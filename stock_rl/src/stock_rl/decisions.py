#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''A signal and a risk state, resolved into one decision a person reads.

A :class:`Decision` is three things at once: the action, the horizon, and
the ordered reasons that produced them. Reasons are sorted by impact,
largest first, because the reader's first question is which input decided
this. Every reason names the input it came from and carries the number
that input held, so ``'12-1 momentum rank fell from 1 to 7 of 50'`` is a
reason and ``'signal changed'`` is not.

**Why this record exists: internal control.** Two honest reasons, and
neither of them is a per-decision regulatory duty:

  * **Strategy-identity evidence.** NSE Detailed Operational Modalities
    paras 9.1 and 9.9 require fresh Exchange registration for a black-box
    algo whenever the logic governing it changes. That re-registration
    duty, not explanation, is what makes this system's documentation
    requirements bite, and a machine-checkable per-decision record is the
    evidence that the deployed code is the registered code.
  * **An operational record of human approval or override.** Who agreed
    with a suggestion, who overrode it, and on what stated grounds.

**There is no Indian requirement to explain an individual BUY/HOLD/SELL
decision.** SEBI CIR/MRD/DP/09/2012 para 8(iv) requires logs of control
parameters, orders, trades and data points and prescribes no reason
schema; nothing in SEBI or NSE defines a sufficient explanation. The
research recorded in
``docs/research/compliance/signal-attribution-and-audit-trail.md`` finds
the audit-trail retention floor at five years under NSE para 10.3 and
states plainly that per-decision rationale and per-decision timeframe
rationale are this project's own engineering choices. No regulation is
cited for them here, and none requires them.

Nothing in this module is evidence about returns. It records which inputs
produced which action. Whether that action earns anything is a separate
question, to be asked of a backtest that first survives the corrections
in ``docs/research/corrections-to-design-docs.md`` -- where the momentum
signal this module reports on is documented as having Indian support and
every indicator threshold used to build it is documented as having none.

**HOLD is an outcome, not the absence of one.** A strong signal on a book
already sitting at its target weight is a HOLD that says so. So is a size
that rounds to no whole share, and so is an order the pre-trade RMS
refuses. Each is a result of the rule tree rather than a fall-through, and
each carries the input that produced it.

**No input may postdate the decision bar.** Every input carries an
``as_of`` and :func:`decide` refuses rather than filters, raising the same
:class:`~stock_rl.sentiment.score.LookAheadError` sentiment raises for the
same failure. Filtering would silently produce a decision built on less
information than the caller believes it used, which is the number a
reviewer cannot reproduce.
'''

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum, StrEnum

from stock_rl.costs import Side
from stock_rl.execution.audit import hash_payload
from stock_rl.execution.sizing import (
  RiskCapBreach,
  SizingInputs,
  StockSizer,
  kelly_method,
)
from stock_rl.risk.checks import AccountState
from stock_rl.risk.killswitch import Trip
from stock_rl.sentiment.score import LookAheadError

__all__ = [
  'Action',
  'Decision',
  'LookAheadError',
  'PositionState',
  'Reason',
  'RiskState',
  'SignalState',
  'Timeframe',
  'decide',
  'decision_schema',
  'entry_rank_max',
  'exit_rank_max',
  'fast_momentum_max',
  'reason_sources',
  'slow_momentum_min',
  'target_tolerance',
  'veto_impact',
]

#: Schema tag carried by every serialised decision and every fingerprint.
#: Bumped when the payload layout changes, so a record written by other
#: logic is recognised as such instead of being read as this one.
decision_schema = 'stock_rl.decisions/1'

#: Impact given to a veto: a latched kill switch, or a pre-trade RMS
#: refusal. A veto is not a driver competing with other drivers, it is
#: the absence of any order at all, so it is put deliberately out of scale
#: with them and therefore always sorts first. Without that, a rank move of
#: six places outranks a halt, and the reader learns the least important
#: thing first.
veto_impact = 1.0e6

#: Largest gap between the target weight and the held weight that still
#: counts as already at target. A tolerance and not an equality, because a
#: target computed in floating point is never exactly the weight the book
#: ends up holding, and a decision that re-trades the rounding residue on
#: every bar is a decision that never stops.
target_tolerance = 1e-4

#: Enter when the cross-sectional 12-1 momentum rank is this good or
#: better, 1 being the strongest of the universe.
entry_rank_max = 10

#: Leave when the rank is this bad or worse. Between the two bands there
#: is a deliberate no-trade zone rather than a third action, so that the
#: strategy does not pay costs to cross from one side of the distribution
#: to the other.
exit_rank_max = 30

#: Upper edge of the fast band for the 12-1 momentum that selects the
#: timeframe, as a fraction.
fast_momentum_max = 0.05

#: Lower edge of the slow band for the same momentum.
slow_momentum_min = 0.15

# Every threshold above is a rule of this project, not a citation. There
# is no published Indian study for a rank band, a momentum band or a
# target tolerance; the corrections file records that the indicator
# thresholds behind the rank have no Indian evidence either.
# ponytail: these four numbers are declared, not fitted, and nothing here
# measures whether they are the right ones. Ceiling hit: a declared band
# is only as good as the evidence behind it and this project has none.
# Upgrade path: fit the bands on a purged, embargoed split
# (src/stock_rl/splits.py) and re-register with the Exchange afterwards,
# because under NSE 9.9 moving them is a change of decision logic.

#: Every input a :class:`Reason` may name. A reason that cannot name one of
#: these is refused, which is how "the reason must name its input" becomes
#: an invariant instead of a convention.
reason_sources = (
  'position.flat',
  'position.target_gap',
  'risk.kill_switch',
  'risk.rms_refusal',
  'signal.momentum',
  'signal.momentum_gate',
  'signal.no_trade_band',
  'signal.rank_entry',
  'signal.rank_exit',
  'signal.rank_move',
  'sizing.rounds_to_nothing',
  'sizing.size',
  'timeframe.band',
)

#: Shape of the hex digest in a fingerprint.
_digest_shape = re.compile(r'[0-9a-f]{64}')


class Action(StrEnum):
  '''What a person is being asked to do.

  **BUY and SELL are :class:`~stock_rl.costs.Side`, reused rather than
  re-declared.** Their values are read off :class:`~stock_rl.costs.Side`
  directly, so there is one table of side strings in the project rather
  than two that can drift: a serialised decision carries ``'buy'`` and
  ``'sell'``, which is what :mod:`stock_rl.execution.audit` already
  records and what :mod:`stock_rl.costs` already prices.

  HOLD is the reason this type exists at all. It is a decision outcome and
  not a side of a trade: there is no order, no cost and no fill for it, so
  it has no :class:`~stock_rl.costs.Side` to live in.

  Attributes:
    BUY: Increase the position.
    HOLD: Do nothing, for a stated reason. First-class, not a default.
    SELL: Reduce the position out of the signal's band.
  '''

  BUY = Side.BUY.value
  HOLD = 'hold'
  SELL = Side.SELL.value


class Timeframe(Enum):
  '''The horizon a decision is expected to play out over, in bars.

  **The horizon is a bar count, not an adjective.** ``'short term'``
  names no length a reader can check against a chart; ``5`` does. Bars
  rather than days because the same number of bars is a different amount
  of time on a one-minute chart and on a weekly one, and the bar count is
  the one unit this project's splits and horizons can verify.

  Attributes:
    FAST: Five bars. Selected when the 12-1 momentum is at or below
      :data:`fast_momentum_max`.
    MID: Twenty bars. The band between the two momentum edges.
    SLOW: Sixty bars. Selected at or above :data:`slow_momentum_min`,
      where a signal is expected to take longer to work.
  '''

  FAST = 5
  MID = 20
  SLOW = 60

  @property
  def bars(self) -> int:
    '''Return the horizon in bars.

    Returns:
      The number of bars this timeframe spans.
    '''
    return int(self.value)


@dataclass(frozen=True, slots=True)
class Reason:
  '''One driver of a decision, named by the input it came from.

  Frozen, because a reason edited after the fact is the one artefact this
  module exists to be trustworthy about.

  Attributes:
    source: The input that produced this reason, one of
      :data:`reason_sources`. Naming the input is what separates a reason
      from a mood: ``'signal.momentum'`` can be checked against the
      recorded state and ``'sentiment feels bad'`` cannot.
    impact: How far this driver moved the decision, as a fraction of the
      source's own full range. :attr:`Decision.reasons` is sorted by this
      descending. Zero marks a reason that qualifies the record rather
      than moving it, which is how the timeframe criterion sorts last
      while still being on the record.
    detail: The sentence a person reads, carrying the numbers.
  '''

  source: str
  impact: float
  detail: str

  def __post_init__(self) -> None:
    '''Validate the reason at construction.

    Raises:
      ValueError: If the source is not one of :data:`reason_sources`, the
        detail is blank, or the impact is negative or not finite. A
        negative impact has no reading, and sorting on absolute values
        would turn it into a positive-looking driver.
    '''
    if self.source not in reason_sources:
      raise ValueError(
        f'reason source must be one of {reason_sources}, got '
        f'{self.source!r}')
    if not self.detail.strip():
      raise ValueError(
        'a reason must say what it observed; a decision whose reason is '
        'blank is the failure this module exists to prevent')
    if not math.isfinite(self.impact) or self.impact < 0.0:
      raise ValueError(
        f'reason impact must be finite and >= 0, got {self.impact}')


@dataclass(frozen=True, slots=True)
class SignalState:
  '''The signal, as it stood at the decision bar.

  **Every field is a reading a trader could have had at the bar.** The
  cross-sectional rank is over :attr:`universe` names from the same bar, so
  a rank is meaningless without the universe it was computed in, and both
  are carried.

  Attributes:
    symbol: NSE scrip the signal is about.
    momentum_rank: Cross-sectional rank of 12-1 momentum at the decision
      bar, 1 being the strongest of :attr:`universe`.
    previous_rank: The same rank one bar earlier. Carried so a reason can
      state the move rather than the level: a rank of 7 is unremarkable
      unless it was 1.
    universe: How many names the rank is out of, at least as large as
      either rank.
    momentum: The 12-1 momentum value, as a fraction.
    as_of: When these readings became available. Never the fetch time.
  '''

  symbol: str
  momentum_rank: int
  previous_rank: int
  universe: int
  momentum: float
  as_of: datetime

  def __post_init__(self) -> None:
    '''Validate the snapshot at construction.

    Raises:
      ValueError: If the symbol is blank, a rank is outside
        ``1..universe``, the universe is smaller than a rank, or ``as_of``
        is naive. Timezone-awareness is required for the same reason
        sentiment requires it: a naive instant on the look-ahead field is
        not a recoverable class of bug.
    '''
    if not self.symbol.strip():
      raise ValueError('a signal needs a symbol')
    if self.universe < 1:
      raise ValueError(f'universe must be >= 1, got {self.universe}')
    for name in ('momentum_rank', 'previous_rank'):
      rank = getattr(self, name)
      if not 1 <= rank <= self.universe:
        raise ValueError(
          f'{name} must be in 1..{self.universe}, got {rank}')
    if not math.isfinite(self.momentum):
      raise ValueError(f'momentum must be finite, got {self.momentum}')
    _require_aware(self.as_of, 'SignalState.as_of')


@dataclass(frozen=True, slots=True)
class PositionState:
  '''The book, as it stood at the decision bar.

  Attributes:
    symbol: Scrip the position is in. Must match the signal's symbol; a
      decision that sizes one name against another name's book is
      refused rather than reconciled.
    target_weight: Weight the book wants to hold, as a fraction of
      capital.
    current_weight: Weight actually held, as a fraction of capital.
    as_of: When this became available.
  '''

  symbol: str
  target_weight: float
  current_weight: float
  as_of: datetime

  def __post_init__(self) -> None:
    '''Validate the snapshot at construction.

    Raises:
      ValueError: If the symbol is blank, either weight is outside
        ``[-1, 1]``, or ``as_of`` is naive.
    '''
    if not self.symbol.strip():
      raise ValueError('a position needs a symbol')
    for name in ('target_weight', 'current_weight'):
      weight = getattr(self, name)
      if not math.isfinite(weight) or not -1.0 <= weight <= 1.0:
        raise ValueError(f'{name} must be in [-1, 1], got {weight}')
    _require_aware(self.as_of, 'PositionState.as_of')

  @property
  def gap(self) -> float:
    '''Return the absolute weight between target and held.

    Returns:
      ``abs(target_weight - current_weight)``, a fraction of capital.
      Compared against :data:`target_tolerance` to answer whether the
      target is already met.
    '''
    return abs(self.target_weight - self.current_weight)


@dataclass(frozen=True, slots=True)
class RiskState:
  '''The kill switch's own record of why trading is halted.

  Carries :class:`~stock_rl.risk.killswitch.Trip` objects rather than a
  boolean and a code, so the decision quotes the switch's own words
  instead of paraphrasing them, and so an unknown trip code cannot be
  invented here: :class:`Trip` validates against
  :data:`~stock_rl.risk.killswitch.trip_codes` at construction.

  Attributes:
    trips: Every trip the switch holds, oldest first. Empty means armed,
      and a latched switch with no trip record is refused: a halt nobody
      can name is the record an inspection asks for.
    as_of: When these trips were recorded.
  '''

  trips: tuple[Trip, ...]
  as_of: datetime

  def __post_init__(self) -> None:
    '''Validate the snapshot at construction.

    Raises:
      ValueError: If ``as_of`` is naive.
    '''
    _require_aware(self.as_of, 'RiskState.as_of')

  @property
  def tripped(self) -> bool:
    '''Return whether the switch is latched.

    Returns:
      True when at least one trip is on record. There is no separate flag
      to disagree with the trips, which is the property a halt depends
      on.
    '''
    return bool(self.trips)

  @property
  def codes(self) -> tuple[str, ...]:
    '''Return every trip code on record, oldest first.'''
    return tuple(trip.code for trip in self.trips)


@dataclass(frozen=True, slots=True)
class Decision:
  '''One decision, with everything that produced it.

  Attributes:
    action: What to do.
    timeframe: How many bars the decision is expected to take to play out.
    sizes: Fraction of capital this decision commits, per symbol. A BUY
      commits the fraction
      :meth:`~stock_rl.execution.sizing.StockSizer.size` returned, so no
      sizing arithmetic happens here. A SELL is a close out of the signal's
      band, so it commits nothing and the recorded weight is zero. A HOLD
      commits nothing at all, which is enforced: a HOLD carrying a size is
      a decision that could still put an order out, and that is the one
      thing a HOLD must never be.
    reasons: Ordered by :attr:`Reason.impact`, largest first, and never
      empty. A decision with no reasons cannot be constructed.
    decided_at: The decision bar, timezone-aware.
    fingerprint: ``<schema>:<sha256 hex>`` over the inputs that produced
      this decision, so two runs on one state are recognisably the same
      decision and any changed input is visible.
  '''

  action: Action
  timeframe: Timeframe
  sizes: tuple[tuple[str, float], ...]
  reasons: tuple[Reason, ...]
  decided_at: datetime
  fingerprint: str

  def __post_init__(self) -> None:
    '''Validate the decision at construction.

    Raises:
      ValueError: If ``decided_at`` is naive, the reasons are empty or
        not in descending impact order, a size is negative, a symbol
        appears twice, an action does not match the sizes it carries, or
        the fingerprint is not this module's shape.
    '''
    _require_aware(self.decided_at, 'Decision.decided_at')
    if not self.reasons:
      raise ValueError(
        'a decision must carry at least one reason; a decision nobody can '
        'explain is not a decision')
    impacts = [reason.impact for reason in self.reasons]
    if impacts != sorted(impacts, reverse=True):
      raise ValueError(
        f'reasons must be ordered by impact, largest first, got {impacts}')
    seen: set[str] = set()
    for symbol, fraction in self.sizes:
      if not symbol.strip():
        raise ValueError('a size needs a symbol')
      if symbol in seen:
        raise ValueError(f'symbol {symbol!r} is sized twice in one decision')
      seen.add(symbol)
      if not math.isfinite(fraction) or fraction < 0.0:
        raise ValueError(
          f'size for {symbol} must be finite and >= 0, got {fraction}')
    if self.action is Action.HOLD and self.sizes:
      raise ValueError(
        f'a HOLD commits no capital, but this one sizes {self.sizes}; a '
        'record that says "hold" and also says what to trade is an order '
        'waiting for a bug to release it')
    if self.action is not Action.HOLD and not self.sizes:
      raise ValueError(
        f'{self.action.value} must say what it commits, but carries no '
        'size')
    scheme, _, digest = self.fingerprint.partition(':')
    if scheme != decision_schema or not _digest_shape.fullmatch(digest):
      raise ValueError(
        f'fingerprint must be {decision_schema}:<64 hex>, got '
        f'{self.fingerprint!r}')

  @property
  def largest_reason(self) -> Reason:
    '''Return the reason that decided this, the first by impact.'''
    return self.reasons[0]

  def to_json(self) -> str:
    '''Return the decision as one canonical JSON line.

    Byte-stable by construction: sorted keys, no insignificant
    whitespace, floats rendered by ``json`` through ``repr`` so the
    recorded digits are the exact double, and no wall-clock read. Two
    processes recording one decision write the same bytes, so a record can
    be compared, diffed or hashed without normalising it first.

    Returns:
      A compact JSON object as a string.
    '''
    payload = {
      'schema': decision_schema,
      'action': self.action.value,
      'timeframe': {
        'name': self.timeframe.name,
        'bars': self.timeframe.bars,
      },
      'sizes': dict(self.sizes),
      'reasons': [
        {'source': reason.source, 'impact': reason.impact,
         'detail': reason.detail}
        for reason in self.reasons
      ],
      'decided_at': self.decided_at.isoformat(),
      'fingerprint': self.fingerprint,
    }
    return json.dumps(payload, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True)

  @classmethod
  def from_json(cls, text: str | Mapping[str, object]) -> 'Decision':
    '''Rebuild a decision from what :meth:`to_json` produced.

    Args:
      text: The JSON line, or an already-parsed mapping of it.

    Returns:
      The :class:`Decision`, whose :meth:`to_json` is byte-identical to
      ``text`` for any line this module wrote.

    Raises:
      ValueError: If the payload is not a JSON object, carries a foreign
        schema, an unknown action, a timeframe whose recorded horizon
        disagrees with the enum, a malformed reason, or a naive timestamp.
        Every one of those is a record written by different logic, which
        is precisely what the re-registration duty exists to catch, so
        none of them is read leniently.
    '''
    payload = json.loads(text) if isinstance(text, str) else dict(text)
    if not isinstance(payload, dict):
      raise ValueError(f'a decision payload must be an object, got {text!r}')
    schema = str(payload.get('schema'))
    if schema != decision_schema:
      raise ValueError(
        f'decision schema {schema!r} is not {decision_schema!r}')
    try:
      return cls(
        action=Action(str(payload['action'])),
        timeframe=_timeframe_from(payload['timeframe']),
        sizes=tuple(
          (str(symbol), float(fraction))
          for symbol, fraction in dict(payload['sizes']).items()
        ),
        reasons=tuple(
          _reason_from(item) for item in payload['reasons']),
        decided_at=_aware_from(str(payload['decided_at'])),
        fingerprint=str(payload['fingerprint']),
      )
    except (KeyError, TypeError, ValueError) as exc:
      raise ValueError(f'malformed decision record: {exc}') from exc

  def render(self) -> str:
    '''Return the decision as the block a person reads before acting.

    Returns:
      Multi-line text: the action and its horizon, what it commits, and
      the reasons in the order they decided it. A SELL is rendered as a
      close to a weight rather than as a size, because a close commits no
      capital and printing ``'size 0.00% of capital'`` for one reads like
      a refusal rather than an instruction.
    '''
    lines = [
      f'{self.action.value.upper()} {self.timeframe.name} '
      f'({self.timeframe.bars} bars) at {self.decided_at.isoformat()}',
    ]
    if self.action is Action.BUY:
      lines.extend(
        f'  size {symbol} {fraction:.2%} of capital'
        for symbol, fraction in self.sizes
      )
    else:
      # Reached only by a SELL. A non-HOLD action always carries a size,
      # enforced in __post_init__, so there is nothing to guard here: a
      # branch that tested self.sizes here would be testing an
      # unreachable state and would hide the invariant.
      lines.extend(
        f'  close {symbol} to {fraction:.2%} of capital'
        for symbol, fraction in self.sizes
      )
    lines.extend(
      f'  {index}. [{reason.source} {reason.impact:.4f}] {reason.detail}'
      for index, reason in enumerate(self.reasons, start=1)
    )
    lines.append(f'  inputs fingerprint {self.fingerprint}')
    return '\n'.join(lines)


def decide(
  signal: SignalState,
  position: PositionState,
  risk: RiskState,
  sizer: StockSizer,
  decision_bar: datetime,
  price: float,
  sizing: SizingInputs = SizingInputs(),
  method: str = kelly_method,
  account: AccountState = AccountState(),
  algo_id: str = '',
) -> Decision:
  '''Turn a signal and a risk state into one decision.

  The rule tree, in the order it is evaluated. The order is the
  defensiveness: a control comes before a preference, so a latched kill
  switch is read before any reason for trading is considered.

  1. **Visibility.** Every input must have been available at
     ``decision_bar``. Refused otherwise, never filtered.
  2. **Kill switch.** Latched means HOLD, with the trip codes quoted. No
     size is computed at all: an order sized while halted is an order
     waiting for a bug to release it.
  3. **Target already met.** HOLD, which is an outcome and not the
     absence of a signal -- this branch is reached with a full-strength
     entry signal whenever the book is already where the signal wants it.
  4. **Exit band.** At or beyond :data:`exit_rank_max` the position is
     closed, or HOLD when there is nothing to close.
  5. **No-trade band and the momentum gate.** Between the bands, or with a
     non-positive momentum, HOLD.
  6. **Entry.** Inside :data:`entry_rank_max` with a positive momentum, the
     order is sized by :meth:`~stock_rl.execution.sizing.StockSizer.size`
     and cleared against the same :class:`~stock_rl.risk.checks.RmsChecker`
     the order path uses. A refusal or a size that rounds to no whole
     share becomes a HOLD naming the refusal, because a sizer that
     shrinks an order to fit hides the fact that the strategy wanted more
     than the account could give.
  7. **Timeframe.** Every branch carries why its horizon was selected,
     with zero impact so it qualifies the record without displacing a
     driver.

  Args:
    signal: The signal at the decision bar.
    position: The book at the decision bar. Must be the same symbol.
    risk: The kill switch's record at the decision bar.
    sizer: The sizer every size comes from.
    decision_bar: The bar the decision is taken at, timezone-aware. No
      input may postdate it.
    price: Price to size against, which must be positive.
    sizing: Edge, haircut and volatility inputs handed to the sizer.
    method: Sizing rule, :data:`~stock_rl.execution.sizing.kelly_method`
      or :data:`~stock_rl.execution.sizing.vol_target_method`.
    account: Account ledger the sized order is cleared against.
    algo_id: Algo ID tagged on the order.

  Returns:
    The :class:`Decision`, reasons sorted by impact.

  Raises:
    LookAheadError: If any input became available after ``decision_bar``.
    ValueError: If the bar is naive, the price is not positive, or the
      signal and position name different symbols.
    ValueError: From the sizer, if the sizing inputs cannot produce a
      size. Left to propagate: a decision whose size cannot be computed
      must not be silently replaced by a smaller one.
  '''
  _require_visible(decision_bar, signal, position, risk)
  if signal.symbol != position.symbol:
    raise ValueError(
      f'signal is for {signal.symbol} but the position is for '
      f'{position.symbol}; a decision that sizes one name against another '
      "name's book is not a decision")
  if price <= 0.0:
    raise ValueError(
      f'price must be positive; nothing can be sized from {price}')
  fingerprint = _fingerprint(
    signal, position, risk, decision_bar, price, sizing, method, account,
    sizer)
  timeframe = _select_timeframe(signal.momentum)
  context = _timeframe_reason(timeframe, signal.momentum)
  symbol = signal.symbol
  gap = position.gap

  if risk.tripped:
    quoted = '; '.join(
      f'{trip.code}: {trip.detail}' for trip in risk.trips)
    return _decision(
      Action.HOLD, timeframe, (),
      [Reason('risk.kill_switch', veto_impact,
              f'the kill switch is latched, so no order may be released '
              f'({quoted})'), context],
      decision_bar, fingerprint)

  if gap <= target_tolerance:
    return _decision(
      Action.HOLD, timeframe, (),
      [Reason('position.target_gap', gap,
              f'target weight {position.target_weight:.2%} is already held '
              f'(current {position.current_weight:.2%}, gap {gap:.4f} at or '
              f'below the {target_tolerance:.4f} tolerance): target '
              'already met, no action needed'), context],
      decision_bar, fingerprint)

  rank = signal.momentum_rank
  universe = signal.universe
  move = abs(rank - signal.previous_rank) / universe

  if rank >= exit_rank_max:
    if position.current_weight <= 0.0:
      return _decision(
        Action.HOLD, timeframe, (),
        [Reason('position.flat',
                (rank - exit_rank_max + 1) / universe,
                f'momentum 12-1 rank {rank} of {universe} is at or beyond '
                f'the exit band {exit_rank_max}-{universe}, but the book '
                f'holds {position.current_weight:+.2%} of capital: there is '
                'nothing to sell'), context],
        decision_bar, fingerprint)
    left = f'close to flat from {position.current_weight:.2%}'
    if position.target_weight > 0.0:
      left += (f', overriding the {position.target_weight:.2%} target '
               'weight: the exit band is a flat close')
    return _decision(
      Action.SELL, timeframe, ((symbol, 0.0),),
      [Reason('signal.rank_exit',
              (rank - exit_rank_max + 1) / universe,
              f'momentum 12-1 rank {rank} of {universe} is at or beyond '
              f'the exit band {exit_rank_max}-{universe}: {left}'),
       Reason('signal.rank_move', move,
              f'cross-sectional rank moved {rank - signal.previous_rank:+d} '
              f'places, from {signal.previous_rank} to {rank} of '
              f'{universe}'),
       Reason('position.target_gap', gap,
              f'the close covers a weight gap of {gap:.2%} of capital'),
       context],
      decision_bar, fingerprint)

  if rank > entry_rank_max:
    return _decision(
      Action.HOLD, timeframe, (),
      [Reason('signal.no_trade_band', (rank - entry_rank_max) / universe,
              f'momentum 12-1 rank {rank} of {universe} is in the '
              f'no-trade band {entry_rank_max + 1}-{exit_rank_max - 1}: '
              'neither strong enough to add nor weak enough to drop'),
       Reason('signal.rank_move', move,
              f'cross-sectional rank moved {rank - signal.previous_rank:+d} '
              f'places, from {signal.previous_rank} to {rank} of '
              f'{universe}'),
       context],
      decision_bar, fingerprint)

  if signal.momentum <= 0.0:
    return _decision(
      Action.HOLD, timeframe, (),
      [Reason('signal.momentum_gate', abs(signal.momentum),
              f'12-1 momentum is {signal.momentum:+.2%}, not positive, so '
              f'rank {rank} of {universe} is not an entry despite sitting '
              f'inside the entry band 1-{entry_rank_max}'),
       Reason('signal.rank_move', move,
              f'cross-sectional rank moved {rank - signal.previous_rank:+d} '
              f'places, from {signal.previous_rank} to {rank} of '
              f'{universe}'),
       context],
      decision_bar, fingerprint)

  drivers = [
    Reason('signal.momentum', abs(signal.momentum),
           f'12-1 momentum is {signal.momentum:+.2%}'),
    Reason('signal.rank_entry', (entry_rank_max - rank) / universe,
           f'momentum 12-1 rank {rank} of {universe} is inside the entry '
           f'band 1-{entry_rank_max}'),
    Reason('signal.rank_move', move,
           f'cross-sectional rank moved {rank - signal.previous_rank:+d} '
           f'places, from {signal.previous_rank} to {rank} of {universe}'),
    Reason('position.target_gap', gap,
           f'target weight {position.target_weight:.2%} against current '
           f'{position.current_weight:.2%}: {gap:.2%} of capital to put on'),
  ]
  try:
    # Action.BUY.value is Side.BUY.value, so the sizer is handed the side
    # the decision says without a translation table in between.
    result = sizer.size(
      symbol=symbol, price=price, side=Action.BUY.value, inputs=sizing,
      method=method, account=account, algo_id=algo_id,
      reference_price=price)
  except RiskCapBreach as exc:
    return _decision(
      Action.HOLD, timeframe, (),
      [Reason('risk.rms_refusal', veto_impact,
              f'the pre-trade RMS refused the sized order, so the strategy '
              f'asked for more than the account could give: '
              f'{exc.report.render()}'), *drivers, context],
      decision_bar, fingerprint)

  if not result.should_trade:
    return _decision(
      Action.HOLD, timeframe, (),
      [Reason('sizing.rounds_to_nothing', result.fraction,
              f'the {method} size of {result.fraction:.2%} of capital is '
              f'{result.fraction:.6f} and rounds to no whole share at '
              f'{price:.2f}: there is no order to place'),
       *drivers, context],
      decision_bar, fingerprint)

  return _decision(
    Action.BUY, timeframe, ((symbol, result.fraction),),
    [*drivers,
     Reason('sizing.size', result.fraction,
            f'the {method} size is {result.fraction:.2%} of capital, '
            f'{result.quantity} shares for {result.notional:.0f} at '
            f'{price:.2f}'), context],
    decision_bar, fingerprint)


def _select_timeframe(momentum: float) -> Timeframe:
  '''Return the horizon the signal's momentum selects.

  Args:
    momentum: The 12-1 momentum value, as a fraction.

  Returns:
    :attr:`Timeframe.FAST` at or below :data:`fast_momentum_max`,
    :attr:`Timeframe.SLOW` at or above :data:`slow_momentum_min`, and
    :attr:`Timeframe.MID` between them. A non-positive momentum selects
    FAST: weak evidence is held to the short horizon.
  '''
  if momentum <= fast_momentum_max:
    return Timeframe.FAST
  if momentum < slow_momentum_min:
    return Timeframe.MID
  return Timeframe.SLOW


def _timeframe_reason(timeframe: Timeframe, momentum: float) -> Reason:
  '''Return the reason stating which horizon was selected and why.

  Carries zero impact deliberately: it qualifies the record rather than
  moving the action, so it sorts after every driver and is never read as
  the reason for the decision.

  Args:
    timeframe: The selected horizon.
    momentum: The 12-1 momentum value that selected it.

  Returns:
    A :class:`Reason` on ``timeframe.band``.
  '''
  return Reason(
    'timeframe.band', 0.0,
    f'12-1 momentum {momentum:+.2%} selects the {timeframe.name} band, a '
    f'horizon of {timeframe.bars} bars')


def _decision(
  action: Action,
  timeframe: Timeframe,
  sizes: tuple[tuple[str, float], ...],
  reasons: list[Reason],
  decided_at: datetime,
  fingerprint: str,
) -> Decision:
  '''Return a Decision with its reasons ordered by impact.

  The sort happens here, once, rather than at each branch: a list ordered
  by hand at seven call sites is a list that will be wrong at one of them.

  Args:
    action: What to do.
    timeframe: The horizon selected.
    sizes: Fraction of capital committed per symbol.
    reasons: The reasons, in any order. ``sorted`` is stable, so reasons
      of equal impact keep the order the branch listed them in, which is
      what makes the fingerprint reproducible.
    decided_at: The decision bar.
    fingerprint: Digest of the inputs.

  Returns:
    The :class:`Decision`.
  '''
  return Decision(
    action=action,
    timeframe=timeframe,
    sizes=sizes,
    reasons=tuple(sorted(reasons, key=lambda reason: -reason.impact)),
    decided_at=decided_at,
    fingerprint=fingerprint,
  )


def _fingerprint(
  signal: SignalState,
  position: PositionState,
  risk: RiskState,
  decision_bar: datetime,
  price: float,
  sizing: SizingInputs,
  method: str,
  account: AccountState,
  sizer: StockSizer,
) -> str:
  '''Return the digest of the inputs a decision was made from.

  **Hashed through the audit chain's own
  :func:`~stock_rl.execution.audit.hash_payload`**, so a decision's
  fingerprint and an audit record's chain hash canonicalise identically.
  Two implementations of "canonical JSON" are two chances for two records
  of the same event to disagree.

  Everything that can change the outcome is in the payload, the sizer's own
  configuration included. A digest that left out the capital base would
  give one fingerprint to a decision that places forty shares and to one
  on the same signal that places none, which makes the fingerprint useless
  for telling two decisions apart.

  The digest covers the inputs and not the output. That makes it a record
  of provenance -- "this action came from this state" -- rather than a
  signature over the rendered decision.
  # ponytail: an edited action would survive a fingerprint check, because
  # nothing here hashes the reasons, and the Algo ID a caller passes for
  # the order tag is not an input to the decision so it is not hashed.
  # Ceiling hit: the fingerprint proves which inputs were used, not that
  # the record was not altered after the fact. Upgrade path: hash the
  # action, timeframe and reasons into the payload too, or chain the record
  # through :class:`~stock_rl.execution.audit.AuditLog`, which already
  # stores a per-record hash for exactly that reason.

  Args:
    signal: The signal snapshot.
    position: The position snapshot.
    risk: The kill switch's record.
    decision_bar: The bar the decision is taken at.
    price: Price sized against.
    sizing: Edge, haircut and volatility inputs.
    method: Sizing rule name.
    account: Account ledger.
    sizer: The sizer, whose capital, lot size, cap and venue configuration
      all bear on the size and on whether an order survives the RMS.

  Returns:
    ``<decision_schema>:<64 lowercase hex digits>``.
  '''
  payload = {
    'schema': decision_schema,
    'decision_bar': decision_bar.isoformat(),
    # ``trips`` is overridden because Trip serialises through its own
    # to_json rather than as a bare dataclass.
    'risk': {**_record(risk), 'as_of': risk.as_of.isoformat(),
             'trips': [trip.to_json() for trip in risk.trips]},
    'position': {**_record(position),
                 'as_of': position.as_of.isoformat()},
    'signal': {**_record(signal), 'as_of': signal.as_of.isoformat()},
    'price': repr(price),
    'sizing': _record(sizing),
    'method': method,
    'account': _record(account),
    'sizer': {
      'capital': repr(sizer.capital),
      'lot_size': repr(sizer.lot_size),
      'cap': repr(sizer.cap),
      # The checker is not a dataclass, so its venue map is read field by
      # field. Its band is what refuses an order, which makes it an input
      # to the decision rather than a detail of the sizer.
      'limits': _record(sizer.checker.limits),
      'securities': {
        symbol: _record(security)
        for symbol, security in sorted(sizer.checker.securities.items())
      },
      'banned': sorted(sizer.checker.banned),
    },
  }
  return f'{decision_schema}:{hash_payload(payload)}'


def _record(value: object) -> dict[str, object]:
  '''Return a frozen dataclass as canonical JSON primitives.

  Args:
    value: Any dataclass instance.

  Returns:
    Mapping of field name to a JSON-stable value. Floats go through
    ``repr`` so the digest records the exact double rather than a rounded
    decimal, and a mapping is keyed by ``str`` and sorted by ``json``
    downstream so two equal ledgers hash equally.

  Raises:
    TypeError: If ``value`` is not a dataclass instance. Every caller in
      this module passes one, and a silent ``{}`` for something else would
      fingerprint two different inputs identically.
  '''
  if not is_dataclass(value) or isinstance(value, type):
    raise TypeError(
      f'expected a dataclass instance, got {type(value).__name__}')
  return {field.name: _plain(getattr(value, field.name))
          for field in fields(value)}


def _plain(value: object) -> object:
  '''Return one value as a JSON-stable primitive.

  Args:
    value: Arbitrary dataclass field value.

  Returns:
    A structure built only of dict, list, str, int and float.
  '''
  if is_dataclass(value) and not isinstance(value, type):
    return _record(value)
  if isinstance(value, Mapping):
    return {str(key): _plain(item) for key, item in sorted(value.items())}
  if isinstance(value, (list, tuple)):
    return [_plain(item) for item in value]
  if isinstance(value, float):
    return repr(value)
  return value


def _require_visible(decision_bar: datetime, *sources: object) -> None:
  '''Refuse when any input became available after the decision bar.

  Args:
    decision_bar: The bar the decision is taken at.
    sources: Inputs carrying an ``as_of`` instant.

  Raises:
    LookAheadError: If any input postdates the bar. Intra-bar look-ahead
      is this project's most repeated bug class and a decision built on it
      is the most consequential place for one to hide, because the
      decision is what a person acts on.
    ValueError: If the bar is naive, or an input carries no ``as_of``.
      The second check exists because an input without a timestamp cannot
      be shown to be legal at any bar.
  '''
  _require_aware(decision_bar, 'decision_bar')
  leaked: list[str] = []
  for source in sources:
    at = getattr(source, 'as_of', None)
    if not isinstance(at, datetime):
      raise ValueError(
        f'{type(source).__name__} carries no aware as_of timestamp, so it '
        'cannot be shown to be visible at any bar')
    if at > decision_bar:
      leaked.append(f'{type(source).__name__}@{at.isoformat()}')
  if leaked:
    listed = ', '.join(leaked)
    raise LookAheadError(
      f'{len(leaked)} input(s) were not visible at '
      f'{decision_bar.isoformat()}: {listed}')


def _require_aware(moment: datetime, label: str) -> None:
  '''Refuse a naive timestamp.

  Args:
    moment: The instant to check.
    label: What the instant belongs to, for the message.

  Raises:
    ValueError: If ``moment`` carries no timezone. A naive instant
      compared against an aware decision bar raises a ``TypeError`` from
      deep inside a backtest, and an IST-versus-UTC slip on the one field
      that defines look-ahead is not a recoverable class of bug.
  '''
  if moment.tzinfo is None:
    raise ValueError(
      f'{label} must be timezone-aware; a naive timestamp on a look-ahead '
      'field is not acceptable')


def _aware_from(text: str) -> datetime:
  '''Return an aware datetime parsed from ISO 8601 text.

  Args:
    text: The timestamp.

  Returns:
    The timezone-aware datetime.

  Raises:
    ValueError: If the text is not an ISO 8601 timestamp, or is naive.
  '''
  try:
    moment = datetime.fromisoformat(text)
  except ValueError as exc:
    raise ValueError(f'not an ISO 8601 timestamp: {text!r}') from exc
  _require_aware(moment, 'decided_at')
  return moment


def _timeframe_from(value: object) -> Timeframe:
  '''Return the :class:`Timeframe` a serialised record names.

  Args:
    value: Mapping with ``name`` and ``bars``.

  Returns:
    The named :class:`Timeframe`.

  Raises:
    ValueError: If the value is not a mapping, the name is unknown, or the
      recorded horizon disagrees with this module's enum. A record whose
      horizon disagrees with the running code is a record written by
      different logic, which is what NSE 9.9 makes a re-registration
      matter, so it is not read leniently.
  '''
  if not isinstance(value, Mapping):
    raise ValueError(f'timeframe must be an object, got {value!r}')
  name = str(value.get('name'))
  try:
    timeframe = Timeframe[name]
  except KeyError as exc:
    raise ValueError(f'unknown timeframe {name!r}') from exc
  bars = int(value.get('bars'))
  if bars != timeframe.bars:
    raise ValueError(
      f'timeframe {name} is {timeframe.bars} bars in this module but the '
      f'record says {bars}; the record and the logic disagree')
  return timeframe


def _reason_from(value: object) -> Reason:
  '''Return the :class:`Reason` a serialised entry names.

  Args:
    value: Mapping with ``source``, ``impact`` and ``detail``.

  Returns:
    The :class:`Reason`.

  Raises:
    ValueError: If the entry is not a mapping or a field is missing.
  '''
  if not isinstance(value, Mapping):
    raise ValueError(f'a reason must be an object, got {value!r}')
  return Reason(
    source=str(value['source']),
    impact=float(value['impact']),
    detail=str(value['detail']),
  )
