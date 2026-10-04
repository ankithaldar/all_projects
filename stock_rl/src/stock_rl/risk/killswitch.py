#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''A latching, persistent, automatically-tripped kill switch.

SEBI CIR/MRD/DP/09/2012 requires a kill switch per Algo ID and says it
is "expected to automatically trigger a halt on trading activity based
on pre-defined conditions". The project's own research is blunter about
what the design docs got wrong: a kill switch drawn as a React button is
not a compliance control, because a button is something an operator must
think to press, and the failure it exists for is the moment nobody is
looking. The automatic trip is therefore the primary path here and the
manual trip is the secondary one.

**It latches.** Once tripped, trading stays halted until a human clears
it with a stated reason. A kill switch that clears itself when a
condition recovers is a pause button. There is no code path in this
class from a recovering metric to an untripped switch, which is the
invariant the class exists to make unbreakable, and :meth:`reset` is the
single method that can break it -- deliberately, and only with a reason.

**It survives a restart.** State lives in a small JSON file written
atomically. A process that dies mid-session and comes back must find
itself still halted, because the crash is exactly the moment a naive
implementation returns armed and starts trading. The file carries the
Algo ID it belongs to: SEBI scopes the switch per Algo ID, and loading
another algo's halted state into this one would either halt the wrong
strategy or, worse, pass a test with the wrong state.

**A corrupt state file refuses rather than resets.** A missing file
means first run, which is a genuinely known state. An unreadable or
malformed file means somebody truncated it, and the only safe reading of
a truncated safety state is "do not trade". So it raises.

**No programmatic escape.** :meth:`reset` demands three things: that the
switch is tripped, a non-blank reason, and the caller's current metrics,
which it checks against the same pre-defined conditions. A job that calls
``reset('ack')`` on a schedule therefore cannot clear a live drawdown,
and a job that calls it with a blank reason gets
:class:`KillSwitchLatched` rather than a cleared switch. Passing no
metrics is not a loophole: it reads as ``0.0``, which is clear, and that
is the documented contract rather than a hidden one.

The thresholds are risk appetite, not law. No Indian instrument
prescribes a 5 percent index trip or a 15 percent drawdown trip; SEBI
prescribes only that the conditions be pre-defined and automatic. They
are constructor arguments for that reason, and the defaults are this
project's own suggestion rather than a citation.
'''

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
  'KillSwitch',
  'KillSwitchError',
  'KillSwitchLatched',
  'KillThresholds',
  'Trip',
  'default_state_file',
  'kill_switch_schema',
  'manual_code',
  'trip_codes',
  'utcnow',
]

#: Schema tag written into the state file. Bump it when the file layout
#: changes, so an old file is recognised as old rather than misread.
kill_switch_schema = 'stock_rl.risk.killswitch/1'

#: A human pressed the switch.
manual_code = 'manual'

#: The reference index fell too far in one session.
index_fall_code = 'index_fall'

#: Peak-to-trough drawdown reached the limit.
drawdown_code = 'max_drawdown'

#: The traded symbol is under an exchange F&O ban.
fno_ban_code = 'fno_ban'

#: Order acknowledgement took longer than the limit allows.
ack_latency_code = 'ack_latency'

#: Every trip code this module can record. Kept as a tuple so a test can
#: assert the automatic codes and the manual code all exist, rather than
#: discovering a missing code from a string literal somewhere.
trip_codes = (
  manual_code, index_fall_code, drawdown_code, fno_ban_code,
  ack_latency_code,
)

#: Source of wall-clock time as an aware UTC datetime. Injectable so the
#: state file is deterministic under test. Nothing here sleeps.
Clock = Callable[[], datetime]


def utcnow() -> datetime:
  '''Return the current time as an aware UTC datetime.

  Isolated as a function so the single wall-clock call in this module is
  easy to find and easy to replace.

  Returns:
    Timezone-aware datetime in UTC.
  '''
  return datetime.now(timezone.utc)


def default_state_file(algo_id: str) -> Path:
  '''Return the conventional state-file path for an Algo ID.

  Per Algo ID, because the switch is per Algo ID. A single shared file
  would make two strategies' switches into one another's problem:
  tripping one would halt the other and clearing one would re-arm the
  other.

  Args:
    algo_id: Exchange-allotted Algo ID.

  Returns:
    A path under the user's state directory.

  Raises:
    ValueError: If the Algo ID is blank. A blank ID would collapse every
      strategy onto one file, which is the same defect under a worse
      name.
  '''
  if not algo_id.strip():
    raise ValueError('algo_id must not be blank for a per-algo state file')
  safe = ''.join(
    character if character.isalnum() or character in '-_' else '_'
    for character in algo_id)
  return Path.home() / '.local' / 'state' / 'stock_rl' / f'kill_{safe}.json'


@dataclass(frozen=True, slots=True)
class KillThresholds:
  '''Pre-defined trip conditions, stated so they can be pre-registered.

  Every level is a risk appetite rather than a legal figure, and none has
  a source in any Indian instrument. SEBI prescribes that the conditions
  exist, be pre-defined and be automatic; it does not prescribe their
  levels.

  Attributes:
    index_fall: Maximum tolerated session fall in the reference index,
      as a fraction. A fall of exactly this value trips: a limit that is
      inclusive on the safe side is not a limit.
    max_drawdown: Maximum tolerated peak-to-trough drawdown, as a
      fraction. Equality trips, for the same reason.
    ack_latency: Maximum tolerated seconds between sending an order and
      receiving its acknowledgement. This is the loop detector: a
      strategy that cannot hear back from the exchange is running
      blind, and one that keeps sending anyway is how an unacknowledged
      cascade starts.
    fno_ban: Trip when the traded symbol is under an exchange F&O ban.
      Bool rather than a level, because the ban list is binary.
  '''

  index_fall: float = 0.05
  max_drawdown: float = 0.15
  ack_latency: float = 2.0
  fno_ban: bool = True

  def __post_init__(self) -> None:
    '''Validate the thresholds at construction.

    Raises:
      ValueError: If any threshold is outside ``(0, 1]`` or the latency
        is not positive. A threshold of zero would halt the strategy on
        every evaluation, which is a misconfiguration and belongs in a
        test rather than in production.
    '''
    for name in ('index_fall', 'max_drawdown'):
      value = getattr(self, name)
      if not 0.0 < value <= 1.0:
        raise ValueError(f'{name} must be in (0, 1], got {value}')
    if self.ack_latency <= 0.0:
      raise ValueError(
        f'ack_latency must be positive, got {self.ack_latency}')


@dataclass(frozen=True, slots=True)
class Trip:
  '''One recorded trip.

  Attributes:
    code: One of :data:`trip_codes`.
    detail: What was observed, in the caller's words.
    at: When it happened, ISO 8601 UTC.
    observed: The measured value that breached the threshold, or 0.0
      for a condition with no scalar.
  '''

  code: str
  detail: str
  at: str
  observed: float = 0.0

  def __post_init__(self) -> None:
    '''Validate the trip record at construction.

    Raises:
      ValueError: If the code is unknown, the detail is empty, or ``at`` is
        not a non-empty string. A trip with no reason is precisely the
        record an inspection asks for, so this class must never be able to
        write one.

        ``at`` is validated as a string because that is what the attribute
        declares and what :meth:`to_json` emits. It was previously
        unchecked, so a ``datetime`` passed here constructed cleanly and
        then failed much later, inside an unrelated caller's JSON encode,
        with a ``TypeError`` that pointed nowhere near the mistake.
    '''
    if self.code not in trip_codes:
      raise ValueError(f'unknown trip code {self.code!r}')
    if not isinstance(self.detail, str) or not self.detail.strip():
      raise ValueError('a trip must record what tripped it')
    if not isinstance(self.at, str) or not self.at.strip():
      raise ValueError(
        f'a trip must record an ISO 8601 string for when it tripped, got '
        f'{self.at!r} of type {type(self.at).__name__}')

  def to_json(self) -> dict[str, object]:
    '''Return the record as JSON-stable primitives.

    Returns:
      Mapping with string keys and JSON-safe values.
    '''
    return {
      'code': self.code,
      'detail': self.detail,
      'at': self.at,
      'observed': self.observed,
    }

  @classmethod
  def from_json(cls, payload: Mapping[str, object]) -> 'Trip':
    '''Rebuild a trip record from parsed JSON.

    Args:
      payload: Mapping produced by :meth:`to_json`.

    Returns:
      The :class:`Trip`.

    Raises:
      ValueError: If the payload is missing a key or carries an unknown
        code.
    '''
    try:
      return cls(
        code=str(payload['code']),
        detail=str(payload['detail']),
        at=str(payload['at']),
        observed=float(payload['observed']),
      )
    except (KeyError, TypeError, ValueError) as exc:
      raise ValueError(f'malformed trip record: {exc}') from exc


class KillSwitchError(RuntimeError):
  '''Base class for every kill-switch refusal.'''


class KillSwitchLatched(KillSwitchError):
  '''Raised when the switch is tripped, or cannot legitimately be cleared.

  Two refusals share one type on purpose. A caller that catches this
  needs to do the same thing for both -- stop trading -- so splitting
  them would invite a handler that clears one and not the other.
  '''


class KillSwitch:
  '''A latching kill switch backed by a JSON state file.

  Not a dataclass: it owns its own constructor because the constructor
  loads persisted state, and a generated ``__init__`` would make that
  load a separate step a caller could forget.

  Attributes:
    algo_id: Algo ID this switch belongs to.
    path: State file backing the latch.
    thresholds: Pre-defined automatic trip conditions.
  '''

  algo_id: str
  path: Path
  thresholds: KillThresholds

  def __init__(
    self,
    algo_id: str,
    path: str | Path | None = None,
    thresholds: KillThresholds = KillThresholds(),
    clock: Clock = utcnow,
  ) -> None:
    '''Build a switch and load any persisted state.

    Args:
      algo_id: Algo ID this switch belongs to. Recorded in the state
        file and checked on load.
      path: State file. Defaults to :func:`default_state_file`. A
        missing file is a first run and starts armed; that is the only
        condition under which this class reports itself armed without
        having been told to.
      thresholds: Pre-defined automatic trip conditions.
      clock: Wall-clock source. Injectable so the state file is
        deterministic under test and so no test waits for time.

    Raises:
      KillSwitchError: If the Algo ID is blank, or a persisted state file
        exists but cannot be parsed, is not a JSON object, carries the
        wrong schema, or belongs to a different Algo ID. Each of those
        raises rather than defaulting, because defaulting is precisely
        how a halted strategy comes back armed.
    '''
    if not isinstance(algo_id, str) or not algo_id.strip():
      raise KillSwitchError(
        'algo_id must not be blank; SEBI scopes the kill switch per Algo '
        'ID and an unnamed switch cannot be shown to be one')
    self.algo_id = algo_id
    self.path = Path(path) if path is not None \
      else default_state_file(algo_id)
    self.thresholds = thresholds
    self._clock = clock
    self._tripped = False
    self._trips: list[Trip] = []
    self._clears: list[str] = []
    self._load()

  @property
  def tripped(self) -> bool:
    '''Return whether trading is halted.

    Sticky in one direction only. There is no setter and no other method
    that clears it, so a recovering metric cannot untrip this switch.
    '''
    return self._tripped

  @property
  def trading_enabled(self) -> bool:
    '''Return whether an order may be released.

    The property an order path should branch on. It is False whenever
    the switch is tripped and there is no other condition under which it
    is False, so a caller that forgets some other risk still stops
    here.
    '''
    return not self._tripped

  @property
  def history(self) -> tuple[Trip, ...]:
    '''Return every trip recorded for this Algo ID, oldest first.'''
    return tuple(self._trips)

  @property
  def clears(self) -> tuple[str, ...]:
    '''Return every recorded clear, oldest first, each with its reason.'''
    return tuple(self._clears)

  def _load(self) -> None:
    '''Read persisted state, or establish that there is none.

    Raises:
      KillSwitchError: If the file exists but is unreadable, is not a
        JSON object, has the wrong schema, or names another Algo ID.
    '''
    if not self.path.exists():
      return
    try:
      payload = json.loads(self.path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
      raise KillSwitchError(
        f'kill switch state at {self.path} cannot be read ({exc}); a '
        'truncated safety state carries no information, so this process '
        'refuses to trade rather than re-arming') from exc
    if not isinstance(payload, dict):
      raise KillSwitchError(
        f'kill switch state at {self.path} is not a JSON object')
    schema = payload.get('schema')
    if schema != kill_switch_schema:
      raise KillSwitchError(
        f'kill switch state schema {schema!r} is not '
        f'{kill_switch_schema!r}; refusing to interpret it')
    if payload.get('algo_id') != self.algo_id:
      owner = payload.get('algo_id')
      raise KillSwitchError(
        f'kill switch state at {self.path} belongs to algo {owner!r}, not '
        f'{self.algo_id!r}')
    trips = payload.get('trips', [])
    clears = payload.get('clears', [])
    if not isinstance(trips, list) or not isinstance(clears, list):
      raise KillSwitchError(
        f'kill switch state at {self.path} has a malformed trips or '
        'clears section')
    self._trips = [
      Trip.from_json(item) for item in trips if isinstance(item, dict)]
    self._clears = [str(item) for item in clears]
    self._tripped = bool(payload.get('tripped'))

  def _persist(self) -> None:
    '''Write the current state atomically.

    A temporary file in the same directory is renamed into place, so a
    crash mid-write leaves the previous state intact rather than a
    half-written one. A half-written state file fails to parse on the
    next start, which is safe but turns a trading process into one that
    cannot start at all.

    Raises:
      KillSwitchError: If the state cannot be written.
    '''
    payload = {
      'schema': kill_switch_schema,
      'algo_id': self.algo_id,
      'tripped': self._tripped,
      'trips': [trip.to_json() for trip in self._trips],
      'clears': list(self._clears),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + '\n'
    temporary = self.path.with_name(f'{self.path.name}.tmp')
    try:
      self.path.parent.mkdir(parents=True, exist_ok=True)
      temporary.write_text(text, encoding='utf-8')
      os.replace(temporary, self.path)
    except OSError as exc:
      raise KillSwitchError(
        f'kill switch state at {self.path} cannot be written ({exc}); a '
        'trip that is not persisted is a trip a restart undoes') from exc

  def trip(self, detail: str, code: str = manual_code,
           observed: float = 0.0) -> Trip:
    '''Trip the switch and persist it.

    Recorded even when the switch is already tripped. Repeated trips are
    evidence of a flapping condition, and dropping them would hide it.

    Args:
      detail: What tripped it, recorded verbatim.
      code: One of :data:`trip_codes`. Defaults to the manual trip.
      observed: The measured value behind the trip.

    Returns:
      The recorded :class:`Trip`.

    Raises:
      KillSwitchError: If the code is unknown or the state cannot be
        written.
    '''
    if code not in trip_codes:
      raise KillSwitchError(
        f'unknown trip code {code!r}; known: {trip_codes}')
    record = Trip(code=code, detail=detail, at=_stamp(self._clock),
                  observed=observed)
    self._trips.append(record)
    self._tripped = True
    self._persist()
    return record

  def evaluate(
    self,
    index_change: float = 0.0,
    drawdown: float = 0.0,
    fno_banned: bool = False,
    ack_latency: float | None = None,
  ) -> Trip | None:
    '''Trip on any breached pre-defined condition.

    This is the SEBI-mandated path: automatic, on conditions declared
    before the fact rather than at the moment of failure. Every condition
    is evaluated, not short-circuited, because the first condition to
    fire is frequently not the one that mattered and the others are the
    explanation.

    Args:
      index_change: Session change in the reference index, as a
        fraction. Negative is a fall.
      drawdown: Peak-to-trough drawdown, as a positive fraction.
      fno_banned: Whether the traded symbol is under an F&O ban.
      ack_latency: Seconds between send and acknowledgement, or None if
        no order is outstanding.

    Returns:
      The first trip recorded on this call, or None if nothing tripped.

    Raises:
      KillSwitchError: If a trip cannot be persisted.
    '''
    recorded: Trip | None = None
    for code, detail, observed in self._breaches(
      index_change, drawdown, fno_banned, ack_latency
    ):
      if recorded is None:
        recorded = self.trip(detail, code=code, observed=observed)
      else:
        # Later conditions in the same evaluation are recorded without
        # a second write, so one bad evaluation writes one file.
        self._trips.append(Trip(
          code=code, detail=detail, at=_stamp(self._clock),
          observed=observed))
    return recorded

  def _breaches(
    self,
    index_change: float,
    drawdown: float,
    fno_banned: bool,
    ack_latency: float | None,
  ) -> tuple[tuple[str, str, float], ...]:
    '''Return every pre-defined condition currently breached.

    Args:
      index_change: Session change in the reference index.
      drawdown: Peak-to-trough drawdown.
      fno_banned: Whether the traded symbol is under an F&O ban.
      ack_latency: Seconds between send and acknowledgement, or None.

    Returns:
      One ``(code, detail, observed)`` triple per breached condition,
      empty when clear.
    '''
    found: list[tuple[str, str, float]] = []
    if index_change <= -self.thresholds.index_fall:
      found.append((
        index_fall_code,
        f'index fell {abs(index_change):.2%} in the session, at or beyond '
        f'the {self.thresholds.index_fall:.2%} limit',
        index_change))
    if drawdown >= self.thresholds.max_drawdown:
      found.append((
        drawdown_code,
        f'drawdown {drawdown:.2%} is at or beyond the '
        f'{self.thresholds.max_drawdown:.2%} limit',
        drawdown))
    if self.thresholds.fno_ban and fno_banned:
      found.append((
        fno_ban_code,
        'the traded symbol is under an exchange F&O ban', 1.0))
    if ack_latency is not None and \
        ack_latency > self.thresholds.ack_latency:
      found.append((
        ack_latency_code,
        f'order acknowledgement took {ack_latency:.2f}s, beyond the '
        f'{self.thresholds.ack_latency:.2f}s limit',
        ack_latency))
    return tuple(found)

  def breached(self, index_change: float = 0.0, drawdown: float = 0.0,
               fno_banned: bool = False,
               ack_latency: float | None = None) -> tuple[str, ...]:
    '''Return the pre-defined conditions currently breached, read-only.

    The counterpart to :meth:`evaluate`, so an operator can be shown
    *why* the switch refuses to clear before being asked to clear it.

    Args:
      index_change: Session change in the reference index.
      drawdown: Peak-to-trough drawdown.
      fno_banned: Whether the traded symbol is under an F&O ban.
      ack_latency: Seconds between send and acknowledgement, or None.

    Returns:
      Human-readable detail per breached condition, empty when clear.
    '''
    return tuple(detail for _, detail, _ in self._breaches(
      index_change, drawdown, fno_banned, ack_latency))

  def reset(
    self,
    reason: str,
    index_change: float = 0.0,
    drawdown: float = 0.0,
    fno_banned: bool = False,
    ack_latency: float | None = None,
  ) -> None:
    '''Clear the switch, explicitly, with a stated reason.

    Three refusals, in the order a caller meets them, and all of them
    loud. This is the only method that can untrip the switch, and it
    demands three things: that the switch is currently tripped, a
    non-blank reason, and the caller's *current* metrics, which it checks
    against the same pre-defined conditions :meth:`evaluate` uses.

    Passing no metrics reads as ``0.0``, which is clear. That is the
    documented contract, not a loophole: it means the caller is asserting
    the conditions are resolved, and a caller asserting that falsely is
    the one thing this cannot detect.

    Args:
      reason: Why a human is clearing it. Persisted with the trip
        history.
      index_change: Session change in the reference index, now.
      drawdown: Peak-to-trough drawdown, now.
      fno_banned: Whether the traded symbol is under an F&O ban, now.
      ack_latency: Seconds between send and acknowledgement, now, or
        None if no order is outstanding.

    Raises:
      KillSwitchLatched: If the switch is not tripped, if the reason is
        blank, or if any pre-defined condition is still breached. The
        message names the breached conditions, because the operator's
        next action depends on which one is the problem.
    '''
    if not self._tripped:
      raise KillSwitchLatched(
        'kill switch is not tripped; there is nothing to clear, and a '
        'clear that does nothing is indistinguishable at runtime from a '
        'clear that worked')
    if not isinstance(reason, str) or not reason.strip():
      raise KillSwitchLatched(
        'clearing the kill switch requires a stated reason; an '
        'unattributed clear is how an automated restart silently re-arms '
        'a halted strategy')
    breached = self.breached(index_change, drawdown, fno_banned,
                             ack_latency)
    if breached:
      raise KillSwitchLatched(
        'kill switch cannot be cleared while a pre-defined condition is '
        'still breached: ' + '; '.join(breached))
    self._tripped = False
    self._clears.append(reason)
    self._persist()

  def guard(self, index_change: float = 0.0, drawdown: float = 0.0,
            fno_banned: bool = False,
            ack_latency: float | None = None) -> None:
    '''Evaluate the conditions and raise if trading is not permitted.

    The single call an order path should make, because it cannot forget
    either half: it evaluates the automatic conditions *and* refuses
    when the switch is already latched.

    Args:
      index_change: Session change in the reference index.
      drawdown: Peak-to-trough drawdown.
      fno_banned: Whether the traded symbol is under an F&O ban.
      ack_latency: Seconds between send and acknowledgement, or None.

    Raises:
      KillSwitchLatched: If the switch is tripped, or a condition
        breached on this call.
    '''
    self.evaluate(index_change, drawdown, fno_banned, ack_latency)
    if not self._tripped:
      return
    reasons = self.breached(index_change, drawdown, fno_banned,
                            ack_latency)
    detail = '; '.join(reasons) if reasons else (
      self._trips[-1].detail if self._trips else 'no condition recorded')
    raise KillSwitchLatched(
      f'kill switch latched for algo {self.algo_id}: {detail}. It stays '
      'tripped until a human clears it with a reason')

  def render(self) -> str:
    '''Return a one-line status for a log.

    Returns:
      Human-readable status naming the Algo ID, the latch state and the
      most recent trip or clear.
    '''
    state = 'TRIPPED' if self._tripped else 'armed'
    if self._tripped and self._trips:
      latest = self._trips[-1].detail
    elif self._clears:
      # Armed with a clear on record: when it was last cleared is the
      # operator's first question, and the trips that led there are still
      # in :attr:`history` if they want them.
      latest = f'last cleared: {self._clears[-1]}'
    else:
      latest = 'no trips recorded'
    return f'kill switch {self.algo_id}: {state}; {latest}'


def _stamp(clock: Clock) -> str:
  '''Return an ISO 8601 timestamp from the injected clock.

  Args:
    clock: Wall-clock source.

  Returns:
    The time in ISO 8601, UTC. A naive datetime is stamped as UTC
    rather than rejected: the only clock this project injects is a test
    clock, and refusing it would be a check against our own tests rather
    than against any risk.
  '''
  moment = clock()
  if moment.tzinfo is None:
    moment = moment.replace(tzinfo=timezone.utc)
  return moment.isoformat()
