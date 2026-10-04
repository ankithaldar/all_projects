#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Circuit bands and corporate-action adjustment, on OHLCV series.

Two unrelated jobs that share a file because they are the same job seen
twice: **reconciling a price series against the rules that move prices**.
A circuit band is the exchange deciding the price is not real, and a
corporate action is the company deciding the price is not comparable.
Both destroy the naive assumption that yesterday's close and today's
close are the same thing.

**Circuit bands.** A band is two-sided and configured, not derived:
``lower_pct`` and ``upper_pct`` are percentages the exchange sets per
scrip and revises on a schedule, and this module does not hardcode them
because a hardcoded band is wrong the day the circular changes. What
this module owns is the arithmetic -- reference close to band, rounded
to the tick -- and the classification of a price against the band.

The three-way classification is the part that matters operationally:

  * **at the band** (``circuit_lower`` / ``circuit_upper``): the market
    is at the band, where only orders at the band price trade;
  * **inside the no-trade window** (``no_trade_lower`` /
    ``no_trade_upper``): trading in the scrip is suspended because a
    market approaching its band is where a crossed book does damage;
  * **normal**: nothing special.

The no-trade percentages are venue configuration, exactly as the band
percentages are. The widths in the project's research corpus were not
verified against a current primary circular, so they are constructor
arguments with no defaults chosen to look authoritative. What *is*
encoded is the mechanism and the ordering: the band is tested before the
window, because a price on the band is at the band regardless of how
wide the window is.

**Corporate actions are where backtests go to die.** An unadjusted
series carries a 1:1 bonus as a 50 percent price crash on the ex-date,
which a momentum strategy reads as a crash and a mean-reversion strategy
reads as a gift. Both are wrong, and the error is silent because the
backtest completes and reports a number.

So the adjustment here is explicit, both series are returned, and the
factors are exposed rather than buried:

  * **bonus** ratio ``b``: price divides by ``1 + b``, volume multiplies
    by ``1 + b``. A 1:1 bonus turns 3500 into 1750 and doubles the
    share count, so value is preserved and share count is not.
  * **split** ratio ``s`` (new shares per old share): price divides by
    ``s``, volume multiplies by ``s``.
  * **dividend** ``d`` per share: the theoretical ex price is
    ``previous close - d``. Prices before the ex-date are scaled by the
    ratio of theoretical ex price to previous close, which is the
    general backward-adjustment step and is what makes several actions in
    one series compose correctly. Volume is **not** adjusted for a
    dividend, because a dividend does not change the share count and a
    volume series inflated by ``close / (close - d)`` is simply wrong.

All bars strictly before the ex-date are adjusted; the ex-date bar is
not, because it already trades at the ex price. Adjusting it would apply
the action twice, which is the mirror-image of not adjusting at all.

An action whose ex-date precedes the first bar is **refused**, not
approximated: the previous close needed for the ratio would have to be
invented, and an invented reference close silently rescales the entire
series. Prepend history instead.
'''

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from stock_rl.bars import Bar
from stock_rl.risk.checks import PriceBand

__all__ = [
  'AdjustedSeries',
  'CircuitBand',
  'CorporateAction',
  'action_kinds',
  'band_for',
  'bonus',
  'circuit_states',
  'classify',
  'corporate_actions',
  'dividend',
  'ex_price',
  'halted_states',
  'normal',
  'no_trade_lower',
  'no_trade_upper',
  'circuit_lower',
  'circuit_upper',
  'round_to_tick',
  'split',
  'split_adjusted_bars',
  'states',
]

#: Recognised corporate action kinds.
bonus = 'bonus'
split = 'split'
dividend = 'dividend'
action_kinds = (bonus, split, dividend)

#: The price is inside the no-trade window below the lower band.
no_trade_lower = 'no_trade_lower'

#: The price is inside the no-trade window above the upper band.
no_trade_upper = 'no_trade_upper'

#: The price is at or below the lower circuit band.
circuit_lower = 'circuit_lower'

#: The price is at or above the upper circuit band.
circuit_upper = 'circuit_upper'

#: The price is trading normally.
normal = 'normal'

#: Every classification :func:`classify` can return.
states = (
  no_trade_lower, no_trade_upper, circuit_lower, circuit_upper, normal,
)

#: Classifications that must not be traded at all.
halted_states = frozenset({no_trade_lower, no_trade_upper})

#: Classifications where only orders at the band price are accepted.
circuit_states = frozenset({circuit_lower, circuit_upper})


@dataclass(frozen=True, slots=True)
class CircuitBand:
  '''A market or scrip circuit band with its no-trade window.

  Attributes:
    lower_pct: Lower band as a fraction of the reference price. A 20
      percent band is ``0.20``.
    upper_pct: Upper band as a fraction of the reference price.
    no_trade_pct: Width of the no-trade window measured inward from
      each band, as a fraction of the reference price. A price inside
      the window cannot be traded even though it is inside the band.
    tick: Exchange price increment the bands are rounded to.
  '''

  lower_pct: float
  upper_pct: float
  no_trade_pct: float = 0.0
  tick: float = 0.05

  def __post_init__(self) -> None:
    '''Validate the band shape at construction.

    The no-trade window is measured inward from **both** edges, so it is
    validated against ``min(lower_pct, upper_pct)`` rather than against
    one side. A one-sided band is legal -- ``lower_pct`` of 0.0 is how a
    scrip with only an upper circuit is expressed -- and checking the
    window against ``upper_pct`` alone accepted ``CircuitBand(0.0, 0.20,
    no_trade_pct=0.15)``, whose window runs from 115 down to 105. A
    shortened chain still verifies clean, and so does a chain that is
    internally consistent but shorter than it was; a window that crosses
    over itself is the same class of error, and it makes
    :func:`classify` report the wrong side of the market.

    A **zero** window is exempt from that comparison. ``lower_pct`` of
    ``0.0`` is how a one-sided band is expressed, and a band with no
    window at all has nothing to cross over, so refusing it would make
    the one-sided band unrepresentable for no gain.

    Raises:
      ValueError: If any percentage is negative, the band is not
        positive on the upper side, a non-zero no-trade window is at
        least as wide as the narrower of the two bands, or the tick is
        not positive. A window that swallows normal trading on one side
        is a configuration error rather than a conservative setting.
    '''
    for name in ('lower_pct', 'upper_pct', 'no_trade_pct'):
      value = getattr(self, name)
      if not 0.0 <= value < 1.0:
        raise ValueError(f'{name} must be in [0, 1), got {value}')
    if self.upper_pct <= 0.0:
      raise ValueError(
        f'upper_pct must be positive, got {self.upper_pct}')
    # The window is measured inward from BOTH edges, so it has to fit
    # inside the NARROWER of the two bands. Comparing it to upper_pct
    # alone is what let an asymmetric band carry an inverted window:
    # lower_pct of 0.0 is a legal one-sided band, and against an
    # upper_pct of 0.20 a window of 0.15 passed the check while running
    # from 115 up to 105. Every price between those two was then tested
    # against the lower edge first, so a price near the top of the band
    # came back no_trade_lower and any caller branching on the state
    # acted on the wrong side of the market.
    narrower = min(self.lower_pct, self.upper_pct)
    if self.no_trade_pct > 0.0 and self.no_trade_pct >= narrower:
      side = 'lower_pct' if self.lower_pct <= self.upper_pct \
        else 'upper_pct'
      raise ValueError(
        f'no_trade_pct {self.no_trade_pct} must be narrower than '
        f'{side} {narrower}, which is the narrower band; a window at or '
        'beyond it swallows normal trading on one side and, measured '
        'inward from both edges, crosses over itself so that the top of '
        'the band reads as the bottom')
    if self.tick <= 0.0:
      raise ValueError(f'tick must be positive, got {self.tick}')

  def band(self, reference: float) -> PriceBand:
    '''Return this band's price bounds for a reference price.

    Args:
      reference: Reference price, normally the previous close.

    Returns:
      The tick-rounded band.

    Raises:
      ValueError: If the reference price is not positive. A reference of
        zero would place the lower band at zero, which is not a price
        anybody can trade.
    '''
    if reference <= 0.0:
      raise ValueError(f'reference must be positive, got {reference}')
    return PriceBand(
      round_to_tick(reference * (1.0 - self.lower_pct), self.tick),
      round_to_tick(reference * (1.0 + self.upper_pct), self.tick))


def round_to_tick(price: float, tick: float) -> float:
  '''Round a price to the nearest multiple of the tick size.

  Rounds half **up**, not to even. The exchange rounds a computed price
  band to the nearest tradable tick and does not use banker's rounding,
  so a band computed with ``round()`` lands on a different tick than the
  exchange's band for exactly the half-way cases -- which is a band that
  is off by one tick, and therefore a price the exchange would not have
  allowed.

  Args:
    price: Price to round.
    tick: Tick size. Must be positive.

  Returns:
    The rounded price.

  Raises:
    ValueError: If the tick is not positive, or the price is not
      finite.
  '''
  if tick <= 0.0:
    raise ValueError(f'tick must be positive, got {tick}')
  if not math.isfinite(price):
    raise ValueError(f'price must be finite, got {price}')
  steps = math.floor(price / tick + 0.5)
  return steps * tick


def band_for(reference: float, band: CircuitBand) -> PriceBand:
  '''Return the price band for a reference price.

  A named function rather than only a method so a caller holding a
  percentage pair from a circular can get a band without constructing a
  :class:`CircuitBand` first.

  Args:
    reference: Reference price, normally the previous close.
    band: The circuit band definition.

  Returns:
    The tick-rounded :class:`~stock_rl.risk.checks.PriceBand`.
  '''
  return band.band(reference)


def classify(price: float, reference: float, band: CircuitBand) -> str:
  '''Classify a traded price against its circuit band.

  Order matters and is deliberate. The band is tested first, because a
  price on the band is at the band whatever the width of the window; the
  window is then tested strictly inside it. Both bands are tested before
  either window, so a price cannot be classified as a no-trade halt when
  it is actually at a circuit.

  Args:
    price: Price to classify.
    reference: Reference price the band was computed from.
    band: The circuit band definition.

  Returns:
    One of :data:`states`.

  Raises:
    ValueError: If the reference price is not positive, or the traded
      price is not positive. A zero price is not "at the lower circuit",
      it is a missing value, and returning a trading state for it would
      be the wrong kind of answer.
  '''
  bounds = band.band(reference)
  if price <= 0.0:
    raise ValueError(f'price must be positive, got {price}')
  if price >= bounds.upper:
    return circuit_upper
  if price <= bounds.lower:
    return circuit_lower
  window_low = bounds.lower + band.no_trade_pct * reference
  window_high = bounds.upper - band.no_trade_pct * reference
  if price <= window_low:
    return no_trade_lower
  if price >= window_high:
    return no_trade_upper
  return normal


@dataclass(frozen=True, slots=True)
class CorporateAction:
  '''One bonus, split or dividend.

  Attributes:
    kind: One of :data:`action_kinds`.
    ex_date: First bar that trades ex the action. Bars strictly before
      it are adjusted; the bar on it is not.
    ratio: For a bonus, shares added per share held, so ``1.0`` is 1:1.
      For a split, new shares per old share, so ``2.0`` is 1:2. Ignored
      for a dividend.
    amount: Dividend per share in rupees. Ignored for a bonus or split.
  '''

  kind: str
  ex_date: datetime
  ratio: float = 0.0
  amount: float = 0.0

  def __post_init__(self) -> None:
    '''Validate the action at construction.

    Raises:
      ValueError: If the kind is unknown, the ratio is not positive for
        a bonus or split, or the amount is not positive for a dividend.
        A zero or negative economic value is rejected here rather than
        being allowed to become a factor of one that adjusts nothing and
        looks like it worked.
    '''
    if self.kind not in action_kinds:
      raise ValueError(
        f'kind must be one of {action_kinds}, got {self.kind!r}')
    if self.kind == dividend:
      if self.amount <= 0.0:
        raise ValueError(
          f'dividend amount must be positive, got {self.amount}')
    elif self.ratio <= 0.0:
      raise ValueError(
        f'{self.kind} ratio must be positive, got {self.ratio}')

  def share_multiplier(self) -> float:
    '''Return the factor by which share count changes.

    Args:
      None.

    Returns:
      ``1 + ratio`` for a bonus, ``ratio`` for a split, ``1.0`` for a
      dividend, because a dividend does not change the share count.
    '''
    if self.kind == bonus:
      return 1.0 + self.ratio
    if self.kind == split:
      return self.ratio
    return 1.0

  def theoretical_price(self, reference: float) -> float:
    '''Return the theoretical ex price for a pre-action price.

    Args:
      reference: Price immediately before the action, normally the
        close of the last bar before the ex-date.

    Returns:
      The ex price the action implies, before tick rounding. The
      exchange publishes its own rounded figure; this is the
      unrounded theoretical value used as the adjustment ratio's
      numerator.

    Raises:
      ValueError: If the reference price is not positive.
    '''
    if reference <= 0.0:
      raise ValueError(f'reference must be positive, got {reference}')
    if self.kind == bonus:
      return reference / (1.0 + self.ratio)
    if self.kind == split:
      return reference / self.ratio
    return reference - self.amount


def ex_price(reference: float, action: CorporateAction) -> float:
  '''Return the theoretical ex price for one action.

  Args:
    reference: Price immediately before the action.
    action: The action being applied.

  Returns:
    The unrounded theoretical ex price.
  '''
  return action.theoretical_price(reference)


def corporate_actions(
  series: Sequence[CorporateAction]) -> tuple[CorporateAction, ...]:
  '''Return the actions in ex-date order.

  Ordering is not cosmetic. Two actions whose ex-dates fall between the
  same two bars must be applied oldest first: applying a split before a
  bonus whose ratio was quoted pre-split produces a compounded factor
  that is wrong by the square of the split. Sorting by date is the only
  order in which the quoted ratios are the ones applied.

  Args:
    series: Actions in any order.

  Returns:
    Actions sorted by ex-date, ties broken by nothing. Two actions on
    the same ex-date are order-dependent and are a data error the caller
    should resolve upstream; the stable sort preserves the caller's order
    for them rather than inventing one.
  '''
  return tuple(sorted(series, key=lambda action: action.ex_date))


@dataclass(frozen=True, slots=True)
class AdjustedSeries:
  '''Both series, plus the factors that relate them.

  Attributes:
    adjusted: Bars with every action applied backward.
    unadjusted: The bars exactly as supplied.
    actions: The actions applied, in the order they were applied.
    share_multiplier: Cumulative factor by which share count changed.
      Divide an original share count by this to get the adjusted count;
      multiply an adjusted count by it to get the original.
    price_multiplier: Cumulative factor by which prices were scaled.
    dividend_total: Total dividend per original share, in rupees.
  '''

  adjusted: tuple[Bar, ...]
  unadjusted: tuple[Bar, ...]
  actions: tuple[CorporateAction, ...]
  share_multiplier: float
  price_multiplier: float
  dividend_total: float


def split_adjusted_bars(
  bars: Sequence[Bar],
  actions: Sequence[CorporateAction],
) -> AdjustedSeries:
  '''Apply corporate actions backward across a bar series.

  Both series are returned. A caller that keeps only the adjusted one
  cannot compute the unadjusted return the exchange printed, which is
  exactly the number an execution-versus-backtest reconciliation needs;
  a caller that keeps only the unadjusted one has a 1:1 bonus in its
  momentum signal. Returning both makes the choice explicit rather than
  silent.

  Args:
    bars: Bars in ascending time order.
    actions: Corporate actions to apply. Sorted internally by ex-date.

  Returns:
    An :class:`AdjustedSeries`.

  Raises:
    ValueError: If the bar series is empty, its timestamps are not
      strictly ascending, an action's ex-date precedes the first bar,
      or an ex-date falls after the last bar. Each of these means the
      adjustment cannot be computed from the data given, and guessing
      would rescale the whole series silently.
  '''
  if not bars:
    raise ValueError('bars must not be empty')
  for index in range(1, len(bars)):
    if bars[index].timestamp <= bars[index - 1].timestamp:
      raise ValueError(
        f'bars must be strictly ascending by timestamp; bar {index} at '
        f'{bars[index].timestamp} does not follow '
        f'{bars[index - 1].timestamp}')
  first = bars[0].timestamp
  last = bars[-1].timestamp
  ordered = corporate_actions(actions)
  for action in ordered:
    if action.ex_date < first:
      raise ValueError(
        f'{action.kind} ex-date {action.ex_date} precedes the first bar '
        f'{first}; the reference close needed to adjust the series is '
        'not in the data. Prepend earlier history rather than letting '
        'the series be rescaled by a guessed reference')
    if action.ex_date > last:
      raise ValueError(
        f'{action.kind} ex-date {action.ex_date} is after the last bar '
        f'{last}; there is no bar to adjust, so this action belongs to '
        'a different series')
  # Walk the actions newest first, extending the adjustment to a longer
  # and longer prefix each time. The earliest action is processed last
  # and therefore reaches every bar, which is what makes the factors
  # compose: a bar before all three actions carries all three ratios, and
  # a bar between the second and third carries only the third.
  steps: list[tuple[float, float] | None] = [None] * len(bars)
  price_factor = 1.0
  share_factor = 1.0
  dividend_total = 0.0
  for action in reversed(ordered):
    last_before = _last_index_before(bars, action.ex_date)
    reference = bars[last_before].close
    price_factor *= action.theoretical_price(reference) / reference
    share_factor *= action.share_multiplier()
    if action.kind == dividend:
      dividend_total += action.amount
    for index in range(last_before + 1):
      steps[index] = (price_factor, share_factor)
  adjusted = tuple(
    _adjust(bar, steps[index]) for index, bar in enumerate(bars))
  return AdjustedSeries(
    adjusted=adjusted,
    unadjusted=tuple(bars),
    actions=ordered,
    share_multiplier=share_factor,
    price_multiplier=price_factor,
    dividend_total=dividend_total,
  )


def _last_index_before(bars: Sequence[Bar], ex_date: datetime) -> int:
  '''Return the index of the last bar trading before an ex-date.

  Args:
    bars: Bars in ascending time order.
    ex_date: Ex-date of the action.

  Returns:
    Zero-based index of the last bar with a timestamp strictly before
    ``ex_date``.

  Raises:
    ValueError: If no bar precedes the ex-date, which
      :func:`split_adjusted_bars` catches earlier with a message naming
      the missing history. This is the defensive branch so the helper is
      safe to read on its own.
  '''
  for index in range(len(bars) - 1, -1, -1):
    if bars[index].timestamp < ex_date:
      return index
  raise ValueError(f'no bar precedes ex-date {ex_date}')


def _adjust(bar: Bar, step: tuple[float, float] | None) -> Bar:
  '''Return one bar with a cumulative adjustment applied.

  Args:
    bar: The original bar.
    step: Tuple of (cumulative price ratio, cumulative share multiplier)
      as at this bar, or None for a bar on or after the last ex-date,
      which is left untouched.

  Returns:
    The adjusted bar, with volume scaled by the share multiplier and
    rounded to a whole number of shares.
  '''
  if step is None:
    return bar
  ratio, shares = step
  volume = round(bar.volume * shares) if bar.volume else 0.0
  return Bar(
    timestamp=bar.timestamp,
    open=bar.open * ratio,
    high=bar.high * ratio,
    low=bar.low * ratio,
    close=bar.close * ratio,
    volume=float(volume),
  )
