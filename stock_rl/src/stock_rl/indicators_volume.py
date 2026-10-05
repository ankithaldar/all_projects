#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Volume-based indicators over a bar series.

**Read this before wiring any of these into a strategy.**

Implementing a published formula is not evidence that the formula has
predictive value. Every function below is the arithmetic its author
published and nothing else. The project's own research record
(``docs/research/methodology/technical-indicators-nse.md``) says so
plainly for this family in particular: "OBV / VWAP -- no Indian
evidence. Rink (2023) excluded volume rules for lack of data -- the
'OBV + MA' backtest is untested territory", and "Volume as a
conditioning variable has evidence; OBV as a signal does not."
Bornholt, Dou and Malin (2015) found volume-based early-stage momentum
beat traditional momentum in 34 of 37 countries, which is an argument
about tilt, not about these signals. FINSABER (KDD 2026) additionally
measured standalone technical signals reading *inverted* out of sample.
This module exists so a caller can choose to test a formula. It does not
assert that any of them works on NSE-listed equities.

**No threshold is fitted anywhere.** The window lengths are arguments
with conventional defaults; nothing here encodes an entry, exit or
regime level, and no function returns a signal. A caller who adds a
threshold owns the evidence for it.

**Contract**, matching :mod:`stock_rl.indicators`:

* every function returns a list the same length as its input;
* ``None`` marks an undefined value -- warm-up, a missing prior bar or a
  guarded division -- and never a substituted zero;
* the value at index ``i`` uses bars up to ``i`` only;
* a non-positive window raises ``ValueError`` naming the parameter;
* every division tests its denominator first and the guarded branch
  returns ``None``, because a plausible wrong number is worse than an
  obviously missing one.

**Recursive accumulators and their seeds.** Four of these functions
(``on_balance_volume``, ``price_volume_trend``, ``positive_volume_index``
and ``negative_volume_index``) integrate rather than average, so their
level depends on every earlier bar. The additive ones seed at zero on
index 0 -- a definitional constant, not a measurement -- and are
therefore defined from index 0. The multiplicative ones cannot: seeding a
product at zero annihilates it, so they are ``None`` until the first bar
that actually qualifies, and each docstring states that boundary.

Every function reads a ``list[Bar]`` from :mod:`stock_rl.bars`, and each
docstring names the fields it touches. Note that
:attr:`stock_rl.bars.Bar.volume` is documented as possibly zero, because
illiquid bars legitimately print none; that makes the volume denominators
here genuinely reachable, not hypothetical.

# ponytail: :func:`chaikin_money_flow`, :func:`ease_of_movement` and
# :func:`volume_oscillator` rebuild their trailing window at every index,
# so each is O(n * window). Ceiling: irrelevant on the daily panels this
# project runs, wasteful in a minute-bar parameter sweep. Upgrade path:
# rolling sums for the volume totals and a running count of the bars a
# window had to drop, which is also what a caller would need to report how
# much of a window was uninformative. No returned value would change.
'''

from __future__ import annotations

from statistics import fmean

from stock_rl.bars import Bar
from stock_rl.indicators import ema

__all__ = [
  'chaikin_money_flow',
  'ease_of_movement',
  'elder_ray',
  'force_index',
  'negative_volume_index',
  'on_balance_volume',
  'positive_volume_index',
  'price_volume_trend',
  'volume_oscillator',
]

#: Bull and bear power as two aligned series. Neither has a warm-up
#: region, so both are plain float lists.
Ray = tuple[list[float], list[float]]

#: Base level for the two volume indices. Fosback published them as
#: starting from an arbitrary level; 100 makes the result a
#: percentage-style index that starts at 100. It cannot be zero: the
#: recursion is multiplicative, so a zero seed would pin every later
#: value at zero and the series would look like a real reading forever.
VOLUME_INDEX_BASE = 100.0


def on_balance_volume(bars: list[Bar]) -> list[float]:
  '''Return Granville's On Balance Volume.

  Add the volume on a bar whose close rose, subtract it on a bar whose
  close fell, and leave it alone when the close is unchanged. The rule is
  entirely additive, so nothing can divide by zero.

  Fields: ``close`` for the direction, ``volume`` for the increment.

  Index 0 is ``0.0``. That is a definitional seed rather than a
  measurement: the running total starts there because there is no prior
  bar to compare against, and the absolute level of an OBV series is
  meaningless in any case since it depends on that seed. The first
  informative value is at index 1.

  Args:
    bars: Price bars in ascending time order.

  Returns:
    List the same length as ``bars``, never containing ``None``.
  '''
  out: list[float] = []
  total = 0.0
  for index, bar in enumerate(bars):
    if index:
      change = bar.close - bars[index - 1].close
      if change > 0.0:
        total += bar.volume
      elif change < 0.0:
        total -= bar.volume
    out.append(total)
  return out


def price_volume_trend(bars: list[Bar]) -> list[float | None]:
  '''Return Price Volume Trend.

  ``PVT += volume * (close - prior_close) / prior_close`` on every bar.
  Unlike OBV the increment is scaled by the size of the move, so a large
  down bar with light volume contributes less than a small down bar with
  heavy volume.

  Fields: ``close`` for the move, ``volume`` for the scale.

  Index 0 is ``0.0``, a definitional seed: there is no prior close to
  divide by, so nothing has been accumulated yet.

  **A non-positive prior close makes the increment undefined.** The
  running total is then frozen -- the value at that index is ``None`` and
  the total carries forward unchanged, so the next usable bar still
  measures from the last bar that actually moved. Zero is not
  substituted: a bar with no prior close has not contributed zero, it has
  contributed an unknown amount, and treating those as the same thing is
  how a divergence quietly becomes a position size.

  Args:
    bars: Price bars in ascending time order.

  Returns:
    List the same length as ``bars``, ``None`` at any index whose prior
    close is not positive.
  '''
  out: list[float | None] = []
  total = 0.0
  for index, bar in enumerate(bars):
    if index == 0:
      out.append(0.0)
      continue
    prior_close = bars[index - 1].close
    if prior_close <= 0.0:
      out.append(None)
      continue
    total += bar.volume * (bar.close - prior_close) / prior_close
    out.append(total)
  return out


def _volume_index(bars: list[Bar], rising: bool) -> list[float | None]:
  '''Return a volume index that moves only on qualifying volume.

  The recursion is multiplicative: on a qualifying bar the level is
  multiplied by the close ratio, on any other bar it is carried forward
  unchanged. Both published indices differ only in which bar qualifies.

  Args:
    bars: Price bars in ascending time order.
    rising: True for the positive volume index, False for the negative.

  Returns:
    List the same length as ``bars``, ``None`` until the first usable
    qualifying bar and at any qualifying bar whose price ratio is
    undefined.
  '''
  out: list[float | None] = [None] * len(bars)
  level: float | None = None
  for index in range(1, len(bars)):
    previous = bars[index - 1]
    current = bars[index]
    if rising:
      qualifies = current.volume > previous.volume
    else:
      qualifies = current.volume < previous.volume
    if not qualifies:
      if level is not None:
        out[index] = level
      continue
    # A non-positive price on either side of the ratio is not a usable
    # multiplier. A zero numerator would drive the level to zero and
    # keep it there, which reads as a real index value forever, so the
    # bar is reported as undefined and the level is frozen instead.
    if previous.close <= 0.0 or current.close <= 0.0:
      continue
    ratio = current.close / previous.close
    level = VOLUME_INDEX_BASE * ratio if level is None else level * ratio
    out[index] = level
  return out


def positive_volume_index(bars: list[Bar]) -> list[float | None]:
  '''Return Fosback's Positive Volume Index.

  The level is multiplied by the close ratio on a bar whose volume rose
  against the prior bar, and carried forward unchanged on every other
  bar. It is a price index of the bars that attracted more interest, not
  a volume total.

  Fields: ``volume`` to qualify a bar, ``close`` on both the current and
  the prior bar for the ratio.

  **The boundary.** The recursion starts at the first bar that qualifies,
  and every earlier index is ``None``, including index 0 which has no
  prior bar. The first defined value is
  ``100 * close[i] / close[i - 1]``. A zero seed would be annihilated by
  the very first multiplication, so :data:`VOLUME_INDEX_BASE` is 100. If
  volume never rises across the whole series, no bar ever qualifies and
  every value is ``None``.

  A qualifying bar whose price ratio is undefined -- a non-positive close
  on either side -- is reported as ``None`` and freezes the level, for
  the same reason: a zero numerator would pin the index at zero.

  Args:
    bars: Price bars in ascending time order.

  Returns:
    List the same length as ``bars``.
  '''
  return _volume_index(bars, rising=True)


def negative_volume_index(bars: list[Bar]) -> list[float | None]:
  '''Return Fosback's Negative Volume Index.

  The mirror of :func:`positive_volume_index`: the level is multiplied by
  the close ratio on a bar whose volume fell, and carried forward
  unchanged otherwise. It is a price index of the bars that attracted
  less interest.

  Fields: ``volume`` to qualify a bar, ``close`` on both the current and
  the prior bar for the ratio.

  **The boundary** is the first bar whose volume fell, at which point the
  level becomes ``100 * close[i] / close[i - 1]``; every earlier index is
  ``None``. If volume never falls, every value is ``None``. A qualifying
  bar with a non-positive close on either side is ``None`` and freezes
  the level.

  Args:
    bars: Price bars in ascending time order.

  Returns:
    List the same length as ``bars``.
  '''
  return _volume_index(bars, rising=False)


def chaikin_money_flow(bars: list[Bar], window: int = 20) -> list[float | None]:
  '''Return Chaikin's Money Flow, bounded in ``[-1, 1]``.

  The sum of money flow volume over the sum of volume, where a bar's
  money flow volume is ``volume * ((close - low) - (high - close)) /
  (high - low)``. The multiplier is where the close sits inside the
  bar's range: +1 at the high, -1 at the low.

  Fields: ``high``, ``low`` and ``close`` for the multiplier, ``volume``
  for the scale. Every field of the bar is read.

  **A bar with no range is dropped from both sums.** When ``high ==
  low`` the multiplier is 0/0: the close is simultaneously the high and
  the low, so the numerator is exactly zero and the denominator is
  exactly zero, and the ratio has no value at all. Zero is not
  substituted, because a zero multiplier would claim the bar showed no
  buying pressure when in fact it showed none at all -- and including its
  volume in the denominator would silently shrink the reading by an
  unknown amount. The bar is excluded from numerator *and* denominator,
  so the result is the money flow of the bars that traded a range,
  normalised by their own volume rather than by the window's total
  volume. That is a documented deviation from "sum over the window" and
  it is the only defensible one: the alternative is a number whose
  meaning depends on how much of the window was uninformative.

  If no bar in the window traded a range, or the summed volume of those
  that did is not positive, the window has no value and ``None`` is
  returned. ``Bar.volume`` is documented as possibly zero for illiquid
  bars, so this case is reachable rather than hypothetical.

  Args:
    bars: Price bars in ascending time order.
    window: Lookback in bars, at least 1.

  Returns:
    List the same length as ``bars``, ``None`` before index
    ``window - 1`` and ``None`` on any window with no usable volume.

  Raises:
    ValueError: If ``window`` is not positive.
  '''
  if window < 1:
    raise ValueError(f'window must be >= 1, got {window}')
  contributions: list[tuple[float, float] | None] = []
  for bar in bars:
    span = bar.high - bar.low
    if span <= 0.0:
      contributions.append(None)
      continue
    multiplier = (
      ((bar.close - bar.low) - (bar.high - bar.close)) / span)
    contributions.append((multiplier * bar.volume, bar.volume))
  out: list[float | None] = [None] * len(bars)
  for index in range(window - 1, len(bars)):
    start = index - window + 1
    usable = [pair for pair in contributions[start:index + 1]
              if pair is not None]
    if not usable:
      continue
    total_volume = sum(volume for _, volume in usable)
    if total_volume <= 0.0:
      continue
    out[index] = sum(flow for flow, _ in usable) / total_volume
  return out


def ease_of_movement(bars: list[Bar], window: int = 14) -> list[float | None]:
  '''Return Ease of Movement.

  Each bar's raw reading is ``distance_moved / box_ratio``, where
  ``distance_moved`` is the change in the bar midpoint ``(high + low) /
  2`` against the prior bar, and ``box_ratio`` is
  ``(volume / window) / (high - low)``. The published pair of definitions
  is then averaged over ``window`` bars. Collapsing the two divisions
  gives the form computed here::

      (midpoint - prior_midpoint) * (high - low) * window / volume

  The sign says which way price moved and the magnitude says how little
  effort it took.

  Fields: ``high`` and ``low`` for the midpoint, the range and the
  volume. ``close`` is *not* used -- this indicator is built on the bar
  midpoint, not on closes.

  **Two guarded denominators, both reachable.** A bar with
  ``high == low`` has no box to divide by, and a bar with no volume has
  nothing to divide by; either makes the raw reading undefined and the
  raw value is ``None``. A raw ``None`` then makes every window
  containing it undefined, because averaging over fewer bars than
  ``window`` would report a different indicator under the same name.

  Index 0 is ``None`` because there is no prior bar to measure a move
  against. The first defined value is therefore at index ``window``, one
  bar later than ``window - 1``.

  Args:
    bars: Price bars in ascending time order.
    window: Lookback in bars for the smoothing average, at least 1.

  Returns:
    List the same length as ``bars``.

  Raises:
    ValueError: If ``window`` is not positive.
  '''
  if window < 1:
    raise ValueError(f'window must be >= 1, got {window}')
  raw: list[float | None] = [None] * len(bars)
  for index in range(1, len(bars)):
    current = bars[index]
    previous = bars[index - 1]
    span = current.high - current.low
    if span <= 0.0 or current.volume <= 0.0:
      continue
    midpoint = (current.high + current.low) / 2.0
    prior_midpoint = (previous.high + previous.low) / 2.0
    raw[index] = (
      (midpoint - prior_midpoint) * span * window / current.volume)
  out: list[float | None] = [None] * len(bars)
  for index in range(window, len(bars)):
    recent = raw[index - window + 1:index + 1]
    if any(value is None for value in recent):
      continue
    out[index] = fmean(recent)
  return out


def force_index(bars: list[Bar], window: int = 13) -> list[float | None]:
  '''Return Elder's Force Index.

  The raw force on each bar is ``(close - prior_close) * volume``, and
  the published index is that series smoothed with an exponential moving
  average of ``window``.

  Fields: ``close`` on both the current and the prior bar, ``volume``.

  **There is no division here, and that is a finding rather than an
  oversight.** The published Force Index is a product, not a ratio, so it
  has no zero-denominator path to guard. The frequently quoted
  *normalised* Force Index divides the raw force by volume (or by ATR),
  which is undefined on exactly the zero-volume bars
  :mod:`stock_rl.bars` documents as legitimate; it is deliberately not
  implemented, because adding a second definition under a parameter
  would leave a caller free to switch into the undefined case without
  being told. Zero volume therefore yields 0.0, which is the
  arithmetically correct answer: no volume, no force.

  The EMA seeds on its first available observation, so the only undefined
  index is 0, where there is no prior close.

  Args:
    bars: Price bars in ascending time order.
    window: EMA span in bars, at least 1.

  Returns:
    List the same length as ``bars``, ``None`` at index 0 only.

  Raises:
    ValueError: If ``window`` is not positive.
  '''
  if window < 1:
    raise ValueError(f'window must be >= 1, got {window}')
  if not bars:
    # Without this the leading None below would be returned for an empty
    # input, which is one value for zero bars: the exact off-by-one this
    # module is not allowed to ship.
    return []
  raw: list[float] = []
  for index in range(1, len(bars)):
    change = bars[index].close - bars[index - 1].close
    raw.append(change * bars[index].volume)
  smoothed = ema(raw, window)
  return [None, *smoothed]


def elder_ray(bars: list[Bar], window: int = 13) -> Ray:
  '''Return Elder's bull power and bear power.

  ``bull = high - EMA(close)`` and ``bear = low - EMA(close)`` on the
  same EMA. Bull power is how far the bar pushed above the mean, bear
  power how far it pushed below, so the pair shows pressure without
  claiming direction.

  Fields: ``high`` and ``low`` for the power, ``close`` for the EMA.
  ``volume`` is unused.

  **There is no division here.** The published Elder-Ray index subtracts
  the two series and neither series divides, so this function has no
  zero-denominator path. The related "Elder-Ray normalised by ATR" is a
  different indicator and is not implemented here.

  Neither output has a warm-up region: the EMA is seeded on the first
  close, so index 0 is defined and equal to ``high - close`` for the bull
  series. That is a deliberate, exact boundary rather than a missing
  warm-up, and it differs from every other function in this module.

  Args:
    bars: Price bars in ascending time order.
    window: EMA span in bars, at least 1.

  Returns:
    ``(bull, bear)``, two plain float lists the same length as ``bars``
    and never containing ``None``.

  Raises:
    ValueError: If ``window`` is not positive.
  '''
  if window < 1:
    raise ValueError(f'window must be >= 1, got {window}')
  middle = ema([bar.close for bar in bars], window)
  bull = [bar.high - centre for bar, centre in zip(bars, middle, strict=True)]
  bear = [bar.low - centre for bar, centre in zip(bars, middle, strict=True)]
  return bull, bear


def volume_oscillator(
  bars: list[Bar],
  short_window: int = 5,
  long_window: int = 10,
) -> list[float | None]:
  '''Return the Volume Oscillator: the two volume means in ratio form.

  ``100 * (SMA_short(volume) - SMA_long(volume)) / SMA_long(volume)``, so
  the reading is a percentage and is comparable across symbols of very
  different liquidity. Zero means the two averages agree.

  Fields: ``volume`` only. Every other field is ignored.

  **The long average is the denominator and it is reachable at zero.**
  :mod:`stock_rl.bars` documents that a bar may carry no volume because
  an illiquid session printed none, so a window of such bars gives a
  long average of exactly zero and the ratio is undefined. ``None`` is
  returned. Substituting zero would report "no change in volume
  activity" for a window with no volume activity whatsoever, which is the
  single most misleading number this function could return.

  A zero *short* average against a positive long one is fine and returns
  ``-100.0``, the arithmetically correct reading.

  Args:
    bars: Price bars in ascending time order.
    short_window: Fast average window in bars, at least 1 and no greater
      than ``long_window``.
    long_window: Slow average window in bars, at least 1.

  Returns:
    List the same length as ``bars``, ``None`` before index
    ``long_window - 1`` and ``None`` wherever the long average is not
    positive.

  Raises:
    ValueError: If a window is not positive or ``short_window`` exceeds
      ``long_window``.
  '''
  if short_window < 1:
    raise ValueError(f'short_window must be >= 1, got {short_window}')
  if long_window < 1:
    raise ValueError(f'long_window must be >= 1, got {long_window}')
  if short_window > long_window:
    raise ValueError(
      f'short_window must be <= long_window, got {short_window} > '
      f'{long_window}')
  volumes = [bar.volume for bar in bars]
  out: list[float | None] = [None] * len(bars)
  for index in range(long_window - 1, len(bars)):
    start = index - long_window + 1
    base = fmean(volumes[start:index + 1])
    if base <= 0.0:
      continue
    fast = fmean(volumes[index - short_window + 1:index + 1])
    out[index] = 100.0 * (fast - base) / base
  return out
