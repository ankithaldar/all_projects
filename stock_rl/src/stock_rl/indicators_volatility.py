#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Volatility and channel indicators over a bar series.

**Read this before wiring any of these into a strategy.**

Implementing a published formula is not evidence that the formula has
predictive value. Every function here is the arithmetic its author
published and nothing more; correctness of the code says nothing about
whether the resulting series forecasts anything. The project's own
research record (``docs/research/methodology/technical-indicators-nse.md``)
is blunt about the evidence base: for Bollinger Bands the best "Indian"
study is eight months of hourly data on fourteen stocks, which that
review calls "not a production basis"; Rink (2023) tested 6,406 rules
with a stepwise Superior Predictive Ability test and concluded technical
trading signals should be treated as noise; and FINSABER (KDD 2026)
measured standalone signals reading *inverted* out of sample. This
module therefore exists so a caller can choose to test a formula, not
because anyone has evidence that it works on NSE-listed equities.

**Every threshold here is unfitted.** Two numbers recur and both are
definitional conventions rather than trading rules:

* Bollinger's *two* standard deviations is the convention of Bollinger's
  own published definition. It is not a parameter fitted to any dataset
  and not a claim about where the bands should be touched.
* Keltner's ATR multiple is a caller-supplied argument with a
  conventional default. Raschke popularised the channel; she did not fit
  the width, and neither does this module.

There is deliberately no ``overbought``, ``oversold`` or ``squeeze``
argument anywhere below, and no function returns a regime label. Each
returns the raw series. A caller who wants a threshold supplies it and
then owns the evidence for it.

**Shared contract**, matching :mod:`stock_rl.indicators`:

* every function returns a container whose length equals its input;
* ``None`` marks the warm-up region, never ``0.0`` and never a partial
  value, because an undefined indicator that reads as a number can size
  a position;
* the value at index ``i`` uses bars up to ``i`` only;
* a non-positive window raises ``ValueError`` naming the parameter;
* every division is guarded by an explicit test of its denominator, and
  the guarded branch returns ``None`` rather than a plausible wrong
  number.

Price-only functions take ``list[float]`` closes and use the ``close``
field alone. Channel functions take ``list[Bar]`` and the docstring of
each names the fields it reads.

# ponytail: :func:`bollinger_bands`, :func:`atr_bands`,
# :func:`choppiness_index`, :func:`vertical_horizontal_filter` and
# :func:`coefficient_of_variation` all slice the trailing window out of
# the series and recompute their statistic from scratch at each index, so
# each is O(n * window) in time. :func:`bollinger_bandwidth` and
# :func:`bollinger_percent_b` inherit the same cost because they call
# :func:`bollinger_bands`. Ceiling: on the panel sizes this project runs,
# a 20-bar window over a few thousand daily bars is irrelevant next to
# the arithmetic around it; it would matter for a minute-bar sweep
# parameter search. Upgrade path: rolling sums for the additive parts and
# a monotonic deque for the rolling extremes, with a two-pass rolling
# mean/variance in place of the naive sum of squares so the variance does
# not lose conditioning. Nothing in the returned series would change.
'''

from __future__ import annotations

import math
from statistics import fmean, stdev

from stock_rl.bars import Bar
from stock_rl.indicators import atr, ema

__all__ = [
  'atr_bands',
  'bollinger_bandwidth',
  'bollinger_bands',
  'bollinger_percent_b',
  'choppiness_index',
  'coefficient_of_variation',
  'keltner_channels',
  'vertical_horizontal_filter',
]

#: Three aligned series, the shape every channel function returns.
Bands = tuple[list[float | None], list[float | None], list[float | None]]

#: Two aligned series: an upper and a lower level.
Envelope = tuple[list[float | None], list[float | None]]

#: Bollinger's published band width, in standard deviations. A definitional
#: convention of the original formula and nothing else -- see the module
#: docstring. Exposed as a name so a caller can see it is a convention and
#: not a fitted default.
BOLLINGER_DEVIATIONS = 2.0

#: Raschke's conventional Keltner multiple, in average true ranges.
#: A convention of the popularised channel, not a fitted width.
KELTNER_MULTIPLE = 2.0


def bollinger_bands(
  closes: list[float],
  window: int = 20,
  deviations: float = BOLLINGER_DEVIATIONS,
) -> Bands:
  '''Return Bollinger's middle, upper and lower bands.

  ``middle`` is the simple moving average of the trailing ``window``
  closes. The band half-width is ``deviations`` times the *sample*
  standard deviation (divisor ``n - 1``) of the same window, matching the
  convention already used by
  :func:`stock_rl.indicators.realized_volatility`, so the two modules
  cannot disagree about what a standard deviation is.

  Fields: ``close`` only. No threshold, squeeze or signal is derived.

  Args:
    closes: Closing prices in ascending time order.
    window: Lookback in bars. Must be at least 2, because the sample
      standard deviation of a single observation has no divisor.
    deviations: Band half-width in standard deviations. Must be positive.

  Returns:
    ``(middle, upper, lower)``, three lists each the length of ``closes``
    with ``None`` before index ``window - 1``.

  Raises:
    ValueError: If ``window`` is below 2 or ``deviations`` is not
      positive. A non-positive width collapses the bands and makes
      :func:`bollinger_percent_b` undefined everywhere, which is a
      caller mistake rather than a data condition.
  '''
  if window < 2:
    raise ValueError(f'window must be >= 2, got {window}')
  if deviations <= 0.0:
    raise ValueError(f'deviations must be > 0, got {deviations}')
  middle: list[float | None] = []
  upper: list[float | None] = []
  lower: list[float | None] = []
  for index in range(len(closes)):
    if index < window - 1:
      middle.append(None)
      upper.append(None)
      lower.append(None)
      continue
    sample = closes[index - window + 1:index + 1]
    centre = fmean(sample)
    half_width = deviations * stdev(sample)
    middle.append(centre)
    upper.append(centre + half_width)
    lower.append(centre - half_width)
  return middle, upper, lower


def bollinger_bandwidth(
  closes: list[float],
  window: int = 20,
  deviations: float = BOLLINGER_DEVIATIONS,
) -> list[float | None]:
  '''Return Bollinger's bandwidth: the band spread over the middle band.

  ``(upper - lower) / middle``. The point of the ratio is scale
  invariance -- it is comparable across a Rs 50 stock and a Rs 5,000 one
  -- which is exactly why the denominator has to be checked: a zero
  middle band makes the ratio undefined, and the published workaround of
  substituting zero for it would report "no volatility" for a series
  that has none at all to measure.

  Fields: ``close`` only, via :func:`bollinger_bands`.

  Args:
    closes: Closing prices in ascending time order.
    window: Lookback in bars, at least 2.
    deviations: Band half-width in standard deviations, positive.

  Returns:
    List the length of ``closes``, ``None`` before index ``window - 1``
    and ``None`` wherever the middle band is exactly zero.

  Raises:
    ValueError: If ``window`` is below 2 or ``deviations`` is not
      positive.
  '''
  middle, upper, lower = bollinger_bands(closes, window, deviations)
  out: list[float | None] = []
  for centre, top, bottom in zip(middle, upper, lower, strict=True):
    # The three bands share one warm-up and are None together, so the
    # explicit checks are belt and braces. They stay separate from the
    # zero-mean branch below because that one has a different cause: a
    # defined band sitting on a zero price level.
    if centre is None or top is None or bottom is None:
      out.append(None)
    elif centre == 0.0:
      out.append(None)
    else:
      out.append((top - bottom) / centre)
  return out


def bollinger_percent_b(
  closes: list[float],
  window: int = 20,
  deviations: float = BOLLINGER_DEVIATIONS,
) -> list[float | None]:
  '''Return Bollinger %b: where the close sits inside the band.

  ``(close - lower) / (upper - lower)``, so 0.0 is at the lower band,
  1.0 at the upper and 0.5 at the middle. The denominator is twice the
  band half-width, so it is zero on a perfectly flat window. The
  mathematically correct reading of 0/0 is "undefined", and both 0.0 and
  0.5 would be plausible-looking lies: the first claims the price is at
  the floor of a band that does not exist, the second that it is
  centred. ``None`` is returned instead.

  Fields: ``close`` only, via :func:`bollinger_bands`.

  Args:
    closes: Closing prices in ascending time order.
    window: Lookback in bars, at least 2.
    deviations: Band half-width in standard deviations, positive.

  Returns:
    List the length of ``closes``, ``None`` before index ``window - 1``
    and ``None`` wherever the window has no dispersion.

  Raises:
    ValueError: If ``window`` is below 2 or ``deviations`` is not
      positive.
  '''
  _, upper, lower = bollinger_bands(closes, window, deviations)
  out: list[float | None] = []
  for index, (top, bottom) in enumerate(zip(upper, lower, strict=True)):
    if top is None or bottom is None:
      out.append(None)
      continue
    spread = top - bottom
    if spread == 0.0:
      out.append(None)
    else:
      out.append((closes[index] - bottom) / spread)
  return out


def keltner_channels(
  bars: list[Bar],
  ema_window: int = 20,
  atr_window: int = 10,
  multiple: float = KELTNER_MULTIPLE,
) -> Bands:
  '''Return Keltner channels: an EMA with an ATR envelope around it.

  ``middle`` is an exponential moving average of the closes;
  ``upper`` and ``lower`` are that average plus and minus ``multiple``
  times Wilder's Average True Range from
  :func:`stock_rl.indicators.atr`. This is the form Linda Bradford
  Raschke popularised; the ATR multiple is a convention, not a fitted
  parameter, and this module makes no claim about which value is right.

  Fields: ``close`` for the middle band; ``high``, ``low`` and the
  previous ``close`` for the ATR.

  The warm-up is governed entirely by the ATR, not the EMA. The EMA
  seeds on its first observation, while ATR needs ``atr_window`` true
  ranges and so its first defined value sits at index ``atr_window``.
  Getting that boundary wrong shifts the whole channel by a bar.

  Args:
    bars: Price bars in ascending time order.
    ema_window: Span of the centre line in bars, at least 1.
    atr_window: Smoothing window of the ATR, at least 1.
    multiple: Envelope width in average true ranges, positive.

  Returns:
    ``(middle, upper, lower)``, each the length of ``bars``. The middle
    band has no ``None`` values; the two envelopes are ``None`` before
    index ``atr_window``.

  Raises:
    ValueError: If a window is not positive or ``multiple`` is not
      positive.
  '''
  if ema_window < 1:
    raise ValueError(f'ema_window must be >= 1, got {ema_window}')
  if atr_window < 1:
    raise ValueError(f'atr_window must be >= 1, got {atr_window}')
  if multiple <= 0.0:
    raise ValueError(f'multiple must be > 0, got {multiple}')
  middle = ema([bar.close for bar in bars], ema_window)
  ranges = atr(bars, atr_window)
  upper: list[float | None] = []
  lower: list[float | None] = []
  for index, spread in enumerate(ranges):
    if spread is None:
      upper.append(None)
      lower.append(None)
      continue
    # ``ema`` seeds on its first observation, so ``middle`` has no None
    # and only the ATR can make this index undefined.
    centre = middle[index]
    upper.append(centre + multiple * spread)
    lower.append(centre - multiple * spread)
  return middle, upper, lower


def atr_bands(
  bars: list[Bar],
  window: int = 14,
  multiple: float = 1.0,
) -> Envelope:
  '''Return an ATR envelope around the running extremes.

  ``upper`` is the highest high of the trailing ``window`` bars plus
  ``multiple`` ATR; ``lower`` is the lowest low minus the same. The
  published description of ATR bands is one sentence -- that they are
  used to signal exits like ATR trailing stops do, without the SAR
  machinery -- which does not pin a formula, so the formula above is
  stated here explicitly as this module's reading of it: the extreme is
  taken over a *rolling* window, not a running maximum since entry,
  because the latter needs a position and this is a pure series
  function. Chandelier Exit is this function at LeBeau's published
  parameters, ``window=22`` and ``multiple=3.0``, so it is not
  implemented again.

  Fields: ``high`` and ``low`` for the extremes; ``high``, ``low`` and
  the previous ``close`` for the ATR.

  Args:
    bars: Price bars in ascending time order.
    window: Lookback in bars for both the extreme and the ATR, at least
      1.
    multiple: Envelope width in average true ranges. Must not be
      negative; zero collapses the bands onto the extremes, which is a
      legitimate degenerate envelope rather than an error.

  Returns:
    ``(upper, lower)``, each the length of ``bars`` and ``None`` before
    index ``window``.

  Raises:
    ValueError: If ``window`` is not positive or ``multiple`` is
      negative.
  '''
  if window < 1:
    raise ValueError(f'window must be >= 1, got {window}')
  if multiple < 0.0:
    raise ValueError(f'multiple must be >= 0, got {multiple}')
  ranges = atr(bars, window)
  upper: list[float | None] = []
  lower: list[float | None] = []
  for index, spread in enumerate(ranges):
    if spread is None:
      upper.append(None)
      lower.append(None)
      continue
    recent = bars[index - window + 1:index + 1]
    upper.append(max(bar.high for bar in recent) + multiple * spread)
    lower.append(min(bar.low for bar in recent) - multiple * spread)
  return upper, lower


def choppiness_index(bars: list[Bar], window: int = 14) -> list[float | None]:
  '''Return Bill Dreiss' Choppiness Index.

  ``100 * log10( sum(H - L) / (max(H) - min(L)) ) / log10(window)``, where
  both the sum and the extremes run over the trailing ``window`` bars. The
  numerator is the total range covered and the denominator is the single
  widest span, so the ratio is large when the window kept overtrading its
  own range and near one when it covered its range once.

  **No clamping, and the published scale is not a guarantee.** The index
  is usually described as living in ``[0, 100]``, and it does on
  continuous price series. It is not clamped here, because a clamp would
  invent a reading: with a window of disjoint price levels the summed
  range can be smaller than the span, the log argument drops below one
  and the value goes negative -- two bars of range 1 at 9-10 and 99-100
  give ``100 * log10(2 / 91) / log10(2) = -550.779``. That is what the
  formula says, and it is also a true description of those two bars.
  Clamping it to zero would report "perfectly choppy" for a window that
  was not.

  **The zero denominator, and why one guard covers both.** The divisor
  is ``max(H) - min(L)``, and the numerator is the summed range. The
  numerator is zero exactly when no bar in the window traded a range, and
  ``log10(0)`` is not a number. Neither ``0.0`` nor ``100.0`` is an
  honest substitute: ``0.0`` claims the window was maximally choppy and
  ``100.0`` claims a perfectly clean trend, both on a window where no
  price moved at all. ``None`` is returned so the caller has to decide
  what an untraded window means.

  Only the numerator is guarded, and that is sufficient by proof rather
  than by hope. If the summed range is positive then some bar has
  ``H > L``, so ``max(H) >= H > L >= min(L)`` and therefore
  ``max(H) - min(L) > 0``. A zero or negative span is thus
  unreachable given a positive numerator, for any input whatsoever --
  including the malformed bars :func:`stock_rl.bars.load_csv` will
  happily load, where ``high < low``. A second guard would be
  unreachable code that could never be tested, which is worse than a
  documented proof.

  Fields: ``high`` and ``low`` only. ``close`` and ``volume`` are unused.

  Args:
    bars: Price bars in ascending time order.
    window: Lookback in bars, at least 2. A window of 1 makes
      ``log10(window)`` zero, so the divisor would vanish.

  Returns:
    List the length of ``bars``, ``None`` before index ``window - 1``
    and ``None`` on any window with no aggregate range.

  Raises:
    ValueError: If ``window`` is below 2.
  '''
  if window < 2:
    raise ValueError(f'window must be >= 2, got {window}')
  out: list[float | None] = [None] * len(bars)
  for index in range(window - 1, len(bars)):
    recent = bars[index - window + 1:index + 1]
    total_range = sum(bar.high - bar.low for bar in recent)
    if total_range <= 0.0:
      continue
    # ``span > 0`` is proven by the numerator guard above: a positive
    # total range means some bar has high > low, hence max(high) > min(low).
    span = max(bar.high for bar in recent) - min(bar.low for bar in recent)
    out[index] = (
      100.0 * math.log10(total_range / span) / math.log10(window))
  return out


def vertical_horizontal_filter(
  values: list[float],
  window: int = 14,
) -> list[float | None]:
  '''Return Adam White's Vertical Horizontal Filter, in ``[0, 100]``.

  ``100 * (highest close - lowest close) / (sum of absolute close to
  close changes)`` over the trailing ``window`` bars. A high reading is
  a directional market that covered its range once; a low one is a
  market that went over the same ground repeatedly.

  **The denominator needs one more bar than the numerator.** Summing
  ``window`` absolute changes requires ``window + 1`` closes, so the
  first defined value is at index ``window``, not ``window - 1``. A flat
  window makes the denominator zero: White's formula is then 0/0 and
  the honest answer is ``None``, because reporting ``0.0`` would claim a
  perfectly sideways market on a series where nothing moved.

  Fields: ``close`` only.

  Args:
    values: Closing prices in ascending time order.
    window: Lookback in bars, at least 1.

  Returns:
    List the length of ``values``, ``None`` before index ``window`` and
    ``None`` wherever the window had no net travel.

  Raises:
    ValueError: If ``window`` is not positive.
  '''
  if window < 1:
    raise ValueError(f'window must be >= 1, got {window}')
  out: list[float | None] = [None] * len(values)
  for index in range(window, len(values)):
    recent = values[index - window + 1:index + 1]
    reach = max(recent) - min(recent)
    travel = sum(
      abs(values[position] - values[position - 1])
      for position in range(index - window + 1, index + 1)
    )
    if travel <= 0.0:
      continue
    out[index] = 100.0 * reach / travel
  return out


def coefficient_of_variation(values: list[float],
                             window: int = 20) -> list[float | None]:
  '''Return the coefficient of variation: sample stdev over mean.

  ``stdev / mean`` on the trailing ``window`` observations, using the
  same ``n - 1`` divisor as :mod:`stock_rl.indicators`. It is the
  simplest scale-free dispersion measure there is, which is why the
  zero-mean case matters: the ratio is undefined at a zero mean, and
  ``None`` is returned rather than a number, because any finite
  substitute would invent a dispersion for a series whose scale is not
  merely unknown but undefined.

  A negative mean yields a negative ratio, which is arithmetically
  correct and economically meaningless. The value is returned as
  computed; nothing here treats it as a volatility magnitude.

  Fields: ``close`` only.

  Args:
    values: Closing prices in ascending time order.
    window: Lookback in bars, at least 2, because the sample standard
      deviation of one observation has no divisor.

  Returns:
    List the length of ``values``, ``None`` before index ``window - 1``
    and ``None`` wherever the window mean is exactly zero.

  Raises:
    ValueError: If ``window`` is below 2.
  '''
  if window < 2:
    raise ValueError(f'window must be >= 2, got {window}')
  out: list[float | None] = [None] * len(values)
  for index in range(window - 1, len(values)):
    recent = values[index - window + 1:index + 1]
    centre = fmean(recent)
    if centre == 0.0:
      continue
    out[index] = stdev(recent) / centre
  return out
