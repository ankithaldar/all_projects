#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Trend-following technical indicators over a bar series.

Pure functions, no state, no I/O, standard library only, standard
library imports only. Every function returns a list the same length as
its input, with ``None`` in the warm-up region where the indicator is
not yet defined, and every value at index ``i`` is computed from bars
up to and including ``i`` only.

That second property is worth stating loudly, because it is the one
this repo has been bitten by. A channel computed over a centred window,
or a lagging span indexed into the future, yields a series that looks
plausible on a chart and silently leaks future information into
anything sized from it. Nothing here reads index ``i + 1``.

READ THIS BEFORE USING ANY OF IT
================================
These are published formulas and nothing more. **Implementing a
published formula is not evidence that the formula has predictive
value.** ``docs/research/methodology/technical-indicators-nse.md``
records that the indicator thresholds behind this project's ranking
have no published Indian evidence behind them, and it lists ADX and
MACD in a table headed "explicitly drop -- zero or negative Indian
evidence". FINSABER (KDD 2026) found standalone signals measuring
*inverted*. So every constant in this module is a definitional
convention of the published indicator -- MACD's 12/26/9, ADX's 14,
Ichimoku's 9/26/52/26, the Mass Index's 9/25/25 -- and **every one of
them here is unfitted on this data**. Nothing in this module was chosen
by looking at Indian returns, and nothing in it was validated out of
sample.

Consequently, and this is the whole design decision: **no function
here returns an ``overbought`` or ``oversold`` flag, names a crossover,
or compares anything against a magic level.** The raw series goes out
and the caller decides what it means. A threshold bolted into a
primitive is a hidden unfitted parameter whose provenance nobody can
check later, which is precisely the failure the research review
documents. Where a platform's default parameters are shown below they
are cited as part of the published definition of the indicator, not as
a trading rule.

Conventions shared with ``stock_rl.indicators``:

* Output length always equals input length, unconditionally.
* Warm-up is ``None``, never ``0.0`` and never a partial value.
* Inputs are ascending in time; the caller guarantees it.
* A function takes exactly the inputs its formula needs: ``highs`` and
  ``lows`` when only the range is needed, and full ``bars`` when the
  formula needs the previous close (ADX true range, Ichimoku Chikou).
'''

from __future__ import annotations

import math
from collections.abc import Sequence

from stock_rl.bars import Bar
from stock_rl.indicators import ema, sma

#: A warm-up-aligned series: one entry per bar, ``None`` where the
#: indicator is undefined.
Series = list[float | None]

#: KST components as ``(roc_window, smoothing_window, weight)``. The
#: windows are the published defaults for Martin Pring's Know Sure
#: Thing, cited as part of the indicator's definition.
_KST_COMPONENTS = ((10, 10, 1), (15, 10, 2), (20, 10, 3), (30, 15, 4))

#: Sum of the KST component weights. Dividing by it keeps the
#: oscillator on the same numeric scale as its largest component.
#: Some published formulations omit this division; see the ``kst``
#: docstring, which names the choice rather than hiding it.
_KST_WEIGHT_SUM = 10

#: Coppock's published smoothing window on the summed rate of change,
#: paired with the 14/11 rate windows above. Cited as part of the
#: indicator's definition.
_COPPOCK_SMOOTHING = 10

__all__ = [
  'adx',
  'aroon_oscillator',
  'coppock',
  'dema',
  'displaced_ma',
  'donchian_channels',
  'hma',
  'ichimoku',
  'kst',
  'linear_regression',
  'macd',
  'macd_histogram',
  'mass_index',
  'tema',
  'trix',
  'wma',
]


# ---------------------------------------------------------------------------
# Wilder smoothing and the windowed primitives every indicator here is built
# from.
# ---------------------------------------------------------------------------


def _wilder_rma(values: list[float], window: int) -> Series:
  '''Return Wilder's smoothed moving average of ``values``.

  This is the recursion ``prev = prev + (value - prev) / window``,
  seeded with the mean of the first ``window`` values. It is *not* a
  simple moving average, and the difference is not cosmetic: a simple
  mean of a step change is correct only at the step and then decays,
  while the RMA keeps ``1/window`` of every value forever, so its
  effective horizon lengthens as the sample grows. Swapping in a simple
  mean produces a plausible number that is wrong everywhere after the
  seed.

  ``stock_rl.indicators`` uses the same recursion for RSI and ATR,
  offset by one bar there because the input series starts at the
  second bar.

  Args:
    values: Dense input series, one value per bar.
    window: Smoothing window in observations.

  Returns:
    List aligned with ``values``, ``None`` until ``window`` values
    have accumulated.
  '''
  result: Series = [None] * len(values)
  if len(values) < window:
    return result
  previous = sum(values[:window]) / window
  result[window - 1] = previous
  for index in range(window, len(values)):
    previous = previous + (values[index] - previous) / window
    result[index] = previous
  return result


def _require_same_length(
  left_name: str,
  left: Sequence[float],
  right_name: str,
  right: Sequence[float],
) -> None:
  '''Raise unless two input series are the same length.

  A silent length mismatch is exactly the bug this repo has shipped
  before: a return list a few entries short of its input reads as a
  warm-up region and nobody notices until a position is sized from the
  wrong bar.

  Args:
    left_name: Name of the first series, for the error message.
    left: The first series.
    right_name: Name of the second series, for the error message.
    right: The second series.

  Raises:
    ValueError: If the two lengths differ.
  '''
  if len(left) != len(right):
    raise ValueError(
      f'{left_name} and {right_name} must be the same length, '
      f'got {len(left)} and {len(right)}')


def _mean_of_window(values: Sequence[float | None], window: int) -> Series:
  '''Average each trailing window, ``None`` if the window is incomplete.

  Args:
    values: Input series, possibly ``None`` in its warm-up region.
    window: Lookback in observations.

  Returns:
    List aligned with ``values``. An entry is ``None`` unless all
    ``window`` inputs behind it are numbers.
  '''
  out: Series = [None] * len(values)
  for index in range(window - 1, len(values)):
    chunk = values[index - window + 1:index + 1]
    usable = [value for value in chunk if value is not None]
    if len(usable) < window:
      continue
    out[index] = sum(usable) / window
  return out


def _rolling_sum(values: Sequence[float | None], window: int) -> Series:
  '''Sum each trailing window, ``None`` if the window is incomplete.

  Args:
    values: Input series, possibly ``None`` in its warm-up region.
    window: Lookback in observations.

  Returns:
    List aligned with ``values``. An entry is ``None`` unless all
    ``window`` inputs behind it are numbers.
  '''
  out: Series = [None] * len(values)
  for index in range(window - 1, len(values)):
    chunk = values[index - window + 1:index + 1]
    usable = [value for value in chunk if value is not None]
    if len(usable) < window:
      continue
    out[index] = sum(usable)
  return out


def _weighted_mean(values: Sequence[float | None], window: int) -> Series:
  '''Linearly weighted mean of each trailing window.

  Weight ``k + 1`` goes to the ``k``-th element of the window, so the
  newest bar carries weight ``window`` and the oldest carries weight
  ``1``. Weights are therefore normalised by ``window * (window + 1)
  / 2`` and the result is a weighted *average*, not a weighted sum.

  Args:
    values: Input series, possibly ``None`` in its warm-up region.
    window: Lookback in observations.

  Returns:
    List aligned with ``values``. An entry is ``None`` unless all
    ``window`` inputs behind it are numbers.
  '''
  out: Series = [None] * len(values)
  divisor = window * (window + 1) / 2.0
  for index in range(window - 1, len(values)):
    chunk = values[index - window + 1:index + 1]
    usable = [value for value in chunk if value is not None]
    if len(usable) < window:
      continue
    weighted = sum(
      (position + 1) * value for position, value in enumerate(usable))
    out[index] = weighted / divisor
  return out


def _rate_of_change(values: Sequence[float], window: int) -> Series:
  '''Return the percentage rate of change over ``window`` observations.

  Args:
    values: Price series in ascending time order.
    window: Lookback in observations.

  Returns:
    List aligned with ``values``, ``None`` for the first ``window``
    entries and wherever the base price is not positive.
  '''
  out: Series = [None] * len(values)
  for index in range(window, len(values)):
    base = values[index - window]
    if base <= 0.0:
      continue
    out[index] = 100.0 * (values[index] / base - 1.0)
  return out


# ---------------------------------------------------------------------------
# Moving average variants.
# ---------------------------------------------------------------------------


def wma(values: list[float], window: int) -> Series:
  '''Return the linearly weighted moving average of ``values``.

  Weights run ``1, 2, ... window`` from oldest to newest, so the newest
  observation dominates. This is the same weighting Coppock applies to
  his summed rate of change, and it is a definitional convention of the
  WMA rather than a fitted parameter.

  Args:
    values: Input series.
    window: Lookback in observations.

  Returns:
    List aligned with ``values``, ``None`` until ``window`` observations
    have accumulated.

  Raises:
    ValueError: If ``window`` is not positive.

  Example:
    ``wma([1.0, 2.0, 3.0], 3)`` is ``(1*1 + 2*2 + 3*3) / 6 = 14/6``.
  '''
  if window < 1:
    raise ValueError(f'window must be >= 1, got {window}')
  return _weighted_mean(values, window)


def hma(values: list[float], window: int) -> Series:
  '''Return Hull's moving average of ``values``.

  Hull's published construction is
  ``WMA(2 * WMA(n/2) - WMA(n), sqrt(n))``. The half window is
  ``window // 2`` and the root window is ``floor(sqrt(window))``, the
  integer form every implementation of Hull's note uses because the
  windows have to be whole numbers of bars.

  The arithmetic is worth reading, because the formula subtracts two
  averages: the difference amplifies curvature, and the outer weighted
  average is what removes most of the resulting noise.

  Args:
    values: Input series.
    window: Base lookback in observations, at least 2.

  Returns:
    List aligned with ``values``, ``None`` until index
    ``window + floor(sqrt(window)) - 2``, the first bar at which the
    outer WMA has a full root window of differences to average.

  Raises:
    ValueError: If ``window`` is less than 2, which leaves no half
      window.

  Example:
    ``hma([1.0, 2.0, 3.0, 4.0, 5.0], 4)`` uses a half window of 2 and a
    root window of 2. WMA2 is ``5/3, 8/3, 11/3, 14/3`` and WMA4 is
    ``3, 4``, so the raw difference is ``13/3, 16/3`` and the outer
    WMA2 gives ``(13/3 + 2*16/3) / 3 = 45/9 = 5`` at index 4.
  '''
  if window < 2:
    raise ValueError(f'window must be >= 2, got {window}')
  half = window // 2
  root = math.isqrt(window)
  half_line = wma(values, half)
  full_line = wma(values, window)
  raw: Series = [None] * len(values)
  for index in range(window - 1, len(values)):
    raw[index] = 2.0 * half_line[index] - full_line[index]
  return _weighted_mean(raw, root)


def dema(values: list[float], window: int) -> Series:
  '''Return the double exponentially smoothed moving average.

  The published definition is ``2 * EMA(n) - EMA(EMA(n))``. Both
  recursions run over the whole input, but the result is reported as
  ``None`` until index ``window - 1`` because an EMA seeded at its
  first observation is not yet a smoothed value. Without that floor the
  DEMA would print a number on bar one that says nothing.

  Args:
    values: Input series.
    window: Span in observations. The smoothing factor is
      ``2 / (window + 1)``.

  Returns:
    List aligned with ``values``, ``None`` for the first ``window - 1``
    entries.

  Raises:
    ValueError: If ``window`` is not positive.

  Example:
    On ``[1, 2, 3, 4]`` with ``window = 2`` the factor is ``2/3``.
    EMA1 is ``1, 5/3, 23/9, 95/27`` and EMA2 is ``1, 13/9, 59/27,
    83/27``, so the DEMA at index 2 is ``2*23/9 - 59/27 = 79/27``.
  '''
  if window < 1:
    raise ValueError(f'window must be >= 1, got {window}')
  first = ema(values, window)
  second = ema(first, window)
  out: Series = [None] * len(values)
  for index in range(window - 1, len(values)):
    out[index] = 2.0 * first[index] - second[index]
  return out


def tema(values: list[float], window: int) -> Series:
  '''Return the triple exponentially smoothed moving average.

  The published definition is
  ``3 * EMA(n) - 3 * EMA(EMA(n)) + EMA(EMA(EMA(n)))``. The warm-up
  floor is the same as for :func:`dema` and for the same reason.

  Args:
    values: Input series.
    window: Span in observations. The smoothing factor is
      ``2 / (window + 1)``.

  Returns:
    List aligned with ``values``, ``None`` for the first ``window - 1``
    entries.

  Raises:
    ValueError: If ``window`` is not positive.

  Example:
    On ``[1, 2, 3, 4]`` with ``window = 2`` the three recursions give
    ``1, 5/3, 23/9, 95/27``, then ``1, 13/9, 59/27, 83/27``, then
    ``1, 35/27, 17/9, 217/81``. The TEMA at index 2 is
    ``3*23/9 - 3*59/27 + 17/9 = 69/9 - 59/9 + 17/9 = 27/9 = 3``.
  '''
  if window < 1:
    raise ValueError(f'window must be >= 1, got {window}')
  first = ema(values, window)
  second = ema(first, window)
  third = ema(second, window)
  out: Series = [None] * len(values)
  for index in range(window - 1, len(values)):
    out[index] = (
      3.0 * first[index] - 3.0 * second[index] + third[index])
  return out


def displaced_ma(
  values: list[float],
  window: int,
  displacement: int,
) -> Series:
  '''Return a moving average displaced earlier by ``displacement`` bars.

  This is Percentage Bands' displaced moving average and Ichimoku's
  leading spans in isolation. The output at index ``i`` is the average
  of bars ``i - displacement - window + 1`` through
  ``i - displacement``, so it is built entirely from bars already in
  the past at ``i``. Displacing *forward* instead would make index
  ``i`` read bar ``i + displacement``, which is look-ahead, so this
  function only ever shifts backwards.

  Args:
    values: Input series.
    window: Lookback in observations for the average.
    displacement: Bars to shift the average earlier by.

  Returns:
    List aligned with ``values``, ``None`` until index
    ``displacement + window - 1``.

  Raises:
    ValueError: If ``window`` or ``displacement`` is not positive.

  Example:
    ``displaced_ma([1.0, 2.0, 3.0, 4.0, 5.0], 2, 1)`` has SMA2 of
    ``None, 1.5, 2.5, 3.5, 4.5``, shifted one bar left to give
    ``None, None, 1.5, 2.5, 3.5``.
  '''
  if window < 1:
    raise ValueError(f'window must be >= 1, got {window}')
  if displacement < 1:
    raise ValueError(f'displacement must be >= 1, got {displacement}')
  average = sma(values, window)
  out: Series = [None] * len(values)
  for index in range(displacement, len(values)):
    source = index - displacement
    out[index] = average[source] if source < len(average) else None
  return out


# ---------------------------------------------------------------------------
# Oscillators and moving average systems.
# ---------------------------------------------------------------------------


def macd(
  closes: list[float],
  fast_window: int = 12,
  slow_window: int = 26,
  signal_window: int = 9,
) -> tuple[Series, Series, Series]:
  '''Return the MACD line, its signal line and their difference.

  The published definition is ``EMA(fast) - EMA(slow)``, with the
  signal line an EMA of the MACD line and the histogram the MACD line
  minus the signal line. The 12/26/9 defaults are part of Gerald
  Aplon's published specification, cited here as a definitional
  convention and not as a trading rule; ``technical-indicators-nse.md``
  records that MACD has worked in only 2 of the 5 developed markets
  Hsu & Ng tested, so nothing about those numbers is a claim here.

  Warm-up is deliberately two-stage. An EMA seeded at its first
  observation is not a smoothed value, so the MACD line is ``None``
  before index ``slow_window - 1``, and the signal line is ``None``
  before ``slow_window + signal_window - 2``. Reporting the line from
  bar one would print a 0.0 that is arithmetically valid and
  meaningless, which is exactly the failure this module exists to
  avoid.

  Args:
    closes: Closing prices in ascending time order.
    fast_window: Fast EMA span, must be less than ``slow_window``.
    slow_window: Slow EMA span.
    signal_window: Span of the signal EMA.

  Returns:
    ``(line, signal, histogram)``, each aligned with ``closes``.

  Raises:
    ValueError: If any window is not positive, or if ``fast_window`` is
      not strictly less than ``slow_window``. The latter is definitional
      rather than a preference: with equal windows the MACD line is
      identically zero.

  Example:
    On ``[1, 2, 3, 4]`` with ``fast_window=2``, ``slow_window=3`` and
    ``signal_window=2`` the factor is ``2/3`` and ``1/2``. Fast EMA is
    ``1, 5/3, 23/9, 95/27`` and slow EMA is ``1, 3/2, 9/4, 25/8``, so
    the line is ``None, None, 23/9 - 9/4 = 11/36,
    95/27 - 25/8 = 85/216``. The signal line seeds on ``11/36`` and
    then gives ``2/3 * 85/216 + 1/3 * 11/36 = 59/162``, so the
    histogram is ``85/216 - 59/162 = 19/648``.
  '''
  if fast_window < 1:
    raise ValueError(f'fast_window must be >= 1, got {fast_window}')
  if slow_window < 1:
    raise ValueError(f'slow_window must be >= 1, got {slow_window}')
  if signal_window < 1:
    raise ValueError(f'signal_window must be >= 1, got {signal_window}')
  if fast_window >= slow_window:
    raise ValueError(
      f'fast_window must be < slow_window, got {fast_window} and '
      f'{slow_window}')
  total = len(closes)
  fast = ema(closes, fast_window)
  slow = ema(closes, slow_window)
  start = slow_window - 1
  line: Series = [None] * total
  for index in range(start, total):
    line[index] = fast[index] - slow[index]
  signal: Series = [None] * total
  tail = ema(line[start:], signal_window)
  for position in range(signal_window - 1, len(tail)):
    signal[start + position] = tail[position]
  histogram: Series = [None] * total
  for index in range(total):
    if line[index] is None or signal[index] is None:
      continue
    histogram[index] = line[index] - signal[index]
  return line, signal, histogram


def macd_histogram(
  closes: list[float],
  fast_window: int = 12,
  slow_window: int = 26,
  signal_window: int = 9,
) -> Series:
  '''Return only the MACD histogram.

  A thin wrapper over :func:`macd` for callers that want the difference
  of the two EMAs without unpacking three lists. The definition and the
  warm-up are identical; nothing is added.

  Args:
    closes: Closing prices in ascending time order.
    fast_window: Fast EMA span.
    slow_window: Slow EMA span.
    signal_window: Span of the signal EMA.

  Returns:
    List aligned with ``closes``, ``None`` before index
    ``slow_window + signal_window - 2``.

  Raises:
    ValueError: Under the same conditions as :func:`macd`.
  '''
  return macd(closes, fast_window, slow_window, signal_window)[2]


def adx(
  bars: list[Bar],
  window: int = 14,
) -> tuple[Series, Series, Series]:
  '''Return Wilder's ADX with its two directional indicators.

  The full directional movement system, as published by J. Welles
  Wilder:

  1. ``+DM`` is the upward move ``high[i] - high[i-1]`` when it exceeds
     the downward move and is positive, else zero. ``-DM`` is the
     mirror image.
  2. True range is the largest of the intrabar span and the two
     cross-bar distances to the previous close, so a gap registers as a
     wide range rather than a quiet one.
  3. All three are smoothed with **Wilder's RMA**, not a simple mean:
     ``prev = prev + (value - prev) / window``, seeded with the mean of
     the first ``window`` values.
  4. ``+DI = 100 * RMA(+DM) / RMA(TR)`` and ``-DI = 100 * RMA(-DM) /
     RMA(TR)``, then ``DX = 100 * |+DI - -DI| / (+DI + -DI)``.
  5. ADX is the Wilder RMA of DX.

  Step 3 is the one that goes wrong quietly. A simple mean of the last
  ``window`` directional moves is *not* an approximation of Wilder's
  value after the seed; see the module tests, which pin the recursion
  with exact fractions.

  ``technical-indicators-nse.md`` records that ADX has no Indian equity
  evidence at all, only foreign-exchange papers, and that Han, Yang &
  Zhou's support for a trend/range overlay is volatility-sorted rather
  than ADX-based. That is absence of evidence rather than evidence of
  absence, and it is a reason to validate nothing here rather than a
  reason to trust the number.

  Args:
    bars: Price bars in ascending time order. True range needs the
      previous close, hence whole bars rather than highs and lows.
    window: Wilder smoothing window.

  Returns:
    ``(adx, plus_di, minus_di)``, each aligned with ``bars``. ``+DI``
    and ``-DI`` start at index ``window``, because there is one fewer
    directional move than there are bars; ADX starts at index
    ``2 * window - 1``, because it is an RMA of DX over the same
    window.

  Raises:
    ValueError: If ``window`` is not positive.

  A bar set with no range at all gives ``0.0`` everywhere rather than
  dividing zero by zero: with neither directional movement nor a range
  there is no directional strength to report, and ``0.0`` is the
  degenerate reading, the same way a flat series gives RSI 50.0.

  Example:
    With ``window = 3`` and highs ``10, 11, 12, 12.5, 13, 12, 12.5,
    13.5``, lows ``9, 9.5, 10, 11, 11.5, 11, 11.5, 12`` and closes
    ``9.5, 10.5, 11.5, 12, 12.5, 11.5, 12, 13``: ``+DM`` is
    ``1, 1, 0.5, 0.5, 0, 0.5, 1``, ``-DM`` is ``0, 0, 0, 0, 0.5, 0, 0``
    and TR is ``1.5, 2, 1.5, 1.5, 1.5, 1, 1.5``. At index 3 the RMA
    seeds on ``(1 + 1 + 0.5) / 3 = 5/6`` and ``1.5 + 2 + 1.5) / 3 =
    5/3``, so ``+DI = 100 * (5/6) / (5/3) = 50`` and ``-DI = 0``.
  '''
  if window < 1:
    raise ValueError(f'window must be >= 1, got {window}')
  strength: Series = [None] * len(bars)
  plus_di, minus_di, dx = _directional_index(bars, window)
  smoothed = _wilder_rma(dx, window)
  for position in range(window - 1, len(smoothed)):
    strength[window + position] = smoothed[position]
  return strength, plus_di, minus_di


def _directional_movement(
  bars: list[Bar],
  window: int,
) -> tuple[Series, Series, Series]:
  '''Return Wilder-smoothed ``+DM``, ``-DM`` and true range.

  Args:
    bars: Price bars in ascending time order.
    window: Wilder smoothing window.

  Returns:
    Three lists indexed by bar, so index ``i - 1`` of each carries the
    smoothed value for bar ``i``. Entries before index ``window - 1``
    are ``None``.
  '''
  count = len(bars) - 1
  up_moves: list[float] = [0.0] * count
  down_moves: list[float] = [0.0] * count
  spans: list[float] = [0.0] * count
  for index in range(1, len(bars)):
    current = bars[index]
    previous_close = bars[index - 1].close
    up_move = current.high - bars[index - 1].high
    down_move = bars[index - 1].low - current.low
    if up_move > down_move and up_move > 0.0:
      up_moves[index - 1] = up_move
    if down_move > up_move and down_move > 0.0:
      down_moves[index - 1] = down_move
    spans[index - 1] = max(
      current.high - current.low,
      abs(current.high - previous_close),
      abs(current.low - previous_close),
    )
  return (
    _wilder_rma(up_moves, window),
    _wilder_rma(down_moves, window),
    _wilder_rma(spans, window),
  )


def _directional_reading(
  up_smoothed: float,
  down_smoothed: float,
  span_smoothed: float,
) -> tuple[float, float, float]:
  '''Return ``(+DI, -DI, DX)`` from smoothed directional movement.

  Args:
    up_smoothed: Wilder-smoothed ``+DM``.
    down_smoothed: Wilder-smoothed ``-DM``.
    span_smoothed: Wilder-smoothed true range.

  Returns:
    The three readings. A non-positive true range gives
    ``(0.0, 0.0, 0.0)``; a combined directional index of zero gives a
    DX of ``0.0``. Both are the degenerate readings rather than a
    division by zero.
  '''
  if span_smoothed <= 0.0:
    return 0.0, 0.0, 0.0
  plus = 100.0 * up_smoothed / span_smoothed
  minus = 100.0 * down_smoothed / span_smoothed
  combined = plus + minus
  if combined <= 0.0:
    return plus, minus, 0.0
  return plus, minus, 100.0 * abs(plus - minus) / combined


def _directional_index(
  bars: list[Bar],
  window: int,
) -> tuple[Series, Series, list[float]]:
  '''Return ``+DI``, ``-DI`` and the raw DX values.

  Args:
    bars: Price bars in ascending time order.
    window: Wilder smoothing window.

  Returns:
    ``plus_di`` and ``minus_di`` aligned with ``bars``, and ``dx`` as a
    dense list whose ``j``-th entry belongs to bar ``window + j``. The
    DX list is dense because its first entry is the seed of the ADX
    recursion, which is consumed at bar ``2 * window - 1``.
  '''
  up_line, down_line, span_line = _directional_movement(bars, window)
  plus_di: Series = [None] * len(bars)
  minus_di: Series = [None] * len(bars)
  dx: list[float] = []
  for index in range(window, len(bars)):
    # Index window - 1 is the Wilder seed, so these are never None.
    reading = _directional_reading(
      up_line[index - 1], down_line[index - 1], span_line[index - 1])
    plus_di[index] = reading[0]
    minus_di[index] = reading[1]
    dx.append(reading[2])
  return plus_di, minus_di, dx


def aroon_oscillator(
  highs: list[float],
  lows: list[float],
  window: int = 25,
) -> Series:
  '''Return Chande's Aroon oscillator in ``[-100, 100]``.

  Defined as bars since the highest high minus bars since the lowest
  low, each scaled to ``[0, 100]`` over a window of ``window + 1``
  bars including the current one:

  ``aroon_up = 100 * (window - ago_high) / window`` and the same for
  the low.

  On a tie the *more recent* extreme wins, because the quantity being
  reported is bars since the extreme and the latest occurrence of an
  equal extreme is the one a reader means. The scan therefore runs from
  the oldest bar of the window towards the current one, so the smallest
  ``ago`` among tied bars is the one left behind. Scanning the other
  way would keep the oldest occurrence and report a materially
  different value on any series with a repeated extreme.

  Args:
    highs: Highs in ascending time order.
    lows: Lows in ascending time order.
    window: Lookback in observations. The scan covers ``window + 1``
      bars.

  Returns:
    List aligned with ``highs``, ``None`` for the first ``window``
    entries. The oscillator's sign is the raw reading; this module
    attaches no level to it.

  Raises:
    ValueError: If ``window`` is not positive, or if ``highs`` and
      ``lows`` differ in length.

  Example:
    ``aroon_oscillator([12, 11, 13], [5, 9, 7], 2)`` looks back 3 bars.
    At index 2 the highest high is the current bar and the lowest low
    is 2 bars back, so the reading is ``100 - 0 = 100``.

    With highs ``10, 10, 10, 1, 2`` and window 3 the maximum of the
    window is 10, reached at bars 1 and 2. The most recent of those is
    bar 2, one bar back, so ``aroon_up = 100 * (3 - 1) / 3 = 200/3``
    rather than the ``0`` an oldest-wins scan would report.
  '''
  if window < 1:
    raise ValueError(f'window must be >= 1, got {window}')
  _require_same_length('highs', highs, 'lows', lows)
  out: Series = [None] * len(highs)
  for index in range(window, len(highs)):
    upper = highs[index]
    lower = lows[index]
    up_ago = 0
    down_ago = 0
    for back in range(window, 0, -1):
      if highs[index - back] >= upper:
        upper = highs[index - back]
        up_ago = back
      if lows[index - back] <= lower:
        lower = lows[index - back]
        down_ago = back
    up_value = 100.0 * (window - up_ago) / window
    down_value = 100.0 * (window - down_ago) / window
    out[index] = up_value - down_value
  return out


def trix(closes: list[float], window: int = 15) -> Series:
  '''Return TRIX, the one-bar rate of change of a triple-smoothed EMA.

  The published definition smooths the prices three times with the same
  span and reports the percentage change of the result over one bar.
  Each smoothing recursion runs over the whole input, but the series is
  reported as ``None`` before index ``window - 1``, so the first printed
  TRIX is at index ``window``: an EMA seeded at its first observation
  carries no information about a trend.

  Args:
    closes: Closing prices in ascending time order.
    window: Span in observations for each of the three smoothings.

  Returns:
    List aligned with ``closes``, ``None`` for the first ``window``
    entries.

  Raises:
    ValueError: If ``window`` is not positive.

  Example:
    On ``[1, 2, 3, 4]`` with ``window = 2`` the factor is ``2/3`` and
    the third smoothing is ``1, 35/27, 17/9, 217/81``. The TRIX at
    index 2 is ``100 * (17/9) / (35/27) - 100 = 100 * 51/35 - 100 =
    1600/35 = 320/7``.
  '''
  if window < 1:
    raise ValueError(f'window must be >= 1, got {window}')
  first = ema(closes, window)
  second = ema(first, window)
  third = ema(second, window)
  out: Series = [None] * len(closes)
  for index in range(window, len(closes)):
    previous = third[index - 1]
    if previous == 0.0:
      continue
    out[index] = 100.0 * (third[index] / previous - 1.0)
  return out


def kst(
  closes: list[float],
  signal_window: int = 9,
) -> tuple[Series, Series]:
  '''Return Pring's Know Sure Thing oscillator and its signal line.

  Four rate-of-change series are each smoothed by a simple moving
  average, then combined with weights ``1, 2, 3, 4``. The four
  ROC windows ``10, 15, 20, 30`` and smoothing windows ``10, 10, 10,
  15`` are the published defaults of the indicator, cited as a
  definitional convention; only the signal window is a parameter here,
  because varying the other eight would be varying the definition.

  One published convention is genuinely split and is therefore named
  rather than hidden: Chester Krier's original sums the weighted
  components, while most later statements of the indicator divide by
  the sum of the weights. This implementation divides by ten, so the
  oscillator sits on the scale of its largest component. To recover
  the undivided form, multiply by :data:`_KST_WEIGHT_SUM`.

  Args:
    closes: Closing prices in ascending time order.
    signal_window: Simple moving average window for the signal line.

  Returns:
    ``(kst, signal)``, each aligned with ``closes``. The slowest
    component needs ``30 + 15 = 45`` bars, so the oscillator starts at
    index 44 and the signal at index ``44 + signal_window - 1``.

  Raises:
    ValueError: If ``signal_window`` is not positive.

  Example:
    On a geometric series ``100 * 1.01 ** i`` every rate of change over
    any window is the same number, so each component equals its own
    rate of change: ``10.462213, 16.096896, 22.019004, 34.784892``.
    The oscillator is then
    ``(10.462213 + 2*16.096896 + 3*22.019004 + 4*34.784892) / 10 =
    247.852582 / 10 = 24.785258``.
  '''
  if signal_window < 1:
    raise ValueError(f'signal_window must be >= 1, got {signal_window}')
  components = [
    _mean_of_window(_rate_of_change(closes, roc_window), smoothing)
    for roc_window, smoothing, _ in _KST_COMPONENTS
  ]
  weights = [weight for _, _, weight in _KST_COMPONENTS]
  out: Series = [None] * len(closes)
  for index in range(len(closes)):
    if any(component[index] is None for component in components):
      continue
    total = 0.0
    for weight, component in zip(weights, components, strict=True):
      total += weight * component[index]
    out[index] = total / _KST_WEIGHT_SUM
  return out, _mean_of_window(out, signal_window)


def coppock(
  closes: list[float],
  long_window: int = 14,
  short_window: int = 11,
  smoothing: int = _COPPOCK_SMOOTHING,
) -> Series:
  '''Return Coppock's rate-of-change oscillator.

  Defined as a weighted moving average of the sum of the
  ``long_window`` and ``short_window`` rates of change. Coppock's
  published parameters are 14, 11 and a 10-bar weighted average, cited
  here as a definitional convention. Whatever the two rate windows are,
  the sum only becomes defined at index ``max(long_window,
  short_window)``, because a rate of change needs its own base bar, and
  the weighted average then needs ``smoothing`` bars of that sum.

  Args:
    closes: Closing prices in ascending time order.
    long_window: Longer rate-of-change window in observations.
    short_window: Shorter rate-of-change window in observations.
    smoothing: Weighted moving average window on the summed rates.

  Returns:
    List aligned with ``closes``, ``None`` until index
    ``max(long_window, short_window) + smoothing - 1``.

  Raises:
    ValueError: If any of the three windows is not positive.

  Example:
    On ``[10, 11, 12, 13]`` with ``long_window=2``, ``short_window=1``
    and ``smoothing=2`` the summed rates of change are
    ``100*(12/10 - 1) + 100*(12/11 - 1) = 320/11`` at index 2 and
    ``100*(13/11 - 1) + 100*(13/12 - 1) = 875/33`` at index 3. The
    2-bar WMA at index 3 is ``(1*320/11 + 2*875/33) / 3 =
    (960 + 1750) / 99 = 2710/99``.
  '''
  if long_window < 1:
    raise ValueError(f'long_window must be >= 1, got {long_window}')
  if short_window < 1:
    raise ValueError(f'short_window must be >= 1, got {short_window}')
  if smoothing < 1:
    raise ValueError(f'smoothing must be >= 1, got {smoothing}')
  long_rate = _rate_of_change(closes, long_window)
  short_rate = _rate_of_change(closes, short_window)
  combined: Series = [None] * len(closes)
  for index in range(len(closes)):
    if long_rate[index] is None or short_rate[index] is None:
      continue
    combined[index] = long_rate[index] + short_rate[index]
  return _weighted_mean(combined, smoothing)


def mass_index(
  highs: list[float],
  lows: list[float],
  fast_window: int = 9,
  slow_window: int = 25,
  sum_window: int = 25,
) -> Series:
  '''Return Donchian's Mass Index.

  Defined as the running sum of ``EMA(fast_window) / EMA(slow_window)``
  of the high-low span, over ``sum_window`` bars. The ``9 / 25 / 25``
  windows are the published defaults of the indicator, cited as a
  definitional convention. Mass Index is a range-expansion measure; no
  reversal claim is made or implied here.

  The warm-up is long and that is arithmetic, not conservatism: the
  slow EMA needs ``slow_window - 1`` bars before it is a ratio, and the
  running sum needs ``sum_window`` ratios after that, so the first
  printed value is at index ``slow_window + sum_window - 2``, which is
  index 48 for the defaults.

  Args:
    highs: Highs in ascending time order.
    lows: Lows in ascending time order.
    fast_window: Fast EMA span on the span series.
    slow_window: Slow EMA span, must exceed ``fast_window``.
    sum_window: Number of ratio bars in the running sum.

  Returns:
    List aligned with ``highs``, ``None`` during warm-up and wherever the
    slow EMA of the span is zero.

  Raises:
    ValueError: If any window is not positive, if ``highs`` and ``lows``
      differ in length, or if ``fast_window`` is not strictly less than
      ``slow_window``. The last is definitional: with equal spans the
      ratio is identically one and the index degenerates to
      ``sum_window``.

  Example:
    A constant span of ``2`` gives fast EMA = slow EMA = ``2``, a ratio
    of exactly ``1`` everywhere, and therefore an index of exactly
    ``sum_window`` once warm. That is the degenerate answer, not a bug.
  '''
  if fast_window < 1:
    raise ValueError(f'fast_window must be >= 1, got {fast_window}')
  if slow_window < 1:
    raise ValueError(f'slow_window must be >= 1, got {slow_window}')
  if sum_window < 1:
    raise ValueError(f'sum_window must be >= 1, got {sum_window}')
  if fast_window >= slow_window:
    raise ValueError(
      f'fast_window must be < slow_window, got {fast_window} and '
      f'{slow_window}')
  _require_same_length('highs', highs, 'lows', lows)
  spans = [high - low for high, low in zip(highs, lows, strict=True)]
  fast = ema(spans, fast_window)
  slow = ema(spans, slow_window)
  ratios: Series = [None] * len(spans)
  for index in range(slow_window - 1, len(spans)):
    if slow[index] == 0.0:
      continue
    ratios[index] = fast[index] / slow[index]
  return _rolling_sum(ratios, sum_window)


def ichimoku(
  bars: list[Bar],
  conversion_window: int = 9,
  base_window: int = 26,
  span_b_window: int = 52,
  displacement: int = 26,
) -> tuple[Series, Series, Series, Series, Series]:
  '''Return the five Ichimoku components.

  Tenkan is the midpoint of the ``conversion_window`` range, Kijun the
  midpoint of the ``base_window`` range, Senkou A the mean of those two
  midpoints and Senkou B the midpoint of the ``span_b_window`` range.
  The ``9 / 26 / 52`` windows and the 26-bar displacement are the
  published system parameters, cited as a definitional convention.

  On displacement, and it is a real decision rather than a detail: the
  two leading spans are drawn forward, so the value plotted at bar ``i``
  is computed from bar ``i - displacement``, which is in the past and
  safe. The lagging span, Chikou, is drawn *backwards*, so returning
  ``closes[i + displacement]`` at index ``i`` would be the textbook
  formulation and would also be look-ahead: index ``i`` would read a
  future bar, and anything sized from it would be sized from
  information the market did not have. Chikou is therefore returned at
  the bar it came from, which means it equals the close series. That is
  not a defect; a lagging line carries no information the price bar does
  not already carry, and a charting caller shifts it left when drawing.

  # ponytail: Chikou is returned undisplaced because this module refuses
  # to emit look-ahead. Ceiling: a caller cannot recover the drawn
  # position of the lagging span from the array alone. Upgrade path:
  # add a separate chart-space projection that maps bar ``i`` to bar
  # ``i - displacement`` as a rendering concern, keeping the value
  # series here free of forward reads.

  Args:
    bars: Price bars in ascending time order. Chikou needs the close, so
      whole bars rather than highs and lows.
    conversion_window: Tenkan window.
    base_window: Kijun window.
    span_b_window: Senkou B window.
    displacement: Bars of displacement for the leading spans.

  Returns:
    ``(tenkan, kijun, senkou_a, senkou_b, chikou)``, each aligned with
    ``bars``. Tenkan starts at index ``conversion_window - 1``, Kijun at
    ``base_window - 1``, Senkou A at ``displacement + base_window - 1``
    and Senkou B at ``displacement + span_b_window - 1``.

  Raises:
    ValueError: If any window is not positive.
  '''
  if conversion_window < 1:
    raise ValueError(f'conversion_window must be >= 1, got {conversion_window}')
  if base_window < 1:
    raise ValueError(f'base_window must be >= 1, got {base_window}')
  if span_b_window < 1:
    raise ValueError(f'span_b_window must be >= 1, got {span_b_window}')
  if displacement < 1:
    raise ValueError(f'displacement must be >= 1, got {displacement}')
  highs = [bar.high for bar in bars]
  lows = [bar.low for bar in bars]
  tenkan = _range_midpoint(highs, lows, conversion_window)
  kijun = _range_midpoint(highs, lows, base_window)
  span_b_source = _range_midpoint(highs, lows, span_b_window)
  total = len(bars)
  senkou_a: Series = [None] * total
  senkou_b: Series = [None] * total
  for index in range(displacement, total):
    source = index - displacement
    if kijun[source] is not None:
      senkou_a[index] = (tenkan[source] + kijun[source]) / 2.0
    senkou_b[index] = span_b_source[source]
  return tenkan, kijun, senkou_a, senkou_b, [bar.close for bar in bars]


def _range_midpoint(
  highs: list[float],
  lows: list[float],
  window: int,
) -> Series:
  '''Return ``(max high + min low) / 2`` over each trailing window.

  Args:
    highs: Highs in ascending time order.
    lows: Lows in ascending time order.
    window: Lookback in observations.

  Returns:
    List aligned with ``highs``, ``None`` until ``window`` observations
    have accumulated.
  '''
  out: Series = [None] * len(highs)
  for index in range(window - 1, len(highs)):
    start = index - window + 1
    out[index] = (
      max(highs[start:index + 1]) + min(lows[start:index + 1])) / 2.0
  return out


def donchian_channels(
  highs: list[float],
  lows: list[float],
  window: int = 20,
) -> tuple[Series, Series]:
  '''Return Donchian's upper and lower channel.

  The upper channel is the highest high and the lower the lowest low of
  the trailing ``window`` bars *including* the current one. A trailing
  window is the whole point: a centred window would read bar
  ``i + window / 2``, which is look-ahead, and this implementation
  never does.

  # ponytail: ``max`` and ``min`` rescan the window on every bar, so the
  # cost is ``O(bars * window)``. Ceiling: a 50-year daily panel at
  # ``window = 200`` is ten million comparisons, which is fine here but
  # is not a shape to grow into. Upgrade path: two monotonic deques,
  # one for highs and one for lows, with the same trailing-window
  # semantics.

  Args:
    highs: Highs in ascending time order.
    lows: Lows in ascending time order.
    window: Lookback in observations.

  Returns:
    ``(upper, lower)``, each aligned with ``highs`` and ``None`` for the
    first ``window - 1`` entries.

  Raises:
    ValueError: If ``window`` is not positive, or if ``highs`` and
      ``lows`` differ in length.

  Example:
    ``donchian_channels([3, 5, 4], [1, 2, 0], 2)`` gives upper
    ``None, 5, 5`` and lower ``None, 1, 0``: bar 1 sees bars 0 and 1,
    bar 2 sees bars 1 and 2.
  '''
  if window < 1:
    raise ValueError(f'window must be >= 1, got {window}')
  _require_same_length('highs', highs, 'lows', lows)
  upper: Series = [None] * len(highs)
  lower: Series = [None] * len(highs)
  for index in range(window - 1, len(highs)):
    start = index - window + 1
    upper[index] = max(highs[start:index + 1])
    lower[index] = min(lows[start:index + 1])
  return upper, lower


def linear_regression(
  values: list[float],
  window: int,
) -> tuple[Series, Series]:
  '''Return the rolling least-squares slope and endpoint of the fit.

  At each bar the trailing ``window`` points are fitted by ordinary
  least squares against ``x = 0, 1, ... window - 1``, and both outputs
  are reported at the current bar:

  * ``slope`` is the ordinary least-squares slope, per bar.
  * ``intercept`` is the fitted line evaluated at the *newest* bar of the
    window, that is ``mean_y + slope * (window - 1 - mean_x)``.

  Returning the endpoint rather than the raw ``x = 0`` intercept is a
  deliberate choice, and the raw intercept of a rolling window is a
  figure ``window - 1`` bars stale with a different scale for every
  window length, which makes it useless as a comparable level. The
  endpoint is what a regression channel ends at.

  Args:
    values: Input series.
    window: Number of trailing observations to fit, at least 2.

  Returns:
    ``(slope, intercept)``, each aligned with ``values`` and ``None``
    for the first ``window - 1`` entries. A flat window gives slope
    ``0.0`` and an intercept equal to that level, which is a meaningful
    reading rather than an undefined one.

  Raises:
    ValueError: If ``window`` is less than 2, which leaves no variance
      in ``x``.

  # ponytail: the fit is recomputed from scratch every bar, so the cost
  # is ``O(bars * window)``. Ceiling: enough for daily panels, not for
  # a tick or minute series over years. Upgrade path: rolling sums of
  # ``x``, ``y``, ``xy`` and ``y**2`` give the identical closed form in
  # ``O(1)`` per bar.

  Example:
    ``linear_regression([1, 2, 3, 4], 4)`` has ``mean_x = 1.5`` and
    ``mean_y = 2.5``, numerator ``2.25 + 0.25 + 0.25 + 2.25 = 5`` and
    denominator ``2.25 + 0.25 + 0.25 + 2.25 = 5``, so the slope is
    ``1`` and the endpoint is ``2.5 + 1 * (3 - 1.5) = 4``.
  '''
  if window < 2:
    raise ValueError(f'window must be >= 2, got {window}')
  slopes: Series = [None] * len(values)
  intercepts: Series = [None] * len(values)
  mean_x = (window - 1) / 2.0
  denominator = sum(
    (index - mean_x) ** 2 for index in range(window))
  for index in range(window - 1, len(values)):
    sample = values[index - window + 1:index + 1]
    mean_y = sum(sample) / window
    numerator = sum(
      (position - mean_x) * (value - mean_y)
      for position, value in enumerate(sample))
    slope = numerator / denominator
    slopes[index] = slope
    intercepts[index] = mean_y + slope * (window - 1 - mean_x)
  return slopes, intercepts
