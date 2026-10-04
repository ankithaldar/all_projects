#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Technical indicators over a bar series.

Pure functions, no state, no I/O, standard library only. Every function
returns a list the same length as its input, with ``None`` in the warm-up
region where the indicator is not yet defined. Returning ``None`` rather
than zero or a partial value matters: a zero would be silently
arithmetically valid and could enter a position size, whereas ``None``
forces the caller to decide what an undefined indicator means.

The selection here is evidence-led rather than exhaustive. Momentum and
realised volatility are included because they are the two anomalies with
the strongest published support on Indian equities; moving averages and
trend slope are included because they are the practical filter on which
signals are conditioned. RSI and ATR are included as exposure and
volatility measures, not as standalone signals.
'''

from __future__ import annotations

import math

from stock_rl.bars import Bar

__all__ = [
  'atr',
  'ema',
  'momentum',
  'realized_volatility',
  'rsi',
  'sma',
  'trend_slope',
]


def sma(values: list[float], window: int) -> list[float | None]:
  '''Return the simple moving average of ``values``.

  Args:
    values: Input series.
    window: Lookback in observations.

  Returns:
    List aligned with ``values``, ``None`` until ``window`` observations
    have accumulated.

  Raises:
    ValueError: If ``window`` is not positive.
  '''
  if window < 1:
    raise ValueError(f'window must be >= 1, got {window}')
  out: list[float | None] = []
  running = 0.0
  for index, value in enumerate(values):
    running += value
    if index >= window:
      running -= values[index - window]
    out.append(running / window if index >= window - 1 else None)
  return out


def ema(values: list[float], window: int) -> list[float | None]:
  '''Return the exponential moving average of ``values``.

  Args:
    values: Input series.
    window: Span in observations. The smoothing factor is ``2/(n+1)``.

  Returns:
    List aligned with ``values``, ``None`` for the warm-up region.
  '''
  if window < 1:
    raise ValueError(f'window must be >= 1, got {window}')
  factor = 2.0 / (window + 1.0)
  out: list[float | None] = []
  current: float | None = None
  for value in values:
    current = value if current is None else (
      value * factor + current * (1.0 - factor))
    out.append(current)
  return out


def momentum(closes: list[float], lookback: int) -> float | None:
  '''Return the trailing total return over ``lookback`` observations.

  This is the raw input to a momentum signal. It is intentionally a plain
  price return with no skip-month adjustment: the skip is applied by the
  caller when constructing a 12-1 style signal, so the primitive stays
  free of hidden convention.

  Args:
    closes: Closing prices in ascending time order.
    lookback: Number of bars to look back.

  Returns:
    Fractional return over the window, or ``None`` if there is not
  enough history.
  '''
  if lookback < 1:
    raise ValueError(f'lookback must be >= 1, got {lookback}')
  if len(closes) <= lookback:
    return None
  past = closes[-lookback - 1]
  if past <= 0.0:
    return None
  return closes[-1] / past - 1.0


def realized_volatility(
  closes: list[float],
  window: int,
  periods: int = 252,
) -> list[float | None]:
  '''Return annualised realised volatility of simple returns.

  Uses the sample standard deviation of per-bar returns scaled by the
  square root of the period count, so the result is directly comparable
  across symbols regardless of bar frequency.

  Args:
    closes: Closing prices in ascending time order.
    window: Lookback in bars.
    periods: Periods per year. 252 for daily NSE bars.

  Returns:
    List aligned with ``closes``, ``None`` during warm-up.

  Raises:
    ValueError: If ``window`` or ``periods`` is not positive.
  '''
  if window < 2:
    raise ValueError(f'window must be >= 2, got {window}')
  if periods < 1:
    raise ValueError(f'periods must be >= 1, got {periods}')
  returns: list[float | None] = [None]
  for index in range(1, len(closes)):
    previous = closes[index - 1]
    returns.append(
      None if previous <= 0.0 else closes[index] / previous - 1.0)
  out: list[float | None] = []
  for index in range(len(closes)):
    if index < window:
      out.append(None)
      continue
    sample = [value for value in returns[index - window + 1:index + 1]
              if value is not None]
    if len(sample) < 2:
      out.append(None)
      continue
    mean = sum(sample) / len(sample)
    variance = sum((value - mean) ** 2 for value in sample)
    variance /= len(sample) - 1
    out.append(math.sqrt(variance) * math.sqrt(periods))
  return out


def rsi(closes: list[float], window: int = 14) -> list[float | None]:
  '''Return Wilder's Relative Strength Index, scaled to ``[0, 100]``.

  The first value appears at index ``window``, because seeding the
  averages needs ``window`` price changes and there is one fewer change
  than there are closes.

  Args:
    closes: Closing prices in ascending time order.
    window: Lookback in bars.

  Returns:
    List aligned with ``closes``, ``None`` during warm-up. A flat series
    yields 50.0, the neutral reading, rather than dividing by zero.
  '''
  if window < 1:
    raise ValueError(f'window must be >= 1, got {window}')
  out: list[float | None] = [None] * len(closes)
  if len(closes) <= window:
    return out
  gains = 0.0
  losses = 0.0
  for index in range(1, window + 1):
    change = closes[index] - closes[index - 1]
    gains += max(change, 0.0)
    losses += max(-change, 0.0)
  average_gain = gains / window
  average_loss = losses / window
  out[window] = _rsi_value(average_gain, average_loss)
  for position in range(window + 1, len(closes)):
    change = closes[position] - closes[position - 1]
    average_gain = (
      average_gain * (window - 1) + max(change, 0.0)) / window
    average_loss = (
      average_loss * (window - 1) + max(-change, 0.0)) / window
    out[position] = _rsi_value(average_gain, average_loss)
  return out


def _rsi_value(average_gain: float, average_loss: float) -> float:
  '''Return the RSI value for smoothed average gain and loss.

  Args:
    average_gain: Smoothed average gain.
    average_loss: Smoothed average loss.

  Returns:
    RSI in ``[0, 100]``. An unbroken advance yields 100.0, which is the
    correct reading, and a completely flat series yields 50.0 rather
    than dividing 0 by 0.
  '''
  if average_gain == 0.0 and average_loss == 0.0:
    return 50.0
  if average_loss == 0.0:
    return 100.0
  strength = average_gain / average_loss
  return 100.0 - 100.0 / (1.0 + strength)


def atr(bars: list[Bar], window: int = 14) -> list[float | None]:
  '''Return Wilder's Average True Range.

  True range is the largest of the high-low span, the absolute move from
  the previous close, and the absolute move into the current close. The
  cross-bar terms are what stop a gap from registering as a small range.

  Args:
    bars: Price bars in ascending time order.
    window: Smoothing window in bars.

  Returns:
    List aligned with ``bars``, ``None`` during warm-up.
  '''
  if window < 1:
    raise ValueError(f'window must be >= 1, got {window}')
  if len(bars) < 2:
    return [None] * len(bars)
  ranges: list[float] = []
  for index in range(1, len(bars)):
    current = bars[index]
    previous_close = bars[index - 1].close
    ranges.append(max(
      current.high - current.low,
      abs(current.high - previous_close),
      abs(current.low - previous_close),
    ))
  return _wilder_smooth(ranges, window, len(bars))


def _wilder_smooth(
  ranges: list[float],
  window: int,
  length: int,
) -> list[float | None]:
  '''Return a Wilder-smoothed series over precomputed values.

  Wilder smoothing seeds with a simple average of the first ``window``
  values, then applies an exponential decay with weight ``1/window`` on
  the new observation.

  Args:
    ranges: Values to smooth, one per bar after the first.
    window: Smoothing window.
    length: Total expected output length, matching the bar count.

  Returns:
    The smoothed series, aligned so index ``i`` corresponds to the value
    derived from ``ranges[i - 1]``.
  '''
  result: list[float | None] = [None] * length
  if len(ranges) < window:
    return result
  average = sum(ranges[:window]) / window
  result[window] = average
  for position in range(window, len(ranges)):
    average = (average * (window - 1) + ranges[position]) / window
    target = position + 1
    if target < length:
      result[target] = average
  return result


def trend_slope(values: list[float], window: int) -> float | None:
  '''Return the ordinary-least-squares slope of the last ``window`` points.

  Normalised by the mean level so the result is comparable across symbols
  priced at different magnitudes: a slope of 0.01 means the series is
  rising one percent of its own average per bar.

  Args:
    values: Input series.
    window: Number of trailing observations to fit.

  Returns:
    Slope per bar, or ``None`` if there is not enough history. A flat
    series legitimately returns 0.0, which is a meaningful "no trend"
    reading rather than an undefined one.
  '''
  if window < 2:
    raise ValueError(f'window must be >= 2, got {window}')
  if len(values) < window:
    return None
  sample = values[-window:]
  mean_x = (window - 1) / 2.0
  mean_y = sum(sample) / window
  numerator = sum((index - mean_x) * (value - mean_y)
                  for index, value in enumerate(sample))
  denominator = sum((index - mean_x) ** 2 for index in range(window))
  if denominator == 0.0 or mean_y == 0.0:
    return None
  return numerator / (denominator * mean_y)
