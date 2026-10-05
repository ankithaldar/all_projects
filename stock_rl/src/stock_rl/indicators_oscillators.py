#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Oscillator indicators over a bar series.

Pure functions, no state, no I/O, standard library only. Every function
returns a list the same length as its input, with ``None`` through the
warm-up region, exactly as ``indicators.py`` does. ``None`` rather than
zero: a zero is arithmetically valid and could silently enter a
position size, whereas ``None`` forces the caller to decide what an
undefined indicator means. ``None`` is also returned where a base price
is non-positive, which means "undefined", not "warm-up".

THESE ARE FORMULAS, NOT SIGNALS
Implementing a published formula is not evidence that the published
formula has predictive value, and nothing here claims otherwise.
``docs/research/methodology/technical-indicators-nse.md`` records that
the indicator thresholds behind this project's ranking have no published
Indian equity evidence at all; that Rink (2023) tested 6,406 technical
rules with a stepwise Superior Predictive Ability test and found the
out-of-sample excess Sharpe of the emerging-market portfolio
significantly negative once costs were charged; and that his own
conclusion is that technical trading signals should be treated as
noise. The project's own headline finding is likewise negative -- on
Nifty-50 panel data the simple baselines beat the reinforcement
learning agents, and buy-and-hold beat the LLM agents (FINSABER, KDD
2026). Every level, threshold and crossover a vendor attaches to the
indicators below is therefore unfitted on this project. This module
deliberately exposes no ``overbought``/``oversold`` API and no crossover
helper: returning a raw series keeps the unvalidated judgement out of
the primitive, and thresholds belong to a strategy layer that has to
earn its keep on held-out Indian data.

The two levels that do appear, Williams %R at -80/+20 and CCI at
+/-100, are definitional conventions of those two indicators -- part of
what "a Williams %R" and "a CCI" are -- not recommendations.

Bar fields used differ by function and are named in each docstring. No
function here reads ``timestamp``: ordering is assumed and validated
upstream by ``bars.load_csv``.

The formulas follow the A-Z reference at
https://www.incrediblecharts.com/indicators/technical-indicators.php
except where a docstring says otherwise: Williams' own published
Ultimate Oscillator, Chande and Kroll's Stochastic RSI scaling, and
Lambert's CCI note are the three places where the reference page
simplifies a published definition.
'''

from __future__ import annotations

from stock_rl.bars import Bar
from stock_rl.indicators import ema, rsi, sma

__all__ = [
  'awesome_oscillator',
  'chaikin_oscillator',
  'commodity_channel_index',
  'money_flow_index',
  'rate_of_change',
  'roc_on_volume',
  'slow_stochastic_d',
  'smoothed_roc',
  'stochastic_d',
  'stochastic_k',
  'stochastic_rsi',
  'ultimate_oscillator',
  'williams_ad',
  'williams_percent_r',
]

#: CCI's published scale factor. Lambert chose it so that roughly seven
#: out of ten readings of a well-behaved series fall inside +/-100, which
#: is what makes the +/-100 line a convention of the indicator rather
#: than a level with independent meaning.
_CCI_SCALE = 0.015


def stochastic_k(bars: list[Bar], window: int = 14) -> list[float | None]:
  '''Return the fast stochastic %K, the close's position in the range.

  ``%K = 100 * (close - lowest low) / (highest high - lowest low)``,
  both extremes taken over the ``window`` bars ending at the current
  bar. This is the single-lookback raw oscillator; every other
  stochastic line in this module is a smoothing of it.

  Bar fields used: ``high``, ``low``, ``close``. Not ``open``, not
  ``volume``.

  Args:
    bars: Price bars in ascending time order.
    window: Lookback in bars, at least 1.

  Returns:
    List aligned with ``bars``. First defined index is ``window - 1``,
    since ``window`` bars end there. Values lie in ``[0, 100]``. A window
    with no intrabar range returns ``50.0``: the close is then
    simultaneously the highest high and the lowest low, so its position
    in the range is undefined and the midpoint of the scale is reported
    rather than dividing by zero.

  Raises:
    ValueError: If ``window`` is not positive.
  '''
  _require_positive('window', window)
  out: list[float | None] = [None] * len(bars)
  for index in range(window - 1, len(bars)):
    lowest, highest = _window_range(bars, index, window)
    if highest <= lowest:
      # ponytail: a zero-width window prints the midpoint of the scale
      # rather than None, because None is reserved for warm-up here and
      # a halted bar is not missing data. Ceiling: the reading is
      # genuinely undefined. Upgrade path: return a companion series of
      # defined-ness flags alongside the values.
      out[index] = 50.0
    else:
      out[index] = 100.0 * (bars[index].close - lowest) / (highest - lowest)
  return out


def stochastic_d(
  bars: list[Bar],
  window: int = 14,
  signal_window: int = 3,
) -> list[float | None]:
  '''Return the fast stochastic %D, a simple average of the fast %K.

  ``%D = SMA(signal_window, %K)``. The published convention is a
  three-bar average; the smoothing is the whole difference from
  ``stochastic_k``, which is why the two are separate functions and why
  the warm-up is one bar longer.

  Bar fields used: ``high``, ``low``, ``close``.

  Args:
    bars: Price bars in ascending time order.
    window: %K lookback in bars, at least 1.
    signal_window: Averaging window for %D, at least 1.

  Returns:
    List aligned with ``bars``. First defined index is
    ``window + signal_window - 2``: ``window - 1`` bars for %K, then
    ``signal_window - 1`` more for the average. Values lie in
    ``[0, 100]``.

  Raises:
    ValueError: If ``window`` or ``signal_window`` is not positive.
  '''
  _require_positive('window', window)
  _require_positive('signal_window', signal_window)
  return _smooth(_stochastic_series(bars, window), window - 1, signal_window)


def slow_stochastic_d(
  bars: list[Bar],
  window: int = 14,
  smooth_k: int = 3,
  smooth_d: int = 3,
) -> list[float | None]:
  '''Return the slow stochastic %D line.

  Slow %K is the fast %D, and slow %D is a further average of slow %K,
  so slow %D smooths the raw %K twice. That double smoothing is what
  separates it from ``stochastic_d``: the published slow stochastic
  trades a faster line for a less noisy one. The slow %K line itself is
  ``stochastic_d(bars, window, smooth_k)`` and is not duplicated here.

  Bar fields used: ``high``, ``low``, ``close``.

  Args:
    bars: Price bars in ascending time order.
    window: %K lookback in bars, at least 1.
    smooth_k: First averaging window applied to %K, at least 1.
    smooth_d: Second averaging window, at least 1.

  Returns:
    List aligned with ``bars``. First defined index is
    ``window + smooth_k + smooth_d - 3``, the sum of the three warm-up
    contributions, because the averaging windows are sequential rather
    than nested.

  Raises:
    ValueError: If any window is not positive.
  '''
  _require_positive('window', window)
  _require_positive('smooth_k', smooth_k)
  _require_positive('smooth_d', smooth_d)
  first_k = _smooth(_stochastic_series(bars, window), window - 1, smooth_k)
  return _smooth(first_k, window + smooth_k - 2, smooth_d)


def williams_percent_r(
  bars: list[Bar],
  window: int = 14,
) -> list[float | None]:
  '''Return Williams %R, the stochastic %K on a ``[-100, 0]`` scale.

  ``%R = -100 * (highest high - close) / (highest high - lowest low)``,
  which is exactly ``%K - 100``: the two are the same information on
  different scales, and both are kept because the vendor literature
  refers to them by different names.

  Bar fields used: ``high``, ``low``, ``close``.

  Args:
    bars: Price bars in ascending time order.
    window: Lookback in bars, at least 1.

  Returns:
    List aligned with ``bars``. First defined index is ``window - 1``.
    Values lie in ``[-100, 0]``. The -80 and +20 lines that give this
    indicator its name are conventions of the published definition, not
    levels validated anywhere in this project; a window with no
    intrabar range returns ``-50.0``, the midpoint of the scale.

  Raises:
    ValueError: If ``window`` is not positive.
  '''
  _require_positive('window', window)
  out: list[float | None] = [None] * len(bars)
  for index in range(window - 1, len(bars)):
    lowest, highest = _window_range(bars, index, window)
    if highest <= lowest:
      # ponytail: same midpoint convention as the fast stochastic, for
      # the same reason. Ceiling: undefined rather than neutral.
      # Upgrade path: a companion series of defined-ness flags.
      out[index] = -50.0
    else:
      span = highest - lowest
      out[index] = -100.0 * (highest - bars[index].close) / span
  return out


def commodity_channel_index(
  bars: list[Bar],
  window: int = 20,
) -> list[float | None]:
  '''Return the Commodity Channel Index.

  ``typical price = (high + low + close) / 3``; ``TPMA`` is its simple
  moving average over ``window`` bars; the mean deviation is that
  deviation from ``TPMA`` smoothed by Wilder's recursive average,
  seeded with the simple mean of the first ``window`` deviations; and

    ``CCI = (typical price - TPMA) / (0.015 * mean deviation)``

  The mean deviation is Wilder-smoothed, not a plain mean, which is
  what distinguishes this from Lambert's original presentation. The
  reference page documents Lambert's variant, which averages
  ``|typical price - TPMA|`` around the *current* ``TPMA`` over the
  window; that is a different number on trending data, not a rounding
  difference.

  The +/-100 line is Lambert's scale constant, which he set so that
  roughly seven in ten readings of a well-behaved series fall inside it.
  It is a convention of the definition, not a cap and not a level this
  project has validated: a jump out of a quiet stretch prints well
  beyond it.

  Bar fields used: ``high``, ``low``, ``close`` via ``Bar.typical_price``.

  Args:
    bars: Price bars in ascending time order.
    window: Lookback in bars, at least 1.

  Returns:
    List aligned with ``bars``. First defined index is
    ``2 * window - 2``: the moving average needs ``window - 1`` bars
    and the Wilder seed needs ``window`` deviations, of which the first
    exists at ``window - 1``. The indicator is unbounded. A zero mean
    deviation returns ``0.0``, which is the correct limit for a series
    whose typical price does not move.

  Raises:
    ValueError: If ``window`` is not positive.
  '''
  _require_positive('window', window)
  out: list[float | None] = [None] * len(bars)
  first = 2 * window - 2
  if len(bars) <= first:
    return out
  typical = [bar.typical_price for bar in bars]
  average = sma(typical, window)
  deviation = [
    abs(value - mean) if mean is not None else 0.0
    for value, mean in zip(typical, average)
  ]
  smoothed = _wilder_seed(deviation, window - 1, window)
  for index in range(first, len(bars)):
    spread = smoothed[index]
    offset = typical[index] - average[index]
    if spread == 0.0:
      # ponytail: a zero mean deviation prints 0.0. Ceiling: on a typical
      # price that does not move the reading is undefined, not neutral.
      # Upgrade path: return the deviation alongside the ratio so a
      # caller can see the zero instead of the quotient.
      out[index] = 0.0
    else:
      out[index] = offset / (_CCI_SCALE * spread)
  return out


def ultimate_oscillator(
  bars: list[Bar],
  fast: int = 7,
  middle: int = 14,
  slow: int = 28,
) -> list[float | None]:
  '''Return Larry Williams' Ultimate Oscillator.

  ``buying pressure = close - min(low, previous close)`` and
  ``true range = max(high, previous close) - min(low, previous close)``
  -- both include the previous close, which is what stops a gap from
  registering as a small range. With ``Avg_n`` the mean of the last
  ``n`` values:

    ``UO = 100 * (4*Avg_fast(BP) + 2*Avg_middle(BP) + Avg_slow(BP))
    / (4*Avg_fast(TR) + 2*Avg_middle(TR) + Avg_slow(TR))``

  That is a ratio of weighted sums, which is not the same number as the
  weighted average of the three separate buying percentages given on
  the reference page; Williams' published form is used here. The three
  periods default to 7, 14 and 28, each double the last, as Williams
  specified.

  Bar fields used: ``high``, ``low``, ``close``, and the previous
  bar's ``close``.

  Args:
    bars: Price bars in ascending time order.
    fast: Shortest buying-pressure window, at least 1.
    middle: Middle window, at least 1.
    slow: Longest window, at least 1.

  Returns:
    List aligned with ``bars``. First defined index is ``max(fast,
    middle, slow)``, which is ``slow`` for the published setup. The
    three lookbacks are *nested*, not additive: the 28-bar sum
    already contains the 14-bar and 7-bar sums, so waiting for
    ``fast + middle + slow`` bars would discard 21 bars of
    computable history for nothing. One extra bar beyond ``slow`` is
    needed only because buying pressure and true range are undefined at
    index 0, where there is no previous close. Values lie in
    ``[0, 100]``: buying pressure cannot be negative, because the close
    of a well-formed bar is never below its own low. A series with no
    true range at all returns ``50.0``.

  Raises:
    ValueError: If any period is not positive.
  '''
  for name, value in (('fast', fast), ('middle', middle), ('slow', slow)):
    _require_positive(name, value)
  out: list[float | None] = [None] * len(bars)
  first = max(fast, middle, slow)
  if len(bars) <= first:
    return out
  buying = [0.0] * len(bars)
  span = [0.0] * len(bars)
  for index in range(1, len(bars)):
    previous_close = bars[index - 1].close
    lowest = min(bars[index].low, previous_close)
    highest = max(bars[index].high, previous_close)
    buying[index] = bars[index].close - lowest
    span[index] = highest - lowest
  for index in range(first, len(bars)):
    numerator = 0.0
    denominator = 0.0
    for period, weight in ((fast, 4.0), (middle, 2.0), (slow, 1.0)):
      stop = index + 1
      numerator += weight * sum(buying[stop - period:stop]) / period
      denominator += weight * sum(span[stop - period:stop]) / period
    if denominator == 0.0:
      # ponytail: a wholly flat series prints 50.0 for the whole window.
      # Ceiling: no true range means no position, so this is undefined
      # rather than neutral. Upgrade path: return the three buying
      # percentages alongside the oscillator.
      out[index] = 50.0
    else:
      out[index] = 100.0 * numerator / denominator
  return out


def stochastic_rsi(
  closes: list[float],
  rsi_window: int = 14,
  window: int = 14,
) -> list[float | None]:
  '''Return Stochastic RSI, Chande and Kroll's stochastic of the RSI.

  ``(RSI - min RSI) / (max RSI - min RSI)`` over ``window`` RSI
  readings, using the Wilder RSI already in ``indicators.rsi``. The
  reference page states it unscaled, in ``[0, 1]``; the widely used
  ``[0, 100]`` variant is the same number times 100 and is left to the
  caller. Applying a second stochastic to a bounded oscillator is what
  makes this far more extreme than the RSI it wraps -- it is not the
  RSI, and it is not a price stochastic either.

  Args:
    closes: Closing prices in ascending time order.
    rsi_window: RSI lookback in bars, at least 1.
    window: Lookback in RSI readings, at least 1.

  Returns:
    List aligned with ``closes``. First defined index is
    ``rsi_window + window - 1``: the RSI itself is undefined until
    ``rsi_window``, and ``window`` RSI readings are then needed. Values
    lie in ``[0, 1]``. A window in which the RSI does not move returns
    ``0.5``, the midpoint, since the position is undefined.

  Raises:
    ValueError: If ``rsi_window`` or ``window`` is not positive.
  '''
  _require_positive('rsi_window', rsi_window)
  _require_positive('window', window)
  out: list[float | None] = [None] * len(closes)
  strength = rsi(closes, rsi_window)
  for index in range(rsi_window + window - 1, len(closes)):
    sample = [value for value in strength[index - window + 1:index + 1]
              if value is not None]
    lowest = min(sample)
    highest = max(sample)
    if highest <= lowest:
      # ponytail: a pinned RSI prints 0.5. Ceiling: an unbroken advance
      # pins the wrapped RSI at 100 and this indicator then cannot report
      # 1.0 on it at all, which is a property of the construction and not
      # a bug. Upgrade path: return the RSI window alongside the ratio.
      out[index] = 0.5
    else:
      out[index] = (strength[index] - lowest) / (highest - lowest)
  return out


def money_flow_index(
  bars: list[Bar],
  window: int = 14,
) -> list[float | None]:
  '''Return the Money Flow Index, the volume-weighted RSI.

  ``typical price = (high + low + close) / 3`` and
  ``money flow = typical price * volume``. Positive money flow sums the
  money flow of every bar whose typical price rose against the previous
  bar, negative money flow sums the bars where it fell; a bar whose
  typical price is unchanged counts as neither, as the published
  definition has no rule for it. Then
  ``MFI = 100 - 100 / (1 + positive / negative)``.

  This is not ``rsi`` with volume substituted: RSI splits price
  *changes* into gains and losses, while MFI splits *money flow* by the
  direction of the typical price, so the volume weighting is inside
  both sums and a high-volume down bar can dominate a low-volume rally.

  Bar fields used: ``high``, ``low``, ``close`` via ``Bar.typical_price``,
  plus ``volume``.

  Args:
    bars: Price bars in ascending time order.
    window: Lookback in bars, at least 1.

  Returns:
    List aligned with ``bars``. First defined index is ``window``, one
    bar later than a window-bounded price oscillator: the window holds
    ``window`` money flows but ``window + 1`` typical prices, because
    the first flow in the window still needs its direction. Values lie
    in ``[0, 100]``. No money flow at either end returns ``50.0``,
    mirroring ``indicators.rsi`` on a flat series.

  Raises:
    ValueError: If ``window`` is not positive.
  '''
  _require_positive('window', window)
  out: list[float | None] = [None] * len(bars)
  if len(bars) <= window:
    return out
  typical = [bar.typical_price for bar in bars]
  flow = [bar.typical_price * bar.volume for bar in bars]
  for index in range(window, len(bars)):
    positive = 0.0
    negative = 0.0
    for position in range(index - window + 1, index + 1):
      change = typical[position] - typical[position - 1]
      if change > 0.0:
        positive += flow[position]
      elif change < 0.0:
        negative += flow[position]
    if negative == 0.0:
      # ponytail: no money flow on the negative leg prints 100, and none
      # on either leg prints 50, mirroring indicators.rsi on a flat
      # series. Ceiling: on a series that never trades this is a
      # constant neutral reading rather than an undefined one. Upgrade
      # path: return the positive and negative sums and let the caller
      # decide what an empty window means.
      out[index] = 50.0 if positive == 0.0 else 100.0
    else:
      out[index] = 100.0 - 100.0 / (1.0 + positive / negative)
  return out


def smoothed_roc(
  closes: list[float],
  ema_window: int = 13,
  lookback: int = 21,
) -> list[float | None]:
  '''Return Smoothed Rate of Change, Schutzman's 1991 two-step form.

  An exponential moving average of the closes over ``ema_window``, then
  the percentage rate of change of that average over ``lookback``.
  Smoothing first is the entire point: it is why the indicator gives
  fewer and later signals than the raw rate of change, not why it is
  better.

  Args:
    closes: Closing prices in ascending time order.
    ema_window: Smoothing span in bars, at least 1.
    lookback: Rate-of-change window in bars, at least 1.

  Returns:
    List aligned with ``closes``. First defined index is ``lookback``:
    the EMA is seeded at the first close exactly as ``indicators.ema``
    does, so only the rate of change contributes warm-up. A
    non-positive base at ``index - lookback`` returns ``None``.

  Raises:
    ValueError: If ``ema_window`` or ``lookback`` is not positive.
  '''
  _require_positive('ema_window', ema_window)
  _require_positive('lookback', lookback)
  smoothed = ema(closes, ema_window)
  out: list[float | None] = [None] * len(closes)
  for index in range(lookback, len(closes)):
    base = smoothed[index - lookback]
    out[index] = (None if base <= 0.0
                  else 100.0 * (smoothed[index] - base) / base)
  return out


def rate_of_change(
  values: list[float],
  lookback: int = 12,
) -> list[float | None]:
  '''Return Rate of Change as a percentage around zero.

  ``100 * (value - value[lookback]) / value[lookback]``, the default
  window being the twelve bars given on the reference page. This is
  ``indicators.momentum`` times 100, returned as a series; the
  primitive is kept separate so the same formula applies to any
  quantity, which is what ``roc_on_volume`` does.

  Args:
    values: Price or volume series in ascending time order.
    lookback: Lookback in bars, at least 1.

  Returns:
    List aligned with ``values``. First defined index is ``lookback``.
    A non-positive base returns ``None``.

  Raises:
    ValueError: If ``lookback`` is not positive.
  '''
  _require_positive('lookback', lookback)
  out: list[float | None] = [None] * len(values)
  for index in range(lookback, len(values)):
    base = values[index - lookback]
    out[index] = (None if base <= 0.0
                  else 100.0 * (values[index] - base) / base)
  return out


def roc_on_volume(bars: list[Bar], lookback: int = 12) -> list[float | None]:
  '''Return Rate of Change of volume, as a percentage.

  The price formula applied to traded quantity. The research file is
  explicit that volume has a role as a conditioning variable and as a
  tilt on momentum, while volume-based standalone signals such as this
  one and OBV have no Indian evidence at all.

  Bar fields used: ``volume`` only.

  Args:
    bars: Price bars in ascending time order.
    lookback: Lookback in bars, at least 1.

  Returns:
    List aligned with ``bars``. First defined index is ``lookback``.

  Raises:
    ValueError: If ``lookback`` is not positive.
  '''
  _require_positive('lookback', lookback)
  return rate_of_change([bar.volume for bar in bars], lookback)


def williams_ad(bars: list[Bar]) -> list[float | None]:
  '''Return Williams Accumulate/Distribute, the range-only cumulative.

  On an up bar, add ``close - min(low, previous close)``; on a down
  bar, subtract ``max(high, previous close) - close``; on an unchanged
  close add nothing. Following Achelis, as the reference page does,
  volume is deliberately omitted, so despite the name this is a
  cumulative trading-range measure and not a volume indicator. The
  previous close enters through both ``min`` and ``max`` so that a gap
  widens the range being accumulated.

  Bar fields used: ``high``, ``low``, ``close``, and the previous
  bar's ``close``. Not ``volume``.

  Args:
    bars: Price bars in ascending time order.

  Returns:
    List aligned with ``bars``. The first value is at index 1: at index
    0 there is no previous close, so no bar has a direction and the
    accumulator is unstarted. The seed is ``0.0``, and a series of
    unchanged closes therefore returns ``0.0`` from index 1 onwards,
    which is the correct reading rather than an undefined one. There
    is no window parameter to validate.
  '''
  out: list[float | None] = [None] * len(bars)
  level = 0.0
  for index in range(1, len(bars)):
    previous_close = bars[index - 1].close
    close = bars[index].close
    if close > previous_close:
      level += close - min(bars[index].low, previous_close)
    elif close < previous_close:
      level -= max(bars[index].high, previous_close) - close
    out[index] = level
  return out


def chaikin_oscillator(
  bars: list[Bar],
  fast: int = 3,
  slow: int = 10,
) -> list[float | None]:
  '''Return the Chaikin Oscillator, the spread of two AD averages.

  Accumulation/distribution for one bar is
  ``volume * ((close - low) - (high - close)) / (high - low)``, and
  the oscillator is ``EMA(fast, AD) - EMA(slow, AD)`` -- a three-period
  exponential average minus a ten-period one, as published. It
  compares money flow against price action and is unbounded.

  Bar fields used: ``high``, ``low``, ``close``, ``volume``.

  Args:
    bars: Price bars in ascending time order.
    fast: Fast EMA span, at least 1.
    slow: Slow EMA span, at least 1.

  Returns:
    List aligned with ``bars``, defined from index 0. Accumulation
    distribution is defined at the first bar and ``indicators.ema``
    seeds at the first value, so there is no warm-up region to report
    as ``None``.

  Raises:
    ValueError: If ``fast`` or ``slow`` is not positive.
  '''
  _require_positive('fast', fast)
  _require_positive('slow', slow)
  line = _accumulation_distribution(bars)
  fast_line = ema(line, fast)
  slow_line = ema(line, slow)
  return [fast_value - slow_value
          for fast_value, slow_value in zip(fast_line, slow_line)]


def awesome_oscillator(
  bars: list[Bar],
  fast: int = 5,
  slow: int = 34,
) -> list[float | None]:
  '''Return the Awesome Oscillator, a dual moving average difference.

  ``SMA(fast, median price) - SMA(slow, median price)`` where the
  median price is ``(high + low) / 2``, with the published 5 and 34
  bar windows. Fast minus slow is the sign convention used here; some
  sources publish the difference the other way round, so the sign is
  worth stating rather than assuming. A trend agent owns the moving
  averages; this is the unvalidated difference between them.

  Bar fields used: ``high``, ``low``. Not ``close``, not ``volume``.

  Args:
    bars: Price bars in ascending time order.
    fast: Fast SMA window, at least 1.
    slow: Slow SMA window, at least 1.

  Returns:
    List aligned with ``bars``. First defined index is ``slow - 1``,
    set by the slower average. The indicator is unbounded and a flat
    series returns ``0.0``.

  Raises:
    ValueError: If ``fast`` or ``slow`` is not positive.
  '''
  _require_positive('fast', fast)
  _require_positive('slow', slow)
  median = [(bar.high + bar.low) / 2.0 for bar in bars]
  fast_line = sma(median, fast)
  slow_line = sma(median, slow)
  return [None if quick is None or steady is None else quick - steady
          for quick, steady in zip(fast_line, slow_line)]


def _require_positive(name: str, value: int) -> None:
  '''Raise ValueError when a window parameter is not positive.

  Args:
    name: Parameter name, so the message names the offending argument.
    value: Value supplied by the caller.

  Raises:
    ValueError: If ``value`` is below 1.
  '''
  if value < 1:
    raise ValueError(f'{name} must be >= 1, got {value}')


def _window_range(
  bars: list[Bar],
  index: int,
  window: int,
) -> tuple[float, float]:
  '''Return the lowest low and highest high of a trailing window.

  Args:
    bars: Price bars in ascending time order.
    index: Index of the bar the window ends at.
    window: Lookback in bars.

  Returns:
    Pair of (lowest low, highest high). A window with no intrabar
    range returns equal values, which callers report as the undefined
    midpoint of their own scale rather than dividing by zero.
  '''
  start = index - window + 1
  return (min(bar.low for bar in bars[start:index + 1]),
          max(bar.high for bar in bars[start:index + 1]))


def _stochastic_series(bars: list[Bar], window: int) -> list[float]:
  '''Return fast %K as a dense series for smoothing.

  Args:
    bars: Price bars in ascending time order.
    window: %K lookback in bars.

  Returns:
    One value per bar, with ``0.0`` placeholders before the first
    defined reading at ``window - 1``. Callers carry that boundary
    themselves; it is not recoverable from the values.
  '''
  series = [0.0] * len(bars)
  for index in range(window - 1, len(bars)):
    lowest, highest = _window_range(bars, index, window)
    if highest > lowest:
      series[index] = 100.0 * (bars[index].close - lowest) / (
        highest - lowest)
    else:
      series[index] = 50.0
  return series


def _smooth(
  values: list[float],
  first: int,
  window: int,
) -> list[float | None]:
  '''Return a simple moving average of an already-defined series.

  Args:
    values: Dense input series, where indices below ``first`` are
      placeholders rather than observations.
    first: Index of the first real observation.
    window: Averaging window in observations, at least 1.

  Returns:
    List aligned with ``values``, ``None`` until ``first + window - 1``.
    The running sum only ever subtracts an observation at or after
    ``first``, so no placeholder can leak into a value.
  '''
  out: list[float | None] = [None] * len(values)
  running = 0.0
  for index in range(first, len(values)):
    running += values[index]
    if index - first >= window:
      running -= values[index - window]
    if index - first >= window - 1:
      out[index] = running / window
  return out


def _wilder_seed(
  values: list[float],
  first: int,
  window: int,
) -> list[float]:
  '''Return Wilder's recursive average seeded with a simple mean.

  Matches ``indicators._wilder_smooth``: seed with the mean of the
  first ``window`` observations, then carry forward with weight
  ``1/window`` on the new observation.

  Args:
    values: Dense series whose entries before ``first`` are
      placeholders.
    first: Index of the first real observation.
    window: Smoothing window.

  Returns:
    Dense list aligned with ``values``, defined from ``first +
    window - 1``.
  '''
  out = [0.0] * len(values)
  smoothed = sum(values[first:first + window]) / window
  out[first + window - 1] = smoothed
  for index in range(first + window, len(values)):
    smoothed = (smoothed * (window - 1) + values[index]) / window
    out[index] = smoothed
  return out


def _accumulation_distribution(bars: list[Bar]) -> list[float]:
  '''Return the accumulation/distribution line.

  Args:
    bars: Price bars in ascending time order.

  Returns:
    One value per bar. A bar with no intrabar range contributes ``0.0``.
  '''
  # ponytail: a zero-range bar contributes 0.0 rather than being
  # skipped. Ceiling: on a halted or illiquid bar the multiplier is
  # genuinely 0/0 and Chaikin gives no guidance. Upgrade path: expose
  # the skipped positions as ``None`` in a variant that returns the
  # line directly; nothing else needs to change.
  out: list[float] = []
  for bar in bars:
    span = bar.high - bar.low
    if span <= 0.0:
      out.append(0.0)
      continue
    out.append(bar.volume * ((bar.close - bar.low)
                             - (bar.high - bar.close)) / span)
  return out
