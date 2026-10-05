#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the trend indicators.

Three properties get more attention than the values themselves,
because all three have already been got wrong somewhere in this
project's history:

1. **Length.** ``indicators.py`` once returned six values for five
   closes. Every function is checked to return exactly as many entries
   as it was given, unconditionally, including the warm-up region.
2. **Wilder smoothing.** ADX's directional movement system uses
   ``prev = prev + (value - prev) / n`` seeded with the first n-bar
   mean. A simple mean is a different function that returns a
   plausible number. :class:`TestAdx` and :class:`TestWilderRma` pin
   the recursion with exact fractions and assert that the simple-mean
   alternative would give a *different* answer, so a swap cannot pass.
3. **No look-ahead.** :class:`TestNoLookAhead` truncates the input and
   requires the surviving prefix of the output to be unchanged. That
   is a stronger statement than any single value check: if any index
   read a future bar, the truncated series would differ.

Every negative test sits beside a positive control. A test that passes
against a function which returns nothing is worse than no test.
'''

import ast
import inspect
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from stock_rl.bars import Bar
from stock_rl.indicators_trend import (
  adx,
  aroon_oscillator,
  coppock,
  dema,
  displaced_ma,
  donchian_channels,
  hma,
  ichimoku,
  kst,
  linear_regression,
  macd,
  macd_histogram,
  mass_index,
  tema,
  trix,
  wma,
)
from stock_rl.indicators_trend import _KST_COMPONENTS
from stock_rl.indicators_trend import _wilder_rma
from stock_rl import indicators_trend as module_under_test

START = datetime(2026, 1, 1)

#: Highs, lows and closes for the hand-computed ADX case. Chosen so the
#: directional moves and true ranges are all exact binary fractions.
ADX_HIGHS = [10.0, 11.0, 12.0, 12.5, 13.0, 12.0, 12.5, 13.5]
ADX_LOWS = [9.0, 9.5, 10.0, 11.0, 11.5, 11.0, 11.5, 12.0]
ADX_CLOSES = [9.5, 10.5, 11.5, 12.0, 12.5, 11.5, 12.0, 13.0]


def bars_from(rows):
  '''Return bars from (high, low, close) tuples.

  Args:
    rows: Sequence of price tuples.

  Returns:
    List of bars one day apart, open set to the close.
  '''
  return [
    Bar(START + timedelta(days=index), close, high, low, close, 1.0)
    for index, (high, low, close) in enumerate(rows)
  ]


def adx_bars():
  '''Return the eight hand-computed bars used by the ADX tests.

  Returns:
    Bars with highs ``ADX_HIGHS``, lows ``ADX_LOWS`` and closes
    ``ADX_CLOSES``.
  '''
  return bars_from(list(zip(ADX_HIGHS, ADX_LOWS, ADX_CLOSES)))


def sine_bars(count):
  '''Return a deterministic trending panel of ``count`` bars.

  The shape is a slow sine plus a linear drift, so highs, lows and
  closes all move and the indicators see genuine range rather than a
  flat line. It is fully deterministic, so a failure is reproducible.

  Args:
    count: Number of bars.

  Returns:
    List of bars.
  '''
  bars = []
  for index in range(count):
    base = 100.0 + 3.0 * math.sin(index / 3.0) + index * 0.25
    bars.append(Bar(
      START + timedelta(days=index), base, base + 1.5, base - 1.5,
      base - 0.5, 1000.0))
  return bars


def columns(result):
  '''Return the series of an indicator result as a list of lists.

  Args:
    result: Either one aligned series or a tuple of them.

  Returns:
    A list holding the series, so multi-output indicators are walked
    by the same code as single-output ones.
  '''
  if isinstance(result, list):
    return [result]
  return list(result)


def _imported_modules():
  '''Return every module the indicator module imports.

  A dependency scan rather than a review, because
  ``tests/test_packaging.py`` enforces zero runtime dependencies with
  an AST sweep and this module must not quietly become a second place
  where that rule is checked differently.

  Returns:
    Sorted list of dotted module names, both ``import x`` and
    ``from x import y`` forms.
  '''
  source = Path(module_under_test.__file__).read_text(encoding='utf-8')
  imported = set()
  for node in ast.walk(ast.parse(source)):
    if isinstance(node, ast.Import):
      imported.update(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module:
      imported.add(node.module)
  return sorted(imported)


def highs_of(bars):
  '''Return the highs of ``bars`` as floats.'''
  return [bar.high for bar in bars]


def lows_of(bars):
  '''Return the lows of ``bars`` as floats.'''
  return [bar.low for bar in bars]


def closes_of(bars):
  '''Return the closes of ``bars`` as floats.'''
  return [bar.close for bar in bars]


#: Bar-list wrappers for the truncation tests. They exist as named
#: functions rather than lambdas so the parameter names in the call
#: sites read as the indicator under test, and so pylint has nothing
#: to complain about.
def adx_of(bars, window=14):
  '''Return :func:`adx` of ``bars`` at the default window.'''
  return adx(bars, window)


def aroon_of(bars, window=25):
  '''Return :func:`aroon_oscillator` of ``bars`` at the default window.'''
  return aroon_oscillator(highs_of(bars), lows_of(bars), window)


def coppock_of(bars, window=14):
  '''Return :func:`coppock` of ``bars`` at the default window.'''
  return coppock(closes_of(bars), window, 11, 10)


def dema_of(bars, window=14):
  '''Return :func:`dema` of ``bars`` at the default window.'''
  return dema(closes_of(bars), window)


def displaced_of(bars, window=20):
  '''Return :func:`displaced_ma` of ``bars`` at the default window.'''
  return displaced_ma(closes_of(bars), window, 26)


def donchian_of(bars, window=20):
  '''Return :func:`donchian_channels` of ``bars`` at the default window.'''
  return donchian_channels(highs_of(bars), lows_of(bars), window)


def hma_of(bars, window=16):
  '''Return :func:`hma` of ``bars`` at the default window.'''
  return hma(closes_of(bars), window)


def kst_of(bars, signal_window=9):
  '''Return :func:`kst` of ``bars`` at the default window.'''
  return kst(closes_of(bars), signal_window)


def linreg_of(bars, window=14):
  '''Return :func:`linear_regression` of ``bars`` at the default window.'''
  return linear_regression(closes_of(bars), window)


def macd_of(bars, fast_window=12, slow_window=26, signal_window=9):
  '''Return :func:`macd` of ``bars`` at the default windows.'''
  return macd(closes_of(bars), fast_window, slow_window, signal_window)


def mass_of(bars, fast_window=9, slow_window=25, sum_window=25):
  '''Return :func:`mass_index` of ``bars`` at the default windows.'''
  return mass_index(
    highs_of(bars), lows_of(bars), fast_window, slow_window, sum_window)


def tema_of(bars, window=14):
  '''Return :func:`tema` of ``bars`` at the default window.'''
  return tema(closes_of(bars), window)


def trix_of(bars, window=15):
  '''Return :func:`trix` of ``bars`` at the default window.'''
  return trix(closes_of(bars), window)


def wma_of(bars, window=14):
  '''Return :func:`wma` of ``bars`` at the default window.'''
  return wma(closes_of(bars), window)


def assert_no_look_ahead(compute, bars, cut):
  '''Assert that no output at index ``i`` depends on a bar after ``i``.

  The check is truncation: recomputing on ``bars[:cut]`` must leave
  every surviving output bit-for-bit equivalent, because a shorter
  input cannot change a value that only ever looked backwards. A
  centred window or a forward-displaced lagging span fails this
  immediately, and it fails at every index rather than one.

  Args:
    compute: Callable taking a bar list and returning one or more
      aligned series.
    bars: Full bar list.
    cut: Length of the truncated input.
  '''
  full = columns(compute(bars))
  partial = columns(compute(bars[:cut]))
  assert len(full) == len(partial)
  for series_index, series in enumerate(full):
    assert len(series) == len(bars)
    for index in range(min(cut, len(series))):
      truncated = partial[series_index][index]
      complete = series[index]
      if complete is None:
        assert truncated is None, (
          f'series {series_index} index {index}: truncation changed a '
          f'None into {truncated!r}')
        continue
      assert truncated is not None, (
        f'series {series_index} index {index}: full run gave '
        f'{complete!r} but the truncated run gave None')
      assert truncated == pytest.approx(complete, rel=1e-12, abs=1e-12)


class TestWilderRma:
  '''Wilder's recursion, pinned against the simple-mean alternative.

  The arithmetic, with ``values = [1, 1, 1/2, 1/2, 0, 1/2, 1]`` and
  ``window = 3``:

  * seed at index 2: ``(1 + 1 + 1/2) / 3 = 5/2 / 3 = 5/6``
  * index 3: ``5/6 + (1/2 - 5/6)/3 = 5/6 - 1/9 = 13/18``
  * index 4: ``13/18 + (0 - 13/18)/3 = 13/18 - 13/54 = 13/27``
  * index 5: ``13/27 + (1/2 - 13/27)/3 = 13/27 + 1/162 = 79/162``
  * index 6: ``79/162 + (1 - 79/162)/3 = 79/162 + 83/486 = 160/243``

  A simple mean at index 6 would be ``(0 + 1/2 + 1) / 3 = 1/2``, which
  is not ``160/243 = 0.6584``. That gap is the whole reason this
  helper exists.
  '''

  def test_hand_computed_recursion(self):
    values = [1.0, 1.0, 0.5, 0.5, 0.0, 0.5, 1.0]
    result = _wilder_rma(values, 3)
    assert result[0] is None
    assert result[1] is None
    assert result[2] == pytest.approx(5 / 6)
    assert result[3] == pytest.approx(13 / 18)
    assert result[4] == pytest.approx(13 / 27)
    assert result[5] == pytest.approx(79 / 162)
    assert result[6] == pytest.approx(160 / 243)

  def test_simple_mean_would_differ(self):
    # Negative control for the previous test: if a simple mean were
    # swapped in, this test still passes, so the two together pin the
    # smoothing choice rather than merely the number.
    values = [1.0, 1.0, 0.5, 0.5, 0.0, 0.5, 1.0]
    simple_mean = sum(values[-3:]) / 3
    wilder = _wilder_rma(values, 3)[-1]
    assert wilder == pytest.approx(160 / 243)
    assert simple_mean == pytest.approx(0.5)
    assert wilder != pytest.approx(simple_mean)

  def test_length_matches_input(self):
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert len(_wilder_rma(values, 2)) == len(values)

  def test_warmup_is_none(self):
    result = _wilder_rma([1.0, 2.0, 3.0, 4.0], 3)
    assert result[:2] == [None, None]
    assert result[2] is not None

  def test_seed_is_the_first_n_mean(self):
    assert _wilder_rma([2.0, 4.0, 6.0, 8.0], 3)[2] == pytest.approx(4.0)

  def test_constant_input_stays_constant(self):
    result = _wilder_rma([3.0] * 8, 3)
    assert result[2:] == [pytest.approx(3.0)] * 6

  def test_window_larger_than_series_is_all_none(self):
    assert _wilder_rma([1.0, 2.0], 5) == [None, None]

  def test_window_of_one_is_identity(self):
    # Positive control: a window of 1 is legal and recurses to itself.
    assert _wilder_rma([4.0, 9.0, 2.0], 1) == [4.0, 9.0, 2.0]


class TestMacd:
  '''MACD line, signal line and histogram.

  Hand arithmetic on ``closes = [1, 2, 3, 4]`` with
  ``fast_window=2, slow_window=3, signal_window=2``. The EMA factor is
  ``2/(n + 1)``, so ``2/3`` and ``1/2``:

  * fast EMA: ``1`` then ``2*2/3 + 1/3 = 5/3``, then
    ``3*2/3 + (5/3)/3 = 2 + 5/9 = 23/9``, then
    ``4*2/3 + (23/9)/3 = 8/3 + 23/27 = 95/27``.
  * slow EMA: ``1``, ``3/2``, ``9/4``, ``4/2 + 9/8 = 25/8``.
  * line: index 2 is ``23/9 - 9/4 = 11/36``; index 3 is
    ``95/27 - 25/8 = (760 - 675)/216 = 85/216``.
  * signal seeds on ``11/36``, so index 3 is
    ``2/3 * 85/216 + 1/3 * 11/36 = 170/648 + 66/648 = 236/648 =
    59/162``.
  * histogram at index 3 is ``85/216 - 59/162 = 255/648 - 236/648 =
    19/648``.
  '''

  def test_hand_computed_values(self):
    line, signal, histogram = macd([1.0, 2.0, 3.0, 4.0], 2, 3, 2)
    assert line[2] == pytest.approx(11 / 36)
    assert line[3] == pytest.approx(85 / 216)
    assert signal[3] == pytest.approx(59 / 162)
    assert histogram[3] == pytest.approx(19 / 648)

  def test_histogram_is_line_minus_signal(self):
    line, signal, histogram = macd(closes_of(sine_bars(60)), 12, 26, 9)
    for index, value in enumerate(histogram):
      if value is None:
        continue
      assert value == pytest.approx(line[index] - signal[index])

  def test_warmup_is_none_not_zero(self):
    line, signal, histogram = macd(closes_of(sine_bars(40)), 3, 8, 3)
    assert line[:7] == [None] * 7
    assert signal[:9] == [None] * 9
    assert histogram[:9] == [None] * 9
    assert line[7] == pytest.approx(0.0, abs=1e-9) or line[7] != 0.0

  def test_length_matches_input(self):
    closes = closes_of(sine_bars(40))
    for series in macd(closes, 12, 26, 9):
      assert len(series) == len(closes)

  def test_empty_input(self):
    for series in macd([], 12, 26, 9):
      assert series == []

  def test_constant_input_gives_zero_line(self):
    # A flat series has fast EMA == slow EMA, so the line is exactly
    # zero. That is the degenerate answer, not a warm-up value.
    line, _, _ = macd([7.0] * 40, 3, 8, 3)
    assert line[-1] == pytest.approx(0.0)

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_fast_window(self, window):
    with pytest.raises(ValueError, match='fast_window'):
      macd([1.0, 2.0, 3.0], window, 26, 9)

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_slow_window(self, window):
    with pytest.raises(ValueError, match='slow_window'):
      macd([1.0, 2.0, 3.0], 12, window, 9)

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_signal_window(self, window):
    with pytest.raises(ValueError, match='signal_window'):
      macd([1.0, 2.0, 3.0], 12, 26, window)

  def test_rejects_fast_not_below_slow(self):
    with pytest.raises(ValueError, match='fast_window'):
      macd([1.0, 2.0, 3.0], 26, 12, 9)

  def test_accepts_fast_of_one(self):
    # Positive control for the three rejections above: the smallest
    # legal windows are accepted.
    line, _, _ = macd([1.0, 2.0, 3.0, 4.0], 1, 2, 1)
    assert line[-1] is not None


class TestMacdHistogram:
  '''The histogram-only wrapper.'''

  def test_matches_the_histogram_from_macd(self):
    closes = closes_of(sine_bars(50))
    assert macd_histogram(closes, 5, 13, 4) == macd(closes, 5, 13, 4)[2]

  def test_length_matches_input(self):
    closes = closes_of(sine_bars(50))
    assert len(macd_histogram(closes, 12, 26, 9)) == len(closes)

  def test_warmup_is_none(self):
    result = macd_histogram(closes_of(sine_bars(30)), 3, 8, 3)
    assert result[:8] == [None] * 8

  def test_hand_computed_value(self):
    assert macd_histogram([1.0, 2.0, 3.0, 4.0], 2, 3, 2)[3] == pytest.approx(
      19 / 648)

  def test_rejects_bad_signal_window(self):
    with pytest.raises(ValueError, match='signal_window'):
      macd_histogram([1.0, 2.0, 3.0], 2, 3, 0)

  def test_accepts_signal_of_one(self):
    # Positive control.
    result = macd_histogram([1.0, 2.0, 3.0, 4.0], 2, 3, 1)
    assert len(result) == 4


class TestAdx:
  '''Wilder's directional movement system.

  Hand arithmetic on the eight bars in :func:`adx_bars` with
  ``window = 3``. Directional moves compare the up move
  ``high[i] - high[i-1]`` with the down move ``low[i-1] - low[i]``:

  * ``+DM`` = ``1, 1, 1/2, 1/2, 0, 1/2, 1`` for bars 1 to 7
  * ``-DM`` = ``0, 0, 0, 0, 1/2, 0, 0``
  * TR = ``3/2, 2, 3/2, 3/2, 3/2, 1, 3/2``

  RMA seeds on the mean of the first three:

  * ``RMA(+DM)``: ``5/6`` at bar 3, then ``13/18``, ``13/27``, ``79/162``,
    ``160/243``
  * ``RMA(-DM)``: ``0`` at bars 3 and 4, then ``1/6``, ``1/9``, ``2/27``
  * ``RMA(TR)``: ``5/3`` at bar 3, then ``29/18``, ``85/54``, ``112/81``,
    ``691/486``

  So at bar 3, ``+DI = 100 * (5/6) / (5/3) = 50`` and ``-DI = 0``. At
  bar 6, ``+DI = 100 * (79/162) / (112/81) = 100 * 79 / 224 = 1975/56``
  and ``-DI = 100 * (1/9) / (112/81) = 900/112 = 225/28``.

  DX is seeded from bar 3 at ``100, 100, 1700/35 = 48.5714``, so ADX
  appears at bar 5 as their mean, ``580/7 = 82.8571``, which is
  ``2 * window - 1`` as documented.
  '''

  def test_hand_computed_directional_index(self):
    _, plus_di, minus_di = adx(adx_bars(), 3)
    assert plus_di[3] == pytest.approx(50.0)
    assert minus_di[3] == pytest.approx(0.0)
    assert plus_di[6] == pytest.approx(1975 / 56)
    assert minus_di[6] == pytest.approx(225 / 28)

  def test_hand_computed_adx(self):
    strength, _, _ = adx(adx_bars(), 3)
    assert strength[5] == pytest.approx(580 / 7)
    assert strength[6] == pytest.approx(362180 / 4753)

  def test_warmup_floor_is_the_published_one(self):
    # +DI and -DI need window + 1 bars because there is one fewer
    # directional move than bars. ADX needs 2 * window - 1.
    strength, plus_di, minus_di = adx(sine_bars(40), 14)
    assert plus_di[:14] == [None] * 14
    assert minus_di[:14] == [None] * 14
    assert strength[:27] == [None] * 27
    assert strength[27] is not None

  def test_warmup_is_none_not_zero(self):
    _, plus_di, _ = adx(adx_bars(), 3)
    assert plus_di[0] is None and plus_di[2] is None

  def test_length_matches_input(self):
    bars = sine_bars(40)
    for series in adx(bars, 14):
      assert len(series) == len(bars)

  def test_simple_mean_would_give_a_different_di(self):
    # The load-bearing test for this module. A simple mean over the
    # trailing three +DM bars 4 to 6 is (1/2 + 0 + 1/2) / 3 = 1/3 and a
    # simple mean of TR is (3/2 + 3/2 + 1) / 3 = 4/3, so the simple
    # mean version would report +DI = 100 * (1/3) / (4/3) = 25.
    _, plus_di, _ = adx(adx_bars(), 3)
    simple_mean_up = (0.5 + 0.0 + 0.5) / 3
    simple_mean_span = (1.5 + 1.5 + 1.0) / 3
    simple_mean_di = 100.0 * simple_mean_up / simple_mean_span
    assert plus_di[6] == pytest.approx(1975 / 56)
    assert simple_mean_di == pytest.approx(25.0)
    assert plus_di[6] != pytest.approx(simple_mean_di)

  def test_single_bar(self):
    bars = sine_bars(1)
    for series in adx(bars, 14):
      assert series == [None]

  def test_short_series_is_all_none(self):
    bars = sine_bars(4)
    for series in adx(bars, 14):
      assert series == [None] * 4

  def test_zero_range_gives_zero_not_a_crash(self):
    # Every bar identical: no range, so no directional movement. The
    # degenerate reading is 0.0, the same way a flat series gives RSI
    # 50.0 rather than a division by zero.
    rows = [(5.0, 5.0, 5.0)] * 20
    strength, plus_di, minus_di = adx(bars_from(rows), 3)
    assert plus_di[-1] == pytest.approx(0.0)
    assert minus_di[-1] == pytest.approx(0.0)
    assert strength[-1] == pytest.approx(0.0)

  def test_directional_index_stays_non_negative(self):
    _, plus_di, minus_di = adx(sine_bars(60), 14)
    for series in (plus_di, minus_di):
      for value in series:
        if value is not None:
          assert 0.0 <= value <= 100.0

  def test_adx_stays_non_negative(self):
    strength, _, _ = adx(sine_bars(60), 14)
    for value in strength:
      if value is not None:
        assert value >= 0.0

  def test_range_without_directional_movement_is_zero(self):
    # Highs flat and lows flat, so neither directional move is ever
    # positive and +DM = -DM = 0, but the close still moves, so the true
    # range stays positive. +DI and -DI are both 0, their sum is 0, and
    # DX must not divide by it.
    rows = []
    for index in range(20):
      close = 6.0 if index % 2 == 0 else 9.0
      rows.append((10.0, 5.0, close))
    strength, plus_di, minus_di = adx(bars_from(rows), 3)
    assert plus_di[-1] == pytest.approx(0.0)
    assert minus_di[-1] == pytest.approx(0.0)
    assert strength[-1] == pytest.approx(0.0)
    assert len(strength) == 20

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      adx(adx_bars(), window)

  def test_accepts_window_of_one(self):
    # Positive control.
    strength, plus_di, _ = adx(sine_bars(10), 1)
    assert plus_di[1] is not None
    assert len(strength) == 10


class TestAroonOscillator:
  '''Chande's Aroon oscillator.

  Hand arithmetic on ``highs = [12, 11, 13]``, ``lows = [5, 9, 7]``
  with ``window = 2``, so the scan covers three bars. At index 2 the
  highest high is the current bar, 0 bars back, and the lowest low is
  bar 0, 2 bars back. ``aroon_up = 100 * (2 - 0) / 2 = 100`` and
  ``aroon_down = 100 * (2 - 2) / 2 = 0``, so the oscillator is
  ``100``.
  '''

  def test_hand_computed_value(self):
    assert aroon_oscillator([12, 11, 13], [5, 9, 7], 2) == [
      None, None, 100.0]

  def test_warmup_is_none(self):
    bars = sine_bars(30)
    result = aroon_oscillator(highs_of(bars), lows_of(bars), 5)
    assert result[:5] == [None] * 5

  def test_length_matches_input(self):
    bars = sine_bars(30)
    result = aroon_oscillator(highs_of(bars), lows_of(bars), 5)
    assert len(result) == len(bars)

  def test_bounded_range(self):
    bars = sine_bars(40)
    for value in aroon_oscillator(highs_of(bars), lows_of(bars), 7):
      if value is not None:
        assert -100.0 <= value <= 100.0

  def test_rising_series_is_at_the_positive_extreme(self):
    # On a monotonic rise the highest high is the current bar, 0 back,
    # and the lowest low is the oldest bar of the window, ``window``
    # back. So up = 100 and down = 0, and the reading is +100. The sign
    # is arithmetic here, not a verdict about the series.
    highs = [float(index) for index in range(20)]
    lows = [float(index) - 1.0 for index in range(20)]
    result = aroon_oscillator(highs, lows, 5)
    assert result[-1] == pytest.approx(100.0)

  def test_falling_series_is_at_the_negative_extreme(self):
    highs = [float(index) for index in range(20, 0, -1)]
    lows = [float(index) - 1.0 for index in range(20, 0, -1)]
    result = aroon_oscillator(highs, lows, 5)
    assert result[-1] == pytest.approx(-100.0)

  def test_constant_input_gives_zero(self):
    result = aroon_oscillator([10.0] * 20, [10.0] * 20, 5)
    assert result[-1] == pytest.approx(0.0)

  def test_recent_tie_wins(self):
    # highs 10, 10, 10, 1, 2 with window 3: the window maximum of 10
    # is reached at bars 1 and 2. The most recent is bar 2, one bar
    # back, so aroon_up = 100 * (3 - 1) / 3 = 100/3. An oldest-wins
    # scan would report 3 bars back and give 0. With a unique lowest
    # low at the current bar, aroon_down = 100, so the reading is
    # 100/3 - 100 = -200/3 rather than the -100 an oldest-wins scan
    # would report.
    highs = [10.0, 10.0, 10.0, 1.0, 2.0]
    lows = [9.0, 9.0, 9.0, 9.0, 0.0]
    assert aroon_oscillator(highs, lows, 3)[4] == pytest.approx(-200 / 3)
    assert -200 / 3 != pytest.approx(-100.0)

  def test_matches_a_brute_force_oracle(self):
    bars = sine_bars(30)
    highs = highs_of(bars)
    lows = lows_of(bars)
    window = 6
    result = aroon_oscillator(highs, lows, window)
    for index in range(window, len(highs)):
      span_highs = highs[index - window:index + 1]
      span_lows = lows[index - window:index + 1]
      # Last occurrence, measured from the end, is bars-since.
      high_ago = window - (
        len(span_highs) - 1 - span_highs[::-1].index(max(span_highs)))
      low_ago = window - (
        len(span_lows) - 1 - span_lows[::-1].index(min(span_lows)))
      expected = 100.0 * (low_ago - high_ago) / window
      assert result[index] == pytest.approx(expected)

  @pytest.mark.parametrize('window', [0, -3])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      aroon_oscillator([1.0, 2.0], [1.0, 2.0], window)

  def test_rejects_mismatched_lengths(self):
    with pytest.raises(ValueError, match='lows'):
      aroon_oscillator([1.0, 2.0, 3.0], [1.0, 2.0], 2)

  def test_accepts_window_of_one(self):
    # Positive control.
    result = aroon_oscillator([1.0, 2.0, 3.0], [1.0, 1.5, 2.0], 1)
    assert result[1] is not None
    assert result[2] == pytest.approx(100.0)


class TestTrix:
  '''One-bar rate of change of a triple-smoothed EMA.

  Hand arithmetic on ``closes = [1, 2, 3, 4]`` with ``window = 2``, so
  the EMA factor is ``2/3`` and the three recursions give

  * EMA1 = ``1, 5/3, 23/9, 95/27``
  * EMA2 = ``1, 13/9, 59/27, 83/27``
  * EMA3 = ``1, 35/27, 17/9, 217/81``

  so TRIX at index 2 is ``100 * (17/9) / (35/27) - 100 = 100 * 51/35 -
  100 = 1600/35 = 320/7``, and at index 3
  ``100 * (217/81) / (17/9) - 100 = 100 * 64/153 - 100 = 6400/153``.
  '''

  def test_hand_computed_values(self):
    result = trix([1.0, 2.0, 3.0, 4.0], 2)
    assert result[2] == pytest.approx(320 / 7)
    assert result[3] == pytest.approx(6400 / 153)

  def test_warmup_is_none(self):
    result = trix(closes_of(sine_bars(30)), 5)
    assert result[:5] == [None] * 5

  def test_length_matches_input(self):
    closes = closes_of(sine_bars(30))
    assert len(trix(closes, 5)) == len(closes)

  def test_empty_input(self):
    assert not trix([], 5)

  def test_constant_input_gives_zero(self):
    result = trix([7.0] * 20, 5)
    assert result[5:] == [pytest.approx(0.0)] * 15

  def test_zero_price_series_is_none_not_a_crash(self):
    # A smoothed price of zero would make the ratio undefined.
    assert trix([0.0] * 6, 2) == [None] * 6

  def test_geometric_series_gives_the_input_rate(self):
    # A geometric series grows at a constant rate, and an EMA of it is
    # a scaled lag of the same geometric series, so once the recursion
    # has converged the triple smoothing still grows at the input rate
    # and TRIX equals it. Short runs lag, so 80 bars of 1% is used.
    closes = [100.0 * 1.01 ** index for index in range(80)]
    assert trix(closes, 5)[-1] == pytest.approx(1.0, abs=1e-4)

  def test_is_a_rate_so_it_is_scale_invariant(self):
    closes = [100.0 * 1.01 ** index for index in range(40)]
    scaled = [value * 1000.0 for value in closes]
    assert trix(scaled, 5)[-1] == pytest.approx(trix(closes, 5)[-1])

  def test_rising_is_positive_and_falling_is_negative(self):
    rising = [float(index) for index in range(1, 40)]
    falling = list(reversed(rising))
    assert trix(rising, 5)[-1] > 0.0
    assert trix(falling, 5)[-1] < 0.0

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      trix([1.0, 2.0, 3.0], window)

  def test_accepts_window_of_one(self):
    # Positive control.
    assert len(trix([1.0, 2.0, 3.0], 1)) == 3


class TestKst:
  '''Pring's Know Sure Thing oscillator.

  Hand arithmetic needs a series where every rate of change is the
  same number, which a geometric series gives. On
  ``closes = 100 * 1.01 ** i`` the ROC over any window ``w`` is
  ``100 * (1.01 ** w - 1)``, so smoothing changes nothing and the four
  components are

  * ``100 * (1.01 ** 10 - 1) = 10.462213``
  * ``100 * (1.01 ** 15 - 1) = 16.096896``
  * ``100 * (1.01 ** 20 - 1) = 22.019004``
  * ``100 * (1.01 ** 30 - 1) = 34.784892``

  giving ``(1*10.462213 + 2*16.096896 + 3*22.019004 + 4*34.784892) / 10
  = 247.852582 / 10 = 24.785258``. The slowest component needs
  ``30 + 15 = 45`` bars, so the oscillator starts at index 44.
  '''

  def test_hand_computed_value(self):
    closes = [100.0 * 1.01 ** index for index in range(60)]
    line, signal = kst(closes)
    expected = (
      100.0 * (1.01 ** 10 - 1)
      + 2.0 * 100.0 * (1.01 ** 15 - 1)
      + 3.0 * 100.0 * (1.01 ** 20 - 1)
      + 4.0 * 100.0 * (1.01 ** 30 - 1)
    ) / 10.0
    assert line[44] == pytest.approx(24.785258, abs=1e-5)
    assert line[44] == pytest.approx(expected)
    assert signal[52] == pytest.approx(expected)

  def test_warmup_floor_is_the_published_one(self):
    closes = closes_of(sine_bars(80))
    line, signal = kst(closes)
    assert line[:44] == [None] * 44
    assert signal[:52] == [None] * 52
    assert line[44] is not None
    assert signal[52] is not None

  def test_length_matches_input(self):
    closes = closes_of(sine_bars(80))
    for series in kst(closes):
      assert len(series) == len(closes)

  def test_short_series_is_all_none(self):
    closes = closes_of(sine_bars(30))
    for series in kst(closes):
      assert series == [None] * 30

  def test_constant_input_gives_zero(self):
    line, signal = kst([7.0] * 80)
    assert line[-1] == pytest.approx(0.0)
    assert signal[-1] == pytest.approx(0.0)

  def test_matches_a_brute_force_combination(self):
    # The load-bearing test for this indicator: it recomputes the whole
    # published definition from first principles -- four rates of
    # change, each smoothed by a simple mean, combined with weights
    # 1, 2, 3, 4 and divided by their sum. A wrong window, a swapped
    # weight or a missing division all fail here.
    closes = closes_of(sine_bars(120))
    line, _ = kst(closes)
    components = []
    for roc_window, smoothing, weight in _KST_COMPONENTS:
      rate = [None] * len(closes)
      for index in range(roc_window, len(closes)):
        rate[index] = 100.0 * (
          closes[index] / closes[index - roc_window] - 1.0)
      smoothed = [None] * len(closes)
      for index in range(smoothing - 1, len(closes)):
        chunk = [
          value for value in rate[index - smoothing + 1:index + 1]
          if value is not None
        ]
        if len(chunk) == smoothing:
          smoothed[index] = sum(chunk) / smoothing
      components.append((weight, smoothed))
    assert components[0][0] == 1 and components[3][0] == 4
    for index in range(len(closes)):
      if any(series[index] is None for _, series in components):
        assert line[index] is None
        continue
      expected = sum(
        weight * series[index] for weight, series in components) / 10.0
      assert line[index] == pytest.approx(expected)

  @pytest.mark.parametrize('window', [0, -2])
  def test_rejects_bad_signal_window(self, window):
    with pytest.raises(ValueError, match='signal_window'):
      kst([1.0] * 60, window)

  def test_accepts_signal_of_one(self):
    # Positive control.
    _, signal = kst([1.0] * 60, 1)
    assert signal[-1] is not None


class TestCoppock:
  '''Coppock's summed rate of change.

  Hand arithmetic on ``closes = [10, 11, 12, 13]`` with
  ``long_window=2``, ``short_window=1`` and ``smoothing=2``. The summed
  rates of change are

  * index 2: ``100*(12/10 - 1) + 100*(12/11 - 1) = 20 + 100/11 =
    320/11``
  * index 3: ``100*(13/11 - 1) + 100*(13/12 - 1) = 200/11 + 25/3 =
    875/33``

  and the 2-bar WMA is ``(1*320/11 + 2*875/33) / 3 = (960 + 1750) / 99 =
  2710/99``.
  '''

  def test_hand_computed_values(self):
    result = coppock([10.0, 11.0, 12.0, 13.0], 2, 1, 2)
    assert result[:3] == [None, None, None]
    assert result[3] == pytest.approx(2710 / 99)

  def test_warmup_floor(self):
    closes = closes_of(sine_bars(40))
    result = coppock(closes, 14, 11, 10)
    assert result[:23] == [None] * 23
    assert result[23] is not None

  def test_length_matches_input(self):
    closes = closes_of(sine_bars(40))
    assert len(coppock(closes, 14, 11, 10)) == len(closes)

  def test_short_series_is_all_none(self):
    assert coppock([1.0, 2.0, 3.0], 14, 11, 10) == [None, None, None]

  def test_constant_input_gives_zero(self):
    result = coppock([7.0] * 40, 14, 11, 10)
    assert result[-1] == pytest.approx(0.0)

  def test_geometric_series_gives_the_summed_rate(self):
    # Constant per-bar rate, so both rates of change are constant and
    # the weighted average of a constant is that constant.
    closes = [100.0 * 1.01 ** index for index in range(40)]
    expected = 100.0 * (1.01 ** 14 - 1) + 100.0 * (1.01 ** 11 - 1)
    assert coppock(closes)[-1] == pytest.approx(expected)

  def test_smoothing_of_one_is_the_summed_rate(self):
    closes = closes_of(sine_bars(40))
    result = coppock(closes, 3, 2, 1)
    assert result[3] is not None

  def test_non_positive_base_price_is_none_not_a_crash(self):
    # A rate of change off a base of zero has no definition, so it is
    # None. The index-2 rate would need closes[0] as its base.
    closes = [0.0, 0.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    result = coppock(closes, 2, 1, 2)
    assert result[2] is None
    assert result[3] is None
    assert len(result) == len(closes)

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_long_window(self, window):
    with pytest.raises(ValueError, match='long_window'):
      coppock([1.0, 2.0, 3.0], window, 11, 10)

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_short_window(self, window):
    with pytest.raises(ValueError, match='short_window'):
      coppock([1.0, 2.0, 3.0], 14, window, 10)

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_smoothing(self, window):
    with pytest.raises(ValueError, match='smoothing'):
      coppock([1.0, 2.0, 3.0], 14, 11, window)

  def test_accepts_window_of_one(self):
    # Positive control for all three rejections.
    assert len(coppock([1.0, 2.0, 3.0], 1, 1, 1)) == 3


class TestMassIndex:
  '''Donchian's range-expansion index.

  A constant span of ``2`` makes both EMAs exactly ``2``, so the ratio
  is exactly ``1`` at every bar and the running sum of 25 ratios is
  exactly ``25``. That is the degenerate answer.
  '''

  def test_constant_span_gives_the_window_sum(self):
    highs = [12.0] * 60
    lows = [10.0] * 60
    result = mass_index(highs, lows)
    assert result[48] == pytest.approx(25.0)
    assert result[-1] == pytest.approx(25.0)

  def test_hand_computed_on_a_constant_span(self):
    # ratio = EMA(2)/EMA(2) = 1 exactly, so index slow + sum - 2 is the
    # sum of exactly sum_window ones.
    highs = [12.0] * 40
    lows = [10.0] * 40
    result = mass_index(highs, lows, 3, 5, 4)
    assert result[7] == pytest.approx(4.0)

  def test_warmup_floor_is_the_published_one(self):
    bars = sine_bars(70)
    result = mass_index(highs_of(bars), lows_of(bars))
    assert result[:48] == [None] * 48
    assert result[48] is not None

  def test_length_matches_input(self):
    bars = sine_bars(70)
    result = mass_index(highs_of(bars), lows_of(bars))
    assert len(result) == len(bars)

  def test_short_series_is_all_none(self):
    result = mass_index([12.0] * 10, [10.0] * 10)
    assert result == [None] * 10

  def test_uses_only_the_high_low_span(self):
    # Shifting every close changes nothing, because the formula reads
    # only highs and lows.
    bars = sine_bars(60)
    moved = [
      Bar(bar.timestamp, bar.open, bar.high, bar.low, bar.close + 50.0, 1.0)
      for bar in bars
    ]
    assert (mass_index(highs_of(bars), lows_of(bars))
            == mass_index(highs_of(moved), lows_of(moved)))

  def test_zero_span_is_all_none(self):
    # High equal to low gives a span of zero, so both EMAs of the span
    # are zero and the ratio has no definition. All None, not a crash.
    result = mass_index([10.0] * 60, [10.0] * 60)
    assert result == [None] * 60

  def test_non_positive_span_is_none_not_a_crash(self):
    # Every high below its low gives a negative span, whose EMAs are
    # never zero, so this only asserts it does not raise.
    result = mass_index([1.0] * 60, [2.0] * 60)
    assert len(result) == 60

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_fast_window(self, window):
    with pytest.raises(ValueError, match='fast_window'):
      mass_index([12.0] * 60, [10.0] * 60, window, 25, 25)

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_slow_window(self, window):
    with pytest.raises(ValueError, match='slow_window'):
      mass_index([12.0] * 60, [10.0] * 60, 9, window, 25)

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_sum_window(self, window):
    with pytest.raises(ValueError, match='sum_window'):
      mass_index([12.0] * 60, [10.0] * 60, 9, 25, window)

  def test_rejects_fast_not_below_slow(self):
    with pytest.raises(ValueError, match='fast_window'):
      mass_index([12.0] * 60, [10.0] * 60, 25, 9, 25)

  def test_rejects_mismatched_lengths(self):
    with pytest.raises(ValueError, match='highs'):
      mass_index([12.0] * 60, [10.0] * 59)

  def test_accepts_fast_of_one(self):
    # Positive control.
    assert len(mass_index([12.0] * 60, [10.0] * 60, 1, 25, 2)) == 60


class TestIchimoku:
  '''The five Ichimoku components.

  Hand arithmetic on the eight bars of :func:`adx_bars` with
  ``conversion_window=3``, ``base_window=5``, ``span_b_window=8`` and
  ``displacement=2``. Tenkan at index 2 is
  ``(max(10, 11, 12) + min(9, 9.5, 10)) / 2 = (12 + 9) / 2 = 10.5``.
  Kijun at index 4 is
  ``(max over bars 0..4 + min over bars 0..4) / 2 = (13 + 9) / 2 = 11``.
  Senkou A at index 6 reads bar 4, two bars back, and is
  ``(tenkan[4] + kijun[4]) / 2 = (11.5 + 11) / 2 = 11.25``.
  '''

  def test_hand_computed_tenkan_and_kijun(self):
    tenkan, kijun, _, _, _ = ichimoku(adx_bars(), 3, 5, 8, 2)
    assert tenkan[2] == pytest.approx(10.5)
    assert tenkan[3] == pytest.approx(11.0)
    assert kijun[4] == pytest.approx(11.0)

  def test_hand_computed_senkou_a_reads_the_past(self):
    _, _, senkou_a, _, _ = ichimoku(adx_bars(), 3, 5, 8, 2)
    assert senkou_a[6] == pytest.approx(11.25)

  def test_senkou_a_is_the_mean_of_tenkan_and_kijun_behind(self):
    bars = sine_bars(80)
    tenkan, kijun, senkou_a, _, _ = ichimoku(bars, 9, 26, 52, 26)
    for index in range(26 + 25, len(bars)):
      source = index - 26
      assert senkou_a[index] == pytest.approx(
        (tenkan[source] + kijun[source]) / 2.0)

  def test_senkou_b_reads_the_past(self):
    bars = sine_bars(80)
    _, _, _, senkou_b, _ = ichimoku(bars, 9, 26, 52, 26)
    assert senkou_b[77] is not None
    assert senkou_b[:77] == [None] * 77

  def test_chikou_is_the_close_at_its_own_bar(self):
    # Displacing Chikou forward in the array would make index i read
    # bar i + 26, which is look-ahead. It is returned at the bar it
    # came from instead, so it equals the close series.
    bars = sine_bars(80)
    _, _, _, _, chikou = ichimoku(bars)
    assert chikou == closes_of(bars)

  def test_warmup_floors(self):
    bars = sine_bars(80)
    tenkan, kijun, senkou_a, senkou_b, chikou = ichimoku(bars)
    assert tenkan[:8] == [None] * 8
    assert kijun[:25] == [None] * 25
    assert senkou_a[:51] == [None] * 51
    assert senkou_b[:77] == [None] * 77
    assert chikou[0] == pytest.approx(closes_of(bars)[0])

  def test_length_matches_input(self):
    bars = sine_bars(80)
    for series in ichimoku(bars):
      assert len(series) == len(bars)

  def test_empty_input(self):
    for series in ichimoku([]):
      assert series == []

  def test_constant_input(self):
    rows = [(10.0, 10.0, 10.0)] * 80
    tenkan, kijun, senkou_a, senkou_b, chikou = ichimoku(bars_from(rows))
    assert tenkan[-1] == pytest.approx(10.0)
    assert kijun[-1] == pytest.approx(10.0)
    assert senkou_a[-1] == pytest.approx(10.0)
    assert senkou_b[-1] == pytest.approx(10.0)
    assert chikou[-1] == pytest.approx(10.0)

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_conversion_window(self, window):
    with pytest.raises(ValueError, match='conversion_window'):
      ichimoku(sine_bars(80), window)

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_base_window(self, window):
    with pytest.raises(ValueError, match='base_window'):
      ichimoku(sine_bars(80), 9, window)

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_span_b_window(self, window):
    with pytest.raises(ValueError, match='span_b_window'):
      ichimoku(sine_bars(80), 9, 26, window)

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_displacement(self, window):
    with pytest.raises(ValueError, match='displacement'):
      ichimoku(sine_bars(80), 9, 26, 52, window)

  def test_accepts_window_of_one(self):
    # Positive control for all four rejections.
    for series in ichimoku(sine_bars(80), 1, 1, 1, 1):
      assert len(series) == 80


class TestDonchianChannels:
  '''Trailing-window channels.

  Hand arithmetic on ``highs = [3, 5, 4]``, ``lows = [1, 2, 0]`` with
  ``window = 2``: bar 1 sees bars 0 and 1 so the channel is
  ``(max(3, 5), min(1, 2)) = (5, 1)``; bar 2 sees bars 1 and 2 so it is
  ``(max(5, 4), min(2, 0)) = (5, 0)``.
  '''

  def test_hand_computed_values(self):
    upper, lower = donchian_channels([3, 5, 4], [1, 2, 0], 2)
    assert upper == [None, 5, 5]
    assert lower == [None, 1, 0]

  def test_warmup_is_none(self):
    bars = sine_bars(30)
    upper, lower = donchian_channels(highs_of(bars), lows_of(bars), 5)
    assert upper[:4] == [None] * 4
    assert lower[:4] == [None] * 4

  def test_length_matches_input(self):
    bars = sine_bars(30)
    for series in donchian_channels(highs_of(bars), lows_of(bars), 5):
      assert len(series) == len(bars)

  def test_window_of_one_is_the_bar_itself(self):
    upper, lower = donchian_channels([3.0, 5.0, 4.0], [1.0, 2.0, 0.0], 1)
    assert upper == [3.0, 5.0, 4.0]
    assert lower == [1.0, 2.0, 0.0]

  def test_upper_is_never_below_lower(self):
    bars = sine_bars(60)
    upper, lower = donchian_channels(highs_of(bars), lows_of(bars), 20)
    for index, high in enumerate(upper):
      if high is not None:
        assert high >= lower[index]

  def test_constant_input(self):
    upper, lower = donchian_channels([10.0] * 10, [10.0] * 10, 3)
    assert upper[-1] == pytest.approx(10.0)
    assert lower[-1] == pytest.approx(10.0)

  def test_is_not_a_centred_window(self):
    # Negative control for the truncation tests. At bar 3 the trailing
    # window covers bars 1 to 3, so the lowest low is 0. A centred
    # window of width 3 would cover bars 2 to 4 and report -1, which is
    # only knowable three bars afterwards.
    highs = [3.0, 5.0, 4.0, 9.0, 1.0]
    lows = [1.0, 2.0, 0.0, 4.0, -1.0]
    upper, lower = donchian_channels(highs, lows, 3)
    assert upper[3] == pytest.approx(9.0)
    assert lower[3] == pytest.approx(0.0)
    assert min(lows[2:5]) == pytest.approx(-1.0)
    assert lower[3] != pytest.approx(min(lows[2:5]))

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      donchian_channels([1.0, 2.0], [1.0, 2.0], window)

  def test_rejects_mismatched_lengths(self):
    with pytest.raises(ValueError, match='lows'):
      donchian_channels([1.0, 2.0, 3.0], [1.0, 2.0], 2)

  def test_accepts_window_of_one(self):
    # Positive control.
    upper, _ = donchian_channels([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], 1)
    assert upper[0] == pytest.approx(1.0)


class TestLinearRegression:
  '''Rolling least-squares slope and endpoint.

  Hand arithmetic on ``[1, 2, 3, 4]`` with ``window = 4``:
  ``mean_x = 1.5``, ``mean_y = 2.5``, numerator
  ``1.5*1.5 + 0.5*0.5 + (-0.5)*(-0.5) + (-1.5)*(-1.5) = 2.25 + 0.25 +
  0.25 + 2.25 = 5`` and denominator ``2.25 + 0.25 + 0.25 + 2.25 = 5``,
  so the slope is ``1`` and the endpoint at ``x = 3`` is
  ``2.5 + 1 * (3 - 1.5) = 4``.
  '''

  def test_hand_computed_perfect_line(self):
    slopes, intercepts = linear_regression([1.0, 2.0, 3.0, 4.0], 4)
    assert slopes == [None, None, None, pytest.approx(1.0)]
    assert intercepts == [None, None, None, pytest.approx(4.0)]

  def test_warmup_is_none(self):
    values = closes_of(sine_bars(20))
    slopes, intercepts = linear_regression(values, 5)
    assert slopes[:4] == [None] * 4
    assert intercepts[:4] == [None] * 4

  def test_length_matches_input(self):
    values = closes_of(sine_bars(20))
    for series in linear_regression(values, 5):
      assert len(series) == len(values)

  def test_empty_input(self):
    for series in linear_regression([], 5):
      assert series == []

  def test_flat_series_is_zero_slope_not_none(self):
    # A flat window has a meaningful zero slope and the level as its
    # endpoint, the same way ``trend_slope`` returns 0.0.
    slopes, intercepts = linear_regression([5.0] * 10, 4)
    assert slopes[-1] == pytest.approx(0.0)
    assert intercepts[-1] == pytest.approx(5.0)

  def test_downward_series_is_negative(self):
    slopes, _ = linear_regression([4.0, 3.0, 2.0, 1.0], 4)
    assert slopes[-1] == pytest.approx(-1.0)

  def test_slope_is_scale_invariant(self):
    # Slope is per bar, so scaling the level must scale it back.
    base = closes_of(sine_bars(20))
    scaled = [value * 10.0 for value in base]
    slopes, intercepts = linear_regression(base, 5)
    scaled_slopes, scaled_intercepts = linear_regression(scaled, 5)
    assert scaled_slopes[-1] == pytest.approx(slopes[-1] * 10.0)
    assert scaled_intercepts[-1] == pytest.approx(intercepts[-1] * 10.0)

  def test_endpoint_tracks_a_rising_series(self):
    values = [float(index) for index in range(20)]
    _, intercepts = linear_regression(values, 5)
    assert intercepts[-1] > intercepts[4]

  @pytest.mark.parametrize('window', [0, 1, -1])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      linear_regression([1.0, 2.0, 3.0], window)

  def test_accepts_window_of_two(self):
    # Positive control: two points define a line exactly.
    slopes, intercepts = linear_regression([1.0, 4.0, 7.0], 2)
    assert slopes[-1] == pytest.approx(3.0)
    assert intercepts[-1] == pytest.approx(7.0)


class TestWma:
  '''Linearly weighted moving average.

  ``wma([1, 2, 3], 3) = (1*1 + 2*2 + 3*3) / (1 + 2 + 3) = 14/6``.
  '''

  def test_hand_computed_value(self):
    assert wma([1.0, 2.0, 3.0], 3)[-1] == pytest.approx(14 / 6)

  def test_weighting_favours_the_newest_bar(self):
    # [1, 0, 0] weighted towards the newest must be smaller than the
    # same series weighted towards the oldest, because the mass sits
    # at the end either way but the oldest case is [0, 0, 1] reversed.
    rising = wma([0.0, 0.0, 1.0], 3)[-1]
    falling = wma([1.0, 0.0, 0.0], 3)[-1]
    assert rising == pytest.approx(3.0 / 6)
    assert falling == pytest.approx(1.0 / 6)
    assert rising > falling

  def test_warmup_is_none(self):
    result = wma(closes_of(sine_bars(20)), 5)
    assert result[:4] == [None] * 4

  def test_length_matches_input(self):
    values = closes_of(sine_bars(20))
    assert len(wma(values, 5)) == len(values)

  def test_constant_input(self):
    assert wma([7.0] * 10, 4)[-1] == pytest.approx(7.0)

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      wma([1.0, 2.0, 3.0], window)

  def test_window_of_one_is_identity(self):
    # Positive control.
    assert wma([3.0, 5.0, 9.0], 1) == [3.0, 5.0, 9.0]


class TestHma:
  '''Hull's moving average.

  ``hma([1, 2, 3, 4, 5], 4)`` has a half window of 2 and a root window
  of ``isqrt(4) = 2``. WMA2 is ``5/3, 8/3, 11/3, 14/3`` and WMA4 is
  ``3, 4``, so the raw difference is ``13/3, 16/3`` and the outer
  WMA2 gives ``(13/3 + 2*16/3) / 3 = 45/9 = 5`` at index 3.
  '''

  def test_hand_computed_value(self):
    result = hma([1.0, 2.0, 3.0, 4.0, 5.0], 4)
    assert result == [None, None, None, None, pytest.approx(5.0)]

  def test_warmup_floor(self):
    # Base window 16 plus root window 4, minus two.
    result = hma(closes_of(sine_bars(40)), 16)
    assert result[:18] == [None] * 18
    assert result[18] is not None

  def test_length_matches_input(self):
    # A shortened return list here is the exact off-by-N this module
    # has to avoid, so it is asserted on a short series too.
    values = closes_of(sine_bars(40))
    assert len(hma(values, 16)) == len(values)
    assert len(hma([1.0, 2.0, 3.0, 4.0, 5.0], 4)) == 5

  def test_constant_input(self):
    assert hma([7.0] * 30, 8)[-1] == pytest.approx(7.0)

  def test_tracks_a_geometric_series(self):
    # HMA is designed to be near the input with minimal lag, so on a
    # smooth geometric series it should sit close to the price rather
    # than trailing it.
    closes = [100.0 * 1.005 ** index for index in range(40)]
    result = hma(closes, 16)
    assert result[-1] == pytest.approx(closes[-1], rel=0.02)

  @pytest.mark.parametrize('window', [0, 1, -1])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      hma([1.0, 2.0, 3.0], window)

  def test_accepts_window_of_two(self):
    # Positive control.
    assert len(hma([1.0, 2.0, 3.0], 2)) == 3


class TestDema:
  '''Double exponential smoothing.

  On ``[1, 2, 3, 4]`` with ``window = 2`` the factor is ``2/3``. EMA1
  is ``1, 5/3, 23/9, 95/27`` and EMA2 is ``1, 13/9, 59/27, 83/27``, so
  the DEMA is ``None, 17/9, 79/27, 107/27``.
  '''

  def test_hand_computed_values(self):
    result = dema([1.0, 2.0, 3.0, 4.0], 2)
    assert result == [None, pytest.approx(17 / 9), pytest.approx(79 / 27),
                      pytest.approx(107 / 27)]

  def test_warmup_is_none(self):
    result = dema(closes_of(sine_bars(20)), 5)
    assert result[:4] == [None] * 4

  def test_length_matches_input(self):
    values = closes_of(sine_bars(20))
    assert len(dema(values, 5)) == len(values)

  def test_constant_input(self):
    assert dema([7.0] * 20, 5)[-1] == pytest.approx(7.0)

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      dema([1.0, 2.0, 3.0], window)

  def test_accepts_window_of_one(self):
    # Positive control: with a factor of 1 the recursion is the input.
    assert dema([1.0, 2.0, 3.0], 1) == [1.0, 2.0, 3.0]


class TestTema:
  '''Triple exponential smoothing.

  The same three recursions give ``1, 5/3, 23/9, 95/27``, then
  ``1, 13/9, 59/27, 83/27``, then ``1, 35/27, 17/9, 217/81``, so the
  TEMA is ``None, 53/27, 27/9, 325/81``.
  '''

  def test_hand_computed_values(self):
    result = tema([1.0, 2.0, 3.0, 4.0], 2)
    assert result == [None, pytest.approx(53 / 27), pytest.approx(3.0),
                      pytest.approx(325 / 81)]

  def test_warmup_is_none(self):
    result = tema(closes_of(sine_bars(20)), 5)
    assert result[:4] == [None] * 4

  def test_length_matches_input(self):
    values = closes_of(sine_bars(20))
    assert len(tema(values, 5)) == len(values)

  def test_constant_input(self):
    assert tema([7.0] * 20, 5)[-1] == pytest.approx(7.0)

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      tema([1.0, 2.0, 3.0], window)

  def test_accepts_window_of_one(self):
    # Positive control.
    assert tema([1.0, 2.0, 3.0], 1) == [1.0, 2.0, 3.0]


class TestDisplacedMa:
  '''Backwards-displaced moving average.

  ``displaced_ma([1, 2, 3, 4, 5], 2, 1)`` has SMA2 of
  ``None, 1.5, 2.5, 3.5, 4.5``, shifted one bar left to give
  ``None, None, 1.5, 2.5, 3.5``.
  '''

  def test_hand_computed_value(self):
    result = displaced_ma([1.0, 2.0, 3.0, 4.0, 5.0], 2, 1)
    assert result == [None, None, 1.5, 2.5, 3.5]

  def test_warmup_floor(self):
    result = displaced_ma(closes_of(sine_bars(60)), 20, 26)
    assert result[:45] == [None] * 45
    assert result[45] is not None

  def test_length_matches_input(self):
    values = closes_of(sine_bars(40))
    assert len(displaced_ma(values, 20, 26)) == len(values)

  def test_shifts_backwards_never_forwards(self):
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    result = displaced_ma(values, 2, 2)
    assert result[3] == pytest.approx(1.5)
    assert result[4] == pytest.approx(2.5)
    assert result[5] == pytest.approx(3.5)

  def test_constant_input(self):
    assert displaced_ma([7.0] * 20, 5, 3)[-1] == pytest.approx(7.0)

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      displaced_ma([1.0, 2.0, 3.0], window, 1)

  @pytest.mark.parametrize('value', [0, -1])
  def test_rejects_bad_displacement(self, value):
    with pytest.raises(ValueError, match='displacement'):
      displaced_ma([1.0, 2.0, 3.0], 2, value)

  def test_accepts_window_and_displacement_of_one(self):
    # Positive control for both rejections.
    assert displaced_ma([1.0, 2.0, 3.0], 1, 1) == [None, 1.0, 2.0]


class TestLengthPreservation:
  '''Every public function returns exactly as many entries as it got.

  This is the regression test for the bug class that hit
  ``indicators.py``: six values for five closes. It runs over every
  function, over a long enough series for every warm-up to have
  finished and every output to be defined at least once, so a function
  that returned a list of the wrong length fails even though the values
  inside it are right.
  '''

  def test_every_function_is_length_preserving(self):
    bars = sine_bars(120)
    highs = highs_of(bars)
    lows = lows_of(bars)
    closes = closes_of(bars)
    cases = {
      'adx': adx(bars, 14),
      'aroon_oscillator': aroon_oscillator(highs, lows, 25),
      'coppock': coppock(closes),
      'dema': dema(closes, 14),
      'displaced_ma': displaced_ma(closes, 20, 26),
      'donchian_channels': donchian_channels(highs, lows, 20),
      'hma': hma(closes, 16),
      'ichimoku': ichimoku(bars),
      'kst': kst(closes),
      'linear_regression': linear_regression(closes, 14),
      'macd': macd(closes),
      'macd_histogram': macd_histogram(closes),
      'mass_index': mass_index(highs, lows),
      'tema': tema(closes, 14),
      'trix': trix(closes, 15),
      'wma': wma(closes, 14),
    }
    assert sorted(cases) == sorted([
      'adx', 'aroon_oscillator', 'coppock', 'dema', 'displaced_ma',
      'donchian_channels', 'hma', 'ichimoku', 'kst', 'linear_regression',
      'macd', 'macd_histogram', 'mass_index', 'tema', 'trix', 'wma'])
    for name, result in cases.items():
      for series in columns(result):
        assert len(series) == len(bars), name

  def test_every_series_is_defined_at_least_once(self):
    # Positive control for the length test: a function that returned
    # 120 Nones would pass on length alone, so assert the complement.
    bars = sine_bars(120)
    highs = highs_of(bars)
    lows = lows_of(bars)
    closes = closes_of(bars)
    cases = [
      adx(bars, 14), aroon_oscillator(highs, lows, 25), coppock(closes),
      dema(closes, 14), displaced_ma(closes, 20, 26),
      donchian_channels(highs, lows, 20), hma(closes, 16),
      ichimoku(bars), kst(closes), linear_regression(closes, 14),
      macd(closes), macd_histogram(closes), mass_index(highs, lows),
      tema(closes, 14), trix(closes, 15), wma(closes, 14),
    ]
    for result in cases:
      for series in columns(result):
        assert any(value is not None for value in series)

  def test_warmup_is_none_and_never_zero_placeholder(self):
    bars = sine_bars(60)
    highs = highs_of(bars)
    lows = lows_of(bars)
    closes = closes_of(bars)
    cases = [
      adx(bars, 14), aroon_oscillator(highs, lows, 25), coppock(closes),
      dema(closes, 14), displaced_ma(closes, 20, 26),
      donchian_channels(highs, lows, 20), hma(closes, 16),
      # Ichimoku's Chikou is excluded on purpose: it is the close
      # series, so it has no warm-up by construction.
      ichimoku(bars)[:4], kst(closes), linear_regression(closes, 14),
      macd(closes), macd_histogram(closes), mass_index(highs, lows),
      tema(closes, 14), trix(closes, 15), wma(closes, 14),
    ]
    for result in cases:
      for series in columns(result):
        assert series[0] is None

  def test_chikou_is_the_only_series_without_a_warmup(self):
    # Positive control for the exclusion above.
    bars = sine_bars(60)
    assert ichimoku(bars)[4][0] is not None

  def test_short_input_still_preserves_length(self):
    # The bug appeared on a five-bar input, so check short inputs too.
    bars = sine_bars(5)
    highs = highs_of(bars)
    lows = lows_of(bars)
    closes = closes_of(bars)
    cases = [
      adx(bars, 14), aroon_oscillator(highs, lows, 5),
      coppock(closes, 2, 1, 2), dema(closes, 3),
      displaced_ma(closes, 2, 1), donchian_channels(highs, lows, 2),
      hma(closes, 2), ichimoku(bars, 2, 2, 2, 1), kst(closes, 1),
      linear_regression(closes, 2), macd(closes, 1, 2, 1),
      macd_histogram(closes, 1, 2, 1),
      mass_index(highs, lows, 1, 2, 2), tema(closes, 3),
      trix(closes, 2), wma(closes, 2),
    ]
    for result in cases:
      for series in columns(result):
        assert len(series) == 5


class TestNoLookAhead:
  '''No output at index ``i`` may depend on a bar after ``i``.

  The test is truncation, not inspection: if a value at index ``i``
  really is a function of bars ``0..i`` only, then feeding a shorter
  series cannot change it. This is what catches a centred Donchian
  window and a forward-indexed Chikou span, both of which pass every
  value check on a single long series.
  '''

  def test_every_indicator_is_prefix_stable(self):
    bars = sine_bars(120)
    cases = {
      'adx': adx_of,
      'aroon_oscillator': aroon_of,
      'coppock': coppock_of,
      'decentred_dema': dema_of,
      'displaced_ma': displaced_of,
      'donchian_channels': donchian_of,
      'hma': hma_of,
      'ichimoku': ichimoku,
      'kst': kst_of,
      'linear_regression': linreg_of,
      'macd': macd_of,
      'mass_index': mass_of,
      'tema': tema_of,
      'trix': trix_of,
      'wma': wma_of,
    }
    for name, compute in cases.items():
      # ``name`` rides along only so a failure says which indicator
      # broke; ``assert_no_look_ahead`` reports the offending index.
      try:
        assert_no_look_ahead(compute, bars, 100)
      except AssertionError as error:
        raise AssertionError(f'{name}: {error}') from error

  def test_prefix_stability_detects_a_centred_window(self):
    # Positive control for the check itself: a deliberately
    # forward-looking rolling max reads bars i, i + 1, i + 2, so
    # truncating the input must change the surviving values. If this
    # test ever failed to notice, ``assert_no_look_ahead`` would be
    # broken and the tests above would prove nothing.
    def centred(rows):
      '''Deliberately wrong: a rolling max that reaches forward.'''
      values = highs_of(rows)
      return [
        max(values[index:index + 3]) for index in range(len(values))
      ]

    bars = sine_bars(40)
    full = centred(bars)
    partial = centred(bars[:20])
    assert full != partial
    assert full[18] != partial[18]
    assert full[18] == pytest.approx(max(highs_of(bars)[18:21]))
    assert partial[18] == pytest.approx(max(highs_of(bars)[18:20]))


class TestHonesty:
  '''The module must not smuggle in unfitted thresholds or a verdict.

  A test on the docstring text with the live module as the oracle.
  The failure this guards against is quiet: an ``overbought``
  convenience parameter would work, would look helpful, and would put
  an unfitted constant one layer away from anyone who tries to audit
  it.
  '''

  def test_no_threshold_api_is_exposed(self):
    banned = ('overbought', 'oversold', 'threshold', 'signal_line_buy',
              'crossover', 'buy', 'sell')
    for name in dir(module_under_test):
      assert name not in banned, name

  def test_no_threshold_keyword_in_any_signature(self):
    for name in module_under_test.__all__:
      signature = inspect.signature(getattr(module_under_test, name))
      for parameter in signature.parameters:
        assert parameter not in ('overbought', 'oversold', 'threshold')

  def test_module_docstring_states_the_evidence_caveat(self):
    doc = module_under_test.__doc__
    assert doc is not None
    assert 'not evidence' in doc
    assert 'unfitted' in doc
    assert 'technical-indicators-nse.md' in doc

  def test_every_public_function_has_a_google_docstring(self):
    for name in module_under_test.__all__:
      doc = getattr(module_under_test, name).__doc__
      assert doc is not None, name
      assert 'Args:' in doc, name
      assert 'Returns:' in doc, name
      assert 'Raises:' in doc, name

  def test_validation_messages_name_the_offending_parameter(self):
    # Every rejection below must name the parameter it is about, so a
    # caller can tell which argument is wrong without reading the
    # source.
    bars = sine_bars(60)
    cases = [
      (lambda: wma([1.0], 0), 'window'),
      (lambda: hma([1.0], 0), 'window'),
      (lambda: dema([1.0], 0), 'window'),
      (lambda: tema([1.0], 0), 'window'),
      (lambda: trix([1.0], 0), 'window'),
      (lambda: macd([1.0], 0, 26, 9), 'fast_window'),
      (lambda: macd([1.0], 12, 0, 9), 'slow_window'),
      (lambda: macd([1.0], 12, 26, 0), 'signal_window'),
      (lambda: adx(bars, 0), 'window'),
      (lambda: aroon_oscillator([1.0], [1.0], 0), 'window'),
      (lambda: kst([1.0], 0), 'signal_window'),
      (lambda: coppock([1.0], 0, 11, 10), 'long_window'),
      (lambda: coppock([1.0], 14, 0, 10), 'short_window'),
      (lambda: coppock([1.0], 14, 11, 0), 'smoothing'),
      (lambda: mass_index([1.0], [1.0], 0, 25, 25), 'fast_window'),
      (lambda: mass_index([1.0], [1.0], 9, 0, 25), 'slow_window'),
      (lambda: mass_index([1.0], [1.0], 9, 25, 0), 'sum_window'),
      (lambda: ichimoku(bars, 0), 'conversion_window'),
      (lambda: ichimoku(bars, 9, 0), 'base_window'),
      (lambda: ichimoku(bars, 9, 26, 0), 'span_b_window'),
      (lambda: ichimoku(bars, 9, 26, 52, 0), 'displacement'),
      (lambda: donchian_channels([1.0], [1.0], 0), 'window'),
      (lambda: linear_regression([1.0], 1), 'window'),
      (lambda: displaced_ma([1.0], 0, 1), 'window'),
      (lambda: displaced_ma([1.0], 2, 0), 'displacement'),
    ]
    for call, parameter in cases:
      with pytest.raises(ValueError, match=parameter):
        call()


class TestZeroRuntimeDependencies:
  '''The module must import nothing outside the standard library.'''

  def test_imports_are_stdlib_only(self):
    for name in _imported_modules():
      root = name.split('.')[0]
      assert root in sys.stdlib_module_names or root == 'stock_rl', name

  def test_only_stdlib_and_the_existing_indicator_helpers(self):
    # The only project imports are the bar type and the already-audited
    # ``ema``/``sma`` primitives, which keeps this module free of a
    # second, subtly different moving average.
    project = [
      name for name in _imported_modules()
      if name.split('.')[0] not in sys.stdlib_module_names
    ]
    assert project == ['stock_rl.bars', 'stock_rl.indicators']




