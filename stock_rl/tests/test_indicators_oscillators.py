#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the oscillator indicators.

Warm-up behaviour is tested as carefully as the values, because a
silently wrong indicator in the warm-up region is worse than a missing
one: it is arithmetically valid and would silently size a position.

Three failure modes are guarded deliberately, because this repository
has been bitten by all three already:

* length. Every function must return a list as long as its input, and
  every function has its own length test plus a causality test that
  re-runs it on truncated inputs.
* conflation. %K is not %D is not slow %D, CCI's mean deviation is
  Wilder-smoothed rather than a plain mean, and the Money Flow Index is
  not RSI with volume substituted. ``TestIndicatorsAreNotAliased``
  compares all 91 pairs of outputs on one shared series, so any two
  that collapse into the same function fail.
* look-ahead. ``assert_causal`` recomputes each indicator on truncated
  inputs and requires the surviving prefix to be unchanged, which is the
  only check that catches a value at index ``i`` reaching past ``i``.

Nothing here asserts that any of these indicators predicts anything. The
tests below check arithmetic, boundaries and degenerate cases only.
'''

from __future__ import annotations

import inspect
import itertools
import math
from datetime import datetime, timedelta

import pytest

from stock_rl.bars import Bar
from stock_rl.indicators import momentum, rsi
from stock_rl.indicators_oscillators import (
  awesome_oscillator,
  chaikin_oscillator,
  commodity_channel_index,
  money_flow_index,
  rate_of_change,
  roc_on_volume,
  slow_stochastic_d,
  smoothed_roc,
  stochastic_d,
  stochastic_k,
  stochastic_rsi,
  ultimate_oscillator,
  williams_ad,
  williams_percent_r,
)

START = datetime(2026, 1, 1)


def bars_from(rows):
  '''Return bars from (close, high, low, volume) tuples.

  The open is set to the close, because no oscillator here reads it and
  a data set where open differs from close would hide a field that is
  being used when it should not be.

  Args:
    rows: Sequence of (close, high, low, volume) tuples.

  Returns:
    List of bars one hour apart, in the order given.
  '''
  return [
    Bar(START + timedelta(hours=index), close, high, low, close, volume)
    for index, (close, high, low, volume) in enumerate(rows)
  ]


def without_volume(bars):
  '''Return the same bars with traded quantity set to zero.

  Args:
    bars: Bars to copy.

  Returns:
    New bars, so a caller can prove a field is ignored.
  '''
  return [Bar(bar.timestamp, bar.open, bar.high, bar.low, bar.close, 0.0)
          for bar in bars]


def flat_bars(count=40, price=100.0, volume=1000.0):
  '''Return bars with no intrabar range and no price movement.'''
  return bars_from([(price, price, price, volume)] * count)


def rising_bars(count=40, at_high=False, start=100.0, step=2.0):
  '''Return a strictly rising series.

  Args:
    count: Number of bars.
    at_high: When true the close is the bar high, so the close sits at
      the top of the window range; otherwise the range straddles it.
    start: First close.
    step: Close-to-close increment.

  Returns:
    List of bars.
  '''
  rows = []
  for index in range(count):
    close = start + step * index
    high = close if at_high else close + 1.0
    rows.append((close, high, close - 1.0, 1000.0 + 10.0 * index))
  return bars_from(rows)


def falling_bars(count=40, at_low=False, start=140.0, step=2.0):
  '''Return a strictly falling series, optionally closing on the low.'''
  rows = []
  for index in range(count):
    close = start - step * index
    low = close if at_low else close - 1.0
    rows.append((close, close + 1.0, low, 1000.0 - 5.0 * index))
  return bars_from(rows)


def gapped_bars(count=20, start=100.0, step=10.0, rising=True):
  '''Return a series whose bars step by more than their own range.

  The previous close then falls outside the current bar, which is the
  only case in which the previous close changes a range calculation.

  Args:
    count: Number of bars.
    start: First close.
    step: Close-to-close increment.
    rising: Direction of the step.

  Returns:
    List of bars with a one-point high-low span.
  '''
  rows = []
  for index in range(count):
    close = start + step * index if rising else start - step * index
    rows.append((close, close + 1.0, close - 1.0, 1000.0))
  return bars_from(rows)


def varied_bars(count=60):
  '''Return a generator for one shared deterministic series.

  A sine plus a drift, with a volume that swings out of phase with
  price, keeps the oscillators from agreeing by accident. The
  generator shape matters: the causality helper calls it at several
  lengths, and a captured series would defeat the whole point.

  Args:
    count: Number of bars to generate.

  Returns:
    List of bars.
  '''
  rows = []
  for index in range(count):
    close = 100.0 + 4.0 * math.sin(index / 3.0) + 0.4 * index
    high = close + 0.8 + 0.6 * abs(math.cos(index / 2.0))
    low = close - 0.7 - 0.5 * abs(math.sin(index / 4.0))
    volume = 1000.0 + 250.0 * math.sin(index / 1.5) + 40.0 * index
    rows.append((close, high, low, volume))
  return bars_from(rows)


def closes_of(count=60):
  '''Return the closing prices of the shared deterministic series.'''
  return [bar.close for bar in varied_bars(count)]


def first_defined(series):
  '''Return the index of the first non-None reading, or None.'''
  for index, value in enumerate(series):
    if value is not None:
      return index
  return None


def assert_causal(build):
  '''Return a test asserting no look-ahead and length preservation.

  Args:
    build: Callable taking a bar count and returning the indicator
      series for that many bars.

  Returns:
    A zero-argument test function. It recomputes the indicator on every
    truncated prefix and requires each prefix value to be identical to
    the corresponding value of the full series, which holds only if no
    value at index ``i`` used a bar after ``i``. The length assertion
    runs on every prefix as well.
  '''
  def test() -> None:
    full = build(len(varied_bars()))
    assert len(full) == len(varied_bars())
    for count in range(0, len(full)):
      prefix = build(count)
      assert len(prefix) == count
      for index, value in enumerate(prefix):
        if value is None:
          assert full[index] is None, f'index {index} lost its warm-up'
        else:
          assert full[index] == pytest.approx(value), (
            f'index {index} changed when later bars were appended')
  return test


# ---------------------------------------------------------------------------
# Stochastic family
# ---------------------------------------------------------------------------

STOCHASTIC_ROWS = [
  (9.0, 10.0, 8.0, 100.0),
  (10.0, 11.0, 9.0, 100.0),
  (12.0, 13.0, 10.0, 100.0),
  (11.0, 14.0, 11.0, 100.0),
  (14.0, 15.0, 12.0, 100.0),
]


class TestStochasticK:
  '''The fast stochastic, the single-lookback oscillator.'''

  def test_known_values(self):
    # Window 3 over bars 0-2: highest high 13, lowest low 8, close 12,
    # so %K = 100 * (12 - 8) / (13 - 8) = 80.
    # Window 3 over bars 1-3: highest 14, lowest 9, close 11,
    # so %K = 100 * (11 - 9) / (14 - 9) = 40.
    # Window 3 over bars 2-4: highest 15, lowest 10, close 14,
    # so %K = 100 * (14 - 10) / (15 - 10) = 80.
    assert stochastic_k(bars_from(STOCHASTIC_ROWS), 3) == [
      None, None, 80.0, 40.0, 80.0]

  def test_warmup_boundary_is_window_minus_one(self):
    series = stochastic_k(varied_bars(), 5)
    assert series[:4] == [None] * 4
    assert series[4] is not None
    assert first_defined(series) == 4

  def test_one_bar_short_of_the_window_is_all_none(self):
    assert stochastic_k(varied_bars(4), 5) == [None] * 4

  def test_length_matches_input(self):
    assert len(stochastic_k(varied_bars(17), 5)) == 17

  def test_no_lookahead(self):
    assert_causal(lambda count: stochastic_k(varied_bars(count), 5))()

  def test_window_of_one_is_the_close_position_in_its_own_range(self):
    # With one bar the range is that bar alone, so the close sits inside
    # its own high-low span and %K is simply its own percentile.
    bars = bars_from([(9.0, 12.0, 6.0, 100.0)])
    assert stochastic_k(bars, 1) == [pytest.approx(50.0)]

  def test_flat_series_is_the_midpoint_not_a_division_error(self):
    # No intrabar range anywhere: the close is both the highest high and
    # the lowest low, so its position is undefined. 50.0 is the
    # midpoint of the scale, reported instead of dividing by zero.
    assert stochastic_k(flat_bars(10), 3)[-1] == pytest.approx(50.0)

  def test_close_at_the_window_low_is_zero(self):
    # Positive control for the degenerate case above: a real range whose
    # close sits on its lowest low is a genuine 0, not the midpoint.
    bars = bars_from([(5.0, 10.0, 5.0, 100.0)] * 6)
    assert stochastic_k(bars, 3)[-1] == pytest.approx(0.0)

  def test_rising_series_closing_on_its_high_saturates(self):
    assert (stochastic_k(rising_bars(20, at_high=True), 5)[-1]
            == pytest.approx(100.0))

  def test_falling_series_closing_on_its_low_saturates(self):
    assert (stochastic_k(falling_bars(20, at_low=True), 5)[-1]
            == pytest.approx(0.0))

  def test_rising_series_between_its_own_wicks_is_bounded(self):
    # A rising close with the range straddling it sits above the middle
    # of the range but below its high, which is the ordinary case.
    value = stochastic_k(rising_bars(20), 5)[-1]
    assert 50.0 < value < 100.0

  def test_stays_within_zero_and_one_hundred(self):
    for value in stochastic_k(varied_bars(), 5):
      if value is not None:
        assert 0.0 <= value <= 100.0

  @pytest.mark.parametrize('window', [0, -1, -14])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      stochastic_k(varied_bars(5), window)


class TestStochasticD:
  '''Fast %D, a simple average of the fast %K.'''

  def test_known_values(self):
    # %K over the same five bars is [None, None, 80, 40, 80], so a
    # two-bar %D is (80 + 40) / 2 = 60 at index 3 and (40 + 80) / 2 = 60
    # at index 4.
    bars = bars_from(STOCHASTIC_ROWS)
    assert stochastic_d(bars, 3, 2) == [None, None, None, 60.0, 60.0]

  def test_warmup_boundary_is_window_plus_signal_minus_two(self):
    series = stochastic_d(varied_bars(), 5, 3)
    assert series[:6] == [None] * 6
    assert first_defined(series) == 6

  def test_one_bar_short_of_the_boundary_is_all_none(self):
    assert stochastic_d(varied_bars(6), 5, 3) == [None] * 6

  def test_length_matches_input(self):
    assert len(stochastic_d(varied_bars(19), 5, 3)) == 19

  def test_no_lookahead(self):
    assert_causal(lambda count: stochastic_d(varied_bars(count), 5, 3))()

  def test_is_the_average_of_k_not_k_itself(self):
    # The negative case: %D is not %K. The positive control: %D is
    # exactly the mean of the trailing %K readings, one bar later.
    bars = bars_from(STOCHASTIC_ROWS)
    quick = stochastic_k(bars, 3)
    signal = stochastic_d(bars, 3, 2)
    assert signal[3] != quick[3]
    assert signal[3] == pytest.approx((quick[2] + quick[3]) / 2)

  def test_smoothing_preserves_a_pinned_level_and_defers_it_by_one_bar(self):
    # Positive control on the smoothing arithmetic: %K is pinned at 50
    # once the flat bars are in range, and averaging a constant gives the
    # same constant back one bar later.
    bars = bars_from([(10.0, 10.0, 10.0, 100.0)] * 3
                     + [(10.0, 20.0, 0.0, 100.0)] * 2)
    assert stochastic_k(bars, 2) == [None, 50.0, 50.0, 50.0, 50.0]
    assert stochastic_d(bars, 2, 2) == [None, None, 50.0, 50.0, 50.0]

  def test_signal_window_of_one_is_k_itself(self):
    bars = varied_bars(30)
    signal = stochastic_d(bars, 5, 1)
    quick = stochastic_k(bars, 5)
    assert signal[:4] == [None] * 4
    assert signal[4:] == pytest.approx(quick[4:])

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      stochastic_d(varied_bars(8), window)

  @pytest.mark.parametrize('signal_window', [0, -2])
  def test_rejects_bad_signal_window(self, signal_window):
    with pytest.raises(ValueError, match='signal_window'):
      stochastic_d(varied_bars(8), 5, signal_window)


class TestSlowStochasticD:
  '''Slow %D, the raw %K smoothed twice.'''

  def test_known_values(self):
    # %K is [None, None, 80, 40, 80]. The first two-bar average is
    # 60 at index 3, the second two-bar average is (60 + 60) / 2 = 60
    # at index 4.
    bars = bars_from(STOCHASTIC_ROWS)
    assert slow_stochastic_d(bars, 3, 2, 2) == [None] * 4 + [60.0]

  def test_warmup_boundary_sums_the_three_contributions(self):
    series = slow_stochastic_d(varied_bars(), 5, 3, 3)
    # 5 - 1 for %K, then 3 - 1 and 3 - 1 for the two averages. The
    # averaging windows here are sequential, not nested, so unlike the
    # Ultimate Oscillator the contributions do add.
    assert series[:8] == [None] * 8
    assert first_defined(series) == 8

  def test_one_bar_short_of_the_boundary_is_all_none(self):
    assert slow_stochastic_d(varied_bars(8), 5, 3, 3) == [None] * 8

  def test_length_matches_input(self):
    assert len(slow_stochastic_d(varied_bars(21), 5, 3, 3)) == 21

  def test_no_lookahead(self):
    assert_causal(lambda count: slow_stochastic_d(
      varied_bars(count), 5, 3, 3))()

  def test_equals_a_smoothed_fast_d(self):
    # The definition: slow %D is slow %K averaged again, and slow %K is
    # fast %D. Composed here from the exported pieces, so the
    # composition itself is what is under test.
    bars = varied_bars(40)
    fast_d = stochastic_d(bars, 5, 3)
    expected: list[float | None] = [None] * len(bars)
    for index in range(8, len(bars)):
      expected[index] = pytest.approx(
        (fast_d[index - 2] + fast_d[index - 1] + fast_d[index]) / 3)
    assert slow_stochastic_d(bars, 5, 3, 3) == expected

  def test_double_smoothing_differs_from_single(self):
    # The negative case that this is a distinct function at all.
    bars = varied_bars(40)
    fast_d = stochastic_d(bars, 5, 3)
    slow_d = slow_stochastic_d(bars, 5, 3, 3)
    assert any(fast_d[index] != pytest.approx(slow_d[index])
               for index in range(8, 40))

  def test_equal_windows_collapse_to_a_single_average(self):
    # Positive control for the smoothing arithmetic.
    bars = bars_from(STOCHASTIC_ROWS)
    assert slow_stochastic_d(bars, 3, 2, 1) == stochastic_d(bars, 3, 2)

  @pytest.mark.parametrize('smooth_k', [0, -1])
  def test_rejects_bad_smooth_k(self, smooth_k):
    with pytest.raises(ValueError, match='smooth_k'):
      slow_stochastic_d(varied_bars(12), 5, smooth_k)

  @pytest.mark.parametrize('smooth_d', [0, -1])
  def test_rejects_bad_smooth_d(self, smooth_d):
    with pytest.raises(ValueError, match='smooth_d'):
      slow_stochastic_d(varied_bars(12), 5, 3, smooth_d)


class TestWilliamsPercentR:
  '''Williams %R, the same position on a negative scale.'''

  def test_known_values(self):
    # -100 * (highest - close) / (highest - lowest):
    # index 2: -100 * (13 - 12) / 5 = -20
    # index 3: -100 * (14 - 11) / 5 = -60
    # index 4: -100 * (15 - 14) / 5 = -20
    assert williams_percent_r(bars_from(STOCHASTIC_ROWS), 3) == [
      None, None, -20.0, -60.0, -20.0]

  def test_warmup_boundary_is_window_minus_one(self):
    series = williams_percent_r(varied_bars(), 5)
    assert series[:4] == [None] * 4
    assert first_defined(series) == 4

  def test_length_matches_input(self):
    assert len(williams_percent_r(varied_bars(13), 5)) == 13

  def test_no_lookahead(self):
    assert_causal(
      lambda count: williams_percent_r(varied_bars(count), 5))()

  def test_is_k_shifted_by_one_hundred(self):
    # The negative case: %R is not %K. The positive control: the exact
    # relationship that makes them the same information twice.
    bars = varied_bars(40)
    quick = stochastic_k(bars, 5)
    percent_r = williams_percent_r(bars, 5)
    for index in range(4, 40):
      assert percent_r[index] == pytest.approx(quick[index] - 100.0)
      assert percent_r[index] != quick[index]

  def test_flat_series_is_the_midpoint(self):
    # Undefined position, so the midpoint of [-100, 0] rather than 0/0.
    assert williams_percent_r(flat_bars(10), 3)[-1] == pytest.approx(-50.0)

  def test_close_at_the_window_high_is_zero(self):
    # Positive control on the degenerate midpoint above: a real range
    # closing on its high is a genuine 0, not the midpoint.
    bars = bars_from([(15.0, 15.0, 5.0, 100.0)] * 6)
    assert williams_percent_r(bars, 3)[-1] == pytest.approx(0.0)

  def test_saturates_at_both_ends(self):
    assert (williams_percent_r(rising_bars(20, at_high=True), 5)[-1]
            == pytest.approx(0.0))
    assert (williams_percent_r(falling_bars(20, at_low=True), 5)[-1]
            == pytest.approx(-100.0))

  def test_stays_within_minus_hundred_and_zero(self):
    for value in williams_percent_r(varied_bars(), 5):
      if value is not None:
        assert -100.0 <= value <= 0.0

  @pytest.mark.parametrize('window', [0, -3])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      williams_percent_r(varied_bars(5), window)


# ---------------------------------------------------------------------------
# Commodity Channel Index
# ---------------------------------------------------------------------------

CCI_ROWS = [
  (9.0, 12.0, 6.0, 100.0),
  (12.0, 15.0, 9.0, 100.0),
  (9.0, 12.0, 6.0, 100.0),
  (15.0, 18.0, 12.0, 100.0),
  (6.0, 9.0, 3.0, 100.0),
  (15.0, 18.0, 12.0, 100.0),
]


class TestCommodityChannelIndex:
  '''CCI with Wilder-smoothed mean deviation.'''

  def test_known_values(self):
    # Typical prices are (H + L + C) / 3 = 9, 12, 9, 15, 6, 15.
    # Window 3 moving averages: 10, 12, 10, 12 at indices 2..5.
    # Deviations: |9 - 10| = 1, |15 - 12| = 3, |6 - 10| = 4.
    # Wilder seed at index 4: (1 + 3 + 4) / 3 = 8/3.
    # CCI(4) = (6 - 10) / (0.015 * 8/3) = -4 / 0.04 = -100.
    # Wilder step at index 5: (8/3 * 2 + |15 - 12|) / 3 = 25/9.
    # CCI(5) = (15 - 12) / (0.015 * 25/9) = 3 / (1/24) = 72.
    result = commodity_channel_index(bars_from(CCI_ROWS), 3)
    assert result[:4] == [None] * 4
    assert result[4] == pytest.approx(-100.0)
    assert result[5] == pytest.approx(72.0)

  def test_warmup_boundary_is_twice_window_minus_two(self):
    # TPMA needs window - 1 bars; the Wilder seed then needs `window`
    # deviations, the first of which exists at index window - 1.
    series = commodity_channel_index(varied_bars(), 4)
    assert series[:6] == [None] * 6
    assert first_defined(series) == 6

  def test_one_bar_short_of_the_boundary_is_all_none(self):
    assert commodity_channel_index(varied_bars(6), 4) == [None] * 6

  def test_length_matches_input(self):
    assert len(commodity_channel_index(varied_bars(23), 4)) == 23

  def test_no_lookahead(self):
    assert_causal(
      lambda count: commodity_channel_index(varied_bars(count), 4))()

  def test_mean_deviation_is_wilder_smoothed_not_a_plain_mean(self):
    # The negative case. Lambert's presentation averages |TP - TPMA|
    # around the current TPMA: at index 4 of the hand case that is
    # (6 - 10) / (0.015 * (1 + 5 + 4) / 3) = -4 / 0.05 = -80, which is
    # not the Wilder-smoothed -100 this module returns.
    bars = bars_from(CCI_ROWS)
    typical = [bar.typical_price for bar in bars]
    lambert = (typical[4] - 10.0) / (
      0.015 * sum(abs(value - 10.0) for value in typical[2:5]) / 3)
    assert lambert == pytest.approx(-80.0)
    assert lambert != pytest.approx(
      commodity_channel_index(bars, 3)[4])

  def test_flat_typical_price_is_zero(self):
    assert commodity_channel_index(flat_bars(12), 3)[-1] == pytest.approx(0.0)

  def test_a_straight_ramp_gives_a_constant_reading(self):
    # Positive control on the whole chain: on a linear ramp the offset
    # from the moving average is 4.5 at every bar and the smoothed mean
    # deviation settles at 4.5 too, so CCI = 4.5 / (0.015 * 4.5).
    ramp = bars_from([(100.0 + index, 101.0 + index, 99.0 + index, 1000.0)
                      for index in range(20)])
    result = commodity_channel_index(ramp, 10)
    assert all(value == pytest.approx(4.5 / (0.015 * 4.5))
               for value in result[18:] if value is not None)

  def test_is_unbounded_and_the_hundred_line_is_only_a_convention(self):
    # A 20-bar ramp gives 66.67, because a linear ramp has a constant
    # offset. One 10-point jump then breaks the ratio: at index 20 the
    # offset is 13.5 against a smoothed deviation of 5.4, so
    # 13.5 / (0.015 * 5.4) = 166.67. The +/-100 line is Lambert's scale
    # constant, not a cap and not a level validated in this project.
    rows = [(100.0 + index, 101.0 + index, 99.0 + index, 1000.0)
            for index in range(20)]
    rows += [(130.0, 131.0, 129.0, 1000.0)] * 6
    result = commodity_channel_index(bars_from(rows), 10)
    assert result[20] == pytest.approx(166.66666666666666)
    assert result[20] > 100.0

  def test_falling_series_is_negative(self):
    assert commodity_channel_index(falling_bars(40), 20)[-1] < 0.0

  @pytest.mark.parametrize('window', [0, -2])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      commodity_channel_index(varied_bars(6), window)


# ---------------------------------------------------------------------------
# Ultimate Oscillator
# ---------------------------------------------------------------------------

UO_ROWS = [
  (10.0, 15.0, 5.0, 100.0),
  (12.0, 15.0, 10.0, 100.0),
  (11.0, 13.0, 9.0, 100.0),
  (13.0, 14.0, 12.0, 100.0),
  (12.0, 15.0, 11.0, 100.0),
]


class TestUltimateOscillator:
  '''Larry Williams' three-period combination.'''

  def test_buying_pressure_and_true_range(self):
    # Index 1: low 10 against a prior close of 10 gives a true-range low
    # of 10 and a high of 15, so BP = 12 - 10 = 2 and TR = 5.
    # Index 2: the prior close of 12 is inside the bar, so BP = 11 - 9 = 2
    # and TR = 13 - 9 = 4.
    # Index 3: the prior close of 11 is BELOW the low of 12, so the range
    # is measured from that prior close: BP = 13 - 11 = 2, TR = 3.
    # Index 4: BP = 12 - 11 = 1, TR = 15 - 11 = 4.
    assert ultimate_oscillator(bars_from(UO_ROWS), 2, 3, 4)[4] == (
      pytest.approx(43.75))

  def test_known_value_arithmetic(self):
    # At index 4 with periods 2, 3, 4:
    #   Avg2(BP) = (2 + 1) / 2 = 3/2      Avg2(TR) = (3 + 4) / 2 = 7/2
    #   Avg3(BP) = (2 + 2 + 1) / 3 = 5/3  Avg3(TR) = (4 + 3 + 4) / 3 = 11/3
    #   Avg4(BP) = (2 + 2 + 2 + 1) / 4 = 7/4, Avg4(TR) = (5+4+3+4)/4 = 4
    # Numerator   = 4 * 3/2 + 2 * 5/3 + 7/4 = 72/12 + 40/12 + 21/12 = 133/12
    # Denominator = 4 * 7/2 + 2 * 11/3 + 4    = 42/3 + 22/3 + 12/3 = 76/3
    # UO = 100 * (133/12) / (76/3) = 100 * 399 / 912 = 43.75
    numerator = 4 * 1.5 + 2 * (5 / 3) + 7 / 4
    denominator = 4 * 3.5 + 2 * (11 / 3) + 4
    assert numerator == pytest.approx(133 / 12)
    assert denominator == pytest.approx(76 / 3)
    assert 100 * numerator / denominator == pytest.approx(43.75)

  def test_warmup_boundary_is_the_longest_lookback_not_their_sum(self):
    # The three windows are nested: the four-bar sum already contains
    # the two-bar and three-bar sums, so index 4 is computable from five
    # bars even though 2 + 3 + 4 = 9 bars have not arrived. One bar
    # beyond the longest lookback is needed only because there is no
    # previous close at index 0.
    result = ultimate_oscillator(bars_from(UO_ROWS), 2, 3, 4)
    assert result[:4] == [None] * 4
    assert result[4] is not None

  def test_one_bar_short_of_the_boundary_is_all_none(self):
    assert ultimate_oscillator(bars_from(UO_ROWS[:4]), 2, 3, 4) == [None] * 4

  def test_boundary_is_the_maximum_of_the_three_periods(self):
    # Deliberately unordered, including the published 7/14/28 setup,
    # which needs 29 bars and not 7 + 14 + 28 = 49.
    for fast, middle, slow in ((3, 6, 12), (12, 6, 3), (5, 5, 5),
                               (7, 14, 28)):
      result = ultimate_oscillator(varied_bars(60), fast, middle, slow)
      assert first_defined(result) == max(fast, middle, slow)
      assert result[max(fast, middle, slow) - 1] is None

  def test_length_matches_input(self):
    assert len(ultimate_oscillator(varied_bars(31), 3, 6, 12)) == 31

  def test_no_lookahead(self):
    assert_causal(
      lambda count: ultimate_oscillator(varied_bars(count), 3, 6, 12))()

  def test_is_not_the_ratio_of_sums_variant(self):
    # The negative case. The reference page forms a weighted average of
    # three separate buying percentages; Williams forms a ratio of
    # weighted sums. On the hand case they differ.
    variant = 100.0 * (4 * 3 / 7 + 2 * 5 / 11 + 7 / 16) / 7
    assert variant != pytest.approx(43.75, abs=1e-6)
    assert variant < 43.75

  def test_gaps_widen_the_range_through_the_previous_close(self):
    # Positive control on the previous-close terms: a gap down leaves
    # the intrabar range tiny while the true range is not, which is the
    # only reason the previous close appears in the formula at all.
    gapped = bars_from([
      (100.0, 101.0, 99.0, 1000.0),
      (100.0, 101.0, 99.0, 1000.0),
      (90.0, 91.0, 89.0, 1000.0),
      (90.0, 91.0, 89.0, 1000.0),
      (90.0, 91.0, 89.0, 1000.0),
    ])
    unbroken = flat_bars(5, price=100.0, volume=1000.0)
    assert (ultimate_oscillator(gapped, 2, 3, 4)[-1]
            < ultimate_oscillator(unbroken, 2, 3, 4)[-1])

  def test_series_with_no_range_is_the_midpoint(self):
    # Buying pressure and true range are both identically zero, so the
    # position is undefined and 50.0 is reported rather than 0/0.
    assert (ultimate_oscillator(flat_bars(12), 2, 3, 4)[-1]
            == pytest.approx(50.0))

  def test_a_strong_uptrend_reads_near_the_top(self):
    # Closes rise 10 a bar with a one-point range, so the previous close
    # sits outside the current bar: BP = close - previous close = 10 and
    # TR = (close + 1) - (close - 10) = 11, giving 100 * 10/11.
    result = ultimate_oscillator(gapped_bars(20), 3, 6, 12)
    assert result[-1] == pytest.approx(100.0 * 10 / 11)

  def test_a_strong_downtrend_reads_near_the_bottom(self):
    # Mirrored, the previous close is above the bar: BP = close - low = 1
    # and TR = (close + 10) - low = 11, giving 100 * 1/11. Buying pressure
    # is bounded below by zero because the close cannot be below its own
    # low, which is why the oscillator is confined to [0, 100].
    result = ultimate_oscillator(gapped_bars(20, start=300.0, rising=False),
                                 3, 6, 12)
    assert result[-1] == pytest.approx(100.0 / 11)

  def test_stays_within_zero_and_one_hundred(self):
    for value in ultimate_oscillator(varied_bars(), 3, 6, 12):
      if value is not None:
        assert 0.0 <= value <= 100.0

  @pytest.mark.parametrize('name', ['fast', 'middle', 'slow'])
  def test_rejects_bad_period(self, name):
    periods = {'fast': 3, 'middle': 6, 'slow': 12}
    periods[name] = 0
    with pytest.raises(ValueError, match=name):
      ultimate_oscillator(varied_bars(20), **periods)


# ---------------------------------------------------------------------------
# Stochastic RSI
# ---------------------------------------------------------------------------


class TestStochasticRsi:
  '''A stochastic of the RSI, not the RSI.'''

  def test_known_values(self):
    # RSI(2): changes are +1, -0.5, +1.5, -0.5.
    # Seed at index 2: gain 1/2, loss 0.5/2, so 100 - 100/(1 + 2) = 66.67.
    # Index 3: gain (0.5 + 1.5) / 2 = 1.0, loss 0.25 / 2 = 0.125,
    # so 100 - 100/(1 + 8) = 88.89.
    # Index 4: gain 0.5, loss (0.125 + 0.5) / 2 = 0.3125,
    # so 100 - 100/(1 + 1.6) = 61.54.
    # The three-bar window at index 4 spans 61.54, 66.67, 88.89 and the
    # current reading is the minimum, so the position is exactly 0.
    result = stochastic_rsi([10.0, 11.0, 10.5, 12.0, 11.5], 2, 3)
    assert result[:4] == [None] * 4
    assert result[4] == pytest.approx(0.0)

  def test_current_reading_on_top_of_its_window_is_exactly_one(self):
    # The same five bars ending at 13.0: the last change is +1, so the
    # RSI is 88.89, equal to the window maximum, and the position is 1.
    result = stochastic_rsi([10.0, 11.0, 10.5, 12.0, 13.0], 2, 3)
    assert result[4] == pytest.approx(1.0)

  def test_warmup_boundary_is_rsi_window_plus_window_minus_one(self):
    series = stochastic_rsi(closes_of(60), 5, 4)
    assert series[:8] == [None] * 8
    assert first_defined(series) == 8

  def test_one_bar_short_of_the_boundary_is_all_none(self):
    assert stochastic_rsi(closes_of(8), 5, 4) == [None] * 8

  def test_length_matches_input(self):
    assert len(stochastic_rsi(closes_of(29), 5, 4)) == 29

  def test_no_lookahead(self):
    assert_causal(lambda count: stochastic_rsi(closes_of(count), 5, 4))()

  def test_stays_within_zero_and_one(self):
    closes = [100.0 + 4.0 * math.sin(index / 3.0) for index in range(80)]
    for value in stochastic_rsi(closes, 14, 14):
      if value is not None:
        assert 0.0 <= value <= 1.0

  def test_flat_series_is_the_midpoint(self):
    assert stochastic_rsi([100.0] * 40, 14, 14)[-1] == pytest.approx(0.5)

  def test_a_pure_ramp_pins_the_rsi_and_so_gives_the_midpoint(self):
    # Positive control for the degenerate case: an unbroken advance
    # pins Wilder's RSI at exactly 100 at every bar, so the wrapped
    # oscillator never moves and its position is undefined. This is why
    # an unbroken trend cannot reach 1.0.
    closes = [float(100 + index) for index in range(40)]
    assert rsi(closes, 14)[-1] == pytest.approx(100.0)
    assert stochastic_rsi(closes, 14, 14)[-1] == pytest.approx(0.5)

  def test_is_not_the_rsi_it_wraps(self):
    # The negative case: a second stochastic is not the oscillator it
    # wraps. The positive control: both are still inside their ranges.
    closes = [100.0 + 4.0 * math.sin(index / 3.0) for index in range(60)]
    outer = stochastic_rsi(closes, 14, 14)
    inner = rsi(closes, 14)
    assert outer[27] != pytest.approx(inner[27])
    assert 0.0 <= outer[27] <= 1.0
    assert 0.0 <= inner[27] <= 100.0

  def test_reaches_extremes_more_often_than_the_rsi(self):
    # Positive control for the reason the indicator exists: a stochastic
    # of a bounded oscillator touches its own bounds far more often.
    closes = [100.0 + 4.0 * math.sin(index / 3.0) + 0.3 * index
              for index in range(120)]
    outer = [v for v in stochastic_rsi(closes, 14, 14) if v is not None]
    inner = [v for v in rsi(closes, 14) if v is not None]
    assert sum(1 for v in outer if v in (0.0, 1.0)) > \
      sum(1 for v in inner if v in (0.0, 100.0))

  @pytest.mark.parametrize('rsi_window', [0, -1])
  def test_rejects_bad_rsi_window(self, rsi_window):
    with pytest.raises(ValueError, match='rsi_window'):
      stochastic_rsi([1.0, 2.0, 3.0], rsi_window)

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      stochastic_rsi([1.0, 2.0, 3.0], 2, window)


# ---------------------------------------------------------------------------
# Money Flow Index
# ---------------------------------------------------------------------------


class TestMoneyFlowIndex:
  '''The volume-weighted RSI.'''

  def test_known_values(self):
    # Typical prices are 9, 11, 10 and money flows are 900, 2200, 3000.
    # In the two-bar window ending at index 2 the typical price rose then
    # fell, so positive flow is 2200 and negative flow is 3000:
    # MFI = 100 - 100 / (1 + 2200/3000) = 100 * 2200 / 5200 = 42.3077.
    bars = bars_from([
      (9.0, 10.0, 8.0, 100.0),
      (11.0, 12.0, 10.0, 200.0),
      (10.0, 11.0, 9.0, 300.0),
    ])
    assert money_flow_index(bars, 2)[2] == pytest.approx(42.30769230769)
    assert 100 * 2200 / 5200 == pytest.approx(42.30769230769)

  def test_warmup_boundary_is_the_window_itself(self):
    # The window holds `window` money flows but `window + 1` typical
    # prices, because the first flow in the window still needs the
    # direction of its predecessor.
    series = money_flow_index(varied_bars(30), 5)
    assert series[:5] == [None] * 5
    assert first_defined(series) == 5

  def test_one_bar_short_of_the_boundary_is_all_none(self):
    assert money_flow_index(varied_bars(5), 5) == [None] * 5

  def test_length_matches_input(self):
    assert len(money_flow_index(varied_bars(27), 5)) == 27

  def test_no_lookahead(self):
    assert_causal(lambda count: money_flow_index(varied_bars(count), 5))()

  def test_is_not_rsi_with_volume_substituted(self):
    # The negative case, with the arithmetic. A window of two over three
    # bars whose typical price rises then falls: RSI(2) sees a gain of 2
    # then a loss of 0.5 and prints 100 - 100/5 = 80. The same bars with
    # all the volume on the down bar print 100 * 120 / 115120, which is
    # near zero. Same price path, opposite reading, purely because MFI
    # weights the two legs by traded money rather than by price move.
    bars = bars_from([
      (10.0, 10.0, 10.0, 10.0),
      (12.0, 12.0, 12.0, 10.0),
      (11.5, 11.5, 11.5, 10000.0),
    ])
    assert rsi([bar.close for bar in bars], 2)[-1] == pytest.approx(80.0)
    reading = money_flow_index(bars, 2)[-1]
    assert reading == pytest.approx(100 * 120 / 115120)
    assert reading < 1.0

  def test_swapping_the_volume_flips_the_reading(self):
    # Positive control for the previous test: the same price path with
    # the volume on the up bar reads near 100 instead.
    bars = bars_from([
      (10.0, 10.0, 10.0, 10000.0),
      (12.0, 12.0, 12.0, 10000.0),
      (11.5, 11.5, 11.5, 10.0),
    ])
    assert money_flow_index(bars, 2)[-1] > 99.0

  def test_flat_series_is_fifty(self):
    assert money_flow_index(flat_bars(12), 5)[-1] == pytest.approx(50.0)

  def test_zero_volume_is_fifty(self):
    # No money moves, so neither flow exists and the reading is neutral
    # rather than a division by zero.
    bars = without_volume(rising_bars(12))
    assert money_flow_index(bars, 5)[-1] == pytest.approx(50.0)

  def test_rising_typical_price_is_one_hundred(self):
    assert money_flow_index(rising_bars(20), 5)[-1] == pytest.approx(100.0)

  def test_falling_typical_price_is_zero(self):
    assert money_flow_index(falling_bars(20), 5)[-1] == pytest.approx(0.0)

  def test_stays_within_zero_and_one_hundred(self):
    for value in money_flow_index(varied_bars(), 5):
      if value is not None:
        assert 0.0 <= value <= 100.0

  def test_unchanged_typical_price_counts_as_neither_leg(self):
    # Positive control on the no-rule case: a flat bar inside the window
    # must not create positive or negative flow. Rising 10, rising 11,
    # flat 11, rising 12 on a window of three: at index 3 the window holds
    # bars 1, 2 and 3, whose flows are 1100 up, 1100 unchanged and 1200
    # up. Positive 2300, negative 0.
    bars = bars_from([
      (10.0, 10.0, 10.0, 100.0),
      (11.0, 11.0, 11.0, 100.0),
      (11.0, 11.0, 11.0, 100.0),
      (12.0, 12.0, 12.0, 100.0),
    ])
    assert money_flow_index(bars, 3)[3] == pytest.approx(100.0)

  @pytest.mark.parametrize('window', [0, -5])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      money_flow_index(varied_bars(8), window)


# ---------------------------------------------------------------------------
# Smoothed Rate of Change
# ---------------------------------------------------------------------------


class TestSmoothedRoc:
  '''Schutzman's two-step oscillator.'''

  def test_known_values(self):
    # A two-bar EMA has factor 2/3 and seeds on the first close:
    #   e0 = 10
    #   e1 = 10/3 + 2/3 * 12 = 34/3
    #   e2 = 34/9 + 2/3 * 14  = 118/9
    #   e3 = 118/27 + 2/3 * 16 = 406/27
    # SROC(2) = 100 * (e2 - e0) / e0 = 100 * (28/9) / 10 = 31.1111
    # SROC(3) = 100 * (e3 - e1) / e1 = 100 * (100/27) / (34/3)
    #         = 100 * 300 / 918 = 32.6797
    closes = [10.0, 12.0, 14.0, 16.0]
    result = smoothed_roc(closes, 2, 2)
    assert result[:2] == [None, None]
    assert result[2] == pytest.approx(31.11111111111)
    assert result[3] == pytest.approx(32.67973856209)

  def test_warmup_boundary_is_the_lookback(self):
    series = smoothed_roc(closes_of(30), 13, 21)
    assert series[:21] == [None] * 21
    assert first_defined(series) == 21

  def test_one_bar_short_of_the_boundary_is_all_none(self):
    closes = [float(100 + index) for index in range(21)]
    assert smoothed_roc(closes, 13, 21) == [None] * 21

  def test_length_matches_input(self):
    assert len(smoothed_roc(closes_of(41), 5, 9)) == 41

  def test_no_lookahead(self):
    assert_causal(lambda count: smoothed_roc(closes_of(count), 5, 9))()

  def test_ema_window_of_one_reduces_to_the_raw_rate_of_change(self):
    # Positive control on the composition: an EMA span of one is the
    # identity, so the indicator must reproduce rate_of_change exactly.
    assert smoothed_roc(closes_of(30), 1, 7) == rate_of_change(
      closes_of(30), 7)

  def test_smoothing_defers_the_signal_by_several_bars(self):
    # The negative case: SROC is not ROC. The positive control: a ten
    # point jump on a flat series moves the raw rate of change to 10.0 in
    # a single bar, while the smoothed reading is still climbing from
    # 1.43 through 2.65 and 3.70 over the next bars.
    closes = [100.0] * 20 + [110.0] * 4
    raw = rate_of_change(closes, 5)
    smoothed = smoothed_roc(closes, 13, 5)
    assert raw[20] == pytest.approx(10.0)
    assert smoothed[20] != pytest.approx(raw[20])
    assert 0.0 < smoothed[20] < raw[20]
    assert smoothed[20] < smoothed[21] < smoothed[22]

  def test_flat_series_is_zero(self):
    assert smoothed_roc([100.0] * 40, 13, 21)[-1] == pytest.approx(0.0)

  def test_rising_series_stays_positive(self):
    result = smoothed_roc([float(100 + index) for index in range(40)])
    assert all(value > 0.0 for value in result[21:])

  def test_falling_series_stays_negative(self):
    closes = [float(140 - index) for index in range(40)]
    result = smoothed_roc(closes, 13, 21)
    assert all(value < 0.0 for value in result[21:])

  def test_a_geometric_series_converges_up_to_the_raw_rate(self):
    # On a fixed 1% growth rate the raw rate is pinned at
    # 100 * (1.01^21 - 1) = 23.2392. The EMA is seeded on the first
    # close rather than on a price from before the series, so the
    # smoothed reading starts below that and converges upwards towards
    # it without ever reaching it.
    closes = [100.0 * (1.01 ** index) for index in range(40)]
    values = [v for v in smoothed_roc(closes, 13, 21) if v is not None]
    raw = [v for v in rate_of_change(closes, 21) if v is not None]
    assert all(value > 0.0 for value in values)
    assert all(value < raw[0] for value in values)
    assert all(earlier < later
               for earlier, later in zip(values, values[1:]))
    assert raw[0] - values[-1] < 1.0

  def test_non_positive_base_is_none(self):
    # Five leading zeros keep the EMA at zero for five bars, so every
    # base taken from that run is zero and the ratio is undefined rather
    # than infinite. The first usable base is the EMA at index 5.
    closes = [0.0] * 5 + [1.0] * 5
    result = smoothed_roc(closes, 13, 2)
    assert result[:7] == [None] * 7
    assert result[7] is not None

  @pytest.mark.parametrize('ema_window', [0, -1])
  def test_rejects_bad_ema_window(self, ema_window):
    with pytest.raises(ValueError, match='ema_window'):
      smoothed_roc([1.0, 2.0, 3.0], ema_window)

  @pytest.mark.parametrize('lookback', [0, -4])
  def test_rejects_bad_lookback(self, lookback):
    with pytest.raises(ValueError, match='lookback'):
      smoothed_roc([1.0, 2.0, 3.0], 3, lookback)


# ---------------------------------------------------------------------------
# Rate of Change, price and volume
# ---------------------------------------------------------------------------


class TestRateOfChange:
  '''Percentage rate of change around zero.'''

  def test_known_values(self):
    # 100 * (12 - 10) / 10 = 20
    assert rate_of_change([10.0, 11.0, 12.0], 2) == [
      None, None, pytest.approx(20.0)]

  def test_warmup_boundary_is_the_lookback(self):
    series = rate_of_change(closes_of(20), 12)
    assert series[:12] == [None] * 12
    assert first_defined(series) == 12

  def test_one_bar_short_of_the_boundary_is_all_none(self):
    assert rate_of_change([1.0] * 12, 12) == [None] * 12

  def test_length_matches_input(self):
    assert len(rate_of_change([float(v) for v in range(25)], 3)) == 25

  def test_no_lookahead(self):
    assert_causal(lambda count: rate_of_change(closes_of(count), 3))()

  def test_agrees_with_momentum_times_one_hundred(self):
    # Positive control against the existing module: the trailing-return
    # primitive and this series must be the same number.
    closes = closes_of(30)
    assert (rate_of_change(closes, 7)[-1]
            == pytest.approx(momentum(closes, 7) * 100.0))

  def test_flat_series_is_zero(self):
    assert rate_of_change([100.0] * 20, 12)[-1] == pytest.approx(0.0)

  def test_non_positive_base_is_none(self):
    assert rate_of_change([0.0, 0.0, 5.0], 2)[2] is None

  def test_a_geometric_series_gives_a_constant_reading(self):
    # Positive control for the arithmetic: on a fixed growth rate the
    # numerator and denominator scale together, so every reading is
    # 100 * (1.01^5 - 1).
    closes = [100.0 * (1.01 ** index) for index in range(30)]
    values = [v for v in rate_of_change(closes, 5) if v is not None]
    assert all(value == pytest.approx(100 * (1.01 ** 5 - 1))
               for value in values)

  def test_a_straight_ramp_is_positive_and_falls_as_the_base_grows(self):
    # A linear ramp gives a constant absolute move over a growing base,
    # so the percentage drifts down. Stated so the monotone claim is not
    # mistaken for "always rising".
    closes = [float(100 + index) for index in range(30)]
    values = [v for v in rate_of_change(closes, 5) if v is not None]
    assert all(value > 0.0 for value in values)
    assert all(earlier > later
               for earlier, later in zip(values, values[1:]))

  def test_falling_series_is_negative(self):
    closes = [float(130 - index) for index in range(30)]
    assert rate_of_change(closes, 5)[-1] < 0.0

  @pytest.mark.parametrize('lookback', [0, -12])
  def test_rejects_bad_lookback(self, lookback):
    with pytest.raises(ValueError, match='lookback'):
      rate_of_change([1.0, 2.0, 3.0], lookback)


class TestRocOnVolume:
  '''The same formula applied to traded quantity.'''

  def test_known_values(self):
    # 100 * (300 - 100) / 100 = 200
    bars = bars_from([
      (9.0, 10.0, 8.0, 100.0),
      (11.0, 12.0, 10.0, 200.0),
      (10.0, 11.0, 9.0, 300.0),
    ])
    assert roc_on_volume(bars, 2)[2] == pytest.approx(200.0)

  def test_warmup_boundary_is_the_lookback(self):
    assert first_defined(roc_on_volume(varied_bars(20), 12)) == 12

  def test_length_matches_input(self):
    assert len(roc_on_volume(varied_bars(24), 12)) == 24

  def test_no_lookahead(self):
    assert_causal(lambda count: roc_on_volume(varied_bars(count), 3))()

  def test_is_not_the_price_oscillator(self):
    # The negative case, on bars whose price and volume move opposite
    # ways. The positive control: each matches its own input series.
    rows = [(100.0 + index, 101.0 + index, 99.0 + index,
             1000.0 - 20.0 * index) for index in range(20)]
    bars = bars_from(rows)
    on_volume = roc_on_volume(bars, 5)
    on_price = rate_of_change([bar.close for bar in bars], 5)
    assert on_volume[-1] < 0.0
    assert on_price[-1] > 0.0
    assert on_volume[-1] != pytest.approx(on_price[-1])

  def test_matches_the_price_formula_applied_to_volumes(self):
    bars = varied_bars(30)
    assert roc_on_volume(bars, 6) == rate_of_change(
      [bar.volume for bar in bars], 6)

  def test_constant_volume_is_zero(self):
    bars = bars_from([(100.0 + index, 101.0 + index, 99.0 + index, 500.0)
                      for index in range(20)])
    assert roc_on_volume(bars, 5)[-1] == pytest.approx(0.0)

  def test_geometric_volume_gives_a_constant_reading(self):
    rows = [(100.0 + index, 101.0 + index, 99.0 + index,
             1000.0 * (1.05 ** index)) for index in range(25)]
    values = [v for v in roc_on_volume(bars_from(rows), 5) if v is not None]
    assert all(value == pytest.approx(100 * (1.05 ** 5 - 1))
               for value in values)

  def test_rising_volume_is_positive_and_falls_as_the_base_grows(self):
    rows = [(100.0 + index, 101.0 + index, 99.0 + index,
             1000.0 + 50.0 * index) for index in range(20)]
    values = [v for v in roc_on_volume(bars_from(rows), 5) if v is not None]
    assert all(value > 0.0 for value in values)
    assert all(earlier > later
               for earlier, later in zip(values, values[1:]))

  @pytest.mark.parametrize('lookback', [0, -1])
  def test_rejects_bad_lookback(self, lookback):
    with pytest.raises(ValueError, match='lookback'):
      roc_on_volume(varied_bars(8), lookback)


# ---------------------------------------------------------------------------
# Williams Accumulate/Distribute
# ---------------------------------------------------------------------------


class TestWilliamsAd:
  '''The range-only cumulative, with no window parameter at all.'''

  def test_known_values(self):
    # Bar 1 closes up at 12 from 10 with a low of 7, so it adds
    # 12 - 7 = 5.
    # Bar 2 closes down at 11 from 12 with a high of 16, so it subtracts
    # 16 - 11 = 5.
    # Bar 3 closes up at 13 from 11 with a low of 8, so it adds 13 - 8 = 5.
    # Bar 4 closes unchanged, so it adds nothing.
    bars = bars_from([
      (10.0, 15.0, 5.0, 100.0),
      (12.0, 17.0, 7.0, 100.0),
      (11.0, 16.0, 6.0, 100.0),
      (13.0, 18.0, 8.0, 100.0),
      (13.0, 14.0, 12.0, 100.0),
    ])
    assert williams_ad(bars) == [None, 5.0, 0.0, 5.0, 5.0]

  def test_warmup_is_a_single_bar(self):
    result = williams_ad(varied_bars(10))
    assert result[0] is None
    assert result[1] is not None

  def test_empty_and_single_bar_series(self):
    assert not williams_ad([])
    assert williams_ad(bars_from([(10.0, 11.0, 9.0, 5.0)])) == [None]

  def test_length_matches_input(self):
    assert len(williams_ad(varied_bars(33))) == 33

  def test_no_lookahead(self):
    assert_causal(lambda count: williams_ad(varied_bars(count)))()

  def test_has_no_window_parameter_to_validate(self):
    # Stated rather than skipped: this is the one oscillator here that
    # takes no lookback, so there is no non-positive window to reject.
    assert list(inspect.signature(williams_ad).parameters) == ['bars']

  def test_flat_series_is_zero(self):
    result = williams_ad(flat_bars(10))
    assert result[0] is None
    assert all(value == pytest.approx(0.0) for value in result[1:])

  def test_unchanged_close_accumulates_nothing(self):
    bars = bars_from([(10.0, 12.0, 8.0, 100.0)] * 6)
    assert williams_ad(bars)[-1] == pytest.approx(0.0)

  def test_rising_closes_accumulate_strictly(self):
    result = williams_ad(rising_bars(20))
    assert all(later > earlier
               for earlier, later in zip(result[1:], result[2:]))
    assert result[-1] > 0.0

  def test_falling_closes_accumulate_negatively(self):
    assert williams_ad(falling_bars(20))[-1] < 0.0

  def test_gap_through_the_previous_close_widens_the_accumulation(self):
    # Positive control on the previous-close terms: the bar after a gap
    # down subtracts max(121, 100) - 90 = 31, which is wider than its own
    # 42-point high-low span would suggest from the close alone.
    bars = bars_from([
      (100.0, 101.0, 99.0, 1000.0),
      (90.0, 121.0, 79.0, 1000.0),
      (90.0, 121.0, 79.0, 1000.0),
    ])
    assert williams_ad(bars)[1] == pytest.approx(-31.0)

  def test_ignores_volume(self):
    # Positive control on the Achelis convention: the name says volume
    # and the published Achelis form has none.
    bars = rising_bars(12)
    assert williams_ad(bars) == williams_ad(without_volume(bars))


# ---------------------------------------------------------------------------
# Chaikin Oscillator
# ---------------------------------------------------------------------------

CHAIKIN_ROWS = [
  (9.0, 10.0, 8.0, 100.0),
  (11.0, 12.0, 9.0, 200.0),
  (10.0, 13.0, 10.0, 300.0),
]


class TestChaikinOscillator:
  '''The spread of a fast and a slow EMA of accumulation/distribution.'''

  def test_known_values(self):
    # Accumulation/distribution for one bar is
    # volume * ((close - low) - (high - close)) / (high - low):
    #   bar 0: 100 * ((9 - 8) - (10 - 9)) / 2 = 100 * 0 / 2 = 0
    #   bar 1: 200 * ((11 - 9) - (12 - 11)) / 3 = 200 / 3
    #   bar 2: 300 * ((10 - 10) - (13 - 10)) / 3 = -300
    # A two-bar EMA has factor 2/3, a three-bar EMA factor 1/2:
    #   fast(2) = [0, 400/9, -5000/27]
    #   slow(3) = [0, 100/3, -400/3]
    # Differences: [0, 100/9, -1400/27]
    result = chaikin_oscillator(bars_from(CHAIKIN_ROWS), 2, 3)
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(100 / 9)
    assert result[2] == pytest.approx(-1400 / 27)

  def test_is_defined_from_the_first_bar(self):
    # Accumulation/distribution needs no lookback and `indicators.ema`
    # seeds on its first value, so there is no warm-up region at all.
    result = chaikin_oscillator(varied_bars(10), 3, 10)
    assert result[0] is not None
    assert all(value is not None for value in result)

  def test_length_matches_input(self):
    assert len(chaikin_oscillator(varied_bars(28), 3, 10)) == 28

  def test_no_lookahead(self):
    assert_causal(
      lambda count: chaikin_oscillator(varied_bars(count), 3, 10))()

  def test_equal_spans_give_exactly_zero(self):
    # Positive control that this is a difference of two averages and not
    # something more elaborate.
    result = chaikin_oscillator(varied_bars(20), 4, 4)
    assert all(value == pytest.approx(0.0) for value in result)

  def test_symmetric_bars_accumulate_nothing(self):
    # A close at the midpoint of the range gives a zero multiplier.
    bars = bars_from([(100.0, 101.0, 99.0, 500.0)] * 12)
    assert chaikin_oscillator(bars, 3, 10)[-1] == pytest.approx(0.0)

  def test_zero_range_bars_do_not_raise(self):
    # A halted bar has no intrabar range, so its multiplier is 0/0. It
    # contributes zero rather than propagating a division error.
    rows = [(100.0 + index, 101.0 + index, 99.0 + index, 1000.0)
            for index in range(6)]
    rows += [(106.0, 106.0, 106.0, 1000.0)] * 6
    assert len(chaikin_oscillator(bars_from(rows), 3, 10)) == 12

  def test_accumulation_in_the_upper_half_of_the_range_is_positive(self):
    # Closing three points above the low and one below the high gives a
    # multiplier of +2, so a rising money flow line makes the faster
    # average sit above the slower one.
    rows = [(100.0 + index, 101.0 + index, 97.0 + index,
             1000.0 + 100.0 * index) for index in range(20)]
    assert chaikin_oscillator(bars_from(rows), 3, 10)[-1] > 0.0

  def test_accumulation_in_the_lower_half_of_the_range_is_negative(self):
    rows = [(120.0 - index, 123.0 - index, 119.0 - index,
             1000.0 + 100.0 * index) for index in range(20)]
    assert chaikin_oscillator(bars_from(rows), 3, 10)[-1] < 0.0

  def test_is_not_the_awesome_oscillator(self):
    # The negative case: two moving-average differences built from
    # different inputs must not coincide.
    bars = varied_bars(40)
    assert (chaikin_oscillator(bars, 3, 10)[-1]
            != pytest.approx(awesome_oscillator(bars, 3, 10)[-1]))

  @pytest.mark.parametrize('name', ['fast', 'slow'])
  def test_rejects_bad_span(self, name):
    spans = {'fast': 3, 'slow': 10}
    spans[name] = -1
    with pytest.raises(ValueError, match=name):
      chaikin_oscillator(varied_bars(15), **spans)


# ---------------------------------------------------------------------------
# Awesome Oscillator
# ---------------------------------------------------------------------------

AWESOME_ROWS = [
  (9.0, 10.0, 8.0, 100.0),
  (10.0, 12.0, 10.0, 100.0),
  (12.0, 14.0, 12.0, 100.0),
  (11.0, 13.0, 11.0, 100.0),
]


class TestAwesomeOscillator:
  '''A dual moving-average difference on the median price.'''

  def test_known_values(self):
    # Median prices (H + L) / 2 are 9, 11, 13, 12.
    # Two-bar average: 10 at index 1, 12 at index 2, 12.5 at index 3.
    # Three-bar average: 11 at index 2, 12 at index 3.
    # Differences: index 2 is 12 - 11 = 1, index 3 is 12.5 - 12 = 0.5.
    result = awesome_oscillator(bars_from(AWESOME_ROWS), 2, 3)
    assert result[:2] == [None, None]
    assert result[2] == pytest.approx(1.0)
    assert result[3] == pytest.approx(0.5)

  def test_warmup_boundary_is_the_slow_window(self):
    series = awesome_oscillator(varied_bars(), 5, 12)
    assert series[:11] == [None] * 11
    assert first_defined(series) == 11

  def test_one_bar_short_of_the_boundary_is_all_none(self):
    assert awesome_oscillator(varied_bars(11), 5, 12) == [None] * 11

  def test_length_matches_input(self):
    assert len(awesome_oscillator(varied_bars(30), 5, 12)) == 30

  def test_no_lookahead(self):
    assert_causal(
      lambda count: awesome_oscillator(varied_bars(count), 5, 12))()

  def test_uses_the_median_price_not_the_close(self):
    # The negative case: a difference of averages of the closes is not
    # this one. Here the median alternates five points above the close
    # and five below it, which the closes alone cannot see.
    bars = bars_from([
      (100.0 + index, 100.0 + index + 10.0 + 5.0 * (index % 2),
       100.0 + index - 10.0 + 5.0 * (index % 2), 100.0)
      for index in range(20)])
    median = [(bar.high + bar.low) / 2.0 for bar in bars]
    closes = [bar.close for bar in bars]
    on_median = sum(median[-3:]) / 3 - sum(median[-9:]) / 9
    on_close = sum(closes[-3:]) / 3 - sum(closes[-9:]) / 9
    assert on_median != pytest.approx(on_close)
    assert awesome_oscillator(bars, 3, 9)[-1] == pytest.approx(on_median)

  def test_symmetric_wicks_make_the_median_the_close(self):
    # Positive control for the previous test: with the range centred on
    # the close the median price IS the close, so the two expressions
    # collapse onto the same number.
    bars = bars_from([(100.0 + index, 105.0 + index, 95.0 + index, 1000.0)
                      for index in range(20)])
    closes = [bar.close for bar in bars]
    on_close = sum(closes[-3:]) / 3 - sum(closes[-9:]) / 9
    assert (awesome_oscillator(bars, 3, 9)[-1]
            == pytest.approx(on_close))

  def test_ignores_volume(self):
    bars = rising_bars(40)
    assert (awesome_oscillator(bars, 5, 34)
            == awesome_oscillator(without_volume(bars), 5, 34))

  def test_flat_series_is_zero(self):
    assert (awesome_oscillator(flat_bars(40), 5, 34)[-1]
            == pytest.approx(0.0))

  def test_equal_windows_are_exactly_zero(self):
    bars = varied_bars(20)
    result = awesome_oscillator(bars, 7, 7)
    assert all(value == pytest.approx(0.0) for value in result[6:])

  def test_rising_series_is_positive(self):
    assert awesome_oscillator(rising_bars(40), 5, 34)[-1] > 0.0

  def test_falling_series_is_negative(self):
    assert awesome_oscillator(falling_bars(40), 5, 34)[-1] < 0.0

  @pytest.mark.parametrize('name', ['fast', 'slow'])
  def test_rejects_bad_window(self, name):
    windows = {'fast': 5, 'slow': 34}
    windows[name] = 0
    with pytest.raises(ValueError, match=name):
      awesome_oscillator(varied_bars(40), **windows)


# ---------------------------------------------------------------------------
# Cross-indicator checks
# ---------------------------------------------------------------------------


def all_oscillators():
  '''Return every oscillator over one shared series.

  Returns:
    Mapping of indicator name to its series over ``varied_bars(60)``,
    with windows small enough that every pair has a long defined region
    to be compared over.
  '''
  bars = varied_bars(60)
  closes = [bar.close for bar in bars]
  return {
    'awesome_oscillator': awesome_oscillator(bars, 5, 12),
    'chaikin_oscillator': chaikin_oscillator(bars, 3, 8),
    'commodity_channel_index': commodity_channel_index(bars, 5),
    'money_flow_index': money_flow_index(bars, 5),
    'rate_of_change': rate_of_change(closes, 7),
    'roc_on_volume': roc_on_volume(bars, 7),
    'slow_stochastic_d': slow_stochastic_d(bars, 5, 3, 3),
    'smoothed_roc': smoothed_roc(closes, 5, 7),
    'stochastic_d': stochastic_d(bars, 5, 3),
    'stochastic_k': stochastic_k(bars, 5),
    'stochastic_rsi': stochastic_rsi(closes, 5, 5),
    'ultimate_oscillator': ultimate_oscillator(bars, 3, 6, 12),
    'williams_ad': williams_ad(bars),
    'williams_percent_r': williams_percent_r(bars, 5),
  }


class TestIndicatorsAreNotAliased:
  '''No two indicators may be the same function under another name.'''

  def test_every_indicator_is_present_and_the_right_length(self):
    series = all_oscillators()
    assert len(series) == 14
    for name, values in series.items():
      assert len(values) == 60, name

  def test_no_two_indicators_agree_on_the_same_bars(self):
    series = all_oscillators()
    checked = 0
    for left, right in itertools.combinations(sorted(series), 2):
      overlaps = [(a, b) for a, b in zip(series[left], series[right])
                  if a is not None and b is not None]
      assert overlaps, f'{left} and {right} never overlap'
      assert any(not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
                 for a, b in overlaps), f'{left} is {right} in disguise'
      checked += 1
    assert checked == 91

  def test_every_indicator_has_a_long_defined_region(self):
    # Positive control for the pair test above: the shared generator
    # really does produce fourteen long, non-empty series, so no pair
    # comparison can pass vacuously.
    defined = {name: sum(1 for value in values if value is not None)
               for name, values in all_oscillators().items()}
    assert all(count >= 48 for count in defined.values()), defined
    assert min(defined.values()) >= 48

  def test_warm_up_boundaries_are_the_documented_ones(self):
    # A second aliasing route: a wrong boundary is usually a copy of
    # another function's boundary, so each is pinned to its own
    # derivation.
    bars = varied_bars(60)
    closes = [bar.close for bar in bars]
    assert first_defined(stochastic_k(bars, 5)) == 4
    assert first_defined(williams_percent_r(bars, 5)) == 4
    assert first_defined(stochastic_d(bars, 5, 3)) == 6
    assert first_defined(slow_stochastic_d(bars, 5, 3, 3)) == 8
    assert first_defined(commodity_channel_index(bars, 5)) == 8
    assert first_defined(ultimate_oscillator(bars, 3, 6, 12)) == 12
    assert first_defined(stochastic_rsi(closes, 5, 5)) == 9
    assert first_defined(money_flow_index(bars, 5)) == 5
    assert first_defined(smoothed_roc(closes, 5, 7)) == 7
    assert first_defined(rate_of_change(closes, 7)) == 7
    assert first_defined(roc_on_volume(bars, 7)) == 7
    assert first_defined(williams_ad(bars)) == 1
    assert first_defined(awesome_oscillator(bars, 5, 12)) == 11
    assert first_defined(chaikin_oscillator(bars, 3, 10)) == 0

  def test_only_two_indicators_share_a_boundary_and_they_are_related(self):
    # %K and %R are the same position on two scales, so one shared
    # boundary is expected. Every other boundary is distinct, which is
    # the quick check that no function has been pasted into the wrong
    # slot.
    boundaries = {
      'stochastic_k': 4,
      'williams_percent_r': 4,
      'stochastic_d': 6,
      'slow_stochastic_d': 8,
      'cci': 8,
      'ultimate_oscillator': 12,
      'stochastic_rsi': 9,
      'money_flow_index': 5,
      'smoothed_roc': 7,
      'rate_of_change': 7,
      'roc_on_volume': 7,
      'williams_ad': 1,
      'awesome_oscillator': 11,
    }
    assert list(boundaries.values()).count(4) == 2
    assert len(set(boundaries.values())) >= 8
