#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the volatility and channel indicators.

Three things are checked for every function, unconditionally, because
all three have been bugs in this repository before:

1. the output length equals the input length, on every input;
2. the warm-up region is ``None`` and the exact first defined index is
   asserted rather than eyeballed;
3. every division has its zero-denominator branch exercised, with a
   positive control beside it proving the branch is the only difference.

The hand-computed cases are worked on short series with the arithmetic
written out, so a reviewer can check the number without running the
code. ``indicators.py`` once returned six values for five closes; a test
that only checked "is the last value right" would not have caught it.
'''

from datetime import datetime, timedelta

import pytest

from stock_rl.bars import Bar
from stock_rl.indicators_volatility import (
  atr_bands,
  bollinger_bandwidth,
  bollinger_bands,
  bollinger_percent_b,
  choppiness_index,
  coefficient_of_variation,
  keltner_channels,
  vertical_horizontal_filter,
)

START = datetime(2026, 1, 1)

#: Closes chosen so the arithmetic is checkable by hand. Window 3 at
#: index 2 averages [10, 12, 14] to 12 with a sample standard deviation
#: of sqrt(((10-12)^2 + 0 + (14-12)^2) / 2) = sqrt(4) = 2.
CLOSES = [10.0, 12.0, 14.0, 11.0, 13.0]

#: A bar with a fixed 2.0 intrabar range and a prior close of 100, so the
#: true range is max(101 - 99, |101 - 100|, |99 - 100|) = 2.0 exactly.
#: High 101, low 99, close 100 on every bar.
STEADY = (100.0, 101.0, 99.0, 100.0)


def bars_from(rows, volume=1000.0):
  '''Return bars from ``(open, high, low, close)`` tuples.

  Args:
    rows: Sequence of price tuples.
    volume: Traded quantity stamped on every bar.

  Returns:
    List of bars one day apart, in ascending time order.
  '''
  return [
    Bar(START + timedelta(days=index), o, h, low, c, volume)
    for index, (o, h, low, c) in enumerate(rows)
  ]


class TestNoLookAhead:
  '''The value at index ``i`` may only use bars up to ``i``.

  One test over every function rather than one per function: rewriting
  the final bar must leave every earlier value byte-identical. A function
  that peeks forward passes every "is the last value right" assertion and
  fails this one.
  '''

  def test_rewriting_the_last_bar_leaves_earlier_values_alone(self):
    closes = [10.0, 12.0, 14.0, 11.0, 13.0, 15.0, 17.0]
    rows = [(c, c + 2.0, c - 2.0, c + 1.0) for c in closes]
    bars = bars_from(rows)
    before = _all_series(closes, bars)
    bumped_rows = list(rows)
    bumped_rows[-1] = (99.0, 101.0, 97.0, 100.0)
    after = _all_series(closes, bars_from(bumped_rows))
    for original, changed in zip(before, after, strict=True):
      assert original[:-1] == changed[:-1]
    # Positive control: the final value does change for a bar-reading
    # function, so the test above is not passing because the mutation did
    # nothing. The close-only functions are unchanged at the last index
    # by construction, since this mutation rewrites bars, not closes.
    before_chop = choppiness_index(bars, 3)
    after_chop = choppiness_index(bars_from(bumped_rows), 3)
    assert before_chop[-1] != after_chop[-1]


def _all_series(closes, bars):
  '''Return every volatility output, flattened to a list of series.

  Args:
    closes: Closing prices in ascending time order.
    bars: Bars matching those closes.

  Returns:
    List of series, each the length of the input.
  '''
  outputs = [
    bollinger_bands(closes, 3),
    bollinger_bandwidth(closes, 3),
    bollinger_percent_b(closes, 3),
    keltner_channels(bars, 2, 3),
    atr_bands(bars, 2),
    choppiness_index(bars, 3),
    vertical_horizontal_filter(closes, 3),
    coefficient_of_variation(closes, 3),
  ]
  return [one for output in outputs
          for one in (output if isinstance(output, tuple) else [output])]


class TestLengthContract:
  '''Every series must match its input length on every input length.

  ``indicators.py`` once returned six values for five closes, which is why
  this is one test over every function and every length from 0 to 11
  rather than a length assertion per function that a new function could
  simply forget.
  '''

  def test_every_series_matches_its_input(self):
    for length in range(12):
      closes = [10.0 + index for index in range(length)]
      bars = bars_from([
        (10.0 + i, 12.0 + i, 8.0 + i, 11.0 + i) for i in range(length)
      ])
      outputs = [
        bollinger_bands(closes, 3),
        bollinger_bandwidth(closes, 3),
        bollinger_percent_b(closes, 3),
        keltner_channels(bars, 2, 3),
        atr_bands(bars, 2),
        choppiness_index(bars, 3),
        vertical_horizontal_filter(closes, 3),
        coefficient_of_variation(closes, 3),
      ]
      for output in outputs:
        series = output if isinstance(output, tuple) else [output]
        for one in series:
          assert len(one) == length


class TestBollingerBands:
  '''Middle, upper and lower bands.'''

  def test_every_band_matches_input_length(self):
    middle, upper, lower = bollinger_bands(CLOSES, 3)
    assert len(middle) == len(CLOSES)
    assert len(upper) == len(CLOSES)
    assert len(lower) == len(CLOSES)

  def test_empty_series_gives_three_empty_bands(self):
    assert bollinger_bands([], 3) == ([], [], [])

  def test_warmup_boundary_is_window_minus_one(self):
    middle, _, _ = bollinger_bands(CLOSES, 3)
    assert middle[:2] == [None, None]
    assert middle[2] is not None
    assert all(value is not None for value in middle[2:])

  def test_hand_computed_bands(self):
    middle, upper, lower = bollinger_bands(CLOSES, 3, 2.0)
    # mean([10, 12, 14]) = 12; sample stdev = 2; half-width = 2 * 2 = 4.
    assert middle[2] == pytest.approx(12.0)
    assert upper[2] == pytest.approx(16.0)
    assert lower[2] == pytest.approx(8.0)

  def test_bands_are_ordered_around_the_middle(self):
    _, upper, lower = bollinger_bands(CLOSES, 3, 2.0)
    for top, bottom in zip(upper, lower, strict=True):
      if top is not None:
        assert top >= bottom

  def test_constant_input_collapses_the_bands(self):
    # A constant series has zero dispersion, so the bands collapse onto
    # the middle. That is the correct degenerate reading, not None.
    middle, upper, lower = bollinger_bands([7.0] * 5, 3)
    assert middle[2:] == [7.0] * 3
    assert upper[2:] == [7.0] * 3
    assert lower[2:] == [7.0] * 3

  def test_wider_deviations_widen_the_bands(self):
    _, tight, _ = bollinger_bands(CLOSES, 3, 1.0)
    _, wide, _ = bollinger_bands(CLOSES, 3, 3.0)
    assert tight[2] < wide[2]

  def test_uses_only_past_closes(self):
    original = bollinger_bands(CLOSES, 3)
    mutated = list(CLOSES)
    mutated[0] = 999.0
    assert bollinger_bands(mutated, 3)[0][3:] == original[0][3:]

  @pytest.mark.parametrize('window', [0, 1, -2])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      bollinger_bands(CLOSES, window)

  @pytest.mark.parametrize('deviations', [0.0, -1.0])
  def test_rejects_non_positive_deviations(self, deviations):
    with pytest.raises(ValueError, match='deviations'):
      bollinger_bands(CLOSES, 3, deviations)


class TestBollingerBandwidth:
  '''Band spread over the middle band.'''

  def test_length_matches_input(self):
    assert len(bollinger_bandwidth(CLOSES, 3)) == len(CLOSES)

  def test_warmup_boundary_is_window_minus_one(self):
    result = bollinger_bandwidth(CLOSES, 3)
    assert result[:2] == [None, None]
    assert all(value is not None for value in result[2:])

  def test_hand_computed_value(self):
    # index 2: upper 16, lower 8, middle 12, so (16 - 8) / 12 = 2/3.
    assert bollinger_bandwidth(CLOSES, 3, 2.0)[2] == pytest.approx(2 / 3)

  def test_constant_input_is_zero_not_none(self):
    # No dispersion means no spread; zero is the correct reading.
    result = bollinger_bandwidth([10.0] * 5, 3)
    assert result[2:] == [0.0, 0.0, 0.0]

  def test_zero_middle_band_is_none(self):
    # [-10, 0, 10] averages to exactly 0, so the ratio has no value.
    # Returning 0.0 or 1.0 here would both be invented readings.
    result = bollinger_bandwidth([-10.0, 0.0, 10.0], 3)
    assert result == [None, None, None]

  def test_zero_middle_has_a_positive_control_alongside(self):
    # Same shape, shifted up by 100: mean 100, stdev 10, half-width 20,
    # so (120 - 80) / 100 = 0.4. Only the zero mean differs.
    shifted = bollinger_bandwidth([90.0, 100.0, 110.0], 3)
    assert shifted[2] is not None
    assert shifted[2] == pytest.approx(0.4)

  @pytest.mark.parametrize('window', [0, 1])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      bollinger_bandwidth(CLOSES, window)


class TestBollingerPercentB:
  '''Where the close sits inside the band.'''

  def test_length_matches_input(self):
    assert len(bollinger_percent_b(CLOSES, 3)) == len(CLOSES)

  def test_warmup_boundary_is_window_minus_one(self):
    result = bollinger_percent_b(CLOSES, 3)
    assert result[:2] == [None, None]
    assert all(value is not None for value in result[2:])

  def test_hand_computed_value(self):
    # index 2: close 14, lower 8, spread 8, so (14 - 8) / 8 = 0.75.
    assert bollinger_percent_b(CLOSES, 3, 2.0)[2] == pytest.approx(0.75)

  def test_a_close_below_the_middle_scores_below_half(self):
    # Window [14, 11, 11]: mean 12, sample stdev
    # sqrt(((14-12)^2 + (11-12)^2 + (11-12)^2) / 2) = sqrt(3) = 1.73205.
    # lower = 12 - 2 * 1.73205 = 8.53590, spread = 4 * 1.73205 = 6.92820,
    # so (11 - 8.53590) / 6.92820 = 0.35566.
    result = bollinger_percent_b([10.0, 12.0, 14.0, 11.0, 11.0], 3, 2.0)
    assert result[4] == pytest.approx(0.3556624, abs=1e-6)
    assert result[4] < 0.5

  def test_zero_spread_window_is_none(self):
    # A flat window makes upper == lower, so 0/0. Reporting 0.0 or 0.5
    # would claim the close is at the floor, or dead centre, of a band
    # that does not exist.
    assert bollinger_percent_b([10.0] * 5, 3) == [None] * 5

  def test_zero_spread_has_a_positive_control_alongside(self):
    # One bar of movement in the window is enough to define the spread.
    mixed = bollinger_percent_b([10.0, 10.0, 12.0, 12.0], 3, 2.0)
    assert mixed[3] is not None

  @pytest.mark.parametrize('window', [0, 1])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      bollinger_percent_b(CLOSES, window)


class TestKeltnerChannels:
  '''EMA centre line with an ATR envelope.'''

  def test_length_matches_input(self):
    bars = bars_from([STEADY] * 6)
    assert len(keltner_channels(bars, 2, 2)[0]) == len(bars)

  def test_middle_has_no_warmup_because_the_ema_seeds_immediately(self):
    middle, _, _ = keltner_channels(bars_from([STEADY] * 6), 2, 2)
    assert all(value is not None for value in middle)

  def test_envelope_warmup_boundary_is_the_atr_window(self):
    _, upper, lower = keltner_channels(bars_from([STEADY] * 6), 2, 3)
    assert upper[:3] == [None, None, None]
    assert lower[:3] == [None, None, None]
    assert upper[3] is not None

  def test_hand_computed_channels(self):
    # ATR of the steady bar is 2.0, EMA of a constant 100 close is 100,
    # so upper = 100 + 2 * 2 = 104 and lower = 100 - 2 * 2 = 96.
    _, upper, lower = keltner_channels(bars_from([STEADY] * 6), 2, 2, 2.0)
    assert upper[2] == pytest.approx(104.0)
    assert lower[2] == pytest.approx(96.0)

  def test_atr_window_shifts_the_envelope_not_the_middle(self):
    bars = bars_from([STEADY] * 8)
    middle, upper, lower = keltner_channels(bars, 3, 5)
    assert middle[0] is not None
    assert upper[4] is None
    assert lower[4] is None
    assert upper[5] is not None

  def test_a_bigger_atr_gap_widens_the_envelope(self):
    rows = [(100.0, 110.0, 90.0, 100.0)] * 6
    _, upper, _ = keltner_channels(bars_from(rows), 2, 2, 2.0)
    # True range here is max(20, 10, 10) = 20, so upper is 100 + 40.
    assert upper[2] == pytest.approx(140.0)

  @pytest.mark.parametrize('ema_window', [0, -1])
  def test_rejects_bad_ema_window(self, ema_window):
    with pytest.raises(ValueError, match='ema_window'):
      keltner_channels(bars_from([STEADY] * 4), ema_window)

  @pytest.mark.parametrize('atr_window', [0, -1])
  def test_rejects_bad_atr_window(self, atr_window):
    with pytest.raises(ValueError, match='atr_window'):
      keltner_channels(bars_from([STEADY] * 4), 2, atr_window)

  @pytest.mark.parametrize('multiple', [0.0, -1.0])
  def test_rejects_non_positive_multiple(self, multiple):
    with pytest.raises(ValueError, match='multiple'):
      keltner_channels(bars_from([STEADY] * 4), 2, 2, multiple)


class TestAtrBands:
  '''Extreme-plus-ATR envelope.'''

  def test_length_matches_input(self):
    bars = bars_from([STEADY] * 6)
    assert len(atr_bands(bars, 2)[0]) == len(bars)

  def test_warmup_boundary_is_the_window(self):
    upper, lower = atr_bands(bars_from([STEADY] * 6), 3)
    assert upper[:3] == [None, None, None]
    assert lower[:3] == [None, None, None]
    assert upper[3] is not None

  def test_hand_computed_bands(self):
    # Extreme high 101 plus ATR 2.0; extreme low 99 minus ATR 2.0.
    upper, lower = atr_bands(bars_from([STEADY] * 6), 2, 1.0)
    assert upper[2] == pytest.approx(103.0)
    assert lower[2] == pytest.approx(97.0)

  def test_zero_multiple_collapses_onto_the_extremes(self):
    upper, lower = atr_bands(bars_from([STEADY] * 6), 2, 0.0)
    assert upper[2] == pytest.approx(101.0)
    assert lower[2] == pytest.approx(99.0)

  def test_short_series_is_all_none(self):
    upper, lower = atr_bands(bars_from([STEADY]), 5)
    assert upper == [None]
    assert lower == [None]

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      atr_bands(bars_from([STEADY] * 4), window)

  def test_rejects_negative_multiple(self):
    with pytest.raises(ValueError, match='multiple'):
      atr_bands(bars_from([STEADY] * 4), 2, -0.5)


class TestChoppinessIndex:
  '''Bill Dreiss' index, including its zero-range window.'''

  def test_length_matches_input(self):
    bars = bars_from([STEADY] * 6)
    assert len(choppiness_index(bars, 3)) == len(bars)

  def test_warmup_boundary_is_window_minus_one(self):
    result = choppiness_index(bars_from([STEADY] * 6), 3)
    assert result[:2] == [None, None]
    assert result[2] is not None

  def test_hand_computed_value(self):
    # Two bars, each of range 4, so the summed range is 8. Extremes are
    # high 13 and low 8, giving a span of 5. Therefore
    # 100 * log10(8 / 5) / log10(2) = 100 * 0.204120 / 0.301030 = 67.80719.
    bars = bars_from([(10.0, 12.0, 8.0, 10.0), (11.0, 13.0, 9.0, 11.0)])
    assert choppiness_index(bars, 2)[1] == pytest.approx(67.8071905)

  def test_a_constant_range_series_is_the_maximum(self):
    # STEADY is a constant bar of range 2, so a 3-bar window sums 6 over
    # a span of 2. 100 * log10(3) / log10(3) = 100, which is the
    # arithmetic ceiling of the formula: the summed range cannot exceed
    # window times the span.
    result = choppiness_index(bars_from([STEADY] * 6), 3)
    assert result[2:] == pytest.approx([100.0] * 4)

  def test_zero_range_window_is_none(self):
    # Every bar has high == low, so the summed range is zero and
    # log10(0) is undefined. Neither 0.0 (maximally choppy) nor 100.0
    # (perfectly trending) is a true statement about an untraded window.
    bars = bars_from([(5.0, 5.0, 5.0, 5.0)] * 5)
    assert choppiness_index(bars, 2) == [None] * 5

  def test_zero_range_has_a_positive_control_alongside(self):
    # One bar of range in the window is enough to make it defined.
    bars = bars_from([
      (5.0, 5.0, 5.0, 5.0),
      (10.0, 12.0, 9.0, 11.0),
    ])
    assert choppiness_index(bars, 2)[1] is not None

  def test_partially_flat_window_still_computes(self):
    # Only the summed range matters, not every bar having one.
    bars = bars_from([
      (5.0, 5.0, 5.0, 5.0),
      (10.0, 12.0, 9.0, 11.0),
      (11.0, 12.0, 10.0, 11.5),
    ])
    result = choppiness_index(bars, 3)
    assert result[2] is not None

  def test_is_not_clamped_to_the_published_scale(self):
    # Two disjoint ranges of 1 each sum to 2, against a span of
    # 100 - 9 = 91. 100 * log10(2 / 91) / log10(2) = -550.779. The
    # formula says so; flooring it to zero would report maximal
    # choppiness for a window that was not.
    bars = bars_from([(9.0, 10.0, 9.0, 9.5), (99.0, 100.0, 99.0, 99.5)])
    assert choppiness_index(bars, 2)[1] == pytest.approx(-550.7794640)

  def test_more_trading_of_the_same_range_raises_the_index(self):
    # Quiet window: three bars of range 2 stepping up, so sum 6 over a
    # span of 13 - 9 = 4. 100 * log10(1.5) / log10(3) = 36.9070.
    quiet = bars_from([
      (10.0, 11.0, 9.0, 10.0),
      (11.0, 12.0, 10.0, 11.0),
      (12.0, 13.0, 11.0, 12.0),
    ])
    # Busy window: three bars of range 8 overlapping one span of 8, so
    # sum 24 over a span of 8. 100 * log10(3) / log10(3) = 100.
    busy = bars_from([
      (10.0, 14.0, 6.0, 12.0),
      (10.0, 14.0, 6.0, 8.0),
      (10.0, 14.0, 6.0, 12.0),
    ])
    assert choppiness_index(quiet, 3)[2] == pytest.approx(36.9070246)
    assert choppiness_index(busy, 3)[2] == pytest.approx(100.0)
    assert choppiness_index(busy, 3)[2] > choppiness_index(quiet, 3)[2]

  def test_does_not_read_the_future(self):
    # Rewriting the LAST bar must leave every earlier value identical:
    # the value at index i may only use bars up to i.
    rows = [(10.0, 12.0, 8.0, 10.0), (11.0, 13.0, 9.0, 11.0),
            (11.0, 15.0, 10.0, 14.0)]
    original = choppiness_index(bars_from(rows), 2)
    mutated = list(rows)
    mutated[-1] = (900.0, 901.0, 899.0, 900.0)
    assert choppiness_index(bars_from(mutated), 2)[:-1] == original[:-1]

  @pytest.mark.parametrize('window', [0, 1, -3])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      choppiness_index(bars_from([STEADY] * 4), window)


class TestVerticalHorizontalFilter:
  '''Adam White's filter.'''

  def test_length_matches_input(self):
    assert len(vertical_horizontal_filter(CLOSES, 2)) == len(CLOSES)

  def test_warmup_boundary_is_window_not_window_minus_one(self):
    # Summing window absolute changes needs window + 1 closes, so the
    # first defined index is one later than a plain rolling window.
    result = vertical_horizontal_filter([10.0, 12.0, 14.0, 16.0], 2)
    assert result[:2] == [None, None]
    assert result[2] is not None

  def test_hand_computed_value(self):
    # Window [12, 14]: reach 14 - 12 = 2. Travel |12 - 10| + |14 - 12| = 4.
    # 100 * 2 / 4 = 50.
    result = vertical_horizontal_filter([10.0, 12.0, 14.0, 16.0], 2)
    assert result[2] == pytest.approx(50.0)

  def test_zero_travel_is_none(self):
    # A window with no net movement makes the denominator zero.
    result = vertical_horizontal_filter([10.0, 10.0, 10.0, 10.0], 2)
    assert result == [None] * 4

  def test_zero_travel_has_a_positive_control_alongside(self):
    moving = vertical_horizontal_filter([10.0, 12.0, 14.0], 2)
    assert moving[2] is not None

  def test_a_round_trip_scores_lower_than_a_clean_run(self):
    trending = vertical_horizontal_filter([10.0, 11.0, 12.0, 13.0], 3)
    round_trip = vertical_horizontal_filter([10.0, 13.0, 10.0, 13.0], 3)
    assert trending[3] > round_trip[3]

  def test_window_of_one_is_the_pure_step(self):
    result = vertical_horizontal_filter([10.0, 12.0, 14.0], 1)
    # Reach 0 over a single close, travel |12 - 10| = 2, so 100 * 0 / 2.
    assert result[1] == pytest.approx(0.0)
    assert result[0] is None

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      vertical_horizontal_filter(CLOSES, window)


class TestCoefficientOfVariation:
  '''Stdev over mean.'''

  def test_length_matches_input(self):
    assert len(coefficient_of_variation(CLOSES, 3)) == len(CLOSES)

  def test_warmup_boundary_is_window_minus_one(self):
    result = coefficient_of_variation(CLOSES, 3)
    assert result[:2] == [None, None]
    assert result[2] is not None

  def test_hand_computed_value(self):
    # Window [10, 20]: mean 15, sample stdev sqrt((25 + 25) / 1) = 7.0711.
    # 7.0711 / 15 = 0.4714.
    result = coefficient_of_variation([10.0, 20.0, 30.0], 2)
    assert result[1] == pytest.approx(0.4714045, abs=1e-6)

  def test_constant_input_is_zero_not_none(self):
    result = coefficient_of_variation([7.0] * 5, 3)
    assert result[2:] == [0.0, 0.0, 0.0]

  def test_zero_mean_is_none(self):
    # [-10, 10] averages to exactly zero, so the ratio has no value.
    result = coefficient_of_variation([-10.0, 10.0, 20.0], 2)
    assert result[1] is None

  def test_zero_mean_has_a_positive_control_alongside(self):
    # The very next window, [10, 20], has a positive mean and computes.
    result = coefficient_of_variation([-10.0, 10.0, 20.0], 2)
    assert result[2] == pytest.approx(0.4714045, abs=1e-6)

  def test_a_negative_mean_returns_the_signed_ratio(self):
    result = coefficient_of_variation([-30.0, -20.0], 2)
    assert result[1] < 0.0

  def test_scaling_the_series_leaves_the_ratio_unchanged(self):
    base = coefficient_of_variation(CLOSES, 3)
    scaled = coefficient_of_variation([value * 100 for value in CLOSES], 3)
    assert scaled[2:] == pytest.approx(base[2:])

  @pytest.mark.parametrize('window', [0, 1, -1])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      coefficient_of_variation(CLOSES, window)
