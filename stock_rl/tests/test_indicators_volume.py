#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the volume indicators.

Every function is checked three ways unconditionally, because all three
have been bugs in this repository before: the output length equals the
input length on every input, the exact first defined index is asserted
rather than eyeballed, and every division has its zero-denominator branch
exercised with a positive control beside it.

The recursive accumulators get extra attention. ``indicators.py`` once
returned six values for five closes, and an accumulator that starts one
bar early looks exactly like a correct one on a rising series. Each of
OBV, PVT, PVI and NVI therefore has its seed and its first qualifying bar
pinned by an explicit index assertion.
'''

import inspect
from datetime import datetime, timedelta

import pytest

from stock_rl.bars import Bar
from stock_rl.indicators_volume import (
  chaikin_money_flow,
  ease_of_movement,
  elder_ray,
  force_index,
  negative_volume_index,
  on_balance_volume,
  positive_volume_index,
  price_volume_trend,
  volume_oscillator,
)

START = datetime(2026, 1, 1)

#: Five bars chosen so every arithmetic path is checkable by hand. The
#: close alternates up, down, up, down and the volume alternates heavy,
#: light, heavy, light, so both volume indices get a qualifying bar and a
#: non-qualifying one.
ROWS = [
  (10.0, 11.0, 9.0, 10.5, 100.0),
  (10.5, 12.0, 10.0, 11.5, 200.0),
  (11.5, 12.0, 10.5, 10.8, 150.0),
  (10.8, 13.0, 10.7, 12.9, 300.0),
  (12.9, 13.5, 12.0, 12.1, 120.0),
]


def bars_from(rows):
  '''Return bars from ``(open, high, low, close, volume)`` tuples.

  Args:
    rows: Sequence of bar tuples.

  Returns:
    List of bars one day apart, in ascending time order.
  '''
  return [
    Bar(START + timedelta(days=index), *row)
    for index, row in enumerate(rows)
  ]


#: A bar with no intrabar range at all. Every indicator that divides by
#: the range must treat this as undefined.
RANGE_LESS = (5.0, 5.0, 5.0, 5.0, 100.0)


class TestNoLookAhead:
  '''The value at index ``i`` may only use bars up to ``i``.

  One test over every function rather than one per function: rewriting
  the final bar must leave every earlier value byte-identical. A function
  that peeks forward passes every "is the last value right" assertion and
  fails this one.
  '''

  def test_rewriting_the_last_bar_leaves_earlier_values_alone(self):
    before = _all_series(bars_from(ROWS))
    mutated = list(ROWS)
    mutated[-1] = (99.0, 101.0, 97.0, 100.0, 999.0)
    after = _all_series(bars_from(mutated))
    for original, changed in zip(before, after, strict=True):
      assert original[:-1] == changed[:-1]
    # Positive control: the final value does change, so the test above is
    # not passing because the mutation did nothing.
    assert before[0][-1] != after[0][-1]


def _all_series(bars):
  '''Return every volume output, flattened to a list of series.

  Args:
    bars: Price bars in ascending time order.

  Returns:
    List of series, each the length of the input.
  '''
  outputs = [
    on_balance_volume(bars),
    price_volume_trend(bars),
    positive_volume_index(bars),
    negative_volume_index(bars),
    chaikin_money_flow(bars, 3),
    ease_of_movement(bars, 3),
    force_index(bars, 3),
    elder_ray(bars, 3),
    volume_oscillator(bars, 2, 3),
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
      bars = bars_from([
        (10.0 + i, 12.0 + i, 8.0 + i, 11.0 + i, 100.0 + i)
        for i in range(length)
      ])
      outputs = [
        on_balance_volume(bars),
        price_volume_trend(bars),
        positive_volume_index(bars),
        negative_volume_index(bars),
        chaikin_money_flow(bars, 3),
        ease_of_movement(bars, 3),
        force_index(bars, 3),
        elder_ray(bars, 3),
        volume_oscillator(bars, 2, 3),
      ]
      for output in outputs:
        series = output if isinstance(output, tuple) else [output]
        for one in series:
          assert len(one) == length


class TestWindowlessFunctions:
  '''The four accumulators take no window, so there is none to validate.

  Every other function in this module raises on a non-positive window.
  These four integrate from the first bar with no lookback parameter at
  all, so pinning that here stops a future window argument arriving
  unvalidated.
  '''

  def test_none_of_them_accepts_a_window(self):
    for function in (on_balance_volume, price_volume_trend,
                     positive_volume_index, negative_volume_index):
      assert list(inspect.signature(function).parameters) == ['bars']


class TestOnBalanceVolume:
  '''Granville's additive accumulator.'''

  def test_length_matches_input(self):
    assert len(on_balance_volume(bars_from(ROWS))) == len(ROWS)

  def test_length_matches_input_on_an_empty_series(self):
    # Empty in, empty out: the seed is only emitted when a bar exists.
    assert on_balance_volume([]) == pytest.approx([])

  def test_index_zero_is_the_zero_seed_not_none(self):
    result = on_balance_volume(bars_from(ROWS))
    assert result[0] == pytest.approx(0.0)
    assert result[0] is not None

  def test_no_warmup_region_at_all(self):
    assert all(value is not None for value in on_balance_volume(
      bars_from(ROWS)))

  def test_hand_computed_running_total(self):
    # index 0: seed 0.
    # index 1: close 10.5 -> 11.5 rose, add 200, total 200.
    # index 2: close 11.5 -> 10.8 fell, subtract 150, total 50.
    # index 3: close 10.8 -> 12.9 rose, add 300, total 350.
    # index 4: close 12.9 -> 12.1 fell, subtract 120, total 230.
    result = on_balance_volume(bars_from(ROWS))
    assert result == pytest.approx([0.0, 200.0, 50.0, 350.0, 230.0])

  def test_an_unchanged_close_leaves_the_total_alone(self):
    rows = [
      (10.0, 11.0, 9.0, 10.0, 100.0),
      (10.0, 11.0, 9.0, 10.0, 500.0),
    ]
    assert on_balance_volume(bars_from(rows)) == [0.0, 0.0]

  def test_constant_input_is_flat_zero(self):
    rows = [(5.0, 6.0, 4.0, 5.0, 100.0)] * 4
    assert on_balance_volume(bars_from(rows)) == [0.0] * 4

  def test_zero_volume_has_a_positive_control_alongside(self):
    # Zero volume contributes exactly zero, which is arithmetically
    # right: nothing traded, so nothing accumulated. The next bar with
    # real volume shows the accumulator is still live.
    quiet = on_balance_volume(bars_from([
      (10.0, 11.0, 9.0, 10.0, 0.0),
      (10.0, 11.0, 9.0, 11.0, 0.0),
      (11.0, 12.0, 10.0, 12.0, 400.0),
    ]))
    assert quiet == pytest.approx([0.0, 0.0, 400.0])


class TestPriceVolumeTrend:
  '''The volume-scaled additive accumulator.'''

  def test_length_matches_input(self):
    assert len(price_volume_trend(bars_from(ROWS))) == len(ROWS)

  def test_index_zero_is_the_zero_seed_not_none(self):
    result = price_volume_trend(bars_from(ROWS))
    assert result[0] == pytest.approx(0.0)
    assert result[0] is not None

  def test_hand_computed_running_total(self):
    # index 1: 0 + 200 * (11.5 - 10.5) / 10.5 = 200 / 10.5 = 19.047619.
    # index 2: + 150 * (10.8 - 11.5) / 11.5 = - 9.130435, total 9.917184.
    # index 3: + 300 * (12.9 - 10.8) / 10.8 = 58.333333, total 68.250517.
    result = price_volume_trend(bars_from(ROWS))
    assert result[1] == pytest.approx(19.0476190)
    assert result[2] == pytest.approx(9.9171842)
    assert result[3] == pytest.approx(68.2505175)

  def test_a_zero_prior_close_is_none_and_freezes_the_total(self):
    rows = [
      (0.0, 1.0, 0.0, 0.0, 100.0),
      (0.0, 2.0, 0.0, 1.0, 200.0),
      (1.0, 3.0, 1.0, 2.0, 300.0),
    ]
    result = price_volume_trend(bars_from(rows))
    # index 1 cannot be computed: there is no usable prior close.
    assert result[1] is None
    # index 2 resumes from the frozen total of 0.0, not from index 1's
    # unknown increment: 0 + 300 * (2 - 1) / 1 = 300.
    assert result[2] == pytest.approx(300.0)

  def test_zero_prior_close_has_a_positive_control_alongside(self):
    positive = price_volume_trend(bars_from([
      (1.0, 2.0, 1.0, 1.0, 100.0),
      (1.0, 3.0, 1.0, 2.0, 300.0),
    ]))
    assert positive[1] == pytest.approx(300.0)

  def test_constant_close_is_flat_zero(self):
    rows = [(5.0, 6.0, 4.0, 5.0, 100.0)] * 4
    result = price_volume_trend(bars_from(rows))
    assert result == pytest.approx([0.0, 0.0, 0.0, 0.0])

  def test_does_not_read_the_future(self):
    original = price_volume_trend(bars_from(ROWS))
    mutated = list(ROWS)
    mutated[-1] = (12.9, 13.5, 12.0, 99.0, 120.0)
    assert price_volume_trend(bars_from(mutated))[:-1] == original[:-1]


class TestPositiveVolumeIndex:
  '''Fosback's multiplicative accumulator.'''

  def test_length_matches_input(self):
    assert len(positive_volume_index(bars_from(ROWS))) == len(ROWS)

  def test_length_matches_input_on_an_empty_series(self):
    assert positive_volume_index([]) == []

  def test_boundary_is_the_first_qualifying_bar(self):
    # Volume goes 100 -> 200 at index 1, the first rise. Index 0 has no
    # prior bar and index 1 is the first bar that qualifies.
    result = positive_volume_index(bars_from(ROWS))
    assert result[0] is None
    assert result[1] is not None
    assert all(value is not None for value in result[1:])

  def test_hand_computed_level(self):
    # index 1: 100 * (11.5 / 10.5) = 109.523810.
    # index 2: volume fell, so the level is carried: 109.523810.
    # index 3: 109.523810 * (12.9 / 10.8) = 130.820106.
    # index 4: volume fell again, carried: 130.820106.
    result = positive_volume_index(bars_from(ROWS))
    assert result[1] == pytest.approx(109.5238095)
    assert result[2] == pytest.approx(109.5238095)
    assert result[3] == pytest.approx(130.8201058)

  def test_a_never_rising_volume_series_is_all_none(self):
    rows = [
      (10.0, 11.0, 9.0, 10.0, 300.0),
      (10.0, 11.0, 9.0, 11.0, 200.0),
      (11.0, 12.0, 10.0, 12.0, 100.0),
    ]
    assert positive_volume_index(bars_from(rows)) == [None] * 3

  def test_a_zero_close_never_annihilates_the_level(self):
    # Volume rises on every bar, so every bar qualifies. Index 1 has a
    # zero close, which would make the ratio zero and pin the index at
    # zero forever, and index 2 has a zero *prior* close so its ratio is
    # undefined too. Both are None. Index 3 has positive closes on both
    # sides and restarts the index from the base at 100 * (4 / 2) = 200
    # rather than inheriting a destroyed zero.
    rows = [
      (10.0, 11.0, 9.0, 10.0, 100.0),
      (0.0, 1.0, 0.0, 0.0, 200.0),
      (1.0, 3.0, 1.0, 2.0, 250.0),
      (2.0, 5.0, 2.0, 4.0, 300.0),
    ]
    result = positive_volume_index(bars_from(rows))
    assert result[1] is None
    assert result[2] is None
    assert result[3] == pytest.approx(200.0)

  def test_zero_close_freezes_then_resumes_from_the_last_level(self):
    rows = [
      (10.0, 11.0, 9.0, 10.0, 100.0),
      (10.0, 12.0, 10.0, 11.0, 200.0),
      (0.0, 1.0, 0.0, 0.0, 300.0),
      (0.0, 2.0, 0.0, 2.0, 400.0),
      (2.0, 4.0, 2.0, 4.0, 500.0),
    ]
    result = positive_volume_index(bars_from(rows))
    assert result[1] == pytest.approx(110.0)
    # The zero-numerator bar is undefined and does not move the level.
    assert result[2] is None
    # Its own ratio is 2 / 0, undefined, so that bar is None too.
    assert result[3] is None
    # Volume rose at index 4 and both closes are positive, so the level
    # resumes from 110 rather than restarting at the base.
    assert result[4] == pytest.approx(110.0 * 4.0 / 2.0)

  def test_does_not_read_the_future(self):
    original = positive_volume_index(bars_from(ROWS))
    mutated = list(ROWS)
    mutated[-1] = (12.9, 13.5, 12.0, 99.0, 999.0)
    assert positive_volume_index(bars_from(mutated))[:-1] == original[:-1]


class TestNegativeVolumeIndex:
  '''The mirror of the positive volume index.'''

  def test_length_matches_input(self):
    assert len(negative_volume_index(bars_from(ROWS))) == len(ROWS)

  def test_boundary_is_the_first_qualifying_bar(self):
    # Volume goes 100 -> 200 at index 1 (not a fall), then 200 -> 150 at
    # index 2, the first fall. Index 1 must therefore be None even
    # though the positive volume index is defined there.
    result = negative_volume_index(bars_from(ROWS))
    assert result[0] is None
    assert result[1] is None
    assert result[2] is not None

  def test_hand_computed_level(self):
    # index 2: 100 * (10.8 / 11.5) = 93.913043.
    # index 3: volume rose, carried: 93.913043.
    # index 4: 93.913043 * (12.1 / 12.9) = 88.088979.
    result = negative_volume_index(bars_from(ROWS))
    assert result[2] == pytest.approx(93.9130435)
    assert result[3] == pytest.approx(93.9130435)
    assert result[4] == pytest.approx(88.0889788)

  def test_a_never_falling_volume_series_is_all_none(self):
    rows = [
      (10.0, 11.0, 9.0, 10.0, 100.0),
      (10.0, 11.0, 9.0, 11.0, 200.0),
      (11.0, 12.0, 10.0, 12.0, 300.0),
    ]
    assert negative_volume_index(bars_from(rows)) == [None] * 3

  def test_the_two_indices_move_on_opposite_bars(self):
    # Volume rises at index 1 and falls at index 2, so the positive index
    # moves at 1 and holds at 2, and the negative index is still undefined
    # at 1 and moves at 2. Comparing the level against the prior bar is
    # the direct test of which bar qualified.
    rising = positive_volume_index(bars_from(ROWS))
    falling = negative_volume_index(bars_from(ROWS))
    assert rising[1] != rising[0]
    assert rising[2] == rising[1]
    assert falling[1] is None
    assert falling[2] != falling[1]


class TestChaikinMoneyFlow:
  '''Money flow volume over volume.'''

  def test_length_matches_input(self):
    assert len(chaikin_money_flow(bars_from(ROWS), 2)) == len(ROWS)

  def test_warmup_boundary_is_window_minus_one(self):
    result = chaikin_money_flow(bars_from(ROWS), 3)
    assert result[:2] == [None, None]
    assert result[2] is not None

  def test_hand_computed_value(self):
    # bar 0: multiplier ((10.5 - 9) - (11 - 10.5)) / 2 = 0.5, flow 50.
    # bar 1: multiplier ((11.5 - 10) - (12 - 11.5)) / 2 = 0.5, flow 100.
    # window total: 150 / 300 = 0.5.
    result = chaikin_money_flow(bars_from(ROWS), 2)
    assert result[1] == pytest.approx(0.5)

  def test_a_close_at_the_high_scores_one(self):
    rows = [
      (10.0, 12.0, 8.0, 12.0, 100.0),
      (10.0, 12.0, 8.0, 12.0, 100.0),
    ]
    # multiplier ((12 - 8) - (12 - 12)) / 4 = 1 for both bars.
    assert chaikin_money_flow(bars_from(rows), 2)[1] == pytest.approx(1.0)

  def test_a_close_at_the_low_scores_minus_one(self):
    rows = [
      (10.0, 12.0, 8.0, 8.0, 100.0),
      (10.0, 12.0, 8.0, 8.0, 100.0),
    ]
    assert chaikin_money_flow(bars_from(rows), 2)[1] == pytest.approx(-1.0)

  def test_stays_inside_minus_one_to_one(self):
    for value in chaikin_money_flow(bars_from(ROWS), 3):
      if value is not None:
        assert -1.0 <= value <= 1.0

  def test_a_rangeless_bar_is_dropped_from_both_sums(self):
    # The window holds one usable bar and one zero-range bar. The usable
    # bar closes at its high, so its multiplier is 1 and the reading is
    # 100 / 100 = 1.0. The rangeless bar's volume is excluded too, which
    # is why it is not 100 / 200.
    rows = [
      (10.0, 12.0, 8.0, 12.0, 100.0),
      RANGE_LESS,
    ]
    assert chaikin_money_flow(bars_from(rows), 2)[1] == pytest.approx(1.0)

  def test_a_window_of_only_rangeless_bars_is_none(self):
    rows = [RANGE_LESS] * 3
    assert chaikin_money_flow(bars_from(rows), 2) == [None] * 3

  def test_a_window_with_no_volume_is_none(self):
    # Bar.volume is documented as possibly zero for illiquid sessions,
    # so the summed-volume denominator can genuinely vanish.
    rows = [
      (10.0, 12.0, 8.0, 11.0, 0.0),
      (10.0, 12.0, 8.0, 11.0, 0.0),
    ]
    assert chaikin_money_flow(bars_from(rows), 2) == [None, None]

  def test_zero_volume_has_a_positive_control_alongside(self):
    rows = [
      (10.0, 12.0, 8.0, 11.0, 0.0),
      (10.0, 12.0, 8.0, 11.0, 500.0),
    ]
    assert chaikin_money_flow(bars_from(rows), 2)[1] is not None

  def test_one_unusable_bar_does_not_poison_the_whole_series(self):
    rows = [
      (10.0, 12.0, 8.0, 11.0, 100.0),
      RANGE_LESS,
      (10.0, 12.0, 8.0, 12.0, 100.0),
    ]
    result = chaikin_money_flow(bars_from(rows), 1)
    assert result[1] is None
    assert result[2] == pytest.approx(1.0)

  def test_does_not_read_the_future(self):
    original = chaikin_money_flow(bars_from(ROWS), 2)
    mutated = list(ROWS)
    mutated[-1] = (12.9, 13.5, 12.0, 12.1, 120.0)
    assert chaikin_money_flow(bars_from(mutated), 2)[:-1] == original[:-1]

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      chaikin_money_flow(bars_from(ROWS), window)


class TestEaseOfMovement:
  '''Distance moved over box ratio.'''

  def test_length_matches_input(self):
    assert len(ease_of_movement(bars_from(ROWS), 2)) == len(ROWS)

  def test_warmup_boundary_is_window_not_window_minus_one(self):
    # Index 0 has no prior bar, and the average needs `window` raw
    # readings, so the first defined index is `window`.
    result = ease_of_movement(bars_from(ROWS), 2)
    assert result[:2] == [None, None]
    assert result[2] is not None

  def test_hand_computed_value(self):
    # bar 1 raw: midpoints 11 and 10, distance 1; span 2; volume 200.
    # 1 * 2 * 2 / 200 = 0.02.
    # bar 2 raw: midpoints 11.25 and 11, distance 0.25; span 1.5; vol 150.
    # 0.25 * 1.5 * 2 / 150 = 0.005.
    # window mean: (0.02 + 0.005) / 2 = 0.0125.
    result = ease_of_movement(bars_from(ROWS), 2)
    assert result[2] == pytest.approx(0.0125)

  def test_a_rising_market_gives_a_positive_reading(self):
    rows = [
      (10.0, 12.0, 8.0, 10.0, 100.0),
      (10.0, 13.0, 9.0, 12.0, 100.0),
      (12.0, 15.0, 11.0, 14.0, 100.0),
    ]
    assert ease_of_movement(bars_from(rows), 2)[2] > 0.0

  def test_a_falling_market_gives_a_negative_reading(self):
    # Ease of Movement follows the bar MIDPOINT, not the close, so these
    # bars close flat to unchanged while the midpoint falls on both legs:
    # 10 -> 9.5 -> 9. Distances -0.5 and -0.5, spans 3 and 2, so the raw
    # readings are -0.5 * 3 * 2 / 100 = -0.03 and -0.5 * 2 * 2 / 100 =
    # -0.02, and the window mean is -0.025.
    rows = [
      (12.0, 12.0, 8.0, 12.0, 100.0),
      (11.0, 11.0, 8.0, 11.0, 100.0),
      (10.0, 10.0, 8.0, 10.0, 100.0),
    ]
    result = ease_of_movement(bars_from(rows), 2)
    assert result[2] == pytest.approx(-0.025)

  def test_a_rangeless_bar_makes_its_windows_none(self):
    rows = [
      (10.0, 12.0, 8.0, 10.0, 100.0),
      RANGE_LESS,
      (10.0, 13.0, 9.0, 12.0, 100.0),
    ]
    result = ease_of_movement(bars_from(rows), 2)
    assert result[2] is None
    assert result[1] is None

  def test_a_zero_volume_bar_makes_its_windows_none(self):
    rows = [
      (10.0, 12.0, 8.0, 10.0, 100.0),
      (10.0, 13.0, 9.0, 12.0, 0.0),
      (12.0, 15.0, 11.0, 14.0, 100.0),
    ]
    assert ease_of_movement(bars_from(rows), 2)[2] is None

  def test_each_zero_denominator_has_a_positive_control_alongside(self):
    rangeless = ease_of_movement(bars_from([
      (10.0, 12.0, 8.0, 10.0, 100.0),
      RANGE_LESS,
      (10.0, 13.0, 9.0, 12.0, 100.0),
    ]), 2)
    ranged = ease_of_movement(bars_from([
      (10.0, 12.0, 8.0, 10.0, 100.0),
      (10.0, 13.0, 9.0, 12.0, 100.0),
      (12.0, 15.0, 11.0, 14.0, 100.0),
    ]), 2)
    assert rangeless[2] is None
    assert ranged[2] is not None

  def test_window_of_one_is_the_raw_reading(self):
    rows = [
      (10.0, 12.0, 8.0, 10.0, 100.0),
      (10.0, 13.0, 9.0, 12.0, 200.0),
    ]
    # distance 1, span 4, window 1, volume 200: 1 * 4 * 1 / 200 = 0.02.
    assert ease_of_movement(bars_from(rows), 1)[1] == pytest.approx(0.02)

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      ease_of_movement(bars_from(ROWS), window)


class TestForceIndex:
  '''Elder's product, smoothed.'''

  def test_length_matches_input(self):
    assert len(force_index(bars_from(ROWS), 2)) == len(ROWS)

  def test_only_index_zero_is_none(self):
    result = force_index(bars_from(ROWS), 2)
    assert result[0] is None
    assert all(value is not None for value in result[1:])

  def test_single_bar_series_is_all_none(self):
    assert force_index(bars_from([ROWS[0]]), 2) == [None]

  def test_length_matches_input_on_an_empty_series(self):
    # The leading None is for the missing prior bar, not a free-standing
    # placeholder: zero bars in means zero values out. Compared as a list
    # rather than by truthiness so the length is what is asserted.
    assert list(force_index([], 2)) == pytest.approx([])

  def test_hand_computed_value(self):
    # Raw force at index 1: (11.5 - 10.5) * 200 = 200. The EMA of span 2
    # has factor 2 / 3, and seeds on its first value, so index 1 is 200.
    result = force_index(bars_from(ROWS), 2)
    assert result[1] == pytest.approx(200.0)
    # index 2 raw: (10.8 - 11.5) * 150 = -105, so
    # -105 * 2/3 + 200 * 1/3 = -3.333333.
    assert result[2] == pytest.approx(-3.3333333)

  def test_constant_close_gives_flat_zero_not_none(self):
    # Zero volume means zero force, and that is arithmetically right:
    # nothing traded, so there was no force. It is not an undefined
    # value and must not be reported as one.
    rows = [(5.0, 6.0, 4.0, 5.0, 100.0)] * 4
    result = force_index(bars_from(rows), 2)
    assert result == [None, 0.0, 0.0, 0.0]

  def test_zero_volume_has_a_positive_control_alongside(self):
    rows = [
      (10.0, 11.0, 9.0, 10.0, 0.0),
      (10.0, 11.0, 9.0, 11.0, 0.0),
      (11.0, 12.0, 10.0, 12.0, 400.0),
    ]
    result = force_index(bars_from(rows), 2)
    assert result[1] == pytest.approx(0.0)
    assert result[2] > 0.0

  def test_a_rising_close_with_volume_is_positive(self):
    rows = [
      (10.0, 11.0, 9.0, 10.0, 100.0),
      (10.0, 12.0, 10.0, 12.0, 500.0),
    ]
    assert force_index(bars_from(rows), 2)[1] == pytest.approx(1000.0)

  def test_does_not_read_the_future(self):
    original = force_index(bars_from(ROWS), 2)
    mutated = list(ROWS)
    mutated[-1] = (12.9, 13.5, 12.0, 99.0, 120.0)
    assert force_index(bars_from(mutated), 2)[:-1] == original[:-1]

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      force_index(bars_from(ROWS), window)


class TestElderRay:
  '''Bull and bear power.'''

  def test_each_series_matches_input_length(self):
    bars = bars_from(ROWS)
    bull, bear = elder_ray(bars, 2)
    assert len(bull) == len(bars)
    assert len(bear) == len(bars)

  def test_no_warmup_region_at_all(self):
    bull, bear = elder_ray(bars_from(ROWS), 2)
    assert bull[0] is not None
    assert bear[0] is not None

  def test_hand_computed_values(self):
    # Span 2 gives a smoothing factor of 2/3. The EMA seeds on the first
    # close, 10.5, then gives 11.5 * 2/3 + 10.5 / 3 = 11.166667.
    bull, bear = elder_ray(bars_from(ROWS), 2)
    assert bull[0] == pytest.approx(11.0 - 10.5)
    assert bear[0] == pytest.approx(9.0 - 10.5)
    assert bull[1] == pytest.approx(12.0 - 11.1666667)
    assert bear[1] == pytest.approx(10.0 - 11.1666667)

  def test_bull_pressure_is_never_below_bear_pressure(self):
    for top, bottom in zip(*elder_ray(bars_from(ROWS), 2), strict=True):
      assert top >= bottom

  def test_a_constant_close_gives_zero_power(self):
    rows = [(5.0, 5.0, 5.0, 5.0, 100.0)] * 3
    bull, bear = elder_ray(bars_from(rows), 2)
    assert bull == [0.0, 0.0, 0.0]
    assert bear == [0.0, 0.0, 0.0]

  def test_empty_series_gives_two_empty_lists(self):
    assert elder_ray([], 2) == ([], [])

  def test_does_not_read_the_future(self):
    bull, bear = elder_ray(bars_from(ROWS), 2)
    mutated = list(ROWS)
    mutated[-1] = (12.9, 13.5, 12.0, 99.0, 120.0)
    moved_bull, moved_bear = elder_ray(bars_from(mutated), 2)
    assert moved_bull[:-1] == bull[:-1]
    assert moved_bear[:-1] == bear[:-1]

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      elder_ray(bars_from(ROWS), window)


class TestVolumeOscillator:
  '''The two volume means in ratio form.'''

  def test_length_matches_input(self):
    assert len(volume_oscillator(bars_from(ROWS), 2, 3)) == len(ROWS)

  def test_warmup_boundary_is_long_window_minus_one(self):
    result = volume_oscillator(bars_from(ROWS), 2, 3)
    assert result[:2] == [None, None]
    assert result[2] is not None

  def test_hand_computed_value(self):
    # Long mean of the first three volumes: (100 + 200 + 150) / 3 = 150.
    # Short mean of the last two: (200 + 150) / 2 = 175.
    # 100 * (175 - 150) / 150 = 16.666667.
    result = volume_oscillator(bars_from(ROWS), 2, 3)
    assert result[2] == pytest.approx(16.6666667)

  def test_equal_windows_are_flat_zero(self):
    result = volume_oscillator(bars_from(ROWS), 3, 3)
    assert result[2:] == [0.0, 0.0, 0.0]

  def test_constant_volume_is_flat_zero(self):
    rows = [(5.0, 6.0, 4.0, 5.0, 100.0)] * 4
    result = volume_oscillator(bars_from(rows), 2, 3)
    assert result[2:] == [0.0, 0.0]

  def test_a_zero_long_mean_is_none(self):
    rows = [
      (5.0, 6.0, 4.0, 5.0, 0.0),
      (5.0, 6.0, 4.0, 5.0, 0.0),
      (5.0, 6.0, 4.0, 5.0, 0.0),
    ]
    # Reporting 0.0 here would say "no change in volume activity" for a
    # window with no volume activity at all.
    assert volume_oscillator(bars_from(rows), 2, 3) == [None] * 3

  def test_zero_long_mean_has_a_positive_control_alongside(self):
    rows = [
      (5.0, 6.0, 4.0, 5.0, 0.0),
      (5.0, 6.0, 4.0, 5.0, 500.0),
      (5.0, 6.0, 4.0, 5.0, 500.0),
    ]
    assert volume_oscillator(bars_from(rows), 2, 3)[2] is not None

  def test_a_zero_short_mean_against_a_positive_long_one(self):
    rows = [
      (5.0, 6.0, 4.0, 5.0, 600.0),
      (5.0, 6.0, 4.0, 5.0, 0.0),
      (5.0, 6.0, 4.0, 5.0, 0.0),
    ]
    # Short mean 0 against a long mean of 200: 100 * (0 - 200) / 200.
    result = volume_oscillator(bars_from(rows), 1, 3)
    assert result[2] == pytest.approx(-100.0)

  def test_heavy_then_light_volume_is_negative(self):
    rows = [
      (5.0, 6.0, 4.0, 5.0, 900.0),
      (5.0, 6.0, 4.0, 5.0, 100.0),
      (5.0, 6.0, 4.0, 5.0, 50.0),
    ]
    assert volume_oscillator(bars_from(rows), 1, 3)[2] < 0.0

  def test_does_not_read_the_future(self):
    original = volume_oscillator(bars_from(ROWS), 2, 3)
    mutated = list(ROWS)
    mutated[-1] = (12.9, 13.5, 12.0, 12.1, 120.0)
    assert volume_oscillator(bars_from(mutated), 2, 3)[:-1] == original[:-1]

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_short_window(self, window):
    with pytest.raises(ValueError, match='short_window'):
      volume_oscillator(bars_from(ROWS), window, 3)

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_long_window(self, window):
    with pytest.raises(ValueError, match='long_window'):
      volume_oscillator(bars_from(ROWS), 2, window)

  def test_rejects_short_window_above_long_window(self):
    with pytest.raises(ValueError, match='short_window'):
      volume_oscillator(bars_from(ROWS), 5, 3)
