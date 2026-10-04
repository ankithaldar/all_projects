#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Tests for technical indicators.

Warm-up behaviour is tested as carefully as the values, because a
silently wrong indicator in the warm-up region is worse than a missing
one: it is arithmetically valid and would silently size a position.
'''

from datetime import datetime, timedelta

import pytest

from stock_rl.bars import Bar
from stock_rl.indicators import (
  atr,
  ema,
  momentum,
  realized_volatility,
  rsi,
  sma,
  trend_slope,
)

START = datetime(2026, 1, 1)


def bars_from(rows):
  '''Return bars from (open, high, low, close) tuples.

  Args:
    rows: Sequence of price tuples.

  Returns:
    List of bars one hour apart.
  '''
  return [
    Bar(START + timedelta(hours=index), o, h, low, c, 1.0)
    for index, (o, h, low, c) in enumerate(rows)
  ]


class TestSma:
  '''Simple moving average values and warm-up.'''

  def test_warmup_is_none_not_zero(self):
    result = sma([1, 2, 3, 4, 5], 2)
    assert result[0] is None
    assert all(value is not None for value in result[1:])

  def test_known_values(self):
    assert sma([1, 2, 3, 4, 5], 2) == [None, 1.5, 2.5, 3.5, 4.5]

  def test_window_of_one_is_identity(self):
    assert sma([3, 5, 9], 1) == [3.0, 5.0, 9.0]

  def test_window_longer_than_series_is_all_none(self):
    assert sma([1, 2], 5) == [None, None]

  def test_length_matches_input(self):
    values = list(range(10))
    assert len(sma(values, 3)) == len(values)

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      sma([1, 2, 3], window)


class TestEma:
  '''Exponential moving average.'''

  def test_known_values(self):
    # factor = 2 / (3 + 1) = 0.5
    result = ema([1, 2, 3], 3)
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(1.5)
    assert result[2] == pytest.approx(2.25)

  def test_seeds_at_first_value(self):
    assert ema([7, 8, 9], 5)[0] == pytest.approx(7.0)

  def test_tracks_a_rising_series(self):
    result = ema([float(value) for value in range(1, 21)], 5)
    assert result[-1] > result[4]

  @pytest.mark.parametrize('window', [0, -3])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      ema([1, 2, 3], window)


class TestMomentum:
  '''Trailing total return.'''

  def test_known_value(self):
    assert momentum([1, 2, 3, 4], 2) == pytest.approx(1.0)

  def test_flat_series_is_zero(self):
    assert momentum([10, 10, 10, 10], 3) == pytest.approx(0.0)

  def test_insufficient_history_is_none(self):
    assert momentum([1, 2], 5) is None

  def test_zero_base_is_none(self):
    assert momentum([0, 0, 0], 1) is None

  @pytest.mark.parametrize('lookback', [0, -2])
  def test_rejects_bad_lookback(self, lookback):
    with pytest.raises(ValueError, match='lookback'):
      momentum([1, 2, 3], lookback)


class TestRealizedVolatility:
  '''Annualised realised volatility.'''

  def test_warmup_is_none(self):
    result = realized_volatility([1.0, 1.01, 1.02, 1.03], 3)
    assert result[:3] == [None, None, None]
    assert result[3] is not None

  def test_constant_returns_give_zero(self):
    # A geometric series has constant per-bar returns, so the sample
    # stdev is zero and the volatility is zero.
    closes = [100.0 * (1.01 ** index) for index in range(30)]
    result = realized_volatility(closes, 10)
    assert result[-1] == pytest.approx(0.0, abs=1e-9)

  def test_more_dispersion_means_higher_volatility(self):
    steady = [100.0 * (1.001 ** index) for index in range(40)]
    jumpy = [100.0, 110.0, 95.0, 105.0, 98.0, 108.0, 94.0, 106.0]
    assert (realized_volatility(jumpy, 6)[-1]
            > realized_volatility(steady, 6)[-1])

  def test_length_matches_input(self):
    values = [100.0, 101.0, 99.0, 102.0, 98.0]
    assert len(realized_volatility(values, 3)) == len(values)

  @pytest.mark.parametrize('window', [1, 0])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      realized_volatility([1.0, 2.0], window)

  def test_rejects_bad_periods(self):
    with pytest.raises(ValueError, match='periods'):
      realized_volatility([1.0, 2.0], 2, periods=0)


class TestRsi:
  '''Wilder RSI, including the degenerate cases.'''

  def test_all_gains_is_one_hundred(self):
    closes = [float(value) for value in range(1, 40)]
    assert rsi(closes, 14)[-1] == pytest.approx(100.0)

  def test_all_losses_is_zero(self):
    closes = [float(value) for value in range(40, 1, -1)]
    assert rsi(closes, 14)[-1] == pytest.approx(0.0)

  def test_flat_series_is_neutral_fifty(self):
    # A flat series must not divide by zero. 50.0 is the neutral reading.
    result = rsi([10.0] * 30, 14)
    assert result[-1] == pytest.approx(50.0)

  def test_warmup_is_none(self):
    result = rsi([10.0, 11.0, 12.0, 13.0], 3)
    assert result[0] is None
    assert any(value is None for value in result[:3])

  def test_length_matches_input(self):
    closes = [10.0, 11.0, 10.5, 12.0, 11.5]
    assert len(rsi(closes, 3)) == len(closes)

  def test_short_series(self):
    assert len(rsi([1.0, 2.0], 14)) == 2

  def test_bounded_range(self):
    closes = [100.0, 102.0, 99.0, 103.0, 98.0, 101.0, 97.0, 104.0]
    for value in rsi(closes, 4):
      if value is not None:
        assert 0.0 <= value <= 100.0

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      rsi([1.0, 2.0], window)


class TestAtr:
  '''Wilder Average True Range.'''

  def test_constant_range(self):
    # Open == close == 100, high 101, low 99, so the high-low span is 2
    # and the cross-bar terms are only 1. True range is 2 throughout.
    rows = [(100.0, 101.0, 99.0, 100.0)] * 20
    assert atr(bars_from(rows), 5)[-1] == pytest.approx(2.0)

  def test_gap_up_increases_true_range(self):
    # A gap must register through the cross-bar terms, otherwise a gap
    # is indistinguishable from a quiet session.
    rows = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (110.0, 110.5, 109.5, 110.0),
        (110.0, 110.5, 109.5, 110.0),
    ]
    # Bar 2 gaps from a prior close of 100 to a low of 109.5, so true
    # range is 109.5 - 100 = 9.5, not the 1.0 intrabar span.
    assert atr(bars_from(rows), 2)[-1] > 2.0

  def test_warmup_is_none(self):
    rows = [(100.0, 101.0, 99.0, 100.0)] * 10
    result = atr(bars_from(rows), 4)
    assert result[0] is None
    assert any(value is None for value in result[:4])

  def test_length_matches_input(self):
    rows = [(100.0, 101.0, 99.0, 100.0)] * 8
    assert len(atr(bars_from(rows), 3)) == len(rows)

  def test_short_series(self):
    assert len(atr(bars_from([(1.0, 2.0, 0.5, 1.5)]), 14)) == 1

  @pytest.mark.parametrize('window', [0, -1])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      atr(bars_from([(1.0, 2.0, 0.5, 1.5)] * 3), window)


class TestTrendSlope:
  '''Normalised least-squares slope.'''

  def test_perfect_line(self):
    # y = 0,1,2,3: numerator 5, denominator 5, mean 1.5, slope 5/(5*1.5).
    assert trend_slope([0.0, 1.0, 2.0, 3.0], 4) == pytest.approx(2 / 3)

  def test_normalised_by_level(self):
    # The same shape scaled by 100 must give the same normalised slope.
    base = trend_slope([1.0, 2.0, 3.0, 4.0], 4)
    scaled = trend_slope([100.0, 200.0, 300.0, 400.0], 4)
    assert base == pytest.approx(scaled)

  def test_downward_trend_is_negative(self):
    assert trend_slope([4.0, 3.0, 2.0, 1.0], 4) < 0.0

  def test_flat_series_is_zero_not_none(self):
    # Zero slope is a meaningful "no trend" reading, not an undefined
    # one, so it must survive as 0.0.
    assert trend_slope([5.0, 5.0, 5.0, 5.0], 4) == pytest.approx(0.0)

  def test_insufficient_history_is_none(self):
    assert trend_slope([1.0, 2.0], 5) is None

  @pytest.mark.parametrize('window', [1, 0])
  def test_rejects_bad_window(self, window):
    with pytest.raises(ValueError, match='window'):
      trend_slope([1.0, 2.0, 3.0], window)
