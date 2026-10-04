#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for performance and risk statistics.'''

import math
from random import Random

import pytest

from stock_rl.metrics import (
  TRADING_DAYS_PER_YEAR,
  calmar_ratio,
  deflated_sharpe,
  drawdown_curve,
  max_drawdown,
  sharpe_ratio,
  sortino_ratio,
  total_return,
  turnover,
)

# Hand-checkable series. Mean 0.02, sample stdev 0.0081650, so the
# unannualised Sharpe is 0.02 / 0.0081650 = 2.4494897.
SIMPLE = [0.02, 0.01, 0.03, 0.02]

# Equity path: rises to 1.2, falls to 0.8 (a 33.33% decline from 1.2),
# then recovers above 1.2 at index 5.
EQUITY = [1.0, 1.2, 0.9, 1.1, 0.8, 1.5]


def series(length: int, seed: int = 7) -> list[float]:
  '''Return a deterministic return series of the requested length.

  Args:
    length: Number of returns.
    seed: Seed for the private Random instance.

  Returns:
    List of returns, reproducible across runs.
  '''
  rng = Random(seed)
  return [rng.gauss(0.0004, 0.011) for _ in range(length)]


class TestSharpe:
  '''Annualisation and degenerate-input behaviour.'''

  def test_known_value_without_annualisation(self):
    assert sharpe_ratio(SIMPLE, periods=1) == pytest.approx(2.4494897)

  def test_annualisation_scales_by_root_periods(self):
    raw = sharpe_ratio(SIMPLE, periods=1)
    annual = sharpe_ratio(SIMPLE, periods=TRADING_DAYS_PER_YEAR)
    assert annual == pytest.approx(raw * math.sqrt(252))

  def test_risk_free_is_subtracted(self):
    with_free = sharpe_ratio(SIMPLE, risk_free=0.02, periods=1)
    assert with_free == pytest.approx(0.0)

  def test_zero_dispersion_is_zero_not_infinite(self):
    assert sharpe_ratio([0.01, 0.01, 0.01]) == 0.0

  def test_single_observation_is_zero(self):
    assert sharpe_ratio([0.01]) == 0.0

  def test_empty_is_zero(self):
    assert sharpe_ratio([]) == 0.0


class TestSortino:
  '''Only downside moves are charged as risk.'''

  def test_gains_only_gives_zero(self):
    assert sortino_ratio([0.01, 0.02, 0.03]) == 0.0

  def test_upside_not_penalised(self):
    # A series with large upside and small downside must score better
    # than the same series with the upside removed.
    steady = [0.01, 0.01, -0.01, 0.01]
    spiky = [0.05, 0.05, -0.01, 0.05]
    assert sortino_ratio(spiky, periods=1) > sortino_ratio(
      steady, periods=1)


class TestDrawdown:
  '''Peak, trough and recovery identification.'''

  def test_known_depth_and_indices(self):
    result = max_drawdown(EQUITY)
    assert result.depth == pytest.approx(1.0 - 0.8 / 1.2)
    assert result.peak_index == 1
    assert result.trough_index == 4

  def test_recovery_index_is_found(self):
    assert max_drawdown(EQUITY).recovery_index == 5

  def test_unrecovered_drawdown_has_no_recovery(self):
    result = max_drawdown([1.0, 0.5, 0.6, 0.7])
    assert result.recovery_index is None

  def test_monotonic_rise_has_no_drawdown(self):
    assert max_drawdown([1.0, 1.1, 1.2]).depth == 0.0

  def test_empty_curve(self):
    result = max_drawdown([])
    assert result.depth == 0.0
    assert result.peak_index == -1

  def test_blowup_leaves_peak_guarded(self):
    # A -100% return takes equity to zero, which would divide by zero in
    # the running-peak calculation. The curve must survive that.
    result = drawdown_curve([1.0, 0.0, 0.0])
    assert all(math.isfinite(value) for value in result)
    assert result[1] == pytest.approx(1.0)

  def test_negative_equity_is_not_a_drawdown(self):
    # Once equity is negative there is no meaningful peak ratio, so the
    # depth must not explode to infinity.
    result = drawdown_curve([-1.0, -2.0])
    assert all(math.isfinite(value) for value in result)

  def test_max_drawdown_skips_negative_peaks(self):
    assert math.isfinite(max_drawdown([-1.0, -2.0]).depth)

  def test_curve_matches_drawdown_curve(self):
    curve = drawdown_curve(EQUITY)
    assert curve[0] == 0.0
    assert curve[2] == pytest.approx(1.0 - 0.9 / 1.2)
    assert curve[4] == pytest.approx(1.0 - 0.8 / 1.2)

  def test_drawdown_never_negative(self):
    assert all(value >= 0.0 for value in drawdown_curve(EQUITY))


class TestTotalReturn:
  '''Compounding of periodic returns.'''

  def test_compounds_multiplicatively(self):
    assert total_return([0.1, 0.1]) == pytest.approx(1.21)

  def test_flat_series_is_flat(self):
    assert total_return([0.0, 0.0]) == pytest.approx(1.0)

  def test_empty_is_flat(self):
    assert total_return([]) == pytest.approx(1.0)


class TestCalmar:
  '''Return per unit of worst pain.'''

  def test_zero_drawdown_is_zero(self):
    assert calmar_ratio([0.01, 0.02, 0.03]) == 0.0

  def test_empty_is_zero(self):
    assert calmar_ratio([]) == 0.0

  def test_deep_drawdown_lowers_ratio(self):
    calm = [0.01] * 200 + [-0.05]
    rough = [0.01] * 200 + [-0.60]
    assert calmar_ratio(rough) < calmar_ratio(calm)


class TestTurnover:
  '''One-way turnover measured from a flat starting book.'''

  def test_opening_a_position_costs_one(self):
    assert turnover([1.0]) == pytest.approx(1.0)

  def test_holding_still_costs_nothing(self):
    assert turnover([0.5, 0.5, 0.5]) == pytest.approx(0.5)

  def test_full_exit_costs_one_more(self):
    assert turnover([1.0, 0.0]) == pytest.approx(2.0)

  def test_empty_is_zero(self):
    assert turnover([]) == 0.0


class TestDeflatedSharpe:
  '''Honest accounting for having searched many configurations.'''

  def test_requires_more_than_one_year(self):
    # 252 daily returns is exactly one year, and the statistic needs
    # T > 1, so this must report 0.0 rather than a misleading number.
    assert deflated_sharpe(series(252), trials=10) == 0.0

  def test_two_years_is_reported(self):
    value = deflated_sharpe(series(504), trials=10)
    assert 0.0 <= value <= 1.0

  def test_more_trials_never_helps(self):
    returns = series(504)
    few = deflated_sharpe(returns, trials=2)
    many = deflated_sharpe(returns, trials=500)
    assert many <= few

  def test_too_few_trials_is_zero(self):
    assert deflated_sharpe(series(504), trials=1) == 0.0

  def test_empty_is_zero(self):
    assert deflated_sharpe([], trials=10) == 0.0

  def test_no_dispersion_has_no_search_threshold(self):
    # A perfectly flat series has zero deviation, so there is no
    # search-induced Sharpe to beat.
    assert deflated_sharpe([0.0] * 504, trials=10) == 0.0

  def test_search_inflates_the_threshold(self):
    # A strategy that beats a low bar should beat a high one, so the
    # probability of clearing a 200-config search must not exceed the
    # probability of clearing a 2-config search on identical returns.
    returns = series(504, seed=11)
    assert (deflated_sharpe(returns, trials=200)
            <= deflated_sharpe(returns, trials=2) + 1e-12)
