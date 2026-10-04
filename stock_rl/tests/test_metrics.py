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
  dsr_from_moments,
  max_z,
  minimum_backtest_length,
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

# Published worked example, Bailey and Lopez de Prado (2014) p.10:
# N=100, V[SR]=1/2, T=1250, annualised SR=2.5, skew=-3, raw kurtosis=10,
# 250 observations per year. Per-period Sharpe is the annualised figure
# divided by sqrt(250); trial dispersion is sqrt(1/2) also divided by
# sqrt(250), which puts both in the same units the formula requires.
paper_sharpe = 2.5 / math.sqrt(250)
paper_dispersion = math.sqrt(0.5) / math.sqrt(250)

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
  '''Honest accounting for having searched many configurations.

  The regression values below are the published worked example from
  Bailey and Lopez de Prado, "The Deflated Sharpe Ratio" (2014), p.10:
  N=100, V[SR]=1/2, T=1250, annualised SR=2.5, skew=-3, kurtosis=10,
  250 observations per year. Pinning the paper's own numbers means any
  future error in the formula shows up as a concrete diff.
  '''

  def test_max_z_matches_the_papers_trials_100(self):
    assert max_z(100) == pytest.approx(2.530603, abs=1e-5)

  def test_expected_max_sharpe_matches_the_paper(self):
    expected = paper_dispersion * max_z(100)
    assert expected == pytest.approx(0.113172, abs=1e-5)

  def test_matches_published_dsr_for_100_trials(self):
    value = dsr_from_moments(
      paper_sharpe,
      paper_dispersion * max_z(100),
      -3.0,
      10.0,
      1250,
    )
    assert value == pytest.approx(0.9004, abs=1e-4)

  def test_matches_published_dsr_for_46_trials(self):
    value = dsr_from_moments(
      paper_sharpe,
      paper_dispersion * max_z(46),
      -3.0,
      10.0,
      1250,
    )
    assert value == pytest.approx(0.9505, abs=1e-4)

  def test_matches_published_dsr_for_88_trials_normal_returns(self):
    # The paper's normal-returns case: skew 0, raw kurtosis 3.
    value = dsr_from_moments(
      paper_sharpe,
      paper_dispersion * max_z(88),
      0.0,
      3.0,
      1250,
    )
    assert value == pytest.approx(0.9505, abs=1e-4)

  def test_strong_strategies_are_not_annihilated(self):
    # Regression guard. An earlier denominator used the Euler-Mascheroni
    # constant in place of skewness and kurtosis, which went negative
    # once annualised Sharpe passed roughly 1.7 and made the function
    # return 0.0 for the very strategies it should endorse.
    returns = series(504, seed=5)
    boosted = [value + 0.004 for value in returns]
    value = deflated_sharpe(boosted, trials=2)
    assert value is not None
    assert value > 0.5

  def test_sub_year_sample_is_reported_not_discarded(self):
    # T > 1 observation, not T > 1 year. A 200-day backtest is a valid
    # sample and must not silently collapse to "no verdict".
    value = deflated_sharpe(series(200, seed=3), trials=5)
    assert value is not None
    assert 0.0 <= value <= 1.0

  def test_more_trials_never_helps(self):
    returns = series(504)
    few = deflated_sharpe(returns, trials=2)
    many = deflated_sharpe(returns, trials=500)
    assert many <= few

  def test_too_few_trials_raises(self):
    with pytest.raises(ValueError, match='trials'):
      deflated_sharpe(series(504), trials=1)

  def test_empty_is_undefined(self):
    assert deflated_sharpe([], trials=10) is None

  def test_no_dispersion_is_undefined(self):
    # Undefined, not 0.0: zero asserts "not credible after search", which
    # is a claim, whereas a flat series supports no claim at all.
    assert deflated_sharpe([0.0] * 504, trials=10) is None

  def test_empirical_trials_widen_the_bar(self):
    # A genuinely wide sweep must not report more significance than the
    # plug-in default. The default is 1/sqrt(years) ~= 0.707 per period
    # here, so the trial Sharpes below are deliberately wider than that.
    returns = series(504, seed=13)
    wide = [1.5, -1.5, 2.0, -2.0, 1.8, -1.8]
    assert (deflated_sharpe(returns, trials=50, trial_sharpes=wide)
            <= deflated_sharpe(returns, trials=50))


class TestMinimumBacktestLength:
  '''Whether the history can support the number of trials claimed.'''

  def test_reproduces_the_paper_two_years_claim(self):
    # "2 years -> no more than 7 trials": the Sharpe that exactly fills
    # two years at N=7 is max_z(7) / sqrt(2).
    target = max_z(7) / math.sqrt(2.0)
    assert minimum_backtest_length(7, target) == pytest.approx(2.0)

  def test_reproduces_the_paper_five_years_claim(self):
    target = max_z(45) / math.sqrt(5.0)
    assert minimum_backtest_length(45, target) == pytest.approx(5.0)

  def test_more_trials_need_more_history(self):
    assert (minimum_backtest_length(500, 1.0)
            > minimum_backtest_length(10, 1.0))

  def test_higher_sharpe_needs_less_history(self):
    assert (minimum_backtest_length(100, 2.0)
            < minimum_backtest_length(100, 1.0))

  def test_rejects_invalid_inputs(self):
    with pytest.raises(ValueError, match='trials'):
      minimum_backtest_length(1, 1.0)
    with pytest.raises(ValueError, match='target_sharpe'):
      minimum_backtest_length(10, 0.0)
