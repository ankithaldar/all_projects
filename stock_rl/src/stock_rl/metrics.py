#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Performance and risk statistics for an equity backtest.

Two rules govern this module.

First, every ratio declares its annualisation convention. A Sharpe of 0.8
means nothing until you know whether it is per-bar or per-year, so the
period count is an explicit argument with a single documented default
(252 NSE trading days) and no hidden scaling.

Second, the Deflated Sharpe Ratio is implemented here rather than left
implicit. Backtesting is a multiple-testing problem: run enough parameter
sweeps and the best result clears zero by luck alone. ``deflated_sharpe``
prices that search honestly, and ``minimum_backtest_length`` reports
whether the available history can support the number of trials claimed at
all.

No numpy or scipy. ``statistics`` supplies mean, sample stdev and the
normal distribution (``cdf`` and ``inv_cdf``), which is everything the
Deflated Sharpe needs.

One caveat worth stating plainly, because it is a property of the
published statistic rather than of this code: the Deflated Sharpe is not a
calibrated significance test. It compares against the *mean* of the null
distribution of the maximum, so a strategy sitting exactly on the
threshold scores 0.5 by construction and the conventional 0.95 bar is a
convention layered on top. Read it as a deflation magnitude, not as a 5
percent error rate.
'''

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist, fmean, stdev

__all__ = [
  'TRADING_DAYS_PER_YEAR',
  'Drawdown',
  'calmar_ratio',
  'deflated_sharpe',
  'drawdown_curve',
  'max_drawdown',
  'sharpe_ratio',
  'sortino_ratio',
  'dsr_from_moments',
  'max_z',
  'minimum_backtest_length',
  'total_return',
  'turnover',
]

#: NSE equity trading days per year. Used to annualise ratios.
TRADING_DAYS_PER_YEAR = 252

#: Euler-Mascheroni constant, used by the expected-maximum-Sharpe term.
EULER_GAMMA = 0.5772156649015329

_NORMAL = NormalDist()


@dataclass(frozen=True, slots=True)
class Drawdown:
  '''The deepest peak-to-trough decline in an equity curve.

  Attributes:
    depth: Peak-to-trough decline as a positive fraction, so 0.23 is a
      23 percent drawdown. Positive rather than negative so that
      comparisons and penalties read naturally.
    peak_index: Index of the equity high that started the decline.
    trough_index: Index of the equity low that ended it.
    recovery_index: Index at which equity regained the prior peak, or
      ``None`` if the curve never got back there.
  '''

  depth: float
  peak_index: int
  trough_index: int
  recovery_index: int | None = None


def _deviation(returns: list[float]) -> float:
  '''Return the sample standard deviation, or zero when undefined.

  Args:
    returns: Periodic returns.

  Returns:
    Sample standard deviation, or 0.0 for fewer than two observations.
  '''
  if len(returns) < 2:
    return 0.0
  return stdev(returns)


def sharpe_ratio(
  returns: list[float],
  risk_free: float = 0.0,
  periods: int = TRADING_DAYS_PER_YEAR,
) -> float:
  '''Return the annualised Sharpe ratio of periodic returns.

  Args:
    returns: Periodic returns as fractions, e.g. 0.01 for one percent.
    risk_free: Periodic risk-free rate, same frequency as ``returns``.
    periods: Periods per year. 252 for daily NSE bars.

  Returns:
    Annualised Sharpe ratio, or 0.0 when the return series has no
    dispersion, because the ratio is undefined rather than infinite
    there and reporting 0.0 keeps downstream comparisons total-ordered.
  '''
  deviation = _deviation(returns)
  if deviation == 0.0 or len(returns) < 2:
    return 0.0
  per_period = (fmean(returns) - risk_free) / deviation
  return per_period * math.sqrt(periods)


def sortino_ratio(
  returns: list[float],
  risk_free: float = 0.0,
  periods: int = TRADING_DAYS_PER_YEAR,
) -> float:
  '''Return the annualised Sortino ratio.

  Unlike Sharpe this charges only for downside deviation, which is the
  relevant risk for a long-only equity book that cannot lose more than it
  holds.

  Args:
    returns: Periodic returns as fractions.
    risk_free: Periodic risk-free rate.
    periods: Periods per year.

  Returns:
    Annualised Sortino ratio, or 0.0 when there is no downside deviation
    or fewer than two observations.
  '''
  if len(returns) < 2:
    return 0.0
  shortfall = [min(0.0, value - risk_free) for value in returns]
  downside = _deviation(shortfall)
  if downside == 0.0:
    return 0.0
  excess = fmean(returns) - risk_free
  return (excess / downside) * math.sqrt(periods)


def drawdown_curve(equity: list[float]) -> list[float]:
  '''Return the distance below the running peak at each point.

  Args:
    equity: Equity curve, starting at or above 1.0.

  Returns:
    List of non-negative drawdown fractions aligned with ``equity``.
  '''
  curve: list[float] = []
  peak = -math.inf
  for value in equity:
    peak = max(peak, value)
    curve.append(0.0 if peak <= 0.0 else max(0.0, 1.0 - value / peak))
  return curve


def max_drawdown(equity: list[float]) -> Drawdown:
  '''Return the deepest peak-to-trough decline in an equity curve.

  Args:
    equity: Equity curve.

  Returns:
    The deepest ``Drawdown``. All zeros for an empty or flat curve.
  '''
  if not equity:
    return Drawdown(0.0, -1, -1, None)
  peak = equity[0]
  peak_index = 0
  best = Drawdown(0.0, 0, 0, None)
  for index, value in enumerate(equity):
    if value > peak:
      peak = value
      peak_index = index
    if peak <= 0.0:
      continue
    depth = 1.0 - value / peak
    if depth > best.depth:
      best = Drawdown(depth, peak_index, index, None)
  return _find_recovery(equity, best)


def _find_recovery(equity: list[float], drawdown: Drawdown) -> Drawdown:
  '''Attach the recovery index to a drawdown, if the curve recovered.

  Args:
    equity: Equity curve the drawdown was measured on.
    drawdown: Drawdown with peak and trough indices filled in.

  Returns:
    The same drawdown with ``recovery_index`` set, or still ``None``.
  '''
  prior_peak = equity[drawdown.peak_index]
  for index in range(drawdown.trough_index + 1, len(equity)):
    if equity[index] >= prior_peak:
      return Drawdown(
        drawdown.depth,
        drawdown.peak_index,
        drawdown.trough_index,
        index,
      )
  return drawdown


def calmar_ratio(
  returns: list[float],
  periods: int = TRADING_DAYS_PER_YEAR,
) -> float:
  '''Return annualised return divided by maximum drawdown.

  Args:
    returns: Periodic returns as fractions.
    periods: Periods per year. 252 for daily NSE bars.

  Returns:
    Calmar ratio, or 0.0 when the drawdown is zero, since an unbounded
    ratio carries no information and would dominate any ranking.
  '''
  if not returns or periods <= 0:
    return 0.0
  equity: list[float] = []
  value = 1.0
  for ret in returns:
    value *= 1.0 + ret
    equity.append(value)
  drawdown = max_drawdown(equity)
  if drawdown.depth <= 0.0:
    return 0.0
  years = len(returns) / periods
  annualised = total_return(returns) ** (1.0 / years) - 1.0
  return annualised / drawdown.depth


def total_return(returns: list[float]) -> float:
  '''Return the cumulative growth factor of periodic returns.

  Args:
    returns: Periodic returns as fractions.

  Returns:
    Product of ``1 + return``, so 0.21 means a 21 percent gain. Returns
    1.0 for an empty series.
  '''
  value = 1.0
  for ret in returns:
    value *= 1.0 + ret
  return value


def turnover(weights: list[float]) -> float:
  '''Return total one-way turnover across a sequence of portfolio weights.

  The first weight is measured against a flat portfolio, so a book that
  opens at 100 percent weight and never trades again scores exactly 1.0.

  Args:
    weights: Target weights per rebalance, as fractions.

  Returns:
    Sum of absolute weight changes.
  '''
  previous = 0.0
  traded = 0.0
  for weight in weights:
    traded += abs(weight - previous)
    previous = weight
  return traded

def max_z(trials: int) -> float:
  '''Return the expected maximum of ``trials`` standard normals.

  This is the Blumenthal/Euler-Mascheroni approximation from Bailey and
  Lopez de Prado (2014), equation 1. The expected maximum of N standard
  normals is far smaller than N, and treating it otherwise is precisely
  the mistake that makes a lucky parameter sweep look significant.

  Args:
    trials: Number of independent configurations tried.

  Returns:
    Expected maximum of ``trials`` standard normal draws.

  Raises:
    ValueError: If fewer than two trials are supplied, since the
      approximation is undefined for a single draw.
  '''
  if trials < 2:
    raise ValueError(f'trials must be >= 2, got {trials}')
  best = _NORMAL.inv_cdf(1.0 - 1.0 / trials)
  tail = _NORMAL.inv_cdf(1.0 - 1.0 / (trials * math.e))
  return (1.0 - EULER_GAMMA) * best + EULER_GAMMA * tail


def _moments(values: list[float]) -> tuple[float, float, float, float]:
  '''Return mean, sample stdev, skewness and raw kurtosis.

  Args:
    values: Return series.

  Returns:
    Tuple of (mean, sample stdev, skewness, raw kurtosis). Skewness and
    kurtosis are 0.0 and 3.0 respectively when they are undefined, which
  is the Gaussian reference case the Mertens term assumes.
  '''
  if len(values) < 2:
    raise ValueError(f'need >= 2 observations, got {len(values)}')
  mean = fmean(values)
  deviation = stdev(values)
  if deviation == 0.0:
    return mean, 0.0, 0.0, 3.0
  total = len(values)
  third = sum((value - mean) ** 3 for value in values) / total
  fourth = sum((value - mean) ** 4 for value in values) / total
  return mean, deviation, third / deviation ** 3, fourth / deviation ** 4


def dsr_from_moments(
  sharpe: float,
  expected_max_sharpe: float,
  skew: float,
  kurtosis: float,
  observations: int,
) -> float:
  '''Return the Deflated Sharpe from precomputed distribution moments.

  Isolated from ``deflated_sharpe`` so the published worked example can be
  reproduced exactly, feeding in the skewness and kurtosis the paper states
  rather than values estimated from a sample.

  Args:
    sharpe: Per-period Sharpe of the selected strategy, not annualised.
    expected_max_sharpe: Expected maximum per-period Sharpe across the
      trial set.
    skew: Skewness of the return distribution.
    kurtosis: Raw kurtosis of the return distribution. 3.0 is Gaussian.
    observations: Number of return observations, T.

  Returns:
    Probability in ``[0, 1]`` that the Sharpe exceeds the search maximum.
  '''
  if observations < 2:
    raise ValueError(f'observations must be >= 2, got {observations}')
  variance = (1.0 - skew * sharpe
              + (kurtosis - 1.0) / 4.0 * sharpe ** 2)
  if variance <= 0.0:
    return 0.0
  numerator = (sharpe - expected_max_sharpe) * math.sqrt(observations - 1)
  return _NORMAL.cdf(numerator / math.sqrt(variance))


def deflated_sharpe(
  returns: list[float],
  trials: int,
  trial_sharpes: list[float] | None = None,
  risk_free: float = 0.0,
  periods: int = TRADING_DAYS_PER_YEAR,
) -> float | None:
  '''Return the probability that a Sharpe survives its own search.

  A strategy chosen as the best of many configurations needs a higher
  Sharpe to be credible than one that was not chosen at all. This returns
  the probability that the observed Sharpe exceeds the expected maximum
  across ``trials`` independent tries, in the sense of Bailey and Lopez de
  Prado (2014), equation 2.

  The computation runs entirely in per-period units. The annualised Sharpe
  reported by ``sharpe_ratio`` is a display statistic and must not be fed
  in here: the ``sqrt(T - 1)`` term counts observations, not years.

  PONYTAIL: the dispersion of trial Sharpes defaults to the theoretical
  plug-in ``1 / sqrt(years)``, which assumes every trial had zero true
  edge. Ceiling: a parameter grid spanning very different volatilities
  has wider dispersion than that, so the plug-in understates the bar.
  Upgrade path: pass ``trial_sharpes`` from your own sweep log, which is
  the empirical estimator the paper prefers and which strictly dominates
  the default.

  Args:
    returns: Periodic returns as fractions.
    trials: How many configurations were tried before selecting this one.
      Only the caller knows this, and undercounting it weakens the
      deflation silently.
    trial_sharpes: Per-period Sharpes of every configuration tried. When
      omitted the theoretical plug-in is used, see the note above.
    risk_free: Periodic risk-free rate.
    periods: Periods per year.

  Returns:
    Probability in ``[0, 1]``, or ``None`` when the statistic is undefined
    because the series has no dispersion or fewer than two observations.
    Undefined is deliberately not reported as 0.0: zero asserts "not
    credible after search", which is a claim, whereas an insufficient
    sample is an absence of one.

  Raises:
    ValueError: If fewer than two trials are supplied.
  '''
  if trials < 2:
    raise ValueError(f'trials must be >= 2, got {trials}')
  if len(returns) < 2:
    return None
  adjusted = [value - risk_free for value in returns]
  mean, deviation, skew, kurtosis = _moments(adjusted)
  if deviation == 0.0:
    return None
  observed = mean / deviation
  years = len(returns) / periods
  if trial_sharpes is None:
    dispersion = 1.0 / math.sqrt(years)
  else:
    dispersion = stdev(trial_sharpes) if len(trial_sharpes) > 1 else 0.0
  return dsr_from_moments(
    observed,
    dispersion * max_z(trials),
    skew,
    kurtosis,
    len(returns),
  )


def minimum_backtest_length(trials: int, target_sharpe: float) -> float:
  '''Return the years of history needed to justify ``trials`` trials.

  Bailey, Borwein, Lopez de Prado and Zhu (2014) show that searching N
  configurations requires a minimum sample length before any Sharpe is
  worth anything. Run this as a gate before trusting ``deflated_sharpe``:
  if the requirement exceeds the history available, no amount of
  implementation correctness rescues the result.

  Args:
    trials: Number of independent configurations tried.
    target_sharpe: Annualised Sharpe being claimed.

  Returns:
    Required sample length in years.

  Raises:
    ValueError: If ``trials`` is below two or the target Sharpe is not
      positive.
  '''
  if trials < 2:
    raise ValueError(f'trials must be >= 2, got {trials}')
  if target_sharpe <= 0.0:
    raise ValueError(f'target_sharpe must be positive, got {target_sharpe}')
  return (max_z(trials) / target_sharpe) ** 2

