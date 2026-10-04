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
prices that search honestly.

No numpy or scipy. ``statistics`` supplies mean, sample stdev and the
normal distribution (``cdf`` and ``inv_cdf``), which is everything the
Deflated Sharpe needs.
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


def _expected_max_sharpe(deviation: float, trials: int) -> float:
  '''Return the expected best Sharpe from ``trials`` independent trials.

  Implements the Blumenthal/Euler-Mascheroni approximation from Bailey and
  Lopez de Prado: the expected maximum of N standard normals is not N but
  a smaller quantity, and assuming otherwise is exactly the mistake that
  makes a lucky sweep look significant.

  Args:
    deviation: Standard deviation of trial Sharpes.
    trials: Number of independent configurations tried.

  Returns:
    Expected maximum Sharpe across those trials.
  '''
  if trials < 2 or deviation <= 0.0:
    return 0.0
  best = _NORMAL.inv_cdf(1.0 - 1.0 / trials)
  tail = _NORMAL.inv_cdf(1.0 - 1.0 / (trials * math.e))
  return deviation * ((1.0 - EULER_GAMMA) * best + EULER_GAMMA * tail)


def deflated_sharpe(
  returns: list[float],
  trials: int,
  risk_free: float = 0.0,
  periods: int = TRADING_DAYS_PER_YEAR,
) -> float:
  '''Return the probability that a Sharpe survives its own search.

  A strategy that was selected as the best of many configurations needs a
  higher Sharpe to be credible than one that was not. This returns the
  probability that the observed Sharpe exceeds the expected maximum from
  ``trials`` independent tries.

  PONYTAIL: this requires at least one year of returns, because the
  statistic is defined in terms of ``sqrt(T - 1)``. Shorter series return
  0.0 rather than a misleading number. Ceiling: sub-year strategies,
  where a rolling or block-bootstrap variant of T would be needed.
  Upgrade path: pass a ``periods``/``years`` estimate that allows T > 1
  once enough history exists; no other change is required.

  Args:
    returns: Periodic returns as fractions.
    trials: How many configurations were tried before selecting this one.
      This is the number that makes the result honest, and the caller is
      the only one who knows it.
    risk_free: Periodic risk-free rate.
    periods: Periods per year.

  Returns:
    Probability in ``[0, 1]`` that the Sharpe exceeds the search-induced
    maximum.
  '''
  if len(returns) < 2 or trials < 2:
    return 0.0
  deviation = _deviation(returns)
  if deviation == 0.0:
    # A series with no dispersion has no Sharpe and therefore no
    # search-induced threshold to beat. Falling through would evaluate
    # cdf(0) = 0.5 and dress up a strategy that did nothing as "50
    # percent credible", so this is undefined and reported as 0.0 to
    # match sharpe_ratio.
    return 0.0
  observed = sharpe_ratio(returns, risk_free, periods)
  scaled = deviation * math.sqrt(periods)
  expected_max = _expected_max_sharpe(scaled, trials)
  years = len(returns) / periods
  if years <= 1.0:
    return 0.0
  numerator = (observed - expected_max) * math.sqrt(years - 1.0)
  spread = 1.0 - EULER_GAMMA * observed + EULER_GAMMA * expected_max ** 2
  if spread <= 0.0:
    return 0.0
  return _NORMAL.cdf(numerator / math.sqrt(spread))
