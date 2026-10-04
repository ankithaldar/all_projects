#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Continuous-weight RL environment for cross-sectional allocation.

This is the SAC-shaped counterpart to :mod:`stock_rl.env.nse`: same bars,
same costs, same no-look-ahead discipline, but the action is a target
weight rather than a direction. It exists because the RL allocation
evidence is a comparison, and a comparison needs both arms measured by the
same instrument. :func:`stock_rl.rl.train.compare` runs this environment
and :func:`stock_rl.portfolio.run_portfolio` over identical panels, costs
and rebalance cadence for exactly that reason.

**Constraints live here, outside the policy, and that is not a
compromise.** The design doc asks for a cvxpy layer inside the policy to
enforce the sector limit. A QP projection is not differentiable, so
back-propagating through one makes the policy gradient silently wrong
rather than approximately wrong. The actor therefore emits raw desired
weights and the environment clips per symbol, scales to a total of at most
1.0, and scales over-cap sectors down. The executed trade is always
feasible and the actor stays unconstrained.

**The drawdown term defaults to zero.** Reward is running Sharpe minus
charged cost minus turnover. An asymmetric per-step drawdown penalty punishes
loss without crediting gain, which is the doubling-strategy hazard: exposure
is raised after losses because the expected recovery gain beats the
incremental penalty, which is a martingale. Francois et al. (2025) had to
add anti-speculation machinery for the same reason, and Zernikov's default
reward collapses to a constant on any non-negative P&L. The term is
exposed for a caller who has decided they want it, with the hazard named in
``drawdown``'s docstring.

**Sizing uses the decision bar's close, not the fill bar's close.** An
order decided at the close of bar ``t`` fills at the open of bar ``t+1``,
and the only portfolio value known when it was sized was the mark at the
close of bar ``t``. Marking at the fill bar's close before sizing, as
``env.nse`` does, quietly buys more shares as the price rises.
'''

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, stdev

from stock_rl.bars import Bar
from stock_rl.costs import DELIVERY, CostModel, Side
from stock_rl.weights import affordable_scale, clamp_weight
from stock_rl.env.gym import Info, Obs
from stock_rl.metrics import TRADING_DAYS_PER_YEAR, max_drawdown
from stock_rl.rl.policy import Action

__all__ = ['AllocationReward', 'WeightAllocationEnv']

#: Default cap on any single symbol's weight.
default_max_weight = 0.10

#: Default cap on any single sector's weight.
default_max_sector_weight = 0.30


@dataclass(frozen=True, slots=True)
class AllocationReward:
  '''Weights on the reward terms.

  Attributes:
    sharpe: Scale on the running Sharpe contribution. Computed over the
      running return history, not the single bar's return, because one bar
      of return is close to pure noise and training on it chases noise.
    cost: Scale on the transaction cost actually charged, expressed as a
      fraction of capital.
    turnover: Scale on the traded weight of the step. Charged separately
      from cost because the two fail differently: cost catches the rupee
      bill, turnover catches churn that is free only under a cost model
      that happens to be set to zero.
    drawdown: Scale on the squared current drawdown. Defaults to zero.
      HAZARD: a term that punishes loss without crediting gain makes the
      doubling strategy optimal, i.e. increasing exposure after losses to
      recover them, which is a martingale. Enable only with the objective
      understood, and prefer a coherent terminal risk measure, which is
      what the hedge environment implements.
  '''

  sharpe: float = 1.0
  cost: float = 1.0
  turnover: float = 0.5
  drawdown: float = 0.0


class WeightAllocationEnv:
  '''Long-only cross-sectional environment with continuous target weights.

  Implements the ``Env`` protocol from :mod:`stock_rl.env.gym` and the
  continuous ``step`` signature of
  :class:`stock_rl.rl.policy.ContinuousEnv`.

  Attributes:
    action_symbols: Symbols the agent may weight, in sorted order.
    n_steps: Length of one episode in steps.
  '''

  action_symbols: tuple[str, ...]
  n_steps: int

  def __init__(
    self,
    panels: dict[str, list[Bar]],
    capital: float = 10_000_000.0,
    costs: CostModel = DELIVERY,
    reward: AllocationReward = AllocationReward(),
    max_weight: float = default_max_weight,
    max_sector_weight: float = default_max_sector_weight,
    sectors: dict[str, str] | None = None,
    history: int = 60,
    rebalance_days: int = 1,
    rebalance_offset: int = 0,
  ) -> None:
    '''Build an environment over aligned per-symbol bar panels.

    ``history`` and ``rebalance_days`` default to the values
    :func:`stock_rl.portfolio.run_portfolio` uses, because the comparison
    between the two engines is only meaningful when the cadence matches.
    With ``rebalance_days=1`` the book rebalances daily; at 21 it trades
    monthly and the rest of the steps hold. ``rebalance_offset`` exists so
    the fill *bars* can be aligned with that function's calendar rather
    than merely the interval; see :func:`stock_rl.rl.train.compare`.

    Args:
      panels: Mapping of symbol to bars in ascending time order.
      capital: Starting cash in rupees.
      costs: Transaction cost model.
      reward: Reward term weights.
      max_weight: Cap on any single symbol's portfolio weight.
      max_sector_weight: Cap on any single sector's weight. Only enforced
        when ``sectors`` is supplied.
      sectors: Optional mapping of symbol to sector name.
      history: Number of warm-up bars before the first decision.
      rebalance_days: Bars between allowed rebalances.
      rebalance_offset: First step index allowed to trade, taken modulo
        ``rebalance_days``.

    Raises:
      ValueError: If panels are empty, misaligned, shorter than
        ``history``, or if an argument is out of range.
    '''
    if not panels:
      raise ValueError('panels must not be empty')
    lengths = {len(bars) for bars in panels.values()}
    if len(lengths) != 1:
      raise ValueError(f'panels must be aligned, got lengths {sorted(lengths)}')
    self.action_symbols = tuple(sorted(panels))
    self.panels = panels
    self.timestamps = [bar.timestamp for bar in panels[self.action_symbols[0]]]
    if len(self.timestamps) <= history:
      raise ValueError(
        f'panels of {len(self.timestamps)} bars must exceed history={history}')
    if capital <= 0.0:
      raise ValueError(f'capital must be positive, got {capital}')
    if not 0.0 < max_weight <= 1.0:
      raise ValueError(f'max_weight must be in (0, 1], got {max_weight}')
    if not 0.0 < max_sector_weight <= 1.0:
      raise ValueError(
        f'max_sector_weight must be in (0, 1], got {max_sector_weight}')
    if rebalance_days < 1:
      raise ValueError(f'rebalance_days must be >= 1, got {rebalance_days}')
    if not 0 <= rebalance_offset < rebalance_days:
      raise ValueError(
        f'rebalance_offset must be in [0, {rebalance_days}), got '
        f'{rebalance_offset}')
    self.capital = capital
    self.costs = costs
    self.reward = reward
    self.max_weight = max_weight
    self.max_sector_weight = max_sector_weight
    self.sectors = sectors or {}
    self.history = history
    self.rebalance_days = rebalance_days
    self.rebalance_offset = rebalance_offset
    self.n_steps = len(self.timestamps) - history
    self._reset_state()

  def _reset_state(self) -> None:
    '''Reset the book and episode bookkeeping to the start.'''
    self._step_index = 0
    self._cash = self.capital
    self._quantity = dict.fromkeys(self.action_symbols, 0)
    self._weights = dict.fromkeys(self.action_symbols, 0.0)
    self._equity = [1.0]
    self._returns: list[float] = []
    self._peak = 1.0
    self._drawdown = 0.0
    self._total_cost = 0.0
    self._turnover_total = 0.0
    self._decisions: list[dict[str, object]] = []

  def _bar_index(self) -> int:
    '''Return the absolute index of the current decision bar.'''
    return self.history + self._step_index - 1

  def reset(self, seed: int | None = None) -> Obs:
    '''Restart the episode from the first decision bar.

    Args:
      seed: Accepted for interface compatibility. This environment is
        deterministic, so the seed is ignored rather than silently
        producing a different episode than a caller expects.

    Returns:
      The first observation.
    '''
    del seed
    self._reset_state()
    return self._observation()

  def _observation(self) -> Obs:
    '''Build the observation visible at the current decision bar.

    The history handed to the policy is sliced to bars at or before the
    decision point, which is what makes look-ahead impossible for an
    actor that reads it: the bar it will be filled on is absent.

    Returns:
      Mapping with per-symbol returns, visible history and book state.
    '''
    index = self._bar_index()
    history = {
      symbol: tuple(panel[:index + 1])
      for symbol, panel in self.panels.items()
    }
    closes = {symbol: [bar.close for bar in bars]
              for symbol, bars in history.items()}
    return {
      'step': self._step_index,
      'timestamp': self.timestamps[index],
      'symbols': self.action_symbols,
      'history': history,
      'returns': {
        symbol: _trailing_return(prices)
        for symbol, prices in closes.items()
      },
      'weights': dict(self._weights),
      'cash': self._cash,
      'equity': self._equity[-1],
      'drawdown': self._drawdown,
    }

  def step(self, action: Action) -> tuple[Obs, float, bool, Info]:
    '''Advance one bar, executing at the next bar's open.

    Args:
      action: Desired weight per symbol. Values outside ``[0, 1]`` are
        clamped by the environment, and missing symbols are treated as
        "hold current", so a partial action is still valid.

    Returns:
      Tuple of (observation, reward, terminated, info). The episode
      terminates on the last bar, because there is no following open to
      fill against.
    '''
    if self._step_index >= self.n_steps:
      return self._observation(), 0.0, True, self._info()
    decision = self._bar_index()
    fill = decision + 1
    known_value = self._portfolio_value(decision)
    target = self._target_weights(action, known_value)
    cost, traded = self._execute(target, known_value, fill)
    self._recount_weights(known_value)
    self._equity.append(self._portfolio_value(fill) / self.capital)
    previous = self._equity[-2]
    self._returns.append(self._equity[-1] / previous - 1.0)
    self._peak = max(self._peak, self._equity[-1])
    self._drawdown = max(self._drawdown, 1.0 - self._equity[-1] / self._peak)
    self._turnover_total += traded
    reward = self._reward(cost, traded)
    self._step_index += 1
    self._decisions.append({
      'step': self._step_index,
      'timestamp': self.timestamps[decision],
      'weights': dict(self._weights),
      'target': dict(target),
      'cost': cost,
      'turnover': traded,
    })
    terminated = self._step_index >= self.n_steps
    return self._observation(), reward, terminated, self._info(cost, traded)

  def _target_weights(
    self, action: Action, value: float) -> dict[str, float]:
    '''Turn a desired-weight action into feasible target weights.

    Feasibility is applied here rather than inside the policy: clip per
    symbol, scale the total to at most 1.0, then scale down any sector
    over its cap. The result is long-only by construction, so a short is
    not expressible even by accident.

    Args:
      action: Desired weight per symbol.
      value: Portfolio value at the decision bar, used only to report.

    Returns:
      Target weight per symbol, summing to at most 1.0.
    '''
    del value
    wanted = {}
    for symbol in self.action_symbols:
      raw = _as_float(action.get(symbol, self._weights[symbol]))
      wanted[symbol] = clamp_weight(float(raw), self.max_weight)
    total = sum(wanted.values())
    if total > 1.0:
      wanted = {symbol: weight / total for symbol, weight in wanted.items()}
    return self._apply_sector_cap(wanted)

  def _apply_sector_cap(
    self, desired: dict[str, float]) -> dict[str, float]:
    '''Scale down weights so no sector exceeds its cap.

    Args:
      desired: Unconstrained desired weights.

    Returns:
      Weights respecting the per-sector cap. Symbols without a recorded
      sector are capped individually only.
    '''
    if not self.sectors:
      return desired
    totals: dict[str, float] = {}
    for symbol, weight in desired.items():
      sector = self.sectors.get(symbol)
      if sector is not None:
        totals[sector] = totals.get(sector, 0.0) + weight
    over = {
      sector: total
      for sector, total in totals.items()
      if total > self.max_sector_weight
    }
    if not over:
      return desired
    scaled = dict(desired)
    for sector, total in over.items():
      factor = self.max_sector_weight / total
      for symbol, weight in desired.items():
        if self.sectors.get(symbol) == sector:
          scaled[symbol] = weight * factor
    return scaled

  def _execute(
    self,
    target: dict[str, float],
    value: float,
    fill: int,
  ) -> tuple[float, float]:
    '''Fill orders at the fill bar's open and return cost and turnover.

    Turnover is measured only on steps that actually trade. Counting the
    drift between the last target and the current weights on a step where
    the book is holding would report churn that never happened.

    Args:
      target: Feasible target weights.
      value: Portfolio value at the decision bar, the sizing basis.
      fill: Index of the bar whose open the fills occur at.

    Returns:
      Tuple of (cost in rupees, absolute weight traded).
    '''
    if self._step_index % self.rebalance_days != self.rebalance_offset:
      return 0.0, 0.0
    deltas: dict[str, int] = {}
    prices: dict[str, float] = {}
    for symbol, wanted in target.items():
      price = self.panels[symbol][fill].open
      if price <= 0.0:
        continue
      prices[symbol] = price
      deltas[symbol] = (int(wanted * value / price)
                        - self._quantity[symbol])
    # Same affordability constraint as portfolio.run_portfolio. Sizing
    # int(wanted * value / price) bounds the notional by the portfolio
    # value but ignores charges, so a fully invested book necessarily
    # overspends and borrows without paying for it. The two engines must
    # agree, so both scale the deltas to the cash actually available.
    scale = affordable_scale(self._cash, prices, deltas, self.costs)
    traded = _absolute_change(target, self._weights)
    cost = 0.0
    for symbol, delta in deltas.items():
      if delta == 0:
        continue
      scaled = int(delta * scale)
      if scaled == 0:
        continue
      notional = abs(scaled) * prices[symbol]
      charge = self.costs.one_way(
        Side.BUY if scaled > 0 else Side.SELL, notional)
      self._cash -= scaled * prices[symbol] + charge
      self._quantity[symbol] += scaled
      cost += charge
    self._total_cost += cost
    return cost, traded

  def _recount_weights(self, value: float) -> None:
    '''Recompute current weights from holdings and the decision bar.

    Args:
      value: Portfolio value at the decision bar.
    '''
    if value <= 0.0:
      self._weights = dict.fromkeys(self.action_symbols, 0.0)
      return
    index = self._bar_index()
    self._weights = {
      symbol: self._quantity[symbol]
      * self.panels[symbol][index].close / value
      for symbol in self.action_symbols
    }

  def _portfolio_value(self, index: int) -> float:
    '''Return cash plus mark-to-market holdings at a bar's close.

    Args:
      index: Bar index at which to mark holdings.

    Returns:
      Portfolio value in rupees.
    '''
    held = sum(
      self._quantity[symbol] * self.panels[symbol][index].close
      for symbol in self.action_symbols
    )
    return self._cash + held

  def _reward(self, cost: float, traded: float) -> float:
    '''Combine the reward terms for one bar.

    Args:
      cost: Transaction cost charged on the bar, in rupees.
      traded: Absolute weight traded on the bar.

    Returns:
      Scalar reward.
    '''
    running_mean = fmean(self._returns)
    running_vol = stdev(self._returns) if len(self._returns) > 1 else 0.0
    sharpe = 0.0 if running_vol <= 0.0 else running_mean / running_vol
    drawdown = self._drawdown * self._drawdown
    return (
      self.reward.sharpe * sharpe
      - self.reward.cost * (cost / self.capital)
      - self.reward.turnover * traded
      - self.reward.drawdown * drawdown
    )

  def _info(self, cost: float = 0.0, traded: float = 0.0) -> Info:
    '''Build the auxiliary info mapping for the current step.

    Args:
      cost: Transaction cost charged on the step just taken.
      traded: Absolute weight traded on the step just taken.

    Returns:
      Info mapping including equity, drawdown, cost and turnover.
    '''
    return {
      'cost': cost,
      'total_cost': self._total_cost,
      'turnover': traded,
      'total_turnover': self._turnover_total,
      'equity': self._equity[-1],
      'drawdown': self._drawdown,
      'gross_exposure': sum(abs(weight) for weight in self._weights.values()),
    }

  def render(self) -> str:
    '''Return a one-line summary of the episode so far.

    Returns:
      Human-readable summary.
    '''
    return (
      f'step {self._step_index}/{self.n_steps} '
      f'equity {self._equity[-1]:.4f} drawdown {self._drawdown:.2%} '
      f'cost {self._total_cost:.0f}'
    )

  @property
  def equity_curve(self) -> list[float]:
    '''Return the equity curve, starting at 1.0.'''
    return list(self._equity)

  @property
  def returns(self) -> list[float]:
    '''Return the per-bar portfolio returns aligned with the equity curve.'''
    return list(self._returns)

  @property
  def total_cost(self) -> float:
    '''Return cumulative transaction cost charged so far, in rupees.'''
    return self._total_cost

  @property
  def total_turnover(self) -> float:
    '''Return cumulative absolute weight traded.'''
    return self._turnover_total

  @property
  def weights(self) -> dict[str, float]:
    '''Return current portfolio weights per symbol.'''
    return dict(self._weights)

  @property
  def positions(self) -> dict[str, int]:
    '''Return current share holdings per symbol.'''
    return dict(self._quantity)

  @property
  def drawdown(self) -> float:
    '''Return current drawdown from the running equity peak.'''
    return self._drawdown

  @property
  def cash(self) -> float:
    '''Return uninvested cash in rupees.'''
    return self._cash

  @property
  def decisions(self) -> list[dict[str, object]]:
    '''Return the per-step decision log, one entry per executed step.'''
    return list(self._decisions)

  def observation(self) -> Obs:
    '''Return the observation visible right now, without advancing.'''
    return self._observation()

  def info(self) -> Info:
    '''Return auxiliary state without advancing the episode.'''
    return self._info()

  @property
  def max_drawdown(self) -> float:
    '''Return the deepest peak-to-trough decline of the equity curve.'''
    return max_drawdown(self._equity).depth

  def annualized_sharpe(self) -> float:
    '''Return the realised annualised Sharpe of the completed episode.

    Reported for convenience only. Any decision about it belongs to the
    Deflated Sharpe, which knows how many trials were tried.
    '''
    if len(self._returns) < 2:
      return 0.0
    deviation = stdev(self._returns)
    if deviation <= 0.0:
      return 0.0
    return (fmean(self._returns) / deviation) * (TRADING_DAYS_PER_YEAR ** 0.5)


def _as_float(value: object) -> float:
  '''Coerce an action entry to a float, treating junk as flat.

  Args:
    value: Raw action entry.

  Returns:
    A float, or 0.0 when the entry cannot be read as one.
  '''
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    return 0.0
  return float(value)


def _absolute_change(
  target: dict[str, float], current: dict[str, float]) -> float:
  '''Return total absolute weight traded between two weight books.

  Args:
    target: Desired weights.
    current: Current weights.

  Returns:
    Sum of absolute weight differences.
  '''
  return sum(abs(target.get(symbol, 0.0) - current.get(symbol, 0.0))
             for symbol in target)


def _trailing_return(prices: list[float], window: int = 21) -> float:
  '''Return the trailing return over ``window`` bars.

  Args:
    prices: Closing prices up to and including the decision bar.
    window: Lookback in bars.

  Returns:
    Fractional return, or 0.0 when there is not enough history.
  '''
  if len(prices) <= window:
    return 0.0
  past = prices[-window - 1]
  if past <= 0.0:
    return 0.0
  return prices[-1] / past - 1.0
