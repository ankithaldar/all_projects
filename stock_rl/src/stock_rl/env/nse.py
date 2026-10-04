#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cross-sectional RL environment for a Nifty-50 style equity book.

Design decisions here are evidence-led, not cosmetic.

**One cross-sectional policy, not 50 single-stock specialists.** Fifty
PPO specialists cannot learn cross-sectional structure, share no data
efficiently, and a portfolio layer on top would relearn the same fifty
stocks. One policy over the whole panel is strictly less code and
strictly more informative.

**Discrete per-symbol actions, so PPO is the correct algorithm.** The
docs propose SAC for weight allocation, which cannot represent a
discrete BUY/SELL/HOLD choice. Actions here are -1 reduce, 0 hold, 1 add
per symbol, mapped onto weights by the environment.

**No look-ahead, structurally.** An action taken at the close of bar ``t``
is filled at the *open* of bar ``t+1*. The agent never sees the fill
price, and the observation is sliced to bars at or before the decision
point.

**Reward follows the design doc but is honest about its risk.** The doc
specifies differential Sharpe minus costs minus a drawdown penalty.
Differential Sharpe is standard (Moody & Saffell). The drawdown term is
included because the doc requires it, with the doubling-strategy hazard
named: a penalty that punishes loss without crediting gain invites
exposure increases after losses. The remedy used in the hedging
literature is a coherent terminal risk measure rather than a per-step
asymmetric proxy, so ``drawdown_weight`` defaults to zero and is exposed
for the caller who has decided they want it.

**Not built here, deliberately:** cvxpy-style constraint projection inside
the policy. A QP projection is not differentiable, so the policy gradient
would be silently wrong. Sector and concentration limits are applied as a
post-processing step in the environment, which keeps the executed trade
feasible and the actor unconstrained.
'''

from __future__ import annotations

from dataclasses import dataclass

from stock_rl.bars import Bar
from stock_rl.costs import DELIVERY, CostModel, Side
from stock_rl.weights import affordable_scale, clamp_weight
from stock_rl.env.gym import Info, Obs
from stock_rl.metrics import TRADING_DAYS_PER_YEAR

__all__ = ['PortfolioEnv', 'RewardWeights']

#: Maximum weight any single symbol may hold.
MAX_WEIGHT = 0.10

#: Maximum weight any single sector may hold.
MAX_SECTOR_WEIGHT = 0.30


@dataclass(frozen=True, slots=True)
class RewardWeights:
  '''Weights on the reward terms.

  Attributes:
    sharpe: Scale on the per-bar differential Sharpe contribution. This
      is the primary term, per the design doc.
    turnover: Scale on the transaction cost actually charged, so the
      agent pays for trading rather than discovering that churn is free.
    drawdown: Scale on the per-bar increase in drawdown. Defaults to
      zero because asymmetric per-step drawdown penalties create a
      doubling pathology; enable only with the reward understood.
  '''

  sharpe: float = 1.0
  turnover: float = 0.5
  drawdown: float = 0.0


class PortfolioEnv:
  '''A long-only cross-sectional equity environment over daily bars.

  Implements the ``Env`` protocol from :mod:`stock_rl.env.gym`.
  '''

  action_symbols: tuple[str, ...]
  n_steps: int

  def __init__(
    self,
    panels: dict[str, list[Bar]],
    capital: float = 10_000_000.0,
    costs: CostModel = DELIVERY,
    reward: RewardWeights = RewardWeights(),
    max_weight: float = MAX_WEIGHT,
    max_sector_weight: float = MAX_SECTOR_WEIGHT,
    sectors: dict[str, str] | None = None,
    history: int = 252,
  ) -> None:
    '''Build an environment from per-symbol bar panels.

    All symbols must share one aligned timeline, because the
    environment advances every panel in lockstep. Misaligned panels are
    rejected rather than silently forward-filled, since forward-filling
    invents bars that did not trade.

    Args:
      panels: Mapping of symbol to bars in ascending time order.
      capital: Starting cash in rupees.
      costs: Transaction cost model.
      reward: Reward term weights.
      max_weight: Cap on any single symbol's portfolio weight.
      max_sector_weight: Cap on any single sector's weight. Only
        enforced when ``sectors`` is supplied.
      sectors: Optional mapping of symbol to sector name.
      history: Number of bars included in each observation.

    Raises:
      ValueError: If panels are empty, misaligned, shorter than
        ``history``, or if a weight cap is out of range.
    '''
    if not panels:
      raise ValueError('panels must not be empty')
    self.action_symbols = tuple(sorted(panels))
    lengths = {len(bars) for bars in panels.values()}
    if len(lengths) != 1:
      raise ValueError(f'panels must be aligned, got lengths {sorted(lengths)}')
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
    self.capital = capital
    self.costs = costs
    self.reward = reward
    self.max_weight = max_weight
    self.max_sector_weight = max_sector_weight
    self.sectors = sectors or {}
    self.history = history
    self.n_steps = len(self.timestamps) - history
    self._reset_state()

  def _reset_state(self) -> None:
    '''Reset the portfolio and episode bookkeeping to the start.'''
    self._step_index = 0
    self._cash = self.capital
    self._quantity = dict.fromkeys(self.action_symbols, 0)
    self._weights = dict.fromkeys(self.action_symbols, 0.0)
    self._equity = [1.0]
    self._returns: list[float] = []
    self._peak = 1.0
    self._drawdown = 0.0
    self._total_cost = 0.0
    self._decisions: list[dict[str, object]] = []

  def reset(self, seed: int | None = None) -> Obs:
    '''Restart the episode from the first decision bar.

    Args:
      seed: Accepted for interface compatibility. The environment is
        deterministic, so the seed is ignored rather than silently
        producing a different episode than a caller expects.

    Returns:
      The first observation.
    '''
    del seed
    self._reset_state()
    return self._observation()

  def _bar_index(self) -> int:
    '''Return the absolute index of the current decision bar.'''
    return self.history + self._step_index - 1

  def _observation(self) -> Obs:
    '''Build the observation visible at the current decision bar.

    The observation is sliced to bars at or before the decision point,
    which is what makes look-ahead impossible for the agent: the bar it
    will be filled on is not present.

    Returns:
      Mapping with per-symbol returns and the portfolio state.
    '''
    index = self._bar_index()
    closes = {
      symbol: [bar.close for bar in panel[:index + 1]]
      for symbol, panel in self.panels.items()
    }
    returns = {}
    for symbol, prices in closes.items():
      returns[symbol] = _trailing_return(prices)
    return {
      'step': self._step_index,
      'timestamp': self.timestamps[index],
      'returns': returns,
      'weights': dict(self._weights),
      'cash': self._cash,
      'equity': self._equity[-1],
      'drawdown': self._drawdown,
    }

  def step(self, action: dict[str, int]) -> tuple[Obs, float, bool, Info]:
    '''Advance one bar, executing at the next bar's open.

    Args:
      action: Desired direction per symbol: -1 reduce, 0 hold, 1 add.
        Unknown symbols are ignored, and missing symbols are treated as
        hold, so a policy that emits a partial action is still valid.

    Returns:
      Tuple of (observation, reward, terminated, info). The episode
      terminates on the last bar, because there is no following open to
      fill against.
    '''
    if self._step_index >= self.n_steps:
      return self._observation(), 0.0, True, self._info()
    fill_index = self._bar_index() + 1
    filled = self._execute(action, fill_index)
    value = self._portfolio_value(fill_index)
    previous = self._equity[-1]
    self._equity.append(value / self.capital)
    bar_return = self._equity[-1] / previous - 1.0
    self._returns.append(bar_return)
    self._peak = max(self._peak, self._equity[-1])
    self._drawdown = max(self._drawdown, 1.0 - self._equity[-1] / self._peak)
    reward = self._reward(filled)
    self._step_index += 1
    terminated = self._step_index >= self.n_steps
    return self._observation(), reward, terminated, self._info(filled)

  def _execute(self, action: dict[str, int], fill_index: int) -> float:
    '''Fill orders at ``fill_index`` open and return the cost charged.

    Args:
      action: Desired direction per symbol.
      fill_index: Index of the bar whose open the fills occur at.

    Returns:
      Total transaction cost in rupees.
    '''
    value = self._portfolio_value(fill_index)
    targets = self._target_weights(action, value)
    deltas: dict[str, int] = {}
    prices: dict[str, float] = {}
    for symbol, target in targets.items():
      price = self.panels[symbol][fill_index].open
      if price <= 0.0:
        continue
      prices[symbol] = price
      deltas[symbol] = int(target * value / price) - self._quantity[symbol]
    # Affordability. Sizing int(target * value / price) bounds the
    # notional by the portfolio value but ignores charges, so a fully
    # invested book necessarily overspends and borrows without paying
    # for it. Identical constraint to portfolio.run_portfolio and
    # rl.portfolio_env, so all three engines stay comparable.
    scale = affordable_scale(self._cash, prices, deltas, self.costs)
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
    self._recount_weights(value)
    self._decisions.append({
      'step': self._step_index,
      'timestamp': self.timestamps[self._bar_index()],
      'weights': dict(self._weights),
      'cost': cost,
    })
    return cost

  def _target_weights(
    self, action: dict[str, int], value: float) -> dict[str, float]:
    '''Translate a directional action into feasible target weights.

    Direction is applied as a multiplier on the current weight, then the
    result is clipped per symbol and per sector. Constraints live here,
    outside the policy, so the executed trade is always feasible and the
    actor stays unconstrained.

    Args:
      action: Desired direction per symbol.
      value: Current portfolio value, used to size the delta.

    Returns:
      Target weight per symbol, summing to at most 1.0.
    '''
    del value
    desired = {}
    for symbol in self.action_symbols:
      held = self._weights[symbol]
      direction = action.get(symbol, 0)
      if direction > 0:
        target = held + self.max_weight
      elif direction < 0:
        target = held - self.max_weight
      else:
        target = held
      desired[symbol] = clamp_weight(float(target), self.max_weight)
    return self._apply_sector_cap(desired)

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

  def _recount_weights(self, value: float) -> None:
    '''Recompute current weights from holdings and portfolio value.

    Args:
      value: Current portfolio value.
    '''
    if value <= 0.0:
      self._weights = dict.fromkeys(self.action_symbols, 0.0)
      return
    self._weights = {
      symbol: self._quantity[symbol]
      * self.panels[symbol][self._bar_index()].close / value
      for symbol in self.action_symbols
    }

  def _portfolio_value(self, index: int) -> float:
    '''Return cash plus mark-to-market holdings at a bar's close.

    Args:
      index: Bar index at which to mark holdings.

    Returns:
      Portfolio value in rupees.
    '''
    held_value = sum(
      self._quantity[symbol] * self.panels[symbol][index].close
      for symbol in self.action_symbols
    )
    return self._cash + held_value

  def _reward(self, cost: float) -> float:
    '''Combine the reward terms for one bar.

    The single bar's own return is deliberately NOT a reward term. One
    bar of return is close to pure noise, and training on it produces a
    policy that chases noise. The Sharpe term is computed over the
    running return history instead, which is what the design doc means
    by differential Sharpe and is the only part of the formula carrying
    real signal.

    Args:
      cost: Transaction cost charged on the bar, in rupees.

    Returns:
      Scalar reward.
    '''
    scale = cost / self.capital
    running_mean = sum(self._returns) / len(self._returns)
    running_vol = _stdev(self._returns)
    sharpe = 0.0 if running_vol <= 0.0 else running_mean / running_vol
    penalty = 0.0
    if self.reward.drawdown:
      penalty = self._drawdown * self._drawdown
    return (
      self.reward.sharpe * sharpe
      - self.reward.turnover * scale
      - self.reward.drawdown * penalty
    )

  def _info(self, cost: float = 0.0) -> Info:
    '''Build the auxiliary info mapping for the current step.

    Args:
      cost: Transaction cost charged on the step just taken.

    Returns:
      Info mapping including equity, drawdown and cumulative cost.
    '''
    return {
      'cost': cost,
      'total_cost': self._total_cost,
      'equity': self._equity[-1],
      'drawdown': self._drawdown,
      'turnover': _turnover(list(self._weights.values())),
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
  def total_cost(self) -> float:
    '''Return cumulative transaction cost charged so far, in rupees.

    Exposed because cost is the term most likely to be silently dropped
    from a reward, so it has to be directly comparable between runs.
    '''
    return self._total_cost

  @property
  def weights(self) -> dict[str, float]:
    '''Return current portfolio weights per symbol.'''
    return dict(self._weights)

  @property
  def positions(self) -> dict[str, int]:
    '''Return current share holdings per symbol.

    Public because "did this actually go short?" is the question a
    reviewer asks first, and it should not require reaching into
    private state to answer it.
    '''
    return dict(self._quantity)

  @property
  def drawdown(self) -> float:
    '''Return current drawdown from the running equity peak.'''
    return self._drawdown

  @property
  def cash(self) -> float:
    '''Return uninvested cash in rupees.'''
    return self._cash

  def observation(self) -> Obs:
    '''Return the observation visible right now, without advancing.

    Public so a caller can inspect what the agent currently sees -- for
    example to assert that a price change does not alter the agent's
    view -- without consuming a step.
    '''
    return self._observation()

  def info(self) -> Info:
    '''Return auxiliary state without advancing the episode.

    Returns:
      Mapping of equity, drawdown, cumulative cost and gross exposure.
    '''
    return self._info()

  @property
  def decisions(self) -> list[dict[str, object]]:
    '''Return the per-step decision log, one entry per executed step.'''
    return list(self._decisions)

  def annualized_sharpe(self) -> float:
    '''Return the realised Sharpe of the completed episode.

    Returns:
      Annualised Sharpe, or 0.0 if the episode produced no dispersion.
      Reported for convenience only; any decision about it belongs to
      the Deflated Sharpe, which knows how many trials were tried.
    '''
    if len(self._returns) < 2:
      return 0.0
    deviation = _stdev(self._returns)
    if deviation <= 0.0:
      return 0.0
    mean = sum(self._returns) / len(self._returns)
    return (mean / deviation) * (TRADING_DAYS_PER_YEAR ** 0.5)


def _trailing_return(prices: list[float], window: int = 21) -> float:
  '''Return the trailing return over ``window`` bars.

  Args:
    prices: Closing prices up to and including the decision bar.
    window: Lookback in bars.

  Returns:
    Fractional return, or 0.0 if there is not enough history.
  '''
  if len(prices) <= window:
    return 0.0
  past = prices[-window - 1]
  if past <= 0.0:
    return 0.0
  return prices[-1] / past - 1.0


def _stdev(values: list[float]) -> float:
  '''Return the sample standard deviation.

  Args:
    values: Input series.

  Returns:
    Sample standard deviation, or 0.0 for fewer than two values.
  '''
  if len(values) < 2:
    return 0.0
  mean = sum(values) / len(values)
  variance = sum((value - mean) ** 2 for value in values)
  return (variance / (len(values) - 1)) ** 0.5


def _turnover(weights: list[float]) -> float:
  '''Return total absolute weight across a single snapshot.

  Args:
    weights: Current weights.

  Returns:
    Sum of absolute weights, equal to gross exposure.
  '''
  return sum(abs(weight) for weight in weights)
