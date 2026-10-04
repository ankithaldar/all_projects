#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Trial accounting and honest evaluation. No gradient descent here.

The single most valuable thing in this module is the length gate. Bailey,
Borwein, Lopez de Prado and Zhu (2014) show that searching N configurations
requires a minimum sample length before any Sharpe is worth anything, and
:func:`stock_rl.metrics.minimum_backtest_length` computes it. When the
required history exceeds the history available, :func:`sharpe_report`
returns no Sharpe at all and says why. A module that reported a Sharpe it
knew to be unsupported would be worse than a module with no training code,
because the number would get quoted.

**No gradient descent, deliberately.** PPO, SAC and DDPG are not
implemented here, and the ceiling is not squeamishness: a correct
implementation needs batched tensors, autodiff and a rollout buffer, which
means numpy or torch, and this project has zero runtime dependencies by
design. What is provided instead is the seam that such a learner plugs
into -- a ``Policy`` Protocol (:mod:`stock_rl.rl.policy`), two
environments with continuous actions, a rollout loop, and a random search
that produces the trial log the Deflated Sharpe needs. The learning
algorithm is the part that should be bought with a dependency, on the
evidence that it is the part worth buying: Grądzki (2026) holds
architecture, data and hyperparameters fixed and still measures a Sharpe
standard deviation of 0.193 on a mean of 0.604, and reports that selecting
the best seed instead of reporting the mean inflates the Sharpe by 44
percent. That noise is larger than the RL-versus-HRP effect anyone is
trying to detect, so an honest evaluator matters more than an extra
algorithm.

Consequently :func:`sharpe_report` reports the **mean across seeds** of
the selected configuration, never the best seed, and carries the selection
inflation alongside so the size of the temptation is visible.

**A length gate is not a formality.** With 16 trials and a claimed Sharpe
of 1.0 the required history is on the order of a decade; with 200 bars of
data no Sharpe is defensible at all, however good the equity curve looks.
'''

from __future__ import annotations

import random
from dataclasses import dataclass, field
from statistics import fmean, stdev
from typing import Protocol, runtime_checkable

from stock_rl import baselines
from stock_rl.bars import Bar
from stock_rl.costs import DELIVERY, CostModel
from stock_rl.env.gym import Info, Obs
from stock_rl.metrics import (
  TRADING_DAYS_PER_YEAR,
  deflated_sharpe,
  max_drawdown,
  minimum_backtest_length,
  sharpe_ratio,
  total_return,
)
from stock_rl.portfolio import run_portfolio
from stock_rl.rl.policy import (
  Action,
  ConstantWeights,
  InverseVolPolicy,
  MomentumPolicy,
  Policy,
  RandomWeights,
  panels_observation,
  policy_fingerprint,
)
from stock_rl.rl.portfolio_env import WeightAllocationEnv

__all__ = [
  'EngineComparison',
  'ReplayEnv',
  'SharpeReport',
  'Trial',
  'TrialLog',
  'compare',
  'equal_weight_sharpe',
  'policy_space',
  'random_search',
  'rollout',
  'sharpe_report',
]

#: Momentum lookbacks in the random search grid.
lookback_grid = (60, 120, 252)

#: Skip lengths in the random search grid, per the 12-1 construction.
skip_grid = (0, 21)

#: Holdings counts in the random search grid.
top_grid = (5, 10)

#: Volatility windows in the random search grid.
vol_window_grid = (20, 60, 120)

#: Per-symbol weight caps in the random search grid.
weight_grid = (0.05, 0.10, 0.15)

#: Policy families in the random search grid. Equal weight is in there on
#: purpose: DeMiguel (2009) is the control arm every learned allocator has
#: to clear, so a search that never evaluates it is not a search.
policy_space = ('equal', 'momentum', 'inverse_vol', 'random')


@runtime_checkable
class ReplayEnv(Protocol):
  '''An environment a policy can be driven to completion on.

  Attributes:
    action_symbols: Names the action keys, in a fixed order.
    n_steps: Length of one episode in steps.
  '''

  action_symbols: tuple[str, ...]
  n_steps: int

  def reset(self, seed: int | None = None) -> Obs:
    '''Restart the episode and return the first observation.

    Args:
      seed: Optional seed for stochastic environments.

    Returns:
      The initial observation.
    '''

  def step(self, action: Action) -> tuple[Obs, float, bool, Info]:
    '''Advance one bar.

    Args:
      action: Target weight or coverage ratio per action key.

    Returns:
      Tuple of (observation, reward, terminated, info).
    '''

  @property
  def equity_curve(self) -> list[float]:
    '''Return the equity curve, starting at 1.0.'''
    return list()


def _returns_from_equity(equity: list[float]) -> list[float]:
  '''Convert an equity curve into per-bar returns.

  Args:
    equity: Equity curve normalised to 1.0.

  Returns:
    Per-bar returns, with a leading 0.0 for the first bar.
  '''
  returns: list[float] = []
  for index, value in enumerate(equity):
    returns.append(0.0 if index == 0 else value / equity[index - 1] - 1.0)
  return returns


def rollout(env: ReplayEnv, policy: Policy,
            seed: int | None = None) -> list[float]:
  '''Run one episode to termination and return the per-bar returns.

  Args:
    env: Any environment satisfying ``ReplayEnv``.
    policy: Policy driving the episode.
    seed: Seed handed to ``env.reset``. Both shipped environments are
      deterministic and discard it, so today this changes nothing; it is
      threaded through anyway because a search that cannot vary its seed
      has no way to measure the dispersion a seed is supposed to cause,
      and a stochastic environment plugged in later would silently be
      evaluated once.

  Returns:
    Per-bar portfolio returns, the unit every metric in
    :mod:`stock_rl.metrics` expects.
  '''
  observation = env.reset(seed)
  done = False
  while not done:
    observation, _, done, _ = env.step(policy.act(observation))
  return _returns_from_equity(env.equity_curve)


@dataclass(frozen=True, slots=True)
class Trial:
  '''One run of one configuration under one seed.

  Attributes:
    name: Configuration label. Seeds of the same configuration share it,
      which is what makes seed dispersion computable.
    seed: Seed the run was started with.
    returns: Per-bar portfolio returns.
    sharpe: Annualised Sharpe, or 0.0 where it is undefined.
    per_period_sharpe: Per-bar Sharpe. This is the unit
      :func:`stock_rl.metrics.deflated_sharpe` needs; feeding an
      annualised figure into it would inflate the statistic.
  '''

  name: str
  seed: int
  returns: list[float]
  sharpe: float = 0.0
  per_period_sharpe: float = 0.0

  @classmethod
  def from_returns(
    cls,
    name: str,
    seed: int,
    returns: list[float],
  ) -> Trial:
    '''Build a trial, computing both Sharpe conventions.

    Args:
      name: Configuration label.
      seed: Seed the run was started with.
      returns: Per-bar portfolio returns.

    Returns:
      A ``Trial`` with both Sharpes filled in.
    '''
    if len(returns) > 1:
      deviation = stdev(returns)
      per_period = fmean(returns) / deviation if deviation > 0.0 else 0.0
    else:
      per_period = 0.0
    return cls(
      name=name,
      seed=seed,
      returns=list(returns),
      sharpe=sharpe_ratio(returns),
      per_period_sharpe=per_period,
    )


@dataclass(frozen=True, slots=True)
class TrialLog:
  '''Every run that was tried, and only the statistics that are honest.

  A *configuration* is a distinct :attr:`Trial.name`, and a *run* is one
  evaluation of one configuration under one seed. The two counts are not
  the same number and conflating them is what made this log lie: the
  search runs ``trials * len(seeds)`` times, but it only ever tried
  ``trials`` configurations, and the Deflated Sharpe is a statement about
  configurations.

  Attributes:
    trials: One entry per run, in the order they were run.
  '''

  trials: list[Trial] = field(default_factory=list)

  def __len__(self) -> int:
    '''Return the number of runs recorded.'''
    return len(self.trials)

  def __iter__(self):
    '''Iterate over the recorded trials.'''
    return iter(self.trials)

  @property
  def names(self) -> tuple[str, ...]:
    '''Return the distinct configuration names, in first-seen order.'''
    seen: dict[str, None] = {}
    for trial in self.trials:
      seen.setdefault(trial.name, None)
    return tuple(seen)

  @property
  def count(self) -> int:
    '''Return the number of independent configurations tried.

    Distinct names, not runs: the seeds of one configuration are the
    same hypothesis measured again, so counting them would inflate the
    search the Deflated Sharpe has to survive. Undercounting this weakens
    the deflation silently, which is why it is a named quantity rather
    than an argument someone can forget to pass, and overcounting it is
    no better: it invents a search that did not happen.
    '''
    return len(self.names)

  @property
  def sharpes(self) -> list[float]:
    '''Return the annualised Sharpe of every run.'''
    return [trial.sharpe for trial in self.trials]

  @property
  def per_period_sharpes(self) -> list[float]:
    '''Return the per-period Sharpe of every run.'''
    return [trial.per_period_sharpe for trial in self.trials]

  @property
  def mean_sharpe(self) -> float:
    '''Return the mean annualised Sharpe across runs.

    The mean, never the maximum. Selecting the best seed instead is worth
    about 44 percent of the reported Sharpe on Grądzki's numbers, which is
    a quantity larger than the effect anyone is hunting for.
    '''
    return fmean(self.sharpes) if self.trials else 0.0

  @property
  def stdev_sharpe(self) -> float:
    '''Return the standard deviation of the annualised Sharpe.

    Reported next to the mean because a mean on its own is the number that
    makes a lucky seed look like an edge.
    '''
    return stdev(self.sharpes) if len(self.sharpes) > 1 else 0.0

  def seed_sharpes(self, name: str) -> list[float]:
    '''Return the Sharpe of each seed of one configuration.

    Args:
      name: Configuration label.

    Returns:
      Annualised Sharpes for that label, in recorded order.
    '''
    return [trial.sharpe for trial in self.trials if trial.name == name]

  def seed_returns(self, name: str) -> list[list[float]]:
    '''Return the per-seed return series of one configuration.

    This is what :func:`sharpe_report` deflates: the selected
    configuration's own returns, not the highest-Sharpe run of some other
    configuration.

    Args:
      name: Configuration label.

    Returns:
      One return series per recorded seed, in recorded order.
    '''
    return [list(trial.returns) for trial in self.trials
            if trial.name == name]

  def seed_stdev(self, name: str | None = None) -> float:
    '''Return the dispersion of the seeds of one configuration.

    This is the number to compare an effect size against. A claimed edge
    smaller than the seed dispersion of the same policy is not an edge.

    It is zero for every configuration produced by
    :func:`random_search`, and that is a fact about the environments
    rather than a measurement: both shipped environments are
    deterministic, so re-running one configuration under three seeds
    returns the same three times. A non-zero value here is meaningful
    only when the environment actually responds to the seed, which is why
    the seeds are threaded into ``reset`` rather than merely recorded.

    Args:
      name: Configuration label. Defaults to the first recorded one.

    Returns:
      Sample standard deviation across seeds, or 0.0 for a single seed.
    '''
    label = name if name is not None else (self.names[0] if self.names else '')
    sharpes = self.seed_sharpes(label)
    return stdev(sharpes) if len(sharpes) > 1 else 0.0

  def best_trial(self) -> Trial:
    '''Return the run with the highest annualised Sharpe.

    Returns:
      The best trial.

    Raises:
      ValueError: If the log is empty.
    '''
    if not self.trials:
      raise ValueError('trial log is empty')
    return max(self.trials, key=lambda trial: trial.sharpe)

  def selection_inflation(self) -> float:
    '''Return best Sharpe divided by mean Sharpe.

    This ratio is the size of the mistake made by reporting the best run
    instead of the mean.

    Returns:
      The ratio, or 1.0 when the mean is not positive.
    '''
    mean = self.mean_sharpe
    if mean <= 0.0:
      return 1.0
    return max(self.sharpes) / mean


@dataclass(frozen=True, slots=True)
class SharpeReport:
  '''What may be reported about a search, and what may not.

  Attributes:
    sharpe: Annualised Sharpe of the selected configuration, averaged
      across its seeds, or ``None`` when the length gate refuses it.
    mean_sharpe: Mean annualised Sharpe across every run.
    stdev_sharpe: Standard deviation across every run.
    seed_stdev: Dispersion of the seeds of the selected configuration.
    best_sharpe: Highest single run, reported only so the gap is visible.
    selection_inflation: ``best_sharpe / mean_sharpe``.
    trials: Number of independent configurations tried.
    required_years: Years of history the search needs to justify itself.
    available_years: Years of history actually available.
    deflated: Deflated Sharpe probability of the selected
      configuration, or ``None`` when the gate refused it. Computed
      from that configuration's own per-seed returns, and reported as
      the weakest of them.
    reason: Why the gate did or did not refuse.
  '''

  sharpe: float | None
  mean_sharpe: float
  stdev_sharpe: float
  seed_stdev: float
  best_sharpe: float
  selection_inflation: float
  trials: int
  required_years: float
  available_years: float
  deflated: float | None
  reason: str


def sharpe_report(log: TrialLog) -> SharpeReport:
  '''Report a Sharpe only if the search history supports one.

  The gate is Bailey, Borwein, Lopez de Prado and Zhu (2014): searching
  ``trials`` configurations to find a Sharpe of ``s`` needs
  ``(E[max N normals] / s) ** 2`` years of returns before the result means
  anything. If that exceeds the history available, this returns a report
  whose ``sharpe`` is ``None``, and a ``reason`` naming the shortfall. No
  amount of implementation correctness rescues a sample that short.

  ``deflated`` is computed from the selected configuration's own per-seed
  returns and reported as the weakest of them. Both halves matter: a
  Deflated Sharpe over a different strategy from the one it is quoted for
  is decoration, and quoting the best seed of a configuration is the
  selection this whole module exists to refuse.

  Args:
    log: Every run that was tried.

  Returns:
    A ``SharpeReport``. ``sharpe`` is ``None`` when it may not be reported.

  Raises:
    ValueError: If the log is empty.
  '''
  if not log.trials:
    raise ValueError('trial log is empty')
  best = log.best_trial()
  selected = max(log.names, key=lambda name: fmean(log.seed_sharpes(name)))
  sharpes = log.seed_sharpes(selected)
  reported = fmean(sharpes)
  # Available history and the deflated probability both come from the
  # SELECTED configuration. Taking them from best_trial() mixes two
  # strategies in one report: the Sharpe on one line is the mean across
  # the seeds of `selected`, and the probability underneath it was
  # computed from the single luckiest run of whichever configuration got
  # the highest number. A Deflated Sharpe that deflates a different
  # strategy from the one it is reported for is decoration.
  seed_returns = log.seed_returns(selected)
  available = (min(len(series) for series in seed_returns)
               / TRADING_DAYS_PER_YEAR)
  trials = log.count
  base = {
    'mean_sharpe': log.mean_sharpe,
    'stdev_sharpe': log.stdev_sharpe,
    'seed_stdev': stdev(sharpes) if len(sharpes) > 1 else 0.0,
    'best_sharpe': best.sharpe,
    'selection_inflation': log.selection_inflation(),
    'trials': trials,
    'available_years': available,
  }
  if trials < 2:
    return SharpeReport(
      sharpe=None, required_years=0.0, deflated=None,
      reason='fewer than two configurations were tried, so there is no '
             'search to deflate and no Sharpe is reportable',
      **base)
  if reported <= 0.0:
    return SharpeReport(
      sharpe=None, required_years=0.0, deflated=None,
      reason=f'selected configuration mean Sharpe {reported:.3f} is not '
             f'positive, so there is nothing to defend',
      **base)
  required = minimum_backtest_length(trials, reported)
  if required > available:
    return SharpeReport(
      sharpe=None, required_years=required, deflated=None,
      reason=f'{trials} trials at Sharpe {reported:.3f} need {required:.2f} '
             f'years of returns, only {available:.2f} are available',
      **base)
  # Deflate every seed of the selected configuration, and report the
  # weakest of them. The seeds are the same hypothesis re-measured, so the
  # honest reading of "does this survive its own search" is the seed that
  # looks least convincing.
  scored = [
    value for value in (
      deflated_sharpe(series, trials,
                      trial_sharpes=log.per_period_sharpes)
      for series in seed_returns)
    if value is not None
  ]
  return SharpeReport(
    sharpe=reported,
    required_years=required,
    deflated=min(scored) if scored else None,
    reason=f'{available:.2f} years available against {required:.2f} required',
    **base)


def _draw_policy(source: random.Random) -> Policy:
  '''Draw one configuration from the random search grid.

  Args:
    source: Seeded random source.

  Returns:
    A policy from one of the four families in ``policy_space``.
  '''
  family = source.choice(policy_space)
  weight = source.choice(weight_grid)
  if family == 'equal':
    return ConstantWeights(weight=weight)
  if family == 'momentum':
    return MomentumPolicy(
      lookback=source.choice(lookback_grid),
      skip=source.choice(skip_grid),
      top=source.choice(top_grid),
      max_weight=weight,
    )
  if family == 'inverse_vol':
    return InverseVolPolicy(
      window=source.choice(vol_window_grid), max_weight=weight)
  return RandomWeights(seed=source.randrange(2 ** 31), max_weight=weight)


def random_search(
  panels: dict[str, list[Bar]],
  trials: int = 8,
  seeds: tuple[int, ...] = (1, 2, 3),
  capital: float = 10_000_000.0,
  costs: CostModel = DELIVERY,
  history: int = 60,
  max_weight: float = 0.10,
  rebalance_days: int = 21,
  seed: int = 0,
) -> TrialLog:
  '''Run random search over the policy space and log every run.

  This is a driver, not a learner. Its job is to produce an honest trial
  log, because a Deflated Sharpe over one run is decoration.

  One configuration is drawn per trial and evaluated under **every**
  seed, which is the only reading in which a seed is a seed. The policy
  used to be drawn inside the seed loop, so the three entries sharing one
  ``name`` were three unrelated configurations: ``TrialLog.count``
  counted one trial where three had been evaluated, which weakens the
  deflation, and ``seed_stdev`` reported the dispersion of *which
  configuration got drawn* while claiming to report seed dispersion.

  Both shipped environments are deterministic and discard the seed, so
  the seeds of one configuration now produce identical runs and
  ``seed_stdev`` is exactly ``0.0``. That is the honest number, and it is
  why the seed is still recorded: it is the key that would vary the run
  the moment a stochastic environment is plugged in, and a search that
  cannot vary its seed cannot honestly report what varying it would do.

  Args:
    panels: Mapping of symbol to bars, all sharing one timeline.
    trials: Number of configurations to draw.
    seeds: Seeds each configuration is evaluated under.
    capital: Starting cash in rupees.
    costs: Transaction cost model.
    history: Warm-up bars before the first decision.
    max_weight: Per-symbol cap applied by the environment.
    rebalance_days: Bars between rebalances.
    seed: Seed for the configuration draw.

  Returns:
    A ``TrialLog`` with ``trials * len(seeds)`` runs covering exactly
    ``trials`` configurations.
  '''
  if trials < 1:
    raise ValueError(f'trials must be >= 1, got {trials}')
  if not seeds:
    raise ValueError('seeds must not be empty')
  source = random.Random(seed)
  log: list[Trial] = []
  for index in range(trials):
    name = f'trial_{index:03d}'
    policy = _draw_policy(source)
    for trial_seed in seeds:
      env = WeightAllocationEnv(
        panels,
        capital=capital,
        costs=costs,
        history=history,
        max_weight=max_weight,
        rebalance_days=rebalance_days,
      )
      returns = rollout(env, policy, trial_seed)
      log.append(Trial.from_returns(name, trial_seed, returns))
  return TrialLog(log)


@dataclass(frozen=True, slots=True)
class EngineComparison:
  '''The same policy measured by two engines over one window.

  Attributes:
    env: Metrics from the RL environment: ``sharpe``, ``max_drawdown``,
      ``total_return``, ``turnover`` and ``total_cost``.
    portfolio: The same five keys from
      :func:`stock_rl.portfolio.run_portfolio`, over the same window.
    fingerprint: Deployment fingerprint of the policy.
  '''

  env: dict[str, float]
  portfolio: dict[str, float]
  fingerprint: str


def compare(
  policy: Policy,
  panels: dict[str, list[Bar]],
  capital: float = 10_000_000.0,
  costs: CostModel = DELIVERY,
  history: int = 60,
  max_weight: float = 0.10,
  rebalance_days: int = 21,
) -> EngineComparison:
  '''Cross-check one policy between the two engines.

  A cost and cadence cross-check, not a parity assertion. The two engines
  share panels, costs, capital, warm-up length and rebalance cadence, and
  the cadence is aligned down to the fill *bar*: ``run_portfolio`` fills
  on bars where ``index % rebalance_days == 0``, while this environment's
  step ``s`` fills on bar ``history + s``, so the environment is built
  with ``rebalance_offset = -history % rebalance_days`` and the two books
  trade the same bars at the same opens.

  Both engines then report over the **same window**.
  :func:`stock_rl.portfolio.run_portfolio` returns one return per bar from
  ``history`` onward and the environment returns one per step, which is
  ``len - history`` of them. The previous claim that their Sharpes and
  turnover "must agree to within whole-share rounding" was not merely
  imprecise, it was unsatisfiable: one series was padded with
  ``history`` zeros and the other was not, so a gap was guaranteed by
  construction and the check carried no information about either engine.

  What remains true, and is what this function is for: **cost agrees
  exactly**, because it is a sum of rupees over the same fills on the same
  bars, and the two engines therefore trade the same book. Sharpe and
  turnover agree approximately, to the extent that the two books round
  their share counts the same way; a divergence there is a rounding
  artefact rather than the whole-share equality the old docstring
  promised, which is why it is reported side by side instead of
  asserted.

  Args:
    policy: Policy to measure.
    panels: Mapping of symbol to bars, all sharing one timeline.
    capital: Starting cash in rupees.
    costs: Transaction cost model.
    history: Warm-up bars before the first decision.
    max_weight: Per-symbol cap applied by both engines.
    rebalance_days: Bars between rebalances.

  Returns:
    An ``EngineComparison`` with both engines' Sharpe, cost, turnover,
    total return and maximum drawdown over the shared window.
  '''
  env = WeightAllocationEnv(
    panels,
    capital=capital,
    costs=costs,
    history=history,
    max_weight=max_weight,
    rebalance_days=rebalance_days,
    rebalance_offset=(-history) % rebalance_days,
  )
  rollout(env, policy)
  baseline = run_portfolio(
    panels,
    lambda visible: policy.act(panels_observation(visible)),
    capital=capital,
    costs=costs,
    rebalance_days=rebalance_days,
    max_weight=max_weight,
    history=history,
  )
  env_returns = env.returns
  # Guard the window claim rather than trusting it: the two series are
  # only comparable one for one, and a future change to either engine's
  # warm-up convention must fail here instead of silently producing two
  # different windows again.
  if len(baseline.returns) != len(env_returns):
    raise ValueError(
      f'the engines report different windows: run_portfolio returned '
      f'{len(baseline.returns)} bars and the environment '
      f'{len(env_returns)}, so no comparison is meaningful')
  return EngineComparison(
    env={
      'sharpe': sharpe_ratio(env_returns),
      'max_drawdown': max_drawdown(env.equity_curve).depth,
      'total_return': total_return(env_returns),
      'turnover': env.total_turnover,
      'total_cost': env.total_cost,
    },
    portfolio={
      'sharpe': baseline.sharpe,
      'max_drawdown': baseline.max_drawdown,
      'total_return': baseline.total_return_multiple,
      'turnover': baseline.turnover,
      'total_cost': baseline.total_cost,
    },
    fingerprint=policy_fingerprint(policy),
  )


def equal_weight_sharpe(panels: dict[str, list[Bar]]) -> float:
  '''Return the annualised Sharpe of the 1/N control arm.

  Exposed as a named function because it is the kill criterion: DeMiguel
  (2009) is the reference result, and a learned policy that cannot clear
  equal weight has not earned its complexity.

  Args:
    panels: Mapping of symbol to bars, all sharing one timeline.

  Returns:
    Annualised Sharpe of equal weight under ``run_portfolio`` defaults.
  '''
  return run_portfolio(panels, baselines.equal_weight).sharpe
