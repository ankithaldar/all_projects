#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Reinforcement learning for NSE allocation and hedging.

Three things live here, in the order they should be read.

``rl.policy``
  The ``Policy`` Protocol, the evidenced baselines wrapped as policies, and
  the deployment fingerprint. The fingerprint is a compliance control, not
  an explainability toy: NSE operational modalities 9.1 and 9.9 require a
  fresh exchange registration for any change to the decision logic of a
  black-box algo, and a fingerprint is how you prove the running code is
  the registered code.

``rl.portfolio_env`` and ``rl.hedge_env``
  Two environments with **continuous** actions. The allocation environment
  emits target weights; the hedging environment emits coverage ratios over
  a **predefined** instrument set. Neither contains a Discrete action,
  because the squashed-Gaussian actor these designs assume cannot represent
  one, and neither applies a constraint inside the policy, because a
  projection is not differentiable and would make the policy gradient
  silently wrong.

``rl.train``
  No gradient descent, on purpose: this project has zero runtime
  dependencies, and PPO or SAC implemented without tensors would be neither
  fast nor correct. What is here instead is the evaluation discipline that
  decides whether a learned policy is real -- a trial counter, multi-seed
  mean *and* standard deviation, the Deflated Sharpe, and a
  minimum-backtest-length gate that refuses to report a Sharpe the available
  history cannot support.

Implemented as a package because the hedge environment and the evaluation
harness belong beside the allocation environment, not inside it.
'''

from stock_rl.rl.hedge_env import (
  Greeks,
  HedgeCosts,
  HedgeEnv,
  HedgeRisk,
  Instrument,
  bs_greeks,
  bs_price,
  cvar,
  default_instruments,
  semi_rmse,
)
from stock_rl.rl.policy import (
  Action,
  ConstantWeights,
  ContinuousEnv,
  InverseVolPolicy,
  MomentumPolicy,
  Policy,
  RandomWeights,
  fingerprint_schema,
  panels_observation,
  policy_fingerprint,
  policy_parameters,
)
from stock_rl.rl.portfolio_env import AllocationReward, WeightAllocationEnv
from stock_rl.rl.train import (
  EngineComparison,
  ReplayEnv,
  SharpeReport,
  Trial,
  TrialLog,
  compare,
  equal_weight_sharpe,
  random_search,
  rollout,
  sharpe_report,
)

__all__ = [
  'Action',
  'AllocationReward',
  'ConstantWeights',
  'ContinuousEnv',
  'EngineComparison',
  'Greeks',
  'HedgeCosts',
  'HedgeEnv',
  'HedgeRisk',
  'Instrument',
  'InverseVolPolicy',
  'MomentumPolicy',
  'Policy',
  'RandomWeights',
  'ReplayEnv',
  'SharpeReport',
  'Trial',
  'TrialLog',
  'WeightAllocationEnv',
  'bs_greeks',
  'bs_price',
  'compare',
  'cvar',
  'default_instruments',
  'equal_weight_sharpe',
  'fingerprint_schema',
  'panels_observation',
  'policy_fingerprint',
  'policy_parameters',
  'random_search',
  'rollout',
  'semi_rmse',
  'sharpe_report',
]
