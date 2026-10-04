#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Decision-side abstractions: the Policy seam and the deployment hash.

This module exists for one reason beyond convenience. The environments in
this package are only worth building if a learned policy can be compared
against the real baselines in :mod:`stock_rl.baselines`, and a learned
policy is only worth deploying if you can prove the code that produced the
orders is the code the exchange was told about. Both needs are served
here: a ``Policy`` Protocol that wraps the evidenced baselines, and a
fingerprint that is stable across processes and changes when any
parameter changes.

**The fingerprint is a compliance control, not an explainability toy.**
NSE operational modalities 9.1 and 9.9 require a fresh exchange
re-registration for any change to the decision logic of a black-box algo.
Answering "is the running process the registered one?" is otherwise
unanswerable, because a Python object gives you a memory address, and
``hash()`` on a string is salted per process. The fingerprint here is a
SHA-256 over a canonical serialisation of the policy class and its
parameters, with a schema prefix so a hashing-convention change cannot be
silently compared against an older registration record.

**Weights are unconstrained on purpose.** Nothing in this module clips,
projects or normalises. Feasibility is the environment's job
(:mod:`stock_rl.rl.portfolio_env`, :mod:`stock_rl.env.nse`), because a
QP projection inside the policy is not differentiable and back-propagating
through one makes the policy gradient silently wrong.

PONYTAIL: ``RandomWeights`` exists so that random search can produce a
genuine trial log for the Deflated Sharpe, not because a random policy is
useful. Ceiling: it draws in sorted symbol order, so the same seed
reproduces the same book only for the same universe. Upgrade path: seed the
draw from the observation hash, which is additive to ``act`` alone.
'''

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Protocol, runtime_checkable

from stock_rl import baselines
from stock_rl.bars import Bar
from stock_rl.env.gym import Info, Obs

__all__ = [
  'Action',
  'ConstantWeights',
  'ContinuousEnv',
  'InverseVolPolicy',
  'MomentumPolicy',
  'Policy',
  'RandomWeights',
  'fingerprint_schema',
  'panels_observation',
  'policy_fingerprint',
  'policy_parameters',
]

#: Version tag prefixed to every fingerprint. Bump it when the hashing
#: convention changes, so an old registration record cannot be compared
#: against a new fingerprint as though they were the same scheme.
fingerprint_schema = 'stock_rl.policy.fingerprint/1'

#: Target weight per symbol, as a fraction of portfolio value. Deliberately
#: unconstrained: feasibility belongs to the environment.
Action = dict[str, float]


@runtime_checkable
class Policy(Protocol):
  '''Maps what is visible now to what should be held.

  The observation contract is deliberately narrow, because the whole point
  is that the evidenced baselines can be written down as policies:

    ``symbols``
      Tuple of tradeable names, in a fixed sorted order, so an emitted
      action is always interpretable.
    ``history``
      Mapping of name to the bars visible at the decision point. It
      excludes the bar the order will be filled on, which is what makes
      look-ahead impossible for a policy that reads it.

  Args:
    observation: Mapping described above.

  Returns:
    Target weight per symbol. Values may fall outside ``[0, 1]``; the
    environment clamps them.
  '''

  def act(self, observation: Obs) -> Action:
    '''Return target weights for the visible observation.

    Args:
      observation: Visible state, sliced to the decision bar.

    Returns:
      Desired weight per symbol.
    '''


@runtime_checkable
class ContinuousEnv(Protocol):
  '''``Env`` narrowed to continuous weight or coverage actions.

  :class:`stock_rl.env.gym.Env` annotates ``step`` with a discrete
  ``dict[str, int]`` action because that is what
  :class:`stock_rl.env.nse.PortfolioEnv` takes. This protocol is the
  continuous counterpart, so a continuous learner has a type to code
  against. It is not a replacement for ``Env``: ``Env`` is
  ``runtime_checkable`` and structural, so both environments satisfy it
  without modification, and only the Gym signature annotation differs.

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

  def render(self) -> str:
    '''Return a one-line summary of the episode so far.

    Returns:
      Human-readable summary.
    '''


def panels_observation(panels: Mapping[str, Sequence[Bar]]) -> Obs:
  '''Build the minimal observation a policy needs from visible panels.

  This is the adapter that lets one policy serve both the RL environments
  and :func:`stock_rl.portfolio.run_portfolio`, which hands a strategy a
  plain mapping of symbol to bars rather than an observation.

  Args:
    panels: Mapping of symbol to the bars visible at the decision point.
      Must exclude the fill bar.

  Returns:
    Observation carrying only ``symbols`` and ``history``.
  '''
  history = {symbol: tuple(bars) for symbol, bars in panels.items()}
  return {
    'symbols': tuple(sorted(history)),
    'history': history,
  }


def _symbols(observation: Obs) -> tuple[str, ...]:
  '''Return the tradeable names in an observation.

  Args:
    observation: Observation from an environment or
      :func:`panels_observation`.

  Returns:
    Sorted symbol names.
  '''
  declared = observation.get('symbols')
  if isinstance(declared, Sequence) and not isinstance(declared, str):
    return tuple(str(name) for name in declared)
  history = observation.get('history', {})
  if isinstance(history, Mapping):
    return tuple(sorted(str(name) for name in history))
  return ()


def _panels(observation: Obs) -> dict[str, tuple[Bar, ...]]:
  '''Return the visible bars in the shape the baselines expect.

  Args:
    observation: Observation carrying a ``history`` mapping.

  Returns:
    Mapping of symbol to visible bars.

  Raises:
    KeyError: If the observation carries no ``history``.
  '''
  history = observation['history']
  if not isinstance(history, Mapping):
    raise KeyError('observation has no history mapping')
  return {str(name): tuple(bars) for name, bars in history.items()}


@dataclass(frozen=True, slots=True)
class ConstantWeights:
  '''Hold a fixed weight in every symbol, the 1/N control arm.

  This is :func:`stock_rl.baselines.equal_weight` expressed as a policy,
  and it is here to be beaten. DeMiguel (2009) is the reason it is the
  kill criterion rather than a warm-up.

  Attributes:
    weight: Weight per symbol, before the environment's feasibility caps.
  '''

  weight: float = 0.1

  def act(self, observation: Obs) -> Action:
    '''Return the same weight for every visible symbol.

    Args:
      observation: Visible state.

    Returns:
      Desired weight per symbol.
    '''
    return {symbol: self.weight for symbol in _symbols(observation)}


@dataclass(frozen=True, slots=True)
class MomentumPolicy:
  '''Cross-sectional momentum, wrapping the momentum baseline.

  :func:`stock_rl.baselines.momentum_ranked`, the strongest anomaly
  family in Indian data and the first thing any learned allocator has to
  clear.

  The strongest anomaly family in Indian data and the first thing any
  learned allocator has to clear.

  Attributes:
    lookback: Total momentum lookback in bars.
    skip: Most recent bars excluded, per the 12-1 construction.
    top: Number of symbols held.
    max_weight: Cap the baseline itself applies before the environment
      sees the weights.
  '''

  lookback: int = 252
  skip: int = 21
  top: int = 10
  max_weight: float = 0.10

  def act(self, observation: Obs) -> Action:
    '''Rank on trailing momentum and weight the leaders.

    Args:
      observation: Visible state.

    Returns:
      Desired weight per symbol, with unheld symbols at zero.
    '''
    return baselines.momentum_ranked(
      _panels(observation), self.lookback, self.skip, self.top,
      self.max_weight)


@dataclass(frozen=True, slots=True)
class InverseVolPolicy:
  '''Inverse realised volatility, wrapping the low-volatility baseline.

  :func:`stock_rl.baselines.low_volatility`, the guardrail baseline the
  RL doc recommends as the control arm.

  The guardrail baseline the RL doc recommends as the control arm: five
  lines, historically close to minimum-variance without the estimation
  error that destroys sample MVO.

  Attributes:
    window: Lookback in bars for the volatility estimate.
    max_weight: Cap the baseline itself applies.
  '''

  window: int = 60
  max_weight: float = 0.10

  def act(self, observation: Obs) -> Action:
    '''Weight each symbol inversely to its realised volatility.

    Args:
      observation: Visible state.

    Returns:
      Desired weight per symbol.
    '''
    return baselines.low_volatility(
      _panels(observation), self.window, self.max_weight)


@dataclass(frozen=True, slots=True)
class RandomWeights:
  '''Seeded random weights, for honest random search.

  Not a strategy. It exists because a Deflated Sharpe needs a trial log,
  and a trial log needs a family of configurations that is not secretly
  tuned. The draw walks symbols in sorted order, so a seed reproduces a
  book exactly for a given universe.

  Attributes:
    seed: Seed for the draw.
    max_weight: Largest weight any symbol may receive.
  '''

  seed: int = 0
  max_weight: float = 0.10

  def act(self, observation: Obs) -> Action:
    '''Return one pseudo-random weight per visible symbol.

    Args:
      observation: Visible state.

    Returns:
      Desired weight per symbol in ``[0, max_weight]``.
    '''
    source = random.Random(self.seed)
    return {
      symbol: source.uniform(0.0, self.max_weight)
      for symbol in _symbols(observation)
    }


def policy_parameters(policy: object) -> dict[str, object]:
  '''Return the declared parameters of a policy.

  A learned policy will not be a dataclass, so the second branch is the
  contract it must satisfy: expose a mapping named ``parameters``. Failing
  loudly here is deliberate, because a policy that cannot be fingerprinted
  cannot be registered.

  Args:
    policy: Any policy object.

  Returns:
    Mapping of parameter name to value.

  Raises:
    TypeError: If the policy exposes no introspectable parameters.
  '''
  if is_dataclass(policy) and not isinstance(policy, type):
    return {field.name: getattr(policy, field.name) for field in fields(policy)}
  declared = getattr(policy, 'parameters', None)
  if isinstance(declared, Mapping):
    return {str(key): value for key, value in declared.items()}
  raise TypeError(
    f'{type(policy).__name__} exposes no parameters: make it a frozen '
    'dataclass or give it a mapping-valued "parameters" attribute, '
    'otherwise its deployment fingerprint cannot be computed')


def _plain(value: object) -> object:
  '''Convert a value into JSON-stable primitives.

  Floats go through ``repr`` rather than ``str`` so the digest records the
  exact double, and mappings are keyed by ``str`` so a symbol-keyed
  parameter hashes identically regardless of insertion order.

  Args:
    value: Arbitrary parameter value.

  Returns:
    A structure built only of dict, list, str and float.
  '''
  if is_dataclass(value) and not isinstance(value, type):
    return {
      field.name: _plain(getattr(value, field.name))
      for field in fields(value)
    }
  if isinstance(value, Mapping):
    return {str(key): _plain(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_plain(item) for item in value]
  if isinstance(value, float):
    return repr(value)
  return value


def policy_fingerprint(policy: object) -> str:
  '''Return a stable hash of a policy's identity and parameters.

  The returned string is ``<schema>:<sha256 hex>``. The class identity is
  part of the payload because two policy classes can accept the same
  parameters and decide differently, and a fingerprint that treated them
  as equal would let a registration record match the wrong code.

  Args:
    policy: Any policy object exposing its parameters, see
      :func:`policy_parameters`.

  Returns:
    Stable fingerprint string.
  '''
  payload = {
    'schema': fingerprint_schema,
    'class': f'{type(policy).__module__}.{type(policy).__qualname__}',
    'parameters': _plain(policy_parameters(policy)),
  }
  text = json.dumps(payload, sort_keys=True, separators=(',', ':'))
  return f'{fingerprint_schema}:{hashlib.sha256(text.encode()).hexdigest()}'
