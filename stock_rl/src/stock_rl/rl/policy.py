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

**Every baseline policy here delegates; none of them re-derives a
weight.** :class:`EqualWeightPolicy`, :class:`BuyAndHoldPolicy`,
:class:`MomentumPolicy`, :class:`TrendFilteredMomentumPolicy` and
:class:`InverseVolPolicy` each call the matching function in
:mod:`stock_rl.baselines` and return its answer unchanged. That is
what gives :mod:`stock_rl.pipeline` something to compare the two
engines against: a wrapper that recomputed the strategy would agree
with itself and prove nothing, and a second implementation that
disagreed would be a bug nobody was looking for. The corollary is that
a wrapper may add no arguments of its own, so the two wrappers whose
baselines take a cap pass that cap through by keyword and the one
whose cap belongs to a different hypothesis (:class:`BuyAndHoldPolicy`)
refuses to supply a default at all.

**Delegation is by keyword throughout, because position does not mean
the same thing twice.** The fifth positional parameter of
``momentum_ranked`` is ``max_weight`` and the fifth positional
parameter of ``trend_filtered_momentum`` is ``ma_window``, so a
positional call written by copying one wrapper's argument order feeds a
weight cap into a moving-average window and produces
``sma(closes, 0.1)`` instead of an error a caller can read.

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
  'BuyAndHoldPolicy',
  'ConstantWeights',
  'ContinuousEnv',
  'EqualWeightPolicy',
  'InverseVolPolicy',
  'MomentumPolicy',
  'Policy',
  'RandomWeights',
  'TrendFilteredMomentumPolicy',
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
class EqualWeightPolicy:
  '''Equal weights across the universe, wrapping the control arm.

  :func:`stock_rl.baselines.equal_weight`, the 1/N rule that DeMiguel
  (2009) makes the kill criterion. **This wrapper exists for parity
  testing and must not be presented as a strategy.** Its entire value
  is that it is a second code path reaching the same function, so a
  divergence between the two engines surfaces on the one baseline
  whose answer is not in doubt.

  Do not confuse it with :class:`ConstantWeights`, which is a
  different thing: ``ConstantWeights`` writes one number into every
  symbol and never looks at how many symbols there are, while
  :func:`stock_rl.baselines.equal_weight` divides by ``N`` and caps.
  On any universe of more than ``1 / max_weight`` symbols the two
  disagree, which is precisely why the control arm needed a wrapper
  that calls the baseline rather than one more reimplementation of it.

  Attributes:
    max_weight: Cap the baseline itself applies. Whatever 1/N leaves
      unused stays in cash rather than being silently over-weighted.
  '''

  max_weight: float = 0.10

  def act(self, observation: Obs) -> Action:
    '''Weight every visible symbol at 1/N.

    Args:
      observation: Visible state.

    Returns:
      Desired weight per symbol.
    '''
    return baselines.equal_weight(
      _panels(observation), max_weight=self.max_weight)


@dataclass(frozen=True, slots=True)
class BuyAndHoldPolicy:
  '''Passive index exposure, wrapping the buy-and-hold control.

  :func:`stock_rl.baselines.buy_and_hold` delegates straight to
  :func:`stock_rl.baselines.equal_weight`, so on a long-only universe
  this emits the same book as :class:`EqualWeightPolicy` under the
  same cap. It is still a separate class, because it is a separate
  hypothesis -- momentum has to be judged against passive exposure,
  not against 1/N -- and because a separate class carries a separate
  fingerprint, so an exchange registration record cannot be satisfied
  by the wrong hypothesis.

  **Why ``max_weight`` defaults to ``None`` rather than to a number.**
  The cap on buy-and-hold is not a knob this hypothesis owns; it is
  ``equal_weight``'s cap reached through a one-line delegation, and a
  wrapper that carried its own default would be choosing a value on
  the baseline's behalf. It would also stop being weight-identical to
  a bare call the moment the baseline's default moved, which is the
  one property a parity wrapper exists to keep. So the default is
  ``None``, and ``None`` means *pass no cap at all*: :meth:`act` calls
  ``baselines.buy_and_hold(panels)`` with nothing added, which is the
  only call a wrapper can make and still honestly say it adds
  nothing. Set ``max_weight`` to impose a cap.

  PONYTAIL: this wrapper delegates, so it adds no decisions and cannot
  disagree with the baseline about what to hold. Ceiling: its only
  parameter is a cap the baseline already had, so there is nothing
  here for a random search to tune, and adding a tunable would turn
  the control arm into a strategy and forfeit the role. Upgrade path:
  none needed -- a richer passive baseline belongs in
  :mod:`stock_rl.baselines`, not in this file.
  '''

  max_weight: float | None = None

  def act(self, observation: Obs) -> Action:
    '''Hold every visible symbol, capped only if a cap was asked for.

    Args:
      observation: Visible state.

    Returns:
      Desired weight per symbol.
    '''
    panels = _panels(observation)
    if self.max_weight is None:
      return baselines.buy_and_hold(panels)
    return baselines.buy_and_hold(panels, max_weight=self.max_weight)


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
      _panels(observation),
      lookback=self.lookback,
      skip=self.skip,
      top=self.top,
      max_weight=self.max_weight,
    )


@dataclass(frozen=True, slots=True)
class TrendFilteredMomentumPolicy:
  '''Momentum behind a moving-average gate, wrapping the trend filter.

  :func:`stock_rl.baselines.trend_filtered_momentum`: Mitra (2011)'s
  SMA(40) regime gate over the same 12-1 momentum
  :class:`MomentumPolicy` already wraps. The gate is the point --
  momentum is documented to turn negative in Indian crises, and the
  cheapest guard against buying that drawdown is to stop buying.

  **The delegation is by keyword, and that is load-bearing rather
  than a style preference.** The fifth *positional* parameter of
  ``trend_filtered_momentum`` is ``ma_window``; the fifth positional
  parameter of ``momentum_ranked`` is ``max_weight``. A positional
  call that copies the argument order of :class:`MomentumPolicy`
  therefore hands ``0.10`` to a moving-average window and computes
  ``sma(closes, 0.1)`` -- a gate that is neither an error a caller
  can read nor a flat book. ``ma_window`` is declared before
  ``max_weight`` here for exactly that reason.

  PONYTAIL: this wrapper exists to make the trend filter checkable
  between :func:`stock_rl.portfolio.run_portfolio` and
  :mod:`stock_rl.rl.portfolio_env`, and it is the wrapper that earned
  its place: ``trend_filtered_momentum`` was the only one of the five
  baselines returning a **partial** weight mapping, while the
  environment read an omitted symbol as *hold current* and
  ``run_portfolio`` read it as *sell to zero*. One decision bar, two
  different books, and no policy to notice, because the only other
  four providers all emitted complete mappings. Both sides now agree on
  :data:`stock_rl.portfolio.omitted_weight` and the baseline completes
  its own mapping. Ceiling: this wrapper still repairs nothing itself,
  because a wrapper whose one job is to add nothing must not paper over
  an engine defect and then be quoted as agreement with it. Upgrade
  path: none needed; widen :mod:`stock_rl.pipeline`'s engine seam to
  call every wrapper here rather than only
  :class:`MomentumPolicy`.

  Attributes:
    window: Momentum lookback in bars.
    skip: Most recent bars excluded from the momentum.
    top: Names held while the regime permits.
    ma_window: Moving-average window of the regime gate.
    max_weight: Cap the baseline itself applies.
  '''

  window: int = 252
  skip: int = 21
  top: int = 10
  ma_window: int = 40
  max_weight: float = 0.10

  def act(self, observation: Obs) -> Action:
    '''Rank on momentum and keep only the names above their average.

    Args:
      observation: Visible state.

    Returns:
      Desired weight per symbol, one entry per symbol in the
      observation. The gate rejects by writing ``0.0``, not by leaving
      a name out.
    '''
    return baselines.trend_filtered_momentum(
      _panels(observation),
      window=self.window,
      skip=self.skip,
      top=self.top,
      ma_window=self.ma_window,
      max_weight=self.max_weight,
    )


@dataclass(frozen=True, slots=True)
class InverseVolPolicy:
  '''Inverse realised volatility, wrapping the low-volatility baseline.

  :func:`stock_rl.baselines.low_volatility`, the guardrail baseline the
  RL doc recommends as the control arm: five lines, historically close
  to minimum-variance without the estimation error that destroys
  sample MVO.

  This is the wrapper that makes low volatility cross-checkable
  between the two engines, and it is here rather than duplicated under
  a second name on purpose: two classes reaching one baseline produce
  two fingerprints for one decision logic, which is precisely the
  situation :func:`policy_fingerprint` exists to refuse.

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
      _panels(observation),
      window=self.window,
      max_weight=self.max_weight,
    )


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
