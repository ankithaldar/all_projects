#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Tests for the RL package: policies, both environments, and evaluation.

The look-ahead, feasibility and reward-symmetry tests matter more than the
rest. Every one of them covers an invariant that fails silently: a leaked
future makes every downstream comparison meaningless while leaving the
equity curve looking fine, a broken constraint produces a book nobody could
have traded, and an asymmetric reward produces a doubling strategy that
shows up as a flattering Sharpe right up until the funding runs out.
'''

from datetime import datetime, timedelta

import pytest

from stock_rl import baselines
from stock_rl.bars import Bar
from stock_rl.costs import CostModel
from stock_rl.env import Env
from stock_rl.rl import (
  AllocationReward,
  ConstantWeights,
  ContinuousEnv,
  EngineComparison,
  HedgeEnv,
  HedgeRisk,
  Instrument,
  InverseVolPolicy,
  MomentumPolicy,
  Policy,
  RandomWeights,
  ReplayEnv,
  Trial,
  TrialLog,
  WeightAllocationEnv,
  bs_greeks,
  bs_price,
  compare,
  cvar,
  default_instruments,
  equal_weight_sharpe,
  panels_observation,
  policy_fingerprint,
  policy_parameters,
  random_search,
  rollout,
  semi_rmse,
  sharpe_report,
)
from stock_rl.rl.hedge_env import min_tail_confidence, norm_cdf, norm_pdf

START = datetime(2026, 1, 1)
HISTORY = 20

FREE = CostModel(stt_buy=0.0, stt_sell=0.0, exchange_pct=0.0,
                 sebi_pct=0.0, stamp_duty_buy=0.0, gst_pct=0.0,
                 dp_charge=0.0, slippage_bps=0.0)


def panel(closes, opens=None):
  '''Return bars one day apart, each opening where the prior one closed.

  Args:
    closes: Sequence of closing prices.
    opens: Optional sequence of opening prices. Defaults to the previous
      close, which makes the open a known quantity rather than a guess.

  Returns:
    List of bars in ascending time order.
  '''
  out = []
  for index, close in enumerate(closes):
    opening = closes[index - 1] if index else closes[0]
    if opens is not None:
      opening = opens[index]
    out.append(Bar(
      START + timedelta(days=index), opening, max(opening, close),
      min(opening, close), close, 1.0,
    ))
  return out


def make_panels(count=200, symbols=('AAA', 'BBB', 'CCC')):
  '''Return aligned panels with distinguishable price paths.

  Args:
    count: Number of bars per panel.
    symbols: Symbols to create.

  Returns:
    Mapping of symbol to bars, all sharing one timeline.
  '''
  panels = {}
  for number, symbol in enumerate(symbols):
    closes = [
      100.0 + number * 40.0
      + 0.8 * index
      + 3.0 * ((-1) ** index) * (number + 1)
      for index in range(count)
    ]
    panels[symbol] = panel(closes)
  return panels


def make_env(**kwargs):
  '''Return an allocation environment over the default panels.

  Args:
    **kwargs: Forwarded to ``WeightAllocationEnv``.

  Returns:
    A constructed environment.
  '''
  return WeightAllocationEnv(make_panels(), history=HISTORY, **kwargs)


def smooth_bars(count=160, drift=0.0006, wiggle=0.0004):
  '''Return a low-volatility underlying panel.

  The hedge environment's vol mask reads trailing realised volatility, so
  a panel with a real zig-zag would sit above the 30 percent ceiling and
  every option leg would be masked out. This one does not.

  Args:
    count: Number of bars.
    drift: Log drift per bar.
    wiggle: Alternating component per bar.

  Returns:
    List of bars.
  '''
  closes = [100.0 * ((1.0 + drift) ** index) * (1.0 + wiggle * ((-1) ** index))
            for index in range(count)]
  return panel(closes)


def journey_bars(count=160, amplitude=0.05):
  '''Return a jagged underlying panel with high realised volatility.

  Args:
    count: Number of bars.
    amplitude: Size of the per-bar zig-zag.

  Returns:
    List of bars.
  '''
  closes = [100.0 * (1.0 + 0.001 * index) * (1.0 + amplitude * ((-1) ** index))
            for index in range(count)]
  return panel(closes)


def hedge_env(**kwargs):
  '''Return a hedge environment over the smooth panel.

  Args:
    **kwargs: Forwarded to ``HedgeEnv``.

  Returns:
    A constructed environment.
  '''
  return HedgeEnv(smooth_bars(), history=40, iv_window=20, **kwargs)


def synthetic_returns(count, spread=0.010, drift=0.005):
  '''Return a deterministic return series with a known positive Sharpe.

  Args:
    count: Number of returns.
    spread: Alternating magnitude about the drift.
    drift: Direction of the alternating pattern, and the series mean.

  Returns:
    List of periodic returns with mean ``drift`` and dispersion
    ``spread``.
  '''
  return [drift + spread if index % 2 == 0 else drift - spread
          for index in range(count)]


def closes_from(returns, start=100.0):
  '''Return a close series that realises exactly the given returns.

  Building the price path from the return list, rather than the other way
  round, is what makes a mirrored return series mirror exactly.

  Args:
    returns: Periodic returns to realise.
    start: First close.

  Returns:
    List of closes, one longer than ``returns``.
  '''
  closes = [start]
  for value in returns:
    closes.append(closes[-1] * (1.0 + value))
  return closes


class TestPolicyProtocol:
  '''The seam a learned policy has to satisfy.'''

  def test_baseline_policies_satisfy_the_protocol(self):
    assert isinstance(ConstantWeights(), Policy)
    assert isinstance(MomentumPolicy(), Policy)
    assert isinstance(InverseVolPolicy(), Policy)
    assert isinstance(RandomWeights(), Policy)

  def test_panels_observation_exposes_symbols_and_history(self):
    panels = make_panels(count=60)
    observation = panels_observation(panels)
    assert observation['symbols'] == ('AAA', 'BBB', 'CCC')
    assert len(observation['history']['AAA']) == 60

  def test_observation_without_symbols_falls_back_to_history(self):
    observation = {'history': {'BBB': (), 'AAA': ()}}
    assert ConstantWeights().act(observation) == {'AAA': 0.1, 'BBB': 0.1}

  def test_missing_history_is_rejected(self):
    with pytest.raises(KeyError):
      MomentumPolicy().act({'symbols': ('AAA',)})

  def test_non_mapping_history_is_rejected(self):
    with pytest.raises(KeyError):
      InverseVolPolicy().act({'symbols': ('AAA',), 'history': 'nope'})


class TestBaselinePolicies:
  '''The policies must agree with the baselines they wrap.'''

  def test_momentum_matches_the_baseline(self):
    panels = make_panels(count=200)
    policy = MomentumPolicy(lookback=60, skip=5, top=2, max_weight=0.5)
    assert policy.act(panels_observation(panels)) == \
      baselines.momentum_ranked(panels, 60, 5, 2, 0.5)

  def test_inverse_vol_matches_the_baseline(self):
    panels = make_panels(count=200)
    policy = InverseVolPolicy(window=30, max_weight=0.5)
    assert policy.act(panels_observation(panels)) == \
      baselines.low_volatility(panels, 30, 0.5)

  def test_momentum_holds_leaders_only(self):
    weights = MomentumPolicy(lookback=60, skip=5, top=1).act(
      panels_observation(make_panels(count=200)))
    assert sorted(weights.values())[-1] > 0.0
    assert sum(1 for value in weights.values() if value > 0.0) == 1

  def test_constant_weights_cover_every_symbol(self):
    weights = ConstantWeights(weight=0.25).act(
      panels_observation(make_panels()))
    assert weights == {'AAA': 0.25, 'BBB': 0.25, 'CCC': 0.25}

  def test_random_weights_are_reproducible_for_a_seed(self):
    observation = panels_observation(make_panels())
    assert RandomWeights(seed=7).act(observation) == \
      RandomWeights(seed=7).act(observation)

  def test_random_weights_differ_across_seeds(self):
    observation = panels_observation(make_panels())
    assert RandomWeights(seed=1).act(observation) != \
      RandomWeights(seed=2).act(observation)

  def test_random_weights_respect_the_cap(self):
    weights = RandomWeights(seed=3, max_weight=0.02).act(
      panels_observation(make_panels()))
    assert all(0.0 <= value <= 0.02 for value in weights.values())


class TestFingerprint:
  '''The compliance control: deployed code == registered code.'''

  def test_stable_for_identical_parameters(self):
    assert policy_fingerprint(MomentumPolicy()) == \
      policy_fingerprint(MomentumPolicy())

  def test_changes_when_a_parameter_changes(self):
    assert policy_fingerprint(MomentumPolicy(lookback=60)) != \
      policy_fingerprint(MomentumPolicy(lookback=252))

  def test_changes_when_only_the_cap_changes(self):
    assert policy_fingerprint(ConstantWeights(weight=0.1)) != \
      policy_fingerprint(ConstantWeights(weight=0.1000001))

  def test_class_identity_is_part_of_the_hash(self):
    # Same parameters, different decision logic: never the same record.
    assert policy_fingerprint(ConstantWeights(weight=0.2)) != \
      policy_fingerprint(RandomWeights(seed=0, max_weight=0.2))

  def test_carries_the_schema_prefix(self):
    fingerprint = policy_fingerprint(ConstantWeights())
    assert fingerprint.startswith('stock_rl.policy.fingerprint/1:')
    assert len(fingerprint.split(':')[1]) == 64

  def test_is_stable_across_mapping_insertion_order(self):
    class Ordered:
      '''A policy exposing parameters as an ordinary mapping.'''

      def __init__(self, mapping):
        self.parameters = dict(mapping)

      def act(self, observation):
        del observation
        return dict(self.parameters)

    first = Ordered([('b', 1), ('a', 2)])
    second = Ordered([('a', 2), ('b', 1)])
    assert policy_fingerprint(first) == policy_fingerprint(second)

  def test_parameters_are_readable(self):
    assert policy_parameters(MomentumPolicy(top=3))['top'] == 3

  def test_unintrospectable_policy_is_rejected(self):
    class Opaque:
      '''A policy that hides its parameters, which cannot be registered.'''

      def act(self, observation):
        del observation
        return {}

    with pytest.raises(TypeError, match='exposes no parameters'):
      policy_fingerprint(Opaque())


class TestAllocationConstruction:
  '''Invalid environments must fail at construction.'''

  def test_satisfies_both_protocols(self):
    env = make_env()
    assert isinstance(env, Env)
    assert isinstance(env, ContinuousEnv)

  def test_step_count_reserves_history(self):
    assert WeightAllocationEnv(make_panels(count=120),
                               history=HISTORY).n_steps == 120 - HISTORY

  def test_action_symbols_are_sorted(self):
    env = WeightAllocationEnv(make_panels(symbols=('CCC', 'AAA', 'BBB')),
                              history=HISTORY)
    assert env.action_symbols == ('AAA', 'BBB', 'CCC')

  def test_rejects_empty_panels(self):
    with pytest.raises(ValueError, match='must not be empty'):
      WeightAllocationEnv({})

  def test_rejects_misaligned_panels(self):
    panels = make_panels()
    panels['BBB'] = panels['BBB'][:50]
    with pytest.raises(ValueError, match='aligned'):
      WeightAllocationEnv(panels, history=HISTORY)

  def test_rejects_history_longer_than_data(self):
    with pytest.raises(ValueError, match='must exceed history'):
      WeightAllocationEnv(make_panels(count=15), history=HISTORY)

  @pytest.mark.parametrize('capital', [0.0, -1.0])
  def test_rejects_non_positive_capital(self, capital):
    with pytest.raises(ValueError, match='capital'):
      make_env(capital=capital)

  @pytest.mark.parametrize('weight', [0.0, -0.1, 1.5])
  def test_rejects_out_of_range_max_weight(self, weight):
    with pytest.raises(ValueError, match='max_weight'):
      make_env(max_weight=weight)

  def test_rejects_bad_rebalance_days(self):
    with pytest.raises(ValueError, match='rebalance_days'):
      make_env(rebalance_days=0)

  def test_rejects_out_of_range_rebalance_offset(self):
    with pytest.raises(ValueError, match='rebalance_offset'):
      make_env(rebalance_days=5, rebalance_offset=5)

  def test_default_drawdown_term_is_off(self):
    # The doubling pathology is why this defaults to zero.
    assert AllocationReward().drawdown == 0.0


class TestAllocationConstraints:
  '''Feasibility is enforced outside the policy.'''

  def _run(self, env, action):
    env.reset()
    while True:
      _, _, done, _ = env.step(action)
      if done:
        return

  def test_weights_sum_to_at_most_one(self):
    env = make_env(max_weight=0.9, costs=FREE)
    self._run(env, {'AAA': 0.9, 'BBB': 0.9, 'CCC': 0.9})
    assert sum(env.weights.values()) <= 1.0 + 1e-9

  def test_single_symbol_cap_holds(self):
    env = make_env(max_weight=0.10, costs=FREE)
    self._run(env, {'AAA': 1.0, 'BBB': 1.0, 'CCC': 1.0})
    assert max(env.weights.values()) <= 0.10 + 1e-9

  def test_sector_cap_holds(self):
    sectors = {'AAA': 'IT', 'BBB': 'IT', 'CCC': 'BANK'}
    env = make_env(sectors=sectors, max_sector_weight=0.20, max_weight=0.5,
                   costs=FREE)
    self._run(env, {'AAA': 0.5, 'BBB': 0.5, 'CCC': 0.5})
    it = env.weights['AAA'] + env.weights['BBB']
    assert it <= 0.20 + 1e-9

  def test_separate_sectors_are_capped_independently(self):
    sectors = {'AAA': 'IT', 'BBB': 'BANK', 'CCC': 'PHARMA'}
    env = make_env(sectors=sectors, max_sector_weight=0.05, max_weight=0.5,
                   costs=FREE)
    self._run(env, {'AAA': 0.5, 'BBB': 0.5, 'CCC': 0.5})
    assert all(value <= 0.05 + 1e-9 for value in env.weights.values())

  def test_weights_are_never_negative(self):
    env = make_env(max_weight=0.5, costs=FREE)
    self._run(env, {'AAA': -5.0, 'BBB': -1.0})
    assert all(value >= 0.0 for value in env.weights.values())

  def test_junk_action_is_treated_as_flat(self):
    env = make_env(max_weight=0.5, costs=FREE)
    env.reset()
    env.step({'AAA': 'lots', 'BBB': None, 'CCC': 0.5})
    assert env.weights['AAA'] == pytest.approx(0.0)
    assert env.weights['BBB'] == pytest.approx(0.0)

  def test_missing_symbol_is_treated_as_hold(self):
    env = make_env(costs=FREE)
    env.reset()
    env.step({'AAA': 0.5})
    assert env.weights['BBB'] == pytest.approx(0.0)
    assert env.weights['AAA'] > 0.0


class TestAllocationNoLookAhead:
  '''The order fills at the next open, sized on the known value.'''

  def test_observation_excludes_the_fill_bar(self):
    env = make_env()
    observation = env.reset()
    decision = HISTORY - 1
    assert len(observation['history']['AAA']) == decision + 1

  def test_changing_future_prices_does_not_change_past_rewards(self):
    base = make_panels(count=80)
    spiked = {
      symbol: list(bars[:HISTORY + 2]) + [
        Bar(bar.timestamp + timedelta(days=900), bar.open * 3,
            bar.high * 3, bar.low * 3, bar.close * 3, 1.0)
        for bar in bars[HISTORY + 2:]
      ]
      for symbol, bars in base.items()
    }
    first = WeightAllocationEnv(base, history=HISTORY, costs=FREE)
    second = WeightAllocationEnv(spiked, history=HISTORY, costs=FREE)
    first.reset()
    second.reset()
    action = {'AAA': 0.4, 'BBB': 0.2}
    assert first.step(action)[1] == pytest.approx(second.step(action)[1])

  def test_agent_never_sees_the_fill_price(self):
    panels = make_panels(count=80)
    env = WeightAllocationEnv(panels, history=HISTORY, costs=FREE)
    env.reset()
    env.step({'AAA': 0.4})
    before = env.observation()['history']['AAA'][-1].close
    # The current decision bar is HISTORY; the fill bar the agent has not
    # reached is one later.
    panels['AAA'][HISTORY + 1] = Bar(
      panels['AAA'][HISTORY + 1].timestamp, 9999.0, 9999.0, 9999.0, 9999.0,
      1.0)
    assert env.observation()['history']['AAA'][-1].close == before

  def test_fill_bar_close_cannot_change_the_executed_size(self):
    # Sizing on the fill bar's close instead of the decision bar's close
    # quietly buys more shares as the price rises. Rewriting the close of
    # the bar the next order fills on must leave the executed size
    # untouched, so two environments differing only in that close must
    # hold identical positions.
    plain = make_panels(count=80)
    rewritten = make_panels(count=80)
    fill = HISTORY + 1
    bar = rewritten['AAA'][fill]
    rewritten['AAA'][fill] = Bar(bar.timestamp, bar.open, 4000.0, 1.0,
                                4000.0, 1.0)
    action = {'AAA': 0.5, 'BBB': 0.5, 'CCC': 0.5}
    first = WeightAllocationEnv(plain, history=HISTORY, costs=FREE)
    second = WeightAllocationEnv(rewritten, history=HISTORY, costs=FREE)
    first.reset()
    second.reset()
    first.step(action)
    second.step(action)
    first.step(action)
    second.step(action)
    assert first.positions == second.positions

  def test_fill_bar_open_does_change_the_executed_size(self):
    # The open is the fill price, so moving it must move the trade. This
    # is the control on the previous test.
    panels = make_panels(count=80)
    quiet = WeightAllocationEnv(panels, history=HISTORY, costs=FREE)
    moved = WeightAllocationEnv(make_panels(count=80), history=HISTORY,
                                costs=FREE)
    moved.panels['AAA'][HISTORY] = Bar(
      moved.panels['AAA'][HISTORY].timestamp, 1.0, 1.0, 1.0, 1.0, 1.0)
    action = {'AAA': 0.5, 'BBB': 0.5, 'CCC': 0.5}
    quiet.reset()
    moved.reset()
    quiet.step(action)
    moved.step(action)
    assert quiet.positions['AAA'] != moved.positions['AAA']


class TestAllocationCostsAndCadence:
  '''Trading must be charged, and only on the days it happens.'''

  def test_charged_costs_are_positive(self):
    env = make_env()
    env.reset()
    assert env.step({'AAA': 0.3})[3]['cost'] > 0.0

  def test_free_model_charges_nothing(self):
    env = make_env(costs=FREE)
    env.reset()
    assert env.step({'AAA': 0.3})[3]['cost'] == 0.0

  def test_churning_costs_more_than_holding(self):
    panels = make_panels(count=80)
    churn = WeightAllocationEnv(panels, history=HISTORY)
    hold = WeightAllocationEnv(panels, history=HISTORY)
    churn.reset()
    hold.reset()
    for index in range(12):
      churn.step({'AAA': 0.4 if index % 2 else 0.0, 'BBB': 0.0})
    for _ in range(12):
      hold.step({'AAA': 0.4, 'BBB': 0.0})
    assert churn.total_cost > hold.total_cost

  def test_non_rebalance_step_trades_nothing(self):
    env = make_env(rebalance_days=5, rebalance_offset=0, costs=FREE,
                   max_weight=0.5)
    env.reset()
    env.step({'AAA': 0.4})
    assert env.total_turnover == pytest.approx(0.4)
    env.step({'AAA': 0.4})
    assert env.total_turnover == pytest.approx(0.4)

  def test_rebalance_offset_shifts_the_calendar(self):
    panels = make_panels(count=80)
    first = WeightAllocationEnv(panels, history=HISTORY, rebalance_days=4,
                                rebalance_offset=0, costs=FREE,
                                max_weight=0.5)
    second = WeightAllocationEnv(panels, history=HISTORY, rebalance_days=4,
                                 rebalance_offset=2, costs=FREE,
                                 max_weight=0.5)
    first.reset()
    second.reset()
    first.step({'AAA': 0.4})
    second.step({'AAA': 0.4})
    assert first.total_turnover == pytest.approx(0.4)
    assert second.total_turnover == pytest.approx(0.0)
    second.step({'AAA': 0.4})
    assert second.total_turnover == pytest.approx(0.0)


class TestAllocationReward:
  '''Reward composition, and the doubling pathology it must not reward.'''

  def test_reward_is_finite(self):
    env = make_env(costs=FREE)
    env.reset()
    reward = env.step({'AAA': 0.4})[1]
    assert isinstance(reward, float)
    assert reward == reward

  def test_cost_penalty_reduces_reward(self):
    panels = make_panels(count=60)
    cheap = WeightAllocationEnv(panels, history=HISTORY,
                                reward=AllocationReward(cost=0.0))
    dear = WeightAllocationEnv(panels, history=HISTORY,
                               reward=AllocationReward(cost=100.0))
    cheap.reset()
    dear.reset()
    assert dear.step({'AAA': 0.4})[1] < cheap.step({'AAA': 0.4})[1]

  def test_turnover_penalty_reduces_reward(self):
    panels = make_panels(count=60)
    cheap = WeightAllocationEnv(panels, history=HISTORY,
                                reward=AllocationReward(turnover=0.0))
    dear = WeightAllocationEnv(panels, history=HISTORY,
                               reward=AllocationReward(turnover=50.0))
    cheap.reset()
    dear.reset()
    assert dear.step({'AAA': 0.4})[1] < cheap.step({'AAA': 0.4})[1]

  def test_drawdown_penalty_defaults_to_zero(self):
    assert AllocationReward().drawdown == 0.0

  def test_reward_is_symmetric_under_mirrored_returns(self):
    # A reward that punishes loss without crediting gain makes the
    # doubling strategy optimal. Negating every return must negate the
    # reward; an asymmetric term cannot do that, a symmetric one does.
    returns = [0.10, -0.05, 0.02]
    reward = AllocationReward(turnover=0.0, cost=0.0)

    def total(series):
      env = WeightAllocationEnv(
        {'AAA': panel(closes_from(series))}, history=2, capital=1e12,
        costs=FREE, max_weight=0.5, reward=reward)
      env.reset()
      running = 0.0
      while True:
        _, step_reward, done, _ = env.step({'AAA': 0.5})
        running += step_reward
        if done:
          return running

    # Whole shares are the only asymmetry left, and a book large enough
    # that truncation is a rounding error leaves the identity exact to
    # the tolerance below. An asymmetric drawdown term breaks it by
    # orders of magnitude more, which the next test shows.
    assert total(returns) == pytest.approx(
      -total([-value for value in returns]), abs=1e-9)

  def test_drawdown_term_is_what_would_break_the_symmetry(self):
    # With the asymmetric term switched on, the mirror no longer scores
    # the negation of the original: the reward now pays the agent for not
    # being down. That is the doubling hazard, and it is why the default
    # is zero.
    returns = [0.10, -0.05, 0.02]
    reward = AllocationReward(turnover=0.0, cost=0.0, drawdown=100.0)

    def total(series):
      env = WeightAllocationEnv(
        {'AAA': panel(closes_from(series))}, history=2, capital=1e12,
        costs=FREE, max_weight=0.5, reward=reward)
      env.reset()
      running = 0.0
      while True:
        _, step_reward, done, _ = env.step({'AAA': 0.5})
        running += step_reward
        if done:
          return running

    assert total(returns) != pytest.approx(
      -total([-value for value in returns]), abs=1e-6)

  def test_growing_after_a_loss_does_not_beat_a_flat_book(self):
    # Same terminal equity, one path that doubles into the loss and one
    # that never trades. The doubling path must not out-score the flat one.
    closes = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 50.0, 50.0,
              1000.0 / 15.0]
    capital = 1_000_000.0

    def total(plan):
      env = WeightAllocationEnv(
        {'AAA': panel(closes)}, capital=capital, history=5, costs=FREE,
        max_weight=1.0)
      env.reset()
      running = 0.0
      for index, target in enumerate(plan):
        _, step_reward, done, _ = env.step({'AAA': target})
        running += step_reward
        assert not done or index == len(plan) - 1
      return running, env.equity_curve[-1]

    flat, flat_equity = total([0.0, 0.0, 0.0, 0.0])
    doubling, doubling_equity = total([0.5, 0.5, 1.0, 1.0])
    assert flat_equity == pytest.approx(doubling_equity)
    assert doubling < flat

  def test_annualised_sharpe_is_zero_before_any_step(self):
    assert make_env().annualized_sharpe() == 0.0

  def test_annualized_sharpe_is_positive_on_a_rising_book(self):
    env = make_env()
    env.reset()
    for _ in range(env.n_steps):
      env.step({'AAA': 0.3, 'BBB': 0.3, 'CCC': 0.3})
    assert env.annualized_sharpe() > 0.0

  def test_max_drawdown_is_reported(self):
    env = make_env()
    env.reset()
    for _ in range(10):
      env.step({'AAA': 0.3})
    assert env.max_drawdown >= 0.0


class TestAllocationLifecycle:
  '''Reset, termination and reporting.'''

  def test_step_count_advances_and_terminates(self):
    env = make_env()
    env.reset()
    action = {'AAA': 0.0}
    for _ in range(env.n_steps - 1):
      assert not env.step(action)[2]
    assert env.step(action)[2]

  def test_step_after_termination_is_inert(self):
    env = make_env()
    env.reset()
    action = {'AAA': 0.0}
    for _ in range(env.n_steps):
      env.step(action)
    equity = env.equity_curve[-1]
    _, reward, done, _ = env.step(action)
    assert done
    assert reward == 0.0
    assert env.equity_curve[-1] == equity

  def test_reset_restores_the_starting_state(self):
    env = make_env()
    env.reset()
    for _ in range(10):
      env.step({'AAA': 0.3})
    env.reset()
    assert env.equity_curve == [1.0]
    assert env.weights['AAA'] == pytest.approx(0.0)
    assert not env.decisions
    assert env.total_cost == 0.0
    assert env.total_turnover == 0.0

  def test_reset_ignores_the_seed(self):
    env = make_env()
    assert env.reset(seed=1)['equity'] == pytest.approx(
      env.reset(seed=99)['equity'])

  def test_equity_curve_grows_by_one_per_step(self):
    env = make_env()
    env.reset()
    env.step({'AAA': 0.3})
    assert len(env.equity_curve) == 2
    assert len(env.returns) == 1

  def test_decision_log_records_target_and_weights(self):
    env = make_env()
    env.reset()
    env.step({'AAA': 0.3})
    entry = env.decisions[0]
    assert entry['cost'] >= 0.0
    assert entry['target']['AAA'] > 0.0

  def test_render_and_info_are_available_without_stepping(self):
    env = make_env()
    env.reset()
    text = env.render()
    assert 'step' in text and 'equity' in text
    assert env.info()['gross_exposure'] == pytest.approx(0.0)
    assert env.observation()['step'] == 0

  def test_positions_and_cash_are_exposed(self):
    env = make_env()
    env.reset()
    env.step({'AAA': 0.3})
    assert env.positions['AAA'] > 0
    assert env.cash < env.capital
    assert env.drawdown >= 0.0


class TestBlackScholes:
  '''Closed-form Greeks and price.'''

  def test_cdf_bounds(self):
    assert norm_cdf(-10.0) == pytest.approx(0.0, abs=1e-12)
    assert norm_cdf(10.0) == pytest.approx(1.0)
    assert norm_cdf(0.0) == pytest.approx(0.5)

  def test_pdf_peaks_at_zero(self):
    assert norm_pdf(0.0) == pytest.approx(0.3989422804, abs=1e-9)
    assert norm_pdf(1.0) < norm_pdf(0.0)

  @pytest.mark.parametrize('spot', [50.0, 100.0, 100.0, 150.0])
  @pytest.mark.parametrize('vol', [0.10, 0.20, 0.60])
  def test_call_delta_is_in_the_unit_interval(self, spot, vol):
    greeks = bs_greeks(spot, 100.0, 0.25, 0.05, vol)
    assert 0.0 <= greeks.delta <= 1.0
    assert greeks.gamma >= 0.0

  @pytest.mark.parametrize('spot', [50.0, 100.0, 100.0, 150.0])
  @pytest.mark.parametrize('vol', [0.10, 0.20, 0.60])
  def test_put_delta_is_negative_and_bounded(self, spot, vol):
    greeks = bs_greeks(spot, 100.0, 0.25, 0.05, vol, 'put')
    assert -1.0 <= greeks.delta <= 0.0
    assert greeks.gamma >= 0.0

  def test_atm_call_delta_is_about_a_half(self):
    assert bs_greeks(100.0, 100.0, 1 / 12, 0.0, 0.2).delta == \
      pytest.approx(0.5, abs=0.03)

  def test_put_call_parity(self):
    call = bs_price(100.0, 95.0, 0.5, 0.06, 0.25)
    put = bs_price(100.0, 95.0, 0.5, 0.06, 0.25, 'put')
    assert call - put == pytest.approx(
      100.0 - 95.0 * 2.718281828459045 ** (-0.06 * 0.5))

  def test_gamma_peaks_at_the_money(self):
    near = bs_greeks(100.0, 100.0, 0.25, 0.05, 0.2).gamma
    far = bs_greeks(100.0, 140.0, 0.25, 0.05, 0.2).gamma
    assert near > far

  def test_expiry_returns_intrinsic_delta_and_no_gamma(self):
    assert bs_greeks(120.0, 100.0, 0.0, 0.05, 0.2) == bs_greeks(
      120.0, 100.0, -1.0, 0.05, 0.2)
    assert bs_greeks(120.0, 100.0, 0.0, 0.05, 0.2).gamma == 0.0
    assert bs_greeks(80.0, 100.0, 0.0, 0.05, 0.2, 'put').delta == -1.0
    assert bs_greeks(120.0, 100.0, 0.0, 0.05, 0.2, 'put').delta == 0.0

  def test_expired_price_is_intrinsic(self):
    assert bs_price(120.0, 100.0, 0.0, 0.05, 0.2) == 20.0
    assert bs_price(80.0, 100.0, 0.0, 0.05, 0.2, 'put') == 20.0

  def test_more_time_is_worth_more_to_a_call(self):
    assert bs_price(100.0, 100.0, 1.0, 0.05, 0.2) > \
      bs_price(100.0, 100.0, 0.1, 0.05, 0.2)

  def test_deep_in_the_money_call_is_a_shot(self):
    assert bs_price(1000.0, 100.0, 0.5, 0.05, 0.2) > 890.0

  @pytest.mark.parametrize('args', [
    (0.0, 100.0, 0.5, 0.05, 0.2),
    (100.0, 0.0, 0.5, 0.05, 0.2),
    (100.0, 100.0, 0.5, 0.05, 0.0),
    (100.0, 100.0, 0.5, 0.05, -0.2),
  ])
  def test_rejects_unusable_inputs(self, args):
    with pytest.raises(ValueError):
      bs_greeks(*args)
    with pytest.raises(ValueError):
      bs_price(*args)

  def test_rejects_unknown_option_kind(self):
    with pytest.raises(ValueError, match='kind'):
      bs_greeks(100.0, 100.0, 0.5, 0.05, 0.2, 'straddle')
    with pytest.raises(ValueError, match='kind'):
      bs_price(100.0, 100.0, 0.5, 0.05, 0.2, 'straddle')


class TestRiskMeasures:
  '''The terminal objective, and the alpha floor it must respect.'''

  def test_cvar_averages_the_worst_tail(self):
    values = [float(value) for value in range(20)]
    assert cvar(values, 1.0) == pytest.approx(0.0)
    assert cvar(values, 0.9) == pytest.approx(0.5)
    assert cvar([10.0] * 20, 0.95) == pytest.approx(10.0)

  def test_cvar_of_an_empty_sample_is_zero(self):
    assert cvar([], 0.95) == 0.0

  @pytest.mark.parametrize('alpha', [0.05, 0.2, 0.5, 0.8])
  def test_low_tail_confidence_is_refused(self, alpha):
    # At alpha <= 20 percent the deep hedging difference strategy earned
    # +1.37 unconditionally: it stopped hedging.
    with pytest.raises(ValueError, match='does not produce a hedge'):
      cvar([1.0, 2.0], alpha)

  def test_alpha_above_one_is_refused(self):
    with pytest.raises(ValueError, match='alpha'):
      cvar([1.0], 1.5)

  def test_min_tail_confidence_is_inclusive(self):
    # 15 percent of 21 observations is 3.15, so the tail is the worst 4.
    assert cvar([-1.0] * 20 + [-9.0], min_tail_confidence) == \
      pytest.approx(-3.0)

  def test_semi_rmse_ignores_gains(self):
    assert semi_rmse([1.0, 2.0, 3.0]) == 0.0
    assert semi_rmse([-4.0, 0.0]) == pytest.approx(4.0 / 2 ** 0.5)

  def test_semi_rmse_grows_with_the_loss_tail(self):
    assert semi_rmse([-2.0, -2.0]) > semi_rmse([-1.0, -1.0])


class TestHedgeConstruction:
  '''Instrument definitions and environment validation.'''

  def test_satisfies_both_protocols(self):
    assert isinstance(hedge_env(), Env)
    assert isinstance(hedge_env(), ContinuousEnv)

  def test_default_set_is_continuous_and_three_wide(self):
    instruments = default_instruments()
    assert [item.name for item in instruments] == [
      'nifty_future', 'put_spread_95', 'collar_upside']
    assert all(item.coverage_cap > 0.0 for item in instruments)

  def test_action_symbols_are_the_instrument_names(self):
    env = hedge_env()
    assert env.action_symbols == (
      'nifty_future', 'put_spread_95', 'collar_upside')

  def test_n_steps_reserves_history(self):
    bars = smooth_bars(count=60)
    assert HedgeEnv(bars, history=40).n_steps == 20

  def test_rejects_short_history(self):
    with pytest.raises(ValueError, match='history'):
      HedgeEnv(smooth_bars(count=10), history=40)

  def test_rejects_bad_capital(self):
    with pytest.raises(ValueError, match='capital'):
      HedgeEnv(smooth_bars(), capital=0.0, history=40)

  def test_rejects_bad_iv_window(self):
    with pytest.raises(ValueError, match='iv_window'):
      HedgeEnv(smooth_bars(), iv_window=1, history=40)

  def test_rejects_bad_base_vol(self):
    with pytest.raises(ValueError, match='base_vol'):
      HedgeEnv(smooth_bars(), base_vol=0.0, history=40)

  def test_rejects_low_tail_confidence(self):
    with pytest.raises(ValueError, match='does not produce a hedge'):
      HedgeEnv(smooth_bars(), risk=HedgeRisk(alpha=0.5), history=40)

  def test_rejects_unknown_risk_measure(self):
    with pytest.raises(ValueError, match='measure'):
      HedgeEnv(smooth_bars(), risk=HedgeRisk(measure='drawdown'), history=40)

  def test_rejects_non_positive_thresholds(self):
    with pytest.raises(ValueError, match='iv_ceiling'):
      HedgeEnv(smooth_bars(), iv_ceiling=0.0, history=40)

  def test_rejects_unknown_instrument_kind(self):
    with pytest.raises(ValueError, match='kind'):
      Instrument('swap', 'swap')

  def test_rejects_option_without_a_tenor(self):
    with pytest.raises(ValueError, match='tenor_bars'):
      Instrument('p', 'option', option_kind='put', strike_ratio=0.95,
                 tenor_bars=0)

  def test_rejects_option_without_a_strike(self):
    with pytest.raises(ValueError, match='strike_ratio'):
      Instrument('p', 'option', option_kind='put', tenor_bars=21)

  def test_rejects_bad_option_kind(self):
    with pytest.raises(ValueError, match='option_kind'):
      Instrument('p', 'option', option_kind='swap', tenor_bars=21)

  @pytest.mark.parametrize('cap', [0.0, 1.5])
  def test_rejects_bad_coverage_cap(self, cap):
    with pytest.raises(ValueError, match='coverage_cap'):
      Instrument('f', 'future', coverage_cap=cap)


class TestHedgeMask:
  '''The hard business rule, expressed as an admissible action set.'''

  def test_low_vol_leaves_options_open(self):
    env = hedge_env()
    env.reset()
    assert env.coverage_bounds()['put_spread_95'] == pytest.approx(1.0)

  def test_high_vol_blocks_new_option_purchases(self):
    env = HedgeEnv(journey_bars(), history=40, iv_window=20)
    env.reset()
    assert env.observation()['iv'] > 0.30
    assert env.coverage_bounds()['put_spread_95'] == pytest.approx(0.0)
    env.step({'put_spread_95': 1.0, 'collar_upside': 1.0,
              'nifty_future': 0.5})
    assert env.positions['put_spread_95'] == 0
    assert env.positions['collar_upside'] == 0
    assert env.positions['nifty_future'] > 0

  def test_mask_survives_the_action_dict(self):
    env = HedgeEnv(journey_bars(), history=40, iv_window=20)
    env.reset()
    for _ in range(5):
      env.step({'put_spread_95': 1.0, 'collar_upside': 1.0})
    assert env.positions['put_spread_95'] == 0
    assert env.positions['collar_upside'] == 0

  def test_observation_carries_the_mask_for_the_critic(self):
    env = HedgeEnv(journey_bars(), history=40, iv_window=20)
    observation = env.reset()
    assert observation['bounds']['put_spread_95'] == pytest.approx(0.0)
    assert observation['bounds']['nifty_future'] == pytest.approx(1.0)

  def test_mask_reads_the_vol_supplied_by_the_observation(self):
    # A critic computes its target from an observation, so the mask must
    # accept one rather than only the live decision bar.
    env = hedge_env()
    assert env.coverage_bounds({'iv': 5.0})['put_spread_95'] == \
      pytest.approx(0.0)

  def test_rich_vol_thins_options_before_the_hard_rule(self):
    env = hedge_env(iv_ceiling=10.0, iv_scale=0.20)
    env.reset()
    vol = env.observation()['iv']
    assert env.coverage_bounds()['put_spread_95'] == \
      pytest.approx(min(1.0, 0.20 / vol))


class TestHedgeConstraints:
  '''Coverage stays inside the instrument caps.'''

  def test_coverage_never_exceeds_the_cap(self):
    env = hedge_env()
    env.reset()
    for _ in range(env.n_steps - 1):
      env.step({'nifty_future': 5.0, 'put_spread_95': 5.0,
                'collar_upside': 5.0})
      for value in env.coverage.values():
        assert 0.0 <= value <= 1.0 + 1e-9

  def test_negative_coverage_is_clamped_to_flat(self):
    env = hedge_env()
    env.reset()
    env.step({'nifty_future': -3.0})
    assert env.coverage['nifty_future'] == pytest.approx(0.0)

  def test_junk_coverage_is_treated_as_flat(self):
    env = hedge_env()
    env.reset()
    env.step({'nifty_future': 'all in', 'put_spread_95': None})
    assert env.coverage['nifty_future'] == pytest.approx(0.0)
    assert env.coverage['put_spread_95'] == pytest.approx(0.0)

  def test_missing_instrument_is_treated_as_flat(self):
    env = hedge_env()
    env.reset()
    env.step({'nifty_future': 0.5})
    assert env.coverage['collar_upside'] == pytest.approx(0.0)

  def test_full_futures_coverage_neutralises_the_liability(self):
    env = hedge_env()
    observation = env.reset()
    assert observation['liability_delta'] == pytest.approx(
      env.exposure_units)
    env.step({'nifty_future': 1.0})
    # Whole contracts, so the residual is at most one index unit.
    assert env.info()['net_delta'] == pytest.approx(0.0, abs=1.0)


class TestHedgeNoLookAhead:
  '''A decision at the close fills at the open, at decision-bar vol.'''

  def test_changing_future_prices_does_not_change_past_rewards(self):
    base = smooth_bars(count=80)
    spiked = [
      *base[:42],
      *[Bar(bar.timestamp + timedelta(days=900), bar.open * 2,
            bar.high * 2, bar.low * 2, bar.close * 2, 1.0)
        for bar in base[42:]],
    ]
    first = HedgeEnv(base, history=40, iv_window=20)
    second = HedgeEnv(spiked, history=40, iv_window=20)
    action = {'nifty_future': 0.5, 'put_spread_95': 0.2}
    assert first.reset()['step'] == 0
    assert second.reset()['step'] == 0
    assert first.step(action)[1] == pytest.approx(second.step(action)[1])
    assert first.positions == second.positions

  def test_fill_bar_close_cannot_change_the_executed_contracts(self):
    plain = smooth_bars(count=80)
    rewritten = smooth_bars(count=80)
    fill = 42
    bar = rewritten[fill]
    rewritten[fill] = Bar(bar.timestamp, bar.open, 500.0, 1.0, 500.0, 1.0)
    action = {'nifty_future': 0.5, 'put_spread_95': 0.3}
    first = HedgeEnv(plain, history=40, iv_window=20)
    second = HedgeEnv(rewritten, history=40, iv_window=20)
    first.reset()
    second.reset()
    for _ in range(3):
      first.step(action)
      second.step(action)
    assert first.positions == second.positions
    assert first.decisions[2]['cost'] == pytest.approx(
      second.decisions[2]['cost'])

  def test_fill_bar_open_is_the_fill_price(self):
    quiet = HedgeEnv(smooth_bars(count=80), history=40, iv_window=20)
    moved_bars = smooth_bars(count=80)
    moved_bars[40] = Bar(moved_bars[40].timestamp, 1.0, 1.0, 1.0, 1.0, 1.0)
    moved = HedgeEnv(moved_bars, history=40, iv_window=20)
    action = {'nifty_future': 0.5}
    quiet.reset()
    moved.reset()
    quiet.step(action)
    moved.step(action)
    assert quiet.total_cost > moved.total_cost


class TestHedgeReward:
  '''Cost per step, coherent risk once at the end.'''

  def test_per_step_reward_is_the_cost_charge(self):
    env = hedge_env()
    env.reset()
    _, reward, done, _ = env.step({'nifty_future': 0.5})
    assert reward == pytest.approx(
      -env.risk.cost * env.decisions[-1]['cost'])
    assert not done

  def test_a_flat_hedge_pays_no_cost(self):
    env = hedge_env()
    env.reset()
    assert env.step({})[1] == pytest.approx(0.0)

  def test_terminal_step_carries_the_risk_penalty(self):
    env = hedge_env()
    env.reset()
    rewards = []
    for _ in range(env.n_steps):
      _, reward, _, _ = env.step({'nifty_future': 0.5})
      rewards.append(reward)
    assert all(value <= 0.0 for value in rewards)
    # The terminal step is the only one charged the risk measure, and it
    # is charged on the whole error path rather than on one bar.
    measure = env.risk_measure()
    assert measure > 0.0
    assert rewards[-1] == pytest.approx(
      -env.risk.cost * env.decisions[-1]['cost'] - env.risk.risk * measure)
    assert rewards[-1] < rewards[-2]

  def test_semi_rmse_measure_is_accepted(self):
    env = HedgeEnv(smooth_bars(), history=40, iv_window=20,
                   risk=HedgeRisk(measure='semi_rmse'))
    env.reset()
    for _ in range(env.n_steps):
      env.step({'nifty_future': 0.5})
    assert env.risk_measure() == pytest.approx(
      semi_rmse(env.hedging_error))

  def test_risk_report_reports_separate_statistics(self):
    env = hedge_env()
    env.reset()
    for _ in range(20):
      env.step({'nifty_future': 0.5})
    report = env.risk_report()
    assert report['error_stdev'] >= 0.0
    assert report['cvar'] <= report['mean_error'] + 1e-9 or \
      report['alpha'] == 0.95
    assert report['max_abs_error'] >= 0.0
    assert len(env.hedging_error) == 20

  def test_error_path_is_scale_free(self):
    # Scale invariance now holds to within whole-share rounding rather
    # than exactly, because the liability is quantised into shares: a
    # 90x larger book cannot hold a proportionally fractional position.
    # This test previously passed only because every path started at the
    # +1.0 offset, which made both values identical by construction.
    small = HedgeEnv(smooth_bars(), capital=1_000_000.0, history=40,
                     iv_window=20)
    large = HedgeEnv(smooth_bars(), capital=90_000_000.0, history=40,
                     iv_window=20)
    for env in (small, large):
      env.reset()
      for _ in range(20):
        env.step({'nifty_future': 0.5})
    assert small.hedging_error[-1] == pytest.approx(
      large.hedging_error[-1], rel=1e-3)

  def test_error_path_starts_at_zero(self):
    # Regression guard for the +1.0 offset: the self-financing error
    # must begin at zero for every policy, or the measure is really
    # liability level rather than hedging error.
    for coverage in (0.0, 0.5, 1.0):
      env = HedgeEnv(smooth_bars(), capital=1_000_000.0, history=40,
                     iv_window=20)
      env.reset()
      env.step({'nifty_future': coverage})
      assert abs(env.hedging_error[0]) < 1e-9, (
          f'coverage {coverage} started at {env.hedging_error[0]}')

  def test_a_better_hedge_has_smaller_error_dispersion(self):
    hedged = hedge_env()
    bare = hedge_env()
    hedged.reset()
    bare.reset()
    for _ in range(30):
      hedged.step({'nifty_future': 1.0})
      bare.step({})
    assert hedged.risk_report()['max_abs_error'] <= \
      bare.risk_report()['max_abs_error']

  def test_churning_options_costs_more(self):
    calm = hedge_env()
    churn = hedge_env()
    calm.reset()
    churn.reset()
    for _ in range(10):
      calm.step({'put_spread_95': 0.2})
      churn.step({'put_spread_95': 0.0 if len(churn.decisions) % 2 else 0.2})
    assert churn.total_cost > calm.total_cost


class TestHedgeLifecycle:
  '''Reset, roll and reporting.'''

  def test_step_count_advances_and_terminates(self):
    env = hedge_env()
    env.reset()
    for index in range(env.n_steps):
      done = env.step({'nifty_future': 0.3})[2]
      assert done is (index == env.n_steps - 1)

  def test_step_after_termination_is_inert(self):
    env = hedge_env()
    env.reset()
    for _ in range(env.n_steps):
      env.step({'nifty_future': 0.3})
    equity = env.equity_curve[-1]
    _, reward, done, _ = env.step({'nifty_future': 0.3})
    assert done
    assert reward == 0.0
    assert env.equity_curve[-1] == equity

  def test_reset_restores_positions_and_error(self):
    env = hedge_env()
    env.reset()
    for _ in range(10):
      env.step({'nifty_future': 0.6})
    env.reset()
    assert env.positions['nifty_future'] == 0
    assert not env.hedging_error
    assert env.equity_curve == [1.0]
    assert env.total_cost == 0.0
    assert not env.decisions
    assert env.average_delta() == 0.0

  def test_reset_ignores_the_seed(self):
    env = hedge_env()
    first = env.reset(seed=1)['spot']
    second = env.reset(seed=2)['spot']
    assert first == pytest.approx(second)

  def test_option_roll_resets_the_strike(self):
    env = hedge_env()
    env.reset()
    before = env.strikes['put_spread_95']
    for _ in range(env.n_steps):
      env.step({'nifty_future': 0.3})
    after = env.strikes['put_spread_95']
    assert after != pytest.approx(before)
    assert after / before > 0.9

  def test_state_carries_the_greeks(self):
    env = hedge_env()
    observation = env.reset()
    for key in ('spot', 'iv', 'rate', 'net_delta', 'gamma', 'coverage',
                'contracts', 'bounds', 'error', 'equity', 'total_cost',
                'tau_years', 'hedge_delta'):
      assert key in observation
    env.step({'nifty_future': 0.5, 'put_spread_95': 0.4})
    assert env.observation()['gamma'] != 0.0
    assert env.decisions[-1]['cost'] > 0.0

  def test_render_and_info_are_available(self):
    env = hedge_env()
    env.reset()
    env.step({'nifty_future': 0.4})
    text = env.render()
    assert 'step' in text and 'error' in text
    assert env.info()['error'] == env.decisions[-1]['error']
    assert env.observation()['step'] == 1

  def test_average_delta_shrinks_with_hedging(self):
    hedged = hedge_env()
    bare = hedge_env()
    hedged.reset()
    bare.reset()
    for _ in range(10):
      hedged.step({'nifty_future': 1.0})
      bare.step({})
    assert abs(hedged.average_delta()) < abs(bare.average_delta())

  def test_equity_curve_tracks_the_hedged_book(self):
    env = hedge_env()
    env.reset()
    for _ in range(5):
      env.step({'nifty_future': 0.5})
    assert len(env.equity_curve) == 6
    assert all(value > 0.0 for value in env.equity_curve)


class TestTrials:
  '''Trial bookkeeping.'''

  def test_from_returns_computes_both_sharpes(self):
    trial = Trial.from_returns('t', 1, synthetic_returns(50))
    assert trial.sharpe > 0.0
    assert trial.per_period_sharpe == pytest.approx(trial.sharpe / 252 ** 0.5)
    assert trial.name == 't'
    assert trial.seed == 1

  def test_single_observation_is_flat(self):
    trial = Trial.from_returns('t', 1, [0.01])
    assert trial.sharpe == 0.0
    assert trial.per_period_sharpe == 0.0

  def test_log_statistics(self):
    log = TrialLog([
      Trial('a', 1, [], sharpe=0.4, per_period_sharpe=0.02),
      Trial('a', 2, [], sharpe=0.6, per_period_sharpe=0.03),
      Trial('b', 1, [], sharpe=0.8, per_period_sharpe=0.04),
    ])
    assert len(log) == 3
    assert log.count == 2
    assert log.names == ('a', 'b')
    assert log.mean_sharpe == pytest.approx(0.6)
    assert log.stdev_sharpe > 0.0
    assert log.seed_stdev('a') == pytest.approx(0.1414213562, abs=1e-9)
    assert log.seed_sharpes('b') == [0.8]
    assert log.best_trial().sharpe == 0.8
    assert log.selection_inflation() == pytest.approx(0.8 / 0.6)
    assert len(list(log)) == 3

  def test_selection_inflation_is_one_on_a_flat_log(self):
    assert TrialLog().selection_inflation() == 1.0
    one = TrialLog([Trial('a', 1, [], sharpe=-0.2)])
    assert one.selection_inflation() == 1.0

  def test_seed_stdev_of_a_single_seed_is_zero(self):
    assert TrialLog([Trial('a', 1, [], sharpe=0.5)]).seed_stdev() == 0.0

  def test_empty_log_has_no_best_trial(self):
    with pytest.raises(ValueError, match='empty'):
      TrialLog().best_trial()


class TestSharpeReport:
  '''The length gate is the point of this module.'''

  def _log(self, count=8, names=4, length=400, annual=0.010):
    '''Build a log whose configurations genuinely differ in quality.

    Args:
      count: Number of runs.
      names: Number of distinct configurations behind those runs.
      length: Returns per run.
      annual: Approximate annualised Sharpe of the weakest configuration.

    Returns:
      A populated ``TrialLog``.
    '''
    spread = 0.01
    trials = []
    for index in range(count):
      factor = 1.0 + 0.15 * (index % names) + 0.01 * index
      drift = annual * spread * factor / (252 ** 0.5)
      returns = synthetic_returns(length, spread=spread, drift=drift)
      trials.append(Trial.from_returns(f'trial_{index % names:02d}', index,
                                       returns))
    return TrialLog(trials)

  def test_refuses_when_the_gate_fails(self):
    log = self._log(names=8, length=200, annual=0.5)
    report = sharpe_report(log)
    assert report.sharpe is None
    assert report.deflated is None
    assert 'need' in report.reason
    assert report.required_years > report.available_years

  def test_refusal_explains_the_shortfall_in_years(self):
    report = sharpe_report(self._log(names=8, length=200, annual=0.5))
    assert f'{report.required_years:.2f}' in report.reason
    assert f'{report.available_years:.2f}' in report.reason

  def test_reports_when_the_gate_passes(self):
    report = sharpe_report(self._log(names=4, length=1200, annual=1.5))
    assert report.sharpe is not None
    assert report.sharpe > 0.0
    assert report.deflated is not None
    assert 0.0 <= report.deflated <= 1.0
    assert report.required_years <= report.available_years

  def test_reported_sharpe_is_the_seed_mean_not_the_best(self):
    report = sharpe_report(self._log(names=4, length=1200, annual=1.5))
    assert report.sharpe < report.best_sharpe
    assert report.seed_stdev > 0.0
    assert report.selection_inflation > 1.0

  def test_reports_dispersion_alongside_the_mean(self):
    report = sharpe_report(self._log(names=4, length=1200, annual=1.5))
    assert report.mean_sharpe > 0.0
    assert report.stdev_sharpe > 0.0

  def test_refuses_a_single_configuration(self):
    log = TrialLog([Trial('only', 1, synthetic_returns(1200))])
    report = sharpe_report(log)
    assert report.sharpe is None
    assert 'two configurations' in report.reason

  def test_refuses_a_non_positive_sharpe(self):
    log = TrialLog([
      Trial('a', 1, synthetic_returns(1200, drift=-0.05)),
      Trial('b', 1, synthetic_returns(1200, drift=-0.05)),
    ])
    report = sharpe_report(log)
    assert report.sharpe is None
    assert 'not positive' in report.reason

  def test_refuses_an_empty_log(self):
    with pytest.raises(ValueError, match='empty'):
      sharpe_report(TrialLog())

  def test_available_years_matches_the_sample(self):
    report = sharpe_report(self._log(names=8, length=252))
    assert report.available_years == pytest.approx(1.0)


class TestRandomSearch:
  '''The driver that produces the trial log.'''

  def test_records_every_configuration_and_seed(self):
    log = random_search(make_panels(count=120), trials=3, seeds=(1, 2),
                        history=HISTORY)
    assert len(log) == 6
    assert log.count == 3

  def test_is_reproducible_for_a_seed(self):
    panels = make_panels(count=120)
    first = random_search(panels, trials=2, seeds=(1,), history=HISTORY,
                          seed=11)
    second = random_search(panels, trials=2, seeds=(1,), history=HISTORY,
                           seed=11)
    assert first.sharpes == second.sharpes

  def test_seeds_produce_different_books(self):
    log = random_search(make_panels(count=120), trials=1, seeds=(1, 2, 3),
                        history=HISTORY)
    assert len(set(log.sharpes)) > 1
    assert log.seed_stdev() > 0.0

  def test_rollout_runs_to_termination(self):
    env = WeightAllocationEnv(make_panels(count=60), history=HISTORY)
    returns = rollout(env, ConstantWeights(weight=0.2))
    assert len(returns) == env.n_steps + 1

  def test_replay_env_protocol_is_structural(self):
    assert isinstance(WeightAllocationEnv(make_panels(count=60),
                                          history=HISTORY), ReplayEnv)
    assert isinstance(HedgeEnv(smooth_bars(), history=40), ReplayEnv)

  def test_rejects_bad_arguments(self):
    panels = make_panels(count=60)
    with pytest.raises(ValueError, match='trials'):
      random_search(panels, trials=0, history=HISTORY)
    with pytest.raises(ValueError, match='seeds'):
      random_search(panels, trials=1, seeds=(), history=HISTORY)


class TestCompareParity:
  '''Two engines, one policy, identical inputs.'''

  def test_returns_both_engines_and_the_fingerprint(self):
    result = compare(ConstantWeights(weight=0.3), make_panels(count=160),
                     history=HISTORY, max_weight=0.5)
    assert isinstance(result, EngineComparison)
    for key in ('sharpe', 'max_drawdown', 'total_return', 'turnover',
                'total_cost'):
      assert key in result.env
      assert key in result.portfolio
    assert result.fingerprint == policy_fingerprint(ConstantWeights(0.3))

  def test_engines_agree_on_cost_and_total_return(self):
    result = compare(MomentumPolicy(lookback=40, skip=5, top=2,
                                    max_weight=0.5),
                     make_panels(count=160), history=HISTORY,
                     max_weight=0.5)
    assert result.env['total_return'] == pytest.approx(
      result.portfolio['total_return'])
    assert result.env['total_cost'] == pytest.approx(
      result.portfolio['total_cost'])

  def test_equal_weight_control_arm_is_positive_on_rising_panels(self):
    assert equal_weight_sharpe(make_panels(count=200)) > 0.0
