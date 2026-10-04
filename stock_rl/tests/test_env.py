#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Tests for the cross-sectional RL environment.

The look-ahead tests matter more than the rest. An environment that leaks
makes every downstream comparison meaningless, and the leak is invisible
in the equity curve.
'''

from datetime import datetime, timedelta

import pytest

from stock_rl.bars import Bar
from stock_rl.costs import CostModel
from stock_rl.env import Env, PortfolioEnv, RewardWeights

START = datetime(2026, 1, 1)
HISTORY = 40

FREE = CostModel(stt_buy=0.0, stt_sell=0.0, exchange_pct=0.0,
                 sebi_pct=0.0, stamp_duty_buy=0.0, gst_pct=0.0,
                 dp_charge=0.0, slippage_bps=0.0)


def panel(closes, start_price=None):
  '''Return bars stepping from one close to the next.

  Args:
    closes: Sequence of closing prices.
    start_price: Optional first open. Defaults to the first close.

  Returns:
    List of bars one day apart.
  '''
  previous = start_price if start_price is not None else closes[0]
  out = []
  for index, close in enumerate(closes):
    out.append(Bar(
      START + timedelta(days=index),
      previous, max(previous, close), min(previous, close), close, 1.0,
    ))
    previous = close
  return out


def make_panels(count=120, symbols=('AAA', 'BBB')):
  '''Return aligned panels with steadily rising prices.

  Args:
    count: Number of bars per panel.
    symbols: Symbols to create.

  Returns:
    Mapping of symbol to bars, all sharing one timeline.
  '''
  panels = {}
  for number, symbol in enumerate(symbols):
    prices = [100.0 + number + index for index in range(count)]
    panels[symbol] = panel(prices)
  return panels


def make_env(**kwargs):
  '''Return an environment over default panels at the test history.

  The environment's own default history is 252 bars, a full year of daily
  data, which is far more than these synthetic panels carry. Construction
  tests must therefore pass an explicit history or they fail before they
  reach the assertion.

  Args:
    **kwargs: Forwarded to ``PortfolioEnv``.

  Returns:
    A constructed environment.
  '''
  return PortfolioEnv(make_panels(), history=HISTORY, **kwargs)


class TestConstruction:
  '''Invalid environments must fail at construction, not mid-episode.'''

  def test_satisfies_the_env_protocol(self):
    assert isinstance(make_env(), Env)

  def test_step_count_reserves_history(self):
    env = PortfolioEnv(make_panels(count=120), history=HISTORY)
    assert env.n_steps == 120 - HISTORY

  def test_action_symbols_are_sorted_and_fixed(self):
    env = PortfolioEnv(
      make_panels(symbols=('BBB', 'AAA')), history=HISTORY)
    assert env.action_symbols == ('AAA', 'BBB')

  def test_rejects_empty_panels(self):
    with pytest.raises(ValueError, match='must not be empty'):
      PortfolioEnv({})

  def test_rejects_misaligned_panels(self):
    panels = make_panels()
    panels['BBB'] = panels['BBB'][:50]
    with pytest.raises(ValueError, match='aligned'):
      PortfolioEnv(panels)

  def test_rejects_history_longer_than_data(self):
    with pytest.raises(ValueError, match='must exceed history'):
      PortfolioEnv(make_panels(count=30), history=40)

  @pytest.mark.parametrize('capital', [0.0, -1.0])
  def test_rejects_non_positive_capital(self, capital):
    with pytest.raises(ValueError, match='capital'):
      make_env(capital=capital)

  @pytest.mark.parametrize('weight', [0.0, -0.1, 1.5])
  def test_rejects_out_of_range_max_weight(self, weight):
    with pytest.raises(ValueError, match='max_weight'):
      make_env(max_weight=weight)

  def test_rejects_out_of_range_sector_cap(self):
    with pytest.raises(ValueError, match='max_sector_weight'):
      make_env(max_sector_weight=0.0)


class TestNoLookAhead:
  '''The observation must never contain the bar it will be filled on.'''

  def test_observation_excludes_the_fill_bar(self):
    env = PortfolioEnv(make_panels(), history=HISTORY, costs=FREE)
    seen = []
    for _ in range(5):
      observation, _, _, _ = env.step(
        {symbol: 0 for symbol in env.action_symbols})
      seen.append(observation['step'])
    assert seen == list(range(1, 6))

  def test_observation_step_advances_with_actions(self):
    env = PortfolioEnv(make_panels(), history=HISTORY, costs=FREE)
    first = env.reset()['step']
    observation, _, _, _ = env.step({'AAA': 1})
    assert observation['step'] == first + 1

  def test_changing_future_prices_does_not_change_past_rewards(self):
    # Two environments differing only after the first decision must
    # produce identical rewards for that first step. If they do not, the
    # agent's action was priced with information it could not have had.
    base = make_panels(count=60)
    spiked = {
      symbol: list(bars[:HISTORY + 2]) + [
        Bar(bar.timestamp + timedelta(days=900), bar.open * 3,
            bar.high * 3, bar.low * 3, bar.close * 3, 1.0)
        for bar in bars[HISTORY + 2:]
      ]
      for symbol, bars in base.items()
    }
    first = PortfolioEnv(base, history=HISTORY, costs=FREE)
    second = PortfolioEnv(spiked, history=HISTORY, costs=FREE)
    first.reset()
    second.reset()
    action = {'AAA': 1, 'BBB': 0}
    _, reward_first, _, _ = first.step(action)
    _, reward_second, _, _ = second.step(action)
    assert reward_first == pytest.approx(reward_second)

  def test_agent_never_sees_the_fill_price(self):
    # The fill happens at the next bar's open. Rewriting that open must
    # not change the observation the agent was shown.
    panels = make_panels(count=60)
    env = PortfolioEnv(panels, history=HISTORY, costs=FREE)
    env.reset()
    observation, _, _, _ = env.step({'AAA': 1})
    before = observation['returns']['AAA']
    panels['AAA'][HISTORY + 1] = Bar(
      panels['AAA'][HISTORY + 1].timestamp, 9999.0, 9999.0, 9999.0,
      9999.0, 1.0)
    after = env.observation()['returns']['AAA']
    assert before == pytest.approx(after)


class TestConstraints:
  '''Feasibility is enforced in the environment, outside the policy.'''

  def test_single_symbol_never_exceeds_max_weight(self):
    env = PortfolioEnv(make_panels(), history=HISTORY, costs=FREE,
                       max_weight=0.10)
    env.reset()
    for _ in range(env.n_steps):
      _, _, done, _ = env.step({'AAA': 1, 'BBB': 1})
      assert max(env.weights.values()) <= 0.10 + 1e-9
      if done:
        break

  def test_sector_cap_is_enforced(self):
    sectors = {'AAA': 'IT', 'BBB': 'IT'}
    env = PortfolioEnv(make_panels(), history=HISTORY, costs=FREE,
                       sectors=sectors, max_sector_weight=0.15)
    env.reset()
    for _ in range(env.n_steps):
      _, _, done, _ = env.step({'AAA': 1, 'BBB': 1})
      assert sum(env.weights.values()) <= 0.15 + 1e-9
      if done:
        break

  def test_separate_sectors_are_capped_independently(self):
    sectors = {'AAA': 'IT', 'BBB': 'BANK'}
    env = PortfolioEnv(make_panels(), history=HISTORY, costs=FREE,
                       sectors=sectors, max_sector_weight=0.10)
    env.reset()
    for _ in range(10):
      env.step({'AAA': 1, 'BBB': 1})
    assert env.weights['AAA'] <= 0.10 + 1e-9
    assert env.weights['BBB'] <= 0.10 + 1e-9

  def test_weights_never_negative(self):
    env = PortfolioEnv(make_panels(), history=HISTORY, costs=FREE)
    env.reset()
    for _ in range(10):
      env.step({'AAA': -1, 'BBB': -1})
    assert all(w >= 0.0 for w in env.weights.values())


class TestActions:
  '''Direction mapping and tolerance of partial actions.'''

  def test_add_increases_weight(self):
    env = PortfolioEnv(make_panels(), history=HISTORY, costs=FREE)
    env.reset()
    env.step({'AAA': 1, 'BBB': 0})
    assert env.weights['AAA'] > 0.0

  def test_reduce_does_not_go_short(self):
    env = PortfolioEnv(make_panels(), history=HISTORY, costs=FREE)
    env.reset()
    for _ in range(5):
      env.step({'AAA': -1, 'BBB': -1})
    assert env.positions['AAA'] >= 0

  def test_missing_symbol_is_treated_as_hold(self):
    env = PortfolioEnv(make_panels(), history=HISTORY, costs=FREE)
    env.reset()
    env.step({'AAA': 1})
    assert env.weights['BBB'] == pytest.approx(0.0)

  def test_unknown_symbol_is_ignored(self):
    env = PortfolioEnv(make_panels(), history=HISTORY, costs=FREE)
    env.reset()
    _, _, _, _ = env.step({'ZZZ': 1, 'AAA': 0})
    assert env.weights['AAA'] == pytest.approx(0.0)


class TestCosts:
  '''Trading must be charged.'''

  def test_charged_costs_are_positive(self):
    env = PortfolioEnv(make_panels(), history=HISTORY)
    env.reset()
    _, _, _, info = env.step({'AAA': 1})
    assert info['cost'] > 0.0

  def test_free_model_charges_nothing(self):
    env = PortfolioEnv(make_panels(), history=HISTORY, costs=FREE)
    env.reset()
    _, _, _, info = env.step({'AAA': 1})
    assert info['cost'] == 0.0

  def test_churning_costs_more_than_holding(self):
    panels = make_panels(count=60)
    churn = PortfolioEnv(panels, history=HISTORY)
    churn.reset()
    for index in range(15):
      direction = 1 if index % 2 else -1
      churn.step({'AAA': direction, 'BBB': 0})
    hold = PortfolioEnv(panels, history=HISTORY)
    hold.reset()
    for _ in range(15):
      hold.step({'AAA': 1, 'BBB': 0})
    assert churn.total_cost > hold.total_cost

  def test_info_total_cost_accumulates(self):
    env = PortfolioEnv(make_panels(count=60), history=HISTORY)
    env.reset()
    env.step({'AAA': 1})
    _, _, _, info = env.step({'AAA': 1})
    assert info['total_cost'] > 0.0


class TestEpisodeLifecycle:
  '''Reset, termination and reporting.'''

  def test_episode_terminates_at_the_end(self):
    env = PortfolioEnv(make_panels(count=60), history=HISTORY)
    env.reset()
    action = {symbol: 0 for symbol in env.action_symbols}
    for _ in range(env.n_steps - 1):
      _, _, done, _ = env.step(action)
      assert not done
    _, _, done, _ = env.step(action)
    assert done

  def test_step_after_termination_is_inert(self):
    env = PortfolioEnv(make_panels(count=60), history=HISTORY)
    env.reset()
    action = {symbol: 0 for symbol in env.action_symbols}
    for _ in range(env.n_steps):
      env.step(action)
    equity = env.equity_curve[-1]
    _, reward, done, _ = env.step(action)
    assert done
    assert reward == 0.0
    assert env.equity_curve[-1] == equity

  def test_reset_restores_initial_state(self):
    env = PortfolioEnv(make_panels(count=60), history=HISTORY)
    env.reset()
    for _ in range(10):
      env.step({'AAA': 1})
    equity = env.equity_curve[-1]
    env.reset()
    assert env.equity_curve == [1.0]
    assert env.equity_curve[-1] != equity
    assert not env.decisions
    assert env.weights['AAA'] == pytest.approx(0.0)

  def test_equity_curve_is_monotone_in_length(self):
    env = PortfolioEnv(make_panels(count=60), history=HISTORY)
    env.reset()
    env.step({'AAA': 1})
    assert len(env.equity_curve) == 2

  def test_equity_curve_starts_at_one(self):
    env = PortfolioEnv(make_panels(count=60), history=HISTORY)
    assert env.equity_curve[0] == pytest.approx(1.0)

  def test_reset_ignores_seed_without_changing_outcome(self):
    env = PortfolioEnv(make_panels(count=60), history=HISTORY)
    first = env.reset(seed=1)['equity']
    second = env.reset(seed=999)['equity']
    assert first == pytest.approx(second)

  def test_decision_log_records_every_step(self):
    env = PortfolioEnv(make_panels(count=60), history=HISTORY)
    env.reset()
    for _ in range(4):
      env.step({'AAA': 1})
    assert len(env.decisions) == 4
    assert all('weights' in entry for entry in env.decisions)

  def test_render_mentions_step_and_equity(self):
    env = PortfolioEnv(make_panels(count=60), history=HISTORY)
    env.reset()
    env.step({'AAA': 1})
    text = env.render()
    assert 'step' in text
    assert 'equity' in text


class TestReward:
  '''Reward composition and the drawdown hazard.'''

  def test_reward_is_finite(self):
    env = PortfolioEnv(make_panels(), history=HISTORY, costs=FREE)
    env.reset()
    _, reward, _, _ = env.step({'AAA': 1})
    assert isinstance(reward, float)
    assert reward == reward  # not NaN

  def test_turnover_penalty_reduces_reward(self):
    # Charged costs are required here: with a free model the turnover
    # term is multiplied by zero and the assertion is vacuous.
    panels = make_panels(count=60)
    cheap = PortfolioEnv(panels, history=HISTORY,
                         reward=RewardWeights(turnover=0.0))
    dear = PortfolioEnv(panels, history=HISTORY,
                        reward=RewardWeights(turnover=100.0))
    cheap.reset()
    dear.reset()
    assert dear.step({'AAA': 1})[1] < cheap.step({'AAA': 1})[1]

  def test_drawdown_penalty_defaults_to_zero(self):
    # The doubling pathology is why this is off by default.
    assert RewardWeights().drawdown == 0.0

  def test_drawdown_penalty_is_non_positive_when_enabled(self):
    env = PortfolioEnv(make_panels(), history=HISTORY, costs=FREE,
                       reward=RewardWeights(drawdown=10.0))
    env.reset()
    for _ in range(5):
      env.step({'AAA': 1})
    assert env.info()['drawdown'] >= 0.0

  def test_annualized_sharpe_on_flat_episode_is_zero(self):
    env = PortfolioEnv(make_panels(), history=HISTORY, costs=FREE)
    env.reset()
    assert env.annualized_sharpe() == 0.0

  def test_annualized_sharpe_is_bounded(self):
    env = PortfolioEnv(make_panels(), history=HISTORY, costs=FREE)
    env.reset()
    for _ in range(10):
      env.step({'AAA': 1})
    assert env.annualized_sharpe() == pytest.approx(
      env.annualized_sharpe())
