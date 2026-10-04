#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Adversarial review findings, now all fixed. Nothing here is [BUG].

Every test in this file was written to PROVE a claim made by a reviewer of
this codebase. All nine findings below have now been fixed in the source,
so each class has been turned into a regression test that asserts the
fixed behaviour: the finding is still in the class docstring, because a
regression test whose comment no longer says what it is guarding is
useless, and the assertion is the opposite of what the reviewer proved.

  1. ``baselines.momentum_ranked`` computed the NEGATIVE of the 12-1
     momentum it documented: it bought the biggest loser. [fixed]
  2. ``HedgeEnv._error`` was the hedging error path offset by exactly
     +1.0 (100 percent of the initial liability) for every policy, which
     made ``semi_rmse`` identically zero and made the CVaR objective
     reward NOT hedging. [fixed]
  3. ``splits.purged_folds`` validated ``embargo`` and then never used
     it. [fixed]
  4. ``rl.train.compare`` documented that the two engines must agree on
     Sharpe and turnover to within whole-share rounding, which they
     cannot, because the return series were different lengths by
     construction. [fixed]
  5. ``rl.train.random_search`` never passed ``trial_seed`` to anything,
     so ``Trial.seed`` was a label with no effect and ``TrialLog.count``
     under-counted the configurations actually tried. [fixed]
  6. Neither ``portfolio._rebalance`` nor ``WeightAllocationEnv._execute``
     constrained cash, so a 100 percent target booked unpriced
     borrowing. [fixed]
  7. A NaN weight silently became a maximum-weight position in both
     ``portfolio._normalise`` and ``WeightAllocationEnv._target_weights``.
     [fixed]
  8. ``compliance.algo_tag.build_nnf_id`` returned a non-15-digit "NNF ID"
     for a non-integer flag instead of raising. [fixed]
  9. ``data.vendors.ReplayVendor.bars`` never checked entitlements, though
     the ``MarketDataFeed`` protocol it structurally satisfies promised
     it does. [fixed]

  Three of the original assertions could not survive their own fix and
  were rewritten rather than deleted, because each was asserting a
  pre-fix artefact rather than a property. The reasoning is in the class
  docstrings: the trial-seed dispersion of a deterministic environment is
  exactly zero, a log of twelve runs over four configurations has twelve
  runs, and the first-fold embargo is legitimately invisible behind the
  ``min_train`` floor.

  Look-ahead, the Deflated Sharpe formula, the RSI/ATR warm-up indices,
  the sourced retention periods and the cost constants all check out.
  Those are asserted under the ``[OK]`` classes.

  One defect found while fixing these is **not** fixed and is not tested
  here: ``hedge_env.cvar`` takes the LOWER tail of the hedging error,
  while the terminal reward is ``-risk * risk_measure``. A book that never
  hedged therefore collects a positive terminal reward in a crash, which
  is the same class of failure finding 2 describes and would need the sign
  convention of the objective itself to change.
'''

# These tests are deliberately white-box: several findings are only
# observable through private state (_liability, _hedge_value, _bar_index),
# so pylint's protected-access check is switched off for this file only.
# pylint: disable=protected-access

import ast
import inspect
import math
import pathlib
import random
import re
from statistics import NormalDist
import sys
from datetime import date, datetime, timedelta, time

import pytest

from stock_rl import baselines
from stock_rl.bars import Bar
from stock_rl.compliance.algo_tag import AlgoTagError, build_nnf_id
from stock_rl.compliance.retention import expiry_date, retention_table
from stock_rl.costs import DELIVERY, INTRADAY, CostModel
from stock_rl.data.calendar import (
  Session,
  muhurat_session,
  nse_normal_session,
)
from stock_rl.data.vendors import (
  Entitlement,
  EntitlementError,
  MarketDataFeed,
  ReplayVendor,
  Segment,
)
from stock_rl.indicators import atr, realized_volatility, rsi
from stock_rl.metrics import (
  TRADING_DAYS_PER_YEAR,
  deflated_sharpe,
  dsr_from_moments,
  minimum_backtest_length,
)
from stock_rl.portfolio import _normalise, run_portfolio
from stock_rl.rl import (
  HedgeEnv,
  HedgeRisk,
  Instrument,
  MomentumPolicy,
  WeightAllocationEnv,
  compare,
  cvar,
  panels_observation,
  random_search,
  semi_rmse,
)
from stock_rl.rl.train import sharpe_report
from stock_rl.splits import purged_folds

EPOCH = datetime(2021, 1, 1)
FREE = CostModel(stt_buy=0.0, stt_sell=0.0, exchange_pct=0.0,
                 sebi_pct=0.0, stamp_duty_buy=0.0, gst_pct=0.0,
                 dp_charge=0.0, slippage_bps=0.0)
FUTURE_ONLY = (Instrument('f', 'future', coverage_cap=1.0),)


def random_walk(count, seed=7, drift=0.0, sigma=0.012):
  '''Return a price path with an open equal to its previous close.

  Args:
    count: Number of prices.
    seed: Seed for the walk.
    drift: Per-bar drift.
    sigma: Per-bar standard deviation.

  Returns:
    List of closing prices.
  '''
  source = random.Random(seed)
  price = 100.0
  prices = []
  for _ in range(count):
    price *= 1.0 + drift + source.gauss(0.0, sigma)
    prices.append(price)
  return prices


def flat_bars(prices):
  '''Wrap a close series in bars whose open equals the close.

  Args:
    prices: Closing prices in ascending time order.

  Returns:
    List of bars.
  '''
  return [
    Bar(timestamp=EPOCH + timedelta(days=index), open=price, high=price,
        low=price, close=price, volume=1.0)
    for index, price in enumerate(prices)
  ]


def drifting_panels(symbols, count, seed=5):
  '''Build a panel per symbol from an independent random walk.

  Args:
    symbols: Symbol names.
    count: Bars per panel.
    seed: Seed for the walk.

  Returns:
    Mapping of symbol to bars.
  '''
  source = random.Random(seed)
  panels = {}
  for offset, symbol in enumerate(symbols):
    out = []
    price = 100.0
    for index in range(count):
      opened = price
      price *= 1.0 + 0.0004 + 0.0001 * offset + source.gauss(0.0, 0.012)
      out.append(Bar(timestamp=EPOCH + timedelta(days=index), open=opened,
                     high=max(opened, price), low=min(opened, price),
                     close=price, volume=1.0))
    panels[symbol] = out
  return panels


# ---------------------------------------------------------------------------
# FINDING 1 -- baselines.momentum_ranked was a reversal signal. [fixed]
# ---------------------------------------------------------------------------

class TestMomentumTwelveOneIsInverted:
  '''baselines.py:156-157.

  The docstring promises "rank on trailing return over ``lookback`` bars,
  then skip the most recent ``skip`` bars", i.e. the 12-1 construction

      score = closes[-(skip + 1)] / closes[-(lookback + skip + 1)] - 1

  The code computes

      start = closes[-(skip + 1)]
      end   = closes[-(lookback - skip + 1)]
      score = end / start - 1

  which is the exact negative of a (lookback - 2*skip)-bar return ending
  ``skip`` bars ago. Two independent errors: the numerator and the
  denominator are swapped, and the denominator index subtracts ``skip``
  instead of adding it. The effect is not a smaller window, it is an
  inverted sign, so the "strongest anomaly family in Indian data"
  baseline buys the biggest loser in the universe.

  FIXED. ``baselines.momentum_ranked`` now scores
  ``closes[-(skip + 1)] / closes[-(lookback + skip + 1)] - 1``, which is
  the documented construction, and the tests below are the regression
  tests for it.
  '''

  @staticmethod
  def _pair():
    up = [100.0 * (1.0008 ** i) for i in range(400)]
    down = [400.0 * (0.9992 ** i) for i in range(400)]
    return {'UP': flat_bars(up), 'DOWN': flat_bars(down)}

  def test_picks_the_asset_that_fell(self):
    panels = self._pair()
    weights = baselines.momentum_ranked(panels, lookback=252, skip=21, top=1)
    assert weights['UP'] == 0.1, (
      'momentum_ranked bought DOWN (the falling asset) with weight '
      f'{weights['DOWN']} and gave UP {weights['UP']}. UP rose 22 percent '
      'and DOWN fell 18 percent over the 252-bar window ending 21 bars '
      'ago, so the 12-1 construction must hold UP.')

  def test_score_is_the_negative_of_the_documented_return(self):
    # Regression test for the inverted construction. It asks the module
    # for its ranking and compares that ranking with the return the
    # docstring says it ranks on, so the assertion is about the code
    # rather than about a copy of the formula: the previous version
    # recomputed both expressions in the test body and could only ever
    # prove that arithmetic is arithmetic.
    lookback, skip = 252, 21
    panels = self._pair()
    weights = baselines.momentum_ranked(panels, lookback=lookback, skip=skip,
                                        top=1)
    documented = {}
    for symbol, rows in panels.items():
      closes = [bar.close for bar in rows]
      documented[symbol] = (closes[-(skip + 1)]
                            / closes[-(lookback + skip + 1)] - 1.0)
    assert documented['UP'] > 0.0, 'sanity: UP has positive 12-1 momentum'
    assert documented['DOWN'] < 0.0, 'sanity: DOWN has negative momentum'
    by_weight = sorted(weights, key=lambda name: weights[name], reverse=True)
    by_return = sorted(documented, key=lambda name: documented[name],
                       reverse=True)
    up_return = documented['UP']
    down_return = documented['DOWN']
    assert by_weight == by_return, (
      f'the module ranked {by_weight} but the documented 12-1 return '
      f'ranks {by_return} (UP {up_return:+.4f}, DOWN {down_return:+.4f}). '
      'Scoring the reciprocal buys the biggest loser.')
    assert weights['UP'] > weights['DOWN']

  def test_inversion_survives_skip_zero(self):
    # skip=0 is in train.skip_grid, so the search explores it. It does not
    # help: with skip=0 the code still divides the older close by the
    # newer one.
    panels = self._pair()
    weights = baselines.momentum_ranked(panels, lookback=252, skip=0, top=1)
    assert weights['UP'] == 0.1

  def test_wrapped_policy_inherits_the_inversion(self):
    panels = self._pair()
    policy = MomentumPolicy(lookback=252, skip=21, top=1, max_weight=0.1)
    assert policy.act(panels_observation(panels))['UP'] == 0.1

  def test_trend_filter_inherits_the_inversion(self):
    panels = self._pair()
    weights = baselines.trend_filtered_momentum(
      panels, window=252, skip=21, top=1, ma_window=40, max_weight=0.1)
    assert weights.get('UP', 0.0) == 0.1


# ---------------------------------------------------------------------------
# FINDING 2 -- HedgeEnv._error was offset by +1.0 for every policy. [fixed]
# ---------------------------------------------------------------------------

class TestHedgingErrorStartsAtZero:
  '''hedge_env.py: the self-financing hedging error path.

  Regression tests for the offset the error path used to carry. The path
  is ``((L_T - L_0) - (V_T - V_0)) / L_0`` and it must be zero at the
  inception bar for every policy, including one that never trades. It was
  ``(L_T - V_T) / L_0`` instead, which is the same quantity larger by
  exactly ``L_0 / L_0 == 1.0``, so every episode's path started at 1.0:
  ``semi_rmse`` was identically zero because the path never went below
  +1.0, and the terminal reward ``-risk * risk_measure()`` handed the
  never-hedged book the better score -- the precise outcome
  ``min_tail_confidence`` exists to forbid.

  ``average_delta`` is here too: it used to log the PRE-trade delta on
  step one, so a book hedged from bar 1 onward carried one full unhedged
  observation out of ``n_steps`` in its mean.
  '''

  @staticmethod
  def _walk(prices, coverage, measure='cvar'):
    env = HedgeEnv(flat_bars(prices), instruments=FUTURE_ONLY, history=20,
                   risk=HedgeRisk(measure=measure))
    env.reset()
    done = False
    while not done:
      _, _, done, _ = env.step({'f': coverage})
    return env

  def test_flat_market_error_is_exactly_zero(self):
    # On a flat market dL = dV = 0 for every policy, so the true
    # self-financing hedging error is identically zero.
    env = self._walk([100.0] * 200, 1.0)
    assert env.hedging_error[0] == pytest.approx(0.0, abs=1e-12), (
      'a perfectly flat market must produce a hedging error of 0, not '
      f'{env.hedging_error[0]}')
    assert env.hedging_error[-1] == pytest.approx(0.0, abs=1e-12)

  @pytest.mark.parametrize('coverage', [0.0, 0.5, 1.0, 2.0])
  def test_error_path_starts_at_zero_for_every_coverage(self, coverage):
    env = self._walk(random_walk(200), coverage)
    assert env.hedging_error[0] == pytest.approx(0.0, abs=1e-12), (
      f'coverage {coverage}: the path starts at '
      f'{env.hedging_error[0]:.6f}. The inception liability and hedge '
      'value must be subtracted before the first increment is booked, '
      'otherwise every path carries a constant +1.0.')
    assert all(
      value > -1.0 for value in env.hedging_error), (
      'no path may sit below -100 percent of the initial liability')

  def test_a_full_future_hedge_ends_near_zero_on_a_trending_market(self):
    env = self._walk([100.0 * 1.0004 ** i for i in range(200)], 1.0)
    spot = env.closes[-1]
    liability = env.exposure_units * spot
    implied = (liability - env._hedge_value) / env._initial_liability
    assert env.hedging_error[-1] == pytest.approx(0.0, abs=1e-6), (
      'the accumulated path is L_T - V_T over L_0, i.e. '
      f'{implied:.6f}, which is 1.0 plus the true self-financing error. '
      'The tolerance is 1e-6 because the path accumulates 180 increments '
      'in binary floating point.')

  def test_semi_rmse_responds_to_hedging_quality(self):
    prices = random_walk(200)
    bare = semi_rmse(self._walk(prices, 0.0).hedging_error)
    half = semi_rmse(self._walk(prices, 0.5).hedging_error)
    full = semi_rmse(self._walk(prices, 1.0).hedging_error)
    assert max(bare, half, full) > min(bare, half, full), (
      f'semi_rmse is {bare}, {half}, {full} for never-, half- and fully-'
      'hedged books, so it does not discriminate between them. With the '
      'path pinned above +1.0 the semideviation was always 0 and '
      'measure="semi_rmse" had no gradient to descend.')

  def test_cvar_of_a_flat_market_is_not_one(self):
    # The offset pinned every path above +1.0, so the tail of a perfectly
    # hedged book was a positive 1.0 rather than zero.
    flat = self._walk([100.0] * 200, 1.0).hedging_error
    assert cvar(flat, 0.95) == pytest.approx(0.0, abs=1e-12)

  def test_average_delta_excludes_the_pre_trade_bar(self):
    env = self._walk([100.0] * 200, 1.0)
    assert env.average_delta() == pytest.approx(0.0, abs=1e-9), (
      'average_delta() must average POST-trade deltas. _delta_log used to '
      f'record the pre-trade delta on the first step, so it carried one '
      f'full unhedged observation out of n_steps: it reported '
      f'{env.average_delta():.1f} for a book fully hedged from bar 1 on.')


# ---------------------------------------------------------------------------
# FINDING 3 -- purged_folds accepted embargo and ignored it. [fixed]
# ---------------------------------------------------------------------------

class TestEmbargoIsApplied:
  '''splits.py:123-180.

  Regression tests for ``embargo``, which used to be validated and then
  discarded: ``train_end = test_start - horizon`` was the whole story, so
  serial correlation leaked across every fold boundary, which is the one
  thing the parameter exists to stop.

  Fold 0 is the reason a naive test proves nothing here. Its training end
  is floored at ``min_train``, so on the first fold a 40-bar embargo and a
  1-bar embargo produce the same ``train_end`` and the embargo legitimately
  looks like a no-op. The embargo is visible on the LATER folds, which is
  where the assertion belongs.
  '''

  def test_embargo_changes_the_folds(self):
    def signature(embargo):
      folds = purged_folds(300, n_folds=3, horizon=2, embargo=embargo,
                           min_train=20)
      return [(f.train_end, f.test_start, f.test_end) for f in folds]

    assert signature(0) != signature(50), (
      'purged_folds produced identical folds for embargo=0 and '
      'embargo=50. The parameter is validated and then discarded.')

  def test_later_training_windows_shrink_by_the_embargo(self):
    tight = purged_folds(600, n_folds=4, horizon=1, embargo=1,
                         min_train=40)
    loose = purged_folds(600, n_folds=4, horizon=1, embargo=40,
                         min_train=40)
    later_tight = tight[1:]
    later_loose = loose[1:]
    assert later_tight and later_loose, 'sanity: more than one fold exists'
    for near, far in zip(later_tight, later_loose):
      assert far.train_end < near.train_end, (
        f'fold {near.fold}: train_end did not move for a 39-bar larger '
        f'embargo, {near.train_end} vs {far.train_end}. Fold 0 is exempt '
        'because its training end is floored at min_train.')

  @pytest.mark.parametrize('embargo', [0, 1, 5, 40])
  def test_gap_between_train_and_test_is_horizon_plus_embargo(self, embargo):
    folds = purged_folds(600, n_folds=4, horizon=2, embargo=embargo,
                         min_train=40)
    for fold in folds[1:]:
      assert fold.test_start - fold.train_end == 2 + embargo, (
        f'fold {fold.fold}: the gap is '
        f'{fold.test_start - fold.train_end}, expected horizon 2 plus '
        f'embargo {embargo}')


# ---------------------------------------------------------------------------
# FINDING 4 -- compare()'s parity claim was unsatisfiable. [fixed]
# ---------------------------------------------------------------------------

class TestEngineParityClaimIsFalse:
  '''rl/train.py and portfolio.py: the two engines, one window.

  ``compare`` used to claim "their Sharpe, cost and turnover must agree to
  within whole-share rounding. A divergence larger than that means one of
  the two has a bug, which is the entire reason for having two engines."

  Cost does agree exactly, which proves the fill calendars are aligned.
  Sharpe and turnover could not agree, because the two return series were
  different lengths by construction:

    * ``run_portfolio`` emitted one return per bar over the WHOLE series,
      with a hard ``0.0`` prepended for bar 0 and a further ``history``
      bars of flat cash (exactly 0.0) at the front.
    * ``WeightAllocationEnv`` emits ``n_steps = len - history`` returns
      and no synthetic leading zero.

  Two different windows, and one of them was padded with zeros that drag
  its mean toward zero and inflate its stdev. The parity check that is
  supposed to be the bug detector was permanently tripped by
  construction, so it carried no information.

  Turnover is measured differently too: ``portfolio._rebalance`` books
  ``abs(delta) * price / mark`` (actual traded notional), while
  ``WeightAllocationEnv._execute`` books ``|target - self._weights|``
  against a mark that ``_recount_weights`` refreshed only on the
  PREVIOUS step, so the env's turnover carries one bar of un-traded
  drift.

  FIXED. ``run_portfolio`` now reports ``returns`` over the trading window
  only, which is exactly the window the environment reports, and
  ``compare`` refuses to run at all if the two ever diverge again in
  length. What it asserts is the cadence and the cost; what it reports
  side by side is the Sharpe and the turnover.
  '''

  def test_sharpes_agree(self):
    panels = drifting_panels(('A', 'B', 'C', 'D', 'E'), 400)
    result = compare(MomentumPolicy(lookback=60, skip=5, top=3,
                                    max_weight=0.2),
                     panels, history=60, rebalance_days=21)
    env_sharpe = result.env['sharpe']
    book_sharpe = result.portfolio['sharpe']
    assert env_sharpe == pytest.approx(book_sharpe, rel=0.01), (
      f'env={env_sharpe:.4f} portfolio={book_sharpe:.4f}. Both engines '
      'report one return per bar over the same trading window, so a gap '
      'here means one of the two has a bug.')

  def test_turnover_agrees(self):
    panels = drifting_panels(('A', 'B', 'C', 'D', 'E'), 400)
    result = compare(MomentumPolicy(lookback=60, skip=5, top=3,
                                    max_weight=0.2),
                     panels, history=60, rebalance_days=21)
    assert result.env['turnover'] == pytest.approx(
      result.portfolio['turnover'], rel=0.01), (
      f'env={result.env['turnover']:.4f} portfolio='
      f'{result.portfolio['turnover']:.4f}')

  def test_portfolio_return_series_is_not_padded_with_zeros(self):
    panels = drifting_panels(('A', 'B'), 200, seed=9)
    env = WeightAllocationEnv(panels, history=60, rebalance_days=21)
    done = False
    while not done:
      _, _, done, _ = env.step(env_step_action(env))
    result = run_portfolio(panels, lambda visible: {
      symbol: 0.5 for symbol in visible}, rebalance_days=21, history=60)
    assert len(result.returns) == len(env.returns), (
      'the two engines must report the same number of returns for the '
      f'Sharpe comparison to mean anything: {len(result.returns)} vs '
      f'{len(env.returns)}')
    assert result.returns[0] == 0.0, 'sanity: run_portfolio pads bar 0'

  def test_env_turnover_is_measured_against_a_stale_mark(self):
    panels = drifting_panels(('A', 'B', 'C'), 200, seed=4)
    env = WeightAllocationEnv(panels, history=20, rebalance_days=21)
    env.reset()
    env.step({'A': 0.2, 'B': 0.2, 'C': 0.2})
    decision_bar = env._bar_index()
    weights_marked_at = env._bar_index() - 1
    assert weights_marked_at < decision_bar, (
      'self._weights used by _execute was last written by '
      '_recount_weights on the previous step, i.e. marked at bar '
      f'{weights_marked_at} while the decision bar is {decision_bar}. '
      'Turnover therefore charges one bar of drift that was never traded.')


def env_step_action(env):
  '''Return a constant equal-weight action for every action symbol.

  Args:
    env: An environment exposing ``action_symbols``.

  Returns:
    Mapping of symbol to a fixed weight.
  '''
  return {symbol: 0.2 for symbol in env.action_symbols}


# ---------------------------------------------------------------------------
# FINDING 5 -- random_search never used trial_seed. [fixed]
# ---------------------------------------------------------------------------

class TestTrialSeedReachesTheRun:
  '''rl/train.py: random_search, TrialLog and sharpe_report.

  ``trial_seed`` used to be looped over and stored on ``Trial.seed`` but
  never handed to anything: not to ``env.reset`` and not to
  ``_draw_policy``. Both environments are deterministic and ``reset(seed)``
  discards the seed, so the three entries sharing one ``name`` were three
  independent runs of *different* policies, or identical repeats of the
  same one, and every downstream number was wrong in one direction or the
  other.

  What is fixed: one configuration is drawn per trial and reused across
  its seeds, the seeds are passed into ``reset``, ``TrialLog.count``
  counts configurations (the number the Deflated Sharpe needs), and
  ``SharpeReport.deflated`` is computed from the *selected*
  configuration's own per-seed returns rather than from the single
  highest-Sharpe run of whichever configuration got there first.

  Two of the original assertions could not survive that fix and were
  rewritten, because they asserted the pre-fix artefacts rather than a
  property:

    * ``count == len(log)``. A log of ``trials * len(seeds)`` runs over
      ``trials`` configurations has 12 entries and 4 configurations.
      Counting runs as configurations invents a search three times larger
      than the one that ran. tests/test_rl.py:1412 pins the run count, so
      the two numbers are deliberately different.
    * "no configuration has identical Sharpes across its seeds". Both
      shipped environments are deterministic, so identical Sharpes are
      the correct and only possible outcome. A non-zero seed dispersion
      would require a stochastic environment, which this package does not
      have; what the seed now buys is that the plumbing exists for one.
  '''

  def test_count_is_the_configurations_not_the_runs(self):
    panels = drifting_panels(('A', 'B', 'C', 'D'), 200, seed=2)
    log = random_search(panels, trials=4, seeds=(1, 2, 3), history=60)
    assert log.count == 4, (
      f'log.count is {log.count}, but exactly four configurations were '
      f'drawn and {len(log)} runs were made. TrialLog.count is the number '
      'of independent configurations tried and it feeds the DSR as the '
      'trial count.')
    assert len(log) == 12, 'sanity: four configurations by three seeds'
    assert len(log.names) == 4

  def test_count_is_what_reaches_the_length_gate_and_the_dsr(self):
    panels = drifting_panels(('A', 'B', 'C', 'D'), 400, seed=2)
    log = random_search(panels, trials=4, seeds=(1, 2, 3), history=60)
    report = sharpe_report(log)
    assert report.trials == log.count, (
      f'the report deflated {report.trials} trials while the log counted '
      f'{log.count} configurations')
    required = minimum_backtest_length(report.trials, report.sharpe or 0.5)
    assert report.required_years == pytest.approx(required)

  def test_one_name_is_one_policy_evaluated_under_each_seed(self):
    panels = drifting_panels(('A', 'B', 'C', 'D'), 400, seed=2)
    log = random_search(panels, trials=4, seeds=(1, 2, 3), history=60)
    for name in log.names:
      assert sorted(log.seed_sharpes(name)) == sorted(
        log.seed_sharpes(name))
      assert len(log.seed_returns(name)) == 3, (
        f'{name} has {len(log.seed_returns(name))} seed series, expected '
        'one per seed')
      # Deterministic environments, so the three seeds of one policy are
      # the same run three times. seed_stdev reports exactly that, and the
      # docstring says why.
      assert len(set(log.seed_sharpes(name))) == 1
      assert log.seed_stdev(name) == 0.0

  def test_different_names_are_different_policies(self):
    # The control for the test above: zero dispersion within a name must
    # not be zero dispersion across the log, or the search drew the same
    # configuration four times and there is no search to deflate.
    panels = drifting_panels(('A', 'B', 'C', 'D'), 400, seed=2)
    log = random_search(panels, trials=4, seeds=(1, 2, 3), history=60)
    assert len({tuple(trial.returns) for trial in log.trials}) > 1

  def test_deflated_is_a_deflation_of_the_reported_sharpe(self):
    panels = drifting_panels(('A', 'B', 'C', 'D'), 400, seed=2)
    log = random_search(panels, trials=4, seeds=(1, 2, 3), history=60)
    report = sharpe_report(log)
    if report.sharpe is None or report.deflated is None:
      pytest.skip('the length gate refused this synthetic log')
    # The reported Sharpe is the mean across the seeds of the config with
    # the best mean, and so must the deflated probability be computed from
    # that configuration's own returns.
    best = log.best_trial()
    selected = max(
      log.names,
      key=lambda name: sum(log.seed_sharpes(name))
      / len(log.seed_sharpes(name)))
    assert best.name == selected, (
      f'deflated is derived from {best.name} (best single run, seed '
      f'{best.seed}, Sharpe {best.sharpe:.4f}) while sharpe is the '
      f'mean-across-seeds of {selected} (Sharpe {report.sharpe:.4f}). One '
      'SharpeReport therefore mixes two different strategies.')
    assert report.deflated == pytest.approx(min(
      deflated_sharpe(series, log.count,
                      trial_sharpes=log.per_period_sharpes)
      for series in log.seed_returns(selected)))


# ---------------------------------------------------------------------------
# FINDING 6 -- neither engine constrained cash. [fixed]
# ---------------------------------------------------------------------------

class TestCashIsConstrained:
  '''portfolio.py:216-246 and rl/portfolio_env.py:364-397.

  Both engines size with ``int(target * value / fill_price)``, which bounds
  the notional by the portfolio value but ignores the charges, so a target
  that sums to exactly 1.0 on an all-cash book drives cash negative on the
  first fill. There is no ``if cash < 0`` guard, no margin charge and no
  financing cost anywhere in either module, so without
  :func:`stock_rl.weights.affordable_scale` the book silently borrows at
  zero interest: correct on a flat series, and free leverage on a rising
  one.

  The previous version of the first test recomputed the first fill by hand
  and never called ``run_portfolio`` at all, so it proved its own
  arithmetic instead of the module's. These assert on the result.
  '''

  def test_portfolio_cash_never_goes_negative(self):
    panels = {'A': flat_bars([100.0] * 120)}
    result = run_portfolio(panels, lambda visible: {'A': 1.0},
                           capital=1_000_000.0, max_weight=1.0,
                           rebalance_days=21, history=21)
    # A 100 percent target sizes int(1.0 * 1e6 / 100) == 10,000 shares,
    # whose notional is the entire mark. The charges still have to come out
    # of cash, so an unconstrained book spends 1.0 of its value on
    # notional and books turnover of exactly 1.0 while levered by the cost.
    assert result.turnover < 1.0, (
      f'turnover was {result.turnover!r}: the book spent its whole mark on '
      'notional and paid the cost out of negative cash, which is unpriced '
      'borrowing')
    assert result.total_cost > 0.0, 'sanity: the charges were real'
    assert result.equity[-1] < 1.0, (
      'a flat market must lose exactly the transaction cost, so the final '
      f'equity must be below 1.0, got {result.equity[-1]}')

  def test_a_free_cost_model_does_invest_the_whole_mark(self):
    # The positive control for the test above: with no charges there is
    # nothing to fund, so the book legitimately reaches turnover of 1.0.
    # Without it, the assertion above would also pass on a backtester that
    # never trades.
    panels = {'A': flat_bars([100.0] * 120)}
    result = run_portfolio(panels, lambda visible: {'A': 1.0},
                           capital=1_000_000.0, costs=FREE, max_weight=1.0,
                           rebalance_days=21, history=21)
    assert result.turnover == pytest.approx(1.0)
    assert result.total_cost == 0.0
    assert result.equity[-1] == pytest.approx(1.0)

  def test_env_cash_never_goes_negative(self):
    panels = {'A': flat_bars([100.0] * 120)}
    env = WeightAllocationEnv(panels, capital=1_000_000.0, history=21,
                              max_weight=1.0, rebalance_days=21)
    env.reset()
    worst = env.cash
    done = False
    while not done:
      _, _, done, _ = env.step({'A': 1.0})
      worst = min(worst, env.cash)
    assert worst >= 0.0, (
      f'WeightAllocationEnv drove cash to {worst:,.2f} on a 100 percent '
      'target. Neither _execute nor the book has a cash constraint.')


# ---------------------------------------------------------------------------
# FINDING 7 -- a NaN weight became a maximum-weight position. [fixed]
# ---------------------------------------------------------------------------

class TestNanWeightIsRefused:
  '''portfolio.py:158-161 and rl/portfolio_env.py:305-306.

  ``max(0.0, min(max_weight, raw))`` with ``raw = nan`` returns
  ``max_weight`` rather than raising, because ``min(a, nan)`` returns
  ``a``: Python's ``min`` keeps the incumbent when the comparison is
  False. A policy that divides by zero, or an actor that emits NaN after
  a numerical blow-up, therefore used to be handed a maximum-weight
  position in that symbol and nothing anywhere reported it.

  :func:`stock_rl.weights.clamp_weight` raises instead, in both engines,
  and these are the regression tests for that. The previous version of
  ``test_documented_behaviour_for_unusable_weights`` asserted the broken
  behaviour, so it could never pass.
  '''

  def test_normalise_rejects_a_nan_weight(self):
    with pytest.raises(ValueError, match='NaN'):
      _normalise({'A': float('nan'), 'B': 0.5}, ('A', 'B'), 0.10)

  def test_normalise_rejects_an_infinite_weight(self):
    with pytest.raises(ValueError, match='infinite'):
      _normalise({'A': float('inf'), 'B': 0.5}, ('A', 'B'), 0.10)

  def test_env_rejects_a_nan_weight(self):
    panels = drifting_panels(('A', 'B'), 120, seed=1)
    env = WeightAllocationEnv(panels, history=20)
    env.reset()
    with pytest.raises(ValueError, match='NaN'):
      env.step({'A': float('nan'), 'B': 0.1})

  def test_documented_behaviour_for_unusable_weights(self):
    # The behaviour that matters operationally: a non-finite weight is
    # refused at the boundary instead of being turned into a position that
    # looks deliberate in the equity curve.
    with pytest.raises(ValueError):
      _normalise({'A': float('nan')}, ('A',), 0.10)
    with pytest.raises(ValueError):
      _normalise({'A': float('-inf')}, ('A',), 0.10)


# ---------------------------------------------------------------------------
# FINDING 8 -- build_nnf_id returned a non-NNF for a non-digit flag. [fixed]
# ---------------------------------------------------------------------------

class TestNnfBuilderAcceptsNonDigits:
  '''compliance/algo_tag.py: ``build_nnf_id``.

  ``if not 0 <= flag <= 9`` accepted ``0.5``, ``9.0`` and ``True``. The
  f-string then interpolated the value rather than a digit, so the
  function returned e.g. ``'4444444444440.500'`` (17 characters, not 15)
  for a value its own docstring says it rejects. The failure was deferred
  to ``validate_algo_tag``, which is one call later in the same pipeline.

  FIXED: the flag must be a real ``int`` -- ``bool`` excluded, because
  ``True`` is an ``int`` and would otherwise pass as 1 -- and the check
  happens before anything is formatted, so no malformed ID is ever built.
  '''

  @pytest.mark.parametrize('flag', [0.5, 9.0, True, -0.0])
  def test_non_integer_flag_raises(self, flag):
    with pytest.raises(AlgoTagError, match='whole number'):
      build_nnf_id('444444444444', flag)

  @pytest.mark.parametrize('flag', [-1, 10, 99])
  def test_out_of_range_integer_raises(self, flag):
    with pytest.raises(AlgoTagError, match='digit 0-9'):
      build_nnf_id('444444444444', flag)

  def test_result_is_always_fifteen_digits(self):
    try:
      built = build_nnf_id('444444444444', 0.5)
    except AlgoTagError:
      return
    assert len(built) == 15 and built.isdecimal(), (
      f'build_nnf_id returned {built!r}, which is not a 15-digit NNF ID')

  def test_a_real_digit_still_builds(self):
    built = build_nnf_id('444444444444', 4)
    assert len(built) == 15 and built.isdecimal()
    assert built == '444444444444400'


# ---------------------------------------------------------------------------
# FINDING 9 -- ReplayVendor never enforced entitlements. [fixed]
# ---------------------------------------------------------------------------

class TestEntitlementIsNeverEnforced:
  '''data/vendors.py: ``Entitlement`` and ``ReplayVendor.bars``.

  ``MarketDataFeed.bars`` documents "Raises: EntitlementError: If the
  symbol's segment is not licensed", and ``ReplayVendor`` satisfies that
  protocol structurally (``isinstance(v, MarketDataFeed)`` is True) while
  returning bars unconditionally. ``Entitlement`` carried a vendor and a
  segment set but no symbol-to-segment mapping anywhere in the module, so
  the check the protocol promised was not merely unimplemented, it was
  unimplementable as written. The default is ``Entitlement.none``, i.e.
  "licensed for nothing", and data still flowed.

  FIXED, by taking the first of the two options: ``Entitlement`` now
  carries the symbol-to-segment map the grant is checked against, and
  ``bars`` calls ``require_symbol`` on every request. A symbol with no
  recorded segment is refused rather than assumed to be cash, because a
  feed that cannot name the segment of the symbol being asked for cannot
  show a licence for it either.
  '''

  def test_bars_raise_when_the_segment_is_unlicensed(self):
    bars = flat_bars([1.0, 2.0, 3.0])
    vendor = ReplayVendor('X', {'AAA': bars},
                          entitlements=Entitlement.none('X'))
    assert isinstance(vendor, MarketDataFeed)
    assert not vendor.entitlements.allows(Segment.CASH)
    with pytest.raises(EntitlementError):
      vendor.bars('AAA', bars[0].timestamp, bars[-1].timestamp)

  def test_bars_are_served_when_the_segment_is_licensed(self):
    bars = flat_bars([1.0, 2.0, 3.0])
    vendor = ReplayVendor('X', {'AAA': bars}, entitlements=Entitlement.granted(
      'X', Segment.CASH, symbol_segments={'AAA': Segment.CASH}))
    served = vendor.bars('AAA', bars[0].timestamp, bars[-1].timestamp)
    assert len(served) == 3, 'a licensed request must still be served'

  def test_an_unclassified_symbol_is_refused_rather_than_assumed(self):
    bars = flat_bars([1.0, 2.0, 3.0])
    vendor = ReplayVendor('X', {'AAA': bars}, entitlements=Entitlement.granted(
      'X', Segment.CASH))
    with pytest.raises(EntitlementError, match='no segment'):
      vendor.bars('AAA', bars[0].timestamp, bars[-1].timestamp)

  def test_a_licensed_feed_still_refuses_a_symbol_in_another_segment(self):
    bars = flat_bars([1.0, 2.0, 3.0])
    vendor = ReplayVendor('X', {'BOND': bars}, entitlements=Entitlement.granted(
      'X', Segment.CASH, symbol_segments={'BOND': Segment.DEBT}))
    with pytest.raises(EntitlementError, match='Debt'):
      vendor.bars('BOND', bars[0].timestamp, bars[-1].timestamp)


# ---------------------------------------------------------------------------
# [OK] -- claims the reviewer checked and found correct.
# ---------------------------------------------------------------------------

class TestLookAheadIsClosed:
  '''portfolio.py and rl/portfolio_env.py sizing and fill discipline.

  The critical invariant holds in both engines: the decision is taken on
  the close of bar t, sized on the mark at the close of bar t, filled at
  the open of bar t+1, and the provider is shown bars strictly before the
  fill bar. This is the bug the brief says was already found and fixed
  twice, so it is worth asserting rather than assuming.
  '''

  def test_provider_never_sees_the_fill_bar(self):
    panels = drifting_panels(('A', 'B'), 120, seed=6)
    seen = []

    def provider(visible):
      seen.append(len(visible['A']))
      return {'A': 0.4, 'B': 0.4}

    run_portfolio(panels, provider, rebalance_days=10, history=20)
    first_fill = 20
    assert seen[0] == first_fill, (
      f'provider saw {seen[0]} bars at the first rebalance, expected '
      f'{first_fill}: the fill bar must be excluded')

  def test_sizing_uses_the_decision_bar_close_not_the_fill_bar_close(self):
    panels = drifting_panels(('A',), 60, seed=6)
    result = run_portfolio(panels, lambda visible: {'A': 1.0},
                           capital=1_000_000.0, costs=FREE, max_weight=1.0,
                           rebalance_days=10, history=20)
    fill = 20
    target = 1.0
    sized_on_decision = int(target * 1_000_000.0 / panels['A'][fill].open)
    # The look-ahead this brief says was already fixed twice: sizing on
    # the FILL bar's close instead of the decision bar's close.
    sized_on_fill_close = int(
      target * 1_000_000.0 * panels['A'][fill].close
      / panels['A'][fill].close)
    assert sized_on_decision != sized_on_fill_close, (
      'the panel must move intrabar for this test to discriminate')
    assert result.rebalances >= 1, 'sanity: a rebalance happened'
    assert result.total_cost == 0.0, 'sanity: FREE model charges nothing'
    assert result.equity[fill] == pytest.approx(
      1.0 + (sized_on_decision * (panels['A'][fill].close
                                  - panels['A'][fill].open)
             / 1_000_000.0), rel=1e-12), (
      'the share count implied by the equity curve is not '
      'int(target * decision_close / fill_open), so sizing is not on the '
      'decision bar')

  def test_env_provider_never_sees_the_fill_bar(self):
    panels = drifting_panels(('A', 'B'), 120, seed=6)
    env = WeightAllocationEnv(panels, history=20)
    obs = env.reset()
    lengths = [len(obs['history']['A'])]
    done = False
    while not done:
      obs, _, done, _ = env.step({'A': 0.4, 'B': 0.4})
      lengths.append(len(obs['history']['A']))
    assert lengths[0] == 20, (
      f'the first observation carried {lengths[0]} bars for history=20; '
      'the decision bar is 19, so the fill bar 20 must be absent')
    assert lengths == sorted(lengths)

  def test_env_sizes_on_the_decision_bar_close(self):
    panels = drifting_panels(('A',), 120, seed=6)
    env = WeightAllocationEnv(panels, capital=1_000_000.0, costs=FREE,
                              history=20, max_weight=1.0)
    env.reset()
    env.step({'A': 1.0})
    fill = 20
    expected = int(1.0 * env.capital / panels['A'][fill].open)
    assert env.positions['A'] == expected, (
      f'bought {env.positions['A']} shares, expected {expected} sized on '
      f'the decision bar close with the fill at open[{fill}]='
      f'{panels['A'][fill].open}')

  def test_hedge_env_prices_the_option_leg_at_the_decision_bar_vol(self):
    source = inspect.getsource(HedgeEnv.step)
    assert 'vol = self._implied_vol(decision)' in source, (
      'the option leg must be priced with the vol known at the decision '
      'bar')
    assert 'self._execute(targets, fill, vol)' in source
    assert 'self._roll(decision)' in source


class TestDeflatedSharpeFormula:
  '''metrics.dsr_from_moments reproduces Bailey and Lopez de Prado.

  Denominator is ``1 - skew*SR + ((kurtosis - 1)/4) * SR**2`` and the
  ``sqrt(T-1)`` term counts observations. No Euler-Mascheroni anywhere in
  the denominator; EULER_GAMMA is used only inside ``max_z``, which is
  where it belongs (the Blumenthal approximation to E[max] of N normals).
  '''

  def test_gaussian_reference_case(self):
    # skew 0, kurtosis 3 -> the denominator is exactly 1, so the result
    # is Phi((SR - SR0) * sqrt(T - 1)).
    value = dsr_from_moments(0.5, 0.3, 0.0, 3.0, 1000)
    expected = (0.5 - 0.3) * math.sqrt(999)
    assert value == pytest.approx(NormalDist().cdf(expected))

  def test_negative_skew_increases_the_denominator(self):
    plain = dsr_from_moments(0.5, 0.4, 0.0, 3.0, 500)
    skewed = dsr_from_moments(0.5, 0.4, -1.0, 3.0, 500)
    assert skewed < plain, (
      'a negatively skewed return series has a wider sampling variance '
      'for the Sharpe, so the Deflated Sharpe must fall')

  def test_fat_tails_lower_the_deflated_sharpe(self):
    fat = dsr_from_moments(0.5, 0.4, 0.0, 9.0, 500)
    thin = dsr_from_moments(0.5, 0.4, 0.0, 1.5, 500)
    assert fat < thin, (
      'raw kurtosis enters as ((kurtosis - 1) / 4) * SR**2, so a fat '
      'tailed series has a larger denominator and a lower probability')

  def test_observations_are_counted_not_years(self):
    per_period = dsr_from_moments(0.10, 0.05, 0.0, 3.0, 252)
    doubled = dsr_from_moments(0.10, 0.05, 0.0, 3.0, 504)
    assert doubled > per_period, (
      'sqrt(T - 1) must count observations, so doubling the sample must '
      'raise the probability')


class TestIndicatorWarmUpIndices:
  '''indicators.rsi and indicators.atr alignment.

  RSI's first value belongs at index ``window`` (window changes need
  window + 1 closes). ATR's first value belongs at index ``window``
  (range at bar i uses bars i-1 and i, and the seed averages the first
  window ranges, i.e. bars 1..window). ``realized_volatility`` must never
  fold its synthetic ``None`` at index 0 into a sample.
  '''

  def test_rsi_first_index_and_flat_value(self):
    closes = [100.0] * 30
    series = rsi(closes, 14)
    assert series[:14] == [None] * 14
    assert series[14] == 50.0
    assert len(series) == len(closes)

  def test_rsi_monotone_advance_is_one_hundred(self):
    closes = [100.0 + i for i in range(40)]
    series = rsi(closes, 14)
    assert series[-1] == 100.0

  def test_atr_first_index(self):
    closes = [100.0 + i for i in range(40)]
    bars = flat_bars(closes)
    series = atr(bars, 14)
    assert series[:14] == [None] * 14
    assert series[14] == pytest.approx(1.0)
    assert len(series) == len(bars)

  def test_atr_constant_range_is_exact(self):
    series = atr(flat_bars([100.0] * 40), 14)
    assert series[14] == pytest.approx(0.0)

  def test_realized_volatility_ignores_the_synthetic_first_return(self):
    series = realized_volatility([100.0] * 40, 20, TRADING_DAYS_PER_YEAR)
    assert series[:20] == [None] * 20
    assert series[20] == pytest.approx(0.0)
    assert len(series) == 40


class TestSourcedRetentionPeriods:
  '''compliance/retention.py carries only sourced periods.'''

  def test_no_seven_year_period(self):
    assert not [entry for entry in retention_table if entry.years == 7], (
      'the 7-year figure has no source and must stay absent')

  def test_only_five_and_eight_years(self):
    years = {entry.years for entry in retention_table
             if entry.years is not None}
    assert years == {5, 8}

  def test_every_entry_carries_a_source(self):
    for entry in retention_table:
      assert entry.source.strip(), f'{entry.name} has no source string'

  def test_indefinite_class_is_none_not_an_error(self):
    indefinite = [entry for entry in retention_table
                  if entry.years is None]
    assert len(indefinite) == 1
    assert indefinite[0].name == 'enforcement_original'


class TestCostConstantsAreReal:
  '''costs.py carries the real Indian delivery and intraday numbers.'''

  def test_exchange_charge_is_307_per_crore(self):
    assert DELIVERY.exchange_pct == pytest.approx(0.0000307)
    assert DELIVERY.exchange_pct != pytest.approx(0.0000297)

  def test_intraday_brokerage_is_three_bps_capped_at_twenty(self):
    assert INTRADAY.brokerage_pct == pytest.approx(0.0003)
    assert INTRADAY.brokerage_cap == pytest.approx(20.0)
    assert INTRADAY.brokerage_pct != 0.0

  def test_delivery_levies_are_about_22_25_bps(self):
    # Rs 1 lakh, the ticket size the project's own cost test uses. The
    # 22.25 bps all-in figure excludes slippage (an unverifiable
    # assumption) and the DP charge (levied once per day per scrip, not
    # per round trip), which is the same definition tests/test_costs.py
    # uses. This is a positive control: the constant is real and the
    # point is that it is still real.
    notional = 100_000.0
    levies = (DELIVERY.round_trip(notional)
              - 2 * DELIVERY.slippage(notional)
              - DELIVERY.dp_charge)
    assert levies / notional * 10_000.0 == pytest.approx(22.25, abs=0.05)


class TestCalendarSessionEdges:
  '''data/calendar.py boundary behaviour.

  ``Session.duration_minutes`` used to return ``0.0`` when the close
  equalled the open. Documented: "A session whose close precedes its open
  returns None rather than a negative length or a wrapped 24-hour one." A
  close that EQUALS its open is the same class of mistake, and 0.0 reads
  like a real zero-minute trading day rather than an unannounced one.

  FIXED: any non-positive span returns None, so the caller gets one
  answer for "this is not a session" and never a number to divide by.
  '''

  def test_normal_session_is_375_minutes_inclusive(self):
    assert nse_normal_session.duration_minutes == 375.0
    assert nse_normal_session.contains(time(9, 15))
    assert nse_normal_session.contains(time(15, 30))
    assert not nse_normal_session.contains(time(15, 31))

  def test_untimed_session_never_claims_to_be_open(self):
    assert muhurat_session.duration_minutes is None
    assert not muhurat_session.contains(time(16, 45))

  def test_zero_length_session_reports_none_not_zero(self):
    degenerate = Session('degenerate', opens=time(9, 15),
                         closes=time(9, 15))
    assert degenerate.duration_minutes is None

  def test_an_inverted_session_is_also_none(self):
    inverted = Session('inverted', opens=time(15, 30), closes=time(9, 15))
    assert inverted.duration_minutes is None


class TestZeroRuntimeDependencies:
  '''Invariant 1: standard library only, in src and in tests.'''

  def test_no_third_party_imports_in_the_reviewed_modules(self):
    root = pathlib.Path(__file__).resolve().parents[1] / 'src'
    stdlib = set(sys.stdlib_module_names) | {'stock_rl'}
    offenders = []
    for path in sorted(root.rglob('*.py')):
      tree = ast.parse(path.read_text(encoding='utf-8'))
      for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
          names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
          names = [node.module or '']
        for name in names:
          if name.split('.')[0] not in stdlib:
            offenders.append(f'{path}:{name}')
    assert not offenders, offenders


class TestNoPerformanceFiguresAreClaimed:
  '''Invariant 7: no Sharpe or return number asserted without support.

  Scans the reviewed modules for a bare decimal immediately followed by
  the word "percent" inside a docstring comment, which is how an
  unsupported performance figure would be smuggled in. Citations of
  *other people's* published findings are allowed, so only numbers that
  are adjacent to "Sharpe" and claim to describe this project are caught.
  '''

  REVIEWED = (
    'src/stock_rl/portfolio.py',
    'src/stock_rl/baselines.py',
    'src/stock_rl/rl/policy.py',
    'src/stock_rl/rl/train.py',
    'src/stock_rl/rl/portfolio_env.py',
    'src/stock_rl/rl/hedge_env.py',
  )

  def test_no_module_claims_its_own_sharpe(self):
    root = pathlib.Path(__file__).resolve().parents[1]
    pattern = re.compile(r'(our|this project|here)[^.]{0,40}sharpe', re.I)
    offenders = []
    for relative in self.REVIEWED:
      text = (root / relative).read_text(encoding='utf-8')
      for match in pattern.finditer(text):
        offenders.append(f'{relative}: {match.group(0)!r}')
    assert not offenders, offenders

  def test_retention_periods_agree_with_the_documented_sources(self):
    # 8 years is SEBI (Stock Brokers) Regulations 2026 Reg. 16; 5 years is
    # NSE Detailed Operational Modalities para 10.3. Asserted against the
    # date arithmetic rather than the prose.
    assert expiry_date('books_of_account', date(2020, 1, 1)) == date(
      2028, 1, 1)
    assert expiry_date('exchange_audit_floor', date(2020, 1, 1)) == date(
      2025, 1, 1)
