#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Tests for the baseline policy wrappers in :mod:`stock_rl.rl.policy`.

Four of the five baselines had no policy wrapper, which left
:mod:`stock_rl.pipeline` able to cross-check the two engines on
``momentum_ranked`` and nothing else: a wrapper that reimplements a
strategy agrees with itself, so it measures nothing. These tests hold
the wrappers to the one property that makes them useful, which is that
``wrapper.act(observation) == baseline(panels, **kwargs)`` exactly.

The second property under test is that every wrapper hands its
arguments to the baseline **by keyword**. That is not pedantry. The
fifth positional parameter of ``momentum_ranked`` is ``max_weight``
and the fifth positional parameter of ``trend_filtered_momentum`` is
``ma_window``, so a positional call written by copying one wrapper's
argument order feeds a weight cap into a moving-average window. Here
it happens to raise, because :func:`stock_rl.indicators.sma` rejects
a window below one; that is luck, and the keyword assertions below
would catch the same mistake on a baseline where it would not.

``buy_and_hold`` is the odd one out. Its cap belongs to
:func:`stock_rl.baselines.equal_weight` rather than to the passive
hypothesis, so :class:`BuyAndHoldPolicy` defaults it to ``None`` and
``None`` means *pass no cap at all*. The tests here check the
documented behaviour by recording what the baseline was actually
called with, because inferring it from an equal output would pass
even if the wrapper injected a cap that happened to match the
baseline's own default today.
'''

from __future__ import annotations

from datetime import datetime, timedelta
from functools import partial

import pytest

from stock_rl import baselines
from stock_rl.bars import Bar
from stock_rl.portfolio import run_portfolio
from stock_rl.rl import train
from stock_rl.rl.policy import (
  BuyAndHoldPolicy,
  EqualWeightPolicy,
  InverseVolPolicy,
  MomentumPolicy,
  Policy,
  TrendFilteredMomentumPolicy,
  panels_observation,
  policy_fingerprint,
  policy_parameters,
)
from stock_rl.rl.portfolio_env import WeightAllocationEnv
from stock_rl.rl.train import Trial, TrialLog, compare, random_search, rollout

#: Symbols per panel. Five against a 0.10 cap keeps ``1 / N`` strictly
#: above the cap, so equal weight and the ranking baselines do not
#: collapse onto the same uniform book and report the same Sharpe.
SYMBOLS = ('AAA', 'BBB', 'CCC', 'DDD', 'EEE')

#: Bars per panel. Long enough for a 60-bar lookback plus a 21-bar skip
#: to have history at the decision bars the engines actually reach.
BARS = 260

#: Warm-up bars before the first decision.
HISTORY = 60

#: Length of one environment episode: every decision bar, plus the
#: leading 0.0 that :func:`stock_rl.rl.train.rollout` prepends for the
#: starting equity curve. Asserted rather than computed so that a
#: change to either engine's window shows up as a failure here.
EPISODE = BARS - HISTORY + 1

#: The baseline, the wrapper that reaches it, the parameters the wrapper
#: is configured with, and the same parameters bound to the bare
#: baseline for the weight-identity comparison. Every entry in
#: :data:`stock_rl.baselines.__all__` appears here, and a test asserts
#: that, so a sixth baseline lands as a failure rather than as another
#: strategy that cannot be cross-checked.
WRAPPERS = (
  ('equal_weight', EqualWeightPolicy, {'max_weight': 0.15},
   partial(baselines.equal_weight, max_weight=0.15)),
  ('buy_and_hold', BuyAndHoldPolicy, {'max_weight': 0.15},
   partial(baselines.buy_and_hold, max_weight=0.15)),
  ('buy_and_hold', BuyAndHoldPolicy, {},
   partial(baselines.buy_and_hold)),
  ('low_volatility', InverseVolPolicy, {'window': 40, 'max_weight': 0.15},
   partial(baselines.low_volatility, window=40, max_weight=0.15)),
  ('momentum_ranked', MomentumPolicy,
   {'lookback': 60, 'skip': 5, 'top': 2, 'max_weight': 0.15},
   partial(baselines.momentum_ranked, lookback=60, skip=5, top=2,
           max_weight=0.15)),
  ('trend_filtered_momentum', TrendFilteredMomentumPolicy,
   {'window': 60, 'skip': 5, 'top': 2, 'ma_window': 40, 'max_weight': 0.15},
   partial(baselines.trend_filtered_momentum, window=60, skip=5, top=2,
           ma_window=40, max_weight=0.15)),
)

#: The five metric keys :func:`stock_rl.rl.train.compare` reports from
#: each engine. Both sides must carry exactly these.
METRIC_KEYS = {
  'sharpe', 'max_drawdown', 'total_return', 'turnover', 'total_cost',
}

#: The trend-filtered wrapper, at the parameters the rest of the tests
#: use. Frozen here so a parameter changed to make a test pass cannot
#: quietly change what every other test is measuring.
TREND = {'window': 60, 'skip': 5, 'top': 2, 'ma_window': 40, 'max_weight': 0.15}

START = datetime(2026, 1, 1)


def bars_for(closes):
  '''Return bars one day apart, each opening where the prior one closed.

  Args:
    closes: Sequence of closing prices in ascending time order.

  Returns:
    List of bars.
  '''
  out = []
  for index, close in enumerate(closes):
    opening = closes[index - 1] if index else closes[0]
    out.append(Bar(
      START + timedelta(days=index), opening,
      max(opening, close), min(opening, close), close, 1.0,
    ))
  return out


def make_panels(count=BARS, symbols=SYMBOLS, rise=2.0):
  '''Return aligned panels whose symbols have distinguishable momentum.

  Each symbol gets its own drift and its own alternating component, so
  the ranking baselines disagree with one another rather than all
  choosing the same name. A purely rising panel is deliberate: the
  trend gate has to be exercised on a *falling* one, which is what
  :func:`digging_panels` is for.

  Args:
    count: Number of bars per panel.
    symbols: Symbols to create.
    rise: Bar-on-bar drift, added to a per-symbol base price.

  Returns:
    Mapping of symbol to bars, all sharing one timeline.
  '''
  panels = {}
  for number, symbol in enumerate(symbols):
    closes = [
      50.0 * (number + 1)
      + (rise + number) * index
      + 0.9 * ((-1) ** index) * (number + 1)
      for index in range(count)
    ]
    panels[symbol] = bars_for(closes)
  return panels


def digging_panels(count=130, symbols=SYMBOLS):
  '''Return panels that rally and then break down at the last bar.

  The trend filter is the only baseline whose answer depends on a
  window that is not a momentum parameter, so testing it needs a panel
  where that window decides the outcome. Here the last close sits below
  the 20-bar average and above the 80-bar average, which is a state a
  rising panel cannot reach: on a panel that only rises, every
  ``ma_window`` passes the gate and the parameter is untested.

  Args:
    count: Number of bars per panel.
    symbols: Symbols to create.

  Returns:
    Mapping of symbol to bars, all sharing one timeline.
  '''
  panels = {}
  for number, symbol in enumerate(symbols):
    base = [50.0 * (number + 1) + (3.0 + number) * index
            for index in range(count - 10)]
    peak = base[-1]
    base += [peak - 9.0 * step for step in range(1, 11)]
    panels[symbol] = bars_for(base)
  return panels


def record_baseline(monkeypatch, name):
  '''Replace one baseline with a recorder that still computes correctly.

  Reading the arguments off the call rather than inferring them from an
  equal output is the point: an output comparison passes when the
  wrapper injects a value that happens to equal the baseline's own
  default, and that is the accident these tests exist to rule out.

  Args:
    monkeypatch: The active ``monkeypatch`` fixture.
    name: Attribute name on :mod:`stock_rl.baselines`.

  Returns:
    List that the recorder appends one ``(args, kwargs)`` tuple to per
    call, in call order.
  '''
  seen = []
  real = getattr(baselines, name)

  def recorder(*args, **kwargs):
    '''Record the call, then delegate to the untouched baseline.'''
    seen.append((args, kwargs))
    return real(*args, **kwargs)

  monkeypatch.setattr(baselines, name, recorder)
  return seen


class TestEveryBaselineIsReachable:
  '''The coverage guarantee the gap was about.'''

  def test_a_wrapper_exists_for_every_baseline(self):
    covered = {name for name, _, _, _ in WRAPPERS}
    assert covered == set(baselines.__all__), (
      f'baselines without a policy wrapper: '
      f'{sorted(set(baselines.__all__) - covered)}. Without one, '
      'pipeline.py cannot cross-check the two engines on that '
      'baseline, which is the entire point of wrapping it.')

  @pytest.mark.parametrize(('name', 'policy_class', 'kwargs', 'bare'), WRAPPERS)
  def test_wrapper_is_weight_identical_to_the_baseline(
      self, name, policy_class, kwargs, bare):
    del name
    panels = make_panels()
    policy = policy_class(**kwargs)
    assert policy.act(panels_observation(panels)) == bare(panels), (
      f'{policy_class.__name__}{kwargs} did not reproduce '
      f'{bare.func.__name__}. A wrapper that disagrees with the '
      'baseline it wraps is a second implementation, not a seam.')

  @pytest.mark.parametrize(('name', 'policy_class', 'kwargs', 'bare'), WRAPPERS)
  def test_wrapper_is_weight_identical_on_a_second_panel(
      self, name, policy_class, kwargs, bare):
    del name
    panels = digging_panels()
    policy = policy_class(**kwargs)
    assert policy.act(panels_observation(panels)) == bare(panels)

  @pytest.mark.parametrize(('name', 'policy_class', 'kwargs', 'bare'), WRAPPERS)
  def test_through_portfolio_the_wrapper_trades_the_same_book(
      self, name, policy_class, kwargs, bare):
    del name
    panels = make_panels()
    policy = policy_class(**kwargs)
    through_policy = run_portfolio(
      panels, lambda visible: policy.act(panels_observation(visible)),
      history=60)
    through_baseline = run_portfolio(
      panels, bare, history=60)
    assert through_policy.total_cost == pytest.approx(
      through_baseline.total_cost)
    assert through_policy.turnover == pytest.approx(through_baseline.turnover)
    assert through_policy.sharpe == pytest.approx(through_baseline.sharpe)

  @pytest.mark.parametrize(('name', 'policy_class', 'kwargs', 'bare'), WRAPPERS)
  def test_wrapper_only_names_symbols_it_can_see(
      self, name, policy_class, kwargs, bare):
    del name, bare
    panels = make_panels()
    weights = policy_class(**kwargs).act(panels_observation(panels))
    assert set(weights) <= set(panels)


class TestKeywordPassThrough:
  '''A cap must never arrive in a window parameter.'''

  @pytest.mark.parametrize(('name', 'policy_class', 'kwargs', 'bare'), WRAPPERS)
  def test_wrapper_sends_exactly_its_declared_keywords(
      self, monkeypatch, name, policy_class, kwargs, bare):
    del bare
    seen = record_baseline(monkeypatch, name)
    policy_class(**kwargs).act(panels_observation(make_panels()))
    assert len(seen) == 1, 'the wrapper called the baseline once'
    args, sent = seen[0]
    assert len(args) == 1, (
      'only the panel may be positional; passing parameters '
      'positionally is what put a weight cap in a moving-average window')
    assert sent == kwargs, (
      f'{policy_class.__name__} sent {sent}, but its own fields are '
      f'{kwargs}. A wrapper that renames or reorders an argument on the '
      'way through is the bug this assertion exists for.')

  def test_trend_filter_keeps_the_cap_out_of_the_window(self, monkeypatch):
    seen = record_baseline(monkeypatch, 'trend_filtered_momentum')
    TrendFilteredMomentumPolicy(**TREND).act(panels_observation(make_panels()))
    sent = seen[0][1]
    assert sent['ma_window'] == TREND['ma_window']
    assert sent['max_weight'] == TREND['max_weight']
    assert isinstance(sent['ma_window'], int)
    assert not isinstance(sent['max_weight'], int)

  def test_handing_the_cap_to_the_window_positionally_is_loud(self):
    # The positional mistake, made deliberately. If this ever starts
    # raising something *other* than the window complaint, a cap has
    # found a home it can silently occupy.
    with pytest.raises(ValueError, match='window must be >= 1'):
      baselines.trend_filtered_momentum(
        make_panels(), 60, 5, 2, TREND['max_weight'])
    # And the wrapper, given the same cap, does not trip it.
    weights = TrendFilteredMomentumPolicy(**TREND).act(
      panels_observation(make_panels()))
    assert weights, 'the wrapper returned an empty book'

  @pytest.mark.parametrize('ma_window', (20, 40, 80))
  def test_ma_window_reaches_the_regime_gate(self, ma_window):
    # On a rising panel every window passes, so a wrapper that ignored
    # ma_window entirely would look correct there. The digging panel
    # is where the parameter is observable.
    panels = digging_panels()
    policy = TrendFilteredMomentumPolicy(**{**TREND, 'ma_window': ma_window})
    assert policy.act(panels_observation(panels)) == \
      baselines.trend_filtered_momentum(
        panels, window=TREND['window'], skip=TREND['skip'], top=TREND['top'],
        ma_window=ma_window, max_weight=TREND['max_weight'])

  def test_a_short_window_closes_the_gate_and_a_long_one_opens_it(self):
    panels = digging_panels()
    short = TrendFilteredMomentumPolicy(**{**TREND, 'ma_window': 20})
    long = TrendFilteredMomentumPolicy(**{**TREND, 'ma_window': 80})
    held_short = [w for w in short.act(panels_observation(panels)).values()
                  if w > 0.0]
    held_long = [w for w in long.act(panels_observation(panels)).values()
                 if w > 0.0]
    assert not held_short, (
      'expected the 20-bar gate to reject the broken-down panel')
    assert held_long, (
      'expected the 80-bar gate to still hold, so the two windows '
      'disagree and ma_window is demonstrably load-bearing')

  def test_default_parameters_match_the_baseline_defaults(self, monkeypatch):
    seen = record_baseline(monkeypatch, 'trend_filtered_momentum')
    TrendFilteredMomentumPolicy().act(panels_observation(make_panels()))
    assert seen[0][1] == {
      'window': 252, 'skip': 21, 'top': 10, 'ma_window': 40,
      'max_weight': 0.10,
    }


class TestBuyAndHoldCap:
  '''``max_weight=None`` is a decision, and it has to be a real one.'''

  def test_the_cap_is_absent_by_default(self):
    assert policy_parameters(BuyAndHoldPolicy()) == {'max_weight': None}

  def test_the_default_call_passes_no_cap_at_all(self, monkeypatch):
    seen = record_baseline(monkeypatch, 'buy_and_hold')
    BuyAndHoldPolicy().act(panels_observation(make_panels()))
    assert seen[0][1] == {}, (
      'an uncapped wrapper must add nothing to the call. Injecting the '
      "baseline's own default here would be indistinguishable from "
      'delegating right up until the baseline default moved, at which '
      'point the parity check would silently start testing the wrong '
      'thing.')

  def test_an_explicit_cap_is_forwarded_by_keyword(self, monkeypatch):
    seen = record_baseline(monkeypatch, 'buy_and_hold')
    BuyAndHoldPolicy(max_weight=0.15).act(panels_observation(make_panels()))
    assert seen[0][1] == {'max_weight': 0.15}

  def test_the_default_matches_a_bare_uncapped_call(self):
    panels = make_panels()
    assert BuyAndHoldPolicy().act(panels_observation(panels)) == \
      baselines.buy_and_hold(panels)

  def test_an_explicit_cap_matches_a_bare_capped_call(self):
    panels = make_panels()
    assert BuyAndHoldPolicy(max_weight=0.15).act(
      panels_observation(panels)) == \
      baselines.buy_and_hold(panels, max_weight=0.15)

  def test_the_uncapped_and_capped_books_can_differ(self):
    # Two symbols against a 0.10 default cap: 1/N is 0.5, so the cap is
    # what decides the book, and a wrapper that quietly injected its own
    # default would be indistinguishable from one that delegates.
    panels = make_panels(symbols=('AAA', 'BBB'))
    uncapped = BuyAndHoldPolicy().act(panels_observation(panels))
    capped = BuyAndHoldPolicy(max_weight=0.15).act(panels_observation(panels))
    assert uncapped == {'AAA': 0.10, 'BBB': 0.10}
    assert capped == {'AAA': 0.15, 'BBB': 0.15}

  def test_the_fingerprint_records_the_cap_or_its_absence(self):
    assert policy_fingerprint(BuyAndHoldPolicy()) != \
      policy_fingerprint(BuyAndHoldPolicy(max_weight=0.10)), (
        'two caps on the same passive book are two registrations, and '
        "None versus the baseline's own default is one of them")

  def test_the_two_controls_are_separate_registrations(self):
    assert policy_fingerprint(BuyAndHoldPolicy(max_weight=0.15)) != \
      policy_fingerprint(EqualWeightPolicy(max_weight=0.15)), (
        'buy-and-hold and 1/N are the same book and different '
        'hypotheses, so they must not share a deployment fingerprint')


class TestTrainingMachinery:
  '''The wrappers have to be usable by :mod:`stock_rl.rl.train`.'''

  @pytest.mark.parametrize(('name', 'policy_class', 'kwargs', 'bare'), WRAPPERS)
  def test_each_wrapper_satisfies_the_policy_protocol(
      self, name, policy_class, kwargs, bare):
    del name, bare
    assert isinstance(policy_class(**kwargs), Policy)

  @pytest.mark.parametrize(('name', 'policy_class', 'kwargs', 'bare'), WRAPPERS)
  def test_rollout_drives_each_wrapper_to_completion(
      self, name, policy_class, kwargs, bare):
    del name, bare
    log = rollout_log(policy_class(**kwargs))
    # len() on a TrialLog counts TRIALS, which here is the number of
    # seeds. The step count is per trial. Line 436 below already reads it
    # the right way round; this assertion was counting seeds as steps and
    # so could never pass for a 2-seed run against a 201-bar episode.
    assert all(len(trial.returns) == EPISODE for trial in log), (
      'the environment must be driven to its last bar, otherwise the '
      'wrapper never met the observations it is supposed to read')

  @pytest.mark.parametrize(('name', 'policy_class', 'kwargs', 'bare'), WRAPPERS)
  def test_random_search_accepts_each_wrapper(self, monkeypatch,
                                              name, policy_class, kwargs, bare):
    del name, bare
    policy = policy_class(**kwargs)

    def draw(source):
      '''Hand back the single wrapper under test, ignoring the draw.'''
      del source
      return policy

    monkeypatch.setattr(train, '_draw_policy', draw)
    log = random_search(make_panels(), trials=2, seeds=(1, 2),
                        history=HISTORY)
    assert len(log) == 4, 'two configurations by two seeds'
    assert log.count == 2
    assert all(len(trial.returns) == EPISODE for trial in log)

  def test_random_search_can_mix_wrappers_and_sharpe_report_accepts(self,
                                                                  monkeypatch):
    drawn = [
      MomentumPolicy(lookback=60, skip=5, top=2, max_weight=0.15),
      TrendFilteredMomentumPolicy(**TREND),
      InverseVolPolicy(window=40, max_weight=0.15),
      EqualWeightPolicy(max_weight=0.15),
      BuyAndHoldPolicy(max_weight=0.15),
    ]
    queue = list(drawn)

    def draw(source):
      '''Hand out the queued wrappers in order, ignoring the draw.'''
      del source
      return queue.pop(0)

    monkeypatch.setattr(train, '_draw_policy', draw)
    log = random_search(make_panels(), trials=len(drawn), seeds=(1, 2),
                        history=HISTORY)
    report = sharpe_report(log)
    assert log.count == len(drawn)
    assert report.trials == len(drawn)
    assert report.reason, 'the report must say why it did or did not refuse'
    assert report.available_years == pytest.approx(EPISODE / 252)

  @pytest.mark.parametrize(('name', 'policy_class', 'kwargs', 'bare'), WRAPPERS)
  def test_sharpe_report_accepts_a_log_of_one_wrapper(
      self, name, policy_class, kwargs, bare):
    del name, bare
    log = rollout_log(policy_class(**kwargs))
    report = sharpe_report(log)
    assert report.trials == 1
    assert report.sharpe is None, (
      'one configuration is no search, so no Sharpe may be reported')
    assert 'fewer than two configurations' in report.reason

  @pytest.mark.parametrize(('name', 'policy_class', 'kwargs', 'bare'), WRAPPERS)
  def test_compare_cross_checks_each_wrapper_between_the_engines(
      self, name, policy_class, kwargs, bare):
    del name, bare
    policy = policy_class(**kwargs)
    result = compare(policy, make_panels(), history=HISTORY)
    assert set(result.env) == METRIC_KEYS
    assert set(result.portfolio) == METRIC_KEYS
    assert result.fingerprint == policy_fingerprint(policy)

  def test_the_trend_filter_seam_is_reachable_at_all(self):
    # :mod:`stock_rl.pipeline` could not reach this seam before, because
    # no policy wrapped the function. What the two engines make of it
    # is deliberately NOT asserted beyond the shared keys: the seam's
    # numbers are a property of the engines, either of which may change
    # them, and a wrapper test that pinned them would break on the fix
    # and pass on the bug.
    result = compare(TrendFilteredMomentumPolicy(**TREND), make_panels(),
                     history=HISTORY)
    assert result.fingerprint.startswith('stock_rl.policy.fingerprint/1:')
    assert set(result.env) == set(result.portfolio)


def rollout_log(policy, panels=None, seeds=(1, 2)):
  '''Return a two-seed :class:`TrialLog` for one policy.

  Built through :func:`stock_rl.rl.train.rollout` and
  :meth:`Trial.from_returns` rather than by hand, so the log is the
  shape :func:`sharpe_report` is written to read.

  Args:
    policy: Policy to drive the environment.
    panels: Panels to run on. Defaults to :func:`make_panels`.
    seeds: Seeds each configuration is evaluated under.

  Returns:
    A ``TrialLog`` with ``len(seeds)`` runs of one configuration.
  '''
  chosen = make_panels() if panels is None else panels
  trials = []
  for seed in seeds:
    env = WeightAllocationEnv(chosen, history=HISTORY)
    trials.append(Trial.from_returns(
      'wrapper', seed, rollout(env, policy, seed)))
  return TrialLog(trials)


def sharpe_report(log):
  '''Return the length-gated report for a trial log.

  Args:
    log: Trial log to report on.

  Returns:
    A :class:`stock_rl.rl.train.SharpeReport`.
  '''
  return train.sharpe_report(log)


