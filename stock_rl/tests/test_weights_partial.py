#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''What an OMITTED symbol in a weight mapping means. One semantic, pinned.

**The semantic under protection: a symbol a weight mapping does not name
is a target of ``0.0``, i.e. an EXIT, in both engines.** An empty mapping
is therefore the flat book, not "no opinion".

The two engines used to implement different readings of the same
mapping, and neither said so:

  * :func:`stock_rl.portfolio.run_portfolio` read
    ``raw.get(symbol, 0.0)`` -- omitted meant SELL.
  * :class:`stock_rl.rl.portfolio_env.WeightAllocationEnv` read
    ``action.get(symbol, self._weights[symbol])`` -- omitted meant HOLD.

Both are defensible in isolation, which is exactly why the divergence was
invisible. Measured on one four-symbol panel, the same intended book
through the two engines differed by Rs 5,593 of cost and a full Sharpe
point, because the backtester liquidated the two names a provider forgot
and paid for it while the environment kept them. A number that changes
with the engine that produced it is not a result.

Why exit rather than hold, stated once so a future reader does not have
to re-derive it:

  * Every provider in :mod:`stock_rl.baselines` expresses exclusion by
    omission. ``momentum_ranked`` funds the top N and drops the rest,
    ``trend_filtered_momentum`` gates on a moving average and drops what
    fails, ``low_volatility`` drops what it cannot measure, and the
    top-N funding rule in :mod:`stock_rl.experiment.harness` funds only
    the chosen names. All of them mean "no money in that name".
  * Under "hold" the trend gate becomes unenforceable. A name that falls
    below its moving average is one the strategy has stopped wanting, and
    hold means the book keeps it forever, quietly and for free.
  * A forgotten name is a bug in this codebase, not an intent. Exit makes
    the bug visible: the position disappears, turnover spikes and the cost
    line moves. Hold hides it, and the strategy's own risk gate is the
    first thing to stop working.
  * Exit keeps the strategy layer honest at zero behavioural cost. The
    providers were made to write their zeros explicitly, so every
    published number in the project is unchanged; see the report.

Why not refuse a partial mapping outright: the top-N funding rule in
``experiment.harness`` and ``trend_filtered_momentum``'s regime gate are
*correct* when they omit names, so a refusal would fail on deliberate
code. Refusing after they were fixed buys nothing this test does not
already buy, and would break every caller who legitimately wants "no
opinion expressed by absence".

The adversarial structure matters more than the assertions. The agreement
test carries its own negative control: it perturbs one engine's default
and asserts the engines then DISAGREE, so a future change that breaks
agreement cannot pass by accident, and the agreement assertion cannot be
satisfied by two engines that have both been broken the same way.
'''

# Deliberately white-box: the divergence lives in private helpers
# (_normalise, _target_weights) and in a module-level constant that a
# future change is most likely to edit by hand.
# pylint: disable=protected-access

import inspect
import random
from datetime import datetime, timedelta

import pytest

from stock_rl import baselines
from stock_rl.bars import Bar
from stock_rl.portfolio import _normalise, omitted_weight, run_portfolio
from stock_rl.rl import portfolio_env
from stock_rl.rl.portfolio_env import WeightAllocationEnv

EPOCH = datetime(2021, 1, 1)
HISTORY = 40
REBALANCE = 21
CAPITAL = 1_000_000.0
SYMBOLS = ('AAA', 'BBB', 'CCC', 'DDD')

#: A mapping that names two of four symbols. Under the protected
#: semantic this is a book in AAA and BBB with CCC and DDD at zero.
PARTIAL = {'AAA': 0.3, 'BBB': 0.3}

#: The same intended book, written out completely. Under the protected
#: semantic the two are the same book, which is the whole claim.
COMPLETE = {'AAA': 0.3, 'BBB': 0.3, 'CCC': 0.0, 'DDD': 0.0}

#: A value no provider here would emit, used to perturb one engine's
#: default and prove the agreement assertion is load-bearing.
PERTURBATION = 0.9


def walk(count, seed, drift=0.0004):
  '''Return one price path as bars, opening where the prior one closed.

  Args:
    count: Number of bars.
    seed: Seed for the walk.
    drift: Per-bar drift.

  Returns:
    List of bars in ascending time order.
  '''
  source = random.Random(seed)
  out = []
  price = 100.0
  for index in range(count):
    opened = price
    price *= 1.0 + drift + source.gauss(0.0, 0.012)
    out.append(Bar(timestamp=EPOCH + timedelta(days=index), open=opened,
                   high=max(opened, price), low=min(opened, price),
                   close=price, volume=1.0))
  return out


def panels(symbols=SYMBOLS, count=140):
  '''Return aligned panels with an independent walk per symbol.

  Args:
    symbols: Symbol names.
    count: Bars per panel.

  Returns:
    Mapping of symbol to bars.
  '''
  return {
    symbol: walk(count, index + 1, 0.0004 + 0.0001 * index)
    for index, symbol in enumerate(symbols)
  }


def flat_panel(count=140):
  '''Return a perfectly flat price path, whose volatility is zero.

  Args:
    count: Number of bars.

  Returns:
    List of bars.
  '''
  return [
    Bar(timestamp=EPOCH + timedelta(days=index), open=100.0, high=100.0,
        low=100.0, close=100.0, volume=1.0)
    for index in range(count)]


def scheduled(schedule):
  '''Return a provider choosing its mapping by visible-bar count.

  Keying on the decision bar rather than on a call counter is what makes
  the two engines comparable: ``run_portfolio`` calls the provider only
  on rebalance bars while the environment calls it on every step, so a
  call counter would hand the two engines different books and the test
  would be measuring the harness.

  Args:
    schedule: Mapping of visible-bar count to weight mapping. Bar counts
      not named repeat the last entry at or below them.

  Returns:
    A ``WeightProvider``.
  '''
  counts = sorted(schedule)

  def provider(visible):
    chosen = counts[0]
    for count in counts:
      if count <= len(visible[SYMBOLS[0]]):
        chosen = count
    return dict(schedule[chosen])

  return provider


def both_engines(panel, provider):
  '''Measure one provider with both engines over one shared window.

  The environment is built with the ``rebalance_offset``
  :func:`stock_rl.rl.train.compare` uses, so the two books fill on the
  same bars at the same opens and a cost gap means a book difference
  rather than a calendar difference.

  Args:
    panel: Mapping of symbol to bars.
    provider: The provider under test.

  Returns:
    Tuple of (PortfolioResult, WeightAllocationEnv).
  '''
  result = run_portfolio(panel, provider, capital=CAPITAL, max_weight=1.0,
                         rebalance_days=REBALANCE, history=HISTORY)
  env = WeightAllocationEnv(panel, capital=CAPITAL, max_weight=1.0,
                            history=HISTORY, rebalance_days=REBALANCE,
                            rebalance_offset=(-HISTORY) % REBALANCE)
  env.reset()
  done = False
  while not done:
    visible = {symbol: bars[:env._bar_index() + 1]
               for symbol, bars in env.panels.items()}
    _, _, done, _ = env.step(provider(visible))
  return result, env


def env_target(action):
  '''Return the environment's target weights for one action, cold book.

  Args:
    action: Weight mapping to hand to the environment.

  Returns:
    The target weight per symbol, before any fill.
  '''
  env = WeightAllocationEnv(panels(), history=HISTORY, max_weight=1.0)
  env.reset()
  return env._target_weights(dict(action), CAPITAL)


class TestOmittedMeansExit:
  '''The semantic, asserted on both engines at once.'''

  def test_an_omitted_symbol_is_zero_in_both_engines(self):
    backtester = _normalise(PARTIAL, SYMBOLS, 1.0)
    environment = env_target(PARTIAL)
    assert backtester['CCC'] == 0.0 and environment['CCC'] == 0.0
    assert backtester['DDD'] == 0.0 and environment['DDD'] == 0.0
    assert backtester == environment, (
      f'portfolio._normalise gave {backtester} and the environment gave '
      f'{environment} for the same partial mapping {PARTIAL}, so the two '
      'engines disagree about what an omitted symbol means')

  def test_the_shared_default_is_zero_and_both_engines_read_it(self):
    # Asserted on the constant and not only on behaviour, so a change
    # that moves one engine's default without moving the other fails
    # here instead of surfacing as a cost difference months later.
    # The name lookup is the point: the environment reads the constant
    # out of its own module globals at call time, so this asserts the
    # binding the perturbation below actually exercises.
    assert omitted_weight == 0.0
    assert portfolio_env.omitted_weight == 0.0
    source = inspect.getsource(WeightAllocationEnv._target_weights)
    assert 'action.get(symbol, omitted_weight)' in source, (
      'WeightAllocationEnv._target_weights no longer reads the shared '
      'omitted_weight constant, so this module can no longer be perturbed '
      'to prove the engines agree, and a hard-coded 0.0 would be just as '
      'silent as the hold default it replaced')

  def test_a_partial_mapping_exits_the_names_it_omits(self):
    # A flat start cannot distinguish exit from hold, so the book is
    # loaded first and the partial mapping arrives afterwards. This is
    # the configuration the original divergence was invisible in.
    panel = panels()
    first = HISTORY + 2 + REBALANCE
    provider = scheduled({0: COMPLETE, first: PARTIAL})
    result, env = both_engines(panel, provider)
    assert env.weights['CCC'] == 0.0, (
      'the environment held a position the mapping stopped naming, so a '
      'partial mapping is a hold here and the semantic has been flipped')
    assert result.weights[-1]['CCC'] == 0.0, (
      'the backtester held a position the mapping stopped naming')
    assert env.total_cost > 0.0, (
      'sanity: the exit was a real trade, not a book that never opened')

  def test_the_two_engines_agree_on_a_partial_mapping(self):
    # The positive and the negative control in one test. If the
    # perturbation below did not make the engines disagree, the
    # assertion above would be proving nothing.
    panel = panels()
    first = HISTORY + 2 + REBALANCE
    schedule = {0: COMPLETE, first: PARTIAL}

    agreed, held = both_engines(panel, scheduled(dict(schedule)))
    assert held.total_cost == pytest.approx(agreed.total_cost), (
      f'the engines charged {agreed.total_cost:,.2f} and '
      f'{held.total_cost:,.2f} rupees for one provider book. Two numbers '
      'for one strategy, and the difference lands in the cost line where '
      'it reads as a result rather than as a bug.')
    assert held.annualized_sharpe() == pytest.approx(agreed.sharpe)

    # Negative control. Perturb ONE engine's default and nothing else.
    original = portfolio_env.omitted_weight
    portfolio_env.omitted_weight = PERTURBATION
    try:
      _, broken = both_engines(panel, scheduled(dict(schedule)))
    finally:
      portfolio_env.omitted_weight = original
    assert broken.total_cost != pytest.approx(agreed.total_cost), (
      f'perturbing the environment default to {PERTURBATION} changed '
      f'neither book ({broken.total_cost:,.2f} against '
      f'{agreed.total_cost:,.2f} rupees), so the agreement assertion above '
      'is not testing the default and would survive the defect it exists '
      'to catch')

  def test_a_complete_mapping_is_unaffected(self):
    panel = panels()
    explicit, _ = both_engines(panel, scheduled({0: COMPLETE}))
    written, _ = both_engines(panel, scheduled({0: PARTIAL}))
    assert explicit.total_cost == pytest.approx(written.total_cost), (
      'a complete mapping and the same book with its zeros written out '
      'cost different amounts, so the change is not inert for a provider '
      'that already returned every symbol')
    assert explicit.sharpe == pytest.approx(written.sharpe)
    assert written.weights[0] == dict(COMPLETE)

  def test_an_empty_mapping_is_the_flat_book_in_both_engines(self):
    # Deliberate, not accidental: every provider here can reach an empty
    # or all-zero book (trend_filtered_momentum when its gate rejects
    # everything), and "no opinion expressed by absence" would make that
    # mean "keep the old book", which is the divergence again.
    panel = panels()
    first = HISTORY + 2 + REBALANCE
    result, env = both_engines(panel, scheduled({0: COMPLETE, first: {}}))
    assert set(env.weights) == set(SYMBOLS)
    assert all(weight == 0.0 for weight in env.weights.values()), (
      f'an empty mapping left the book at {env.weights}: absence is being '
      'read as a hold, so a provider that stops naming a name keeps it')
    assert set(result.weights[-1]) == set(SYMBOLS)
    assert all(weight == 0.0 for weight in result.weights[-1].values())
    assert result.total_cost == pytest.approx(env.total_cost)
    assert result.total_cost > 0.0, (
      'sanity: flattening a loaded book is a real trade')

  def test_a_complete_mapping_needs_no_sentinel_from_the_caller(self):
    # The property the providers rely on: writing the zeros out and
    # leaving them out are the same book, so a provider author never has
    # to remember to emit a name in order to exclude it.
    panel = panels()
    sparse, _ = both_engines(panel, scheduled({0: {'AAA': 0.5}}))
    dense, _ = both_engines(
      panel, scheduled({0: {'AAA': 0.5, 'BBB': 0.0, 'CCC': 0.0,
                            'DDD': 0.0}}))
    assert sparse.total_cost == pytest.approx(dense.total_cost)
    assert sparse.sharpe == pytest.approx(dense.sharpe)


class TestProvidersNameEverySymbol:
  '''No provider may still produce a partial mapping.'''

  @pytest.mark.parametrize('name', baselines.__all__)
  def test_a_baseline_names_every_symbol_on_a_normal_panel(self, name):
    panel = panels(count=300)
    mapping = getattr(baselines, name)(panel)
    assert set(mapping) == set(SYMBOLS), (
      f'{name} returned {sorted(mapping)} for {sorted(panel)}, so it is '
      'still producing a partial mapping')

  def test_the_trend_gate_writes_the_names_it_drops(self):
    # trend_filtered_momentum was the only provider that could still
    # omit a name on a healthy panel, which is what made the divergence
    # reachable from shipped code rather than from a hypothetical caller.
    panel = panels(count=300)
    mapping = baselines.trend_filtered_momentum(panel)
    assert set(mapping) == set(SYMBOLS)
    assert 0.0 in mapping.values(), (
      'the trend gate dropped nothing on this panel, so the test proves '
      'nothing about the names it used to omit')

  def test_the_trend_gate_is_all_zeros_when_nothing_qualifies(self):
    # A flat book is the documented answer when the gate rejects
    # everything. It must be complete, or "flat" is indistinguishable
    # from "hold the previous book" under either engine.
    panel = panels(symbols=('AAA',), count=300)
    flat = [Bar(timestamp=EPOCH + timedelta(days=index), open=100.0,
                high=100.0, low=100.0, close=100.0, volume=1.0)
            for index in range(300)]
    panel['AAA'] = flat
    mapping = baselines.trend_filtered_momentum(panel)
    assert mapping == {'AAA': 0.0}

  def test_low_volatility_zeros_a_symbol_it_cannot_measure(self):
    # The second provider that could omit a name: a perfectly flat path
    # has zero realised volatility, so it is excluded from the inverse-
    # volatility weighting and used to fall out of the mapping.
    panel = panels(count=300)
    panel['CCC'] = flat_panel(300)
    mapping = baselines.low_volatility(panel)
    assert set(mapping) == set(SYMBOLS)
    assert mapping['CCC'] == 0.0, (
      f'low_volatility returned {mapping} for a panel with a flat symbol, '
      'so it is still producing a partial mapping')

  def test_no_baseline_returns_an_empty_mapping_on_a_populated_panel(self):
    panel = panels(count=300)
    for name in baselines.__all__:
      mapping = getattr(baselines, name)(panel)
      assert mapping, (
        f'{name} returned an empty mapping for a four-symbol panel, which '
        'the engines would read as the flat book rather than as a refusal')


class TestTheSemanticIsStated:
  '''A silent flip has to fail, and the docs have to agree with it.

  Asserting on the docstring text as well as on the live computation is
  deliberate. A test that only asserts correct behaviour passes while a
  default is still wrong in one engine, because the arithmetic can look
  right on the panel it happens to use.
  '''

  def test_the_backtester_docstring_names_the_semantic(self):
    text = _normalise.__doc__ or ''
    assert 'omitted_weight' in text, (
      'portfolio._normalise no longer documents what an omitted symbol '
      'means, so the next reader has to infer it from the arithmetic')
    assert 'exit' in text

  def test_the_environment_docstring_names_the_semantic(self):
    text = WeightAllocationEnv._target_weights.__doc__ or ''
    assert 'omitted_weight' in text
    assert 'exit' in text
    action_doc = WeightAllocationEnv.step.__doc__ or ''
    assert 'omitted_weight' in action_doc, (
      'WeightAllocationEnv.step is the public surface an actor reads, and '
      'it is where "a partial action is still valid" used to imply hold')

  def test_a_future_default_change_breaks_the_agreement_test(self):
    # The guard on the guard: the semantic is one constant, and this
    # pins it rather than pinning the number it happens to hold today, so
    # a deliberate flip must be made in both engines at once.
    assert portfolio_env.omitted_weight is omitted_weight
    assert not isinstance(omitted_weight, bool), (
      'a bool default would compare equal to 0.0 and 1.0 while reading as '
      'neither')
    assert _normalise({}, SYMBOLS, 1.0) == dict.fromkeys(SYMBOLS, 0.0)
    assert env_target({}) == dict.fromkeys(SYMBOLS, 0.0)
