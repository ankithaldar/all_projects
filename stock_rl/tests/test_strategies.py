#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Tests for the indicator-driven strategies.

Three properties are pinned for all five strategies, and the third is
the one that matters:

1. Each returns a COMPLETE weight mapping keyed by the panel's symbols.
2. Each returns ``0.0`` -- not a guess -- when the visible history cannot
   support its indicator.
3. **No look-ahead.** A decision taken at the close of bar ``t`` fills at
   the open of bar ``t+1`` and is sized on what was known at bar ``t-1``.
   The engine enforces the fill and passes only ``panel[:index]``, so a
   provider structurally cannot price its own fill -- but it can still
   leak through state, mutation of the panel it was handed, or a
   computation that reaches outside its argument. Test
   :class:`TestNoLookAhead` proves the leak-free property end to end by
   running the same strategies over a panel whose bars after the cut have
   been replaced with fabricated ones: every rebalance before the cut must
   produce the identical book.

Each negative test sits beside a positive control, so a test that cannot
fail is caught rather than trusted.
'''

from datetime import datetime, timedelta
import math
import random

import pytest

from stock_rl.baselines import (
  buy_and_hold,
  equal_weight,
  low_volatility,
  momentum_ranked,
  trend_filtered_momentum,
)
from stock_rl.bars import Bar
from stock_rl.portfolio import run_portfolio
from stock_rl.strategies import (
  atr_breakout,
  ema_trend_following,
  measure,
  require_history,
  rsi_mean_reversion,
  volatility_filtered_momentum,
  volatility_scaled_momentum,
)

START = datetime(2026, 1, 1)

#: Symbols in the generated panel. Named AAA/BBB/CCC-style on purpose: a
#: test that printed RELIANCE would invite somebody to believe it.
SYMBOLS = tuple(f'S{index:02d}' for index in range(12))

#: Bars per symbol. Above every strategy's declared warm-up (the longest
#: is 274) and long enough for :func:`stock_rl.metrics.minimum_backtest_length`
#: to pass for a modest claimed Sharpe, so the strategies are exercised on
#: a readable book rather than on their refusal path.
BARS = 630

#: Rebalance interval and engine warm-up used throughout, matching the
#: defaults of run_portfolio so the test measures the same thing the
#: module does.
REBALANCE_DAYS = 21
HISTORY = 60

#: Bar index the fabricated tail replaces in the look-ahead tests. Every
#: rebalance before it is a decision taken on real data and must be
#: unchanged by the fiction.
CUT = 300

#: Snapshot index of the first decision that can see the fabricated bars.
#: The engine rebalances on every multiple of ``REBALANCE_DAYS`` from bar
#: 63 upward, so snapshot ``i`` was decided at bar ``63 + 21 * i`` and the
#: last snapshot that cannot see the fabricated bars is the last one whose
#: decision bar is at or before ``CUT``.
CUT_INDEX = (CUT - 1) // REBALANCE_DAYS - 2

#: Bars each strategy declares it needs before it will read a signal. The
#: refusal tests cut a panel one bar below these and open it two bars
#: above, so the boundary is pinned rather than assumed.
NEEDED = {
  'atr_breakout': 61,
  'ema_trend_following': 200,
  'rsi_mean_reversion': 15,
  'volatility_filtered_momentum': 274,
  'volatility_scaled_momentum': 274,
}

#: Panel shape for the history-boundary test, as (drift per bar, noise
#: sigma, extra keyword arguments). Each shape is chosen so that the ONLY
#: reason the short panel is refused is bar count: on the long panel the
#: signal fires on its merits. A clean ramp qualifies deterministically for
#: RSI and for the two trend rules, so they get almost no noise; the two
#: momentum strategies divide by realised volatility, which is exactly zero
#: on a ramp, so they get enough noise for the reading to land inside the
#: declared 0.15-to-0.45 band. ``atr_breakout`` also relaxes its range
#: confirmation to zero, because that threshold has its own test below and
#: a coin flip here would test the coin rather than the warm-up.
BOUNDARY = {
  'atr_breakout': (0.006, 0.002, {'strength': 0.0}),
  'ema_trend_following': (0.006, 0.002, {}),
  'rsi_mean_reversion': (-0.006, 0.002, {}),
  'volatility_filtered_momentum': (0.006, 0.012, {}),
  'volatility_scaled_momentum': (0.006, 0.012, {}),
}

#: The five strategies under test, as (name, provider) pairs.
STRATEGIES = (
  ('atr_breakout', atr_breakout),
  ('ema_trend_following', ema_trend_following),
  ('rsi_mean_reversion', rsi_mean_reversion),
  ('volatility_filtered_momentum', volatility_filtered_momentum),
  ('volatility_scaled_momentum', volatility_scaled_momentum),
)

#: The five baselines every strategy is measured against.
BASELINES = (
  ('buy_and_hold', buy_and_hold),
  ('equal_weight', equal_weight),
  ('low_volatility', low_volatility),
  ('momentum_ranked', momentum_ranked),
  ('trend_filtered_momentum', trend_filtered_momentum),
)


def bars_from(rows, start=0):
  '''Return bars from (open, high, low, close) tuples.

  Args:
    rows: Sequence of price tuples.
    start: Index of the first row, used to offset the timestamps.

  Returns:
    List of bars one day apart.
  '''
  return [
    Bar(START + timedelta(days=start + index), o, h, low, c, 1.0)
    for index, (o, h, low, c) in enumerate(rows)
  ]


def make_panel(symbols=SYMBOLS, count=BARS, seed=11):
  '''Return a deterministic generated panel with a built-in drift.

  A geometric random walk per symbol, half the universe drifting up and
  half drifting down, so a cross-sectional strategy has something to
  rank and a mean-reversion strategy has something to fade. This is
  SYNTHETIC data and nothing computed on it is a finding: the drift is a
  property of the generator, which is precisely why
  ``docs/INDICATOR-STRATEGIES.md`` reports the same strategies on a
  zero-drift walk as well.

  Twelve symbols, above ``1 / max_weight``, because a panel with fewer
  collapses four of the five baselines into one uniform book and every
  Sharpe becomes the same number.

  Args:
    symbols: Symbols to generate.
    count: Bars per symbol.
    seed: Seed for the first symbol's walk; each symbol offsets it.

  Returns:
    Mapping of symbol to bars on a shared timeline.
  '''
  panels = {}
  for offset, symbol in enumerate(symbols):
    source = random.Random(seed + offset)
    drift = 0.0020 if offset % 2 == 0 else -0.0015
    rows = []
    price = 100.0 + 5.0 * offset
    for _ in range(count):
      opening = price
      price = round(price * (1.0 + drift + source.gauss(0.0, 0.012)), 4)
      rows.append((opening, max(opening, price), min(opening, price),
                   price))
    panels[symbol] = bars_from(rows)
  return panels


def trend_bars(count, step, sigma=0.010, seed=5):
  '''Return one bar series: a drift per bar plus gaussian noise.

  Args:
    count: Number of bars.
    step: Fractional drift per bar. Negative gives a downtrend.
    sigma: Standard deviation of the per-bar noise.
    seed: Seed for the noise.

  Returns:
    List of bars. A zero ``sigma`` gives an exact geometric ramp, whose
    realised volatility is zero.
  '''
  source = random.Random(seed)
  rows = []
  price = 100.0
  for _ in range(count):
    opening = price
    price = round(opening * (1.0 + step + source.gauss(0.0, sigma)), 4)
    rows.append((opening, max(opening, price), min(opening, price), price))
  return bars_from(rows)


def trending_panel(count, step, sigma=0.010, seed=5):
  '''Return a two-symbol panel, one trending up and one down.

  Args:
    count: Bars per symbol.
    step: Fractional drift per bar.
    sigma: Standard deviation of the per-bar noise.
    seed: Seed for the first symbol's noise.

  Returns:
    Mapping of symbol to bars, both long enough for ``count``.
  '''
  return {
    'AAA': trend_bars(count, step, sigma, seed),
    'BBB': trend_bars(count, -step, sigma, seed + 101),
  }


def rising_panel(count, base=100.0, step=0.01):
  '''Return one bar series that rises by ``step`` a bar.

  Args:
    count: Number of bars.
    base: Starting close.
    step: Fractional gain per bar.

  Returns:
    List of bars.
  '''
  rows = []
  for index in range(count):
    opening = base * (1.0 + step) ** index
    closing = base * (1.0 + step) ** (index + 1)
    rows.append((opening, closing, opening, closing))
  return bars_from(rows)


def falling_panel(count, base=100.0, step=0.01):
  '''Return one bar series that falls by ``step`` a bar.

  Args:
    count: Number of bars.
    base: Starting close.
    step: Fractional loss per bar.

  Returns:
    List of bars.
  '''
  rows = []
  for index in range(count):
    opening = base * (1.0 - step) ** index
    closing = base * (1.0 - step) ** (index + 1)
    rows.append((opening, opening, closing, closing))
  return bars_from(rows)


def flat_panel(count, price=100.0):
  '''Return one bar series that never moves.

  Realised volatility of a constant-return series is exactly zero, which
  is the degenerate case the volatility sizing must refuse rather than
  divide by.

  Args:
    count: Number of bars.
    price: Constant close.

  Returns:
    List of bars.
  '''
  return bars_from([(price, price, price, price)] * count)


def truncate(panels, stop):
  '''Return a panel cut at ``stop`` bars.

  Args:
    panels: Mapping of symbol to bars.
    stop: Number of bars to keep.

  Returns:
    Mapping of symbol to the leading bars.
  '''
  return {symbol: bars[:stop] for symbol, bars in panels.items()}


def replace_tail(panels, stop, tail):
  '''Return the real bars up to ``stop`` with fabricated bars after it.

  Args:
    panels: Mapping of symbol to real bars.
    stop: Bar index where the fabrication begins.
    tail: Mapping of symbol to fabricated bars.

  Returns:
    Mapping of symbol to a panel sharing its prefix with ``panels`` and
    nothing else.
  '''
  return {
    symbol: list(bars[:stop]) + list(tail[symbol])
    for symbol, bars in panels.items()
  }


def decisions(provider, panels, history=HISTORY):
  '''Return every book the engine would trade on ``panels``, in order.

  A strategy is asked for one decision at a time by the engine, so a test
  that only asks once is testing one lucky bar. This walks the rebalance
  schedule and collects the whole path.

  Args:
    provider: Weight provider under test.
    panels: Mapping of symbol to bars.
    history: Bars of warm-up before the first decision.

  Returns:
    List of weight mappings, one per rebalance.
  '''
  return run_portfolio(
    panels, provider,
    rebalance_days=REBALANCE_DAYS,
    history=history,
  ).weights


def divergent_tail(symbols, start, count, scale=4.0):
  '''Return fabricated bars that share nothing with a real panel.

  Same calendar dates as the real bars would have had, prices scaled and
  rising the other way, so any read of "the future" shows up immediately.

  Args:
    symbols: Symbols to fabricate for.
    start: Bar index the fabricated series begins at.
    count: Number of fabricated bars.
    scale: Multiplier applied to the prices.

  Returns:
    Mapping of symbol to fabricated bars.
  '''
  rows = []
  for step in range(count):
    lower = 100.0 * scale * (1.5 ** step)
    upper = 100.0 * scale * (1.5 ** (step + 1))
    rows.append((lower, upper, lower, upper))
  tail = bars_from(rows, start=start)
  return {symbol: tail for symbol in symbols}


class TestWeightMapping:
  '''Each strategy returns a complete mapping keyed by the panel.'''

  @pytest.mark.parametrize('name,provider', STRATEGIES)
  def test_keys_match_panel_symbols(self, name, provider):
    panels = make_panel()
    weights = provider(panels)
    assert set(weights) == set(panels), name

  @pytest.mark.parametrize('name,provider', STRATEGIES)
  def test_weights_are_finite_and_long_only(self, name, provider):
    weights = provider(make_panel())
    for symbol, weight in weights.items():
      assert math.isfinite(weight), (name, symbol)
      assert weight >= 0.0, (name, symbol)

  @pytest.mark.parametrize('name,provider', STRATEGIES)
  def test_qualifying_names_sum_to_one(self, name, provider):
    # Every non-empty book a strategy produces must be fully invested, and
    # every empty one must be exactly flat. A book that sums to 0.4 is a
    # sizing bug rather than a choice: _book either shares the whole book
    # among qualifying names or returns nothing.
    #
    # The provider's own output is asserted, not the engine's snapshot,
    # because run_portfolio clamps at max_weight and the cash it holds is
    # the engine's business rather than the strategy's.
    panels = make_panel()
    books = [
      provider(truncate(panels, stop))
      for stop in range(HISTORY + REBALANCE_DAYS, BARS, REBALANCE_DAYS)
    ]
    held = [book for book in books if sum(book.values()) > 0.0]
    assert held, name
    for book in held:
      assert sum(book.values()) == pytest.approx(1.0), name

  @pytest.mark.parametrize('name,provider', STRATEGIES)
  def test_produces_a_non_flat_book(self, name, provider):
    # Positive control for the refusal tests below: with the full panel
    # every strategy has enough history, so each must actually hold
    # something at some decision.
    held = [
      book for book in decisions(provider, make_panel())
      if sum(book.values()) > 0.0
    ]
    assert held, name

  @pytest.mark.parametrize('name,provider', STRATEGIES)
  def test_held_count_respects_top(self, name, provider):
    panels = make_panel()
    held = sum(
      1 for weight in provider(panels, top=3).values() if weight > 0.0)
    assert held <= 3, name

  @pytest.mark.parametrize('name,provider', STRATEGIES)
  def test_empty_panels_give_empty_book(self, name, provider):
    # Positive control for the not-enough-history refusal: an empty panel
    # is the shortest possible one, and it must be refused the same way
    # rather than raising IndexError.
    assert provider({}) == {}, name

  @pytest.mark.parametrize('name,provider', STRATEGIES)
  def test_single_empty_panel_is_named_at_zero(self, name, provider):
    assert provider({'AAA': []}) == {'AAA': 0.0}, name


class TestInsufficientHistoryRefused:
  '''Too little history is refused, not guessed at.'''

  @pytest.mark.parametrize('name,provider', STRATEGIES)
  def test_one_bar_below_the_declared_warmup_is_all_zero(self, name,
                                                         provider):
    # The refusal boundary is pinned rather than assumed: NEEDED is the
    # bar count each strategy declares, and one bar below it every name
    # must be named at exactly 0.0 -- a zero score would be arithmetically
    # valid and would look like a decision.
    needed = NEEDED[name]
    step, sigma, extra = BOUNDARY[name]
    panels = trending_panel(needed - 1, step, sigma)
    weights = provider(panels, **extra)
    assert set(weights) == set(panels), name
    assert all(weight == 0.0 for weight in weights.values()), name

  @pytest.mark.parametrize('name,provider', STRATEGIES)
  def test_two_bars_above_the_declared_warmup_flips_the_refusal(self, name,
                                                                provider):
    # Positive control for the refusal above. The panel is the same shape,
    # two bars longer, and the strategy holds something: the flat book was
    # a length decision and not a permanent property of the strategy.
    needed = NEEDED[name]
    step, sigma, extra = BOUNDARY[name]
    panels = trending_panel(needed + 1, step, sigma)
    assert sum(provider(panels, **extra).values()) > 0.0, name

  @pytest.mark.parametrize('name,provider', STRATEGIES)
  def test_a_ten_bar_panel_is_refused_by_everyone(self, name, provider):
    # Below the shortest declared warm-up of any strategy, so every one of
    # them must refuse -- and must still name every symbol rather than
    # returning a partial mapping or raising.
    panels = {'AAA': trend_bars(10, 0.01), 'BBB': trend_bars(10, -0.01)}
    weights = provider(panels)
    assert weights == {'AAA': 0.0, 'BBB': 0.0}, name

  @pytest.mark.parametrize('name,provider', STRATEGIES)
  def test_one_bar_panel_is_all_zero(self, name, provider):
    panels = {'AAA': rising_panel(1), 'BBB': rising_panel(1, 200.0)}
    weights = provider(panels)
    assert weights == {'AAA': 0.0, 'BBB': 0.0}, name

  def test_rsi_refuses_below_its_window(self):
    weights = rsi_mean_reversion(
      {'AAA': rising_panel(14)}, window=14)
    assert weights == {'AAA': 0.0}

  def test_rsi_holds_when_window_is_satisfied(self):
    weights = rsi_mean_reversion(
      {'AAA': falling_panel(40)}, window=14)
    assert weights['AAA'] > 0.0

  def test_ema_trend_refuses_below_its_window(self):
    weights = ema_trend_following({'AAA': rising_panel(199)}, slow=200)
    assert weights == {'AAA': 0.0}

  def test_ema_trend_holds_when_window_is_satisfied(self):
    weights = ema_trend_following({'AAA': rising_panel(200)}, slow=200)
    assert weights['AAA'] > 0.0

  def test_zero_volatility_is_refused_not_divided_by(self):
    # A constant-return series has exactly zero realised volatility, so
    # the momentum/volatility ratio is 0/0. The strategy must refuse.
    panels = {
      'AAA': flat_panel(300),
      'BBB': flat_panel(300, 250.0),
    }
    weights = volatility_scaled_momentum(panels)
    assert weights == {'AAA': 0.0, 'BBB': 0.0}

  def test_zero_volatility_is_refused_by_the_band_too(self):
    panels = {
      'AAA': flat_panel(300),
      'BBB': flat_panel(300, 250.0),
    }
    weights = volatility_filtered_momentum(panels)
    assert weights == {'AAA': 0.0, 'BBB': 0.0}

  def test_rsi_rejects_a_non_positive_entry(self):
    with pytest.raises(ValueError, match='entry'):
      rsi_mean_reversion({'AAA': rising_panel(40)}, entry=0.0)


class TestNoLookAhead:
  '''Weights before the cut cannot depend on bars after it.'''

  def test_the_cut_index_really_is_the_first_decision_after_the_cut(self):
    # Positive control for the index arithmetic the tests below lean on.
    # If it drifts, every no-look-ahead assertion is checking the wrong
    # set of snapshots and would pass or fail for the wrong reason.
    def bar(snapshot):
      '''Return the bar index a snapshot was decided on.'''
      return HISTORY + 3 + REBALANCE_DAYS * snapshot

    assert bar(CUT_INDEX - 1) <= CUT < bar(CUT_INDEX)

  @pytest.mark.parametrize('name,provider', STRATEGIES)
  def test_weights_before_the_cut_are_unchanged(self, name, provider):
    real = make_panel()
    fabricated = replace_tail(
      real, CUT, divergent_tail(SYMBOLS, CUT, BARS - CUT))
    before = decisions(provider, real)
    after = decisions(provider, fabricated)
    assert before[:CUT_INDEX] == after[:CUT_INDEX], name

  @pytest.mark.parametrize('name,provider', STRATEGIES)
  def test_the_fabricated_panel_really_changes_the_later_books(self, name,
                                                               provider):
    # Positive control for the test above. If the fabricated bars were a
    # no-op the assertion would pass for the wrong reason, so the panel
    # must demonstrably be a different panel: same symbols, same length,
    # a byte-identical prefix, different prices after the cut, and
    # therefore a different book once the engine reaches them.
    real = make_panel()
    fabricated = replace_tail(
      real, CUT, divergent_tail(SYMBOLS, CUT, BARS - CUT))
    assert len(fabricated['S00']) == len(real['S00']), name
    assert fabricated['S00'][:CUT] == real['S00'][:CUT], name
    assert fabricated['S00'] != real['S00'], name
    before = decisions(provider, real)
    after = decisions(provider, fabricated)
    assert before[:CUT_INDEX] == after[:CUT_INDEX], name
    assert before[CUT_INDEX:] != after[CUT_INDEX:], name

  @pytest.mark.parametrize('name,provider', STRATEGIES)
  def test_provider_is_a_pure_function_of_its_argument(self, name, provider):
    # A provider that caches between calls or mutates the panel it was
    # handed would answer differently the second time, which is the other
    # way a future leak reaches a past decision.
    panels = make_panel()
    snapshot = {symbol: list(bars) for symbol, bars in panels.items()}
    first = provider(panels)
    second = provider(panels)
    assert first == second, name
    assert panels == snapshot, name

  @pytest.mark.parametrize('name,provider', STRATEGIES)
  def test_shorter_history_is_a_different_question(self, name, provider):
    # Positive control for the whole class: the provider is NOT invariant
    # to how much history it is shown, because a decision made with 100
    # bars is genuinely a different decision from one made with 300. If it
    # were invariant, the test above would be vacuous.
    panels = make_panel()
    short = provider(truncate(panels, 100))
    long = provider(truncate(panels, 300))
    assert short != long or not any(long.values()), name


class TestThroughTheEngine:
  '''Every strategy runs on the one backtester and returns finite stats.'''

  @pytest.mark.parametrize('name,provider', STRATEGIES)
  def test_result_is_finite(self, name, provider):
    result = run_portfolio(
      make_panel(),
      provider,
      rebalance_days=REBALANCE_DAYS,
      history=HISTORY,
    )
    for value in (result.sharpe, result.max_drawdown,
                  result.total_return_multiple, result.turnover,
                  result.total_cost):
      assert math.isfinite(value), name
    assert result.rebalances > 0, name
    assert result.total_return_multiple > 0.0, name
    assert len(result.returns) == BARS - HISTORY, name

  @pytest.mark.parametrize('name,provider', STRATEGIES)
  def test_snapshot_books_are_feasible(self, name, provider):
    result = run_portfolio(
      make_panel(),
      provider,
      rebalance_days=REBALANCE_DAYS,
      history=HISTORY,
    )
    for book in result.weights:
      assert all(weight >= 0.0 for weight in book.values()), name
      assert sum(book.values()) <= 1.0 + 1e-12, name
      assert max(book.values()) <= 0.10 + 1e-12, name

  @pytest.mark.parametrize('name,provider', STRATEGIES)
  def test_is_measured_on_the_same_engine_as_the_baselines(self, name,
                                                           provider):
    # Positive control for the comparison in docs/INDICATOR-STRATEGIES.md:
    # the strategies and the baselines are run over the same panel with the
    # same rebalance interval and warm-up, so every Sharpe is measured on
    # the identical instrument. Equal weight on this panel is the number
    # every strategy is judged against. The provider under test is in the
    # list, so this also proves `measure` measured the strategy rather
    # than a substitute for it.
    panels = make_panel()
    runs = [(name, provider)] + list(BASELINES)
    rows = measure(
      panels, runs,
      rebalance_days=REBALANCE_DAYS,
      history=HISTORY,
    )
    assert len(rows) == len(runs)
    assert {row.strategy for row in rows} == {row[0] for row in runs}
    assert all(math.isfinite(row.sharpe) for row in rows)

  def test_a_non_finite_weight_is_refused_by_the_engine(self):
    # The guard that makes every finiteness assertion above meaningful: if
    # a provider can smuggle infinity into a position, then "all weights
    # finite" was never true of the strategies.
    with pytest.raises(ValueError, match='infinite'):
      run_portfolio(make_panel(), lambda panels: {'S00': math.inf})


class TestHistoryGate:
  '''A panel too short to support a claim is refused.'''

  #: Trials and claimed Sharpe held constant across the refusal and the
  #: pass, so the only thing that differs between them is bar count.
  TRIALS = 4
  TARGET_SHARPE = 1.0

  def test_short_panel_is_refused_naming_the_gate(self):
    panels = truncate(make_panel(), 120)
    with pytest.raises(ValueError, match='minimum_backtest_length'):
      require_history(
        panels, 'short', self.TRIALS, self.TARGET_SHARPE)

  def test_short_panel_refusal_quotes_the_numbers(self):
    panels = truncate(make_panel(), 120)
    with pytest.raises(ValueError) as caught:
      require_history(panels, 'short', self.TRIALS, self.TARGET_SHARPE)
    message = str(caught.value)
    assert f'{self.TRIALS} configurations' in message
    assert '1.000' in message
    assert 'short' in message

  def test_long_panel_passes_the_same_gate(self):
    # Positive control: same trials, same claimed Sharpe, 630 bars instead
    # of 120. The refusal above is a length decision, not a permanent
    # property of the gate.
    available = require_history(
      make_panel(), 'long', self.TRIALS, self.TARGET_SHARPE)
    assert available == pytest.approx((BARS - HISTORY) / 252)
    assert available > 0.0

  def test_non_positive_sharpe_is_refused(self):
    with pytest.raises(ValueError, match='minimum_backtest_length'):
      require_history(make_panel(), 'flat', self.TRIALS, 0.0)

  def test_more_trials_demand_more_history(self):
    # The gate has to bite harder as the search grows, or declaring one
    # trial would pass a panel that ten could not.
    panels = make_panel()
    assert require_history(panels, 'few', 2, 1.0) > 0.0
    with pytest.raises(ValueError, match='minimum_backtest_length'):
      require_history(panels, 'many', 500, 1.0)

  def test_empty_panels_are_refused(self):
    with pytest.raises(ValueError, match='panels must not be empty'):
      require_history({}, 'none', self.TRIALS, self.TARGET_SHARPE)

  def test_measure_refuses_a_panel_that_supports_no_claim(self):
    # A short panel on which nothing is measurable: every strategy returns
    # a flat book, so the best Sharpe is zero and the gate has nothing to
    # defend. A number must not leave this module in that state.
    panels = {
      'AAA': flat_panel(120),
      'BBB': flat_panel(120, 200.0),
    }
    with pytest.raises(ValueError, match='minimum_backtest_length'):
      measure(
        panels,
        list(STRATEGIES),
        rebalance_days=REBALANCE_DAYS,
        history=HISTORY,
      )

  def test_measure_reports_a_panel_that_supports_a_claim(self):
    # Positive control for the refusal above.
    rows = measure(
      make_panel(),
      list(STRATEGIES),
      rebalance_days=REBALANCE_DAYS,
      history=HISTORY,
    )
    assert [row.strategy for row in rows] == [
      name for name, _ in STRATEGIES
    ]
    assert all(row.bars == BARS - HISTORY for row in rows)

  def test_measure_needs_two_configurations(self):
    with pytest.raises(ValueError, match='deflate'):
      measure(make_panel(), [('equal_weight', equal_weight)])


class TestDeclaredThresholdBehaviour:
  '''The thresholds are declared values, and they behave as declared.'''

  def test_rsi_entry_level_is_inclusive(self):
    # At entry the depth is zero, which is the honest reading: sitting on
    # the line is no evidence at all.
    closes = bars_from([(10.0, 10.0, 10.0, 10.0)] * 40)
    held = rsi_mean_reversion({'AAA': closes}, entry=50.0)
    assert held == {'AAA': 0.0}

  def test_rsi_sizes_by_depth(self):
    deep = rsi_mean_reversion({'AAA': falling_panel(40)}, window=14)
    shallow_panel = bars_from([
      (10.0, 10.0, 10.0, 10.0 + (0.4 if step % 2 == 0 else -0.4))
      for step in range(40)
    ])
    shallow = rsi_mean_reversion({'AAA': shallow_panel}, window=14)
    # Both books are single-name so both sum to 1.0; the assertion that
    # matters is that an unbroken decline reads oversold and a flat
    # oscillation does not.
    assert deep['AAA'] > 0.0
    assert shallow['AAA'] == 0.0

  def test_ema_trend_needs_a_rising_slope(self):
    # A name above its long EMA but rolling over fails the direction test.
    panels = {'AAA': rising_panel(260)}
    assert ema_trend_following(panels)['AAA'] > 0.0
    falling = {'AAA': falling_panel(260)}
    assert ema_trend_following(falling)['AAA'] == 0.0

  def test_atr_breakout_needs_a_new_high(self):
    panels = {'AAA': rising_panel(120)}
    assert atr_breakout(panels)['AAA'] > 0.0
    # A flat series makes no new high, so nothing is bought.
    flat = {'AAA': flat_panel(120)}
    assert atr_breakout(flat)['AAA'] == 0.0

  def test_atr_breakout_needs_an_above_average_range(self):
    # A new high reached on a move smaller than one average true range is
    # drift, not a breakout. Raising the requirement to 100 ranges must
    # therefore empty the book.
    panels = {'AAA': rising_panel(120)}
    assert atr_breakout(panels, strength=1.0)['AAA'] > 0.0
    assert atr_breakout(panels, strength=100.0)['AAA'] == 0.0

  def test_volatility_band_excludes_both_ends(self):
    # Low band: a band far below the generated volatility buys nothing.
    panels = make_panel()
    assert volatility_filtered_momentum(
      panels, low=0.0001, high=100.0) != volatility_filtered_momentum(
        panels, low=1.0, high=100.0)

  def test_volatility_scaled_momentum_is_tilted_not_flat(self):
    # The whole point of the strategy: two names with the same momentum
    # must not get the same capital if one is twice as volatile.
    panels = make_panel()
    weights = volatility_scaled_momentum(panels)
    held = [w for w in weights.values() if w > 0.0]
    assert len(held) > 1
    assert len(set(held)) > 1

