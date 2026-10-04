#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the three-arm context experiment harness.

Every test here is written to fail if the logic were wrong, not to
confirm that a function runs. The specific things that would be easy to
get wrong and expensive to get wrong are:

* the control arm's state drifting from the price block, which would
  make "the arms differ only by context" false;
* two arms quietly measured on different parameters, which would turn
  the comparison into two unrelated backtests;
* a failing kill criterion being softened into a near-miss somewhere in
  the reporting path;
* a verdict that a caller can supply or that disagrees with
  ``KillCriteriaResult.passed``;
* seed dispersion reported as zero when the seeds genuinely differ;
* a reading timestamped after the decision bar reaching a state vector.

The panels here are synthetic. A verdict computed on generated data is
a statement about this harness, never about the context layer.
'''

from __future__ import annotations

import json
import random
# The statistics helpers are private on purpose -- they are the module's
# own arithmetic, not an API -- and the degenerate cases below (a
# zero-variance paired difference, a flat benchmark, a too-short series)
# are only reachable by calling them directly.
# pylint: disable=protected-access

from datetime import datetime, timedelta, timezone
from statistics import fmean, stdev

import pytest

from stock_rl.bars import Bar
from stock_rl.context import CONTEXT_WIDTH, SymbolContext
from stock_rl.context.fuse import KillCriteria, evaluate_kill_criteria
from stock_rl.costs import CostModel
from stock_rl.experiment import harness
from stock_rl.experiment import (
  Arm,
  ArmConfig,
  ArmRun,
  ArmSummary,
  Comparison,
  ExperimentReport,
  LookAheadError,
  Verdict,
  arm_state,
  benchmark_returns,
  build_contexts,
  compare_arms,
  linear_scorer,
  price_features,
  price_width,
  run_arm,
  run_experiment,
)
from stock_rl.graph import nifty50_seed
from stock_rl.sentiment import SentimentReading

IST = timezone(timedelta(hours=5, minutes=30))
START = datetime(2026, 1, 1, 15, 30, tzinfo=IST)
BEFORE = START - timedelta(hours=6)
LATER = START + timedelta(days=1)
SYMBOLS = ('AAA', 'BBB', 'CCC')
FREE = CostModel(stt_buy=0.0, stt_sell=0.0, exchange_pct=0.0, sebi_pct=0.0,
                 stamp_duty_buy=0.0, gst_pct=0.0, dp_charge=0.0,
                 slippage_bps=0.0)


def panel(closes: list[float]) -> list[Bar]:
  '''Return bars one day apart, each opening where the prior one closed.

  Args:
    closes: Closing prices in ascending order.

  Returns:
    Bars with naive-at-call but timezone-aware timestamps, so the
      look-ahead machinery sees real aware datetimes.
  '''
  out: list[Bar] = []
  for index, close in enumerate(closes):
    opening = closes[index - 1] if index else closes[0]
    out.append(Bar(
      START + timedelta(days=index),
      opening,
      max(opening, close),
      min(opening, close),
      close,
      1.0,
    ))
  return out


def synth_panels(count: int = 700, symbols: tuple[str, ...] = SYMBOLS,
                 seed: int = 3) -> dict[str, list[Bar]]:
  '''Return generated panels with per-symbol drift and noise.

  Args:
    count: Bars per symbol.
    symbols: Symbols to generate.
    seed: Seed for the price generator.

  Returns:
    Mapping of symbol to bars on a shared timeline.
  '''
  panels: dict[str, list[Bar]] = {}
  for index, symbol in enumerate(symbols):
    source = random.Random(seed + index)
    closes: list[float] = []
    price = 100.0 + 10.0 * index
    for _ in range(count):
      price *= 1.0 + source.gauss(0.0004, 0.013)
      closes.append(round(price, 4))
    panels[symbol] = panel(closes)
  return panels


def readings(symbols: tuple[str, ...] = SYMBOLS, count: int = 700,
             available: datetime = BEFORE) -> list[SentimentReading]:
  '''Return three-source readings every three bars.

  Three sources rather than two because
  :data:`stock_rl.sentiment.default_min_sources` is two, and a
  single-source aggregate is refused as unusable -- which is correct
  behaviour but would make every context arm's block all zeros and the
  comparison vacuous.

  Args:
    symbols: Symbols to write readings for.
    count: Bars per symbol the readings span.
    available: Timestamp stamped on every reading. Move this past the
      decision bar to build a leak.

  Returns:
    Readings for every symbol, spread over the panel.
  '''
  source = random.Random(19)
  out: list[SentimentReading] = []
  for index, symbol in enumerate(symbols):
    for bar in range(0, count, 3):
      base = source.gauss(-0.5 + 0.5 * index, 0.35)
      for name in ('reuters', 'ndtv', 'mint'):
        out.append(SentimentReading(
          symbol,
          max(-1.0, min(1.0, source.gauss(base, 0.15))),
          'news',
          0.6,
          START + timedelta(days=bar) + (available - BEFORE),
          name,
        ))
  return out


def config(**overrides) -> ArmConfig:
  '''Return a synthetic-panel configuration.

  Args:
    **overrides: Fields to replace.

  Returns:
    A config naming the panels synthetic, with costs zeroed so a
      turnover difference cannot be blamed on the cost model.
  '''
  base = {
    'costs': FREE,
    'history': 60,
    'holdings': 2,
    'max_weight': 0.5,
    'n_folds': 5,
    'min_train': 60,
    'seeds': (1, 2, 3),
    'panel_kind': 'synthetic',
  }
  base.update(overrides)
  return ArmConfig(**base)


def scalars(symbols: tuple[str, ...] = SYMBOLS) -> dict[str, float]:
  '''Return one injected LLM scalar per symbol.

  Args:
    symbols: Symbols to cover.

  Returns:
    Distinct per-symbol scalars, so arm C is not arm B with a constant
      bolted on.
  '''
  return {symbol: 0.4 - 0.3 * index
          for index, symbol in enumerate(symbols)}


def hand_context(symbol: str = 'AAA') -> SymbolContext:
  '''Return a context with hand-checkable field values.

  Args:
    symbol: Symbol the context describes.

  Returns:
    A context whose numbers a reader can verify by eye, so the
      control-arm byte-identity test compares against arithmetic rather
      than against another function call.
  '''
  return SymbolContext(
    symbol=symbol,
    sentiment_score=0.4,
    sentiment_disagreement=0.1,
    sentiment_intensity=0.6,
    sentiment_usable=1.0,
    sentiment_source_ratio=0.6,
    graph_depth=2.0,
    commodity_dependencies=1.0,
    macro_dependencies=1.0,
    sector_dependencies=2.0,
    options_pcr=1.1,
    options_iv_rank=0.7,
    options_net_oi_change=0.05,
  )


def test_control_arm_state_is_the_price_block_byte_for_byte() -> None:
  '''The control arm must reproduce the hand-computed price state.

  The price block is spelled out in the test rather than passed through
  :func:`stock_rl.experiment.price_features`, so a bug that changed
  ``price_features`` would be caught here rather than being confirmed
  by both sides of the comparison moving together.
  '''
  closes = [100.0 + index for index in range(30)]
  bars = panel(closes)
  computed = price_features(bars)
  assert len(computed) == price_width
  hand = (closes[-1] / closes[-21] - 1.0, computed[1], computed[2])
  assert computed == pytest.approx(hand, abs=1e-12)

  control = arm_state(Arm.CONTROL, computed)
  assert control == tuple(computed)
  assert len(control) == price_width


def test_control_state_is_identical_whether_or_not_context_exists() -> None:
  '''Passing a context to the control arm must not change its state.

  This is the "byte-identical, not merely close" requirement. A control
  that appended zeros would pass an equality test on the first
  ``price_width`` elements and would quietly be a wider state.
  '''
  price = (0.11, 0.22, 0.33)
  without = arm_state(Arm.CONTROL, price)
  with_context = arm_state(Arm.CONTROL, price, hand_context())
  assert with_context == without
  assert with_context == (0.11, 0.22, 0.33)
  assert len(with_context) == len(without)


def test_context_arms_prefix_the_control_state_exactly() -> None:
  '''Arms B and C must keep the price block unchanged and in order.

  Tuple equality, not approximate: a context arm that rescaled or
  reordered the price block would change the control's contribution and
  the experiment would be measuring two edits at once.
  '''
  price = (0.05, 0.5, 1.0)
  context = hand_context()
  control = arm_state(Arm.CONTROL, price)
  for arm, scalar in ((Arm.CONTEXT, None),
                      (Arm.PRICE_CONTEXT, 0.25)):
    state = arm_state(arm, price, context, scalar)
    assert state[:price_width] == control
    assert state[:price_width] == price


def test_context_block_is_shared_between_arm_b_and_arm_c() -> None:
  '''Arm C must not carry a second, independently fused context block.

  Two fusions of the same readings produce the same values by accident,
  so the test pins the identity by construction: C's context block is
  B's, byte for byte, and the only difference is the trailing scalar.
  '''
  price = (0.05, 0.5, 1.0)
  context = hand_context()
  arm_b = arm_state(Arm.CONTEXT, price, context)
  arm_c = arm_state(Arm.PRICE_CONTEXT, price, context, 0.25)
  block_b = arm_b[price_width:price_width + CONTEXT_WIDTH]
  block_c = arm_c[price_width:price_width + CONTEXT_WIDTH]
  assert block_b == block_c
  assert len(arm_c) == len(arm_b) + 1
  assert arm_c[:price_width] == arm_b[:price_width]


def test_price_features_of_an_empty_panel_are_all_zero() -> None:
  '''No history must give zeroed features of the right width.

  The width must not collapse to zero, because the state vector's width
  is what every saved policy was built against.
  '''
  assert price_features([]) == (0.0,) * price_width
  assert len(price_features([])) == price_width


def test_wrong_price_width_is_refused() -> None:
  '''A short price block must raise, not silently widen the state.

  Without this the control arm would return a 2-float state and the
  saved-policy-width argument in the context module would be false.
  '''
  with pytest.raises(ValueError, match='price block must be'):
    arm_state(Arm.CONTROL, (0.1, 0.2))


def test_linear_scorer_truncates_and_hand_computes() -> None:
  '''The scorer must equal the plain dot product of its inputs.

  Args:
    None: No arguments; the inputs are fixed below.
  '''
  state = (1.0, -2.0, 3.0)
  assert linear_scorer(state, (0.5, 0.25, 2.0)) == pytest.approx(
    0.5 * 1.0 + 0.25 * -2.0 + 2.0 * 3.0)
  assert linear_scorer(state, (1.0,)) == pytest.approx(1.0)
  assert linear_scorer((), (1.0, 2.0)) == 0.0


def test_context_arm_needs_a_context_and_price_context_needs_a_scalar(
) -> None:
  '''Arm B without context and arm C without a scalar must both refuse.

  Defaulting the missing piece would make arm C byte-identical to arm B
  and answer the LLM question by omission.
  '''
  with pytest.raises(ValueError, match='needs a SymbolContext'):
    arm_state(Arm.CONTEXT, (0.1, 0.2, 0.3))
  with pytest.raises(ValueError, match='llm_scalar'):
    arm_state(Arm.PRICE_CONTEXT, (0.1, 0.2, 0.3), hand_context())


def test_build_contexts_rejects_a_reading_from_the_future() -> None:
  '''A reading public after the decision bar must raise, not be dropped.

  The leak is the whole reason the harness exists as more than a
  backtest: a silently dropped reading still produces a Sharpe, and that
  Sharpe is built on data no trader could have had.
  '''
  leaked = SentimentReading('AAA', 0.9, 'news', 0.9, LATER, 'reuters')
  public = [
    SentimentReading('AAA', 0.2, 'news', 0.5, BEFORE, 'ndtv'),
    SentimentReading('AAA', 0.4, 'news', 0.5, BEFORE, 'mint'),
  ]
  with pytest.raises(LookAheadError, match='not public at'):
    build_contexts(('AAA',), [*public, leaked], START)
  kept = build_contexts(('AAA',), public, START)
  assert set(kept) == {'AAA'}
  assert kept['AAA'].sentiment_usable == 1.0
  assert kept['AAA'].sentiment_score == pytest.approx(0.3)


def test_build_contexts_rejects_a_naive_decision_bar() -> None:
  '''A naive decision bar must raise rather than compare timestamps.

  A naive-versus-aware comparison raises ``TypeError`` from deep inside
  the backtest otherwise, and the field at issue is the one that defines
  look-ahead.
  '''
  with pytest.raises(ValueError, match='timezone-aware'):
    build_contexts(('AAA',), [], datetime(2026, 1, 1, 15, 30))


def test_build_contexts_needs_a_shock_when_given_a_graph() -> None:
  '''A graph without a shock key must raise rather than traverse zero.

  Depth 0 for every symbol would look like a legitimate "nothing is
  downstream" reading when it actually means nobody said where to look.
  '''
  with pytest.raises(ValueError, match='shock node key'):
    build_contexts(('AAA',), [], START, graph=nifty50_seed())


def test_build_contexts_uses_graph_depth() -> None:
  '''A reachable symbol must carry its hop count as graph depth.

  Args:
    None: No arguments; the panel and shock are fixed below.
  '''
  graph = nifty50_seed()
  contexts = build_contexts(('RELIANCE',), [], START, graph=graph,
                            shock='commodity:crude')
  assert contexts['RELIANCE'].graph_depth == 1.0
  absent = build_contexts(('ZZZZNOTLISTED',), [], START, graph=graph,
                          shock='commodity:crude')
  assert absent['ZZZZNOTLISTED'].graph_depth == 0.0


def test_run_arm_does_not_leak_readings_past_the_decision_bar() -> None:
  '''A run must not see a reading published after the decision bar.

  The full reading history is handed to :func:`run_arm`, including
  readings stamped after the panel ends, so the per-bar selection is the
  only thing standing between the run and the future.
  '''
  panels = synth_panels(count=400)
  book = readings(count=400)
  future = [SentimentReading(symbol, 1.0, 'news', 1.0, LATER, name)
            for symbol in SYMBOLS for name in ('reuters', 'ndtv', 'mint')]
  with_future = run_arm(Arm.CONTEXT, panels, config(), 1,
                        readings=[*book, *future])
  without = run_arm(Arm.CONTEXT, panels, config(), 1, readings=book)
  assert with_future.returns == without.returns
  # A leak would also have saturated sentiment_score at +1.0 for every
  # symbol, which no real reading set does.
  context = build_contexts(
    SYMBOLS,
    [item for item in book if item.available_from <= START + timedelta(
      days=3)],
    START + timedelta(days=3),
  )
  assert min(item.sentiment_score for item in context.values()) < 0.9


def test_run_arm_stores_the_shared_config_verbatim() -> None:
  '''Every run must record the exact config it was measured under.

  Args:
    None: No arguments; the config is fixed below.
  '''
  shared = config()
  panels = synth_panels(count=300)
  run = run_arm(Arm.CONTROL, panels, shared, 1)
  assert run.config is shared
  assert run.config.costs is shared.costs


def test_price_context_arm_requires_an_injected_scalar() -> None:
  '''Arm C must refuse to run without its injected scalar.

  Args:
    None: No arguments; the panel is fixed below.
  '''
  panels = synth_panels(count=300)
  with pytest.raises(ValueError, match='injected llm_scalar'):
    run_arm(Arm.PRICE_CONTEXT, panels, config(), 1, readings=readings(
      count=300))


def test_benchmark_is_equal_weight_on_the_same_parameters() -> None:
  '''The benchmark must face the same costs, caps and warm-up.

  Args:
    None: No arguments; the panel is fixed below.
  '''
  panels = synth_panels(count=400)
  shared = config()
  benchmark = benchmark_returns(panels, shared)
  control = run_arm(Arm.CONTROL, panels, shared, 1)
  assert len(benchmark) == len(control.returns)


def test_compare_arms_rejects_mismatched_seeds() -> None:
  '''Comparing different seed sets must raise.

  A comparison that silently paired seed 1 of one arm with seed 2 of
  another would report a difference created by the seed draw.
  '''
  panels = synth_panels(count=300)
  shared = config()
  control = [run_arm(Arm.CONTROL, panels, shared, seed) for seed in (1, 2)]
  treatment = [run_arm(Arm.CONTEXT, panels, shared, 9) for _ in (1, 2)]
  benchmark = benchmark_returns(panels, shared)
  with pytest.raises(ValueError, match='seeds differ'):
    compare_arms(control, treatment, benchmark, shared)


def test_compare_arms_rejects_a_short_benchmark() -> None:
  '''A benchmark shorter than the arms must raise.

  Truncating it instead would silently compare against a different
  market window than the one the arms traded.
  '''
  panels = synth_panels(count=300)
  shared = config()
  control = [run_arm(Arm.CONTROL, panels, shared, seed) for seed in (1, 2)]
  treatment = [run_arm(Arm.CONTROL, panels, shared, seed) for seed in (1, 2)]
  with pytest.raises(ValueError, match='benchmark has'):
    compare_arms(control, treatment, [0.0, 0.1], shared)


def test_compare_arms_rejects_an_empty_arm_list() -> None:
  '''An empty control list must raise rather than divide by zero.

  Args:
    None: No arguments; the panel is fixed below.
  '''
  panels = synth_panels(count=300)
  shared = config()
  run = [run_arm(Arm.CONTROL, panels, shared, 1)]
  with pytest.raises(ValueError, match='same non-zero seeds'):
    compare_arms([], run, [0.0] * 200, shared)


def test_identical_arms_compare_to_exactly_zero() -> None:
  '''An arm compared against itself must produce exactly zero deltas.

  This is the null test that makes every non-zero number meaningful. It
  would fail if the paired statistics were computed on the wrong series,
  or if a small non-zero floor were quietly added anywhere.
  '''
  panels = synth_panels(count=400)
  shared = config()
  runs = [run_arm(Arm.CONTROL, panels, shared, seed) for seed in (1, 2, 3)]
  benchmark = benchmark_returns(panels, shared)
  result = compare_arms(runs, runs, benchmark, shared)
  assert result.return_delta_mean == 0.0
  assert result.return_delta_stdev == 0.0
  assert result.sharpe_delta_mean == 0.0
  assert result.sharpe_delta_stdev == 0.0
  assert result.alpha_mean == 0.0
  assert result.abs_t == 0.0
  assert result.folds_passed == 0


def test_multi_seed_stdev_is_non_zero_when_seeds_differ() -> None:
  '''Seed dispersion must be a measurement, not a constant.

  Grądzki (2026) measures a Sharpe standard deviation of 0.193 on a
  mean of 0.604, so a harness that reports 0.0 for differing seeds is
  reporting that it did not vary anything -- which is exactly the
  failure this asserts against.
  '''
  panels = synth_panels(count=500)
  shared = config()
  per_seed = {
    seed: run_arm(Arm.CONTROL, panels, shared, seed).sharpe
    for seed in (1, 2, 3, 4)
  }
  assert len(set(per_seed.values())) > 1
  book = readings(count=500)
  report = run_experiment(
    panels,
    readings=book,
    llm_scalars=scalars(),
    config=shared,
  )
  injected = scalars()
  for summary in report.arms:
    assert summary.seeds == shared.seeds
    # The mean and stdev must be those of *that* arm's own seeds. An
    # arm that quietly reported the control's Sharpe would show zero
    # dispersion here, and a summary that reported the best seed instead
    # of the mean would miss this equality.
    runs = [run_arm(summary.arm, panels, shared, seed,
                    readings=book, llm_scalars=injected)
            for seed in summary.seeds]
    sharpes = [run.sharpe for run in runs]
    assert len(set(sharpes)) > 1
    assert summary.sharpe_mean == pytest.approx(fmean(sharpes))
    assert summary.sharpe_stdev == pytest.approx(stdev(sharpes))
    assert summary.sharpe_mean < max(sharpes)


def test_report_refuses_arms_that_disagree_on_parameters() -> None:
  '''A report must not survive arms run under different configurations.

  Two arms measured with different cost models or cadences is not a weak
  comparison, it is not a comparison, and the cheapest place to say so
  is before anything is reported.
  '''
  panels = synth_panels(count=300)
  shared = config()
  summaries = tuple(ArmSummary(arm=arm, seeds=shared.seeds,
                               config=shared, mean_returns=[0.0])
                    for arm in Arm)
  assert len(panels) == len(SYMBOLS)
  ExperimentReport(config=shared, arms=summaries)
  # Same seeds, different rebalance cadence: the arm names and the seed
  # set both look right, so only a config comparison catches this.
  mismatched = config(rebalance_days=7)
  bad = [summary if summary.arm is not Arm.CONTEXT else
         ArmSummary(arm=summary.arm, seeds=shared.seeds,
                    config=mismatched, mean_returns=[0.0])
         for summary in summaries]
  with pytest.raises(ValueError, match='different config'):
    ExperimentReport(config=shared, arms=tuple(bad))
  # A different cost model must be caught the same way.
  other_costs = config(costs=CostModel(slippage_bps=20.0))
  bad_costs = [summary if summary.arm is not Arm.PRICE_CONTEXT else
               ArmSummary(arm=summary.arm, seeds=shared.seeds,
                          config=other_costs, mean_returns=[0.0])
               for summary in summaries]
  with pytest.raises(ValueError, match='different config'):
    ExperimentReport(config=shared, arms=tuple(bad_costs))
  # A different seed set is caught too.
  bad_seeds = list(summaries)
  bad_seeds[0] = ArmSummary(arm=Arm.CONTROL, seeds=(7, 8, 9), config=shared,
                            mean_returns=[0.0])
  with pytest.raises(ValueError, match='ran under seeds'):
    ExperimentReport(config=shared, arms=tuple(bad_seeds))


def test_report_refuses_a_missing_arm_and_duplicate_seeds() -> None:
  '''The report must insist on exactly three arms and distinct seeds.

  A missing arm would leave a comparison with no control, and duplicate
  seeds would inflate the seed count that the dispersion is computed
  over.
  '''
  shared = config()
  summaries = tuple(ArmSummary(arm=arm, seeds=shared.seeds, config=shared)
                    for arm in Arm)
  ExperimentReport(config=shared, arms=summaries)
  with pytest.raises(ValueError, match='one summary per arm'):
    ExperimentReport(config=shared, arms=summaries[:2])
  duplicated = config(seeds=(1, 1, 2))
  with pytest.raises(ValueError, match='duplicate seeds'):
    ExperimentReport(config=duplicated, arms=summaries)


def test_report_refuses_an_unknown_arm_set() -> None:
  '''An arm list that is not exactly the three must be rejected.

  Args:
    None: No arguments; the config is fixed below.
  '''
  shared = config()
  summaries = (ArmSummary(arm=Arm.CONTROL, seeds=shared.seeds),
               ArmSummary(arm=Arm.CONTROL, seeds=shared.seeds),
               ArmSummary(arm=Arm.CONTROL, seeds=shared.seeds))
  with pytest.raises(ValueError, match='expected arms'):
    ExperimentReport(config=shared, arms=summaries)


def test_verdict_follows_passed_and_cannot_be_supplied() -> None:
  '''The verdict must be KEEP iff passed, with no caller override.

  A harness that accepts a verdict argument is a harness whose verdict
  can be a preference, and the whole point of pre-registering the
  thresholds is that the result cannot be edited after the fact.
  '''
  passed = evaluate_kill_criteria(0.01, 0.01, 4.0, 5, 0.1)
  failed = evaluate_kill_criteria(0.9, 0.01, 4.0, 5, 0.1)
  assert passed.passed and failed.passed is False
  shared = config()
  summaries = tuple(ArmSummary(arm=arm, seeds=shared.seeds, config=shared)
                    for arm in Arm)
  assert ExperimentReport(config=shared, arms=summaries,
                          kill=passed).verdict is Verdict.KEEP
  assert ExperimentReport(config=shared, arms=summaries,
                          kill=failed).verdict is Verdict.KILL
  assert ExperimentReport(config=shared, arms=summaries,
                          kill=None).verdict is Verdict.INCONCLUSIVE
  assert 'verdict' not in ExperimentReport.__slots__


def test_a_single_failing_threshold_fails_the_whole_verdict() -> None:
  '''One unmet threshold must be enough to produce a KILL.

  This is the "mostly passed" test. Four of five criteria clearing is
  not a pass; it is four failures the report must list.
  '''
  panels = synth_panels(count=700)
  shared = config()
  report = run_experiment(
    panels,
    readings=readings(count=700),
    llm_scalars=scalars(),
    config=shared,
  )
  assert report.kill is not None
  assert report.verdict is Verdict.KILL
  assert report.kill.passed is False
  assert report.kill.failures
  payload = json.loads(report.to_json())
  assert payload['failures'] == list(report.kill.failures)
  assert payload['kill_passed'] is False
  # Four of the five criteria cleared and the fifth did not. That is a
  # fail, and every unmet threshold is listed rather than counted.
  near_miss = evaluate_kill_criteria(
    alpha_pvalue=0.01,
    sharpe_pvalue=0.01,
    abs_t=3.5,
    folds_passed=5,
    monthly_one_sided_turnover=0.9,
    criteria=KillCriteria(),
  )
  assert near_miss.passed is False
  assert len(near_miss.failures) == 1
  assert 'turnover' in near_miss.failures[0]
  # A flat null effect fails all five.
  null_result = evaluate_kill_criteria(
    alpha_pvalue=0.9,
    sharpe_pvalue=0.9,
    abs_t=0.4,
    folds_passed=0,
    monthly_one_sided_turnover=0.9,
    criteria=KillCriteria(),
  )
  assert null_result.passed is False
  assert len(null_result.failures) == 5


def test_failures_are_surfaced_verbatim() -> None:
  '''The report's failure strings must be the criteria module's own.

  A paraphrase is how a near-miss becomes "mostly passed", so the test
  compares the report's JSON against
  :func:`stock_rl.context.evaluate_kill_criteria` output character for
  character.
  '''
  panels = synth_panels(count=300)
  shared = config()
  report = run_experiment(
    panels,
    readings=readings(count=300),
    llm_scalars=scalars(),
    config=shared,
  )
  summaries = tuple(ArmSummary(arm=arm, seeds=shared.seeds)
                    for arm in Arm)
  verdict_failures = ('alpha p=0.900 > 0.1: delete all numeric context work',)
  manual = evaluate_kill_criteria(
    alpha_pvalue=0.9,
    sharpe_pvalue=0.01,
    abs_t=4.0,
    folds_passed=5,
    monthly_one_sided_turnover=0.1,
  )
  assert manual.failures == verdict_failures
  stamped = ExperimentReport(config=shared, arms=summaries, kill=manual)
  payload = json.loads(stamped.to_json())
  assert payload['failures'] == list(manual.failures)
  assert payload['verdict'] == 'kill'
  if report.kill is not None:
    assert payload['failures'] == list(report.kill.failures)


def test_inconclusive_verdict_when_history_is_too_short() -> None:
  '''Too little history must yield INCONCLUSIVE, not a confident verdict.

  The gate is
  :func:`stock_rl.metrics.minimum_backtest_length`, the same one
  :mod:`stock_rl.rl.train` gates reporting on. Reporting a Sharpe the
  sample cannot support is worse than reporting nothing.
  '''
  # 200 bars is 0.67 years of returns and the claimed Sharpe is under
  # 0.1, for which
  # :func:`stock_rl.metrics.minimum_backtest_length` demands a
  # double-digit number of years. The gate must refuse rather than
  # evaluate thresholds against a Sharpe it has just said is
  # indefensible.
  panels = synth_panels(count=200, symbols=('AAA', 'BBB'), seed=3)
  shared = config(seeds=(1, 2, 3, 4, 5), min_train=30, n_folds=2,
                  history=30, holdings=1)
  report = run_experiment(
    panels,
    readings=readings(('AAA', 'BBB'), count=200),
    llm_scalars={'AAA': 0.2, 'BBB': -0.3},
    config=shared,
  )
  assert report.kill is None
  assert report.verdict is Verdict.INCONCLUSIVE
  assert report.gate_reason
  assert report.required_years > report.available_years
  assert 'kill criterion was evaluated' in report.gate_reason
  payload = json.loads(report.to_json())
  assert payload['kill_passed'] is None
  assert payload['failures'] is None
  assert payload['verdict'] == 'inconclusive'
  # The measurements are still published; only the verdict is withheld.
  assert payload['comparisons']


def test_run_experiment_rejects_empty_panels() -> None:
  '''No panels must raise rather than report a vacuous null result.

  Args:
    None: No arguments; the input is fixed below.
  '''
  with pytest.raises(ValueError, match='no panels'):
    run_experiment({}, readings=[], llm_scalars={}, config=config())


def test_report_json_is_byte_stable_across_identical_runs() -> None:
  '''Two identical runs must serialise to identical bytes.

  The report exists to be diffed between runs. If it were not
  byte-stable, a verdict change and floating-point jitter would be
  indistinguishable, which defeats the purpose of a machine-readable
  artefact.
  '''
  panels = synth_panels(count=400)
  shared = config()
  book = readings(count=400)
  injected = scalars()
  first = run_experiment(panels, readings=book, llm_scalars=injected,
                         config=shared)
  second = run_experiment(panels, readings=book, llm_scalars=injected,
                          config=shared)
  assert first.to_json() == second.to_json()
  assert first.to_json().endswith('\n')
  payload = json.loads(first.to_json())
  assert payload['panel_kind'] == 'synthetic'
  assert [item['label'] for item in payload['comparisons']] == [
    'context_minus_a', 'price_context_minus_a']
  assert [arm['arm'] for arm in payload['arms']] == [
    'control', 'context', 'price_context']
  assert payload['verdict'] in {'keep', 'kill', 'inconclusive'}
  assert 'folds_passed' in payload['comparisons'][0]
  assert 'alpha_pvalue' in payload['criteria_inputs']
  assert payload['config']['costs']['stt_buy'] == 0.0


def test_experiment_report_carries_both_comparisons_and_all_arms() -> None:
  '''The report must contain A, B, C and both contrasts.

  An experiment that reported only B minus A would hide arm C entirely,
  and arm C is the arm the roadmap is about to spend money on.
  '''
  panels = synth_panels(count=400)
  shared = config()
  report = run_experiment(
    panels,
    readings=readings(count=400),
    llm_scalars=scalars(),
    config=shared,
  )
  assert {summary.arm for summary in report.arms} == set(Arm)
  labels = [item.label for item in report.comparisons]
  assert labels == ['context_minus_a', 'price_context_minus_a']
  for comparison in report.comparisons:
    assert isinstance(comparison, Comparison)
    assert comparison.folds_total == shared.n_folds
    assert comparison.seeds == shared.seeds
  assert set(report.criteria_inputs) == {
    'alpha_pvalue', 'sharpe_pvalue', 'abs_t', 'folds_passed',
    'monthly_one_sided_turnover'}
  assert report.panel_kind == 'synthetic'


def test_arms_are_measured_on_identical_parameters() -> None:
  '''Every arm's run must carry the identical config object.

  The harness constructs one config and hands the same object to all
  three arms, so identity is available and is what the report asserts.
  A copy per arm would still compare equal, but only equality of values
  would be checked, and a mutable default such as ``costs`` could then
  differ without notice.
  '''
  panels = synth_panels(count=300)
  shared = config()
  runs = {arm: [run_arm(arm, panels, shared, seed,
                        readings=readings(count=300), llm_scalars=scalars())
                for seed in shared.seeds]
          for arm in Arm}
  identities = {id(run.config) for group in runs.values() for run in group}
  assert identities == {id(shared)}
  assert all(len(run.returns) == len(panels['AAA']) - shared.history
             for group in runs.values() for run in group)


def test_naive_panel_timestamps_are_localised_before_the_context_path() -> None:
  '''A naive bar timestamp must be localised, not compared naive.

  The sentiment module refuses a naive decision bar outright, so a panel
  built from naive timestamps would otherwise abort every context arm
  while the control arm ran fine -- the arms would not be comparable and
  the failure would look like a data problem rather than a clock one.
  '''
  naive: dict[str, list[Bar]] = {}
  for symbol in SYMBOLS:
    bars = synth_panels(count=300, symbols=(symbol,))[symbol]
    naive[symbol] = [
      Bar(bar.timestamp.replace(tzinfo=None), bar.open, bar.high, bar.low,
          bar.close, bar.volume)
      for bar in bars
    ]
  assert naive['AAA'][0].timestamp.tzinfo is None
  run = run_arm(Arm.CONTEXT, naive, config(), 1,
                readings=readings(count=300))
  assert len(run.returns) > 0


def test_a_mis_shaped_state_is_refused(monkeypatch) -> None:
  '''A fused state of the wrong length must raise, not be used anyway.

  The three guards in :func:`arm_state` are the only thing standing
  between a future edit to the context module and a silently
  double-counted arm C. They are exercised here by handing
  :func:`arm_state` a state vector that violates each post-condition in
  turn, because a guard that is never tripped is a comment.
  '''
  price = (0.1, 0.2, 0.3)
  context = hand_context()

  def truncated(*_args, **_kwargs):
    '''Return a control state missing its last element.

    Args:
      *_args: Ignored.
      **_kwargs: Ignored.

    Returns:
      A state one float short of the control width.
    '''
    return (0.1, 0.2)

  monkeypatch.setattr(harness, 'state_vector', truncated)
  with pytest.raises(ValueError, match='produced 2 floats, expected 3'):
    arm_state(Arm.CONTROL, price)

  def rescaled(*_args, **_kwargs):
    '''Return a context state whose price block has been rescaled.

    Args:
      *_args: Ignored.
      **_kwargs: Ignored.

    Returns:
      A full-width state with the price block doubled.
    '''
    return (0.2, 0.4, 0.6) + (0.0,) * 12

  monkeypatch.setattr(harness, 'state_vector', rescaled)
  with pytest.raises(ValueError, match='altered the price block'):
    arm_state(Arm.CONTEXT, price, context)

  def poisoned(*_args, **_kwargs):
    '''Return a context state carrying NaN in the context block.

    Args:
      *_args: Ignored.
      **_kwargs: Ignored.

    Returns:
      A full-width state with NaN where the sentiment score sits.
    '''
    return (0.1, 0.2, 0.3) + (float('nan'),) + (0.0,) * 11

  monkeypatch.setattr(harness, 'state_vector', poisoned)
  with pytest.raises(ValueError, match='NaN in the context block'):
    arm_state(Arm.CONTEXT, price, context)


def test_equal_weight_baseline_handles_an_empty_cross_section() -> None:
  '''An empty cross-section must give no weights rather than dividing by 0.

  Args:
    None: No arguments; the input is fixed below.
  '''
  assert not harness._equal_weight({})
  assert harness._equal_weight({'AAA': [], 'BBB': []}) == {
    'AAA': 0.5, 'BBB': 0.5}


def test_seed_averaging_refuses_ragged_series() -> None:
  '''Averaging seed series of different lengths must raise.

  Zero-padding the short one would dilute the mean and change the
  Sharpe of every arm in the report without changing any of the runs.

  Args:
    None: No arguments; the inputs are fixed below.
  '''
  assert harness._mean_series([[0.0, 1.0], [2.0, 3.0]]) == [1.0, 2.0]
  with pytest.raises(ValueError, match='differ in length'):
    harness._mean_series([[0.0, 1.0], [2.0]])


def test_degenerate_statistics_do_not_pass_the_thresholds() -> None:
  '''Too-short, zero-dispersion and flat inputs must give 0.0, not blow up.

  Every degenerate case returns zero rather than infinity. Infinity would
  pass ``|t| > 3`` and a raised exception would abort the whole report,
  so both failure modes here would be worse than the honest zero.

  Args:
    None: No arguments; the inputs are fixed below.
  '''
  assert harness._paired_t([0.1]) == 0.0
  assert harness._paired_t([0.1, 0.1, 0.1]) == 0.0
  assert harness._paired_t([0.1, -0.1]) == pytest.approx(0.0, abs=1e-12)
  assert harness._paired_t([0.01] * 10 + [0.02] * 10) > 0.0
  assert harness._capm_alpha([0.1, 0.2], [0.1, 0.2]) == (0.0, 0.0)
  assert harness._capm_alpha([0.1, 0.2, 0.3],
                             [0.0, 0.0, 0.0]) == (0.0, 0.0)
  assert harness._capm_alpha([0.1, 0.2, 0.3], [1.0, 2.0]) == (0.0, 0.0)


def test_summarising_an_empty_arm_list_yields_an_empty_summary() -> None:
  '''An arm with no runs must summarise to nothing, not to a zero Sharpe.

  The report then rejects it for missing arms, which is the right layer
  to catch it: a fabricated 0.0 Sharpe would look like a measurement.

  Args:
    None: No arguments; the arm is fixed below.
  '''
  empty = harness._summarise(Arm.CONTROL, [])
  assert empty.arm is Arm.CONTROL
  assert not empty.seeds
  assert not empty.mean_returns


def test_arms_reporting_different_windows_are_refused() -> None:
  '''Two arms with different return counts must raise, not be truncated.

  Padding the shorter series with zeros would compare a 300-bar arm
  against a 200-bar arm and attribute the difference in exposure to the
  context vector.
  '''
  shared = config()
  short = ArmRun(arm=Arm.CONTROL, seed=1, config=shared,
                 returns=[0.01] * 300, sharpe=1.0)
  long_run = ArmRun(arm=Arm.CONTEXT, seed=1, config=shared,
                    returns=[0.01] * 400, sharpe=1.0)
  with pytest.raises(ValueError, match='different windows'):
    compare_arms([short], [long_run], [0.0] * 400, shared)


def test_a_flat_benchmark_makes_the_alpha_regression_undefined() -> None:
  '''A flat benchmark must yield zero alpha, not a division by zero.

  A constant market series has no variance, so beta is undefined. The
  harness must report 0.0 rather than raise mid-report or invent a beta.
  '''
  shared = config()
  panels = synth_panels(count=300)
  control = [run_arm(Arm.CONTROL, panels, shared, seed) for seed in (1, 2)]
  treatment = [run_arm(Arm.CONTROL, panels, shared, seed) for seed in (1, 2)]
  flat = [0.0] * len(control[0].returns)
  result = compare_arms(control, treatment, flat, shared)
  assert result.alpha_mean == 0.0
  assert result.alpha_stdev == 0.0


def test_a_non_finite_statistic_is_refused_by_the_report() -> None:
  '''A NaN Sharpe must raise rather than serialise as ``NaN``.

  ``NaN`` is not valid JSON. A report that emitted it would not parse,
  and the artefact exists to be machine-read and diffed.
  '''
  shared = config()
  summaries = tuple(
    ArmSummary(arm=arm, seeds=shared.seeds, config=shared,
               sharpe_mean=float('nan') if index == 0 else 0.0)
    for index, arm in enumerate(Arm)
  )
  with pytest.raises(ValueError, match='cannot serialise'):
    ExperimentReport(config=shared, arms=summaries).to_json()


def test_control_arm_never_touches_the_context_path() -> None:
  '''Arm A must run with no readings at all and produce the same result.

  If the control consulted the context path even to ignore it, then a
  bug in the context plumbing would move the control too and every
  B minus A number would be contaminated at the source.
  '''
  panels = synth_panels(count=300)
  shared = config()
  with_readings = run_arm(Arm.CONTROL, panels, shared, 1,
                          readings=readings(count=300))
  without = run_arm(Arm.CONTROL, panels, shared, 1)
  assert with_readings.returns == without.returns
  assert with_readings.sharpe == without.sharpe
  assert with_readings.weights == without.weights


def test_arm_run_records_turnover_in_monthly_units() -> None:
  '''Monthly one-sided turnover must be a fraction, not a raw sum.

  The threshold in :class:`stock_rl.context.KillCriteria` is 0.50 as a
  fraction, so a harness reporting the raw turnover sum would compare
  rupees against a fraction and pass everything.
  '''
  panels = synth_panels(count=400)
  shared = config()
  run = run_arm(Arm.CONTROL, panels, shared, 1)
  assert 0.0 < run.monthly_one_sided_turnover < 1.0
  assert run.total_cost >= 0.0
  assert run.total_return_multiple > 0.0
