#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for context fusion and its gate.

The gate tests are the point of this file. Everything in
:mod:`stock_rl.context` exists to express the design doc's context
proposal, and the evidence review says that proposal has never cleared a
fair baseline for Nifty-50 names. So the module ships **off**, and the
tests assert it is off, assert it stays off, and assert that turning it
on requires an explicit override that only the pre-registered
experiment is allowed to pass.

The width tests are the other point. A context vector whose length
changes with the data silently invalidates every saved policy, so
:data:`CONTEXT_WIDTH` is asserted rather than derived.
'''

import importlib
from datetime import datetime, timedelta, timezone

import pytest

from stock_rl.context import (
  CONTEXT_ENABLED,
  CONTEXT_WIDTH,
  LLM_SCALAR_ENABLED,
  ContextArm,
  KillCriteria,
  SymbolContext,
  context_enabled,
  context_names,
  evaluate_kill_criteria,
  fuse,
  fuse_panel,
  llm_scalar_enabled,
  max_sources,
  state_vector,
  symbol_context,
  zero_context,
)
from stock_rl.graph import nifty50_seed
from stock_rl.graph.traversal import affected_stocks, dependencies
from stock_rl.sentiment import SentimentReading, aggregate

IST = timezone(timedelta(hours=5, minutes=30))
BAR = datetime(2026, 1, 2, 15, 30, tzinfo=IST)
BEFORE = BAR - timedelta(hours=6)
CRUDE = 'commodity:crude'


def reading(
  score: float,
  source: str,
  available_from: datetime = BEFORE,
  symbol: str = 'RELIANCE',
) -> SentimentReading:
  '''Return a valid reading for the fusion tests.

  Args:
    score: Directional score in ``[-1, 1]``.
    source: Source identifier, so distinct sources survive aggregation.
    available_from: Public availability timestamp.
    symbol: Canonical symbol.

  Returns:
    A constructed reading.
  '''
  return SentimentReading(
    symbol=symbol,
    score=score,
    event_type='earnings',
    intensity=0.6,
    available_from=available_from,
    source=source,
  )


def two_sources() -> list[SentimentReading]:
  '''Return two agreeing-enough readings that clear the source minimum.

  Returns:
    Readings for ``RELIANCE`` with different scores, so disagreement is
    non-zero and a test can see it survive fusion.
  '''
  return [reading(0.8, 'reuters'), reading(-0.2, 'cnbc')]


# --- the gate -------------------------------------------------------------

def test_context_is_disabled_by_default() -> None:
  '''Off, under both spellings, on a module that must not be trusted.'''
  assert context_enabled is False
  assert CONTEXT_ENABLED is False
  assert CONTEXT_ENABLED is context_enabled
  assert llm_scalar_enabled is False
  assert LLM_SCALAR_ENABLED is False


def test_disabled_context_fuses_to_zeros_of_the_right_width() -> None:
  '''Off means a zero vector, not a missing one: widths never move.'''
  context = symbol_context('RELIANCE', two_sources(), BAR)
  vector = fuse(context)
  assert vector == zero_context()
  assert len(vector) == CONTEXT_WIDTH
  assert all(value == 0.0 for value in vector)


def test_gate_can_be_overridden_only_explicitly(monkeypatch) -> None:
  '''An experiment may turn it on; nothing else may.'''
  context = symbol_context('RELIANCE', two_sources(), BAR)
  assert fuse(context, enabled=True) != zero_context()
  # The package re-exports the function ``fuse``, whose name shadows the
  # module of the same name, so the module object is fetched explicitly.
  module = importlib.import_module('stock_rl.context.fuse')
  monkeypatch.setattr(module, 'context_enabled', True)
  assert fuse(context) != zero_context()


# --- width and layout -----------------------------------------------------

def test_width_matches_the_named_fields() -> None:
  '''The dense vector is only usable if its layout is documented.'''
  assert CONTEXT_WIDTH == len(context_names())
  assert CONTEXT_WIDTH == 12
  assert context_names()[0] == 'sentiment_score'
  assert 'symbol' not in context_names()


def test_every_context_produces_the_same_length() -> None:
  '''Two different inputs of the same shape, one length. Always.'''
  quiet = symbol_context('RELIANCE', two_sources(), BAR)
  loud = symbol_context('TCS', [
    reading(1.0, 'a', symbol='TCS'), reading(-1.0, 'b', symbol='TCS'),
  ], BAR, graph_depth=3, commodity_dependencies=2)
  assert len(fuse(quiet, True)) == len(fuse(loud, True)) == CONTEXT_WIDTH
  assert fuse(quiet, True) != fuse(loud, True)


def test_fused_vector_is_all_floats_and_zero_when_disabled() -> None:
  '''Concatenating into an RL state needs floats, not ints or Nones.'''
  context = symbol_context('RELIANCE', two_sources(), BAR)
  for vector in (fuse(context, True), fuse(context)):
    assert all(isinstance(value, float) for value in vector)


def test_panel_layout_is_flat_and_ordered() -> None:
  '''Cross-sectional fusion is a documented flatten, not a dict.'''
  first = symbol_context('RELIANCE', two_sources(), BAR)
  second = symbol_context('TCS', two_sources(), BAR, graph_depth=2)
  fused = fuse_panel([first, second], True)
  assert len(fused) == 2 * CONTEXT_WIDTH
  assert fused[:CONTEXT_WIDTH] == fuse(first, True)
  assert fused[CONTEXT_WIDTH:] == fuse(second, True)
  assert not fuse_panel([], True)
  assert len(fuse_panel([first], False)) == CONTEXT_WIDTH


# --- building a context ---------------------------------------------------

def test_symbol_context_keeps_disagreement_as_its_own_field() -> None:
  '''Uncertainty survives fusion instead of being averaged into the score.'''
  context = symbol_context('RELIANCE', two_sources(), BAR)
  values = dict(zip(context_names(), fuse(context, True)))
  assert values['sentiment_score'] == pytest.approx(0.3)
  assert values['sentiment_disagreement'] == pytest.approx(0.5)
  assert values['sentiment_usable'] == 1.0
  assert values['sentiment_source_ratio'] == pytest.approx(
    2 / max_sources)


def test_single_source_context_is_flagged_and_its_score_zeroed() -> None:
  '''An unusable aggregate cannot reach the vector as a real number.'''
  context = symbol_context('RELIANCE', [reading(0.9, 'finbert_local')], BAR)
  values = dict(zip(context_names(), fuse(context, True)))
  assert values['sentiment_usable'] == 0.0
  assert values['sentiment_score'] == 0.0
  assert context.symbol == 'RELIANCE'


def test_leaked_reading_never_reaches_the_vector() -> None:
  '''The leak test at the fusion layer: public time or nothing.'''
  leaked = [reading(0.9, 'reuters'), reading(-0.9, 'cnbc',
                                             available_from=BAR + timedelta(
                                                 days=1))]
  context = symbol_context('RELIANCE', leaked, BAR)
  values = dict(zip(context_names(), fuse(context, True)))
  assert values['sentiment_usable'] == 0.0
  assert values['sentiment_score'] == 0.0


def test_precomputed_aggregate_is_accepted() -> None:
  '''An end-of-day cache must not force a second aggregation.'''
  cached = aggregate(two_sources(), 'RELIANCE', BAR)
  context = symbol_context('RELIANCE', aggregate_result=cached)
  assert context.sentiment_usable == 1.0
  assert context.sentiment_score == pytest.approx(cached.score)


def test_symbol_context_validates_its_inputs() -> None:
  '''Neither a blank symbol nor an empty input is silently accepted.'''
  with pytest.raises(ValueError, match='symbol'):
    symbol_context(' ', two_sources(), BAR)
  with pytest.raises(ValueError, match='readings or a pre-computed'):
    symbol_context('RELIANCE')


def test_context_combines_with_the_dependency_graph() -> None:
  '''Graph reachability is a count and a depth, nothing fancier.'''
  graph = nifty50_seed()
  reached = affected_stocks(graph, CRUDE, max_depth=1)
  ons = [key for key in dependencies(graph, 'stock:ONGC')]
  context = symbol_context('ONGC', [], BAR, aggregate_result=aggregate(
      [], 'ONGC', BAR), graph_depth=reached['ONGC'],
      commodity_dependencies=len(ons), macro_dependencies=0,
      sector_dependencies=1)
  values = dict(zip(context_names(), fuse(context, True)))
  assert values['graph_depth'] == 1.0
  assert values['commodity_dependencies'] == float(len(ons))
  assert values['sector_dependencies'] == 1.0


def test_options_positioning_defaults_to_a_neutral_market() -> None:
  '''PCR 1.0 and IV rank 0.5 mean "no options signal", not "no signal".'''
  context = SymbolContext('RELIANCE')
  values = dict(zip(context_names(), fuse(context, True)))
  assert values['options_pcr'] == 1.0
  assert values['options_iv_rank'] == 0.5
  assert values['options_net_oi_change'] == 0.0


# --- the three arms -------------------------------------------------------

def test_control_arm_leaves_the_price_state_untouched() -> None:
  '''Arm A is a control, not a zero-padded arm B.'''
  prices = [0.01, -0.02, 0.03]
  context = symbol_context('RELIANCE', two_sources(), BAR)
  assert state_vector(prices, context) == tuple(prices)
  assert state_vector(prices, context, ContextArm.CONTROL, True) == (
    tuple(prices))


def test_numeric_arm_appends_context_only_when_enabled() -> None:
  '''Arm B is reachable, but only by an explicit experiment override.'''
  prices = [0.01]
  context = symbol_context('RELIANCE', two_sources(), BAR)
  off = state_vector(prices, context, ContextArm.NUMERIC_CONTEXT)
  on = state_vector(prices, context, ContextArm.NUMERIC_CONTEXT, True)
  assert len(off) == len(prices) + CONTEXT_WIDTH
  assert off[len(prices):] == zero_context()
  assert on[len(prices):] == fuse(context, True)


def test_llm_arm_requires_the_gate_and_an_injected_scalar() -> None:
  '''No LLM call exists here, so the scalar must arrive from outside.'''
  prices = [0.01]
  context = symbol_context('RELIANCE', two_sources(), BAR)
  arm = ContextArm.LLM_SCALAR
  with pytest.raises(ValueError, match='llm_scalar_enabled'):
    state_vector(prices, context, arm, True, False, 0.5)
  with pytest.raises(ValueError, match='llm_scalar_enabled'):
    state_vector(prices, context, arm, True, None, 0.5)
  with pytest.raises(ValueError, match='injected llm_scalar'):
    state_vector(prices, context, arm, True, True)
  built = state_vector(prices, context, arm, True, True, 0.25)
  assert built == tuple(prices) + fuse(context, True) + (0.25,)
  with pytest.raises(ValueError, match=r'llm_scalar must be in'):
    state_vector(prices, context, arm, True, True, 2.0)


def test_arms_b_and_c_require_a_context() -> None:
  '''Asking for context without supplying it is a wiring bug.'''
  with pytest.raises(ValueError, match='needs a SymbolContext'):
    state_vector([0.01], None, ContextArm.NUMERIC_CONTEXT)


# --- pre-registered kill criteria ----------------------------------------

def test_kill_criteria_pass_only_when_everything_clears() -> None:
  '''One failure is enough, and the reason is listed.'''
  passed = evaluate_kill_criteria(
    alpha_pvalue=0.01, sharpe_pvalue=0.02, abs_t=3.4, folds_passed=5,
    monthly_one_sided_turnover=0.30)
  assert passed.passed
  assert not passed.failures
  borderline = evaluate_kill_criteria(
    alpha_pvalue=0.10, sharpe_pvalue=0.10, abs_t=3.01, folds_passed=4,
    monthly_one_sided_turnover=0.50)
  assert borderline.passed
  # "Require |t| > 3.0" is strict: exactly 3.0 does not clear it.
  assert not evaluate_kill_criteria(
    alpha_pvalue=0.01, sharpe_pvalue=0.01, abs_t=3.0, folds_passed=5,
    monthly_one_sided_turnover=0.1).passed


def test_kill_criteria_name_every_failure() -> None:
  '''The four pre-registered kills, each visible in the output.'''
  result = evaluate_kill_criteria(
    alpha_pvalue=0.4, sharpe_pvalue=0.5, abs_t=1.1, folds_passed=2,
    monthly_one_sided_turnover=0.9)
  assert not result.passed
  assert len(result.failures) == 5
  joined = ' '.join(result.failures)
  assert 'alpha' in joined
  assert 'sharpe' in joined
  assert '|t|' in joined
  assert 'folds' in joined
  assert 'turnover' in joined


def test_kill_criteria_thresholds_are_configurable() -> None:
  '''Pre-registering means writing them down, so they are data.'''
  strict = KillCriteria(alpha_pvalue=0.01, min_abs_t=5.0, folds_passed=5)
  result = evaluate_kill_criteria(
    alpha_pvalue=0.02, sharpe_pvalue=0.01, abs_t=3.5, folds_passed=5,
    monthly_one_sided_turnover=0.1, criteria=strict)
  assert not result.passed
  assert KillCriteria().min_abs_t == 3.0


def test_kill_criteria_reject_nonsense_input() -> None:
  '''A p-value of -1 is a bug in the harness, not a passing result.'''
  with pytest.raises(ValueError, match='p-values'):
    evaluate_kill_criteria(-0.1, 0.01, 3.0, 5, 0.1)
  with pytest.raises(ValueError, match='abs_t'):
    evaluate_kill_criteria(0.01, 0.01, -3.0, 5, 0.1)
  with pytest.raises(ValueError, match='folds_passed'):
    evaluate_kill_criteria(0.01, 0.01, 3.0, -1, 0.1)
  with pytest.raises(ValueError, match='turnover'):
    evaluate_kill_criteria(0.01, 0.01, 3.0, 5, -0.1)
