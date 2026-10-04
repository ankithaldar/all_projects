#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Run arms A, B and C over one panel set and report a pre-registered
verdict.

**This module is the cheap test that comes before the expensive layer.**
The 90-day roadmap builds a data fabric and a sentiment LLM and *then*
asks whether they help. That order is backwards: the fabric is the
expensive part and the measurement is free. Everything here exists so
the answer arrives before the fabric is paid for.

Three arms, one panel set, identical everything else:

* **A / :attr:`Arm.CONTROL`** -- price features only. The state is the
  tuple the no-context path already produces, unchanged and in the same
  order. :func:`arm_state` returns it by delegation to
  :func:`stock_rl.context.fuse.state_vector`, so "byte-identical" is a
  property of one shared code path rather than a promise in a docstring.
* **B / :attr:`Arm.CONTEXT`** -- A plus the fused context vector.
* **C / :attr:`Arm.PRICE_CONTEXT`** -- B plus one injected LLM scalar.
  This is the arm a naive implementation double-counts in, so
  :func:`arm_state` asserts that C's price block is element-for-element
  A's and that C's context block is element-for-element B's. A second
  independent fusion of the same readings would produce the same
  *values* only by accident; making the identity explicit makes the
  mistake loud.

**Every parameter is shared and asserted to be shared.** One
:class:`ArmConfig` is constructed by :func:`run_experiment` and stored
on every :class:`ArmRun`. :class:`ExperimentReport` re-checks that in
``__post_init__`` and raises rather than reporting a comparison whose
arms were not measured on the same panels, history, cadence, cost model
and folds. A difference in parameters between arms is not a small
inconvenience: it is the entire experiment, gone.

**Statistics, computed with :mod:`stock_rl.metrics` and
:mod:`stock_rl.splits`.** Two distinct numbers are reported per arm
pair and neither substitutes for the other:

``paired t`` / ``sharpe_pvalue``
  A paired t-test on the **per-bar return differences**, treatment
  minus control, which is the daily analogue of the review's paired
  t-test on per-stock Sharpe. ``abs_t`` is its ``|t|``.
``alpha`` / ``alpha_pvalue``
  A CAPM regression of the *same* paired difference on the benchmark
  return series. The intercept is the incremental alpha and its t
  statistic gets its own p-value, because an effect that is entirely
  explained by beta is not an effect.

The benchmark is equal weight over the same panels
(:func:`benchmark_returns`), because DeMiguel (2009) found 1/N beats
almost every optimised allocation and the review is explicit that the
comparison must not be against another trained policy.

``folds_passed`` counts the walk-forward folds, from
:func:`stock_rl.splits.purged_folds`, in which the treatment beat the
control on the paired difference. The review demands the effect hold in
at least 4 of 5; a mean over folds is not that.

**Seeds are reported as a distribution, never as one number.** Each
:class:`ArmSummary` and :class:`Comparison` carries a mean and a sample
standard deviation across seeds, and the kill criteria are evaluated on
the **least favourable seed** for every threshold. Pre-registered and
symmetric: a criterion has to hold for every seed to count, and the
rule is decided here rather than after seeing which seed disagreed.

**The verdict is not a parameter.** :attr:`ExperimentReport.verdict` is
derived from :attr:`~stock_rl.context.fuse.KillCriteriaResult.passed`
and nothing else, so no caller can pass a KEEP in. It is ``KEEP`` when
the pre-registered thresholds all cleared, ``KILL`` when any of them
failed, and ``INCONCLUSIVE`` when the length gate in
:func:`stock_rl.metrics.minimum_backtest_length` says the available
history cannot support the claimed Sharpe at all -- in which case no
threshold was evaluated and no verdict about the context layer may be
spoken. The failure strings are carried verbatim from
:func:`~stock_rl.context.fuse.evaluate_kill_criteria`; they are never
paraphrased, counted, or rounded into "mostly passed".

Nothing here flips :data:`stock_rl.context.fuse.context_enabled`. It is
``False`` before this module runs and ``False`` after, whatever the
verdict, because a verdict is evidence and the gate is a decision.

**A synthetic panel cannot support a conclusion.** Every number this
module produces is a function of the panels handed to it. On generated
data the verdict is a test of the harness, not a finding about the
context layer, and the report says so on its face through
:attr:`ExperimentReport.panel_kind`.

PONYTAIL: two-sided p-values come from the normal distribution
(:class:`statistics.NormalDist`), not a Student t. Ceiling: the t
tail is heavier at small samples, so a borderline ``|t|`` here is
slightly anti-conservative. Upgrade path: replace
:func:`_two_sided_p` with a Student t CDF; it is one function and every
caller already routes through it.

PONYTAIL: the ranking scorer is a seeded linear map over the state
vector, not a trained policy. Ceiling: it cannot learn interactions, so
the harness measures whether context *reaches the decision* at all, not
whether a learner could extract more from it. Upgrade path: pass a
learner that consumes the same state tuples; the arms, the folds, the
cost model and the kill criteria are unchanged by that swap, which is
the reason they are separate from the scorer.
'''

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from statistics import NormalDist, fmean, stdev
from typing import Final

from stock_rl.bars import Bar
from stock_rl.context.fuse import (
  CONTEXT_WIDTH,
  ContextArm,
  KillCriteria,
  KillCriteriaResult,
  SymbolContext,
  evaluate_kill_criteria,
  state_vector,
  symbol_context,
)
from stock_rl.costs import DELIVERY, CostModel
from stock_rl.graph.edges import DependencyGraph
from stock_rl.graph.traversal import affected_stocks
from stock_rl.indicators import momentum, realized_volatility
from stock_rl.metrics import (
  TRADING_DAYS_PER_YEAR,
  minimum_backtest_length,
  sharpe_ratio,
  total_return,
  turnover,
)
from stock_rl.portfolio import run_portfolio
from stock_rl.sentiment.score import (
  LookAheadError,
  SentimentReading,
  require_visible,
  visible_readings,
)
from stock_rl.splits import purged_folds

__all__ = [
  'Arm',
  'ArmConfig',
  'ArmRun',
  'ArmSummary',
  'Comparison',
  'ExperimentReport',
  'LookAheadError',
  'Verdict',
  'arm_state',
  'benchmark_returns',
  'build_contexts',
  'compare_arms',
  'linear_scorer',
  'momentum_lookback',
  'price_features',
  'price_width',
  'run_arm',
  'run_experiment',
  'vol_window',
]

#: Number of floats in the price block, identical for all three arms.
#: Fixed, and asserted by the tests, for the same reason
#: :data:`stock_rl.context.fuse.CONTEXT_WIDTH` is: a state that changes
#: width with the data invalidates every saved policy silently.
price_width: Final[int] = 3

#: Lookback for the trailing-return feature. 20 bars, because the review
#: notes India's documented short-horizon effect is momentum.
momentum_lookback: Final[int] = 20

#: Lookback for the realised-volatility feature.
vol_window: Final[int] = 20

#: Trading days in a calendar month, used to annualise turnover to a
#: monthly figure. Novy-Marx & Velikov quote monthly one-sided turnover,
#: so the harness must produce the same unit or the threshold is
#: meaningless.
_trading_days_per_month: Final[int] = 21

#: Normal distribution for two-sided p-values. See the module docstring's
#: note on why this is not a Student t.
_normal: Final[NormalDist] = NormalDist()

class Arm(StrEnum):
  '''The three arms of the pre-registered experiment.

  Attributes:
    CONTROL: **A** -- price and technical features only. Byte-identical
      to the no-context path, which is what makes it a control rather
      than a zero-padded imitation of one.
    CONTEXT: **B** -- the control plus the fused context vector. The arm
      the review says to run first, because A and B cost nothing to
      compare.
    PRICE_CONTEXT: **C** -- B plus one injected scalar per symbol. The
      arm where price and context are both live and a naive
      implementation counts them twice.
  '''

  CONTROL = 'control'
  CONTEXT = 'context'
  PRICE_CONTEXT = 'price_context'


#: Arm name to the context module's arm. Reused rather than re-declared:
#: the whole point of that gate is that there is exactly one place where
#: a state vector is assembled. Kept as a table because the mapping is
#: the entire difference between the arms and must be readable at a
#: glance; a branch per arm would be the same three lines spread out.
_context_arm: Final[dict[Arm, ContextArm]] = {
  Arm.CONTROL: ContextArm.CONTROL,
  Arm.CONTEXT: ContextArm.NUMERIC_CONTEXT,
  Arm.PRICE_CONTEXT: ContextArm.LLM_SCALAR,
}


class Verdict(StrEnum):
  '''What the pre-registered thresholds decided.

  Attributes:
    KEEP: Every threshold in :class:`stock_rl.context.fuse.KillCriteria`
      cleared. Evidence, not a decision: the gate in
      :mod:`stock_rl.context.fuse` stays off regardless.
    KILL: At least one threshold failed. The context layer does not earn
      its build cost.
    INCONCLUSIVE: The history cannot support the claimed Sharpe at all
      (:func:`stock_rl.metrics.minimum_backtest_length`), so no threshold
      was evaluated and no claim may be made in either direction. Not a
      softer version of KILL: it is an absence of a claim.
  '''

  KEEP = 'keep'
  KILL = 'kill'
  INCONCLUSIVE = 'inconclusive'


@dataclass(frozen=True, slots=True)
class ArmConfig:
  '''Parameters shared by every arm, with no per-arm overrides.

  There is deliberately no field here that an arm could read
  differently. Cost model, warm-up, cadence, capital, cap, fold count,
  label horizon, embargo, minimum train size and seeds are the whole
  list of things that have to be identical for the comparison to mean
  anything, and :class:`ExperimentReport` refuses a report whose arms
  disagree.

  Attributes:
    capital: Starting cash in rupees.
    costs: Transaction cost model, shared by all arms.
    history: Warm-up bars before the first decision.
    rebalance_days: Bars between rebalances.
    max_weight: Per-symbol weight cap.
    holdings: Names held per rebalance, chosen by rank on the arm's
      score. Without a holding count every symbol would be sized
      identically and the ranking would be theatre: the three arms
      would trade the same equal-weight book and the comparison would
      measure nothing at all.
    n_folds: Walk-forward folds, matching
      :attr:`stock_rl.context.fuse.KillCriteria.folds_total`.
    horizon: Forward label horizon in bars, for the fold purge.
    embargo: Bars dropped after each test window.
    min_train: Minimum training bars in the first fold.
    seeds: Seeds every arm is evaluated under.
    criteria: Pre-registered thresholds.
    panel_kind: What the panels are. Free text so the report can say
      ``'synthetic'``, which is the only honest description of generated
      data and the reason nobody may quote such a verdict.
  '''

  capital: float = 10_000_000.0
  costs: CostModel = DELIVERY
  history: int = 60
  rebalance_days: int = 21
  max_weight: float = 0.10
  holdings: int = 5
  n_folds: int = 5
  horizon: int = 1
  embargo: int = 1
  min_train: int = 60
  seeds: tuple[int, ...] = (1, 2, 3)
  criteria: KillCriteria = KillCriteria()
  panel_kind: str = 'unknown'


@dataclass(frozen=True, slots=True)
class ArmRun:
  '''One arm's backtest under one seed.

  Attributes:
    arm: Which arm produced this run.
    seed: Seed it ran under.
    config: The shared configuration. Stored so a report can prove the
      arms were measured identically.
    returns: Per-bar portfolio returns over the trading window.
    weights: Target weight snapshot per rebalance.
    sharpe: Annualised Sharpe.
    total_return_multiple: Growth factor.
    total_cost: Transaction cost charged, in rupees.
    monthly_one_sided_turnover: One-way turnover divided by the number
      of months in the window, i.e. the unit Novy-Marx & Velikov quote.
  '''

  arm: Arm
  seed: int
  config: ArmConfig
  returns: list[float] = field(default_factory=list)
  weights: list[dict[str, float]] = field(default_factory=list)
  sharpe: float = 0.0
  total_return_multiple: float = 1.0
  total_cost: float = 0.0
  monthly_one_sided_turnover: float = 0.0


@dataclass(frozen=True, slots=True)
class ArmSummary:
  '''One arm across every seed, as a distribution.

  Attributes:
    arm: Which arm.
    seeds: Seeds the arm ran under.
    config: The configuration this arm was measured under, or ``None``
      for a hand-built summary. Carried on the summary rather than only
      on the report so :meth:`ExperimentReport.__post_init__` can prove
      the arms agreed, which is the property that makes the comparison
      mean anything.
    sharpe_mean: Mean annualised Sharpe across seeds.
    sharpe_stdev: Sample standard deviation of the Sharpe. Never
      omitted: a single seed's Sharpe is not a result.
    total_return_mean: Mean growth factor.
    total_return_stdev: Sample standard deviation of the growth factor.
    monthly_turnover_mean: Mean monthly one-sided turnover.
    monthly_turnover_max: Worst seed's turnover. This, not the mean, is
      what the turnover threshold is judged on.
    mean_returns: Seed-averaged per-bar returns, the series the paired
      tests are run on so that every seed contributes.
  '''

  arm: Arm
  seeds: tuple[int, ...]
  config: ArmConfig | None = None
  sharpe_mean: float = 0.0
  sharpe_stdev: float = 0.0
  total_return_mean: float = 1.0
  total_return_stdev: float = 0.0
  monthly_turnover_mean: float = 0.0
  monthly_turnover_max: float = 0.0
  mean_returns: list[float] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Comparison:
  '''One arm measured against the control.

  Attributes:
    label: ``'b_minus_a'`` or ``'c_minus_a'``.
    treatment: The arm being tested.
    seeds: Seeds both arms ran under.
    return_delta_mean: Mean difference in growth factor.
    return_delta_stdev: Its sample standard deviation.
    sharpe_delta_mean: Mean difference in annualised Sharpe.
    sharpe_delta_stdev: Its sample standard deviation.
    alpha_mean: Mean CAPM intercept of the paired difference regressed
      on the benchmark.
    alpha_stdev: Its sample standard deviation.
    alpha_pvalue: Worst seed's two-sided p-value on the intercept.
    sharpe_pvalue: Worst seed's two-sided p-value on the paired t.
    abs_t: Worst seed's ``|t|``, i.e. the smallest across seeds.
    monthly_one_sided_turnover: Worst seed's turnover, the maximum.
    folds_passed: Folds in which the treatment beat the control on
      **every** bar of the seed-averaged paired difference. Stated as
      all-bars rather than mean-positive, because a fold where context
      helped on average while hurting on 40 percent of bars is not a
      fold the effect held in.
    folds_total: Folds the effect had to hold in.
  '''

  label: str
  treatment: Arm
  seeds: tuple[int, ...]
  return_delta_mean: float = 0.0
  return_delta_stdev: float = 0.0
  sharpe_delta_mean: float = 0.0
  sharpe_delta_stdev: float = 0.0
  alpha_mean: float = 0.0
  alpha_stdev: float = 0.0
  alpha_pvalue: float = 1.0
  sharpe_pvalue: float = 1.0
  abs_t: float = 0.0
  monthly_one_sided_turnover: float = 0.0
  folds_passed: int = 0
  folds_total: int = 0


@dataclass(frozen=True, slots=True)
class ExperimentReport:
  '''The machine-readable verdict, and nothing softer than it.

  Attributes:
    config: The shared configuration every arm ran under.
    arms: One :class:`ArmSummary` per arm, in control-first order.
    comparisons: B minus A and C minus A.
    criteria_inputs: The exact five numbers handed to
      :func:`stock_rl.context.fuse.evaluate_kill_criteria`, kept so a
      reader can re-derive the verdict without rerunning the harness.
      ``folds_passed`` is carried as a float here purely so the whole
      mapping serialises with one numeric type; the call itself passes
      the integer the criteria expect.
    kill: The pre-registered result, failures verbatim, or ``None``
      when the length gate refused to evaluate anything.
    required_years: History the claimed Sharpe needs.
    available_years: History actually available.
    gate_reason: Why the gate did or did not refuse.
    benchmark_sharpe: Sharpe of 1/N on the same panels, costs and
      folds, so a reader can see what "no edge" looked like.
    panel_kind: Copied from the config; ``'synthetic'`` on generated
      data.
  '''

  config: ArmConfig
  arms: tuple[ArmSummary, ...] = ()
  comparisons: tuple[Comparison, ...] = ()
  criteria_inputs: dict[str, float] = field(default_factory=dict)
  kill: KillCriteriaResult | None = None
  required_years: float = 0.0
  available_years: float = 0.0
  gate_reason: str = ''
  benchmark_sharpe: float = 0.0

  def __post_init__(self) -> None:
    '''Refuse a report whose arms were not measured identically.

    Raises:
      ValueError: If the arms disagree on any shared parameter, on the
        seed set, or if the arms are not exactly the three the
        experiment defines. A comparison between differently-parameterised
        arms is not a weak result, it is not a result, and the cheapest
        place to say so is before anything is reported.
    '''
    if len(self.arms) != len(Arm):
      raise ValueError(
        f'expected one summary per arm ({len(Arm)}), got {len(self.arms)}')
    if {summary.arm for summary in self.arms} != set(Arm):
      raise ValueError(
        f'expected arms {sorted(arm.value for arm in Arm)}, got '
        f'{sorted(summary.arm.value for summary in self.arms)}')
    if len(self.config.seeds) != len(set(self.config.seeds)):
      raise ValueError(f'duplicate seeds: {self.config.seeds}')
    for summary in self.arms:
      if summary.seeds != self.config.seeds:
        raise ValueError(
          f'arm {summary.arm.value} ran under seeds {summary.seeds}, '
          f'config says {self.config.seeds}')
    for summary in self.arms:
      if summary.config is not None and summary.config != self.config:
        raise ValueError(
          f'arm {summary.arm.value} was measured under a different '
          f'config: {summary.config} != {self.config}')

  @property
  def verdict(self) -> Verdict:
    '''Return the verdict, derived from the kill result alone.

    There is no ``verdict`` field and no argument anywhere that sets
    one. ``passed`` decides KEEP or KILL, and the absence of a result
    decides INCONCLUSIVE: a gate that refused to evaluate a threshold
    has not earned a KEEP, and calling it a KILL would overstate what
    was actually tested.

    Returns:
      :attr:`Verdict.KEEP`, :attr:`Verdict.KILL` or
        :attr:`Verdict.INCONCLUSIVE`.
    '''
    if self.kill is None:
      return Verdict.INCONCLUSIVE
    return Verdict.KEEP if self.kill.passed else Verdict.KILL

  def to_json(self) -> str:
    '''Return the report as byte-stable JSON.

    Deterministic by construction: keys are sorted, separators are
    fixed, and every float is rounded to a fixed number of decimals
    before serialisation. Two runs over identical inputs produce
    identical bytes, so two runs can be diffed and a change in verdict
    is visible as a change in bytes rather than as a change in tone.

    Returns:
      The JSON document, two-space indented, ending in a newline.
    '''
    payload = {
      'available_years': _round(self.available_years),
      'arms': [_arm_payload(summary) for summary in self.arms],
      'benchmark_sharpe': _round(self.benchmark_sharpe),
      'comparisons': [_comparison_payload(item) for item
                      in self.comparisons],
      'config': _config_payload(self.config),
      'criteria_inputs': {key: _round(value) for key, value
                          in sorted(self.criteria_inputs.items())},
      'failures': list(self.kill.failures) if self.kill else None,
      'gate_reason': self.gate_reason,
      'kill_passed': self.kill.passed if self.kill else None,
      'panel_kind': self.panel_kind,
      'required_years': _round(self.required_years),
      'verdict': self.verdict.value,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + '\n'

  @property
  def panel_kind(self) -> str:
    '''Return what the panels are, as recorded in the config.'''
    return self.config.panel_kind


def price_features(bars: list[Bar]) -> tuple[float, ...]:
  '''Return the price block of the state vector.

  Three scale-free numbers, in a fixed order, for every arm. Scale-free
  matters more than it looks: the linear scorer multiplies these by
  seeded coefficients, and a raw rupee price would swamp a 2 percent
  momentum term by three orders of magnitude, so the seed would decide
  nothing and the seed dispersion the report must publish would be a
  rounding artefact of the price level.

  Args:
    bars: Visible history, oldest first. Only bars before the decision
      are ever passed in, so this cannot see the fill price.

  Returns:
    Tuple of exactly :data:`price_width` floats: trailing
      ``momentum_lookback``-bar return, annualised realised volatility
      over ``vol_window`` bars, and the close's position in that
      window's range in ``[0, 1]``. Warm-up regions are 0.0 rather than
      dropped, so the width never depends on how much history arrived.
  '''
  closes = [bar.close for bar in bars]
  if not closes:
    return (0.0,) * price_width
  trend = momentum(closes, momentum_lookback) or 0.0
  volatility = realized_volatility(closes, vol_window)[-1] or 0.0
  window = closes[-vol_window:] if len(closes) >= vol_window else closes
  low = min(window)
  high = max(window)
  position = 0.0 if high <= low else (closes[-1] - low) / (high - low)
  return (trend, volatility, position)


def linear_scorer(
  state: tuple[float, ...],
  coefficients: tuple[float, ...],
) -> float:
  '''Return a scalar score from a state tuple and a coefficient vector.

  The seed enters only through ``coefficients``, drawn once per run as
  ``uniform(-1, 1)`` of the state's width. Because the draw is
  sequential from a single :class:`random.Random`, the first
  :data:`price_width` coefficients are the same for every arm under the
  same seed, so arm A and arm B score their shared price block
  identically and the difference between them is attributable to the
  context block alone.

  Args:
    state: State tuple for one symbol.
    coefficients: One coefficient per state element. A shorter vector
      truncates rather than raising, which keeps a caller that hardcodes
      a width from crashing the control arm.

  Returns:
    The dot product, truncated to ``min(len(state), len(coefficients))``
      terms.
  '''
  total = 0.0
  for index in range(min(len(state), len(coefficients))):
    total += coefficients[index] * state[index]
  return total


def arm_state(
  arm: Arm,
  price: tuple[float, ...] | list[float],
  context: SymbolContext | None = None,
  llm_scalar: float | None = None,
) -> tuple[float, ...]:
  '''Return the state vector for one arm, with the shape enforced.

  Delegates to :func:`stock_rl.context.fuse.state_vector`, so there is
  one implementation of "what a state looks like" and the control arm's
  state is that function's output rather than a re-derivation of it. The
  two context gates are overridden to ``True`` here and **only** here:
  this is the pre-registered experiment, which is the one caller
  :func:`stock_rl.context.fuse.fuse` documents as allowed to pass it.

  The post-conditions are the point of this function. Arms B and C must
  return the price state unchanged and in exactly the control shape, and
  C must not contain a second, independently computed context block. A
  double-count passes every output-shape test a reviewer would write by
  eye, so it is asserted instead.

  Args:
    arm: Which arm.
    price: Price features, unchanged and unnormalised.
    context: Fused context, required for arms B and C.
    llm_scalar: The single injected scalar for arm C. This module never
      computes one; it has no model and makes no network call.

  Returns:
    Tuple of floats: the price block, then the context block, then the
      scalar for arm C.

  Raises:
    ValueError: If the fused shape does not satisfy the post-conditions,
      which means the state assembly double-counted or reordered.
    LookAheadError: Never from here; a leaked reading is rejected
      earlier, in :func:`build_contexts`.
  '''
  state = state_vector(
    price,
    context,
    arm=_context_arm[arm],
    enabled=True,
    llm_enabled=True,
    llm_scalar=llm_scalar,
  )
  block = tuple(float(value) for value in price)
  if len(block) != price_width:
    raise ValueError(
      f'price block must be {price_width} floats, got {len(block)}')
  extra = 0 if arm is Arm.CONTROL else CONTEXT_WIDTH
  expected = price_width + extra + (1 if arm is Arm.PRICE_CONTEXT else 0)
  if len(state) != expected:
    raise ValueError(
      f'arm {arm.value} produced {len(state)} floats, expected {expected}')
  if state[:price_width] != block:
    raise ValueError(
      f'arm {arm.value} altered the price block: '
      f'{state[:price_width]} != {block}')
  if arm is not Arm.CONTROL:
    context_block = state[price_width:price_width + CONTEXT_WIDTH]
    if any(math.isnan(value) for value in context_block):
      raise ValueError(f'arm {arm.value} has NaN in the context block')
  return state


def build_contexts(
  symbols: tuple[str, ...],
  readings: list[SentimentReading],
  decision_bar: datetime,
  graph: DependencyGraph | None = None,
  shock: str | None = None,
) -> dict[str, SymbolContext]:
  '''Fuse one bar's context, rejecting anything not yet public.

  Every reading handed over is checked with
  :func:`stock_rl.sentiment.score.require_visible` **before** it is
  aggregated, so a reading stamped after the decision bar aborts the
  build instead of being quietly averaged out. Silently dropping it
  would still produce a number, and that number would be built on data
  no trader could have had.

  ``readings`` here are therefore the readings **this decision** may
  use, not the experiment's whole reading history. :func:`run_arm`
  makes that selection per bar with
  :func:`stock_rl.sentiment.score.visible_readings` before calling
  this. The split is deliberate: selecting is unavoidable -- a 2,750
  day run holds thousands of readings that were not public on its
  first bar -- while *refusing* belongs at this boundary, so a caller
  that hands over an unselected list fails loudly here instead of
  getting a number. An all-zero context is a legitimate result, no news
  that day; a leaked one is not.

  Args:
    symbols: Symbols to build context for.
    readings: Readings of any symbol available at this decision.
    decision_bar: The bar the decision is taken at. Must be
      timezone-aware.
    graph: Dependency graph, for the graph half of the context block.
    shock: Node key the graph traversal starts from, e.g.
      ``'commodity:crude'``. Required when ``graph`` is given.

  Returns:
    Mapping of symbol to context, in ``symbols`` order. Depth is the
      minimum hop count from ``shock``, 0.0 where not reached.

  Raises:
    ValueError: If ``decision_bar`` is naive, or ``graph`` is given
      without ``shock``.
    LookAheadError: If any reading became public after the decision bar.
  '''
  if decision_bar.tzinfo is None:
    raise ValueError(
      f'decision_bar must be timezone-aware, got {decision_bar!r}')
  if graph is not None and shock is None:
    raise ValueError('a graph needs a shock node key to traverse from')
  depths = affected_stocks(graph, shock) if graph is not None else {}
  contexts: dict[str, SymbolContext] = {}
  for symbol in symbols:
    mine = [reading for reading in readings
            if reading.symbol == symbol]
    # Refuse loudly rather than filter: a leak is a bug in the data
    # plumbing, not a thin news day.
    require_visible(mine, decision_bar)
    contexts[symbol] = symbol_context(
      symbol,
      mine,
      decision_bar,
      graph_depth=float(depths.get(symbol, 0.0)),
    )
  return contexts


def benchmark_returns(panels: dict[str, list[Bar]],
                      config: ArmConfig) -> list[float]:
  '''Return 1/N per-bar returns on the shared panels, costs and folds.

  The benchmark the review names, not another trained policy: DeMiguel
  (2009) found 1/N beats almost every optimised allocation, so a context
  arm that cannot beat equal weight has not shown anything, and
  measuring it against a weak learned baseline would hide that.

  Args:
    panels: Symbol to bars, all on one timeline.
    config: The shared configuration, so the benchmark faces the same
      warm-up, cadence, caps and costs as every arm.

  Returns:
    Per-bar equal-weight returns over the trading window.
  '''
  result = run_portfolio(
    panels,
    _equal_weight,
    capital=config.capital,
    costs=config.costs,
    rebalance_days=config.rebalance_days,
    max_weight=config.max_weight,
    history=config.history,
  )
  return list(result.returns)


def _equal_weight(visible: dict[str, list[Bar]]) -> dict[str, float]:
  '''Return an equal weight per symbol.

  Args:
    visible: Visible price history, which this baseline ignores.

  Returns:
    One weight per symbol, all equal.
  '''
  count = len(visible)
  if count == 0:
    return {}
  weight = 1.0 / count
  return {symbol: weight for symbol in visible}


def run_arm(
  arm: Arm,
  panels: dict[str, list[Bar]],
  config: ArmConfig,
  seed: int,
  readings: list[SentimentReading] | None = None,
  graph: DependencyGraph | None = None,
  shock: str | None = None,
  llm_scalars: dict[str, float] | None = None,
) -> ArmRun:
  '''Run one arm over the panels under one seed.

  Arms B and C build their context through :func:`build_contexts` at
  every rebalance, from the last **visible** bar's timestamp as the
  decision bar, over the readings
  :func:`stock_rl.sentiment.score.visible_readings` says were public by
  then. Arm A never touches the context path at all, which is what makes
  it a control and not a third reading of the same code.

  Args:
    arm: Which arm to run.
    panels: Symbol to bars, all on one timeline.
    config: The shared configuration.
    seed: Seed for the scorer's coefficients. The backtest itself is
      deterministic given the panels; the seed varies the ranking
      coefficients, which is what makes the seed dispersion in the
      report a measurement rather than a formality.
    readings: Sentiment readings for the experiment.
    graph: Dependency graph for the graph half of the context block.
    shock: Node key the traversal starts from.
    llm_scalars: One injected scalar per symbol, for arm C. Required
      for arm C and refused when missing, so arm C can never silently
      degenerate into arm B.

  Returns:
    An :class:`ArmRun`.

  Raises:
    ValueError: If arm C has no scalar for a symbol, or the panels are
      unusable for the shared configuration.
    LookAheadError: If a reading was not public at a decision bar.
  '''
  readings = list(readings or ())
  symbols = tuple(sorted(panels))
  # One Random per arm would give arm C's price block different
  # coefficients from arm A's and the comparison would measure the
  # draw, not the context. Drawing from a single seeded source fixes the
  # price prefix and lets the context block extend it.
  source = random.Random(seed)
  coefficients = tuple(
    source.uniform(-1.0, 1.0)
    for _ in range(price_width + CONTEXT_WIDTH + 1))

  def provider(visible: dict[str, list[Bar]]) -> dict[str, float]:
    '''Rank symbols on their arm state and size the book.

    Args:
      visible: Price history strictly before the fill bar.

    Returns:
      Raw per-symbol weights; the engine caps and renormalises them.
    '''
    if not visible:
      return {}
    bar = next(iter(visible.values()))[-1]
    decision_bar = bar.timestamp
    if decision_bar.tzinfo is None:
      decision_bar = decision_bar.replace(tzinfo=timezone.utc)
    contexts: dict[str, SymbolContext] = {}
    if arm is not Arm.CONTROL:
      contexts = build_contexts(
        symbols,
        list(visible_readings(readings, decision_bar)),
        decision_bar,
        graph,
        shock,
      )
    scored: list[tuple[float, str]] = []
    for symbol, bars in visible.items():
      state = arm_state(
        arm,
        price_features(bars),
        contexts.get(symbol),
        _llm_scalar(arm, symbol, llm_scalars),
      )
      scored.append((linear_scorer(state, coefficients), symbol))
    scored.sort(reverse=True)
    # Only the top `holdings` names are funded. Sizing every symbol
    # equally would make the rank irrelevant, and then arms A, B and C
    # would all trade the equal-weight book and the experiment would
    # report a perfectly null result about nothing.
    chosen = scored[:max(1, config.holdings)]
    return {symbol: config.max_weight for _, symbol in chosen}

  result = run_portfolio(
    panels,
    provider,
    capital=config.capital,
    costs=config.costs,
    rebalance_days=config.rebalance_days,
    max_weight=config.max_weight,
    history=config.history,
  )
  months = max(1.0, len(result.returns) / _trading_days_per_month)
  traded = sum(turnover([snapshot[symbol] for snapshot in result.weights])
               for symbol in symbols) / max(1, len(symbols))
  return ArmRun(
    arm=arm,
    seed=seed,
    config=config,
    returns=list(result.returns),
    weights=[dict(snapshot) for snapshot in result.weights],
    sharpe=result.sharpe,
    total_return_multiple=result.total_return_multiple,
    total_cost=result.total_cost,
    monthly_one_sided_turnover=traded / months,
  )


def _llm_scalar(
  arm: Arm,
  symbol: str,
  llm_scalars: dict[str, float] | None,
) -> float | None:
  '''Return the injected LLM scalar for arm C, or refuse.

  Args:
    arm: Which arm is running.
    symbol: Symbol being scored.
    llm_scalars: Injected scalars, if any.

  Returns:
    ``None`` for arms A and B, and the injected scalar for arm C.

  Raises:
    ValueError: If arm C has no scalar for ``symbol``. Defaulting to
      0.0 would make arm C byte-identical to arm B and the LLM question
      would be answered "no" by a missing argument rather than by a
      measurement.
  '''
  if arm is not Arm.PRICE_CONTEXT:
    return None
  if not llm_scalars or symbol not in llm_scalars:
    raise ValueError(
      f'arm price_context needs an injected llm_scalar for {symbol}; '
      f'this module calls no model')
  return llm_scalars[symbol]


def compare_arms(
  control: list[ArmRun],
  treatment: list[ArmRun],
  benchmark: list[float],
  config: ArmConfig,
) -> Comparison:
  '''Measure one arm against the control across every seed.

  Two statistics, both on the per-bar paired difference, and neither
  substituting for the other: a paired t-test, which is the daily
  analogue of the review's paired per-stock Sharpe test, and a CAPM
  regression of that difference on the benchmark, whose intercept is
  the alpha left after market exposure is removed.

  The paired series is the **seed-averaged** return of each arm, so
  every seed contributes and no seed is selected after the fact. The
  thresholds are then judged on the least favourable seed for each
  statistic independently -- smallest ``|t|``, largest p-value, largest
  turnover -- because a criterion that holds for the lucky seed and not
  the unlucky ones has not held.

  Args:
    control: Control runs, one per seed.
    treatment: Treatment runs, one per seed.
    benchmark: Per-bar equal-weight returns over the same window.
    config: The shared configuration.

  Returns:
    A :class:`Comparison`.

  Raises:
    ValueError: If the two arms ran under different seeds, if any run
      covers a different number of bars, or if the benchmark is shorter
      than the arms.
  '''
  if len(control) != len(treatment) or not control:
    raise ValueError('control and treatment need the same non-zero seeds')
  control_by_seed = {run.seed: run for run in control}
  treatment_by_seed = {run.seed: run for run in treatment}
  if set(control_by_seed) != set(treatment_by_seed):
    raise ValueError(
      f'seeds differ: control {sorted(control_by_seed)} vs treatment '
      f'{sorted(treatment_by_seed)}')
  seeds = tuple(sorted(control_by_seed))
  # Every run of every arm must cover the same window. Comparing against
  # the *shortest* and truncating the rest would let a longer arm win on
  # exposure and have the difference booked as context.
  length = len(control[0].returns)
  odd = [run for run in list(control) + list(treatment)
         if len(run.returns) != length]
  if odd:
    raise ValueError(
      f'arms report different windows: {length} vs '
      f'{sorted({len(run.returns) for run in odd})}')
  if len(benchmark) < length:
    raise ValueError(
      f'benchmark has {len(benchmark)} bars, need {length}')
  market = benchmark[:length]
  control_mean = _mean_series([control_by_seed[seed].returns[:length]
                               for seed in seeds])
  treatment_mean = _mean_series([treatment_by_seed[seed].returns[:length]
                                 for seed in seeds])
  differences = [after - before
                 for after, before in zip(treatment_mean, control_mean)]

  per_seed_alpha: list[float] = []
  per_seed_p: list[float] = []
  per_seed_t: list[float] = []
  for seed in seeds:
    paired = [after - before for after, before
              in zip(treatment_by_seed[seed].returns[:length],
                     control_by_seed[seed].returns[:length])]
    alpha, _ = _capm_alpha(paired, market)
    per_seed_alpha.append(alpha)
    per_seed_p.append(_two_sided_p(_paired_t(paired)))
    per_seed_t.append(abs(_paired_t(paired)))
  folds = purged_folds(
    length,
    n_folds=config.n_folds,
    horizon=config.horizon,
    embargo=config.embargo,
    min_train=config.min_train,
  )
  held = [all(value > 0.0 for value in
              differences[fold.test_start:fold.test_end])
          for fold in folds]
  return Comparison(
    label=f'{treatment[0].arm.value}_minus_a',
    treatment=treatment[0].arm,
    seeds=seeds,
    return_delta_mean=_delta_mean(
      control, treatment, total_return),
    return_delta_stdev=_delta_stdev(control, treatment, total_return),
    sharpe_delta_mean=_delta_mean(control, treatment, sharpe_ratio),
    sharpe_delta_stdev=_delta_stdev(control, treatment, sharpe_ratio),
    alpha_mean=fmean(per_seed_alpha),
    alpha_stdev=stdev(per_seed_alpha) if len(per_seed_alpha) > 1 else 0.0,
    alpha_pvalue=max(per_seed_p),
    sharpe_pvalue=max(per_seed_p),
    abs_t=min(per_seed_t),
    monthly_one_sided_turnover=max(
      run.monthly_one_sided_turnover for run in treatment),
    folds_passed=sum(held),
    folds_total=len(folds),
  )


def _delta_mean(control: list[ArmRun], treatment: list[ArmRun],
                measure) -> float:
  '''Return the mean treatment-minus-control difference of ``measure``.

  Args:
    control: Control runs.
    treatment: Treatment runs.
    measure: Callable applied to a run's returns.

  Returns:
    Mean of ``measure(treatment) - measure(control)`` over seeds.
  '''
  control_by_seed = {run.seed: run for run in control}
  return fmean([
    measure(run.returns) - measure(control_by_seed[run.seed].returns)
    for run in treatment
  ])


def _delta_stdev(control: list[ArmRun], treatment: list[ArmRun],
                 measure) -> float:
  '''Return the sample stdev of a per-seed difference.

  Args:
    control: Control runs.
    treatment: Treatment runs.
    measure: Callable applied to a run's returns.

  Returns:
    Sample standard deviation, or 0.0 for a single seed.
  '''
  control_by_seed = {run.seed: run for run in control}
  values = [measure(run.returns) - measure(control_by_seed[run.seed].returns)
            for run in treatment]
  return stdev(values) if len(values) > 1 else 0.0


def _mean_series(series: list[list[float]]) -> list[float]:
  '''Return the per-index mean of several equal-length series.

  Args:
    series: Equal-length return series, one per seed.

  Returns:
    The seed-averaged series.

  Raises:
    ValueError: If the series are not all the same length.
  '''
  length = len(series[0])
  if any(len(item) != length for item in series):
    raise ValueError('seed series differ in length')
  return [fmean([item[index] for item in series])
          for index in range(length)]


def _paired_t(differences: list[float]) -> float:
  '''Return the paired t statistic of a per-bar difference series.

  Args:
    differences: Paired per-bar differences.

  Returns:
    The t statistic, or 0.0 where it is undefined: fewer than two
      observations, or no dispersion. Zero rather than infinity because
      zero is a value the threshold can fail, and reporting infinity
      would let a degenerate series pass ``|t| > 3``.
  '''
  if len(differences) < 2:
    return 0.0
  deviation = stdev(differences)
  if deviation == 0.0:
    return 0.0
  return fmean(differences) / (deviation / math.sqrt(len(differences)))


def _capm_alpha(portfolio: list[float],
                market: list[float]) -> tuple[float, float]:
  '''Return the CAPM intercept and beta of a series against the market.

  Ordinary least squares on the paired difference against the equal
  weight benchmark, with no risk-free term because
  :func:`stock_rl.metrics.sharpe_ratio` is used at a zero risk-free rate
  throughout this module and the two must agree about the unit.

  Args:
    portfolio: Series to decompose.
    market: Benchmark returns, at least as long as ``portfolio``.

  Returns:
    Tuple of ``(alpha, beta)``. Both 0.0 when the regression is
      undefined, i.e. fewer than three observations or a flat benchmark.

  Raises:
    ValueError: Never. Inputs are truncated to the shorter length.
  '''
  count = min(len(portfolio), len(market))
  if count < 3:
    return 0.0, 0.0
  y = portfolio[:count]
  x = market[:count]
  mean_x = fmean(x)
  mean_y = fmean(y)
  spread = sum((value - mean_x) ** 2 for value in x)
  if spread == 0.0:
    return 0.0, 0.0
  beta = sum((xi - mean_x) * (yi - mean_y)
             for xi, yi in zip(x, y)) / spread
  return mean_y - beta * mean_x, beta


def _two_sided_p(t_stat: float) -> float:
  '''Return the two-sided normal p-value for a t statistic.

  Args:
    t_stat: The statistic.

  Returns:
    Probability in ``[0, 1]``, symmetric in the sign. See the module
      docstring: this is a normal tail, not a Student t.
  '''
  return 2.0 * (1.0 - _normal.cdf(abs(t_stat)))


def run_experiment(
  panels: dict[str, list[Bar]],
  readings: list[SentimentReading] | None = None,
  graph: DependencyGraph | None = None,
  shock: str | None = None,
  llm_scalars: dict[str, float] | None = None,
  config: ArmConfig | None = None,
) -> ExperimentReport:
  '''Run all three arms and return the pre-registered verdict.

  Every arm receives the same ``config`` object, so there is no code
  path by which two arms could be measured on different panels, history,
  cadence, cost model or folds. :class:`ExperimentReport` re-checks
  that on construction.

  The length gate runs **before** the kill criteria. Claiming a Sharpe
  that the available history cannot support is the exact mistake
  :func:`stock_rl.metrics.minimum_backtest_length` exists to prevent, and
  evaluating thresholds against a number that should never have been
  computed would give a verdict on nothing. A gate refusal yields
  ``kill is None`` and :attr:`Verdict.INCONCLUSIVE`.

  Args:
    panels: Symbol to bars, all on one timeline.
    readings: Sentiment readings for the experiment.
    graph: Dependency graph for the graph half of the context block.
    shock: Node key the traversal starts from.
    llm_scalars: One injected scalar per symbol, required by arm C.
    config: Shared configuration. Defaults are the review's: five
      folds, three seeds, and ``panel_kind='unknown'``. Pass
      ``panel_kind='synthetic'`` for generated panels, which is what
      keeps a demo verdict from being quoted as a finding.

  Returns:
    An :class:`ExperimentReport`.

  Raises:
    ValueError: If the panels or configuration cannot support the
      experiment, or if arm C has no injected scalar.
    LookAheadError: If a reading was not public at a decision bar.
  '''
  config = config or ArmConfig()
  if not panels:
    raise ValueError('no panels to run the experiment on')
  runs: dict[Arm, list[ArmRun]] = {}
  for arm in Arm:
    runs[arm] = [
      run_arm(arm, panels, config, seed, readings, graph, shock,
              llm_scalars)
      for seed in config.seeds
    ]
  summaries = tuple(_summarise(arm, runs[arm]) for arm in Arm)
  benchmark = benchmark_returns(panels, config)
  comparisons = (
    compare_arms(runs[Arm.CONTROL], runs[Arm.CONTEXT], benchmark, config),
    compare_arms(runs[Arm.CONTROL], runs[Arm.PRICE_CONTEXT], benchmark,
                 config),
  )
  primary = comparisons[0]
  length = min(len(summary.mean_returns) for summary in summaries)
  available = length / TRADING_DAYS_PER_YEAR
  claimed = max(summary.sharpe_mean for summary in summaries)
  trials = len(Arm)
  required = (minimum_backtest_length(trials, claimed)
              if claimed > 0.0 else 0.0)
  inputs = {
    'alpha_pvalue': primary.alpha_pvalue,
    'sharpe_pvalue': primary.sharpe_pvalue,
    'abs_t': primary.abs_t,
    'folds_passed': float(primary.folds_passed),
    'monthly_one_sided_turnover': primary.monthly_one_sided_turnover,
  }
  if required > available:
    return ExperimentReport(
      config=config,
      arms=summaries,
      comparisons=comparisons,
      criteria_inputs=inputs,
      kill=None,
      required_years=required,
      available_years=available,
      gate_reason=(
        f'{trials} arms claiming Sharpe {claimed:.3f} need {required:.2f} '
        f'years of returns, only {available:.2f} are available, so no '
        f'kill criterion was evaluated'),
      benchmark_sharpe=sharpe_ratio(benchmark),
    )
  kill = evaluate_kill_criteria(
    alpha_pvalue=primary.alpha_pvalue,
    sharpe_pvalue=primary.sharpe_pvalue,
    abs_t=primary.abs_t,
    folds_passed=primary.folds_passed,
    monthly_one_sided_turnover=primary.monthly_one_sided_turnover,
    criteria=config.criteria,
  )
  return ExperimentReport(
    config=config,
    arms=summaries,
    comparisons=comparisons,
    criteria_inputs=inputs,
    kill=kill,
    required_years=required,
    available_years=available,
    gate_reason=f'{available:.2f} years available against '
                f'{required:.2f} required',
    benchmark_sharpe=sharpe_ratio(benchmark),
  )


def _summarise(arm: Arm, runs: list[ArmRun]) -> ArmSummary:
  '''Summarise one arm across its seeds.

  Args:
    arm: Which arm.
    runs: One run per seed.

  Returns:
    An :class:`ArmSummary` with means, standard deviations and the
      seed-averaged return series.

  Raises:
    ValueError: Never; an empty run list yields an empty summary, which
      :class:`ExperimentReport` then rejects for missing seeds.
  '''
  if not runs:
    return ArmSummary(arm=arm, seeds=())
  sharpes = [run.sharpe for run in runs]
  growth = [run.total_return_multiple for run in runs]
  turnovers = [run.monthly_one_sided_turnover for run in runs]
  return ArmSummary(
    arm=arm,
    seeds=tuple(run.seed for run in runs),
    config=runs[0].config,
    sharpe_mean=fmean(sharpes),
    sharpe_stdev=stdev(sharpes) if len(sharpes) > 1 else 0.0,
    total_return_mean=fmean(growth),
    total_return_stdev=stdev(growth) if len(growth) > 1 else 0.0,
    monthly_turnover_mean=fmean(turnovers),
    monthly_turnover_max=max(turnovers),
    mean_returns=_mean_series([list(run.returns) for run in runs]),
  )


def _round(value: float) -> float:
  '''Return a float rounded for byte-stable serialisation.

  Args:
    value: The number.

  Returns:
    Rounded to 12 decimals. Enough to be exact for every quantity the
      report carries, few enough that a platform difference in the last
      bit of an intermediate does not show up as a changed byte.
  '''
  if math.isnan(value) or math.isinf(value):
    raise ValueError(f'report cannot serialise {value}')
  return round(value, 12)


def _config_payload(config: ArmConfig) -> dict[str, object]:
  '''Return the JSON payload for an :class:`ArmConfig`.

  Args:
    config: The shared configuration.

  Returns:
    A JSON-safe mapping, including the cost model field by field so a
      diff shows which cost changed rather than only that one did.
  '''
  payload: dict[str, object] = {
    'capital': config.capital,
    'history': config.history,
    'horizon': config.horizon,
    'embargo': config.embargo,
    'max_weight': config.max_weight,
    'holdings': config.holdings,
    'min_train': config.min_train,
    'n_folds': config.n_folds,
    'rebalance_days': config.rebalance_days,
    'seeds': list(config.seeds),
  }
  payload['costs'] = asdict(config.costs)
  payload['criteria'] = asdict(config.criteria)
  return payload


def _arm_payload(summary: ArmSummary) -> dict[str, object]:
  '''Return the JSON payload for an :class:`ArmSummary`.

  Args:
    summary: The arm summary.

  Returns:
    A JSON-safe mapping. The full return series is summarised by its
      own length and total rather than inlined: a 2,750-element array
      would make the report undiffable and the byte-stability test
      slow, and no decision reads it.
  '''
  return {
    'arm': summary.arm.value,
    'bars': len(summary.mean_returns),
    'monthly_turnover_max': _round(summary.monthly_turnover_max),
    'monthly_turnover_mean': _round(summary.monthly_turnover_mean),
    'seeds': list(summary.seeds),
    'sharpe_mean': _round(summary.sharpe_mean),
    'sharpe_stdev': _round(summary.sharpe_stdev),
    'total_return_mean': _round(summary.total_return_mean),
    'total_return_stdev': _round(summary.total_return_stdev),
  }


def _comparison_payload(item: Comparison) -> dict[str, object]:
  '''Return the JSON payload for a :class:`Comparison`.

  Args:
    item: The comparison.

  Returns:
    A JSON-safe mapping.
  '''
  return {
    'abs_t': _round(item.abs_t),
    'alpha_mean': _round(item.alpha_mean),
    'alpha_pvalue': _round(item.alpha_pvalue),
    'alpha_stdev': _round(item.alpha_stdev),
    'folds_passed': item.folds_passed,
    'folds_total': item.folds_total,
    'label': item.label,
    'monthly_one_sided_turnover':
        _round(item.monthly_one_sided_turnover),
    'return_delta_mean': _round(item.return_delta_mean),
    'return_delta_stdev': _round(item.return_delta_stdev),
    'seeds': list(item.seeds),
    'sharpe_delta_mean': _round(item.sharpe_delta_mean),
    'sharpe_delta_stdev': _round(item.sharpe_delta_stdev),
    'sharpe_pvalue': _round(item.sharpe_pvalue),
    'treatment': item.treatment.value,
  }
