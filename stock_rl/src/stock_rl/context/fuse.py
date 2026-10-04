#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Fusing sentiment and graph reachability into a numeric context vector.

This is the design doc's "Contextual State Augmenter", reduced to what
survives the evidence review. The doc proposes a cross-attention module
between context and the market state; the review of the doc's own claims
notes that nothing trains that attention, the RL reward does not
supervise attention weights, and the claimed "core innovation" is a
blank. So: no attention, a fixed-length float vector, and concatenation.

**The gate is off, and it stays off.** :data:`context_enabled` is
``False``. The research review's verdict is that non-price context has
never been shown to add predictive value for Nifty-50 names against a
fair baseline, and it prescribes the experiment before the feature:

* three arms -- **A** price/technical only, **B** A plus ~20 numeric
  scalars, **C** B plus **one** LLM scalar per ticker per day;
* time-consistent splits only (2 of the field's 19 primary studies even
  do this), >= 5 seeds, 5 expanding walk-forward folds;
* Diebold-Mariano on equity curves, paired t-tests on per-stock Sharpe,
  **CAPM alpha with a p-value**, ``|t| > 3.0`` per Harvey, Liu & Zhu,
  and it must hold in >= 4 of 5 folds;
* the benchmark is **1/N and buy-and-hold over Nifty-50**, never a
  trained RL baseline, because DeMiguel (2009) showed 1/N beats almost
  every optimised allocation and Stanford found debate agents lose to
  1/N two thirds of the time.

:func:`evaluate_kill_criteria` encodes those pre-registered thresholds so
"it cleared the bar" is a checkable statement rather than a memory.

**PONYTAIL: naive concatenation of context scalars, and here is why.**
Concatenating ~20 scalars onto a price state is the cheapest thing that
could possibly work, and it is untested. The RL policy is already an
ensemble learner over its state, so a handful of decorrelated scalars is
strictly more sample-efficient than asking ten LLMs each to emit a
sentiment adjective for a judge to pick. It may well add nothing, and
the honest prior is that it adds nothing: Harvey, Liu & Zhu found 158
of 296 published "significant" factors were false discoveries, and
McLean & Pontiff showed published predictors earn 58% less after
publication.

**CEILING: a fixed-length vector loses which signal mattered.** Index 3
is not "sentiment disagreement" to anything downstream except this
function. Nothing in the vector can be attributed, so a gain cannot be
traced to a feature, and a feature cannot be dropped without rebuilding
every saved policy's input layer.

**UPGRADE PATH: a sparse ``(symbol, value)`` mapping is a drop-in
replacement for this dense vector.** Same gate, same width semantics, but
each entry carries its name, so :func:`fuse` and :func:`fuse_panel` can
be swapped for a mapping-returning pair without touching the call site.
The RL state then keeps a name alongside every number, and the audit
record from :mod:`stock_rl.compliance.retention` can say *which* context
signal produced a decision.

**One free fact is worth getting regardless of the experiment's
outcome:** options positioning from the free daily bhavcopy. It is the
only context signal with a genuine informational-asymmetry prior --
what large participants actually did, rather than what the public
already reads. The three option fields below are where it goes. Note
also the constraint the review names: context features have far less
history than price features (bhavcopy ~6-10 years, RBI DBIE less), so
an 8-year price-trained policy is learning a higher-dimensional state
from a shorter sample, and that alone can explain why context "doesn't
help".
'''

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime
from enum import StrEnum
from math import isfinite, isnan
from typing import Final

from stock_rl.sentiment.score import (
  SentimentAggregate,
  SentimentReading,
  aggregate,
)

__all__ = [
  'CONTEXT_ENABLED',
  'CONTEXT_WIDTH',
  'ContextArm',
  'KillCriteria',
  'KillCriteriaResult',
  'LLM_SCALAR_ENABLED',
  'SymbolContext',
  'context_enabled',
  'context_names',
  'evaluate_kill_criteria',
  'fuse',
  'fuse_panel',
  'llm_scalar_enabled',
  'max_sources',
  'state_vector',
  'symbol_context',
  'zero_context',
]

#: Master gate on non-price context. **Default off.** Flipping this is a
#: research decision, not a configuration one: it must stay ``False``
#: until :func:`evaluate_kill_criteria` passes on the three-arm A/B
#: (price-only vs numeric context vs LLM scalar) described in the module
#: docstring, with pre-registered thresholds decided before the results
#: were seen.
context_enabled: Final[bool] = False

#: The same gate under the name the design doc uses, so a grep for the
#: doc's spelling lands here. Kept as an alias rather than as the only
#: name so both spellings answer the same question; both must stay
#: ``False``.
CONTEXT_ENABLED: Final[bool] = context_enabled

#: Gate on the single LLM scalar of arm C. Off, and for a stronger
#: reason than the numeric gate: the review's kill criteria say that if
#: B minus A is insignificant the numeric context is deleted outright,
#: and if C minus B is insignificant then **no LLM call is ever built**.
#: The scalar itself is injected by the caller -- this module makes no
#: network call and holds no model.
llm_scalar_enabled: Final[bool] = False

#: The same gate under the design doc's spelling.
LLM_SCALAR_ENABLED: Final[bool] = llm_scalar_enabled

#: Number of floats :func:`fuse` returns per symbol. Fixed, and
#: asserted by the tests, because a context vector that changes width
#: with the data would silently invalidate every saved policy.
CONTEXT_WIDTH = 12

#: Source count at which the sentiment corroboration ratio saturates.
#: Readings beyond this do not make the mean more trustworthy, they just
#: make the raw count a worse feature.
max_sources = 5


class ContextArm(StrEnum):
  '''The three arms of the pre-registered experiment.

  Attributes:
    CONTROL: **A** -- price and technical features only. The state is
      unchanged by this module, which is what makes it a control.
    NUMERIC_CONTEXT: **B** -- control plus the fused context vector.
      The arm the review says to run *first*, because arms A and B cost
      nothing to compare.
    LLM_SCALAR: **C** -- numeric context plus one scalar per ticker
      from a **single** LLM call per day. Never seven to ten agents: the
      review found debate loses to equal-weight 67% of the time, and
      that a plain mean beat the judge.
  '''

  CONTROL = 'control'
  NUMERIC_CONTEXT = 'numeric_context'
  LLM_SCALAR = 'llm_scalar'


@dataclass(frozen=True, slots=True)
class SymbolContext:
  '''The fused context for one symbol, before it becomes a vector.

  Every field is a plain number already computed by
  :mod:`stock_rl.sentiment` or :mod:`stock_rl.graph`. There is no model
  here and no text.

  Attributes:
    symbol: NSE symbol the context describes.
    sentiment_score: Equal-weight aggregate sentiment in ``[-1, 1]``.
    sentiment_disagreement: Cross-source spread in ``[0, 1]``. **Kept as
      its own field on purpose**: uncertainty is information, and an
      average cannot represent it.
    sentiment_intensity: Mean event intensity in ``[0, 1]``.
    sentiment_usable: 1.0 when the aggregate cleared its source
      minimum, else 0.0. A separate flag rather than a folded-in
      sentinel, because "no usable reading" and "a reading that says
      nothing" must not look alike downstream.
    sentiment_source_ratio: Distinct sources over :data:`max_sources`,
      capped at 1.0.
    graph_depth: Hops from the reference commodity shock, 0.0 when not
      reached. Depth rather than a boolean, because a one-hop exposure
      and a three-hop one are not the same claim.
    commodity_dependencies: Count of commodity nodes this symbol
      depends on.
    macro_dependencies: Count of macro nodes this symbol depends on.
      Global drivers are counted, never duplicated per symbol as
      features: FII/DII flow is one India-wide number, and treating it
      as per-stock is what the doc's "FII Flow Agent" gets wrong.
    sector_dependencies: Count of sector nodes this symbol depends on.
    options_pcr: Put-call ratio from the free daily bhavcopy, the one
      context signal with an informational-asymmetry prior.
    options_iv_rank: Implied-volatility rank in ``[0, 1]``.
    options_net_oi_change: Fractional net change in open interest over
      the window, signed.
  '''

  symbol: str
  sentiment_score: float = 0.0
  sentiment_disagreement: float = 0.0
  sentiment_intensity: float = 0.0
  sentiment_usable: float = 0.0
  sentiment_source_ratio: float = 0.0
  graph_depth: float = 0.0
  commodity_dependencies: float = 0.0
  macro_dependencies: float = 0.0
  sector_dependencies: float = 0.0
  options_pcr: float = 1.0
  options_iv_rank: float = 0.5
  options_net_oi_change: float = 0.0


@dataclass(frozen=True, slots=True)
class KillCriteria:
  '''Thresholds decided **before** the experiment is run.

  Attributes:
    alpha_pvalue: Largest acceptable p-value on CAPM alpha for B minus
      A. Above this, all numeric context work is deleted.
    sharpe_pvalue: Same for the Sharpe comparison.
    min_abs_t: Minimum ``|t|``, from Harvey, Liu & Zhu (2016), who show
      conventional ``t > 2.0`` is inadequate once 300+ factors are
      mined and require ``t > 3.0``; of 296 published significant
      factors, 158 were false discoveries.
    folds_passed: Minimum number of the walk-forward folds in which the
      effect must hold.
    folds_total: Folds in the experiment.
    max_monthly_one_sided_turnover: Novy-Marx & Velikov (2016) find only
      anomalies with monthly one-sided turnover below ~50% survive
      costs; a daily context-driven signal is high-turnover by
      construction, so this is a hard cap rather than a diagnostic.
  '''

  alpha_pvalue: float = 0.10
  sharpe_pvalue: float = 0.10
  min_abs_t: float = 3.0
  folds_passed: int = 4
  folds_total: int = 5
  max_monthly_one_sided_turnover: float = 0.50


@dataclass(frozen=True, slots=True)
class KillCriteriaResult:
  '''Outcome of the pre-registered check.

  Attributes:
    passed: True only when every threshold cleared. One failure is
      enough, and the failures are listed rather than summarised so the
      decision cannot be reinterpreted after seeing them.
    failures: Human-readable reasons, one per unmet threshold.
  '''

  passed: bool
  failures: tuple[str, ...]


def context_names() -> tuple[str, ...]:
  '''Return the context field names in vector order.

  This is the mapping the fixed-length vector does **not** carry. Keep
  it beside the policy that consumes the vector: index ``i`` of
  :func:`fuse` is ``context_names()[i]``, and the whole point of the
  sparse upgrade path is to stop needing this indirection.

  Returns:
    Field names of :class:`SymbolContext`, minus ``symbol``, in
    declaration order. Length is :data:`CONTEXT_WIDTH`.
  '''
  return tuple(field.name for field in fields(SymbolContext)
               if field.name != 'symbol')


def symbol_context(
  symbol: str,
  readings: list[SentimentReading] | None = None,
  decision_bar: datetime | None = None,
  aggregate_result: SentimentAggregate | None = None,
  graph_depth: float = 0.0,
  commodity_dependencies: float = 0.0,
  macro_dependencies: float = 0.0,
  sector_dependencies: float = 0.0,
  options_pcr: float = 1.0,
  options_iv_rank: float = 0.5,
  options_net_oi_change: float = 0.0,
) -> SymbolContext:
  '''Build a :class:`SymbolContext` from sentiment inputs.

  Thin on purpose: it exists so a caller has one obvious way to turn
  readings plus graph counts into a context, and so the sentiment rules
  cannot be bypassed by assembling the dataclass field by field with
  different conventions.

  Args:
    symbol: NSE symbol.
    readings: Readings for the decision, of any symbol. Ignored when
      ``aggregate_result`` is supplied.
    decision_bar: Bar the decision is taken at, passed to
      :func:`stock_rl.sentiment.aggregate`. Should be a timezone-aware
      datetime; anything else is that module's error, not this one's.
    aggregate_result: A pre-computed aggregate, e.g. from an end-of-day
      cache. Takes precedence over ``readings``.
    graph_depth: Hops from the reference commodity shock.
    commodity_dependencies: Count of commodity dependencies.
    macro_dependencies: Count of macro dependencies.
    sector_dependencies: Count of sector dependencies.
    options_pcr: Put-call ratio from the bhavcopy.
    options_iv_rank: Implied-volatility rank.
    options_net_oi_change: Fractional net open-interest change.

  Returns:
    The context, with ``sentiment_usable`` set to 0.0 and the score
      zeroed when the aggregate is unusable, so a single-source or
      leaked reading cannot reach a state vector as a real-looking
      number.

  Raises:
    ValueError: If ``symbol`` is blank, or neither ``readings`` nor
      ``aggregate_result`` supplies anything to aggregate.
  '''
  if not symbol.strip():
    raise ValueError('symbol_context needs a symbol')
  result = aggregate_result
  if result is None:
    if readings is None:
      raise ValueError(
        f'{symbol}: supply readings or a pre-computed aggregate')
    if decision_bar is None:
      # A missing bar must not mean "no filter". It used to, and the
      # permissive default was the one that let the future in: readings
      # dated a year past the decision bar produced sentiment_usable=1.0
      # at score 1.0, with nothing recorded. For a backtesting library the
      # unattributed call is the dangerous one, so refusing is the only
      # safe reading of a bar the caller did not name.
      return SymbolContext(
        symbol=symbol,
        sentiment_score=0.0,
        sentiment_disagreement=0.0,
        sentiment_intensity=0.0,
        sentiment_usable=0.0,
        sentiment_source_ratio=0.0,
        graph_depth=graph_depth,
        commodity_dependencies=commodity_dependencies,
        macro_dependencies=macro_dependencies,
        sector_dependencies=sector_dependencies,
        options_pcr=options_pcr,
        options_iv_rank=options_iv_rank,
        options_net_oi_change=options_net_oi_change,
      )
    result = aggregate(readings, symbol, decision_bar)
  usable = 1.0 if result.usable else 0.0
  return SymbolContext(
    symbol=symbol,
    sentiment_score=result.score if result.usable else 0.0,
    sentiment_disagreement=result.disagreement,
    sentiment_intensity=result.intensity,
    sentiment_usable=usable,
    sentiment_source_ratio=min(1.0, result.source_count / max_sources),
    graph_depth=graph_depth,
    commodity_dependencies=commodity_dependencies,
    macro_dependencies=macro_dependencies,
    sector_dependencies=sector_dependencies,
    options_pcr=options_pcr,
    options_iv_rank=options_iv_rank,
    options_net_oi_change=options_net_oi_change,
  )


def zero_context() -> tuple[float, ...]:
  '''Return the neutral context vector.

  Returns:
    A zero tuple of length :data:`CONTEXT_WIDTH`. Zero rather than the
    field defaults, because the defaults are a *neutral options market*
    (``pcr=1.0``, ``iv_rank=0.5``) while zero means "no context at all",
    and the two must not be confused in an audit of a run made with the
    gate off.
  '''
  return (0.0,) * CONTEXT_WIDTH


def fuse(
  context: SymbolContext,
  enabled: bool | None = None,
) -> tuple[float, ...]:
  '''Fuse one symbol's context into a fixed-length float vector.

  Args:
    context: Context to fuse.
    enabled: Override for the module gate. ``None`` uses
      :data:`context_enabled`. A caller may pass ``True`` **only** to run
      the pre-registered arm B/C experiment; production code must not
      override a gate that exists to stop it.

  Returns:
    Tuple of exactly :data:`CONTEXT_WIDTH` floats in
    :func:`context_names` order, or an all-zero vector of the same
    length when the gate is off.

  Raises:
    ValueError: Never. A bad context is the caller's dataclass
      validation, not this function's.
  '''
  if not (context_enabled if enabled is None else enabled):
    return zero_context()
  values = [getattr(context, name) for name in context_names()]
  return tuple(float(value) for value in values)


def fuse_panel(
  contexts: list[SymbolContext] | tuple[SymbolContext, ...],
  enabled: bool | None = None,
) -> tuple[float, ...]:
  '''Fuse a cross-section into one flat, fixed-length vector.

  Args:
    contexts: Per-symbol contexts, in the caller's order. Order is
      preserved and **is** the layout, so a caller passing a
      differently-ordered list silently reorders the state; the tests
      pin the layout rather than sorting it.
    enabled: Override for the module gate, as in :func:`fuse`.

  Returns:
    Tuple of ``len(contexts) * CONTEXT_WIDTH`` floats. Empty input gives
    an empty tuple, not zeros: "no symbols" is not "50 symbols with no
    context".

  Raises:
    ValueError: Never.
  '''
  out: list[float] = []
  for context in contexts:
    out.extend(fuse(context, enabled))
  return tuple(out)


def state_vector(
  price_features: list[float] | tuple[float, ...],
  context: SymbolContext | None = None,
  arm: ContextArm = ContextArm.CONTROL,
  enabled: bool | None = None,
  llm_enabled: bool | None = None,
  llm_scalar: float | None = None,
) -> tuple[float, ...]:
  '''Assemble the RL state for one arm of the experiment.

  This is the only place the design doc's context proposal is
  expressible, and it is expressible only through two gates that are
  both ``False`` in this module. Nothing else in the codebase
  concatenates context into a state, so the experiment cannot be run by
  accident from a feature builder.

  * ``control`` (arm A) -- ``price_features`` unchanged. The
    context argument is ignored, which is what makes it a control
    rather than a zero-padded one.
  * ``numeric_context`` (arm B) -- price features plus
    :func:`fuse` output.
  * ``llm_scalar`` (arm C) -- arm B plus **one** injected scalar.

  Args:
    price_features: Price/technical features for this symbol, unchanged
      and unnormalised. This function does not rescale anything.
    context: Context to fuse. Required for arms B and C.
    arm: Which arm to assemble.
    enabled: Override for the context gate.
    llm_enabled: Override for the LLM gate.
    llm_scalar: The single LLM scalar for arm C, in ``[-1, 1]``. Must be
      supplied by the caller: **this module never calls a model**, and
      the review's costed prescription is one call per day per ticker,
      not 500-650.

  Returns:
    Tuple of floats: the price features, then the context vector, then
      the scalar for arm C.

  Raises:
    ValueError: If an arm needs a context and none was given, if arm C
      has no scalar, or if a scalar is outside ``[-1, 1]``.
  '''
  arm = ContextArm(arm)
  features = [float(value) for value in price_features]
  if arm is ContextArm.CONTROL:
    return tuple(features)
  if context is None:
    raise ValueError(f'arm {arm.value} needs a SymbolContext')
  features.extend(fuse(context, enabled))
  if arm is ContextArm.NUMERIC_CONTEXT:
    return tuple(features)
  if not (llm_scalar_enabled if llm_enabled is None else llm_enabled):
    raise ValueError(
      f'arm {arm.value} needs {llm_scalar_enabled=}: the review says do '
      f'not build LLM calls unless numeric context cleared its kill '
      f'criteria')
  if llm_scalar is None:
    raise ValueError('arm llm_scalar needs an injected llm_scalar')
  if not -1.0 <= llm_scalar <= 1.0:
    raise ValueError(f'llm_scalar must be in [-1, 1], got {llm_scalar}')
  features.append(float(llm_scalar))
  return tuple(features)


def _reject_underspecified(
  measured: Mapping[str, float],
  failures: list[str],
) -> None:
  '''Record a failure for every measurement that is not a real number.

  This gate is the only thing standing between a p-hacked context
  experiment and "keep the feature", so it must fail **closed**. It did
  not. Every validation and every threshold test below was a comparison
  against a bound, and every comparison involving NaN is False, so a NaN
  skipped its own validation, skipped its own failure check, and was
  reported as a threshold that cleared:

      honest bad result   alpha p=0.9, |t|=0.1, turnover 900%
                          -> passed=False, five failures
      same fields as NaN  -> passed=True, failures=()

  A NaN is what a division by a zero-variance window, an empty fold or a
  dropped observation produces, so this was reachable rather than
  theoretical -- and it failed in the permissive direction, on the one
  decision that governs the whole programme. The module sat at 100%
  statement and 100% branch coverage with every line of this function
  executed by the shipped suite, which is the clearest evidence in this
  repository that coverage is not evidence of correctness.

  Args:
    measured: The measured values, keyed by the name used in the message.
    failures: Accumulator that each offending name is appended to.
  '''
  for name, value in measured.items():
    if isfinite(value):
      continue
    if isnan(value):
      failures.append(
        f'{name} is NaN: the measurement does not exist, so the threshold '
        f'cannot have cleared. Fix the measurement, do not treat this as '
        f'a pass')
    else:
      raise ValueError(
        f'{name} must be finite, got {value!r}')


def evaluate_kill_criteria(
  alpha_pvalue: float,
  sharpe_pvalue: float,
  abs_t: float,
  folds_passed: int,
  monthly_one_sided_turnover: float,
  criteria: KillCriteria = KillCriteria(),
) -> KillCriteriaResult:
  '''Check the pre-registered experiment against its kill criteria.

  Every threshold here is from the research review and was fixed before
  the experiment is run. That is the entire mechanism: the review's
  complaint is not that the results were wrong, it is that nobody decided
  in advance what would count as failure, which is how a t-statistic of
  1.8 becomes "promising".

  Args:
    alpha_pvalue: p-value on CAPM alpha for B minus A.
    sharpe_pvalue: p-value on the Sharpe comparison for B minus A.
    abs_t: Absolute t-statistic on the primary comparison.
    folds_passed: Walk-forward folds in which the effect held.
    monthly_one_sided_turnover: Monthly one-sided turnover as a
      fraction, e.g. 0.4 for 40%.
    criteria: Thresholds to apply.

  Returns:
    A :class:`KillCriteriaResult`. ``passed`` is False if any threshold
      failed, and every failure is listed.

  Raises:
    ValueError: If a p-value or turnover is negative, ``folds_passed`` is
      negative, or ``abs_t`` is negative.
  '''
  failures: list[str] = []
  _reject_underspecified(
    {
      'alpha_pvalue': alpha_pvalue,
      'sharpe_pvalue': sharpe_pvalue,
      'abs_t': abs_t,
      'monthly_one_sided_turnover': monthly_one_sided_turnover,
    },
    failures)
  # NaN is a missing measurement and becomes a named failure. A negative
  # number is a caller bug and raises. Both checks are needed: replacing
  # one with the other is a regression the existing suite caught.
  if alpha_pvalue < 0.0 or sharpe_pvalue < 0.0:
    raise ValueError(
      f'p-values must be >= 0, got alpha={alpha_pvalue!r} '
      f'sharpe={sharpe_pvalue!r}')
  if abs_t < 0.0:
    raise ValueError(f'abs_t must be >= 0, got {abs_t!r}')
  if folds_passed < 0:
    raise ValueError(f'folds_passed must be >= 0, got {folds_passed!r}')
  if monthly_one_sided_turnover < 0.0:
    raise ValueError(
      f'turnover must be >= 0, got {monthly_one_sided_turnover!r}')
  if alpha_pvalue > criteria.alpha_pvalue:
    failures.append(
      f'alpha p={alpha_pvalue:.3f} > {criteria.alpha_pvalue}: delete all '
      f'numeric context work')
  if sharpe_pvalue > criteria.sharpe_pvalue:
    failures.append(
      f'sharpe p={sharpe_pvalue:.3f} > {criteria.sharpe_pvalue}: numeric '
      f'context did not beat price-only')
  if abs_t <= criteria.min_abs_t:
    failures.append(
      f'|t|={abs_t:.2f} <= {criteria.min_abs_t}: conventional t>2.0 is '
      f'inadequate after data mining')
  if folds_passed < criteria.folds_passed:
    failures.append(
      f'{folds_passed} of {criteria.folds_total} folds passed, need '
      f'{criteria.folds_passed}')
  if monthly_one_sided_turnover > criteria.max_monthly_one_sided_turnover:
    failures.append(
      f'monthly one-sided turnover '
      f'{monthly_one_sided_turnover:.2%} > '
      f'{criteria.max_monthly_one_sided_turnover:.0%}: abandon regardless '
      f'of backtest return')
  return KillCriteriaResult(passed=not failures, failures=tuple(failures))

