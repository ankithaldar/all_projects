#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The three-arm experiment that decides whether context is built.

One module, one question: **does adding the fused context vector to the
RL state make the strategy better than the same strategy on price
alone?** The roadmap puts the data fabric and the sentiment LLM first
and the measurement later, and that order is backwards, because the
fabric is the expensive layer and the measurement is free.

So this package is the cheap test, and it is deliberately a harness
rather than a strategy. :mod:`stock_rl.experiment.harness` runs three
arms over the same panels, the same history, the same rebalance
cadence, the same cost model and the same walk-forward folds:

======  ==========================  =================================
arm     state vector                 what it answers
======  ==========================  =================================
A       price features only          the control, byte-identical to
                                      the no-context path
B       A + fused context vector     does context add anything at all
C       B + one injected LLM scalar  is the extra call worth its cost
======  ==========================  =================================

Arm C is the one where a naive implementation double-counts, which is
why its context block is required to be a byte-identical prefix-copy of
arm B's rather than a second, separately computed fusion.

The thresholds are not decided here. They live in
:mod:`stock_rl.context.fuse` as :class:`~stock_rl.context.fuse
.KillCriteria`, and this package only feeds measurements into
:func:`~stock_rl.context.fuse.evaluate_kill_criteria` and reports the
result verbatim. The gate itself
(:data:`~stock_rl.context.fuse.context_enabled`) stays ``False``
regardless of the verdict, and nothing in this package flips it: a
verdict is a report, not a switch.

The harness reports a **mean and a standard deviation across seeds**
and refuses a Sharpe the available history cannot support, reusing
:func:`stock_rl.metrics.minimum_backtest_length` rather than a second,
competing threshold. A single seed's Sharpe is not a result, and
Grądzki (2026) measured how much of one is selection.

The evidence this is built against is in
``docs/research/agents/llm-agents-in-finance.md``: buy-and-hold beats
LLM agents significantly across 63-91 unbiased symbols, no agent
produces significant alpha, multi-agent debate loses to equal weight
67% of the time, and of 296 published "significant" factors 158 were
false discoveries. The prior is that this experiment ends in KILL. That
is the reason to run it before writing the fabric, not after.
'''

from stock_rl.experiment.harness import (
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
  'price_features',
  'price_width',
  'run_arm',
  'run_experiment',
]
