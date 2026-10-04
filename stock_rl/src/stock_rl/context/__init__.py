#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Numeric context fusion, behind a gate that is off.

One module, one job: turn sentiment aggregates and dependency-graph
reachability into a fixed-length float vector that can be concatenated
onto an RL state, or into nothing at all.

**The default is nothing at all.** :data:`~stock_rl.context.fuse
.context_enabled` is ``False`` and
:data:`~stock_rl.context.fuse.CONTEXT_ENABLED` is the same value under
the design doc's spelling. It stays ``False`` until the three-arm
experiment clears pre-registered kill criteria -- price-only against
numeric context against one LLM scalar -- with ``|t| > 3.0``, a CAPM
alpha p-value, at least 4 of 5 walk-forward folds, and monthly one-sided
turnover under 50%, benchmarked against 1/N and buy-and-hold over
Nifty-50 rather than against another trained policy.

The design doc's "Contextual State Augmenter" is a cross-attention
module whose nothing trains it, so it is not rebuilt here. The review's
prescription for the one thing that did work -- an anti-convergence
guard, the Jensen-Shannon intervention with Sharpe +0.14 at p = 0.028 --
is implemented where it belongs, on the sentiment side, as
:func:`stock_rl.sentiment.converged`.

There is no LLM call in this package, no network call, and no model.
:func:`~stock_rl.context.fuse.state_vector` will assemble arm C of the
experiment only if the caller injects a scalar it computed elsewhere,
which is the only shape in which an LLM scalar is defensible: one call
per day, one number per ticker, compared against a plain mean.
'''

from stock_rl.context.fuse import (
  CONTEXT_ENABLED,
  CONTEXT_WIDTH,
  LLM_SCALAR_ENABLED,
  ContextArm,
  KillCriteria,
  KillCriteriaResult,
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
