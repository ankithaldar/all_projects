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
:func:`stock_rl.sentiment.converged`, which
:func:`stock_rl.sentiment.aggregate` now calls on every set of
per-source scores it publishes.

:func:`~stock_rl.context.fuse.state_vector` will assemble arm C of the
experiment only if the caller injects a scalar it computed elsewhere,
which is the only shape in which an LLM scalar is defensible: one call
per day, one number per ticker, compared against a plain mean.

**No LLM call, no network call and no model in
:mod:`stock_rl.context`, :mod:`stock_rl.sentiment` or
:mod:`stock_rl.graph`.** That claim is scoped to those three packages
because it is only true of those three, and the previous unqualified
"no network call in this package" was false: elsewhere in
``stock_rl`` there are real sockets -- :mod:`stock_rl.compliance.robots`
calls ``urllib.request.urlopen`` and :mod:`stock_rl.api` serves HTTP on
``server.serve_forever``. Those are deliberate and named; a claim that
sweeps them in is not.

**This package re-exports nothing named ``fuse``.** The fusion function
is :func:`~stock_rl.context.fuse.fuse_vector`, and the submodule keeps
the name ``fuse``. A re-exported function called ``fuse`` shadowed the
submodule, so ``import stock_rl.context.fuse as m`` bound the function
and the gate this package exists to keep off was unreadable through the
path its own docstrings print. The submodule is still callable for the
old spelling; see :mod:`stock_rl.context.fuse`.
'''

from typing import TYPE_CHECKING

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
  fuse_panel,
  fuse_vector,
  llm_scalar_enabled,
  max_sources,
  state_vector,
  symbol_context,
  zero_context,
)

if TYPE_CHECKING:
  # At runtime the name ``stock_rl.context.fuse`` is the MODULE, because
  # binding a function to it is the defect this package just fixed. The
  # module is itself callable and delegates to ``fuse_vector``, so
  # ``from stock_rl.context import fuse; fuse(ctx)`` runs -- but a static
  # analyser cannot see a ``sys.modules[...].__class__`` assignment, so
  # the alias is declared here for type checkers and linters only. It is
  # never true at runtime, so it shadows nothing.
  fuse = fuse_vector

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
  'fuse_panel',
  'fuse_vector',
  'llm_scalar_enabled',
  'max_sources',
  'state_vector',
  'symbol_context',
  'zero_context',
]
