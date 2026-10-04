#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Sentiment as an injected input: scoring and symbol resolution.

Two modules, because they answer two different questions and keep two
different failure modes apart:

``score``
  Given readings that already exist, what is the number, and may it be
  used at this bar? The rules that live here are the ones a leak would
  exploit: ``available_from`` is public availability rather than fetch
  time, and a single source is not corroboration.

``entity_link``
  Given ``"Reliance"``, which NSE symbol is it? With an explicit refusal
  when the answer is not unique.

**Nothing here calls a model, and that is the finding, not a gap.** The
research review measured the alternatives: buy-and-hold beats FinMem and
FinAgent significantly across 63-91 unbiased symbols (p <= 0.006) with no
significant alpha from either (all p > 0.34), FinBERT-class sentiment
was found **inverted** as a standalone signal, and multi-agent debate
lost to equal-weight 67% of the time over 210 runs. So sentiment is a
*feature* that arrives from outside, it is triaged rather than believed,
and where the sources disagree the disagreement is reported instead of
averaged into a tidy number that hides it.
'''

from stock_rl.sentiment.entity_link import (
  LinkStatus,
  Resolution,
  SymbolLinker,
  SymbolRecord,
  default_linker,
  min_prefix_chars,
  resolve,
)
from stock_rl.sentiment.score import (
  LookAheadError,
  Rejection,
  SentimentAggregate,
  SentimentReading,
  aggregate,
  converged,
  convergence_tolerance,
  default_min_sources,
  disagreement,
  require_visible,
  visible_readings,
)

__all__ = [
  'LinkStatus',
  'LookAheadError',
  'Rejection',
  'Resolution',
  'SentimentAggregate',
  'SentimentReading',
  'SymbolLinker',
  'SymbolRecord',
  'aggregate',
  'converged',
  'convergence_tolerance',
  'default_linker',
  'default_min_sources',
  'disagreement',
  'min_prefix_chars',
  'require_visible',
  'resolve',
  'visible_readings',
]
