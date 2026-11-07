#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Shared scoring/embedding defaults (single home for core constants).

services/scorer.py re-exports these so existing imports keep working;
core modules import from here, not from services
except core/bootstrap.py composition root (documented) (no core->services
arrow).
'''

from __future__ import annotations

import math
from typing import Any

DEFAULT_WEIGHTS: dict[str, float] = {
  'skills_must': 0.30,
  'skills_nice': 0.10,
  'semantic': 0.25,
  'seniority': 0.10,
  'title_fit': 0.05,
  'salary_fit': 0.10,
  'recency': 0.05,
  'company_fit': 0.05,
}

DEFAULT_SALARY_FLOOR_LPA = 45.0
'''Default salary hard floor in LPA.'''

DEFAULT_EMBED_MODEL = 'BAAI/bge-small-en-v1.5'
'''Default embedding model id.'''

MAX_SQLITE_ID = 2**63 - 1
'''SQLite INTEGER upper bound for path ids (signed 64-bit).'''


def clamp_priority(v: Any) -> int:
  '''Clamp a company priority into the supported 1..5 range.

  Args:
    v: Raw priority value.

  Returns:
    Priority clamped to 1..5.

  Raises:
    ValueError: On bool, non-finite values (inf/nan) or non-numeric input.
  '''
  # bool is an int subclass but is never a valid priority.
  if isinstance(v, bool):
    raise ValueError(f'invalid priority: {v!r}')
  # Int-first: float() on a huge int (e.g. 10**400) raises
  # OverflowError, so clamp ints before any float conversion.
  if isinstance(v, int):
    return max(1, min(v, 5))
  try:
    value = float(v)
  except OverflowError as exc:
    raise ValueError(f'invalid priority: {v!r}') from exc
  if not math.isfinite(value):
    raise ValueError(f'invalid priority: {v!r}')
  return max(1, min(int(value), 5))
