#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle-15 pins: docs match cited code truth (Cycle-14 findings).'''

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs'
ARCH = (DOCS / 'architecture.md').read_text(encoding='utf-8')
API = (DOCS / 'api.md').read_text(encoding='utf-8')
SRC = ROOT / 'src/job_hunter/job_hunter'


def _section(text: str, start: str, end: str) -> str:
  '''Extract a markdown section between two headings.

  Args:
    text: Full document text.
    start: Heading marker for the section start.
    end: Heading marker for the section end.

  Returns:
    Text between the markers.
  '''
  begin = text.index(start)
  finish = text.index(end, begin)
  return text[begin:finish]


def test_fanout_cooled_sql_and_deferred_count() -> None:
  '''Fan-out: cooled SQL join/pre-filter, aggregate skips, stamp, deferred.'''
  s10 = _section(ARCH, '### 10.2', '## 11.')
  assert 'cooled join' in s10.lower()
  assert 'list_cooled_due' in s10
  assert 'count_due' in s10
  assert '_filtered_due_count' in s10
  assert 'cooled_skips' in s10
  assert 'mark_checked' in s10
  assert 'untruncated filtered-eligible count' in s10
  assert 'deferred_targets' in s10
  assert 'MAX_FANOUT' in s10
  # §11 cross-cutting cooldown text agrees (fetch_pair + cooled_skips).
  s11 = _section(ARCH, '## 11.', '## 12.')
  assert 'cooled_skips' in s11
  assert 'fetch_pair' in s11


def test_fanout_ats_nullif_both_run_kinds() -> None:
  '''Fan-out: list_due ATS/NULLIF-level applies to both run kinds.'''
  s10 = _section(ARCH, '### 10.2', '## 11.')
  assert 'NULLIF' in s10
  assert 'list_due' in s10
  assert 'both run kinds' in s10
  assert 'truthy' in s10
  assert 'quick' in s10.lower()
  assert 'priority >= 5' in s10 or 'priority>=5' in s10


def test_constants_home_layout_row() -> None:
  '''Constants: §4 layout has core/constants.py single-home row.'''
  s4 = _section(ARCH, '## 4.', '## 5.')
  assert 'core/constants.py' in s4
  assert 'DEFAULT_WEIGHTS' in s4
  assert 'DEFAULT_SALARY_FLOOR_LPA' in s4
  assert 'DEFAULT_EMBED_MODEL' in s4
  assert 'MAX_SQLITE_ID' in s4
  assert 'clamp_priority' in s4
  assert 'single home' in s4.lower()
  assert 'scorer' in s4.lower() and 're-export' in s4.lower()
  # Code truth behind the row.
  const_src = (SRC / 'core/constants.py').read_text(encoding='utf-8')
  for name in (
    'DEFAULT_WEIGHTS', 'DEFAULT_SALARY_FLOOR_LPA', 'DEFAULT_EMBED_MODEL',
    'MAX_SQLITE_ID', 'clamp_priority',
  ):
    assert name in const_src
  scorer_src = (SRC / 'services/scorer.py').read_text(encoding='utf-8')
  assert 'from job_hunter.core.constants import' in scorer_src


def test_api_max_sqlite_id_symmetric_negative_huge() -> None:
  '''API bounds: MAX_SQLITE_ID symmetric, negative-huge 422.'''
  assert 'MAX_SQLITE_ID' in API
  assert 'ge=-MAX_SQLITE_ID' in API
  assert 'le=+MAX_SQLITE_ID' in API or 'le=MAX_SQLITE_ID' in API
  assert 'negative-huge' in API.lower()
  assert '422' in API
  # Code truth: routers use the symmetric bound.
  for name in ('jobs.py', 'companies.py', 'recommendations.py', 'runs.py'):
    src = (SRC / f'api/routes/{name}').read_text(encoding='utf-8')
    assert 'ge=-MAX_SQLITE_ID' in src, name
    assert 'le=MAX_SQLITE_ID' in src, name


def test_source_key_contextvar_absent_actual() -> None:
  '''source_key: logging has no source_key_var; docs say (run_id, node).'''
  logging_src = (SRC / 'core/logging.py').read_text(encoding='utf-8')
  assert 'source_key_var' not in logging_src
  assert 'run_id_var' in logging_src and 'node_var' in logging_src
  s21 = _section(ARCH, '## 21.', '## 22.')
  assert '(run_id, node)' in s21
  assert 'run_id, node, source_key' not in ARCH
  assert 'source_key' not in s21


def test_semantic_trunc_embedding_scope_fallback() -> None:
  '''Semantic trunc scoped to embedding creation; fallback separate.'''
  s14 = _section(ARCH, '## 14.', '## 15.')
  assert 'embedding creation' in s14.lower()
  assert 'candidate_doc[:2000]' in s14
  assert 'description_text[:1800]' in s14
  assert 'full candidate_doc' in s14
  assert 'description[:2500]' in s14 or 'description_text[:2500]' in s14
  assert 'semantic_fallback' in s14
  # Code truth behind the scope.
  nodes_src = (SRC / 'graph/nodes.py').read_text(encoding='utf-8')
  assert 'candidate_doc[:2000]' in nodes_src
  assert "[:1800]" in nodes_src
  assert "[:2500]" in nodes_src


def test_api_bounds_discovery_salary_weights() -> None:
  '''API bounds: discovery-cap, salary-floor, scoring-weights rows.'''
  assert 'discovery-cap' in API.lower() or 'discovery cap' in API.lower()
  assert '1..10000' in API
  assert 'salary-floor' in API.lower() or 'salary floor' in API.lower()
  assert '0..1000' in API
  assert 'scoring-weights' in API.lower() or 'scoring_weights' in API
  assert 'finite' in API.lower()
  assert 'overflow' in API.lower()
  # Code truth behind the rows.
  settings_src = (SRC / 'api/routes/settings.py').read_text(encoding='utf-8')
  assert '1 <= value <= 10000' in settings_src
  assert '0 <= number <= 1000' in settings_src
  assert 'math.isfinite' in settings_src
  assert 'OverflowError' in settings_src


def test_bulk_empty_failed_split() -> None:
  '''Bulk: empty vs failed split in §11 (and report fields).'''
  s11 = _section(ARCH, '## 11.', '## 12.')
  assert 'empty' in s11.lower() and 'failed' in s11.lower()
  assert 'report.empty' in s11 or '`empty`' in s11
  assert 'report.failed' in s11 or '`failed`' in s11
  # Code truth: fetch_source returns an ok sentinel splitting the two.
  bulk_src = (SRC / 'services/bulk_fetch.py').read_text(encoding='utf-8')
  assert 'report.empty' in bulk_src and 'report.failed' in bulk_src


def test_upsert_last_checked_invariant() -> None:
  '''Upsert preserves last_checked_at; mark_checked is the only writer.'''
  s12 = _section(ARCH, '## 12.', '## 13.')
  assert 'upsert' in s12.lower()
  assert 'last_checked_at' in s12
  assert 'preserves' in s12.lower()
  assert 'mark_checked' in s12
  assert 'only writer' in s12.lower()
  # Code truth: upsert SQL never mentions last_checked_at; mark_checked does.
  comp_src = (SRC / 'db/repositories/companies.py').read_text(encoding='utf-8')
  upsert_src = comp_src[comp_src.index('def upsert'):comp_src.index('def find_by_name')]
  assert 'last_checked_at' not in upsert_src
  assert 'last_checked_at' in comp_src
