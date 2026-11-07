#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle-20 pins: docs match cited code truth (Cycle-19 findings).'''

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs'
ARCH = (DOCS / 'architecture.md').read_text(encoding='utf-8')
OVERVIEW = (DOCS / 'app_overview.md').read_text(encoding='utf-8')
RUNBOOK = (DOCS / 'runbook.md').read_text(encoding='utf-8')
TEST_WRITER = (ROOT / 'agents/test-writer.md').read_text(encoding='utf-8')
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


def test_fanout_deferred_filtered_eligible_via_filtered_due_count() -> None:
  '''§10.2: deferred is filtered-eligible via _filtered_due_count.'''
  s10 = _section(ARCH, '### 10.2', '## 11.')
  assert '_filtered_due_count' in s10
  assert 'filtered-eligible' in s10
  assert 'untruncated filtered-eligible count' in s10
  assert 'max(0, eligible_count - MAX_FANOUT)' in s10
  assert 'max(0, count_due() - MAX_FANOUT)' not in s10
  assert '_DUE_FROM' in s10
  assert 'source_enabled' in s10
  assert 'priority >= 5' in s10
  # Code truth: functions and filters behind the claim.
  nodes_src = (SRC / 'graph/nodes.py').read_text(encoding='utf-8')
  assert 'def _filtered_due_count' in nodes_src
  assert 'eligible_count = _filtered_due_count' in nodes_src
  assert 'deferred = max(0, eligible_count - MAX_FANOUT)' in nodes_src
  assert 'CompaniesRepository._DUE_FROM' in nodes_src
  assert "if quick and int(raw['priority'] or 0) < 5" in nodes_src
  assert 'if not source_enabled(sources, raw[' in nodes_src


def test_fanout_deferred_examples_600_and_60_disabled() -> None:
  '''§10.2: 600 eligible -> 550; 60 due 10 disabled -> 0.'''
  s10 = _section(ARCH, '### 10.2', '## 11.')
  assert '600 due companies all eligible' in s10
  assert 'deferred_targets = 550' in s10
  assert '60 due with 10 on a disabled source' in s10
  assert 'deferred_targets = 0' in s10


def test_quick_poll_pinned_ats_only_no_aggregator() -> None:
  '''§19: quick_poll is pinned ATS only, no aggregator fetch.'''
  s19 = _section(ARCH, '## 19.', '## 20.')
  assert 'pinned ATS companies only' in s19
  assert "kind='refresh'" in s19
  assert 'priority >= 5' in s19
  assert 'no aggregator fetch' in s19
  assert 'T2 aggregators +' not in ARCH
  # Code truth: scheduler refresh kind, run_manager mapping, nodes filter.
  sched_src = (SRC / 'workers/scheduler.py').read_text(encoding='utf-8')
  assert "kind='refresh'" in sched_src
  assert 'scheduler:quick_poll' in sched_src
  mgr_src = (SRC / 'services/run_manager.py').read_text(encoding='utf-8')
  assert "== 'refresh'" in mgr_src
  nodes_src = (SRC / 'graph/nodes.py').read_text(encoding='utf-8')
  assert "if row.get('priority', 0) >= 5" in nodes_src


def test_app_overview_nightly_all_due_vs_quick_pinned_only() -> None:
  '''Overview step 2: nightly all-due vs quick pinned-only distinguished.'''
  assert 'all due companies' in OVERVIEW
  assert 'pinned-only every 2h' in OVERVIEW
  assert 'priority >= 5' in OVERVIEW
  assert "kind='refresh'" in OVERVIEW
  assert 'no aggregator' in OVERVIEW


def test_enrich_cap_semantics_arch_s15_and_runbook() -> None:
  '''Enrich cap: default 40, attempted before skips, capped remainder.'''
  s15 = _section(ARCH, '## 15.', '## 16.')
  assert 'enrich_per_run_cap' in s15
  assert 'defaults to 40' in s15
  assert 'increments before' in s15
  assert 'skipped attempts' in s15
  assert 'enrich_capped' in s15
  assert 'cap - len(inserted)' in s15
  assert 'jobs_needing_enrichment' in s15
  assert 'defaults to 40' in RUNBOOK
  assert 'enrich_capped' in RUNBOOK
  assert 'cap - len(inserted)' in RUNBOOK
  # Code truth: default, attempted order, capped, backlog shortfall.
  nodes_src = (SRC / 'graph/nodes.py').read_text(encoding='utf-8')
  assert "int(settings.discovery.get('enrich_per_run_cap', 40))" in nodes_src
  assert 'attempted += 1' in nodes_src
  assert 'enrich_skipped_budget' in nodes_src
  assert 'enrich_skipped_short' in nodes_src
  assert 'enrich_capped' in nodes_src
  assert 'shortfall = cap - len(targets)' in nodes_src
  assert 'jobs_needing_enrichment(limit=shortfall)' in nodes_src


def test_test_module_count_dynamic_recount() -> None:
  '''Counts in §4 and test-writer match live tests/unit/test_*.py.'''
  count = len(list((ROOT / 'tests/unit').glob('test_*.py')))
  assert count >= 60, count
  assert f'{count} unit test modules' in ARCH
  assert f'{count} unit test modules' in TEST_WRITER
  assert 'live list is tests/unit/' in ARCH
  assert 'counted dynamically' in TEST_WRITER
