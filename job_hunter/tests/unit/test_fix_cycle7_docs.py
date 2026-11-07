#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle-7 docs/web pins: throttle rstrip, priority clamp, MCP2, a11y.'''


from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src' / 'job_hunter'


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


def test_docs_throttle_snippet_rstrips_trailing_dot() -> None:
  '''§11 throttle snippet strips trailing dots so hosts share a throttle.'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s11 = _section(arch, '## 11.', '## 12.')
  assert ".rstrip('.')" in s11
  assert 'lower' in s11.lower()


def test_docs_priority_clamp_arch_17_and_api_companies() -> None:
  '''§17 and api.md state the priority clamp and import error rule.'''
  phrase = (
    'priority clamped 1..5 on create/patch/import; '
    'non-numeric/non-finite import priority increments errors; '
    'numeric truncated via int() then clamped 1..5 '
    '(missing/None->3; 0->1 (clamped) on import/seed)'
  )
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s17 = _section(arch, '## 17.', '## 18.')
  assert phrase in s17
  api = (ROOT / 'docs' / 'api.md').read_text(encoding='utf-8')
  assert phrase in api


def test_docs_mcp2_fallback_all_three_in_arch_8() -> None:
  '''§8 MCP 2.x fallback sentence covers all three servers.'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s8 = _section(arch, '## 8.', '## 9.')
  assert 'all three' in s8.lower()
  assert 'sources_server.py' in s8
  assert 'resume_server.py' in s8
  assert 'store_server.py' in s8
  assert 'MCPServer' in s8


def test_dashboard_summary_cards_plain_div_no_role_status() -> None:
  '''Polling summary-cards container is a plain div, not a live region.'''
  html = (
    SRC / 'llm_gateway' / 'templates' / 'dashboard.html'
  ).read_text(encoding='utf-8')
  summary_line = next(
    (line for line in html.splitlines() if 'summary-cards' in line), '',
  )
  assert summary_line, 'summary-cards container not found'
  assert "role='status'" not in summary_line
  assert 'role="status"' not in summary_line
  assert 'aria-live' not in summary_line
  assert summary_line.strip().startswith('<div')
  dashboard_js = (
    SRC / 'llm_gateway' / 'templates' / 'dashboard.js'
  ).read_text(encoding='utf-8')
  assert 'aria-live' not in dashboard_js


def test_docs_checkpoint_10_1_pragmas_still_accurate() -> None:
  '''§10.1 checkpoint pragmas incl. foreign_keys=ON remain documented.'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s101 = _section(arch, '### 10.1', '### 10.2')
  assert 'journal_mode=WAL' in s101
  assert 'busy_timeout=5000' in s101
  assert 'foreign_keys=ON' in s101


def test_docs_settings_put_four_keys_still_accurate() -> None:
  '''§17 and api.md list exactly the 4 PUT-able settings keys.'''
  keys = (
    'scoring_weights',
    'salary_hard_floor_lpa',
    'discovery.max_llm_calls_per_run',
    'discovery.enrich_per_run_cap',
  )
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s17 = _section(arch, '## 17.', '## 18.')
  for key in keys:
    assert key in s17, key
  api = (ROOT / 'docs' / 'api.md').read_text(encoding='utf-8')
  for key in keys:
    assert key in api, key
