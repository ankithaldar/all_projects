#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle-9 docs pins: priority, count, one-line guards, resume raise.'''

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS_UNIT = ROOT / 'tests' / 'unit'

PRIORITY_PHRASE = (
  'non-numeric/non-finite import priority increments errors; '
  'numeric truncated via int() then clamped 1..5 '
  '(missing/None->3; 0->1 (clamped) on import/seed)'
)


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


def _recount_number() -> int:
  '''Count unit test modules on disk.

  Returns:
    Number of tests/unit/test_*.py files.
  '''
  return len(list(TESTS_UNIT.glob('test_*.py')))


def test_docs_test_module_counts_match_recount() -> None:
  '''Docs state the live test-module recount (dynamic pin).'''
  count = _recount_number()
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  writer = (ROOT / 'agents' / 'test-writer.md').read_text(encoding='utf-8')
  assert f'{count} unit test modules' in arch
  assert f'{count} unit test modules' in writer
  assert 'live list is tests/unit/' in arch
  assert 'live list is every module in `tests/unit/`' in writer


def test_docs_priority_wording_arch_12_17_and_api() -> None:
  '''§12/§17 and api.md state missing/None->3; 0->1 clamped phrase.'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s12 = _section(arch, '## 12.', '## 13.')
  assert 'missing/None->3; 0->1 (clamped)' in s12
  s17 = _section(arch, '## 17.', '## 18.')
  assert PRIORITY_PHRASE in s17
  api = (ROOT / 'docs' / 'api.md').read_text(encoding='utf-8')
  assert PRIORITY_PHRASE in api
  for text in (arch, api):
    assert 'missing/falsy' not in text
    assert 'non-integer' not in text


def test_docs_send_error_501_headers_api_dashboard() -> None:
  '''api.md dashboard note covers send_error 501 security headers.'''
  api = (ROOT / 'docs' / 'api.md').read_text(encoding='utf-8')
  assert 'send_error' in api
  assert '501' in api
  assert 'security headers' in api
  assert 'X-Content-Type-Options' in api
  assert 'X-Frame-Options' in api


def test_docs_taxonomy_non_dict_guard_14() -> None:
  '''§14 notes the non-dict skill-spec guard.'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s14 = _section(arch, '## 14.', '## 15.')
  assert 'skipping non-dict skill spec' in s14


def test_docs_score_int_guard_15_and_20() -> None:
  '''§15 and §20 note the non-int job_id score guard.'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s15 = _section(arch, '## 15.', '## 16.')
  assert 'skipping job with non-int job_id' in s15
  s20 = _section(arch, '## 20.', '## 21.')
  assert 'skipping job with non-int job_id' in s20


def test_docs_processed_skipped_on_scan_11() -> None:
  '''§11 T4 line states _processed files are skipped on scan.'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s11 = _section(arch, '## 11.', '## 12.')
  t4_line = next(
    (line for line in s11.splitlines() if 'T4 manual export inbox' in line),
    '',
  )
  assert t4_line, 'T4 row not found'
  assert 'skipped on scan' in t4_line
  assert '_processed' in t4_line
  assert 'rescan-dedupe' not in t4_line


def test_docs_dangling_resume_raise_14_and_api() -> None:
  '''§14 and api resume row note the dangling-resume RuntimeError.'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s14 = _section(arch, '## 14.', '## 15.')
  assert 'resume row missing after save_resume_ex' in s14
  assert 'RuntimeError' in s14
  api = (ROOT / 'docs' / 'api.md').read_text(encoding='utf-8')
  resume_row = next(
    (line for line in api.splitlines() if '/api/profile/resume' in line), '',
  )
  assert resume_row, 'resume row not found'
  assert 'resume row missing after save_resume_ex' in resume_row
  assert 'RuntimeError' in resume_row
