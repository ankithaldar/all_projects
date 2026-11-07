#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle-8 docs pins: counts, priority, seeds, duplicate, identity log.'''

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS_UNIT = ROOT / 'tests' / 'unit'


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
  '''Docs state the live test-module recount (live-list hedge kept).'''
  count = _recount_number()
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  writer = (ROOT / 'agents' / 'test-writer.md').read_text(encoding='utf-8')
  assert f'{count} unit test modules' in arch
  assert f'{count} unit test modules' in writer
  assert 'live list is tests/unit/' in arch
  assert 'live list is every module in `tests/unit/`' in writer


def test_docs_priority_wording_arch_17_and_api() -> None:
  '''§17 and api.md state non-numeric/non-finite + int() + clamp 1..5.'''
  phrase = (
    'non-numeric/non-finite import priority increments errors; '
    'numeric truncated via int() then clamped 1..5 '
    '(missing/None->3; 0->1 (clamped) on import/seed)'
  )
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s17 = _section(arch, '## 17.', '## 18.')
  assert phrase in s17
  assert 'clamped' in s17
  assert 'non-numeric' in s17
  api = (ROOT / 'docs' / 'api.md').read_text(encoding='utf-8')
  assert phrase in api
  assert 'clamped' in api
  assert 'non-numeric' in api


def test_docs_seed_clamp_sentence_in_12() -> None:
  '''§12 notes seeds clamped 1..5 and bad priority skipped w/ warning.'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s12 = _section(arch, '## 12.', '## 13.')
  assert 'clamped 1..5' in s12
  lowered = s12.lower()
  assert 'bad priority' in lowered
  assert 'skipped' in lowered
  assert 'warning' in lowered


def test_docs_duplicate_sha_note_api_resume_and_14() -> None:
  '''api resume row and §14 note duplicate sha -> stored path.'''
  api = (ROOT / 'docs' / 'api.md').read_text(encoding='utf-8')
  resume_row = next(
    (line for line in api.splitlines() if '/api/profile/resume' in line), '',
  )
  assert resume_row, 'resume row not found'
  assert 'duplicate' in resume_row.lower()
  assert 'resume_id' in resume_row
  assert 'stored path' in resume_row.lower()
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s14 = _section(arch, '## 14.', '## 15.')
  assert 'duplicate' in s14.lower()
  assert 'resume_id' in s14
  assert 'stored path' in s14.lower()


def test_docs_identity_log_string_in_20() -> None:
  '''§20 quotes the exact identity-log string from state.py.'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s20 = _section(arch, '## 20.', '## 21.')
  assert 'skipping inserted item without identity' in s20
