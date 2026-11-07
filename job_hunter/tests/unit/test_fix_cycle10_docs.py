#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle-10 pins: bool priority, taxonomy guards, row-first, API guards.'''


from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip('fastapi')
TestClient = pytest.importorskip('fastapi.testclient').TestClient

ROOT = Path(__file__).resolve().parents[2]

BOOL_PHRASE = (
  'bool priority is rejected, not clamped '
  '(422 create/patch; errors import; skipped seed)'
)

HUGE_INT = 10**400


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


def _companies_rows(text: str) -> list[str]:
  '''Return the priority-bearing companies rows from a docs table.

  Only create/import/patch rows carry the priority sentence; the
  DELETE and planned-refresh rows are intentionally excluded.

  Args:
    text: Document text.

  Returns:
    Lines listing priority-bearing company endpoints.
  '''
  return [
    line for line in text.splitlines()
    if '/api/companies' in line and 'priority' in line
  ]


@pytest.fixture(name='client')
def _client_fixture(tmp_path: Path, monkeypatch) -> TestClient:
  '''Build the app against a temporary data directory.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest monkeypatch fixture.

  Returns:
    Configured TestClient (server exceptions surfaced as responses).
  '''
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  config_path = tmp_path / 'app.yaml'
  config_path.write_text('salary_hard_floor_lpa: 45\n', encoding='utf-8')
  from job_hunter.api.main import create_app
  app = create_app(str(config_path))
  client = TestClient(app, raise_server_exceptions=False)
  yield client
  client.close()


def test_docs_bool_priority_sentence_12_17_and_api() -> None:
  '''Bool priority sentence lives in §12, §17 and api companies rows.'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s12 = _section(arch, '## 12.', '## 13.')
  assert BOOL_PHRASE in s12
  s17 = _section(arch, '## 17.', '## 18.')
  assert BOOL_PHRASE in s17
  for row in _companies_rows(s17):
    assert BOOL_PHRASE in row, row
  api = (ROOT / 'docs' / 'api.md').read_text(encoding='utf-8')
  for row in _companies_rows(api):
    assert BOOL_PHRASE in row, row


def test_docs_int_first_clamp_phrase_still_present() -> None:
  '''Int-first clamp outcome (huge int -> 5/1) stays documented.'''
  phrase = 'numeric truncated via int() then clamped 1..5'
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s12 = _section(arch, '## 12.', '## 13.')
  # §12 wraps across lines, so compare with whitespace collapsed.
  assert phrase in ' '.join(s12.split())
  s17 = _section(arch, '## 17.', '## 18.')
  assert phrase in s17
  api = (ROOT / 'docs' / 'api.md').read_text(encoding='utf-8')
  assert phrase in api


def test_docs_taxonomy_guards_14() -> None:
  '''§14 lists the full taxonomy guard set.'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s14 = _section(arch, '## 14.', '## 15.')
  assert 'skipping non-dict skill spec' in s14
  assert 'non-dict payload' in s14
  assert 'returns 0' in s14
  assert 'non-str name' in s14
  assert 'not counted' in s14
  assert 'non-list aliases' in s14
  assert 'empty' in s14
  assert 'warn' in s14
  assert 'non-str aliases' in s14
  assert 'skipped' in s14


def test_docs_row_first_14_and_api_resume_row() -> None:
  '''Row-first sentence lives in §14 and the api resume row.'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s14 = _section(arch, '## 14.', '## 15.')
  assert 'deletes owned resume row first' in s14
  assert 'file kept if' in s14
  api = (ROOT / 'docs' / 'api.md').read_text(encoding='utf-8')
  resume_row = next(
    (line for line in api.splitlines() if '/api/profile/resume' in line), '',
  )
  assert resume_row, 'resume row not found'
  assert 'deletes owned resume row first' in resume_row
  assert 'file kept if' in resume_row


def test_profile_skills_non_str_422(client: TestClient) -> None:
  '''Skills API rejects a non-str name with 422.'''
  response = client.put('/api/profile/skills', json={'names': [123]})
  assert response.status_code == 422, response.text


def test_settings_scoring_weights_huge_int_422(
  client: TestClient,
) -> None:
  '''Scoring weights reject a huge int without overflowing float().'''
  response = client.put(
    '/api/settings?key=scoring_weights', json={'w': HUGE_INT},
  )
  assert response.status_code == 422, response.text


def test_settings_salary_floor_huge_int_422(client: TestClient) -> None:
  '''Salary floor rejects a huge int without overflowing float().'''
  response = client.put(
    '/api/settings?key=salary_hard_floor_lpa', json=HUGE_INT,
  )
  assert response.status_code == 422, response.text


def test_raw_infinity_priority_422(client: TestClient) -> None:
  '''Raw Infinity priority is a clean 422, not a 500.'''
  headers = {'Content-Type': 'application/json'}
  response = client.post(
    '/api/companies',
    content='{"name": "Inf Co", "domain": "inf.example.com",'
    ' "priority": Infinity}',
    headers=headers,
  )
  assert response.status_code == 422, response.text


def test_inbox_db_failure_no_rename(
  tmp_path: Path, monkeypatch,
) -> None:
  '''A DB failure in _insert_records leaves the inbox file in place.'''
  import asyncio

  import job_hunter.adapters.manual_inbox as inbox_mod
  from job_hunter.core.config import AppSettings
  from job_hunter.core.db import run_migrations

  async def _boom(settings, records):
    raise RuntimeError('db boom')

  monkeypatch.setattr(inbox_mod, '_insert_records', _boom)
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  inbox = tmp_path / 'inbox'
  inbox.mkdir(parents=True, exist_ok=True)
  run_migrations(tmp_path / 'app.db')
  source = inbox / 'saved.csv'
  source.write_text(
    'Job Title,Company,Location\nStuff,Acme Corp,Mumbai\n',
    encoding='utf-8',
  )
  config_path = tmp_path / 'app.yaml'
  config_path.write_text('salary_hard_floor_lpa: 45\n', encoding='utf-8')
  settings = AppSettings(config_path)
  count = asyncio.run(inbox_mod.scan_inbox(settings))
  assert count == 0, count
  assert source.exists(), 'inbox file must stay after DB failure'
  processed = inbox / '_processed'
  moved = list(processed.iterdir()) if processed.exists() else []
  assert moved == [], moved
