#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''RED tests for cycle-5 graph/db fixes (one per numbered spec item).'''

from __future__ import annotations

import inspect
import sqlite3
import subprocess
import sys
from pathlib import Path

from job_hunter.core.models import NormalizedJob
from job_hunter.graph.state import _normalize_inserted


def _good_dict() -> dict:
  return {
    'raw': {
      'source_key': 'lever',
      'url': 'https://j.example/1',
      'title': 'Good DS',
      'company_name': 'GoodCo',
    },
    'canonical_url': 'https://j.example/1',
    'content_hash': 'hash-good',
    'city': 'Bengaluru',
  }


def test_normalize_inserted_skips_malformed_and_string_items() -> None:
  '''One corrupt item never kills the node; string items are skipped.'''
  malformed = {
    'job_id': 'not-an-int',
    'raw': 'not-a-dict',
    'canonical_url': 'https://j.example/bad',
    'content_hash': 'bad',
  }
  state = {'inserted': [_good_dict(), malformed, 'string-item']}
  result = _normalize_inserted(state)
  assert len(result) == 1
  assert isinstance(result[0], NormalizedJob)
  assert result[0].content_hash == 'hash-good'
  assert state['inserted'] == result


def test_normalize_inserted_flat_fallback_preserves_fields() -> None:
  '''Flat fallback keeps city/work_mode/salary and raw fields.'''
  flat = {
    'job_id': 7,
    'canonical_url': 'https://j.example/flat',
    'content_hash': 'hash-flat',
    'company_id': 3,
    'city': 'Pune',
    'region': 'MH',
    'country': 'IN',
    'work_mode': 'remote',
    'employment_type': 'contract',
    'salary_min_lpa': 40.0,
    'salary_max_lpa': 60.0,
    'source_key': 'lever',
    'external_id': 'ext-1',
    'url': 'https://j.example/flat',
    'title': 'Flat Data Scientist',
    'company_name': 'FlatCo',
    'location_text': 'Pune, India',
    'description_text': 'flat description',
    'description_html': '<p>flat</p>',
    'posted_at': '2026-08-01T00:00:00+00:00',
    'employment_type_raw': 'Contract',
    'work_mode_hint': 'remote',
    'salary_raw': '40-60 LPA',
  }
  result = _normalize_inserted({'inserted': [flat]})
  assert len(result) == 1
  job = result[0]
  assert job.job_id == 7
  assert job.company_id == 3
  assert job.city == 'Pune'
  assert job.region == 'MH'
  assert job.country == 'IN'
  assert job.work_mode == 'remote'
  assert job.employment_type == 'contract'
  assert job.salary_min_lpa == 40.0
  assert job.salary_max_lpa == 60.0
  assert job.raw.title == 'Flat Data Scientist'
  assert job.raw.company_name == 'FlatCo'
  assert job.raw.source_key == 'lever'
  assert job.raw.external_id == 'ext-1'
  assert job.raw.url == 'https://j.example/flat'
  assert job.raw.location_text == 'Pune, India'
  assert job.raw.description_text == 'flat description'
  assert job.raw.description_html == '<p>flat</p>'
  assert job.raw.posted_at == '2026-08-01T00:00:00+00:00'
  assert job.raw.employment_type_raw == 'Contract'
  assert job.raw.work_mode_hint == 'remote'
  assert job.raw.salary_raw == '40-60 LPA'


def test_runs_create_and_has_active_docstrings_mark_deprecated() -> None:
  '''Deprecated test-only helpers carry the deprecation docstring.'''
  from job_hunter.db.repositories.runs import RunsRepository

  phrase = (
    'Deprecated: test-only; production must use '
    'create_if_not_active (atomic).'
  )
  assert phrase in (inspect.getdoc(RunsRepository.create) or '')
  assert phrase in (inspect.getdoc(RunsRepository.has_active) or '')


def test_cache_connect_closes_conn_on_pragma_failure(monkeypatch) -> None:
  '''A pragma failure in cache _connect closes the connection.'''
  import llm_gateway.cache as cache_mod

  class FakeConn:
    '''Connection whose execute raises on PRAGMA.'''

    def __init__(self) -> None:
      self.closed = False

    def execute(self, sql, *args, **kwargs):
      '''Raise on any PRAGMA like a locked/corrupt database.'''
      raise sqlite3.OperationalError('pragma boom')

    def close(self) -> None:
      '''Record close.'''
      self.closed = True

  fake = FakeConn()
  monkeypatch.setattr(cache_mod.sqlite3, 'connect', lambda *a, **k: fake)
  try:
    cache_mod._connect(':memory:')
  except sqlite3.OperationalError:
    pass
  else:
    raise AssertionError('expected OperationalError from pragma')
  assert fake.closed is True


def test_db_connect_retries_wal_on_operational_error(
  tmp_path: Path, monkeypatch,
) -> None:
  '''core.db connect retries WAL pragma once and then succeeds.'''
  import job_hunter.core.db as db_mod

  class FlakyConn:
    '''Connection failing the first WAL pragma only.'''

    def __init__(self) -> None:
      self.wal_attempts = 0
      self.closed = False
      self.row_factory = None

    def execute(self, sql, *args, **kwargs):
      '''Raise OperationalError on first WAL, succeed afterwards.'''
      if 'journal_mode=WAL' in sql:
        self.wal_attempts += 1
        if self.wal_attempts == 1:
          raise sqlite3.OperationalError('database is locked')

    def close(self) -> None:
      '''Mark closed.'''
      self.closed = True

  flaky = FlakyConn()
  monkeypatch.setattr(db_mod.sqlite3, 'connect', lambda *a, **k: flaky)
  conn = db_mod.connect(tmp_path / 'app.db')
  assert conn is flaky
  assert flaky.wal_attempts == 2
  conn.close()


def test_f821_no_undefined_names_in_matcher_and_client() -> None:
  '''ruff F821 is clean for matcher.py and llm/client.py.'''
  root = Path(__file__).resolve().parents[2]
  matcher = (
    root
    / 'src'
    / 'job_hunter'
    / 'job_hunter'
    / 'services'
    / 'matcher.py'
  )
  client = root / 'src' / 'job_hunter' / 'job_hunter' / 'llm' / 'client.py'
  result = subprocess.run(
    [sys.executable, '-m', 'ruff', 'check', '--select', 'F821',
     str(matcher), str(client)],
    capture_output=True,
    text=True,
    cwd=str(root),
    check=False,
  )
  if result.returncode != 0 and 'No module named ruff' in result.stderr:
    result = subprocess.run(
      ['ruff', 'check', '--select', 'F821', str(matcher), str(client)],
      capture_output=True,
      text=True,
      cwd=str(root),
      check=False,
    )
  assert result.returncode == 0, result.stdout + result.stderr
