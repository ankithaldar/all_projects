#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle 13 backend fixes: fan-out rotation, bulk_fetch enablement,
raw_jobs normalization, constants move, huge-int path params.'''

from __future__ import annotations

import asyncio
import logging
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

fastapi = pytest.importorskip('fastapi')
TestClient = pytest.importorskip('fastapi.testclient').TestClient

_HUGE = '1' + '0' * 400
_MAX_I64 = '9223372036854775807'


def _settings(tmp_path: Path, monkeypatch, *, extra: str = ''):
  '''Build migrated settings for direct service tests.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest fixture.
    extra: Extra YAML appended to the config.

  Returns:
    Migrated AppSettings.
  '''
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  config_path = tmp_path / 'app.yaml'
  config_path.write_text(
    'salary_hard_floor_lpa: 45\n' + extra, encoding='utf-8',
  )
  from job_hunter.core.config import AppSettings
  from job_hunter.core.db import run_migrations
  settings = AppSettings(config_path)
  settings.data_dir.mkdir(parents=True, exist_ok=True)
  run_migrations(settings.db_path)
  return settings


def _make_client(tmp_path: Path, monkeypatch) -> TestClient:
  '''Build an app client against a temporary data directory.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest monkeypatch fixture.

  Returns:
    Configured TestClient.
  '''
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  monkeypatch.setenv('APP_HOST', '127.0.0.1')
  monkeypatch.setenv('JH_API_TOKEN', '')
  monkeypatch.setenv('APP_LOG_LEVEL', 'INFO')
  config_path = tmp_path / 'app.yaml'
  config_path.write_text(
    'salary_hard_floor_lpa: 45\n', encoding='utf-8',
  )
  from job_hunter.api.main import create_app
  app = create_app(str(config_path))
  return TestClient(app, raise_server_exceptions=False)


def _seed_company(
  db_path: Path, name: str, last_checked: str | None,
  *, status: str = 'active', priority: int = 3,
) -> int:
  '''Insert one company row directly (preserving last_checked_at).

  Args:
    db_path: Database path.
    name: Company name.
    last_checked: last_checked_at value or None.
    status: Lifecycle status.
    priority: Priority value.

  Returns:
    New company id.
  '''
  with closing(sqlite3.connect(str(db_path))) as conn:
    cursor = conn.execute(
      'INSERT INTO companies (name, normalized_name, domain, '
      'ats_provider, board_ref, priority, status, last_checked_at) '
      'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
      (
        name, name.lower(), f'{name.lower()}.example.com',
        'lever', f'board-{name.lower()}', priority, status, last_checked,
      ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def _seed_60_companies(db_path: Path) -> list[int]:
  '''Seed 60 active companies with staggered oldest-first timestamps.

  Args:
    db_path: Database path.

  Returns:
    Company ids in oldest-first order.
  '''
  return [
    _seed_company(db_path, f'Co{i:02d}', f'2026-01-01 00:00:{i:02d}')
    for i in range(60)
  ]


class _EmptyAdapter:
  '''Adapter returning no records (fetch succeeds, checks the company).'''

  def __init__(self, http) -> None:
    '''Accept the shared client.

    Args:
      http: Unused client.
    '''
    _ = http

  async def fetch(self, target, limit=200):
    '''Return no records.

    Args:
      target: Target.
      limit: Cap.

    Returns:
      Empty record list.
    '''
    _ = (target, limit)
    return []

  async def health(self, target):
    '''Always healthy.

    Args:
      target: Target.

    Returns:
      True.
    '''
    _ = target
    return True


def _stub_adapters(monkeypatch) -> None:
  '''Route every build_adapter call to the empty stub adapter.

  Args:
    monkeypatch: Pytest fixture.
  '''
  import job_hunter.graph.nodes as nodes_mod

  def stub(key, http):
    '''Return the empty stub for any source key.

    Args:
      key: Source key.
      http: Shared client.

    Returns:
      Empty stub adapter.
    '''
    _ = key
    return _EmptyAdapter(http)

  monkeypatch.setattr(nodes_mod, 'build_adapter', stub)


# 1. Fan-out rotation ----------------------------------------------------


def test_list_due_orders_oldest_first_nulls_first(
  tmp_path: Path, monkeypatch,
) -> None:
  '''list_due returns active companies oldest-first, NULL stamps first.'''
  settings = _settings(tmp_path, monkeypatch)
  newest = _seed_company(settings.db_path, 'Newest', '2026-01-03 00:00:00')
  never = _seed_company(settings.db_path, 'Never', None)
  oldest = _seed_company(settings.db_path, 'Oldest', '2026-01-01 00:00:00')
  _seed_company(
    settings.db_path, 'Paused', '2025-01-01 00:00:00', status='paused',
  )
  from job_hunter.db.repositories.companies import CompaniesRepository
  rows = CompaniesRepository(settings.db_path).list_due(10)
  assert [row['id'] for row in rows] == [never, oldest, newest]


def test_build_plan_rotation_with_deferred_targets(
  tmp_path: Path, monkeypatch,
) -> None:
  '''60 due companies: first run defers 10; after fetching the first 50,
  the next plan rotates to the deferred batch.'''
  settings = _settings(tmp_path, monkeypatch)
  ids = _seed_60_companies(settings.db_path)
  _stub_adapters(monkeypatch)
  import job_hunter.graph.nodes as nodes_mod

  config = {'configurable': {'settings': settings}}

  def run_plan():
    '''Run build_plan and return the planned company ids.'''
    update = asyncio.run(
      nodes_mod.build_plan(
        {'run_id': 1, 'plan': {'quick_poll': False}}, config,
      ),
    )
    assert 'errors' not in update or not update['errors'], update
    targets = update['plan']['targets']
    return update, [int(t['company_id']) for t in targets]

  first_update, first_ids = run_plan()
  assert first_ids[:50] == ids[:50], first_ids[:5]
  assert first_update['stats']['deferred_targets'] == 10, first_update
  first_fifty = first_ids[:50]

  async def fetch_first_fifty() -> None:
    '''Fetch each planned target once (marks companies checked).'''
    for company_id in first_fifty:
      target = next(
        t for t in first_update['plan']['targets']
        if int(t['company_id']) == company_id
      )
      await nodes_mod.fetch_pair(target, config)

  asyncio.run(fetch_first_fifty())

  _second_update, second_ids = run_plan()
  assert second_ids[:50] != first_ids[:50], second_ids[:5]
  assert second_ids[0] == ids[50], second_ids[:3]


def test_fetch_pair_marks_company_checked(
  tmp_path: Path, monkeypatch,
) -> None:
  '''fetch_pair stamps companies.last_checked_at on success.'''
  settings = _settings(tmp_path, monkeypatch)
  company_id = _seed_company(
    settings.db_path, 'Checked', '2020-01-01 00:00:00',
  )
  _stub_adapters(monkeypatch)
  import job_hunter.graph.nodes as nodes_mod

  update = asyncio.run(
    nodes_mod.fetch_pair(
      {
        'company_id': company_id, 'name': 'Checked',
        'source_key': 'lever', 'board_ref': 'board-checked',
      },
      {'configurable': {'settings': settings}},
    ),
  )
  assert not update['errors'], update
  from job_hunter.db.repositories.companies import CompaniesRepository
  row = CompaniesRepository(settings.db_path).get(company_id)
  assert row is not None
  assert row['last_checked_at'] not in (None, '2020-01-01 00:00:00'), row


# 2. bulk_fetch enablement -------------------------------------------------


def test_resolve_sources_filters_disabled_defaults(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Disabled sources drop out of the default fetch; an explicit
  request still overrides the disabled flag.'''
  settings = _settings(
    tmp_path, monkeypatch,
    extra='sources:\n  remotive:\n    enabled: false\n',
  )
  from job_hunter.services.bulk_fetch import resolve_sources
  defaults = resolve_sources([], settings)
  assert 'remotive' not in defaults, defaults
  assert 'arbeitnow' in defaults, defaults
  # Explicit --sources overrides the enabled flag (documented contract).
  assert resolve_sources(['remotive'], settings) == ['remotive']
  # Backwards compatible: no settings means unfiltered defaults.
  assert 'remotive' in resolve_sources([])


def test_bulk_fetch_default_skips_disabled_source(
  tmp_path: Path, monkeypatch,
) -> None:
  '''bulk_fetch with no explicit sources never fetches a disabled one.'''
  settings = _settings(
    tmp_path, monkeypatch,
    extra='sources:\n  remotive:\n    enabled: false\n',
  )
  from job_hunter.services import bulk_fetch as bulk_mod

  seen: list[str] = []

  async def fake_fetch_source(http, source_key, limit):
    '''Record the source and return nothing.

    Args:
      http: Unused client.
      source_key: Source being fetched.
      limit: Cap.

    Returns:
      Empty record list.
    '''
    _ = (http, limit)
    seen.append(source_key)
    return []

  class _FakeHttp:
    '''No-op HTTP client.'''

    async def close(self) -> None:
      '''Do nothing.'''

  monkeypatch.setattr(bulk_mod, 'fetch_source', fake_fetch_source)
  asyncio.run(bulk_mod.bulk_fetch(settings, http=_FakeHttp(), persist=False))
  assert 'remotive' not in seen, seen
  assert 'arbeitnow' in seen, seen


# 3. raw_jobs normalization ------------------------------------------------


def test_normalize_raw_jobs_dict_to_objects_skips_malformed(
  caplog,
) -> None:
  '''Dict raw_jobs become RawJobRecords; malformed items are skipped.'''
  from job_hunter.core.models import RawJobRecord
  from job_hunter.graph.state import _normalize_raw_jobs

  existing = RawJobRecord(
    source_key='lever', url='https://j.example/9', title='Existing',
  )
  state = {
    'raw_jobs': [
      {
        'source_key': 'lever', 'url': 'https://j.example/1',
        'title': 'DS', 'company_name': 'Acme',
      },
      {'title': 'no url, no source'},  # malformed: missing required fields
      'not-a-dict',
      existing,
    ],
  }
  with caplog.at_level(logging.WARNING):
    normalized = _normalize_raw_jobs(state)
  assert len(normalized) == 2, normalized
  assert all(
    isinstance(record, RawJobRecord) for record in normalized
  ), normalized
  assert state['raw_jobs'] == normalized
  assert normalized[0].title == 'DS'
  assert any(
    record.levelno >= logging.WARNING for record in caplog.records
  ), [r.getMessage() for r in caplog.records]


def test_normalize_dedupe_accepts_dict_raw_jobs(
  tmp_path: Path, monkeypatch,
) -> None:
  '''normalize_dedupe persists dict-shaped raw_jobs (checkpoint form).'''
  settings = _settings(tmp_path, monkeypatch)
  import job_hunter.graph.nodes as nodes_mod

  state = {
    'run_id': 1,
    'raw_jobs': [
      {
        'source_key': 'lever', 'url': 'https://j.example/dict-1',
        'title': 'Staff Data Scientist', 'company_name': 'DictCo',
        'description_text': 'python sql experimentation 55 LPA',
      },
      {'title': 'malformed'},  # skipped with a warning, not an error
    ],
  }
  update = asyncio.run(
    nodes_mod.normalize_dedupe(
      state, {'configurable': {'settings': settings}},
    ),
  )
  assert 'errors' not in update or not update['errors'], update
  assert update['stats']['new_jobs'] == 1, update
  assert len(update['inserted']) == 1, update


# 4. Constants move ---------------------------------------------------------


def test_constants_live_in_core_and_scorer_reexports() -> None:
  '''core.constants is the single home; scorer re-exports the same
  objects; owned core files no longer import from services.'''
  from job_hunter.core.constants import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_SALARY_FLOOR_LPA,
    DEFAULT_WEIGHTS,
  )
  from job_hunter.services.scorer import (
    DEFAULT_EMBED_MODEL as scorer_model,
  )
  from job_hunter.services.scorer import (
    DEFAULT_SALARY_FLOOR_LPA as scorer_floor,
  )
  from job_hunter.services.scorer import DEFAULT_WEIGHTS as scorer_weights
  assert DEFAULT_SALARY_FLOOR_LPA == 45.0
  assert DEFAULT_EMBED_MODEL == 'BAAI/bge-small-en-v1.5'
  assert DEFAULT_WEIGHTS['skills_must'] == 0.30
  assert scorer_floor is DEFAULT_SALARY_FLOOR_LPA
  assert scorer_model is DEFAULT_EMBED_MODEL
  assert scorer_weights is DEFAULT_WEIGHTS

  core_dir = (
    Path(__file__).resolve().parents[2]
    / 'src/job_hunter/job_hunter/core'
  )
  for name in ('constants.py', 'config.py', 'models.py'):
    text = (core_dir / name).read_text(encoding='utf-8')
    assert 'from job_hunter.services' not in text, name
    assert 'import job_hunter.services' not in text, name


# 5. Huge-int path/query params ---------------------------------------------


def test_huge_int_params_422(tmp_path: Path, monkeypatch) -> None:
  '''10**400 in page/after_id/path ids is rejected with 422.'''
  client = _make_client(tmp_path, monkeypatch)
  cases = [
    ('GET', f'/api/recommendations?page={_HUGE}'),
    ('GET', f'/api/jobs/{_HUGE}'),
    ('GET', f'/api/runs/1/events?after_id={_HUGE}'),
    ('GET', f'/api/runs/{_HUGE}/events'),
    ('POST', f'/api/runs/{_HUGE}/cancel'),
  ]
  for method, url in cases:
    response = client.request(method, url)
    assert response.status_code == 422, (method, url, response.text)
  patch = client.patch(f'/api/companies/{_HUGE}', json={})
  assert patch.status_code == 422, patch.text
  delete = client.request('DELETE', f'/api/companies/{_HUGE}')
  assert delete.status_code == 422, delete.text
  rec_patch = client.patch(
    f'/api/recommendations/{_HUGE}?status=saved',
  )
  assert rec_patch.status_code == 422, rec_patch.text


def test_int64_boundary_ids_are_not_422(tmp_path: Path, monkeypatch) -> None:
  '''2**63-1 stays valid (404), proving the bound is inclusive.'''
  client = _make_client(tmp_path, monkeypatch)
  job = client.get(f'/api/jobs/{_MAX_I64}')
  assert job.status_code == 404, job.text
  over = client.get(f'/api/jobs/{int(_MAX_I64) + 1}')
  assert over.status_code == 422, over.text


# 6. Docs overclaim code counterparts ---------------------------------------


def test_nodes_has_no_escalation_or_contextvar_overclaim() -> None:
  '''Owned graph code makes no watchdog failed-escalation or
  source_key contextvar claim; docs files are out of scope here.'''
  nodes_path = (
    Path(__file__).resolve().parents[2]
    / 'src/job_hunter/job_hunter/graph/nodes.py'
  )
  text = nodes_path.read_text(encoding='utf-8')
  assert 'escalat' not in text.lower()
  assert 'contextvar' not in text.lower()
