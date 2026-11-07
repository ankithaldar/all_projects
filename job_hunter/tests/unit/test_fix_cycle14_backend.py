#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle 14 backend fixes (W1-backend).

Covers paren-except normalization, dead-code removal, settings-cache
dedup, source_enabled single home, MAX_SQLITE_ID single home with
negative-huge 422, clamp_priority single home, list_due ATS + cooled
filtering, upsert last_checked_at preservation, bulk empty/failed
split, and config docstring accuracy.
'''

from __future__ import annotations

import asyncio
import inspect
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

fastapi = pytest.importorskip('fastapi')
TestClient = pytest.importorskip('fastapi.testclient').TestClient

_HUGE = '1' + '0' * 400
_NEG_HUGE = '-1' + '0' * 400


def _settings(tmp_path: Path, monkeypatch, *, extra: str = ''):
  '''Build migrated settings for service tests.

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


def _seed_company_raw(
  db_path: Path, name: str, last_checked: str | None,
  *, ats: str | None = 'lever', board: object = 'auto',
  priority: int = 3, status: str = 'active',
) -> int:
  '''Insert one company row directly.

  Args:
    db_path: Database path.
    name: Company name.
    last_checked: last_checked_at value or None.
    ats: ATS provider or None.
    board: Board ref, None for NULL, or 'auto' for a generated ref.
    priority: Priority value.
    status: Lifecycle status.

  Returns:
    New company id.
  '''
  if board == 'auto':
    board = f'board-{name.lower()}'
  with closing(sqlite3.connect(str(db_path))) as conn:
    cursor = conn.execute(
      'INSERT INTO companies (name, normalized_name, domain, '
      'ats_provider, board_ref, priority, status, last_checked_at) '
      'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
      (
        name, name.lower(), f'{name.lower()}.example.com',
        ats, board, priority, status, last_checked,
      ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def _cool_company(db_path: Path, company_id: int, hours: int = 48) -> None:
  '''Mark one company scope as cooled down.

  Args:
    db_path: Database path.
    company_id: Company id.
    hours: Cooldown length in hours.
  '''
  with closing(sqlite3.connect(str(db_path))) as conn:
    conn.execute(
      'INSERT INTO crawl_state (scope, cooldown_until) '
      'VALUES (?, datetime(\'now\', ?)) '
      'ON CONFLICT(scope) DO UPDATE SET cooldown_until = excluded.cooldown_until',
      (f'company:{company_id}', f'+{hours} hours'),
    )
    conn.commit()


# 1. Paren-except normalization --------------------------------------------


def test_except_sites_use_parens() -> None:
  '''scheduler/dedupe use parenthesized multi-except (not comma).'''
  root = (
    Path(__file__).resolve().parents[2]
    / 'src/job_hunter/job_hunter'
  )
  sched = (root / 'workers/scheduler.py').read_text(encoding='utf-8')
  dedupe = (root / 'services/dedupe.py').read_text(encoding='utf-8')
  assert 'except KeyboardInterrupt, SystemExit:' not in sched
  assert 'except (KeyboardInterrupt, SystemExit):' in sched
  assert 'except TypeError, ValueError:' not in dedupe
  assert 'except (TypeError, ValueError):' in dedupe


# 2. Dead-code removal ------------------------------------------------------


def test_nodes_has_no_top_level_json_import() -> None:
  '''nodes.py drops the unused top-level json import.'''
  import job_hunter.graph.nodes as nodes_mod
  src = inspect.getsource(nodes_mod)
  head = '\n'.join(src.splitlines()[:30])
  assert 'import json' not in head, head[:500]


def test_web_search_dead_helpers_removed() -> None:
  '''company_from_result_title/_TITLE_AT_RE/looks_like_domain/_URL_RE gone.'''
  import job_hunter.services.web_search as ws
  assert not hasattr(ws, 'company_from_result_title')
  assert not hasattr(ws, '_TITLE_AT_RE')
  assert not hasattr(ws, 'looks_like_domain')
  assert not hasattr(ws, '_URL_RE')


def test_domain_verifier_dead_helpers_removed() -> None:
  '''verified_site_row/company_domain_map are deleted.'''
  import job_hunter.services.domain_verifier as dv
  assert not hasattr(dv, 'verified_site_row')
  assert not hasattr(dv, 'company_domain_map')


def test_skills_taxonomy_load_seed_file_removed() -> None:
  '''load_seed_file + sole yaml import gone; Path kept (used in __init__).'''
  from job_hunter.services.skills_taxonomy import SkillsTaxonomy
  assert not hasattr(SkillsTaxonomy, 'load_seed_file')
  src = inspect.getsource(__import__(
    'job_hunter.services.skills_taxonomy',
    fromlist=['x'],
  ))
  assert 'import yaml' not in src
  # Path is still used for the db_path annotation in __init__.
  assert 'Path' in src


# 3. Settings-cache dedup ----------------------------------------------------


def test_scheduler_delegates_to_jobs_settings_cache(
  tmp_path: Path, monkeypatch,
) -> None:
  '''scheduler._cached_bootstrap returns jobs._settings object (same id).'''
  import job_hunter.workers.jobs as jobs_mod
  import job_hunter.workers.scheduler as sched_mod
  # Scheduler must not own a duplicate cache dict.
  assert not hasattr(sched_mod, '_SETTINGS_CACHE')
  # Clear jobs cache so the identity check is meaningful.
  cache = getattr(jobs_mod, '_SETTINGS_CACHE', None)
  if isinstance(cache, dict):
    cache.clear()
  config_path = tmp_path / 'app.yaml'
  config_path.write_text('salary_hard_floor_lpa: 45\n', encoding='utf-8')
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  via_sched = sched_mod._cached_bootstrap(config_path)
  via_jobs = jobs_mod._settings(config_path)
  assert via_sched is via_jobs


# 4. source_enabled single home ------------------------------------------------


def test_source_enabled_single_home() -> None:
  '''core.config.source_enabled is the only implementation.'''
  import job_hunter.core.config as cfg
  assert hasattr(cfg, 'source_enabled')
  func = cfg.source_enabled
  assert func({}, 'lever') is True
  assert func({'lever': {'enabled': False}}, 'lever') is False
  assert func({'lever': {'enabled': True}}, 'lever') is True
  assert func({'lever': {}}, 'lever') is True
  assert func(None, 'lever') is True

  import job_hunter.graph.nodes as nodes_mod
  from job_hunter.services import bulk_fetch as bulk_mod
  assert not hasattr(nodes_mod, '_source_enabled')
  assert not hasattr(bulk_mod, '_source_enabled')
  root = (
    Path(__file__).resolve().parents[2]
    / 'src/job_hunter/job_hunter'
  )
  assert 'def _source_enabled' not in (
    root / 'graph/nodes.py'
  ).read_text(encoding='utf-8')
  assert 'def _source_enabled' not in (
    root / 'services/bulk_fetch.py'
  ).read_text(encoding='utf-8')


# 5. MAX_SQLITE_ID single home + negative-huge 422 ----------------------------


def test_max_sqlite_id_single_home() -> None:
  '''core.constants.MAX_SQLITE_ID exists; routers import it (no local).'''
  import job_hunter.core.constants as const
  assert hasattr(const, 'MAX_SQLITE_ID')
  assert const.MAX_SQLITE_ID == 2**63 - 1
  root = (
    Path(__file__).resolve().parents[2]
    / 'src/job_hunter/job_hunter/api/routes'
  )
  for name in ('jobs.py', 'companies.py', 'recommendations.py', 'runs.py'):
    src = (root / name).read_text(encoding='utf-8')
    assert '_MAX_ID =' not in src, name
    assert 'MAX_SQLITE_ID' in src, name


def test_negative_huge_path_ids_422(tmp_path: Path, monkeypatch) -> None:
  '''-10**400 on every Path id returns 422 (ge=-MAX_SQLITE_ID).'''
  client = _make_client(tmp_path, monkeypatch)
  assert client.get(f'/api/jobs/{_NEG_HUGE}').status_code == 422
  assert client.patch(f'/api/companies/{_NEG_HUGE}', json={}).status_code == 422
  assert client.request('DELETE', f'/api/companies/{_NEG_HUGE}').status_code == 422
  assert client.patch(
    f'/api/recommendations/{_NEG_HUGE}?status=saved',
  ).status_code == 422
  assert client.get(f'/api/runs/{_NEG_HUGE}/events').status_code == 422


# 6. clamp_priority single home -----------------------------------------------


def test_clamp_priority_single_home() -> None:
  '''core.constants.clamp_priority owns the 1..5 clamp logic.'''
  import job_hunter.core.constants as const
  assert hasattr(const, 'clamp_priority')
  func = const.clamp_priority
  assert func(10**400) == 5
  assert func(-(10**400)) == 1
  assert func(999) == 5
  assert func(-5) == 1
  assert func(3) == 3
  with pytest.raises(ValueError):
    func(True)
  with pytest.raises(ValueError):
    func(float('inf'))
  with pytest.raises(ValueError):
    func(float('nan'))

  # Private names stay as delegating wrappers (tests import them).
  from job_hunter.api.routes.companies import _clamp_priority
  assert _clamp_priority(999) == 5
  assert _clamp_priority(10**400) == func(10**400)
  root = (
    Path(__file__).resolve().parents[2]
    / 'src/job_hunter/job_hunter'
  )
  comp_src = (root / 'api/routes/companies.py').read_text(encoding='utf-8')
  disc_src = (root / 'services/company_discovery.py').read_text(
    encoding='utf-8',
  )
  assert 'from job_hunter.core.constants import' in comp_src
  assert 'clamp_priority' in comp_src
  assert 'from job_hunter.core.constants import' in disc_src
  assert 'clamp_priority' in disc_src


# 7. list_due ATS filter + build_plan cooled filter ---------------------------


def test_list_due_requires_ats_and_board(
  tmp_path: Path, monkeypatch,
) -> None:
  '''list_due excludes active companies without ATS provider/board_ref.'''
  settings = _settings(tmp_path, monkeypatch)
  _seed_company_raw(
    settings.db_path, 'NoAts', None, ats=None, board=None,
  )
  _seed_company_raw(
    settings.db_path, 'NoBoard', None, ats='lever', board=None,
  )
  good = _seed_company_raw(
    settings.db_path, 'HasBoth', None, ats='lever', board='b1',
  )
  from job_hunter.db.repositories.companies import CompaniesRepository
  rows = CompaniesRepository(settings.db_path).list_due(10)
  assert [row['id'] for row in rows] == [good]


def test_build_plan_filters_cooled_and_counts_skips(
  tmp_path: Path, monkeypatch,
) -> None:
  '''50 cooled + 50 fresh: only fresh planned, cooled_skips=50.'''
  settings = _settings(tmp_path, monkeypatch)
  cooled_ids = [
    _seed_company_raw(
      settings.db_path, f'Cooled{i:02d}', '2026-01-01 00:00:00',
    )
    for i in range(50)
  ]
  fresh_ids = [
    _seed_company_raw(
      settings.db_path, f'Fresh{i:02d}', '2026-02-01 00:00:00',
    )
    for i in range(50)
  ]
  for company_id in cooled_ids:
    _cool_company(settings.db_path, company_id)
  import job_hunter.graph.nodes as nodes_mod
  config = {'configurable': {'settings': settings}}
  update = asyncio.run(
    nodes_mod.build_plan(
      {'run_id': 1, 'plan': {'quick_poll': False}}, config,
    ),
  )
  assert 'errors' not in update or not update['errors'], update
  planned = [int(t['company_id']) for t in update['plan']['targets']]
  assert set(planned) == set(fresh_ids), planned[:10]
  assert update['stats'].get('cooled_skips') == 50, update['stats']
  assert update['stats'].get('deferred_targets') == 0, update['stats']
  # Cooled heads were stamped checked so they rotate out.
  from job_hunter.db.repositories.companies import CompaniesRepository
  repo = CompaniesRepository(settings.db_path)
  for company_id in cooled_ids[:3]:
    row = repo.get(company_id)
    assert row is not None
    assert row['last_checked_at'] not in ('2026-01-01 00:00:00', None), row


def test_fetch_pair_cooled_marks_checked(
  tmp_path: Path, monkeypatch,
) -> None:
  '''fetch_pair cooled-skip also stamps mark_checked (rotation).'''
  settings = _settings(tmp_path, monkeypatch)
  company_id = _seed_company_raw(
    settings.db_path, 'CooledPair', '2020-01-01 00:00:00',
  )
  _cool_company(settings.db_path, company_id)
  import job_hunter.graph.nodes as nodes_mod
  config = {'configurable': {'settings': settings}}
  update = asyncio.run(
    nodes_mod.fetch_pair(
      {
        'company_id': company_id, 'name': 'CooledPair',
        'source_key': 'lever', 'board_ref': 'board-cooledpair',
      },
      config,
    ),
  )
  assert update['stats'].get('cooled_skips') == 1, update
  from job_hunter.db.repositories.companies import CompaniesRepository
  row = CompaniesRepository(settings.db_path).get(company_id)
  assert row is not None
  assert row['last_checked_at'] != '2020-01-01 00:00:00', row


# 8. Upsert must not bump last_checked_at --------------------------------------


def test_upsert_existing_does_not_bump_last_checked(
  tmp_path: Path, monkeypatch,
) -> None:
  '''upsert on an existing company preserves last_checked_at.'''
  settings = _settings(tmp_path, monkeypatch)
  company_id = _seed_company_raw(
    settings.db_path, 'Stable', '2020-01-01 00:00:00',
  )
  from job_hunter.db.repositories.companies import CompaniesRepository
  repo = CompaniesRepository(settings.db_path)
  assert repo.get(company_id)['last_checked_at'] == '2020-01-01 00:00:00'
  repo.upsert(name='Stable', domain='stable.example.com')
  row = repo.get(company_id)
  assert row is not None
  assert row['last_checked_at'] == '2020-01-01 00:00:00', row


# 9. bulk empty/failed split ---------------------------------------------------


def test_fetch_source_returns_records_and_ok_sentinel(
  tmp_path: Path, monkeypatch,
) -> None:
  '''fetch_source returns (records, True) on empty, (records, False) on raise.'''
  settings = _settings(tmp_path, monkeypatch)
  _ = settings
  from job_hunter.services import bulk_fetch as bulk_mod

  class _EmptyAdapter:
    '''Adapter returning no records.'''

    def __init__(self, http) -> None:
      '''Accept client.'''
      _ = http

    async def fetch(self, target, limit=200):
      '''Return empty.'''
      _ = (target, limit)
      return []

  class _RaisingAdapter:
    '''Adapter that always raises.'''

    def __init__(self, http) -> None:
      '''Accept client.'''
      _ = http

    async def fetch(self, target, limit=200):
      '''Raise boom.'''
      _ = (target, limit)
      raise RuntimeError('boom')

  class _FakeHttp:
    '''No-op client.'''

    async def close(self) -> None:
      '''Do nothing.'''

  def _stub(key, http):
    '''Return adapter per key.'''
    if key == 'empty_src':
      return _EmptyAdapter(http)
    return _RaisingAdapter(http)

  monkeypatch.setattr(bulk_mod, 'build_adapter', _stub)
  empty_result = asyncio.run(
    bulk_mod.fetch_source(_FakeHttp(), 'empty_src', 10),
  )
  assert isinstance(empty_result, tuple), empty_result
  empty_records, empty_ok = empty_result
  assert empty_records == []
  assert empty_ok is True
  bad_result = asyncio.run(
    bulk_mod.fetch_source(_FakeHttp(), 'bad_src', 10),
  )
  assert isinstance(bad_result, tuple), bad_result
  bad_records, bad_ok = bad_result
  assert bad_records == []
  assert bad_ok is False


def test_bulk_fetch_counts_empty_vs_failed_separately(
  tmp_path: Path, monkeypatch,
) -> None:
  '''bulk_fetch: empty source -> empty=1, raising source -> failed=1.'''
  settings = _settings(tmp_path, monkeypatch)
  from job_hunter.services import bulk_fetch as bulk_mod

  class _EmptyAdapter:
    '''Adapter returning no records.'''

    def __init__(self, http) -> None:
      '''Accept client.'''
      _ = http

    async def fetch(self, target, limit=200):
      '''Return empty.'''
      _ = (target, limit)
      return []

  class _RaisingAdapter:
    '''Adapter that always raises.'''

    def __init__(self, http) -> None:
      '''Accept client.'''
      _ = http

    async def fetch(self, target, limit=200):
      '''Raise boom.'''
      _ = (target, limit)
      raise RuntimeError('boom')

  class _FakeHttp:
    '''No-op client.'''

    async def close(self) -> None:
      '''Do nothing.'''

  def _stub(key, http):
    '''Return adapter per key.'''
    if key == 'empty_src':
      return _EmptyAdapter(http)
    return _RaisingAdapter(http)

  monkeypatch.setattr(bulk_mod, 'build_adapter', _stub)
  report = asyncio.run(
    bulk_mod.bulk_fetch(
      settings,
      source_keys=['empty_src', 'bad_src'],
      http=_FakeHttp(),
      persist=False,
    ),
  )
  assert hasattr(report, 'empty'), vars(report)
  assert report.empty == 1, vars(report)
  assert report.failed == 1, vars(report)
  assert report.fetched == 0
  assert report.per_source.get('empty_src') == 0
  assert report.per_source.get('bad_src') == 0


# 10. config docstring accuracy -------------------------------------------------


def test_config_docstring_has_no_dotenv_override_claim() -> None:
  '''config docstring: os.getenv only; gateway .env fallback noted.'''
  root = (
    Path(__file__).resolve().parents[2]
    / 'src/job_hunter/job_hunter/core/config.py'
  )
  text = root.read_text(encoding='utf-8')
  first = text.splitlines()[3] if len(text.splitlines()) > 3 else ''
  assert '.env overrides' not in first, first
  assert 'os.getenv' in first or 'os.getenv' in text[:2000], text[:2000]
  # Root-.env fallback for the gateway env path is documented.
  lower = text.lower()
  assert 'gateway' in lower
  assert 'fallback' in lower
