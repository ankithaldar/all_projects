#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle-14 tester: fan-out cooldown, list_due ATS filter, negative huge
IDs, plus verify-only pins for sanitized cache / embed / wheel / JS /
checkpoint and the tautology scan.

Hermetic: tmp_path DBs, stub adapters, TestClient, no network.

W1 status: initially not landed at first probe (build_plan had
no cooldown filter, list_due had no ATS filter, Path params had
no lower bound -> 500). W1 landed mid-run: build_plan now
filters cooled companies and counts cooled_skips in stats,
list_due requires ats_provider + board_ref, and Path params use
ge=-MAX_SQLITE_ID -> 422. The desired-behavior tests below were
RED (xfail) before W1 and XPASS after; markers were then
removed and the tests are now ordinary passing tests.
'''

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

fastapi = pytest.importorskip('fastapi')
TestClient = pytest.importorskip('fastapi.testclient').TestClient

ROOT = Path(__file__).resolve().parents[2]
_HUGE = '1' + '0' * 400
_HUGE_NEG = '-' + _HUGE


def _settings(tmp_path: Path, monkeypatch):
  '''Build migrated settings for direct service tests.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest fixture.

  Returns:
    Migrated AppSettings.
  '''
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  config_path = tmp_path / 'app.yaml'
  config_path.write_text(
    'salary_hard_floor_lpa: 45\n', encoding='utf-8',
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
    monkeypatch: Pytest fixture.

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
  db_path: Path,
  name: str,
  last_checked: str | None,
  *,
  provider: str | None = 'lever',
  status: str = 'active',
  priority: int = 3,
) -> int:
  '''Insert one company row directly.

  Args:
    db_path: Database path.
    name: Company name.
    last_checked: last_checked_at value or None.
    provider: ATS provider key or None (unverified).
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
        name,
        name.lower(),
        f'{name.lower()}.example.com',
        provider,
        f'board-{name.lower()}' if provider else None,
        priority,
        status,
        last_checked,
      ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def _set_cooldown(db_path: Path, company_id: int) -> None:
  '''Open a future cooldown for company:<id>.

  Args:
    db_path: Database path.
    company_id: Company id.
  '''
  with closing(sqlite3.connect(str(db_path))) as conn:
    conn.execute(
      'INSERT INTO crawl_state (scope, cooldown_until) '
      "VALUES (?, datetime('now', '+24 hours')) "
      'ON CONFLICT(scope) DO UPDATE SET '
      "cooldown_until = datetime('now', '+24 hours')",
      (f'company:{company_id}',),
    )
    conn.commit()


class _CountingAdapter:
  '''Adapter returning no records, counting fetch calls.'''

  calls = 0

  def __init__(self, http) -> None:
    '''Accept the shared client.

    Args:
      http: Unused client.
    '''
    _ = http

  async def fetch(self, target, limit=200):
    '''Count and return no records.

    Args:
      target: Target.
      limit: Cap.

    Returns:
      Empty record list.
    '''
    _ = (target, limit)
    _CountingAdapter.calls += 1
    return []


def _seed_50_cooled_50_fresh(db_path: Path) -> tuple[list[int], list[int]]:
  '''Seed 50 cooled (oldest) + 50 fresh companies.

  Args:
    db_path: Database path.

  Returns:
    (cooled ids, fresh ids) in oldest-first order.
  '''
  cooled = [
    _seed_company(db_path, f'Cooled{i:02d}', f'2026-01-01 00:00:{i:02d}')
    for i in range(50)
  ]
  for cid in cooled:
    _set_cooldown(db_path, cid)
  fresh = [
    _seed_company(db_path, f'Fresh{i:02d}', f'2026-01-02 00:00:{i:02d}')
    for i in range(50)
  ]
  return cooled, fresh


# 1. Fan-out cooldown -----------------------------------------------------

def test_fanout_cooldown_fetch_pair_counts_cooled_skips_current(
  tmp_path: Path, monkeypatch,
) -> None:
  '''CURRENT contract: cooled_skips counted at fetch_pair layer.

  50 cooled + 50 fresh seeded. build_plan + route_targets fan out
  at most 50 Sends. Running fetch_pair over every routed target:
  each cooled target yields cooled_skips==1 with no adapter call,
  each fresh target is fetched. Sum of cooled_skips equals the
  number of cooled targets among the routed set, whichever set
  build_plan currently routes (cooled-first today, fresh-only
  after W1) -- this test pins the counting, not the routing.
  '''
  settings = _settings(tmp_path, monkeypatch)
  cooled, fresh = _seed_50_cooled_50_fresh(settings.db_path)
  cooled_set = set(cooled)
  _CountingAdapter.calls = 0

  import job_hunter.graph.nodes as nodes_mod
  from job_hunter.graph.discovery_graph import route_targets

  monkeypatch.setattr(
    nodes_mod,
    'build_adapter',
    lambda key, http: _CountingAdapter(http),
  )
  config = {'configurable': {'settings': settings}}
  update = asyncio.run(
    nodes_mod.build_plan(
      {'run_id': 1, 'plan': {'quick_poll': False}}, config,
    ),
  )
  assert 'errors' not in update or not update['errors'], update
  sends = route_targets({'plan': update['plan']})
  assert len(sends) == 50, len(sends)
  routed_ids = [int(s.arg['company_id']) for s in sends]
  cooled_routed = [cid for cid in routed_ids if cid in cooled_set]
  fresh_routed = [cid for cid in routed_ids if cid not in cooled_set]
  assert len(cooled_routed) + len(fresh_routed) == 50
  assert set(fresh_routed) <= set(fresh)

  total_cooled_skips = 0
  for send in sends:
    result = asyncio.run(nodes_mod.fetch_pair(send.arg, config))
    assert not result['errors'], result
    total_cooled_skips += int(result['stats'].get('cooled_skips', 0))
    cid = int(send.arg['company_id'])
    if cid in cooled_set:
      assert result['stats']['cooled_skips'] == 1, result
      assert result['raw_jobs'] == [], result
    else:
      assert result['stats'].get('cooled_skips', 0) == 0, result
  assert total_cooled_skips == len(cooled_routed), (
    total_cooled_skips,
    len(cooled_routed),
  )
  assert _CountingAdapter.calls == len(fresh_routed), (
    _CountingAdapter.calls,
    len(fresh_routed),
  )


def test_fanout_cooldown_build_plan_plans_fresh_only_desired_w1(
  tmp_path: Path, monkeypatch,
) -> None:
  '''build_plan plans fresh only, cooled excluded, skips counted.'''
  settings = _settings(tmp_path, monkeypatch)
  _cooled, fresh = _seed_50_cooled_50_fresh(settings.db_path)

  import job_hunter.graph.nodes as nodes_mod

  update = asyncio.run(
    nodes_mod.build_plan(
      {'run_id': 1, 'plan': {'quick_poll': False}},
      {'configurable': {'settings': settings}},
    ),
  )
  assert 'errors' not in update or not update['errors'], update
  planned_ids = [int(t['company_id']) for t in update['plan']['targets']]
  assert planned_ids == fresh, planned_ids[:5]
  assert update['stats'].get('cooled_skips') == 50, update['stats']
  assert update['stats'].get('deferred_targets') == 0, update['stats']


# 2. list_due ATS filter --------------------------------------------------

def test_build_plan_excludes_unverified_null_ats_current(
  tmp_path: Path, monkeypatch,
) -> None:
  '''CURRENT: build_plan excludes NULL-ats_provider companies.

  list_due itself is unfiltered today, but its consumer
  build_plan drops rows with falsy ats_provider/board_ref, so
  unverified companies never become fetch targets. Holds both
  before and after a list_due ATS filter lands.
  '''
  settings = _settings(tmp_path, monkeypatch)
  _seed_company(settings.db_path, 'NullCo', None, provider=None)
  _seed_company(settings.db_path, 'LeverCo', None, provider='lever')

  import job_hunter.graph.nodes as nodes_mod

  update = asyncio.run(
    nodes_mod.build_plan(
      {'run_id': 1, 'plan': {'quick_poll': False}},
      {'configurable': {'settings': settings}},
    ),
  )
  assert 'errors' not in update or not update['errors'], update
  names = [t['name'] for t in update['plan']['targets']]
  assert names == ['LeverCo'], names


def test_list_due_excludes_unverified_null_ats_desired_w1(
  tmp_path: Path, monkeypatch,
) -> None:
  '''list_due excludes NULL ats_provider rows.'''
  settings = _settings(tmp_path, monkeypatch)
  _seed_company(settings.db_path, 'NullCo', None, provider=None)
  _seed_company(settings.db_path, 'LeverCo', None, provider='lever')

  from job_hunter.db.repositories.companies import CompaniesRepository

  names = [r['name'] for r in CompaniesRepository(settings.db_path).list_due()]
  assert 'NullCo' not in names, names
  assert 'LeverCo' in names, names


# 3. Negative huge IDs ----------------------------------------------------

def test_recommendations_page_huge_negative_422(
  tmp_path: Path, monkeypatch,
) -> None:
  '''GET /api/recommendations?page=-10**400 -> 422 (ge=1).'''
  client = _make_client(tmp_path, monkeypatch)
  response = client.get(f'/api/recommendations?page={_HUGE_NEG}')
  assert response.status_code == 422, response.text


def test_jobs_huge_negative_422_desired_w1(
  tmp_path: Path, monkeypatch,
) -> None:
  '''GET /api/jobs/-10**400 -> 422, not 500.'''
  client = _make_client(tmp_path, monkeypatch)
  response = client.get(f'/api/jobs/{_HUGE_NEG}')
  assert response.status_code == 422, response.text


def test_companies_patch_huge_negative_422_desired_w1(
  tmp_path: Path, monkeypatch,
) -> None:
  '''PATCH /api/companies/-10**400 -> 422, not 500.'''
  client = _make_client(tmp_path, monkeypatch)
  response = client.patch(f'/api/companies/{_HUGE_NEG}', json={})
  assert response.status_code == 422, response.text


# 4-6. Verify-only pins ---------------------------------------------------

def test_cycle13_sanitized_astream_cache_and_embed_fetch_exist() -> None:
  '''Cycle-13 covers sanitized astream cache, embed, fetch failures.

  Verify-only: the real gateway test lives in
  test_fix_cycle13_tester.py::test_sanitized_astream_cache_twice_second_is_cache
  and is run separately as evidence; duplicating it here would
  only re-prove the same gateway path.
  '''
  cycle13 = (ROOT / 'tests/unit/test_fix_cycle13_tester.py').read_text(
    encoding='utf-8',
  )
  assert 'def test_sanitized_astream_cache_twice_second_is_cache' in cycle13
  assert 'def test_compute_embeddings_success_cycle13' in cycle13
  assert 'def test_fetch_pair_raising_yields_fetch_failures_count' in cycle13
  cycle5 = (ROOT / 'tests/unit/test_fix_cycle5_docs.py').read_text(
    encoding='utf-8',
  )
  cycle6 = (ROOT / 'tests/unit/test_fix_cycle6_docs.py').read_text(
    encoding='utf-8',
  )
  assert 'def test_fetch_pair_adapter_raising_fetch_failure' in cycle5
  assert 'def test_compute_embeddings_success_saves_db_row' in cycle6


def test_wheel_js_checkpoint_tests_exist() -> None:
  '''Wheel, JS harness, and checkpoint round-trip tests exist.

  Verify-only: each is run separately as evidence.
  '''
  cycle5 = (ROOT / 'tests/unit/test_fix_cycle5_docs.py').read_text(
    encoding='utf-8',
  )
  cycle6 = (ROOT / 'tests/unit/test_fix_cycle6_docs.py').read_text(
    encoding='utf-8',
  )
  assert 'def test_wheel_contains_runtime_assets' in cycle5
  assert 'def test_wheel_install_import_and_dashboard_assets' in cycle6
  js_harness = ROOT / 'tests/unit/test_js_harness.py'
  assert js_harness.exists()
  assert 'api.js' in js_harness.read_text(encoding='utf-8')
  workers = (ROOT / 'tests/unit/test_fix_graph_workers.py').read_text(
    encoding='utf-8',
  )
  assert 'def test_checkpoint_round_trip' in workers


# 7. Tautology scan pin ---------------------------------------------------

def test_throttle_tautology_assert_removed() -> None:
  '''The >=0.4 or >=0.35 throttle tautology is gone (exact >=0.4).'''
  cycle6 = (ROOT / 'tests/unit/test_fix_cycle6_adapters.py').read_text(
    encoding='utf-8',
  )
  assert 'elapsed >= 0.4 or elapsed >= 0.35' not in cycle6
  assert 'assert elapsed >= 0.4' in cycle6
