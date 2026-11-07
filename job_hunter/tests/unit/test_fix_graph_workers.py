#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for W5 graph/workers fixes.

Covers worker entrypoint config argument, safe_node error conversion,
scheduler add_job kwargs, persist_record dedupe, run_manager
ConflictError handling, checkpoint round-trip, route_targets fan-out
cap, and similar_title_exists company filtering.
'''

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from job_hunter.core.db import run_migrations


def test_worker_main_passes_default_config_path(monkeypatch) -> None:
  '''Worker __main__ passes a Path config arg to start_scheduler.'''
  import runpy

  import job_hunter.workers.scheduler as sched

  captured: dict = {}

  def fake_start(path) -> None:
    captured['path'] = path

  monkeypatch.setattr(sched, 'start_scheduler', fake_start)
  runpy.run_module('job_hunter.worker', run_name='__main__')
  assert 'path' in captured, (
    'start_scheduler was not called by worker entrypoint'
  )
  assert isinstance(captured['path'], Path), (
    f'expected Path arg, got {type(captured['path'])}'
  )
  assert str(captured['path']).endswith(str(Path('config') / 'app.yaml'))


def test_safe_node_converts_exception_to_errors_entry() -> None:
  '''safe_node wraps a failing async node into an errors entry.'''
  from job_hunter.graph.nodes import safe_node

  @safe_node
  async def failing_node(state, config):
    raise RuntimeError('boom ' + 'x' * 1000)

  result = asyncio.run(failing_node({}, {}))
  assert 'errors' in result and len(result['errors']) == 1
  err = result['errors'][0]
  # NodeError pydantic model or dict-like: check fields both ways
  node = (
    err.get('node') if isinstance(err, dict) else getattr(err, 'node', None)
  )
  message = (
    err.get('message')
    if isinstance(err, dict)
    else getattr(err, 'message', None)
  )
  recoverable = (
    err.get('recoverable')
    if isinstance(err, dict)
    else getattr(err, 'recoverable', None)
  )
  assert node == 'failing_node'
  assert message is not None and len(message) <= 500 and 'boom' in message
  assert recoverable is True


def test_safe_node_applied_to_load_profile() -> None:
  '''load_profile sync node: bad config becomes errors.'''
  from job_hunter.graph.nodes import load_profile

  result = load_profile({}, {})
  assert 'errors' in result and len(result['errors']) >= 1
  err = result['errors'][0]
  node = (
    err.get('node') if isinstance(err, dict) else getattr(err, 'node', None)
  )
  assert node == 'load_profile'


def test_safe_node_applied_to_build_plan() -> None:
  '''build_plan (async node) is wrapped: bad config becomes errors.'''
  from job_hunter.graph.nodes import build_plan

  result = asyncio.run(build_plan({}, {}))
  assert 'errors' in result and len(result['errors']) >= 1
  err = result['errors'][0]
  node = (
    err.get('node') if isinstance(err, dict) else getattr(err, 'node', None)
  )
  assert node == 'build_plan'


def test_route_targets_caps_fanout_at_50() -> None:
  '''route_targets truncates fan-out to at most 50 Sends.'''
  from job_hunter.graph.discovery_graph import route_targets

  targets = [
    {'company_id': i, 'name': f'C{i}', 'source_key': 'lever', 'board_ref': 'x'}
    for i in range(60)
  ]
  sends = route_targets({'plan': {'targets': targets}})
  assert len(sends) == 50


def test_scheduler_add_job_kwargs(monkeypatch, tmp_path: Path) -> None:
  '''All scheduler add_job calls use the required kwargs.'''
  import job_hunter.workers.scheduler as sched_mod
  import job_hunter.workers.jobs as jobs_mod

  calls: list = []

  class FakeScheduler:
    def add_job(self, func, trigger, **kwargs):
      calls.append(kwargs)

    def start(self):
      pass

  fake_settings = SimpleNamespace(
    log_level='INFO',
    data_dir=tmp_path,
    schedule={},
    discovery={},
  )
  monkeypatch.setattr(sched_mod, 'BlockingScheduler', FakeScheduler)
  monkeypatch.setattr(sched_mod, 'setup_logging', lambda *a, **k: None)
  # W1 cycle14: scheduler._cached_bootstrap delegates to
  # workers.jobs._settings, which calls the top-level imported
  # bootstrap in jobs_mod -- patch that name (and the original
  # bootstrap_mod name for belt-and-braces).
  import job_hunter.core.bootstrap as bootstrap_mod

  monkeypatch.setattr(bootstrap_mod, 'bootstrap', lambda path: fake_settings)
  monkeypatch.setattr(jobs_mod, 'bootstrap', lambda path: fake_settings)
  # recover_orphans imported inside start_scheduler from workers.jobs
  monkeypatch.setattr(jobs_mod, 'recover_orphans', lambda path: [])
  # clear any cached settings so bootstrap fake is used
  for mod in (sched_mod, jobs_mod):
    cache = getattr(mod, '_SETTINGS_CACHE', None) or getattr(
      mod, '_BOOTSTRAP_CACHE', None
    )
    if isinstance(cache, dict):
      cache.clear()

  sched_mod.start_scheduler(tmp_path / 'app.yaml')
  assert len(calls) >= 5, f'expected >=5 add_job calls, got {len(calls)}'
  for kwargs in calls:
    assert kwargs.get('max_instances') == 1, (
      f'max_instances missing/wrong in {kwargs}'
    )
    assert kwargs.get('coalesce') is True, f'coalesce missing/wrong in {kwargs}'
    assert kwargs.get('misfire_grace_time') == 3600, (
      f'misfire_grace_time missing/wrong in {kwargs}'
    )


def test_jobs_settings_cached_once(monkeypatch, tmp_path: Path) -> None:
  '''workers.jobs._settings caches bootstrap result per process/config.'''
  import job_hunter.workers.jobs as jobs_mod

  config_path = tmp_path / 'app.yaml'
  config_path.write_text('salary_hard_floor_lpa: 45\n', encoding='utf-8')
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  # clear cache if present
  cache = getattr(jobs_mod, '_SETTINGS_CACHE', None)
  if isinstance(cache, dict):
    cache.clear()
  first = jobs_mod._settings(config_path)
  second = jobs_mod._settings(config_path)
  assert first is second


def test_persist_record_dedupe_dict(tmp_path: Path) -> None:
  '''persist_record inserts once; second identical record is duplicate.'''
  from job_hunter.db.repositories.companies import CompaniesRepository
  from job_hunter.db.repositories.jobs import JobsRepository
  from job_hunter.services.persist import persist_record

  db = tmp_path / 'app.db'
  run_migrations(db)
  jobs_repo = JobsRepository(db)
  companies_repo = CompaniesRepository(db)
  record = {
    'source_key': 'manual',
    'external_id': '1',
    'url': 'https://example.com/j/1',
    'company_name': 'Acme',
    'title': 'Data Scientist',
    'location_text': 'Bengaluru',
    'description_text': 'python jobs',
    'salary_raw': '',
    'posted_at': None,
    'employment_type_raw': 'full_time',
    'work_mode_hint': '',
    'extra': {},
  }
  first = asyncio.run(
    persist_record(
      jobs_repo=jobs_repo, record=record, company_repo=companies_repo
    )
  )
  second = asyncio.run(
    persist_record(
      jobs_repo=jobs_repo, record=record, company_repo=companies_repo
    )
  )
  assert first == 'inserted'
  assert second == 'duplicate'
  assert len(jobs_repo.list_jobs()) == 1


def test_persist_record_dedupe_raw_job(tmp_path: Path) -> None:
  '''persist_record also accepts a RawJobRecord and dedupes it.'''
  from job_hunter.core.models import RawJobRecord
  from job_hunter.db.repositories.jobs import JobsRepository
  from job_hunter.services.persist import persist_record

  db = tmp_path / 'app.db'
  run_migrations(db)
  jobs_repo = JobsRepository(db)
  record = RawJobRecord(
    source_key='manual',
    external_id='2',
    url='https://example.com/j/2',
    company_name='Acme',
    title='ML Engineer',
    location_text='Mumbai',
    description_text='ml work',
  )
  assert (
    asyncio.run(persist_record(jobs_repo=jobs_repo, record=record))
    == 'inserted'
  )
  assert (
    asyncio.run(persist_record(jobs_repo=jobs_repo, record=record))
    == 'duplicate'
  )
  assert len(jobs_repo.list_jobs()) == 1


def test_run_manager_conflict_on_duplicate_enqueue(
  tmp_path: Path, monkeypatch
) -> None:
  '''Second enqueue while a run is active raises ConflictError.'''
  from job_hunter.core.config import AppSettings
  from job_hunter.core.errors import ConflictError
  from job_hunter.services.run_manager import RunManager

  run_migrations(tmp_path / 'app.db')
  config_path = tmp_path / 'app.yaml'
  config_path.write_text('salary_hard_floor_lpa: 45\n', encoding='utf-8')
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  settings = AppSettings(config_path)
  manager = RunManager(settings)
  manager.enqueue('discovery', 'test')
  with pytest.raises(ConflictError):
    manager.enqueue('discovery', 'test')


def test_run_manager_conflict_on_integrity_error(
  tmp_path: Path, monkeypatch
) -> None:
  '''Duplicate active run becomes ConflictError via create_if_not_active.

  Covers both paths in RunManager.enqueue: create_if_not_active
  returning None (active run exists / lost race) and raising
  IntegrityError (unique/active race) both raise ConflictError.
  '''
  from job_hunter.core.config import AppSettings
  from job_hunter.core.errors import ConflictError
  from job_hunter.services.run_manager import RunManager

  run_migrations(tmp_path / 'app.db')
  config_path = tmp_path / 'app.yaml'
  config_path.write_text('salary_hard_floor_lpa: 45\n', encoding='utf-8')
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  settings = AppSettings(config_path)
  manager = RunManager(settings)
  # create_if_not_active returns None when an active run exists
  monkeypatch.setattr(
    manager.runs,
    'create_if_not_active',
    lambda *a, **k: None,
  )
  with pytest.raises(ConflictError):
    manager.enqueue('discovery', 'test')
  # create_if_not_active raises IntegrityError on a race
  monkeypatch.setattr(
    manager.runs,
    'create_if_not_active',
    lambda *a, **k: (_ for _ in ()).throw(
      sqlite3.IntegrityError('dup active run')
    ),
  )
  with pytest.raises(ConflictError):
    manager.enqueue('discovery', 'test')


def test_run_manager_no_duplicate_contextvar(
  tmp_path: Path, monkeypatch
) -> None:
  '''run_manager uses run_id_var, no duplicate ContextVar.'''
  import job_hunter.services.run_manager as rm
  from job_hunter.core.logging import run_id_var

  # current_run_id must be deleted (not a separate ContextVar)
  assert not hasattr(rm, 'current_run_id') or rm.current_run_id is run_id_var, (
    'duplicate current_run_id ContextVar still present'
  )


def test_checkpoint_round_trip(tmp_path: Path) -> None:
  '''Checkpoint save/load round-trips when AsyncSqliteSaver is available.'''
  try:
    import aiosqlite  # noqa: F401
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: F401
  except ImportError:
    pytest.skip(
      'langgraph sqlite checkpointer / aiosqlite not available in this env'
    )

  from job_hunter.graph.checkpointing import open_checkpointer, thread_config

  db_path = tmp_path / 'nested' / 'ckpt.sqlite'

  async def _round_trip():
    async with open_checkpointer(db_path) as saver:
      assert saver is not None
      cfg = thread_config(42)
      ckpt = {
        'v': 1,
        'id': '00000000-0000-0000-0000-0000000000aa',
        'ts': '2026-01-01T00:00:00+00:00',
        'channel_values': {'x': 1},
        'channel_versions': {'x': '00000000-0000-0000-0000-0000000000aa'},
      }
      meta = {'source': 'input', 'step': 0, 'writes': {}, 'parents': {}}
      await saver.aput(cfg, ckpt, meta, {})
      got = await saver.aget(cfg)
      assert got is not None
      assert got['channel_values'] == {'x': 1}
    assert db_path.exists()

  asyncio.run(_round_trip())


def test_similar_title_exists_filters_by_company() -> None:
  '''similar_title_exists only matches candidates for the given company.'''
  from job_hunter.db.repositories.jobs import JobsRepository
  from job_hunter.services.dedupe import DedupeEngine

  engine = DedupeEngine(JobsRepository(':memory:'))
  # Structured candidates: match exists only for company 2, not company 1
  candidates = [
    {'company_id': 2, 'title': 'Staff Data Scientist'},
    {'company_id': 1, 'title': 'Cook'},
  ]
  assert (
    engine.similar_title_exists(
      'Staff Data Scientist', company_id=1, candidates=candidates
    )
    is False
  )
  assert (
    engine.similar_title_exists(
      'Staff Data Scientist', company_id=2, candidates=candidates
    )
    is True
  )
  # Plain string candidates (pre-fetched for the company) still work
  assert (
    engine.similar_title_exists(
      'Staff Data Scientist', company_id=1, candidates=['Staff Data Scientist']
    )
    is True
  )
  assert (
    engine.similar_title_exists(
      'Staff Data Scientist', company_id=1, candidates=[]
    )
    is False
  )


def test_nodes_no_stub_fake_and_no_inline_json() -> None:
  '''nodes.py has no Stub fake, no inline json, and no json. usage.

  The json module is unused in nodes.py (grep: no ``json.`` call
  sites), so W1 removes the legacy top-level ``import json`` pin.
  This test therefore asserts absence of json usage rather than
  presence of the import: it passes with the import present
  (unused) and after W1 removes it, and fails if any code starts
  using ``json.`` or reintroduces an inline ``import json as _json``.
  '''
  import inspect

  import job_hunter.graph.nodes as nodes_mod

  src = inspect.getsource(nodes_mod)
  assert "type('Stub'" not in src and 'type("Stub"' not in src
  assert 'import json as _json' not in src
  assert 'json.' not in src, 'nodes.py must not use the json module'
