#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=import-outside-toplevel,too-few-public-methods

'''Repo-switch tests: callers delegate to repository methods.'''


from __future__ import annotations

import asyncio
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from job_hunter.core.db import run_migrations
from job_hunter.db.repositories.taxonomy import TaxonomyRepository
from job_hunter.services.skills_taxonomy import SkillsTaxonomy


def _db(tmp_path: Path) -> Path:
  '''Create a migrated app database.

  Args:
    tmp_path: Pytest temporary directory.

  Returns:
    Database path.
  '''
  db = tmp_path / 'app.db'
  run_migrations(db)
  return db


def test_taxonomy_all_skills_uses_repo(
  tmp_path: Path, monkeypatch,
) -> None:
  '''all_skills delegates to TaxonomyRepository.skill_rows.

  Args:
    tmp_path: Pytest temporary directory.
    monkeypatch: Pytest fixture.
  '''
  _db(tmp_path)
  calls: list = []

  def fake_skill_rows(self):
    '''Return canned skill rows.

    Args:
      self: Repository instance.

    Returns:
      Canned rows.
    '''
    _ = self
    calls.append('skill_rows')
    return [{
      'id': 1, 'canonical_name': 'Python', 'category': 'language',
    }]

  monkeypatch.setattr(
    TaxonomyRepository, 'skill_rows', fake_skill_rows,
  )
  taxonomy = SkillsTaxonomy(tmp_path / 'app.db')
  rows = taxonomy.all_skills()
  assert rows == [{
    'id': 1, 'canonical_name': 'Python', 'category': 'language',
  }]
  assert calls == ['skill_rows']


def test_taxonomy_resolve_uses_repo(
  tmp_path: Path, monkeypatch,
) -> None:
  '''resolve delegates to TaxonomyRepository.resolve_alias.

  Args:
    tmp_path: Pytest temporary directory.
    monkeypatch: Pytest fixture.
  '''
  _db(tmp_path)
  calls: list = []

  def fake_resolve_alias(self, name):
    '''Return a canned skill id.

    Args:
      self: Repository instance.
      name: Raw skill text.

    Returns:
      Canned skill id.
    '''
    _ = self
    calls.append(name)
    return 42

  monkeypatch.setattr(
    TaxonomyRepository, 'resolve_alias', fake_resolve_alias,
  )
  taxonomy = SkillsTaxonomy(tmp_path / 'app.db')
  assert taxonomy.resolve('PYTHON') == 42
  assert calls


def test_watchdog_uses_fetch_helpers(
  tmp_path: Path, monkeypatch,
) -> None:
  '''build_report routes SQL through core.db fetch helpers.

  Args:
    tmp_path: Pytest temporary directory.
    monkeypatch: Pytest fixture.
  '''
  _db(tmp_path)
  import job_hunter.services.watchdog as watchdog_mod
  from job_hunter.services.watchdog import build_report

  calls_all: list = []
  calls_one: list = []

  @contextmanager
  def fake_fetch_all(_db_path, sql, _params=()):
    '''Yield canned rows based on the query.

    Args:
      _db_path: Database path.
      sql: SQL text.
      _params: Bound parameters.

    Yields:
      Canned rows.
    '''
    calls_all.append(sql)
    if 'crawl_state' in sql:
      yield [{'scope': 'company:1'}]
    elif 'ats_provider IS NULL' in sql:
      yield [{'name': 'Acme'}]
    else:
      yield [{'key': 'lever', 'jobs': 3, 'freshest': '2026-01-01'}]

  @contextmanager
  def fake_fetch_one(_db_path, sql, _params=()):
    '''Yield a canned count row.

    Args:
      _db_path: Database path.
      sql: SQL text.
      _params: Bound parameters.

    Yields:
      Canned count row.
    '''
    calls_one.append(sql)
    yield {'n': 2}

  monkeypatch.setattr(
    watchdog_mod, 'fetch_all', fake_fetch_all, raising=False,
  )
  monkeypatch.setattr(
    watchdog_mod, 'fetch_one', fake_fetch_one, raising=False,
  )
  settings = SimpleNamespace(db_path=tmp_path / 'app.db')
  report = build_report(settings)
  assert report['companies_needs_review'] == 2
  assert report['quarantined_jobs'] == 2
  assert report['cooled_down_sources'] == ['company:1']
  assert report['sample_unverified_companies'] == ['Acme']
  assert report['source_freshness'] == [{
    'key': 'lever', 'jobs': 3, 'freshest': '2026-01-01',
  }]
  assert calls_all and calls_one


class _FakeJobsRepo:
  '''Jobs repo double exposing only the batched skill API.'''

  def __init__(self, rows: list) -> None:
    '''Store canned rows.

    Args:
      rows: Canned skill rows.
    '''
    self._rows = rows
    self.calls: list = []

  def job_skill_rows(self, job_ids):
    '''Record the batched call and return canned rows.

    Args:
      job_ids: Requested job ids.

    Returns:
      Canned rows.
    '''
    self.calls.append(list(job_ids))
    return list(self._rows)


def test_nodes_job_skill_rows_batched() -> None:
  '''_job_skill_rows calls JobsRepository.job_skill_rows batched.'''
  from job_hunter.graph.nodes import _job_skill_rows

  repo = _FakeJobsRepo([
    {'job_id': 1, 'skill_id': 7, 'kind': 'must_have'},
    {'job_id': 2, 'skill_id': 8, 'kind': 'nice_to_have'},
  ])
  rows = _job_skill_rows(repo, [1, 2])
  assert rows == [
    {'job_id': 1, 'skill_id': 7, 'kind': 'must_have'},
    {'job_id': 2, 'skill_id': 8, 'kind': 'nice_to_have'},
  ]
  assert repo.calls == [[1, 2]]
  assert not hasattr(repo, '_db_path')


def test_nodes_job_skill_rows_single_id_compat() -> None:
  '''_job_skill_rows still accepts a single job id.'''
  from job_hunter.graph.nodes import _job_skill_rows

  repo = _FakeJobsRepo([
    {'job_id': 5, 'skill_id': 9, 'kind': 'must_have'},
  ])
  rows = _job_skill_rows(repo, 5)
  assert rows == [{'job_id': 5, 'skill_id': 9, 'kind': 'must_have'}]
  assert repo.calls == [[5]]


def test_settings_llm_stats_uses_gateway_repo(
  tmp_path: Path, monkeypatch,
) -> None:
  '''/llm/stats delegates to GatewayStatsRepository.llm_stats.

  Args:
    tmp_path: Pytest temporary directory.
    monkeypatch: Pytest fixture.
  '''
  import job_hunter.api.routes.settings as settings_mod
  from job_hunter.api.routes.settings import llm_stats

  calls: list = []
  sentinel = {
    'by_provider': [{'provider': 'p1', 'calls': 1}],
    'by_node': [],
    'totals': {'calls': 1, 'tokens': 10, 'cost': 0.5},
  }

  class FakeGatewayStats:
    '''Gateway stats double recording llm_stats calls.'''

    def __init__(self, db_path=None) -> None:
      '''Accept an optional default path.

      Args:
        db_path: Optional default gateway path.
      '''
      calls.append(('init', db_path))

    def llm_stats(self, db_path=None, days=7):
      '''Record the call and return the sentinel shape.

      Args:
        db_path: Gateway database path.
        days: Lookback window.

      Returns:
        Sentinel stats mapping.
      '''
      calls.append(('llm_stats', db_path, days))
      return sentinel

  monkeypatch.setattr(
    settings_mod, 'GatewayStatsRepository', FakeGatewayStats,
    raising=False,
  )
  path = tmp_path / 'gateway.db'
  settings = SimpleNamespace(gateway_db_path=path)
  result = llm_stats(days=3, settings=settings)
  assert result == sentinel
  assert ('llm_stats', path, 3) in calls


def test_checkpoint_resume_round_trip(tmp_path: Path) -> None:
  '''Checkpoint resume round-trips when sqlite saver exists.

  Args:
    tmp_path: Pytest temporary directory.
  '''
  try:
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    _ = (aiosqlite, AsyncSqliteSaver)
  except ImportError:
    pytest.skip(
      'langgraph-checkpoint-sqlite not available in this env; '
      'Lead to verify checkpoint resume in Cycle 2'
    )
  from job_hunter.graph.checkpointing import open_checkpointer, thread_config

  db_path = tmp_path / 'ckpt.sqlite'

  async def _round_trip():
    '''Put and get one checkpoint.

    Returns:
      None.
    '''
    async with open_checkpointer(db_path) as saver:
      assert saver is not None
      cfg = thread_config(42)
      ckpt = {
        'v': 1,
        'id': '00000000-0000-0000-0000-0000000000aa',
        'ts': '2026-01-01T00:00:00+00:00',
        'channel_values': {'x': 1},
        'channel_versions': {
          'x': '00000000-0000-0000-0000-0000000000aa',
        },
      }
      meta = {
        'source': 'input', 'step': 0, 'writes': {}, 'parents': {},
      }
      await saver.aput(cfg, ckpt, meta, {})
      got = await saver.aget(cfg)
      assert got is not None
      assert got['channel_values'] == {'x': 1}
    assert db_path.exists()

  asyncio.run(_round_trip())
