#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''RED tests for cycle-3 DB fixes (F2-db-gateway).'''

from __future__ import annotations

import inspect
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

import pytest
from job_hunter.core.db import run_migrations, session
from job_hunter.db.repositories.companies import CompaniesRepository
from job_hunter.db.repositories.jobs import JobsRepository
from job_hunter.db.repositories.recommendations import (
  RecommendationsRepository,
)
from job_hunter.db.repositories.runs import RunsRepository
from job_hunter.db.repositories.settings import GatewayStatsRepository


def _db(tmp_path: Path) -> Path:
  db = tmp_path / 'app.db'
  run_migrations(db)
  return db


def _job(repo, suffix: str) -> int:
  payload = {
    'source_key': 'lever',
    'url': f'https://j.example/{suffix}',
    'canonical_url': f'https://j.example/{suffix}',
    'title': f'Job {suffix}',
    'raw_json': '{}',
    'content_hash': f'hash-{suffix}',
    'description_text': 'data scientist python',
  }
  return int(repo.insert(payload))


def _manager(tmp_path: Path, monkeypatch):
  from job_hunter.core.config import AppSettings
  from job_hunter.services.run_manager import RunManager

  run_migrations(tmp_path / 'app.db')
  config_path = tmp_path / 'app.yaml'
  config_path.write_text('salary_hard_floor_lpa: 45\n', encoding='utf-8')
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  settings = AppSettings(config_path)
  return RunManager(settings)


# 1. repo negative LIMIT clamp -------------------------------------------


def test_list_jobs_negative_per_page_clamped(tmp_path: Path) -> None:
  repo = JobsRepository(_db(tmp_path))
  for i in range(3):
    _job(repo, f'neg{i}')
  rows = repo.list_jobs(per_page=-1)
  assert len(rows) == 1


def test_list_jobs_huge_per_page_clamped_to_cap(tmp_path: Path) -> None:
  repo = JobsRepository(_db(tmp_path))
  for i in range(3):
    _job(repo, f'huge{i}')
  rows = repo.list_jobs(per_page=10000)
  assert len(rows) <= 100


def test_recommendations_negative_per_page_clamped(tmp_path: Path) -> None:
  db = _db(tmp_path)
  jobs = JobsRepository(db)
  job_ids = [_job(jobs, f'rec{i}') for i in range(3)]
  run_id = RunsRepository(db).create('discovery', 'test')
  recs = RecommendationsRepository(db)
  recs.upsert_many(
    run_id,
    1,
    [
      {
        'job_id': jid,
        'total_score': float(90 - i),
        'gate_pass': True,
        'gate_failures': [],
        'breakdown': {},
        'rationale': '',
      }
      for i, jid in enumerate(job_ids)
    ],
  )
  rows = recs.list_for_candidate(1, per_page=-1)
  assert len(rows) == 1


def test_companies_negative_limit_clamped(tmp_path: Path) -> None:
  repo = CompaniesRepository(_db(tmp_path))
  for i in range(3):
    repo.upsert(f'Acme {i} Ltd', vertical='fintech')
  rows = repo.list_companies(limit=-1)
  assert len(rows) == 1


def test_runs_negative_limit_clamped(tmp_path: Path) -> None:
  repo = RunsRepository(_db(tmp_path))
  for _ in range(3):
    repo.create('discovery', 'test')
  rows = repo.list_runs(limit=-1)
  assert len(rows) == 1


def test_runs_events_negative_limit_clamped(tmp_path: Path) -> None:
  repo = RunsRepository(_db(tmp_path))
  run_id = repo.create('discovery', 'test')
  for i in range(3):
    repo.log_event(run_id, 'info', 'node', f'msg {i}')
  rows = repo.events_since(run_id, 0, limit=-1)
  assert len(rows) == 1


# 2. cache non-dict --------------------------------------------------------


@pytest.mark.parametrize('raw', ['12345', '"hello"', '[1, 2]', 'true'])
def test_cache_non_dict_payload_returns_none(tmp_path: Path, raw: str) -> None:
  from llm_gateway.cache import SQLiteCache

  db = tmp_path / 'cache.db'
  cache = SQLiteCache(db)
  with closing(sqlite3.connect(str(db))) as conn:
    conn.execute(
      'INSERT INTO cache (key, created_at, payload) VALUES (?, ?, ?)',
      ('k', 9999999999.0, raw),
    )
    conn.commit()
  assert cache.get('k') is None
  with closing(sqlite3.connect(str(db))) as conn:
    assert (
      conn.execute('SELECT COUNT(*) FROM cache WHERE key = ?', ('k',))
      .fetchone()[0]
      == 0
    )


# 3. atomic enqueue --------------------------------------------------------


def test_create_if_not_active_sequential(tmp_path: Path) -> None:
  repo = RunsRepository(_db(tmp_path))
  first = repo.create_if_not_active('discovery', 'test')
  assert first is not None
  assert repo.create_if_not_active('discovery', 'test') is None
  repo.finish(first, 'success', {})
  second = repo.create_if_not_active('discovery', 'test')
  assert second is not None and second != first


def test_run_manager_second_enqueue_conflicts(
  tmp_path: Path, monkeypatch
) -> None:
  from job_hunter.core.errors import ConflictError

  manager = _manager(tmp_path, monkeypatch)
  manager.enqueue('discovery', 'test')
  with pytest.raises(ConflictError):
    manager.enqueue('discovery', 'test')


def test_run_manager_concurrent_enqueue_exactly_one_wins(
  tmp_path: Path, monkeypatch
) -> None:
  from job_hunter.core.errors import ConflictError

  manager = _manager(tmp_path, monkeypatch)
  winners: list[int] = []
  conflicts: list[int] = []
  barrier = threading.Barrier(4)

  def attempt() -> None:
    barrier.wait(timeout=10)
    try:
      winners.append(manager.enqueue('discovery', 'test'))
    except ConflictError:
      conflicts.append(1)

  threads = [threading.Thread(target=attempt) for _ in range(4)]
  for t in threads:
    t.start()
  for t in threads:
    t.join(timeout=20)
  assert len(winners) == 1, f'winners={winners} conflicts={conflicts}'
  assert len(conflicts) == 3


def test_run_manager_comment_no_partial_index_claim() -> None:
  import job_hunter.services.run_manager as rm

  src = inspect.getsource(rm.RunManager.enqueue)
  assert 'partial' not in src.lower()
  assert 'partial unique index' not in src.lower()


# 4. llm_stats error leak --------------------------------------------------


def test_llm_stats_error_is_generic(tmp_path: Path) -> None:
  gateway = tmp_path / 'gateway.db'
  gateway.write_bytes(b'')
  stats = GatewayStatsRepository().llm_stats(gateway)
  assert stats.get('error') == 'unavailable'
  assert 'no such table' not in str(stats.get('error'))
  assert stats['totals'] == {'calls': 0, 'tokens': 0, 'cost': 0.0}


# 5. gateway writers WAL ---------------------------------------------------


def test_cache_db_is_wal(tmp_path: Path) -> None:
  from llm_gateway.cache import SQLiteCache

  db = tmp_path / 'cache.db'
  SQLiteCache(db)
  with closing(sqlite3.connect(str(db))) as conn:
    mode = conn.execute('PRAGMA journal_mode').fetchone()[0]
  assert mode.lower() == 'wal'


def test_db_logger_db_is_wal(tmp_path: Path) -> None:
  from llm_gateway.db_logger import DBLogger

  db = tmp_path / 'gw.db'
  logger = DBLogger(db)
  try:
    with closing(sqlite3.connect(str(db))) as conn:
      mode = conn.execute('PRAGMA journal_mode').fetchone()[0]
    assert mode.lower() == 'wal'
  finally:
    logger.close()


# 6. llm_calls indexes -----------------------------------------------------


def test_db_logger_creates_llm_indexes(tmp_path: Path) -> None:
  from llm_gateway.db_logger import DBLogger

  db = tmp_path / 'gw.db'
  logger = DBLogger(db)
  try:
    with closing(sqlite3.connect(str(db))) as conn:
      names = {
        row[0]
        for row in conn.execute(
          "SELECT name FROM sqlite_master WHERE type='index'"
        )
      }
  finally:
    logger.close()
  assert {'idx_llm_timestamp', 'idx_llm_provider_ts', 'idx_llm_session_ts'} <= names


# 7. session() docstring warning -------------------------------------------


def test_session_docstring_warns_not_to_nest() -> None:
  doc = session.__doc__ or ''
  assert 'Do not nest session()' in doc


# 8. load_embeddings job_ids filter ----------------------------------------


def test_load_embeddings_filters_by_job_ids(tmp_path: Path) -> None:
  repo = JobsRepository(_db(tmp_path))
  ids = [_job(repo, f'emb{i}') for i in range(3)]
  for i, jid in enumerate(ids):
    repo.save_embedding(jid, 'model-x', [float(i), 1.0])
  assert set(repo.load_embeddings('model-x')) == set(ids)
  filtered = repo.load_embeddings('model-x', job_ids=[ids[1]])
  assert filtered == {ids[1]: [1.0, 1.0]}
  assert repo.load_embeddings('model-x', job_ids=[]) == {}
  assert repo.load_embeddings('model-x', job_ids=[999999]) == {}
