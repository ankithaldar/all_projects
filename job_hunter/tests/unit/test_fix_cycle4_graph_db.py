#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''RED tests for cycle-4 graph/db gateway fixes (one per spec item).'''

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import time
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Dict

import pytest


def test_dblogger_worker_survives_insert_failure(
  tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
  '''DROP TABLE then log() -> worker alive, error logged, close in time.'''
  from llm_gateway.db_logger import DBLogger
  from llm_gateway.schemas import LogRecord

  db_path = tmp_path / 'gw.db'
  logger = DBLogger(db_path)
  try:
    with closing(sqlite3.connect(str(db_path))) as conn:
      conn.execute('DROP TABLE llm_calls')
      conn.commit()
    record = LogRecord(provider='p', model_used='m', status='success')
    with caplog.at_level(logging.ERROR, logger='llm_gateway.db_logger'):
      logger.log(record)
      deadline = time.monotonic() + 3.0
      while time.monotonic() < deadline:
        if any(
          r.levelno >= logging.ERROR for r in caplog.records
        ):
          break
        time.sleep(0.05)
    assert logger._thread.is_alive(), 'worker thread died on insert error'
    assert any(
      r.levelno >= logging.ERROR for r in caplog.records
    ), 'insert error was not logged'
    closer = threading.Thread(target=logger.close)
    closer.start()
    closer.join(timeout=5.0)
    assert not closer.is_alive(), 'close() hung past 5s timeout guard'
  finally:
    if not logger._closed:
      logger.close()


def test_connect_pragma_failure_closes_connection(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  '''_connect pragma failure closes the connection and re-raises.'''
  import llm_gateway.db_logger as db_logger_mod

  closed = {'called': False}

  class FailingConn:
    '''Connection whose PRAGMA execution always fails.'''

    def execute(self, *args: Any, **kwargs: Any):
      '''Raise on any pragma execution.'''
      raise sqlite3.OperationalError('pragma boom')

    def close(self) -> None:
      '''Record close calls.'''
      closed['called'] = True

  failing = FailingConn()
  monkeypatch.setattr(
    db_logger_mod.sqlite3, 'connect', lambda *a, **kw: failing
  )
  with pytest.raises(sqlite3.OperationalError):
    db_logger_mod._connect(str(Path('/tmp/fake.db')))
  assert closed['called'], '_connect leaked connection on pragma failure'


def test_checkpoint_connection_enforces_foreign_keys(
  tmp_path: Path,
) -> None:
  '''open_checkpointer connection reports foreign_keys=1.'''
  from job_hunter.graph.checkpointing import open_checkpointer

  async def _run() -> int:
    async with open_checkpointer(tmp_path / 'ckpt.sqlite') as saver:
      assert saver is not None
      cursor = await saver.conn.execute('PRAGMA foreign_keys')
      row = await cursor.fetchone()
      return int(row[0])

  assert asyncio.run(_run()) == 1


def test_run_migrations_skips_applied_inside_txn(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  '''Applied set changing after pre-check is honoured under BEGIN.'''
  import job_hunter.core.db as db_mod
  from job_hunter.core.db import run_migrations

  db = tmp_path / 'app.db'
  run_migrations(db)

  @contextmanager
  def lying_fetch_all(db_path, sql, params=()):
    '''Pretend no migration is applied (pre-check lies).'''
    yield []

  monkeypatch.setattr(db_mod, 'fetch_all', lying_fetch_all)
  # Must skip inside the txn and not raise DatabaseError (duplicate
  # column on the 0003 ALTER would be raised by the old code).
  run_migrations(db)


def test_0004_embedding_index_used_by_unfiltered_load(
  tmp_path: Path,
) -> None:
  '''Fresh migrated DB has idx_job_embeddings_model_job and uses it.'''
  from job_hunter.core.db import connect, run_migrations

  db = tmp_path / 'app.db'
  run_migrations(db)
  with closing(connect(db, readonly=True)) as conn:
    names = {
      row['name']
      for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
      )
    }
    assert 'idx_job_embeddings_model_job' in names
    rows = conn.execute(
      'EXPLAIN QUERY PLAN SELECT job_id, dim, vector '
      'FROM job_embeddings WHERE model = ?',
      ('m',),
    ).fetchall()
  details = ' '.join(str(row[-1]) for row in rows)
  assert 'idx_job_embeddings_model_job' in details, details
  assert 'SEARCH' in details, details


def test_state_annotations_match_runtime_dicts() -> None:
  '''plan and enriched annotations are dict, not pydantic models.'''
  from typing import get_type_hints

  from job_hunter.graph.state import DiscoveryState

  hints = get_type_hints(DiscoveryState)

  def _is_dict_annotation(annotation: Any) -> bool:
    return (
      annotation is dict
      or getattr(annotation, '__origin__', None) is dict
    )

  assert _is_dict_annotation(hints['plan']), hints['plan']
  assert _is_dict_annotation(hints['enriched']), hints['enriched']


def test_normalize_inserted_converts_dicts() -> None:
  '''Checkpoint-resumed dict inserted -> objects with .job_id.'''
  try:
    from job_hunter.graph.nodes import _normalize_inserted
  except ImportError as exc:
    pytest.fail(f'_normalize_inserted helper missing: {exc}')

  from job_hunter.core.models import NormalizedJob, RawJobRecord

  job = NormalizedJob(
    job_id=42,
    raw=RawJobRecord(
      source_key='lever', url='https://x.example/1', title='T',
    ),
    canonical_url='https://x.example/1',
    content_hash='h1',
  )
  dumped: Dict[str, Any] = job.model_dump()
  state: Dict[str, Any] = {'inserted': [dumped, job]}
  normalized = _normalize_inserted(state)
  assert normalized[0].job_id == 42
  assert normalized[0].content_hash == 'h1'
  assert normalized[0].raw.title == 'T'
  assert normalized[1] is job
  assert state['inserted'][0].job_id == 42


def test_get_client_key_includes_config_and_env(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  '''Same db/cache but different config/env -> different clients.'''
  from job_hunter.llm import client as client_mod

  client_mod.reset_client()

  class FakeSettings:
    '''Minimal settings surface used by get_client.'''

    def __init__(self, config: str, env: str = '/tmp/.env') -> None:
      '''Bind the four gateway paths.'''
      self.gateway_db_path = Path('/tmp/db.db')
      self.gateway_cache_path = Path('/tmp/cache.db')
      self.gateway_config_path = Path(config)
      self.gateway_env_path = Path(env)

  class FakeClient:
    '''Stand-in for GatewayClient that avoids real gateway wiring.'''

    def __init__(self, settings: Any) -> None:
      '''Store settings.'''
      self.settings = settings

  monkeypatch.setattr(client_mod, 'GatewayClient', FakeClient)
  first = client_mod.get_client(FakeSettings('/tmp/a.yaml'))
  second = client_mod.get_client(FakeSettings('/tmp/b.yaml'))
  assert first is not second
  different_env = client_mod.get_client(
    FakeSettings('/tmp/a.yaml', env='/tmp/other.env')
  )
  assert different_env is not first
  same = client_mod.get_client(FakeSettings('/tmp/a.yaml'))
  assert same is first
  client_mod.reset_client()


def test_nodes_lines_within_80_chars() -> None:
  '''nodes.py has no line longer than 80 chars (D13 wrapping).'''
  path = (
    Path(__file__).resolve().parents[2]
    / 'src'
    / 'job_hunter'
    / 'job_hunter'
    / 'graph'
    / 'nodes.py'
  )
  lines = path.read_text(encoding='utf-8').splitlines()
  for number, line in enumerate(lines, start=1):
    assert len(line) <= 80, f'line {number} len={len(line)}: {line}'
