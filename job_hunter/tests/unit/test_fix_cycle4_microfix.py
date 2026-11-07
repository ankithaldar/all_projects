#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''RED tests for cycle-4 microfixes: dashboard connection close and
manual-inbox processed move.'''

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from llm_gateway.dashboard import _DashboardHandler

LINKEDIN_HTML = '''
<div class="base-card relative job-search-card">
  <a class="base-card__full-link" href="https://www.linkedin.com/jobs/view/1234?trk=x">
    <h3>Staff Data Scientist</h3>
  </a>
  <div class="base-search-card__subtitle">Razorpay</div>
  <div class="job-search-card__location">Bengaluru, Karnataka, India</div>
</div>
'''


def _handler(db_path: Path) -> _DashboardHandler:
  '''Build a handler bound to a tmp gateway database.

  Args:
    db_path: Path to the gateway logging database.

  Returns:
    Handler instance whose stats builder reads that database.
  '''
  handler = object.__new__(_DashboardHandler)
  handler.db_path = str(db_path)
  return handler


def _seed_calls(db_path: Path) -> None:
  '''Create the llm_calls table and seed two call rows.

  Args:
    db_path: Path to the gateway logging database.
  '''
  with closing(sqlite3.connect(str(db_path))) as conn:
    conn.execute(
      '''
      CREATE TABLE llm_calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        provider TEXT NOT NULL,
        model_used TEXT NOT NULL,
        input_tokens INTEGER NOT NULL,
        output_tokens INTEGER NOT NULL,
        latency_ms REAL NOT NULL,
        status TEXT NOT NULL,
        error TEXT,
        prompt_chars INTEGER NOT NULL,
        response_chars INTEGER NOT NULL,
        cost REAL NOT NULL,
        temperature REAL NOT NULL,
        system_prompt TEXT NOT NULL,
        session_id TEXT
      )
      '''
    )
    conn.execute(
      '''
      INSERT INTO llm_calls (
        timestamp, provider, model_used, input_tokens, output_tokens,
        latency_ms, status, prompt_chars, response_chars, cost,
        temperature, system_prompt, session_id
      ) VALUES (
        '2026-01-01T00:00:00Z', 'dummy', 'dummy-model', 10, 20,
        100.0, 'success', 5, 5, 0.5, 0.0, '', 'sess-1'
      )
      '''
    )
    conn.execute(
      '''
      INSERT INTO llm_calls (
        timestamp, provider, model_used, input_tokens, output_tokens,
        latency_ms, status, prompt_chars, response_chars, cost,
        temperature, system_prompt, session_id
      ) VALUES (
        '2026-01-01T00:01:00Z', 'dummy', 'dummy-model', 30, 40,
        300.0, 'error', 5, 5, 1.5, 0.0, '', 'sess-1'
      )
      '''
    )
    conn.commit()


def _assert_connection_closed(conn: sqlite3.Connection) -> None:
  '''Assert a sqlite3 connection is closed.

  Args:
    conn: Connection to check.

  Raises:
    AssertionError: If the connection is still usable.
  '''
  with pytest.raises(sqlite3.ProgrammingError):
    conn.execute('SELECT 1')


def test_build_stats_closes_connection(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  '''_build_stats closes its sqlite connection and keeps stats identical.

  Args:
    tmp_path: Pytest temporary directory.
    monkeypatch: Pytest fixture.
  '''
  db_path = tmp_path / 'gateway.db'
  _seed_calls(db_path)

  real_connect = sqlite3.connect
  created: list[sqlite3.Connection] = []

  def tracking_connect(*args, **kwargs) -> sqlite3.Connection:
    conn = real_connect(*args, **kwargs)
    created.append(conn)
    return conn

  monkeypatch.setattr(sqlite3, 'connect', tracking_connect)

  stats = _handler(db_path)._build_stats()

  assert stats['summary']['total_calls'] == 2
  assert stats['summary']['input_tokens'] == 40
  assert stats['error_rate'] == 0.5
  assert created, '_build_stats did not open a sqlite connection'
  for conn in created:
    _assert_connection_closed(conn)


def test_scan_inbox_moves_file_to_processed(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  '''scan_inbox moves ingested files to _processed/ with timestamp prefix.

  Args:
    tmp_path: Pytest temporary directory.
    monkeypatch: Pytest fixture.
  '''
  from job_hunter.adapters.manual_inbox import scan_inbox
  from job_hunter.core.config import AppSettings
  from job_hunter.core.db import run_migrations

  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  inbox = tmp_path / 'inbox'
  (inbox / 'linkedin').mkdir(parents=True)
  run_migrations(tmp_path / 'app.db')
  source = inbox / 'linkedin' / 'saved.html'
  source.write_text(LINKEDIN_HTML, encoding='utf-8')
  config_path = tmp_path / 'app.yaml'
  config_path.write_text('salary_hard_floor_lpa: 45\n', encoding='utf-8')
  settings = AppSettings(config_path)

  count = asyncio.run(scan_inbox(settings))

  assert count == 1
  assert not source.exists(), 'inbox file was not moved out of inbox'
  processed = tmp_path / 'inbox' / '_processed'
  moved = list(processed.glob('*_saved.html'))
  assert len(moved) == 1, f'expected 1 processed file, got {list(processed.iterdir()) if processed.exists() else "no _processed dir"}'
  prefix = moved[0].name.split('_', 1)[0]
  assert prefix.isdigit(), f'expected timestamp prefix, got {moved[0].name}'
