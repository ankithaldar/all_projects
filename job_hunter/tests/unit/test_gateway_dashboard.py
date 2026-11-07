#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the LLM gateway dashboard stats builder.'''


from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from llm_gateway.dashboard import _DashboardHandler


def _handler(db_path: Path) -> _DashboardHandler:
  '''Build a handler bound to a tmp gateway database.

  Args:
    db_path: Path to the gateway logging database.

  Returns:
    Handler instance whose stats builder reads that database.
  '''
  # BaseHTTPRequestHandler needs a live socket; _build_stats only
  # reads self.db_path, so skip __init__ entirely.
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


def test_build_stats_aggregates_seeded_calls(tmp_path: Path) -> None:
  '''Stats builder aggregates totals, error rate, and grouping.

  Args:
    tmp_path: Pytest temporary directory.
  '''
  db_path = tmp_path / 'gateway.db'
  _seed_calls(db_path)
  stats = _handler(db_path)._build_stats()
  summary = stats['summary']
  assert summary['total_calls'] == 2
  assert summary['input_tokens'] == 40
  assert summary['output_tokens'] == 60
  assert summary['total_cost'] == 2.0
  assert abs(summary['avg_latency_ms'] - 200.0) < 1e-9
  assert stats['error_rate'] == 0.5
  assert {row['status'] for row in stats['statuses']} == {
    'success', 'error',
  }
  assert len(stats['per_provider']) == 1
  assert stats['per_provider'][0]['provider'] == 'dummy'
  assert len(stats['recent_calls']) == 2


def test_build_stats_empty_database_is_zero(tmp_path: Path) -> None:
  '''Stats builder reports zeros when no calls table exists.

  Args:
    tmp_path: Pytest temporary directory.
  '''
  stats = _handler(tmp_path / 'missing.db')._build_stats()
  assert stats['summary']['total_calls'] == 0
  assert stats['summary']['input_tokens'] == 0
  assert stats['error_rate'] == 0.0
  assert stats['statuses'] == []
  assert stats['per_provider'] == []
  assert stats['recent_calls'] == []
