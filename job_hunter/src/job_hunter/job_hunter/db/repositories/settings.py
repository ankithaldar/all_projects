#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Settings repository: JSON values with config-file defaults.'''

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict

from job_hunter.core.config import AppSettings
from job_hunter.core.db import fetch_all, fetch_one, session


class SettingsRepository:
  '''Read/write the settings table and seed defaults from app.yaml.'''

  def __init__(self, db_path: str | Path) -> None:
    '''Initialize the repository.

    Args:
      db_path: Application database path.
    '''
    self._db_path = str(db_path)

  def get(self, key: str, default: Any = None) -> Any:
    '''Fetch one setting.

    Args:
      key: Setting name.
      default: Value when missing.

    Returns:
      Decoded JSON value.
    '''
    with fetch_one(
      self._db_path,
      'SELECT value_json FROM settings WHERE key = ?',
      (key,),
    ) as row:
      return json.loads(row['value_json']) if row else default

  def all_settings(self) -> Dict[str, Any]:
    '''Return every stored setting.

    Returns:
      Mapping of key to decoded value.
    '''
    with fetch_all(
      self._db_path,
      'SELECT key, value_json FROM settings',
    ) as rows:
      return {row['key']: json.loads(row['value_json']) for row in rows}

  def put(self, key: str, value: Any) -> None:
    '''Upsert one setting.

    Args:
      key: Setting name.
      value: JSON-serializable value.
    '''
    with session(self._db_path) as conn:
      conn.execute(
        'INSERT INTO settings (key, value_json) VALUES (?, ?) '
        'ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json, '
        "updated_at = datetime('now')",
        (key, json.dumps(value, ensure_ascii=False)),
      )

  def ensure_defaults(self, settings: AppSettings) -> None:
    '''Insert any missing default keys derived from app.yaml.

    Args:
      settings: Loaded application settings.
    '''
    defaults: Dict[str, Any] = {
      'schedule': settings.schedule,
      'sources': settings.sources,
      'discovery': settings.discovery,
      'scoring_weights': settings.scoring_weights,
      'embeddings': settings.embeddings,
      'salary_hard_floor_lpa': settings.salary_floor_lpa,
    }
    existing = self.all_settings()
    for key, value in defaults.items():
      if key not in existing:
        self.put(key, value)


class GatewayStatsRepository:
  '''Read-only summaries over the gateway ``llm_calls`` table.

  The gateway database is a different file from the app database,
  so the path is passed to :meth:`llm_stats` (or at construction)
  rather than assumed to be the app path.
  '''

  def __init__(self, db_path: str | Path | None = None) -> None:
    '''Initialize the repository.

    Args:
      db_path: Optional default gateway database path.
    '''
    self._db_path = str(db_path) if db_path is not None else None

  def llm_stats(
    self,
    db_path: str | Path | None = None,
    days: int = 7,
  ) -> Dict[str, Any]:
    '''Summarize gateway call costs grouped by provider and node.

    Args:
      db_path: Gateway database path; falls back to the path given
        at construction when omitted.
      days: Lookback window.

    Returns:
      Mapping ``{by_provider, by_node, totals}`` matching the
      ``/api/llm/stats`` route shape, with an ``error`` key instead
      of raising on a missing or unreadable database.
    '''
    empty: Dict[str, Any] = {
      'by_provider': [],
      'by_node': [],
      'totals': {'calls': 0, 'tokens': 0, 'cost': 0.0},
    }
    path = (
      Path(db_path)
      if db_path is not None
      else (Path(self._db_path) if self._db_path is not None else None)
    )
    if path is None or not path.exists():
      return empty
    window = f'-{max(1, days)} days'
    try:
      with fetch_all(
        path,
        'SELECT provider, COUNT(*) AS calls, '
        'SUM(input_tokens + output_tokens) AS tokens, SUM(cost) AS cost '
        "FROM llm_calls WHERE timestamp >= datetime('now', ?) "
        'GROUP BY provider ORDER BY cost DESC',
        (window,),
      ) as by_provider_rows:
        by_provider = [dict(row) for row in by_provider_rows]
      with fetch_all(
        path,
        "SELECT COALESCE(session_id, '') AS node, COUNT(*) AS calls, "
        'SUM(input_tokens + output_tokens) AS tokens, SUM(cost) AS cost '
        "FROM llm_calls WHERE timestamp >= datetime('now', ?) "
        'GROUP BY session_id ORDER BY cost DESC LIMIT 25',
        (window,),
      ) as by_node_rows:
        by_node = [dict(row) for row in by_node_rows]
      with fetch_one(
        path,
        'SELECT COUNT(*) AS calls, '
        'COALESCE(SUM(input_tokens + output_tokens), 0) AS tokens, '
        'COALESCE(SUM(cost), 0.0) AS cost FROM llm_calls '
        "WHERE timestamp >= datetime('now', ?)",
        (window,),
      ) as totals_row:
        totals = dict(totals_row) if totals_row else empty['totals']
    except sqlite3.Error:
      return {**empty, 'error': 'unavailable'}
    return {
      'by_provider': by_provider,
      'by_node': by_node,
      'totals': totals,
    }
