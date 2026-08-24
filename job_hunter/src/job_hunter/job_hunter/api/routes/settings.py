#!/usr/bin/env python
# -- coding: utf-8 --

'''Settings and dashboard/LLM statistics endpoints.'''


from __future__ import annotations

import sqlite3
from typing import Any, Dict, List

from fastapi import APIRouter, Body, Depends
from job_hunter.api.deps import get_settings
from job_hunter.core.config import AppSettings
from job_hunter.db.repositories.recommendations import RecommendationsRepository
from job_hunter.db.repositories.settings import SettingsRepository

router = APIRouter(prefix='/api', tags=['settings'])


@router.get('/settings')
def get_all_settings(settings: AppSettings = Depends(get_settings)) -> Dict[str, Any]:
  '''Return every stored setting.

  Args:
    settings: App settings.

  Returns:
    Settings mapping.
  '''
  return SettingsRepository(settings.db_path).all_settings()


@router.put('/settings')
def put_setting(
  key: str,
  value: Any = Body(...),
  settings: AppSettings = Depends(get_settings),
) -> Dict[str, str]:
  '''Upsert one setting by key.

  Args:
    key: Setting name.
    value: JSON value.
    settings: App settings.

  Returns:
    Acknowledgement.
  '''
  SettingsRepository(settings.db_path).put(key, value)
  return {'key': key, 'status': 'saved'}


@router.get('/stats/dashboard')
def dashboard(settings: AppSettings = Depends(get_settings)) -> Dict[str, Any]:
  '''Return dashboard KPI counters.

  Args:
    settings: App settings.

  Returns:
    Counter mapping.
  '''
  return RecommendationsRepository(settings.db_path).counts()


@router.get('/llm/stats')
def llm_stats(
  days: int = 7,
  settings: AppSettings = Depends(get_settings),
) -> Dict[str, Any]:
  '''Summarize gateway call costs grouped by provider and node.

  Args:
    days: Lookback window.
    settings: App settings.

  Returns:
    {by_provider: [...], by_node: [...], totals: {...}}.
  '''
  db_path = settings.gateway_db_path
  empty: Dict[str, Any] = {
    'by_provider': [], 'by_node': [],
    'totals': {'calls': 0, 'tokens': 0, 'cost': 0.0},
  }
  if not db_path.exists():
    return empty
  try:
    conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    by_provider = conn.execute(
      "SELECT provider, COUNT(*) AS calls, SUM(input_tokens + output_tokens) AS tokens, "
      'SUM(cost) AS cost FROM llm_calls '
      "WHERE timestamp >= datetime('now', ?) GROUP BY provider ORDER BY cost DESC",
      (f'-{max(1, days)} days',),
    ).fetchall()
    by_node = conn.execute(
      "SELECT COALESCE(session_id, '') AS node, COUNT(*) AS calls, "
      'SUM(input_tokens + output_tokens) AS tokens, SUM(cost) AS cost '
      "FROM llm_calls WHERE timestamp >= datetime('now', ?) "
      'GROUP BY session_id ORDER BY cost DESC LIMIT 25',
      (f'-{max(1, days)} days',),
    ).fetchall()
    totals = conn.execute(
      "SELECT COUNT(*) AS calls, "
      'COALESCE(SUM(input_tokens + output_tokens), 0) AS tokens, '
      "COALESCE(SUM(cost), 0.0) AS cost FROM llm_calls "
      "WHERE timestamp >= datetime('now', ?)",
      (f'-{max(1, days)} days',),
    ).fetchone()
    conn.close()
  except sqlite3.Error as exc:
    return {**empty, 'error': str(exc)}
  return {
    'by_provider': [dict(row) for row in by_provider],
    'by_node': [dict(row) for row in by_node],
    'totals': dict(totals) if totals else empty['totals'],
  }
