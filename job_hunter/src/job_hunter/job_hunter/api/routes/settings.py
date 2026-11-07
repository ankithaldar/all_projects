#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Settings and dashboard/LLM statistics endpoints.'''


from __future__ import annotations

import math
from typing import Any, Dict

from fastapi import APIRouter, Body, Depends, HTTPException
from job_hunter.api.deps import get_settings
from job_hunter.core.config import AppSettings
from job_hunter.db.repositories.recommendations import (
  RecommendationsRepository,
)
from job_hunter.db.repositories.settings import (
  GatewayStatsRepository,
  SettingsRepository,
)

router = APIRouter(prefix='/api', tags=['settings'])

_ALLOWED_SETTINGS = {
  'scoring_weights',
  'salary_hard_floor_lpa',
  'discovery.max_llm_calls_per_run',
  'discovery.enrich_per_run_cap',
}
_CAP_SETTINGS = {
  'discovery.max_llm_calls_per_run',
  'discovery.enrich_per_run_cap',
}


def _reject(detail: str) -> None:
  '''Reject a setting value.

  Args:
    detail: Generic error detail.

  Raises:
    HTTPException: Always 422.
  '''
  raise HTTPException(status_code=422, detail=detail)


def _check_weights(value: Any) -> None:
  '''Validate scoring weights.

  Args:
    value: Candidate value.

  Raises:
    HTTPException: 422 on invalid weights.
  '''
  if not isinstance(value, dict):
    _reject('scoring_weights must be an object')
  for item in value.values():
    if isinstance(item, bool) or not isinstance(item, (int, float)):
      _reject('scoring_weights values must be numbers')
    try:
      number = float(item)
    except OverflowError:
      _reject('scoring_weights values must be finite')
    if not math.isfinite(number):
      _reject('scoring_weights values must be finite')


def _check_floor(value: Any) -> None:
  '''Validate the salary floor.

  Args:
    value: Candidate value.

  Raises:
    HTTPException: 422 on invalid floor.
  '''
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    _reject('salary floor must be a number')
  try:
    number = float(value)
  except OverflowError:
    _reject('salary floor out of range')
  if not math.isfinite(number) or not 0 <= number <= 1000:
    _reject('salary floor out of range')


def _validate_setting(key: str, value: Any) -> None:
  '''Validate one setting value against its allowlist rule.

  Args:
    key: Setting name.
    value: Candidate value.

  Raises:
    HTTPException: 422 on unknown key or invalid value.
  '''
  if key not in _ALLOWED_SETTINGS:
    _reject('unknown setting key')
  if key == 'scoring_weights':
    _check_weights(value)
  elif key == 'salary_hard_floor_lpa':
    _check_floor(value)
  elif key in _CAP_SETTINGS:
    if isinstance(value, bool) or not isinstance(value, int):
      _reject('discovery cap must be an integer')
    if not 1 <= value <= 10000:
      _reject('discovery cap out of range')


@router.get('/settings')
def get_all_settings(
  settings: AppSettings = Depends(get_settings),
) -> Dict[str, Any]:
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
  _validate_setting(key, value)
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
  return GatewayStatsRepository().llm_stats(
    settings.gateway_db_path, days=days,
  )
