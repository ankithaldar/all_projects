#!/usr/bin/env python
# -- coding: utf-8 --

'''Settings repository: JSON values with config-file defaults.'''


from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from job_hunter.core.config import AppSettings
from job_hunter.core.db import connect, session


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
    row = connect(self._db_path, readonly=True).execute(
      'SELECT value_json FROM settings WHERE key = ?', (key,),
    ).fetchone()
    return json.loads(row['value_json']) if row else default

  def all_settings(self) -> Dict[str, Any]:
    '''Return every stored setting.

    Returns:
      Mapping of key to decoded value.
    '''
    rows = connect(self._db_path, readonly=True).execute(
      'SELECT key, value_json FROM settings',
    ).fetchall()
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
