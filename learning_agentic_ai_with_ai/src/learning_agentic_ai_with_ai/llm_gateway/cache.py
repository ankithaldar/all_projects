#!/usr/bin/env python
# -- coding: utf-8 --

'''SQLite-backed response cache.'''


from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

from llm_gateway.schemas import GatewayRequest


class SQLiteCache:
  '''Persistent cache for identical gateway requests.'''

  def __init__(self, db_path: str | Path, ttl_seconds: int = 86400) -> None:
    '''Initialize the cache database.

    Args:
      db_path: SQLite database path.
      ttl_seconds: Cache TTL in seconds. Zero or negative disables expiry.
    '''
    self._db_path = str(db_path)
    self._ttl_seconds = ttl_seconds

    Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(self._db_path) as conn:
      conn.execute(
        '''
        CREATE TABLE IF NOT EXISTS cache (
          key TEXT PRIMARY KEY,
          created_at REAL NOT NULL,
          payload TEXT NOT NULL
        )
        '''
      )
      conn.commit()

  def get(self, key: str) -> Optional[Dict[str, Any]]:
    '''Fetch a cached payload.

    Args:
      key: Cache key.

    Returns:
      Cached payload dictionary or None.
    '''
    with sqlite3.connect(self._db_path) as conn:
      row = conn.execute(
        'SELECT created_at, payload FROM cache WHERE key = ?',
        (key,),
      ).fetchone()

    if row is None:
      return None

    created_at, payload = row

    if self._ttl_seconds > 0 and time.time() - created_at > self._ttl_seconds:
      return None

    return json.loads(payload)

  def set(self, key: str, payload: Dict[str, Any]) -> None:
    '''Store a cached payload.

    Args:
      key: Cache key.
      payload: JSON-serializable payload.
    '''
    serialized = json.dumps(payload, default=str)

    with sqlite3.connect(self._db_path) as conn:
      conn.execute(
        '''
        INSERT OR REPLACE INTO cache (key, created_at, payload)
        VALUES (?, ?, ?)
        ''',
        (key, time.time(), serialized),
      )
      conn.commit()

  @staticmethod
  def make_key(
    request: GatewayRequest,
    include_temperature: bool = False,
  ) -> str:
    '''Create a stable cache key for a gateway request.

    Args:
      request: Gateway request.
      include_temperature: Whether temperature should affect the cache key.

    Returns:
      SHA256 cache key.
    '''
    payload: Dict[str, Any] = {
      'prompt': request.prompt,
      'system_prompt': request.system_prompt,
      'messages': [
        message.model_dump(exclude_none=True, mode='json')
        for message in request.messages or []
      ],
      'tools': [
        tool.model_dump(exclude_none=True, mode='json')
        for tool in request.tools or []
      ],
      'tool_choice': request.tool_choice,
    }

    if include_temperature:
      payload['temperature'] = request.temperature

    canonical = json.dumps(
      payload,
      sort_keys=True,
      separators=(',', ':'),
      default=str,
    )

    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
