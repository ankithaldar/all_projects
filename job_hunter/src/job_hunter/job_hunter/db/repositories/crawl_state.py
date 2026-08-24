#!/usr/bin/env python
# -- coding: utf-8 --

'''Crawl watermark state per source or company scope.'''


from __future__ import annotations

from pathlib import Path
from typing import Optional

from job_hunter.core.db import connect, session


class CrawlStateRepository:
  '''Track cursors, failures, and cooldowns per crawl scope.'''

  def __init__(self, db_path: str | Path) -> None:
    '''Initialize the repository.

    Args:
      db_path: Application database path.
    '''
    self._db_path = str(db_path)

  def get_cursor(self, scope: str) -> Optional[str]:
    '''Return the stored cursor for a scope.

    Args:
      scope: Scope key like 'source:greenhouse' or 'company:12'.

    Returns:
      Cursor value or None.
    '''
    row = connect(self._db_path, readonly=True).execute(
      'SELECT cursor FROM crawl_state WHERE scope = ?', (scope,),
    ).fetchone()
    return row['cursor'] if row else None

  def set_success(self, scope: str, cursor: Optional[str]) -> None:
    '''Record a successful fetch.

    Args:
      scope: Scope key.
      cursor: New watermark.
    '''
    with session(self._db_path) as conn:
      conn.execute(
        'INSERT INTO crawl_state (scope, cursor, last_success_at, '
        "consecutive_failures) VALUES (?, ?, datetime('now'), 0) "
        'ON CONFLICT(scope) DO UPDATE SET cursor = excluded.cursor, '
        "last_success_at = datetime('now'), consecutive_failures = 0",
        (scope, cursor),
      )

  def set_failure(self, scope: str, cooldown_hours: int = 24) -> int:
    '''Record a failure and open a cooldown after repeated issues.

    Args:
      scope: Scope key.
      cooldown_hours: Cooldown applied from the fifth consecutive failure.

    Returns:
      Updated consecutive-failure count.
    '''
    with session(self._db_path) as conn:
      conn.execute(
        'INSERT INTO crawl_state (scope, consecutive_failures) VALUES (?, 1) '
        'ON CONFLICT(scope) DO UPDATE SET '
        'consecutive_failures = consecutive_failures + 1',
        (scope,),
      )
      row = conn.execute(
        'SELECT consecutive_failures FROM crawl_state WHERE scope = ?',
        (scope,),
      ).fetchone()
      count = int(row['consecutive_failures'])
      if count >= 5:
        conn.execute(
          'UPDATE crawl_state SET cooldown_until = datetime(\'now\', ?) '
          'WHERE scope = ?',
          (f'+{cooldown_hours * count // 5} hours', scope),
        )
      return count

  def is_cooled_down(self, scope: str) -> bool:
    '''Return whether a scope is inside its cooldown window.

    Args:
      scope: Scope key.

    Returns:
      True when cooling down.
    '''
    row = connect(self._db_path, readonly=True).execute(
      "SELECT 1 FROM crawl_state WHERE scope = ? AND cooldown_until > datetime('now')",
      (scope,),
    ).fetchone()
    return row is not None
