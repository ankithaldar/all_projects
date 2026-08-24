#!/usr/bin/env python
# -- coding: utf-8 --

'''Runs and run_events persistence plus the pending-run claim protocol.'''


from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from job_hunter.core.db import connect, session


class RunsRepository:
  '''Manage run lifecycle rows and their event stream.'''

  def __init__(self, db_path: str | Path) -> None:
    '''Initialize the repository.

    Args:
      db_path: Application database path.
    '''
    self._db_path = str(db_path)

  def create(self, kind: str, triggered_by: str = 'scheduler') -> int:
    '''Insert a pending run.

    Args:
      kind: Run kind.
      triggered_by: Origin of the run.

    Returns:
      New run id.
    '''
    with session(self._db_path) as conn:
      cur = conn.execute(
        'INSERT INTO runs (kind, triggered_by) VALUES (?, ?)',
        (kind, triggered_by),
      )
      return int(cur.lastrowid)

  def has_active(self) -> bool:
    '''Return whether a run is pending or running.

    Returns:
      True when an active run exists.
    '''
    row = connect(self._db_path, readonly=True).execute(
      "SELECT COUNT(*) AS n FROM runs WHERE status IN ('pending','running')",
    ).fetchone()
    return bool(row and row['n'])

  def claim_pending(self, run_id: Optional[int] = None) -> Optional[int]:
    '''Atomically claim one pending run for execution.

    Args:
      run_id: Specific run to claim, else oldest pending.

    Returns:
      Claimed run id or None.
    '''
    conn = connect(self._db_path)
    try:
      conn.execute('BEGIN IMMEDIATE')
      if run_id is not None:
        row = conn.execute(
          "SELECT id FROM runs WHERE id = ? AND status = 'pending'",
          (run_id,),
        ).fetchone()
      else:
        row = conn.execute(
          "SELECT id FROM runs WHERE status = 'pending' ORDER BY id LIMIT 1",
        ).fetchone()
      if row is None:
        conn.execute('ROLLBACK')
        return None
      claimed = int(row['id'])
      conn.execute(
        "UPDATE runs SET status = 'running', started_at = datetime('now') "
        'WHERE id = ?',
        (claimed,),
      )
      conn.execute('COMMIT')
      return claimed
    except Exception:
      if conn.in_transaction:
        conn.execute('ROLLBACK')
      raise
    finally:
      conn.close()

  def finish(
    self,
    run_id: int,
    status: str,
    stats: Dict[str, Any],
    error_text: Optional[str] = None,
  ) -> None:
    '''Finalize a run with stats and terminal status.

    Args:
      run_id: Run id.
      status: One of success|partial|failed|cancelled.
      stats: Counter mapping persisted as JSON.
      error_text: Optional error summary.
    '''
    with session(self._db_path) as conn:
      conn.execute(
        "UPDATE runs SET status = ?, finished_at = datetime('now'), "
        'stats_json = ?, error_text = ? WHERE id = ?',
        (status, json.dumps(stats, ensure_ascii=False), error_text, run_id),
      )

  def mark_orphans_failed(self, ttl_minutes: int = 120) -> List[int]:
    '''Fail stale running runs (crash recovery).

    Args:
      ttl_minutes: Age threshold in minutes.

    Returns:
      List of affected run ids.
    '''
    with session(self._db_path) as conn:
      rows = conn.execute(
        "SELECT id FROM runs WHERE status = 'running' AND started_at < "
        "datetime('now', ?)",
        (f'-{ttl_minutes} minutes',),
      ).fetchall()
      ids = [int(row['id']) for row in rows]
      if ids:
        marks = ','.join('?' * len(ids))
        conn.execute(
          f'UPDATE runs SET status = "failed", finished_at = datetime(\'now\'), '
          f"error_text = 'orphaned run recovered' WHERE id IN ({marks})",
          ids,
        )
      return ids

  def get(self, run_id: int) -> Optional[Dict[str, Any]]:
    '''Fetch one run row.

    Args:
      run_id: Run id.

    Returns:
      Row mapping or None.
    '''
    row = connect(self._db_path, readonly=True).execute(
      'SELECT * FROM runs WHERE id = ?', (run_id,),
    ).fetchone()
    return dict(row) if row else None

  def list_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
    '''List recent runs newest first.

    Args:
      limit: Page size.

    Returns:
      Rows as mappings.
    '''
    rows = connect(self._db_path, readonly=True).execute(
      'SELECT * FROM runs ORDER BY id DESC LIMIT ?', (limit,),
    ).fetchall()
    return [dict(row) for row in rows]

  def log_event(
    self,
    run_id: int,
    level: str,
    node: str,
    message: str,
    data: Optional[Dict[str, Any]] = None,
  ) -> int:
    '''Append a run event.

    Args:
      run_id: Run id.
      level: debug|info|warn|error.
      node: Node or component name.
      message: Human readable text.
      data: Optional structured payload.

    Returns:
      Event id.
    '''
    with session(self._db_path) as conn:
      cur = conn.execute(
        'INSERT INTO run_events (run_id, level, node, message, data_json) '
        'VALUES (?, ?, ?, ?, ?)',
        (
          run_id,
          level,
          node,
          message,
          json.dumps(data, ensure_ascii=False) if data else None,
        ),
      )
      return int(cur.lastrowid)

  def events_since(self, run_id: int, last_id: int, limit: int = 200) -> List[Dict[str, Any]]:
    '''Return events after a cursor id.

    Args:
      run_id: Run id.
      last_id: Exclusive lower bound.
      limit: Max rows.

    Returns:
      Event rows.
    '''
    rows = connect(self._db_path, readonly=True).execute(
      'SELECT * FROM run_events WHERE run_id = ? AND id > ? ORDER BY id LIMIT ?',
      (run_id, last_id, limit),
    ).fetchall()
    return [dict(row) for row in rows]
