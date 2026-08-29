#!/usr/bin/env python
# -- coding: utf-8 --

'''Local SQLite persistence for agent sessions, events, memory, and audits.

Why SQLite? It is embedded (no server), transactional, and universally
available - perfect for a course that must run anywhere. Production systems
would swap this class for Postgres; the interface stays identical.
'''


from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

_SCHEMA = '''
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  meta_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  ts TEXT NOT NULL,
  event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);

CREATE TABLE IF NOT EXISTS memory (
  session_id TEXT NOT NULL,
  key TEXT NOT NULL,
  value_json TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (session_id, key)
);

CREATE TABLE IF NOT EXISTS tool_calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  ts TEXT NOT NULL,
  server TEXT NOT NULL,
  tool TEXT NOT NULL,
  args_json TEXT NOT NULL DEFAULT '{}',
  result_json TEXT NOT NULL DEFAULT '{}',
  ok INTEGER NOT NULL,
  error TEXT,
  latency_ms REAL NOT NULL DEFAULT 0,
  approved INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_session ON tool_calls(session_id);
'''


class SessionRecord(BaseModel):
  '''A persisted agent session.'''

  model_config = ConfigDict(extra='ignore')

  session_id: str
  created_at: str
  meta: Dict[str, Any] = Field(default_factory=dict)


class EventRecord(BaseModel):
  '''One execution-history event (audit trail).'''

  model_config = ConfigDict(extra='ignore')

  id: int
  session_id: str
  ts: str
  event_type: str
  payload: Dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
  '''Persisted audit row for one tool invocation.'''

  model_config = ConfigDict(extra='ignore')

  id: int
  session_id: str
  ts: str
  server: str
  tool: str
  args: Dict[str, Any] = Field(default_factory=dict)
  result: Dict[str, Any] = Field(default_factory=dict)
  ok: bool = True
  error: Optional[str] = None
  latency_ms: float = 0.0
  approved: bool = True


def _utc_now() -> str:
  '''Return the current UTC time as ISO string.

  Returns:
    ISO 8601 timestamp.
  '''
  return datetime.now(timezone.utc).isoformat()


class AgentStore:
  '''Thread-safe SQLite store for sessions, events, memory, tool audits.'''

  def __init__(self, db_path: Path | str) -> None:
    '''Open (and initialize) the store.

    Args:
      db_path: Path to the SQLite database file.
    '''
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    self._lock = threading.Lock()
    self._conn = sqlite3.connect(str(path), check_same_thread=False)
    self._conn.row_factory = sqlite3.Row
    self._conn.executescript(_SCHEMA)
    self._conn.commit()

  def close(self) -> None:
    '''Close the database connection.'''
    with self._lock:
      self._conn.close()

  def ensure_session(
    self,
    session_id: str,
    meta: Optional[Dict[str, Any]] = None,
  ) -> SessionRecord:
    '''Create a session row if absent.

    Args:
      session_id: Unique session identifier.
      meta: Arbitrary session metadata.

    Returns:
      The persisted session record.
    '''
    with self._lock:
      row = self._conn.execute(
        'SELECT session_id, created_at, meta_json '
        'FROM sessions WHERE session_id = ?',
        (session_id,),
      ).fetchone()
      if row is None:
        created = _utc_now()
        self._conn.execute(
          'INSERT INTO sessions (session_id, created_at, meta_json) '
          'VALUES (?, ?, ?)',
          (session_id, created, json.dumps(meta or {})),
        )
        self._conn.commit()
        return SessionRecord(
          session_id=session_id,
          created_at=created,
          meta=meta or {},
        )

      return SessionRecord(
        session_id=row['session_id'],
        created_at=row['created_at'],
        meta=json.loads(row['meta_json'] or '{}'),
      )

  def log_event(
    self,
    session_id: str,
    event_type: str,
    payload: Optional[Dict[str, Any]] = None,
  ) -> EventRecord:
    '''Append one event to the execution history.

    Args:
      session_id: Owning session.
      event_type: Event name, e.g. 'llm_response', 'task_start'.
      payload: Structured event data.

    Returns:
      The persisted event record.
    '''
    ts = _utc_now()
    payload_json = json.dumps(payload or {}, default=str)
    with self._lock:
      cursor = self._conn.execute(
        'INSERT INTO events (session_id, ts, event_type, payload_json) '
        'VALUES (?, ?, ?, ?)',
        (session_id, ts, event_type, payload_json),
      )
      self._conn.commit()
      return EventRecord(
        id=int(cursor.lastrowid or 0),
        session_id=session_id,
        ts=ts,
        event_type=event_type,
        payload=payload or {},
      )

  def history(self, session_id: str) -> List[EventRecord]:
    '''List all events for a session, oldest first.

    Args:
      session_id: Session to query.

    Returns:
      Ordered event records.
    '''
    with self._lock:
      rows = self._conn.execute(
        'SELECT id, session_id, ts, event_type, payload_json '
        'FROM events WHERE session_id = ? ORDER BY id ASC',
        (session_id,),
      ).fetchall()

    return [
      EventRecord(
        id=row['id'],
        session_id=row['session_id'],
        ts=row['ts'],
        event_type=row['event_type'],
        payload=json.loads(row['payload_json'] or '{}'),
      )
      for row in rows
    ]

  def remember(
    self,
    session_id: str,
    key: str,
    value: Dict[str, Any],
  ) -> None:
    '''Upsert a memory entry for a session.

    Args:
      session_id: Owning session.
      key: Memory key.
      value: JSON-serializable payload.
    '''
    with self._lock:
      self._conn.execute(
        'INSERT INTO memory (session_id, key, value_json, updated_at) '
        'VALUES (?, ?, ?, ?) '
        'ON CONFLICT(session_id, key) DO UPDATE SET '
        'value_json=excluded.value_json, updated_at=excluded.updated_at',
        (session_id, key, json.dumps(value, default=str), _utc_now()),
      )
      self._conn.commit()

  def recall(self, session_id: str, key: str) -> Optional[Dict[str, Any]]:
    '''Fetch one memory entry.

    Args:
      session_id: Owning session.
      key: Memory key.

    Returns:
      Stored payload or None.
    '''
    with self._lock:
      row = self._conn.execute(
        'SELECT value_json FROM memory WHERE session_id = ? AND key = ?',
        (session_id, key),
      ).fetchone()

    if row is None:
      return None
    return json.loads(row['value_json'])

  def all_memory(self, session_id: str) -> Dict[str, Dict[str, Any]]:
    '''List all memory entries for a session.

    Args:
      session_id: Session to query.

    Returns:
      Mapping of key to payload.
    '''
    with self._lock:
      rows = self._conn.execute(
        'SELECT key, value_json FROM memory WHERE session_id = ? ORDER BY key',
        (session_id,),
      ).fetchall()

    return {row['key']: json.loads(row['value_json']) for row in rows}

  def log_tool_call(
    self,
    session_id: str,
    server: str,
    tool: str,
    args: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
    ok: bool = True,
    error: Optional[str] = None,
    latency_ms: float = 0.0,
    approved: bool = True,
  ) -> ToolCallRecord:
    '''Persist one tool-call audit row.

    Args:
      session_id: Owning session.
      server: MCP server name.
      tool: Tool name.
      args: Arguments used.
      result: (Truncated) tool result.
      ok: Whether the call succeeded.
      error: Error message when failed.
      latency_ms: Call duration.
      approved: Whether policy approved the call.

    Returns:
      The persisted record.
    '''
    ts = _utc_now()
    with self._lock:
      cursor = self._conn.execute(
        'INSERT INTO tool_calls (session_id, ts, server, tool, args_json, '
        'result_json, ok, error, latency_ms, approved) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (
          session_id,
          ts,
          server,
          tool,
          json.dumps(args or {}, default=str),
          json.dumps(result or {}, default=str),
          int(ok),
          error,
          float(latency_ms),
          int(approved),
        ),
      )
      self._conn.commit()

    return ToolCallRecord(
      id=int(cursor.lastrowid or 0),
      session_id=session_id,
      ts=ts,
      server=server,
      tool=tool,
      args=args or {},
      result=result or {},
      ok=ok,
      error=error,
      latency_ms=latency_ms,
      approved=approved,
    )

  def tool_calls(self, session_id: str) -> List[ToolCallRecord]:
    '''List tool-call audit rows for a session.

    Args:
      session_id: Session to query.

    Returns:
      Ordered tool call records.
    '''
    with self._lock:
      rows = self._conn.execute(
        'SELECT id, session_id, ts, server, tool, args_json, result_json, '
        'ok, error, latency_ms, approved '
        'FROM tool_calls WHERE session_id = ? ORDER BY id ASC',
        (session_id,),
      ).fetchall()

    return [
      ToolCallRecord(
        id=row['id'],
        session_id=row['session_id'],
        ts=row['ts'],
        server=row['server'],
        tool=row['tool'],
        args=json.loads(row['args_json'] or '{}'),
        result=json.loads(row['result_json'] or '{}'),
        ok=bool(row['ok']),
        error=row['error'],
        latency_ms=row['latency_ms'],
        approved=bool(row['approved']),
      )
      for row in rows
    ]
