#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Persistent SQLite interaction log for bot runs (plan v11 section 8).

All statements are parameterised and the on-disk paths are confined to the
project log directory.  All values serialised to JSON use the same canonical
encoding (ASCII, sorted keys) so rows are reproducible.
"""

import json
import os
import sqlite3
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

from cat_game_bot.config import _resolve_path, _json_default
from cat_game_bot.errors import DbError
from cat_game_bot.models import ActionKind

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOG_ROOT = _PROJECT_ROOT / 'logs'

_SCHEMA = (
  'CREATE TABLE IF NOT EXISTS sessions ('
  'id INTEGER PRIMARY KEY AUTOINCREMENT, '
  'started_at REAL NOT NULL, finished_at REAL, '
  'config_json TEXT NOT NULL, result TEXT NOT NULL)',
  'CREATE TABLE IF NOT EXISTS events ('
  'id INTEGER PRIMARY KEY AUTOINCREMENT, '
  'session_id INTEGER NOT NULL REFERENCES sessions(id), '
  'ts REAL NOT NULL, event TEXT NOT NULL, state TEXT NOT NULL, '
  'detail_json TEXT NOT NULL)',
  'CREATE INDEX IF NOT EXISTS idx_events_session_ts '
  'ON events(session_id, ts)',
  'CREATE TABLE IF NOT EXISTS actions ('
  'id INTEGER PRIMARY KEY AUTOINCREMENT, '
  'session_id INTEGER NOT NULL REFERENCES sessions(id), '
  'ts REAL NOT NULL, kind TEXT NOT NULL, target TEXT, '
  'x INTEGER, y INTEGER, ok INTEGER NOT NULL, detail TEXT NOT NULL)',
  'CREATE INDEX IF NOT EXISTS idx_actions_session_ts '
  'ON actions(session_id, ts)',
  'CREATE TABLE IF NOT EXISTS observations ('
  'id INTEGER PRIMARY KEY AUTOINCREMENT, '
  'session_id INTEGER NOT NULL REFERENCES sessions(id), '
  'ts REAL NOT NULL, state TEXT NOT NULL, matches_json TEXT NOT NULL, '
  'counts_json TEXT NOT NULL, screen_hash TEXT NOT NULL)',
  'CREATE INDEX IF NOT EXISTS idx_observations_session_ts '
  'ON observations(session_id, ts)',
  'CREATE TABLE IF NOT EXISTS goal_progress ('
  'id INTEGER PRIMARY KEY AUTOINCREMENT, '
  'session_id INTEGER NOT NULL REFERENCES sessions(id), '
  'ts REAL NOT NULL, item TEXT NOT NULL, count INTEGER NOT NULL, '
  'required INTEGER NOT NULL, met INTEGER NOT NULL)',
  'CREATE INDEX IF NOT EXISTS idx_goal_session_ts '
  'ON goal_progress(session_id, ts)',
  'CREATE TABLE IF NOT EXISTS gifts ('
  'id INTEGER PRIMARY KEY AUTOINCREMENT, '
  'session_id INTEGER NOT NULL REFERENCES sessions(id), '
  'ts REAL NOT NULL, coins_before INTEGER NOT NULL, '
  'coins_after INTEGER NOT NULL, delta INTEGER NOT NULL)',
  'CREATE INDEX IF NOT EXISTS idx_gifts_session_ts '
  'ON gifts(session_id, ts)',
)


def _canonical(value):
  """Return canonical JSON for an arbitrary value.

  Args:
      value: Value to serialise.

  Returns:
      str: ASCII, key-sorted JSON text.
  """
  return json.dumps(value, ensure_ascii=True, sort_keys=True,
                    default=_json_default)


class InteractionLog:
  """SQLite-backed log with confined paths and canonical JSON payloads."""

  def __init__(self, db_path):
    """Open (or create) a database, applying the locked schema.

    The literal ``:memory:`` path is supported for tests and never touches
    the filesystem.  On-disk paths are confined to the project log root,
    created with mode 0700 and the database file forced to mode 0600.

    Args:
        db_path: Filesystem path, or ``':memory:'`` for an in-memory DB.

    Raises:
        DbError: On path, directory, open, chmod or schema failure.
    """
    self._closed = False
    memory = str(db_path) == ':memory:'
    if memory:
      target = ':memory:'
    else:
      resolved = _resolve_path(str(db_path), _LOG_ROOT)
      target = str(resolved)
      try:
        os.makedirs(resolved.parent, mode=0o700, exist_ok=True)
        os.chmod(resolved.parent, 0o700)
      except OSError as exc:
        raise DbError(f'cannot create log directory: {exc}') from exc
    try:
      self.connection = sqlite3.connect(target)
    except (sqlite3.Error, OSError) as exc:
      raise DbError(f'cannot open database: {exc}') from exc
    if not memory:
      try:
        os.chmod(target, 0o600)
      except OSError as exc:
        self.connection.close()
        raise DbError(f'cannot secure database file: {exc}') from exc
    try:
      for statement in _SCHEMA:
        self.connection.execute(statement)
      self.connection.commit()
    except sqlite3.Error as exc:
      self.connection.close()
      raise DbError(f'cannot initialise schema: {exc}') from exc

  def _write(self, sql, params):
    """Execute one parameterised write and commit it.

    Args:
        sql: Parameterised SQL statement.
        params: Sequence of bound values.

    Raises:
        DbError: If SQLite rejects the write.
    """
    try:
      self.connection.execute(sql, params)
      self.connection.commit()
    except sqlite3.Error as exc:
      raise DbError(f'database write failed: {exc}') from exc

  def open_session(self, config_json, initial_result='INTERRUPTED'):
    """Insert a session row with a placeholder result.

    Args:
        config_json: Serialised configuration text.
        initial_result: Placeholder terminal result.

    Returns:
        int: New session id.

    Raises:
        DbError: If the insert fails.
    """
    try:
      cursor = self.connection.execute(
          'INSERT INTO sessions(started_at, finished_at, config_json, '
          'result) VALUES (?, NULL, ?, ?)',
          (time.time(), config_json, initial_result))
      self.connection.commit()
    except sqlite3.Error as exc:
      raise DbError(f'cannot open session: {exc}') from exc
    return int(cursor.lastrowid)

  def log_event(self, session_id, ts, event, state, detail):
    """Append one event row.

    Args:
        session_id: Owning session id.
        ts: Event timestamp.
        event: Event name.
        state: Game state at the event.
        detail: Detail mapping to JSON-encode.
    """
    self._write(
        'INSERT INTO events(session_id, ts, event, state, detail_json) '
        'VALUES (?, ?, ?, ?, ?)',
        (session_id, ts, event, state, _canonical(detail)))

  def log_action(self, session_id, ts, action, result):
    """Append one action row, encoding swipe ends in ``target``.

    Args:
        session_id: Owning session id.
        ts: Action timestamp.
        action: Emitted action.
        result: Outcome of the action.
    """
    x = None
    y = None
    target = None
    if action.kind is ActionKind.SWIPE:
      if action.point is not None:
        x = action.point.x
        y = action.point.y
      if action.end is not None:
        target = f'{action.end.x},{action.end.y}'
    elif action.point is not None:
      x = action.point.x
      y = action.point.y
    self._write(
        'INSERT INTO actions(session_id, ts, kind, target, x, y, ok, '
        'detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        (session_id, ts, action.kind.value, target, x, y,
         int(result.ok), result.detail))

  def log_observation(self, session_id, ts, state, matches, counts,
                      screen_hash):
    """Append one observation row.

    Args:
        session_id: Owning session id.
        ts: Observation timestamp.
        state: Detected game state.
        matches: Detected interactive matches.
        counts: Current item counts.
        screen_hash: Short hash of the screen pixels.
    """
    payload = [asdict(match) if is_dataclass(match) else match
               for match in matches]
    self._write(
        'INSERT INTO observations(session_id, ts, state, matches_json, '
        'counts_json, screen_hash) VALUES (?, ?, ?, ?, ?, ?)',
        (session_id, ts, state, _canonical(payload),
         _canonical(counts), screen_hash))

  def log_goal_progress(self, session_id, ts, item, count, required, met):
    """Append one goal-progress row.

    Args:
        session_id: Owning session id.
        ts: Observation timestamp.
        item: Canonical item id.
        count: Observed count.
        required: Required count.
        met: Whether the target is met.
    """
    self._write(
        'INSERT INTO goal_progress(session_id, ts, item, count, required, '
        'met) VALUES (?, ?, ?, ?, ?, ?)',
        (session_id, ts, item, count, required, int(met)))

  def log_gift(self, session_id, ts, before, after):
    """Append one verified gift row.

    Args:
        session_id: Owning session id.
        ts: Verification timestamp.
        before: Coin count before collection.
        after: Coin count after collection.
    """
    self._write(
        'INSERT INTO gifts(session_id, ts, coins_before, coins_after, '
        'delta) VALUES (?, ?, ?, ?, ?)',
        (session_id, ts, before, after, after - before))

  def close_session(self, session_id, result):
    """Write the terminal result and finish time of a session.

    Args:
        session_id: Session to finalise.
        result: Terminal result string.
    """
    self._write('UPDATE sessions SET finished_at=?, result=? WHERE id=?',
                (time.time(), result, session_id))

  def close(self):
    """Close the connection idempotently; never raise."""
    if self._closed:
      return
    self._closed = True
    try:
      self.connection.close()
    except sqlite3.Error:
      pass
