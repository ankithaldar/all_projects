#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''SQLite connection helpers and the migration runner.'''

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from job_hunter.core.errors import DatabaseError

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / 'db' / 'migrations'


def connect(db_path: str | Path, readonly: bool = False) -> sqlite3.Connection:
  '''Open a WAL-mode SQLite connection.

  Args:
    db_path: Database file path.
    readonly: Whether to open read-only (creates parent dirs either way).

  Returns:
    Configured connection.

  Raises:
    sqlite3.Error: If the connection cannot be opened.
  '''
  path = Path(db_path)
  path.parent.mkdir(parents=True, exist_ok=True)
  conn = sqlite3.connect(
    f'file:{path}?mode=ro' if readonly else str(path),
    timeout=10.0,
    uri=readonly,
    isolation_level=None,
  )
  conn.row_factory = sqlite3.Row
  try:
    if not readonly:
      # WAL is persistent once set by a writer; forcing it on a
      # readonly connection would try to write and fail on
      # databases that are not already in WAL mode.
      for attempt in range(3):
        try:
          conn.execute('PRAGMA journal_mode=WAL')
          break
        except sqlite3.OperationalError:
          if attempt == 2:
            raise
          time.sleep(0.1)
    conn.execute('PRAGMA foreign_keys=ON')
    conn.execute('PRAGMA busy_timeout=5000')
  except Exception:
    conn.close()
    raise
  return conn


@contextmanager
def session(db_path: str | Path) -> Iterator[sqlite3.Connection]:
  '''Yield a connection with atomic commit/rollback.

  Do not nest session() blocks (each opens its own BEGIN IMMEDIATE
  connection; nesting deadlocks until busy_timeout).

  Args:
    db_path: Database file path.

  Yields:
    Active connection inside a ``BEGIN IMMEDIATE`` transaction.

  Raises:
    Exception: Re-raised after rolling back on any failure.
  '''
  conn = connect(db_path)
  began = False
  try:
    if not conn.in_transaction:
      conn.execute('BEGIN IMMEDIATE')
      began = True
    yield conn
    if began and conn.in_transaction:
      conn.execute('COMMIT')
  except Exception:
    if began and conn.in_transaction:
      conn.execute('ROLLBACK')
    elif conn.in_transaction:
      # The caller began a transaction; roll back rather than leak it.
      conn.execute('ROLLBACK')
    raise
  finally:
    conn.close()


@contextmanager
def fetch_all(
  db_path: str | Path,
  sql: str,
  params: Sequence[Any] = (),
) -> Iterator[List[sqlite3.Row]]:
  '''Fetch all rows on a readonly connection that always closes.

  Args:
    db_path: Database file path.
    sql: Parameterized SELECT statement.
    params: Bound parameter values.

  Yields:
    List of result rows.

  Raises:
    sqlite3.Error: If the query fails.
  '''
  conn = connect(db_path, readonly=True)
  try:
    yield conn.execute(sql, tuple(params)).fetchall()
  finally:
    conn.close()


@contextmanager
def fetch_one(
  db_path: str | Path,
  sql: str,
  params: Sequence[Any] = (),
) -> Iterator[Optional[sqlite3.Row]]:
  '''Fetch one row on a readonly connection that always closes.

  Args:
    db_path: Database file path.
    sql: Parameterized SELECT statement.
    params: Bound parameter values.

  Yields:
    Single result row or None.

  Raises:
    sqlite3.Error: If the query fails.
  '''
  conn = connect(db_path, readonly=True)
  try:
    yield conn.execute(sql, tuple(params)).fetchone()
  finally:
    conn.close()


def quote_identifier(name: str) -> str:
  '''Quote a trusted SQL identifier.

  Args:
    name: Column or table name already checked against the schema.

  Returns:
    Double-quoted identifier.
  '''
  return '"' + name.replace('"', '""') + '"'


def table_columns(
  conn: sqlite3.Connection,
  table: str,
) -> set[str]:
  '''Return the real column names of a table.

  Args:
    conn: Open connection.
    table: Trusted table name (never user input).

  Returns:
    Set of column names from ``PRAGMA table_info``.
  '''
  quoted = quote_identifier(table)
  rows = conn.execute(f'PRAGMA table_info({quoted})').fetchall()
  return {str(row[1]) for row in rows}


def filter_to_columns(
  conn: sqlite3.Connection,
  table: str,
  payload: Dict[str, Any],
) -> Dict[str, Any]:
  '''Keep only payload keys that are real columns of a table.

  Args:
    conn: Open connection.
    table: Trusted table name (never user input).
    payload: Candidate column mapping.

  Returns:
    Mapping containing only real columns; unknown keys are dropped
    with a warning and never interpolated into SQL.
  '''
  valid = table_columns(conn, table)
  kept: Dict[str, Any] = {}
  for key, value in payload.items():
    if key in valid:
      kept[key] = value
    else:
      logger.warning('dropping unknown column %r for table %s', key, table)
  return kept


def _split_statements(sql: str) -> List[str]:
  '''Split a script into complete statements.

  Args:
    sql: Script text possibly containing triggers, quoted
      semicolons, and comments.

  Returns:
    List of complete statement strings in script order.
  '''
  statements: List[str] = []
  buffer: List[str] = []
  in_single = in_double = in_line = in_block = False
  index = 0
  while index < len(sql):
    char = sql[index]
    nxt = sql[index + 1] if index + 1 < len(sql) else ''
    buffer.append(char)
    if in_line:
      if char == '\n':
        in_line = False
    elif in_block:
      if char == '*' and nxt == '/':
        buffer.append(nxt)
        index += 1
        in_block = False
    elif in_single:
      if char == "'" and nxt == "'":
        buffer.append(nxt)
        index += 1
      elif char == "'":
        in_single = False
    elif in_double:
      if char == '"' and nxt == '"':
        buffer.append(nxt)
        index += 1
      elif char == '"':
        in_double = False
    elif char == '-' and nxt == '-':
      buffer.append(nxt)
      index += 1
      in_line = True
    elif char == '/' and nxt == '*':
      buffer.append(nxt)
      index += 1
      in_block = True
    elif char == "'":
      in_single = True
    elif char == '"':
      in_double = True
    elif char == ';':
      joined = ''.join(buffer)
      if sqlite3.complete_statement(joined):
        if joined.strip():
          statements.append(joined)
        buffer = []
    index += 1
  tail = ''.join(buffer)
  if tail.strip():
    statements.append(tail)
  return statements


def _execute_statements(conn: sqlite3.Connection, sql: str) -> None:
  '''Execute a SQL script statement by statement.

  Args:
    conn: Connection inside an explicit transaction.
    sql: Script text possibly containing triggers.

  Raises:
    sqlite3.Error: If any statement fails.
  '''
  for statement in _split_statements(sql):
    conn.execute(statement)


def run_migrations(db_path: str | Path) -> None:
  '''Apply pending numbered SQL migrations exactly once.

  Args:
    db_path: Database file path.

  Raises:
    DatabaseError: If a migration fails mid-application.
  '''
  with session(db_path) as conn:
    conn.execute(
      'CREATE TABLE IF NOT EXISTS schema_migrations ('
      'name TEXT PRIMARY KEY, applied_at TEXT NOT NULL '
      "DEFAULT (datetime('now')))"
    )
  with fetch_all(db_path, 'SELECT name FROM schema_migrations') as rows:
    applied = {row['name'] for row in rows}
  for script in sorted(MIGRATIONS_DIR.glob('*.sql')):
    if script.name in applied:
      continue
    sql = script.read_text(encoding='utf-8')
    conn = connect(db_path)
    try:
      conn.execute('BEGIN IMMEDIATE')
      try:
        # Re-check inside the txn: a concurrent runner may have
        # applied this migration between the pre-check and BEGIN.
        already = conn.execute(
          'SELECT name FROM schema_migrations WHERE name = ?',
          (script.name,),
        ).fetchone()
        if already is None:
          _execute_statements(conn, sql)
          conn.execute(
            'INSERT INTO schema_migrations (name) VALUES (?)',
            (script.name,),
          )
      except sqlite3.Error as exc:
        if conn.in_transaction:
          conn.execute('ROLLBACK')
        raise DatabaseError(f'migration {script.name} failed: {exc}') from exc
      if conn.in_transaction:
        conn.execute('COMMIT')
    finally:
      conn.close()
