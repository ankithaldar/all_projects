#!/usr/bin/env python
# -- coding: utf-8 --

'''SQLite connection helpers and the migration runner.'''


from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from job_hunter.core.errors import DatabaseError

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / 'db' / 'migrations'


def connect(db_path: str | Path, readonly: bool = False) -> sqlite3.Connection:
  '''Open a WAL-mode SQLite connection.

  Args:
    db_path: Database file path.
    readonly: Whether to open read-only (creates parent dirs either way).

  Returns:
    Configured connection.
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
  conn.execute('PRAGMA journal_mode=WAL')
  conn.execute('PRAGMA foreign_keys=ON')
  conn.execute('PRAGMA busy_timeout=5000')
  return conn


@contextmanager
def session(db_path: str | Path) -> Iterator[sqlite3.Connection]:
  '''Yield a connection with automatic commit/rollback.

  Args:
    db_path: Database file path.

  Yields:
    Active connection.
  '''
  conn = connect(db_path)
  try:
    yield conn
    if conn.in_transaction:
      conn.execute('COMMIT')
  except Exception:
    if conn.in_transaction:
      conn.execute('ROLLBACK')
    raise
  finally:
    conn.close()


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
      'name TEXT PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT (datetime(\'now\')))'
    )
  applied = {
    row['name']
    for row in connect(db_path, readonly=True).execute('SELECT name FROM schema_migrations')
  }
  for script in sorted(MIGRATIONS_DIR.glob('*.sql')):
    if script.name in applied:
      continue
    sql = script.read_text(encoding='utf-8')
    conn = connect(db_path)
    try:
      conn.executescript(sql)
      conn.execute('INSERT INTO schema_migrations (name) VALUES (?)', (script.name,))
    except sqlite3.Error as exc:
      raise DatabaseError(f'migration {script.name} failed: {exc}') from exc
    finally:
      conn.close()
