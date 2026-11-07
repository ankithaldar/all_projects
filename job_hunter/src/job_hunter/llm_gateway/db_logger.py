#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Background SQLite logger.'''

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from typing import Optional

from llm_gateway.schemas import LogRecord

logger = logging.getLogger(__name__)


def _connect(db_path: str) -> sqlite3.Connection:
  '''Open a WAL-mode connection with a busy timeout.

  Args:
    db_path: SQLite database path.

  Returns:
    Configured connection; caller must close it.
  '''
  conn = sqlite3.connect(db_path, timeout=10)
  try:
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA busy_timeout=5000')
    conn.execute('PRAGMA foreign_keys=ON')
  except BaseException:
    conn.close()
    raise
  return conn


class DBLogger:
  '''Logs gateway calls asynchronously using a worker thread.'''

  def __init__(self, db_path: str | Path) -> None:
    '''Initialize logger database and worker thread.

    Args:
      db_path: SQLite database path.
    '''
    self._db_path = str(db_path)
    self._queue: queue.Queue[Optional[LogRecord]] = queue.Queue()
    self._closed = False

    Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
    self._init_db()

    self._thread = threading.Thread(target=self._worker, daemon=True)
    self._thread.start()

  def _init_db(self) -> None:
    '''Create the logging table if needed.'''
    with closing(_connect(self._db_path)) as conn:
      conn.execute(
        '''
        CREATE TABLE IF NOT EXISTS llm_calls (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          timestamp TEXT NOT NULL,
          provider TEXT NOT NULL,
          model_used TEXT NOT NULL,
          input_tokens INTEGER NOT NULL,
          output_tokens INTEGER NOT NULL,
          latency_ms REAL NOT NULL,
          status TEXT NOT NULL,
          error TEXT,
          prompt_chars INTEGER NOT NULL,
          response_chars INTEGER NOT NULL,
          cost REAL NOT NULL,
          temperature REAL NOT NULL,
          system_prompt TEXT NOT NULL,
          session_id TEXT
        )
        '''
      )
      conn.execute(
        'CREATE INDEX IF NOT EXISTS idx_llm_timestamp '
        'ON llm_calls(timestamp)'
      )
      conn.execute(
        'CREATE INDEX IF NOT EXISTS idx_llm_provider_ts '
        'ON llm_calls(provider, timestamp)'
      )
      conn.execute(
        'CREATE INDEX IF NOT EXISTS idx_llm_session_ts '
        'ON llm_calls(session_id, timestamp)'
      )
      conn.commit()

  def log(self, record: LogRecord) -> None:
    '''Enqueue a log record without blocking the caller.

    Args:
      record: Log record to persist.
    '''
    if self._closed:
      return

    self._queue.put(record)

  def close(self) -> None:
    '''Drain the queue (bounded) and stop the worker thread.'''
    if self._closed:
      return

    self._closed = True
    joiner = threading.Thread(target=self._queue.join, daemon=True)
    joiner.start()
    joiner.join(timeout=5.0)
    self._queue.put(None)
    self._thread.join(timeout=2.0)

  def _worker(self) -> None:
    '''Background worker that persists queued records.'''
    while True:
      record = self._queue.get()

      if record is None:
        self._queue.task_done()
        break

      try:
        self._insert(record)
      except Exception:
        logger.exception('failed to persist log record')
      finally:
        self._queue.task_done()

  def _insert(self, record: LogRecord) -> None:
    '''Persist one log record.

    Args:
      record: Log record to persist.
    '''
    with closing(_connect(self._db_path)) as conn:
      conn.execute(
        '''
        INSERT INTO llm_calls (
          timestamp,
          provider,
          model_used,
          input_tokens,
          output_tokens,
          latency_ms,
          status,
          error,
          prompt_chars,
          response_chars,
          cost,
          temperature,
          system_prompt,
          session_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        (
          record.timestamp.isoformat(),
          record.provider,
          record.model_used,
          record.input_tokens,
          record.output_tokens,
          record.latency_ms,
          record.status,
          record.error,
          record.prompt_chars,
          record.response_chars,
          record.cost,
          record.temperature,
          record.system_prompt,
          record.session_id,
        ),
      )
      conn.commit()
