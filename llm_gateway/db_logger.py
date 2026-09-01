#!/usr/bin/env python
# -- coding: utf-8 --

'''Background SQLite logger.'''

from __future__ import annotations

import queue
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from llm_gateway.schemas import LogRecord


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
    with sqlite3.connect(self._db_path) as conn:
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
    '''Drain the queue and stop the worker thread.'''
    if self._closed:
      return

    self._closed = True
    self._queue.join()
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
      finally:
        self._queue.task_done()

  def _insert(self, record: LogRecord) -> None:
    '''Persist one log record.

    Args:
      record: Log record to persist.
    '''
    with sqlite3.connect(self._db_path) as conn:
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
