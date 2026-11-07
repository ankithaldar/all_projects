#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Checkpointing wiring for resumable graph runs.'''

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


@asynccontextmanager
async def open_checkpointer(db_path: str | Path):
  '''Yield an AsyncSqliteSaver when the dependency is available.

  Args:
    db_path: Checkpoint database path.

  Yields:
    Checkpointer instance or None when langgraph sqlite extras are missing.
  '''
  try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    import aiosqlite
  except ImportError as exc:
    logger.warning(
      'checkpoint dependency missing, running without checkpointer: %s', exc
    )
    yield None
    return
  path = Path(db_path)
  path.parent.mkdir(parents=True, exist_ok=True)
  conn = await aiosqlite.connect(str(path))
  try:
    await conn.execute('PRAGMA journal_mode=WAL')
    await conn.execute('PRAGMA busy_timeout=5000')
    await conn.execute('PRAGMA foreign_keys=ON')
    saver = AsyncSqliteSaver(conn)
    await saver.setup()
    yield saver
  finally:
    await conn.close()


def thread_config(run_id: int, settings=None) -> dict:
  '''Build runnable config carrying settings and checkpoint thread id.

  Args:
    run_id: Run row id.
    settings: Application settings to inject into nodes.

  Returns:
    LangGraph config dict.
  '''
  configurable: dict = {
    'thread_id': f'run-{run_id}',
    'checkpoint_ns': '',
  }
  if settings is not None:
    configurable['settings'] = settings
  return {'configurable': configurable}
