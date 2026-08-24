#!/usr/bin/env python
# -- coding: utf-8 --

'''Checkpointing wiring for resumable graph runs.'''


from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional


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
  except ImportError:
    yield None
    return
  conn = await aiosqlite.connect(str(db_path))
  try:
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
  configurable: dict = {'thread_id': f'run-{run_id}'}
  if settings is not None:
    configurable['settings'] = settings
  return {'configurable': configurable}


