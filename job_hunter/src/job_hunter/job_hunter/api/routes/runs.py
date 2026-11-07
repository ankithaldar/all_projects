#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Run management endpoints including the SSE progress stream.'''


from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from job_hunter.api.deps import get_settings
from job_hunter.core.config import AppSettings
from job_hunter.db.repositories.runs import RunsRepository

router = APIRouter(prefix='/api/runs', tags=['runs'])


@router.get('')
def list_runs(
  limit: int = Query(50, ge=1, le=100),
  settings: AppSettings = Depends(get_settings),
) -> List[Dict[str, Any]]:
  '''List recent runs.

  Args:
    limit: Page size.
    settings: App settings.

  Returns:
    Run rows.
  '''
  return RunsRepository(settings.db_path).list_runs(limit=limit)


@router.post('/discovery')
def trigger_discovery(
  settings: AppSettings = Depends(get_settings),
) -> Dict[str, Any]:
  '''Enqueue a discovery run for immediate execution by a worker.

  Args:
    settings: App settings.

  Returns:
    New run id (status pending).

  Raises:
    HTTPException: 409 when a run is already active.
  '''
  repo = RunsRepository(settings.db_path)
  run_id = repo.create_if_not_active('discovery', 'ui')
  if run_id is None:
    raise HTTPException(status_code=409, detail='a run is already active')
  return {'run_id': run_id}


@router.get('/{run_id}/events')
def run_events(
  run_id: int,
  after_id: int = Query(0, ge=0),
  limit: int = Query(200, ge=1, le=500),
  settings: AppSettings = Depends(get_settings),
) -> List[Dict[str, Any]]:
  '''Return events after a cursor id.

  Args:
    run_id: Run id.
    after_id: Exclusive cursor.
    limit: Max rows.
    settings: App settings.

  Returns:
    Event rows.

  Raises:
    HTTPException: When the run does not exist.
  '''
  repo = RunsRepository(settings.db_path)
  if repo.get(run_id) is None:
    raise HTTPException(status_code=404, detail='run not found')
  return repo.events_since(run_id, after_id, limit=limit)


@router.post('/{run_id}/cancel')
def cancel_run(
  run_id: int,
  settings: AppSettings = Depends(get_settings),
) -> Dict[str, Any]:
  '''Mark a pending or running run failed (cooperative cancel).

  Args:
    run_id: Run id.
    settings: App settings.

  Returns:
    Final status.
  '''
  repo = RunsRepository(settings.db_path)
  row = repo.get(run_id)
  if row is None:
    raise HTTPException(status_code=404, detail='run not found')
  if row['status'] in ('pending', 'running'):
    repo.finish(run_id, 'failed', {}, error_text='cancelled via ui')
  row = repo.get(run_id)
  return {'run_id': run_id, 'status': row['status']}


@router.get('/{run_id}/stream')
async def stream_run(
  run_id: int,
  settings: AppSettings = Depends(get_settings),
) -> StreamingResponse:
  '''Server-Sent Events stream of run events until completion.

  Args:
    run_id: Run id.
    settings: App settings.

  Returns:
    SSE response.
  '''
  repo = RunsRepository(settings.db_path)
  if repo.get(run_id) is None:
    raise HTTPException(status_code=404, detail='run not found')

  async def generator() -> AsyncIterator[str]:
    '''Poll events and forward them as SSE frames.

    Yields:
        SSE text frames.
    '''
    last_id = 0
    idle_polls = 0
    while True:
      events = repo.events_since(run_id, last_id)
      for event in events:
        last_id = int(event['id'])
        payload = json.dumps({
          'id': event['id'],
          'level': event['level'],
          'node': event['node'],
          'message': event['message'],
        }, ensure_ascii=False)
        yield f'event: log\ndata: {payload}\n\n'
      row = repo.get(run_id)
      if row and row['finished_at'] and not events:
        raw_stats = row['stats_json']
        stats = (
          json.loads(row['stats_json'])
          if isinstance(raw_stats, str) and raw_stats
          else {}
        )
        yield (
          'event: done\ndata: '
          + json.dumps({'status': row['status'], 'stats': stats})
          + '\n\n'
        )
        return
      idle_polls = idle_polls + 1 if not events else 0
      if idle_polls > 600:
        yield 'event: timeout\ndata: {}\n\n'
        return
      await asyncio.sleep(0.7)

  return StreamingResponse(generator(), media_type='text/event-stream')
