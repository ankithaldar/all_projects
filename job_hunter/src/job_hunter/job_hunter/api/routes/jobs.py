#!/usr/bin/env python
# -- coding: utf-8 --

'''Job explorer endpoints.'''


from __future__ import annotations

from typing import Any, Dict, List

import json as _json
from fastapi import APIRouter, Depends
from job_hunter.api.deps import get_settings
from job_hunter.core.config import AppSettings
from job_hunter.db.repositories.jobs import JobsRepository

router = APIRouter(prefix='/api/jobs', tags=['jobs'])


@router.get('')
def list_jobs(
  q: str = '',
  city: str = '',
  work_mode: str = '',
  vertical: str = '',
  status: str = 'active',
  page: int = 1,
  per_page: int = 25,
  settings: AppSettings = Depends(get_settings),
) -> List[Dict[str, Any]]:
  '''Search and list postings.

  Args:
    q: FTS query.
    city: City filter.
    work_mode: Work-mode filter.
    vertical: Vertical filter.
    status: Status filter.
    page: Page number.
    per_page: Page size.
    settings: App settings.

  Returns:
    Job rows.
  '''
  repo = JobsRepository(settings.db_path)
  rows = repo.list_jobs(
    q=q, city=city, work_mode=work_mode, vertical=vertical,
    status=status or 'active', page=page, per_page=min(per_page, 100),
  )
  for row in rows:
    row.pop('raw_json', None)
    row['description_text'] = (row.get('description_text') or '')[:600]
  _ = _json
  return rows


@router.get('/{job_id}')
def get_job(
  job_id: int,
  settings: AppSettings = Depends(get_settings),
) -> Dict[str, Any]:
  '''Return one posting with its raw payload.

  Args:
    job_id: Job id.
    settings: App settings.

  Returns:
    Full row including raw_json decoded.
  '''
  row = JobsRepository(settings.db_path).get(job_id)
  if row is None:
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail='job not found')
  try:
    row['raw'] = _json.loads(row.get('raw_json') or '{}')
  except _json.JSONDecodeError:
    row['raw'] = {}
  row.pop('raw_json', None)
  return row
