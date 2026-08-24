#!/usr/bin/env python
# -- coding: utf-8 --

'''Company management endpoints.'''


from __future__ import annotations

from typing import Any, Dict, List, Optional

import yaml
from fastapi import APIRouter, Depends, HTTPException
from job_hunter.api.deps import get_settings
from job_hunter.core.config import AppSettings
from job_hunter.db.repositories.companies import CompaniesRepository
from pydantic import BaseModel

router = APIRouter(prefix='/api/companies', tags=['companies'])


class CompanyCreate(BaseModel):
  '''Manual company creation payload.'''

  name: str
  domain: str = ''
  vertical: str = ''
  priority: int = 3


class CompanyPatch(BaseModel):
  '''Allow-listed company edits.'''

  priority: Optional[int] = None
  vertical: Optional[str] = None
  status: Optional[str] = None
  notes: Optional[str] = None


@router.get('')
def list_companies(
  status: str = '',
  vertical: str = '',
  settings: AppSettings = Depends(get_settings),
) -> List[Dict[str, Any]]:
  '''List companies with filters.

  Args:
    status: Status filter.
    vertical: Vertical filter.
    settings: App settings.

  Returns:
    Company rows.
  '''
  return CompaniesRepository(settings.db_path).list_companies(
    status=status or None, vertical=vertical or None,
  )


@router.post('')
def create_company(
  payload: CompanyCreate,
  settings: AppSettings = Depends(get_settings),
) -> Dict[str, Any]:
  '''Create or update one company manually.

  Args:
    payload: Company fields.
    settings: App settings.

  Returns:
    Company id.
  '''
  company_id = CompaniesRepository(settings.db_path).upsert(
    name=payload.name,
    domain=payload.domain,
    vertical=payload.vertical or None,
    priority=payload.priority,
    discovered_via='manual',
  )
  return {'company_id': company_id}


@router.post('/import')
async def import_companies(
  content: str,
  settings: AppSettings = Depends(get_settings),
) -> Dict[str, Any]:
  '''Import seed-style YAML with a companies list.

  Args:
    content: Raw YAML text.
    settings: App settings.

  Returns:
    Ingestion counters.

  Raises:
    HTTPException: On invalid YAML.
  '''
  try:
    data = yaml.safe_load(content) or {}
  except yaml.YAMLError as exc:
    raise HTTPException(status_code=422, detail=f'invalid yaml: {exc}') from exc
  entries = data.get('companies') if isinstance(data, dict) else None
  if not isinstance(entries, list):
    raise HTTPException(status_code=422, detail='expected top-level companies list')
  repo = CompaniesRepository(settings.db_path)
  count = 0
  for entry in entries:
    if not isinstance(entry, dict) or not entry.get('name'):
      continue
    repo.upsert(
      name=str(entry['name']),
      domain=str(entry.get('domain') or ''),
      vertical=entry.get('vertical_hint'),
      priority=int(entry.get('priority') or 3),
      discovered_via='import',
    )
    count += 1
  return {'imported': count}


@router.patch('/{company_id}')
def patch_company(
  company_id: int,
  payload: CompanyPatch,
  settings: AppSettings = Depends(get_settings),
) -> Dict[str, Any]:
  '''Apply allow-listed edits to one company.

  Args:
    company_id: Id.
    payload: Edits.
    settings: App settings.

  Returns:
    Updated row.

  Raises:
    HTTPException: When the company is missing.
  '''
  repo = CompaniesRepository(settings.db_path)
  if repo.get(company_id) is None:
    raise HTTPException(status_code=404, detail='company not found')
  fields = payload.model_dump(exclude_none=True)
  status = fields.pop('status', None)
  if status:
    repo.set_status(company_id, status, notes=fields.get('notes'))
  repo.patch(company_id, fields)
  return repo.get(company_id)


@router.delete('/{company_id}')
def blacklist_company(
  company_id: int,
  settings: AppSettings = Depends(get_settings),
) -> Dict[str, str]:
  '''Blacklist a company so its jobs are skipped.

  Args:
    company_id: Id.
    settings: App settings.

  Returns:
    Acknowledgement.
  '''
  CompaniesRepository(settings.db_path).set_status(
    company_id, 'blacklisted', notes='blacklisted via ui',
  )
  return {'status': 'blacklisted'}
