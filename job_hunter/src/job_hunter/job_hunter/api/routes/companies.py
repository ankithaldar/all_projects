#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Company management endpoints.'''


from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional

import yaml
from fastapi import APIRouter, Body, Depends, HTTPException, Path
from job_hunter.adapters.base import valid_domain
from job_hunter.api.deps import get_settings
from job_hunter.core.config import AppSettings
from job_hunter.core.constants import MAX_SQLITE_ID, clamp_priority
from job_hunter.db.repositories.companies import CompaniesRepository
from pydantic import BaseModel, field_validator

router = APIRouter(prefix='/api/companies', tags=['companies'])


def _clamp_priority(priority: Any) -> int:
  '''Clamp a company priority into the supported 1..5 range.

  Args:
    priority: Raw priority value.

  Returns:
    Priority clamped to 1..5.

  Raises:
    ValueError: On bool, non-finite values (inf/nan) or non-numeric
      input.
  '''
  return clamp_priority(priority)


def _reject_bool_priority(value: Any) -> Any:
  '''Reject bool priority values before pydantic coerces them to int.

  Args:
    value: Raw priority value from the request body.

  Returns:
    The value unchanged when it is not a bool.

  Raises:
    ValueError: When the value is a bool.
  '''
  if isinstance(value, bool):
    raise ValueError('invalid priority')
  return value


def _validate_domain(domain: str) -> str:
  '''Validate a company domain.

  Args:
    domain: Candidate domain (empty means unknown).

  Returns:
    Normalized domain.

  Raises:
    HTTPException: 422 on invalid domain or IP-as-domain.
  '''
  try:
    return valid_domain(domain)
  except ValueError:
    raise HTTPException(status_code=422, detail='invalid domain') from None


class CompanyCreate(BaseModel):
  '''Manual company creation payload.'''

  name: str
  domain: str = ''
  vertical: str = ''
  priority: int = 3

  @field_validator('priority', mode='before')
  @classmethod
  def _priority_not_bool(cls, value: Any) -> Any:
    '''Reject bool priorities before int coercion.

    Args:
      value: Raw priority value.

    Returns:
      The value unchanged when it is not a bool.
    '''
    return _reject_bool_priority(value)


class CompanyPatch(BaseModel):
  '''Allow-listed company edits.'''

  domain: Optional[str] = None
  priority: Optional[int] = None
  vertical: Optional[str] = None
  status: Optional[str] = None
  notes: Optional[str] = None

  @field_validator('priority', mode='before')
  @classmethod
  def _priority_not_bool(cls, value: Any) -> Any:
    '''Reject bool priorities before int coercion.

    Args:
      value: Raw priority value.

    Returns:
      The value unchanged when it is not a bool.
    '''
    return _reject_bool_priority(value)


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
  domain = _validate_domain(payload.domain)
  try:
    priority = _clamp_priority(payload.priority)
  except (TypeError, ValueError, OverflowError):
    raise HTTPException(status_code=422, detail='invalid priority') from None
  company_id = CompaniesRepository(settings.db_path).upsert(
    name=payload.name,
    domain=domain,
    vertical=payload.vertical or None,
    priority=priority,
    discovered_via='manual',
  )
  return {'company_id': company_id}


@router.post('/import')
async def import_companies(
  content: str = Body(...),
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
  if len(content) > 1_000_000:
    raise HTTPException(status_code=422, detail='content too large')
  try:
    data = yaml.safe_load(content) or {}
  except yaml.YAMLError as exc:
    raise HTTPException(status_code=422, detail='invalid yaml') from exc
  entries = data.get('companies') if isinstance(data, dict) else None
  if not isinstance(entries, list):
    raise HTTPException(
      status_code=422, detail='expected top-level companies list'
    )
  repo = CompaniesRepository(settings.db_path)
  count = 0
  errors = 0
  for entry in entries:
    if not isinstance(entry, dict) or not entry.get('name'):
      continue
    domain = str(entry.get('domain') or '')
    try:
      domain = valid_domain(domain)
    except ValueError:
      errors += 1
      continue
    raw_priority = entry.get('priority', 3)
    if raw_priority is None:
      raw_priority = 3
    try:
      priority = _clamp_priority(raw_priority)
    except (TypeError, ValueError, OverflowError):
      errors += 1
      continue
    repo.upsert(
      name=str(entry['name']),
      domain=domain,
      vertical=entry.get('vertical_hint'),
      priority=priority,
      discovered_via='import',
    )
    count += 1
  return {'imported': count, 'errors': errors}


@router.patch('/{company_id}')
def patch_company(
  company_id: Annotated[int, Path(ge=-MAX_SQLITE_ID, le=MAX_SQLITE_ID)],
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
  fields = payload.model_dump(exclude_none=True)
  if 'domain' in fields:
    fields['domain'] = _validate_domain(fields['domain'])
  if fields.get('priority') is not None:
    try:
      fields['priority'] = _clamp_priority(fields['priority'])
    except (TypeError, ValueError, OverflowError):
      raise HTTPException(
        status_code=422, detail='invalid priority',
      ) from None
  # NOTE: two-session mirror-write is intentional here: the status
  # update (set_status) and the field patch (patch) run in separate
  # sessions, so a failure between them leaves the status mirror
  # updated without the field patch. Accepted LOW risk; documented
  # for parity with the put_profile settings mirror + profile row.
  repo = CompaniesRepository(settings.db_path)
  if repo.get(company_id) is None:
    raise HTTPException(status_code=404, detail='company not found')
  status = fields.pop('status', None)
  if status:
    repo.set_status(company_id, status, notes=fields.get('notes'))
  repo.patch(company_id, fields)
  return repo.get(company_id)


@router.delete('/{company_id}')
def blacklist_company(
  company_id: Annotated[int, Path(ge=-MAX_SQLITE_ID, le=MAX_SQLITE_ID)],
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
