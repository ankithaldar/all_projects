#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Profile, resume upload, and skills endpoints.'''


from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from job_hunter.api.deps import get_settings
from job_hunter.core.config import AppSettings
from job_hunter.db.repositories.profile import ProfileRepository
from job_hunter.db.repositories.settings import SettingsRepository
from pydantic import BaseModel, Field, field_validator

router = APIRouter(prefix='/api', tags=['profile'])


class ProfilePayload(BaseModel):
  '''Editable profile fields.'''

  target_roles: list = Field(default_factory=lambda: ['Data Scientist'])
  seniority_keywords: list = Field(default_factory=list)
  target_verticals: list = Field(default_factory=list)
  blocked_verticals: list = Field(default_factory=list)
  cities: list = Field(default_factory=list)
  relocate_ok: bool = False
  remote_pref: str = 'any'
  salary_floor_lpa: float | None = None
  experience_years: float | None = Field(default=None, ge=0, le=60)
  employment_types: list = Field(default_factory=lambda: ['full_time'])
  summary: str = ''

  @field_validator(
    'salary_floor_lpa', 'experience_years', mode='before'
  )
  @classmethod
  def _reject_bool(cls, value: Any) -> Any:
    '''Reject bool values before float coercion.

    Args:
      value: Raw numeric value.

    Returns:
      The value unchanged when it is not a bool.

    Raises:
      ValueError: When the value is a bool.
    '''
    if isinstance(value, bool):
      raise ValueError('invalid number')
    return value


@router.get('/profile')
def get_profile(
  settings: AppSettings = Depends(get_settings),
) -> Dict[str, Any]:
  '''Return the current candidate profile.

  Args:
    settings: App settings.

  Returns:
    Profile mapping (defaults when unset).
  '''
  repo = ProfileRepository(settings.db_path)
  data = repo.get_profile(1)
  if data is None:
    return {
      'target_roles': ['Data Scientist'], 'seniority_keywords': [],
      'target_verticals': [], 'blocked_verticals': [], 'cities': [],
      'relocate_ok': False, 'remote_pref': 'any',
      'salary_floor_lpa': SettingsRepository(settings.db_path).get(
        'salary_hard_floor_lpa',
      ), 'experience_years': 0,
      'employment_types': ['full_time'], 'summary': '', 'skills': [],
    }
  data.setdefault('skills', [])
  if data.get('salary_floor_lpa') is None:
    data['salary_floor_lpa'] = SettingsRepository(settings.db_path).get(
      'salary_hard_floor_lpa',
    )
  return data


@router.put('/profile')
def put_profile(
  payload: ProfilePayload,
  settings: AppSettings = Depends(get_settings),
) -> Dict[str, Any]:
  '''Persist a new profile version.

  Args:
    payload: Editable fields.
    settings: App settings.

  Returns:
    Saved profile mapping.
  '''
  repo = ProfileRepository(settings.db_path)
  fields = payload.model_dump(exclude_unset=True)
  if 'salary_floor_lpa' in fields:
    floor = fields['salary_floor_lpa']
    if floor is not None:
      if (
        isinstance(floor, bool)
        or not math.isfinite(float(floor))
        or not 0 <= float(floor) <= 1000
      ):
        raise HTTPException(
          status_code=422, detail='salary floor out of range',
        )
      # NOTE: two-session mirror-write is intentional: the settings
      # mirror (salary_hard_floor_lpa) is written in its own session
      # before the profile row is saved below, so a failure after
      # the mirror write leaves the settings mirror updated without
      # the profile row. Accepted LOW risk.
      repo_settings = SettingsRepository(settings.db_path)
      repo_settings.put('salary_hard_floor_lpa', float(floor))
    else:
      # Column is NOT NULL; explicit null means keep existing.
      del fields['salary_floor_lpa']
  existing = repo.get_profile(1) or {}
  merged = {**existing, **fields}
  for key in (
    'version', 'id', 'candidate_id', 'created_at',
    'confidence', 'resume_id'
  ):
    merged.pop(key, None)
  repo.save_profile(merged, candidate_id=1)
  return repo.get_profile(1)


@router.post('/profile/resume')
async def upload_resume(
  file: UploadFile,
  request: Request,
  settings: AppSettings = Depends(get_settings),
) -> Dict[str, Any]:
  '''Ingest a PDF resume through the Profile Curator agent.

  Args:
    file: Uploaded PDF.
    settings: App settings.

  Returns:
    Parse/extraction result summary.

  Raises:
    HTTPException: On non-PDF uploads or parse failure.
  '''
  if not (file.filename or '').lower().endswith('.pdf'):
    raise HTTPException(
      status_code=415, detail='only PDF resumes are supported'
    )
  content_length = request.headers.get('content-length') or ''
  try:
    declared = int(content_length) if content_length else 0
  except ValueError:
    declared = 0
  if declared > 10 * 1024 * 1024:
    raise HTTPException(status_code=422, detail='file too large')
  chunks: list[bytes] = []
  total = 0
  while chunk := await file.read(1_000_000):
    chunks.append(chunk)
    total += len(chunk)
    if total > 10 * 1024 * 1024:
      raise HTTPException(status_code=413, detail='file too large')
  data = b''.join(chunks)
  from job_hunter.llm.client import get_client
  from job_hunter.services.profile_curator import ProfileCurator
  try:
    curator = ProfileCurator(settings, client=get_client(settings))
    result = await curator.ingest_upload(
      file.filename or f'resume_{int(time.time())}.pdf', data
    )
  except Exception as exc:
    created = (
      getattr(exc, 'file_path', None) or getattr(exc, 'path', None)
    )
    if created:
      try:
        Path(created).unlink(missing_ok=True)
      except OSError:
        pass
    raise HTTPException(
      status_code=422, detail='resume curation unavailable',
    ) from exc
  return result


@router.get('/skills')
def list_skills(settings: AppSettings = Depends(get_settings)) -> list:
  '''Return the canonical skill taxonomy.

  Args:
    settings: App settings.

  Returns:
    Skill rows.
  '''
  from job_hunter.services.skills_taxonomy import SkillsTaxonomy
  return SkillsTaxonomy(settings.db_path).all_skills()


@router.put('/profile/skills')
def set_skills(
  names: List[str],
  settings: AppSettings = Depends(get_settings),
) -> Dict[str, Any]:
  '''Replace the candidate's manual skill list.

  Args:
    names: Raw skill names.
    settings: App settings.

  Returns:
    Resolved count.

  Raises:
    HTTPException: 422 when names is not a list of strings.
  '''
  taxonomy = SkillsTaxonomyForApp(settings)
  try:
    ids = taxonomy.resolve_many(names)
  except TypeError:
    raise HTTPException(
      status_code=422, detail='names must be a list of str',
    ) from None
  ProfileRepository(settings.db_path).set_candidate_skills(ids)
  return {'resolved': len(ids), 'requested': len(names)}


def SkillsTaxonomyForApp(settings: AppSettings):
  '''Build a taxonomy instance for the app database.

  Args:
    settings: App settings.

  Returns:
    SkillsTaxonomy.
  '''
  from job_hunter.services.skills_taxonomy import SkillsTaxonomy
  return SkillsTaxonomy(settings.db_path)
