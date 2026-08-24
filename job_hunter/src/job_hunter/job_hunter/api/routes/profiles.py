#!/usr/bin/env python
# -- coding: utf-8 --

'''Profile, resume upload, and skills endpoints.'''


from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from job_hunter.api.deps import get_settings
from job_hunter.core.config import AppSettings
from job_hunter.db.repositories.profile import ProfileRepository
from pydantic import BaseModel, Field

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
  experience_years: float | None = None
  employment_types: list = Field(default_factory=lambda: ['full_time'])
  summary: str = ''


@router.get('/profile')
def get_profile(settings: AppSettings = Depends(get_settings)) -> Dict[str, Any]:
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
      'salary_floor_lpa': None, 'experience_years': 0,
      'employment_types': ['full_time'], 'summary': '', 'skills': [],
    }
  data.setdefault('skills', [])
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
  fields = payload.model_dump()
  floor = fields.pop('salary_floor_lpa', None)
  if floor is not None:
    from job_hunter.db.repositories.settings import SettingsRepository
    SettingsRepository(settings.db_path).put('salary_hard_floor_lpa', float(floor))
  existing = repo.get_profile(1) or {}
  merged = {**existing, **fields}
  for key in ('version', 'id', 'candidate_id', 'created_at', 'confidence', 'resume_id'):
    merged.pop(key, None)
  repo.save_profile(merged, candidate_id=1)
  return repo.get_profile(1)


@router.post('/profile/resume')
async def upload_resume(
  file: UploadFile,
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
    raise HTTPException(status_code=415, detail='only PDF resumes are supported')
  data = await file.read()
  if len(data) > 10 * 1024 * 1024:
    raise HTTPException(status_code=413, detail='file too large')
  from job_hunter.llm.client import get_client
  from job_hunter.services.profile_curator import ProfileCurator
  curator = ProfileCurator(settings, client=get_client(settings))
  try:
    result = await curator.ingest_upload(file.filename or f'resume_{int(time.time())}.pdf', data)
  except Exception as exc:
    raise HTTPException(status_code=502, detail=f'curation failed: {exc}') from exc
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
  names: list,
  settings: AppSettings = Depends(get_settings),
) -> Dict[str, Any]:
  '''Replace the candidate's manual skill list.

  Args:
    names: Raw skill names.
    settings: App settings.

  Returns:
    Resolved count.
  '''
  taxonomy = SkillsTaxonomyForApp(settings)
  ids = taxonomy.resolve_many(names)
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
