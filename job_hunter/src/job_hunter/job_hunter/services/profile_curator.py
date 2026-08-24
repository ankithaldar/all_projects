#!/usr/bin/env python
# -- coding: utf-8 --

'''Profile Curator agent: resume text to canonical profile via the gateway.'''


from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from job_hunter.core.config import AppSettings
from job_hunter.db.repositories.profile import ProfileRepository
from job_hunter.llm.client import GatewayClient, get_client
from job_hunter.llm.structured import complete_structured
from job_hunter.services.resume_parser import ResumeParser
from job_hunter.services.skills_taxonomy import SkillsTaxonomy


class ProfileExtraction(BaseModel):
  '''Schema the curator asks the LLM to fill from a resume.'''

  skills: List[str] = Field(default_factory=list)
  summary: str = ''
  current_title: str = ''
  experience_years: float = 0.0
  seniority_keywords: List[str] = Field(default_factory=list)


INSTRUCTION = (
  'You are a precise recruitment-data extractor. From the resume text, list '
  'the candidate\'s concrete technical and domain skills (canonical names like '
  '"Python", "PyTorch", "A/B testing"), a 3-4 sentence professional summary, '
  'their current or most recent title, total years of professional experience, '
  'and seniority keywords found (e.g. "staff", "senior", "principal", "lead").'
)


class ProfileCurator:
  '''Orchestrates resume ingestion into the candidate profile.'''

  def __init__(self, settings: AppSettings, client: Optional[GatewayClient] = None) -> None:
    '''Initialize with settings and an optional gateway client override.

    Args:
      settings: Application settings.
      client: Gateway client override (tests).
    '''
    self._settings = settings
    self._client = client or get_client(settings)
    self._parser = ResumeParser()

  async def ingest_upload(
    self,
    file_name: str,
    data: bytes,
    candidate_id: int = 1,
  ) -> Dict[str, Any]:
    '''Store, parse, extract, and persist an uploaded resume.

    Args:
      file_name: Original upload filename.
      data: File bytes.
      candidate_id: Candidate id.

    Returns:
      Mapping with resume_id, profile fields, parse metadata.
    '''
    target_dir = self._settings.data_dir / 'resumes'
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = f'{int(time.time())}'
    safe_name = Path(file_name).name.replace(' ', '_') or f'resume_{stamp}.pdf'
    file_path = target_dir / f'{stamp}_{safe_name}'
    file_path.write_bytes(data)

    parsed = self._parser.parse_file(file_path)
    profiles = ProfileRepository(self._settings.db_path)
    resume_id = profiles.save_resume(
      file_path=str(file_path),
      sha256=parsed['sha256'],
      mime='application/pdf',
      parsed_ok=bool(parsed['ok']),
      candidate_id=candidate_id,
    )
    if not parsed['ok']:
      return {'resume_id': resume_id, 'parsed_ok': False, 'reason': 'no text extracted'}

    extraction = await self.extract(parsed['text'])
    skill_ids = self._persist_skills(extraction.skills, candidate_id)
    existing = profiles.get_profile(candidate_id) or {}
    version_fields: Dict[str, Any] = {
      'skills_resolved': len(skill_ids),
      'summary': extraction.summary or existing.get('summary', ''),
      'experience_years': extraction.experience_years or existing.get('experience_years', 0),
      'seniority_keywords': extraction.seniority_keywords or json_loads(existing.get('seniority_keywords')),
      'target_roles': [extraction.current_title] if extraction.current_title else json_loads(existing.get('target_roles')),
      'resume_id': resume_id,
      'confidence': 0.8,
    }
    profiles.save_profile(version_fields, candidate_id=candidate_id)
    return {
      'resume_id': resume_id,
      'parsed_ok': True,
      'pages': parsed['pages'],
      'skills': extraction.skills,
      'skill_ids': skill_ids,
      'summary': extraction.summary,
      'current_title': extraction.current_title,
      'experience_years': extraction.experience_years,
    }

  async def extract(self, resume_text: str) -> ProfileExtraction:
    '''Run structured extraction over cleaned resume text.

    Args:
      resume_text: Cleaned plain text.

    Returns:
      Validated extraction.
    '''
    return await complete_structured(
      self._client,
      ProfileExtraction,
      INSTRUCTION,
      resume_text,
      session_id=f'profile:{int(time.time())}',
    )

  def _persist_skills(self, raw_skills: list, candidate_id: int) -> list:
    '''Resolve and link extracted skills for the candidate.

    Args:
      raw_skills: Raw LLM skill names.
      candidate_id: Candidate id.

    Returns:
      Resolved canonical ids.
    '''
    taxonomy = SkillsTaxonomy(self._settings.db_path)
    resolved_ids = taxonomy.resolve_many(raw_skills)
    if not resolved_ids:
      taxonomy.load_seed({'skills': {name: None for name in raw_skills}})
      resolved_ids = taxonomy.resolve_many(raw_skills)
    ProfileRepository(self._settings.db_path).set_candidate_skills(resolved_ids, candidate_id)
    return resolved_ids


def json_loads(value: Any) -> list:
  '''Decode a JSON column that may arrive as str, list, or None.

  Args:
    value: Raw stored value.

  Returns:
    List value.
  '''
  import json
  if isinstance(value, list):
    return value
  if isinstance(value, str) and value.strip():
    try:
      decoded = json.loads(value)
      return decoded if isinstance(decoded, list) else []
    except json.JSONDecodeError:
      return []
  return []
