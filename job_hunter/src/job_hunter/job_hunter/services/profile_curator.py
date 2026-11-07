#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Profile Curator agent: resume text to canonical profile via the gateway.'''


from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, Field
from job_hunter.core.config import AppSettings
from job_hunter.db.repositories.profile import ProfileRepository
from job_hunter.llm.client import GatewayClient, get_client
from job_hunter.llm.structured import complete_structured
from job_hunter.services.resume_parser import ResumeParser
from job_hunter.services.skills_taxonomy import SkillsTaxonomy

_Skill = Annotated[str, Field(max_length=100)]


class ProfileExtraction(BaseModel):
  '''Schema the curator asks the LLM to fill from a resume.'''

  skills: List[_Skill] = Field(
    default_factory=list, max_length=200
  )
  summary: str = Field(default='', max_length=20000)
  current_title: str = ''
  experience_years: float = Field(default=0.0, ge=0, le=60)
  seniority_keywords: List[_Skill] = Field(
    default_factory=list, max_length=200
  )


INSTRUCTION = (
  'You are a precise recruitment-data extractor. From the resume text, list '
  'the candidate\'s concrete technical and domain skills (canonical names like '
  '"Python", "PyTorch", "A/B testing"), a 3-4 sentence professional summary, '
  'their current or most recent title, total years of professional experience, '
  'and seniority keywords found (e.g. "staff", "senior", "principal", "lead").'
)


class ProfileCurator:
  '''Orchestrates resume ingestion into the candidate profile.'''

  def __init__(
    self,
    settings: AppSettings,
    client: Optional[GatewayClient] = None,
  ) -> None:
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
    file_path: Optional[Path] = None
    resume_id: Optional[int] = None
    resume_created: bool = False
    stored_file_path: Optional[str] = None
    try:
      target_dir = self._settings.data_dir / 'resumes'
      target_dir.mkdir(parents=True, exist_ok=True)
      # Full hex: the former uuid.uuid4().hex[:8] truncation risked
      # same-second collisions; the full 32-char hex does not.
      file_path = target_dir / (
        f'{int(time.time())}_{uuid.uuid4().hex}_{Path(file_name).name}'
      )
      file_path.write_bytes(data)
      parsed = self._parser.parse_file(file_path)
      profiles = ProfileRepository(self._settings.db_path)
      if not parsed['ok']:
        resume_id, resume_created = profiles.save_resume_ex(
          file_path=str(file_path),
          sha256=parsed['sha256'],
          mime='application/pdf',
          parsed_ok=bool(parsed['ok']),
          candidate_id=candidate_id,
        )
        if not resume_created:
          row = profiles.get_resume(resume_id)
          if row is None:
            raise RuntimeError('resume row missing after save_resume_ex')
          stored_file_path = str(row['file_path'])
          try:
            file_path.unlink(missing_ok=True)
          except OSError:
            pass
          return {
            'resume_id': resume_id,
            'parsed_ok': False,
            'reason': 'no text extracted',
            'file_path': stored_file_path,
            'path': stored_file_path,
          }
        stored_file_path = str(file_path)
        return {
          'resume_id': resume_id,
          'parsed_ok': False,
          'reason': 'no text extracted',
          'file_path': str(file_path),
          'path': str(file_path),
        }

      extraction = await self.extract(parsed['text'])
      skill_ids = self._persist_skills(extraction.skills, candidate_id)
      resume_id, resume_created = profiles.save_resume_ex(
        file_path=str(file_path),
        sha256=parsed['sha256'],
        mime='application/pdf',
        parsed_ok=bool(parsed['ok']),
        candidate_id=candidate_id,
      )
      if not resume_created:
        row = profiles.get_resume(resume_id)
        if row is None:
          raise RuntimeError('resume row missing after save_resume_ex')
        stored_file_path = str(row['file_path'])
        try:
          file_path.unlink(missing_ok=True)
        except OSError:
          pass
      else:
        stored_file_path = str(file_path)
      existing = profiles.get_profile(candidate_id) or {}
      default_exp = existing.get('experience_years', 0)
      seniority = extraction.seniority_keywords
      seniority = seniority or json_loads(existing.get('seniority_keywords'))
      title = extraction.current_title
      roles = [title] if title else json_loads(existing.get('target_roles'))
      version_fields: Dict[str, Any] = {
        'skills_resolved': len(skill_ids),
        'summary': extraction.summary or existing.get('summary', ''),
        'experience_years': extraction.experience_years or default_exp,
        'seniority_keywords': seniority,
        'target_roles': roles,
        'resume_id': resume_id,
        'confidence': 0.8,
      }
      profiles.save_profile(version_fields, candidate_id=candidate_id)
      return_path = (
        stored_file_path if not resume_created else str(file_path)
      )
      return {
        'resume_id': resume_id,
        'parsed_ok': True,
        'pages': parsed['pages'],
        'skills': extraction.skills,
        'skill_ids': skill_ids,
        'summary': extraction.summary,
        'current_title': extraction.current_title,
        'experience_years': extraction.experience_years,
        'file_path': return_path,
        'path': return_path,
      }
    except BaseException as exc:
      # Atomic cleanup on ANY failure (including CancelledError /
      # KeyboardInterrupt, which are BaseExceptions): remove our
      # own file and any resume row written before the failure.
      # Only rows created by this invocation may be deleted, and
      # only when the stored path matches this invocation's file.
      owns_row = (
        resume_id is not None
        and resume_created
        and stored_file_path == str(file_path)
      )
      file_retained = False
      if owns_row:
        # Delete our row first: if a duplicate has adopted it and
        # saved a profile referencing it, the delete is FK-blocked
        # (IntegrityError) and the stored file must be kept. Only
        # unlink when the row is actually gone.
        try:
          ProfileRepository(self._settings.db_path).delete_resume(
            resume_id,  # type: ignore[arg-type]
          )
        except Exception:  # pylint: disable=broad-exception-caught
          file_retained = True
        else:
          if file_path is not None:
            try:
              file_path.unlink(missing_ok=True)
            except OSError:
              pass
      elif file_path is not None:
        # Not our stored row (e.g. duplicate temp file, or no row
        # written yet): unlinking our own file cannot remove the
        # stored file, whose path differs.
        try:
          file_path.unlink(missing_ok=True)
        except OSError:
          pass
      try:
        # Only hand the route a path to unlink when deletion
        # succeeded (or no row existed); when FK-blocked the stored
        # file is retained, so file_path must be falsy.
        if file_retained:
          exc.file_path = None
        elif file_path is not None:
          exc.file_path = str(file_path)
        else:
          exc.file_path = None
      except Exception:  # pylint: disable=broad-exception-caught
        pass
      raise

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
      session_id=f'profile:{uuid.uuid4().hex}',
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
    repo = ProfileRepository(self._settings.db_path)
    repo.set_candidate_skills(resolved_ids, candidate_id)
    return resolved_ids


def json_loads(value: Any) -> list:
  '''Decode a JSON column that may arrive as str, list, or None.

  Args:
    value: Raw stored value.

  Returns:
    List value.
  '''
  if isinstance(value, list):
    return value
  if isinstance(value, str) and value.strip():
    try:
      decoded = json.loads(value)
      return decoded if isinstance(decoded, list) else []
    except json.JSONDecodeError:
      return []
  return []
