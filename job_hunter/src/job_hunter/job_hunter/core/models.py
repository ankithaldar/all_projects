#!/usr/bin/env python
# -- coding: utf-8 --

'''Domain models shared across services, graph, and API.'''


from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def utcnow() -> datetime:
  '''Return current UTC time.

  Returns:
    Timezone-aware datetime.
  '''
  return datetime.now(timezone.utc)


class WorkMode:
  '''Work-mode enum values.'''

  REMOTE = 'remote'
  HYBRID = 'hybrid'
  ONSITE = 'onsite'
  UNKNOWN = 'unknown'


class CandidateProfile(BaseModel):
  '''Canonical candidate profile used by matching.'''

  candidate_id: int = 1
  target_roles: List[str] = Field(default_factory=lambda: ['Data Scientist'])
  seniority_keywords: List[str] = Field(
    default_factory=lambda: ['staff', 'senior', 'principal', 'lead'],
  )
  target_verticals: List[str] = Field(default_factory=list)
  blocked_verticals: List[str] = Field(default_factory=list)
  cities: List[str] = Field(default_factory=list)
  relocate_ok: bool = False
  remote_pref: str = 'any'
  salary_floor_lpa: float = 45.0
  experience_years: float = 0.0
  employment_types: List[str] = Field(default_factory=lambda: ['full_time'])
  skills: List[str] = Field(default_factory=list)
  summary: str = ''
  version: int = 1


class CompanyTarget(BaseModel):
  '''A company plus the source through which its jobs are fetched.'''

  company_id: Optional[int] = None
  name: str
  domain: str = ''
  source_key: str
  board_ref: str = ''


class RawJobRecord(BaseModel):
  '''Posting exactly as produced by one adapter, pre-normalization.'''

  source_key: str
  external_id: str = ''
  url: str
  company_name: str = ''
  title: str
  location_text: str = ''
  description_html: str = ''
  description_text: str = ''
  posted_at: Optional[str] = None
  employment_type_raw: str = ''
  work_mode_hint: str = ''
  salary_raw: str = ''
  extra: Dict[str, Any] = Field(default_factory=dict)


class NormalizedJob(BaseModel):
  '''Canonical job row ready for persistence.'''

  raw: RawJobRecord
  canonical_url: str
  content_hash: str
  company_id: Optional[int] = None
  city: str = ''
  region: str = ''
  country: str = 'IN'
  work_mode: str = WorkMode.UNKNOWN
  employment_type: str = 'full_time'
  salary_min_lpa: Optional[float] = None
  salary_max_lpa: Optional[float] = None
  is_new: bool = True


class EnrichedJob(BaseModel):
  '''LLM-extracted structured fields for one posting.'''

  must_have_skills: List[str] = Field(default_factory=list)
  nice_to_have_skills: List[str] = Field(default_factory=list)
  experience_min_years: Optional[float] = None
  experience_max_years: Optional[float] = None
  salary_min_lpa: Optional[float] = None
  salary_max_lpa: Optional[float] = None
  work_mode: str = WorkMode.UNKNOWN
  employment_type: str = 'full_time'
  confidence: float = 0.6
  needs_review: bool = False


class ScoredJob(BaseModel):
  '''A normalized job with match components and a final score.'''

  job_id: int
  title: str
  url: str
  company_id: Optional[int] = None
  company_name: str = ''
  vertical: str = 'unknown'
  total_score: float = 0.0
  gate_pass: bool = True
  gate_failures: List[str] = Field(default_factory=list)
  breakdown: Dict[str, float] = Field(default_factory=dict)
  rationale: str = ''
  semantic_score: float = 0.0
  posted_at: Optional[str] = None


class RunPlan(BaseModel):
  '''The set of fetch tasks selected for one run.'''

  run_id: int
  kind: str = 'discovery'
  targets: List[CompanyTarget] = Field(default_factory=list)
  scan_inbox: bool = False
  quick_poll: bool = False


class NodeError(BaseModel):
  '''One recoverable failure captured inside a graph node.'''

  node: str
  message: str
  recoverable: bool = True


class TokenBudget(BaseModel):
  '''Per-run LLM call budget with exhaustion tracking.'''

  max_calls: int = 300
  used: int = 0

  def allow(self) -> bool:
    '''Return whether another LLM call fits the budget.

    Returns:
      True when under budget.
    '''
    return self.used < self.max_calls

  def spend(self) -> None:
    '''Consume one call from the budget.'''
    self.used += 1
