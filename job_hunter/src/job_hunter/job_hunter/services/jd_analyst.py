#!/usr/bin/env python
# -- coding: utf-8 --

'''JD Analyst: LLM extraction of structured fields from postings.'''


from __future__ import annotations

import time
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field
from job_hunter.core.models import EnrichedJob, WorkMode
from job_hunter.llm.client import GatewayClient
from job_hunter.llm.structured import complete_structured


class JDExtraction(BaseModel):
  '''Schema requested from the model for one posting.'''

  must_have_skills: List[str] = Field(default_factory=list)
  nice_to_have_skills: List[str] = Field(default_factory=list)
  experience_min_years: Optional[float] = None
  experience_max_years: Optional[float] = None
  salary_min_lpa: Optional[float] = None
  salary_max_lpa: Optional[float] = None
  work_mode: str = WorkMode.UNKNOWN
  employment_type: str = 'full_time'
  confidence: float = 0.6


INSTRUCTION = (
  'You are a meticulous job-description parser for the Indian market. '
  'Extract from the posting: concrete technical/domain skills split into '
  '"must_have_skills" (explicitly required) and "nice_to_have_skills" '
  '(preferred/bonus); experience bounds in years; any stated CTC/salary in '
  'INR lakhs per annum (numbers only, e.g. 45 means 45 LPA); work mode '
  '(remote|hybrid|onsite|unknown); and employment type '
  '(full_time|part_time|contract|internship). Be conservative: omit what is '
  'not stated. confidence reflects overall extraction certainty in [0,1].'
)


def apply_salary_floor(
  enriched: EnrichedJob,
  floor_lpa: float,
) -> Tuple[bool, str]:
  '''Evaluate the hard salary floor against extracted/normalized values.

  Args:
    enriched: Extraction result.
    floor_lpa: Minimum acceptable LPA.

  Returns:
    (passes_floor, reason) — reason explains a failure.
  '''
  if enriched.salary_max_lpa is not None and enriched.salary_max_lpa < floor_lpa:
    return False, f'salary_max {enriched.salary_max_lpa} below floor {floor_lpa}'
  return True, ''


class JDAnalyst:
  '''Extract structured job data through the gateway with budgeting.'''

  def __init__(self, client: GatewayClient) -> None:
    '''Initialize the analyst.

    Args:
      client: Gateway client.
    '''
    self._client = client

  async def extract(self, title: str, description_text: str, session_id: str) -> EnrichedJob:
    '''Run structured extraction over one posting.

    Args:
      title: Posting title.
      description_text: Cleaned description text.
      session_id: Correlation id (run-scoped).

    Returns:
      Validated enrichment.
    '''
    payload = f'TITLE: {title}\n\nDESCRIPTION:\n{description_text[:9000]}'
    extraction = await complete_structured(
      self._client,
      JDExtraction,
      INSTRUCTION,
      payload,
      session_id=session_id,
      temperature=0.0,
    )
    mode = extraction.work_mode.lower()
    if mode not in (WorkMode.REMOTE, WorkMode.HYBRID, WorkMode.ONSITE):
      mode = WorkMode.UNKNOWN
    return EnrichedJob(
      must_have_skills=extraction.must_have_skills,
      nice_to_have_skills=extraction.nice_to_have_skills,
      experience_min_years=extraction.experience_min_years,
      experience_max_years=extraction.experience_max_years,
      salary_min_lpa=extraction.salary_min_lpa,
      salary_max_lpa=extraction.salary_max_lpa,
      work_mode=mode,
      employment_type=extraction.employment_type or 'full_time',
      confidence=float(extraction.confidence),
    )


