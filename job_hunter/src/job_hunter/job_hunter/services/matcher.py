#!/usr/bin/env python
# -- coding: utf-8 --

'''Match component computation and hard gates.'''


from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from job_hunter.core.models import CandidateProfile, WorkMode

_TOKEN_RE = re.compile(r'[a-z+#][a-z0-9+.#-]{2,}')


def tokens(text: str) -> set:
  '''Lowercase word-ish tokens of length >= 3.

  Args:
    text: Any text.

  Returns:
    Token set.
  '''
  return set(_TOKEN_RE.findall((text or '').lower()))


def _overlap_ratio(a: set, b: set) -> float:
  '''Overlap scaled by the smaller set.

  Args:
    a: Token set A.
    b: Token set B.

  Returns:
    Ratio in [0, 1].
  '''
  if not a or not b:
    return 0.0
  return len(a & b) / max(1.0, min(len(a), len(b)))


def gate_failures(
  candidate: CandidateProfile,
  job: Dict[str, Any],
  enriched: Optional[Dict[str, Any]],
) -> List[str]:
  '''Evaluate all hard gates for one job.

  Args:
    candidate: Candidate profile.
    job: Job row mapping.
    enriched: Optional enrichment payload.

  Returns:
    Failed gate names; empty when everything passes.
  '''
  failures: List[str] = []
  floor = candidate.salary_floor_lpa
  sal_max = (enriched or {}).get('salary_max_lpa') or job.get('salary_max_lpa')
  if sal_max is not None and float(sal_max) < floor:
    failures.append(f'salary_floor<{floor}')

  mode = job.get('work_mode') or WorkMode.UNKNOWN
  pref = candidate.remote_pref
  allowed = {
    'any': {WorkMode.REMOTE, WorkMode.HYBRID, WorkMode.ONSITE, WorkMode.UNKNOWN},
    'remote': {WorkMode.REMOTE, WorkMode.UNKNOWN},
    'hybrid': {WorkMode.HYBRID, WorkMode.REMOTE},
    'onsite': {WorkMode.ONSITE},
  }.get(pref)
  if allowed is not None and mode not in allowed and mode != WorkMode.UNKNOWN:
    failures.append(f'work_mode:{mode}!={pref}')

  city = (job.get('city') or '').strip()
  if city and city.lower() != 'remote':
    wanted = {c.lower() for c in candidate.cities}
    if city.lower() not in wanted and not candidate.relocate_ok:
      failures.append(f'location:{city}')

  exp_min = (enriched or {}).get('experience_min_years')
  exp_min = exp_min if exp_min is not None else job.get('experience_min_yrs')
  exp_max = (enriched or {}).get('experience_max_years')
  exp_max = exp_max if exp_max is not None else job.get('experience_max_yrs')
  years = float(candidate.experience_years or 0)
  if years > 0 and exp_min is not None:
    upper = exp_max if exp_max is not None else float(exp_min) + 2
    if float(exp_min) > years + 1.5 or upper < years - 1.5:
      failures.append(f'experience:{exp_min}-{exp_max}')

  emp = (job.get('employment_type') or 'full_time').lower()
  wanted_types = [e.lower() for e in candidate.employment_types]
  if emp in ('full_time', 'part_time', 'contract', 'internship'):
    if emp not in wanted_types:
      failures.append(f'employment_type:{emp}')
  return failures


def skill_coverage(
  candidate_skill_ids: List[int],
  job_skills: List[Dict[str, Any]],
) -> Tuple[float, float]:
  '''Must/nice coverage against the candidate's skills.

  Args:
    candidate_skill_ids: Candidate's resolved skill ids.
    job_skills: Rows with skill_id and kind.

  Returns:
    (must_coverage, nice_coverage); (0.5, 0.5) when the job lists no musts.
  '''
  cand = set(candidate_skill_ids)
  musts = {row['skill_id'] for row in job_skills if row['kind'] == 'must_have'}
  nices = {row['skill_id'] for row in job_skills if row['kind'] == 'nice_to_have'}
  must_cov = len(musts & cand) / len(musts) if musts else 0.5
  nice_cov = len(nices & cand) / len(nices) if nices else 0.5
  return min(1.0, must_cov), min(1.0, nice_cov)


def seniority_fit(
  candidate_years: float,
  job_min: Optional[float],
  job_max: Optional[float],
) -> float:
  '''Triangular fit peaking when the band centers on the candidate.

  Args:
    candidate_years: Experience in years.
    job_min: Lower bound.
    job_max: Upper bound.

  Returns:
    Score in [0, 1]; 0.5 neutral when unknown.
  '''
  if candidate_years <= 0 or job_min is None:
    return 0.5
  upper = job_max if job_max is not None else job_min + 2.0
  center = (job_min + upper) / 2.0
  distance = abs(candidate_years - center)
  width = max(2.0, (upper - job_min) / 2.0 + 1.5)
  return max(0.0, 1.0 - distance / width)


def recency_score(posted_at: Optional[str], half_life_days: float = 10.0) -> float:
  '''Exponential recency with configurable half-life.

  Args:
    posted_at: ISO timestamp.
    half_life_days: Decay half-life.

  Returns:
    Score in (0, 1]; 0.5 neutral when unknown.
  '''
  if not posted_at:
    return 0.5
  try:
    then = datetime.fromisoformat(str(posted_at).replace('Z', '+00:00'))
    age_days = max(0.0, (datetime.now(timezone.utc) - then).total_seconds() / 86400.0)
  except ValueError:
    return 0.5
  return math.pow(0.5, age_days / half_life_days)


def title_fit(target_roles: List[str], title: str) -> float:
  '''Token overlap between target roles and posting title.

  Args:
    target_roles: Desired role names.
    title: Posting title.

  Returns:
    Score in [0, 1].
  '''
  role_tokens = tokens(' '.join(target_roles))
  title_tokens = tokens(title)
  return min(1.0, _overlap_ratio(role_tokens, title_tokens) * 2.0)


def semantic_fallback(candidate_text: str, job_text: str) -> float:
  '''Keyword-overlap stand-in when embeddings are unavailable.

  Args:
    candidate_text: Summary + skills text.
    job_text: Title + description text.

  Returns:
    Score in [0, 1].
  '''
  return min(1.0, _overlap_ratio(tokens(candidate_text), tokens(job_text)) * 3.0)


def salary_fit(
  salary_min: Optional[float],
  salary_max: Optional[float],
  floor_lpa: float,
) -> float:
  '''Floor-aware salary component.

  Args:
    salary_min: Posted lower bound.
    salary_max: Posted upper bound.
    floor_lpa: Hard floor.

  Returns:
    Score in [0, 1]; 0.5 neutral when unknown.
  '''
  if salary_min is None and salary_max is None:
    return 0.5
  mid_low = salary_min if salary_min is not None else salary_max
  mid_high = salary_max if salary_max is not None else salary_min
  mid = (float(mid_low) + float(mid_high)) / 2.0
  return min(1.0, mid / floor_lpa)


def company_fit(
  vertical: Optional[str],
  target_verticals: List[str],
  company_priority: int = 3,
) -> float:
  '''Vertical alignment with a pinned-company bump.

  Args:
    vertical: Job company's vertical.
    target_verticals: Candidate's preferred verticals.
    company_priority: 1..5.

  Returns:
    Score in [0, 1].
  '''
  if not vertical:
    base = 0.4
  elif not target_verticals or vertical in target_verticals:
    base = 1.0
  else:
    base = 0.5
  if company_priority >= 5:
    base = min(1.0, base + 0.2)
  return base
