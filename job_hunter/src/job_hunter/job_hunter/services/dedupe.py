#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Dedupe policy on top of content hashes and canonical URLs.'''

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

from rapidfuzz import fuzz
from job_hunter.core.models import NormalizedJob
from job_hunter.db.repositories.jobs import JobsRepository


def _candidate_title_and_company(
  candidate: Any,
) -> Tuple[Optional[str], Optional[Any]]:
  '''Extract (title, company_id) from a candidate in any supported shape.

  Args:
    candidate: Plain title string, mapping with title/company_id,
      (company_id, title) pair, or object with title/company_id attrs.

  Returns:
    (title or None, company_id or None) where None company means the
    candidate was pre-fetched for the target company.
  '''
  if isinstance(candidate, str):
    return candidate, None
  if isinstance(candidate, dict):
    title = candidate.get('title') or candidate.get('name')
    return (str(title) if title else None), candidate.get('company_id')
  if isinstance(candidate, (tuple, list)) and len(candidate) >= 2:
    # (company_id, title) pair; a pair of strings is treated as (title, ...)
    first, second = candidate[0], candidate[1]
    if isinstance(first, int) and isinstance(second, str):
      return second, first
    if isinstance(second, str):
      return second, first
    return None, None
  title = getattr(candidate, 'title', None)
  if isinstance(title, str):
    return title, getattr(candidate, 'company_id', None)
  return None, None


class DedupeEngine:
  '''Decide whether a normalized job is new, a duplicate, or a refresh.'''

  def __init__(self, jobs_repo: JobsRepository) -> None:
    '''Initialize the engine.

    Args:
      jobs_repo: Jobs repository.
    '''
    self._jobs = jobs_repo

  def classify(self, job: NormalizedJob) -> Tuple[str, int]:
    '''Classify one normalized job.

    Args:
      job: Normalized candidate row.

    Returns:
      (verdict, existing_job_id) where verdict is new|hash_dup|url_dup.
    '''
    if self._jobs.exists_hash(job.content_hash):
      return 'hash_dup', 0
    existing = self._jobs.find_same_url(job.canonical_url)
    if existing is not None:
      return 'url_dup', int(existing['id'])
    return 'new', 0

  def similar_title_exists(
    self,
    title: str,
    company_id: Optional[int] = None,
    threshold: float = 95.0,
    candidates: Sequence[Any] = (),
  ) -> bool:
    '''Fuzzy guard against near-identical reposts at one company.

    Args:
      title: New posting title.
      company_id: Company id; when given, structured candidates
        carrying a different company_id are filtered out. Plain
        string candidates are treated as pre-fetched for this
        company. None disables filtering.
      threshold: Token-sort-ratio cutoff.
      candidates: Pre-fetched titles for the company (optional),
        as strings or mappings/pairs/objects with company_id.

    Returns:
      True when a very similar live title exists.
    '''
    if not candidates:
      return False
    for candidate in candidates:
      candidate_title, candidate_company = _candidate_title_and_company(
        candidate
      )
      if not candidate_title:
        continue
      if company_id is not None and candidate_company is not None:
        try:
          if int(candidate_company) != int(company_id):
            continue
        except TypeError, ValueError:
          if str(candidate_company) != str(company_id):
            continue
      if (
        float(fuzz.token_sort_ratio(title.lower(), candidate_title.lower()))
        >= threshold
      ):
        return True
    return False
