#!/usr/bin/env python
# -- coding: utf-8 --

'''Dedupe policy on top of content hashes and canonical URLs.'''


from __future__ import annotations

from typing import List, Tuple

from rapidfuzz import fuzz
from job_hunter.core.models import NormalizedJob
from job_hunter.db.repositories.jobs import JobsRepository


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
    company_id: int,
    threshold: float = 95.0,
    candidates: List[str] = (),
  ) -> bool:
    '''Fuzzy guard against near-identical reposts at one company.

    Args:
      title: New posting title.
      company_id: Company id.
      threshold: Token-sort-ratio cutoff.
      candidates: Pre-fetched titles for the company (optional).

    Returns:
      True when a very similar live title exists.
    '''
    if not candidates:
      return False
    for candidate in candidates:
      if float(fuzz.token_sort_ratio(title.lower(), candidate.lower())) >= threshold:
        return True
    _ = company_id
    return False
