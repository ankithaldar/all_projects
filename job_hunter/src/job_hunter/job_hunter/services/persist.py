#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Shared persist sequence for normalized job records.

Extracted from the three previously triplicated call sites
(graph/nodes.py normalize_dedupe, services/bulk_fetch.py
_persist_records, and adapters/manual_inbox.py _insert_records):
compute content_hash and canonical_url, detect city/work-mode,
resolve/create the company, insert, and report inserted/duplicate.
'''

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, Optional, Union

from job_hunter.core.models import RawJobRecord
from job_hunter.services.normalizer import (
  canonicalize_url,
  content_hash,
  detect_city,
  html_to_text,
  infer_work_mode,
  normalize_employment,
  parse_posted_at,
  parse_salary_lpa,
)

logger = logging.getLogger(__name__)


def _coerce_record(record: Union[RawJobRecord, Dict[str, Any]]) -> RawJobRecord:
  '''Coerce a dict or RawJobRecord into a RawJobRecord.

  Args:
    record: Raw record object or mapping with RawJobRecord-like keys.

  Returns:
    RawJobRecord instance.
  '''
  if isinstance(record, RawJobRecord):
    return record
  if isinstance(record, dict):
    company_name = (
      record.get('company_name') or record.get('company_raw_name') or ''
    )
    employment_raw = record.get('employment_type_raw')
    if employment_raw is None:
      # dict payloads often carry the already-canonical employment_type;
      # normalize_employment maps it back to a canonical value.
      employment_raw = record.get('employment_type') or ''
    return RawJobRecord(
      source_key=record.get('source_key') or 'manual',
      external_id=str(record.get('external_id') or ''),
      url=record.get('url') or '',
      company_name=str(company_name),
      title=record.get('title') or '',
      location_text=record.get('location_text') or '',
      description_html=record.get('description_html') or '',
      description_text=record.get('description_text') or '',
      posted_at=record.get('posted_at'),
      employment_type_raw=str(employment_raw),
      work_mode_hint=record.get('work_mode_hint') or '',
      salary_raw=record.get('salary_raw') or '',
      extra=dict(record.get('extra') or {}),
    )
  raise TypeError(
    f'record must be RawJobRecord or dict, got {type(record).__name__}'
  )


async def persist_record(
  *,
  jobs_repo,
  record: Union[RawJobRecord, Dict[str, Any]],
  company_repo=None,
  quality_score: Optional[float] = None,
  create_company: bool = False,
  company_status: str = 'needs_review',
  company_priority: int = 2,
  discovered_via: Optional[str] = None,
  source_key: Optional[str] = None,
  employment_type: Optional[str] = None,
  description_text: Optional[str] = None,
  parse_salary: bool = True,
  company_stats: Optional[Dict[str, int]] = None,
) -> str:
  '''Persist one raw job record if it is not already stored.

  Args:
    jobs_repo: Jobs repository with exists_hash/find_same_url/insert.
    record: RawJobRecord or dict shaped like one.
    company_repo: Optional companies repository for alias resolution
      and (when create_company) company creation.
    quality_score: When not None, stored on the job row (bulk 0.5,
      manual 0.3; graph path leaves it at the column default).
    create_company: Create the company when alias resolution misses
      (bulk path only; graph/manual resolve only).
    company_status: Status for a created company.
    company_priority: Priority for a created company.
    discovered_via: Origin label for a created company.
    source_key: Override the record's source key (manual forces 'manual').
    employment_type: Override the normalized employment type
      (manual forces 'full_time').
    description_text: Override the description text used for hashing.
    parse_salary: Parse salary LPA fields (manual path skips this).
    company_stats: Optional dict updated with new_companies /
      companies_linked counters when a company is created/linked.

  Returns:
    'inserted' when a new row was written, 'duplicate' when the
    content hash or canonical URL is already stored.

  Raises:
    Exception: Non-duplicate insert failures propagate to the caller,
      which logs and skips the row (preserving per-record resilience).
  '''
  raw = _coerce_record(record)
  text_desc = (
    description_text
    if description_text is not None
    else (raw.description_text or html_to_text(raw.description_html))
  )
  digest = content_hash(raw, text_desc)
  if jobs_repo.exists_hash(digest):
    return 'duplicate'
  canonical_url = canonicalize_url(raw.url)
  if raw.url and hasattr(jobs_repo, 'find_same_url'):
    try:
      if jobs_repo.find_same_url(canonical_url) is not None:
        return 'duplicate'
    except Exception:  # url check is best-effort
      logger.debug(
        'find_same_url check failed for %s', canonical_url, exc_info=True
      )

  company_id = None
  if company_repo is not None and raw.company_name:
    company_id = company_repo.resolve_alias(raw.company_name)
    if company_id is None and create_company:
      try:
        company_id = company_repo.upsert(
          name=raw.company_name,
          status=company_status,
          discovered_via=discovered_via or f'fetch:{raw.source_key}',
          priority=company_priority,
        )
      except Exception as exc:  # noqa: BLE001 - a clash must not stop the run
        logger.debug('company insert failed for %s: %s', raw.company_name, exc)
        company_id = None
      if company_id:
        company_repo.add_alias(raw.company_name, company_id)
        if company_stats is not None:
          company_stats['new_companies'] = (
            company_stats.get('new_companies', 0) + 1
          )
    if company_id and company_stats is not None:
      company_stats['companies_linked'] = (
        company_stats.get('companies_linked', 0) + 1
      )

  sal_min, sal_max = (
    parse_salary_lpa(raw.salary_raw, text_desc)
    if parse_salary
    else (None, None)
  )
  payload: Dict[str, Any] = {
    'source_key': source_key or raw.source_key,
    'external_id': raw.external_id,
    'url': raw.url,
    'canonical_url': canonical_url,
    'company_id': company_id,
    'company_raw_name': raw.company_name,
    'title': raw.title,
    'location_text': raw.location_text,
    'city': detect_city(raw.location_text, text_desc),
    'work_mode': infer_work_mode(
      raw.work_mode_hint, raw.location_text, f'{raw.title} {text_desc[:300]}'
    ),
    'employment_type': employment_type
    if employment_type is not None
    else normalize_employment(raw.employment_type_raw),
    'salary_min_lpa': sal_min,
    'salary_max_lpa': sal_max,
    'salary_raw': raw.salary_raw,
    'posted_at': parse_posted_at(raw.posted_at),
    'description_text': text_desc[:20000],
    'raw_json': raw.model_dump_json(),
    'content_hash': digest,
  }
  if quality_score is not None:
    payload['quality_score'] = quality_score
  try:
    jobs_repo.insert(payload)
  except sqlite3.IntegrityError:
    # Lost a race with a concurrent insert of the same hash/URL.
    if jobs_repo.exists_hash(digest):
      return 'duplicate'
    raise
  return 'inserted'
