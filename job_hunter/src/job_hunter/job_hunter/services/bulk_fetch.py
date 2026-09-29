#!/usr/bin/env python
# -- coding: utf-8 --

'''Aggressive multi-source job ingestion.

This module widens the fetch surface beyond the per-company ATS crawl that
the discovery graph performs. It pulls whole feeds from the public job
aggregator APIs, which is where the highest posting volume lives, then runs
every record through the same normalization and dedupe path used elsewhere
so the normal scoring pipeline is unaffected.
'''


from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from job_hunter.adapters.http_client import HttpClient
from job_hunter.adapters.registry import build_adapter
from job_hunter.core.config import AppSettings
from job_hunter.core.models import CompanyTarget, RawJobRecord
from job_hunter.db.repositories.companies import CompaniesRepository
from job_hunter.db.repositories.jobs import JobsRepository
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

'''Aggregators pulled by a bulk run when no explicit list is given.'''
DEFAULT_SOURCES: tuple = (
  'arbeitnow', 'jobicy', 'himalayas', 'remotive', 'remoteok', 'weworkremotely',
)


@dataclass
class FetchReport:
  '''Counters for one bulk fetch pass.

  Attributes:
    per_source: Records returned by each source key.
    fetched: Total records returned across sources.
    inserted: Postings written as new rows.
    duplicates: Postings rejected by the dedupe engine.
    failed: Sources that raised an error.
    companies_linked: Postings attached to a known company.
    new_companies: Company rows created from posting company names.
  '''

  per_source: Dict[str, int] = field(default_factory=dict)
  fetched: int = 0
  inserted: int = 0
  duplicates: int = 0
  failed: int = 0
  companies_linked: int = 0
  new_companies: int = 0

  def summary(self) -> str:
    '''Render a one-line human summary of the pass.

    Returns:
      Space-separated counters.
    '''
    parts = [f'{key}={value}' for key, value in sorted(self.per_source.items())]
    return (
      f'{self.fetched} fetched ({", ".join(parts) or "no sources"}) | '
      f'{self.inserted} new | {self.duplicates} dupes | '
      f'{self.companies_linked} linked | {self.new_companies} new companies | '
      f'{self.failed} source failures'
    )


def resolve_sources(keys: Sequence[str]) -> List[str]:
  '''Normalize a requested source list, falling back to the defaults.

  Args:
    keys: Requested source keys; may be empty.

  Returns:
    De-duplicated source keys in request order.
  '''
  wanted = [str(key).strip().lower() for key in keys if str(key).strip()]
  if not wanted:
    return list(DEFAULT_SOURCES)
  seen: set = set()
  ordered: List[str] = []
  for key in wanted:
    if key not in seen:
      seen.add(key)
      ordered.append(key)
  return ordered


async def fetch_source(
  http: HttpClient,
  source_key: str,
  limit: int,
) -> List[RawJobRecord]:
  '''Fetch one aggregator feed, returning an empty list on failure.

  Args:
    http: Shared polite HTTP client.
    source_key: Registered source key.
    limit: Safety cap on returned records.

  Returns:
    Raw job records for the source.
  '''
  target = CompanyTarget(name=source_key, source_key=source_key)
  try:
    adapter = build_adapter(source_key, http)
    return await adapter.fetch(target, limit=limit)
  except Exception as exc:  # noqa: BLE001 - one bad source must not stop a run
    logger.warning('source %s failed: %s', source_key, exc)
    return []


def _persist_records(
  settings: AppSettings,
  records: Sequence[RawJobRecord],
  report: FetchReport,
) -> None:
  '''Normalize and store fetched records, creating companies as needed.

  Args:
    settings: Application settings.
    records: Records to persist.
    report: Report updated in place with the insertion counters.
  '''
  jobs_repo = JobsRepository(settings.db_path)
  companies_repo = CompaniesRepository(settings.db_path)
  for record in records:
    if not record.title or not record.url:
      continue
    text_desc = record.description_text or html_to_text(record.description_html)
    digest = content_hash(record, text_desc)
    if jobs_repo.exists_hash(digest):
      report.duplicates += 1
      continue
    company_id = None
    if record.company_name:
      company_id = companies_repo.resolve_alias(record.company_name)
      if company_id is None:
        try:
          company_id = companies_repo.upsert(
            name=record.company_name,
            status='needs_review',
            discovered_via=f'fetch:{record.source_key}',
            priority=2,
          )
        except Exception as exc:  # noqa: BLE001 - a clash must not stop the run
          logger.debug('company insert failed for %s: %s', record.company_name, exc)
          company_id = None
        if company_id:
          companies_repo.add_alias(record.company_name, company_id)
          report.new_companies += 1
      if company_id:
        report.companies_linked += 1
    sal_min, sal_max = parse_salary_lpa(record.salary_raw, text_desc)
    try:
      jobs_repo.insert({
        'source_key': record.source_key,
        'external_id': record.external_id,
        'url': record.url,
        'canonical_url': canonicalize_url(record.url),
        'company_id': company_id,
        'company_raw_name': record.company_name,
        'title': record.title,
        'location_text': record.location_text,
        'city': detect_city(record.location_text, text_desc),
        'work_mode': infer_work_mode(
          record.work_mode_hint, record.location_text, f'{record.title} {text_desc[:300]}',
        ),
        'employment_type': normalize_employment(record.employment_type_raw),
        'salary_min_lpa': sal_min,
        'salary_max_lpa': sal_max,
        'salary_raw': record.salary_raw,
        'posted_at': parse_posted_at(record.posted_at),
        'description_text': text_desc[:20000],
        'raw_json': record.model_dump_json(),
        'content_hash': digest,
        'quality_score': 0.5,
      })
      report.inserted += 1
    except Exception as exc:  # noqa: BLE001 - skip unpersistable rows
      logger.debug('insert failed for %s: %s', record.url, exc)


async def bulk_fetch(
  settings: AppSettings,
  source_keys: Sequence[str] = (),
  limit_per_source: int = 2000,
  persist: bool = True,
  http: Optional[HttpClient] = None,
) -> FetchReport:
  '''Fetch every requested aggregator and store the unseen postings.

  Args:
    settings: Application settings.
    source_keys: Source keys to fetch; empty means the default aggregators.
    limit_per_source: Safety cap per source.
    persist: Whether to write records to the database.
    http: Optional client to reuse; one is created and closed otherwise.

  Returns:
      A :class:`FetchReport` describing the pass.
  '''
  sources = resolve_sources(source_keys)
  report = FetchReport()
  owns_client = http is None
  client = http or HttpClient()
  try:
    seen: set = set()
    collected: List[RawJobRecord] = []
    for source_key in sources:
      records = await fetch_source(client, source_key, limit_per_source)
      if not records:
        report.failed += 1
        report.per_source[source_key] = 0
        continue
      report.per_source[source_key] = len(records)
      report.fetched += len(records)
      for record in records:
        key = record.url or record.external_id
        if key in seen:
          report.duplicates += 1
          continue
        seen.add(key)
        collected.append(record)
    if persist:
      _persist_records(settings, collected, report)
  finally:
    if owns_client:
      await client.close()
  return report
