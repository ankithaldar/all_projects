#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from typing import Dict, List, Optional, Sequence, Tuple

from job_hunter.adapters.http_client import HttpClient
from job_hunter.adapters.registry import build_adapter
from job_hunter.core.config import AppSettings, source_enabled
from job_hunter.core.models import CompanyTarget, RawJobRecord
from job_hunter.db.repositories.companies import CompaniesRepository
from job_hunter.db.repositories.jobs import JobsRepository
from job_hunter.services.persist import persist_record

logger = logging.getLogger(__name__)

'''Aggregators pulled by a bulk run when no explicit list is given.'''
DEFAULT_SOURCES: tuple = (
  'arbeitnow',
  'jobicy',
  'himalayas',
  'remotive',
  'remoteok',
  'weworkremotely',
  'cutshort',
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
    empty: Sources that returned zero records without error.
    companies_linked: Postings attached to a known company.
    new_companies: Company rows created from posting company names.
  '''

  per_source: Dict[str, int] = field(default_factory=dict)
  fetched: int = 0
  inserted: int = 0
  duplicates: int = 0
  failed: int = 0
  empty: int = 0
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
      f'{self.failed} source failures | {self.empty} empty sources'
    )


def resolve_sources(
  keys: Sequence[str],
  settings: AppSettings | Dict[str, object] | None = None,
) -> List[str]:
  '''Normalize a requested source list, falling back to the defaults.

  With no explicit keys, DEFAULT_SOURCES is filtered through
  settings.sources enablement (same check as graph build_plan).
  An explicit --sources list always overrides the enabled flag:
  the caller asked for those sources by name.

  Args:
    keys: Requested source keys; may be empty.
    settings: Application settings (or raw sources mapping); None
      leaves the defaults unfiltered.

  Returns:
    De-duplicated source keys in request order.
  '''
  wanted = [str(key).strip().lower() for key in keys if str(key).strip()]
  if not wanted:
    sources: Dict[str, object] = {}
    if isinstance(settings, dict):
      sources = settings
    elif settings is not None:
      sources = settings.sources or {}
    return [
      key for key in DEFAULT_SOURCES if source_enabled(sources, key)
    ]
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
) -> Tuple[List[RawJobRecord], bool]:
  '''Fetch one aggregator feed, reporting whether it succeeded.

  Args:
    http: Shared polite HTTP client.
    source_key: Registered source key.
    limit: Safety cap on returned records.

  Returns:
    (records, ok): Raw job records for the source and True when the
    fetch succeeded (even with zero records); ([], False) on failure
    so callers can split empty successes from failures.
  '''
  target = CompanyTarget(name=source_key, source_key=source_key)
  try:
    adapter = build_adapter(source_key, http)
    return await adapter.fetch(target, limit=limit), True
  except Exception as exc:  # noqa: BLE001 - one bad source must not stop a run
    logger.warning('source %s failed: %s', source_key, exc)
    return [], False


async def _persist_records(
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
    company_stats: Dict[str, int] = {}
    try:
      outcome = await persist_record(
        jobs_repo=jobs_repo,
        record=record,
        company_repo=companies_repo,
        quality_score=0.5,
        create_company=True,
        company_status='needs_review',
        company_priority=2,
        discovered_via=f'fetch:{record.source_key}',
        company_stats=company_stats,
      )
    except Exception as exc:  # noqa: BLE001 - skip unpersistable rows
      # Surfaced at warning, not debug: a whole source failing to insert
      # looks identical to a source with no new postings in counters.
      logger.warning(
        'insert failed for %s: %s: %s',
        record.url,
        type(exc).__name__,
        exc,
      )
      continue
    report.new_companies += company_stats.get('new_companies', 0)
    report.companies_linked += company_stats.get('companies_linked', 0)
    if outcome == 'duplicate':
      report.duplicates += 1
    else:
      report.inserted += 1


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
  sources = resolve_sources(source_keys, settings)
  report = FetchReport()
  owns_client = http is None
  client = http or HttpClient()
  try:
    seen: set = set()
    collected: List[RawJobRecord] = []
    for source_key in sources:
      result = await fetch_source(client, source_key, limit_per_source)
      # Back-compat for monkeypatched fakes returning a bare list:
      # treat a list as a successful fetch.
      if isinstance(result, tuple):
        records, ok = result
      else:
        records, ok = list(result), True
      report.per_source[source_key] = len(records)
      report.fetched += len(records)
      if not ok:
        report.failed += 1
        continue
      if not records:
        report.empty += 1
        continue
      for record in records:
        key = record.url or record.external_id
        if key in seen:
          report.duplicates += 1
          continue
        seen.add(key)
        collected.append(record)
    if persist:
      await _persist_records(settings, collected, report)
  finally:
    if owns_client:
      await client.close()
  return report
