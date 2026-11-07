#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Watchdog: post-run data-quality and freshness report.'''


from __future__ import annotations

from typing import Any, Dict, List

from job_hunter.core.config import AppSettings
from job_hunter.core.db import fetch_all, fetch_one


def build_report(settings: AppSettings) -> Dict[str, Any]:
  '''Assemble the health snapshot surfaced after each run.

  Args:
    settings: Application settings.

  Returns:
    Report mapping with per-source freshness and issue counts.
  '''
  # SQL-backed by design: no repository method covers these
  # aggregate watchdog queries (source freshness, review and
  # cooldown counts, error counts, unverified samples), and
  # db/ repositories are outside this task's ownership, so the
  # queries stay here but run via fetch_all/fetch_one, which
  # always close their readonly connections.
  db_path = settings.db_path
  with fetch_all(
    db_path,
    'SELECT j.source_key AS key, COUNT(*) AS jobs, '
    'MAX(COALESCE(j.posted_at, j.first_seen_at)) AS freshest '
    'FROM jobs j WHERE j.status = \'active\' GROUP BY j.source_key',
  ) as sources:
    sources = list(sources)
  with fetch_one(
    db_path,
    'SELECT COUNT(*) AS n FROM companies '
    'WHERE status = \'needs_review\'',
  ) as stale_row:
    stale_companies = stale_row['n']
  with fetch_all(
    db_path,
    'SELECT scope FROM crawl_state '
    'WHERE cooldown_until > datetime(\'now\')',
  ) as cooldown_rows:
    cooldowns = [row['scope'] for row in cooldown_rows]
  with fetch_one(
    db_path,
    'SELECT COUNT(*) AS n FROM jobs WHERE status = \'error\'',
  ) as quarantined_row:
    quarantined = quarantined_row['n']
  with fetch_all(
    db_path,
    'SELECT name FROM companies WHERE status = \'active\' '
    'AND ats_provider IS NULL LIMIT 10',
  ) as unverified:
    unverified = list(unverified)
  return {
    'source_freshness': [dict(row) for row in sources],
    'companies_needs_review': int(stale_companies),
    'cooled_down_sources': list(cooldowns),
    'quarantined_jobs': int(quarantined),
    'sample_unverified_companies': [row['name'] for row in unverified],
  }


def summarize_issues(report: Dict[str, Any]) -> List[str]:
  '''Reduce a report to a short human-readable issue list.

  Args:
    report: Output of build_report.

  Returns:
    Issue strings (empty when healthy).
  '''
  issues: List[str] = []
  cooled = report.get('cooled_down_sources')
  if cooled:
    joined = ', '.join(cooled)
    issues.append(f'sources cooling down: {joined}')
  needs_review = report.get('companies_needs_review')
  if needs_review:
    issues.append(f'{needs_review} companies need review')
  quarantined = report.get('quarantined_jobs')
  if quarantined:
    issues.append(f'{quarantined} quarantined postings')
  return issues
