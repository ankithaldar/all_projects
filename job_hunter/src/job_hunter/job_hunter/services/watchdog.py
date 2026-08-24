#!/usr/bin/env python
# -- coding: utf-8 --

'''Watchdog: post-run data-quality and freshness report.'''


from __future__ import annotations

from typing import Any, Dict, List

from job_hunter.core.config import AppSettings
from job_hunter.core.db import connect


def build_report(settings: AppSettings) -> Dict[str, Any]:
    """Assemble the health snapshot surfaced after each run.

    Args:
      settings: Application settings.

    Returns:
      Report mapping with per-source freshness and issue counts.
    """
    conn = connect(settings.db_path, readonly=True)

    sources = conn.execute(
      'SELECT j.source_key AS key, COUNT(*) AS jobs, '
      "MAX(COALESCE(j.posted_at, j.first_seen_at)) AS freshest "
      'FROM jobs j WHERE j.status = \'active\' GROUP BY j.source_key'
    ).fetchall()

    stale_companies = conn.execute(
      "SELECT COUNT(*) AS n FROM companies WHERE status = 'needs_review'",
    ).fetchone()['n']

    cooldowns = [
      row['scope'] for row in conn.execute(
        "SELECT scope FROM crawl_state WHERE cooldown_until > datetime('now')",
      )
    ]

    quarantined = conn.execute(
      "SELECT COUNT(*) AS n FROM jobs WHERE status = 'error'",
    ).fetchone()['n']

    unverified = conn.execute(
      "SELECT name FROM companies WHERE status = 'active' AND ats_provider IS NULL LIMIT 10",
    ).fetchall()
    conn.close()

    return {
      'source_freshness': [dict(row) for row in sources],
      'companies_needs_review': int(stale_companies),
      'cooled_down_sources': list(cooldowns),
      'quarantined_jobs': int(quarantined),
      'sample_unverified_companies': [row['name'] for row in unverified],
    }


def summarize_issues(report: Dict[str, Any]) -> List[str]:
    """Reduce a report to a short human-readable issue list.

    Args:
      report: Output of build_report.

    Returns:
      Issue strings (empty when healthy).
    """
    issues: List[str] = []
    if report.get('cooled_down_sources'):
        issues.append(f"sources cooling down: {', '.join(report['cooled_down_sources'])}")
    if report.get('companies_needs_review'):
        issues.append(f"{report['companies_needs_review']} companies need review")
    if report.get('quarantined_jobs'):
        issues.append(f"{report['quarantined_jobs']} quarantined postings")
    return issues
