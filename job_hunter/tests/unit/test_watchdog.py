#!/usr/bin/env python
# -- coding: utf-8 --

'''Tests for the watchdog report.'''


from __future__ import annotations

from pathlib import Path

from job_hunter.core.config import AppSettings
from job_hunter.core.db import run_migrations
from job_hunter.services.watchdog import build_report, summarize_issues


def test_report_on_empty_db(tmp_path: Path, monkeypatch) -> None:
    """Report builds cleanly with no data and lists no issues.

    Args:
      tmp_path: Pytest temporary directory.
      monkeypatch: Pytest fixture.
    """
    run_migrations(tmp_path / 'app.db')
    config_path = tmp_path / 'app.yaml'
    config_path.write_text('salary_hard_floor_lpa: 45\n', encoding='utf-8')
    monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
    settings = AppSettings(config_path)
    report = build_report(settings)
    assert report['source_freshness'] == []
    assert report['companies_needs_review'] == 0
    assert summarize_issues(report) == []


def test_issue_summary_counts(tmp_path: Path) -> None:
    """Issue summarizer reflects report contents."""
    issues = summarize_issues({
      'cooled_down_sources': ['company:7'],
      'companies_needs_review': 3,
      'quarantined_jobs': 1,
    })
    assert len(issues) == 3
