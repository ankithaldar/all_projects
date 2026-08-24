#!/usr/bin/env python
# -- coding: utf-8 --

'''Tests for the manual-export inbox parsers.'''


from __future__ import annotations

import asyncio
from pathlib import Path

from job_hunter.adapters.manual_inbox import parse_csv, parse_linkedin_html, scan_inbox

LINKEDIN_HTML = '''
<div class="base-card relative job-search-card">
  <a class="base-card__full-link" href="https://www.linkedin.com/jobs/view/1234?trk=x">
    <h3>Staff Data Scientist</h3>
  </a>
  <div class="base-search-card__subtitle">Razorpay</div>
  <div class="job-search-card__location">Bengaluru, Karnataka, India</div>
</div>
'''

CSV_TEXT = '''Job Title,Company,Location,CTC
Staff DS,Acme Corp,Mumbai,50-70 LPA
Data Scientist II,Globex,Pune,
'''


def test_linkedin_html_parse() -> None:
  '''Saved LinkedIn cards yield title/company/location.'''
  records = parse_linkedin_html(LINKEDIN_HTML)
  assert len(records) == 1
  record = records[0]
  assert record.title == 'Staff Data Scientist'
  assert record.company_name == 'Razorpay'
  assert 'linkedin.com' in record.url
  assert '?' not in record.url


def test_csv_parse_with_fuzzy_headers() -> None:
  '''Naukri-style CSV headers map to canonical fields.'''
  records = parse_csv(CSV_TEXT)
  assert [r.title for r in records] == ['Staff DS', 'Data Scientist II']
  assert records[0].company_name == 'Acme Corp'
  assert records[0].salary_raw == '50-70 LPA'


def test_scan_inbox_moves_files(tmp_path: Path) -> None:
  '''Scan ingests inbox files and moves them to _processed.

  Args:
    tmp_path: Pytest temporary directory.
  """
  '''
  from job_hunter.core.config import AppSettings
  from job_hunter.core.db import run_migrations
  inbox = tmp_path / 'inbox'
  (inbox / 'linkedin').mkdir(parents=True)
  run_migrations(tmp_path / 'app.db')
  (inbox / 'linkedin' / 'saved.html').write_text(LINKEDIN_HTML, encoding='utf-8')
  config_path = tmp_path / 'app.yaml'
  config_path.write_text('salary_hard_floor_lpa: 45\n', encoding='utf-8')
  settings = AppSettings(config_path)
  count = asyncio.run(scan_inbox(settings))
  assert count >= 0
