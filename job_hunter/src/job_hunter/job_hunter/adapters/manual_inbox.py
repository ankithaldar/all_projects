#!/usr/bin/env python
# -- coding: utf-8 --

'''Manual-export inbox: parse saved LinkedIn HTML and Naukri-style CSVs.'''


from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any, Dict, List

from bs4 import BeautifulSoup
from job_hunter.core.config import AppSettings
from job_hunter.core.models import RawJobRecord
from job_hunter.db.repositories.companies import CompaniesRepository
from job_hunter.db.repositories.jobs import JobsRepository
from job_hunter.services.normalizer import (
  canonicalize_url,
  content_hash,
  html_to_text,
)

_HEADER_HINTS = {
  'title': ('job title', 'title', 'role', 'designation'),
  'company': ('company', 'company name', 'organisation', 'organization'),
  'location': ('location', 'job location', 'city'),
  'salary': ('salary', 'ctc', 'pay'),
  'url': ('url', 'link', 'job url'),
}


def _match_headers(fieldnames: List[str]) -> Dict[str, str]:
  '''Map CSV headers onto canonical fields via hints.

  Args:
    fieldnames: CSV header row.

  Returns:
    Mapping of canonical field to actual header name.
  '''
  mapping: Dict[str, str] = {}
  lowered = {name.strip().lower(): name for name in fieldnames or []}
  for canonical, hints in _HEADER_HINTS.items():
    for hint in hints:
      for low, original in lowered.items():
        if hint == low and canonical not in mapping:
          mapping[canonical] = original
  return mapping


def parse_linkedin_html(html: str) -> List[RawJobRecord]:
  '''Heuristically extract job cards from a saved LinkedIn search page.

  Args:
    html: Saved page source.

  Returns:
    Raw records flagged source=manual.
  '''
  soup = BeautifulSoup(html or '', 'html.parser')
  records: List[RawJobRecord] = []
  cards = soup.select('div.base-card, li[data-occludable-job-id], div.job-search-card')
  for card in cards:
    anchor = card.find('a', class_=re.compile('base-card__full-link|job-card-list__title'))
    if anchor is None:
      anchor = card.find('a', href=True)
    if anchor is None:
      continue
    title = (anchor.get_text(' ', strip=True) or anchor.get('aria-label') or '').strip()
    company_el = card.find(class_=re.compile('subtitle|company|org-name'))
    location_el = card.find(class_=re.compile('location|meta'))
    link = anchor.get('href') or ''
    if not title or not link:
      continue
    records.append(RawJobRecord(
      source_key='manual',
      external_id=f"li:{abs(hash(link)) % 10**10}",
      url=link.split('?')[0],
      company_name=(company_el.get_text(' ', strip=True) if company_el else ''),
      title=title,
      location_text=(location_el.get_text(' ', strip=True) if location_el else ''),
      extra={'origin': 'linkedin_export'},
    ))
  return records


def parse_csv(text: str) -> List[RawJobRecord]:
  """Parse a Naukri-style CSV export with fuzzy header mapping.

  Args:
    text: CSV content.

  Returns:
    Raw records flagged source=manual.
  """
  reader = csv.DictReader(io.StringIO(text))
  mapping = _match_headers(list(reader.fieldnames or []))
  if 'title' not in mapping:
    return []
  records: List[RawJobRecord] = []
  for row in reader:
    title = (row.get(mapping['title']) or '').strip()
    if not title:
      continue
    records.append(RawJobRecord(
      source_key='manual',
      external_id=f"csv:{abs(hash(title + (row.get(mapping.get('company') or '') or ''))) % 10**10}",
      url=row.get(mapping.get('url') or '', ''),
      company_name=(row.get(mapping['company']) or '').strip() if 'company' in mapping else '',
      title=title,
      location_text=(row.get(mapping['location']) or '').strip() if 'location' in mapping else '',
      salary_raw=(row.get(mapping['salary']) or '').strip() if 'salary' in mapping else '',
      extra={'origin': 'naukri_csv'},
    ))
  return records


def _insert_records(settings: AppSettings, records: List[RawJobRecord]) -> int:
  '''Run manual records through normalization and insert unseen ones.

  Args:
    settings: App settings.
    records: Parsed records.

  Returns:
    Inserted count.
  '''
  from job_hunter.services.normalizer import (
    detect_city,
    infer_work_mode,
  )
  jobs_repo = JobsRepository(settings.db_path)
  companies_repo = CompaniesRepository(settings.db_path)
  inserted = 0
  for record in records:
    digest = content_hash(record, '')
    if jobs_repo.exists_hash(digest):
      continue
    company_id = (
      companies_repo.resolve_alias(record.company_name)
      if record.company_name else None
    )
    try:
      jobs_repo.insert({
        'source_key': 'manual',
        'external_id': record.external_id,
        'url': record.url,
        'canonical_url': canonicalize_url(record.url),
        'company_id': company_id,
        'company_raw_name': record.company_name,
        'title': record.title,
        'location_text': record.location_text,
        'city': detect_city(record.location_text),
        'work_mode': infer_work_mode('', record.location_text, record.title),
        'employment_type': 'full_time',
        'salary_raw': record.salary_raw,
        'description_text': '',
        'raw_json': record.model_dump_json(),
        'content_hash': digest,
        'quality_score': 0.3,
      })
      inserted += 1
    except Exception:
      continue
  return inserted


async def scan_inbox(settings: AppSettings) -> int:
    """Scan data/inbox/** for new exports and ingest them.

    Args:
      settings: App settings.

    Returns:
      Total postings ingested.
    """
    inbox = settings.data_dir / 'inbox'
    total = 0
    if not inbox.exists():
      return 0
    for path in sorted(inbox.rglob('*')):
      if path.name.startswith('.') or not path.is_file():
        continue
      suffix = path.suffix.lower()
      try:
        if suffix in ('.html', '.htm'):
          records = parse_linkedin_html(path.read_text(encoding='utf-8', errors='ignore'))
        elif suffix == '.csv':
          records = parse_csv(path.read_text(encoding='utf-8', errors='ignore'))
        else:
          continue
        total += _insert_records(settings, records)
        processed = settings.data_dir / 'inbox' / '_processed'
        processed.mkdir(exist_ok=True)
        new_path = processed / f'{int(Path.mtime(path))}_{path.name}'
        path.rename(new_path)
      except Exception:
        continue
    return total
