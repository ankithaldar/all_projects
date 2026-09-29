#!/usr/bin/env python
# -- coding: utf-8 --

'''Remote-job aggregator adapters: Remotive, RemoteOK, WeWorkRemotely.'''


from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

import feedparser
from bs4 import BeautifulSoup
from job_hunter.adapters.base import SourceAdapter
from job_hunter.core.models import CompanyTarget, RawJobRecord


def _strip_html(html: str) -> str:
  '''Convert HTML to plain text.

  Args:
    html: Raw HTML fragment.

  Returns:
    Plain text.
  '''
  return ' '.join(BeautifulSoup(html or '', 'html.parser').get_text(' ').split())


def parse_himalayas(payload: list) -> List[RawJobRecord]:
  '''Parse Himalayas /jobs/api output.

  The live API returns camelCase keys under a ``jobs`` array; older
  snake_case keys are also accepted so cached payloads keep parsing.

  Args:
    payload: Decoded JSON object or array.

  Returns:
    Raw job records.
  '''
  items = payload.get('jobs') if isinstance(payload, dict) else payload
  records: List[RawJobRecord] = []
  for item in items or []:
    if not isinstance(item, dict):
      continue
    url = str(item.get('applicationLink') or item.get('application_link') or '')
    title = str(item.get('title') or '').strip()
    if not title or not url:
      continue
    records.append(RawJobRecord(
      source_key='himalayas',
      external_id=str(item.get('guid') or item.get('id') or url),
      url=url,
      company_name=str(item.get('companyName') or item.get('company_name') or '').strip(),
      title=title,
      location_text=_himalayas_location(item),
      description_text=_strip_html(str(item.get('description') or '')),
      employment_type_raw=str(item.get('employmentType') or ''),
      work_mode_hint='remote',
      salary_raw=_himalayas_salary(item),
      posted_at=_epoch_to_iso(item.get('pubDate') or item.get('posted_at')),
      extra={
        'company_slug': str(item.get('companySlug') or ''),
        'seniority': str(item.get('seniority') or ''),
      },
    ))
  return records


def _himalayas_location(item: dict) -> str:
  '''Render a Himalayas location string from its varied field shapes.

  Args:
    item: One Himalayas job mapping.

  Returns:
    Human-readable location text, possibly empty.
  '''
  for key in ('location_strings', 'locationRestrictions'):
    value = item.get(key)
    if isinstance(value, list) and value:
      return '; '.join(str(part) for part in value)
    if isinstance(value, str) and value.strip():
      return value.strip()
  for key in ('location', 'country', 'timezone'):
    value = str(item.get(key) or '').strip()
    if value:
      return value
  return 'Remote'


def _himalayas_salary(item: dict) -> str:
  '''Render a Himalayas salary range as readable text.

  Args:
    item: One Himalayas job mapping.

  Returns:
    Salary text such as 'GBP 50000-50000 annual', or an empty string.
  '''
  low = item.get('minSalary')
  high = item.get('maxSalary')
  if low is None and high is None:
    return ''
  parts = [str(item.get('currency') or '').strip()]
  if low is not None and high is not None and str(low) != str(high):
    parts.append(f'{low}-{high}')
  else:
    parts.append(str(low if low is not None else high))
  period = str(item.get('salaryPeriod') or '').strip()
  if period:
    parts.append(period)
  return ' '.join(part for part in parts if part)


def parse_remotive(payload: dict) -> List[RawJobRecord]:
  '''Parse Remotive /api/remote-jobs output.

  Args:
    payload: Decoded JSON.

  Returns:
    Raw job records.
  '''
  records: List[RawJobRecord] = []
  for item in payload.get('jobs') or []:
    salary = str(item.get('salary') or '')
    records.append(RawJobRecord(
      source_key='remotive',
      external_id=str(item.get('id') or ''),
      url=str(item.get('url') or ''),
      company_name=str(item.get('company_name') or ''),
      title=str(item.get('title') or '').strip(),
      location_text=str(item.get('candidate_required_location') or ''),
      description_text=_strip_html(str(item.get('description') or '')),
      employment_type_raw=str(item.get('job_type') or ''),
      work_mode_hint='remote',
      salary_raw=salary,
      posted_at=item.get('publication_date'),
    ))
  return [record for record in records if record.title and record.url]


def parse_remoteok(payload: list) -> List[RawJobRecord]:
  '''Parse RemoteOK /api output (first element is a legal notice).

  Args:
    payload: Decoded JSON array.

  Returns:
    Raw job records.
  '''
  records: List[RawJobRecord] = []
  tail = payload[1:] if payload and isinstance(payload[0], dict) and 'legal' in (payload[0] or {}) else payload
  for item in tail or []:
    if not isinstance(item, dict):
      continue
    records.append(RawJobRecord(
      source_key='remoteok',
      external_id=str(item.get('id') or item.get('slug') or ''),
      url=str(item.get('url') or ''),
      company_name=str(item.get('company') or '').strip(),
      title=str(item.get('position') or '').strip(),
      location_text='Remote' if item.get('location') in (None, '', 'Worldwide') else str(item['location']),
      description_text=_strip_html(str(item.get('description') or '')),
      salary_raw=f"{item.get('salary_min', '')}-{item.get('salary_max', '')}".strip('-'),
      posted_at=str(item.get('date') or '') or None,
      extra={'tags': item.get('tags') or []},
    ))
  return [record for record in records if record.title and record.url]


def parse_wwr_rss(text: str) -> List[RawJobRecord]:
  '''Parse a WeWorkRemotely RSS category feed.

  Args:
    text: RSS XML text.

  Returns:
    Raw job records (title format often "Company: Role").
  '''
  records: List[RawJobRecord] = []
  feed = feedparser.parse(text)
  for entry in feed.entries:
    title = entry.get('title', '')
    company, _, role = title.partition(':')
    if not role.strip():
      company, role = '', title
    link = str(entry.get('link') or '')
    records.append(RawJobRecord(
      source_key='weworkremotely',
      external_id=link,
      url=link,
      company_name=company.strip(),
      title=role.strip(),
      location_text='Remote',
      work_mode_hint='remote',
      description_text=_strip_html(str(entry.get('summary') or '')),
      posted_at=None,
    ))
  return [record for record in records if record.title and record.url]


def parse_arbeitnow(payload: dict) -> List[RawJobRecord]:
  '''Parse Arbeitnow /api/job-board-api output.

  Each item is a single posting rather than a list of jobs, and the payload
  carries the employer's own website as the posting URL.

  Args:
    payload: Decoded JSON with a 'data' array.

  Returns:
    Raw job records.
  '''
  records: List[RawJobRecord] = []
  for item in payload.get('data') or []:
    if not isinstance(item, dict):
      continue
    url = str(item.get('url') or '')
    if not url:
      continue
    location = str(item.get('location') or '').strip()
    records.append(RawJobRecord(
      source_key='arbeitnow',
      external_id=str(item.get('slug') or ''),
      url=url,
      company_name=str(item.get('company_name') or '').strip(),
      title=str(item.get('title') or '').strip(),
      location_text=location,
      description_text=_strip_html(str(item.get('description') or '')),
      employment_type_raw='; '.join(item.get('job_types') or []),
      work_mode_hint='remote' if item.get('remote') else '',
      posted_at=_epoch_to_iso(item.get('created_at')),
      extra={'tags': item.get('tags') or []},
    ))
  return [record for record in records if record.title]


def parse_jobicy(payload: dict) -> List[RawJobRecord]:
  '''Parse Jobicy /api/v2/remote-jobs output.

  Args:
    payload: Decoded JSON with a 'jobs' array.

  Returns:
    Raw job records.
  '''
  records: List[RawJobRecord] = []
  for item in payload.get('jobs') or []:
    if not isinstance(item, dict):
      continue
    url = str(item.get('url') or '')
    if not url:
      continue
    records.append(RawJobRecord(
      source_key='jobicy',
      external_id=str(item.get('jobSlug') or item.get('id') or ''),
      url=url,
      company_name=str(item.get('companyName') or '').strip(),
      title=str(item.get('jobTitle') or '').strip(),
      location_text=str(item.get('jobGeo') or ''),
      description_text=_strip_html(str(item.get('jobDescription') or '')),
      employment_type_raw='; '.join(item.get('jobType') or []),
      work_mode_hint='remote',
      salary_raw=_jobicy_salary(item),
      posted_at=item.get('pubDate'),
      extra={'industry': item.get('jobIndustry') or []},
    ))
  return [record for record in records if record.title]


def _jobicy_salary(item: dict) -> str:
  '''Render a Jobicy salary range as readable text.

  Args:
    item: One Jobicy job mapping.

  Returns:
    Salary text such as 'USD 208000-240000 yearly', or an empty string.
  '''
  low = item.get('salaryMin')
  high = item.get('salaryMax')
  if low is None and high is None:
    return ''
  parts = [str(item.get('salaryCurrency') or '').strip()]
  if low is not None and high is not None:
    parts.append(f'{low}-{high}')
  else:
    parts.append(str(low if low is not None else high))
  period = str(item.get('salaryPeriod') or '').strip()
  if period:
    parts.append(period)
  return ' '.join(part for part in parts if part)


def _epoch_to_iso(value: object) -> Optional[str]:
  '''Convert a Unix timestamp to an ISO 8601 date string.

  Args:
    value: Seconds since the epoch, or None.

  Returns:
    Date portion of the UTC timestamp, or None when unparseable.
  '''
  try:
    seconds = int(value)  # type: ignore[arg-type]
  except (TypeError, ValueError):
    return None
  if seconds <= 0:
    return None
  return datetime.fromtimestamp(seconds, tz=timezone.utc).date().isoformat()


class AggregatorAdapter(SourceAdapter):
  '''Generic aggregator fetcher driven by a source key.'''

  SOURCE_URLS = {
    'remotive': ('json', 'https://remotive.com/api/remote-jobs'),
    'remoteok': ('json', 'https://remoteok.com/api'),
    'weworkremotely': (
      'rss',
      'https://weworkremotely.com/categories/remote-programming-jobs.rss',
    ),
    'himalayas': ('json', 'https://himalayas.app/jobs/api?limit=100'),
    'arbeitnow': ('json', 'https://www.arbeitnow.com/api/job-board-api'),
    'jobicy': ('json', 'https://jobicy.com/api/v2/remote-jobs?count=100'),
  }

  async def fetch(self, target: CompanyTarget, limit: int = 200) -> List[RawJobRecord]:
    '''Fetch and parse one aggregator feed.

    Args:
      target: Only target.source_key matters; board_ref filters company.
      limit: Safety cap.

    Returns:
      Raw job records, optionally filtered to the target company.

    Raises:
      AdapterError: On unknown source key.
    '''
    kind, url = self.SOURCE_URLS[target.source_key]
    if kind == 'rss':
      text = await self._http.get_text(url, rpm=10)
      records = parse_wwr_rss(text)
    elif target.source_key == 'himalayas':
      records = await self._fetch_himalayas(url, limit)
    else:
      payload = await self._http.get_json(url, rpm=10)
      parser = {
        'remotive': parse_remotive,
        'remoteok': parse_remoteok,
        'arbeitnow': parse_arbeitnow,
        'jobicy': parse_jobicy,
      }[target.source_key]
      records = parser(payload)
    if target.board_ref:
      lowered = target.board_ref.lower()
      records = [
        record for record in records
        if lowered in record.company_name.lower()
        or lowered in record.url.lower()
      ]
    return records[:limit]

  async def _fetch_himalayas(self, url: str, limit: int) -> List[RawJobRecord]:
    '''Page through the Himalayas feed using its cursor.

    Himalayas caps each response at 20 records regardless of the requested
    limit, so a plain single call would leave most of the feed unharvested.

    Args:
      url: Feed base URL.
      limit: Safety cap on returned records.

    Returns:
      Raw job records across as many pages as needed to reach the limit.
    '''
    records: List[RawJobRecord] = []
    cursor: Optional[str] = None
    seen_cursors: set = set()
    while len(records) < limit:
      payload = await self._http.get_json(
        url, params={'cursor': cursor} if cursor else None, rpm=10,
      )
      if not isinstance(payload, dict):
        break
      batch = parse_himalayas(payload)
      if not batch:
        break
      records.extend(batch)
      cursor = str(payload.get('nextCursor') or '')
      if not cursor or cursor in seen_cursors:
        break
      seen_cursors.add(cursor)
    return records[:limit]

  async def health(self, target: CompanyTarget) -> bool:
    '''Probe the configured feed URL.

    Args:
      target: Company whose source_key selects the feed.

    Returns:
      True when the feed answers.
    '''
    try:
      kind, url = self.SOURCE_URLS[target.source_key]
      if kind == 'rss':
        return bool(await self._http.get_text(url, rpm=10))
      await self._http.get_json(url, rpm=10)
      return True
    except Exception:
      return False


