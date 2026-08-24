#!/usr/bin/env python
# -- coding: utf-8 --

'''Remote-job aggregator adapters: Remotive, RemoteOK, WeWorkRemotely.'''


from __future__ import annotations

import feedparser
from bs4 import BeautifulSoup
from job_hunter.adapters.base import SourceAdapter
from job_hunter.core.models import RawJobRecord


def _strip_html(html: str) -> str:
  '''Convert HTML to plain text.

  Args:
    html: Raw HTML fragment.

  Returns:
    Plain text.
  '''
  return ' '.join(BeautifulSoup(html or '', 'html.parser').get_text(' ').split())


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


class AggregatorAdapter(SourceAdapter):
  '''Generic aggregator fetcher driven by a source key.'''

  SOURCE_URLS = {
    'remotive': ('json', 'https://remotive.com/api/remote-jobs'),
    'remoteok': ('json', 'https://remoteok.com/api'),
    'weworkremotely': (
      'rss',
      'https://weworkremotely.com/categories/remote-programming-jobs.rss',
    ),
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
    else:
      payload = await self._http.get_json(url, rpm=10)
      parser = parse_remotive if target.source_key == 'remotive' else parse_remoteok
      records = parser(payload)
    if target.board_ref:
      lowered = target.board_ref.lower()
      records = [
        record for record in records
        if lowered in record.company_name.lower()
        or lowered in record.url.lower()
      ]
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


