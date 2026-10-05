#!/usr/bin/env python
# -- coding: utf-8 --

'''Cutshort (cutshort.io) adapter: India-native tech job board.

Cutshort is a hiring platform for technology companies in India, in the same
category as Naukri or LinkedIn but for the startup/tech segment. Unlike the
global remote boards already wired in (Remotive, RemoteOK, Himalayas), its
inventory is India-first: measured across nine category pages, 76% of postings
carry an Indian city and 99% carry a structured INR salary.

There is no public read API without a key -- ``/api/v1/jobs`` answers 401
``invalid_api_key``. The keyless path is the category pages themselves, which
are server-rendered Next.js and embed their job array in a ``__NEXT_DATA__``
script island. ``/jobs/<category>`` is not disallowed by robots.txt, and the
site publishes an ``llms.txt`` advertising agent access.

Two upstream quirks this adapter has to absorb:

  - Pagination does not work. ``?page=2`` and ``?page=3`` return byte-identical
    payloads, so a category yields its first 50 postings and nothing more. The
    remaining volume is only reachable through a keyed API, so the limit here
    is per-category rather than a global cap.
  - Five of nine category URLs return HTTP 200 with no embedded data at all
    (client-rendered only), while others embed 50 records. A 200 therefore says
    nothing about whether a category yielded anything, so each category is
    probed independently and empties are skipped silently.

The structured ``salaryRange.min``/``max`` are deliberately not trusted. They
disagree with the rendered text: a posting displayed as ``Rs 15L - Rs 20L``
reports ``min=750000``, because the pair is a low-midpoint bracket rather than
the true bounds. The display string is the value the poster typed, so salary is
derived from ``salaryRangeText`` and the numbers are used only to annualize a
monthly posting.
'''

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from job_hunter.adapters.base import SourceAdapter
from job_hunter.adapters.http_client import HttpClient
from job_hunter.core.errors import AdapterError
from job_hunter.core.models import CompanyTarget, RawJobRecord

logger = logging.getLogger(__name__)

SOURCE_KEY = 'cutshort'
BASE = 'https://cutshort.io'

'''Category slugs worth pulling, ordered by relevance to data/ML hiring.

Slugs that 404 or render client-side only are dropped by the probe rather than
special-cased, so a future Cutshort URL change degrades to fewer categories
instead of an exception.
'''
CATEGORY_SLUGS: tuple = (
  'datascience-jobs',
  'machine-learning-jobs',
  'ai-jobs',
  'data-engineer-jobs',
  'data-science-analytics-jobs',
  'business-analytics-jobs',
  'devops-jobs',
  'software-development-jobs',
  'product-management-jobs',
)

_NEXT_DATA_RE = re.compile(r'id="__NEXT_DATA__"[^>]*>(\{.*?\})</script>', re.DOTALL)

'''``Rs 15L - Rs 20L / yr`` and friends, as printed on the posting.'''
_LPA_RANGE_RE = re.compile(
  r'(?:₹|\brs\.?\s*)?(\d{1,3}(?:\.\d)?)\s*(?:l|lakh|lakhs|lp|lpa)\b'
  r'\s*(?:-|–|—|to)\s*'
  r'(?:₹|\brs\.?\s*)?(\d{1,3}(?:\.\d)?)\s*(?:l|lakh|lakhs|lp|lpa)\b',
  re.IGNORECASE,
)
_LPA_SINGLE_RE = re.compile(
  r'(?:₹|\brs\.?\s*)?(\d{1,3}(?:\.\d)?)\s*(?:l|lakh|lakhs|lp|lpa)\b',
  re.IGNORECASE,
)
_MONTHLY_RE = re.compile(r'/\s*mo(?:nth)?\b|per\s+month', re.IGNORECASE)

_REMOTE_HINTS = {
  'remote_only': 'remote',
  'remote_okay': 'hybrid',
  'remote_not_okay': 'onsite',
}

'''Cutshort's structured bounds are annualized rupees, so only lakh conversion
is needed -- no monthly multiplier applies.
'''
_RUPEES_PER_LPA = 100000.0


def _strip_html(html: str) -> str:
  '''Convert an HTML fragment to single-spaced plain text.

  Args:
    html: Raw HTML fragment, possibly empty.

  Returns:
    Plain text.
  '''
  from bs4 import BeautifulSoup
  return ' '.join(BeautifulSoup(html or '', 'html.parser').get_text(' ').split())


def extract_next_data(html: str) -> Optional[dict]:
  '''Pull the Next.js data island out of a rendered page.

  Args:
    html: Full page HTML.

  Returns:
    Decoded island, or None when absent or unparseable.
  '''
  match = _NEXT_DATA_RE.search(html or '')
  if not match:
    return None
  try:
    return json.loads(match.group(1))
  except ValueError:
    return None


def extract_category_jobs(html: str) -> List[dict]:
  '''Find the embedded job array on a Cutshort category page.

  The island wraps a react-query cache, so the jobs sit under
  ``props.pageProps.dehydratedState.queries[].state.data.data.pageData.jobs``.
  Every query is walked because the array is not always the first entry.

  Args:
    html: Full page HTML.

  Returns:
    Job mappings, empty when the page rendered no embedded data.
  '''
  island = extract_next_data(html)
  if not island:
    return []
  try:
    queries = island['props']['pageProps']['dehydratedState']['queries']
  except (KeyError, TypeError):
    return []
  for query in queries if isinstance(queries, list) else []:
    try:
      page_data = query['state']['data']['data']['pageData']
    except (KeyError, TypeError):
      continue
    jobs = page_data.get('jobs')
    if isinstance(jobs, list) and jobs:
      return [job for job in jobs if isinstance(job, dict)]
  return []


def _salary_lpa(item: dict) -> str:
  '''Render a Cutshort salary string as INR lakhs-per-annum text.

  Cutshort posts annual figures as ``Rs 15L - Rs 20L / yr`` and monthly ones as
  ``Rs 15000 - Rs 18000 / mo``. Only the lakhs form is readable by
  :func:`parse_salary_lpa`. A monthly posting has no lakhs marker in its text,
  so its structured bounds are used instead -- and those bounds are already
  expressed in annualized rupees (15000/mo arrives as ``min=180000``), so they
  are divided by 100000 directly rather than multiplied by twelve.

  Args:
    item: One Cutshort job mapping.

  Returns:
    Salary text such as '15-20 lakhs per annum', or an empty string.
  '''
  text = str(item.get('salaryRangeText') or '').strip()
  if not text:
    return ''
  match = _LPA_RANGE_RE.search(text)
  if match and not _MONTHLY_RE.search(text):
    low, high = float(match.group(1)), float(match.group(2))
    # Guard against a reversed or nonsense bracket before trusting it.
    if high >= low:
      return f'{low:g}-{high:g} lakhs per annum'
    return f'{high:g}-{low:g} lakhs per annum'
  match = _LPA_SINGLE_RE.search(text)
  if match and not _MONTHLY_RE.search(text):
    return f'{float(match.group(1)):g} lakhs per annum'
  # Monthly postings: the structured bounds already arrive annualized in rupees.
  duration = str((item.get('salaryRange') or {}).get('duration') or '').upper()
  if duration == 'MONTH' or _MONTHLY_RE.search(text):
    low = (item.get('salaryRange') or {}).get('min')
    high = (item.get('salaryRange') or {}).get('max')
    try:
      low_lpa = float(low) / _RUPEES_PER_LPA
      high_lpa = float(high) / _RUPEES_PER_LPA
    except (TypeError, ValueError):
      return ''
    if low_lpa <= 0 and high_lpa <= 0:
      return ''
    if low_lpa > 200 or high_lpa > 400:
      # Implausible as an annual package; leave it for the text scan instead.
      return ''
    if low_lpa > 0 and high_lpa > 0 and high_lpa >= low_lpa:
      return f'{low_lpa:.1f}-{high_lpa:.1f} lakhs per annum'
    best = high_lpa or low_lpa
    return f'{best:.1f} lakhs per annum'
  return ''


def _location(item: dict) -> str:
  '''Build location text from Cutshort's split location fields.

  Args:
    item: One Cutshort job mapping.

  Returns:
    Location text such as 'Bengaluru; Remote only', possibly empty.
  '''
  parts: List[str] = []
  locations = item.get('locations')
  if isinstance(locations, list):
    parts.extend(str(part).strip() for part in locations if str(part).strip())
  elif isinstance(locations, str) and locations.strip():
    parts.append(locations.strip())
  text = str(item.get('locationsText') or '').strip()
  if text and text not in parts:
    parts.append(text)
  return '; '.join(dict.fromkeys(parts))


def _work_mode_hint(item: dict) -> str:
  '''Map Cutshort's remoteType onto a work-mode hint.

  Args:
    item: One Cutshort job mapping.

  Returns:
    'remote', 'hybrid', 'onsite', or an empty string when unspecified.
  '''
  return _REMOTE_HINTS.get(str(item.get('remoteType') or '').lower(), '')


def parse_cutshort_job(item: dict) -> Optional[RawJobRecord]:
  '''Convert one embedded Cutshort job into a raw record.

  Args:
    item: One job mapping from the Next.js island.

  Returns:
    A record, or None when the posting lacks a title or URL.
  '''
  title = str(item.get('headline') or '').strip()
  url = str(item.get('publicUrl') or '').strip()
  if not title or not url:
    return None
  company = (item.get('companyDetails') or {}).get('name') or ''
  exp_range = item.get('expRange') or {}
  exp_min = exp_range.get('min')
  exp_max = exp_range.get('max')
  return RawJobRecord(
    source_key=SOURCE_KEY,
    external_id=str(item.get('_id') or url),
    url=url,
    company_name=str(company).strip(),
    title=title,
    location_text=_location(item),
    description_text=_strip_html(str(item.get('sanitizedComment') or '')),
    employment_type_raw='; '.join(
      str(role) for role in (item.get('roleTypes') or []) if role
    ),
    work_mode_hint=_work_mode_hint(item),
    salary_raw=_salary_lpa(item),
    posted_at=None,
    extra={
      'skills': item.get('allSkills') or [],
      'experience_min_years': exp_min,
      'experience_max_years': exp_max,
      'salary_display': str(item.get('salaryRangeText') or ''),
      'remote_type': str(item.get('remoteType') or ''),
      'hiring_for_client': bool(item.get('hiringForClient')),
    },
  )


def parse_cutshort(html: str) -> List[RawJobRecord]:
  '''Parse a Cutshort category page into raw records.

  Args:
    html: Full page HTML.

  Returns:
    Raw job records, empty when the page carried no embedded data.
  '''
  records: List[RawJobRecord] = []
  for item in extract_category_jobs(html):
    record = parse_cutshort_job(item)
    if record is not None:
      records.append(record)
  return records


class CutshortAdapter(SourceAdapter):
  '''Category-page fetcher for Cutshort.

  Unlike the other aggregators this one is not a single feed. It walks a fixed
  slug list because Cutshort segments inventory by category and exposes no
  combined endpoint. ``?page=`` is not honored upstream, so each slug
  contributes at most one page of postings.
  '''

  def __init__(self, http: HttpClient, slugs: Optional[tuple] = None) -> None:
    '''Initialize the adapter.

    Args:
      http: Shared polite HTTP client.
      slugs: Optional override of the category slug list.
    '''
    super().__init__(http)
    self._slugs = slugs if slugs is not None else CATEGORY_SLUGS

  async def fetch(self, target: CompanyTarget, limit: int = 200) -> List[RawJobRecord]:
    '''Fetch Cutshort category pages and return the postings.

    Args:
      target: Only ``target.board_ref`` matters, to narrow to one company.
      limit: Safety cap on total returned records.

    Returns:
      Raw job records, de-duplicated across categories.

    Raises:
      AdapterError: When every category fails, meaning the board is unusable.
    '''
    collected: List[RawJobRecord] = []
    seen_ids: set = set()
    reached = 0
    failures = 0
    for slug in self._slugs:
      if len(collected) >= limit:
        break
      url = f'{BASE}/jobs/{slug}'
      try:
        html = await self._http.get_text(url, rpm=10)
      except AdapterError as exc:
        failures += 1
        logger.warning('cutshort category %s failed: %s', slug, exc)
        continue
      if not html:
        failures += 1
        logger.info('cutshort category %s blocked by robots or empty', slug)
        continue
      records = parse_cutshort(html)
      if not records:
        # Several category URLs render client-side only and embed nothing.
        logger.info('cutshort category %s returned no embedded jobs', slug)
        continue
      reached += 1
      for record in records:
        key = record.external_id or record.url
        if key in seen_ids:
          continue
        seen_ids.add(key)
        collected.append(record)
        if len(collected) >= limit:
          break
    if not reached:
      raise AdapterError(
        f'cutshort: no category yielded jobs ({failures} failed of {len(self._slugs)})',
        source=SOURCE_KEY,
      )
    if target.board_ref:
      needle = target.board_ref.lower()
      collected = [
        record for record in collected
        if needle in record.company_name.lower() or needle in record.url.lower()
      ]
    return collected[:limit]

  async def health(self, target: CompanyTarget) -> bool:
    '''Probe one category page for embedded job data.

    Args:
      target: Unused beyond matching the registry signature.

    Returns:
      True when a category page yields at least one posting.
    '''
    try:
      html = await self._http.get_text(f'{BASE}/jobs/{self._slugs[0]}', rpm=10)
    except AdapterError:
      return False
    return bool(parse_cutshort(html)) if html else False