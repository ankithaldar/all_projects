#!/usr/bin/env python
# -- coding: utf-8 --

'''Field normalization: URLs, cities, work modes, salaries, hashes.'''


from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Optional, Tuple

from bs4 import BeautifulSoup
from job_hunter.core.models import RawJobRecord, WorkMode

_TRACKING_PARAMS = re.compile(r'[?&](utm_[a-z]+|fbclid|gclid|ref|source)=[^&]*')
_CITY_MAP = {
  'bangalore': 'Bengaluru', 'bengaluru': 'Bengaluru', 'blr': 'Bengaluru',
  'mumbai': 'Mumbai', 'navi mumbai': 'Mumbai', 'thane': 'Mumbai',
  'gurgaon': 'Gurugram', 'gurugram': 'Gurugram',
  'noida': 'Noida', 'greater noida': 'Noida', 'delhi': 'Delhi NCR',
  'new delhi': 'Delhi NCR', 'ghaziabad': 'Delhi NCR', 'faridabad': 'Delhi NCR',
  'hyderabad': 'Hyderabad', 'chennai': 'Chennai', 'pune': 'Pune',
  'kolkata': 'Kolkata', 'ahmedabad': 'Ahmedabad', 'jaipur': 'Jaipur',
  'indore': 'Indore', 'kochi': 'Kochi', 'coimbatore': 'Coimbatore',
  'chandigarh': 'Chandigarh', 'remote': 'Remote',
}
_SALARY_RANGE_RE = re.compile(
  r'(\d{1,2}(?:\.\d)?)\s*(?:-|–|to)\s*(\d{1,2}(?:\.\d)?)\s*'
  r'(?:lpa|lakhs? per annum|lakhs?|lacs?|l\b)',
  re.IGNORECASE,
)
_SALARY_SINGLE_RE = re.compile(r'(\d{1,2}(?:\.\d)?)\s*(?:lpa|lakhs?|lacs?)\b', re.IGNORECASE)


def canonicalize_url(url: str) -> str:
  '''Strip tracking params and fragments from a posting URL.

  Args:
    url: Original URL.

  Returns:
    Canonical URL string.
  '''
  from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
  parsed = urlparse(url)
  kept = [
    (key, value)
    for key, value in parse_qsl(parsed.query, keep_blank_values=True)
    if not key.lower().startswith('utm_') and key.lower() not in (
      'fbclid', 'gclid', 'ref', 'source', 'trk',
    )
  ]
  return urlunparse(parsed._replace(
    query=urlencode(kept),
    fragment='',
  ))


def html_to_text(html: str) -> str:
  '''Convert HTML to compact plain text.

  Args:
    html: Raw HTML.

    Returns:
      Plain text.
  '''
  if '<' not in (html or ''):
    return ' '.join((html or '').split())
  soup = BeautifulSoup(html, 'html.parser')
  for tag in soup(['script', 'style', 'noscript']):
    tag.decompose()
  return ' '.join(soup.get_text(' ').split())


def content_hash(record: RawJobRecord, description_text: str) -> str:
  '''Stable hash of the semantic content of a posting.

  Args:
    record: Raw record.
    description_text: Cleaned description text.

  Returns:
    Hex digest.
  '''
  parts = [
    canonicalize_url(record.url),
    record.company_name.strip().lower(),
    record.title.strip().lower(),
    description_text[:600],
  ]
  return hashlib.sha256('\x1f'.join(parts).encode('utf-8')).hexdigest()


def detect_city(location_text: str, description_text: str = '') -> str:
  '''Map free location text to a canonical Indian city or Remote.

  Args:
    location_text: Location field.
    description_text: Description fallback.

  Returns:
    City name or empty string when unknown.
  '''
  haystack = f'{location_text} {description_text[:300]}'.lower()
  if 'remote' in location_text.lower() and not any(
    city in haystack for city in ('bangalore', 'bengaluru', 'mumbai', 'pune')
  ):
    return 'Remote'
  for variant, city in _CITY_MAP.items():
    if variant in haystack:
      return city
  return ''


def infer_work_mode(hint: str, location_text: str, text: str) -> str:
  '''Infer remote/hybrid/onsite from hints and text signals.

  Args:
    hint: Adapter-supplied workplace type.
    location_text: Location field.
    text: Title + description text.

  Returns:
    One of the WorkMode values.
  '''
  lowered_hint = (hint or '').lower()
  lowered_all = f'{location_text} {text}'.lower()
  if 'remote' in lowered_hint or 'remote' in location_text.lower():
    return WorkMode.REMOTE
  if 'hybrid' in lowered_hint or 'hybrid' in lowered_all[:400]:
    return WorkMode.HYBRID
  if any(token in lowered_all for token in ('work from home', 'fully remote')):
    return WorkMode.REMOTE
  if location_text.strip():
    return WorkMode.ONSITE
  return WorkMode.UNKNOWN


def normalize_employment(raw: str) -> str:
  '''Map employment-type wording to a canonical value.

  Args:
    raw: Adapter-provided employment text.

  Returns:
    full_time|part_time|contract|internship|other.
  '''
  lowered = (raw or '').lower()
  for token in ('intern',):
    if token in lowered:
      return 'internship'
  for token in ('contract', 'c2h', 'freelance'):
    if token in lowered:
      return 'contract'
  if 'part' in lowered:
    return 'part_time'
  if 'full' in lowered or not lowered:
    return 'full_time'
  return 'other'


def parse_salary_lpa(salary_raw: str, text: str = '') -> Tuple[Optional[float], Optional[float]]:
  '''Extract an INR LPA range from salary text or description.

  Args:
    salary_raw: Structured salary string if any.
    text: Fallback text to scan.

  Returns:
    (min_lpa, max_lpa); None components when unknown or implausible.
  '''
  for source in (salary_raw or '', (text or '')[:1500]):
    if not source:
      continue
    match = _SALARY_RANGE_RE.search(source)
    if match:
      low = float(match.group(1))
      high = float(match.group(2))
      if high >= low and 1 <= low <= 200 and high <= 200:
        return low, high
    match = _SALARY_SINGLE_RE.search(source)
    if match:
      value = float(match.group(1))
      if 1 <= value <= 200:
        return value * 0.95, value * 1.05
  return None, None


def parse_posted_at(value: object) -> Optional[str]:
  '''Normalize assorted posted-at formats to ISO 8601 UTC.

  Args:
    value: ISO string, epoch seconds/millis, or None.

  Returns:
    ISO timestamp string or None.
  '''
  if value in (None, ''):
    return None
  if isinstance(value, (int, float)):
    millis = float(value)
    if millis > 10_000_000_000:
      millis /= 1000.0
    try:
      return datetime.fromtimestamp(millis, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
      return None
  text = str(value).strip()
  if text.isdigit():
    return parse_posted_at(int(text))
  try:
    return datetime.fromisoformat(text.replace('Z', '+00:00')).astimezone(
      timezone.utc,
    ).isoformat()
  except ValueError:
    return None
