#!/usr/bin/env python
# -- coding: utf-8 --

'''Careers-page ATS fingerprinting: domain to (provider, board_ref).'''

from __future__ import annotations

import re
from typing import Optional, Tuple

import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from job_hunter.adapters.http_client import HttpClient

warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)

FINGERPRINTS = [
  ('greenhouse', re.compile(r'job-boards\.greenhouse\.io/([A-Za-z0-9_-]+)|boards\.greenhouse\.io/(?:embed_job_board\?for=|greenhouse/)([A-Za-z0-9_-]+)')),
  ('lever', re.compile(r'jobs\.lever\.co/([A-Za-z0-9_-]+)|jobs\.eu\.lever\.co/([A-Za-z0-9_-]+)')),
  ('ashby', re.compile(r'jobs\.ashbyhq\.com/([A-Za-z0-9_.-]+)|jobs\.ashbyhq\.com/api/job-board/([A-Za-z0-9_.-]+)')),
  ('workable', re.compile(r'apply\.workable\.com/([A-Za-z0-9_-]{2,})')),
  ('workday', re.compile(r'([a-z0-9-]+\.wd\d+\.myworkdayjobs\.com)(?:/([A-Za-z0-9_-]+))?')),
  ('smartrecruiters', re.compile(r'careers\.smartrecruiters\.com/([A-Za-z0-9_-]+)|jobs\.smartrecruiters\.com/([A-Za-z0-9_-]+)')),
  ('recruitee', re.compile(r'([A-Za-z0-9_-]+)\.recruitee\.com')),
  ('personio', re.compile(r'([A-Za-z0-9_-]+)\.jobs\.personio\.(?:de|com)')),
]

CAREERS_PATHS = ['/careers', '/careers/', '/jobs', '/company/careers', '/about/careers']


def extract_fingerprint(html: str) -> Optional[Tuple[str, str]]:
  '''Scan page HTML for known ATS board URLs.

  Args:
    html: Page source.

  Returns:
    (provider, board_ref) for the first match, else None.
  '''
  if not html:
    return None
  hrefs = ' '.join(a.get('href', '') or '' for a in BeautifulSoup(html, 'html.parser').find_all('a'))
  haystack = f'{hrefs} {html[:40000]}'
  for provider, pattern in FINGERPRINTS:
    match = pattern.search(haystack)
    if match:
      groups = [g for g in match.groups() if g]
      ref = '/'.join(groups) if provider == 'workday' else (groups[0] if groups else '')
      ref = ref.split('?')[0].strip('/')
      if ref and len(ref) >= 2 and ref.lower() not in {'j', 'jobs', 'careers'}:
        return provider, ref
  return None


class CareerPageDetector:
  '''Probe a company's site to discover its ATS board.'''

  def __init__(self, http: HttpClient) -> None:
    '''Initialize the detector.

    Args:
      http: Shared HTTP client.
    '''
    self._http = http

  async def detect(self, domain: str, max_probes: int = 3, name: str = '') -> Optional[Tuple[str, str, str]]:
    '''Attempt fingerprinting via common careers paths.

    Args:
      domain: Registrable company domain.
      max_probes: Maximum pages fetched.

    Returns:
      (provider, board_ref, careers_url) or None.
    '''
    domain = domain.strip().lower().replace('http://', '').replace('https://', '').strip('/')
    probes = [f'https://www.{domain}{path}' for path in CAREERS_PATHS]
    probes.insert(1, f'https://{domain}/careers')
    seen: set = set()
    fetched = 0
    for url in probes:
      if url in seen or fetched >= max_probes:
        continue
      seen.add(url)
      try:
        html = await self._http.get_text(url, rpm=20)
      except Exception:
        continue
      fetched += 1
      found = extract_fingerprint(html)
      if found:
        return found[0], found[1], url
    return await self._guess_slugs(domain, name)

  async def _guess_slugs(self, domain: str, name: str) -> Optional[Tuple[str, str, str]]:
    '''Probe likely greenhouse/lever board slugs from domain and name.

    Args:
      domain: Registrable domain.
      name: Display name.

    Returns:
      (provider, ref, url) when a live board answers.
    '''
    core = domain.split('.')[0].lower().replace('-', '')
    guesses = {core}
    cleaned = ''.join(ch for ch in name.lower() if ch.isalnum())
    if cleaned:
      guesses.add(cleaned)
    for guess in list(guesses)[:2]:
      gh = f'https://job-boards.greenhouse.io/{guess}'
      try:
        text = await self._http.get_text(gh, rpm=20)
        if text and 'greenhouse' in text.lower():
          return 'greenhouse', guess, gh
      except Exception:
        pass
      lv = f'https://jobs.lever.co/{guess}'
      try:
        text = await self._http.get_text(lv, rpm=20)
        if text and 'lever.co' in text.lower():
          return 'lever', guess, lv
      except Exception:
        pass
    return None
