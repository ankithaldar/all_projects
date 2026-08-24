#!/usr/bin/env python
# -- coding: utf-8 --

'''Careers-page ATS fingerprinting: domain to (provider, board_ref).'''

from __future__ import annotations

import re
from typing import Optional, Tuple

from bs4 import BeautifulSoup
from job_hunter.adapters.http_client import HttpClient

FINGERPRINTS = [
  ('greenhouse', re.compile(r'job-boards\.greenhouse\.io/([A-Za-z0-9_-]+)|boards\.greenhouse\.io/(?:embed_job_board\?for=|greenhouse/)([A-Za-z0-9_-]+)')),
  ('lever', re.compile(r'jobs\.lever\.co/([A-Za-z0-9_-]+)|jobs\.eu\.lever\.co/([A-Za-z0-9_-]+)')),
  ('ashby', re.compile(r'jobs\.ashbyhq\.com/([A-Za-z0-9_.-]+)|jobs\.ashbyhq\.com/api/job-board/([A-Za-z0-9_.-]+)')),
  ('workable', re.compile(r'apply\.workable\.com/([A-Za-z0-9_-]+)|(?:[a-z0-9-]+\.)?workable\.com/j\?lvn=')),
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
      ref = next((group for group in match.groups() if group), '')
      ref = ref.split('?')[0].strip('/')
      if ref:
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

  async def detect(self, domain: str, max_probes: int = 3) -> Optional[Tuple[str, str, str]]:
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
    return None
