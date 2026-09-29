#!/usr/bin/env python
# -- coding: utf-8 --

'''Robots-permitted web search used for company and job discovery.

The search backend is the DuckDuckGo HTML endpoint, which its robots.txt
declares as fully crawlable (``Allow: /``). The main duckduckgo.com host is
disallowed outright, so it is never used. Results are cached on disk with a
TTL to stay polite under repeated scouting runs.
'''


from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from urllib.parse import unquote

import httpx
from bs4 import BeautifulSoup
from job_hunter.adapters.http_client import USER_AGENT

SEARCH_ENDPOINT = 'https://html.duckduckgo.com/html/'

'''Result hosts that are never useful as a company website.'''
_NOISE_HOSTS = (
  'duckduckgo.com', 'bing.com', 'yahoo.com', 'ecosia.org', 'startpage.com',
  'brave.com', 'mojeek.com', 'qwant.com', 'yandex.com', 'google.com',
)

_TITLE_SUFFIXES = (
  ' | LinkedIn', '- LinkedIn', ' | linkedin', ' on LinkedIn', ' LinkedIn',
  ' | Indeed', ' | Glassdoor', ' | Naukri', ' | Wellfound', ' | CutShort',
  ' | AmbitionBox', ' | Foundit', ' | Jobberman', ' | Shine', ' | SimplyHired',
)

_SLUG_AT_RE = re.compile(r'-at-(?P<slug>[a-z0-9][a-z0-9-]{1,60}?)-\d{5,}\s*$', re.IGNORECASE)
_TITLE_AT_RE = re.compile(r'\s+at\s+(?P<company>[^-]{2,60}?)\s*(?:\||\u2013|\u2014|-{1,2}\s+LinkedIn|$)', re.IGNORECASE)
_URL_RE = re.compile(r'^[a-z0-9][a-z0-9.-]*\.[a-z]{2,}$', re.IGNORECASE)


class SearchResult(dict):
  '''One search hit: ``title``, ``href``, and ``snippet`` keys.'''


class WebSearchIndex:
  '''Rate-limited, cached search client over the DuckDuckGo HTML endpoint.

  The class implements a narrow interface -- ``search(query, limit)`` and
  ``urls(query, limit)`` -- so callers never depend on the backend.
  '''

  def __init__(
    self,
    cache_path: Optional[Path] = None,
    ttl_seconds: int = 86400,
    min_interval: float = 3.0,
    rpm: int = 15,
    timeout: float = 25.0,
  ) -> None:
    '''Initialize the search client.

    Args:
      cache_path: Optional JSON file used to persist results between runs.
      ttl_seconds: Age after which a cached result is refetched.
      min_interval: Minimum seconds between outbound search requests.
      rpm: Requests-per-minute cap used to derive the throttle interval.
      timeout: Per-request timeout in seconds.
    '''
    self._cache_path = Path(cache_path) if cache_path else None
    self._ttl = ttl_seconds
    self._min_interval = max(min_interval, 60.0 / max(rpm, 1))
    self._timeout = timeout
    self._last_request = 0.0
    self._cache: Dict[str, Dict[str, object]] = {}
    self._client = httpx.AsyncClient(
      headers={
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml',
      },
      timeout=httpx.Timeout(connect=10.0, read=timeout, write=10.0, pool=10.0),
      follow_redirects=True,
    )
    if self._cache_path and self._cache_path.exists():
      try:
        import json
        self._cache = json.loads(self._cache_path.read_text(encoding='utf-8'))
      except Exception:  # noqa: BLE001 - a corrupt cache is simply ignored
        self._cache = {}

  def _load_cached(self, query: str) -> Optional[List[SearchResult]]:
    '''Return cached results for a query when still fresh.

    Args:
      query: Search query text.

    Returns:
      Cached result list, or None when absent or stale.
    '''
    entry = self._cache.get(query)
    if not isinstance(entry, dict):
      return None
    stored = entry.get('results')
    ts = float(entry.get('ts') or 0.0)
    if not isinstance(stored, list) or (time.time() - ts) > self._ttl:
      return None
    return [SearchResult(item) for item in stored if isinstance(item, dict)]

  def _store_cached(self, query: str, results: Sequence[SearchResult]) -> None:
    '''Persist results for a query to the in-memory and on-disk caches.

    Args:
      query: Search query text.
      results: Results to cache.
    '''
    self._cache[query] = {'ts': time.time(), 'results': list(results)}
    if not self._cache_path:
      return
    try:
      import json
      self._cache_path.parent.mkdir(parents=True, exist_ok=True)
      self._cache_path.write_text(json.dumps(self._cache), encoding='utf-8')
    except Exception:  # noqa: BLE001 - caching must never break a run
      pass

  async def _throttle(self) -> None:
    '''Sleep so that consecutive requests respect the configured interval.'''
    delta = self._min_interval - (time.monotonic() - self._last_request)
    if delta > 0:
      import asyncio
      await asyncio.sleep(delta)
    self._last_request = time.monotonic()

  @staticmethod
  def _decode_href(href: str) -> str:
    '''Unwrap a DuckDuckGo redirect into its destination URL.

    Args:
      href: Raw anchor href, possibly a ``/l/?uddg=`` redirect.

    Returns:
      The absolute destination URL, or the input when not a redirect.
    '''
    match = re.search(r'uddg=([^&]+)', href)
    if match:
      return unquote(match.group(1))
    return href

  @staticmethod
  def _parse(html: str, limit: int) -> List[SearchResult]:
    '''Extract result items from a DuckDuckGo HTML response.

    Args:
      html: Response body.
      limit: Maximum number of results to return.

    Returns:
      Result mappings ordered as returned by the engine.
    '''
    soup = BeautifulSoup(html or '', 'html.parser')
    results: List[SearchResult] = []
    for anchor in soup.select('a.result__a'):
      if len(results) >= limit:
        break
      href = WebSearchIndex._decode_href(str(anchor.get('href') or ''))
      if not href.startswith('http'):
        continue
      snippet_el = anchor.find_parent(class_=re.compile(r'^result'))
      snippet = ''
      if snippet_el:
        found = snippet_el.find(class_=re.compile(r'result__snippet|result-snippet'))
        if found:
          snippet = found.get_text(' ', strip=True)
      results.append(SearchResult({
        'title': anchor.get_text(' ', strip=True),
        'href': href,
        'snippet': snippet,
      }))
    return results

  async def search(self, query: str, limit: int = 15) -> List[SearchResult]:
    '''Run a search query, returning cached results when possible.

    Args:
      query: Search query text.
      limit: Maximum number of results.

    Returns:
      Result mappings; empty when the backend answers with a challenge.
    '''
    cached = self._load_cached(query)
    if cached is not None:
      return cached[:limit]
    await self._throttle()
    try:
      response = await self._client.get(SEARCH_ENDPOINT, params={'q': query})
    except Exception:  # noqa: BLE001 - search failures degrade gracefully
      return []
    if response.status_code != 200:
      return []
    results = self._parse(response.text, max(1, min(limit, 30)))
    if results:
      self._store_cached(query, results)
    return results[:limit]

  async def urls(self, query: str, limit: int = 15) -> List[str]:
    '''Return only the result URLs for a query.

    Args:
      query: Search query text.
      limit: Maximum number of URLs.

    Returns:
      Destination URLs in engine order.
    '''
    return [str(item.get('href')) for item in await self.search(query, limit) if item.get('href')]

  async def close(self) -> None:
    '''Release the underlying connection pool.'''
    await self._client.aclose()


def host_of(url: str) -> str:
  '''Extract the lowercase host from a URL.

  Args:
    url: Absolute URL or bare host.

  Returns:
    Host without scheme, port, or leading ``www.``; empty on failure.
  '''
  candidate = url.strip()
  if '//' not in candidate:
    candidate = f'//{candidate}'
  match = re.match(r'^[a-z]+://([^/?#]+)', candidate, re.IGNORECASE)
  if not match:
    return ''
  host = match.group(1).lower().split(':')[0]
  return host[4:] if host.startswith('www.') else host


def is_search_noise(url: str) -> bool:
  '''Report whether a result URL comes from a search engine or social site.

  Args:
    url: Result URL.

  Returns:
    True when the host is a search engine or a social/job aggregator.
  '''
  host = host_of(url)
  if not host:
    return True
  return any(host == item or host.endswith(f'.{item}') for item in _NOISE_HOSTS)


def clean_result_title(title: str) -> str:
  '''Strip search-engine and job-board decorations from a result title.

  Args:
    title: Raw result title.

  Returns:
    Title without trailing site names.
  '''
  cleaned = ' '.join((title or '').split())
  for suffix in _TITLE_SUFFIXES:
    if cleaned.lower().endswith(suffix.lower()):
      cleaned = cleaned[: -len(suffix)].strip()
  return cleaned.strip(' -\u2013\u2014|,')


def company_from_linkedin_url(url: str) -> str:
  '''Recover a company slug from a LinkedIn job URL.

  LinkedIn slugs follow the pattern ``<role>-at-<company-slug>-<jobid>``.
  The slug is converted into a spaced, title-cased display name.

  Args:
    url: LinkedIn job URL.

  Returns:
    A human-readable company name, or an empty string when the URL carries
    no company slug.
  '''
  if 'linkedin.com' not in (url or ''):
    return ''
  path = re.sub(r'^[a-z]+://[^/]+', '', url.strip(), flags=re.IGNORECASE)
  match = _SLUG_AT_RE.search(path.rstrip('/'))
  if not match:
    return ''
  slug = match.group('slug')
  return re.sub(r'[-_]+', ' ', slug).strip().title()


def company_from_result_title(title: str) -> str:
  '''Recover a company name from a ``<Role> at <Company>`` result title.

  Args:
    title: Search result title.

  Returns:
    The company name, or an empty string when the pattern does not match.
  '''
  cleaned = clean_result_title(title)
  match = _TITLE_AT_RE.search(cleaned)
  if not match:
    return ''
  return match.group('company').strip(' -\u2013\u2014|,')


def looks_like_domain(value: str) -> bool:
  '''Report whether a string is a bare hostname.

  Args:
    value: Candidate string.

  Returns:
    True when the value matches a simple domain shape.
  '''
  return bool(_URL_RE.match((value or '').strip()))
