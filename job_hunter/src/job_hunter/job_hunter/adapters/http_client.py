#!/usr/bin/env python
# -- coding: utf-8 --

'''Shared async HTTP client with per-host politeness and robots checks.'''


from __future__ import annotations

import asyncio
import time
from typing import Dict, Optional
from urllib.parse import urlparse

import httpx
from job_hunter.core.errors import AdapterError

USER_AGENT = 'job-hunter-personal/0.1 (local personal job search; contact: self)'


class HostThrottle:
  '''Minimum-interval throttle between requests to one host.'''

  def __init__(self, rpm: int) -> None:
    '''Initialize the throttle.

    Args:
      rpm: Allowed requests per minute for this host.
    '''
    self._interval = 60.0 / max(rpm, 1)
    self._last = 0.0

  async def wait(self) -> None:
    '''Sleep until this host may be hit again.'''
    now = time.monotonic()
    delta = self._interval - (now - self._last)
    if delta > 0:
      await asyncio.sleep(delta)
    self._last = time.monotonic()


class RobotsCache:
  '''Robots.txt allowance cache keyed by origin.'''

  def __init__(self) -> None:
    '''Create an empty cache.'''
    self._entries: Dict[str, Optional[object]] = {}

  async def allowed(self, client: httpx.AsyncClient, url: str) -> bool:
    '''Return whether a URL is permitted by its robots.txt.

    Args:
      client: HTTP client used to fetch robots.txt.
      url: Target URL.

    Returns:
      True when explicitly allowed or when rules are unavailable.
    '''
    from urllib.robotparser import RobotFileParser
    parsed = urlparse(url)
    origin = f'{parsed.scheme}://{parsed.netloc}'
    if origin not in self._entries:
      parser = RobotFileParser()
      try:
        response = await client.get(f'{origin}/robots.txt', follow_redirects=True)
        if response.status_code == 200:
          parser.parse(response.text.splitlines())
          self._entries[origin] = parser
        else:
          self._entries[origin] = None
      except httpx.HTTPError:
        self._entries[origin] = None
    robot_parser = self._entries[origin]
    if robot_parser is None:
      return True
    try:
      return robot_parser.can_fetch(USER_AGENT, url)
    except Exception:
      return True


class HttpClient:
  '''Polite async HTTP client shared by all adapters.'''

  def __init__(self, default_rpm: int = 30) -> None:
    '''Initialize the client.

    Args:
      default_rpm: Per-host request cap when not overridden.
    '''
    self._client = httpx.AsyncClient(
      headers={'User-Agent': USER_AGENT, 'Accept': '*/*'},
      timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
      follow_redirects=True,
    )
    self._default_rpm = default_rpm
    self._throttles: Dict[str, HostThrottle] = {}
    self._robots = RobotsCache()

  def throttle_for(self, url: str, rpm: Optional[int] = None) -> HostThrottle:
    '''Return the throttle for a URL's host.

    Args:
      url: Target URL.
      rpm: Optional override of the default cap.

    Returns:
      Host-scoped throttle.
    '''
    host = urlparse(url).netloc
    if host not in self._throttles:
      self._throttles[host] = HostThrottle(rpm or self._default_rpm)
    return self._throttles[host]

  async def get_json(
    self,
    url: str,
    params: Optional[dict] = None,
    rpm: Optional[int] = None,
    method: str = 'GET',
  ) -> object:
    '''Fetch JSON with throttling and one retry.

    Args:
      url: Endpoint URL.
      params: Query parameters.
      rpm: Per-host RPM override.
      method: HTTP method (Ashby accepts POST; GET used by default).

    Returns:
      Decoded JSON value.

    Raises:
      AdapterError: On transport failure or non-JSON response.
    '''
    await self.throttle_for(url, rpm).wait()
    for attempt in range(2):
      try:
        if method.upper() == 'POST':
          response = await self._client.post(url, json=params or {})
        else:
          response = await self._client.get(url, params=params)
        if response.status_code in (429, 500, 502, 503, 504):
          await asyncio.sleep(1.5 * (attempt + 1))
          continue
        return response.json()
      except (httpx.HTTPError, ValueError) as exc:
        if attempt == 1:
          raise AdapterError(f'get_json failed {url}: {exc}') from exc
        await asyncio.sleep(1.5)
    raise AdapterError(f'get_json exhausted retries: {url}')

  async def get_text(self, url: str, rpm: Optional[int] = None) -> str:
    '''Fetch a page body respecting robots.txt.

    Args:
      url: Page URL.
      rpm: Per-host RPM override.

    Returns:
      Response text (empty when blocked).

    Raises:
      AdapterError: On transport failure.
    '''
    await self.throttle_for(url, rpm).wait()
    if not await self._robots.allowed(self._client, url):
      return ''
    try:
      response = await self._client.get(url)
      return response.text or ''
    except httpx.HTTPError as exc:
      raise AdapterError(f'get_text failed {url}: {exc}') from exc

  async def close(self) -> None:
    '''Release the underlying connection pool.'''
    await self._client.aclose()
