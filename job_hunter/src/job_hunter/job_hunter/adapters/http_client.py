#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Shared async HTTP client with per-host politeness and robots checks.'''

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import math
import random
import socket
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

from job_hunter.core.errors import AdapterError

logger = logging.getLogger(__name__)

USER_AGENT = (
  'job-hunter-personal/0.1 (local personal job search; contact: self)'
)

MAX_RESPONSE_BYTES = 5_000_000
'''Hard cap on any single fetched body; larger responses are rejected.'''

MAX_REDIRECTS = 5
'''Redirect hops followed before giving up (each hop is SSRF-validated).'''

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

_MAX_RETRY_AFTER = 30.0
'''Retry-After seconds honoured on 429; larger values are capped.'''


class SSRFBlocked(ValueError):
  '''Raised when a URL fails public-https validation (fail closed).'''


def _is_blocked_ip(ip_str: str) -> bool:
  '''Report whether an IP literal is non-public.

  Args:
    ip_str: IP address text.

  Returns:
    True when the address is private, loopback, link-local, reserved,
    multicast, or unspecified.
  '''
  try:
    ip = ipaddress.ip_address(ip_str)
  except ValueError:
    return True
  if isinstance(ip, ipaddress.IPv6Address):
    mapped = ip.ipv4_mapped
    if mapped is not None and _is_blocked_ip(str(mapped)):
      return True
  return bool(
    ip.is_private
    or ip.is_loopback
    or ip.is_link_local
    or ip.is_reserved
    or ip.is_multicast
    or ip.is_unspecified
  )


_NUMERIC_HOST_CHARS = frozenset('0123456789abcdefABCDEFxX.')


def _canonical_numeric_host(host: str) -> Optional[str]:
  '''Decode an obfuscated numeric host to a dotted IP when possible.

  Args:
    host: Hostname taken from a URL.

  Returns:
    Dotted-quad IP text, or None when the host is not numeric or cannot
    be decoded by :func:`socket.inet_aton`.
  '''
  if not host or any(char not in _NUMERIC_HOST_CHARS for char in host):
    return None
  try:
    return socket.inet_ntoa(socket.inet_aton(host))
  except (OSError, UnicodeEncodeError):
    return None


def _extract_ips(resolved: object) -> list:
  '''Normalize getaddrinfo-style tuples (or plain IP strings) to IP texts.

  Args:
    resolved: Resolver return value.

  Returns:
    List of IP literal strings.
  '''
  ips: list = []
  for entry in resolved or []:
    if isinstance(entry, str):
      ips.append(entry)
    elif isinstance(entry, (tuple, list)) and len(entry) >= 5:
      sockaddr = entry[4]
      if isinstance(sockaddr, (tuple, list)) and sockaddr:
        ips.append(str(sockaddr[0]))
  return ips


def validate_public_url(url: str, resolver: Optional[Callable] = None) -> str:
  '''Validate that a URL is public https and safe to fetch (SSRF guard).

  Rejects non-https schemes, missing hostnames, userinfo (``@`` in the
  netloc), non-public literal IPs, and hostnames that resolve to any
  non-public IP. Resolution failure fails closed.

  Args:
    url: Candidate URL.
    resolver: Optional getaddrinfo-compatible callable (tests inject a
      fake); defaults to :func:`socket.getaddrinfo`.

  Returns:
    The URL unchanged.

  Raises:
    SSRFBlocked: When any check fails.
  '''
  if not isinstance(url, str) or not url.strip():
    raise SSRFBlocked(f'empty URL: {url!r}')
  parsed = urlparse(url)
  if parsed.scheme.lower() != 'https':
    raise SSRFBlocked(f'scheme must be https: {url!r}')
  if '@' in parsed.netloc or parsed.username or parsed.password:
    raise SSRFBlocked(f'userinfo not allowed: {url!r}')
  host = (parsed.hostname or '').strip().rstrip('.')
  if not host:
    raise SSRFBlocked(f'missing hostname: {url!r}')
  try:
    port = parsed.port or 443
  except ValueError:
    raise SSRFBlocked(f'invalid port: {url!r}') from None
  # Literal IP hosts need no DNS.
  try:
    ipaddress.ip_address(host)
  except ValueError:
    pass
  else:
    if _is_blocked_ip(host):
      raise SSRFBlocked(f'non-public IP host: {url!r}')
    return url
  # Obfuscated numeric hosts (0x7f000001, 2130706433, 0177.0.0.1, 127.1)
  # decode via inet_aton and are checked in canonical form; a public
  # resolver answer must never rescue them.
  canonical = _canonical_numeric_host(host)
  if canonical is not None:
    if _is_blocked_ip(canonical):
      raise SSRFBlocked(f'non-public IP host: {url!r}')
    return url
  resolve = resolver or socket.getaddrinfo
  try:
    resolved = resolve(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
  except Exception as exc:
    logger.warning('SSRF validation: DNS failed for %s: %s', host, exc)
    raise SSRFBlocked(f'DNS resolution failed for {host}: {exc}') from exc
  ips = _extract_ips(resolved)
  if not ips:
    logger.warning('SSRF validation: no addresses for %s', host)
    raise SSRFBlocked(f'DNS resolution returned no addresses for {host}')
  for ip_str in ips:
    if _is_blocked_ip(ip_str):
      logger.warning(
        'SSRF validation: %s resolves to non-public %s', host, ip_str
      )
      raise SSRFBlocked(f'{host} resolves to non-public address {ip_str}')
  return url


async def read_capped(response: object, limit: Optional[int] = None) -> bytes:
  '''Stream a response body, enforcing a hard size cap.

  Args:
    response: Streaming httpx response (``aiter_bytes`` source).
    limit: Byte cap; defaults to :data:`MAX_RESPONSE_BYTES`.

  Returns:
    The full body bytes.

  Raises:
    AdapterError: When the body exceeds the cap.
  '''
  cap = MAX_RESPONSE_BYTES if limit is None else limit
  chunks: list = []
  total = 0
  async for chunk in response.aiter_bytes():  # type: ignore[attr-defined]
    total += len(chunk)
    if total > cap:
      raise AdapterError(f'response exceeds {cap} bytes')
    chunks.append(chunk)
  return b''.join(chunks)


@dataclass
class _FetchResult:
  '''One completed fetch after manual redirect hops.

  Attributes:
    status_code: Final HTTP status.
    headers: Final response headers (Retry-After source).
    body: Capped response body bytes.
    encoding: Response charset when available.
    url: Final URL after redirects.
  '''

  status_code: int
  headers: object
  body: bytes
  encoding: object
  url: str


def _retry_delay(headers: object, attempt: int) -> float:
  '''Compute a retry sleep honouring Retry-After plus jitter.

  Args:
    headers: Response headers carrying optional Retry-After seconds.
    attempt: Zero-based retry attempt.

  Returns:
    Sleep seconds: Retry-After (capped at 30) when present and parsable,
    else the legacy linear backoff, plus ``random.uniform(0, 0.5)``.
  '''
  base: Optional[float] = None
  raw = None
  try:
    raw = headers.get('Retry-After')  # type: ignore[union-attr]
  except Exception:  # noqa: BLE001 - headers are mapping-like or empty
    raw = None
  if raw is not None:
    try:
      base = float(str(raw).strip())
    except (TypeError, ValueError):
      base = None
    if base is not None and not math.isfinite(base):
      base = None
  if base is None:
    base = 1.5 * (attempt + 1)
  return min(max(base, 0.0), _MAX_RETRY_AFTER) + random.uniform(0, 0.5)


async def _fetch_with_redirects(
  client: httpx.AsyncClient,
  method: str,
  url: str,
  params: Optional[dict] = None,
  json_data: Optional[dict] = None,
  resolver: Optional[Callable] = None,
) -> _FetchResult:
  '''Fetch one URL following redirects with a hop-by-hop SSRF check.

  Each Location (relative ones resolved against the current URL) is
  validated with :func:`validate_public_url` before it is requested, so
  a public URL can never redirect the client to a private address.

  Args:
    client: httpx client used for every hop (``follow_redirects`` off).
    method: HTTP method, GET or POST.
    url: First URL to request.
    params: Query params sent on the first hop (GET only).
    json_data: JSON body sent on the first hop (POST only).
    resolver: Optional DNS resolver forwarded to URL validation.

  Returns:
    Final status, headers, capped body, charset and URL.

  Raises:
    SSRFBlocked: When any hop target fails validation.
    AdapterError: On redirect overflow or oversize body.
  '''
  validate_public_url(url, resolver=resolver)
  current = url
  verb = method.upper()
  for hop in range(MAX_REDIRECTS + 1):
    kwargs: Dict[str, object] = {}
    if hop == 0:
      if verb == 'GET' and params is not None:
        kwargs['params'] = params
      if verb == 'POST' and json_data is not None:
        kwargs['json'] = json_data
    async with client.stream(verb, current, **kwargs) as response:
      status = response.status_code
      location = response.headers.get('Location')
      if status in _REDIRECT_STATUSES and location:
        if hop >= MAX_REDIRECTS:
          raise AdapterError(f'redirect limit exceeded: {url}')
        next_url = urljoin(current, location)
        validate_public_url(next_url, resolver=resolver)
        if status == 303:
          verb = 'GET'
        current = next_url
        continue
      headers = response.headers
      body = await read_capped(response)
      encoding = getattr(response, 'encoding', None)
    return _FetchResult(
      status_code=status, headers=headers, body=body,
      encoding=encoding, url=current,
    )
  raise AdapterError(f'redirect limit exceeded: {url}')


def _decode_body(response: object, body: bytes) -> str:
  '''Decode streamed bytes using the response charset when available.

  Args:
    response: Streaming response carrying an optional ``encoding``.
    body: Raw body bytes.

  Returns:
    Decoded text.
  '''
  encoding = getattr(response, 'encoding', None) or 'utf-8'
  try:
    return body.decode(encoding, errors='replace')
  except (LookupError, ValueError):
    return body.decode('utf-8', errors='replace')


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
    '''Sleep until this host may be hit again.

    Assumes single-loop ownership: an HttpClient (and thus its
    throttles) is used from one event loop at a time, so ``_last``
    needs no asyncio.Lock (an asyncio lock would also bind to the
    first awaiting loop and break legitimate cross-loop reuse in
    tests that drive the client via repeated ``asyncio.run``).
    '''
    now = time.monotonic()
    delta = self._interval - (now - self._last)
    if delta > 0:
      # Reserve the slot before sleeping so concurrent callers space
      # out by one interval each instead of waking together.
      self._last = now + delta
      await asyncio.sleep(delta)
    else:
      self._last = now


class RobotsCache:
  '''Robots.txt allowance cache keyed by origin.'''

  def __init__(self, resolver: Optional[Callable] = None) -> None:
    '''Create an empty cache.

    Args:
      resolver: Optional DNS resolver forwarded to URL validation.
    '''
    self._entries: Dict[str, Optional[object]] = {}
    self._resolver = resolver

  async def allowed(self, client: httpx.AsyncClient, url: str) -> bool:
    '''Return whether a URL is permitted by its robots.txt.

    Fail-open (True, with a warning) when robots.txt is missing or
    unreachable; fail-closed (False) when the robots URL itself is
    SSRF-blocked.

    Args:
      client: HTTP client used to fetch robots.txt.
      url: Target URL.

    Returns:
      True when explicitly allowed or when rules are unavailable.
    '''
    parsed = urlparse(url)
    origin = f'{parsed.scheme}://{parsed.netloc}'
    if origin not in self._entries:
      parser = RobotFileParser()
      robots_url = f'{origin}/robots.txt'
      try:
        validate_public_url(robots_url, resolver=self._resolver)
      except SSRFBlocked as exc:
        logger.warning('robots.txt SSRF-blocked for %s: %s', origin, exc)
        return False
      try:
        fetched = await _fetch_with_redirects(
          client, 'GET', robots_url, resolver=self._resolver
        )
        if fetched.status_code == 200:
          text = _decode_body(fetched, fetched.body)
          parser.parse(text.splitlines())
          self._entries[origin] = parser
        else:
          logger.warning(
            'robots.txt unavailable (HTTP %s) for %s; failing open',
            fetched.status_code,
            origin,
          )
          self._entries[origin] = None
      except SSRFBlocked as exc:
        logger.warning('robots.txt SSRF-blocked for %s: %s', origin, exc)
        return False
      except (httpx.HTTPError, AdapterError) as exc:
        logger.warning(
          'robots.txt fetch failed for %s: %s; failing open',
          origin, exc,
        )
        self._entries[origin] = None
    robot_parser = self._entries[origin]
    if robot_parser is None:
      return True
    try:
      return robot_parser.can_fetch(USER_AGENT, url)
    except Exception:  # noqa: BLE001 - parser failure fails open
      return True


class HttpClient:
  '''Polite async HTTP client shared by all adapters.'''

  def __init__(
    self,
    default_rpm: int = 30,
    resolver: Optional[Callable] = None,
    transport: Optional[object] = None,
  ) -> None:
    '''Initialize the client.

    Args:
      default_rpm: Per-host request cap when not overridden.
      resolver: Optional DNS resolver forwarded to URL validation
        (tests inject a fake).
      transport: Optional httpx transport (tests inject a mock).
    '''
    self._client = httpx.AsyncClient(
      headers={'User-Agent': USER_AGENT, 'Accept': '*/*'},
      timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
      follow_redirects=False,
      transport=transport,
    )
    self._default_rpm = default_rpm
    self._resolver = resolver
    self._throttles: Dict[tuple, HostThrottle] = {}
    self._throttle_lock = threading.Lock()
    self._robots = RobotsCache(resolver=resolver)

  def throttle_for(self, url: str, rpm: Optional[int] = None) -> HostThrottle:
    '''Return the throttle for a URL's host at a given rpm.

    Keyed by ``(host, rpm)`` (D15): each (host, rpm) pair pins one
    HostThrottle, so a later call with a different rpm for the same
    host gets its own throttle instead of silently inheriting the
    first-seen rpm. A threading lock guards throttle creation.

    Args:
      url: Target URL.
      rpm: Optional override of the default cap.

    Returns:
      Host-scoped throttle for the (host, rpm) pair.
    '''
    parsed = urlparse(url)
    if parsed.hostname:
      host = (parsed.hostname or '').lower().rstrip('.')
    else:
      # Bare host[:port][/path] without a scheme.
      host = str(url).split('/')[0].split(':')[0].lower().rstrip('.')
    key = (host, rpm or self._default_rpm)
    with self._throttle_lock:
      if key not in self._throttles:
        self._throttles[key] = HostThrottle(key[1])
      return self._throttles[key]

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
      SSRFBlocked: When the URL (or its final redirect target) is not
        public https.
      AdapterError: On transport failure, oversize body, or non-JSON
        response.
    '''
    validate_public_url(url, resolver=self._resolver)
    await self.throttle_for(url, rpm).wait()
    verb = method.upper()
    for attempt in range(2):
      try:
        fetched = await _fetch_with_redirects(
          self._client,
          verb,
          url,
          params=None if verb == 'POST' else params,
          json_data=(params or {}) if verb == 'POST' else None,
          resolver=self._resolver,
        )
        if fetched.status_code in (429, 500, 502, 503, 504):
          if attempt == 1:
            raise AdapterError(
              f'get_json retryable status {fetched.status_code}: {url}'
            )
          await asyncio.sleep(_retry_delay(fetched.headers, attempt))
          continue
        return json.loads(fetched.body)
      except (SSRFBlocked, AdapterError):
        raise
      except (httpx.HTTPError, ValueError) as exc:
        if attempt == 1:
          raise AdapterError(f'get_json failed {url}: {exc}') from exc
        await asyncio.sleep(_retry_delay({}, attempt))
    raise AdapterError(f'get_json exhausted retries: {url}')

  async def get_text(self, url: str, rpm: Optional[int] = None) -> str:
    '''Fetch a page body respecting robots.txt.

    Args:
      url: Page URL.
      rpm: Per-host RPM override.

    Returns:
      Response text (empty when blocked).

    Raises:
      SSRFBlocked: When the URL (or its final redirect target) is not
        public https.
      AdapterError: On transport failure or oversize body.
    '''
    validate_public_url(url, resolver=self._resolver)
    await self.throttle_for(url, rpm).wait()
    if not await self._robots.allowed(self._client, url):
      return ''
    try:
      fetched = await _fetch_with_redirects(
        self._client, 'GET', url, resolver=self._resolver
      )
      return _decode_body(fetched, fetched.body)
    except httpx.HTTPError as exc:
      raise AdapterError(f'get_text failed {url}: {exc}') from exc

  async def close(self) -> None:
    '''Release the underlying connection pool.'''
    await self._client.aclose()


_shared_client: Optional[HttpClient] = None
'''Process-wide shared HttpClient; see get_shared_http_client.'''

_shared_lock = threading.Lock()
'''Guards shared-client creation/reset against concurrent races.'''


def get_shared_http_client() -> HttpClient:
  '''Return the process-wide shared HttpClient, creating it on first use.

  Creation is double-checked under a threading lock, so concurrent
  first callers (threads or tasks) never build two clients. The
  returned client is single-loop owned: callers must not share it
  across event loops concurrently.

  Returns:
    The singleton HttpClient. Callers must not close it; use
    reset_shared_http_client (tests) or process shutdown instead.
  '''
  global _shared_client
  if _shared_client is None:
    with _shared_lock:
      if _shared_client is None:
        _shared_client = HttpClient()
  return _shared_client


def reset_shared_http_client() -> None:
  '''Drop the shared HttpClient reference (tests only).

  The dropped client is not awaited; call close() on it first when the
  caller still holds it. Next get_shared_http_client() creates anew.
  '''
  global _shared_client
  with _shared_lock:
    _shared_client = None
