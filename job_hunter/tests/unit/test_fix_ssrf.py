#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''SSRF, board_ref injection, parser-type and ReDoS regression tests.

Written RED-first. All network is faked: DNS via an injectable resolver,
HTTP via httpx MockTransport or a stub client. No real requests are made.
'''

from __future__ import annotations

import json
import socket
import time

import httpx
import pytest
from job_hunter.adapters.aggregators import (
  AggregatorAdapter,
  parse_arbeitnow,
  parse_himalayas,
  parse_jobicy,
  parse_remoteok,
  parse_remotive,
)
from job_hunter.adapters.ashby import parse_payload as parse_ashby
from job_hunter.adapters.career_page import extract_fingerprint
from job_hunter.adapters.greenhouse import GreenhouseAdapter
from job_hunter.adapters.greenhouse import parse_payload as parse_greenhouse
from job_hunter.adapters.http_client import (
  HttpClient,
  RobotsCache,
  SSRFBlocked,
  read_capped,
  validate_public_url,
)
from job_hunter.adapters.lever import LeverAdapter
from job_hunter.adapters.lever import parse_payload as parse_lever
from job_hunter.adapters.recruitee import parse_payload as parse_recruitee
from job_hunter.adapters.registry import build_adapter
from job_hunter.adapters.smartrecruiters import parse_list_payload as parse_sr
from job_hunter.adapters.workable import parse_payload as parse_workable
from job_hunter.adapters.workday import parse_payload as parse_workday
from job_hunter.core.errors import AdapterError
from job_hunter.core.models import CompanyTarget
from job_hunter.mcp_servers import sources_server
from job_hunter.services import web_search
from job_hunter.services.domain_verifier import WebsiteVerifier

PUBLIC_IP = '93.184.216.34'


def public_resolver(
  unused_host, port, unused_family=0, unused_typ=0, unused_proto=0,
  unused_flags=0,
):
  '''Fake getaddrinfo returning one public IP for any host.'''
  return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (PUBLIC_IP, port or 443))]


def private_resolver(
  unused_host, port, unused_family=0, unused_typ=0, unused_proto=0,
  unused_flags=0,
):
  '''Fake getaddrinfo returning a private IP (DNS-rebinding style).'''
  return [
    (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('10.0.0.5', port or 443))
  ]


def failing_resolver(
  unused_host, unused_port, unused_family=0, unused_typ=0, unused_proto=0,
  unused_flags=0,
):
  '''Fake getaddrinfo that always fails resolution.'''
  raise socket.gaierror('Name or service not known')


def make_http(resolver=public_resolver, handler=None) -> HttpClient:
  '''Build an HttpClient with fake DNS and (optionally) a MockTransport.

  Args:
    resolver: Fake getaddrinfo-compatible resolver.
    handler: Optional MockTransport handler; when given, HTTP is faked.

  Returns:
    Configured HttpClient.
  '''
  transport = httpx.MockTransport(handler) if handler is not None else None
  return HttpClient(resolver=resolver, transport=transport)


# --- validate_public_url ---


@pytest.mark.parametrize(
  'url',
  [
    'http://example.com/jobs',
    'https://127.0.0.1/admin',
    'https://10.1.2.3/x',
    'https://169.254.169.254/latest/meta-data',
    'https://192.168.1.1/x',
    'https://[::1]/x',
    'https://user:pass@example.com/x',
    'https://example.com@evil.example/x',
    'https:///no-host',
    'ftp://example.com/x',
    '',
  ],
)
def test_validate_public_url_rejects(url: str) -> None:
  '''Non-https, private-literal, and userinfo URLs are rejected.'''
  with pytest.raises(SSRFBlocked):
    validate_public_url(url, resolver=public_resolver)


def test_validate_public_url_accepts_public_https() -> None:
  '''A public https URL resolving to a public IP passes.'''
  url = 'https://example.com/jobs'
  assert validate_public_url(url, resolver=public_resolver)


def test_validate_public_url_rejects_private_dns() -> None:
  '''A hostname resolving to a private IP is rejected (rebinding).'''
  with pytest.raises(SSRFBlocked):
    url = 'https://rebind.example.com/x'
    validate_public_url(url, resolver=private_resolver)


def test_validate_public_url_rejects_any_private_in_mixed() -> None:
  '''One private IP among public ones still rejects the host.'''

  def mixed(
    unused_host, unused_port, unused_family=0, unused_typ=0, unused_proto=0,
    unused_flags=0,
  ):
    return [
      (socket.AF_INET, socket.SOCK_STREAM, 6, '', (PUBLIC_IP, 443)),
      (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('192.168.0.1', 443)),
    ]

  with pytest.raises(SSRFBlocked):
    validate_public_url('https://mixed.example.com/', resolver=mixed)


def test_validate_public_url_fails_closed_on_dns_failure() -> None:
  '''Resolution failure raises instead of allowing the fetch.'''
  with pytest.raises(SSRFBlocked):
    validate_public_url('https://nope.example.com/', resolver=failing_resolver)


def test_validate_public_url_accepts_public_literal_ip() -> None:
  '''A public literal IP needs no DNS and passes.'''
  url = f'https://{PUBLIC_IP}/x'
  assert validate_public_url(url, resolver=failing_resolver)


# --- HttpClient SSRF + size cap + redirects ---


async def test_get_text_blocks_private_url_before_request() -> None:
  '''get_text raises SSRFBlocked for a private URL without any request.'''
  client = make_http()
  with pytest.raises(SSRFBlocked):
    await client.get_text('https://127.0.0.1/secret')
  await client.close()


async def test_get_json_blocks_http_url() -> None:
  '''get_json rejects plain http before requesting.'''
  client = make_http()
  with pytest.raises(SSRFBlocked):
    await client.get_json('http://example.com/jobs')
  await client.close()


async def test_get_text_rejects_redirect_to_private() -> None:
  '''A redirect whose final URL is private raises SSRFBlocked.'''

  def handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == '/robots.txt':
      return httpx.Response(404, text='nope')
    if request.url.path == '/start':
      return httpx.Response(302, headers={'Location': 'https://127.0.0.1/evil'})
    return httpx.Response(200, text='evil')

  client = make_http(handler=handler)
  with pytest.raises(SSRFBlocked):
    await client.get_text('https://example.com/start')
  await client.close()


async def test_get_text_enforces_size_cap() -> None:
  '''A body larger than 5MB raises instead of being buffered.'''

  def handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == '/robots.txt':
      return httpx.Response(404, text='nope')
    return httpx.Response(200, content=b'x' * 5_000_001)

  client = make_http(handler=handler)
  with pytest.raises(AdapterError):
    await client.get_text('https://example.com/big')
  await client.close()


class _ChunkedResponse:
  '''Minimal streaming-response stand-in for read_capped.'''

  def __init__(self, chunks: list) -> None:
    self._chunks = chunks

  async def aiter_bytes(self):
    for chunk in self._chunks:
      yield chunk


async def test_read_capped_raises_over_limit() -> None:
  '''read_capped aborts as soon as the limit is exceeded.'''
  with pytest.raises(AdapterError):
    await read_capped(_ChunkedResponse([b'12345', b'67890']), limit=9)


async def test_read_capped_returns_bytes_under_limit() -> None:
  '''read_capped concatenates chunks under the limit.'''
  assert await read_capped(_ChunkedResponse([b'ab', b'cd']), limit=9) == b'abcd'


async def test_robots_fail_closed_on_ssrf_blocked() -> None:
  '''RobotsCache denies when the robots URL itself is SSRF-blocked.'''
  async with httpx.AsyncClient() as raw:
    cache = RobotsCache(resolver=public_resolver)
    assert await cache.allowed(raw, 'https://127.0.0.1/x') is False


async def test_robots_fail_open_on_404_with_warning() -> None:
  '''Missing robots.txt stays fail-open (and is logged).'''

  def handler(unused_request: httpx.Request) -> httpx.Response:
    return httpx.Response(404, text='nope')

  async with httpx.AsyncClient(
    transport=httpx.MockTransport(handler), follow_redirects=True
  ) as raw:
    cache = RobotsCache(resolver=public_resolver)
    assert await cache.allowed(raw, 'https://example.com/x') is True


# --- board_ref validation ---


def _target(source_key: str, board_ref: str) -> CompanyTarget:
  return CompanyTarget(name='Acme', source_key=source_key, board_ref=board_ref)


@pytest.mark.parametrize(
  'source_key',
  [
    'greenhouse',
    'lever',
    'ashby',
    'workable',
    'smartrecruiters',
    'recruitee',
    'personio',
    'workday',
  ],
)
@pytest.mark.parametrize(
  'bad_ref', ['../../x', 'evil.com@x', 'a/b', 'x' * 65, '']
)
async def test_board_ref_injection_rejected(
  source_key: str, bad_ref: str
) -> None:
  '''Path-traversal / userinfo board refs raise AdapterError, no network.'''
  http = make_http()
  adapter = build_adapter(source_key, http)
  try:
    with pytest.raises(AdapterError):
      await adapter.fetch(_target(source_key, bad_ref))
  finally:
    await http.close()


async def test_workday_rejects_foreign_host() -> None:
  '''A workday board_ref on a non-myworkdayjobs host is rejected.'''
  http = make_http()
  adapter = build_adapter('workday', http)
  try:
    with pytest.raises(AdapterError):
      await adapter.fetch(_target('workday', 'evil.com/site'))
  finally:
    await http.close()


async def test_workday_accepts_real_host_shape() -> None:
  '''tenant.wdN.myworkdayjobs.com/site parses a canned MockTransport hit.'''
  payload = {
    'total': 1,
    'jobPostings': [
      {
        'externalPath': '/job/123',
        'title': 'Data Scientist',
        'locationsText': 'Remote, India',
      },
    ],
  }

  def handler(request: httpx.Request) -> httpx.Response:
    assert request.url.host == 'acme.wd1.myworkdayjobs.com'
    assert request.url.path == '/wday/cxs/acme/site/jobs'
    return httpx.Response(200, json=payload)

  http = make_http(handler=handler)
  adapter = build_adapter('workday', http)
  try:
    records = await adapter.fetch(
      _target('workday', 'acme.wd1.myworkdayjobs.com/site')
    )
  finally:
    await http.close()
  assert len(records) == 1
  assert records[0].title == 'Data Scientist'
  assert records[0].external_id == '/job/123'
  assert (
    records[0].url
    == 'https://acme.wd1.myworkdayjobs.com/en-US/site/job/123'
  )
  assert records[0].location_text == 'Remote, India'


# --- registry ---


async def test_aggregator_unknown_source_key_raises_adapter_error() -> None:
  '''AggregatorAdapter.fetch maps unknown keys to AdapterError, not KeyError.'''
  http = make_http()
  adapter = AggregatorAdapter(http)
  try:
    with pytest.raises(AdapterError):
      await adapter.fetch(_target('nope', ''))
  finally:
    await http.close()


async def test_build_adapter_unknown_raises_adapter_error() -> None:
  '''build_adapter rejects unknown source keys with AdapterError.'''
  http = make_http()
  try:
    with pytest.raises(AdapterError):
      build_adapter('nope', http)
  finally:
    await http.close()


# --- career_page ReDoS bound ---


def test_extract_fingerprint_bounded_on_pathological_input() -> None:
  '''A 40k pathological page is fingerprinted in well under 2 seconds.'''
  html = (
    '<html><body>'
    '<a href="https://job-boards.greenhouse.io/'
    + 'x' * 5000
    + '">x</a>'
    + 'a' * 40000
    + '</body></html>'
  )
  start = time.monotonic()
  extract_fingerprint(html)
  assert time.monotonic() - start < 2.0


def test_extract_fingerprint_no_match_pathological() -> None:
  '''A long no-match string completes fast and returns None.'''
  html = '<html><body>' + 'a' * 40000 + '</body></html>'
  start = time.monotonic()
  assert extract_fingerprint(html) is None
  assert time.monotonic() - start < 2.0


# --- parser type guards ---


def test_parser_type_matrix_returns_empty() -> None:
  '''Wrong top-level payload types parse to [] instead of crashing.'''
  assert parse_remotive([]) == []
  assert parse_remoteok({}) == []
  assert parse_greenhouse([]) == []
  assert parse_lever({}) == []
  assert parse_ashby([]) == []
  assert parse_workable([]) == []
  assert parse_arbeitnow([]) == []
  assert parse_jobicy([]) == []
  assert not parse_himalayas('not-a-payload')
  assert parse_recruitee([]) == []
  assert not parse_workday([], 'acme.wd1.myworkdayjobs.com', 'site')
  assert parse_sr([], 'co') == []
  assert parse_sr({}, 'co') == []


class _StubHttp:
  '''Stub HttpClient returning a fixed get_json payload.'''

  def __init__(self, payload) -> None:
    self._payload = payload

  async def get_json(self, unused_url, **unused_kwargs):
    return self._payload


async def test_greenhouse_fetch_wrong_type_returns_empty() -> None:
  '''Greenhouse fetch with a list payload returns [] (no dict() crash).'''
  adapter = GreenhouseAdapter(_StubHttp([]))
  assert await adapter.fetch(_target('greenhouse', 'acme')) == []


async def test_lever_fetch_wrong_type_returns_empty() -> None:
  '''Lever fetch with a dict payload returns [] (no list() crash).'''
  adapter = LeverAdapter(_StubHttp({}))
  assert await adapter.fetch(_target('lever', 'acme')) == []


# --- sources_server: search_web delegation + domain validation ---


def test_search_web_delegates_to_web_search_index(monkeypatch) -> None:
  '''search_web uses WebSearchIndex instead of an inline DDG client.'''
  calls = {}

  class FakeIndex:
    '''Fake WebSearchIndex recording how it was called.'''

    def __init__(self, *unused_args, **unused_kwargs) -> None:
      calls['init'] = True

    async def search(self, query, limit=15):
      calls['query'] = query
      calls['limit'] = limit
      return [{'title': 'T', 'href': 'https://example.com'}]

    async def close(self) -> None:
      return None

  monkeypatch.setattr(sources_server, 'WebSearchIndex', FakeIndex)
  out = json.loads(sources_server.search_web('acme jobs', max_results=4))
  assert out == [{'title': 'T', 'href': 'https://example.com'}]
  assert calls['query'] == 'acme jobs'


def test_detect_careers_rejects_userinfo_and_ip() -> None:
  '''detect_careers rejects userinfo / IP-literal domains without probing.'''
  for bad in ('evil.com@x', '127.0.0.1', 'https://user@evil.com'):
    out = json.loads(sources_server.detect_careers(bad))
    assert out.get('found') is False, bad


# --- web_search + domain_verifier fetch-layer validation ---


async def test_web_search_validates_endpoint_before_fetch(monkeypatch) -> None:
  '''WebSearchIndex.search fails gracefully when validation blocks.'''
  def boom(url, resolver=None):
    raise SSRFBlocked('blocked')

  monkeypatch.setattr(web_search, 'validate_public_url', boom)
  index = web_search.WebSearchIndex()
  try:
    assert await index.search('acme', limit=3) == []
  finally:
    await index.close()


async def test_domain_verifier_probe_blocks_ssrf_without_fetch() -> None:
  '''WebsiteVerifier.probe never fetches an SSRF-blocked URL.'''
  class SpyHttp:
    '''Fake HttpClient recording get_text calls.'''

    def __init__(self) -> None:
      self.calls: list = []

    async def get_text(self, url, unused_rpm=None):
      self.calls.append(url)
      return 'x' * 500

  spy = SpyHttp()
  verifier = WebsiteVerifier(http=spy)
  probe = await verifier.probe('evil.com@x')
  assert probe.ok is False
  assert not spy.calls
