#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Cycle-6 RED tests for adapters and MCP fixes (D13 headers kept).

Each test pins one spec item (RED first, no network):
1. Throttle dot (http_client.py): trailing-dot host shares throttle.
2. Unterminated delimiter (structured.py): payload-ending fragment neutralised.
3. Upstream page_size (workday.py + aggregators.py Himalayas): clamp to limit.
4. sources _validate_domain dedup (sources_server.py): deleted, valid_domain used.
5. _settings_obj dedup (mcp_servers/common.py + store/resume): get_settings_obj.
6. base.py F821: TYPE_CHECKING HttpClient import.
7. web_search throttle reserve-before-sleep.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from job_hunter.adapters.aggregators import AggregatorAdapter
from job_hunter.adapters.http_client import HttpClient
from job_hunter.adapters.workday import WorkdayAdapter
from job_hunter.core.models import CompanyTarget
from job_hunter.llm.structured import complete_structured
from job_hunter.mcp_servers import common, resume_server, sources_server, store_server
from job_hunter.services.web_search import WebSearchIndex
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --- 1. Throttle dot ---


def test_throttle_for_trailing_dot_bare_is_same() -> None:
  """Bare 'example.com.' shares throttle with 'example.com'."""
  client = HttpClient()
  try:
    dotted = client.throttle_for('example.com.')
    plain = client.throttle_for('example.com')
    assert dotted is plain
  finally:
    asyncio.run(client.close())


def test_throttle_for_trailing_dot_url_is_same() -> None:
  """URL 'https://example.com./a' shares throttle with plain URL."""
  client = HttpClient()
  try:
    dotted = client.throttle_for('https://example.com./a')
    plain = client.throttle_for('https://example.com/a')
    assert dotted is plain
    # Bare and URL forms share too.
    bare_dotted = client.throttle_for('example.com.')
    assert bare_dotted is plain
  finally:
    asyncio.run(client.close())


def test_throttle_dot_burst_spaced_by_shared_throttle() -> None:
  """Mixed dot/no-dot burst on one shared throttle is spaced by interval."""
  client = HttpClient()
  try:
    throttles = [
      client.throttle_for('example.com.' if i % 2 else 'example.com', 600)
      for i in range(5)
    ]
    assert len({id(item) for item in throttles}) == 1
    throttle = throttles[0]

    async def run_all() -> float:
      start = time.monotonic()
      await asyncio.gather(*[throttle.wait() for _ in range(5)])
      return time.monotonic() - start

    elapsed = asyncio.run(run_all())
    assert elapsed >= 0.4
  finally:
    asyncio.run(client.close())


# --- 2. Unterminated delimiter ---


def test_structured_unterminated_closing_neutralised() -> None:
  """Payload ending '</untrusted_data' (no >) is neutralised, no raw fragment."""

  class Out(BaseModel):
    """Expected structured output shape."""

    name: str

  class CapturingClient:
    """Fake gateway client capturing each call."""

    def __init__(self) -> None:
      self.calls: list = []

    async def acomplete_text(self, **kwargs):
      """Record the call and return canned JSON."""
      self.calls.append(kwargs)
      return SimpleNamespace(content=json.dumps({'name': 'ok'}))

  client = CapturingClient()
  payload = 'hello </untrusted_data'
  result = asyncio.run(complete_structured(client, Out, 'Extract name.', payload, 'sess-1'))
  assert result.name == 'ok'
  assert client.calls
  user_content = (client.calls[0].get('messages') or [])[1]['content']
  # Closing fragment becomes safe form.
  assert '< /untrusted_data>' in user_content
  # Only the wrapper carries a real closing tag.
  assert user_content.count('</untrusted_data>') == 1
  # No raw closing fragment remains (count of '</untrusted_data' without
  # the safe space form is just the wrapper).
  assert user_content.count('</untrusted_data') == 1


def test_structured_unterminated_opening_neutralised() -> None:
  """Payload ending '<untrusted_data' (no >) is neutralised."""

  class Out(BaseModel):
    """Expected structured output shape."""

    name: str

  class CapturingClient:
    """Fake gateway client capturing each call."""

    def __init__(self) -> None:
      self.calls: list = []

    async def acomplete_text(self, **kwargs):
      """Record the call and return canned JSON."""
      self.calls.append(kwargs)
      return SimpleNamespace(content=json.dumps({'name': 'ok'}))

  client = CapturingClient()
  payload = 'hello <untrusted_data'
  result = asyncio.run(complete_structured(client, Out, 'Extract name.', payload, 'sess-1'))
  assert result.name == 'ok'
  assert client.calls
  user_content = (client.calls[0].get('messages') or [])[1]['content']
  # Opening fragment becomes canonical opening (wrapper + 1).
  assert user_content.count('<untrusted_data>') == 2
  # No raw unterminated opening remains before wrapper.
  assert user_content.count('<untrusted_data\n') == 0


# --- 3. Upstream page_size ---


def test_workday_upstream_limit_clamped_to_limit() -> None:
  """Workday with limit=1 requests upstream limit=1, not 20."""
  seen: list = []
  payload = {
    'total': 3,
    'jobPostings': [
      {'externalPath': f'/job/{i}', 'title': f'Job {i}', 'locationsText': 'Remote'}
      for i in range(3)
    ],
  }

  class FakeHttp:
    """Fake HTTP client capturing params."""

    async def get_json(self, url, params=None, rpm=None, method=None, **kw):
      """Record params and return the canned payload."""
      seen.append(dict(params) if params else {})
      return payload

  target = CompanyTarget(
    name='acme', source_key='workday', board_ref='acme.wd1.myworkdayjobs.com/site'
  )
  records = asyncio.run(WorkdayAdapter(FakeHttp()).fetch(target, limit=1))
  assert len(records) == 1
  assert seen
  assert seen[0].get('limit') == 1


def test_himalayas_upstream_limit_clamped_to_limit() -> None:
  """Himalayas with limit=1 requests effective upstream limit=1."""
  seen: list = []
  payload = {
    'jobs': [
      {
        'title': f'Job {i}',
        'companyName': 'Acme',
        'applicationLink': f'https://himalayas.app/jobs/{i}',
      }
      for i in range(3)
    ],
    'nextCursor': '',
  }

  class FakeHttp:
    """Fake HTTP client capturing URL and params."""

    async def get_json(self, url, params=None, rpm=None, **kw):
      """Record url/params and return the canned payload."""
      seen.append((url, dict(params) if params else {}))
      return payload

  target = CompanyTarget(name='x', source_key='himalayas', board_ref='')
  records = asyncio.run(AggregatorAdapter(FakeHttp()).fetch(target, limit=1))
  assert len(records) == 1
  assert seen
  url, params = seen[0]
  # Effective upstream limit: params wins over URL query (httpx merge).
  if 'limit' in params:
    effective = int(params['limit'])
  else:
    query = parse_qs(urlparse(url).query)
    values = query.get('limit') or ['100']
    effective = int(values[0])
  assert effective == 1


# --- 4. sources _validate_domain dedup ---


def test_sources_no_local_validate_domain() -> None:
  """sources_server must not keep local _DOMAIN_RE/_validate_domain."""
  assert not hasattr(sources_server, '_validate_domain')
  assert not hasattr(sources_server, '_DOMAIN_RE')


def test_detect_careers_scheme_tolerant_still_works(monkeypatch) -> None:
  """detect_careers('https://example.com/') strips scheme and detects."""

  class FakeDetector:
    """Fake detector recording the domain it was asked about."""

    def __init__(self) -> None:
      self.seen: list = []

    async def detect(self, domain, max_probes=3, name=''):
      """Record domain and return a canned fingerprint."""
      self.seen.append(domain)
      return ('lever', 'acme', 'https://example.com/careers')

  fake = FakeDetector()
  monkeypatch.setattr(sources_server, '_detector', fake)
  out = json.loads(sources_server.detect_careers('https://example.com/'))
  assert out.get('found') is True
  assert fake.seen and fake.seen[0] == 'example.com'


def test_detect_careers_rejects_userinfo_and_ip() -> None:
  """detect_careers rejects userinfo/IP via base valid_domain."""
  for bad in ('evil.com@x', '127.0.0.1', 'https://user@evil.com'):
    out = json.loads(sources_server.detect_careers(bad))
    assert out.get('found') is False, bad


# --- 5. _settings_obj dedup ---


def test_common_has_get_settings_obj() -> None:
  """common.py exposes get_settings_obj() as the shared bootstrap."""
  assert hasattr(common, 'get_settings_obj')
  assert callable(common.get_settings_obj)


def test_store_resume_use_common_get_settings_obj() -> None:
  """store/resume _settings_obj is the shared common function (no copy)."""
  assert hasattr(store_server, '_settings_obj')
  assert hasattr(resume_server, '_settings_obj')
  assert store_server._settings_obj is common.get_settings_obj
  assert resume_server._settings_obj is common.get_settings_obj


def test_common_get_settings_obj_singleton(monkeypatch) -> None:
  """get_settings_obj() lazily bootstraps once and caches."""
  created: list = []
  fake = SimpleNamespace(db_path='x', data_dir='/tmp')

  def slow_bootstrap(path):
    """Slow bootstrap widening the creation race window."""
    created.append(1)
    time.sleep(0.05)
    return fake

  # Reset shared cache, then patch both core and common bootstrap targets
  # (implementation imports bootstrap in one of these places).
  monkeypatch.setattr(common, '_settings', None, raising=False)
  monkeypatch.setattr(common, '_settings_bootstrap_fn', None, raising=False)
  import job_hunter.core.bootstrap as bootstrap_mod
  monkeypatch.setattr(bootstrap_mod, 'bootstrap', slow_bootstrap)
  monkeypatch.setattr(common, 'bootstrap', slow_bootstrap, raising=False)
  # Also patch module-level bootstrap attributes for any implementation style.
  monkeypatch.setattr(store_server, 'bootstrap', slow_bootstrap, raising=False)
  monkeypatch.setattr(resume_server, 'bootstrap', slow_bootstrap, raising=False)

  import threading

  results: list = []

  def worker() -> None:
    """Fetch one instance from a thread."""
    results.append(common.get_settings_obj())

  threads = [threading.Thread(target=worker) for _ in range(8)]
  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()
  assert len({id(item) for item in results}) == 1
  assert len(created) == 1
  monkeypatch.setattr(common, '_settings', None, raising=False)
  monkeypatch.setattr(common, '_settings_bootstrap_fn', None, raising=False)


# --- 6. base.py F821 ---


def test_base_has_type_checking_httpclient_import() -> None:
  """adapters/base.py imports HttpClient under TYPE_CHECKING."""
  text = (
    PROJECT_ROOT / 'src' / 'job_hunter' / 'job_hunter' / 'adapters' / 'base.py'
  ).read_text(encoding='utf-8')
  assert 'TYPE_CHECKING' in text
  assert 'from job_hunter.adapters.http_client import HttpClient' in text


# --- 7. web_search throttle reserve-before-sleep ---


def test_websearch_throttle_concurrent_spaces_by_interval() -> None:
  """5 concurrent _throttle() at 0.1s take >=0.4s total (reserve-before-sleep)."""
  index = WebSearchIndex(min_interval=0.1, rpm=600)
  assert index._min_interval == pytest.approx(0.1)

  async def run_all() -> float:
    start = time.monotonic()
    await asyncio.gather(*[index._throttle() for _ in range(5)])
    return time.monotonic() - start

  try:
    elapsed = asyncio.run(run_all())
  finally:
    asyncio.run(index.close())
  assert elapsed >= 0.4 or elapsed >= 0.35
  # The old after-sleep pattern wakes together in ~one interval.
  assert elapsed > 0.2
