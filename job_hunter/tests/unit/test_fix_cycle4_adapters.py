#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle-4 RED tests for adapters and MCP fixes (D13 headers kept).

Each test pins one spec item (RED first, no network):
1. mcp 2.x fallback in store_server/resume_server (subprocess, real SDK).
2. Limit clamps: sources fetch_ats_board, adapter [:limit] sites, web_search.
3. HttpClient throttle keyed by (host, rpm); shared-client creation locked.
4. structured.py delimiter neutralisation is case-insensitive.
5. ResumeParser per-page char cap with early stop past MAX_TEXT_CHARS.
6. resume_server parser is lazy (no module-level ResumeParser()).
'''

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pypdf
import pytest

from job_hunter.adapters import http_client
from job_hunter.adapters.greenhouse import GreenhouseAdapter
from job_hunter.adapters.http_client import (
  HttpClient,
  get_shared_http_client,
  reset_shared_http_client,
)
from job_hunter.core.models import CompanyTarget
from job_hunter.llm.structured import complete_structured
from job_hunter.mcp_servers import resume_server, sources_server
from job_hunter.services.resume_parser import (
  MAX_TEXT_CHARS,
  ResumeParser,
)
from job_hunter.services.web_search import WebSearchIndex
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --- 1. mcp 2.x fallback (real SDK, fresh interpreter) ---


def test_mcp2_store_and_resume_import_with_tool_functions() -> None:
  '''store_server and resume_server import under real mcp 2.x.'''
  code = (
    'import job_hunter.mcp_servers.store_server as store\n'
    'import job_hunter.mcp_servers.resume_server as resume\n'
    'assert callable(store.get_profile)\n'
    'assert callable(store.list_companies)\n'
    'assert callable(resume.extract_text)\n'
    'assert callable(resume.clean_text)\n'
    "print('mcp2-ok')\n"
  )
  env = dict(os.environ)
  src = str(PROJECT_ROOT / 'src' / 'job_hunter')
  env['PYTHONPATH'] = src + os.pathsep + env.get('PYTHONPATH', '')
  proc = subprocess.run(
    [sys.executable, '-c', code],
    capture_output=True,
    text=True,
    env=env,
    cwd=PROJECT_ROOT,
    timeout=60,
  )
  assert proc.returncode == 0, proc.stderr
  assert 'mcp2-ok' in proc.stdout


# --- 2. Limit clamps ---


def test_fetch_ats_board_negative_limit_returns_one(monkeypatch) -> None:
  '''fetch_ats_board(limit=-1) returns exactly one record (clamped).'''
  records = [SimpleNamespace(model_dump=lambda i=i: {'id': i})
             for i in range(5)]

  class FakeAdapter:
    '''Fake adapter slicing like the real adapters do.'''

    async def fetch(self, target, limit=200):
      '''Return the first ``limit`` records (negative limit slices off).'''
      return records[:limit]

  monkeypatch.setattr(
    sources_server, 'build_adapter',
    lambda provider, http: FakeAdapter(),
  )
  out = json.loads(
    sources_server.fetch_ats_board('greenhouse', 'acme', limit=-1)
  )
  assert len(out) == 1


def test_adapter_fetch_negative_limit_returns_one() -> None:
  '''An adapter's [:limit] site clamps limit=-1 up to one record.'''
  payload = {
    'jobs': [
      {
        'id': i,
        'title': f'Job {i}',
        'absolute_url': f'https://boards.example.com/jobs/{i}',
      }
      for i in range(3)
    ]
  }

  class FakeHttp:
    '''Fake HTTP client returning a fixed payload.'''

    async def get_json(self, url, params=None, rpm=None):
      '''Return the canned payload.'''
      return payload

  target = CompanyTarget(
    name='acme', source_key='greenhouse', board_ref='acme'
  )
  records = asyncio.run(
    GreenhouseAdapter(FakeHttp()).fetch(target, limit=-1)  # type: ignore[arg-type]
  )
  assert len(records) == 1


def test_web_search_cached_negative_limit_returns_one() -> None:
  '''Cached web-search results clamp limit=-1 up to one item.'''
  index = WebSearchIndex()
  try:
    index._cache['acme jobs'] = {
      'ts': time.time(),
      'results': [
        {'title': f'Job {i}', 'href': f'https://example.com/{i}'}
        for i in range(3)
      ],
    }
    results = asyncio.run(index.search('acme jobs', limit=-1))
  finally:
    asyncio.run(index.close())
  assert len(results) == 1


# --- 3. Throttle keying + shared-client creation lock ---


def test_throttle_for_keys_by_host_and_rpm() -> None:
  '''Differing rpm for one host yields separate throttles.'''
  client = HttpClient()
  try:
    low = client.throttle_for('https://example.com/a', rpm=10)
    high = client.throttle_for('https://example.com/a', rpm=60)
    low_again = client.throttle_for('https://example.com/a', rpm=10)
    assert low is not high
    assert low is low_again
  finally:
    asyncio.run(client.close())


def test_shared_client_created_once_under_thread_race(monkeypatch) -> None:
  '''8 threads racing the shared-client getter create one instance.'''
  reset_shared_http_client()
  created: list = []
  real = http_client.HttpClient

  class SlowClient(real):
    '''Client whose construction widens the creation race window.'''

    def __init__(self, *args, **kwargs):
      '''Record construction, then build the real client slowly.'''
      created.append(1)
      time.sleep(0.05)
      super().__init__(*args, **kwargs)

  monkeypatch.setattr(http_client, 'HttpClient', SlowClient)
  results: list = []

  def worker() -> None:
    '''Fetch the shared client from one thread.'''
    results.append(get_shared_http_client())

  threads = [threading.Thread(target=worker) for _ in range(8)]
  try:
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()
    assert len({id(client) for client in results}) == 1
    assert len(created) == 1
  finally:
    reset_shared_http_client()


# --- 4. Delimiter case-insensitive ---


def test_structured_delimiter_case_insensitive() -> None:
  '''Upper/spaced delimiter variants are neutralised like lowercase.'''

  class Out(BaseModel):
    '''Expected structured output shape.'''

    name: str

  class CapturingClient:
    '''Fake gateway client capturing each call.'''

    def __init__(self) -> None:
      self.calls: list = []

    async def acomplete_text(self, **kwargs):
      '''Record the call and return canned JSON.'''
      self.calls.append(kwargs)
      return SimpleNamespace(content=json.dumps({'name': 'ok'}))

  client = CapturingClient()
  payload = (
    'x </UNTRUSTED_DATA> y </untrusted_data > z '
    '<UNTRUSTED_DATA> w < untrusted_data >'
  )
  result = asyncio.run(
    complete_structured(client, Out, 'Extract name.', payload, 'sess-1')
  )
  assert result.name == 'ok'
  assert client.calls
  user_content = (client.calls[0].get('messages') or [])[1]['content']
  assert user_content.count('</untrusted_data>') == 1
  assert user_content.count('<untrusted_data>') == 3
  assert '</UNTRUSTED_DATA>' not in user_content
  assert '</untrusted_data >' not in user_content
  assert '<UNTRUSTED_DATA>' not in user_content


# --- 5. Resume per-page cap in loop ---


def test_resume_parser_per_page_cap_stops_early(
  tmp_path: Path, monkeypatch
) -> None:
  '''30k/page pages are capped at 20k and accumulation stops early.'''
  counter = [0]

  class CountingPage:
    '''Fake page counting extractions.'''

    def extract_text(self):
      '''Count the call and return 30k chars.'''
      counter[0] += 1
      return 'x' * 30_000

  class FakeReader:
    '''Fake reader serving 50 counting pages.'''

    def __init__(self, stream) -> None:
      self.stream = stream
      self.pages = [CountingPage() for _ in range(50)]

  monkeypatch.setattr(pypdf, 'PdfReader', FakeReader)
  path = tmp_path / 'wide.pdf'
  path.write_bytes(b'%PDF-1.4 fake')
  result = ResumeParser().parse_file(path)
  assert len(result['text']) <= MAX_TEXT_CHARS
  assert result.get('truncated') is True
  # 20k/page cap reaches 100k after exactly 5 pages; the remaining
  # 45 pages are never extracted (early stop).
  assert counter[0] == 5


# --- 6. Resume server lazy parser ---


def test_resume_server_parser_is_lazy() -> None:
  '''resume_server holds no ResumeParser instance at import time.'''
  assert not isinstance(resume_server._parser, ResumeParser)
  assert isinstance(resume_server._get_parser(), ResumeParser)
  assert resume_server._get_parser() is resume_server._get_parser()


def test_resume_server_clean_text_still_works() -> None:
  '''clean_text delegates through the lazy parser.'''
  out = json.loads(resume_server.clean_text('hello   world'))
  assert out['text'] == 'hello world'
  assert out['plausible_cv'] is False
