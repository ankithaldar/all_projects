#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Hermetic coverage-gap tests for bulk, career, runs, gateway.

Every test uses tmp_path, fakes, or a mock transport: no
network and no LLM access.
'''


from __future__ import annotations

import asyncio
import socket
import types
from pathlib import Path

import httpx
from job_hunter.adapters.career_page import CareerPageDetector
from job_hunter.adapters.http_client import HttpClient
from job_hunter.core.models import RawJobRecord
from job_hunter.services import bulk_fetch as bulk_mod
from job_hunter.services.bulk_fetch import bulk_fetch
from llm_gateway.app import LLMGateway
from llm_gateway.sanitized_gateway import SanitizedLLMGateway
from llm_gateway.schemas import GatewayRequest, GatewayResponse


def test_bulk_persist_error_branch_skips_bad_record(
  tmp_path: Path, monkeypatch
) -> None:
  '''A failing persist skips one record without aborting the batch.

  Args:
    tmp_path: Pytest temporary directory.
    monkeypatch: Pytest fixture.
  '''
  settings = types.SimpleNamespace(db_path=tmp_path / 'app.db')
  good = RawJobRecord(
    source_key='src', url='https://j.example/good', title='Good Job',
  )
  bad = RawJobRecord(
    source_key='src', url='https://j.example/bad', title='Bad Job',
  )

  async def fake_fetch_source(http, source_key, limit):
    '''Return one good and one bad record.

    Args:
      http: Unused HTTP client.
      source_key: Unused source key.
      limit: Unused record cap.

    Returns:
      Two raw records.
    '''
    _ = (http, source_key, limit)
    return [good, bad]

  async def fake_persist_record(*, jobs_repo, record, **kwargs):
    '''Raise for the bad record; insert the good one.

    Args:
      jobs_repo: Unused jobs repository.
      record: Record to persist.
      **kwargs: Forwarded persist options.

    Returns:
      Outcome string.

    Raises:
      RuntimeError: For the bad record.
    '''
    _ = (jobs_repo, kwargs)
    if 'bad' in record.url:
      raise RuntimeError('persist boom')
    return 'inserted'

  class _FakeHttp:
    '''No-op HTTP client satisfying the bulk fetch close contract.'''

    async def close(self) -> None:
      '''Do nothing; no connection was opened.'''

  monkeypatch.setattr(bulk_mod, 'fetch_source', fake_fetch_source)
  monkeypatch.setattr(bulk_mod, 'persist_record', fake_persist_record)
  report = asyncio.run(
    bulk_fetch(settings, source_keys=['src'], http=_FakeHttp())
  )
  assert report.fetched == 2
  assert report.inserted == 1


def _public_resolver(host, port, *args):
  '''Fake DNS resolver returning one public address.

  Args:
    host: Hostname being resolved.
    port: Port being resolved.
    *args: Forwarded getaddrinfo arguments.

  Returns:
    getaddrinfo-style tuples with a public IPv4 address.
  '''
  _ = (host, port, args)
  return [(
    socket.AF_INET,
    socket.SOCK_STREAM,
    6,
    '',
    ('93.184.216.34', 443),
  )]


def _mock_handler(request: httpx.Request) -> httpx.Response:
  '''Serve canned careers HTML and a 404 robots.txt.

  Args:
    request: Incoming request.

  Returns:
    Mock response.
  '''
  if request.url.path == '/robots.txt':
    return httpx.Response(404)
  html = (
    '<html><body>'
    '<a href="https://jobs.lever.co/acme">Jobs</a>'
    '</body></html>'
  )
  return httpx.Response(200, text=html)


def test_career_detect_finds_lever_via_mock_transport() -> None:
  '''Detector fingerprints a lever board from canned careers HTML.'''
  transport = httpx.MockTransport(_mock_handler)
  http = HttpClient(resolver=_public_resolver, transport=transport)
  try:
    result = asyncio.run(CareerPageDetector(http).detect('example.com'))
  finally:
    asyncio.run(http.close())
  assert result is not None
  provider, board_ref, careers_url = result
  assert provider == 'lever'
  assert board_ref == 'acme'
  assert 'example.com/careers' in careers_url


def test_runs_list_and_missing_run_404(client) -> None:
  '''Runs list starts empty; missing runs report 404.

  Args:
    client: TestClient fixture from tests/conftest.py.
  '''
  listed = client.get('/api/runs')
  assert listed.status_code == 200
  assert listed.json() == []
  events = client.get('/api/runs/999999/events')
  assert events.status_code == 404
  cancel = client.post('/api/runs/999999/cancel')
  assert cancel.status_code == 404


def test_sanitized_complete_passthrough_without_think(monkeypatch) -> None:
  '''Complete passes think-free content through unchanged.

  Args:
    monkeypatch: Pytest fixture.
  '''
  gateway = object.__new__(SanitizedLLMGateway)
  gateway._cache = None

  def fake_complete(self, request):
    '''Return a canned think-free response.

    Args:
      self: Gateway instance.
      request: Gateway request.

    Returns:
      Canned gateway response.
    '''
    _ = (self, request)
    return GatewayResponse(
      provider='dummy', model='dummy-model', content='plain answer',
    )

  monkeypatch.setattr(LLMGateway, 'complete', fake_complete)
  result = gateway.complete(GatewayRequest(prompt='hi'))
  assert result.content == 'plain answer'
  assert result.provider == 'dummy'


def test_sanitize_strips_think_tokens() -> None:
  '''Sanitizer strips think blocks from completion content.'''
  gateway = object.__new__(SanitizedLLMGateway)
  gateway._cache = None
  response = GatewayResponse(
    provider='dummy',
    model='dummy-model',
    content='<think>hidden reasoning</think>visible answer',
  )
  result = gateway._sanitize_complete_response(
    GatewayRequest(prompt='hi'), response,
  )
  assert result.content == 'visible answer'
  assert 'hidden' not in result.content
