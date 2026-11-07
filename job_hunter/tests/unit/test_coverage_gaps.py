#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Hermetic tests for previously 0%-covered modules.

Every test uses tmp_path, fakes, or monkeypatch only: no network,
no LLM, and no stdio MCP transport. MCP handlers run directly.
'''


from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest
import yaml
from job_hunter.core.errors import ConflictError
from job_hunter.db.repositories.companies import CompaniesRepository
from job_hunter.db.repositories.runs import RunsRepository
from job_hunter.graph.checkpointing import open_checkpointer, thread_config
from job_hunter.services.company_discovery import ingest_seeds
from job_hunter.services.domain_verifier import registrable_domain
from job_hunter.workers.jobs import (
  _settings,
  enqueue_run,
  execute_pending_run,
  recover_orphans,
  stale_sweep,
)
from llm_gateway.sanitize import StreamThinkSanitizer, sanitize_think_tokens

try:
  import importlib as _importlib

  _importlib.import_module('mcp.server.fastmcp')
except ModuleNotFoundError:
  _fake = types.ModuleType('mcp.server.fastmcp')

  class _FakeFastMCP:
    '''Minimal stand-in whose tool decorator returns the handler.'''

    def __init__(self, *args, **unused_kwargs) -> None:
      '''Accept any constructor arguments.'''
      _ = (args, unused_kwargs)

    def tool(self, *args, **unused_kwargs):
      '''Return a decorator returning the wrapped handler unchanged.'''
      def decorator(func):
        '''Return the original handler function.'''
        return func

      if args and callable(args[0]):
        return args[0]
      return decorator

    def run(self) -> None:
      '''Do nothing; stdio serving is never exercised in tests.'''

  _fake.FastMCP = _FakeFastMCP
  sys.modules['mcp.server.fastmcp'] = _fake

# pylint: disable=ungrouped-imports
from job_hunter.mcp_servers import (
  resume_server,
  sources_server,
  store_server,
)


def test_company_discovery_registrable() -> None:
  '''Prove canonical registrable_domain extracts domains.'''
  assert (
    registrable_domain('https://jobs.example.com/careers')
    == 'example.com'
  )
  assert registrable_domain('https://example.com/') == 'example.com'
  assert (
    registrable_domain('https://careers.acme.co.in/x') == 'acme.co.in'
  )


def test_company_discovery_ingest_seeds(
  tmp_path: Path, app_settings, db_path,
) -> None:
  '''Prove ingest_seeds upserts every companies_*.yaml entry.'''
  seeds = tmp_path / 'seeds'
  seeds.mkdir()
  (seeds / 'companies_test.yaml').write_text(
    yaml.safe_dump({'companies': [
      {'name': 'Acme Pvt Ltd', 'domain': 'acme.example',
       'vertical_hint': 'saas', 'priority': 5},
      {'name': 'Beta', 'domain': 'beta.example', 'priority': 2},
    ]}),
    encoding='utf-8',
  )
  (seeds / 'ignored.txt').write_text('companies: []', encoding='utf-8')
  count = asyncio.run(ingest_seeds(app_settings, seeds_dir=seeds))
  assert count == 2
  rows = CompaniesRepository(db_path).list_companies(limit=10)
  assert {row['name'] for row in rows} == {'Acme Pvt Ltd', 'Beta'}
  acme = next(row for row in rows if row['name'] == 'Acme Pvt Ltd')
  assert acme['vertical'] == 'saas'
  assert acme['priority'] == 5


def test_checkpointing_thread_config() -> None:
  '''Prove thread_config carries thread_id and optional settings.'''
  first = thread_config(7)
  assert first == {
    'configurable': {
      'thread_id': 'run-7', 'checkpoint_ns': '',
    },
  }
  assert 'checkpoint_ns' in first['configurable']
  assert first['configurable']['checkpoint_ns'] == ''
  sentinel = object()
  cfg = thread_config(3, settings=sentinel)
  assert cfg['configurable']['thread_id'] == 'run-3'
  assert 'checkpoint_ns' in cfg['configurable']
  assert cfg['configurable']['settings'] is sentinel


def test_checkpointing_open_returns_none_or_saver(
  tmp_path: Path,
) -> None:
  '''Prove open_checkpointer yields None or a saver without crashing.'''
  async def _probe():
    '''Open the checkpointer and return the yielded saver.'''
    async with open_checkpointer(tmp_path / 'ckpt.sqlite') as saver:
      return saver

  saver = asyncio.run(_probe())
  try:
    # pylint: disable=import-outside-toplevel
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    assert AsyncSqliteSaver is not None
    assert saver is not None
  except ModuleNotFoundError:
    assert saver is None


def test_sanitize_think_strip() -> None:
  '''Prove closed, unclosed, bracket, and orphan think markers strip.'''
  assert sanitize_think_tokens('<think>p</think>hello') == 'hello'
  assert sanitize_think_tokens('hi<thinking>p</thinking>t') == 'hit'
  assert sanitize_think_tokens('keep<think>never closed') == 'keep'
  assert sanitize_think_tokens('[think]p[/think]shown') == 'shown'
  assert sanitize_think_tokens('a</think>b') == 'ab'
  assert sanitize_think_tokens('') == ''
  assert sanitize_think_tokens('plain answer') == 'plain answer'


def test_sanitize_pii_preserved_outside_think() -> None:
  '''Prove outside text is verbatim and think-block PII is stripped.'''
  text = '<think>alice@example.com</think>bob@example.com'
  assert sanitize_think_tokens(text) == 'bob@example.com'


def test_sanitize_streaming() -> None:
  '''Prove StreamThinkSanitizer hides think content across chunks.'''
  sanitizer = StreamThinkSanitizer()
  out = sanitizer.feed('start <thi')
  out += sanitizer.feed('nk>hidden</think> end')
  out += sanitizer.flush()
  assert out == 'startend'
  assert 'hidden' not in out


def test_workers_enqueue_and_nothing_pending(app_settings) -> None:
  '''Prove enqueue refuses a second active run and sweeps are no-ops.'''
  config_path = app_settings.config_path
  assert execute_pending_run(config_path) == {
    'run_id': None, 'status': 'nothing_pending', 'stats': {},
  }
  run_id = enqueue_run(config_path, triggered_by='test')
  assert isinstance(run_id, int) and run_id >= 1
  with pytest.raises(ConflictError):
    enqueue_run(config_path)
  assert recover_orphans(config_path) == []
  assert stale_sweep(config_path) == 0


def test_workers_settings_helper(tmp_path: Path, app_settings) -> None:
  '''Prove _settings bootstraps and migrates a tmp data dir.'''
  loaded = _settings(app_settings.config_path)
  assert loaded.db_path.exists()
  assert loaded.data_dir == tmp_path


@pytest.fixture()
def store(app_settings, db_path, monkeypatch):
  '''Return the store server module bound to tmp settings and DB.'''
  _ = db_path
  monkeypatch.setattr(store_server, '_settings', app_settings)
  return store_server


def test_store_server_handlers(  # pylint: disable=redefined-outer-name
  store,
) -> None:
  '''Prove store handlers list, upsert, and return empty collections.'''
  assert json.loads(store.get_profile()) == {
    'error': 'no profile yet',
  }
  created = json.loads(
    store.upsert_company('StoreCo', 'store.example', 'saas'),
  )
  assert created['company_id'] >= 1
  companies = json.loads(store.list_companies())
  assert [c['name'] for c in companies] == ['StoreCo']
  assert json.loads(store.jobs_recent()) == []
  assert json.loads(store.recommendations_top()) == []


def test_store_server_record_run_event(  # pylint: disable=redefined-outer-name
  store, db_path,
) -> None:
  '''Prove record_run_event persists an event for a real run.'''
  run_id = RunsRepository(db_path).create('discovery', 'test')
  event = json.loads(
    store.record_run_event(run_id, 'info', 'node-x', 'hello'),
  )
  assert event['event_id'] >= 1


def test_sources_server_fetch_unknown_provider() -> None:
  '''Prove fetch_ats_board errors cleanly for an unknown provider.'''
  payload = json.loads(sources_server.fetch_ats_board('nope', 'x'))
  assert 'unknown source key' in payload['error']


def test_sources_server_fetch_page_and_detect(monkeypatch) -> None:
  '''Prove fetch_page cleans HTML and detect_careers maps results.'''
  class _FakeHttp:
    '''Fake HTTP client returning canned HTML without network.'''

    async def get_text(self, url):
      '''Return canned page HTML for any URL.'''
      _ = url
      html = ('<html><head><title>T</title></head>'
              '<body><script>x()</script>Hi</body></html>')
      return html

  class _FakeDetector:
    '''Fake detector returning a canned ATS fingerprint.'''

    async def detect(self, domain):
      '''Return a canned detection tuple for any domain.'''
      _ = domain
      return ('lever', 'acme', 'https://example.com/careers')

  monkeypatch.setattr(sources_server, '_http', _FakeHttp())
  url = 'https://example.com/'
  payload = json.loads(asyncio.run(sources_server.fetch_page(url)))
  assert payload['title'] == 'T'
  assert payload['text'] == 'T Hi'
  monkeypatch.setattr(sources_server, '_detector', _FakeDetector())
  found = json.loads(sources_server.detect_careers('example.com'))
  assert found == {
    'found': True, 'provider': 'lever', 'board_ref': 'acme',
    'careers_url': 'https://example.com/careers',
  }


def test_sources_server_detect_not_found(monkeypatch) -> None:
  '''Prove detect_careers reports found=false when detector is None.'''
  class _NoneDetector:
    '''Fake detector that never finds anything.'''

    async def detect(self, domain):
      '''Return None for any domain.'''
      _ = domain

  monkeypatch.setattr(sources_server, '_detector', _NoneDetector())
  payload = json.loads(sources_server.detect_careers('example.com'))
  assert payload == {'found': False}


def test_resume_server_clean_text() -> None:
  '''Prove clean_text normalizes ligatures and flags plausible CVs.'''
  raw = '  ﬁ experience education skills project work ' + 'x ' * 300
  payload = json.loads(resume_server.clean_text(raw))
  assert payload['text'].startswith('fi experience')
  assert payload['plausible_cv'] is True
  short = json.loads(resume_server.clean_text('hello'))
  assert short == {'text': 'hello', 'plausible_cv': False}


def test_resume_server_extract_missing_file(tmp_path: Path) -> None:
  '''Prove extract_text returns ok=false for a missing PDF file.'''
  payload = json.loads(
    resume_server.extract_text(str(tmp_path / 'nope.pdf')),
  )
  assert payload['ok'] is False
  assert payload['error']
