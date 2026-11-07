#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle-13 tester: fan-out, bulk enablement, raw resume, huge ids,
sanitized astream cache, embed/fetch-failure counts.

Hermetic: tmp_path DBs, stub adapters/providers, no network.
V1 contracts aligned with test_fix_cycle13_backend (list_due,
resolve_sources(settings), _normalize_raw_jobs, huge-int bounds).
'''

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import AsyncIterator, Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
import yaml

fastapi = pytest.importorskip('fastapi')
TestClient = pytest.importorskip('fastapi.testclient').TestClient

ROOT = Path(__file__).resolve().parents[2]
_HUGE = '1' + '0' * 400


def _settings(tmp_path: Path, monkeypatch, *, extra: str = ''):
  '''Build migrated settings for direct service tests.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest fixture.
    extra: Extra YAML appended to the config.

  Returns:
    Migrated AppSettings.
  '''
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  config_path = tmp_path / 'app.yaml'
  config_path.write_text(
    'salary_hard_floor_lpa: 45\n' + extra, encoding='utf-8',
  )
  from job_hunter.core.config import AppSettings
  from job_hunter.core.db import run_migrations
  settings = AppSettings(config_path)
  settings.data_dir.mkdir(parents=True, exist_ok=True)
  run_migrations(settings.db_path)
  return settings


def _make_client(tmp_path: Path, monkeypatch) -> TestClient:
  '''Build an app client against a temporary data directory.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest fixture.

  Returns:
    Configured TestClient.
  '''
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  monkeypatch.setenv('APP_HOST', '127.0.0.1')
  monkeypatch.setenv('JH_API_TOKEN', '')
  monkeypatch.setenv('APP_LOG_LEVEL', 'INFO')
  config_path = tmp_path / 'app.yaml'
  config_path.write_text(
    'salary_hard_floor_lpa: 45\n', encoding='utf-8',
  )
  from job_hunter.api.main import create_app
  app = create_app(str(config_path))
  return TestClient(app, raise_server_exceptions=False)


def _seed_company(
  db_path: Path, name: str, last_checked: str | None,
  *, provider: str = 'lever', status: str = 'active', priority: int = 3,
) -> int:
  '''Insert one company row directly (preserving last_checked_at).

  Args:
    db_path: Database path.
    name: Company name.
    last_checked: last_checked_at value or None.
    provider: ATS provider key.
    status: Lifecycle status.
    priority: Priority value.

  Returns:
    New company id.
  '''
  with closing(sqlite3.connect(str(db_path))) as conn:
    cursor = conn.execute(
      'INSERT INTO companies (name, normalized_name, domain, '
      'ats_provider, board_ref, priority, status, last_checked_at) '
      'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
      (
        name, name.lower(), f'{name.lower()}.example.com',
        provider, f'board-{name.lower()}', priority, status, last_checked,
      ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def _seed_60(db_path: Path) -> list[int]:
  '''Seed 60 active companies with staggered oldest-first stamps.

  Args:
    db_path: Database path.

  Returns:
    Company ids in oldest-first (insertion) order.
  '''
  return [
    _seed_company(db_path, f'Co{i:02d}', f'2026-01-01 00:00:{i:02d}')
    for i in range(60)
  ]


class _EmptyAdapter:
  '''Adapter returning no records.'''

  def __init__(self, http) -> None:
    '''Accept the shared client.

    Args:
      http: Unused client.
    '''
    _ = http

  async def fetch(self, target, limit=200):
    '''Return no records.

    Args:
      target: Target.
      limit: Cap.

    Returns:
      Empty record list.
    '''
    _ = (target, limit)
    return []


def _stub_adapters(monkeypatch) -> None:
  '''Route every build_adapter call to the empty stub adapter.

  Args:
    monkeypatch: Pytest fixture.
  '''
  import job_hunter.graph.nodes as nodes_mod

  def stub(key, http):
    '''Return the empty stub for any source key.

    Args:
      key: Source key.
      http: Shared client.

    Returns:
      Empty stub adapter.
    '''
    _ = key
    return _EmptyAdapter(http)

  monkeypatch.setattr(nodes_mod, 'build_adapter', stub)


# 1. Fan-out rotation -----------------------------------------------------

def test_fanout_first_50_oldest_via_route_cap(
  tmp_path: Path, monkeypatch,
) -> None:
  '''60 due companies: build_plan first 50 are oldest; route caps at 50.

  Current contract (priority/name) and V1 (last_checked_at oldest)
  agree here because names and stamps are both oldest-first.
  '''
  settings = _settings(tmp_path, monkeypatch)
  ids = _seed_60(settings.db_path)
  _stub_adapters(monkeypatch)
  import job_hunter.graph.nodes as nodes_mod
  from job_hunter.graph.discovery_graph import MAX_FANOUT, route_targets

  assert MAX_FANOUT == 50
  update = asyncio.run(
    nodes_mod.build_plan(
      {'run_id': 1, 'plan': {'quick_poll': False}},
      {'configurable': {'settings': settings}},
    ),
  )
  assert 'errors' not in update or not update['errors'], update
  targets = update['plan']['targets']
  planned_ids = [int(t['company_id']) for t in targets]
  assert planned_ids[:50] == ids[:50], planned_ids[:5]
  sends = route_targets({'plan': update['plan']})
  assert len(sends) == 50, len(sends)
  sent_ids = [int(s.arg['company_id']) for s in sends]
  assert sent_ids == planned_ids[:50]


def test_fanout_rotation_after_marking_checked(
  tmp_path: Path, monkeypatch,
) -> None:
  '''V1: after first 50 are marked checked, next plan starts at id 50.'''
  settings = _settings(tmp_path, monkeypatch)
  ids = _seed_60(settings.db_path)
  _stub_adapters(monkeypatch)
  import job_hunter.graph.nodes as nodes_mod

  config = {'configurable': {'settings': settings}}

  def run_plan_ids() -> list[int]:
    '''Run build_plan and return planned company ids.'''
    update = asyncio.run(
      nodes_mod.build_plan(
        {'run_id': 1, 'plan': {'quick_poll': False}}, config,
      ),
    )
    assert 'errors' not in update or not update['errors'], update
    return [int(t['company_id']) for t in update['plan']['targets']]

  first_ids = run_plan_ids()
  assert first_ids[:50] == ids[:50], first_ids[:5]
  with closing(sqlite3.connect(str(settings.db_path))) as conn:
    conn.executemany(
      'UPDATE companies SET last_checked_at = ? WHERE id = ?',
      [('2027-01-01 00:00:00', cid) for cid in ids[:50]],
    )
    conn.commit()
  second_ids = run_plan_ids()
  assert second_ids[0] == ids[50], second_ids[:3]


def test_fetch_pair_marks_company_checked_v1(
  tmp_path: Path, monkeypatch,
) -> None:
  '''V1: fetch_pair stamps companies.last_checked_at on success.'''
  settings = _settings(tmp_path, monkeypatch)
  company_id = _seed_company(
    settings.db_path, 'Checked', '2020-01-01 00:00:00',
  )
  _stub_adapters(monkeypatch)
  import job_hunter.graph.nodes as nodes_mod
  from job_hunter.db.repositories.companies import CompaniesRepository

  update = asyncio.run(
    nodes_mod.fetch_pair(
      {
        'company_id': company_id, 'name': 'Checked',
        'source_key': 'lever', 'board_ref': 'board-checked',
      },
      {'configurable': {'settings': settings}},
    ),
  )
  assert not update['errors'], update
  row = CompaniesRepository(settings.db_path).get(company_id)
  assert row is not None
  assert row['last_checked_at'] not in (None, '2020-01-01 00:00:00'), row


# 2. bulk_fetch enablement -------------------------------------------------

def test_resolve_sources_current_defaults_and_dedupe() -> None:
  '''resolve_sources([]) falls back to defaults; dedupes/lowercases.'''
  from job_hunter.services.bulk_fetch import DEFAULT_SOURCES, resolve_sources

  assert resolve_sources([]) == list(DEFAULT_SOURCES)
  assert resolve_sources(['Jobicy', 'jobicy', ' arbeitnow ']) == [
    'jobicy', 'arbeitnow',
  ]


def test_build_plan_excludes_disabled_source(
  tmp_path: Path, monkeypatch,
) -> None:
  '''build_plan drops companies whose source is enabled:false.'''
  settings = _settings(
    tmp_path, monkeypatch,
    extra='sources:\n  lever:\n    enabled: false\n',
  )
  _seed_company(settings.db_path, 'LeverCo', None, provider='lever')
  _seed_company(settings.db_path, 'GhCo', None, provider='greenhouse')
  _stub_adapters(monkeypatch)
  import job_hunter.graph.nodes as nodes_mod

  update = asyncio.run(
    nodes_mod.build_plan(
      {'run_id': 1, 'plan': {'quick_poll': False}},
      {'configurable': {'settings': settings}},
    ),
  )
  assert 'errors' not in update or not update['errors'], update
  names = [t['name'] for t in update['plan']['targets']]
  assert 'GhCo' in names, names
  assert 'LeverCo' not in names, names


def test_resolve_sources_filters_disabled_defaults_v1(
  tmp_path: Path, monkeypatch,
) -> None:
  '''V1: resolve_sources([], settings) drops disabled defaults.'''
  settings = _settings(
    tmp_path, monkeypatch,
    extra='sources:\n  remotive:\n    enabled: false\n',
  )
  from job_hunter.services.bulk_fetch import resolve_sources

  defaults = resolve_sources([], settings)
  assert 'remotive' not in defaults, defaults
  assert 'arbeitnow' in defaults, defaults
  assert resolve_sources(['remotive'], settings) == ['remotive']
  assert 'remotive' in resolve_sources([])


def test_bulk_fetch_default_skips_disabled_source_v1(
  tmp_path: Path, monkeypatch,
) -> None:
  '''V1: bulk_fetch with no explicit sources skips disabled ones.'''
  settings = _settings(
    tmp_path, monkeypatch,
    extra='sources:\n  remotive:\n    enabled: false\n',
  )
  from job_hunter.services import bulk_fetch as bulk_mod

  seen: list[str] = []

  async def fake_fetch_source(http, source_key, limit):
    '''Record the source and return nothing.

    Args:
      http: Unused client.
      source_key: Source being fetched.
      limit: Cap.

    Returns:
      Empty record list.
    '''
    _ = (http, limit)
    seen.append(source_key)
    return []

  class _FakeHttp:
    '''No-op HTTP client.'''

    async def close(self) -> None:
      '''Do nothing.'''

  monkeypatch.setattr(bulk_mod, 'fetch_source', fake_fetch_source)
  asyncio.run(bulk_mod.bulk_fetch(settings, http=_FakeHttp(), persist=False))
  assert 'remotive' not in seen, seen
  assert 'arbeitnow' in seen, seen


# 3. raw_jobs resume -------------------------------------------------------

def test_normalize_dedupe_object_raw_jobs_inserts(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Object raw_jobs flow through normalize_dedupe to inserted objects.'''
  settings = _settings(tmp_path, monkeypatch)
  import job_hunter.graph.nodes as nodes_mod
  from job_hunter.core.models import NormalizedJob, RawJobRecord
  from job_hunter.graph.state import initial_state

  state = initial_state(run_id=1, triggered_by='test')
  state['raw_jobs'] = [RawJobRecord(
    source_key='lever', url='https://j.example/obj-1',
    title='Staff Data Scientist', company_name='ObjCo',
    description_text='python sql experimentation 55 LPA data pipelines',
  )]
  update = asyncio.run(
    nodes_mod.normalize_dedupe(
      state, {'configurable': {'settings': settings}},
    ),
  )
  assert 'errors' not in update or not update['errors'], update
  assert update['stats']['new_jobs'] == 1, update
  assert len(update['inserted']) == 1
  assert isinstance(update['inserted'][0], NormalizedJob)


def test_normalize_dedupe_dict_raw_jobs_v1(
  tmp_path: Path, monkeypatch,
) -> None:
  '''V1: dict-shaped raw_jobs (checkpoint form) persist via normalize.'''
  settings = _settings(tmp_path, monkeypatch)
  import job_hunter.graph.nodes as nodes_mod

  state = {
    'run_id': 1,
    'raw_jobs': [
      {
        'source_key': 'lever', 'url': 'https://j.example/dict-c13-1',
        'title': 'Staff Data Scientist', 'company_name': 'DictCo13',
        'description_text': 'python sql experimentation 55 LPA',
      },
      {'title': 'malformed'},
    ],
  }
  update = asyncio.run(
    nodes_mod.normalize_dedupe(
      state, {'configurable': {'settings': settings}},
    ),
  )
  assert 'errors' not in update or not update['errors'], update
  assert update['stats']['new_jobs'] == 1, update
  assert len(update['inserted']) == 1, update


def test_normalize_raw_jobs_helper_v1(caplog) -> None:
  '''V1: _normalize_raw_jobs coerces dicts, skips malformed, in place.'''
  from job_hunter.core.models import RawJobRecord
  from job_hunter.graph.state import _normalize_raw_jobs

  existing = RawJobRecord(
    source_key='lever', url='https://j.example/9', title='Existing',
  )
  state = {
    'raw_jobs': [
      {
        'source_key': 'lever', 'url': 'https://j.example/1',
        'title': 'DS', 'company_name': 'Acme',
      },
      {'title': 'no url, no source'},
      'not-a-dict',
      existing,
    ],
  }
  with caplog.at_level(logging.WARNING):
    normalized = _normalize_raw_jobs(state)
  assert len(normalized) == 2, normalized
  assert all(
    isinstance(record, RawJobRecord) for record in normalized
  ), normalized
  assert state['raw_jobs'] == normalized
  assert normalized[0].title == 'DS'


# 4. Huge path IDs ---------------------------------------------------------

def test_jobs_page_huge_422_control(tmp_path: Path, monkeypatch) -> None:
  '''GET /api/jobs?page=10**400 -> 422 (le=10000 already landed).'''
  client = _make_client(tmp_path, monkeypatch)
  response = client.get(f'/api/jobs?page={_HUGE}')
  assert response.status_code == 422, response.text


def test_huge_job_path_422_v1(tmp_path: Path, monkeypatch) -> None:
  '''V1: GET /api/jobs/{10**400} -> 422, not 500 OverflowError.'''
  client = _make_client(tmp_path, monkeypatch)
  response = client.get(f'/api/jobs/{_HUGE}')
  assert response.status_code == 422, response.text


def test_huge_company_patch_422_v1(tmp_path: Path, monkeypatch) -> None:
  '''V1: PATCH /api/companies/{10**400} -> 422, not 500.'''
  client = _make_client(tmp_path, monkeypatch)
  response = client.patch(f'/api/companies/{_HUGE}', json={})
  assert response.status_code == 422, response.text


def test_huge_recommendations_page_422_v1(
  tmp_path: Path, monkeypatch,
) -> None:
  '''V1: GET /api/recommendations?page=10**400 -> 422, not 500.'''
  client = _make_client(tmp_path, monkeypatch)
  response = client.get(f'/api/recommendations?page={_HUGE}')
  assert response.status_code == 422, response.text


# 5. Sanitized cache-stream astream ---------------------------------------

class _ChunkedDummyProvider:
  '''Dummy provider yielding fixed chunks.'''

  def __init__(self, name: str, chunks: list) -> None:
    '''Initialize chunked provider.

    Args:
      name: Provider name.
      chunks: Text pieces streamed in order.
    '''
    self.name = name
    self.chunks = chunks
    self.stream_calls = 0

  def stream(self, request) -> Iterator[Any]:
    '''Yield the text in small chunks.

    Args:
      request: Provider request.

    Yields:
      Provider chunks.
    '''
    from llm_gateway.schemas import ProviderChunk

    self.stream_calls += 1
    pieces = list(self.chunks)
    for index, piece in enumerate(pieces):
      last = index == len(pieces) - 1
      yield ProviderChunk(
        provider=self.name,
        model=request.model,
        delta_content=piece,
        finish_reason='stop' if last else None,
      )

  async def astream(self, request) -> AsyncIterator[Any]:
    '''Yield the text in small chunks asynchronously.

    Args:
      request: Provider request.

    Yields:
      Provider chunks.
    '''
    from llm_gateway.schemas import ProviderChunk

    self.stream_calls += 1
    pieces = list(self.chunks)
    for index, piece in enumerate(pieces):
      last = index == len(pieces) - 1
      yield ProviderChunk(
        provider=self.name,
        model=request.model,
        delta_content=piece,
        finish_reason='stop' if last else None,
      )


def _make_cache_gateway(tmp_path: Path, provider) -> object:
  '''Build a cache-enabled SanitizedLLMGateway with one provider.

  Args:
    tmp_path: Pytest temp dir.
    provider: Dummy provider instance.

  Returns:
    Configured gateway (caller must close it).
  '''
  from llm_gateway.sanitized_gateway import SanitizedLLMGateway

  config = {
    'default_system_prompt': 'test',
    'retry': {
      'max_attempts': 1,
      'base_delay_seconds': 0.0,
      'max_delay_seconds': 0.0,
    },
    'cache': {
      'enabled': True,
      'ttl_seconds': 60,
      'include_temperature': False,
    },
    'providers': {
      'p1': {
        'enabled': True,
        'base_url': 'http://dummy.invalid/v1',
        'api_key_env': '',
        'rpm': 100,
        'timeout_seconds': 1.0,
        'models': {
          'fast': {
            'model': 'dummy-model',
            'temperature': 0.0,
            'max_tokens': 10,
            'cost_per_million_input': 0.0,
            'cost_per_million_output': 0.0,
            'supports_tools': True,
          },
        },
      },
    },
    'execution_order': [{'provider': 'p1', 'alias': 'fast'}],
    'fallback': None,
  }
  config_path = tmp_path / 'gateway.yaml'
  config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
  env_path = tmp_path / '.env'
  env_path.write_text('', encoding='utf-8')
  return SanitizedLLMGateway(
    config_path=str(config_path),
    env_path=str(env_path),
    db_path=tmp_path / 'logs.db',
    cache_path=tmp_path / 'cache.db',
    provider_overrides={'p1': provider},
  )


async def test_sanitized_astream_cache_twice_second_is_cache(
  tmp_path: Path,
) -> None:
  '''Cache-enabled astream twice: second is provider=cache, sanitized.

  Args:
    tmp_path: Pytest temp dir.
  '''
  from llm_gateway.schemas import GatewayRequest

  chunks = ['<thi', 'nk>secret reasoning</th', 'ink>visible answer']
  provider = _ChunkedDummyProvider('p1', chunks)
  gateway = _make_cache_gateway(tmp_path, provider)
  try:
    request = GatewayRequest(prompt='hi')
    first = []
    async for chunk in gateway.astream(request):
      first.append(chunk)
    second = []
    async for chunk in gateway.astream(request):
      second.append(chunk)
  finally:
    gateway.close()
  first_text = ''.join(chunk.delta_content or '' for chunk in first)
  assert 'secret' not in first_text
  assert 'visible answer' in first_text
  assert all(chunk.provider != 'cache' for chunk in first)
  assert provider.stream_calls == 1, provider.stream_calls
  assert len(second) == 1, second
  assert second[0].provider == 'cache'
  second_text = second[0].delta_content or ''
  assert 'secret' not in second_text
  assert 'visible answer' in second_text


# 6. Embed success + fetch-failure counts ---------------------------------

def test_preexisting_embed_and_fetch_failure_tests_exist() -> None:
  '''Cycle-5/6 already cover embed success and fetch_failures.'''
  cycle5 = (ROOT / 'tests/unit/test_fix_cycle5_docs.py').read_text(
    encoding='utf-8',
  )
  cycle6 = (ROOT / 'tests/unit/test_fix_cycle6_docs.py').read_text(
    encoding='utf-8',
  )
  assert 'def test_fetch_pair_adapter_raising_fetch_failure' in cycle5
  assert 'def test_compute_embeddings_success_saves_db_row' in cycle6


async def test_fetch_pair_raising_yields_fetch_failures_count(
  tmp_path: Path, monkeypatch,
) -> None:
  '''fetch_pair with raising adapter: fetch_failures==1 + NodeError.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest monkeypatch fixture.
  '''
  import job_hunter.graph.nodes as nodes_mod
  from job_hunter.graph.nodes import fetch_pair

  settings = _settings(tmp_path, monkeypatch)

  class BoomAdapter:
    '''Adapter whose fetch always raises.'''

    def __init__(self, http) -> None:
      '''Accept client.

      Args:
        http: Unused client.
      '''
      self._http = http

    async def fetch(self, target, limit=200):
      '''Always raise.

      Args:
        target: Target.
        limit: Cap.

      Returns:
        Never returns.
      '''
      raise RuntimeError('board down c13')

  monkeypatch.setattr(
    nodes_mod, 'build_adapter', lambda key, http: BoomAdapter(http),
  )
  update = await fetch_pair({
    'company_id': 7, 'name': 'Y', 'source_key': 'greenhouse',
    'board_ref': 'y',
  }, {'configurable': {'settings': settings}})
  assert update['stats']['fetch_failures'] == 1, update
  assert len(update['errors']) == 1
  assert update['errors'][0].node == 'fetch_pair'
  assert 'board down c13' in update['errors'][0].message


async def test_compute_embeddings_success_cycle13(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Fake embedder vectors persist: embeddings==1 and a DB row exists.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest monkeypatch fixture.
  '''
  import job_hunter.services.embedder as embedder_mod
  from job_hunter.core.db import run_migrations
  from job_hunter.core.models import NormalizedJob, RawJobRecord
  from job_hunter.db.repositories.jobs import JobsRepository
  from job_hunter.graph.nodes import compute_embeddings
  from job_hunter.graph.state import initial_state

  settings = _settings(tmp_path, monkeypatch)
  run_migrations(settings.db_path)
  jobs_repo = JobsRepository(settings.db_path)
  job_id = jobs_repo.insert({
    'source_key': 'lever',
    'url': 'https://jobs.example.com/embed/c13/1',
    'canonical_url': 'https://jobs.example.com/embed/c13/1',
    'company_raw_name': 'ExampleCo13',
    'title': 'Data Scientist',
    'description_text': 'python machine learning pipelines c13',
    'raw_json': '{}',
    'content_hash': 'cycle13embedhash1',
  })

  class FakeEmbedder:
    '''Fake embedder returning one fixed vector.'''

    model_id = 'fake-embed-model-c13'

    def embed(self, texts):
      '''Return one vector per input text.

      Args:
        texts: Input texts.

      Returns:
        One fixed vector per text.
      '''
      return [[0.4, 0.5, 0.6] for _ in texts]

  monkeypatch.setattr(
    embedder_mod, 'get_embedder', lambda settings: FakeEmbedder(),
  )
  state = initial_state(run_id=1, triggered_by='test')
  state['inserted'] = [NormalizedJob(
    job_id=job_id,
    raw=RawJobRecord(
      source_key='lever',
      url='https://jobs.example.com/embed/c13/1',
      title='Data Scientist',
    ),
    canonical_url='https://jobs.example.com/embed/c13/1',
    content_hash='cycle13embedhash1',
  )]
  update = await compute_embeddings(
    state, {'configurable': {'settings': settings}},
  )
  assert update['stats']['embeddings'] == 1, update
  rows = jobs_repo.load_embeddings('fake-embed-model-c13')
  assert job_id in rows
  assert rows[job_id] == pytest.approx([0.4, 0.5, 0.6])
