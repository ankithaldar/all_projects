#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle-5 docs/tests: wheel, checkpoint, persist, gateway, nodes pins.'''


from __future__ import annotations

import logging
import sqlite3
import subprocess
import sys
import zipfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _make_settings(tmp_path: Path, monkeypatch) -> object:
  '''Build AppSettings against a tmp data dir.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest monkeypatch fixture.

  Returns:
    Settings instance.
  '''
  from job_hunter.core.config import AppSettings

  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  config_path = tmp_path / 'app.yaml'
  config_path.write_text(
    'salary_hard_floor_lpa: 45\n', encoding='utf-8',
  )
  return AppSettings(str(config_path))


def _section(text: str, start: str, end: str) -> str:
  '''Extract a markdown section between two headings.

  Args:
    text: Full document text.
    start: Heading marker for the section start.
    end: Heading marker for the section end.

  Returns:
    Text between the markers.
  '''
  begin = text.index(start)
  finish = text.index(end, begin)
  return text[begin:finish]


def test_wheel_contains_runtime_assets(tmp_path: Path) -> None:
  '''Built wheel ships dashboard assets, migration, gateway config.

  Args:
    tmp_path: Pytest temp dir.
  '''
  wheel_dir = tmp_path / 'wheels'
  wheel_dir.mkdir()
  proc = subprocess.run(
    [
      sys.executable, '-m', 'pip', 'wheel', '.',
      '-w', str(wheel_dir), '--no-deps', '--no-build-isolation', '-q',
    ],
    cwd=ROOT, capture_output=True, text=True, timeout=120, check=False,
  )
  assert proc.returncode == 0, proc.stderr
  wheels = list(wheel_dir.glob('*.whl'))
  assert len(wheels) == 1, wheels
  with zipfile.ZipFile(wheels[0]) as zf:
    names = zf.namelist()
  assert any(name.endswith('dashboard.css') for name in names), names
  assert any(name.endswith('dashboard.js') for name in names), names
  assert any('0004_job_embeddings_index.sql' in name for name in names)
  assert any(name.endswith('config/gateway.yaml') for name in names)


async def test_checkpoint_missing_dep_yields_none_with_warning(
  tmp_path: Path, monkeypatch, caplog,
) -> None:
  '''Blocked checkpoint import yields None and a warning.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest monkeypatch fixture.
    caplog: Log capture fixture.
  '''
  from job_hunter.graph.checkpointing import open_checkpointer

  monkeypatch.setitem(
    sys.modules, 'langgraph.checkpoint.sqlite.aio', None,
  )
  with caplog.at_level(
    logging.WARNING, logger='job_hunter.graph.checkpointing',
  ):
    async with open_checkpointer(tmp_path / 'ckpt.db') as saver:
      assert saver is None
  assert 'checkpoint dependency missing' in caplog.text


class _RaceJobsRepo:
  '''Jobs repo fake reproducing an insert race.'''

  def __init__(self, duplicate_after_race: bool) -> None:
    '''Initialize race behaviour.

    Args:
      duplicate_after_race: When True, second exists_hash is True.
    '''
    self._duplicate_after_race = duplicate_after_race
    self._exists_calls = 0

  def exists_hash(self, digest: str) -> bool:
    '''Report duplicate only after a race when configured.

    Args:
      digest: Content hash.

    Returns:
      Whether the hash exists.
    '''
    self._exists_calls += 1
    return self._duplicate_after_race and self._exists_calls > 1

  def find_same_url(self, canonical_url: str):
    '''Never find a same-URL row.

    Args:
      canonical_url: Canonical URL.

    Returns:
      None always.
    '''
    return

  def insert(self, payload) -> int:
    '''Always raise IntegrityError to simulate a lost race.

    Args:
      payload: Job payload.

    Returns:
      Never returns.
    '''
    raise sqlite3.IntegrityError('UNIQUE constraint failed: jobs.content_hash')


_RECORD = {
  'source_key': 'lever',
  'url': 'https://example.com/cycle5/1',
  'company_name': 'Acme',
  'title': 'Data Scientist',
  'description_text': 'python ml',
}


async def test_persist_integrity_race_duplicate_vs_reraise() -> None:
  '''Lost race maps to duplicate only when the hash now exists.'''
  from job_hunter.services.persist import persist_record

  duplicate_repo = _RaceJobsRepo(duplicate_after_race=True)
  outcome = await persist_record(jobs_repo=duplicate_repo, record=dict(_RECORD))
  assert outcome == 'duplicate'

  reraising_repo = _RaceJobsRepo(duplicate_after_race=False)
  with pytest.raises(sqlite3.IntegrityError):
    await persist_record(jobs_repo=reraising_repo, record=dict(_RECORD))


async def test_persist_type_error_on_record_42() -> None:
  '''record=42 (not a mapping or RawJobRecord) raises TypeError.'''
  from job_hunter.services.persist import persist_record

  repo = _RaceJobsRepo(duplicate_after_race=False)
  with pytest.raises(TypeError):
    await persist_record(jobs_repo=repo, record=42)


class _OkJobsRepo:
  '''Minimal jobs repo whose insert always succeeds.'''

  def exists_hash(self, digest: str) -> bool:
    '''Never a duplicate.

    Args:
      digest: Content hash.

    Returns:
      False always.
    '''
    return False

  def find_same_url(self, canonical_url: str):
    '''Never find a same-URL row.

    Args:
      canonical_url: Canonical URL.

    Returns:
      None always.
    '''
    return

  def insert(self, payload) -> int:
    '''Pretend the row was written.

    Args:
      payload: Job payload.

    Returns:
      New job id.
    '''
    return 1


class _ClashingCompanyRepo:
  '''Company repo fake whose upsert always fails.'''

  def resolve_alias(self, name: str):
    '''Never resolve an alias.

    Args:
      name: Company name.

    Returns:
      None always.
    '''
    return

  def upsert(self, **kwargs) -> int:
    '''Always clash.

    Args:
      **kwargs: Upsert fields.

    Returns:
      Never returns.
    '''
    raise RuntimeError('company clash')


async def test_persist_company_upsert_failure_still_inserts() -> None:
  '''A failing company upsert does not block the job insert.'''
  from job_hunter.services.persist import persist_record

  outcome = await persist_record(
    jobs_repo=_OkJobsRepo(),
    record=dict(_RECORD),
    company_repo=_ClashingCompanyRepo(),
    create_company=True,
  )
  assert outcome == 'inserted'


async def test_verify_pending_detector_raising_needs_review(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Detector raising marks the company failed + needs_review.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest monkeypatch fixture.
  '''
  import job_hunter.services.company_discovery as discovery
  from job_hunter.core.db import run_migrations
  from job_hunter.db.repositories.companies import CompaniesRepository

  settings = _make_settings(tmp_path, monkeypatch)
  run_migrations(settings.db_path)
  repos = CompaniesRepository(settings.db_path)
  company_id = repos.upsert(name='Boom', domain='boom.com')

  class FakeHttp:
    '''Fake HttpClient with async close.'''

    async def close(self) -> None:
      '''No-op close.'''
      return

  class RaisingDetector:
    '''Detector that always raises.'''

    def __init__(self, unused_http) -> None:
      '''Accept http client.

      Args:
        unused_http: Fake http client.
      '''
      self._http = unused_http

    async def detect(self, domain, max_probes=3, name=''):
      '''Always raise.

      Args:
        domain: Company domain.
        max_probes: Probe budget (unused).
        name: Company name (unused).

      Returns:
        Never returns.
      '''
      raise RuntimeError(f'detect boom for {domain}')

  monkeypatch.setattr(discovery, 'HttpClient', FakeHttp)
  monkeypatch.setattr(discovery, 'CareerPageDetector', RaisingDetector)
  result = await discovery.verify_pending(settings, chunk=50)
  assert result == {'verified': 0, 'failed': 1}
  row = repos.get(company_id)
  assert row is not None and row['status'] == 'needs_review'
  assert row['notes'] == 'ats detection failed'


class _ChunkedDummyProvider:
  '''Dummy provider yielding fixed chunks (tags split across chunks).'''

  def __init__(self, name: str, chunks: list) -> None:
    '''Initialize chunked provider.

    Args:
      name: Provider name.
      chunks: Text pieces streamed in order.
    '''
    self.name = name
    self.chunks = chunks

  def _slices(self) -> list:
    '''Return the configured chunks.

    Returns:
      Text pieces.
    '''
    return list(self.chunks)

  def stream(self, request) -> Iterator[Any]:
    '''Yield the text in small chunks.

    Args:
      request: Provider request.

    Yields:
      Provider chunks.
    '''
    from llm_gateway.schemas import ProviderChunk

    pieces = self._slices()
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

    pieces = self._slices()
    for index, piece in enumerate(pieces):
      last = index == len(pieces) - 1
      yield ProviderChunk(
        provider=self.name,
        model=request.model,
        delta_content=piece,
        finish_reason='stop' if last else None,
      )


def _make_sanitized_gateway(tmp_path: Path, provider) -> object:
  '''Build a SanitizedLLMGateway with one dummy provider.

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
      'enabled': False,
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


def test_sanitized_gateway_stream_strips_split_think(tmp_path: Path) -> None:
  '''Sync stream strips a <think> tag split across chunks.

  Args:
    tmp_path: Pytest temp dir.
  '''
  from llm_gateway.schemas import GatewayRequest

  chunks = ['<thi', 'nk>secret reasoning</th', 'ink>visible answer']
  provider = _ChunkedDummyProvider('p1', chunks)
  gateway = _make_sanitized_gateway(tmp_path, provider)
  try:
    text = ''.join(
      chunk.delta_content or ''
      for chunk in gateway.stream(GatewayRequest(prompt='hi'))
    )
  finally:
    gateway.close()
  assert 'secret' not in text
  assert 'visible answer' in text


async def test_sanitized_gateway_astream_strips_split_think(
  tmp_path: Path,
) -> None:
  '''Async stream strips a <think> tag split across chunks.

  Args:
    tmp_path: Pytest temp dir.
  '''
  from llm_gateway.schemas import GatewayRequest

  chunks = ['<thi', 'nk>secret reasoning</th', 'ink>visible answer']
  provider = _ChunkedDummyProvider('p1', chunks)
  gateway = _make_sanitized_gateway(tmp_path, provider)
  try:
    parts = []
    async for chunk in gateway.astream(GatewayRequest(prompt='hi')):
      parts.append(chunk.delta_content or '')
    text = ''.join(parts)
  finally:
    gateway.close()
  assert 'secret' not in text
  assert 'visible answer' in text


async def test_enrich_jds_long_description_enriched(
  tmp_path: Path, monkeypatch,
) -> None:
  '''A >200-char description reaches the fake extractor.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest monkeypatch fixture.
  '''
  from job_hunter.core.db import run_migrations
  from job_hunter.core.models import EnrichedJob, NormalizedJob, RawJobRecord
  from job_hunter.db.repositories.jobs import JobsRepository
  from job_hunter.graph.nodes import enrich_jds
  from job_hunter.graph.state import initial_state

  settings = _make_settings(tmp_path, monkeypatch)
  run_migrations(settings.db_path)
  jobs_repo = JobsRepository(settings.db_path)
  description = 'python machine learning data pipelines ' * 15
  assert len(description) > 200
  job_id = jobs_repo.insert({
    'source_key': 'lever',
    'url': 'https://jobs.example.com/enrich/1',
    'canonical_url': 'https://jobs.example.com/enrich/1',
    'company_raw_name': 'ExampleCo',
    'title': 'Data Scientist',
    'description_text': description,
    'raw_json': '{}',
    'content_hash': 'cycle5enrichhash1',
  })
  row = jobs_repo.get(job_id)
  assert row is not None and len(row['description_text'] or '') > 200

  class FakeAnalyst:
    '''Fake JDAnalyst returning fixed extraction.'''

    def __init__(self, client) -> None:
      '''Accept client.

      Args:
        client: Fake gateway client.
      '''
      self._client = client

    async def extract(self, title, description_text, session_id):
      '''Return a fixed extraction.

      Args:
        title: Posting title.
        description_text: Description text.
        session_id: Correlation id.

      Returns:
        Fixed enrichment.
      '''
      _ = (title, session_id)
      assert len(description_text) > 200
      return EnrichedJob(
        must_have_skills=['Python'],
        salary_max_lpa=60.0,
        confidence=0.9,
      )

  monkeypatch.setattr(
    'job_hunter.llm.client.get_client', lambda settings: object(),
  )
  monkeypatch.setattr('job_hunter.services.jd_analyst.JDAnalyst', FakeAnalyst)

  state = initial_state(run_id=1, triggered_by='test')
  state['inserted'] = [NormalizedJob(
    job_id=job_id,
    raw=RawJobRecord(
      source_key='lever',
      url='https://jobs.example.com/enrich/1',
      title='Data Scientist',
    ),
    canonical_url='https://jobs.example.com/enrich/1',
    content_hash='cycle5enrichhash1',
  )]
  update = await enrich_jds(
    state, {'configurable': {'settings': settings}},
  )
  assert update['stats'].get('enriched', 0) >= 1, update
  assert 'cycle5enrichhash1' in update['enriched']


async def test_compute_embeddings_null_provider_skips(
  tmp_path: Path, monkeypatch,
) -> None:
  '''NullProvider embedder skips embedding with a counter.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest monkeypatch fixture.
  '''
  import job_hunter.services.embedder as embedder_mod
  from job_hunter.core.db import run_migrations
  from job_hunter.core.models import NormalizedJob, RawJobRecord
  from job_hunter.graph.nodes import compute_embeddings
  from job_hunter.graph.state import initial_state
  from job_hunter.services.embedder import NullProvider

  settings = _make_settings(tmp_path, monkeypatch)
  run_migrations(settings.db_path)
  monkeypatch.setattr(
    embedder_mod, 'get_embedder', lambda settings: NullProvider(),
  )
  state = initial_state(run_id=1, triggered_by='test')
  state['inserted'] = [NormalizedJob(
    job_id=1,
    raw=RawJobRecord(
      source_key='lever', url='https://jobs.example.com/e/1', title='DS',
    ),
    canonical_url='https://jobs.example.com/e/1',
    content_hash='embedhash1',
  )]
  update = await compute_embeddings(
    state, {'configurable': {'settings': settings}},
  )
  assert update['stats']['embeddings'] == 0
  assert update['stats']['embedding_skipped'] == 1


async def test_fetch_pair_adapter_raising_fetch_failure(
  tmp_path: Path, monkeypatch,
) -> None:
  '''An adapter raising yields fetch_failures=1 and a NodeError.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest monkeypatch fixture.
  '''
  import job_hunter.graph.nodes as nodes_mod
  from job_hunter.core.db import run_migrations
  from job_hunter.graph.nodes import fetch_pair

  settings = _make_settings(tmp_path, monkeypatch)
  run_migrations(settings.db_path)

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
      raise RuntimeError('board down')

  monkeypatch.setattr(
    nodes_mod, 'build_adapter', lambda key, http: BoomAdapter(http),
  )
  update = await fetch_pair({
    'company_id': 3, 'name': 'X', 'source_key': 'lever', 'board_ref': 'x',
  }, {'configurable': {'settings': settings}})
  assert update['stats']['fetch_failures'] == 1, update
  assert len(update['errors']) == 1
  assert update['errors'][0].node == 'fetch_pair'


def test_docs_state_snippet_matches_runtime_truth() -> None:
  '''§10.1 snippet matches graph/state.py runtime truth.'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s101 = _section(arch, '### 10.1', '### 10.2')
  assert 'run_id: int' in s101
  assert 'plan: Dict[str, Any]' in s101
  assert 'enriched: Dict[str, Any]' in s101
  assert 'inserted: List[NormalizedJob]' in s101
  assert 'RunPlan' not in s101
  assert 'normalized: list' not in s101
  assert '_normalize_inserted' in s101
  assert '_normalize_candidate' in s101


def test_docs_throttle_is_per_host_rpm() -> None:
  '''§11 names the per-(host, rpm) throttle with default 30 rpm.'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s11 = _section(arch, '## 11.', '## 12.')
  assert 'per-(host, rpm) throttle, default 30 rpm' in s11
  assert 'D15' in s11


def test_docs_resume_sizes_declared_vs_actual() -> None:
  '''Resume size errors: 422 declared, 413 actual (arch + api).'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  api = (ROOT / 'docs' / 'api.md').read_text(encoding='utf-8')
  for text in (arch, api):
    assert '422 when the declared' in text
    assert '413 when the actual' in text


def test_docs_learnings_mcp_post_deletion_truth() -> None:
  '''Learnings describe post-MCPClientManager-deletion truth.'''
  text = (ROOT / 'skills' / 'learnings.md').read_text(encoding='utf-8')
  assert 'ensure_sys_path()' in text
  assert 'standalone stdio' in text
  assert '`MCPClientManager._child_env()` injects' not in text


def test_test_writer_count_bumped() -> None:
  '''test-writer no longer claims the stale 23-module suite.'''
  text = (ROOT / 'agents' / 'test-writer.md').read_text(encoding='utf-8')
  assert '23 unit test' not in text
  assert 'tests/unit/' in text
