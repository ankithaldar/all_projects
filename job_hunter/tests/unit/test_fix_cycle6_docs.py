#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle-6 docs/tests: wheel install, cache-stream, embed, doc pins.'''


from __future__ import annotations

import os
import subprocess
import sys
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


def test_gitignore_ignores_build() -> None:
  '''job_hunter/.gitignore ignores build/ output.'''
  lines = (ROOT / '.gitignore').read_text(encoding='utf-8').splitlines()
  assert 'build/' in [line.strip() for line in lines]


def test_wheel_install_import_and_dashboard_assets(tmp_path: Path) -> None:
  '''Wheel installs --no-deps and imports with dashboard assets.

  Args:
    tmp_path: Pytest temp dir.
  '''
  wheel_dir = tmp_path / 'wheels'
  wheel_dir.mkdir()
  wheel_proc = subprocess.run(
    [
      sys.executable, '-m', 'pip', 'wheel', '.',
      '-w', str(wheel_dir), '--no-deps', '--no-build-isolation', '-q',
    ],
    cwd=ROOT, capture_output=True, text=True, timeout=180, check=False,
  )
  assert wheel_proc.returncode == 0, wheel_proc.stderr
  wheels = list(wheel_dir.glob('*.whl'))
  assert len(wheels) == 1, wheels
  target = tmp_path / 'target'
  install_proc = subprocess.run(
    [
      sys.executable, '-m', 'pip', 'install', '--target', str(target),
      '--no-deps', '-q', str(wheels[0]),
    ],
    cwd=tmp_path, capture_output=True, text=True, timeout=180, check=False,
  )
  assert install_proc.returncode == 0, install_proc.stderr
  script = (
    'import job_hunter\n'
    'from importlib.resources import files\n'
    "base = files('llm_gateway')\n"
    "assert (base / 'templates' / 'dashboard.html').is_file()\n"
    "assert (base / 'templates' / 'dashboard.css').is_file()\n"
    "assert (base / 'templates' / 'dashboard.js').is_file()\n"
    "assert (base / 'config' / 'gateway.yaml').is_file()\n"
    "app = files('job_hunter')\n"
    "assert (app / 'db' / 'migrations'\n"
    "        / '0004_job_embeddings_index.sql').is_file()\n"
    "print('wheel-install-ok')\n"
  )
  env = dict(os.environ)
  env['PYTHONPATH'] = str(target)
  import_proc = subprocess.run(
    [sys.executable, '-c', script],
    cwd=tmp_path, capture_output=True, text=True,
    timeout=60, check=False, env=env,
  )
  assert import_proc.returncode == 0, import_proc.stderr
  assert 'wheel-install-ok' in import_proc.stdout


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
    self.stream_calls = 0

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

    self.stream_calls += 1
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

    self.stream_calls += 1
    pieces = self._slices()
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


def test_sanitized_gateway_cache_stream_second_is_cache_and_sanitized(
  tmp_path: Path,
) -> None:
  '''Cache-enabled stream twice: second is provider=cache, sanitized.

  Args:
    tmp_path: Pytest temp dir.
  '''
  from llm_gateway.schemas import GatewayRequest

  chunks = ['<thi', 'nk>secret reasoning</th', 'ink>visible answer']
  provider = _ChunkedDummyProvider('p1', chunks)
  gateway = _make_cache_gateway(tmp_path, provider)
  try:
    request = GatewayRequest(prompt='hi')
    first = list(gateway.stream(request))
    second = list(gateway.stream(request))
  finally:
    gateway.close()
  first_text = ''.join(chunk.delta_content or '' for chunk in first)
  assert 'secret' not in first_text
  assert 'visible answer' in first_text
  assert all(chunk.provider != 'cache' for chunk in first)
  assert provider.stream_calls == 1
  assert len(second) == 1
  assert second[0].provider == 'cache'
  second_text = second[0].delta_content or ''
  assert 'secret' not in second_text
  assert 'visible answer' in second_text


def test_sanitized_gateway_stream_tail_flush_trailing_lt(
  tmp_path: Path,
) -> None:
  '''Stream ending in a lone '<' emits it via the tail flush.

  Args:
    tmp_path: Pytest temp dir.
  '''
  from llm_gateway.schemas import GatewayRequest

  provider = _ChunkedDummyProvider('p1', ['hello <'])
  gateway = _make_cache_gateway(tmp_path, provider)
  try:
    text = ''.join(
      chunk.delta_content or ''
      for chunk in gateway.stream(GatewayRequest(prompt='hi'))
    )
  finally:
    gateway.close()
  assert text == 'hello<', repr(text)


async def test_compute_embeddings_success_saves_db_row(
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

  settings = _make_settings(tmp_path, monkeypatch)
  run_migrations(settings.db_path)
  jobs_repo = JobsRepository(settings.db_path)
  job_id = jobs_repo.insert({
    'source_key': 'lever',
    'url': 'https://jobs.example.com/embed/1',
    'canonical_url': 'https://jobs.example.com/embed/1',
    'company_raw_name': 'ExampleCo',
    'title': 'Data Scientist',
    'description_text': 'python machine learning pipelines',
    'raw_json': '{}',
    'content_hash': 'cycle6embedhash1',
  })

  class FakeEmbedder:
    '''Fake embedder returning one fixed vector.'''

    model_id = 'fake-embed-model'

    def embed(self, texts):
      '''Return one vector per input text.

      Args:
        texts: Input texts.

      Returns:
        One fixed vector per text.
      '''
      return [[0.1, 0.2, 0.3] for _ in texts]

  monkeypatch.setattr(
    embedder_mod, 'get_embedder', lambda settings: FakeEmbedder(),
  )
  state = initial_state(run_id=1, triggered_by='test')
  state['inserted'] = [NormalizedJob(
    job_id=job_id,
    raw=RawJobRecord(
      source_key='lever',
      url='https://jobs.example.com/embed/1',
      title='Data Scientist',
    ),
    canonical_url='https://jobs.example.com/embed/1',
    content_hash='cycle6embedhash1',
  )]
  update = await compute_embeddings(
    state, {'configurable': {'settings': settings}},
  )
  assert update['stats']['embeddings'] == 1, update
  rows = jobs_repo.load_embeddings('fake-embed-model')
  assert job_id in rows
  assert rows[job_id] == pytest.approx([0.1, 0.2, 0.3])


def test_docs_checkpoint_pragmas_in_10_1() -> None:
  '''§10.1 names the checkpoint connection pragmas.'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s101 = _section(arch, '### 10.1', '### 10.2')
  assert 'journal_mode=WAL' in s101
  assert 'busy_timeout=5000' in s101
  assert 'foreign_keys=ON' in s101


def test_docs_limit_clamps_in_section_8() -> None:
  '''§8 states the effective limit clamps per tool.'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s8 = _section(arch, '## 8.', '## 9.')
  assert 'min(limit, 300)' in s8
  assert 'min(max_results, 15)' in s8
  assert 'min(limit, 500)' in s8
  assert 'min(limit, 200)' in s8
  assert 'min(limit, 100)' in s8


def test_docs_manual_inbox_processed_move_in_t4() -> None:
  '''§11 T4 row notes the _processed/ move after scan.'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s11 = _section(arch, '## 11.', '## 12.')
  assert '_processed/' in s11


def test_docs_throttle_reservation_and_host_normalization_in_11() -> None:
  '''§11 notes throttle slot reservation + host lowercasing.'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s11 = _section(arch, '## 11.', '## 12.')
  assert 'reserv' in s11.lower()
  assert 'lower' in s11.lower()


def test_docs_tenacity_claim_replaced_with_actual_retry() -> None:
  '''No tenacity-retry claim; actual get_json/get_text truth stated.'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  assert 'tenacity' not in arch.lower()
  s20 = _section(arch, '## 20.', '## 21.')
  assert 'get_json' in s20
  assert 'get_text' in s20
  assert 'Retry-After' in s20
  assert '30' in s20


def test_docs_ua_literal_synced_and_no_d15_suffix() -> None:
  '''§11 UA matches code literal; D15 written without (c).'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s11 = _section(arch, '## 11.', '## 12.')
  assert '(local personal job search; contact: self)' in s11
  assert '(+contact: local)' not in s11
  assert 'D15(c)' not in arch
  assert 'D15' in s11


def test_docs_quarantine_claim_qualified() -> None:
  '''§20 qualifies quarantine: malformed inserted items logged+skipped.'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s20 = _section(arch, '## 20.', '## 21.')
  assert 'skipping malformed inserted item' in s20
  assert 'logged and skipped' in s20
  assert 'nothing silently dropped' not in s20


def test_docs_api_mcp2_fallback_all_three_and_chunk_note() -> None:
  '''api.md: mcp2 fallback for all 3 servers + 1 MB chunk note.'''
  api = (ROOT / 'docs' / 'api.md').read_text(encoding='utf-8')
  assert 'resume_server' in api
  assert 'store_server' in api
  assert 'sources_server' in api
  assert 'MCPServer' in api
  assert '1 MB chunks' in api or '1MB chunks' in api


def test_docs_no_stale_test_module_counts() -> None:
  '''No stale 23/40 test-module counts in docs or agent briefs.'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  writer = (ROOT / 'agents' / 'test-writer.md').read_text(encoding='utf-8')
  assert '23 unit test' not in arch
  assert '23 unit test' not in writer
  assert '40 unit test' not in writer


def test_docs_sections_18_19_still_accurate() -> None:
  '''§19 is the 6-job table (no maintenance); §18 settings row.'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  s19 = _section(arch, '## 19.', '## 20.')
  assert '6 real jobs' in s19
  assert 'no `maintenance` job' in s19
  s18 = _section(arch, '## 18.', '## 19.')
  assert 'API-only' in s18
