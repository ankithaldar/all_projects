#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle-4 P4 docs/config/tests: hermetic pins for post-fix truth.'''


from __future__ import annotations

import socket
import sqlite3
import threading
import tomllib
import urllib.error
import urllib.request
from contextlib import closing
from http.server import ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _pyproject() -> dict:
  '''Parse pyproject.toml.

  Returns:
    Parsed TOML mapping.
  '''
  return tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))


def _app_yaml_text() -> str:
  '''Read config/app.yaml as text.

  Returns:
    File text.
  '''
  return (ROOT / 'config' / 'app.yaml').read_text(encoding='utf-8')


def test_pyproject_package_data_covers_assets() -> None:
  '''setuptools package-data ships gateway + app data files.'''
  data = _pyproject()
  package_data = data['tool']['setuptools']['package-data']
  assert package_data['llm_gateway'] == ['templates/*', 'config/*']
  assert package_data['job_hunter'] == ['db/migrations/*.sql']


def test_pyproject_declares_starlette_and_typing_extensions() -> None:
  '''Direct starlette / typing_extensions imports are declared.'''
  deps = _pyproject()['project']['dependencies']
  assert any(dep.startswith('starlette>=0.37') for dep in deps), deps
  assert any(dep.startswith('typing_extensions>=4.8') for dep in deps), deps


def test_app_yaml_has_no_rate_limit_rpm_and_is_enabled_only() -> None:
  '''app.yaml sources are enabled-only; per-source RPM is adapter-owned.'''
  text = _app_yaml_text()
  assert 'rate_limit_rpm' not in text
  assert 'adapter-owned' in text
  assert 'adapters pass explicit rpm=' in text
  data = yaml.safe_load(text)
  for key, value in data['sources'].items():
    assert set(value.keys()) == {'enabled'}, (key, value)
    assert isinstance(value['enabled'], bool)


def test_docs_describe_post_fix_truth() -> None:
  '''Docs pin settings, CSRF, checkpoint, dashboard, atomic, AppError.'''
  arch = (ROOT / 'docs' / 'architecture.md').read_text(encoding='utf-8')
  api = (ROOT / 'docs' / 'api.md').read_text(encoding='utf-8')
  runbook = (ROOT / 'docs' / 'runbook.md').read_text(encoding='utf-8')
  # Settings: GET seeded read-only vs PUT 4 keys; no schedule/UI claims.
  for text in (arch, api):
    assert 'seeded' in text.lower()
    assert 'read-only' in text.lower() or 'read only' in text.lower()
  assert 'scoring_weights' in api and 'salary_hard_floor_lpa' in api
  assert 'discovery.max_llm_calls_per_run' in api
  assert 'discovery.enrich_per_run_cap' in api
  # CSRF wording, shape, Referer alternative (arch + api).
  csrf = (
    'non-GET with cross-origin Origin/Referer is rejected 403'
  )
  assert csrf in arch and csrf in api
  assert 'headerless requests not checked' in arch
  assert 'headerless requests not checked' in api
  assert "{'detail': 'forbidden origin'}" in arch or '"forbidden origin"' in arch
  assert 'Referer' in arch and 'Referer' in api
  # Checkpoint deps + conditional resume.
  assert 'langgraph-checkpoint-sqlite>=3.1' in arch
  assert 'aiosqlite>=0.20' in arch
  assert 'resume' in arch.lower() and 'install' in arch.lower()
  # Dashboard assets.
  for asset in ('/dashboard.css', '/dashboard.js', '/api/stats'):
    assert asset in arch, asset
  assert 'dashboard.html' in arch and 'dashboard.css' in arch
  assert 'dashboard.js' in arch
  # Atomic enqueue both paths.
  assert 'create_if_not_active' in arch
  assert 'BEGIN IMMEDIATE' in arch
  assert 'create_if_not_active' in api
  assert '409' in api
  # AppError mapping in api.md error section.
  assert 'ConflictError' in api
  assert '502' in api and '503' in api
  assert 'GatewayUnavailableError' in api or 'gateway unavailable' in api.lower()
  # rate_limit_rpm removed from config docs (DDL mention allowed once).
  assert arch.count('rate_limit_rpm') <= 1
  if 'rate_limit_rpm' in arch:
    assert 'INTEGER' in arch
  assert 'rate_limit_rpm' not in runbook
  # State normalization + mcp fallback mentions.
  assert 'CandidateProfile' in arch
  assert 'checkpoint' in arch.lower()
  assert 'mcp' in arch.lower()
  # runbook covers atomic/orphan + checkpoint install note.
  assert '409' in runbook or 'pending' in runbook.lower()


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
  monkeypatch.setenv('APP_LOG_LEVEL', 'INFO')
  config_path = tmp_path / 'app.yaml'
  config_path.write_text(
    'salary_hard_floor_lpa: 45\n', encoding='utf-8',
  )
  return AppSettings(str(config_path))


async def test_verify_pending_fake_detector(tmp_path: Path, monkeypatch) -> None:
  '''verify_pending patches verified rows and fails unknown ones.

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
  acme_id = repos.upsert(name='Acme', domain='acme.com')
  other_id = repos.upsert(name='Other', domain='other.com')

  class FakeHttp:
    '''Fake HttpClient with async close.'''

    async def close(self) -> None:
      '''No-op close.'''
      return

  class FakeDetector:
    '''Fake detector returning greenhouse for acme only.'''

    def __init__(self, unused_http) -> None:
      '''Accept http client.

      Args:
        unused_http: Fake http client.
      '''
      self._http = unused_http

    async def detect(self, domain, max_probes=3, name=''):
      '''Return a fingerprint for acme.com, else None.

      Args:
        domain: Company domain.
        max_probes: Probe budget (unused).
        name: Company name (unused).

      Returns:
        Fingerprint tuple or None.
      '''
      if domain == 'acme.com':
        return ('greenhouse', 'acme', 'https://acme.com/careers')
      return None

  monkeypatch.setattr(discovery, 'HttpClient', FakeHttp)
  monkeypatch.setattr(discovery, 'CareerPageDetector', FakeDetector)
  result = await discovery.verify_pending(settings, chunk=50)
  assert result == {'verified': 1, 'failed': 1}
  acme = repos.get(acme_id)
  assert acme is not None and acme['ats_provider'] == 'greenhouse'
  assert acme['board_ref'] == 'acme'
  other = repos.get(other_id)
  assert other is not None and other['status'] == 'needs_review'
  assert other['notes'] == 'ats detection failed'


class _RaceJobsRepo:
  '''Jobs repo fake for persist race paths.'''

  def __init__(self, duplicate_after_race: bool) -> None:
    '''Initialize race behaviour.

    Args:
      duplicate_after_race: When True, second exists_hash is True.
    '''
    self._duplicate_after_race = duplicate_after_race
    self._exists_calls = 0
    self.inserted = []

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
  'url': 'https://example.com/j/1',
  'company_name': 'Acme',
  'title': 'Data Scientist',
  'description_text': 'python ml',
}


async def test_persist_integrity_race_duplicate_vs_reraise() -> None:
  '''IntegrityError maps to duplicate when the hash now exists, else raises.'''
  from job_hunter.services.persist import persist_record

  duplicate_repo = _RaceJobsRepo(duplicate_after_race=True)
  outcome = await persist_record(jobs_repo=duplicate_repo, record=dict(_RECORD))
  assert outcome == 'duplicate'

  reraising_repo = _RaceJobsRepo(duplicate_after_race=False)
  with pytest.raises(sqlite3.IntegrityError):
    await persist_record(jobs_repo=reraising_repo, record=dict(_RECORD))


async def test_persist_type_error_on_bad_record() -> None:
  '''Non-dict, non-RawJobRecord records raise TypeError.'''
  from job_hunter.services.persist import persist_record

  repo = _RaceJobsRepo(duplicate_after_race=False)
  with pytest.raises(TypeError):
    await persist_record(jobs_repo=repo, record=123)


def test_csrf_host_match_allow_and_ipv6_helper(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Same-host Origin/Referer passes; cross-origin is 403; IPv6 helper works.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest monkeypatch fixture.
  '''
  from job_hunter.api.main import _host_header_host, create_app

  assert _host_header_host('[::1]:8088') == '::1'
  assert _host_header_host('[2001:db8::1]') == '2001:db8::1'
  assert _host_header_host('app.example:8088') == 'app.example'

  from fastapi.testclient import TestClient

  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  monkeypatch.setenv('APP_HOST', '127.0.0.1')
  monkeypatch.setenv('JH_API_TOKEN', '')
  config_path = tmp_path / 'app.yaml'
  config_path.write_text('salary_hard_floor_lpa: 45\n', encoding='utf-8')
  app = create_app(str(config_path))
  # TrustedHost would 400 app.example before CSRF; remove it to
  # isolate the CSRF layer under test (production keeps both).
  app.user_middleware = [
    entry for entry in app.user_middleware
    if 'TrustedHost' not in str(entry)
  ]
  app.middleware_stack = None
  client = TestClient(app, raise_server_exceptions=False)
  same = client.put(
    '/api/settings?key=salary_hard_floor_lpa',
    json=50,
    headers={'Host': 'app.example', 'Origin': 'https://app.example'},
  )
  assert same.status_code != 403, same.text
  referer = client.put(
    '/api/settings?key=salary_hard_floor_lpa',
    json=51,
    headers={
      'Host': 'app.example',
      'Referer': 'https://app.example/settings',
    },
  )
  assert referer.status_code != 403, referer.text
  cross = client.put(
    '/api/settings?key=salary_hard_floor_lpa',
    json=52,
    headers={'Host': 'app.example', 'Origin': 'https://evil.example'},
  )
  assert cross.status_code == 403
  assert cross.json() == {'detail': 'forbidden origin'}
  headerless = client.put(
    '/api/settings?key=salary_hard_floor_lpa', json=53,
  )
  assert headerless.status_code != 403, headerless.text


async def test_sanitized_gateway_acomplete_strips_think(tmp_path: Path) -> None:
  '''acomplete removes <think> blocks from displayed content.

  Args:
    tmp_path: Pytest temp dir.
  '''
  import yaml as _yaml
  from llm_gateway.sanitized_gateway import SanitizedLLMGateway
  from llm_gateway.schemas import GatewayRequest
  from llm_gateway.tests.conftest import DummyProvider

  provider = DummyProvider('p1', text='<think>secret reasoning</think>visible')
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
  config_path.write_text(_yaml.safe_dump(config), encoding='utf-8')
  env_path = tmp_path / '.env'
  env_path.write_text('', encoding='utf-8')
  gateway = SanitizedLLMGateway(
    config_path=str(config_path),
    env_path=str(env_path),
    db_path=tmp_path / 'logs.db',
    cache_path=tmp_path / 'cache.db',
    provider_overrides={'p1': provider},
  )
  try:
    response = await gateway.acomplete(GatewayRequest(prompt='hi'))
  finally:
    gateway.close()
  assert 'secret' not in response.content
  assert 'visible' in response.content


def _public_resolver(unused_host, port, *args):
  '''Fake DNS returning one public IP.

  Args:
    unused_host: Host (unused).
    port: Port.
    *args: Forwarded args.

  Returns:
    getaddrinfo-style tuples.
  '''
  return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', port or 443))]


async def test_redirect_loop_cap_six_hops() -> None:
  '''A redirect loop raises AdapterError after 6 hops.'''
  from job_hunter.adapters.http_client import _fetch_with_redirects
  from job_hunter.core.errors import AdapterError

  calls = {'count': 0}

  def handler(request: httpx.Request) -> httpx.Response:
    '''Always redirect to the loop URL.

    Args:
      request: Incoming request.

    Returns:
      302 redirect response.
    '''
    calls['count'] += 1
    return httpx.Response(
      302, headers={'Location': 'https://example.com/loop'},
    )

  transport = httpx.MockTransport(handler)
  async with httpx.AsyncClient(
    transport=transport, follow_redirects=False,
  ) as client:
    with pytest.raises(AdapterError):
      await _fetch_with_redirects(
        client, 'GET', 'https://example.com/loop',
        resolver=_public_resolver,
      )
  assert calls['count'] == 6


def _seed_gateway_db(db_path: Path) -> None:
  '''Seed llm_calls with two rows.

  Args:
    db_path: Gateway db path.
  '''
  with closing(sqlite3.connect(str(db_path))) as conn:
    conn.execute(
      '''
      CREATE TABLE llm_calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL, provider TEXT NOT NULL,
        model_used TEXT NOT NULL, input_tokens INTEGER NOT NULL,
        output_tokens INTEGER NOT NULL, latency_ms REAL NOT NULL,
        status TEXT NOT NULL, error TEXT, prompt_chars INTEGER NOT NULL,
        response_chars INTEGER NOT NULL, cost REAL NOT NULL,
        temperature REAL NOT NULL, system_prompt TEXT NOT NULL,
        session_id TEXT
      )
      '''
    )
    conn.execute(
      '''
      INSERT INTO llm_calls (
        timestamp, provider, model_used, input_tokens, output_tokens,
        latency_ms, status, prompt_chars, response_chars, cost,
        temperature, system_prompt, session_id
      ) VALUES (
        '2026-01-01T00:00:00Z', 'dummy', 'm', 10, 20, 100.0,
        'success', 5, 5, 0.5, 0.0, '', 's1'
      )
      '''
    )
    conn.commit()


def test_dashboard_stats_http_and_404(tmp_path: Path) -> None:
  '''Live dashboard serves /api/stats JSON and 404s unknown paths.

  Args:
    tmp_path: Pytest temp dir.
  '''
  import json

  from llm_gateway.dashboard import _DashboardHandler

  db_path = tmp_path / 'gateway.db'
  _seed_gateway_db(db_path)
  _DashboardHandler.db_path = str(db_path)
  server = ThreadingHTTPServer(('127.0.0.1', 0), _DashboardHandler)
  port = server.server_address[1]
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()
  try:
    base = f'http://127.0.0.1:{port}'
    with urllib.request.urlopen(base + '/api/stats') as response:
      assert response.status == 200
      payload = json.loads(response.read().decode('utf-8'))
      assert payload['summary']['total_calls'] == 1
      assert payload['summary']['input_tokens'] == 10
    with pytest.raises(urllib.error.HTTPError) as excinfo:
      urllib.request.urlopen(base + '/nope')
    assert excinfo.value.code == 404
    excinfo.value.close()
  finally:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
