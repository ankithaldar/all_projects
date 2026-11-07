#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle4 API fixes: atomic runs, AppError mapping, resume caps, imports.'''

from __future__ import annotations

import threading
from pathlib import Path

import job_hunter.llm.client as llm_client_mod
import pytest
from job_hunter.core.errors import AdapterError as _AdapterError
from job_hunter.core.errors import ConflictError as _ConflictError
from job_hunter.core.errors import DatabaseError as _DatabaseError
from job_hunter.core.errors import (
  GatewayUnavailableError as _GatewayUnavailableError,
)

fastapi = pytest.importorskip('fastapi')
TestClient = pytest.importorskip('fastapi.testclient').TestClient


def _make_client(
  tmp_path: Path, monkeypatch, *, token: str = '',
) -> TestClient:
  '''Build an app client against a temporary data directory.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest monkeypatch fixture.
    token: Bearer token to configure.

  Returns:
    Configured TestClient.
  '''
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  monkeypatch.setenv('APP_HOST', '127.0.0.1')
  monkeypatch.setenv('JH_API_TOKEN', token)
  monkeypatch.setenv('APP_LOG_LEVEL', 'INFO')
  config_path = tmp_path / 'app.yaml'
  config_path.write_text(
    'salary_hard_floor_lpa: 45\n', encoding='utf-8',
  )
  from job_hunter.api.main import create_app
  app = create_app(str(config_path))
  return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(name='client')
def _client_fixture(tmp_path: Path, monkeypatch) -> TestClient:
  '''Provide a default local-only client.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest monkeypatch fixture.

  Returns:
    Configured TestClient.
  '''
  return _make_client(tmp_path, monkeypatch)


# 1. Atomic discovery trigger ------------------------------------------


def test_runs_discovery_concurrent_one_wins(
  tmp_path: Path, monkeypatch,
) -> None:
  '''8 concurrent POSTs create exactly one pending run.'''
  client = _make_client(tmp_path, monkeypatch)
  barrier = threading.Barrier(8)
  results: list[int] = []
  lock = threading.Lock()

  def attempt() -> None:
    barrier.wait(timeout=10)
    resp = client.post('/api/runs/discovery')
    with lock:
      results.append(resp.status_code)

  threads = [threading.Thread(target=attempt) for _ in range(8)]
  for t in threads:
    t.start()
  for t in threads:
    t.join(timeout=30)
  successes = [s for s in results if s in (200, 201)]
  conflicts = [s for s in results if s == 409]
  assert len(successes) == 1, f'results={results}'
  assert len(conflicts) == 7, f'results={results}'
  from job_hunter.db.repositories.runs import RunsRepository
  runs = RunsRepository(tmp_path / 'app.db').list_runs(limit=100)
  pending = [r for r in runs if r['status'] == 'pending']
  assert len(pending) == 1, f'runs={runs}'


# 2. AppError mapping ---------------------------------------------------


def _raise_conflict(self, *args, **kwargs) -> None:
  raise _ConflictError('SECRET_CONFLICT_LEAK')


def _raise_database(self, *args, **kwargs) -> None:
  raise _DatabaseError('SECRET_DB_LEAK')


def _raise_adapter(self, *args, **kwargs) -> None:
  raise _AdapterError('SECRET_ADAPTER_LEAK')


def _raise_gateway(self, *args, **kwargs) -> None:
  raise _GatewayUnavailableError('SECRET_GATEWAY_LEAK')


def _patch_repo(monkeypatch, raiser) -> None:
  '''Monkeypatch repo methods used by candidate routes to raise.

  Args:
    monkeypatch: Pytest fixture.
    raiser: Callable raising the desired AppError.
  '''
  from job_hunter.db.repositories.runs import RunsRepository
  from job_hunter.db.repositories.settings import (
    SettingsRepository,
  )
  for cls in (RunsRepository, SettingsRepository):
    monkeypatch.setattr(cls, 'create_if_not_active', raiser, raising=False)
    monkeypatch.setattr(cls, 'create', raiser, raising=False)
    monkeypatch.setattr(cls, 'has_active', raiser, raising=False)
    monkeypatch.setattr(cls, 'list_runs', raiser, raising=False)
    monkeypatch.setattr(cls, 'all_settings', raiser, raising=False)


def test_apperror_conflict_maps_409(
  client: TestClient, monkeypatch,
) -> None:
  '''ConflictError raised by a route maps to generic 409.'''
  _patch_repo(monkeypatch, _raise_conflict)
  response = client.post('/api/runs/discovery')
  assert response.status_code == 409, response.text
  assert 'SECRET_CONFLICT' not in response.text


def test_apperror_database_maps_500(
  client: TestClient, monkeypatch,
) -> None:
  '''DatabaseError maps to generic 500 without leaking.'''
  _patch_repo(monkeypatch, _raise_database)
  response = client.post('/api/runs/discovery')
  assert response.status_code == 500, response.text
  assert 'SECRET_DB' not in response.text


def test_apperror_adapter_maps_502(
  client: TestClient, monkeypatch,
) -> None:
  '''AdapterError maps to generic 502 without leaking.'''
  _patch_repo(monkeypatch, _raise_adapter)
  response = client.post('/api/runs/discovery')
  assert response.status_code == 502, response.text
  assert 'SECRET_ADAPTER' not in response.text


def test_apperror_gateway_maps_503(
  client: TestClient, monkeypatch,
) -> None:
  '''GatewayUnavailableError maps to generic 503 without leaking.'''
  _patch_repo(monkeypatch, _raise_gateway)
  response = client.post('/api/runs/discovery')
  assert response.status_code == 503, response.text
  assert 'SECRET_GATEWAY' not in response.text


# 3. Resume Content-Length cap before read -------------------------------


def test_resume_oversize_content_length_422_before_read(
  client: TestClient, monkeypatch,
) -> None:
  '''Oversize Content-Length is rejected before curator is reached.'''
  called: list[str] = []

  def boom(_settings):
    called.append('get_client')
    raise RuntimeError('should not reach curator')

  monkeypatch.setattr(llm_client_mod, 'get_client', boom)
  files = {'file': ('r.pdf', b'%PDF-1.4 fake', 'application/pdf')}
  headers = {'Content-Length': str(11 * 1024 * 1024)}
  response = client.post(
    '/api/profile/resume', files=files, headers=headers,
  )
  assert response.status_code == 422, (
    f'status={response.status_code} body={response.text}'
  )
  assert not called, 'curator/get_client must not be reached'


# 4. Companies import bad priority counts as error ----------------------


def test_companies_import_bad_priority_counts_error(
  client: TestClient,
) -> None:
  '''Import with priority 'x' counts an error, imports the rest.'''
  yaml_text = (
    'companies:\n'
    '  - name: Bad Priority Co\n'
    '    domain: badpriority.com\n'
    "    priority: 'x'\n"
    '  - name: Good Co\n'
    '    domain: goodco.com\n'
    '    priority: 3\n'
  )
  response = client.post('/api/companies/import', json=yaml_text)
  assert response.status_code == 200, response.text
  body = response.json()
  assert body['imported'] == 1, body
  assert body['errors'] == 1, body


# 5. valid_domain dedup normalizes case ----------------------------------


def test_companies_import_uppercase_domain_accepted(
  client: TestClient,
) -> None:
  '''Import accepts mixed-case domains via base valid_domain.'''
  yaml_text = (
    'companies:\n'
    '  - name: Example Co\n'
    '    domain: Example.COM\n'
    '    priority: 3\n'
  )
  response = client.post('/api/companies/import', json=yaml_text)
  assert response.status_code == 200, response.text
  body = response.json()
  assert body['imported'] == 1, body
  assert body['errors'] == 0, body


# 6. models salary floor single-sourced ---------------------------------


def test_candidate_profile_salary_floor_uses_default_constant() -> None:
  '''CandidateProfile default comes from scorer constant.'''
  import inspect

  import job_hunter.core.models as models_mod
  from job_hunter.services.scorer import DEFAULT_SALARY_FLOOR_LPA
  source = inspect.getsource(models_mod)
  assert 'DEFAULT_SALARY_FLOOR_LPA' in source
  profile = models_mod.CandidateProfile()
  assert profile.salary_floor_lpa == DEFAULT_SALARY_FLOOR_LPA
