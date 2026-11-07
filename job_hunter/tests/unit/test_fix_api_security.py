#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''API security fixes: host/token guard, pagination and allowlists.'''


from __future__ import annotations

from pathlib import Path

import job_hunter.api.routes.settings as settings_mod
import job_hunter.llm.client as llm_client_mod
import pytest
from job_hunter.api.main import create_app
from job_hunter.core.config import AppSettings, is_loopback_host

fastapi = pytest.importorskip('fastapi')
TestClient = pytest.importorskip('fastapi.testclient').TestClient


def _make_client(
  tmp_path: Path, monkeypatch, *, token: str = '', host: str = '127.0.0.1',
) -> TestClient:
  '''Build an app client against a temporary data directory.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest monkeypatch fixture.
    token: Bearer token to configure.
    host: Bind host to configure.

  Returns:
    Configured TestClient.
  '''
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  monkeypatch.setenv('APP_HOST', host)
  monkeypatch.setenv('JH_API_TOKEN', token)
  monkeypatch.setenv('APP_LOG_LEVEL', 'INFO')
  config_path = tmp_path / 'app.yaml'
  config_path.write_text(
    'salary_hard_floor_lpa: 45\n', encoding='utf-8',
  )
  app = create_app(str(config_path))
  return TestClient(app, raise_server_exceptions=False)


def _put_raw(client: TestClient, url: str, raw: str):
  '''PUT a raw JSON body (allows NaN/Infinity literals).

  Args:
    client: Test client.
    url: Request URL.
    raw: Raw JSON text.

  Returns:
    Response from the client.
  '''
  headers = {'Content-Type': 'application/json'}
  return client.put(url, content=raw, headers=headers)


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


def test_evil_host_rejected_400(client: TestClient) -> None:
  '''TrustedHostMiddleware rejects an unknown Host.'''
  response = client.get('/healthz', headers={'Host': 'evil.example.com'})
  assert response.status_code == 400


def test_security_headers_and_csp_api_only(client: TestClient) -> None:
  '''API responses carry security headers and CSP.'''
  response = client.get('/api/profile')
  assert response.status_code == 200
  assert response.headers.get('x-content-type-options') == 'nosniff'
  assert response.headers.get('x-frame-options') == 'DENY'
  csp = response.headers.get('content-security-policy') or ''
  assert 'default-src' in csp
  root = client.get('/')
  if root.status_code == 200:
    root_csp = root.headers.get('content-security-policy') or ''
    assert 'default-src' not in root_csp


def test_generic_500_no_leak(client: TestClient, monkeypatch) -> None:
  '''Unhandled errors return a generic 500 without leaks.'''

  def boom(self):
    raise RuntimeError('SECRET_LEAK_xyz_123')

  monkeypatch.setattr(settings_mod.SettingsRepository, 'all_settings', boom)
  response = client.get('/api/settings')
  assert response.status_code == 500
  body = response.json()
  expected = {
    'title': 'Internal Server Error',
    'status': 500,
    'detail': 'internal error',
  }
  assert body == expected
  assert 'SECRET_LEAK' not in response.text


def test_healthz_shape_kept(client: TestClient) -> None:
  '''Health probe keeps its original shape.'''
  body = client.get('/healthz').json()
  assert body['status'] == 'ok'
  assert 'db' in body and 'gateway_db' in body and 'version' in body


def test_per_page_negative_jobs_422(client: TestClient) -> None:
  '''Negative jobs per_page is rejected.'''
  assert client.get('/api/jobs?per_page=-1').status_code == 422


def test_per_page_negative_recommendations_422(client: TestClient) -> None:
  '''Negative recommendations per_page is rejected.'''
  assert client.get('/api/recommendations?per_page=-1').status_code == 422


def test_limit_negative_runs_422(client: TestClient) -> None:
  '''Negative runs limit is rejected.'''
  assert client.get('/api/runs?limit=-1').status_code == 422


def test_per_page_over_max_422(client: TestClient) -> None:
  '''Oversized pagination values are rejected.'''
  assert client.get('/api/jobs?per_page=101').status_code == 422
  recs = client.get('/api/recommendations?per_page=101')
  assert recs.status_code == 422
  assert client.get('/api/runs?limit=101').status_code == 422


def test_runs_events_limit_over_max_422(client: TestClient) -> None:
  '''Run events limit is capped at 500.'''
  created = client.post('/api/runs/discovery')
  assert created.status_code == 200
  run_id = created.json()['run_id']
  url = f'/api/runs/{run_id}/events?limit=501'
  assert client.get(url).status_code == 422
  url_neg = f'/api/runs/{run_id}/events?limit=-1'
  assert client.get(url_neg).status_code == 422


def test_unknown_settings_key_422(client: TestClient) -> None:
  '''Unknown settings keys are rejected.'''
  url = '/api/settings?key=arbitrary_evil_key'
  assert client.put(url, json={'x': 1}).status_code == 422


def test_nan_scoring_weight_422(client: TestClient) -> None:
  '''NaN scoring weights are rejected.'''
  url = '/api/settings?key=scoring_weights'
  assert _put_raw(client, url, '{"w": NaN}').status_code == 422


def test_inf_and_non_number_scoring_weight_422(client: TestClient) -> None:
  '''Infinity and non-number weights are rejected.'''
  url = '/api/settings?key=scoring_weights'
  assert _put_raw(client, url, '{"w": Infinity}').status_code == 422
  assert client.put(url, json={'w': 'high'}).status_code == 422
  assert client.put(url, json=['not', 'object']).status_code == 422


def test_settings_salary_floor_bounds(client: TestClient) -> None:
  '''Settings salary floor must be within 0..1000.'''
  url = '/api/settings?key=salary_hard_floor_lpa'
  assert client.put(url, json=-1).status_code == 422
  assert client.put(url, json=1001).status_code == 422
  assert _put_raw(client, url, 'NaN').status_code == 422
  assert client.put(url, json=50).status_code == 200


def test_settings_discovery_caps_bounds(client: TestClient) -> None:
  '''Discovery caps must be integers within 1..10000.'''
  cap_url = '/api/settings?key=discovery.max_llm_calls_per_run'
  enrich_url = '/api/settings?key=discovery.enrich_per_run_cap'
  assert client.put(cap_url, json=0).status_code == 422
  assert client.put(enrich_url, json=10001).status_code == 422
  assert client.put(cap_url, json=10).status_code == 200


def test_import_with_client_json_body_200(client: TestClient) -> None:
  '''Import accepts the frontend JSON string body.'''
  yaml_text = (
    'companies:\n'
    '  - name: Acme Corp\n'
    '    domain: acme.com\n'
    '    priority: 3\n'
  )
  response = client.post('/api/companies/import', json=yaml_text)
  assert response.status_code == 200, response.text
  assert response.json()['imported'] == 1


def test_import_oversize_rejected(client: TestClient) -> None:
  '''Oversized import content is rejected.'''
  response = client.post('/api/companies/import', json='x' * 1_000_001)
  assert response.status_code == 422


def test_company_domain_validation(client: TestClient) -> None:
  '''Company create validates domains and rejects IPs.'''
  bad_domains = ['not a domain', '127.0.0.1', 'a@b.com']
  for index, domain in enumerate(bad_domains):
    payload = {'name': f'Evil{index}', 'domain': domain, 'priority': 3}
    assert client.post('/api/companies', json=payload).status_code == 422
  payload = {
    'name': 'Good Co', 'domain': 'good.example.com', 'priority': 3,
  }
  assert client.post('/api/companies', json=payload).status_code == 200


def test_company_patch_domain_validation(client: TestClient) -> None:
  '''Company patch validates domains.'''
  payload = {
    'name': 'Patch Co', 'domain': 'patch.example.com', 'priority': 3,
  }
  created = client.post('/api/companies', json=payload)
  assert created.status_code == 200
  company_id = created.json()['company_id']
  patch_url = f'/api/companies/{company_id}'
  bad = client.patch(patch_url, json={'domain': 'bad_domain'})
  assert bad.status_code == 422
  good = client.patch(patch_url, json={'domain': 'new.example.com'})
  assert good.status_code == 200


def test_profile_partial_preserves_omitted_fields(
  client: TestClient,
) -> None:
  '''Partial profile PUT preserves omitted fields.'''
  first = client.put('/api/profile', json={
    'target_roles': ['Data Scientist'],
    'seniority_keywords': ['senior', 'staff'],
    'blocked_verticals': ['crypto'],
    'employment_types': ['full_time', 'contract'],
    'cities': ['Bengaluru'],
  })
  assert first.status_code == 200, first.text
  partial = client.put('/api/profile', json={'target_roles': ['ML Engineer']})
  assert partial.status_code == 200, partial.text
  saved = client.get('/api/profile').json()
  assert saved['target_roles'] == ['ML Engineer']
  assert saved['seniority_keywords'] == ['senior', 'staff']
  assert saved['blocked_verticals'] == ['crypto']
  expects = ['full_time', 'contract']
  assert saved['employment_types'] == expects


def test_profile_salary_bounds(client: TestClient) -> None:
  '''Profile salary floor must be within 0..1000.'''
  assert client.put(
    '/api/profile', json={'salary_floor_lpa': -5},
  ).status_code == 422
  assert client.put(
    '/api/profile', json={'salary_floor_lpa': 2000},
  ).status_code == 422
  nan_body = '{"salary_floor_lpa": NaN}'
  assert _put_raw(client, '/api/profile', nan_body).status_code == 422
  inf_body = '{"salary_floor_lpa": Infinity}'
  assert _put_raw(client, '/api/profile', inf_body).status_code == 422


def test_resume_parse_failure_422_generic(
  client: TestClient, monkeypatch,
) -> None:
  '''Resume parse failures return generic 422 without leaks.'''

  def boom(settings):
    raise RuntimeError('SECRET_RESUME_LEAK')

  monkeypatch.setattr(llm_client_mod, 'get_client', boom)
  files = {'file': ('r.pdf', b'%PDF-1.4 fake', 'application/pdf')}
  response = client.post('/api/profile/resume', files=files)
  assert response.status_code == 422, response.text
  assert 'SECRET_RESUME_LEAK' not in response.text


def test_nonloopback_without_token_runtimeerror(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Non-loopback without a token fails fast.'''
  with pytest.raises(RuntimeError):
    _make_client(tmp_path, monkeypatch, token='', host='0.0.0.0')


def test_token_wrong_bearer_401_and_correct_ok(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Bearer token is enforced when configured.'''
  client = _make_client(
    tmp_path, monkeypatch, token='s3cret', host='0.0.0.0',
  )
  assert client.get('/api/profile').status_code == 401
  wrong = {'Authorization': 'Bearer wrong'}
  assert client.get('/api/profile', headers=wrong).status_code == 401
  right = {'Authorization': 'Bearer s3cret'}
  ok = client.get('/api/profile', headers=right)
  assert ok.status_code == 200, ok.text


def test_log_level_validation(tmp_path: Path, monkeypatch) -> None:
  '''Invalid log levels raise ValueError.'''
  monkeypatch.setenv('APP_LOG_LEVEL', 'NOPE')
  config_path = tmp_path / 'app.yaml'
  config_path.write_text(
    'salary_hard_floor_lpa: 45\n', encoding='utf-8',
  )
  settings = AppSettings(str(config_path))
  with pytest.raises(ValueError):
    _ = settings.log_level


def test_is_loopback_host_and_api_token(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Loopback helper and token env are wired correctly.'''
  monkeypatch.setenv('JH_API_TOKEN', 'tok123')
  config_path = tmp_path / 'app.yaml'
  config_path.write_text(
    'salary_hard_floor_lpa: 45\n', encoding='utf-8',
  )
  settings = AppSettings(str(config_path))
  assert settings.api_token == 'tok123'
  assert is_loopback_host('127.0.0.1')
  assert is_loopback_host('localhost')
  assert is_loopback_host('::1')
  assert not is_loopback_host('0.0.0.0')
  assert not is_loopback_host('192.168.1.5')
