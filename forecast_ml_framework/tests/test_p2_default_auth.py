#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Default deployment auth tests (FIX-ADV-1).

RED: the default Dockerfile binds 0.0.0.0 while the
module-level ``app = create_app()`` never calls
``validate_bind_host``, and the Helm chart defaults to
``env: {}`` with no auth, so POST /predict returns 200
with no key. GREEN: Helm defaults and the container
entrypoint require auth, and module-level creation is
fail-closed on a non-loopback bind without a key.
'''

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip('httpx')
pytest.importorskip('fastapi')

_AUTH_ENV_NAMES = (
    'FORECAST_API_KEY',
    'FORECAST_REQUIRE_AUTH',
    'FORECAST_ENV',
    'ENVIRONMENT',
    'APP_ENV',
    'FORECAST_OIDC_ISSUER',
    'FORECAST_OIDC_AUDIENCE',
    'FORECAST_OIDC_JWKS_URL',
    'FORECAST_BIND_HOST',
)

_VALID_KEY = 'default-auth-key-1234567890'


class DummyModel:
  '''Dummy model returning deterministic predictions.'''

  def predict(self, rows):
    '''Predict sum of each row.'''
    return [sum(row) for row in rows]


def dummy_loader():
  '''Return dummy model.'''
  return DummyModel()


def _root():
  '''Return the forecast_ml_framework project root.'''
  return Path(__file__).resolve().parents[1]


def _read(relative):
  '''Read a project-relative file as text.

  Args:
    relative: Path relative to the project root.

  Returns:
    File contents as text.
  '''
  return (_root() / relative).read_text(encoding='utf-8')


def _values():
  '''Load and return the Helm values mapping.'''
  return yaml.safe_load(_read(
      'deploy/helm/forecast-ml/values.yaml'))


def _clean_auth_env(monkeypatch):
  '''Clear every auth and bind env var for one test.

  Args:
    monkeypatch: Pytest monkeypatch fixture.
  '''
  for name in _AUTH_ENV_NAMES:
    monkeypatch.delenv(name, raising=False)


def _subprocess_env(overrides):
  '''Build a clean subprocess env for module import.

  Args:
    overrides: Mapping of extra or replacement values.

  Returns:
    Environment mapping for the subprocess.
  '''
  env = dict(os.environ)
  for name in _AUTH_ENV_NAMES:
    env.pop(name, None)
  env['PYTHONPATH'] = str(_root() / 'src')
  env.update(overrides)
  return env


def test_helm_values_require_auth_by_default():
  '''Helm defaults require auth and carry no empty env.'''
  values = _values()
  env = values['env']
  assert isinstance(env, dict)
  assert env, 'default env must not be empty'
  require = str(env.get('FORECAST_REQUIRE_AUTH', '')).lower()
  profile = str(env.get('FORECAST_ENV', '')).lower()
  assert require in ('1', 'true', 'yes', 'on') or profile in (
      'prod', 'production')
  assert 'FORECAST_API_KEY' not in env


def test_dockerfile_entrypoint_validates_bind_host():
  '''Container entrypoint validates the bind host first.'''
  text = _read('Dockerfile')
  assert 'validate_bind_host' in text
  lines = [line for line in text.splitlines()
           if line.strip().startswith('ENTRYPOINT')]
  assert lines, 'missing ENTRYPOINT line'
  entrypoint = lines[0]
  assert 'uvicorn forecast_ml_framework.serving.app:app' in entrypoint
  assert '--host 0.0.0.0' in entrypoint
  assert entrypoint.index(
      'validate_bind_host') < entrypoint.index('uvicorn')
  assert 'FORECAST_REQUIRE_AUTH' in text or (
      'FORECAST_BIND_HOST' in text)


def test_create_app_bind_env_without_key_refused(monkeypatch):
  '''Bind host from env refuses startup without a key.'''
  from forecast_ml_framework.serving.app import create_app
  _clean_auth_env(monkeypatch)
  monkeypatch.setenv('FORECAST_BIND_HOST', '0.0.0.0')
  with pytest.raises(RuntimeError, match='FORECAST_API_KEY'):
    create_app(dummy_loader)


def test_module_level_import_refused_without_key():
  '''Module-level app creation refuses a non-loopback bind.'''
  result = subprocess.run(
      [sys.executable, '-c',
       'import forecast_ml_framework.serving.app'],
      cwd=str(_root()), env=_subprocess_env(
          {'FORECAST_BIND_HOST': '0.0.0.0'}),
      capture_output=True, text=True, check=False)
  assert result.returncode != 0
  assert 'FORECAST_API_KEY' in result.stderr


def test_helm_default_env_refuses_startup_without_key(monkeypatch):
  '''Applying Helm default env refuses startup sans key.'''
  from forecast_ml_framework.serving.app import create_app
  _clean_auth_env(monkeypatch)
  for name, value in _values()['env'].items():
    monkeypatch.setenv(str(name), str(value))
  with pytest.raises(RuntimeError, match='FORECAST_API_KEY'):
    create_app(dummy_loader)


def test_predict_without_key_returns_401(monkeypatch):
  '''Authed deployment rejects unauthenticated predicts.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  _clean_auth_env(monkeypatch)
  for name, value in _values()['env'].items():
    monkeypatch.setenv(str(name), str(value))
  monkeypatch.setenv('FORECAST_API_KEY', _VALID_KEY)
  client = TestClient(create_app(dummy_loader))
  assert client.get('/health').status_code == 200
  denied = client.post('/predict', json={'features': {'a': 1.0}})
  assert denied.status_code == 401
  allowed = client.post(
      '/predict', json={'features': {'a': 1.0}},
      headers={'X-API-Key': _VALID_KEY})
  assert allowed.status_code == 200


def test_create_app_with_key_on_non_loopback_serves(monkeypatch):
  '''Non-loopback bind with a key serves normally.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  _clean_auth_env(monkeypatch)
  monkeypatch.setenv('FORECAST_BIND_HOST', '0.0.0.0')
  monkeypatch.setenv('FORECAST_API_KEY', _VALID_KEY)
  client = TestClient(create_app(dummy_loader, host='0.0.0.0'))
  response = client.post(
      '/predict', json={'features': {'a': 1.0}},
      headers={'X-API-Key': _VALID_KEY})
  assert response.status_code == 200


def test_loopback_bind_env_without_key_allowed(monkeypatch):
  '''Loopback bind from env stays usable in dev.'''
  from forecast_ml_framework.serving.app import create_app
  _clean_auth_env(monkeypatch)
  monkeypatch.setenv('FORECAST_BIND_HOST', '127.0.0.1')
  app = create_app(dummy_loader)
  assert app is not None


def test_oidc_config_exempts_bind_host_key_check(monkeypatch):
  '''Complete OIDC config exempts the shared-key check.'''
  from forecast_ml_framework.serving.app import create_app
  _clean_auth_env(monkeypatch)
  monkeypatch.setenv('FORECAST_BIND_HOST', '0.0.0.0')
  monkeypatch.setenv('FORECAST_OIDC_ISSUER', 'https://issuer.test')
  monkeypatch.setenv('FORECAST_OIDC_AUDIENCE', 'forecast-ml')
  app = create_app(dummy_loader)
  assert app is not None
