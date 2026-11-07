#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Direct tests for DegradedLoader and the app fallback.

Proves the real DegradedLoader class (not a hand-rolled
lambda) raises the typed error and that create_app wires
it into a degraded health + 503 predict roundtrip.
'''

import pytest

from forecast_ml_framework.serving.model_loader import (
  DegradedLoader,
  ModelNotConfiguredError,
)


def test_degraded_loader_load_raises(tmp_path):
  '''load() raises ModelNotConfiguredError, no tmp leak.

  Fails if load returns a model or leaks a path in the
  message instead of the exact generic message.
  '''
  loader = DegradedLoader()
  before = set(tmp_path.iterdir())
  with pytest.raises(ModelNotConfiguredError) as excinfo:
    loader.load()
  assert str(excinfo.value) == 'model not configured'
  assert str(tmp_path) not in str(excinfo.value)
  assert set(tmp_path.iterdir()) == before


def test_degraded_loader_call_raises():
  '''__call__ raises the same ModelNotConfiguredError.

  Fails if calling the loader returns a model.
  '''
  loader = DegradedLoader()
  with pytest.raises(ModelNotConfiguredError) as excinfo:
    loader()
  assert str(excinfo.value) == 'model not configured'


def test_create_app_degraded_roundtrip_real_loader(monkeypatch):
  '''create_app() with MODEL_* unset degrades for real.

  Fails if create_app raises on a missing loader or if
  /health and /predict do not use the real DegradedLoader.
  '''
  pytest.importorskip('fastapi')
  pytest.importorskip('httpx')
  for name in ('MODEL_ARTIFACT_PATH', 'MODEL_REGISTRY_URI',
               'MODEL_NAME', 'MODEL_ALIAS'):
    monkeypatch.delenv(name, raising=False)
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app())
  health = client.get('/health')
  assert health.status_code == 200
  assert health.json() == {
    'status': 'degraded', 'model_loaded': False}
  result = client.post(
    '/predict', json={'features': {'a': 1.0}})
  assert result.status_code == 503
