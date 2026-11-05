#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for serving app.'''

import sys
import types

import pytest

pytest.importorskip('httpx')
pytest.importorskip('fastapi')


class DummyModel:
  '''Dummy model returning deterministic predictions.'''

  def predict(self, rows):
    '''Predict sum of each row.'''
    return [sum(row) for row in rows]


def dummy_loader():
  '''Return dummy model.'''
  return DummyModel()


def _install_fake_mlflow(monkeypatch):
  '''Install a fake mlflow tree; return the call log.'''
  fake = types.ModuleType('mlflow')
  fake.__path__ = []
  sklearn = types.ModuleType('mlflow.sklearn')
  pyfunc = types.ModuleType('mlflow.pyfunc')
  exceptions = types.ModuleType('mlflow.exceptions')

  class MlflowException(Exception):
    pass

  MlflowException.__module__ = 'mlflow.exceptions'
  exceptions.MlflowException = MlflowException
  calls = {'set_tracking_uri': [], 'sklearn': [], 'pyfunc': []}
  fake.set_tracking_uri = lambda uri: calls[
    'set_tracking_uri'].append(uri)

  def _sklearn_load(uri):
    calls['sklearn'].append(uri)
    return DummyModel()

  def _pyfunc_load(uri):
    calls['pyfunc'].append(uri)
    return DummyModel()

  sklearn.load_model = _sklearn_load
  pyfunc.load_model = _pyfunc_load
  fake.sklearn = sklearn
  fake.pyfunc = pyfunc
  fake.exceptions = exceptions
  monkeypatch.setitem(sys.modules, 'mlflow', fake)
  monkeypatch.setitem(sys.modules, 'mlflow.sklearn', sklearn)
  monkeypatch.setitem(sys.modules, 'mlflow.pyfunc', pyfunc)
  monkeypatch.setitem(sys.modules, 'mlflow.exceptions', exceptions)
  return calls


def test_health():
  '''Health returns status ok.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))
  response = client.get('/health')
  assert response.status_code == 200
  assert response.json() == {'status': 'ok'}


def test_predict_features():
  '''Predict accepts single features dict.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))
  response = client.post(
    '/predict', json={'features': {'a': 1.0, 'b': 2.0}})
  assert response.status_code == 200
  assert response.json() == {'predictions': [3.0]}


def test_predict_records():
  '''Predict accepts list of records.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))
  payload = {'records': [{'a': 1.0, 'b': 2.0}, {'a': 3.0, 'b': 4.0}]}
  response = client.post('/predict', json=payload)
  assert response.status_code == 200
  assert response.json() == {'predictions': [3.0, 7.0]}


def test_predict_empty_rejected():
  '''Predict rejects empty payload.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))
  response = client.post('/predict', json={})
  assert response.status_code == 422


def test_predict_missing_feature_422():
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))
  payload = {'records': [{'a': 1.0, 'b': 2.0}, {'a': 3.0}]}
  r = client.post('/predict', json=payload)
  assert r.status_code == 422
  assert 'b' in r.text


def test_predict_nested_dict_422():
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))
  r = client.post('/predict', json={'features': {'a': {'x': 1.0}}})
  assert r.status_code == 422


def test_predict_wrong_type_422():
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))
  r = client.post('/predict', json={'features': {'a': 'not-a-number'}})
  assert r.status_code == 422


def test_predict_nonfinite_422():
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))
  for token in ('NaN', 'Infinity', '-Infinity'):
    r = client.post(
        '/predict', content='{"features": {"a": %s}}' % token,
        headers={'Content-Type': 'application/json'})
    assert r.status_code == 422


def test_predict_too_many_records_413():
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))
  payload = {'records': [{'a': 1.0}] * 10001}
  r = client.post('/predict', json=payload)
  assert r.status_code == 413


def test_predict_too_many_feature_keys_413():
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))
  payload = {'features': {'f%d' % i: 1.0 for i in range(1001)}}
  r = client.post('/predict', json=payload)
  assert r.status_code == 413


def test_create_app_from_registry_health_and_predict(monkeypatch):
  '''App built from registry loader serves health and predict.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app_from_registry
  calls = _install_fake_mlflow(monkeypatch)
  client = TestClient(create_app_from_registry(
      'sqlite:///registry.db', 'my-model', alias='production'))
  assert client.get('/health').json() == {'status': 'ok'}
  r = client.post('/predict', json={'features': {'a': 1.0, 'b': 2.0}})
  assert r.status_code == 200
  assert r.json() == {'predictions': [3.0]}
  assert calls['set_tracking_uri'] == ['sqlite:///registry.db']
  assert calls['sklearn'] == ['models:/my-model@production']


def test_create_app_from_env_registry_config(monkeypatch):
  '''create_app without loader uses MODEL_* env vars.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  monkeypatch.setenv('MODEL_REGISTRY_URI', 'sqlite:///registry.db')
  monkeypatch.setenv('MODEL_NAME', 'my-model')
  monkeypatch.setenv('MODEL_ALIAS', 'production')
  calls = _install_fake_mlflow(monkeypatch)
  client = TestClient(create_app())
  assert client.get('/health').status_code == 200
  r = client.post('/predict', json={'features': {'a': 2.0}})
  assert r.json() == {'predictions': [2.0]}
  assert calls['set_tracking_uri'] == ['sqlite:///registry.db']
  assert calls['sklearn'] == ['models:/my-model@production']


def test_create_app_loader_injection_preferred_over_env(monkeypatch):
  '''Injected loader wins over MODEL_* env vars.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  monkeypatch.setenv('MODEL_REGISTRY_URI', 'sqlite:///secret.db')
  monkeypatch.setenv('MODEL_NAME', 'missing-model')
  client = TestClient(create_app(dummy_loader))
  r = client.post('/predict', json={'features': {'a': 5.0}})
  assert r.status_code == 200
  assert r.json() == {'predictions': [5.0]}


def test_registry_loader_failure_500_no_leak():
  '''Failed registry load yields generic 500 without leak.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app

  def bad_loader():
    raise RuntimeError('/secret/dir/model.pkl not found')

  client = TestClient(create_app(bad_loader))
  r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 500
  assert '/secret' not in r.text
  assert 'model.pkl' not in r.text
