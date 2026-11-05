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


def test_predict_huge_int_422():
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))
  r = client.post(
      '/predict',
      content='{"features": {"a": 1' + '0' * 400 + '}}',
      headers={'Content-Type': 'application/json'})
  assert r.status_code == 422


def test_predict_unknown_feature_key_422():
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app

  class StrictModel:
    feature_names_in_ = ['a', 'b']

    def predict(self, rows):
      return [sum(row) for row in rows]

  client = TestClient(create_app(lambda: StrictModel()))
  r = client.post(
      '/predict', json={'features': {'a': 1.0, 'b': 2.0, 'c': 3.0}})
  assert r.status_code == 422
  assert 'c' in r.text


def test_predict_nonfinite_output_422():
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app

  class InfModel:
    def predict(self, rows):
      return [float('inf') for _ in rows]

  client = TestClient(create_app(lambda: InfModel()))
  r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 422
  assert 'predictions' in r.text


def test_predict_nan_output_422():
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app

  class NanModel:
    def predict(self, rows):
      return [float('nan') for _ in rows]

  client = TestClient(create_app(lambda: NanModel()))
  r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 422


def test_predict_body_too_large_413():
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))
  body = '{"features": {"a": 1.0}, "pad": "' + ('x' * (11 * 1024 * 1024)) + '"}'
  r = client.post(
      '/predict', content=body,
      headers={'Content-Type': 'application/json',
               'Content-Length': str(len(body))})
  assert r.status_code == 413

# --- cycle2 real sklearn ndarray feature_names_in_ (RED) ---
def test_predict_real_sklearn_logreg():
  '''Real sklearn LogisticRegression (ndarray feature_names_in_) serves 200.'''
  import numpy as np
  import pandas as pd
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  from sklearn.linear_model import LogisticRegression
  rng = np.random.RandomState(42)
  x = pd.DataFrame(rng.randn(60, 3), columns=['f1', 'f2', 'f3'])
  y = (x['f1'] + x['f2'] > 0).astype(int)
  model = LogisticRegression(max_iter=1000).fit(x, y)
  # feature_names_in_ is a numpy ndarray on real sklearn models
  assert hasattr(model.feature_names_in_, 'tolist')
  client = TestClient(create_app(lambda: model))
  r = client.post(
    '/predict', json={'features': {'f1': 1.0, 'f2': 2.0, 'f3': 3.0}})
  assert r.status_code == 200, r.text
  assert len(r.json()['predictions']) == 1
