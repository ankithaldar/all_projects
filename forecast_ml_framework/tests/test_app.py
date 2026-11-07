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
    '''Mlflow Exception test double.'''
    pass

  MlflowException.__module__ = 'mlflow.exceptions'
  exceptions.MlflowException = MlflowException
  calls = {'set_tracking_uri': [], 'sklearn': [], 'pyfunc': []}
  fake.set_tracking_uri = lambda uri: calls[
    'set_tracking_uri'].append(uri)

  def _sklearn_load(uri):
    '''Load with sklearn.'''
    calls['sklearn'].append(uri)
    return DummyModel()

  def _pyfunc_load(uri):
    '''Load with pyfunc.'''
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


def _install_fake_mlflow_client(monkeypatch):
  '''Install fake mlflow client downloading a real pickle.'''
  import pickle
  import tempfile
  from pathlib import Path
  from sklearn.linear_model import LinearRegression
  fake = types.ModuleType('mlflow')
  fake.__path__ = []
  tracking = types.ModuleType('mlflow.tracking')
  exceptions = types.ModuleType('mlflow.exceptions')
  sklearn = types.ModuleType('mlflow.sklearn')
  pyfunc = types.ModuleType('mlflow.pyfunc')

  class MlflowException(Exception):
    '''Mlflow Exception test double.'''
    pass

  MlflowException.__module__ = 'mlflow.exceptions'
  exceptions.MlflowException = MlflowException
  calls = {'client': [], 'version': [], 'download': [],
           'set_tracking_uri': [], 'sklearn': [], 'pyfunc': []}
  fake.set_tracking_uri = lambda uri: calls[
      'set_tracking_uri'].append(uri)

  class _Version:
    '''Version test double.'''
    version = '3'
    run_id = 'run-1'
    source = 'fake-source'

  class _FakeClient:
    '''Fake Client test double.'''
    def __init__(self, tracking_uri=None, registry_uri=None):
      '''Initialize the test double.'''
      calls['client'].append(
          {'tracking_uri': tracking_uri,
           'registry_uri': registry_uri})

    def get_model_version_by_alias(self, name, alias):
      '''Return model version by alias.'''
      calls['version'].append((name, alias))
      return _Version()

    def download_artifacts(self, run_id, path, dst_path=None):
      '''Download artifacts.'''
      calls['download'].append((run_id, path, dst_path))
      model = LinearRegression().fit([[1.0, 2.0]], [3.0])
      base = Path(dst_path) if dst_path else Path(
          tempfile.mkdtemp())
      base.mkdir(parents=True, exist_ok=True)
      target = base / 'model.pkl'
      target.write_bytes(pickle.dumps(model))
      return str(target)

  tracking.MlflowClient = _FakeClient

  def _sklearn_load(uri):
    '''Load with sklearn.'''
    calls['sklearn'].append(uri)
    raise AssertionError('sklearn.load_model must not be used')

  sklearn.load_model = _sklearn_load
  pyfunc.load_model = lambda uri: DummyModel()
  fake.sklearn = sklearn
  fake.pyfunc = pyfunc
  fake.tracking = tracking
  fake.exceptions = exceptions
  monkeypatch.setitem(sys.modules, 'mlflow', fake)
  monkeypatch.setitem(sys.modules, 'mlflow.tracking', tracking)
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
  assert response.json() == {'status': 'ok', 'model_loaded': True}


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
  '''Predict missing feature 422.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))
  payload = {'records': [{'a': 1.0, 'b': 2.0}, {'a': 3.0}]}
  r = client.post('/predict', json=payload)
  assert r.status_code == 422
  # R-09: generic detail, no field-name echo.
  assert r.json() == {'detail': 'validation failed'}


def test_predict_nested_dict_422():
  '''Predict nested dict 422.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))
  r = client.post('/predict', json={'features': {'a': {'x': 1.0}}})
  assert r.status_code == 422


def test_predict_wrong_type_422():
  '''Predict wrong type 422.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))
  r = client.post('/predict', json={'features': {'a': 'not-a-number'}})
  assert r.status_code == 422


def test_predict_nonfinite_422():
  '''Predict nonfinite 422.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))
  for token in ('NaN', 'Infinity', '-Infinity'):
    r = client.post(
        '/predict', content='{"features": {"a": %s}}' % token,
        headers={'Content-Type': 'application/json'})
    assert r.status_code == 422


def test_predict_too_many_records_413():
  '''Predict too many records 413.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))
  payload = {'records': [{'a': 1.0}] * 10001}
  r = client.post('/predict', json=payload)
  assert r.status_code == 413


def test_predict_too_many_feature_keys_413():
  '''Predict too many feature keys 413.'''
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
  calls = _install_fake_mlflow_client(monkeypatch)
  client = TestClient(create_app_from_registry(
      'sqlite:///registry.db', 'my-model', alias='production'))
  assert client.get('/health').json() == {
      'status': 'ok', 'model_loaded': True}
  r = client.post('/predict', json={'features': {'a': 1.0, 'b': 2.0}})
  assert r.status_code == 200
  assert r.json() == {'predictions': [3.0]}
  assert calls['client'] == [{
      'tracking_uri': 'sqlite:///registry.db',
      'registry_uri': 'sqlite:///registry.db'}]
  assert calls['version'] == [('my-model', 'production')]
  assert calls['sklearn'] == []


def test_create_app_from_env_registry_config(monkeypatch):
  '''create_app without loader uses MODEL_* env vars.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  monkeypatch.setenv('MODEL_REGISTRY_URI', 'sqlite:///registry.db')
  monkeypatch.setenv('MODEL_NAME', 'my-model')
  monkeypatch.setenv('MODEL_ALIAS', 'production')
  calls = _install_fake_mlflow_client(monkeypatch)
  client = TestClient(create_app())
  assert client.get('/health').status_code == 200
  r = client.post(
      '/predict', json={'features': {'a': 1.0, 'b': 2.0}})
  assert r.json() == {'predictions': [3.0]}
  assert calls['client'] == [{
      'tracking_uri': 'sqlite:///registry.db',
      'registry_uri': 'sqlite:///registry.db'}]
  assert calls['version'] == [('my-model', 'production')]


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
    '''Return a bad loader.'''
    raise RuntimeError('/secret/dir/model.pkl not found')

  client = TestClient(create_app(bad_loader))
  r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 500
  assert '/secret' not in r.text
  assert 'model.pkl' not in r.text


def test_predict_huge_int_422():
  '''Predict huge int 422.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))
  r = client.post(
      '/predict',
      content='{"features": {"a": 1' + '0' * 400 + '}}',
      headers={'Content-Type': 'application/json'})
  assert r.status_code == 422


def test_predict_unknown_feature_key_422():
  '''Predict unknown feature key 422.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app

  class StrictModel:
    '''Strict Model test double.'''
    feature_names_in_ = ['a', 'b']

    def predict(self, rows):
      '''Return a prediction.'''
      return [sum(row) for row in rows]

  client = TestClient(create_app(lambda: StrictModel()))
  r = client.post(
      '/predict', json={'features': {'a': 1.0, 'b': 2.0, 'c': 3.0}})
  assert r.status_code == 422
  # R-09: generic detail, no field-name echo.
  assert r.json() == {'detail': 'validation failed'}
  assert 'unknown feature' not in r.text


def test_predict_nonfinite_output_500():
  '''Predict nonfinite output 500.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app

  class InfModel:
    '''Inf Model test double.'''
    def predict(self, rows):
      '''Return a prediction.'''
      return [float('inf') for _ in rows]

  client = TestClient(create_app(lambda: InfModel()))
  r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 500
  assert r.json() == {'detail': 'prediction failed'}


def test_predict_nan_output_500():
  '''Predict nan output 500.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app

  class NanModel:
    '''Nan Model test double.'''
    def predict(self, rows):
      '''Return a prediction.'''
      return [float('nan') for _ in rows]

  client = TestClient(create_app(lambda: NanModel()))
  r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 500
  assert r.json() == {'detail': 'prediction failed'}


def test_predict_body_too_large_413():
  '''Predict body too large 413.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))
  body = '{"features": {"a": 1.0}, "pad": "' + ('x' * (11 * 1024 * 1024)) + '"}'
  r = client.post(
      '/predict', content=body,
      headers={'Content-Type': 'application/json',
               'Content-Length': str(len(body))})
  assert r.status_code == 413

def test_predict_second_record_unknown_feature_422():
  '''Unknown feature in 2nd record rejected (not only first).'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))
  payload = {'records': [{'a': 1.0, 'b': 2.0},
                         {'a': 3.0, 'b': 4.0, 'evil': 5.0}]}
  r = client.post('/predict', json=payload)
  assert r.status_code == 422
  # R-09: generic detail, no field-name echo.
  assert r.json() == {'detail': 'validation failed'}
  assert 'evil' not in r.text


def test_predict_second_record_unknown_vs_model_features_422():
  '''Predict second record unknown vs model features 422.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app

  class StrictModel:
    '''Strict Model test double.'''
    feature_names_in_ = ['a', 'b']

    def predict(self, rows):
      '''Return a prediction.'''
      return [sum(row) for row in rows]

  client = TestClient(create_app(lambda: StrictModel()))
  payload = {'records': [{'a': 1.0, 'b': 2.0},
                         {'a': 3.0, 'b': 4.0, 'c': 9.0}]}
  r = client.post('/predict', json=payload)
  assert r.status_code == 422
  # R-09: generic detail, no field-name echo.
  assert r.json() == {'detail': 'validation failed'}


def test_predict_chunked_body_over_cap_413_no_content_length():
  '''Chunked stream >10MB rejected by streaming counter.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))

  def chunks():
    '''Chunks.'''
    yield b'{"features": {"a": 1.0}, "pad": "'
    yield b'x' * (6 * 1024 * 1024)
    yield b'x' * (6 * 1024 * 1024)
    yield b'"}'

  r = client.post('/predict', content=chunks(),
                  headers={'Content-Type': 'application/json'})
  assert r.status_code == 413


def test_api_key_auth_401_without_and_200_with(monkeypatch):
  '''FORECAST_API_KEY set: missing/wrong key 401, correct 200.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  monkeypatch.setenv('FORECAST_API_KEY', 's3cret-test-key-1234567890')
  client = TestClient(create_app(dummy_loader))
  payload = {'features': {'a': 1.0}}
  assert client.post('/predict', json=payload).status_code == 401
  assert client.post('/predict', json=payload,
                     headers={'X-API-Key': 'wrong'}).status_code == 401
  r = client.post('/predict', json=payload,
                  headers={'X-API-Key': 's3cret-test-key-1234567890'})
  assert r.status_code == 200
  r2 = client.post(
      '/predict', json=payload,
      headers={'Authorization':
               'Bearer s3cret-test-key-1234567890'})
  assert r2.status_code == 200
  # Health stays public for probes.
  assert client.get('/health').status_code == 200


def test_api_key_not_set_means_no_auth(monkeypatch):
  '''Api key not set means no auth.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  monkeypatch.delenv('FORECAST_API_KEY', raising=False)
  client = TestClient(create_app(dummy_loader))
  r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 200


def test_security_headers_present():
  '''Security headers present.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))
  for resp in (client.get('/health'),
               client.post('/predict',
                           json={'features': {'a': 1.0}})):
    assert resp.headers.get('x-content-type-options') == 'nosniff'
    assert resp.headers.get('x-frame-options') == 'DENY'
    assert 'no-referrer' in resp.headers.get('referrer-policy', '')
    assert resp.headers.get('cache-control') == 'no-store'
    assert 'default-src \'none\'' in resp.headers.get(
        'content-security-policy', '')
  # Also on error responses.
  r = client.post('/predict', json={})
  assert r.headers.get('x-content-type-options') == 'nosniff'


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


def _clear_model_env(monkeypatch):
  '''Clear MODEL_* env vars for degraded-loader tests.'''
  for name in ('MODEL_ARTIFACT_PATH', 'MODEL_REGISTRY_URI',
               'MODEL_NAME', 'MODEL_ALIAS'):
    monkeypatch.delenv(name, raising=False)


def test_create_app_no_loader_degraded_no_raise(monkeypatch):
  '''create_app() without loader/config degrades, never raises.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  _clear_model_env(monkeypatch)
  client = TestClient(create_app())
  health = client.get('/health')
  assert health.status_code == 200
  assert health.json() == {
      'status': 'degraded', 'model_loaded': False}
  r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 503
  assert '/secret' not in r.text


def test_predict_typed_loader_errors(monkeypatch):
  '''Typed loader errors map to 503/503/500 without leak.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  from forecast_ml_framework.serving.model_loader import (
    ModelCorruptError,
    ModelNotConfiguredError,
    ModelNotFoundError,
  )
  secret = '/secret/dir/model.pkl'

  def make(exc):
    '''Make a value.'''
    def loader():
      '''Return a loader.'''
      raise exc
    return loader

  cases = [
      (ModelNotConfiguredError(secret), 503),
      (ModelNotFoundError(secret), 503),
      (ModelCorruptError(secret), 500),
  ]
  for exc, expected in cases:
    client = TestClient(create_app(make(exc)))
    r = client.post('/predict', json={'features': {'a': 1.0}})
    assert r.status_code == expected, r.text
    assert '/secret' not in r.text
    assert 'model.pkl' not in r.text
    health = client.get('/health')
    assert health.json() == {
        'status': 'degraded', 'model_loaded': False}


def test_create_app_artifact_path_env_priority(monkeypatch, tmp_path):
  '''MODEL_ARTIFACT_PATH file loader wins over MODEL_NAME.'''
  import pickle
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  from sklearn.linear_model import LinearRegression
  model = LinearRegression().fit([[1.0, 2.0]], [3.0])
  path = tmp_path / 'model.pkl'
  path.write_bytes(pickle.dumps(model))
  monkeypatch.setenv('MODEL_ARTIFACT_PATH', str(path))
  monkeypatch.setenv('MODEL_REGISTRY_URI', 'sqlite:///registry.db')
  monkeypatch.setenv('MODEL_NAME', 'my-model')
  client = TestClient(create_app())
  assert client.get('/health').json() == {
      'status': 'ok', 'model_loaded': True}
  r = client.post('/predict', json={'features': {'a': 1.0, 'b': 2.0}})
  assert r.status_code == 200, r.text
  assert r.json() == {'predictions': [3.0]}


def test_module_level_app_exists():
  '''Module-level app is created via create_app().'''
  from forecast_ml_framework.serving import app as app_module
  assert app_module.app is not None
  assert hasattr(app_module.app, 'routes')


def test_predict_typed_loader_detail_strings():
  '''NotFound -> 'model artifact not found' 503, load fail -> 500.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  from forecast_ml_framework.serving.model_loader import (
    ModelCorruptError,
    ModelNotFoundError,
  )

  def make(exc):
    '''Make a value.'''
    def loader():
      '''Return a loader.'''
      raise exc
    return loader

  client = TestClient(create_app(make(
    ModelNotFoundError('/secret/dir/model.pkl'))))
  r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 503
  assert r.json() == {'detail': 'model artifact not found'}
  assert '/secret' not in r.text

  client = TestClient(create_app(make(
    ModelCorruptError('/secret/dir/model.pkl'))))
  r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 500
  assert r.json() == {'detail': 'model failed to load'}
  assert '/secret' not in r.text

  def bad_loader():
    '''Return a bad loader.'''
    raise RuntimeError('/secret/dir/model.pkl boom')

  client = TestClient(create_app(bad_loader))
  r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 500
  assert r.json() == {'detail': 'model failed to load'}
  assert '/secret' not in r.text


def test_predict_model_value_error_generic_500_no_leak():
  '''model.predict ValueError -> generic 500, log only, no leak.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app

  class ValueErrorModel:
    '''Value Error Model test double.'''
    def predict(self, rows):
      '''Return a prediction.'''
      raise ValueError('/secret/internal detail')

  client = TestClient(create_app(lambda: ValueErrorModel()))
  r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 500
  assert r.json() == {'detail': 'prediction failed'}
  assert '/secret' not in r.text
  assert 'internal' not in r.text


def test_predict_validation_error_still_422():
  '''_validate_records ValueError stays 422 with generic detail.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))
  r = client.post('/predict', json={'features': {'a': 'x'}})
  assert r.status_code == 422
  assert r.json() == {'detail': 'validation failed'}


def test_health_caches_probe_on_app_instance():
  '''Second /health does not re-probe loader; cache on app.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  calls = {'n': 0}

  def counting_loader():
    '''Return a counting loader.'''
    calls['n'] += 1
    return DummyModel()

  app = create_app(counting_loader)
  client = TestClient(app)
  assert client.get('/health').json() == {
    'status': 'ok', 'model_loaded': True}
  assert calls['n'] == 1
  assert client.get('/health').json() == {
    'status': 'ok', 'model_loaded': True}
  assert calls['n'] == 1
  # Cache lives on the app instance (state or attribute).
  cached = (getattr(app.state, 'health_cache', None)
            or getattr(app, '_health_cache', None)
            or getattr(app, 'health_cache', None))
  assert cached is not None


# --- loop6 dict predictions (RED) ---

def test_check_predictions_finite_dict_values():
  '''Check predictions finite dict values.'''
  import pytest
  from forecast_ml_framework.serving.app import _check_predictions_finite
  with pytest.raises(ValueError, match='finite'):
    _check_predictions_finite({'a': float('inf')})
  with pytest.raises(ValueError, match='finite'):
    _check_predictions_finite({'a': [1.0, float('nan')]})
  with pytest.raises(ValueError, match='finite'):
    _check_predictions_finite([{'a': float('-inf')}])
  _check_predictions_finite({'a': [1.0, 2.0]})


def test_predict_dict_nonfinite_output_500():
  '''Predict dict nonfinite output 500.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app

  class DictInfModel:
    '''Dict Inf Model test double.'''
    def predict(self, rows):
      '''Return a prediction.'''
      return {'pred': [float('inf')]}

  client = TestClient(create_app(lambda: DictInfModel()))
  r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 500
  assert r.json() == {'detail': 'prediction failed'}


# --- loop7 serving finite (RED) ---

def test_check_predictions_finite_numpy_decimal_complex():
  '''Check predictions finite numpy decimal complex.'''
  import numpy as np
  from decimal import Decimal
  from forecast_ml_framework.serving.app import _check_predictions_finite
  bad = [
    np.float32(float('inf')),
    np.float64(float('nan')),
    np.array([1.0, np.float32(float('inf'))]),
    np.array([Decimal('Infinity')], dtype=object),
    Decimal('Infinity'),
    Decimal('NaN'),
    complex(float('inf'), 0.0),
    complex(0.0, float('nan')),
    np.complex128(complex(float('inf'), 0.0)),
    {'a': np.float32(float('nan'))},
    {'a': Decimal('-Infinity')},
    {'a': [complex(0.0, float('inf'))]},
    {'a': np.array([1.0, float('inf')])},
    [np.float32(float('-inf'))],
  ]
  for value in bad:
    with pytest.raises(ValueError, match='finite'):
      _check_predictions_finite(value)
  _check_predictions_finite(np.float32(1.0))
  _check_predictions_finite(np.array([1.0, 2.0]))
  _check_predictions_finite(Decimal('1.5'))
  _check_predictions_finite(complex(1.0, 2.0))
  _check_predictions_finite({'a': np.float32(1.0)})


# --- loop8 None/pd.NA/non-string leaf (RED) ---

def test_check_predictions_finite_none_and_pd_na():
  '''Check predictions finite none and pd na.'''
  import numpy as np
  import pandas as pd
  from forecast_ml_framework.serving.app import _check_predictions_finite
  bad = [
    None,
    [None],
    {'a': None},
    [pd.NA],
    {'a': pd.NA},
    pd.NA,
    np.array([None], dtype=object),
    np.array([pd.NA], dtype=object),
    [pd.NaT],
    b'bytes',
    [object()],
  ]
  for value in bad:
    with pytest.raises(ValueError, match='finite'):
      _check_predictions_finite(value)
  _check_predictions_finite(['cat', 'dog'])
  _check_predictions_finite({'a': 'cat'})
  _check_predictions_finite('cat')


def test_predict_none_output_500():
  '''Predict none output 500.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app

  class NoneModel:
    '''None Model test double.'''
    def predict(self, rows):
      '''Return a prediction.'''
      return [None for _ in rows]

  client = TestClient(create_app(lambda: NoneModel()))
  r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 500
  assert r.json() == {'detail': 'prediction failed'}


def test_predict_string_label_output_200():
  '''Predict string label output 200.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app

  class LabelModel:
    '''Label Model test double.'''
    def predict(self, rows):
      '''Return a prediction.'''
      return ['cat' for _ in rows]

  client = TestClient(create_app(lambda: LabelModel()))
  r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 200
  assert r.json() == {'predictions': ['cat']}


# --- loop9 masked/generator (RED) ---

def test_check_predictions_finite_masked_constant_raises():
  '''Check predictions finite masked constant raises.'''
  import numpy as np
  from forecast_ml_framework.serving.app import _check_predictions_finite
  with pytest.raises(ValueError, match='finite'):
    _check_predictions_finite(np.ma.masked)
  with pytest.raises(ValueError, match='finite'):
    _check_predictions_finite([np.ma.masked])
  with pytest.raises(ValueError, match='finite'):
    _check_predictions_finite({'a': np.ma.masked})


def test_check_predictions_finite_masked_array_raises_when_masked():
  '''Check predictions finite masked array raises when masked.'''
  import numpy as np
  from forecast_ml_framework.serving.app import _check_predictions_finite
  masked = np.ma.array([1.0, 2.0], mask=[False, True])
  with pytest.raises(ValueError, match='finite'):
    _check_predictions_finite(masked)
  with pytest.raises(ValueError, match='finite'):
    _check_predictions_finite([masked])
  clean = np.ma.array([1.0, 2.0], mask=[False, False])
  _check_predictions_finite(clean)
  with pytest.raises(ValueError, match='finite'):
    _check_predictions_finite(
      np.ma.array([1.0, float('inf')], mask=[False, False]))


def test_predict_generator_predictions_return_list():
  '''Predict generator predictions return list.'''
  from forecast_ml_framework.serving.app import _predict

  class GenModel:
    '''Gen Model test double.'''
    def predict(self, rows):
      '''Return a prediction.'''
      return (float(v) for v in [1.0, 2.0][:len(rows)])

  assert _predict(GenModel(), [{'a': 1.0}, {'a': 2.0}]) == [1.0, 2.0]


def test_predict_generator_http_200_not_empty():
  '''Predict generator http 200 not empty.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app

  class GenModel:
    '''Gen Model test double.'''
    def predict(self, rows):
      '''Return a prediction.'''
      return (sum(row) for row in rows)

  client = TestClient(create_app(lambda: GenModel()))
  r = client.post('/predict', json={'features': {'a': 3.0}})
  assert r.status_code == 200, r.text
  assert r.json() == {'predictions': [3.0]}


def test_predict_masked_output_500():
  '''Predict masked output 500.'''
  import numpy as np
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app

  class MaskedModel:
    '''Masked Model test double.'''
    def predict(self, rows):
      '''Return a prediction.'''
      return np.ma.masked

  client = TestClient(create_app(lambda: MaskedModel()))
  r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 500
  assert r.json() == {'detail': 'prediction failed'}


# --- W3 RED: fail-closed auth ---

def test_require_auth_no_key_raises_at_startup(monkeypatch):
  '''FORECAST_REQUIRE_AUTH=1 without key raises RuntimeError.'''
  from forecast_ml_framework.serving.app import create_app
  monkeypatch.setenv('FORECAST_REQUIRE_AUTH', '1')
  monkeypatch.delenv('FORECAST_API_KEY', raising=False)
  monkeypatch.delenv('FORECAST_ENV', raising=False)
  monkeypatch.delenv('ENVIRONMENT', raising=False)
  monkeypatch.delenv('APP_ENV', raising=False)
  with pytest.raises(RuntimeError, match='FORECAST_API_KEY'):
    create_app(dummy_loader)


def test_prod_profile_no_key_raises_at_startup(monkeypatch):
  '''Production profile without key raises RuntimeError.'''
  from forecast_ml_framework.serving.app import create_app
  monkeypatch.setenv('FORECAST_ENV', 'prod')
  monkeypatch.delenv('FORECAST_API_KEY', raising=False)
  monkeypatch.delenv('FORECAST_REQUIRE_AUTH', raising=False)
  with pytest.raises(RuntimeError, match='FORECAST_API_KEY'):
    create_app(dummy_loader)


def test_unset_key_dev_logs_warning(monkeypatch, caplog):
  '''Dev without key still serves but logs a WARNING.'''
  import logging
  from forecast_ml_framework.serving.app import create_app
  monkeypatch.delenv('FORECAST_API_KEY', raising=False)
  monkeypatch.delenv('FORECAST_REQUIRE_AUTH', raising=False)
  monkeypatch.delenv('FORECAST_ENV', raising=False)
  monkeypatch.delenv('ENVIRONMENT', raising=False)
  monkeypatch.delenv('APP_ENV', raising=False)
  with caplog.at_level(logging.WARNING):
    app = create_app(dummy_loader)
  assert app is not None
  assert any(
      record.levelno == logging.WARNING
      and 'FORECAST_API_KEY' in record.getMessage()
      for record in caplog.records)


def test_short_api_key_raises_at_startup(monkeypatch):
  '''API key shorter than the minimum is refused at startup.'''
  from forecast_ml_framework.serving.app import create_app
  monkeypatch.setenv('FORECAST_API_KEY', 'short')
  with pytest.raises(RuntimeError, match='FORECAST_API_KEY'):
    create_app(dummy_loader)


def test_non_loopback_bind_without_key_in_prod_refuses(monkeypatch):
  '''--host 0.0.0.0 without key in prod is refused.'''
  from forecast_ml_framework.serving.app import create_app
  monkeypatch.setenv('FORECAST_ENV', 'prod')
  monkeypatch.delenv('FORECAST_API_KEY', raising=False)
  with pytest.raises(RuntimeError):
    create_app(dummy_loader, host='0.0.0.0')


def test_non_loopback_bind_helper_refuses_without_key(monkeypatch):
  '''validate_bind_host refuses non-loopback without a key.'''
  from forecast_ml_framework.serving.app import validate_bind_host
  monkeypatch.delenv('FORECAST_API_KEY', raising=False)
  monkeypatch.delenv('FORECAST_REQUIRE_AUTH', raising=False)
  monkeypatch.delenv('FORECAST_ENV', raising=False)
  with pytest.raises(RuntimeError):
    validate_bind_host('0.0.0.0')
  validate_bind_host('127.0.0.1')


# --- W3 RED: scrubbed exception logging ---

def test_secret_exception_never_in_caplog(monkeypatch, caplog):
  '''Secret-bearing loader exception never reaches logs.'''
  import logging
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  secret = 'supersecretpassword12345'

  def bad_loader():
    '''Return a bad loader.'''
    raise RuntimeError(f'password={secret} /secret/dir/model.pkl')

  monkeypatch.delenv('FORECAST_API_KEY', raising=False)
  monkeypatch.delenv('FORECAST_REQUIRE_AUTH', raising=False)
  monkeypatch.delenv('FORECAST_ENV', raising=False)
  with caplog.at_level(logging.DEBUG):
    client = TestClient(create_app(bad_loader))
    r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 500
  assert secret not in caplog.text
  assert '/secret' not in caplog.text
  assert 'model.pkl' not in caplog.text


def test_secret_predict_exception_never_in_caplog(monkeypatch, caplog):
  '''Secret-bearing predict exception never reaches logs.'''
  import logging
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  secret = 'supersecretpassword99999'

  class SecretModel:
    '''Secret Model test double.'''
    def predict(self, rows):
      '''Return a prediction.'''
      raise ValueError(f'token={secret} internal-detail')

  monkeypatch.delenv('FORECAST_API_KEY', raising=False)
  monkeypatch.delenv('FORECAST_REQUIRE_AUTH', raising=False)
  monkeypatch.delenv('FORECAST_ENV', raising=False)
  with caplog.at_level(logging.DEBUG):
    client = TestClient(create_app(lambda: SecretModel()))
    r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 500
  assert secret not in caplog.text
  assert 'internal-detail' not in caplog.text


# --- W3 RED: rate limiting ---

def test_rate_limit_429_on_predict(monkeypatch):
  '''Exceeding the per-minute rate limit returns 429.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  monkeypatch.delenv('FORECAST_API_KEY', raising=False)
  monkeypatch.delenv('FORECAST_REQUIRE_AUTH', raising=False)
  monkeypatch.delenv('FORECAST_ENV', raising=False)
  monkeypatch.setenv('FORECAST_RATE_LIMIT_PER_MINUTE', '3')
  client = TestClient(create_app(dummy_loader))
  codes = [
      client.post(
          '/predict', json={'features': {'a': 1.0}}).status_code
      for _ in range(5)
  ]
  assert 429 in codes
  assert codes[:3] == [200, 200, 200]


# --- W3 RED: HSTS + access log + health refresh ---

def test_hsts_header_when_tls(monkeypatch):
  '''HTTPS responses carry Strict-Transport-Security.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  monkeypatch.delenv('FORECAST_API_KEY', raising=False)
  monkeypatch.delenv('FORECAST_REQUIRE_AUTH', raising=False)
  monkeypatch.delenv('FORECAST_ENV', raising=False)
  client = TestClient(
      create_app(dummy_loader), base_url='https://testserver')
  r = client.get('/health')
  assert 'max-age=' in r.headers.get(
      'strict-transport-security', '')
  assert r.headers.get('x-content-type-options') == 'nosniff'


def test_access_log_hashed_key_id_no_raw_key(monkeypatch, caplog):
  '''/predict access log hashes key-id and omits raw key.'''
  import hashlib
  import hmac
  import logging
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  key = 's3cret-test-key-1234567890'
  monkeypatch.setenv('FORECAST_API_KEY', key)
  monkeypatch.delenv('FORECAST_REQUIRE_AUTH', raising=False)
  monkeypatch.delenv('FORECAST_ENV', raising=False)
  # R-04: key-id is HMAC-SHA256 keyed by the server secret.
  hashed = hmac.new(
      key.encode(), key.encode(), hashlib.sha256).hexdigest()[:16]
  with caplog.at_level(logging.INFO):
    client = TestClient(create_app(dummy_loader))
    r = client.post(
        '/predict', json={'features': {'a': 1.0}},
        headers={'X-API-Key': key})
  assert r.status_code == 200
  assert 'predict_access' in caplog.text
  assert hashed in caplog.text
  assert key not in caplog.text
  assert 'outcome=success' in caplog.text


def test_health_cache_refresh_after_ttl(monkeypatch):
  '''Health cache re-probes after its TTL expires.'''
  import time
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  monkeypatch.delenv('FORECAST_API_KEY', raising=False)
  monkeypatch.delenv('FORECAST_REQUIRE_AUTH', raising=False)
  monkeypatch.delenv('FORECAST_ENV', raising=False)
  calls = {'n': 0}

  def counting_loader():
    '''Return a counting loader.'''
    calls['n'] += 1
    return DummyModel()

  app = create_app(counting_loader)
  client = TestClient(app)
  assert client.get('/health').json()['status'] == 'ok'
  assert calls['n'] == 1
  cache = app.state.health_cache
  cache['probed_at'] = time.monotonic() - 1000.0
  assert client.get('/health').json()['status'] == 'ok'
  assert calls['n'] == 2


def test_health_cache_refresh_on_predict_success(monkeypatch):
  '''Successful /predict refreshes a degraded health cache.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  monkeypatch.delenv('FORECAST_API_KEY', raising=False)
  monkeypatch.delenv('FORECAST_REQUIRE_AUTH', raising=False)
  monkeypatch.delenv('FORECAST_ENV', raising=False)
  state = {'fail': True}

  def flaky_loader():
    '''Flaky loader.'''
    if state['fail']:
      raise RuntimeError('not yet')
    return DummyModel()

  client = TestClient(create_app(flaky_loader))
  assert client.get('/health').json() == {
      'status': 'degraded', 'model_loaded': False}
  state['fail'] = False
  r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 200
  assert client.get('/health').json() == {
      'status': 'ok', 'model_loaded': True}


# --- X1 RED: audit persistence, 413 logging, generic 422 ---

@pytest.mark.parametrize(
    'payload,marker', [
        ({'records': [{'a': 1.0, 'b': 2.0}, {'a': 3.0}]}, 'b'),
        ({'records': [{'a': 1.0, 'b': 2.0},
                      {'a': 3.0, 'b': 4.0, 'evil_field': 5.0}]},
         'evil_field'),
        ({'features': {'a': 'not-a-number'}}, 'a'),
    ])
def test_validation_422_generic_no_field_echo(payload, marker):
  '''Fails if a 422 response echoes the offending field name.

  Mutation kill: restoring ``detail=str(exc)`` in the predict
  endpoint must make this test fail ('validation failed' gone
  and the field marker present in the body).
  '''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))
  r = client.post('/predict', json=payload)
  assert r.status_code == 422
  assert r.json() == {'detail': 'validation failed'}
  assert 'unknown feature' not in r.text
  assert 'missing feature' not in r.text
  assert 'must be a finite number' not in r.text
  # Single-char markers ('a','b') can appear spuriously; only
  # assert absence for distinctive markers.
  if len(marker) > 1:
    assert marker not in r.text


def test_predict_413_leaves_access_and_audit_entries(
    monkeypatch, caplog):
  '''_check_caps 413 writes access log AND audit entry.'''
  import json
  import logging
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  key = 's3cret-test-key-1234567890'
  monkeypatch.setenv('FORECAST_API_KEY', key)
  monkeypatch.delenv('FORECAST_REQUIRE_AUTH', raising=False)
  monkeypatch.delenv('FORECAST_ENV', raising=False)
  app = create_app(dummy_loader)
  client = TestClient(app)
  with caplog.at_level(logging.INFO):
    r = client.post(
        '/predict', json={'records': [{'a': 1.0}] * 10001},
        headers={'X-API-Key': key})
  assert r.status_code == 413
  assert 'predict_access' in caplog.text
  assert 'outcome=payload_too_large' in caplog.text
  audit = app.state.audit_log
  assert audit
  entry = audit[-1]
  assert entry['event'] == 'predict'
  assert entry['actor'] == 'serving'
  assert entry['details']['outcome'] == 'payload_too_large'
  assert entry['details']['records'] == 10001
  assert 'latency_ms' in entry['details']
  assert 'key_id_hmac' in entry['details']
  blob = json.dumps(entry)
  assert key not in blob
  assert key not in caplog.text


def test_middleware_body_413_leaves_access_and_audit_entries(
    monkeypatch, caplog):
  '''Streaming body-cap 413 writes access log AND audit entry.'''
  import json
  import logging
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  key = 's3cret-test-key-1234567890'
  monkeypatch.setenv('FORECAST_API_KEY', key)
  monkeypatch.delenv('FORECAST_REQUIRE_AUTH', raising=False)
  monkeypatch.delenv('FORECAST_ENV', raising=False)
  app = create_app(dummy_loader)
  client = TestClient(app)
  body = '{"features": {"a": 1.0}, "pad": "' + ('x' * (11 * 1024 * 1024)) + '"}'
  with caplog.at_level(logging.INFO):
    r = client.post(
        '/predict', content=body,
        headers={'Content-Type': 'application/json',
                 'X-API-Key': key,
                 'Content-Length': str(len(body))})
  assert r.status_code == 413
  assert 'predict_access' in caplog.text
  assert 'outcome=payload_too_large' in caplog.text
  audit = app.state.audit_log
  assert audit
  entry = audit[-1]
  assert entry['event'] == 'predict'
  assert entry['actor'] == 'serving'
  assert entry['details']['outcome'] == 'payload_too_large'
  assert key not in json.dumps(entry)


def test_predict_success_writes_audit_entry_no_raw_key(
    monkeypatch):
  '''Successful /predict persists an audit entry without raw key.'''
  import json
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  key = 's3cret-test-key-1234567890'
  monkeypatch.setenv('FORECAST_API_KEY', key)
  monkeypatch.delenv('FORECAST_REQUIRE_AUTH', raising=False)
  monkeypatch.delenv('FORECAST_ENV', raising=False)
  app = create_app(dummy_loader)
  client = TestClient(app)
  r = client.post(
      '/predict', json={'features': {'a': 1.0}},
      headers={'X-API-Key': key})
  assert r.status_code == 200
  audit = app.state.audit_log
  assert audit
  entry = audit[-1]
  assert entry['details']['outcome'] == 'success'
  assert entry['details']['records'] == 1
  assert key not in json.dumps(audit)


def test_hmac_key_id_differs_per_key_and_hides_raw(monkeypatch):
  '''HMAC key-id differs per key and never equals the raw key.'''
  from forecast_ml_framework.serving.app import _hmac_key_id
  monkeypatch.setenv('FORECAST_API_KEY', 'server-secret-1234567890')
  first = _hmac_key_id('client-key-aaaaaaaa')
  second = _hmac_key_id('client-key-bbbbbbbb')
  assert first != second
  assert first != 'client-key-aaaaaaaa'
  assert 'client-key' not in first
  assert _hmac_key_id('') == 'anonymous'


# --- N1 S3-R2 RED: schema 422 repudiation ---
def test_predict_schema_422_leaves_access_and_audit_entries(
    monkeypatch, caplog):
  '''Schema-invalid body (pydantic 422) leaves access+audit entries.

  Fails if middleware does not log FastAPI/pydantic schema 422s —
  repudiation must cover malformed-schema attempts that never
  reach the endpoint.
  '''
  import json
  import logging
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  key = 's3cret-test-key-1234567890'
  monkeypatch.setenv('FORECAST_API_KEY', key)
  monkeypatch.delenv('FORECAST_REQUIRE_AUTH', raising=False)
  monkeypatch.delenv('FORECAST_ENV', raising=False)
  app = create_app(dummy_loader)
  client = TestClient(app)
  with caplog.at_level(logging.INFO):
    r = client.post(
        '/predict', json={'features': 123},
        headers={'X-API-Key': key})
  assert r.status_code == 422
  assert 'predict_access' in caplog.text
  assert 'outcome=validation_error' in caplog.text
  audit = app.state.audit_log
  assert audit
  entry = audit[-1]
  assert entry['event'] == 'predict'
  assert entry['actor'] == 'serving'
  assert entry['details']['outcome'] == 'validation_error'
  assert key not in json.dumps(entry)
  assert key not in caplog.text


# --- C4-R1 RED: per-request scope marker dedup ---


def test_middleware_schema_422_logged_despite_concurrent_growth(
    monkeypatch, caplog):
  '''Concurrent growth must not mask a schema-422 log.

  Simulates a concurrent success appending to the audit
  list between the middleware snapshot and its 422 check
  (list growth without endpoint logging). The schema 422
  must still leave exactly one access + one audit entry.
  Fails under length-based dedup (lost log); passes with
  the scope-marker check.
  '''
  import asyncio
  import logging
  from forecast_ml_framework.serving.app import (
    _CapAuthHeadersMiddleware,
  )
  monkeypatch.delenv('FORECAST_API_KEY', raising=False)
  monkeypatch.delenv('FORECAST_REQUIRE_AUTH', raising=False)
  monkeypatch.delenv('FORECAST_ENV', raising=False)
  audit = []

  async def downstream(scope, receive, send):
    # Concurrent success grows the list during downstream.
    '''Downstream.'''
    audit.append({
      'event': 'predict', 'actor': 'serving',
      'details': {'outcome': 'success',
                  'key_id_hmac': 'concurrent',
                  'latency_ms': 1.0, 'records': 1},
      'timestamp': '2026-01-01T00:00:00+00:00'})
    await send({'type': 'http.response.start',
                'status': 422, 'headers': []})
    await send({'type': 'http.response.body',
                'body': b'{"detail": "validation failed"}'})

  middleware = _CapAuthHeadersMiddleware(
    downstream, rate_limit_per_minute=600, audit_log=audit)
  scope = {'type': 'http', 'path': '/predict',
           'method': 'POST', 'headers': [], 'scheme': 'http'}

  async def receive():
    '''Receive.'''
    return {'type': 'http.request', 'body': b'{}',
            'more_body': False}

  sent = []

  async def send(message):
    '''Send.'''
    sent.append(message)

  with caplog.at_level(logging.INFO):
    asyncio.run(middleware(scope, receive, send))
  assert sent[0]['status'] == 422
  accesses = [r for r in caplog.records
              if 'predict_access' in r.getMessage()]
  assert len(accesses) == 1
  validation = [e for e in audit
                if e['details']['outcome'] == 'validation_error']
  assert len(validation) == 1
  assert len(audit) == 2


def test_saturated_audit_endpoint_422_single_access_and_audit(
    monkeypatch, caplog):
  '''Saturated audit list: endpoint 422 logs exactly once.

  With the list at MAX_AUDIT_ENTRIES (append+trim leaves
  len unchanged) length-based dedup double-logs. Exactly
  one access log and one validation_error audit entry
  must remain; the list stays capped.
  '''
  import logging
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import (
    MAX_AUDIT_ENTRIES,
    create_app,
  )
  key = 's3cret-test-key-1234567890'
  monkeypatch.setenv('FORECAST_API_KEY', key)
  monkeypatch.delenv('FORECAST_REQUIRE_AUTH', raising=False)
  monkeypatch.delenv('FORECAST_ENV', raising=False)
  app = create_app(dummy_loader)
  audit = app.state.audit_log
  for _ in range(MAX_AUDIT_ENTRIES):
    audit.append({
      'event': 'predict', 'actor': 'serving',
      'details': {'outcome': 'seed', 'key_id_hmac': 'seed',
                  'latency_ms': 0.0, 'records': 0},
      'timestamp': '2026-01-01T00:00:00+00:00'})
  assert len(audit) == MAX_AUDIT_ENTRIES
  client = TestClient(app)
  with caplog.at_level(logging.INFO):
    r = client.post(
      '/predict', json={'features': {'a': 'not-a-number'}},
      headers={'X-API-Key': key})
  assert r.status_code == 422
  accesses = [row for row in caplog.records
              if 'predict_access' in row.getMessage()]
  assert len(accesses) == 1
  validation = [e for e in audit
                if e['details']['outcome'] == 'validation_error']
  assert len(validation) == 1
  assert len(audit) == MAX_AUDIT_ENTRIES


def test_audit_failure_middleware_does_not_double_log(
    monkeypatch, caplog):
  '''Audit failure: endpoint 422 leaves single access log.

  When audit persistence raises, the list does not grow;
  length-based dedup lets the middleware log a second
  access. Exactly one access log must remain and no
  audit entry is persisted.
  '''
  import logging
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  key = 's3cret-test-key-1234567890'
  monkeypatch.setenv('FORECAST_API_KEY', key)
  monkeypatch.delenv('FORECAST_REQUIRE_AUTH', raising=False)
  monkeypatch.delenv('FORECAST_ENV', raising=False)

  def failing_audit_log(*args, **kwargs):
    '''Failing audit log.'''
    raise RuntimeError('audit sink down')

  monkeypatch.setattr(
    'forecast_ml_framework.governance.audit_log',
    failing_audit_log)
  app = create_app(dummy_loader)
  client = TestClient(app)
  with caplog.at_level(logging.INFO):
    r = client.post(
      '/predict', json={'features': {'a': 'not-a-number'}},
      headers={'X-API-Key': key})
  assert r.status_code == 422
  accesses = [row for row in caplog.records
              if 'predict_access' in row.getMessage()]
  assert len(accesses) == 1
  assert app.state.audit_log == []


# --- S15 C13-01 RED: non-ASCII API key byte ---

def test_non_ascii_api_key_byte_401_with_audit_entry(
    monkeypatch, caplog):
  '''Non-ASCII X-API-Key byte 0xe9 yields 401 plus audit entry.

  Raw byte 0xe9 decodes as latin-1 e-acute; str
  compare_digest raises TypeError on non-ASCII, which
  previously crashed to 500 with no audit entry. The
  request must instead be a clean 401 with an
  unauthorized access and audit entry, no exception.
  '''
  import logging
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  key = 's3cret-test-key-1234567890'
  monkeypatch.setenv('FORECAST_API_KEY', key)
  monkeypatch.delenv('FORECAST_REQUIRE_AUTH', raising=False)
  monkeypatch.delenv('FORECAST_ENV', raising=False)
  app = create_app(dummy_loader)
  client = TestClient(app, raise_server_exceptions=False)
  with caplog.at_level(logging.INFO):
    r = client.post(
        '/predict', json={'features': {'a': 1.0}},
        headers=[(b'x-api-key', b'\xe9')])
  assert r.status_code == 401
  assert r.json() == {'detail': 'unauthorized'}
  assert 'predict_access' in caplog.text
  assert 'outcome=unauthorized' in caplog.text
  audit = app.state.audit_log
  assert len(audit) == 1
  assert audit[-1]['event'] == 'predict'
  assert audit[-1]['actor'] == 'serving'


# --- S15 C13-02 RED: validation log forging ---

def test_validation_error_log_has_no_raw_newline(caplog):
  '''Validation log emits one record with no raw newline.

  A record key containing a newline reaches
  _log_validation_error via the type-check message;
  scrub_text keeps the newline, so logging it raw
  would forge a second log line. Exactly one record
  must be emitted and its message must contain no
  newline or carriage-return characters.
  '''
  import logging
  from forecast_ml_framework.serving.app import (
    _log_validation_error,
  )
  exc = ValueError(
      'feature a\nFORGED log line\rforged must be a '
      'finite number')
  with caplog.at_level(logging.WARNING):
    _log_validation_error(exc)
  assert len(caplog.records) == 1
  message = caplog.records[0].getMessage()
  assert '\n' not in message
  assert '\r' not in message
  assert 'validation_error' in message


# --- S15 C13-03 RED: duplicate auth headers ---

def test_duplicate_api_key_headers_canonical_audit(monkeypatch):
  '''Duplicate X-API-Key headers cannot misattribute audit.

  Middleware _header_map is last-wins while
  dict(headers) is first-wins, so attacker then valid
  previously passed auth as valid but audited the
  attacker HMAC. The request must either be rejected
  401 or, if it succeeds, its audit key_id must be
  the canonical last (valid) key.
  '''
  import hashlib
  import hmac
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  key = 's3cret-test-key-1234567890'
  monkeypatch.setenv('FORECAST_API_KEY', key)
  monkeypatch.delenv('FORECAST_REQUIRE_AUTH', raising=False)
  monkeypatch.delenv('FORECAST_ENV', raising=False)
  app = create_app(dummy_loader)
  client = TestClient(app, raise_server_exceptions=False)
  r = client.post(
      '/predict', json={'features': {'a': 1.0}},
      headers=[('x-api-key', 'attacker'),
               ('x-api-key', key)])
  canonical = hmac.new(
      key.encode(), key.encode(),
      hashlib.sha256).hexdigest()[:16]
  audit = app.state.audit_log
  assert r.status_code in (200, 401)
  assert audit
  assert audit[-1]['details']['key_id_hmac'] == canonical
  if r.status_code == 401:
    assert r.json() == {'detail': 'unauthorized'}
