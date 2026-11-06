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
  assert client.get('/health').json() == {
      'status': 'ok', 'model_loaded': True}
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


def test_predict_nonfinite_output_500():
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app

  class InfModel:
    def predict(self, rows):
      return [float('inf') for _ in rows]

  client = TestClient(create_app(lambda: InfModel()))
  r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 500
  assert r.json() == {'detail': 'prediction failed'}


def test_predict_nan_output_500():
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app

  class NanModel:
    def predict(self, rows):
      return [float('nan') for _ in rows]

  client = TestClient(create_app(lambda: NanModel()))
  r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 500
  assert r.json() == {'detail': 'prediction failed'}


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

def test_predict_second_record_unknown_feature_422():
  '''Unknown feature in 2nd record rejected (not only first).'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))
  payload = {'records': [{'a': 1.0, 'b': 2.0},
                         {'a': 3.0, 'b': 4.0, 'evil': 5.0}]}
  r = client.post('/predict', json=payload)
  assert r.status_code == 422
  assert 'evil' in r.text


def test_predict_second_record_unknown_vs_model_features_422():
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app

  class StrictModel:
    feature_names_in_ = ['a', 'b']

    def predict(self, rows):
      return [sum(row) for row in rows]

  client = TestClient(create_app(lambda: StrictModel()))
  payload = {'records': [{'a': 1.0, 'b': 2.0},
                         {'a': 3.0, 'b': 4.0, 'c': 9.0}]}
  r = client.post('/predict', json=payload)
  assert r.status_code == 422
  assert 'c' in r.text


def test_predict_chunked_body_over_cap_413_no_content_length():
  '''Chunked stream >10MB rejected by streaming counter.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))

  def chunks():
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
  monkeypatch.setenv('FORECAST_API_KEY', 's3cret-test-key')
  client = TestClient(create_app(dummy_loader))
  payload = {'features': {'a': 1.0}}
  assert client.post('/predict', json=payload).status_code == 401
  assert client.post('/predict', json=payload,
                     headers={'X-API-Key': 'wrong'}).status_code == 401
  r = client.post('/predict', json=payload,
                  headers={'X-API-Key': 's3cret-test-key'})
  assert r.status_code == 200
  r2 = client.post('/predict', json=payload,
                   headers={'Authorization': 'Bearer s3cret-test-key'})
  assert r2.status_code == 200
  # Health stays public for probes.
  assert client.get('/health').status_code == 200


def test_api_key_not_set_means_no_auth(monkeypatch):
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  monkeypatch.delenv('FORECAST_API_KEY', raising=False)
  client = TestClient(create_app(dummy_loader))
  r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 200


def test_security_headers_present():
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
    assert "default-src 'none'" in resp.headers.get(
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
    def loader():
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
    def loader():
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
    def predict(self, rows):
      raise ValueError('/secret/internal detail')

  client = TestClient(create_app(lambda: ValueErrorModel()))
  r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 500
  assert r.json() == {'detail': 'prediction failed'}
  assert '/secret' not in r.text
  assert 'internal' not in r.text


def test_predict_validation_error_still_422():
  '''_validate_records ValueError stays 422 with field detail.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  client = TestClient(create_app(dummy_loader))
  r = client.post('/predict', json={'features': {'a': 'x'}})
  assert r.status_code == 422


def test_health_caches_probe_on_app_instance():
  '''Second /health does not re-probe loader; cache on app.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  calls = {'n': 0}

  def counting_loader():
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
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app

  class DictInfModel:
    def predict(self, rows):
      return {'pred': [float('inf')]}

  client = TestClient(create_app(lambda: DictInfModel()))
  r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 500
  assert r.json() == {'detail': 'prediction failed'}


# --- loop7 serving finite (RED) ---

def test_check_predictions_finite_numpy_decimal_complex():
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
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app

  class NoneModel:
    def predict(self, rows):
      return [None for _ in rows]

  client = TestClient(create_app(lambda: NoneModel()))
  r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 500
  assert r.json() == {'detail': 'prediction failed'}


def test_predict_string_label_output_200():
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app

  class LabelModel:
    def predict(self, rows):
      return ['cat' for _ in rows]

  client = TestClient(create_app(lambda: LabelModel()))
  r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 200
  assert r.json() == {'predictions': ['cat']}


# --- loop9 masked/generator (RED) ---

def test_check_predictions_finite_masked_constant_raises():
  import numpy as np
  from forecast_ml_framework.serving.app import _check_predictions_finite
  with pytest.raises(ValueError, match='finite'):
    _check_predictions_finite(np.ma.masked)
  with pytest.raises(ValueError, match='finite'):
    _check_predictions_finite([np.ma.masked])
  with pytest.raises(ValueError, match='finite'):
    _check_predictions_finite({'a': np.ma.masked})


def test_check_predictions_finite_masked_array_raises_when_masked():
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
  from forecast_ml_framework.serving.app import _predict

  class GenModel:
    def predict(self, rows):
      return (float(v) for v in [1.0, 2.0][:len(rows)])

  assert _predict(GenModel(), [{'a': 1.0}, {'a': 2.0}]) == [1.0, 2.0]


def test_predict_generator_http_200_not_empty():
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app

  class GenModel:
    def predict(self, rows):
      return (sum(row) for row in rows)

  client = TestClient(create_app(lambda: GenModel()))
  r = client.post('/predict', json={'features': {'a': 3.0}})
  assert r.status_code == 200, r.text
  assert r.json() == {'predictions': [3.0]}


def test_predict_masked_output_500():
  import numpy as np
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app

  class MaskedModel:
    def predict(self, rows):
      return np.ma.masked

  client = TestClient(create_app(lambda: MaskedModel()))
  r = client.post('/predict', json={'features': {'a': 1.0}})
  assert r.status_code == 500
  assert r.json() == {'detail': 'prediction failed'}
