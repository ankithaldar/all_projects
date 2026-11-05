#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for serving app.'''

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
