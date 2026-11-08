#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the /metrics producer (FIX-STUB-1).'''

import json
from pathlib import Path

import pytest

pytest.importorskip('httpx')
pytest.importorskip('fastapi')

_ROOT = Path(__file__).resolve().parents[1]
_OBS_DIR = _ROOT / 'deploy' / 'observability'
_PROM = _OBS_DIR / 'prometheus.yaml'
_DRIFT = _OBS_DIR / 'dashboards' / 'model-drift.json'
_LATENCY = _OBS_DIR / 'dashboards' / 'api-latency.json'


class DummyModel:
  '''Dummy model returning deterministic predictions.'''

  def predict(self, rows):
    '''Predict sum of each row.'''
    return [sum(row) for row in rows]


def dummy_loader():
  '''Return dummy model.'''
  return DummyModel()


def _client():
  '''Return a TestClient for a fresh app.'''
  from fastapi.testclient import TestClient
  from forecast_ml_framework.serving.app import create_app
  return TestClient(create_app(dummy_loader))


def _metric_value(body, name):
  '''Return the value of a label-free metric line.'''
  for line in body.splitlines():
    if line.startswith(name + ' '):
      return float(line.rsplit(' ', 1)[1])
  raise AssertionError(f'metric not found: {name}')


def _dashboard_text(path):
  '''Return dashboard JSON as text.'''
  return path.read_text()


def test_metrics_endpoint_returns_prometheus_text():
  '''GET /metrics is 200 Prometheus text with all series.'''
  client = _client()
  response = client.get('/metrics')
  assert response.status_code == 200
  assert response.headers['content-type'].startswith('text/plain')
  body = response.text
  assert '# TYPE http_request_duration_seconds histogram' in body
  assert 'http_request_duration_seconds_sum' not in body
  assert '# TYPE predict_requests_total counter' in body
  assert '# TYPE predict_errors_total counter' in body
  assert '# TYPE drift_psi gauge' in body
  assert '# TYPE drift_ks gauge' in body
  assert 'drift_psi{feature="prediction"} 0' in body
  assert 'drift_ks{feature="prediction"} 0' in body


def test_predict_increments_counters_and_histogram():
  '''One successful /predict increments request + histogram.'''
  client = _client()
  response = client.post(
    '/predict', json={'features': {'a': 1.0, 'b': 2.0}})
  assert response.status_code == 200
  body = client.get('/metrics').text
  assert _metric_value(body, 'predict_requests_total') == 1
  assert _metric_value(body, 'predict_errors_total') == 0
  count = 'http_request_duration_seconds_count{code="200"' \
    ',route="/predict"} 1'
  assert count in body
  assert 'le="+Inf"} 1' in body


def test_failed_predict_increments_error_counter():
  '''A rejected /predict increments requests and errors.'''
  client = _client()
  response = client.post('/predict', json={'features': {}})
  assert response.status_code == 422
  body = client.get('/metrics').text
  assert _metric_value(body, 'predict_requests_total') == 1
  assert _metric_value(body, 'predict_errors_total') == 1
  assert 'http_request_duration_seconds_count{code="422"' \
    ',route="/predict"} 1' in body


def test_metrics_registry_histogram_buckets_cumulative():
  '''Histogram buckets are cumulative over observations.'''
  from forecast_ml_framework.serving.metrics import MetricsRegistry
  registry = MetricsRegistry()
  registry.observe_http(0.003, '/predict', 200)
  registry.observe_http(0.4, '/predict', 200)
  body = registry.render()
  assert 'le="0.005"} 1' in body
  assert 'le="0.25"} 1' in body
  assert 'le="0.5"} 2' in body
  assert 'le="+Inf"} 2' in body
  assert 'http_request_duration_seconds_count{code="200"' \
    ',route="/predict"} 2' in body
  assert 'http_request_duration_seconds_sum{code="200"' \
    ',route="/predict"}' in body


def test_drift_gauge_defaults_and_set():
  '''Drift gauges default to 0 and update via set_drift.'''
  from forecast_ml_framework.serving.metrics import MetricsRegistry
  registry = MetricsRegistry()
  assert 'drift_psi{feature="prediction"} 0' in registry.render()
  registry.set_drift(0.3, 0.4)
  body = registry.render()
  assert 'drift_psi{feature="prediction"} 0.3' in body
  assert 'drift_ks{feature="prediction"} 0.4' in body


def test_update_from_monitoring_report_prediction():
  '''Pipeline hook copies prediction drift into gauges.'''
  from forecast_ml_framework.serving.metrics import MetricsRegistry
  registry = MetricsRegistry()
  report = {
    'prediction': {'psi': 0.42, 'ks': 0.12},
    'features': {'num': {'psi': 0.01, 'ks': 0.01}},
    'performance': {},
  }
  assert registry.update_from_monitoring_report(report) is True
  body = registry.render()
  assert 'drift_psi{feature="prediction"} 0.42' in body
  assert 'drift_ks{feature="prediction"} 0.12' in body


def test_update_from_monitoring_report_max_feature():
  '''Hook falls back to max feature drift without prediction.'''
  from forecast_ml_framework.serving.metrics import MetricsRegistry
  registry = MetricsRegistry()
  report = {
    'features': {
      'num': {'psi': 0.33, 'ks': 0.22},
      'cat': {'psi': 0.1, 'ks': 0.05},
    },
  }
  assert registry.update_from_monitoring_report(report) is True
  body = registry.render()
  assert 'drift_psi{feature="prediction"} 0.33' in body
  assert 'drift_ks{feature="prediction"} 0.22' in body


def test_update_from_monitoring_report_empty_keeps_previous():
  '''Empty report returns False and keeps prior gauges.'''
  from forecast_ml_framework.serving.metrics import MetricsRegistry
  registry = MetricsRegistry()
  registry.set_drift(0.3, 0.4)
  assert registry.update_from_monitoring_report({}) is False
  assert registry.update_from_monitoring_report(None) is False
  body = registry.render()
  assert 'drift_psi{feature="prediction"} 0.3' in body
  assert 'drift_ks{feature="prediction"} 0.4' in body


def test_set_drift_rejects_negative_values():
  '''Negative drift values raise ValueError.'''
  from forecast_ml_framework.serving.metrics import MetricsRegistry
  registry = MetricsRegistry()
  with pytest.raises(ValueError):
    registry.set_drift(-0.1, 0.2)
  with pytest.raises(ValueError):
    registry.set_drift(0.1, -0.2)


def test_observe_http_rejects_negative_duration():
  '''Negative durations raise ValueError.'''
  from forecast_ml_framework.serving.metrics import MetricsRegistry
  registry = MetricsRegistry()
  with pytest.raises(ValueError):
    registry.observe_http(-0.1, '/predict', 200)


def test_metrics_public_when_api_key_set(monkeypatch):
  '''/metrics scrapes without a key; /predict still 401s.'''
  monkeypatch.setenv('FORECAST_API_KEY', 'k' * 32)
  monkeypatch.delenv('FORECAST_REQUIRE_AUTH', raising=False)
  monkeypatch.delenv('FORECAST_ENV', raising=False)
  client = _client()
  assert client.get('/metrics').status_code == 200
  response = client.post(
    '/predict', json={'features': {'a': 1.0, 'b': 2.0}})
  assert response.status_code == 401
  body = client.get('/metrics').text
  assert _metric_value(body, 'predict_requests_total') == 1
  assert _metric_value(body, 'predict_errors_total') == 1


def test_dashboards_reference_real_metrics_only():
  '''Dashboards cite emitted names, with no stub wording.'''
  for path in (_DRIFT, _LATENCY):
    text = _dashboard_text(path)
    assert 'STUB' not in text
    assert 'stub' not in text.lower()
    assert 'placeholder' not in text.lower()
    dashboard = json.loads(text)
    assert dashboard['panels']
  drift = json.loads(_dashboard_text(_DRIFT))
  drift_exprs = ' '.join(
    target['expr']
    for panel in drift['panels']
    for target in panel['targets'])
  assert 'drift_psi' in drift_exprs
  assert 'drift_ks' in drift_exprs
  latency = json.loads(_dashboard_text(_LATENCY))
  latency_exprs = ' '.join(
    target['expr']
    for panel in latency['panels']
    for target in panel['targets'])
  assert 'http_request_duration_seconds_bucket' in latency_exprs
  assert 'http_request_duration_seconds_count' in latency_exprs
  assert 'predict_requests_total' in latency_exprs
  assert 'predict_errors_total' in latency_exprs


def test_prometheus_config_targets_metrics_producer():
  '''Prometheus scrapes the app /metrics, no stub wording.'''
  import yaml
  raw = _PROM.read_text()
  assert 'STUB' not in raw
  assert 'placeholder' not in raw.lower()
  config = yaml.safe_load(raw)
  jobs = config['scrape_configs']
  forecast = next(
    job for job in jobs if job['job_name'] == 'forecast-ml')
  assert forecast['metrics_path'] == '/metrics'
  assert forecast['static_configs'][0]['targets'] == [
    'forecast-ml:8000']
  assert 'http_request_duration_seconds' in raw
  assert 'predict_requests_total' in raw
  assert 'drift_psi' in raw
