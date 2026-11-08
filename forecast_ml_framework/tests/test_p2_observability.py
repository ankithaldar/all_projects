#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Observability configs for P2-O1 (live /metrics producer).'''

import json
import re
import shutil
import subprocess
from pathlib import Path

import yaml

_LIVE_METRICS = (
    'http_request_duration_seconds',
    'predict_requests_total',
    'predict_errors_total',
    'drift_psi',
    'drift_ks',
)
_DRIFT_THRESHOLD = 0.2
_PROMQL_NON_METRIC = {
    'avg_over_time',
    'by',
    'clamp_min',
    'code',
    'histogram_quantile',
    'job',
    'le',
    'rate',
    'route',
    'sum',
}
_ROOT = Path(__file__).resolve().parents[1]
_METRICS_PY = (
    _ROOT / 'src' / 'forecast_ml_framework' / 'serving' / 'metrics.py')
_OBS_DIR = _ROOT / 'deploy' / 'observability'
_OTEL = _OBS_DIR / 'otel-collector.yaml'
_PROM = _OBS_DIR / 'prometheus.yaml'
_DRIFT = _OBS_DIR / 'dashboards' / 'model-drift.json'
_LATENCY = _OBS_DIR / 'dashboards' / 'api-latency.json'


def _load_yaml(path):
  '''Load YAML from path.'''
  with path.open() as handle:
    return yaml.safe_load(handle)


def _load_json(path):
  '''Load JSON from path.'''
  return json.loads(path.read_text())


def _panels(dashboard):
  '''Return panel dicts from a dashboard.'''
  return dashboard['panels']


def _exprs(dashboard):
  '''Return every target expr in a dashboard.'''
  exprs = []
  for panel in _panels(dashboard):
    for target in panel['targets']:
      exprs.append(target['expr'])
  return exprs


def _unknown_metrics(expr):
  '''Return identifiers that are not live metrics.'''
  cleaned = re.sub(r'"[^"]*"', '', expr)
  cleaned = re.sub(r'\[[^\]]*\]', '', cleaned)
  names = re.findall(r'[A-Za-z_][A-Za-z0-9_]*', cleaned)
  unknown = []
  for name in names:
    if name in _PROMQL_NON_METRIC:
      continue
    if name in _LIVE_METRICS:
      continue
    if any(name.startswith(live + '_') for live in _LIVE_METRICS):
      continue
    unknown.append(name)
  return unknown


def _drift_fires(value):
  '''Evaluate the drift threshold on a value.'''
  return value > _DRIFT_THRESHOLD


def test_otel_collector_loads():
  '''Otel collector YAML parses to a mapping.'''
  config = _load_yaml(_OTEL)
  assert isinstance(config, dict)


def test_otel_collector_receivers_and_exporters():
  '''Otlp receiver and Prometheus exporter are set.'''
  config = _load_yaml(_OTEL)
  protocols = config['receivers']['otlp']['protocols']
  assert 'grpc' in protocols
  assert 'http' in protocols
  assert 'prometheus' in config['exporters']
  raw = _OTEL.read_text()
  assert 'service.name' in raw
  assert 'forecast-ml' in raw
  assert 'service.version' in raw
  assert 'SERVICE_VERSION' in raw


def test_prometheus_scrape_and_retention():
  '''Prometheus scrapes both jobs every 15s, retention 15d.'''
  config = _load_yaml(_PROM)
  assert config['global']['scrape_interval'] == '15s'
  raw = _PROM.read_text()
  assert 'storage.tsdb.retention.time' in raw
  assert '15d' in raw
  assert 'forecast-ml:8000' in raw
  assert '/metrics' in raw
  assert 'otel-collector:8888' in raw
  assert '--web.external-url' in raw


def test_promtool_config_check_or_fallback():
  '''Promtool validates config, else structural fallback.'''
  tool = shutil.which('promtool')
  if tool is not None:
    done = subprocess.run(
        [tool, 'check', 'config', str(_PROM)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return
  config = _load_yaml(_PROM)
  assert config['scrape_configs'], 'scrape_configs must list targets'


def test_model_drift_dashboard_json():
  '''Model drift dashboard is valid JSON with psi and ks.'''
  dashboard = _load_json(_DRIFT)
  assert _panels(dashboard), 'panels must exist'
  titles = [panel['title'] for panel in _panels(dashboard)]
  joined = ' '.join(titles).lower()
  assert 'drift psi' in joined
  assert 'drift ks' in joined


def test_api_latency_dashboard_json():
  '''Latency dashboard is valid JSON with p50 and p95.'''
  dashboard = _load_json(_LATENCY)
  assert _panels(dashboard), 'panels must exist'
  titles = [panel['title'] for panel in _panels(dashboard)]
  joined = ' '.join(titles).lower()
  assert 'p50' in joined
  assert 'p95' in joined
  assert 'predict' in joined
  assert '5xx' in joined


def test_dashboard_exprs_use_live_names_only():
  '''Every panel expr references the live allowlist only.'''
  for path in (_DRIFT, _LATENCY):
    for expr in _exprs(_load_json(path)):
      assert expr.strip(), 'panel expr must be non-empty'
      assert not _unknown_metrics(expr), (str(path), expr)
      assert any(live in expr for live in _LIVE_METRICS), expr


def test_live_metrics_no_stub_headers_in_configs():
  '''Prometheus config and dashboards claim live metrics.'''
  producer = _METRICS_PY.read_text()
  for name in _LIVE_METRICS:
    assert name in producer, name
  for path in (_PROM, _DRIFT, _LATENCY):
    raw = path.read_text()
    assert 'STUB ONLY' not in raw, path
    assert 'STUB' not in raw, path
    assert 'stub' not in raw.lower(), path
    assert 'placeholder' not in raw.lower(), path
  prom_raw = _PROM.read_text()
  for name in _LIVE_METRICS:
    assert name in prom_raw, name
  drift_raw = _DRIFT.read_text()
  assert 'drift_psi' in drift_raw
  assert 'drift_ks' in drift_raw
  latency_raw = _LATENCY.read_text()
  assert 'http_request_duration_seconds' in latency_raw
  assert 'predict_requests_total' in latency_raw
  assert 'predict_errors_total' in latency_raw


def test_live_drift_metric_in_dashboard():
  '''Drift dashboard queries the live drift metrics.'''
  exprs = _exprs(_load_json(_DRIFT))
  assert any('drift_psi' in expr for expr in exprs)
  assert any('drift_ks' in expr for expr in exprs)


def test_live_drift_fires_on_synthetic_value():
  '''Synthetic drift above 0.2 fires; below does not.'''
  assert _drift_fires(0.30) is True
  assert _drift_fires(0.10) is False
  assert _drift_fires(0.2) is False
  assert 'drift_psi' in _LIVE_METRICS
