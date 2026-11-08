#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Race and drift tests for the /metrics producer.

Pins two fixes: ``render`` renders from a deep-copied
snapshot taken under the lock, so concurrent requests
cannot tear a render; negative drift in a monitoring
report is treated as missing, so the pipeline hook
returns False and keeps the gauges instead of raising
via ``set_drift``.
'''

import sys
import threading

import pytest

from forecast_ml_framework.serving.metrics import (
  MetricsRegistry,
  _finite_stat,
)


def _snapshot(body):
  '''Return (buckets, inf, count, summed) for one series.'''
  buckets = []
  inf = None
  count = None
  summed = None
  prefix = 'http_request_duration_seconds_bucket'
  for line in body.splitlines():
    if line.startswith(prefix):
      value = int(line.rsplit(' ', 1)[1])
      if 'le="+Inf"' in line:
        inf = value
      else:
        buckets.append(value)
    elif line.startswith(
        'http_request_duration_seconds_count'):
      count = int(line.rsplit(' ', 1)[1])
    elif line.startswith(
        'http_request_duration_seconds_sum'):
      summed = float(line.rsplit(' ', 1)[1])
  assert count is not None
  assert inf is not None
  assert summed is not None
  return buckets, inf, count, summed


def _assert_coherent(body):
  '''Assert histogram invariants of one rendered body.'''
  buckets, inf, count, summed = _snapshot(body)
  assert buckets
  assert inf == count
  assert all(value <= inf for value in buckets)
  assert buckets == sorted(buckets)
  # Duration 0.01 hits every bucket from le=0.01 up.
  assert buckets[0] == 0
  assert all(value == count for value in buckets[1:])
  assert summed == pytest.approx(count * 0.01)


def test_render_concurrent_observe_no_torn_snapshot():
  '''Concurrent observe and render never tear a snapshot.

  Fails on the shallow copy: buckets read outside the
  lock can exceed the count read before them.
  '''
  registry = MetricsRegistry()
  stop = threading.Event()
  first_observed = threading.Event()

  def observe_loop():
    '''Observe fixed durations until stopped.'''
    while not stop.is_set():
      registry.observe_http(0.01, '/predict', 200)
      registry.record_predict(200)
      first_observed.set()

  worker = threading.Thread(target=observe_loop)
  previous = sys.getswitchinterval()
  sys.setswitchinterval(1e-6)
  try:
    worker.start()
    # Wait approach: block until the worker performs its first
    # observe_http before asserting on a histogram, since an
    # empty histogram is valid pre-traffic state and render()
    # before the first observation has no count line to parse.
    assert first_observed.wait(timeout=5), (
        'worker did not observe within 5 seconds')
    for _ in range(200):
      _assert_coherent(registry.render())
  finally:
    stop.set()
    worker.join()
    sys.setswitchinterval(previous)
  _assert_coherent(registry.render())


def test_update_negative_psi_returns_false_keeps_gauges():
  '''Negative prediction psi returns False, keeps gauges.

  Fails before the fix: set_drift raises ValueError.
  '''
  registry = MetricsRegistry()
  registry.set_drift(0.3, 0.4)
  report = {'prediction': {'psi': -0.1, 'ks': 0.2}}
  assert registry.update_from_monitoring_report(report) is False
  body = registry.render()
  assert 'drift_psi{feature="prediction"} 0.3' in body
  assert 'drift_ks{feature="prediction"} 0.4' in body


def test_update_negative_ks_returns_false_keeps_gauges():
  '''Negative prediction ks returns False, keeps gauges.'''
  registry = MetricsRegistry()
  registry.set_drift(0.3, 0.4)
  report = {'prediction': {'psi': 0.2, 'ks': -0.1}}
  assert registry.update_from_monitoring_report(report) is False
  body = registry.render()
  assert 'drift_psi{feature="prediction"} 0.3' in body
  assert 'drift_ks{feature="prediction"} 0.4' in body


def test_update_negative_feature_returns_false():
  '''Negative-only feature drift returns False, no raise.'''
  registry = MetricsRegistry()
  registry.set_drift(0.3, 0.4)
  report = {'features': {'num': {'psi': -0.5, 'ks': 0.1}}}
  assert registry.update_from_monitoring_report(report) is False
  body = registry.render()
  assert 'drift_psi{feature="prediction"} 0.3' in body
  assert 'drift_ks{feature="prediction"} 0.4' in body


def test_finite_stat_rejects_negative_allows_zero():
  '''_finite_stat rejects negatives, allows zero.'''
  assert _finite_stat({'psi': -0.1}, 'psi') is None
  assert _finite_stat({'psi': 0.0}, 'psi') == 0.0
  assert _finite_stat({'psi': 0.1}, 'psi') == 0.1


def test_update_zero_drift_returns_true():
  '''Zero drift is valid and updates the gauges.'''
  registry = MetricsRegistry()
  registry.set_drift(0.3, 0.4)
  report = {'prediction': {'psi': 0.0, 'ks': 0.0}}
  assert registry.update_from_monitoring_report(report) is True
  body = registry.render()
  assert 'drift_psi{feature="prediction"} 0' in body
  assert 'drift_ks{feature="prediction"} 0' in body
