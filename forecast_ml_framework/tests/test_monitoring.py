#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the monitoring / drift pipeline (Phase 3).'''

import logging

import numpy as np
import pandas as pd
import pytest

from forecast_ml_framework.pipelines.monitoring.nodes import (
  build_monitoring_report,
  compute_drift,
  evaluate_alerts,
  performance_decay,
  prediction_drift,
)
from forecast_ml_framework.pipelines.monitoring.pipeline import (
  create_pipeline,
)


def _params(**overrides):
  '''Return monitoring params for tests.'''
  base = {
    'bins': 10,
    'epsilon': 1e-4,
    'thresholds': {
      'psi': {'warning': 0.1, 'critical': 0.25},
      'ks': {'warning': 0.1, 'critical': 0.25},
      'missing_rate_delta': {'warning': 0.05, 'critical': 0.1},
    },
  }
  base.update(overrides)
  return base


def _frames(shift=0.0, n=1000, seed=42):
  '''Return reference/current frames (numeric + categorical).'''
  rng = np.random.default_rng(seed)
  ref = pd.DataFrame({
    'num': rng.normal(0.0, 1.0, n),
    'cat': rng.choice(['a', 'b', 'c'], n, p=[0.5, 0.3, 0.2]),
  })
  cur = pd.DataFrame({
    'num': rng.normal(shift, 1.0, n),
    'cat': rng.choice(['a', 'b', 'c'], n,
                       p=[0.1, 0.3, 0.6] if shift else [0.5, 0.3, 0.2]),
  })
  return ref, cur


def test_compute_drift_no_drift_clean():
  '''Identical distributions give near-zero PSI/KS.'''
  ref, _ = _frames(shift=0.0)
  cur = ref.copy()
  report = compute_drift(ref, cur, _params())
  assert set(report) == {'num', 'cat'}
  assert report['num']['psi'] == pytest.approx(0.0, abs=0.05)
  assert report['num']['ks'] == pytest.approx(0.0, abs=0.05)
  assert report['num']['mean_shift'] == pytest.approx(0.0, abs=0.1)
  assert report['cat']['psi'] == pytest.approx(0.0, abs=0.05)


def test_compute_drift_detects_shifted_numeric():
  '''A mean shift is detected by PSI and KS.'''
  ref, cur = _frames(shift=2.0)
  report = compute_drift(ref, cur, _params())
  assert report['num']['psi'] > 0.25
  assert report['num']['ks'] > 0.25
  assert report['num']['mean_shift'] == pytest.approx(2.0, abs=0.2)
  # KS alias and standardised shift are present.
  assert report['num']['ks_statistic'] == pytest.approx(
    report['num']['ks'])
  assert report['num']['mean_shift_std'] > 1.0


def test_compute_drift_detects_shifted_categorical():
  '''A category proportion shift gives high categorical PSI.'''
  ref, cur = _frames(shift=2.0)
  report = compute_drift(ref, cur, _params())
  assert report['cat']['psi'] > 0.25
  assert report['cat']['ks'] > 0.1
  assert np.isnan(report['cat']['mean_shift'])


def test_compute_drift_missing_rate_delta():
  '''Missing-rate delta reflects added NaNs in current.'''
  ref, cur = _frames(shift=0.0)
  cur.loc[:99, 'num'] = np.nan
  report = compute_drift(ref, cur, _params())
  assert report['num']['missing_rate_delta'] == pytest.approx(
    0.1, abs=0.01)
  assert report['num']['missing_rate_current'] == pytest.approx(
    0.1, abs=0.01)


def test_compute_drift_empty_warns_not_crashes(caplog):
  '''Empty frames return NaN stats with a warning.'''
  empty = pd.DataFrame({'num': pd.Series([], dtype=float),
                        'cat': pd.Series([], dtype=object)})
  _, cur = _frames()
  with caplog.at_level(logging.WARNING):
    report = compute_drift(empty, cur, _params())
  assert np.isnan(report['num']['psi'])
  assert np.isnan(report['num']['ks'])
  assert any('empty' in r.message.lower() for r in caplog.records)


def test_compute_drift_small_warns_not_crashes(caplog):
  '''One-row frames return NaN stats with a warning.'''
  ref = pd.DataFrame({'num': [1.0], 'cat': ['a']})
  cur = pd.DataFrame({'num': [9.0], 'cat': ['b']})
  with caplog.at_level(logging.WARNING):
    report = compute_drift(ref, cur, _params())
  assert np.isnan(report['num']['psi'])
  assert any('small' in r.message.lower() or 'empty' in r.message.lower()
             for r in caplog.records)


def test_prediction_drift_shifted():
  '''Shifted continuous predictions give high PSI/KS.'''
  rng = np.random.default_rng(7)
  ref = rng.normal(0.0, 1.0, 1000)
  cur = rng.normal(2.0, 1.0, 1000)
  report = prediction_drift(ref, cur, _params())
  assert report['psi'] > 0.25
  assert report['ks'] > 0.25
  assert report['prediction']['psi'] == pytest.approx(report['psi'])


def test_prediction_drift_from_frames():
  '''Prediction drift accepts frames with a prediction column.'''
  ref = pd.DataFrame({'prediction': [0, 1] * 500})
  cur = pd.DataFrame({'prediction': [0] * 900 + [1] * 100})
  report = prediction_drift(ref, cur, _params())
  assert report['psi'] > 0.25


def test_prediction_drift_empty_warns(caplog):
  '''Empty predictions return NaN PSI with a warning.'''
  with caplog.at_level(logging.WARNING):
    report = prediction_drift([], [], _params())
  assert np.isnan(report['psi'])
  assert any('empty' in r.message.lower() for r in caplog.records)


def test_performance_decay_delta():
  '''Decay reports current, baseline, and delta per metric.'''
  report = performance_decay(
    {'accuracy': 0.80, 'rmse': 1.5},
    _params(baseline_metrics={'accuracy': 0.90, 'rmse': 1.2}))
  assert report['accuracy']['current'] == pytest.approx(0.80)
  assert report['accuracy']['baseline'] == pytest.approx(0.90)
  assert report['accuracy']['delta'] == pytest.approx(-0.10)
  assert report['rmse']['delta'] == pytest.approx(0.30)


def test_performance_decay_empty_warns(caplog):
  '''Empty metrics return an empty report with a warning.'''
  with caplog.at_level(logging.WARNING):
    report = performance_decay({}, _params(baseline_metrics={}))
  assert report == {}


def test_evaluate_alerts_fires_on_drift():
  '''Shifted features fire warning/critical alerts with fields.'''
  ref, cur = _frames(shift=2.0)
  report = compute_drift(ref, cur, _params())
  alerts = evaluate_alerts(report, _params())
  assert alerts
  for alert in alerts:
    assert set(alert) == {
      'feature', 'metric', 'value', 'threshold', 'severity'}
  by_feature = {(a['feature'], a['metric']) for a in alerts}
  assert ('num', 'psi') in by_feature
  psi_alert = next(a for a in alerts
                   if a['feature'] == 'num' and a['metric'] == 'psi')
  assert psi_alert['severity'] == 'critical'
  assert psi_alert['value'] > 0.25
  assert psi_alert['threshold'] == pytest.approx(0.25)


def test_evaluate_alerts_clean_no_alerts():
  '''Clean drift produces no alerts.'''
  ref, cur = _frames(shift=0.0)
  report = compute_drift(ref, cur, _params())
  assert evaluate_alerts(report, _params()) == []


def test_evaluate_alerts_warning_severity():
  '''Values between warning and critical are warnings.'''
  report = {'num': {'psi': 0.15, 'ks': 0.02, 'missing_rate_delta': 0.0}}
  alerts = evaluate_alerts(report, _params())
  assert len(alerts) == 1
  assert alerts[0]['severity'] == 'warning'
  assert alerts[0]['threshold'] == pytest.approx(0.1)


def test_evaluate_alerts_nan_values_skipped():
  '''NaN stats never alert and never crash.'''
  report = {'num': {'psi': float('nan'), 'ks': float('nan')}}
  assert evaluate_alerts(report, _params()) == []


def test_build_monitoring_report_combines_sections():
  '''Combined report carries drift, prediction, performance.'''
  ref, cur = _frames(shift=0.0)
  report = build_monitoring_report(
    compute_drift(ref, cur, _params()),
    np.zeros(100), np.zeros(100),
    {'accuracy': 0.85},
    _params(baseline_metrics={'accuracy': 0.90}))
  assert 'features' in report
  assert report['prediction']['psi'] == pytest.approx(0.0, abs=0.05)
  assert report['performance']['accuracy']['delta'] == pytest.approx(
    -0.05)


def test_evaluate_alerts_on_combined_report():
  '''Alerts also fire on prediction/performance sections.'''
  combined = {
    'features': {'num': {'psi': 0.0, 'ks': 0.0}},
    'prediction': {'psi': 0.5, 'ks': 0.4},
    'performance': {'accuracy': {'delta': -0.2, 'decay': 0.2}},
  }
  alerts = evaluate_alerts(combined, _params())
  features = {a['feature'] for a in alerts}
  assert 'prediction' in features
  assert 'accuracy' in features


def test_pipeline_has_three_nodes():
  '''Pipeline wires drift -> combined report -> alerts.'''
  pipe = create_pipeline()
  names = [n.name for n in pipe.nodes]
  assert names == ['compute_drift', 'build_monitoring_report',
                   'evaluate_alerts']
  assert len(pipe.nodes) == 3
  nodes = {n.name: n for n in pipe.nodes}
  assert 'params:monitoring_params' in nodes['compute_drift'].inputs
  assert 'params:monitoring_params' in (
    nodes['evaluate_alerts'].inputs)
  assert nodes['evaluate_alerts'].outputs == ['alerts']


def test_pipeline_end_to_end_via_node_funcs():
  '''Node funcs chain drift -> report -> alerts without catalog.'''
  nodes = {n.name: n for n in create_pipeline().nodes}
  ref, cur = _frames(shift=2.0)
  drift = nodes['compute_drift'].func(ref, cur, _params())
  report = nodes['build_monitoring_report'].func(
    drift, np.zeros(100), np.ones(100), {'accuracy': 0.8},
    _params(baseline_metrics={'accuracy': 0.9}))
  alerts = nodes['evaluate_alerts'].func(report, _params())
  assert alerts
  assert any(a['feature'] == 'num' for a in alerts)


def test_compute_drift_invalid_bins_raises_value_error():
  '''Non-integer bins raise an informative ValueError.'''
  ref, cur = _frames()
  with pytest.raises(ValueError, match='bins must be a positive int'):
    compute_drift(ref, cur, _params(bins='abc'))


def test_compute_drift_ragged_dict_raises_value_error():
  '''Ragged dict inputs raise an informative ValueError.'''
  _, cur = _frames()
  ragged = {'num': [1.0] * 20, 'cat': ['a']}
  with pytest.raises(ValueError, match='same length'):
    compute_drift(ragged, cur, _params())


def test_compute_drift_epsilon_clamped_finite_psi():
  '''Zero epsilon is clamped so disjoint categories stay finite.'''
  ref = pd.DataFrame({'cat': ['a'] * 100})
  cur = pd.DataFrame({'cat': ['b'] * 100})
  report = compute_drift(ref, cur, _params(epsilon=0))
  assert np.isfinite(report['cat']['psi'])
  assert report['cat']['psi'] > 0


def test_compute_drift_inf_reference_psi_is_nan_with_warning(caplog):
  '''All-inf reference yields NaN PSI plus a warning, not inf.'''
  ref = pd.DataFrame({'num': [np.inf] * 100})
  cur = pd.DataFrame({'num': [1.0] * 100})
  with caplog.at_level(logging.WARNING):
    report = compute_drift(ref, cur, _params())
  assert np.isnan(report['num']['psi'])
  assert any('psi' in r.message.lower() for r in caplog.records)


def test_performance_decay_non_numeric_is_nan_skip(caplog):
  '''Non-numeric metric values become NaN deltas, not a crash.'''
  with caplog.at_level(logging.WARNING):
    report = performance_decay(
      {'accuracy': 'bad'},
      _params(baseline_metrics={'accuracy': 0.9}))
  assert np.isnan(report['accuracy']['delta'])
  assert np.isnan(report['accuracy']['decay'])


def test_performance_decay_inf_is_nan_not_passthrough(caplog):
  '''Inf metric values become NaN deltas, not inf passthrough.'''
  with caplog.at_level(logging.WARNING):
    report = performance_decay(
      {'accuracy': float('inf')},
      _params(baseline_metrics={'accuracy': 0.9}))
  assert np.isnan(report['accuracy']['delta'])
  assert np.isnan(report['accuracy']['decay'])
