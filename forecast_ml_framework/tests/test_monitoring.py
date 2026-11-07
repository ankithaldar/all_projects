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


# --- loop4: uncovered branches in monitoring nodes ---


def test_bins_variants_and_errors():
  '''_bins handles aliases, float ints, and bad values.'''
  from forecast_ml_framework.pipelines.monitoring.nodes import _bins
  assert _bins({'n_bins': 5}) == 5
  assert _bins({'num_bins': 6}) == 6
  assert _bins({'bins': 5.0}) == 5
  assert _bins({}) == 10
  for bad in (True, 5.5, float('inf'), 'abc', 0, -1):
    with pytest.raises(ValueError, match='bins must be'):
      _bins({'bins': bad})


def test_epsilon_variants_and_errors():
  '''_epsilon clamps/bad values as documented.'''
  from forecast_ml_framework.pipelines.monitoring.nodes import _epsilon
  assert _epsilon({'epsilon': 0.01}) == pytest.approx(0.01)
  assert _epsilon({'epsilon': -1}) == pytest.approx(1e-9)
  assert _epsilon({'epsilon': float('inf')}) == pytest.approx(1e-9)
  for bad in (True, 'abc'):
    with pytest.raises(ValueError, match='epsilon must be'):
      _epsilon({'epsilon': bad})


def test_min_rows_bad_value():
  '''_min_rows rejects non-int values.'''
  from forecast_ml_framework.pipelines.monitoring.nodes import _min_rows
  assert _min_rows({'min_rows': '20'}) == 20
  with pytest.raises(ValueError, match='min_rows must be'):
    _min_rows({'min_rows': 'abc'})


def test_psi_helpers_edge_cases():
  '''PSI helpers handle bad epsilon, inf, constant ref.'''
  from forecast_ml_framework.pipelines.monitoring.nodes import (
    _numeric_psi, _psi_from_props)
  assert _psi_from_props([0.5, 0.5], [0.5, 0.5], 'bad') == pytest.approx(
    0.0, abs=1e-9)
  assert _psi_from_props([0.5], [0.5], -1) == pytest.approx(0.0, abs=1e-9)
  assert np.isnan(_psi_from_props(
    [float('inf')], [float('inf')], 1e-4))
  const = np.ones(100)
  assert np.isfinite(_numeric_psi(const, const * 2, 10, 1e-4))


def test_compute_drift_empty_no_columns():
  '''Empty frames with no columns return empty report.'''
  assert compute_drift(pd.DataFrame(), pd.DataFrame(),
                       _params()) == {}


def test_prediction_drift_series_forms_and_kwargs():
  '''prediction_drift accepts proba/first-col/dict/kwargs forms.'''
  ref_proba = pd.DataFrame({'proba': [0.1, 0.9] * 50})
  cur_proba = pd.DataFrame({'proba': [0.9, 0.1] * 50})
  assert 'psi' in prediction_drift(ref_proba, cur_proba, _params())
  ref_other = pd.DataFrame({'other': [0, 1] * 50})
  assert 'psi' in prediction_drift(ref_other, ref_other, _params())
  assert 'psi' in prediction_drift(
    pd.DataFrame(), pd.DataFrame(), _params())
  assert 'psi' in prediction_drift(
    {'prediction': [0, 1] * 50}, {'prediction': [1, 0] * 50},
    _params())
  assert 'psi' in prediction_drift(
    reference_predictions=[0, 1] * 50,
    current_predictions=[1, 0] * 50, params=_params())
  assert 'psi' in prediction_drift(
    ref=[0, 1] * 50, cur=[1, 0] * 50, params=_params())
  assert 'psi' in prediction_drift(
    pd.Series([0, 1] * 50), None, _params())


def test_performance_decay_signature_variants():
  '''performance_decay handles wrapped/callable/kwargs forms.'''
  rep = performance_decay(
    {'current_metrics': {'accuracy': 0.8}},
    {'baseline_metrics': {'accuracy': 0.9}})
  assert rep['accuracy']['delta'] == pytest.approx(-0.1)
  rep2 = performance_decay(lambda: None, {'accuracy': 0.8},
                           _params(baseline_metrics={
                             'accuracy': 0.9}))
  assert np.isnan(rep2['accuracy']['delta'])
  rep2b = performance_decay({'accuracy': 0.8}, None,
                            _params(baseline_metrics={
                              'accuracy': 0.9}))
  assert rep2b['accuracy']['delta'] == pytest.approx(-0.1)
  rep3 = performance_decay(current={'accuracy': 0.8},
                           baseline={'accuracy': 0.9})
  assert rep3['accuracy']['delta'] == pytest.approx(-0.1)
  rep4 = performance_decay({'accuracy': 0.8},
                           _params(baseline_metrics={
                             'accuracy': 0.9}))
  assert rep4['accuracy']['delta'] == pytest.approx(-0.1)


def test_performance_decay_missing_and_bad_types():
  '''performance_decay handles missing metrics and bad types.'''
  rep = performance_decay({'accuracy': 0.9, 'extra': 0.5},
                          _params(baseline_metrics={'accuracy': 0.9}))
  assert np.isnan(rep['extra']['delta'])
  with pytest.raises(ValueError, match='current_metrics must be'):
    performance_decay('bad', {})
  with pytest.raises(ValueError, match='baseline_metrics must be'):
    performance_decay({'accuracy': 0.9}, 'bad')


def test_performance_decay_nonfinite_delta_and_relative():
  '''performance_decay returns NaN for non-finite delta/relative.'''
  rep = performance_decay({'accuracy': 1e308},
                          _params(baseline_metrics={
                            'accuracy': -1e308}))
  assert np.isnan(rep['accuracy']['delta'])
  rep2 = performance_decay({'accuracy': 1e308},
                           _params(baseline_metrics={
                             'accuracy': 1e-308}))
  assert np.isnan(rep2['accuracy']['relative_delta'])


def test_threshold_pair_and_resolve_variants():
  '''Threshold helpers handle scalar/None/alias/bare maps.'''
  from forecast_ml_framework.pipelines.monitoring.nodes import (
    _resolve_thresholds, _threshold_pair)
  assert _threshold_pair(0.2) == (0.2, None)
  assert _threshold_pair(None) == (None, None)
  assert _threshold_pair({'low': 0.1, 'high': 0.3}) == (0.1, 0.3)
  assert _threshold_pair({'medium': 0.15}) == (0.15, None)
  merged = _resolve_thresholds({'thresholds': {
    'warning': 0.2, 'critical': 0.3}, 'psi_threshold': 0.33})
  assert merged['psi']['warning'] == pytest.approx(0.33)
  assert merged['ks']['warning'] == pytest.approx(0.2)
  assert merged['ks']['critical'] == pytest.approx(0.3)
  assert merged['missing_delta'] == merged['missing_rate_delta']


def test_evaluate_alerts_invalid_and_skipped_sections(caplog):
  '''evaluate_alerts handles invalid reports and bad stats.'''
  assert evaluate_alerts({'num': 'bad'}, _params()) == []
  with caplog.at_level(logging.WARNING):
    assert evaluate_alerts('bad-report', _params()) == []
  assert evaluate_alerts(report={'num': {'psi': 0.5}},
                         params=_params())
  combined = {'prediction': 'bad', 'performance': {
    'accuracy': 'bad', 'rmse': {'delta': -0.5}}}
  alerts = evaluate_alerts(combined, _params())
  assert any(a['feature'] == 'rmse' for a in alerts)


def test_build_monitoring_report_kwargs_forms():
  '''build_monitoring_report accepts predictions/metrics kwargs.'''
  ref, cur = _frames(shift=0.0)
  report = build_monitoring_report(
    compute_drift(ref, cur, _params()),
    params=_params(baseline_metrics={'accuracy': 0.9}),
    predictions=(np.zeros(100), np.ones(100)),
    metrics={'accuracy': 0.8})
  assert report['performance']['accuracy']['delta'] == pytest.approx(
    -0.1)
  assert 'prediction' in report


# --- W9 cycle-1 fixes (RED) ---

def test_psi_clipped_then_renormalized_value():
  '''PSI matches clip-then-renormalize reference computation.'''
  from forecast_ml_framework.pipelines.monitoring.nodes import (
    _psi_from_props)
  ref = [0.1] * 10
  cur = [0.1111111111111111] * 9 + [0.0]
  eps = 1e-4

  def _expected(ref_props, cur_props):
    ref_c = np.clip(np.asarray(ref_props, dtype=float), eps, None)
    cur_c = np.clip(np.asarray(cur_props, dtype=float), eps, None)
    ref_n = ref_c / ref_c.sum()
    cur_n = cur_c / cur_c.sum()
    assert cur_c.sum() > 1.0
    assert ref_n.sum() == pytest.approx(1.0)
    assert cur_n.sum() == pytest.approx(1.0)
    return float(np.sum((cur_n - ref_n) * np.log(cur_n / ref_n)))

  assert _psi_from_props(ref, cur, eps) == pytest.approx(
    _expected(ref, cur))


def test_psi_epsilon_monotonic_sensible():
  '''Larger epsilon smooths PSI downward for a fixed shift.'''
  from forecast_ml_framework.pipelines.monitoring.nodes import (
    _psi_from_props)
  ref = [0.1] * 10
  cur = [0.1111111111111111] * 9 + [0.0]
  small = _psi_from_props(ref, cur, 1e-4)
  large = _psi_from_props(ref, cur, 0.01)
  assert 0.0 < large < small


def test_psi_clipped_props_sum_to_one():
  '''Clipped proportion vectors are renormalized to sum 1.'''
  from forecast_ml_framework.pipelines.monitoring.nodes import (
    _clipped_props)
  for props in ([0.0] * 200, [0.1] * 10, [0.5, 0.0, 0.5]):
    clipped = _clipped_props(props, 0.01)
    assert float(np.sum(clipped)) == pytest.approx(1.0)
    assert np.all(clipped >= 0.01 / 2.0)


def test_performance_decay_rmse_improvement_is_negative():
  '''RMSE improvement (lower is better) gives negative decay.'''
  report = performance_decay(
    {'rmse': 0.4}, _params(baseline_metrics={'rmse': 0.5}))
  assert report['rmse']['decay'] == pytest.approx(-0.1)
  assert report['rmse']['delta'] == pytest.approx(-0.1)


def test_performance_decay_accuracy_drop_is_positive():
  '''Accuracy drop (higher is better) gives positive decay.'''
  report = performance_decay(
    {'accuracy': 0.8}, _params(baseline_metrics={'accuracy': 0.9}))
  assert report['accuracy']['decay'] == pytest.approx(0.1)


def test_performance_decay_r2_family_higher_better():
  '''r2-family metrics are higher-is-better, not lower.'''
  report = performance_decay(
    {'r2': 0.8, 'adj_r2': 0.85},
    _params(baseline_metrics={'r2': 0.9, 'adj_r2': 0.8}))
  assert report['r2']['decay'] == pytest.approx(0.1)
  assert report['adj_r2']['decay'] == pytest.approx(-0.05)


def test_performance_decay_lower_better_family():
  '''MAE/log-loss worsening gives positive decay.'''
  report = performance_decay(
    {'mae': 1.2, 'log_loss': 0.3},
    _params(baseline_metrics={'mae': 1.0, 'log_loss': 0.4}))
  assert report['mae']['decay'] == pytest.approx(0.2)
  assert report['log_loss']['decay'] == pytest.approx(-0.1)


def test_resolve_thresholds_conf_aligned_defaults():
  '''Defaults match conf: ks warning 0.05, psi 0.1/0.25.'''
  from forecast_ml_framework.pipelines.monitoring.nodes import (
    _resolve_thresholds)
  merged = _resolve_thresholds({})
  assert merged['psi'] == {'warning': 0.1, 'critical': 0.25}
  assert merged['ks'] == {'warning': 0.05, 'critical': 0.25}
  assert merged['missing_rate_delta'] == {
    'warning': 0.05, 'critical': 0.1}


def test_resolve_thresholds_flat_critical_and_missing_keys():
  '''Flat critical/missing params keys override thresholds.'''
  from forecast_ml_framework.pipelines.monitoring.nodes import (
    _resolve_thresholds)
  merged = _resolve_thresholds({
    'psi_critical_threshold': 0.3,
    'ks_threshold': 0.07,
    'ks_critical_threshold': 0.4,
    'missing_rate_delta_threshold': 0.06,
    'missing_rate_delta_critical_threshold': 0.2,
  })
  assert merged['psi']['critical'] == pytest.approx(0.3)
  assert merged['ks'] == {'warning': 0.07, 'critical': 0.4}
  assert merged['missing_rate_delta'] == {
    'warning': 0.06, 'critical': 0.2}
  assert merged['missing_delta'] == merged['missing_rate_delta']
