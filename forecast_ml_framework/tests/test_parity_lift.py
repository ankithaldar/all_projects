#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for pandas lift table (parity Task 4).'''

import numpy as np
import pandas as pd
import pytest

from forecast_ml_framework.pipelines.modeling.lift_table import (
  compute_lift_table,
)

COLUMNS = [
  'decile', 'min_prob', 'max_prob', 'avg_prob', 'decile_count',
  'non_target_count', 'target_count', 'target_percentage',
  'non_target_percentage', 'cum_target_percentage',
  'cum_non_target_percentage', 'ks', 'lift',
  'decile_disconnect_rate', 'predicted_positive', 'predicted_negative',
  'tp', 'fp', 'tn', 'fn', 'precision', 'recall',
  'customers_at_or_above_threshold',
  'true_positive_at_or_above_threshold',
  'precision_at_or_above_threshold',
  'recall_at_or_above_threshold', 'tp_above_threshold',
  'fp_above_threshold', 'tn_above_threshold', 'fn_above_threshold',
  'precision_above_threshold', 'recall_above_threshold',
]


def _separable(n=100):
  '''Return y/proba where proba perfectly separates classes.'''
  y_true = np.array([0] * (n // 2) + [1] * (n // 2))
  proba = np.concatenate([
    np.linspace(0.0, 0.4, n // 2),
    np.linspace(0.6, 0.9, n // 2),
  ])
  y_pred = (proba >= 0.5).astype(int)
  return y_true, y_pred, proba


def test_lift_table_columns_and_row_count():
  '''Lift table has parity columns (corrected spelling), 10 rows.'''
  y_true, y_pred, proba = _separable()
  out = compute_lift_table(y_true, y_pred, proba)
  assert list(out.columns) == COLUMNS
  assert len(out) == 10


def test_lift_table_sorted_decile_desc_and_top_lift():
  '''Rows sorted decile desc; top decile lift > 1 on separable data.'''
  y_true, y_pred, proba = _separable()
  out = compute_lift_table(y_true, y_pred, proba)
  assert list(out['decile']) == sorted(out['decile'], reverse=True)
  assert out.iloc[0]['lift'] > 1


def test_lift_table_ks_and_cumulative():
  '''KS within [0, 100] and cumulative percentages end at 100.'''
  y_true, y_pred, proba = _separable()
  out = compute_lift_table(y_true, y_pred, proba)
  assert out['ks'].between(0, 100).all()
  assert out.iloc[-1]['cum_target_percentage'] == 100
  assert out.iloc[-1]['cum_non_target_percentage'] == 100


def test_lift_table_confusion_sums_to_decile_count():
  '''tp+fp+tn+fn equals decile_count per row.'''
  y_true, y_pred, proba = _separable()
  out = compute_lift_table(y_true, y_pred, proba)
  total = out['tp'] + out['fp'] + out['tn'] + out['fn']
  assert (total == out['decile_count']).all()


def test_lift_table_empty_guard():
  '''Empty input returns empty DataFrame, never crashes.'''
  out = compute_lift_table(
    np.array([]), np.array([]), np.array([]))
  assert isinstance(out, pd.DataFrame)
  assert len(out) == 0
  assert list(out.columns) == COLUMNS


def test_lift_table_single_class_guard():
  '''Single-class y_true returns empty DataFrame, never crashes.'''
  out = compute_lift_table(
    np.ones(20, dtype=int), np.ones(20, dtype=int),
    np.linspace(0.1, 0.9, 20))
  assert isinstance(out, pd.DataFrame)
  assert len(out) == 0


def _pct_expected(part, whole):
  '''Return part/whole*100 for test expectations.'''
  return part / whole * 100 if whole else 0.0


def test_lift_table_threshold_filters_at_or_above_columns():
  '''Threshold subset cumsums: proba>=threshold drives at-or-above.'''
  y_true, y_pred, proba = _separable()
  none_out = compute_lift_table(y_true, y_pred, proba, threshold=None)
  out = compute_lift_table(y_true, y_pred, proba, threshold=0.6)
  expected_customers = int((proba >= 0.6).sum())
  assert out.iloc[-1]['customers_at_or_above_threshold'] == (
      expected_customers)
  assert out.iloc[-1]['true_positive_at_or_above_threshold'] == int(
      y_true[proba >= 0.6].sum())
  assert out.iloc[-1]['customers_at_or_above_threshold'] < (
      none_out.iloc[-1]['customers_at_or_above_threshold'])
  assert out.iloc[-1]['precision_at_or_above_threshold'] == pytest.approx(
      _pct_expected(y_true[proba >= 0.6].sum(), expected_customers))


def test_lift_table_threshold_none_keeps_top_decile_cumsum():
  '''Threshold None retains cumulative-from-top at-or-above columns.'''
  y_true, y_pred, proba = _separable()
  out = compute_lift_table(y_true, y_pred, proba, threshold=None)
  assert out.iloc[-1]['customers_at_or_above_threshold'] == len(y_true)
  assert out.iloc[-1]['true_positive_at_or_above_threshold'] == int(
      y_true.sum())


def test_lift_table_nan_proba_raises():
  '''NaN proba raises ValueError (validation pattern), not empty.'''
  y_true, y_pred, proba = _separable()
  proba = proba.copy()
  proba[0] = np.nan
  with pytest.raises(ValueError, match='proba'):
    compute_lift_table(y_true, y_pred, proba)


def test_lift_table_validates_n_lengths_and_binary_labels():
  '''n outside 1..100, mismatched lengths, non-binary labels raise.'''
  y_true, y_pred, proba = _separable()
  with pytest.raises(ValueError, match='n'):
    compute_lift_table(y_true, y_pred, proba, n=0)
  with pytest.raises(ValueError, match='n'):
    compute_lift_table(y_true, y_pred, proba, n=101)
  with pytest.raises(ValueError, match='length'):
    compute_lift_table(y_true[:-1], y_pred, proba)
  with pytest.raises(ValueError, match='binary'):
    bad = y_true.copy()
    bad[0] = 2
    compute_lift_table(bad, y_pred, proba)
  with pytest.raises(ValueError, match='binary'):
    bad_pred = y_pred.copy()
    bad_pred[0] = 2
    compute_lift_table(y_true, bad_pred, proba)


def test_lift_table_caps_q_when_n_exceeds_length():
  '''n larger than input length is capped, never crashes.'''
  y_true = np.array([0, 1] * 10)
  y_pred = y_true.copy()
  proba = np.linspace(0.05, 0.95, 20)
  out = compute_lift_table(y_true, y_pred, proba, n=100)
  assert len(out) <= 20
  assert len(out) >= 1


def test_lift_table_inf_proba_raises():
  '''Inf proba raises ValueError like NaN.'''
  y_true, y_pred, proba = _separable()
  for bad in (np.inf, -np.inf):
    proba = proba.copy()
    proba[0] = bad
    with pytest.raises(ValueError, match='proba'):
      compute_lift_table(y_true, y_pred, proba)


def test_lift_table_threshold_bool_rejected():
  '''Bool threshold (True/False) raises ValueError, not 1.0/0.0.'''
  y_true, y_pred, proba = _separable()
  for bad in (True, False):
    with pytest.raises(ValueError, match='threshold'):
      compute_lift_table(y_true, y_pred, proba, threshold=bad)


def test_lift_table_out_of_range_proba_raises():
  '''Proba outside [0, 1] (e.g. 999, -0.5) raises ValueError.'''
  y_true, y_pred, proba = _separable()
  for bad in (999.0, -0.5, 1.5):
    proba = proba.copy()
    proba[0] = bad
    with pytest.raises(ValueError, match='proba'):
      compute_lift_table(y_true, y_pred, proba)
