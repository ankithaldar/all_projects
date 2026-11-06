#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for select_threshold (parity Task 5).'''

import numpy as np

from forecast_ml_framework.pipelines.modeling.threshold import (
  select_threshold,
)


def _separable(n=100):
  '''Return y/proba where proba perfectly separates classes.'''
  y_true = np.array([0] * (n // 2) + [1] * (n // 2))
  proba = np.concatenate([
    np.linspace(0.0, 0.4, n // 2),
    np.linspace(0.6, 0.9, n // 2),
  ])
  return y_true, proba


def test_pr_curve_threshold_in_range():
  '''pr_curve returns a threshold in (0, 1) on separable data.'''
  y_true, proba = _separable()
  thr = select_threshold(y_true, proba, 'pr_curve')
  assert 0 < thr < 1


def test_roc_curve_threshold_in_range():
  '''roc_curve returns a threshold in (0, 1) on separable data.'''
  y_true, proba = _separable()
  thr = select_threshold(y_true, proba, 'roc_curve')
  assert 0 < thr < 1


def test_mcc_curve_threshold_in_range():
  '''mcc_curve returns a threshold in (0, 1) on separable data.'''
  y_true, proba = _separable()
  thr = select_threshold(y_true, proba, 'mcc_curve')
  assert 0 < thr < 1


def test_ks_threshold_on_perfect_separation():
  '''ks_statistics returns >= 0.5 on perfect separation.'''
  y_true, proba = _separable()
  thr = select_threshold(y_true, proba, 'ks_statistics')
  assert 0 < thr < 1
  assert thr >= 0.5


def test_unknown_method_returns_half():
  '''Unknown method falls back to 0.5.'''
  y_true, proba = _separable()
  assert select_threshold(y_true, proba, 'not_a_method') == 0.5


def test_empty_and_single_class_guard():
  '''Empty / single-class input falls back to 0.5, never crashes.'''
  assert select_threshold(np.array([]), np.array([]),
                          'ks_statistics') == 0.5
  assert select_threshold(np.ones(10, dtype=int),
                          np.linspace(0.1, 0.9, 10),
                          'pr_curve') == 0.5


def test_roc_curve_never_returns_inf():
  '''roc_curve excludes the inf entry; result is finite in [0, 1].'''
  y_true, proba = _separable()
  thr = select_threshold(y_true, proba, 'roc_curve')
  assert np.isfinite(thr)
  assert 0 <= thr <= 1


def test_roc_curve_fallback_half_when_best_gmean_zero():
  '''Constant proba gives best g-mean 0, so roc_curve falls back 0.5.'''
  y_true = np.array([0] * 50 + [1] * 50)
  proba = np.full(100, 0.5)
  assert select_threshold(y_true, proba, 'roc_curve') == 0.5


def test_threshold_nan_proba_returns_half():
  '''NaN proba returns 0.5 for every method, never crashes.'''
  y_true, proba = _separable()
  proba = proba.copy()
  proba[3] = np.nan
  for method in ('pr_curve', 'roc_curve', 'mcc_curve',
                 'ks_statistics'):
    assert select_threshold(y_true, proba, method) == 0.5


def test_threshold_non_finite_proba_returns_half():
  '''Inf/-inf proba returns 0.5 for every method.'''
  y_true, proba = _separable()
  for bad in (np.inf, -np.inf):
    proba = proba.copy()
    proba[3] = bad
    for method in ('pr_curve', 'roc_curve', 'mcc_curve',
                   'ks_statistics'):
      assert select_threshold(y_true, proba, method) == 0.5


def test_threshold_out_of_range_proba_returns_half():
  '''Out-of-range proba (<0 or >1) returns 0.5.'''
  y_true, proba = _separable()
  proba = proba.copy()
  proba[3] = -0.5
  for method in ('pr_curve', 'roc_curve', 'mcc_curve',
                 'ks_statistics'):
    assert select_threshold(y_true, proba, method) == 0.5
  proba = proba.copy()
  proba[3] = 4.36
  for method in ('pr_curve', 'roc_curve', 'mcc_curve',
                 'ks_statistics'):
    assert select_threshold(y_true, proba, method) == 0.5


def test_threshold_length_mismatch_raises():
  '''Mismatched y_true/proba lengths raise ValueError.'''
  y_true, proba = _separable()
  import pytest
  with pytest.raises(ValueError, match='length'):
    select_threshold(y_true[:-1], proba, 'pr_curve')
  with pytest.raises(ValueError, match='length'):
    select_threshold(y_true, proba[:-1], 'ks_statistics')


def test_threshold_method_case_insensitive():
  '''Method is case-insensitive (upper/mixed still works).'''
  y_true, proba = _separable()
  assert select_threshold(y_true, proba, 'PR_CURVE') == (
    select_threshold(y_true, proba, 'pr_curve'))
  assert select_threshold(y_true, proba, 'Roc_Curve') == (
    select_threshold(y_true, proba, 'roc_curve'))


def test_threshold_unknown_method_warns():
  '''Unknown method warns and returns 0.5.'''
  import warnings
  y_true, proba = _separable()
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    assert select_threshold(
      y_true, proba, 'not_a_method') == 0.5
  assert caught
  assert issubclass(caught[0].category, UserWarning)
