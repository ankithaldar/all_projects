#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Regression tests for cycle-1 metrics/losses fixes (ML-5 to ML-24a, F4).'''

import math

import numpy as np
import pytest

from forecast_ml_framework.losses import huber_loss
from forecast_ml_framework.pipelines.modeling.metrics import (
  _at_k,
  _huber,
  _ks,
  _log_loss,
  _pos_proba,
  _smape,
  _top_decile,
  _top_k,
  compute_metrics,
)


def test_log_loss_single_class_one():
  '''ML-5: single-class 1 log loss is -log(0.8), not NaN/crash.'''
  vals = compute_metrics(
      [1, 1, 1], [1, 1, 1], [[0.2, 0.8]] * 3, 'classification',
      selected={'log_loss'})
  assert vals['log_loss'] == pytest.approx(-math.log(0.8))
  assert _log_loss([1, 1, 1], [[0.2, 0.8]] * 3) == pytest.approx(
      0.2231435513142097)


def test_log_loss_single_class_zero():
  '''ML-5: single-class 0 log loss is -log(0.8), not NaN/crash.'''
  vals = compute_metrics(
      [0, 0, 0], [0, 0, 0], [[0.8, 0.2]] * 3, 'classification',
      selected={'log_loss'})
  assert vals['log_loss'] == pytest.approx(-math.log(0.8))
  assert _log_loss([0, 0, 0], [[0.8, 0.2]] * 3) == pytest.approx(
      0.2231435513142097)


def test_g_mean_binary_geometric():
  '''ML-6: binary g-mean is sqrt(sens*spec), not arithmetic mean.'''
  vals = compute_metrics(
      [1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 1],
      [[0.1, 0.9]] * 8, 'classification',
      selected={'g_mean', 'sensitivity', 'specificity'})
  assert vals['sensitivity'] == pytest.approx(0.5)
  assert vals['specificity'] == pytest.approx(0.75)
  assert vals['g_mean'] == pytest.approx(math.sqrt(0.5 * 0.75))
  assert vals['g_mean'] == pytest.approx(0.6123724356957945)
  assert vals['g_mean'] != pytest.approx(0.625)


def test_ks_minus_one_one():
  '''ML-7: KS on {-1,1} matches KS on remapped {0,1}.'''
  proba = [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
  assert _ks([-1, -1, 1, 1], proba) == pytest.approx(1.0)
  assert _ks([-1, -1, 1, 1], proba) == pytest.approx(
      _ks([0, 0, 1, 1], proba))
  vals = compute_metrics(
      [-1, -1, 1, 1], [-1, -1, 1, 1], proba, 'classification',
      selected={'ks'})
  assert vals['ks'] == pytest.approx(1.0)


def test_ks_yes_no():
  '''ML-7: KS on {no,yes} matches KS on remapped {0,1}.'''
  proba = [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
  assert _ks(['no', 'no', 'yes', 'yes'], proba) == pytest.approx(1.0)
  assert _ks(['no', 'no', 'yes', 'yes'], proba) == pytest.approx(
      _ks([0, 0, 1, 1], proba))
  vals = compute_metrics(
      ['no', 'no', 'yes', 'yes'], ['no', 'no', 'yes', 'yes'], proba,
      'classification', selected={'ks'})
  assert vals['ks'] == pytest.approx(1.0)


def test_pos_proba_and_top_k_non_zero_one():
  '''ML-7: positive proba/top-k use max sorted label as positive.'''
  proba = [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
  for labels in ([-1, -1, 1, 1], ['no', 'no', 'yes', 'yes']):
    assert _pos_proba(proba, labels).tolist() == pytest.approx(
        [0.1, 0.2, 0.8, 0.9])
    assert _top_k(labels, proba, 1) == pytest.approx(1.0)
    assert _top_k(labels, proba, 1) == pytest.approx(
        _top_k([0, 0, 1, 1], proba, 1))


def test_pr_auc_is_trapezoid_not_average_precision():
  '''ML-8: pr_auc is auc(recall, precision), AP stays separate.'''
  y_true = [0, 0, 1, 1, 0, 1]
  scores = [0.1, 0.4, 0.35, 0.8, 0.2, 0.9]
  proba = [[1 - s, s] for s in scores]
  vals = compute_metrics(
      y_true, y_true, proba, 'classification',
      selected={'pr_auc', 'pr_auc_ovr', 'average_precision', 'map'})
  assert vals['pr_auc'] == pytest.approx(0.9027777777777777)
  assert vals['pr_auc_ovr'] == pytest.approx(0.9027777777777777)
  assert vals['average_precision'] == pytest.approx(
      0.9166666666666665)
  assert vals['map'] == pytest.approx(0.9166666666666665)
  assert vals['pr_auc'] != pytest.approx(vals['average_precision'])


@pytest.mark.parametrize('name', [
    'msle', 'tweedie_deviance', 'poisson_deviance', 'gamma_deviance'])
def test_out_of_support_negative_y_true_is_nan(name):
  '''ML-21: negative y_true gives NaN, not fabricated 0.0.'''
  vals = compute_metrics(
      [-5, 10], [-5, 10], None, 'regression', selected={name})
  assert math.isnan(vals[name])
  assert vals[name] != 0.0


@pytest.mark.parametrize('name', [
    'msle', 'tweedie_deviance', 'poisson_deviance', 'gamma_deviance'])
def test_in_support_positive_deviances_are_finite(name):
  '''ML-21: in-support positive data still gives finite deviances.'''
  vals = compute_metrics(
      [1, 2, 3], [1, 2, 3], None, 'regression', selected={name})
  assert vals[name] == pytest.approx(0.0)


def test_at_k_multiclass_documented_formula():
  '''ML-22: multiclass recall_at_k is correct_top / n (documented).'''
  y_true = [0, 1, 2, 0, 1, 2]
  proba = np.array(
      [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.35, 0.3, 0.35],
       [0.35, 0.3, 0.35], [0.35, 0.35, 0.3], [0.35, 0.3, 0.35]])
  precision, recall, f1 = _at_k(y_true, y_true, proba, 2)
  assert precision == pytest.approx(1.0)
  assert recall == pytest.approx(2 / 6)
  assert f1 == pytest.approx(0.5)


def test_at_k_binary_unchanged():
  '''ML-22: binary recall_at_k stays TP / all positives.'''
  proba = [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
  precision, recall, f1 = _at_k([0, 0, 1, 1], [0, 0, 1, 1], proba, 2)
  assert precision == pytest.approx(1.0)
  assert recall == pytest.approx(1.0)
  assert f1 == pytest.approx(1.0)


def test_top_decile_multiclass_documented_formula():
  '''ML-22: multiclass lift is accuracy_top / accuracy_overall.'''
  y_true = [0, 1, 2, 0, 1, 2]
  proba = np.array(
      [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.35, 0.3, 0.35],
       [0.35, 0.3, 0.35], [0.35, 0.35, 0.3], [0.35, 0.3, 0.35]])
  lift, gain = _top_decile(y_true, y_true, proba)
  # Top decile is ceil(0.1*6)=1 row, correct; overall accuracy 1.0.
  assert lift == pytest.approx(1.0)
  assert gain == pytest.approx(1 / 6)


def test_smape_lists_match_arrays():
  '''ML-24a: _smape accepts lists exactly like arrays.'''
  assert _smape([1, 2], [1, 2]) == pytest.approx(
      _smape(np.array([1.0, 2.0]), np.array([1.0, 2.0])))
  assert _smape([1, 2], [2, 4]) == pytest.approx(2 / 3)


def test_huber_matches_losses_single_home():
  '''F4: metrics._huber delegates to losses.huber_loss with delta.'''
  y_true = [0.0, 1.0, 2.0, 3.0]
  y_pred = [0.5, 1.0, 1.0, 4.0]
  assert _huber(y_true, y_pred) == pytest.approx(
      huber_loss(y_true, y_pred, delta=1.35))
  assert _huber(y_true, y_pred, 0.5) == pytest.approx(
      huber_loss(y_true, y_pred, delta=0.5))
  vals = compute_metrics(
      y_true, y_pred, None, 'regression', selected={'huber_loss'})
  assert vals['huber_loss'] == pytest.approx(
      huber_loss(y_true, y_pred, delta=1.35))
