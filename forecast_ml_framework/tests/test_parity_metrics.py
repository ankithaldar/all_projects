#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Parity tests for regression descriptive stats, confusion table, adj_r2.'''

import pytest

from forecast_ml_framework.pipelines.modeling.metrics import (
  compute_metrics,
  confusion_rates_table,
)


def test_descriptive_selected():
  '''Selected descriptive metrics on [1,2,3]/[1,1,4].'''
  vals = compute_metrics(
      [1, 2, 3], [1, 1, 4], None, 'regression',
      selected={'target_sum', 'pred_mean', 'total_count',
                'max_pred_error'})
  assert vals['target_sum'] == 6
  assert vals['total_count'] == 3
  assert vals['pred_mean'] == pytest.approx(2.0)
  assert vals['max_pred_error'] == 1


def test_descriptive_all():
  '''All descriptive target/pred stats are correct.'''
  vals = compute_metrics(
      [1, 2, 3], [1, 1, 4], None, 'regression',
      selected={'target_sum', 'target_min', 'target_max',
                'target_mean', 'pred_sum', 'pred_min',
                'pred_max', 'pred_mean'})
  assert vals['target_min'] == 1
  assert vals['target_max'] == 3
  assert vals['target_mean'] == pytest.approx(2.0)
  assert vals['pred_sum'] == 6
  assert vals['pred_min'] == 1
  assert vals['pred_max'] == 4


def test_max_pred_error_signed_not_abs():
  '''Signed max(target-pred), not max abs error.'''
  vals = compute_metrics(
      [1, 2, 3], [4, 1, 4], None, 'regression',
      selected={'max_pred_error', 'max_error'})
  assert vals['max_pred_error'] == 1
  assert vals['max_error'] == 3


def test_confusion_rates_table():
  '''Confusion rates table rows for [1,1,0,0]/[1,0,0,0].'''
  rows = confusion_rates_table([1, 1, 0, 0], [1, 0, 0, 0])
  assert rows[0]['tpr'] == pytest.approx(0.5)
  assert rows[0]['fpr'] == pytest.approx(0.0)
  assert rows[0]['fnr'] == pytest.approx(0.5)
  assert rows[0]['tnr'] == pytest.approx(1.0)
  labels = [r.get('label', r.get('Actual')) for r in rows]
  assert 'Actual Positive' in labels
  assert 'Actual Negative' in labels


def test_confusion_rates_table_rows_differ_asymmetric():
  '''ML-24b: Actual Positive/Negative rows differ on asymmetric cm.

  y_true=[0,0,1,1], y_pred=[0,1,1,1] gives tn=1, fp=1,
  fn=0, tp=2, so tpr=1.0, fpr=0.5, fnr=0.0, tnr=0.5.
  The Actual Positive row carries those rates; the
  Actual Negative row inverts roles (Negative as the
  reference class), so its tpr is tnr and its tnr is
  tpr. Identical rows would mean the old duplicated
  dict bug is still present.
  '''
  rows = confusion_rates_table([0, 0, 1, 1], [0, 1, 1, 1])
  pos, neg = rows[0], rows[1]
  assert pos['label'] == 'Actual Positive'
  assert neg['label'] == 'Actual Negative'
  assert pos['tpr'] == pytest.approx(1.0)
  assert pos['fnr'] == pytest.approx(0.0)
  assert pos['fpr'] == pytest.approx(0.5)
  assert pos['tnr'] == pytest.approx(0.5)
  assert neg['tpr'] == pytest.approx(0.5)
  assert neg['tnr'] == pytest.approx(1.0)
  assert neg['fpr'] == pytest.approx(0.0)
  assert neg['fnr'] == pytest.approx(0.5)
  assert pos['tpr'] != neg['tpr']


def test_adj_r2_n_features():
  '''adj_r2 with n_features=2 differs from old approx.'''
  from sklearn.metrics import r2_score
  y_true = [1, 2, 3, 4, 5]
  y_pred = [1, 2, 3, 4, 4]
  vals = compute_metrics(
      y_true, y_pred, None, 'regression',
      selected={'adj_r2'}, n_features=2)
  r2 = r2_score(y_true, y_pred)
  expected = 1 - (1 - r2) * 4 / (5 - 2 - 1)
  old = 1 - (1 - r2) * 4 / max(5 - 2, 1)
  assert vals['adj_r2'] == pytest.approx(expected)
  assert vals['adj_r2'] != pytest.approx(old)


def test_adj_r2_default_one():
  '''Default adj_r2 matches old p=1 formula.'''
  from sklearn.metrics import r2_score
  y_true = [1, 2, 3, 4, 5]
  y_pred = [1, 2, 3, 4, 4]
  vals = compute_metrics(
      y_true, y_pred, None, 'regression', selected={'adj_r2'})
  r2 = r2_score(y_true, y_pred)
  assert vals['adj_r2'] == pytest.approx(1 - (1 - r2) * 4 / 3)


def test_compute_metrics_task_aliases_cls_reg():
  '''Task aliases cls/binary/reg (any case) dispatch correctly.'''
  proba = [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]]
  for alias in ('cls', 'binary', 'CLASS', 'Classification'):
    vals = compute_metrics(
        [0, 1, 0, 1], [0, 1, 0, 1], proba, alias,
        selected={'accuracy'})
    assert vals['accuracy'] == pytest.approx(1.0), alias
  for alias in ('reg', 'REG', 'Regression'):
    vals = compute_metrics(
        [1, 2, 3], [1, 1, 4], None, alias,
        selected={'target_sum'})
    assert vals['target_sum'] == 6, alias
