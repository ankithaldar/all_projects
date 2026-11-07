#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Parity Task 2: evaluate_model by_centile + selected_ntile.'''

import warnings

import numpy as np
import pandas as pd
import pytest

from forecast_ml_framework.pipelines.modeling.evaluate import evaluate_model


def _cls_data(n=1000):
  rng = np.random.default_rng(0)
  proba_pos = rng.random(n)
  y_true = (proba_pos > 0.5).astype(int)
  y_pred = (proba_pos >= 0.5).astype(int)
  y_proba = np.column_stack([1 - proba_pos, proba_pos])
  x = pd.DataFrame({'f1': rng.random(n)})
  return y_true, y_pred, y_proba, x


def _reg_data(n=1000):
  rng = np.random.default_rng(1)
  y_true = rng.normal(size=n)
  y_pred = y_true + rng.normal(scale=0.1, size=n)
  x = pd.DataFrame({'f1': rng.random(n)})
  return y_true, y_pred, None, x


def test_cls_lift_decile_selected_ntile_matches_top_decile():
  '''LIFT_DECILE_10 selects the top (1-based label 10) decile.

  The selected score is the tile-vs-overall lift (top-tile
  positive rate / overall prevalence), not the within-tile
  lift_top_decile diagnostic stored in by_decile.
  '''
  y_true, y_pred, y_proba, x = _cls_data()
  params = {
      'task_type': 'classification',
      'metrics': ['lift_top_decile', 'roc_auc', 'accuracy'],
      'eval_metric': 'LIFT_DECILE_10',
  }
  out = evaluate_model(y_true, y_pred, y_proba, x, params)
  assert out['selected_ntile']['eval_metric'] == 'LIFT_DECILE_10'
  assert out['selected_ntile']['metric'] == 'lift'
  assert out['selected_ntile']['ntile'] == 10
  assert isinstance(out['selected_ntile']['score'], float)
  assert '10' in out['by_decile']
  from forecast_ml_framework.pipelines.modeling.evaluate import (
    _deciles,
  )
  labels = _deciles(y_true, y_pred, y_proba, 'classification')
  top_mask = labels == labels.max()
  prevalence = float(np.mean(np.asarray(y_true) == 1))
  expected = float(np.mean(np.asarray(y_true)[top_mask] == 1)) / prevalence
  assert out['selected_ntile']['score'] == pytest.approx(expected)
  assert 'by_centile' not in out


def test_reg_rmse_centile_selected_ntile_present():
  '''RMSE_CENTILE_100 builds by_centile and selects top centile.'''
  y_true, y_pred, y_proba, x = _reg_data()
  params = {
      'task_type': 'regression',
      'metrics': ['rmse', 'mae'],
      'eval_metric': 'RMSE_CENTILE_100',
  }
  out = evaluate_model(y_true, y_pred, y_proba, x, params)
  assert 'by_centile' in out
  assert len(out['by_centile']) <= 100
  assert out['selected_ntile'] is not None
  assert out['selected_ntile']['metric'] == 'rmse'
  assert out['selected_ntile']['ntile'] == 100
  top_key = max(out['by_centile'], key=lambda k: int(k))
  assert out['selected_ntile']['score'] == out['by_centile'][top_key][
      'rmse']


def test_no_eval_metric_selected_none_old_keys_unchanged():
  '''Without eval_metric, selected_ntile is None, old keys present.'''
  y_true, y_pred, y_proba, x = _cls_data()
  params = {
      'task_type': 'classification',
      'metrics': ['roc_auc', 'accuracy'],
  }
  out = evaluate_model(y_true, y_pred, y_proba, x, params)
  assert out['selected_ntile'] is None
  assert 'by_centile' not in out
  assert 'global' in out and 'by_bucket' in out and 'by_decile' in out
  assert out['global']['roc_auc'] is not None


def test_reg_overall_eval_metric_uses_global():
  '''Bare RMSE (overall) selects the global score.'''
  y_true, y_pred, y_proba, x = _reg_data()
  params = {
      'task_type': 'regression',
      'metrics': ['rmse'],
      'eval_metric': 'RMSE',
  }
  out = evaluate_model(y_true, y_pred, y_proba, x, params)
  assert out['selected_ntile']['ntile'] == 0
  assert out['selected_ntile']['score'] == out['global']['rmse']
  assert 'by_centile' not in out


def test_unknown_eval_metric_warns_and_falls_back():
  '''Unknown eval_metric warns and still returns a selection.'''
  y_true, y_pred, y_proba, x = _cls_data()
  params = {
      'task_type': 'classification',
      'metrics': ['lift_top_decile', 'roc_auc'],
      'eval_metric': 'FOO_DECILE_3',
  }
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    out = evaluate_model(y_true, y_pred, y_proba, x, params)
  assert any(issubclass(w.category, UserWarning) for w in caught)
  assert out['selected_ntile'] is not None
  assert isinstance(out['selected_ntile']['score'], float)


def test_lift_quintile_top_quintile():
  '''LIFT_QUINTILE_5 selects top quintile, not 5th decile.'''
  from forecast_ml_framework.pipelines.modeling.evaluate import (
    _by_group,
    _deciles,
  )
  from forecast_ml_framework.pipelines.modeling.metrics import (
    resolve_selected_metrics,
  )
  y_true, y_pred, y_proba, x = _cls_data()
  params = {
      'task_type': 'classification',
      'metrics': ['lift_top_decile', 'roc_auc', 'accuracy'],
      'eval_metric': 'LIFT_QUINTILE_5',
  }
  out = evaluate_model(y_true, y_pred, y_proba, x, params)
  assert out['selected_ntile']['ntile'] == 5
  assert out['selected_ntile']['metric'] == 'lift'
  selected = resolve_selected_metrics(
      params['metrics'], params['task_type'])
  effective = list(selected)
  if 'lift_top_decile' not in effective:
    effective.append('lift_top_decile')
  labels = _deciles(y_true, y_pred, y_proba, params['task_type'], n=5)
  expected = _by_group(y_true, y_pred, y_proba, labels,
                       params['task_type'], effective)
  top_label = max(expected, key=lambda k: int(k))
  assert top_label == '5'
  top_mask = labels == labels.max()
  prevalence = float(np.mean(np.asarray(y_true) == 1))
  expected_lift = float(
      np.mean(np.asarray(y_true)[top_mask] == 1)) / prevalence
  assert out['selected_ntile']['score'] == pytest.approx(expected_lift)


def test_evaluate_task_alias_cls_and_reg():
  '''evaluate_model honours task aliases cls/reg like validation.'''
  y_true, y_pred, y_proba, x = _cls_data()
  params = {
      'task_type': 'cls',
      'metrics': ['accuracy', 'lift_top_decile'],
      'eval_metric': 'LIFT_DECILE_10',
  }
  out = evaluate_model(y_true, y_pred, y_proba, x, params)
  assert 'accuracy' in out['global']
  assert out['selected_ntile']['metric'] == 'lift'
  _y_true, _y_pred, _y_proba, _x = _reg_data()
  params_reg = {
      'task_type': 'REG',
      'metrics': ['rmse'],
      'eval_metric': 'RMSE',
  }
  out_reg = evaluate_model(_y_true, _y_pred, _y_proba, _x, params_reg)
  assert 'rmse' in out_reg['global']
  assert out_reg['selected_ntile']['score'] == out_reg['global']['rmse']


def test_pipeline_predict_proba_task_alias():
  '''_predict_proba returns proba for cls alias, None for reg.'''
  from forecast_ml_framework.pipelines.modeling.pipeline import (
    _predict_proba,
  )

  class _Model:

    def predict_proba(self, x):
      return [[0.5, 0.5]]

  assert _predict_proba(_Model(), None, 'cls') == [[0.5, 0.5]]
  assert _predict_proba(_Model(), None, 'BINARY') == [[0.5, 0.5]]
  assert _predict_proba(_Model(), None, 'reg') is None
  assert _predict_proba(_Model(), None, 'Regression') is None
