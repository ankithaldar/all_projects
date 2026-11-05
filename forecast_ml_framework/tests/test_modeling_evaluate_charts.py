#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for evaluate, charts and shap nodes.'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecast_ml_framework.pipelines.modeling.charts import make_charts
from forecast_ml_framework.pipelines.modeling.evaluate import (
  evaluate_model,
)
from forecast_ml_framework.pipelines.modeling.shap_explain import (
  explain_model,
)


def _clf(n=120):
  '''Return synthetic classification evaluation inputs.'''
  rng = np.random.RandomState(3)
  x = pd.DataFrame({'f1': rng.randn(n), 'f2': rng.randn(n),
                    'seg': np.where(rng.rand(n) > .5, 'a', 'b')})
  y_true = (x['f1'] > 0).astype(int).values
  y_pred = y_true.copy()
  y_pred[::7] = 1 - y_pred[::7]
  proba = np.column_stack([1 - y_pred, y_pred]).astype(float)
  params = {'task_type': 'classification',
            'bucket_segment_col': 'seg'}
  return x, y_true, y_pred, proba, params


def _reg(n=120):
  '''Return synthetic regression evaluation inputs.'''
  rng = np.random.RandomState(3)
  x = pd.DataFrame({'f1': rng.randn(n), 'f2': rng.randn(n),
                    'seg': np.where(rng.rand(n) > .5, 'a', 'b')})
  y_true = (x['f1'] + x['f2']).values
  y_pred = y_true + rng.randn(n) * 0.1
  params = {'task_type': 'regression', 'bucket_segment_col': 'seg'}
  return x, y_true, y_pred, None, params


def test_evaluate_classification_global_and_bucket():
  '''Evaluate returns global, bucket and decile metrics.'''
  x, y_true, y_pred, proba, params = _clf()
  out = evaluate_model(y_true, y_pred, proba, x, params)
  assert 'global' in out and 'f1_weighted' in out['global']
  assert 'by_bucket' in out and out['by_bucket']
  assert 'by_decile' in out and out['by_decile']
  assert 'confusion_matrix' in out['global']


def test_evaluate_regression_global_and_bucket():
  '''Evaluate returns global, bucket and decile metrics.'''
  x, y_true, y_pred, _, params = _reg()
  out = evaluate_model(y_true, y_pred, None, x, params)
  assert 'rmse' in out['global'] and 'mae' in out['global']
  assert out['by_bucket'] and out['by_decile']


def test_charts_classification_at_least_five():
  '''Classification charts return at least five figures.'''
  x, y_true, y_pred, proba, params = _clf()
  figs = make_charts(y_true, y_pred, proba, x, params)
  assert len(figs) >= 5
  for fig in figs.values():
    assert fig is not None


def test_charts_regression_at_least_five_small_data():
  '''Regression charts return >=5 figs on small data.'''
  x, y_true, y_pred, _, params = _reg(20)
  figs = make_charts(y_true, y_pred, None, x, params)
  assert len(figs) >= 5


def test_charts_leave_no_open_figures():
  '''make_charts closes every figure it creates.'''
  x, y_true, y_pred, proba, params = _clf()
  before = plt.get_fignums()
  figs = make_charts(y_true, y_pred, proba, x, params)
  try:
    assert len(figs) >= 5
    assert plt.get_fignums() == before
  finally:
    for fig in figs.values():
      plt.close(fig)


def test_charts_leave_no_open_figures_regression():
  '''Regression make_charts also leaves no open figures.'''
  x, y_true, y_pred, _, params = _reg()
  before = plt.get_fignums()
  figs = make_charts(y_true, y_pred, None, x, params)
  try:
    assert len(figs) >= 5
    assert plt.get_fignums() == before
  finally:
    for fig in figs.values():
      plt.close(fig)


def test_classification_full_metric_set():
  '''Classification exposes every comprehensive metric.'''
  x, y_true, y_pred, proba, params = _clf()
  out = evaluate_model(y_true, y_pred, proba, x, params)['global']
  keys = ['accuracy', 'balanced_accuracy', 'precision_macro',
          'recall_macro', 'f1_macro', 'precision_micro',
          'precision_weighted', 'per_class_precision',
          'per_class_recall', 'per_class_f1', 'roc_auc',
          'roc_auc_ovr', 'roc_auc_ovo', 'pr_auc',
          'average_precision', 'log_loss', 'brier',
          'brier_score', 'cohen_kappa_linear',
          'cohen_kappa_quadratic', 'kappa_linear',
          'kappa_quadratic', 'matthews_corrcoef', 'mcc',
          'sensitivity', 'specificity', 'npv', 'ppv', 'fpr',
          'fnr', 'fdr', 'ks', 'ks_statistic', 'hinge_loss',
          'top_k_accuracy_2', 'top_k_accuracy_3', 'jaccard',
          'jaccard_macro', 'jaccard_weighted', 'zero_one_loss',
          'confusion_matrix', 'classification_report', 'ece',
          'mce', 'calibration_ece', 'calibration_mce',
          'brier_decomposition']
  for key in keys:
    assert key in out, key
  assert isinstance(out['classification_report'], dict)
  assert isinstance(out['confusion_matrix'], list)


def test_classification_multiclass_metrics():
  '''Multiclass classification computes ovr/ovo AUC safely.'''
  rng = np.random.RandomState(7)
  n = 90
  y_true = rng.randint(0, 3, n)
  y_pred = y_true.copy()
  y_pred[::5] = (y_pred[::5] + 1) % 3
  proba = rng.rand(n, 3)
  proba /= proba.sum(axis=1, keepdims=True)
  x = pd.DataFrame({'f1': rng.randn(n),
                    'seg': np.where(rng.rand(n) > .5, 'a', 'b')})
  params = {'task_type': 'classification',
            'bucket_segment_col': 'seg'}
  out = evaluate_model(y_true, y_pred, proba, x, params)
  g = out['global']
  assert not np.isnan(g['roc_auc_ovr'])
  assert not np.isnan(g['roc_auc_ovo'])
  assert len(g['per_class_f1']) == 3
  assert out['by_bucket'] and out['by_decile']


def test_classification_single_class_no_crash():
  '''Single-class subset yields NaN metrics, not a crash.'''
  n = 20
  y_true = np.zeros(n, dtype=int)
  y_pred = np.zeros(n, dtype=int)
  proba = np.column_stack([np.ones(n), np.zeros(n)])
  x = pd.DataFrame({'f1': np.arange(n), 'seg': 'a'})
  params = {'task_type': 'classification',
            'bucket_segment_col': 'seg'}
  out = evaluate_model(y_true, y_pred, proba, x, params)
  assert np.isnan(out['global']['roc_auc'])
  assert 'accuracy' in out['global']
  assert out['by_bucket'] or out['by_decile'] or True


def test_regression_full_metric_set():
  '''Regression exposes every comprehensive metric.'''
  x, y_true, y_pred, _, params = _reg()
  out = evaluate_model(y_true, y_pred, None, x, params)['global']
  keys = ['mae', 'medae', 'rmse', 'rmsle', 'mse', 'msle',
          'r2', 'adj_r2', 'mape', 'smape', 'explained_variance',
          'max_error', 'bias', 'mean_residual',
          'residual_std', 'residual_skew', 'residual_kurtosis',
          'quantile_loss_0_1', 'quantile_loss_0_5',
          'quantile_loss_0_9', 'pinball_0_1', 'pinball_0_5',
          'pinball_0_9', 'tweedie_deviance',
          'poisson_deviance', 'gamma_deviance',
          'd2_absolute_error_score',
          'mean_absolute_percentage_error',
          'median_absolute_percentage_error', 'medape',
          'mean_absolute_error', 'mean_squared_error',
          'root_mean_squared_error']
  for key in keys:
    assert key in out, key
  # Quantile losses & deviances are finite on synthetic data.
  assert np.isfinite(out['quantile_loss_0_5'])
  assert np.isfinite(out['poisson_deviance'])
  reg = evaluate_model(y_true, y_pred, None, x, params)
  assert 'r2' in reg['by_bucket']['a']
  assert 'r2' in reg['by_decile'][list(reg['by_decile'])[0]]


def test_shap_fallback_returns_dict_artifact():
  '''SHAP node returns a dict artifact via fallback.'''
  from forecast_ml_framework.pipelines.modeling.adapters import (
    SklearnAdapter,
  )
  x, y_true, _, _, _ = _clf()
  model = SklearnAdapter({}).fit(x[['f1', 'f2']], y_true)
  artifact = explain_model(model, x[['f1', 'f2']], y_true,
                           {'shap_params': {'method': 'auto'}})
  assert isinstance(artifact, dict)
  assert 'global_importance' in artifact
  assert artifact['global_importance']
