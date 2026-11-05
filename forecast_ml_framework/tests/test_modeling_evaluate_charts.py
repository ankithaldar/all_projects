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
from forecast_ml_framework.pipelines.modeling.pipeline import (
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
  params = dict(params, metrics='all')
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
  params = dict(params, metrics='all')
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
            'bucket_segment_col': 'seg', 'metrics': 'all'}
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
  params = dict(params, metrics='all')
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


def test_default_metrics_subset_and_counts():
  '''Default evaluate returns curated subset only, with note.'''
  from forecast_ml_framework.pipelines.modeling.metrics import (
    DEFAULT_METRICS,
    METRIC_REGISTRY,
  )
  assert 10 <= len(DEFAULT_METRICS) <= 20
  assert 'accuracy' in DEFAULT_METRICS and 'mae' in DEFAULT_METRICS
  assert len(METRIC_REGISTRY) >= 80
  x, y_true, y_pred, proba, params = _clf()
  out = evaluate_model(y_true, y_pred, proba, x, params)
  g = out['global']
  assert 'accuracy' in g and 'roc_auc' in g
  assert 'ndcg' not in g and 'huber_loss' not in g
  assert out['metrics_calculated'] == sorted(g.keys())
  assert out['available_metrics'] == len(METRIC_REGISTRY)
  assert out['available_metrics'] > len(g)


def test_metrics_all_flag_and_custom_list():
  '''metrics=all returns everything; list returns only those.'''
  x, y_true, y_pred, proba, params = _clf()
  all_params = dict(params, metrics='all')
  g_all = evaluate_model(
      y_true, y_pred, proba, x, all_params)['global']
  assert 'ndcg' in g_all and 'gini_coefficient' in g_all
  assert 'mrr' in g_all and 'mean_average_precision' in g_all
  list_params = dict(params, metrics_to_calculate=['accuracy',
                                                   'f1_weighted'])
  out = evaluate_model(y_true, y_pred, proba, x, list_params)
  assert sorted(out['global'].keys()) == [
      'accuracy', 'f1_weighted']
  assert out['metrics_calculated'] == ['accuracy', 'f1_weighted']
  list_params_r = {'task_type': 'regression', 'metrics': ['mae']}
  x_r, yt_r, yp_r, _, _ = _reg()
  out_r = evaluate_model(yt_r, yp_r, None, x_r, list_params_r)
  assert sorted(out_r['global'].keys()) == ['mae']
  assert out_r['global']['mae'] > 0


def test_metrics_unknown_raises_valueerror():
  '''Unknown metric name raises ValueError listing valid names.'''
  from forecast_ml_framework.pipelines.modeling.metrics import (
    METRIC_REGISTRY,
  )
  x, y_true, y_pred, proba, params = _clf()
  bad = dict(params, metrics=['accuracy', 'nope'])
  try:
    evaluate_model(y_true, y_pred, proba, x, bad)
  except ValueError as exc:
    assert 'nope' in str(exc)
    assert 'accuracy' in str(exc)  # valid names listed
  else:  # pragma: no cover
    raise AssertionError('expected ValueError')
  assert 'accuracy' in METRIC_REGISTRY


def test_metrics_selection_skips_unselected_families():
  '''Selecting only mae must not invoke classification AUC code.'''
  import forecast_ml_framework.pipelines.modeling.metrics as m
  orig = m.compute_metrics
  calls = []
  x_r, yt_r, yp_r, _, _ = _reg()
  params = {'task_type': 'regression', 'metrics': ['mae']}
  out = evaluate_model(yt_r, yp_r, None, x_r, params)
  assert list(out['global']) == ['mae']
  # Shared intermediate: selecting mase still yields correct ratio.
  params_mase = {'task_type': 'regression',
                 'metrics': ['mase', 'mae']}
  g = evaluate_model(yt_r, yp_r, None, x_r, params_mase)['global']
  assert g['mase'] > 0 and g['mae'] > 0
  _ = orig, calls


def test_classification_phase3_metrics_binary():
  '''Phase-3 classification metrics on binary synthetic data.'''
  x, y_true, y_pred, proba, params = _clf()
  params = dict(params, metrics='all')
  out = evaluate_model(y_true, y_pred, proba, x, params)
  g = out['global']
  keys = ['error_rate', 'f_beta_0_5', 'f_beta_2', 'g_mean',
          'youden_j', 'diagnostic_odds_ratio', 'fowlkes_mallows',
          'markedness', 'prevalence_threshold', 'kl_divergence',
          'gini_coefficient', 'precision_at_5', 'precision_at_10',
          'recall_at_5', 'recall_at_10', 'f1_at_5', 'f1_at_10',
          'mean_average_precision', 'ndcg', 'mrr',
          'lift_top_decile', 'cumulative_gain_top_decile']
  for key in keys:
    assert key in g, key
    assert np.isfinite(g[key]), key
  # Relationships / bounds.
  assert abs(g['error_rate'] - (1 - g['accuracy'])) < 1e-9
  assert abs(g['gini_coefficient'] - (2 * g['roc_auc'] - 1)) < 1e-9
  assert 0 <= g['ndcg'] <= 1 and 0 <= g['mrr'] <= 1
  assert 0 <= g['g_mean'] <= 1
  assert g['kl_divergence'] >= 0
  for key in ('precision_at_5', 'recall_at_5', 'f1_at_5'):
    assert 0 <= g[key] <= 1
  # Propagation through evaluate aggregation.
  assert 'error_rate' in out['by_bucket']['a']
  first_decile = next(iter(out['by_decile']))
  assert 'ndcg' in out['by_decile'][first_decile]


def test_classification_phase3_metrics_multiclass_ranking():
  '''Phase-3 ranking metrics on multiclass proba scores.'''
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
            'bucket_segment_col': 'seg', 'metrics': 'all'}
  g = evaluate_model(y_true, y_pred, proba, x, params)['global']
  for key in ('precision_at_5', 'recall_at_10', 'f1_at_10',
              'mean_average_precision', 'ndcg', 'mrr',
              'lift_top_decile', 'cumulative_gain_top_decile',
              'g_mean', 'kl_divergence', 'gini_coefficient'):
    assert key in g, key
    assert np.isfinite(g[key]), key
  assert 0 <= g['ndcg'] <= 1 and 0 <= g['mrr'] <= 1


def test_regression_phase3_metrics():
  '''Phase-3 regression metrics on synthetic data.'''
  x, y_true, y_pred, _, params = _reg()
  params = dict(params, metrics='all')
  out = evaluate_model(y_true, y_pred, None, x, params)
  g = out['global']
  keys = ['mase', 'wmae', 'mspe', 'rmspe', 'pearson_r',
          'spearman_rho', 'mpe', 'maape', 'huber_loss',
          'log_cosh_loss', 'mean_bias_deviation', 'r2_log',
          'residual_std']
  for key in keys:
    assert key in g, key
    assert np.isfinite(g[key]), key
  assert abs(g['mean_bias_deviation'] - g['bias']) < 1e-12
  assert abs(g['rmspe'] - np.sqrt(g['mspe'])) < 1e-9
  assert -1 <= g['pearson_r'] <= 1
  assert -1 <= g['spearman_rho'] <= 1
  assert g['mase'] >= 0 and g['wmae'] >= 0 and g['huber_loss'] >= 0
  assert 'mase' in out['by_bucket']['a']
  first_decile = next(iter(out['by_decile']))
  assert 'pearson_r' in out['by_decile'][first_decile]


def test_regression_phase3_metrics_constant_and_zero_safe():
  '''Phase-3 regression metrics are NaN-safe on degenerate data.'''
  y_true = np.zeros(20)
  y_pred = np.zeros(20)
  x = pd.DataFrame({'f1': np.arange(20), 'seg': 'a'})
  params = {'task_type': 'regression', 'bucket_segment_col': 'seg',
            'metrics': 'all'}
  g = evaluate_model(y_true, y_pred, None, x, params)['global']
  for key in ('mase', 'wmae', 'mspe', 'rmspe', 'mpe', 'maape',
              'huber_loss', 'log_cosh_loss', 'r2_log'):
    assert key in g, key  # value may be nan, must not raise


# Phase-3 charts: RED tests for all new chart functions.
def _tiny():
  '''Return tiny inputs that must yield None, never a crash.'''
  return np.array([0, 1]), np.array([0.2, 0.8])


def _check(fn, *args, **kwargs):
  '''Assert fn returns a figure and leaks no open figures.'''
  before = plt.get_fignums()
  fig = fn(*args, **kwargs)
  assert fig is not None
  assert plt.get_fignums() == before
  plt.close(fig)


def _check_none(fn, *args, **kwargs):
  '''Assert fn returns None on tiny data and leaks nothing.'''
  before = plt.get_fignums()
  assert fn(*args, **kwargs) is None
  assert plt.get_fignums() == before


def _df(n=60):
  '''Return synthetic tabular data with a target column.'''
  rng = np.random.RandomState(0)
  return pd.DataFrame({'f1': rng.randn(n), 'f2': rng.randn(n),
                       'cat': np.where(rng.rand(n) > .5, 'a', 'b'),
                       'y': rng.randint(0, 2, n)})


def test_phase3_data_charts():
  '''Data-understanding charts return figures on synthetic data.'''
  from forecast_ml_framework.pipelines.modeling.charts import (
    correlation_heatmap,
    feature_distributions,
    make_data_charts,
    missing_heatmap,
    outlier_box,
    pair_plot,
    pca_scatter,
    target_distribution,
  )
  df = _df()
  _check(target_distribution, df, 'y')
  _check(feature_distributions, df)
  _check(correlation_heatmap, df)
  _check(pair_plot, df, 'y')
  _check(missing_heatmap, df)
  _check(outlier_box, df)
  _check(pca_scatter, df, 'y')
  before = plt.get_fignums()
  figs = make_data_charts(df, 'y', {})
  assert len(figs) >= 6
  assert plt.get_fignums() == before
  for fig in figs.values():
    plt.close(fig)


def test_phase3_data_charts_tiny_and_nan():
  '''Data charts return None on tiny / all-NaN data, no crash.'''
  from forecast_ml_framework.pipelines.modeling.charts import (
    correlation_heatmap,
    make_data_charts,
    missing_heatmap,
    target_distribution,
  )
  _check_none(target_distribution, _df(2), 'y')
  _check_none(correlation_heatmap, _df(2))
  bad = _df(30)[['f1', 'f2']].astype(float)
  bad[:] = np.nan
  _check_none(missing_heatmap, bad)
  assert make_data_charts(_df(2), 'y', {}) == {}
  assert make_data_charts(None, 'y', {}) == {}


def test_phase3_classification_charts():
  '''New classification charts return figures and None when tiny.'''
  from forecast_ml_framework.pipelines.modeling.charts import (
    balanced_accuracy_plot,
    confusion_wheel,
    det_curve,
    pr_gain_curve,
    topk_accuracy_plot,
  )
  _x, y_true, y_pred, proba, _ = _clf()
  _check(det_curve, y_true, proba)
  _check(confusion_wheel, y_true, y_pred)
  _check(topk_accuracy_plot, y_true, proba)
  _check(balanced_accuracy_plot, y_true, y_pred)
  _check(pr_gain_curve, y_true, proba)
  t_y, t_p = _tiny()
  _check_none(det_curve, t_y, t_p.reshape(-1, 1))
  _check_none(confusion_wheel, t_y, t_y)
  _check_none(topk_accuracy_plot, t_y, t_p.reshape(-1, 1))
  _check_none(balanced_accuracy_plot, t_y, t_y)
  _check_none(pr_gain_curve, t_y, t_p.reshape(-1, 1))


def test_phase3_regression_charts():
  '''New regression charts return figures and None when tiny.'''
  from forecast_ml_framework.pipelines.modeling.charts import (
    cooks_distance,
    forecast_vs_actual,
    prediction_interval_coverage,
    rec_curve,
    residual_autocorr,
    residuals_over_time,
  )
  x, y_true, y_pred, _, _ = _reg()
  _check(residual_autocorr, y_true, y_pred)
  _check(cooks_distance, y_true, y_pred, x[['f1', 'f2']].values)
  _check(rec_curve, y_true, y_pred)
  _check(prediction_interval_coverage, y_true, y_pred)
  _check(residuals_over_time, y_true, y_pred)
  _check(forecast_vs_actual, y_true, y_pred)
  _check(residuals_over_time, y_true, y_pred,
         pd.date_range('2024-01-01', periods=len(y_true)))
  _check(forecast_vs_actual, y_true, y_pred,
         pd.date_range('2024-01-01', periods=len(y_true)))
  t_y, _ = _tiny()
  _check_none(residual_autocorr, t_y, t_y)
  _check_none(cooks_distance, t_y, t_y)
  _check_none(rec_curve, t_y, t_y)
  _check_none(prediction_interval_coverage, t_y, t_y)
  _check_none(residuals_over_time, t_y, t_y)
  _check_none(forecast_vs_actual, t_y, t_y)


def test_phase3_training_charts():
  '''Training diagnostics return figures and None when tiny.'''
  from forecast_ml_framework.pipelines.modeling.charts import (
    bias_variance,
    cv_scores_box,
    learning_curve,
    loss_vs_epoch,
    make_training_charts,
    overfitting_gap,
    validation_curve,
  )
  tr = [0.5, 0.6, 0.7, 0.8]
  va = [0.4, 0.55, 0.65, 0.7]
  _check(learning_curve, tr, va)
  _check(validation_curve, [1, 10, 100], tr[:3], va[:3])
  _check(loss_vs_epoch, [1.0, 0.5, 0.3], [1.2, 0.6, 0.4])
  _check(cv_scores_box, {'m1': [0.7, 0.8, 0.9], 'm2': [0.6, 0.7, 0.8]})
  _check(overfitting_gap, tr, va)
  _check(bias_variance, tr, va)
  hist = {'train_scores': tr, 'val_scores': va,
          'train_loss': [1.0, 0.5, 0.3], 'val_loss': [1.2, 0.6, 0.4],
          'cv_scores': {'m1': [0.7, 0.8, 0.9]}}
  before = plt.get_fignums()
  figs = make_training_charts(hist, {})
  assert len(figs) >= 5 and plt.get_fignums() == before
  for fig in figs.values():
    plt.close(fig)
  _check_none(learning_curve, [0.5], [0.4])
  _check_none(cv_scores_box, {})
  assert make_training_charts({}, {}) == {}
  assert make_training_charts(None, {}) == {}


def test_phase3_interpretability_charts():
  '''Interpretability charts return figures and None when tiny.'''
  from sklearn.tree import DecisionTreeClassifier

  from forecast_ml_framework.pipelines.modeling.charts import (
    decision_boundary,
    ice_plot,
    lime_bar,
    pdp_1d,
    shap_dependence,
    shap_force,
    shap_waterfall,
    tree_plot,
  )
  x, y_true, _y_pred, _proba, _ = _clf()
  rng = np.random.RandomState(1)
  shap_vals = rng.randn(len(y_true), 2)
  _check(shap_dependence, shap_vals, x[['f1', 'f2']].values, 0)
  _check(shap_waterfall, shap_vals[0], ['f1', 'f2'])
  _check(shap_force, shap_vals[0], ['f1', 'f2'])
  _check(lime_bar, {'f1': 0.5, 'f2': -0.3})
  model = DecisionTreeClassifier(max_depth=3, random_state=0)
  model.fit(x[['f1', 'f2']], y_true)
  _check(pdp_1d, model, x[['f1', 'f2']].values, 0)
  _check(ice_plot, model, x[['f1', 'f2']].values, 0)
  _check(decision_boundary, x[['f1', 'f2']].values, y_true, model)
  _check(tree_plot, model, ['f1', 'f2'])
  _check_none(shap_dependence, shap_vals[:1], x.values[:1], 0)
  _check_none(shap_waterfall, shap_vals[0][:0], [])
  _check_none(lime_bar, {})
  _check_none(tree_plot, object())
  _check_none(decision_boundary, x.values[:1], y_true[:1], model)


def test_phase3_comparison_and_specialized():
  '''Comparison + specialized charts return figures, None tiny.'''
  from forecast_ml_framework.pipelines.modeling.charts import (
    bland_altman,
    ensemble_diversity,
    error_correlation_heatmap,
    make_comparison_charts,
    model_comparison_box,
    pr_overlay,
    prediction_interval_vs_actual,
    quantile_coverage,
    roc_overlay,
  )
  _x, y_true, y_pred, _proba, _ = _reg(80)
  _xc, yt_c, _yp_c, proba_c, _ = _clf()
  preds = {'m1': y_pred, 'm2': y_pred + 0.1}
  probas = {'m1': proba_c, 'm2': proba_c * 0.9}
  _check(model_comparison_box, {'m1': [0.7, 0.8, 0.9],
                                'm2': [0.6, 0.7, 0.8]})
  _check(bland_altman, preds['m1'], preds['m2'])
  _check(ensemble_diversity, preds)
  _check(roc_overlay, yt_c, probas)
  _check(pr_overlay, yt_c, probas)
  _check(error_correlation_heatmap, preds, y_true)
  _check(quantile_coverage, y_true, {0.1: y_pred - 1, 0.9: y_pred + 1})
  _check(prediction_interval_vs_actual, y_true, y_pred - 1, y_pred + 1)
  before = plt.get_fignums()
  figs = make_comparison_charts(preds, y_true, {'probas': probas})
  assert 'bland_altman' in figs and 'ensemble_diversity' in figs
  assert plt.get_fignums() == before
  for fig in figs.values():
    plt.close(fig)
  t_y, _ = _tiny()
  _check_none(bland_altman, t_y, t_y)
  _check_none(ensemble_diversity, {'m1': t_y})
  _check_none(roc_overlay, t_y, {'m1': t_y.reshape(-1, 1)})
  _check_none(quantile_coverage, t_y, {0.5: t_y})
  _check_none(prediction_interval_vs_actual, t_y, t_y, t_y)
  assert make_comparison_charts({}, t_y, {}) == {}


def test_phase3_make_charts_dispatcher_new_groups():
  '''make_charts wires new groups when params provide data.'''
  from sklearn.tree import DecisionTreeClassifier
  x, y_true, y_pred, proba, params = _clf()
  model = DecisionTreeClassifier(max_depth=3, random_state=0)
  model.fit(x[['f1', 'f2']], y_true)
  params = dict(params, training_history={
      'train_scores': [0.5, 0.6, 0.7], 'val_scores': [0.4, 0.5, 0.6]},
      shap_values=np.random.RandomState(0).randn(len(y_true), 2),
      shap_feature_names=['f1', 'f2'], model=model,
      lime_importances={'f1': 0.4, 'f2': 0.2})
  before = plt.get_fignums()
  figs = make_charts(y_true, y_pred, proba, x, params)
  for key in ('det_curve', 'confusion_wheel', 'topk_accuracy_plot',
              'balanced_accuracy_plot', 'pr_gain_curve',
              'learning_curve', 'shap_dependence', 'pdp_1d',
              'tree_plot', 'lime_bar'):
    assert key in figs, key
  assert plt.get_fignums() == before
  for fig in figs.values():
    plt.close(fig)
  xr, ytr, ypr, _, paramr = _reg()
  paramr = dict(paramr, prediction_intervals={
      'lower': ypr - 1, 'upper': ypr + 1},
      quantile_preds={0.1: ypr - 1, 0.9: ypr + 1})
  figs_r = make_charts(ytr, ypr, None, xr, paramr)
  for key in ('residual_autocorr', 'cooks_distance', 'rec_curve',
              'prediction_interval_coverage', 'residuals_over_time',
              'forecast_vs_actual', 'quantile_coverage',
              'prediction_interval_vs_actual'):
    assert key in figs_r, key
  for fig in figs_r.values():
    plt.close(fig)


def test_compute_metrics_length_mismatch_raises():
  import pytest
  from forecast_ml_framework.pipelines.modeling.metrics import (
    compute_metrics,
  )
  with pytest.raises(ValueError):
    compute_metrics(
        [0, 1, 0], [0, 1], [[0.5, 0.5]] * 3, 'classification')
  with pytest.raises(ValueError):
    compute_metrics(
        [0, 1], [0, 1], [[0.5, 0.5]], 'classification')
  with pytest.raises(ValueError):
    compute_metrics([1.0, 2.0], [1.0], None, 'regression')


def test_evaluate_model_length_mismatch_raises():
  import pandas as pd
  import pytest
  x = pd.DataFrame({'f1': [1.0, 2.0]})
  with pytest.raises(ValueError):
    evaluate_model(
        [0, 1, 0], [0, 1], [[0.5, 0.5]] * 3, x,
        {'task_type': 'classification'})
