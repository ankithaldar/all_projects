#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Charts node: matplotlib figures for both task types.'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _pd_abs_err(y_true, y_pred):
  '''Return absolute errors as a pandas Series.

  Args:
    y_true: True labels or values.
    y_pred: Predicted labels or values.

  Returns:
    Pandas Series of absolute errors aligned by position.
  '''
  import pandas as pd
  return pd.Series(np.abs(np.asarray(y_true, dtype=float)
                           - np.asarray(y_pred, dtype=float)))


def _new_fig(title):
  '''Create a figure with a title.'''
  fig, ax = plt.subplots()
  ax.set_title(title)
  return fig, ax


def _safe(figs, name, fn, *args):
  '''Add a figure if fn succeeds, else skip gracefully.

  Args:
    figs: Dict of chart name to figure, updated in place.
    name: Chart name to store the figure under.
    fn: Callable returning a matplotlib Figure.
    *args: Arguments forwarded to fn.

  Returns:
    None. Figures created by a failing fn are closed so
    no figure leaks.
  '''
  before = set(plt.get_fignums())
  try:
    fig = fn(*args)
    if fig is not None:
      figs[name] = fig
  except Exception:  # noqa: BLE001 - charts skip on bad data
    for num in set(plt.get_fignums()) - before:
      plt.close(num)


def _roc(y_true, proba):
  '''Plot the ROC curve.'''
  from sklearn.metrics import roc_curve, roc_auc_score
  fpr, tpr, _ = roc_curve(y_true, proba[:, 1])
  fig, ax = _new_fig('ROC curve')
  ax.plot(fpr, tpr, label=f'auc={roc_auc_score(y_true, proba[:, 1]):.3f}')
  ax.plot([0, 1], [0, 1], '--')
  ax.legend()
  return fig


def _pr(y_true, proba):
  '''Plot the precision-recall curve.'''
  from sklearn.metrics import precision_recall_curve
  prec, rec, _ = precision_recall_curve(y_true, proba[:, 1])
  fig, ax = _new_fig('PR curve')
  ax.plot(rec, prec)
  ax.set_xlabel('recall')
  ax.set_ylabel('precision')
  return fig


def _confusion(y_true, y_pred):
  '''Plot the confusion matrix heatmap.'''
  from sklearn.metrics import confusion_matrix
  cm = confusion_matrix(y_true, y_pred)
  fig, ax = _new_fig('Confusion matrix')
  ax.imshow(cm, cmap='Blues')
  for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
      ax.text(j, i, str(cm[i, j]), ha='center', va='center')
  return fig


def _calibration(y_true, proba):
  '''Plot the calibration curve.'''
  from sklearn.calibration import calibration_curve
  prob_true, prob_pred = calibration_curve(
      y_true, proba[:, 1], n_bins=5)
  fig, ax = _new_fig('Calibration curve')
  ax.plot(prob_pred, prob_true, marker='o')
  ax.plot([0, 1], [0, 1], '--')
  return fig


def _threshold(y_true, proba):
  '''Plot precision/recall vs threshold.'''
  from sklearn.metrics import precision_recall_curve
  prec, rec, thr = precision_recall_curve(y_true, proba[:, 1])
  fig, ax = _new_fig('Threshold curve')
  ax.plot(thr, prec[:-1], label='precision')
  ax.plot(thr, rec[:-1], label='recall')
  ax.legend()
  return fig


def _lift(y_true, proba):
  '''Plot lift and gain curves.'''
  from sklearn.metrics import roc_curve
  order = np.argsort(-proba[:, 1])
  y_sorted = np.asarray(y_true)[order]
  gain = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
  frac = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
  fig, ax = _new_fig('Lift / gain')
  ax.plot(frac, gain, label='gain')
  ax.plot(frac, gain / np.maximum(frac, 1e-9), label='lift')
  ax.legend()
  return fig


def _ks_plot(y_true, proba):
  '''Plot cumulative score distributions by class.'''
  pos = np.sort(proba[:, 1][np.asarray(y_true) == 1])
  neg = np.sort(proba[:, 1][np.asarray(y_true) == 0])
  fig, ax = _new_fig('KS plot')
  ax.plot(pos, np.arange(1, len(pos) + 1) / max(len(pos), 1),
          label='pos')
  ax.plot(neg, np.arange(1, len(neg) + 1) / max(len(neg), 1),
          label='neg')
  ax.legend()
  return fig


def _prob_dist(y_true, proba):
  '''Plot probability distributions by class.'''
  fig, ax = _new_fig('Probability distribution')
  for label in (0, 1):
    ax.hist(proba[:, 1][np.asarray(y_true) == label],
            bins=10, alpha=0.5, label=str(label))
  ax.legend()
  return fig


def _per_class_bars(y_true, y_pred):
  '''Plot per-class F1 bars.'''
  from sklearn.metrics import f1_score
  vals = f1_score(y_true, y_pred, average=None, zero_division=0)
  fig, ax = _new_fig('Per-class F1')
  ax.bar(range(len(vals)), vals)
  ax.set_xlabel('class')
  return fig


def _cost_curve(_y_true, proba):  # pylint: disable=unused-argument
  '''Plot a cost-curve stub.'''
  fig, ax = _new_fig('Cost curve (stub)')
  ax.plot([0, 1], [proba.mean(), 1 - proba.mean()])
  ax.set_xlabel('fraction of population')
  return fig


def _pred_vs_actual(y_true, y_pred):
  '''Plot predicted vs actual.'''
  fig, ax = _new_fig('Predicted vs actual')
  ax.scatter(y_true, y_pred, s=8)
  ax.plot([y_true.min(), y_true.max()],
          [y_true.min(), y_true.max()], '--')
  return fig


def _residuals_fitted(y_pred, resid):
  '''Plot residuals vs fitted.'''
  fig, ax = _new_fig('Residuals vs fitted')
  ax.scatter(y_pred, resid, s=8)
  ax.axhline(0, color='r')
  return fig


def _residuals_actual(y_true, resid):
  '''Plot residuals vs actual plus residual histogram.'''
  fig, ax = _new_fig('Residuals vs actual')
  ax.scatter(y_true, resid, s=8)
  ax.axhline(0, color='r')
  return fig


def _residual_hist_qq(y_true, y_pred):
  '''Plot residual histogram and QQ plot side by side.'''
  from scipy.stats import probplot
  resid = np.asarray(y_true, dtype=float) - np.asarray(
      y_pred, dtype=float)
  fig, axes = plt.subplots(1, 2)
  axes[0].hist(resid, bins=min(10, max(2, len(resid) // 2)))
  axes[0].set_title('Residual histogram')
  probplot(resid, plot=axes[1])
  axes[1].set_title('QQ plot')
  fig.suptitle('Residuals hist + QQ')
  return fig


def _scale_location(y_true, y_pred):
  '''Plot sqrt(|resid|) vs fitted (scale-location).'''
  resid = np.asarray(y_true, dtype=float) - np.asarray(
      y_pred, dtype=float)
  fig, ax = _new_fig('Scale-location')
  ax.scatter(y_pred, np.sqrt(np.abs(resid)), s=8)
  return fig


def _error_actual(y_true, y_pred):
  '''Plot absolute error vs actual.'''
  err = np.abs(np.asarray(y_true, dtype=float) - np.asarray(
      y_pred, dtype=float))
  fig, ax = _new_fig('Error vs actual')
  ax.scatter(y_true, err, s=8)
  return fig


def _error_bucket(y_true, y_pred, x, params):
  '''Plot mean absolute error by bucket segment.'''
  col = params.get('bucket_segment_col')
  if not col or not hasattr(x, 'columns') or col not in x.columns:
    raise ValueError('no bucket column')
  err = _pd_abs_err(y_true, y_pred)
  means = err.groupby(x[col].values).mean()
  fig, ax = _new_fig('Error by bucket')
  ax.bar([str(k) for k in means.index], means.values)
  return fig


def _segment_curves(y_true, y_pred, x, params):
  '''Plot predicted vs actual per segment.'''
  col = params.get('bucket_segment_col')
  if not col or not hasattr(x, 'columns') or col not in x.columns:
    raise ValueError('no bucket column')
  fig, ax = _new_fig('Segment curves')
  for label in list(dict.fromkeys(x[col].astype(str))):
    mask = (x[col].astype(str) == label).values
    ax.scatter(np.asarray(y_true)[mask],
               np.asarray(y_pred)[mask], s=8, label=str(label))
  ax.legend()
  return fig


def _feature_residual(y_true, y_pred, x):
  '''Plot residuals vs the first numeric feature.'''
  resid = np.asarray(y_true, dtype=float) - np.asarray(
      y_pred, dtype=float)
  numeric = x.select_dtypes(include='number') if hasattr(
      x, 'select_dtypes') else None
  if numeric is None or numeric.shape[1] == 0:
    raise ValueError('no numeric feature')
  fig, ax = _new_fig('Feature vs residual')
  ax.scatter(numeric.iloc[:, 0], resid, s=8)
  ax.set_xlabel(str(numeric.columns[0]))
  return fig


def make_charts(y_true, y_pred, y_proba, x_test, params):
  '''Build all charts for the task type.

  Args:
    y_true: True labels or values.
    y_pred: Predicted labels or values.
    y_proba: Probabilities or None.
    x_test: Test features.
    params: modeling_params dict.

  Returns:
    Dict of chart name to matplotlib Figure. Charts whose
    data are insufficient are skipped gracefully. Every
    figure is closed before returning, so no figure leaks;
    callers that need the image must save it inside this
    call (figures are returned closed, not open).
  '''
  params = params or {}
  task = params.get('task_type', 'classification')
  figs = {}
  if task == 'classification':
    if y_proba is not None:
      proba = np.asarray(y_proba)
      _safe(figs, 'roc', _roc, y_true, proba)
      _safe(figs, 'pr', _pr, y_true, proba)
      _safe(figs, 'calibration_curve', _calibration, y_true, proba)
      _safe(figs, 'threshold_curve', _threshold, y_true, proba)
      _safe(figs, 'lift_gain', _lift, y_true, proba)
      _safe(figs, 'ks_plot', _ks_plot, y_true, proba)
      _safe(figs, 'prob_distribution', _prob_dist, y_true, proba)
      _safe(figs, 'cost_curve', _cost_curve, y_true, proba)
    _safe(figs, 'confusion_heatmap', _confusion, y_true, y_pred)
    _safe(figs, 'per_class_bars', _per_class_bars, y_true, y_pred)
  else:
    resid = np.asarray(y_true, dtype=float) - np.asarray(
        y_pred, dtype=float)
    _safe(figs, 'pred_vs_actual', _pred_vs_actual, y_true, y_pred)
    _safe(figs, 'residuals_vs_fitted', _residuals_fitted,
          y_pred, resid)
    _safe(figs, 'residuals_vs_actual', _residuals_actual,
          y_true, resid)
    _safe(figs, 'residual_hist_qq', _residual_hist_qq,
          y_true, y_pred)
    _safe(figs, 'scale_location', _scale_location, y_true, y_pred)
    _safe(figs, 'error_vs_actual', _error_actual, y_true, y_pred)
    _safe(figs, 'error_by_bucket', _error_bucket,
          y_true, y_pred, x_test, params)
    _safe(figs, 'segment_curves', _segment_curves,
          y_true, y_pred, x_test, params)
    _safe(figs, 'feature_vs_residual', _feature_residual,
          y_true, y_pred, x_test)
  try:
    return figs
  finally:
    for fig in list(figs.values()):
      plt.close(fig)
