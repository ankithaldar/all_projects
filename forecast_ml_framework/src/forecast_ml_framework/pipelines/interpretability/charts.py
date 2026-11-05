#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Interpretability and diagnostics charts (Phase 3).

Matplotlib figures for the structured artifacts produced
by ``pipelines.interpretability.nodes``. Every public
function returns a Figure or None on insufficient data,
never raises, and closes its figure in a finally block
(via :func:`_safe_fig`) so no figure slot leaks.

Where ``pipelines.modeling.charts`` already provides an
equivalent function (SHAP dependence/waterfall/force,
LIME bar, decision boundary, tree, Cook, REC, residuals
over time, learning/validation/loss/CV/overfitting), it
is reused defensively: the import happens inside the
function and any failure falls back to the local
implementation below. ``modeling/charts.py`` is never
modified from here.

Interpretability: SHAP beeswarm, SHAP bar, SHAP
heatmap, SHAP dependence, SHAP waterfall, SHAP force,
PDP 1D (+ICE overlay), PDP 2D heatmap, permutation
importance bar, LIME bar, decision boundary contour,
tree plot.
Diagnostics: calibration curve, residual vs fitted,
residual histogram+QQ, scale-location, error by
segment bar, prediction distribution by class,
confidence histogram, residuals over time, Cook's
distance, REC curve, learning curve, validation
curve, loss vs epoch, CV box, overfitting gap.
'''

import logging
from statistics import NormalDist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LOG = logging.getLogger(__name__)


def _safe_fig(fn):
  '''Wrap a chart fn: never raise, always close figure.

  Returns the (closed) Figure or None; only figures
  created by a failing call are closed on exception.
  '''
  def wrapper(*args, **kwargs):
    '''Call the wrapped chart function safely.

    Args:
      *args: Positional arguments forwarded to fn.
      **kwargs: Keyword arguments forwarded to fn.

    Returns:
      The (closed) Figure from fn, or None on error.
    '''
    before = set(plt.get_fignums())
    fig = None
    try:
      fig = fn(*args, **kwargs)
      return fig
    except Exception:  # noqa: BLE001 - charts skip bad data
      for num in set(plt.get_fignums()) - before:
        plt.close(num)
      return None
    finally:
      if fig is not None:
        plt.close(fig)
  return wrapper


def _new_fig(title):
  '''Create a figure with a title.'''
  fig, ax = plt.subplots()
  ax.set_title(title)
  return fig, ax


def _delegate(name, *args, **kwargs):
  '''Try a modeling.charts function of the same name.

  Returns (figure, True) when the modeling function
  exists (figure may be None = insufficient data),
  else (None, False) so the caller uses its local
  implementation. Never raises.
  '''
  try:
    from forecast_ml_framework.pipelines.modeling import (
      charts as modeling_charts,
    )
    func = getattr(modeling_charts, name, None)
    if not callable(func):
      return None, False
    return func(*args, **kwargs), True
  except Exception as exc:  # noqa: BLE001 - local fallback
    LOG.warning('modeling chart %r unavailable: %s' % (name, exc))
    return None, False


def _pairs(y_true, y_pred):
  '''Return finite (true, pred) arrays of size >= 5.'''
  yt = np.asarray(y_true, dtype=float).ravel()
  yp = np.asarray(y_pred, dtype=float).ravel()
  if yt.size != yp.size or yt.size < 5:
    return None
  mask = np.isfinite(yt) & np.isfinite(yp)
  if mask.sum() < 5:
    return None
  return yt[mask], yp[mask]


def _matrix(shap_values, feature_names=None):
  '''Return (2D SHAP matrix, names) from rows or array.'''
  names = list(feature_names) if feature_names else None
  if isinstance(shap_values, dict):
    rows = shap_values.get('per_row') or []
    shap_values = rows
  if isinstance(shap_values, (list, tuple)) and shap_values \
      and isinstance(shap_values[0], dict):
    rows = list(shap_values)
    if names is None:
      first = (rows[0].get('shap_values') or {})
      names = list(first)
    matrix = np.array([[float((r.get('shap_values') or {})
                                .get(n, 0.0)) for n in names]
                       for r in rows], dtype=float)
  else:
    matrix = np.asarray(shap_values, dtype=float)
    if matrix.ndim == 1:
      matrix = matrix.reshape(-1, 1)
    if names is None:
      names = ['f%d' % i for i in range(matrix.shape[1])]
  if matrix.ndim != 2 or matrix.shape[0] < 2 or not matrix.shape[1]:
    return None, names
  return matrix, names


def _importance_pairs(importance, feature_names=None):
  '''Return (labels, values) from importance artifacts.'''
  if isinstance(importance, dict) and 'importance' in importance:
    return [str(importance.get('feature', 'f0'))], \
      np.array([float(importance['importance'])])
  if isinstance(importance, (list, tuple)) and importance \
      and isinstance(importance[0], dict):
    labels = [str(e.get('feature', 'f%d' % i))
              for i, e in enumerate(importance)]
    values = np.array([float(e.get('importance',
                                   e.get('weight', 0.0)))
                       for e in importance])
    return labels, values
  if isinstance(importance, dict):
    return [str(k) for k in importance], \
      np.asarray(list(importance.values()), dtype=float)
  values = np.asarray(importance, dtype=float).ravel() \
    if importance is not None else np.array([])
  labels = list(feature_names) if feature_names else \
    ['f%d' % i for i in range(values.size)]
  return [str(v) for v in labels[:values.size]], values


def _barh(title, labels, values):
  '''Create a horizontal bar figure, or None if empty.'''
  if values is None or values.size < 1:
    return None
  order = np.argsort(-np.abs(values))
  labels = [str(labels[i]) if i < len(labels) else 'f%d' % i
            for i in order]
  fig, ax = _new_fig(title)
  ax.barh(labels, values[order])
  ax.set_xlabel('importance')
  return fig


# --- Interpretability charts ---

@_safe_fig
def shap_beeswarm(shap_values, feature_names=None):
  '''SHAP beeswarm: strip of SHAP values per feature.'''
  matrix, names = _matrix(shap_values, feature_names)
  if matrix is None:
    return None
  fig, ax = _new_fig('SHAP beeswarm')
  for j, name in enumerate(names):
    vals = matrix[:, j]
    jitter = np.random.RandomState(j).uniform(-0.2, 0.2, len(vals))
    ax.scatter(vals, j + jitter, s=10, c=vals, cmap='viridis')
  ax.set_yticks(range(len(names)))
  ax.set_yticklabels(names)
  ax.set_xlabel('SHAP value')
  return fig


@_safe_fig
def shap_bar(importance, feature_names=None):
  '''SHAP bar: sorted global importance.'''
  labels, values = _importance_pairs(importance, feature_names)
  return _barh('SHAP bar', labels, values)


@_safe_fig
def shap_heatmap(shap_values, feature_names=None):
  '''SHAP heatmap: per-row SHAP values by feature.'''
  matrix, names = _matrix(shap_values, feature_names)
  if matrix is None:
    return None
  fig, ax = _new_fig('SHAP heatmap')
  ax.imshow(matrix.T, aspect='auto', cmap='coolwarm')
  ax.set_yticks(range(len(names)))
  ax.set_yticklabels(names)
  ax.set_xlabel('row')
  return fig


@_safe_fig
def shap_dependence(shap_values, x=None, feature_index=0):
  '''SHAP dependence for one feature (delegates if able).'''
  matrix, _names = _matrix(shap_values)
  if matrix is None or not (0 <= feature_index < matrix.shape[1]):
    return None
  fig, done = _delegate('shap_dependence', matrix, x, feature_index)
  if done:
    return fig
  vals = matrix[:, feature_index]
  xs = np.arange(matrix.shape[0])
  if x is not None:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == matrix.shape[0] \
        and arr.shape[1] > feature_index:
      xs = arr[:, feature_index]
  fig, ax = _new_fig('SHAP dependence')
  ax.scatter(xs, vals, s=10, c=vals)
  ax.set_xlabel('feature %d' % feature_index)
  ax.set_ylabel('SHAP value')
  return fig


@_safe_fig
def shap_waterfall(shap_values, feature_names=None, baseline=None):
  '''SHAP waterfall: single row as cumulative steps.'''
  fig, done = _delegate('shap_waterfall', shap_values, feature_names)
  if done:
    return fig
  vals = np.asarray(shap_values, dtype=float).ravel() \
    if not isinstance(shap_values, dict) else np.array([])
  if vals.size < 1:
    return None
  labels = list(feature_names) if feature_names else \
    ['f%d' % i for i in range(vals.size)]
  order = np.argsort(-np.abs(vals))
  start = float(baseline or 0.0)
  cumulative = start + np.cumsum(vals[order])
  fig, ax = _new_fig('SHAP waterfall')
  ax.plot(range(len(cumulative) + 1),
          [start] + list(cumulative), marker='o')
  ax.set_xticks(range(len(cumulative) + 1))
  ax.set_xticklabels(['base'] + [str(labels[i]) for i in order],
                     rotation=45)
  ax.set_ylabel('cumulative SHAP value')
  return fig


@_safe_fig
def shap_force(shap_values, feature_names=None, baseline=None):
  '''SHAP force: single-row bars around the baseline.'''
  fig, done = _delegate('shap_force', shap_values, feature_names)
  if done:
    return fig
  vals = np.asarray(shap_values, dtype=float).ravel() \
    if not isinstance(shap_values, dict) else np.array([])
  if vals.size < 1:
    return None
  labels = list(feature_names) if feature_names else \
    ['f%d' % i for i in range(vals.size)]
  fig, ax = _new_fig('SHAP force')
  ax.bar([str(v) for v in labels[:vals.size]], vals)
  if baseline is not None:
    ax.axvline(0, color='r')
  ax.set_ylabel('SHAP value')
  return fig


@_safe_fig
def pdp_1d_chart(pdp_result, ice_curves=None):
  '''PDP 1D line with optional ICE overlay curves.

  Args:
    pdp_result: ``{'grid', 'values'}`` from nodes.pdp_1d.
    ice_curves: Optional list of per-row curve value
      lists aligned with the grid.
  '''
  if not isinstance(pdp_result, dict):
    return None
  grid = np.asarray(pdp_result.get('grid'), dtype=float).ravel()
  values = np.asarray(pdp_result.get('values'), dtype=float).ravel()
  if grid.size < 2 or values.size != grid.size:
    return None
  fig, ax = _new_fig('PDP 1D')
  for curve in ice_curves or []:
    curve = np.asarray(curve, dtype=float).ravel()
    if curve.size == grid.size:
      ax.plot(grid, curve, alpha=0.3)
  ax.plot(grid, values, linewidth=2)
  ax.set_xlabel(str(pdp_result.get('feature', 'feature')))
  ax.set_ylabel('mean prediction')
  return fig


pdp_1d = pdp_1d_chart


@_safe_fig
def pdp_2d_heatmap(model, x, feature_a=0, feature_b=1):
  '''PDP 2D heatmap over the top-2 features.

  Args:
    model: Fitted model with ``predict``.
    x: Feature frame/array with at least 5 rows.
    feature_a: First feature (name or index).
    feature_b: Second feature (name or index).
  '''
  import pandas as pd
  frame = x if isinstance(x, pd.DataFrame) else \
    pd.DataFrame(np.asarray(x, dtype=float))
  if len(frame) < 5:
    return None
  cols = list(frame.columns)
  def _idx(feature, default):
    '''Resolve a feature name or index to a column index.

    Args:
      feature: Feature name or index to resolve.
      default: Index to use when the feature is absent.

    Returns:
      Column index of feature, or default when absent.
    '''
    return cols.index(feature) if feature in cols else default
  ia, ib = _idx(feature_a, 0), _idx(feature_b, 1)
  if ia == ib or max(ia, ib) >= len(cols):
    return None
  def _grid(col):
    '''Build a percentile grid for one column.

    Args:
      col: Column name to summarise.

    Returns:
      Sorted unique percentile values for the column.
    '''
    return np.unique(np.percentile(
      frame[col].astype(float), np.linspace(0, 100, 10)))
  ga, gb = _grid(cols[ia]), _grid(cols[ib])
  if ga.size < 2 or gb.size < 2:
    return None
  base = frame.median(numeric_only=True)
  scores = np.zeros((len(gb), len(ga)))
  for j, vb in enumerate(gb):
    for i, va in enumerate(ga):
      row = base.copy()
      row[cols[ia]] = va
      row[cols[ib]] = vb
      scores[j, i] = float(np.asarray(
        model.predict(pd.DataFrame([row]))).ravel()[0])
  fig, ax = _new_fig('PDP 2D heatmap')
  ax.imshow(scores, aspect='auto', origin='lower',
            extent=[ga[0], ga[-1], gb[0], gb[-1]])
  ax.set_xlabel(str(cols[ia]))
  ax.set_ylabel(str(cols[ib]))
  return fig


@_safe_fig
def permutation_importance_bar(importance, feature_names=None):
  '''Bar chart of permutation importance.'''
  labels, values = _importance_pairs(importance, feature_names)
  return _barh('Permutation importance', labels, values)


@_safe_fig
def lime_bar(importances, feature_names=None):
  '''LIME bar from rows, dict, explainer, or array.'''
  rows = importances
  if isinstance(rows, (list, tuple)) and rows \
      and isinstance(rows[0], dict) and 'top_features' in rows[0]:
    first = rows[0].get('top_features') or []
    labels = [str(e.get('feature', 'f%d' % i))
              for i, e in enumerate(first)]
    values = np.array([float(e.get('weight', 0.0)) for e in first])
    return _barh('LIME bar', labels, values)
  fig, done = _delegate('lime_bar', importances, feature_names)
  if done:
    return fig
  labels, values = _importance_pairs(importances, feature_names)
  return _barh('LIME bar', labels, values)


@_safe_fig
def decision_boundary_chart(x, y_true, model=None):
  '''Decision boundary contour over the top-2 features.'''
  fig, done = _delegate('decision_boundary', x, y_true, model)
  if done:
    return fig
  data = np.asarray(x, dtype=float)
  yt = np.asarray(y_true).ravel()
  if data.ndim != 2 or data.shape[1] < 2 or data.shape[0] < 5 \
      or data.shape[0] != yt.size:
    return None
  fig, ax = _new_fig('Decision boundary')
  if model is not None:
    mins = data[:, :2].min(axis=0)
    maxs = data[:, :2].max(axis=0)
    gx, gy = np.meshgrid(np.linspace(mins[0], maxs[0], 20),
                         np.linspace(mins[1], maxs[1], 20))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    try:
      scores = np.asarray(model.predict(grid), dtype=float).ravel()
    except Exception:  # noqa: BLE001 - fill with medians
      full = np.tile(np.median(data, axis=0), (grid.shape[0], 1))
      full[:, :2] = grid
      scores = np.asarray(
        model.predict(full), dtype=float).ravel()
    ax.contourf(gx, gy, scores.reshape(gx.shape), alpha=0.3)
  ax.scatter(data[:, 0], data[:, 1], s=8, c=yt)
  ax.set_xlabel('feature 0')
  ax.set_ylabel('feature 1')
  return fig


@_safe_fig
def tree_plot(model, feature_names=None):
  '''Plot a sklearn tree (or first forest estimator).'''
  fig, done = _delegate('tree_plot', model, feature_names)
  if done and fig is not None:
    return fig
  target = model
  if not hasattr(target, 'tree_'):
    estimators = getattr(model, 'estimators_', None)
    if estimators is None or len(estimators) == 0:
      return None
    target = estimators[0]
  if not hasattr(target, 'tree_'):
    return None
  from sklearn.tree import plot_tree
  fig, ax = plt.subplots()
  plot_tree(target, ax=ax, feature_names=feature_names, filled=True)
  ax.set_title('Decision tree')
  return fig


# --- Diagnostics charts ---

@_safe_fig
def calibration_curve(data, y_pred=None):
  '''Calibration curve from bins or (y_true, y_pred).'''
  bins = data.get('calibration_bins') if isinstance(data, dict) \
    else data
  if y_pred is not None:
    pairs = _pairs(data, y_pred)
    if pairs is None:
      return None
    yt, yp = pairs
    edges = np.linspace(yp.min(), yp.max(), 6)
    bins = []
    for i in range(len(edges) - 1):
      mask = (yp >= edges[i]) & (yp <= edges[i + 1])
      if mask.sum():
        bins.append({'mean_predicted': float(yp[mask].mean()),
                     'mean_actual': float(yt[mask].mean())})
  if not bins:
    return None
  xs = np.array([float(b['mean_predicted']) for b in bins])
  ys = np.array([float(b['mean_actual']) for b in bins])
  if xs.size < 1:
    return None
  fig, ax = _new_fig('Calibration curve')
  ax.plot(xs, ys, marker='o')
  ax.plot([xs.min(), xs.max()], [xs.min(), xs.max()], '--')
  ax.set_xlabel('mean predicted')
  ax.set_ylabel('mean actual')
  return fig


@_safe_fig
def residual_vs_fitted(y_true, y_pred):
  '''Residuals vs fitted values scatter.'''
  pairs = _pairs(y_true, y_pred)
  if pairs is None:
    return None
  yt, yp = pairs
  fig, ax = _new_fig('Residuals vs fitted')
  ax.scatter(yp, yt - yp, s=10)
  ax.axhline(0, color='r')
  ax.set_xlabel('fitted')
  ax.set_ylabel('residual')
  return fig


@_safe_fig
def residual_hist_qq(y_true, y_pred):
  '''Residual histogram with normal QQ (two panels).'''
  pairs = _pairs(y_true, y_pred)
  if pairs is None:
    return None
  resid = pairs[0] - pairs[1]
  fig, axes = plt.subplots(1, 2)
  axes[0].hist(resid, bins=10)
  axes[0].set_title('Residual histogram')
  ordered = np.sort(resid)
  probs = (np.arange(1, len(ordered) + 1) - 0.5) / len(ordered)
  theoretical = np.array([NormalDist().inv_cdf(p) for p in probs])
  axes[1].scatter(theoretical, ordered, s=10)
  axes[1].set_title('Normal QQ')
  return fig


@_safe_fig
def scale_location(y_true, y_pred):
  '''Scale-location: sqrt |standardised residual| vs fitted.'''
  pairs = _pairs(y_true, y_pred)
  if pairs is None:
    return None
  yt, yp = pairs
  resid = yt - yp
  std = float(np.std(resid)) or 1.0
  scale = np.sqrt(np.abs(resid / std))
  fig, ax = _new_fig('Scale-location')
  ax.scatter(yp, scale, s=10)
  ax.set_xlabel('fitted')
  ax.set_ylabel('sqrt |standardised residual|')
  return fig


@_safe_fig
def error_by_segment_bar(error_by_segment):
  '''Bar chart of MAE by segment.'''
  if not error_by_segment:
    return None
  labels = [str(e.get('segment', 'segment_%d' % i))
            for i, e in enumerate(error_by_segment)]
  values = np.array([float(e.get('mae', 0.0))
                     for e in error_by_segment])
  return _barh('Error by segment', labels, values)


@_safe_fig
def prediction_distribution_by_class(y_true, y_pred):
  '''Histograms of predictions grouped by true class.'''
  pairs = _pairs(y_true, y_pred)
  if pairs is None:
    return None
  yt, yp = pairs
  labels = yt if np.unique(yt).size <= 10 else \
    (yt > np.median(yt)).astype(float)
  fig, ax = _new_fig('Prediction distribution by class')
  for label in sorted(set(labels)):
    ax.hist(yp[labels == label], bins=10, alpha=0.5,
            label=str(label))
  ax.legend()
  ax.set_xlabel('prediction')
  return fig


@_safe_fig
def confidence_histogram(proba):
  '''Histogram of max-probability confidence scores.'''
  arr = np.asarray(proba, dtype=float)
  if arr.size < 5:
    return None
  scores = arr.max(axis=1) if arr.ndim == 2 else arr.ravel()
  scores = scores[np.isfinite(scores)]
  if scores.size < 5:
    return None
  fig, ax = _new_fig('Confidence histogram')
  ax.hist(scores, bins=10)
  ax.set_xlabel('confidence')
  return fig


@_safe_fig
def residuals_over_time_chart(y_true, y_pred, dates=None):
  '''Residuals over time / sample order.'''
  fig, done = _delegate('residuals_over_time', y_true, y_pred, dates)
  if done:
    return fig
  pairs = _pairs(y_true, y_pred)
  if pairs is None:
    return None
  axis = dates if dates is not None and len(dates) == len(pairs[0]) \
    else np.arange(len(pairs[0]))
  fig, ax = _new_fig('Residuals over time')
  ax.plot(axis, pairs[0] - pairs[1])
  ax.axhline(0, color='r')
  return fig


@_safe_fig
def cooks_distance(y_true, y_pred, x=None):
  '''Cook's distance stem plot.'''
  fig, done = _delegate('cooks_distance', y_true, y_pred, x)
  if done:
    return fig
  pairs = _pairs(y_true, y_pred)
  if pairs is None:
    return None
  yt, yp = pairs
  resid = yt - yp
  design = np.asarray(x, dtype=float) if x is not None else yp
  if design.ndim == 1:
    design = design.reshape(-1, 1)
  design = np.column_stack([np.ones(len(yt)), design])
  p = design.shape[1]
  if len(yt) <= p + 1:
    return None
  hat = design @ np.linalg.pinv(design.T @ design) @ design.T
  leverage = np.clip(np.diag(hat), 0, 0.999)
  mse = float(np.mean(resid ** 2)) or 1.0
  cooks = (resid ** 2 / (p * mse)) * (leverage / (1 - leverage) ** 2)
  fig, ax = _new_fig('Cooks distance')
  ax.stem(cooks)
  ax.set_xlabel('observation')
  return fig


@_safe_fig
def rec_curve(y_true, y_pred):
  '''REC curve: accuracy within tolerance.'''
  fig, done = _delegate('rec_curve', y_true, y_pred)
  if done:
    return fig
  pairs = _pairs(y_true, y_pred)
  if pairs is None:
    return None
  err = np.sort(np.abs(pairs[0] - pairs[1]))
  acc = np.arange(1, len(err) + 1) / len(err)
  fig, ax = _new_fig('REC curve')
  ax.plot(err, acc)
  ax.set_xlabel('tolerance')
  ax.set_ylabel('accuracy within tolerance')
  return fig


@_safe_fig
def learning_curve(train_scores, val_scores):
  '''Learning curve: train vs validation scores.'''
  fig, done = _delegate('learning_curve', train_scores, val_scores)
  if done:
    return fig
  tr = np.asarray(train_scores, dtype=float).ravel()
  va = np.asarray(val_scores, dtype=float).ravel()
  if min(tr.size, va.size) < 2:
    return None
  n = min(tr.size, va.size)
  fig, ax = _new_fig('Learning curve')
  ax.plot(range(n), tr[:n], label='train')
  ax.plot(range(n), va[:n], label='validation')
  ax.legend()
  return fig


@_safe_fig
def validation_curve(param_values, train_scores, val_scores):
  '''Validation curve vs hyperparameter values.'''
  fig, done = _delegate('validation_curve', param_values,
                        train_scores, val_scores)
  if done:
    return fig
  pv = np.asarray(param_values, dtype=float).ravel()
  tr = np.asarray(train_scores, dtype=float).ravel()
  va = np.asarray(val_scores, dtype=float).ravel()
  if min(pv.size, tr.size, va.size) < 2:
    return None
  n = min(pv.size, tr.size, va.size)
  fig, ax = _new_fig('Validation curve')
  ax.plot(pv[:n], tr[:n], marker='o', label='train')
  ax.plot(pv[:n], va[:n], marker='s', label='validation')
  ax.legend()
  ax.set_xlabel('parameter')
  return fig


@_safe_fig
def loss_vs_epoch(train_loss, val_loss=None):
  '''Loss vs epoch.'''
  fig, done = _delegate('loss_vs_epoch', train_loss, val_loss)
  if done:
    return fig
  tr = np.asarray(train_loss, dtype=float).ravel()
  if tr.size < 2:
    return None
  fig, ax = _new_fig('Loss vs epoch')
  ax.plot(tr, label='train')
  if val_loss is not None:
    va = np.asarray(val_loss, dtype=float).ravel()
    if va.size >= 2:
      ax.plot(va, label='validation')
  ax.legend()
  ax.set_xlabel('epoch')
  return fig


@_safe_fig
def cv_box(cv_scores):
  '''Box plot of cross-validation scores.'''
  fig, done = _delegate('cv_scores_box', cv_scores)
  if done:
    return fig
  if isinstance(cv_scores, dict):
    groups = {str(k): np.asarray(v, dtype=float).ravel()
              for k, v in cv_scores.items()}
    groups = {k: v for k, v in groups.items() if v.size >= 2}
  else:
    arr = np.asarray(cv_scores, dtype=float).ravel()
    groups = {'scores': arr} if arr.size >= 2 else {}
  if not groups:
    return None
  fig, ax = _new_fig('CV scores')
  ax.boxplot(list(groups.values()), tick_labels=list(groups.keys()))
  ax.set_ylabel('score')
  return fig


@_safe_fig
def overfitting_gap(train_scores, val_scores):
  '''Train-validation gap per point.'''
  fig, done = _delegate('overfitting_gap', train_scores, val_scores)
  if done:
    return fig
  tr = np.asarray(train_scores, dtype=float).ravel()
  va = np.asarray(val_scores, dtype=float).ravel()
  if min(tr.size, va.size) < 2:
    return None
  n = min(tr.size, va.size)
  fig, ax = _new_fig('Overfitting gap')
  ax.bar(range(n), tr[:n] - va[:n])
  ax.axhline(0, color='r')
  ax.set_xlabel('fold / epoch')
  return fig
