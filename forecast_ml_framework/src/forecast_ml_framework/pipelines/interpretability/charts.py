#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Interpretability and diagnostics charts (Phase 3).

Unique interpretability charts live here: SHAP
beeswarm/bar/heatmap, PDP 1D (+ICE), PDP 2D heatmap,
permutation importance bar, calibration curve, and
diagnostics charts with no ``modeling.charts``
equivalent.

Charts already provided by
``pipelines.modeling.charts`` (SHAP dependence/
waterfall/force, LIME bar, decision boundary, tree,
Cook, REC, residuals over time, learning/validation/
loss/CV/overfitting) are imported directly below --
no local copies and no delegation shim. Ambiguous
inputs are adapted at the boundary only: ``lime_bar``
accepts ``lime_explain`` rows and ``tree_plot``
accepts a forest via its first estimator.

Every public function returns a Figure or None on
insufficient data, never raises, and closes its
figure in a finally block so no figure slot leaks.
'''

import logging
from statistics import NormalDist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from forecast_ml_framework.pipelines.modeling.charts import (
  _new_fig,
  _pairs,
  _safe_fig,
  cooks_distance,
  cv_scores_box,
  decision_boundary,
  learning_curve,
  lime_bar as _modeling_lime_bar,
  loss_vs_epoch,
  overfitting_gap,
  rec_curve,
  residuals_over_time,
  shap_dependence,
  shap_force,
  shap_waterfall,
  tree_plot as _modeling_tree_plot,
  validation_curve,
)

LOG = logging.getLogger(__name__)

# Direct re-exports under the interpretability names.
decision_boundary_chart = decision_boundary
residuals_over_time_chart = residuals_over_time
cv_box = cv_scores_box

__all__ = [
  'calibration_curve',
  'confidence_histogram',
  'cooks_distance',
  'cv_box',
  'decision_boundary_chart',
  'error_by_segment_bar',
  'learning_curve',
  'lime_bar',
  'loss_vs_epoch',
  'overfitting_gap',
  'pdp_1d',
  'pdp_1d_chart',
  'pdp_2d_heatmap',
  'permutation_importance_bar',
  'prediction_distribution_by_class',
  'rec_curve',
  'residual_hist_qq',
  'residual_vs_fitted',
  'residuals_over_time_chart',
  'scale_location',
  'shap_bar',
  'shap_beeswarm',
  'shap_dependence',
  'shap_force',
  'shap_heatmap',
  'shap_waterfall',
  'tree_plot',
  'validation_curve',
]


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


@_safe_fig
def lime_bar(importances, feature_names=None):
  '''LIME bar from rows, dict, explainer, or array.

  ``lime_explain`` rows (``top_features``) are adapted
  to a feature->weight dict, then rendered by
  ``modeling.charts.lime_bar``.
  '''
  if isinstance(importances, (list, tuple)) and importances \
      and isinstance(importances[0], dict) \
      and 'top_features' in importances[0]:
    first = importances[0].get('top_features') or []
    weights = {str(e.get('feature', 'f%d' % i)):
               float(e.get('weight', 0.0))
               for i, e in enumerate(first)}
    return _modeling_lime_bar(weights, feature_names)
  return _modeling_lime_bar(importances, feature_names)


@_safe_fig
def tree_plot(model, feature_names=None):
  '''Plot a sklearn tree (or first forest estimator).'''
  target = model
  if not hasattr(target, 'tree_'):
    estimators = getattr(model, 'estimators_', None)
    if estimators is None or len(estimators) == 0:
      return None
    target = estimators[0]
  return _modeling_tree_plot(target, feature_names)


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
