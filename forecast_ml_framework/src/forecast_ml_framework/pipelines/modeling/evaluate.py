#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Evaluation node: global, per-bucket and per-decile metrics.

Metric selection: ``params['metrics']`` (alias
``params['metrics_to_calculate']``), provisioned from
``parameters.yml`` by the config worker, controls which
metrics are calculated:

- absent -> ``DEFAULT_METRICS`` only (curated subset, ~15
  per task; the user does not see every metric by default);
- ``'all'`` -> every name in ``METRIC_REGISTRY`` for the task;
- a list -> only those names (unknown -> ``ValueError``
  listing valid names).

Only selected metrics are computed (shared intermediates
once, unselected families skipped). The result includes
``metrics_calculated`` (sorted list actually computed) and
``available_metrics`` (registry count, so the user knows
more metrics exist and can select them).
'''

import numpy as np
import pandas as pd

from forecast_ml_framework.pipelines.modeling.metrics import (
  CLASSIFICATION_METRICS, METRIC_REGISTRY, REGRESSION_METRICS,
  compute_metrics, resolve_selected_metrics,
)
from forecast_ml_framework.pipelines.modeling.ntile import (
  normalize_task_kind,
  parse_eval_metric,
)

# Map parity/ntile metric names to forecast registry names.
_METRIC_ALIASES = {
    'lift': 'lift_top_decile',
    'auc': 'roc_auc',
    'brier': 'brier_score',
    'precision': 'precision_weighted',
    'recall': 'recall_weighted',
    'f1': 'f1_weighted',
}


def _parse_eval_metric(metric_name, task):
  '''Parse the eval metric via ``ntile.parse_eval_metric``.

  Delegates entirely to ntile (single owner of the parity
  grammar); the 1-based ntile conversion happens in
  ``_selected_ntile``.
  '''
  return parse_eval_metric(metric_name, task)


def _registry_name(metric):
  '''Map a parsed metric name to a forecast registry name.'''
  name = str(metric or '').strip().lower()
  mapped = _METRIC_ALIASES.get(name, name)
  if mapped in METRIC_REGISTRY:
    return mapped
  if name in METRIC_REGISTRY:
    return name
  return mapped


def _deciles(_y_true, y_pred, proba, task, n=10):
  # pylint: disable=unused-argument
  '''Assign prediction-decile labels.

  Args:
    _y_true: Unused (kept for signature symmetry).
    y_pred: Predicted labels or values.
    proba: Probabilities or None.
    task: 'classification' or 'regression'.
    n: Nominal number of deciles.

  Returns:
    Array of integer decile labels.
  '''
  if normalize_task_kind(task) == 'classification':
    arr = np.asarray(proba, dtype=float)
    score = arr[:, 1] if arr.ndim == 2 and arr.shape[1] > 1 else arr
  else:
    score = np.asarray(y_pred, dtype=float)
  try:
    return pd.qcut(score, q=min(n, max(2, len(score) // 2)),
                   labels=False, duplicates='drop')
  except (ValueError, TypeError):
    return np.zeros(len(score), dtype=int)


def _by_group(y_true, y_pred, proba, groups, task, selected):
  '''Compute metrics per group label.

  Args:
    y_true: True labels or values.
    y_pred: Predicted labels or values.
    proba: Probabilities or None.
    groups: Group label per row.
    task: 'classification' or 'regression'.
    selected: Selected metric names (same as global).

  Returns:
    Dict of group label to metrics dict.
  '''
  out = {}
  groups = np.asarray(groups)
  for label in pd.unique(groups):
    mask = groups == label
    if mask.sum() < 1:
      continue
    sub = None if proba is None else proba[mask]
    out[str(label)] = compute_metrics(
        np.asarray(y_true)[mask], np.asarray(y_pred)[mask],
        sub, task, selected=selected)
  return out


def evaluate_model(y_true, y_pred, y_proba, x_test, params):
  '''Evaluate predictions globally and by bucket/decile.

  Args:
    y_true: True labels or values.
    y_pred: Predicted labels or values.
    y_proba: Probabilities (classification) or None.
    x_test: Test features (DataFrame) for bucket column.
    params: modeling_params dict with task_type,
      bucket_segment_col, and optionally ``metrics`` /
      ``metrics_to_calculate``: absent for the curated
      ``DEFAULT_METRICS`` subset, ``'all'`` for every
      registry metric, or a list of metric names to
      calculate (unknown name -> ``ValueError``).

  Returns:
    Dict with global, by_bucket and by_decile metrics, plus
    ``metrics_calculated`` (sorted names actually computed)
    and ``available_metrics`` (count of metrics in
    ``METRIC_REGISTRY``, so more exist than the default).
  '''
  params = params or {}
  task = normalize_task_kind(params.get('task_type', 'classification'))
  requested = params.get('metrics', params.get(
      'metrics_to_calculate'))
  selected = resolve_selected_metrics(requested, task)
  proba = None if y_proba is None else np.asarray(y_proba)
  result = {'global': compute_metrics(
      y_true, y_pred, proba, task, selected=selected),
      'metrics_calculated': sorted(selected),
      'available_metrics': len(METRIC_REGISTRY)}
  bucket_col = params.get('bucket_segment_col')
  if bucket_col and hasattr(x_test, 'columns') and (
          bucket_col in x_test.columns):
    result['by_bucket'] = _by_group(
        y_true, y_pred, proba, x_test[bucket_col], task,
        selected)
  else:
    result['by_bucket'] = {}
  labels = _deciles(y_true, y_pred, proba, task)
  result['by_decile'] = _by_group(
      y_true, y_pred, proba, labels, task, selected)
  result['selected_ntile'] = None
  eval_metric = params.get('eval_metric')
  if eval_metric:
    result['selected_ntile'] = _selected_ntile(
        eval_metric, y_true, y_pred, proba, task, selected, result)
  return result


def _selected_ntile(eval_metric, y_true, y_pred, proba, task, selected,
                    result):
  '''Build the selected_ntile dict for ``eval_metric``.

  ``_deciles`` labels are 0-based (0..n-1) while parity ntile
  labels are 1-based (1..n), so the requested ntile ``n``
  maps to group label ``str(n - 1)``; when that label is
  absent (e.g. qcut collapsed groups), fall back to the
  largest available ntile label.
  '''
  metric, kind, ntile = _parse_eval_metric(eval_metric, task)
  registry = _registry_name(metric)
  task_name = normalize_task_kind(task)
  valid = (CLASSIFICATION_METRICS if task_name == 'classification'
           else REGRESSION_METRICS)
  effective = list(selected)
  if registry in valid and registry not in effective:
    effective.append(registry)
  groups = result.get('by_decile') or {}
  if kind == 'centile':
    centile_labels = _deciles(y_true, y_pred, proba, task, n=100)
    groups = _by_group(
        y_true, y_pred, proba, centile_labels, task, effective)
    result['by_centile'] = groups
  elif kind != 'overall':
    # Generic branch: decile/quintile/quartile recompute with
    # their own ``n`` so the top tile matches the request.
    tile_labels = _deciles(y_true, y_pred, proba, task,
                           n=ntile if ntile else 10)
    groups = _by_group(
        y_true, y_pred, proba, tile_labels, task, effective)
  score = None
  if kind == 'overall':
    score = result['global'].get(registry)
    if score is None and registry in valid:
      single = compute_metrics(
          y_true, y_pred, proba, task, selected=[registry])
      score = single.get(registry)
  elif groups:
    label = str(ntile - 1)
    if label not in groups:
      label = max(groups, key=lambda k: int(k))
    group = groups[label]
    if registry in group:
      score = group[registry]
  if score is None:
    score = result['global'].get(registry)
  try:
    score = float(score)
  except (TypeError, ValueError):
    score = float('nan')
  return {'eval_metric': eval_metric, 'metric': metric,
          'score': score, 'ntile': int(ntile)}
