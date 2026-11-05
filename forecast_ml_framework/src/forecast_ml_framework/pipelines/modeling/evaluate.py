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
  METRIC_REGISTRY, compute_metrics, resolve_selected_metrics,
)


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
  if task == 'classification':
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
  task = params.get('task_type', 'classification')
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
  return result
