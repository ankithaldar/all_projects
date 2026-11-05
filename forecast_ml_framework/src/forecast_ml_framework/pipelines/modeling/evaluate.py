#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Evaluation node: global, per-bucket and per-decile metrics.'''

import numpy as np
import pandas as pd

from forecast_ml_framework.pipelines.modeling.metrics import (
  compute_metrics,
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


def _by_group(y_true, y_pred, proba, groups, task):
  '''Compute metrics per group label.

  Args:
    y_true: True labels or values.
    y_pred: Predicted labels or values.
    proba: Probabilities or None.
    groups: Group label per row.
    task: 'classification' or 'regression'.

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
        sub, task)
  return out


def evaluate_model(y_true, y_pred, y_proba, x_test, params):
  '''Evaluate predictions globally and by bucket/decile.

  Args:
    y_true: True labels or values.
    y_pred: Predicted labels or values.
    y_proba: Probabilities (classification) or None.
    x_test: Test features (DataFrame) for bucket column.
    params: modeling_params dict with task_type and
      bucket_segment_col.

  Returns:
    Dict with global, by_bucket and by_decile metrics.
  '''
  params = params or {}
  task = params.get('task_type', 'classification')
  proba = None if y_proba is None else np.asarray(y_proba)
  result = {'global': compute_metrics(
      y_true, y_pred, proba, task)}
  bucket_col = params.get('bucket_segment_col')
  if bucket_col and hasattr(x_test, 'columns') and (
          bucket_col in x_test.columns):
    result['by_bucket'] = _by_group(
        y_true, y_pred, proba, x_test[bucket_col], task)
  else:
    result['by_bucket'] = {}
  labels = _deciles(y_true, y_pred, proba, task)
  result['by_decile'] = _by_group(
      y_true, y_pred, proba, labels, task)
  return result
