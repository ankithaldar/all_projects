#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Class-imbalance statistics for classification targets.'''
# pylint: disable=broad-exception-caught

from decimal import Decimal

import numpy as np
import pandas as pd

from forecast_ml_framework.pipelines.modeling.ntile import (
  normalize_task_kind,
)


def _label(value):
  '''Return a plain Python label for dict keys.'''
  item = getattr(value, 'item', None)
  if callable(item):
    try:
      return item()
    except Exception:  # noqa: BLE001
      return value
  return value


def _is_positive_label(label):
  '''Return True if a sole label denotes the positive class.

  Positive is a non-zero numeric label (1 for 0/1 targets) or a
  truthy string label; 0 / False / '0' / 'false' are negative.
  '''
  if isinstance(label, (bool, np.bool_)):
    return bool(label)
  if isinstance(label, (int, float, np.integer, np.floating)):
    return float(label) != 0.0
  text = str(label).strip().lower()
  if text in ('0', 'false', 'no', 'negative', 'neg'):
    return False
  return True


def _validate_label_item(item):
  '''Reject non-scalar/NaN/Inf/None inside a label, recursing tuples.'''
  if isinstance(item, tuple):
    for sub in item:
      _validate_label_item(sub)
    return
  if isinstance(item, dict):
    for sub in list(item.keys()) + list(item.values()):
      _validate_label_item(sub)
    raise ValueError(
      'compute_imbalance y labels must be scalar: dict labels '
      'are not supported')
  if isinstance(item, (list, np.ndarray)):
    for sub in np.asarray(item, dtype=object).ravel() \
            if isinstance(item, np.ndarray) else item:
      _validate_label_item(sub)
    raise ValueError(
      'compute_imbalance y labels must be scalar: list/array '
      'labels inside a tuple are not supported')
  if isinstance(item, (set, frozenset, bytearray)):
    for sub in item:
      _validate_label_item(sub)
    raise ValueError(
      'compute_imbalance y labels must be scalar: set/frozenset/'
      'bytearray labels inside a tuple are not supported')
  if isinstance(item, (complex, np.complexfloating)):
    if not (np.isfinite(float(item.real))
            and np.isfinite(float(item.imag))):
      raise ValueError(
        'compute_imbalance y must not contain non-finite (inf) labels')
    return
  if isinstance(item, Decimal) and not item.is_finite():
    raise ValueError(
      'compute_imbalance y must not contain non-finite (inf) labels')
  if item is None:
    raise ValueError(
      'compute_imbalance y must not contain NaN/NaT/NA/None labels')
  try:
    isna = pd.isna(item)
  except Exception:  # noqa: BLE001
    isna = False
  if isna is True:
    raise ValueError(
      'compute_imbalance y must not contain NaN/NaT/NA/None labels')
  if isinstance(item, (float, np.floating)) and not np.isfinite(
          float(item)):
    raise ValueError(
      'compute_imbalance y must not contain non-finite (inf) labels')


def compute_imbalance(y, task='classification'):
  '''Compute imbalance statistics for a target.

  Args:
    y: Target labels (array-like or Series).
    task: 'classification' or 'regression'.

  Returns:
    Dict with real_rate (positive-class rate for binary,
    minority-class rate for multiclass, positive rate (0 or 1)
    for single-class, 0.0 for regression), scale_pos_weight
    (n_neg / n_pos for binary, 1.0 otherwise), and class_weights
    (balanced weights n / (k * count) per class, {} for
    regression). Single-class real_rate is the positive-class
    rate, not a minority rate: all-positive gives 1.0 and
    all-negative gives 0.0.

  Raises:
    ValueError: If y is None, empty, contains NaN, or is 2-D
      (list-of-lists / 2-D array labels are rejected, not
      flattened; use tuple labels for multi-column classes).
  '''
  if y is None:
    raise ValueError('compute_imbalance requires a non-None y')
  if not isinstance(y, (list, tuple, np.ndarray, pd.Series)):
    raise ValueError(
      'compute_imbalance y must be a list, tuple, ndarray, or '
      'Series array-like; str/generator/other iterables are '
      'not supported')
  if isinstance(y, np.ndarray) and y.ndim > 1:
    raise ValueError(
      'compute_imbalance y must be 1-D: 2-D array labels are '
      'not supported; pass 1-D labels (tuple labels for '
      'multi-column classes)')
  if isinstance(y, (list, tuple)) and any(
          isinstance(item, (list, np.ndarray)) for item in y):
    raise ValueError(
      'compute_imbalance y must be 1-D: list-of-lists labels '
      'are not supported; pass 1-D labels (tuple labels for '
      'multi-column classes)')
  if hasattr(y, 'to_numpy'):
    values = y.to_numpy()
  elif isinstance(y, (list, tuple)) and any(
          isinstance(item, tuple) for item in y):
    # List-of-tuples: np.asarray would build a 2-D array and
    # ravel() would flatten the tuple elements into scalars.
    # Build a 1-D object array preserving each tuple as one label
    # (the Series path already does this via to_numpy()).
    values = np.empty(len(y), dtype=object)
    for idx, item in enumerate(y):
      values[idx] = item
  else:
    values = np.asarray(y)
  values = np.asarray(values)
  if values.ndim > 1:
    raise ValueError(
      'compute_imbalance y must be 1-D: 2-D labels are not '
      'supported; pass 1-D labels (tuple labels for '
      'multi-column classes)')
  values = values.ravel()
  if values.dtype == object and any(
          isinstance(item, (list, np.ndarray)) for item in values):
    raise ValueError(
      'compute_imbalance y must be 1-D: list-of-lists labels '
      'are not supported; pass 1-D labels (tuple labels for '
      'multi-column classes)')
  if values.size == 0:
    raise ValueError('compute_imbalance requires a non-empty y')
  # Reject NaN/inf for numeric and object targets alike.
  if values.dtype.kind in 'fc':
    if np.isnan(values).any():
      raise ValueError('compute_imbalance y must not contain NaN')
    if np.isinf(values).any():
      raise ValueError(
        'compute_imbalance y must not contain non-finite (inf) labels')
  elif values.dtype.kind in 'Mm':
    # Datetime/timedelta NaT is missing, not a class.
    try:
      isnat = np.isnat(values)
    except Exception:  # noqa: BLE001
      isnat = pd.isna(values)
    if bool(np.asarray(isnat).any()):
      raise ValueError('compute_imbalance y must not contain NaT')
  elif values.dtype == object:
    for item in values:
      _validate_label_item(item)
  if normalize_task_kind(task) != 'classification':
    return {'real_rate': 0.0, 'scale_pos_weight': 1.0,
            'class_weights': {}}
  labels, counts = np.unique(values, return_counts=True)
  n = float(values.size)
  k = float(len(labels))
  class_weights = {}
  for label, count in zip(labels, counts):
    class_weights[_label(label)] = float(n / (k * count))
  if len(labels) == 2:
    order = np.argsort(labels)
    # Positive class is the larger label (1 for 0/1 targets).
    pos_count = float(counts[order[-1]])
    neg_count = float(counts[order[0]])
    real_rate = float(pos_count / n)
    if pos_count:
      scale_pos_weight = float(neg_count / pos_count)
    else:
      scale_pos_weight = 1.0
  elif len(labels) == 1:
    # Single-class: real_rate is the positive-class rate (0 or 1),
    # not a minority rate (which would always be 1.0 here).
    real_rate = 1.0 if _is_positive_label(labels[0]) else 0.0
    scale_pos_weight = 1.0
  else:
    real_rate = float(counts.min() / n)
    scale_pos_weight = 1.0
  return {'real_rate': real_rate,
          'scale_pos_weight': scale_pos_weight,
          'class_weights': class_weights}
