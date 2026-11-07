#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Threshold selection (select_threshold parity, Task 5).'''

import warnings

import numpy as np
from sklearn.metrics import (
  matthews_corrcoef,
  precision_recall_curve,
  roc_curve,
)


def _round_threshold(value):
  '''Round a threshold for display only (never for selection).

  Parity display rounding: <0.0001 to 6dp, <0.01 to 4dp,
  else 2dp. ``select_threshold`` returns the full-precision
  optimum; callers that display a threshold may pass it
  through this helper.
  '''
  if value < 0.0001:
    return float(np.round(value, 6))
  if value < 0.01:
    return float(np.round(value, 4))
  return float(np.round(value, 2))


def _mcc_curve(y_true, proba):
  '''Return the MCC-optimal threshold plateau midpoint.

  Scores the parity 0.01 grid by MCC, then returns the
  midpoint of the contiguous plateau of thresholds tied at
  the best score (the first argmax is the fragile left
  edge of the plateau). Finer grids are probed when the
  coarse optimum sits at 0.
  '''
  def score(thresholds):
    '''Score thresholds by MCC, 0 where undefined.'''
    scores = []
    for thr in thresholds:
      pred = (proba >= thr).astype(int)
      if len(np.unique(pred)) < 2:
        scores.append(0.0)
      else:
        scores.append(matthews_corrcoef(y_true, pred))
    return np.asarray(scores, dtype=float)

  def plateau_midpoint(thresholds, scores):
    '''Return midpoint of the best-score plateau around argmax.'''
    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]
    if best_score == 0:
      return float(thresholds[best_idx])
    left = best_idx
    while left > 0 and scores[left - 1] == best_score:
      left -= 1
    right = best_idx
    while right < len(scores) - 1 and scores[right + 1] == best_score:
      right += 1
    return float((thresholds[left] + thresholds[right]) / 2.0)

  thresholds = np.arange(0, 1, 0.01)
  scores = score(thresholds)
  best = plateau_midpoint(thresholds, scores)
  if best == 0:
    thresholds = np.arange(0, 0.01, 0.0001)
    best = plateau_midpoint(thresholds, score(thresholds))
    if best == 0:
      thresholds = np.arange(0, 0.0001, 0.000001)
      best = plateau_midpoint(thresholds, score(thresholds))
  return float(best)


def select_threshold(y_true, proba, method):
  '''Select a classification threshold by the given method.

  Args:
    y_true: Binary target (0/1), 1-D.
    proba: Predicted probability of the positive class, 1-D,
      or (n, 2) binary predict_proba output (column 1 is
      used), or (n, 1) output (column 0 is used). Multiclass
      proba (n, k>2), other shapes, and multiclass y_true
      warn and return 0.5 instead of crashing.
    method: One of pr_curve, roc_curve, mcc_curve,
      ks_statistics. Unknown methods return 0.5.

  Returns:
    Selected threshold as float at full precision (use
    ``_round_threshold`` only to display it).
  '''
  y_true = np.asarray(y_true)
  try:
    proba_raw = np.asarray(proba, dtype=float)
  except (TypeError, ValueError):
    warnings.warn(
      'proba must be numeric probabilities; using default 0.5',
      stacklevel=2)
    return 0.5
  if proba_raw.ndim == 0 or y_true.ndim == 0:
    raise ValueError(
      'length mismatch: y_true and proba must have equal lengths')
  # Shape validation precedes the length check so invalid
  # shapes (e.g. y_true (1, n), proba (2, n), multiclass
  # proba (n, k>2)) warn and fall back to 0.5 instead of
  # crashing downstream in sklearn / IndexError paths.
  if y_true.ndim == 2 and y_true.shape[1] == 1:
    y_true = y_true.ravel()
  elif y_true.ndim != 1:
    warnings.warn(
      'y_true must be 1-D binary; using default 0.5',
      stacklevel=2)
    return 0.5
  if proba_raw.ndim == 1:
    proba = proba_raw
  elif proba_raw.ndim == 2 and proba_raw.shape[1] == 2:
    # Binary predict_proba output: use the positive class (col 1).
    proba = proba_raw[:, 1]
  elif proba_raw.ndim == 2 and proba_raw.shape[1] == 1:
    proba = proba_raw[:, 0]
  else:
    warnings.warn(
      'proba must be 1-D binary (or (n, 2) binary predict_proba); '
      'multiclass or other shapes fall back to 0.5',
      stacklevel=2)
    return 0.5
  if len(y_true) != len(proba_raw):
    raise ValueError(
      'length mismatch: y_true and proba must have equal lengths')
  # Non-numeric (e.g. string) labels cannot be scored by
  # sklearn curves / ks as floats: warn and fall back.
  try:
    y_numeric = y_true.astype(float)
  except (TypeError, ValueError):
    warnings.warn(
      'y_true must be numeric binary for threshold selection; '
      'using default 0.5', stacklevel=2)
    return 0.5
  if not np.isfinite(y_numeric).all():
    warnings.warn(
      'y_true must be finite for threshold selection; '
      'using default 0.5', stacklevel=2)
    return 0.5
  y_true = y_numeric
  if len(y_true) == 0 or len(np.unique(y_true)) < 2:
    return 0.5
  if len(np.unique(y_true)) > 2:
    warnings.warn(
      'y_true must be binary for threshold selection; '
      'multiclass targets fall back to 0.5', stacklevel=2)
    return 0.5
  if not np.isfinite(proba).all():
    return 0.5
  if proba.size and (proba.min() < 0 or proba.max() > 1):
    return 0.5
  method_name = str(method or '').strip().lower()
  try:
    return _select_threshold_method(
      y_true, proba, method_name, method)
  except (IndexError, TypeError, ValueError):
    warnings.warn(
      'threshold selection failed for method '
      f'{method!r}; using default 0.5', stacklevel=2)
    return 0.5


def _select_threshold_method(y_true, proba, method_name, method):
  '''Dispatch threshold selection for validated binary data.

  Args:
    y_true: Numeric binary target, 1-D.
    proba: Positive-class probabilities, 1-D.
    method_name: Normalised method name.
    method: Original method (for warnings).

  Returns:
    Selected threshold as float at full precision.

  Raises:
    ValueError: Propagated from sklearn metrics, caught by
      select_threshold which warns and returns 0.5.
  '''
  if method_name == 'pr_curve':
    precision, recall, thresholds = precision_recall_curve(
      y_true, proba)
    fscore = np.divide(
      2 * precision[:-1] * recall[:-1],
      precision[:-1] + recall[:-1],
      out=np.zeros(len(precision) - 1), where=(
        precision[:-1] + recall[:-1]) != 0)
    selected = float(thresholds[np.argmax(fscore)])
  elif method_name == 'roc_curve':
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    # Exclude the leading inf threshold (fpr=0, tpr=1) and clip
    # rates/thresholds to [0, 1]; it would otherwise always win.
    fpr, tpr, thresholds = fpr[1:], tpr[1:], thresholds[1:]
    if len(thresholds) == 0:
      return 0.5
    thresholds = np.clip(thresholds, 0, 1)
    gmeans = np.sqrt(np.clip(tpr, 0, 1) * np.clip(1 - fpr, 0, 1))
    best = int(np.argmax(gmeans))
    if gmeans[best] == 0:
      return 0.5
    selected = float(thresholds[best])
  elif method_name == 'mcc_curve':
    selected = _mcc_curve(y_true, proba)
  elif method_name == 'ks_statistics':
    order = np.argsort(-proba)
    target = y_true[order].astype(float)
    n_pos = target.sum()
    n_neg = len(target) - n_pos
    if n_pos == 0 or n_neg == 0:
      return 0.5
    tpr = np.cumsum(target) / n_pos
    fpr = np.cumsum(1 - target) / n_neg
    selected = float(proba[order[np.argmax(tpr - fpr)]])
  else:
    warnings.warn(f'unknown threshold method {method!r}: '
                  f'using default 0.5', stacklevel=2)
    return 0.5
  return float(selected)
