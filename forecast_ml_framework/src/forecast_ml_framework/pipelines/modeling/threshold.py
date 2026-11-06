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
  '''Round per parity: <0.0001 to 6dp, <0.01 to 4dp, else 2dp.'''
  if value < 0.0001:
    return float(np.round(value, 6))
  if value < 0.01:
    return float(np.round(value, 4))
  return float(np.round(value, 2))


def _mcc_curve(y_true, proba):
  '''Return threshold maximising MCC over parity step grids.'''
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

  thresholds = np.arange(0, 1, 0.01)
  best = thresholds[np.argmax(score(thresholds))]
  if best == 0:
    thresholds = np.arange(0, 0.01, 0.0001)
    best = thresholds[np.argmax(score(thresholds))]
    if best == 0:
      thresholds = np.arange(0, 0.0001, 0.000001)
      best = thresholds[np.argmax(score(thresholds))]
  return float(best)


def select_threshold(y_true, proba, method):
  '''Select a classification threshold by the given method.

  Args:
    y_true: Binary target (0/1).
    proba: Predicted probability of the positive class.
    method: One of pr_curve, roc_curve, mcc_curve,
      ks_statistics. Unknown methods return 0.5.

  Returns:
    Selected threshold as float, rounded per parity.
  '''
  y_true = np.asarray(y_true)
  proba = np.asarray(proba, dtype=float)
  if len(y_true) != len(proba):
    raise ValueError(
      'length mismatch: y_true and proba must have equal lengths')
  if len(y_true) == 0 or len(np.unique(y_true)) < 2:
    return 0.5
  if not np.isfinite(proba).all():
    return 0.5
  if proba.size and (proba.min() < 0 or proba.max() > 1):
    return 0.5
  method_name = str(method or '').strip().lower()
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
  return _round_threshold(selected)
