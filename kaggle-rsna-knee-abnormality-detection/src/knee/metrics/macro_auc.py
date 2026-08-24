#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Macro-averaged ROC-AUC with NaN-class accounting.

Rare findings (Fracture, Baker's) are frequently absent from a fold's
validation split; per-class AUC is undefined there and must be skipped
-- never zero-filled -- so the macro mean stays an honest ranking
metric (BLUEPRINT sections 1 and 6).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def macro_auc(
  y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[float, np.ndarray, int]:
  """Per-class ROC-AUC averaged over classes with two label states.

  Args:
      y_true: Binary ground truth ``(N, C)`` (bool or 0/1 ints).
      y_prob: Predicted probabilities ``(N, C)``.

  Returns:
      Tuple ``(mean_auc, per_class, skipped)`` where ``per_class`` is a
      length-C array holding NaN for undefined classes and ``skipped``
      counts them. ``mean_auc`` is NaN when every class was skipped.
  """
  y_true = np.asarray(y_true)
  y_prob = np.asarray(y_prob, dtype=np.float64)
  n_classes = y_true.shape[1] if y_true.ndim > 1 else 1
  per_class = np.full(n_classes, np.nan, dtype=np.float64)
  skipped = 0
  for c in range(n_classes):
    t = y_true[:, c].astype(bool) if y_true.ndim > 1 else y_true.astype(bool)
    p = y_prob[:, c] if y_prob.ndim > 1 else y_prob[:, 0]
    if t.size == 0 or not np.isfinite(p).all() or np.unique(t).size < 2:
      skipped += 1
      continue
    per_class[c] = roc_auc_score(t, p)
  valid = per_class[np.isfinite(per_class)]
  mean = float(valid.mean()) if valid.size else float('nan')
  return mean, per_class, skipped
