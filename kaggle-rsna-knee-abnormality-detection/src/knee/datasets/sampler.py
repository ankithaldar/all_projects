#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Balanced multi-label sampling for rare-class exposure.

WeightedRandomSampler weights that equalize the expected positive
exposure of every class per epoch: a study's weight is the mean inverse
prevalence of its positive labels, normalized to mean 1 and clamped so
no study is oversampled beyond ``oversample_cap`` (default 8x,
BLUEPRINT section 7).
"""

from __future__ import annotations

import numpy as np


class BalancedMultiLabelSampler:
  """Computes per-study sampling weights from the label matrix.

  Args:
      oversample_cap: Upper clamp on normalized sample weights.
      eps: Floor applied to class prevalences.
  """

  def __init__(self, oversample_cap: float = 8.0, eps: float = 1e-6) -> None:
    """Validate the cap.

    Args:
        oversample_cap: Max oversampling factor vs. uniform sampling.
        eps: Prevalence floor for classes with zero positives.

    Raises:
        ValueError: If ``oversample_cap`` is below 1.
    """
    if oversample_cap < 1.0:
      raise ValueError('oversample_cap must be >= 1')
    self.oversample_cap = float(oversample_cap)
    self.eps = eps

  def compute_weights(self, targets: np.ndarray) -> np.ndarray:
    """Inverse-prevalence mean weights, normalized and capped.

    Args:
        targets: Binary label matrix ``(N, C)``.

    Returns:
        Positive weights ``(N,)`` with mean 1 and maximum at most
        ``oversample_cap``; all-negative studies keep weight 1.
    """
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[0] == 0:
      raise ValueError('targets must be a non-empty (N, C) matrix')
    prevalence = np.clip(targets.mean(axis=0), self.eps, None)
    inverse = 1.0 / prevalence  # (C,)
    pos_counts = targets.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
      raw = np.where(
        pos_counts > 0, (targets * inverse).sum(axis=1) / np.maximum(
          pos_counts, 1.0
        ), 1.0,
      )
    mean = raw.mean()
    weights = np.where(mean > 0, raw / max(mean, self.eps), 1.0)
    return np.minimum(weights, self.oversample_cap).astype(np.float64)

  __call__ = compute_weights
