#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Per-class and macro ROC-AUC tracking for multi-target evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

IGNORE_INDEX = -1.0


class MultilabelAUC:
  """Accumulate predictions/targets and report per-class + macro AUC.

  Classes without both outcomes in the accumulated set are reported as
  ``nan`` and excluded from the macro average (macro-AUC on the private LB
  is computed by the organizers over all 12, but during CV a missing class
  simply carries no information).
  """

  def __init__(self, target_columns: list[str]) -> None:
    """Initialize accumulators.

    Args:
        target_columns: Canonical target order used in reports.
    """
    self.target_columns = list(target_columns)
    self.reset()

  def reset(self) -> None:
    """Clear all accumulated state between epochs."""
    self._probs: list[np.ndarray] = []
    self._targets: list[np.ndarray] = []
    self._uids: list[str] = []

  def update(
    self,
    probs: np.ndarray,
    targets: np.ndarray,
    study_uids: list[str],
  ) -> None:
    """Append one batch of results.

    Args:
        probs: ``(batch, n_targets)`` probabilities after sigmoid.
        targets: ``(batch, n_targets)`` values possibly containing -1.
        study_uids: Study identifiers aligned with rows.
    """
    self._probs.append(np.asarray(probs, dtype=np.float64))
    self._targets.append(np.asarray(targets))
    self._uids.extend(study_uids)

  def per_class(self) -> dict[str, float]:
    """Compute AUC per class ignoring unknown targets.

    Returns:
        Mapping target -> AUC (nan when the class lacks both outcomes,
        or when no validation batch has been accumulated yet - e.g.
        ``limit_val_batches=0`` or an empty val split).
    """
    if not self._probs:
      return {name: float('nan') for name in self.target_columns}
    probs = np.concatenate(self._probs, axis=0)
    targets = np.concatenate(self._targets, axis=0)
    scores: dict[str, float] = {}
    for col, name in enumerate(self.target_columns):
      mask = targets[:, col] != IGNORE_INDEX
      values = targets[mask, col]
      if values.size == 0 or len(np.unique(values)) < 2:
        scores[name] = float('nan')
        continue
      scores[name] = float(roc_auc_score(values, probs[mask, col]))
    return scores

  def macro(self) -> float:
    """Compute the mean AUC across computable classes.

    Returns:
        Macro AUC; 0.5 when no class is computable yet.
    """
    values = [v for v in self.per_class().values() if not np.isnan(v)]
    return float(np.mean(values)) if values else 0.5

  def summary(self) -> dict[str, float]:
    """Return a flat metric dictionary ready for logging.

    Returns:
        Dictionary with ``auc_macro`` plus one entry per target.
    """
    result = {f'auc/{name}': value for name, value in self.per_class().items()}
    result['auc/macro'] = self.macro()
    return result

  @property
  def study_uids(self) -> list[str]:
    """Return all accumulated study identifiers in order.

    Returns:
        List of StudyInstanceUID strings.
    """
    return self._uids

  def stacked(self) -> tuple[np.ndarray, np.ndarray]:
    """Return concatenated probability and target matrices.

    Returns:
        Tuple ``(probs, targets)`` suitable for OOF persistence;
        zero-row ``(0, n_targets)`` arrays when nothing accumulated.
    """
    if not self._probs:
      empty = np.empty((0, len(self.target_columns)), dtype=np.float64)
      return empty, empty.copy()
    return (
      np.concatenate(self._probs, axis=0),
      np.concatenate(self._targets, axis=0),
    )
