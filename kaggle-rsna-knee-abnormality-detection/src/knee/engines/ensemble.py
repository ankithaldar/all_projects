#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ensembling: greedy hill-climb blend + submission serialization.

Members are scored on honest gold-OOF macro-AUC (the experiment metric,
BLUEPRINT section 6); the blender adds members only while the pooled
gold-OOF improves, tracking each member's share of the mixture.
Optional rank-averaging replaces probabilities with per-class ranks
before blending (scipy ``rankdata``), an alternative fusion that can be
more robust to probability miscalibration across members.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from knee.config_params.schema import TARGETS
from knee.metrics.macro_auc import macro_auc

_DEFAULT_ALPHAS = (0.1, 0.25, 0.5, 0.75, 0.9)
_MIN_IMPROVEMENT = 1e-5


def save_submission(frame: pd.DataFrame, path: str | Path) -> None:
  """Write a competition submission with the canonical column order.

  Args:
      frame: Frame holding StudyInstanceUID plus TARGET columns.
      path: Destination CSV; parent directories are created.
  """
  columns = ['StudyInstanceUID'] + [t for t in TARGETS if t in frame.columns]
  out = frame[columns].copy()
  targets = [c for c in columns if c != 'StudyInstanceUID']
  if targets:
    out[targets] = out[targets].clip(lower=0.0, upper=1.0)
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  out.to_csv(destination, index=False)


class GreedyBlender:
  """Greedy ensemble selection over OOF probability frames.

  Strategy Strategy-pattern free on purpose: one fit() pass implements
  the hill climb; weights_ always sums to 1 so downstream consumers can
  apply it to test submissions directly.

  Args:
      rank_average: Blend per-class ranks instead of raw probabilities.
      alphas: Candidate shares tried for every new member.
      max_members: Hard cap on selected members (None = unlimited).
  """

  def __init__(
    self,
    rank_average: bool = False,
    alphas: tuple[float, ...] = _DEFAULT_ALPHAS,
    max_members: int | None = None,
  ) -> None:
    """Store search hyperparameters.

    Args:
        rank_average: Enable rank-based fusion.
        alphas: Grid of new-member mixture shares.
        max_members: Selection cap.

    Raises:
        ValueError: If any alpha lies outside (0, 1).
    """
    if not all(0 < alpha < 1 for alpha in alphas):
      raise ValueError('every alpha must lie in (0, 1)')
    self.rank_average = bool(rank_average)
    self.alphas = tuple(alphas)
    self.max_members = max_members
    self.selected_: list[str] = []
    self.weights_: dict[str, float] = {}
    self.best_score_: float = float('-inf')

  def _prepare(self, values: np.ndarray) -> np.ndarray:
    """Optionally rank-transform member probabilities per class.

    Args:
        values: Probability matrix ``(N, C)``.

    Returns:
        Rank matrix when rank_average is set, else the input values.
    """
    if not self.rank_average:
      return values
    from scipy.stats import (  # pylint: disable=import-outside-toplevel
      rankdata,
    )

    return np.stack(
      [rankdata(values[:, c]) for c in range(values.shape[1])], axis=1
    ).astype(np.float64)

  @staticmethod
  def _score(matrix: np.ndarray, gold_values: np.ndarray) -> float:
    """Macro-AUC of blended ranks/probabilities against gold labels.

    Args:
        matrix: Blended matrix ``(N, C)``.
        gold_values: Binary gold matrix ``(N, C)``.

    Returns:
        Mean AUC; NaN-skipping classes are ignored by the metric.
    """
    score, _, _ = macro_auc(gold_values > 0.5, matrix)
    return score

  def _blend(
    self,
    prepared: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
  ) -> np.ndarray:
    """Recompute the weighted mixture from member shares.

    Args:
        prepared: Prepared matrices keyed by member name.
        weights: Share per member; defaults to the fitted ``weights_``.

    Returns:
        Weighted average matrix ``(N, C)``.
    """
    active = weights if weights is not None else self.weights_
    total = sum(active.values()) or 1.0
    mixture = None
    for name, weight in active.items():
      contribution = prepared[name] * (weight / total)
      mixture = contribution if mixture is None else mixture + contribution
    return mixture

  def fit(
    self, oof_frames: dict[str, pd.DataFrame], gold: pd.DataFrame
  ) -> GreedyBlender:
    """Hill-climb members against the shared gold-OOF metric.

    Args:
        oof_frames: Member name -> OOF frame with StudyInstanceUID +
            TARGET columns.
        gold: Frame with StudyInstanceUID and binary TARGET columns;
            rows with missing labels are dropped before scoring.

    Returns:
        Self, with ``selected_``, ``weights_`` and ``best_score_`` set.

    Raises:
        ValueError: When no member or no fully labeled rows remain.
    """
    targets = [t for t in TARGETS if t in gold.columns]
    if not targets:
      raise ValueError('gold frame lacks all target columns')
    clean_gold = gold.dropna(subset=targets).reset_index(drop=True)

    aligned: dict[str, np.ndarray] = {}
    for name, frame in oof_frames.items():
      merged = clean_gold[['StudyInstanceUID']].merge(
        frame[['StudyInstanceUID', *TARGETS]],
        on='StudyInstanceUID',
        how='inner',
      )
      if len(merged) != len(clean_gold):
        raise ValueError(f'member {name} misses gold studies')
      aligned[name] = merged[list(TARGETS)].to_numpy(np.float64)

    gold_values = clean_gold[list(TARGETS)].to_numpy(np.float64)
    prepared = {name: self._prepare(values) for name, values in aligned.items()}
    start = max(prepared, key=lambda n: self._score(prepared[n], gold_values))
    self.selected_, self.weights_ = [start], {start: 1.0}
    self.best_score_ = self._score(prepared[start], gold_values)

    remaining = set(prepared) - {start}
    while remaining and (
      self.max_members is None or len(self.selected_) < self.max_members
    ):
      best_pick, best_alpha = None, 0.0
      best_gain, next_score = _MIN_IMPROVEMENT, self.best_score_
      for candidate in sorted(remaining):
        for alpha in self.alphas:
          trial_weights = {
            name: weight * (1.0 - alpha)
            for name, weight in self.weights_.items()
          }
          trial_weights[candidate] = alpha
          score = self._score(self._blend(prepared, trial_weights), gold_values)
          if score - self.best_score_ > best_gain:
            best_gain, next_score = score - self.best_score_, score
            best_pick, best_alpha = candidate, alpha
      if best_pick is None:
        break
      self.weights_ = {
        name: weight * (1.0 - best_alpha)
        for name, weight in self.weights_.items()
      }
      self.weights_[best_pick] = best_alpha
      self.selected_.append(best_pick)
      self.best_score_ = next_score
      remaining.discard(best_pick)
    return self
