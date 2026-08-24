#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Label fusion: gold > rule seeds > teacher probabilities.

Implements the priority cascade of BLUEPRINT section 3 producing the
canonical ``weak_labels.parquet``: per class a probability, a source tag
(``gold | rule | teacher | none``), and a trust ``weight`` consumed as
the sample weight in the distillation loss. Rule seeds survive only
when their measured precision on the gold subset clears
``rule_confidence_floor``; everything else defers to the teacher.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from knee.config_params.schema import TARGETS

_FALLBACK_WEIGHT = 0.05


class WeakLabelBuilder:
  """Fuses rule seed matrices, teacher OOF and gold into one table.

  Args:
      rule_confidence_floor: Minimum measured per-class rule precision.
      teacher_weight: Sample weight assigned to teacher-sourced labels.
      min_positive_prob: Below this teacher prob a class counts as a
          confident negative (still teacher-weighted).
  """

  def __init__(
    self,
    rule_confidence_floor: float = 0.9,
    teacher_weight: float = 1.0,
    min_positive_prob: float = 0.35,
  ) -> None:
    """Validate fusion parameters.

    Args:
        rule_confidence_floor: Precision gate in [0, 1].
        teacher_weight: Non-negative trust for teacher labels.
        min_positive_prob: Confident-negative threshold in [0, 1].

    Raises:
        ValueError: If thresholds fall outside their valid ranges.
    """
    if not 0 <= rule_confidence_floor <= 1:
      raise ValueError('rule_confidence_floor must lie in [0, 1]')
    if teacher_weight < 0:
      raise ValueError('teacher_weight must be non-negative')
    if not 0 <= min_positive_prob <= 1:
      raise ValueError('min_positive_prob must lie in [0, 1]')
    self.rule_floor = float(rule_confidence_floor)
    self.teacher_weight = float(teacher_weight)
    self.min_positive_prob = float(min_positive_prob)

  def build(
    self,
    train_csv: str,
    rule_probs: pd.DataFrame,
    rule_mask: pd.DataFrame,
    rule_precision: pd.Series | None = None,
    teacher_oof: pd.DataFrame | None = None,
  ) -> tuple[pd.DataFrame, dict]:
    """Fuse every signal source into the weak-label table.

    Args:
        train_csv: Study CSV with StudyInstanceUID + optional gold cols.
        rule_probs: Frame [StudyInstanceUID, *TARGETS] of seed probs.
        rule_mask: Same shape; 1 where the labeler made any claim.
        rule_precision: Per-target rule precision measured on the gold
            subset (NaN where unmeasurable); None disables rules.
        teacher_oof: Optional [StudyInstanceUID, *TARGETS] soft probs.

    Returns:
        Tuple ``(weak, stats)``. ``weak`` carries StudyInstanceUID,
        TARGETS probabilities, ``f'{target}_source'`` tags and a
        study-level ``weight`` column; ``stats`` summarizes sources.
    """
    studies = pd.read_csv(train_csv)
    base = studies[['StudyInstanceUID']]
    table = base.merge(rule_probs, on='StudyInstanceUID', how='left')
    claims = base.merge(rule_mask, on='StudyInstanceUID', how='left')
    n = len(base)
    n_targets = len(TARGETS)

    gold = np.full((n, n_targets), np.nan, np.float32)
    for c, target in enumerate(TARGETS):
      if target in studies.columns:
        gold[:, c] = (
          pd.to_numeric(studies[target], errors='coerce').to_numpy(np.float32)
        )
    teacher_values = np.full((n, n_targets), np.nan, np.float32)
    if teacher_oof is not None:
      merged = base.merge(teacher_oof, on='StudyInstanceUID', how='left')
      teacher_values = merged[list(TARGETS)].to_numpy(np.float32)

    precision = np.full(n_targets, np.nan, np.float32)
    if rule_precision is not None:
      for c, target in enumerate(TARGETS):
        if target in rule_precision.index:
          value = float(rule_precision[target])
          precision[c] = value if np.isfinite(value) else np.nan

    probs = np.full((n, n_targets), 0.5, np.float32)
    weights = np.full((n, n_targets), _FALLBACK_WEIGHT, np.float32)
    sources = np.empty((n, n_targets), dtype='<U8')
    sources[:] = 'none'

    is_gold = np.isfinite(gold)
    rule_claim = claims[list(TARGETS)].fillna(0).to_numpy(np.float32) > 0.5
    rule_positive = table[list(TARGETS)].fillna(0).to_numpy(np.float32) > 0.5
    gated = rule_claim & (np.isnan(precision) | (precision >= self.rule_floor))

    for c in range(n_targets):
      use_rule = gated[:, c] & ~is_gold[:, c]
      use_teacher = (
        np.isfinite(teacher_values[:, c])
        & ~is_gold[:, c] & ~use_rule
      )
      probs[is_gold[:, c], c] = np.clip(gold[is_gold[:, c], c], 0, 1)
      weights[is_gold[:, c], c] = 1.0
      sources[is_gold[:, c], c] = 'gold'

      probs[use_rule, c] = rule_positive[use_rule, c].astype(np.float32)
      floor = 0.75 if np.isnan(precision[c]) else float(precision[c])
      weights[use_rule, c] = floor
      sources[use_rule, c] = 'rule'

      probs[use_teacher, c] = teacher_values[use_teacher, c]
      weights[use_teacher, c] = self.teacher_weight
      sources[use_teacher, c] = 'teacher'

    weak = pd.DataFrame({'StudyInstanceUID': base['StudyInstanceUID']})
    stats: dict = {'n_studies': int(n), 'positives': {}, 'sources': {}}
    for c, target in enumerate(TARGETS):
      weak[target] = probs[:, c]
      weak[f'{target}_source'] = sources[:, c]
      stats['sources'][target] = (
        pd.Series(sources[:, c]).value_counts().astype(int).to_dict()
      )
      stats['positives'][target] = float((probs[:, c] >= 0.5).sum())
    weak['weight'] = weights.mean(axis=1).astype(np.float32)
    return weak.reset_index(drop=True), stats
