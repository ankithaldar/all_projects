#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frequency-aware study sampling for imbalanced multi-target training.

The 12 targets are heavily skewed (EDA: fracture/contusion/Baker's/
synovitis are rare), while the competition metric is the MACRO-AUC -
every class contributes equally. Uniform study sampling therefore gives
rare positives too few gradient updates.

This module converts per-target prevalence into per-study sampling
weights:

1. prevalence of every target over UNMASKED entries (``-1`` uncertain
   labels never count in either numerator or denominator);
2. per-target weight ``prevalence ** -tempering`` (``tempering: 0.0``
   reproduces uniform sampling, ``1.0`` full inverse frequency),
   normalized to mean 1.0 so the scale stays comparable across knobs;
3. per-study weight = ``aggregation`` (max or mean) of the weights of
   the targets the study is POSITIVE for; studies with no positive
   target (healthy or fully unknown) receive ``baseline``.

The resulting sampler keeps epoch size (``num_samples = len(dataset)``
with replacement) so step counts, schedules, and resume math are
unchanged; only the sampling distribution tilts toward studies carrying
rare positives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

UNKNOWN = -1.0
POSITIVE = 1.0
AGGREGATIONS = ('max', 'mean')


def positive_prevalences(label_matrix: np.ndarray) -> np.ndarray:
  """Compute per-target prevalence over unmasked entries.

  Args:
      label_matrix: ``(n_studies, n_targets)`` array with values in
          ``{-1, 0, 1}`` (NaN counts as unknown).

  Returns:
      ``(n_targets,)`` float array of fractions in [0, 1]; targets with
      no unmasked entries yield 0.0 (and are excluded downstream).
  """
  matrix = np.nan_to_num(
    np.asarray(label_matrix, dtype=np.float64), nan=UNKNOWN
  )
  unmasked = matrix != UNKNOWN
  positives = (matrix == POSITIVE) & unmasked
  counts = positives.sum(axis=0)
  totals = unmasked.sum(axis=0)
  with np.errstate(divide='ignore', invalid='ignore'):
    return np.where(totals > 0, counts / np.maximum(totals, 1), 0.0)


def study_weights(
  label_matrix: np.ndarray,
  tempering: float = 0.5,
  aggregation: str = 'max',
  baseline: float = 1.0,
) -> np.ndarray:
  """Compute one sampling weight per study.

  Args:
      label_matrix: ``(n_studies, n_targets)`` array with values in
          ``{-1, 0, 1}`` (NaN counts as unknown).
      tempering: Skew exponent on inverse prevalence; 0.0 disables
          weighting, 1.0 is full inverse-frequency.
      aggregation: How to combine the per-target weights of a study's
          positive targets: ``max`` (sharper rare-class boost) or
          ``mean``.
      baseline: Weight for studies with no positive target at all.

  Returns:
      ``(n_studies,)`` float64 array of positive sampling weights.

  Raises:
      ValueError: If ``aggregation`` is unknown or ``tempering`` is
          negative.
  """
  if aggregation not in AGGREGATIONS:
    raise ValueError(
      f'aggregation must be one of {AGGREGATIONS}: {aggregation!r}'
    )
  if tempering < 0:
    raise ValueError(f'tempering must be >= 0: {tempering!r}')
  matrix = np.nan_to_num(
    np.asarray(label_matrix, dtype=np.float64), nan=UNKNOWN
  )
  prevalence = positive_prevalences(matrix)
  target_weight = np.ones_like(prevalence)
  known = prevalence > 0
  if tempering > 0 and known.any():
    raw = np.power(prevalence[known], -tempering)
    target_weight[known] = raw / raw.mean()
  positive = matrix == POSITIVE
  if aggregation == 'max':
    # -inf sentinel avoids np.nanmax RuntimeWarnings on rows without
    # any positive target; the isfinite mask restores the baseline.
    combined = np.where(positive, target_weight, -np.inf)
    raw = combined.max(axis=1)
    weights = np.where(np.isfinite(raw), raw, baseline)
  else:
    counts = positive.sum(axis=1)
    sums = np.where(positive, target_weight, 0.0).sum(axis=1)
    weights = np.where(counts > 0, sums / np.maximum(counts, 1), baseline)
  return np.maximum(weights, np.finfo(np.float64).tiny)


class StudyWeightedRandomSampler(WeightedRandomSampler):
  """Sample studies proportionally to their rare-positive content.

  Weights derive from the labels frame aligned to the dataset's
  ``study_ids``; studies absent from the frame (or fully unknown) fall
  back to ``baseline``. ``num_samples`` defaults to ``len(study_ids)``
  with replacement, preserving epoch size and step-based schedules.
  """

  def __init__(
    self,
    study_ids: list[str],
    label_frame: pd.DataFrame,
    target_columns: list[str],
    tempering: float = 0.5,
    aggregation: str = 'max',
    baseline: float = 1.0,
    num_samples: int | None = None,
    replacement: bool = True,
  ) -> None:
    """Compute aligned weights and hand them to the torch sampler.

    Args:
        study_ids: Dataset order; weight[i] applies to study_ids[i].
        label_frame: Labels frame with StudyInstanceUID + target cols.
        target_columns: Canonical 12-target order.
        tempering: Inverse-prevalence exponent (0 = uniform).
        aggregation: ``max`` or ``mean`` over positive targets.
        baseline: Weight for studies without any positive target.
        num_samples: Draws per epoch; None keeps ``len(study_ids)``.
        replacement: Draw with replacement (required for upweighting).
    """
    lookup = label_frame.set_index('StudyInstanceUID').reindex(study_ids)
    matrix = (
      lookup[target_columns]
      .to_numpy(dtype=np.float64)
      .reshape(len(study_ids), len(target_columns))
    )
    weights = study_weights(matrix, tempering, aggregation, baseline)
    super().__init__(
      torch.as_tensor(weights, dtype=torch.double),
      num_samples=int(num_samples or len(study_ids)),
      replacement=replacement,
    )
