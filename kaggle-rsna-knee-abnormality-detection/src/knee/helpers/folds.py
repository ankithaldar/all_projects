#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cross-validation fold construction.

Builds a ``folds.csv`` mapping StudyInstanceUID -> fold id using any
scikit-learn splitter expressible in ``configs/folds.yaml``
(class_path/init_params convention). Grouping defaults to the study id and
upgrades automatically to PatientID whenever the header index reveals
multi-study patients (EDA sampled 1000/1000 unique, but the full scan in
notebook 02 is authoritative).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GROUP_CANDIDATES = ('PatientID', 'StudyInstanceUID')


def build_strata(
  labels: pd.DataFrame,
  rare_targets: list[str],
  anchor_targets: list[str],
) -> pd.Series:
  """Compose a stratification key from rare-target buckets plus anchors.

  The key concatenates binary flags for each configured target; because
  macro-AUC weights every class equally, keeping the four rarest targets
  represented per fold matters more than balancing frequent ones.

  Args:
      labels: Frame with one row per study and target columns.
      rare_targets: Targets bucketed individually into the stratum key.
      anchor_targets: Extra high-signal targets appended to the key.

  Returns:
      String series of composite strata, aligned to ``labels`` index.
  """
  parts = []
  for column in [*rare_targets, *anchor_targets]:
    if column not in labels.columns:
      raise KeyError(f'Stratify target missing from labels frame: {column}')
    parts.append(labels[column].fillna(0).astype(int).astype(str))
  return _join_parts(parts, labels.index)


def _join_parts(parts: list[pd.Series], index: pd.Index) -> pd.Series:
  """Join per-column stratum fragments into one series.

  Args:
      parts: Fragment series sharing a common index.
      index: Target index for the result.

  Returns:
      Single string series of joined strata.
  """
  joined = parts[0].astype(str)
  for fragment in parts[1:]:
    joined = joined + '-' + fragment.astype(str)
  return pd.Series(joined.values, index=index)


def resolve_group_column(labels: pd.DataFrame) -> tuple[str, pd.Series]:
  """Pick the strongest available grouping column.

  Args:
      labels: Frame potentially carrying PatientID alongside StudyInstanceUID.

  Returns:
      Tuple of (column name used, group values).

  Raises:
      KeyError: If no known grouping column exists.
  """
  for candidate in GROUP_CANDIDATES:
    if candidate in labels.columns:
      return candidate, labels[candidate].astype(str)
  raise KeyError(f'None of {GROUP_CANDIDATES} found in labels frame')


def make_folds(
  labels: pd.DataFrame,
  splitter,
  rare_targets: list[str],
  anchor_targets: list[str],
) -> pd.Series:
  """Assign fold ids with grouped stratification.

  Args:
      labels: One-row-per-study frame including grouping and target columns.
      splitter: Instantiated scikit-learn CV object exposing ``split``.
      rare_targets: Rare-target columns prioritized by the strata builder.
      anchor_targets: Anchor columns appended to strata.

  Returns:
      Integer series of fold ids indexed like ``labels``.

  Raises:
      ValueError: If the splitter yields fewer folds than requested rows allow.
  """
  _, groups = resolve_group_column(labels)
  strata = build_strata(labels, rare_targets, anchor_targets)
  fold_ids = np.full(len(labels), -1, dtype=int)
  dummy_x = np.zeros((len(labels), 1))
  for fold_id, (_, valid_idx) in enumerate(
    splitter.split(dummy_x, y=strata, groups=groups)
  ):
    if fold_id >= getattr(splitter, 'n_splits', len(fold_ids)):
      break
    fold_ids[valid_idx] = fold_id
  if (fold_ids < 0).any():
    raise ValueError('Splitter left unlabeled rows; check groups and strata')
  return pd.Series(fold_ids, index=labels.index, name='fold')
