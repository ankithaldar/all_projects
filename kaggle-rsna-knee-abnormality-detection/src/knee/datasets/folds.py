#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen multi-label stratified fold assignment.

One fold CSV must be shared by every experiment and ensemble member;
this module is its single producer (BLUEPRINT section 6). Iterative
stratification preserves all 12 label marginals per fold, which plain
KFold cannot for this prevalence profile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from knee.config_params.schema import TARGETS


def _label_columns(df: pd.DataFrame) -> list[str]:
  """Pick the label columns present in a train/folds frame.

  Args:
      df: Frame with StudyInstanceUID and zero or more target columns.

  Returns:
      Ordered list of target columns found (submission order).
  """
  return [t for t in TARGETS if t in df.columns]


def make_iterative_multilabel_folds(
  train_csv: str, n_folds: int = 5, seed: int = 42
) -> pd.DataFrame:
  """Assign every study to one fold via iterative stratification.

  Missing labels are treated as negatives for stratification purposes
  only; the original NaN pattern is preserved in the returned frame.

  Args:
      train_csv: Study-level labels CSV with StudyInstanceUID.
      n_folds: Number of folds to create.
      seed: RNG seed; identical inputs yield byte-identical folds.

  Returns:
      The input frame plus an integer ``fold`` column in [0, n_folds).
  """
  # lazy: keeps this dataset helper importable without iterstrat
  from iterstrat.ml_stratifiers import (  # pylint: disable=import-outside-toplevel
    MultilabelStratifiedKFold,
  )

  df = pd.read_csv(train_csv)
  cols = _label_columns(df)
  y = df[cols].fillna(0).to_numpy(dtype=np.int64) if cols else None
  splitter = MultilabelStratifiedKFold(
    n_splits=n_folds, shuffle=True, random_state=seed
  )
  df['fold'] = -1
  dummy = pd.DataFrame({'x': range(len(df))})
  for k, (_, val_idx) in enumerate(splitter.split(dummy, y)):
    df.loc[val_idx, 'fold'] = k
  if (df['fold'] < 0).any():  # pragma: no cover - splitter contract
    raise RuntimeError('iterative stratification left rows unassigned')
  return df


def fold_summary(folds_df: pd.DataFrame) -> pd.DataFrame:
  """Per-fold study counts and positive counts per target class.

  Args:
      folds_df: Frame produced by :func:`make_iterative_multilabel_folds`.

  Returns:
      Summary indexed by fold with ``n_studies`` plus one positive-count
      column per present target.
  """
  cols = _label_columns(folds_df)
  grouped = folds_df.groupby('fold')
  summary = grouped.size().rename('n_studies').to_frame()
  for col in cols:
    summary[col] = grouped[col].apply(lambda s: float((s.fillna(0) > 0).sum()))
  return summary
