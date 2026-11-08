#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Purged cross-validation folds over frozen assignment.

Expanding-window folds built per fold by calling the single
assignment owner in ``data_split``. This module computes only
fold geometry and delegates ordering, filtering, and the
leakage check to that owner.
'''

import math

import pandas as pd

from forecast_ml_framework.pipelines.modeling.data_split import (
  _temporal_fold_assignment,
  assert_no_leakage,
)


def purged_cv_folds(
    df: pd.DataFrame, params: dict | None = None) -> list:
  '''Build expanding-window purged folds.

  Args:
    df: Frame with the time column.
    params: Dict with time_col (required), n_splits
      (int >= 1, default 3), test_size (float in (0, 1)
      or None, default 0.2), embargo (default 0), purge
      (default None, meaning embargo), window (only
      'expanding', default 'expanding'), and
      min_train_rows (int >= 1, default 1).

  Returns:
    List of fold dicts with fold_index, train_df,
    test_df, and fold_meta. The fold_meta embargo field
    holds the effective checked gap (the larger of
    embargo and purge) while purge is recorded apart.

  Raises:
    ValueError: On missing time_col, bad window,
      bad n_splits/test_size/min_train_rows, empty
      frame, helper boundary refusal, small train
      fold, or a leaking fold.
  '''
  if not isinstance(params, dict):
    params = {}
  time_col = params.get('time_col')
  if time_col is None:
    raise ValueError(
      'purged_cv_folds: time_col is required')
  window = params.get('window', 'expanding')
  if window != 'expanding':
    raise ValueError(
      'purged_cv_folds: window must be expanding')
  n_splits = params.get('n_splits', 3)
  if isinstance(n_splits, bool) or not isinstance(
      n_splits, int) or n_splits < 1:
    raise ValueError(
      'purged_cv_folds: n_splits must be an int >= 1')
  test_size = params.get('test_size', 0.2)
  if test_size is not None:
    if isinstance(test_size, bool) or not isinstance(
        test_size, (int, float)) or not 0 < test_size < 1:
      raise ValueError(
        'purged_cv_folds: test_size must be in (0, 1)')
  embargo_raw = params.get('embargo', 0)
  purge_raw = params.get('purge', None)
  min_train_rows = params.get('min_train_rows', 1)
  if isinstance(min_train_rows, bool) or not isinstance(
      min_train_rows, int) or min_train_rows < 1:
    raise ValueError(
      'purged_cv_folds: min_train_rows must be >= 1')
  if df is None or len(df) == 0:
    raise ValueError(
      'purged_cv_folds: df must be non-empty')
  total = len(df)
  if test_size is None:
    n_test_block = math.ceil(total / (n_splits + 1))
  else:
    n_test_block = math.ceil(test_size * total)
  test_meta = None
  if test_size is not None:
    test_meta = float(test_size)
  folds = []
  for fold_index in range(n_splits):
    test_start = total - (n_splits - fold_index) * n_test_block
    result = _temporal_fold_assignment(
      df, time_col=time_col, test_start=test_start,
      n_test_block=n_test_block, embargo=embargo_raw,
      purge=purge_raw)
    fold_meta = {
      'sampling_method': 'purged_cv',
      'fold_index': fold_index,
      'n_splits': n_splits,
      'time_col': time_col,
      'test_size': test_meta,
      'embargo': result['embargo'],
      'purge': result['purge'],
      'n_train': result['n_train'],
      'n_test': result['n_test'],
      'n_embargoed': result['n_embargoed'],
      'n_purged': result['n_purged'],
      'max_time_train': result['max_time_train'],
      'min_time_test': result['min_time_test'],
      'leakage_checked': True,
      'window': 'expanding',
    }
    n_train = fold_meta['n_train']
    if n_train < min_train_rows:
      raise ValueError(
        f'purged_cv_folds: fold {fold_index} n_train '
        f'{n_train} below min_train_rows '
        f'{min_train_rows}')
    assert_no_leakage(fold_meta)
    folds.append({
      'fold_index': fold_index,
      'train_df': result['train_df'],
      'test_df': result['test_df'],
      'fold_meta': fold_meta,
    })
  return folds


def fold_leakage_check(folds: list[dict]) -> None:
  '''Check every fold against the frozen leakage rule.

  Args:
    folds: List of fold dicts holding fold_meta.

  Returns:
    None when every fold passes.

  Raises:
    ValueError: If folds is empty or malformed, or a
      fold fails the leakage check naming its index.
  '''
  if not isinstance(folds, list) or len(folds) == 0:
    raise ValueError(
      'fold_leakage_check: folds must be non-empty')
  for pos, fold in enumerate(folds):
    label = pos
    if isinstance(fold, dict):
      label = fold.get('fold_index', pos)
    fold_meta = None
    if isinstance(fold, dict):
      fold_meta = fold.get('fold_meta')
    if not isinstance(fold_meta, dict):
      raise ValueError(
        'fold_leakage_check: fold '
        f'{label} missing fold_meta')
    try:
      assert_no_leakage(fold_meta)
    except ValueError as exc:
      raise ValueError(
        'fold_leakage_check: fold '
        f'{label} leaks ({exc})') from exc
  return None
