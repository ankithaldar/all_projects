#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Walk-forward backtesting over purged folds.

The harness consumes purged folds read-only, refits a caller
injected model for every fold, and scores predictions on the
fold test frame. It owns no fold geometry and performs no
ordering of its own.
'''

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecast_ml_framework.pipelines.modeling.purged_cv import (
  fold_leakage_check,
  purged_cv_folds,
)
from sklearn.metrics import accuracy_score, mean_absolute_error


@dataclass
class BacktestResult:
  '''Result of a walk-forward backtest run.

  Attributes:
    per_fold: List of dicts with fold_index, n_train,
      n_test, score, and fold_meta.
    aggregate: Mean of the per-fold scores as float.
    n_folds: Number of scored folds.
    task_type: Task type used for default scoring.
  '''
  per_fold: list
  aggregate: float
  n_folds: int
  task_type: str


def _neg_mae(y_true, y_pred):
  '''Return the negated mean absolute error.

  Args:
    y_true: True labels as array-like.
    y_pred: Predicted labels as array-like.

  Returns:
    Negative MAE as float so higher is better.
  '''
  return -float(mean_absolute_error(y_true, y_pred))


def default_metric(task_type: str):
  '''Return the documented default scoring callable.

  Args:
    task_type: Task type name as a string.

  Returns:
    Callable(y_true, y_pred) -> float where classification
    maps to accuracy and regression maps to negated MAE.

  Raises:
    ValueError: On an unknown task type.
  '''
  name = str(task_type).strip().lower()
  if name == 'classification':
    return accuracy_score
  if name == 'regression':
    return _neg_mae
  raise ValueError(f'default_metric: unknown task {task_type!r}')


def run_backtest(
    df: pd.DataFrame, model_fn, metric_fn=None, *,
    params: dict | None = None) -> BacktestResult:
  '''Walk-forward backtest over purged folds.

  Args:
    df: Frame with the time column and target column.
    model_fn: Callable that takes a train frame and returns
      a fitted estimator with a predict method. Called fresh
      for every fold with that fold train frame only.
    metric_fn: Callable(y_true, y_pred) -> float scoring
      predictions. When None, the task_type default is used.
    params: Fold params forwarded to the fold builder, plus
      target_col (default 'target'), task_type (default
      'classification'), and feature_cols (default None,
      meaning all columns except target and time).

  Returns:
    BacktestResult with per-fold scores and mean aggregate.

  Raises:
    ValueError: On an empty frame, a missing target or time
      column, a non-callable model_fn, a None estimator or
      prediction, a bad prediction shape, an empty fold
      list, or a leaking fold.
  '''
  if df is None or len(df) == 0:
    raise ValueError('run_backtest: df must be non-empty')
  if not callable(model_fn):
    raise ValueError('run_backtest: model_fn must be callable')
  cfg = params if isinstance(params, dict) else {}
  target_col = cfg.get('target_col', 'target')
  task_type = cfg.get('task_type', 'classification')
  if target_col not in df.columns:
    raise ValueError(
      f'run_backtest: target_col {target_col!r} missing')
  time_col = cfg.get('time_col')
  if time_col is None or time_col not in df.columns:
    raise ValueError('run_backtest: time_col is required')
  feature_cols = cfg.get('feature_cols')
  if feature_cols is None:
    feature_cols = [
      c for c in df.columns
      if c != target_col and c != time_col]
  score_metric = metric_fn
  if score_metric is None:
    score_metric = default_metric(task_type)
  folds = purged_cv_folds(df, cfg)
  if not folds:
    raise ValueError('run_backtest: folds must be non-empty')
  fold_leakage_check(folds)
  per_fold = []
  for fold in folds:
    train_df = fold['train_df']
    test_df = fold['test_df']
    fold_index = fold['fold_index']
    estimator = model_fn(train_df)
    if estimator is None:
      raise ValueError(
        f'run_backtest: fold {fold_index} model_fn '
        'returned None')
    predict = getattr(estimator, 'predict', None)
    if not callable(predict):
      raise ValueError(
        f'run_backtest: fold {fold_index} missing predict')
    preds = predict(test_df[feature_cols])
    if preds is None:
      raise ValueError(
        f'run_backtest: fold {fold_index} predict '
        'returned None')
    pred_arr = np.asarray(preds)
    if pred_arr.ndim != 1 or len(pred_arr) != len(test_df):
      raise ValueError(
        f'run_backtest: fold {fold_index} bad shape')
    y_true = test_df[target_col].to_numpy()
    score = float(score_metric(y_true, pred_arr))
    per_fold.append({
      'fold_index': fold_index,
      'n_train': len(train_df),
      'n_test': len(test_df),
      'score': score,
      'fold_meta': fold['fold_meta'],
    })
  total = sum(entry['score'] for entry in per_fold)
  aggregate = float(total / len(per_fold))
  return BacktestResult(
    per_fold=per_fold,
    aggregate=aggregate,
    n_folds=len(per_fold),
    task_type=task_type)
