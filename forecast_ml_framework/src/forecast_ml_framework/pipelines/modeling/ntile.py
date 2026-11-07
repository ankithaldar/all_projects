#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''This framework metrics parity: ntile eval-metric helpers.

Three small helpers shared by modeling evaluation so that a
requested eval metric maps deterministically onto a rank tile:

- ``parse_eval_metric``: metric name + task -> ``(metric, kind,
  n)``. Classification defaults to ``lift``/``decile``/10,
  regression to ``rmse``/``overall``/0; an unknown metric name
  warns and falls back to the task default.
- ``assign_ntile``: score -> 1-based ``pd.qcut`` tile ids
  (lowest scores in tile 1); ``overall`` maps to all-zero.
- ``select_ntile_row``: DataFrame row for a tile, falling back
  to the row with the maximum tile value.
'''

import re
import warnings

import numpy as np
import pandas as pd

from forecast_ml_framework.pipelines.modeling.metrics import (
  CLASSIFICATION_METRICS,
  REGRESSION_METRICS,
)

_KIND_N = {'decile': 10, 'quintile': 5, 'quartile': 4,
           'centile': 100}
_CLASS_ALIASES = {'lift', 'gain', 'cumulative_gain',
                  'cumulative_gain_top_decile'}
_REG_ALIASES = {'root_mean_squared_error'}
_CLS_SELECTABLE = {
    'LIFT': 'lift', 'PRECISION': 'precision', 'RECALL': 'recall',
    'ACCURACY': 'accuracy', 'F1': 'f1', 'AUC': 'auc',
    'BRIER_SCORE': 'brier_score', 'BRIER': 'brier_score',
}
_REG_SELECTABLE = {
    'MAE': 'mae', 'MAPE': 'mape', 'MSE': 'mse', 'MSLE': 'msle',
    'RMSE': 'rmse', 'RMSLE': 'rmsle', 'ADJ_R2': 'adj_r2',
    'R2': 'r2',
}
# Forecast registry names aliasing the canonical selectable
# metric (reverse of evaluate._METRIC_ALIASES, plus aliases).
_FORECAST_ALIASES = {
    'lift_top_decile': 'lift',
    'cumulative_gain_top_decile': 'lift',
    'cum_gain_top_decile': 'lift',
    'roc_auc': 'auc',
    'brier': 'brier_score',
    'precision_weighted': 'precision',
    'recall_weighted': 'recall',
    'f1_weighted': 'f1',
    'root_mean_squared_error': 'rmse',
}
_PARITY_RE = re.compile(
    r'^(.+?)_(DECILE|CENTILE|QUINTILE|QUARTILE)_(\d+)$')
_KIND_SUFFIX_RE = re.compile(
    r'^(.+?)_(DECILE|CENTILE|QUINTILE|QUARTILE)$')


def normalize_task_kind(task):
  '''Normalize a task name to 'classification' or 'regression'.

  Args:
    task: Task name (case-insensitive).

  Returns:
    'classification' or 'regression'; unknown tasks warn and
    default to 'classification'.
  '''
  name = str(task or '').strip().lower()
  if name.startswith('reg'):
    return 'regression'
  if name.startswith('class') or name in ('cls', 'binary'):
    return 'classification'
  warnings.warn(f'unknown task {task!r}: defaulting to '
                f'classification', stacklevel=3)
  return 'classification'


def _task_kind(task):
  '''Normalize a task name (alias of ``normalize_task_kind``).'''
  return normalize_task_kind(task)


def parse_eval_metric(metric_name, task):
  '''Parse an eval metric name into (metric, kind, n).

  Args:
    metric_name: Requested metric name, or None/empty for the
      task default (no warning).
    task: 'classification' or 'regression'.

  Returns:
    Tuple of (metric, kind, n): classification defaults to
    ('lift', 'decile', 10), regression to ('rmse', 'overall', 0).
    Parity grammar (``<M>_DECILE_<n>`` / ``<M>_CENTILE_<n>`` with
    ``M`` a selectable metric for the task) is supported first
    and canonicalises to the selectable metric name; forecast
    names (e.g. ``lift_top_decile``) alias to the same
    canonical metric. Names containing
    'decile'/'quintile'/'quartile' select that kind and count;
    unknown names warn and return the task default.
  '''
  task_kind = _task_kind(task)
  default = (('lift', 'decile', 10) if task_kind == 'classification'
             else ('rmse', 'overall', 0))
  if metric_name is None or not str(metric_name).strip():
    return default
  metric = str(metric_name).strip().lower()
  upper = metric.upper()
  selectable = (_CLS_SELECTABLE if task_kind == 'classification'
                else _REG_SELECTABLE)
  # Forecast aliases first: bare, parity (<alias>_<KIND>_<n>),
  # and kind-suffix (<alias>_<KIND>) forms map to the canonical
  # metric with kind/count preserved.
  if metric in _FORECAST_ALIASES:
    return _FORECAST_ALIASES[metric], default[1], default[2]
  match = _PARITY_RE.match(upper)
  if match:
    name, kind, count = match.groups()
    base = name.lower()
    if base in _FORECAST_ALIASES:
      return _FORECAST_ALIASES[base], kind.lower(), int(count)
  suffix = _KIND_SUFFIX_RE.match(upper)
  if suffix:
    name, kind = suffix.groups()
    base = name.lower()
    if base in _FORECAST_ALIASES:
      return (_FORECAST_ALIASES[base], kind.lower(),
              _KIND_N[kind.lower()])
    if name in selectable:
      return (selectable[name], kind.lower(),
              _KIND_N[kind.lower()])
  # Parity grammar: <M>_DECILE_<n> / <M>_CENTILE_<n> etc.
  if match:
    name, kind, count = match.groups()
    if name in selectable:
      return selectable[name], kind.lower(), int(count)
    warnings.warn(f'unknown eval metric {metric_name!r} for '
                  f'{task_kind}: using default {default[0]!r}',
                  stacklevel=2)
    return default
  # Bare parity/selectable names use the task default kind.
  if upper in selectable:
    return selectable[upper], default[1], default[2]
  known = (CLASSIFICATION_METRICS if task_kind == 'classification'
           else REGRESSION_METRICS)
  aliases = (_CLASS_ALIASES if task_kind == 'classification'
             else _REG_ALIASES)
  if metric not in known and metric not in aliases:
    warnings.warn(f'unknown eval metric {metric_name!r} for '
                  f'{task_kind}: using default {default[0]!r}',
                  stacklevel=2)
    return default
  for kind, count in _KIND_N.items():
    if kind in metric:
      return metric, kind, count
  if 'overall' in metric or 'global' in metric:
    return metric, 'overall', 0
  return metric, default[1], default[2]


def assign_ntile(score, kind='decile', n=10):
  '''Assign 1-based qcut tile ids to scores.

  Args:
    score: One-dimensional scores (array-like or Series).
    kind: 'decile', 'quintile', 'quartile', or 'overall'.
    n: Number of tiles; falls back to the kind default when
      falsy (except for 'overall').

  Returns:
    Integer array of tile ids in [1, n] (lowest scores in tile
    1); 'overall' returns all zeros, constant scores all
    tile 1, and NaN scores tile 1 (never tile 0). This is the
    single 1-based tiling used everywhere (evaluation,
    lift tables and scoring). When ``n`` exceeds the number
    of rows, the tile count degrades to at most one tile per
    row and a ``UserWarning`` is issued: the maximum tile
    may then be below ``n`` (e.g. a centile request on 50
    rows yields at most 50 tiles).
  '''
  values = pd.Series(np.asarray(score, dtype=float))
  if len(values) == 0:
    return np.zeros(0, dtype=int)
  kind_name = str(kind or 'overall').strip().lower()
  if kind_name in ('overall', 'global', 'none'):
    return np.zeros(len(values), dtype=int)
  q = int(n) if n else _KIND_N.get(kind_name, 10)
  if q <= 0:
    return np.zeros(len(values), dtype=int)
  if q > len(values):
    warnings.warn(
        f'requested {q} tiles for only {len(values)} rows: tile '
        'count degrades to at most one tile per row (max tile '
        f'<= {len(values)})',
        UserWarning, stacklevel=2)
  try:
    labels = pd.qcut(values, q=min(q, len(values)),
                     labels=False, duplicates='drop')
  except (ValueError, TypeError):
    return np.ones(len(values), dtype=int)
  if labels.isna().all():
    # Constant scores form no bins: everything is tile 1.
    return np.ones(len(values), dtype=int)
  return (labels.fillna(0) + 1).astype(int).to_numpy()


def select_ntile_row(df, ntile_col, ntile):
  '''Return the row for a tile, falling back to the max tile.

  Args:
    df: DataFrame with a numeric tile column.
    ntile_col: Name of the tile column in ``df``.
    ntile: Requested tile value, or None for the max row.

  Returns:
    The matching row as a Series; if no row matches, the row
    with the maximum ``ntile_col`` value.

  Raises:
    ValueError: If ``df`` is empty or lacks ``ntile_col``.
  '''
  if df is None or len(df) == 0:
    raise ValueError('select_ntile_row: empty dataframe')
  if ntile_col not in df.columns:
    raise ValueError(
        f'select_ntile_row: missing column {ntile_col!r}')
  if ntile is not None:
    match = df[df[ntile_col] == ntile]
    if not match.empty:
      return match.iloc[0]
  return df.loc[df[ntile_col].idxmax()]
