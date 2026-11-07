#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Data splitting and sampling nodes.'''

import warnings

import numpy as np

from forecast_ml_framework.pipelines.modeling.ntile import (
  normalize_task_kind,
)


def _ordered_split(x, y, test_size):
  '''Split rows in order: first rows train, last rows test.

  Args:
    x: Feature rows in split order (DataFrame or array).
    y: Target rows in split order.
    test_size: Fraction of rows for test, in (0, 1).

  Returns:
    Tuple (x_train, x_test, y_train, y_test) as positional
    slices preserving the input order.
  '''
  n = len(x)
  n_test = max(1, int(round(n * test_size)))
  if n_test >= n:
    n_test = n - 1
  cut = n - n_test
  return x.iloc[:cut], x.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def _resolve_split_options(params, sampling):
  '''Resolve shuffle / time_col / method options for split_data.

  Args:
    params: modeling_params dict.
    sampling: sampling_params dict (may be empty).

  Returns:
    Tuple (method, shuffle, time_col) where method is one of
    'stratified', 'random' or 'time'. An explicit
    method='time' (or a time_col with shuffle=False) selects
    the ordered time split; shuffle defaults to True except
    when a time_col is given without an explicit shuffle.
  '''
  method = str(sampling.get('method', params.get('method',
                                                  'stratified')))
  method = method.strip().lower()
  time_col = sampling.get('time_col', params.get('time_col'))
  shuffle = sampling.get('shuffle', params.get('shuffle'))
  if method in ('time', 'ordered', 'temporal'):
    return 'time', False, time_col
  if time_col is not None and shuffle is False:
    return 'time', False, time_col
  if method == 'random':
    return 'random', True if shuffle is None else bool(shuffle), time_col
  if shuffle is False and time_col is None:
    return 'random', False, time_col
  return 'stratified', True if shuffle is None else bool(shuffle), time_col


def _resolve_id_cols(params, sampling):
  '''Resolve identifier columns excluded from default features.

  Mirrors the presplit path (ML-24c): the ``id_cols``
  param when given, else ``id_col``, else a
  variables-style id list passed via ``params``
  (``variables`` dict with ``id_cols``/``id_col``/``id``),
  else ``['id']``. Explicit sources present together
  are unioned; a string counts as a single column.
  Only consulted when ``feature_cols`` is None, so an
  explicit feature list is always honoured as given.

  Args:
    params: Modeling-params dict.
    sampling: Sampling-params dict (may be empty).

  Returns:
    List of identifier column names.
  '''
  found = []

  def _collect(raw):
    if raw is None:
      return
    if isinstance(raw, str):
      found.append(raw)
    else:
      found.extend(str(c) for c in raw)

  _collect(params.get('id_cols'))
  _collect(params.get('id_col'))
  variables = params.get('variables')
  if isinstance(variables, dict):
    _collect(variables.get('id_cols'))
    _collect(variables.get('id_col'))
    _collect(variables.get('id'))
  _collect(sampling.get('id_cols'))
  _collect(sampling.get('id_col'))
  seen = []
  for col in found:
    if col not in seen:
      seen.append(col)
  return seen or ['id']


def split_data(df, params):
  '''Split a frame into train/test features and target.

  When ``feature_cols`` is None, identifier columns
  are excluded from the default features (ML-24c):
  see :func:`_resolve_id_cols` (``id_cols``/``id_col``/
  variables id list, else ``['id']``). An explicit
  ``feature_cols`` list is honoured exactly as given.

  Args:
    df: Input DataFrame with a target column.
    params: modeling_params dict. Keys used: target_col,
      feature_cols, id_cols, id_col, variables, task_type,
      test_size, random_state, time_col, shuffle, and
      sampling_params {method: stratified|random|time,
      test_size, stratify_col, time_col, shuffle}.

  Returns:
    Tuple (x_train, x_test, y_train, y_test, meta). Meta
    ``sampling_method`` is the method actually used:
    'stratified' only when stratification succeeded, else
    'random' (with a loud warning) or 'time' for the
    ordered split. Time splits order by ``time_col`` when
    given (otherwise by row order), so every train index
    precedes every test index. ``shuffle=False`` with a
    random method preserves row order (first rows train).
    Stratification is only attempted for classification.
  '''
  from sklearn.model_selection import train_test_split
  params = params or {}
  sampling = params.get('sampling_params', {}) or {}
  if df is None or len(df) == 0:
    raise ValueError('split_data requires a non-empty dataframe')
  target = params.get('target_col', 'target')
  if target not in df.columns:
    raise ValueError(f'target column not found: {target}')
  test_size = sampling.get('test_size',
                           params.get('test_size', 0.2))
  if isinstance(test_size, bool) or not isinstance(
      test_size, (int, float)) or not 0 < test_size < 1:
    raise ValueError('test_size must be in (0, 1)')
  features = params.get('feature_cols')
  if features is None:
    id_cols = _resolve_id_cols(params, sampling)
    features = [c for c in df.columns
                if c != target and c not in id_cols]
  seed = params.get('random_state', sampling.get('random_state', 42))
  if seed is not None and (isinstance(seed, bool)
                           or not isinstance(seed, (int, np.integer))
                           or int(seed) < 0):
    raise ValueError(
      f'random_state must be a non-negative int or None, got {seed!r}')
  seed = None if seed is None else int(seed)
  method, shuffle, time_col = _resolve_split_options(params, sampling)
  x = df[features]
  y = df[target]
  if method == 'time':
    if time_col is not None:
      if time_col not in df.columns:
        raise ValueError(f'time column not found: {time_col}')
      order = df[time_col].sort_values(kind='stable').index
      x = x.loc[order]
      y = y.loc[order]
    x_tr, x_te, y_tr, y_te = _ordered_split(x, y, test_size)
    actual = 'time'
  elif method == 'random' and not shuffle:
    x_tr, x_te, y_tr, y_te = _ordered_split(x, y, test_size)
    actual = 'random'
  else:
    stratify = None
    if method == 'stratified':
      stratify_col = sampling.get('stratify_col', target)
      is_classification = (
        normalize_task_kind(params.get('task_type', 'classification'))
        == 'classification')
      # O5: explicit regression stratify on target is dropped.
      if (not is_classification and 'stratify_col' in sampling
          and stratify_col == target):
        stratify = None
      else:
        stratify = df[stratify_col] if (
            is_classification or 'stratify_col' in sampling) else None
      if stratify is not None and stratify.nunique() < 2:
        stratify = None
    try:
      x_tr, x_te, y_tr, y_te = train_test_split(
          x, y, test_size=test_size, random_state=seed,
          stratify=stratify, shuffle=shuffle)
      # Residuals 1-2: actual is random whenever stratify
      # is None after guards (regression default, dropped
      # or single-class nulled), never a false stratified.
      actual = 'random' if stratify is None else method
    except ValueError:
      warnings.warn(
          'stratified split failed (a class has too few members); '
          'falling back to a random unstratified split',
          UserWarning, stacklevel=2)
      x_tr, x_te, y_tr, y_te = train_test_split(
          x, y, test_size=test_size, random_state=seed,
          shuffle=shuffle)
      actual = 'random'
  meta = {'task_type': params.get('task_type', 'classification'),
          'test_size': test_size, 'random_state': seed,
          'sampling_method': actual,
          'requested_method': method,
          'shuffle': shuffle, 'time_col': time_col,
          'n_train': int(len(y_tr)), 'n_test': int(len(y_te))}
  return x_tr, x_te, np.asarray(y_tr), np.asarray(y_te), meta
