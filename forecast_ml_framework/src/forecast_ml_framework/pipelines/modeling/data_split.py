#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Data splitting and sampling nodes.'''

import datetime
import math
import re
import warnings

import numpy as np
import pandas as pd

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
    '''Append raw id column name(s) to ``found``.

    Args:
      raw: Identifier column name, iterable of names,
        or None (a no-op). Strings count as a single
        column, never as an iterable of characters.

    Returns:
      None. ``found`` is updated in place.
    '''
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


def _value_is_aware(value):
  '''Return True if aware, False if naive, None if neutral.

  Args:
    value: Single time_col value.

  Returns:
    True for aware datetime, False for naive/numeric,
    None for missing values skipped from the check.
  '''
  if value is None or value is pd.NaT:
    return None
  if isinstance(value, float) and math.isnan(value):
    return None
  if isinstance(value, (datetime.datetime, pd.Timestamp)):
    tz = getattr(value, 'tzinfo', None)
    if tz is None:
      return False
    return value.utcoffset() is not None
  if isinstance(value, str):
    try:
      parsed = pd.Timestamp(value)
      return getattr(parsed, 'tzinfo', None) is not None
    except Exception:
      return None
  if isinstance(value, np.datetime64):
    return False
  if isinstance(value, (bool, int, float, np.integer, np.floating)):
    return False
  tz = getattr(value, 'tzinfo', None)
  if tz is not None:
    return True
  return False


def _check_no_mixed_tz(series):
  '''Check whole column for mixed naive/aware before conversion.

  Args:
    series: time_col series to check.

  Raises:
    ValueError: If any value is naive and any is aware,
      raised before any ``pd.to_datetime`` conversion
      so mixed columns never surface as TypeError.
  '''
  if series is None or len(series) == 0:
    return
  has_naive = False
  has_aware = False
  values = series.tolist() if hasattr(series, 'tolist') else series
  for value in values:
    aware = _value_is_aware(value)
    if aware is True:
      has_aware = True
    elif aware is False:
      has_naive = True
    if has_naive and has_aware:
      break
  if has_naive and has_aware:
    raise ValueError(
      'temporal_split: mixed naive/aware time_col values '
      'cannot be normalized together')


def _series_is_numeric(series):
  '''Return True if series is numeric time values.

  Args:
    series: time_col series.

  Returns:
    True for numeric dtype or object dtype holding
    only numeric values, else False (datetime path).
  '''
  if pd.api.types.is_numeric_dtype(series):
    return True
  values = [v for v in series.tolist()
            if v is not None and v is not pd.NaT]
  if values and all(
      isinstance(v, (int, float, np.integer, np.floating))
      and not isinstance(v, bool) for v in values):
    return True
  return False


def _parse_embargo(raw, *, is_datetime):
  '''Normalize embargo to Timedelta or float.

  Args:
    raw: embargo as timedelta, int days/units, or
      duration string such as 'P7D'.
    is_datetime: True to return Timedelta days.

  Returns:
    Timedelta for datetime, float for numeric.

  Raises:
    ValueError: For bool or unparsable embargo.
  '''
  if raw is None:
    return pd.Timedelta(0) if is_datetime else 0.0
  if isinstance(raw, pd.Timedelta):
    if is_datetime:
      return raw
    return float(raw.total_seconds())
  if isinstance(raw, datetime.timedelta):
    delta = pd.Timedelta(raw)
    return delta if is_datetime else float(delta.total_seconds())
  if isinstance(raw, bool):
    raise ValueError('temporal_split: invalid embargo value')
  if isinstance(raw, (int, float, np.integer, np.floating)):
    number = float(raw)
    return pd.Timedelta(days=number) if is_datetime else number
  if isinstance(raw, str):
    text = raw.strip()
    if not text:
      return pd.Timedelta(0) if is_datetime else 0.0
    if text.startswith('P'):
      try:
        delta = pd.Timedelta(text)
      except Exception:
        raise ValueError(
          f'temporal_split: invalid embargo {raw!r}')
      return delta if is_datetime else float(delta.total_seconds())
    if re.fullmatch(r'-?\d+(\.\d+)?([eE][+-]?\d+)?', text):
      number = float(text)
      return pd.Timedelta(days=number) if is_datetime else number
    try:
      delta = pd.Timedelta(text)
    except Exception:
      raise ValueError(f'temporal_split: invalid embargo {raw!r}')
    return delta if is_datetime else float(delta.total_seconds())
  raise ValueError(f'temporal_split: invalid embargo {raw!r}')


def _embargo_str(parsed, *, is_datetime):
  '''Render normalized embargo duration string.

  Args:
    parsed: Parsed embargo from :func:`_parse_embargo`.
    is_datetime: True for datetime embargo.

  Returns:
    Duration string, e.g. 'P7D' or '7'.
  '''
  if is_datetime:
    delta = parsed if isinstance(parsed, pd.Timedelta) else pd.Timedelta(
      parsed)
    days = delta.total_seconds() / 86400
    if float(days).is_integer():
      return f'P{int(days)}D'
    return f'P{days:g}D'
  number = float(parsed)
  if number.is_integer():
    return str(int(number))
  return str(number)


def _parse_meta_time(value):
  '''Parse meta max/min time value.

  Args:
    value: Timestamp, numeric, or string time value.

  Returns:
    Tuple (parsed, is_datetime) where parsed is
    Timestamp or float.
  '''
  if isinstance(value, bool):
    raise ValueError('temporal leakage: invalid time value')
  if isinstance(value, (int, float, np.integer, np.floating)):
    return float(value), False
  if isinstance(value, (datetime.datetime, pd.Timestamp)):
    return pd.to_datetime(value, utc=True), True
  if isinstance(value, str):
    text = value.strip()
    if re.fullmatch(r'-?\d+(\.\d+)?([eE][+-]?\d+)?', text):
      return float(text), False
    return pd.to_datetime(text, utc=True), True
  return pd.to_datetime(value, utc=True), True


def assert_no_leakage(meta):
  '''Assert single strict leakage formula on split meta.

  Formula: ``max_time_train < min_time_test - embargo``
  (gap must be strictly greater than embargo).

  Args:
    meta: Dict with max_time_train, min_time_test, embargo.

  Raises:
    ValueError: Naming the gap when formula fails.
  '''
  if not isinstance(meta, dict):
    raise ValueError('temporal leakage: meta must be a dict')
  max_raw = meta.get('max_time_train')
  min_raw = meta.get('min_time_test')
  if max_raw is None or min_raw is None:
    return
  max_value, max_is_dt = _parse_meta_time(max_raw)
  min_value, min_is_dt = _parse_meta_time(min_raw)
  is_datetime = bool(max_is_dt or min_is_dt)
  embargo = _parse_embargo(meta.get('embargo'), is_datetime=is_datetime)
  gap = min_value - max_value
  if not max_value < min_value - embargo:
    raise ValueError(
      f'temporal leakage: train/test overlap '
      f'(gap={gap} <= embargo={embargo}, '
      f'max_time_train={max_raw!r}, min_time_test={min_raw!r})')


def embargo_gap(meta):
  '''Return (min_time_test - max_time_train) as float, pinned unit.

  Args:
    meta: Dict with max_time_train and min_time_test.

  Returns:
    Float gap. Datetime meta yields float days,
    numeric meta yields float time units.

  Raises:
    ValueError: If max/min is missing/None, the pair
      mixes datetime and numeric values, or meta is
      not a dict.
  '''
  if not isinstance(meta, dict):
    raise ValueError('temporal leakage: meta must be a dict')
  max_raw = meta.get('max_time_train')
  min_raw = meta.get('min_time_test')
  if max_raw is None or min_raw is None:
    raise ValueError(
      'embargo_gap: missing max_time_train/min_time_test')
  max_value, max_is_dt = _parse_meta_time(max_raw)
  min_value, min_is_dt = _parse_meta_time(min_raw)
  if bool(max_is_dt) != bool(min_is_dt):
    raise ValueError(
      'embargo_gap: mixed datetime/numeric time values')
  gap = min_value - max_value
  if max_is_dt:
    return float(gap.total_seconds() / 86400.0)
  return float(gap)


def assert_no_overlap(folds):
  '''Assert pairwise time and index disjointness across folds.

  Args:
    folds: List of fold dicts, each with fold_index,
      test_df, and fold_meta.

  Raises:
    ValueError: If folds is empty, any fold is
      malformed, test indices overlap, a fold leaks,
      or test blocks are not time-increasing.
  '''
  if not isinstance(folds, list) or len(folds) == 0:
    raise ValueError('assert_no_overlap: folds must be non-empty')
  parsed_mins = []
  parsed_kinds = []
  test_indices = []
  for pos, fold in enumerate(folds):
    label = pos
    if isinstance(fold, dict) and 'fold_index' in fold:
      label = fold['fold_index']
    if not isinstance(fold, dict):
      raise ValueError(
        f'assert_no_overlap: fold {label} must be a dict')
    test_df = fold.get('test_df')
    fold_meta = fold.get('fold_meta')
    if test_df is None or not hasattr(test_df, 'index'):
      raise ValueError(
        f'assert_no_overlap: fold {label} missing test_df')
    if not isinstance(fold_meta, dict):
      raise ValueError(
        f'assert_no_overlap: fold {label} missing fold_meta')
    max_raw = fold_meta.get('max_time_train')
    min_raw = fold_meta.get('min_time_test')
    if max_raw is None or min_raw is None:
      raise ValueError(
        f'assert_no_overlap: fold {label} missing '
        'max_time_train/min_time_test')
    assert_no_leakage(fold_meta)
    min_value, min_is_dt = _parse_meta_time(min_raw)
    parsed_mins.append(min_value)
    parsed_kinds.append(bool(min_is_dt))
    test_indices.append(set(test_df.index))
  for i in range(len(folds)):
    for j in range(i + 1, len(folds)):
      label_i = folds[i].get('fold_index', i)
      label_j = folds[j].get('fold_index', j)
      if test_indices[i] & test_indices[j]:
        raise ValueError(
          'assert_no_overlap: folds '
          f'{label_i} and {label_j} overlap '
          '(shared test index)')
      if parsed_kinds[i] != parsed_kinds[j]:
        raise ValueError(
          'assert_no_overlap: folds '
          f'{label_i} and {label_j} mixed time types')
      if not parsed_mins[j] > parsed_mins[i]:
        raise ValueError(
          'assert_no_overlap: folds '
          f'{label_i} and {label_j} not time-increasing '
          '(test min_time_test)')
  return None


def _temporal_fold_assignment(df, *, time_col, test_start,
                               n_test_block, embargo=0, purge=None):
  '''Return one purged-fold assignment using frozen order.

  Args:
    df: Full frame with time_col.
    time_col: Time column name.
    test_start: Positional start of the test block
      in sorted order.
    n_test_block: Number of rows in the test block.
    embargo: Embargo gap (timedelta/int/str).
    purge: Purge gap; None defaults to embargo.

  Returns:
    Dict with train_df, test_df, counts, max/min
    times, and embargo/purge strings.

  Raises:
    ValueError: On empty/missing time_col, invalid
      geometry, mixed tz, or embargo consuming data.
  '''
  if time_col is None:
    raise ValueError(
      'temporal_split: time_col is required for method=temporal')
  if df is None or len(df) == 0:
    raise ValueError('temporal_split requires a non-empty dataframe')
  if time_col not in df.columns:
    raise ValueError(f'time column not found: {time_col}')
  total = len(df)
  if isinstance(test_start, bool) or not isinstance(
      test_start, int):
    raise ValueError(
      '_temporal_fold_assignment: test_start must be an int')
  if isinstance(n_test_block, bool) or not isinstance(
      n_test_block, int):
    raise ValueError(
      '_temporal_fold_assignment: n_test_block must be int')
  if not test_start > 0 or not n_test_block > 0:
    raise ValueError(
      '_temporal_fold_assignment: test_start and n_test_block '
      'must be positive')
  if not test_start < total:
    raise ValueError(
      '_temporal_fold_assignment: test_start out of range')
  if not test_start + n_test_block <= total:
    raise ValueError(
      '_temporal_fold_assignment: test block out of range')
  time_series = df[time_col]
  _check_no_mixed_tz(time_series)
  is_numeric = _series_is_numeric(time_series)
  is_datetime = not is_numeric
  embargo_parsed = _parse_embargo(embargo, is_datetime=is_datetime)
  if purge is None:
    purge_parsed = embargo_parsed
  else:
    purge_parsed = _parse_embargo(purge, is_datetime=is_datetime)
  effective = max(embargo_parsed, purge_parsed)
  if is_numeric:
    numeric_times = pd.to_numeric(time_series, errors='raise')
    order = numeric_times.sort_values(kind='stable').index
    sorted_df = df.loc[order]
    sorted_times = numeric_times.loc[order].to_numpy(dtype=float)
    min_value = float(sorted_times[test_start])
    horizon = min_value - float(effective)
    train_end = -1
    for idx in range(test_start):
      if float(sorted_times[idx]) < horizon:
        train_end = idx
    if train_end < 0:
      raise ValueError(
        'temporal_split: embargo consumes <test_size> fraction')
    embargo_horizon = min_value - float(embargo_parsed)
    purge_horizon = min_value - float(purge_parsed)
    n_embargoed = 0
    n_purged = 0
    for pos in range(train_end + 1, test_start):
      value = float(sorted_times[pos])
      if value >= embargo_horizon:
        n_embargoed += 1
      elif purge_horizon <= value < embargo_horizon:
        n_purged += 1
    max_train_value = float(sorted_times[train_end])
    if float(max_train_value).is_integer():
      max_str = str(int(max_train_value))
    else:
      max_str = str(max_train_value)
    if float(min_value).is_integer():
      min_str = str(int(min_value))
    else:
      min_str = str(min_value)
  else:
    normalized = pd.to_datetime(time_series, utc=True, errors='raise')
    order = normalized.sort_values(kind='stable').index
    sorted_df = df.loc[order]
    sorted_times = normalized.loc[order]
    min_time = sorted_times.iloc[test_start]
    horizon_time = min_time - effective
    train_end = -1
    for idx in range(test_start):
      if sorted_times.iloc[idx] < horizon_time:
        train_end = idx
    if train_end < 0:
      raise ValueError(
        'temporal_split: embargo consumes <test_size> fraction')
    embargo_horizon = min_time - embargo_parsed
    purge_horizon = min_time - purge_parsed
    n_embargoed = 0
    n_purged = 0
    for pos in range(train_end + 1, test_start):
      value = sorted_times.iloc[pos]
      if value >= embargo_horizon:
        n_embargoed += 1
      elif purge_horizon <= value < embargo_horizon:
        n_purged += 1
    max_str = sorted_times.iloc[train_end].isoformat()
    min_str = min_time.isoformat()
  train = sorted_df.iloc[:train_end + 1]
  test = sorted_df.iloc[test_start:test_start + n_test_block]
  check = {
    'max_time_train': max_str,
    'min_time_test': min_str,
    'embargo': _embargo_str(effective, is_datetime=is_datetime),
  }
  assert_no_leakage(check)
  return {
    'train_df': train,
    'test_df': test,
    'n_train': int(len(train)),
    'n_test': int(len(test)),
    'n_embargoed': int(n_embargoed),
    'n_purged': int(n_purged),
    'max_time_train': max_str,
    'min_time_test': min_str,
    'embargo': _embargo_str(effective, is_datetime=is_datetime),
    'purge': _embargo_str(purge_parsed, is_datetime=is_datetime),
  }


def _frames_max_min(train_df, test_df, time_col):
  '''Compute max_train / min_test from frames.

  Args:
    train_df: Train frame with time_col.
    test_df: Test frame with time_col.
    time_col: Column name.

  Returns:
    Tuple (max_str, min_str, gap_str, embargo_raw_placeholder)
    with ISO strings or numeric strings.
  '''
  train_series = train_df[time_col]
  test_series = test_df[time_col]
  combined = pd.concat([train_series, test_series], ignore_index=True)
  _check_no_mixed_tz(combined)
  if _series_is_numeric(train_series) and _series_is_numeric(test_series):
    max_value = float(train_series.max())
    min_value = float(test_series.min())
    gap = min_value - max_value
    # ISO-like numeric strings keep contract str type.
    if float(max_value).is_integer():
      max_str = str(int(max_value))
    else:
      max_str = str(max_value)
    if float(min_value).is_integer():
      min_str = str(int(min_value))
    else:
      min_str = str(min_value)
    return max_str, min_str, str(gap), False
  train_times = pd.to_datetime(train_series, utc=True, errors='raise')
  test_times = pd.to_datetime(test_series, utc=True, errors='raise')
  max_value = train_times.max()
  min_value = test_times.min()
  gap = min_value - max_value
  return max_value.isoformat(), min_value.isoformat(), str(gap), True


def temporal_split(df, params=None, test_df=None):
  '''Split by temporal fixed point or validate presplit cut.

  Args:
    df: Full frame for method='temporal', or train
      frame for method='presplit'.
    params: Dict with target_col, time_col, method,
      test_size, embargo, random_state, task_type.
    test_df: Test frame for presplit (frozen signature:
      ``temporal_split(train_df, params, test_df)``;
      tuple/list input and train/test params keys are
      not supported).

  Returns:
    Tuple (train_raw, test_raw, split_meta).

  Raises:
    ValueError: Missing time_col, mixed tz, embargo
      consuming data, leaky presplit cut, tuple input,
      or presplit without a test frame.
  '''
  if isinstance(df, (tuple, list)):
    raise ValueError(
      'temporal_split: presplit requires train/test frames; '
      'tuple/list input is not supported')
  params = params if isinstance(params, dict) else {}
  sampling = params.get('sampling_params', {}) or {}
  method = str(sampling.get('method', params.get('method', 'temporal')))
  method = method.strip().lower()
  time_col = sampling.get('time_col', params.get('time_col'))
  test_size = sampling.get('test_size', params.get('test_size', 0.2))
  if isinstance(test_size, bool) or not isinstance(
      test_size, (int, float)) or not 0 < test_size < 1:
    raise ValueError('test_size must be in (0, 1)')
  test_size = float(test_size)
  embargo_raw = sampling.get('embargo', params.get('embargo', 0))
  if params.get('sampling_params') is not None and 'embargo' in sampling:
    embargo_raw = sampling.get('embargo')
  if method == 'presplit':
    if test_df is None:
      raise ValueError(
        'temporal_split: presplit requires train/test frames')
    return _temporal_split_presplit(df, test_df, params)
  if time_col is None:
    raise ValueError(
      'temporal_split: time_col is required for method=temporal')
  if df is None or len(df) == 0:
    raise ValueError('temporal_split requires a non-empty dataframe')
  if time_col not in df.columns:
    raise ValueError(f'time column not found: {time_col}')
  time_series = df[time_col]
  _check_no_mixed_tz(time_series)
  is_numeric = _series_is_numeric(time_series)
  is_datetime = not is_numeric
  embargo = _parse_embargo(embargo_raw, is_datetime=is_datetime)
  if is_numeric:
    numeric_times = pd.to_numeric(time_series, errors='raise')
    order = numeric_times.sort_values(kind='stable').index
    sorted_df = df.loc[order]
    sorted_times = numeric_times.loc[order].to_numpy(dtype=float)
  else:
    normalized = pd.to_datetime(time_series, utc=True, errors='raise')
    order = normalized.sort_values(kind='stable').index
    sorted_df = df.loc[order]
    sorted_times = normalized.loc[order]
  total = len(sorted_df)
  n_test_target = int(math.ceil(test_size * total))
  test_start = total - n_test_target
  if test_start <= 0:
    raise ValueError(
      'temporal_split: embargo consumes <test_size> fraction')
  if is_numeric:
    horizon = float(sorted_times[test_start]) - float(embargo)
    train_end = -1
    for idx in range(test_start):
      if float(sorted_times[idx]) < horizon:
        train_end = idx
    if train_end < 0:
      raise ValueError(
        'temporal_split: embargo consumes <test_size> fraction')
  else:
    horizon_time = sorted_times.iloc[test_start] - embargo
    train_end = -1
    for idx in range(test_start):
      if sorted_times.iloc[idx] < horizon_time:
        train_end = idx
    if train_end < 0:
      raise ValueError(
        'temporal_split: embargo consumes <test_size> fraction')
  train = sorted_df.iloc[:train_end + 1]
  test = sorted_df.iloc[test_start:]
  embargoed = sorted_df.iloc[train_end + 1:test_start]
  n_embargoed = int(len(embargoed))
  if is_numeric:
    max_train_value = float(sorted_times[train_end])
    min_test_value = float(sorted_times[test_start])
    if float(max_train_value).is_integer():
      max_str = str(int(max_train_value))
    else:
      max_str = str(max_train_value)
    if float(min_test_value).is_integer():
      min_str = str(int(min_test_value))
    else:
      min_str = str(min_test_value)
  else:
    max_str = sorted_times.iloc[train_end].isoformat()
    min_str = sorted_times.iloc[test_start].isoformat()
  meta = {
    'sampling_method': 'temporal',
    'requested_method': method,
    'shuffle': False,
    'time_col': time_col,
    'test_size': float(test_size),
    'random_state': None,
    'n_train': int(len(train)),
    'n_test': int(len(test)),
    'n_embargoed': n_embargoed,
    'embargo': _embargo_str(embargo, is_datetime=is_datetime),
    'max_time_train': max_str,
    'min_time_test': min_str,
    'leakage_checked': True,
    'task_type': params.get('task_type', 'classification'),
  }
  assert_no_leakage(meta)
  return train, test, meta


def _temporal_split_presplit(train_df, test_df, params):
  '''Presplit helper: validate caller cut, never move rows.

  Args:
    train_df: Caller train frame.
    test_df: Caller test frame.
    params: Temporal params dict.

  Returns:
    Tuple (train, test, meta) unchanged frames.

  Raises:
    ValueError: On leaky caller cut naming the gap, or
      malformed test_size (fail-loud, matching temporal).
  '''
  if train_df is None or test_df is None:
    raise ValueError(
      'temporal_split: presplit requires train and test frames')
  sampling = (params or {}).get('sampling_params', {}) or {}
  time_col = sampling.get('time_col', (params or {}).get('time_col'))
  embargo_raw = sampling.get(
    'embargo', (params or {}).get('embargo', 0))
  test_size = sampling.get(
    'test_size', (params or {}).get('test_size', 0.2))
  if isinstance(test_size, bool) or not isinstance(
      test_size, (int, float)) or not 0 < test_size < 1:
    raise ValueError('test_size must be in (0, 1)')
  max_str = None
  min_str = None
  gap_str = None
  is_datetime = True
  embargo_parsed = None
  if time_col is not None and time_col in train_df.columns \
      and time_col in test_df.columns:
    max_str, min_str, gap_str, is_datetime = _frames_max_min(
      train_df, test_df, time_col)
    embargo_parsed = _parse_embargo(embargo_raw, is_datetime=is_datetime)
  leakage_checked = bool(max_str is not None and min_str is not None)
  meta = {
    'sampling_method': 'presplit',
    'requested_method': 'presplit',
    'shuffle': False,
    'time_col': time_col,
    'test_size': float(test_size),
    'random_state': None,
    'n_train': int(len(train_df)),
    'n_test': int(len(test_df)),
    'n_embargoed': 0,
    'embargo': _embargo_str(
      embargo_parsed if embargo_parsed is not None else _parse_embargo(
        embargo_raw, is_datetime=True), is_datetime=is_datetime)
    if leakage_checked else None,
    'max_time_train': max_str,
    'min_time_test': min_str,
    'leakage_checked': leakage_checked,
    'leakage_gap': gap_str,
    'task_type': (params or {}).get('task_type', 'classification'),
  }
  if leakage_checked:
    check = {
      'max_time_train': max_str,
      'min_time_test': min_str,
      'embargo': embargo_raw,
    }
    assert_no_leakage(check)
  return train_df, test_df, meta


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
  requested = str(sampling.get(
    'method', params.get('method', 'stratified')))
  requested = requested.strip().lower()
  x = df[features]
  y = df[target]
  if method == 'time':
    if time_col is not None:
      if time_col not in df.columns:
        raise ValueError(f'time column not found: {time_col}')
      embargo_raw = sampling.get('embargo', params.get('embargo', 0))
      temporal_params = {
        'target_col': target,
        'time_col': time_col,
        'method': 'temporal',
        'test_size': test_size,
        'embargo': embargo_raw,
        'task_type': params.get('task_type', 'classification'),
      }
      train_raw, test_raw, temporal_meta = temporal_split(
        df, temporal_params)
      x_tr = train_raw[features]
      x_te = test_raw[features]
      y_tr = np.asarray(train_raw[target])
      y_te = np.asarray(test_raw[target])
      meta = {
        'task_type': params.get('task_type', 'classification'),
        'test_size': test_size,
        'random_state': None,
        'sampling_method': 'time',
        'requested_method': 'time',
        'shuffle': False,
        'time_col': time_col,
        'n_train': int(temporal_meta['n_train']),
        'n_test': int(temporal_meta['n_test']),
        'n_embargoed': int(temporal_meta['n_embargoed']),
        'embargo': temporal_meta['embargo'],
        'max_time_train': temporal_meta['max_time_train'],
        'min_time_test': temporal_meta['min_time_test'],
        'leakage_checked': True,
      }
      return x_tr, x_te, y_tr, y_te, meta
    if requested == 'temporal':
      raise ValueError(
        'temporal_split: time_col is required for method=temporal')
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
          'n_train': int(len(y_tr)), 'n_test': int(len(y_te)),
          'n_embargoed': 0, 'embargo': None,
          'max_time_train': None, 'min_time_test': None,
          'leakage_checked': False}
  return x_tr, x_te, np.asarray(y_tr), np.asarray(y_te), meta
