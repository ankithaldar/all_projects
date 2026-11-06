#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Fitted preprocessing registry: fit on train, replay at score.

Boundary: covers exactly the seven stateless steps in
``data_prep/nodes.py`` that have fit/apply pairs: Imputations,
Normalizations, QuantileDiscretizer, HighLabelBinning,
CategoricalEncoding, TargetEncoding, CapOutliers. KNN
imputation lives in ``nodes.py`` only (not here). Non-fitted
or unknown steps are rejected with ``ValueError`` at fit and
at apply time -- never silently skipped. Registry values are
JSON-serializable (numpy converted to lists) and carry a
``version`` / ``schema`` field.
'''

import datetime
import json
import logging
import uuid
from decimal import Decimal

import numpy as np
import pandas as pd

from forecast_ml_framework.pipelines.data_prep.nodes import (
  _iter_steps,
  _resolve_cols,
  _resolve_ohe_max_cols,
)

LOG = logging.getLogger(__name__)

FITTED_REGISTRY_VERSION = 1
REGISTRY_VERSION = FITTED_REGISTRY_VERSION
REGISTRY_SCHEMA = 'fitted_registry.v1'
FITTED_REGISTRY_SCHEMA = REGISTRY_SCHEMA
MAX_NUM_BUCKETS = 1000
_REGISTRY_META_KEYS = frozenset({
  'version', '__version__', 'schema', '__schema__', '__meta__', 'steps',
})


def _native(v):
  '''Convert numpy / pandas scalars to native JSON-safe values.'''
  if isinstance(v, (np.integer,)):
    return int(v)
  if isinstance(v, (np.floating,)):
    return float(v)
  if isinstance(v, (np.bool_,)):
    return bool(v)
  if isinstance(v, np.ndarray):
    return [_native(x) for x in v.tolist()]
  if isinstance(v, (list, tuple)):
    return [_native(x) for x in v]
  if isinstance(v, dict):
    return {str(k): _native(x) for k, x in v.items()}
  if isinstance(v, pd.Timestamp):
    return v.isoformat()
  if isinstance(v, datetime.datetime):
    return {'__type__': 'datetime', 'value': v.isoformat()}
  if isinstance(v, datetime.date):
    return {'__type__': 'date', 'value': v.isoformat()}
  if isinstance(v, Decimal):
    return {'__type__': 'decimal', 'value': str(v)}
  if isinstance(v, uuid.UUID):
    return {'__type__': 'uuid', 'value': str(v)}
  if isinstance(v, (bytes, bytearray)):
    return {'__type__': 'bytes', 'value': bytes(v).hex()}
  if isinstance(v, pd.Period):
    return {'__type__': 'period', 'value': str(v),
            'freq': v.freqstr}
  if isinstance(v, pd.Timedelta):
    return {'__type__': 'pd_timedelta', 'value': str(v)}
  if isinstance(v, datetime.timedelta):
    return {'__type__': 'timedelta', 'value': str(v)}
  return v


def _jsonable(obj):
  '''Convert obj to JSON-serializable form via :func:`_native`.'''
  return _native(obj)


def _encode_typed(value):
  '''Encode a category preserving tuple/Timestamp type losslessly.

  Tuples become ``{"__type__": "tuple", "value": [...]}`` and
  Timestamps become ``{"__type__": "timestamp", "value": iso}``
  so JSON serialization does not turn them into lists/strings
  that ``isin``/``map`` at apply time can never match. Dates,
  datetimes, Decimals, UUIDs, bytes, Periods, and Timedeltas
  are likewise typed (via :func:`_native`) so they survive JSON
  and decode back to the original type. Other values use
  :func:`_native` and are unchanged.
  '''
  if isinstance(value, tuple):
    return {'__type__': 'tuple',
            'value': [_encode_typed(v) for v in value]}
  if isinstance(value, pd.Timestamp):
    return {'__type__': 'timestamp', 'value': value.isoformat()}
  if isinstance(value, (list, np.ndarray)):
    return [_encode_typed(v) for v in value]
  return _native(value)


def _decode_typed(value):
  '''Inverse of :func:`_encode_typed`; plain values pass through.'''
  if isinstance(value, dict):
    if value.get('__type__') == 'tuple' and 'value' in value:
      return tuple(_decode_typed(v) for v in value['value'])
    if value.get('__type__') == 'timestamp' and 'value' in value:
      return pd.Timestamp(value['value'])
    if value.get('__type__') == 'datetime' and 'value' in value:
      return datetime.datetime.fromisoformat(value['value'])
    if value.get('__type__') == 'date' and 'value' in value:
      return datetime.date.fromisoformat(value['value'])
    if value.get('__type__') == 'decimal' and 'value' in value:
      return Decimal(value['value'])
    if value.get('__type__') == 'uuid' and 'value' in value:
      return uuid.UUID(value['value'])
    if value.get('__type__') == 'bytes' and 'value' in value:
      return bytes.fromhex(value['value'])
    if value.get('__type__') == 'period' and 'value' in value:
      freq = value.get('freq')
      if freq:
        return pd.Period(value['value'], freq=freq)
      return pd.Period(value['value'])
    if value.get('__type__') == 'pd_timedelta' and 'value' in value:
      return pd.Timedelta(value['value'])
    if value.get('__type__') == 'timedelta' and 'value' in value:
      return pd.Timedelta(value['value']).to_pytimedelta()
    return {k: _decode_typed(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_decode_typed(v) for v in value]
  return value


def _is_number(value):
  '''Return True for a finite, non-bool numeric value.'''
  if isinstance(value, (bool, np.bool_)):
    return False
  if not isinstance(value, (int, float, np.integer, np.floating)):
    return False
  return bool(np.isfinite(float(value)))


def _finite_float(value, col, what):
  '''Return ``value`` as float, else raise ValueError naming col.'''
  if not _is_number(value):
    raise ValueError(
      'Invalid %s for column %r: %r' % (what, col, value))
  return float(value)


def _reject_inf_numeric(df, cols, step):
  '''Raise ValueError if a listed numeric column contains inf.'''
  for col in cols:
    if col not in df.columns:
      continue
    series = df[col]
    if not pd.api.types.is_numeric_dtype(series):
      continue
    if bool(np.isinf(
            pd.to_numeric(series, errors='coerce')
            .fillna(0).to_numpy(dtype=float)).any()):
      raise ValueError(
        'Invalid %s data for column %r: non-finite (inf) value'
        % (step, col))


def _assert_finite_state(state, step, col=None):
  '''Recursively reject NaN/inf floats in a fitted state dict.'''
  if isinstance(state, dict):
    for key, val in state.items():
      if isinstance(key, (float, np.floating)) and not np.isfinite(
              float(key)):
        raise ValueError(
          'Invalid %s state for column %r: non-finite key %r'
          % (step, col, key))
      _assert_finite_state(val, step, col)
  elif isinstance(state, (list, tuple)):
    for val in state:
      _assert_finite_state(val, step, col)
  elif isinstance(state, (float, np.floating)):
    if not np.isfinite(float(state)):
      raise ValueError(
        'Invalid %s state for column %r: non-finite value %r'
        % (step, col, state))


def _validate_pair(col, pair):
  '''Validate a (lo, hi) pair and return floats.

  Raises:
    ValueError: If pair is not a two-number list/tuple or
      lo > hi, naming the column.
  '''
  if not isinstance(pair, (list, tuple)) or len(pair) != 2:
    raise ValueError(
      'Invalid bound pair for column %r: %r' % (col, pair))
  lo, hi = pair
  if not _is_number(lo) or not _is_number(hi):
    raise ValueError(
      'Invalid bound pair for column %r: %r' % (col, pair))
  lo, hi = float(lo), float(hi)
  if lo > hi:
    raise ValueError(
      'Invalid bound pair for column %r: lo > hi (%r > %r)'
      % (col, lo, hi))
  return lo, hi


def _validate_quantile_bound(name, value):
  '''Validate a quantile bound in [0, 1].

  Raises:
    ValueError: If value is not a number in [0, 1].
  '''
  if not _is_number(value):
    raise ValueError(
      'Invalid quantile bound %s: %r' % (name, value))
  value = float(value)
  if not 0.0 <= value <= 1.0:
    raise ValueError(
      'Invalid quantile bound %s: %r not in [0, 1]' % (name, value))
  return value


def _check_dummy_collision(expected_dummies, existing):
  '''Raise ValueError on duplicate or existing-column collisions.'''
  seen = set()
  for dummy in expected_dummies:
    if dummy in seen:
      raise ValueError(
        'CategoricalEncoding OHE dummy name collision: duplicate '
        'dummy column %r' % (dummy,))
    seen.add(dummy)
  for dummy in expected_dummies:
    if dummy in existing:
      raise ValueError(
        'CategoricalEncoding OHE dummy name collision: dummy '
        'column %r collides with existing column %r'
        % (dummy, dummy))


def _check_registry_version(registry):
  '''Raise ValueError when a present registry version/schema differs.'''
  if not isinstance(registry, dict):
    return
  candidates = []
  for key in ('version', '__version__'):
    if key in registry:
      candidates.append(('version', registry[key]))
  for key in ('schema', '__schema__'):
    if key in registry:
      candidates.append(('schema', registry[key]))
  meta = registry.get('__meta__')
  if isinstance(meta, dict):
    for key in ('version', '__version__'):
      if key in meta:
        candidates.append(('version', meta[key]))
    for key in ('schema', '__schema__'):
      if key in meta:
        candidates.append(('schema', meta[key]))
  for kind, value in candidates:
    expected = (FITTED_REGISTRY_VERSION if kind == 'version'
                else REGISTRY_SCHEMA)
    if value != expected:
      raise ValueError(
        'fitted_registry %s mismatch: got %r, expected %r'
        % (kind, value, expected))


def _registry_steps(registry):
  '''Return step->state mapping excluding registry meta keys.'''
  registry = registry or {}
  if not isinstance(registry, dict):
    raise ValueError('fitted_registry must be a dict')
  _check_registry_version(registry)
  steps = {}
  nested = registry.get('steps')
  if isinstance(nested, dict):
    steps.update(nested)
  for name, state in registry.items():
    if name in _REGISTRY_META_KEYS:
      continue
    if name in steps:
      raise ValueError(
        'Duplicate preprocessing step name: %s' % name)
    steps[name] = state
  return steps


def _registry_has_typed_categories(registry):
  '''Return True when registry declares typed-category encoding.'''
  if not isinstance(registry, dict):
    return False
  meta = registry.get('__meta__')
  if isinstance(meta, dict) and meta.get('typed_categories') is True:
    return True
  return registry.get('typed_categories') is True


def _is_typed_category_value(value):
  '''Return True for values requiring typed category encoding.'''
  return isinstance(value, (tuple, pd.Timestamp, datetime.datetime,
                            datetime.date, Decimal, uuid.UUID, bytes,
                            bytearray, pd.Period, pd.Timedelta,
                            datetime.timedelta))


def _check_legacy_typed_data(df, registry, steps):
  '''Fail loud for legacy registries when typed data is scored.

  Pre-typed registries (version 1, no ``typed_categories`` flag)
  stored tuples as plain lists, so ``isin``/``map`` at apply time
  can never match tuple data. They cannot be distinguished by
  version/schema (both unchanged), so refuse and instruct refit
  instead of silently corrupting the output.
  '''
  if _registry_has_typed_categories(registry):
    return
  for name in ('HighLabelBinning', 'CategoricalEncoding',
               'TargetEncoding'):
    state = steps.get(name)
    if not isinstance(state, dict):
      continue
    for col in state:
      if col not in df.columns:
        continue
      for value in df[col]:
        if _is_typed_category_value(value):
          raise ValueError(
            'legacy fitted_registry (no typed_categories flag) '
            'cannot apply to column %r containing tuple/typed '
            'categories: categories were stored as plain lists '
            'and would not match; refit preprocessing to create '
            'a typed registry' % (col,))


def _fit_impute(df, args, variables):
  '''Fit imputation fill values per column.'''
  cols = _resolve_cols(args, variables.get('num_cols', []))
  strategy = args.get('strategy', 'none')
  bounds = args.get('custom_bounds') or {}
  _reject_inf_numeric(df, cols, 'Imputations')
  state = {}
  for col in cols:
    if col not in df.columns:
      continue
    fill = None
    if strategy == 'mean':
      if not df[col].notna().any():
        raise ValueError(
          'Imputations cannot fit all-NaN column %r' % (col,))
      fill = df[col].mean()
    elif strategy == 'median':
      if not df[col].notna().any():
        raise ValueError(
          'Imputations cannot fit all-NaN column %r' % (col,))
      fill = df[col].median()
    elif strategy == 'mode':
      mode = df[col].mode(dropna=True)
      if len(mode) == 0:
        raise ValueError(
          'Imputations cannot fit all-NaN column %r' % (col,))
      fill = mode.iloc[0]
    elif col in bounds:
      fill = bounds[col]
    if fill is not None:
      if isinstance(fill, (float, np.floating)) and not np.isfinite(
              float(fill)):
        raise ValueError(
          'Imputations cannot fit column %r: non-finite fill %r'
          % (col, fill))
      if isinstance(fill, (int, float, np.integer, np.floating)) and (
              not _is_number(fill)):
        raise ValueError(
          'Imputations cannot fit column %r: non-finite fill %r'
          % (col, fill))
      state[col] = {'fill': _native(fill), 'strategy': strategy}
  for col, st in state.items():
    _assert_finite_state(st, 'Imputations', col)
  return state


def _apply_impute(df, state):
  '''Apply stored fill values.'''
  for col, st in (state or {}).items():
    if col in df.columns:
      if not isinstance(st, dict) or 'fill' not in st:
        raise ValueError(
          'Invalid Imputations state for column %r: %r' % (col, st))
      fill = st.get('fill')
      if fill is None or isinstance(fill, (list, tuple, dict)):
        raise ValueError(
          'Invalid Imputations fill for column %r: %r' % (col, fill))
      if isinstance(fill, (float, np.floating)) and not np.isfinite(
              float(fill)):
        raise ValueError(
          'Invalid Imputations fill for column %r: %r' % (col, fill))
      if pd.api.types.is_numeric_dtype(df[col]) and not _is_number(fill):
        raise ValueError(
          'Invalid Imputations fill for column %r: %r' % (col, fill))
      df[col] = df[col].fillna(fill)
  return df


def _fit_normalize(df, args, variables):
  '''Fit normalization bounds per column.'''
  cols = _resolve_cols(args, variables.get('num_cols', []))
  handle = str(args.get('handle', 'MinMaxScaler'))
  _reject_inf_numeric(df, cols, 'Normalizations')
  state = {}
  for col in cols:
    if col not in df.columns:
      continue
    if not df[col].notna().any():
      raise ValueError(
        'Normalizations cannot fit all-NaN column %r' % (col,))
    if 'Standard' in handle or 'ZScore' in handle:
      mu = float(df[col].mean())
      sd = float(df[col].std())
      if np.isnan(sd) or sd == 0.0:
        sd = 1.0
      if not _is_number(mu) or not _is_number(sd):
        raise ValueError(
          'Normalizations cannot fit column %r: mu=%r sd=%r'
          % (col, mu, sd))
      state[col] = {'kind': 'standard', 'mu': mu, 'sd': sd}
    else:
      lo = float(df[col].min())
      hi = float(df[col].max())
      if not _is_number(lo) or not _is_number(hi):
        raise ValueError(
          'Normalizations cannot fit column %r: lo=%r hi=%r'
          % (col, lo, hi))
      state[col] = {'kind': 'minmax', 'lo': lo, 'hi': hi}
  for col, st in state.items():
    _assert_finite_state(st, 'Normalizations', col)
  return state


def _apply_normalize(df, state):
  '''Apply stored normalization bounds verbatim.'''
  for col, st in (state or {}).items():
    if col not in df.columns:
      continue
    if not isinstance(st, dict):
      raise ValueError(
        'Invalid Normalizations state for column %r: %r' % (col, st))
    kind = st.get('kind', 'minmax')
    if kind == 'standard':
      mu = _finite_float(st.get('mu'), col, 'Normalizations mu')
      sd = _finite_float(st.get('sd'), col, 'Normalizations sd')
      if sd == 0.0:
        raise ValueError(
          'Invalid Normalizations sd for column %r: %r' % (col, sd))
      df[col] = (df[col] - mu) / sd
    elif kind == 'minmax':
      lo = _finite_float(st.get('lo'), col, 'Normalizations lo')
      hi = _finite_float(st.get('hi'), col, 'Normalizations hi')
      rng = (hi - lo) or 1.0
      df[col] = (df[col] - lo) / rng
    else:
      raise ValueError(
        'Invalid Normalizations kind for column %r: %r' % (col, kind))
  return df


def _fit_discretize(df, args, variables):
  '''Fit quantile bin edges per column from train data.'''
  cols = _resolve_cols(args, variables.get('num_cols', []))
  raw_buckets = (args.get('params') or {}).get('numBuckets', 10)
  if isinstance(raw_buckets, (bool, np.bool_)) or not isinstance(
      raw_buckets, (int, np.integer)):
    raise ValueError(
      'Invalid numBuckets for QuantileDiscretizer: %r'
      % (raw_buckets,))
  buckets = int(raw_buckets)
  if buckets < 1:
    raise ValueError(
      'Invalid numBuckets for QuantileDiscretizer: %r' % (buckets,))
  if buckets > MAX_NUM_BUCKETS:
    raise ValueError(
      'Invalid numBuckets for QuantileDiscretizer: %d exceeds '
      'cap %d' % (buckets, MAX_NUM_BUCKETS))
  _reject_inf_numeric(df, cols, 'QuantileDiscretizer')
  state = {}
  for col in cols:
    if col not in df.columns:
      continue
    vals = pd.to_numeric(df[col], errors='coerce').dropna()
    if len(vals) == 0:
      raise ValueError(
        'QuantileDiscretizer cannot fit all-NaN column %r' % (col,))
    if not np.isfinite(vals.to_numpy(dtype=float)).all():
      raise ValueError(
        'QuantileDiscretizer cannot fit column %r: non-finite value'
        % (col,))
    eff_buckets = max(1, min(buckets, len(vals)))
    try:
      edges = np.unique(np.quantile(
        vals, np.linspace(0, 1, eff_buckets + 1)))
    except Exception:
      edges = np.array([float(vals.min()), float(vals.max())])
    edges = [float(e) for e in edges]
    if len(edges) < 2:
      v = edges[0] if edges else 0.0
      edges = [v - 0.5, v + 0.5]
    if not all(_is_number(e) for e in edges):
      raise ValueError(
        'QuantileDiscretizer cannot fit column %r: non-finite edges %r'
        % (col, edges))
    state[col] = {'edges': edges}
  for col, st in state.items():
    _assert_finite_state(st, 'QuantileDiscretizer', col)
  return state


def _apply_discretize(df, state):
  '''Apply stored bin edges via digitize; never recompute.'''
  for col, st in (state or {}).items():
    if col not in df.columns:
      continue
    if not isinstance(st, dict):
      raise ValueError(
        'Invalid QuantileDiscretizer state for column %r: %r'
        % (col, st))
    raw_edges = st.get('edges')
    if not isinstance(raw_edges, (list, tuple)) or len(raw_edges) < 2:
      raise ValueError(
        'Invalid QuantileDiscretizer edges for column %r: %r'
        % (col, raw_edges))
    edges = []
    for edge in raw_edges:
      edges.append(_finite_float(
        edge, col, 'QuantileDiscretizer edge'))
    if any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
      raise ValueError(
        'Invalid QuantileDiscretizer edges for column %r: '
        'edges must be strictly increasing: %r' % (col, edges))
    inner = edges[1:-1]
    vals = pd.to_numeric(df[col], errors='coerce')
    df[col] = np.digitize(vals, inner) if inner else 0
    df[col] = df[col].astype(int)
  return df


def _fit_bin_labels(df, args, variables):
  '''Fit high-label binning keep-set per column.'''
  cols = _resolve_cols(args, variables.get('cat_cols', []))
  val_count = int(args.get('val_count', 300))
  state = {}
  for col in cols:
    if col not in df.columns:
      continue
    counts = df[col].value_counts()
    keep = [k for k in counts[counts >= val_count].index]
    state[col] = {'keep': [_encode_typed(k) for k in keep]}
  return state


def _apply_bin_labels(df, state):
  '''Apply stored keep-set, else ``Other``.'''
  for col, st in (state or {}).items():
    if col in df.columns:
      keep = set(_decode_typed(k) for k in (st.get('keep') or []))
      df[col] = df[col].where(df[col].isin(keep), 'Other')
  return df


def _fit_encode_cat(df, args, variables):
  '''Fit categorical encoding categories per column.'''
  cols = _resolve_cols(args, variables.get('cat_cols', []))
  ohe = bool(args.get('ohe_flag'))
  max_cols = _resolve_ohe_max_cols(args)
  state = {}
  for col in cols:
    if col not in df.columns:
      continue
    cats = list(pd.factorize(df[col])[1])
    if ohe and max_cols is not None and len(cats) > max_cols:
      raise ValueError(
        'CategoricalEncoding exceeds max_cols=%d: categorical '
        'column %r has %d distinct values'
        % (max_cols, col, len(cats)))
    state[col] = {
      'categories': [_encode_typed(c) for c in cats], 'ohe': ohe,
      'max_cols': max_cols}
  if ohe and state:
    # Dummy-name collision check from fitted categories: distinct
    # (column, category) pairs (a/b_c vs a_b/c -> a_b_c) or a dummy
    # vs an existing column would emit duplicate columns.
    present = list(state)
    expected = ['%s_%s' % (col, _decode_typed(cat))
                for col, st in state.items() for cat in st['categories']]
    _check_dummy_collision(expected, set(df.columns) - set(present))
  if ohe and max_cols is not None and state:
    present = list(state)
    expanded = len(df.columns) - len(present)
    expanded += sum(
      len(st['categories']) for st in state.values())
    if expanded > max_cols:
      raise ValueError(
        'CategoricalEncoding exceeds max_cols=%d: get_dummies '
        'expansion would produce %d columns for categorical '
        'columns %r' % (max_cols, expanded, present))
  return state


def _apply_encode_cat(df, state):
  '''Apply stored categories; unseen maps to -1 (or no dummy).'''
  state = state or {}
  ohe_present = [c for c, st in state.items()
                 if c in df.columns and (st or {}).get('ohe')]
  if ohe_present:
    expected = ['%s_%s' % (col, _decode_typed(cat))
                for col in ohe_present
                for cat in ((state[col] or {}).get('categories') or [])]
    _check_dummy_collision(
      expected, set(df.columns) - set(ohe_present))
  if ohe_present:
    caps = []
    for col in ohe_present:
      st = state[col] or {}
      if 'max_cols' in st:
        caps.append(st.get('max_cols'))
      else:
        caps.append(_resolve_ohe_max_cols({}))
    max_cols = next((c for c in caps if c is not None), None)
    if max_cols is not None:
      for col in ohe_present:
        cats = list((state[col] or {}).get('categories') or [])
        if len(cats) > max_cols:
          raise ValueError(
            'CategoricalEncoding exceeds max_cols=%d: '
            'categorical column %r has %d fitted categories'
            % (max_cols, col, len(cats)))
      expanded = len(df.columns) - len(ohe_present)
      expanded += sum(
        len(list((state[c] or {}).get('categories') or []))
        for c in ohe_present)
      if expanded > max_cols:
        raise ValueError(
          'CategoricalEncoding exceeds max_cols=%d: get_dummies '
          'expansion would produce %d columns for categorical '
          'columns %r' % (max_cols, expanded, ohe_present))
  for col, st in state.items():
    if col not in df.columns:
      continue
    cats = [_decode_typed(c) for c in (st.get('categories') or [])]
    if st.get('ohe'):
      filtered = df[col].where(df[col].isin(cats))
      dummies = pd.get_dummies(filtered, prefix=col)
      expected = ['%s_%s' % (col, c) for c in cats]
      dummies = dummies.reindex(
        columns=expected, fill_value=0)
      df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
      if len(set(df.columns)) != len(df.columns):
        raise ValueError(
          'CategoricalEncoding OHE dummy name collision: '
          'duplicate dummy columns %r' % (list(df.columns),))
      continue
    mapping = {c: i for i, c in enumerate(cats)}
    df[col] = df[col].map(lambda v: mapping.get(v, -1))
  return df


def _fit_target_encode(df, args, variables):
  '''Fit target encoding group means per column.'''
  cols = _resolve_cols(args, variables.get('cat_cols', []))
  target = variables.get('target_col', 'target')
  state = {}
  if target not in df.columns:
    return state
  if pd.api.types.is_numeric_dtype(df[target]):
    _reject_inf_numeric(df, [target], 'TargetEncoding')
  global_mean = float(df[target].mean())
  if not _is_number(global_mean):
    raise ValueError(
      'TargetEncoding cannot fit target %r: non-finite global %r'
      % (target, global_mean))
  for col in cols:
    if col not in df.columns:
      continue
    mapping = df.groupby(col)[target].mean()
    pairs = [[_encode_typed(k), float(v)] for k, v in mapping.items()]
    for _key, val in pairs:
      if not _is_number(val):
        raise ValueError(
          'TargetEncoding cannot fit column %r: non-finite mapping '
          'value %r' % (col, val))
    state[col] = {
      'mapping': pairs, 'global': global_mean, 'target': target}
  for col, st in state.items():
    _assert_finite_state(st, 'TargetEncoding', col)
  return state


def _apply_target_encode(df, state):
  '''Apply stored group means, global mean fallback.'''
  for col, st in (state or {}).items():
    if col not in df.columns:
      continue
    raw = st.get('mapping') or []
    if isinstance(raw, dict):
      for key, val in raw.items():
        if not _is_number(val):
          raise ValueError(
            'Invalid TargetEncoding mapping for column %r: %r'
            % (col, raw))
      mapping = {_decode_typed(k): float(v) for k, v in raw.items()}
    else:
      pairs = []
      for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
          raise ValueError(
            'Invalid TargetEncoding mapping pair for column %r: %r'
            % (col, pair))
        if not _is_number(pair[1]):
          raise ValueError(
            'Invalid TargetEncoding mapping pair for column %r: %r'
            % (col, pair))
        pairs.append(pair)
      mapping = {_decode_typed(k): float(v) for k, v in pairs}
    if 'global' in st:
      global_mean = _finite_float(
        st.get('global'), col, 'TargetEncoding global')
    else:
      global_mean = 0.0
    df[col] = df[col].map(
      lambda v: mapping.get(v, global_mean)).astype(float)
  return df


def _fit_cap(df, args, variables):
  '''Fit cap outlier lo/hi per column.'''
  cols = _resolve_cols(args, variables.get('num_cols', []))
  bound = args.get('bound') or {}
  if not isinstance(bound, dict):
    raise ValueError('Invalid bound for CapOutliers: %r' % (bound,))
  lower = _validate_quantile_bound(
    'lower', bound.get('lower', 0.01))
  upper = _validate_quantile_bound(
    'upper', bound.get('upper', 0.99))
  if lower > upper:
    raise ValueError(
      'Invalid bound for CapOutliers: lower > upper (%r > %r)'
      % (lower, upper))
  custom = args.get('custom_bound') or {}
  if not isinstance(custom, dict):
    raise ValueError(
      'Invalid custom_bound for CapOutliers: %r' % (custom,))
  _reject_inf_numeric(df, cols, 'CapOutliers')
  state = {}
  for col in cols:
    if col not in df.columns:
      continue
    if col in custom:
      lo, hi = _validate_pair(col, custom[col])
    else:
      if not df[col].notna().any():
        raise ValueError(
          'CapOutliers cannot fit all-NaN column %r' % (col,))
      try:
        lo = df[col].quantile(lower)
        hi = df[col].quantile(upper)
      except Exception as exc:
        raise ValueError(
          'CapOutliers cannot fit column %r: %s' % (col, exc)
        ) from exc
      if not _is_number(lo) or not _is_number(hi):
        raise ValueError(
          'CapOutliers cannot fit column %r: bounds %r, %r'
          % (col, lo, hi))
      lo, hi = float(lo), float(hi)
      if np.isnan(lo) or np.isnan(hi):
        raise ValueError(
          'CapOutliers cannot fit all-NaN column %r' % (col,))
      if lo > hi:
        raise ValueError(
          'Invalid bound pair for column %r: lo > hi (%r > %r)'
          % (col, lo, hi))
    state[col] = {'lo': _native(float(lo)), 'hi': _native(float(hi))}
  return state


def _apply_cap(df, state):
  '''Apply stored lo/hi clip.'''
  for col, st in (state or {}).items():
    if col in df.columns:
      st = st or {}
      lo, hi = st.get('lo'), st.get('hi')
      if lo is not None and hi is not None:
        if not _is_number(lo) or not _is_number(hi):
          raise ValueError(
            'Invalid CapOutliers bounds for column %r: %r'
            % (col, st))
        if float(lo) > float(hi):
          raise ValueError(
            'Invalid bound pair for column %r: lo > hi (%r > %r)'
            % (col, lo, hi))
      df[col] = df[col].clip(lower=lo, upper=hi)
  return df


_FIT = {
  'Imputations': _fit_impute,
  'Normalizations': _fit_normalize,
  'QuantileDiscretizer': _fit_discretize,
  'HighLabelBinning': _fit_bin_labels,
  'CategoricalEncoding': _fit_encode_cat,
  'TargetEncoding': _fit_target_encode,
  'CapOutliers': _fit_cap,
}

_APPLY = {
  'Imputations': _apply_impute,
  'Normalizations': _apply_normalize,
  'QuantileDiscretizer': _apply_discretize,
  'HighLabelBinning': _apply_bin_labels,
  'CategoricalEncoding': _apply_encode_cat,
  'TargetEncoding': _apply_target_encode,
  'CapOutliers': _apply_cap,
}

FITTED_STEP_NAMES = frozenset(_FIT)
FITTED_STEPS = FITTED_STEP_NAMES


def fit_preprocess(df, prep_params, variables):
  '''Fit step state on df and return transformed df + registry.

  Args:
    df: Train dataframe.
    prep_params: List of step dicts as in preprocessing.yml.
    variables: Dict with column groups and target.

  Returns:
    Tuple of (transformed df, registry dict) where registry
    is ``{step_name: {col: state}}`` JSON-serializable.
  '''
  variables = variables or {}
  df = df.copy()
  registry = {}
  seen = set()
  for name, skip, args in _iter_steps(prep_params):
    if skip:
      continue
    if name in seen:
      raise ValueError(
        'Duplicate preprocessing step name: %s' % name)
    seen.add(name)
    fit_fn = _FIT.get(name)
    if fit_fn is None:
      raise ValueError(
        'Unsupported preprocessing step for fitted '
        'preprocessing: %s (only %s are fitted)'
        % (name, sorted(_FIT)))
    state = fit_fn(df, args, variables)
    _assert_finite_state(state, name)
    registry[name] = state
    df = _APPLY[name](df, state)
  registry_out = {
    'version': FITTED_REGISTRY_VERSION,
    '__version__': FITTED_REGISTRY_VERSION,
    'schema': REGISTRY_SCHEMA,
    '__schema__': REGISTRY_SCHEMA,
    '__meta__': {
      'version': FITTED_REGISTRY_VERSION,
      'schema': REGISTRY_SCHEMA,
      'typed_categories': True,
    },
  }
  registry_out.update(registry)
  return df, _jsonable(registry_out)


def apply_preprocess(df, registry):
  '''Replay fitted registry state verbatim on df.

  Args:
    df: Scoring dataframe.
    registry: Dict from :func:`fit_preprocess` (or JSON
      round-trip of it).

  Returns:
    Transformed dataframe copy.
  '''
  df = df.copy()
  if isinstance(registry, str):
    registry = registry_from_json(registry) if registry else {}
  steps = _registry_steps(registry)
  _check_legacy_typed_data(df, registry, steps)
  for name, state in steps.items():
    fn = _APPLY.get(name)
    if fn is None:
      raise ValueError(
        'Unsupported preprocessing step in fitted_registry: %s '
        '(only %s are fitted)' % (name, sorted(_APPLY)))
    df = fn(df, state)
  return df


def registry_to_json(registry):
  '''Serialize registry to a JSON string.'''
  return json.dumps(_jsonable(registry or {}), allow_nan=False)


def registry_from_json(text):
  '''Deserialize registry from a JSON string.'''
  return json.loads(text)
