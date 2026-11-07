#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Batch scoring nodes: load scoring data, predict, write.

Scheduled-ready design: the Kedro catalog owns all IO. The
input dataset (``scoring_input``) may be backed by a core
:class:`SQLQueryDataSet` / :class:`SparkSQLDataSet` (BigQuery
via GCP, Athena via AWS) or any ``kedro_datasets`` dataset;
the output dataset (``predictions``) is written by the catalog
after the :func:`write_predictions` node returns. No cloud
credentials or connection strings live here -- they come from
catalog/params configuration, never hardcoded.

Model loading is file/registry-agnostic: a callable loader is
injected via the ``model_loader`` argument or
``params['model_loader']`` / ``params['model']``. A callable
returning a model (e.g. ``FileModelLoader(path)`` or a
registry loader) is called once; an object that already has
``predict`` is used as-is. ``params`` may instead name a file
(``model_path``) or a dotted ``model_handle``; those are
resolved lazily inside :func:`predict` only. String handles
are allowlist-checked (see :data:`_ALLOWED_MODEL_MODULES`):
only estimator-like classes (with ``fit``/``predict``) from
known ML modules may be resolved, never ``os``/``subprocess``/
``builtins``.

Scoring guards: prediction is chunked by ``batch_size``
(default 1000), capped by ``max_rows`` (default 100000) /
``max_cols`` (default 1000), and rejects non-finite feature
values with a ``ValueError`` naming the offending column.

Cloud scaling (100M rows x 500 cols): caps are env/config
driven -- ``params['max_rows']`` / ``params['max_cols']``
win, else ``SCORING_MAX_ROWS`` / ``SCORING_MAX_COLS`` env,
else the local-safe defaults above. ``max_rows=None`` (null
in YAML) is unbounded and is only allowed when
``spark_scoring.enabled`` is true; with Spark scoring
disabled it raises ``ValueError`` instead of scoring an
unbounded pandas frame. When ``params['spark_scoring']['enabled']``
is true and the input is a Spark DataFrame (exposes
``mapPartitions``), scoring runs partitioned
(mapPartitions-style): each partition is scored in chunks of
``partition_batch_size`` rows (default 10000, falling back to
``batch_size``) WITHOUT calling ``toPandas()`` on the driver,
so there is no driver OOM. Pandas input keeps the existing
chunked-by-``batch_size`` path.
'''

import logging
import math
import os
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from forecast_ml_framework.core import spark_factory
from forecast_ml_framework.core.handle import (
  ALLOWED_MODEL_MODULES, resolve_handle_strict)
from forecast_ml_framework.core.predictions import (
  check_predictions_finite, materialize_predictions)

_check_predictions_finite = check_predictions_finite
_materialize_predictions = materialize_predictions

LOG = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 1000
DEFAULT_MAX_ROWS = 100_000
DEFAULT_MAX_COLS = 1000
DEFAULT_PARTITION_BATCH_SIZE = 10_000

# Compat alias: the allowlist lives in core.handle (R1).
_ALLOWED_MODEL_MODULES = ALLOWED_MODEL_MODULES


def _is_classification(params):
  '''Return True when params task_type normalizes to classification.'''
  from forecast_ml_framework.core.handle import normalize_task_kind
  return normalize_task_kind(
      (params or {}).get('task_type', 'classification')) == (
          'classification')


def _model_threshold(params):
  '''Validate and return ``model_threshold`` (default 0.5).

  Args:
    params: Scoring params dict.

  Returns:
    Threshold as float in [0, 1].

  Raises:
    ValueError: If the threshold is non-numeric, non-finite,
      or outside [0, 1].
  '''
  value = (params or {}).get('model_threshold', 0.5)
  if value is None:
    return 0.5
  if isinstance(value, bool):
    raise ValueError(
        f'model_threshold must be numeric in [0, 1], got {value!r}')
  try:
    threshold = float(value)
  except (TypeError, ValueError):
    raise ValueError(
        f'model_threshold must be numeric in [0, 1], got {value!r}'
    ) from None
  if not np.isfinite(threshold) or not 0 <= threshold <= 1:
    raise ValueError(
        f'model_threshold must be numeric in [0, 1], got {value!r}')
  return threshold


def _pandas_conversion_params(params):
  '''Extract pandas conversion params from scoring params.

  Args:
    params: Scoring params dict.

  Returns:
    Dict for ``spark_factory.to_pandas`` (empty dict enforces
    the default pandas limit).
  '''
  params = params or {}
  nested = params.get('pandas_conversion_params')
  if isinstance(nested, dict):
    return nested
  nested = params.get('pandas_conversion')
  if isinstance(nested, dict):
    return nested
  return {}


def _spark_scoring_enabled(params):
  '''Return True when partitioned Spark scoring is enabled.

  Args:
    params: Scoring params dict; ``spark_scoring`` may be a
      bool or a mapping with ``enabled``.

  Returns:
    True when Spark scoring was requested.
  '''
  params = params or {}
  flag = params.get('spark_scoring')
  if isinstance(flag, dict):
    return bool(flag.get('enabled'))
  if flag is not None:
    return bool(flag)
  return bool(params.get('spark_scoring_enabled'))


def _partition_batch_size(params):
  '''Resolve rows per partition chunk for Spark scoring.

  Args:
    params: Scoring params dict.

  Returns:
    Positive int chunk size.
  '''
  params = params or {}
  flag = params.get('spark_scoring')
  candidates = [params.get('partition_batch_size')]
  if isinstance(flag, dict):
    candidates.append(flag.get('partition_batch_size'))
  candidates.append(params.get('batch_size'))
  for value in candidates:
    if value is not None:
      size = int(value)
      if size <= 0:
        raise ValueError(
          'partition_batch_size must be a positive int, got %r'
          % (value,))
      return size
  return DEFAULT_PARTITION_BATCH_SIZE


def _is_spark_partitionable(df):
  '''Return True for Spark-like frames exposing mapPartitions.

  Args:
    df: Candidate dataframe.

  Returns:
    True when ``df`` has ``mapPartitions`` directly or via
    ``rdd.mapPartitions`` (and is not a pandas frame).
  '''
  if isinstance(df, pd.DataFrame):
    return False
  if hasattr(df, 'mapPartitions'):
    return True
  rdd = getattr(df, 'rdd', None)
  return rdd is not None and hasattr(rdd, 'mapPartitions')


def _cap_value(params, key, env_key, default):
  '''Resolve a cap: params win, then env, then default.

  A params value of ``None`` means unbounded (returned as
  ``None``). Env values ``null`` / ``none`` / ``''`` also mean
  unbounded; other env values are parsed as ints.

  Args:
    params: Scoring params dict.
    key: Cap key in params.
    env_key: Environment variable name.
    default: Local-safe default when neither is set.

  Returns:
    Int cap, or None when unbounded.
  '''
  if key in params:
    value = params[key]
    return None if value is None else int(value)
  env = os.environ.get(env_key)
  if env is not None:
    text = str(env).strip().lower()
    if text in ('', 'null', 'none'):
      return None
    return int(text)
  return default


def _row_count(df):
  '''Return row count without converting Spark frames.

  Args:
    df: Pandas or Spark-like dataframe.

  Returns:
    Row count via ``len`` or ``count()``.
  '''
  if isinstance(df, pd.DataFrame):
    return len(df)
  counter = getattr(df, 'count', None)
  if callable(counter):
    return int(counter())
  return len(df)


# Canonical resolver lives in core.handle (R1 layering); this
# name is kept as a compat re-export so existing imports of
# ``scoring.nodes._resolve_handle_strict`` keep working.
_resolve_handle_strict = resolve_handle_strict  # pylint: disable=invalid-name


def resolve_handle(handle):
  '''Resolve a dotted handle under the model-loader allowlist.

  Compat wrapper (F7): the canonical implementation lives in
  ``core.handle.resolve_handle_strict`` (R1 layering) so
  there is a single owner of the policy; the private name
  is kept as an alias for backwards compatibility.

  Args:
    handle: Dotted importable string, e.g.
      ``sklearn.ensemble.RandomForestClassifier``.

  Returns:
    The estimator-like class named by handle.

  Raises:
    ValueError: If handle is not allowlisted, cannot be
      resolved, or is not an estimator-like class.
  '''
  return _resolve_handle_strict(handle)


def _check_caps(df, params):
  '''Enforce max_rows / max_cols caps on a frame.

  Caps are env/config driven (see :func:`_cap_value`); a
  ``max_rows`` of ``None`` is unbounded and is only allowed
  with ``spark_scoring.enabled`` true. Spark frames are
  counted via ``count()`` / ``columns`` without
  ``toPandas()``.

  Raises:
    ValueError: If the frame exceeds either cap, or if
      ``max_rows`` is ``None`` with Spark scoring disabled.
  '''
  params = params or {}
  max_rows = _cap_value(
      params, 'max_rows', 'SCORING_MAX_ROWS', DEFAULT_MAX_ROWS)
  if max_rows is None and not _spark_scoring_enabled(params):
    raise ValueError(
        'max_rows=None (unbounded) is only allowed with '
        'spark_scoring.enabled=true; set max_rows or enable '
        'Spark scoring instead of scoring an unbounded '
        'pandas frame')
  max_cols = _cap_value(
      params, 'max_cols', 'SCORING_MAX_COLS', DEFAULT_MAX_COLS)
  if max_rows is not None:
    rows = _row_count(df)
    if rows > max_rows:
      raise ValueError(
          'scoring input exceeds max_rows=%d: got %d rows'
          % (max_rows, rows))
  if max_cols is not None and len(df.columns) > max_cols:
    raise ValueError(
        'scoring input exceeds max_cols=%d: got %d columns'
        % (max_cols, len(df.columns)))


def _registry_loader(params):
  '''Build a RegistryModelLoader from model_name/alias params.

  Args:
    params: Scoring params dict with ``model_name`` and
      optional ``alias`` / ``registry_uri`` (or
      ``tracking_uri``).

  Returns:
    A RegistryModelLoader instance (callable returning the
    loaded model).
  '''
  params = params or {}
  model_name = params.get('model_name') or params.get(
      'registered_model_name')
  alias = params.get('alias') or params.get('model_alias') \
      or 'production'
  model_uri = params.get('model_uri') or ''
  if not model_name and str(model_uri).startswith('models:/'):
    remainder = str(model_uri)[len('models:/'):].lstrip('/')
    model_name, _, uri_alias = remainder.partition('@')
    if uri_alias:
      alias = uri_alias
  registry_uri = params.get('registry_uri') or params.get(
      'tracking_uri') or params.get('registry') or params.get(
      'mlflow_tracking_uri')
  loader_cls = globals().get('RegistryModelLoader')
  if loader_cls is None:
    from forecast_ml_framework.serving.model_loader import (
      RegistryModelLoader,
    )
    loader_cls = RegistryModelLoader
  try:
    return loader_cls(registry_uri, model_name, alias=alias)
  except TypeError:
    try:
      return loader_cls(model_name, alias=alias)
    except TypeError:
      return loader_cls(registry_uri, model_name, alias)


def _resolve_loader(params, model_loader):
  '''Resolve the injected model loader to a loaded model.

  Args:
    params: Scoring params dict (may hold model_loader/model,
      model_path, or model_handle).
    model_loader: Explicit loader argument, winning over params.

  Returns:
    A fitted model exposing ``predict`` (not yet called for
    empty inputs; callers guard that case).
  '''
  params = params or {}
  loader = model_loader
  if loader is None:
    loader = params.get('model_loader', params.get('model'))
  if loader is None and params.get('model_path'):
    from forecast_ml_framework.serving.model_loader import (
      FileModelLoader,
    )
    loader = FileModelLoader(str(params['model_path']))
  if loader is None and params.get('model_handle'):
    loader = _resolve_handle_strict(str(params['model_handle']))
  if loader is None and (
          params.get('model_name') or params.get('model_uri')):
    loader = _registry_loader(params)
  if loader is None:
    raise ValueError(
        'scoring requires a model loader: pass model_loader=..., '
        'params["model_loader"], params["model_path"], '
        'params["model_handle"], or params["model_name"]/'
        'params["alias"] for registry loading')
  if isinstance(loader, str):
    loader = _resolve_handle_strict(loader)
  if hasattr(loader, 'predict'):
    return loader
  if callable(loader):
    model = loader()
    if not hasattr(model, 'predict'):
      raise TypeError('model loader returned object without predict')
    return model
  if hasattr(loader, 'load'):
    model = loader.load()
    if not hasattr(model, 'predict'):
      raise TypeError('model loader returned object without predict')
    return model
  raise TypeError('Unsupported model loader: %r' % type(loader))


def _feature_cols(df, params, id_cols):
  '''Resolve feature columns for prediction.

  Args:
    df: Scoring dataframe.
    params: Scoring params dict.
    id_cols: Identifier columns to exclude.

  Returns:
    List of feature column names present in df.
  '''
  target = params.get('target_col', 'target')
  requested = params.get('feature_cols')
  if requested:
    return [c for c in requested if c in df.columns]
  excluded = set(id_cols) | {target}
  return [c for c in df.columns if c not in excluded]


def _ntile_labels(values, n):
  '''Assign 1-based ntile labels via qcut, fallback to 1s.

  Uses ``assign_ntile`` from core.handle (which re-exports
  modeling.ntile) when it has landed; otherwise a local
  qcut. Constant/short input falls back to all 1s, never a
  crash.
  '''
  scores = np.asarray(values, dtype=float)
  if len(scores) == 0:
    return np.array([], dtype=int)
  kind = 'decile' if n == 10 else 'centile'
  try:
    from forecast_ml_framework.core.handle import assign_ntile
    labels = np.asarray(assign_ntile(scores, kind, n), dtype=int)
    if len(labels) == len(scores) and len(labels):
      if labels.min() == 0:
        labels = labels + 1
      return labels
  except (ImportError, ValueError, TypeError, AttributeError) as exc:
    LOG.debug('_ntile_labels: assign_ntile fallback (%s)', exc)
  try:
    labels = pd.qcut(scores, q=min(n, max(2, len(scores) // 2)),
                     labels=False, duplicates='drop')
    if labels is None or len(labels) != len(scores):
      raise ValueError('qcut failed')
    return np.asarray(labels, dtype=int) + 1
  except (ValueError, TypeError):
    return np.ones(len(scores), dtype=int)


def _positive_class_proba(raw):
  '''Extract binary positive-class probabilities (ML-19).

  Only a binary ``(n, 2)`` array (column 1) or a 1-D
  per-row array is a valid positive-class proba. A 1-D
  array is used as-is with a warning, never reduced
  through ``max`` (which would collapse to a scalar).
  Any other shape -- ``(n, 1)``, multiclass ``(n, >2)``,
  or other ndim -- warns as non-binary and falls back
  to row-wise ``max`` so multiclass output is never
  silently mistaken for binary column 1.

  Args:
    raw: ``predict_proba`` output (array-like).

  Returns:
    1-D array of positive-class probabilities.
  '''
  arr = np.asarray(raw, dtype=float)
  if arr.size == 0:
    return np.asarray(
        raw, dtype=float).reshape(0) if arr.ndim != 1 else arr
  if arr.ndim == 2 and arr.shape[1] == 2:
    return arr[:, 1]
  if arr.ndim == 1:
    warnings.warn(
        'predict_proba returned 1-D probabilities; treating them '
        'as positive-class proba', stacklevel=3)
    return arr
  warnings.warn(
      'predict_proba shape %r is not binary; positive-class '
      'proba requires (n, 2) or 1-D, falling back to row-wise '
      'max' % (getattr(arr, 'shape', None),), stacklevel=3)
  return np.max(arr, axis=-1)


def _ntile_edges(scores, n):
  '''Return unique quantile edges for ``n`` tiles.

  Bounded: ``np.quantile`` over the driver-held scores (or
  ``approxQuantile`` edges for Spark) -- no full-frame collect.
  '''
  values = np.asarray(scores, dtype=float)
  values = values[np.isfinite(values)]
  if len(values) == 0 or n < 2:
    return np.array([-float('inf'), float('inf')])
  edges = np.unique(np.quantile(values, np.linspace(0, 1, n + 1)))
  if len(edges) < 2:
    return np.array([-float('inf'), float('inf')])
  return edges


def _labels_from_edges(scores, edges, n):
  '''Map scores to 1-based tiles from global edges.'''
  values = np.asarray(scores, dtype=float)
  edges = np.asarray(edges, dtype=float)
  if len(values) == 0:
    return np.array([], dtype=int)
  if len(edges) < 2:
    return np.ones(len(values), dtype=int)
  interior = edges[1:-1]
  labels = np.digitize(values, interior, right=False) + 1
  labels = np.clip(labels, 1, max(1, n))
  labels[~np.isfinite(values)] = 1
  return labels.astype(int)


def _calib_values(df, params):
  '''Return calibrated values when scoring params name them.

  Accepts ``calib_proba`` (and aliases) as a column name
  present in ``df``, an array-like of matching length, or
  ``True`` meaning look for a calib column by name.
  Returns None when no calib source is present.
  '''
  params = params or {}
  keys = ('calib_proba', 'calib_score', 'calibrated_proba',
          'calibrated_score', 'calib_col', 'calibrated_col',
          'calib')
  for key in keys:
    if key not in params:
      continue
    value = params[key]
    if isinstance(value, str):
      if value in df.columns:
        return np.asarray(df[value], dtype=float)
      continue
    if value is None or value is True:
      for col in ('calib_proba', 'calibrated_proba',
                  'calib_score', 'calibrated_score', 'calib'):
        if col in df.columns:
          return np.asarray(df[col], dtype=float)
      continue
    try:
      arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
      continue
    if arr.ndim == 1 and len(arr) == len(df):
      return arr
  return None


def _calibrate_probability(proba, real_rate, sample_rate):
  '''Correct positive-class proba for prior-rate shift.

  Prefers ``calibrate_probability`` from modeling.calibration
  (vz-high Task 6); falls back to the same formula locally when
  that module has not landed yet, so scoring still works.
  Empty frames lack ``calib_proba`` (see :func:`_empty_predictions`).
  '''
  try:
    from forecast_ml_framework.pipelines.modeling.calibration import (
      calibrate_probability,
    )
    return np.asarray(
        calibrate_probability(np.asarray(proba, dtype=float),
                              float(real_rate), float(sample_rate)),
        dtype=float)
  except ImportError:
    p = np.asarray(proba, dtype=float)
    real_rate = float(real_rate)
    sample_rate = float(sample_rate)
    if real_rate == 0 or sample_rate == 0 or real_rate == sample_rate:
      return p
    numerator = p * real_rate / sample_rate
    denominator = ((1.0 - p) * (1.0 - real_rate) / (1.0 - sample_rate)
                   + numerator)
    return np.clip(numerator / denominator, 0.0, 1.0)


def _ohe_with_caps(x, params):
  '''One-hot encode non-numeric columns with caps (F6).

  Args:
    x: Feature frame with mixed numeric/categorical columns.
    params: Scoring params dict for cap resolution.

  Returns:
    Numeric frame after ``astype(float)`` or one-hot
    encoding, with non-finite values rejected.

  Raises:
    ValueError: If caps are exceeded, dummy names collide,
      or feature values are non-finite.
  '''
  try:
    x = x.astype(float)
  except (TypeError, ValueError):
    x = _ohe_encode(x, params)
  for col in x.columns:
    values = np.asarray(x[col], dtype=float)
    if not np.isfinite(values).all():
      raise ValueError(
          'non-finite feature values in column %r' % (col,))
  return x


def _ohe_encode(x, params):
  '''Expand categorical columns via get_dummies with caps.

  Args:
    x: Feature frame that failed ``astype(float)``.
    params: Scoring params dict for cap resolution.

  Returns:
    One-hot encoded numeric frame.

  Raises:
    ValueError: If caps are exceeded or dummy names collide.
  '''
  cat_cols = [
    c for c in x.columns if not pd.api.types.is_numeric_dtype(x[c])]
  max_cols = _cap_value(
    params or {}, 'max_cols', 'SCORING_MAX_COLS', DEFAULT_MAX_COLS)
  if cat_cols and max_cols is not None:
    # Cardinality cap before expansion: get_dummies would
    # otherwise expand e.g. 2000 distinct categories into 2000
    # columns, bypassing the max_cols check done on the
    # pre-expansion frame (DoS amplification).
    numeric_cols = len(x.columns) - len(cat_cols)
    expanded = numeric_cols
    for col in cat_cols:
      distinct = int(x[col].nunique(dropna=False))
      if distinct > max_cols:
        raise ValueError(
          'scoring input exceeds max_cols=%d: categorical '
          'column %r has %d distinct values'
          % (max_cols, col, distinct))
      expanded += distinct
    if expanded > max_cols:
      raise ValueError(
        'scoring input exceeds max_cols=%d: get_dummies '
        'expansion would produce %d columns'
        % (max_cols, expanded))
  # Dummy-name collision check before expansion: distinct
  # (column, category) pairs (or a dummy vs an existing
  # column) can produce the same name, e.g. a/b_c and
  # a_b/c both expand to a_b_c; get_dummies would then
  # emit duplicate columns and silently misalign features.
  expected_dummies = []
  for col in cat_cols:
    for cat in x[col].unique():
      expected_dummies.append('%s_%s' % (col, cat))
  seen_dummies = set()
  for dummy in expected_dummies:
    if dummy in seen_dummies:
      raise ValueError(
        'scoring OHE dummy name collision: duplicate dummy '
        'column %r' % (dummy,))
    seen_dummies.add(dummy)
  existing = set(x.columns) - set(cat_cols)
  for dummy in expected_dummies:
    if dummy in existing:
      raise ValueError(
        'scoring OHE dummy name collision: dummy column %r '
        'collides with existing column %r' % (dummy, dummy))
  x = pd.get_dummies(x, columns=cat_cols)
  if len(set(x.columns)) != len(x.columns):
    raise ValueError(
      'scoring OHE dummy name collision: duplicate dummy '
      'columns %r' % (list(x.columns),))
  # Re-check caps after expansion.
  if max_cols is not None and len(x.columns) > max_cols:
    raise ValueError(
      'scoring input exceeds max_cols=%d: got %d columns '
      'after get_dummies expansion'
      % (max_cols, len(x.columns)))
  return x


def _align_to_fitted(x, model):
  '''Reindex ``x`` to the model's fitted feature order (F6).

  Appearance order after ``get_dummies`` may differ from
  the order the model was fitted on, which would silently
  misalign features positionally. Missing fitted features
  are filled with 0; extra columns are dropped.

  Args:
    x: Encoded feature frame.
    model: Fitted model exposing ``feature_names_in_``.

  Returns:
    Frame reindexed to fitted columns when available,
    otherwise unchanged.
  '''
  fitted_names = getattr(model, 'feature_names_in_', None)
  if fitted_names is not None:
    try:
      fitted_list = list(fitted_names)
    except TypeError:
      fitted_list = None
    if fitted_list:
      x = x.reindex(columns=fitted_list, fill_value=0)
  return x


def _score_pandas_chunk(df, model, params, id_cols, chunk_size,
                        defer_ntile=False):
  '''Score one pandas chunk, returning a predictions frame.

  Args:
    df: Pandas chunk.
    model: Fitted model exposing ``predict``.
    params: Scoring params dict.
    id_cols: Identifier columns present in ``df``.
    chunk_size: Rows per ``model.predict`` call.
    defer_ntile: When True, skip tile assignment (Spark first
      pass; a global second pass assigns tiles from the full
      score column so they are comparable across partitions).
      Calibrated values are stashed for that pass.

  Returns:
    Predictions frame for this chunk (same schema as
    :func:`predict` output, minus tiles when deferred).

  Threshold: when ``params['model_threshold']`` is present (not
  None) and the task is classification, ``prediction`` is
  ``(positive-class proba >= threshold).astype(int)`` for
  binary ``predict_proba`` output; ``proba`` (positive-class
  probability) is left unchanged. Multiclass or ``(n, 1)``
  proba output warns and falls back to ``model.predict``.
  ``high_score_ind`` uses the same ``>=`` threshold (ML-19),
  so ``proba == threshold`` gives prediction 1 and ``Y``
  when the tile is >= 5.
  When both ``real_rate`` and ``sample_rate`` are present, a
  ``calib_proba`` column corrected via ``calibrate_probability``
  is appended; ``proba``/``prediction`` are never mutated by
  that correction.
  '''
  features = _feature_cols(df, params, id_cols)
  x = df[features] if features else df.drop(columns=id_cols)
  x = _ohe_with_caps(x, params)
  x = _align_to_fitted(x, model)
  pred_parts = []
  proba_parts = []
  is_classification = (
      _is_classification(params)
      and hasattr(model, 'predict_proba'))
  params = params or {}
  threshold_active = (
      is_classification
      and params.get('model_threshold') is not None)
  threshold = _model_threshold(params) if threshold_active else None
  for start in range(0, len(x), chunk_size):
    chunk = x.iloc[start:start + chunk_size]
    if is_classification:
      raw_proba = materialize_predictions(model.predict_proba(chunk))
      _check_predictions_finite(raw_proba)
      proba_parts.append(_positive_class_proba(raw_proba))
      if threshold_active:
        arr = np.asarray(raw_proba, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2:
          pred_parts.append((arr[:, 1] >= threshold).astype(int))
        elif arr.ndim == 1 and arr.size:
          pred_parts.append((arr >= threshold).astype(int))
        else:
          warnings.warn(
              'model_threshold is binary-only; predict_proba shape '
              '%r is not binary, falling back to model.predict'
              % (getattr(arr, 'shape', None),), stacklevel=2)
          raw_pred = materialize_predictions(model.predict(chunk))
          _check_predictions_finite(raw_pred)
          pred_parts.append(np.asarray(raw_pred))
      else:
        raw_pred = materialize_predictions(model.predict(chunk))
        _check_predictions_finite(raw_pred)
        pred_parts.append(np.asarray(raw_pred))
    else:
      raw_pred = materialize_predictions(model.predict(chunk))
      _check_predictions_finite(raw_pred)
      pred_parts.append(np.asarray(raw_pred))
  preds = np.concatenate(pred_parts) if pred_parts else np.array([])
  proba = (np.concatenate(proba_parts) if proba_parts
           else np.full(len(df), np.nan, dtype=float))
  out = df[id_cols].copy() if id_cols else pd.DataFrame(index=df.index)
  out['prediction'] = preds
  out['proba'] = proba
  if (is_classification and params.get('real_rate') is not None
          and params.get('sample_rate') is not None):
    out['calib_proba'] = _calibrate_probability(
        proba, params['real_rate'], params['sample_rate'])
  score_values = proba if is_classification else preds
  try:
    score_arr = np.asarray(score_values, dtype=float)
  except (TypeError, ValueError):
    score_arr = np.asarray(preds, dtype=float)
  if len(score_arr) and np.isnan(score_arr).all():
    score_arr = np.asarray(preds, dtype=float)
  calib = _calib_values(df, params)
  if defer_ntile:
    if calib is not None:
      out['__calib_score'] = np.asarray(calib, dtype=float)
  else:
    out['score_decile'] = _ntile_labels(score_arr, 10)
    out['score_centile'] = _ntile_labels(score_arr, 100)
    if calib is not None:
      out['calib_decile'] = _ntile_labels(calib, 10)
      out['calib_centile'] = _ntile_labels(calib, 100)
    if is_classification:
      threshold = _model_threshold(params)
      out['high_score_ind'] = np.where(
          (np.asarray(score_arr, dtype=float) >= threshold)
          & (out['score_decile'] >= 5), 'Y', 'N')
  out['score_model_id'] = (
      params.get('score_model_id') or params.get('model_id')
      or params.get('model_name') or params.get('alias') or 'unknown')
  out['scored_at'] = pd.Timestamp(datetime.now(timezone.utc))
  return out


def _empty_predictions(params):
  '''Build the empty predictions frame (schema only).'''
  params = params or {}
  id_cols = list(params.get('id_cols') or ['id'])
  return pd.DataFrame({
    **{c: pd.Series([], dtype=object) for c in id_cols},
    'prediction': pd.Series([], dtype=float),
    'proba': pd.Series([], dtype=float),
    'score_decile': pd.Series([], dtype=int),
    'score_centile': pd.Series([], dtype=int),
    'calib_decile': pd.Series([], dtype=int),
    'calib_centile': pd.Series([], dtype=int),
    'high_score_ind': pd.Series([], dtype=object),
    'score_model_id': pd.Series([], dtype=object),
    'scored_at': pd.Series([], dtype=object),
  })


def _assign_global_ntiles_frame(scored, params):
  '''Reassign tiles globally on a driver-held predictions frame.

  Spark ``mapPartitions`` fakes return concatenated pandas, so
  the global tiles are recomputed here from the full ``proba``
  (classification) or ``prediction`` (regression) column --
  the same score that tiling and ``high_score_ind`` use.
  '''
  if scored is None or len(scored) == 0:
    return scored
  is_classification = _is_classification(params)
  score_arr = None
  if is_classification:
    try:
      score_arr = np.asarray(scored['proba'], dtype=float)
    except (KeyError, TypeError, ValueError):
      score_arr = None
    if score_arr is not None and np.isnan(score_arr).all():
      score_arr = None
  if score_arr is None:
    score_arr = np.asarray(scored['prediction'], dtype=float)
  scored['score_decile'] = _ntile_labels(score_arr, 10)
  scored['score_centile'] = _ntile_labels(score_arr, 100)
  if '__calib_score' in scored.columns:
    calib = np.asarray(scored['__calib_score'], dtype=float)
    scored['calib_decile'] = _ntile_labels(calib, 10)
    scored['calib_centile'] = _ntile_labels(calib, 100)
    scored.drop(columns=['__calib_score'], inplace=True)
  if is_classification:
    threshold = _model_threshold(params)
    scored['high_score_ind'] = np.where(
        (np.asarray(score_arr, dtype=float) >= threshold)
        & (scored['score_decile'] >= 5), 'Y', 'N')
  return _order_predictions_columns(scored, is_classification)


def _order_predictions_columns(scored, is_classification):
  '''Order predictions columns as the pandas path emits them.'''
  excluded = {
    'prediction', 'proba', 'score_decile', 'score_centile',
    'calib_decile', 'calib_centile', 'high_score_ind',
    'score_model_id', 'scored_at'}
  id_cols = [c for c in scored.columns if c not in excluded]
  canonical = list(id_cols) + [
    'prediction', 'proba', 'score_decile', 'score_centile']
  if 'calib_decile' in scored.columns:
    canonical += ['calib_decile', 'calib_centile']
  if is_classification and 'high_score_ind' in scored.columns:
    canonical += ['high_score_ind', 'score_model_id', 'scored_at']
  else:
    canonical += ['score_model_id', 'scored_at']
  ordered = [c for c in canonical if c in scored.columns]
  for col in scored.columns:
    if col not in ordered:
      ordered.append(col)
  return scored[ordered]


def _spark_global_reassign(scored, params):
  '''Second Spark pass: global tiles from approxQuantile edges.

  ``approxQuantile`` (bounded, approximate) needs no driver
  collect, so there is no OOM/large-collect path; edges are
  then broadcast to a second ``mapPartitions``. Frames without
  ``approxQuantile`` pass through unchanged (legacy fakes).
  '''
  approx = getattr(scored, 'approxQuantile', None)
  if not callable(approx):
    return scored
  is_classification = _is_classification(params)
  score_col = 'proba' if is_classification else 'prediction'
  columns = [c for c in (getattr(scored, 'columns', []) or [])]
  if score_col not in columns:
    return scored

  def _edges_for(col, n):
    '''Return percentile edges for col via approxQuantile.

    Args:
      col: Score column name.
      n: Number of tiles (edges are n + 1 percentiles).

    Returns:
      Sorted unique edge array, or [-inf, inf] when
      approxQuantile yields no usable edges.
    '''
    percentages = [i / n for i in range(n + 1)]
    try:
      raw = approx(col, percentages)
    except TypeError:
      raw = approx(col, percentages, 0.001)
    if raw is None:
      return np.array([-float('inf'), float('inf')])
    edges = np.unique(np.asarray(raw, dtype=float))
    if len(edges) < 2:
      return np.array([-float('inf'), float('inf')])
    return edges

  score_dec_edges = _edges_for(score_col, 10)
  score_cent_edges = _edges_for(score_col, 100)
  calib_dec_edges = _edges_for('__calib_score', 10) \
      if '__calib_score' in columns else None
  calib_cent_edges = _edges_for('__calib_score', 100) \
      if '__calib_score' in columns else None
  threshold = _model_threshold(params)

  def _assign_edges(part):
    '''Assign score/calibration tile labels to one partition.

    Args:
      part: One Spark partition (or pandas DataFrame).

    Returns:
      The partition with score/calibration tile columns
      added and the helper calibration score dropped.
    '''
    pandas_conversion_params = _pandas_conversion_params(params)
    part = (part if isinstance(part, pd.DataFrame)
            else spark_factory.to_pandas(part, pandas_conversion_params))
    if part.empty:
      return part
    values = np.asarray(part[score_col], dtype=float)
    part['score_decile'] = _labels_from_edges(values, score_dec_edges, 10)
    part['score_centile'] = _labels_from_edges(
        values, score_cent_edges, 100)
    if calib_dec_edges is not None and '__calib_score' in part.columns:
      calib = np.asarray(part['__calib_score'], dtype=float)
      part['calib_decile'] = _labels_from_edges(calib, calib_dec_edges, 10)
      part['calib_centile'] = _labels_from_edges(
          calib, calib_cent_edges, 100)
      part = part.drop(columns=['__calib_score'])
    if is_classification:
      part['high_score_ind'] = np.where(
          (values >= threshold) & (part['score_decile'] >= 5), 'Y', 'N')
    return _order_predictions_columns(part, is_classification)

  mapper = getattr(scored, 'mapPartitions', None)
  if mapper is None:
    rdd = getattr(scored, 'rdd', None)
    mapper = getattr(rdd, 'mapPartitions', None)
  if mapper is None:
    return scored
  return mapper(_assign_edges)


def load_scoring_data(data, params=None):
  '''Load scoring data from the catalog input dataset.

  Args:
    data: Catalog input (pandas or Spark-like dataframe).
    params: Scoring params dict (unused for loading; kept so
      the Kedro node can receive ``params:scoring_params``).

  Returns:
    Scoring data as a pandas DataFrame, or -- when
    ``spark_scoring`` is enabled and ``data`` is a Spark
    DataFrame -- the Spark frame unchanged (no ``toPandas()``,
    so no driver OOM). Empty input returns an empty frame
    with a warning, never a crash.
  '''
  params = params or {}
  if _spark_scoring_enabled(params) and _is_spark_partitionable(data):
    _check_caps(data, params)
    if _row_count(data) == 0:
      LOG.warning('load_scoring_data: empty scoring input')
    return data
  pandas_conversion_params = _pandas_conversion_params(params)
  df = spark_factory.to_pandas(data, pandas_conversion_params)
  if isinstance(data, pd.DataFrame):
    df = df.copy()
  _check_caps(df, params)
  if df.empty:
    LOG.warning('load_scoring_data: empty scoring input')
  return df


def predict(data, params=None, model_loader=None):
  '''Score data with the loaded model.

  Args:
    data: Scoring dataframe (pandas or Spark-like). When
      ``spark_scoring`` is enabled and ``data`` exposes
      ``mapPartitions``, scoring is partitioned: each Spark
      partition is scored in ``partition_batch_size``-row
      chunks WITHOUT ``toPandas()`` (no driver OOM); the
      model must be picklable/broadcastable by the caller in
      real Spark deployments.
    params: Scoring params dict with optional ``id_cols``
      (default ``['id']``), ``feature_cols``, ``task_type``,
      ``score_model_id`` / ``model_id``, ``batch_size``,
      ``partition_batch_size``, ``spark_scoring`` (bool or
      ``{'enabled': bool}``), and env/config caps
      ``max_rows`` / ``max_cols`` (``None`` = unbounded).
    model_loader: Injected callable loader or ready model;
      falls back to ``params['model_loader']`` etc. (see
      :func:`_resolve_loader`). Never called for empty input.

  Returns:
    Predictions dataframe with columns ``id_cols`` +
    ``prediction``, ``proba`` (positive-class probability, NaN
    for regression/no proba), ``score_model_id`` and
    ``scored_at``. In real Spark, ``mapPartitions`` may
    return a Spark object, which is returned unchanged for
    the catalog to write (e.g. to BigQuery). Empty input
    returns an empty frame with that schema plus a warning,
    never a crash.
  '''
  params = params or {}
  _check_caps(data, params)
  batch_size = int(params.get('batch_size', DEFAULT_BATCH_SIZE))
  if batch_size <= 0:
    raise ValueError(
        'batch_size must be a positive int, got %r' % (batch_size,))
  if _spark_scoring_enabled(params) and _is_spark_partitionable(data):
    if _row_count(data) == 0:
      LOG.warning('predict: empty scoring input, no rows scored')
      return _empty_predictions(params)
    model = _resolve_loader(params, model_loader)
    chunk_size = _partition_batch_size(params)
    id_cols = [c for c in (params.get('id_cols') or ['id'])
               if c in data.columns]

    def _score_partition(part):
      '''Score one Spark partition in sub-chunks.'''
      pandas_conversion_params = _pandas_conversion_params(params)
      part = (part if isinstance(part, pd.DataFrame)
              else spark_factory.to_pandas(part, pandas_conversion_params))
      if part.empty:
        return part
      part_ids = [c for c in id_cols if c in part.columns]
      return _score_pandas_chunk(part, model, params, part_ids,
                                 chunk_size, defer_ntile=True)

    mapper = getattr(data, 'mapPartitions', None)
    if mapper is None:
      mapper = data.rdd.mapPartitions
    scored = mapper(_score_partition)
    if isinstance(scored, pd.DataFrame):
      return _assign_global_ntiles_frame(scored, params)
    return _spark_global_reassign(scored, params)
  pandas_conversion_params = _pandas_conversion_params(params)
  df = spark_factory.to_pandas(data, pandas_conversion_params)
  if isinstance(data, pd.DataFrame):
    df = df.copy()
  _check_caps(df, params)
  if df.empty:
    LOG.warning('predict: empty scoring input, no rows scored')
    return _empty_predictions(params)
  id_cols = [c for c in (params.get('id_cols') or ['id'])
             if c in df.columns]
  model = _resolve_loader(params, model_loader)
  return _score_pandas_chunk(df, model, params, id_cols, batch_size)


def write_predictions(predictions, params=None, destination=None):
  '''Write predictions to the catalog output dataset.

  In a Kedro run the catalog writes the returned frame to
  the ``predictions`` dataset (BigQuery/Athena via
  ``kedro_datasets`` or a lazy core dataset), so by default
  this node is a pass-through. When a destination dataset
  object is injected (``destination`` argument or
  ``params['destination']`` with a ``save`` method) it is
  saved through that object instead; a read-only core
  dataset (``SQLQueryDataSet``/``SparkSQLDataSet`` raise
  ``NotImplementedError`` on save) warns and passes through
  so the catalog write still happens. Empty input warns,
  skips the save, and returns empty -- never a crash.

  Args:
    predictions: Predictions dataframe from :func:`predict`.
    params: Scoring params dict (may hold ``destination``).
    destination: Optional dataset object exposing ``save``.

  Returns:
    The predictions dataframe, unchanged.
  '''
  params = params or {}
  if _spark_scoring_enabled(params) and _is_spark_partitionable(
          predictions):
    return predictions
  pandas_conversion_params = _pandas_conversion_params(params)
  df = spark_factory.to_pandas(predictions, pandas_conversion_params)
  if isinstance(predictions, pd.DataFrame):
    df = df.copy()
  if df.empty:
    LOG.warning('write_predictions: empty predictions, nothing written')
    return df
  dest = destination
  if dest is None:
    dest = params.get('destination') or params.get(
      'destination_dataset') or params.get('target_dataset')
  if dest is not None and hasattr(dest, 'save'):
    try:
      dest.save(df)
    except NotImplementedError as exc:
      LOG.warning(
          'write_predictions: destination is read-only (%s); '
          'leaving the write to the catalog dataset', exc)
  return df
