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

import importlib
import inspect
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from forecast_ml_framework.core import spark_factory

LOG = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 1000
DEFAULT_MAX_ROWS = 100_000
DEFAULT_MAX_COLS = 1000
DEFAULT_PARTITION_BATCH_SIZE = 10_000

_ALLOWED_MODEL_MODULES = (
  'sklearn',
  'xgboost',
  'lightgbm',
  'catboost',
  'h2o',
  'forecast_ml_framework',
)


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


def _resolve_handle_strict(handle):
  '''Resolve a dotted handle under the model-loader allowlist.

  Args:
    handle: Dotted importable string, e.g.
      ``sklearn.ensemble.RandomForestClassifier``.

  Returns:
    The estimator-like class named by handle.

  Raises:
    ValueError: If handle is not a dotted string, its top-level
      module is not allowlisted, it cannot be imported, or it
      does not name a class with ``fit`` and ``predict``.
  '''
  if not isinstance(handle, str) or '.' not in handle:
    raise ValueError(
        'expected a dotted importable string, got %r' % (handle,))
  top_level = handle.split('.', 1)[0]
  if top_level not in _ALLOWED_MODEL_MODULES:
    raise ValueError(
        'model_handle %r is not allowlisted: top-level module %r '
        'is not one of %s'
        % (handle, top_level, ', '.join(_ALLOWED_MODEL_MODULES)))
  module_path, _, attr = handle.rpartition('.')
  try:
    module = importlib.import_module(module_path)
  except ImportError as exc:
    raise ValueError(
        'cannot resolve handle %r: module %r failed to import'
        % (handle, module_path)) from exc
  try:
    obj = getattr(module, attr)
  except AttributeError as exc:
    raise ValueError(
        'cannot resolve handle %r: module %r has no %r'
        % (handle, module_path, attr)) from exc
  if not inspect.isclass(obj) or not (
          hasattr(obj, 'fit') and hasattr(obj, 'predict')):
    raise ValueError(
        'model_handle %r must resolve to an estimator-like class '
        'with fit and predict, got %r' % (handle, obj))
  return obj


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


def _score_pandas_chunk(df, model, params, id_cols, chunk_size):
  '''Score one pandas chunk, returning a predictions frame.

  Args:
    df: Pandas chunk.
    model: Fitted model exposing ``predict``.
    params: Scoring params dict.
    id_cols: Identifier columns present in ``df``.
    chunk_size: Rows per ``model.predict`` call.

  Returns:
    Predictions frame for this chunk (same schema as
    :func:`predict` output).
  '''
  features = _feature_cols(df, params, id_cols)
  x = df[features] if features else df.drop(columns=id_cols)
  try:
    x = x.astype(float)
  except (TypeError, ValueError):
    x = pd.get_dummies(x, columns=[
      c for c in x.columns if not pd.api.types.is_numeric_dtype(x[c])])
  for col in x.columns:
    values = np.asarray(x[col], dtype=float)
    if not np.isfinite(values).all():
      raise ValueError(
          'non-finite feature values in column %r' % (col,))
  pred_parts = []
  proba_parts = []
  is_classification = (
      params.get('task_type', 'classification') == 'classification'
      and hasattr(model, 'predict_proba'))
  for start in range(0, len(x), chunk_size):
    chunk = x.iloc[start:start + chunk_size]
    pred_parts.append(np.asarray(model.predict(chunk)))
    if is_classification:
      proba_parts.append(np.max(
          np.asarray(model.predict_proba(chunk), dtype=float), axis=1))
  preds = np.concatenate(pred_parts) if pred_parts else np.array([])
  proba = (np.concatenate(proba_parts) if proba_parts
           else np.full(len(df), np.nan, dtype=float))
  out = df[id_cols].copy() if id_cols else pd.DataFrame(index=df.index)
  out['prediction'] = preds
  out['proba'] = proba
  out['score_model_id'] = (
      params.get('score_model_id') or params.get('model_id')
      or params.get('model_name') or params.get('alias') or 'unknown')
  out['scored_at'] = pd.Timestamp(datetime.now(timezone.utc))
  return out


def _empty_predictions(params):
  '''Build the empty predictions frame (schema only).'''
  id_cols = list(params.get('id_cols') or ['id'])
  return pd.DataFrame({
    **{c: pd.Series([], dtype=object) for c in id_cols},
    'prediction': pd.Series([], dtype=float),
    'proba': pd.Series([], dtype=float),
    'score_model_id': pd.Series([], dtype=object),
    'scored_at': pd.Series([], dtype=object),
  })


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
  """Score data with the loaded model.

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
    ``prediction``, ``proba`` (probability of the predicted
    class, NaN for regression/no proba), ``score_model_id``,
    and ``scored_at``. In real Spark, ``mapPartitions`` may
    return a Spark object, which is returned unchanged for
    the catalog to write (e.g. to BigQuery). Empty input
    returns an empty frame with that schema plus a warning,
    never a crash.
  """
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
      """Score one Spark partition in sub-chunks."""
      pandas_conversion_params = _pandas_conversion_params(params)
      part = (part if isinstance(part, pd.DataFrame)
              else spark_factory.to_pandas(part, pandas_conversion_params))
      if part.empty:
        return part
      part_ids = [c for c in id_cols if c in part.columns]
      return _score_pandas_chunk(part, model, params, part_ids,
                                 chunk_size)

    mapper = getattr(data, 'mapPartitions', None)
    if mapper is None:
      mapper = data.rdd.mapPartitions
    scored = mapper(_score_partition)
    if isinstance(scored, pd.DataFrame):
      return scored
    return scored
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
