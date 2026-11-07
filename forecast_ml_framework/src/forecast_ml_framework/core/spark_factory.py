#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Spark session creation with graceful local fallback.

Cloud scaling (100M rows x 500 cols):
- ``get_spark`` applies autoscaling-friendly defaults: dynamic
  allocation, Adaptive Query Execution (AQE), and shuffle
  partitions from config ``spark_params``. Cluster mode note:
  on Dataproc/EMR (``master`` = ``yarn`` / cluster manager) the
  driver runs on the cloud cluster; this factory only builds
  the session config -- it does not submit/ manage clusters
  (no submit automation, no credentials here).
- ``check_pandas_limits`` guards the pandas-conversion path
  (data_prep / modeling): frames larger than
  ``pandas_conversion_params.max_rows_for_pandas`` (default
  5,000,000 rows) raise an informative error directing to
  Spark-native feature engineering / distributed training
  (H2O DAI/Spark, XGBoost DMatrix distributed) instead of
  OOM-ing on the driver. ``to_pandas`` enforces the guard when
  ``config`` is supplied.
'''
# pylint: disable=import-outside-toplevel, broad-exception-caught

import logging
import os

LOG = logging.getLogger(__name__)

DEFAULT_MAX_ROWS_FOR_PANDAS = 5_000_000
DEFAULT_SHUFFLE_PARTITIONS = 200

_AUTOSCALING_DEFAULTS = {
  'spark.dynamicAllocation.enabled': 'true',
  'spark.sql.adaptive.enabled': 'true',
  'spark.sql.adaptive.coalescePartitions.enabled': 'true',
}

# Spark conf allowlist (cycle-1 F-11 / cycle-2 antisec R-10).
# Only keys that cannot load or execute arbitrary code are
# allowed: resource sizing (driver/executor memory + cores +
# instances), shuffle partitions, dynamic allocation, AQE
# (``spark.sql.adaptive.*``), and explicit query-tuning keys.
# Everything else -- ``spark.jars`` / ``*.jars.*``,
# ``*executorEnv*``, ``*extraClassPath*``,
# ``*extraJavaOptions*``, ``*classloader*``,
# ``*userClassPath*``, ``spark.kubernetes.*``, unknown keys --
# raises ``ValueError`` naming the key (fail-closed, never
# passed to ``SparkSession.builder.config``). Note the
# ``spark.sql.*`` surface is an explicit key list, not a
# prefix: a bare prefix would admit catalog/hive/provider
# keys that can load code.
_ALLOWED_SPARK_KEYS = frozenset({
  # Shipped conf (conf/base/parameters/platform.yml).
  'spark.sql.shuffle.partitions',
  'spark.driver.memory',
  'spark.driver.cores',
  'spark.executor.memory',
  'spark.executor.cores',
  'spark.executor.instances',
  # Documented query tuning (100M-row sizing, README/infra).
  'spark.sql.autoBroadcastJoinThreshold',
  'spark.sql.broadcastTimeout',
  'spark.sql.codegen.wholeStage',
  'spark.sql.files.maxPartitionBytes',
  'spark.sql.files.openCostInBytes',
})

_ALLOWED_SPARK_PREFIXES = (
  'spark.sql.adaptive.',
  'spark.dynamicAllocation.',
)

# Keys that are not Spark conf entries: session wiring handled
# elsewhere (master/app name/enabled) or aliases handled by
# ``spark_builder_config`` itself (shuffle/dynamic allocation).
_NON_CONF_KEYS = frozenset({
  'shuffle_partitions',
  'enabled',
  'master',
  'app_name',
  'config',
  'spark_params',
  'spark_parameters',
  'dynamic_allocation',
})


def _is_allowed_spark_key(key):
  '''Return True when a Spark conf key is on the allowlist.

  Args:
    key: Candidate conf key.

  Returns:
    True for exact allowlist keys and for the
    ``spark.sql.adaptive.*`` / ``spark.dynamicAllocation.*``
    tuning prefixes, else False.
  '''
  if not isinstance(key, str):
    return False
  if key in _ALLOWED_SPARK_KEYS:
    return True
  return key.startswith(_ALLOWED_SPARK_PREFIXES)


def _validate_spark_key(key):
  '''Reject a Spark conf key that is not on the allowlist.

  Args:
    key: Candidate conf key.

  Raises:
    ValueError: Naming the rejected key; the caller must not
      pass it to Spark (fail-closed, never merge-then-filter).
  '''
  if not _is_allowed_spark_key(key):
    raise ValueError(
      f'spark conf key {key!r} is not in the allowlist; only '
      'shipped/tuning keys (shuffle, dynamic allocation, '
      'spark.sql.adaptive.*, driver/executor sizing, explicit '
      'query tuning) are allowed')


def _apply_conf_source(merged, source):
  '''Validate and merge one conf source dict into ``merged``.

  Unwraps the shipped ``spark_params.spark`` nested shape and
  the empty ``jar_packages`` placeholder; a non-empty
  ``jar_packages`` is rejected like any other code-loading
  key. Non-conf wiring keys are skipped.

  Args:
    merged: Destination conf dict (mutated in place).
    source: Candidate conf mapping.

  Raises:
    ValueError: Naming the first non-allowlisted key.
  '''
  if not isinstance(source, dict):
    return
  for key, value in source.items():
    if key in _NON_CONF_KEYS:
      continue
    if key == 'spark' and isinstance(value, dict):
      _apply_conf_source(merged, value)
      continue
    if key == 'jar_packages':
      if value:
        raise ValueError(
          f'spark conf key {key!r} is not in the allowlist; '
          'arbitrary packages are not allowed')
      continue
    _validate_spark_key(key)
    merged[key] = value


def _spark_section(config):
  '''Extract the Spark configuration mapping.

  Args:
    config: Raw configuration mapping.

  Returns:
    The nested ``spark`` mapping, or ``config`` itself when it
    already holds Spark keys, else an empty dict.
  '''
  if not isinstance(config, dict):
    return {}
  nested = config.get('spark')
  if isinstance(nested, dict):
    return nested
  if isinstance(nested, bool):
    return {'enabled': nested}
  return config


def _spark_enabled(config):
  '''Decide whether Spark execution was requested.

  Spark runs only when explicitly enabled (``enabled`` /
  ``spark_enabled`` / ``use_spark`` truthy) or a Spark section
  carrying other keys is supplied. Empty config means disabled.

  Args:
    config: Raw configuration mapping.

  Returns:
    True when Spark was requested.
  '''
  if not isinstance(config, dict) or not config:
    return False
  section = _spark_section(config)
  for key in ('enabled', 'spark_enabled', 'use_spark'):
    if key in section:
      return bool(section[key])
    if key in config:
      return bool(config[key])
  if section is not config:
    return bool(section)
  return 'spark' in config and bool(config.get('spark'))


def _spark_params(config, section):
  '''Merge ``spark_params`` from section / top-level config.

  Args:
    config: Raw configuration mapping.
    section: Nested spark section.

  Returns:
    Dict of spark_params (empty when absent).
  '''
  for source in (section, config if isinstance(config, dict) else {}):
    for key in ('spark_params', 'spark_parameters'):
      value = source.get(key) if isinstance(source, dict) else None
      if isinstance(value, dict):
        return dict(value)
  return {}


def spark_builder_config(config=None):
  '''Return the Spark conf dict ``get_spark`` will apply.

  Autoscaling-friendly defaults (dynamic allocation + AQE) are
  applied first, then ``shuffle_partitions`` from
  ``spark_params`` (default 200), then explicit ``config`` /
  ``spark_params`` entries override. Cloud cluster mode
  (Dataproc/EMR, ``master='yarn'``) uses the same config; only
  session creation happens here.

  Only allowlisted keys (see ``_ALLOWED_SPARK_KEYS`` /
  ``_ALLOWED_SPARK_PREFIXES``) may be supplied; any other key
  from ``config`` raises ``ValueError`` naming it, so code
  loading conf (``spark.jars``, ``executorEnv``,
  ``extraClassPath``/``extraJavaOptions``, ``classloader``,
  ``userClassPath``, ``spark.kubernetes.*``) never reaches
  ``SparkSession.builder.config`` (cycle-1 F-11 / R-10).

  Args:
    config: Raw configuration mapping.

  Returns:
    Dict mapping Spark conf keys to values.

  Raises:
    ValueError: Naming a supplied conf key that is not on the
      allowlist.
  '''
  section = _spark_section(config)
  params = _spark_params(config, section)
  merged = dict(_AUTOSCALING_DEFAULTS)
  shuffle = (
    params.get('shuffle_partitions')
    or section.get('shuffle_partitions')
    or params.get('spark.sql.shuffle.partitions')
    or section.get('spark.sql.shuffle.partitions')
    or DEFAULT_SHUFFLE_PARTITIONS)
  merged['spark.sql.shuffle.partitions'] = shuffle
  if isinstance(section.get('dynamic_allocation'), bool):
    merged['spark.dynamicAllocation.enabled'] = (
      'true' if section['dynamic_allocation'] else 'false')
  for source in (section, section.get('config'), params):
    _apply_conf_source(merged, source)
  return merged


def _pandas_limit_from_config(config):
  '''Resolve ``max_rows_for_pandas`` from nested/flat config.

  Args:
    config: Raw configuration mapping.

  Returns:
    Row threshold (int, default 5,000,000).
  '''
  if not isinstance(config, dict):
    return DEFAULT_MAX_ROWS_FOR_PANDAS
  for key in ('max_rows_for_pandas', 'max_rows'):
    if key in config and config[key] is not None:
      return int(config[key])
  for section_key in ('pandas_conversion_params',
                      'pandas_conversion', 'spark'):
    nested = config.get(section_key)
    if isinstance(nested, dict):
      for key in ('max_rows_for_pandas', 'max_rows'):
        if key in nested and nested[key] is not None:
          return int(nested[key])
      if section_key == 'pandas_conversion_params':
        return DEFAULT_MAX_ROWS_FOR_PANDAS
  env = os.environ.get('PANDAS_MAX_ROWS')
  if env:
    try:
      return int(env)
    except ValueError:
      return DEFAULT_MAX_ROWS_FOR_PANDAS
  return DEFAULT_MAX_ROWS_FOR_PANDAS


def _frame_row_count(df):
  '''Return row count for pandas/Spark-like frames, else None.'''
  try:
    import pandas as pd
    if isinstance(df, pd.DataFrame):
      return len(df)
  except (ImportError, TypeError, ValueError, AttributeError) as exc:
    LOG.debug('_frame_row_count: pandas probe skipped (%s)',
        type(exc).__name__)
  count = getattr(df, 'count', None)
  if callable(count):
    try:
      return int(count())
    except (RuntimeError, TypeError, ValueError,
            AttributeError) as exc:
      LOG.debug('_frame_row_count: count() unreadable (%s)',
          type(exc).__name__)
      return None
  try:
    return len(df)
  except (RuntimeError, TypeError, AttributeError) as exc:
    LOG.debug('_frame_row_count: len() unreadable (%s)',
        type(exc).__name__)
    return None


def _frame_col_count(df):
  '''Return column count for pandas/Spark-like frames, else None.'''
  columns = getattr(df, 'columns', None)
  if columns is not None:
    try:
      return len(columns)
    except (RuntimeError, TypeError, AttributeError) as exc:
      LOG.debug('_frame_col_count: columns unreadable (%s)',
          type(exc).__name__)
      return None
  return None


def check_pandas_limits(df, config=None):
  '''Guard the pandas-conversion path against driver OOM.

  Args:
    df: Pandas or Spark-like frame (``count()``/``columns`` used,
      never ``toPandas()``).
    config: Mapping that may carry ``pandas_conversion_params``
      with ``max_rows_for_pandas`` (default 5,000,000). Flat
      ``max_rows_for_pandas`` is also accepted.

  Returns:
    None when the frame is within the threshold.

  Raises:
    ValueError: If row count exceeds the threshold, directing
      to Spark-native feature engineering / distributed
      training (H2O DAI/Spark, XGBoost DMatrix distributed).
  '''
  limit = _pandas_limit_from_config(config)
  rows = _frame_row_count(df)
  if rows is None:
    return None
  if rows > limit:
    cols = _frame_col_count(df)
    cells = rows * cols if cols is not None else rows
    raise ValueError(
      'pandas conversion exceeds max_rows_for_pandas=%d: got '
      '%d rows x %s cols (%d cells). Use Spark-native feature '
      'engineering / distributed training (H2O DAI/Spark, '
      'XGBoost DMatrix distributed) instead of converting to '
      'pandas on the driver.' % (limit, rows, cols, cells))
  return None


def to_pandas(df_or_spark_df, config=None):
  '''Convert a Spark DataFrame to pandas, passing others through.

  :func:`check_pandas_limits` always runs first (with the
  default 5,000,000-row threshold when ``config`` is ``None``)
  so oversized frames raise instead of OOM-ing.

  Args:
    df_or_spark_df: A Spark DataFrame, a pandas object, or any
      other object to return unchanged.
    config: Optional configuration; used for the pandas limit
      guard (``pandas_conversion_params`` /
      ``max_rows_for_pandas``).

  Returns:
    The pandas representation, or the input unchanged when it
    is not a Spark DataFrame.
  '''
  check_pandas_limits(
      df_or_spark_df, config if config is not None else {})
  converter = getattr(df_or_spark_df, 'toPandas', None)
  if callable(converter):
    return converter()
  return df_or_spark_df


class SparkSessionFactory:
  '''Factory for Spark sessions with local-mode fallback.'''

  @staticmethod
  def get_spark(config=None):
    '''Return a Spark session, or None when unavailable.

    Returns None gracefully when Spark is disabled or when
    ``pyspark`` cannot be imported/started, so local pandas
    execution is never broken by missing Spark.

    Autoscaling-friendly config (see
    :func:`spark_builder_config`): dynamic allocation, AQE,
    shuffle partitions from ``spark_params``. Cloud cluster
    mode note: on Dataproc/EMR the driver is the cluster
    driver; this factory only builds the session -- cluster
    submission/management is out of scope (config only).

    Args:
      config: Mapping that may carry a ``spark`` section with
        ``enabled``, ``master``, ``app_name``, ``config``, and
        ``spark_params`` entries.

    Returns:
      A ``SparkSession`` when Spark is enabled and available,
      else None.
    '''
    if not _spark_enabled(config):
      return None
    section = _spark_section(config)
    try:
      from pyspark.sql import SparkSession
    except ImportError as exc:
      LOG.debug('get_spark: pyspark unavailable (%s)',
          type(exc).__name__)
      return None
    try:
      builder = SparkSession.builder
      builder = builder.master(section.get('master', 'local[*]'))
      builder = builder.appName(section.get('app_name', 'forecast-ml'))
      for key, value in spark_builder_config(config).items():
        builder = builder.config(key, value)
      return builder.getOrCreate()
    except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
      LOG.debug('get_spark: session build failed (%s)',
          type(exc).__name__)
      return None

  @staticmethod
  def to_pandas(df_or_spark_df, config=None):
    '''Convert a Spark DataFrame to pandas.

    Args:
      df_or_spark_df: A Spark DataFrame or a passthrough object.
      config: Optional configuration mapping.

    Returns:
      The pandas representation, or the input unchanged.
    '''
    return to_pandas(df_or_spark_df, config)

  @staticmethod
  def check_pandas_limits(df, config=None):
    '''Guard a frame against pandas-conversion OOM.

    Args:
      df: Pandas or Spark-like frame.
      config: Configuration mapping.

    Returns:
      None when within the configured threshold.
    '''
    return check_pandas_limits(df, config)
