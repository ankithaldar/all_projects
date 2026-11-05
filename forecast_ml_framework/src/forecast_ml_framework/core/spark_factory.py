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

import os

DEFAULT_MAX_ROWS_FOR_PANDAS = 5_000_000
DEFAULT_SHUFFLE_PARTITIONS = 200

_AUTOSCALING_DEFAULTS = {
  'spark.dynamicAllocation.enabled': 'true',
  'spark.sql.adaptive.enabled': 'true',
  'spark.sql.adaptive.coalescePartitions.enabled': 'true',
}


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

  Args:
    config: Raw configuration mapping.

  Returns:
    Dict mapping Spark conf keys to values.
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
  for source in (section.get('config'), params):
    if isinstance(source, dict):
      for key, value in source.items():
        if key in ('shuffle_partitions', 'enabled', 'master',
                   'app_name'):
          continue
        merged[key] = value
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
  except Exception:
    pass
  count = getattr(df, 'count', None)
  if callable(count):
    try:
      return int(count())
    except Exception:
      return None
  try:
    return len(df)
  except Exception:
    return None


def _frame_col_count(df):
  '''Return column count for pandas/Spark-like frames, else None.'''
  columns = getattr(df, 'columns', None)
  if columns is not None:
    try:
      return len(columns)
    except Exception:
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

  When ``config`` is supplied, :func:`check_pandas_limits` runs
  first so oversized frames raise instead of OOM-ing.

  Args:
    df_or_spark_df: A Spark DataFrame, a pandas object, or any
      other object to return unchanged.
    config: Optional configuration; used for the pandas limit
      guard.

  Returns:
    The pandas representation, or the input unchanged when it
    is not a Spark DataFrame.
  '''
  if config is not None:
    check_pandas_limits(df_or_spark_df, config)
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
    except ImportError:
      return None
    try:
      builder = SparkSession.builder
      builder = builder.master(section.get('master', 'local[*]'))
      builder = builder.appName(section.get('app_name', 'forecast-ml'))
      for key, value in spark_builder_config(config).items():
        builder = builder.config(key, value)
      return builder.getOrCreate()
    except Exception:
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
