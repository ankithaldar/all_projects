#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Spark session creation with graceful local fallback.'''
# pylint: disable=import-outside-toplevel, broad-exception-caught


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


def to_pandas(df_or_spark_df, config=None):
  '''Convert a Spark DataFrame to pandas, passing others through.

  Args:
    df_or_spark_df: A Spark DataFrame, a pandas object, or any
      other object to return unchanged.
    config: Optional configuration; unused beyond the signature
      contract, kept for call-site symmetry.

  Returns:
    The pandas representation, or the input unchanged when it
    is not a Spark DataFrame.
  '''
  del config  # kept for call-site symmetry
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

    Args:
      config: Mapping that may carry a ``spark`` section with
        ``enabled``, ``master``, and ``app_name`` entries.

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
      for key, value in (section.get('config') or {}).items():
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
