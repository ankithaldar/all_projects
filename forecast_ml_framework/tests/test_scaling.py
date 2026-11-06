#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for Phase 2 cloud scaling: partitioned Spark scoring,
unbounded caps, pandas conversion guard, autoscaling Spark config.'''

import sys
import types

import numpy as np
import pandas as pd
import pytest

from forecast_ml_framework.pipelines.scoring.nodes import (
  load_scoring_data,
  predict,
)


class _FakeModel:
  '''Classifier using first feature column.'''

  def predict(self, x):
    '''Return one label per row.'''
    return (np.asarray(x, dtype=float)[:, 0] > 0).astype(int)

  def predict_proba(self, x):
    '''Return two-column proba.'''
    pos = np.where(np.asarray(x, dtype=float)[:, 0] > 0, 0.9, 0.1)
    return np.column_stack([1.0 - pos, pos])


class _RecordingModel:
  '''Model recording chunk sizes.'''

  def __init__(self):
    self.sizes = []

  def predict(self, x):
    '''Record size and return labels.'''
    self.sizes.append(len(x))
    return (np.asarray(x, dtype=float)[:, 0] > 0).astype(int)


class _FakeSparkDF:
  '''Fake Spark DF: mapPartitions over pandas slices, no toPandas.

  ``mapPartitions(func)`` calls ``func(pandas_slice)`` per
  partition (in-memory slice of ``partition_size`` rows) and
  concatenates the pandas results. Real Spark returns a
  Spark/RDD object; this fake returns pandas so tests can assert
  on the scored output directly.
  '''

  def __init__(self, frame, partition_size=2):
    self._frame = frame.copy()
    self.columns = list(frame.columns)
    self._partition_size = partition_size
    self.to_pandas_calls = 0
    self.map_partitions_calls = 0

  def toPandas(self):
    '''Record call and return frame (should not be called).'''
    self.to_pandas_calls += 1
    return self._frame.copy()

  def count(self):
    '''Return row count without converting.'''
    return len(self._frame)

  def mapPartitions(self, func):
    '''Apply func to each pandas partition slice.'''
    self.map_partitions_calls += 1
    parts = []
    for start in range(0, len(self._frame), self._partition_size):
      part = self._frame.iloc[start:start + self._partition_size]
      parts.append(func(part))
    if not parts:
      return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _frame(n=5):
  '''Small scoring frame.'''
  return pd.DataFrame({
    'id': list(range(n)),
    'f1': [1.0, -1.0, 2.0, -2.0, 3.0][:n] if n <= 5 else [1.0] * n,
    'f2': [0.0] * n,
  })


def _params(**overrides):
  '''Scoring params for scaling tests.'''
  base = {
    'id_cols': ['id'],
    'feature_cols': ['f1', 'f2'],
    'task_type': 'classification',
    'score_model_id': 'model-v1',
    'spark_scoring': {'enabled': True},
  }
  base.update(overrides)
  return base


def test_spark_scoring_does_not_call_to_pandas():
  '''Spark path scores via mapPartitions, never toPandas.'''
  spark_df = _FakeSparkDF(_frame(5), partition_size=2)
  out = predict(spark_df, _params(), model_loader=lambda: _FakeModel())
  assert spark_df.to_pandas_calls == 0
  assert spark_df.map_partitions_calls == 1
  assert out['prediction'].tolist() == [1, 0, 1, 0, 1]
  assert list(out.columns) == [
    'id', 'prediction', 'proba', 'score_decile', 'score_centile',
    'high_score_ind', 'score_model_id', 'scored_at']
  for col in ('score_decile', 'score_centile', 'high_score_ind'):
    assert col in out.columns


def test_spark_scoring_partition_batch_chunks_within_partition():
  '''partition_batch_size sub-chunks each Spark partition.'''
  spark_df = _FakeSparkDF(_frame(5), partition_size=5)
  model = _RecordingModel()
  out = predict(
    spark_df, _params(partition_batch_size=2),
    model_loader=lambda: model)
  assert model.sizes == [2, 2, 1]
  assert len(out) == 5


def test_load_scoring_data_spark_passthrough_when_enabled():
  '''load node passes Spark DF through when spark_scoring on.'''
  spark_df = _FakeSparkDF(_frame(3))
  out = load_scoring_data(spark_df, _params())
  assert out is spark_df
  assert spark_df.to_pandas_calls == 0


def test_max_rows_null_raises_without_spark():
  '''max_rows=None with spark disabled raises (unbounded pandas).'''
  frame = pd.DataFrame({
    'id': np.arange(250_000), 'f1': np.ones(250_000),
    'f2': np.zeros(250_000)})
  with pytest.raises(ValueError, match='max_rows'):
    predict(
      frame, _params(max_rows=None, spark_scoring={'enabled': False}),
      model_loader=lambda: _FakeModel())


def test_max_rows_null_allowed_with_spark_enabled():
  '''max_rows=None is allowed only when spark_scoring is enabled.'''
  frame = pd.DataFrame({
    'id': np.arange(250_000), 'f1': np.ones(250_000),
    'f2': np.zeros(250_000)})
  out = predict(
    frame, _params(max_rows=None, spark_scoring={'enabled': True}),
    model_loader=lambda: _FakeModel())
  assert len(out) == 250_000


def test_max_rows_null_unbounded_spark():
  '''max_rows=None means unbounded for Spark path.'''
  frame = pd.DataFrame({
    'id': np.arange(250_000), 'f1': np.ones(250_000),
    'f2': np.zeros(250_000)})
  spark_df = _FakeSparkDF(frame, partition_size=100_000)
  out = predict(
    spark_df, _params(max_rows=None),
    model_loader=lambda: _FakeModel())
  assert spark_df.to_pandas_calls == 0
  assert len(out) == 250_000


def test_max_cols_null_unbounded_wide_frame():
  '''max_cols=None allows >1000 columns (500-col wide frame).'''
  extra = pd.DataFrame({('x%d' % i): np.zeros(4) for i in range(498)})
  frame = pd.concat([_frame(4), extra], axis=1)
  assert len(frame.columns) == 501
  params = _params(max_cols=None, max_rows=1000,
                   spark_scoring={'enabled': False})
  del params['feature_cols']
  out = predict(frame, params, model_loader=lambda: _FakeModel())
  assert out['prediction'].tolist() == [1, 0, 1, 0]


def test_wide_500_col_chunked_predict():
  '''Chunked predict works on a 500-column wide frame.'''
  n = 6
  cols = {'id': list(range(n)),
          'f0': [1.0, -1.0, 2.0, -2.0, 3.0, -3.0]}
  for i in range(1, 500):
    cols['f%d' % i] = np.zeros(n)
  frame = pd.DataFrame(cols)
  params = _params(max_cols=None, max_rows=1000, batch_size=2,
                   spark_scoring={'enabled': False})
  del params['feature_cols']
  model = _RecordingModel()
  out = predict(frame, params, model_loader=lambda: model)
  assert model.sizes == [2, 2, 2]
  assert out['prediction'].tolist() == [1, 0, 1, 0, 1, 0]


def test_check_pandas_limits_raises_informative():
  '''Guard raises directing to Spark-native / distributed paths.'''
  from forecast_ml_framework.core.spark_factory import (
    check_pandas_limits,
  )
  frame = pd.DataFrame({'id': range(100), 'f1': np.ones(100)})
  with pytest.raises(ValueError) as excinfo:
    check_pandas_limits(
      frame, {'pandas_conversion_params': {'max_rows_for_pandas': 10}})
  msg = str(excinfo.value).lower()
  assert 'spark' in msg
  assert 'distributed' in msg


def test_check_pandas_limits_default_threshold():
  '''Default 5M row threshold rejects >5M rows.'''
  from forecast_ml_framework.core.spark_factory import (
    check_pandas_limits,
  )
  frame = pd.DataFrame({'id': np.arange(5_000_001),
                        'f1': np.ones(5_000_001)})
  with pytest.raises(ValueError, match='max_rows_for_pandas'):
    check_pandas_limits(frame, {})


def test_check_pandas_limits_passes_small():
  '''Small frames pass the guard and return None.'''
  from forecast_ml_framework.core.spark_factory import (
    check_pandas_limits,
  )
  assert check_pandas_limits(_frame(3), {}) is None


def test_check_pandas_limits_spark_count():
  '''Guard uses Spark count() without toPandas.'''
  from forecast_ml_framework.core.spark_factory import (
    check_pandas_limits,
  )
  spark_df = _FakeSparkDF(_frame(5))
  with pytest.raises(ValueError, match='max_rows_for_pandas'):
    check_pandas_limits(
      spark_df, {'max_rows_for_pandas': 2})
  assert spark_df.to_pandas_calls == 0


def test_get_spark_autoscaling_config(monkeypatch):
  '''get_spark applies dynamic allocation, AQE, shuffle partitions.'''
  configs = {}

  class _Builder:
    '''Fake Spark builder recording config calls.'''

    def master(self, value):
      '''Record master, return self.'''
      configs['master'] = value
      return self

    def appName(self, value):
      '''Record app, return self.'''
      configs['app_name'] = value
      return self

    def config(self, key, value):
      '''Record config entry, return self.'''
      configs[key] = value
      return self

    def getOrCreate(self):
      '''Return sentinel session.'''
      return 'fake-session'

  fake_sql = types.ModuleType('pyspark.sql')
  fake_sql.SparkSession = types.SimpleNamespace(builder=_Builder())
  fake_pyspark = types.ModuleType('pyspark')
  fake_pyspark.sql = fake_sql
  monkeypatch.setitem(sys.modules, 'pyspark', fake_pyspark)
  monkeypatch.setitem(sys.modules, 'pyspark.sql', fake_sql)

  from forecast_ml_framework.core.spark_factory import SparkSessionFactory
  session = SparkSessionFactory.get_spark({
    'spark': {'enabled': True, 'master': 'yarn',
              'spark_params': {'shuffle_partitions': 400}}})
  assert session == 'fake-session'
  assert configs['spark.dynamicAllocation.enabled'] == 'true'
  assert configs['spark.sql.adaptive.enabled'] == 'true'
  assert configs['spark.sql.shuffle.partitions'] == 400


def test_spark_factory_to_pandas_guards():
  '''spark_factory.to_pandas raises before converting huge frames.'''
  from forecast_ml_framework.core.spark_factory import (
    SparkSessionFactory,
  )
  frame = pd.DataFrame({'id': range(100)})
  with pytest.raises(ValueError, match='max_rows_for_pandas'):
    SparkSessionFactory.to_pandas(
      frame, {'pandas_conversion_params': {'max_rows_for_pandas': 10}})


class _FakeSparkCountOnly:
  '''Fake Spark frame reporting >5M rows via count() only.'''

  def __init__(self, rows):
    self._rows = rows
    self.columns = ['id', 'f1']
    self.to_pandas_calls = 0

  def count(self):
    '''Return row count without materialising data.'''
    return self._rows

  def toPandas(self):
    '''Record call; the guard must prevent this.'''
    self.to_pandas_calls += 1
    raise AssertionError('toPandas must not be called')


def test_fake_spark_over_5m_rows_raises_without_to_pandas():
  '''Fake Spark frame >5M rows raises through guarded to_pandas.'''
  from forecast_ml_framework.core.spark_factory import to_pandas
  spark_df = _FakeSparkCountOnly(5_000_001)
  with pytest.raises(ValueError, match='max_rows_for_pandas'):
    to_pandas(spark_df, {})
  assert spark_df.to_pandas_calls == 0


def test_scoring_load_fake_spark_over_pandas_limit_raises():
  '''Scoring load path enforces the pandas guard via factory.'''
  spark_df = _FakeSparkCountOnly(5_000_001)
  with pytest.raises(ValueError, match='max_rows_for_pandas'):
    load_scoring_data(spark_df, _params(spark_scoring={'enabled': False}))
  assert spark_df.to_pandas_calls == 0


def test_load_scoring_data_max_rows_null_without_spark_raises():
  '''load_scoring_data rejects max_rows=None when spark is off.'''
  with pytest.raises(ValueError, match='max_rows'):
    load_scoring_data(
      _frame(3), _params(max_rows=None, spark_scoring={'enabled': False}))
