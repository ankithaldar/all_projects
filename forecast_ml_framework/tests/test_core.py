#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for forecast_ml_framework.core abstractions.'''

import pytest


def _loader(variables=None, parameters=None):
  from forecast_ml_framework.core.config_loader import ConfigLoader
  return ConfigLoader(variables or {}, parameters or {})


def test_config_get_from_parameters():
  loader = _loader(parameters={'model': {'lr': 0.1}})
  assert loader.get('model', 'lr') == 0.1


def test_config_get_falls_back_to_variables():
  loader = _loader(variables={'paths': {'raw': 'data/raw'}})
  assert loader.get('paths', 'raw') == 'data/raw'


def test_config_get_returns_default():
  loader = _loader()
  assert loader.get('missing', 'key', 'fallback') == 'fallback'
  assert loader.get('missing', 'key') is None


def test_task_type_explicit_override():
  loader = _loader(parameters={'task_type': 'classification'})
  assert loader.task_type() == 'classification'


def test_task_type_autodetect_classification_from_target():
  loader = _loader(
      parameters={'target_col': 'label', 'target_type': 'classification'})
  assert loader.task_type() == 'classification'


def test_task_type_autodetect_regression_from_target():
  loader = _loader(
      parameters={'target_col': 'price', 'target_type': 'regression'})
  assert loader.task_type() == 'regression'


def test_task_type_defaults_to_regression():
  loader = _loader(parameters={'target_col': 'price'})
  assert loader.task_type() == 'regression'


def test_merged_features_union_minus_exclude():
  loader = _loader(
      parameters={'business_features': ['b1', 'a'],
                  'exclude_features': ['x']})
  result = loader.merged_feature_list(['a', 'x', 'c'])
  # exclude must remove 'x'; dedup preserves first-seen order.
  assert result == ['a', 'c', 'b1']


def test_merged_features_override_wins():
  loader = _loader(
      parameters={'feature_override': ['only1', 'only2'],
                  'business_features': ['b1'],
                  'exclude_features': ['only2']})
  assert loader.merged_feature_list(['a']) == ['only1', 'only2']


def test_merged_features_dedup_preserves_order():
  loader = _loader(parameters={'business_features': ['a', 'a', 'b']})
  assert loader.merged_feature_list(['a', 'b', 'a']) == ['a', 'b']


def test_local_artifact_store_roundtrip(tmp_path):
  from forecast_ml_framework.core.artifact_store import LocalArtifactStore
  store = LocalArtifactStore(str(tmp_path))
  assert not store.exists('a.txt')
  store.save('a.txt', 'hello')
  assert store.exists('a.txt')
  assert store.load('a.txt') == 'hello'
  store.save('sub/b.txt', 'world')
  assert sorted(store.list('')) == ['a.txt', 'sub/b.txt']
  assert store.list('sub') == ['sub/b.txt']


def test_local_artifact_store_dict_roundtrip(tmp_path):
  from forecast_ml_framework.core.artifact_store import LocalArtifactStore
  store = LocalArtifactStore(str(tmp_path))
  store.save('m.pkl', {'coef': [1, 2]})
  assert store.load('m.pkl') == {'coef': [1, 2]}


def test_factory_selects_local(tmp_path):
  from forecast_ml_framework.core.artifact_store import (
      LocalArtifactStore, create_artifact_store)
  store = create_artifact_store(
      {'type': 'local', 'base_path': str(tmp_path)})
  assert isinstance(store, LocalArtifactStore)


def test_factory_selects_gcs_without_cloud_lib():
  from forecast_ml_framework.core.artifact_store import (
      GCSArtifactStore, create_artifact_store)
  store = create_artifact_store({'type': 'gcs', 'bucket': 'bkt'})
  assert isinstance(store, GCSArtifactStore)
  assert store.bucket == 'bkt'


def test_factory_selects_s3_without_cloud_lib():
  from forecast_ml_framework.core.artifact_store import (
      S3ArtifactStore, create_artifact_store)
  store = create_artifact_store({'type': 's3', 'bucket': 'bkt2'})
  assert isinstance(store, S3ArtifactStore)
  assert store.bucket == 'bkt2'


def test_factory_unknown_type_raises():
  from forecast_ml_framework.core.artifact_store import (
      create_artifact_store)
  with pytest.raises(ValueError):
      create_artifact_store({'type': 'azure'})


def test_spark_factory_disabled_returns_none():
  from forecast_ml_framework.core.spark_factory import SparkSessionFactory
  assert SparkSessionFactory.get_spark({'spark': {'enabled': False}}) is None
  assert SparkSessionFactory.get_spark({'enabled': False}) is None
  assert SparkSessionFactory.get_spark({}) is None


def _hide_pyspark(monkeypatch):
  '''Make pyspark import fail regardless of env (absent or present).'''
  import sys
  import types
  fake = types.ModuleType('pyspark')
  monkeypatch.setitem(sys.modules, 'pyspark', fake)
  monkeypatch.delitem(sys.modules, 'pyspark.sql', raising=False)


def test_spark_factory_unavailable_returns_none_gracefully(monkeypatch):
  from forecast_ml_framework.core.spark_factory import SparkSessionFactory
  _hide_pyspark(monkeypatch)
  result = SparkSessionFactory.get_spark({'spark': {'enabled': True}})
  assert result is None


def test_to_pandas_passthrough_for_pandas_like():
  from forecast_ml_framework.core.spark_factory import to_pandas
  sentinel = object()
  assert to_pandas(sentinel, {}) is sentinel


def test_sql_query_dataset_loads_from_sqlite(tmp_path):
  from forecast_ml_framework.core.datasets import SQLQueryDataSet
  db = tmp_path / 'data.db'
  import sqlite3
  conn = sqlite3.connect(str(db))
  conn.execute('CREATE TABLE t (id INTEGER, val TEXT)')
  conn.execute('INSERT INTO t VALUES (1, \'a\')')
  conn.commit()
  conn.close()
  ds = SQLQueryDataSet(
      connection='sqlite:///' + str(db), query='SELECT id, val FROM t')
  df = ds.load()
  assert list(df['val']) == ['a']
  assert ds.exists() is True


def test_spark_sql_dataset_stub_raises_without_spark(monkeypatch):
  from forecast_ml_framework.core.datasets import SparkSQLDataSet
  _hide_pyspark(monkeypatch)
  ds = SparkSQLDataSet(query='SELECT 1')
  # Kedro wraps load errors in DatasetError; accept any failure.
  with pytest.raises(Exception):
      ds.load()


def test_local_store_rejects_traversal(tmp_path):
  from forecast_ml_framework.core.artifact_store import LocalArtifactStore
  store = LocalArtifactStore(str(tmp_path))
  for bad in ('../escape.txt', 'a/../../escape.txt', '/etc/hostname',
              str(tmp_path / 'abs.txt')):
    with pytest.raises(ValueError):
      store.save(bad, 'x')
    with pytest.raises(ValueError):
      store.load(bad)
    with pytest.raises(ValueError):
      store.exists(bad)
    with pytest.raises(ValueError):
      store.list(bad)


def test_local_store_pickle_evil_rejected(tmp_path):
  import os
  import pickle
  from forecast_ml_framework.core.artifact_store import LocalArtifactStore

  class Evil:
    def __reduce__(self):
      return (os.system, ('echo pwned',))

  store = LocalArtifactStore(str(tmp_path))
  (tmp_path / 'evil.pkl').write_bytes(pickle.dumps(Evil()))
  with pytest.raises(ValueError):
    store.load('evil.pkl')


def test_cloud_store_rejects_bad_keys():
  from forecast_ml_framework.core.artifact_store import (
      GCSArtifactStore, S3ArtifactStore)
  for store in (GCSArtifactStore('b'), S3ArtifactStore('b')):
    for bad in ('../escape', '/abs', '..\\escape'):
      with pytest.raises(ValueError):
        store.save(bad, 'x')
      with pytest.raises(ValueError):
        store.load(bad)
      with pytest.raises(ValueError):
        store.exists(bad)


def test_merged_override_str_is_single_feature():
  loader = _loader(parameters={'feature_override': 'only_feature'})
  assert loader.merged_feature_list(['a']) == ['only_feature']


def test_merged_override_non_list_rejected():
  loader = _loader(parameters={'feature_override': {'a': 1}})
  with pytest.raises(ValueError):
    loader.merged_feature_list(['a'])


def test_merged_override_cap():
  loader = _loader(
      parameters={'feature_override': ['f%d' % i for i in range(10001)]})
  with pytest.raises(ValueError):
    loader.merged_feature_list(['a'])


def test_merged_override_nonexistent_passthrough_documented():
  # Existence validation is caller job; override passes through.
  loader = _loader(parameters={'feature_override': ['ghost_col']})
  assert loader.merged_feature_list(['a']) == ['ghost_col']


def test_sql_dataset_rejects_write_and_multi_statement(tmp_path):
  from forecast_ml_framework.core.datasets import SQLQueryDataSet
  db = tmp_path / 'd.db'
  import sqlite3
  sqlite3.connect(str(db)).close()
  conn = 'sqlite:///' + str(db)
  for bad in ('DELETE FROM t', 'INSERT INTO t VALUES (1)',
              'SELECT 1; SELECT 2', 'DROP TABLE t'):
    # Kedro wraps load errors; the cause must be our ValueError.
    with pytest.raises(Exception) as excinfo:
      SQLQueryDataSet(connection=conn, query=bad).load()
    cause = excinfo.value.__cause__ or excinfo.value
    assert isinstance(cause, ValueError)


def test_restricted_loads_chain_read_pickle_blocked(tmp_path):
  import os
  import pickle
  import pandas as pd
  from forecast_ml_framework.core.artifact_store import restricted_loads
  marker = tmp_path / 'pwned'

  class Evil:
    def __reduce__(self):
      return (os.system, (f'touch {marker}',))

  inner = pickle.dumps(Evil())

  class Outer:
    def __reduce__(self):
      return (pd.read_pickle, (inner,))

  with pytest.raises(ValueError):
    restricted_loads(pickle.dumps(Outer()))
  assert not marker.exists()


def test_restricted_loads_read_csv_blocked(tmp_path):
  import pickle
  import pandas as pd
  from forecast_ml_framework.core.artifact_store import restricted_loads
  secret = tmp_path / 'secret.csv'
  secret.write_text('secret_col\n42\n', encoding='utf-8')

  class Outer:
    def __reduce__(self):
      return (pd.read_csv, (str(secret),))

  with pytest.raises(ValueError):
    restricted_loads(pickle.dumps(Outer()))


def test_restricted_loads_legit_roundtrips():
  import pickle
  import numpy as np
  import pandas as pd
  from forecast_ml_framework.core.artifact_store import restricted_loads
  from sklearn.linear_model import LinearRegression
  arr = np.array([[1.0, 2.0], [3.0, 4.0]])
  np.testing.assert_array_equal(restricted_loads(pickle.dumps(arr)), arr)
  df = pd.DataFrame({'a': [1, 2], 'b': [3.5, 4.5]})
  pd.testing.assert_frame_equal(
      restricted_loads(pickle.dumps(df)), df)
  model = LinearRegression().fit([[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0])
  loaded = restricted_loads(pickle.dumps(model))
  assert loaded.predict([[1.0, 2.0]])[0] == pytest.approx(1.0)


def test_sql_validator_blocks_cte_delete_and_into_outfile(tmp_path):
  from forecast_ml_framework.core.datasets import (
      SQLQueryDataSet, _validate_read_only_query)
  db = tmp_path / 'd2.db'
  import sqlite3
  sqlite3.connect(str(db)).close()
  conn = 'sqlite:///' + str(db)
  bad = (
      'WITH x AS (DELETE FROM t RETURNING id) SELECT id FROM x',
      'SELECT id INTO OUTFILE "/tmp/x" FROM t',
      'SELECT * INTO new_table FROM t',
      'SELECT pg_sleep(5)',
      'WITH x AS (SELECT 1) DELETE FROM t',
      'SELECT 1; DELETE FROM t',
  )
  for query in bad:
    with pytest.raises(ValueError):
      _validate_read_only_query(query)
    with pytest.raises(Exception) as excinfo:
      SQLQueryDataSet(connection=conn, query=query).load()
    cause = excinfo.value.__cause__ or excinfo.value
    assert isinstance(cause, ValueError)
  # UNION is a legitimate read; SELECT exfiltration is inherent to
  # SELECT, so UNION stays allowed by design.
  assert _validate_read_only_query(
      'SELECT id FROM t UNION SELECT 1') is not None


def _has_value_error(exc):
  '''True when ``exc`` or any chained cause/context is ValueError.'''
  seen = set()
  while exc is not None and id(exc) not in seen:
    seen.add(id(exc))
    if isinstance(exc, ValueError):
      return True
    exc = exc.__cause__ or exc.__context__
  return False


def test_spark_sql_dataset_validates_before_session(monkeypatch):
  from forecast_ml_framework.core.datasets import SparkSQLDataSet
  _hide_pyspark(monkeypatch)
  calls = []

  class FakeSpark:
    '''Session stub: any sql() call proves validation was skipped.'''

    def sql(self, query):
      calls.append(query)
      raise AssertionError('spark.sql must not be reached')

  for bad in ('WITH x AS (DELETE FROM t RETURNING id) SELECT id FROM x',
              'SELECT id INTO OUTFILE "/tmp/x" FROM t',
              'INSERT INTO t VALUES (1)'):
    # No session at all: validation must win over RuntimeError.
    with pytest.raises(Exception) as excinfo:
      SparkSQLDataSet(query=bad).load()
    assert _has_value_error(excinfo.value)
    # Session present: validation must still win, sql() not called.
    with pytest.raises(Exception) as excinfo:
      SparkSQLDataSet(spark=FakeSpark(), query=bad).load()
    assert _has_value_error(excinfo.value)
  assert calls == []
