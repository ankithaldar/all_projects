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


def test_restricted_loads_xgboost_roundtrip():
  import pickle
  import numpy as np
  import pytest as _pytest
  _pytest.importorskip('xgboost')
  from forecast_ml_framework.core.artifact_store import restricted_loads
  from xgboost import XGBRegressor
  x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
  y = np.array([1.0, 2.0, 3.0, 4.0])
  model = XGBRegressor(n_estimators=2, max_depth=2).fit(x, y)
  loaded = restricted_loads(pickle.dumps(model))
  assert loaded.predict(x) == pytest.approx(model.predict(x))


def test_restricted_loads_lightgbm_roundtrip():
  import pickle
  import numpy as np
  import pytest as _pytest
  _pytest.importorskip('lightgbm')
  from forecast_ml_framework.core.artifact_store import restricted_loads
  from lightgbm import LGBMRegressor
  x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
  y = np.array([1.0, 2.0, 3.0, 4.0])
  model = LGBMRegressor(n_estimators=2, verbose=-1).fit(x, y)
  loaded = restricted_loads(pickle.dumps(model))
  assert loaded.predict(x) == pytest.approx(model.predict(x))


def test_restricted_loads_catboost_roundtrip():
  import pickle
  import numpy as np
  import pytest as _pytest
  _pytest.importorskip('catboost')
  from forecast_ml_framework.core.artifact_store import restricted_loads
  from catboost import CatBoostRegressor
  x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
  y = np.array([1.0, 2.0, 3.0, 4.0])
  model = CatBoostRegressor(iterations=2, verbose=False).fit(x, y)
  loaded = restricted_loads(pickle.dumps(model))
  assert loaded.predict(x) == pytest.approx(model.predict(x))


def test_restricted_loads_zoo_classifiers_roundtrip():
  import pickle
  import numpy as np
  import pytest as _pytest
  from forecast_ml_framework.core.artifact_store import restricted_loads
  x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
  y = np.array([0, 1, 0, 1])
  _pytest.importorskip('xgboost')
  from xgboost import XGBClassifier
  model = XGBClassifier(n_estimators=2, max_depth=2).fit(x, y)
  loaded = restricted_loads(pickle.dumps(model))
  assert list(loaded.predict(x)) == list(model.predict(x))
  _pytest.importorskip('lightgbm')
  from lightgbm import LGBMClassifier
  model = LGBMClassifier(n_estimators=2, verbose=-1).fit(x, y)
  loaded = restricted_loads(pickle.dumps(model))
  assert list(loaded.predict(x)) == list(model.predict(x))
  _pytest.importorskip('catboost')
  from catboost import CatBoostClassifier
  model = CatBoostClassifier(iterations=2, verbose=False).fit(x, y)
  loaded = restricted_loads(pickle.dumps(model))
  assert list(loaded.predict(x)) == list(model.predict(x))


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


# --- Coverage: artifact_store restricted_loads / cloud / decode ----


def test_restricted_loads_rejects_non_bytes():
  from forecast_ml_framework.core.artifact_store import restricted_loads
  for bad in ('text', None, 123, ['x']):
    with pytest.raises(ValueError, match='requires bytes'):
      restricted_loads(bad)


def test_restricted_loads_rejects_malformed_and_truncated():
  import pickle
  from forecast_ml_framework.core.artifact_store import restricted_loads
  for bad in (b'', b'not a pickle', b'\x80\x04\x95', b'\xff\xfe\x00'):
    with pytest.raises(ValueError):
      restricted_loads(bad)
  full = pickle.dumps({'a': 1})
  with pytest.raises(ValueError):
    restricted_loads(full[:3])
  with pytest.raises(ValueError):
    restricted_loads(full[:-2])


def test_restricted_loads_rejects_chained_os_system():
  import os
  import pickle

  class Evil:
    def __reduce__(self):
      return (os.system, ('echo pwned',))

  from forecast_ml_framework.core.artifact_store import restricted_loads
  with pytest.raises(ValueError, match='Forbidden'):
    restricted_loads(pickle.dumps(Evil()))


def test_decode_payload_pickle_text_and_raw():
  import pickle
  from forecast_ml_framework.core.artifact_store import _decode_payload
  assert _decode_payload(pickle.dumps([1, 2])) == [1, 2]
  assert _decode_payload('hello'.encode('utf-8')) == 'hello'
  assert _decode_payload(b'\xff\xfe\x00binary') == b'\xff\xfe\x00binary'
  assert _decode_payload(b'') == ''


def test_validate_cloud_key_rejects_absolute_and_escape():
  from forecast_ml_framework.core.artifact_store import _validate_cloud_key
  for bad in ('/abs/key', '\\abs\\key', '../escape', 'a/../b',
              'a\\..\\b', '..', 'x/../../y'):
    with pytest.raises(ValueError):
      _validate_cloud_key(bad)
  assert _validate_cloud_key('a/b.txt') == 'a/b.txt'
  assert _validate_cloud_key('') == ''
  assert _validate_cloud_key('a..b/c') == 'a..b/c'


def test_local_store_bytes_and_empty_path(tmp_path):
  from forecast_ml_framework.core.artifact_store import LocalArtifactStore
  store = LocalArtifactStore(str(tmp_path))
  assert store._full_path('') == str(tmp_path.resolve())
  store.save('raw.bin', b'\xff\x00\xfe')
  assert store.load('raw.bin') == b'\xff\x00\xfe'
  assert store.exists('raw.bin') is True
  assert store.list() == ['raw.bin']


def _fake_google(monkeypatch):
  import sys
  import types

  class FakeBlob:
    def __init__(self, bucket, name):
      self._bucket = bucket
      self.name = name

    def upload_from_string(self, data):
      self._bucket.objects[self.name] = data

    def download_as_bytes(self):
      return self._bucket.objects[self.name]

    def exists(self):
      return self.name in self._bucket.objects

  class FakeBucket:
    def __init__(self, name):
      self.name = name
      self.objects = {}

    def blob(self, key):
      return FakeBlob(self, key)

    def list_blobs(self, prefix=''):
      return [FakeBlob(self, k) for k in sorted(self.objects)
              if k.startswith(prefix)]

  shared_bucket = FakeBucket('bkt')

  class FakeClient:
    def __init__(self):
      self._bucket = shared_bucket

    def bucket(self, name):
      assert name == 'bkt'
      return self._bucket

  storage_mod = types.ModuleType('google.cloud.storage')
  storage_mod.Client = FakeClient
  cloud_mod = types.ModuleType('google.cloud')
  cloud_mod.storage = storage_mod
  google_mod = types.ModuleType('google')
  google_mod.cloud = cloud_mod
  monkeypatch.setitem(sys.modules, 'google', google_mod)
  monkeypatch.setitem(sys.modules, 'google.cloud', cloud_mod)
  monkeypatch.setitem(sys.modules, 'google.cloud.storage', storage_mod)
  return FakeClient


def test_gcs_store_branches_via_fake_sdk(monkeypatch):
  import pickle
  _fake_google(monkeypatch)
  from forecast_ml_framework.core.artifact_store import GCSArtifactStore
  store = GCSArtifactStore('bkt')
  store.save('a.txt', 'hello')
  store.save('b.bin', b'\xff\x00')
  store.save('c.pkl', {'x': 1})
  assert store.load('a.txt') == 'hello'
  assert store.load('b.bin') == b'\xff\x00'
  assert store.load('c.pkl') == {'x': 1}
  objects = store._client().bucket('bkt').objects
  assert pickle.loads(objects['c.pkl']) == {'x': 1}
  assert store.exists('a.txt') is True
  assert store.exists('missing') is False
  assert store.list('') == ['a.txt', 'b.bin', 'c.pkl']
  assert store.list('a') == ['a.txt']
  with pytest.raises(ValueError):
    store.save('/etc/passwd', 'x')
  with pytest.raises(ValueError):
    store.list('../escape')
  blob = store._client().bucket('bkt').objects
  assert blob['a.txt'] == b'hello'


def test_gcs_client_missing_sdk_raises_import_error(monkeypatch):
  import sys
  monkeypatch.setitem(sys.modules, 'google', None)
  from forecast_ml_framework.core.artifact_store import GCSArtifactStore
  with pytest.raises(ImportError, match='google-cloud-storage'):
    GCSArtifactStore('bkt')._client()


def _fake_boto3(monkeypatch):
  import io
  import sys
  import types

  class FakePaginator:
    def __init__(self, client):
      self._client = client

    def paginate(self, Bucket, Prefix=''):
      keys = sorted(k for k in self._client.objects
                    if k.startswith(Prefix))
      yield {'Contents': [{'Key': k} for k in keys]} if keys else {}

  class FakeS3:
    def __init__(self):
      self.objects = {}

    def put_object(self, Bucket, Key, Body):
      assert Bucket == 'bkt'
      self.objects[Key] = Body

    def get_object(self, Bucket, Key):
      return {'Body': io.BytesIO(self.objects[Key])}

    def head_object(self, Bucket, Key):
      if Key not in self.objects:
        raise KeyError(Key)
      return {}

    def get_paginator(self, name):
      assert name == 'list_objects_v2'
      return FakePaginator(self)

  client = FakeS3()
  boto_mod = types.ModuleType('boto3')
  boto_mod.client = lambda svc: client
  monkeypatch.setitem(sys.modules, 'boto3', boto_mod)
  return client


def test_s3_store_branches_via_fake_sdk(monkeypatch):
  import pickle
  client = _fake_boto3(monkeypatch)
  from forecast_ml_framework.core.artifact_store import S3ArtifactStore
  store = S3ArtifactStore('bkt')
  store.save('a.txt', 'hello')
  store.save('b.bin', b'\xff\x00')
  store.save('c.pkl', [1, 2])
  assert store.load('a.txt') == 'hello'
  assert store.load('b.bin') == b'\xff\x00'
  assert client.objects['a.txt'] == b'hello'
  assert pickle.loads(client.objects['c.pkl']) == [1, 2]
  assert store.exists('a.txt') is True
  assert store.exists('missing') is False
  assert store.list('') == ['a.txt', 'b.bin', 'c.pkl']
  assert store.list('a') == ['a.txt']
  with pytest.raises(ValueError):
    store.save('../escape', 'x')
  with pytest.raises(ValueError):
    store.list('/abs')


def test_s3_client_missing_sdk_raises_import_error(monkeypatch):
  import sys
  monkeypatch.setitem(sys.modules, 'boto3', None)
  from forecast_ml_framework.core.artifact_store import S3ArtifactStore
  with pytest.raises(ImportError, match='boto3 is required'):
    S3ArtifactStore('bkt')._client()


def test_create_artifact_store_nested_and_aliases(tmp_path):
  from forecast_ml_framework.core.artifact_store import (
    GCSArtifactStore, LocalArtifactStore, S3ArtifactStore,
    create_artifact_store)
  nested = create_artifact_store(
      {'artifact_store': {'type': 'local', 'base_path': str(tmp_path)}})
  assert isinstance(nested, LocalArtifactStore)
  nested2 = create_artifact_store(
      {'store': {'store_type': 'gcs', 'bucket_name': 'b1'}})
  assert isinstance(nested2, GCSArtifactStore)
  assert nested2.bucket == 'b1'
  assert isinstance(
      create_artifact_store({'backend': 'aws', 'bucket': 'b2'}),
      S3ArtifactStore)
  assert isinstance(create_artifact_store(None), LocalArtifactStore)


# --- Coverage: datasets SQL quoting / kwargs alias / pickle dataset ----


def test_strip_sql_literals_blanks_quoted_keywords():
  from forecast_ml_framework.core.datasets import _strip_sql_literals
  out = _strip_sql_literals("SELECT 'DROP TABLE x' AS a, \"DELETE\" FROM t")
  assert 'DROP' not in out and 'DELETE' not in out
  assert 'SELECT' in out and 'FROM' in out
  escaped = _strip_sql_literals("SELECT 'it''s DROP' FROM t")
  assert 'DROP' not in escaped
  unterminated = _strip_sql_literals("SELECT 'abc")
  assert 'abc' not in unterminated
  assert _strip_sql_literals('SELECT 1') == 'SELECT 1'


def test_validate_read_only_query_edges():
  from forecast_ml_framework.core.datasets import _validate_read_only_query
  assert _validate_read_only_query('SELECT 1;') == 'SELECT 1;'
  assert _validate_read_only_query('SELECT 1 ; ') == 'SELECT 1 ; '
  assert _validate_read_only_query(
      "SELECT ';' FROM t WHERE x = 'a;b'") is not None
  for bad in (None, '', '   ', 123, b'SELECT 1'):
    with pytest.raises(ValueError):
      _validate_read_only_query(bad)


def test_sql_dataset_dict_and_kwargs_alias_merging():
  from forecast_ml_framework.core.datasets import SQLQueryDataSet
  ds = SQLQueryDataSet(
      {'connection': 'dict-conn', 'query': 'SELECT 1'},
      sql='SELECT 2', url='kwarg-url')
  assert ds.connection == 'dict-conn'
  assert ds.query == 'SELECT 1'  # merged dict query wins over sql alias
  ds_alias = SQLQueryDataSet({'connection': 'c'}, sql='SELECT 2')
  assert ds_alias.query == 'SELECT 2'
  ds2 = SQLQueryDataSet(url='u1')
  assert ds2.connection == 'u1'
  ds3 = SQLQueryDataSet(dsn='d1', sql='SELECT 3')
  assert ds3.connection == 'd1' and ds3.query == 'SELECT 3'
  ds4 = SQLQueryDataSet(engine='e1', query='SELECT 4')
  assert ds4.connection == 'e1' and ds4.query == 'SELECT 4'
  ds5 = SQLQueryDataSet({'url': 'u2', 'sql': 'SELECT 5'})
  assert ds5.connection == 'u2' and ds5.query == 'SELECT 5'
  ds6 = SQLQueryDataSet({'dsn': 'd2'})
  assert ds6.connection == 'd2'
  ds7 = SQLQueryDataSet({'engine': 'e2'})
  assert ds7.connection == 'e2'


def test_sql_dataset_engine_branches(tmp_path):
  from forecast_ml_framework.core.datasets import SQLQueryDataSet
  class _Engine:
    def connect(self):
      return None

  sentinel = _Engine()
  ds = SQLQueryDataSet(connection=sentinel, query='SELECT 1')
  assert ds._engine() is sentinel
  with pytest.raises(ValueError, match='requires a connection'):
    SQLQueryDataSet(query='SELECT 1')._engine()
  real = SQLQueryDataSet(connection='sqlite:///:memory:',
                         query='SELECT 1 AS one')
  engine = real._engine()
  assert engine is not None
  frame = real.load()
  assert list(frame['one']) == [1]
  assert real.exists() is True
  bad = SQLQueryDataSet(connection='sqlite:///:memory:',
                        query='SELECT 1')
  with pytest.raises(NotImplementedError):
    bad._save(object())
  assert real._describe() == {
      'type': 'SQLQueryDataSet', 'has_query': True}
  broken = SQLQueryDataSet(connection='sqlite:////nonexistent-dir-xyz/db',
                           query='SELECT 1')
  assert broken.exists() is False


def test_sql_dataset_engine_missing_sqlalchemy(monkeypatch):
  import sys
  monkeypatch.setitem(sys.modules, 'sqlalchemy', None)
  from forecast_ml_framework.core.datasets import SQLQueryDataSet
  with pytest.raises(ImportError, match='sqlalchemy is required'):
    SQLQueryDataSet(connection='sqlite:///:memory:',
                    query='SELECT 1')._engine()


def test_spark_dataset_aliases_and_fake_session():
  from forecast_ml_framework.core.datasets import SparkSQLDataSet
  ds = SparkSQLDataSet({'jdbc_config': {'url': 'jdbc:x'}},
                       sql='SELECT 1')
  assert ds.config == {'jdbc_config': {'url': 'jdbc:x'}}
  assert ds.query == 'SELECT 1'
  assert ds.spark is None
  ds2 = SparkSQLDataSet(session='sess', query='SELECT 1',
                        jdbc_config={'a': 1})
  assert ds2.spark == 'sess' and ds2.config == {'a': 1}

  class FakeSpark:
    def sql(self, query):
      return ('ran', query)

  ds3 = SparkSQLDataSet(spark=FakeSpark(), query='SELECT 1')
  assert ds3.load() == ('ran', 'SELECT 1')
  assert ds3._describe()['type'] == 'SparkSQLDataSet'
  assert ds3.exists() is True
  with pytest.raises(NotImplementedError):
    ds3._save(object())


def test_spark_dataset_no_query_and_no_session(monkeypatch):
  from forecast_ml_framework.core.datasets import SparkSQLDataSet
  with pytest.raises(ValueError, match='requires a query'):
    SparkSQLDataSet(spark=object())._load()
  no_spark = SparkSQLDataSet(query='SELECT 1')
  monkeypatch.setattr(
      'forecast_ml_framework.core.spark_factory.SparkSessionFactory.get_spark',
      staticmethod(lambda config: None), raising=False)
  with pytest.raises(RuntimeError, match='requires a Spark session'):
    no_spark._resolve_spark()
  assert no_spark.exists() is False


def test_restricted_pickle_dataset_roundtrip_and_aliases(tmp_path):
  import pickle
  from forecast_ml_framework.core.datasets import RestrictedPickleDataset
  path = tmp_path / 'sub' / 'model.pkl'
  ds = RestrictedPickleDataset(str(path))
  ds.save({'a': [1, 2]})
  assert ds.load() == {'a': [1, 2]}
  assert ds.exists() is True
  assert ds._describe()['filepath'] == str(path)
  ds2 = RestrictedPickleDataset(path=str(path), protocol=4)
  assert ds2.load() == {'a': [1, 2]}
  ds3 = RestrictedPickleDataset(str(path), load_args={})
  assert ds3.load() == {'a': [1, 2]}
  missing = RestrictedPickleDataset(str(tmp_path / 'nope.pkl'))
  assert missing.exists() is False
  empty = RestrictedPickleDataset()
  assert empty._describe()['filepath'] is None
  with pytest.raises(ValueError, match='requires a filepath'):
    empty._load()
  with pytest.raises(ValueError, match='requires a filepath'):
    empty._save(object())
  assert empty.exists() is False
  file_uri = RestrictedPickleDataset('file://' + str(path))
  assert file_uri.load() == {'a': [1, 2]}
  # Tampered pickle is rejected, not silently loaded.
  path.write_bytes(pickle.dumps(os_system_payload()))
  with pytest.raises(Exception) as excinfo:
    RestrictedPickleDataset(str(path)).load()
  assert _has_value_error(excinfo.value)


def os_system_payload():
  import os

  class Evil:
    def __reduce__(self):
      return (os.system, ('echo pwned',))

  return Evil()
