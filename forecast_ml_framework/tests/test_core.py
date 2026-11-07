#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for forecast_ml_framework.core abstractions.'''

import pytest


def _loader(variables=None, parameters=None):
  '''Loader.'''
  from forecast_ml_framework.core.config_loader import ConfigLoader
  return ConfigLoader(variables or {}, parameters or {})


def test_config_get_from_parameters():
  '''Test config get from parameters.'''
  loader = _loader(parameters={'model': {'lr': 0.1}})
  assert loader.get('model', 'lr') == 0.1


def test_config_get_falls_back_to_variables():
  '''Test config get falls back to variables.'''
  loader = _loader(variables={'paths': {'raw': 'data/raw'}})
  assert loader.get('paths', 'raw') == 'data/raw'


def test_config_get_returns_default():
  '''Test config get returns default.'''
  loader = _loader()
  assert loader.get('missing', 'key', 'fallback') == 'fallback'
  assert loader.get('missing', 'key') is None


def test_task_type_explicit_override():
  '''Test task type explicit override.'''
  loader = _loader(parameters={'task_type': 'classification'})
  assert loader.task_type() == 'classification'


def test_task_type_autodetect_classification_from_target():
  '''Test task type autodetect classification from target.'''
  loader = _loader(
      parameters={'target_col': 'label', 'target_type': 'classification'})
  assert loader.task_type() == 'classification'


def test_task_type_autodetect_regression_from_target():
  '''Test task type autodetect regression from target.'''
  loader = _loader(
      parameters={'target_col': 'price', 'target_type': 'regression'})
  assert loader.task_type() == 'regression'


def test_task_type_defaults_to_classification():
  '''Test task type defaults to classification.'''
  loader = _loader(parameters={'target_col': 'price'})
  assert loader.task_type() == 'classification'


def test_task_type_explicit_regression_wins():
  '''Test task type explicit regression wins.'''
  loader = _loader(parameters={'task_type': 'regression'})
  assert loader.task_type() == 'regression'


def test_merged_features_union_minus_exclude():
  '''Test merged features union minus exclude.'''
  loader = _loader(
      parameters={'business_features': ['b1', 'a'],
                  'exclude_features': ['x']})
  result = loader.merged_feature_list(['a', 'x', 'c'])
  # exclude must remove 'x'; dedup preserves first-seen order.
  assert result == ['a', 'c', 'b1']


def test_merged_features_override_wins():
  '''Test merged features override wins.'''
  loader = _loader(
      parameters={'feature_override': ['only1', 'only2'],
                  'business_features': ['b1'],
                  'exclude_features': ['only2']})
  assert loader.merged_feature_list(['a']) == ['only1', 'only2']


def test_merged_features_dedup_preserves_order():
  '''Test merged features dedup preserves order.'''
  loader = _loader(parameters={'business_features': ['a', 'a', 'b']})
  assert loader.merged_feature_list(['a', 'b', 'a']) == ['a', 'b']


def test_local_artifact_store_roundtrip(tmp_path):
  '''Test local artifact store roundtrip.'''
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
  '''Test local artifact store dict roundtrip.'''
  from forecast_ml_framework.core.artifact_store import LocalArtifactStore
  store = LocalArtifactStore(str(tmp_path))
  store.save('m.pkl', {'coef': [1, 2]})
  assert store.load('m.pkl') == {'coef': [1, 2]}


def test_factory_selects_local(tmp_path):
  '''Test factory selects local.'''
  from forecast_ml_framework.core.artifact_store import (
      LocalArtifactStore, create_artifact_store)
  store = create_artifact_store(
      {'type': 'local', 'base_path': str(tmp_path)})
  assert isinstance(store, LocalArtifactStore)


def test_factory_selects_gcs_without_cloud_lib():
  '''Test factory selects gcs without cloud lib.'''
  from forecast_ml_framework.core.artifact_store import (
      GCSArtifactStore, create_artifact_store)
  store = create_artifact_store({'type': 'gcs', 'bucket': 'bkt'})
  assert isinstance(store, GCSArtifactStore)
  assert store.bucket == 'bkt'


def test_factory_selects_s3_without_cloud_lib():
  '''Test factory selects s3 without cloud lib.'''
  from forecast_ml_framework.core.artifact_store import (
      S3ArtifactStore, create_artifact_store)
  store = create_artifact_store({'type': 's3', 'bucket': 'bkt2'})
  assert isinstance(store, S3ArtifactStore)
  assert store.bucket == 'bkt2'


def test_factory_unknown_type_raises():
  '''Test factory unknown type raises.'''
  from forecast_ml_framework.core.artifact_store import (
      create_artifact_store)
  with pytest.raises(ValueError):
    create_artifact_store({'type': 'azure'})


def test_spark_factory_disabled_returns_none():
  '''Test spark factory disabled returns none.'''
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
  '''Test spark factory unavailable returns none gracefully.'''
  from forecast_ml_framework.core.spark_factory import SparkSessionFactory
  _hide_pyspark(monkeypatch)
  result = SparkSessionFactory.get_spark({'spark': {'enabled': True}})
  assert result is None


def test_to_pandas_passthrough_for_pandas_like():
  '''Test to pandas passthrough for pandas like.'''
  from forecast_ml_framework.core.spark_factory import to_pandas
  sentinel = object()
  assert to_pandas(sentinel, {}) is sentinel


def test_sql_query_dataset_loads_from_sqlite(tmp_path):
  '''Test sql query dataset loads from sqlite.'''
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
  '''Test spark sql dataset stub raises without spark.'''
  from forecast_ml_framework.core.datasets import SparkSQLDataSet
  _hide_pyspark(monkeypatch)
  ds = SparkSQLDataSet(query='SELECT 1')
  # Kedro wraps load errors in DatasetError; accept any failure.
  with pytest.raises(Exception):
    ds.load()


def test_local_store_rejects_traversal(tmp_path):
  '''Test local store rejects traversal.'''
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
  '''Test local store pickle evil rejected.'''
  import os
  import pickle
  from forecast_ml_framework.core.artifact_store import LocalArtifactStore

  class Evil:
    '''Test double Evil.'''
    def __reduce__(self):
      '''Return the reduction tuple.'''
      return (os.system, ('echo pwned',))

  store = LocalArtifactStore(str(tmp_path))
  (tmp_path / 'evil.pkl').write_bytes(pickle.dumps(Evil()))
  with pytest.raises(ValueError):
    store.load('evil.pkl')


def test_cloud_store_rejects_bad_keys():
  '''Test cloud store rejects bad keys.'''
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
  '''Test merged override str is single feature.'''
  loader = _loader(parameters={'feature_override': 'only_feature'})
  assert loader.merged_feature_list(['a']) == ['only_feature']


def test_merged_override_non_list_rejected():
  '''Test merged override non list rejected.'''
  loader = _loader(parameters={'feature_override': {'a': 1}})
  with pytest.raises(ValueError):
    loader.merged_feature_list(['a'])


def test_merged_override_cap():
  '''Test merged override cap.'''
  loader = _loader(
      parameters={'feature_override': ['f%d' % i for i in range(10001)]})
  with pytest.raises(ValueError):
    loader.merged_feature_list(['a'])


def test_merged_override_nonexistent_passthrough_documented():
  # Existence validation is caller job; override passes through.
  '''Test merged override nonexistent passthrough documented.'''
  loader = _loader(parameters={'feature_override': ['ghost_col']})
  assert loader.merged_feature_list(['a']) == ['ghost_col']


def test_sql_dataset_rejects_write_and_multi_statement(tmp_path):
  '''Test sql dataset rejects write and multi statement.'''
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
  '''Test restricted loads chain read pickle blocked.'''
  import os
  import pickle
  import pandas as pd
  from forecast_ml_framework.core.artifact_store import restricted_loads
  marker = tmp_path / 'pwned'

  class Evil:
    '''Test double Evil.'''
    def __reduce__(self):
      '''Return the reduction tuple.'''
      return (os.system, (f'touch {marker}',))

  inner = pickle.dumps(Evil())

  class Outer:
    '''Test double Outer.'''
    def __reduce__(self):
      '''Return the reduction tuple.'''
      return (pd.read_pickle, (inner,))

  with pytest.raises(ValueError):
    restricted_loads(pickle.dumps(Outer()))
  assert not marker.exists()


def test_restricted_loads_read_csv_blocked(tmp_path):
  '''Test restricted loads read csv blocked.'''
  import pickle
  import pandas as pd
  from forecast_ml_framework.core.artifact_store import restricted_loads
  secret = tmp_path / 'secret.csv'
  secret.write_text('secret_col\n42\n', encoding='utf-8')

  class Outer:
    '''Test double Outer.'''
    def __reduce__(self):
      '''Return the reduction tuple.'''
      return (pd.read_csv, (str(secret),))

  with pytest.raises(ValueError):
    restricted_loads(pickle.dumps(Outer()))


def test_restricted_loads_legit_roundtrips():
  '''Test restricted loads legit roundtrips.'''
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
  '''Test restricted loads xgboost roundtrip.'''
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
  '''Test restricted loads lightgbm roundtrip.'''
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
  '''Test restricted loads catboost roundtrip.'''
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
  '''Test restricted loads zoo classifiers roundtrip.'''
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
  '''Test sql validator blocks cte delete and into outfile.'''
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
  '''Test spark sql dataset validates before session.'''
  from forecast_ml_framework.core.datasets import SparkSQLDataSet
  _hide_pyspark(monkeypatch)
  calls = []

  class FakeSpark:
    '''Session stub: any sql() call proves validation was skipped.'''

    def sql(self, query):
      '''Sql.'''
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
  '''Test restricted loads rejects non bytes.'''
  from forecast_ml_framework.core.artifact_store import restricted_loads
  for bad in ('text', None, 123, ['x']):
    with pytest.raises(ValueError, match='requires bytes'):
      restricted_loads(bad)


def test_restricted_loads_rejects_malformed_and_truncated():
  '''Test restricted loads rejects malformed and truncated.'''
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
  '''Test restricted loads rejects chained os system.'''
  import os
  import pickle

  class Evil:
    '''Test double Evil.'''
    def __reduce__(self):
      '''Return the reduction tuple.'''
      return (os.system, ('echo pwned',))

  from forecast_ml_framework.core.artifact_store import restricted_loads
  with pytest.raises(ValueError, match='Forbidden'):
    restricted_loads(pickle.dumps(Evil()))


def test_restricted_loads_rejects_tzfile_global():
  '''Test restricted loads rejects tzfile global.'''
  from forecast_ml_framework.core.artifact_store import (
    _SAFE_GLOBALS, restricted_loads)
  assert ('dateutil.tz.tz', 'tzfile') not in _SAFE_GLOBALS
  assert ('dateutil.tz._tzfile', 'tzfile') not in _SAFE_GLOBALS
  # Handcrafted protocol-0 pickle naming the allowlisted alias:
  # c<module>\n<name>\n then REDUCE on the probed path.
  raw = (b'cdateutil.tz.tz\ntzfile\n'
         b'(S\'/tmp/definitely_missing_tz_xyz\'\ntR.')
  with pytest.raises(ValueError, match='Forbidden pickle global'):
    restricted_loads(raw)


def test_restricted_loads_booster_path_errors_are_generic():
  '''Test restricted loads booster path errors are generic.'''
  import pickle
  import pytest as _pytest
  _pytest.importorskip('xgboost')
  from forecast_ml_framework.core.artifact_store import restricted_loads
  from xgboost import Booster
  probe = '/tmp/definitely_missing_secret_path_12345.model'

  class Evil:
    '''Test double Evil.'''
    def __reduce__(self):
      '''Return the reduction tuple.'''
      return (Booster, (None, None, probe))

  with _pytest.raises(ValueError) as excinfo:
    restricted_loads(pickle.dumps(Evil()))
  assert probe not in str(excinfo.value)
  cause = excinfo.value.__cause__ or excinfo.value.__context__
  assert cause is None or probe not in str(cause)


def test_restricted_loads_lightgbm_booster_path_errors_generic():
  '''Test restricted loads lightgbm booster path errors generic.'''
  import pickle
  import pytest as _pytest
  _pytest.importorskip('lightgbm')
  from forecast_ml_framework.core.artifact_store import restricted_loads
  from lightgbm import Booster
  probe = '/tmp/definitely_missing_secret_path_12345.txt'

  class Evil:
    '''Test double Evil.'''
    def __reduce__(self):
      '''Return the reduction tuple.'''
      return (Booster, (None, None, probe))

  with _pytest.raises(ValueError) as excinfo:
    restricted_loads(pickle.dumps(Evil()))
  assert probe not in str(excinfo.value)
  seen = set()
  cause = excinfo.value.__cause__ or excinfo.value.__context__
  while cause is not None and id(cause) not in seen:
    seen.add(id(cause))
    assert probe not in str(cause)
    cause = cause.__cause__ or cause.__context__


def test_restricted_loads_xgboost_booster_valid_file_rejected(tmp_path):
  '''Test restricted loads xgboost booster valid file rejected.'''
  import pickle
  import numpy as np
  import pytest as _pytest
  _pytest.importorskip('xgboost')
  from forecast_ml_framework.core.artifact_store import restricted_loads
  from xgboost import Booster, XGBRegressor
  x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
  y = np.array([1.0, 2.0, 3.0, 4.0])
  model = XGBRegressor(n_estimators=2, max_depth=2).fit(x, y)
  model_path = tmp_path / 'valid.model'
  model.get_booster().save_model(str(model_path))
  assert model_path.is_file()

  class Evil:
    '''Test double Evil.'''
    def __reduce__(self):
      '''Return the reduction tuple.'''
      return (Booster, (None, None, str(model_path)))

  with _pytest.raises(ValueError):
    restricted_loads(pickle.dumps(Evil()))


def test_restricted_loads_lightgbm_booster_valid_file_rejected(tmp_path):
  '''Test restricted loads lightgbm booster valid file rejected.'''
  import pickle
  import numpy as np
  import pytest as _pytest
  _pytest.importorskip('lightgbm')
  from forecast_ml_framework.core.artifact_store import restricted_loads
  from lightgbm import Booster, LGBMRegressor
  x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
  y = np.array([1.0, 2.0, 3.0, 4.0])
  model = LGBMRegressor(n_estimators=2, verbose=-1).fit(x, y)
  model_path = tmp_path / 'valid.txt'
  model.booster_.save_model(str(model_path))
  assert model_path.is_file()

  class Evil:
    '''Test double Evil.'''
    def __reduce__(self):
      '''Return the reduction tuple.'''
      return (Booster, (None, None, str(model_path)))

  with _pytest.raises(ValueError):
    restricted_loads(pickle.dumps(Evil()))


def test_restricted_loads_xgboost_booster_kwargs_model_file_rejected():
  '''Test restricted loads xgboost booster kwargs model file rejected.'''
  import copyreg
  import pickle
  import pytest as _pytest
  _pytest.importorskip('xgboost')
  from forecast_ml_framework.core.artifact_store import restricted_loads
  from xgboost import Booster

  class Evil:
    '''Test double Evil.'''
    def __reduce__(self):
      '''Return the reduction tuple.'''
      return (
          copyreg.__newobj_ex__,
          (Booster, (), {'model_file': '/etc/passwd'}))

  with _pytest.raises(ValueError) as excinfo:
    restricted_loads(pickle.dumps(Evil()))
  assert '/etc/passwd' not in str(excinfo.value)


def test_restricted_loads_lightgbm_booster_kwargs_model_file_rejected():
  '''Test restricted loads lightgbm booster kwargs model file rejected.'''
  import copyreg
  import pickle
  import pytest as _pytest
  _pytest.importorskip('lightgbm')
  from forecast_ml_framework.core.artifact_store import restricted_loads
  from lightgbm import Booster

  class Evil:
    '''Test double Evil.'''
    def __reduce__(self):
      '''Return the reduction tuple.'''
      return (
          copyreg.__newobj_ex__,
          (Booster, (), {'model_file': '/etc/passwd'}))

  with _pytest.raises(ValueError) as excinfo:
    restricted_loads(pickle.dumps(Evil()))
  assert '/etc/passwd' not in str(excinfo.value)


def test_restricted_loads_booster_fifo_rejected_promptly(tmp_path):
  '''Test restricted loads booster fifo rejected promptly.'''
  import os
  import pickle
  import threading
  import pytest as _pytest
  _pytest.importorskip('xgboost')
  from forecast_ml_framework.core.artifact_store import restricted_loads
  from xgboost import Booster
  fifo_path = tmp_path / 'fifo'
  os.mkfifo(str(fifo_path))

  class Evil:
    '''Test double Evil.'''
    def __reduce__(self):
      '''Return the reduction tuple.'''
      return (Booster, (None, None, str(fifo_path)))

  raw = pickle.dumps(Evil())
  outcome = {}

  def run():
    '''Run.'''
    try:
      outcome['value'] = restricted_loads(raw)
    except BaseException as exc:  # pylint: disable=broad-exception-caught
      outcome['error'] = exc

  thread = threading.Thread(target=run, daemon=True)
  thread.start()
  thread.join(timeout=5)
  assert not thread.is_alive(), 'restricted_loads blocked on FIFO'
  assert isinstance(outcome.get('error'), ValueError)


def test_restricted_loads_oversized_pickle_rejected(monkeypatch):
  '''Test restricted loads oversized pickle rejected.'''
  import pickle
  from forecast_ml_framework.core.artifact_store import restricted_loads
  raw = pickle.dumps({'a': [1, 2, 3]})
  assert len(raw) > 10
  monkeypatch.setenv('MAX_ARTIFACT_BYTES', '10')
  with pytest.raises(ValueError, match='MAX_ARTIFACT_BYTES'):
    restricted_loads(raw)


def test_local_store_oversized_pickle_rejected(monkeypatch, tmp_path):
  '''Test local store oversized pickle rejected.'''
  import pickle
  from forecast_ml_framework.core.artifact_store import LocalArtifactStore
  store = LocalArtifactStore(str(tmp_path))
  (tmp_path / 'big.pkl').write_bytes(pickle.dumps({'a': [1, 2, 3]}))
  monkeypatch.setenv('MAX_ARTIFACT_BYTES', '10')
  with pytest.raises(ValueError, match='MAX_ARTIFACT_BYTES'):
    store.load('big.pkl')


def test_decode_payload_pickle_text_and_raw():
  '''Test decode payload pickle text and raw.'''
  import pickle
  from forecast_ml_framework.core.artifact_store import _decode_payload
  assert _decode_payload(pickle.dumps([1, 2])) == [1, 2]
  assert _decode_payload('hello'.encode('utf-8')) == 'hello'
  assert _decode_payload(b'\xff\xfe\x00binary') == b'\xff\xfe\x00binary'
  assert _decode_payload(b'') == ''


def test_validate_cloud_key_rejects_absolute_and_escape():
  '''Test validate cloud key rejects absolute and escape.'''
  from forecast_ml_framework.core.artifact_store import _validate_cloud_key
  for bad in ('/abs/key', '\\abs\\key', '../escape', 'a/../b',
              'a\\..\\b', '..', 'x/../../y'):
    with pytest.raises(ValueError):
      _validate_cloud_key(bad)
  assert _validate_cloud_key('a/b.txt') == 'a/b.txt'
  assert _validate_cloud_key('') == ''
  assert _validate_cloud_key('a..b/c') == 'a..b/c'


def test_local_store_bytes_and_empty_path(tmp_path):
  '''Test local store bytes and empty path.'''
  from forecast_ml_framework.core.artifact_store import LocalArtifactStore
  store = LocalArtifactStore(str(tmp_path))
  assert store._full_path('') == str(tmp_path.resolve())
  store.save('raw.bin', b'\xff\x00\xfe')
  assert store.load('raw.bin') == b'\xff\x00\xfe'
  assert store.exists('raw.bin') is True
  assert store.list() == ['raw.bin']


def _fake_google(monkeypatch):
  '''Fake google.'''
  import io
  import sys
  import types

  class FakeBlob:
    '''Test double FakeBlob.'''
    def __init__(self, bucket, name):
      '''Initialize the instance.'''
      self._bucket = bucket
      self.name = name

    def upload_from_string(self, data):
      '''Upload from string.'''
      self._bucket.objects[self.name] = data

    def open(self, mode='rb'):
      '''Open.'''
      return io.BytesIO(self._bucket.objects[self.name])

    def download_as_bytes(self):
      '''Download as bytes.'''
      raise AssertionError('must stream via open(), not download whole')

    def exists(self):
      '''Exists.'''
      return self.name in self._bucket.objects

  class FakeBucket:
    '''Test double FakeBucket.'''
    def __init__(self, name):
      '''Initialize the instance.'''
      self.name = name
      self.objects = {}

    def blob(self, key):
      '''Blob.'''
      return FakeBlob(self, key)

    def list_blobs(self, prefix=''):
      '''List blobs.'''
      return [FakeBlob(self, k) for k in sorted(self.objects)
              if k.startswith(prefix)]

  shared_bucket = FakeBucket('bkt')

  class FakeClient:
    '''Test double FakeClient.'''
    def __init__(self):
      '''Initialize the instance.'''
      self._bucket = shared_bucket

    def bucket(self, name):
      '''Bucket.'''
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
  '''Test gcs store branches via fake sdk.'''
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
  '''Test gcs client missing sdk raises import error.'''
  import sys
  monkeypatch.setitem(sys.modules, 'google', None)
  from forecast_ml_framework.core.artifact_store import GCSArtifactStore
  with pytest.raises(ImportError, match='google-cloud-storage'):
    GCSArtifactStore('bkt')._client()


def _fake_boto3(monkeypatch):
  '''Fake boto3.'''
  import io
  import sys
  import types

  class FakePaginator:
    '''Test double FakePaginator.'''
    def __init__(self, client):
      '''Initialize the instance.'''
      self._client = client

    def paginate(self, Bucket, Prefix=''):
      '''Paginate.'''
      keys = sorted(k for k in self._client.objects
                    if k.startswith(Prefix))
      yield {'Contents': [{'Key': k} for k in keys]} if keys else {}

  class FakeS3:
    '''Test double FakeS3.'''
    def __init__(self):
      '''Initialize the instance.'''
      self.objects = {}

    def put_object(self, Bucket, Key, Body):
      '''Put object.'''
      assert Bucket == 'bkt'
      self.objects[Key] = Body

    def get_object(self, Bucket, Key):
      '''Get object.'''
      return {'Body': io.BytesIO(self.objects[Key])}

    def head_object(self, Bucket, Key):
      '''Head object.'''
      if Key not in self.objects:
        raise KeyError(Key)
      return {}

    def get_paginator(self, name):
      '''Get paginator.'''
      assert name == 'list_objects_v2'
      return FakePaginator(self)

  client = FakeS3()
  boto_mod = types.ModuleType('boto3')
  boto_mod.client = lambda svc: client
  monkeypatch.setitem(sys.modules, 'boto3', boto_mod)
  return client


def test_s3_store_branches_via_fake_sdk(monkeypatch):
  '''Test s3 store branches via fake sdk.'''
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
  '''Test s3 client missing sdk raises import error.'''
  import sys
  monkeypatch.setitem(sys.modules, 'boto3', None)
  from forecast_ml_framework.core.artifact_store import S3ArtifactStore
  with pytest.raises(ImportError, match='boto3 is required'):
    S3ArtifactStore('bkt')._client()


def test_create_artifact_store_nested_and_aliases(tmp_path):
  '''Test create artifact store nested and aliases.'''
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
  '''Test strip sql literals blanks quoted keywords.'''
  from forecast_ml_framework.core.datasets import _strip_sql_literals
  out = _strip_sql_literals('SELECT \'DROP TABLE x\' AS a, "DELETE" FROM t')
  # R8-01: single-quoted literals are blanked, but double-quoted
  # identifiers stay visible so the denylist can see quoted names.
  assert 'DROP' not in out and 'DELETE' in out
  assert 'SELECT' in out and 'FROM' in out
  escaped = _strip_sql_literals('SELECT \'it\'\'s DROP\' FROM t')
  assert 'DROP' not in escaped
  unterminated = _strip_sql_literals('SELECT \'abc')
  assert 'abc' not in unterminated
  assert _strip_sql_literals('SELECT 1') == 'SELECT 1'


def test_validate_blocks_double_quoted_forbidden_function():
  '''Double-quoted identifier names stay visible to denylist (R8-01).'''
  from forecast_ml_framework.core.datasets import (
      _strip_sql_literals, _validate_read_only_query)
  out = _strip_sql_literals(
      'SELECT "pg_read_binary_file"(\'/etc/passwd\')')
  assert 'pg_read_binary_file' in out
  for bad in ('SELECT "pg_read_binary_file"(\'/etc/passwd\')',
              'SELECT "PG_READ_BINARY_FILE"(\'/etc/passwd\')',
              'SELECT "writefile"(\'/tmp/x\', \'hi\')'):
    with pytest.raises(ValueError, match='Forbidden'):
      _validate_read_only_query(bad)


def test_validate_allows_double_quoted_identifiers():
  '''Legit double-quoted identifier columns still pass (R8-01).'''
  from forecast_ml_framework.core.datasets import _validate_read_only_query
  assert _validate_read_only_query(
      'SELECT "user_id", "updated_at", "created_at" FROM t') is not None
  assert _validate_read_only_query(
      'SELECT "my_readfile" FROM t') is not None
  # Single-quoted literals are still blanked, not scanned.
  assert _validate_read_only_query(
      'SELECT \'pg_read_binary_file\' FROM t') is not None


def test_validate_read_only_query_edges():
  '''Test validate read only query edges.'''
  from forecast_ml_framework.core.datasets import _validate_read_only_query
  assert _validate_read_only_query('SELECT 1;') == 'SELECT 1;'
  assert _validate_read_only_query('SELECT 1 ; ') == 'SELECT 1 ; '
  assert _validate_read_only_query(
      'SELECT \';\' FROM t WHERE x = \'a;b\'') is not None
  for bad in (None, '', '   ', 123, b'SELECT 1'):
    with pytest.raises(ValueError):
      _validate_read_only_query(bad)


def test_sql_dataset_dict_and_kwargs_alias_merging():
  '''Test sql dataset dict and kwargs alias merging.'''
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
  '''Test sql dataset engine branches.'''
  from forecast_ml_framework.core.datasets import SQLQueryDataSet
  class _Engine:
    '''Test double _Engine.'''
    def connect(self):
      '''Connect.'''
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
  '''Test sql dataset engine missing sqlalchemy.'''
  import sys
  monkeypatch.setitem(sys.modules, 'sqlalchemy', None)
  from forecast_ml_framework.core.datasets import SQLQueryDataSet
  with pytest.raises(ImportError, match='sqlalchemy is required'):
    SQLQueryDataSet(connection='sqlite:///:memory:',
                    query='SELECT 1')._engine()


def test_spark_dataset_aliases_and_fake_session():
  '''Test spark dataset aliases and fake session.'''
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
    '''Test double FakeSpark.'''
    def sql(self, query):
      '''Sql.'''
      return ('ran', query)

  ds3 = SparkSQLDataSet(spark=FakeSpark(), query='SELECT 1')
  assert ds3.load() == ('ran', 'SELECT 1')
  assert ds3._describe()['type'] == 'SparkSQLDataSet'
  assert ds3.exists() is True
  with pytest.raises(NotImplementedError):
    ds3._save(object())


def test_spark_dataset_no_query_and_no_session(monkeypatch):
  '''Test spark dataset no query and no session.'''
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
  '''Test restricted pickle dataset roundtrip and aliases.'''
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
  '''Os system payload.'''
  import os

  class Evil:
    '''Test double Evil.'''
    def __reduce__(self):
      '''Return the reduction tuple.'''
      return (os.system, ('echo pwned',))

  return Evil()


# --- loop4: spark_factory coverage ---


def test_spark_section_and_enabled_variants():
  '''Test spark section and enabled variants.'''
  from forecast_ml_framework.core.spark_factory import (
    _spark_enabled, _spark_section)
  assert _spark_section(None) == {}
  assert _spark_section('x') == {}
  assert _spark_section({'spark': True}) == {'enabled': True}
  assert _spark_section({'spark': {'a': 1}}) == {'a': 1}
  assert _spark_enabled(None) is False
  assert _spark_enabled({'spark_enabled': True}) is True
  assert _spark_enabled({'use_spark': True}) is True
  assert _spark_enabled({'spark': {'enabled': True}}) is True
  assert _spark_enabled({'spark': {'master': 'local'}}) is True
  assert _spark_enabled({'spark': True}) is True
  assert _spark_enabled({'other': 1}) is False
  assert _spark_enabled({'enabled': False}) is False
  assert _spark_enabled({'enabled': True,
    'spark': {'master': 'local'}}) is True
  assert _spark_enabled({'spark_enabled': False,
    'spark': {'master': 'local'}}) is False


def test_spark_builder_config_variants():
  '''Test spark builder config variants.'''
  from forecast_ml_framework.core.spark_factory import (
    spark_builder_config)
  cfg = spark_builder_config({'spark': {
    'dynamic_allocation': False, 'shuffle_partitions': 7,
    'config': {'spark.executor.memory': '2g', 'master': 'yarn',
               'enabled': True, 'app_name': 'x',
               'shuffle_partitions': 9}}})
  assert cfg['spark.sql.shuffle.partitions'] == 7
  assert cfg['spark.dynamicAllocation.enabled'] == 'false'
  assert cfg['spark.executor.memory'] == '2g'
  assert 'master' not in cfg and 'enabled' not in cfg
  with pytest.raises(ValueError) as excinfo:
    spark_builder_config({'spark_params': {'a': 1}})
  assert '\'a\'' in str(excinfo.value)
  with pytest.raises(ValueError) as excinfo:
    spark_builder_config({'spark_parameters': {'b': 2}})
  assert '\'b\'' in str(excinfo.value)
  cfg13 = spark_builder_config({'spark': {'spark_params': {
    'spark.sql.shuffle.partitions': 13}}})
  assert cfg13['spark.sql.shuffle.partitions'] == 13
  cfg14 = spark_builder_config({'spark': {'spark_parameters': {
    'shuffle_partitions': 14}}})
  assert cfg14['spark.sql.shuffle.partitions'] == 14
  with pytest.raises(ValueError) as excinfo:
    spark_builder_config({'spark': {'dynamic_allocation': True,
      'spark_params': {'extra': '1'}}})
  assert '\'extra\'' in str(excinfo.value)
  default = spark_builder_config(None)
  assert default['spark.sql.shuffle.partitions'] == 200
  assert default['spark.dynamicAllocation.enabled'] == 'true'
  assert default['spark.sql.adaptive.enabled'] == 'true'


def test_pandas_limit_from_config_variants(monkeypatch):
  '''Test pandas limit from config variants.'''
  from forecast_ml_framework.core.spark_factory import (
    _pandas_limit_from_config)
  assert _pandas_limit_from_config(None) == 5000000
  assert _pandas_limit_from_config({'max_rows': 5}) == 5
  assert _pandas_limit_from_config(
    {'max_rows_for_pandas': None, 'max_rows': 6}) == 6
  assert _pandas_limit_from_config(
    {'pandas_conversion_params': {'max_rows': 9}}) == 9
  assert _pandas_limit_from_config(
    {'pandas_conversion_params': {}}) == 5000000
  assert _pandas_limit_from_config(
    {'pandas_conversion': {'max_rows_for_pandas': 8}}) == 8
  assert _pandas_limit_from_config({'spark': {'max_rows': 7}}) == 7
  monkeypatch.setenv('PANDAS_MAX_ROWS', '123')
  assert _pandas_limit_from_config({}) == 123
  assert _pandas_limit_from_config({'spark': {}}) == 123
  monkeypatch.setenv('PANDAS_MAX_ROWS', 'abc')
  assert _pandas_limit_from_config({}) == 5000000


def test_frame_counts_and_pandas_limits(monkeypatch):
  '''Test frame counts and pandas limits.'''
  import sys
  import types
  import pandas as pd
  from forecast_ml_framework.core.spark_factory import (
    SparkSessionFactory, _frame_col_count, _frame_row_count,
    check_pandas_limits)
  frame = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
  assert _frame_row_count(frame) == 2
  assert _frame_col_count(frame) == 2

  class Countable:
    '''Test double Countable.'''
    def count(self):
      '''Count.'''
      return 3
  assert _frame_row_count(Countable()) == 3

  class BadCount:
    '''Test double BadCount.'''
    def count(self):
      '''Count.'''
      raise RuntimeError('no')
    def __len__(self):
      '''Return the length.'''
      return 4
  assert _frame_row_count(BadCount()) is None

  class LenOnly:
    '''Test double LenOnly.'''
    def __len__(self):
      '''Return the length.'''
      return 4
  assert _frame_row_count(LenOnly()) == 4

  class BadCountNoLen:
    '''Test double BadCountNoLen.'''
    def count(self):
      '''Count.'''
      raise RuntimeError('no')
  assert _frame_row_count(BadCountNoLen()) is None
  assert _frame_row_count(object()) is None

  fake_pd = types.ModuleType('pandas')
  monkeypatch.setitem(sys.modules, 'pandas', fake_pd)
  assert _frame_row_count(Countable()) == 3
  monkeypatch.undo()
  import pandas as pd  # noqa: E402

  class BadCols:
    '''Test double BadCols.'''
    columns = object()
  assert _frame_col_count(BadCols()) is None
  assert _frame_col_count(object()) is None
  assert check_pandas_limits(frame, {}) is None
  assert check_pandas_limits(object(), {}) is None
  with pytest.raises(ValueError, match='max_rows_for_pandas'):
    check_pandas_limits(frame, {'max_rows_for_pandas': 1})

  class CountNoCols:
    '''Test double CountNoCols.'''
    def count(self):
      '''Count.'''
      return 10
  with pytest.raises(ValueError, match='max_rows_for_pandas'):
    check_pandas_limits(CountNoCols(), {'max_rows_for_pandas': 1})
  assert SparkSessionFactory.check_pandas_limits(frame, {}) is None
  sentinel = object()
  assert SparkSessionFactory.to_pandas(sentinel) is sentinel

  class WithPandas:
    '''Test double WithPandas.'''
    def toPandas(self):
      '''ToPandas.'''
      return 'converted'
  assert SparkSessionFactory.to_pandas(WithPandas()) == 'converted'


def _install_fake_builder(monkeypatch, builder):
  '''Install fake builder.'''
  import sys
  import types
  pyspark = types.ModuleType('pyspark')
  sql = types.ModuleType('pyspark.sql')

  class SparkSession:
    '''Test double SparkSession.'''
    pass
  SparkSession.builder = builder
  sql.SparkSession = SparkSession
  pyspark.sql = sql
  monkeypatch.setitem(sys.modules, 'pyspark', pyspark)
  monkeypatch.setitem(sys.modules, 'pyspark.sql', sql)


def test_get_spark_with_fake_builder(monkeypatch):
  '''Test get spark with fake builder.'''
  from forecast_ml_framework.core.spark_factory import (
    SparkSessionFactory)

  class Builder:
    '''Test double Builder.'''
    def __init__(self):
      '''Initialize the instance.'''
      self.calls = []
    def master(self, value):
      '''Master.'''
      self.calls.append(('master', value))
      return self
    def appName(self, value):
      '''AppName.'''
      self.calls.append(('appName', value))
      return self
    def config(self, key, value):
      '''Config.'''
      self.calls.append(('config', key, value))
      return self
    def getOrCreate(self):
      '''GetOrCreate.'''
      return 'session'
  builder = Builder()
  _install_fake_builder(monkeypatch, builder)
  assert SparkSessionFactory.get_spark(
    {'spark': {'enabled': True, 'master': 'yarn',
               'app_name': 'app'}}) == 'session'
  assert ('master', 'yarn') in builder.calls
  assert ('appName', 'app') in builder.calls
  assert any(c[0] == 'config' for c in builder.calls)


def test_get_spark_builder_failure_returns_none(monkeypatch):
  '''Test get spark builder failure returns none.'''
  from forecast_ml_framework.core.spark_factory import (
    SparkSessionFactory)

  class Builder:
    '''Test double Builder.'''
    def master(self, value):
      '''Master.'''
      raise RuntimeError('boom')
  _install_fake_builder(monkeypatch, Builder())
  assert SparkSessionFactory.get_spark(
    {'spark': {'enabled': True}}) is None


def test_sql_denylist_blocks_privileged_functions():
  '''Read-only validator blocks privileged pg_* functions (A8).'''
  from forecast_ml_framework.core.datasets import _validate_read_only_query
  for bad in ('SELECT pg_read_file(1)',
              'SELECT pg_ls_dir(1)',
              'SELECT lo_export(1, 1)',
              'SELECT dblink(1)',
              'SELECT pg_advisory_lock(1)'):
    with pytest.raises(ValueError, match='Forbidden'):
      _validate_read_only_query(bad)


def test_sql_denylist_blocks_server_file_primitives():
  '''Read-only validator blocks server-file primitives (S7-01).'''
  from forecast_ml_framework.core.datasets import _validate_read_only_query
  bad = (
      'SELECT pg_read_binary_file(\'/etc/passwd\')',
      'SELECT PG_READ_BINARY_FILE(\'/etc/passwd\')',
      'SELECT pg_catalog.pg_read_binary_file(\'/etc/passwd\')',
      'SELECT lo_import(\'/tmp/x\')',
      'SELECT pg_stat_file(\'/etc/passwd\')',
      'SELECT pg_ls_waldir()',
      'SELECT pg_logdir_ls()',
      'SELECT readfile(\'/etc/passwd\')',
      'SELECT writefile(\'/tmp/x\', \'hi\')',
      'SELECT pg_read_binary_file_extra(\'/etc/passwd\')',
      'SELECT pg_stat_file_data(\'/etc/passwd\')',
      'SELECT pg_ls_waldir_x()',
      'SELECT pg_logdir_ls_backup()',
      'SELECT lo_import_1(\'/tmp/x\')',
      'SELECT readfile_data(\'/etc/passwd\')',
      'SELECT writefile2(\'/tmp/x\', \'hi\')',
  )
  for query in bad:
    with pytest.raises(ValueError, match='Forbidden'):
      _validate_read_only_query(query)
  assert _validate_read_only_query(
      'SELECT id, val FROM t') is not None
  assert _validate_read_only_query(
      'SELECT updated_at, created_at FROM t') is not None
  assert _validate_read_only_query(
      'SELECT my_readfile FROM t') is not None


def test_core_params_canonical_helpers():
  '''core.params owns canonical _int/_float/_get/_parse_datetime (F8).'''
  from datetime import datetime, timezone

  from forecast_ml_framework.core import params as core_params
  assert core_params.parse_int('3', 0) == 3
  assert core_params.parse_int('bad', 9) == 9
  assert core_params.parse_float('1.5', 0.0) == 1.5
  assert core_params.parse_float(None, 2.5) == 2.5
  assert core_params.get_first({'a': None, 'b': 2}, 'a', 'b') == 2
  assert core_params.get_first({'a': 1}, 'a', 'b') == 1
  assert core_params.get_first({}, 'a') is None
  parsed = core_params.parse_datetime('2026-01-01T00:00:00Z')
  assert parsed == datetime(2026, 1, 1, tzinfo=timezone.utc)
  with pytest.raises(ValueError):
    core_params.parse_datetime('not-a-date')


def test_core_init_lazy_no_kedro_import():
  '''Importing core does not eagerly import kedro (V11).'''
  import subprocess
  import sys
  code = (
    'import sys; '
    'import forecast_ml_framework.core as core; '
    'assert "kedro" not in sys.modules, "kedro eagerly imported"; '
    'assert "kedro.io" not in sys.modules; '
    'from forecast_ml_framework.core.config_loader import ConfigLoader; '
    'c = ConfigLoader(variables={"a": 1}, parameters={"b": 2}); '
    'assert c.get("a") == 1 and c.get("b") == 2; '
    'assert core.ConfigLoader is ConfigLoader; '
    'print("lazy-ok")')
  import os
  from pathlib import Path
  root = Path(__file__).resolve().parents[1]
  env = dict(os.environ)
  env['PYTHONPATH'] = str(root / 'src') + os.pathsep + env.get(
    'PYTHONPATH', '')
  result = subprocess.run(
    [sys.executable, '-c', code], capture_output=True, text=True,
    cwd=str(root), env=env)
  assert result.returncode == 0, result.stderr
  assert 'lazy-ok' in result.stdout


# --- X2: chunked capped reads (C2-F2/R-03) ---

_CHUNK_BYTES = 1024 * 1024


class _TrackingStream:
  '''File-like stream counting bytes returned by read().'''

  def __init__(self, data):
    '''Initialize the instance.'''
    import io
    self._buf = io.BytesIO(data)
    self.bytes_read = 0

  def read(self, size=-1):
    '''Read.'''
    chunk = self._buf.read(size)
    self.bytes_read += len(chunk)
    return chunk

  def __enter__(self):
    '''Enter the context.'''
    return self

  def __exit__(self, *args):
    '''Exit the context.'''
    return False


def _patch_tracking_open(monkeypatch):
  '''Patch builtins.open to count bytes read; return counter dict.'''
  import builtins
  real_open = builtins.open
  counter = {'bytes': 0}

  class _TrackingFile:
    '''File wrapper counting bytes returned by read().'''

    def __init__(self, handle):
      '''Initialize the instance.'''
      self._handle = handle

    def read(self, size=-1):
      '''Read.'''
      data = self._handle.read(size)
      counter['bytes'] += len(data)
      return data

    def __enter__(self):
      '''Enter the context.'''
      return self

    def __exit__(self, *args):
      '''Exit the context.'''
      return False

  def _tracking_open(file, mode='r', *args, **kwargs):
    '''Tracking open.'''
    return _TrackingFile(real_open(file, mode, *args, **kwargs))

  monkeypatch.setattr(builtins, 'open', _tracking_open)
  return counter


@pytest.mark.parametrize('cap,size', [
  (10, 3 * _CHUNK_BYTES),
  (2 * _CHUNK_BYTES, 5 * _CHUNK_BYTES),
  (0, 1),
])
def test_capped_read_rejects_over_cap_bounded(cap, size):
  '''Fails if the capped reader materialises the whole payload
  before rejecting it instead of stopping near cap+chunk.'''
  from forecast_ml_framework.core.artifact_store import _read_capped
  stream = _TrackingStream(b'x' * size)
  with pytest.raises(ValueError, match='MAX_ARTIFACT_BYTES'):
    _read_capped(stream, cap)
  assert stream.bytes_read <= cap + _CHUNK_BYTES


def test_capped_read_returns_bytes_under_cap():
  '''Capped reader returns the full payload when it fits.'''
  from forecast_ml_framework.core.artifact_store import _read_capped
  assert _read_capped(_TrackingStream(b'hello'), 1024) == b'hello'
  assert _read_capped(_TrackingStream(b''), 1024) == b''


def test_local_store_sparse_over_cap_rejected_without_full_read(
    monkeypatch, tmp_path):
  '''Fails if LocalArtifactStore.load reads a sparse oversized
  file fully into memory before rejecting it.'''
  from forecast_ml_framework.core.artifact_store import LocalArtifactStore
  cap = _CHUNK_BYTES
  monkeypatch.setenv('MAX_ARTIFACT_BYTES', str(cap))
  target = tmp_path / 'big.bin'
  with open(target, 'wb') as handle:
    handle.truncate(cap * 8)
  counter = _patch_tracking_open(monkeypatch)
  with pytest.raises(ValueError, match='MAX_ARTIFACT_BYTES'):
    LocalArtifactStore(str(tmp_path)).load('big.bin')
  assert counter['bytes'] <= cap + _CHUNK_BYTES


def test_restricted_pickle_dataset_sparse_over_cap_rejected(
    monkeypatch, tmp_path):
  '''Fails if RestrictedPickleDataset._load reads an oversized
  pickle file fully into memory before rejecting it.'''
  from forecast_ml_framework.core.datasets import RestrictedPickleDataset
  cap = _CHUNK_BYTES
  monkeypatch.setenv('MAX_ARTIFACT_BYTES', str(cap))
  target = tmp_path / 'big.pkl'
  with open(target, 'wb') as handle:
    handle.truncate(cap * 8)
  counter = _patch_tracking_open(monkeypatch)
  with pytest.raises(ValueError, match='MAX_ARTIFACT_BYTES'):
    RestrictedPickleDataset(str(target))._load()
  assert counter['bytes'] <= cap + _CHUNK_BYTES


def test_s3_store_over_cap_rejected_without_full_read(monkeypatch):
  '''Fails if S3ArtifactStore.load reads the whole Body stream
  before rejecting it as oversized.'''
  import sys
  import types
  from forecast_ml_framework.core.artifact_store import S3ArtifactStore
  monkeypatch.setenv('MAX_ARTIFACT_BYTES', '10')
  stream = _TrackingStream(b'x' * (3 * _CHUNK_BYTES))

  class FakeS3:
    '''Test double FakeS3.'''
    def get_object(self, Bucket, Key):
      '''Get object.'''
      return {'Body': stream}

  boto_mod = types.ModuleType('boto3')
  boto_mod.client = lambda svc: FakeS3()
  monkeypatch.setitem(sys.modules, 'boto3', boto_mod)
  with pytest.raises(ValueError, match='MAX_ARTIFACT_BYTES'):
    S3ArtifactStore('bkt').load('k.bin')
  assert stream.bytes_read <= 10 + _CHUNK_BYTES


def test_gcs_store_over_cap_rejected_without_full_download(monkeypatch):
  '''Fails if GCSArtifactStore.load downloads the whole blob
  before rejecting it as oversized instead of streaming capped.'''
  import sys
  import types
  from forecast_ml_framework.core.artifact_store import GCSArtifactStore
  monkeypatch.setenv('MAX_ARTIFACT_BYTES', '10')
  stream = _TrackingStream(b'x' * (3 * _CHUNK_BYTES))

  class FakeBlob:
    '''Test double FakeBlob.'''
    def open(self, mode='rb'):
      '''Open.'''
      return stream

    def download_as_bytes(self):
      '''Download as bytes.'''
      raise AssertionError('must stream via open(), not download whole')

  class FakeBucket:
    '''Test double FakeBucket.'''
    def blob(self, key):
      '''Blob.'''
      return FakeBlob()

  class FakeClient:
    '''Test double FakeClient.'''
    def bucket(self, name):
      '''Bucket.'''
      return FakeBucket()

  storage_mod = types.ModuleType('google.cloud.storage')
  storage_mod.Client = FakeClient
  cloud_mod = types.ModuleType('google.cloud')
  cloud_mod.storage = storage_mod
  google_mod = types.ModuleType('google')
  google_mod.cloud = cloud_mod
  monkeypatch.setitem(sys.modules, 'google', google_mod)
  monkeypatch.setitem(sys.modules, 'google.cloud', cloud_mod)
  monkeypatch.setitem(sys.modules, 'google.cloud.storage', storage_mod)
  with pytest.raises(ValueError, match='MAX_ARTIFACT_BYTES'):
    GCSArtifactStore('bkt').load('k.bin')
  assert stream.bytes_read <= 10 + _CHUNK_BYTES


def test_gcs_store_no_open_raises_promptly_without_download(monkeypatch):
  '''Fails if GCS load materialises the whole blob via
  download_as_bytes when the blob has no streaming open().'''
  import sys
  import types
  from forecast_ml_framework.core.artifact_store import GCSArtifactStore
  calls = {'download': 0}

  class FakeBlob:
    '''Blob without open(): download must never be called.'''

    def download_as_bytes(self):
      '''Download as bytes.'''
      calls['download'] += 1
      return b'x' * (3 * _CHUNK_BYTES)

  class FakeBucket:
    '''Test double FakeBucket.'''
    def blob(self, key):
      '''Blob.'''
      return FakeBlob()

  class FakeClient:
    '''Test double FakeClient.'''
    def bucket(self, name):
      '''Bucket.'''
      return FakeBucket()

  storage_mod = types.ModuleType('google.cloud.storage')
  storage_mod.Client = FakeClient
  cloud_mod = types.ModuleType('google.cloud')
  cloud_mod.storage = storage_mod
  google_mod = types.ModuleType('google')
  google_mod.cloud = cloud_mod
  monkeypatch.setitem(sys.modules, 'google', google_mod)
  monkeypatch.setitem(sys.modules, 'google.cloud', cloud_mod)
  monkeypatch.setitem(sys.modules, 'google.cloud.storage', storage_mod)
  with pytest.raises(RuntimeError, match='GCS blob has no streaming open'):
    GCSArtifactStore('bkt').load('k.bin')
  assert calls['download'] == 0
