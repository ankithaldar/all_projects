#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the batch scoring pipeline (Phase 2).'''

import logging

import numpy as np
import pandas as pd
import pytest

from forecast_ml_framework.pipelines.scoring.nodes import (
  _resolve_loader,
  load_scoring_data,
  predict,
  write_predictions,
)
from forecast_ml_framework.pipelines.scoring.pipeline import (
  create_pipeline,
)


class _FakeModel:
  '''Minimal classifier returning deterministic predictions.'''

  def predict(self, x):
    '''Return one label per row from the first feature.'''
    return (np.asarray(x, dtype=float)[:, 0] > 0).astype(int)

  def predict_proba(self, x):
    '''Return two-column proba aligned with predict().'''
    pos = np.where(
      np.asarray(x, dtype=float)[:, 0] > 0, 0.9, 0.1)
    return np.column_stack([1.0 - pos, pos])


class _FakeRegressor:
  '''Minimal regressor without proba.'''

  def predict(self, x):
    '''Return the first feature scaled by two.'''
    return 2.0 * np.asarray(x, dtype=float)[:, 0]


class _SparkLike:
  '''Spark-like frame exposing toPandas only.'''

  def __init__(self, frame):
    self._frame = frame

  def toPandas(self):
    '''Return the wrapped frame.'''
    return self._frame


class _RecordingDataset:
  '''Fake catalog destination recording save calls.'''

  def __init__(self):
    self.saved = []

  def save(self, data):
    '''Record the payload.'''
    self.saved.append(data)


class _ReadOnlyDataset:
  '''Fake destination mimicking core read-only datasets.'''

  def save(self, data):
    '''Raise like SQLQueryDataSet.save.'''
    raise NotImplementedError('read-only')


def _frame():
  '''Return small scoring frame.'''
  return pd.DataFrame({
    'id': [1, 2, 3],
    'f1': [1.0, -1.0, 2.0],
    'f2': [0.0, 0.0, 0.0],
  })


def _params(**overrides):
  '''Return scoring params for tests.'''
  base = {
    'id_cols': ['id'],
    'feature_cols': ['f1', 'f2'],
    'task_type': 'classification',
    'score_model_id': 'model-v1',
  }
  base.update(overrides)
  return base


def test_load_scoring_data_pandas_passthrough():
  '''Pandas input is copied and returned unchanged.'''
  frame = _frame()
  out = load_scoring_data(frame, _params())
  pd.testing.assert_frame_equal(out, frame)
  assert out is not frame


def test_load_scoring_data_spark_like_converted():
  '''Spark-like input is converted via toPandas.'''
  out = load_scoring_data(_SparkLike(_frame()), _params())
  pd.testing.assert_frame_equal(out, _frame())


def test_load_scoring_data_empty_warns_not_crashes(caplog):
  '''Empty input returns an empty frame with a warning.'''
  empty = pd.DataFrame({'id': pd.Series([], dtype=int),
                        'f1': pd.Series([], dtype=float)})
  with caplog.at_level(logging.WARNING):
    out = load_scoring_data(empty, _params())
  assert len(out) == 0
  assert any('empty' in r.message.lower() for r in caplog.records)


def test_predict_classification_schema_and_proba():
  '''Predict adds prediction, proba, ntile, model id, timestamp.'''
  out = predict(_frame(), _params(), model_loader=lambda: _FakeModel())
  assert list(out.columns) == [
    'id', 'prediction', 'proba', 'score_decile', 'score_centile',
    'high_score_ind', 'score_model_id', 'scored_at']
  for col in ('score_decile', 'score_centile', 'high_score_ind'):
    assert col in out.columns
  assert out['prediction'].tolist() == [1, 0, 1]
  assert out['proba'].tolist() == [0.9, 0.1, 0.9]
  assert (out['score_model_id'] == 'model-v1').all()
  assert out['scored_at'].notna().all()


def test_predict_uses_injected_loader_callable():
  '''Loader callable is invoked to obtain the model.'''
  calls = []

  def loader():
    '''Record the call and return a model.'''
    calls.append(1)
    return _FakeModel()

  out = predict(_frame(), _params(), model_loader=loader)
  assert calls == [1]
  assert len(out) == 3


def test_predict_loader_from_params():
  '''Loader may also arrive inside params (file/registry agnostic).'''
  params = _params(model_loader=lambda: _FakeModel())
  out = predict(_frame(), params)
  assert out['prediction'].tolist() == [1, 0, 1]


def test_predict_accepts_model_directly():
  '''A model object passed as loader is used without calling.'''
  out = predict(_frame(), _params(), model_loader=_FakeModel())
  assert len(out) == 3


def test_predict_regression_proba_is_nan():
  '''Regressors without proba get NaN proba, not a crash.'''
  params = _params(task_type='regression')
  out = predict(_frame(), params, model_loader=lambda: _FakeRegressor())
  assert out['prediction'].tolist() == [2.0, -2.0, 4.0]
  assert out['proba'].isna().all()


def test_predict_empty_returns_schema_with_warning(caplog):
  '''Empty input yields empty predictions, warning, no load call.'''
  calls = []
  empty = pd.DataFrame({'id': pd.Series([], dtype=int),
                        'f1': pd.Series([], dtype=float),
                        'f2': pd.Series([], dtype=float)})
  with caplog.at_level(logging.WARNING):
    out = predict(empty, _params(),
                  model_loader=lambda: calls.append(1) or _FakeModel())
  assert len(out) == 0
  assert list(out.columns) == [
    'id', 'prediction', 'proba', 'score_decile', 'score_centile',
    'calib_decile', 'calib_centile', 'high_score_ind',
    'score_model_id', 'scored_at']
  assert calls == []
  assert any('empty' in r.message.lower() for r in caplog.records)


def test_write_predictions_saves_to_destination():
  '''Write node saves through a destination dataset when given.'''
  preds = predict(_frame(), _params(),
                  model_loader=lambda: _FakeModel())
  dest = _RecordingDataset()
  out = write_predictions(preds, _params(), destination=dest)
  pd.testing.assert_frame_equal(out, preds)
  assert len(dest.saved) == 1
  pd.testing.assert_frame_equal(dest.saved[0], preds)


def test_write_predictions_readonly_destination_warns(caplog):
  '''Read-only core datasets warn and pass predictions through.'''
  preds = predict(_frame(), _params(),
                  model_loader=lambda: _FakeModel())
  with caplog.at_level(logging.WARNING):
    out = write_predictions(preds, _params(),
                            destination=_ReadOnlyDataset())
  pd.testing.assert_frame_equal(out, preds)
  assert any('read-only' in r.message.lower()
             or 'not supported' in r.message.lower()
             for r in caplog.records)


def test_write_predictions_without_destination_passthrough():
  '''No destination (catalog handles IO) returns predictions.'''
  preds = predict(_frame(), _params(),
                  model_loader=lambda: _FakeModel())
  out = write_predictions(preds, _params())
  pd.testing.assert_frame_equal(out, preds)


def test_write_predictions_empty_warns_not_crashes(caplog):
  '''Empty predictions warn and return empty without saving.'''
  dest = _RecordingDataset()
  empty = pd.DataFrame({
    'id': pd.Series([], dtype=int),
    'prediction': pd.Series([], dtype=float),
    'proba': pd.Series([], dtype=float),
    'score_model_id': pd.Series([], dtype=object),
    'scored_at': pd.Series([], dtype=object),
  })
  with caplog.at_level(logging.WARNING):
    out = write_predictions(empty, _params(), destination=dest)
  assert len(out) == 0
  assert dest.saved == []
  assert any('empty' in r.message.lower() for r in caplog.records)


def test_score_data_prep_empty_registry_with_fitted_steps_raises():
  from forecast_ml_framework.pipelines.scoring.pipeline import (
    score_data_prep,
  )
  params = {'data_prep_params': [{'Normalizations': {'args': {}}}]}
  with pytest.raises(ValueError, match='fitted_registry'):
    score_data_prep(_frame(), {}, params)
  with pytest.raises(ValueError, match='fitted_registry'):
    score_data_prep(_frame(), None, params)


def test_score_data_prep_empty_registry_no_fitted_steps_identity():
  from forecast_ml_framework.pipelines.scoring.pipeline import (
    score_data_prep,
  )
  frame = _frame()
  out = score_data_prep(frame, {}, _params())
  pd.testing.assert_frame_equal(out, frame)


def test_score_data_prep_applies_non_empty_registry():
  from forecast_ml_framework.pipelines.scoring.pipeline import (
    score_data_prep,
  )
  from forecast_ml_framework.pipelines.data_prep.fitted import (
    fit_preprocess,
  )
  train = pd.DataFrame({'f1': [0.0, 10.0], 'f2': [0.0, 0.0],
                        'target': [0, 1]})
  params = [{'Normalizations': {'args': {
    'handle': 'MinMaxScaler', 'cols': {'include_cols': ['f1']}}}}]
  _, registry = fit_preprocess(train, params, {'num_cols': ['f1']})
  out = score_data_prep(_frame(), registry, _params())
  assert abs(out['f1'].iloc[0] - 0.1) < 1e-9


def test_pipeline_has_three_named_nodes():
  '''Pipeline wires score_data_prep -> predict -> write_predictions.'''
  pipe = create_pipeline()
  names = [n.name for n in pipe.nodes]
  assert names == ['score_data_prep', 'predict', 'write_predictions']
  assert len(pipe.nodes) == 3
  nodes = {n.name: n for n in pipe.nodes}
  assert 'scoring_input' in nodes['score_data_prep'].inputs
  assert 'fitted_registry' in nodes['score_data_prep'].inputs
  assert 'params:scoring_params' in nodes['score_data_prep'].inputs
  assert 'params:scoring_params' in nodes['predict'].inputs
  assert 'params:scoring_params' in nodes['write_predictions'].inputs


def test_pipeline_end_to_end_via_node_funcs():
  '''Node funcs chain load -> predict -> write without catalog.'''
  nodes = {n.name: n for n in create_pipeline().nodes}
  data = nodes['score_data_prep'].func(_frame(), {}, _params())
  preds = nodes['predict'].func(
      data, _params(), model_loader=lambda: _FakeModel())
  out = nodes['write_predictions'].func(preds, _params())
  assert out['prediction'].tolist() == [1, 0, 1]
  assert out['score_model_id'].unique().tolist() == ['model-v1']


# --- Phase 2 fix: registry, I/O wiring, caps, finite, allowlist ---


def test_registry_includes_scoring_twelve_keys():
  '''Registry returns 12 keys including scoring (Phase 3).'''
  from forecast_ml_framework.pipeline_registry import register_pipelines
  pipelines = register_pipelines()
  assert set(pipelines) == {
    'data', 'preprocessing', 'feature_selection',
    'modeling', 'evaluation', 'predict', 'scoring',
    'eda', 'monitoring', 'retraining', 'cost_tracking',
    'interpretability'}
  assert len(pipelines) == 12
  assert 'scoring' in pipelines


def test_pipeline_uses_params_io():
  '''Pipeline I/O comes from scoring_params when given.'''
  pipe = create_pipeline(scoring_params={
    'input_dataset': 'my_in', 'output_dataset': 'my_out'})
  nodes = {n.name: n for n in pipe.nodes}
  assert 'my_in' in nodes['score_data_prep'].inputs
  assert nodes['write_predictions'].outputs == ['my_out']


def test_pipeline_defaults_backcompat_io():
  '''Without params, pipeline defaults to scoring_input/predictions.'''
  nodes = {n.name: n for n in create_pipeline().nodes}
  assert 'scoring_input' in nodes['score_data_prep'].inputs
  assert nodes['write_predictions'].outputs == ['predictions']


class _RecordingModel:
  '''Model recording chunk sizes it was called with.'''

  def __init__(self):
    self.sizes = []

  def predict(self, x):
    '''Record chunk size and return labels.'''
    self.sizes.append(len(x))
    return (np.asarray(x, dtype=float)[:, 0] > 0).astype(int)


def test_predict_chunked_by_batch_size():
  '''Prediction honours batch_size in chunks.'''
  frame = pd.DataFrame({
    'id': range(5), 'f1': [1.0, -1.0, 2.0, -2.0, 3.0],
    'f2': [0.0] * 5})
  model = _RecordingModel()
  out = predict(frame, _params(batch_size=2),
                model_loader=lambda: model)
  assert model.sizes == [2, 2, 1]
  assert out['prediction'].tolist() == [1, 0, 1, 0, 1]


def test_predict_invalid_batch_size():
  '''Non-positive batch_size raises ValueError.'''
  with pytest.raises(ValueError, match='batch_size'):
    predict(_frame(), _params(batch_size=0),
            model_loader=lambda: _FakeModel())


def test_predict_max_rows_exceeded():
  '''Rows beyond max_rows raise ValueError.'''
  with pytest.raises(ValueError, match='max_rows'):
    predict(_frame(), _params(max_rows=2),
            model_loader=lambda: _FakeModel())


def test_predict_max_cols_exceeded():
  '''Cols beyond max_cols raise ValueError.'''
  with pytest.raises(ValueError, match='max_cols'):
    predict(_frame(), _params(max_cols=2),
            model_loader=lambda: _FakeModel())


def test_predict_default_max_rows_rejects_large():
  '''Default cap rejects 200k rows.'''
  frame = pd.DataFrame({
    'id': np.arange(200_000), 'f1': np.ones(200_000),
    'f2': np.zeros(200_000)})
  with pytest.raises(ValueError, match='max_rows'):
    predict(frame, _params(), model_loader=lambda: _FakeModel())


def test_predict_rejects_inf_naming_column():
  '''Inf features raise ValueError naming the column.'''
  frame = _frame()
  frame.loc[0, 'f1'] = np.inf
  with pytest.raises(ValueError, match='f1'):
    predict(frame, _params(), model_loader=lambda: _FakeModel())


def test_predict_rejects_nan_naming_column():
  '''NaN features raise ValueError naming the column.'''
  frame = _frame()
  frame.loc[1, 'f2'] = np.nan
  with pytest.raises(ValueError, match='f2'):
    predict(frame, _params(), model_loader=lambda: _FakeModel())


@pytest.mark.parametrize('handle', [
  'os.system', 'subprocess.Popen', 'builtins.eval', 'builtins.exec',
])
def test_model_handle_rejects_non_allowlisted(handle):
  '''model_handle outside allowlist raises ValueError.'''
  with pytest.raises(ValueError):
    predict(_frame(), _params(model_handle=handle))


def test_string_loader_rejects_non_allowlisted():
  '''String model_loader outside allowlist raises ValueError.'''
  with pytest.raises(ValueError):
    predict(_frame(), _params(), model_loader='os.system')


def test_model_handle_accepts_allowlisted_estimator_class():
  '''Allowlisted estimator class resolves via model_handle.'''
  resolved = _resolve_loader(
    _params(model_handle='sklearn.ensemble.RandomForestClassifier'),
    None)
  assert resolved.__name__ == 'RandomForestClassifier'
  assert hasattr(resolved, 'fit') and hasattr(resolved, 'predict')


def test_model_handle_rejects_non_estimator_allowlisted():
  '''Allowlisted non-estimator (function) raises ValueError.'''
  with pytest.raises(ValueError):
    _resolve_loader(
      _params(model_handle='sklearn.datasets.load_iris'), None)


def test_predict_max_rows_null_without_spark_raises():
  '''max_rows=None with spark_scoring disabled raises ValueError.'''
  with pytest.raises(ValueError, match='max_rows'):
    predict(_frame(), _params(max_rows=None,
                               spark_scoring={'enabled': False}),
            model_loader=lambda: _FakeModel())


def test_load_scoring_data_max_rows_null_without_spark_raises():
  '''load_scoring_data rejects max_rows=None when spark is off.'''
  with pytest.raises(ValueError, match='max_rows'):
    load_scoring_data(_frame(), _params(max_rows=None,
                                        spark_scoring={'enabled': False}))


class _FakeSparkCountOnly:
  '''Fake Spark frame reporting >5M rows via count() only.'''

  def __init__(self, rows):
    self._rows = rows
    self.columns = ['id', 'f1', 'f2']
    self.to_pandas_calls = 0

  def count(self):
    '''Return row count without materialising data.'''
    return self._rows

  def toPandas(self):
    '''Record call; the guard must prevent this.'''
    self.to_pandas_calls += 1
    raise AssertionError('toPandas must not be called')


class _FakeRegistryLoader:
  '''Fake RegistryModelLoader returning _FakeModel.'''

  def __init__(self, registry_uri=None, model_name=None,
               alias='production', *args, **kwargs):
    self.registry_uri = registry_uri
    self.model_name = model_name
    self.alias = alias

  def __call__(self):
    '''Return the fake model.'''
    return _FakeModel()


def test_resolve_loader_model_name_alias_uses_registry(monkeypatch):
  '''model_name/alias without loader resolves via RegistryModelLoader.'''
  import forecast_ml_framework.serving.model_loader as loader_mod
  monkeypatch.setattr(
      loader_mod, 'RegistryModelLoader', _FakeRegistryLoader)
  resolved = _resolve_loader(
      _params(model_name='my-model', alias='production'), None)
  assert isinstance(resolved, _FakeModel)


def test_predict_model_name_alias_loads_via_registry(monkeypatch):
  '''predict with model_name/alias loads via registry, no loader.'''
  import forecast_ml_framework.serving.model_loader as loader_mod
  monkeypatch.setattr(
      loader_mod, 'RegistryModelLoader', _FakeRegistryLoader)
  out = predict(
      _frame(), _params(model_name='my-model', alias='production'))
  assert out['prediction'].tolist() == [1, 0, 1]


def test_scoring_fake_spark_over_pandas_limit_raises():
  '''Scoring load enforces pandas guard on fake Spark >5M rows.'''
  spark_df = _FakeSparkCountOnly(5_000_001)
  with pytest.raises(ValueError, match='max_rows_for_pandas'):
    load_scoring_data(spark_df, _params(spark_scoring={'enabled': False}))
  assert spark_df.to_pandas_calls == 0


# --- loop4: uncovered branches in scoring nodes ---


def test_pandas_conversion_and_spark_enabled_variants():
  '''Conversion/spark_enabled helpers cover aliases.'''
  from forecast_ml_framework.pipelines.scoring.nodes import (
    _pandas_conversion_params, _spark_scoring_enabled)
  assert _pandas_conversion_params(
    {'pandas_conversion_params': {'a': 1}}) == {'a': 1}
  assert _pandas_conversion_params(
    {'pandas_conversion': {'b': 2}}) == {'b': 2}
  assert _pandas_conversion_params({}) == {}
  assert _spark_scoring_enabled({'spark_scoring': True}) is True
  assert _spark_scoring_enabled(
    {'spark_scoring_enabled': True}) is True
  assert _spark_scoring_enabled({}) is False


def test_partition_batch_size_variants():
  '''_partition_batch_size covers candidates and errors.'''
  from forecast_ml_framework.pipelines.scoring.nodes import (
    _partition_batch_size)
  assert _partition_batch_size({'partition_batch_size': 5}) == 5
  assert _partition_batch_size(
    {'spark_scoring': {'partition_batch_size': 6}}) == 6
  assert _partition_batch_size({'batch_size': 7}) == 7
  assert _partition_batch_size({}) == 10000
  with pytest.raises(ValueError, match='partition_batch_size'):
    _partition_batch_size({'partition_batch_size': 0})


class _Rdd:
  '''Fake RDD exposing mapPartitions.'''

  def __init__(self, frame):
    self._frame = frame

  def mapPartitions(self, func):
    '''Apply func to the wrapped frame.'''
    return func(self._frame)


class _SparkPartitionable:
  '''Fake Spark frame with mapPartitions.'''

  def __init__(self, frame):
    self._frame = frame
    self.columns = list(frame.columns)

  def count(self):
    '''Return row count.'''
    return len(self._frame)

  def mapPartitions(self, func):
    '''Apply func to the wrapped frame.'''
    return func(self._frame)


def test_partitionable_variants():
  '''_is_spark_partitionable covers mapPartitions/rdd forms.'''
  from forecast_ml_framework.pipelines.scoring.nodes import (
    _is_spark_partitionable)

  class ViaRdd:
    def __init__(self, frame):
      self.rdd = _Rdd(frame)
  assert _is_spark_partitionable(_SparkPartitionable(_frame())) is True
  assert _is_spark_partitionable(ViaRdd(_frame())) is True
  assert _is_spark_partitionable(_frame()) is False
  assert _is_spark_partitionable(object()) is False


def test_cap_value_env_variants(monkeypatch):
  '''_cap_value covers env int/unbounded and params.'''
  from forecast_ml_framework.pipelines.scoring.nodes import _cap_value
  assert _cap_value({'max_rows': 5}, 'max_rows', 'X', 9) == 5
  assert _cap_value({'max_rows': None}, 'max_rows', 'X', 9) is None
  monkeypatch.setenv('SCORING_TEST_CAP', '42')
  assert _cap_value({}, 'max_rows', 'SCORING_TEST_CAP', 9) == 42
  for text in ('', 'null', 'none', ' NULL '):
    monkeypatch.setenv('SCORING_TEST_CAP', text)
    assert _cap_value({}, 'max_rows', 'SCORING_TEST_CAP', 9) is None
  monkeypatch.delenv('SCORING_TEST_CAP', raising=False)
  assert _cap_value({}, 'max_rows', 'SCORING_TEST_CAP', 9) == 9


def test_row_count_variants():
  '''_row_count covers count() and len() forms.'''
  from forecast_ml_framework.pipelines.scoring.nodes import _row_count
  assert _row_count(_frame()) == 3
  assert _row_count(_SparkPartitionable(_frame())) == 3

  class LenOnly:
    def __len__(self):
      return 2
  assert _row_count(LenOnly()) == 2


def test_resolve_handle_strict_errors():
  '''_resolve_handle_strict rejects bad/non-importable handles.'''
  from forecast_ml_framework.pipelines.scoring.nodes import (
    _resolve_handle_strict)
  for bad in (123, 'nodot', 'sklearn.no_such_module.Foo',
              'sklearn.ensemble.NoSuchAttr'):
    with pytest.raises(ValueError):
      _resolve_handle_strict(bad)


def test_registry_loader_model_uri_parsing(monkeypatch):
  '''_registry_loader parses models:/ URIs and aliases.'''
  import forecast_ml_framework.serving.model_loader as loader_mod
  from forecast_ml_framework.pipelines.scoring.nodes import (
    _registry_loader)
  monkeypatch.setattr(
    loader_mod, 'RegistryModelLoader', _FakeRegistryLoader)
  loader = _registry_loader({'model_uri': 'models:/my-model@prod'})
  assert loader.model_name == 'my-model'
  assert loader.alias == 'prod'
  loader2 = _registry_loader({'registered_model_name': 'm2',
                              'model_alias': 'staging',
                              'tracking_uri': 'mlruns'})
  assert loader2.model_name == 'm2' and loader2.alias == 'staging'
  assert loader2.registry_uri == 'mlruns'


def test_registry_loader_signature_fallbacks(monkeypatch):
  '''_registry_loader falls back across loader signatures.'''
  import forecast_ml_framework.serving.model_loader as loader_mod
  from forecast_ml_framework.pipelines.scoring.nodes import (
    _registry_loader)

  class TwoArg:
    def __init__(self, model_name, alias='production'):
      self.args = (model_name, alias)
  monkeypatch.setattr(loader_mod, 'RegistryModelLoader', TwoArg)
  assert _registry_loader({'model_name': 'm'}).args == (
    'm', 'production')

  class ThreePositional:
    def __init__(self, uri, name, alias, /):
      self.args = (uri, name, alias)
  monkeypatch.setattr(
    loader_mod, 'RegistryModelLoader', ThreePositional)
  out = _registry_loader({'model_name': 'm', 'alias': 'a'})
  assert out.args == (None, 'm', 'a')


def test_resolve_loader_path_load_and_errors(monkeypatch):
  '''_resolve_loader covers model_path, load(), and errors.'''
  import forecast_ml_framework.serving.model_loader as loader_mod

  class FakeFileLoader:
    def __init__(self, path):
      self.path = path
    def __call__(self):
      return _FakeModel()
  monkeypatch.setattr(loader_mod, 'FileModelLoader', FakeFileLoader)
  assert isinstance(_resolve_loader({'model_path': '/tmp/m'}, None),
                    _FakeModel)

  class Loadable:
    def load(self):
      return _FakeModel()
  assert isinstance(_resolve_loader({}, Loadable()), _FakeModel)

  class BadLoadable:
    def load(self):
      return object()
  with pytest.raises(TypeError):
    _resolve_loader({}, BadLoadable())
  with pytest.raises(TypeError):
    _resolve_loader({}, lambda: object())
  with pytest.raises(TypeError):
    _resolve_loader({}, 123)
  with pytest.raises(ValueError):
    _resolve_loader({}, None)
  resolved = _resolve_loader(
    {}, 'sklearn.ensemble.RandomForestClassifier')
  assert resolved.__name__ == 'RandomForestClassifier'


def test_feature_cols_requested_filtering():
  '''_feature_cols filters requested cols to present ones.'''
  from forecast_ml_framework.pipelines.scoring.nodes import _feature_cols
  assert _feature_cols(_frame(), {'feature_cols': ['f1', 'ghost']},
                       ['id']) == ['f1']
  assert _feature_cols(_frame(), {}, ['id']) == ['f1', 'f2']


def test_predict_non_numeric_feature_dummies():
  '''Non-numeric features are dummy-encoded before predict.'''
  frame = pd.DataFrame({'id': [1, 2], 'cat': ['a', 'b'],
                        'f1': [1.0, -1.0]})
  out = predict(frame, _params(feature_cols=['cat', 'f1']),
                model_loader=lambda: _FakeModel())
  assert out['prediction'].tolist() == [1, 0]


def test_load_and_predict_spark_partitionable():
  '''Spark scoring load/predict run via mapPartitions.'''
  spark = _SparkPartitionable(_frame())
  params = _params(spark_scoring={'enabled': True})
  assert load_scoring_data(spark, params) is spark
  out = predict(spark, params, model_loader=lambda: _FakeModel())
  assert out['prediction'].tolist() == [1, 0, 1]


def test_predict_spark_via_rdd_and_empty(caplog):
  '''Spark predict works via rdd and warns on empty.'''
  class ViaRdd:
    def __init__(self, frame):
      self._frame = frame
      self.columns = list(frame.columns)
      self.rdd = _Rdd(frame)
    def count(self):
      return len(self._frame)
  params = _params(spark_scoring=True)
  out = predict(ViaRdd(_frame()), params,
                model_loader=lambda: _FakeModel())
  assert out['prediction'].tolist() == [1, 0, 1]
  empty_spark = _SparkPartitionable(pd.DataFrame({
    'id': pd.Series([], dtype=int),
    'f1': pd.Series([], dtype=float),
    'f2': pd.Series([], dtype=float)}))
  with caplog.at_level(logging.WARNING):
    empty_out = predict(empty_spark, params,
                        model_loader=lambda: _FakeModel())
  assert len(empty_out) == 0
  assert load_scoring_data(empty_spark, params) is empty_spark


def test_write_predictions_spark_partitionable_passthrough():
  '''write_predictions passes Spark frames through unchanged.'''
  spark = _SparkPartitionable(_frame())
  params = _params(spark_scoring={'enabled': True})
  assert write_predictions(spark, params) is spark


def test_predict_spark_empty_and_non_df_results():
  '''Spark predict handles empty partitions and non-DF results.'''

  class EmptyPartition:
    def __init__(self):
      self.columns = ['id', 'f1', 'f2']
    def count(self):
      return 3
    def mapPartitions(self, func):
      empty = pd.DataFrame({'id': pd.Series([], dtype=int),
                            'f1': pd.Series([], dtype=float)})
      return func(empty)

  class NonDfResult:
    def __init__(self):
      self.columns = ['id', 'f1', 'f2']
    def count(self):
      return 3
    def mapPartitions(self, func):
      return 'spark-result'
  params = _params(spark_scoring={'enabled': True})
  assert len(predict(EmptyPartition(), params,
                     model_loader=lambda: _FakeModel())) == 0
  assert predict(NonDfResult(), params,
                 model_loader=lambda: _FakeModel()) == 'spark-result'


# --- loop7: get_dummies cardinality bypasses max_cols (RED) ---

def test_predict_dummies_cardinality_exceeds_max_cols():
  '''2000 distinct cats with max_cols=10 raise ValueError.'''
  frame = pd.DataFrame({
    'id': range(2000),
    'cat': [f'c{i}' for i in range(2000)],
  })
  with pytest.raises(ValueError, match='max_cols'):
    predict(frame, _params(feature_cols=['cat'], max_cols=10),
            model_loader=lambda: _FakeModel())


def test_predict_dummies_total_expansion_exceeds_max_cols():
  '''Combined dummy expansion beyond max_cols raises ValueError.'''
  frame = pd.DataFrame({
    'id': range(6),
    'cat_a': [f'a{i}' for i in range(6)],
    'cat_b': [f'b{i}' for i in range(6)],
  })
  with pytest.raises(ValueError, match='max_cols'):
    predict(frame, _params(feature_cols=['cat_a', 'cat_b'],
                           max_cols=10),
            model_loader=lambda: _FakeModel())
