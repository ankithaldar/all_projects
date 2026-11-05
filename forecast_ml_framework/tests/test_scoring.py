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
  '''Predict adds prediction, proba, model id, and timestamp.'''
  out = predict(_frame(), _params(), model_loader=lambda: _FakeModel())
  assert list(out.columns) == [
    'id', 'prediction', 'proba', 'score_model_id', 'scored_at']
  assert out['prediction'].tolist() == [1, 0, 1]
  assert out['proba'].tolist() == [0.9, 0.9, 0.9]
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
    'id', 'prediction', 'proba', 'score_model_id', 'scored_at']
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


def test_pipeline_has_three_named_nodes():
  '''Pipeline wires score_data -> predict -> write_predictions.'''
  pipe = create_pipeline()
  names = [n.name for n in pipe.nodes]
  assert names == ['score_data', 'predict', 'write_predictions']
  assert len(pipe.nodes) == 3
  nodes = {n.name: n for n in pipe.nodes}
  assert 'scoring_input' in nodes['score_data'].inputs
  assert 'params:scoring_params' in nodes['score_data'].inputs
  assert 'params:scoring_params' in nodes['predict'].inputs
  assert 'params:scoring_params' in nodes['write_predictions'].inputs


def test_pipeline_end_to_end_via_node_funcs():
  '''Node funcs chain load -> predict -> write without catalog.'''
  nodes = {n.name: n for n in create_pipeline().nodes}
  data = nodes['score_data'].func(_frame(), _params())
  preds = nodes['predict'].func(
      data, _params(), model_loader=lambda: _FakeModel())
  out = nodes['write_predictions'].func(preds, _params())
  assert out['prediction'].tolist() == [1, 0, 1]
  assert out['score_model_id'].unique().tolist() == ['model-v1']


# --- Phase 2 fix: registry, I/O wiring, caps, finite, allowlist ---


def test_registry_includes_scoring_seven_keys():
  '''Registry returns 7 keys including scoring.'''
  from forecast_ml_framework.pipeline_registry import register_pipelines
  pipelines = register_pipelines()
  assert set(pipelines) == {
    'data', 'preprocessing', 'feature_selection',
    'modeling', 'evaluation', 'predict', 'scoring'}
  assert len(pipelines) == 7


def test_pipeline_uses_params_io():
  '''Pipeline I/O comes from scoring_params when given.'''
  pipe = create_pipeline(scoring_params={
    'input_dataset': 'my_in', 'output_dataset': 'my_out'})
  nodes = {n.name: n for n in pipe.nodes}
  assert 'my_in' in nodes['score_data'].inputs
  assert nodes['write_predictions'].outputs == ['my_out']


def test_pipeline_defaults_backcompat_io():
  '''Without params, pipeline defaults to scoring_input/predictions.'''
  nodes = {n.name: n for n in create_pipeline().nodes}
  assert 'scoring_input' in nodes['score_data'].inputs
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
