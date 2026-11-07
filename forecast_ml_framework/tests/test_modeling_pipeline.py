#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the Kedro modeling pipeline wiring.'''

import numpy as np
import pandas as pd

from forecast_ml_framework.pipelines.modeling.pipeline import (
  create_pipeline,
)


def test_pipeline_nodes_in_order():
  '''Pipeline has split, train, calibrate, evaluate, charts, shap.'''
  pipe = create_pipeline()
  names = [n.name for n in pipe.nodes]
  for name in ('split_data_node', 'train_models_node',
               'calibrate_model_node', 'evaluate_model_node',
               'make_charts_node', 'explain_model_node'):
    assert name in names


def test_pipeline_end_to_end_sklearn():
  '''Nodes run end to end on synthetic classification data.'''
  rng = np.random.RandomState(42)
  frame = pd.DataFrame({'f1': rng.randn(120), 'f2': rng.randn(120),
                        'seg': rng.randint(0, 2, 120),
                        'target': rng.randint(0, 2, 120)})
  params = {
    'task_type': 'classification', 'target_col': 'target',
    'models_to_use': ['sklearn'],
    'selection_metric': 'f1_weighted', 'random_state': 42,
    'test_size': 0.25,
    'sampling_params': {'method': 'stratified'},
    'calibration': {'method': 'sigmoid', 'cv': 2},
    'bucket_segment_col': 'seg',
    'model_params': {'sklearn': {}},
    'shap_params': {'method': 'auto'},
  }
  pipe = create_pipeline()
  nodes = {n.name: n for n in pipe.nodes}
  x_tr, x_te, y_tr, y_te, _ = nodes['split_data_node'].func(
      frame, params)
  model, per_model, _ = nodes['train_models_node'].func(
      x_tr, y_tr, x_te, y_te, params)
  calibrated, _ = nodes['calibrate_model_node'].func(
      model, x_tr, y_tr, params)
  y_pred = calibrated.predict(x_te)
  proba = calibrated.predict_proba(x_te)
  metrics = nodes['evaluate_model_node'].func(
      y_te, y_pred, proba, x_te, params)
  figs = nodes['make_charts_node'].func(
      y_te, y_pred, proba, x_te, params)
  artifact = nodes['explain_model_node'].func(
      calibrated, x_te, y_te, params)
  assert 'f1_weighted' in metrics['global']
  assert len(figs) >= 5
  assert artifact['global_importance']


def test_modeling_shap_explain_module_deleted():
  '''modeling.shap_explain deleted; pipeline uses interpretability.'''
  import importlib.util
  assert importlib.util.find_spec(
      'forecast_ml_framework.pipelines.modeling.shap_explain') is None


def test_modeling_pipeline_explain_artifact_shape():
  '''Explain node preserves legacy shap_artifact shape.'''
  import numpy as np
  import pandas as pd
  rng = np.random.RandomState(42)
  frame = pd.DataFrame({'f1': rng.randn(120), 'f2': rng.randn(120),
                        'seg': rng.randint(0, 2, 120),
                        'target': rng.randint(0, 2, 120)})
  params = {
    'task_type': 'classification', 'target_col': 'target',
    'models_to_use': ['sklearn'],
    'selection_metric': 'f1_weighted', 'random_state': 42,
    'test_size': 0.25,
    'sampling_params': {'method': 'stratified'},
    'calibration': {'method': 'sigmoid', 'cv': 2},
    'bucket_segment_col': 'seg',
    'model_params': {'sklearn': {}},
    'shap_params': {'method': 'auto'},
  }
  nodes = {n.name: n for n in create_pipeline().nodes}
  x_tr, x_te, y_tr, y_te, _ = nodes['split_data_node'].func(
      frame, params)
  model, _pm, _ = nodes['train_models_node'].func(
      x_tr, y_tr, x_te, y_te, params)
  calibrated, _ = nodes['calibrate_model_node'].func(
      model, x_tr, y_tr, params)
  artifact = nodes['explain_model_node'].func(
      calibrated, x_te, y_te, params)
  assert artifact['global_importance']
  assert 'method' in artifact and 'per_row' in artifact
  assert 'shap_available' in artifact


# --- loop4: _predict/_predict_proba/explain_model branches ---


class _StubModel:
  '''Stub model for predict helpers.'''

  def predict(self, x):
    '''Return fixed labels.'''
    return np.asarray([1, 0])

  def predict_proba(self, x):
    '''Return fixed proba.'''
    return np.asarray([[0.2, 0.8], [0.9, 0.1]])


def test_predict_helpers_regression_and_classification():
  '''_predict delegates; _predict_proba is None for regression.'''
  from forecast_ml_framework.pipelines.modeling.pipeline import (
    _predict, _predict_proba)
  model = _StubModel()
  assert list(_predict(model, None)) == [1, 0]
  assert _predict_proba(model, None, 'regression') is None
  assert _predict_proba(model, None, 'classification').shape == (2, 2)
  names = [n.name for n in create_pipeline().nodes]
  assert 'predict_node' in names and 'predict_proba_node' in names


def test_explain_model_skips_bad_rows_and_maps_permutation(
    monkeypatch):
  '''explain_model skips non-dict rows and maps permutation.'''
  import forecast_ml_framework.pipelines.modeling.pipeline as pipe_mod

  def fake_explain(model, x, params):
    '''Return canned interpretability output.'''
    assert params == {
      'max_rows': 7, 'n_repeats': 3, 'random_state': 9}
    return {
      'method': 'permutation_fallback',
      'global_importance': [
        {'feature': 'f1', 'importance': 2}, 'bad', {'x': 1}],
      'per_row': [
        'bad', {'row': 3, 'shap_values': {'f1': 0.5}},
        {'shap_values': {}}],
    }
  monkeypatch.setattr(pipe_mod, 'shap_explain', fake_explain)
  params = {'shap_params': {
    'sample_rows': 7, 'n_repeats': 3, 'random_state': 9}}
  out = pipe_mod.explain_model(object(), object(), object(), params)
  assert out['method'] == 'permutation'
  assert out['shap_available'] is False
  assert out['global_importance'] == {'f1': 2.0}
  assert out['per_row'] == {'3': {'f1': 0.5}, '1': {}}


def test_explain_model_shap_available_and_none_params(monkeypatch):
  '''explain_model reports shap_available for shap method.'''
  import forecast_ml_framework.pipelines.modeling.pipeline as pipe_mod
  monkeypatch.setattr(
    pipe_mod, 'shap_explain',
    lambda m, x, p: {'method': 'shap', 'global_importance': [],
                     'per_row': []})
  out = pipe_mod.explain_model(object(), object(), object(), None)
  assert out == {
    'method': 'shap', 'global_importance': {}, 'per_row': {},
    'shap_available': True}


# --- loop4: task aliases through normalize_task_kind ---

import sys
import types


class _RecordingClassifier:
  '''Fake classifier recording construction.'''

  def __init__(self, **kwargs):
    '''Store constructor kwargs for inspection.'''
    self.kwargs = kwargs

  def fit(self, x, y):
    '''Return self without fitting.'''
    return self


class _RecordingRegressor:
  '''Fake regressor recording construction.'''

  def __init__(self, **kwargs):
    '''Store constructor kwargs for inspection.'''
    self.kwargs = kwargs

  def fit(self, x, y):
    '''Return self without fitting.'''
    return self


def _fake_module(name, **attrs):
  '''Build a fake module with the given attributes.'''
  module = types.ModuleType(name)
  for key, value in attrs.items():
    setattr(module, key, value)
  return module


def test_zoo_adapters_task_alias_uses_classifier(monkeypatch):
  '''cls/BINARY/CASE variants select Classifier, not Regressor.'''
  from forecast_ml_framework.pipelines.modeling.adapters import (
    CatBoostAdapter, LightGBMAdapter, XGBoostAdapter)
  cases = [
    (XGBoostAdapter, 'xgboost',
     {'XGBClassifier': _RecordingClassifier,
      'XGBRegressor': _RecordingRegressor}),
    (LightGBMAdapter, 'lightgbm',
     {'LGBMClassifier': _RecordingClassifier,
      'LGBMRegressor': _RecordingRegressor}),
    (CatBoostAdapter, 'catboost',
     {'CatBoostClassifier': _RecordingClassifier,
      'CatBoostRegressor': _RecordingRegressor}),
  ]
  x = np.zeros((4, 2))
  y = np.array([0, 1, 0, 1])
  for alias in ('cls', 'BINARY', 'Classification', 'CLASS'):
    for adapter_cls, module, attrs in cases:
      monkeypatch.setitem(sys.modules, module,
                          _fake_module(module, **attrs))
      adapter = adapter_cls({'task_type': alias})
      adapter.fit(x, y)
      assert isinstance(adapter.model, _RecordingClassifier), (
        alias, adapter_cls.__name__)


def test_train_models_task_alias_binary_uses_proba():
  '''train_models with BINARY alias computes classification metrics.'''
  from forecast_ml_framework.pipelines.modeling.adapters import (
    train_models)
  rng = np.random.RandomState(0)
  x = rng.randn(60, 2)
  y = (x[:, 0] > 0).astype(int)
  params = {
    'task_type': 'BINARY', 'models_to_use': ['sklearn'],
    'selection_metric': 'f1_weighted', 'random_state': 42,
    'model_params': {'sklearn': {'model_type': 'logistic_regression'}},
  }
  _, per_model, best = train_models(x, y, x, y, params)
  assert best == 'sklearn'
  assert 'f1_weighted' in per_model['sklearn']
  assert 'roc_auc' in per_model['sklearn']


def test_make_charts_task_alias_binary_is_classification():
  '''make_charts with BINARY alias builds classification charts.'''
  from forecast_ml_framework.pipelines.modeling.charts import (
    make_charts)
  n = 60
  y_true = np.tile([0, 1], n // 2)
  y_pred = y_true.copy()
  proba = np.column_stack([1 - y_true * 0.8 - 0.1,
                           y_true * 0.8 + 0.1])
  x_test = np.zeros((n, 2))
  for alias in ('BINARY', 'cls', 'Classification'):
    figs = make_charts(
        y_true, y_pred, proba, x_test, {'task_type': alias})
    assert 'roc' in figs, alias
    assert 'pred_vs_actual' not in figs, alias
