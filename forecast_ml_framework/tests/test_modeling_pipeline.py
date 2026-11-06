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
