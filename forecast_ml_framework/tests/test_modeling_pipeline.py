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
