#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Tests for FIX-ARCH-1 best_model writer envelope.'''

# pylint: disable=import-outside-toplevel, invalid-name

import pickle

import numpy as np


def _clf_data():
  '''Return small classification data.'''
  rng = np.random.RandomState(42)
  x = rng.randn(120, 3)
  y = (x[:, 0] + x[:, 1] > 0).astype(int)
  return x, y


def _reg_data():
  '''Return small regression data.'''
  rng = np.random.RandomState(42)
  x = rng.randn(120, 3)
  y = x[:, 0] * 2.0 + x[:, 1] - x[:, 2]
  return x, y


def _clf_params():
  '''Return classification modeling params.'''
  return {
    'task_type': 'classification',
    'models_to_use': ['sklearn'],
    'selection_metric': 'f1_weighted',
    'random_state': 42,
    'model_params': {
      'sklearn': {'model_type': 'logistic_regression'},
    },
  }


def test_train_models_returns_envelope():
  '''train_models returns envelope dict, not adapter.'''
  from forecast_ml_framework.pipelines.modeling.adapters import (
    ModelAdapter,
    train_models,
  )
  from forecast_ml_framework.pipelines.modeling.envelope import (
    model_classes,
    unwrap_envelope,
    validate_envelope,
  )
  x, y = _clf_data()
  envelope, per_model, best = train_models(
      x, y, x, y, _clf_params())
  assert best == 'sklearn'
  assert 'f1_weighted' in per_model['sklearn']
  assert isinstance(envelope, dict)
  assert set(envelope.keys()) == {
    'model', 'label_set', 'registry_ref', 'onnx_sidecar'}
  assert not isinstance(envelope['model'], ModelAdapter)
  raw = unwrap_envelope(envelope)
  assert raw is envelope['model']
  assert hasattr(raw, 'predict_proba')
  assert hasattr(raw, 'predict')
  classes = model_classes(envelope)
  assert classes is not None
  assert envelope['label_set'] == [str(c) for c in classes]
  assert envelope['registry_ref'] is None
  assert envelope['onnx_sidecar'] is None
  validate_envelope(envelope, 'classification')


def test_train_models_raw_estimator_type():
  '''Envelope model is the raw estimator, not the adapter.'''
  from sklearn.linear_model import LogisticRegression

  from forecast_ml_framework.pipelines.modeling.adapters import (
    train_models,
  )
  x, y = _clf_data()
  envelope, _per, _best = train_models(
      x, y, x, y, _clf_params())
  assert isinstance(envelope['model'], LogisticRegression)


def test_train_models_regression_envelope():
  '''Regression envelope has label_set None.'''
  from forecast_ml_framework.pipelines.modeling.envelope import (
    validate_envelope,
  )
  from forecast_ml_framework.pipelines.modeling.adapters import (
    train_models,
  )
  x, y = _reg_data()
  params = {
    'task_type': 'regression',
    'models_to_use': ['sklearn'],
    'selection_metric': 'rmse',
    'random_state': 42,
    'model_params': {
      'sklearn': {'model_type': 'linear_regression'},
    },
  }
  envelope, per_model, best = train_models(x, y, x, y, params)
  assert best == 'sklearn'
  assert isinstance(envelope, dict)
  assert envelope['label_set'] is None
  validate_envelope(envelope, 'regression')
  assert len(envelope['model'].predict(x)) == len(y)


def test_train_models_registry_onnx_passthrough():
  '''registry_ref and onnx_sidecar pass through.'''
  from forecast_ml_framework.pipelines.modeling.adapters import (
    train_models,
  )
  x, y = _clf_data()
  params = _clf_params()
  params['registry_ref'] = {
    'name': 'my-model', 'alias': 'production', 'version': '1'}
  params['onnx_sidecar'] = {
    'path': 'm.onnx', 'opset': 14, 'sha256': None}
  envelope, _per, _best = train_models(x, y, x, y, params)
  assert envelope['registry_ref'] == params['registry_ref']
  assert envelope['onnx_sidecar'] == params['onnx_sidecar']


def test_pipeline_best_model_node_outputs_envelope():
  '''train_models_node outputs envelope dict.'''
  from forecast_ml_framework.pipelines.modeling.pipeline import (
    create_pipeline,
  )
  from forecast_ml_framework.pipelines.modeling.envelope import (
    validate_envelope,
  )
  x, y = _clf_data()
  params = _clf_params()
  nodes = {n.name: n for n in create_pipeline().nodes}
  envelope, per_model, best = nodes[
    'train_models_node'].func(x, y, x, y, params)
  assert isinstance(envelope, dict)
  assert set(envelope.keys()) == {
    'model', 'label_set', 'registry_ref', 'onnx_sidecar'}
  assert 'sklearn' in per_model
  assert best == 'sklearn'
  validate_envelope(envelope, 'classification')


def test_calibrate_consumes_train_envelope():
  '''calibrate_model consumes train envelope output.'''
  from forecast_ml_framework.pipelines.modeling.adapters import (
    train_models,
  )
  from forecast_ml_framework.pipelines.modeling.calibration import (
    calibrate_model,
  )
  from forecast_ml_framework.pipelines.modeling.envelope import (
    unwrap_envelope,
  )
  x, y = _clf_data()
  envelope, _per, _best = train_models(
      x, y, x, y, _clf_params())
  params = _clf_params()
  params['calibration'] = {'method': 'sigmoid', 'cv': 2}
  cal_env, meta = calibrate_model(envelope, x, y, params)
  assert isinstance(cal_env, dict)
  assert cal_env['label_set'] == envelope['label_set']
  assert meta['calibrated'] is True
  proba = unwrap_envelope(cal_env).predict_proba(x)
  assert proba.shape == (len(x), 2)
  assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_restricted_roundtrip_train_envelope(tmp_path):
  '''Restricted round-trip preserves predict_proba.'''
  from forecast_ml_framework.core.artifact_store import (
    restricted_loads,
  )
  from forecast_ml_framework.core.datasets import (
    RestrictedPickleDataset,
  )
  from forecast_ml_framework.pipelines.modeling.adapters import (
    train_models,
  )
  from forecast_ml_framework.pipelines.modeling.envelope import (
    unwrap_envelope,
  )
  x, y = _clf_data()
  envelope, _per, _best = train_models(
      x, y, x, y, _clf_params())
  before = unwrap_envelope(envelope).predict_proba(x)
  restored = restricted_loads(pickle.dumps(envelope))
  assert isinstance(restored, dict)
  after = unwrap_envelope(restored).predict_proba(x)
  assert np.allclose(before, after)
  path = tmp_path / 'best_model.pkl'
  dataset = RestrictedPickleDataset(filepath=str(path))
  dataset.save(envelope)
  loaded = dataset.load()
  assert isinstance(loaded, dict)
  assert loaded['label_set'] == envelope['label_set']
  assert np.allclose(
      before, unwrap_envelope(loaded).predict_proba(x))


def test_restricted_roundtrip_calibrated_envelope(tmp_path):
  '''Calibrated envelope survives restricted round-trip.'''
  from forecast_ml_framework.core.artifact_store import (
    restricted_loads,
  )
  from forecast_ml_framework.core.datasets import (
    RestrictedPickleDataset,
  )
  from forecast_ml_framework.pipelines.modeling.adapters import (
    train_models,
  )
  from forecast_ml_framework.pipelines.modeling.calibration import (
    calibrate_model,
  )
  from forecast_ml_framework.pipelines.modeling.envelope import (
    unwrap_envelope,
  )
  x, y = _clf_data()
  envelope, _per, _best = train_models(
      x, y, x, y, _clf_params())
  params = _clf_params()
  params['calibration'] = {'method': 'sigmoid', 'cv': 2}
  cal_env, _meta = calibrate_model(envelope, x, y, params)
  before = unwrap_envelope(cal_env).predict_proba(x)
  restored = restricted_loads(pickle.dumps(cal_env))
  assert isinstance(restored, dict)
  assert np.allclose(
      before, unwrap_envelope(restored).predict_proba(x))
  path = tmp_path / 'calibrated.pkl'
  dataset = RestrictedPickleDataset(filepath=str(path))
  dataset.save(cal_env)
  loaded = dataset.load()
  assert isinstance(loaded, dict)
  assert np.allclose(
      before, unwrap_envelope(loaded).predict_proba(x))
