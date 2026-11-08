#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Envelope enforcement: wrap, train and load validate.'''
# pylint: disable=import-outside-toplevel, invalid-name

import pickle
from unittest import mock

import numpy as np
import pytest


def _tiny_classification():
  '''Return a fitted classifier and its data.'''
  from sklearn.linear_model import LogisticRegression
  x = np.array([[0.0], [1.0], [2.0], [3.0], [0.5], [2.5],
                [1.5], [3.5]])
  y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
  est = LogisticRegression(max_iter=1000).fit(x, y)
  return est, x, y


def _tiny_regression():
  '''Return a fitted regressor and its data.'''
  from sklearn.linear_model import LinearRegression
  x = np.array([[0.0], [1.0], [2.0], [3.0]])
  y = np.array([0.0, 1.0, 2.0, 3.0])
  est = LinearRegression().fit(x, y)
  return est, x, y


def _manual_envelope(model, label_set):
  '''Return a raw envelope dict without using wrap.'''
  return {
    'model': model,
    'label_set': label_set,
    'registry_ref': None,
    'onnx_sidecar': None,
  }


def test_wrap_envelope_accepts_valid():
  '''Wrap accepts valid classification and regression.'''
  from forecast_ml_framework.pipelines.modeling.envelope import (
    wrap_envelope,
  )
  est, _, _ = _tiny_classification()
  env = wrap_envelope(est, ['0', '1'])
  assert env['model'] is est
  assert env['label_set'] == ['0', '1']
  rest, _, _ = _tiny_regression()
  env_r = wrap_envelope(rest, None)
  assert env_r['label_set'] is None


def test_wrap_envelope_rejects_label_set_mismatch():
  '''Wrap rejects label_set not matching classes.'''
  from forecast_ml_framework.pipelines.modeling.envelope import (
    wrap_envelope,
  )
  est, _, _ = _tiny_classification()
  with pytest.raises(ValueError):
    wrap_envelope(est, ['1', '0'])


def test_wrap_envelope_rejects_none_label_set_for_classifier():
  '''Wrap rejects a classifier envelope with None label.'''
  from forecast_ml_framework.pipelines.modeling.envelope import (
    wrap_envelope,
  )
  est, _, _ = _tiny_classification()
  with pytest.raises(ValueError):
    wrap_envelope(est, None)


def test_wrap_envelope_rejects_list_for_regressor():
  '''Wrap rejects a regressor envelope with a label list.'''
  from forecast_ml_framework.pipelines.modeling.envelope import (
    wrap_envelope,
  )
  rest, _, _ = _tiny_regression()
  with pytest.raises(ValueError):
    wrap_envelope(rest, ['0', '1'])


def test_wrap_envelope_rejects_none_model():
  '''Wrap rejects a None model.'''
  from forecast_ml_framework.pipelines.modeling.envelope import (
    wrap_envelope,
  )
  with pytest.raises(ValueError):
    wrap_envelope(None, None)


def test_wrap_envelope_rejects_bad_onnx_sidecar():
  '''Wrap rejects a malformed onnx_sidecar.'''
  from forecast_ml_framework.pipelines.modeling.envelope import (
    wrap_envelope,
  )
  est, _, _ = _tiny_classification()
  with pytest.raises(ValueError):
    wrap_envelope(
      est, ['0', '1'],
      onnx_sidecar={'path': 'm.txt', 'opset': 14,
                    'sha256': None})


def test_wrap_envelope_rejects_bad_registry_ref():
  '''Wrap rejects a malformed registry_ref.'''
  from forecast_ml_framework.pipelines.modeling.envelope import (
    wrap_envelope,
  )
  est, _, _ = _tiny_classification()
  with pytest.raises(ValueError):
    wrap_envelope(
      est, ['0', '1'], registry_ref={'name': 'm'})


def test_train_models_classification_without_classes_raises():
  '''Classification train raises when classes are missing.'''
  from forecast_ml_framework.pipelines.modeling.adapters import (
    train_models,
  )
  _, x, y = _tiny_classification()
  params = {
    'task_type': 'classification',
    'models_to_use': ['sklearn'],
    'selection_metric': 'f1_weighted',
    'random_state': 42,
    'model_params': {
      'sklearn': {'model_type': 'logistic_regression'},
    },
  }
  target = (
    'forecast_ml_framework.pipelines.modeling.'
    'adapters.model_classes')
  with mock.patch(target, return_value=None), pytest.raises(
      ValueError):
    train_models(x, y, x, y, params)


def test_load_model_accepts_valid_envelope():
  '''load_model returns the model from a valid envelope.'''
  from forecast_ml_framework.pipelines.modeling.envelope import (
    wrap_envelope,
  )
  from forecast_ml_framework.serving.model_loader import (
    load_best_model,
    load_model,
  )
  est, _, _ = _tiny_classification()
  env = wrap_envelope(est, ['0', '1'])
  assert load_model(env) is est
  assert load_best_model(env) is est
  rest, _, _ = _tiny_regression()
  env_r = wrap_envelope(rest, None)
  assert load_model(env_r) is rest
  assert load_best_model(env_r) is rest


def test_load_model_rejects_label_set_mismatch():
  '''load_model rejects an envelope with a bad label_set.'''
  from forecast_ml_framework.serving.model_loader import (
    ModelCorruptError,
    load_best_model,
    load_model,
  )
  est, _, _ = _tiny_classification()
  bad = _manual_envelope(est, ['1', '0'])
  with pytest.raises(ModelCorruptError):
    load_model(bad)
  with pytest.raises(ModelCorruptError):
    load_best_model(bad)


def test_load_model_rejects_missing_keys():
  '''load_model rejects a dict with only a model key.'''
  from forecast_ml_framework.serving.model_loader import (
    ModelCorruptError,
    load_model,
  )
  est, _, _ = _tiny_classification()
  with pytest.raises(ModelCorruptError):
    load_model({'model': est})


def test_load_model_rejects_extra_keys():
  '''load_model rejects an envelope with an extra key.'''
  from forecast_ml_framework.pipelines.modeling.envelope import (
    wrap_envelope,
  )
  from forecast_ml_framework.serving.model_loader import (
    ModelCorruptError,
    load_model,
  )
  est, _, _ = _tiny_classification()
  bad = wrap_envelope(est, ['0', '1'])
  bad['extra'] = 1
  with pytest.raises(ModelCorruptError):
    load_model(bad)


def test_load_model_rejects_none_model_envelope():
  '''load_model rejects an envelope whose model is None.'''
  from forecast_ml_framework.serving.model_loader import (
    ModelCorruptError,
    load_model,
  )
  bad = _manual_envelope(None, None)
  with pytest.raises(ModelCorruptError):
    load_model(bad)


def test_load_model_rejects_classifier_none_label_set():
  '''load_model rejects a classifier envelope with None label.'''
  from forecast_ml_framework.serving.model_loader import (
    ModelCorruptError,
    load_model,
  )
  est, _, _ = _tiny_classification()
  bad = _manual_envelope(est, None)
  with pytest.raises(ModelCorruptError):
    load_model(bad)


def test_load_model_rejects_invalid_pickle_envelope(tmp_path):
  '''load_model rejects an invalid envelope stored in a file.'''
  from forecast_ml_framework.serving.model_loader import (
    ModelCorruptError,
    load_model,
  )
  est, _, _ = _tiny_classification()
  path = tmp_path / 'bad.pkl'
  path.write_bytes(pickle.dumps(_manual_envelope(est, ['1', '0'])))
  with pytest.raises(ModelCorruptError):
    load_model(str(path))
