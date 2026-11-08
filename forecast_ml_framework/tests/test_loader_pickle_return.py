#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Pickle envelope via load_model path is unwrapped.'''
# pylint: disable=import-outside-toplevel

import pickle


def _tiny_classification():
  '''Return fitted LogisticRegression and its data.'''
  import numpy as np
  from sklearn.linear_model import LogisticRegression
  x = np.array([[0.0], [1.0], [2.0], [3.0],
                [0.5], [2.5], [1.5], [3.5]])
  y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
  est = LogisticRegression(max_iter=1000).fit(x, y)
  return est, x


def test_pickle_envelope_unwrapped_via_path(tmp_path):
  '''Pickle envelope via path returns estimator, not dict.'''
  from forecast_ml_framework.pipelines.modeling.envelope import (
    wrap_envelope,
  )
  from forecast_ml_framework.serving.model_loader import load_model
  est, x = _tiny_classification()
  path = tmp_path / 'model.pkl'
  path.write_bytes(pickle.dumps(wrap_envelope(est, ['0', '1'])))
  loaded = load_model(str(path))
  assert not isinstance(loaded, dict)
  assert loaded.predict(x).tolist() == est.predict(x).tolist()


def test_pickle_bare_estimator_unchanged(tmp_path):
  '''Pickle bare estimator via path returns it unchanged.'''
  from forecast_ml_framework.serving.model_loader import load_model
  est, x = _tiny_classification()
  path = tmp_path / 'model.pkl'
  path.write_bytes(pickle.dumps(est))
  loaded = load_model(str(path))
  assert not isinstance(loaded, dict)
  assert loaded.predict(x).tolist() == est.predict(x).tolist()


def test_pickle_and_dict_same_shape(tmp_path):
  '''Same artifact via dict and path both return estimator.'''
  from forecast_ml_framework.pipelines.modeling.envelope import (
    wrap_envelope,
  )
  from forecast_ml_framework.serving.model_loader import load_model
  est, x = _tiny_classification()
  env = wrap_envelope(est, ['0', '1'])
  path = tmp_path / 'model.pkl'
  path.write_bytes(pickle.dumps(env))
  from_path = load_model(str(path))
  from_dict = load_model(env)
  assert not isinstance(from_path, dict)
  assert not isinstance(from_dict, dict)
  assert from_path.predict(x).tolist() == from_dict.predict(
    x).tolist()
