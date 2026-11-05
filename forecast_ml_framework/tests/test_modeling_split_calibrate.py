#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for split, sampling and calibration nodes.'''

import numpy as np
import pandas as pd

from forecast_ml_framework.pipelines.modeling.calibration import (
  calibrate_model,
)
from forecast_ml_framework.pipelines.modeling.data_split import (
  split_data,
)


def _frame(n=100):
  '''Return a frame with target and bucket columns.'''
  rng = np.random.RandomState(0)
  return pd.DataFrame({
    'f1': rng.randn(n),
    'f2': rng.randn(n),
    'seg': np.where(rng.rand(n) > 0.5, 'a', 'b'),
    'target': rng.randint(0, 2, n),
  })


def test_split_stratified_classification():
  '''Stratified split preserves class ratio roughly.'''
  params = {'target_col': 'target', 'task_type': 'classification',
            'test_size': 0.25, 'random_state': 42,
            'sampling_params': {'method': 'stratified'}}
  x_tr, x_te, y_tr, y_te, meta = split_data(_frame(), params)
  assert len(x_te) == 25
  assert len(x_tr) == 75
  assert 'target' not in x_tr.columns
  assert 'seg' in x_tr.columns
  assert meta['task_type'] == 'classification'


def test_split_random_sampling():
  '''Random sampling ignores stratification.'''
  params = {'target_col': 'target', 'task_type': 'regression',
            'test_size': 0.2, 'random_state': 1,
            'sampling_params': {'method': 'random'}}
  x_tr, x_te, _, _, _ = split_data(_frame(), params)
  assert len(x_te) == 20
  assert len(x_tr) == 80


def test_calibrate_classification_returns_proba_model():
  '''Calibrated classification model predicts proba.'''
  from forecast_ml_framework.pipelines.modeling.adapters import (
    SklearnAdapter,
  )
  frame = _frame(120)
  x = frame[['f1', 'f2']]
  y = frame['target']
  base = SklearnAdapter({}).fit(x, y)
  params = {'task_type': 'classification',
            'calibration': {'method': 'sigmoid', 'cv': 2}}
  calibrated, meta = calibrate_model(base, x, y, params)
  assert meta['calibrated'] is True
  assert calibrated.predict_proba(x).shape[1] == 2


def test_calibrate_accepts_calibration_handle():
  '''Dotted calibration_handle resolves and calibrates.'''
  from forecast_ml_framework.pipelines.modeling.adapters import (
    SklearnAdapter,
  )
  frame = _frame(120)
  x = frame[['f1', 'f2']]
  y = frame['target']
  base = SklearnAdapter({}).fit(x, y)
  params = {
    'task_type': 'classification',
    'calibration_handle':
      'sklearn.calibration.CalibratedClassifierCV',
    'calibration': {'method': 'sigmoid', 'cv': 2},
  }
  calibrated, meta = calibrate_model(base, x, y, params)
  assert meta['calibrated'] is True
  assert calibrated.predict_proba(x).shape[1] == 2


def test_calibrate_bogus_calibration_handle_raises():
  '''Bogus calibration_handle raises an informative error.'''
  from forecast_ml_framework.pipelines.modeling.adapters import (
    SklearnAdapter,
  )
  import pytest
  frame = _frame(60)
  x = frame[['f1', 'f2']]
  y = frame['target']
  base = SklearnAdapter({}).fit(x, y)
  params = {
    'task_type': 'classification',
    'calibration_handle': 'bogus.module.Nope',
    'calibration': {'method': 'sigmoid', 'cv': 2},
  }
  with pytest.raises(ValueError, match='bogus.module.Nope'):
    calibrate_model(base, x, y, params)


def test_calibrate_regression_passthrough():
  '''Calibration is a passthrough for regression.'''
  from forecast_ml_framework.pipelines.modeling.adapters import (
    SklearnAdapter,
  )
  frame = _frame(60)
  x = frame[['f1', 'f2']]
  y = frame['f1'] + frame['f2']
  base = SklearnAdapter({'model_type': 'linear_regression'}).fit(x, y)
  params = {'task_type': 'regression',
            'calibration': {'method': 'sigmoid', 'cv': 2}}
  calibrated, meta = calibrate_model(base, x, y, params)
  assert meta['calibrated'] is False
  assert calibrated is base
