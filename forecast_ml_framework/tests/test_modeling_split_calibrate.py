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


def test_split_rejects_bad_test_size():
  '''split_data rejects out-of-range test_size values.'''
  import pytest
  for bad in (0, 1, 1.5, -0.2):
    with pytest.raises(ValueError):
      split_data(
          _frame(), {'target_col': 'target', 'test_size': bad})


def test_split_rejects_empty_frame():
  '''split_data rejects an empty frame.'''
  import pytest
  with pytest.raises(ValueError):
    split_data(
        _frame(0), {'target_col': 'target', 'test_size': 0.2})


def test_split_rejects_missing_target():
  '''split_data rejects a missing target column.'''
  import pytest
  with pytest.raises(ValueError):
    split_data(
        _frame(), {'target_col': 'missing', 'test_size': 0.2})


# --- Coverage: betacal missing-library fallback + wrapper ---


def test_calibrate_betacal_missing_library_falls_back_to_sigmoid(
    monkeypatch):
  '''betacal ImportError falls back to sigmoid calibration.'''
  import sys
  monkeypatch.setitem(sys.modules, 'betacal', None)
  from forecast_ml_framework.pipelines.modeling.adapters import (
    SklearnAdapter,
  )
  frame = _frame(80)
  x = frame[['f1', 'f2']]
  y = frame['target']
  base = SklearnAdapter({}).fit(x, y)
  params = {
    'task_type': 'classification',
    'calibration': {'method': 'betacal', 'cv': 2},
  }
  calibrated, meta = calibrate_model(base, x, y, params)
  assert meta['calibrated'] is True
  assert meta['method'] == 'sigmoid'
  assert meta['requested_method'] == 'betacal'
  assert meta['fallback'] is True
  assert calibrated.predict_proba(x).shape[1] == 2


def test_calibrate_betacal_handle_string_falls_back_when_missing(
    monkeypatch):
  '''A BetaCalibration handle also routes through betacal fallback.'''
  import sys
  monkeypatch.setitem(sys.modules, 'betacal', None)
  from forecast_ml_framework.pipelines.modeling.adapters import (
    SklearnAdapter,
  )
  frame = _frame(80)
  x = frame[['f1', 'f2']]
  y = frame['target']
  base = SklearnAdapter({}).fit(x, y)
  params = {
    'task_type': 'classification',
    'calibration_handle': 'betacal.BetaCalibration',
    'calibration': {'method': 'sigmoid', 'cv': 2},
  }
  calibrated, meta = calibrate_model(base, x, y, params)
  assert meta['method'] == 'sigmoid'
  assert meta['calibrated'] is True


def test_beta_cal_wrapper_delegates_to_base_and_calibrator():
  '''_BetaCalWrapper maps proba through base then calibrator.'''
  from forecast_ml_framework.pipelines.modeling.calibration import (
    _BetaCalWrapper,
  )

  class FakeBase:
    '''Base stub returning fixed class probabilities.'''
    classes_ = np.array(['neg', 'pos'])

    def predict_proba(self, x):
      '''Return fixed two-class probabilities.'''
      return np.array([[0.9, 0.1], [0.2, 0.8]])

  class FakeCalibrator:
    '''Calibrator stub scaling probabilities.'''
    def predict_proba(self, proba):
      '''Scale the given probabilities by half.'''
      return proba * 0.5

  wrapped = _BetaCalWrapper(FakeBase(), FakeCalibrator())
  assert wrapped.base is not None
  assert wrapped.calibrator is not None
  proba = wrapped.predict_proba(None)
  assert proba.shape == (2, 2)
  assert list(wrapped.predict(None)) == ['neg', 'pos']


def test_calibrate_task_alias_binary_calibrates():
  '''calibrate_model with BINARY/cls alias calibrates (not passthrough).'''
  from forecast_ml_framework.pipelines.modeling.adapters import (
    SklearnAdapter,
  )
  frame = _frame(120)
  x = frame[['f1', 'f2']]
  y = frame['target']
  for alias in ('BINARY', 'cls', 'Classification'):
    base = SklearnAdapter({}).fit(x, y)
    params = {'task_type': alias,
              'calibration': {'method': 'sigmoid', 'cv': 2}}
    calibrated, meta = calibrate_model(base, x, y, params)
    assert meta['calibrated'] is True, alias
    assert calibrated.predict_proba(x).shape[1] == 2


def test_split_task_alias_stratifies(monkeypatch):
  '''split_data with cls/BINARY alias stratifies by target.'''
  import sklearn.model_selection
  captured = {}
  real = sklearn.model_selection.train_test_split

  def spy(*args, **kwargs):
    '''Record kwargs then delegate to the real split.'''
    captured.update(kwargs)
    return real(*args, **kwargs)
  monkeypatch.setattr(
      sklearn.model_selection, 'train_test_split', spy)
  for alias in ('cls', 'BINARY', 'Classification'):
    captured.clear()
    params = {'target_col': 'target', 'task_type': alias,
              'test_size': 0.25, 'random_state': 42,
              'sampling_params': {'method': 'stratified'}}
    split_data(_frame(), dict(params))
    assert captured.get('stratify') is not None, alias
