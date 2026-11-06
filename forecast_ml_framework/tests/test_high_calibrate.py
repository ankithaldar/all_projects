#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for prior-shift probability calibration.'''

import numpy as np
import pandas as pd
import pytest

from forecast_ml_framework.pipelines.modeling.calibration import (
  calibrate_probability,
)


def test_identity_when_rates_equal():
  '''Equal real/sample rates leave probabilities unchanged.'''
  proba = np.array([0.1, 0.5, 0.9])
  out = calibrate_probability(proba, 0.3, 0.3)
  assert np.allclose(out, proba)


def test_lower_real_rate_lowers_probability():
  '''Lower real prevalence lowers calibrated probabilities.'''
  proba = np.array([0.5, 0.9])
  out = calibrate_probability(proba, 0.1, 0.5)
  assert np.all(out < proba)
  assert np.all(out >= 0)
  assert np.all(out <= 1)
  # Odds formula: p=0.5 -> 0.1, p=0.9 -> 0.5
  assert out[0] == pytest.approx(0.1)
  assert out[1] == pytest.approx(0.5)


def test_higher_real_rate_raises_probability():
  '''Higher real prevalence raises calibrated probabilities.'''
  proba = np.array([0.5])
  out = calibrate_probability(proba, 0.5, 0.1)
  assert out[0] == pytest.approx(0.9)
  assert out[0] > 0.5


def test_series_in_series_out():
  '''Series input returns a Series with the same index.'''
  proba = pd.Series([0.2, 0.8], index=['a', 'b'], name='p')
  out = calibrate_probability(proba, 0.1, 0.5)
  assert isinstance(out, pd.Series)
  assert list(out.index) == ['a', 'b']
  assert out.name == 'p'
  assert np.all((out >= 0) & (out <= 1))


def test_extremes_are_clipped():
  '''Probabilities stay in [0, 1] at the extremes.'''
  proba = np.array([0.0, 1.0])
  out = calibrate_probability(proba, 0.1, 0.5)
  assert out[0] == pytest.approx(0.0)
  assert out[1] == pytest.approx(1.0)


def test_monotonic_preserved():
  '''Calibration preserves probability ordering.'''
  proba = np.linspace(0.01, 0.99, 20)
  out = calibrate_probability(proba, 0.2, 0.4)
  assert np.all(np.diff(out) > 0)


@pytest.mark.parametrize('real_rate,sample_rate', [
    (np.nan, 0.5), (0.5, np.nan), (np.inf, 0.5), (0.5, -np.inf),
    (-0.1, 0.5), (1.1, 0.5), (0.5, -0.1), (0.5, 1.1),
    ('bad', 0.5), (0.5, 'bad'), (None, 0.5), (0.5, None),
])
def test_calibrate_probability_invalid_rates_raise(real_rate,
                                                   sample_rate):
  '''Non-finite or out-of-range real/sample rates raise ValueError.'''
  with pytest.raises(ValueError, match='rate'):
    calibrate_probability(np.array([0.5]), real_rate, sample_rate)


@pytest.mark.parametrize('bad_proba', [
    np.array([np.nan]), np.array([np.inf]), np.array([-0.1]),
    np.array([1.1]),
])
def test_calibrate_probability_invalid_proba_raises(bad_proba):
  '''NaN or out-of-range proba raises ValueError.'''
  with pytest.raises(ValueError, match='proba'):
    calibrate_probability(bad_proba, 0.1, 0.5)


def _fitted_base(n=60):
  '''Return a fitted base adapter and data.'''
  from forecast_ml_framework.pipelines.modeling.adapters import (
    SklearnAdapter,
  )
  rng = np.random.RandomState(0)
  x = pd.DataFrame({'f1': rng.randn(n), 'f2': rng.randn(n)})
  y = pd.Series(rng.randint(0, 2, n))
  # Ensure both classes present with enough samples for cv=2.
  y.iloc[0] = 0
  y.iloc[1] = 1
  base = SklearnAdapter({}).fit(x, y)
  return base, x, y


def test_calibrate_model_bogus_method_raises():
  '''Unknown calibration method raises ValueError.'''
  from forecast_ml_framework.pipelines.modeling.calibration import (
    calibrate_model,
  )
  base, x, y = _fitted_base()
  params = {'task_type': 'classification',
            'calibration': {'method': 'bogus', 'cv': 2}}
  with pytest.raises(ValueError, match='method'):
    calibrate_model(base, x, y, params)


@pytest.mark.parametrize('bad_cv', [1, 0, 2.0, True, '3', None])
def test_calibrate_model_invalid_cv_type_raises(bad_cv):
  '''Non-int or <2 calibration cv raises ValueError.'''
  from forecast_ml_framework.pipelines.modeling.calibration import (
    calibrate_model,
  )
  base, x, y = _fitted_base()
  params = {'task_type': 'classification',
            'calibration': {'method': 'sigmoid', 'cv': bad_cv}}
  with pytest.raises(ValueError, match='cv'):
    calibrate_model(base, x, y, params)


def test_calibrate_model_cv_above_min_class_raises():
  '''Calibration cv above smallest class count raises ValueError.'''
  from forecast_ml_framework.pipelines.modeling.calibration import (
    calibrate_model,
  )
  base, x, y = _fitted_base(20)
  y = pd.Series([0] * 18 + [1] * 2)
  base = base.__class__({}).fit(x, y)
  params = {'task_type': 'classification',
            'calibration': {'method': 'isotonic', 'cv': 3}}
  with pytest.raises(ValueError, match='cv'):
    calibrate_model(base, x, y, params)


def test_calibrate_probability_multiclass_raises():
  '''Multiclass (n, k>2) proba raises instead of corrupting rows.'''
  proba = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
  with pytest.raises(ValueError, match='proba'):
    calibrate_probability(proba, 0.1, 0.5)


def test_calibrate_probability_3d_raises():
  '''3-D proba raises ValueError.'''
  with pytest.raises(ValueError, match='proba'):
    calibrate_probability(np.zeros((2, 2, 1)), 0.1, 0.5)
