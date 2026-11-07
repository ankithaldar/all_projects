#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Prior-rate correction application at scoring (vz-high Task 8).'''

import numpy as np
import pandas as pd

from forecast_ml_framework.pipelines.scoring.nodes import predict


class _Fake:
  '''Fake model loader with predict and predict_proba.'''
  def predict(self, x):
    '''Return ones for each row.'''
    return np.array([1] * len(x))

  def predict_proba(self, x):
    '''Return equal class probabilities.'''
    return np.array([[0.5, 0.5]] * len(x))


def test_prior_rate_adds_calib_proba_column():
  '''Prior rate adds calib_proba column.'''
  df = pd.DataFrame({'id': [1], 'f1': [0.0]})
  out = predict(df, {'task_type': 'classification', 'real_rate': 0.1,
                     'sample_rate': 0.5, 'id_cols': ['id']},
                model_loader=_Fake())
  assert 'calib_proba' in out.columns
  assert abs(out['calib_proba'].iloc[0] - 0.1) < 1e-9
  assert out['proba'].iloc[0] == 0.5  # unchanged max/positive proba
  assert list(out['prediction']) == [1]  # prediction semantics unchanged


def test_no_prior_rate_params_no_new_column():
  '''No prior-rate params means no new column.'''
  out = predict(pd.DataFrame({'id': [1], 'f1': [0.0]}),
                {'task_type': 'classification', 'id_cols': ['id']},
                model_loader=_Fake())
  assert 'calib_proba' not in out.columns


def test_prior_rate_requires_both_params():
  '''Prior rate requires both real and sample params.'''
  df = pd.DataFrame({'id': [1], 'f1': [0.0]})
  out = predict(df, {'task_type': 'classification', 'real_rate': 0.1,
                     'id_cols': ['id']}, model_loader=_Fake())
  assert 'calib_proba' not in out.columns
