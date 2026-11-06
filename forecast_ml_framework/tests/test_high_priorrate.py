"""Prior-rate correction application at scoring (vz-high Task 8)."""

import numpy as np
import pandas as pd

from forecast_ml_framework.pipelines.scoring.nodes import predict


class _Fake:
  def predict(self, x):
    return np.array([1] * len(x))

  def predict_proba(self, x):
    return np.array([[0.5, 0.5]] * len(x))


def test_prior_rate_adds_calib_proba_column():
  df = pd.DataFrame({'id': [1], 'f1': [0.0]})
  out = predict(df, {'task_type': 'classification', 'real_rate': 0.1,
                     'sample_rate': 0.5, 'id_cols': ['id']},
                model_loader=_Fake())
  assert 'calib_proba' in out.columns
  assert abs(out['calib_proba'].iloc[0] - 0.1) < 1e-9
  assert out['proba'].iloc[0] == 0.5  # unchanged max/positive proba
  assert list(out['prediction']) == [1]  # prediction semantics unchanged


def test_no_prior_rate_params_no_new_column():
  out = predict(pd.DataFrame({'id': [1], 'f1': [0.0]}),
                {'task_type': 'classification', 'id_cols': ['id']},
                model_loader=_Fake())
  assert 'calib_proba' not in out.columns


def test_prior_rate_requires_both_params():
  df = pd.DataFrame({'id': [1], 'f1': [0.0]})
  out = predict(df, {'task_type': 'classification', 'real_rate': 0.1,
                     'id_cols': ['id']}, model_loader=_Fake())
  assert 'calib_proba' not in out.columns
