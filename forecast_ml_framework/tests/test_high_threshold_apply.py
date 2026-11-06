"""Threshold application at scoring (vz-high Task 7)."""

import numpy as np
import pandas as pd

from forecast_ml_framework.pipelines.scoring.nodes import predict


class _Fake:
  def predict(self, x):
    return np.array([1] * len(x))

  def predict_proba(self, x):
    return np.array([[0.6, 0.4]] * len(x))


def test_threshold_changes_labels():
  df = pd.DataFrame({'id': [1, 2], 'f1': [0.0, 1.0]})
  out = predict(
      df,
      {'task_type': 'classification', 'model_threshold': 0.35,
       'id_cols': ['id']},
      model_loader=_Fake())
  assert list(out['prediction']) == [1, 1]  # 0.4 >= 0.35
  assert list(out['proba']) == [0.4, 0.4]  # proba column unchanged
  out2 = predict(
      df,
      {'task_type': 'classification', 'model_threshold': 0.45,
       'id_cols': ['id']},
      model_loader=_Fake())
  assert list(out2['prediction']) == [0, 0]  # 0.4 < 0.45
  out3 = predict(df, {'task_type': 'classification', 'id_cols': ['id']},
                 model_loader=_Fake())
  assert list(out3['prediction']) == [1, 1]  # verbatim predict


def test_threshold_ignored_for_regression():
  class Reg:
    def predict(self, x):
      return np.array([2.5] * len(x))

  df = pd.DataFrame({'id': [1], 'f1': [0.0]})
  out = predict(df, {'task_type': 'regression', 'model_threshold': 0.5,
                     'id_cols': ['id']}, model_loader=Reg())
  assert list(out['prediction']) == [2.5]


def test_threshold_multiclass_falls_back_to_predict():
  import warnings

  class Multi:
    def predict(self, x):
      return np.array([2] * len(x))

    def predict_proba(self, x):
      return np.array([[0.1, 0.2, 0.7]] * len(x))

  df = pd.DataFrame({'id': [1], 'f1': [0.0]})
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    out = predict(df, {'task_type': 'classification',
                       'model_threshold': 0.5, 'id_cols': ['id']},
                  model_loader=Multi())
  assert list(out['prediction']) == [2]
  assert caught


def test_threshold_single_column_proba_falls_back():
  import warnings

  class OneCol:
    def predict(self, x):
      return np.array([1] * len(x))

    def predict_proba(self, x):
      return np.array([[0.9]] * len(x))

  df = pd.DataFrame({'id': [1], 'f1': [0.0]})
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    out = predict(df, {'task_type': 'classification',
                       'model_threshold': 0.5, 'id_cols': ['id']},
                  model_loader=OneCol())
  assert list(out['prediction']) == [1]
  assert caught
