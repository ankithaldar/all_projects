#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''P3-B backtest harness tests (Contract 4).'''

import inspect
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest


def _reg_frame(n=100, start='2024-01-01'):
  '''Return daily regression frame with feature and target.'''
  times = pd.date_range(start=start, periods=n, freq='D', tz='UTC')
  return pd.DataFrame({
    'time': times,
    'feature': np.arange(n, dtype=float),
    'target': np.arange(n, dtype=float),
  })


def _class_frame(n=100, start='2024-01-01'):
  '''Return daily classification frame with feature and target.'''
  times = pd.date_range(start=start, periods=n, freq='D', tz='UTC')
  return pd.DataFrame({
    'time': times,
    'feature': np.arange(n, dtype=float),
    'target': [0, 1] * (n // 2),
  })


class _MeanEstimator:
  '''Estimator predicting the train target mean.'''

  def __init__(self, mean_val):
    '''Store the train mean value.'''
    self._mean = mean_val

  def predict(self, x_frame):
    '''Return the train mean for every test row.'''
    assert list(x_frame.columns) == ['feature']
    return np.full(len(x_frame), self._mean)


class _ZeroEstimator:
  '''Estimator predicting zero for every row.'''

  def predict(self, x_frame):
    '''Return zeros for every test row.'''
    return np.zeros(len(x_frame), dtype=float)


class _NonePredictEstimator:
  '''Estimator whose predict returns None.'''

  def predict(self, unused_x_frame):
    '''Return None to trigger the fail-loud path.'''
    return None


class _LongPredictEstimator:
  '''Estimator predicting one row too many.'''

  def __init__(self, mean_val):
    '''Store the train mean value.'''
    self._mean = mean_val

  def predict(self, x_frame):
    '''Return one extra prediction row.'''
    return np.full(len(x_frame) + 1, self._mean)


class _ScalarPredictEstimator:
  '''Estimator predicting a scalar value.'''

  def __init__(self, mean_val):
    '''Store the train mean value.'''
    self._mean = mean_val

  def predict(self, unused_x_frame):
    '''Return a scalar mean value.'''
    return float(self._mean)


class _TwoDPredictEstimator:
  '''Estimator predicting a 2-D column vector.'''

  def __init__(self, mean_val):
    '''Store the train mean value.'''
    self._mean = mean_val

  def predict(self, x_frame):
    '''Return a 2-D column of predictions.'''
    return np.full((len(x_frame), 1), self._mean)


class _RaisingPredictEstimator:
  '''Estimator whose predict raises.'''

  def predict(self, unused_x_frame):
    '''Raise a marker error like a fitted model would.'''
    raise RuntimeError('boom')


class _NoPredictEstimator:
  '''Estimator without a predict method.'''


def _mean_model_fn(calls):
  '''Return a train-mean model_fn recording train lengths.'''
  def model_fn(train_df):
    '''Fit by recording length and returning the mean.'''
    calls.append(len(train_df))
    return _MeanEstimator(float(train_df['target'].mean()))
  return model_fn


def _regress_params():
  '''Return the pinned regression backtest params.'''
  return {'time_col': 'time', 'n_splits': 2, 'test_size': 0.2,
          'embargo': timedelta(days=7), 'purge': None,
          'target_col': 'target', 'task_type': 'regression'}


def test_exact_scores_regression_train_mean():
  '''Pinned train-mean scores are -43.5, -53.5 and -48.5.'''
  from forecast_ml_framework.pipelines.modeling.backtest import (
    run_backtest,
  )
  df = _reg_frame(100)
  calls = []
  result = run_backtest(
    df, _mean_model_fn(calls), metric_fn=None,
    params=_regress_params())
  assert calls == [53, 73]
  assert result.n_folds == 2
  assert result.task_type == 'regression'
  assert result.per_fold[0]['score'] == -43.5
  assert result.per_fold[1]['score'] == -53.5
  assert result.aggregate == -48.5
  assert result.per_fold[0]['n_train'] == 53
  assert result.per_fold[1]['n_train'] == 73
  assert result.per_fold[0]['n_test'] == 20
  assert result.per_fold[1]['n_test'] == 20
  assert result.per_fold[0]['fold_index'] == 0
  assert result.per_fold[1]['fold_index'] == 1
  for entry in result.per_fold:
    assert entry['fold_meta']['leakage_checked'] is True


def test_one_fold_golden_via_harness():
  '''One fold through the harness pins 73/20/7.'''
  from forecast_ml_framework.pipelines.modeling.backtest import (
    run_backtest,
  )
  df = _reg_frame(100)
  calls = []
  params = {'time_col': 'time', 'n_splits': 1, 'test_size': 0.2,
            'embargo': timedelta(days=7), 'purge': None,
            'target_col': 'target', 'task_type': 'regression'}
  result = run_backtest(
    df, _mean_model_fn(calls), metric_fn=None, params=params)
  assert result.n_folds == 1
  assert calls == [73]
  meta = result.per_fold[0]['fold_meta']
  assert meta['n_train'] == 73
  assert meta['n_test'] == 20
  assert meta['n_embargoed'] == 7
  assert result.per_fold[0]['n_train'] == 73
  assert result.per_fold[0]['n_test'] == 20


def test_classification_backtest_scores():
  '''A zero estimator scores accuracy 0.5 on both folds.'''
  from forecast_ml_framework.pipelines.modeling.backtest import (
    run_backtest,
  )
  df = _class_frame(100)
  params = {'time_col': 'time', 'n_splits': 2, 'test_size': 0.2,
            'embargo': timedelta(days=7), 'purge': None,
            'target_col': 'target', 'task_type': 'classification'}

  def model_fn(unused_train_df):
    '''Return a zero estimator for each fold.'''
    return _ZeroEstimator()
  result = run_backtest(
    df, model_fn, metric_fn=None, params=params)
  assert result.n_folds == 2
  assert result.task_type == 'classification'
  assert result.per_fold[0]['score'] == 0.5
  assert result.per_fold[1]['score'] == 0.5
  assert result.aggregate == 0.5


def test_model_fn_none_second_call_raises():
  '''A None fit on fold 1 raises with no placebo row.'''
  from forecast_ml_framework.pipelines.modeling.backtest import (
    run_backtest,
  )
  df = _reg_frame(100)
  calls = []

  def model_fn(train_df):
    '''Return None on the second call only.'''
    calls.append(len(train_df))
    if len(calls) == 2:
      return None
    return _MeanEstimator(float(train_df['target'].mean()))
  with pytest.raises(ValueError, match='fold 1'):
    run_backtest(df, model_fn, params=_regress_params())
  assert calls == [53, 73]


def test_predict_none_raises():
  '''A None prediction raises ValueError naming the fold.'''
  from forecast_ml_framework.pipelines.modeling.backtest import (
    run_backtest,
  )
  df = _reg_frame(100)

  def model_fn(unused_train_df):
    '''Return an estimator predicting None.'''
    return _NonePredictEstimator()
  with pytest.raises(ValueError, match='fold 0'):
    run_backtest(df, model_fn, params=_regress_params())


def test_predict_wrong_length_raises():
  '''A prediction one row too long raises ValueError.'''
  from forecast_ml_framework.pipelines.modeling.backtest import (
    run_backtest,
  )
  df = _reg_frame(100)

  def model_fn(train_df):
    '''Return a long prediction estimator variant.'''
    return _LongPredictEstimator(float(train_df['target'].mean()))
  with pytest.raises(ValueError, match='fold 0'):
    run_backtest(df, model_fn, params=_regress_params())


def test_predict_scalar_raises():
  '''A scalar prediction raises ValueError.'''
  from forecast_ml_framework.pipelines.modeling.backtest import (
    run_backtest,
  )
  df = _reg_frame(100)

  def model_fn(train_df):
    '''Return an estimator predicting a scalar.'''
    return _ScalarPredictEstimator(float(train_df['target'].mean()))
  with pytest.raises(ValueError, match='fold 0'):
    run_backtest(df, model_fn, params=_regress_params())


def test_predict_two_dimensional_raises():
  '''A 2-D prediction raises ValueError.'''
  from forecast_ml_framework.pipelines.modeling.backtest import (
    run_backtest,
  )
  df = _reg_frame(100)

  def model_fn(train_df):
    '''Return an estimator predicting a column.'''
    return _TwoDPredictEstimator(float(train_df['target'].mean()))
  with pytest.raises(ValueError, match='fold 0'):
    run_backtest(df, model_fn, params=_regress_params())


def test_model_fn_none_raises():
  '''A None model_fn raises before any fold.'''
  from forecast_ml_framework.pipelines.modeling.backtest import (
    run_backtest,
  )
  df = _reg_frame(100)
  with pytest.raises(ValueError, match='model_fn'):
    run_backtest(df, None, params=_regress_params())


def test_model_fn_not_callable_raises():
  '''A non-callable model_fn raises ValueError.'''
  from forecast_ml_framework.pipelines.modeling.backtest import (
    run_backtest,
  )
  df = _reg_frame(100)
  with pytest.raises(ValueError, match='model_fn'):
    run_backtest(df, 42, params=_regress_params())


def test_estimator_without_predict_raises():
  '''An estimator without predict raises ValueError.'''
  from forecast_ml_framework.pipelines.modeling.backtest import (
    run_backtest,
  )
  df = _reg_frame(100)

  def model_fn(unused_train_df):
    '''Return an estimator without predict.'''
    return _NoPredictEstimator()
  with pytest.raises(ValueError, match='fold 0'):
    run_backtest(df, model_fn, params=_regress_params())


def test_metric_fn_raises_propagates():
  '''A raising metric propagates as the original type.'''
  from forecast_ml_framework.pipelines.modeling.backtest import (
    run_backtest,
  )
  df = _reg_frame(100)
  calls = []

  def bad_metric(unused_y_true, unused_y_pred):
    '''Always raise a marker error.'''
    raise RuntimeError('boom')
  with pytest.raises(RuntimeError, match='boom'):
    run_backtest(
      df, _mean_model_fn(calls), metric_fn=bad_metric,
      params=_regress_params())


def test_predict_raises_propagates():
  '''A raising predict propagates with no wrapping.'''
  from forecast_ml_framework.pipelines.modeling.backtest import (
    run_backtest,
  )
  df = _reg_frame(100)

  def model_fn(unused_train_df):
    '''Return an estimator whose predict raises.'''
    return _RaisingPredictEstimator()
  with pytest.raises(RuntimeError, match='boom'):
    run_backtest(df, model_fn, params=_regress_params())


def test_empty_df_raises():
  '''An empty or missing frame raises ValueError.'''
  from forecast_ml_framework.pipelines.modeling.backtest import (
    run_backtest,
  )
  calls = []

  def model_fn(train_df):
    '''Record and return the mean estimator.'''
    calls.append(len(train_df))
    return _MeanEstimator(float(train_df['target'].mean()))
  with pytest.raises(ValueError, match='non-empty'):
    run_backtest(pd.DataFrame({'time': [], 'feature': [],
                               'target': []}),
                 model_fn, params=_regress_params())
  with pytest.raises(ValueError, match='non-empty'):
    run_backtest(None, model_fn, params=_regress_params())


def test_missing_target_raises():
  '''A frame without the target column raises ValueError.'''
  from forecast_ml_framework.pipelines.modeling.backtest import (
    run_backtest,
  )
  df = _reg_frame(100).drop(columns=['target'])
  calls = []
  with pytest.raises(ValueError, match='target_col'):
    run_backtest(df, _mean_model_fn(calls), params=_regress_params())


def test_missing_time_col_raises():
  '''A missing time column raises ValueError.'''
  from forecast_ml_framework.pipelines.modeling.backtest import (
    run_backtest,
  )
  df = _reg_frame(100)
  calls = []
  with pytest.raises(ValueError, match='time_col'):
    run_backtest(df, _mean_model_fn(calls), params={})
  with pytest.raises(ValueError, match='time_col'):
    run_backtest(
      df, _mean_model_fn(calls),
      params={'time_col': 'missing', 'n_splits': 2,
              'target_col': 'target',
              'task_type': 'regression'})


def test_params_none_raises():
  '''None params raise ValueError via the missing time col.'''
  from forecast_ml_framework.pipelines.modeling.backtest import (
    run_backtest,
  )
  df = _reg_frame(100)
  calls = []
  with pytest.raises(ValueError, match='time_col'):
    run_backtest(df, _mean_model_fn(calls), params=None)


def test_default_metric_classification_scores():
  '''The classification default is plain accuracy.'''
  from forecast_ml_framework.pipelines.modeling.backtest import (
    default_metric,
  )
  score_fn = default_metric('classification')
  assert score_fn([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75
  assert score_fn([0, 1], [0, 1]) == 1.0


def test_default_metric_regression_scores():
  '''The regression default is negated MAE.'''
  from forecast_ml_framework.pipelines.modeling.backtest import (
    default_metric,
  )
  score_fn = default_metric('regression')
  assert score_fn([0, 2], [0, 0]) == -1.0
  assert score_fn([1, 3], [1, 3]) == 0.0


def test_default_metric_unknown_raises():
  '''An unknown task type raises ValueError.'''
  from forecast_ml_framework.pipelines.modeling.backtest import (
    default_metric,
  )
  with pytest.raises(ValueError, match='unknown'):
    default_metric('unknown')


def test_determinism_identical_scores():
  '''Two identical runs return identical floats.'''
  from forecast_ml_framework.pipelines.modeling.backtest import (
    run_backtest,
  )
  df = _reg_frame(100)
  first = run_backtest(
    df, _mean_model_fn([]), params=_regress_params())
  second = run_backtest(
    df, _mean_model_fn([]), params=_regress_params())
  assert first.per_fold[0]['score'] == second.per_fold[0]['score']
  assert first.per_fold[1]['score'] == second.per_fold[1]['score']
  assert first.aggregate == second.aggregate


def test_custom_metric_fn_used():
  '''A caller metric overrides the task default.'''
  from forecast_ml_framework.pipelines.modeling.backtest import (
    run_backtest,
  )
  df = _reg_frame(100)
  calls = []

  def flat_metric(unused_y_true, unused_y_pred):
    '''Return a flat marker score.'''
    return 5.0
  result = run_backtest(
    df, _mean_model_fn(calls), metric_fn=flat_metric,
    params=_regress_params())
  assert result.per_fold[0]['score'] == 5.0
  assert result.per_fold[1]['score'] == 5.0
  assert result.aggregate == 5.0
  assert calls == [53, 73]


def test_explicit_feature_cols_used():
  '''Explicit feature cols limit the scored columns.'''
  from forecast_ml_framework.pipelines.modeling.backtest import (
    run_backtest,
  )
  df = _reg_frame(100)
  df['extra'] = np.arange(100, dtype=float)
  calls = []
  params = _regress_params()
  params['feature_cols'] = ['feature']
  result = run_backtest(
    df, _mean_model_fn(calls), params=params)
  assert result.per_fold[0]['score'] == -43.5
  assert result.per_fold[1]['score'] == -53.5
  assert calls == [53, 73]


def test_empty_folds_raises(monkeypatch):
  '''An empty fold list raises ValueError.'''
  import forecast_ml_framework.pipelines.modeling.backtest as mod
  from forecast_ml_framework.pipelines.modeling.backtest import (
    run_backtest,
  )
  df = _reg_frame(100)
  calls = []
  monkeypatch.setattr(
    mod, 'purged_cv_folds', lambda d, p: [])
  with pytest.raises(ValueError, match='folds'):
    run_backtest(df, _mean_model_fn(calls), params=_regress_params())


def test_leaky_fold_raises(monkeypatch):
  '''A leaking fold raises before any scoring.'''
  import forecast_ml_framework.pipelines.modeling.backtest as mod
  from forecast_ml_framework.pipelines.modeling.backtest import (
    run_backtest,
  )
  from forecast_ml_framework.pipelines.modeling.purged_cv import (
    purged_cv_folds,
  )
  df = _reg_frame(100)
  folds = purged_cv_folds(df, _regress_params())
  folds[0]['fold_meta']['max_time_train'] = (
    folds[0]['fold_meta']['min_time_test'])
  calls = []
  monkeypatch.setattr(
    mod, 'purged_cv_folds', lambda d, p: folds)
  with pytest.raises(ValueError, match='fold 0'):
    run_backtest(df, _mean_model_fn(calls), params=_regress_params())
  assert not calls


def test_grep_guard_single_builder():
  '''Backtest reuses folds with no second sort.'''
  import forecast_ml_framework.pipelines.modeling.backtest as mod
  source = inspect.getsource(mod)
  assert 'purged_cv_folds' in source
  assert 'fold_leakage_check' in source
  assert 'temporal_split' not in source
  assert 'sort_values' not in source
  assert '_temporal_fold_assignment' not in source
  assert '_parse_embargo' not in source
  assert 'train_end' not in source
  assert 'Timedelta' not in source
  assert 'return []' not in source
  assert 'return {}' not in source
  assert 'except: pass' not in source
