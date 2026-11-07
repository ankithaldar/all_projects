#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for hyperparameter tuning (high-value Tasks 3-6).'''

import numpy as np
import pandas as pd
import pytest

from forecast_ml_framework.pipelines.modeling.tuning import tune_model


def _data():
  '''Return small classification data.'''
  rng = np.random.RandomState(42)
  x = rng.randn(120, 3)
  y = (x[:, 0] + x[:, 1] > 0).astype(int)
  return pd.DataFrame(x, columns=['f1', 'f2', 'f3']), y


@pytest.mark.serial
def test_grid_tuning_returns_best_and_report():
  '''Grid search returns fitted best, params and report.'''
  from sklearn.linear_model import LogisticRegression
  x, y = _data()
  params = {
    'method': 'grid',
    'param_grid': {'C': [0.1, 1.0]},
    'cv': 3,
    'scoring': 'accuracy',
  }
  best, best_params, report = tune_model(
    LogisticRegression(max_iter=1000), x, y, params)
  assert set(best_params) == {'C'}
  assert best_params['C'] in (0.1, 1.0)
  assert best.get_params()['C'] == best_params['C']
  assert hasattr(best, 'predict')
  assert report['method'] == 'grid'
  assert report['best_params'] == best_params
  assert report['best_score'] is not None
  assert report['cv'] == 3
  assert report['scoring'] == 'accuracy'
  assert report['n_jobs'] == 1


@pytest.mark.serial
def test_random_tuning_returns_best():
  '''Randomized search returns a fitted best estimator.'''
  from sklearn.linear_model import LogisticRegression
  x, y = _data()
  params = {
    'method': 'random',
    'param_grid': {'C': [0.1, 1.0, 10.0]},
    'cv': 3,
    'scoring': 'accuracy',
    'n_iter': 2,
    'random_state': 42,
  }
  best, best_params, report = tune_model(
    LogisticRegression(max_iter=1000), x, y, params)
  assert best_params['C'] in (0.1, 1.0, 10.0)
  assert report['method'] == 'random'
  assert hasattr(best, 'predict')


@pytest.mark.serial
def test_halving_tuning_returns_best():
  '''Successive halving returns a fitted best estimator.'''
  from sklearn.linear_model import LogisticRegression
  x, y = _data()
  params = {
    'method': 'halving',
    'param_grid': {'C': [0.1, 1.0, 10.0]},
    'cv': 3,
    'scoring': 'accuracy',
    'random_state': 42,
  }
  best, best_params, report = tune_model(
    LogisticRegression(max_iter=1000), x, y, params)
  assert best_params['C'] in (0.1, 1.0, 10.0)
  assert report['method'] == 'halving'
  assert hasattr(best, 'predict')


def test_unknown_method_raises():
  '''Unknown tuning method raises ValueError (exact map).'''
  from sklearn.linear_model import LogisticRegression
  x, y = _data()
  params = {
    'method': 'bogus',
    'param_grid': {'C': [1.0]},
    'cv': 3,
  }
  with pytest.raises(ValueError, match='method'):
    tune_model(LogisticRegression(max_iter=1000), x, y, params)


def test_empty_grid_list_raises():
  '''Empty param_grid list raises ValueError.'''
  from sklearn.linear_model import LogisticRegression
  x, y = _data()
  params = {
    'method': 'grid',
    'param_grid': {'C': []},
    'cv': 3,
  }
  with pytest.raises(ValueError, match='param_grid'):
    tune_model(LogisticRegression(max_iter=1000), x, y, params)


def test_non_list_grid_raises():
  '''Non-list param_grid value raises ValueError.'''
  from sklearn.linear_model import LogisticRegression
  x, y = _data()
  params = {
    'method': 'grid',
    'param_grid': {'C': 1.0},
    'cv': 3,
  }
  with pytest.raises(ValueError, match='param_grid'):
    tune_model(LogisticRegression(max_iter=1000), x, y, params)


def test_empty_grid_raises():
  '''Empty param_grid dict raises ValueError.'''
  from sklearn.linear_model import LogisticRegression
  x, y = _data()
  params = {'method': 'grid', 'param_grid': {}, 'cv': 3}
  with pytest.raises(ValueError, match='param_grid'):
    tune_model(LogisticRegression(max_iter=1000), x, y, params)


def test_cv_below_two_raises():
  '''cv < 2 raises ValueError.'''
  from sklearn.linear_model import LogisticRegression
  x, y = _data()
  params = {
    'method': 'grid',
    'param_grid': {'C': [1.0]},
    'cv': 1,
  }
  with pytest.raises(ValueError, match='cv'):
    tune_model(LogisticRegression(max_iter=1000), x, y, params)


def test_cv_above_min_class_raises():
  '''cv > smallest class count raises ValueError.'''
  from sklearn.linear_model import LogisticRegression
  x, y = _data()
  params = {
    'method': 'grid',
    'param_grid': {'C': [1.0]},
    'cv': 1000,
  }
  with pytest.raises(ValueError, match='cv'):
    tune_model(LogisticRegression(max_iter=1000), x, y, params)


def test_non_string_scoring_raises():
  '''Non-string scoring raises ValueError.'''
  from sklearn.linear_model import LogisticRegression
  x, y = _data()
  params = {
    'method': 'grid',
    'param_grid': {'C': [1.0]},
    'cv': 3,
    'scoring': 123,
  }
  with pytest.raises(ValueError, match='scoring'):
    tune_model(LogisticRegression(max_iter=1000), x, y, params)


def test_n_jobs_override_and_default():
  '''n_jobs defaults to 1 and can be overridden.'''
  from sklearn.linear_model import LogisticRegression
  x, y = _data()
  params = {
    'method': 'grid',
    'param_grid': {'C': [1.0]},
    'cv': 2,
    'scoring': 'accuracy',
    'n_jobs': 2,
  }
  _, _, report = tune_model(
    LogisticRegression(max_iter=1000), x, y, params)
  assert report['n_jobs'] == 2


def test_grid_product_over_1000_raises():
  '''Grid with >1000 total candidates raises ValueError pre-fit.'''
  from sklearn.linear_model import LogisticRegression
  x, y = _data()
  params = {
    'method': 'grid',
    'param_grid': {'C': list(np.linspace(0.1, 4.0, 40)),
                   'max_iter': list(range(40))},
    'cv': 3,
  }
  with pytest.raises(ValueError, match='1000'):
    tune_model(LogisticRegression(max_iter=1000), x, y, params)


def test_n_iter_over_1000_raises():
  '''n_iter > 1000 raises ValueError.'''
  from sklearn.linear_model import LogisticRegression
  x, y = _data()
  params = {
    'method': 'random',
    'param_grid': {'C': [0.1, 1.0]},
    'cv': 3,
    'n_iter': 1001,
  }
  with pytest.raises(ValueError, match='n_iter'):
    tune_model(LogisticRegression(max_iter=1000), x, y, params)


def test_n_iter_non_positive_raises():
  '''n_iter < 1 raises ValueError.'''
  from sklearn.linear_model import LogisticRegression
  x, y = _data()
  params = {
    'method': 'random',
    'param_grid': {'C': [0.1, 1.0]},
    'cv': 3,
    'n_iter': 0,
  }
  with pytest.raises(ValueError, match='n_iter'):
    tune_model(LogisticRegression(max_iter=1000), x, y, params)


@pytest.mark.parametrize('bad_n_jobs', [0, -1, 1.5, True])
def test_invalid_n_jobs_raises(bad_n_jobs):
  '''n_jobs outside [1, cpu] or non-int raises ValueError.'''
  from sklearn.linear_model import LogisticRegression
  x, y = _data()
  params = {
    'method': 'grid',
    'param_grid': {'C': [1.0]},
    'cv': 2,
    'n_jobs': bad_n_jobs,
  }
  with pytest.raises(ValueError, match='n_jobs'):
    tune_model(LogisticRegression(max_iter=1000), x, y, params)


def test_n_jobs_above_cpu_raises():
  '''n_jobs greater than cpu_count raises ValueError.'''
  import os
  from sklearn.linear_model import LogisticRegression
  x, y = _data()
  params = {
    'method': 'grid',
    'param_grid': {'C': [1.0]},
    'cv': 2,
    'n_jobs': (os.cpu_count() or 1) + 1,
  }
  with pytest.raises(ValueError, match='n_jobs'):
    tune_model(LogisticRegression(max_iter=1000), x, y, params)


def test_regression_cv_validated_against_n_samples():
  '''Regression cv may exceed min unique count but not n_samples.'''
  from sklearn.linear_model import Ridge
  rng = np.random.RandomState(0)
  x = rng.randn(24, 2)
  y = x[:, 0] * 2.0 + rng.randn(24) * 0.01
  params = {
    'method': 'grid',
    'param_grid': {'alpha': [0.1, 1.0]},
    'cv': 6,
    'scoring': 'r2',
  }
  best, best_params, report = tune_model(Ridge(), x, y, params)
  assert best_params['alpha'] in (0.1, 1.0)
  assert report['cv'] == 6
  assert hasattr(best, 'predict')


def test_regression_cv_above_n_samples_raises():
  '''Regression cv > n_samples raises ValueError.'''
  from sklearn.linear_model import Ridge
  rng = np.random.RandomState(0)
  x = rng.randn(12, 2)
  y = rng.randn(12)
  params = {
    'method': 'grid',
    'param_grid': {'alpha': [1.0]},
    'cv': 20,
  }
  with pytest.raises(ValueError, match='cv'):
    tune_model(Ridge(), x, y, params)


def test_single_class_y_raises_pre_fit():
  '''Single-class classification target raises ValueError pre-fit.'''
  from sklearn.linear_model import LogisticRegression
  rng = np.random.RandomState(0)
  x = rng.randn(30, 2)
  y = np.zeros(30, dtype=int)
  params = {
    'method': 'grid',
    'param_grid': {'C': [1.0]},
    'cv': 2,
  }
  with pytest.raises(ValueError, match='class'):
    tune_model(LogisticRegression(max_iter=1000), x, y, params)


@pytest.mark.serial
def test_halving_random_tuning_returns_best():
  '''Halving random search maps n_iter to n_candidates.'''
  from sklearn.linear_model import LogisticRegression
  x, y = _data()
  params = {
    'method': 'halving_random',
    'param_grid': {'C': [0.1, 1.0, 10.0]},
    'cv': 3,
    'scoring': 'accuracy',
    'n_iter': 2,
    'random_state': 42,
  }
  best, best_params, report = tune_model(
    LogisticRegression(max_iter=1000), x, y, params)
  assert best_params['C'] in (0.1, 1.0, 10.0)
  assert report['method'] == 'halving_random'
  assert hasattr(best, 'predict')


def test_fit_zoo_helper_fits_estimator_and_records_imbalance():
  '''_fit_zoo builds kwargs, fits cls and stores imbalance.'''
  from forecast_ml_framework.pipelines.modeling.adapters import (
    XGBoostAdapter,
    _fit_zoo,
  )

  class _Dummy:
    '''Dummy estimator recording fit state.'''
    def __init__(self, **kwargs):
      '''Store kwargs and mark unfitted.'''
      self.kwargs = kwargs
      self.fitted = False

    def fit(self, x, y):
      '''Mark fitted and return self.'''
      self.fitted = True
      return self

  adapter = XGBoostAdapter({'task_type': 'classification'})
  result = _fit_zoo(adapter, [[0.0], [1.0]], [0, 1], _Dummy)
  assert result is adapter
  assert isinstance(adapter.model, _Dummy)
  assert adapter.model.fitted is True
  assert adapter.imbalance_ is None


def test_fit_zoo_helper_rejects_tuning_config():
  '''_fit_zoo fails loud on a tuning config (zoo adapters).'''
  from forecast_ml_framework.pipelines.modeling.adapters import (
    XGBoostAdapter,
    _fit_zoo,
  )

  class _Dummy:
    '''Dummy estimator for tuning rejection.'''
    def __init__(self, **kwargs):
      '''Accept and ignore kwargs.'''
      pass

  adapter = XGBoostAdapter(
    {'tuning': {'method': 'grid', 'param_grid': {'C': [1.0]}}})
  with pytest.raises(ValueError, match='tuning is not supported'):
    _fit_zoo(adapter, [[0.0]], [0], _Dummy)


def test_hooks_note_run_id_records_first_run_id_only():
  '''_note_run_id sets run_id once and ignores empty values.'''
  from forecast_ml_framework.hooks import ForecastHooks

  hooks = ForecastHooks()
  hooks._note_run_id(None)
  assert hooks.run_id is None
  hooks._note_run_id('run-1')
  assert hooks.run_id == 'run-1'
  hooks._note_run_id('run-2')
  assert hooks.run_id == 'run-1'


def test_imbalance_none_and_nat_still_rejected():
  '''compute_imbalance rejects None/NaT (dead-branch removal).'''
  import pandas as pd
  from forecast_ml_framework.pipelines.modeling.imbalance import (
    compute_imbalance)
  for bad in ([0, None], [0, pd.NaT], [0, float('nan')]):
    with pytest.raises(ValueError):
      compute_imbalance(bad)


@pytest.mark.parametrize('bad_y', [
  np.zeros((30, 1), dtype=int),
  np.zeros((30, 2), dtype=int),
  [[0], [1]] * 15,
  [[0, 1], [1, 0]] * 15,
])
def test_non_1d_y_raises(bad_y):
  '''2-D / list-of-lists y raises ValueError pre-fit.'''
  from sklearn.linear_model import LogisticRegression
  rng = np.random.RandomState(0)
  x = rng.randn(30, 2)
  params = {
    'method': 'grid',
    'param_grid': {'C': [1.0]},
    'cv': 2,
  }
  with pytest.raises(ValueError, match='1-D'):
    tune_model(LogisticRegression(max_iter=1000), x, bad_y, params)


def test_fold_balanced_estimator_tags_delegate_to_wrapped():
  '''Wrapper is a classifier (regressor) iff the wrapped is.'''
  from sklearn.base import is_classifier, is_regressor
  from sklearn.linear_model import LogisticRegression, Ridge
  from forecast_ml_framework.pipelines.modeling.tuning import (
    _FoldBalancedEstimator,
  )
  clf_wrapper = _FoldBalancedEstimator(
      LogisticRegression(max_iter=1000), balance=True)
  reg_wrapper = _FoldBalancedEstimator(Ridge(), balance=True)
  assert is_classifier(clf_wrapper) is True
  assert is_regressor(clf_wrapper) is False
  assert is_classifier(reg_wrapper) is False
  assert is_regressor(reg_wrapper) is True
  empty_wrapper = _FoldBalancedEstimator(None, balance=True)
  assert is_classifier(empty_wrapper) is False
  assert is_regressor(empty_wrapper) is False


def test_balance_ordered_imbalance_uses_stratified_cv():
  '''Ordered 80/20 y + balance=True uses stratified folds.

  Regression for cycle-7 Finding 1: the wrapper lacked
  ``__sklearn_tags__`` so GridSearchCV degraded StratifiedKFold
  to KFold, single-class fold 3 raised FitFailedWarning and
  best_score was nan. Stratified fold weights here are
  53/13=4.08, 53/14=3.79 and 54/13=4.15 (refit 80/20=4.0);
  KFold weights were 2.3/2.35/inf.
  '''
  import warnings
  from sklearn.exceptions import FitFailedWarning
  from sklearn.linear_model import LogisticRegression
  rng = np.random.RandomState(0)
  y = np.array([0] * 80 + [1] * 20)
  x = pd.DataFrame({'f1': rng.randn(len(y))})
  params = {'method': 'grid',
            'param_grid': {'C': [0.1, 1.0]},
            'cv': 3, 'balance': True}
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    _, _, report = tune_model(
        LogisticRegression(max_iter=1000), x, y, params)
  fit_failed = [w for w in caught
                if issubclass(w.category, FitFailedWarning)]
  assert fit_failed == []
  assert report['best_score'] is not None
  assert np.isfinite(report['best_score'])
  weights = report.get('balance_per_fold_weights') or []
  assert len(weights) == 2 * 3 + 1
  search_weights = weights[:6]
  assert all(3.5 < w < 4.5 for w in search_weights)
  assert weights[-1] == pytest.approx(4.0)
