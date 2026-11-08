#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for per-adapter loss function mapping.'''

import sys
import types

import numpy as np
import pandas as pd
import pytest

from forecast_ml_framework.pipelines.modeling.adapters import (
  CatBoostAdapter,
  H2OAutoMLAdapter,
  LightGBMAdapter,
  SklearnAdapter,
  XGBoostAdapter,
  get_loss,
  train_models,
)
from forecast_ml_framework.pipelines.modeling.envelope import (
  unwrap_envelope,
)


def _data():
  '''Return small regression data.'''
  rng = np.random.RandomState(42)
  x = pd.DataFrame(rng.randn(60, 3), columns=['f1', 'f2', 'f3'])
  y = x['f1'].values + rng.randn(60) * 0.1
  return x, y


class _FakeEstimator:
  '''Estimator stub capturing constructor kwargs.'''

  def __init__(self, **kwargs):
    '''Store kwargs for inspection.'''
    self.kwargs = kwargs

  def fit(self, x, y):
    '''No-op fit returning self.'''
    return self


def _fake_module(name, **attrs):
  '''Build a fake module with the given attributes.'''
  module = types.ModuleType(name)
  for key, value in attrs.items():
    setattr(module, key, value)
  return module


@pytest.mark.parametrize('adapter,loss,expected', [
  ('xgboost', 'logloss', 'binary:logistic'),
  ('xgboost', 'squared_error', 'reg:squarederror'),
  ('xgboost', 'poisson', 'count:poisson'),
  ('lightgbm', 'logloss', 'binary'),
  ('lightgbm', 'squared_error', 'regression'),
  ('lightgbm', 'huber', 'huber'),
  ('catboost', 'logloss', 'Logloss'),
  ('catboost', 'rmse', 'RMSE'),
  ('catboost', 'mae', 'MAE'),
  ('sklearn', 'log_loss', 'log_loss'),
  ('sklearn', 'squared_error', 'squared_error'),
  ('sklearn', 'huber', 'huber'),
  ('h2o_automl', 'logloss', 'logloss'),
  ('h2o_automl', 'squared_error', 'squared_error'),
  ('h2o_automl', 'poisson', 'poisson'),
])
def test_get_loss_maps_per_adapter(adapter, loss, expected):
  '''get_loss maps plain strings to native values.'''
  assert get_loss(adapter, loss) == expected


def test_get_loss_unknown_raises_listing_valid():
  '''Unknown loss raises listing valid losses.'''
  with pytest.raises(ValueError) as exc:
    get_loss('xgboost', 'not_a_loss')
  msg = str(exc.value)
  assert 'xgboost' in msg
  assert 'binary:logistic' in msg


def test_get_loss_custom_callable_string():
  '''Allowlisted dotted callable string resolves.'''
  from sklearn.metrics import mean_absolute_error
  assert get_loss(
      'sklearn',
      'sklearn.metrics.mean_absolute_error') is mean_absolute_error


def test_get_loss_custom_bogus_string_raises():
  '''Bogus allowlisted dotted loss string raises informative error.'''
  with pytest.raises(ValueError, match='sklearn.metrics.nope_loss'):
    get_loss('sklearn', 'sklearn.metrics.nope_loss')


def test_get_loss_custom_non_callable_raises():
  '''Dotted loss resolving to non-callable raises.'''
  with pytest.raises(ValueError, match='not allowlisted'):
    get_loss('sklearn', 'sklearn.__version__')


def test_get_loss_custom_os_system_rejected():
  '''Non-allowlisted dotted loss (os.system) is rejected.'''
  with pytest.raises(ValueError, match='not allowlisted'):
    get_loss('sklearn', 'os.system')


def test_xgboost_loss_overrides_objective(monkeypatch):
  '''XGBoost loss_function overrides objective kwarg.'''
  fake = _fake_module('xgboost', XGBClassifier=_FakeEstimator,
                      XGBRegressor=_FakeEstimator)
  monkeypatch.setitem(sys.modules, 'xgboost', fake)
  x, y = _data()
  params = {'task_type': 'classification',
            'loss_function': 'logloss',
            'objective': 'reg:squarederror'}
  original = dict(params)
  adapter = XGBoostAdapter(params)
  adapter.fit(x, y)
  assert adapter.model.kwargs['objective'] == 'binary:logistic'
  assert 'loss_function' not in adapter.model.kwargs
  assert params == original


def test_lightgbm_loss_maps_to_objective(monkeypatch):
  '''LightGBM loss_function maps to objective.'''
  fake = _fake_module('lightgbm', LGBMClassifier=_FakeEstimator,
                      LGBMRegressor=_FakeEstimator)
  monkeypatch.setitem(sys.modules, 'lightgbm', fake)
  x, y = _data()
  adapter = LightGBMAdapter(
      {'task_type': 'regression', 'loss_function': 'squared_error'})
  adapter.fit(x, y)
  assert adapter.model.kwargs['objective'] == 'regression'
  assert 'loss_function' not in adapter.model.kwargs


def test_catboost_loss_maps_to_loss_function(monkeypatch):
  '''CatBoost loss_function maps to native value.'''
  fake = _fake_module('catboost',
                      CatBoostClassifier=_FakeEstimator,
                      CatBoostRegressor=_FakeEstimator)
  monkeypatch.setitem(sys.modules, 'catboost', fake)
  x, y = _data()
  adapter = CatBoostAdapter(
      {'task_type': 'regression', 'loss_function': 'mae'})
  adapter.fit(x, y)
  assert adapter.model.kwargs['loss_function'] == 'MAE'


def test_sklearn_loss_maps_to_loss_kwarg():
  '''Sklearn loss_function maps to loss kwarg.'''
  x, y = _data()
  adapter = SklearnAdapter(
      {'model_type': 'gradient_boosting_regressor',
       'loss_function': 'absolute_error'})
  adapter.fit(x, y)
  assert adapter.model.get_params()['loss'] == 'absolute_error'


def test_h2o_loss_maps_to_loss_function(monkeypatch):
  '''H2O loss_function maps to native kwarg.'''
  captured = {}

  class FakeEstimator:
    '''Capturing H2O estimator stub.'''

    def __init__(self, **kwargs):
      '''Store kwargs for inspection.'''
      captured.update(kwargs)

    def train(self, x=None, y=None):
      '''No-op train.'''

  automl = types.SimpleNamespace(H2OAutoMLEstimator=FakeEstimator)
  fake = _fake_module('h2o', automl=automl)
  fake.init = lambda **kwargs: None
  monkeypatch.setitem(sys.modules, 'h2o', fake)
  x, y = _data()
  adapter = H2OAutoMLAdapter({'loss_function': 'poisson'})
  adapter.fit(x, y)
  assert captured['loss_function'] == 'poisson'
  assert 'loss_function' not in {
      k: v for k, v in adapter.params.items() if k == 'nope'}


def test_default_no_loss_unchanged(monkeypatch):
  '''No loss_function leaves estimator kwargs unchanged.'''
  fake = _fake_module('xgboost', XGBClassifier=_FakeEstimator,
                      XGBRegressor=_FakeEstimator)
  monkeypatch.setitem(sys.modules, 'xgboost', fake)
  x, y = _data()
  adapter = XGBoostAdapter({'task_type': 'classification',
                            'n_estimators': 5})
  adapter.fit(x, y)
  assert adapter.model.kwargs == {'n_estimators': 5}


def test_objective_passthrough_back_compat(monkeypatch):
  '''Unrecognised native objective passes through.'''
  fake = _fake_module('xgboost', XGBClassifier=_FakeEstimator,
                      XGBRegressor=_FakeEstimator)
  monkeypatch.setitem(sys.modules, 'xgboost', fake)
  x, y = _data()
  adapter = XGBoostAdapter({'task_type': 'classification',
                            'objective': 'multi:softprob'})
  adapter.fit(x, y)
  assert adapter.model.kwargs['objective'] == 'multi:softprob'


def test_unknown_loss_in_adapter_raises(monkeypatch):
  '''Unknown loss_function in adapter raises ValueError.'''
  fake = _fake_module('lightgbm', LGBMClassifier=_FakeEstimator,
                      LGBMRegressor=_FakeEstimator)
  monkeypatch.setitem(sys.modules, 'lightgbm', fake)
  x, y = _data()
  adapter = LightGBMAdapter({'loss_function': 'nope'})
  with pytest.raises(ValueError, match='binary'):
    adapter.fit(x, y)


def test_train_models_top_level_loss_function():
  '''Top-level modeling_params.loss_function is applied.'''
  x, y = _data()
  params = {
    'task_type': 'regression',
    'models_to_use': ['sklearn'],
    'selection_metric': 'rmse',
    'loss_function': 'absolute_error',
    'model_params': {
        'sklearn': {'model_type': 'gradient_boosting_regressor'},
    },
  }
  envelope, _, best = train_models(x, y, x, y, params)
  assert best == 'sklearn'
  assert set(envelope) == {
    'model', 'label_set', 'registry_ref', 'onnx_sidecar'}
  assert unwrap_envelope(envelope).get_params()['loss'] == (
    'absolute_error')


# --- cycle3 loss audit (RED) ---

@pytest.mark.parametrize('dotted', [
  'forecast_ml_framework.cli._run_kedro',
  'forecast_ml_framework.core.datasets.SQLQueryDataSet',
  'forecast_ml_framework.pipelines.modeling.adapters.get_loss',
  'sklearn.metrics.accuracy_score',
  'sklearn.linear_model.LogisticRegression',
  'sklearn.ensemble.RandomForestClassifier',
])
def test_get_loss_rejects_non_loss_allowlisted_top_level(dotted):
  '''Only exact loss callables resolve, not any allowlisted module.'''
  with pytest.raises(ValueError, match='not allowlisted'):
    get_loss('sklearn', dotted)


def test_get_loss_registered_custom_loss_resolves():
  '''Registered custom loss under forecast_ml_framework.losses resolves.'''
  from forecast_ml_framework.losses import huber_loss
  assert get_loss(
      'sklearn', 'forecast_ml_framework.losses.huber_loss') is huber_loss
  assert huber_loss([0.0], [1.0]) == pytest.approx(0.5)
  assert get_loss(
      'sklearn',
      'forecast_ml_framework.losses.quantile_loss') is not None


def test_get_loss_standard_sklearn_metric_resolves():
  '''Standard sklearn metric losses resolve by exact dotted name.'''
  from sklearn.metrics import mean_squared_error
  assert get_loss(
      'sklearn',
      'sklearn.metrics.mean_squared_error') is mean_squared_error


def test_sklearn_adapter_unsupported_loss_kwarg_fails_loudly():
  '''Estimator rejecting loss kwarg raises, not silent default.'''
  x, y = _data()
  adapter = SklearnAdapter(
      {'model_type': 'linear_regression',
       'loss_function': 'squared_error'})
  with pytest.raises(ValueError, match='rejected|loss'):
    adapter.fit(x, y)


def test_handle_adapter_unsupported_loss_kwarg_fails_loudly():
  '''HandleAdapter does not silently fall back to cls().'''
  x, y = _data()
  from forecast_ml_framework.pipelines.modeling.adapters import (
    create_adapter,
  )
  adapter = create_adapter(
      'sklearn.linear_model.LinearRegression',
      {'loss_function': 'squared_error'})
  with pytest.raises(ValueError, match='rejected|loss'):
    adapter.fit(x, y)


# --- losses.py coverage: quantile/pinball math + register_loss ---

@pytest.mark.parametrize('y_true,y_pred,quantile,expected', [
  ([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], 0.5, 0.5),
  ([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], 0.9, 0.9),
  ([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], 0.1, 0.1),
  ([0.0, 0.0], [1.0, 2.0], 0.5, 0.75),
  ([0.0, 0.0], [1.0, 2.0], 0.9, 0.15),
  ([2.0, 2.0], [2.0, 2.0], 0.5, 0.0),
])
def test_quantile_loss_math(y_true, y_pred, quantile, expected):
  '''quantile_loss matches hand-computed pinball values.'''
  from forecast_ml_framework.losses import quantile_loss
  assert quantile_loss(
      y_true, y_pred, quantile=quantile) == pytest.approx(expected)


def test_pinball_loss_is_quantile_alias_and_matches_sklearn():
  '''pinball_loss is quantile_loss and matches sklearn.'''
  from sklearn.metrics import mean_pinball_loss

  from forecast_ml_framework.losses import pinball_loss, quantile_loss
  assert pinball_loss is quantile_loss
  y_true = [1.0, 2.0, 3.0, 0.0]
  y_pred = [1.0, 1.0, 1.0, 1.0]
  for quantile in (0.1, 0.5, 0.9):
    expected = mean_pinball_loss(y_true, y_pred, alpha=quantile)
    assert pinball_loss(
        y_true, y_pred, quantile=quantile) == pytest.approx(expected)


@pytest.mark.parametrize('bad_name', [
  'my_losses.my_loss',
  'forecast_ml_framework.losses.bad-name',
  'forecast_ml_framework.losses.',
  'forecast_ml_framework.losses.has space',
  'sklearn.metrics.mean_squared_error',
  '',
  123,
  None,
])
def test_register_loss_invalid_name_raises(bad_name):
  '''register_loss rejects names outside the losses namespace.'''
  from forecast_ml_framework.losses import register_loss
  with pytest.raises(ValueError, match='must be'):
    register_loss(bad_name, lambda y_true, y_pred: 0.0)


@pytest.mark.parametrize('not_callable', [42, None, 'loss', []])
def test_register_loss_non_callable_raises(not_callable):
  '''register_loss rejects non-callable func values.'''
  from forecast_ml_framework.losses import register_loss
  with pytest.raises(ValueError, match='not callable'):
    register_loss(
        'forecast_ml_framework.losses.not_callable_loss', not_callable)


def test_register_loss_duplicate_raises():
  '''register_loss rejects a duplicate dotted name.'''
  from forecast_ml_framework.losses import (
    LOSS_CALLABLES,
    register_loss,
  )
  name = 'forecast_ml_framework.losses.duplicate_loss'
  try:
    register_loss(name, lambda y_true, y_pred: 0.0)
    with pytest.raises(ValueError, match='already'):
      register_loss(name, lambda y_true, y_pred: 1.0)
  finally:
    LOSS_CALLABLES.pop(name, None)


def test_register_loss_duplicate_same_callable_raises():
  '''register_loss raises even when re-registering the same callable.'''
  from forecast_ml_framework.losses import huber_loss, register_loss
  with pytest.raises(ValueError, match='already'):
    register_loss(
        'forecast_ml_framework.losses.huber_loss', huber_loss)
