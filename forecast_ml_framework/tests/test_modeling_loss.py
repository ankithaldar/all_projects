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


def test_get_loss_custom_callable_string(monkeypatch):
  '''Dotted callable string resolves via resolve_handle.'''
  def my_loss(y_true, y_pred):
    '''Dummy custom loss.'''
    return 0.0

  module = _fake_module('my_pkg.losses', my_loss=my_loss)
  monkeypatch.setitem(sys.modules, 'my_pkg.losses', module)
  assert get_loss('sklearn', 'my_pkg.losses.my_loss') is my_loss


def test_get_loss_custom_bogus_string_raises(monkeypatch):
  '''Bogus dotted loss string raises informative error.'''
  monkeypatch.setitem(sys.modules, 'my_pkg.losses', None)
  with pytest.raises(ValueError, match='my_pkg.losses.my_loss'):
    get_loss('sklearn', 'my_pkg.losses.my_loss')


def test_get_loss_custom_non_callable_raises(monkeypatch):
  '''Dotted loss resolving to non-callable raises.'''
  module = _fake_module('my_pkg.losses', my_loss=42)
  monkeypatch.setitem(sys.modules, 'my_pkg.losses', module)
  with pytest.raises(ValueError, match='non-callable'):
    get_loss('sklearn', 'my_pkg.losses.my_loss')


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
  model, _, best = train_models(x, y, x, y, params)
  assert best == 'sklearn'
  assert model.model.get_params()['loss'] == 'absolute_error'
