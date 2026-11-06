#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for tuning/balance wiring through train_models.'''

import numpy as np
import pandas as pd

from forecast_ml_framework.pipelines.modeling.adapters import (
  SklearnAdapter,
  train_models,
)


def _clf_data():
  '''Return small linearly separable classification data.'''
  rng = np.random.RandomState(42)
  x = rng.randn(120, 3)
  y = (x[:, 0] + x[:, 1] > 0).astype(int)
  return pd.DataFrame(x, columns=['f1', 'f2', 'f3']), y


def test_train_models_with_tuning_report():
  '''train_models tunes sklearn model and stores report.'''
  x, y = _clf_data()
  params = {
    'task_type': 'classification',
    'models_to_use': ['sklearn'],
    'selection_metric': 'f1_weighted',
    'random_state': 42,
    'tuning': {
      'method': 'grid',
      'param_grid': {'C': [0.1, 1.0]},
      'cv': 3,
      'scoring': 'accuracy',
    },
    'model_params': {'sklearn': {'model_type': 'logistic_regression'}},
  }
  model, per_model, best = train_models(x, y, x, y, params)
  assert best == 'sklearn'
  assert 'tuning' in per_model['sklearn']
  assert per_model['sklearn']['tuning']['method'] == 'grid'
  assert per_model['sklearn']['tuning']['best_params']['C'] in (
    0.1, 1.0)
  assert model.model.get_params()['C'] == (
    per_model['sklearn']['tuning']['best_params']['C'])
  assert len(model.predict(x)) == len(y)


def test_tuning_report_persisted_via_pipeline_node():
  '''train_models per_model tuning flows into build_tuning_report.'''
  from forecast_ml_framework.pipelines.modeling.pipeline import (
    build_tuning_report,
  )
  x, y = _clf_data()
  params = {
    'task_type': 'classification',
    'models_to_use': ['sklearn'],
    'selection_metric': 'f1_weighted',
    'random_state': 42,
    'tuning': {'method': 'grid', 'param_grid': {'C': [1.0]},
               'cv': 2, 'scoring': 'accuracy'},
    'model_params': {'sklearn': {'model_type': 'logistic_regression'}},
  }
  _model, per_model, _best = train_models(x, y, x, y, params)
  report = build_tuning_report(per_model)
  assert report['sklearn']['method'] == 'grid'
  assert report['sklearn']['best_params'] == {'C': 1.0}


def test_train_models_without_tuning_unchanged():
  '''Without tuning no tuning key is added to per-model metrics.'''
  x, y = _clf_data()
  params = {
    'task_type': 'classification',
    'models_to_use': ['sklearn'],
    'selection_metric': 'f1_weighted',
    'random_state': 42,
    'model_params': {'sklearn': {'model_type': 'logistic_regression'}},
  }
  _, per_model, best = train_models(x, y, x, y, params)
  assert best == 'sklearn'
  assert 'tuning' not in per_model['sklearn']


def test_train_models_per_model_tuning():
  '''Per-model tuning dict in model_params is honoured.'''
  x, y = _clf_data()
  params = {
    'task_type': 'classification',
    'models_to_use': ['sklearn'],
    'selection_metric': 'f1_weighted',
    'random_state': 42,
    'model_params': {
      'sklearn': {
        'model_type': 'logistic_regression',
        'tuning': {
          'method': 'grid',
          'param_grid': {'C': [1.0]},
          'cv': 2,
          'scoring': 'accuracy',
        },
      },
    },
  }
  _, per_model, best = train_models(x, y, x, y, params)
  assert best == 'sklearn'
  assert per_model['sklearn']['tuning']['best_params'] == {'C': 1.0}


def test_balance_injects_class_weight_sklearn():
  '''balance=True injects class_weight into sklearn estimator.'''
  x, y = _clf_data()
  # Make data imbalanced by keeping most negative rows.
  keep = np.concatenate([np.where(y == 0)[0],
                         np.where(y == 1)[0][:10]])
  x_small, y_small = x.iloc[keep], y[keep]
  adapter = SklearnAdapter({
    'model_type': 'logistic_regression',
    'balance': True,
  })
  adapter.fit(x_small, y_small)
  assert 'balance' not in adapter.model.get_params()
  class_weight = adapter.model.get_params().get('class_weight')
  assert class_weight is not None
  assert class_weight[1] > class_weight[0]


def test_balance_injects_scale_pos_weight_xgboost(monkeypatch):
  '''balance=True prefers scale_pos_weight when supported.'''
  import sys
  import types

  class FakeXGB:
    '''Fake XGB estimator recording kwargs.'''

    def __init__(self, **kwargs):
      '''Store kwargs.'''
      self.kwargs = kwargs

    def fit(self, x, y):
      '''No-op fit.'''
      return self

    def predict(self, x):
      '''Return zeros.'''
      return np.zeros(len(x), dtype=int)

  module = types.ModuleType('xgboost')
  module.XGBClassifier = FakeXGB
  module.XGBRegressor = FakeXGB
  monkeypatch.setitem(sys.modules, 'xgboost', module)
  from forecast_ml_framework.pipelines.modeling.adapters import (
    XGBoostAdapter,
  )
  x, y = _clf_data()
  adapter = XGBoostAdapter({'task_type': 'classification',
                            'balance': True})
  adapter.fit(x, y)
  assert 'balance' not in adapter.model.kwargs
  assert adapter.model.kwargs.get('scale_pos_weight') is not None
  assert adapter.model.kwargs['scale_pos_weight'] > 0


def test_train_models_balance_via_params():
  '''Top-level balance flag is propagated to the model.'''
  x, y = _clf_data()
  params = {
    'task_type': 'classification',
    'models_to_use': ['sklearn'],
    'selection_metric': 'f1_weighted',
    'random_state': 42,
    'balance': True,
    'model_params': {'sklearn': {'model_type': 'logistic_regression'}},
  }
  model, _per_model, best = train_models(x, y, x, y, params)
  assert best == 'sklearn'
  assert model.model.get_params().get('class_weight') is not None


def _fake_module(monkeypatch, name, **attrs):
  '''Install a fake optional library module.'''
  import sys
  import types
  module = types.ModuleType(name)
  for key, value in attrs.items():
    setattr(module, key, value)
  monkeypatch.setitem(sys.modules, name, module)
  return module


def test_zoo_tuning_raises_for_xgboost(monkeypatch):
  '''Tuning config for XGBoost raises ValueError (fail loud).'''
  import pytest

  class FakeXGB:
    '''Fake XGB estimator.'''

    def __init__(self, **kwargs):
      '''Store kwargs.'''

    def fit(self, x, y):
      '''No-op fit.'''

  _fake_module(monkeypatch, 'xgboost',
               XGBClassifier=FakeXGB, XGBRegressor=FakeXGB)
  from forecast_ml_framework.pipelines.modeling.adapters import (
    XGBoostAdapter,
  )
  x, y = _clf_data()
  adapter = XGBoostAdapter({
    'task_type': 'classification',
    'tuning': {'method': 'grid', 'param_grid': {'max_depth': [3]}},
  })
  with pytest.raises(ValueError, match='tuning'):
    adapter.fit(x, y)


def test_zoo_tuning_raises_for_lightgbm_and_catboost(monkeypatch):
  '''Tuning config for LightGBM/CatBoost raises ValueError.'''
  import pytest

  class Fake:
    '''Fake estimator.'''

    def __init__(self, **kwargs):
      '''Store kwargs.'''

    def fit(self, x, y):
      '''No-op fit.'''

  _fake_module(monkeypatch, 'lightgbm',
               LGBMClassifier=Fake, LGBMRegressor=Fake)
  _fake_module(monkeypatch, 'catboost',
               CatBoostClassifier=Fake, CatBoostRegressor=Fake)
  from forecast_ml_framework.pipelines.modeling.adapters import (
    CatBoostAdapter,
    LightGBMAdapter,
  )
  x, y = _clf_data()
  tuning = {'method': 'grid', 'param_grid': {'num_leaves': [7]}}
  with pytest.raises(ValueError, match='tuning'):
    LightGBMAdapter({'task_type': 'classification',
                     'tuning': tuning}).fit(x, y)
  with pytest.raises(ValueError, match='tuning'):
    CatBoostAdapter({'task_type': 'classification',
                     'tuning': tuning}).fit(x, y)


def test_h2o_tuning_and_balance_raise(monkeypatch):
  '''Tuning and balance for H2O raise ValueError (fail loud).'''
  import pytest

  class FakeH2O:
    '''Fake h2o module holder.'''

  _fake_module(monkeypatch, 'h2o')
  from forecast_ml_framework.pipelines.modeling.adapters import (
    H2OAutoMLAdapter,
  )
  x, y = _clf_data()
  with pytest.raises(ValueError, match='tuning'):
    H2OAutoMLAdapter({
      'tuning': {'method': 'grid', 'param_grid': {'x': [1]}},
    }).fit(x, y)
  with pytest.raises(ValueError, match='balance'):
    H2OAutoMLAdapter({'balance': True}).fit(x, y)
