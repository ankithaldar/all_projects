#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for model adapters and the train node.'''

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
  create_adapter,
  resolve_handle,
  train_models,
)
from forecast_ml_framework.pipelines.modeling.metrics import (
  compute_metrics,
)


def _clf_data():
  '''Return small linearly separable classification data.'''
  rng = np.random.RandomState(42)
  x = rng.randn(120, 3)
  y = (x[:, 0] + x[:, 1] > 0).astype(int)
  cols = ['f1', 'f2', 'f3']
  return pd.DataFrame(x, columns=cols), y, cols


def test_sklearn_adapter_trains_and_predicts():
  '''Sklearn adapter fits, predicts and returns proba.'''
  x, y, _ = _clf_data()
  adapter = SklearnAdapter({'model_type': 'logistic_regression'})
  adapter.fit(x, y)
  pred = adapter.predict(x)
  proba = adapter.predict_proba(x)
  assert len(pred) == len(y)
  assert proba.shape == (len(y), 2)
  assert np.allclose(proba.sum(axis=1), 1.0)


def test_sklearn_adapter_save_load_roundtrip(tmp_path):
  '''Sklearn adapter save/load preserves predictions.'''
  x, y, _ = _clf_data()
  adapter = SklearnAdapter({})
  adapter.fit(x, y)
  path = str(tmp_path / 'model.pkl')
  adapter.save(path)
  loaded = SklearnAdapter.load(path)
  assert (loaded.predict(x) == adapter.predict(x)).all()


class _EvilReduce:
  '''Pickle payload that would execute a shell command.'''

  def __init__(self, marker):
    '''Store marker path the payload would create.'''
    self.marker = str(marker)

  def __reduce__(self):
    '''Return an os.system call as the reconstruction.'''
    import os
    return (os.system, (f'touch {self.marker}',))


def test_adapter_load_blocks_evil_pickle(tmp_path):
  '''Adapter load rejects a malicious pickle, no side effect.'''
  import pickle
  marker = tmp_path / 'pwned'
  path = tmp_path / 'evil.pkl'
  path.write_bytes(pickle.dumps(_EvilReduce(marker)))
  with pytest.raises(ValueError):
    SklearnAdapter.load(str(path))
  assert not marker.exists()


def test_unsupported_lib_raises_informative_error(monkeypatch):
  '''Missing optional lib suggests disabling it in params.

  Mocks the import failure so the test passes whether or not
  xgboost is installed in the current environment.
  '''
  monkeypatch.setitem(sys.modules, 'xgboost', None)
  adapter = create_adapter('xgboost', {})
  with pytest.raises(RuntimeError) as exc:
    adapter.fit(*_clf_data()[:2])
  msg = str(exc.value)
  assert 'xgboost' in msg
  assert 'models_to_use' in msg


class _FakeEstimator:
  '''Minimal estimator stub that records kwargs.'''

  def __init__(self, **kwargs):
    '''Store kwargs for later inspection.'''
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


@pytest.mark.parametrize('adapter_cls,module,attrs', [
  (XGBoostAdapter, 'xgboost',
   {'XGBClassifier': _FakeEstimator,
    'XGBRegressor': _FakeEstimator}),
  (LightGBMAdapter, 'lightgbm',
   {'LGBMClassifier': _FakeEstimator,
    'LGBMRegressor': _FakeEstimator}),
  (CatBoostAdapter, 'catboost',
   {'CatBoostClassifier': _FakeEstimator,
    'CatBoostRegressor': _FakeEstimator}),
])
def test_fit_does_not_mutate_params_and_double_fit_works(
    monkeypatch, adapter_cls, module, attrs):
  '''Fit keeps params unchanged so a second fit still works.'''
  monkeypatch.setitem(sys.modules, module,
                      _fake_module(module, **attrs))
  x, y, _ = _clf_data()
  params = {'task_type': 'regression'}
  original = dict(params)
  adapter = adapter_cls(params)
  adapter.fit(x, y)
  adapter.fit(x, y)
  assert adapter.params == original
  assert params == original


class _FakeH2OModel:
  '''Fake H2O model exposing predict and predict_proba.'''

  def predict(self, x):
    '''Return dummy labels distinct from probabilities.'''
    return np.zeros(len(x), dtype=int)

  def predict_proba(self, x):
    '''Return a valid two-column probability array.'''
    n = len(x)
    return np.column_stack([np.full(n, 0.25), np.full(n, 0.75)])


def test_h2o_predict_proba_returns_real_proba():
  '''H2O predict_proba returns proba, not predict labels.'''
  x, y, _ = _clf_data()
  adapter = H2OAutoMLAdapter({})
  adapter.model = _FakeH2OModel()
  proba = adapter.predict_proba(x)
  assert proba.shape == (len(y), 2)
  assert np.allclose(proba.sum(axis=1), 1.0)


class _FakeH2OModelNoProba:
  '''Fake H2O model without predict_proba.'''

  def predict(self, x):
    '''Return dummy labels.'''
    return np.zeros(len(x), dtype=int)


def test_h2o_predict_proba_without_support_raises():
  '''H2O predict_proba raises clearly when unsupported.'''
  x, _, _ = _clf_data()
  adapter = H2OAutoMLAdapter({})
  adapter.model = _FakeH2OModelNoProba()
  with pytest.raises(NotImplementedError, match='predict_proba'):
    adapter.predict_proba(x)


def test_train_models_dedupes_models_to_use(monkeypatch):
  '''Duplicate model names are fitted only once.'''
  x, y, _ = _clf_data()
  original_fit = SklearnAdapter.fit
  count = [0]

  def counting_fit(self, x_data, y_data):
    '''Count fits while delegating to the real fit.'''
    count[0] += 1
    return original_fit(self, x_data, y_data)

  monkeypatch.setattr(SklearnAdapter, 'fit', counting_fit)
  params = {
    'task_type': 'classification',
    'models_to_use': ['sklearn'] * 500,
    'selection_metric': 'f1_weighted',
    'random_state': 42,
    'model_params': {'sklearn': {'model_type': 'logistic_regression'}},
  }
  _, per_model, best = train_models(x, y, x, y, params)
  assert count[0] == 1
  assert best == 'sklearn'
  assert list(per_model) == ['sklearn']


def test_unknown_adapter_name_raises():
  '''Unknown adapter names raise a clear error.'''
  with pytest.raises(ValueError):
    create_adapter('nope', {})


def test_train_models_selects_best_by_metric():
  '''Train node returns fitted model and per-model metrics.'''
  x, y, _ = _clf_data()
  params = {
    'task_type': 'classification',
    'models_to_use': ['sklearn'],
    'selection_metric': 'f1_weighted',
    'random_state': 42,
    'model_params': {'sklearn': {'model_type': 'logistic_regression'}},
  }
  model, per_model, best = train_models(x, y, x, y, params)
  assert best == 'sklearn'
  assert 'sklearn' in per_model
  assert 'f1_weighted' in per_model['sklearn']
  assert len(model.predict(x)) == len(y)


def test_resolve_handle_real_sklearn():
  '''Dotted sklearn handle resolves to the real class.'''
  from sklearn.linear_model import LogisticRegression
  assert resolve_handle(
      'sklearn.linear_model.LogisticRegression'
  ) is LogisticRegression


def test_resolve_handle_bogus_raises_informative():
  '''Bogus dotted handle raises an informative error.'''
  with pytest.raises(ValueError) as exc:
    resolve_handle('bogus.module.DoesNotExist')
  msg = str(exc.value)
  assert 'bogus.module.DoesNotExist' in msg
  assert 'model_handle' in msg


def test_create_adapter_accepts_dotted_handle():
  '''create_adapter accepts a dotted importable handle.'''
  x, y, _ = _clf_data()
  adapter = create_adapter(
      'sklearn.linear_model.LogisticRegression', {})
  adapter.fit(x, y)
  assert adapter.predict_proba(x).shape == (len(y), 2)


def test_create_adapter_short_name_still_works():
  '''Short adapter names remain supported for back-compat.'''
  adapter = create_adapter('sklearn', {})
  assert isinstance(adapter, SklearnAdapter)


def test_train_models_accepts_model_handle_in_params():
  '''train_models honours model_handle in per-model params.'''
  x, y, _ = _clf_data()
  params = {
    'task_type': 'classification',
    'models_to_use': ['sklearn'],
    'selection_metric': 'f1_weighted',
    'random_state': 42,
    'model_params': {
      'sklearn': {
        'model_handle':
          'sklearn.ensemble.RandomForestClassifier',
        'n_estimators': 5,
        'random_state': 42,
      },
    },
  }
  model, per_model, best = train_models(x, y, x, y, params)
  assert best == 'sklearn'
  assert 'f1_weighted' in per_model['sklearn']
  assert len(model.predict(x)) == len(y)


def test_train_models_accepts_handle_as_model_name():
  '''models_to_use may hold a dotted handle directly.'''
  x, y, _ = _clf_data()
  params = {
    'task_type': 'classification',
    'models_to_use': ['sklearn.linear_model.LogisticRegression'],
    'selection_metric': 'f1_weighted',
    'random_state': 42,
    'model_params': {},
  }
  model, per_model, best = train_models(x, y, x, y, params)
  assert best == 'sklearn.linear_model.LogisticRegression'
  assert len(model.predict(x)) == len(y)


def test_compute_metrics_classification_keys():
  '''Classification metrics include the approved set.'''
  y_true = np.array([0, 1, 1, 0, 1, 0])
  y_pred = np.array([0, 1, 0, 0, 1, 1])
  proba = np.array([[.9, .1], [.2, .8], [.6, .4],
                    [.8, .2], [.3, .7], [.4, .6]])
  got = compute_metrics(y_true, y_pred, proba, 'classification')
  for key in ('accuracy', 'roc_auc', 'f1_weighted', 'kappa',
              'mcc', 'brier', 'ece', 'confusion_matrix'):
    assert key in got


def test_compute_metrics_regression_keys():
  '''Regression metrics include the approved set.'''
  y_true = np.array([1.0, 2.0, 3.0, 4.0])
  y_pred = np.array([1.1, 1.9, 3.2, 3.8])
  got = compute_metrics(y_true, y_pred, None, 'regression')
  for key in ('mae', 'rmse', 'r2', 'mape', 'bias', 'max_error'):
    assert key in got

# --- cycle2 handles hardening (RED) ---

@pytest.mark.parametrize('handle', [
  'os.system', 'subprocess.Popen', 'builtins.eval', 'builtins.exec',
  'os.popen', 'subprocess.run', 'sys.modules', 'builtins.__import__',
])
def test_resolve_handle_rejects_non_allowlisted(handle):
  '''Dangerous callables are rejected by resolve_handle.'''
  with pytest.raises(ValueError):
    resolve_handle(handle)

def test_handle_adapter_fit_rejects_os_system():
  '''HandleAdapter.fit cannot instantiate os.system.'''
  x, y, _ = _clf_data()
  adapter = create_adapter('os.system', {})
  with pytest.raises(ValueError):
    adapter.fit(x, y)

def test_get_loss_rejects_os_system_dotted():
  '''get_loss dotted os.system is rejected.'''
  from forecast_ml_framework.pipelines.modeling.adapters import get_loss
  with pytest.raises(ValueError):
    get_loss('sklearn', 'os.system')

def test_get_loss_rejects_subprocess_popen_dotted():
  '''get_loss dotted subprocess.Popen is rejected.'''
  from forecast_ml_framework.pipelines.modeling.adapters import get_loss
  with pytest.raises(ValueError):
    get_loss('sklearn', 'subprocess.Popen')

def test_calibration_handle_os_system_rejected_in_train():
  '''train_models rejects calibration_handle=os.system.'''
  x, y, _ = _clf_data()
  params = {
    'task_type': 'classification',
    'models_to_use': ['sklearn'],
    'calibration_handle': 'os.system',
    'model_params': {'sklearn': {}},
  }
  with pytest.raises(ValueError):
    train_models(x, y, x, y, params)

def test_sklearn_handle_still_fits():
  '''Allowlisted sklearn handle still fits via HandleAdapter.'''
  x, y, _ = _clf_data()
  adapter = create_adapter(
    'sklearn.linear_model.LogisticRegression', {})
  adapter.fit(x, y)
  assert adapter.predict_proba(x).shape == (len(y), 2)
