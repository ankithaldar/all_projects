#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''W8b modeling-core cycle-1 fixes: RED regression tests.'''

import sys
import types

import numpy as np
import pandas as pd
import pytest


def _frame(n=100, seed=0):
  '''Return a binary frame with a time column.'''
  rng = np.random.RandomState(seed)
  return pd.DataFrame({
      'f1': rng.randn(n),
      'time': np.arange(n),
      'target': (rng.rand(n) > 0.5).astype(int),
  })


# --- ML-1: train_models selects inside x_train, never y_test ---

class _StubModel:
  '''Stub whose predictions are perfect on train-like rows only.'''

  def __init__(self, perfect_on_train):
    '''Store whether this stub is perfect on train-like rows.'''
    self.perfect = perfect_on_train

  def fit(self, x, y):
    '''Accept fit, expose binary classes_ like a fitted model.'''
    self.classes_ = np.array([0, 1])
    return self

  def _pred(self, x):
    '''Return y_like (or its flip) as predictions.'''
    like = np.asarray(x['y_like']).astype(int)
    return like if self.perfect else 1 - like

  def predict(self, x):
    '''Predict labels from the y_like column.'''
    return self._pred(x)

  def predict_proba(self, x):
    '''Return two-column proba matching predictions.'''
    pred = self._pred(x).astype(float)
    return np.column_stack([1 - pred, pred])


def test_train_models_selects_by_train_not_test(monkeypatch):
  '''A wins on train, B wins on test: returns A, ignores y_test.'''
  from forecast_ml_framework.pipelines.modeling import adapters
  y_train = np.array([0, 1] * 30)
  x_train = pd.DataFrame({'y_like': y_train})
  y_test = np.array([0, 1] * 10)
  x_test = pd.DataFrame({'y_like': 1 - y_test})

  def fake_create(name, params=None):
    '''Return stub A for model_a and stub B for model_b.'''
    return _StubModel(perfect_on_train=(name == 'model_a'))

  monkeypatch.setattr(adapters, 'create_adapter', fake_create)
  params = {'task_type': 'classification',
            'models_to_use': ['model_a', 'model_b'],
            'selection_metric': 'f1_weighted'}
  _, _, best = adapters.train_models(
      x_train, y_train, x_test, y_test, params)
  assert best == 'model_a'


def test_train_models_y_test_nan_still_selects_by_train(monkeypatch):
  '''NaN y_test must not affect selection (never read for picking).'''
  from forecast_ml_framework.pipelines.modeling import adapters
  y_train = np.array([0, 1] * 30)
  x_train = pd.DataFrame({'y_like': y_train})
  x_test = pd.DataFrame({'y_like': np.zeros(20, dtype=int)})
  y_test = np.full(20, np.nan)

  def fake_create(name, params=None):
    '''Return stub A for model_a and stub B for model_b.'''
    return _StubModel(perfect_on_train=(name == 'model_a'))

  monkeypatch.setattr(adapters, 'create_adapter', fake_create)
  params = {'task_type': 'classification',
            'models_to_use': ['model_a', 'model_b'],
            'selection_metric': 'f1_weighted'}
  _, _, best = adapters.train_models(
      x_train, y_train, x_test, y_test, params)
  assert best == 'model_a'


# --- ML-4: split_data shuffle / time / stratify-fallback meta ---

def test_split_time_method_is_ordered():
  '''method=time: every train index precedes every test index.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    split_data,
  )
  frame = _frame(100)
  params = {'target_col': 'target', 'task_type': 'classification',
            'sampling_params': {'method': 'time', 'time_col': 'time',
                                'test_size': 0.2}}
  x_tr, x_te, _, _, meta = split_data(frame, params)
  assert meta['sampling_method'] == 'time'
  assert x_tr['time'].max() < x_te['time'].min()


def test_split_no_shuffle_preserves_order():
  '''shuffle=False with random method keeps the first rows in train.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    split_data,
  )
  frame = _frame(100)
  params = {'target_col': 'target', 'task_type': 'classification',
            'sampling_params': {'method': 'random', 'shuffle': False,
                                'test_size': 0.2}}
  x_tr, x_te, _, _, meta = split_data(frame, params)
  assert meta['sampling_method'] == 'random'
  assert list(x_tr['time']) == list(range(80))
  assert list(x_te['time']) == list(range(80, 100))


def test_split_stratify_fallback_warns_and_meta_is_honest():
  '''Impossible stratify warns loudly and meta says random.'''
  from forecast_ml_framework.pipelines.modeling.data_split import (
    split_data,
  )
  frame = pd.DataFrame({'f1': np.arange(100.0),
                        'target': [0] * 99 + [1]})
  params = {'target_col': 'target', 'task_type': 'classification',
            'sampling_params': {'method': 'stratified',
                                'test_size': 0.2}}
  with pytest.warns(UserWarning, match='stratified'):
    _, _, _, _, meta = split_data(frame, params)
  assert meta['sampling_method'] == 'random'


# --- ML-10: full-precision threshold + MCC plateau midpoint ---

def test_threshold_full_precision_not_rounded():
  '''Optimum at proba 0.1234 is returned at full precision.'''
  from forecast_ml_framework.pipelines.modeling.threshold import (
    select_threshold,
  )
  y_true = np.array([0, 1])
  proba = np.array([0.05, 0.1234])
  thr = select_threshold(y_true, proba, 'pr_curve')
  assert thr == pytest.approx(0.1234, abs=1e-6)


def test_mcc_threshold_is_plateau_midpoint():
  '''MCC optimum plateau returns its midpoint, not the left edge.'''
  from forecast_ml_framework.pipelines.modeling.threshold import (
    select_threshold,
  )
  y_true = np.array([0, 0, 1, 1])
  proba = np.array([0.1, 0.2, 0.8, 0.9])
  thr = select_threshold(y_true, proba, 'mcc_curve')
  assert thr == pytest.approx(0.505, abs=0.01)
  assert thr > 0.4


# --- ML-11: betacal OOF + warn/record + binary validation ---

class _FakeBetaCalibration:
  '''Fake betacal recording the probas it was fitted on.'''

  recorded = []

  def fit(self, proba, y):
    '''Record fit inputs and return self.'''
    type(self).recorded.append(np.asarray(proba, dtype=float).copy())
    return self

  def predict_proba(self, proba):
    '''Return identity two-column proba for binary input.'''
    arr = np.asarray(proba, dtype=float)
    pos = arr[:, 1] if arr.ndim == 2 and arr.shape[1] == 2 else arr
    return np.column_stack([1 - pos, pos])


def _install_fake_betacal(monkeypatch):
  '''Install the fake betacal module and clear its record.'''
  module = types.ModuleType('betacal')
  module.BetaCalibration = _FakeBetaCalibration
  _FakeBetaCalibration.recorded = []
  monkeypatch.setitem(sys.modules, 'betacal', module)


def test_betacal_fit_on_out_of_fold_probas(monkeypatch):
  '''Betacal calibrator input comes from held-out folds, not in-sample.'''
  from forecast_ml_framework.pipelines.modeling.adapters import (
    SklearnAdapter,
  )
  from forecast_ml_framework.pipelines.modeling.calibration import (
    calibrate_model,
  )
  _install_fake_betacal(monkeypatch)
  frame = _frame(80)
  x = frame[['f1']]
  y = frame['target']
  base = SklearnAdapter({}).fit(x, y)
  in_sample = base.model.predict_proba(x)[:, 1]
  params = {'task_type': 'classification',
            'calibration': {'method': 'betacal', 'cv': 2}}
  _, meta = calibrate_model(base, x, y, params)
  assert meta['method'] == 'betacal'
  assert len(_FakeBetaCalibration.recorded) == 1
  oof = _FakeBetaCalibration.recorded[0]
  assert oof.shape == in_sample.shape
  assert not np.allclose(oof, in_sample)


def test_betacal_missing_warns_and_records_requested(monkeypatch):
  '''Missing betacal warns and records requested vs used method.'''
  from forecast_ml_framework.pipelines.modeling.adapters import (
    SklearnAdapter,
  )
  from forecast_ml_framework.pipelines.modeling.calibration import (
    calibrate_model,
  )
  monkeypatch.setitem(sys.modules, 'betacal', None)
  frame = _frame(80)
  x = frame[['f1']]
  y = frame['target']
  base = SklearnAdapter({}).fit(x, y)
  params = {'task_type': 'classification',
            'calibration': {'method': 'betacal', 'cv': 2}}
  with pytest.warns(UserWarning, match='betacal'):
    _, meta = calibrate_model(base, x, y, params)
  assert meta['requested_method'] == 'betacal'
  assert meta['method'] == 'sigmoid'


def test_betacal_multiclass_raises(monkeypatch):
  '''Betacal on a multiclass target raises ValueError.'''
  from forecast_ml_framework.pipelines.modeling.adapters import (
    SklearnAdapter,
  )
  from forecast_ml_framework.pipelines.modeling.calibration import (
    calibrate_model,
  )
  _install_fake_betacal(monkeypatch)
  rng = np.random.RandomState(0)
  x = pd.DataFrame({'f1': rng.randn(60)})
  y = np.array([0, 1, 2] * 20)
  base = SklearnAdapter({}).fit(x, y)
  params = {'task_type': 'classification',
            'calibration': {'method': 'betacal', 'cv': 2}}
  with pytest.raises(ValueError, match='binary'):
    calibrate_model(base, x, y, params)


# --- ML-12: per-fold balance weights in tuning ---

def test_tune_model_balance_weights_differ_per_fold():
  '''Tuning with balance=True recomputes weights per fold.

  Post cycle-7 Finding 1 fix the folds are stratified (not
  unshuffled KFold): ordered 80/20 y with cv=3 gives per-fold
  train ratios 53/13, 53/14 and 54/13 (all near the full
  80/20=4.0 ratio, KFold gave 2.3/2.35/inf). Rounding makes
  the stratified weights differ slightly, and best_score
  must be finite (KFold produced nan via single-class folds).
  '''
  from sklearn.linear_model import LogisticRegression
  from forecast_ml_framework.pipelines.modeling.tuning import tune_model
  rng = np.random.RandomState(0)
  y = np.array([0] * 80 + [1] * 20)
  x = pd.DataFrame({'f1': rng.randn(len(y))})
  params = {'method': 'grid',
            'param_grid': {'C': [0.1, 1.0]},
            'cv': 3, 'balance': True}
  _, _, report = tune_model(LogisticRegression(max_iter=1000), x, y, params)
  weights = report.get('balance_per_fold_weights') or []
  assert len(set(weights)) > 1
  assert all(3.5 < w < 4.5 for w in weights)
  assert report['best_score'] is not None
  assert np.isfinite(report['best_score'])


# --- ML-13: selected_ntile is tile-vs-overall lift ---

def test_selected_ntile_top_decile_lift_vs_overall():
  '''Perfect top decile at prevalence 0.1 gives lift 10.0.'''
  from forecast_ml_framework.pipelines.modeling.evaluate import (
    evaluate_model,
  )
  n = 100
  y_true = np.array([0] * 90 + [1] * 10)
  proba_pos = np.concatenate([
      np.linspace(0.0, 0.89, 90),
      np.linspace(0.9, 0.99, 10),
  ])
  y_proba = np.column_stack([1 - proba_pos, proba_pos])
  y_pred = (proba_pos >= 0.5).astype(int)
  x = pd.DataFrame({'f1': np.arange(n, dtype=float)})
  params = {'task_type': 'classification',
            'metrics': ['lift_top_decile'],
            'eval_metric': 'LIFT_DECILE_10'}
  out = evaluate_model(y_true, y_pred, y_proba, x, params)
  assert out['selected_ntile']['score'] == pytest.approx(10.0)


# --- ML-14: lift = precision_decile / prevalence, unequal tiles ---

def test_lift_table_lift_uses_precision_over_prevalence():
  '''Unequal tiles: lift equals (target/decile_count)/prevalence.'''
  from forecast_ml_framework.pipelines.modeling.lift_table import (
    compute_lift_table,
  )
  proba = np.array([0.1] * 50 + [0.5] * 30 + [0.9] * 20)
  y_true = np.array([0] * 45 + [1] * 5 + [0] * 20 + [1] * 10
                    + [0] * 10 + [1] * 10)
  y_pred = (proba >= 0.5).astype(int)
  table = compute_lift_table(y_true, y_pred, proba, n=10)
  assert len(table) > 0
  prevalence = y_true.mean()
  expected = (table['target_count'] / table['decile_count']) / prevalence
  assert np.allclose(table['lift'].to_numpy(), expected.to_numpy())


# --- ML-15: single 1-based tiling, n>len warns ---

def test_assign_ntile_warns_when_tiles_exceed_rows():
  '''Centile request on 10 rows warns and caps tiles.'''
  from forecast_ml_framework.pipelines.modeling.ntile import assign_ntile
  with pytest.warns(UserWarning, match='tiles'):
    tiles = assign_ntile(np.arange(10.0), 'centile', 100)
  assert tiles.max() <= 10
  assert tiles.min() == 1


def test_deciles_are_one_based():
  '''evaluate._deciles labels start at 1, matching assign_ntile.'''
  from forecast_ml_framework.pipelines.modeling.evaluate import _deciles
  rng = np.random.RandomState(0)
  proba = rng.rand(200)
  labels = _deciles(np.zeros(200, dtype=int), np.zeros(200, dtype=int),
                    np.column_stack([1 - proba, proba]),
                    'classification')
  assert int(np.min(labels)) == 1
  assert int(np.max(labels)) == 10


# --- ML-18: promotion direction map + signed baselines ---

def test_logloss_improvement_promotes_classification():
  '''Lower-better log-loss improvement promotes in classification.'''
  from forecast_ml_framework.pipelines.modeling.validation import (
    should_promote,
  )
  promote, dev = should_promote(0.4, 0.5, 'classification', 'log_loss')
  assert promote is True
  assert dev == pytest.approx(-20.0)


def test_signed_kappa_baseline_promotes():
  '''Kappa from -0.1 to 0.2 (signed baseline) promotes.'''
  from forecast_ml_framework.pipelines.modeling.validation import (
    should_promote,
  )
  promote, _ = should_promote(0.2, -0.1, 'classification', 'cohen_kappa')
  assert promote is True


# --- ML-23: unknown / mismatched model_type raises ---

def test_unknown_model_type_raises():
  '''SklearnAdapter with an unknown model_type raises ValueError.'''
  from forecast_ml_framework.pipelines.modeling.adapters import (
    SklearnAdapter,
  )
  frame = _frame(20)
  with pytest.raises(ValueError, match='model_type'):
    SklearnAdapter({'model_type': 'bogus'}).fit(
        frame[['f1']], frame['target'])


def test_regression_with_random_forest_classifier_raises():
  '''Regression task with random_forest classifier raises.'''
  from forecast_ml_framework.pipelines.modeling.adapters import (
    SklearnAdapter,
  )
  frame = _frame(20)
  with pytest.raises(ValueError, match='regression'):
    SklearnAdapter({'model_type': 'random_forest',
                    'task_type': 'regression'}).fit(
        frame[['f1']], frame['target'].astype(float))
