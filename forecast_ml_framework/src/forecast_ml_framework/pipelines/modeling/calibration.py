#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Calibration node (classification only).'''

import numpy as np
import pandas as pd

from forecast_ml_framework.pipelines.modeling.adapters import (
  SklearnAdapter,
  resolve_handle,
)
from forecast_ml_framework.pipelines.modeling.ntile import (
  normalize_task_kind,
)


class _BetaCalWrapper:
  '''Adapt betacal (proba in/out) to a sklearn-style API.'''

  def __init__(self, base, calibrator):
    '''Store the base model and the fitted calibrator.'''
    self.base = base
    self.calibrator = calibrator

  def predict_proba(self, x):
    '''Map features through base proba then calibrator.'''
    return self.calibrator.predict_proba(self.base.predict_proba(x))

  def predict(self, x):
    '''Predict the most probable class.'''
    return self.base.classes_[self.predict_proba(x).argmax(axis=1)]


def _wrap(estimator):
  '''Wrap a fitted estimator in a sklearn-compatible adapter.'''
  adapter = SklearnAdapter({})
  adapter.model = estimator
  return adapter


def _validate_rate(name, value):
  '''Validate a prevalence rate is finite and in [0, 1].'''
  if isinstance(value, bool) or value is None:
    raise ValueError(
      f'{name} must be a finite number in [0, 1], got {value!r}')
  try:
    rate = float(value)
  except (TypeError, ValueError) as exc:
    raise ValueError(
      f'{name} must be a finite number in [0, 1], '
      f'got {value!r}') from exc
  if not np.isfinite(rate) or rate < 0.0 or rate > 1.0:
    raise ValueError(
      f'{name} must be a finite number in [0, 1], got {value!r}')
  return rate


def _validate_calibrate_cv(cv, y_train):
  '''Validate calibration cv is an int in [2, min class count].'''
  if isinstance(cv, bool) or not isinstance(cv, int):
    raise ValueError(f'cv must be an integer, got {cv!r}')
  if cv < 2:
    raise ValueError(f'cv must be at least 2, got {cv}')
  values = np.asarray(y_train).ravel()
  if values.size == 0:
    raise ValueError('calibrate_model requires a non-empty y_train')
  _, counts = np.unique(values, return_counts=True)
  if len(counts) < 2:
    raise ValueError(
      'calibrate_model requires at least 2 classes in y_train, '
      f'got {len(counts)} class(es)')
  minimum = int(counts.min())
  if cv > minimum:
    raise ValueError(
      f'cv={cv} exceeds the smallest class count '
      f'{minimum}: cv must be in [2, {minimum}]')
  return cv


_CALIBRATION_METHODS = ('sigmoid', 'isotonic', 'betacal')


def calibrate_probability(proba, real_rate, sample_rate):
  '''Adjust probabilities for prior (prevalence) shift.

  Pure Saerens-style odds adjustment: the odds of each
  probability are rescaled by
  ``(real_rate / (1 - real_rate)) /
  (sample_rate / (1 - sample_rate))`` and mapped back to
  a probability, so the result is monotonic in ``proba``
  and clipped to [0, 1].

  Args:
    proba: Positive-class probabilities (array-like or
      Series; a two-column array adjusts column 1 and
      renormalises column 0).
    real_rate: Positive rate in the real population.
    sample_rate: Positive rate in the sample the model
      was trained on.

  Returns:
    Calibrated probabilities of the same type/shape as
    the input (Series in, Series out). Identity when
    ``real_rate == sample_rate``.
  Raises:
    ValueError: If real_rate/sample_rate is not finite or outside
      [0, 1], proba is not finite or outside [0, 1], or proba
      is multiclass / 3-D (only 1-D, (n, 1) and (n, 2) binary
      proba are supported; other shapes would corrupt row
      sums instead of summing to 1).
  '''
  is_series = isinstance(proba, pd.Series)
  is_frame = isinstance(proba, pd.DataFrame)
  real = _validate_rate('real_rate', real_rate)
  sample = _validate_rate('sample_rate', sample_rate)
  try:
    values = np.asarray(proba, dtype=float)
  except (TypeError, ValueError) as exc:
    raise ValueError(
      f'proba must be numeric probabilities in [0, 1], '
      f'got {proba!r}') from exc
  if values.size and (not np.isfinite(values).all()
                      or values.min() < 0.0 or values.max() > 1.0):
    raise ValueError(
      'proba must be finite probabilities in [0, 1], '
      'got NaN/inf or out-of-range values')
  if values.ndim > 2 or (values.ndim == 2
                         and values.shape[1] > 2):
    raise ValueError(
      'proba must be 1-D, (n, 1), or (n, 2) binary '
      f'probabilities, got shape {values.shape}: multiclass '
      'proba (n, k>2) is not supported by '
      'calibrate_probability (row sums would not be 1)')
  if real == sample:
    out = values.copy()
  elif real <= 0.0:
    out = np.zeros_like(values)
    if values.ndim == 2 and values.shape[1] == 2:
      out[:, 0] = 1.0
  elif real >= 1.0:
    out = np.ones_like(values)
    if values.ndim == 2 and values.shape[1] == 2:
      out[:, 0] = 0.0
  else:
    eps = 1e-15
    sample_c = min(max(sample, eps), 1.0 - eps)
    factor = (real / (1.0 - real)) / (
      sample_c / (1.0 - sample_c))
    if values.ndim == 2 and values.shape[1] == 2:
      pos = np.clip(values[:, 1], 0.0, 1.0)
      odds = np.divide(pos, 1.0 - pos,
                       out=np.full_like(pos, np.inf),
                       where=pos < 1.0)
      with np.errstate(divide='ignore', invalid='ignore'):
        adjusted = (odds * factor) / (1.0 + odds * factor)
      adjusted = np.where(pos >= 1.0, 1.0, adjusted)
      adjusted = np.where(pos <= 0.0, 0.0, adjusted)
      out = np.empty_like(values)
      out[:, 1] = np.clip(adjusted, 0.0, 1.0)
      out[:, 0] = 1.0 - out[:, 1]
    else:
      pos = np.clip(values, 0.0, 1.0)
      odds = np.divide(pos, 1.0 - pos,
                       out=np.full_like(pos, np.inf),
                       where=pos < 1.0)
      with np.errstate(divide='ignore', invalid='ignore'):
        adjusted = (odds * factor) / (1.0 + odds * factor)
      adjusted = np.where(pos >= 1.0, 1.0, adjusted)
      adjusted = np.where(pos <= 0.0, 0.0, adjusted)
      out = np.clip(adjusted, 0.0, 1.0)
  if is_series:
    return pd.Series(out, index=proba.index, name=proba.name)
  if is_frame:
    return pd.DataFrame(out, index=proba.index, columns=proba.columns)
  if isinstance(proba, np.ndarray):
    return out
  return out


def _oof_proba(base, x_train, y_train, cv):
  '''Return out-of-fold positive-class probas for base.

  Fits a clone of ``base`` on each training fold and predicts
  the held-out fold, so the returned probas never come from
  rows the scoring clone was fitted on (unlike in-sample
  probas from the already-fitted ``base``).

  Args:
    base: Fitted sklearn-style estimator with predict_proba.
    x_train: Training features.
    y_train: Training target.
    cv: Number of stratified folds.

  Returns:
    1-D array of out-of-fold positive-class probabilities.

  Raises:
    ValueError: If base cannot be cloned for OOF fitting or
      its predict_proba is not binary (n, 2).
  '''
  import copy
  from sklearn.base import clone
  from sklearn.model_selection import StratifiedKFold
  y_values = np.asarray(y_train)
  splitter = StratifiedKFold(n_splits=cv, shuffle=True,
                             random_state=42)
  oof = np.zeros(len(y_values), dtype=float)
  for fit_idx, val_idx in splitter.split(x_train, y_values):
    try:
      fold_model = clone(base)
    except Exception:  # noqa: BLE001 - deepcopy fallback
      try:
        fold_model = copy.deepcopy(base)
      except Exception as exc:
        raise ValueError(
            'betacal calibration requires a cloneable base '
            f'estimator for out-of-fold probas: {exc}') from exc
    fold_model.fit(_take(x_train, fit_idx), y_values[fit_idx])
    proba = np.asarray(
        fold_model.predict_proba(_take(x_train, val_idx)), dtype=float)
    if proba.ndim != 2 or proba.shape[1] != 2:
      raise ValueError(
          'betacal calibration requires binary (n, 2) proba, '
          f'got shape {proba.shape}')
    oof[val_idx] = proba[:, 1]
  return oof


def _take(x, idx):
  '''Return rows ``idx`` of array-like or DataFrame ``x``.'''
  if hasattr(x, 'iloc'):
    return x.iloc[idx]
  return np.asarray(x)[idx]


def calibrate_model(model, x_train, y_train, params):
  '''Calibrate a fitted classification model.

  Args:
    model: Fitted adapter (or sklearn estimator).
    x_train: Training features for calibration CV.
    y_train: Training target.
    params: modeling_params dict with calibration
      {method: sigmoid|isotonic|betacal|none, cv: int}.
      Betacal (binary only) is fitted on out-of-fold probas
      from ``_oof_proba``; when the ``betacal`` library is
      missing it warns and falls back to sigmoid, recording
      both the requested and the used method in meta.

  Returns:
    Tuple (calibrated model, meta dict) with keys
    calibrated, method (used), requested_method, and
    fallback (True only on the betacal->sigmoid fallback).

  Raises:
    ValueError: If method is not one of sigmoid|isotonic|betacal
      (or 'none' for passthrough), cv is not an int in
      [2, smallest class count], y_train has < 2 classes, or
      betacal is requested for a non-binary target.
  '''
  params = params or {}
  cfg = params.get('calibration', {}) or {}
  method = cfg.get('method', 'sigmoid')
  handle = (params.get('calibration_handle')
            or cfg.get('handle') or cfg.get('calibration_handle'))
  calibrator_cls = None
  if isinstance(handle, str) and '.' in handle:
    if handle.endswith('BetaCalibration'):
      method = 'betacal'
    elif handle.endswith('CalibratedClassifierCV'):
      calibrator_cls = resolve_handle(handle)
    else:
      # Validation-only resolve: assigned to _ and not
      # used further. A handle that is neither
      # BetaCalibration nor CalibratedClassifierCV must still
      # be allowlisted (bogus handles raise here), but
      # calibration then falls through to the default
      # CalibratedClassifierCV below with calibrator_cls
      # left as None.
      _ = resolve_handle(handle)
  if (normalize_task_kind(params.get('task_type', 'classification'))
          != 'classification' or method == 'none'):
    return model, {'calibrated': False, 'method': method,
                   'requested_method': method, 'fallback': False}
  if not isinstance(method, str):
    raise ValueError(
      f'calibration method must be one of '
      f'{list(_CALIBRATION_METHODS)}, got {method!r}')
  method = method.strip().lower()
  if method not in _CALIBRATION_METHODS:
    raise ValueError(
      f'calibration method must be one of '
      f'{list(_CALIBRATION_METHODS)}, got {method!r}')
  cv = _validate_calibrate_cv(cfg.get('cv', 5), y_train)
  base = getattr(model, 'model', model)
  requested = method
  if method == 'betacal':
    classes = np.unique(np.asarray(y_train))
    if len(classes) != 2:
      raise ValueError(
          'betacal calibration requires a binary target, got '
          f'{len(classes)} classes: betacal only supports '
          'binary (n, 2) proba')
    try:
      from betacal import BetaCalibration
    except ImportError:
      import warnings
      warnings.warn(
          'betacal calibration requested but the betacal library '
          'is not installed; falling back to sigmoid calibration',
          UserWarning, stacklevel=2)
      method = 'sigmoid'
    else:
      fitted = BetaCalibration()
      fitted.fit(_oof_proba(base, x_train, y_train, cv), y_train)
      wrapped = _BetaCalWrapper(base, fitted)
      return _wrap(wrapped), {'calibrated': True,
                              'method': 'betacal',
                              'requested_method': 'betacal',
                              'fallback': False}
  from sklearn.calibration import CalibratedClassifierCV
  calibrator_cls = calibrator_cls or CalibratedClassifierCV
  calibrated = calibrator_cls(base, method=method, cv=cv)
  calibrated.fit(x_train, y_train)
  return _wrap(calibrated), {'calibrated': True, 'method': method,
                             'requested_method': requested,
                             'fallback': requested != method}
