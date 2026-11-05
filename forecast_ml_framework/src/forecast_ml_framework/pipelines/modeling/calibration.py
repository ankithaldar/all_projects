#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Calibration node (classification only).'''

from forecast_ml_framework.pipelines.modeling.adapters import (
  SklearnAdapter,
  resolve_handle,
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


def calibrate_model(model, x_train, y_train, params):
  '''Calibrate a fitted classification model.

  Args:
    model: Fitted adapter (or sklearn estimator).
    x_train: Training features for calibration CV.
    y_train: Training target.
    params: modeling_params dict with calibration
      {method: sigmoid|isotonic|betacal|none, cv: int}.

  Returns:
    Tuple (calibrated model, meta dict).
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
      resolve_handle(handle)
  if params.get('task_type') != 'classification' or method == 'none':
    return model, {'calibrated': False, 'method': method}
  base = getattr(model, 'model', model)
  if method == 'betacal':
    try:
      from betacal import BetaCalibration
      fitted = BetaCalibration()
      fitted.fit(base.predict_proba(x_train)[:, 1], y_train)
      wrapped = _BetaCalWrapper(base, fitted)
      return _wrap(wrapped), {'calibrated': True,
                              'method': 'betacal'}
    except ImportError:
      method = 'sigmoid'
  from sklearn.calibration import CalibratedClassifierCV
  cv = cfg.get('cv', 5)
  calibrator_cls = calibrator_cls or CalibratedClassifierCV
  calibrated = calibrator_cls(base, method=method, cv=cv)
  calibrated.fit(x_train, y_train)
  return _wrap(calibrated), {'calibrated': True, 'method': method}
