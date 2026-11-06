#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Registry of allowlisted loss callables.

Only exact dotted names in this registry may be resolved from a
string ``loss_function``. Arbitrary callables from allowlisted
top-level modules (for example ``forecast_ml_framework.cli`` or
``sklearn.metrics.accuracy_score``) are rejected, so config
strings cannot reach non-loss code.

Standard losses are the scikit-learn metric losses below plus
the custom losses defined in this module. Trusted code may add
a custom loss by calling :func:`register_loss` with a dotted
name under ``forecast_ml_framework.losses``; config strings
themselves can never register a new loss.
'''

import numpy as np

#: Dotted names of standard scikit-learn metric losses.
_SKLEARN_LOSS_NAMES = (
  'mean_squared_error',
  'mean_absolute_error',
  'mean_squared_log_error',
  'root_mean_squared_error',
  'root_mean_squared_log_error',
  'median_absolute_error',
  'mean_absolute_percentage_error',
  'log_loss',
  'hinge_loss',
  'brier_score_loss',
  'mean_poisson_deviance',
  'mean_gamma_deviance',
  'mean_tweedie_deviance',
  'mean_pinball_loss',
  'zero_one_loss',
)


def huber_loss(y_true, y_pred, delta=1.0):
  '''Huber loss between ``y_true`` and ``y_pred`` (mean).

  Args:
    y_true: True values.
    y_pred: Predicted values.
    delta: Threshold where loss switches to linear.

  Returns:
    Mean Huber loss as a float.
  '''
  residuals = np.asarray(y_true, dtype=float) - np.asarray(
      y_pred, dtype=float)
  magnitude = np.abs(residuals)
  quadratic = magnitude <= delta
  loss = np.where(
      quadratic, 0.5 * residuals ** 2,
      delta * (magnitude - 0.5 * delta))
  return float(np.mean(loss))


def quantile_loss(y_true, y_pred, quantile=0.5):
  '''Pinball (quantile) loss between ``y_true`` and ``y_pred``.

  Args:
    y_true: True values.
    y_pred: Predicted values.
    quantile: Quantile in (0, 1).

  Returns:
    Mean pinball loss as a float.
  '''
  residuals = np.asarray(y_true, dtype=float) - np.asarray(
      y_pred, dtype=float)
  return float(np.mean(np.where(
      residuals >= 0, quantile * residuals,
      (quantile - 1) * residuals)))


#: Alias matching the pinball-loss name used by some libraries.
pinball_loss = quantile_loss


def _build_registry():
  '''Build the exact dotted-name -> callable loss registry.'''
  registry = {}
  try:
    from sklearn import metrics
  except ImportError:
    metrics = None
  if metrics is not None:
    for name in _SKLEARN_LOSS_NAMES:
      func = getattr(metrics, name, None)
      if callable(func):
        registry[f'sklearn.metrics.{name}'] = func
  registry['forecast_ml_framework.losses.huber_loss'] = huber_loss
  registry['forecast_ml_framework.losses.quantile_loss'] = quantile_loss
  registry['forecast_ml_framework.losses.pinball_loss'] = pinball_loss
  return registry


#: Exact allowlist of resolvable loss callables, by dotted name.
LOSS_CALLABLES = _build_registry()


def allowed_loss_names():
  '''Return the sorted exact dotted names that may be resolved.'''
  return sorted(LOSS_CALLABLES)


def register_loss(dotted_name, func):
  '''Register a trusted custom loss under this module.

  Args:
    dotted_name: Dotted name that must start with
      ``forecast_ml_framework.losses.`` and otherwise be a valid
      identifier.
    func: The callable loss to register.

  Raises:
    ValueError: If the name is not under
      ``forecast_ml_framework.losses``, is not callable, or is
      already registered. Duplicate registration always raises,
      even when re-registering the same callable.
  '''
  prefix = 'forecast_ml_framework.losses.'
  if (not isinstance(dotted_name, str)
          or not dotted_name.startswith(prefix)
          or not dotted_name[len(prefix):].isidentifier()):
    raise ValueError(
      f'Cannot register loss {dotted_name!r}: name must be '
      f"'{prefix}<identifier>'.")
  if not callable(func):
    raise ValueError(
      f'Cannot register loss {dotted_name!r}: not callable.')
  if dotted_name in LOSS_CALLABLES:
    raise ValueError(
      f'Cannot register loss {dotted_name!r}: already registered.')
  LOSS_CALLABLES[dotted_name] = func


def get_loss_callable(dotted_name):
  '''Return the allowlisted loss callable for ``dotted_name``.

  Args:
    dotted_name: Exact dotted name, e.g.
      ``'sklearn.metrics.mean_absolute_error'``.

  Returns:
    The registered callable.

  Raises:
    ValueError: If ``dotted_name`` is not in the exact
      allowlist.
  '''
  func = LOSS_CALLABLES.get(dotted_name)
  if func is None:
    raise ValueError(
      f'loss_function {dotted_name!r} is not allowlisted: it is '
      'not an exact registered loss callable. Valid loss '
      f'callables: {allowed_loss_names()}.')
  return func
