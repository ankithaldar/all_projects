#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Hyperparameter tuning via sklearn search strategies.'''
# pylint: disable=unused-import,import-outside-toplevel
# pylint: disable=broad-exception-caught,unexpected-keyword-arg

import inspect
import logging
import os

import numpy as np
from sklearn.base import BaseEstimator

_MAX_CANDIDATES = 1000
_MAX_N_ITER = 1000

LOG = logging.getLogger(__name__)

#: Exact method map: normalised name -> search class name.
_METHODS = {
  'grid': 'grid',
  'grid_search': 'grid',
  'gridsearchcv': 'grid',
  'random': 'random',
  'random_search': 'random',
  'randomized': 'random',
  'randomized_search': 'random',
  'randomizedsearchcv': 'random',
  'halving': 'halving',
  'halving_grid': 'halving',
  'halvinggridsearchcv': 'halving',
  'halving_random': 'halving_random',
  'halving_random_search': 'halving_random',
  'halvingrandomsearchcv': 'halving_random',
}

_GRID_KEYS = ('param_grid', 'param_distributions', 'grid', 'params')
_TUNING_KEYS = ('tuning', 'tuning_params', 'tuning_config',
                'hyperparameter_tuning')


class _FoldBalancedEstimator(BaseEstimator):
  '''Estimator wrapper recomputing balance weights per fit.

  ``fit(x, y)`` clones the wrapped estimator, computes
  imbalance statistics from the ``y`` it is actually fitted
  on (one CV fold inside ``GridSearchCV``, the full data on
  the final refit) and injects ``scale_pos_weight`` /
  ``class_weight`` / ``class_weights`` into the clone before
  fitting -- so folds with different class ratios get
  different weights instead of one full-train constant.
  Each fit appends its ``scale_pos_weight`` to the shared
  class-level ``balance_fit_log`` audit list (clones made by
  the search share the class, so the log aggregates across
  folds and candidates). Parameter grids tune the wrapped
  estimator via ``estimator__<param>`` keys (rewritten by
  ``tune_model`` when balance is on).
  '''

  #: Audit log of per-fit scale_pos_weight values (shared).
  balance_fit_log = []

  def __init__(self, estimator=None, balance=True):
    '''Store the wrapped estimator and balance flag.

    Args:
      estimator: Unfitted sklearn-style estimator to wrap.
      balance: When True (default), recompute balance
        weights from each fit's ``y``.
    '''
    self.estimator = estimator
    self.balance = balance

  def _balance_param(self, estimator, info):
    '''Return (param_name, value) to inject, or (None, None).'''
    try:
      signature = inspect.signature(type(estimator))
    except (TypeError, ValueError):
      return None, None
    names = set(signature.parameters)
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values())
    if 'scale_pos_weight' in names or has_var_kw:
      return 'scale_pos_weight', info['scale_pos_weight']
    if 'class_weight' in names:
      return 'class_weight', info['class_weights']
    if 'class_weights' in names:
      return 'class_weights', info['class_weights']
    return None, None

  def fit(self, x, y):
    '''Clone, rebalance on this fit's y, and fit the clone.'''
    from sklearn.base import clone
    from forecast_ml_framework.pipelines.modeling.imbalance import (
      compute_imbalance,
    )
    fitted = clone(self.estimator)
    if self.balance:
      info = compute_imbalance(y, 'classification')
      name, value = self._balance_param(fitted, info)
      if name is not None and name in fitted.get_params():
        fitted.set_params(**{name: value})
      type(self).balance_fit_log.append(
          float(info['scale_pos_weight']))
    fitted.fit(x, y)
    self.estimator_ = fitted
    self.classes_ = getattr(fitted, 'classes_', None)
    return self

  def predict(self, x):
    '''Predict with the fitted clone.'''
    return self.estimator_.predict(x)

  def predict_proba(self, x):
    '''Predict probas with the fitted clone.'''
    return self.estimator_.predict_proba(x)

  def score(self, x, y):
    '''Score with the fitted clone (search default scoring).'''
    return self.estimator_.score(x, y)

  def __sklearn_tags__(self):
    '''Delegate sklearn tags to the wrapped estimator.

    Without this, ``is_classifier(wrapper)`` is False and
    ``GridSearchCV(cv=int)`` degrades StratifiedKFold to
    KFold for the balanced path, which can produce
    single-class folds, FitFailedWarnings and a nan
    best_score on ordered imbalanced data. Falls back to
    the generic tags when there is no wrapped estimator
    with tags (e.g. estimator is None).
    '''
    estimator = self.__dict__.get('estimator')
    if estimator is not None:
      tags = getattr(estimator, '__sklearn_tags__', None)
      if callable(tags):
        return tags()
    return super().__sklearn_tags__()

  def __getattr__(self, name):
    '''Delegate unknown attributes to the fitted clone.'''
    fitted = self.__dict__.get('estimator_')
    if fitted is None:
      raise AttributeError(name)
    return getattr(fitted, name)


def _min_class_count(y):
  '''Return the smallest class count in y, or len(y).'''
  try:
    _, counts = np.unique(np.asarray(y), return_counts=True)
  except Exception:  # noqa: BLE001
    return len(y)
  if len(counts) == 0:
    return 0
  return int(counts.min())


def _resolve_params(params):
  '''Unwrap a nested tuning dict if present.'''
  params = dict(params or {})
  for key in _TUNING_KEYS:
    nested = params.get(key)
    if isinstance(nested, dict):
      merged = dict(nested)
      for extra, value in params.items():
        if extra not in _TUNING_KEYS:
          merged.setdefault(extra, value)
      return merged
  return params


def _param_grid(params, method):
  '''Return and validate the parameter grid/distributions.'''
  grid = None
  for key in _GRID_KEYS:
    if key in params:
      grid = params[key]
      break
  if not isinstance(grid, dict) or not grid:
    raise ValueError(
      f'param_grid must be a non-empty dict of non-empty lists, '
      f'got {grid!r} for tuning method {method!r}')
  for name, values in grid.items():
    if isinstance(values, (list, tuple)):
      if len(values) == 0:
        raise ValueError(
          f'param_grid[{name!r}] must be a non-empty list, '
          f'got an empty list')
    elif method in ('grid', 'halving'):
      raise ValueError(
        f'param_grid[{name!r}] must be a non-empty list, '
        f'got {values!r}')
  total = 1
  for values in grid.values():
    if isinstance(values, (list, tuple)):
      total *= len(values)
      if total > _MAX_CANDIDATES:
        raise ValueError(
          f'param_grid has {total} total candidates, exceeding '
          f'the cap of {_MAX_CANDIDATES}: reduce list lengths '
          f'(product of list lengths must be <= {_MAX_CANDIDATES})')
  return grid


def _is_regression(estimator, params):
  '''Return True for regression estimators / explicit task.'''
  for key in ('task_type', 'task', 'task_kind'):
    raw = params.get(key)
    if raw is not None:
      name = str(raw).strip().lower()
      if name.startswith('reg'):
        return True
      if name.startswith('class') or name in ('cls', 'binary'):
        return False
  try:
    from sklearn.base import is_regressor
    if is_regressor(estimator):
      return True
  except (ImportError, AttributeError, TypeError, ValueError) as exc:
    LOG.debug('is_regressor probe failed; using class-name check: %r', exc)
  name = type(estimator).__name__.lower()
  return 'regressor' in name


def _validate_n_iter(params):
  '''Validate optional n_iter is an int in [1, 1000].'''
  for key in ('n_iter', 'n_candidates'):
    if key not in params or params.get(key) is None:
      continue
    value = params[key]
    if isinstance(value, bool) or not isinstance(value, int):
      raise ValueError(f'{key} must be an integer, got {value!r}')
    if value < 1 or value > _MAX_N_ITER:
      raise ValueError(
        f'{key} must be in [1, {_MAX_N_ITER}], got {value}')


def _validate_n_jobs(n_jobs):
  '''Validate n_jobs is an int in [1, cpu_count].'''
  if isinstance(n_jobs, bool) or not isinstance(n_jobs, int):
    raise ValueError(f'n_jobs must be an integer, got {n_jobs!r}')
  cpu = os.cpu_count() or 1
  if n_jobs < 1 or n_jobs > cpu:
    raise ValueError(
      f'n_jobs must be in [1, {cpu}], got {n_jobs}')
  return n_jobs


def _validate_cv(cv, y, is_regression=False):
  '''Validate cv against min class (classification) or n (regression).'''
  if isinstance(cv, bool) or not isinstance(cv, int):
    raise ValueError(f'cv must be an integer, got {cv!r}')
  if cv < 2:
    raise ValueError(f'cv must be at least 2, got {cv}')
  if is_regression:
    n = len(np.asarray(y))
    if n < 2:
      raise ValueError(
        f'cv={cv} requires at least 2 samples, got n_samples={n}')
    if cv > n:
      raise ValueError(
        f'cv={cv} exceeds n_samples={n}: cv must be in [2, {n}]')
    return cv
  minimum = _min_class_count(y)
  if minimum and cv > minimum:
    raise ValueError(
      f'cv={cv} exceeds the smallest class count '
      f'{minimum}: cv must be in [2, {minimum}]')
  return cv


def tune_model(estimator, x, y, params):
  '''Tune an estimator with grid/random/halving search.

  Args:
    estimator: Unfitted sklearn-style estimator.
    x: Training features.
    y: Training target.
    params: Tuning dict with method (grid|random|halving|
      halving_random; halving_random maps n_iter to the
      HalvingRandomSearchCV n_candidates kwarg),
      param_grid (dict of non-empty lists; product of list
      lengths must be <= 1000), cv (int in [2, min class count]
      for classification, or [2, n_samples] for regression),
      scoring (string), n_jobs (int in [1, cpu_count],
      default 1), and optional n_iter (int in [1, 1000]) /
      random_state. Classification targets must have >= 2
      classes (single-class raises ValueError pre-fit).

  Returns:
    Tuple (best fitted estimator, best_params dict, report
    dict with method/best_score/best_params/cv/scoring/n_jobs).

  Raises:
    ValueError: If method is unknown, param_grid is invalid,
      cv is out of range, or scoring is not a string.
  '''
  params = _resolve_params(params)
  raw_method = params.get('method', params.get('search_type',
                      params.get('tuning_method', 'grid')))
  method_key = str(raw_method or '').strip().lower()
  if method_key not in _METHODS:
    raise ValueError(
      f'Unknown tuning method {raw_method!r}: method must be '
      f'one of {sorted(set(_METHODS.values()))} (exact map)')
  method = _METHODS[method_key]
  grid = _param_grid(params, method)
  _validate_n_iter(params)
  is_regression = _is_regression(estimator, params)
  try:
    y_array = np.asarray(y)
  except Exception:  # noqa: BLE001
    raise ValueError(
      'y must be 1-D: 2-D / list-of-lists targets are not supported'
    ) from None
  if y_array.ndim != 1:
    raise ValueError(
      f'y must be 1-D, got shape {y_array.shape}: '
      '2-D / list-of-lists targets are not supported')
  y_values = y_array.ravel()
  if y_values.size == 0:
    raise ValueError('tune_model requires a non-empty y')
  if not is_regression and len(np.unique(y_values)) < 2:
    raise ValueError(
      'tune_model requires at least 2 classes in y for '
      f'classification, got {len(np.unique(y_values))} class(es): '
      'single-class targets cannot be tuned')
  if is_regression and y_values.size < 2:
    raise ValueError(
      'tune_model requires at least 2 samples for regression, '
      f'got {y_values.size}')
  scoring = params.get('scoring')
  if scoring is not None and not isinstance(scoring, str):
    raise ValueError(
      f'scoring must be a string, got {scoring!r} '
      f'({type(scoring).__name__})')
  if scoring is not None and not scoring.strip():
    raise ValueError('scoring must be a non-empty string')
  cv = params.get('cv', 3)
  cv = _validate_cv(cv, y, is_regression=is_regression)
  n_jobs = _validate_n_jobs(params.get('n_jobs', 1))
  balance = bool(params.get('balance'))
  if balance:
    # Recompute balance weights inside each CV fold instead
    # of reusing one full-train constant baked into the
    # estimator: wrap it and route grid keys to the wrapped
    # estimator (estimator__<param>).
    _FoldBalancedEstimator.balance_fit_log = []
    estimator = _FoldBalancedEstimator(estimator, balance=True)
    grid = {f'estimator__{key}': values
            for key, values in grid.items()}
  random_state = params.get('random_state', params.get('seed'))
  common = {'cv': cv, 'scoring': scoring, 'n_jobs': n_jobs}
  if method == 'grid':
    from sklearn.model_selection import GridSearchCV
    search = GridSearchCV(estimator, grid, **common)
  elif method == 'random':
    from sklearn.model_selection import RandomizedSearchCV
    n_iter = params.get('n_iter', 10)
    search = RandomizedSearchCV(
      estimator, grid, n_iter=n_iter,
      random_state=random_state, **common)
  elif method == 'halving':
    try:
      from sklearn.experimental import enable_halving_search_cv  # noqa
    except ImportError:
      # Safe: older sklearn without the halving export;
      # HalvingGridSearchCV import below is the real guard.
      pass
    from sklearn.model_selection import HalvingGridSearchCV
    search = HalvingGridSearchCV(
      estimator, grid, random_state=random_state, **common)
  else:
    try:
      from sklearn.experimental import enable_halving_search_cv  # noqa
    except ImportError:
      # Safe: older sklearn without the halving export;
      # HalvingRandomSearchCV import below is the real guard.
      pass
    from sklearn.model_selection import HalvingRandomSearchCV
    # HalvingRandomSearchCV has no n_iter kwarg; it takes
    # n_candidates (alias accepted: n_candidates wins, then
    # n_iter, default 10).
    n_candidates = params.get('n_candidates',
                              params.get('n_iter', 10))
    search = HalvingRandomSearchCV(
      estimator, grid, n_candidates=n_candidates,
      random_state=random_state, **common)
  search.fit(x, y)
  best_params = dict(search.best_params_)
  report = {
    'method': method,
    'best_score': float(search.best_score_)
    if search.best_score_ is not None else None,
    'best_params': best_params,
    'best_index': int(search.best_index_),
    'cv': cv,
    'scoring': scoring,
    'n_jobs': n_jobs,
    'n_candidates': len(search.cv_results_['params'])
    if hasattr(search, 'cv_results_') else None,
  }
  if method == 'random':
    report['n_iter'] = params.get('n_iter', 10)
  if method == 'halving_random':
    report['n_candidates'] = params.get(
      'n_candidates', params.get('n_iter', 10))
    report['n_iter'] = report['n_candidates']
  if balance:
    # Per-fit weights seen during the search (CV folds plus
    # the final full-data refit); folds with different class
    # ratios show different values by construction.
    report['balance'] = True
    report['balance_per_fold_weights'] = list(
        _FoldBalancedEstimator.balance_fit_log)
  return search.best_estimator_, best_params, report
