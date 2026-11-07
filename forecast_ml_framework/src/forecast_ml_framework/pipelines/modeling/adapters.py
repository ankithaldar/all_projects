#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Model adapter ABC, zoo adapters and the train node.'''

import inspect
import pickle
from abc import ABC, abstractmethod

import numpy as np

from forecast_ml_framework.pipelines.modeling.ntile import (
  normalize_task_kind,
)

#: Max distinct models fitted by train_models in one call.
MAX_MODELS = 10

#: Param keys that hold dotted importable handles, not model kwargs.
HANDLE_KEYS = ('model_handle', 'calibration_handle',
               'tuning_handle', 'handle')

#: Native estimator kwarg that carries the loss, per adapter.
LOSS_KWARGS = {
  'sklearn': 'loss',
  'xgboost': 'objective',
  'lightgbm': 'objective',
  'catboost': 'loss_function',
  'h2o_automl': 'loss_function',
  'h2o': 'loss_function',
  'handle': 'loss',
}

#: Loss registry: adapter -> normalised name -> native value.
#: Keys are looked up case-insensitively; native values are
#: accepted as keys too (they map to themselves).
LOSS_REGISTRY = {
  'xgboost': {
    'binary:logistic': 'binary:logistic',
    'logloss': 'binary:logistic',
    'logistic': 'binary:logistic',
    'binary_logistic': 'binary:logistic',
    'binary': 'binary:logistic',
    'reg:squarederror': 'reg:squarederror',
    'squared_error': 'reg:squarederror',
    'mse': 'reg:squarederror',
    'rmse': 'reg:squarederror',
    'regression': 'reg:squarederror',
    'reg:gamma': 'reg:gamma',
    'gamma': 'reg:gamma',
    'reg:tweedie': 'reg:tweedie',
    'tweedie': 'reg:tweedie',
    'reg:absoluteerror': 'reg:absoluteerror',
    'absolute_error': 'reg:absoluteerror',
    'mae': 'reg:absoluteerror',
    'count:poisson': 'count:poisson',
    'poisson': 'count:poisson',
    'reg:logistic': 'reg:logistic',
    'multi:softprob': 'multi:softprob',
    'multi:softmax': 'multi:softmax',
    'reg:pseudohubererror': 'reg:pseudohubererror',
    'huber': 'reg:pseudohubererror',
  },
  'lightgbm': {
    'binary': 'binary',
    'logloss': 'binary',
    'binary_crossentropy': 'binary',
    'regression': 'regression',
    'squared_error': 'regression',
    'mse': 'regression',
    'rmse': 'regression',
    'regression_l1': 'regression_l1',
    'absolute_error': 'regression_l1',
    'mae': 'regression_l1',
    'huber': 'huber',
    'fair': 'fair',
    'poisson': 'poisson',
    'gamma': 'gamma',
    'tweedie': 'tweedie',
    'quantile': 'quantile',
    'multiclass': 'multiclass',
    'multiclassova': 'multiclassova',
    'xentropy': 'xentropy',
    'lambdarank': 'lambdarank',
    'rank_xendcg': 'rank_xendcg',
  },
  'catboost': {
    'logloss': 'Logloss',
    'log_loss': 'Logloss',
    'binary': 'Logloss',
    'rmse': 'RMSE',
    'squared_error': 'RMSE',
    'mse': 'RMSE',
    'mae': 'MAE',
    'absolute_error': 'MAE',
    'poisson': 'Poisson',
    'gamma': 'Gamma',
    'tweedie': 'Tweedie',
    'quantile': 'Quantile',
    'huber': 'Huber',
    'fair': 'Fair',
    'hinge': 'Hinge',
    'crossentropy': 'CrossEntropy',
  },
  'sklearn': {
    'log_loss': 'log_loss',
    'logloss': 'log_loss',
    'squared_error': 'squared_error',
    'mse': 'squared_error',
    'rmse': 'squared_error',
    'absolute_error': 'absolute_error',
    'mae': 'absolute_error',
    'huber': 'huber',
    'quantile': 'quantile',
    'epsilon_insensitive': 'epsilon_insensitive',
    'poisson': 'poisson',
    'gamma': 'gamma',
    'tweedie': 'tweedie',
    'hinge': 'hinge',
    'squared_hinge': 'squared_hinge',
    'modified_huber': 'modified_huber',
    'perceptron': 'perceptron',
    'exponential': 'exponential',
  },
  'h2o_automl': {
    'logloss': 'logloss',
    'log_loss': 'logloss',
    'binary': 'logloss',
    'classification': 'logloss',
    'squared_error': 'squared_error',
    'mse': 'squared_error',
    'rmse': 'squared_error',
    'regression': 'squared_error',
    'absolute_error': 'absolute_error',
    'mae': 'absolute_error',
    'poisson': 'poisson',
    'gamma': 'gamma',
    'tweedie': 'tweedie',
    'huber': 'huber',
    'quantile': 'quantile',
  },
}


def resolve_handle(handle):
  '''Resolve a dotted handle under the scoring allowlist.

  Delegates to
  ``core.handle.resolve_handle_strict`` so
  modeling handles share its allowlist (sklearn / xgboost /
  lightgbm / catboost / h2o / forecast_ml_framework) and
  estimator check (class with ``fit`` and ``predict``).
  Arbitrary importables such as ``os.system``,
  ``subprocess.Popen`` or ``builtins.eval`` are rejected,
  preventing RCE via ``HandleAdapter.fit`` /
  ``calibration_handle``.

  Args:
    handle: Dotted importable string, e.g.
      'sklearn.linear_model.LogisticRegression'.

  Returns:
    The allowlisted estimator-like class named by handle.

  Raises:
    ValueError: If handle is not allowlisted, cannot be
      resolved, or is not an estimator-like class.
  '''
  from forecast_ml_framework.core.handle import (
    resolve_handle_strict as _resolve_handle_strict,
  )
  if not isinstance(handle, str) or '.' not in handle:
    raise ValueError(
      f'Cannot resolve handle {handle!r}: expected a dotted '
      'importable string such as '
      "'sklearn.linear_model.LogisticRegression' for "
      'model_handle / calibration_handle / tuning_handle, or a '
      'short adapter name (sklearn, xgboost, lightgbm, catboost, '
      'h2o_automl).')
  try:
    return _resolve_handle_strict(handle)
  except ValueError as exc:
    raise ValueError(
      f'Cannot resolve handle {handle!r}: {exc}. Check '
      'model_handle / calibration_handle / tuning_handle is an '
      'allowlisted estimator handle.') from exc


def _resolve_loss_callable(loss):
  '''Resolve a dotted custom loss by exact allowlist match.

  Only exact dotted names registered in
  ``forecast_ml_framework.losses`` (standard scikit-learn metric
  losses and registered custom losses under
  ``forecast_ml_framework.losses``) are accepted. A top-level
  module allowlist is not sufficient: arbitrary callables from
  allowlisted modules such as
  ``forecast_ml_framework.cli._run_kedro`` or
  ``forecast_ml_framework.core.datasets.SQLQueryDataSet`` must
  not resolve as losses.

  Args:
    loss: Dotted importable string naming a callable.

  Returns:
    The resolved callable.

  Raises:
    ValueError: If ``loss`` is not an exact registered loss
      callable.
  '''
  from forecast_ml_framework.losses import get_loss_callable
  try:
    return get_loss_callable(loss)
  except ValueError as exc:
    raise ValueError(
      f'loss_function {loss!r} is not allowlisted: {exc}') from exc


def _canonical_adapter(adapter_name):
  '''Return the registry key for an adapter or handle name.

  Args:
    adapter_name: Short adapter name or dotted handle.

  Returns:
    Canonical adapter key, or 'handle' when unknown.
  '''
  if not isinstance(adapter_name, str):
    return 'handle'
  root = adapter_name.split('.')[0].strip().lower()
  if root == 'h2o':
    return 'h2o_automl'
  if root in LOSS_REGISTRY:
    return root
  return 'handle'


def get_loss(adapter_name, loss):
  '''Map a plain loss string to an adapter's native value.

  Args:
    adapter_name: Short adapter name ('sklearn',
      'xgboost', 'lightgbm', 'catboost', 'h2o_automl') or
      a dotted handle whose module implies the adapter.
    loss: Plain loss string (case-insensitive, e.g.
      'logloss', 'squared_error', 'poisson'), an already
      native value (e.g. 'binary:logistic', 'Logloss'), a
      callable, or a dotted importable string resolving
      to a callable (e.g. 'forecast_ml_framework.losses.'
      'huber_loss' / 'sklearn.metrics.mean_absolute_error').

  Returns:
    Native loss value for the adapter: a string, or the
    resolved callable for dotted/callable custom losses.

  Raises:
    ValueError: If the loss is unknown for the adapter, the
      dotted string cannot be resolved, or it resolves to
      a non-callable. The message lists valid losses.
  '''
  if callable(loss):
    return loss
  if not isinstance(loss, str) or not loss.strip():
    raise ValueError(
      f'Unknown loss_function {loss!r} for adapter '
      f'{adapter_name!r}: expected a plain string such as '
      "'logloss' / 'squared_error', or a dotted callable "
      "string such as 'forecast_ml_framework.losses.'"
      "huber_loss' / 'sklearn.metrics.'"
      "'mean_absolute_error'.")
  if '.' in loss:
    try:
      return _resolve_loss_callable(loss)
    except ValueError as exc:
      raise ValueError(
        f'Cannot resolve loss_function {loss!r} for adapter '
        f'{adapter_name!r}: {exc}') from exc
  canonical = _canonical_adapter(adapter_name)
  registry = LOSS_REGISTRY.get(canonical)
  if registry is None:
    raise ValueError(
      f'Unknown adapter {adapter_name!r} for loss_function '
      f'{loss!r}. Known adapters: {sorted(LOSS_REGISTRY)}')
  key = loss.strip().lower()
  if key not in registry:
    raise ValueError(
      f'Unknown loss_function {loss!r} for adapter '
      f'{adapter_name!r}. Valid losses for {canonical}: '
      f'{sorted(registry)}. Or pass a dotted callable '
      "string such as 'forecast_ml_framework.losses.'"
      "huber_loss' / 'sklearn.metrics.'"
      "'mean_absolute_error'.")
  return registry[key]


def _loss_kwargs(adapter_name, params):
  '''Return estimator kwargs with loss_function applied.

  Args:
    adapter_name: Short adapter name or dotted handle.
    params: Adapter hyperparameters (never mutated).

  Returns:
    Dict of estimator kwargs. An explicit 'loss_function'
    overrides any 'objective'/'loss' in params; a recognised
    'objective' is normalised to the native value, an
    unrecognised native objective passes through unchanged
    (back-compat). No loss key means defaults are unchanged.

  Raises:
    ValueError: If 'loss_function' is unknown/unresolvable.
  '''
  kwargs = _validated_handle_kwargs(params)
  kwargs.pop('loss_function', None)
  canonical = _canonical_adapter(adapter_name)
  objective = kwargs.get('objective')
  if isinstance(objective, str) and '.' not in objective:
    try:
      kwargs['objective'] = get_loss(adapter_name, objective)
    except ValueError:
      # Safe: unrecognised native objective passes through
      # unchanged (back-compat, see _loss_kwargs docstring).
      pass
  loss = params.get('loss_function')
  if loss is not None:
    kwarg = LOSS_KWARGS.get(canonical, 'loss')
    kwargs[kwarg] = get_loss(adapter_name, loss)
  return kwargs


def _estimator_kwargs(params):
  '''Return estimator kwargs without reserved handle keys.

  Args:
    params: Adapter hyperparameters (never mutated).

  Returns:
    Dict of kwargs safe to pass to an estimator constructor.
  '''
  skip = set(HANDLE_KEYS) | {'task_type', 'model_type',
                             'loss_function', 'tuning',
                             'tuning_params', 'tuning_config',
                             'hyperparameter_tuning', 'balance',
                             'class_balancing'}
  return {k: v for k, v in params.items() if k not in skip}


def _validated_handle_kwargs(params):
  '''Validate dotted handles in params, returning clean kwargs.

  Args:
    params: Adapter hyperparameters (never mutated).

  Returns:
    Dict of estimator kwargs with handles stripped out.

  Raises:
    ValueError: If a dotted handle cannot be resolved.
  '''
  for key in HANDLE_KEYS:
    value = params.get(key)
    if isinstance(value, str) and '.' in value:
      resolve_handle(value)
  return _estimator_kwargs(params)


#: Param keys that may carry a tuning config dict.
TUNING_KEYS = ('tuning', 'tuning_params', 'tuning_config',
               'hyperparameter_tuning')


def _tuning_config(params):
  '''Return the tuning config dict in params, or None.

  Args:
    params: Adapter hyperparameters (never mutated).

  Returns:
    Tuning dict if present, else None.
  '''
  for key in TUNING_KEYS:
    value = (params or {}).get(key)
    if isinstance(value, dict):
      return value
  return None


def _balance_kwargs(cls, params, y, kwargs, task):
  '''Inject balance kwargs into estimator kwargs if enabled.

  Args:
    cls: Estimator class to inspect for supported kwargs.
    params: Adapter hyperparameters (never mutated).
    y: Training target used to compute imbalance.
    kwargs: Estimator kwargs (updated in place).
    task: Task kind string.

  Returns:
    Tuple (kwargs, imbalance dict or None). Prefers
    ``scale_pos_weight`` when the estimator signature
    supports it, else ``class_weight`` / ``class_weights``.
    The ``balance`` flag itself is never passed through.
  '''
  if not (params or {}).get('balance'):
    return kwargs, None
  from forecast_ml_framework.pipelines.modeling.imbalance import (
    compute_imbalance,
  )
  info = compute_imbalance(y, task)
  try:
    signature = inspect.signature(cls)
  except (TypeError, ValueError):
    signature = None
  names = set(signature.parameters) if signature is not None else set()
  has_var_kw = signature is not None and any(
    p.kind == inspect.Parameter.VAR_KEYWORD
    for p in signature.parameters.values())
  if 'scale_pos_weight' in names or has_var_kw:
    kwargs.setdefault('scale_pos_weight',
                      info['scale_pos_weight'])
  elif 'class_weight' in names:
    kwargs.setdefault('class_weight', info['class_weights'])
  elif 'class_weights' in names:
    kwargs.setdefault('class_weights', info['class_weights'])
  return kwargs, info


def _check_zoo_tuning(adapter_name, params):
  '''Raise ValueError if tuning is set for a zoo adapter.

  Tuning is only supported for 'sklearn' (and handle
  adapters wrapping a sklearn-style estimator, which tune
  via tune_model). Zoo adapters (xgboost / lightgbm /
  catboost / h2o_automl) never tune silently: a tuning
  config must fail loud instead of being ignored.

  Args:
    adapter_name: Zoo adapter name.
    params: Adapter hyperparameters (never mutated).

  Raises:
    ValueError: If a tuning config dict is present.
  '''
  if _tuning_config(params) is not None:
    raise ValueError(
      f'tuning is not supported for {adapter_name!r} adapter: '
      'tuning config is only supported for sklearn and handle '
      'adapters; remove the tuning config or switch '
      'models_to_use to sklearn/handle (fail loud, never '
      'silently ignore tuning)')


def _missing(lib, name):
  '''Build the informative missing-library error.'''
  return RuntimeError(
    f'{name} requires the {lib!r} library, which is not installed. '
    f'Disable it by removing {name!r} from '
    'params.modeling_params.models_to_use '
    '(params.models_to_use).')


def _fit_zoo(adapter, x, y, cls):
  '''Fit a zoo adapter (xgboost/lightgbm/catboost) and return it.

  Shared body of the zoo ``fit`` methods: fail loud on a
  tuning config, resolve task/kwargs/balance, build ``cls``
  and fit it. Lazy library imports stay in each adapter so
  importing this module never requires a zoo library.

  Args:
    adapter: Zoo adapter instance (its ``name``/``params``
      drive task, loss and balance handling).
    x: Training features.
    y: Training target.
    cls: Estimator class to build and fit.

  Returns:
    The same adapter, with ``model`` fitted and
    ``imbalance_`` recorded.

  Raises:
    ValueError: If a tuning config is present, or the
      loss/balance handling rejects the params.
  '''
  _check_zoo_tuning(adapter.name, adapter.params)
  task = normalize_task_kind(
    adapter.params.get('task_type', 'classification'))
  kwargs, info = _balance_kwargs(
    cls, adapter.params, y,
    _loss_kwargs(adapter.name, adapter.params), task)
  adapter.imbalance_ = info
  adapter.model = cls(**kwargs)
  adapter.model.fit(x, y)
  return adapter


class ModelAdapter(ABC):
  '''Abstract model adapter interface.

  Attributes:
    name: Registry name of this adapter.
    params: Hyperparameters for the underlying model.
  '''

  name = 'base'

  def __init__(self, params=None):
    '''Store adapter hyperparameters.

    Args:
      params: Dict of hyperparameters (may be empty).
    '''
    self.params = dict(params or {})
    self.model = None
    self.tuning_report = None
    self.tuning_best_params = None
    self.imbalance_ = None

  @abstractmethod
  def fit(self, x, y):
    '''Fit the model. Returns self.'''

  def predict(self, x):
    '''Predict labels or values.'''
    return self.model.predict(x)

  def predict_proba(self, x):
    '''Predict class probabilities.'''
    return self.model.predict_proba(x)

  def save(self, path):
    '''Serialize the fitted model to path.'''
    with open(path, 'wb') as handle:
      pickle.dump(self.model, handle)

  @classmethod
  def load(cls, path):
    '''Load a saved model and wrap it in an adapter.

    Reads the raw bytes and unpickles via
    ``core.artifact_store.restricted_loads`` (exact allowlist),
    never raw ``pickle.load``. Only sklearn/numpy/pandas
    objects whose globals are in the allowlist roundtrip;
    zoo models (xgboost/lightgbm/catboost/h2o) whose pickles
    reference non-allowlisted globals are rejected with
    ``ValueError`` by design (sklearn-only loading). Callers
    needing those libraries must add their globals to the
    artifact_store allowlist in a reviewed change — this
    module must not widen it unilaterally.

    Args:
      path: Filesystem path of the saved model.

    Returns:
      An adapter instance wrapping the loaded model.

    Raises:
      ValueError: When the pickle is malformed or references
        a forbidden (non-allowlisted) global.
    '''
    from forecast_ml_framework.core.artifact_store import (
      restricted_loads,
    )
    with open(path, 'rb') as handle:
      raw = handle.read()
    model = restricted_loads(raw)
    obj = cls({})
    obj.model = model
    return obj


class SklearnAdapter(ModelAdapter):
  '''Adapter for scikit-learn models.'''

  name = 'sklearn'

  def _build(self, task_type=None, y=None):
    '''Build the underlying sklearn estimator.

    Raises:
      ValueError: On an unknown ``model_type`` or a
        classifier/regressor ``model_type`` that mismatches
        the task (fail loud, never silently train the wrong
        kind of model).
    '''
    from sklearn.ensemble import (GradientBoostingClassifier,
                                  GradientBoostingRegressor,
                                  RandomForestClassifier,
                                  RandomForestRegressor)
    from sklearn.linear_model import (LinearRegression,
                                      LogisticRegression)
    task = normalize_task_kind(
        task_type or self.params.get('task_type', 'classification'))
    kind = self.params.get('model_type')
    if kind is None:
      # No explicit model_type: default follows the task, so
      # a regression adapter without model_type builds a
      # regressor instead of silently training a classifier.
      kind = ('linear_regression' if task == 'regression'
              else 'logistic_regression')
    table = {
      'logistic_regression': LogisticRegression,
      'linear_regression': LinearRegression,
      'random_forest': RandomForestClassifier,
      'random_forest_regressor': RandomForestRegressor,
      'gradient_boosting': GradientBoostingClassifier,
      'gradient_boosting_regressor': GradientBoostingRegressor,
    }
    if kind not in table:
      raise ValueError(
        f'Unknown model_type {kind!r} for sklearn adapter: '
        f'model_type must be one of {sorted(table)}')
    cls = table[kind]
    # NOTE: 'logistic_regression' ends with '_regression' but is
    # a classifier, so membership is explicit, not suffix-based.
    is_regressor = kind in {'linear_regression',
                            'random_forest_regressor',
                            'gradient_boosting_regressor'}
    # Mismatch is only enforced when the task was declared
    # explicitly: without a task_type, model_type itself
    # declares the estimator kind (back-compat for callers
    # that pass e.g. linear_regression alone).
    explicit_task = (task_type is not None
                     or 'task_type' in self.params)
    if not explicit_task:
      task = 'regression' if is_regressor else 'classification'
    if task == 'regression' and not is_regressor:
      raise ValueError(
        f'model_type {kind!r} is a classifier but task_type is '
        'regression: use an explicit *_regressor model_type '
        '(random_forest_regressor / gradient_boosting_regressor / '
        'linear_regression) for regression')
    if task == 'classification' and is_regressor:
      raise ValueError(
        f'model_type {kind!r} is a regressor but task_type is '
        'classification: use a classifier model_type '
        '(logistic_regression / random_forest / gradient_boosting) '
        'for classification')
    kwargs = _loss_kwargs(self.name, self.params)
    if y is not None:
      task = normalize_task_kind(
        self.params.get('task_type', 'classification'))
      kwargs, info = _balance_kwargs(
        cls, self.params, y, kwargs, task)
      if info is not None:
        self.imbalance_ = info
    try:
      return cls(**kwargs)
    except TypeError as exc:
      raise ValueError(
        f'Estimator {cls.__name__} rejected kwargs {kwargs!r} '
        f'(including loss): {exc}. Check loss_function and '
        'model hyperparameters for this model_type.') from exc

  def fit(self, x, y):
    '''Fit the sklearn model, tuning first if configured.'''
    tuning = _tuning_config(self.params)
    if tuning is not None and self.params.get('balance'):
      # Let tune_model recompute balance weights per CV fold
      # (via _FoldBalancedEstimator) instead of reusing the
      # full-train weights baked into ``base``.
      tuning = dict(tuning)
      tuning.setdefault('balance', True)
    base = self._build(y=y)
    if tuning is not None:
      from forecast_ml_framework.pipelines.modeling.tuning import (
        tune_model,
      )
      best, best_params, report = tune_model(
        base, x, y, tuning)
      self.model = best
      self.tuning_best_params = best_params
      self.tuning_report = report
      return self
    self.model = base
    self.model.fit(x, y)
    return self


class XGBoostAdapter(ModelAdapter):
  '''Adapter for XGBoost models (lazy import).'''

  name = 'xgboost'

  def fit(self, x, y):
    '''Fit the XGBoost model. Returns self.'''
    try:
      import xgboost as xgb
    except ImportError as exc:
      raise _missing('xgboost', self.name) from exc
    task = normalize_task_kind(
        self.params.get('task_type', 'classification'))
    cls = xgb.XGBClassifier if task == 'classification' else xgb.XGBRegressor
    return _fit_zoo(self, x, y, cls)


class LightGBMAdapter(ModelAdapter):
  '''Adapter for LightGBM models (lazy import).'''

  name = 'lightgbm'

  def fit(self, x, y):
    '''Fit the LightGBM model. Returns self.'''
    try:
      import lightgbm as lgb
    except ImportError as exc:
      raise _missing('lightgbm', self.name) from exc
    task = normalize_task_kind(
        self.params.get('task_type', 'classification'))
    cls = lgb.LGBMClassifier if task == 'classification' else lgb.LGBMRegressor
    return _fit_zoo(self, x, y, cls)


class CatBoostAdapter(ModelAdapter):
  '''Adapter for CatBoost models (lazy import).'''

  name = 'catboost'

  def fit(self, x, y):
    '''Fit the CatBoost model. Returns self.'''
    try:
      from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError as exc:
      raise _missing('catboost', self.name) from exc
    task = normalize_task_kind(
        self.params.get('task_type', 'classification'))
    cls = (CatBoostClassifier if task == 'classification'
           else CatBoostRegressor)
    return _fit_zoo(self, x, y, cls)


class H2OAutoMLAdapter(ModelAdapter):
  '''Adapter for H2O AutoML models (lazy import).'''

  name = 'h2o_automl'

  def fit(self, x, y):
    '''Fit the H2O AutoML model. Returns self.'''
    _check_zoo_tuning(self.name, self.params)
    if (self.params or {}).get('balance'):
      raise ValueError(
        'balance is not supported for h2o_automl adapter: '
        'class balancing cannot be applied to H2O AutoML; '
        'remove balance or switch models_to_use to '
        'sklearn/handle (fail loud, never silently ignore '
        'balance)')
    try:
      import h2o
    except ImportError as exc:
      raise _missing('h2o', self.name) from exc
    h2o.init(disable_server=True)
    import pandas as pd
    frame = pd.DataFrame(x)
    frame['target'] = np.asarray(y)
    kwargs = _loss_kwargs(self.name, self.params)
    self.model = h2o.automl.H2OAutoMLEstimator(**kwargs)
    self.model.train(x=frame, y='target')
    return self

  def predict(self, x):
    '''Predict with the H2O model.'''
    import pandas as pd
    return self.model.predict(pd.DataFrame(x))

  def predict_proba(self, x):
    '''Predict class probabilities with the H2O model.

    Raises:
      NotImplementedError: If the fitted H2O model has no
        callable predict_proba. H2O predict() returns labels,
        not probabilities, so it is never used as a proba
        substitute.
    '''
    predict_proba = getattr(self.model, 'predict_proba', None)
    if not callable(predict_proba):
      raise NotImplementedError(
        'H2OAutoMLAdapter.predict_proba requires the fitted '
        'model to expose predict_proba; this model does not. '
        'Use predict() for labels instead.')
    import pandas as pd
    return predict_proba(pd.DataFrame(x))


class HandleAdapter(ModelAdapter):
  '''Adapter for any estimator named by a dotted handle.'''

  name = 'handle'

  def __init__(self, params=None, handle=None):
    '''Store hyperparameters and the dotted handle to resolve.

    Args:
      params: Hyperparameters for the resolved estimator.
      handle: Dotted importable string; defaults to the
        'model_handle' (or 'handle') entry in params.
    '''
    super().__init__(params)
    self.handle = handle or self.params.get(
        'model_handle') or self.params.get('handle')

  def fit(self, x, y):
    '''Resolve the handle, build and fit the estimator.'''
    if not self.handle:
      raise ValueError(
        'HandleAdapter requires a model_handle dotted '
        'importable string.')
    cls = resolve_handle(self.handle)
    task = normalize_task_kind(
      self.params.get('task_type', 'classification'))
    kwargs, info = _balance_kwargs(
      cls, self.params, y,
      _loss_kwargs(self.handle, self.params), task)
    self.imbalance_ = info
    try:
      base = cls(**kwargs)
    except TypeError as exc:
      raise ValueError(
        f'Estimator {cls.__name__} rejected kwargs {kwargs!r} '
        f'(including loss): {exc}. Check loss_function and '
        'model hyperparameters for this handle.') from exc
    tuning = _tuning_config(self.params)
    if tuning is not None and self.params.get('balance'):
      tuning = dict(tuning)
      tuning.setdefault('balance', True)
    if tuning is not None:
      from forecast_ml_framework.pipelines.modeling.tuning import (
        tune_model,
      )
      best, best_params, report = tune_model(base, x, y, tuning)
      self.model = best
      self.tuning_best_params = best_params
      self.tuning_report = report
      return self
    self.model = base
    self.model.fit(x, y)
    return self


ADAPTERS = {
  'sklearn': SklearnAdapter,
  'xgboost': XGBoostAdapter,
  'lightgbm': LightGBMAdapter,
  'catboost': CatBoostAdapter,
  'h2o_automl': H2OAutoMLAdapter,
  'h2o': H2OAutoMLAdapter,
}


def create_adapter(name, params=None):
  '''Create an adapter by registry name or dotted handle.

  Args:
    name: Short adapter name ('sklearn', 'xgboost',
      'lightgbm', 'catboost', 'h2o_automl') or a dotted
      importable string such as
      'sklearn.linear_model.LogisticRegression'.
    params: Hyperparameters for the adapter. May carry
      'model_handle' / 'calibration_handle' /
      'tuning_handle' dotted strings, which are resolved
      via resolve_handle() and never passed through to
      the estimator constructor.

  Returns:
    A ModelAdapter instance.

  Raises:
    ValueError: If the name is unknown or a dotted handle
      cannot be resolved.
  '''
  params = dict(params or {})
  if isinstance(name, str) and '.' in name:
    return HandleAdapter(params, handle=name)
  if name not in ADAPTERS:
    raise ValueError(f'Unknown model adapter: {name!r}. '
                     f'Known: {sorted(ADAPTERS)}')
  handle = params.get('model_handle')
  if isinstance(handle, str) and '.' in handle:
    return HandleAdapter(params, handle=handle)
  _validated_handle_kwargs(params)
  return ADAPTERS[name](params)


def _pick_best(per_model, params):
  '''Pick best model name by the selection metric.'''
  from forecast_ml_framework.pipelines.modeling.metrics import (
    LOWER_BETTER,
  )
  metric = params.get('selection_metric', 'f1_weighted')
  higher = params.get('selection_metric_higher_better')
  if higher is None:
    higher = metric not in LOWER_BETTER
  best, best_val = None, None
  for name, metrics in per_model.items():
    val = metrics.get(metric)
    if val is None or (isinstance(val, float) and np.isnan(val)):
      continue
    if best_val is None or (val > best_val if higher
                            else val < best_val):
      best, best_val = name, val
  return best


def _model_names(params):
  '''Return the deduplicated, capped model names to train.'''
  raw_names = params.get('models_to_use', ['sklearn'])
  if params.get('model_handle') and 'models_to_use' not in params:
    raw_names = [params['model_handle']]
  return list(dict.fromkeys(raw_names))[:MAX_MODELS]


def _per_model_params(name, params, task):
  '''Build the hyperparameter dict for one model name.'''
  model_params = params.get('model_params', {}) or {}
  by_name = params.get('model_params_by_name', {}) or {}
  source = model_params.get(name)
  if not isinstance(source, dict):
    source = by_name.get(name, {})
  if not isinstance(source, dict):
    source = {}
  if not source and model_params and not any(
      isinstance(v, dict) for v in model_params.values()):
    source = model_params
  mp = dict(source)
  if params.get('model_handle'):
    mp.setdefault('model_handle', params['model_handle'])
  if params.get('loss_function') is not None:
    mp.setdefault('loss_function', params['loss_function'])
  for key in TUNING_KEYS:
    if isinstance(params.get(key), dict):
      mp.setdefault(key, params[key])
  if params.get('balance') is not None:
    mp.setdefault('balance', params['balance'])
  mp.setdefault('task_type', task)
  return mp


def _selection_split(x_train, y_train, params, task):
  '''Split x_train into fit/validation parts for selection.

  Args:
    x_train: Training features.
    y_train: Training target.
    params: modeling_params dict (``random_state`` and
      optional ``selection_test_size`` are honoured).
    task: Normalised task kind.

  Returns:
    Tuple (x_fit, y_fit, x_val, y_val), or None when the
    training set is too small for a held-out slice
    (< 10 rows): selection then falls back to scoring on
    the full training set.
  '''
  y_values = np.asarray(y_train)
  if len(y_values) < 10:
    return None
  from sklearn.model_selection import train_test_split
  test_size = params.get('selection_test_size', 0.2)
  seed = params.get('random_state', 42)
  stratify = None
  if task == 'classification':
    _, counts = np.unique(y_values, return_counts=True)
    if counts.min() >= 2:
      stratify = y_values
  try:
    return train_test_split(
        x_train, y_values, test_size=test_size,
        random_state=seed, stratify=stratify)
  except ValueError:
    return train_test_split(
        x_train, y_values, test_size=test_size, random_state=seed)


def _score_on(adapter, x, y, task):
  '''Fit is assumed done; score adapter predictions on (x, y).'''
  from forecast_ml_framework.pipelines.modeling.metrics import (
    compute_metrics,
  )
  y_pred = adapter.predict(x)
  proba = None
  if task == 'classification':
    proba = adapter.predict_proba(x)
  return compute_metrics(y, y_pred, proba, task)


def _train_one(name, params, task, x_train, y_train, split):
  '''Fit, select-score and refit one model; return its parts.

  The selection score is computed on a held-out slice of
  ``x_train`` (fit on the other slice), never on the test
  set. The returned adapter is then refitted on the full
  ``x_train`` so the shipped model uses all training data.
  '''
  mp = _per_model_params(name, params, task)
  if split is None:
    adapter = create_adapter(name, mp)
    adapter.fit(x_train, y_train)
    metrics = _score_on(adapter, x_train, y_train, task)
  else:
    x_fit, x_val, y_fit, y_val = split
    adapter = create_adapter(name, mp)
    adapter.fit(x_fit, y_fit)
    metrics = _score_on(adapter, x_val, y_val, task)
    adapter = create_adapter(name, mp)
    adapter.fit(x_train, y_train)
  tuning_report = getattr(adapter, 'tuning_report', None)
  if tuning_report is not None:
    metrics['tuning'] = tuning_report
  return adapter, metrics


def train_models(x_train, y_train, x_test, y_test, params):
  '''Train every model in the zoo and pick the best.

  Model selection uses a held-out validation slice INSIDE
  ``x_train`` (fit on the rest of train, scored on the
  slice); ``x_test``/``y_test`` are never read here, so the
  test set stays an honest hold-out for the final report.
  The chosen adapter is refitted on the full ``x_train``.
  ``per_model`` metrics are those train-validation scores.

  Args:
    x_train: Training features.
    y_train: Training target.
    x_test: Test features (unused; kept for pipeline compat).
    y_test: Test target (unused; never used for selection).
    params: modeling_params dict (see pipeline header).

  Returns:
    Tuple of (best fitted adapter, per-model metrics, best name).
  '''
  # pylint: disable=unused-argument
  task = normalize_task_kind(
      params.get('task_type', 'classification'))
  names = _model_names(params)
  for key in ('calibration_handle', 'tuning_handle'):
    value = params.get(key)
    if isinstance(value, str) and '.' in value:
      resolve_handle(value)
  split = _selection_split(x_train, y_train, params, task)
  per_model, fitted = {}, {}
  for name in names:
    adapter, metrics = _train_one(
        name, params, task, x_train, y_train, split)
    per_model[name] = metrics
    fitted[name] = adapter
  best = _pick_best(per_model, params) or names[0]
  return fitted[best], per_model, best
