#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Model adapter ABC, zoo adapters and the train node.'''

import importlib
import pickle
from abc import ABC, abstractmethod

import numpy as np

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
  ``pipelines.scoring.nodes._resolve_handle_strict`` so
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
  from forecast_ml_framework.pipelines.scoring.nodes import (
    _resolve_handle_strict,
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
  '''Resolve a dotted custom-loss under the allowlist.

  Only callables whose top-level module is allowlisted
  (same list as ``_resolve_handle_strict``) are accepted;
  arbitrary callables such as ``os.system`` are rejected.

  Args:
    loss: Dotted importable string naming a callable.

  Returns:
    The resolved callable.

  Raises:
    ValueError: If not allowlisted, unresolvable, or
      non-callable.
  '''
  from forecast_ml_framework.pipelines.scoring.nodes import (
    _ALLOWED_MODEL_MODULES,
  )
  top_level = loss.split('.', 1)[0]
  if top_level not in _ALLOWED_MODEL_MODULES:
    raise ValueError(
      f'loss_function {loss!r} is not allowlisted: top-level '
      f'module {top_level!r} is not one of '
      + ', '.join(_ALLOWED_MODEL_MODULES))
  module_path, _, attr = loss.rpartition('.')
  try:
    module = importlib.import_module(module_path)
  except ImportError as exc:
    raise ValueError(
      f'Cannot resolve loss_function {loss!r}: module '
      f'{module_path!r} could not be imported ({exc}).') from exc
  try:
    resolved = getattr(module, attr)
  except AttributeError as exc:
    raise ValueError(
      f'Cannot resolve loss_function {loss!r}: module '
      f'{module_path!r} has no attribute {attr!r}.') from exc
  if not callable(resolved):
    raise ValueError(
      f'loss_function {loss!r} resolved to a non-callable '
      'object; a custom loss must be callable.')
  return resolved


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
      to a callable (e.g. 'my_pkg.losses.my_loss').

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
      "string such as 'my_pkg.losses.my_loss'.")
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
      "string such as 'my_pkg.losses.my_loss'.")
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
                             'loss_function'}
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


def _missing(lib, name):
  '''Build the informative missing-library error.'''
  return RuntimeError(
    f'{name} requires the {lib!r} library, which is not installed. '
    f'Disable it by removing {name!r} from '
    'params.modeling_params.models_to_use '
    '(params.models_to_use).')


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

  def _build(self, task_type=None):
    '''Build the underlying sklearn estimator.'''
    from sklearn.ensemble import (GradientBoostingClassifier,
                                  GradientBoostingRegressor,
                                  RandomForestClassifier,
                                  RandomForestRegressor)
    from sklearn.linear_model import (LinearRegression,
                                      LogisticRegression)
    kind = self.params.get('model_type', 'logistic_regression')
    table = {
      'logistic_regression': LogisticRegression,
      'linear_regression': LinearRegression,
      'random_forest': RandomForestClassifier,
      'random_forest_regressor': RandomForestRegressor,
      'gradient_boosting': GradientBoostingClassifier,
      'gradient_boosting_regressor': GradientBoostingRegressor,
    }
    cls = table.get(kind, LogisticRegression)
    kwargs = _loss_kwargs(self.name, self.params)
    try:
      return cls(**kwargs)
    except TypeError:
      fallback = {'linear_regression': LinearRegression,
                  'random_forest_regressor':
                  RandomForestRegressor,
                  'gradient_boosting_regressor':
                  GradientBoostingRegressor}
      return fallback.get(kind, LogisticRegression)()

  def fit(self, x, y):
    '''Fit the sklearn model. Returns self.'''
    self.model = self._build()
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
    task = self.params.get('task_type', 'classification')
    cls = xgb.XGBClassifier if task == 'classification' else xgb.XGBRegressor
    self.model = cls(**_loss_kwargs(self.name, self.params))
    self.model.fit(x, y)
    return self


class LightGBMAdapter(ModelAdapter):
  '''Adapter for LightGBM models (lazy import).'''

  name = 'lightgbm'

  def fit(self, x, y):
    '''Fit the LightGBM model. Returns self.'''
    try:
      import lightgbm as lgb
    except ImportError as exc:
      raise _missing('lightgbm', self.name) from exc
    task = self.params.get('task_type', 'classification')
    cls = lgb.LGBMClassifier if task == 'classification' else lgb.LGBMRegressor
    self.model = cls(**_loss_kwargs(self.name, self.params))
    self.model.fit(x, y)
    return self


class CatBoostAdapter(ModelAdapter):
  '''Adapter for CatBoost models (lazy import).'''

  name = 'catboost'

  def fit(self, x, y):
    '''Fit the CatBoost model. Returns self.'''
    try:
      from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError as exc:
      raise _missing('catboost', self.name) from exc
    task = self.params.get('task_type', 'classification')
    cls = (CatBoostClassifier if task == 'classification'
           else CatBoostRegressor)
    self.model = cls(**_loss_kwargs(self.name, self.params))
    self.model.fit(x, y)
    return self


class H2OAutoMLAdapter(ModelAdapter):
  '''Adapter for H2O AutoML models (lazy import).'''

  name = 'h2o_automl'

  def fit(self, x, y):
    '''Fit the H2O AutoML model. Returns self.'''
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
    kwargs = _loss_kwargs(self.handle, self.params)
    try:
      self.model = cls(**kwargs)
    except TypeError:
      self.model = cls()
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


def train_models(x_train, y_train, x_test, y_test, params):
  '''Train every model in the zoo and pick the best.

  Args:
    x_train: Training features.
    y_train: Training target.
    x_test: Test features.
    y_test: Test target.
    params: modeling_params dict (see pipeline header).

  Returns:
    Tuple of (best fitted adapter, per-model metrics, best name).
  '''
  from forecast_ml_framework.pipelines.modeling.metrics import (
    compute_metrics,
  )
  task = params.get('task_type', 'classification')
  raw_names = params.get('models_to_use', ['sklearn'])
  if params.get('model_handle') and 'models_to_use' not in params:
    raw_names = [params['model_handle']]
  names = list(dict.fromkeys(raw_names))[:MAX_MODELS]
  model_params = params.get('model_params', {}) or {}
  by_name = params.get('model_params_by_name', {}) or {}
  for key in ('calibration_handle', 'tuning_handle'):
    value = params.get(key)
    if isinstance(value, str) and '.' in value:
      resolve_handle(value)
  per_model, fitted = {}, {}
  for name in names:
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
    mp.setdefault('task_type', task)
    adapter = create_adapter(name, mp)
    adapter.fit(x_train, y_train)
    y_pred = adapter.predict(x_test)
    proba = None
    if task == 'classification':
      proba = adapter.predict_proba(x_test)
    per_model[name] = compute_metrics(
        y_test, y_pred, proba, task)
    fitted[name] = adapter
  best = _pick_best(per_model, params) or names[0]
  return fitted[best], per_model, best
