#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Kedro hooks: MLflow run-lifecycle tracking (vz-high Task 11).

``ForecastHooks`` starts an MLflow run before a pipeline run,
logs reproducibility params (seed, git SHA, data version) and
node tags plus real scalar input params, ends the run FINISHED
on success and FAILED on error. Tracking URI / experiment
resolve from run_params mlflow_params, else shipped config
via load_config(), else ambient MLflow state (with a warning).
All MLflow access is lazy (imported inside each method); when
``mlflow`` is not importable every method is a no-op that logs
a warning, so a Kedro run never fails because of hooks. No
hook method ever raises; failures log at WARNING with a fixed
code and exception type only (no exception message, no
traceback), so secrets carried by an exception never reach
the log.

Method names/signatures match Kedro 1.7
``kedro.framework.hooks.specs`` (PipelineSpecs/NodeSpecs):
``before_pipeline_run(run_params, pipeline, catalog)``,
``after_pipeline_run(run_params, run_result, pipeline, catalog)``,
``on_pipeline_error(error, run_params, pipeline, catalog)``,
``before_node_run(node, catalog, inputs, is_async, run_id)``,
``after_node_run(node, catalog, inputs, outputs, is_async,
run_id)``. ``on_error`` is kept as an alias of
``on_pipeline_error`` (legacy name used by earlier callers).

Kedro discovery: ``ForecastHooks`` is a plain class (no Kedro
base class) whose five hook methods carry ``@hook_impl``
markers, and ``settings.HOOKS`` registers an instance, as
Kedro 1.7 ``KedroSession`` requires (``_register_hooks``
raises ``TypeError`` for a class; pluggy only exposes
``@hook_impl``-marked methods). Discovery is pinned by tests
in ``tests/test_hooks_repro.py``.
'''

import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

try:
  from kedro.framework.hooks import hook_impl
except ImportError:  # pragma: no cover - kedro optional at import
  def hook_impl(func):
    '''No-op ``hook_impl`` when Kedro is unavailable.

    Args:
      func: The hook method being decorated.

    Returns:
      The undecorated method.
    '''
    return func

LOG = logging.getLogger(__name__)

# Allowlist for pipeline / tag names logged to MLflow. Anything
# outside this set (control chars, separators, injection chars)
# is replaced with '_'; the result is capped at _MAX_NAME_LEN.
_ALLOWLIST_RE = re.compile(r'[^A-Za-z0-9 _\-.:]')
_CONTROL_RE = re.compile(r'[\x00-\x1f\x7f]')
_MAX_NAME_LEN = 100

# Caps for node input params logged to MLflow: at most this many
# params, scalar values truncated to this many chars (MLflow
# limits params to 500 chars), so a node never spams MLflow with
# high-cardinality per-run keys.
_MAX_NODE_PARAMS = 20
_MAX_PARAM_VALUE_LEN = 250


def sanitize_name(value, default='kedro'):
  '''Sanitize a pipeline/tag name for MLflow.

  Strips control characters, replaces characters outside the
  allowlist ``[A-Za-z0-9 _\\-.:]`` with ``_``, collapses
  surrounding whitespace and caps the length at 100 chars.
  Empty/None input yields ``default``.
  '''
  if value is None:
    return default
  text = _CONTROL_RE.sub('', str(value))
  text = _ALLOWLIST_RE.sub('_', text).strip()
  text = text[:_MAX_NAME_LEN].strip()
  return text or default


# Public alias for callers/tests that look for a sanitize helper.
sanitize = sanitize_name


def _mlflow_or_warn():
  '''Return the mlflow module, or None (with a warning).'''
  try:
    import mlflow  # pylint: disable=import-outside-toplevel
    return mlflow
  except ImportError as exc:
    LOG.warning(
        'ForecastHooks: mlflow not available code=%s type=%s',
        'mlflow_unavailable', type(exc).__name__)
    return None


def _git_sha():
  '''Best-effort git SHA: GIT_SHA env, else git rev-parse.

  Returns the SHA string, or 'unknown' when neither works.
  Never raises.
  '''
  env_sha = os.environ.get('GIT_SHA', '').strip()
  if env_sha:
    return env_sha
  try:
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], capture_output=True,
        text=True, timeout=5, check=False)
    sha = (result.stdout or '').strip()
    if result.returncode == 0 and sha:
      return sha
  except Exception as exc:  # pylint: disable=broad-except
    LOG.warning(
        'ForecastHooks: git SHA lookup failed code=%s type=%s',
        'git_sha_failed', type(exc).__name__)
  return 'unknown'


def _find_param(params, keys):
  '''Find the first non-empty value among keys in run params.

  Checks the flat run-params dict, then the nested 'parameters'
  dict (Kedro context shape). Returns None when absent.
  '''
  if not isinstance(params, dict):
    return None
  for key in keys:
    value = params.get(key)
    if value is not None and value != '':
      return value
  nested = params.get('parameters')
  if isinstance(nested, dict):
    for key in keys:
      value = nested.get(key)
      if value is not None and value != '':
        return value
  return None


def _load_shipped_config():
  '''Load shipped project config, or {} when unavailable.'''
  try:
    # Lazy import avoids a cycle at module import time.
    # pylint: disable=import-outside-toplevel
    from forecast_ml_framework.core.config_schema import load_config
    # pylint: enable=import-outside-toplevel
    config = load_config()
    return config if isinstance(config, dict) else {}
  except Exception as exc:  # pylint: disable=broad-except
    LOG.warning(
        'ForecastHooks: could not load shipped config code=%s type=%s',
        'config_load_failed', type(exc).__name__)
    return {}


def _mlflow_params_from(params, hook=None):
  '''Resolve mlflow_params tracking config (dict or {}).

  Precedence: explicit hook config, run_params mlflow_params /
  mlflow / nested parameters, then shipped config via
  load_config(). Returns the resolved dict (possibly {}).
  '''
  hook_params = getattr(hook, 'mlflow_params', None)
  if isinstance(hook_params, dict) and hook_params:
    return hook_params
  section = None
  if isinstance(params, dict):
    section = params.get('mlflow_params') or params.get('mlflow')
    nested = params.get('parameters')
    if section is None and isinstance(nested, dict):
      section = nested.get('mlflow_params') or nested.get('mlflow')
  if isinstance(section, dict) and section:
    return section
  shipped = _load_shipped_config()
  shipped_params = shipped.get('parameters') or {}
  if isinstance(shipped_params, dict):
    section = shipped_params.get('mlflow_params')
    if isinstance(section, dict) and section:
      return section
  return {}


def _seed_from_config():
  '''Seed from shipped config, or None when absent.

  Search order: ``parameters.modeling_params`` then
  ``parameters.global_params`` (then ``global`` / top-level)
  for ``random_state`` or ``seed``. Never raises.
  '''
  shipped = _load_shipped_config()
  params = shipped.get('parameters') if isinstance(shipped, dict) else None
  if not isinstance(params, dict):
    return None
  containers = (
      params.get('modeling_params'),
      params.get('global_params'),
      params.get('global'),
      params,
  )
  for container in containers:
    if isinstance(container, dict):
      for key in ('random_state', 'seed'):
        value = container.get(key)
        if value is not None and value != '':
          return value
  return None


def _repro_params(params):
  '''Reproducibility params logged at pipeline start.

  seed from seed/random_state keys in run_params, else from
  shipped config (modeling_params/global random_state via
  load_config), else the explicit string 'unknown' -- never
  silently omitted. git SHA (best-effort), data version/hash
  ('unknown' fallback). Never raises.
  '''
  repro = {}
  seed = _find_param(params, ('seed', 'random_state'))
  if seed is None:
    seed = _seed_from_config()
  repro['seed'] = (
      str(seed)[:_MAX_PARAM_VALUE_LEN] if seed is not None
      else 'unknown')
  repro['git_sha'] = str(_git_sha())[:_MAX_PARAM_VALUE_LEN]
  version = _find_param(
      params,
      ('data_version', 'data_hash', 'data_snapshot',
       'data_snapshot_hash'))
  repro['data_version'] = (
      str(version)[:_MAX_PARAM_VALUE_LEN] if version
      is not None else 'unknown')
  return repro


def _scalar_node_params(inputs):
  '''Real scalar input params for a node, capped and sanitized.

  Only str/int/float/bool values from a dict ``inputs`` are
  returned (keys sanitized, values capped); everything else is
  dropped so param keys stay low-cardinality and stable.
  '''
  if not isinstance(inputs, dict):
    return {}
  result = {}
  for key, value in inputs.items():
    if len(result) >= _MAX_NODE_PARAMS:
      break
    if value is None or not isinstance(
        value, (str, int, float, bool)):
      continue
    clean_key = sanitize_name(value=str(key), default='input')
    result[clean_key] = str(value)[:_MAX_PARAM_VALUE_LEN]
  return result


def _run_params_from(args, kwargs):
  '''Extract run_params dict from Kedro or legacy call styles.'''
  for value in args:
    if isinstance(value, dict):
      return value
  for key in ('run_params', 'params', 'context'):
    if isinstance(kwargs.get(key), dict):
      return kwargs[key]
  return {}


class ForecastHooks:
  '''MLflow run-lifecycle hooks (plain object, no Kedro base).

  Discovery: the five hook methods carry ``@hook_impl``
  markers and ``settings.HOOKS`` registers an instance;
  see the module docstring. Pinned by discovery tests in
  ``tests/test_hooks_repro.py``.

  Args:
    mlflow_params: Optional {'tracking_uri':...,
      'experiment_name':...} mapping overriding run_params
      and shipped config when resolving MLflow targets.

  Attributes:
    run_id: MLflow run id of the active run, or None.
    warning_count: Number of hook failures recorded so far.
    hook_warning_count: Alias of ``warning_count``.
    mlflow_params: Explicit MLflow target config, or None.
  '''

  def __init__(self, mlflow_params=None):
    '''Initialise with no active MLflow run.

    Args:
      mlflow_params: Optional MLflow target mapping (see
        class docs). Kedro instantiates this with no args.
    '''
    self.run_id = None
    self.warning_count = 0
    self.hook_warning_count = 0
    self._run_active = False
    self.mlflow_params = mlflow_params

  def _record_warning(self, mlflow, message):
    '''Count a hook failure and log it as an MLflow metric.

    Must be called from an ``except`` block: the warning is
    logged at WARNING with a fixed message plus exception
    type only (no message text, no traceback), so failures
    stay visible without ever raising out of a hook.
    '''
    self.warning_count += 1
    self.hook_warning_count = self.warning_count
    exc = sys.exc_info()[1]
    LOG.warning(
        '%s code=%s type=%s', message, 'hook_failed',
        type(exc).__name__ if exc is not None else 'Unknown')
    try:
      if mlflow is not None and hasattr(mlflow, 'log_metrics'):
        mlflow.log_metrics({'hook_warnings': self.warning_count})
    except Exception as exc2:  # pylint: disable=broad-except
      LOG.warning(
          'ForecastHooks: could not log warning metric code=%s type=%s',
          'hook_metric_failed', type(exc2).__name__)

  def _note_run_id(self, run_id):
    '''Record run_id from a node hook when not already set.'''
    if run_id and not self.run_id:
      self.run_id = str(run_id)

  def _capture_run_id(self, mlflow):
    '''Capture run_id from ``mlflow.active_run()`` when present.'''
    try:
      active_run_fn = getattr(mlflow, 'active_run', None)
      if callable(active_run_fn):
        active = active_run_fn()
        if active is not None:
          info = getattr(active, 'info', None)
          run_id = getattr(info, 'run_id', None) if info else None
          if run_id:
            self.run_id = str(run_id)
    except Exception as exc:  # pylint: disable=broad-except
      LOG.warning(
          'ForecastHooks: could not capture run_id code=%s type=%s',
          'run_id_capture_failed', type(exc).__name__)

  def _is_run_active(self, mlflow):
    '''Return True when an MLflow run is active.

    ``mlflow.active_run()`` is authoritative when available;
    otherwise fall back to whether ``before_pipeline_run``
    successfully started a run on this hook instance.
    '''
    try:
      active_run_fn = getattr(mlflow, 'active_run', None)
      if callable(active_run_fn):
        return active_run_fn() is not None
    except Exception as exc:  # pylint: disable=broad-except
      LOG.warning(
          'ForecastHooks: could not check active run code=%s type=%s',
          'active_run_check_failed', type(exc).__name__)
      return self._run_active
    return self._run_active

  def _end_run(self, mlflow, status):
    '''End the active run, guarded so no-run never calls end_run.'''
    if not self._is_run_active(mlflow):
      return
    mlflow.end_run(status=status)
    self._run_active = False
    self.run_id = None

  def _apply_mlflow_config(self, mlflow, params):
    '''Point mlflow at the resolved tracking URI / experiment.

    Config URIs win over ambient MLflow state. When nothing
    resolves, fall back to ambient with a WARNING. Never raises.
    '''
    resolved = _mlflow_params_from(params, self)
    uri = resolved.get('tracking_uri') or resolved.get('uri')
    experiment = (resolved.get('experiment_name')
                  or resolved.get('experiment'))
    applied = False
    if uri:
      set_uri = getattr(mlflow, 'set_tracking_uri', None)
      if callable(set_uri):
        set_uri(uri)
        applied = True
    if experiment:
      set_exp = getattr(mlflow, 'set_experiment', None)
      if callable(set_exp):
        set_exp(experiment)
        applied = True
    if not applied:
      LOG.warning(
          'ForecastHooks: no mlflow_params tracking config '
          'resolved; using ambient MLflow configuration')

  # pylint: disable=keyword-arg-before-vararg
  @hook_impl
  def before_pipeline_run(self, run_params=None, pipeline=None,
                          catalog=None, *args, **kwargs):
    '''Start an MLflow run named ``<pipeline>-<timestamp>``.

    Resolves tracking URI/experiment from run_params
    mlflow_params (else shipped config, else ambient with a
    warning), then logs reproducibility params (seed, git SHA,
    data version). Kedro 1.7 signature: ``(run_params,
    pipeline, catalog)``. Legacy ``(context, run_params,
    pipeline)`` calls are also accepted (the dict argument is
    honoured either way).
    '''
    mlflow = _mlflow_or_warn()
    if mlflow is None:
      return
    try:
      params = _run_params_from((run_params, pipeline, catalog) + args,
                                kwargs)
      self._apply_mlflow_config(mlflow, params)
      name = sanitize_name(
          params.get('pipeline_name') or params.get('pipeline')
          or 'kedro')
      stamp = params.get('timestamp') or datetime.now(
          timezone.utc).strftime('%Y%m%d-%H%M%S')
      mlflow.start_run(run_name=f'{name}-{stamp}')
      self._run_active = True
      self._capture_run_id(mlflow)
      repro = _repro_params(params)
      if repro and hasattr(mlflow, 'log_params'):
        mlflow.log_params(repro)
    except Exception:  # pylint: disable=broad-except
      self._record_warning(mlflow,
                           'ForecastHooks.before_pipeline_run failed')
  # pylint: enable=keyword-arg-before-vararg

  # pylint: disable=unused-argument,keyword-arg-before-vararg
  @hook_impl
  def after_pipeline_run(self, run_params=None, run_result=None,
                         pipeline=None, catalog=None, *args, **kwargs):
    '''End the active MLflow run with status FINISHED.'''
    mlflow = _mlflow_or_warn()
    if mlflow is None:
      return
    try:
      self._end_run(mlflow, 'FINISHED')
    except Exception:  # pylint: disable=broad-except
      self._record_warning(mlflow,
                           'ForecastHooks.after_pipeline_run failed')

  # pylint: enable=unused-argument,keyword-arg-before-vararg

  # pylint: disable=unused-argument,keyword-arg-before-vararg
  @hook_impl
  def before_node_run(self, node=None, catalog=None, inputs=None,
                      is_async=False, run_id=None, *args, **kwargs):
    '''Log node tag and real scalar input params before it runs.'''
    mlflow = _mlflow_or_warn()
    if mlflow is None:
      return
    try:
      self._note_run_id(run_id)
      name = sanitize_name(getattr(node, 'name', str(node)),
                           default='node')
      mlflow.set_tag('node', name)
      scalars = _scalar_node_params(inputs)
      if scalars and hasattr(mlflow, 'log_params'):
        mlflow.log_params(scalars)
    except Exception:  # pylint: disable=broad-except
      self._record_warning(mlflow,
                           'ForecastHooks.before_node_run failed')

  # pylint: enable=unused-argument,keyword-arg-before-vararg

  # pylint: disable=unused-argument,keyword-arg-before-vararg
  @hook_impl
  def after_node_run(self, node=None, catalog=None, inputs=None,
                     outputs=None, is_async=False, run_id=None, *args,
                     **kwargs):
    '''Log node name as a tag after it runs.'''
    mlflow = _mlflow_or_warn()
    if mlflow is None:
      return
    try:
      self._note_run_id(run_id)
      name = sanitize_name(getattr(node, 'name', str(node)),
                           default='node')
      mlflow.set_tag('node', name)
    except Exception:  # pylint: disable=broad-except
      self._record_warning(mlflow,
                           'ForecastHooks.after_node_run failed')

  # pylint: enable=unused-argument,keyword-arg-before-vararg

  # pylint: disable=unused-argument,keyword-arg-before-vararg
  @hook_impl
  def on_pipeline_error(self, error=None, run_params=None, pipeline=None,
                        catalog=None, *args, **kwargs):
    '''End the active MLflow run with status FAILED.'''
    mlflow = _mlflow_or_warn()
    if mlflow is None:
      return
    try:
      self._end_run(mlflow, 'FAILED')
    except Exception:  # pylint: disable=broad-except
      self._record_warning(mlflow,
                           'ForecastHooks.on_pipeline_error failed')
  # pylint: enable=unused-argument,keyword-arg-before-vararg

  def on_error(self, *args, **kwargs):
    '''Alias of :meth:`on_pipeline_error` (legacy name).'''
    self.on_pipeline_error(*args, **kwargs)
