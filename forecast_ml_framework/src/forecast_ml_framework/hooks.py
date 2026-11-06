#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Kedro hooks: MLflow run-lifecycle tracking (vz-high Task 11).

``ForecastHooks`` starts an MLflow run before a pipeline run,
logs node params/tags, ends the run FINISHED on success and
FAILED on error. All MLflow access is lazy (imported inside
each method); when ``mlflow`` is not importable every method is
a no-op that logs a warning, so a Kedro run never fails because
of hooks. No hook method ever raises.

Method names/signatures match Kedro 1.7
``kedro.framework.hooks.specs`` (PipelineSpecs/NodeSpecs):
``before_pipeline_run(run_params, pipeline, catalog)``,
``after_pipeline_run(run_params, run_result, pipeline, catalog)``,
``on_pipeline_error(error, run_params, pipeline, catalog)``,
``before_node_run(node, catalog, inputs, is_async, run_id)``,
``after_node_run(node, catalog, inputs, outputs, is_async,
run_id)``. ``on_error`` is kept as an alias of
``on_pipeline_error`` (legacy name used by earlier callers).
'''

import logging
import re
from datetime import datetime, timezone

LOG = logging.getLogger(__name__)

# Allowlist for pipeline / tag names logged to MLflow. Anything
# outside this set (control chars, separators, injection chars)
# is replaced with '_'; the result is capped at _MAX_NAME_LEN.
_ALLOWLIST_RE = re.compile(r'[^A-Za-z0-9 _\-.:]')
_CONTROL_RE = re.compile(r'[\x00-\x1f\x7f]')
_MAX_NAME_LEN = 100


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
    LOG.warning('ForecastHooks: mlflow not available (%s); no-op', exc)
    return None


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

  Attributes:
    run_id: MLflow run id of the active run, or None.
    warning_count: Number of hook failures recorded so far.
    hook_warning_count: Alias of ``warning_count``.
  '''

  def __init__(self):
    '''Initialise with no active MLflow run.'''
    self.run_id = None
    self.warning_count = 0
    self.hook_warning_count = 0
    self._run_active = False

  def _record_warning(self, mlflow, message):
    '''Count a hook failure and log it as an MLflow metric.'''
    self.warning_count += 1
    self.hook_warning_count = self.warning_count
    LOG.warning(message)
    try:
      if mlflow is not None and hasattr(mlflow, 'log_metrics'):
        mlflow.log_metrics({'hook_warnings': self.warning_count})
    except Exception:  # pylint: disable=broad-except
      LOG.warning('ForecastHooks: could not log warning metric',
                  exc_info=True)

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
    except Exception:  # pylint: disable=broad-except
      LOG.warning('ForecastHooks: could not capture run_id',
                  exc_info=True)

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
    except Exception:  # pylint: disable=broad-except
      LOG.warning('ForecastHooks: could not check active run',
                  exc_info=True)
      return self._run_active
    return self._run_active

  def _end_run(self, mlflow, status):
    '''End the active run, guarded so no-run never calls end_run.'''
    if not self._is_run_active(mlflow):
      return
    mlflow.end_run(status=status)
    self._run_active = False
    self.run_id = None

  def before_pipeline_run(self, run_params=None, pipeline=None,
                          catalog=None, *args, **kwargs):
    '''Start an MLflow run named ``<pipeline>-<timestamp>``.

    Kedro 1.7 signature: ``(run_params, pipeline, catalog)``.
    Legacy ``(context, run_params, pipeline)`` calls are also
    accepted (the dict argument is honoured either way).
    '''
    mlflow = _mlflow_or_warn()
    if mlflow is None:
      return
    try:
      params = _run_params_from((run_params, pipeline, catalog) + args,
                                kwargs)
      name = sanitize_name(
          params.get('pipeline_name') or params.get('pipeline')
          or 'kedro')
      stamp = params.get('timestamp') or datetime.now(
          timezone.utc).strftime('%Y%m%d-%H%M%S')
      mlflow.start_run(run_name=f'{name}-{stamp}')
      self._run_active = True
      self._capture_run_id(mlflow)
    except Exception:  # pylint: disable=broad-except
      self._record_warning(mlflow,
                           'ForecastHooks.before_pipeline_run failed')

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

  def before_node_run(self, node=None, catalog=None, inputs=None,
                      is_async=False, run_id=None, *args, **kwargs):
    '''Log node name as a tag/param before it runs.'''
    mlflow = _mlflow_or_warn()
    if mlflow is None:
      return
    try:
      self._note_run_id(run_id)
      name = sanitize_name(getattr(node, 'name', str(node)),
                           default='node')
      stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
      mlflow.set_tag('node', name)
      mlflow.log_params({f'{name}_{stamp}': name})
    except Exception:  # pylint: disable=broad-except
      self._record_warning(mlflow,
                           'ForecastHooks.before_node_run failed')

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

  def on_error(self, *args, **kwargs):
    '''Alias of :meth:`on_pipeline_error` (legacy name).'''
    self.on_pipeline_error(*args, **kwargs)
