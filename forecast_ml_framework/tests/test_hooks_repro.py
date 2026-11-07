#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Reproducibility and config-resolution tests for ForecastHooks.

Covers W6 spec: before_pipeline_run logs seed / git SHA / data
version, node hooks log real scalar inputs (no timestamp param),
and mlflow tracking config resolves from run_params (else
ambient with a warning). All tests use a fake mlflow module.
'''


import logging
import sys
import types


def _fake_mlflow(monkeypatch, calls):
  '''Install a fake mlflow module recording calls in ``calls``.'''
  fake = types.ModuleType('mlflow')
  fake.start_run = lambda run_name=None: calls.append(
      ('start', run_name))
  fake.end_run = lambda status=None: calls.append(('end', status))
  fake.set_tracking_uri = lambda uri: calls.append(
      ('uri', uri))
  fake.set_experiment = lambda name: calls.append(('exp', name))
  fake.set_tag = lambda k, v: calls.append(('tag', k, v))
  fake.log_params = lambda d: calls.append(('params', dict(d)))
  fake.log_metrics = lambda d: calls.append(('metrics', dict(d)))
  fake.active_run = lambda: None
  monkeypatch.setitem(sys.modules, 'mlflow', fake)
  return fake


def _param_dicts(calls):
  '''Return all logged param dicts from the call list.'''
  return [c[1] for c in calls if c[0] == 'params']


def test_before_pipeline_logs_repro_metadata(monkeypatch):
  '''Seed, git SHA and data version are logged as params.'''
  calls = []
  _fake_mlflow(monkeypatch, calls)
  monkeypatch.setenv('GIT_SHA', 'abc123')
  from forecast_ml_framework.hooks import ForecastHooks
  h = ForecastHooks()
  run_params = {
      'pipeline_name': 'modeling',
      'seed': 42,
      'data_version': 'snap-2026-11',
      'mlflow_params': {
          'tracking_uri': 'sqlite:////tmp/x.db',
          'experiment_name': 'exp-config',
      },
  }
  h.before_pipeline_run(run_params, None, None)
  dicts = _param_dicts(calls)
  merged = {}
  for d in dicts:
    merged.update(d)
  assert merged.get('seed') == '42'
  assert merged.get('git_sha') == 'abc123'
  assert merged.get('data_version') == 'snap-2026-11'


def test_repro_metadata_unknown_defaults(monkeypatch):
  '''Missing seed/data/git fall back to unknown, never crash.'''
  import subprocess as _sp
  calls = []
  _fake_mlflow(monkeypatch, calls)
  monkeypatch.delenv('GIT_SHA', raising=False)

  def _boom(*args, **kwargs):
    '''Raise an error.'''
    raise OSError('no git')

  def _no_config(env_dir='base'):
    '''No config.'''
    raise RuntimeError('no config')

  monkeypatch.setattr(_sp, 'run', _boom)
  monkeypatch.setattr(
      'forecast_ml_framework.core.config_schema.load_config', _no_config)
  from forecast_ml_framework.hooks import ForecastHooks
  h = ForecastHooks()
  h.before_pipeline_run({'pipeline_name': 'modeling'}, None, None)
  merged = {}
  for d in _param_dicts(calls):
    merged.update(d)
  assert merged.get('data_version') == 'unknown'
  assert merged.get('git_sha') == 'unknown'
  # mlops R3: seed is explicit 'unknown', never silently omitted.
  assert merged.get('seed') == 'unknown'


def test_tracking_uri_config_wins_over_ambient(monkeypatch):
  '''run_params mlflow_params URI is applied via set_tracking_uri.'''
  calls = []
  _fake_mlflow(monkeypatch, calls)
  monkeypatch.setenv('GIT_SHA', 'abc123')
  from forecast_ml_framework.hooks import ForecastHooks
  h = ForecastHooks()
  run_params = {
      'pipeline_name': 'modeling',
      'mlflow_params': {
          'tracking_uri': 'sqlite:////tmp/x.db',
          'experiment_name': 'exp-config',
      },
  }
  h.before_pipeline_run(run_params, None, None)
  assert ('uri', 'sqlite:////tmp/x.db') in calls
  assert ('exp', 'exp-config') in calls


def test_tracking_from_load_config(monkeypatch):
  '''Shipped config URI is used when run_params carries none.'''
  calls = []
  _fake_mlflow(monkeypatch, calls)
  monkeypatch.setenv('GIT_SHA', 'abc123')
  shipped = {
      'parameters': {
          'mlflow_params': {
              'tracking_uri': 'sqlite:////tmp/shipped.db',
              'experiment_name': 'exp-shipped',
          },
      },
  }
  monkeypatch.setattr(
      'forecast_ml_framework.core.config_schema.load_config',
      lambda env_dir='base': shipped)
  from forecast_ml_framework.hooks import ForecastHooks
  h = ForecastHooks()
  h.before_pipeline_run({'pipeline_name': 'modeling'}, None, None)
  assert ('uri', 'sqlite:////tmp/shipped.db') in calls
  assert ('exp', 'exp-shipped') in calls


def test_tracking_ambient_fallback_warns(monkeypatch, caplog):
  '''No config anywhere warns and still starts the run.'''
  calls = []
  _fake_mlflow(monkeypatch, calls)
  monkeypatch.setenv('GIT_SHA', 'abc123')

  def _boom(env_dir='base'):
    '''Raise an error.'''
    raise RuntimeError('no config')

  monkeypatch.setattr(
      'forecast_ml_framework.core.config_schema.load_config', _boom)
  from forecast_ml_framework.hooks import ForecastHooks
  h = ForecastHooks()
  with caplog.at_level(logging.WARNING):
    h.before_pipeline_run({'pipeline_name': 'modeling'}, None, None)
  starts = [c for c in calls if c[0] == 'start']
  assert starts
  assert not [c for c in calls if c[0] == 'uri']
  assert 'ambient' in caplog.text


def test_node_params_log_real_scalars_no_timestamp(monkeypatch):
  '''Node hook logs real scalar inputs, without timestamp keys.'''

  class _Node:
    '''Node test double.'''
    name = 'train_node'

  calls = []
  _fake_mlflow(monkeypatch, calls)
  monkeypatch.setenv('GIT_SHA', 'abc123')
  from forecast_ml_framework.hooks import ForecastHooks
  h = ForecastHooks()
  h.before_pipeline_run({'pipeline_name': 'modeling'}, None, None)
  inputs = {'lr': 0.1, 'epochs': 10, 'data': 'frame', 'df': object()}
  h.before_node_run(_Node(), None, inputs, False, None)
  node_dicts = _param_dicts(calls)[1:]
  assert node_dicts
  merged = {}
  for d in node_dicts:
    merged.update(d)
  assert merged.get('lr') == '0.1'
  assert merged.get('epochs') == '10'
  assert merged.get('data') == 'frame'
  assert 'df' not in merged
  for d in node_dicts:
    for key in d:
      assert '_2026' not in key
      assert key != 'train_node_20261106-120000-000000'


def test_node_param_keys_stable_low_cardinality(monkeypatch):
  '''Repeated node runs reuse the same param keys.'''

  class _Node:
    '''Node test double.'''
    name = 'train_node'

  calls = []
  _fake_mlflow(monkeypatch, calls)
  monkeypatch.setenv('GIT_SHA', 'abc123')
  from forecast_ml_framework.hooks import ForecastHooks
  h = ForecastHooks()
  h.before_pipeline_run({'pipeline_name': 'modeling'}, None, None)
  inputs = {'lr': 0.1, 'epochs': 10}
  h.before_node_run(_Node(), None, dict(inputs), False, None)
  h.before_node_run(_Node(), None, dict(inputs), False, None)
  node_dicts = _param_dicts(calls)[1:]
  assert len(node_dicts) == 2
  assert set(node_dicts[0]) == set(node_dicts[1]) == {
      'lr', 'epochs'}


def test_hooks_warn_without_exc_info_on_failure(monkeypatch, caplog):
  '''Failure path logs WARNING code+type only, never raises.'''
  calls = []
  fake = _fake_mlflow(monkeypatch, calls)

  def _boom(run_name=None):
    '''Raise an error.'''
    raise RuntimeError('mlflow down SECRET-MARKER-xyz')

  fake.start_run = _boom
  monkeypatch.setenv('GIT_SHA', 'abc123')
  from forecast_ml_framework.hooks import ForecastHooks
  h = ForecastHooks()
  with caplog.at_level(logging.WARNING):
    h.before_pipeline_run({'pipeline_name': 'modeling'}, None, None)
  assert h.warning_count == 1
  warns = [r for r in caplog.records
           if r.levelno >= logging.WARNING]
  assert warns
  # R-02: no traceback and no raw exception text in logs.
  assert not any(r.exc_info for r in warns)
  assert 'SECRET-MARKER-xyz' not in caplog.text
  assert 'mlflow down' not in caplog.text


def test_hooks_secret_exception_never_in_caplog(monkeypatch, caplog):
  '''Secret-bearing exception never reaches hook logs.'''
  calls = []
  fake = _fake_mlflow(monkeypatch, calls)
  secret = 'SECRET-MARKER-hooks-12345'

  def _boom(run_name=None):
    '''Raise an error.'''
    raise RuntimeError(f'token={secret} /secret/dir/x')

  fake.start_run = _boom
  monkeypatch.setenv('GIT_SHA', 'abc123')
  from forecast_ml_framework.hooks import ForecastHooks
  h = ForecastHooks()
  with caplog.at_level(logging.DEBUG):
    h.before_pipeline_run({'pipeline_name': 'modeling'}, None, None)
  assert secret not in caplog.text
  assert '/secret' not in caplog.text
  assert not any(r.exc_info for r in caplog.records)


def test_repro_seed_fallback_from_modeling_params(monkeypatch):
  '''No run_params seed: modeling_params random_state from config.'''
  calls = []
  _fake_mlflow(monkeypatch, calls)
  monkeypatch.setenv('GIT_SHA', 'abc123')
  shipped = {
      'parameters': {
          'modeling_params': {'random_state': 7},
          'mlflow_params': {
              'tracking_uri': 'sqlite:////tmp/x.db',
              'experiment_name': 'exp',
          },
      },
  }
  monkeypatch.setattr(
      'forecast_ml_framework.core.config_schema.load_config',
      lambda env_dir='base': shipped)
  from forecast_ml_framework.hooks import ForecastHooks
  h = ForecastHooks()
  h.before_pipeline_run({'pipeline_name': 'modeling'}, None, None)
  merged = {}
  for d in _param_dicts(calls):
    merged.update(d)
  assert merged.get('seed') == '7'


def test_repro_seed_fallback_from_global_params(monkeypatch):
  '''No run_params seed: global_params random_state from config.'''
  calls = []
  _fake_mlflow(monkeypatch, calls)
  monkeypatch.setenv('GIT_SHA', 'abc123')
  shipped = {
      'parameters': {
          'global_params': {'random_state': 99},
      },
  }
  monkeypatch.setattr(
      'forecast_ml_framework.core.config_schema.load_config',
      lambda env_dir='base': shipped)
  from forecast_ml_framework.hooks import ForecastHooks
  h = ForecastHooks()
  h.before_pipeline_run({'pipeline_name': 'modeling'}, None, None)
  merged = {}
  for d in _param_dicts(calls):
    merged.update(d)
  assert merged.get('seed') == '99'


def test_kedro_hook_manager_registers_settings_hooks_instance():
  '''Kedro HookManager registers ``settings.HOOKS`` without error.

  ``settings.HOOKS`` is a ``ForecastHooks`` instance (Kedro 1.7
  requires instances), so ``_register_hooks`` (the exact call
  ``KedroSession.__init__`` makes) succeeds and every hook
  spec exposes a hookimpl from the registered instance.
  '''
  from kedro.framework.hooks.manager import (
    _create_hook_manager,
    _register_hooks,
  )
  from forecast_ml_framework.hooks import ForecastHooks
  from forecast_ml_framework.settings import HOOKS
  assert len(HOOKS) == 1
  assert isinstance(HOOKS[0], ForecastHooks)
  manager = _create_hook_manager()
  _register_hooks(manager, HOOKS)
  for spec in ('before_pipeline_run', 'after_pipeline_run',
               'on_pipeline_error', 'before_node_run',
               'after_node_run'):
    hookimpls = getattr(manager.hook, spec).get_hookimpls()
    assert [impl.function.__name__ for impl in hookimpls] == [spec]


def test_kedro_hook_manager_discovers_marked_instance_hooks():
  '''Kedro HookManager discovers all 5 marked instance hooks.

  pluggy only exposes ``@hook_impl``-marked methods; with the
  markers in ``hooks.py`` an instance yields exactly one
  hookimpl per hook spec and hooks can fire.
  '''
  from kedro.framework.hooks.manager import (
    _create_hook_manager,
    _register_hooks,
  )
  from forecast_ml_framework.hooks import ForecastHooks
  manager = _create_hook_manager()
  _register_hooks(manager, (ForecastHooks(),))
  for spec in ('before_pipeline_run', 'after_pipeline_run',
               'on_pipeline_error', 'before_node_run',
               'after_node_run'):
    hookimpls = getattr(manager.hook, spec).get_hookimpls()
    assert [impl.function.__name__ for impl in hookimpls] == [spec]
