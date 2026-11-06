"""MLflow run-lifecycle hooks (vz-high Task 11)."""

import sys
import types


def test_hooks_noop_without_mlflow(monkeypatch):
  from forecast_ml_framework.hooks import ForecastHooks
  monkeypatch.setitem(sys.modules, 'mlflow', None)
  h = ForecastHooks()
  h.before_pipeline_run(None, {}, None)  # must not raise
  h.on_error(None, {}, None)  # must not raise


def test_hooks_lifecycle_with_fake_mlflow(monkeypatch):
  calls = []
  fake = types.ModuleType('mlflow')
  fake.start_run = lambda run_name=None: calls.append(('start', run_name))
  fake.end_run = lambda status=None: calls.append(('end', status))
  fake.set_tag = lambda k, v: calls.append(('tag', k, v))
  fake.log_params = lambda d: calls.append(('params', d))
  fake.log_metrics = lambda d: None
  monkeypatch.setitem(sys.modules, 'mlflow', fake)
  from forecast_ml_framework.hooks import ForecastHooks
  h = ForecastHooks()
  h.before_pipeline_run(None, {'pipeline_name': 'modeling'}, None)
  h.after_pipeline_run(None, {}, None)
  assert calls[0][0] == 'start'
  assert calls[0][1].startswith('modeling-')
  assert ('end', 'FINISHED') in calls


def test_hooks_kedro17_signatures(monkeypatch):
  calls = []
  fake = types.ModuleType('mlflow')
  fake.start_run = lambda run_name=None: calls.append(('start', run_name))
  fake.end_run = lambda status=None: calls.append(('end', status))
  fake.set_tag = lambda k, v: calls.append(('tag', k, v))
  fake.log_params = lambda d: calls.append(('params', d))
  monkeypatch.setitem(sys.modules, 'mlflow', fake)
  from forecast_ml_framework.hooks import ForecastHooks
  h = ForecastHooks()
  h.before_pipeline_run({'pipeline_name': 'scoring'}, None, None)
  h.after_pipeline_run({}, {}, None, None)
  # end_run is guarded: no active run after FINISHED, so a new
  # run must start before an error can end it as FAILED.
  h.before_pipeline_run({'pipeline_name': 'scoring'}, None, None)
  h.on_pipeline_error(RuntimeError('boom'), {}, None, None)
  assert calls[0][0] == 'start'
  assert ('end', 'FINISHED') in calls
  assert ('end', 'FAILED') in calls


def _fake_mlflow(monkeypatch, calls):
  fake = types.ModuleType('mlflow')
  state = {'active': None}

  class _Info:
    run_id = 'run-123'

  class _Run:
    info = _Info()

  def _start(run_name=None):
    calls.append(('start', run_name))
    state['active'] = _Run()

  def _end(status=None):
    calls.append(('end', status))
    state['active'] = None

  fake.start_run = _start
  fake.end_run = _end
  fake.active_run = lambda: state['active']
  fake.set_tag = lambda k, v: calls.append(('tag', k, v))
  fake.log_params = lambda d: calls.append(('params', d))
  fake.log_metrics = lambda d: calls.append(('metrics', d))
  monkeypatch.setitem(sys.modules, 'mlflow', fake)
  return fake


def test_hooks_captures_run_id_from_active_run(monkeypatch):
  calls = []
  _fake_mlflow(monkeypatch, calls)
  from forecast_ml_framework.hooks import ForecastHooks
  h = ForecastHooks()
  assert h.run_id is None
  h.before_pipeline_run({'pipeline_name': 'modeling'}, None, None)
  assert h.run_id == 'run-123'


def test_hooks_end_run_guarded_when_no_active_run(monkeypatch):
  calls = []
  _fake_mlflow(monkeypatch, calls)
  from forecast_ml_framework.hooks import ForecastHooks
  h = ForecastHooks()
  h.after_pipeline_run({}, {}, None, None)
  h.on_pipeline_error(RuntimeError('x'), {}, None, None)
  assert ('end', 'FINISHED') not in calls
  assert ('end', 'FAILED') not in calls


def test_hooks_node_params_distinct_per_node(monkeypatch):
  calls = []
  _fake_mlflow(monkeypatch, calls)
  from forecast_ml_framework.hooks import ForecastHooks

  class _Node:
    def __init__(self, name):
      self.name = name

  h = ForecastHooks()
  h.before_pipeline_run({'pipeline_name': 'modeling'}, None, None)
  h.before_node_run(_Node('split_data_node'), None, {}, False, 'run-123')
  h.before_node_run(_Node('split_data_node'), None, {}, False, 'run-123')
  param_dicts = [c[1] for c in calls if c[0] == 'params']
  assert len(param_dicts) == 2
  keys = [next(iter(d)) for d in param_dicts]
  assert keys[0] != keys[1]
  assert keys[0].startswith('split_data_node')


def test_hooks_sanitizes_pipeline_name_and_tag(monkeypatch):
  calls = []
  _fake_mlflow(monkeypatch, calls)
  from forecast_ml_framework.hooks import ForecastHooks

  class _Node:
    name = 'evil\x00;DROP TABLE\x1f' + 'x' * 200

  h = ForecastHooks()
  h.before_pipeline_run(
    {'pipeline_name': 'bad\x00;name' + 'y' * 200}, None, None)
  start_name = calls[0][1]
  assert '\x00' not in start_name
  assert ';' not in start_name
  assert len(start_name) <= 130  # name capped at 100 + timestamp suffix
  h.before_node_run(_Node(), None, {}, False, None)
  tags = [c for c in calls if c[0] == 'tag']
  assert tags
  assert '\x00' not in tags[-1][2]
  assert ';' not in tags[-1][2]
  assert len(tags[-1][2]) <= 100


def test_hooks_warning_count_and_metric_on_failure(monkeypatch):
  calls = []
  fake = _fake_mlflow(monkeypatch, calls)

  def _boom(run_name=None):
    raise RuntimeError('mlflow down')

  fake.start_run = _boom
  from forecast_ml_framework.hooks import ForecastHooks
  h = ForecastHooks()
  h.before_pipeline_run({'pipeline_name': 'modeling'}, None, None)
  assert getattr(h, 'warning_count', 0) == 1
  metrics = [c[1] for c in calls if c[0] == 'metrics']
  assert metrics and list(metrics[-1].values()) == [1]


def test_hooks_failed_on_error(monkeypatch):
  import sys as _sys
  import types as _types

  calls = []
  fake = _types.ModuleType('mlflow')
  fake.start_run = lambda run_name=None: None
  fake.end_run = lambda status=None: calls.append(status)
  fake.set_tag = lambda k, v: None
  fake.log_params = lambda d: None
  monkeypatch.setitem(_sys.modules, 'mlflow', fake)
  from forecast_ml_framework.hooks import ForecastHooks
  h = ForecastHooks()
  h.before_pipeline_run(None, {'pipeline_name': 'modeling'}, None)
  h.on_error(None, {}, None)
  assert 'FAILED' in calls
