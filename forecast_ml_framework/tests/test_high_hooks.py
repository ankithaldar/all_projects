"""MLflow run-lifecycle hooks (vz-high Task 11)."""

import sys
import types

import pytest


def test_hooks_noop_without_mlflow(monkeypatch):
  '''No-mlflow calls are no-ops keeping counters unchanged.

  Fails if any hook raises on missing mlflow or records a
  warning/run_id while mlflow is unavailable.
  '''
  from forecast_ml_framework.hooks import ForecastHooks
  monkeypatch.setitem(sys.modules, 'mlflow', None)
  h = ForecastHooks()
  h.before_pipeline_run(None, {}, None)  # must not raise
  h.on_error(None, {}, None)  # must not raise
  assert h.warning_count == 0
  assert h.hook_warning_count == 0
  assert h.run_id is None


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
  '''Distinct nodes log distinct real scalar params.

  Fails if node params use a timestamp key, ignore real
  scalar inputs, or log identical dicts for different nodes.
  '''
  calls = []
  _fake_mlflow(monkeypatch, calls)
  from forecast_ml_framework.hooks import ForecastHooks

  class _Node:
    def __init__(self, name):
      self.name = name

  h = ForecastHooks()
  h.before_pipeline_run({'pipeline_name': 'modeling'}, None, None)
  h.before_node_run(_Node('split_data_node'), None, {}, False, 'run-123')
  h.before_node_run(
      _Node('split_data_node'), None, {'test_size': 0.2}, False, 'run-123')
  h.before_node_run(
      _Node('train_node'), None, {'lr': 0.1}, False, 'run-123')
  param_dicts = [c[1] for c in calls if c[0] == 'params']
  # First dict is the repro dict from before_pipeline_run;
  # empty inputs above must not add a node param dict.
  node_dicts = param_dicts[1:]
  assert len(node_dicts) == 2
  assert node_dicts[0] == {'test_size': '0.2'}
  assert node_dicts[1] == {'lr': '0.1'}
  assert set(node_dicts[0]) != set(node_dicts[1])
  for d in node_dicts:
    for key in d:
      assert not key.startswith('split_data_node')
      assert not key.startswith('train_node')


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


@pytest.mark.parametrize('value,expected', [
  (None, 'kedro'),
  ('', 'kedro'),
  ('   ', 'kedro'),
  (42, '42'),
  ('\x00a\x1f', 'a'),
  ('a;b', 'a_b'),
  ('a/b', 'a_b'),
])
def test_sanitize_name_cases(value, expected):
  '''Sanitize maps empty/disallowed input to safe names.

  Fails if sanitize_name returns empty output, keeps
  separators, or stringifies numbers incorrectly.
  '''
  from forecast_ml_framework.hooks import sanitize_name
  assert sanitize_name(value) == expected


def test_sanitize_name_length_boundaries():
  '''Sanitize caps at 100 chars (99/100/101 boundary).

  Fails if the length cap is off by one in either direction.
  '''
  from forecast_ml_framework.hooks import sanitize_name
  assert len(sanitize_name('x' * 99)) == 99
  assert len(sanitize_name('x' * 100)) == 100
  assert len(sanitize_name('x' * 101)) == 100


def test_sanitize_name_cap_then_strip():
  '''Cap-then-strip drops trailing space and overflow.

  Fails if a trailing space survives or overflow chars
  beyond 100 appear in the result.
  '''
  from forecast_ml_framework.hooks import sanitize_name
  assert sanitize_name('x' * 99 + ' ') == 'x' * 99
  result = sanitize_name('x' * 100 + 'y')
  assert result == 'x' * 100
  assert 'y' not in result


def test_sanitize_name_default_override_and_alias():
  '''Default override wins; sanitize aliases sanitize_name.

  Fails if the default override is ignored or the alias
  points at a different function.
  '''
  from forecast_ml_framework.hooks import (
    sanitize,
    sanitize_name,
  )
  assert sanitize_name(None, default='zzz') == 'zzz'
  assert sanitize is sanitize_name


def test_run_params_from_kwargs_keys():
  '''run_params/params/context kwargs yield the dict.

  Fails if any documented kwargs key stops resolving.
  '''
  from forecast_ml_framework.hooks import _run_params_from
  assert _run_params_from((), {'run_params': {'a': 1}}) == {'a': 1}
  assert _run_params_from((), {'params': {'a': 1}}) == {'a': 1}
  assert _run_params_from((), {'context': {'a': 1}}) == {'a': 1}


def test_run_params_from_non_dict_and_fallback():
  '''Non-dict kwargs ignored; empty falls back to {}.

  Fails if non-dict kwargs raise or the fallback is not {}.
  '''
  from forecast_ml_framework.hooks import _run_params_from
  assert _run_params_from((), {'run_params': 'not-a-dict'}) == {}
  assert _run_params_from((), {}) == {}
  assert _run_params_from(({'x': 1},), {}) == {'x': 1}


def test_hooks_no_mlflow_early_returns(monkeypatch):
  '''All hook methods early-return without mlflow.

  Fails if any method raises, warns, or records a run_id
  while mlflow is unavailable.
  '''
  from forecast_ml_framework.hooks import ForecastHooks
  monkeypatch.setitem(sys.modules, 'mlflow', None)
  h = ForecastHooks()
  h.before_pipeline_run({'pipeline_name': 'x'}, None, None)
  h.after_pipeline_run({}, {}, None, None)
  h.before_node_run(None, None, {}, False, None)
  h.after_node_run(None, None, {}, {}, False, None)
  h.on_pipeline_error(RuntimeError('x'), {}, None, None)
  h.on_error(RuntimeError('x'), {}, None, None)
  assert h.warning_count == 0
  assert h.hook_warning_count == 0
  assert h.run_id is None


@pytest.mark.serial
def test_hooks_real_mlflow_smoke(tmp_path, monkeypatch):
  '''Real-mlflow sqlite hooks smoke (no server needed).

  Fails if ForecastHooks stops writing pipeline and node
  activity to real MLflow, or if warning_count or run_id
  behaviour drifts from the real SDK.
  '''
  pytest.importorskip('mlflow')
  import uuid
  import mlflow
  from mlflow.tracking import MlflowClient
  from forecast_ml_framework.hooks import ForecastHooks

  class _Node:
    def __init__(self, name):
      self.name = name

  monkeypatch.chdir(tmp_path)
  db = 'sqlite:///' + str(tmp_path / 'mlflow.db')
  experiment_name = f'test-hooks-{uuid.uuid4().hex[:12]}'
  hooks = ForecastHooks()
  run_params = {
    'pipeline_name': 'hooks-smoke',
    'mlflow_params': {
      'tracking_uri': db,
      'experiment_name': experiment_name,
    },
  }
  try:
    hooks.before_pipeline_run(run_params, None, None)
    captured_run_id = hooks.run_id
    assert captured_run_id is not None
    assert hooks.warning_count == 0
    node = _Node('hooks-smoke-node')
    hooks.before_node_run(node, None, {'lr': 0.1}, False,
                          captured_run_id)
    hooks.after_node_run(node, None, {'lr': 0.1}, {}, False,
                         captured_run_id)
    hooks.after_pipeline_run({}, {}, None, None)
    assert hooks.warning_count == 0
    assert hooks.hook_warning_count == 0
    client = MlflowClient(tracking_uri=db)
    experiment = client.get_experiment_by_name(experiment_name)
    assert experiment is not None
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.run_id == captured_run_id
    run = client.get_run(captured_run_id)
    assert run.info.status == 'FINISHED'
    assert 'seed' in run.data.params
    assert 'git_sha' in run.data.params
    assert 'data_version' in run.data.params
    assert run.data.params.get('lr') == '0.1'
    assert run.data.tags.get('node') == 'hooks-smoke-node'
  finally:
    try:
      mlflow.set_tracking_uri(None)
    except Exception:  # pylint: disable=broad-exception-caught
      pass
