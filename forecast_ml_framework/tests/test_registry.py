#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for forecast_ml_framework.registry.model_registry.'''
# pylint: disable=import-outside-toplevel
# pylint: disable=unused-variable, invalid-name

import logging
import sys
import types
import uuid
from types import SimpleNamespace

import pytest


def _fake_mlflow(monkeypatch):
  '''Install a fake mlflow module tree and return its parts.'''
  fake = types.ModuleType('mlflow')
  fake.__path__ = []
  sklearn = types.ModuleType('mlflow.sklearn')
  tracking = types.ModuleType('mlflow.tracking')

  calls = {'log_model': [], 'register_model': [],
           'set_tracking_uri': [], 'set_registry_uri': [],
           'log_params': [], 'log_metrics': [], 'set_tags': []}
  state = {'run_id': 'run-123', 'version': '1'}

  class FakeRun:
    '''Context manager mimicking an mlflow run.'''

    def __init__(self):
      self.info = SimpleNamespace(run_id=state['run_id'])

    def __enter__(self):
      return self

    def __exit__(self, *exc):
      return False

  fake.start_run = lambda **kw: FakeRun()
  fake.log_params = lambda params: calls['log_params'].append(params)
  fake.log_metrics = lambda metrics: calls['log_metrics'].append(
    metrics)
  fake.set_tags = lambda tags: calls['set_tags'].append(tags)
  fake.set_tracking_uri = lambda uri: calls[
    'set_tracking_uri'].append(uri)
  fake.set_registry_uri = lambda uri: calls[
    'set_registry_uri'].append(uri)

  def _log_model(sk_model=None, name=None, registered_model_name=None,
                 tags=None, **kw):
    calls['log_model'].append({
      'sk_model': sk_model, 'name': name,
      'registered_model_name': registered_model_name, 'tags': tags})
    return SimpleNamespace(
      model_uri='models:/%s/%s' % (registered_model_name,
                                   state['version']),
      run_id=state['run_id'], version=state['version'])

  sklearn.log_model = _log_model
  sklearn.load_model = lambda uri: ('loaded', uri)

  def _register_model(model_uri, name, **kw):
    calls['register_model'].append((model_uri, name))
    return SimpleNamespace(version=state['version'],
                           run_id=state['run_id'])

  fake.register_model = _register_model
  fake.sklearn = sklearn

  class FakeClient:
    '''Minimal MlflowClient fake recording alias operations.'''

    def __init__(self, tracking_uri=None, registry_uri=None):
      self.tracking_uri = tracking_uri
      self.registry_uri = registry_uri
      self.aliases = {}
      self.versions = [SimpleNamespace(
        name='my-model', version='1', run_id='run-123',
        source='models:/my-model/1')]
      FakeClient.last = self

    def set_registered_model_alias(self, name, alias, version):
      self.aliases[(name, alias)] = str(version)

    def get_model_version_by_alias(self, name, alias):
      version = self.aliases.get((name, alias))
      if version is None:
        raise KeyError('no alias %s for %s' % (alias, name))
      return SimpleNamespace(name=name, alias=alias,
                             version=version,
                             run_id='run-123')

    def search_model_versions(self, filter_string=None, **kw):
      if filter_string and 'run_id' in filter_string:
        return list(self.versions)
      return list(self.versions)

    def search_registered_models(self, filter_string=None, **kw):
      return [SimpleNamespace(name='my-model')]

    def get_experiment_by_name(self, name):
      return SimpleNamespace(experiment_id='exp-1', name=name)

    def search_runs(self, experiment_ids, **kw):
      return [
        SimpleNamespace(
          info=SimpleNamespace(run_id='run-b', status='FINISHED'),
          data=SimpleNamespace(
            metrics={'rmse': 0.5}, params={'lr': '0.1'},
            tags={})),
        SimpleNamespace(
          info=SimpleNamespace(run_id='run-a', status='FINISHED'),
          data=SimpleNamespace(
            metrics={'rmse': 0.2}, params={'lr': '0.2'},
            tags={})),
      ]

  tracking.MlflowClient = FakeClient
  fake.tracking = tracking
  monkeypatch.setitem(sys.modules, 'mlflow', fake)
  monkeypatch.setitem(sys.modules, 'mlflow.sklearn', sklearn)
  monkeypatch.setitem(sys.modules, 'mlflow.tracking', tracking)
  return fake, sklearn, FakeClient, calls


def test_import_without_mlflow_is_lazy():
  '''Importing the module must not import mlflow.'''
  from forecast_ml_framework.registry import model_registry
  assert model_registry is not None
  # Lazy: a registry can be constructed without mlflow present.
  registry = model_registry.ModelRegistry(
    {'tracking_uri': 'sqlite:////tmp/x.db'})
  assert registry.tracking_uri == 'sqlite:////tmp/x.db'


def test_missing_mlflow_raises_clear_error(monkeypatch):
  '''Absent mlflow raises ImportError naming mlflow.'''
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  monkeypatch.setitem(sys.modules, 'mlflow', None)
  registry = ModelRegistry({'tracking_uri': 'mlruns'})
  registry._mlflow = None
  registry._client = None
  registry._sklearn = None
  with pytest.raises(ImportError, match='mlflow'):
    registry.register_model(object(), 'my-model')


@pytest.mark.parametrize('alias', [
  'production', 'staging', 'challenger', 'canary', 'v2', 'a1',
])
def test_validate_alias_accepts_good(alias):
  from forecast_ml_framework.registry.model_registry import (
    validate_alias)
  assert validate_alias(alias) == alias


@pytest.mark.parametrize('alias', [
  '', 'Production', 'PRODUCTION', 'prod uction', 'prod!',
  '-bad', 'bad alias', 'bad/alias', None, 123,
  'production\n', 'production\ninjected', 'production\r\n',
  'production\nstaging', 'staging\n',
])
def test_validate_alias_rejects_bad(alias):
  from forecast_ml_framework.registry.model_registry import (
    validate_alias)
  with pytest.raises(ValueError):
    validate_alias(alias)


def test_known_aliases_includes_champion():
  from forecast_ml_framework.registry.model_registry import (
    KNOWN_ALIASES)
  assert 'champion' in KNOWN_ALIASES
  assert 'production' in KNOWN_ALIASES


@pytest.mark.parametrize('name', [
  '../../etc/passwd', '../model', 'a/b', 'a\\b', 'a b', 'a\nb',
  'Bad Name!', 'BadName', 'model.name', 'model@production',
  'model:1', '', '   ', None, 123, 'a\n', '-bad', '_bad',
])
def test_validate_model_name_rejects_bad(name):
  from forecast_ml_framework.registry.model_registry import (
    validate_model_name)
  with pytest.raises(ValueError):
    validate_model_name(name)


@pytest.mark.parametrize('name', [
  'my-model', 'my_model', 'model1', 'a', 'v2-final',
])
def test_validate_model_name_accepts_good(name):
  from forecast_ml_framework.registry.model_registry import (
    validate_model_name)
  assert validate_model_name(name) == name


@pytest.mark.parametrize('uri', [
  'sqlite:////tmp/mlflow.db', 'sqlite:///mlflow.db',
  'http://localhost:5000', 'https://tracking.example.com',
  'http://127.0.0.1:5000', 'mlruns', './mlruns', '',
])
def test_validate_registry_uri_accepts_good(uri):
  from forecast_ml_framework.registry.model_registry import (
    validate_registry_uri)
  assert validate_registry_uri(uri) == uri


@pytest.mark.parametrize('uri', [
  'file:///etc/passwd', 'file://localhost/etc/passwd',
  'javascript:alert(1)', 'data:text/html,hi', 'ftp://example.com/x',
  'http://user:pass@localhost:5000',
  'https://user@localhost:5000', 'sqlite://user:pass@/tmp/x.db',
  'http://169.254.169.254/latest/meta-data/',
  'https://169.254.169.254/', 'http://169.254.1.2/',
  'http://metadata.google.internal/computeMetadata/v1/',
  'http://metadata/', 'http://100.100.100.200/',
  'http://localhost:5000\ninjected', ' http://localhost:5000',
  'http://localhost:5000 ', 123,
])
def test_validate_registry_uri_rejects_bad(uri):
  from forecast_ml_framework.registry.model_registry import (
    validate_registry_uri)
  with pytest.raises(ValueError):
    validate_registry_uri(uri)


@pytest.mark.parametrize('uri', [
  'http://0xA9FEA9FE/',
  'http://2852039166/',
  'http://0xa9.0xfe.0xa9.0xfe/',
  'http://0251.0376.0251.0376/',
  '//169.254.169.254',
  '//169.254.169.254/latest/meta-data/',
  'http://metadata%2egoogle.internal/',
  'https://metadata%2egoogle.internal/computeMetadata/v1/',
  'http://0x7f000001/',
  'http://2130706433/',
  'http://0177.0.0.1/',
  'http://0x7f.0.0.1/',
])
def test_validate_registry_uri_rejects_encoded_hosts(uri):
  '''Encoded metadata/loopback hosts must be rejected.

  Canonical 127.0.0.1 is allowed by design (see good URIs) but
  non-canonical numeric encodings of it are not: str(ip) != host.
  '''
  from forecast_ml_framework.registry.model_registry import (
    validate_registry_uri)
  with pytest.raises(ValueError):
    validate_registry_uri(uri)


@pytest.mark.parametrize('uri', [
  'http://169\u3002254\u3002169\u3002254/',
  'http://169\uff0e254\uff0e169\uff0e254/',
  'http://169\uff61254\uff61169\uff61254/',
  'http://169\u2024254\u2024169\u2024254/',
  '//169\u3002254\u3002169\u3002254',
  'http://metadata\u3002google\u3002internal/',
  'http://metadata\uff0egoogle\uff0einternal/',
  'http://metadata\uff61google\uff61internal/',
  'http://metadata\u2024google\u2024internal/',
  'http://127\u30020\u30020\u30021/',
  'http://127\uff0e0\uff0e0\uff0e1/',
  'http://127\uff610\uff610\uff611/',
  'http://127\u20240\u20240\u20241/',
])
def test_validate_registry_uri_rejects_unicode_dot_hosts(uri):
  '''Unicode-dot hosts resolving to metadata/loopback are rejected.

  getaddrinfo normalises U+3002/U+FF0E/U+FF61/U+2024 to '.',
  so 169.254.169.254 / metadata.google.internal / 127.0.0.1
  written with those codepoints must not bypass host checks.
  '''
  from forecast_ml_framework.registry.model_registry import (
    validate_registry_uri)
  with pytest.raises(ValueError):
    validate_registry_uri(uri)


def test_model_registry_rejects_bad_tracking_uri():
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  with pytest.raises(ValueError):
    ModelRegistry({'tracking_uri': 'file:///tmp/x.db'})
  with pytest.raises(ValueError):
    ModelRegistry({'tracking_uri': 'http://user:pass@host:5000'})


def test_find_version_only_swallows_mlflow_exception(
    monkeypatch, caplog):
  _fake_mlflow(monkeypatch)
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  registry = ModelRegistry({'tracking_uri': 'sqlite:////tmp/r.db'})

  class MlflowException(Exception):
    pass

  MlflowException.__module__ = 'mlflow.exceptions'

  class RaisingClient:
    def __init__(self, exc):
      self._exc = exc

    def search_model_versions(self, filter_string=None, **kw):
      raise self._exc

  # MlflowException (by type/module name) is swallowed + logged.
  exc = MlflowException('boom')
  registry._client = RaisingClient(exc)
  with caplog.at_level(logging.WARNING):
    assert registry._find_version('my-model', 'run-1') is None
  assert caplog.records
  # Non-Mlflow errors must propagate, not be hidden.
  registry._client = RaisingClient(RuntimeError('hidden failure'))
  with pytest.raises(RuntimeError, match='hidden failure'):
    registry._find_version('my-model', 'run-1')
  # ValueError from validation-style errors also propagates.
  registry._client = RaisingClient(ValueError('bad input'))
  with pytest.raises(ValueError, match='bad input'):
    registry._find_version('my-model', 'run-1')


def test_register_alias_load_roundtrip_mock(monkeypatch):
  '''register -> alias -> get/load roundtrip with fake mlflow.'''
  fake, sklearn, _client_cls, calls = _fake_mlflow(monkeypatch)
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  registry = ModelRegistry({
    'tracking_uri': 'sqlite:////tmp/reg.db',
    'registry_uri': 'sqlite:////tmp/reg.db',
    'experiment_name': 'forecast-ml-framework',
  })
  model = object()
  result = registry.register_model(
    model, 'my-model', alias='production',
    metrics={'rmse': 0.2}, params={'lr': 0.1},
    tags={'env': 'test'})
  assert result['run_id'] == 'run-123'
  assert result['uri'] == 'models:/my-model/1'
  assert result['model_uri'] == result['uri']
  assert result['version'] == '1'
  assert calls['log_model'][0]['sk_model'] is model
  assert calls['log_model'][0]['registered_model_name'] == 'my-model'
  assert calls['log_params'] == [{'lr': 0.1}]
  assert calls['log_metrics'] == [{'rmse': 0.2}]
  assert 'sqlite:////tmp/reg.db' in calls['set_tracking_uri']
  # Alias was set and can be read back (roundtrip).
  version = registry.get_model_by_alias('my-model', 'production')
  assert version.version == '1'
  assert version.alias == 'production'
  assert registry._load_sklearn().load_model(
    version.source if hasattr(version, 'source') else result['uri']
  ) == ('loaded', 'models:/my-model/1')


def test_register_rejects_bad_alias(monkeypatch):
  _fake_mlflow(monkeypatch)
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  registry = ModelRegistry({'tracking_uri': 'mlruns'})
  with pytest.raises(ValueError):
    registry.register_model(object(), 'my-model', alias='Bad Alias')
  with pytest.raises(ValueError):
    registry.register_model(object(), '', alias='production')


def test_promote_alias_by_run_id_mock(monkeypatch):
  fake, _sk, _cls, _calls = _fake_mlflow(monkeypatch)
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  registry = ModelRegistry({'tracking_uri': 'sqlite:////tmp/r.db'})
  registry.register_model(object(), 'my-model', alias='staging')
  promoted = registry.promote_alias(
    'my-model', 'production', 'run-123')
  assert str(promoted) == '1'
  version = registry.get_model_by_alias('my-model', 'production')
  assert version.version == '1'
  with pytest.raises(ValueError):
    registry.promote_alias('my-model', 'Bad!', 'run-123')


def test_list_models_mock(monkeypatch):
  _fake_mlflow(monkeypatch)
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  registry = ModelRegistry({'tracking_uri': 'mlruns'})
  models = registry.list_models('my-model')
  assert len(models) == 1
  assert models[0].version == '1'


def test_compare_runs_sorted_by_metric_mock(monkeypatch):
  _fake_mlflow(monkeypatch)
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  registry = ModelRegistry({
    'tracking_uri': 'mlruns', 'experiment_id': 'exp-1'})
  ranked = registry.compare_runs(metric='rmse')
  assert [row['run_id'] for row in ranked] == ['run-a', 'run-b']
  assert ranked[0]['metric_value'] == pytest.approx(0.2)
  assert ranked[0]['champion'] is True
  assert ranked[1]['challenger'] is True


def test_compare_runs_higher_better(monkeypatch):
  _fake_mlflow(monkeypatch)
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  registry = ModelRegistry({
    'tracking_uri': 'mlruns', 'experiment_id': 'exp-1'})
  ranked = registry.compare_runs(metric='rmse', higher_better=True)
  assert ranked[0]['run_id'] == 'run-b'


def test_config_from_nested_mlflow_params():
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  registry = ModelRegistry({
    'mlflow_params': {'tracking_uri': 'sqlite:////tmp/n.db',
                      'registry_uri': 'sqlite:////tmp/n.db',
                      'experiment_name': 'exp-x'}})
  assert registry.tracking_uri == 'sqlite:////tmp/n.db'
  assert registry.registry_uri == 'sqlite:////tmp/n.db'
  assert registry.experiment_name == 'exp-x'


@pytest.fixture(autouse=False)
def _reset_mlflow_state():
  '''Reset global mlflow state around real-mlflow tests.'''
  pytest.importorskip('mlflow')
  import mlflow
  try:
    yield
  finally:
    for reset in (mlflow.set_tracking_uri, mlflow.set_registry_uri):
      try:
        reset(None)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        logging.getLogger(__name__).warning(
          'mlflow state reset failed: %s', exc)


@pytest.mark.serial
@pytest.mark.usefixtures('_reset_mlflow_state')
def test_real_mlflow_roundtrip_sqlite(tmp_path, monkeypatch):
  '''Real mlflow sqlite registry roundtrip (no server needed).'''
  pytest.importorskip('mlflow')
  pytest.importorskip('sklearn')
  monkeypatch.chdir(tmp_path)
  from sklearn.linear_model import LinearRegression
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  db = 'sqlite:///' + str(tmp_path / 'mlflow.db')
  experiment_name = f'test-registry-{uuid.uuid4().hex[:12]}'
  registry = ModelRegistry({
    'tracking_uri': db, 'registry_uri': db,
    'experiment_name': experiment_name})
  model = LinearRegression().fit([[0.0], [1.0]], [0.0, 1.0])
  result = registry.register_model(
    model, 'my-model', alias='production',
    metrics={'rmse': 0.0}, params={'fit_intercept': True},
    tags={'env': 'test'})
  assert result['run_id']
  assert result['uri']
  version = registry.get_model_by_alias('my-model', 'production')
  assert version is not None
  models = registry.list_models('my-model')
  assert len(models) >= 1
  ranked = registry.compare_runs(metric='rmse', higher_better=False)
  assert ranked and ranked[0]['run_id'] == result['run_id']
