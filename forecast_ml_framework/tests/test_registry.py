#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for forecast_ml_framework.registry.model_registry.'''
# pylint: disable=import-outside-toplevel
# pylint: disable=unused-variable, invalid-name

import logging
import socket
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
      '''Initialize the test double.'''
      self.info = SimpleNamespace(run_id=state['run_id'])

    def __enter__(self):
      '''Enter the context.'''
      return self

    def __exit__(self, *exc):
      '''Exit the context.'''
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
    '''Log model.'''
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
    '''Register model.'''
    calls['register_model'].append((model_uri, name))
    return SimpleNamespace(version=state['version'],
                           run_id=state['run_id'])

  fake.register_model = _register_model
  fake.sklearn = sklearn

  class FakeClient:
    '''Minimal MlflowClient fake recording alias operations.'''

    def __init__(self, tracking_uri=None, registry_uri=None):
      '''Initialize the test double.'''
      self.tracking_uri = tracking_uri
      self.registry_uri = registry_uri
      self.aliases = {}
      self.versions = [SimpleNamespace(
        name='my-model', version='1', run_id='run-123',
        source='models:/my-model/1')]
      FakeClient.last = self

    def set_registered_model_alias(self, name, alias, version):
      '''Set registered model alias.'''
      self.aliases[(name, alias)] = str(version)

    def get_model_version_by_alias(self, name, alias):
      '''Return model version by alias.'''
      version = self.aliases.get((name, alias))
      if version is None:
        raise KeyError('no alias %s for %s' % (alias, name))
      return SimpleNamespace(name=name, alias=alias,
                             version=version,
                             run_id='run-123')

    def search_model_versions(self, filter_string=None, **kw):
      '''Search model versions.'''
      if filter_string and 'run_id' in filter_string:
        return list(self.versions)
      return list(self.versions)

    def search_registered_models(self, filter_string=None, **kw):
      '''Search registered models.'''
      return [SimpleNamespace(name='my-model')]

    def get_experiment_by_name(self, name):
      '''Get experiment by name.'''
      return SimpleNamespace(experiment_id='exp-1', name=name)

    def search_runs(self, experiment_ids, **kw):
      '''Search runs.'''
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
  '''Validate alias accepts good.'''
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
  '''Validate alias rejects bad.'''
  from forecast_ml_framework.registry.model_registry import (
    validate_alias)
  with pytest.raises(ValueError):
    validate_alias(alias)


def test_known_aliases_includes_champion():
  '''Known aliases includes champion.'''
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
  '''Validate model name rejects bad.'''
  from forecast_ml_framework.registry.model_registry import (
    validate_model_name)
  with pytest.raises(ValueError):
    validate_model_name(name)


@pytest.mark.parametrize('name', [
  'my-model', 'my_model', 'model1', 'a', 'v2-final',
])
def test_validate_model_name_accepts_good(name):
  '''Validate model name accepts good.'''
  from forecast_ml_framework.registry.model_registry import (
    validate_model_name)
  assert validate_model_name(name) == name


@pytest.mark.parametrize('uri', [
  'sqlite:////tmp/mlflow.db', 'sqlite:///mlflow.db',
  'http://localhost:5000', 'https://tracking.example.com',
  'http://127.0.0.1:5000', 'mlruns', './mlruns', '',
])
def test_validate_registry_uri_accepts_good(uri):
  '''Validate registry uri accepts good.'''
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
  '''Validate registry uri rejects bad.'''
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


@pytest.mark.parametrize('uri', [
  'http://[::ffff:169.254.169.254]/',
  'http://[::ffff:a9fe:a9fe]/',
  'http://[::ffff:127.0.0.1]/',
  'http://[::ffff:10.0.0.1]/',
  'http://[::ffff:192.168.1.1]/',
  'http://[::ffff:172.16.0.1]/',
  '//[::ffff:169.254.169.254]/',
  'http://10.0.0.1/',
  'http://192.168.1.1/',
  'http://172.16.0.1/',
  'http://172.31.255.255/',
  'http://127.0.0.2/',
  # Loopback via mapped / 6to4 / teredo forms (except explicit
  # canonical 127.0.0.1 and localhost, which stay allowed per
  # pre-existing good-URI contract)
  'http://[2002:a9fe:a9fe::1]/',
  'http://[2002:7f00:1::1]/',
  'http://[2001:0:4136:e378:8000:63bf:3fff:fdd2]/',
  'http://240.0.0.1/',
  'http://255.255.255.255/',
  'http://0.0.0.0/',
  'http://[fe80::1]/',
  'http://[fd00::1]/',
])
def test_validate_registry_uri_rejects_private_mapped_6to4_teredo(uri):
  '''Mapped/6to4/Teredo/private/reserved SSRF forms are rejected.'''
  from forecast_ml_framework.registry.model_registry import (
    validate_registry_uri)
  with pytest.raises(ValueError):
    validate_registry_uri(uri)


@pytest.mark.parametrize('uri', [
  'http://100.64.0.1/',
  'http://100.127.255.255/',
  'http://100.64.0.1:5000/path',
  '//100.64.0.1',
  'http://[::ffff:100.64.0.1]/',
  'http://[::ffff:100.127.0.1]/',
])
def test_validate_registry_uri_rejects_cgnat_100_64_10(uri):
  '''CGNAT 100.64.0.0/10 is not global (is_private False,
  is_global False) and must be rejected like other non-public
  IPs, except explicit canonical localhost.'''
  from forecast_ml_framework.registry.model_registry import (
    validate_registry_uri)
  with pytest.raises(ValueError):
    validate_registry_uri(uri)


@pytest.mark.parametrize('uri', [
  'http://localhost:5000', 'http://127.0.0.1:5000',
  'http://[::1]:5000',
])
def test_validate_registry_uri_explicit_localhost_allowed(uri):
  '''Explicit localhost (name/canonical) stays allowed.'''
  from forecast_ml_framework.registry.model_registry import (
    validate_registry_uri)
  assert validate_registry_uri(uri) == uri


def test_model_registry_rejects_bad_tracking_uri():
  '''Model registry rejects bad tracking uri.'''
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  with pytest.raises(ValueError):
    ModelRegistry({'tracking_uri': 'file:///tmp/x.db'})
  with pytest.raises(ValueError):
    ModelRegistry({'tracking_uri': 'http://user:pass@host:5000'})


def test_find_version_only_swallows_mlflow_exception(
    monkeypatch, caplog):
  '''Find version only swallows mlflow exception.'''
  _fake_mlflow(monkeypatch)
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  registry = ModelRegistry({'tracking_uri': 'sqlite:////tmp/r.db'})

  class MlflowException(Exception):
    '''Mlflow Exception test double.'''
    pass

  MlflowException.__module__ = 'mlflow.exceptions'

  class RaisingClient:
    '''Raising Client test double.'''
    def __init__(self, exc):
      '''Initialize the test double.'''
      self._exc = exc

    def search_model_versions(self, filter_string=None, **kw):
      '''Search model versions.'''
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
  '''Register rejects bad alias.'''
  _fake_mlflow(monkeypatch)
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  registry = ModelRegistry({'tracking_uri': 'mlruns'})
  with pytest.raises(ValueError):
    registry.register_model(object(), 'my-model', alias='Bad Alias')
  with pytest.raises(ValueError):
    registry.register_model(object(), '', alias='production')


def test_promote_alias_by_run_id_mock(monkeypatch):
  '''Promote alias by run id mock.'''
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
  '''List models mock.'''
  _fake_mlflow(monkeypatch)
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  registry = ModelRegistry({'tracking_uri': 'mlruns'})
  models = registry.list_models('my-model')
  assert len(models) == 1
  assert models[0].version == '1'


def test_compare_runs_sorted_by_metric_mock(monkeypatch):
  '''Compare runs sorted by metric mock.'''
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
  '''Compare runs higher better.'''
  _fake_mlflow(monkeypatch)
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  registry = ModelRegistry({
    'tracking_uri': 'mlruns', 'experiment_id': 'exp-1'})
  ranked = registry.compare_runs(metric='rmse', higher_better=True)
  assert ranked[0]['run_id'] == 'run-b'


def test_config_from_nested_mlflow_params():
  '''Config from nested mlflow params.'''
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


# --- loop4: *.localhost / localhost.localdomain (RED) ---

@pytest.mark.parametrize('uri', [
  'http://foo.localhost:5000',
  'http://sub.localhost/',
  'https://foo.localhost:5000/path',
  '//foo.localhost:5000',
  'http://foo.localhost./',
  'http://localhost.localdomain:5000',
  'http://localhost.localdomain./',
])
def test_validate_registry_uri_rejects_localhost_subdomains(uri):
  '''*.localhost and localhost.localdomain are rejected.

  They resolve to ::1 via getaddrinfo, so only the exact
  canonical ``localhost`` hostname stays allowed by design.
  ``localhost.localdomain`` is the RFC6761 alias for localhost
  and is rejected for the same reason (documented here).
  '''
  from forecast_ml_framework.registry.model_registry import (
    validate_registry_uri)
  with pytest.raises(ValueError):
    validate_registry_uri(uri)


# --- loop5: *.localhost.localdomain bypass (RED) ---

@pytest.mark.parametrize('uri', [
  'http://foo.localhost.localdomain:5000',
  'http://a.b.localhost.localdomain:5000',
  'https://foo.localhost.localdomain/path',
  '//foo.localhost.localdomain:5000',
  'http://foo.localhost.localdomain./',
])
def test_validate_registry_uri_rejects_localhost_localdomain_subdomains(uri):
  '''*.localhost.localdomain is rejected like *.localhost.

  Only the exact ``localhost`` hostname stays allowed; any
  subdomain chaining via ``localhost.localdomain`` resolves
  to loopback and must not bypass the allowance.
  '''
  from forecast_ml_framework.registry.model_registry import (
    validate_registry_uri)
  with pytest.raises(ValueError):
    validate_registry_uri(uri)


# --- W4 registry hardening (cycle-1 MLops 1-5, Sec F3/F4) ---


@pytest.fixture(autouse=True)
def _public_dns_stub(monkeypatch):
  '''Stub DNS to a public IP so URI tests are offline-safe.

  Production code resolves hostnames via ``socket.getaddrinfo``
  and fails closed on DNS errors. Existing good-URI tests use
  ``tracking.example.com`` which is not guaranteed to resolve
  in CI, so the default stub returns a public IP. DNS tests
  below override this stub via ``monkeypatch.setattr`` to
  private/error cases.
  '''
  def _stub(host, port, *args, **kwargs):
    '''Stub.'''
    return [
      (socket.AF_INET, socket.SOCK_STREAM, 6, '',
       ('8.8.8.8', 0)),
    ]
  monkeypatch.setattr(socket, 'getaddrinfo', _stub)
  yield


def _gate_run(run_id, metric, value):
  '''Build one fake MLflow run row for gate tests.'''
  return SimpleNamespace(
    info=SimpleNamespace(run_id=run_id, status='FINISHED'),
    data=SimpleNamespace(
      metrics={metric: value}, params={}, tags={}))


def _setup_incumbent(registry, name, alias, version, run_id):
  '''Set incumbent alias and fake alias lookup.'''
  registry._get_client()
  registry._client.aliases[(name, alias)] = version

  def _by_alias(model_name, model_alias):
    '''By alias.'''
    return SimpleNamespace(
      name=model_name, alias=model_alias,
      version=version, run_id=run_id)

  registry._client.get_model_version_by_alias = _by_alias


def test_find_version_returns_none_on_mismatch(monkeypatch):
  '''Unknown run_id yields no version (no wrong fallback).'''
  _fake_mlflow(monkeypatch)
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  registry = ModelRegistry({'tracking_uri': 'mlruns'})
  assert registry._find_version('my-model', 'run-999') is None


def test_promote_alias_typo_run_id_raises(monkeypatch):
  '''Typo run_id raises instead of promoting wrong version.'''
  _fake_mlflow(monkeypatch)
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  registry = ModelRegistry({'tracking_uri': 'mlruns'})
  registry.register_model(
    object(), 'my-model', alias='staging')
  with pytest.raises(ValueError):
    registry.promote_alias('my-model', 'production', 'run-000')


def test_promote_alias_digit_run_id_not_version(monkeypatch):
  '''Digit-only run_id is not treated as a version string.'''
  _fake_mlflow(monkeypatch)
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  registry = ModelRegistry({'tracking_uri': 'mlruns'})
  registry.register_model(
    object(), 'my-model', alias='staging')
  with pytest.raises(ValueError):
    registry.promote_alias('my-model', 'production', '123')
  assert ('my-model', 'production') not in registry._client.aliases


def test_register_model_default_alias_none(monkeypatch):
  '''Default alias is None: no silent production promotion.'''
  _fake_mlflow(monkeypatch)
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  registry = ModelRegistry({'tracking_uri': 'mlruns'})
  result = registry.register_model(object(), 'my-model')
  assert result['alias'] is None
  assert result['promoted'] is False
  assert registry._get_client().aliases == {}


def test_register_model_explicit_alias_promoted_true(monkeypatch):
  '''Explicit alias promotes and reports promoted=True.'''
  _fake_mlflow(monkeypatch)
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  registry = ModelRegistry({'tracking_uri': 'mlruns'})
  result = registry.register_model(
    object(), 'my-model', alias='production')
  assert result['promoted'] is True
  assert result['alias'] == 'production'
  assert registry._client.aliases[('my-model', 'production')] == '1'


def test_register_model_gate_worse_not_promoted(monkeypatch):
  '''Worse candidate metric leaves alias unchanged.'''
  _fake_mlflow(monkeypatch)
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  registry = ModelRegistry({
    'tracking_uri': 'sqlite:////tmp/r.db',
    'experiment_id': 'exp-1',
    'promote_on_metric': 'rmse',
  })
  _setup_incumbent(
    registry, 'my-model', 'production', '0', 'run-old')
  registry._client.search_runs = lambda experiment_ids, **kw: [
    _gate_run('run-123', 'rmse', 0.5),
    _gate_run('run-old', 'rmse', 0.2),
  ]
  result = registry.register_model(
    object(), 'my-model', alias='production',
    metrics={'rmse': 0.5})
  assert result['promoted'] is False
  assert registry._client.aliases[('my-model', 'production')] == '0'


def test_register_model_gate_better_promoted(monkeypatch):
  '''Better candidate metric promotes the new version.'''
  _fake_mlflow(monkeypatch)
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  registry = ModelRegistry({
    'tracking_uri': 'sqlite:////tmp/r.db',
    'experiment_id': 'exp-1',
    'promote_on_metric': 'rmse',
  })
  _setup_incumbent(
    registry, 'my-model', 'production', '0', 'run-old')
  registry._client.search_runs = lambda experiment_ids, **kw: [
    _gate_run('run-123', 'rmse', 0.1),
    _gate_run('run-old', 'rmse', 0.2),
  ]
  result = registry.register_model(
    object(), 'my-model', alias='production',
    metrics={'rmse': 0.1})
  assert result['promoted'] is True
  assert registry._client.aliases[('my-model', 'production')] == '1'


def test_register_model_gate_higher_better_f1(monkeypatch):
  '''F1 is higher-better: higher candidate wins, lower loses.'''
  _fake_mlflow(monkeypatch)
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  registry = ModelRegistry({
    'tracking_uri': 'sqlite:////tmp/r.db',
    'experiment_id': 'exp-1',
    'promote_on_metric': 'f1',
  })
  _setup_incumbent(
    registry, 'my-model', 'production', '0', 'run-old')
  registry._client.search_runs = lambda experiment_ids, **kw: [
    _gate_run('run-123', 'f1', 0.9),
    _gate_run('run-old', 'f1', 0.8),
  ]
  result = registry.register_model(
    object(), 'my-model', alias='production',
    metrics={'f1': 0.9})
  assert result['promoted'] is True
  assert registry._client.aliases[('my-model', 'production')] == '1'
  # Reset incumbent, then try a worse F1 candidate.
  registry._client.aliases[('my-model', 'production')] = '0'
  registry._client.search_runs = lambda experiment_ids, **kw: [
    _gate_run('run-123', 'f1', 0.7),
    _gate_run('run-old', 'f1', 0.8),
  ]
  result = registry.register_model(
    object(), 'my-model', alias='production',
    metrics={'f1': 0.7})
  assert result['promoted'] is False
  assert registry._client.aliases[('my-model', 'production')] == '0'


def test_register_model_gate_calls_compare_runs(monkeypatch):
  '''Promotion gate must call compare_runs at runtime.'''
  _fake_mlflow(monkeypatch)
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  registry = ModelRegistry({
    'tracking_uri': 'sqlite:////tmp/r.db',
    'experiment_id': 'exp-1',
    'promote_on_metric': 'rmse',
  })
  _setup_incumbent(
    registry, 'my-model', 'production', '0', 'run-old')
  registry._client.search_runs = lambda experiment_ids, **kw: [
    _gate_run('run-123', 'rmse', 0.1),
    _gate_run('run-old', 'rmse', 0.2),
  ]
  calls = []
  original = registry.compare_runs

  def _spy(*args, **kwargs):
    '''Spy.'''
    calls.append((args, kwargs))
    return original(*args, **kwargs)

  registry.compare_runs = _spy
  registry.register_model(
    object(), 'my-model', alias='production',
    metrics={'rmse': 0.1})
  assert calls, 'compare_runs was not called by the gate'


def test_validate_uri_redacts_password_in_error():
  '''Error messages must not echo passwords or query secrets.'''
  from forecast_ml_framework.registry.model_registry import (
    validate_registry_uri)
  uri = (
    'http://user:supersecret@localhost:5000'
    '/path?token=abc123')
  with pytest.raises(ValueError) as excinfo:
    validate_registry_uri(uri)
  message = str(excinfo.value)
  assert 'supersecret' not in message
  assert 'abc123' not in message
  assert 'token' not in message


def test_validate_uri_error_truncates_long_uri():
  '''Long URIs are redacted and truncated in error messages.'''
  from forecast_ml_framework.registry.model_registry import (
    validate_registry_uri)
  long_path = 'a' * 500
  uri = (
    'http://user:supersecret@10.0.0.1/%s'
    '?token=xyz' % long_path)
  with pytest.raises(ValueError) as excinfo:
    validate_registry_uri(uri)
  message = str(excinfo.value)
  assert 'supersecret' not in message
  assert 'xyz' not in message
  assert long_path not in message
  assert len(message) < 400


def test_validate_uri_dns_private_rejected(monkeypatch):
  '''Hostnames resolving to private IPs are rejected.'''
  from forecast_ml_framework.registry.model_registry import (
    validate_registry_uri)

  def _private(host, port, *args, **kwargs):
    '''Private.'''
    return [
      (socket.AF_INET, socket.SOCK_STREAM, 6, '',
       ('10.0.0.8', 0)),
    ]

  monkeypatch.setattr(socket, 'getaddrinfo', _private)
  with pytest.raises(ValueError):
    validate_registry_uri('http://evil.example.com:5000')


def test_validate_uri_dns_public_allowed(monkeypatch):
  '''Hostnames resolving only to public IPs are allowed.'''
  from forecast_ml_framework.registry.model_registry import (
    validate_registry_uri)

  def _public(host, port, *args, **kwargs):
    '''Public.'''
    return [
      (socket.AF_INET, socket.SOCK_STREAM, 6, '',
       ('8.8.8.8', 0)),
    ]

  monkeypatch.setattr(socket, 'getaddrinfo', _public)
  uri = 'http://public.example.com:5000'
  assert validate_registry_uri(uri) == uri


def test_validate_uri_dns_any_private_rejected(monkeypatch):
  '''Any single non-public resolved IP rejects the URI.'''
  from forecast_ml_framework.registry.model_registry import (
    validate_registry_uri)

  def _mixed(host, port, *args, **kwargs):
    '''Mixed.'''
    return [
      (socket.AF_INET, socket.SOCK_STREAM, 6, '',
       ('8.8.8.8', 0)),
      (socket.AF_INET, socket.SOCK_STREAM, 6, '',
       ('169.254.169.254', 0)),
    ]

  monkeypatch.setattr(socket, 'getaddrinfo', _mixed)
  with pytest.raises(ValueError):
    validate_registry_uri('http://mixed.example.com:5000')


def test_validate_uri_dns_failure_fails_closed(monkeypatch):
  '''DNS errors fail closed instead of allowing the URI.'''
  from forecast_ml_framework.registry.model_registry import (
    validate_registry_uri)

  def _fail(host, port, *args, **kwargs):
    '''Fail.'''
    raise socket.gaierror('no such host')

  monkeypatch.setattr(socket, 'getaddrinfo', _fail)
  with pytest.raises(ValueError):
    validate_registry_uri('http://unresolvable.example.com:5000')


# --- X2: C2-F3/C2-F4, MLops R1 gate, R-05 TTL-0 ---


@pytest.mark.parametrize('uri,secret', [
  ('https:user:SUPERSECRET@host.invalid/path?token=QUERYSECRET',
   'SUPERSECRET'),
  ('https:user:SUPERSECRET@host.invalid/path?token=QUERYSECRET',
   'QUERYSECRET'),
  ('sqlite:user:SUPERSECRET@/tmp/x.db?token=QUERYSECRET',
   'SUPERSECRET'),
  ('mlruns?token=QUERYSECRET#frag', 'QUERYSECRET'),
  ('mlruns?token=QUERYSECRET#frag', 'frag'),
  ('https://user:SUPERSECRET@host.invalid/path?token=QUERYSECRET#frag',
   'SUPERSECRET'),
  ('//user:SUPERSECRET@host.invalid/path?token=QUERYSECRET',
   'SUPERSECRET'),
])
def test_safe_uri_redacts_opaque_and_query(uri, secret):
  '''Fails if _safe_uri echoes userinfo, query or fragment from
  opaque or scheme-less URIs.'''
  from forecast_ml_framework.registry.model_registry import _safe_uri
  out = _safe_uri(uri)
  assert secret not in out
  assert '?' not in out and '#' not in out


def _mlflow_exc_class(error_code='INTERNAL_ERROR'):
  '''Build a fake MlflowException class with an error_code.'''

  class MlflowException(Exception):
    '''Fake MlflowException carrying an error_code.'''

    def __init__(self, message):
      '''Initialize the test double.'''
      super().__init__(message)
      self.error_code = error_code

  MlflowException.__module__ = 'mlflow.exceptions'
  return MlflowException


def test_find_version_log_type_only(monkeypatch, caplog):
  '''_find_version logs only the exception type name (C2-F3).'''
  _fake_mlflow(monkeypatch)
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  registry = ModelRegistry({'tracking_uri': 'mlruns'})
  exc_cls = _mlflow_exc_class()

  class RaisingClient:
    '''Raising Client test double.'''
    def search_model_versions(self, filter_string=None, **kw):
      '''Search model versions.'''
      raise exc_cls(
        'boom http://user:SUPERSECRET@host/path?token=QUERYSECRET')

  registry._client = RaisingClient()
  with caplog.at_level(logging.WARNING):
    assert registry._find_version('my-model', 'run-1') is None
  assert 'SUPERSECRET' not in caplog.text
  assert 'QUERYSECRET' not in caplog.text
  assert 'boom' not in caplog.text
  assert 'MlflowException' in caplog.text


def test_register_fallback_log_type_only(monkeypatch, caplog):
  '''register_model fallback logs only the exception type name.'''
  fake, sklearn, _cls, _calls = _fake_mlflow(monkeypatch)
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  exc_cls = _mlflow_exc_class()
  sklearn.log_model = lambda **kw: SimpleNamespace(
    model_uri='models:/my-model/1', version=None, run_id='run-123')

  def _boom(*args, **kwargs):
    '''Raise an error.'''
    raise exc_cls(
      'boom http://user:SUPERSECRET@host/path?token=QUERYSECRET')

  fake.register_model = _boom
  registry = ModelRegistry({'tracking_uri': 'mlruns'})
  with caplog.at_level(logging.WARNING):
    result = registry.register_model(object(), 'my-model')
  assert result['promoted'] is False
  assert 'SUPERSECRET' not in caplog.text
  assert 'QUERYSECRET' not in caplog.text
  assert 'boom' not in caplog.text
  assert 'MlflowException' in caplog.text


@pytest.mark.parametrize('config', [
  {'tracking_uri': 'mlruns', 'experiment_id': 'exp-1',
   'promote_on_metric': 'f1_weighted'},
  {'registry_params': {'tracking_uri': 'mlruns',
                       'experiment_id': 'exp-1',
                       'promote_on_metric': 'f1_weighted'}},
  {'mlflow_params': {'tracking_uri': 'mlruns',
                     'experiment_id': 'exp-1',
                     'promote_on_metric': 'f1_weighted'}},
])
def test_registry_config_promote_on_metric_defaults_gate(
    monkeypatch, config):
  '''Fails if ModelRegistry ignores promote_on_metric in the
  supplied config and promotes unconditionally.'''
  _fake_mlflow(monkeypatch)
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  registry = ModelRegistry(config)
  assert registry.promote_on_metric == 'f1_weighted'
  _setup_incumbent(
    registry, 'my-model', 'production', '0', 'run-old')
  registry._client.search_runs = lambda experiment_ids, **kw: [
    _gate_run('run-123', 'f1_weighted', 0.7),
    _gate_run('run-old', 'f1_weighted', 0.8),
  ]
  result = registry.register_model(
    object(), 'my-model', alias='production',
    metrics={'f1_weighted': 0.7})
  assert result['promoted'] is False
  assert registry._client.aliases[('my-model', 'production')] == '0'


def test_gate_internal_error_fails_closed(monkeypatch):
  '''INTERNAL_ERROR during incumbent lookup fails closed.'''
  _fake_mlflow(monkeypatch)
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  registry = ModelRegistry({
    'tracking_uri': 'sqlite:////tmp/r.db',
    'experiment_id': 'exp-1',
    'promote_on_metric': 'f1_weighted',
  })
  _setup_incumbent(
    registry, 'my-model', 'production', '0', 'run-old')
  exc_cls = _mlflow_exc_class('INTERNAL_ERROR')

  def _boom(model_name, model_alias):
    '''Raise an error.'''
    raise exc_cls('internal failure')

  registry._client.get_model_version_by_alias = _boom
  result = registry.register_model(
    object(), 'my-model', alias='production',
    metrics={'f1_weighted': 0.9})
  assert result['promoted'] is False
  assert registry._client.aliases[('my-model', 'production')] == '0'


def test_gate_not_found_still_promotes_first_version(monkeypatch):
  '''RESOURCE_DOES_NOT_EXIST lookup means no incumbent: promote.'''
  _fake_mlflow(monkeypatch)
  from forecast_ml_framework.registry.model_registry import (
    ModelRegistry)
  registry = ModelRegistry({
    'tracking_uri': 'sqlite:////tmp/r.db',
    'experiment_id': 'exp-1',
    'promote_on_metric': 'f1_weighted',
  })
  registry._get_client()
  exc_cls = _mlflow_exc_class('RESOURCE_DOES_NOT_EXIST')

  def _missing(model_name, model_alias):
    '''Return a missing value.'''
    raise exc_cls('alias not found')

  registry._client.get_model_version_by_alias = _missing
  result = registry.register_model(
    object(), 'my-model', alias='production',
    metrics={'f1_weighted': 0.9})
  assert result['promoted'] is True
  assert registry._client.aliases[('my-model', 'production')] == '1'


def test_dns_rebinding_ttl0_accepted_risk_validate_time_only(monkeypatch):
  '''TTL-0 DNS rebinding is an accepted risk (R-05).

  validate_registry_uri checks DNS at validation time only; it
  does not pin or cache the resolution, so a second resolution
  can differ (TTL-0 rebinding). Production deployments must
  therefore enforce an egress allowlist at the network layer;
  this test pins the time-of-check-only behaviour.
  '''
  from forecast_ml_framework.registry.model_registry import (
    validate_registry_uri)
  responses = iter([
    [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('8.8.8.8', 0))],
    [(socket.AF_INET, socket.SOCK_STREAM, 6, '',
      ('169.254.169.254', 0))],
  ])

  def _rebind(host, port, *args, **kwargs):
    '''Rebind.'''
    return next(responses)

  monkeypatch.setattr(socket, 'getaddrinfo', _rebind)
  uri = 'http://rebind.example.com:5000'
  assert validate_registry_uri(uri) == uri
  with pytest.raises(ValueError):
    validate_registry_uri(uri)
