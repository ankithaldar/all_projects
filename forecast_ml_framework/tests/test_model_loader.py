#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for model loaders.'''

import pickle
import sys
import types

import pytest


class Dummy:
  '''Dummy picklable model.'''

  def predict(self, rows):
    '''Predict sum.'''
    return [sum(row) for row in rows]


def _install_fake_mlflow(monkeypatch, sklearn_exc=None,
                         pyfunc_exc=None, sklearn_result=None,
                         pyfunc_result=None):
  '''Install a fake mlflow tree; return (calls, exceptions).'''
  fake = types.ModuleType('mlflow')
  fake.__path__ = []
  sklearn = types.ModuleType('mlflow.sklearn')
  pyfunc = types.ModuleType('mlflow.pyfunc')
  exceptions = types.ModuleType('mlflow.exceptions')

  class MlflowException(Exception):
    pass

  MlflowException.__module__ = 'mlflow.exceptions'
  exceptions.MlflowException = MlflowException
  calls = {'set_tracking_uri': [], 'sklearn': [], 'pyfunc': []}
  fake.set_tracking_uri = lambda uri: calls[
    'set_tracking_uri'].append(uri)

  def _sklearn_load(uri):
    calls['sklearn'].append(uri)
    if sklearn_exc is not None:
      raise sklearn_exc
    return sklearn_result if sklearn_result is not None else Dummy()

  def _pyfunc_load(uri):
    calls['pyfunc'].append(uri)
    if pyfunc_exc is not None:
      raise pyfunc_exc
    return pyfunc_result if pyfunc_result is not None else Dummy()

  sklearn.load_model = _sklearn_load
  pyfunc.load_model = _pyfunc_load
  fake.sklearn = sklearn
  fake.pyfunc = pyfunc
  fake.exceptions = exceptions
  monkeypatch.setitem(sys.modules, 'mlflow', fake)
  monkeypatch.setitem(sys.modules, 'mlflow.sklearn', sklearn)
  monkeypatch.setitem(sys.modules, 'mlflow.pyfunc', pyfunc)
  monkeypatch.setitem(sys.modules, 'mlflow.exceptions', exceptions)
  return calls, exceptions


def test_file_loader_loads_pickle(tmp_path):
  '''File loader loads pickled model lazily.'''
  from forecast_ml_framework.serving.model_loader import (
    FileModelLoader,
  )
  from sklearn.linear_model import LinearRegression
  model = LinearRegression().fit([[1.0, 2.0]], [3.0])
  path = tmp_path / 'model.pkl'
  path.write_bytes(pickle.dumps(model))
  loader = FileModelLoader(str(path))
  loaded = loader()
  assert loaded.predict([[1.0, 2.0]])[0] == pytest.approx(3.0)


def test_registry_loader_loads_via_mocked_mlflow(monkeypatch):
  '''Registry loader loads downloaded bytes via mlflow client.'''
  calls, _exc = _install_fake_mlflow_client(monkeypatch)
  from forecast_ml_framework.serving.model_loader import (
    RegistryModelLoader,
  )
  loader = RegistryModelLoader(
      'sqlite:///registry.db', 'my-model', alias='production')
  loaded = loader()
  assert calls['client'] == [{
      'tracking_uri': 'sqlite:///registry.db',
      'registry_uri': 'sqlite:///registry.db'}]
  assert calls['version'] == [('my-model', 'production')]
  assert calls['sklearn'] == []
  assert loaded.predict([[1.0, 2.0]])[0] == pytest.approx(3.0)


def test_registry_loader_default_alias_and_caches(monkeypatch):
  '''Registry loader defaults to production alias and caches.'''
  calls, _exc = _install_fake_mlflow_client(monkeypatch)
  from forecast_ml_framework.serving.model_loader import (
    RegistryModelLoader,
  )
  loader = RegistryModelLoader('sqlite:///registry.db', 'my-model')
  assert loader() is loader()
  assert calls['version'] == [('my-model', 'production')]
  assert len(calls['download']) == 1


def test_registry_loader_requires_model_name():
  '''Registry loader rejects missing model name generically.'''
  import pytest
  from forecast_ml_framework.serving.model_loader import (
    RegistryModelLoader,
  )
  with pytest.raises(ValueError) as excinfo:
    RegistryModelLoader('sqlite:///registry.db', '')
  assert 'model name' in str(excinfo.value)


@pytest.mark.parametrize('alias', [
  '', None, 'production\n', 'production\ninjected', '../../etc',
  'Bad Alias', 'prod!', 'a/b',
])
def test_registry_loader_rejects_bad_alias(alias):
  '''Bad alias raises ValueError before any mlflow call.

  Fails if RegistryModelLoader stops validating alias values
  and passes empty, newline, or path-traversal aliases through.
  '''
  from forecast_ml_framework.serving.model_loader import (
    RegistryModelLoader,
  )
  with pytest.raises(ValueError):
    RegistryModelLoader(
        'sqlite:///registry.db', 'my-model', alias=alias)


@pytest.mark.parametrize('name', [
  '../../etc/passwd', 'a/b', 'a b', 'Bad Name!', 'model.name',
  'model@production', 'a\nb', None, 123,
])
def test_registry_loader_rejects_bad_model_name(name):
  from forecast_ml_framework.serving.model_loader import (
    RegistryModelLoader,
  )
  with pytest.raises(ValueError):
    RegistryModelLoader('sqlite:///registry.db', name)


@pytest.mark.parametrize('uri', [
  'file:///etc/passwd', 'javascript:alert(1)',
  'http://user:pass@localhost:5000', 'https://user@localhost/',
  'http://169.254.169.254/', 'http://metadata.google.internal/',
  'http://100.100.100.200/', 'ftp://example.com/',
])
def test_registry_loader_rejects_bad_uri(uri):
  from forecast_ml_framework.serving.model_loader import (
    RegistryModelLoader,
  )
  with pytest.raises(ValueError):
    RegistryModelLoader(uri, 'my-model', alias='production')


def test_registry_loader_empty_alias_not_coerced():
  '''Empty alias must fail, not silently become production.'''
  from forecast_ml_framework.serving.model_loader import (
    RegistryModelLoader,
  )
  with pytest.raises(ValueError):
    RegistryModelLoader(
        'sqlite:///registry.db', 'my-model', alias='')
  with pytest.raises(ValueError):
    RegistryModelLoader(
        'sqlite:///registry.db', 'my-model', alias=None)


def test_typed_errors_are_runtime_errors():
  '''Typed loader errors subclass RuntimeError.'''
  from forecast_ml_framework.serving.model_loader import (
    ModelCorruptError,
    ModelNotConfiguredError,
    ModelNotFoundError,
  )
  for cls in (ModelNotConfiguredError, ModelNotFoundError,
              ModelCorruptError):
    assert issubclass(cls, RuntimeError)


def test_registry_loader_alias_not_found_no_pyfunc_fallback(
    monkeypatch, caplog):
  '''Alias-not-found MlflowException must not fall back.'''
  import logging
  from forecast_ml_framework.serving.model_loader import (
    ModelNotFoundError,
    RegistryModelLoader,
  )
  _calls, exceptions = _install_fake_mlflow_client(monkeypatch)
  not_found = exceptions.MlflowException(
      'No model version found for alias production')
  calls, _exc = _install_fake_mlflow_client(
      monkeypatch, version_exc=not_found)
  with caplog.at_level(logging.WARNING):
    with pytest.raises(ModelNotFoundError):
      RegistryModelLoader(
          'sqlite:///registry.db', 'my-model',
          alias='production')()
  assert calls['pyfunc'] == []
  assert caplog.records


def test_registry_loader_flavour_error_rejected_by_default(
    monkeypatch, caplog):
  '''Unrestricted bytes must NOT fall back to pyfunc code.'''
  import logging
  from forecast_ml_framework.serving.model_loader import (
    ModelCorruptError,
    RegistryModelLoader,
  )
  calls, _exc = _install_fake_mlflow_client(
      monkeypatch, artifact_bytes=b'not-a-pickle')
  with caplog.at_level(logging.WARNING):
    with pytest.raises(ModelCorruptError):
      RegistryModelLoader(
          'sqlite:///registry.db', 'my-model',
          alias='production')()
  assert calls['pyfunc'] == []
  assert caplog.records


def test_registry_loader_malicious_pyfunc_rejected(monkeypatch):
  '''Malicious payload is never executed via pyfunc by default.'''
  executed = []
  from forecast_ml_framework.serving.model_loader import (
    ModelCorruptError,
    RegistryModelLoader,
  )
  calls, _exc = _install_fake_mlflow_client(
      monkeypatch, artifact_bytes=b'not-a-pickle')
  import sys as _sys
  _sys.modules['mlflow.pyfunc'].load_model = lambda uri: executed.append(
      uri) or Dummy()
  with pytest.raises(ModelCorruptError):
    RegistryModelLoader(
        'sqlite:///registry.db', 'my-model',
        alias='production')()
  assert calls['pyfunc'] == []
  assert executed == []


def test_registry_loader_allow_pyfunc_explicit_warns(
    monkeypatch, caplog):
  '''Opt-in pyfunc requires explicit flag and startup warning.'''
  import logging
  from forecast_ml_framework.serving.model_loader import (
    RegistryModelLoader,
  )
  calls, _exc = _install_fake_mlflow_client(
      monkeypatch, artifact_bytes=b'not-a-pickle')
  with caplog.at_level(logging.WARNING):
    loaded = RegistryModelLoader(
        'sqlite:///registry.db', 'my-model', alias='production',
        allow_pyfunc=True)()
  assert calls['pyfunc'] == ['models:/my-model@production']
  assert loaded.predict([[1.0, 2.0]]) == [3.0]
  assert any('pyfunc' in r.getMessage().lower()
             for r in caplog.records)


def test_registry_loader_no_mlflow_not_configured(monkeypatch):
  '''Missing mlflow maps to ModelNotConfiguredError.'''
  import sys
  from forecast_ml_framework.serving.model_loader import (
    ModelNotConfiguredError,
    RegistryModelLoader,
  )
  monkeypatch.setitem(sys.modules, 'mlflow', None)
  with pytest.raises(ModelNotConfiguredError) as excinfo:
    RegistryModelLoader(
        'sqlite:///registry.db', 'my-model', alias='production')()
  assert 'sqlite' not in str(excinfo.value)


def test_registry_loader_failure_generic_no_leak(monkeypatch):
  '''Registry loader failure does not leak registry details.'''
  from forecast_ml_framework.serving.model_loader import (
    ModelCorruptError,
    RegistryModelLoader,
  )
  secret = 'sqlite:///secret-dir/registry.db'
  _install_fake_mlflow_client(
      monkeypatch, version_exc=RuntimeError(secret))
  with pytest.raises(ModelCorruptError) as excinfo:
    RegistryModelLoader(secret, 'my-model', alias='production')()
  assert secret not in str(excinfo.value)
  assert 'secret-dir' not in str(excinfo.value)


def test_file_loader_evil_pickle_rejected(tmp_path):
  import os
  import pickle
  import pytest
  from forecast_ml_framework.serving.model_loader import FileModelLoader

  class Evil:
    def __reduce__(self):
      return (os.system, ('echo pwned',))

  path = tmp_path / 'evil.pkl'
  path.write_bytes(pickle.dumps(Evil()))
  from forecast_ml_framework.serving.model_loader import (
    ModelCorruptError,
  )
  with pytest.raises(ModelCorruptError):
    FileModelLoader(str(path)).load()


def test_file_loader_missing_no_path_disclosure(tmp_path):
  import pytest
  from forecast_ml_framework.serving.model_loader import (
    FileModelLoader,
    ModelNotFoundError,
  )
  secret = str(tmp_path / 'secret-dir' / 'model.pkl')
  with pytest.raises(ModelNotFoundError) as excinfo:
    FileModelLoader(secret).load()
  assert secret not in str(excinfo.value)
  assert str(tmp_path) not in str(excinfo.value)


def test_file_loader_malformed_corrupt(tmp_path):
  import pytest
  from forecast_ml_framework.serving.model_loader import (
    FileModelLoader,
    ModelCorruptError,
  )
  path = tmp_path / 'model.pkl'
  path.write_bytes(b'not-a-pickle')
  with pytest.raises(ModelCorruptError) as excinfo:
    FileModelLoader(str(path)).load()
  assert str(path) not in str(excinfo.value)


def test_file_loader_chain_read_pickle_blocked(tmp_path):
  import os
  import pickle
  import pandas as pd
  import pytest
  from forecast_ml_framework.serving.model_loader import FileModelLoader
  marker = tmp_path / 'pwned'

  class Evil:
    def __reduce__(self):
      return (os.system, (f'touch {marker}',))

  inner = pickle.dumps(Evil())

  class Outer:
    def __reduce__(self):
      return (pd.read_pickle, (inner,))

  path = tmp_path / 'chain.pkl'
  path.write_bytes(pickle.dumps(Outer()))
  from forecast_ml_framework.serving.model_loader import (
    ModelCorruptError,
  )
  with pytest.raises(ModelCorruptError):
    FileModelLoader(str(path)).load()
  assert not marker.exists()


def test_file_loader_read_csv_blocked(tmp_path):
  import pickle
  import pandas as pd
  import pytest
  from forecast_ml_framework.serving.model_loader import FileModelLoader
  secret = tmp_path / 'secret.csv'
  secret.write_text('secret_col\n42\n', encoding='utf-8')

  class Outer:
    def __reduce__(self):
      return (pd.read_csv, (str(secret),))

  path = tmp_path / 'csv.pkl'
  path.write_bytes(pickle.dumps(Outer()))
  from forecast_ml_framework.serving.model_loader import (
    ModelCorruptError,
  )
  with pytest.raises(ModelCorruptError):
    FileModelLoader(str(path)).load()


# --- W3 RED: registry via restricted_loads on downloaded bytes ---

def _linear_pickle_bytes():
  '''Return pickle bytes of a fitted LinearRegression.'''
  from sklearn.linear_model import LinearRegression
  model = LinearRegression().fit([[1.0, 2.0]], [3.0])
  return pickle.dumps(model)


def _install_fake_mlflow_client(monkeypatch, artifact_bytes=None,
                               version='3', run_id='run-1',
                               version_exc=None, tracking_exc=None):
  '''Install fake mlflow tracking client; return call log.'''
  import tempfile
  from pathlib import Path
  fake = types.ModuleType('mlflow')
  fake.__path__ = []
  tracking = types.ModuleType('mlflow.tracking')
  exceptions = types.ModuleType('mlflow.exceptions')
  sklearn = types.ModuleType('mlflow.sklearn')
  pyfunc = types.ModuleType('mlflow.pyfunc')

  class MlflowException(Exception):
    pass

  MlflowException.__module__ = 'mlflow.exceptions'
  exceptions.MlflowException = MlflowException

  class _Version:
    def __init__(self):
      self.version = version
      self.run_id = run_id
      self.source = 'fake-source'

  calls = {'client': [], 'version': [], 'download': [],
           'set_tracking_uri': [], 'set_registry_uri': [],
           'sklearn': [], 'pyfunc': []}
  fake.set_tracking_uri = lambda uri: calls[
      'set_tracking_uri'].append(uri)

  def _set_registry_uri(uri):
    calls['set_registry_uri'].append(uri)

  fake.set_registry_uri = _set_registry_uri

  class _FakeClient:
    def __init__(self, tracking_uri=None, registry_uri=None):
      calls['client'].append(
          {'tracking_uri': tracking_uri,
           'registry_uri': registry_uri})

    def get_model_version_by_alias(self, name, alias):
      calls['version'].append((name, alias))
      if version_exc is not None:
        raise version_exc
      return _Version()

    def download_artifacts(self, run_id_arg, path, dst_path=None):
      calls['download'].append((run_id_arg, path, dst_path))
      data = artifact_bytes if artifact_bytes is not None \
          else _linear_pickle_bytes()
      base = Path(dst_path) if dst_path else Path(
          tempfile.mkdtemp())
      base.mkdir(parents=True, exist_ok=True)
      target = base / 'model.pkl'
      target.write_bytes(data)
      return str(target)

  tracking.MlflowClient = _FakeClient

  def _sklearn_load(uri):
    calls['sklearn'].append(uri)
    raise AssertionError('sklearn.load_model must not be used')

  def _pyfunc_load(uri):
    calls['pyfunc'].append(uri)
    return Dummy()

  sklearn.load_model = _sklearn_load
  pyfunc.load_model = _pyfunc_load
  fake.sklearn = sklearn
  fake.pyfunc = pyfunc
  fake.tracking = tracking
  fake.exceptions = exceptions
  monkeypatch.setitem(sys.modules, 'mlflow', fake)
  monkeypatch.setitem(sys.modules, 'mlflow.tracking', tracking)
  monkeypatch.setitem(sys.modules, 'mlflow.sklearn', sklearn)
  monkeypatch.setitem(sys.modules, 'mlflow.pyfunc', pyfunc)
  monkeypatch.setitem(sys.modules, 'mlflow.exceptions', exceptions)
  return calls, exceptions


def test_registry_loader_downloads_and_restricted_loads(monkeypatch):
  '''Registry load uses downloaded bytes + restricted_loads.'''
  calls, _exc = _install_fake_mlflow_client(monkeypatch)
  from forecast_ml_framework.serving.model_loader import (
    RegistryModelLoader,
  )
  loaded = RegistryModelLoader(
      'sqlite:///registry.db', 'my-model', alias='production')()
  assert loaded.predict([[1.0, 2.0]])[0] == pytest.approx(3.0)
  assert calls['sklearn'] == []
  assert calls['version'] == [('my-model', 'production')]
  assert calls['download'] != []


def test_registry_loader_malicious_pickle_rejected(monkeypatch, tmp_path):
  '''Malicious registry pickle is rejected, never executed.'''
  import os
  marker = tmp_path / 'pwned'

  class Evil:
    def __reduce__(self):
      return (os.system, (f'touch {marker}',))

  _install_fake_mlflow_client(
      monkeypatch, artifact_bytes=pickle.dumps(Evil()))
  from forecast_ml_framework.serving.model_loader import (
    ModelCorruptError,
    RegistryModelLoader,
  )
  with pytest.raises(ModelCorruptError):
    RegistryModelLoader(
        'sqlite:///registry.db', 'my-model')()
  assert not marker.exists()


def test_registry_loader_sha256_pin_match_and_mismatch(monkeypatch):
  '''MODEL_SHA256 pin is verified before unpickling.'''
  raw = _linear_pickle_bytes()
  digest = __import__('hashlib').sha256(raw).hexdigest()
  _install_fake_mlflow_client(monkeypatch, artifact_bytes=raw)
  from forecast_ml_framework.serving.model_loader import (
    ModelCorruptError,
    RegistryModelLoader,
  )
  monkeypatch.setenv('MODEL_SHA256', digest)
  loaded = RegistryModelLoader(
      'sqlite:///registry.db', 'my-model')()
  assert loaded.predict([[1.0, 2.0]])[0] == pytest.approx(3.0)
  monkeypatch.setenv('MODEL_SHA256', '0' * 64)
  with pytest.raises(ModelCorruptError):
    RegistryModelLoader(
        'sqlite:///registry.db', 'my-model')()


def test_registry_loader_logs_resolved_version(monkeypatch, caplog):
  '''Resolved registry version is logged at load.'''
  import logging
  _install_fake_mlflow_client(monkeypatch, version='7')
  from forecast_ml_framework.serving.model_loader import (
    RegistryModelLoader,
  )
  with caplog.at_level(logging.INFO):
    RegistryModelLoader('sqlite:///registry.db', 'my-model')()
  assert '7' in caplog.text
  assert 'my-model' in caplog.text


def test_registry_loader_separate_tracking_registry_uris(monkeypatch):
  '''Tracking and registry URIs are passed separately.'''
  calls, _exc = _install_fake_mlflow_client(monkeypatch)
  from forecast_ml_framework.serving.model_loader import (
    RegistryModelLoader,
  )
  RegistryModelLoader(
      'sqlite:///registry2.db', 'my-model', alias='production',
      tracking_uri='sqlite:///tracking.db')()
  assert calls['client'] == [{
      'tracking_uri': 'sqlite:///tracking.db',
      'registry_uri': 'sqlite:///registry2.db'}]


def test_registry_loader_secret_exception_not_logged(
    monkeypatch, caplog):
  '''Secret detail from version lookup never reaches logs.'''
  import logging
  secret = 'supersecretpassword77777'
  _calls, exceptions = _install_fake_mlflow_client(monkeypatch)
  exc = exceptions.MlflowException(f'password={secret}')
  _install_fake_mlflow_client(monkeypatch, version_exc=exc)
  from forecast_ml_framework.serving.model_loader import (
    ModelCorruptError,
    RegistryModelLoader,
  )
  with caplog.at_level(logging.DEBUG):
    with pytest.raises(ModelCorruptError):
      RegistryModelLoader(
          'sqlite:///registry.db', 'my-model')()
  assert secret not in caplog.text


def test_file_loader_size_cap_rejected(tmp_path):
  '''Oversized file artifact is rejected before unpickling.'''
  from forecast_ml_framework.serving.model_loader import (
    FileModelLoader,
    ModelCorruptError,
  )
  model = _linear_pickle_bytes()
  path = tmp_path / 'model.pkl'
  path.write_bytes(model)
  with pytest.raises(ModelCorruptError):
    FileModelLoader(str(path), max_artifact_bytes=10).load()


def test_file_loader_sha256_mismatch_rejected(monkeypatch, tmp_path):
  '''File artifact with wrong MODEL_SHA256 is rejected.'''
  from forecast_ml_framework.serving.model_loader import (
    FileModelLoader,
    ModelCorruptError,
  )
  path = tmp_path / 'model.pkl'
  path.write_bytes(_linear_pickle_bytes())
  monkeypatch.setenv('MODEL_SHA256', '0' * 64)
  with pytest.raises(ModelCorruptError):
    FileModelLoader(str(path)).load()


def test_is_alias_not_found_bare_alias_word_is_not_not_found():
  '''Bare word alias in an error is not a not-found signal.'''
  from forecast_ml_framework.serving.model_loader import (
    _is_alias_not_found,
  )

  class _Exc(Exception):
    error_code = 'INTERNAL_ERROR'

  assert _is_alias_not_found(
      _Exc('alias config invalid for model')) is False
  assert _is_alias_not_found(
      _Exc('alias not found for model')) is True
  assert _is_alias_not_found(
      _Exc('model version not found')) is True


  class _Coded(Exception):
    error_code = 'RESOURCE_DOES_NOT_EXIST'

  assert _is_alias_not_found(_Coded('something vague')) is True
