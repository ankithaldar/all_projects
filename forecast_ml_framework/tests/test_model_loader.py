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
  '''Registry loader loads models:/name@alias via mlflow.'''
  calls, _exc = _install_fake_mlflow(monkeypatch)
  from forecast_ml_framework.serving.model_loader import (
    RegistryModelLoader,
  )
  loader = RegistryModelLoader(
      'sqlite:///registry.db', 'my-model', alias='production')
  loaded = loader()
  assert calls['set_tracking_uri'] == ['sqlite:///registry.db']
  assert calls['sklearn'] == ['models:/my-model@production']
  assert loaded.predict([[1.0, 2.0]]) == [3.0]


def test_registry_loader_default_alias_and_caches(monkeypatch):
  '''Registry loader defaults to production alias and caches.'''
  calls, _exc = _install_fake_mlflow(monkeypatch)
  from forecast_ml_framework.serving.model_loader import (
    RegistryModelLoader,
  )
  loader = RegistryModelLoader('sqlite:///registry.db', 'my-model')
  assert loader() is loader()
  assert calls['sklearn'] == ['models:/my-model@production']


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
  _calls, exceptions = _install_fake_mlflow(monkeypatch)
  # Re-install with the not-found error now that the class exists.
  not_found = exceptions.MlflowException(
      'No model version found for alias production')
  calls, _exc = _install_fake_mlflow(
      monkeypatch, sklearn_exc=not_found)
  with caplog.at_level(logging.WARNING):
    with pytest.raises(ModelNotFoundError):
      RegistryModelLoader(
          'sqlite:///registry.db', 'my-model',
          alias='production')()
  assert calls['pyfunc'] == []
  assert caplog.records


def test_registry_loader_flavour_error_rejected_by_default(
    monkeypatch, caplog):
  '''Flavour error must NOT fall back to arbitrary pyfunc code.'''
  import logging
  _calls, exceptions = _install_fake_mlflow(monkeypatch)
  from forecast_ml_framework.serving.model_loader import (
    ModelCorruptError,
    RegistryModelLoader,
  )
  flavour = exceptions.MlflowException(
      'The model does not have the specified flavour')
  calls, _exc = _install_fake_mlflow(
      monkeypatch, sklearn_exc=flavour)
  with caplog.at_level(logging.WARNING):
    with pytest.raises(ModelCorruptError):
      RegistryModelLoader(
          'sqlite:///registry.db', 'my-model',
          alias='production')()
  assert calls['pyfunc'] == []
  assert caplog.records


def test_registry_loader_malicious_pyfunc_rejected(monkeypatch):
  '''Malicious pyfunc payload is never executed by default.'''
  executed = []
  _calls, exceptions = _install_fake_mlflow(monkeypatch)
  from forecast_ml_framework.serving.model_loader import (
    ModelCorruptError,
    RegistryModelLoader,
  )
  flavour = exceptions.MlflowException(
      'The model does not have the specified flavour')
  calls, _exc = _install_fake_mlflow(
      monkeypatch, sklearn_exc=flavour,
      pyfunc_result=None)
  # Even if pyfunc would return a model, loader must not call it.
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
  _calls, exceptions = _install_fake_mlflow(monkeypatch)
  from forecast_ml_framework.serving.model_loader import (
    RegistryModelLoader,
  )
  flavour = exceptions.MlflowException(
      'The model does not have the specified flavour')
  calls, _exc = _install_fake_mlflow(
      monkeypatch, sklearn_exc=flavour)
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
  _install_fake_mlflow(
      monkeypatch, sklearn_exc=RuntimeError(secret),
      pyfunc_exc=RuntimeError(secret))
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
