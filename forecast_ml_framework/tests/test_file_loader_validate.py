#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''File/registry loaders validate dict envelopes.'''
# pylint: disable=import-outside-toplevel, unused-argument

import pickle
import sys
import tempfile
import types
from pathlib import Path

import pytest


def _tiny_classification():
  '''Return fitted LogisticRegression and its data.'''
  import numpy as np
  from sklearn.linear_model import LogisticRegression
  x = np.array([[0.0], [1.0], [2.0], [3.0],
                [0.5], [2.5], [1.5], [3.5]])
  y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
  est = LogisticRegression(max_iter=1000).fit(x, y)
  return est, x


def _valid_envelope_bytes():
  '''Return pickle bytes of a valid envelope.'''
  from forecast_ml_framework.pipelines.modeling.envelope import (
    wrap_envelope,
  )
  est, _ = _tiny_classification()
  return pickle.dumps(wrap_envelope(est, ['0', '1']))


def _invalid_envelope_bytes():
  '''Return pickle bytes of an envelope with bad label_set.'''
  est, _ = _tiny_classification()
  env = {
    'model': est,
    'label_set': ['1', '0'],
    'registry_ref': None,
    'onnx_sidecar': None,
  }
  return pickle.dumps(env)


def _install_fake_mlflow(monkeypatch, artifact_bytes):
  '''Install fake mlflow client serving artifact_bytes.'''
  fake = types.ModuleType('mlflow')
  fake.__path__ = []
  tracking = types.ModuleType('mlflow.tracking')
  pyfunc = types.ModuleType('mlflow.pyfunc')
  exceptions = types.ModuleType('mlflow.exceptions')

  class MlflowException(Exception):
    '''Mlflow exception test double.'''

  MlflowException.__module__ = 'mlflow.exceptions'
  exceptions.MlflowException = MlflowException

  class _Version:
    '''Version test double.'''

    def __init__(self):
      '''Initialize the test double.'''
      self.version = '3'
      self.run_id = 'run-1'
      self.source = 'fake-source'

  class _FakeClient:
    '''Fake client test double.'''

    def __init__(self, tracking_uri=None, registry_uri=None):
      '''Initialize the test double.'''

    def get_model_version_by_alias(self, name, alias):
      '''Return a version object.'''
      return _Version()

    def download_artifacts(self, run_id, path, dst_path=None):
      '''Write artifact bytes and return their path.'''
      base = Path(dst_path) if dst_path else Path(
        tempfile.mkdtemp())
      base.mkdir(parents=True, exist_ok=True)
      target = base / 'model.pkl'
      target.write_bytes(artifact_bytes)
      return str(target)

  tracking.MlflowClient = _FakeClient
  pyfunc.load_model = lambda uri: (_ for _ in ()).throw(
    AssertionError('pyfunc must not be called'))
  fake.tracking = tracking
  fake.pyfunc = pyfunc
  fake.exceptions = exceptions
  monkeypatch.setitem(sys.modules, 'mlflow', fake)
  monkeypatch.setitem(sys.modules, 'mlflow.tracking', tracking)
  monkeypatch.setitem(sys.modules, 'mlflow.pyfunc', pyfunc)
  monkeypatch.setitem(sys.modules, 'mlflow.exceptions', exceptions)


def test_file_loader_valid_envelope_unwrapped(tmp_path):
  '''File loader serves unwrapped estimator for valid envelope.'''
  from forecast_ml_framework.serving.model_loader import (
    FileModelLoader,
  )
  est, x = _tiny_classification()
  path = tmp_path / 'model.pkl'
  path.write_bytes(_valid_envelope_bytes())
  loaded = FileModelLoader(str(path)).load()
  assert not isinstance(loaded, dict)
  assert loaded.predict(x).tolist() == est.predict(x).tolist()


def test_file_loader_invalid_envelope_rejected(tmp_path):
  '''File loader raises ModelCorruptError for invalid envelope.'''
  from forecast_ml_framework.serving.model_loader import (
    FileModelLoader,
    ModelCorruptError,
  )
  path = tmp_path / 'model.pkl'
  path.write_bytes(_invalid_envelope_bytes())
  with pytest.raises(ModelCorruptError):
    FileModelLoader(str(path)).load()


def test_registry_loader_valid_envelope_unwrapped(monkeypatch):
  '''Registry loader serves unwrapped estimator if valid.'''
  from forecast_ml_framework.serving.model_loader import (
    RegistryModelLoader,
  )
  est, x = _tiny_classification()
  _install_fake_mlflow(monkeypatch, _valid_envelope_bytes())
  loaded = RegistryModelLoader(
    'sqlite:///registry.db', 'my-model', alias='production')()
  assert not isinstance(loaded, dict)
  assert loaded.predict(x).tolist() == est.predict(x).tolist()


def test_registry_loader_invalid_envelope_rejected(monkeypatch):
  '''Registry loader raises ModelCorruptError if invalid.'''
  from forecast_ml_framework.serving.model_loader import (
    ModelCorruptError,
    RegistryModelLoader,
  )
  _install_fake_mlflow(monkeypatch, _invalid_envelope_bytes())
  with pytest.raises(ModelCorruptError):
    RegistryModelLoader(
      'sqlite:///registry.db', 'my-model', alias='production')()
