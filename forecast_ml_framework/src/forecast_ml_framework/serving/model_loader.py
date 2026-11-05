#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Model loaders for serving.

File-based loading is supported in v1. The MLflow registry
loader is a Phase 2 stub.

Trust boundary: model files are trusted framework artifacts
only. Pickle payloads are decoded with the restricted
unpickler from :mod:`forecast_ml_framework.core.artifact_store`
(exact ``(module, name)`` allowlist, shared with the artifact
store) so chained ``__reduce__`` payloads reaching loader
functions such as ``pandas.read_pickle``/``read_csv`` are
rejected with ``ValueError`` instead of executing code.

No auth or Docker is provided by this module.
'''

import os

from forecast_ml_framework.core.artifact_store import restricted_loads


class FileModelLoader:
  '''Load a pickled model from a file path lazily.

  Args:
    artifact_path: Path to a pickle model file.
  '''

  def __init__(self, artifact_path: str) -> None:
    self.artifact_path = artifact_path
    self._model = None

  def load(self):
    '''Load and cache the model, returning it.

    Raises:
      FileNotFoundError: When the artifact is missing; the
        message never contains the server filesystem path.
      ValueError: When the pickle references forbidden globals.
    '''
    if self._model is None:
      self._model = self._load_file(self.artifact_path)
    return self._model

  def __call__(self):
    '''Return the cached model, loading it on first call.'''
    return self.load()

  @staticmethod
  def _load_file(path: str):
    '''Load model from a pickle file via the restricted unpickler.

    Args:
      path: Filesystem path to the pickle artifact.

    Returns:
      The unpickled model.

    Raises:
      FileNotFoundError: With a generic message (no path leak).
      ValueError: When the pickle is forbidden or malformed.
    '''
    if not os.path.isfile(str(path)):
      raise FileNotFoundError('model artifact not found')
    with open(path, 'rb') as handle:
      raw = handle.read()
    return restricted_loads(raw)


class RegistryModelLoader:
  '''Stub for MLflow registry loading (Phase 2).

  Args:
    model_name: Registered model name.
    alias: Registry alias, for example ``latest``.
  '''

  def __init__(self, model_name: str, alias: str = 'latest') -> None:
    self.model_name = model_name
    self.alias = alias

  def load(self):
    '''Raise NotImplementedError until Phase 2.'''
    raise NotImplementedError(
      'MLflow registry loading is Phase 2; use FileModelLoader'
    )

  def __call__(self):
    '''Raise NotImplementedError until Phase 2.'''
    return self.load()
