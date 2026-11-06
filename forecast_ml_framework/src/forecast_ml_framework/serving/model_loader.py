#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Model loaders for serving.

File-based loading is supported in v1. The MLflow registry
loader loads ``models:/<name>@<alias>`` via the lazy mlflow
sklearn entry point only (``mlflow.sklearn.load_model``).

Trust boundary / registry trust: the registry (and any
model registered in it) must be fully trusted. sklearn
loading still unpickles model artifacts, so a malicious
or compromised registry can execute code at load time --
only load from registries you trust. The pyfunc flavour
is NOT used by default: ``mlflow.pyfunc.load_model``
executes arbitrary code shipped with a model, so it is
gated behind explicit ``allow_pyfunc=True`` (with a
startup warning). Model files are trusted framework
artifacts only. Pickle payloads are decoded with the
restricted unpickler from
:mod:`forecast_ml_framework.core.artifact_store`
(exact ``(module, name)`` allowlist, shared with the artifact
store) so chained ``__reduce__`` payloads reaching loader
functions such as ``pandas.read_pickle``/``read_csv`` are
rejected with ``ValueError`` instead of executing code.

No auth or Docker is provided by this module.
'''

import logging
import os

from forecast_ml_framework.core.artifact_store import restricted_loads
from forecast_ml_framework.registry.model_registry import (
  _is_mlflow_exception, validate_alias, validate_model_name,
  validate_registry_uri)

logger = logging.getLogger(__name__)

_FLAVOUR_HINTS = (
  'flavour', 'flavor', 'does not have the specified',
  'does not have a specified', 'not have the specified',
  'no sklearn flavor', 'no sklearn flavour',
)

_NOT_FOUND_HINTS = (
  'alias', 'does not exist', 'resource_does_not_exist',
  'not found', 'no such', 'no model version',
)


def _is_alias_not_found(exc):
  '''Return True when ``exc`` signals alias/model not found.'''
  if type(exc).__name__ == 'ResourceDoesNotExistException':
    return True
  text = str(exc).lower()
  code = str(getattr(exc, 'error_code', '') or '').lower()
  if 'RESOURCE_DOES_NOT_EXIST' in code:
    return True
  return any(hint in text for hint in _NOT_FOUND_HINTS)


def _is_flavour_error(exc):
  '''Return True when ``exc`` signals a flavour mismatch.'''
  text = str(exc).lower()
  return any(hint in text for hint in _FLAVOUR_HINTS)


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
  '''Load a model from the MLflow registry lazily.

  Only the sklearn flavour is loaded by default. The
  registry must be trusted (see module docstring).

  Args:
    registry_uri: MLflow tracking/registry URI, or None to
      keep the ambient mlflow configuration.
    model_name: Registered model name.
    alias: Registry alias, default ``production``.
    allow_pyfunc: Explicit opt-in to the pyfunc flavour
      fallback. Default False. When True a startup warning
      is logged: pyfunc executes arbitrary code shipped
      with the model, so only enable for fully trusted
      registries.
  '''

  def __init__(
      self,
      registry_uri: str | None = None,
      model_name: str | None = None,
      alias: str = 'production',
      allow_pyfunc: bool = False,
  ) -> None:
    validate_model_name(model_name)
    validate_alias(alias)
    if registry_uri:
      validate_registry_uri(registry_uri)
    self.registry_uri = registry_uri
    self.model_name = model_name
    self.alias = alias
    self.allow_pyfunc = bool(allow_pyfunc)
    if self.allow_pyfunc:
      logger.warning(
        'allow_pyfunc=True: mlflow pyfunc fallback enabled; '
        'pyfunc executes arbitrary code from the registry '
        'model -- only use with a fully trusted registry')
    self._model = None

  def load(self):
    '''Load and cache the registry model, returning it.

    Raises:
      RuntimeError: With a generic message (no registry URI,
        model path, or underlying error leak) when loading
        fails for any reason.
    '''
    if self._model is None:
      self._model = self._load_registry()
    return self._model

  def __call__(self):
    '''Return the cached model, loading it on first call.'''
    return self.load()

  def _load_registry(self):
    '''Load ``models:/<name>@<alias>`` via lazy mlflow imports.

    sklearn flavour only by default. A flavour-mismatch
    MlflowException is a generic RuntimeError and never
    falls back to pyfunc unless ``allow_pyfunc=True`` was
    explicitly given (startup warning covers that case).
    Alias/model-not-found and every other MlflowException
    likewise become a generic RuntimeError so no
    URI/detail leaks.
    '''
    try:
      import mlflow
    except ImportError:
      raise RuntimeError('model unavailable') from None
    try:
      if self.registry_uri:
        mlflow.set_tracking_uri(self.registry_uri)
      uri = f'models:/{self.model_name}@{self.alias}'
      try:
        return mlflow.sklearn.load_model(uri)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        if not _is_mlflow_exception(exc):
          raise RuntimeError('model unavailable') from None
        if _is_alias_not_found(exc):
          logger.warning(
            'registry model %r alias %r not found',
            self.model_name, self.alias)
          raise RuntimeError('model unavailable') from None
        if not _is_flavour_error(exc):
          logger.warning(
            'sklearn load failed for %r@%r: %s',
            self.model_name, self.alias, exc)
          raise RuntimeError('model unavailable') from None
        if not self.allow_pyfunc:
          logger.warning(
            'sklearn flavour unavailable for %r@%r; '
            'pyfunc fallback disabled (allow_pyfunc=False): %s',
            self.model_name, self.alias, exc)
          raise RuntimeError('model unavailable') from None
        logger.warning(
          'sklearn flavour unavailable for %r@%r, '
          'falling back to pyfunc (allow_pyfunc=True): %s',
          self.model_name, self.alias, exc)
        return mlflow.pyfunc.load_model(uri)
    except RuntimeError:
      raise
    except Exception:  # pylint: disable=broad-exception-caught
      raise RuntimeError('model unavailable') from None
