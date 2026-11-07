#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Model loaders for serving.

File-based loading is supported in v1. The MLflow registry
loader resolves ``models:/<name>@<alias>`` via the lazy
mlflow tracking client, downloads the artifact bytes, and
decodes them with the restricted unpickler from
:mod:`forecast_ml_framework.core.artifact_store`. It never
calls the mlflow sklearn entry point on registry bytes:
that entry point unpickles without the restricted allowlist, so
a malicious registry pickle could execute code at load
time. The pyfunc flavour is NOT used by default:
``mlflow.pyfunc.load_model`` executes arbitrary code
shipped with a model, so it is gated behind explicit
``allow_pyfunc=True`` (with a startup warning) and is
only reached when restricted loading of the downloaded
bytes fails. Model files are trusted framework artifacts
only. Pickle payloads are decoded with the restricted
unpickler (exact ``(module, name)`` allowlist, shared
with the artifact store) so chained ``__reduce__``
payloads reaching loader functions such as
``pandas.read_pickle``/``read_csv`` are rejected instead
of executing code.

Integrity and size: artifact reads are capped at
``MAX_ARTIFACT_BYTES`` (512 MiB default, overridable via
``FORECAST_MAX_ARTIFACT_BYTES``/``MAX_ARTIFACT_BYTES`` or
``max_artifact_bytes``) before any unpickling, and an
optional SHA-256 pin (``MODEL_SHA256`` env var or
``expected_sha256``) is verified against the downloaded
bytes before unpickling. Tracking and registry URIs are
kept separate (``tracking_uri`` vs ``registry_uri``);
when only one is given it is used for both, which
matches sqlite file registries where the two coincide.

No auth or Docker is provided by this module.
'''
# pylint: disable=import-outside-toplevel

import hashlib
import importlib
import logging
import os
import secrets
import tempfile
import threading
from pathlib import Path

from forecast_ml_framework.core.artifact_store import restricted_loads
from forecast_ml_framework.registry.model_registry import (
  validate_alias, validate_model_name, validate_registry_uri)

logger = logging.getLogger(__name__)

#: Default cap for a single model artifact read.
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024

_CHUNK = 1024 * 1024

_FLAVOUR_HINTS = (
  'flavour', 'flavor', 'does not have the specified',
  'does not have a specified', 'not have the specified',
  'no sklearn flavor', 'no sklearn flavour',
)

#: Exact phrases (never the bare word ``alias``) that, with
#: the exact error code, signal a missing alias/model.
_NOT_FOUND_PHRASES = (
  'does not exist', 'resource_does_not_exist', 'not found',
  'no such model', 'no such alias', 'no model version',
  'alias not found', 'alias does not exist',
  'model version not found', 'model not found',
  'registered model not found',
)

_NOT_FOUND_CODES = frozenset({'RESOURCE_DOES_NOT_EXIST'})


def _is_alias_not_found(exc):
  '''Return True when ``exc`` signals alias/model not found.

  Matching is exact on the MLflow error code and on
  specific phrases only; an error that merely mentions
  the word ``alias`` (for example a config error) is not
  classified as not-found.
  '''
  if type(exc).__name__ == 'ResourceDoesNotExistException':
    return True
  code = str(getattr(exc, 'error_code', '') or '').strip().upper()
  if code in _NOT_FOUND_CODES:
    return True
  text = str(exc).lower()
  return any(phrase in text for phrase in _NOT_FOUND_PHRASES)


def _is_flavour_error(exc):
  '''Return True when ``exc`` signals a flavour mismatch.'''
  text = str(exc).lower()
  return any(hint in text for hint in _FLAVOUR_HINTS)


def _max_artifact_bytes(explicit=None):
  '''Return the effective artifact size cap in bytes.

  Args:
    explicit: Caller-supplied cap, winning over env vars.

  Returns:
    Cap in bytes; the default when no source sets one.

  Raises:
    ValueError: When an explicit or env cap is invalid.
  '''
  raw = explicit
  if raw is None:
    raw = os.environ.get('FORECAST_MAX_ARTIFACT_BYTES')
  if raw is None:
    raw = os.environ.get('MAX_ARTIFACT_BYTES')
  if raw is None or raw == '':
    return MAX_ARTIFACT_BYTES
  try:
    cap = int(str(raw), 10)
  except (TypeError, ValueError):
    raise ValueError('invalid artifact size cap') from None
  if cap <= 0:
    raise ValueError('invalid artifact size cap')
  return cap


def _expected_sha256(explicit=None):
  '''Return the expected SHA-256 hex digest, or None.

  Args:
    explicit: Caller-supplied digest, winning over env.

  Returns:
    Lowercased 64-char hex digest, or None when no pin
    is configured.

  Raises:
    ValueError: When a configured digest is malformed.
  '''
  raw = explicit
  if raw is None:
    raw = os.environ.get('MODEL_SHA256')
  if raw is None:
    raw = os.environ.get('FORECAST_MODEL_SHA256')
  if raw is None or str(raw).strip() == '':
    return None
  digest = str(raw).strip().lower()
  if len(digest) != 64 or any(
      char not in '0123456789abcdef' for char in digest):
    raise ValueError('invalid MODEL_SHA256 digest')
  return digest


def _verify_sha256(raw, expected):
  '''Verify ``raw`` against an expected SHA-256 pin.

  Args:
    raw: Downloaded artifact bytes.
    expected: Expected hex digest, or None to skip.

  Raises:
    ModelCorruptError: When the digest mismatches.
  '''
  if expected is None:
    return
  actual = hashlib.sha256(bytes(raw)).hexdigest()
  if not secrets.compare_digest(actual, expected):
    raise ModelCorruptError('model artifact corrupt')


def _read_file_capped(path, cap):
  '''Read a file in chunks, rejecting reads over ``cap``.

  Args:
    path: Filesystem path to read.
    cap: Maximum bytes to accept.

  Returns:
    File bytes.

  Raises:
    ModelNotFoundError: When the file is missing.
    ModelCorruptError: When the file exceeds ``cap``.
  '''
  if not os.path.isfile(str(path)):
    raise ModelNotFoundError('model artifact not found')
  chunks = []
  total = 0
  try:
    with open(path, 'rb') as handle:
      while True:
        chunk = handle.read(_CHUNK)
        if not chunk:
          break
        total += len(chunk)
        if total > cap:
          raise ModelCorruptError('model artifact corrupt')
        chunks.append(chunk)
  except FileNotFoundError:
    raise ModelNotFoundError('model artifact not found') from None
  except OSError:
    raise ModelCorruptError('model artifact corrupt') from None
  return b''.join(chunks)


def _pickle_path_in(downloaded):
  '''Return the pickle file inside a downloaded path.

  Args:
    downloaded: File or directory path returned by an
      artifact download.

  Returns:
    Path to the pickle file.

  Raises:
    ModelCorruptError: When no pickle file is present.
  '''
  path = Path(str(downloaded))
  if path.is_file():
    return path
  if path.is_dir():
    candidates = sorted(
        list(path.rglob('*.pkl')) + list(path.rglob('*.pickle')))
    preferred = [item for item in candidates
                 if item.name in ('model.pkl', 'model.pickle')]
    if preferred:
      return preferred[0]
    if candidates:
      return candidates[0]
  raise ModelCorruptError('model corrupt')


def _log_failure(event, exc):
  '''Log a load failure with code and type only.

  The exception message is never logged and no traceback
  (``exc_info``) is attached, so secrets, URIs, and
  paths carried by the exception never reach the log.
  Scrubbing alone is not sufficient here: it can leave
  path prefixes intact.
  '''
  logger.warning(
      '%s code=%s type=%s', event, 'load_failed',
      type(exc).__name__)


class ModelNotConfiguredError(RuntimeError):
  '''Raised when no model loader/configuration is available.'''


class ModelNotFoundError(RuntimeError):
  '''Raised when the requested model artifact/version is missing.'''


class ModelCorruptError(RuntimeError):
  '''Raised when the model artifact is malformed or unusable.'''


class DegradedLoader:
  '''Loader that always reports the model as not configured.'''

  def load(self):
    '''Raise ModelNotConfiguredError.'''
    raise ModelNotConfiguredError('model not configured')

  def __call__(self):
    '''Raise ModelNotConfiguredError.'''
    return self.load()


class FileModelLoader:
  '''Load a pickled model from a file path lazily.

  Args:
    artifact_path: Path to a pickle model file.
    max_artifact_bytes: Artifact size cap, overriding
      the env vars and the default.
    expected_sha256: Expected SHA-256 hex digest of the
      artifact bytes, overriding ``MODEL_SHA256``.
  '''

  def __init__(self, artifact_path: str,
               max_artifact_bytes=None,
               expected_sha256=None) -> None:
    self.artifact_path = artifact_path
    self._max_artifact_bytes = max_artifact_bytes
    self._expected_sha256 = expected_sha256
    if expected_sha256 is not None:
      _expected_sha256(expected_sha256)
    if max_artifact_bytes is not None:
      _max_artifact_bytes(max_artifact_bytes)
    self._model = None
    self._lock = threading.Lock()

  def load(self):
    '''Load and cache the model, returning it.

    Raises:
      ModelNotFoundError: When the artifact is missing; the
        message never contains the server filesystem path.
      ModelCorruptError: When the pickle is malformed,
        oversized, fails its SHA-256 pin, or references
        forbidden globals.
    '''
    if self._model is None:
      with self._lock:
        if self._model is None:
          self._model = self._load_file(
              self.artifact_path, self._max_artifact_bytes,
              self._expected_sha256)
    return self._model

  def __call__(self):
    '''Return the cached model, loading it on first call.'''
    return self.load()

  @staticmethod
  def _load_file(path: str, max_artifact_bytes=None,
                 expected_sha256=None):
    '''Load model from a pickle file via restricted unpickler.

    Args:
      path: Filesystem path to the pickle artifact.
      max_artifact_bytes: Size cap overriding env/default.
      expected_sha256: SHA-256 pin overriding env.

    Returns:
      The unpickled model.

    Raises:
      ModelNotFoundError: With a generic message (no leak).
      ModelCorruptError: When the pickle is forbidden,
        malformed, oversized, or fails its SHA pin.
    '''
    cap = _max_artifact_bytes(max_artifact_bytes)
    expected = _expected_sha256(expected_sha256)
    raw = _read_file_capped(path, cap)
    _verify_sha256(raw, expected)
    try:
      return restricted_loads(raw)
    except Exception:  # pylint: disable=broad-exception-caught
      raise ModelCorruptError('model artifact corrupt') from None


class RegistryModelLoader:
  '''Load a model from the MLflow registry lazily.

  The registry must be trusted for URIs and versions,
  but artifact bytes are still decoded with the
  restricted unpickler, size-capped, and optionally
  SHA-256-pinned before use.

  Args:
    registry_uri: MLflow registry URI, or None for the
      ambient configuration.
    model_name: Registered model name.
    alias: Registry alias, default ``production``.
    allow_pyfunc: Explicit opt-in to the pyfunc
      fallback. Default False. When True a startup
      warning is logged: pyfunc executes arbitrary code
      from the registry model.
    tracking_uri: MLflow tracking URI, kept separate
      from ``registry_uri``. When None, ``registry_uri``
      is used for both (sqlite registries coincide).
    max_artifact_bytes: Artifact size cap, overriding
      the env vars and the default.
    expected_sha256: Expected SHA-256 hex digest of the
      artifact bytes, overriding ``MODEL_SHA256``.
  '''

  def __init__(
      self,
      registry_uri: str | None = None,
      model_name: str | None = None,
      alias: str = 'production',
      allow_pyfunc: bool = False,
      tracking_uri: str | None = None,
      max_artifact_bytes=None,
      expected_sha256=None,
  ) -> None:
    validate_model_name(model_name)
    validate_alias(alias)
    if registry_uri:
      validate_registry_uri(registry_uri)
    if tracking_uri:
      validate_registry_uri(tracking_uri)
    self.registry_uri = registry_uri
    self.tracking_uri = tracking_uri
    self.model_name = model_name
    self.alias = alias
    self.allow_pyfunc = bool(allow_pyfunc)
    self._max_artifact_bytes = max_artifact_bytes
    self._expected_sha256 = expected_sha256
    if expected_sha256 is not None:
      _expected_sha256(expected_sha256)
    if max_artifact_bytes is not None:
      _max_artifact_bytes(max_artifact_bytes)
    if self.allow_pyfunc:
      logger.warning(
        'allow_pyfunc=True: mlflow pyfunc fallback enabled; '
        'pyfunc executes arbitrary code from the registry '
        'model -- only use with a fully trusted registry')
    self._model = None
    self._lock = threading.Lock()

  def load(self):
    '''Load and cache the registry model, returning it.

    Raises:
      ModelNotConfiguredError: When mlflow is unavailable.
      ModelNotFoundError: When the alias/model is not found.
      ModelCorruptError: For any other load failure.
        Messages never contain the registry URI, model
        path, or underlying error.
    '''
    if self._model is None:
      with self._lock:
        if self._model is None:
          self._model = self._load_registry()
    return self._model

  def __call__(self):
    '''Return the cached model, loading it on first call.'''
    return self.load()

  def _make_client(self, mlflow):
    '''Build an MlflowClient with separate tracking/registry.

    Args:
      mlflow: Imported mlflow module.

    Returns:
      Configured MlflowClient.
    '''
    tracking = None
    try:
      tracking = importlib.import_module('mlflow.tracking')
    except ImportError:
      tracking = getattr(mlflow, 'tracking', None)
    if tracking is None or not hasattr(
        tracking, 'MlflowClient'):
      raise ModelNotConfiguredError('model not configured')
    tracking_uri = self.tracking_uri or self.registry_uri
    registry_uri = self.registry_uri or self.tracking_uri
    return tracking.MlflowClient(
        tracking_uri=tracking_uri, registry_uri=registry_uri)

  @staticmethod
  def _version_field(version_obj, name):
    '''Return a field from a version object or mapping.'''
    if isinstance(version_obj, dict):
      return version_obj.get(name)
    return getattr(version_obj, name, None)

  def _download_bytes(self, client, version_obj):
    '''Download artifact bytes for a resolved version.

    Args:
      client: Configured MlflowClient.
      version_obj: Resolved model version object.

    Returns:
      Downloaded artifact bytes, size-capped.

    Raises:
      ModelCorruptError: When bytes cannot be obtained
        or exceed the size cap.
    '''
    cap = _max_artifact_bytes(self._max_artifact_bytes)
    run_id = self._version_field(version_obj, 'run_id')
    source = self._version_field(version_obj, 'source')
    if source is None:
      source = self._version_field(version_obj, 'artifact_uri')
    downloaded = None
    if run_id and hasattr(client, 'download_artifacts'):
      with tempfile.TemporaryDirectory() as tmp:
        try:
          downloaded = client.download_artifacts(
              str(run_id), '', dst_path=tmp)
        except TypeError:
          downloaded = client.download_artifacts(
              str(run_id), '', tmp)
        return _read_file_capped(
            _pickle_path_in(downloaded), cap)
    if source:
      local = str(source)
      if local.startswith('file://'):
        local = local[len('file://'):]
      if os.path.isfile(local):
        return _read_file_capped(local, cap)
      artifacts = getattr(self._import_mlflow(), 'artifacts', None)
      if artifacts is not None and hasattr(
          artifacts, 'download_artifacts_from_uri'):
        local_path = artifacts.download_artifacts_from_uri(local)
        return _read_file_capped(
            _pickle_path_in(local_path), cap)
    raise ModelCorruptError('model corrupt')

  @staticmethod
  def _import_mlflow():
    '''Import mlflow or raise ModelNotConfiguredError.'''
    try:
      import mlflow
    except ImportError:
      raise ModelNotConfiguredError(
        'model not configured') from None
    return mlflow

  def _load_registry(self):
    '''Load ``models:/<name>@<alias>`` via downloaded bytes.

    The alias is resolved through the tracking client
    and the resolved version is logged. Artifact bytes
    are downloaded, size-capped, SHA-pinned, and decoded
    with ``restricted_loads`` only. Alias/model-not-found
    becomes a ModelNotFoundError, missing mlflow a
    ModelNotConfiguredError, and every other failure a
    ModelCorruptError, so no URI/detail leaks. The
    pyfunc fallback runs only when ``allow_pyfunc=True``
    and restricted loading fails.
    '''
    mlflow = self._import_mlflow()
    try:
      client = self._make_client(mlflow)
    except ModelNotConfiguredError:
      raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
      _log_failure('registry client failed', exc)
      raise ModelCorruptError('model corrupt') from None
    try:
      version_obj = client.get_model_version_by_alias(
          self.model_name, self.alias)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      if _is_alias_not_found(exc):
        logger.warning(
          'registry model %r alias %r not found',
          self.model_name, self.alias)
        raise ModelNotFoundError('model not found') from None
      _log_failure('registry version lookup failed', exc)
      raise ModelCorruptError('model corrupt') from None
    version = self._version_field(version_obj, 'version')
    logger.info(
      'registry model resolved name=%r alias=%r version=%r',
      self.model_name, self.alias, str(version))
    try:
      raw = self._download_bytes(client, version_obj)
    except (ModelCorruptError, ModelNotFoundError):
      raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
      _log_failure('registry artifact download failed', exc)
      raise ModelCorruptError('model corrupt') from None
    _verify_sha256(raw, _expected_sha256(self._expected_sha256))
    try:
      return restricted_loads(raw)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      if not self.allow_pyfunc:
        _log_failure('registry artifact load failed', exc)
        raise ModelCorruptError('model corrupt') from None
      logger.warning(
        'restricted load failed for %r@%r; falling back to '
        'pyfunc (allow_pyfunc=True)',
        self.model_name, self.alias)
      try:
        uri = f'models:/{self.model_name}@{self.alias}'
        return mlflow.pyfunc.load_model(uri)
      except Exception:  # pylint: disable=broad-exception-caught
        raise ModelCorruptError('model corrupt') from None
