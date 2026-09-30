#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Object-storage persistence helpers.

The framework round-trips two kinds of artefact through cloud object storage:
the fitted feature registry and the live trainer instance. Both are pickled,
which is the framework's central packaging decision and also its single largest
reproducibility risk . These helpers centralise that access so
a version stamp can be attached in one place.
"""

from __future__ import annotations

import pickle
from typing import Any
from urllib.parse import urlparse

from forecasting_ml_framework.exceptions import FrameworkError
from forecasting_ml_framework.observability.logging import get_logger

LOGGER = get_logger(__name__)

#: Bumped whenever the pickled layout of a model handle or registry changes in a
#: way that is not forward- or backward-compatible. A stored stamp lets a
#: scoring run detect a mismatch instead of failing deep inside ``pickle``.
ARTEFACT_FORMAT_VERSION = '1.0.0'


def get_file_system(path: str | None = None, project: str | None = None) -> Any:
  """Return a filesystem handle chosen by the address, not by what is installed.

  The implementation is *derived from the path*. The previous version tried the
  cloud filesystem first and unconditionally, so on any machine with it installed
  a bare local path was silently routed through it -- which is precisely the
  address-form hazard the local artefact root introduces, and precisely the case a
  developer hits first. Dispatching on the scheme removes the whole class: a
  managed address reaches the managed store, and a bare path reaches the host.

  This is also the one place the two artefact roots are reconciled. Both are the
  *same* filesystem interface at different roots, so the substitution is exact for
  reads and writes and the only thing that needs care is not mangling the address.

  Args:
    path: The address the filesystem will be used for. Its scheme selects the
      implementation. When absent, the cloud store is used, which is the correct
      default for a managed run and the wrong one for a local one -- so a caller
      working against a local root must pass its path.
    project: The cloud project, applied only when the address resolves to the cloud.

  Returns:
    An ``fsspec``-compatible filesystem instance.

  Raises:
    FrameworkError: If no supported filesystem implementation is importable.
  """
  import fsspec  # noqa: PLC0415 - deferred so the core imports without fsspec

  scheme = urlparse(str(path or '')).scheme
  if scheme in ('', 'file'):
    return fsspec.filesystem('file')
  try:
    import gcsfs  # noqa: PLC0415 - deferred so the core imports without gcsfs
  except ImportError as error:
    raise FrameworkError(
      f'The address {path!r} names the {scheme!r} scheme but no filesystem implementation for it '
      f'is installed. Artefact access must not silently fall back to the local filesystem: a run '
      'would write its model weights somewhere it cannot find them.',
      path=str(path),
      scheme=scheme,
    ) from error
  return gcsfs.GCSFileSystem(token='google_default', project=project)


def save_pickle(obj: Any, path: str) -> str:
  """Serialise an object to object storage with a format stamp.

  Args:
    obj: The object to persist. It must be picklable.
    path: The destination path. A directory path receives ``artefact.pkl``.

  Returns:
    The resolved path that was written.

  Raises:
    FrameworkError: If the object cannot be serialised.
  """
  payload = {'__format_version__': ARTEFACT_FORMAT_VERSION, 'payload': obj}
  target = _resolve_filepath(path)
  filesystem = get_file_system(target)
  try:
    with filesystem.open(target, 'wb') as handle:
      pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
  except (OSError, pickle.PicklingError, TypeError, AttributeError) as error:
    raise FrameworkError(
      f'Failed to persist artefact to {target!r}. Pickled model handles pin the estimator class '
      'definition and the installed library versions, so a failure here usually indicates a '
      'non-picklable member rather than a storage problem.',
      path=target,
      error=str(error),
    ) from error
  LOGGER.info('Saved pickle artefact', extra={'path': target})
  return target


def load_pickle(path: str) -> Any:
  """Deserialise an object previously written by :func:`save_pickle`.

  Args:
    path: The path to read.

  Returns:
    The unpickled object.

  Raises:
    FrameworkError: If the artefact is missing or unreadable.
    FrameworkError: If the stored format version is incompatible.
  """
  target = _resolve_filepath(path)
  filesystem = get_file_system(target)
  try:
    with filesystem.open(target, 'rb') as handle:
      payload = pickle.load(handle)  # noqa: S301 - artefacts are written by this framework only
  except FileNotFoundError as error:
    raise FrameworkError(f'Pickle artefact not found: {target!r}', path=target) from error
  except Exception as error:  # noqa: BLE001 - unpickling can raise almost anything
    raise FrameworkError(
      f'Failed to unpickle artefact at {target!r}. This is almost always a library-version '
      'mismatch: the artefact was written by a container image whose pinned versions differ '
      'from the current one.',
      path=target,
      error=str(error),
    ) from error

  if isinstance(payload, dict) and '__format_version__' in payload:
    stored = payload['__format_version__']
    if stored != ARTEFACT_FORMAT_VERSION:
      raise FrameworkError(
        f'Artefact format mismatch at {target!r}: stored {stored!r}, current {ARTEFACT_FORMAT_VERSION!r}',
        path=target,
      )
    return payload['payload']
  return payload


def _resolve_filepath(path: str) -> str:
  """Normalise a destination path to a concrete object path.

  Args:
    path: A file path or a directory path.

  Returns:
    A path ending in a filename.
  """
  if path.endswith('.pkl') or path.endswith('.pkt'):
    return path
  return f'{path.rstrip("/")}/artefact.pkl'
