#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Versioned parquet dataset over any ``fsspec`` filesystem.

This is the natural replacement when a versioned, partitioned
on-object-storage layout is wanted, and it is a supported catalog ``type:``.
"""

from __future__ import annotations

from typing import Any

from kedro.io.core import AbstractDataset

from forecasting_ml_framework.exceptions import DatasetError
from forecasting_ml_framework.observability.logging import get_logger

LOGGER = get_logger(__name__)


class ParquetDataSet(AbstractDataset):
  """A pandas ``DataFrame`` stored as parquet, single-file or partitioned.

  Attributes:
    _version: An optional path segment, which is what makes a rerun reversible.
    _partitioned: Whether the path is a directory of parts rather than a file.
  """

  def __init__(
    self,
    filepath: str,
    version: str | None = None,
    partitioned: bool = True,
    credentials: dict[str, Any] | None = None,
    load_args: dict[str, Any] | None = None,
    save_args: dict[str, Any] | None = None,
  ) -> None:
    """Validate the configuration eagerly; perform no I/O.

    Args:
      filepath: A ``fsspec`` path, with or without the parquet suffix.
      version: An optional version segment appended to the path.
      partitioned: ``True`` when the path is a directory of parts.
      credentials: Optional cloud credentials.
      load_args: Overrides for ``pandas.read_parquet``.
      save_args: Overrides for ``DataFrame.to_parquet``.

    Raises:
      DatasetError: If no filepath is supplied.
    """
    if not filepath:
      raise DatasetError("Missing required configuration 'filepath'")
    self._base_filepath = filepath
    self._version = version
    self._partitioned = partitioned
    self._credentials = credentials
    self._load_args = dict(load_args or {})
    self._save_args = dict(save_args or {})
    self._versioned_path = f'{filepath.rstrip("/")}/{version}' if version else filepath.rstrip('/')

  def _describe(self) -> dict[str, Any]:
    """Return every configuration key.

    Returns:
      A mapping suitable for the run log.
    """
    return {
      'filepath': self._versioned_path,
      'version': self._version,
      'partitioned': self._partitioned,
    }

  def _load(self) -> Any:
    """Read the parquet payload.

    Returns:
      A pandas ``DataFrame``.

    Raises:
      DatasetError: If the payload is absent or unreadable.
    """
    import pandas as pd  # noqa: PLC0415

    from forecasting_ml_framework.utils.storage import get_file_system  # noqa: PLC0415

    path = self._resolve(exists_only=True)
    filesystem = get_file_system(path)
    try:
      with filesystem.open(path, 'rb') as handle:
        return pd.read_parquet(handle, **self._load_args)
    except FileNotFoundError as error:
      raise DatasetError(f'Parquet payload not found: {path}', filepath=path) from error
    except Exception as error:  # noqa: BLE001 - pyarrow raises many types
      raise DatasetError(f'Unable to read the parquet payload at {path}', filepath=path, error=str(error)) from error

  def _save(self, data: Any) -> None:
    """Write the parquet payload.

    Args:
      data: The pandas ``DataFrame`` to write.
    """
    from forecasting_ml_framework.utils.storage import get_file_system  # noqa: PLC0415

    path = self._resolve(exists_only=False)
    filesystem = get_file_system(path)
    try:
      with filesystem.open(path, 'wb') as handle:
        data.to_parquet(handle, **self._save_args)
    except OSError as error:
      raise DatasetError(f'Unable to write the parquet payload at {path}', filepath=path, error=str(error)) from error
    LOGGER.info('Wrote parquet dataset', extra={'path': path, 'rows': len(data)})

  def _resolve(self, exists_only: bool) -> str:
    """Resolve the concrete path for a payload.

    The payload is a single columnar file whether or not ``partitioned`` is set.
    The flag is retained for catalogue compatibility and is reported by
    :meth:`_describe`, but it does not select between a file and a directory: the
    read and write paths were byte-identical in both branches, so a "partitioned"
    dataset was in fact a single file and the flag described nothing.

    Args:
      exists_only: Whether the path is being resolved for a read. Accepted so the
        caller reads the same at both ends; the suffix does not depend on it.

    Returns:
      The resolved path.
    """
    del exists_only
    return f'{self._versioned_path}.parquet'
