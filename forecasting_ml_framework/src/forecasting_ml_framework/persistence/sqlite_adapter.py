#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Embedded SQLite dataset adapter for the persistence layer.

This adapter stands in for the distributed BigQuery adapter when the local
execution flag selects the embedded relational back end. It wraps the
:class:`sqlite_warehouse.SQLiteWarehouse` so the production adapter
contract -- ``_load``, ``_save``, ``_exists``, ``_describe``, and the five
write dispositions -- is satisfied by a single file instead of a managed
warehouse.

The adapter does not replicate BigQuery-specific features such as partitioned
upsert staging, scratch datasets, or connector-level session management. Those
are out of scope for a single-file store.
'''

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from kedro.io.core import AbstractDataset

from forecasting_ml_framework.local.sqlite_warehouse import (
 WRITE_TRUNCATE,
 WRITE_APPEND,
 WRITE_EMPTY,
 SQLiteWarehouse,
 SqliteWarehouseError,
)
from forecasting_ml_framework import constants
from forecasting_ml_framework.observability.logging import get_logger

LOGGER = get_logger(__name__)

WRITE_MODES = (WRITE_TRUNCATE, WRITE_APPEND, WRITE_EMPTY)


class EmbeddedSqlDataset(AbstractDataset):
 '''A SQLite-backed dataset adapter using the embedded warehouse.

 The adapter opens a single SQLite file per dataset and provides the same
 load, save, describe, and existence interface as the production
 ``CustomSparkBQDataSet`` adapter, so a local run exercises the same
 contract without requiring a managed engine.
 '''

 def __init__(
  self,
  table_name: str | None = None,
  database: str | None = None,
  path: str | Path | None = None,
  write_mode: str = WRITE_TRUNCATE,
  if_exists: str | None = None,
 ) -> None:
  '''Validate eagerly; open no connection yet.

  Args:
   table_name: The table or dataset identifier.
   database: The fully-qualified name used for description.
   path: The SQLite file path. Defaults to the local artefact root if
    ``is_local`` is set.
   write_mode: One of ``WRITE_TRUNCATE``, ``WRITE_APPEND``,
    ``WRITE_EMPTY``.
   if_exists: Overrides ``write_mode`` with ``'replace'`` or
    ``'append'``.

  Raises:
   ValueError: If ``write_mode`` or ``if_exists`` is unsupported.
  '''
  super().__init__()
  self.table_name = table_name or (database or 'dataset')
  self.database = database
  resolved_path = path or constants.LOCAL_ARTEFACT_ROOT
  self.path = Path(resolved_path)
  if if_exists is not None:
   self.write_mode = if_exists.lower() if if_exists == 'append' else WRITE_TRUNCATE
  else:
   self.write_mode = write_mode
  if self.write_mode not in WRITE_MODES:
   raise ValueError(
    f'Unsupported write mode: {self.write_mode!r}; '
    f'supported: {list(WRITE_MODES)}.',
   )
  self._warehouse: SQLiteWarehouse | None = None
  LOGGER.info('Initialised embedded adapter', extra={
   'table': self.table_name, 'path': str(self.path),
   'write_mode': self.write_mode,
  })

 def _load(self) -> Any:
  '''Read the table into a frame.

  Returns:
   A pandas ``DataFrame``.
  '''
  LOGGER.info('Loading embedded dataset', extra={'table': self.table_name})
  if self._warehouse is None:
   self._warehouse = SQLiteWarehouse(str(self.path))
  if not self._warehouse.exists(self.table_name):
   LOGGER.warning(
    'Table not present; returning empty frame.',
    extra={'table': self.table_name},
   )
   return pd.DataFrame()
  return self._warehouse.query(f'SELECT * FROM "{self.table_name}"')

 def _save(self, data: Any) -> None:
  '''Write the frame under the configured disposition.

  Args:
   data: A ``pandas.DataFrame``.
  '''
  LOGGER.info('Saving embedded dataset', extra={
   'table': self.table_name, 'write_mode': self.write_mode,
  })
  if self._warehouse is None:
   self._warehouse = SQLiteWarehouse(str(self.path))
  self._warehouse.write(
   data,
   self.table_name,
   disposition=self.write_mode,
   if_exists='replace' if self.write_mode == WRITE_TRUNCATE else 'append',
  )

 def _exists(self) -> bool:
  '''Report whether the table exists.

  Returns:
   ``True`` when present.
  '''
  if self._warehouse is None:
   self._warehouse = SQLiteWarehouse(str(self.path))
  return self._warehouse.exists(self.table_name)

 def _describe(self) -> dict[str, Any]:
  '''Return every configuration key for audit and comparison.

  Returns:
   A dictionary of the adapter's settings.
  '''
  return {
   'table_name': self.table_name,
   'database': self.database,
   'path': str(self.path),
   'write_mode': self.write_mode,
   'adapter_type': 'embedded_sqlite',
  }

 def _getstate__(self) -> dict[str, Any]:
  '''Remove live engine objects from the state.

  Returns:
   The serialisable settings.
  '''
  state = self._describe()
  return state
