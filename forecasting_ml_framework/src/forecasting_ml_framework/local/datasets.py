#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kedro dataset adapters backed by the local SQLite warehouse.

These are the local counterparts of the three production warehouse adapters --
the Spark-native query dataset, the pandas table dataset and the score table.
Each keeps the same constructor shape as its production counterpart so a catalog
entry written for the warehouse resolves to one of these with only the ``type:``
line changed, and each implements the same write disposition semantics.

The Spark adapter is the one that matters most. The feature-engineering tier runs
as Spark transformations over a Spark DataFrame, and a local run that handed it
a pandas frame would be testing a different code path than production. So the
Spark adapter converts at the boundary, in both directions, and the routines
never learn that the warehouse is not BigQuery.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from kedro.io.core import AbstractDataset

from forecasting_ml_framework.local.sql_translate import resolve_parameters, to_sqlite
from forecasting_ml_framework.local.sqlite_warehouse import (
  WRITE_TRUNCATE,
  SQLiteWarehouse,
  SqliteWarehouseError,
)
from forecasting_ml_framework.observability.logging import get_logger

LOGGER = get_logger(__name__)

#: Write modes the Spark adapter accepts, named as the production catalog names
#: them. Each maps onto one write disposition.
_WRITE_MODE_DISPOSITIONS = {
  'overwrite': WRITE_TRUNCATE,
  'create_replace': WRITE_TRUNCATE,
  'insert': 'WRITE_APPEND',
  'insert_overwrite': 'WRITE_APPEND',
  'upsert': 'WRITE_APPEND',
}


class SQLiteSparkDataSet(AbstractDataset):
  """A Spark DataFrame bound to a SQLite table or to a query.

  Args:
    table_name: The destination table for writes.
    sql: An optional query, used for reads. Takes precedence over ``table_name``.
    database: Accepted for catalog compatibility; the local warehouse is a
      single file, so the qualifier carries no information and is ignored.
    write_mode: One of :data:`_WRITE_MODE_DISPOSITIONS`.
    warehouse_path: The SQLite file backing this dataset.
  """

  def __init__(
    self,
    table_name: str,
    warehouse_path: str,
    sql: str | None = None,
    database: str | None = None,
    write_mode: str = 'overwrite',
    warehouse_path_is_dir: bool = False,
  ) -> None:
    self._table_name = table_name
    self._sql = sql
    self._database = database
    self._warehouse_path = warehouse_path
    self._warehouse_path_is_dir = warehouse_path_is_dir
    if write_mode not in _WRITE_MODE_DISPOSITIONS:
      raise SqliteWarehouseError(
        f'Unsupported write_mode {write_mode!r}; expected one of {sorted(_WRITE_MODE_DISPOSITIONS)}.'
      )
    self._write_mode = write_mode

  def _describe(self) -> dict[str, Any]:
    """Describe the dataset for ``kedro catalog`` and the run report.

    Returns:
      A mapping of the adapter's configuration.
    """
    return {
      'table_name': self._table_name,
      'warehouse_path': self._warehouse_path,
      'write_mode': self._write_mode,
      'backend': 'sqlite',
    }

  def _warehouse(self) -> SQLiteWarehouse:
    """Open the warehouse for the duration of one operation.

    Returns:
      A connected warehouse. The caller closes it, so a long-lived driver
      cannot leak a file handle by holding a dataset.
    """
    return SQLiteWarehouse(self._warehouse_path)

  def _load(self) -> Any:
    """Read into a Spark DataFrame.

    Returns:
      A Spark ``DataFrame``.
    """
    from pyspark.sql import SparkSession  # noqa: PLC0415

    warehouse = self._warehouse()
    try:
      if self._sql is not None:
        frame = warehouse.query(to_sqlite(self._sql))
      else:
        if not warehouse.exists(self._table_name):
          raise SqliteWarehouseError(
            f'Table {self._table_name!r} does not exist in the local warehouse '
            f'at {self._warehouse_path}. Available: {warehouse.table_names()}'
          )
        frame = warehouse.query(f'SELECT * FROM "{self._table_name}"')
    finally:
      warehouse.close()

    LOGGER.info('Read a warehouse dataset', extra={'table': self._table_name, 'rows': len(frame)})
    return SparkSession.getActiveSession().createDataFrame(frame)


class SQLiteTableDataSet(AbstractDataset):
  """A pandas ``DataFrame`` bound to a SQLite table.

  The pandas counterpart of the Spark adapter, used for the metric tables and
  the score table.

  Args:
    table_name: The destination table.
    warehouse_path: The SQLite file backing this dataset.
    write_disposition: The catalog spelling of the disposition.
    if_exists: The ``to_gbq`` spelling, which takes precedence.
    partition_field: Accepted for catalog compatibility and recorded in the
      description. The local warehouse has no partitioning, so the column is
      carried in the data rather than in the table definition.
  """

  def __init__(
    self,
    table_name: str,
    warehouse_path: str,
    write_disposition: str = WRITE_TRUNCATE,
    if_exists: str | None = None,
    partition_field: str | None = None,
    partition_value: str | None = None,
    clustering_fields: list[str] | None = None,
  ) -> None:
    self._table_name = table_name
    self._warehouse_path = warehouse_path
    self._write_disposition = write_disposition
    self._if_exists = if_exists
    self._partition_field = partition_field
    self._partition_value = partition_value
    self._clustering_fields = list(clustering_fields or [])

  def _describe(self) -> dict[str, Any]:
    """Describe the dataset.

    Returns:
      A mapping of the adapter's configuration.
    """
    return {
      'table_name': self._table_name,
      'warehouse_path': self._warehouse_path,
      'write_disposition': self._write_disposition,
      'partition_field': self._partition_field,
      'backend': 'sqlite',
    }

  def _load(self) -> Any:
    """Read the table.

    Returns:
      A pandas ``DataFrame``.
    """
    warehouse = SQLiteWarehouse(self._warehouse_path)
    try:
      if not warehouse.exists(self._table_name):
        raise SqliteWarehouseError(
          f'Table {self._table_name!r} does not exist. Available: {warehouse.table_names()}'
        )
      return warehouse.query(f'SELECT * FROM "{self._table_name}"')
    finally:
      warehouse.close()

  def _save(self, data: Any) -> None:
    """Write the table under the configured disposition.

    Args:
      data: The pandas ``DataFrame`` to write.
    """
    if data is None:
      return
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if frame.empty:
      LOGGER.warning('Skipping an empty warehouse write', extra={'table': self._table_name})
      return
    warehouse = SQLiteWarehouse(self._warehouse_path)
    try:
      written = warehouse.write(
        frame, self._table_name, disposition=self._write_disposition, if_exists=self._if_exists
      )
    finally:
      warehouse.close()
    LOGGER.info('Wrote a warehouse table', extra={'table': self._table_name, 'rows': written})


class SQLiteQueryDataSet(SQLiteTableDataSet):
  """A pandas ``DataFrame`` produced by a query rather than read from a table.

  The label-derivation queries are the reason this exists: the local run must
  execute the same SQL the production catalog declares, so the query is
  translated and run rather than replaced by an in-memory join.
  """

  def __init__(self, sql: str, warehouse_path: str, **kwargs: Any) -> None:
    self._sql = sql
    super().__init__(table_name='<query>', warehouse_path=warehouse_path, **kwargs)

  def _describe(self) -> dict[str, Any]:
    """Describe the dataset.

    Returns:
      A mapping including the query, for the run report.
    """
    description = super()._describe()
    description['sql'] = self._sql
    return description

  def _load(self) -> Any:
    """Execute the query.

    Returns:
      A pandas ``DataFrame``.
    """
    warehouse = SQLiteWarehouse(self._warehouse_path)
    try:
      return warehouse.query(to_sqlite(self._sql))
    finally:
      warehouse.close()


def ensure_parent(path: str | Path) -> Path:
  """Create a directory's parent and return the path.

  Args:
    path: The file path whose parent should exist.

  Returns:
    The same path, with its parent created.
  """
  resolved = Path(path)
  resolved.parent.mkdir(parents=True, exist_ok=True)
  return resolved


def load_from_query(
  warehouse_path: str,
  sql: str,
  parameters: dict[str, Any] | None = None,
  globals_: dict[str, Any] | None = None,
) -> pd.DataFrame:
  """Resolve, translate and run a catalog query, returning a frame.

  The three steps are kept together because each is a precondition of the next
  and separating them at the call site is how a query ends up executed with an
  unexpanded ``${...}`` reference.

  Args:
    warehouse_path: The SQLite file.
    sql: The catalog query.
    parameters: The parameters document, for ``${section:key}`` expansion.
    globals_: The globals document, for ``${globals:key}`` expansion.

  Returns:
    A pandas ``DataFrame``.
  """
  expanded = resolve_parameters(sql, parameters or {}, globals_ or {})
  warehouse = SQLiteWarehouse(warehouse_path)
  try:
    return warehouse.query(to_sqlite(expanded))
  finally:
    warehouse.close()
