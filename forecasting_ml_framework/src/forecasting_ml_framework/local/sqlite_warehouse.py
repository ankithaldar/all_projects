#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A local SQLite substitute for the production warehouse.

Everything a production run gets from a cloud data warehouse is a table you can
query, a table you can write, and a write disposition that says what happens
when the table already exists. This module provides exactly those three things
against a single SQLite file, and it is deliberately small: a local harness
whose job is to let the real catalog SQL execute, not to be a database.

What is faithful to the production semantics, and why each matters:

* **Write dispositions.** ``WRITE_TRUNCATE`` replaces a table outright, which is
  what makes a metric snapshot idempotent -- re-running a day must replace the
  day's rows, not append a second copy. ``WRITE_APPEND`` extends a partitioned
  time series. Treating them as the same operation is how a metrics table grows
  duplicate rows for the same period without any error.
* **A query is a query.** Reading goes through SQL, not through a cached
  DataFrame, so a local run exercises the catalog's joins, filters and window
  functions rather than skipping them.
* **Types are declared.** SQLite is dynamically typed, and a table created by
  inference will silently accept a string into an integer column. Every write
  therefore carries an explicit column type list, so a type regression surfaces
  as an error at write time.

What is not faithful, and is not claimed to be: there is no concurrency control,
no partitioning, no clustering, and no cost model. A local run is a correctness
harness, not a performance one.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

#: Write dispositions, named as the catalog names them so a catalog entry
#: written for the production warehouse runs here unchanged.
WRITE_TRUNCATE = 'WRITE_TRUNCATE'
WRITE_APPEND = 'WRITE_APPEND'
WRITE_EMPTY = 'WRITE_EMPTY'

SUPPORTED_WRITE_DISPOSITIONS = (WRITE_TRUNCATE, WRITE_APPEND, WRITE_EMPTY)

#: pandas dtype -> SQLite column type. SQLite has no boolean or datetime
#: storage class worth using here, so both are mapped to something that round
#: trips through pandas rather than to a type that loses information.
_SQL_TYPES = {
  'int64': 'INTEGER',
  'int32': 'INTEGER',
  'int16': 'INTEGER',
  'float64': 'REAL',
  'float32': 'REAL',
  'bool': 'INTEGER',
  'object': 'TEXT',
  'string': 'TEXT',
  'category': 'TEXT',
  'datetime64[ns]': 'TEXT',
}


class SQLiteWarehouse:
  """A single-file SQLite warehouse standing in for the cloud data warehouse.

  The class is a thin, explicit wrapper rather than a connection pool. Every
  method is a direct statement of what the production adapter would do, so the
  two can be read side by side.
  """

  def __init__(self, path: str | Path) -> None:
    """Open or create the warehouse file.

    Args:
      path: The SQLite file. Parent directories are created.

    Raises:
      SqliteWarehouseError: If the file cannot be opened.
    """
    self._path = Path(path)
    try:
      self._path.parent.mkdir(parents=True, exist_ok=True)
      self._connection = sqlite3.connect(str(self._path))
    except (OSError, sqlite3.Error) as error:
      raise SqliteWarehouseError(f'Could not open the local warehouse at {self._path}: {error}') from error
    self._connection.row_factory = sqlite3.Row

  @property
  def path(self) -> Path:
    """Return the warehouse file path.

    Returns:
      The path this warehouse was opened at.
    """
    return self._path

  @property
  def connection(self) -> sqlite3.Connection:
    """Return the underlying connection.

    Returns:
      The live ``sqlite3`` connection, for callers composing their own SQL.

    Raises:
      SqliteWarehouseError: If the warehouse has been closed. Reopening lazily would
        hide a use-after-close, which on a single-file store means writing through a
        handle nobody will commit.
    """
    if self._connection is None:
      raise SqliteWarehouseError(
        f'The warehouse at {self._path} is closed. Open a new one rather than reusing this handle.'
      )
    return self._connection

  # ------------------------------------------------------------------ #
  # Reads
  # ------------------------------------------------------------------ #
  def query(self, sql: str, parameters: Sequence[Any] | None = None) -> pd.DataFrame:
    """Execute a query and return the result as a frame.

    Args:
      sql: The query, already translated to the SQLite dialect.
      parameters: Bind parameters, positional.

    Returns:
      A pandas ``DataFrame``. An empty result yields an empty frame with the
      columns the query selected, so a downstream column check fails on a
      missing column rather than on a missing frame.
    """
    cursor = self.connection.execute(sql, parameters or ())
    columns = [description[0] for description in cursor.description or ()]
    rows = cursor.fetchall()
    if not rows:
      return pd.DataFrame(columns=columns)
    return pd.DataFrame([tuple(row) for row in rows], columns=columns)

  def table_names(self) -> list[str]:
    """List the tables the warehouse holds.

    Returns:
      Table names, sorted.
    """
    return [row['name'] for row in self.connection.execute(
      "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )]

  def exists(self, table_name: str) -> bool:
    """Report whether a table is present.

    Args:
      table_name: The table name.

    Returns:
      ``True`` when the table exists.
    """
    row = self.connection.execute(
      "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table_name,)
    ).fetchone()
    return row is not None

  def count(self, table_name: str) -> int:
    """Count the rows in a table.

    Args:
      table_name: The table name.

    Returns:
      The row count, or ``0`` when the table does not exist.
    """
    if not self.exists(table_name):
      return 0
    row = self.connection.execute(f'SELECT COUNT(*) AS n FROM "{table_name}"').fetchone()
    return int(row['n'])

  # ------------------------------------------------------------------ #
  # Writes
  # ------------------------------------------------------------------ #
  def write(
    self,
    frame: pd.DataFrame,
    table_name: str,
    disposition: str = WRITE_TRUNCATE,
    if_exists: str | None = None,
  ) -> int:
    """Write a frame to a table under an explicit disposition.

    Args:
      frame: The rows to write.
      table_name: The destination table.
      disposition: One of :data:`WRITE_TRUNCATE`, :data:`WRITE_APPEND` or
        :data:`WRITE_EMPTY`. :data:`WRITE_TRUNCATE` replaces the table.
      if_exists: Overrides ``disposition`` with the ``to_gbq`` spelling
        (``'replace'`` or ``'append'``), because the score-table adapter uses
        that vocabulary and the catalog must not have to be rewritten for it.

    Returns:
      The number of rows written.

    Raises:
      SqliteWarehouseError: If the disposition is unsupported, or the frame has
        no columns to write.
    """
    resolved = self._resolve_disposition(disposition, if_exists)
    if frame is None or frame.empty:
      raise SqliteWarehouseError(
        f'Refusing to write an empty frame to {table_name!r}: an empty write is a bug, not a no-op.'
      )
    if frame.shape[1] == 0:
      raise SqliteWarehouseError(f'Refusing to write a frame with no columns to {table_name!r}.')

    if resolved == WRITE_APPEND and not self.exists(table_name):
      self._create_table(table_name, list(frame.columns), list(frame.dtypes))
    elif resolved == WRITE_TRUNCATE or not self.exists(table_name):
      self._replace_table(table_name, list(frame.columns), list(frame.dtypes))
    else:
      self._check_columns(table_name, list(frame.columns))

    self._insert(frame, table_name)
    return len(frame)

  def _resolve_disposition(self, disposition: str, if_exists: str | None) -> str:
    """Map a catalog write disposition onto the internal vocabulary.

    Args:
      disposition: The catalog spelling.
      if_exists: The ``to_gbq`` spelling, which takes precedence when supplied.

    Returns:
      One of the ``WRITE_`` constants.

    Raises:
      SqliteWarehouseError: If neither spelling is recognised.
    """
    if if_exists is not None:
      mapping = {'replace': WRITE_TRUNCATE, 'append': WRITE_APPEND, 'fail': WRITE_EMPTY}
      if if_exists not in mapping:
        raise SqliteWarehouseError(
          f'Unsupported if_exists {if_exists!r}; expected one of {sorted(mapping)}.'
        )
      return mapping[if_exists]
    if disposition not in SUPPORTED_WRITE_DISPOSITIONS:
      raise SqliteWarehouseError(
        f'Unsupported write disposition {disposition!r}; expected one of {list(SUPPORTED_WRITE_DISPOSITIONS)}.'
      )
    return disposition

  def _create_table(self, table_name: str, columns: list[str], dtypes: Iterable[Any]) -> None:
    """Create a table with declared column types.

    Args:
      table_name: The table to create.
      columns: The column names, in order.
      dtypes: The matching pandas dtypes, used to declare SQLite types.
    """
    declarations = ', '.join(
      f'"{column}" {_sql_type(dtype)}' for column, dtype in zip(columns, dtypes, strict=True)
    )
    self.connection.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({declarations})')
    self.connection.commit()

  def _replace_table(self, table_name: str, columns: list[str], dtypes: Iterable[Any]) -> None:
    """Replace a table's contents, creating it when absent.

    Args:
      table_name: The table to replace.
      columns: The column names, in order.
      dtypes: The matching pandas dtypes.
    """
    self.connection.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    self._create_table(table_name, columns, dtypes)

  def _check_columns(self, table_name: str, columns: list[str]) -> None:
    """Verify an append's columns match the existing table.

    Args:
      table_name: The destination table.
      columns: The incoming column names.

    Raises:
      SqliteWarehouseError: If a column is absent from the table. An append
        that silently skips a column produces a table whose shape depends on
        write history.
    """
    existing = {
      row['name'] for row in self.connection.execute(f'PRAGMA table_info("{table_name}")')
    }
    missing = [column for column in columns if column not in existing]
    if missing:
      raise SqliteWarehouseError(
        f'Append to {table_name!r} is missing column(s) {missing} that the table already has. '
        f'A metrics table whose shape depends on write history cannot be read back.'
      )

  def _insert(self, frame: pd.DataFrame, table_name: str) -> None:
    """Insert every row of a frame.

    Args:
      frame: The rows to insert.
      table_name: The destination table.
    """
    columns = list(frame.columns)
    placeholders = ', '.join('?' for _ in columns)
    quoted = ', '.join(f'"{column}"' for column in columns)
    statement = f'INSERT INTO "{table_name}" ({quoted}) VALUES ({placeholders})'
    rows = [tuple(_sql_value(value) for value in row) for row in frame.itertuples(index=False, name=None)]
    self.connection.executemany(statement, rows)
    self.connection.commit()

  def drop(self, table_name: str) -> None:
    """Drop a table if it exists.

    Args:
      table_name: The table to drop.
    """
    self.connection.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    self.connection.commit()

  def execute(self, sql: str, parameters: Sequence[Any] | None = None) -> list[sqlite3.Row]:
    """Execute an arbitrary statement and return any rows.

    Args:
      sql: The statement.
      parameters: Bind parameters.

    Returns:
      The rows produced, which is empty for a statement that returns none.
    """
    cursor = self.connection.execute(sql, parameters or ())
    self.connection.commit()
    try:
      return cursor.fetchall()
    except sqlite3.Error:
      return []

  def close(self) -> None:
    """Close the connection.

    The close is idempotent, so a ``finally`` block and a ``with`` block can both
    release the handle without the second call raising.
    """
    if self._connection is not None:
      self._connection.close()
      self._connection = None

  def __enter__(self) -> SQLiteWarehouse:
    """Open the scope.

    Returns:
      This warehouse.
    """
    return self

  def __exit__(self, *unused: object) -> None:
    """Close the connection on scope exit, whatever happened inside.

    Args:
      unused: The exception triple, unused.

    Returns:
      ``None``, so an exception raised inside the scope propagates.
    """
    self.close()


def _sql_type(dtype: Any) -> str:
  """Map a pandas dtype onto a SQLite column type.

  Args:
    dtype: The pandas dtype.

  Returns:
    The SQLite column type.
  """
  return _SQL_TYPES.get(str(dtype), 'TEXT')


def _sql_value(value: Any) -> Any:
  """Coerce a value into something the SQLite driver accepts.

  ``NaN`` has no SQLite representation and ``None`` is the correct encoding of
  "absent", so a missing value is written as ``NULL`` rather than as the string
  ``'nan'`` -- a string that would reappear as a real category downstream.

  Timestamps are written in ISO-8601 rather than as their epoch, because the
  score table is read by humans and by SQL that compares against a date
  literal, and an epoch integer is neither.

  Args:
    value: A value from a frame cell.

  Returns:
    ``None``, a Python scalar, or a string.
  """
  import datetime  # noqa: PLC0415

  if value is None:
    return None
  if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
    return value.isoformat()
  if isinstance(value, float) and pd.isna(value):
    return None
  try:
    if pd.isna(value):
      return None
  except (TypeError, ValueError):
    pass
  if isinstance(value, (str, bytes)):
    return value
  if hasattr(value, 'item'):
    try:
      item = value.item()
    except (AttributeError, ValueError):
      return value
    return _sql_value(item) if isinstance(item, float) else item
  return value


class SqliteWarehouseError(RuntimeError):
  """Raised when the local warehouse cannot satisfy a request.

  Distinct from :class:`forecasting_ml_framework.exceptions.DatasetError` so the
  local harness's own failures are separable from the framework's.
  """
