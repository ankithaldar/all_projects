#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the local execution harness.

The local package exists so the framework's own node functions can be run against
SQLite and a local disk. Two of its modules are therefore a correctness surface
in their own right rather than scaffolding:

* the SQL translator decides what the production catalog's business rules *mean*
  when they run locally, and a mistranslation returns plausible numbers rather
  than an error;
* the warehouse's write dispositions decide whether re-running a day replaces its
  rows or duplicates them, and that distinction is invisible until someone counts
  rows weeks later.

So both are tested directly, and the tests are written adversarially -- the
question is not whether the translator round-trips its own examples, but whether
it refuses to guess at something it does not understand.

Spark and the dataset download are deliberately not exercised here. A Spark
session needs a Java runtime, and the pipeline driver needs the network; those
belong to the end-to-end run, which reports its own failures by name. A test
that silently skips is worse than no test, so this module tests what can be
tested everywhere and leaves the rest to the run that reports it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from forecasting_ml_framework.local.sql_translate import (
  SqlTranslationError,
  resolve_parameters,
  to_sqlite,
)
from forecasting_ml_framework.local.sqlite_warehouse import (
  WRITE_APPEND,
  WRITE_TRUNCATE,
  SQLiteWarehouse,
  SqliteWarehouseError,
)


@pytest.fixture
def warehouse(tmp_path: Path) -> SQLiteWarehouse:
  """Return a warehouse in a temporary directory.

  Args:
    tmp_path: pytest's temporary directory.

  Returns:
    An open warehouse, closed at the end of the test.
  """
  instance = SQLiteWarehouse(tmp_path / 'warehouse.db')
  yield instance
  instance.close()


def _frame(rows: int = 3, offset: float = 0.0) -> pd.DataFrame:
  """Build a small frame for a write test.

  Args:
    rows: The row count.
    offset: A value added to the numeric column, so two writes are
      distinguishable.

  Returns:
    A two-column frame.
  """
  return pd.DataFrame({
    'entity_id': [f'E{index:03d}' for index in range(rows)],
    'value': [float(index) + offset for index in range(rows)],
  })


class TestTranslation:
  """The dialect differences between the warehouse and SQLite."""

  def test_qualified_name_becomes_a_bare_identifier(self) -> None:
    """A project/dataset/table path loses its qualifiers.

    The local warehouse is a single file with a single dataset, so the
    qualifiers have nothing to disambiguate and keeping them would only make
    every reference unresolvable.
    """
    result = to_sqlite('SELECT * FROM `proj.ds.nba_features`')
    assert '"nba_features"' in result
    assert 'proj' not in result
    assert '.' not in result.replace('nba_features', '')

  def test_backticks_become_double_quotes(self) -> None:
    """Backticks are BigQuery's identifier delimiter; SQLite's is double quotes."""
    assert to_sqlite('SELECT `a` FROM `b`').count('"') == 4

  def test_date_add_becomes_a_modifier(self) -> None:
    """``DATE_ADD(x, INTERVAL n DAY)`` becomes SQLite's modifier form."""
    result = to_sqlite("SELECT DATE_ADD(DATE('2024-01-01'), INTERVAL 88 DAY)")
    assert "date(DATE('2024-01-01'), '+88 days')" in result
    assert 'INTERVAL' not in result

  @pytest.mark.parametrize(
    ('unit', 'expected'),
    [
      ('DAY', '+5 days'),
      ('HOUR', '+5 hours'),
      ('MINUTE', '+5 minutes'),
      ('SECOND', '+5 seconds'),
      ('MONTH', '+5 months'),
      ('YEAR', '+5 years'),
    ],
  )
  def test_every_supported_unit_translates(self, unit: str, expected: str) -> None:
    """Each interval unit must carry through, not be assumed to be days.

    A translator that always emitted ``days`` would produce a query that runs and
    returns the wrong date, which is the worst available outcome.

    Args:
      unit: The SQL interval unit.
      expected: The SQLite modifier suffix and count.
    """
    result = to_sqlite(f"SELECT DATE_ADD(DATE('x'), INTERVAL 5 {unit})")
    assert expected in result

  def test_week_translates_to_days(self) -> None:
    """SQLite has no week modifier, so weeks must be scaled to days."""
    assert "'+14 days'" in to_sqlite("SELECT DATE_ADD(DATE('x'), INTERVAL 2 WEEK)")

  def test_quarter_translates_to_months(self) -> None:
    """A quarter has no SQLite modifier either, so it scales to months."""
    assert "'+6 months'" in to_sqlite("SELECT DATE_ADD(DATE('x'), INTERVAL 2 QUARTER)")

  def test_singular_unit_is_accepted(self) -> None:
    """``INTERVAL 1 DAY`` and ``INTERVAL 1 DAYS`` are the same request."""
    assert "'+1 days'" in to_sqlite("SELECT DATE_ADD(DATE('x'), INTERVAL 1 DAY)")

  def test_nested_date_add_collapses(self) -> None:
    """A nested interval must resolve innermost-first, not mis-nest.

    A regex applied once would rewrite the outer call and leave the inner one
    behind as a bare expression, so the query would then reference an undefined
    name.
    """
    result = to_sqlite("SELECT DATE_ADD(DATE_ADD(DATE('x'), INTERVAL 1 MONTH), INTERVAL 58 DAY)")
    assert result.count('date(') == 2
    assert 'INTERVAL' not in result
    assert "'+58 days'" in result
    assert "'+1 months'" in result

  def test_current_date_translates(self) -> None:
    """Both spellings of the current date must become ``date('now')``."""
    assert "date('now')" in to_sqlite('SELECT CURRENT_DATE')
    assert "date('now')" in to_sqlite('SELECT CURRENT_DATE()')

  def test_window_and_case_survive(self) -> None:
    """The constructs the label query relies on must pass through unchanged."""
    sql = 'SELECT ROW_NUMBER() OVER (PARTITION BY a ORDER BY b DESC) AS rn, CASE WHEN c IS NULL THEN 0 ELSE 1 END'
    result = to_sqlite(sql)
    assert 'ROW_NUMBER() OVER (PARTITION BY a ORDER BY b DESC)' in result
    assert 'CASE WHEN c IS NULL THEN 0 ELSE 1 END' in result

  @pytest.mark.parametrize(
    'sql',
    [
      'SELECT ML.PREDICT(model, t) FROM t',
      'SELECT * FROM EXPORT DATA OPTIONS(uri="gs://x") AS y',
      'CREATE TEMP FUNCTION f(x INT64) AS (x + 1)',
    ],
  )
  def test_untranslatable_constructs_are_refused(self, sql: str) -> None:
    """A construct with no exact translation must raise, not be guessed at.

    A mistranslated query runs and returns numbers. A refusal fails at
    translation time and names what has to be rewritten, which is the only
    outcome that cannot be mistaken for a result.

    Args:
      sql: A query using an unsupported construct.
    """
    with pytest.raises(SqlTranslationError):
      to_sqlite(sql)

  def test_translation_is_idempotent(self) -> None:
    """Translating twice must equal translating once.

    The driver resolves and translates in separate steps, so a caller that
    translates an already-translated query must not corrupt it.
    """
    sql = "SELECT * FROM `p.d.t` WHERE d > DATE_ADD(DATE('x'), INTERVAL 3 DAY)"
    once = to_sqlite(sql)
    assert to_sqlite(once) == once


class TestParameterResolution:
  """Reference expansion, which happens before translation."""

  def test_globals_reference_expands(self) -> None:
    """A ``${globals:key}`` reference reads the globals document."""
    assert 'nba_features' in resolve_parameters('SELECT ${globals:table}', {}, {'table': 'nba_features'})

  def test_section_reference_expands(self) -> None:
    """A ``${section:key}`` reference reads that section of the parameters."""
    resolved = resolve_parameters('SELECT ${data_project:project}', {'data_project': {'project': 'p'}}, {})
    assert resolved == 'SELECT p'

  def test_numeric_reference_is_stringified(self) -> None:
    """A numeric value must reach the query as text, not as a Python repr."""
    assert resolve_parameters('SELECT ${globals:n}', {}, {'n': 88}) == 'SELECT 88'

  def test_unknown_key_names_itself(self) -> None:
    """A missing key must name the section and the key.

    A ``KeyError`` with no message leaves the reader guessing which of a dozen
    documents was wrong, and the query is the only place the reference appears.
    """
    with pytest.raises(KeyError) as caught:
      resolve_parameters('SELECT ${globals:absent}', {}, {'present': 1})
    message = str(caught.value)
    assert 'absent' in message
    assert 'globals' in message

  def test_expansion_then_translation_composes(self) -> None:
    """The two steps must compose, since the driver runs them in that order."""
    sql = "SELECT * FROM `${project:dataset}.<table>` WHERE d <= DATE_ADD(DATE('${run:date}'), INTERVAL 88 DAY)"
    resolved = resolve_parameters(sql, {'project': {'dataset': 'local.nba'}, 'run': {'date': '2024-01-01'}}, {})
    result = to_sqlite(resolved)
    assert '"<table>"' in result
    assert "'+88 days'" in result
    assert '2024-01-01' in result


class TestWriteDispositions:
  """The distinction between replacing a snapshot and extending a series."""

  def test_truncate_replaces(self, warehouse: SQLiteWarehouse) -> None:
    """A truncate write leaves exactly the rows just written."""
    warehouse.write(_frame(3), 'snapshot', disposition=WRITE_TRUNCATE)
    warehouse.write(_frame(2, offset=100), 'snapshot', disposition=WRITE_TRUNCATE)
    assert warehouse.count('snapshot') == 2

  def test_append_extends(self, warehouse: SQLiteWarehouse) -> None:
    """An append write accumulates, which is what a time series needs."""
    warehouse.write(_frame(3), 'series', disposition=WRITE_TRUNCATE)
    warehouse.write(_frame(2, offset=100), 'series', disposition=WRITE_APPEND)
    assert warehouse.count('series') == 5

  def test_append_to_absent_table_creates_it(self, warehouse: SQLiteWarehouse) -> None:
    """Appending to a table that does not exist must create it, not fail.

    A partitioned time series is written for the first time on its first run, so
    treating "no table yet" as an error would make the first period impossible.
    """
    warehouse.write(_frame(2), 'new_series', disposition=WRITE_APPEND)
    assert warehouse.count('new_series') == 2

  def test_truncate_then_append_is_idempotent_per_period(self, warehouse: SQLiteWarehouse) -> None:
    """Re-running a period must not duplicate it.

    The score metrics table is a snapshot, so a re-run replaces. A duplicate row
    for the same period is the classic symptom of getting this wrong, and it only
    shows up when someone aggregates.
    """
    for _ in range(3):
      warehouse.write(_frame(4), 'metrics', disposition=WRITE_TRUNCATE)
    assert warehouse.count('metrics') == 4

  def test_gbq_spelling_overrides_the_disposition(self, warehouse: SQLiteWarehouse) -> None:
    """``if_exists`` from the score-table adapter must win over the disposition.

    The catalog uses two vocabularies for the same three behaviours, and the
    adapter is declared with whichever its own upstream expects.
    """
    warehouse.write(_frame(2), 'scores', disposition=WRITE_TRUNCATE, if_exists='append')
    warehouse.write(_frame(2, offset=50), 'scores', disposition=WRITE_TRUNCATE, if_exists='append')
    assert warehouse.count('scores') == 4

  def test_unknown_disposition_is_refused(self, warehouse: SQLiteWarehouse) -> None:
    """An unrecognised disposition must raise rather than default.

    Defaulting to a truncate would silently discard a time series, and to an
    append would silently duplicate a snapshot.
    """
    with pytest.raises(SqliteWarehouseError):
      warehouse.write(_frame(2), 't', disposition='UPSERT_MAYBE')

  def test_append_with_an_unknown_column_is_refused(self, warehouse: SQLiteWarehouse) -> None:
    """An append whose shape differs must fail rather than partially write.

    A table whose columns depend on write history cannot be read back reliably,
    and the failure is otherwise discovered at the next aggregate.
    """
    warehouse.write(_frame(2), 'series', disposition=WRITE_TRUNCATE)
    with pytest.raises(SqliteWarehouseError) as caught:
      warehouse.write(
        pd.DataFrame({'entity_id': ['X'], 'different_column': [1.0]}),
        'series',
        disposition=WRITE_APPEND,
      )
    assert 'different_column' in str(caught.value)

  def test_empty_frame_is_refused(self, warehouse: SQLiteWarehouse) -> None:
    """Writing no rows must raise, not be treated as a no-op.

    An empty write usually means the upstream stage produced nothing, and
    silently succeeding turns that into an empty table that reads as a valid
    result for the day.
    """
    with pytest.raises(SqliteWarehouseError):
      warehouse.write(pd.DataFrame(columns=['entity_id', 'value']), 'empty', disposition=WRITE_TRUNCATE)


class TestRoundTrip:
  """Values must survive a write and a read unchanged."""

  @pytest.mark.parametrize(
    'value',
    [0, -17, 3.25, 1e-9, 'text', '', True, False],
  )
  def test_scalars_round_trip(self, warehouse: SQLiteWarehouse, value: object) -> None:
    """A written value reads back equal.

    Args:
      warehouse: The warehouse fixture.
      value: A value to write and read back.
    """
    warehouse.write(pd.DataFrame({'k': ['only'], 'v': [value]}), 'scalars', disposition=WRITE_TRUNCATE)
    assert warehouse.query('SELECT v FROM scalars').iloc[0]['v'] == value

  def test_null_survives_as_null(self, warehouse: SQLiteWarehouse) -> None:
    """A missing value must read back as null, not as the string ``'nan'``.

    The string form is the dangerous one: it reappears downstream as a real
    category, which is how a null becomes a legitimate value.
    """
    warehouse.write(
      pd.DataFrame({'k': ['a', 'b'], 'v': [1.0, float('nan')]}), 'nulls', disposition=WRITE_TRUNCATE
    )
    values = warehouse.query('SELECT v FROM nulls')['v'].tolist()
    assert values[0] == 1.0
    assert pd.isna(values[1])

  def test_timestamp_is_written_as_iso(self, warehouse: SQLiteWarehouse) -> None:
    """A timestamp must be stored in a form SQL can compare against a literal.

    An epoch integer is neither readable nor comparable to ``DATE('2024-01-01')``,
    and the metric tables are queried with date literals.
    """
    warehouse.write(
      pd.DataFrame({'k': ['a'], 'ts': [pd.Timestamp('2024-01-01T00:00:00Z')]}), 'stamped', disposition=WRITE_TRUNCATE
    )
    stored = warehouse.query("SELECT ts FROM stamped WHERE ts >= DATE('2023-12-31')")
    assert len(stored) == 1

  def test_query_preserves_declared_columns(self, warehouse: SQLiteWarehouse) -> None:
    """An empty result keeps its columns.

    A downstream column check should fail on a missing column, not on a missing
    frame -- otherwise "no rows" and "no such column" are indistinguishable.
    """
    warehouse.write(_frame(2), 't', disposition=WRITE_TRUNCATE)
    empty = warehouse.query('SELECT entity_id, value FROM t WHERE value > 1000')
    assert list(empty.columns) == ['entity_id', 'value']
    assert empty.empty

  def test_aggregates_compute(self, warehouse: SQLiteWarehouse) -> None:
    """A real aggregate must run, so the local warehouse is a database and not a dict."""
    warehouse.write(_frame(5), 't', disposition=WRITE_TRUNCATE)
    result = warehouse.query('SELECT COUNT(*) AS n, MAX(value) AS m FROM t')
    assert int(result.iloc[0]['n']) == 5
    assert float(result.iloc[0]['m']) == 4.0


class TestWarehouseHandle:
  """The handle's own surface."""

  def test_parent_directory_is_created(self, tmp_path: Path) -> None:
    """A path whose parent does not exist must still work.

    The local run points at a workspace that has just been created, and a
    warehouse that insists its directory pre-exists would need the caller to know
    an implementation detail of the filesystem layout.
    """
    instance = SQLiteWarehouse(tmp_path / 'deep' / 'nested' / 'w.db')
    try:
      assert instance.path.exists()
    finally:
      instance.close()

  def test_tables_are_listed_sorted(self, warehouse: SQLiteWarehouse) -> None:
    """Table listing must exclude SQLite's internal tables."""
    warehouse.write(_frame(1), 'b_table', disposition=WRITE_TRUNCATE)
    warehouse.write(_frame(1), 'a_table', disposition=WRITE_TRUNCATE)
    assert warehouse.table_names() == ['a_table', 'b_table']

  def test_absent_table_reports_zero(self, warehouse: SQLiteWarehouse) -> None:
    """Counting an absent table is zero, not an error."""
    assert warehouse.count('never_created') == 0

  def test_drop_is_idempotent(self, warehouse: SQLiteWarehouse) -> None:
    """Dropping a table that is not there must succeed."""
    warehouse.drop('nothing')
    warehouse.write(_frame(1), 't', disposition=WRITE_TRUNCATE)
    warehouse.drop('t')
    assert not warehouse.exists('t')

  def test_unwritable_path_raises_a_warehouse_error(self, tmp_path: Path) -> None:
    """A path that cannot be opened must raise the warehouse's own error type.

    A raw ``sqlite3.OperationalError`` would propagate a driver message to a
    caller that has no way to distinguish it from a query failure.
    """
    blocker = tmp_path / 'blocker'
    blocker.write_text('not a directory', encoding='utf-8')
    with pytest.raises(SqliteWarehouseError):
      SQLiteWarehouse(blocker / 'w.db')

  def test_execute_returns_rows_for_a_select(self, warehouse: SQLiteWarehouse) -> None:
    """The raw statement helper must work for callers composing their own SQL."""
    warehouse.execute('CREATE TABLE raw (x INTEGER)')
    warehouse.execute('INSERT INTO raw VALUES (7)')
    assert warehouse.execute('SELECT x FROM raw')[0]['x'] == 7

  def test_execute_tolerates_a_statement_returning_nothing(self, warehouse: SQLiteWarehouse) -> None:
    """A DDL statement returns no rows, and that is not an error."""
    assert warehouse.execute('CREATE TABLE t (x INTEGER)') == []

  def test_connection_is_exposed(self, warehouse: SQLiteWarehouse) -> None:
    """The live connection is available for callers needing their own SQL."""
    assert isinstance(warehouse.connection, sqlite3.Connection)
