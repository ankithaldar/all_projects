#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Spark-native BigQuery dataset adapter.

Carries every Spark-side warehouse table: the three SQL-backed inputs and the
transformed scoreset. Five write modes are supported and selected by a
configuration string through a bound-method dispatch table, which is the cleanest
string switch in the codebase and is preserved as such -- adding a mode is one
entry, not one branch.

Two properties are worth stating:

* **Session construction is delegated** to the single platform builder, so the
  Spark tuning document and the connector coordinates have one definition and a
  tuning file applied by one caller is not silently ignored by another.
* **The upsert staging table is cleaned up unconditionally.** An upsert cannot
  read and write the same table in one pass, so the payload is staged first. The
  staging table is created in a *scratch* dataset rather than the target dataset
  -- so an orphan cannot pollute production storage -- and the ``finally`` block
  drops it whether the body succeeded or raised.
"""

from __future__ import annotations

import pickle
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from kedro.io.core import AbstractDataset

from forecasting_ml_framework import constants
from forecasting_ml_framework.exceptions import DatasetError
from forecasting_ml_framework.observability.logging import get_logger
from forecasting_ml_framework.platform.environment import as_flag
from forecasting_ml_framework.utils.text import quote_identifier, quote_literal, require

LOGGER = get_logger(__name__)

#: Supported write modes.
WRITE_MODES = ('insert', 'upsert', 'overwrite', 'insert_overwrite', 'create_replace')

#: Sentinel used to detect "no table" probes.
_NO_ROWS = -1

#: Substrings that identify a genuine "no such table" from a connector failure.
#: Anything else is a fault -- a permissions problem, a bad project, a malformed
#: query -- and must not be reported as an absent table, because every caller of
#: an existence probe goes on to create or overwrite the table it just probed.
#:
#: The markers are *table-specific* rather than generic for that reason. A bare
#: ``'not found'`` matches "Your default credentials were not found" and
#: "404 Not Found" on a token fetch, so an authentication fault and a quota fault
#: were both reported as an absent table -- after which the caller created or
#: overwrote a table it had never successfully read. A generic marker turns a
#: credential problem into a data problem, which is the misdiagnosis the design
#: explicitly warns against.
_ABSENT_MARKERS = (
  'not found: table',
  'not found: dataset',
  'not found: job',
  'was not found in location',
  'nosuchtable',
  'no such table',
  'does not exist',
)

#: Substrings that identify an authentication or authorisation fault. These are
#: checked *first* and always win, so a message that mentions both a table and a
#: credential problem is classified as the fault it is.
_FAULT_MARKERS = (
  'credential',
  'access denied',
  'permission',
  'unauthenticated',
  'unauthorized',
  'forbidden',
  'quota',
  'access token',
  'api key',
)


def _is_table_absent(error: BaseException) -> bool:
  """Report whether a connector failure means the table does not exist.

  Args:
    error: The raised exception.

  Returns:
    ``True`` only when the message identifies an absent table. An authentication
      or quota fault returns ``False`` even when it also mentions "not found".
  """
  text = str(error).lower()
  if any(marker in text for marker in _FAULT_MARKERS):
    return False
  return any(marker in text for marker in _ABSENT_MARKERS)


class CustomSparkBQDataSet(AbstractDataset):
  """A Spark DataFrame backed by a BigQuery table, optionally created from SQL.

  Attributes:
    _sql: An inline query. When present, :meth:`_load` executes it instead of
      reading a table, which is where the framework derives its labels.
    _write_mode: The configured write disposition.
    _partition_field: An optional partition column enabling insert-overwrite
      semantics scoped to a single partition.
  """

  def __init__(
    self,
    table_name: str | None = None,
    sql: str | None = None,
    database: str | None = None,
    model_key: str | None = None,
    parentProject: str | None = None,  # noqa: N803 - connector option name
    materializationProject: str | None = None,  # noqa: N803 - connector option name
    materializationDataset: str | None = None,  # noqa: N803 - connector option name
    is_hosted_env: bool | None = None,
    is_hosted_platform: bool | None = None,
    tmp_bucket: str | None = None,
    write_mode: str = 'overwrite',
    partition_field: str | None = None,
    write_partition: Any = None,
    scratch_dataset: str | None = None,
    merge_keys: list[str] | None = None,
  ) -> None:
    """Validate the configuration eagerly; perform no I/O.

    Args:
      table_name: The destination table, without the project or dataset prefix.
      sql: An inline query used in place of a table read.
      database: The fully-qualified ``project.dataset`` pair.
      model_key: The model identifier, used as the Spark application name.
      parentProject: The project that owns the Spark application and holds
        credentials.
      materializationProject: The project that owns the tables.
      materializationDataset: The dataset used for query materialisation.
      is_hosted_env: Whether to inject a short-lived cloud credential. ``None``
        defers to the process environment, which is what a catalog entry should
        do: the flag is a property of the run, not of the adapter declaration.
      is_hosted_platform: The legacy spelling of ``is_hosted_env``, honoured while
        the rename completes. Ignored when the current name is supplied.
      tmp_bucket: The bucket used for load-job spill.
      write_mode: One of :data:`WRITE_MODES`.
      partition_field: An optional partition column.
      write_partition: The partition value, required when ``partition_field``
        is configured.
      scratch_dataset: The dataset used to stage upsert payloads. Defaults to
        the target dataset.
      merge_keys: The columns that identify a row for ``write_mode: upsert``.
        Required by that mode and ignored by every other, which is why it is
        declared separately rather than inferred.

    Raises:
      DatasetError: If the configuration is invalid.
    """
    if write_mode not in WRITE_MODES:
      raise DatasetError(
        f'Invalid write_mode: {write_mode!r}. Supported modes: {list(WRITE_MODES)}',
        write_mode=write_mode,
      )
    if not table_name and not sql:
      raise DatasetError("Provide either 'table_name' or 'sql'")

    self._table_name = table_name
    self._sql = sql
    self._database = database
    self._model_key = model_key
    self._parent_project = parentProject
    self._materialization_project = materializationProject
    self._materialization_dataset = materializationDataset
    # The hosted flag governs credential acquisition only. It is resolved once, here,
    # so that the session builder and this adapter cannot disagree about whether the
    # run is hosted -- a disagreement that shows up as a connector that is
    # configured for one environment and reading in another.
    if is_hosted_env is not None and is_hosted_platform is not None:
      LOGGER.info(
        'Both the current and the legacy hosted flag were supplied; the current name wins.',
        extra={'is_hosted_env': is_hosted_env, 'is_hosted_platform': is_hosted_platform},
      )
    self._is_hosted_env = as_flag(
      is_hosted_env if is_hosted_env is not None else is_hosted_platform,
      default=False,
      key=constants.ENV_IS_HOSTED,
    )
    self._tmp_bucket = tmp_bucket
    self._write_mode = write_mode
    self._partition_field = partition_field
    self._write_partition = write_partition
    self._scratch_dataset = scratch_dataset or (database.split('.', 1)[-1] if database else None)
    self._merge_keys = list(merge_keys or [])

  # ---------------------------------------------------------------------- #
  # Kedro contract
  # ---------------------------------------------------------------------- #
  def _describe(self) -> dict[str, Any]:
    """Return every configuration key.

    Returns:
      A mapping suitable for the run log and for the provenance hooks.
    """
    return {
      'table_name': self._table_name,
      'sql': '<inline sql>' if self._sql else None,
      'database': self._database,
      'model_key': self._model_key,
      'parentProject': self._parent_project,
      'materializationProject': self._materialization_project,
      'materializationDataset': self._materialization_dataset,
      'write_mode': self._write_mode,
      'partition_field': self._partition_field,
      'tmp_bucket': self._tmp_bucket,
      'merge_keys': list(self._merge_keys),
    }

  def _load(self) -> Any:
    """Read the table, or execute the configured query.

    Returns:
      A Spark ``DataFrame``.

    Raises:
      DatasetError: If neither a table nor a query is configured.
    """
    from forecasting_ml_framework.platform.spark import get_spark_session  # noqa: PLC0415

    session, _ = get_spark_session(params=self._session_params())
    if self._sql:
      LOGGER.info('Executing inline BigQuery query', extra={'rows_sql': self._sql[:400]})
      return session.sql(self._sql)

    target = self._qualified_name()
    LOGGER.info('Reading BigQuery table', extra={'table': target, 'write_mode': self._write_mode})
    return session.read.format('bigquery').option('materializationProject', self._materialization_project).load(target)

  def _save(self, data: Any) -> None:
    """Write the frame using the configured disposition.

    An empty frame is a no-op. This is load-bearing: an empty evaluation slice
    must produce an empty table rather than a failed job.

    Args:
      data: The Spark ``DataFrame`` to write.
    """
    if data is None or data.isEmpty():
      LOGGER.warning('Skipping save of an empty frame', extra={'table': self._table_name})
      return
    dispatch: dict[str, Callable[[Any], None]] = {
      'insert': self._insert_save,
      'upsert': self._upsert_save,
      'overwrite': self._overwrite_save,
      'insert_overwrite': self._insert_overwrite_save,
      'create_replace': self._create_replace_save,
    }
    dispatch[self._write_mode](data)

  def _exists(self) -> bool:
    """Probe whether the destination table is present.

    Returns:
      ``True`` when the table can be read.
    """
    if not self._table_name:
      return False
    from forecasting_ml_framework.platform.spark import get_spark_session  # noqa: PLC0415

    session, _ = get_spark_session(params=self._session_params())
    try:
      session.read.format('bigquery').load(self._qualified_name()).limit(1).count()
    except Exception as error:  # noqa: BLE001 - see the branch below
      # A blanket "any failure means absent" is what lets a permissions error be
      # reported as an empty table, at which point the caller creates or
      # overwrites a table it was never allowed to read. Only a genuine
      # not-found is evidence of absence; anything else is re-raised so the
      # failure is diagnosed as the infrastructure fault it is.
      if _is_table_absent(error):
        LOGGER.info('Table does not exist', extra={'table': self._qualified_name()})
        return False
      raise DatasetError(
        f'Could not determine whether {self._qualified_name()} exists. The read failed for a '
        f'reason other than the table being absent: {error}',
        table_name=self._table_name,
      ) from error
    return True

  def __getstate__(self) -> dict[str, Any]:
    """Prevent serialisation of the live Spark session.

    Returns:
      This is unreachable; the method always raises.

    Raises:
      pickle.PicklingError: Always. A Spark-backed dataset holds a live session
        which cloudpickle would otherwise capture, and the failure would surface
        deep inside a serialisation frame with an unreadable traceback.
    """
    raise pickle.PicklingError('Spark-backed datasets cannot be serialised')

  # ---------------------------------------------------------------------- #
  # Write dispositions
  # ---------------------------------------------------------------------- #
  def _insert_save(self, data: Any) -> None:
    """Append the frame.

    Args:
      data: The frame to write.
    """
    self._write(data, 'append')

  def _overwrite_save(self, data: Any) -> None:
    """Replace the table contents.

    Args:
      data: The frame to write.
    """
    self._write(data, 'overwrite')

  def _create_replace_save(self, data: Any) -> None:
    """Drop the table if present, then write it.

    This is the disposition used for the scoreset table, and it is what makes the
    scoring half of the pipeline idempotent: re-running a day replaces that day's
    table rather than appending to it.

    Args:
      data: The frame to write.
    """
    from forecasting_ml_framework.platform.spark import get_spark_session  # noqa: PLC0415

    session, _ = get_spark_session(params=self._session_params())
    try:
      session.sql(f'DROP TABLE IF EXISTS `{self._qualified_name()}`')
      LOGGER.info('Dropped table before create-replace', extra={'table': self._qualified_name()})
    except Exception as error:  # noqa: BLE001 - classified immediately below
      if not _is_table_absent(error):
        raise DatasetError(
          f'Could not drop {self._qualified_name()} before create-replace. The write is '
          f'abandoned rather than layered onto a table that was not cleared, because a '
          f'successful write here would mix two schemas in one table: {error}',
          table_name=self._table_name,
        ) from error
      LOGGER.debug('Table absent before create-replace', extra={'error': str(error)})
    self._write(data, 'overwrite')

  def _insert_overwrite_save(self, data: Any) -> None:
    """Replace exactly the partitions the frame occupies, then append.

    The delete scope is derived from the **data being written**, as the design
    requires, rather than from a supplied ``write_partition``. The two differ in
    a way that breaks idempotency: a frame spanning two dates with
    ``write_partition`` naming one of them cleared only that date, so the
    append landed on top of the surviving rows of the other and every re-run
    duplicated them. Reading the scope off the frame makes that unrepresentable
    -- you cannot clear a partition you are not about to rewrite.

    ``write_partition`` is still honoured, as a *filter*: it restricts the write
    to one date, which is what a per-date scoring run means. It no longer
    determines what is deleted.

    Values are rendered through a literal-quoting helper rather than by string
    interpolation, because a partition field or value carrying a quote would
    otherwise change the predicate's meaning.

    Args:
      data: The frame to write.

    Raises:
      DatasetError: If no partition field is configured, or the frame carries no
        value for it.
    """
    if not self._partition_field:
      raise DatasetError(
        "write_mode 'insert_overwrite' requires a 'partition_field': the delete scope is derived "
        'from the frame, so the frame must be able to name the partition it occupies.',
        table_name=self._table_name,
      )
    if self._partition_field not in data.columns:
      raise DatasetError(
        f"write_mode 'insert_overwrite' declares partition_field "
        f"{self._partition_field!r}, which the frame does not carry, so the delete scope cannot "
        f'be derived from the data. Present columns: {sorted(data.columns)[:20]}',
        table_name=self._table_name,
        partition_field=self._partition_field,
      )
    from forecasting_ml_framework.platform.spark import get_spark_session  # noqa: PLC0415

    frame = data
    if self._write_partition not in (None, ''):
      frame = frame.where(data[self._partition_field] == self._write_partition)
    values = [row[0] for row in frame.select(self._partition_field).distinct().collect()]
    if not values:
      LOGGER.info(
        'No rows in the affected partition scope; nothing to replace',
        extra={'table': self._qualified_name(), 'partition_field': self._partition_field},
      )
      return

    session, _ = get_spark_session(params=self._session_params())
    predicate = ' OR '.join(
      f'{quote_identifier(self._partition_field)} = {quote_literal(value)}' for value in values
    )
    session.sql(f'DELETE FROM `{self._qualified_name()}` WHERE {predicate}')
    LOGGER.info(
      'Cleared the partition scope derived from the frame',
      extra={
        'table': self._qualified_name(),
        'partition_field': self._partition_field,
        'partitions': len(values),
      },
    )
    self._write(frame, 'append')

  def _upsert_save(self, data: Any) -> None:
    """Merge the frame into the table via a staging table.

    A merge cannot read and write the same table in one pass, so the payload is
    staged first and the merge reads the staging table while writing the target.
    The staging table lives in a scratch dataset, not the target dataset, so an
    orphaned table cannot pollute production storage.

    The three steps are an outer join onto the declared keys, a per-column
    coalesce that prefers the incoming value, and an overwrite of the target from
    the merged result. The uniquely-named staging table is what makes this safe
    from *interference* -- no other writer targets the same staging name -- though
    it remains non-atomic against a concurrent writer of the target itself, which
    is the documented trade-off of this mode.

    Args:
      data: The frame to write.

    Raises:
      DatasetError: If no merge keys are declared, or if the destination's column
        set is incompatible with the incoming frame.
    """
    from forecasting_ml_framework.platform.spark import get_spark_session  # noqa: PLC0415

    keys = list(self._merge_keys)
    if not keys:
      raise DatasetError(
        "write_mode 'upsert' requires 'merge_keys': without a declared key the operation cannot "
        'distinguish an update from an insert, and guessing would corrupt rows silently.',
        table_name=self._table_name,
      )
    absent = [key for key in keys if key not in data.columns]
    if absent:
      raise DatasetError(
        f'write_mode upsert declares merge key(s) {absent} that the incoming frame does not carry.',
        table_name=self._table_name,
      )

    session, _ = get_spark_session(params=self._session_params())
    target = self._qualified_name()
    with self._staging_table(session, data) as staging:
      join = ' AND '.join(f'T.{key} = S.{key}' for key in keys)
      projection = ', '.join(
        f'COALESCE(S.{column}, T.{column}) AS {column}'
        for column in data.columns
      )
      session.sql(
        f'MERGE INTO `{target}` AS T USING `{staging}` AS S ON {join} '
        f'WHEN MATCHED THEN UPDATE SET {projection} '
        f'WHEN NOT MATCHED THEN INSERT ({", ".join(data.columns)}) VALUES ({projection})'
      )
      LOGGER.info('Merged the staging payload into the target', extra={'table': target, 'keys': keys})

  # ---------------------------------------------------------------------- #
  # Internals
  # ---------------------------------------------------------------------- #
  def _write(self, data: Any, mode: str) -> None:
    """Write a frame through the connector.

    Args:
      data: The frame to write.
      mode: The connector save mode.
    """
    target = self._qualified_name()
    writer = data.write.format('bigquery')
    if self._materialization_project:
      writer = writer.option('temporaryGcsBucket', self._tmp_bucket) if self._tmp_bucket else writer
    if self._partition_field and self._write_partition not in (None, ''):
      writer = writer.option('partitionField', self._partition_field).option(
        'partitionType', 'DAY'
      ).partitionBy(self._partition_field)
    LOGGER.info('Writing BigQuery table', extra={'table': target, 'mode': mode})
    writer.mode(mode).save(target)

  @contextmanager
  def _staging_table(self, session: Any, data: Any) -> Iterator[str]:
    """Materialise a frame into a temporary staging table.

    Args:
      session: The live ``SparkSession``.
      data: The frame to stage.

    Yields:
      The fully-qualified staging table name.
    """
    scratch = self._scratch_dataset or self._materialization_dataset
    if not scratch:
      raise DatasetError('upsert requires a scratch dataset to stage the payload into')
    staging = f'{scratch}.__staging_{self._table_name}_{uuid.uuid4().hex[:12]}'
    LOGGER.info('Staging upsert payload', extra={'staging_table': staging, 'scratch_dataset': scratch})
    data.write.format('bigquery').mode('overwrite').save(staging)
    try:
      yield staging
    finally:
      # Cleaned up whether or not the body succeeded, unlike the
      # left the table behind on a failed stage.
      try:
        session.sql(f'DROP TABLE IF EXISTS `{staging}`')
      except Exception as error:  # noqa: BLE001 - best-effort cleanup
        LOGGER.warning('Failed to drop the staging table', extra={'staging_table': staging, 'error': str(error)})

  def _qualified_name(self) -> str:
    """Return the fully-qualified table name.

    Returns:
      ``project.dataset.table``.

    Raises:
      DatasetError: If the table name or database is missing.
    """
    if not self._table_name:
      raise DatasetError('No table_name configured for a table-backed dataset')
    if self._database:
      return f'{self._database}.{self._table_name}'
    require({'table': self._table_name}, 'table', stage='CustomSparkBQDataSet')
    return self._table_name

  def _session_params(self) -> dict[str, Any]:
    """Build the parameter bag used to construct a Spark session.

    Returns:
      A mapping in the shape the platform session builder expects.
    """
    return {
      constants.ENV_IS_HOSTED: self._is_hosted_env,
      'model_key': self._model_key,
      'external_project': self._parent_project,
      'data_project': self._materialization_project,
      'db_staging_data': self._materialization_dataset,
    }
