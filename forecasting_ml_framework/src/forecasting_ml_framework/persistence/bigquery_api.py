#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BigQuery REST dataset adapters.

Two adapters share this module:

``BQTableDataSet``
  A pandas frame bound to a table, written through the load-job API. It is the
  carrier for all nine metric and lift tables.

``BQQueryDataSet``
  A parameterised query with ``{param}`` templating, resolved from a file or from
  inline SQL. It is a supported catalog ``type:`` and the natural choice when a
  dataset's shape is defined by a query rather than by a table schema.

The partition-decorator mechanism deserves a note because it is subtle and is the
framework's only interaction with BigQuery decorators. Writing a load job to
``table$20260929`` targets exactly one partition in a single atomic load, with no
delete-then-insert window and no possibility of a partial write being observed.
That is materially safer than the pattern the enterprise egress SQL uses.
"""

from __future__ import annotations

import re
from typing import Any

from kedro.io.core import AbstractDataset

from forecasting_ml_framework.exceptions import DatasetError, PartitionValueMissingError
from forecasting_ml_framework.observability.logging import get_logger

LOGGER = get_logger(__name__)

#: Default load disposition. Metric tables are small single-partition snapshots,
#: so truncate-and-write is the correct semantic.
DEFAULT_WRITE_DISPOSITION = 'WRITE_TRUNCATE'


class BQTableDataSet(AbstractDataset):
  """A pandas ``DataFrame`` bound to a BigQuery table.

  Attributes:
    _project_id: The project that owns the table.
    _dataset: The dataset that owns the table.
    _table_name: The table name.
    _partition_field: An optional partition column.
    _partition_value: The partition value used for both reads and writes.
  """

  def __init__(
    self,
    table_name: str,
    dataset: str,
    project_id: str,
    credentials: dict[str, Any] | None = None,
    write_disposition: str = DEFAULT_WRITE_DISPOSITION,
    partition_field: str | None = None,
    partition_value: str | None = None,
    partition_type: str = 'DAY',
    clustering_fields: list[str] | None = None,
    description: str | None = None,
  ) -> None:
    """Validate the configuration eagerly; perform no I/O.

    Args:
      table_name: The destination table.
      dataset: The destination dataset.
      project_id: The destination project.
      credentials: Optional service-account credentials.
      write_disposition: A BigQuery load-job write disposition.
      partition_field: The partition column, or ``None`` for an unpartitioned
        table.
      partition_value: The partition value, required for partitioned use.
      partition_type: The partition granularity, e.g. ``'DAY'``.
      clustering_fields: Columns to cluster on. ``['model_key']`` is the correct
        access pattern for score lookups and metric trends.
      description: An optional table description.

    Raises:
      DatasetError: If mandatory configuration is absent.
    """
    for key, value in (('table_name', table_name), ('dataset', dataset), ('project_id', project_id)):
      if not value:
        raise DatasetError(f"Missing required configuration '{key}'")
    self._table_name = table_name
    self._dataset = dataset
    self._project_id = project_id
    self._credentials = credentials
    self._write_disposition = write_disposition
    self._partition_field = partition_field
    self._partition_value = partition_value
    self._partition_type = partition_type
    self._clustering_fields = list(clustering_fields or [])
    self._description = description

  def _describe(self) -> dict[str, Any]:
    """Return every configuration key.

    Returns:
      A mapping suitable for the run log.
    """
    return {
      'table_name': self._table_name,
      'dataset': self._dataset,
      'project_id': self._project_id,
      'write_disposition': self._write_disposition,
      'partition_field': self._partition_field,
      'partition_value': self._partition_value,
      'partition_type': self._partition_type,
      'clustering_fields': self._clustering_fields,
    }

  def _load(self) -> Any:
    """Read the table, optionally filtered to a single partition.

    Returns:
      A pandas ``DataFrame``.

    Raises:
      PartitionValueMissingError: If a partition field is configured without a
        partition value.
    """
    from google.cloud import bigquery  # noqa: PLC0415

    client = bigquery.Client(project=self._project_id, credentials=self._credentials)
    condition = ''
    if self._partition_field is not None:
      if self._partition_value is None:
        raise PartitionValueMissingError(
          f'Partition field {self._partition_field!r} is configured but no partition value was '
          'supplied; reading a partitioned table requires an explicit value',
          table_name=self._table_name,
        )
      condition = f' WHERE {self._partition_field} = "{self._partition_value}"'
    sql = f'SELECT * FROM {self._qualified_name()}{condition}'
    LOGGER.info('Reading metric table', extra={'table': self._qualified_name(), 'partition': self._partition_value})
    return client.query(sql).to_dataframe()

  def _save(self, data: Any) -> None:
    """Write the frame through the load-job API.

    An empty frame is a no-op, which is what allows an empty evaluation slice to
    complete rather than fail.

    Args:
      data: The pandas ``DataFrame`` to write.
    """
    if data is None or getattr(data, 'empty', True):
      LOGGER.warning('Skipping save of an empty frame', extra={'table': self._table_name})
      return

    from google.cloud import bigquery  # noqa: PLC0415

    client = bigquery.Client(project=self._project_id, credentials=self._credentials)
    job_config = bigquery.LoadJobConfig(
      write_disposition=self._write_disposition,
      time_partitioning=(
        bigquery.table.TimePartitioning(type_=self._partition_type, field=self._partition_field)
        if self._partition_field
        else None
      ),
      clustering_fields=self._clustering_fields or None,
    )
    decorator = self.partition_decorator()
    destination = f'{self._qualified_name()}{decorator}'
    LOGGER.info(
      'Writing metric table',
      extra={'table': self._qualified_name(), 'decorator': decorator, 'disposition': self._write_disposition},
    )
    client.load_table_from_dataframe(dataframe=data, destination=destination, job_config=job_config).result()

  def partition_decorator(self) -> str:
    """Return the BigQuery partition decorator for the configured value.

    The decorator encodes the partition value in compact ``YYYYMMDD`` form, so
    an ISO ``run_date`` of ``2026-09-29`` becomes ``$20260929``.

    Returns:
      The decorator, or an empty string for an unpartitioned table.

    Raises:
      PartitionValueMissingError: If a partition field is configured without a
        partition value.
    """
    if self._partition_field is None:
      return ''
    if self._partition_value is None:
      raise PartitionValueMissingError(
        f'Partition field {self._partition_field!r} is configured but no partition value was supplied',
        table_name=self._table_name,
      )
    return '$' + self._partition_value.replace('-', '')

  def _qualified_name(self) -> str:
    """Return the fully-qualified table name.

    Returns:
      ``project.dataset.table``.
    """
    return f'{self._project_id}.{self._dataset}.{self._table_name}'


class BQQueryDataSet(AbstractDataset):
  """A parameterised BigQuery query with ``{param}`` templating.

  The template is resolved from a file on ``fsspec`` or from inline SQL, then
  rendered with the supplied parameters. This is the adapter to use when a
  dataset's shape is defined by a query rather than by a table schema.
  """

  _placeholder_pattern = re.compile(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}')

  def __init__(
    self,
    sql: str | None = None,
    filepath: str | None = None,
    dataset: str | None = None,
    project_id: str | None = None,
    credentials: dict[str, Any] | None = None,
    load_args: dict[str, Any] | None = None,
  ) -> None:
    """Validate the configuration eagerly; perform no I/O.

    Args:
      sql: An inline query template.
      filepath: A ``fsspec`` path to a file containing the template.
      dataset: The dataset used as the query's default.
      project_id: The project used as the query's default.
      credentials: Optional service-account credentials.
      load_args: Extra query-job configuration, e.g.
        ``{'dry_run': True, 'maximum_bytes_billed': 10**9}``.

    Raises:
      DatasetError: If neither a template nor a file path is supplied.
    """
    if not sql and not filepath:
      raise DatasetError("Provide either 'sql' or 'filepath'")
    self._sql = sql
    self._filepath = filepath
    self._dataset = dataset
    self._project_id = project_id
    self._credentials = credentials
    self._load_args = dict(load_args or {})

  def _describe(self) -> dict[str, Any]:
    """Return every configuration key.

    Returns:
      A mapping suitable for the run log. ``filepath`` is present even when
      unused, because the provenance hooks read it unconditionally.
    """
    return {
      'sql': '<inline sql>' if self._sql else None,
      'filepath': self._filepath,
      'dataset': self._dataset,
      'project_id': self._project_id,
    }

  def _load(self) -> Any:
    """Render the template, execute it, and return the result.

    Returns:
      A pandas ``DataFrame``.
    """
    from google.cloud import bigquery  # noqa: PLC0415

    template = self._read_template()
    rendered = template.format(**self._load_args)
    client = bigquery.Client(project=self._project_id, credentials=self._credentials)
    LOGGER.info('Executing parameterised BigQuery query', extra={'sql': rendered[:400]})
    return client.query(rendered).to_dataframe()

  def _save(self, data: Any) -> None:
    """Refuse writes.

    Args:
      data: Ignored.

    Raises:
      DatasetError: Always. A query-backed dataset is read-only.
    """
    raise DatasetError('BQQueryDataSet is read-only; a query does not define a writable schema')

  def _read_template(self) -> str:
    """Return the query template from its source.

    Returns:
      The template text.

    Raises:
      DatasetError: If the configured file cannot be read.
    """
    if self._sql:
      return self._sql
    from forecasting_ml_framework.utils.storage import get_file_system  # noqa: PLC0415

    filesystem = get_file_system(self._filepath, self._project_id)
    try:
      with filesystem.open(self._filepath, 'r') as handle:
        return handle.read()
    except OSError as error:
      raise DatasetError('Unable to read the query template', filepath=self._filepath, error=str(error)) from error
