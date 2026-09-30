#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""pandas-gbq dataset adapter for the production score table.

The score table is the framework's most important external contract, so its
adapter is the one that most needs to be explicit about its write semantics.
``if_exists: 'replace'`` is what makes the scoring half of the pipeline
idempotent: the same input produces the same scores, and re-running a day
replaces that day's table instead of appending to it. Combined with
``create_replace`` on the scoreset, the entire scoring half can be re-run safely
at any time.
"""

from __future__ import annotations

from typing import Any

from kedro.io.core import AbstractDataset

from forecasting_ml_framework.exceptions import DatasetError
from forecasting_ml_framework.observability.logging import get_logger

LOGGER = get_logger(__name__)

#: Documented write behaviours. ``replace`` is the framework default because the
#: score table is a per-run snapshot, not a time series.
SUPPORTED_IF_EXISTS = ('fail', 'replace', 'append')

DEFAULT_LOAD_ARGS: dict[str, Any] = {'project_id': None, 'location': None, 'credentials': None, 'dtype': None}
DEFAULT_SAVE_ARGS: dict[str, Any] = {'project_id': None, 'location': None, 'credentials': None, 'if_exists': 'replace'}


class GBQTableDataSet(AbstractDataset):
  """A pandas ``DataFrame`` bound to a BigQuery table via ``pandas-gbq``."""

  def __init__(
    self,
    table_name: str,
    dataset: str,
    project_id: str,
    credentials: dict[str, Any] | None = None,
    load_args: dict[str, Any] | None = None,
    save_args: dict[str, Any] | None = None,
  ) -> None:
    """Validate the configuration eagerly; perform no I/O.

    Args:
      table_name: The destination table.
      dataset: The destination dataset.
      project_id: The destination project.
      credentials: Optional service-account credentials.
      load_args: Overrides for ``pandas.read_gbq``.
      save_args: Overrides for ``DataFrame.to_gbq``.

    Raises:
      DatasetError: If mandatory configuration is absent or ``if_exists`` is
        unsupported.
    """
    for key, value in (('table_name', table_name), ('dataset', dataset), ('project_id', project_id)):
      if not value:
        raise DatasetError(f"Missing required configuration '{key}'")

    self._table_name = table_name
    self._dataset = dataset
    self._project_id = project_id
    self._credentials = credentials
    self._load_args = {**DEFAULT_LOAD_ARGS, **(load_args or {})}
    self._save_args = {**DEFAULT_SAVE_ARGS, **(save_args or {})}

    if_exists = self._save_args.get('if_exists', 'replace')
    if if_exists not in SUPPORTED_IF_EXISTS:
      raise DatasetError(
        f"Invalid 'if_exists' value: {if_exists!r}. Supported: {list(SUPPORTED_IF_EXISTS)}",
        if_exists=if_exists,
      )

  def _describe(self) -> dict[str, Any]:
    """Return every configuration key.

    Returns:
      A mapping suitable for the run log.
    """
    return {
      'table_name': self._table_name,
      'dataset': self._dataset,
      'project_id': self._project_id,
      'if_exists': self._save_args.get('if_exists'),
      'filepath': self.qualified_name,
    }

  @property
  def qualified_name(self) -> str:
    """Return the fully-qualified table name.

    Returns:
      ``project.dataset.table``.
    """
    return f'{self._project_id}.{self._dataset}.{self._table_name}'

  def _load(self) -> Any:
    """Read the score table.

    Returns:
      A pandas ``DataFrame``.
    """
    import pandas_gbq  # noqa: PLC0415

    LOGGER.info('Reading score table', extra={'table': self.qualified_name})
    return pandas_gbq.read_gbq(
      self.qualified_name,
      project_id=_pick(self._load_args.get('project_id'), self._project_id),
      location=self._load_args.get('location'),
      credentials=_pick(self._load_args.get('credentials'), self._credentials),
      dtype=self._load_args.get('dtype'),
    )

  def _save(self, data: Any) -> None:
    """Write the score table with the configured disposition.

    Args:
      data: The pandas ``DataFrame`` to write.
    """
    if data is None or getattr(data, 'empty', True):
      LOGGER.warning('Skipping save of an empty score table', extra={'table': self.qualified_name})
      return
    LOGGER.info(
      'Writing score table',
      extra={'table': self.qualified_name, 'if_exists': self._save_args.get('if_exists'), 'rows': len(data)},
    )
    data.to_gbq(
      self.qualified_name,
      project_id=_pick(self._save_args.get('project_id'), self._project_id),
      location=self._save_args.get('location'),
      credentials=_pick(self._save_args.get('credentials'), self._credentials),
      if_exists=self._save_args.get('if_exists') or 'replace',
    )


def _pick(configured: Any, declared: Any) -> Any:
  """Return the first value that was actually supplied.

  The defaults are merged into ``load_args`` / ``save_args`` at construction, so
  every key *exists* and ``args.get(key, declared)`` therefore never reaches its
  default -- the declared ``credentials`` and ``project_id`` were silently
  discarded, and the adapter fell back to ambient application-default
  credentials. That is a quiet change of identity: a catalog entry that declares
  a service account was read and written as whoever the process happened to be
  running as.

  A supplied value is therefore found by value, not by key presence. ``None`` is
  the sentinel for "not supplied", which is why the defaults themselves are
  ``None`` rather than empty strings or zero.

  Args:
    configured: The value from the merged argument mapping.
    declared: The value passed to the constructor.

  Returns:
    ``configured`` when it is not ``None``, otherwise ``declared``.
  """
  return declared if configured is None else configured
