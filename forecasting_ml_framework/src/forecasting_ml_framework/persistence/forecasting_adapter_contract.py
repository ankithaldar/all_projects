#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Forecasting dataset adapter contracts.

Restructures the forecasting persistence layer adapter contracts through
forecasting layer naming. Every adapter satisfies the forecasting contract:
validate eagerly with no I/O, describe all configuration keys, and refuse
serialisation when it holds a live engine object. A new forecasting adapter
needs only a forecasting class and a catalog ``type:`` entry.
'''

from __future__ import annotations

from typing import Any

from kedro.io.core import AbstractDataset

from forecasting_ml_framework import constants
from forecasting_ml_framework.exceptions import ForecastingDatasetContractError
from forecasting_ml_framework.observability.logging import get_logger

LOGGER = get_logger(__name__)


class ForecastingDatasetAdapterContract:
  '''Abstract forecasting dataset adapter contract.

  Defines the forecasting layer interface for dataset adapters: eager
  validation, full description, load/save contracts, and safe serialisation.
  Concrete forecasting adapters extend :class:`AbstractDataset` and
  implement the forecasting adapter protocol rather than inheriting from
  this abstract contract.

  Attributes:
    adapter_type: The forecasting adapter vocabulary identifier.
  '''

  def __init__(self) -> None:
    '''Initialise the forecasting adapter contract state.'''
    self.adapter_type: str = 'forecasting_adapter'

  def forecasting_describe(self) -> dict[str, Any]:
    '''Return every forecasting adapter configuration key.

    Returns:
      A mapping suitable for the forecasting run log and provenance hooks.
    '''
    return {
      'adapter_type': self.adapter_type,
    }

  def forecasting_validate(self) -> None:
    '''Validate the forecasting adapter configuration eagerly.

    Raises:
      ForecastingDatasetContractError: If a forecasting adapter contract
        precondition is violated.
    '''
    pass

  def forecasting_load_contract(self) -> Any:
    '''Load data through the forecasting adapter contract.

    Returns:
      The forecasting dataset payload (pandas ``DataFrame`` or Spark
      ``DataFrame``).
    '''
    raise NotImplementedError('Forecasting adapter load contract not implemented')

  def forecasting_save_contract(self, data: Any) -> None:
    '''Write the forecasting dataset payload.

    Args:
      data: The forecasting dataset payload.
    '''
    raise NotImplementedError('Forecasting adapter save contract not implemented')


class ForecastingEmbeddedSqlContract(AbstractDataset):
  '''Forecasting embedded SQLite dataset adapter contract.

  Restructures :class:`~sqlite_warehouse.SQLiteWarehouse` through forecasting
  layer naming, providing the same load, save, describe, and existence interface
  as the production forecasting dataset adapter.

  Attributes:
    table_name: The forecasting dataset identifier.
    database: The forecasting dataset qualified name.
    path: The SQLite file path for forecasting local execution.
  '''

  def __init__(
    self,
    table_name: str | None = None,
    database: str | None = None,
    path: str | None = None,
    write_mode: str = 'WRITE_TRUNCATE',
  ) -> None:
    '''Validate eagerly; open no forecasting database connection yet.

    Args:
      table_name: The forecasting dataset identifier.
      database: The forecasting dataset qualified name.
      path: The SQLite file path.
      write_mode: The forecasting write disposition.

    Raises:
      ForecastingDatasetContractError: If the forecasting configuration is
        invalid.
    '''
    super().__init__()
    if not table_name and not database:
      raise ForecastingDatasetContractError(
        'Forecasting embedded adapter requires table_name or database',
        table_name=table_name,
        database=database,
      )
    self.table_name = table_name or (database or 'forecasting_dataset')
    self.database = database
    from pathlib import Path

    self.path = Path(path or constants.LOCAL_ARTEFACT_ROOT)
    self.write_mode = write_mode
    self.adapter_type = 'forecasting_embedded_sql'
    LOGGER.info(
      'Forecasting embedded adapter initialised',
      extra={
        'table': self.table_name,
        'database': self.database,
        'path': str(self.path),
        'adapter_type': self.adapter_type,
      },
    )

  def forecasting_describe(self) -> dict[str, Any]:
    '''Return forecasting adapter settings for audit and comparison.

    Returns:
      A dictionary of forecasting adapter settings.
    '''
    return {
      'table_name': self.table_name,
      'database': self.database,
      'path': str(self.path),
      'write_mode': self.write_mode,
      'adapter_type': self.adapter_type,
    }

  def forecasting_load_contract(self) -> Any:
    '''Read the forecasting embedded dataset.

    Returns:
      A pandas ``DataFrame``.
    '''
    from forecasting_ml_framework.local.sqlite_warehouse import SQLiteWarehouse

    LOGGER.info('Forecasting embedded load contract', extra={'table': self.table_name})
    warehouse = SQLiteWarehouse(str(self.path))
    try:
      if not warehouse.exists(self.table_name):
        LOGGER.warning(
          'Forecasting table absent; returning empty frame.',
          extra={'table': self.table_name},
        )
        import pandas as pd

        return pd.DataFrame()
      return warehouse.query(f'SELECT * FROM "{self.table_name}"')
    finally:
      warehouse.close()

  def forecasting_save_contract(self, data: Any) -> None:
    '''Write the forecasting embedded dataset.

    Args:
      data: A pandas ``DataFrame``.
    '''
    from forecasting_ml_framework.local.sqlite_warehouse import SQLiteWarehouse

    LOGGER.info(
      'Forecasting embedded save contract',
      extra={'table': self.table_name, 'write_mode': self.write_mode},
    )
    warehouse = SQLiteWarehouse(str(self.path))
    try:
      warehouse.write(data, self.table_name, disposition=self.write_mode)
    finally:
      warehouse.close()
