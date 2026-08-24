#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data loader for SQLite databases.

Reads a single table, or the result of an arbitrary SQL query, from a local
SQLite database file into a pandas DataFrame.
"""


# imports
import sqlite3
import logging
from typing import Any, Dict
from pathlib import Path

import pandas as pd
#    script imports
from .base import DataLoader
# imports


# constants
# constants


# classes

class SQLiteDataLoader(DataLoader):
  """Data loader for SQLite databases.

  The source is selected through the ``sqlite`` section of the config. Either
  a ``table`` name or a full ``query`` must be provided; when both are present
  the query takes precedence, which allows filtering or column projection.
  """

  def __init__(self, config: Dict[str, Any]):
    """Initialise the loader from the application config.

    Args:
      config: Full application config, expected to hold a ``sqlite`` section
        with a ``file_path`` and either a ``table`` or a ``query`` entry.

    Raises:
      FileNotFoundError: If ``sqlite.file_path`` does not exist.
    """
    super().__init__(config)
    self.file_path = Path(config['sqlite']['file_path'])
    if not self.file_path.exists():
      raise FileNotFoundError(f'SQLite database not found: {self.file_path}')
    self.table = config['sqlite'].get('table')
    self.query = config['sqlite'].get('query')
    self.logger = logging.getLogger(self.logger_name)

  def load_data(self) -> pd.DataFrame:
    """Load the configured table or query into a DataFrame.

    Returns:
      pd.DataFrame: The rows selected from the database.

    Raises:
      ValueError: If neither ``sqlite.table`` nor ``sqlite.query`` is set.
      sqlite3.Error: If the table or query cannot be executed.
    """
    if not self.query and not self.table:
      raise ValueError(
        'Provide either "sqlite.table" or "sqlite.query" in the config.')

    if self.query:
      self.logger.info('Running custom query against %s...', self.file_path)
      query = self.query
    else:
      self.logger.info(
        'Loading table %r from %s...', self.table, self.file_path)
      query = f'SELECT * FROM "{self.table}"'

    with sqlite3.connect(self.file_path) as conn:
      try:
        df = pd.read_sql_query(query, conn)
      except Exception as e:
        self.logger.error(
          'Failed to load SQLite data from %s: %s', self.file_path, e)
        raise

    self.logger.info(
      'Loaded %d rows and %d columns.', len(df), len(df.columns))
    return df
# classes
