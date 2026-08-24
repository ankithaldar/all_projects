#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data loader for Parquet files."""


# imports
import pandas as pd
import logging
from typing import Dict, Any
from pathlib import Path
#    script imports
from .base import DataLoader
# imports


# constants
# constants


# classes
class ParquetDataLoader(DataLoader):
  """Data loader for Parquet files."""

  def __init__(self, config: Dict[str, Any]):
    super().__init__(config)
    self.file_path = Path(config['parquet']['file_path'])
    if not self.file_path.exists():
      raise FileNotFoundError(f"Parquet file not found: {self.file_path}")
    self.logger = logging.getLogger(self.logger_name)

  def load_data(self) -> pd.DataFrame:
    self.logger.info('Loading Parquet data from %s...', self.file_path)
    df = pd.read_parquet(self.file_path)
    self.logger.info(
      'Loaded %d rows and %d columns.', len(df), len(df.columns))
    return df
# classes
