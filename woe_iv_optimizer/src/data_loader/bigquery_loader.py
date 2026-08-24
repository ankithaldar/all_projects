#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Load Data from BigQuery Table"""


# imports
from google.cloud import bigquery
import pandas as pd
import logging
from typing import Dict, List
import os
#    script imports
from .base import DataLoader
# imports


# constants
# constants


# classes
class BigQueryDataLoader(DataLoader):
  """Load Data from BigQuery Table"""

  def __init__(self, config: Dict[str, Any]):
    super().__init__(config)
    self.client = bigquery.Client(project=config['bigquery']['external_project'])
    self.logger = logging.getLogger(self.logger_name)

  def load_data(self) -> pd.DataFrame:
    query = f"""
    SELECT * FROM `{self.config['bigquery']['project_id']}.{self.config['bigquery']['dataset_id']}.{self.config['bigquery']['table_id']}`
    """
    self.logger.info("Loading data from BigQuery...")
    df = self.client.query(query).to_dataframe()
    self.logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
    return df
# cslasses
