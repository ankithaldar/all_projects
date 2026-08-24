#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Load Data from BigQuery Table"""


# imports
from google.cloud import bigquery
import pandas as pd
import logging
from typing import Any, Dict
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
    settings = config['bigquery']
    self.client = bigquery.Client(project=settings['external_project'])
    self.logger = logging.getLogger(self.logger_name)

  def load_data(self) -> pd.DataFrame:
    settings = self.config['bigquery']
    table = '.'.join([settings['project_id'], settings['dataset_id'],
                      settings['table_id']])
    self.logger.info('Loading data from BigQuery...')
    df = self.client.query(f'SELECT * FROM `{table}`').to_dataframe()
    self.logger.info('Loaded %d rows and %d columns.', len(df), len(df.columns))
    return df
# cslasses
