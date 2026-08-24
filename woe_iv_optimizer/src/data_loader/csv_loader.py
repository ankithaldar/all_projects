#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data loader for CSV files."""


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

class CSVDataLoader(DataLoader):
  """Data loader for CSV files."""

  def __init__(self, config: Dict[str, Any]):
    super().__init__(config)
    self.source = config['csv']['file_path']  # Can be local path or URL
    self.logger = logging.getLogger(self.logger_name)

  def load_data(self) -> pd.DataFrame:
    self.logger.info(f"Loading CSV data from: {self.source}")

    # Use pandas' native support for URLs (http, https, s3, gcs, etc.)
    try:
      df = pd.read_csv(
        self.source,
        sep=self.config['csv'].get('separator', ','),
        encoding=self.config['csv'].get('encoding', 'utf-8'),
        low_memory=False,
        storage_options=self._get_storage_options()
      )
    except Exception as e:
      self.logger.error(f"Failed to load CSV from {self.source}: {e}")
      raise

    self.logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
    return df

  def _get_storage_options(self) -> dict:
    """
    Return storage options for cloud backends (e.g., GCS, S3).
    These are passed to pandas.read_csv via `storage_options`.

    Example config for GCS:
      csv:
        file_path: "gs://bucket/data.csv"
        gcp_project: "my-project"

    Example for S3:
      csv:
        file_path: "s3://bucket/data.csv"
        s3_credentials:
          key: "..."
          secret: "..."
    """
    parsed = urlparse(self.source)
    scheme = parsed.scheme.lower()

    if scheme == 'gs':
      project = self.config['csv'].get('gcp_project')
      if project:
        return {'token': 'anon', 'project': project}  # or use default credentials
      return {}  # rely on ADC

    elif scheme == 's3':
      creds = self.config['csv'].get('s3_credentials', {})
      if creds:
        return {
          'key': creds.get('key'),
          'secret': creds.get('secret'),
          'token': creds.get('token')  # optional
        }
      # Otherwise, rely on boto3 default credentials (env vars, IAM, etc.)

    # For http/https, no storage options needed
    return {}
# classes
