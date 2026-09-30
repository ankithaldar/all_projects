#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Persistence layer: dataset adapters.

Each adapter satisfies the same contract: validate configuration eagerly with no
I/O, return every configuration key from ``_describe``, and refuse serialisation
when it holds a live engine object. A new backend therefore needs only a class
and a catalog ``type:`` entry.
"""

from forecasting_ml_framework.persistence.bigquery_api import BQQueryDataSet, BQTableDataSet
from forecasting_ml_framework.persistence.bigquery_spark import CustomSparkBQDataSet
from forecasting_ml_framework.persistence.gbq_pandas import GBQTableDataSet
from forecasting_ml_framework.persistence.parquet import ParquetDataSet

__all__ = [
  'BQQueryDataSet',
  'BQTableDataSet',
  'CustomSparkBQDataSet',
  'GBQTableDataSet',
  'ParquetDataSet',
]
