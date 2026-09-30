#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Local execution: the same framework, a SQLite warehouse and a local disk.

This package is the counterpart the production code does not have. The framework
targets a cloud warehouse and object storage; a developer, a test or a review
needs to run the *same* node functions, the *same* catalog queries and the *same*
trainer against data on a workstation.

What lives here, and why each piece is a substitution rather than a rewrite:

* :mod:`sql_translate` renders the production catalog's SQL into SQLite, so the
  business rules encoded in those queries -- the outcome window, the positive
  class, label maturity -- are the rules the local run executes.
* :mod:`sqlite_warehouse` supplies the table and write-disposition semantics the
  production adapters get from BigQuery.
* :mod:`datasets` presents those as Kedro datasets, so a catalog entry changes
  only in its ``type:`` line.
* :mod:`spark_session` handles the two environment facts that stop Spark
  starting on a workstation.
* :mod:`nba` provides the source data and seeds the warehouse.

Nothing in the framework's own layers imports this package. The dependency runs
one way, from here into the framework, which is what keeps a local harness from
becoming a second implementation of the thing it is testing.
"""

from __future__ import annotations

from forecasting_ml_framework.local.nba import (
  NBA_SOURCE_URL,
  default_data_dir,
  download,
  reset_workspace,
  seed_warehouse,
  workspace_paths,
)
from forecasting_ml_framework.local.spark_session import (
  configure_local_environment,
  create_local_session,
)
from forecasting_ml_framework.local.sql_translate import to_sqlite
from forecasting_ml_framework.local.sqlite_warehouse import SQLiteWarehouse

__all__ = [
  'NBA_SOURCE_URL',
  'SQLiteWarehouse',
  'configure_local_environment',
  'create_local_session',
  'default_data_dir',
  'download',
  'reset_workspace',
  'seed_warehouse',
  'to_sqlite',
  'workspace_paths',
]
