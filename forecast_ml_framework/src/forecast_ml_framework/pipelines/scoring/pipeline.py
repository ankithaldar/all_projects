#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Kedro pipeline for batch scoring (Phase 2).

Wiring: ``score_data`` -> ``predict`` -> ``write_predictions``.

Canonical catalog datasets (configured in ``conf/``, not here):
- ``scoring_input`` (in): batch to score. Back it with a
  core ``SQLQueryDataSet`` / ``SparkSQLDataSet`` or a
  ``kedro_datasets`` BigQuery/Athena dataset; credentials
  come from catalog config/env, never hardcoded.
- ``scoring_output`` (out): canonical scored output written
  by the catalog, e.g. a BigQuery (GCP) / Athena (AWS)
  table via ``kedro_datasets``. Scheduled-ready: re-running
  the pipeline re-scores from the catalog input.

Back-compat defaults: when ``create_pipeline`` is called
without ``scoring_params``, the pipeline wires the legacy
datasets ``scoring_input`` (in) and ``predictions`` (out).
When ``scoring_params`` supplies ``input_dataset`` /
``output_dataset``, those names are wired instead. The
conf worker aligns ``conf/`` to the canonical names above
(``scoring_input`` / ``scoring_output``).

Params (``params:scoring_params``, a dict). Expected keys
for conf alignment:
  input_dataset: catalog name of the scoring input
    (canonical ``scoring_input``).
  output_dataset: catalog name of the scored output
    (canonical ``scoring_output``).
  model_path: pickle path, loaded via FileModelLoader.
    Required unless another loader is given.
  model_name / alias: registered model name / MLflow alias
    for registry loading (e.g. via RegistryModelLoader);
    also recorded as ``score_model_id`` when no explicit
    id is set.
  model_handle: dotted string for an estimator-like class
    (with ``fit``/``predict``) from an allowlisted module
    (``sklearn``, ``xgboost``, ``lightgbm``, ``catboost``,
    ``h2o``, ``forecast_ml_framework``); other modules
    (``os``, ``subprocess``, ``builtins``, ...) are
    rejected.
  model_loader / model: injected callable loader or ready
    model (tests). File/registry-agnostic: any callable
    returning a model with ``predict`` works.
  id_cols: identifier columns carried into the output
    (default ``['id']``).
  feature_cols: feature columns to score on; default is
    every column except id/target columns.
  task_type: 'classification' (default) or 'regression';
    controls whether proba is populated.
  batch_size: rows per scoring chunk (positive int,
    default 1000). Larger batches use more memory.
  max_rows: maximum input rows accepted (default 100000
    for local safety; env ``SCORING_MAX_ROWS`` overrides).
    ``null`` means unbounded (required for the 100M-row
    cloud path). Beyond a finite cap, scoring raises
    ``ValueError``.
  max_cols: maximum input columns accepted (default
    1000; env ``SCORING_MAX_COLS`` overrides). ``null``
    means unbounded (500-feature ingestion needs this).
  spark_scoring: ``{'enabled': True}`` (or bool) to score
    Spark DataFrames partitioned via ``mapPartitions``
    without ``toPandas()``. Default off (pandas path).
  partition_batch_size: rows per chunk inside each Spark
    partition (default 10000, falls back to ``batch_size``).
    Larger values use more executor memory.

Cloud scaling path (100M rows x 500 cols, no driver OOM):
  BigQuery (catalog ``scoring_input``) -> Spark on Dataproc
  with autoscaling (``spark.dynamicAllocation.enabled``,
  AQE, shuffle partitions from ``spark_params``; see
  ``core.spark_factory.get_spark`` -- config only, cluster
  submission is out of scope) -> partitioned scoring
  (``spark_scoring.enabled``, ``max_rows: null``,
  ``max_cols: null``, ``partition_batch_size`` per
  partition) -> BigQuery write via the catalog
  ``scoring_output`` dataset. The driver never materialises
  the full frame in pandas; executor partitions hold only
  ``partition_batch_size`` rows at a time. Note: for real
  Spark the model object must be broadcast/picklable --
  the partition function closes over it.

Conf keys for the config worker (documented here, not set
in ``conf/`` by this task): ``scoring_params.max_rows``,
``scoring_params.max_cols``, ``scoring_params.spark_scoring``
(``enabled``, ``partition_batch_size``),
``scoring_params.partition_batch_size``, and
``pandas_conversion_params.max_rows_for_pandas`` (default
5,000,000; guard in ``core.spark_factory.check_pandas_limits``
directs oversized pandas conversions to Spark-native FE /
distributed training).
  score_model_id / model_id: recorded on every output
    row (default falls back to model_name/alias, then
    'unknown').
  destination: optional dataset object with ``save``;
    default is the catalog output dataset.
'''

from kedro.pipeline import Pipeline, node

from forecast_ml_framework.pipelines.scoring.nodes import (
  load_scoring_data,
  predict,
  write_predictions,
)


def create_pipeline(scoring_params=None, **kwargs):
  '''Create the batch scoring Kedro pipeline.

  Args:
    scoring_params: Optional dict; its ``input_dataset`` /
      ``output_dataset`` entries wire the first/last node
      I/O. Defaults to ``scoring_input`` / ``predictions``
      for back-compat when omitted.
    **kwargs: Forwarded to Pipeline (e.g. tags, namespace).

  Returns:
    Kedro Pipeline: score_data -> predict -> write_predictions.
  '''
  params = scoring_params or {}
  input_dataset = params.get('input_dataset') or 'scoring_input'
  output_dataset = params.get('output_dataset') or 'predictions'
  return Pipeline([
    node(func=load_scoring_data,
         inputs=[input_dataset, 'params:scoring_params'],
         outputs='scoring_data',
         name='score_data'),
    node(func=predict,
         inputs=['scoring_data', 'params:scoring_params'],
         outputs='scored_predictions',
         name='predict'),
    node(func=write_predictions,
         inputs=['scored_predictions', 'params:scoring_params'],
         outputs=output_dataset,
         name='write_predictions'),
  ], **kwargs)
