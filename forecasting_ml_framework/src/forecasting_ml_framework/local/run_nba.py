#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The NBA end-to-end run: warehouse to warehouse, on a workstation.

This driver executes the framework's own node functions -- the same
``load_input_data``, ``fit_registry``, ``apply_registry``, ``model_training``,
``model_evaluation``, ``lift_calculation`` and ``model_scoring`` that the
production pipeline graph wires together. Nothing about the modelling is
reimplemented here. What changes is where the data lives: SQLite in place of
BigQuery, a local directory in place of object storage, and a local Spark
session. The catalog's SQL is translated rather than replaced, so the business
rules encoded in it are the rules this run applies.

The run is organised as the pipeline stages it mirrors, and each stage is a
function returning its output -- so a failing stage fails at a named place with a
report of what it had produced, rather than somewhere inside a graph runner.

Two properties of the framework are genuinely under test here, and both are
structural rather than numerical:

* **Train/serve parity.** The scoring stage reloads the serialised registry from
  local disk and applies it to a frame it has never seen. If the registry
  captured anything at fit time that is not in the artefact, the scoring frame
  gets a different shape from the training frame and the run fails on the model,
  not on the registry.
* **The self-describing model handle.** The trainer is pickled, dropped, and
  reloaded from disk before anything is scored. A model that cannot be scored
  from its own artefact is a model that cannot be deployed.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from forecasting_ml_framework import constants
from forecasting_ml_framework.core.metadata import Metadata
from forecasting_ml_framework.core.registry import Registry
from forecasting_ml_framework.local import spark_session
from forecasting_ml_framework.local.nba import (
  FEATURE_COLUMNS,
  describe_source,
  reset_workspace,
  seed_warehouse,
  workspace_paths,
)
from forecasting_ml_framework.local.sql_translate import to_sqlite
from forecasting_ml_framework.local.sqlite_warehouse import WRITE_APPEND, WRITE_TRUNCATE, SQLiteWarehouse
from forecasting_ml_framework.nodes.feature_nodes import (
  apply_registry,
  build_registry,
  fit_registry,
)
from forecasting_ml_framework.nodes.model_nodes import (
  lift_calculation,
  model_evaluation,
  model_scoring,
  model_training,
  spark_to_pandas,
)
from forecasting_ml_framework.observability.logging import configure_logging, get_logger

LOGGER = get_logger(__name__)

#: The model key the run is registered under.
MODEL_KEY = 'nba_retention'

#: The suffix every artefact and dataset name carries. The framework's central
#: isolation mechanism, and it is exercised here for the same reason it exists
#: in production: one checkout, several model instances, no collisions.
SUFFIX = 'nba'

#: The date the run pretends to be. The label query is anchored to it, so it is
#: a configuration value rather than today's date -- a run must be reproducible.
TRAIN_RUN_DATE = '2024-01-01'

#: The positive class, declared as the catalog declares it.
RETAINED_STATE = 'retained'
ATTRITION_STATE = 'attrition'

#: Placeholder warehouse coordinates. SQLite has neither projects nor datasets,
#: so these names are never resolved -- but the run declares them so it reaches
#: the platform session builder through the same path production does, rather
#: than a local-only bypass.
LOCAL_PROJECT = 'local'
LOCAL_DATASET = 'nba'
LOCAL_SOURCE_DATASET = 'nba_source'

#: The label-derivation query, in the catalog's shape. The three business rules
#: it encodes are the framework's, and the local run executes them rather than
#: reading a label that was handed to it.
TRAIN_INPUT_SQL = """
WITH deactivations AS (
  SELECT esn, deactivation_reason, deactivation_date
  FROM `${data_project}.${db_source_data}.nba_deactivations`
  WHERE DATE(deactivation_date) > DATE('${train_run_dt}')
    AND DATE(deactivation_date) <= DATE_ADD(DATE('${train_run_dt}'), INTERVAL 88 DAY)
), dedup_deactivations AS (
  SELECT esn, deactivation_reason, deactivation_date
  FROM (
    SELECT esn, deactivation_reason, deactivation_date,
           ROW_NUMBER() OVER (PARTITION BY esn ORDER BY deactivation_date DESC) AS rn
    FROM deactivations
  ) WHERE rn = 1
), customer_info AS (
  SELECT service_id, cust_id, line_seq_id, account_ref, mobile_ref
  FROM (
    SELECT service_id, cust_id, line_seq_id, account_ref, mobile_ref,
           ROW_NUMBER() OVER (PARTITION BY service_id) AS rn
    FROM `${data_project}.${db_source_data}.nba_customer_info`
    WHERE prcs_date = DATE_ADD(DATE('${train_run_dt}'), INTERVAL 88 DAY)
  ) WHERE rn = 1
), features AS (
  SELECT * FROM `${data_project}.${db_staging_data}.nba_features`
  WHERE run_date = DATE('${train_run_dt}')
)
SELECT
  f.*,
  c.cust_id, c.line_seq_id, c.account_ref, c.mobile_ref,
  CASE WHEN d.deactivation_date IS NULL THEN '${retained_state}'
       WHEN d.deactivation_date < DATE_ADD(DATE('${train_run_dt}'), INTERVAL 58 DAY) THEN '${retained_state}'
       ELSE '${attrition_state}' END AS raw_state,
  CASE WHEN d.deactivation_date IS NULL THEN 0
       WHEN d.deactivation_date < DATE_ADD(DATE('${train_run_dt}'), INTERVAL 58 DAY) THEN 0
       ELSE 1 END AS label,
  '${model_key}' AS model_key
FROM features f
LEFT JOIN dedup_deactivations d ON f.service_id = d.esn
LEFT JOIN customer_info c ON f.service_id = c.service_id
"""

#: The scoring query: features, identifiers and the model key, and no label at
#: all. A production scoring frame never carries an outcome, and neither does
#: this one -- so a leak between the two paths is a structural error, not a
#: number that happens to look too good.
SCORE_INPUT_SQL = """
WITH customer_info AS (
  SELECT service_id, cust_id, line_seq_id, account_ref, mobile_ref
  FROM `${data_project}.${db_source_data}.nba_customer_info`
)
SELECT f.*, c.cust_id, c.line_seq_id, c.account_ref, c.mobile_ref,
       '${model_key}' AS model_key
FROM `${data_project}.${db_staging_data}.nba_features` f
LEFT JOIN customer_info c ON f.service_id = c.service_id
WHERE f.run_date = DATE('${train_run_dt}')
"""


@dataclass
class RunReport:
  """Accumulates what each stage produced, for the final summary.

  A long pipeline that fails at stage six should be able to say what stages one
  to five did, and a run that succeeds should be able to show the numbers rather
  than only assert on them.

  Attributes:
    stages: Stage name to elapsed seconds, in execution order.
    facts: Stage name to a small mapping of reported values.
    tables: Warehouse table name to row count after the run.
    artefacts: Artefact name to the path written.
  """

  stages: list[tuple[str, float]] = field(default_factory=list)
  facts: dict[str, dict[str, Any]] = field(default_factory=dict)
  tables: dict[str, int] = field(default_factory=dict)
  artefacts: dict[str, str] = field(default_factory=dict)

  def record(self, name: str, seconds: float, **facts: Any) -> None:
    """Record one completed stage.

    Args:
      name: The stage name.
      seconds: How long it took.
      **facts: Reported values for the summary.
    """
    self.stages.append((name, seconds))
    self.facts[name] = facts
    LOGGER.info('Stage complete', extra={'stage': name, 'seconds': round(seconds, 2), **facts})


def _parameters() -> dict[str, Any]:
  """Build the parameters document the node functions read.

  Returns:
    A parameters document carrying the model program, the modelling block and
    the globals the routines resolve their column roles from.
  """
  return {
    # These are read from the *top level* of the parameters document, not from
    # the globals or from a modelling block. A metric row that omits them is
    # still a valid frame -- every column exists -- and it is only discoverable
    # as wrong when someone queries the warehouse by model and finds nothing.
    #
    # The session block is not optional once a model key is declared: the
    # platform builder treats a named model as a real run and requires the full
    # configuration. A local run is a real run, so it declares one. The project
    # and dataset names are strings the local warehouse never uses -- SQLite has
    # no projects -- but declaring them is what lets the run reach the session
    # builder through the same path production does.
    constants.MODEL_KEY: MODEL_KEY,
    constants.RUN_DATE: TRAIN_RUN_DATE,
    constants.TRAIN_RUN_DATE: TRAIN_RUN_DATE,
    constants.SCORE_RUN_DATE: TRAIN_RUN_DATE,
    # The local environment: the embedded store backs every data transaction and no
    # credentials are required, which is the combination that makes a local run a
    # valid correctness test rather than a production run on an unmanaged machine.
    constants.ENV_IS_LOCAL: True,
    constants.ENV_IS_HOSTED: False,
    'external_project': LOCAL_PROJECT,
    'data_project': LOCAL_PROJECT,
    'db_staging_data': LOCAL_DATASET,
    'db_source_data': LOCAL_SOURCE_DATASET,
    'globals': {},
    'modeling_params': _modeling_block(),
    'data_prep_params': _routine_program(),
  }


def _globals() -> dict[str, Any]:
  """Build the globals document that declares the column roles.

  Returns:
    A mapping of role key to column names, plus the slices the runner injects.
  """
  return {
    constants.ROLE_IDENTIFIER: ['service_id'],
    constants.ROLE_CATEGORICAL: [],
    constants.ROLE_NUMERICAL: list(FEATURE_COLUMNS),
    constants.MODEL_KEY: MODEL_KEY,
    constants.RUN_DATE: TRAIN_RUN_DATE,
    constants.TRAIN_RUN_DATE: TRAIN_RUN_DATE,
    constants.SCORE_RUN_DATE: TRAIN_RUN_DATE,
  }


def _routine_program() -> list[dict[str, Any]]:
  """Build the feature-transform program.

  Each entry is one routine with its configuration. The program is the framework's
  central declarative artefact: the same list is used to fit the registry and,
  after serialisation, to apply it at scoring time.

  Returns:
    The routine configuration list.
  """
  # The catalog passes *resolved column names* here, via `${globals:NUM_COL}`,
  # not the role keys. The distinction matters: a role key is a label the
  # contract carries, and a routine that received one would treat the string
  # 'NUM_COL' as a column name and fail on a frame that has no such column.
  numeric = list(FEATURE_COLUMNS)
  return [
    {
      'Imputations': {
        'skip': False,
        'args': {
          'strategy': 'mean',
          'cols': {'exclude_cols': [], 'include_cols': numeric},
        },
      }
    },
    {
      'Normalizations': {
        'skip': False,
        'args': {
          'handle': 'pyspark.ml.feature.MinMaxScaler',
          'min_val': 0.0,
          'max_val': 1.0,
          'cols': {'exclude_cols': [], 'include_cols': numeric},
        },
      }
    },
    {
      'CapOutliers': {
        'skip': True,
        'args': {
          'method': 'bound',
          'bound': {'lower': 0.01, 'upper': 0.99},
          'cols': {'exclude_cols': [], 'include_cols': numeric},
        },
      }
    },
  ]


def _modeling_block() -> dict[str, Any]:
  """Build the modelling configuration block.

  Returns:
    The modelling block, resolved by the node layer as ``modeling_params``.
  """
  return {
    'model_handle': 'sklearn.ensemble.HistGradientBoostingClassifier',
    'model_params': {
      'max_depth': 4,
      'max_iter': 150,
      'learning_rate': 0.06,
      'l2_regularization': 0.5,
      'min_samples_leaf': 20,
      'early_stopping': False,
      'random_state': 42,
    },
    # No ``fit_params``: the framework splats them into the estimator's ``fit``,
    # and ``eval_set`` is the gradient-boosting-library convention, not
    # scikit-learn's. A scikit-learn estimator configures its own validation
    # split through its constructor, so a holdout passed at fit time is a
    # TypeError rather than an early-stopping signal.
    'fit_params': {},
    'calibration_handle': 'sklearn.calibration.CalibratedClassifierCV',
    'calibration_params': {'method': 'isotonic', 'cv': 3},
    'balance_class_weight': False,
    'eval_size': 0.25,
    'validation_flag': False,
    'prediction_col': constants.DEFAULT_PREDICTION_COLUMN,
    'prediction_probability_col': constants.DEFAULT_PREDICTION_PROBABILITY_COLUMN,
    # 'mcc_curve' rather than an invented name: an unrecognised method falls back
    # to 0.5 with a warning, which on a 12% positive rate predicts almost nobody
    # positive and makes the run look successful while the threshold is doing
    # nothing. MCC is the right objective here because the operating point is a
    # business decision on a rare outcome, not an accuracy decision.
    'threshold_selection_method': 'mcc_curve',
    'include_cols': list(FEATURE_COLUMNS),
    'exclude_cols': [],
  }


def _build_metadata(artefact_root: str) -> Metadata:
  """Build the schema contract the pipeline threads through every stage.

  Args:
    artefact_root: The local directory that routines persist their fitted Spark
      pipelines under. A routine whose fitted state cannot be pickled writes a
      Spark pipeline model instead, and it needs somewhere to put it -- which in
      production is object storage and here is a directory.

  Returns:
    A contract with the identifier, the target and the feature roles declared.
  """
  Path(artefact_root).mkdir(parents=True, exist_ok=True)
  metadata = Metadata()
  # The field names are the contract's own: `id_cols` and `seq_cols`. Writing
  # `identifier_cols` and `sequence_cols` created two brand-new attributes that
  # nothing else reads, so the contract reported no identifiers and no sequences
  # while appearing to have declared both. Every downstream consequence followed:
  # the registry projection dropped the identifier, and the published score frame
  # carried no key to join on.
  metadata.id_cols = ['service_id']
  metadata.target_col = 'label'
  metadata.numerical_cols = list(FEATURE_COLUMNS)
  metadata.categorical_cols = []
  metadata.seq_cols = []
  metadata.feature_cols = list(FEATURE_COLUMNS)
  metadata.intermediate_file_path = artefact_root
  return metadata


def _resolve_sql(sql: str) -> str:
  """Expand the references in a catalog query and translate it for SQLite.

  Args:
    sql: The query.

  Returns:
    A SQLite query.
  """
  resolved = (
    sql.replace('${data_project}', LOCAL_PROJECT)
      .replace('${db_source_data}', LOCAL_SOURCE_DATASET)
      .replace('${db_staging_data}', LOCAL_DATASET)
      .replace('${train_run_dt}', TRAIN_RUN_DATE)
      .replace('${model_key}', MODEL_KEY)
      .replace('${retained_state}', RETAINED_STATE)
      .replace('${attrition_state}', ATTRITION_STATE)
  )
  return to_sqlite(resolved)


def stage_load_input(warehouse_path: str) -> pd.DataFrame:
  """Execute the label-derivation query and return the labelled frame.

  Args:
    warehouse_path: The SQLite file.

  Returns:
    A pandas ``DataFrame`` with a derived label.
  """
  warehouse = SQLiteWarehouse(warehouse_path)
  try:
    return warehouse.query(_resolve_sql(TRAIN_INPUT_SQL))
  finally:
    warehouse.close()


def stage_load_score_input(warehouse_path: str) -> pd.DataFrame:
  """Execute the scoring query, which carries no outcome.

  Args:
    warehouse_path: The SQLite file.

  Returns:
    A pandas ``DataFrame`` of features and identifiers.
  """
  warehouse = SQLiteWarehouse(warehouse_path)
  try:
    return warehouse.query(_resolve_sql(SCORE_INPUT_SQL))
  finally:
    warehouse.close()


def _coerce_types(frame: pd.DataFrame) -> pd.DataFrame:
  """Give every column a concrete dtype before it reaches Spark.

  SQLite is dynamically typed, so a query returns every column as ``object``.
  Spark's schema inference refuses a column it cannot type, and the error names a
  data type rather than a column -- which is a poor way to learn that a numeric
  feature arrived as strings. Each object column is therefore coerced here, and a
  column that coerces to *neither* all-numeric nor all-text is a genuine mixed
  column and is reported by name rather than guessed at.

  Args:
    frame: The frame as the query returned it.

  Returns:
    A frame with no ``object`` dtype remaining, other than text.
  """
  coerced = frame.copy()
  for column in coerced.columns:
    if coerced[column].dtype != object:
      continue
    if coerced[column].isna().all():
      raise LocalRunError(
        f'Column {column!r} is entirely null. That is a join that matched nothing, not '
        f'a data problem: check that the referenced table is populated and that the '
        f'join key is the same column on both sides.'
      )
    if coerced[column].isna().any():
      raise LocalRunError(
        f'Column {column!r} is partially null ({int(coerced[column].isna().sum())} of '
        f'{len(coerced)} rows). A nullable identifier or key column usually means an '
        f'outer join did not match every row.'
      )

    # The null check must precede the numeric test, and the numeric test must be
    # reached only when the column is genuinely non-numeric. Reading the order the
    # other way round is what turned a nullable numeric feature into text: a
    # single null made `notna().all()` false, so the column fell through to the
    # text branch, where `astype(str)` wrote the literal string 'nan' into the
    # data -- a value that then passed every downstream check as a real category.
    as_number = pd.to_numeric(coerced[column], errors='coerce')
    if as_number.notna().all():
      coerced[column] = as_number.astype('float64') if as_number.dtype.kind == 'f' else as_number.astype('int64')
      continue
    coerced[column] = coerced[column].astype(str)
  return coerced


def _to_spark(frame: pd.DataFrame) -> Any:
  """Convert a frame to a Spark DataFrame using the active session.

  Args:
    frame: The pandas frame.

  Returns:
    A Spark ``DataFrame``.
  """
  from pyspark.sql import SparkSession  # noqa: PLC0415

  return SparkSession.getActiveSession().createDataFrame(_coerce_types(frame))


def _assert_row_count_preserved(stage: str, before: int, after: int) -> None:
  """Assert that a stage neither added nor lost rows.

  A row-count change across a transform is either a bug or an undeclared
  sampling step, and both must be visible. A fan-out join is the case that
  matters: it inflates the count, the run still trains, and every metric is then
  computed over duplicated entities -- so the number looks plausible and means
  something else entirely.

  Args:
    stage: The stage name, for the message.
    before: The row count entering the stage.
    after: The row count leaving it.

  Raises:
    LocalRunError: If the counts differ.
  """
  if before != after:
    raise LocalRunError(
      f'{stage} changed the row count from {before} to {after}. A transform must be '
      f'reshape-only. If the change is intended, the stage must declare it rather '
      f'than relying on the caller to notice.'
    )


def run(report: RunReport | None = None) -> RunReport:
  """Execute the whole pipeline and return what each stage produced.

  Args:
    report: An existing report to append to. A new one is created when omitted.

  Returns:
    The populated report.
  """
  report = report or RunReport()
  paths = workspace_paths()

  # ------------------------------------------------------------------ #
  # 0. Workspace
  # ------------------------------------------------------------------ #
  started = time.perf_counter()
  configure_logging()
  reset_workspace()
  counts = seed_warehouse(paths['warehouse'])
  warehouse = SQLiteWarehouse(paths['warehouse'])
  report.record('seed_warehouse', time.perf_counter() - started, **counts)

  started = time.perf_counter()
  source = describe_source('/tmp/opencode/nba/nba.csv') if Path('/tmp/opencode/nba/nba.csv').exists() else {}
  environment = spark_session.configure_local_environment()
  spark = spark_session.create_local_session()
  report.record(
    'start_spark',
    time.perf_counter() - started,
    spark_version=spark.version,
    **{f'env_{key.lower()}': value for key, value in environment.items()},
  )

  parameters = _parameters()
  metadata = _build_metadata(str(Path(paths['objects']) / 'routine_artefacts'))
  globals_ = _globals()

  # ------------------------------------------------------------------ #
  # 1. fe_training: load -> fit -> apply -> pandas
  # ------------------------------------------------------------------ #
  started = time.perf_counter()
  labelled = stage_load_input(paths['warehouse'])
  warehouse.write(labelled, 'nba_train_input', disposition=WRITE_TRUNCATE)
  report.record(
    'load_input_data',
    time.perf_counter() - started,
    rows=len(labelled),
    source_rows=counts['nba_features'],
    positives=int(labelled['label'].sum()),
    positive_rate=round(float(labelled['label'].mean()), 4),
  )

  started = time.perf_counter()
  spark_frame = _to_spark(labelled)
  registry = build_registry(globals_, parameters['data_prep_params'])
  fitted, transformed, transformed_meta = fit_registry(
    spark_frame, metadata, registry, parameters
  )
  _assert_row_count_preserved('fit_registry', spark_frame.count(), transformed.count())
  report.record(
    'fit_registry',
    time.perf_counter() - started,
    routines=len(fitted.get_routines()),
    columns_in=len(spark_frame.columns),
    columns_out=len(transformed.columns),
  )

  registry_path = Path(paths['objects']) / f'registry_{SUFFIX}.pkl'
  with registry_path.open('wb') as handle:
    pickle.dump(fitted, handle)
  report.artefacts['fitted_registry'] = str(registry_path)

  started = time.perf_counter()
  reloaded: Registry = pickle.loads(registry_path.read_bytes())
  spark_score = _to_spark(stage_load_score_input(paths['warehouse']))
  score_shape_before = len(spark_score.columns)
  scored_frame, score_meta = apply_registry(
    spark_score, metadata, parameters, 'score', reloaded
  )
  _assert_row_count_preserved('apply_registry', spark_score.count(), scored_frame.count())
  report.record(
    'apply_registry',
    time.perf_counter() - started,
    routines=len(reloaded.get_routines()),
    rows=scored_frame.count(),
    columns_in=score_shape_before,
    columns_out=len(scored_frame.columns),
  )
  # Train/serve parity, stated as the invariant it actually is. The two frames
  # are NOT expected to have the same columns: the training frame carries the
  # outcome and the scoring frame must not, so equality of column counts would be
  # the wrong assertion. What must hold is that the registry produced the same
  # *feature* schema on both, and that the scoring frame gained no column the
  # training frame lacks.
  train_columns = set(transformed.columns)
  score_columns = set(scored_frame.columns)
  extra_on_score = sorted(score_columns - train_columns)
  if extra_on_score:
    raise LocalRunError(
      f'Train/serve parity broken: the scoring frame has column(s) {extra_on_score} that the '
      f'training frame does not. The serialised registry produced a different schema.'
    )
  missing_features = [column for column in FEATURE_COLUMNS if column not in score_columns]
  if missing_features:
    raise LocalRunError(
      f'Train/serve parity broken: the scoring frame is missing feature(s) {missing_features} '
      f'that the training frame carries.'
    )
  report.facts['parity'] = {
    'train_columns': len(train_columns),
    'score_columns': len(score_columns),
    'outcome_only_on_train': sorted(train_columns - score_columns),
  }

  started = time.perf_counter()
  preprocessed, preprocessed_meta = spark_to_pandas(transformed, transformed_meta, True)
  report.record(
    'spark_to_pandas',
    time.perf_counter() - started,
    rows=len(preprocessed),
    columns=len(preprocessed.columns),
  )
  _assert_row_count_preserved('spark_to_pandas', transformed.count(), len(preprocessed))

  # ------------------------------------------------------------------ #
  # 2. model_training
  # ------------------------------------------------------------------ #
  started = time.perf_counter()
  trainer, fitted_model, calibrator, train_slice, holdout = model_training(
    parameters, preprocessed, preprocessed_meta, 'sklearn'
  )
  report.record(
    'model_training',
    time.perf_counter() - started,
    estimator=type(fitted_model).__name__,
    features=len(getattr(trainer, 'features_to_use', [])),
    train_rows=len(train_slice),
    holdout_rows=len(holdout),
    calibrator=type(calibrator).__name__ if calibrator else 'prior-shift',
    threshold=round(float(getattr(trainer, 'optimal_model_threshold', float('nan'))), 6),
  )

  # The three model artefacts are persisted and then re-read from disk, and
  # everything downstream uses only the re-read copies. This is the round trip
  # the catalog expresses as ``model_handle`` / ``fitted_model`` /
  # ``calibrated_model``, and it is the only version of this test that means
  # anything: an in-memory trainer would satisfy every assertion below without
  # once crossing a serialisation boundary.
  started = time.perf_counter()
  objects = Path(paths['objects'])
  handle_path = objects / f'model_handle_{SUFFIX}.pkl'
  fitted_path = objects / f'fitted_model_{SUFFIX}.pkl'
  calibrator_path = objects / f'calibrated_model_{SUFFIX}.pkl'
  with handle_path.open('wb') as handle:
    pickle.dump(trainer, handle)
  with fitted_path.open('wb') as handle:
    pickle.dump(fitted_model, handle)
  with calibrator_path.open('wb') as handle:
    pickle.dump(calibrator, handle)
  report.artefacts.update({
    'model_handle': str(handle_path),
    'fitted_model': str(fitted_path),
    'calibrated_model': str(calibrator_path),
  })
  report.record(
    'persist_model_trio',
    time.perf_counter() - started,
    handle_bytes=handle_path.stat().st_size,
    fitted_bytes=fitted_path.stat().st_size,
  )

  restored = pickle.loads(handle_path.read_bytes())
  restored_model = pickle.loads(fitted_path.read_bytes())
  restored_calibrator = pickle.loads(calibrator_path.read_bytes())
  del trainer, fitted_model, calibrator

  # The train slice and the holdout are kept, not deleted. The holdout is the
  # only frame in this run that no part of fitting or model selection ever saw,
  # so it is the honest test slice, and the per-slice metric and lift tables need
  # each slice scored on its own rows.
  if train_slice.empty or holdout.empty:
    raise LocalRunError(
      f'The trainer returned an empty slice: train={len(train_slice)}, holdout={len(holdout)}. A '
      f'test slice cannot be evaluated, and evaluating one is the point of the per-slice tables.'
    )

  # The two slices must partition the frame, with no row in both. They do not
  # merely happen to: the holdout is drawn FROM `preprocessed`, so anything that
  # evaluates the parent frame as if it were the train slice silently includes
  # every test row. This is the invariant that catches it, and it is checked
  # before any metric is computed rather than inferred from the counts afterwards,
  # because a contaminated train row does not look wrong in isolation -- it looks
  # like a good model.
  if len(train_slice) + len(holdout) != len(preprocessed):
    raise LocalRunError(
      f'The train slice and the holdout do not partition the frame: '
      f'{len(train_slice)} + {len(holdout)} != {len(preprocessed)}. Either the split '
      f'drops or duplicates rows, and every per-slice metric below would be computed '
      f'over the wrong population.'
    )
  # The identifier is read from the metadata rather than hard-coded, so the check
  # follows whatever the contract declares. A hard-coded name here would silently
  # degrade to a no-op if the contract changed, and a no-op overlap check is worse
  # than none: it reads as a passing assertion.
  id_columns = [name for name in (preprocessed_meta.id_cols or []) if name in train_slice.columns]
  if not id_columns:
    raise LocalRunError(
      f'None of the declared id columns {list(preprocessed_meta.id_cols or [])} is present in the '
      f'train slice, so the slices cannot be proven disjoint. Declared columns: '
      f'{sorted(train_slice.columns)[:10]}.'
    )
  id_column = id_columns[0]
  overlap = set(train_slice[id_column]) & set(holdout[id_column])
  if overlap:
    raise LocalRunError(
      f'{len(overlap)} row(s) appear in both the train slice and the holdout, so the '
      f'test metrics are not out-of-sample. First offenders: {sorted(overlap)[:5]}.'
    )

  # ------------------------------------------------------------------ #
  # 3. Scoring, evaluation and lift
  # ------------------------------------------------------------------ #
  started = time.perf_counter()
  # The train slice is scored on ITS OWN rows. Scoring the parent frame here --
  # which is what this did before -- reports a "train" number that contains every
  # test row: the confusion counts summed to 1340 against a 1005-row train slice,
  # and the train/test overlap was the full 335-row holdout.
  train_scored, calibrated_scored, _ = model_scoring(
    restored, restored_model, restored_calibrator, train_slice, preprocessed_meta, 'train'
  )
  report.record(
    'model_scoring',
    time.perf_counter() - started,
    rows=len(train_scored),
    score_column=constants.SCORE_VALUE if constants.SCORE_VALUE in train_scored.columns else 'score_value',
    calibrated_rows=len(calibrated_scored),
  )

  # The test slice is scored through the same restored artefacts, so the metrics
  # published under `test` are produced by exactly the path a production scoring
  # run would use -- a pickle that was written, dropped and re-read -- rather
  # than by the in-memory objects the trainer happened to still hold.
  test_scored, test_calibrated_scored, _ = model_scoring(
    restored, restored_model, restored_calibrator, holdout, preprocessed_meta, 'test'
  )
  report.record(
    'model_scoring_test',
    time.perf_counter() - started,
    rows=len(test_scored),
    calibrated_rows=len(test_calibrated_scored),
  )

  started = time.perf_counter()
  # Each slice is evaluated on ITS OWN frame. The previous version called
  # `model_evaluation` three times on the same scored frame with three different
  # `data_type` labels, so `train`, `valid` and `test` reported byte-identical
  # metrics and the only thing distinguishing the rows was the label itself. A
  # test row that is a copy of the training row is worse than no test row: it
  # invites a reviewer to read an in-sample number as out-of-sample.
  #
  # The trainer's own holdout is the honest test slice -- it is the only frame
  # here that no part of fitting or model selection ever saw. It is re-scored
  # through the restored artefacts, so the test path also crosses the same
  # serialisation boundary the production scoring path does.
  train_rows = len(train_slice)
  holdout_rows = len(holdout)
  # Keyed ``{slice}_{scale}`` so the publish step can address a table by name
  # without re-deriving the label. ``raw`` and ``calib`` are the two score
  # scales the framework carries side by side.
  metric_frames: dict[str, pd.DataFrame] = {}
  lift_frames: dict[str, pd.DataFrame] = {}
  slice_rows: dict[str, int] = {}

  # `multi_class` is passed positionally because it occupies the seventh slot in
  # the node signature, ahead of the sequencing token. The graph binds eight
  # inputs in that order, so a call that omits it shifts `sequence_flag` into the
  # wrong parameter and raises before the first metric is computed. The flag is
  # read from the same document the graph reads, so the harness and a production
  # run agree on which metric path is taken.
  multiclass_flag = bool((parameters.get('evaluation_params') or {}).get('multiclass_flag', False))

  # Each slice carries its OWN pair of scored frames: the raw one and the
  # calibrated one. The pair is held together deliberately. An earlier version
  # built the raw/calibrated mapping once, outside this loop, from the TRAIN
  # slice's frames, and then indexed it by slice name -- so the test slice was
  # evaluated against the training rows and every test row came back identical
  # to its train twin (roc_auc 0.897 on both). The mapping has to be per-slice,
  # because the calibrated frame is a function of the rows it was calibrated on.
  #
  # The two scales are the evidence the architecture calls the 2x2 matrix:
  # {raw, calibrated} x {train, holdout}. Publishing only one scale hides the
  # diagnostic that matters most, because a calibrated score that has collapsed
  # towards the base rate is the signature of a prior-correction
  # misconfiguration, and it is invisible in the raw metrics.
  slices: tuple[tuple[str, pd.DataFrame, pd.DataFrame], ...] = (
    ('train', train_scored, calibrated_scored),
    ('test', test_scored, test_calibrated_scored),
  )
  # The column prefix that selects a scale's columns from a frame. The raw scale
  # uses the bare names; the calibrated scale prefixes both with `calib_`, so the
  # calibrated read is a different column of the SAME frame rather than a
  # different frame.
  scale_prefixes = (('raw', ''), ('calib', constants.CALIBRATION_PREFIX))

  for data_type, slice_frame, slice_calibrated in slices:
    if slice_frame is None:
      continue
    slice_rows[data_type] = len(slice_frame)
    for scale_name, prefix in scale_prefixes:
      # Both scales are read off the slice's own frame. The calibrated columns
      # only exist on it when calibration actually ran, so the calibrated scale is
      # only attempted when the columns are present.
      if scale_name == 'calib' and (slice_calibrated is None or slice_calibrated.empty):
        LOGGER.warning(
          'Skipping a score scale: the slice has no calibrated frame',
          extra={'slice': data_type, 'scale': scale_name},
        )
        continue
      frame = slice_calibrated if scale_name == 'calib' else slice_frame
      prediction_column = f'{prefix}{constants.DEFAULT_PREDICTION_COLUMN}'
      probability_column = f'{prefix}{constants.DEFAULT_PREDICTION_PROBABILITY_COLUMN}'
      if probability_column not in frame.columns:
        # Without the probability column the metrics are undefined, and
        # evaluating it anyway would report a number that does not describe
        # anything. Skip the scale and say so rather than publish a null row.
        LOGGER.warning(
          'Skipping a score scale: the probability column is absent',
          extra={'slice': data_type, 'scale': scale_name, 'column': probability_column},
        )
        continue
      key = f'{data_type}_{scale_name}'
      metric_frame, _ = model_evaluation(
        frame,
        parameters,
        preprocessed_meta,
        prediction_column,
        probability_column,
        data_type,
        multiclass_flag,
        sequence_flag=False,
      )
      metric_frames[key] = metric_frame
      slice_lift, _ = lift_calculation(
        frame,
        parameters,
        preprocessed_meta,
        prediction_column,
        probability_column,
        data_type,
        sequence_flag=False,
      )
      lift_frames[key] = slice_lift

  # The headline is the raw train slice, which is the first key written by the
  # loop. It is a summary convenience only: the per-slice tables are the record,
  # and the test-slice row is the one a reviewer should read.
  headline = next(
    (frame.iloc[0] for frame in metric_frames.values() if not frame.empty),
    pd.Series(dtype=object),
  )
  report.record(
    'model_evaluation',
    time.perf_counter() - started,
    evaluations=len(metric_frames),
    scales=sorted({key.rsplit('_', 1)[-1] for key in metric_frames}),
    train_rows=train_rows,
    holdout_rows=holdout_rows,
    tested_rows=slice_rows.get('test'),
    roc_auc=round(float(headline.get('roc_auc', float('nan'))), 4),
    accuracy=round(float(headline.get('accuracy', float('nan'))), 4),
    precision=round(float(headline.get('precision', float('nan'))), 4),
    recall=round(float(headline.get('recall', float('nan'))), 4),
    f1=round(float(headline.get('f1', float('nan'))), 4),
  )

  # The confusion counts must agree with the scalar metrics derived from them.
  # The weaker "the four counts sum to the population" check cannot catch a
  # transposition, because a transposed matrix still sums to the same total.
  # Every scale and every slice is checked, not just the headline: the two
  # thresholds are selected independently, so a disagreement on one scale says
  # nothing about the other and skipping the check would hide it.
  for key, frame in metric_frames.items():
    if frame.empty or not all(column in frame.columns for column in ('tn', 'fp', 'fn', 'tp')):
      continue
    row = frame.iloc[0]
    predicted_positive = float(row['tp']) + float(row['fp'])
    actual_positive = float(row['tp']) + float(row['fn'])
    if predicted_positive and abs(float(row['precision']) - float(row['tp']) / predicted_positive) > 0.01:
      raise LocalRunError(
        f'Confusion counts disagree with precision on {key}: tp={row["tp"]} fp={row["fp"]} '
        f'gives {float(row["tp"]) / predicted_positive:.3f}, but the reported precision is '
        f'{row["precision"]}.'
      )
    if actual_positive and abs(float(row['recall']) - float(row['tp']) / actual_positive) > 0.01:
      raise LocalRunError(
        f'Confusion counts disagree with recall on {key}: tp={row["tp"]} fn={row["fn"]} '
        f'gives {float(row["tp"]) / actual_positive:.3f}, but the reported recall is '
        f'{row["recall"]}.'
      )

  # The lift table is a warehouse table, so it carries the same provenance the
  # metric row does. A band table with no model key cannot be attributed to a
  # run, and a NULL column is a valid value rather than a visible error.
  #
  # ``score_scale`` is stamped onto every band row. Without it a lift table from
  # the raw scale and one from the calibrated scale are indistinguishable, which
  # matters because the two answer different questions: the raw lift measures
  # ranking, the calibrated lift measures whether the corrected probability is
  # trustworthy as a probability.
  lift_frames = {
    key: frame.assign(model_key=MODEL_KEY, run_date=TRAIN_RUN_DATE, score_scale=key.rsplit('_', 1)[-1])
    for key, frame in lift_frames.items()
    if not frame.empty
  }
  report.record(
    'lift_calculation',
    time.perf_counter() - started,
    tables=len(lift_frames),
    bands=sum(len(frame) for frame in lift_frames.values()),
    slices=sorted(slice_rows),
  )

  # ------------------------------------------------------------------ #
  # 4. Score the untouched frame through the restored artefacts
  # ------------------------------------------------------------------ #
  started = time.perf_counter()
  score_frame, score_frame_meta = spark_to_pandas(scored_frame, score_meta, True)
  published, _, _ = model_scoring(
    restored, restored_model, restored_calibrator, score_frame, score_frame_meta, 'score'
  )
  warehouse.write(published, 'nba_score_table', disposition=WRITE_TRUNCATE)
  report.record(
    'publish_score_table',
    time.perf_counter() - started,
    rows=len(published),
    mean_score=round(float(published[constants.SCORE_VALUE].mean()), 4)
    if constants.SCORE_VALUE in published.columns else None,
  )

  # ------------------------------------------------------------------ #
  # 5. Publish to the warehouse
  # ------------------------------------------------------------------ #
  started = time.perf_counter()
  metric_frames = {
    key: frame.assign(model_key=MODEL_KEY, run_date=TRAIN_RUN_DATE, score_scale=key.rsplit('_', 1)[-1])
    for key, frame in metric_frames.items()
    if not frame.empty
  }
  try:
    # One metric table and one lift table per {slice, scale} pair, named for what
    # they contain. Previously every slice was written to a single `train_`-prefixed
    # table, so the row labelled `test` sat in a table called `train` and a
    # reviewer reading the test row had no way to tell it was not the training
    # number. The table name now states the slice, which is the same convention
    # the production catalog uses (`train_model_evaluation_*`, `test_lift_df_*`,
    # each with a `_calib_` twin).
    #
    # Every per-slice table is a full snapshot for one slice and one scale, so it
    # truncates. Re-running must replace it rather than accumulate rows, otherwise
    # the band counts double and every rate in the table is wrong.
    for key in sorted(metric_frames):
      slice_name, _, scale_name = key.rpartition('_')
      table_prefix = f'{slice_name}_{scale_name}'
      warehouse.write(metric_frames[key], f'{table_prefix}_metrics_{SUFFIX}', disposition=WRITE_TRUNCATE)
      slice_lift = lift_frames.get(key)
      if slice_lift is not None and not slice_lift.empty:
        warehouse.write(slice_lift, f'{table_prefix}_lift_{SUFFIX}', disposition=WRITE_TRUNCATE)

    # A score-slice table is a partitioned time series, so it appends. A truncate
    # there would make re-running a day look identical to running it once, which
    # is precisely the distinction the two dispositions exist to express. It
    # carries the *test* slice's raw metric row only -- previously the whole
    # multi-slice frame was relabelled `data_type='test'` and written, so the
    # table held three rows all claiming to be the test slice.
    test_raw = metric_frames.get('test_raw')
    if test_raw is not None and not test_raw.empty:
      warehouse.write(test_raw, f'score_metrics_{SUFFIX}', disposition=WRITE_APPEND)
    report.tables = {name: warehouse.count(name) for name in warehouse.table_names()}
  finally:
    warehouse.close()
  report.record(
    'publish_metrics',
    time.perf_counter() - started,
    tables=len(report.tables),
    slices=sorted(slice_rows),
    scales=sorted({key.rsplit('_', 1)[-1] for key in metric_frames}),
  )

  spark.stop()
  report.facts['source'] = source
  report.facts['globals'] = globals_
  return report


class LocalRunError(RuntimeError):
  """Raised when a local run violates a framework invariant."""


def format_report(report: RunReport) -> str:
  """Render a run report as readable text.

  Args:
    report: The report to render.

  Returns:
    A multi-line summary.
  """
  lines = ['=' * 78, 'NBA END-TO-END RUN'.center(78), '=' * 78, '']
  source = report.facts.get('source', {})
  if source:
    lines.append(
      f"dataset : {source.get('rows')} rows x {source.get('columns')} columns, "
      f"positive rate {source.get('positive_rate')}"
    )
  lines.append('')
  lines.append(f"{'STAGE':<26}{'SECONDS':>9}   DETAIL")
  lines.append('-' * 78)
  for name, seconds in report.stages:
    detail = ', '.join(
      f'{key}={value}' for key, value in report.facts.get(name, {}).items() if not key.startswith('env_')
    )
    lines.append(f'{name:<26}{seconds:>9.2f}   {detail[:150]}')
  if report.artefacts:
    lines += ['', 'ARTEFACTS', '-' * 78]
    lines += [f'  {key:<22}{value}' for key, value in report.artefacts.items()]
  if report.tables:
    lines += ['', 'WAREHOUSE TABLES', '-' * 78]
    lines += [f'  {key:<34}{value:>8} rows' for key, value in sorted(report.tables.items())]
  lines.append('')
  lines.append('=' * 78)
  return '\n'.join(lines)


def main() -> int:
  """Run the pipeline and print the report.

  Returns:
    A process exit code.
  """
  report = run()
  print(format_report(report))
  print(json.dumps(report.tables, indent=2))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
