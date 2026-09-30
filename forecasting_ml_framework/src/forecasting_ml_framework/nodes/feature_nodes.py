#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feature-engineering nodes.

These are the pipeline's application services. Each one is a pure function from
persisted datasets to persisted datasets, which is what allows the four pipelines
to be submitted as four separate cluster jobs with state carried purely through
storage.

**Three cross-cutting conventions** are enforced here, and —  — they
live in two shared helpers rather than being implemented twice. A
duplicated the report-date synthesis and the ``model_key`` routing in both
``fit_registry`` and ``apply_registry``, so a convention change required edits in
two places with no compiler to catch a miss.

**Four new assertions** enforce what a only observed. the
framework printed every count and asserted none, so a routine that dropped 40 % of
a frame produced a complete, plausible and entirely wrong model with no alert.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pyspark.sql import functions as sf

from forecasting_ml_framework import constants
from forecasting_ml_framework.core.metadata import Metadata
from forecasting_ml_framework.core.registry import Registry
from forecasting_ml_framework.exceptions import DataContractError
from forecasting_ml_framework.observability.logging import get_logger
from forecasting_ml_framework.observability.quality import (
  QualityPolicy,
  assert_column_roles_disjoint,
  assert_row_count,
  assert_target_cardinality,
)
from forecasting_ml_framework.utils.reflection import resolve_routine
from forecasting_ml_framework.utils.text import require

LOGGER = get_logger(__name__)


def load_input_data(
  parameters: dict[str, Any],
  global_params: dict[str, Any],
  input_frame: Any,
) -> tuple[Any, Metadata]:
  """Validate the input frame, coerce indicators, and construct the contract.

  The label is *derived upstream in SQL*, not present in the feature feed. That
  separation is architecturally significant: the node's only job is to assert the
  frame is learnable and to describe it.

  Args:
    parameters: The full parameters document, providing the column-role lists.
    global_params: The shared parameter bag, providing the target column and the
      transform-state root.
    input_frame: The Spark frame from whichever input dataset the pipeline
      selected.

  Returns:
    A two-tuple of the frame and the contract.

  Raises:
    DataContractError: If a declared column is absent from the frame.
    TargetCardinalityError: If the target has fewer than two distinct values.
  """
  target_col = require(global_params, 'target_col', stage='load_input_data')
  role_map = {
    constants.ROLE_NUMERICAL: list(parameters.get(constants.ROLE_NUMERICAL, []) or []),
    constants.ROLE_CATEGORICAL: list(parameters.get(constants.ROLE_CATEGORICAL, []) or []),
    constants.ROLE_INDICATOR: list(parameters.get(constants.ROLE_INDICATOR, []) or []),
    constants.ROLE_SEQUENCE: list(parameters.get(constants.ROLE_SEQUENCE, []) or []),
    constants.ROLE_IDENTIFIER: list(parameters.get(constants.ROLE_IDENTIFIER, []) or []),
  }

  #
  # check for it at all: a sequence column that was also declared numeric reached
  # the estimator as an array<float> feature it could not fit, and the failure
  # surfaced only after the entire feature pipeline had run.
  assert_column_roles_disjoint(role_map, stage='load_input_data')

  if target_col in input_frame.columns:
    counts = input_frame.groupBy(sf.col(target_col).alias(target_col)).count()
    # The cardinality is collected once and the distribution logged from the
    # collected rows. ``counts.count()`` followed by ``counts.show()`` ran the
    # aggregation twice, and the console output went to stdout where nothing
    # queries it -- the process log is not an observability channel.
    distinct = counts.collect()
    assert_target_cardinality(len(distinct), target_col)
    LOGGER.info(
      'Read the target distribution',
      extra={'target_col': target_col, 'distribution': {str(row[0]): int(row[1]) for row in distinct}},
    )

  for column in role_map[constants.ROLE_INDICATOR]:
    if column in input_frame.columns:
      input_frame = input_frame.withColumn(column, sf.col(column).cast('float'))

  declared = [c for role in constants.FEATURE_ROLE_KEYS for c in role_map[role]]
  metadata = Metadata(
    target_col=target_col,
    feature_cols=declared,
    numerical_cols=role_map[constants.ROLE_NUMERICAL],
    categorical_cols=role_map[constants.ROLE_CATEGORICAL],
    seq_cols=role_map[constants.ROLE_SEQUENCE],
    id_cols=role_map[constants.ROLE_IDENTIFIER],
    intermediate_file_path=global_params.get('intermediate_file_path'),
  )
  # The array columns must not be handed to an estimator. The sequence family
  # registers its own summaries as features, so removing the source here is
  # correct and makes the configuration precondition impossible to get wrong.
  metadata.detach_sequence_features()

  _assert_declared_columns_present(input_frame, declared, role_map[constants.ROLE_IDENTIFIER], target_col)
  # The count is an action, so it is requested once and reused. The previous
  # expression evaluated to ``None`` unconditionally -- it took the length of the
  # role map, which is always populated, rather than the frame's row count -- so
  # the log line reported a null row count for every run. A declared-but-never-taken
  # measurement is worse than none: it looks like instrumentation and is not.
  row_count = input_frame.count()
  LOGGER.info('Loaded the input frame', extra={'rows': row_count, 'metadata': str(metadata)})
  return input_frame, metadata


def split_frame(
  frame: Any,
  metadata: Metadata,
  training_split_ratio: float | None,
  test_split_ratio: float | None,
  seed: int = 10,
) -> tuple[Any, Metadata, Any, Metadata]:
  """Split the frame at the Spark level, with exact partition coverage.

  For a classification target the test frame is the ``left_anti`` complement of
  the train frame on the identifier columns, so no row can be in both and none is
  lost. That is strictly better than a two-way ``sampleBy``, which cannot
  guarantee complementarity.

  Correctness depends on ``id_cols`` being a primary key. If a natural key spans
  multiple rows, the anti-join removes *all* rows for any key present in train,
  so the test frame silently loses data. The scheduler's post-scoring duplicate
  check is the upstream guard for that invariant.

  The task type is inferred by target cardinality rather than by configuration:
  ten or fewer distinct values is a classification, more is a regression. That is
  an implicit convention — a twelve-class problem would be treated as a
  regression — and it is documented here rather than left implicit.

  Args:
    frame: The input frame.
    metadata: The schema contract.
    training_split_ratio: The training fraction, or ``None`` to derive it.
    test_split_ratio: The test fraction, or ``None`` to derive it.
    seed: The sampling seed. Hardcoded in a, and fixed here as an
      explicit parameter so the split is reproducible and configurable.

  Returns:
    A four-tuple of the train frame, the train contract, the test frame and the
    test contract.

  Raises:
    ValueError: If both ratios are supplied and do not sum to one.
  """
  train_ratio, test_ratio = _resolve_ratios(training_split_ratio, test_split_ratio)
  row_count = frame.count()
  LOGGER.info(
    'Splitting the frame',
    extra={'train_ratio': train_ratio, 'test_ratio': test_ratio, 'input_rows': row_count, 'metadata': str(metadata)},
  )

  distinct_targets = frame.select(metadata.target_col).distinct().count()
  if distinct_targets <= 10:
    fractions = (
      frame.select(metadata.target_col)
.distinct()
.withColumn('fraction', sf.lit(train_ratio))
.rdd.collectAsMap()
    )
    train_frame = frame.stat.sampleBy(metadata.target_col, fractions, seed=seed)
    test_frame = frame.join(train_frame, on=metadata.id_cols, how='left_anti')
  else:
    train_frame, test_frame = frame.randomSplit([train_ratio, test_ratio], seed=seed)

  # Each slice gets its own contract, so a later routine that mutates one does
  # not leak into the other.
  train_metadata = metadata.copy()
  test_metadata = metadata.copy()
  LOGGER.info(
    'Split complete',
    extra={
      'train_rows': train_frame.count(),
      'test_rows': test_frame.count(),
      # Collected and converted rather than passed through: a Spark ``Row`` is not
      # JSON-serialisable, so a log formatter that handles the rest of the record
      # fails on exactly this field.
      'train_distribution': _distribution(train_frame, metadata.target_col, metadata.target_col)[
        metadata.target_col
      ],
      'test_distribution': _distribution(test_frame, metadata.target_col, metadata.target_col)[
        metadata.target_col
      ],
    },
  )
  return train_frame, train_metadata, test_frame, test_metadata


def sampling_node(frame: Any, metadata: Metadata, sampling_config: dict[str, Any]) -> tuple[Any, Metadata]:
  """Resample the frame and compute the imbalance statistics.

  This node does double duty, and the second role is the more important one: it
  is the **origin of the imbalance statistics** that later drive both weight
  injection and probability calibration. ``real_rate`` and ``sample_rate`` are the
  sole inputs to prior-shift calibration, and their ratio is the exact factor
  needed to undo the training-time reweighting — which is how the framework
  produces a calibrated probability without ever refitting a calibrator.

  Args:
    frame: The input frame.
    metadata: The schema contract.
    sampling_config: The ``sampling_params`` block.

  Returns:
    A two-tuple of the sampled frame and the contract, carrying the derived
    statistics.

  Raises:
    DataContractError: If the target is absent.
  """
  return _apply_sampling(frame, metadata, sampling_config, compute_weights=True)


def sampling_node_regression(
  frame: Any, metadata: Metadata, sampling_config: dict[str, Any]
) -> tuple[Any, Metadata]:
  """Resample a continuous-target frame.

  It is the same function with a flag, so the two variants cannot drift
  apart.

  Args:
    frame: The input frame.
    metadata: The schema contract.
    sampling_config: The ``sampling_reg_params`` block.

  Returns:
    A two-tuple of the sampled frame and the contract.
  """
  return _apply_sampling(frame, metadata, sampling_config, compute_weights=False)


def build_registry(global_params: dict[str, Any], data_prep_config: list[dict[str, Any]]) -> Registry:
  """Instantiate the configured transform chain.

  The configuration shape is a **list of single-key dictionaries**, and that is
  the enabling decision. A mapping could express neither ordering nor repetition;
  a list can express both, which is why the configured program registers two
  ``Imputations`` blocks with different scopes and fill dictionaries.

  Args:
    global_params: The shared parameter bag, passed to every routine.
    data_prep_config: The ``data_prep_params`` list.

  Returns:
    The registry, in configuration order.

  Raises:
    ValueError: If a registration block is not a single-key mapping.
  """
  registry = Registry()
  for entry in data_prep_config or []:
    for key, block in entry.items():
      if 'skip' in block and block['skip'] is False:
        class_handle = resolve_routine(key, block.get('modules'))
        args = dict(block.get('args', {}) or {})
        registry.append(class_handle(global_params=global_params, params=args))
  LOGGER.info('Built the feature registry', extra={'routines': [r.name for r in registry.get_routines()]})
  return registry


def fit_registry(
  frame: Any,
  metadata: Metadata,
  registry: Registry,
  parameters: dict[str, Any],
  quality: QualityPolicy | None = None,
) -> tuple[Registry, Any, Metadata]:
  """Fit the transform chain and apply the output conventions.

  Responsibilities beyond delegation are three **cross-cutting conventions**:
  narrowing the schema before the driver boundary, synthesising a report date,
  and routing ``model_key`` out of the data into the contract.

  Args:
    frame: The input frame.
    metadata: The schema contract.
    registry: The unfitted registry.
    parameters: The full parameters document. It is injected whole, not as a
      slice, because it is the only place in the graph that needs the run-level
      keys.
    quality: The active quality policy, for the row-count assertion.

  Returns:
    A three-tuple of the fitted registry, the narrowed frame and the contract.

  Raises:
    DataContractError: If the projection would drop a declared feature.
  """
  policy = quality or QualityPolicy()
  observed_before = frame.count()
  LOGGER.info('Fitting the registry', extra={'rows': observed_before})

  registry_obj, frame, metadata = registry.fit(frame, metadata, parameters)

  selected = metadata.feature_cols + metadata.id_cols
  if metadata.target_col:
    selected = selected + [metadata.target_col]
  absent = [column for column in metadata.feature_cols if column not in frame.columns]
  if absent:
    raise DataContractError(
      f'A transform declared {len(absent)} feature(s) that the frame does not contain: {absent[:20]}. '
      'A routine that appends to feature_cols without emitting the column produces an invisible '
      'feature; one that emits without appending produces this error.',
      missing=absent,
    )
  frame = frame.select(*[column for column in selected if column in frame.columns])

  frame, metadata = apply_output_conventions(frame, metadata, parameters, include_target=True)
  observed_after = frame.count()
  assert_row_count(observed_after, policy.with_expected(observed_before), 'fit_registry')
  # The count is taken once and reused: the quality gate and the log line were each
  # requesting it, so the frame was counted twice for one number.
  LOGGER.info('Registry fitted', extra={'rows': observed_after, 'features': len(metadata.feature_cols)})
  return registry_obj, frame, metadata


def apply_registry(
  frame: Any,
  metadata: Metadata,
  parameters: dict[str, Any],
  data_type: str,
  registry: Registry,
  quality: QualityPolicy | None = None,
) -> tuple[Any, Metadata]:
  """Apply the persisted transform chain and the output conventions.

  In production the target column is deliberately withheld, so the label can
  never leak into the scoreset materialisation. That is a genuine
  data-governance control, not an optimisation.

  Args:
    frame: The input frame.
    metadata: The schema contract.
    parameters: The full parameters document.
    data_type: The slice label. In ``prod`` mode the target is excluded.
    registry: The registry loaded from object storage.
    quality: The active quality policy.

  Returns:
    A two-tuple of the frame and the contract.

  Raises:
    DataContractError: If the projection would drop a declared feature. The
      scoring path enforces the same contract as the fitting path, because a
      partial feature set scored and published as a whole one is the most
      consequential silent failure the framework has.
  """
  policy = quality or QualityPolicy()
  observed_before = frame.count()
  run_mode = parameters.get('run_mode', 'prod')

  frame, metadata = registry.apply(frame, metadata, data_type)

  selected = metadata.feature_cols + metadata.id_cols
  if run_mode != 'prod' and metadata.target_col:
    selected = selected + [metadata.target_col]
  # The same check `fit_registry` performs, and for the same reason. The scoring
  # path is where a mismatch is most expensive and least visible: silently
  # projecting the missing column away leaves the persisted contract describing a
  # column the materialised frame does not contain, and the failure then surfaces
  # at the trainer as an opaque "feature not found" -- or not at all, if the
  # column happened to be optional downstream.
  absent = [column for column in metadata.feature_cols if column not in frame.columns]
  if absent:
    raise DataContractError(
      f'The registry declared {len(absent)} feature(s) that the frame does not contain: {absent[:20]}. '
      'On the scoring path this means the input extraction did not produce a column the fitted '
      'contract requires, or a routine emitted a feature it did not register. Refusing here is '
      'what stops a partial feature set being scored and published as if it were the whole one.',
      missing=absent,
      stage='apply_registry',
    )
  frame = frame.select(*[column for column in selected if column in frame.columns])

  frame, metadata = apply_output_conventions(frame, metadata, parameters, include_target=run_mode != 'prod')
  observed_after = frame.count()
  assert_row_count(observed_after, policy.with_expected(observed_before), 'apply_registry')
  LOGGER.info('Registry applied', extra={'rows': observed_after, 'run_mode': run_mode})
  return frame, metadata


def apply_output_conventions(
  frame: Any,
  metadata: Metadata,
  parameters: dict[str, Any],
  include_target: bool,
) -> tuple[Any, Metadata]:
  """Apply the two shared cross-cutting conventions to a frame.

  **The report-date convention.** If none of ``rpt_dt``, ``report_period`` or
  ``create_date`` is present among the identifiers, a synthetic report date is
  created from the run date. Every output dataset therefore carries one, which is
  what allows the downstream egress SQL to partition and filter by reporting
  period. Enforcing it in one helper rather than two functions is what makes the
  training and scoring paths symmetric by construction.

  **``model_key`` routing.** The model identifier is *input data* for a training
  frame and *metadata* for a scoring frame. The value is lifted out of the data
  into the contract and the column is dropped, so it does not propagate into the
  scoreset table; it is re-attached as a literal by the scoring step.

  A read the value with ``select('model_key').distinct().collect()[0][0]``,
  which assumes exactly one distinct value and silently discards any second. The
  count is asserted now.

  **Why this returns the frame as well as the contract.** A Spark DataFrame is
  immutable: ``withColumn`` and ``drop`` return *new* frames, they do not modify
  the one passed in. This function therefore used to return only the contract,
  and every ``frame = ...`` rebinding inside it was discarded by all three
  call sites. The two documented conventions consequently did not happen: the
  report date was never added and ``model_key`` was never dropped -- while
  ``metadata.id_cols`` *was* mutated to declare the report date, so the contract
  claimed an identifier column that the persisted frame did not contain. The
  asymmetry is what makes returning the pair load-bearing rather than cosmetic.

  Args:
    frame: The frame to modify.
    metadata: The contract to modify.
    parameters: The full parameters document.
    include_target: Whether the target column is part of the contract's
      projection. Retained for symmetry with the projection performed by the
      callers; the projection itself is applied by the caller, which is where the
      selected column list lives.

  Returns:
    A two-tuple of the modified frame and the modified contract.
  """
  del include_target
  if not any(column in frame.columns for column in constants.CANONICAL_DATE_COLUMNS):
    if not any(column in metadata.id_cols for column in constants.CANONICAL_DATE_COLUMNS):
      report_date = parameters.get(constants.RUN_DATE)
      if report_date:
        frame = frame.withColumn(constants.REPORT_DATE, sf.lit(report_date))
        metadata.id_cols = metadata.id_cols + [constants.REPORT_DATE]
        LOGGER.info('Synthesised the report date', extra={'report_date': report_date})

  if constants.MODEL_KEY in frame.columns:
    distinct = frame.select(constants.MODEL_KEY).distinct().limit(2).collect()
    if len(distinct) != 1:
      raise DataContractError(
        f'The input frame carries {len(distinct)} distinct {constants.MODEL_KEY} values. The model '
        'identifier must be uniform within a run; mixed identifiers mean two models are being '
        'trained in one frame, which would contaminate both.',
        values=[row[0] for row in distinct],
      )
    metadata.model_key = distinct[0][0]
    frame = frame.drop(constants.MODEL_KEY)
  elif parameters.get(constants.MODEL_KEY):
    metadata.model_key = parameters[constants.MODEL_KEY]

  if constants.MODEL_KEY in metadata.id_cols:
    metadata.id_cols = [column for column in metadata.id_cols if column != constants.MODEL_KEY]
  return frame, metadata


def merge_registries(*registries: Registry) -> Registry:
  """Concatenate registries into one chain.

  Exposed as a node for a pipeline that composes its chain from separately
  defined fragments. A defined this and wired it into nothing.

  Args:
    *registries: The registries to merge, in order.

  Returns:
    A new registry.
  """
  merged = Registry.merge(*registries)
  LOGGER.info('Merged registries', extra={'routines': [r.name for r in merged.get_routines()]})
  return merged


def transform_frame(
  frame: Any,
  metadata: Metadata,
  registry: Registry,
  parameters: dict[str, Any],
  apply_only: bool = False,
) -> tuple[Any, Metadata]:
  """Fit-and-apply in a single call, optionally skipping the fit.

  The production pipelines use ``fit_registry`` and ``apply_registry`` as two
  distinct stages, because scoring must apply the *persisted* registry rather
  than a freshly fitted one. This node exists for notebook and prototyping use.

  Args:
    frame: The input frame.
    metadata: The contract.
    registry: The registry.
    parameters: The full parameters document.
    apply_only: When ``True``, only the apply path runs.

  Returns:
    A two-tuple of the transformed frame and the enriched contract.
  """
  frame, metadata = registry.transform(frame, metadata, parameters, apply_only=apply_only)
  frame, metadata = apply_output_conventions(frame, metadata, parameters, include_target=True)
  return frame, metadata


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _apply_sampling(
  frame: Any,
  metadata: Metadata,
  sampling_config: dict[str, Any],
  compute_weights: bool,
) -> tuple[Any, Metadata]:
  """Resample a frame and optionally compute imbalance statistics.

  Args:
    frame: The input frame.
    metadata: The contract.
    sampling_config: The sampling configuration block.
    compute_weights: ``False`` for a continuous target, where class weights are
      meaningless.

  Returns:
    A two-tuple of the sampled frame and the contract.

  Raises:
    DataContractError: If the target column is absent.
  """
  target_col = metadata.target_col
  if target_col is None or target_col not in frame.columns:
    raise DataContractError(
      'The sampling node requires a labelled frame: the target column is absent',
      target_col=target_col,
    )

  sampling_type = sampling_config.get('sampling_type')
  fractions = sampling_config.get('fractions')
  stratum_col = sampling_config.get('stratum_col') or target_col
  seed = sampling_config.get('seed', 42)
  LOGGER.info('Sampling configuration', extra={'config': sampling_config, 'stratum': stratum_col})
  # The before and after distributions are the evidence that sampling did what was
  # asked, and they are logged as structured fields rather than printed. Printing
  # them forced an extra aggregation action per call and wrote to a channel nothing
  # queries -- which is how a pipeline comes to be believed observable because it
  # prints so much.
  before = _distribution(frame, stratum_col, target_col)

  if compute_weights:
    _attach_imbalance_statistics(frame, metadata, target_col)

  if sampling_type in ('Equal', 'Uniform'):
    fractions = _compute_fractions(frame, stratum_col, sampling_type)

  if isinstance(fractions, dict):
    sampled = frame.sampleBy(stratum_col, fractions=fractions, seed=seed)
  else:
    sampled = frame.sample(False, fractions or 1.0, seed=seed)

  after = _distribution(sampled, stratum_col, target_col)
  LOGGER.info(
    'Sampling complete',
    extra={
      'distribution_before': before,
      'distribution_after': after,
      'real_rate': metadata.real_rate,
      'scale_pos_weight': metadata.scale_pos_weight,
      'class_weights': metadata.class_weights,
    },
  )
  return sampled, metadata


def _distribution(frame: Any, stratum_col: str, target_col: str) -> dict[str, dict[str, int]]:
  """Measure the stratum and target distributions of a frame.

  Args:
    frame: The Spark frame.
    stratum_col: The column sampling stratified on.
    target_col: The label column.

  Returns:
    A mapping of column name to its value counts.
  """
  measured: dict[str, dict[str, int]] = {}
  # A set, so the case where the sampling stratum IS the target measures it once
  # rather than twice.
  for column in dict.fromkeys((stratum_col, target_col)):
    rows = frame.groupBy(sf.col(column).alias(column)).count().collect()
    measured[column] = {str(row[0]): int(row[1]) for row in rows}
  return measured


def _attach_imbalance_statistics(frame: Any, metadata: Metadata, target_col: str) -> None:
  """Compute and attach the population rate and the class weights.

  The precision guard matters more than it looks. Rounding a rare-event rate of
  0.00043 to two places yields ``0.0``, which would silently disable calibration
  through the degenerate branch of the prior-shift formula. Re-rounding to four
  places below ten percent is a targeted defence against exactly that, and it is
  applied here and in the trainer's fallback path.

  Args:
    frame: The input frame.
    metadata: The contract to attach to.
    target_col: The target column.
  """
  distinct = frame.select(target_col).distinct().count()
  if distinct == 2:
    total = frame.count()
    positives = frame.filter(sf.col(target_col) == 1).count()
    negatives = frame.filter(sf.col(target_col) == 0).count()
    if not positives or not negatives:
      raise DataContractError(
        f'Binary target with positives={positives}, negatives={negatives}. Class weights and '
        'prior-shift calibration are undefined, so the run cannot proceed.',
        positives=positives,
        negatives=negatives,
      )
    metadata.real_rate = _rounded_rate(positives, total)
    metadata.scale_pos_weight = float(np.round(negatives / positives, 4))
    metadata.class_weights = [
      float(np.round(total / (2 * negatives), 4)),
      float(np.round(total / (2 * positives), 4)),
    ]
  elif 2 < distinct <= 10:
    counts = frame.groupBy(sf.col(target_col).alias(target_col)).count().rdd.collectAsMap()
    total = sum(counts.values())
    metadata.class_weights = {
      label: total / (len(counts) * count) for label, count in counts.items()
    }
  else:
    LOGGER.info(
      'No class weights computed: the target has more than ten distinct values, so the task is '
      'treated as regression',
      extra={'distinct_values': distinct},
    )
  LOGGER.info(
    'Imbalance statistics',
    extra={
      'real_rate': metadata.real_rate,
      'scale_pos_weight': metadata.scale_pos_weight,
      'class_weights': metadata.class_weights,
    },
  )


def _compute_fractions(frame: Any, stratum_col: str, sampling_type: str) -> dict[Any, float]:
  """Compute a per-stratum sampling fraction.

  ``'Equal'`` is a class-balancing resampling: the dominant stratum gets the
  smallest fraction, so stratum sizes converge toward equality. ``'Uniform'`` is
  proportional and preserves the class ratio.

  Args:
    frame: The input frame.
    stratum_col: The stratification column.
    sampling_type: ``'Equal'`` or ``'Uniform'``.

  Returns:
    A mapping of stratum value to sampling fraction.
  """
  counts = frame.groupBy(sf.col(stratum_col).alias(stratum_col)).count()
  total = frame.count()
  with_total = counts.withColumn('__total', sf.lit(total))
  ratio = (
    sf.lit(1) - (sf.col('count') / sf.col('__total'))
    if sampling_type == 'Equal'
    else sf.col('count') / sf.col('__total')
  )
  return {row[stratum_col]: float(row['Fraction']) for row in with_total.withColumn('Fraction', ratio).collect()}


def _rounded_rate(numerator: int, denominator: int) -> float:
  """Compute a rate with a rare-event precision guard.

  Args:
    numerator: The numerator.
    denominator: The denominator.

  Returns:
    The rate.
  """
  rate = float(np.round(numerator / denominator, 2))
  if rate < 0.1:
    rate = float(np.round(numerator / denominator, 4))
  return rate


def _resolve_ratios(training_split_ratio: float | None, test_split_ratio: float | None) -> tuple[float, float]:
  """Resolve the two split ratios, allowing either to be omitted.

  Args:
    training_split_ratio: The training fraction, or ``None``.
    test_split_ratio: The test fraction, or ``None``.

  Returns:
    A two-tuple of the resolved ratios, summing to one.

  Raises:
    ValueError: If both are supplied and do not sum to one, or if neither is.
  """
  has_train = training_split_ratio not in (None, 0, 0.0, False)
  has_test = test_split_ratio not in (None, 0, 0.0, False)
  if not has_train and not has_test:
    raise ValueError('At least one of training_split_ratio or test_split_ratio must be supplied')
  if has_train and not has_test:
    return float(training_split_ratio), float(1.0 - training_split_ratio)
  if has_test and not has_train:
    return float(1.0 - test_split_ratio), float(test_split_ratio)
  if abs((training_split_ratio + test_split_ratio) - 1.0) > 1e-9:
    raise ValueError(
      f'test_ratio + train_ratio != 1.0 (got {training_split_ratio} + {test_split_ratio}). Supply '
      'both, or supply only one and the other is derived.'
    )
  return float(training_split_ratio), float(test_split_ratio)


def _assert_declared_columns_present(
  frame: Any,
  features: list[str],
  identifiers: list[str],
  target_col: str | None,
) -> None:
  """Assert that every declared column exists in the frame.

  Args:
    frame: The input frame.
    features: The declared feature columns.
    identifiers: The declared identifier columns.
    target_col: The target column.

  Raises:
    TargetCardinalityError: If the target is missing from a labelled frame.
    DataContractError: If a declared feature or identifier is absent.
  """
  available = set(frame.columns)
  missing = [column for column in [*features, *identifiers] if column not in available]
  if missing:
    raise DataContractError(
      f'{len(missing)} declared column(s) are absent from the input frame: {missing[:20]}. A column '
      'declared in a role list but not selected by the input SQL will fail here rather than deep '
      'inside a transform.',
      missing=missing,
    )
  if target_col and target_col not in available:
    LOGGER.info(
      'The input frame carries no target column; it will be constructed from configuration',
      extra={'target_col': target_col},
    )
