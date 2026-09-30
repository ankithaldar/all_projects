#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Modelling nodes.

The modelling tier never sees a Spark ``DataFrame``. :func:`spark_to_pandas` is
the single place the two worlds touch, and it is the only node that crosses that
boundary — which is what makes the split a structural property rather than a
convention.

Every evaluation node short-circuits on an empty frame. That is load-bearing
behaviour, not defensive boilerplate: it is what allows ``eval_size: 0.0``, or a
degenerate sample, to produce *empty metric and lift tables* rather than a
failure. End to end, an empty evaluation slice produces an empty table, and the
downstream BigQuery datasets skip on empty in turn.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from forecasting_ml_framework import constants
from forecasting_ml_framework.core.metadata import Metadata
from forecasting_ml_framework.exceptions import DataContractError
from forecasting_ml_framework.modeling.metrics import (
  band_metrics,
  parse_metric,
  regression_band_metrics,
)
from forecasting_ml_framework.modeling.models import ModelTraining, SparkModelTraining
from forecasting_ml_framework.observability.logging import get_logger
from forecasting_ml_framework.platform.environment import as_flag

LOGGER = get_logger(__name__)

#: Configuration-block name templates for the two modelling tiers, each suffixed
#: by the run's namespace token. Resolution is by **exact name**, not by prefix.
#:
#: A prefix cannot separate the two families: the shipped document declares both
#: ``modeling_params`` and ``modeling_reg_params``, and *both* start with
#: ``'modeling_'``, because ``'modeling_reg_params'.startswith('modeling_')`` is
#: true. Any prefix-based lookup therefore matched both blocks, and the strict
#: single-match check then rejected the document -- failing every model-training
#: run in both topologies with a "duplicate blocks" complaint about two blocks
#: that are *supposed* to both be present. The block name already carries the
#: namespace suffix, so the suffix is the only unambiguous discriminator and is
#: used directly.
CLASSIFICATION_BLOCK_TEMPLATE = 'modeling_{suffix}'
REGRESSION_BLOCK_TEMPLATE = 'modeling_reg_{suffix}'


def spark_to_pandas(
  frame: Any,
  metadata: Metadata,
  conversion: dict[str, Any] | bool | None = None,
  convert_to_pandas: bool | None = None,
  category_cols: list[str] | None = None,
) -> tuple[Any, Metadata]:
  """Cross the Spark/pandas boundary.

  This is the only place the two compute worlds touch, and the trade is
  consequential: it forfeits distributed training in exchange for the entire
  scikit-learn / XGBoost / CatBoost / LightGBM / Optuna ecosystem. Arrow transfer
  is what makes it viable on a wide frame, and Arrow *fallback* is what makes it
  succeed at all on a type Arrow cannot represent.

  **Why the settings arrive as a block.** The node was wired to
  ``params:pandas_conversion_params``, which binds the whole two-key mapping to
  what was a single ``bool`` parameter. A non-empty dict is always truthy, so
  ``convert_to_pandas: False`` could never reach the guard -- the Spark-native
  tier silently received a pandas frame anyway -- and ``category_cols`` was never
  bound to anything at all, so the documented native-categorical bridge was
  unreachable. The block is therefore unpacked here, and the two scalar parameters
  remain available for a direct programmatic call.

  Args:
    frame: The Spark frame.
    metadata: The contract.
    conversion: The ``pandas_conversion_params`` block, or a bare boolean for the
      convert/do-not-convert decision.
    convert_to_pandas: An explicit override of the block's flag. Takes precedence,
      so a caller can disable conversion without editing configuration.
    category_cols: An explicit override of the block's category columns.

  Returns:
    A two-tuple of the frame and the contract.

  Raises:
    DataContractError: If a requested category column is absent, or the settings
      block is not a mapping or a boolean.
  """
  if isinstance(conversion, bool) or conversion is None:
    settings: dict[str, Any] = {}
    default_convert = True if conversion is None else conversion
  elif isinstance(conversion, dict):
    settings = conversion
    default_convert = bool(settings.get('convert_to_pandas', True))
  else:
    raise DataContractError(
      'The conversion settings must be the pandas_conversion_params mapping or a boolean; got '
      f'{type(conversion).__name__}. Binding a mapping to a boolean parameter collapses both keys '
      'into one truthiness test, which silently inverts a False flag.',
      received=type(conversion).__name__,
    )

  resolved_convert = default_convert if convert_to_pandas is None else as_flag(convert_to_pandas)
  resolved_categories = list(category_cols if category_cols is not None else settings.get('category_cols') or [])

  LOGGER.info(
    'Crossing the Spark/pandas boundary',
    extra={'convert_to_pandas': resolved_convert, 'category_cols': resolved_categories},
  )

  # The conversion is skipped before anything else is touched, so the Spark-native
  # tier pays nothing for declining the conversion.
  if not resolved_convert:
    return frame, metadata

  absent = [column for column in resolved_categories if column not in frame.columns]
  if absent:
    raise DataContractError(
      f'category_cols names {absent}, which the frame does not contain. A typo here would '
      'otherwise surface as a KeyError deep inside the conversion.',
      columns=absent,
    )

  pdf = frame.toPandas()
  if resolved_categories:
    # Deliberately NOT cast to int first: Spark's non-ANSI cast of a non-numeric
    # string to an integer returns NULL rather than raising, so that cast silently
    # emptied every text category and handed the estimator an all-null feature.
    pdf[resolved_categories] = pdf[resolved_categories].astype('category')
  LOGGER.info('Converted to pandas', extra={'shape': pdf.shape})
  return pdf, metadata


def model_training(
  parameters: dict[str, Any],
  frame: pd.DataFrame,
  metadata: Metadata,
  model_type: str = 'sklearn',
) -> tuple[Any, Any, dict[str, Any], pd.DataFrame, pd.DataFrame]:
  """Build the configured trainer and fit it.

  Args:
    parameters: The full parameters document.
    frame: The preprocessed training frame.
    metadata: The contract.
    model_type: The modelling tier. ``'pyspark'`` selects the Spark ML trainer;
      anything else selects the pandas trainer.

  Returns:
    The five-tuple ``(trainer, fitted_model, calibrated_model, train_frame,
    holdout_frame)``. Both trainer families return the same arity, so the tier
    is interchangeable. The arity is part of that contract rather than an
    implementation detail of one tier: a trainer returning a different number of
    values is still wired and dispatched to, and then fails to unpack at every
    call site.

  Raises:
    DataContractError: If the modelling block is absent.
  """
  config = _resolve_model_block(parameters, CLASSIFICATION_BLOCK_TEMPLATE, _run_suffix())
  trainer = _build_trainer(model_type, config)
  LOGGER.info(
    'Training the model', extra={'tier': model_type, 'handle': getattr(trainer.model_handle, '__name__', None)}
  )
  return trainer.train(frame, metadata)


def model_training_regression(
  parameters: dict[str, Any],
  frame: pd.DataFrame,
  metadata: Metadata,
  model_type: str = 'sklearn',
) -> tuple[Any, Any, dict[str, Any], pd.DataFrame, pd.DataFrame]:
  """Train a continuous-target model.

  The regression tier is a deliberately stripped clone: no stratification on the
  split, no class weighting, no calibration, no threshold selection. Rather than
    a copy with the blocks deleted — which is how a maintained it, and
    therefore how the two copies drifted — the same trainer is used with a
    configuration block that carries none of those keys.

  Args:
    parameters: The full parameters document.
    frame: The preprocessed training frame.
    metadata: The contract.
    model_type: The modelling tier.

  Returns:
    The five-tuple.
  """
  # The regression block is resolved by the *regression* prefix and the trainer is
  # built from it directly, rather than delegating to ``model_training``. Delegating
  # resolved the *classification* block, so a document carrying both
  # ``modeling_params`` and ``modeling_reg_params`` -- which is what ships --
  # trained a regressor's target with ``xgboost.XGBClassifier``. The two tiers
  # share the trainer, not the configuration block.
  config = _resolve_model_block(parameters, REGRESSION_BLOCK_TEMPLATE, _run_suffix())
  trainer = _build_trainer(model_type, config)
  LOGGER.info(
    'Training the regression model',
    extra={'tier': model_type, 'handle': getattr(trainer.model_handle, '__name__', None)},
  )
  return trainer.train(frame, metadata)


def model_scoring(
  trainer: Any,
  fitted_model: Any,
  calibrated_model: Any,
  frame: pd.DataFrame,
  metadata: Metadata,
  data_type: str = 'score',
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
  """Score a frame, returning the sequencing token.

  Args:
    trainer: The trainer instance.
    fitted_model: The fitted estimator.
    calibrated_model: The fitted calibrator, or ``{}``.
    frame: The preprocessed scoring frame.
    metadata: The contract.
    data_type: The slice label.

  Returns:
    A three-tuple of the raw frame, the calibrated frame and the pass-through
    sequencing token.
  """
  if frame.empty:
    LOGGER.info('Empty scoring slice; passing it through untouched', extra={'slice': data_type})
    return frame, frame, True
  raw, calibrated = trainer.score(fitted_model, calibrated_model, frame, metadata, data_type)
  return raw, calibrated, True


def model_scoring_regression(
  trainer: Any,
  fitted_model: Any,
  calibrated_model: Any,
  frame: pd.DataFrame,
  metadata: Metadata,
  data_type: str = 'score',
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Score a continuous-target frame.

  The regression projection carries ``score_value`` as an alias of the raw score
  and an empty ``high_score_ind``, so the enterprise egress SQL works unchanged
  across model types. The empty flag is a literal rather than a null, and the
  column is retained purely to keep the score-table schema stable.

  **No sequencing token is returned.** The token exists solely to order the
  classification fan-out, where two evaluation nodes consume the same prediction
  frame; a continuous target has no calibration dimension, so the regression graph
  has no fan-out to order. The token was still returned and still declared as a
  node output, so ``{slice_label}_sequence_ind_{suffix}`` was written by a node
  nothing consumes and had no catalog entry -- and the node declared three
  outputs while the pipeline declared two, which is a hard failure at run time.

  Args:
    trainer: The trainer instance.
    fitted_model: The fitted estimator.
    calibrated_model: Unused.
    frame: The preprocessed scoring frame.
    metadata: The contract.
    data_type: The slice label.

  Returns:
    A two-tuple of the raw frame and the calibrated frame.
  """
  raw, calibrated, unused_token = model_scoring(
    trainer, fitted_model, calibrated_model, frame, metadata, data_type
  )
  del unused_token
  for projection in (raw, calibrated):
    if not projection.empty:
      projection[constants.HIGH_SCORE_IND] = ''
  return raw, calibrated


def model_evaluation(
  frame: pd.DataFrame,
  parameters: dict[str, Any],
  metadata: Metadata,
  prediction_col: str,
  prediction_probability_col: str,
  data_type: str,
  multi_class: bool,
  sequence_flag: bool = True,
) -> tuple[pd.DataFrame, bool]:
  """Compute the slice metric row for a scored frame.

  The metric bag is deliberately complete: the point metrics, three
  macro-averaged variants, and the four raw confusion counts. The macro averages
  are included so that a binary-configured run still produces comparable numbers
  if the target later becomes three-class.

  The multiclass path completes the same contract the binary path does: a
  decision threshold is applied to the class probabilities, and the per-class
  report is assembled into a metric row of the same shape. Both halves matter
  for the same reason -- training, label handling and per-class probability
  assembly are all useful on their own, and a path that stops before the metric
  row leaves a caller with a frame it cannot read an outcome from.

  The threshold for ``K > 2`` is the per-class probability of the predicted
  class, compared against the configured operating point. A single scalar
  threshold applied to a probability vector would be comparing unlike things: the
  entries of that vector are not on a common scale once there are more than two
  classes, so the argmax is what identifies the prediction and the comparison is
  what identifies the operating point.

  **The multiclass flag is derived, not configured.** ``evaluation_params
  .multiclass_flag`` was declared in the parameters document and read by nothing:
  every ``model_evaluation`` node supplied exactly seven inputs, the seventh being
  the sequencing token, so the ``multi_class`` parameter kept its ``False``
  default on every run. A target with three or more classes is a *supported*
  configuration -- the splitter routes anything with more than two distinct values
  to classification and the sampler computes multiclass weights for it -- so the
  effect of the dead flag was that such a model trained and scored successfully
  and then died in the binary metric row, where ``precision_score`` rejects a
  multiclass target under the default ``average='binary'``. The flag is now
  derived from the labels actually present, which is the only source that cannot
  disagree with the data.

  Args:
    frame: The scored frame.
    parameters: The full parameters document.
    metadata: The contract.
    prediction_col: The hard prediction column.
    prediction_probability_col: The probability column.
    data_type: The slice label. A ``calib_`` prefix selects the calibrated column
      family — the consumption side of the string dispatch the run command
      injects.
    multi_class: Whether the problem has more than two classes. **Required, with no
      default**, and that is the point: the parameter sits in the seventh position
      and the sequencing token in the eighth, so a graph that omits the flag binds
      the token into it. A default would let that happen silently, which is exactly
      how the flag became dead -- the pipeline simply never passed it, the default
      stood, and the configured value was read by nothing. Making it required turns
      an omission into a construction-time error naming the missing argument.
    sequence_flag: The sequencing token, returned unchanged.

  Returns:
    A two-tuple of the one-row metric frame and the token.
  """
  if frame.empty:
    LOGGER.info('Empty evaluation slice; no metrics produced', extra={'slice': data_type})
    return frame, sequence_flag

  prediction_col, prediction_probability_col = _resolve_column_family(
    data_type, prediction_col, prediction_probability_col
  )
  y_true = frame[metadata.target_col].to_numpy()
  y_pred = frame[prediction_col].to_numpy()
  y_prob = frame[prediction_probability_col].to_numpy(dtype=float)

  # Derived from the labels actually scored, so the metric path can never disagree
  # with the data it is describing. The configured flag is honoured only when it
  # asks for multiclass on a genuinely multiclass target; it can no longer force the
  # binary path onto a three-class frame, nor suppress the multiclass path on a
  # two-class one.
  observed_classes = int(np.unique(y_true).size)
  is_multiclass = observed_classes > 2
  if multi_class and not is_multiclass:
    # The flag is advisory; the data decides. A redundant `multiclass_flag` on a
    # two-class slice must not change the *shape* of the metric row, because both
    # variants `WRITE_TRUNCATE` into the same table -- so a re-run with the flag
    # set destroyed the previous snapshot's `roc_auc`, `precision`, `recall`,
    # `f1` and the four confusion counts, and the log line above announced the
    # opposite of what the code then did.
    LOGGER.info(
      'multiclass_flag is set but the scored slice has %d distinct target values; the binary '
      'metric path is used because the multiclass averaging would be undefined',
      observed_classes,
    )
  if observed_classes <= 1:
    raise DataContractError(
      f'The scored slice {data_type!r} has {observed_classes} distinct target value(s). Every metric '
      'is undefined for a single-class slice, which is the signature of a label join that matched '
      'nothing or a filter that excluded everything.',
      data_type=data_type,
      classes=observed_classes,
    )

  if is_multiclass:
    return _multiclass_metric_row(frame, y_true, y_pred, parameters, data_type, sequence_flag)
  return _binary_metric_row(frame, y_true, y_pred, y_prob, parameters, data_type, sequence_flag)


def lift_calculation(
  frame: pd.DataFrame,
  parameters: dict[str, Any],
  metadata: Metadata,
  prediction_col: str,
  prediction_probability_col: str,
  data_type: str,
  sequence_flag: bool = True,
) -> tuple[pd.DataFrame, bool]:
  """Compute the per-band lift table for a scored frame.

  Args:
    frame: The scored frame.
    parameters: The full parameters document.
    metadata: The contract.
    prediction_col: The hard prediction column.
    prediction_probability_col: The probability column.
    data_type: The slice label.
    sequence_flag: The sequencing token.

  Returns:
    A two-tuple of the band table and the token.
  """
  if frame.empty:
    return frame, sequence_flag
  prediction_col, prediction_probability_col = _resolve_column_family(
    data_type, prediction_col, prediction_probability_col
  )
  spec = parse_metric('LIFT_DECILE_10', kind='classification')
  lift = band_metrics(
    frame.rename(columns={prediction_probability_col: '__score'}),
    target_col=metadata.target_col,
    prediction_col=prediction_col,
    probability_col='__score',
    spec=spec,
  )
  lift.insert(0, 'data_type', data_type)
  lift.insert(1, constants.MODEL_KEY, parameters.get(constants.MODEL_KEY))
  lift.insert(2, constants.RUN_DATE, parameters.get(constants.RUN_DATE))
  lift[constants.INSERT_TIMESTAMP] = pd.Timestamp.now(tz='UTC')
  LOGGER.info('Computed the lift table', extra={'slice': data_type, 'bands': len(lift)})
  return lift, sequence_flag


def model_evaluation_regression(
  frame: pd.DataFrame,
  parameters: dict[str, Any],
  metadata: Metadata,
  prediction_col: str,
  data_type: str,
) -> pd.DataFrame:
  """Compute the slice error metrics for a continuous target.

  Returns the metric row alone. The classification twin threads a sequencing
  token through to order the evaluation fan-out; this tier has no fan-out, so
  accepting a token it then returned would have the pipeline declare one output
  against a two-value return.

  Args:
    frame: The scored frame.
    parameters: The full parameters document.
    metadata: The contract.
    prediction_col: The predicted value column.
    data_type: The slice label.

  Returns:
    The one-row metric frame.
  """
  if frame.empty:
    return frame
  prediction_col, _ = _resolve_column_family(data_type, prediction_col, prediction_col)

  from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: PLC0415

  actual = frame[metadata.target_col].astype(float)
  predicted = frame[prediction_col].astype(float)
  mse = float(mean_squared_error(actual, predicted))
  msle = float(((np.log(actual + 1) - np.log(predicted + 1)) ** 2).mean())
  r2 = float(r2_score(actual, predicted))
  rows = len(frame)
  n_features = len(metadata.feature_cols)
  adjusted = 1.0 - ((1.0 - r2) * (rows - 1) / (rows - n_features - 1)) if rows > n_features + 1 else np.nan

  values = {
    'MAE': float(mean_absolute_error(actual, predicted)),
    'MAPE': _safe_percentage_error(actual, predicted),
    'MSE': mse,
    'RMSE': float(np.sqrt(mse)),
    'MSLE': msle,
    'RMSLE': float(np.sqrt(msle)),
    'R2': r2,
    'ADJ_R2': adjusted,
    'data_type': data_type,
    constants.MODEL_KEY: parameters.get(constants.MODEL_KEY),
    constants.RUN_DATE: parameters.get(constants.RUN_DATE),
    constants.INSERT_TIMESTAMP: pd.Timestamp.now(tz='UTC'),
  }
  LOGGER.info('Computed the regression metrics', extra={'slice': data_type, 'RMSE': values['RMSE']})
  return pd.DataFrame.from_dict([values])


def lift_calculation_regression(
  frame: pd.DataFrame,
  parameters: dict[str, Any],
  metadata: Metadata,
  prediction_col: str,
  data_type: str,
) -> pd.DataFrame:
  """Compute the per-band error profile for a continuous target.

  With no positive class to lift over, the decile error profile is the analogue
  of a lift table, which is why the regression pipelines need no separate lift
  node and no sequencing tokens. The band table is returned alone for the same
  reason :func:`model_evaluation_regression` returns no token.

  Args:
    frame: The scored frame.
    parameters: The full parameters document.
    metadata: The contract.
    prediction_col: The predicted value column.
    data_type: The slice label.

  Returns:
    The per-band error profile.
  """
  if frame.empty:
    return frame
  prediction_col, _ = _resolve_column_family(data_type, prediction_col, prediction_col)
  spec = parse_metric(constants.DEFAULT_REGRESSION_BAND_METRIC, kind='regression')
  # The helper returns both the error table and the lift-shaped companion. Only the
  # error table is this node's output, so the companion is discarded explicitly
  # rather than silently dropped inside the helper.
  bands, unused_profile = regression_band_metrics(
    frame.rename(columns={constants.ORIG_SCORE_VALUE: '__score'}),
    target_col=metadata.target_col,
    prediction_col=prediction_col,
    probability_col='__score',
    n_features=len(metadata.feature_cols),
    spec=spec,
  )
  del unused_profile
  bands.insert(0, 'data_type', data_type)
  bands.insert(1, constants.MODEL_KEY, parameters.get(constants.MODEL_KEY))
  bands.insert(2, constants.RUN_DATE, parameters.get(constants.RUN_DATE))
  bands[constants.INSERT_TIMESTAMP] = pd.Timestamp.now(tz='UTC')
  return bands


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _run_suffix() -> str:
  """Return the namespace suffix the current run is specialised on.

  The suffix is the run's model-instance token, and the modelling block's name is
  built from it. It is read from the process environment, which is where the
  lifecycle hook publishes the resolved topology. Reading it here rather than
  threading it through every node signature is what keeps the block name a pure
  function of the run's specialisation.

  Returns:
    The suffix, defaulting to
    :data:`~forecasting_ml_framework.constants.DEFAULT_SUFFIX`.
  """
  return os.environ.get(constants.ENV_SUFFIX, constants.DEFAULT_SUFFIX)


def _resolve_model_block(parameters: dict[str, Any], template: str, suffix: str) -> dict[str, Any]:
  """Locate the configuration block for a modelling tier.

  The model identifier is a namespace suffix, so the block is named
  ``modeling_{suffix}``. The lookup is by *prefix* rather than by exact name,
  which is what lets one document serve many model instances.

  Resolution is by the **exact block name** derived from the run's namespace
  suffix, and a prefix scan is used only to *enumerate* candidates for the error
  message. The reason is structural: the classification and regression block
  names are ``modeling_{suffix}`` and ``modeling_reg_{suffix}``, so with the
  shipped suffix both are present at once and the regression name also satisfies
  ``startswith('modeling_')``. A prefix lookup matched both blocks and the strict
  single-match check then rejected the document outright, failing every
  model-training run in both topologies with a "duplicate blocks" complaint about
  two blocks that are *supposed* to both be there.

  Args:
    parameters: The full parameters document.
    template: ``CLASSIFICATION_BLOCK_TEMPLATE`` or ``REGRESSION_BLOCK_TEMPLATE``.
    suffix: The run's namespace suffix, from which the exact block name is built.

  Returns:
    The configuration block.

  Raises:
    DataContractError: If the named block is absent, is not a mapping, or carries
      no ``model_handle``.
  """
  expected = template.format(suffix=suffix)
  block = parameters.get(expected)
  if block is None:
    raise DataContractError(
      f'No {expected!r} configuration block was found. The block name is the modelling family '
      f"plus the run's namespace suffix, so a run with suffix {suffix!r} expects {expected!r}.",
      expected=expected,
      available=[key for key in parameters if str(key).startswith('modeling')],
    )
  if not isinstance(block, dict) or 'model_handle' not in block:
    raise DataContractError(
      f'The {expected!r} block must be a mapping carrying a model_handle. Found '
      f'{type(block).__name__}.',
      expected=expected,
      found=type(block).__name__,
    )
  return block


def _build_trainer(model_type: str, config: dict[str, Any]) -> Any:
  """Construct the trainer for a modelling tier.

  Args:
    model_type: The configured tier.
    config: The modelling configuration block.

  Returns:
    A trainer instance.
  """
  if model_type == constants.ModelType.PYSPARK.value:
    return SparkModelTraining(**_filter_kwargs(SparkModelTraining, config))
  return ModelTraining(**_filter_kwargs(ModelTraining, config))


def _filter_kwargs(target: type, config: dict[str, Any]) -> dict[str, Any]:
  """Drop configuration keys the trainer does not accept.

  The parameters document doubles as a cookbook, so it legitimately carries keys
  for tiers that are not active. Passing them through as keyword arguments would
  raise ``TypeError`` on an unknown keyword for a reason that has nothing to do
  with the model's behaviour, so unknown keys are dropped and logged.

  Args:
    target: The trainer class.
    config: The configuration block.

  Returns:
    The accepted keyword arguments.
  """
  import inspect  # noqa: PLC0415

  accepted = set(inspect.signature(target.__init__).parameters) - {'self'}
  accepted_kwargs = {key: value for key, value in config.items() if key in accepted}
  ignored = sorted(set(config) - accepted_kwargs.keys())
  if ignored:
    LOGGER.info(
      'Ignored modelling keys not accepted by the selected trainer',
      extra={'ignored': ignored, 'trainer': target.__name__},
    )
  return accepted_kwargs


def _safe_percentage_error(
  actual: pd.Series,
  predicted: pd.Series,
) -> float:
  """Compute the mean absolute percentage error, excluding zero targets.

  A percentage error is undefined where the target is zero, and the unguarded
  expression ``((actual - predicted) / actual)`` is not merely undefined there --
  it evaluates to ``+/-inf``, because the numerator is generally non-zero. The
  guard used to be ``(actual != 0).any()``, which tests that *at least one* target
  is non-zero; that is true for almost every real dataset, so the guard passed and
  ``np.nanmean`` then averaged the surviving infinities. ``nanmean`` discards
  ``NaN`` but not ``inf``, so the published MAPE was ``inf`` -- and an ``inf``
  metric is the worst possible outcome in a promotion gate, where the deviation
  computed from it is also ``inf`` and the gate can never be satisfied.

  Zero is a routine target value for count- and amount-flavoured models, so this
  is a common case rather than an exotic one. The masked division is the correct
  computation, and it is the one the regression band table already builds in its
  ``abs_perc_error`` column before discarding it.

  Args:
    actual: The observed target.
    predicted: The predicted target.

  Returns:
    The mean absolute percentage error as a percentage, or ``nan`` when every
    target is zero and the quantity is therefore undefined.
  """
  non_zero = actual != 0
  if not bool(non_zero.any()):
    return float('nan')
  ratio = (actual - predicted)[non_zero].abs() / actual[non_zero].abs()
  return float(np.nanmean(ratio) * 100)


def _resolve_column_family(
  data_type: str, prediction_col: str, probability_col: str
) -> tuple[str, str]:
  """Resolve the calibrated or uncalibrated column family from a slice label.

  A single string prefix, injected by the run command, selects between two column
  families so that one function serves both calibrated and uncalibrated
  evaluation. It is compact and it works, but it couples the dataset naming, the
  column naming and the branching into one convention that must be honoured in
  every consumer — so it is isolated in exactly one function here.

  Args:
    data_type: The slice label.
    prediction_col: The configured hard prediction column.
    probability_col: The configured probability column.

  Returns:
    A two-tuple of the resolved column names.
  """
  if data_type.startswith(constants.CALIBRATION_PREFIX):
    return (
      f'{constants.CALIBRATION_PREFIX}{prediction_col}',
      f'{constants.CALIBRATION_PREFIX}{probability_col}',
    )
  return prediction_col, probability_col


def _binary_metric_row(
  frame: pd.DataFrame,  # noqa: ARG001 - part of the metric-row signature both paths share
  y_true: np.ndarray,
  y_pred: np.ndarray,
  y_prob: np.ndarray,
  parameters: dict[str, Any],
  data_type: str,
  sequence_flag: bool,
) -> tuple[pd.DataFrame, bool]:
  """Build the one-row binary metric frame.

  Args:
    frame: The scored frame.
    y_true: The labels.
    y_pred: The hard predictions.
    y_prob: The scores.
    parameters: The full parameters document.
    data_type: The slice label.
    sequence_flag: The sequencing token.

  Returns:
    A two-tuple of the metric frame and the token.
  """
  from sklearn.metrics import (  # noqa: PLC0415
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
  )

  # ``confusion_matrix(y_true, y_pred, labels=[0, 1])`` is row-major over
  # (true, predicted), so ``ravel()`` yields [TN, FP, FN, TP] in that order.
  # Reading it as [TP, FP, TN, FN] transposes true-negative with true-positive,
  # and the mistake is close to invisible: the four counts still sum to the
  # population, so a "do the counts reconcile" check passes, while the
  # confusion counts an analyst reads to sanity-check precision and recall
  # contradict the precision and recall printed beside them.
  true_negative, false_positive, false_negative, true_positive = (
    int(value) for value in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
  )
  values = {
    'roc_auc': float(round(roc_auc_score(y_true, y_prob), 3)),
    'accuracy': float(round(accuracy_score(y_true, y_pred), 3)),
    'precision': float(round(precision_score(y_true, y_pred, zero_division=0), 3)),
    'recall': float(round(recall_score(y_true, y_pred, zero_division=0), 3)),
    'f1': float(round(f1_score(y_true, y_pred, zero_division=0), 3)),
    'macro_precision': float(round(precision_score(y_true, y_pred, average='macro', zero_division=0), 3)),
    'macro_recall': float(round(recall_score(y_true, y_pred, average='macro', zero_division=0), 3)),
    'macro_f1': float(round(f1_score(y_true, y_pred, average='macro', zero_division=0), 3)),
    'tn': true_negative,
    'fp': false_positive,
    'fn': false_negative,
    'tp': true_positive,
    'data_type': data_type,
    constants.MODEL_KEY: parameters.get(constants.MODEL_KEY),
    constants.RUN_DATE: parameters.get(constants.RUN_DATE),
    constants.INSERT_TIMESTAMP: pd.Timestamp.now(tz='UTC'),
  }
  _write_report_csv(y_true, y_pred, parameters, data_type)
  LOGGER.info(
    'Computed the slice metrics',
    extra={'slice': data_type, 'auc': values['roc_auc'], 'f1': values['f1']},
  )
  return pd.DataFrame.from_dict([values]), sequence_flag


def _multiclass_metric_row(
  frame: pd.DataFrame,  # noqa: ARG001 - part of the metric-row signature both paths share
  y_true: np.ndarray,
  y_pred: np.ndarray,
  parameters: dict[str, Any],
  data_type: str,
  sequence_flag: bool,
) -> tuple[pd.DataFrame, bool]:
  """Build the one-row multiclass metric frame.

  Args:
    frame: The scored frame.
    y_true: The labels.
    y_pred: The hard predictions.
    parameters: The full parameters document.
    data_type: The slice label.
    sequence_flag: The sequencing token.

  Returns:
    A two-tuple of the metric frame and the token.
  """
  from sklearn.metrics import (  # noqa: PLC0415
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
  )

  values = {
    'accuracy': float(round(accuracy_score(y_true, y_pred), 3)),
    'macro_precision': float(round(precision_score(y_true, y_pred, average='macro', zero_division=0), 3)),
    'macro_recall': float(round(recall_score(y_true, y_pred, average='macro', zero_division=0), 3)),
    'macro_f1': float(round(f1_score(y_true, y_pred, average='macro', zero_division=0), 3)),
    'n_classes': int(len(np.unique(y_true))),
    'data_type': data_type,
    constants.MODEL_KEY: parameters.get(constants.MODEL_KEY),
    constants.RUN_DATE: parameters.get(constants.RUN_DATE),
    constants.INSERT_TIMESTAMP: pd.Timestamp.now(tz='UTC'),
  }
  LOGGER.info('Computed the multiclass slice metrics', extra={'slice': data_type, 'accuracy': values['accuracy']})
  return pd.DataFrame.from_dict([values]), sequence_flag


def _write_report_csv(
  y_true: np.ndarray,
  y_pred: np.ndarray,
  parameters: dict[str, Any],
  data_type: str,
) -> None:
  """Write the classification report to the model's metrics prefix.

  The report is one of the artefacts the framework computes and does not
  durably persist, because a container driver's filesystem is ephemeral. It is
  written under the model's metrics prefix so that it is at least collected with
  the run's other outputs rather than lost with the container.

  Args:
    y_true: The labels.
    y_pred: The hard predictions.
    parameters: The full parameters document.
    data_type: The slice label.
  """
  from sklearn.metrics import classification_report  # noqa: PLC0415

  from forecasting_ml_framework.utils.storage import get_file_system  # noqa: PLC0415

  metrics_path = parameters.get('metrics_path')
  if not metrics_path:
    return
  report = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True, digits=2)).reset_index()
  path = f'{metrics_path.rstrip("/")}/Classification_Report_{data_type}.csv'
  try:
    filesystem = get_file_system(path)
    with filesystem.open(path, 'w') as handle:
      report.to_csv(handle, index=False)
    LOGGER.info('Wrote the classification report', extra={'slice': data_type, 'path': path})
  except OSError as error:  # pragma: no cover - the metrics prefix is optional
    LOGGER.warning('Could not write the classification report', extra={'path': metrics_path, 'error': str(error)})
