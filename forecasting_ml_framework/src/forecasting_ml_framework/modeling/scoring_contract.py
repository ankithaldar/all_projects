#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=import-outside-toplevel,broad-exception-caught,unused-argument,unexpected-keyword-arg,invalid-name,inconsistent-quotes

'''Scoring contract: classification and regression metric evaluation.

This module restructures the forecasting metric comparison layer into a
standalone contract layer, using forecasting constants and layer naming.
Every public function preserves the forecasting arity: 2-tuple for
classification evaluation (metric row plus sequencing token), single
DataFrame for regression evaluation.
'''

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import pandas as pd

from forecasting_ml_framework import constants
from forecasting_ml_framework.core.metadata import Metadata
from forecasting_ml_framework.exceptions import DataContractError, MetricBandEmptyError
from forecasting_ml_framework.modeling.metrics import (
  band_metrics,
  parse_metric,
  select_band,
)
from forecasting_ml_framework.observability.logging import get_logger

LOGGER = get_logger(__name__)


def classification_scoring_contract(
  frame: pd.DataFrame,
  parameters: dict[str, Any],
  metadata: Metadata,
  prediction_col: str,
  prediction_probability_col: str,
  data_type: str,
  sequence_flag: bool = True,
) -> tuple[pd.DataFrame, bool]:
  '''Compute the classification metric evaluation row.

  The metric bag is complete: point metrics, three macro-averaged variants,
  and the raw confusion counts. The multiclass path produces the same row
  shape as the binary path, because both families write to the same
  evaluation table.

  Args:
    frame: The scored frame.
    parameters: The full parameters document.
    metadata: The contract.
    prediction_col: The hard prediction column.
    prediction_probability_col: The probability column.
    data_type: The slice label. A ``calib_`` prefix selects the calibrated
      column family via :func:`_resolve_column_family`.
    sequence_flag: The sequencing token, returned unchanged.

  Returns:
    A two-tuple of the one-row metric frame and the token.

  Raises:
    DataContractError: When the scored slice has fewer than two distinct
      target values, because every metric is undefined for a single-class
      slice.
  '''
# pylint: disable=import-outside-toplevel
  if frame.empty:
    LOGGER.info('Empty evaluation slice; no metrics produced', extra={'slice': data_type})
    return frame, sequence_flag

  prediction_col, prediction_probability_col = _resolve_column_family(
    data_type, prediction_col, prediction_probability_col
  )
  y_true = frame[metadata.target_col].to_numpy()
  y_pred = frame[prediction_col].to_numpy()
  y_prob = frame[prediction_probability_col].to_numpy(dtype=float)

  observed_classes = int(np.unique(y_true).size)
  is_multiclass = observed_classes > 2

  if observed_classes <= 1:
    raise DataContractError(
      f'The scored slice {data_type!r} has {observed_classes} distinct target value(s). '
      'Every metric is undefined for a single-class slice.',
      data_type=data_type,
      classes=observed_classes,
    )

  if is_multiclass:
    return _multiclass_scoring_contract(y_true, y_pred, parameters, data_type, sequence_flag)

  metric_frame, token = _binary_scoring_contract(y_true, y_pred, y_prob, parameters, data_type, sequence_flag)
  if not metric_frame.empty:
    _add_selected_bucket_contract(metric_frame, frame, metadata, prediction_col, prediction_probability_col, parameters)
  return metric_frame, token


def regression_scoring_contract(
  frame: pd.DataFrame,
  parameters: dict[str, Any],
  metadata: Metadata,
  prediction_col: str,
  data_type: str,
) -> pd.DataFrame:
  '''Compute the regression metric evaluation row.

  A continuous target has no sequencing token: the regression graph has no
  evaluation fan-out to order, so returning a token would declare one output
  against a zero-value return in the pipeline binding.

  Args:
    frame: The scored frame.
    parameters: The full parameters document.
    metadata: The contract.
    prediction_col: The predicted value column.
    data_type: The slice label.

  Returns:
    The one-row metric frame.
  '''
  if frame.empty:
    return frame
  prediction_col, _ = _resolve_column_family(data_type, prediction_col, prediction_col)

  from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: PLC0415

  actual = frame[metadata.target_col].astype(float)
  predicted = frame[prediction_col].astype(float)
  mse = float(mean_squared_error(actual, predicted))
  msle_value = float(((np.log(actual + 1) - np.log(predicted + 1)) ** 2).mean())
  r2 = float(r2_score(actual, predicted))
  rows = len(frame)
  n_features = len(metadata.feature_cols)
  adjusted_r2 = 1.0 - ((1.0 - r2) * (rows - 1) / (rows - n_features - 1)) if rows > n_features + 1 else np.nan

  values = {
    'MAE': float(mean_absolute_error(actual, predicted)),
    'MAPE': _safe_percentage_error_contract(actual, predicted),
    'MSE': mse,
    'RMSE': float(np.sqrt(mse)),
    'MSLE': msle_value,
    'RMSLE': float(np.sqrt(msle_value)),
    'R2': r2,
    'ADJ_R2': adjusted_r2,
    'data_type': data_type,
    constants.MODEL_KEY: parameters.get(constants.MODEL_KEY),
    constants.RUN_DATE: parameters.get(constants.RUN_DATE),
    constants.INSERT_TIMESTAMP: pd.Timestamp.now(tz='UTC'),
  }
  LOGGER.info('Computed the regression scoring contract', extra={'slice': data_type, 'RMSE': values['RMSE']})
  return pd.DataFrame.from_dict([values])


# --------------------------------------------------------------------------- #
# Internal scoring-contract helpers
# --------------------------------------------------------------------------- #


def _binary_scoring_contract(
  y_true: np.ndarray,
  y_pred: np.ndarray,
  y_prob: np.ndarray,
  parameters: dict[str, Any],
  data_type: str,
  sequence_flag: bool,
) -> tuple[pd.DataFrame, bool]:
  '''Build the one-row binary metric frame.

  Args:
    y_true: The true labels.
    y_pred: The hard predictions.
    y_prob: The probability scores.
    parameters: The full parameters document.
    data_type: The slice label.
    sequence_flag: The sequencing token.

  Returns:
    A two-tuple of the metric frame and the token.
  '''
  from sklearn.metrics import (  # noqa: PLC0415
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
  )

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
  LOGGER.info(
    'Computed the binary scoring contract',
    extra={'slice': data_type, 'auc': values['roc_auc'], 'f1': values['f1']},
  )
  return pd.DataFrame.from_dict([values]), sequence_flag


def _multiclass_scoring_contract(
  y_true: np.ndarray,
  y_pred: np.ndarray,
  parameters: dict[str, Any],
  data_type: str,
  sequence_flag: bool,
) -> tuple[pd.DataFrame, bool]:
  '''Build the one-row multiclass metric frame.

  Args:
    y_true: The true labels.
    y_pred: The hard predictions.
    parameters: The full parameters document.
    data_type: The slice label.
    sequence_flag: The sequencing token.

  Returns:
    A two-tuple of the metric frame and the token.
  '''
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
  LOGGER.info(
    'Computed the multiclass scoring contract',
    extra={'slice': data_type, 'accuracy': values['accuracy']},
  )
  return pd.DataFrame.from_dict([values]), sequence_flag


def _add_selected_bucket_contract(
  metric_frame: pd.DataFrame,
  frame: pd.DataFrame,
  metadata: Metadata,
  prediction_col: str,
  prediction_probability_col: str,
  parameters: dict[str, Any],
) -> None:
  '''Emit the selected-bucket metric vector into ``metric_frame``.

  The promotion gate selects a single score band and reads one scalar from
  it. The evaluation table carries the full metric vector for that same
  band, so a reviewer can see how the selected segment behaves beyond the
  promoted scalar.

  Args:
    metric_frame: The one-row metric frame, mutated in place.
    frame: The scored frame the metrics describe.
    metadata: The contract, used to name the target column.
    prediction_col: The hard prediction column.
    prediction_probability_col: The score column.
    parameters: The full parameters document.
  '''
  if frame.empty:
    return
  target_col = metadata.target_col
  spec = parse_metric(_eval_metric_name_contract(parameters), kind='classification', strict=False)
  try:
    bands = band_metrics(
      frame.copy(),
      target_col=target_col,
      prediction_col=prediction_col,
      probability_col=prediction_probability_col,
      spec=spec,
    )
  except Exception:  # noqa: BLE001 - degenerate slice must not break evaluation
    LOGGER.debug('Could not compute the selected-bucket contract', exc_info=True)
    return
  if bands.empty:
    return
  try:
    selected = select_band(bands, spec)
  except MetricBandEmptyError:
    selected = bands[bands['score_ntile'] == bands['score_ntile'].max()]
  if selected.empty:
    selected = bands.iloc[[0]]
  row = selected.iloc[0]

  for column in bands.columns:
    if column == 'score_ntile':
      continue
    value = row[column]
    with contextlib.suppress(AttributeError, ValueError):
      value = value.item() if hasattr(value, 'item') else value
    metric_frame[column] = value

  metric_aliases = {
    'AUC': ('selected_auc', 'selected_roc_auc'),
    'PRECISION': ('selected_precision',),
    'RECALL': ('selected_recall',),
    'ACCURACY': ('selected_accuracy',),
    'F1': ('selected_f1',),
    'BRIER_SCORE': ('selected_brier_score',),
    'LIFT': ('selected_lift',),
  }
  for source, aliases in metric_aliases.items():
    if source not in row.index:
      continue
    value = row[source]
    with contextlib.suppress(AttributeError, ValueError):
      value = value.item() if hasattr(value, 'item') else value
    for alias in aliases:
      metric_frame[alias] = value
    metric_frame[f'selected_{source}'] = value


def _safe_percentage_error_contract(
  actual: pd.Series,
  predicted: pd.Series,
) -> float:
  '''Compute the mean absolute percentage error, excluding zero targets.

  Args:
    actual: The observed target.
    predicted: The predicted target.

  Returns:
    The mean absolute percentage error as a percentage, or ``nan`` when
    every target is zero.
  '''
  non_zero = actual != 0
  if not bool(non_zero.any()):
    return float('nan')
  ratio = (actual - predicted)[non_zero].abs() / actual[non_zero].abs()
  return float(np.nanmean(ratio) * 100)


def _eval_metric_name_contract(parameters: dict[str, Any]) -> str | None:
  '''Resolve the configured promotion metric name, if any.

  Args:
    parameters: The full parameters document.

  Returns:
    The configured metric name, or ``None``.
  '''
  if not isinstance(parameters, dict):
    return None
  block = parameters.get('modeling_params')
  if isinstance(block, dict) and block.get('eval_metric') is not None:
    return block.get('eval_metric')
  if parameters.get('eval_metric') is not None:
    return parameters.get('eval_metric')
  return None


def _resolve_column_family(
  data_type: str,
  prediction_col: str,
  probability_col: str,
) -> tuple[str, str]:
  '''Resolve the calibrated or uncalibrated column family from a slice label.

  Args:
    data_type: The slice label.
    prediction_col: The configured hard prediction column.
    probability_col: The configured probability column.

  Returns:
    A two-tuple of the resolved column names.
  '''
  if data_type.startswith(constants.CALIBRATION_PREFIX):
    return (
      f'{constants.CALIBRATION_PREFIX}{prediction_col}',
      f'{constants.CALIBRATION_PREFIX}{probability_col}',
    )
  return prediction_col, probability_col
