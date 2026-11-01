#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Lift contract: per-band lift evaluation using forecasting naming.

This module restructures the forecasting lift calculation layer into a
standalone contract, preserving the forecasting layer's 2-tuple arity
(lift table plus sequencing token) and using forecasting constants for
model identity and provenance.
'''

from __future__ import annotations

from typing import Any

import pandas as pd

from forecasting_ml_framework import constants
from forecasting_ml_framework.core.metadata import Metadata
from forecasting_ml_framework.modeling.metrics import band_metrics, parse_metric, regression_band_metrics
from forecasting_ml_framework.observability.logging import get_logger

LOGGER = get_logger(__name__)


def lift_contract_evaluation(
  frame: pd.DataFrame,
  parameters: dict[str, Any],
  metadata: Metadata,
  prediction_col: str,
  prediction_probability_col: str,
  data_type: str,
  sequence_flag: bool = True,
) -> tuple[pd.DataFrame, bool]:
  '''Compute the per-band lift table for a scored frame.

  The contract uses forecasting layer naming and inserts forecasting
  constants (``MODEL_KEY``, ``RUN_DATE``, ``INSERT_TIMESTAMP``) as the
  provenance columns required by the downstream warehouse contract.

  Args:
    frame: The scored frame.
    parameters: The full parameters document.
    metadata: The contract.
    prediction_col: The hard prediction column.
    prediction_probability_col: The probability column.
    data_type: The slice label.
    sequence_flag: The sequencing token, returned unchanged.

  Returns:
    A two-tuple of the band table and the token.
  '''
  if frame.empty:
    LOGGER.info('Empty lift evaluation slice; passing through', extra={'slice': data_type})
    return frame, sequence_flag

  prediction_col, prediction_probability_col = _resolve_column_family_contract(
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
  LOGGER.info('Computed the lift contract', extra={'slice': data_type, 'bands': len(lift)})
  return lift, sequence_flag


def regression_lift_contract_evaluation(
  frame: pd.DataFrame,
  parameters: dict[str, Any],
  metadata: Metadata,
  prediction_col: str,
  data_type: str,
) -> pd.DataFrame:
  '''Compute the per-band error profile for a continuous target.

  A continuous target has no positive class to lift over, so the decile
  error profile is the analogue of a lift table. This contract uses the
  forecasting regression band metric (``DEFAULT_REGRESSION_BAND_METRIC``)
  rather than a hard-coded literal.

  Args:
    frame: The scored frame.
    parameters: The full parameters document.
    metadata: The contract.
    prediction_col: The predicted value column.
    data_type: The slice label.

  Returns:
    The per-band error profile frame.
  '''
  if frame.empty:
    return frame
  prediction_col, _ = _resolve_column_family_contract(data_type, prediction_col, prediction_col)
  spec = parse_metric(constants.DEFAULT_REGRESSION_BAND_METRIC, kind='regression')
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
  LOGGER.info('Computed the regression lift contract', extra={'slice': data_type, 'bands': len(bands)})
  return bands


def _resolve_column_family_contract(
  data_type: str,
  prediction_col: str,
  probability_col: str,
) -> tuple[str, str]:
  '''Resolve the calibrated or uncalibrated column family.

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
