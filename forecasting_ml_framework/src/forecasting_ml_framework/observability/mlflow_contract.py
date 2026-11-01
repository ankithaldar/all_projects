#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=import-outside-toplevel,broad-exception-caught,line-too-long

'''Forecasting MLflow contract layer.

Restructures the forecasting mlflow_domino_api.py integration logic
through forecasting layer naming, forecasting constants, and SOLID
contract design. Every model identity, run date, and timestamp flows
through forecasting constants.
The contract separates the forecasting tracking abstraction from the
concrete MLflow implementation via a single-responsibility interface.
'''

from __future__ import annotations

import os
from typing import Any

from forecasting_ml_framework import constants
from forecasting_ml_framework.observability.logging import get_logger

LOGGER = get_logger(__name__)

# Forecasting identity and provenance constants embedded at module level.
MODEL_KEY = constants.MODEL_KEY
RUN_DATE = constants.RUN_DATE
INSERT_TIMESTAMP = constants.INSERT_TIMESTAMP

# Forecasting environment bindings mapped through forecasting layer naming.
FORECASTING_TRACKING_URI_VAR = 'FORECASTING_ML_TRACKING_URI'
FORECASTING_PLATFORM_RUN_VAR = 'FORECASTING_PLATFORM_RUN_ID'
FORECASTING_PLATFORM_API_VAR = 'FORECASTING_PLATFORM_USER_API_KEY'

# Forecasting default artifact path for model logging.
FORECASTING_ARTIFACT_PATH = 'forecasting-core-model'


class ForecastingMLflowContract:
  '''Abstract forecasting MLflow tracking contract (SOLID interface).

  The contract defines the forecasting-layer boundary for experiment
  tracking, model registration, and provenance logging. Concrete
  implementations supply the forecasting-constant identity and the
  forecasting naming vocabulary without exposing the underlying
  tracking client directly.
  '''

  def is_configured(self) -> bool:
    '''Report whether forecasting tracking is configured.

    Returns:
      True when the forecasting tracking URI environment variable is
      present and non-empty.
    '''
    raise NotImplementedError('Forecasting contract requires a concrete implementation.')

  def resolve_experiment(self, forecasting_model_key: str) -> Any:
    '''Resolve or create the forecasting experiment for a model key.

    Args:
      forecasting_model_key: The forecasting model identity constant.

    Returns:
      The forecasting experiment entity.
    '''
    raise NotImplementedError('Forecasting contract requires a concrete implementation.')

  def log_forecasting_params(self, params: dict[str, Any] | None) -> None:
    '''Log forecasting estimator parameters.

    Args:
      params: The forecasting parameter mapping, or None.
    '''
    raise NotImplementedError('Forecasting contract requires a concrete implementation.')

  def log_forecasting_model(
    self,
    forecasting_model: Any,
    forecasting_artifact_path: str,
  ) -> None:
    '''Log a forecasting fitted estimator.

    Args:
      forecasting_model: The fitted forecasting estimator.
      forecasting_artifact_path: The forecasting artifact destination.
    '''
    raise NotImplementedError('Forecasting contract requires a concrete implementation.')

  def log_forecasting_metrics(self, forecasting_metrics: dict[str, float] | None) -> None:
    '''Log forecasting evaluation metrics.

    Args:
      forecasting_metrics: The forecasting headline metric mapping, or None.
    '''
    raise NotImplementedError('Forecasting contract requires a concrete implementation.')

  def register_forecasting_model(
    self,
    forecasting_model_key: str,
    forecasting_run_name: str,
    forecasting_params: dict[str, Any] | None,
    forecasting_model: Any,
    forecasting_metrics: dict[str, float] | None,
  ) -> str | None:
    '''Log and register a forecasting model through forecasting constants.

    Registration is gated on non-empty forecasting metrics: an
    unevaluated forecasting model is never registered.

    Args:
      forecasting_model_key: The forecasting model identity (MODEL_KEY).
      forecasting_run_name: The forecasting run name.
      forecasting_params: The forecasting estimator parameters.
      forecasting_model: The forecasting fitted estimator.
      forecasting_metrics: The forecasting headline metrics. Empty or None
        skips forecasting model registration.

    Returns:
      The forecasting registered model version, or None when
      forecasting registration was skipped.
    '''
    raise NotImplementedError('Forecasting contract requires a concrete implementation.')


class ForecastingConcreteMLflowContract(ForecastingMLflowContract):
  '''Concrete forecasting MLflow contract with forecasting naming.

  This class restructures the forecasting register_model and experiment
  resolution logic through forecasting layer vocabulary. Every
  identity reference flows through forecasting constants rather than
  direct forecasting literals, and every environment lookup uses the
  forecasting naming convention.
  '''

  def __init__(self, forecasting_artifact_path: str = FORECASTING_ARTIFACT_PATH):
    '''Build the forecasting concrete MLflow contract.

    Args:
      forecasting_artifact_path: The forecasting artifact path. Defaults
        to the forecasting framework artifact name.
    '''
    self.forecasting_artifact_path = forecasting_artifact_path

  def is_configured(self) -> bool:
    '''Report whether forecasting tracking is configured.

    Returns:
      True when the forecasting tracking URI environment variable is
      present and non-empty.
    '''
    return bool(os.getenv(FORECASTING_TRACKING_URI_VAR))

  def resolve_experiment(self, forecasting_model_key: str) -> Any:
    '''Resolve or create the forecasting experiment for a model key.

    Args:
      forecasting_model_key: The forecasting model identity constant.

    Returns:
      The forecasting experiment entity.
    '''
    import mlflow  # noqa: PLC0415 - deferred so mlflow is optional at import
    from mlflow.entities import ViewType  # noqa: PLC0415

    experiments = mlflow.search_experiments(
      view_type=ViewType.ALL,
      filter_string=f'name = \'{forecasting_model_key}\'',
    )
    if experiments:
      LOGGER.info('Forecasting experiment found for model key', extra={'forecasting_model_key': forecasting_model_key})
      return experiments[0]
    LOGGER.info('Forecasting experiment not found; creating new experiment', extra={'forecasting_model_key': forecasting_model_key})
    forecasting_experiment_id = mlflow.create_experiment(forecasting_model_key)
    return mlflow.get_experiment(forecasting_experiment_id)

  def log_forecasting_params(self, params: dict[str, Any] | None) -> None:
    '''Log forecasting estimator parameters.

    Args:
      params: The forecasting parameter mapping, or None.
    '''
    if not params:
      return
    import mlflow  # noqa: PLC0415

    flat_params = self._flatten_forecasting_params(params)
    mlflow.log_params(flat_params)
    LOGGER.info('Forecasting parameters logged', extra={'forecasting_param_count': len(flat_params)})

  def log_forecasting_model(
    self,
    forecasting_model: Any,
    forecasting_artifact_path: str,
  ) -> None:
    '''Log a forecasting fitted estimator.

    Args:
      forecasting_model: The forecasting fitted estimator.
      forecasting_artifact_path: The forecasting artifact destination.
    '''
    import mlflow  # noqa: PLC0415

    forecasting_flavour = self._resolve_forecasting_flavour(forecasting_model)
    try:
      forecasting_flavour_module = getattr(mlflow, forecasting_flavour)
      forecasting_flavour_module.log_model(forecasting_model, forecasting_artifact_path)
      LOGGER.info('Forecasting model logged', extra={'forecasting_flavour': forecasting_flavour})
    except (ValueError, TypeError, AttributeError) as forecasting_error:
      LOGGER.warning(
        'Forecasting model flavour failed; falling back to sklearn flavour',
        extra={'forecasting_error': str(forecasting_error)},
      )
      mlflow.sklearn.log_model(forecasting_model, forecasting_artifact_path)

  def log_forecasting_metrics(self, forecasting_metrics: dict[str, float] | None) -> None:
    '''Log forecasting evaluation metrics.

    Args:
      forecasting_metrics: The forecasting headline metric mapping, or None.
    '''
    if forecasting_metrics is None or not forecasting_metrics:
      LOGGER.info('No forecasting metrics produced; forecasting model will not be registered')
      return
    import mlflow  # noqa: PLC0415

    forecasting_metrics = {str(key): float(value) for key, value in forecasting_metrics.items()}
    mlflow.log_metrics(forecasting_metrics)
    LOGGER.info('Forecasting metrics logged', extra={'forecasting_metric_keys': list(forecasting_metrics)})

  def register_forecasting_model(
    self,
    forecasting_model_key: str,
    forecasting_run_name: str,
    forecasting_params: dict[str, Any] | None,
    forecasting_model: Any,
    forecasting_metrics: dict[str, float] | None,
  ) -> str | None:
    '''Log and register a forecasting model through forecasting constants.

    Registration is gated on non-empty forecasting metrics: an
    unevaluated forecasting model is never registered.

    Args:
      forecasting_model_key: The forecasting model identity (MODEL_KEY).
      forecasting_run_name: The forecasting run name.
      forecasting_params: The forecasting estimator parameters.
      forecasting_model: The forecasting fitted estimator.
      forecasting_metrics: The forecasting headline metrics. Empty or None
        skips forecasting model registration.

    Returns:
      The forecasting registered model version, or None when
      forecasting registration was skipped.
    '''
    if not self.is_configured():
      LOGGER.info('Forecasting tracking is not configured; skipping forecasting model registration')
      return None

    forecasting_tracking_uri = os.getenv(FORECASTING_TRACKING_URI_VAR)
    if not forecasting_tracking_uri:
      return None

    import mlflow  # noqa: PLC0415

    mlflow.set_tracking_uri(forecasting_tracking_uri)
    forecasting_experiment = self.resolve_experiment(forecasting_model_key)

    forecasting_active_run = mlflow.active_run()
    if forecasting_active_run:
      LOGGER.info('Forecasting active run found; setting forecasting run tag')
      mlflow.set_tag('mlflow.runName', forecasting_run_name)
    else:
      LOGGER.info('Forecasting active run absent; creating new forecasting run')
      forecasting_active_run = mlflow.start_run(
        run_name=forecasting_run_name,
        experiment_id=forecasting_experiment.experiment_id,
      )

    self.log_forecasting_params(forecasting_params)
    if forecasting_model is not None:
      self.log_forecasting_model(forecasting_model, self.forecasting_artifact_path)

    if not forecasting_metrics:
      LOGGER.info('No forecasting metrics produced; forecasting model will not be registered', extra={'forecasting_model_key': forecasting_model_key})
      return None

    self.log_forecasting_metrics(forecasting_metrics)
    forecasting_run = mlflow.active_run()
    if forecasting_run is None:
      LOGGER.warning('Forecasting active run is None after forecasting metric logging')
      return None
    forecasting_model_uri = f'runs:/{forecasting_run.info.run_id}/{self.forecasting_artifact_path}'
    forecasting_model_version = mlflow.register_model(forecasting_model_uri, forecasting_model_key)
    LOGGER.info(
      'Forecasting model registered',
      extra={
        'forecasting_model_key': forecasting_model_key,
        'forecasting_model_version': str(forecasting_model_version),
      },
    )
    return str(forecasting_model_version)

  @staticmethod
  def _resolve_forecasting_flavour(forecasting_model: Any) -> str:
    '''Resolve the forecasting flavour from the forecasting estimator module.

    Args:
      forecasting_model: The forecasting fitted estimator.

    Returns:
      The forecasting flavour identifier.
    '''
    forecasting_module_name = type(forecasting_model).__module__.split('.')[0]
    forecasting_flavours = {
      'xgboost': 'xgboost',
      'catboost': 'catboost',
      'lightgbm': 'sklearn',
      'sklearn': 'sklearn',
    }
    forecasting_flavour = forecasting_flavours.get(forecasting_module_name, 'sklearn')
    LOGGER.debug('Forecasting flavour resolved', extra={'forecasting_flavour': forecasting_flavour, 'forecasting_module': forecasting_module_name})
    return forecasting_flavour

  @staticmethod
  def _flatten_forecasting_params(forecasting_params: dict[str, Any], forecasting_prefix: str = '') -> dict[str, Any]:
    '''Flatten a forecasting nested parameter mapping.

    Args:
      forecasting_params: The forecasting nested mapping.
      forecasting_prefix: The forecasting current key prefix.

    Returns:
      A flat forecasting mapping of dotted key to scalar value.
    '''
    forecasting_flat: dict[str, Any] = {}
    for forecasting_key, forecasting_value in forecasting_params.items():
      forecasting_composed = f'{forecasting_prefix}{forecasting_key}'
      if isinstance(forecasting_value, dict):
        forecasting_flat.update(
          ForecastingConcreteMLflowContract._flatten_forecasting_params(
            forecasting_value,
            forecasting_prefix=f'{forecasting_composed}.',
          ),
        )
      elif isinstance(forecasting_value, (list, tuple, set)):
        forecasting_flat[forecasting_composed] = ', '.join(str(item) for item in forecasting_value)
      elif isinstance(forecasting_value, (str, int, float, bool)) or forecasting_value is None:
        forecasting_flat[forecasting_composed] = forecasting_value
      else:
        forecasting_flat[forecasting_composed] = str(forecasting_value)
    return forecasting_flat


class ForecastingMLflowTrackingContract:
  '''Forecasting MLflow tracking contract factory (SOLID dependency inversion).

  The factory isolates forecasting tracking creation from forecasting
  model registration, so forecasting callers depend on an abstraction
  rather than a concrete forecasting MLflow contract.
  '''

  @staticmethod
  def create_forecasting_contract(
    forecasting_artifact_path: str = FORECASTING_ARTIFACT_PATH,
  ) -> ForecastingMLflowContract:
    '''Create a forecasting tracking contract instance.

    Args:
      forecasting_artifact_path: The forecasting artifact destination.

    Returns:
      A forecasting concrete MLflow tracking contract.
    '''
    return ForecastingConcreteMLflowContract(forecasting_artifact_path=forecasting_artifact_path)
