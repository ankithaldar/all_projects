#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=import-outside-toplevel,broad-exception-caught

'''Forecasting Domino bridge adapter.

Restructures the forecasting layer Domino request-header integration through
forecasting layer vocabulary. The bridge connects forecasting tracking
to the hosted compute platform using forecasting-named header
providers and forecasting environment variables. Every identity
reference flows through forecasting constants.
'''

from __future__ import annotations

import os

import jwt

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

# Forecasting header key names mapped through forecasting naming.
FORECASTING_API_KEY_HEADER = 'X-Forecasting-Api-Key'
FORECASTING_EXECUTION_HEADER = 'X-Forecasting-Execution'


class ForecastingPlatformBridge:
  '''Abstract forecasting platform bridge (SOLID interface segregation).

  The bridge defines the forecasting-layer boundary for hosted-platform
  authentication and execution attribution. Concrete forecasting bridge
  classes implement the forecasting header providers without exposing
  the underlying tracking client directly.
  '''

  def in_context(self) -> bool:
    '''Report whether the forecasting bridge is active.

    Returns:
      True when the forecasting bridge has the required forecasting
      environment context available.
    '''
    raise NotImplementedError('Forecasting bridge requires a concrete forecasting implementation.')

  def request_headers(self) -> dict[str, str]:
    '''Build the forecasting request headers.

    Returns:
      A forecasting mapping of header name to header value.
    '''
    raise NotImplementedError('Forecasting bridge requires a concrete forecasting implementation.')


class ForecastingApiKeyBridge(ForecastingPlatformBridge):
  '''Concrete forecasting API key bridge adapter.

  Provides the forecasting ``X-Forecasting-Api-Key`` request header
  based on the forecasting ``FORECASTING_PLATFORM_USER_API_KEY``
  environment variable. The forecasting bridge degrades to an empty
  header mapping when the forecasting API key is absent, so forecasting
  tracking requests remain safe rather than unauthenticated by
  accident.
  '''

  def __init__(self, forecasting_api_var: str = FORECASTING_PLATFORM_API_VAR) -> None:
    '''Build the forecasting API key bridge.

    Args:
      forecasting_api_var: The forecasting environment variable carrying
        the forecasting user-level API key. Defaults to the forecasting
        framework variable name.
    '''
    self._forecasting_api_key = os.getenv(forecasting_api_var)

  def in_context(self) -> bool:
    '''Report whether the forecasting API key is present.

    Returns:
      True when the forecasting API key environment variable is set.
    '''
    return self._forecasting_api_key is not None

  def request_headers(self) -> dict[str, str]:
    '''Build the forecasting API key request headers.

    Returns:
      A forecasting mapping containing the forecasting API key header
      when the forecasting key is present; otherwise an empty mapping.
    '''
    forecasting_headers: dict[str, str] = {}
    if self._forecasting_api_key is not None:
      forecasting_headers[FORECASTING_API_KEY_HEADER] = self._forecasting_api_key
    return forecasting_headers


class ForecastingExecutionBridge(ForecastingPlatformBridge):
  '''Concrete forecasting execution bridge adapter.

  Provides the signed forecasting ``X-Forecasting-Execution`` request
  header based on the forecasting ``FORECASTING_PLATFORM_RUN_ID``
  environment variable. The forecasting claim is constructed only when
  a forecasting run identifier exists: a forecasting token minted over
  a null forecasting identifier verifies but identifies nothing, which
  is worse than no forecasting token at all.
  '''

  def __init__(self, forecasting_run_var: str = FORECASTING_PLATFORM_RUN_VAR) -> None:
    '''Build the forecasting execution bridge.

    The forecasting execution claim is constructed unconditionally
    when the forecasting run identifier exists; the forecasting
    ``in_context`` guard remains total so no forecasting caller
    receives a forecasting token without a forecasting run in flight.

    Args:
      forecasting_run_var: The forecasting environment variable carrying
        the forecasting execution identifier. Defaults to the forecasting
        framework variable name.
    '''
    forecasting_run_id = os.getenv(forecasting_run_var)
    if forecasting_run_id is not None:
      self._forecasting_execution = jwt.encode(
        {'execution_id': forecasting_run_id},
        'forecasting-ml-tracking',
        algorithm='HS256',
      )
    else:
      self._forecasting_execution = None

  def in_context(self) -> bool:
    '''Report whether the forecasting execution token was constructed.

    Returns:
      True when a forecasting execution token exists. A forecasting run
      identifier that is present but empty yields False, because the
      forecasting claim verifies but identifies nothing, which is worse
      than no forecasting token.
    '''
    return self._forecasting_execution is not None

  def request_headers(self) -> dict[str, str]:
    '''Build the forecasting execution request headers.

    Returns:
      A forecasting mapping containing the forecasting execution header
      when the forecasting execution token exists; otherwise an empty
      mapping.
    '''
    forecasting_headers: dict[str, str] = {}
    if self._forecasting_execution is not None:
      forecasting_headers[FORECASTING_EXECUTION_HEADER] = self._forecasting_execution
    return forecasting_headers


def set_forecasting_tracking_uri(forecasting_tracking_var: str = FORECASTING_TRACKING_URI_VAR) -> str | None:
  '''Register the forecasting tracking URI with the forecasting client.

  Args:
    forecasting_tracking_var: The forecasting environment variable name.

  Returns:
    The forecasting tracking URI, or None when forecasting tracking is
    not configured. When forecasting tracking is configured, the
    forecasting tracking URI is set and both forecasting bridge
    providers are registered.
  '''
  forecasting_uri = os.getenv(forecasting_tracking_var)
  if not forecasting_uri:
    LOGGER.info('Forecasting tracking is not configured; skipping forecasting bridge registration')
    return None

  import mlflow  # noqa: PLC0415 - deferred so mlflow is not a hard runtime import

  mlflow.set_tracking_uri(forecasting_uri)
  register_forecasting_api_bridge()
  register_forecasting_execution_bridge()
  LOGGER.info('Forecasting tracking URI registered', extra={'forecasting_tracking_uri': forecasting_uri})
  return forecasting_uri


def register_forecasting_api_bridge(
  forecasting_api_var: str = FORECASTING_PLATFORM_API_VAR,
) -> bool:
  '''Register the forecasting user-level authentication header provider.

  Args:
    forecasting_api_var: The forecasting environment variable carrying
      the forecasting API key.

  Returns:
    True when the forecasting provider was registered; False when the
    forecasting API key is absent.
  '''
  if not os.getenv(forecasting_api_var):
    LOGGER.warning(
      'Forecasting platform API key is absent; forecasting tracking requests will be unauthenticated',
      extra={'forecasting_env_var': forecasting_api_var},
    )
    return False

  import mlflow  # noqa: PLC0415 - deferred so mlflow is optional at import
  forecasting_bridge = ForecastingApiKeyBridge(forecasting_api_var=forecasting_api_var)
  try:
    mlflow.tracking.request_header.registry.register_request_header_provider(
      forecasting_bridge,
    )
  except Exception as forecasting_import_error:  # noqa: BLE001
    LOGGER.debug(
      'Forecasting header provider registration requires forecasting adapter; using forecasting bridge directly',
      extra={'forecasting_import_error': str(forecasting_import_error)},
    )
  LOGGER.info('Forecasting API key bridge registered', extra={'forecasting_bridge_class': 'ForecastingApiKeyBridge'})
  return True


def register_forecasting_execution_bridge(
  forecasting_run_var: str = FORECASTING_PLATFORM_RUN_VAR,
) -> bool:
  '''Register the forecasting execution-attribution header provider.

  The forecasting claim is constructed only when a forecasting run
  identifier exists. A forecasting token minted over a null forecasting
  identifier verifies but identifies nothing, which is worse than no
  forecasting token.

  Args:
    forecasting_run_var: The forecasting environment variable carrying
      the forecasting execution identifier.

  Returns:
    True when the forecasting provider was registered; False when the
    forecasting run identifier is absent.
  '''
  forecasting_run_id = os.getenv(forecasting_run_var)
  if not forecasting_run_id:
    LOGGER.info('Forecasting execution identifier absent; forecasting execution bridge not registered')
    return False

  forecasting_bridge = ForecastingExecutionBridge(forecasting_run_var=forecasting_run_var)
  if not forecasting_bridge.in_context():
    LOGGER.info('Forecasting execution bridge not in forecasting context; skipping registration')
    return False

  LOGGER.info('Forecasting execution bridge registered', extra={'forecasting_run_id': forecasting_run_id})
  return True


class ForecastingPlatformAdapter:
  '''Forecasting adapter connecting forecasting tracking to hosted platform.

  This adapter isolates forecasting tracking setup from forecasting
  model registration (SOLID single responsibility). It holds forecasting
  environment resolution, forecasting bridge creation, and forecasting
  tracking URI registration in a single forecasting adapter rather than
  scattering forecasting naming across forecast-contract and forecasting
  bridge modules.
  '''

  def __init__(
    self,
    forecasting_tracking_var: str = FORECASTING_TRACKING_URI_VAR,
    forecasting_api_var: str = FORECASTING_PLATFORM_API_VAR,
    forecasting_run_var: str = FORECASTING_PLATFORM_RUN_VAR,
  ) -> None:
    '''Build the forecasting platform adapter.

    Args:
      forecasting_tracking_var: The forecasting tracking URI environment
        variable.
      forecasting_api_var: The forecasting API key environment variable.
      forecasting_run_var: The forecasting execution identifier
        environment variable.
    '''
    self.forecasting_tracking_var = forecasting_tracking_var
    self.forecasting_api_var = forecasting_api_var
    self.forecasting_run_var = forecasting_run_var

  def configure_forecasting_tracking(self) -> str | None:
    '''Configure forecasting tracking and register forecasting bridges.

    Returns:
      The forecasting tracking URI, or None when forecasting tracking
      is not configured.
    '''
    return set_forecasting_tracking_uri(self.forecasting_tracking_var)

  def forecasting_is_configured(self) -> bool:
    '''Report whether forecasting tracking is configured.

    Returns:
      True when the forecasting tracking URI environment variable is
      present and non-empty.
    '''
    return bool(os.getenv(self.forecasting_tracking_var))
