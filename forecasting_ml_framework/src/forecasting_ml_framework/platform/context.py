#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Programmatic pipeline execution and the Spark-aware Kedro context.

A Spark-aware context is what lets the framework be exercised outside a cluster:
it constructs the session before any node runs and hands the run's parameters to
the single session builder, so the connector coordinates and the credential
refresh are configured identically however the session came into being.

Both components are enabled. The context forwards the run's cloud credential
under the name the session builder binds, and the configuration loader reads
only attributes its parent actually assigns -- a name that is read but never set
is an ``AttributeError`` at first use, which is the worst possible moment to
discover it.
"""

from __future__ import annotations

from typing import Any

from forecasting_ml_framework.exceptions import SparkSessionError
from forecasting_ml_framework.observability.logging import get_logger
from forecasting_ml_framework.platform.environment import resolve_environment
from forecasting_ml_framework.platform.spark import ACCESS_TOKEN_CONFIG_KEY, get_spark_session

LOGGER = get_logger(__name__)

try:  # pragma: no cover - kedro is a hard dependency but the guard aids tooling
  from kedro.framework.context import KedroContext

  _CONTEXT_BASE: Any = KedroContext
except Exception:  # noqa: BLE001 - allow import without kedro installed
  _CONTEXT_BASE = object  # type: ignore[assignment,misc]


class ForecastingSparkContext(_CONTEXT_BASE):  # type: ignore[misc,valid-type]
  """A Kedro context that constructs a Spark session with the BigQuery connector.

  The context delegates entirely to the single session builder in
  :mod:`forecasting_ml_framework.platform.spark`, so the connector coordinates
  and the credential refresh have exactly one definition regardless of which
  component created the session first.
  """

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    """Build the context.

    Args:
      *args: Positional arguments forwarded to the parent context.
      **kwargs: Keyword arguments forwarded to the parent context.
    """
    super().__init__(*args, **kwargs)

  def _get_spark_session(self) -> Any:
    """Return the session, constructing it on first use.

    Returns:
      The live ``SparkSession``.
    """
    if getattr(self, '_spark', None) is not None:
      return self._spark
    params = dict(getattr(self, 'params', {}) or {})
    conf_loader = getattr(self, 'config_loader', None)
    if conf_loader is not None:
      try:
        params = {**dict(conf_loader['spark']), **params}
      except Exception:  # noqa: BLE001 - the spark group is optional
        LOGGER.debug('No spark configuration group found in the loader')
    session, _ = get_spark_session(params=params, apply_tuning=True)
    self._spark = session
    return session

  def close(self) -> None:
    """Release the Spark session held by this context."""
    if getattr(self, '_spark', None) is not None:
      self._spark.stop()
      self._spark = None


def init_spark_session(runtime_params: dict[str, Any] | None = None) -> Any:
  """Construct the Spark session for a Kedro session.

  This is a ``kedro_spark.init_spark_session`` shape, extended so the connector
  credential is configured from the resolved environment rather than being assumed.

  Args:
    runtime_params: The run parameters.

  Returns:
    The live ``SparkSession``.

  Raises:
    SparkSessionError: If hosted mode is requested but no credential was obtained.
    A hosted run without a credential would otherwise fail at its first warehouse
    read, which points at the connector rather than at the missing credential.
  """
  session, _ = get_spark_session(params=runtime_params or {}, apply_tuning=True)
  configured = ACCESS_TOKEN_CONFIG_KEY in session.conf.getAll()
  environment = resolve_environment(runtime_params)
  if environment.is_hosted and not configured:
    raise SparkSessionError(
      'Hosted mode was requested but the connector credential was not configured. Either the '
      'Cloud SDK is unauthenticated or the token could not be fetched; a hosted run cannot '
      'proceed without one.',
      is_hosted_env=True,
    )
  LOGGER.info(
    'Spark session initialised for the Kedro context',
    extra={'token_configured': configured, 'environment': str(environment)},
  )
  return session
