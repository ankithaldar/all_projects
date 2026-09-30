#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Experiment-tracking integration.

Experiment tracking follows the platform split: authentication is only available
when the run executes on the hosted compute platform, so the integration is
written to be inert anywhere else. ``mlflow`` and ``jwt`` are imported lazily
inside the functions that need them, which means neither is imported at all on a
run that never tracks anything.

Two properties make the integration safe to enable by default:

1. It **degrades to a no-op** when the tracking environment is absent, so a
   developer need not install or configure MLflow to run the framework, and a
   missing backend can never be the reason a production run fails.
2. The execution-token provider's ``in_context`` guard is **total**. A token is
   minted only when a run identifier actually exists, so no caller can receive a
   JWT over a null identifier -- a token that verifies but identifies nothing is
   worse than no token, because it looks valid.
"""

from __future__ import annotations

import os
from typing import Any

from forecasting_ml_framework.observability.logging import get_logger

LOGGER = get_logger(__name__)

#: Environment variable carrying the user-level tracking API key.
API_KEY_ENV_VAR = 'HOSTED_PLATFORM_USER_API_KEY'
#: Environment variable carrying the execution identifier.
RUN_ID_ENV_VAR = 'HOSTED_PLATFORM_RUN_ID'
#: Environment variable carrying the tracking server URI.
TRACKING_URI_ENV_VAR = 'MLFLOW_TRACKING_URI'

#: Artifact path under which the fitted estimator is logged.
DEFAULT_ARTIFACT_PATH = 'core-model'


def tracking_is_configured() -> bool:
  """Report whether an experiment-tracking server has been configured.

  Returns:
    ``True`` when ``MLFLOW_TRACKING_URI`` is set, ``False`` otherwise.
  """
  return bool(os.getenv(TRACKING_URI_ENV_VAR))


def set_tracking_uri() -> str | None:
  """Register the hosted-platform request headers with the tracking client.

  Returns:
    The tracking URI in use, or ``None`` when tracking is not configured.
  """
  uri = os.getenv(TRACKING_URI_ENV_VAR)
  if not uri:
    LOGGER.info('Experiment tracking is not configured; skipping registration')
    return None
  import mlflow  # noqa: PLC0415 - deferred so mlflow is not a hard runtime import

  mlflow.set_tracking_uri(uri)
  register_api_key_request_header_provider()
  register_execution_request_header_provider()
  return uri


def register_api_key_request_header_provider() -> bool:
  """Register the user-level authentication header provider.

  Returns:
    ``True`` when the provider was registered, ``False`` when the API key is
    absent.
  """
  if not os.getenv(API_KEY_ENV_VAR):
    LOGGER.warning(
      'Hosted-platform API key is absent; tracking requests will be unauthenticated',
      extra={'env_var': API_KEY_ENV_VAR},
    )
    return False
  import mlflow  # noqa: PLC0415
  from mlflow.tracking.request_header import registry  # noqa: PLC0415

  class HostedPlatformApiKeyRequestHeaderProvider(_BaseProvider):
    """Provides the ``X-HostedPlatform-Api-Key`` request header."""

    def __init__(self) -> None:
      self._api_key = os.getenv(API_KEY_ENV_VAR)

    def in_context(self) -> bool:
      """Report whether the provider is active.

      Returns:
        ``True`` when an API key is present.
      """
      return self._api_key is not None

    def request_headers(self) -> dict[str, str]:
      """Build the request headers.

      Returns:
        A mapping of header name to value.
      """
      return {'X-HostedPlatform-Api-Key': self._api_key} if self._api_key else {}

  mlflow.tracking.request_header.registry.register_request_header_provider(
    HostedPlatformApiKeyRequestHeaderProvider()
  )
  _ = registry  # keep the import meaningful for readers of the module
  return True


def register_execution_request_header_provider() -> bool:
  """Register the execution-attribution header provider.

  The claim is constructed **only** when a run identifier exists. A
  implementation built the token unconditionally, so ``in_context()`` returned
  ``True`` even with no run in flight.

  Returns:
    ``True`` when the provider was registered, ``False`` when the run identifier
    is absent.
  """
  run_id = os.getenv(RUN_ID_ENV_VAR)
  if not run_id:
    return False
  import jwt  # noqa: PLC0415
  import mlflow  # noqa: PLC0415

  class HostedPlatformExecutionRequestHeaderProvider(_BaseProvider):
    """Provides the signed ``X-HostedPlatform-Execution`` request header."""

    def __init__(self) -> None:
      # A signature over a claim nobody can influence is an attribution token,
      # not an integrity proof; the tracking server is the verifier.
      self._execution = jwt.encode({'execution_id': run_id}, 'forecasting-ml-tracking', algorithm='HS256')

    def in_context(self) -> bool:
      """Report whether the provider is active.

      Returns:
        ``True`` when an execution token was constructed.
      """
      return self._execution is not None

    def request_headers(self) -> dict[str, str]:
      """Build the request headers.

      Returns:
        A mapping of header name to value.
      """
      return {'X-HostedPlatform-Execution': self._execution} if self._execution else {}

  mlflow.tracking.request_header.registry.register_request_header_provider(
    HostedPlatformExecutionRequestHeaderProvider()
  )
  return True


def register_model(
  model_key: str,
  run_name: str,
  params: dict[str, Any] | None,
  model: Any,
  metrics: dict[str, float] | None,
) -> str | None:
  """Log a trained model to the tracking server and register it.

  One experiment and one registered model are created per ``model_key`` — the
  same identifier the framework uses for the metadata registry, the metrics path
  and the intermediate path, so a single key spans four systems.

  Registration is gated on non-empty metrics: an unevaluated model is never
  registered.

  Args:
    model_key: The model identifier used as experiment and registered-model name.
    run_name: The run name.
    params: The effective estimator parameters.
    model: The fitted estimator.
    metrics: The headline metric row. Empty or ``None`` skips registration.

  Returns:
    The registered model version, or ``None`` when registration was skipped.
  """
  if not tracking_is_configured():
    return None
  uri = set_tracking_uri()
  if uri is None:
    return None

  import mlflow  # noqa: PLC0415
  from mlflow.entities import ViewType  # noqa: PLC0415

  experiment = _resolve_experiment(mlflow, ViewType, model_key)

  active_run = mlflow.active_run()
  if active_run:
    mlflow.set_tag('mlflow.runName', run_name)
  else:
    mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id)

  if params:
    mlflow.log_params(_flatten_params(params))
  if model is not None:
    _log_model(mlflow, model, DEFAULT_ARTIFACT_PATH)
  if not metrics:
    LOGGER.info('No metrics produced; model will not be registered', extra={'model_key': model_key})
    return None

  mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
  run = mlflow.active_run()
  if run is None:  # pragma: no cover - defensive
    return None
  model_uri = f'runs:/{run.info.run_id}/{DEFAULT_ARTIFACT_PATH}'
  model_version = mlflow.register_model(model_uri, model_key)
  return str(model_version)


def end_active_runs(status: str | None = None) -> int:
  """Close every active tracking run.

  The framework is invoked once per pipeline stage in a single process, so a
  leaked run would survive into the next stage. The loop is defensive because
  MLflow permits nested runs.

  Args:
    status: An optional terminal status such as ``'FAILED'``.

  Returns:
    The number of runs that were closed.
  """
  if not tracking_is_configured():
    return 0
  try:
    import mlflow  # noqa: PLC0415
  except ImportError:  # pragma: no cover - optional dependency
    return 0
  closed = 0
  while mlflow.active_run():
    mlflow.end_run(status)
    closed += 1
  return closed


def _resolve_experiment(mlflow: Any, view_type: Any, model_key: str) -> Any:
  """Return the experiment for a model key, creating it when absent.

  Args:
    mlflow: The imported ``mlflow`` module.
    view_type: The ``ViewType`` enum class.
    model_key: The model identifier.

  Returns:
    The experiment entity.
  """
  experiments = mlflow.search_experiments(view_type=view_type.ALL, filter_string=f"name = '{model_key}'")
  if experiments:
    return experiments[0]
  experiment_id = mlflow.create_experiment(model_key)
  return mlflow.get_experiment(experiment_id)


def _log_model(mlflow: Any, model: Any, artifact_path: str) -> None:
  """Log an estimator using the flavour that matches its type.

  Args:
    mlflow: The imported ``mlflow`` module.
    model: The fitted estimator.
    artifact_path: The artifact destination.
  """
  module_name = type(model).__module__.split('.')[0]
  flavours = {
    'xgboost': 'xgboost',
    'catboost': 'catboost',
    'lightgbm': 'sklearn',
    'sklearn': 'sklearn',
  }
  flavour = flavours.get(module_name, 'sklearn')
  try:
    getattr(mlflow, flavour).log_model(model, artifact_path)
  except (ValueError, TypeError, AttributeError) as error:
    LOGGER.warning('Falling back to the sklearn flavour for model logging', extra={'error': str(error)})
    mlflow.sklearn.log_model(model, artifact_path)


def _flatten_params(params: dict[str, Any], prefix: str = '') -> dict[str, Any]:
  """Flatten a nested parameter mapping into MLflow-compatible scalar values.

  Args:
    params: The nested mapping.
    prefix: The current key prefix.

  Returns:
    A flat mapping of dotted key to scalar value.
  """
  flat: dict[str, Any] = {}
  for key, value in params.items():
    composed = f'{prefix}{key}'
    if isinstance(value, dict):
      flat.update(_flatten_params(value, prefix=f'{composed}.'))
    elif isinstance(value, (list, tuple, set)):
      flat[composed] = ', '.join(str(item) for item in value)
    elif isinstance(value, (str, int, float, bool)) or value is None:
      flat[composed] = value
    else:
      flat[composed] = str(value)
  return flat


try:  # pragma: no cover - exercised only when mlflow is installed
  from mlflow.tracking.request_header.request_header_provider import (
    RequestHeaderProvider as _BaseProvider,
  )
except Exception:  # noqa: BLE001 - mlflow layout differs across versions
  class _BaseProvider:  # type: ignore[no-redef]
    """Minimal stand-in used when MLflow's provider base class is unavailable."""

    def in_context(self) -> bool:
      """Report whether the provider is active.

      Returns:
        ``False`` by default.
      """
      return False

    def request_headers(self) -> dict[str, str]:
      """Build the request headers.

      Returns:
        An empty mapping.
      """
      return {}
