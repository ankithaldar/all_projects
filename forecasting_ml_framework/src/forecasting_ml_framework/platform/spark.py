#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Process-wide Spark session lifecycle.

Three components in this framework need a Spark session: the registry's
fit/apply loop, the modelling nodes, and the BigQuery-Spark dataset adapter.
This module is the single builder. Every other component delegates here, so the
tuning configuration, the BigQuery connector coordinates and the credential
refresh have exactly one definition -- and therefore cannot drift apart
depending on which component happened to create the session first.

Three properties are load-bearing and worth stating explicitly:

* **No mutable default arguments.** Both parameter bags default to ``None`` and
  are normalised into a local mapping. A shared default that a caller writes into
  would contaminate every later call in the process, and the damage is silent:
  the session still builds, it just has no connector configuration, and the
  failure surfaces at the first warehouse read with an error that points at the
  connector rather than at its cause.
* **The credential is refreshed on the live session.** ``getOrCreate`` ignores a
  builder's configuration once a session exists, so a freshly-minted token passed
  to a builder is discarded. ``spark.conf.set`` on the existing session is what
  actually updates the connector credential. Refreshes are rate-limited, because
  minting a token shells out to the cloud SDK and the registry calls this
  accessor once per transform.
* **Configuration errors are raised, not deferred.** Mandatory keys are
  validated at construction, so a missing ``model_key`` fails here with a
  message naming it, rather than as a ``KeyError`` from a frame several layers
  away.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from forecasting_ml_framework import constants
from forecasting_ml_framework.exceptions import (
  SparkSessionError,
  SparkTuningConfigNotFoundError,
)
from forecasting_ml_framework.observability.logging import get_logger
from forecasting_ml_framework.platform.environment import as_flag, resolve_environment
from forecasting_ml_framework.utils.text import require

LOGGER = get_logger(__name__)

#: Spark configuration keys read from ``conf/<env>/spark.yml``.
SPARK_TUNING_FILENAME = 'spark.yml'

#: Runtime config key carrying the connector credential.
ACCESS_TOKEN_CONFIG_KEY = 'gcpAccessToken'

#: Minimum seconds between credential refreshes. Refreshing on every registry
#: routine would spawn one subprocess per routine for no benefit.
TOKEN_REFRESH_INTERVAL_SECONDS = 1800

#: Mandatory parameter keys for a fully-configured session. The hosted flag is
#: deliberately absent: it is resolved through :func:`resolve_environment`, which
#: accepts either the current or the legacy spelling and defaults to false, so
#: requiring it here would reject a correctly configured run for omitting a flag
#: whose absence already means the same thing.
_MANDATORY_PARAMS = (
  'model_key',
  'external_project',
  'data_project',
  'db_staging_data',
)

#: Bookkeeping for the credential refresh, keyed by session identity.
_TOKEN_STATE: dict[str, float] = {}


def get_spark_session(
  global_params: dict[str, Any] | None = None,
  params: dict[str, Any] | None = None,
  apply_tuning: bool = False,
  conf_dir: str | Path | None = None,
  conf_env: str | None = None,
) -> tuple[Any, Any]:
  """Build or reuse the process-wide Spark session.

  The first call in a process creates the session and applies the BigQuery
  connector configuration. Every subsequent call returns the same object and, if
  credentials are in use, refreshes the connector token on the **live** session
  rather than rebuilding it.

  Args:
    global_params: A shared parameter bag. Its keys win over ``params``, which
      is a merge direction and is preserved deliberately: the node
      layer compensates by passing a single dictionary.
    params: The run parameters.
    apply_tuning: Whether to apply the environment's Spark tuning document.
      The custom run command passes ``True``; the dataset adapter and the
      registry pass ``False`` so the tuning is applied exactly once per process.
    conf_dir: The configuration root. Defaults to the project's ``conf``
      directory.
    conf_env: The configuration environment. Defaults to ``base``.

  Returns:
    A two-tuple of the ``SparkSession`` and its ``SparkContext``.

  Raises:
    MissingParameterError: If a mandatory parameter is absent.
    SparkTuningConfigNotFoundError: If tuning is requested but no document exists.
  """
  from pyspark.sql import SparkSession  # noqa: PLC0415 - deferred so the core imports without pyspark

  merged = dict(params or {})
  merged.update(global_params or {})

  builder = SparkSession.builder
  if merged:
    if not merged.get('model_key'):
      # A bare session is legitimate for local exploration, so only the
      # fully-configured path is validated.
      if all(key in merged for key in _MANDATORY_PARAMS):
        _validate_session_params(merged)
    else:
      _validate_session_params(merged)

  existing = SparkSession.getActiveSession()
  if existing is None:
    if apply_tuning:
      # `SparkSession.Builder.config` takes a `SparkConf` in the `conf`
      # keyword and a plain mapping in `map`. A mapping passed to `conf` is
      # read with `.getAll()`, which a dict does not have, so the whole tuned
      # path raised AttributeError before a session existed. This is the entire
      # production entry point, so the failure was not a tuning defect -- it was
      # a total outage of every tuned run.
      builder = builder.config(map=_load_spark_conf(conf_dir, conf_env))
    if merged.get('model_key'):
      builder = builder.appName(str(merged['model_key']))
    builder = builder.config('viewsEnabled', 'true')
    if merged.get('external_project'):
      builder = builder.config('parentProject', str(merged['external_project']))
    if merged.get('data_project'):
      builder = builder.config('materializationProject', str(merged['data_project']))
    if merged.get('db_staging_data'):
      builder = builder.config('materializationDataset', str(merged['db_staging_data']))
    if resolve_environment(merged).is_hosted:
      builder = builder.config(ACCESS_TOKEN_CONFIG_KEY, _fetch_access_token())

    try:
      session = builder.getOrCreate()
    except Exception as error:  # noqa: BLE001 - spark raises many types
      raise SparkSessionError('Unable to create or reuse a Spark session', error=str(error)) from error
  else:
    session = existing
    _maybe_refresh_token(session, merged)

  LOGGER.info(
    'Spark session ready',
    extra={
      'app_name': session.sparkContext.appName,
      'master': str(session.sparkContext.master),
      'reused': existing is not None,
    },
  )
  return session, session.sparkContext


def refresh_access_token(session: Any) -> str | None:
  """Force-refresh the BigQuery connector credential on a live session.

  Args:
    session: The live ``SparkSession``.

  Returns:
    The new token, or ``None`` when hosted-platform mode is off.
  """
  token = _fetch_access_token()
  if not token:
    return None
  session.conf.set(ACCESS_TOKEN_CONFIG_KEY, token)
  return token


def _maybe_refresh_token(session: Any, params: dict[str, Any]) -> None:
  """Refresh the connector token on an existing session when it is stale.

  Args:
    session: The live ``SparkSession``.
    params: The merged run parameters.
  """
  if not resolve_environment(params).is_hosted:
    return
  key = str(session.sparkContext.applicationId)
  now = time.monotonic()
  if now - _TOKEN_STATE.get(key, 0.0) < TOKEN_REFRESH_INTERVAL_SECONDS:
    return
  token = _fetch_access_token()
  if token:
    session.conf.set(ACCESS_TOKEN_CONFIG_KEY, token)
    _TOKEN_STATE[key] = now


def _validate_session_params(params: dict[str, Any]) -> None:
  """Assert that every mandatory session parameter is present.

  Args:
    params: The merged run parameters.

  Raises:
    MissingParameterError: If a mandatory key is absent.
  """
  for key in _MANDATORY_PARAMS:
    require(params, key, stage='get_spark_session')


def _is_hosted(value: Any) -> bool:
  """Interpret the hosted-environment flag, which arrives as a string or a bool.

  Args:
    value: The raw flag.

  Returns:
    ``True`` when hosted mode is enabled.

  Raises:
    ConfigurationError: If the value is not recognisable as a boolean. An
      unrecognised value is rejected rather than read as false, because the
      credential path chosen on a misread flag fails later and somewhere unrelated.
  """
  return as_flag(value, default=False, key=constants.ENV_IS_HOSTED)


def _fetch_access_token() -> str:
  """Shell out to the cloud SDK for a short-lived access token.

  Raises:
    SparkSessionError: If the token cannot be obtained.
  """
  try:
    token = subprocess.getoutput('gcloud auth application-default print-access-token').strip()
  except Exception as error:  # noqa: BLE001 - defensive
    raise SparkSessionError('Failed to shell out for a cloud access token', error=str(error)) from error
  if not token or 'ERROR' in token.upper():
    raise SparkSessionError(
      'Could not obtain a cloud access token. The Cloud SDK must be installed and authenticated '
      'in the runtime image. The failure is reported here rather than at the first BigQuery read, '
      'where the error would point at the connector instead of at its cause.'
    )
  return token


def _load_spark_conf(conf_dir: str | Path | None, conf_env: str | None) -> dict[str, str]:
  """Load the Spark tuning document for an environment, with a base fallback.

  Args:
    conf_dir: The configuration root. Defaults to ``<project>/conf``.
    conf_env: The configuration environment. Defaults to ``base``.

  Returns:
    The Spark configuration as a mapping.

  Raises:
    SparkTuningConfigNotFoundError: If no document exists for the environment or
      for ``base``.
  """
  import yaml  # noqa: PLC0415

  root = Path(conf_dir) if conf_dir else Path(__file__).resolve().parents[3] / 'conf'
  candidates = []
  if conf_env:
    candidates.append(root / conf_env / SPARK_TUNING_FILENAME)
  candidates.append(root / 'base' / SPARK_TUNING_FILENAME)

  for candidate in candidates:
    if candidate.is_file():
      with candidate.open(encoding='utf-8') as handle:
        loaded = yaml.safe_load(handle) or {}
      LOGGER.info('Loaded Spark tuning document', extra={'path': str(candidate), 'keys': len(loaded)})
      return {str(key): str(value) for key, value in loaded.items()}

  searched = ', '.join(str(candidate) for candidate in candidates)
  raise SparkTuningConfigNotFoundError(
    f'No Spark tuning document found. Searched: {searched}. The tuning document is applied only '
    'here, so if it is missing the run proceeds with Spark defaults and Arrow-based pandas '
    'conversion may be unavailable.',
    searched=searched,
  )
