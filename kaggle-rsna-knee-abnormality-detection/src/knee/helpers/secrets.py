#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Secret resolution: environment -> .env file -> Kaggle User Secrets.

Every external credential (Discord webhook, W&B API key, Kaggle token) is
referenced by *name* in configuration; values never live in code or YAML.
"""

from __future__ import annotations

import os
from pathlib import Path

from knee.helpers.utils import get_logger

_LOGGER = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ENV_LOADED = False
_LOADED_PATH: Path | None = None


def load_project_env(env_path: str | None = None) -> bool:
  """Load a project ``.env`` file into ``os.environ``.

  The default repository-root file loads once per process; explicit paths
  load when they differ from the previously loaded one. Existing process
  variables are never overridden (``override=False``).

  Args:
      env_path: Optional explicit path; defaults to the repository root.

  Returns:
      True when a file was loaded (values are exported to os.environ).
  """
  global _ENV_LOADED, _LOADED_PATH  # pylint: disable=global-statement
  target = Path(env_path) if env_path else _PROJECT_ROOT / '.env'
  if _ENV_LOADED and _LOADED_PATH == target:
    return True
  loaded = False
  if target.exists():
    try:
      # Optional dependency; environments without it simply skip .env.
      from dotenv import load_dotenv  # pylint: disable=import-outside-toplevel

      load_dotenv(target, override=False)
      loaded = True
    except ImportError:
      _LOGGER.warning('python-dotenv missing; skipped %s', target)
  _ENV_LOADED = True
  _LOADED_PATH = target
  return loaded


def _from_kaggle_secrets(name: str) -> str | None:
  """Read one secret from the Kaggle notebook store when available.

  Args:
      name: Secret label registered under Add-ons > Secrets.

  Returns:
      Secret value or None outside Kaggle notebooks / when unregistered.
  """
  try:
    # Notebook-only dependency, mirrors helpers.kaggle_io policy.
    from kaggle_secrets import (  # pylint: disable=import-outside-toplevel
      UserSecretsClient,
    )

    return UserSecretsClient().get_secret(name)
  except Exception:  # pylint: disable=broad-exception-caught
    return None


def get_secret(
  name: str,
  default: str | None = None,
  env_path: str | None = None,
) -> str | None:
  """Resolve a secret by name across all supported backends.

  Lookup order:

  1. Process environment (already set on CI / exported shells).
  2. Project ``.env`` (loaded lazily).
  3. Kaggle User Secrets (notebook runs).

  Args:
      name: Secret variable name, e.g. ``DISCORD_WEBHOOK_URL``.
      default: Value returned when every backend misses.
      env_path: Optional explicit ``.env`` path forwarded to
          ``load_project_env`` (tests and non-root checkouts).

  Returns:
      The secret value or ``default``.
  """
  load_project_env(env_path)
  value = os.environ.get(name)
  if value:
    return value
  return _from_kaggle_secrets(name) or default
