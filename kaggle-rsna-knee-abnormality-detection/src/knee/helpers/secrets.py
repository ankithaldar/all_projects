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


def load_project_env(env_path: str | None = None) -> bool:
  """Load the project ``.env`` file exactly once per process.

  Args:
      env_path: Optional explicit path; defaults to the repository root.

  Returns:
      True when a file was loaded (values are exported to os.environ).
  """
  global _ENV_LOADED  # pylint: disable=global-statement
  if _ENV_LOADED:
    return True
  target = Path(env_path) if env_path else _PROJECT_ROOT / '.env'
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


def get_secret(name: str, default: str | None = None) -> str | None:
  """Resolve a secret by name across all supported backends.

  Lookup order:

  1. Process environment (already set on CI / exported shells).
  2. Project ``.env`` (loaded lazily).
  3. Kaggle User Secrets (notebook runs).

  Args:
      name: Secret variable name, e.g. ``DISCORD_WEBHOOK_URL``.
      default: Value returned when every backend misses.

  Returns:
      The secret value or ``default``.
  """
  load_project_env()
  value = os.environ.get(name)
  if value:
    return value
  return _from_kaggle_secrets(name) or default
