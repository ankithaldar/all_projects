#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Secret management: .env locally, Kaggle Secrets inside kernels.

Priority order for a key ``K``:
1. already present in os.environ (kernel env vars, CI, ...)
2. ``.env`` file at project root (python-dotenv)
3. Kaggle Secrets (UserSecretsClient) -- wrapped in try/except because it
   only exists on Kaggle and requires the secret to be attached.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

from dotenv import load_dotenv

_ENV_LOADED = False


def _try_load_dotenv() -> None:
  global _ENV_LOADED
  if _ENV_LOADED:
    return
  for candidate in (
    Path.cwd() / '.env',
    Path(__file__).resolve().parents[3] / '.env',
  ):
    if candidate.exists():
      load_dotenv(candidate, override=False)
      break
  _ENV_LOADED = True


def _kaggle_secret(key: str) -> str | None:
  try:  # pragma: no cover - only importable inside Kaggle kernels
    # pylint: disable=import-outside-toplevel
    from kaggle_secrets import UserSecretsClient  # type: ignore

    return UserSecretsClient().get_secret(key)
  except Exception:  # pylint: disable=broad-exception-caught
    return None


@cache
def get_secret(key: str, default: str | None = None) -> str | None:
  """Resolve a secret with the documented priority order."""
  _try_load_dotenv()
  value = os.environ.get(key)
  if value is not None:
    return value
  return _kaggle_secret(key) or default


def load_env() -> None:
  """Idempotently hydrate os.environ from .env; called by config loader."""
  _try_load_dotenv()
