#!/usr/bin/env python
# -- coding: utf-8 --

'''FastAPI dependency providers.'''


from __future__ import annotations

from functools import lru_cache

from fastapi import Request
from job_hunter.core.config import AppSettings


def get_settings(request: Request) -> AppSettings:
  '''Return the app-scoped settings instance.

  Args:
    request: Current request.

  Returns:
    Application settings.
  '''
  return request.app.state.settings


@lru_cache(maxsize=1)
def default_config_path() -> str:
  '''Resolve the default config path from this file's location.

  Returns:
    Path string to config/app.yaml.
  '''
  from pathlib import Path
  return str(Path(__file__).resolve().parents[4] / 'config' / 'app.yaml')
