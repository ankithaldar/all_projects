#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''FastAPI dependency providers.'''


from __future__ import annotations

from functools import lru_cache
from secrets import compare_digest

from fastapi import HTTPException, Request
from job_hunter.core.config import AppSettings


def get_settings(request: Request) -> AppSettings:
  '''Return the app-scoped settings instance.

  Args:
    request: Current request.

  Returns:
    Application settings.
  '''
  return request.app.state.settings


def require_token(request: Request) -> None:
  '''Require Bearer token when JH_API_TOKEN is configured.

  Args:
    request: Current request.

  Raises:
    HTTPException: 401 when a token is configured and the header is missing
      or does not match.
  '''
  settings: AppSettings = request.app.state.settings
  token = getattr(settings, 'api_token', '') or ''
  if not token:
    return
  authorization = request.headers.get('authorization') or ''
  expected = f'Bearer {token}'
  if not compare_digest(authorization, expected):
    raise HTTPException(status_code=401, detail='unauthorized')


@lru_cache(maxsize=1)
def default_config_path() -> str:
  '''Resolve the default config path from this file's location.

  Returns:
    Path string to config/app.yaml.
  '''
  from pathlib import Path
  return str(Path(__file__).resolve().parents[4] / 'config' / 'app.yaml')
