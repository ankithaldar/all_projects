#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''FastAPI application factory with static UI mounting.'''


from __future__ import annotations

import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from job_hunter.api.deps import default_config_path, require_token
from job_hunter.core.bootstrap import bootstrap
from job_hunter.core.errors import (
  AdapterError,
  AppError,
  ConflictError,
  DatabaseError,
  GatewayUnavailableError,
)
# Gateway errors are wrapped at llm/client.py into
# core.errors.GatewayUnavailableError; there is no
# gateway errors module class to fall back to.

logger = logging.getLogger(__name__)

_ALLOWED_ORIGIN_HOSTS = {'127.0.0.1', 'localhost', 'testserver'}


def _origin_host(value: str) -> str:
  '''Extract the hostname from an Origin or Referer value.

  Args:
    value: Header value such as 'https://example.com/path'.

  Returns:
    Lowercase hostname, or empty string when absent.
  '''
  return (urlparse(value).hostname or '').lower()


def _host_header_host(value: str) -> str:
  '''Extract the hostname from a Host header value.

  Args:
    value: Host header value, possibly 'host:port'.

  Returns:
    Lowercase hostname without port or brackets.
  '''
  host = value.strip().lower()
  if host.startswith('['):
    closing = host.find(']')
    return host[1:closing] if closing > 0 else host
  return host.split(':', 1)[0]


def _sanitize_nonfinite(value: Any) -> Any:
  '''Replace non-finite floats with strings for JSON serialization.

  Args:
    value: Value possibly containing inf/nan floats.

  Returns:
    Sanitized value with non-finite floats replaced by str.
  '''
  if isinstance(value, float):
    return value if math.isfinite(value) else str(value)
  if isinstance(value, dict):
    return {
      key: _sanitize_nonfinite(item) for key, item in value.items()
    }
  if isinstance(value, (list, tuple)):
    return [_sanitize_nonfinite(item) for item in value]
  return value


def create_app(config_path: Optional[str] = None) -> FastAPI:
  '''Build the configured FastAPI application.

  Args:
    config_path: Optional app.yaml override; defaults to repo layout.

  Returns:
    Ready FastAPI instance.
  '''
  settings = bootstrap(config_path or default_config_path())
  if not settings.is_loopback_host() and not settings.api_token:
    raise RuntimeError('non-loopback host requires JH_API_TOKEN')

  @asynccontextmanager
  async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    '''Emit a startup marker.

    Args:
      app: The running app.

    Yields:
      Nothing; startup marker only.
    '''
    logger.info('job_hunter api ready (data=%s)', settings.data_dir)
    yield

  app = FastAPI(
    title='Job Hunter', version='0.1.0', lifespan=lifespan,
    dependencies=[Depends(require_token)],
  )
  app.state.settings = settings
  app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=['127.0.0.1', 'localhost', 'testserver'],
  )

  @app.middleware('http')
  async def csrf_origin_check(request: Request, call_next):
    '''Reject cross-origin non-GET API requests.

    Args:
      request: Current request.
      call_next: Next handler.

    Returns:
      Response, or 403 for a forbidden origin.
    '''
    if request.method != 'GET' and request.url.path.startswith('/api/'):
      host_host = _host_header_host(request.headers.get('host', ''))
      for header in ('origin', 'referer'):
        raw = request.headers.get(header)
        if not raw:
          continue
        origin_host = _origin_host(raw)
        if origin_host in _ALLOWED_ORIGIN_HOSTS:
          continue
        if host_host and origin_host == host_host:
          continue
        return JSONResponse(
          status_code=403, content={'detail': 'forbidden origin'},
        )
    return await call_next(request)

  @app.middleware('http')
  async def security_headers(request: Request, call_next):
    '''Attach security headers; CSP only for API JSON responses.

    Args:
      request: Current request.
      call_next: Next handler.

    Returns:
      Response with security headers.
    '''
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    if request.url.path.startswith('/api/'):
      response.headers['Content-Security-Policy'] = "default-src 'self'"
    return response

  from job_hunter.api.routes.companies import router as companies_router
  from job_hunter.api.routes.jobs import router as jobs_router
  from job_hunter.api.routes.profiles import router as profiles_router
  from job_hunter.api.routes.recommendations import router as recs_router
  from job_hunter.api.routes.runs import router as runs_router
  from job_hunter.api.routes.settings import router as settings_router

  app.include_router(profiles_router)
  app.include_router(companies_router)
  app.include_router(jobs_router)
  app.include_router(recs_router)
  app.include_router(runs_router)
  app.include_router(settings_router)

  @app.get('/healthz')
  def healthz() -> dict:
    '''Liveness probe including storage status.

    Returns:
      Status mapping.
    '''
    return {
      'status': 'ok',
      'db': settings.db_path.exists(),
      'gateway_db': settings.gateway_db_path.exists(),
      'version': app.version,
    }

  @app.exception_handler(RequestValidationError)
  async def request_validation(
    request: Request, exc: RequestValidationError,
  ) -> JSONResponse:
    '''Return sanitized validation errors as 422 JSON.

    Args:
      request: Current request.
      exc: Raised validation error.

    Returns:
      JSON error response with non-finite floats stringified.
    '''
    errors = _sanitize_nonfinite(jsonable_encoder(exc.errors()))
    return JSONResponse(status_code=422, content={'detail': errors})

  @app.exception_handler(AppError)
  async def app_error(request: Request, exc: AppError) -> JSONResponse:
    '''Map application errors to generic status codes.

    Args:
      request: Current request.
      exc: Raised application error.

    Returns:
      JSON error response with a generic detail per class.
    '''
    if isinstance(exc, ConflictError):
      status, detail, title = 409, 'conflict', 'Conflict'
    elif isinstance(exc, DatabaseError):
      status, detail, title = 500, 'database error', 'Internal Server Error'
    elif isinstance(exc, AdapterError):
      status, detail, title = 502, 'adapter error', 'Bad Gateway'
    elif isinstance(exc, GatewayUnavailableError):
      status = 503
      detail = 'gateway unavailable'
      title = 'Service Unavailable'
    else:
      status, detail, title = 500, 'internal error', 'Internal Server Error'
    logger.exception('application error on %s', request.url.path)
    return JSONResponse(
      status_code=status,
      content={'title': title, 'status': status, 'detail': detail},
    )

  @app.exception_handler(Exception)
  async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    '''Convert unexpected errors to problem-style JSON.

    Args:
      request: Current request.
      exc: Raised exception.

    Returns:
      JSON error response.
    '''
    logger.exception('unhandled error on %s', request.url.path)
    content = {
      'title': 'Internal Server Error',
      'status': 500,
      'detail': 'internal error',
    }
    return JSONResponse(status_code=500, content=content)

  web_dir = Path(__file__).resolve().parents[4] / 'web'
  if web_dir.exists():
    app.mount('/', StaticFiles(directory=str(web_dir), html=True), name='web')
  return app
