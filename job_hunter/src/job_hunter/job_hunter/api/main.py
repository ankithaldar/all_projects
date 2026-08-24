#!/usr/bin/env python
# -- coding: utf-8 --

'''FastAPI application factory with static UI mounting.'''


from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from job_hunter.api.deps import default_config_path
from job_hunter.core.bootstrap import bootstrap

logger = logging.getLogger(__name__)


def create_app(config_path: Optional[str] = None) -> FastAPI:
  '''Build the configured FastAPI application.

  Args:
    config_path: Optional app.yaml override; defaults to repo layout.

  Returns:
    Ready FastAPI instance.
  '''
  settings = bootstrap(config_path or default_config_path())

  @asynccontextmanager
  async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    '''Emit a startup marker.

    Args:
      app: The running app.
    '''
    logger.info('job_hunter api ready (data=%s)', settings.data_dir)
    yield

  app = FastAPI(title='Job Hunter', version='0.1.0', lifespan=lifespan)
  app.state.settings = settings

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
    return JSONResponse(
      status_code=500,
      content={'title': 'internal error', 'status': 500, 'detail': str(exc)[:300]},
    )

  web_dir = Path(__file__).resolve().parents[4] / 'web'
  if web_dir.exists():
    app.mount('/', StaticFiles(directory=str(web_dir), html=True), name='web')
  return app


