#!/usr/bin/env python
# -- coding: utf-8 --

'''API entrypoint: python -m job_hunter.api.'''


import uvicorn

from job_hunter.core.config import AppSettings


def _app_root():
  '''Return the project root directory.

  Returns:
    Path object.
  '''
  from pathlib import Path
  return Path(__file__).resolve().parents[4]


if __name__ == '__main__':
  settings = AppSettings(_app_root() / 'config' / 'app.yaml')
  uvicorn.run(
    'job_hunter.api.main:create_app',
    host=settings.host,
    port=settings.port,
    log_level='warning',
    factory=True,
  )
