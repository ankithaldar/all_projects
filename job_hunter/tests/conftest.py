#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Shared hermetic fixtures for the test suite.

Fixtures are function-scoped, use tmp_path only, and perform no
network access. Test-only helpers live here, not in product code.
'''


from __future__ import annotations

from pathlib import Path

import pytest
from job_hunter.api.main import create_app
from job_hunter.core.config import AppSettings
from job_hunter.core.db import run_migrations


@pytest.fixture()
def app_settings(tmp_path: Path, monkeypatch) -> object:
  '''Return AppSettings backed by a tmp data dir and minimal yaml.'''
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  config_path = tmp_path / 'app.yaml'
  config_path.write_text(
    'salary_hard_floor_lpa: 45\n', encoding='utf-8',
  )
  return AppSettings(config_path)


@pytest.fixture()
def db_path(  # pylint: disable=redefined-outer-name
  tmp_path: Path, app_settings: object,
) -> Path:
  '''Return a freshly migrated tmp database path.'''
  _ = app_settings
  db = tmp_path / 'app.db'
  run_migrations(db)
  return db


@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
  '''Return a TestClient for create_app with tmp settings.'''
  pytest.importorskip('fastapi')
  test_client = pytest.importorskip('fastapi.testclient').TestClient
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  config_path = tmp_path / 'app.yaml'
  config_path.write_text(
    'salary_hard_floor_lpa: 45\n', encoding='utf-8',
  )
  app = create_app(str(config_path))
  client = test_client(app)
  yield client
  client.close()
