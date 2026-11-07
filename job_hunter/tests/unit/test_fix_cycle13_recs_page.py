#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle 13 micro-fix: recommendations page bound (mirror jobs le=10000).

Route accepts page up to 2**63-1, repo computes OFFSET (page-1)*per_page
-> SQLite OverflowError (500) at page=2**63-1. Fix: route le=10000 and
repo-side page clamp so direct repo calls return [] instead of raising.
'''

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip('fastapi')
TestClient = pytest.importorskip('fastapi.testclient').TestClient

_MAX_I64 = '9223372036854775807'


def _settings(tmp_path: Path, monkeypatch):
  '''Build migrated settings for direct repo tests.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest fixture.

  Returns:
    Migrated AppSettings.
  '''
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  config_path = tmp_path / 'app.yaml'
  config_path.write_text(
    'salary_hard_floor_lpa: 45\n', encoding='utf-8',
  )
  from job_hunter.core.config import AppSettings
  from job_hunter.core.db import run_migrations
  settings = AppSettings(config_path)
  settings.data_dir.mkdir(parents=True, exist_ok=True)
  run_migrations(settings.db_path)
  return settings


def _make_client(tmp_path: Path, monkeypatch) -> TestClient:
  '''Build an app client against a temporary data directory.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest monkeypatch fixture.

  Returns:
    Configured TestClient.
  '''
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  monkeypatch.setenv('APP_HOST', '127.0.0.1')
  monkeypatch.setenv('JH_API_TOKEN', '')
  monkeypatch.setenv('APP_LOG_LEVEL', 'INFO')
  config_path = tmp_path / 'app.yaml'
  config_path.write_text(
    'salary_hard_floor_lpa: 45\n', encoding='utf-8',
  )
  from job_hunter.api.main import create_app
  app = create_app(str(config_path))
  return TestClient(app, raise_server_exceptions=False)


def test_recommendations_huge_page_is_422_not_500(
  tmp_path: Path, monkeypatch,
) -> None:
  '''GET /api/recommendations?page=2**63-1 is rejected with 422.'''
  client = _make_client(tmp_path, monkeypatch)
  response = client.get(f'/api/recommendations?page={_MAX_I64}')
  assert response.status_code == 422, response.text


def test_recommendations_repo_huge_page_returns_empty(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Repo list_for_candidate with huge page returns [] not OverflowError.'''
  settings = _settings(tmp_path, monkeypatch)
  from job_hunter.db.repositories.recommendations import (
    RecommendationsRepository,
  )
  rows = RecommendationsRepository(settings.db_path).list_for_candidate(
    page=2**63 - 1,
  )
  assert rows == [], rows


def test_recommendations_page_bound_matches_jobs(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Page 10001 is 422 (mirrors jobs le=10000); page 10000 is not 422.'''
  client = _make_client(tmp_path, monkeypatch)
  over = client.get('/api/recommendations?page=10001')
  assert over.status_code == 422, over.text
  at_bound = client.get('/api/recommendations?page=10000')
  assert at_bound.status_code != 422, at_bound.text
