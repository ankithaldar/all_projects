#!/usr/bin/env python
# -- coding: utf-8 --

'''API smoke tests via TestClient.'''


from __future__ import annotations

import os
from pathlib import Path

import pytest
fastapi = pytest.importorskip('fastapi')
TestClient = pytest.importorskip('fastapi.testclient').TestClient


@pytest.fixture()
def client(tmp_path: Path, monkeypatch) -> TestClient:
  '''Build the app against a temporary data directory.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest fixture.

  Returns:
    Configured TestClient.
  '''
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  config_path = tmp_path / 'app.yaml'
  config_path.write_text('salary_hard_floor_lpa: 45\n', encoding='utf-8')
  from job_hunter.api.main import create_app
  app = create_app(str(config_path))
  return TestClient(app)


def test_healthz(client: TestClient) -> None:
  '''Health probe reports ok with databases initialized.'''
  response = client.get('/healthz')
  assert response.status_code == 200
  body = response.json()
  assert body['status'] == 'ok'
  assert body['db'] is True


def test_profile_roundtrip(client: TestClient) -> None:
  '''Profile defaults then update persist correctly.'''
  first = client.get('/api/profile').json()
  assert first['target_roles'] == ['Data Scientist']
  updated = client.put('/api/profile', json={
    'target_roles': ['Staff Data Scientist'],
    'cities': ['Bengaluru', 'Remote'],
    'remote_pref': 'any',
    'relocate_ok': True,
    'experience_years': 9,
    'summary': 'DS leader',
  })
  assert updated.status_code == 200
  saved = client.get('/api/profile').json()
  assert saved['target_roles'] == ['Staff Data Scientist']
  assert saved['cities'] == ['Bengaluru', 'Remote']
  assert saved['version'] >= 1


def test_recommendations_empty_and_runs_conflict(client: TestClient) -> None:
  '''Empty recommendation list and run trigger flow work.'''
  assert client.get('/api/recommendations').json() == []
  first = client.post('/api/runs/discovery')
  assert first.status_code == 200
  second = client.post('/api/runs/discovery')
  assert second.status_code == 409


def test_settings_roundtrip(client: TestClient) -> None:
  '''Settings persist arbitrary JSON values.'''
  put = client.put('/api/settings?key=scoring_weights', json={
    'skills_must': 0.4,
  })
  assert put.status_code == 200
  stored = client.get('/api/settings').json()
  assert stored['scoring_weights']['skills_must'] == 0.4
