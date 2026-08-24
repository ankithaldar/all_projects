#!/usr/bin/env python
# -- coding: utf-8 --

'''Smoke tests for schema creation and the migration runner.'''


from __future__ import annotations

from pathlib import Path

from job_hunter.core.db import connect, run_migrations


def test_migrations_create_core_tables(tmp_path: Path) -> None:
  '''All v1 tables exist after migrations run once and only once.

  Args:
    tmp_path: Pytest temporary directory.
  '''
  db = tmp_path / 'app.db'
  run_migrations(db)
  run_migrations(db)
  names = {
    row['name']
    for row in connect(db, readonly=True).execute(
      "SELECT name FROM sqlite_master WHERE type = 'table'",
    )
  }
  expected = {
    'candidates', 'candidate_profiles', 'resumes', 'skills', 'skill_aliases',
    'candidate_skills', 'companies', 'company_aliases', 'sources', 'jobs',
    'jobs_fts', 'job_skills', 'job_embeddings', 'recommendations', 'runs',
    'run_events', 'crawl_state', 'settings', 'schema_migrations',
  }
  assert expected.issubset(names)


def test_sources_seeded(tmp_path: Path) -> None:
  '''Source registry contains the built-in adapters.

  Args:
    tmp_path: Pytest temporary directory.
  '''
  db = tmp_path / 'app.db'
  run_migrations(db)
  keys = {
    row['key']
    for row in connect(db, readonly=True).execute('SELECT key FROM sources')
  }
  assert {'greenhouse', 'lever', 'ashby', 'workable', 'manual'}.issubset(keys)
