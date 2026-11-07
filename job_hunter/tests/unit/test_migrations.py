#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for schema creation, idempotency, indexes, and FTS.'''


from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from job_hunter.core.db import MIGRATIONS_DIR, connect, run_migrations


def _names(db: Path, kind: str) -> set:
  '''Return sqlite_master names of one kind, closing the connection.'''
  with closing(connect(db, readonly=True)) as conn:
    return {
      row['name']
      for row in conn.execute(
        'SELECT name FROM sqlite_master WHERE type = ?', (kind,),
      )
    }


def _counts(db: Path, tables) -> dict:
  '''Return row counts for tables, closing the connection.'''
  with closing(connect(db, readonly=True)) as conn:
    return {
      table: conn.execute(
        f'SELECT COUNT(*) FROM {table}',
      ).fetchone()[0]
      for table in tables
    }


def _insert_job(conn: sqlite3.Connection, content_hash: str) -> None:
  '''Insert one minimal valid job row with the given hash.'''
  conn.execute(
    'INSERT INTO jobs (source_key, url, canonical_url, title,'
    ' description_text, raw_json, content_hash)'
    ' VALUES (?, ?, ?, ?, ?, ?, ?)',
    (
      'lever',
      f'https://jobs.example.com/{content_hash}',
      f'https://jobs.example.com/{content_hash}',
      'Staff Data Scientist',
      'python experimentation',
      '{}',
      content_hash,
    ),
  )


def test_migrations_create_core_tables(tmp_path: Path) -> None:
  '''Prove the full v1 table set (incl. FTS shadows) exists exactly.'''
  db = tmp_path / 'app.db'
  run_migrations(db)
  run_migrations(db)
  expected = {
    'candidates', 'candidate_profiles', 'resumes', 'skills',
    'skill_aliases', 'candidate_skills', 'companies', 'company_aliases',
    'sources', 'jobs', 'jobs_fts', 'job_skills', 'job_embeddings',
    'recommendations', 'runs', 'run_events', 'crawl_state', 'settings',
    'schema_migrations', 'notifications',
    'jobs_fts_config', 'jobs_fts_data', 'jobs_fts_docsize',
    'jobs_fts_idx',
  }
  assert _names(db, 'table') == expected
  assert _names(db, 'trigger') == {'jobs_ai', 'jobs_ad', 'jobs_au'}


def test_migrations_0002_indexes_exist(tmp_path: Path) -> None:
  '''Prove 0002 and 0001 indexes and applied migration rows exist.'''
  db = tmp_path / 'app.db'
  run_migrations(db)
  indexes = _names(db, 'index')
  for index in (
    'idx_jobs_city_status', 'idx_jobs_mode_status',
    'idx_recs_status_score', 'idx_companies_ats',
    'idx_jobs_company_posted', 'idx_jobs_status', 'idx_jobs_hash',
    'idx_recs_score', 'idx_events_run',
  ):
    assert index in indexes, index
  with closing(connect(db, readonly=True)) as conn:
    applied = {
      row['name']
      for row in conn.execute('SELECT name FROM schema_migrations')
    }
  # Parallel workers may add later migrations; require 0001+0002 and
  # that applied matches every migration file present on disk.
  on_disk = {p.name for p in MIGRATIONS_DIR.glob('*.sql')}
  assert {'0001_init.sql', '0002_perf.sql'}.issubset(applied)
  assert applied == on_disk


def test_migrations_double_run_preserves_row_counts(
  tmp_path: Path,
) -> None:
  '''Prove re-running migrations preserves seeded row counts.'''
  db = tmp_path / 'app.db'
  run_migrations(db)
  tables = ('sources', 'candidates', 'schema_migrations')
  before = _counts(db, tables)
  assert before['candidates'] == 1
  assert before['sources'] >= 16
  assert before['schema_migrations'] == len(
    list(MIGRATIONS_DIR.glob('*.sql')),
  )
  run_migrations(db)
  run_migrations(db)
  assert _counts(db, tables) == before


def test_jobs_unique_content_hash_enforced(tmp_path: Path) -> None:
  '''Prove a duplicate content_hash raises IntegrityError.'''
  db = tmp_path / 'app.db'
  run_migrations(db)
  with closing(connect(db)) as conn:
    _insert_job(conn, 'hash-dup')
    with pytest.raises(sqlite3.IntegrityError):
      _insert_job(conn, 'hash-dup')


def test_jobs_fts_insert_and_match(tmp_path: Path) -> None:
  '''Prove FTS triggers index jobs and MATCH finds only matches.'''
  db = tmp_path / 'app.db'
  run_migrations(db)
  with closing(connect(db)) as conn:
    _insert_job(conn, 'hash-fts')
    rows = conn.execute(
      'SELECT rowid FROM jobs_fts WHERE jobs_fts MATCH ?',
      ('experimentation',),
    ).fetchall()
    assert len(rows) == 1
    miss = conn.execute(
      'SELECT rowid FROM jobs_fts WHERE jobs_fts MATCH ?',
      ('nonexistentterm',),
    ).fetchall()
    assert miss == []


def test_sources_seeded(tmp_path: Path) -> None:
  '''Prove the source registry contains the built-in adapters.'''
  db = tmp_path / 'app.db'
  run_migrations(db)
  with closing(connect(db, readonly=True)) as conn:
    keys = {row['key'] for row in conn.execute('SELECT key FROM sources')}
  builtins = {'greenhouse', 'lever', 'ashby', 'workable', 'manual'}
  assert builtins.issubset(keys)
