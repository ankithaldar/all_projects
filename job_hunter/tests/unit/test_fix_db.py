#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''RED tests for W2-db database fixes (one per numbered spec item).'''

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from job_hunter.core.db import fetch_all, fetch_one, run_migrations, session
from job_hunter.db.repositories.companies import CompaniesRepository
from job_hunter.db.repositories.jobs import JobsRepository
from job_hunter.db.repositories.profile import ProfileRepository
from job_hunter.db.repositories.recommendations import (
  RecommendationsRepository,
)
from job_hunter.db.repositories.runs import RunsRepository
from job_hunter.db.repositories.settings import GatewayStatsRepository
from job_hunter.db.repositories.taxonomy import TaxonomyRepository
from job_hunter.services.skills_taxonomy import SkillsTaxonomy


def _db(tmp_path: Path) -> Path:
  db = tmp_path / 'app.db'
  run_migrations(db)
  return db


def _job(repo, suffix: str) -> int:
  payload = {
    'source_key': 'lever',
    'url': f'https://j.example/{suffix}',
    'canonical_url': f'https://j.example/{suffix}',
    'title': f'Job {suffix}',
    'raw_json': '{}',
    'content_hash': f'hash-{suffix}',
    'description_text': 'data scientist python',
  }
  return int(repo.insert(payload))


def test_recommendations_upsert_rank_is_position(tmp_path: Path) -> None:

  db = _db(tmp_path)
  jobs = JobsRepository(db)
  job1 = _job(jobs, 'r1')
  job2 = _job(jobs, 'r2')
  run_id = RunsRepository(db).create('discovery', 'test')
  recs = RecommendationsRepository(db)
  recs.upsert_many(
    run_id,
    1,
    [
      {
        'job_id': job1,
        'total_score': 90.0,
        'gate_pass': True,
        'rank': 42,
        'gate_failures': [],
        'breakdown': {},
        'rationale': 'a',
      },
      {
        'job_id': job2,
        'total_score': 80.0,
        'gate_pass': True,
        'rank': 99,
        'gate_failures': [],
        'breakdown': {},
        'rationale': 'b',
      },
    ],
  )
  # SQLite tolerates duplicate INSERT columns (first wins), so the
  # duplicate-rank bug is also asserted at the SQL-source level: the
  # INSERT column list must contain 'rank' exactly once.
  source = (
    Path(__file__).resolve().parents[2]
    / 'src'
    / 'job_hunter'
    / 'job_hunter'
    / 'db'
    / 'repositories'
    / 'recommendations.py'
  ).read_text(encoding='utf-8')
  insert_sql = source.split('INSERT INTO recommendations', 1)[1].split(
    'ON CONFLICT', 1
  )[0]
  assert insert_sql.count('rank') == 1
  rows = recs.list_for_candidate(1)
  assert [row['rank'] for row in rows] == [1, 2]
  with closing(sqlite3.connect(str(db))) as conn:
    stored = [
      r[0]
      for r in conn.execute(
        'SELECT rank FROM recommendations ORDER BY total_score DESC'
      )
    ]
  assert stored == [1, 2]


def test_recommendations_counts(tmp_path: Path) -> None:

  db = _db(tmp_path)
  job_id = _job(JobsRepository(db), 'counts1')
  run_id = RunsRepository(db).create('discovery', 'test')
  recs = RecommendationsRepository(db)
  recs.upsert_many(
    run_id,
    1,
    [
      {
        'job_id': job_id,
        'total_score': 10.0,
        'gate_pass': True,
        'gate_failures': [],
        'breakdown': {},
        'rationale': '',
      }
    ],
  )
  counts = recs.counts(1)
  assert counts['new_today'] == 1
  assert counts['active_jobs'] == 1


def test_session_rolls_back_on_exception(tmp_path: Path) -> None:

  db = _db(tmp_path)
  with pytest.raises(RuntimeError), session(db) as conn:
    conn.execute(
      'INSERT INTO companies (name, normalized_name) '
      "VALUES ('Atomic Co', 'atomic co')"
    )
    raise RuntimeError('boom')
  with closing(sqlite3.connect(str(db))) as conn:
    count = conn.execute(
      "SELECT COUNT(*) FROM companies WHERE normalized_name = 'atomic co'"
    ).fetchone()[0]
  assert count == 0


def test_jobs_insert_drops_evil_column(tmp_path: Path) -> None:

  db = _db(tmp_path)
  repo = JobsRepository(db)
  payload = {
    'source_key': 'lever',
    'url': 'https://j.example/evil',
    'canonical_url': 'https://j.example/evil',
    'title': 'Safe title',
    'raw_json': '{}',
    'content_hash': 'hash-evil',
    'evil"; DROP TABLE jobs; --': 'pwned',
    'title) VALUES (1); DROP TABLE jobs; --': 'pwned',
  }
  job_id = repo.insert(payload)
  assert job_id > 0
  row = repo.get(job_id)
  assert row is not None and row['title'] == 'Safe title'
  with closing(sqlite3.connect(str(db))) as conn:
    assert (
      conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='jobs'"
      ).fetchone()[0]
      == 1
    )


def test_profile_save_drops_skills_resolved(tmp_path: Path) -> None:

  repo = ProfileRepository(_db(tmp_path))
  new_id = repo.save_profile({'skills_resolved': 3, 'summary': 'hello world'})
  assert new_id > 0
  profile = repo.get_profile(1)
  assert profile is not None and profile['summary'] == 'hello world'
  assert 'skills_resolved' not in profile


def test_migration_idempotent_and_0003_indexes(tmp_path: Path) -> None:
  db = _db(tmp_path)
  run_migrations(db)  # second run is a no-op
  with closing(sqlite3.connect(str(db))) as conn:
    indexes = {
      row[0]: (row[0], row[1] or '')
      for row in conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='index'"
      )
    }
    sources = {r[0] for r in conn.execute('SELECT key FROM sources')}
    runs_cols = {r[1] for r in conn.execute('PRAGMA table_info(runs)')}
  expected = {
    'idx_jobs_canonical_url',
    'idx_jobs_source_external',
    'idx_candidate_profiles_candidate_version',
    'idx_companies_status_vertical_priority',
    'idx_runs_status_created',
  }
  assert expected.issubset(set(indexes))
  for name in expected:
    assert 'UNIQUE' not in indexes[name][1].upper(), name
  assert 'workday' in sources
  assert 'created_at' in runs_cols


def test_save_resume_fk_raises_value_error(tmp_path: Path) -> None:

  repo = ProfileRepository(_db(tmp_path))
  with pytest.raises(ValueError, match='candidate not found'):
    repo.save_resume(
      '/tmp/x.pdf', 'sha-fk', 'application/pdf', True, candidate_id=999
    )


def test_save_resume_duplicate_sha_returns_existing(tmp_path: Path) -> None:

  repo = ProfileRepository(_db(tmp_path))
  first = repo.save_resume('/tmp/a.pdf', 'sha-dup', 'application/pdf', True)
  second = repo.save_resume('/tmp/b.pdf', 'sha-dup', 'application/pdf', True)
  assert first == second


def test_list_jobs_empty_fts_no_crash(tmp_path: Path) -> None:

  repo = JobsRepository(_db(tmp_path))
  _job(repo, 'fts1')
  rows = repo.list_jobs(q='   ')
  assert len(rows) == 1
  rows = repo.list_jobs(q='')
  assert len(rows) == 1


def test_mark_orphans_failed_age_gates_pending(tmp_path: Path) -> None:

  db = _db(tmp_path)
  repo = RunsRepository(db)
  fresh = repo.create('discovery', 'test')
  assert repo.mark_orphans_failed() == []
  assert repo.get(fresh)['status'] == 'pending'
  with closing(sqlite3.connect(str(db))) as conn:
    conn.execute(
      "UPDATE runs SET created_at = datetime('now', '-2 hours') WHERE id = ?",
      (fresh,),
    )
    conn.commit()
  assert repo.mark_orphans_failed() == [fresh]
  assert repo.get(fresh)['status'] == 'failed'


def test_mark_orphans_failed_still_fails_old_running(tmp_path: Path) -> None:

  db = _db(tmp_path)
  repo = RunsRepository(db)
  run_id = repo.create('discovery', 'test')
  with closing(sqlite3.connect(str(db))) as conn:
    conn.execute(
      "UPDATE runs SET status = 'running', "
      "started_at = datetime('now', '-3 hours') WHERE id = ?",
      (run_id,),
    )
    conn.commit()
  assert repo.mark_orphans_failed() == [run_id]


def test_companies_upsert_same_name_same_id(tmp_path: Path) -> None:

  repo = CompaniesRepository(_db(tmp_path))
  first = repo.upsert('Acme Technologies', vertical='fintech')
  second = repo.upsert('Acme Technologies', vertical='saas')
  assert first == second
  row = repo.get(first)
  assert row is not None and row['vertical'] == 'saas'
  by_domain = repo.upsert('Other Name', domain='acme.example', vertical='x')
  assert repo.upsert('Another Name', domain='acme.example') == by_domain


def test_fetch_helpers(tmp_path: Path) -> None:

  db = _db(tmp_path)
  with fetch_one(
    db, 'SELECT key FROM sources WHERE key = ?', ('lever',)
  ) as row:
    assert row is not None and row['key'] == 'lever'
  with fetch_all(
    db, 'SELECT key FROM sources WHERE key = ?', ('lever',)
  ) as rows:
    assert [r['key'] for r in rows] == ['lever']
  with fetch_one(db, 'SELECT key FROM sources WHERE key = ?', ('nope',)) as row:
    assert row is None


def test_jobs_job_skill_rows(tmp_path: Path) -> None:

  db = _db(tmp_path)
  taxonomy = SkillsTaxonomy(db)
  taxonomy.load_seed({'skills': {'Python': {'aliases': ['py']}}})
  skill_id = taxonomy.resolve('py')
  assert skill_id is not None
  repo = JobsRepository(db)
  job_id = _job(repo, 'skills1')
  repo.attach_skill_rows(job_id, [{'skill_id': skill_id, 'kind': 'must_have'}])
  rows = repo.job_skill_rows([job_id])
  assert len(rows) == 1
  assert rows[0]['skill_name'] == 'Python'
  assert not repo.job_skill_rows([])


def test_gateway_stats_repository(tmp_path: Path) -> None:

  gateway = tmp_path / 'gateway.db'
  with closing(sqlite3.connect(str(gateway))) as conn:
    conn.execute(
      'CREATE TABLE llm_calls (provider TEXT, session_id TEXT, '
      'timestamp TEXT, input_tokens INTEGER, output_tokens INTEGER, '
      'cost REAL)'
    )
    conn.execute(
      "INSERT INTO llm_calls VALUES ('p1', 's1', datetime('now'), 10, 5, 0.5)"
    )
    conn.execute(
      "INSERT INTO llm_calls VALUES ('p1', 's2', datetime('now'), 20, 5, 1.5)"
    )
    conn.commit()
  stats = GatewayStatsRepository().llm_stats(gateway)
  assert stats['totals']['calls'] == 2
  assert stats['totals']['tokens'] == 40
  assert stats['by_provider'][0]['provider'] == 'p1'
  missing = GatewayStatsRepository().llm_stats(tmp_path / 'missing.db')
  assert missing['totals'] == {'calls': 0, 'tokens': 0, 'cost': 0.0}


def test_taxonomy_repository(tmp_path: Path) -> None:

  db = _db(tmp_path)
  taxonomy = SkillsTaxonomy(db)
  taxonomy.load_seed({'skills': {'Python': {'aliases': ['py']}}})
  repo = TaxonomyRepository(db)
  rows = repo.skill_rows()
  assert [r['canonical_name'] for r in rows] == ['Python']
  assert repo.resolve_alias('PY') == repo.resolve_alias('python')
  assert repo.resolve_alias('python') is not None
  assert repo.resolve_alias('unknown-xyz') is None
