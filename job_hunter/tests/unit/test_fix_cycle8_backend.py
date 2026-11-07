#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle 8 backend polish: quote reuse, priority 0, dangling resume,
taxonomy guard, score int guard, dashboard send_error.'''

from __future__ import annotations

import asyncio
import hashlib
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

fastapi = pytest.importorskip('fastapi')
TestClient = pytest.importorskip('fastapi.testclient').TestClient

SRC = (
  Path(__file__).resolve().parents[2]
  / 'src'
  / 'job_hunter'
  / 'job_hunter'
)


def _settings(tmp_path: Path, monkeypatch):
  '''Build migrated settings for direct service tests.

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


def _make_client(
  tmp_path: Path, monkeypatch, *, token: str = '',
) -> TestClient:
  '''Build an app client against a temporary data directory.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest monkeypatch fixture.
    token: Bearer token to configure.

  Returns:
    Configured TestClient.
  '''
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  monkeypatch.setenv('APP_HOST', '127.0.0.1')
  monkeypatch.setenv('JH_API_TOKEN', token)
  monkeypatch.setenv('APP_LOG_LEVEL', 'INFO')
  config_path = tmp_path / 'app.yaml'
  config_path.write_text(
    'salary_hard_floor_lpa: 45\n', encoding='utf-8',
  )
  from job_hunter.api.main import create_app
  app = create_app(str(config_path))
  return TestClient(app, raise_server_exceptions=False)


def _make_curator(settings):
  '''Build a curator with a stub client (no gateway network).

  Args:
    settings: Application settings.

  Returns:
    ProfileCurator instance.
  '''
  from job_hunter.services.profile_curator import ProfileCurator
  return ProfileCurator(settings, client=object())


def _parsed_ok(path) -> dict:
  '''Fake successful parse result for a written file.

  Args:
    path: Written resume file path.

  Returns:
    Parse mapping with ok True.
  '''
  data = Path(path).read_bytes()
  return {
    'text': 'python data scientist experience',
    'pages': 1,
    'sha256': hashlib.sha256(data).hexdigest(),
    'ok': True,
    'truncated': False,
  }


def _parsed_fail(path) -> dict:
  '''Fake failed parse result for a written file.

  Args:
    path: Written resume file path.

  Returns:
    Parse mapping with ok False.
  '''
  data = Path(path).read_bytes()
  return {
    'text': '',
    'pages': 0,
    'sha256': hashlib.sha256(data).hexdigest(),
    'ok': False,
    'truncated': False,
  }


def _extraction():
  '''Build a fake extraction result.

  Returns:
    ProfileExtraction instance.
  '''
  from job_hunter.services.profile_curator import ProfileExtraction
  return ProfileExtraction(
    skills=['Python'],
    summary='Experienced data scientist.',
    current_title='Data Scientist',
    experience_years=5.0,
    seniority_keywords=['senior'],
  )


def _configure_curator_ok(curator):
  '''Stub parser, skills, and extract for a successful ingest.

  Args:
    curator: ProfileCurator instance.
  '''
  curator._parser.parse_file = _parsed_ok
  curator._persist_skills = lambda raw, cid: []

  async def fake_extract(_text):
    '''Return a fixed extraction.

    Args:
      _text: Resume text.

    Returns:
      Fake extraction.
    '''
    return _extraction()

  curator.extract = fake_extract  # type: ignore[method-assign]


# 1. quote_identifier reuse ---------------------------------------------------


def test_quote_identifier_used_in_profile_and_db(tmp_path: Path) -> None:
  '''profile.py and core/db.py use shared quote_identifier, no inline.'''
  profile_text = (SRC / 'db' / 'repositories' / 'profile.py').read_text(
    encoding='utf-8',
  )
  db_text = (SRC / 'core' / 'db.py').read_text(encoding='utf-8')
  # Shared helper is imported in profile.py and called in both files.
  assert 'quote_identifier' in profile_text, profile_text
  # Inline concatenation patterns must be gone from both sites.
  assert '\'"\' + c.replace' not in profile_text, profile_text
  assert '\'"\' + table.replace' not in db_text, db_text
  # table_columns body uses the helper (behavior identical).
  table_cols = db_text.split('def table_columns', 1)[1].split(
    'def filter_to_columns', 1,
  )[0]
  assert 'quote_identifier' in table_cols, table_cols
  profile_save = profile_text.split('def save_profile', 1)[1].split(
    'def save_resume_ex', 1,
  )[0]
  assert 'quote_identifier' in profile_save, profile_save
  # Behavior identical: a profile save still round-trips.
  from job_hunter.core.db import run_migrations
  from job_hunter.db.repositories.profile import ProfileRepository
  db = tmp_path / 'app.db'
  run_migrations(db)
  repo = ProfileRepository(db)
  new_id = repo.save_profile({'summary': 'quote reuse works'})
  assert new_id > 0
  profile = repo.get_profile(1)
  assert profile is not None and profile['summary'] == 'quote reuse works'


# 2. dead _valid_domain --------------------------------------------------------


def test_valid_domain_wrapper_removed_cycle8() -> None:
  '''Dead _valid_domain is gone; live _validate_domain remains.'''
  from job_hunter.api.routes import companies as companies_route
  assert hasattr(companies_route, '_validate_domain')
  assert not hasattr(companies_route, '_valid_domain'), (
    'dead _valid_domain wrapper must be removed'
  )


# 3. base.py docstring single-source -------------------------------------------


def test_base_docstring_single_source_cycle8() -> None:
  '''base.py states single-source truth, no stale own-copy note.'''
  text = (SRC / 'adapters' / 'base.py').read_text(encoding='utf-8')
  assert 'API keeps its own copy' not in text
  assert 'API wraps valid_domain' in text
  assert 'Single-source truth' in text


# 4. priority 0 -> clamp to 1 (None/missing -> 3) ------------------------------


def test_company_import_priority_zero_clamps_to_one(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Import: priority 0 stored 1, null/missing stored 3.'''
  client = _make_client(tmp_path, monkeypatch)
  yaml_text = (
    'companies:\n'
    '  - name: Zero Co\n'
    '    domain: zeroco.example\n'
    '    priority: 0\n'
    '  - name: Null Co\n'
    '    domain: nullco.example\n'
    '    priority:\n'
    '  - name: Missing Co\n'
    '    domain: missingco.example\n'
  )
  response = client.post('/api/companies/import', json=yaml_text)
  assert response.status_code == 200, response.text
  body = response.json()
  assert body['imported'] == 3, body
  assert body['errors'] == 0, body
  from job_hunter.db.repositories.companies import CompaniesRepository
  rows = {
    row['domain']: row
    for row in CompaniesRepository(
      tmp_path / 'app.db'
    ).list_companies(limit=10)
  }
  assert rows['zeroco.example']['priority'] == 1, rows
  assert rows['nullco.example']['priority'] == 3, rows
  assert rows['missingco.example']['priority'] == 3, rows


def test_seed_priority_zero_clamps_to_one(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Seed ingestion: priority 0 stored 1, null/missing stored 3.'''
  settings = _settings(tmp_path, monkeypatch)
  seeds = tmp_path / 'seeds'
  seeds.mkdir()
  (seeds / 'companies_test.yaml').write_text(
    'companies:\n'
    '  - name: Seed Zero\n'
    '    domain: seedzero.example\n'
    '    priority: 0\n'
    '  - name: Seed Null\n'
    '    domain: seednull.example\n'
    '    priority:\n'
    '  - name: Seed Missing\n'
    '    domain: seedmissing.example\n',
    encoding='utf-8',
  )
  from job_hunter.services.company_discovery import ingest_seeds
  count = asyncio.run(ingest_seeds(settings, seeds_dir=seeds))
  assert count == 3, count
  from job_hunter.db.repositories.companies import CompaniesRepository
  rows = {
    row['domain']: row
    for row in CompaniesRepository(settings.db_path).list_companies(limit=10)
  }
  assert rows['seedzero.example']['priority'] == 1, rows
  assert rows['seednull.example']['priority'] == 3, rows
  assert rows['seedmissing.example']['priority'] == 3, rows


# 5. get_resume None -> raise ---------------------------------------------------


def test_curator_raises_when_get_resume_none_unparseable(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Duplicate unparseable: get_resume None raises, no dangling return.'''
  settings = _settings(tmp_path, monkeypatch)
  curator = _make_curator(settings)
  curator._parser.parse_file = _parsed_fail
  first = asyncio.run(
    curator.ingest_upload('first.pdf', b'%PDF-unparseable-cycle8'),
  )
  assert first['resume_id'] == 1, first
  assert first['parsed_ok'] is False, first
  from job_hunter.db.repositories.profile import ProfileRepository
  monkeypatch.setattr(
    ProfileRepository, 'get_resume', lambda self, rid: None,
  )
  with pytest.raises(RuntimeError, match='resume row missing'):
    asyncio.run(
      curator.ingest_upload('second.pdf', b'%PDF-unparseable-cycle8'),
    )


def test_curator_raises_when_get_resume_none_success(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Duplicate success: get_resume None raises, no dangling return.'''
  settings = _settings(tmp_path, monkeypatch)
  curator = _make_curator(settings)
  _configure_curator_ok(curator)
  first = asyncio.run(
    curator.ingest_upload('first.pdf', b'%PDF-same-cycle8'),
  )
  assert first['resume_id'] == 1, first
  assert first['parsed_ok'] is True, first
  from job_hunter.db.repositories.profile import ProfileRepository
  monkeypatch.setattr(
    ProfileRepository, 'get_resume', lambda self, rid: None,
  )
  with pytest.raises(RuntimeError, match='resume row missing'):
    asyncio.run(
      curator.ingest_upload('second.pdf', b'%PDF-same-cycle8'),
    )


# 6. taxonomy non-dict spec guard ----------------------------------------------


def test_taxonomy_load_seed_string_spec_no_raise(
  tmp_path: Path, monkeypatch, caplog,
) -> None:
  '''String spec does not raise AttributeError; dict specs still work.'''
  settings = _settings(tmp_path, monkeypatch)
  from job_hunter.services.skills_taxonomy import SkillsTaxonomy
  taxonomy = SkillsTaxonomy(settings.db_path)
  import logging
  with caplog.at_level(logging.WARNING):
    count = taxonomy.load_seed({
      'skills': {
        'Python': 'string-spec',
        'GoodSkill': {'aliases': ['goodalias']},
      },
    })
  assert count >= 1, count
  assert taxonomy.resolve('goodalias') is not None
  # A warning was logged for the non-dict spec.
  assert any(
    'non-dict' in record.getMessage().lower()
    or 'string' in record.getMessage().lower()
    or 'spec' in record.getMessage().lower()
    for record in caplog.records
    if record.levelno >= logging.WARNING
  ), [r.getMessage() for r in caplog.records]


# 7. score_rank non-int job_id guard --------------------------------------------


def test_score_rank_skips_non_int_job_id(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Phantom job_id='x' is skipped; the real job still scores.'''
  settings = _settings(tmp_path, monkeypatch)
  from job_hunter.core.models import (
    CandidateProfile,
    NormalizedJob,
    RawJobRecord,
  )
  from job_hunter.db.repositories.jobs import JobsRepository
  from job_hunter.db.repositories.runs import RunsRepository
  from job_hunter.graph.nodes import score_rank_persist
  from job_hunter.services.persist import persist_record

  repo = JobsRepository(settings.db_path)
  assert asyncio.run(persist_record(
    jobs_repo=repo,
    record={
      'source_key': 'lever',
      'url': 'https://j.example/cycle8-1',
      'company_name': 'X',
      'title': 'Data Scientist',
      'description_text': 'python sql data science role 55 LPA',
    },
  )) == 'inserted'
  real_id = int(repo.list_jobs()[0]['id'])
  run_id = RunsRepository(settings.db_path).create('discovery', 'test')
  real_job = NormalizedJob(
    job_id=real_id,
    raw=RawJobRecord(
      source_key='lever', url='https://j.example/cycle8-1',
      title='Data Scientist',
    ),
    canonical_url='', content_hash='hash-cycle8-1',
  )
  phantom = NormalizedJob.model_construct(
    job_id='x',
    raw=RawJobRecord(
      source_key='lever', url='https://j.example/phantom',
      title='Phantom',
    ),
    canonical_url='',
    content_hash='phantom-hash',
  )
  candidate = CandidateProfile(
    skills=[], target_roles=['Data Scientist'], salary_floor_lpa=45.0,
    experience_years=9.0, employment_types=['full_time'],
  )
  result = asyncio.run(score_rank_persist(
    {
      'run_id': run_id,
      'candidate': candidate,
      'inserted': [phantom, real_job],
      'enriched': {},
    },
    {'configurable': {'settings': settings}},
  ))
  assert 'errors' not in result, result.get('errors')
  scored_ids = [s.job_id for s in result['scored']]
  assert scored_ids == [real_id], scored_ids


# 8. dashboard send_error -------------------------------------------------------


def test_dashboard_post_501_has_csp() -> None:
  '''POST / -> 501 WITH Content-Security-Policy header.'''
  from llm_gateway.dashboard import _DashboardHandler
  server = ThreadingHTTPServer(('127.0.0.1', 0), _DashboardHandler)
  port = server.server_address[1]
  thread = threading.Thread(
    target=server.serve_forever, daemon=True,
  )
  thread.start()
  try:
    base = f'http://127.0.0.1:{port}'
    req = urllib.request.Request(
      base + '/', data=b'x=1', method='POST',
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
      urllib.request.urlopen(req)
    assert excinfo.value.code == 501, excinfo.value.code
    csp = excinfo.value.headers.get('Content-Security-Policy') or ''
    assert "default-src 'self'" in csp, f'CSP missing: {csp!r}'
    nosniff = excinfo.value.headers.get('X-Content-Type-Options') or ''
    assert nosniff == 'nosniff', nosniff
    excinfo.value.close()
  finally:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
