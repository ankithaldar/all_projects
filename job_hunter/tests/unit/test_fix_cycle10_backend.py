#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle 10 backend fixes: taxonomy guards, overflow/non-finite, curator
unlink retention, inbox rename-on-success, profile ranges, jobs page cap.'''

from __future__ import annotations

import asyncio
import hashlib
import logging
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

import pytest

fastapi = pytest.importorskip('fastapi')
TestClient = pytest.importorskip('fastapi.testclient').TestClient


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


def _resume_rows(db_path: Path) -> list:
  '''Return rows from the resumes table.

  Args:
    db_path: Application database path.

  Returns:
    List of (id, file_path) tuples.
  '''
  with closing(sqlite3.connect(str(db_path))) as conn:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
      'SELECT id, file_path FROM resumes ORDER BY id'
    ).fetchall()
    return [(row['id'], row['file_path']) for row in rows]


# 1. resolve / resolve_many guards ------------------------------------------


def test_resolve_non_str_returns_none(
  tmp_path: Path, monkeypatch,
) -> None:
  '''resolve(None/123/list/bytes) returns None, no AttributeError.'''
  settings = _settings(tmp_path, monkeypatch)
  from job_hunter.services.skills_taxonomy import SkillsTaxonomy
  taxonomy = SkillsTaxonomy(settings.db_path)
  taxonomy.load_seed({'skills': {'Python': {'category': 'lang'}}})
  assert taxonomy.resolve('Python') is not None
  assert taxonomy.resolve(None) is None  # type: ignore[arg-type]
  assert taxonomy.resolve(123) is None  # type: ignore[arg-type]
  assert taxonomy.resolve(['Python']) is None  # type: ignore[arg-type]
  assert taxonomy.resolve(b'Python') is None  # type: ignore[arg-type]
  assert taxonomy.resolve({'x': 'Python'}) is None  # type: ignore[arg-type]


def test_resolve_many_bare_str_and_non_list_raises_typeerror(
  tmp_path: Path, monkeypatch,
) -> None:
  '''resolve_many('Python'/b'...'/123/None) raises TypeError.'''
  settings = _settings(tmp_path, monkeypatch)
  from job_hunter.services.skills_taxonomy import SkillsTaxonomy
  taxonomy = SkillsTaxonomy(settings.db_path)
  taxonomy.load_seed({'skills': {'Python': {'category': 'lang'}}})
  with pytest.raises(TypeError, match='names must be a list of str'):
    taxonomy.resolve_many('Python')  # type: ignore[arg-type]
  with pytest.raises(TypeError, match='names must be a list of str'):
    taxonomy.resolve_many(b'Python')  # type: ignore[arg-type]
  with pytest.raises(TypeError, match='names must be a list of str'):
    taxonomy.resolve_many(123)  # type: ignore[arg-type]
  with pytest.raises(TypeError, match='names must be a list of str'):
    taxonomy.resolve_many(None)  # type: ignore[arg-type]
  with pytest.raises(TypeError, match='names must be a list of str'):
    taxonomy.resolve_many({'Python': 1})  # type: ignore[arg-type]
  # Tuple is allowed.
  assert taxonomy.resolve_many(('Python',)) == [
    taxonomy.resolve('Python')
  ]


def test_resolve_many_mixed_skips_non_str_with_warning(
  tmp_path: Path, monkeypatch, caplog,
) -> None:
  '''resolve_many(['Python',123]) resolves Python only, warns on 123.'''
  settings = _settings(tmp_path, monkeypatch)
  from job_hunter.services.skills_taxonomy import SkillsTaxonomy
  taxonomy = SkillsTaxonomy(settings.db_path)
  taxonomy.load_seed({'skills': {'Python': {'category': 'lang'}}})
  with caplog.at_level(logging.WARNING):
    ids = taxonomy.resolve_many(['Python', 123])  # type: ignore[list-item]
  assert ids == [taxonomy.resolve('Python')]
  assert any(
    record.levelno >= logging.WARNING for record in caplog.records
  ), [r.getMessage() for r in caplog.records]


def test_put_profile_skills_non_str_422(
  tmp_path: Path, monkeypatch,
) -> None:
  '''PUT /profile/skills with [123] -> 422; bare string -> 422.'''
  settings = _settings(tmp_path, monkeypatch)
  from job_hunter.services.skills_taxonomy import SkillsTaxonomy
  SkillsTaxonomy(settings.db_path).load_seed(
    {'skills': {'Python': {'category': 'lang'}}}
  )
  client = _make_client(tmp_path, monkeypatch)
  bad = client.put('/api/profile/skills', json=[123])
  assert bad.status_code == 422, bad.text
  bad2 = client.put('/api/profile/skills', json=['Python', 123])
  assert bad2.status_code == 422, bad2.text
  bare = client.put('/api/profile/skills', json='Python')
  assert bare.status_code == 422, bare.text
  good = client.put('/api/profile/skills', json=['Python'])
  assert good.status_code == 200, good.text
  assert good.json()['resolved'] == 1, good.text


# 2. Non-dict spec skip ------------------------------------------------------


def test_load_seed_non_dict_preserves_category(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Non-dict/non-None spec returns 0 without nulling category.'''
  settings = _settings(tmp_path, monkeypatch)
  from job_hunter.services.skills_taxonomy import SkillsTaxonomy
  taxonomy = SkillsTaxonomy(settings.db_path)
  count = taxonomy.load_seed(
    {'skills': {'Go': {'category': 'lang', 'aliases': []}}}
  )
  assert count == 1, count
  before = {
    row['canonical_name']: row for row in taxonomy.all_skills()
  }
  assert before['Go']['category'] == 'lang', before
  count2 = taxonomy.load_seed({'skills': {'Go': 'not-a-dict-spec'}})
  assert count2 == 0, count2
  after = {
    row['canonical_name']: row for row in taxonomy.all_skills()
  }
  assert after['Go']['category'] == 'lang', after


def test_load_seed_non_dict_skipped_mixed(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Mixed payload: good dict loads, bad spec skipped, count only good.'''
  settings = _settings(tmp_path, monkeypatch)
  from job_hunter.services.skills_taxonomy import SkillsTaxonomy
  taxonomy = SkillsTaxonomy(settings.db_path)
  taxonomy.load_seed(
    {'skills': {'Rust': {'category': 'lang', 'aliases': []}}}
  )
  count = taxonomy.load_seed({
    'skills': {
      'Rust': 12345,
      'NewSkill': {'category': 'new', 'aliases': []},
    },
  })
  assert count == 1, count
  rows = {row['canonical_name']: row for row in taxonomy.all_skills()}
  assert rows['Rust']['category'] == 'lang', rows
  assert rows['NewSkill']['category'] == 'new', rows


# 3. Settings OverflowError --------------------------------------------------


def test_settings_weights_huge_int_422(
  tmp_path: Path, monkeypatch,
) -> None:
  '''PUT scoring_weights with 10**400 -> 422 not 500.'''
  client = _make_client(tmp_path, monkeypatch)
  response = client.put(
    '/api/settings?key=scoring_weights', json={'score': 10**400},
  )
  assert response.status_code == 422, response.text


def test_settings_floor_huge_int_422(
  tmp_path: Path, monkeypatch,
) -> None:
  '''PUT salary_hard_floor_lpa with 10**400 -> 422 not 500.'''
  client = _make_client(tmp_path, monkeypatch)
  response = client.put(
    '/api/settings?key=salary_hard_floor_lpa', json=10**400,
  )
  assert response.status_code == 422, response.text


# 4. Raw non-finite 500 ------------------------------------------------------


def test_companies_raw_infinity_422(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Raw Infinity priority -> 422 not 500.'''
  client = _make_client(tmp_path, monkeypatch)
  response = client.post(
    '/api/companies',
    content='{"name":"x","domain":"x.example.com","priority":Infinity}',
    headers={'content-type': 'application/json'},
  )
  assert response.status_code == 422, response.text
  # Response body must be valid JSON.
  assert response.json()['detail'], response.text


def test_companies_raw_nan_and_1e400_422(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Raw NaN / 1e400 priority -> 422 not 500.'''
  client = _make_client(tmp_path, monkeypatch)
  nan_resp = client.post(
    '/api/companies',
    content='{"name":"x","domain":"x.example.com","priority":NaN}',
    headers={'content-type': 'application/json'},
  )
  assert nan_resp.status_code == 422, nan_resp.text
  big_resp = client.post(
    '/api/companies',
    content='{"name":"x","domain":"x.example.com","priority":1e400}',
    headers={'content-type': 'application/json'},
  )
  assert big_resp.status_code == 422, big_resp.text


# 5. Route unlink retention --------------------------------------------------


def test_curator_creator_fail_keeps_file_when_duplicate_adopted(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Creator fails at save_profile after duplicate adopted the row.

  Stored file and row must survive; exc.file_path must be falsy so the
  route does not unlink the retained stored file.
  '''
  settings = _settings(tmp_path, monkeypatch)
  curator = _make_curator(settings)
  _configure_curator_ok(curator)

  from job_hunter.db.repositories.profile import ProfileRepository
  orig_save = ProfileRepository.save_profile
  creator_entered = threading.Event()
  duplicate_done = threading.Event()
  calls = {'n': 0}

  def flaky_save(self, fields, candidate_id=1):
    '''First (creator) call blocks, then fails; others use real save.

    Args:
      self: Repository instance.
      fields: Profile fields.
      candidate_id: Candidate id.

    Returns:
      Real save_profile result for non-creator calls.

    Raises:
      RuntimeError: For the creator, after the duplicate completes.
    '''
    calls['n'] += 1
    if calls['n'] == 1:
      creator_entered.set()
      assert duplicate_done.wait(timeout=10)
      raise RuntimeError('creator save_profile failed')
    return orig_save(self, fields, candidate_id)

  monkeypatch.setattr(ProfileRepository, 'save_profile', flaky_save)

  errors: list = []

  def run_creator() -> None:
    '''Run the creator ingest in its own thread / event loop.'''
    try:
      asyncio.run(
        curator.ingest_upload('first.pdf', b'%PDF-concurrent-cycle10'),
      )
    except RuntimeError as exc:
      errors.append(exc)

  thread = threading.Thread(target=run_creator, daemon=True)
  thread.start()
  nested = None
  try:
    assert creator_entered.wait(timeout=10), (
      'creator never reached save_profile'
    )
    nested = asyncio.run(
      curator.ingest_upload('second.pdf', b'%PDF-concurrent-cycle10'),
    )
  finally:
    duplicate_done.set()
  thread.join(timeout=10)
  assert not thread.is_alive(), 'creator thread hung'
  assert errors, 'creator should have raised RuntimeError'
  assert 'creator save_profile failed' in str(errors[0])
  assert nested is not None
  stored_path = nested['file_path']
  assert Path(stored_path).exists(), stored_path
  assert _resume_rows(settings.db_path) == [
    (nested['resume_id'], stored_path),
  ]
  profile = ProfileRepository(settings.db_path).get_profile(1)
  assert profile is not None, profile
  assert profile['resume_id'] == nested['resume_id'], profile
  # Curator must not hand the route a retained stored path to unlink.
  assert not getattr(errors[0], 'file_path', None), getattr(
    errors[0], 'file_path', None,
  )


def test_route_concurrent_duplicate_creator_fail_keeps_file(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Same scenario via API: creator 422, duplicate 200, file kept.'''
  client = _make_client(tmp_path, monkeypatch)
  # Stub curator internals for route-created instances.
  import job_hunter.llm.client as llm_client_mod
  from job_hunter.services.profile_curator import ProfileCurator
  from job_hunter.services.resume_parser import ResumeParser

  monkeypatch.setattr(llm_client_mod, 'get_client', lambda _s: object())

  def fake_parse(self, path):
    '''Stub parse for route-created curator instances.

    Args:
      self: Parser instance.
      path: Written resume file path.

    Returns:
      Parse mapping with ok True.
    '''
    return _parsed_ok(path)

  monkeypatch.setattr(ResumeParser, 'parse_file', fake_parse)
  monkeypatch.setattr(
    ProfileCurator, '_persist_skills', lambda self, raw, cid: [],
  )

  async def fake_extract(self, _text):
    '''Return a fixed extraction for route curator instances.

    Args:
      self: Curator instance.
      _text: Resume text.

    Returns:
      Fake extraction.
    '''
    return _extraction()

  monkeypatch.setattr(ProfileCurator, 'extract', fake_extract)

  from job_hunter.db.repositories.profile import ProfileRepository
  orig_save = ProfileRepository.save_profile
  creator_entered = threading.Event()
  duplicate_done = threading.Event()
  calls = {'n': 0}

  def flaky_save(self, fields, candidate_id=1):
    '''First (creator) call blocks, then fails; others use real save.

    Args:
      self: Repository instance.
      fields: Profile fields.
      candidate_id: Candidate id.

    Returns:
      Real save_profile result for non-creator calls.

    Raises:
      RuntimeError: For the creator, after the duplicate completes.
    '''
    calls['n'] += 1
    if calls['n'] == 1:
      creator_entered.set()
      assert duplicate_done.wait(timeout=10)
      raise RuntimeError('creator save_profile failed')
    return orig_save(self, fields, candidate_id)

  monkeypatch.setattr(ProfileRepository, 'save_profile', flaky_save)

  payload = b'%PDF-concurrent-route-cycle10'
  creator_results: list = []

  def run_creator() -> None:
    '''POST the creator resume upload via the API.'''
    creator_results.append(
      client.post(
        '/api/profile/resume',
        files={'file': ('first.pdf', payload, 'application/pdf')},
      )
    )

  thread = threading.Thread(target=run_creator, daemon=True)
  thread.start()
  duplicate_resp = None
  try:
    assert creator_entered.wait(timeout=10), (
      'creator never reached save_profile'
    )
    duplicate_resp = client.post(
      '/api/profile/resume',
      files={'file': ('second.pdf', payload, 'application/pdf')},
    )
  finally:
    duplicate_done.set()
  thread.join(timeout=15)
  assert not thread.is_alive(), 'creator thread hung'
  assert creator_results, 'creator request never completed'
  assert creator_results[0].status_code == 422, creator_results[0].text
  assert duplicate_resp is not None
  assert duplicate_resp.status_code == 200, duplicate_resp.text
  stored_path = duplicate_resp.json()['file_path']
  resume_id = duplicate_resp.json()['resume_id']
  assert Path(stored_path).exists(), stored_path
  assert _resume_rows(tmp_path / 'app.db') == [(resume_id, stored_path)]
  profile = ProfileRepository(tmp_path / 'app.db').get_profile(1)
  assert profile is not None, profile
  assert profile['resume_id'] == resume_id, profile


# 6. Inbox rename-on-success -------------------------------------------------


def test_scan_inbox_insert_raising_keeps_file(
  tmp_path: Path, monkeypatch,
) -> None:
  '''_insert_records raising -> file NOT moved.'''
  from job_hunter.adapters.manual_inbox import scan_inbox
  from job_hunter.core.config import AppSettings
  from job_hunter.core.db import run_migrations
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  run_migrations(tmp_path / 'app.db')
  inbox = tmp_path / 'inbox'
  inbox.mkdir(parents=True)
  target = inbox / 'jobs.csv'
  target.write_text(
    'Job Title,Company,Location\nMy Job,Acme,Delhi\n', encoding='utf-8',
  )
  config_path = tmp_path / 'app.yaml'
  config_path.write_text('salary_hard_floor_lpa: 45\n', encoding='utf-8')
  settings = AppSettings(config_path)

  import job_hunter.adapters.manual_inbox as manual_inbox_mod

  async def boom(settings, records):
    '''Simulate a DB failure during inserts.

    Args:
      settings: App settings.
      records: Parsed records.

    Raises:
      RuntimeError: Simulated DB failure.
    '''
    raise RuntimeError('db boom')

  monkeypatch.setattr(manual_inbox_mod, '_insert_records', boom)
  count = asyncio.run(scan_inbox(settings))
  assert count == 0, count
  assert target.exists(), 'file must stay when inserts fail'
  assert not list((inbox / '_processed').iterdir()) if (
    inbox / '_processed'
  ).exists() else True


def test_scan_inbox_db_exception_per_record_keeps_file(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Per-record DB exception (persist_record) -> file NOT moved.'''
  from job_hunter.adapters.manual_inbox import scan_inbox
  from job_hunter.core.config import AppSettings
  from job_hunter.core.db import run_migrations
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  run_migrations(tmp_path / 'app.db')
  inbox = tmp_path / 'inbox'
  inbox.mkdir(parents=True)
  target = inbox / 'jobs.csv'
  target.write_text(
    'Job Title,Company,Location\nMy Job,Acme,Delhi\n', encoding='utf-8',
  )
  config_path = tmp_path / 'app.yaml'
  config_path.write_text('salary_hard_floor_lpa: 45\n', encoding='utf-8')
  settings = AppSettings(config_path)

  import job_hunter.adapters.manual_inbox as manual_inbox_mod

  async def db_boom(*args, **kwargs):
    '''Simulate a SQLite failure for one record.

    Args:
      *args: Positional args.
      **kwargs: Keyword args.

    Raises:
      sqlite3.OperationalError: Simulated DB failure.
    '''
    raise sqlite3.OperationalError('db boom')

  monkeypatch.setattr(manual_inbox_mod, 'persist_record', db_boom)
  count = asyncio.run(scan_inbox(settings))
  assert count == 0, count
  assert target.exists(), 'file must stay on DB exception'
  processed = inbox / '_processed'
  names = sorted(p.name for p in processed.iterdir()) if (
    processed.exists()
  ) else []
  assert not any(name.endswith('_jobs.csv') for name in names), names


# 7. Profiles bool/range -----------------------------------------------------


def test_profile_salary_floor_bool_422(
  tmp_path: Path, monkeypatch,
) -> None:
  '''PUT salary_floor True -> 422 (bool is not a number).'''
  client = _make_client(tmp_path, monkeypatch)
  response = client.put('/api/profile', json={'salary_floor_lpa': True})
  assert response.status_code == 422, response.text


def test_profile_experience_nan_inf_422(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Raw NaN/Infinity experience_years -> 422 not 200/500.'''
  client = _make_client(tmp_path, monkeypatch)
  nan_resp = client.put(
    '/api/profile',
    content='{"experience_years": NaN}',
    headers={'content-type': 'application/json'},
  )
  assert nan_resp.status_code == 422, nan_resp.text
  inf_resp = client.put(
    '/api/profile',
    content='{"experience_years": Infinity}',
    headers={'content-type': 'application/json'},
  )
  assert inf_resp.status_code == 422, inf_resp.text
  neg_inf = client.put(
    '/api/profile',
    content='{"experience_years": -Infinity}',
    headers={'content-type': 'application/json'},
  )
  assert neg_inf.status_code == 422, neg_inf.text


def test_profile_experience_range_422(
  tmp_path: Path, monkeypatch,
) -> None:
  '''PUT experience_years -5 / 61 -> 422; 30 / 0 / 60 -> 200.'''
  client = _make_client(tmp_path, monkeypatch)
  low = client.put('/api/profile', json={'experience_years': -5})
  assert low.status_code == 422, low.text
  high = client.put('/api/profile', json={'experience_years': 61})
  assert high.status_code == 422, high.text
  for ok_val in (0, 30, 60):
    ok = client.put('/api/profile', json={'experience_years': ok_val})
    assert ok.status_code == 200, (ok_val, ok.text)


# 8. Jobs page cap -----------------------------------------------------------


def test_jobs_page_huge_422(
  tmp_path: Path, monkeypatch,
) -> None:
  '''GET /api/jobs?page=10**400 -> 422 not 500.'''
  client = _make_client(tmp_path, monkeypatch)
  huge = '1' + '0' * 400
  response = client.get(f'/api/jobs?page={huge}')
  assert response.status_code == 422, response.text
  over = client.get('/api/jobs?page=10001')
  assert over.status_code == 422, over.text
  edge = client.get('/api/jobs?page=10000')
  assert edge.status_code == 200, edge.text
