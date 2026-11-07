#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle 9 backend fixes: priority overflow/bool, curator unlink order,
taxonomy aliases guard, manual_inbox _processed skip.'''

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


# 1. _clamp_priority overflow / bool ------------------------------------------


def test_clamp_priority_huge_int_and_bool() -> None:
  '''Huge int clamps to 5/-5 to 1 without OverflowError; bool rejected.'''
  from job_hunter.api.routes.companies import _clamp_priority
  assert _clamp_priority(10**400) == 5
  assert _clamp_priority(-(10**400)) == 1
  assert _clamp_priority(999) == 5
  assert _clamp_priority(3) == 3
  with pytest.raises(ValueError):
    _clamp_priority(True)
  with pytest.raises(ValueError):
    _clamp_priority(False)


def test_clamp_seed_priority_huge_int_and_bool() -> None:
  '''Seed clamp mirrors companies: huge int -> 5, bool rejected.'''
  from job_hunter.services.company_discovery import _clamp_seed_priority
  assert _clamp_seed_priority(10**400) == 5
  assert _clamp_seed_priority(-(10**400)) == 1
  with pytest.raises(ValueError):
    _clamp_seed_priority(True)


def test_company_create_huge_priority_stored_five(
  tmp_path: Path, monkeypatch,
) -> None:
  '''POST create priority=10**400 -> 200 stored 5 (not 500).'''
  client = _make_client(tmp_path, monkeypatch)
  response = client.post('/api/companies', json={
    'name': 'Huge Co',
    'domain': 'huge.example',
    'priority': 10**400,
  })
  assert response.status_code == 200, response.text
  from job_hunter.db.repositories.companies import CompaniesRepository
  rows = CompaniesRepository(tmp_path / 'app.db').list_companies(limit=10)
  assert len(rows) == 1, rows
  assert rows[0]['priority'] == 5, rows


def test_company_create_bool_priority_422(
  tmp_path: Path, monkeypatch,
) -> None:
  '''POST create priority=True -> 422 (bool is not a valid priority).'''
  client = _make_client(tmp_path, monkeypatch)
  response = client.post('/api/companies', json={
    'name': 'Bool Co',
    'domain': 'boolco.example',
    'priority': True,
  })
  assert response.status_code == 422, response.text


def test_company_patch_bool_priority_422_and_huge_clamps(
  tmp_path: Path, monkeypatch,
) -> None:
  '''PATCH priority=True -> 422; PATCH huge int -> 200 stored 5.'''
  client = _make_client(tmp_path, monkeypatch)
  created = client.post('/api/companies', json={
    'name': 'Patch Co',
    'domain': 'patchco.example',
    'priority': 3,
  })
  assert created.status_code == 200, created.text
  company_id = created.json()['company_id']
  bad = client.patch(f'/api/companies/{company_id}', json={
    'priority': True,
  })
  assert bad.status_code == 422, bad.text
  huge = client.patch(f'/api/companies/{company_id}', json={
    'priority': 10**400,
  })
  assert huge.status_code == 200, huge.text
  from job_hunter.db.repositories.companies import CompaniesRepository
  row = CompaniesRepository(tmp_path / 'app.db').get(company_id)
  assert row is not None and row['priority'] == 5, row


def test_company_import_bool_priority_counts_error(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Import with priority: true counts an error, no 500.'''
  client = _make_client(tmp_path, monkeypatch)
  yaml_text = (
    'companies:\n'
    '  - name: Good One\n'
    '    domain: goodone.example\n'
    '    priority: 3\n'
    '  - name: Bool Bad\n'
    '    domain: boolbad.example\n'
    '    priority: true\n'
  )
  response = client.post('/api/companies/import', json=yaml_text)
  assert response.status_code == 200, response.text
  body = response.json()
  assert body['imported'] == 1, body
  assert body['errors'] == 1, body
  from job_hunter.db.repositories.companies import CompaniesRepository
  stored = {
    row['domain'] for row in CompaniesRepository(
      tmp_path / 'app.db'
    ).list_companies()
  }
  assert stored == {'goodone.example'}, stored


def test_company_import_huge_priority_clamps_to_five(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Import with a huge int priority clamps to 5 instead of erroring.'''
  client = _make_client(tmp_path, monkeypatch)
  huge = '1' + '0' * 400
  yaml_text = (
    'companies:\n'
    '  - name: Huge Import\n'
    '    domain: hugeimport.example\n'
    f'    priority: {huge}\n'
  )
  response = client.post('/api/companies/import', json=yaml_text)
  assert response.status_code == 200, response.text
  body = response.json()
  assert body['imported'] == 1, body
  assert body['errors'] == 0, body
  from job_hunter.db.repositories.companies import CompaniesRepository
  rows = CompaniesRepository(tmp_path / 'app.db').list_companies(limit=10)
  assert rows[0]['priority'] == 5, rows


# 2. Curator concurrent-duplicate unlink ---------------------------------------


def test_curator_creator_fail_keeps_file_when_duplicate_adopted(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Creator fails at save_profile after duplicate adopted the row.

  The stored file and the duplicate's saved profile must survive:
  cleanup must delete the row first and keep the file when that
  delete is blocked by the duplicate's profile reference.
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
      # Wait for the duplicate upload to adopt + save its profile.
      assert duplicate_done.wait(timeout=10)
      raise RuntimeError('creator save_profile failed')
    return orig_save(self, fields, candidate_id)

  monkeypatch.setattr(ProfileRepository, 'save_profile', flaky_save)

  errors: list = []

  def run_creator() -> None:
    '''Run the creator ingest in its own thread / event loop.'''
    try:
      asyncio.run(
        curator.ingest_upload('first.pdf', b'%PDF-concurrent-cycle9'),
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
    # Second upload (duplicate sha) adopts the creator's resume row
    # and saves a profile referencing it.
    nested = asyncio.run(
      curator.ingest_upload('second.pdf', b'%PDF-concurrent-cycle9'),
    )
  finally:
    duplicate_done.set()
  thread.join(timeout=10)
  assert not thread.is_alive(), 'creator thread hung'
  assert errors, 'creator should have raised RuntimeError'
  assert 'creator save_profile failed' in str(errors[0])
  assert nested is not None
  stored_path = nested['file_path']
  # Stored file retained despite creator cleanup.
  assert Path(stored_path).exists(), stored_path
  # Resume row retained (FK-blocked delete) and profile intact.
  assert _resume_rows(settings.db_path) == [
    (nested['resume_id'], stored_path),
  ]
  profile = ProfileRepository(settings.db_path).get_profile(1)
  assert profile is not None, profile
  assert profile['resume_id'] == nested['resume_id'], profile


# 3. Taxonomy aliases validation ------------------------------------------------


def test_taxonomy_aliases_non_list_skipped_no_raise(
  tmp_path: Path, monkeypatch, caplog,
) -> None:
  '''aliases=123 does not crash load_seed; valid skills still load.'''
  settings = _settings(tmp_path, monkeypatch)
  from job_hunter.services.skills_taxonomy import SkillsTaxonomy
  taxonomy = SkillsTaxonomy(settings.db_path)
  with caplog.at_level(logging.WARNING):
    count = taxonomy.load_seed({
      'skills': {
        'BadAlias': {'aliases': 123},
        'GoodSkill': {'aliases': ['goodalias']},
      },
    })
  assert count >= 1, count
  assert taxonomy.resolve('goodalias') is not None
  assert any(
    record.levelno >= logging.WARNING for record in caplog.records
  ), [r.getMessage() for r in caplog.records]


def test_taxonomy_aliases_non_str_elements_skipped(
  tmp_path: Path, monkeypatch,
) -> None:
  '''aliases=[1, None, 'realalias'] loads the skill, skips bad elements.'''
  settings = _settings(tmp_path, monkeypatch)
  from job_hunter.services.skills_taxonomy import SkillsTaxonomy
  taxonomy = SkillsTaxonomy(settings.db_path)
  count = taxonomy.load_seed({
    'skills': {
      'MixedSkill': {'aliases': [1, None, 'realalias']},
      'TupleSkill': {'aliases': ('tuplealias',)},
    },
  })
  assert count == 2, count
  assert taxonomy.resolve('realalias') is not None
  assert taxonomy.resolve('tuplealias') is not None
  assert taxonomy.resolve('mixedskill') is not None


def test_taxonomy_non_str_name_key_skipped(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Non-str skill name key is skipped; valid entries still load.'''
  settings = _settings(tmp_path, monkeypatch)
  from job_hunter.services.skills_taxonomy import SkillsTaxonomy
  taxonomy = SkillsTaxonomy(settings.db_path)
  count = taxonomy.load_seed({
    'skills': {
      123: {'aliases': ['shouldnotload']},
      'GoodSkill': {'aliases': ['goodalias']},
    },
  })
  assert count == 1, count
  assert taxonomy.resolve('goodalias') is not None
  assert taxonomy.resolve('shouldnotload') is None


# 4. manual_inbox _processed skip ----------------------------------------------


def test_scan_inbox_skips_processed_files(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Pre-seeded _processed file is not re-ingested or renamed.'''
  from job_hunter.core.config import AppSettings
  from job_hunter.core.db import run_migrations
  from job_hunter.db.repositories.jobs import JobsRepository
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  inbox = tmp_path / 'inbox'
  processed = inbox / '_processed'
  processed.mkdir(parents=True)
  run_migrations(tmp_path / 'app.db')
  processed_csv = (
    'Job Title,Company,Location\n'
    'Processed Only Job,ProcessedCorp,Delhi\n'
  )
  (processed / 'already.csv').write_text(processed_csv, encoding='utf-8')
  (inbox / 'fresh.csv').write_text(
    'Job Title,Company,Location\n'
    'Fresh Job,FreshCorp,Mumbai\n',
    encoding='utf-8',
  )
  config_path = tmp_path / 'app.yaml'
  config_path.write_text('salary_hard_floor_lpa: 45\n', encoding='utf-8')
  settings = AppSettings(config_path)
  from job_hunter.adapters.manual_inbox import scan_inbox
  count = asyncio.run(scan_inbox(settings))
  rows = JobsRepository(settings.db_path).list_jobs()
  titles = [row['title'] for row in rows]
  # Only the fresh inbox file was ingested.
  assert count == 1, (count, titles)
  assert titles == ['Fresh Job'], titles
  # _processed file untouched: same path, same content, no rename.
  assert (processed / 'already.csv').exists()
  assert (processed / 'already.csv').read_text(encoding='utf-8') == (
    processed_csv
  )
  processed_names = sorted(p.name for p in processed.iterdir())
  # The pre-seeded file must not have been renamed to a timestamped copy.
  assert not any(
    name.endswith('_already.csv') for name in processed_names
  ), processed_names
  # Second scan re-ingests nothing from _processed.
  count2 = asyncio.run(scan_inbox(settings))
  assert count2 == 0, count2
  assert len(JobsRepository(settings.db_path).list_jobs()) == 1
