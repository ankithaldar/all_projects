#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle7 curator/import fixes: duplicate-sha, inf priority, seed clamp.'''

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

fastapi = pytest.importorskip('fastapi')
TestClient = pytest.importorskip('fastapi.testclient').TestClient


def _settings(tmp_path: Path, monkeypatch):
  '''Build migrated settings for direct curator tests.

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


# 1. Duplicate-sha ownership --------------------------------------------------


def test_save_resume_ex_reports_created(tmp_path: Path) -> None:
  '''save_resume_ex returns (id, created) distinguishing new vs duplicate.'''
  from job_hunter.core.db import run_migrations
  from job_hunter.db.repositories.profile import ProfileRepository
  db = tmp_path / 'app.db'
  run_migrations(db)
  repo = ProfileRepository(db)
  assert hasattr(repo, 'save_resume_ex'), (
    'ProfileRepository must expose save_resume_ex'
  )
  first_id, first_created = repo.save_resume_ex(
    '/tmp/a.pdf', 'sha-dup-ex', 'application/pdf', True,
  )
  second_id, second_created = repo.save_resume_ex(
    '/tmp/b.pdf', 'sha-dup-ex', 'application/pdf', True,
  )
  assert first_created is True, (first_id, first_created)
  assert second_created is False, (second_id, second_created)
  assert first_id == second_id
  # Compat wrapper still returns plain id.
  wrapped = repo.save_resume(
    '/tmp/c.pdf', 'sha-dup-ex', 'application/pdf', True,
  )
  assert wrapped == first_id
  # Stored path for the duplicate sha remains the first file.
  assert hasattr(repo, 'get_resume'), 'repo must expose get_resume'
  row = repo.get_resume(first_id)
  assert row is not None
  assert row['file_path'] == '/tmp/a.pdf', row


def test_curator_duplicate_success_returns_stored_path(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Duplicate upload returns stored path and unlinks new file only.'''
  settings = _settings(tmp_path, monkeypatch)
  curator = _make_curator(settings)
  _configure_curator_ok(curator)
  first = asyncio.run(
    curator.ingest_upload('first.pdf', b'%PDF-same-bytes'),
  )
  old_path = first['file_path']
  old_id = first['resume_id']
  assert Path(old_path).exists()
  second = asyncio.run(
    curator.ingest_upload('second.pdf', b'%PDF-same-bytes'),
  )
  assert second['resume_id'] == old_id, second
  assert second['file_path'] == old_path, second
  assert second['path'] == old_path, second
  # New upload file was removed; only the stored file remains.
  resumes_dir = settings.data_dir / 'resumes'
  leftovers = sorted(p.name for p in resumes_dir.iterdir())
  assert leftovers == [Path(old_path).name], leftovers
  assert Path(old_path).exists()
  assert _resume_rows(settings.db_path) == [(old_id, old_path)]


def test_curator_duplicate_failure_preserves_existing(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Duplicate + save_profile boom preserves row and old file.'''
  settings = _settings(tmp_path, monkeypatch)
  curator = _make_curator(settings)
  _configure_curator_ok(curator)
  first = asyncio.run(
    curator.ingest_upload('first.pdf', b'%PDF-same-bytes'),
  )
  old_path = first['file_path']
  old_id = first['resume_id']
  assert Path(old_path).exists()

  from job_hunter.db.repositories.profile import ProfileRepository

  def boom(self, fields, candidate_id=1):
    '''Raise as a failing profile save would.

    Args:
      self: Repository instance.
      fields: Profile fields.
      candidate_id: Candidate id.

    Raises:
      RuntimeError: Simulated save failure.
    '''
    raise RuntimeError('save_profile failed')

  monkeypatch.setattr(ProfileRepository, 'save_profile', boom)
  deleted: list = []
  _orig_delete = ProfileRepository.delete_resume

  def spy_delete(self, resume_id):
    '''Record delete_resume calls while preserving real behavior.

    Args:
      self: Repository instance.
      resume_id: Resume id passed to delete.
    '''
    deleted.append(resume_id)
    return _orig_delete(self, resume_id)

  monkeypatch.setattr(ProfileRepository, 'delete_resume', spy_delete)
  with pytest.raises(RuntimeError, match='save_profile failed'):
    asyncio.run(
      curator.ingest_upload('second.pdf', b'%PDF-same-bytes'),
    )
  # No delete attempt against the pre-existing row; row and file survive.
  assert not deleted, f'delete_resume called for duplicate: {deleted}'
  assert _resume_rows(settings.db_path) == [(old_id, old_path)]
  assert Path(old_path).exists()
  resumes_dir = settings.data_dir / 'resumes'
  leftovers = sorted(p.name for p in resumes_dir.iterdir())
  assert leftovers == [Path(old_path).name], leftovers


def test_curator_duplicate_unparseable_returns_stored(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Duplicate with no text extracted returns stored path, unlinks new.'''
  settings = _settings(tmp_path, monkeypatch)
  curator = _make_curator(settings)
  curator._parser.parse_file = _parsed_fail
  first = asyncio.run(
    curator.ingest_upload('first.pdf', b'%PDF-unparseable'),
  )
  old_path = first['file_path']
  old_id = first['resume_id']
  second = asyncio.run(
    curator.ingest_upload('second.pdf', b'%PDF-unparseable'),
  )
  assert second['resume_id'] == old_id, second
  assert second['file_path'] == old_path, second
  assert second['path'] == old_path, second
  resumes_dir = settings.data_dir / 'resumes'
  leftovers = sorted(p.name for p in resumes_dir.iterdir())
  assert leftovers == [Path(old_path).name], leftovers


# 2. Import inf ---------------------------------------------------------------


def test_clamp_priority_rejects_nonfinite() -> None:
  '''_clamp_priority raises ValueError for inf/nan, still clamps ints.'''
  from job_hunter.api.routes.companies import _clamp_priority
  with pytest.raises(ValueError):
    _clamp_priority(float('inf'))
  with pytest.raises(ValueError):
    _clamp_priority(float('-inf'))
  with pytest.raises(ValueError):
    _clamp_priority(float('nan'))
  assert _clamp_priority(999) == 5
  assert _clamp_priority(-5) == 1
  assert _clamp_priority(3) == 3


def test_company_import_inf_counts_error(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Import with .inf priority counts an error, imports others, no 500.'''
  client = _make_client(tmp_path, monkeypatch)
  yaml_text = (
    'companies:\n'
    '  - name: Good One\n'
    '    domain: goodone.com\n'
    '    priority: 3\n'
    '  - name: Bad Inf\n'
    '    domain: badinf.com\n'
    '    priority: .inf\n'
    '  - name: Good Two\n'
    '    domain: goodtwo.com\n'
    '    priority: 2\n'
  )
  response = client.post('/api/companies/import', json=yaml_text)
  assert response.status_code == 200, response.text
  body = response.json()
  assert body['imported'] == 2, body
  assert body['errors'] == 1, body
  from job_hunter.db.repositories.companies import CompaniesRepository
  stored = {
    row['domain'] for row in CompaniesRepository(
      tmp_path / 'app.db'
    ).list_companies()
  }
  assert stored == {'goodone.com', 'goodtwo.com'}, stored


# 3. Seed clamp ---------------------------------------------------------------


def test_company_discovery_seed_bad_priority_skipped(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Bad seed priority skips entry with warning, good entries ingest.'''
  settings = _settings(tmp_path, monkeypatch)
  seeds = tmp_path / 'seeds'
  seeds.mkdir()
  (seeds / 'companies_test.yaml').write_text(
    'companies:\n'
    '  - name: Seed Good\n'
    '    domain: seedgood.example\n'
    '    priority: 999\n'
    '  - name: Seed Bad X\n'
    '    domain: seedbadx.example\n'
    "    priority: 'x'\n"
    '  - name: Seed Bad Inf\n'
    '    domain: seedbadinf.example\n'
    '    priority: .inf\n'
    '  - name: Seed Good Two\n'
    '    domain: seedgoodtwo.example\n'
    '    priority: 2\n',
    encoding='utf-8',
  )
  from job_hunter.services.company_discovery import ingest_seeds
  count = asyncio.run(ingest_seeds(settings, seeds_dir=seeds))
  assert count == 2, count
  from job_hunter.db.repositories.companies import CompaniesRepository
  rows = CompaniesRepository(settings.db_path).list_companies(limit=10)
  names = {row['name'] for row in rows}
  assert names == {'Seed Good', 'Seed Good Two'}, names
  by_name = {row['name']: row for row in rows}
  assert by_name['Seed Good']['priority'] == 5, by_name


# 4. Dead _valid_domain wrapper ------------------------------------------------


def test_valid_domain_wrapper_removed() -> None:
  '''Dead _valid_domain is gone; _validate_domain remains everywhere.'''
  from job_hunter.api.routes import companies as companies_route
  assert hasattr(companies_route, '_validate_domain')
  assert not hasattr(companies_route, '_valid_domain'), (
    'dead _valid_domain wrapper must be removed'
  )


# 5. Non-dict seed entry + duplicate edge paths -------------------------------


def test_company_discovery_non_dict_seed_skipped(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Non-dict seed entry skipped, good entry counts once.'''
  settings = _settings(tmp_path, monkeypatch)
  seeds = tmp_path / 'seeds'
  seeds.mkdir()
  (seeds / 'companies_test.yaml').write_text(
    'companies:\n'
    '  - not-a-dict\n'
    '  - name: Seed Good One\n'
    '    domain: seedgoodone.example\n'
    '    priority: 3\n',
    encoding='utf-8',
  )
  from job_hunter.services.company_discovery import ingest_seeds
  count = asyncio.run(ingest_seeds(settings, seeds_dir=seeds))
  assert count == 1, count
  from job_hunter.db.repositories.companies import CompaniesRepository
  names = {row['name'] for row in CompaniesRepository(
    settings.db_path,
  ).list_companies(limit=10)}
  assert names == {'Seed Good One'}, names


def test_curator_duplicate_get_resume_none_raises_and_preserves(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Duplicate with get_resume->None raises RuntimeError, keeps original.'''
  settings = _settings(tmp_path, monkeypatch)
  curator = _make_curator(settings)
  _configure_curator_ok(curator)
  first = asyncio.run(
    curator.ingest_upload('first.pdf', b'%PDF-none-row-bytes'),
  )
  old_id = first['resume_id']
  old_path = first['file_path']

  from job_hunter.db.repositories.profile import ProfileRepository

  def none_get(self, resume_id):
    '''Return None as if the row vanished mid-flight.

    Args:
      self: Repository instance.
      resume_id: Resume id.

    Returns:
      None.
    '''
    return

  monkeypatch.setattr(ProfileRepository, 'get_resume', none_get)
  with pytest.raises(RuntimeError, match='resume row missing'):
    asyncio.run(
      curator.ingest_upload('second.pdf', b'%PDF-none-row-bytes'),
    )
  # Original row/file survive; this invocation's file was cleaned up.
  assert _resume_rows(settings.db_path) == [(old_id, old_path)]
  assert Path(old_path).exists()
  resumes_dir = settings.data_dir / 'resumes'
  leftovers = sorted(p.name for p in resumes_dir.iterdir())
  assert leftovers == [Path(old_path).name], leftovers


def test_curator_duplicate_unlink_oserror_no_crash(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Unlink raising OSError on duplicate does not crash ingest.'''
  settings = _settings(tmp_path, monkeypatch)
  curator = _make_curator(settings)
  _configure_curator_ok(curator)
  first = asyncio.run(
    curator.ingest_upload('first.pdf', b'%PDF-unlink-bytes'),
  )
  old_path = first['file_path']
  old_id = first['resume_id']

  _orig_unlink = Path.unlink

  def boom(self, missing_ok=False):
    '''Raise OSError as a locked file would.

    Args:
      self: Path being unlinked.
      missing_ok: Missing-ok flag.

    Raises:
      OSError: Simulated unlink failure.
    '''
    raise OSError('unlink boom')

  monkeypatch.setattr(Path, 'unlink', boom)
  second = asyncio.run(
    curator.ingest_upload('second.pdf', b'%PDF-unlink-bytes'),
  )
  assert second['resume_id'] == old_id, second
  assert second['file_path'] == old_path, second
  assert second['path'] == old_path, second
  monkeypatch.setattr(Path, 'unlink', _orig_unlink)
  assert _resume_rows(settings.db_path) == [(old_id, old_path)]
  assert Path(old_path).exists()
