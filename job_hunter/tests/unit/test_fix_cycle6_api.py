#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle6 API/web fixes: curator atomicity, priority clamp, stream, a11y.'''

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

fastapi = pytest.importorskip('fastapi')
TestClient = pytest.importorskip('fastapi.testclient').TestClient

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / 'web'
SRC = ROOT / 'src' / 'job_hunter'


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


def _resume_rows(db_path: Path) -> list:
  '''Return rows from the resumes table.

  Args:
    db_path: Application database path.

  Returns:
    List of file_path values.
  '''
  with closing(sqlite3.connect(str(db_path))) as conn:
    conn.row_factory = sqlite3.Row
    rows = conn.execute('SELECT file_path FROM resumes').fetchall()
    return [row['file_path'] for row in rows]


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


# 1. Curator atomicity -----------------------------------------------------


def test_curator_save_profile_failure_no_file_no_resume_row(
  tmp_path: Path, monkeypatch,
) -> None:
  '''save_profile raising leaves no file and no resume DB row.'''
  settings = _settings(tmp_path, monkeypatch)
  curator = _make_curator(settings)
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

  curator.extract = fake_extract

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
  with pytest.raises(RuntimeError):
    asyncio.run(curator.ingest_upload('resume.pdf', b'%PDF-good'))
  resumes_dir = settings.data_dir / 'resumes'
  leftover = list(resumes_dir.iterdir()) if resumes_dir.exists() else []
  assert leftover == [], f'leftover files={leftover}'
  assert _resume_rows(settings.db_path) == []


def test_curator_cancelled_error_unlinks_file_and_reraises(
  tmp_path: Path, monkeypatch,
) -> None:
  '''CancelledError during extract unlinks the file and re-raises.'''
  settings = _settings(tmp_path, monkeypatch)
  curator = _make_curator(settings)
  curator._parser.parse_file = _parsed_ok

  async def cancelled(_text):
    '''Raise CancelledError as cancellation would.

    Args:
      _text: Resume text.

    Raises:
      asyncio.CancelledError: Simulated cancellation.
    '''
    raise asyncio.CancelledError()

  curator.extract = cancelled
  with pytest.raises(asyncio.CancelledError):
    asyncio.run(curator.ingest_upload('resume.pdf', b'%PDF-good'))
  resumes_dir = settings.data_dir / 'resumes'
  leftover = list(resumes_dir.iterdir()) if resumes_dir.exists() else []
  assert leftover == [], f'leftover files={leftover}'


def test_curator_base_exception_unlinks_file_and_reraises(
  tmp_path: Path, monkeypatch,
) -> None:
  '''A non-Exception BaseException during extract unlinks and re-raises.'''

  class _Abort(BaseException):
    '''Test-only BaseException subclass.'''

  settings = _settings(tmp_path, monkeypatch)
  curator = _make_curator(settings)
  curator._parser.parse_file = _parsed_ok

  async def abort(_text):
    '''Raise a BaseException.

    Args:
      _text: Resume text.

    Raises:
      _Abort: Simulated abort.
    '''
    raise _Abort()

  curator.extract = abort
  with pytest.raises(_Abort):
    asyncio.run(curator.ingest_upload('resume.pdf', b'%PDF-good'))
  resumes_dir = settings.data_dir / 'resumes'
  leftover = list(resumes_dir.iterdir()) if resumes_dir.exists() else []
  assert leftover == [], f'leftover files={leftover}'


def test_curator_full_uuid_in_filename_and_session_id(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Filename uuid segment and session_id use the full 32-char hex.'''
  settings = _settings(tmp_path, monkeypatch)
  curator = _make_curator(settings)
  curator._parser.parse_file = _parsed_fail
  result = asyncio.run(curator.ingest_upload('resume.pdf', b'%PDF-one'))
  name = Path(result['file_path']).name
  parts = name.split('_')
  assert len(parts) >= 3, name
  assert len(parts[1]) == 32, f'uuid segment not full hex: {name}'
  int(parts[1], 16)

  captured: dict = {}

  async def fake_structured(_client, _model, _instr, _text, session_id=None):
    '''Capture session_id and return a fixed extraction.

    Args:
      _client: Gateway client.
      _model: Extraction model.
      _instr: Instruction text.
      _text: Resume text.
      session_id: Session correlation id.

    Returns:
      Fake extraction.
    '''
    captured['session_id'] = session_id
    return _extraction()

  monkeypatch.setattr(
    'job_hunter.services.profile_curator.complete_structured',
    fake_structured,
  )
  asyncio.run(curator.extract('some resume text'))
  session_id = captured.get('session_id') or ''
  assert session_id.startswith('profile:'), session_id
  tail = session_id.split(':', 1)[1]
  assert len(tail) == 32, f'session_id not full uuid hex: {session_id}'
  int(tail, 16)


def test_profile_repository_delete_resume(tmp_path: Path, monkeypatch) -> None:
  '''delete_resume removes exactly the row with the given id.'''
  settings = _settings(tmp_path, monkeypatch)
  from job_hunter.db.repositories.profile import ProfileRepository
  repo = ProfileRepository(settings.db_path)
  resume_id = repo.save_resume(
    file_path='x.pdf', sha256='a' * 64, mime='application/pdf',
    parsed_ok=True, candidate_id=1,
  )
  assert _resume_rows(settings.db_path) == ['x.pdf']
  repo.delete_resume(resume_id)
  assert _resume_rows(settings.db_path) == []


# 2. Priority clamp --------------------------------------------------------


def test_company_create_priority_clamped(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Create clamps priority 999 -> 5 and -5 -> 1.'''
  client = _make_client(tmp_path, monkeypatch)
  from job_hunter.db.repositories.companies import CompaniesRepository
  repo = CompaniesRepository(tmp_path / 'app.db')
  high = client.post(
    '/api/companies',
    json={'name': 'High Co', 'domain': 'highco.com', 'priority': 999},
  )
  assert high.status_code == 200, high.text
  low = client.post(
    '/api/companies',
    json={'name': 'Low Co', 'domain': 'lowco.com', 'priority': -5},
  )
  assert low.status_code == 200, low.text
  stored = {row['domain']: row['priority'] for row in repo.list_companies()}
  assert stored['highco.com'] == 5, stored
  assert stored['lowco.com'] == 1, stored


def test_company_patch_priority_clamped(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Patch clamps priority 999 -> 5 and -5 -> 1.'''
  client = _make_client(tmp_path, monkeypatch)
  created = client.post(
    '/api/companies',
    json={'name': 'Patch Co', 'domain': 'patchco.com', 'priority': 3},
  )
  assert created.status_code == 200, created.text
  company_id = created.json()['company_id']
  high = client.patch(
    f'/api/companies/{company_id}', json={'priority': 999},
  )
  assert high.status_code == 200, high.text
  assert high.json()['priority'] == 5, high.text
  low = client.patch(
    f'/api/companies/{company_id}', json={'priority': -5},
  )
  assert low.status_code == 200, low.text
  assert low.json()['priority'] == 1, low.text


def test_company_import_priority_clamped_bad_type_errors(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Import clamps priority; a bad-typed priority still counts an error.'''
  client = _make_client(tmp_path, monkeypatch)
  yaml_text = (
    'companies:\n'
    '  - name: Import High\n'
    '    domain: importhigh.com\n'
    '    priority: 999\n'
    '  - name: Import Low\n'
    '    domain: importlow.com\n'
    '    priority: -5\n'
    '  - name: Import Bad\n'
    '    domain: importbad.com\n'
    "    priority: 'x'\n"
  )
  response = client.post('/api/companies/import', json=yaml_text)
  assert response.status_code == 200, response.text
  body = response.json()
  assert body['imported'] == 2, body
  assert body['errors'] == 1, body
  from job_hunter.db.repositories.companies import CompaniesRepository
  stored = {
    row['domain']: row['priority']
    for row in CompaniesRepository(tmp_path / 'app.db').list_companies()
  }
  assert stored['importhigh.com'] == 5, stored
  assert stored['importlow.com'] == 1, stored


# 3. stream_run ------------------------------------------------------------


def test_stream_unknown_run_404(
  tmp_path: Path, monkeypatch,
) -> None:
  '''GET stream for an unknown run_id is 404, not a long poll.'''
  client = _make_client(tmp_path, monkeypatch)
  response = client.get('/api/runs/999999/stream')
  assert response.status_code == 404, response.text


def test_stream_known_run_stats_is_dict(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Done frame carries stats as a parsed object, not a JSON string.'''
  client = _make_client(tmp_path, monkeypatch)
  created = client.post('/api/runs/discovery')
  assert created.status_code == 200, created.text
  run_id = created.json()['run_id']
  from job_hunter.db.repositories.runs import RunsRepository
  RunsRepository(tmp_path / 'app.db').finish(
    run_id, 'success', {'jobs': 3},
  )
  response = client.get(f'/api/runs/{run_id}/stream')
  assert response.status_code == 200, response.text
  done_payload = None
  for line in response.text.splitlines():
    if line.startswith('data: ') and '"status"' in line:
      done_payload = json.loads(line[len('data: '):])
  assert done_payload is not None, response.text
  assert isinstance(done_payload['stats'], dict), done_payload
  assert done_payload['stats'] == {'jobs': 3}, done_payload


# 4. Dead fallback import --------------------------------------------------


def test_main_no_llm_gateway_errors_fallback() -> None:
  '''api/main.py imports GatewayUnavailableError from core.errors only.'''
  source = (
    SRC / 'job_hunter' / 'api' / 'main.py'
  ).read_text(encoding='utf-8')
  assert 'llm_gateway.errors' not in source
  assert 'GatewayUnavailableError' in source
  assert 'core.errors' in source
  assert 'llm/client.py' in source
  assert 'wrapped' in source.lower()


def test_main_import_still_works(tmp_path: Path, monkeypatch) -> None:
  '''create_app importable and health endpoint returns 200.'''
  from job_hunter.api.main import create_app
  assert callable(create_app)
  client = _make_client(tmp_path, monkeypatch)
  response = client.get('/healthz')
  assert response.status_code == 200, response.text
  assert response.json()['status'] == 'ok', response.text


# 5. Atomicity documentation -----------------------------------------------


def test_put_profile_and_patch_company_document_mirror_write() -> None:
  '''Both routes document the intentional two-session mirror-write.'''
  profiles = (
    SRC / 'job_hunter' / 'api' / 'routes' / 'profiles.py'
  ).read_text(encoding='utf-8')
  companies = (
    SRC / 'job_hunter' / 'api' / 'routes' / 'companies.py'
  ).read_text(encoding='utf-8')
  for label, text in (('profiles', profiles), ('companies', companies)):
    assert 'mirror' in text.lower(), label
    assert 'intentional' in text.lower(), label
    assert 'LOW' in text, label
  assert 'two-session' in profiles.lower() or 'two session' in profiles.lower()
  assert 'settings mirror' in profiles.lower()
  assert 'profile row' in profiles.lower()
  assert 'two-session' in companies.lower() or (
    'two session' in companies.lower()
  )


# 6. Web href control chars ------------------------------------------------


def test_api_js_href_rejects_control_chars() -> None:
  '''api.js guards hrefs against control chars before the allowlist.'''
  text = (WEB / 'js' / 'api.js').read_text(encoding='utf-8')
  assert r'[\u0000-\u0020]' in text
  guard = re.compile(r'[\x00-\x20]')
  for bad in (
    'java\tscript:alert(1)', '/path\n', '\x01/path', '/pa\x1fth',
    ' https://example.com', '/path ',
  ):
    assert guard.search(bad) is not None, repr(bad)
  for good in ('/path', 'https://example.com/path', '#frag', '/'):
    assert guard.search(good) is None, repr(good)


# 7. Web button / th scope / aria-live --------------------------------------


def test_runs_html_uses_button_for_run_action() -> None:
  '''Run action is a real button, not an anchor with href="#".'''
  text = (WEB / 'runs.html').read_text(encoding='utf-8')
  assert "jh.el('button'" in text
  assert "type: 'button'" in text
  assert "href: '#'" not in text
  assert 'jh.el(\'a\', { href: \'#\'' not in text


def test_th_scope_col_in_components_and_dashboard() -> None:
  '''Generated table headers carry scope="col" in both renderers.'''
  components = (WEB / 'js' / 'components.js').read_text(encoding='utf-8')
  assert 'scope' in components
  assert "'col'" in components or '"col"' in components
  dashboard_js = (
    SRC / 'llm_gateway' / 'templates' / 'dashboard.js'
  ).read_text(encoding='utf-8')
  assert 'scope' in dashboard_js
  assert "'col'" in dashboard_js or '"col"' in dashboard_js


def test_dashboard_summary_cards_no_aria_live() -> None:
  '''Polling summary-cards container is a plain div: no aria-live/role.'''
  html = (
    SRC / 'llm_gateway' / 'templates' / 'dashboard.html'
  ).read_text(encoding='utf-8')
  summary_line = next(
    (line for line in html.splitlines() if 'summary-cards' in line), '',
  )
  assert summary_line, 'summary-cards container not found'
  assert 'aria-live' not in summary_line, summary_line
  assert "role='status'" not in summary_line
  assert 'role="status"' not in summary_line
  assert summary_line.strip().startswith('<div')
  dashboard_js = (
    SRC / 'llm_gateway' / 'templates' / 'dashboard.js'
  ).read_text(encoding='utf-8')
  assert 'aria-live' not in dashboard_js
