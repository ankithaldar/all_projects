#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle5 API/web fixes: streaming cap, orphan race, CSP, href, SSE, auth.'''

from __future__ import annotations

import asyncio
import hashlib
import re
import sqlite3
import threading
import urllib.error
import urllib.request
from contextlib import closing
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

fastapi = pytest.importorskip('fastapi')
TestClient = pytest.importorskip('fastapi.testclient').TestClient

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / 'web'
API_JS = WEB / 'js' / 'api.js'
PROFILE_HTML = WEB / 'profile.html'


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


# 1. Streaming upload cap -------------------------------------------------


def test_streaming_upload_cap_chunked_413(
  tmp_path: Path, monkeypatch,
) -> None:
  '''11MB via chunked reads -> 413 and every read is size-bounded.'''
  client = _make_client(tmp_path, monkeypatch)
  from starlette.datastructures import UploadFile
  calls: list = []
  offsets: dict = {}
  total = 11 * 1024 * 1024

  async def fake_read(self, size=-1):
    '''Return sequential chunks, recording the requested size.

    Args:
      self: UploadFile instance.
      size: Requested chunk size.

    Returns:
      Next chunk bytes.
    '''
    calls.append(size)
    if size is None or size <= 0:
      return b'x' * total
    off = offsets.get(id(self), 0)
    if off >= total:
      return b''
    chunk_len = min(size, total - off)
    offsets[id(self)] = off + chunk_len
    return b'x' * chunk_len

  monkeypatch.setattr(UploadFile, 'read', fake_read)
  files = {'file': ('r.pdf', b'%PDF small', 'application/pdf')}
  response = client.post('/api/profile/resume', files=files)
  assert response.status_code == 413, response.text
  assert calls, 'UploadFile.read was never called'
  assert all(call == 1_000_000 for call in calls), f'calls={calls}'
  assert max(calls) <= 1_000_000


# 2. Orphan cleanup race --------------------------------------------------


def test_curator_unique_filename_same_second(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Same-second same-name uploads get distinct paths, no overwrite.'''
  settings = _settings(tmp_path, monkeypatch)
  monkeypatch.setattr(
    'job_hunter.services.profile_curator.time.time',
    lambda: 1700000000.0,
  )
  curator = _make_curator(settings)
  curator._parser.parse_file = _parsed_fail
  first = asyncio.run(curator.ingest_upload('resume.pdf', b'%PDF-one'))
  second = asyncio.run(curator.ingest_upload('resume.pdf', b'%PDF-two'))
  assert first.get('file_path'), first
  assert second.get('file_path'), second
  assert first['file_path'] != second['file_path']
  assert Path(first['file_path']).exists()
  assert Path(second['file_path']).exists()
  assert Path(first['file_path']).read_bytes() == b'%PDF-one'
  assert Path(second['file_path']).read_bytes() == b'%PDF-two'
  source = (
    ROOT / 'src' / 'job_hunter' / 'job_hunter' / 'services'
    / 'profile_curator.py'
  ).read_text(encoding='utf-8')
  assert 'uuid.uuid4().hex[:8]' in source


def test_failed_extract_no_file_no_row(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Extract failure leaves no resume file and no dangling DB row.'''
  settings = _settings(tmp_path, monkeypatch)
  curator = _make_curator(settings)
  curator._parser.parse_file = _parsed_ok

  async def boom(_text):
    '''Raise as a failed LLM extraction would.

    Args:
      _text: Resume text.

    Raises:
      RuntimeError: Simulated extraction failure.
    '''
    raise RuntimeError('extract failed')

  curator.extract = boom
  with pytest.raises(RuntimeError):
    asyncio.run(curator.ingest_upload('resume.pdf', b'%PDF-good'))
  resumes_dir = settings.data_dir / 'resumes'
  leftover = list(resumes_dir.iterdir()) if resumes_dir.exists() else []
  assert leftover == [], f'leftover files={leftover}'
  assert _resume_rows(settings.db_path) == []


def test_route_deletes_exact_path_not_concurrent(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Route deletes only the curator path, not a concurrent file.'''
  client = _make_client(tmp_path, monkeypatch)
  from job_hunter.services.profile_curator import ProfileCurator
  resumes_dir = tmp_path / 'resumes'
  resumes_dir.mkdir(parents=True, exist_ok=True)
  own = resumes_dir / 'own.pdf'
  concurrent = resumes_dir / 'concurrent.pdf'

  async def fake_ingest(self, file_name, data, candidate_id=1):
    '''Create own and concurrent files, then fail with exact path.

    Args:
      self: Curator instance.
      file_name: Upload name.
      data: Upload bytes.
      candidate_id: Candidate id.

    Raises:
      RuntimeError: Simulated curator failure.
    '''
    own.write_bytes(b'own')
    concurrent.write_bytes(b'concurrent')
    err = RuntimeError('curator failed')
    err.file_path = str(own)
    raise err

  monkeypatch.setattr(ProfileCurator, 'ingest_upload', fake_ingest)
  files = {'file': ('r.pdf', b'%PDF small', 'application/pdf')}
  response = client.post('/api/profile/resume', files=files)
  assert response.status_code == 422, response.text
  assert not own.exists(), 'exact curator file must be deleted'
  assert concurrent.exists(), 'concurrent file must not be deleted'


# 3. Dashboard 404 headers -------------------------------------------------


def test_dashboard_404_has_csp() -> None:
  '''GET /nope -> 404 WITH Content-Security-Policy header.'''
  from llm_gateway.dashboard import _DashboardHandler
  server = ThreadingHTTPServer(('127.0.0.1', 0), _DashboardHandler)
  port = server.server_address[1]
  thread = threading.Thread(
    target=server.serve_forever, daemon=True,
  )
  thread.start()
  try:
    base = f'http://127.0.0.1:{port}'
    with pytest.raises(urllib.error.HTTPError) as excinfo:
      urllib.request.urlopen(base + '/nope')
    assert excinfo.value.code == 404
    csp = excinfo.value.headers.get('Content-Security-Policy') or ''
    assert "default-src 'self'" in csp, f'CSP missing: {csp!r}'
    excinfo.value.close()
  finally:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


# 4. href backslash --------------------------------------------------------


def test_href_regex_rejects_backslash_allows_root() -> None:
  '''Href pattern from api.js rejects /\\evil.com, allows / and /path.'''
  text = API_JS.read_text(encoding='utf-8')
  anchor = text.find('else if (/^')
  assert anchor != -1, 'relative href branch not found'
  start = text.find('/^', anchor)
  end = text.find('/.test(value)', start)
  assert start != -1 and end != -1, 'href regex literal not found'
  pattern_str = text[start + 1:end]
  assert '^\\/$' in pattern_str, pattern_str
  assert '(?![\\/\\\\])' in pattern_str, pattern_str
  pattern = re.compile(pattern_str)
  assert pattern.search('/') is not None
  assert pattern.search('/path') is not None
  assert pattern.search('/\\evil.com') is None
  assert pattern.search('//evil.com') is None
  assert pattern.search('/\\') is None


# 5. SSE multi-line --------------------------------------------------------


def test_sse_multiline_join_single_space() -> None:
  '''SSE parser joins data lines with \\n, strips one leading space.'''
  text = API_JS.read_text(encoding='utf-8')
  assert "join('\\n')" in text, 'multi-line data must join with newline'
  assert 'line.slice(5).trim()' not in text
  assert "data += line" not in text
  assert ".startsWith(' ')" in text
  assert 'slice(1)' in text


# 6. Resume via fetchWithAuth ---------------------------------------------


def test_profile_html_uses_fetch_with_auth() -> None:
  '''profile.html uses fetchWithAuth and never reads sessionStorage.'''
  text = PROFILE_HTML.read_text(encoding='utf-8')
  assert 'sessionStorage' not in text
  assert "fetchWithAuth('/api/profile/resume'" in text
  assert "method:'POST'" in text
  assert 'body: formData' in text
  assert "fetch('/api/profile/resume'" not in text
