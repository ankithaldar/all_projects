#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle-3 fixes for API security, token UI, and dashboard CSP.'''


from __future__ import annotations

import re
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from job_hunter.api.main import create_app
from job_hunter.core.config import is_loopback_host
from llm_gateway.dashboard import _DashboardHandler

fastapi = pytest.importorskip('fastapi')
TestClient = pytest.importorskip('fastapi.testclient').TestClient

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / 'web'
GATEWAY = ROOT / 'src' / 'job_hunter' / 'llm_gateway'
API_JS = WEB / 'js' / 'api.js'
PROFILE_HTML = WEB / 'profile.html'
DASHBOARD_HTML = GATEWAY / 'templates' / 'dashboard.html'


def _make_client(
  tmp_path: Path, monkeypatch, *, token: str = '', host: str = '127.0.0.1',
) -> TestClient:
  '''Build an app client against a temporary data directory.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest monkeypatch fixture.
    token: Bearer token to configure.
    host: Bind host to configure.

  Returns:
    Configured TestClient.
  '''
  monkeypatch.setenv('APP_DATA_DIR', str(tmp_path))
  monkeypatch.setenv('APP_HOST', host)
  monkeypatch.setenv('JH_API_TOKEN', token)
  monkeypatch.setenv('APP_LOG_LEVEL', 'INFO')
  config_path = tmp_path / 'app.yaml'
  config_path.write_text(
    'salary_hard_floor_lpa: 45\n', encoding='utf-8',
  )
  app = create_app(str(config_path))
  return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(name='client')
def _client_fixture(tmp_path: Path, monkeypatch) -> TestClient:
  '''Provide a default local-only client.

  Args:
    tmp_path: Pytest temp dir.
    monkeypatch: Pytest monkeypatch fixture.

  Returns:
    Configured TestClient.
  '''
  return _make_client(tmp_path, monkeypatch)


def test_csrf_evil_origin_post_403(client: TestClient) -> None:
  '''Cross-origin POST to the API is rejected with 403.'''
  response = client.post(
    '/api/runs/discovery', headers={'Origin': 'https://evil.com'},
  )
  assert response.status_code == 403
  assert response.json() == {'detail': 'forbidden origin'}


def test_csrf_evil_referer_post_403(client: TestClient) -> None:
  '''Cross-origin Referer on a POST is rejected with 403.'''
  headers = {'Referer': 'https://evil.example.com/page'}
  response = client.post('/api/runs/discovery', headers=headers)
  assert response.status_code == 403
  assert response.json() == {'detail': 'forbidden origin'}


def test_csrf_same_origin_post_not_403(client: TestClient) -> None:
  '''Same-origin POST from the local UI is not blocked.'''
  headers = {'Origin': 'http://testserver'}
  response = client.post('/api/runs/discovery', headers=headers)
  assert response.status_code != 403


def test_csrf_loopback_origin_post_not_403(client: TestClient) -> None:
  '''Loopback Origin is allowed even when Host differs.'''
  headers = {'Origin': 'http://127.0.0.1:8088'}
  response = client.post('/api/runs/discovery', headers=headers)
  assert response.status_code != 403


def test_csrf_get_not_checked(client: TestClient) -> None:
  '''GET requests are not subject to the origin check.'''
  response = client.get(
    '/api/profile', headers={'Origin': 'https://evil.com'},
  )
  assert response.status_code == 200


def test_api_js_token_ui_fetch_stream() -> None:
  '''api.js stores a token and sends it without a query fallback.'''
  text = API_JS.read_text(encoding='utf-8')
  assert 'sessionStorage' in text
  assert 'jh_token' in text
  assert 'Authorization' in text
  assert 'Bearer ' in text
  assert 'prompt' in text
  assert 'JH_API_TOKEN' in text
  assert 'token=' not in text
  assert 'EventSource' not in text
  assert 'getReader' in text


def test_api_js_href_rejects_protocol_relative() -> None:
  '''Relative hrefs reject protocol-relative // URLs.'''
  text = API_JS.read_text(encoding='utf-8')
  assert r'(?![\/\\])' in text
  assert r'^\/(?!\/)' not in text
  assert r'^\/$' in text
  assert r'^#' in text
  assert r'^\.' in text
  assert 'https?:' in text


def test_api_js_href_regex_logic() -> None:
  '''Actual api.js href patterns block // and /\\, allow safe.'''
  text = API_JS.read_text(encoding='utf-8')
  bodies: list[tuple[str, int]] = []
  for match in re.finditer(r'/\^', text):
    pos = match.start() + 1
    chars: list[str] = []
    while pos < len(text):
      char = text[pos]
      if char == '\\':
        chars.append(text[pos:pos + 2])
        pos += 2
        continue
      if char == '/':
        break
      chars.append(char)
      pos += 1
    else:
      raise AssertionError('unterminated href pattern')
    flags = 0
    if text[pos + 1:pos + 2] == 'i':
      flags = re.IGNORECASE
    bodies.append((''.join(chars), flags))
  assert len(bodies) == 4
  patterns = [re.compile(body, flags) for body, flags in bodies]
  safe = ['/', '/jobs', '#', 'https://example.com']
  for value in safe:
    assert any(p.search(value) for p in patterns), value
  unsafe = ['//evil.com', r'/\evil.com']
  for value in unsafe:
    assert not any(p.search(value) for p in patterns), value


def test_api_js_sse_timeout_notifies() -> None:
  '''SSE timeout closes the stream and reports timeout status.'''
  text = API_JS.read_text(encoding='utf-8')
  assert 'timeout' in text
  assert 'es.close()' in text
  assert "status: 'timeout'" in text or 'status:' in text


def test_profile_html_floor_zero() -> None:
  '''Profile form sends an explicit zero salary floor.'''
  text = PROFILE_HTML.read_text(encoding='utf-8')
  floor_line = [
    line for line in text.splitlines() if 'salary_floor_lpa' in line
  ]
  assert floor_line
  joined = '\n'.join(floor_line)
  assert '|| null' not in joined
  assert 'trim()' in text


def test_profile_html_resume_auth() -> None:
  '''Resume upload delegates bearer auth to fetchWithAuth.'''
  text = PROFILE_HTML.read_text(encoding='utf-8')
  assert 'fetchWithAuth' in text
  assert '/api/profile/resume' in text
  assert 'sessionStorage' not in text
  assert 'Authorization' not in text
  api_text = API_JS.read_text(encoding='utf-8')
  assert 'fetchWithAuth' in api_text
  assert 'sessionStorage' in api_text
  assert 'Authorization' in api_text
  assert 'Bearer ' in api_text
  assert 'jh_token' in api_text


def test_profile_salary_floor_zero_persists(
  client: TestClient,
) -> None:
  '''PUT with salary_floor_lpa 0 persists and reads back 0.'''
  response = client.put(
    '/api/profile', json={'salary_floor_lpa': 0},
  )
  assert response.status_code == 200, response.text
  saved = client.get('/api/profile').json()
  assert saved['salary_floor_lpa'] == 0


def test_is_loopback_empty_false() -> None:
  '''Empty host is not loopback.'''
  assert is_loopback_host('') is False


def test_app_host_empty_without_token_raises(
  tmp_path: Path, monkeypatch,
) -> None:
  '''Empty bind host without a token fails fast.'''
  with pytest.raises(RuntimeError):
    _make_client(tmp_path, monkeypatch, token='', host='')


def test_companies_import_invalid_domain_skipped(
  client: TestClient,
) -> None:
  '''Import skips entries whose domain fails create validation.'''
  yaml_text = (
    'companies:\n'
    '  - name: Good Co\n'
    '    domain: good.example.com\n'
    '    priority: 3\n'
    '  - name: Evil Co\n'
    '    domain: not a domain\n'
    '    priority: 3\n'
  )
  response = client.post('/api/companies/import', json=yaml_text)
  assert response.status_code == 200, response.text
  body = response.json()
  assert body['imported'] == 1
  assert body['errors'] == 1
  names = [row['name'] for row in client.get('/api/companies').json()]
  assert 'Good Co' in names
  assert 'Evil Co' not in names


def test_dashboard_template_no_inline() -> None:
  '''Dashboard template links external assets, no inline bodies.'''
  text = DASHBOARD_HTML.read_text(encoding='utf-8')
  assert '<script>' not in text
  assert '<style>' not in text
  assert '/dashboard.js' in text
  assert '/dashboard.css' in text
  assert '<script src=' in text
  assert '<link' in text


def test_dashboard_assets_and_csp() -> None:
  '''Dashboard serves external JS/CSS with a strict CSP.'''
  server = ThreadingHTTPServer(('127.0.0.1', 0), _DashboardHandler)
  port = server.server_address[1]
  thread = threading.Thread(
    target=server.serve_forever, daemon=True,
  )
  thread.start()
  try:
    base = f'http://127.0.0.1:{port}'
    with urllib.request.urlopen(base + '/') as response:
      html = response.read().decode('utf-8')
      csp = response.headers.get('Content-Security-Policy') or ''
      assert response.status == 200
      assert 'unsafe-inline' not in csp
      assert "default-src 'self'" in csp
      assert '<script>' not in html
      assert '<style>' not in html
    with urllib.request.urlopen(base + '/dashboard.js') as response:
      assert response.status == 200
    with urllib.request.urlopen(base + '/dashboard.css') as response:
      assert response.status == 200
  finally:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
