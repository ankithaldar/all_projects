#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Contract tests for W6 frontend and gateway fixes.'''


from __future__ import annotations

import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from llm_gateway.cache import SQLiteCache
from llm_gateway.sanitize import sanitize_think_tokens
from llm_gateway.schemas import GatewayRequest

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / 'web'
GATEWAY = ROOT / 'src' / 'job_hunter' / 'llm_gateway'
DASHBOARD_HTML = GATEWAY / 'templates' / 'dashboard.html'
DASHBOARD_PY = GATEWAY / 'dashboard.py'
APP_PY = GATEWAY / 'app.py'
API_JS = WEB / 'js' / 'api.js'
COMPONENTS_JS = WEB / 'js' / 'components.js'
CSS = WEB / 'css' / 'style.css'
WEB_HTMLS = sorted(WEB.glob('*.html'))


def _read(path: Path) -> str:
  '''Read a file as UTF-8 text.

  Args:
    path: File to read.

  Returns:
    File contents.
  '''
  return path.read_text(encoding='utf-8')


def test_dashboard_no_innerhtml_interpolation() -> None:
  '''Dashboard must not interpolate data via innerHTML.'''
  text = _read(DASHBOARD_HTML)
  assert 'innerHTML' not in text
  assert 'textContent' in text
  assert 'createElement' in text


def test_api_js_href_allowlist() -> None:
  '''api.js must allowlist http(s) hrefs in jh.el.'''
  text = _read(API_JS)
  assert re.search(r'\^https\?:', text)
  assert 'noopener noreferrer' in text
  assert 'href' in text


def test_components_jh_el_links() -> None:
  '''components.js must use jh.el for links, no raw href set.'''
  text = _read(COMPONENTS_JS)
  assert 'setAttribute' not in text
  assert re.search(r'jh\.el\(.a.', text)


def test_api_js_sse_timeout_and_safe_json() -> None:
  '''SSE must handle timeout, safe JSON, and close on done.'''
  text = _read(API_JS)
  assert 'timeout' in text
  assert 'es.close()' in text
  assert 'try' in text and 'catch' in text
  assert 'JSON.parse' in text


def test_jobs_and_companies_load_clears_list() -> None:
  '''jobs/companies load() must clear the list before append.'''
  for name in ('jobs.html', 'companies.html'):
    text = _read(WEB / name)
    assert 'replaceChildren()' in text


def test_all_web_htmls_viewport_and_single_h1() -> None:
  '''Every web page has a viewport meta and exactly one h1.'''
  assert len(WEB_HTMLS) == 7
  for path in WEB_HTMLS:
    text = _read(path)
    assert 'viewport' in text, path.name
    assert 'width=device-width' in text, path.name
    assert text.count('<h1') == 1, path.name


def test_all_web_htmls_labels_for_and_inputs_id() -> None:
  '''Every label has for= and every input/select/textarea an id.'''
  for path in WEB_HTMLS:
    text = _read(path)
    for match in re.finditer(r'<label([^>]*)>', text):
      assert 'for=' in match.group(1), path.name
    for match in re.finditer(r'<input([^>]*)>', text):
      assert 'id=' in match.group(1), path.name
    for match in re.finditer(r'<(select|textarea)([^>]*)>', text):
      assert 'id=' in match.group(2), path.name


def test_all_web_htmls_buttons_and_status_role() -> None:
  '''Every button has type=; toast has role status live.'''
  for path in WEB_HTMLS:
    text = _read(path)
    for match in re.finditer(r'<button([^>]*)>', text):
      assert 'type=' in match.group(1), path.name
  api_text = _read(API_JS)
  assert 'role' in api_text and 'status' in api_text
  assert 'aria-live' in api_text


def test_css_table_wrap_nav_focus_dismiss_debug() -> None:
  '''CSS has table-wrap, nav wrap, focus, dismiss, debug.'''
  text = _read(CSS)
  assert '.table-wrap' in text
  assert 'overflow-x:auto' in text.replace(' ', '')
  nav = re.search(r'\.nav\s*\{[^}]*\}', text)
  assert nav and 'flex-wrap:wrap' in nav.group(0).replace(' ', '')
  assert '.btn.save.saved' not in text
  assert '.btn.dismiss' in text
  assert '.console .debug' in text
  assert ':focus-visible' in text
  assert 'table-wrap' in _read(COMPONENTS_JS)


def test_every_load_has_try_catch() -> None:
  '''Every page load() must try/catch and show errors.'''
  for path in WEB_HTMLS:
    text = _read(path)
    if 'load()' in text or 'async function load' in text:
      assert 'try' in text and 'catch' in text, path.name
      assert 'rror' in text, path.name


def test_profile_and_settings_save_disabled() -> None:
  '''Profile/settings save buttons start disabled.'''
  for name in ('profile.html', 'settings.html'):
    text = _read(WEB / name)
    assert 'disabled' in text, name


def test_companies_priority_plain_text() -> None:
  '''Companies priority is plain P{n} text, not a score badge.'''
  text = _read(WEB / 'companies.html')
  assert 'scoreBadge' not in text
  assert 'P${' in text


def test_runs_runid_no_innerhtml() -> None:
  '''Runs page must not use innerHTML for the run id.'''
  text = _read(WEB / 'runs.html')
  assert 'innerHTML' not in text
  assert 'textContent' in text or 'jh.el' in text


def test_dashboard_py_security_headers() -> None:
  '''Dashboard handler sends security headers, uses with.'''
  text = _read(DASHBOARD_PY)
  assert 'default-src' in text
  assert 'X-Content-Type-Options' in text and 'nosniff' in text
  assert 'X-Frame-Options' in text and 'DENY' in text
  assert 'with sqlite3.connect' in text


def test_cache_key_differs_for_model_tokens_session() -> None:
  '''Cache key differs for model, max_tokens, and session.'''
  req = GatewayRequest(prompt='hello')
  base = SQLiteCache.make_key(req, include_temperature=False)
  key_m1 = SQLiteCache.make_key(
    req, include_temperature=False, model='m1')
  key_m2 = SQLiteCache.make_key(
    req, include_temperature=False, model='m2')
  assert key_m1 != base and key_m1 != key_m2
  key_t10 = SQLiteCache.make_key(
    req, include_temperature=False, max_tokens=10)
  key_t99 = SQLiteCache.make_key(
    req, include_temperature=False, max_tokens=99)
  assert key_t10 != key_t99
  key_s1 = SQLiteCache.make_key(
    req, include_temperature=False, session_id='s1')
  key_s2 = SQLiteCache.make_key(
    req, include_temperature=False, session_id='s2')
  assert key_s1 != key_s2
  req_a = GatewayRequest(prompt='hello', max_tokens=10, session_id='a')
  req_b = GatewayRequest(prompt='hello', max_tokens=99, session_id='b')
  key_a = SQLiteCache.make_key(req_a, include_temperature=False)
  key_b = SQLiteCache.make_key(req_b, include_temperature=False)
  assert key_a != key_b


def test_cache_corrupt_returns_none(tmp_path: Path) -> None:
  '''Corrupt cache payload returns None and is deleted.'''
  cache = SQLiteCache(tmp_path / 'cache.db', ttl_seconds=60)
  cache.set('k1', {'content': 'ok'})
  with closing(sqlite3.connect(tmp_path / 'cache.db')) as conn:
    conn.execute(
      'UPDATE cache SET payload = ? WHERE key = ?',
      ('not-json{', 'k1'))
    conn.commit()
  assert cache.get('k1') is None
  with closing(sqlite3.connect(tmp_path / 'cache.db')) as conn:
    row = conn.execute(
      'SELECT payload FROM cache WHERE key = ?', ('k1',)).fetchone()
  assert row is None


def test_cache_expired_deleted_on_access(tmp_path: Path) -> None:
  '''Expired cache entries are deleted on access.'''
  cache = SQLiteCache(tmp_path / 'cache.db', ttl_seconds=1)
  cache.set('k2', {'content': 'old'})
  with closing(sqlite3.connect(tmp_path / 'cache.db')) as conn:
    conn.execute(
      'UPDATE cache SET created_at = created_at - 100 WHERE key = ?',
      ('k2',))
    conn.commit()
  assert cache.get('k2') is None
  with closing(sqlite3.connect(tmp_path / 'cache.db')) as conn:
    row = conn.execute(
      'SELECT payload FROM cache WHERE key = ?', ('k2',)).fetchone()
  assert row is None


def test_sanitize_normal_think_block_stripped() -> None:
  '''Normal think/reasoning/bracket blocks are stripped.'''
  assert sanitize_think_tokens(
    'before <think>secret</think> after') == 'before  after'
  assert sanitize_think_tokens(
    'hi <reasoning>secret</reasoning> bye') == 'hi  bye'
  assert sanitize_think_tokens(
    'hi [think]secret[/think] bye') == 'hi  bye'
  assert sanitize_think_tokens('hi <think>unclosed secret') == 'hi'


def test_sanitize_pathological_input_fast() -> None:
  '''Pathological unclosed think input completes in under 1s.'''
  payload = '<think>' * 5000 + 'hello'
  start = time.perf_counter()
  out = sanitize_think_tokens(payload)
  elapsed = time.perf_counter() - start
  assert elapsed < 1.0
  assert '<think>' not in out.lower()
  assert 'hello' not in out


def test_sanitize_caps_input() -> None:
  '''Sanitizer caps input at 200k characters.'''
  out = sanitize_think_tokens('a' * 300000)
  assert len(out) <= 200000


def test_app_fallback_respects_limiter_and_cache() -> None:
  '''app.py fallback respects limiter; cache key passes args.'''
  text = _read(APP_PY)
  assert 'not is_fallback' not in text
  assert text.count('max_tokens=') >= 1
  assert text.count('session_id=') >= 1
