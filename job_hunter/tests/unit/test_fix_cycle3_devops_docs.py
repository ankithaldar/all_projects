#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Contract tests for cycle-3 devops/docs fixes (config/docs/scripts).'''

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(relpath: str) -> str:
  '''Read a repo-relative text file.

  Args:
    relpath: Path relative to the job_hunter root.

  Returns:
    File contents as text.
  '''
  return (ROOT / relpath).read_text(encoding='utf-8')


def _section(text: str, start: str, end: str) -> str:
  '''Extract a markdown section between two headings.

  Args:
    text: Full document text.
    start: Heading marker for the section start.
    end: Heading marker for the section end.

  Returns:
    Text between the markers.
  '''
  begin = text.index(start)
  finish = text.index(end, begin)
  return text[begin:finish]


def test_pyproject_has_checkpoint_sqlite_dep() -> None:
  '''pyproject declares langgraph-checkpoint-sqlite>=3.1.'''
  data = tomllib.loads(_read('pyproject.toml'))
  deps = data['project']['dependencies']
  assert any(dep.startswith('langgraph-checkpoint-sqlite>=3.1') for dep in deps)


def test_readme_has_no_requirements_txt_refs() -> None:
  '''README no longer points at a deleted requirements.txt.'''
  assert 'requirements.txt' not in _read('README.md')


def test_readme_python314_gateway_db_token_trustedhost() -> None:
  '''README documents Python >=3.14, gateway_db, token UI, TrustedHost.'''
  text = _read('README.md')
  assert 'Python >=3.14' in text or 'Python >= 3.14' in text
  assert 'gateway_db' in text
  assert '401' in text
  assert 'TrustedHost' in text


def test_api_md_resume_is_422_not_502() -> None:
  '''API docs describe resume failures as 422, never 502.'''
  text = _read('docs/api.md')
  assert '422' in text
  # Post-F1: 502 is the AdapterError mapping, NOT a resume mapping.
  # Resume failures must stay 422; 502 may appear only for adapters.
  for line in text.splitlines():
    if 'resume' in line.lower():
      assert '502' not in line, line
  assert 'AdapterError' in text


def test_api_md_error_shapes_401_csrf_and_params() -> None:
  '''API docs cover both error shapes, 401, CSRF, gated/page params.'''
  text = _read('docs/api.md')
  assert '401' in text
  assert '{detail}' in text
  assert 'title' in text and 'status' in text
  assert 'Origin' in text
  assert 'status=new' in text or "'new'" in text or '`new`' in text
  assert 'include_gated' in text
  assert 'page' in text
  assert 'gateway_db' in text
  assert 'TrustedHost' in text


def test_architecture_s8_lists_actual_mcp_tools() -> None:
  '''Architecture section 8 matches the real @mcp.tool defs.'''
  text = _read('docs/architecture.md')
  s8 = _section(text, '## 8.', '## 9.')
  for name in (
    'detect_careers',
    'clean_text',
    'fetch_ats_board',
    'fetch_page',
    'search_web',
    'extract_text',
    'get_profile',
    'list_companies',
    'upsert_company',
    'jobs_recent',
    'recommendations_top',
    'record_run_event',
  ):
    assert name in s8, f'missing MCP tool {name} in section 8'
  for stale in (
    'fetch_json',
    'fetch_rss',
    'parse_sections',
    'get_companies_due',
    'insert_raw_jobs',
    'get_jobs_without_enrichment',
    'save_recommendations',
  ):
    assert stale not in s8, f'stale MCP tool {stale} still in section 8'


def test_architecture_postfix_truth() -> None:
  '''Architecture describes post-fix graph/MCP/error behaviour.'''
  text = _read('docs/architecture.md')
  s8 = _section(text, '## 8.', '## 9.')
  assert '422' in text
  assert '401' in text
  assert 'checkpoint_ns' in text
  assert 'persist_record' in text
  assert 'services/persist.py' in text
  assert 'db/repositories/taxonomy.py' in text
  assert 'api/__main__.py' in text
  count = len(list((ROOT / 'tests' / 'unit').glob('test_*.py')))
  assert f'{count} unit test' in text, (
    f'recount {count} unit test modules not in architecture.md'
  )
  assert 'services/llm/client.py' not in text
  assert 'web/js/pages/*.js' not in text
  assert 'GatewayChatModel' not in s8
  assert 'no in-process tool loop' in s8.lower()
  assert 'LLM tool loop' not in s8


def test_runbook_ttl_numbers_and_422() -> None:
  '''Runbook documents orphan TTLs and 422 placeholder-key errors.'''
  text = _read('docs/runbook.md')
  assert '422' in text
  assert '502' not in text
  assert '120' in text
  assert '5 min' in text or '5m' in text
  assert '1h' in text or '1 hour' in text or 'one hour' in text


def test_roadmap_no_502_claim() -> None:
  '''Roadmap states 422 (not 502) for placeholder-key failures.'''
  assert '502' not in _read('docs/roadmap.md')


def test_run_sh_uses_app_host_port() -> None:
  '''run.sh reads APP_HOST/APP_PORT with 127.0.0.1/8088 defaults.'''
  text = _read('run.sh')
  assert 'APP_HOST' in text
  assert 'APP_PORT' in text
  assert '127.0.0.1' in text
  assert '8088' in text


def test_gitignore_patterns() -> None:
  '''.gitignore ignores .env but not .env.example templates.'''
  text = _read('.gitignore')
  assert '.env' in text
  assert '!.env.example' in text
  assert 'src/job_hunter/llm_gateway/.env' in text
  assert '*.egg-info/' in text
  assert '*.db' in text


def test_git_check_ignore_env_vs_example() -> None:
  '''git check-ignore: .env ignored, .env.example not ignored.'''
  ignored = subprocess.run(
    ['git', 'check-ignore', '-q', '.env'],
    cwd=ROOT,
    check=False,
  )
  assert ignored.returncode == 0
  not_ignored = subprocess.run(
    ['git', 'check-ignore', '-q', '.env.example'],
    cwd=ROOT,
    check=False,
  )
  assert not_ignored.returncode == 1


def test_test_writer_covers_all_modules() -> None:
  '''test-writer brief points at the full unit suite.'''
  text = _read('agents/test-writer.md')
  assert '46 unit test' in text or 'tests/unit/' in text
