#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Contract tests for W7 devops/docs fixes (config/docs/scripts).'''

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _pyproject() -> dict:
  '''Parse pyproject.toml.

  Returns:
    The parsed TOML mapping.
  '''
  text = (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
  return tomllib.loads(text)


def test_pyproject_parses_and_ruff_indent_is_space() -> None:
  '''pyproject parses and ruff indent-style is valid 'space'.'''
  data = _pyproject()
  assert data['tool']['ruff']['format']['indent-style'] == 'space'


def test_pyproject_testpaths_includes_gateway_tests() -> None:
  '''pytest collects both app and gateway test suites.'''
  data = _pyproject()
  testpaths = data['tool']['pytest']['ini_options']['testpaths']
  assert 'tests' in testpaths
  assert 'src/job_hunter/llm_gateway/tests' in testpaths


def test_pyproject_deps_requests_in_no_tenacity_no_pydantic_settings() -> None:
  '''requests declared; unused tenacity/pydantic-settings absent.'''
  data = _pyproject()
  deps = data['project']['dependencies']
  assert any(d.startswith('requests>=2.31') for d in deps)
  joined = '\n'.join(deps)
  assert 'tenacity' not in joined
  assert 'pydantic-settings' not in joined


def test_pyproject_dev_extras_include_pytest_cov() -> None:
  '''dev extras include pytest-cov.'''
  data = _pyproject()
  dev = data['project']['optional-dependencies']['dev']
  assert any(d.startswith('pytest-cov') for d in dev)


def test_gitignore_ignores_env_and_egg_info_and_db() -> None:
  '''job_hunter/.gitignore blocks env, egg-info, and db files.'''
  text = (ROOT / '.gitignore').read_text(encoding='utf-8')
  needles = (
    '.env', '.env.*', '*.egg-info/',
    'src/job_hunter/llm_gateway/.env', '*.db',
  )
  for needle in needles:
    assert needle in text, f'missing {needle} in .gitignore'


def test_setup_sh_copies_gateway_template_and_gates_python() -> None:
  '''setup.sh copies gateway .env.example and enforces python >=3.14.'''
  text = (ROOT / 'setup.sh').read_text(encoding='utf-8')
  assert 'src/job_hunter/llm_gateway/.env.example' in text
  assert '3.14' in text
  assert 'exit 1' in text


def test_run_sh_waits_for_healthz() -> None:
  '''run.sh polls /healthz before printing the UI URL.'''
  text = (ROOT / 'run.sh').read_text(encoding='utf-8')
  assert '/healthz' in text


def test_readme_quick_start_has_discover_companies_and_semantic() -> None:
  '''README quick-start mentions discover-companies and .[dev,semantic].'''
  text = (ROOT / 'README.md').read_text(encoding='utf-8')
  assert 'discover-companies' in text
  assert '.[dev,semantic]' in text


def test_app_yaml_parses_and_has_workday() -> None:
  '''config/app.yaml parses and registers the workday source.'''
  path = ROOT / 'config' / 'app.yaml'
  data = yaml.safe_load(path.read_text(encoding='utf-8'))
  assert 'workday' in data['sources']


def test_runbook_and_api_docs_exist() -> None:
  '''docs/runbook.md and docs/api.md are present.'''
  assert (ROOT / 'docs' / 'runbook.md').is_file()
  assert (ROOT / 'docs' / 'api.md').is_file()


def test_root_env_example_is_app_only() -> None:
  '''Root .env.example carries APP_* vars only, pointing at gateway template.'''
  text = (ROOT / '.env.example').read_text(encoding='utf-8')
  assert 'APP_' in text
  assert 'llm_gateway/.env.example' in text
  assert 'OPENROUTER_API_KEYS' not in text


def test_gateway_main_uses_package_relative_paths() -> None:
  '''llm_gateway/main.py resolves config/.env from __file__, not CWD.'''
  path = ROOT / 'src' / 'job_hunter' / 'llm_gateway' / 'main.py'
  text = path.read_text(encoding='utf-8')
  assert '__file__' in text
  assert 'PACKAGE_ROOT' in text
  assert 'gateway.yaml' in text
