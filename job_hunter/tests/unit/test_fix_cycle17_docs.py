#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle-17 pins: docs match cited code truth (Cycle-16 findings).'''

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs'
ARCH = (DOCS / 'architecture.md').read_text(encoding='utf-8')
DOCS_TEXT = '\n'.join(
  p.read_text(encoding='utf-8') for p in sorted(DOCS.glob('*.md'))
)
SRC = ROOT / 'src/job_hunter/job_hunter'
GATEWAY = ROOT / 'src/job_hunter/llm_gateway'


def test_no_conda_fallback_or_312_claim() -> None:
  '''Risks: no documented 3.12 conda fallback (requires 3.14, setup refuses).'''
  assert 'conda fallback' not in DOCS_TEXT
  assert 'conda fallback' not in ARCH
  assert '3.12' not in DOCS_TEXT
  assert '3.12' not in ARCH
  # Code truth: pyproject requires >=3.14 and setup.sh refuses older.
  pyproject = (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
  assert "requires-python = '>=3.14'" in pyproject
  setup = (ROOT / 'setup.sh').read_text(encoding='utf-8')
  assert 'sys.version_info >= (3, 14)' in setup
  assert 'requires Python >= 3.14' in setup


def test_llm_gateway_import_path() -> None:
  '''Import: `from llm_gateway import LLMGateway`, not job_hunter prefix.'''
  assert 'from llm_gateway import LLMGateway' in ARCH
  assert 'from job_hunter.llm_gateway import LLMGateway' not in ARCH
  # Code truth: gateway __init__ re-exports Sanitized as LLMGateway via
  # the top-level `llm_gateway` package (Sanitized subclass note).
  init_src = (GATEWAY / '__init__.py').read_text(encoding='utf-8')
  assert (
    'from llm_gateway.sanitized_gateway import '
    'SanitizedLLMGateway as LLMGateway'
  ) in init_src
  sanitized_src = (GATEWAY / 'sanitized_gateway.py').read_text(
    encoding='utf-8'
  )
  assert 'from llm_gateway.app import LLMGateway' in sanitized_src
  assert 'class SanitizedLLMGateway(LLMGateway)' in sanitized_src


def test_nice_to_have_neutral() -> None:
  '''Nice-to-have: unknown maps to 0.5 neutral like must-have.'''
  assert '(unknown => 0.5 neutral)' in ARCH
  # The phrase must sit on the nice-to-have coverage row.
  nice_row = next(
    line for line in ARCH.splitlines() if 'nice-to-have coverage' in line
  )
  assert '(unknown => 0.5 neutral)' in nice_row
  # Code truth: matcher returns 0.5 when there are no nices.
  matcher_src = (SRC / 'services/matcher.py').read_text(encoding='utf-8')
  assert 'if nices else 0.5' in matcher_src
  assert "if row['kind'] == 'nice_to_have'" in matcher_src


def test_ddl_not_null_india_and_quality() -> None:
  '''DDL: india_presence and quality_score are NOT NULL in docs.'''
  assert 'india_presence INTEGER NOT NULL DEFAULT 1' in ARCH
  assert 'quality_score REAL NOT NULL DEFAULT 0.5' in ARCH
  # Code truth: 0001_init.sql declares both columns NOT NULL.
  sql = (
    SRC / 'db/migrations/0001_init.sql'
  ).read_text(encoding='utf-8')
  assert 'india_presence INTEGER NOT NULL DEFAULT 1' in sql
  assert 'quality_score REAL NOT NULL DEFAULT 0.5' in sql


def test_config_no_dotenv_overrides_parenthetical() -> None:
  '''Config: no stale `.env overrides` parenthetical in architecture.'''
  assert '.env overrides' not in ARCH
  # Truthful remainder is kept: env read via os.getenv, not file-loaded.
  assert 'not loaded from a file' in ARCH
  # Code truth: config docstring says os.getenv only, no .env loading,
  # with no `.env overrides` mention.
  config_src = (SRC / 'core/config.py').read_text(encoding='utf-8')
  head = config_src[:2000]
  assert 'os.getenv' in head
  assert '.env overrides' not in config_src.split("'''")[1]
