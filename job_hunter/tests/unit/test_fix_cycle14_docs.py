#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle-14 pins: docs match cited code truth (Cycle-13 findings).'''

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs'
ARCH = (DOCS / 'architecture.md').read_text(encoding='utf-8')
API = (DOCS / 'api.md').read_text(encoding='utf-8')
OVERVIEW = (DOCS / 'app_overview.md').read_text(encoding='utf-8')
ROADMAP = (DOCS / 'roadmap.md').read_text(encoding='utf-8')
RUNBOOK = (DOCS / 'runbook.md').read_text(encoding='utf-8')
README = (ROOT / 'README.md').read_text(encoding='utf-8')
WRITER = (ROOT / 'agents' / 'test-writer.md').read_text(encoding='utf-8')


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


def test_roadmap_feedback_planned_bounded() -> None:
  '''Roadmap: feedback planned/not-done, bounded ±0.05 weight.'''
  s10 = ROADMAP[ROADMAP.index('feature_10'):ROADMAP.index('feature_11')]
  assert '±0.05' in s10
  assert 'planned' in s10.lower()
  assert 'shifts scores within' not in ROADMAP
  assert 'feedback_adjust' in ROADMAP


def test_settings_dotted_keys_stored_never_read() -> None:
  '''Settings: 2 of 4 PUT keys stored-but-never-read (discovery caps).'''
  for text in (ARCH, API, RUNBOOK):
    low = text.lower()
    assert 'stored-but-never-read' in low or 'never read' in low
    assert 'scoring_weights' in text
    assert 'salary_hard_floor_lpa' in text
  assert 'discovery.max_llm_calls_per_run' in API
  assert 'discovery.enrich_per_run_cap' in API
  assert 'api-only' in API.lower()


def test_env_table_jh_app_config_and_ollama() -> None:
  '''README env: JH_APP_CONFIG (MCP only), OLLAMA, split keys.'''
  assert 'JH_APP_CONFIG' in README
  assert 'MCP only' in README or 'MCP-only' in README
  assert 'OLLAMA_BASE_URL' in README
  assert 'gateway.yaml' in README
  assert 'JH_API_TOKEN' in README
  assert 'standalone-only' in README
  # 4 referenced vs 5 template-only split remains stated.
  assert 'template only' in README.lower()


def test_jobs_raw_cancel_adapter_profile_setup() -> None:
  '''Jobs raw decoded, cancel immediate, fetch-only, profile, setup.'''
  assert 'decoded `raw`' in API
  assert 'no payload key' in API.lower() or 'no raw' in API.lower()
  s17 = _section(ARCH, '## 17.', '## 18.')
  assert 'immediate' in s17.lower()
  assert 'no cooperative cancel flag' in s17
  s11 = _section(ARCH, '## 11.', '## 12.')
  assert 'fetch-only' in s11
  assert 'defaults dict' in ARCH
  assert 'candidate 1' in ARCH
  for text in (README, RUNBOOK):
    assert 'fetch-jobs' in text
    assert 'scout' in text.lower()
    assert 'pytest' in text.lower()


def test_csrf_api_only_allowed_host_healthz() -> None:
  '''CSRF API-only scope, allowed-host bypass, healthz under token.'''
  assert 'API-only' in API or 'API only' in API
  assert 'allowed-host bypass' in API.lower() or 'allowed host' in API.lower()
  assert 'require_token' in API
  assert '/healthz' in API
  s23 = _section(ARCH, '## 23.', '## 24.')
  assert 'API-only' in s23 or 'API only' in s23


def test_structured_signature_attempts_no_fallback() -> None:
  '''Structured: exact signature, 3 attempts, no few-shot/fallback.'''
  s9 = _section(ARCH, '## 9.', '## 10.')
  assert 'complete_structured(client, schema' in s9
  assert 'max_repairs=2' in s9
  assert '3 attempts' in s9
  assert 'no few-shot' in s9.lower()
  assert 'no rule-based fallback' in ARCH
  assert 'counts' in s9 and 'enrich_failures' in s9


def test_harvest_per_path_no_tiebreak_no_sitemap() -> None:
  '''Harvest: bulk-only create, graph/manual no-create, rules-only.'''
  s12 = _section(ARCH, '## 12.', '## 13.')
  assert 'create_company=True' in s12
  assert 'no auto-create' in s12.lower()
  assert 'no LLM tie-break' in s12
  assert 'sitemap is never probed' in s12.lower()
  assert 'rules-only' in s12.lower()
  assert 'LLM tie-break only on ambiguity' not in ARCH


def test_embedding_trunc_no_chunk_token_fallback() -> None:
  '''Embedding: 1800-char trunc, no chunking, token fallback, doc.'''
  s14 = _section(ARCH, '## 14.', '## 15.')
  assert '1800' in s14
  assert 'no chunking' in s14.lower()
  assert 'token-overlap' in s14
  assert 'summary + skills' in s14


def test_resume_pypdf_only_service_not_mcp() -> None:
  '''Resume: pypdf-only service parse, no DOCX, not MCP path.'''
  s14 = _section(ARCH, '## 14.', '## 15.')
  assert 'pypdf-only' in s14.lower()
  assert 'no DOCX' in s14
  assert 'ResumeParser service' in s14
  assert 'not the API' in s14 or 'not MCP' in s14


def test_api_transports_bare_and_query() -> None:
  '''Transports: bare array/YAML, key/status query params, 1MB.'''
  assert 'bare array' in API.lower()
  assert '?key=' in API
  assert 'bare JSON body' in API
  assert '?status=' in API
  assert 'bare YAML' in API
  assert '1_000_000' in API


def test_refresh_cron_pages_overview_split() -> None:
  '''Refresh weekly cron, §18 pages, overview sources split.'''
  s19 = _section(ARCH, '## 19.', '## 20.')
  assert '30 3 * * 0' in s19
  assert 'company_refresh' in s19
  s18 = _section(ARCH, '## 18.', '## 19.')
  assert 'New today' in s18
  assert 'per_page=8' in s18
  assert 'per_page=50' in s18
  assert 'number inputs' in s18
  assert 'discovery graph' in OVERVIEW.lower()
  assert 'fetch-jobs' in OVERVIEW
  assert 'rules-only' in OVERVIEW.lower()


def test_roadmap_writer_ddl_diagram_headers() -> None:
  '''Roadmap/writer/DDL/diagram/headers match code truth.'''
  assert 'MCPClientManager' in ROADMAP
  assert 'historical' in ROADMAP.lower()
  assert 'test_fix_cycle13_recs_page' in WRITER
  assert 'test_fix_cycle14_docs' in WRITER
  s16 = _section(ARCH, '## 16.', '## 17.')
  assert 'jobs_ai' in s16
  assert 'idx_job_embeddings_model_job' in s16
  assert 'read-only views' not in ARCH
  s5 = _section(ARCH, '## 5.', '## 6.')
  assert '# -*- coding: utf-8 -*-' in s5
  assert '# -- coding' not in s5
  assert 'no `.pylintrc`' in s5.lower() or 'no .pylintrc' in s5.lower()
  assert 'ruff' in s5.lower()
  s22 = _section(ARCH, '## 22.', '## 23.')
  assert 'not loaded from a file' in s22.lower()


def test_fanout_bulk_bounds_mcp_throttle() -> None:
  '''Fan-out rotation, bulk override, bounds, MCP returns, throttle.'''
  s10 = _section(ARCH, '### 10.2', '## 11.')
  assert 'list_due' in s10
  assert 'mark_checked' in s10
  assert 'deferred_targets' in s10
  assert 'cooled_skips' in s10
  for text in (README, ARCH, RUNBOOK):
    assert 'overrides the enabled flag' in text
  assert '_MAX_ID' in API
  assert 'RequestValidationError' in API
  assert 'le=10000' in API or 'le = 10000' in API or '10000' in API
  s8 = _section(ARCH, '## 8.', '## 9.')
  assert 'plausible_cv' in s8
  assert 'unvalidated' in s8.lower()
  assert 'robots/blocks → `cooldown_until`' not in ARCH
  s11 = _section(ARCH, '## 11.', '## 12.')
  assert 'get_text only' in s11.lower()
  assert 'fetch_pair' in s11
  assert 'insert-only' in RUNBOOK.lower()


def test_counts_dynamic_not_hardcoded() -> None:
  '''Test-module counts are dynamic (60+), not a stale literal.'''
  modules = list((ROOT / 'tests' / 'unit').glob('test_*.py'))
  assert len(modules) >= 60
  assert '60+' in WRITER or 'counted dynamically' in WRITER.lower()
