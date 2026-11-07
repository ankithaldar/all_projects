#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cycle-13 pins: docs match cited code truth (HIGH/MEDIUM drifts).'''

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

CASCADE = [
  'openrouter_free',
  'nvidia_deepseek_ai_deepseek_v4_flash',
  'groq_qwen_qwen3_8_27b',
  'openrouter_nvidia_nemotron_3_ultra',
  'nvidia_moonshotai_kimi_k3',
  'groq_openai_gpt_oss_120b',
  'openrouter_thinking_machines_inkling',
  'nvidia_zai_glm_53_flash',
  'cerebras',
]

NODES = [
  'load_profile',
  'build_plan',
  'fetch_pair',
  'normalize_dedupe',
  'enrich_jds',
  'compute_embeddings',
  'score_rank_persist',
  'summarize_run',
]


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


def test_cascade_execution_order_exact() -> None:
  '''§0 cascade lists gateway.yaml execution_order + ollama fallback.'''
  s0 = _section(ARCH, '## 0.', '## 1.')
  for name in CASCADE:
    assert name in s0, name
  assert 'execution_order' in s0
  assert 'fallback ollama' in s0
  # Ordered exactly as gateway.yaml (positions increasing).
  positions = [s0.index(name) for name in CASCADE]
  assert positions == sorted(positions)


def test_topology_linear_eight_nodes() -> None:
  '''§10.2 is the actual linear 8-node graph with Send fan-out.'''
  s10 = _section(ARCH, '### 10.2', '## 11.')
  for node in NODES:
    assert node in s10, node
  chain = ' → '.join(NODES)
  assert chain in s10.replace('\n', ' ').replace('  ', ' ') or chain in s10
  assert 'RefreshGraph' not in s10
  assert 'fetch_node' not in s10
  assert 'classify_new_companies' not in s10
  assert 'hard_filter' not in s10
  assert 'match_and_score' not in s10
  assert 'persist_recommendations' not in s10


def test_gates_match_matcher_exact() -> None:
  '''§15 gates mirror matcher.gate_failures remote/location/exp sets.'''
  s15 = _section(ARCH, '## 15.', '## 16.')
  assert 'REMOTE, UNKNOWN' in s15 or 'remote {REMOTE' in s15
  assert 'HYBRID, REMOTE' in s15 or 'hybrid {HYBRID' in s15
  assert 'ONSITE' in s15
  assert '+1.5' in s15 or '±1.5' in s15 or 'years + 1.5' in s15
  assert 'salary_unverified' not in ARCH
  assert 'vertical' in s15.lower()
  # Verticals are not a hard gate in matcher.py.
  assert 'no vertical gate' in s15.lower()


def test_semantic_math_and_sort_and_template_rationale() -> None:
  '''§15 semantic uses (cosine+1)/2, title overlap, sort, template.'''
  s15 = _section(ARCH, '## 15.', '## 16.')
  assert '(cosine + 1) / 2' in s15
  assert 'token-overlap' in s15 or 'token overlap' in s15
  assert 'posted_at' in s15 and 'asc' in s15
  assert 'template-only' in s15 or 'template-built' in s15
  assert 'calibrated min-max' not in s15
  assert 'cosine(title' not in s15


def test_env_table_effective_vars_only() -> None:
  '''README env table: effective vars + JH_API_TOKEN + 9 keys split.'''
  for var in (
    'APP_HOST', 'APP_PORT', 'APP_LOG_LEVEL', 'APP_DATA_DIR',
    'SALARY_HARD_FLOOR_LPA', 'JH_API_TOKEN',
  ):
    assert var in README, var
  for var in (
    'BYTEZ_API_KEYS', 'CEREBRAS_API_KEYS', 'GITHUB_API_KEYS',
    'GROQ_API_KEYS', 'MISTRAL_API_KEYS', 'NVIDIA_NIM_API_KEYS',
    'OMNIROUTE_API_KEYS', 'OPENROUTER_API_KEYS', 'ORCAROUTER_API_KEYS',
  ):
    assert var in README, var
  assert 'standalone-only' in README
  assert 'gateway.yaml' in README


def test_jobs_detail_raw_and_cancel_immediate() -> None:
  '''Jobs detail exposes decoded `raw`; cancel finishes immediately.'''
  assert 'raw_json' not in API
  assert 'decoded `raw`' in API or '`raw` decoded' in API or 'decoded raw' in API
  s17 = _section(ARCH, '## 17.', '## 18.')
  assert 'no cooperative cancel flag' in s17
  assert 'cancelled via ui' in s17 or 'cancelled via UI' in s17
  assert 'immediate' in s17.lower()


def test_feedback_taxonomy_adapter_profile_setup() -> None:
  '''Feedback planned, flat taxonomy, fetch-only adapter, profile, setup.'''
  s15 = _section(ARCH, '## 15.', '## 16.')
  assert 'feedback_adjust' in s15
  assert 'never called' in s15 or 'planned' in s15.lower()
  assert 'gently retunes' not in OVERVIEW
  s13 = _section(ARCH, '## 13.', '## 14.')
  assert 'flat' in s13.lower()
  for vertical in (
    'ai_ml_infra', 'fintech', 'ecommerce', 'saas', 'healthtech',
    'edtech', 'logistics', 'gaming_media', 'retail_tech',
    'analytics_consulting',
  ):
    assert vertical in s13, vertical
  assert 'no `energy` vertical' in s13
  s11 = _section(ARCH, '## 11.', '## 12.')
  assert 'fetch-only' in s11 or 'only `fetch`' in s11
  assert 'discover_targets()' not in s11
  assert 'defaults dict' in ARCH or 'defaults when unset' in ARCH
  assert 'candidate 1' in ARCH
  assert 'fetch-jobs' in README
  assert 'scout' in README.lower()
  assert 'pytest' in README.lower()


def test_mediums_csrf_healthz_structured_fallback() -> None:
  '''CSRF/healthz, structured 3 attempts, no rule fallback, harvest.'''
  assert '/healthz' in API
  assert 'require_token' in API or 'behind the token' in API.lower()
  s9 = _section(ARCH, '## 9.', '## 10.')
  assert '3 attempts' in s9
  assert 'no few-shot' in s9.lower()
  assert 'no rule-based fallback' in ARCH
  assert 'fall back to rules' not in ARCH
  s12 = _section(ARCH, '## 12.', '## 13.')
  assert 'create_company' in s12 or 'bulk' in s12.lower()
  assert 'graph' in s12.lower() and 'no auto-create' in s12.lower()


def test_mediums_embedding_resume_transports_refresh() -> None:
  '''Embedding trunc, pypdf-only, transports, company-refresh filter.'''
  assert '1800' in ARCH
  assert 'no chunking' in ARCH.lower()
  s14 = _section(ARCH, '## 14.', '## 15.')
  assert 'pypdf-only' in s14.lower() or 'pypdf only' in s14.lower()
  assert 'python-docx' not in ARCH
  assert 'bare array' in API.lower()
  assert 'key query' in API.lower() or '?key=' in API
  assert 'status query' in API.lower() or '?status=' in API
  assert 'bare YAML' in API or 'bare yaml' in API.lower()
  assert '1MB' in API or '1_000_000' in API
  s19 = _section(ARCH, '## 19.', '## 20.')
  assert 'domain' in s19.lower()
  assert 'ats_provider' in s19
  assert 'last_checked_at > 7d' not in s19


def test_mediums_pages_overview_roadmap_ddl_diagram() -> None:
  '''§18 pages, overview rules-only, roadmap, §16 DDL, diagram.'''
  s18 = _section(ARCH, '## 18.', '## 19.')
  assert 'New today' in s18
  assert 'number inputs' in s18 or 'type="number"' in s18 or 'number input' in s18.lower()
  assert 'rules-only' in OVERVIEW.lower() or 'rules only' in OVERVIEW.lower()
  assert 'MCPClientManager' in ROADMAP
  assert 'removed' in ROADMAP.lower()
  assert 'historical' in ROADMAP.lower()
  s16 = _section(ARCH, '## 16.', '## 17.')
  assert 'parsed_json' not in s16
  assert 'seniority_keywords' in s16
  assert 'salary_floor_lpa' in s16
  assert 'created_at' in s16
  assert 'INSERT OR IGNORE INTO candidates' in s16
  assert 'single writer = worker' not in ARCH.replace('single-writer', '')


def test_mediums_runbook_422_and_settings_insert_only() -> None:
  '''Runbook 422 is resume-only; settings seeding is insert-only.'''
  assert 'resume upload only' in RUNBOOK.lower()
  assert 'do not return HTTP 422' in RUNBOOK.split('## Placeholder keys')[1].split('## ')[0]
  assert 'insert-only' in ARCH.lower() or 'insert only' in ARCH.lower()


def test_test_writer_cycle7_to_13_mappings() -> None:
  '''agents/test-writer maps cycle7-13 modules (or states exhaustiveness).'''
  for module in (
    'test_fix_cycle7_curator',
    'test_fix_cycle7_docs',
    'test_fix_cycle8_backend',
    'test_fix_cycle9_docs',
    'test_fix_cycle10_docs',
    'test_fix_cycle11_docs',
    'test_fix_cycle13_docs',
  ):
    assert module in WRITER, module
