# Job Hunter — Personal Agentic Job Discovery System (India)

Status: Draft v1.0 for review
Owner: helder
Scope: local-only, single-user, discovery + recommendation first; automated applier is a future extension.

---

## 0. Purpose and inputs consumed

This document defines the full architecture for a personal, locally-run Agentic AI
system that hunts jobs in India for a configured role and industry vertical,
recommends ranked matches, and later (separate roadmap) applies automatically.

Inputs already fixed by the repository:

1. All code lives under `job_hunter/`. Nothing outside that folder is modified.
2. An existing `llm_gateway` package at `src/job_hunter/llm_gateway/` is the
   **only** path for every agentic LLM call. Its observed public surface:

   - `from job_hunter.llm_gateway import LLMGateway` (exports `SanitizedLLMGateway`)
   - `LLMGateway(config_path, env_path, db_path=None, cache_path=None, provider_overrides=None)`
   - `complete(request) -> GatewayResponse`, `acomplete(...)`,
     `stream(request) -> Iterator[ProviderChunk]`, `astream(...)`
   - `GatewayRequest(prompt | messages, system_prompt, tools, tool_choice,
     temperature, max_tokens, session_id, metadata)`
   - `GatewayResponse(content, tool_calls, usage, cached, latency_ms, cost,
     provider, model, alias, session_id)`
   - YAML config cascade: openrouter_ox_alpha → nvidia_glm → nvidia_llama →
     groq_qwen → groq_gpt_oss → openrouter → cerebras, fallback ollama(local).
   - Built-in per-provider RPM limiting, retry with backoff, SQLite response
     cache (`data/cache.db`) and call-cost log DB (`data/gateway.db`), plus an
     optional dashboard on port 8099. Dashboard assets (all under
     `src/job_hunter/llm_gateway/templates/`): `/` serves `dashboard.html`,
     `/dashboard.css` serves `dashboard.css`, `/dashboard.js` serves
     `dashboard.js`, and `/api/stats` serves JSON stats from the gateway
     DB (unknown paths → 404). The gateway config ships as
     `llm_gateway/config/gateway.yaml`; both are wheel package-data
     (see `pyproject.toml`).

3. Consequences of the gateway surface that shape this design:

   - There is **no native `response_format: json_schema`** support, so agents
     use JSON-mode prompting plus a strict parse → validate → repair loop.
   - Tool calling works only on providers with `supports_tools: true`; the
     Ollama fallback model has it disabled, so agents must degrade gracefully.
   - `session_id` gives free correlation between LangGraph runs and gateway
     cost logs — used everywhere as `{run_id}:{node_or_agent}`.
   - Gateway paths default to relative `config/gateway.yaml` / `.env`; the app
     must pass absolute paths resolved from the package location.

---

## 1. Goals and non-goals

Goals (v1):

- G1 Maximum-coverage automated job **discovery** across Indian employers with
  zero paid job-board APIs and no cloud deployment.
- G2 Personalized **ranking** using resume, skills, location, salary
  expectation, remote preference, experience level.
- G3 Full automation of the pipeline on a schedule; UI for review, not operation.
- G4 Transparent scoring with per-component breakdowns (no black box).
- G5 Every LLM call routed through `llm_gateway` with cost/latency visibility.

Non-goals (v1):

- Automated applying (designed now, built later — see roadmap).
- Multi-user/multi-tenant anything. Schema tolerates multiple candidates but UX assumes one.
- Mobile apps, authN/authZ beyond localhost binding.
- Cloud, containers, or distributed queues.

---

## 2. Assumptions and defaults chosen (confirm or override)

These are defaults I chose because they were not specified. None block the
design; each is cheap to change before implementation starts.

| # | Topic | Assumed default |
|---|-------|-----------------|
| A1 | Runtime | **Confirmed: Python 3.14+.** Dedicated venv inside `job_hunter/.venv`; `fastembed` chosen partly for its 3.14 wheel safety. |
| A2 | User | Single candidate; resume supplied as a **LaTeX-generated PDF** (pypdf parse path primary; DOCX left as an optional cheap fallback). |
| A3 | Salary | INR LPA parsing (₹ / lakhs / CTC phrasing). **Hard floor: 45 LPA — postings that verifiably top out below are excluded outright** (see gate in §15). |
| A4 | Scale | ~100–500 target companies, ≤ 5k active jobs, daily runs. SQLite is comfortable here. |
| A5 | Sources policy | Safe set: official ATS board APIs + aggregator RSS/APIs + career-page discovery, plus a watched **manual-export inbox** for LinkedIn/Naukri saves. No scraping of ToS-restricted portals. |
| A6 | Notifications | Dashboard + SSE live updates only for MVP; email/desktop deferred to a small feature branch. |
| A7 | Embeddings | `fastembed` (ONNX MiniLM-class, CPU, small wheels). Graceful degradation to SQLite FTS5 keyword scoring if unavailable. |
| A8 | Privacy | Resume/JD text goes to whichever providers are enabled in `gateway.yaml`. Switching `execution_order` to ollama-only makes the system fully offline. |
| A9 | Role seeds | **Locked: Staff-level Data Scientist roles, India metros + remote.** Industry vertical(s) still open (OQ-1); taxonomy ships with ai_ml_infra / fintech / saas / ecommerce starters. |

---

## 3. System overview

Four long-lived processes, one shared SQLite dataset (WAL mode):

```
                        ┌──────────────────────────────────────────────┐
                        │                job_hunter host               │
                        │                                              │
  Browser ──HTTP/SSE──▶ │  FastAPI app (127.0.0.1:8088)                │
  (vanilla JS)          │   ├─ REST API  (api/)                        │
                        │   ├─ static web/ (HTML+CSS+JS)               │
                        │   └─ read-only views over data/app.db        │
                        │                                              │
                        │  Worker (python main.py worker)              │
                        │   ├─ APScheduler (BlockingScheduler) jobs    │
                        │   ├─ claims runs from runs table             │
                        │   └─ executes LangGraph DiscoveryGraph       │
                        │        │                                     │
                        │        ├─▶ repositories (graph nodes use     │
                        │        │    repos directly, per D2)          │
                        │        └─▶ llm_gateway (all LLM calls)       │
                        │              ├─ provider cascade + fallback  │
                        │              ├─ cache.db   gateway.db        │
                        │              └─ dashboard :8099 (optional)   │
                        │                                              │
                        │  data/app.db  ← single writer = worker       │
                        └──────────────────────────────────────────────┘
```

Process responsibilities:

| Process | Entry point | Notes |
|---------|-------------|-------|
| API server | `python main.py api` | FastAPI + Uvicorn, binds host/port from AppSettings, serves `web/` statically. Reads DB; writes only profile/settings/manual actions. |
| Worker/scheduler | `python main.py worker` | Sole writer during runs; owns APScheduler (BlockingScheduler); invokes graphs. |
| MCP servers | standalone stdio subprocesses (3: sources, resume, store), run via `python main.py mcp <server>` / `config/app.yaml` `mcp:` commands | LLM/discovery tool surface per D2 — the graph nodes themselves use repositories directly, not MCP; no in-process MCP client spawns them. |
| Gateway dashboard | optional `llm_gateway` dashboard | Cost inspection UI on 8099, already provided. |

SQLite concurrency contract: WAL + `busy_timeout=5000`; worker uses
`BEGIN IMMEDIATE` to claim pending runs; API writes are short transactions;
FTS5 external-content table kept in sync by triggers.

---

## 4. Repository and module layout

Everything below lives under `job_hunter/`. The source root is
`src/job_hunter/`, which contains **two top-level packages**: the existing
`llm_gateway` (untouched imports like `from llm_gateway.app import ...`
keep working) and the new application package `job_hunter`.

```
job_hunter/
├── docs/
│   ├── architecture.md            ← this file
│   ├── roadmap.md                 ← phased feature-branch plan
│   ├── app_overview.md            ← short overview
│   ├── runbook.md                 ← operations runbook
│   └── api.md                     ← endpoint reference
├── agents/
│   ├── bug-hunter.md  doc-keeper.md  pipeline-runner.md  test-writer.md
├── config/
│   └── app.yaml                   ← committed app defaults (§22)
├── pyproject.toml                 ← packaging (canonical deps), ruff, pytest config
├── .env.example                   ← APP_* only; gateway template is canonical for keys
├── main.py                        ← thin launcher (seed-db|api|worker|run-discovery|
│                                    discover-companies|verify-ats|scout-companies|
│                                    fetch-jobs|mcp)
├── setup.sh / run.sh              ← one-time setup / api+worker runner
├── src/job_hunter/
│   ├── llm_gateway/               ← EXISTING gateway (config/, providers/, tests/)
│   └── job_hunter/
│       ├── __init__.py
│       ├── core/
│       │   ├── config.py          ← AppSettings (+paths)
│       │   ├── logging.py         ← JSON logs, run contextvars
│       │   ├── db.py              ← connection factory, WAL, migrations runner
│       │   ├── bootstrap.py       ← first-time init (migrations, taxonomy, defaults)
│       │   ├── models.py          ← domain Pydantic models
│       │   └── errors.py          ← exception hierarchy
│       ├── db/
│       │   ├── migrations/        ← numbered SQL scripts
│       │   └── repositories/      ← Repository pattern per aggregate
│       │       ├── companies.py  crawl_state.py  jobs.py  profile.py
│       │       ├── recommendations.py  runs.py  settings.py
│       │       ├── taxonomy.py    ← db/repositories/taxonomy.py
│       ├── llm/
│       │   ├── client.py          ← GatewayClient facade (singleton)
│       │   ├── structured.py      ← complete_structured(): JSON+repair loop
│       │   │                        (langchain_bridge.py / GatewayChatModel
│       │   │                        and mcp_tools.py / MCPClientManager:
│       │   │                        removed — see §7–§9)
│       ├── mcp_servers/
│       │   ├── common.py
│       │   ├── sources_server.py  ← fetch_ats_board, fetch_page, search_web,
│       │   │                        detect_careers (see §8 for exact signatures)
│       │   ├── resume_server.py   ← extract_text, clean_text
│       │   └── store_server.py    ← get_profile, list_companies,
│       │                            upsert_company, jobs_recent,
│       │                            recommendations_top, record_run_event
│       ├── adapters/              ← SourceAdapter implementations (§11)
│       │   ├── base.py  greenhouse.py  lever.py  ashby.py  workable.py
│       │   ├── workday.py  smartrecruiters.py  recruitee.py  personio.py
│       │   ├── aggregators.py     ← remotive/remoteok/weworkremotely/himalayas/
│       │   │                        arbeitnow/jobicy (shared adapter)
│       │   ├── cutshort.py  career_page.py  manual_inbox.py
│       │   ├── http_client.py  registry.py
│       ├── services/
│       │   ├── company_discovery.py  vertical_classifier.py  domain_verifier.py
│       │   ├── normalizer.py  dedupe.py  skills_taxonomy.py  web_search.py
│       │   ├── embedder.py  matcher.py  scorer.py  jd_analyst.py
│       │   ├── profile_curator.py  resume_parser.py  bulk_fetch.py
│       │   ├── linkedin_scout.py  run_manager.py  watchdog.py
│       │   ├── persist.py         ← persist_record(): shared persist sequence
│       ├── graph/
│       │   ├── state.py           ← TypedDict states + reducers
│       │   ├── nodes.py           ← node functions
│       │   ├── discovery_graph.py ← build_discovery_graph()
│       │   └── checkpointing.py   ← SqliteSaver wiring
│       ├── workers/
│       │   ├── scheduler.py       ← APScheduler (BlockingScheduler) jobs
│       │   └── jobs.py            ← claim-run + execute + finalize
│       ├── worker/__main__.py
│       └── api/
│           ├── __main__.py        ← api/__main__.py (`python -m job_hunter.api`)
│           ├── main.py            ← FastAPI app factory, static mount
│           ├── deps.py            ← DI (settings, db, gateway)
│           └── routes/
│               ├── profiles.py  companies.py  jobs.py
│               ├── recommendations.py  runs.py  settings.py
├── web/
│   ├── index.html  profile.html  companies.html  jobs.html
│   ├── recommendations.html  runs.html  settings.html
│   ├── css/style.css
│   └── js/api.js  js/components.js
├── seeds/
│   ├── companies_*.yaml           ← seed companies per vertical
│   ├── verticals.yaml             ← taxonomy (§13)
│   └── skills_aliases.yaml        ← starter skill taxonomy
└── tests/
    └── unit/…                     ← 56 unit test modules (pre-existing;
                                     live list is tests/unit/; pytest;
                                     gateway tests collected via testpaths)
```

Packaging notes:

```toml
# pyproject.toml (excerpt)
[tool.setuptools.packages.find]
where = ['src/job_hunter']
include = ['job_hunter*', 'llm_gateway*']

[tool.pytest.ini_options]
pythonpath = ['src/job_hunter']
```

Rationale: the gateway's modules import each other as top-level `llm_gateway.*`;
making `src/job_hunter` the package root preserves those imports verbatim while
exposing the application as top-level `job_hunter`.

---

## 5. Coding standards (enforced)

Every Python file begins exactly with:

```python
#!/usr/bin/env python
# -- coding: utf-8 --
```

Rules (matching both your requirements and the existing gateway code):

1. Google-style docstrings on every module, class, function, method
   (Args / Returns / Raises / Yields where applicable).
2. 2-space indentation; no tabs.
3. Single quotes for all string literals.
4. SOLID: adapters/strategies behind ABCs (`SourceAdapter`, `Embedder`,
   `ScoringComponent`); repositories behind protocols; composition via
   constructors; no import-time side effects.
5. Patterns used deliberately: Facade (`GatewayClient`), Adapter
   (`SourceAdapter`), Repository, Strategy (scorers),
   Factory (app/graph builders), Observer-lite (SSE broadcaster).
6. Type hints everywhere; `from __future__ import annotations`.
7. Lint: repo-root `.pylintrc` applies; add ruff for speed; CI-less local
   pre-commit hook optional.

Skeleton example (the shape all new files follow):

```python
#!/usr/bin/env python
# -- coding: utf-8 --

'''Short module purpose line.'''


from __future__ import annotations


class Example:
  '''One-line summary.

  Args:
    name: Description.
  '''

  def __init__(self, name: str) -> None:
    '''Initialize the example.

    Args:
      name: Display name.
    '''
    self._name = name
```

---

## 6. Git workflow

Base branch in this repo is `agentic_ai/job-hunter` (hyphen — actual name).
Feature branches follow your naming convention under it.

```bash
git checkout agentic_ai/job-hunter && git pull
git checkout -b agentic_ai/job-hunter/feature_03_source_adapters

# ...commit freely on the feature branch...

git checkout agentic_ai/job-hunter
git merge --squash agentic_ai/job-hunter/feature_03_source_adapters
git commit -m 'feat(adapters): tier-1 ATS source adapters with contract tests'
git branch -D agentic_ai/job-hunter/feature_03_source_adapters
```

Conventions: conventional-commit messages on squashes
(`feat|fix|docs|chore(scope): summary`); one feature = one branch =
one squash commit; tags `v0.x.y` at roadmap milestones.

---

## 7. Agent roles

Agents are LangGraph subgraphs/nodes composed of prompts + gateway calls +
repository access. Per D2, the graph nodes use repositories directly — the
MCP servers are a standalone stdio tool surface for discovery/LLM use
(`python main.py mcp <sources|resume|store>`; commands in
`config/app.yaml`), not the graph's data path. There is no in-process LLM
tool loop: `MCPClientManager` (`llm/mcp_tools.py`) was deleted, so MCP
tools are not loaded into the graph or an LLM session at runtime; graph
nodes call repositories and services (`persist_record`, §10) directly.
Each agent has a single responsibility (SRP) and a typed state slice.

| Agent | Responsibility | LLM usage | Data access |
|-------|----------------|-----------|-------------|
| Profile Curator | Parse resume → canonical profile; keep skills taxonomy fresh | Structured extraction | repositories/services (resume-mcp `extract_text`/`clean_text` available as standalone stdio tools) |
| Company Scout | Expand seed list into verified targets: resolve domain, detect ATS provider/board token, classify vertical | Light (classification, disambiguation) | repositories/services (sources-mcp `search_web`/`fetch_page`/`detect_careers` available as standalone stdio tools) |
| Job Fetcher | Pull postings from assigned (source, target) pairs; paginate; respect limits | None (deterministic) | adapters + repositories (browser-mcp later) |
| Normalizer | Canonicalize fields, map cities/regions, dedupe | Rare fallback for messy locations | repositories/services |
| JD Analyst | Extract must/nice skills, experience band, salary, work mode, employment type from raw description | Heavy structured extraction | none |
| Matcher | Compute component scores vs candidate (skills coverage, semantic sim, seniority…) | None (embeddings + rules) | repositories/services |
| Ranker | Weighted aggregation, gates, rank, human-readable rationale | One-line rationale (cheap alias) | repositories/services |
| Watchdog | Post-run QA: stale detection, adapter health, quarantine counts, alert flags | Optional summary | repositories/services |
| Applier (future) | Tailor documents, fill forms, submit after approval gate | Heavy generation | browser-mcp |

Design rule: **no agent talks to HTTP directly** — everything flows through
repositories/services (or the standalone MCP stdio tools above), keeping
tools testable and swappable (DIP).

> **D2 overload note:** "D2" is overloaded in this document. In §3/§7/§8,
> "per D2" means "graph nodes use repositories directly; MCP is a separate
> tool surface". In the §25 decision list, D2 is the resume-input decision
> (LaTeX-generated PDF via pypdf).

---

## 8. MCP servers and tools

Three local MCP servers (official `mcp` Python SDK, stdio transport), one
process per concern. Per D2 they form a standalone LLM/discovery tool
surface — run via `python main.py mcp <sources|resume|store>` or the
`mcp:` commands in `config/app.yaml` — while graph nodes use repositories
directly. There is no in-process tool loop and no MCP client manager
(`MCPClientManager` deleted); the tables below are the exact
`@mcp.tool` definitions in `mcp_servers/*.py`. MCP 2.x fallback: all
three servers (`mcp_servers/sources_server.py`,
`mcp_servers/resume_server.py`, `mcp_servers/store_server.py`) try
`mcp.server.fastmcp.FastMCP` first and fall back to
`mcp.server.mcpserver.MCPServer` (aliased as `FastMCP`) on
`ModuleNotFoundError`, so all three servers import on both mcp 1.x
and mcp 2.x.

### sources-mcp (`mcp_servers/sources_server.py`)

| Tool (exact signature) | Output |
|------|--------|
| `fetch_ats_board(provider: str, board_ref: str, limit: int = 100) -> str` | normalized raw posting list as JSON |
| `fetch_page(url: str) -> str` (`async def`) | page body as cleaned text (robots-aware) |
| `search_web(query: str, max_results: int = 8) -> str` | results[] (web-search index) |
| `detect_careers(domain: str) -> str` | ATS fingerprint for a domain's careers board |

### resume-mcp (`mcp_servers/resume_server.py`)

| Tool (exact signature) | Output |
|------|--------|
| `extract_text(path: str) -> str` | extracted text + metadata from a PDF resume |
| `clean_text(text: str) -> str` | normalized text (ligatures, unicode, whitespace) |

### store-mcp (`mcp_servers/store_server.py`)

Allow-listed operations only (never raw SQL passthrough):

| Tool (exact signature) | Output |
|------|--------|
| `get_profile(candidate_id: int = 1) -> str` | candidate profile as JSON |
| `list_companies(status: str = 'active', limit: int = 100) -> str` | company rows |
| `upsert_company(name: str, domain: str = '', vertical: str = '') -> str` | created/updated company id |
| `jobs_recent(limit: int = 25, city: str = '', work_mode: str = '') -> str` | recent active jobs |
| `recommendations_top(limit: int = 10) -> str` | top gate-passing recommendations |
| `record_run_event(run_id: int, level: str, node: str, message: str) -> str` | event id (level: debug\|info\|warn\|error) |

Effective limit clamps (callers pass a raw limit; the server clamps it):
`fetch_ats_board` uses `max(1, min(limit, 300))`, `search_web` uses
`max(1, min(max_results, 15))`, `list_companies` uses `min(limit, 500)`,
`jobs_recent` uses `min(limit, 200)`, and `recommendations_top` uses
`min(limit, 100)`.

### Planned (not built)

A browser-mcp (Playwright) is planned for the applier track only; it is not
part of the current 3-server surface and is not needed for discovery.

---

## 9. LLM integration (via llm_gateway)

`llm/client.py` provides the only construction site for the gateway
(Singleton scoped per process):

```python
gateway = LLMGateway(
  config_path=APP_ROOT / 'src' / 'job_hunter' / 'llm_gateway'
              / 'config' / 'gateway.yaml',
  env_path=GATEWAY_ENV_PATH,
  db_path=DATA_DIR / 'gateway.db',
  cache_path=DATA_DIR / 'cache.db',
)
```

Two call patterns, all through the gateway:

1. **Plain completion** — rationales, summaries:
   `gateway.acomplete(GatewayRequest(prompt=..., session_id=f'{run_id}:ranker'))`.
2. **Structured extraction** — `structured.complete_structured(schema, payload)`
   helper: builds a JSON-only instruction + few-shot, calls `acomplete`,
   strips code fences/think tokens (sanitizer already helps), parses with
   Pydantic; on failure replays once with the validator error appended
   (max 2 repairs), then raises `StructuredOutputError` so the node can fall
   back to rules. Every attempt shares one `session_id` for cost attribution.

Removed: the LangChain bridge (`GatewayChatModel` in
`llm/langchain_bridge.py`) and the MCP client tool loader
(`MCPClientManager` in `llm/mcp_tools.py`) are deleted. LLM calls go
directly through `GatewayClient` (`llm/client.py`) and the structured
helper above; there is no LangChain chat-model adapter and no in-process
tool loop.

Degradation ladder per task class:

- Extraction quality critical (JD Analyst): try cascade; on
  `AllProvidersFailedError` mark job `needs_review` rather than guessing.
- Nice-to-have (rationale lines): skip silently if LLM unavailable.
- Ollama fallback lacks tools: it is a last resort for completion calls
  and agents switch to deterministic plans when reached.

Cost governance: gateway RPM limits already cap burn; the app additionally
caches extractions keyed by job `content_hash` (a re-crawl of an unchanged JD
costs zero tokens) and enforces a per-run token budget from settings.

---

## 10. LangGraph workflows and state design

### 10.1 State

```python
class DiscoveryState(TypedDict):
  '''State for the nightly discovery graph.'''
  run_id: int
  triggered_by: str
  candidate: CandidateProfile
  plan: Dict[str, Any]                     # build_plan returns plan.model_dump();
                                           # callers seed {'quick_poll': ...}
  raw_jobs: Annotated[List[RawJobRecord], operator.add]
  inserted: List[NormalizedJob]
  enriched: Dict[str, Any]                 # keyed by content_hash; plain dicts
  scored: List[ScoredJob]
  errors: Annotated[List[NodeError], operator.add]
  stats: Annotated[Dict[str, int], merge_stats]  # counters merged by reducer
  budget: TokenBudget
```

Reducers: `raw_jobs`/`errors` append (fan-out safe); `stats` merges dicts;
everything else last-write-wins. State normalization: `DiscoveryState`
is a `typing_extensions.TypedDict` (`total=False`); after checkpoint
serialization the candidate may come back as a plain dict, so nodes
normalize inserted items via `graph/state.py: _normalize_inserted`
(dict → `NormalizedJob`) and the candidate via
`graph/nodes.py: _normalize_candidate` (dict →
`CandidateProfile(**value)`, instance passes through) before use.
Checkpointing via LangGraph's `SqliteSaver` into
`data/checkpoints.sqlite` → any run resumes after crash/restart.
Dependencies: `langgraph-checkpoint-sqlite>=3.1` + `aiosqlite>=0.20`
(declared in `pyproject.toml`); resume is conditional on install —
`graph/checkpointing.py: open_checkpointer` yields `None` (and logs a
warning) when those imports are missing, and the graph runs without a
checkpointer instead of crashing. The checkpointer connection sets
`PRAGMA journal_mode=WAL`, `PRAGMA busy_timeout=5000` and
`PRAGMA foreign_keys=ON` before `AsyncSqliteSaver` setup.
`graph/checkpointing.py:
thread_config(run_id)` builds the runnable config with
`'checkpoint_ns': ''` (empty namespace) and `thread_id=f'run-{run_id}'`.

Persistence: `services/persist.py: persist_record(...)` is the shared
persist sequence (content-hash/canonical-URL, city/work-mode detection,
company resolution, insert → `inserted`/`duplicate`) used by the graph
nodes, `bulk_fetch`, and the manual inbox — graph nodes do not persist
via MCP tools.

### 10.2 Graph topology

```
load_profile → build_plan ──Send(fetch_pair)──▶ fetch_node (fan-out N)
                    ▲                               │
                    │                     collect_raw_jobs
                    │                               │
              (loop weekly)              normalize_and_dedupe
                                                    │
                                        classify_new_companies (cond.)
                                                    │
                                                hard_filter (gates)
                                                    │
                                    ┌─ cond: content_hash unseen? ─┐
                                      yes: enrich_jds (batched LLM)
                                      no:  skip
                                                    │
                                              compute_embeddings
                                                    │
                                            match_and_score
                                                    │
                                       persist_recommendations
                                                    │
                                          summarize_run (Watchdog)
```

- `build_plan` emits one `Send('fetch_node', {source, target})` per due pair →
  natural parallelism with per-source concurrency caps in the node itself.
- Every node is wrapped by `safe_node(fn)`: catches exceptions, appends
  `NodeError(node, message, recoverable)` to state, writes a `run_events` row,
  returns partial updates → **partial-success semantics** (one broken board
  never kills the run).
- Conditional edges: enrichment only for unseen hashes; Watchdog escalates to
  `status='failed'` only if ≥ 1 hard-required node errored.
- `RefreshGraph` = subset subgraph (single company/source) reused by the API's
  "refresh now" button; same nodes, different `plan` dict.

---

## 11. Job discovery strategy (tiered, India-aware)

Tier model, cheapest/most-stable first. All tiers feed the same
`RawJobRecord` shape; adapters implement `SourceAdapter` (ABC):
`discover_targets()`, `fetch(pair, since)`, `parse(payload) -> [RawJobRecord]`.

| Tier | Source | Mechanism | Coverage notes for India |
|------|--------|-----------|--------------------------|
| T1 official ATS APIs | Greenhouse, Lever, Ashby, Workable, Workday, SmartRecruiters, Recruitee, Personio(XML) | Public JSON endpoints, no auth | Strong for product companies, GCCs, startups: e.g. `boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true`, `api.lever.co/v0/postings/{co}?mode=json`, `api.ashbyhq.com/posting-api/job-board/{org}`, `apply.workable.com/api/v1/widget/accounts/{ac}?details=true`, `api.smartrecruiters.com/v1/companies/{co}/postings` |
| T2 remote/global aggregators | Remotive `/api/remote-jobs`, RemoteOK `/api`, Himalayas, WeWorkRemotely RSS, Arbeitnow, Jobicy, Cutshort (India-first, server-rendered category pages) | Public API/RSS/HTML | Filters to India-eligible + remote-timezone-overlap |
| T3 career pages | `{domain}/careers|/jobs` crawl → ATS fingerprint (URL/CNAME patterns) → route to matching T1 adapter | Discovery-driven | Catches companies missing from seed lists |
| T4 manual export inbox | Watched dirs `data/inbox/{linkedin,naukri,instahyre}/` | Parse saved search-result HTML/CSV you export by hand; after a successful parse the file is moved to `data/inbox/_processed/<mtime>_<name>` (skipped on scan, so it is not re-ingested) | Compliant way to include ToS-restricted portals; records flagged `source='manual'` |

Cross-cutting behavior: per-(host, rpm) throttle, default 30 rpm
(`adapters/http_client.py: throttle_for` keys throttles by `(host, rpm)`
per D15 — the same host at different rpm values runs independent
throttles, so the aggregate rate can exceed the minimum rpm; adapters
pass explicit `rpm=` per source). The throttle host is normalized
lowercase with trailing dots stripped
(`parsed.hostname.lower().rstrip('.')`, with the same lowercasing and
`.rstrip('.')` for a bare `host[:port]` URL, so trailing-dot hosts
share the same throttle), and `HostThrottle.wait()` reserves its slot
before sleeping (`_last = now + delta`), so concurrent callers space
out by one interval each instead of waking together. Polite UA
is `job-hunter-personal/0.1 (local personal job search; contact: self)`,
robots.txt respected for HTML,
ETag/Last-Modified reuse via `crawl_state`, pagination until `since` watermark,
per-adapter circuit breaker (5 consecutive failures → cooldown 24h, surfaced in
Watchdog report).

Freshness: postings not re-seen for 45 days → `status='stale'` (hidden from
recommendations, retained for history).

---

## 12. Company discovery strategy

Pipeline (Company Scout agent + `services/company_discovery.py`):

1. **Seeds** — user-curated YAML per vertical/city
   (`name, domain?, vertical_hint, priority`). Ships with fintech/Bengaluru examples.
   Seed priorities are clamped 1..5 (numeric truncated via int() then
   clamped 1..5; missing/None->3; 0->1 (clamped) on import/seed); a seed entry with a
   bad priority (non-numeric/non-finite) is skipped with a warning and
   not counted; bool priority is rejected, not clamped (422 create/patch; errors import; skipped seed).
2. **Harvest** — every T1/T2 posting carries `company_raw_name` + board token;
   unknown names become candidate companies automatically (this alone covers
   most of the ATS ecosystem over time).
3. **Expansion** — `sources-mcp.search_web` queries like
   `'{vertical} product companies in {city} careers'`; results resolved to
   registrable domains (tldextract-style suffix rules, offline list).
4. **Verification/enrichment** — probe `{domain}/careers`, `/jobs`, sitemap;
   fingerprint ATS provider; capture `board_token/org/account`; store
   `ats_provider` + `careers_url`. Failures park company in
   `status='needs_review'` instead of dropping.
5. **Aliases** — `company_aliases` table maps spelling variants harvested from
   boards ('Flipkart Internet Pvt Ltd' → flipkart.com).

Identity rule: `domain` is the canonical key; name matching is fuzzy
(RapidFuzz token_sort_ratio ≥ 92) with LLM tie-break only on ambiguity.

---

## 13. Industry vertical filtering

Taxonomy in `seeds/verticals.yaml`: two levels (vertical → sub_vertical),
keyword lists per node, and inclusion/exclusion terms. Examples:
`fintech` (payments, lending, neobank, wealth…), `saas`, `ecommerce`,
`ai_ml_infra`, `gaming`, `healthtech`, `edtech`, `logistics`.

Classification order (cheap→expensive, confidence recorded):

1. Domain keyword hit (careers page/about meta) → conf 0.9
2. Rule keywords in description/company blurb → conf 0.7
3. LLM classifier via gateway (enum-constrained output + confidence) → conf 0.6
4. Unknown → `vertical='unknown'`; included only if user setting
   `include_unknown_vertical=false` is off... default **excluded**, visible in
   Companies UI for one-click labeling (labeling feeds rule learning).

The candidate profile declares `target_verticals[]` + `blocked_verticals[]`;
hard filter drops non-targets unless priority ≥ 4 (hand-pinned company).

---

## 14. Candidate profile matching

Inputs captured (profile UI + resume upload): resume file, skills
(auto + manual), location preferences (multi-city + willingness to relocate),
salary expectation (min/target LPA), remote preference
(remote|hybrid|onsite|any), experience level (years + band), target roles,
target/blocked verticals, employment types.

Mechanics:

- **Parsing**: pypdf/python-docx → text → sections (resume-mcp) →
  Curator agent extracts structured profile (Pydantic-validated) → user edits
  in UI (human always confirms the parse).
- **Skills taxonomy**: `skills` + aliases seeded from
  `seeds/skills_aliases.yaml`; Curator proposes new aliases; matcher resolves
  JD skills ↔ canonical ids (alias map first, embedding-nearest within
  threshold second). None spec seeded with empty category/aliases; non-None non-dict spec skipped/not counted, category preserved (non-None non-dict specs log `skipping non-dict skill spec for <name>: <spec>`; dict specs preserve their category/aliases). resolve_many(non-str elements skipped with warning; bare str/bytes/non-list raises TypeError -> 422 on PUT /api/profile/skills). Further guards: a non-dict payload returns 0, a non-str name is skipped/not counted, non-list aliases are empty+warn, and non-str aliases are skipped.
- **Semantic similarity**: `fastembed` MiniLM-class vectors; candidate doc =
  summary + skills + recent titles; job doc = title + cleaned description
  (chunked at ~1k chars, mean-pooled). Cosine → scaled to [0,1]. Vectors stored
  in `job_embeddings` (BLOB) keyed by model id; recomputed only when hash changes.
- **Resume duplicates**: duplicate sha256 returns the existing resume_id
  and stored path (the new upload file is unlinked; the stored row is kept).
  A dangling duplicate (sha matches but the stored row is missing) raises
  `RuntimeError: resume row missing after save_resume_ex`. On failure it
  deletes owned resume row first; file kept if referenced (row delete is
  FK-blocked when a profile references it). FK-blocked/retained failure sets exc.file_path falsy so API does not unlink stored file.

---

## 15. Recommendation ranking and scoring logic

Two stages: **hard gates** then **weighted score**. Everything persisted for
explainability (`score_breakdown_json` shown as a bar breakdown in the UI).

Gates (fail ⇒ excluded, reason stored):

| Gate | Logic |
|------|-------|
| work_mode_compat | job mode ∈ allowed set for pref ('any' passes all; remote-pref users pass remote/hybrid-remote-flagged) |
| location_compat | job city ∈ prefs OR relocation_ok AND priority ≥ 3 OR remote |
| experience_band | overlap between [exp_min, exp_max] and candidate ±1y window |
| employment_type | ∈ enabled types (full_time default) |
| salary_floor (hard) | settings.salary_hard_floor_lpa = **45**; exclude when posted_max_lpa < 45. Unknown ranges pass but are flagged `salary_unverified` and score neutral on the salary component |

Weighted components (defaults; editable live in Settings, persisted per user):

| Component | Default w | Score sᵢ ∈ [0,1] |
|-----------|-----------|------------------|
| must-have skill coverage | 0.30 | matched_must / total_must (unknown ⇒ 0.5 neutral) |
| nice-to-have coverage | 0.10 | matched_nice / total_nice |
| semantic similarity | 0.25 | cosine, calibrated min-max over run cohort |
| seniority fit | 0.10 | triangular peak at candidate years |
| title fit | 0.05 | cosine(title, candidate target-role docs) |
| salary fit | 0.10 | clamp(mid_posted / 45 LPA, 0, 1); `salary_unverified` ⇒ 0.5 neutral |
| recency | 0.05 | exp(−ln2 · age_days / 10) half-life 10d |
| company/vertical fit | 0.05 | exact vertical 1.0, sub-vertical parent 0.7, pinned +0.2 cap 1 |

`total_score = round(100 · Σ wᵢ·sᵢ, 1)` for gate-passers; ties broken by
recency then company priority. Jobs with a non-int `job_id` (incl. bool)
are skipped with `skipping job with non-int job_id` before any `int()`
conversion, so one phantom job cannot crash the scoring pass. Recommendations table stores rank per run +
rationale (template-built; optional one-line LLR summary via cheapest alias).

Feedback loop: dismiss/save actions feed simple learned adjustments
(dismiss-on-high-score demotes similar titles/skills slightly — logistic
reweighting of `title_fit`/`semantic` weights bounded ±20%).

---

## 16. SQLite schema (v1 DDL summary)

File: `data/app.db`, WAL, foreign keys ON, migrated by numbered SQL scripts.

```sql
CREATE TABLE candidates (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  email TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE resumes (
  id INTEGER PRIMARY KEY,
  candidate_id INTEGER NOT NULL REFERENCES candidates(id),
  file_path TEXT NOT NULL,
  sha256 TEXT NOT NULL UNIQUE,
  mime TEXT,
  uploaded_at TEXT NOT NULL DEFAULT (datetime('now')),
  parsed_ok INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE candidate_profiles (
  id INTEGER PRIMARY KEY,
  candidate_id INTEGER NOT NULL REFERENCES candidates(id),
  version INTEGER NOT NULL DEFAULT 1,
  target_roles TEXT NOT NULL DEFAULT '[]',      -- json array
  target_verticals TEXT NOT NULL DEFAULT '[]',  -- json array
  blocked_verticals TEXT NOT NULL DEFAULT '[]', -- json array
  cities TEXT NOT NULL DEFAULT '[]',            -- json array
  relocate_ok INTEGER NOT NULL DEFAULT 0,
  remote_pref TEXT NOT NULL DEFAULT 'any'
    CHECK (remote_pref IN ('remote','hybrid','onsite','any')),
  salary_min_lpa REAL, salary_target_lpa REAL,
  experience_years REAL, experience_band TEXT,
  employment_types TEXT NOT NULL DEFAULT '["full_time"]',
  summary TEXT,
  parsed_json TEXT,
  confidence REAL,
  resume_id INTEGER REFERENCES resumes(id),
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE skills (
  id INTEGER PRIMARY KEY,
  canonical_name TEXT NOT NULL UNIQUE,
  category TEXT
);
CREATE TABLE skill_aliases (
  alias TEXT PRIMARY KEY,
  skill_id INTEGER NOT NULL REFERENCES skills(id)
);
CREATE TABLE candidate_skills (
  candidate_id INTEGER NOT NULL REFERENCES candidates(id),
  skill_id INTEGER NOT NULL REFERENCES skills(id),
  weight REAL NOT NULL DEFAULT 1.0,
  PRIMARY KEY (candidate_id, skill_id)
);

CREATE TABLE companies (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  normalized_name TEXT NOT NULL UNIQUE,
  domain TEXT UNIQUE,
  careers_url TEXT,
  ats_provider TEXT, board_ref TEXT,        -- token/org/account per provider
  vertical TEXT, sub_vertical TEXT, vertical_confidence REAL,
  hq_city TEXT, india_presence INTEGER DEFAULT 1,
  size_band TEXT, funding_stage TEXT,
  priority INTEGER NOT NULL DEFAULT 3,       -- 1..5, 5 = pinned
  status TEXT NOT NULL DEFAULT 'active'
    CHECK (status IN ('active','paused','needs_review','blacklisted','merged')),
  discovered_via TEXT, notes TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  last_checked_at TEXT
);
CREATE TABLE company_aliases (
  alias TEXT PRIMARY KEY,
  company_id INTEGER NOT NULL REFERENCES companies(id)
);

CREATE TABLE sources (
  key TEXT PRIMARY KEY,                      -- e.g. greenhouse, remotive, manual
  kind TEXT NOT NULL,                        -- ats|aggregator|career|manual
  enabled INTEGER NOT NULL DEFAULT 1,
  rate_limit_rpm INTEGER NOT NULL DEFAULT 30,
  config_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE jobs (
  id INTEGER PRIMARY KEY,
  source_key TEXT NOT NULL REFERENCES sources(key),
  external_id TEXT,
  url TEXT NOT NULL,
  canonical_url TEXT NOT NULL,
  company_id INTEGER REFERENCES companies(id),
  company_raw_name TEXT,
  title TEXT NOT NULL,
  location_text TEXT, city TEXT, region TEXT, country TEXT DEFAULT 'IN',
  work_mode TEXT CHECK (work_mode IN ('remote','hybrid','onsite','unknown')),
  employment_type TEXT,
  salary_min_lpa REAL, salary_max_lpa REAL, salary_raw TEXT,
  experience_min_yrs REAL, experience_max_yrs REAL,
  posted_at TEXT, first_seen_at TEXT NOT NULL DEFAULT (datetime('now')),
  description_text TEXT,
  raw_json TEXT NOT NULL,
  content_hash TEXT NOT NULL UNIQUE,         -- sha256(canonical url+company+title+desc-trimmed)
  status TEXT NOT NULL DEFAULT 'active'
    CHECK (status IN ('active','stale','closed','error')),
  quality_score REAL DEFAULT 0.5,
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_jobs_company_posted ON jobs(company_id, posted_at DESC);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE VIRTUAL TABLE jobs_fts USING fts5(
  title, description_text, content='jobs', content_rowid='id'
);

CREATE TABLE job_skills (
  job_id INTEGER NOT NULL REFERENCES jobs(id),
  skill_id INTEGER NOT NULL REFERENCES skills(id),
  kind TEXT NOT NULL CHECK (kind IN ('must_have','nice_to_have')),
  confidence REAL NOT NULL DEFAULT 0.8,
  PRIMARY KEY (job_id, skill_id)
);

CREATE TABLE job_embeddings (
  job_id INTEGER NOT NULL REFERENCES jobs(id),
  model TEXT NOT NULL,
  dim INTEGER NOT NULL,
  vector BLOB NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (job_id, model)
);

CREATE TABLE recommendations (
  id INTEGER PRIMARY KEY,
  job_id INTEGER NOT NULL REFERENCES jobs(id),
  candidate_id INTEGER NOT NULL REFERENCES candidates(id),
  run_id INTEGER NOT NULL REFERENCES runs(id),
  total_score REAL NOT NULL,
  rank INTEGER,
  gate_pass INTEGER NOT NULL,
  gate_failures TEXT,                        -- json array of reasons
  score_breakdown_json TEXT NOT NULL,        -- per-component values+weights
  rationale TEXT,
  status TEXT NOT NULL DEFAULT 'new'
    CHECK (status IN ('new','saved','dismissed','applied','expired')),
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  reviewed_at TEXT,
  UNIQUE (job_id, candidate_id)
);
CREATE INDEX idx_recs_score ON recommendations(candidate_id, total_score DESC);

CREATE TABLE runs (
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL CHECK (kind IN ('discovery','refresh','on_demand','maintenance')),
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending','running','success','partial','failed')),
  triggered_by TEXT,                         -- scheduler|ui|cli
  started_at TEXT, finished_at TEXT,
  stats_json TEXT, error_text TEXT
);
CREATE TABLE run_events (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES runs(id),
  ts TEXT NOT NULL DEFAULT (datetime('now')),
  level TEXT NOT NULL,                       -- debug|info|warn|error
  node TEXT, message TEXT NOT NULL, data_json TEXT
);
CREATE INDEX idx_events_run ON run_events(run_id, id);

CREATE TABLE crawl_state (
  id INTEGER PRIMARY KEY,
  scope TEXT NOT NULL,                       -- source:key or company:{id}
  cursor TEXT, etag TEXT, last_success_at TEXT,
  consecutive_failures INTEGER NOT NULL DEFAULT 0,
  cooldown_until TEXT,
  UNIQUE (scope)
);

CREATE TABLE notifications (
  id INTEGER PRIMARY KEY, channel TEXT, payload_json TEXT,
  status TEXT NOT NULL DEFAULT 'pending', created_at TEXT, sent_at TEXT
);

CREATE TABLE settings (
  key TEXT PRIMARY KEY, value_json TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Future applier extension (created in its own migration when Phase 20 starts):
-- applications, application_events, tailored_documents
```

---

## 17. Python API endpoints (FastAPI, prefix `/api`)

| Method & path | Purpose |
|---------------|---------|
| GET `/healthz` | liveness + db/gateway status (`gateway_db: false` = gateway DB not created yet, no LLM call yet — not unhealthy) |
| GET/PUT `/api/profile` | current candidate profile (auto-creates single candidate) |
| POST `/api/profile/resume` | multipart upload → parse → preview (422 if LLM unavailable, e.g. placeholder keys; 415 non-PDF; over 10 MB: 422 when the declared Content-Length exceeds it, 413 when the actual body does) |
| PUT `/api/profile/skills` | edit skill list/weights |
| GET `/api/skills` | full canonical skill taxonomy (no `q` param) |
| GET/POST `/api/companies` | list (filters)/create manually; priority clamped 1..5 on create/patch/import; non-numeric/non-finite import priority increments errors; numeric truncated via int() then clamped 1..5 (missing/None->3; 0->1 (clamped) on import/seed) bool priority is rejected, not clamped (422 create/patch; errors import; skipped seed) |
| POST `/api/companies/import` | import seed companies from YAML content (422 on invalid YAML); priority clamped 1..5 on create/patch/import; non-numeric/non-finite import priority increments errors; numeric truncated via int() then clamped 1..5 (missing/None->3; 0->1 (clamped) on import/seed) bool priority is rejected, not clamped (422 create/patch; errors import; skipped seed) |
| PATCH `/api/companies/{id}` | pause/pin/label vertical/edit; priority clamped 1..5 on create/patch/import; non-numeric/non-finite import priority increments errors; numeric truncated via int() then clamped 1..5 (missing/None->3; 0->1 (clamped) on import/seed) bool priority is rejected, not clamped (422 create/patch; errors import; skipped seed) |
| DELETE `/api/companies/{id}` | blacklist |
| POST `/api/companies/{id}/refresh` | **planned** — not implemented |
| GET `/api/jobs` | paginated explorer (`q` → FTS5; filters: city/work_mode/vertical/status; params `page`, `per_page`) |
| GET `/api/jobs/{id}` | detail incl. raw_json |
| GET `/api/recommendations` | ranked list (min_score, status, vertical, `include_gated`, `page`, `per_page` filters) |
| GET `/api/recommendations/{id}` | **planned** — not implemented (list carries the breakdown) |
| PATCH `/api/recommendations/{id}` | status transitions (new/saved/dismissed/applied) |
| GET `/api/runs` | history + stats |
| POST `/api/runs/discovery` | trigger now (409 if one running; atomic via `create_if_not_active` single `BEGIN IMMEDIATE` session — sequential 409 semantics, API route now atomic too, see §19) |
| POST `/api/runs/{id}/cancel` | cooperative cancel flag |
| GET `/api/runs/{id}/events` | paged events |
| GET `/api/runs/{id}/stream` | **SSE** live progress (worker publishes to events table; API tails it) |
| GET `/api/settings` | seeded read-only copies of the settings table (keys seeded from `config/app.yaml` by `ensure_defaults`: `schedule`, `sources`, `discovery`, `scoring_weights`, `embeddings`, `salary_hard_floor_lpa`) |
| PUT `/api/settings` | 4 keys only: `scoring_weights`, `salary_hard_floor_lpa`, `discovery.max_llm_calls_per_run`, `discovery.enrich_per_run_cap` (unknown keys → 422) |
| GET `/api/stats/dashboard` | counts, top companies, score histogram, freshness |
| GET `/api/llm/stats?days=` | read-only rollup from `data/gateway.db` (calls, tokens, cost, by provider/node) |
| POST `/api/inbox/import` | **planned** — inbox is scanned by the worker interval job, not an API |
| GET `/api/inbox/preview` | **planned** — not implemented |

Errors: two shapes. Unhandled errors (500) use problem-style JSON
`{title, status, detail}` (no `instance` field); expected `HTTPException`
errors use FastAPI's `{detail}` shape. Status rows:
401 (token configured and Bearer header missing/wrong),
404 (unknown id), 409 (concurrent run attempt), 422 (Pydantic/domain
validation, incl. resume curation unavailable on placeholder keys),
500 (unexpected). `AppError` mapping (`api/main.py` handler,
problem-style `{title, status, detail}`):
`ConflictError` → 409 `{'title': 'Conflict', 'status': 409, 'detail':
'conflict'}`, `AdapterError` → 502 `'adapter error'` /
`'Bad Gateway'`, `GatewayUnavailableError` → 503 `'gateway
unavailable'` / `'Service Unavailable'`, `DatabaseError` → 500
`'database error'`, other `AppError` → 500 `'internal error'` — see
`docs/api.md` error section. Direct `HTTPException` 409s from the runs
route use the `{detail}` shape ('a run is already active'; see
`docs/api.md`), unlike the problem-style `AppError` handler responses
above. Per D1, non-loopback `APP_HOST`
requires `JH_API_TOKEN` (Bearer) — see `docs/api.md`. CSRF: non-GET with cross-origin Origin/Referer is rejected 403; headerless requests not checked (browser always sends Origin on POST); the 403 shape is
`{'detail': 'forbidden origin'}`. `Referer` is accepted as an
alternative to `Origin` when it is same-host. The web UI sends it
automatically.

---

## 18. Frontend (HTML + vanilla JavaScript only)

No framework, no build step. FastAPI serves `web/` at `/`; pages share
`css/style.css`, `js/api.js` (fetch wrapper, SSE helper, toast), and
`js/components.js` (card, badge, slider, table renderers, tiny hash-router).

| Page | Content |
|------|---------|
| `index.html` | Dashboard: KPI cards (new recs today, active companies, jobs indexed, run health, LLM spend 7d), latest top-10 recs, recent runs strip |
| `profile.html` | Profile form (roles chips, cities, remote select, salary sliders, exp), resume upload + parsed-preview editor, skills manager with alias suggestions |
| `companies.html` | Table w/ filters (vertical/status/ATS), inline actions (pin/pause/refresh/label), import dialog, add-manually form |
| `recommendations.html` | Card list sorted by score: total + component bar chart (pure CSS), gates, rationale, links; actions save/dismiss; keyboard nav; bulk dismiss |
| `jobs.html` | Raw explorer with FTS search box, filter sidebar, duplicate/stale badges |
| `runs.html` | Run history table + detail drawer with live SSE event console (level-colored), stats counters, errors panel |
| `settings.html` | Scoring weights sliders (live-sum validation = 1.0; page PUTs only `scoring_weights` — the other 3 PUT-able keys are API-only, see §17); no schedule editor, no source toggles, no token-budget UI — schedule and source enabling are yaml-only (`config/app.yaml`, see §22) |

SSE protocol (actual `routes/runs.py`): `event: log`
(`{id, level, node, message}`), `event: done` (`{status, stats}`),
`event: timeout` (`{}`). Reconnect handled in `api.js`.

---

## 19. Scheduled crawling/polling workers

Worker entrypoint `python main.py worker` runs APScheduler
(`BlockingScheduler`, per D6) with jobs defined in `workers/scheduler.py`:

| Job | Cadence (default) | Action |
|-----|--------------------|--------|
| discovery_run | daily 06:15 | insert `runs(pending, discovery)` then execute DiscoveryGraph end-to-end |
| quick_poll | every 120 min | T2 aggregators + pinned (priority 5) companies only — cheap incremental fetch |
| company_refresh | weekly, staggered chunk/night | Scout verification for companies `last_checked_at > 7d`, batch of ≤ 50/night |
| stale_sweep | daily 04:00 | mark unseen > 45d jobs stale, expire their recommendations |
| inbox_scan | every 30 min | parse new files in `data/inbox/**` |
| pending_executor | every 30 s | drain UI/CLI-enqueued pending runs |

That is 6 real jobs. There is no `maintenance` job (that row was aspirational
and is removed). All times are in the host's local timezone — there is no
timezone attached to the scheduler (no IST/UTC conversion); the cron values
above are literal local times. Execution contract: worker claims `runs` rows
with `BEGIN IMMEDIATE` (single-writer discipline), streams progress into
`run_events` (which powers SSE), finalizes status success/partial/failed with
stats, releases lock. Missed schedules coalesce (max_instances=1,
misfire_grace=1h). Cron strings live in `config/app.yaml` under `schedule:`
(yaml-only; not editable via the Settings UI — see §17/§18).
Atomic enqueue: `RunManager.enqueue` and the API route
`POST /api/runs/discovery` both use
`RunsRepository.create_if_not_active` (single `BEGIN IMMEDIATE`
session): the active-run check and insert are one atomic step, so a
concurrent caller loses the race and gets 409 (`RunManager`:
'another run is pending or running'; API route: 'a run is already
active') instead of a duplicate run. Sequential (non-concurrent)
semantics are the same: a second enqueue while one is
pending/running is 409.

---

## 20. Error handling

Layered, fail-soft by design:

| Layer | Policy |
|-------|--------|
| HTTP/tool calls | httpx timeouts (connect 10s/read 30s); `HttpClient.get_json` makes 2 attempts total (one retry) on 429/500/502/503/504 and on transport errors, sleeping `_retry_delay` (Retry-After honoured, capped at 30s, plus jitter); `get_text` has no retry (single attempt after the robots check); circuit breaker per source; robots/blocks → `cooldown_until`, never tight-loop |
| LLM (inside gateway) | provider cascade + retries + fallback already handled; app adds: structured-parse repair loop (≤ 2), then rule-based fallback flagged `confidence<0.5`; `AllProvidersFailedError` ⇒ node degrades per §9 ladder |
| Graph nodes | `safe_node` wrapper: catch-all → NodeError + run_event, continue; required-node failure ⇒ run `partial`/`failed` but partial results still persisted |
| Validation | Pydantic models at every boundary; checkpoint-resumed malformed, non-dict, or identity-less `inserted` items are logged and skipped (`graph/state.py: _normalize_inserted` logs `skipping inserted item without identity` / `skipping malformed inserted item` / `skipping non-dict inserted item`) — that path is never silent; scoring skips non-int `job_id` with `skipping job with non-int job_id` (§15). No code path currently writes `jobs.status='error'` quarantine rows (the Watchdog only counts such rows if present), so the old blanket quarantine claim does not hold elsewhere |
| API | exception handlers → 500 problem-style `{title, status, detail}`, expected errors → `HTTPException` `{detail}`; unexpected errors logged with run/request ids; 401/404/409/422 semantics as in §17 |
| Data integrity | content_hash uniqueness (idempotent reruns); FK constraints ON; migration transaction-per-script with version table |
| Crash recovery | LangGraph SqliteSaver checkpoints; orphan recovery: worker/scheduler fails `running` runs older than 120 min, CLI `run-discovery` uses >5 min (5m), and `pending` runs older than 1h (1 hour) are failed by the same sweep → marks failed, optionally resumes from checkpoint |

User-visible errors always carry a run/event reference so Runs page explains them.

---

## 21. Logging and observability

- Stdlib `logging`; JSON formatter; `RotatingFileHandler`
  `data/logs/app.log` (10 MB × 5). Invalid log levels raise `ValueError`.
- Context propagation via `contextvars` (run_id, node, source_key) injected by
  `safe_node`/middleware — every line correlates to UI-visible entities.
- Levels: DEBUG dev default off; INFO pipeline milestones; WARN degraded
  paths (fallback used, breaker opened); ERROR with stack (once per issue).
- Dual audit trail: app-side `run_events` (pipeline story) +
  gateway-side `data/gateway.db` (tokens/cost/latency/provider per call,
  correlated via `session_id='{run_id}:{node}'`). `/api/llm/stats` joins both
  worlds for the dashboard's spend card.
- No PII in logs: resume text/token counts yes, contents no; secrets never
  logged (gateway already logs only lengths/system-prompt presence).

---

## 22. Configuration and secrets

- `src/job_hunter/llm_gateway/.env` (gitignored; **the canonical key file**,
  created by `setup.sh` from the 9-provider
  `src/job_hunter/llm_gateway/.env.example` template): provider keys,
  `GATEWAY_DB_PATH`, `GATEWAY_CACHE_PATH`, `GATEWAY_DASHBOARD_PORT`.
  The repo-root `.env.example` is APP-only and points at the gateway template.
- `config/app.yaml` (committed defaults + local overrides): `app` bind
  host/port/log level, `salary_hard_floor_lpa`, `schedule` crons/intervals,
  `sources` enabled-only toggles (incl. `workday`; `manual` is the inbox
  scan, not an adapter), `scout` (company discovery), `discovery` (run
  knobs), `scoring_weights`, `embeddings` model id, `mcp` stdio commands.
  Per-source RPM is adapter-owned (adapters pass explicit `rpm=`), not
  read from config. Enabling/disabling sources and editing the schedule
  are yaml-only: change `config/app.yaml` and restart (no Settings UI
  for schedule or source toggles).
- `AppSettings` (`core/config.py`) reads yaml + `APP_*`/`SALARY_HARD_FLOOR_LPA`
  env at boot; the `settings` table is seeded read-only from yaml
  (`ensure_defaults`) and only the 4 PUT keys in §17 may be changed via
  the API/UI. `APP_*` vars must be exported/sourced in the shell — see
  `docs/runbook.md`.

---

## 23. Security, privacy, ToS posture

- Servers bind 127.0.0.1 only; no inbound exposure; no telemetry.
  `TrustedHostMiddleware` is hardcoded to `127.0.0.1`/`localhost`/
  `testserver` — a `JH_API_TOKEN` alone does not enable LAN exposure.
  When a token is set, the UI prompts for it on `401` and sends the
  Bearer header thereafter. CSRF: non-GET with cross-origin Origin/Referer is rejected 403; headerless requests not checked (browser always sends Origin on POST). The 403 body is
  `{'detail': 'forbidden origin'}`; `Referer` is an accepted
  alternative to `Origin` when same-host.
- Egress limited to: configured job sources, DDG lite search, LLM providers.
- ToS: no scraping of LinkedIn/Naukri/Instahyre/Hirist; manual-export inbox
  keeps those ecosystems available without violations. robots.txt honored for
  HTML fetching; JSON board APIs used as intended for public distribution.
- Privacy: all data local; resume leaves machine only toward enabled LLM
  providers — flipping `execution_order` to ollama-only yields a fully offline
  system (documented tradeoff: lower extraction quality).

---

## 24. Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| ATS schema/UI drift breaks adapters | silent coverage loss | contract tests against fixtures + Watchdog freshness alerts per source |
| Anti-bot on career pages | some companies unfetchable | graceful skip + needs_review; optional browser-mcp later |
| Free-tier LLM flakiness/quota | degraded enrichment | gateway cascade + rotation already; caches make reruns nearly free; rule fallbacks |
| Hallucinated extraction (skills/salary) | bad matches | strict schemas, repair loop, confidence thresholds, spot-check queue in Jobs UI |
| Py3.14 wheel lag (onnx/torch) | install friction | fastembed-first; documented 3.12 conda fallback |
| SQLite write contention | slow UI spikes | WAL + busy_timeout + single-writer worker; reads are fine concurrently |
| Over-scraping a host | IP blocks | token buckets, ETag reuse, low cadence, per-source breakers |
| Scope creep toward applier too early | stalled v1 | applier explicitly parked; its schema/abstractions designed but unbuilt (roadmap) |

---

## 25. Decisions and remaining questions

Resolved (2026-08-24):

- **D1 — Target role:** Staff-level Data Scientist positions in India.
- **D2 — Resume input:** LaTeX-generated PDF; pypdf-based parsing (LaTeX PDFs
  extract cleanly; ligature/unicode-normalization step included). DOCX optional later.
  (D2 overload note: in §3/§7/§8 "per D2" means graph nodes use repositories
  directly — see the note in §7. Here D2 is this resume-input decision.)
- **D3 — Compensation:** hard floor 45 LPA, enforced as a gate (§15); nothing
  below is recommended. Unknown-salary postings pass but flagged `salary_unverified`.
- **D4 — Runtime:** Python 3.14+ dedicated venv.
- **D5 — llm_gateway:** unchanged. Its existing `tools` / `tool_choice`
  support on `supports_tools: true` providers covers every tool-calling need;
  integration happens purely in external adapter code (`llm/client.py`).
  The former LangChain bridge (`llm/langchain_bridge.py`,
  `GatewayChatModel`) and MCP client tool loader (`llm/mcp_tools.py`,
  `MCPClientManager`) are deleted — see §7–§9.
  If a future gap appears (e.g., native JSON-schema response_format), a minimal
  gateway change may be proposed explicitly first.
- **D6 — Ports:** app API 8088; gateway dashboard 8099; both localhost-only.

Remaining:

- **OQ-1 — Industry vertical(s) to prioritize** for a Staff DS search
  (taxonomy ready; until you choose, all verticals are in scope weighted
  toward product companies and GCCs).
- **OQ-2 — Notifications:** dashboard + SSE assumed for MVP.

---

*Companion document:* `roadmap.md` maps every item above onto sequenced
feature branches with acceptance criteria and the squash-merge workflow.

---

## 26. Hardening changelog

### Loop 1 — 2026-08-24 (`feature_16_hardening_run1`)

End-to-end run/debug pass; nine defects found by exercising real paths:

| # | Defect | Fix |
|---|--------|-----|
| 1 | Double-claim: `execute_pending_run` pre-claimed oldest pending, then `RunManager.execute` re-claimed → 'skipped' runs | `execute(run_id=None)` now owns claiming; worker helper no longer pre-claims |
| 2 | UI-triggered runs stayed `pending` forever unless nightly cron | Worker gained 30s `pending_executor` interval job that drains queued runs |
| 3 | `load_profile` never loaded candidate skills → skill-coverage always neutral | Reads `candidate_skill_names()` from profile repo |
| 4 | Raw user text hit FTS5 MATCH unescaped (`+`, quotes) → 500s on `/api/jobs?q=` | `sanitize_fts()` builds AND-of-quoted terms |
| 5 | Profile UI showed empty salary floor even though settings had 45 | GET `/api/profile` injects floor from settings when unset |
| 6 | CLI `discover-companies --seeds` silently ignored its argument | `ingest_seeds(settings, seeds_dir)` override plumbed through |
| 7 | Crawl watermark never stored (`set_success(scope, None)`) → full refetch each poll | fetch_pair stores max posted_at cursor per scope |
| 8 | Watchdog test polluted global env via `os.environ` direct write | monkeypatch.setenv isolation |
| 9 | `GatewayChatModel._generate` used deprecated `get_event_loop().run_until_complete` | running-loop detection with `asyncio.run` fallback |

Verification: 46 tests (5 new regressions), live API re-smoke
(floor injection=45.0, FTS special-chars=200).

---

## 27. Doc keeper

`docs/architecture.md` is updated in the same commit as any behavior
change it describes (see §26 for the format). Agent briefs live in
`agents/*.md`; session learnings land in `skills/learnings.md`.

### Loop 2 — 2026-08-24 (`fix_gateway_root_path`)

Live-log triage of repeated `/api/profile/resume` 500s:

| # | Defect | Fix |
|---|--------|-----|
| 10 | `AppSettings.gateway_root` resolved one level too deep (`job_hunter/job_hunter/llm_gateway`) → `ConfigError` on every resume upload | corrected to `parents[2]`; regression test asserts the YAML exists |
| 11 | Curator construction sat outside the route's try-block → unhandled 500 instead of clean 502 | moved inside try; clients now get `resume curation unavailable: …` |
| 12 | With the path fixed, `enrich_jds` re-wrapped an already-built `TokenBudget` (`**model`) and crashed | accepts model or dict |

Verification: 47 tests green; live upload attempt returns structured 502
(remaining failures are placeholder keys, by design).

> **History note (superseded by 422):** the structured 502 responses
> recorded in Loop 2 above are superseded by 422 — current
> resume-curation failures return 422 (`{"detail": "resume curation
> unavailable"}`), see §17 and `docs/api.md` / `docs/runbook.md`.

### Loop 3 — 2026-08-24 (`feature_18_expanded_coverage_v2`)

Coverage expansion to every DS-hiring employer regardless of industry:
seeds 30 → 117 across all verticals + GCCs (retail_tech/energy taxonomy
added); Workday CXS adapter; Himalayas aggregator; career-page detector
v2 (Workday host fingerprint joined as board_ref, Greenhouse/Lever
slug-guess fallback, bogus-ref guard, bs4 XML warning filtered); Ashby
plain-GET fix; enrichment per-run cap with self-healing backlog;
scoring backlog fallback so recommendations converge without refetching;
DB-settings overrides for run knobs; CLI recovery covers stale pending
with configurable TTL and a guaranteed finalizer; SIGTERM graceful
finalization in main.py. Quote-style normalization pass (61 literals).
Verified: 48 unit tests green; live discovery runs progressing
(enrichment converges across runs via backlog).
