# Job Hunter — Roadmap (feature-branch plan)

Workflow per your convention: one feature branch
`agentic_ai/job-hunter/feature_<xx>_<name>` off `agentic_ai/job-hunter`,
then squash-merge and delete the branch. One feature = one squash commit.
Phases are strictly ordered except where marked ∥ (parallel-safe).

| Branch | Phase | Scope | Key deliverables | Done when |
|--------|-------|-------|------------------|-----------|
| `feature_00_project_scaffold` | P0 | Skeleton + tooling | pyproject (dual top-level pkgs), AppSettings, logging setup, db.py + migrations runner, `/healthz`, ruff/pylint clean, CI-less test harness | `pytest` green; api boots and serves health |
| `feature_01_candidate_profile` | P0 | Profile domain | candidates/profiles/resumes/skills tables, resume parser (LaTeX-PDF via pypdf), Profile Curator agent (structured extract via gateway), profile repositories | upload → parsed profile persisted, editable via API |
| `feature_02_mcp_servers` ∥ | P1 | Tool layer | sources-mcp, store-mcp, resume-mcp with typed tools + tests; MCPClientManager lifecycle | tools callable from a smoke script; robots/token-bucket enforced |
| `feature_03_source_adapters_t1` ∥ | P1 | T1 ATS adapters | SourceAdapter ABC; greenhouse/lever/ashby/workable/smartrecruiters/recruitee/personio + fixtures contract tests | each adapter returns normalized RawJobRecords from saved payloads live-fetch smoke-tested |
| `feature_04_company_discovery` | P1 | Scout pipeline | seeds loader, name harvest from postings, domain resolution, ATS fingerprinting, company_aliases, vertical classifier (rules+LLM) | seed yaml → verified companies w/ board refs + verticals in DB |
| `feature_05_discovery_graph` | P2 | LangGraph core | state.py reducers, safe_node wrapper, DiscoveryGraph wiring, Send fan-out fetch, SqliteSaver checkpointing, run_manager claim/finalize | graph runs end-to-end against stubbed adapters; resumes after kill -9 mid-run |
| `feature_06_scheduler_worker` | P2 | Worker process | APScheduler jobs table (§19), quick_poll, inbox scan skeleton, orphan-run adoption | cron fires discovery daily; SSE-visible progress events land in run_events |
| `feature_07_data_quality` | P2 | Data-quality depth | stale sweep scheduling, FTS search hardening, quality scoring, quarantine review path | rerunning a source inserts zero dupes; 45d staleness works |
| `feature_08_jd_enrichment` | P3 | JD Analyst | structured extraction (skills must/nice, salary LPA regex+LLM, exp band, mode), repair loop, extraction cache by hash, token budget, 45-LPA floor gating | enriched jobs ≥ 95% schema-valid on fixture set; unchanged re-crawl costs 0 tokens; sub-45 postings excluded |
| `feature_09_embeddings_matching` | P3 | Semantic layer | fastembed embedder (+FTS5 fallback), job_embeddings storage, candidate doc builder, cosine service | vectors computed once per unique posting; similarity sane on golden pairs |
| `feature_10_scoring_ranking` | P3 | Recommender | gates + weighted components + breakdown json, rank persistence, rationale templates, feedback reweighting (bounded) | recommendations ranked with explainable breakdowns; dismiss feedback shifts scores within ±20% |
| `feature_11_api_layer` | P4 | REST surface | all routes §17 incl. problem+json handlers, settings persistence, llm stats rollup | endpoint integration tests green; OpenAPI docs accurate |
| `feature_12_web_ui` ∥ | P4 | Frontend | 7 pages + shared js/css, SSE console, score-breakdown bars, keyboard review flow | full happy path usable mouse-only and keyboard-only |
| `feature_13_manual_inbox` | P4 | T4 channel | watched-folder scanner, LinkedIn-saved-page & Naukri-CSV parsers, preview/confirm API | dropping an export produces deduped jobs flagged source=manual |
| `feature_14_watchdog_polish` | P5 | Reliability | Watchdog agent report, adapter freshness alerts, breaker cooldown surfacing, log rotation verify | run summary lists degraded sources with reasons |
| `feature_15_hardening_qa` | P5 | Pre-release | perf pass (indexes, batch sizes), staff-DS seed set (metros + remote; verticals per OQ-1), README runbook, v0.1.0 tag | 7 consecutive nightly runs without manual intervention |
| **Future applier track** | | | | |
| `feature_20_application_schema` | P6 | Foundation | applications/application_events/tailored_documents migrations; tracker states; UI tab (read-only) | schema live, zero behavior change elsewhere |
| `feature_21_tailoring_agent` | P6 | Docs generation | JD-aware resume bullet tailoring + cover letters (gateway), DOCX/PDF render, human edit loop | tailored doc previewed/approved in UI, stored per application |
| `feature_22_apply_adapters` | P7 | Submission | Playwright browser-mcp apply flows for Greenhouse/Lever/Workable forms; dry-run mode with screenshots + filled-field diff | dry-run produces accurate preview artifacts for sample boards |
| `feature_23_approval_guardrails` | P7 | Autonomy control | approval queue (default: nothing sends without you), rate limits/day, audit trail, per-company autonomy levels | submit happens only after explicit approval; every action audited |

Milestone tags: `v0.1.0` after feature_15 (discovery MVP),
`v0.2.0` after feature_23 (assisted applying).

## Squash-merge recipe

```bash
git checkout agentic_ai/job-hunter && git pull
git checkout -b agentic_ai/job-hunter/feature_05_discovery_graph
# ... work, commit ...
git checkout agentic_ai/job-hunter
git merge --squash agentic_ai/job-hunter/feature_05_discovery_graph
git commit -m 'feat(graph): discovery graph with fan-out fetch and checkpoints'
git branch -D agentic_ai/job-hunter/feature_05_discovery_graph
```

Dependency notes: 01→(02∥03)→04→05→06→07→08→09→10→11→(12∥13)→14→15;
applier track starts only after v0.1.0.
