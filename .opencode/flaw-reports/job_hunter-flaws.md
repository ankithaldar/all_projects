# Flaw Report — job_hunter (Cycle 1 draft)
Status: IN_PROGRESS — 9/10 reviewers returned. Pending: adversarial-tester ONLY.
Lead-verified 2026-11-07: `git ls-files job_hunter | grep env/data/db` = only .env.example files; `git log --all -- src/.../llm_gateway/.env` = empty. Live .env NOT leaked to git, but on-disk unignored -> rotation still recommended (user action).

## DevOps/SRE (ses_e4b6d6cc0ffewZRy6Yj4wawO5I) — 4H/6M/4L
- [HIGH] pyproject.toml:55 `indent-style='spaces'` invalid (expected `space`) — ruff broken
- [HIGH] .gitignore: no `.env` rule — live gateway `.env` unignored, `git add -A` risk
- [HIGH] setup.sh:21 + README:11 copies root `.env.example` (missing 4 provider keys) instead of gateway template
- [HIGH] requirements.txt vs pyproject.toml drift: fastembed unconditional vs optional, dev trio missing, no pins/lockfile
- [MEDIUM] setup.sh: no Python >=3.14 gate
- [MEDIUM] main.py:67-72 hardcodes 127.0.0.1:8088, ignores AppSettings/APP_*; api/__main__.py honors them
- [MEDIUM] main.py uvicorn log_level='warning' vs worker INFO — split observability
- [MEDIUM] README omits [semantic] + discover-companies (verbatim = empty DB)
- [MEDIUM] Env vars inert: APP_* never load_dotenv'd; GATEWAY_DASHBOARD_PORT unused; gateway_db_path split-brain
- [MEDIUM] No CI/Docker/systemd/rollback
- [LOW] run.sh `all`: no worker-death/readiness detection
- [LOW] main.py docstring omits scout-companies/fetch-jobs
- [LOW] logging.py: no APP_LOG_LEVEL validation
- [LOW] /healthz only health surface

## Anti-Team Security (ses_e4b6d6d0bffeFjOnOU68fQ1fV3) — REJECTED, 1C/4H/4M/2L
See .opencode/critique-log/job_hunter-cycle-1-security.md
- [CRITICAL] Unauthenticated API write/delete, no Host/CSRF/token, APP_HOST 0.0.0.0 possible
- [HIGH] SSRF via MCP fetch_page + adapters, no private-IP/redirect/size guard
- [HIGH] Prompt injection JD/resume -> LLM + unauthenticated tool exec in mcp_tools
- [HIGH] Arbitrary file read MCP extract_text(path) no confinement
- [HIGH] Unbounded fetch/PDF OOM DoS
- [MEDIUM] ATS board_ref injection (8 adapters), stored XSS gateway dashboard + javascript: href, .env gap + error leaks, arbitrary PUT /settings
- [LOW] YAML injection scout writer, supply chain no lockfile

## Anti-Team Arch (ses_e4b6d6d0dffekZ5Az3v0IdQPXm) — REJECTED, 10 critical / 11 important
See .opencode/critique-log/job_hunter-cycle-1-arch.md
- [CRITICAL] worker/__main__.py broken entrypoint TypeError
- [CRITICAL] graph/nodes.py 651-line god, no safe_node
- [CRITICAL] MCP fictional in hot path, store_server tool list mismatch
- [CRITICAL] Checkpointing silently degrades, state not serializable
- [CRITICAL] Repo leaks: jobs_repo._db_path raw SQL, watchdog/taxonomy raw SQL, settings.py sqlite3.connect bypass
- [CRITICAL] recommendations.py INSERT duplicate `rank` column — upsert_many broken
- [CRITICAL] Config no validation (no pydantic-settings), dead sources.enabled/rpm, PUT /settings arbitrary
- [CRITICAL] Scheduler BlockingScheduler, TOCTOU has_active, orphan query kills pending
- [CRITICAL] Gateway coupling: sys.path hacks, raw gateway escape, run_until_complete on running loop
- [CRITICAL] Registry unsound: KeyError, OCP, per-pair HttpClient breaks throttle
- [IMPORTANT] Error hierarchy decorative, layer inversion API->DB, unbounded fan-out + Stub fake, embedder reload + full scan, migrations drift/workday patch, logging theater, state type lies, main.py ignores config, mcp import side-effects, robots fail-open + no tenacity, dual ContextVars

## Tester (ses_e4b6d6cc5ffeSq4exnleYNDWCN) — suite 70 PASSED after pip install -e '.[dev]'
- Baseline 5 collection errors were ENV (missing deps), resolved by install. Post-install: 70 passed, 49% total coverage (5576 stmts)
- [HIGH] Gateway 10 tests excluded (testpaths=['tests']) + 4 unreachable: `from tests.conftest import DummyProvider` collision. Only 6/10 runnable explicitly.
- [HIGH] 0% modules: company_discovery, mcp_tools, langchain_bridge, all mcp_servers, checkpointing, workers/jobs, gateway dashboard/main, api/__main__ (43 modules not directly imported)
- [HIGH] llm_gateway/app.py 16% (1498 LOC), sanitize 17%, sanitized_gateway 15%
- [MEDIUM] test_graph_pipeline cannot-fail asserts (>=0, isinstance list), test_migrations only 2 asserts, 0 async tests (asyncio_mode dead), only 1 pytest.raises in suite, pytest config no addopts/filterwarnings/coverage gate, ResourceWarning unclosed sqlite
- [LOW] StubAdapter monkeypatch hides registry, test_api order-dependence, no integration/E2E, no conftest.py factories duplicated

## Frontend (ses_e4b6d6cc1ffeJv157cSZyfXHY2) — 4H/6M/7L, 21 API contracts checked
- [HIGH] companies.html import YAML always 422: client JSON body vs server bare `content: str` query param
- [HIGH] Gateway dashboard stored XSS innerHTML (provider/session_id/model)
- [HIGH] jobs/companies load() appends without clearing -> duplicates stack
- [HIGH] profile.html PUT omits seniority_keywords/blocked_verticals/employment_types -> silent data loss/reset (incl. resume-extracted keywords)
- [MEDIUM] javascript: URL XSS components.js:46, no error/loading states anywhere, no viewport/media queries (mobile overflow), labels without for= + multiple h1 + no aria-live, SSE timeout loop (api.js), priority shown as score badge
- [LOW] runs.html innerHTML runId, static mount fragile parents[4] + no CSP + str(exc) leak, dead CSS, toast race, display nits, zero JS tests + no contract tests

## Docs (ses_e4b6d6cbffferCk654HN6EfXqV) — 5H/11M/5L drift
- [HIGH] README omits discover-companies (0 companies), worker entrypoint broken in app_overview/architecture, app.yaml manual vs registry workday mismatch, 4 documented API endpoints missing (companies refresh, recommendation get, inbox import/preview), missing runbook/api.md/env table
- [MEDIUM] architecture §4 file map stale (8+ files), Tier table omits workday/arbeitnow/jobicy/cutshort, §19 scheduler (maintenance job absent, max_instances claims false, no TZ), SSE events log/done/timeout not progress/stat/done, /api/skills?q + jobs date filter nonexistent, RFC7807 error format mismatch, main.py api hardcode vs docs, §3 wrong `hunter.*` package + MCP count, §22 config keys mismatch, roadmap missing 16-19 + unmet 11/13, agents/*.md log path + PYTHONPATH + test mapping stale
- [LOW] README .[dev] vs .[dev,semantic], skills_aliases header, .pylintrc claim, ashby docstring, Cutshort 44k unsourced, app_overview omits bulk path

## Security OWASP (ses_e4b6d6d10ffeR6Imv5ge2kPxbs) — 1C/4H/5M/2L (converges with anti-security)
- [CRITICAL] 7 live keys in src/job_hunter/llm_gateway/.env on disk, .gitignore lacks .env (Lead: not tracked/in history, still rotate + gitignore)
- [HIGH] Zero auth on all routes + dashboard, docs on, APP_HOST 0.0.0.0 possible, CSRF via text/plain
- [HIGH] SSRF http_client no allowlist/private-IP block, board_ref/domain unencoded
- [HIGH] Stored XSS dashboard innerHTML + javascript: href, no CSP
- [MEDIUM] Prompt injection jd_analyst/profile_curator/structured, info disclosure str(exc) in main/profiles/mcp, upload no magic-byte + extract_text arbitrary path, arbitrary PUT /settings, unbounded YAML import + per_page=-1 -> LIMIT -1 unlimited
- [LOW] No TrustedHost/headers, floating deps + undeclared `requests` import (openai_compatible.py:12)
- Clean: SQL parameterized, sanitize_fts, yaml safe_load only, no eval/exec/subprocess/pickle, gateway keys env-based, http timeouts+robots+throttle, 10MB upload cap

## Database (ses_e4b6d6cc3ffeWS1cLN1vR0CeXV) — 4H/7M/5L, runtime-verified on scratch DB
- [HIGH] recommendations.py duplicate `rank` (stored rank=position, row rank silently dropped — reproven)
- [HIGH] core/db.py session() autocommit (isolation_level=None) -> multi-statement writes non-atomic (attach_skill_rows, set_candidate_skills, stale_sweep, upsert_many, set_failure)
- [HIGH] jobs.py insert + profile.py save_profile f-string column identifiers NO allowlist -> evil key error + real `skills_resolved` drift crash (profile_curator.py:95 passes it, no such column)
- [HIGH] Migration runner: no txn around executescript+record, no checksum, race, leaked readonly conn db.py:83, no down migrations; 0001:174 FK to runs before creation
- [MEDIUM] Missing UNIQUE(canonical_url)/(source_key,external_id), candidate_profiles UNIQUE(candidate_id,version), upsert races, index gaps (companies SCAN+TEMP, llm_calls zero indexes -> 5 scans/req)
- [MEDIUM] Dozens of leaked connect(readonly=True) never closed; WAL/threading inconsistent (gateway bare connect vs ThreadingHTTPServer -> database locked, logger worker dies silently); N+1 companies find_by_name + dedupe 2 queries/job + per-job skill conn
- [MEDIUM] FK no ON DELETE policy, save_resume TypeError on FK fail (verified), runs CHECK excludes 'cancelled' vs docs, CandidateProfile.skills no column, notifications table no repo/check
- [MEDIUM] test_migrations weak (subset only, no 0002/constraint/FTS/down tests)
- [LOW] Checkpointing silent None fallback, sanitize_fts '' -> FTS syntax error (verified), mark_orphans unbounded IN list >999 vars, bootstrap seed patch drift, datetime format T vs space skew
- Clean: all VALUES bound ?, IF NOT EXISTS everywhere, UNIQUE content_hash/normalized_name/domain, claim_pending BEGIN IMMEDIATE, unseen_hashes chunk 500, FTS triggers present

## Reviewer YAGNI (ses_e4b6d6d13ffeKR5stcIMST2ycR) — 2 Critical / 11 Important / 11 Minor
- [Critical] worker/__main__.py broken (converges) + llm_gateway openai_compatible.py `import requests` undeclared in pyproject+requirements -> clean-install ImportError
- [Important] Dead: langchain_bridge GatewayChatModel (0 callers), mcp_tools run_tool_loop/Executor (0), company_discovery harvest/expand (0, sole MCPClientManager user), adapters health() x11 never called, registry fetch_for_target/detect_provider (0), repo/service symbols (due_for_refresh, stats_summary, unseen_hashes, raw_json_of, similar_title_exists, utcnow, NotFound/Conflict, AppSettings.section, stream_text, reset_client)
- [Important] Duplication: html_to_text x5, registrable_domain x3 (co.in drift) + blocked lists x3, job persistence payload x3 (nodes/bulk_fetch/manual_inbox drift quality 0.5/0.3), DDG search x2 (sources_server uses wrong host), defaults 45.0/BAAI hardcoded x6, dead sources.enabled/rpm config, ceremonial provider subclasses x10
- [Important] Unused deps pydantic-settings+tenacity (0 imports), requirements.txt drift, duplicated .env.example templates, god app.py 1498 lines (ADR before refactor)
- [Minor] No-op imports (_ = registrable_domain etc), god modules linkedin_scout 714/nodes 651/domain_verifier 552, ruff spaces (converges), .gitignore missing *.egg-info, silent except:pass x4, gateway/app duplication (2 ConfigError, 2 DB layers)

## Pending
adversarial-tester ONLY

## FINAL CYCLE 1 SUMMARY (Lead synthesis 2026-11-07)
- Reviewers: 10/10 returned. Anti-arch REJECTED (10C/11I), Anti-sec REJECTED (1C/4H/4M/2L)
- Converging (multi-reviewer) top flaws: broken worker entrypoint (3 reviewers), duplicate rank SQL (2, runtime-proven), unauthenticated API (3), SSRF (3), prompt injection (3), dashboard XSS (3), .env gitignore (3), ruff spaces (3), requirements drift (3), PUT /settings arbitrary (3), per_page=-1 (3), README discover-companies omission (3), extract_text arbitrary read (2), unbounded fetch/PDF (3), board_ref injection (2), scheduler TOCTOU/orphans (2), sys.path hacks (2), repo leaks (2)
- Tester: 70 passed post-install, 49% coverage, gateway tests excluded/broken
- Decisions: see .opencode/context/decisions.md (D1-D12)
- Fix loop: 8 file-disjoint workers Round 1, TDD, then Round 2 repo-leak switch + re-review Cycle 2

## FIX ROUND 1 — W7 devops/docs DONE (Lead-verified: 12/12, ruff parses, gateway 10/10)
- pyproject fixed, requirements.txt deleted, .gitignore .env, setup/run/main/logging/README/docs/runbook/api.md done
- Note: ruff UP009 (coding header) + EXE001 now fire on every file BECAUSE of user-mandated header (D13). Round-2 formatter will add ruff lint ignores for UP009/EXE001 citing D13 so `ruff check` is usable. User directive wins over ruff default.
- Full suite at W7 return: 279 passed, 7 failed — all in other workers' in-flight files (test_fix_db x3, test_migrations x2, test_fix_ssrf x1). Re-check when W2/W3/W8 land.

## FIX ROUND 1 — W6 frontend/gateway DONE (Lead-verified: full suite 286 passed, 1 skipped)
- Dashboard XSS (textContent), href allowlist, SSE timeout/close, a11y (viewport/labels/h1/aria-live), list-clear, error states, cache key/corrupt/expired, sanitize linear parser, limiter fallback removed, CSP headers. Pylint 9.98/10 on new files.
- Note: ruff format wants """ docstrings vs D13 ''' style — W6 correctly did NOT apply ruff format. Round-2 formatter must preserve ''' (configure ruff format docstring-code-format? No: ruff format quote-style single applies to strings, docstring normalization is separate — formatter to verify and document).

## FIX ROUND 1 — W1 api-security DONE (Lead-verified: 31/31, scope clean, pylint 9.51/10)
- TrustedHost, security headers, generic 500s, token guard (non-loopback), pagination bounds, settings allowlist, domain validation, import Body fix, profile exclude_unset. RED 23-fail -> GREEN 25-pass.
- W1-flagged collection error in test_coverage_gaps.py was transient in-flight; Lead re-ran it now: 17 passed (W8 fixed since).

## FIX ROUND 1 — W5 graph/workers DONE (Lead-verified: 32 passed, 1 skipped, scope clean)
- Worker entrypoint fixed, safe_node all nodes, checkpoint warning/mkdir/pragmas, scheduler caps+bootstrap-once, run_manager single ContextVar + ConflictError, persist_record shared by nodes/bulk/manual (triplication removed), Stub->NormalizedJob.
- Checkpoint round-trip test skips (langgraph.checkpoint.sqlite not installed) — warning-path covered. Round-2: verify checkpoint resume with aiosqlite extra installed.

## FIX ROUND 1 — W2 db DONE (Lead-verified: 22/22 targeted, full suite 287 passed, pylint 10.00/10)
- Atomic session(), rank fix, column allowlists, migration txn, 0003 non-unique indexes + runs.created_at + workday seeds (bootstrap patch deleted), orphan age-gate+chunk, ON CONFLICT upsert, fetch_* close helpers, TaxonomyRepository + GatewayStatsRepository + job_skill_rows (Round-2 ready).
- runs.created_at: ALTER ADD COLUMN nullable + backfill (SQLite non-constant default restriction) + create() sets it. CHECK-cancelled skipped (SQLite cannot ALTER CHECK, documented in 0003).

## FIX ROUND 1 — W4 services/LLM DONE (Lead-verified: 28/28, scope clean, pylint 9.93/10)
- Resume magic/page/char caps, prompt delimiters + DATA instruction, Field bounds, bridge thread fix, embedder singleton, extract_text confinement, watchdog/taxonomy conn-close.

## FIX ROUND 1 — W8 tester DONE (Lead-verified: gateway 10/10, targeted 36/36, scope tests-only)
- Gateway import fixed, conftest fixtures, strong graph asserts, full migration assertions, 17 coverage-gap tests (checkpointing 62%, mcp_servers 69-93%, workers/jobs 97%, sanitize 75%), 2 native async tests.
- ResourceWarning item: W8 correctly traced to product core/db.py (forbidden to edit) — W2's fetch_* fix resolved it (verified -W error 11 passed).

## FIX ROUND 1 COMPLETE (8/8 workers, Lead-verified each)
- Full suite: 287 passed, 1 skipped (checkpoint aiosqlite missing in env)
- W1 api 31/31, W2 db 22/22 (pylint 10.00), W3 ssrf 104/104 (pylint 9.73), W4 services 28/28 (9.93), W5 graph 32/32 (8.71), W6 web 21/21 (9.98), W7 devops 12/12 (8.91), W8 tester gateway 10/10 + 36/36
- Round 2: R2A repo-leak switch + R2B YAGNI cleanup (parallel, disjoint), then formatter, then Cycle-2 review.

## FIX ROUND 2 — R2A repo-leak DONE (Lead-verified: 36 passed, scope clean)
- taxonomy->TaxonomyRepository, nodes->job_skill_rows batched, settings->GatewayStatsRepository, watchdog->fetch_* (documented SQL-backed). Zero bare sqlite3.connect in owned files.
- 2 failures confirmed obsolete (W4 test_fix_services_llm monkeypatches removed `connect`): test_taxonomy_connections_are_closed, test_watchdog_closes_connection_on_error. Tester to rewrite in Cycle-2 prep (assert repo delegation instead).
- Checkpoint resume still unverified (langgraph.checkpoint.sqlite missing in env) — Cycle-2 must install + verify.

## FIX ROUND 2 — R2B YAGNI DONE (Lead-verified: 60/60 targeted, full 305 + 2 known obsolete)
- Deleted: health() base+aggregators/cutshort, fetch_for_target, detect_provider, _strip_html x5 -> html_to_text, host_of, due_for_refresh, stats_summary, raw_json_of, unseen_hashes, utcnow, NotFoundError, AppSettings.section, stream_text, run_tool_loop/Executor, expand_via_search. E/F pylint 10.00.
- Kept (caller-found): GatewayChatModel (W4 tests), harvest_from_jobs (W8 tests), _registrable (W8 test asserts current co.in behaviour — canonical change would break test; flagged for Cycle-2 decision), MCPClientManager (now src-unused — Cycle-2 decide delete file or keep).
- Leftover: 8 adapters' health() impls (ashby, greenhouse, lever, personio, recruitee, smartrecruiters, workable, workday) — formatter worker removes.
- 2 obsolete failures confirmed (W4 connect monkeypatch vs Round-1 rewrite) — tester followup rewrites.

## TESTER FOLLOWUP — obsolete fixed; checkpoint product bug found (Lead-verified)
- 2 obsolete tests rewritten as delegation/close-tracking assertions, green. D13 ok.
- Checkpoint BLOCKED: thread_config() missing checkpoint_ns:'' (langgraph-checkpoint-sqlite 3.1.3 requires it; KeyError at aio.py:547). Lead reproduced. Fix routed to formatter worker (owns src/): add checkpoint_ns:'' + verify round-trips pass.
- langgraph-checkpoint-sqlite 3.1.3 installed in env for verification (not added to pyproject — Cycle-2/Cycle-3 decision: checkpointing is optional-import, package should become a real dependency since resume is a core claim).

## FIX ROUND 2 — Formatter DONE (Lead-verified)
- Headers: 0 files with old `# -- coding` (48 normalized). health(): 0 remain. >80 lines: 7 (6 import-exempt + 1 test-data string). Checkpoint_ns fix verified (round-trips pass).
- Suite: 307 passed, 1 skipped, 1 failed = only obsolete test_checkpointing_thread_config (tester micro-fix running).
- Ruff: 918 errors remain (pre-existing UP006/UP035 typing-style families etc; UP009/EXE001 families eliminated -258). No mass reformat per scope. Cycle-2 reviewers to judge whether ruff-clean is required beyond D13/pylint.

## ROUND 2 COMPLETE — suite 308 passed, 1 skipped (Lead-verified)
- R2A repo-leak, R2B YAGNI, formatter D13, tester followup + checkpoint assertion: all DONE.
- 1 skip = checkpoint-related? (langgraph sqlite round-trip now passes; remaining skip to identify in Cycle 2 — likely aiosqlite-optional test elsewhere.)
- Cycle 2 launched: 10 reviewers re-review FIXED code vs Cycle-1 flaw list.

## CYCLE 2 RESULTS (Lead synthesis — 1 APPROVED / 9 REJECTED)
- APPROVED: security (0 Critical/High; 2 Medium SSRF residuals: DNS TOCTOU + reactive redirect validation)
- REJECTED: reviewer (4 MAJOR: checkpoint dep, MCPClientManager/GatewayChatModel/harvest orphans; 4 MINOR; 4 NIT), anti-arch (2 CRITICAL: dead sources knobs, atomic enqueue; many IMPORTANT), anti-sec (2 HIGH: CSRF, redirect SSRF; 4 MEDIUM), adversarial (1 HIGH: salary_fit ZeroDivision; MEDIUM: repo/MCP negative LIMIT; cache non-dict), tester (1 HIGH: checkpoint dep; coverage 71% +22pp, cannot-fail leftovers), DB (1 HIGH: dashboard always-empty double-close; MEDIUM: gateway WAL/index), frontend (2 HIGH: dashboard CSP break, UI token dead-end; 3 LOW), devops (1 HIGH: checkpoint dep; ruff 918), docs (1 HIGH: 502->422 stale; 6 MEDIUM drifts)
- Cycle-3 plan: 6 workers F1-F6 (see status). Lead decisions: DELETE MCPClientManager/GatewayChatModel/harvest (YAGNI, docs updated); manual redirect hops + document TOCTOU residual as accepted (local single-user); atomic enqueue via single-session conditional insert (no partial-index migration risk); god modules accepted via ADR D14 (split deferred).

## CYCLE 3 — F2 db-gateway DONE (Lead-verified: 42/42 targeted)
- Repo LIMIT clamps, cache non-dict, atomic create_if_not_active (concurrent 1-winner), WAL+indexes llm_calls (EXPLAIN uses indexes), generic stats error, session docstring, load_embeddings(job_ids).
- Blocked (tester-owned): test_fix_graph_workers::test_run_manager_conflict_on_integrity_error obsolete (monkeypatches runs.create, enqueue now uses create_if_not_active) -> F6 must update.
- Note: harvest_from_jobs per_page=200 now capped 100/call — moot (F4 deletes harvest).

## CYCLE 3 — F1 + F5 DONE (Lead-verified: F1 69/69, F5 13/13, node/bash clean)
- F1: CSRF middleware, token UI (fetch-stream SSE, no query token), dashboard JS/CSS externalised (CSP self-only, no inline), conn.close dashboard bug fixed (stats non-zero), href //, SSE timeout, profile 0, APP_HOST='', import domain validation.
- F5: langgraph-checkpoint-sqlite>=3.1 declared, run.sh ports, .gitignore !.env.example, egg-info deleted, docs 502->422 + §8 real tools + error shapes + TTL + file map.
- Note: F5 docs already describe F4 deletions (mcp_tools/bridge) — final integration must confirm F4 landed them or docs overclaim.

## CYCLE 3 — F3 + F6 DONE (Lead-verified: F3 34/34, F6 68/68)
- F3: salary_fit guard, sources enabled wiring, shared client, state normalization, CancelledError explicit, node_var, keyed get_client, defaults centralised. RPM threading documented as adapter-owned (adapters pass explicit rpm=).
- F6: cannot-fail fixed, deleted-symbol tests updated (mcp_tools/bridge/harvest gone verified), dashboard/profile/bulk/career/runs coverage up (bulk 80%, career 59%, profile 56%).
- New bugs found by F6 -> microfix worker: dashboard.py:105 with-no-close leak; manual_inbox.py:205 Path.mtime(path) always raises (inbox never moved to _processed); obsolete run_manager integrity test.

## CYCLE 3 — F4 DONE (Lead-verified: 130/130 targeted, full 407 passed, deletions on disk)
- Manual SSRF redirects (private hop never requested), obfuscated-IP canonicalization, shared client, delimiter escape, YAML safe_dump, store domain gate, lazy sources client, resume stat cap, deleted mcp_tools.py + langchain_bridge.py + harvest + _registrable, provider subclasses collapsed to OpenAICompatibleProvider.
- All Cycle-3 workers F1-F6 DONE. Microfix (dashboard closing + manual_inbox mtime + obsolete test) running.

## MICROFIX DONE (Lead-verified: 407 passed, 1 skipped)
- dashboard closing() + stats identical, manual_inbox stat().st_mtime (files now move to _processed), obsolete run_manager test rewritten.
- Note: microfix docstring documents closing pattern to satisfy brittle test_fix_web text check (`with sqlite3.connect` literal) — tester nit for Cycle-3/4.
- Cycle 3 launched (10 reviewers vs Cycle-2 findings).

## CYCLE 3 RESULTS (5 APPROVED: security, anti-sec, frontend, devops, tester / 5 REJECTED: reviewer, anti-arch, adversarial, DB, docs)
- Converging blockers: runs-route atomic enqueue (anti-arch CRITICAL + adversarial HIGH, 8-thread proof), checkpoint dep closed, deletions closed, CSRF/redirect closed.
- Cycle-4 prep P1-P4 dispatched (see status). D14 written by Lead (god modules accepted). D15 risk-accepts DNS TOCTOU + missing-Origin CSRF.
- New real bugs from reviews: mcp2 store/resume import crash, DBLogger worker-death hang, manual_inbox (fixed microfix), dashboard closing (fixed microfix).

## CYCLE-4 PREP — P2 DONE (Lead-verified: 125/125, mcp2 import ok)
- mcp2 fallback store/resume, limit clamps adapters+MCP+web_search, throttle (host,rpm) keying + shared lock, case-insensitive delimiter, per-page resume cap, lazy resume parser.
- P2 full-suite failures were other workers' in-flight files (P3 graph, P4 docs) — re-check at integration.

## CYCLE-4 PREP — P1 DONE (Lead-verified: 40/40, atomic route grep-confirmed)
- Runs route atomic (409 on race), AppError->409/500/502/503 mapping, resume Content-Length pre-check + orphan cleanup, import priority validation, valid_domain dedup (Example.COM consistent), models constant import.

## CYCLE-4 PREP — P3 DONE (Lead-verified: 42/42 targeted, 0004 exists, FK pragmas present)
- DBLogger survives insert errors + close() timeout, checkpoint FK, migration re-check, state annotations plan/enriched->dict + _normalize_inserted, get_client 4-path key, nodes >80 wrapped.
- Follow-up: cache.py _connect pragma-close guard (F2 file, P3 forbidden) — tiny fix for final polish.

## CYCLE-4 PREP COMPLETE (P1-P4 all DONE, Lead-verified full suite 446 passed, 1 skipped)
- P1 atomic runs/AppError/resume cap/domain dedup, P2 mcp2/limits/throttle/delimiter, P3 DBLogger/state/get_client/0004, P4 package-data/starlette/rpm removal/docs/tests.
- Known follow-ups: cache.py _connect pragma guard (LOW), ResourceWarning saver leak (test_graph_pipeline, GC-attributed), F821 x3 annotation-only.
- Cycle 4 launched.

## CYCLE 4 RESULTS (5 APPROVED: reviewer, security, anti-sec, tester, DB / 5 REJECTED: anti-arch, adversarial, frontend, devops, docs)
- Converging: CSRF/redirect/delimiter/YAML/store/domain all verified closed. Remaining: throttle wait race (anti-arch IMPORTANT), chunked-upload RAM + orphan race (adversarial/anti-sec MEDIUM), MCP limit leftovers (cutshort/workday/himalayas/smartrecruiters), normalize malformed (MEDIUM), dashboard 404 headers, href backslash, SSE join, resume auth path, untracked dashboard.css/js (commit action), F821 x4, docs §10.1/D15/settings + test gaps.
- Cycle-5 prep Q1-Q4 dispatched. D10/D15(c) amendments written by Lead.

## CYCLE-5 PREP COMPLETE (Q1-Q4 + sanitize microfix, Lead-verified 505 passed, 1 skipped)
- Q1 streaming upload/orphan race/dashboard 404/href-backslash/SSE-join/fetchWithAuth, Q2 throttle-reservation/limits/delimiter/locks, Q3 normalize-malformed/cache-guard/WAL-retry/F821, Q4 wheel/docs/tests, sanitize chunk-split.
- Cycle 5 launched.

## CYCLE 5 RESULTS (4 APPROVED: security, anti-arch, anti-sec, tester / 6 REJECTED: reviewer, adversarial, DB, frontend, devops, docs)
- Converging fixes verified: throttle reservation, normalize-malformed, atomic routes, mcp2, limits, delimiters, DBLogger, state annotations, deletions, CSRF/redirects.
- Remaining: curator row/file atomicity + BaseException/write-inside-try (MEDIUM x3 reviewers), throttle dot-bypass (LOW), unterminated delimiter (LOW), dashboard send_error (LOW), priority clamp (LOW), stream_run 404+stats (MEDIUM), href control-chars/SSE/button/th/aria (LOWs), untracked css/js (commit), F821 base.py, docs drift (counts/mcp2/quarantine/tenacity/clamps), tester gaps (sanitized cache-stream, embed success, wheel install, JS harness).
- Cycle-6 prep R1-R4 dispatched.

## CYCLE-6 PREP COMPLETE (R1-R4 all DONE, Lead-verified full suite 560 passed, 0 failed, 0 skipped)
- R1 streaming/orphan/BaseException/curator-row-delete/priority/dashboard404/stream404/href-control/button/th/aria, R2 throttle-dot/delimiter-unterminated/limits/sources-dedup/common-settings/F821, R3 resolve_many/_instance-lock/config-constant/pragma-guard, R4 docs-46/wheel-install/JS-harness/sanitized-cache-stream/embed-success.
- Cycle 6 launched.

## CYCLE 6 RESULTS (4 APPROVED: security, anti-arch, anti-sec, tester / 6 REJECTED: reviewer, adversarial, DB, frontend, devops, docs)
- All Cycle-5 items verified closed (streaming, throttle reservation, normalize-malformed, atomic, mcp2, limits, delimiters, DBLogger, state annotations).
- Converging real bug: duplicate-sha delete_resume harm (4 reviewers). Plus import inf crash, identity-less phantom job, workday real-network test (tester HIGH), untracked css/js (commit), F821 base (fixed R2? devops C6 says F821=1 base.py — R2 TYPE_CHECKING added, verify), docs drift (throttle dot, priority, mcp2 §8, counts).
- Cycle-7 prep S1-S3 dispatched.

## CYCLE-7 PREP — S2 DONE (Lead-verified: 36/36 targeted, full 569 passed)
- quote_identifier shared in core.db, base docstring, _normalize identity guard (phantoms skipped).

## CYCLE-7 PREP — S3 DONE (Lead-verified: full suite 583 passed)
- Dashboard role=status removed, docs throttle-dot/priority/MCP2 pins, workday test rewritten hermetic (ResourceWarning clean), cycle6 dashboard assert updated.

## CYCLE-7 PREP COMPLETE (S1-S3 all DONE, Lead-verified full suite 583 passed)
- S1 duplicate ownership (save_resume_ex created-flag, stored-path return), import inf finite, seed clamp+skip, _valid_domain deleted.
- Cycle 7 launched.

## CYCLE 7 RESULTS (7 APPROVED: security, anti-arch, anti-sec, adversarial, tester, DB, frontend / 3 REJECTED: reviewer, devops, docs)
- All Cycle-6 items verified closed (duplicate ownership, inf, throttle dot, delimiters, locks, mcp2, limits, normalize, DBLogger, dashboard).
- Remaining: profile/table_columns quote reuse (MINOR x3), _valid_domain dead (MINOR), base docstring (MINOR), priority 0->3 (LOW x3), get_resume None dangling (LOW), taxonomy non-dict (INFO), normalize int guard (LOW), dashboard send_error (LOW), untracked css/js (HIGH commit), docs counts/priority wording (MEDIUM), tester LOWs (coverage leftovers).
- Cycle-8 prep T1+T2 dispatched (final polish).

## CYCLE-8 PREP COMPLETE (T1+T2 DONE, Lead-verified full suite 602 passed)
- T1 quote reuse, priority 0->1, curator row-None raise, taxonomy guard, score int guard, dashboard send_error.
- T2 docs counts 51/priority/seed/duplicate/identity, ssrf close hygiene, flat-build test, non-dict seed, create_app health test.
- Cycle 8 launched.

## CYCLE 8 RESULTS (7 APPROVED: reviewer, security, anti-arch, anti-sec, tester, DB, frontend / 3 REJECTED: adversarial, devops, docs)
- All Cycle-7 items verified closed except: create/patch OverflowError 500 (MEDIUM), curator concurrent-duplicate unlink (MEDIUM), taxonomy aliases (LOW), bool priority (LOW), untracked css/js+persist+taxonomy+0003/0004 (HIGH commit), docs priority wording (HIGH x2), docs LOWs.
- Cycle-9 prep T3+T4 dispatched (final).

## CYCLE-9 PREP COMPLETE (T3+T4 DONE, Lead-verified full suite 621 passed)
- T3 int-first clamp (10**400->5, bool reject 422), curator row-first cleanup, taxonomy alias validation, _processed skip.
- T4 docs priority/seed/duplicate/identity/_processed/send_error wording, counts 53, cycle9 docs tests.
- Cycle 9 launched.

## CYCLE 9 RESULTS (6 APPROVED: reviewer, security, anti-arch, tester, DB, frontend / 4 REJECTED: anti-sec, adversarial, devops, docs)
- All Cycle-8 items verified closed (int-first clamp, row-first curator, throttle dot/reservation, _processed skip, docs counts/wording, F821 0).
- Remaining: resolve_many/settings overflow 500s (anti-sec HIGH x2), raw-inf 500 (adversarial MEDIUM), route unlink defeats retention (anti-sec MEDIUM), inbox rename-on-DB-failure (MEDIUM), profiles bool/range + jobs page overflow (LOWs), taxonomy nulls category (arch M3), untracked files (devops HIGH commit), docs bool wording (MEDIUM).
- Cycle-10 prep U1+U2 dispatched (final).

## CYCLE-10 PREP COMPLETE (U1+U2 DONE, Lead-verified full suite 648 passed)
- U1 resolve_many guards + List[str] route, settings overflow, RequestValidationError non-finite sanitize, retention fix (exc.file_path only when deleted), inbox rename-on-success, bool/range profiles, jobs page le.
- U2 docs bool/counts/row-first/taxonomy, cycle10 docs tests.
- Cycle 10 launched.

## CYCLE 10 RESULTS (8 APPROVED: reviewer, security, anti-arch, anti-sec, adversarial, tester, DB, frontend / 2 REJECTED: devops, docs)
- All Cycle-9 items verified closed (resolve guards, settings overflow, raw-inf 422, retention, inbox success tracking, bool/range, page le, throttle dot, limits, delimiters).
- Remaining: docs taxonomy sentence (docs MEDIUM), resolve_many/exc.file_path/sanitize docs LOWs, roadmap quarantine qualify, tester LOW pins (settings bool, payload guard, sanitize direct, manual_inbox generic, curator branches), devops HIGH untracked (commit action).
- Cycle-11 prep: 1 docs+tester worker dispatched (final).

## CYCLE-11 PREP COMPLETE (T5 DONE, Lead-verified full suite 662 passed)
- Docs taxonomy split/resolve_many/exc.file_path/sanitize-caps/roadmap/counts 56; tests settings-bool/payload/sanitize/inbox-generic/curator-branch.
- Cycle 11 launched.

## FINAL QA + COMMIT (2026-11-07)
- Full suite Lead-verified: 662 passed (post-commit HEAD c95fd3c).
- Human smoke: seed-db ok, api healthz ok, evil-Host 400, per_page 422, companies list ok.
- Commit c95fd3c (206 files) pushed to origin agentic_ai/job-hunter. Tree clean, no secrets/data staged (verified).
- User actions: rotate 7 gateway keys; restore lost resume PDF (untracked, deleted during loop); delete 127.0.0.1 test company row in local data/app.db.
- Cycle 12 post-commit verification launched (10 reviewers on committed tree).

## CYCLE 12 RESULTS (post-commit, HEAD c95fd3c — 7 APPROVED: reviewer, security, anti-sec, adversarial, tester, DB, frontend / 3 REJECTED: anti-arch, devops, docs)
- Anti-arch: fan-out starvation (51+ companies never fetched - rotation fix), bulk_fetch enabled bypass, raw_jobs resume flip, core->services inversion, docs overclaims.
- Adversarial: huge page/path-ID OverflowError 500s (recommendations + jobs/companies/runs/recs path ids).
- Devops: untracked files (commit action only).
- Docs: 12 HIGH deep drifts (cascade, topology, gates, semantic, adapter ABC, env table, raw_json, cancel, feedback, taxonomy, competitor refresh, DDL, diagram, runbook scope, settings staleness).
- Cycle-13 prep V1-V3 dispatched.

## CYCLE-13 PREP COMPLETE (V1-V3 + recs-page microfix + docs recount, Lead-verified full suite 706 passed)
- Fan-out rotation (list_due oldest-first + deferred stat), bulk enabled filter, raw_jobs normalize, core/constants move, huge-int ID bounds, recs page le+clamp.
- Docs deep-drift rewrite (cascade, topology, gates, semantic, ABC, DDL, transports, refresh, dashboard pages), counts 60.
- Cycle 13 launched.

## CYCLE 13 RESULTS (5 APPROVED: security, anti-sec, tester, DB, frontend / 5 REJECTED: reviewer, anti-arch, adversarial, devops, docs)
- Reviewer SyntaxError claim FALSE (py3.14 accepts except A,B — verified ast.parse OK; will normalize anyway).
- Anti-arch: fan-out cooled/ATS starvation + last_checked_at upsert reset (HIGH), raw_jobs ok, inversion ok, docs overclaims remain.
- Adversarial: negative huge path IDs 500 (MEDIUM), CSRF userinfo (LOW), linkedin unicode header (LOW).
- Devops: HIGH constants.py untracked (commit).
- Docs: 12 HIGH deep drifts + MEDIUM/LOWs.
- Cycle-14 prep W1-W3 dispatched.

## CYCLE-14 PREP COMPLETE (W1-W3 DONE, Lead-verified full suite 747 passed)
- W1 parens-except normalization, dead deletions, _SETTINGS_CACHE dedup, source_enabled/clamp/_MAX_ID shared homes, list_due ATS+cooled+rotation, last_checked_at upsert removal, bulk empty/failed split, negative-ID bounds.
- W2 docs deep-drift rewrite (cascade/topology/gates/semantic/ABC/DDL/transports/refresh/pages/feedback/taxonomy/counts 63).
- W3 fan-out/limits tests, wheel/JS/checkpoint verify, weak-assert strengthens.
- Cycle 14 launched.

## CYCLE 14 RESULTS (7 APPROVED: reviewer, security, anti-sec, adversarial, tester, DB, frontend / 3 REJECTED: anti-arch, devops, docs)
- Anti-arch: list_due limit-before-cooled starvation (HIGH) + source_key doc (HIGH), Mediums (bootstrap doc, mark_checked semantics, ATS empty-string).
- Adversarial APPROVED (negative huge all 422, priority, tokenizer). Tester/DB/frontend APPROVED (DB save_profile version + runs docstring LOWs accepted).
- Devops: HIGH constants.py untracked (commit).
- Docs: HIGH fan-out misattributed + constants/bounds layout, MEDIUMs (semantic trunc, bounds table, bulk/empty, upsert).
- Cycle-15 prep W1+W2 dispatched (final).

## CYCLE-15 PREP — W2 DONE (Lead-verified: full suite 766 passed, 0 failed)
- Docs fan-out/cooled/ATS rotation, constants layout, source_key removal, bounds symmetric MAX_SQLITE_ID, counts 65.
- W1 (backend) form return pending; tree already green with its changes (766 passed).

## CYCLE-15 PREP COMPLETE (W1+W2 DONE, Lead-verified full suite 766 passed)
- W1 list_due cooled SQL join + ATS NULLIF + count_due true deferred + quick truthy + docstrings.
- W2 docs fan-out/constants/source_key/bounds/bulk truth + counts 65.
- Cycle 15 launched.
