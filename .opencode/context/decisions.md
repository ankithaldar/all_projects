# ADR — job_hunter review loop (2026-11-07, Lead tie-breaks)
## D1 Auth model (anti-sec Critical)
Local single-user. Fix: TrustedHost allowlist [127.0.0.1, localhost, testserver] + security headers + generic 500s. Non-loopback APP_HOST requires JH_API_TOKEN (Bearer, checked in api/deps.py) else fail-fast in config. No OAuth/session (out of scope local-only).

## D2 MCP architecture (anti-arch Critical)
Keep 3 MCP servers (documented feature, used by company_discovery meta path + LLM tool loop). Nodes using repos directly is ACCEPTED drift: update architecture.md §7/§8 to state graph uses repos, MCP is LLM/discovery tool surface. Fix store_server tool list docs, add SSRF/path guards to tools, structured error envelopes. Do NOT delete MCP, do NOT rewire graph through MCP this loop.

## D3 Gateway god app.py (1498 lines)
No refactor this loop (README: untouched, regression risk). Fix only: cache key/corrupt handling, sanitize ReDoS, dashboard XSS, undeclared `requests` dep. ADR accepted duplication gateway vs app (2 ConfigError, 2 DB layers).

## D4 UNIQUE canonical_url
0003 migration creates NON-unique indexes (canonical_url, candidate_id+version, companies status/vertical, llm_calls timestamp/provider/session). UNIQUE deferred: existing DBs may hold dupes; content_hash UNIQUE already guards dedupe. Document.

## D5 Deps
Remove unused `tenacity`, `pydantic-settings` from pyproject+requirements (0 imports). Add `requests>=2.31` (critical ImportError). Delete requirements.txt duplicate (pyproject canonical; grep/update references in README/setup). Add lockfile note to docs, no lockfile generation this loop (CI-less local app).

## D6 Scheduler
Keep BlockingScheduler (AsyncIO migration risk > benefit local). Add max_instances=1, coalesce=True, misfire_grace_time to all add_job, bootstrap-once (cache AppSettings in scheduler), fix mark_orphans age-gate for pending, partial UNIQUE index on runs active (pending/running) for atomic enqueue. Update arch §19 (no maintenance job, no TZ, Blocking not AsyncIO).

## D7 Worker entrypoint + safe_node + checkpoint
worker/__main__.py passes default config path. Implement safe_node wrapper in graph/nodes.py + apply to all nodes. Checkpoint: log warning on missing aiosqlite (not silent), mkdir parents, WAL/busy pragmas. Full Pydantic state refactor deferred; normalize budget/enriched/errors shapes minimally + resume test.

## D8 LangChain bridge
Fix running-loop bug via dedicated thread + new event loop (never run_until_complete on running loop). Keep file (integration point).

## D9 SSRF guard
Central validate_public_url() in adapters/http_client.py: https-only, literal+resolved IP block (loopback/private/link-local/reserved/metadata), redirect re-validation, 5MB streaming cap. Resolver injectable for tests. board_ref per-provider regex. fetch_page in MCP goes through same guard.

## D10 Repo leaks (staged)
Round 1: W2 adds repo methods (job_skill_rows, watchdog queries, taxonomy queries, gateway stats) + closes all connections. Round 2: W4/W5 switch services/nodes/settings route to repo methods. Round 1 services/nodes only close conns.

## D11 .env keys
Code fix: .gitignore .env*/egg-info. User action: ROTATE 7 live keys (Lead verified never tracked in git). Document in final report.

## D12 Root .gitignore .opencode/ change
Unexpected modification found during review (whitespace + .opencode/). At commit: keep .opencode docs tracked via `git add -f` for specific files OR remove ignore entry. Decision: remove `.opencode/` ignore lines (AGENTS.md shared files are intended persistent), keep whitespace revert.

## D13 Strict formatting (user directive 2026-11-07, binding on all workers + Round-2 formatter)
- SOLID + Google-style docstrings on all modules/classes/functions/methods in new/modified code.
- 2-space indent (no tabs/4-space), single quotes for all string literals (exception: strings containing a single quote may use double quotes to avoid escapes).
- Follow repo-root `.pylintrc` (agentic_ai_main/.pylintrc, Google style: indent '  ', max-line-length 80, check-quote-consistency). DO NOT modify .pylintrc.
- Every Python file begins exactly:
  ```
  #!/usr/bin/env python
  # -*- coding: utf-8 -*-
  <blank line>
  '''<module docstring>'''
  ```
  Current repo uses `# -- coding: utf-8 --` in all 126 files (0 with `# -*-` form). Round-2 formatter normalizes to `# -*-` form repo-wide + verifies blank line + ''' docstring.
- Line length: pylintrc 80 vs ruff/pyproject 100 conflict. New/modified code MUST be <=80. Existing >80 lines (402/19978 at directive time) grandfathered unless touched; Round-2 formatter wraps touched files to <=80 and reports remaining count. Ruff line-length stays 100 (W7) to avoid mass reformat churn; pylint is the 80-col gate.
- Baseline at directive: 0 tab-indented files, 123/126 files have ''' module docstring (3 new worker files missing: services/persist.py, test_fix_graph_workers.py, test_fix_services_llm.py — workers to fix in follow-up).

## D14 God modules (Cycle-3 gate, Lead ADR 2026-11-07)
`llm_gateway/app.py` (~1508 lines, D3 deferral) and `graph/nodes.py` (~921 lines, 8 nodes + helpers) are ACCEPTED as monoliths for this loop. Rationale: safe_node + RED/GREEN tests cover the seams; splitting nodes per-stage or gateway orchestration mid-loop carries regression risk > benefit for a local single-user app. Split trigger: next feature touching 3+ nodes, or nodes.py >1200 lines, or gateway app.py change beyond registry/cache (then split per-stage modules + step helpers first). Accepted debt recorded; Cycle-3 reviewers downgraded accordingly.

## D15 DNS-TOCTOU + CSRF missing-Origin residuals (risk-accepted)
Security (Cycle-3, APPROVED) + anti-sec (APPROVED): (a) DNS rebinding TOCTOU (validate-then-fetch re-resolves) accepted — https-only + TLS cert verification + 5MB cap bound impact in personal loopback profile; proper fix needs IP-pinned transport. (b) CSRF middleware allows requests with BOTH Origin+Referer absent (curl/non-browser) — browsers always send Origin on POST, Bearer token blocks token mode; documented intent, tester pins with a test. (c) Shared HttpClient throttle pinning by first rpm — document in code.

## D10 amendment (Cycle-4)
Watchdog keeps inline SQL aggregates via fetch_all/fetch_one (code-comment rationale in services/watchdog.py:24-29); TaxonomyRepository covers reads, load_seed writes stay in service via session(). Both accepted — D10 repo-method migration closed for these two.

## D15(c) amendment (Cycle-4, supersedes first-rpm pinning)
Throttle keying is per-(host, rpm) (adapters/http_client.py throttle_for). Same host at different rpm values runs independent throttles; aggregate rate can exceed the minimum rpm. Accepted — adapters pass explicit rpm= per source, uniqueness by (host,rpm) is the intended semantics.
