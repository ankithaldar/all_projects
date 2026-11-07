# Anti-Team Architect Critique — job_hunter cycle-1
Verdict: REJECTED (ses_e4b6d6d0dffekZ5Az3v0IdQPXm)
Date: 2026-11-07

CRITICAL:
- worker/__main__.py:9 vs workers/scheduler.py:67 — documented `python -m job_hunter.worker` calls start_scheduler() no-arg -> TypeError. Fix default config path + smoke test.
- graph/nodes.py 651 lines god orchestrator, no safe_node (grep 0 vs arch docs). Split enrich_jds/score_rank_persist, add safe_node wrapper.
- MCP fictional in hot path: nodes bypass MCP, grep MCPClientManager=1 call site, store_server tool list != arch §8. Wire or delete/re-scope claim.
- graph/checkpointing.py:24-29 missing aiosqlite -> yield None silently, crash-resume false. State mixes CandidateProfile object / model_dump dict / TokenBudget obj+dict. Fail loudly, normalize state, checkpoint-resume test.
- Repo pattern leaks: nodes reach jobs_repo._db_path raw SQL, watchdog/skills_taxonomy raw SQL, api/routes/settings.py:89-111 sqlite3.connect to gateway.db bypassing WAL pragmas + conn leak. Add repo methods, forbid connect() outside repos (grep gate).
- db/repositories/recommendations.py:47-64 BROKEN SQL: INSERT lists `rank` TWICE, binds position + row.get('rank'). upsert_many cannot work. Fix single rank, real-DB round-trip test.
- core/config.py:17-161 no pydantic-settings (grep 0, dep declared), scoring_weights/sources/scout/cron untyped, defaults duplicated, sources.enabled/rate_limit_rpm never read in graph, PUT /api/settings arbitrary key/value. Typed validated model, allowlist, wire/delete dead knobs.
- workers/scheduler.py:77,137-144 BlockingScheduler single-thread long discovery blocks poll/sweep, no max_instances/coalesce/misfire, bootstrap() every job, has_active TOCTOU, mark_orphans_failed `(running AND old) OR pending` kills all pending. AsyncIO/threadpool + caps, bootstrap once, atomic enqueue, age-gate pending.
- llm/client.py gateway coupling: sys.path hacks, GatewayClient.gateway raw escape, get_client singleton ignores settings, langchain_bridge `running.run_until_complete` on running loop -> RuntimeError. Remove escape, key singleton, sync-from-async via new thread, real packaging.
- adapters/registry.py:26-42 AggregatorAdapter KeyError not AdapterError, OCP violation, fetch_pair new HttpClient per Send breaks per-host throttle, rpm hardcoded 10, manual/career_page unregistered. Shared HttpClient, AdapterError, register/exclude.

IMPORTANT:
- core/errors.py decorative: NotFound/Conflict/Database never raised, RunManager bare RuntimeError, API maps no AppError, ~65 bare except pass. Map at API, ban bare except.
- Layer inversion systemic: API->DB direct all routes, services take db_path, manual_inbox imports repos. Thin service layer, adapters DB-free, repo protocols.
- graph/nodes unbounded Send fan-out no caps, Stub type('Stub') fake NormalizedJob, taxonomy resolve twice/skill, embedder re-instantiated per run.
- embedder get_embedder new FastembedProvider every call (model reload), load_embeddings full scan O(all jobs).
- Migrations drift: 0001 seeds 16 sources not workday, bootstrap.py:40-45 INSERT OR IGNORE patch, arch §16 candidate_profiles != SQL != models, run_migrations non-atomic + readonly connect leak.
- Logging theater: run_id/node contextvars only set in RunManager, nodes never set, no per-run logs, _job_hunter_configured blocks reconfig, many modules no logger.
- State/reducer types lie: errors list[NodeError] but dicts returned, enriched dict[str,EnrichedJob] but plain dicts, budget obj<->dict flip.
- main.py:64-72 hardcodes 127.0.0.1:8088 ignores settings, log_level warning, sys.path hacks double-import risk.
- mcp_servers import-time ensure_sys_path() side effects, tool errors swallowed to strings indistinguishable from results.
- http_client RobotsCache fail-open (True on error), get_json 2 retries fixed 1.5s no jitter, tenacity dep never used, no Retry-After/circuit-breaker.
- run_manager dual ContextVars (current_run_id vs run_id_var), token never reset leak.

Blind spots: llm_gateway internals, web/, tests not run, live behaviour, linkedin_scout/domain_verifier skimmed only.
Verdict REJECTED. Re-review requires all critical fixes with regression tests.
