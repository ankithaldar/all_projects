# Agent: test-writer

## Task
Own regression coverage for every fix; keep the suite fast (<10s).

## Rules
1. New tests go into the existing unit module covering that area
   (the live list is every module in `tests/unit/` — 66 unit test modules
   (60+; counted dynamically as `tests/unit/test_*.py`, not a hardcoded
   literal; last counted 2026-11-07, includes
   `test_fix_cycle5_docs.py`, `test_fix_cycle6_docs.py`,
   `test_fix_cycle11_docs.py` and
   `test_js_harness.py`), plus any new contract modules added since —
   this mapping is non-exhaustive, the live directory wins):
   - runs/claiming → `test_graph_pipeline.py`
   - API routes → `test_api.py`
   - adapters/parsing → `test_adapter_parsing.py`
   - scoring/gates → `test_scoring.py`
   - scheduler/cron → `test_scheduler_helpers.py`
   - company discovery → `test_company_discovery.py`
   - scout/linkedin → `test_scout.py`
   - manual inbox → `test_manual_inbox.py`
   - migrations → `test_migrations.py`
   - profile services → `test_profile_services.py`
   - embeddings → `test_embeddings.py`
   - watchdog → `test_watchdog.py`
   - async smoke → `test_async_smoke.py`
   - coverage gaps → `test_coverage_gaps.py`
   - API security → `test_fix_api_security.py`
   - DB fixes → `test_fix_db.py`
   - devops/config/docs contracts → `test_fix_devops.py`
   - graph/worker fixes → `test_fix_graph_workers.py`
   - repo-switch fixes → `test_fix_repo_switch.py`
   - services/LLM fixes → `test_fix_services_llm.py`
   - SSRF fixes → `test_fix_ssrf.py`
   - web fixes → `test_fix_web.py`
   - YAGNI fixes → `test_fix_yagni.py`
   - cycle-3 devops/docs contracts → `test_fix_cycle3_devops_docs.py`
   - coverage gaps 2 → `test_coverage_gaps2.py`
   - profile curator gaps → `test_profile_curator_gaps.py`
   - gateway dashboard → `test_gateway_dashboard.py`
   - JS harness (node api.js) → `test_js_harness.py`
   - cycle-3 adapters/MCP → `test_fix_cycle3_adapters_mcp.py`
   - cycle-3 API/web → `test_fix_cycle3_api_web.py`
   - cycle-3 DB → `test_fix_cycle3_db.py`
   - cycle-3 graph → `test_fix_cycle3_graph.py`
   - cycle-4 adapters → `test_fix_cycle4_adapters.py`
   - cycle-4 API → `test_fix_cycle4_api.py`
   - cycle-4 docs/tests → `test_fix_cycle4_docs_tests.py`
   - cycle-4 graph/DB → `test_fix_cycle4_graph_db.py`
   - cycle-4 microfixes → `test_fix_cycle4_microfix.py`
   - cycle-5 adapters → `test_fix_cycle5_adapters.py`
   - cycle-5 API/web → `test_fix_cycle5_api_web.py`
   - cycle-5 docs → `test_fix_cycle5_docs.py`
   - cycle-5 graph/DB → `test_fix_cycle5_graph_db.py`
   - cycle-5 sanitize → `test_fix_cycle5_sanitize.py`
   - cycle-6 API/web → `test_fix_cycle6_api.py`
   - cycle-6 docs → `test_fix_cycle6_docs.py`
   - cycle-6 graph/DB → `test_fix_cycle6_graph_db.py`
   - cycle-6 adapters → `test_fix_cycle6_adapters.py`
   - cycle-7 curator → `test_fix_cycle7_curator.py`
   - cycle-7 docs → `test_fix_cycle7_docs.py`
   - cycle-7 quote/state → `test_fix_cycle7_quote_state.py`
   - cycle-8 backend → `test_fix_cycle8_backend.py`
   - cycle-8 docs → `test_fix_cycle8_docs.py`
   - cycle-9 backend → `test_fix_cycle9_backend.py`
   - cycle-9 docs → `test_fix_cycle9_docs.py`
   - cycle-10 backend → `test_fix_cycle10_backend.py`
   - cycle-10 docs → `test_fix_cycle10_docs.py`
   - cycle-11 docs → `test_fix_cycle11_docs.py`
   - cycle-13 backend → `test_fix_cycle13_backend.py`
   - cycle-13 tester → `test_fix_cycle13_tester.py`
   - cycle-13 docs → `test_fix_cycle13_docs.py`
   - cycle-13 recs page → `test_fix_cycle13_recs_page.py`
   - cycle-14 docs → `test_fix_cycle14_docs.py`
   - cycle-15 backend → `test_fix_cycle15_backend.py`
   - cycle-15 docs → `test_fix_cycle15_docs.py`
   - cycle-17 docs → `test_fix_cycle17_docs.py`
   When in doubt, list `tests/unit/` and add the test to the module
   whose subject it covers rather than creating a new file.
2. Every test isolates storage: tmp_path db + `APP_DATA_DIR` env via
   monkeypatch (never raw os.environ writes).
3. Network is banned; stub adapters/providers at the registry boundary.
4. Run `PYTHONPATH=src/job_hunter pytest tests/unit -q` and report the
   exact count; a fix without its regression test is not done.
