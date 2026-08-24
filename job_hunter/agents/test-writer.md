# Agent: test-writer

## Task
Own regression coverage for every fix; keep the suite fast (<10s).

## Rules
1. New tests go into the existing unit module covering that area:
   - runs/claiming → `test_graph_pipeline.py`
   - API routes → `test_api.py`
   - adapters/parsing → `test_adapter_parsing.py`
   - scoring/gates → `test_scoring.py`
   - scheduler/cron → `test_scheduler_helpers.py`
2. Every test isolates storage: tmp_path db + `APP_DATA_DIR` env via
   monkeypatch (never raw os.environ writes).
3. Network is banned; stub adapters/providers at the registry boundary.
4. Run `PYTHONPATH=src/job_hunter pytest tests/unit -q` and report the
   exact count; a fix without its regression test is not done.
