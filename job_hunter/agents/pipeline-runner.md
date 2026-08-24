# Agent: pipeline-runner

## Task
Execute the Job Hunter discovery pipeline safely and diagnose failures.

## Steps
1. `python main.py seed-db` — must print 'database seeded'.
2. `python main.py run-discovery` — expect a JSON-ish status line ending
   in success or partial (never a traceback).
3. On failure, read the newest `data/logs/runs/*.log` slice and the
   `run_events` table for the failing node.

## Guardrails
- Never run against production `data/`; set `APP_DATA_DIR` to a temp dir.
- The gateway is read-only: no edits under `src/job_hunter/llm_gateway/`.

## Acceptance
- `pytest tests/unit -q` exits 0 and a discovery run reaches
  `score_rank_persist` without unhandled node errors.
