# Job Hunter — Runbook

Operations for the local API + worker pair. See `docs/api.md` for endpoints
and `docs/architecture.md` §19 for the scheduler contract.

## Start / stop

```bash
./setup.sh          # one-time: venv, deps, gateway .env, seed-db, discover, verify-ats
./run.sh            # api + worker together (waits for /healthz, then prints the URL)
./run.sh api        # API only
./run.sh worker     # worker only
```

- Stop both: `Ctrl-C` in the `run.sh` foreground terminal (the worker is
  killed by the trap; the API exits with it).
- Worker death note: if the worker exits while the API stays up, the run
  queue silently stalls — `run.sh` prints a hint on exit. Check
  `data/logs/app.log`, then restart with `./run.sh worker`.

## Enqueueing runs (409 semantics)

`POST /api/runs/discovery` and `RunManager.enqueue` are both atomic:
`create_if_not_active` checks for a pending/running run and inserts
inside a single `BEGIN IMMEDIATE` session, so concurrent enqueues
produce exactly one run and the loser gets 409. Sequential calls while
one run is pending/running also get 409 (`ConflictError` →
409; see `docs/api.md`). If a 409 persists, a run is stuck — see
Orphan recovery below.

## Checkpoint / resume

Resume from `data/checkpoints.sqlite` is conditional on
`langgraph-checkpoint-sqlite>=3.1` + `aiosqlite>=0.20` being installed
(declared in `pyproject.toml`). Without them the worker logs a warning
and runs without a checkpointer (no resume after crash). Check with
`python -c "import langgraph.checkpoint.sqlite.aio, aiosqlite"`.

## Gateway dashboard (8099)

Served from `src/job_hunter/llm_gateway/templates/`: `/` (dashboard.html),
`/dashboard.css`, `/dashboard.js`, `/api/stats`. Stats read
`data/gateway.db`; a missing DB yields zeroed stats, not a 500.

## Settings / sources: yaml-only

`GET /api/settings` returns seeded read-only copies; `PUT` accepts
only `scoring_weights`, `salary_hard_floor_lpa`,
`discovery.max_llm_calls_per_run`, `discovery.enrich_per_run_cap`.
Schedule and source enabling are yaml-only — edit `config/app.yaml`
(`sources:` is `enabled`-only; per-source RPM is adapter-owned) and
restart. CSRF: non-GET with cross-origin Origin/Referer is rejected
403 with `{'detail': 'forbidden origin'}`; headerless requests not
checked (browser always sends Origin on POST).

## Logs

- App log: `data/logs/app.log` (JSON lines, rotating 10 MB × 5), written by
  both API and worker via the same handler.
- Run events: the `run_events` table (visible at `GET /api/runs/{id}/events`
  and streamed via `/api/runs/{id}/stream`).
- Gateway costs: `data/gateway.db` (per-call tokens/cost via `session_id`).

## Orphan recovery

A run stuck in `running` after a crash/kill is adopted on worker start and
by `recover_orphans` (stale `running` rows older than the TTL are finalized
as failed). The TTLs are:

- **Worker/scheduler:** `running` rows older than **120 min** (default
  `recover_orphans(..., ttl_minutes=120)`).
- **CLI `run-discovery` (`python main.py run-discovery`):** `running` rows
  older than **5 min** (`ttl_minutes=5`), and a guaranteed finalizer so no
  run is left `running` forever.
- **Pending rows:** older than **1h** (one hour) are failed by the same
  sweep, so a freshly enqueued pending run is never failed before it can
  be claimed.

Debug a stuck run:

```bash
python - <<'PY'
import sqlite3
conn = sqlite3.connect('data/app.db')
print(conn.execute("select id,status,started_at from runs where status in ('pending','running')").fetchall())
PY
```

## Zero companies / zero jobs after a fresh install

A fresh install fetches zero jobs until seeds are ingested **and** ATS board
refs are verified. In order:

1. `python main.py seed-db` — must print `database seeded`.
2. `python main.py discover-companies` — ingests `seeds/companies_*.yaml`.
3. `python main.py verify-ats` — populates `ats_provider`/`board_ref`;
   without these, company-driven fetching has nothing to call.
4. `python main.py fetch-jobs` — bulk pull from aggregators (Arbeitnow,
   Jobicy, Cutshort, …).
5. `python main.py scout-companies` — discover more companies, then
   re-run `verify-ats` so the new ones get board refs.

Check counts in the DB:

```bash
python - <<'PY'
import sqlite3
conn = sqlite3.connect('data/app.db')
print('companies:', conn.execute('select count(*) from companies').fetchone())
print('ats refs:', conn.execute('select count(*) from companies where ats_provider is not null').fetchone())
print('jobs:', conn.execute('select count(*) from jobs').fetchone())
PY
```

## Placeholder keys → 422 errors

Resume upload, JD enrichment, and any gateway call with placeholder keys
(`your-*-key`, `sk-or-key_one`, …) fail at the provider and surface as a
structured **422** (`{"detail": "resume curation unavailable"}`) — this is
by design, not a bug. Fill real keys into
`src/job_hunter/llm_gateway/.env` (template:
`src/job_hunter/llm_gateway/.env.example`) and restart the worker.

## Env vars

`APP_*` must be exported/sourced in the shell (the app does not load a
`.env` file). See the README environment-variable table.
