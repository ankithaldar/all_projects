# Job Hunter — API Reference

Base: `http://127.0.0.1:8088` (host/port from `AppSettings`, i.e.
`APP_HOST`/`APP_PORT` or `config/app.yaml` `app:`). All routes are
implemented in `src/job_hunter/job_hunter/api/routes/`.

## Endpoints

| Method & path | Purpose |
|---------------|---------|
| GET `/healthz` | liveness + db/gateway status (`gateway_db: false` = gateway DB not created yet, no LLM call yet — not unhealthy); behind the app-wide `require_token` dependency, so it returns 401 without the Bearer header when `JH_API_TOKEN` is set |
| GET/PUT `/api/profile` | current candidate profile; GET returns a defaults dict when unset (candidate 1 seeded by migration, not auto-created) |
| POST `/api/profile/resume` | multipart upload → parse → preview (422 if LLM unavailable, e.g. placeholder keys; 415 non-PDF; over 10 MB: 422 when the declared Content-Length exceeds it, 413 when the actual body does); duplicate sha returns existing resume_id + stored path; dangling duplicate raises RuntimeError 'resume row missing after save_resume_ex'; failure deletes owned resume row first, file kept if referenced; FK-blocked/retained failure sets exc.file_path falsy so API does not unlink stored file |
| PUT `/api/profile/skills` | edit skill list/weights; body is a bare array of names (not an object) |
| GET `/api/skills` | full canonical skill taxonomy (no `q` param) |
| GET/POST `/api/companies` | list (filters `status`, `vertical`) / create manually; priority clamped 1..5 on create/patch/import; non-numeric/non-finite import priority increments errors; numeric truncated via int() then clamped 1..5 (missing/None->3; 0->1 (clamped) on import/seed) bool priority is rejected, not clamped (422 create/patch; errors import; skipped seed) |
| POST `/api/companies/import` | import seed companies from bare YAML string body (422 on invalid YAML; 1MB / 1_000_000-char cap, over → 422 `content too large`); priority clamped 1..5 on create/patch/import; non-numeric/non-finite import priority increments errors; numeric truncated via int() then clamped 1..5 (missing/None->3; 0->1 (clamped) on import/seed) bool priority is rejected, not clamped (422 create/patch; errors import; skipped seed) |
| PATCH `/api/companies/{id}` | pause/pin/label vertical/edit; priority clamped 1..5 on create/patch/import; non-numeric/non-finite import priority increments errors; numeric truncated via int() then clamped 1..5 (missing/None->3; 0->1 (clamped) on import/seed) bool priority is rejected, not clamped (422 create/patch; errors import; skipped seed) |
| DELETE `/api/companies/{id}` | blacklist |
| GET `/api/jobs` | paginated explorer (`q` → FTS5; filters `city`, `work_mode`, `vertical`, `status`; params `page`, `per_page`) |
| GET `/api/jobs/{id}` | job detail incl. decoded `raw` (the stored payload decoded to JSON under the `raw` key; no payload key is echoed back — the response carries `raw`, not the stored payload column) |
| GET `/api/recommendations` | ranked list (`min_score`, `status`, `vertical`, `include_gated`, `page`, `per_page`) |
| PATCH `/api/recommendations/{id}` | status transition (`status=new|saved|dismissed|applied`); `status` is a status query param (`?status=saved`), not a body |
| GET `/api/runs` | run history (`limit`) |
| POST `/api/runs/discovery` | enqueue a discovery run (409 if one is already pending/running; atomic via `create_if_not_active` single `BEGIN IMMEDIATE` session — the API route is atomic too, not just `RunManager.enqueue`; sequential second call also 409s) |
| POST `/api/runs/{id}/cancel` | immediate finish: marks a pending/running run `failed` (`cancelled via ui`) at once; no cooperative cancel flag is read by the worker |
| GET `/api/runs/{id}/events` | run events (`after_id`, `limit`) |
| GET `/api/runs/{id}/stream` | SSE stream: `event: log` (`{id, level, node, message}`), `event: done` (`{status, stats}`), `event: timeout` (`{}`) |
| GET `/api/settings` | seeded read-only copies of every stored setting (seeded from `config/app.yaml` by `ensure_defaults`) |
| PUT `/api/settings` | update one setting; 4 keys only — `scoring_weights`, `salary_hard_floor_lpa`, `discovery.max_llm_calls_per_run`, `discovery.enrich_per_run_cap` (all other keys, incl. schedule/source toggles, → 422; schedule/source enabling is yaml-only); `key` is a key query param (`?key=...`) with a bare JSON body value. Only `scoring_weights` and `salary_hard_floor_lpa` are effective (read via `SettingsRepository.get` in graph nodes); the two `discovery.*` dotted keys are stored-but-never-read under their literal dotted key — effective discovery caps come from `config/app.yaml` / the top-level `discovery` settings dict, so the dotted PUT keys are API-only |
| GET `/api/stats/dashboard` | dashboard KPI counters |
| GET `/api/llm/stats` | gateway rollup from `data/gateway.db` (`days`) |

Planned but **not implemented**: `POST /api/companies/{id}/refresh`,
`GET /api/recommendations/{id}`, `POST /api/inbox/import`,
`GET /api/inbox/preview` (the inbox is scanned by the worker interval job).

## Error bodies

Two shapes exist — do not assume only one:

1. **Unhandled errors (500)** use problem-style JSON from the
   `api/main.py` exception handler:

   ```json
   {"title": "Internal Server Error", "status": 500, "detail": "internal error"}
   ```

2. **Expected `HTTPException` errors** (401, 404, 409, 415, 422, …) use
   FastAPI's default `{detail}` shape:

   ```json
   {"detail": "unauthorized"}
   ```

   Pydantic request-validation failures are also `422` with
   `{"detail": [...]}` (field errors).

_sanitize_nonfinite stringifies non-finite floats in validation errors (422 not 500); jobs page le=10000; profile experience 0-60.

Status codes:

- `401` — a token is configured (`JH_API_TOKEN`) and the
  `Authorization: Bearer <token>` header is missing or wrong
  (`{"detail": "unauthorized"}`).
- `404` — unknown run/company/job/recommendation id (`detail` names the
  resource).
- `409` — a discovery run is already pending/running (`ConflictError`:
  'another run is pending or running'; API route detail
  'a run is already active').
- `502` — upstream failure (`AdapterError`: 'adapter error').
- `503` — gateway unavailable (`GatewayUnavailableError`: 'gateway
  unavailable'; also surfaced as 422 'resume curation unavailable'
  on the resume-upload path, see below).
- `422` — validation or domain failure: Pydantic field errors, invalid
  company domain, invalid recommendation status, salary floor out of
  range, invalid YAML import, or resume curation failing (e.g.
  placeholder provider keys causing `resume curation unavailable`). See
  `docs/runbook.md`.
- `500` — unexpected error; problem-style body above, full traceback is
  logged server-side.

`AppError` mapping (F1, `api/main.py` handler, problem-style
`{title, status, detail}`): `ConflictError` → 409 `{'title':
'Conflict', 'status': 409, 'detail': 'conflict'}`; `AdapterError` →
502 `{'title': 'Bad Gateway', 'status': 502, 'detail': 'adapter
error'}`; `GatewayUnavailableError` → 503 `{'title': 'Service
Unavailable', 'status': 503, 'detail': 'gateway unavailable'}`;
`DatabaseError` → 500 `'database error'`; other `AppError` → 500
`'internal error'`. (Direct `HTTPException` 409s from the runs route
use the `{detail}` shape; `AppError` handler responses use the
problem-style shape above.)

## Bounds and validation (422 table)

Out-of-range ids/params fail Pydantic validation as 422 (never 500),
via the `RequestValidationError` handler in `api/main.py` (`{"detail":
errors}`, non-finite floats stringified by `_sanitize_nonfinite`).

| Bound | Value | Applies to |
|-------|-------|------------|
| `MAX_SQLITE_ID = 2**63 - 1` (formerly `_MAX_ID`, now single home in `core/constants.py`; routers import it, no local copy) | SQLite signed int64 max, symmetric `ge=-MAX_SQLITE_ID, le=+MAX_SQLITE_ID` | path ids: jobs, recommendations, runs, companies; runs `after_id` (`ge=0, le=MAX_SQLITE_ID`); a negative-huge id (e.g. `-10**400`) fails as 422, never 500 |
| `page` | 1..10000 (`le=10000`) | jobs, recommendations |
| `per_page` | 1..100 | jobs (default 25), recommendations (default 30) |
| `limit` | 1..100 (runs list), 1..500 (run events) | runs |
| `experience_years` | 0..60 | profile payload |
| discovery-cap | 1..10000, integer (non-bool) | `PUT /api/settings?key=discovery.max_llm_calls_per_run` and `?key=discovery.enrich_per_run_cap`; other types → 422 `discovery cap must be an integer`, out of range → 422 `discovery cap out of range` |
| salary-floor | 0..1000, finite number | `PUT /api/settings?key=salary_hard_floor_lpa`; non-number/bool → 422 `salary floor must be a number`, non-finite or out of range → 422 `salary floor out of range` |
| scoring-weights | object with finite numeric values | `PUT /api/settings?key=scoring_weights`; non-object → 422 `scoring_weights must be an object`, non-number/bool value → 422 `scoring_weights values must be numbers`, non-finite or overflow value (inf/nan/huge `OverflowError` on `float()`) → 422 `scoring_weights values must be finite` |

## Auth / token note (D1)

Local single-user app; no login. Defaults bind `127.0.0.1` only with a
hardcoded TrustedHost allowlist (`127.0.0.1`, `localhost`, `testserver`)
and security headers. If `APP_HOST` is set to a non-loopback address, the
app requires `JH_API_TOKEN` (Bearer token, checked in `api/deps.py`) and
fails fast at startup without it — never expose the API without it.

`/healthz` is also behind the app-wide `require_token` dependency: with a token set, health checks without the Bearer header get 401.

Setting a token alone does **not** enable LAN exposure: `TrustedHost`
is hardcoded, so requests with a LAN hostname/IP are rejected regardless
of the token. The web UI prompts for the token on a `401`, keeps it for
the session, and retries with the Bearer header.

## CSRF note

CSRF: non-GET with cross-origin Origin/Referer is rejected 403 —
API-only scope (`/api/` paths only, non-GET only; `api/main.py:
csrf_origin_check`); headerless requests not checked (browser always
sends Origin on POST). The 403 shape is `{'detail': 'forbidden origin'}`.
Allowed-host bypass: an Origin/Referer host in
`127.0.0.1`/`localhost`/`testserver` bypasses the check, and a same-host
match against the `Host` header also bypasses it (`Referer` is an
accepted alternative to `Origin` when same-host; the browser UI sends
`Origin` automatically). A malicious site cannot drive the local API
from a victim's browser because its cross-origin `Origin`/`Referer`
will not match the `Host` header.

## Gateway dashboard (port 8099, separate process)

Not part of this API, listed for completeness: `/` (dashboard.html),
`/dashboard.css`, `/dashboard.js`, and `/api/stats` (JSON) are served
from `src/job_hunter/llm_gateway/templates/`; unknown paths → 404.
`send_error` (e.g. 501 for unsupported methods) also returns the standard
security headers (Content-Security-Policy, X-Content-Type-Options,
X-Frame-Options) via `_send`.
Checkpoint note: resume is conditional on
`langgraph-checkpoint-sqlite>=3.1` + `aiosqlite>=0.20` being installed
(see `docs/architecture.md` §10); without them runs execute without a
checkpointer. State normalization: checkpoint-deserialized candidates
are coerced dict → `CandidateProfile` via `_normalize_candidate`
(`graph/nodes.py`). MCP 2.x fallback: all three MCP servers
(`sources_server.py`, `resume_server.py`, `store_server.py`) try
`mcp.server.fastmcp.FastMCP` first and fall back to
`mcp.server.mcpserver.MCPServer` (aliased as `FastMCP`) on
`ModuleNotFoundError`, so they import on both mcp 1.x and mcp 2.x
(see `docs/architecture.md` §8).

Resume upload reads the body in 1 MB chunks (`file.read(1_000_000)`)
and returns 413 when the actual total exceeds 10 MB.
