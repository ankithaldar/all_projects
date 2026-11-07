# Job Hunter — API Reference

Base: `http://127.0.0.1:8088` (host/port from `AppSettings`, i.e.
`APP_HOST`/`APP_PORT` or `config/app.yaml` `app:`). All routes are
implemented in `src/job_hunter/job_hunter/api/routes/`.

## Endpoints

| Method & path | Purpose |
|---------------|---------|
| GET `/healthz` | liveness + db/gateway status (`gateway_db: false` = gateway DB not created yet, no LLM call yet — not unhealthy) |
| GET/PUT `/api/profile` | current candidate profile |
| POST `/api/profile/resume` | multipart upload → parse → preview (422 if LLM unavailable, e.g. placeholder keys; 415 non-PDF; over 10 MB: 422 when the declared Content-Length exceeds it, 413 when the actual body does); duplicate sha returns existing resume_id + stored path; dangling duplicate raises RuntimeError 'resume row missing after save_resume_ex'; failure deletes owned resume row first, file kept if referenced; FK-blocked/retained failure sets exc.file_path falsy so API does not unlink stored file |
| PUT `/api/profile/skills` | edit skill list/weights |
| GET `/api/skills` | full canonical skill taxonomy (no `q` param) |
| GET/POST `/api/companies` | list (filters `status`, `vertical`) / create manually; priority clamped 1..5 on create/patch/import; non-numeric/non-finite import priority increments errors; numeric truncated via int() then clamped 1..5 (missing/None->3; 0->1 (clamped) on import/seed) bool priority is rejected, not clamped (422 create/patch; errors import; skipped seed) |
| POST `/api/companies/import` | import seed companies from YAML content (422 on invalid YAML); priority clamped 1..5 on create/patch/import; non-numeric/non-finite import priority increments errors; numeric truncated via int() then clamped 1..5 (missing/None->3; 0->1 (clamped) on import/seed) bool priority is rejected, not clamped (422 create/patch; errors import; skipped seed) |
| PATCH `/api/companies/{id}` | pause/pin/label vertical/edit; priority clamped 1..5 on create/patch/import; non-numeric/non-finite import priority increments errors; numeric truncated via int() then clamped 1..5 (missing/None->3; 0->1 (clamped) on import/seed) bool priority is rejected, not clamped (422 create/patch; errors import; skipped seed) |
| DELETE `/api/companies/{id}` | blacklist |
| GET `/api/jobs` | paginated explorer (`q` → FTS5; filters `city`, `work_mode`, `vertical`, `status`; params `page`, `per_page`) |
| GET `/api/jobs/{id}` | job detail incl. raw payload |
| GET `/api/recommendations` | ranked list (`min_score`, `status`, `vertical`, `include_gated`, `page`, `per_page`) |
| PATCH `/api/recommendations/{id}` | status transition (`status=new|saved|dismissed|applied`) |
| GET `/api/runs` | run history (`limit`) |
| POST `/api/runs/discovery` | enqueue a discovery run (409 if one is already pending/running; atomic via `create_if_not_active` single `BEGIN IMMEDIATE` session — the API route is atomic too, not just `RunManager.enqueue`; sequential second call also 409s) |
| POST `/api/runs/{id}/cancel` | mark a pending/running run failed (cancelled via ui) |
| GET `/api/runs/{id}/events` | run events (`after_id`, `limit`) |
| GET `/api/runs/{id}/stream` | SSE stream: `event: log` (`{id, level, node, message}`), `event: done` (`{status, stats}`), `event: timeout` (`{}`) |
| GET `/api/settings` | seeded read-only copies of every stored setting (seeded from `config/app.yaml` by `ensure_defaults`) |
| PUT `/api/settings` | update one setting (`key`, `value`); 4 keys only — `scoring_weights`, `salary_hard_floor_lpa`, `discovery.max_llm_calls_per_run`, `discovery.enrich_per_run_cap` (all other keys, incl. schedule/source toggles, → 422; schedule/source enabling is yaml-only) |
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

## Auth / token note (D1)

Local single-user app; no login. Defaults bind `127.0.0.1` only with a
hardcoded TrustedHost allowlist (`127.0.0.1`, `localhost`, `testserver`)
and security headers. If `APP_HOST` is set to a non-loopback address, the
app requires `JH_API_TOKEN` (Bearer token, checked in `api/deps.py`) and
fails fast at startup without it — never expose the API without it.

Setting a token alone does **not** enable LAN exposure: `TrustedHost`
is hardcoded, so requests with a LAN hostname/IP are rejected regardless
of the token. The web UI prompts for the token on a `401`, keeps it for
the session, and retries with the Bearer header.

## CSRF note

CSRF: non-GET with cross-origin Origin/Referer is rejected 403;
headerless requests not checked (browser always sends Origin on POST).
The 403 shape is `{'detail': 'forbidden origin'}`. `Referer` is an
accepted alternative to `Origin` when same-host (the browser UI sends
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
