# Learnings — Job Hunter build & hardening

Append-only log. Format: **Context** → Lesson → Rule.

---

## LangGraph node config injection
- Context: `load_profile(state, config)` crashed with "missing positional
  argument" only when `from __future__ import annotations` was active.
- Lesson: LangGraph inspects resolved annotations; a second param typed
  `Dict[str, Any]` is treated as a plain dependency, not the runnable config.
- Rule: type the second node parameter exactly as `RunnableConfig`
  (imported from `langchain_core.runnables`) — never a loose alias.

## Claim-once run protocol
- Context: CLI `execute_pending_run` pre-claimed the oldest pending run,
  then `RunManager.execute` claimed again and got None → runs 'skipped'.
- Lesson: ownership of state transitions must live in exactly one layer.
- Rule: only `RunManager.execute` moves pending→running; callers pass ids,
  they never claim.

## Pending-run pickup needs an executor
- Context: API-enqueued runs stayed pending until the nightly cron.
- Lesson: enqueueing without a consumer is a silent stall, not a queue.
- Rule: any producer of work units gets a matching recurring drainer
  (worker polls pending every 30s).

## pytest piped through tail masks failures
- Context: a commit chain gated on `pytest -q | tail -1 && git ...` merged
  with 1 failing test because `$?` was tail's exit code.
- Lesson: pipelines report the LAST command's status.
- Rule: capture `${PIPESTATUS[0]}` or run pytest bare when gating logic.

## Read-only SQLite URIs fail on missing files
- Context: `file:x?mode=ro` raised 'unable to open database file' in tests
  that skipped bootstrap.
- Lesson: ro mode never creates the file; missing ≠ empty.
- Rule: repositories assume migrations ran (bootstrap first); tests must
  call `run_migrations(tmp_path/'app.db')` before read-only access.

## Background uvicorn wedges a persistent shell
- Context: `nohup uvicorn ... &` kept the bash session blocked past its
  timeout even after curls finished.
- Lesson: background children inherit the session's stdout/stderr pipes;
  the tool waits on pipe closure, not on the foreground job.
- Rule: launch with explicit redirection + `&`, capture `$!`, then
  `kill $PID; wait $PID` inside the SAME command.

## stdio MCP servers own stdout
- Context: app JSON logs on stdout corrupted JSONRPC framing; the MCP
  client spewed 'Failed to parse JSONRPC message' while still working.
- Lesson: any subprocess speaking a line protocol must not share stdout
  with logging.
- Rule: console logging always goes to stderr (`StreamHandler(sys.stderr)`)
  — stdout is reserved for IPC.

## MCP child processes need explicit PYTHONPATH
- Context: spawned `-m job_hunter.mcp_servers.*` died with ModuleNotFound
  despite working in the parent shell.
- Lesson: env vars are not inherited from an interactive shell by spawned
  service processes.
- Rule: `MCPClientManager._child_env()` injects the src root into
  PYTHONPATH plus JH_APP_CONFIG for every child; per-server launch
  failures are logged and skipped, never fatal.

## Cron vs APScheduler day-of-week numbering
- Context: standard cron `0=Sun`; APScheduler uses `0=Mon..6=Sun`.
- Lesson: silently reusing cron DOW shifts schedules by one weekday.
- Rule: convert tokens via `(d-1)%7` for digits, pass names through.

## fastembed as a Py3.14-safe embedding path
- Context: torch/onnx wheels lag new CPython releases; semantic scoring
  needed to survive their absence.
- Lesson: optional heavy deps behind an ABC let the system degrade to FTS5
  keyword overlap without code churn.
- Rule: providers return None when unavailable; matcher switches to the
  keyword fallback keyed off `NullProvider.model_id`.
