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

## Undeclared transitive deps surface only in clean venvs
- Context: setup.sh's fresh venv failed on `tiktoken` though tests passed
  for weeks in the system Anaconda env.
- Lesson: ad-hoc system installs hide missing dependency declarations.
- Rule: any package imported by shipped code goes into pyproject
  dependencies immediately; verify via the clean-room setup.sh path.

## Coerce types at API boundaries
- Context: `bootstrap(seeds_dir=...)` crashed because argparse passed a
  string where a Path was expected.
- Lesson: trust nothing crossing module boundaries — normalize there.
- Rule: boundary functions do `Path(x)` / type coercion themselves.

## pgrep patterns match your own command line
- Context: `pgrep -fc 'python main.py worker'` returned inflated counts;
  a group-kill then hung the shell session.
- Lesson: bash -c wrappers embed the literal pattern, and negative-pid
  kills misfire when $! is not a real process-group leader.
- Rule: bracket-trick patterns (`[m]ain.py worker`), kill by exact PID,
  and stop foreground children first so EXIT traps fire naturally.

## Path arithmetic breaks silently until exercised
- Context: resume uploads 500'd four times; unit tests stayed green
  because none constructed the real gateway.
- Lesson: relative parents[N] chains are invisible bugs until a request
  touches them; tests that stub the boundary hide layout mistakes.
- Rule: add existence assertions for resolved config paths
  (test_gateway_paths_resolve_to_real_files) and smoke the real route.

## Guard rails belong around construction, not just use
- Context: the 500 became a 502 only after moving get_client() inside try.
- Lesson: object construction can fail as meaningfully as method calls.
- Rule: wrap dependency construction and usage in the same error policy.

## `from __future__ import annotations` + Pydantic = lazy NameErrors
- Context: `ProfileExtraction` used `List[str]` but profile_curator.py
  never imported List; every resume upload failed at validation time
  with "not fully defined ... call model_rebuild()".
- Lesson: postponed annotations turn missing typing imports into runtime
  failures on first validation, not at class definition.
- Rule: when a module defines Pydantic models under future-annotations,
  import every annotation name from typing, and add a construct +
  model_json_schema() test to force resolution.

## Ashby public API rejects POST-with-compensation
- Context: posting-api returned 401 once includeCompensation was sent via POST.
- Lesson: 'public' ATS endpoints still gate premium params behind auth.
- Rule: probe with the plainest call first (GET, no params); add params only after confirming baseline access.

## Enterprise ATS coverage = Workday CXS
- Context: most GCC seed companies failed Greenhouse/Lever detection — they are Workday shops.
- Lesson: the dominant enterprise pattern is {tenant}.wd{n}.myworkdayjobs.com with a public CXS JSON endpoint.
- Rule: capture full host + site path as board_ref and paginate CXS by limit/offset.

## Slug-guessing rescues boards hidden from careers pages
- Context: several companies host greenhouse/lever boards without linking them from probed pages.
- Lesson: absence of a fingerprint on the company site does not mean absence of a board.
- Rule: after page probes fail, guess slugs from domain core + alnum name against board-host URL patterns and verify by marker text.

## Caps + backlog healing beat monolithic runs
- Context: enriching hundreds of JDs at ~40s/call exceeded any command timeout, killing runs mid-flight and orphaning work.
- Lesson: bounded per-run side-effectful work plus persistent backlog queries convert timeouts into progress-by-installments.
- Rule: cap side-effectful LLM work per run; on later runs, fall back to oldest unfinished backlog before declaring success.

## Mass string-literal rewrites need placeholder escaping
- Context: a tokenize-based double→single quote conversion corrupted files containing \\' escapes and docstrings with apostrophes.
- Lesson: delimiter swaps interact with escape sequences; naive chained replaces break both directions.
- Rule: placeholder-protect existing escapes before swapping delimiters, gate such passes behind the full test suite, and commit immediately after verification.
