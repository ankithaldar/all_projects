# Agent: bug-hunter

## Task
Find and fix defects by exercising real code paths, not by reading alone.

## Method
1. Boot the API (`uvicorn --factory job_hunter.api.main:create_app`) with
   `APP_DATA_DIR` pointed at a temp dir; curl every route with happy-path
   and adversarial inputs (empty strings, quotes, `+`, unicode).
2. Trace each 500/unhandled exception to its root cause before editing.
3. Prefer fixing the cause over patching the symptom; keep fixes minimal.
4. Add one regression test per defect inside the existing test module that
   covers that area — no new test files unless a new area opens up.

## Known trap patterns in this codebase
- LangGraph node second params must be typed `RunnableConfig`.
- `pytest | tail` masks exit codes; check `$?` separately.
- Read-only SQLite URIs fail on missing files — bootstrap first.
- Background servers must be killed with an explicit PID, not left to hold
  the shell session.
