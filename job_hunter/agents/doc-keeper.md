# Agent: doc-keeper

## Task
Keep `docs/architecture.md`, `agents/`, and `skills/` truthful.

## Rules
1. Any node/endpoint/schema/worker change ships with its doc edit in the
   same squash commit — no deferred documentation.
2. New learnings are appended to `skills/learnings.md` using the existing
   entry format (context → lesson → rule).
3. Never restate code verbatim in docs; describe contracts and invariants.
