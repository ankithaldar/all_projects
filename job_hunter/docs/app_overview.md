# Job Hunter — App Overview (short)

A local-only Python app that finds jobs in India for you, ranks them against
your resume/preferences, and shows them in a small web dashboard. It runs on
one machine: no cloud, no accounts, SQLite storage, plain HTML/JS frontend.

## What the app does (end-to-end)

1. **You set your profile once** — upload resume, pick target roles, industry
   vertical(s), cities, remote preference, salary expectation, experience.
   An agent parses the resume into a structured profile; you confirm it.
2. **Every night (and quick polls every 2h)** the discovery graph pulls fresh
   postings from official ATS board APIs (Greenhouse, Lever, Ashby, Workable,
   SmartRecruiters…) for companies it tracks (fan-out rotates through the
   backlog). Bulk aggregator pulls are a separate CLI path: `fetch-jobs`
   pulls whole feeds from remote-job aggregator APIs/RSS (Arbeitnow, Jobicy,
   Himalayas, Remotive, RemoteOK, WeWorkRemotely, Cutshort), and
   `scout-companies` is the CLI company-discovery path. A watched inbox also
   accepts saved LinkedIn/Naukri search exports you drop in manually (no
   ToS-violating scraping).
3. **A company scout** grows your target list via `scout-companies`
   (LinkedIn job pages located through the search index, website verified,
   ATS board detected, vertical tagged rules-only) plus seeds you provide.
   Names harvested from postings only auto-create companies on the
   `fetch-jobs` (bulk) path — the discovery graph resolves existing
   companies only and never auto-creates.
4. **New postings** are deduped (content hash), filtered by hard gates
   (location / work-mode / experience band / employment type), then enriched
   by an LLM through `llm_gateway`: must-have vs nice-to-have skills,
   salary range, experience band extracted from each JD.
5. **Scoring** combines skill coverage, embedding similarity, seniority fit,
   salary fit, recency, and vertical fit into one 0–100 score with a visible
   per-component breakdown — no black box.
6. **You review** ranked recommendations in the browser: open the job, save
   or dismiss with one keypress. (Dismiss/save feedback retuning is planned:
   `feedback_adjust` exists but is never called, so it does not retune
   rankings yet.)
7. Everything (runs, errors, tokens, cost per LLM call) is logged and visible
   on a Runs page with live progress.

Applying to jobs is intentionally **out of scope for v1** — the schema and
agent design leave a clean slot for a later applier extension.

## Main pieces

| Piece | Tech | Role |
|-------|------|------|
| API server | FastAPI + Uvicorn (127.0.0.1) | REST + SSE + serves `web/` |
| Worker | APScheduler process | nightly discovery, polls, sweeps; runs LangGraph graphs |
| Agent pipeline | LangGraph | load_profile → build_plan → fetch_pair → normalize_dedupe → enrich_jds → compute_embeddings → score_rank_persist → summarize_run |
| Tools | MCP servers (stdio, standalone CLI/LLM surface) | source fetching, resume parsing, safe DB ops, web search — the discovery graph itself uses repositories directly, not MCP |
| LLM calls | existing `llm_gateway` only | provider cascade + fallback, caching, cost logs |
| Storage | SQLite (WAL) | jobs, companies, profile, recommendations, runs, settings |
| UI | HTML/CSS/vanilla JS | 7 pages: dashboard, profile, companies, recommendations, jobs, runs, settings |

## Layout in one glance

```
job_hunter/
├── src/job_hunter/
│   ├── llm_gateway/     ← existing, untouched
│   └── job_hunter/      ← core, db, llm client, mcp_servers, adapters,
│                          services, graph, workers, api
├── web/                 ← HTML pages + js/css (no framework)
├── seeds/               ← companies, verticals taxonomy, skills aliases
├── docs/                ← architecture.md (full detail), roadmap.md,
│                          runbook.md, api.md
└── data/                ← app.db, gateway.db, cache.db, resumes/, inbox/
```

## Running it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev,semantic]'
cp src/job_hunter/llm_gateway/.env.example src/job_hunter/llm_gateway/.env
python main.py seed-db
python main.py api            # terminal 1 → http://127.0.0.1:8088
python main.py worker         # terminal 2 → schedules + discovery runs
python main.py fetch-jobs     # bulk pull from public aggregators
python main.py scout-companies # find + verify new companies
```

Full engineering detail (schemas, scoring math, endpoints, graph states):
see `docs/architecture.md`. Build order per feature branch: `docs/roadmap.md`.
Operations: `docs/runbook.md`; endpoint reference: `docs/api.md`.
