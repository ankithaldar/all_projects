# Job Hunter — App Overview (short)

A local-only Python app that finds jobs in India for you, ranks them against
your resume/preferences, and shows them in a small web dashboard. It runs on
one machine: no cloud, no accounts, SQLite storage, plain HTML/JS frontend.

## What the app does (end-to-end)

1. **You set your profile once** — upload resume, pick target roles, industry
   vertical(s), cities, remote preference, salary expectation, experience.
   An agent parses the resume into a structured profile; you confirm it.
2. **Every night (and quick polls every 2h)** a worker pulls fresh postings
   from safe public sources: official ATS board APIs (Greenhouse, Lever,
   Ashby, Workable, SmartRecruiters…), remote-job aggregator APIs/RSS, and
   career pages of companies it tracks. A watched inbox also accepts saved
   LinkedIn/Naukri search exports you drop in manually (no ToS-violating scraping).
3. **A company scout** keeps growing your target list: seeds you provide,
   names harvested from fetched postings, and web-search expansion — each
   company gets its careers page fingerprinted to find its ATS feed, and is
   tagged with an industry vertical (rules first, LLM fallback).
4. **New postings** are deduped (content hash), filtered by hard gates
   (location / work-mode / experience band / employment type), then enriched
   by an LLM through `llm_gateway`: must-have vs nice-to-have skills,
   salary range, experience band extracted from each JD.
5. **Scoring** combines skill coverage, embedding similarity, seniority fit,
   salary fit, recency, and vertical fit into one 0–100 score with a visible
   per-component breakdown — no black box.
6. **You review** ranked recommendations in the browser: open the job, save
   or dismiss with one keypress. Your feedback gently retunes future rankings.
7. Everything (runs, errors, tokens, cost per LLM call) is logged and visible
   on a Runs page with live progress.

Applying to jobs is intentionally **out of scope for v1** — the schema and
agent design leave a clean slot for a later applier extension.

## Main pieces

| Piece | Tech | Role |
|-------|------|------|
| API server | FastAPI + Uvicorn (127.0.0.1) | REST + SSE + serves `web/` |
| Worker | APScheduler process | nightly discovery, polls, sweeps; runs LangGraph graphs |
| Agent pipeline | LangGraph (+LangChain utils) | plan → fetch → normalize → filter → enrich → score → persist |
| Tools | MCP servers (stdio) | source fetching, resume parsing, safe DB ops, web search |
| LLM calls | existing `llm_gateway` only | provider cascade + fallback, caching, cost logs |
| Storage | SQLite (WAL) | jobs, companies, profile, recommendations, runs, settings |
| UI | HTML/CSS/vanilla JS | 7 pages: dashboard, profile, companies, recommendations, jobs, runs, settings |

## Layout in one glance

```
job_hunter/
├── src/job_hunter/
│   ├── llm_gateway/     ← existing, untouched
│   └── job_hunter/      ← core, db, llm bridge, mcp_servers, adapters,
│                          services, graph, workers, api
├── web/                 ← HTML pages + js/css (no framework)
├── seeds/               ← companies, verticals taxonomy, skills aliases
├── docs/                ← architecture.md (full detail), roadmap.md
└── data/                ← app.db, gateway.db, cache.db, resumes/, inbox/
```

## Running it (planned)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .
python -m job_hunter.api      # terminal 1 → http://127.0.0.1:8088
python -m job_hunter.worker   # terminal 2 → schedules + discovery runs
```

Full engineering detail (schemas, scoring math, endpoints, graph states):
see `docs/architecture.md`. Build order per feature branch: `docs/roadmap.md`.
