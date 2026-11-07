# Job Hunter

Local-only agentic job discovery system for India. See `docs/app_overview.md`
for the short version, `docs/architecture.md` for full design,
`docs/runbook.md` for operations, and `docs/api.md` for the HTTP API.

## Quick start

> **Python >=3.14 is required upfront** (the project declares
> `requires-python = '>=3.14'`; `setup.sh` refuses older interpreters).
> Check `python --version` before the manual steps below.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev,semantic]'
cp src/job_hunter/llm_gateway/.env.example src/job_hunter/llm_gateway/.env   # add your real keys

python main.py seed-db            # migrations + taxonomy + default settings
python main.py discover-companies # ingest seed companies into the DB
python main.py api                # terminal 1 -> http://127.0.0.1:8088
python main.py worker             # terminal 2 -> schedules + discovery runs

python main.py run-discovery      # one-off discovery run now
python main.py verify-ats         # fill in ATS board refs for new companies
python main.py mcp sources        # inspect an MCP server standalone
```

`./setup.sh` runs the full flow above (requires Python >= 3.14);
`./run.sh` starts API + worker together (waits for `/healthz`, honouring
`APP_HOST`/`APP_PORT`).

Health check: `GET /healthz` returns `status`, `db`, `gateway_db`, and
`version`. `gateway_db: false` simply means the gateway cost-log DB
(`data/gateway.db`) does not exist yet — no LLM call has run yet. It is
**not** unhealthy and is **not** an error; the first gateway call creates
the file. `db: false` likewise means the app DB has not been seeded yet
(run `python main.py seed-db`).

Auth / browser token UI: when `JH_API_TOKEN` is set, API calls without a
matching `Authorization: Bearer <token>` header get `401`. The web UI
prompts you for the token on a `401`, stores it for the session, and
retries with the Bearer header. The `TrustedHost` allowlist is hardcoded
to `127.0.0.1`, `localhost`, and `testserver` — setting a token alone
does **not** enable LAN exposure; non-loopback hosts are rejected by
`TrustedHostMiddleware` regardless of the token.

## Environment variables

`APP_*` variables must be exported (or sourced) in your shell — the app does
not load them from a file. LLM provider keys live in
`src/job_hunter/llm_gateway/.env` (copy from the gateway `.env.example`).

| Variable | Default | Purpose |
|----------|---------|---------|
| `APP_HOST` | `127.0.0.1` | API bind host |
| `APP_PORT` | `8088` | API bind port |
| `APP_LOG_LEVEL` | `INFO` | Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL) |
| `APP_DATA_DIR` | `./data` | Data directory (DBs, logs, inbox) |
| `SALARY_HARD_FLOOR_LPA` | `45` | Hard salary floor in LPA |
| `GATEWAY_DB_PATH` | `./data/gateway.db` | Gateway call/cost log DB |
| `GATEWAY_CACHE_PATH` | `./data/cache.db` | Gateway response cache DB |
| `GATEWAY_DASHBOARD_PORT` | `8099` | Gateway dashboard port |
| `OPENROUTER_API_KEYS` | — | OpenRouter keys (comma-separated) |
| `GROQ_API_KEYS` | — | Groq keys |
| `CEREBRAS_API_KEYS` | — | Cerebras keys |
| `NVIDIA_NIM_API_KEYS` | — | NVIDIA NIM keys |
| `GITHUB_API_KEYS` | — | GitHub keys (gateway templates also include BYTEZ/MISTRAL/OMNIROUTE/ORCAROUTER) |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434/v1` | Local Ollama fallback base URL |

## Security note

Keys in `src/job_hunter/llm_gateway/.env` are gitignored. If a `.env` file
containing real keys is ever committed, **rotate every key in it** before
pushing — git history retains them even after deletion.

## Widening the net

```bash
python main.py fetch-jobs                    # bulk pull all aggregators
python main.py fetch-jobs --sources arbeitnow,jobicy --limit 500

python main.py scout-companies                # find + verify new companies
python main.py scout-companies --queries 20 --max 15
```

`fetch-jobs` pulls whole feeds from the public aggregator APIs
(Arbeitnow, Jobicy, Himalayas, Remotive, RemoteOK, WeWorkRemotely,
Cutshort) and stores unseen postings, creating company rows for names not
yet tracked.

Cutshort is the India-first source. The other six are global remote boards
that only incidentally carry India roles (measured single-page India share:
RemoteOK 5%, Jobicy 3%, Arbeitnow 1%, Himalayas and Remotive 0%). Cutshort
returns 76% India-located postings with 99% carrying a structured INR
salary, which is what `salary_hard_floor_lpa` needs.

It has no public read API without a key, so the adapter reads the
server-rendered category pages instead. `?page=` is ignored upstream and
several category URLs render client-side only, so a pull yields roughly 200
postings rather than the ~44k in its sitemap. Reaching the full inventory
needs an API key from Cutshort.

`scout-companies` looks for hiring companies on LinkedIn job pages, proves
each company's website, detects its ATS board, and appends the result to
`seeds/companies_scouted.yaml`. LinkedIn is never crawled directly: its
`robots.txt` disallows `/jobs-guest/`, so job pages are located through the
DuckDuckGo HTML search index, which its `robots.txt` allows. A domain is
only accepted when a distinctive company-name token appears on the live
page, so guesses like `exl.ai` are rejected in favour of `exlservice.com`.

## Layout

- `src/job_hunter/llm_gateway/` existing LLM gateway (untouched).
- `src/job_hunter/job_hunter/` application package.
- `web/` HTML/CSS/vanilla-JS frontend served by the API on port 8088.
- `seeds/` companies, verticals taxonomy, skills aliases.
- `data/` SQLite databases, resumes, manual-export inbox (created at runtime).
- `docs/` architecture, roadmap, overview, `runbook.md`, `api.md`.

Notes on tools: the three MCP servers (`sources`, `resume`, `store`) are
stdio tools for discovery/LLM use; the discovery graph itself uses
repositories directly. There is no in-process LLM tool loop and no
LangChain bridge — both were removed (see `docs/architecture.md` §7–§9).
