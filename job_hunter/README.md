# Job Hunter

Local-only agentic job discovery system for India. See `docs/app_overview.md`
for the short version, `docs/architecture.md` for full design.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
cp .env.example src/job_hunter/llm_gateway/.env   # add your real keys

python main.py seed-db        # migrations + taxonomy + default settings
python main.py api            # terminal 1 -> http://127.0.0.1:8088
python main.py worker         # terminal 2 -> schedules + discovery runs

python main.py run-discovery  # one-off discovery run now
python main.py verify-ats    # fill in ATS board refs for new companies
python main.py mcp sources    # inspect an MCP server standalone
```

## Widening the net

```bash
python main.py fetch-jobs                    # bulk pull all aggregators
python main.py fetch-jobs --sources arbeitnow,jobicy --limit 500

python main.py scout-companies                # find + verify new companies
python main.py scout-companies --queries 20 --max 15
```

`fetch-jobs` pulls whole feeds from the public aggregator APIs
(Arbeitnow, Jobicy, Himalayas, Remotive, RemoteOK, WeWorkRemotely) and
stores unseen postings, creating company rows for names not yet tracked.

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
- `docs/` architecture, roadmap, overview.
