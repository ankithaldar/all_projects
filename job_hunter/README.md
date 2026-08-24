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
python main.py mcp sources    # inspect an MCP server standalone
```

## Layout

- `src/job_hunter/llm_gateway/` existing LLM gateway (untouched).
- `src/job_hunter/job_hunter/` application package.
- `web/` HTML/CSS/vanilla-JS frontend served by the API on port 8088.
- `seeds/` companies, verticals taxonomy, skills aliases.
- `data/` SQLite databases, resumes, manual-export inbox (created at runtime).
- `docs/` architecture, roadmap, overview.
