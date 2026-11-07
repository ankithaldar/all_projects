## Project: job_hunter Full Review Loop
### Summary
Run full review team on complete `./job_hunter` (168 files, ~130 code files: Python app + llm_gateway, SQL migrations, HTML/JS/CSS web, shell scripts, config/seeds/docs). Loop: review -> flaw report -> workers fix (TDD) -> anti-team gate -> re-review until 0 issues. Then document, commit, push.

### Tech Stack
- Python >=3.14, FastAPI/Pydantic, httpx, SQLAlchemy?/aiosqlite SQLite, langgraph/langchain-core, mcp, APScheduler
- SQL: SQLite migrations 0001_init.sql, 0002_perf.sql (job_hunter/db/migrations)
- Frontend: HTML5 + vanilla JS/CSS in job_hunter/web/
- Shell: setup.sh, run.sh
- No CI config found yet (devops to verify)

### Architecture (as-documented)
- src/job_hunter/job_hunter/ — app: api/, adapters/, core/, db/, graph/, llm/, services/, workers/, mcp_servers/
- src/job_hunter/llm_gateway/ — existing LLM gateway (README: "untouched")
- web/ static frontend served by API :8088
- seeds/, config/app.yaml, data/ (gitignored SQLite, runtime)

### MVP Scope (v1) — THIS TASK
- [ ] Cycle 1 parallel read-only review by 10 roles: reviewer, security, anti-team-arch, anti-team-security, adversarial-tester, tester, database-engineer, worker-frontend, devops-sre, docs-engineer/solution-architect
- [ ] Synthesize into .opencode/flaw-reports/job_hunter-flaws.md with file:line, severity, evidence
- [ ] Anti-team gate cycle-1 critique-log entries (APPROVED/REJECTED per finding validity)
- [ ] Fix loop: file-disjoint worker tasks, TDD (write failing test first, watch RED, minimal GREEN, full suite), max 200 cycles, retry budget 3/task
- [ ] Re-review until 0 open issues from all reviewers + tester full suite green
- [ ] Documentation update (README/docs if drift found), commit + push
OUT OF SCOPE: New features, llm_gateway redesign (only fix if security/critical flaw proven), external API keys/network-dependent live scrapes, cloud deploy.

### Acceptance Criteria
- [ ] Every reviewer returns 0 issues (or issues are documented as accepted-risk ADR with Lead tie-break)
- [ ] `pytest -q` in job_hunter/ passes fully (deps installed per pyproject)
- [ ] `ruff check` passes (config fixed if broken)
- [ ] No tracked secrets / .env / db / pdf / logs in git
- [ ] Flaw report + critique logs + final QA report committed
- [ ] Commit pushed to origin branch agentic_ai/job-hunter

### Baseline (Lead, 2026-11-07, before Cycle 1)
- git branch: agentic_ai/job-hunter, HEAD 56fde14, clean status
- pytest: 5 collection errors — missing modules in THIS env (feedparser, tiktoken, langgraph, apscheduler) — env gap, not yet proven code flaw; workers must install `pip install -e '.[dev]'` in venv and re-run
- ruff: FAIL to parse pyproject.toml — `[tool.ruff.format] indent-style='spaces'` invalid (expected 'space'); proven flaw F-001 candidate
- git ls-files: no .env/.db/.pdf/.log/egg-info tracked (good)
- pytest warning: Unknown config option asyncio_mode (pytest-asyncio not installed in env)

### Risks / Open Questions
- Env deps missing may mask real test failures until venv installed
- llm_gateway is large & described untouched — review it for security anyway, do not refactor without proof
- 200-cycle loop budget per AGENTS.md; deadlock >3 cycles on same issue -> deadlock.md + Lead tie-break + ADR

Does this look right? User pre-approved by instructing "run ... till 0 issue ... document then commit and push" — proceeding without interview gate (brainstorming opt-out).
