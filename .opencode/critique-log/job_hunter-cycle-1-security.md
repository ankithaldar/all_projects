# Anti-Team Security Critique — job_hunter cycle-1
Verdict: REJECTED (ses_e4b6d6d0bffeFjOnOU68fQ1fV3)
Date: 2026-11-07

[CRITICAL] api/main.py:44,61-90 + api/deps.py:15 + all api/routes/*.py — Unauthenticated localhost API with write/delete endpoints (PUT /api/profile, POST /api/profile/resume, PUT /api/settings, DELETE /api/companies/{id}, POST /api/runs/discovery). No auth/Host/CSRF middleware, no TrustedHost, no CORS control. api/__main__.py uses settings.host (APP_HOST) with no 0.0.0.0 guard. Mitigation: Host allowlist or localhost token (data/.token 0600), Origin/CSRF check, hard-fail non-loopback unless token, rate-limit.

[HIGH] mcp_servers/sources_server.py:53-74 + adapters/http_client.py:48-78,90-94,114-172 — SSRF via MCP fetch_page/detect_careers and adapter fetchers; follow_redirects=True, RobotsCache fail-open, no private-IP/redirect/scheme guard. Mitigation: central URL guard https-only, DNS private-IP block, redirect/size caps, ATS host allowlist.

[HIGH] llm/structured.py:76-83 + services/jd_analyst.py:84 + profile_curator.py:124-130 + llm/mcp_tools.py:129-186 — Indirect prompt injection: raw JD/resume payload in single user message, no delimiters/system instruction; run_tool_loop executes any LLM-proposed tool, no allowlist/approval. Mitigation: delimit <untrusted_data>, system-role instruction, per-task tool allowlist + approval for writes, arg schema validation.

[HIGH] mcp_servers/resume_server.py:24-38 + services/resume_parser.py:27-45 — Arbitrary file read via MCP extract_text(path) -> PdfReader + read_bytes, no confinement to data/resumes. Mitigation: resolve + is_relative_to(data_dir/resumes), reject absolute/parent.

[HIGH] adapters/http_client.py:90-172 + resume_parser.py:36-43 + web_search.py:198,203 — Unbounded fetch/PDF OOM/CPU DoS: no max-bytes, ResumeParser iterates all pages. Mitigation: 2-5MB cap, 30-50 page cap, 50k char cap, timeouts.

[MEDIUM] ATS board_ref host/path injection: greenhouse.py:59, lever.py:65, ashby.py:91, workable.py:66, smartrecruiters.py:106, recruitee.py:58, personio.py:66, workday.py:66-70, sources_server.py:43-47. Mitigation: per-provider regex ^[A-Za-z0-9_-]{1,64}$, httpx.URL quoting.

[MEDIUM] Stored XSS gateway dashboard: llm_gateway/templates/dashboard.html:115-139 + dashboard.py:36-192 — innerHTML with session_id/provider/model_used persisted in llm_calls. web/js/components.js:46 href=rec.url no javascript: check (web/js/api.js safe via createTextNode). Mitigation: textContent only, href scheme allowlist http/https, CSP.

[MEDIUM] .env gitignore gap + error leakage: .gitignore no .env (docs/architecture.md:920 claims gitignored), setup.sh creates gateway .env, api/main.py:86-89 returns detail=str(exc)[:300], profiles.py:122-123 leaks. Mitigation: gitignore .env*, secret scan, generic 500s.

[MEDIUM] Arbitrary settings write: api/routes/settings.py:34-51 + repositories/settings.py:54-67 + graph/nodes.py:277-300 + scheduler.py:136-144 — PUT /settings any key/value, nodes reads max_llm_calls/enrich caps, scheduler no max_instances cap. Mitigation: key allowlist + ranges, max_instances=1+coalesce, budget caps.

[LOW] YAML injection linkedin_scout.py:597-651 manual flow-mapping without "/newline escaping. Mitigation: yaml.safe_dump.

[LOW] Supply chain: setup.sh pip upgrade + >= bounds, no lock/hashes. No curl|sh found (clean).

Checked clean: verify=False, shell=True, eval/exec, pickle/yaml.load, subprocess, CORS*, debug/0.0.0.0/reload = 0 matches. SQL parameterized ? everywhere + allowlists. yaml.safe_load everywhere. profile_curator Path(name).name traversal block, profiles 10MB cap, manual_inbox .html/.csv only.

Blockers before APPROVED: 1) Host allowlist+token+CSRF loopback, 2) SSRF guard, 3) prompt-injection hardening, 4) extract_text confinement + size caps, 5) board_ref validation + dashboard escaping + .env gitignore + settings allowlist + generic errors, 6) re-attack PoCs.
