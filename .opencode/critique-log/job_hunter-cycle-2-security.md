# Anti-Team Security Cycle-2 — REJECTED (ses_e4b3465f5ffejtKSZ3jPwldDa7)
HIGH: (1) No Origin/Referer CSRF check — simple POST from evil site passes TrustedHost+empty token, enqueues LLM run. Required: Origin/Referer allowlist for non-GET /api/* + evil-Origin 403 test. (2) SSRF redirect/TOCTOU — follow_redirects=True sends GET to private hop before post-validation; DNS resolve-then-fetch race. Required: manual hop validation + pinning/peer check + test private hop never requested.
MEDIUM: gateway CSP blocks dashboard inline script/style (functional break); resume read_bytes unbounded before caps; prompt delimiter breakable via </untrusted_data>; companies/import domain unvalidated + YAML flow-writer injection.
LOW: StaticFiles bypasses require_token; /healthz token probe break; dashboard unauthenticated; APP_HOST='' binds all; MCP str(exc); token timing OK (compare_digest).
PASS: board_ref, dashboard XSS, href allowlist, settings allowlist, .env gitignore, generic 500.
