# Status — job_hunter-review
| Task | Owner | State | Retries left |
|---|---|---|---|
| review-cycle-1 | 10 reviewers | DONE (2 REJECTED gates) | 0 |
| fix-round-1 W1 api-security | worker-backend | DONE (25 new tests, 31/31 with test_api, pylint 9.51) | 3 |
| fix-round-1 W2 db | database-engineer | DONE (16 new tests, 22/22, full 287 passed, pylint 10.00) | 3 |
| fix-round-1 W3 adapters-ssrf | worker-backend | DONE (77 new tests, 104/104, full 287 passed, pylint 9.73) | 3 |
| fix-round-1 W4 services-llm | worker-backend | DONE (18 new tests, 28/28, pylint 9.93) | 3 |
| fix-round-1 W5 graph-workers | worker-backend | DONE (16 new tests, 32 passed, pylint 8.71) | 3 |
| fix-round-1 W6 frontend-gateway | worker-frontend | DONE (21/21 tests, full suite 286 passed Lead-verified) | 3 |
| fix-round-1 W7 devops-docs | devops-sre | DONE (12/12 tests, ruff parses, pylint 8.91) | 3 |
| fix-round-1 W8 tester | tester | DONE (gateway 10/10, targeted 36/36, coverage gaps >0%) | 3 |
| fix-round-2 R2A repo-leak switch | worker-backend | DONE (36/36, 0 bare connects; 2 obsolete W4 tests -> Cycle-2 tester) | 3 |
| fix-round-2 R2B YAGNI cleanup | worker-backend | DONE (14 new tests, 60/60 targeted, E/F pylint 10.00) | 3 |
| fix-round-2 formatter (D13 headers/80col/docstrings repo-wide) | worker-backend | DONE (headers 0 old, >80=7 exempt, checkpoint fixed, ruff 918 pre-existing) | 3 |
| fix-round-2 tester followup (2 obsolete + checkpoint) | tester | DONE (308 passed, 1 skipped Lead-verified) | 3 |
| review-cycle-2 | full team (10 reviewers) | DONE (1 APPROVED / 9 REJECTED) | 0 |
| fix-cycle-3 F2 db-gateway | database-engineer | DONE (20 new tests, 42/42 targeted) | 3 |
| fix-cycle-3 F1 api-web-dashboard | worker-backend | DONE (17 new tests, 69/69, dashboard externalised) | 3 |
| fix-cycle-3 F5 devops-docs | devops-sre | DONE (13 new tests, checkpoint dep declared) | 3 |
| fix-cycle-3 F3 graph-arch | worker-backend | DONE (14 new tests, 34/34 targeted) | 3 |
| fix-cycle-3 F6 tester | tester | DONE (68/68 subsets, deletions verified) | 3 |
| fix-cycle-3 F4 adapters-mcp | worker-backend | DONE (26 new tests, 130/130, full 407 passed) | 3 |
| microfix dashboard/inbox/test | worker-backend | DONE (407 passed, 1 skipped Lead-verified) | 3 |
| review-cycle-3 | full team (10 reviewers) | DONE (5 APPROVED / 5 REJECTED) | 0 |
| review-cycle-4 | full team (10 reviewers) | DONE (5 APPROVED / 5 REJECTED) | 0 |
| prep-cycle-5 Q1-Q4 + microfixes | workers | DONE (505 passed) | 0 |
| review-cycle-5 | full team (10 reviewers) | DONE (4 APPROVED / 6 REJECTED) | 0 |
| review-cycle-6 | full team (10 reviewers) | DONE (4 APPROVED / 6 REJECTED) | 0 |
| prep-cycle-7 S1 curator-import | worker-backend | DONE (8 new tests, 31/31 targeted) | 3 |
| prep-cycle-7 S2 adapters-dbquote | worker-backend | DONE (9 new tests, 36/36 targeted, full 569) | 3 |
| prep-cycle-7 S3 docs-tester-web | devops-sre+tester | DONE (583 passed, workday test hermetic) | 3 |
| prep-cycle-6 R1 api-curator-web | worker-backend | DONE (Cycle 6 reviewed) | 0 |
| prep-cycle-6 R2 adapters-mcp | worker-backend | DONE | 0 |
| prep-cycle-6 R3 graph-db | worker-backend | DONE | 0 |
| prep-cycle-6 R4 devops-docs-tester | devops-sre+tester | DONE | 0 |
| prep-cycle-5 Q1 api-web | worker-backend | DONE
| prep-cycle-5 Q2 adapters-mcp | worker-backend | DONE
| prep-cycle-5 Q3 graph-db | worker-backend | DONE
| prep-cycle-5 Q4 devops-docs-tester | devops-sre+tester | DONE
| prep-cycle-4 P1 api | worker-backend | DONE (9 new tests, 40/40, pylint 9.52) | 3 |
| prep-cycle-4 P2 adapters-mcp | worker-backend | DONE (10 new tests, 125/125, mcp2-ok) | 3 |
| prep-cycle-4 P3 graph-db-gateway | worker-backend | DONE (9 new tests, 42/42 targeted, 0004 migration) | 3 |
| prep-cycle-4 P4 devops-docs-tester | devops-sre+tester | DONE (446 passed, wheel verified, rpm removed) | 3 |
| final-qa-commit-push | tester + Lead | queued | 3 |
Cycle: 1/200 -> fix loop. Flaws: .opencode/flaw-reports/job_hunter-flaws.md Decisions: .opencode/context/decisions.md
