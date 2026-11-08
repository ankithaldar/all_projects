# Flaw report — ADV-FULL-REPO (transcribed, FIX-DOCS-1, 2026-11-08)

Source: `.opencode/handoffs/ADV-FULL-REPO.md`. Verdict: **REJECTED**.

Findings (transcribed in substance, file:line as in the handoff):

- F1 [HIGH auth/deploy] Default Docker/Helm deployment unauthenticated:
  `Dockerfile:32` ENTRYPOINT uvicorn `--host 0.0.0.0` + module-level
  `app=create_app()` (`serving/app.py:1281`) skips `validate_bind_host`
  (`serving/app.py:1095`); Helm defaults `values.yaml:9,13` set neither
  auth env. Proven: no-env `create_app` → `POST /predict` = 200.
- F2 [HIGH/MED injection] `woe_iv_optimizer` SQLite UNION exfiltration
  proven (`sqlite_loader.py:75` unescaped f-string); BigQuery backtick
  breakout (`bigquery_loader.py:36`) and GBQ `sql.format(**ctx)`
  (`run_etl_in_levels.py:520`).
- F3 [MED consent] Consent opt-in, fail-open by omission:
  `guard_write` returns when `consent_ledger is None`
  (`governance/pii.py:216-217`); PII detection false negatives proven
  (phone, full_name, SSN, IBAN PASS with `pii_classes=['none']`);
  unsalted `subject_hash` dictionary-reversible (`governance/scrub.py:70`).
- F4 [MED extraction, amplifies F1] `/predict` 600/min/process token
  bucket (`serving/app.py:483-516,952-961`), 2 replicas default;
  proven 626/650 burst successes.
- F5 [LOW] OIDC token without `sub` VERIFIED (`serving/auth.py:394-401`).
- F6 [LOW supply-chain] Dockerfile `pip install uv` unpinned +
  `python:3.14-slim` no digest (`Dockerfile:4,9,16`).
- F7 [LOW] JWKS `default_fetcher` no scheme allowlist
  (`serving/auth.py:161-191`).

Gate verdict: REJECTED (see `../critique-log/ADV-FULL-REPO.md`).
This file transcribes the handoff; it does not create a new verdict.
