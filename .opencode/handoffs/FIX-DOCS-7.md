# FIX-DOCS-7 — Reviewer Cycle9 doc drift (docs-engineer handoff)

**Owns (exclusive):** `forecast_ml_framework/infra/README.md`,
`forecast_ml_framework/docs/runbook-p2.md`,
`forecast_ml_framework/README.md` (serving refs + tree only). No `src/` touched.

## Changes

1. **infra/README.md** — "Two independent stacks" → three; added Azure row
   (private storage/KV/VNet/Entra/AKS stub off, verified against
   `infra/azure/main.tf` + `infra/azure/README.md`); Azure env-auth
   (`ARM_*` / `az login`) + required `subscription_id`/`tenant_id`;
   Azure apply block (`infra/azure/README.md:62-66` commands verbatim);
   "No Docker (A5)" flipped to shipped-image note (`Dockerfile`,
   fail-closed, P11-gated promotion); `azurerm` backend noted; Validation
   now lists `infra/gcp, infra/aws, and infra/azure`.
2. **runbook-p2.md** — Retention/erasure step 3 now cites the **frozen**
   ADR `D-ERASURE-CLI-1` (anti-team `APPROVED` 2026-11-08, decisions.md
   D-ERASURE-CLI-1, contract `.opencode/plans/D-ERASURE-CLI-1.md`:
   env-only rows-only CLI, `artifacts={}`, no markers; P11-only gate).
   New mandatory step 4 (library-caller duty): supply artifact dicts
   (fitted registry + best-model envelope) → `erase_subject` writes
   `{subject_hash, requested_at, action='retrain_required'}` markers
   (`__meta__.erasure` / top-level `erasure`) → re-save via existing writers;
   rows-only CLI exit 0 ≠ complete GDPR erasure. Audit renumbered step 5.
3. **README.md (serving refs + tree only)** — Serving API + Features +
   env-table refs re-verified by grep/read against
   `serving/app.py` / `serving/model_loader.py`: `create_app` 1112,
   `create_app_from_registry` 1351, `_validate_auth_config` 287,
   `validate_bind_host` 257, `_hmac_key_id` 373, `_audit_predict` 395,
   `_log_access` 424, `MIN_API_KEY_LENGTH` 96, typed-error classes 406,
   pyfunc (class 518, default 544, warning 590, gated fallback 756),
   restricted path 756; non-named refs updated or dropped.
   Tree: added `conf/base/platform.yml` (governance_params.retention_days).

## Evidence

- `grep -i 'no[- ]docker' forecast_ml_framework/infra/README.md` → empty.
- `grep -i azure forecast_ml_framework/infra/README.md` → row L9, apply L32-41, validate L160.
- `grep -i 'library-caller|retrain_required|frozen under' forecast_ml_framework/docs/runbook-p2.md` → L137,148-159.
- `grep 'serving/app.py:[0-9]|serving/model_loader.py:[0-9]' forecast_ml_framework/README.md` (Serving API/Features/env) → only verified def/const lines (96,99,105,257,287,325,373,395,424,444,467,510,548,604,809,847,1112,1168,1188,1195,1219,1333,1351,705; model_loader 406,518,544,590,756).
- `grep 'platform.yml' forecast_ml_framework/README.md` → tree `conf/base/platform.yml` + existing `conf/base/parameters/platform.yml` distinct.

## Assumptions

- README Known-Issues (#19/#21/…) + Change Log carry dated (2026-11-07)
  serving refs left as history — outside "Serving API serving refs"
  scope for this task; re-verify on next docs pass if reviewer requires.
- Azure apply/validate commands copied verbatim from
  `infra/azure/README.md:62-66,114-118`; no terraform run in this env.

## Blockers

None. Awaiting anti-team-arch gate (docs-engineer never self-approves).
