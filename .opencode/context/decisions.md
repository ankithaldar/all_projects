# Architecture Decision Record (ADR) Log — Tabular ML Platform Program

> **Owner of record:** `solution-architect` (contract content) · **Planner skeleton:** `product-planner` (this file, P0-1)
> **Program gate:** APPROVED by `anti-team-arch` Cycle 2
> **Rule:** This file freezes *which* decisions exist and their status. Contract details are owned by `solution-architect` — this log records freeze status by reference to `.opencode/plans/contracts.md` and intentionally contains **no invented contract content**.
> **P0 freeze:** 2026-11-08 — contracts.md dual APPROVED (reviewer PASS + anti-team APPROVED on FIX4); edge-cases.md dual APPROVED. P0 entry criteria met.

| ADR | Title | Status |
|-----|-------|--------|
| D-README-1 | README tie-break | frozen |
| D-MVP-READINESS-1 | MVP readiness vs production promotion | frozen |
| D-BEST-MODEL-V2-1 | `best_model` v2 contract (BLOCKING) | frozen |
| D-FITTED-REG-V2-1 | `fitted_registry` v2 dual-read | frozen |
| D-TEMPORAL-SPLIT-1 | Temporal split API | frozen |
| D-SECRETS-PII-GDPR-1 | Secrets / retention / PII / GDPR | frozen |
| D-CATALOG-V2-1 | Catalog v2 (ONE catalog.yml Option B; fragment merge deferred) | frozen |
| D-ERASURE-CLI-1 | Operator erasure CLI channel (secure env only) | frozen |
| D-P3-COVERAGE-1 | P3 `modeling.pipeline` coverage gate (parity suites, option 1) | frozen |
| D-SCOPE-FORECAST-1 | Cycle 6/7 approval scope = `forecast_ml_framework` ONLY; F2 woe/gbq + F6 supply-chain periphery OUT of forecast verdict | frozen |
| D-ACCEPTED-RISK-1 | Accepted risk: JWKS DNS TOCTOU (R2) + extraction rate limit (R3) | frozen |
| D-ACCEPTED-RISK-2 | Accepted risk: PII detection residuals N1/N6 (+ Cycle 7 N7–N9) | frozen |

---

## D-README-1 — README tie-break (frozen)

- **Status:** frozen
- **Context:** Where README guidance and program phase plan conflicted during P0 planning, a single tie-break rule was required so downstream phases have one source of truth.
- **Decision:** Program phase plan (`.opencode/plans/program.md` + gate verdicts in `.opencode/critique-log/`) governs on conflict; README is descriptive only and never overrides a frozen ADR or an `APPROVED` gate.
- **Consequences:** P0 contracts freeze proceeds against this rule; any README change that contradicts a frozen ADR must be re-routed through `solution-architect` + anti-team gate before adoption.
- **Owner:** Lead / `solution-architect`

## D-MVP-READINESS-1 — MVP readiness vs production promotion interpretation (frozen)

- **Status:** frozen
- **Context:** MVP v1 is defined as **P0 + P1 contracts + P2 platform production-readiness**. A prior ambiguity read "production-readiness" as automatic production promotion.
- **Decision (interpretation recorded):** **Readiness ≠ production promotion.** Completing P2 proves the platform is *ready* (build/deploy/observability criteria met); it does **not** authorize production traffic, model promotion, or release. Production promotion is solely controlled by the **P11 release gate** (and per-phase entry/exit gates in `program.md`).
- **Consequences:**
  - MVP v1 exit = P2 exit gate satisfied; no serving traffic beyond readiness checks.
  - No phase may claim "production" status before P11 exit gate passes.
  - Serving (P10) work is gated behind executed MLOps (P9) and cannot pre-empt P11.
- **Owner:** Lead / `anti-team-arch` (gate)

## D-BEST-MODEL-V2-1 — `best_model` v2 contract (BLOCKING) (frozen)

- **Status:** frozen
- **Freeze date:** 2026-11-08
- **Gate evidence:** reviewer PASS + anti-team APPROVED on FIX4 (contracts.md dual APPROVED)
- **Contract of record:** `.opencode/plans/contracts.md` Contract 3 (§§3.1–3.10) FIX4 — field table, example payload, and rejection criteria live there; this entry records freeze status only.
- **Context:** `best_model` in v1 is ambiguous; v2 must be frozen before P1 contract implementation and before any P2 IaC that provisions model-serving surfaces.
- **Decision (frozen):** As frozen in contracts.md Contract 3 (FIX4): `best_model` v2 is a dict with keys `{model, label_set, registry_ref, onnx_sidecar}` — field semantics, types, versioning, and validation per §§3.2–3.10 of the contract of record.
- **Acceptance:** ADR flipped to `frozen` with full field table + example payload + rejection criteria per contracts.md FIX4 (§§3.2–3.9, acceptance trace); P1 entry gate checks this ADR.
- **Owner:** `solution-architect`

## D-FITTED-REG-V2-1 — `fitted_registry` v2 dual-read (frozen)

- **Status:** frozen
- **Freeze date:** 2026-11-08
- **Gate evidence:** reviewer PASS + anti-team APPROVED on FIX4 (contracts.md dual APPROVED)
- **Contract of record:** `.opencode/plans/contracts.md` Contract 2 (§§2.1–2.8) FIX4 — read-precedence, fallback, migration/shadow, and deprecation window live there; this entry records freeze status only.
- **Context:** Existing `fitted_registry` consumers must keep working after v2 lands; dual-read (v1 + v2) compatibility must be decided before P1 and P3 (time-split) rely on it.
- **Decision (frozen):** As frozen in contracts.md Contract 2 (FIX4): dual-read strategy for `fitted_registry` v2 — canonical read order + fallback behavior + test hook per §§2.2–2.5 of the contract of record.
- **Acceptance:** ADR flipped to `frozen` naming canonical read order + fallback behavior + test hook per contracts.md FIX4 (§§2.2–2.8, acceptance trace).
- **Owner:** `solution-architect`

## D-TEMPORAL-SPLIT-1 — Temporal split API (frozen)

- **Status:** frozen
- **Freeze date:** 2026-11-08
- **Gate evidence:** reviewer PASS + anti-team APPROVED on FIX4 (contracts.md dual APPROVED)
- **Contract of record:** `.opencode/plans/contracts.md` Contract 4 (§§4.1–4.9) FIX4 — API signature, leakage invariants, and determinism guarantees live there; this entry records freeze status only.
- **Context:** P3 Time-Split requires a single temporal split contract so leakage cannot differ per caller.
- **Decision (frozen):** As frozen in contracts.md Contract 4 (FIX4): temporal split API — signature, time column / timezone, gap/embargo, and determinism guarantees per §§4.2–4.5 of the contract of record.
- **Acceptance:** ADR flipped to `frozen` with API signature + leakage invariants + determinism check per contracts.md FIX4 (§§4.2–4.9, acceptance trace).
- **Owner:** `solution-architect`

## D-SECRETS-PII-GDPR-1 — Secrets / retention / PII / GDPR (frozen)

- **Status:** frozen
- **Freeze date:** 2026-11-08
- **Gate evidence:** reviewer PASS + anti-team APPROVED on FIX4 (contracts.md dual APPROVED)
- **Contract of record:** `.opencode/plans/contracts.md` Contract 5 (§§5.1–5.9) FIX4 — secret source/rotation, retention windows, PII tagging/redaction, residency, and erasure flow live there; this entry records freeze status only.
- **Context:** P2 Platform/IaC needs a single posture for secrets handling, artifact/data retention, PII classification, and GDPR duties before any infra or data code is written.
- **Decision (frozen):** As frozen in contracts.md Contract 5 (FIX4): Secrets / retention / PII / GDPR posture per §§5.2(a)–(e) of the contract of record (including rotation out-of-band via cloud, no in-repo rotation in P0; residency region-agnostic, decision at P2 IaC before P11).
- **Acceptance:** ADR flipped to `frozen` per contracts.md FIX4 (§§5.2–5.9, acceptance trace); P2 entry gate checks this ADR.
- **Owner:** `solution-architect` + `security`

## D-CATALOG-V2-1 — Catalog v2 (frozen)

- **Status:** frozen
- **Freeze date:** 2026-11-08
- **Gate evidence:** reviewer PASS + anti-team APPROVED on FIX4 (contracts.md dual APPROVED)
- **Contract of record:** `.opencode/plans/contracts.md` D-CATALOG-V2-1 proposal + Contract 1 (§§1.1–1.9) FIX4 — this entry records freeze status only.
- **Context:** Contract 1 attached the catalog split to D-BEST-MODEL-V2-1 (sidecar half) with no catalog-specific ADR, orphaning the merge/collision contract.
- **Decision (frozen):** Catalog v2 per D-CATALOG-V2-1 / Contract 1 FIX4: ONE `catalog.yml` Option B in P0, fragment merge deferred (see Acceptance).
- **Acceptance:** Per contracts.md §1 / D-CATALOG-V2-1 FIX4, all three required and met: (1) fragment merge rules as deferred target only — P0 claims no fragment-merge mechanism; ONE `catalog.yml` with section banners (Option B); (2) ONNX sidecar entry deferred to P10 — in P0 there is no `best_model_onnx` catalog entry (`.onnx` loads rejected at the loader per Contract 3 §3.4); (3) per-task split: the 6 fragments listed in §1.2(b) are retained as the deferred target; P0 freezes ONE-catalog alphabetical grouping (Option B) now.
- **Owner:** `solution-architect`

## D-ERASURE-CLI-1 — Operator erasure CLI channel (secure env only) (frozen)

- **Status:** frozen
- **Freeze date:** 2026-11-08
- **Gate evidence:** anti-team APPROVED on `.opencode/plans/D-ERASURE-CLI-1.md` FIX
- **Contract of record:** `.opencode/plans/D-ERASURE-CLI-1.md` — decision / acceptance detail lives there; this entry records freeze status only.
- **Context:** P2 Platform/IaC needs a frozen operator erasure CLI channel before production CLI erasure can run; the frozen `erase_subject` semantics are not re-decided here.
- **Decision (frozen):** As frozen in `.opencode/plans/D-ERASURE-CLI-1.md` FIX: env-only rows-only CLI (clear subject id from `FORECAST_ERASE_SUBJECT_ID` only, never argv/logs) + library caller re-save duty (CLI passes `artifacts={}` and writes no markers; artifact retrain-flagging requires a library caller supplying artifact dicts + re-save via existing writers) + P11 gate (freeze alone does not authorize production traffic, model promotion, or release — D-MVP-READINESS-1, P11 only).
- **Acceptance:** ADR flipped to `frozen` per `.opencode/plans/D-ERASURE-CLI-1.md` FIX (anti-team APPROVED); P3 entry proceeds on frozen channel, production CLI erasure gated as above.
- **Owner:** `solution-architect`

## D-P3-COVERAGE-1 — P3 `modeling.pipeline` coverage gate, parity suites (option 1) (frozen)

- **Status:** frozen
- **Date:** 2026-11-08
- **Gate evidence / source:** Lead coverage ruling (option 1, recorded 2026-11-08) at `.opencode/plans/P3.md:537`; P3-D handoff coverage evidence at `.opencode/handoffs/P3-D.md:104-117` (literal named suites 63%, with parity suites 90% rounded there, 89.61% measured in P3.md:537).
- **Context:** The literal `modeling.pipeline` per-module gate in P3-D (`.opencode/plans/P3.md:517`, P3.md Global Constraints per-module gate) reached only 63% with the P3-D named suites (`test_p3_pipeline_integration`, `test_modeling_pipeline`, `test_modeling_split_calibrate`), because those suites never exercise `build_lift_table` / `build_promotion` / threshold / tuning-report lines in `modeling/pipeline.py` — lines owned by the parity suites (`.opencode/handoffs/P3-D.md:104-114`). P3-D handoff therefore left the coverage-gate arithmetic unresolved pending a Lead ruling (`.opencode/handoffs/P3-D.md:172-183`, options 1-3), while `.opencode/status.md` already marked P3 `done` — a contradiction this ADR clears.
- **Decision (coverage ruling, option 1):** The `modeling.pipeline` per-module coverage gate includes `forecast_ml_framework/tests/test_parity_pipeline.py` + `forecast_ml_framework/tests/test_interpretability.py` (parity suites) alongside the P3-D suites, reaching 89.61% whole-file coverage (≥85% gate satisfied), per `.opencode/plans/P3.md:537`.
- **Consequences:** P3-D gate passes; the P3-D coverage-gate item is cleared and P3 is consistently `done` across `.opencode/status.md`, `.opencode/critique-log/P3-D.md`, and `.opencode/critique-log/P3-gate.md` (all pointing to this ADR). No P3 product code changes. The ruling is ADR-backed for `.opencode/plans/P3.md:537` (kept as-is, now backed by this ADR).
- **Alternatives considered:**
  - Option 2 — accept whole-suite 85% total instead of the per-module gate (`.opencode/handoffs/P3-D.md:176-179`): rejected. P3.md Global Constraints define the gate per-module with no combined averaging; a total would weaken the per-module bar this ADR upholds.
  - Option 3 — authorize out-of-scope lift/promotion/threshold tests in the P3-D file (`.opencode/handoffs/P3-D.md:176-180`): rejected. That duplicates `test_parity_pipeline` coverage and is YAGNI scope creep in the P3-D file (`.opencode/handoffs/P3-D.md:112-114`).
- **Owner:** Lead (coverage ruling, 2026-11-08, per P3.md:537) / `docs-engineer` (this ADR transcription, FIX-GATE-DOCS)

## D-SCOPE-FORECAST-1 — Cycle 6/7 approval scope = `forecast_ml_framework` ONLY (frozen)

- **Status:** frozen
- **Date:** 2026-11-08
- **Gate evidence / source:** Lead task FIX-GATE-EVIDENCE (2026-11-08) recording the Cycle 6/7 scope adjudication; arch workstream handoffs `.opencode/handoffs/ADV-CYCLE6.md:8-9,15-16` and `.opencode/handoffs/ADV-CYCLE7.md:9-11,17-19` (both: scope `forecast_ml_framework` only, Verdict **APPROVED**, 0 blocking, forecast scope).
- **Context:** Arch Cycle 7 reported the scope ADR missing: the Cycle 6/7 APPROVED verdicts are stated for forecast scope, while the full-repo adversarial review (`.opencode/handoffs/ADV-FULL-REPO.md:10`, REJECTED, F1–F7 in `.opencode/flaw-reports/ADV-FULL-REPO-flaws.md:12-28`) ranges over the whole monorepo. Without a scope ADR, a reader could cite the forecast APPROVED verdict as a whole-monorepo approval, or cite F2/F6 against the forecast verdict.
- **Decision:** The Cycle 6/7 approval scope is **`forecast_ml_framework` ONLY**. Findings **F2** (`woe_iv_optimizer` SQLite UNION / BigQuery backtick / GBQ `sql.format` — `.opencode/flaw-reports/ADV-FULL-REPO-flaws.md:12-15`; no `woe_iv_optimizer` / `bigquery_loader` match under `forecast_ml_framework/` on grep, 2026-11-08) and **F6** (supply-chain periphery — Dockerfile/uv pin, `gbq_parallel_runner` + woe requirements unpinned, `.opencode/flaw-reports/ADV-FULL-REPO-flaws.md:25-26`) are explicitly **OUT of the forecast verdict**. They are monorepo follow-up, owned separately, and neither block nor are cleared by the forecast APPROVED verdicts.
- **Consequences:** P1/P2/P3 gate and flaw-report verdicts in `.opencode/critique-log/` / `.opencode/flaw-reports/` (FIX-GATE-EVIDENCE pass) are forecast-only verdicts; citing them for woe/gbq or supply-chain periphery is a scope error. F2/F6 remain open monorepo items under their own follow-up; closing them requires their own gate, not a re-read of Cycle 6/7.
- **Alternatives considered:**
  - Treat Cycle 6/7 APPROVED as whole-monorepo approval: rejected. Both handoffs state scope `forecast_ml_framework` only (`ADV-CYCLE6.md:8-9`, `ADV-CYCLE7.md:9-11`); the verdicts never attacked F2/F6 surfaces.
  - Keep F2/F6 as forecast blockers: rejected. F2 code is not under `forecast_ml_framework/` (grep, 2026-11-08, no matches), and F6 as scoped here is the periphery (gbq/woe requirements) — blocking the forecast verdict on non-forecast code contradicts the stated Cycle 6/7 scope.
- **Owner:** Lead (scope adjudication, FIX-GATE-EVIDENCE) / `docs-engineer` (this ADR transcription)

## D-ACCEPTED-RISK-1 — Accepted risk: JWKS DNS TOCTOU (R2) + extraction rate limit (R3) (frozen)

- **Status:** frozen
- **Date:** 2026-11-08
- **Gate evidence / source:** Lead task FIX-GATE-EVIDENCE (2026-11-08, Lead adjudication); residuals as documented in `.opencode/handoffs/ADV-CYCLE6.md:86-87` (R2/R3) and `.opencode/handoffs/ADV-CYCLE7.md:96-97` (R2/R3 carried), with the underlying vectors blocked in Cycle 7 (`ADV-CYCLE7.md:70-72` JWKS, `ADV-CYCLE7.md:80-81` extraction).
- **Context:** Cycle 6/7 APPROVED is a 0-blocking bar, but two non-blocking residuals remained carried as issues rather than accepted-risk ADRs: R2 JWKS DNS TOCTOU (validate-time DNS check vs fetch-time resolution; documented in `serving/auth.py`, no repro) and R3 extraction rate limit (token bucket 600/min/process, 2 Helm replicas by default; limiter proven working, residual is the at-limit aggregate across replicas). Arch Cycle 7 flagged both as carried residuals; the original full-repo F4/F7 forms (extraction amplification, JWKS scheme allowlist) were remediated in the shipped forecast paths and re-proven blocked in Cycle 6/7.
- **Decision (accepted risk, Lead adjudication):** R2 and R3 are **accepted risks, not 0-issue blockers**, for the forecast verdict. R2: JWKS validate+fetcher blocks file/http/private/metadata/localhost/encoded/userinfo forms and redirects to private targets (Cycle 7, all blocked); the residual DNS TOCTOU between validation and fetch is accepted with no repro on record. R3: the in-process token bucket (600/min) is accepted as the shipped limiter; the residual is aggregate extraction at-limit across 2 replicas (authenticated burst 650 → 641×200/9×429 in Cycle 7), documented, with no per-principal limit claimed.
- **Consequences:** P1/P2 gate and flaw-report PASS verdicts (FIX-GATE-EVIDENCE) stand with R2/R3 recorded here instead of as open issues. Operators must not read the verdict as claiming per-principal extraction limits or TOCTOU-proof JWKS resolution; tightening either (e.g. pinned resolution at fetch, distributed/per-principal limiter) is follow-up work under a new ADR if pursued. Readiness ≠ production promotion (D-MVP-READINESS-1) is unchanged.
- **Alternatives considered:**
  - Keep R2/R3 as blocking issues: rejected. Both are carried, documented residuals with the mandated vectors blocked and no repro (R2) / limiter proven working (R3); Lead adjudication under the 0-blocking bar accepts them rather than holding the forecast verdict open.
  - Claim R2/R3 fixed: rejected. No code change in this docs pass proves otherwise; the ADR records acceptance, not remediation.
- **Owner:** Lead (accepted-risk adjudication, FIX-GATE-EVIDENCE) / `docs-engineer` (this ADR transcription)

## D-ACCEPTED-RISK-2 — Accepted risk: PII detection residuals N1/N6 (+ Cycle 7 N7–N9) (frozen)

- **Status:** frozen
- **Date:** 2026-11-08
- **Gate evidence / source:** Lead task FIX-GATE-EVIDENCE (2026-11-08, Lead adjudication); residuals as documented in `.opencode/handoffs/ADV-CYCLE6.md:88-94` (N1, N6) and `.opencode/handoffs/ADV-CYCLE7.md:98-116` (N1, N6 carried; N7–N9 new, disclosed), with all mandated bare 11-digit / separator payloads blocked in Cycle 7 (`ADV-CYCLE7.md:32-56`).
- **Context:** PII detection in shipped forecast paths blocks every mandated payload form (Cycle 6 C-2 and Cycle 7 bare 11-digit suites, all blocked on every write surface), but a family of non-canonical / false-positive-tradeoff forms still PASS-WRITE: N1 names/columns (`alice smith` / `ALICE SMITH`, UK `+44` phone, generic name columns — FP tradeoffs, Lead adjudication since Cycle 5) and N6 dashed lowercase IBAN (`de89-3704-0044-0532-0130-00`; canonical compact/space-grouped IBAN forms are blocked). Cycle 7 additionally disclosed N7 (phone glued to a letter suffix, `15551234567x` / `…ext123`), N8 (fullwidth / arabic-indic unicode digits), and N9 (longer/non-US digit strings, e.g. `25551234567`, 12-digit, `015551234567` — boundaries prevent substring FP on numeric IDs). None is a mandated bare form.
- **Decision (accepted risk, Lead adjudication):** N1, N6, N7, N8, and N9 are **accepted risks, not 0-issue blockers**, for the forecast verdict. The shipped canonical forms (bare 10/11-digit, separator, SSN, compact/spaced IBAN, declared/undeclared consent paths) remain blocked as proven in Cycle 6/7; the residual family is documented FP-tradeoff / non-canonical surface, accepted as-is.
- **Consequences:** P1/P2 gate and flaw-report PASS verdicts (FIX-GATE-EVIDENCE) stand with N1/N6/N7–N9 recorded here instead of as open issues. No claim is made that name-column, UK-phone, dash-grouped-IBAN, letter-glued, unicode-digit, or non-US digit-string forms are detected; callers needing those forms must declare/track them explicitly. R1 (auth dev-mode bare `create_app()` 200, loopback by design — `ADV-CYCLE7.md:94-95`) is likewise carried documentation, not a blocker, and is recorded here by reference.
- **Alternatives considered:**
  - Keep N1/N6/N7–N9 as blocking issues: rejected. They are non-canonical forms in the same FP-tradeoff family already adjudicated at Cycle 5, with every mandated form blocked; Lead adjudication accepts the residual family rather than holding the forecast verdict open.
  - Claim N1/N6/N7–N9 fixed: rejected. No code change in this docs pass; the ADR records acceptance, not remediation. The Cycle 7 note stands: if Lead wants letter-glued suffixes in scope, the fix is a `(?!\d)`-style boundary (`ADV-CYCLE7.md:104-108`) — future work, not this verdict.
- **Owner:** Lead (accepted-risk adjudication, FIX-GATE-EVIDENCE) / `docs-engineer` (this ADR transcription)

---

*Skeleton created by `product-planner` under P0-1. P0 freeze 2026-11-08 (reviewer PASS + anti-team APPROVED on FIX4): D-BEST-MODEL-V2-1, D-FITTED-REG-V2-1, D-TEMPORAL-SPLIT-1, D-SECRETS-PII-GDPR-1, D-CATALOG-V2-1 flipped to frozen. No product code. Contract details remain reserved to `solution-architect` (`.opencode/plans/contracts.md`).*
