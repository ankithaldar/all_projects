# P1-F DVC Execution Decision Record

- **Status:** Recorded
- **Record ID:** P1-F
- **Date:** 2026-11-08
- **Owner:** docs-engineer
- **Depends On:** P0 frozen only per `.opencode/context/decisions.md` (7 ADRs frozen, 2026-11-08) and `.opencode/plans/contracts.md` FIX4 (dual APPROVED). No P1 code task is a dependency.
- **Scope:** DVC execution question for P1 only. Documentation only. This record creates no code, no configuration, no remote, and no infrastructure.
- **Verdict:** DEFER

**Rationale (single sentence):** DVC remains configuration-only today through `dvc_params` in `platform.yml` with no execution code and no observed volume, retention, audit, or remote-backend evidence that satisfies the EXEC bar.

## Context

P0 froze the contracts that P1 now implements. P1 implements those contracts without introducing IaC. The DVC question is whether P1 should also introduce DVC execution — initializing DVC, tracking data files, adding stages, configuring remotes, and wiring hooks. This record decides that question.

DVC remains parameters and configuration only today in the codebase. Kedro and MLflow carry the current reproducibility and tracking paths. Artifact stores carry deletion and retention. No DVC stage, no tracked file, and no remote exists in the working tree today.

This record uses one verdict vocabulary only. The permitted terms are `DEFER`, `EXEC`, and `EXEC-LITE`. `DEFER` is the default. `EXEC` or `EXEC-LITE` requires every criterion in its list to be met with cited evidence. Any other verdict wording is outside the vocabulary.

## Current control (config-only today)

The DVC control today is configuration plus schema validation only:

- `forecast_ml_framework/conf/base/parameters/platform.yml:79-83` defines `dvc_params` with `enabled: true`, `remote: 'local'`, `remote_url: ''`, and `push_on_run: false`. The empty `remote_url` means no remote exists today.
- `forecast_ml_framework/src/forecast_ml_framework/core/config_schema.py:282-288` defines the `DvcParams` schema, and `forecast_ml_framework/src/forecast_ml_framework/core/config_schema.py:575` wires it into the top-level configuration. The schema validates shape only.
- `forecast_ml_framework/conf/local/parameters.yml:36-40` sets `enabled: false` and `push_on_run: false` for local development, so local runs never push.
- `forecast_ml_framework/README.md:163-166` states that DVC remains parameters and configuration only with no execution code.
- `forecast_ml_framework/pyproject.toml:49-52` records that no `import dvc` appears in `src/` and keeps `dvc>=3,<4` as an optional extra.
- Search of `forecast_ml_framework/src` for `import dvc`, `from dvc`, and `dvc init|repro|push|pull` returns no hits. Search of the working tree for `dvc.yaml` returns no hits. No execution path exists today.

## Criteria that gate EXEC and EXEC-LITE

| ID | Criterion | Evidence today (repo path or `Assumption:`) | EXEC implication |
|----|-----------|---------------------------------------------|------------------|
| C1 | Data volume and retention interplay with P1-D2 | `Assumption:` P1-D2 deletion paths have no DVC-aware handling today. Volume evidence: `remote_url: ''` at `forecast_ml_framework/conf/base/parameters/platform.yml:79-83`. | Unmet. EXEC needs a design that prevents erasure and retention from forcing DVC history rewrites, plus measured volume data. |
| C2 | GDPR and legal-hold constraint | `Assumption:` Erasure and legal-hold rules have no DVC history handling today. Anchor: `.opencode/context/decisions.md` D-SECRETS-PII-GDPR-1 frozen. | Unmet. EXEC needs explicit handling for subject deletion and legal hold within versioned history. |
| C3 | Cloud backend cost | `forecast_ml_framework/conf/base/parameters/platform.yml:79-83` shows `remote_url: ''` with no selected backend. Anchor: `.opencode/plans/P1.md` P1-F spec requires a costed backend before execution. | Unmet. No remote, no storage bill, and no cost sign-off exists today. |
| C4 | Team familiarity and operability | `forecast_ml_framework/pyproject.toml:49-52` keeps `dvc` as an optional extra outside `src/`. No runbook for DVC exists in the tree. | Unmet. EXEC needs a named owner, a runbook, and training evidence. |
| C5 | P2 IaC coupling | Anchor: `.opencode/plans/P1.md` Global Constraints defer IaC to P2. Anchor: `.opencode/plans/program.md` P2 row places IaC after P1. | Unmet by design in P1. EXEC must wait for P2 backend and secret handling choices. |
| C6 | Rollback cost and reversibility | `Assumption:` No DVC history exists, so rollback today is the absence of DVC. No reversal test exists. | Unmet. EXEC needs a tested reversal path that restores prior data pointers without data loss. |

## EXEC bar (all items must hold with evidence)

`EXEC` means full DVC execution: `dvc init`, a stage file, versioned data, and a configured remote with push and pull. All of the following must hold:

- E1 Volume: measured tracked volume exceeds the flip-signal threshold in the DEFER section below, with file sizes and version counts cited.
- E2 Retention and erasure: a DVC-aware retention and erasure design proves that P1-D2 `delete` and `delete_entity` semantics never require history rewrites and never resurrect erased subjects.
- E3 GDPR and legal hold: subject deletion and legal-hold retention are specified for versioned history, including what remains after deletion and who approves it.
- E4 Backend and cost: a single remote is selected by backend type and region with a storage and transfer cost estimate approved within the P2 IaC budget.
- E5 Operability: a named owner and a runbook for `repro`, `push`, `pull`, history inspection, and recovery are recorded.
- E6 P2 coupling: P2 IaC provisions the remote and the credential path, with secrets from the environment only.
- E7 Reversal: a tested reversal restores prior pointers and verifies data integrity on a clean checkout.

No item above holds today with evidence. `DEFER` therefore holds.

## EXEC-LITE bar (all items must hold with evidence)

`EXEC-LITE` means data-only versioning: tracked dataset files only, no pipeline stage file, no repro wiring, and a local remote only. All of the following must hold:

- L1 Volume: the same measured volume evidence as E1, cited with file sizes and version counts.
- L2 Retention and erasure: the same DVC-aware deletion design as E2, limited to tracked dataset files.
- L3 GDPR and legal hold: the same subject deletion and legal-hold specification as E3, limited to tracked dataset files.
- L4 Backend and cost: cost estimate for local-only operation plus an explicit decision that no cloud remote is used yet.
- L5 Operability: a named owner and a short runbook for `add`, `push`, `pull`, and recovery for dataset files only.
- L6 P2 coupling: confirmation that no IaC is needed for a local remote, so P2 remains unaffected.
- L7 Reversal: a tested reversal for dataset pointers only.

No item above holds today with evidence. `DEFER` therefore holds for `EXEC-LITE` as well.

## Assessment (why DEFER holds today)

- **C1 (volume and retention):** No measured volume exists in the record. The existing retention and erasure surfaces delete expired snapshots and feature rows. Versioning those paths without a DVC-aware design would preserve erased bytes in history or force history rewrites. That risk is sufficient on its own to keep `DEFER`.
- **C2 (GDPR and legal hold):** No specification exists for erased subjects inside versioned history. Until that specification exists, versioning would create a retention and audit conflict.
- **C3 (backend and cost):** No remote exists. `remote_url: ''` means there is no backend to cost. Selecting a backend in P1 would pre-empt P2 IaC choices.
- **C4 (operability):** No owner or runbook exists. Introducing execution without those artifacts would create an unowned production path.
- **C5 (P2 coupling):** P1 defers IaC to P2 by plan. A P1 remote decision would front-run P2 backend, region, and secret handling.
- **C6 (rollback):** No reversal path exists because no DVC history exists. Creating history first and testing reversal later reverses the required order.

`Assumption:` Each finding above is the current state. New measurements or a future ADR can change any finding with cited evidence.

## If verdict were EXEC or EXEC-LITE (follow-on outline — not authorized)

This section is an outline for a future decision. It does not authorize any code. The outline applies only if a future record selects `EXEC` or `EXEC-LITE` with evidence.

**Gate for any follow-on (mandatory):** A new ADR is required, plus a program amendment in `.opencode/plans/program.md`, plus a Lead re-plan. No `dvc.yaml`, no remote, no tracked file, and no hook may land before that gate is `APPROVED` by the gating reviewer. This record alone never authorizes execution.

**Outline for `EXEC` (full execution):**

- Stage file: new `dvc.yaml` at the project root with stages that mirror Kedro boundaries, for example `data_prep` and `train`, each with explicit `deps`, `params`, and `outs`. Stage names and parameters require ADR approval.
- Remote: one remote only, named to match `dvc_params.remote`, with `remote_url` supplied from the environment. The backend choice (GCS or S3) and the bucket come from the P2 IaC decision. No secret value is stored in YAML.
- Tracked files: new `.dvc` files for the dataset payloads under the catalog filepaths chosen by the ADR, for example one `.dvc` file per large dataset. No `.dvc` file lands before the ADR.
- Hooks: no hook lands without ADR approval. Any hook that touches retention, erasure, or publication requires P1-D2 compatibility evidence.
- Wiring: `dvc_params.enabled`, `remote`, `remote_url`, and `push_on_run` in `platform.yml` become the sole configuration source. Code reads them through the existing schema.

**Outline for `EXEC-LITE` (data-only versioning):**

- Stage file: none. No `dvc.yaml` is created under `EXEC-LITE`. Repro wiring stays out of scope.
- Remote: local remote only. No cloud backend is selected.
- Tracked files: new `.dvc` files for dataset payloads only, chosen by the ADR. No artifact or model output is tracked under `EXEC-LITE`.
- Hooks: none.
- Limits: `push_on_run` stays `false`. `EXEC-LITE` never implies pipeline execution.

**Common requirements for either outline:** Version pins and the optional extra status are reviewed in the ADR. GDPR, retention, and legal-hold handling are specified before any tracked file is created. Rollback is tested on a clean checkout before any remote push is enabled.

## If verdict is DEFER (what would change it)

`DEFER` changes to `EXEC` or `EXEC-LITE` only when measurable signals cross explicit thresholds with cited evidence. The thresholds below are proposed decision thresholds, marked as `Assumption:`. They are not measured values today.

- **Volume signal:** `Assumption:` Tracked datasets under catalog filepaths exceed 5 GB in total, or a single dataset exceeds 1 GB and is shared across 2 or more environments for reproduction. Evidence required: file sizes, version counts, and sharing scope cited from the catalog and the artifact store.
- **Audit signal:** `Assumption:` Reproduction across time or across machines becomes mandatory for an audit or compliance check. Evidence required: the audit requirement, the required retention window, and the showing that MLflow plus the artifact store cannot meet it alone.
- **Backend signal:** P2 IaC selects a versioned remote bucket with a cost estimate and credential path. Evidence required: the P2 decision, the bucket location, and cost sign-off.
- **Erasure signal:** A DVC-aware retention and erasure design is recorded and tested. Evidence required: the design plus tests showing deletion never requires history rewrites.
- **Operability signal:** A named owner and a runbook exist. Evidence required: the owner name and the runbook path.
- **Reversal signal:** A tested reversal restores prior pointers on a clean checkout. Evidence required: the test and its output.

Any one signal alone never flips the verdict. `EXEC` needs all signals together. `EXEC-LITE` needs the subset in its bar above. A future record must cite evidence for each signal it claims.

## Consequences of DEFER

- P1 ships without DVC execution. Reproducibility stays with MLflow and the artifact store, as today.
- P1-D2 erasure and retention remain honest without history complications.
- No storage cost, no remote, and no IaC change occur in P1.
- P2 IaC remains the single place that selects any future remote backend.
- A future ADR remains the only path to execution. No silent introduction of DVC execution can occur through configuration alone.

## Alternatives considered

- **EXEC in P1:** Rejected today. No volume evidence, no erasure design, no remote, no owner, and no reversal test exist. Each gap maps to a criterion in C1 through C6.
- **EXEC-LITE in P1:** Rejected today. Even data-only versioning needs the erasure design, the GDPR specification, and the owner plus runbook. Those items are missing today.
- **DEFER in P1:** Chosen. It matches the current configuration-only control, keeps P1 inside its no-IaC scope, and leaves a clear evidence bar for any future change.

## No-IaC affirmation

This record creates no infrastructure. It creates no `dvc.yaml`, no `.dvc` directory, no remote, and no hook. It provisions no bucket, no cluster, and no secret. P2 remains the only phase that provisions infrastructure. Scope support: `.opencode/plans/P1.md` Global Constraints plus `.opencode/plans/program.md` P2 row.

## References

- `.opencode/context/decisions.md` — 7 ADRs frozen, 2026-11-08 (P0 entry evidence).
- `.opencode/plans/contracts.md` — FIX4 contract of record (P0 frozen).
- `.opencode/plans/program.md` — P0 through P11 chain; P2 owns IaC.
- `.opencode/plans/P1.md` — P1-F task spec (decision record, documentation only).
- `forecast_ml_framework/conf/base/parameters/platform.yml:79-83` — `dvc_params` definition.
- `forecast_ml_framework/conf/local/parameters.yml:36-40` — local override (`enabled: false`).
- `forecast_ml_framework/src/forecast_ml_framework/core/config_schema.py:282-288` and `:575` — `DvcParams` schema and wiring.
- `forecast_ml_framework/README.md:163-166` — configuration-only statement.
- `forecast_ml_framework/pyproject.toml:49-52` — optional extra status with no `src/` import.

## Verification (for the reviewer)

- Verdict is `DEFER`, with one rationale sentence, the criteria table C1 through C6, the `EXEC` bar E1 through E7, the `EXEC-LITE` bar L1 through L7, the follow-on outline, the flip signals, and this affirmation present.
- Only the three permitted verdict terms appear in this record. Any other verdict wording is absent.
- `git diff --stat` for this task shows this file only. No `dvc.yaml`, no `.dvc` directory, and no `src/` or `conf/` paths appear as new files in this task.
- Search of `src/` for DVC execution imports and commands returns no hits.
- Search of the working tree for `dvc.yaml` returns no hits.
