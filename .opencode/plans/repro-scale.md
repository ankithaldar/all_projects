---
---
# Repro / Scale Implementation Plan (REPRO-PLAN-FIX)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Each task below is independently assignable, testable, and file-disjoint. Workers read their task section only, plus `contracts.md` for their contract.

I'm using the writing-plans skill to create the implementation plan. (Product-planner autonomous mode: planned to Lead only via the blackboard; no direct user dispatch.)

**Goal:** Deliver MVP v1 repro/scale as **R1-feasibility + R3-pins + S2-contract only** (no S1, R2 doc-only), with S1 Spark GBM deferred to a separate ADR phase, R2 DVC doc-only until `D-DVC-EXEC-1`, and every deferred item (R4 manifest, S3 partition/PIT, S5a/b/c) defined as ADR-gated tasks that cannot block MVP — resolving anti-team REJECTED (S1 breaks best_model v2/fitted/serving, R1 AQE-off bit-identical unproven, R2 DVC exec pre-empts DEFER + mutates frozen `split_meta` without ADR, no Owns matrix, no per-task acceptance commands).

**Architecture:** MVP is measurement + pins + one new pure reader module only. R1 first *measures* determinism variance before any bit-identical claim is decided; R3 pins the exact environment bytes; S2 freezes a chunked Parquet reader contract aligned to P1-B2 watermark semantics. No Spark GBM estimator, no DVC execution, no `split_meta` mutation, and no IaC land in MVP. All frozen P0/P1 surfaces (`best_model` v2 envelope, `fitted_registry` dual-read, `temporal_split`, `governance/scrub.py` sole leaf) are consumed read-only.

**Tech Stack:** Python (pandas, numpy, pyarrow, pytest, pytest-cov), Spark 3.5.x + Java 17 LTS (repro env only, feasibility-measured — not a training estimator), Docker (pinned base digest), uv (frozen lockfile), GitHub Actions ubuntu runner.

**Spec:** `.opencode/plans/contracts.md` (FIX4, dual APPROVED 2026-11-08) · `.opencode/context/decisions.md` (7 ADRs frozen; D-BEST-MODEL-V2-1, D-TEMPORAL-SPLIT-1, D-SECRETS-PII-GDPR-1 read-only here) · `.opencode/plans/program.md` (P0→P1→P2 chain; readiness ≠ promotion per D-MVP-READINESS-1) · `.opencode/plans/P1.md` (P1-B2 streaming semantics, P1-D1 scrub sole leaf, P1-F DVC DEFER) · `.opencode/plans/P1-dvc-decision.md` (P1-F verdict `DEFER` — this plan does not overturn it) · `.opencode/plans/P2.md` (P2-C1 container, P2-R1 CI, P2-I1 Azure IaC patterns — consumed, never duplicated)

**Owner of this plan:** `product-planner` (`.opencode/plans/repro-scale.md` only; no `src/` edits in this planning task)
**Contract owner:** `solution-architect` (`.opencode/plans/contracts.md` FIX4). Any field/semantics change vs frozen ADRs requires a new ADR + anti-team gate — workers must not reinterpret frozen contracts. `split_meta.data_hash` freeze (RS-R2D) is owned by `solution-architect`, never by a repro/scale worker.
**All paths below are repo-relative.** `forecast_ml_framework/` is the project root. Do not edit outside your task's `Owns:`.

---

## Global Constraints (apply to every repro/scale task)

- TDD mandatory: RED test commit first, then GREEN, then REFACTOR. Every task's sequence below is RED→GREEN→REFACTOR→gate. Every worker leaves both commits (squashed history must show RED failure log in `.opencode/handoffs/RS-<task>.md`).
- **Single coverage gate definition (sole definition — all per-task commands below reference this name and never redefine it):** per-task Python gate = `pytest --cov=<exact-module> --cov-fail-under=85 <exact-test-file> -q` where `<exact-module>` is the full import path without `src.` prefix (e.g. `forecast_ml_framework.pipelines.data_prep.chunked_reader`) and `<exact-test-file>` is the task's `tests/test_rs_*.py` path. For doc-only / config-only / IaC tasks with no new Python production module (RS-S1D, RS-R2D, RS-R3, RS-S5A, RS-S5B, RS-S5C), the gate is the named test-file command + named validation command (`grep` guard, `terraform validate`, `docker build`) — no per-task Python coverage is claimed, and this carve-out is stated per task, never silently omitted. Full MVP check = `pytest forecast_ml_framework/tests/test_rs_r1_feasibility.py forecast_ml_framework/tests/test_rs_r3_pins.py forecast_ml_framework/tests/test_rs_s2_chunked_reader.py -q` (3 test files; deferred tasks' test files join only when their ADR is APPROVED).
- Frozen-contract read-only: workers must not edit `.opencode/plans/contracts.md`, `.opencode/context/decisions.md`, `pipelines/data_prep/fitted.py`, `pipelines/modeling/envelope.py`, `pipelines/modeling/data_split.py` (`temporal_split` + `split_meta` shape), `governance/scrub.py` (sole leaf), `governance/pii.py`, `governance/retention.py`, `governance/erasure.py`, `core/artifact_store.py` (allowlist), `conf/base/catalog.yml` grouping, or any P1/P2-owned file listed in § File footprint matrix. Consume those surfaces read-only; if a task appears to need a frozen-semantic change, stop and raise `BLOCKED:` to Lead (new ADR + gate required).
- **No S1 in MVP:** no Spark GBM estimator, no `SparkGBMAdapter`, no Spark pickle in `best_model` envelope, no `model_classes` change, no loader dispatch change in MVP. S1 is charter-only (RS-S1D) until ADR `D-SPARK-GBM-1` is APPROVED as a separate phase.
- **R2 doc-only:** no `dvc init`, no `dvc.yaml`, no `.dvc` directory, no remote, no hook, no DVC execution code in any MVP task. P1-F verdict `DEFER` stands until ADR `D-DVC-EXEC-1` is APPROVED + `program.md` amendment + Lead re-plan.
- **No `split_meta` mutation:** no MVP task adds, removes, or renames any `split_meta` field (including any `data_hash`) in code. The `data_hash` freeze is a `solution-architect` ADR (RS-R2D charter records the required freeze inputs; implementation is a later ADR-gated task, not MVP).
- `governance/scrub.py` is the sole scrub/hash leaf (stdlib-only). No new module may import `hashlib` for scrubbing/subject-hashing or run regex masking (`re.sub`/`re.compile` for PII). The RS-R2D + RS-R4 grep guard scans every new module in this plan for `hashlib.sha256`, `import hashlib`, `re.sub`, `re.compile` — any hit outside `scrub.py` fails the gate.
- Pickle boundary: never `pickle.load` — always `restricted_loads`. No Spark pickle may be added to `_SAFE_GLOBALS` in MVP.
- Secrets from env only (`GOOGLE_APPLICATION_CREDENTIALS`, `AWS_PROFILE`, `MODEL_SHA256`); no secret value in YAML/params/IaC/docs. Rotation out-of-band via cloud (D-SECRETS-PII-GDPR-1 §5.2(e)); residency region-agnostic, no residency pin in this plan.
- File-disjoint: no file appears in two tasks' `Owns:` (see § File footprint matrix). Parallel workers run in separate worktrees; merge via Lead / `devops-sre`. Shared-file collisions are planning bugs — if you need another task's file, stop and raise `BLOCKED:` to Lead; do not edit it.
- **No clear-ID logging:** every log/audit/detail string that carries entity/subject/actor must carry `scrub.subject_hash(value)` (64-hex), never the clear value.
- No production promotion: completing MVP proves repro/scale *readiness evidence* only; it does not authorize production traffic, model promotion, or release per D-MVP-READINESS-1. Production promotion is P11 only.

## Review Focus (five failure modes most likely to bite — each pinned to an owning task's test)

1. **R1 claims bit-identical determinism that AQE/shuffle cannot deliver** (multi-partition Spark shuffle is order-nondeterministic; AQE-on changes partitioning silently). → RS-R1F: feasibility measurement runs the frozen config N=5 times, records variance, and only then decides the claim (bit-identical vs allclose); `spark.sql.adaptive.enabled=false` + frozen `spark.sql.shuffle.partitions` asserted in test.
2. **S1 Spark GBM silently breaks best_model v2** (Spark estimator has no `classes_`, its pickle globals are not in `_SAFE_GLOBALS`, loader dispatch assumes sklearn). → RS-S1D: charter-only in this plan; adapter name, pickletools residual globals, `model_classes` semantics, and loader RED→GREEN are specified as ADR-gated future work — no MVP code path can emit or load a Spark model.
3. **DVC exec pre-empts DEFER and mutates `split_meta`** (a `dvc.yaml`/remote lands, or a worker adds `split_meta.data_hash` in code without an ADR, breaking the frozen split contract + erasure/history semantics). → RS-R2D: doc-only charter; `D-DVC-EXEC-1` + `data_hash` freeze inputs recorded; grep guards assert no `dvc.yaml`, no `.dvc`, no `data_hash` code mutation.
4. **Chunked reader silently drops late rows or diverges from P1-B2 watermark** (late rows dropped instead of routed; watermark recomputed differently per caller). → RS-S2C: `filter_late_rows` invariant (`len(on_time)+len(late)==len(df)`) + watermark formula frozen to P1-B2 (`max_event_time - allowed_lateness`, strict `<`); S3 PIT parity test later proves partitioned reads match.
5. **Pins look pinned but drift** (floating base tag, unrecorded uv/Java/Spark versions, lockfile re-resolve, network fetch at build). → RS-R3: base digest (`image@sha256:…`), uv version, ubuntu runner, Java, Spark tarball sha256, `uv.lock` frozen-check, and no-network build all asserted in `test_rs_r3_pins.py`; R4 manifest later reads these pins (never re-invents them).

---

## Problem

P0 froze *what* the contracts are and P1 implemented them, P2 delivered platform readiness, but repro/scale was rejected: S1 proposed a Spark GBM estimator that breaks the frozen `best_model` v2 envelope (no `classes_` semantics, no allowlist entries, no loader dispatch), R1 asserted AQE-off bit-identical determinism without measurement, and R2 proposed DVC execution that pre-empts the P1-F `DEFER` verdict and mutates frozen `split_meta` without an ADR. Without this revised plan, the team either ships an unproven determinism claim or re-litigates frozen contracts per caller.

## User stories (Lead / downstream consumers)

- As a **Lead**, I want MVP scoped to R1-feasibility + R3-pins + S2-contract with file-disjoint tasks and machine-checkable Done so I can dispatch MVP workers in parallel without merge collisions and without S1/R2 blocking the cut, so that MVP entry is never blocked by a deferred ADR.
- As a **reproducibility reviewer**, I want an R1 feasibility measurement that freezes shuffle/AQE/Spark/Java and *then* decides the determinism claim from measured variance so that no bit-identical claim ships unproven, so that tolerance (allclose metrics, exact hash only for the single-threaded reference) is evidence, not assertion.
- As a **security/GDPR reviewer**, I want R2 doc-only until `D-DVC-EXEC-1` and `split_meta.data_hash` frozen by `solution-architect` (field/type/canonicalization over exact bytes, lineage/MLflow propagation, scrub sole-leaf grep guard) so that DVC history/erasure and frozen split semantics are never mutated by a repro/scale worker, so that erasure remains honest.
- As a **data platform operator**, I want a chunked Parquet reader contract (signature/chunksize/catalog key/watermark) plus a partition-key + PIT parity test defined so that streaming scale reads are watermark-consistent with P1-B2 and point-in-time correct, so that scale never silently drops late rows or leaks future features.

## Scope IN / OUT (explicit — no other scope may be added without a Lead decision)

### IN — MVP v1 (exactly these three tasks)

- **R1-feasibility (RS-R1F):** deterministic shuffle + AQE-off + Spark 3.5.x + Java freeze, tolerance definition (allclose metrics; exact hash only single-threaded reference), `PYTHONHASHSEED`/numpy/Spark seed read sites pinned for the R4 manifest, and a measurement harness that runs N times and decides the claim from data. No bit-identical claim is made before measurement.
- **R3-pins (RS-R3):** reproducible environment pins — base image digest, uv version, ubuntu runner, Java version, Spark tarball checksum, `uv.lock` frozen check, no-network build — each asserted by test.
- **S2-contract (RS-S2C):** chunked Parquet reader contract — exact function signature, `chunksize`, Parquet catalog key, watermark semantics aligned to P1-B2 — as one new pure module + tests. No streaming runtime, no Kafka/Kinesis, no cluster service.

### IN — Deferred / ADR-gated (defined in this plan, NOT dispatched in MVP)

- **S1 Spark GBM (RS-S1D, charter-only):** adapter name, exact pickle globals via pickletools, `model_classes` semantics, loader RED→GREEN — all specified in a deferred charter doc; implementation requires ADR `D-SPARK-GBM-1` as a separate phase. Not in MVP.
- **R2 DVC (RS-R2D, doc-only):** `D-DVC-EXEC-1` ADR inputs (backend GCS/S3/local, residency, erasure/history, cost, owner, reversal) + `split_meta.data_hash` freeze inputs owned by `solution-architect`. No DVC execution, no `split_meta` code mutation in MVP.
- **R4 manifest (RS-R4, Wave 1):** reproducibility manifest consuming RS-R1F seed read sites + RS-R3 pins. Depends On RS-R1F + RS-R3 GREEN; not part of the MVP cut.
- **S3 partition + PIT (RS-S3, Wave 1):** partition key + PIT parity test. Depends On RS-S2C GREEN; not part of the MVP cut.
- **S5 split (RS-S5A/B/C, each ADR-frozen):** HPA load test / MLflow Postgres IaC / per-principal bucket design — three separate ADR-gated tasks, each frozen by its own ADR before dispatch. None in MVP.

### OUT (this plan deliberately does not do this)

- No Spark GBM estimator code, no `SparkGBMAdapter`, no Spark pickle allowlist entries, no `best_model` envelope change, no fitted-registry change, no serving/loader change (all S1 — deferred to `D-SPARK-GBM-1` phase).
- No DVC execution: no `dvc init`, no `dvc.yaml`, no `.dvc` files, no remote, no hook, no pipeline wiring (all R2 — doc-only until `D-DVC-EXEC-1`).
- No `split_meta.data_hash` code (field/type/canonicalization frozen by `solution-architect` ADR; this plan records freeze inputs only).
- No bit-identical determinism claim in MVP: RS-R1F measures first; the claim (bit-identical vs allclose) is decided from measured variance, and exact-hash applies only to the single-threaded reference run regardless.
- No streaming platform (no Kafka/Kinesis, no watermark *service*, no scheduler wiring), no HPA/MLflow-IaC/bucket provisioning in MVP (S5 — each ADR-frozen separately).
- No IaC / cloud provisioning in MVP (no Terraform/Pulumi, no bucket/secret-manager/region provisioning) — S5B/S5C IaC is deferred and ADR-gated.
- No ONNX InferenceSession/adapter, no `best_model_onnx` catalog entry (deferred to P10 per D-CATALOG-V2-1).
- No production traffic, model promotion, or production claims (readiness ≠ promotion per D-MVP-READINESS-1; production promotion is P11 only).
- No `src/` edits in *this planning task* (`repro-scale.md` only); workers edit only inside their own `Owns:`.

### ADR-gated scope (each item: ADR + program.md amendment required before dispatch; if ADR not APPROVED the task is deferred without blocking MVP)

| Item | Tasks | ADR required | Program amendment | `Assumption:` label below |
|---|---|---|---|---|
| S1 Spark GBM implementation | RS-S1D (charter) → future impl | `D-SPARK-GBM-1` (adapter name, pickletools globals, `model_classes`, loader RED→GREEN) | `program.md` P5/P10 scope line (separate phase) | `Assumption:` adapter name + `label_set` source (see § Assumptions) |
| R2 DVC execution | RS-R2D (doc-only) → future exec | `D-DVC-EXEC-1` (backend, residency, erasure/history, cost, owner, reversal) | `program.md` P2/P9 scope line | `Assumption:` DVC verdict stays `DEFER` |
| `split_meta.data_hash` freeze | RS-R2D (freeze inputs) → impl | `D-SPLIT-DATA-HASH-1` (field/type/canonicalization, lineage/MLflow, scrub guard) owned by `solution-architect` | `program.md` P3 scope line | `Assumption:` `data_hash` shape (see § Assumptions) |
| R4 manifest | RS-R4 | `D-REPRO-MANIFEST-1` (manifest schema + seed/pin read sites) | `program.md` P9 scope line | `Assumption:` manifest storage (caller-persists) |
| S3 partition + PIT | RS-S3 | `D-PARTITION-PIT-1` (partition key + PIT parity semantics) | `program.md` P3/P4 scope line | `Assumption:` partition key (see § Assumptions) |
| S5 HPA load test | RS-S5A | `D-S5A-HPA-LOAD-1` (load thresholds + SLOs) | `program.md` P2/P10 scope line | `Assumption:` load thresholds |
| S5 MLflow Postgres IaC | RS-S5B | `D-S5B-MLFLOW-PG-1` (Postgres backend + IaC) | `program.md` P2 scope line | `Assumption:` MLflow backend (see § Assumptions) |
| S5 per-principal buckets | RS-S5C | `D-S5C-BUCKET-1` (bucket layout + ACLs) | `program.md` P2 scope line | `Assumption:` bucket layout |

- **Scope-creep resolution (per item, one choice each):** S1/R2/S5 are **kept but ADR-gated and deferred** (not removed, not in MVP) — each task's **Spec** header repeats its ADR requirement and its **`Assumption:`** line states the default if the ADR does not land. No deferred item ships without its ADR + anti-team gate. MVP ships only the three IN tasks above.

---

## MVP v1 redefined (frozen cut for this plan)

> **MVP v1 = RS-R1F (R1-feasibility) + RS-R3 (R3-pins) + RS-S2C (S2-contract).**
> - **No S1:** RS-S1D is charter-only; no Spark GBM code, no envelope/fitted/serving change.
> - **R2 doc-only:** RS-R2D records `D-DVC-EXEC-1` + `data_hash` freeze inputs; no DVC execution, no `split_meta` mutation.
> - **Not in MVP cut:** RS-R4 (Wave 1), RS-S3 (Wave 1), RS-S5A/B/C (ADR-gated) — defined here, dispatched only after MVP GREEN + their ADR APPROVED.
> - MVP exit = the 3-task full-suite command in § Global Constraints green + anti-team `APPROVED` on this plan + no `src/` edit outside the 2 new MVP modules + no `dvc.yaml`/`.dvc`/`split_meta` mutation (grep guards green).

## Entry criteria checklist (all must be checked before any MVP worker starts)

- [ ] P0 freeze recorded: `.opencode/plans/contracts.md` FIX4 dual APPROVED (reviewer PASS + anti-team APPROVED), 2026-11-08.
- [ ] `.opencode/context/decisions.md` — all 7 ADRs `frozen` (D-README-1, D-MVP-READINESS-1, D-BEST-MODEL-V2-1, D-FITTED-REG-V2-1, D-TEMPORAL-SPLIT-1, D-SECRETS-PII-GDPR-1, D-CATALOG-V2-1).
- [ ] P1 exit evidence available read-only: P1-B2 watermark semantics (`assign_time_columns`/`watermark_cutoff`/`filter_late_rows`), P1-D1 scrub sole leaf, P1-F DVC `DEFER` (`P1-dvc-decision.md`). Lead confirms no frozen-semantic edit since P1.
- [ ] Anti-team-arch gate on **this plan** APPROVED (critique log `.opencode/critique-log/repro-scale-cycle-N.md`) before dispatch. This plan proposes; only Lead approves.
- [ ] ADR-gated/deferred tasks (RS-S1D impl, RS-R2D exec, RS-R4, RS-S3, RS-S5A/B/C) dispatch only after their ADR in § ADR-gated scope above is APPROVED; MVP tasks (RS-R1F/RS-R3/RS-S2C) are not blocked by those ADRs.

**Assumptions (every line labeled `Assumption:` — workers plan against these, not additional scope):**

- `Assumption:` **Spark version** — repro env narrows to Spark 3.5.x (`pyspark>=3.5,<4` in the repro harness/pins, even though `pyproject.toml:61` allows `pyspark>=3.5,<5` for lazy use). **Java** is 17 LTS (the 3.5.x LTS pairing); the exact base-image digest + Spark tarball sha256 are *recorded* by RS-R3 from the real image/tarball (this plan invents no digest value).
- `Assumption:` **Deterministic shuffle** uses a frozen seed (`random_state=42`, matching `temporal_split` default) + frozen `spark.sql.shuffle.partitions=200` + `spark.sql.adaptive.enabled=false` (AQE-off). The exact partition count is frozen by RS-R1F and read back by RS-R4; changing it is a new measurement, not a config tweak.
- `Assumption:` **Tolerance** — metrics compared with `np.allclose(rtol=1e-6, atol=1e-8)`; exact hash equality applies *only* to the single-threaded reference run (`spark.sql.shuffle.partitions=1`, single executor, single thread). Multi-partition runs never claim exact-hash equality.
- `Assumption:` **Seed read sites for R4 manifest** — exactly three: (1) `os.environ['PYTHONHASHSEED']` (must be `'0'`, set before interpreter start), (2) numpy seed passed to `np.random.default_rng(seed)` in the feasibility harness (default `42`), (3) Spark conf dict read from the feasibility config (`spark.sql.shuffle.partitions`, `spark.sql.adaptive.enabled`, Spark seed keys as frozen by RS-R1F). RS-R4 reads these sites; it never re-invents a seed.
- `Assumption:` **S1 adapter name** — future adapter is `SparkGBMAdapter` (in `pipelines/modeling/adapters.py`, mirroring `ModelAdapter`); **S1 `label_set` source** is an explicit `label_set: list[str]` param (Spark GBM exposes no `classes_`), validated against proba width — never derived from a nonexistent `classes_`. Placeholder names only; RS-S1D charter must confirm or replace them under `D-SPARK-GBM-1`.
- `Assumption:` **`split_meta.data_hash`** — `str` (64-hex sha256), canonicalization = `hashlib.sha256(json.dumps({train/test canonical rows}, sort_keys=True, separators=(',',':')).encode('utf-8')).hexdigest()` over exact bytes defined by `solution-architect` under `D-SPLIT-DATA-HASH-1`; this plan records the freeze *inputs*, not the frozen bytes. The hashing itself runs through the `scrub` sole leaf (no second `hashlib` in new modules).
- `Assumption:` **S2 Parquet catalog key** — `raw_events_parquet` (new key name, additive; no existing key renamed). **S2 chunksize** default `100_000` rows. **S3 partition key** — `event_date` = UTC date of `event_time`, Hive-style `event_date=YYYY-MM-DD/` directory segments.
- `Assumption:` **R3 runner** — `ubuntu-24.04` GitHub Actions runner; **uv version** is recorded from the real `uv --version` output in the pins file (this plan invents no version number); **no-network build** = `docker build --network=none` + `uv sync --frozen --locked --no-index` (lockfile check via `--frozen --locked`).
- `Assumption:` **S5 MLflow backend** — Postgres (not SQLite file) for scale; **S5 bucket layout** — per-principal buckets (one bucket per principal, principal = OIDC `sub` hash prefix, never clear ID). Both are ADR defaults to be frozen by `D-S5B-MLFLOW-PG-1` / `D-S5C-BUCKET-1`, not MVP decisions.

## File footprint matrix (proves file-disjointness — every file in exactly one Owns)

| File (repo-relative) | Task | Group | MVP? |
|---|---|---|---|
| `forecast_ml_framework/src/forecast_ml_framework/repro/feasibility.py` (new) | RS-R1F | R1 | ✅ MVP |
| `forecast_ml_framework/tests/test_rs_r1_feasibility.py` (new) | RS-R1F | R1 | ✅ MVP |
| `forecast_ml_framework/Dockerfile.repro` (new) | RS-R3 | R3 | ✅ MVP |
| `forecast_ml_framework/ci/repro-pins.txt` (new) | RS-R3 | R3 | ✅ MVP |
| `forecast_ml_framework/ci/build-repro.sh` (new) | RS-R3 | R3 | ✅ MVP |
| `forecast_ml_framework/tests/test_rs_r3_pins.py` (new) | RS-R3 | R3 | ✅ MVP |
| `forecast_ml_framework/src/forecast_ml_framework/pipelines/data_prep/chunked_reader.py` (new) | RS-S2C | S2 | ✅ MVP |
| `forecast_ml_framework/tests/test_rs_s2_chunked_reader.py` (new) | RS-S2C | S2 | ✅ MVP |
| `forecast_ml_framework/src/forecast_ml_framework/repro/manifest.py` (new) | RS-R4 | R4 | Wave 1 |
| `forecast_ml_framework/tests/test_rs_r4_manifest.py` (new) | RS-R4 | R4 | Wave 1 |
| `forecast_ml_framework/src/forecast_ml_framework/pipelines/data_prep/partitioning.py` (new) | RS-S3 | S3 | Wave 1 |
| `forecast_ml_framework/tests/test_rs_s3_partition_pit.py` (new) | RS-S3 | S3 | Wave 1 |
| `.opencode/plans/s1-spark-gbm-deferred.md` (new) | RS-S1D | S1 | Deferred |
| `forecast_ml_framework/tests/test_rs_s1_deferred.py` (new) | RS-S1D | S1 | Deferred |
| `.opencode/plans/r2-dvc-doc-only.md` (new) | RS-R2D | R2 | Deferred |
| `forecast_ml_framework/tests/test_rs_r2_doc_only.py` (new) | RS-R2D | R2 | Deferred |
| `forecast_ml_framework/deploy/load/hpa-load-test.md` (new) | RS-S5A | S5 | Deferred |
| `forecast_ml_framework/tests/test_rs_s5a_load.py` (new) | RS-S5A | S5 | Deferred |
| `forecast_ml_framework/infra/mlflow-postgres/main.tf` (new) | RS-S5B | S5 | Deferred |
| `forecast_ml_framework/infra/mlflow-postgres/variables.tf` (new) | RS-S5B | S5 | Deferred |
| `forecast_ml_framework/tests/test_rs_s5b_mlflow_pg.py` (new) | RS-S5B | S5 | Deferred |
| `.opencode/plans/s5c-bucket-design.md` (new) | RS-S5C | S5 | Deferred |
| `forecast_ml_framework/tests/test_rs_s5c_buckets.py` (new) | RS-S5C | S5 | Deferred |

No other file may be edited by any repro/scale worker without a Lead-approved re-plan (new ADR if semantics change). In particular: `forecast_ml_framework/Dockerfile` (P2-C1), `.github/workflows/forecast-ml-ci.yml` + `forecast_ml_framework/ci/run_checks.sh` (P2-R1), `conf/base/catalog.yml` (P1-E), `pipelines/modeling/adapters.py` + `envelope.py` + `serving/model_loader.py` + `core/artifact_store.py` (P1-A2/P1-D1), `pipelines/modeling/data_split.py` (P1-B1), `governance/scrub.py` (P1-D1), and `.opencode/plans/contracts.md` / `.opencode/context/decisions.md` are **never edited** in this plan.

## Task order / parallelism

| Wave | Tasks | Parallel? | Depends On |
|---|---|---|---|
| **Wave 0 — MVP (no inter-dependency)** | RS-R1F (feasibility) · RS-R3 (pins) · RS-S2C (chunked-reader contract) | All 3 parallel (disjoint files) | P0+P1 frozen only |
| **Wave 1 — post-MVP consumers** | RS-R4 `Depends On` RS-R1F + RS-R3 (seed read sites + pins must be GREEN before manifest reads them) · RS-S3 `Depends On` RS-S2C (reader contract must be GREEN before partition/PIT) | RS-R4 ∥ RS-S3 (disjoint files) | As listed; ADR-gated (see § ADR-gated scope) |
| **Wave D — deferred (ADR-gated, NOT MVP)** | RS-S1D (S1 charter) · RS-R2D (R2 doc-only) · RS-S5A (HPA load) · RS-S5B (MLflow PG) · RS-S5C (buckets) | Charter/doc tasks may run anytime (disjoint); implementation dispatch only after each ADR APPROVED | Each task's ADR (see table) |

**Risks sized first:** the riskiest unknown (R1 determinism variance) is in Wave 0 (RS-R1F) so failure surfaces before R4 manifest or any bit-identical claim is built on top. No task is a giant frontend+backend+data step; each ends with an independently testable deliverable.

---

## Tasks (file-disjoint, TDD, ≥85% coverage where applicable, each <1–2h equivalent)

> **How to read each task:** `Owns:` = exact paths the worker may touch (exclusive). `Depends On:` = must be GREEN first. `Spec:` = exact signatures/types/semantics. `Acceptance:` = machine-checkable Done (command + output). `TDD evidence:` = RED→GREEN trail the worker must leave. `Anti gate:` = must be `APPROVED` before Lead accepts.

### RS-R1F — R1 feasibility measurement: deterministic shuffle / AQE-off / Spark 3.5.x+Java (MVP, Wave 0)

**Owns:**
- `forecast_ml_framework/src/forecast_ml_framework/repro/feasibility.py` (new)
- `forecast_ml_framework/tests/test_rs_r1_feasibility.py` (new)

**Depends On:** P0+P1 frozen only (parallel with RS-R3/RS-S2C).

**Spec (measurement first — claim decided from data, never asserted upfront):**

```python
# feasibility.py — deterministic repro config + variance measurement harness.
REPRO_CONFIG: dict = {
  'PYTHONHASHSEED': '0',                       # read site 1 (env, must be set pre-interpreter)
  'numpy_seed': 42,                            # read site 2 (np.random.default_rng(42))
  'spark_conf': {
    'spark.sql.adaptive.enabled': 'false',     # AQE-off (frozen)
    'spark.sql.shuffle.partitions': '200',     # deterministic shuffle (frozen)
    # Spark seed keys as frozen by this task (recorded verbatim for RS-R4)
  },
  'spark_version': '3.5.x',                    # narrowed from pyproject pyspark>=3.5,<5
  'java_version': '17',                        # LTS pairing (exact digest in RS-R3 pins)
  'random_state': 42,                          # matches temporal_split default
  'single_threaded_reference': {'spark.sql.shuffle.partitions': '1'},
}
def run_repro_once(df: pd.DataFrame, config: dict = REPRO_CONFIG) -> dict:
  """Run ONE deterministic shuffle+aggregate pass under config; return
  {'metrics': dict, 'hash': str} where hash is sha256 over canonical metric bytes.
  Uses np.random.default_rng(config['numpy_seed']) and the frozen spark_conf
  (when Spark present); never reads a seed from anywhere else."""
def measure_variance(df: pd.DataFrame, *, n_runs: int = 5, config: dict = REPRO_CONFIG) -> dict:
  """Run run_repro_once n_runs times; return {'hashes': list[str], 'metrics_list': list[dict],
  'bit_identical': bool, 'allclose': bool} where bit_identical = all hashes equal,
  allclose = every metrics pair np.allclose(rtol=1e-6, atol=1e-8). The CLAIM is decided
  from this output: bit_identical only if measured True; otherwise allclose-only."""
```

Tolerance (frozen): metrics `np.allclose(rtol=1e-6, atol=1e-8)`; exact hash equality only for the single-threaded reference config (`shuffle.partitions=1`). Multi-partition runs never claim exact-hash.
`Assumption:` Spark 3.5.x + Java 17 + shuffle/AQE values per § Assumptions (worker records the exact Spark/Java versions from the real env; plan invents no digest).

**Acceptance (all must pass):**
- [ ] `pytest forecast_ml_framework/tests/test_rs_r1_feasibility.py -q` — asserts: `REPRO_CONFIG['spark_conf']['spark.sql.adaptive.enabled']=='false'`; `REPRO_CONFIG['PYTHONHASHSEED']=='0'`; `numpy_seed==42`; `measure_variance(..., n_runs=5)` returns hashes/metrics with `bit_identical`/`allclose` computed (not hardcoded); single-threaded reference config has `shuffle.partitions=='1'`; metric comparison uses `rtol=1e-6, atol=1e-8` (asserted via a perturbed-metrics fixture that fails allclose).
- [ ] `pytest --cov=forecast_ml_framework.repro.feasibility --cov-fail-under=85 forecast_ml_framework/tests/test_rs_r1_feasibility.py -q` passes (single gate, § Global Constraints; module `forecast_ml_framework.repro.feasibility`, ≥85%).
- [ ] Seed read-site test: RS-R4 contract — the three read sites (env `PYTHONHASHSEED`, `numpy_seed` param, `spark_conf` dict) are present and are the *only* seed sources (grep: no other `random_state`/`default_rng` seed literal in `feasibility.py`).

**TDD evidence required:** `test_rs_r1_feasibility.py` RED commit (fails with `ImportError` for missing `repro.feasibility`) → GREEN commit implementing § Spec. Worker leaves both commits (history must show RED failure log in handoff note).
**Anti gate:** `anti-team-arch` (determinism semantics: AQE-off, shuffle freeze, tolerance, measure-then-claim). Max 3 cycles, then deadlock protocol to Lead.
**Estimate:** 1.5h (config 0.25h, harness 0.75h, variance tests 0.5h).

### RS-R3 — R3 reproducible environment pins (MVP, Wave 0)

**Owns:**
- `forecast_ml_framework/Dockerfile.repro` (new)
- `forecast_ml_framework/ci/repro-pins.txt` (new)
- `forecast_ml_framework/ci/build-repro.sh` (new)
- `forecast_ml_framework/tests/test_rs_r3_pins.py` (new)

**Depends On:** P0+P1 frozen only (parallel with RS-R1F/RS-S2C).

**Spec:** `Dockerfile.repro` — multi-stage repro image: `FROM ubuntu:24.04@sha256:<digest>` (digest recorded from the real base image; plan invents none), installs Java 17 LTS, copies `pyproject.toml`+`uv.lock`, runs `uv sync --frozen --locked --no-dev`, copies `src/`+`conf/`. `ci/repro-pins.txt` — one `key=value` per line, exact keys: `BASE_DIGEST`, `UV_VERSION`, `RUNNER=ubuntu-24.04`, `JAVA_VERSION=17`, `SPARK_VERSION=3.5.x`, `SPARK_TARBALL_SHA256`, `PYTHON_VERSION`. `ci/build-repro.sh` — `set -euo pipefail`; builds with `docker build --network=none -f Dockerfile.repro` and verifies `uv sync --frozen --locked` + `uv.lock` unchanged (`git diff --exit-code uv.lock`).
`Assumption:` runner `ubuntu-24.04`, Java 17, Spark 3.5.x per § Assumptions; worker records real digest/version/checksum values.

**Acceptance (all must pass):**
- [ ] `pytest forecast_ml_framework/tests/test_rs_r3_pins.py -q` — asserts: `repro-pins.txt` parses with all 7 keys present; `BASE_DIGEST` matches `sha256:[0-9a-f]{64}`; `RUNNER=='ubuntu-24.04'`; `JAVA_VERSION=='17'`; `SPARK_VERSION` starts with `3.5.`; `Dockerfile.repro` contains the pinned digest + `uv sync --frozen --locked`; `build-repro.sh` contains `--network=none` + `--frozen --locked`.
- [ ] `bash forecast_ml_framework/ci/build-repro.sh` (or its `--network=none` fallback evidence in handoff) exits 0 and `git diff --exit-code forecast_ml_framework/uv.lock` is empty (lockfile unchanged).
- [ ] No-network proof: `grep -c 'network=none' forecast_ml_framework/ci/build-repro.sh` ≥ 1; `grep -rn 'http' forecast_ml_framework/Dockerfile.repro` returns only the base-image ref (no curl/wget fetch at build).

**TDD evidence required:** RED (fails with `FileNotFoundError` for missing `repro-pins.txt`) → GREEN (pins + Dockerfile.repro + script + test).
**Coverage/validation gate:** No new Python production module (pins/script/Dockerfile + test file only) → no pytest-cov gate. Validation gate = `test_rs_r3_pins.py` green + `build-repro.sh` exit 0. (Carve-out stated here per § Global Constraints; never silently omitted.)
**Anti gate:** `anti-team-arch` (pin completeness: digest/uv/runner/Java/Spark/lock/no-network). Max 3 cycles.
**Estimate:** 1h (pins 0.25h, Dockerfile.repro 0.25h, script+test 0.5h).

### RS-S2C — S2 streaming chunked reader contract (MVP, Wave 0)

**Owns:**
- `forecast_ml_framework/src/forecast_ml_framework/pipelines/data_prep/chunked_reader.py` (new)
- `forecast_ml_framework/tests/test_rs_s2_chunked_reader.py` (new)

**Depends On:** P0+P1 frozen only (parallel; consumes P1-B2 watermark semantics read-only, never edits `streaming.py`).

**Spec (P1-B2-aligned contract; Parquet catalog key + watermark frozen here):**

```python
# chunked_reader.py — chunked Parquet reader contract (catalog key + watermark).
CATALOG_KEY = 'raw_events_parquet'   # Parquet catalog key (additive; no key renamed)
DEFAULT_CHUNKSIZE = 100_000          # rows per chunk (int, >0 else ValueError)
def read_parquet_chunks(path: str, *, chunksize: int = DEFAULT_CHUNKSIZE,
                        columns: list[str] | None = None,
                        filters: list[tuple] | None = None) -> Iterator[pd.DataFrame]:
  """Yield Parquet chunks via pyarrow.parquet.ParquetFile.iter_batches
  (chunked reader; never loads whole file). columns/filters passed through to
  iter_batches (projection/predicate pushdown). Empty file -> zero chunks (no raise)."""
def watermark_cutoff_for_chunks(max_event_time: pd.Timestamp, *,
                                allowed_lateness: pd.Timedelta) -> pd.Timestamp:
  """watermark = max_event_time - allowed_lateness (P1-B2 semantics; negative
  lateness -> ValueError). Late row = event_time < watermark (strict <)."""
def split_late_chunks(df: pd.DataFrame, *, watermark: pd.Timestamp,
                      event_time_col: str = 'event_time') -> tuple[pd.DataFrame, pd.DataFrame]:
  """Split into (on_time, late) with invariant len(on_time)+len(late)==len(df);
  late rows are ROUTED (returned), never dropped silently."""
```

`Assumption:` catalog key `raw_events_parquet` + chunksize `100_000` per § Assumptions. Watermark semantics identical to P1-B2 (`watermark_cutoff`/`filter_late_rows`); this module must not re-define different semantics.

**Acceptance (all must pass):**
- [ ] `pytest forecast_ml_framework/tests/test_rs_s2_chunked_reader.py -q` — asserts: `CATALOG_KEY=='raw_events_parquet'`; signature defaults (`chunksize==100_000`); chunked read of a 250k-row Parquet fixture yields exactly `ceil(250k/chunksize)` chunks with row-count sum == input; `columns` projection respected; `split_late_chunks` invariant on a fixture with 3 late rows; watermark math (`max - lateness`); `chunksize=0` raises `ValueError`; empty file yields zero chunks.
- [ ] `pytest --cov=forecast_ml_framework.pipelines.data_prep.chunked_reader --cov-fail-under=85 forecast_ml_framework/tests/test_rs_s2_chunked_reader.py -q` passes (single gate, § Global Constraints; ≥85%).
- [ ] Grep guard in test: no `kafka*`/`kinesis`/`pyspark*streaming*` import in `chunked_reader.py` (no streaming runtime); no `hashlib`/`re.sub` (scrub sole leaf).

**TDD evidence:** RED (import fails / no chunking) → GREEN.
**Anti gate:** `anti-team-arch` (reader signature, chunksize, catalog key, watermark semantics). Max 3 cycles.
**Estimate:** 1h.

### RS-R4 — R4 reproducibility manifest (Wave 1, ADR-gated)

**Owns:**
- `forecast_ml_framework/src/forecast_ml_framework/repro/manifest.py` (new)
- `forecast_ml_framework/tests/test_rs_r4_manifest.py` (new)

**Depends On:** RS-R1F + RS-R3 GREEN (seed read sites + pins must land first). **ADR-gated:** dispatch requires `D-REPRO-MANIFEST-1` APPROVED + `program.md` amendment. If ADR not APPROVED, this task is deferred (MVP not blocked).
**Spec:**
```python
def build_manifest(config: dict | None = None) -> dict:
  """Return manifest dict with exactly: repro_config (RS-R1F REPRO_CONFIG),
  seed_read_sites {pythonhashseed: os.environ['PYTHONHASHSEED'], numpy_seed: int,
  spark_conf: dict} (read from RS-R1F sites, never re-invented),
  pins (parsed from RS-R3 repro-pins.txt: base_digest, uv_version, runner,
  java_version, spark_version, spark_tarball_sha256), tolerance {rtol: 1e-6, atol: 1e-8}.
  Caller persists the manifest (catalog/MLflow lineage); this function writes nothing."""
```
`Assumption:` manifest storage is caller-persists (see § Assumptions); `split_meta.data_hash` propagates into the manifest only after `D-SPLIT-DATA-HASH-1` is frozen (RS-R2D).

**Acceptance (all must pass):**
- [ ] `pytest forecast_ml_framework/tests/test_rs_r4_manifest.py -q` — manifest contains the 3 seed read sites + all RS-R3 pin keys; tolerance `rtol/atol` exact; manifest is JSON-serializable (`json.dumps` round-trip).
- [ ] `pytest --cov=forecast_ml_framework.repro.manifest --cov-fail-under=85 forecast_ml_framework/tests/test_rs_r4_manifest.py -q` passes (single gate; ≥85%).
- [ ] Grep guard: no `hashlib`/`re.sub` in `manifest.py` (scrub sole leaf); no seed literal outside the RS-R1F read sites.

**TDD evidence:** RED → GREEN. **Anti gate:** `anti-team-arch`. **Estimate:** 1h.

### RS-S3 — S3 partition key + PIT parity test (Wave 1, ADR-gated)

**Owns:**
- `forecast_ml_framework/src/forecast_ml_framework/pipelines/data_prep/partitioning.py` (new)
- `forecast_ml_framework/tests/test_rs_s3_partition_pit.py` (new)

**Depends On:** RS-S2C GREEN (reader contract first). **ADR-gated:** dispatch requires `D-PARTITION-PIT-1` APPROVED + `program.md` amendment.
**Spec:**
```python
def partition_key(event_time: pd.Timestamp) -> str:
  """Return Hive-style partition segment 'event_date=YYYY-MM-DD' from the UTC
  date of event_time (event_time normalized via pd.to_datetime(..., utc=True);
  mixed naive/aware -> ValueError, matching P1-B2)."""
def get_pit_features_partitioned(partition_root: str, entity_ids: list[str],
                                 as_of: pd.Timestamp) -> pd.DataFrame:
  """Read only partitions with event_date <= as_of date (partition pruning) and
  return point-in-time-correct rows (event_time <= as_of). Must equal
  get_pit_features_unpartitioned on the same data (parity test below)."""
```
`Assumption:` partition key `event_date` (UTC date of `event_time`) per § Assumptions.

**Acceptance (all must pass):**
- [ ] `pytest forecast_ml_framework/tests/test_rs_s3_partition_pit.py -q` — `partition_key` format `event_date=YYYY-MM-DD`; PIT parity: partitioned `get_pit_features_partitioned` == unpartitioned reference on a fixture with rows straddling `as_of` (frame equality, including late-row fixtures); future rows (`event_time > as_of`) excluded.
- [ ] `pytest --cov=forecast_ml_framework.pipelines.data_prep.partitioning --cov-fail-under=85 forecast_ml_framework/tests/test_rs_s3_partition_pit.py -q` passes (single gate; ≥85%).

**TDD evidence:** RED → GREEN. **Anti gate:** `anti-team-arch` (partition key + PIT parity). **Estimate:** 1.5h.

### RS-S1D — S1 Spark GBM DEFERRED charter (Wave D, charter-only — NOT in MVP, ADR-gated)

**Owns:**
- `.opencode/plans/s1-spark-gbm-deferred.md` (new)
- `forecast_ml_framework/tests/test_rs_s1_deferred.py` (new)

**Depends On:** None (charter-only). **ADR-gated for implementation:** any S1 code requires ADR `D-SPARK-GBM-1` APPROVED as a separate phase + `program.md` amendment + Lead re-plan. This task produces the charter only.
**Spec:** Charter doc must specify, with exact names (no TBD): (1) adapter name (`SparkGBMAdapter`, per `Assumption:`), (2) exact pickle globals enumerated via `pickletools.genops` `GLOBAL`/`STACK_GLOBAL` scan (residual list after subtracting current `_SAFE_GLOBALS` — worker runs the scan and records the exact tuples; plan invents none), (3) `model_classes` semantics (Spark GBM has no `classes_` → explicit `label_set` param validated against proba width), (4) loader RED→GREEN (RED: `restricted_loads` rejects Spark pickle; GREEN: allowlist additions + loader dispatch for the Spark flavor only). Charter must state the frozen-contract breakage it resolves: envelope `model` raw-estimator invariant, `label_set` vs `model_classes`, `_SAFE_GLOBALS` additions, loader flavor dispatch.
**Acceptance (all must pass):**
- [ ] `pytest forecast_ml_framework/tests/test_rs_s1_deferred.py -q` — asserts charter exists and contains the 4 exact sections (adapter name, pickletools globals, `model_classes`, loader RED→GREEN); asserts pickletools residual list is non-empty and enumerated (not "TBD"); asserts `git diff --stat` shows no `src/` edit from this task.
- [ ] `grep -c 'D-SPARK-GBM-1' .opencode/plans/s1-spark-gbm-deferred.md` ≥ 1; `grep -rn 'SparkGBMAdapter\|pickletools' .opencode/plans/s1-spark-gbm-deferred.md` non-empty.

**TDD evidence:** RED (missing charter → test fails) → GREEN (charter + test). **Coverage/validation gate:** No new Python production module (charter doc + test only) → no pytest-cov gate. Validation gate = `test_rs_s1_deferred.py` green + grep guards green. (Carve-out stated here per § Global Constraints.)
**Anti gate:** `anti-team-arch` + `security` (pickle/allowlist boundary). **Estimate:** 1h (pickletools scan 0.5h, charter 0.5h).

### RS-R2D — R2 DVC doc-only + split_meta.data_hash freeze inputs (Wave D, doc-only — NOT in MVP, ADR-gated)

**Owns:**
- `.opencode/plans/r2-dvc-doc-only.md` (new)
- `forecast_ml_framework/tests/test_rs_r2_doc_only.py` (new)

**Depends On:** None (doc-only). **ADR-gated for execution:** any DVC execution requires ADR `D-DVC-EXEC-1` APPROVED + `program.md` amendment + Lead re-plan. P1-F `DEFER` stands until then.
**Spec:** Doc must record: (1) `D-DVC-EXEC-1` required fields — backend (GCS/S3/local, single choice), residency (region-agnostic per D-SECRETS-PII-GDPR-1, decision at P2 IaC before P11), erasure/history (DVC-aware deletion never requires history rewrites, never resurrects erased subjects), cost (storage+transfer estimate), owner (named), reversal (tested clean-checkout restore); (2) `split_meta.data_hash` freeze inputs owned by `solution-architect` under `D-SPLIT-DATA-HASH-1` — field `data_hash: str` (64-hex), type, canonicalization over exact bytes (`json.dumps(sort_keys=True, separators=(',',':')).encode('utf-8')` → sha256 hex), lineage/MLflow propagation (split_meta → run params/tags + lineage record), scrub sole-leaf grep guard (no second `hashlib` in new modules). Doc must state: doc-only until both ADRs APPROVED; no `dvc.yaml`, no `.dvc`, no `split_meta` code mutation.
**Acceptance (all must pass):**
- [ ] `pytest forecast_ml_framework/tests/test_rs_r2_doc_only.py -q` — asserts doc contains all 6 `D-DVC-EXEC-1` fields + all 4 `data_hash` freeze inputs (field/type/canonicalization/lineage+MLflow/scrub guard); asserts no `dvc.yaml`/`.dvc` in working tree (`glob` empty); asserts no `data_hash` code mutation (`grep -rn 'data_hash' src/` shows no new repro/scale code — only pre-existing `hooks.py` reference).
- [ ] `grep -c 'D-DVC-EXEC-1\|D-SPLIT-DATA-HASH-1' .opencode/plans/r2-dvc-doc-only.md` ≥ 2; `grep -rn 'hashlib' forecast_ml_framework/src/forecast_ml_framework/repro/` returns empty (scrub sole leaf; `scrub.py` excluded).

**TDD evidence:** RED → GREEN. **Coverage/validation gate:** No new Python production module (doc + test only) → no pytest-cov gate. Validation gate = `test_rs_r2_doc_only.py` green + grep guards green. (Carve-out stated here per § Global Constraints.)
**Anti gate:** `anti-team-arch` + `security` (residency/erasure/cost/owner/reversal; scrub guard). **Estimate:** 1h.

### RS-S5A — S5 HPA load test (Wave D, ADR-gated — NOT in MVP)

**Owns:**
- `forecast_ml_framework/deploy/load/hpa-load-test.md` (new)
- `forecast_ml_framework/tests/test_rs_s5a_load.py` (new)

**Depends On:** None for the design doc. **ADR-gated:** dispatch requires `D-S5A-HPA-LOAD-1` APPROVED (load thresholds + SLOs) + `program.md` amendment. Load-test execution is out of MVP.
**Spec:** Doc freezes: HPA min/max replicas, CPU/memory thresholds, load profile (concurrent `/predict` RPS), SLOs (p50/p95 latency, error budget), pass/fail thresholds (exact numbers, no TBD). `Assumption:` thresholds recorded by the worker from the ADR (plan invents no numbers).
**Acceptance (all must pass):**
- [ ] `pytest forecast_ml_framework/tests/test_rs_s5a_load.py -q` — asserts doc contains HPA min/max, thresholds, load profile, SLOs, pass/fail (all numeric, no "TBD" substring).
- [ ] `grep -c 'D-S5A-HPA-LOAD-1' forecast_ml_framework/deploy/load/hpa-load-test.md` ≥ 1.

**TDD evidence:** RED → GREEN. **Coverage/validation gate:** No new Python production module (doc + test only) → no pytest-cov gate. Validation gate = `test_rs_s5a_load.py` green. (Carve-out stated here per § Global Constraints.)
**Anti gate:** `anti-team-arch`. **Estimate:** 0.5h.

### RS-S5B — S5 MLflow Postgres IaC (Wave D, ADR-gated — NOT in MVP)

**Owns:**
- `forecast_ml_framework/infra/mlflow-postgres/main.tf` (new)
- `forecast_ml_framework/infra/mlflow-postgres/variables.tf` (new)
- `forecast_ml_framework/tests/test_rs_s5b_mlflow_pg.py` (new)

**Depends On:** None for the design. **ADR-gated:** dispatch requires `D-S5B-MLFLOW-PG-1` APPROVED (Postgres backend + IaC) + `program.md` amendment. No IaC apply in MVP.
**Spec:** Terraform ≥1.3; Postgres backend for MLflow (not SQLite file): `postgresql` database + user (from vars, secrets via env/Secret Manager references only — never values), connection string output (reference only). `Assumption:` Postgres backend per § Assumptions.
**Acceptance (all must pass):**
- [ ] `pytest forecast_ml_framework/tests/test_rs_s5b_mlflow_pg.py -q` — asserts `main.tf` declares a Postgres resource, `variables.tf` has no secret-value default (grep guard for secret literals empty), MLflow backend string references Postgres.
- [ ] `(cd forecast_ml_framework/infra/mlflow-postgres && terraform init -backend=false -input=false && terraform validate)` exits 0.
- [ ] `terraform fmt -check -recursive forecast_ml_framework/infra/mlflow-postgres` exits 0.

**TDD evidence:** RED (empty dir → `terraform validate` fails) → GREEN. **Coverage/validation gate:** No Python production module → no pytest-cov gate. Validation gate = `test_rs_s5b_mlflow_pg.py` green + `terraform validate` green. (Carve-out stated here per § Global Constraints.)
**Anti gate:** `anti-team-arch` + `security` (no secret in state, Postgres backend). **Estimate:** 1h.

### RS-S5C — S5 per-principal bucket design (Wave D, ADR-gated — NOT in MVP)

**Owns:**
- `.opencode/plans/s5c-bucket-design.md` (new)
- `forecast_ml_framework/tests/test_rs_s5c_buckets.py` (new)

**Depends On:** None for the design doc. **ADR-gated:** dispatch requires `D-S5C-BUCKET-1` APPROVED (bucket layout + ACLs) + `program.md` amendment. No bucket provisioning in MVP.
**Spec:** Doc freezes: per-principal bucket layout (one bucket per principal; principal key = hash prefix of OIDC `sub` via `scrub.subject_hash`, never clear ID), bucket naming, ACL/IAM policy (least-privilege, no public access), lifecycle/retention alignment with P1-D2. `Assumption:` per-principal layout per § Assumptions.
**Acceptance (all must pass):**
- [ ] `pytest forecast_ml_framework/tests/test_rs_s5c_buckets.py -q` — asserts doc contains bucket naming, per-principal layout, ACL/IAM, lifecycle; asserts no clear-ID bucket naming (grep for clear `sub` pattern empty; test asserts `subject_hash` referenced).
- [ ] `grep -c 'D-S5C-BUCKET-1' .opencode/plans/s5c-bucket-design.md` ≥ 1.

**TDD evidence:** RED → GREEN. **Coverage/validation gate:** No new Python production module (doc + test only) → no pytest-cov gate. Validation gate = `test_rs_s5c_buckets.py` green. (Carve-out stated here per § Global Constraints.)
**Anti gate:** `anti-team-arch` + `security` (per-principal isolation, no clear ID, scrub leaf). **Estimate:** 0.5h.

---

## Exit criteria (MVP v1)

- [ ] Full MVP suite green: `pytest forecast_ml_framework/tests/test_rs_r1_feasibility.py forecast_ml_framework/tests/test_rs_r3_pins.py forecast_ml_framework/tests/test_rs_s2_chunked_reader.py -q` (3 files, all pass).
- [ ] Per-task coverage gates green (RS-R1F ≥85% `repro.feasibility`, RS-S2C ≥85% `chunked_reader`; RS-R3 validation gate green with carve-out stated).
- [ ] No-S1 proof: `git diff --stat` shows no edit to `adapters.py`/`envelope.py`/`model_loader.py`/`artifact_store.py`/`fitted.py`; no Spark pickle in any MVP artifact.
- [ ] R2 doc-only proof: no `dvc.yaml`, no `.dvc`, no `split_meta.data_hash` code mutation (RS-R2D grep guards green when that task runs; MVP workers must not create them).
- [ ] Anti-team `APPROVED` on this plan (critique log `repro-scale-cycle-N.md`) before Lead accepts MVP.
- [ ] Flaw-report pass (`tester`): PASS/FAIL with file:line findings in `.opencode/flaw-reports/repro-scale-flaws.md`.

*Plan created by `product-planner` under REPRO-PLAN-FIX (anti-team REJECTED revision). Planning artifact only — no product code. MVP = R1-feasibility + R3-pins + S2-contract; S1 deferred to `D-SPARK-GBM-1` phase; R2 doc-only until `D-DVC-EXEC-1` + `D-SPLIT-DATA-HASH-1`.*
