# P0 Contracts Freeze — Catalog, Fitted Registry, Best Model, Temporal Split, Secrets/Retention/PII/GDPR

> **Owner:** `solution-architect` (this file only). **Planner owns** `.opencode/context/decisions.md`; this file **references** ADR IDs only and does not edit that file.
> **Status:** `frozen` — FIX4 banner (2026-11-08): dual APPROVED (reviewer PASS + anti-team `APPROVED` on FIX4, per `.opencode/context/decisions.md` P0 freeze 2026-11-08). Planner sync (FIX-DOCS-1, 2026-11-08): status headers below flipped `proposed` → `frozen` to align with `decisions.md` (D-BEST-MODEL-V2-1, D-FITTED-REG-V2-1, D-TEMPORAL-SPLIT-1, D-SECRETS-PII-GDPR-1, D-CATALOG-V2-1 all `frozen`). No contract body change in this sync; body below is the FIX4 frozen text.
> **Scope guard:** No `src/` edits, no `README.md` edits, no `decisions.md` edits in this task. Signatures + pseudocode only; workers write product code.
> **ADRs referenced:** D-BEST-MODEL-V2-1, D-FITTED-REG-V2-1, D-TEMPORAL-SPLIT-1, D-SECRETS-PII-GDPR-1, D-MVP-READINESS-1, D-README-1, D-CATALOG-V2-1 (all `frozen`, FIX4 2026-11-08 — see below).
> **All paths relative to forecast_ml_framework/** — every `src/...`, `conf/...`, `catalog.yml:...`, and `tests/...` path in this file is relative to `forecast_ml_framework/` unless an absolute repo path is shown.

## P0-2-FIX revision note (reviewer FAIL — one section per flaw; body edits below are normative)

### B1 — Gate reconciliation (Blocker)
- **Choice:** Single gate statement, reconciled in §4.8 only. It now reads: Blocks P3 entry; P5 inherits via P3→P4→P5 chain (P5 entry = P4 exit). §4.8 normative; :467 header cross-references it; no other gate claim remains. `program.md` / `decisions.md` were not edited (Lead-owned).

### B2 — Catalog ADR orphan (Blocker)
- **Choice (explicit):** Add `D-CATALOG-V2-1` as a new proposal section at the top of this file (immediately below this note, before Contract 1), owner `solution-architect`. Alternative (scope Contract 1 down to ONNX sidecar under D-BEST and defer the 6-file split with no new ADR) was **not** chosen.
- Acceptance for D-CATALOG-V2-1: fragment merge rules + ONNX sidecar entry + per-task split (see that section). Planner will add the ADR row to `decisions.md` on freeze; this file only references the ID and does not edit `decisions.md`.

### B3 — Secrets rotation + residency (Major)
- **Choice (explicit):** Add explicit dispositions (not a non-freezable marker): rotation = out-of-band via cloud, no in-repo rotation in P0; residency = region-agnostic; decision at P2 IaC before P11. See §5.2(e). D-SECRETS-PII-GDPR-1 remains freezable on those dispositions.

### B4 — YAGNI cuts (Major, (a)–(d))
- **(a) ONNX:** Full ONNX InferenceSession/OnnxAdapter deferred to P10. P0 freezes only dict-key semantics + `.onnx` rejection rule. See §3.4: ONNX session deferred to P10 — no session/adapter is frozen or built in P0. P0 adds **no** `best_model_onnx` catalog entry (C4).
- **(b) Catalog:** ONE-catalog grouping now (Option B). The 6-fragment migration is deferred (target design retained in D-CATALOG-V2-1 acceptance, not implemented in P0). See §1.2(b)/§1.3/§1.4.
- **(c) Temporal:** Freeze only `temporal_split` + `assert_no_leakage` now. `purged_cv_folds` + `calibrate_model` prohibition moved to P5/P7 charter (deferred, not frozen). See §4.2/§4.9.
- **(d) Migration echo:** The `__meta__` migration echo is deleted. No consumer/validator is named because the echo does not exist in P0. See §2.2/§2.5.

### P0-2-FIX2 revision note (anti-team REJECTED — C1–C6, H1–H2; body edits below are normative)
- **C1 unwrap:** single `unwrap_envelope(obj)` mandated; all consumers listed in §3.6/§3.10 with modeling/pipeline.py call-site pseudocode. `calibrate_model` regression passthrough returns envelope (not bare).
- **C2 pickle/registry:** `envelope['model']` is the raw allowlisted estimator (`adapter.model`, `adapters.py:532-535`), not the adapter; ~~no `_SAFE_GLOBALS` additions~~ (**superseded by FIX3 for the calibrated case** — 7 additions in §3.2; uncalibrated case needs none); `mlflow.sklearn.log_model` (`model_registry.py:785-787`) receives the unwrapped estimator; `RegistryModelLoader` rebuilds `label_set`/`registry_ref` from bare-estimator artifact via MLflow tags + `model_classes` fallback (§3.2/§3.4/§3.6).
- **C3 `classes_`:** `model_classes(model)` unwraps `ModelAdapter`/`CalibratedClassifierCV`/`_BetaCalWrapper` (§3.3); multiclass `proba[:,1]` behavior stated explicitly.
- **C4 catalog/ONNX:** **no** `best_model_onnx` catalog entry in P0 (Parquet placeholder removed); no fragment-merge/`CatalogMergeError` mechanism claimed; ONE `catalog.yml` with section banners (Option B). Deferred fragments require an explicit future loader change (§1.2/§1.7).
- **C5 temporal:** series-level tz normalization with named mixed-naive/aware `ValueError` check; embargo drops from test head only; single strict formula `max_train < min_test - embargo` everywhere; purged folds fully deferred (no return fields); named calibrate+temporal detection param (§4.2/§4.5).
- **C6 delete/erasure:** `delete()` on `ArtifactStore` ABC + Local/GCS/S3 + online memory/Redis purge; sole `subject_hash` sha256 leaf in `governance/scrub.py`; single marker keys `__meta__.erasure` (registry) / top-level `erasure` (envelope) with re-save paths (§5.2(c)–(d)).
- **H1 `require_v2`:** params wired into the `data_prep` `apply_preprocess` node (§2.2/§2.5).
- **H2 dual-write:** writer emits nested `steps` only in v2; `_check_registry_version` enforces internal consistency (§2.2/§2.5).

### P0-2-FIX3 revision note (anti-team re-gate REJECTED — C2, H1, mediums, low; body edits below are normative)
- **C2-critical calibrated `_SAFE_GLOBALS`:** the prior "no additions" claim (ex-§3.2/§3.9) is **deleted**. Calibrated `envelope['model']` (CCC at `calibration.py:322-327` or `_BetaCalWrapper`+`BetaCalibration` at `:317-318`) is **not** in `_SAFE_GLOBALS` (`artifact_store.py:47-338`). P1 adds exactly the 7 reviewed tuples in §3.2 (pickletools-verified) and a calibrated-envelope `restricted_loads` round-trip RED test (§3.9). Envelope storage for the calibrated case is **unchanged** (still pickled dict) — the allowlist is extended, not bypassed.
- **C2-high loader:** `RegistryModelLoader.resolved_version` is stored/exposed (set from the `model_loader.py:575` local, not log parsing); `label_set` retrieval is named exactly as `MlflowClient.get_model_version_by_alias(name, alias).tags` + `json.loads`, with tags set via `log_model(tags=...)` at `model_registry.py:785-787` (not run `set_tags` at `:783-784`); fallback is `model_classes(estimator)` where `estimator` is the `restricted_loads` return at `model_loader.py:588` (not raw bytes). Mocked-client test added (§3.9).
- **H1 flag:** **both** `modeling_params.fitted_registry_v2` **and** `scoring_params.fitted_registry_v2` are required (no single top-level flag — it is read by neither node). Operator procedure + node-level wiring tests for both nodes (§2.2/§2.5/§2.8).
- **Medium temporal:** `test_size` is **post-embargo** (final fraction), with the overshoot/fixed-point formula in §4.2 so the final fraction matches; post-embargo fraction test in §4.9. Reconciles ex-`:454` vs ex-`:465`.
- **Medium duplicate guard:** deferred-migration duplicate check is now a named custom loader mechanism (`yaml.compose` + node line numbers, §1.2(b)) — no un-named "naming both locations" claim remains.
- **Medium online delete:** `OnlineFeatureStore.delete` calls Redis delete **only** when `self._client()` is not `None` (`online.py:59-82`); unconditional purge removed. Fake-client delete test (§5.2(c)/§5.9).
- **Low:** `RestrictPickleDataset` → `RestrictedPickleDataset` (`core/datasets.py:471`) in §5.2(d).
- B1–B4 / C1 / C3 / C4 / C6 / H2 from FIX/FIX2 are **intact** (no body edit below reverses them).

---

## D-CATALOG-V2-1 — Catalog v2 ADR proposal (for planner row on freeze)

**ADR:** D-CATALOG-V2-1 · **Owner:** `solution-architect` · **Status:** `frozen` (FIX4, 2026-11-08; planner row in `decisions.md` — this file does not edit `decisions.md`)

- **Context:** Contract 1 (§§1.1–1.9) previously attached the catalog split to D-BEST-MODEL-V2-1 (sidecar half) with no catalog-specific ADR. That orphaned the merge/collision contract.
- **Proposed decision:** Catalog v2 = (i) NO ONNX sidecar entry in P0 (C4; session deferred to P10), (ii) per-task split target, (iii) fragment merge rules (§1.2(b) as deferred target; P0 builds Option B grouping per B4(b)).
- **Acceptance (freeze requires all three):**
  1. **Fragment merge rules (deferred target only):** deferred 6-fragment migration needs a future loader change (single-file catalog loader → multi-file include mechanism with duplicate-key guard); **P0 claims no fragment-merge mechanism** — ONE `catalog.yml` with section banners (Option B per B4(b)). No `CatalogMergeError` or fragment merge is frozen or built in P0 (C4).
  2. **ONNX sidecar entry (deferred to P10 per B4(a)):** in **P0 there is no `best_model_onnx` catalog entry** (C4) — `.onnx` loads are rejected at the loader (Contract 3 §3.4); the `onnx_sidecar` key shape in the `best_model` dict is recorded but unused. A future P10 entry must name a real importable dataset class at that time; P0 freezes no placeholder entry.
  3. **Per-task split:** the 6 fragments as listed in §1.2(b) (dataprep, modeling, scoring, monitoring, cost, retraining), each file-disjoint per pipeline. **P0 cut (B4(b)):** migration to these fragments is deferred; P0 freezes ONE-catalog alphabetical grouping (Option B) now. The 6-fragment layout remains the D-CATALOG-V2-1 target and is not P0 implementation scope. Retaining the fragments as a deferred target requires the loader change stated in §1.2(b) (C4).
- **Consequence:** Contract 1 is re-parented from "D-BEST-MODEL-V2-1 (sidecar half)" to D-CATALOG-V2-1 for catalog layout/merge, retaining D-BEST-MODEL-V2-1 only for `best_model` dict semantics (Contract 3).

## Problem, constraints, assumptions

- **Problem:** `best_model` single-pickle + `calibrate_model(best_model)` and `fitted_registry` v1 (`fitted.py:46` raises on version mismatch) are blocking ambiguities for P4/P5. Catalog v1 has no ONNX sidecar and one shared file, so new pipelines collide. Temporal split has three overlapping spellings (`time`/`ordered`/`temporal` → `time`) with no purge/embargo or `split_meta` contract. Retention/GDPR are pure functions with no enforcement reaching artifacts.
- **Constraints:** Python/SQL/HTML-JS, Kedro catalog, restricted-unpickle security boundary, TDD + ≥85% coverage for every contract change, additive/dual-read migrations only (Hyrum's Law: observable v1 behavior is a de facto contract).
- **Assumptions (stated to Lead, design against these):**
  1. Kedro remains the orchestrator; `conf/base/catalog.yml` remains the base file (per-task split is additive include, not a new framework).
  2. `onnxruntime` is an optional serving dependency; pickle-restricted remains the default path. Pyfunc stays gated (`allow_pyfunc=False` default).
  3. Temporal correctness is judged on `time_col` values when present, else row order. Timezones are UTC-normalized at the boundary (per `feature_store` `_naive_utc` precedent).
  4. No new secret store is introduced in P0; secrets come from env (`GOOGLE_APPLICATION_CREDENTIALS`, `AWS_PROFILE`, `MODEL_SHA256`) as today (`catalog.yml:3-4`).

---

# Contract 1 — `catalog.yml` v2 (ONE-catalog grouping now; no ONNX entry and no fragment migration in P0)

**ADR:** D-CATALOG-V2-1 (catalog layout/merge; see proposal at top) · D-BEST-MODEL-V2-1 retained only for `best_model` dict semantics (Contract 3) · **Owner:** `solution-architect` · **Status:** `frozen` (FIX4, 2026-11-08)

## 1.1 Current-state evidence (file:line)

| Claim | Evidence |
|---|---|
| Single shared catalog, no per-task split | `conf/base/catalog.yml:1-758` — one file holds raw, parquet, pickle, BQ, scoring, governance, modeling, monitoring, cost, retraining datasets for all pipelines |
| 12 display pipelines share that one catalog | `src/forecast_ml_framework/pipeline_registry.py:38-51` (`_PIPELINE_MODULES`: `data`/`preprocessing`/`feature_selection`→`data_prep`, `modeling`/`evaluation`/`predict`→`modeling`, `scoring`, `eda`, `monitoring`, `retraining`, `cost_tracking`, `interpretability`) |
| `best_model` is a bare pickle | `conf/base/catalog.yml:565-567` `best_model: RestrictedPickleDataset filepath: artifacts/models/best_model.pkl` |
| `calibrated_model` / registry placeholders are also bare pickles | `catalog.yml:581-585` `calibrated_model.pkl`; `catalog.yml:426-428` `registry_model.pkl`; `catalog.yml:113-120` `xgb.pkl`/`calib.pkl` |
| `fitted_registry` is JSON dict | `catalog.yml:643-645` `fitted_registry: JSONDataset filepath: data/processed/fitted_registry.json` |
| Catalog v1 has no ONNX entry | `grep onnx` over `forecast_ml_framework/` → zero matches (verified at freeze); P0 keeps it that way (C4 — no entry frozen, session deferred to P10) |
| Model load path only understands pickle | `src/forecast_ml_framework/serving/model_loader.py:211-236` `_pickle_path_in` (`*.pkl`/`*.pickle` only); `model_loader.py:277-357` `FileModelLoader` → `restricted_loads`; `model_loader.py:360-601` `RegistryModelLoader` → downloaded bytes → `restricted_loads`, pyfunc gated |
| Pickle boundary is restricted unpickle | `catalog.yml:15-18` (comment: only `RestrictedPickleDataset` for model/calibration pickle); `src/forecast_ml_framework/core/datasets.py:472-480` (save `pickle.dumps`, load via restricted unpickler, never raw `pickle.load`) |

## 1.2 Proposed v2

**a) NO ONNX catalog entry in P0 (C4; ONNX deferred to P10 per B4(a)).** P0 adds **no** `best_model_onnx` dataset name and no placeholder entry. A frozen Parquet (or any other) placeholder for `.onnx` is explicitly removed — it is not a real importable dataset for ONNX. The ONNX artifact remains only as the recorded-but-unused `onnx_sidecar` key shape inside the `best_model` dict (Contract 3 §3.2); P0 serving always uses pickle-restricted and rejects `.onnx` loads (see Contract 3 §3.4, ONNX session deferred to P10). A real `best_model_onnx` entry with a real importable dataset class may be added only at P10 with its own gate; P0 freezes nothing there.
- No existing pipeline wires an ONNX dataset in v1; the modeling pipeline may not emit a `.onnx` sidecar file in P0 (gated emission is a P10 concern, not `modeling_params.export_onnx` in P0).

**b) ONE-catalog grouping now (Option B, B4(b)); per-task 6-fragment migration deferred.** P0 keeps the single `conf/base/catalog.yml` and freezes alphabetical section grouping with banner comments (`# --- dataprep ---`, `# --- modeling ---`, …) so ownership/collision boundaries are visible without moving keys. The deferred 6-fragment target (retained as D-CATALOG-V2-1 acceptance, **not** P0 scope) is:

```
conf/base/catalog.yml            # P0: sole file, alphabetical grouping banners (v1 names unchanged)
# DEFERRED TARGET (not P0): conf/base/catalog/{dataprep,modeling,scoring,monitoring,cost,retraining}.yml
```

- **No fragment-merge mechanism is frozen in P0 (C4).** Dropping the `CatalogMergeError` / fragment-merge claim: it is not a Kedro mechanism, and P0's single-file catalog has no merge step. The deferred migration (when gated) requires an explicit future loader change: replace the single-file catalog load (`DataCatalog.from_config` on one file) with a multi-file include mechanism (e.g. OmegaConf include list or Kedro `CatalogConfigResolver` with grouped files) **plus** the named custom duplicate-key guard below. File moves alone do not constitute the migration. Env overlays (`conf/local`, `conf/gcp`, `conf/aws`) keep overriding only `filepath`/URI per entry, as today (`catalog.yml:2-4`).
  - **Custom duplicate-guard mechanism (deferred, FIX3-medium):** a P1-when-gated helper `check_catalog_duplicates(paths) -> None` (not a claimed Kedro API) that, for each catalog file in `paths`, runs `yaml.compose(open(path))` and walks every `MappingNode`, recording for each `(key_node, value_node)` in `MappingNode.value` the entry `(file, key_node.start_mark.line)` — the **key** node's start line, not the mapping node's `node.start_mark.line`; on a repeated key it raises `ValueError(f'duplicate catalog key {key!r}: {file1}:{line1} and {file2}:{line2}')` — i.e. both locations are named **by this helper's own compose walk**, not by Kedro. The loader calls it before `DataCatalog.from_config`. P0 builds no such helper (single file, no merge); the deferred migration must land the helper + its test (duplicate key in two fragments → error naming both `file:line`s) in the same change as the multi-file include.

## 1.3 Alternatives considered

| Option | Cost | Failure mode | Verdict |
|---|---|---|---|
| **A. Fragments + includes (deferred target)** | Small: file moves **plus** the future loader change in §1.2(b) (multi-file include + duplicate-key guard); not Kedro-native fragment merge | Mis-merge if a new fragment redefines a base key → guarded by YAML duplicate detection at load (not a claimed Kedro API) | Deferred beyond P0 (B4(b)) — retained as D-CATALOG-V2-1 target, not P0 implementation |
| **B. One catalog.yml + alphabetical grouping (chosen for P0)** | Zero migration | New pipelines still collide on the same file; mitigated by grouping banners + CODEOWNERS per section | **Chosen for P0** — YAGNI cut: freezes visible ownership now, defers 6-file migration |
| C. Catalog per env only (`catalog_<pipeline>.yml`) | Full duplication of anchors; drift | Three copies of `_spark_bq_data`/`_save_parquet` diverge; Hyrum risk on every rename | Rejected — violates one-version rule, 3× maintenance |

## 1.4 Trade-off record

- **Decision:** Option B for P0, Option A deferred. **Gain:** zero migration risk, byte-stable `filepath` values, visible section ownership. **Give up:** new pipelines still edit one file until the deferred migration lands (mitigation: alphabetical banners + per-section ownership). **Reverse if:** fragment migration is re-proposed and gated — then apply the frozen merge rules above with no key renames.

## 1.5 Migration / dual-read / rollout

- v1 `catalog.yml` stays loadable alone; P0 adds alphabetical grouping banners only. **No `best_model_onnx` entry is added in P0** (C4). No fragment files are created in P0.
- Rollout: (1) add grouping banners (no key moves), (2) **no** ONNX entry (deferred to P10), (3) deferred 6-fragment migration is a separate gated task that must include the loader change in §1.2(b) (no merge mechanism exists in P0). No dual `filepath` shadowing.

## 1.6 Catalog / serving impact

- Serving (`serving/app.py`, `model_loader.py`) gains no flavor dispatch in P0 (C4) — pickle-restricted loading + `.onnx` rejection only (Contract 3 §3.4); serving never imports catalog YAML directly.
- IaC (P2) provisions storage paths from catalog `filepath` values only — no hardcoded paths in Terraform/Pulumi.

## 1.7 Failure modes

| Failure | Observable behavior | Handling |
|---|---|---|
| Duplicate key with different `type` (deferred fragment migration) | `ValueError` from the custom `check_catalog_duplicates` helper (`yaml.compose` walk, §1.2(b)) at catalog load, naming both `file:line` locations (when the deferred migration lands; P0 single-file grouping has no merge step and claims no Kedro fragment-merge mechanism or helper) | Fail fast; no partial run |
| `.onnx` load (no catalog entry in P0) | `ModelCorruptError('model artifact corrupt')`, no URI leak (per `model_loader.py:239-250` logging rule) | P0: `.onnx` load is rejected (§3.4); pickle path unchanged |
| Grouping banner missing for a new pipeline section | Existing pipelines unaffected | Warning in review; pipeline fails with `ModelNotConfiguredError` only if it needs the dataset |

## 1.8 Module boundaries

- **Owner:** `solution-architect` (contract) → `worker` (catalog grouping) → `tester` (catalog load tests). **Must never depend on:** `src/` logic (YAML only).

## 1.9 Test strategy (observable behavior, TDD, ≥85% coverage)

- **RED first:** `test_catalog_v2.py` — load catalog; assert all v1 names resolve; assert **no `best_model_onnx` entry** exists in P0; assert alphabetical grouping banners cover all v1 names (each name under exactly one banner). No fragment/merge or ONNX-session test in P0 (both deferred). Every assertion is on catalog load behavior, never on file text.
- **GREEN:** grouping banners only (no ONNX entry, no fragments in P0). **REFACTOR:** anchors dedup.
- **Coverage:** ≥85% of the new catalog validation helper; threshold gate in `ci/run_checks.sh` extended to `tests/test_catalog_v2.py`.

**P1 handoff (pseudocode only):** `load_catalog(base_dir) -> Catalog` // single file, grouped; env overlay applied as today.

---

# Contract 2 — `fitted_registry` v2 dual-read (v1 + v2, migration flag, `_check_registry_version` update, scoring consumers)

**ADR:** D-FITTED-REG-V2-1 · **Owner:** `solution-architect` · **Status:** `frozen` (FIX4, 2026-11-08)

## 2.1 Current-state evidence (file:line)

| Claim | Evidence |
|---|---|
| Version is a single int + schema string | `src/forecast_ml_framework/pipelines/data_prep/fitted.py:46-49` `FITTED_REGISTRY_VERSION=1`, `REGISTRY_SCHEMA='fitted_registry.v1'` (aliased `REGISTRY_VERSION`/`FITTED_REGISTRY_SCHEMA`) |
| Mismatch raises, no dual-read | `fitted.py:259-284` `_check_registry_version` — collects `version`/`__version__`/`schema`/`__schema__`/`__meta__.*`, raises `ValueError('fitted_registry %s mismatch…')` when any present value ≠ expected |
| Meta keys excluded from steps | `fitted.py:51-53` `_REGISTRY_META_KEYS = {version, __version__, schema, __schema__, __meta__, steps}`; `fitted.py:287-304` `_registry_steps` calls the check then merges nested `steps` |
| Writer stamps v1 + typed flag | `fitted.py:937-949` `fit_preprocess` writes `version/__version__/schema/__schema__/__meta__{version, schema, typed_categories: True}` + step states; `fitted.py:325-351` `_check_legacy_typed_data` fails loud for pre-typed registries on tuple/typed data |
| Catalog shape is JSON | `conf/base/catalog.yml:643-645` `fitted_registry: JSONDataset → data/processed/fitted_registry.json` |
| Scoring replays via `apply_preprocess` | `src/forecast_ml_framework/pipelines/scoring/pipeline.py:185-227` `score_data_prep` (empty-registry guard, then `apply_preprocess`); `scoring/pipeline.py:247-251` node wiring `[input_dataset, 'fitted_registry', 'params:scoring_params'] → scoring_data` |
| Data-prep side also replays | `src/forecast_ml_framework/pipelines/data_prep/pipeline.py:91-96` `apply_preprocess` node `['test_raw','fitted_registry'] → test_preprocessed_data`; `data_prep/nodes.py:836-855` `apply_preprocess` wraps `fitted.apply_preprocess` |
| Seven fitted steps only | `fitted.py:874-895` `_FIT`/`_APPLY` (Imputations, Normalizations, QuantileDiscretizer, HighLabelBinning, CategoricalEncoding, TargetEncoding, CapOutliers); unknown step → `ValueError` at fit (`fitted.py:922-926`) and apply (`fitted.py:970-973`) |

## 2.2 Proposed v2

**Canonical v2 shape (what `fit_preprocess` writes):**

```python
{
  "version": 2, "__version__": 2,
  "schema": "fitted_registry.v2", "__schema__": "fitted_registry.v2",
  "__meta__": {
    "version": 2, "schema": "fitted_registry.v2",
    "typed_categories": True
  },
  "steps": { "<StepName>": { "<col>": <state> }, ... }       # nested canonical
}
```

- New fields are **additive only**: `steps` nesting is canonical for v2 (v1 flat keys still accepted on read). **Writer rule (H2):** `fit_preprocess` (v2) writes **nested `steps` only** — it must stop writing flat step keys (`registry_out.update(registry)` at `fitted.py:948` is removed for v2; the seven step states go under the single `"steps"` key, never duplicated flat and nested). The migration echo under `__meta__` is deleted (B4(d)) — no such field is written, read, or validated in P0.
- **Migration flag (H1, FIX3 — both-flags, no single location):** there is **no** single flag location and **no** top-level `fitted_registry_v2` read by either node (the top-level key is deleted from this contract — nothing reads it). Each node reads **its own** pipeline params, and **both** must be set consistently by the operator:
  - `data_prep` node reads `params:modeling_params.fitted_registry_v2` (bool, default `False`);
  - `scoring` node reads `params:scoring_params.fitted_registry_v2` (bool, default `False`).
  **Operator procedure:** set the **same** bool in both `modeling_params` (`conf/base/parameters/modeling.yml`) and `scoring_params` (`conf/base/parameters/platform.yml`) — e.g. to enable v2-required, set `fitted_registry_v2: true` under **both** param blocks in the same change; to roll back, clear both. Setting only one is a misconfiguration: each node enforces only its own flag (the other node keeps dual-read), so a half-set flag silently leaves one path accepting v1 — reviewers must check both files. When a node's flag is `True`, that node **requires** a v2 registry (version 2 present) and logs the upgrade path; when `False` (default), dual-read accepts v1 or v2 with no behavior change. The flag never changes fitted statistics — it only gates the required version on read. The reader function `read_registry(obj, require_v2=...)` takes the already-resolved per-node bool; it never reads params itself.
- **Scoring + data-prep consumers** (`score_data_prep`, `data_prep.apply_preprocess` node, direct `fitted.apply_preprocess` callers) all funnel through the same reader (no per-caller version logic). **H1 wiring (exact):** the flag must reach the `data_prep` apply node, which today has no params input — see §2.5 for the exact node input change.

## 2.3 Alternatives considered

| Option | Cost | Failure mode | Verdict |
|---|---|---|---|
| **A. Dual-read + flag (chosen)** | Small: one reader branch; writer stamps v2 additively | A buggy v2 writer could emit v2 schema with v1 steps → guarded by step-shape validation tests | **Chosen** — additive, Hyrum-safe, flag gives staged rollout |
| B. Hard cutover v1→v2 (bump only) | Trivial code | Every v1 artifact on disk breaks scoring immediately; violates dual-read ADR | Rejected — no migration window |
| C. Fork new registry name (`fitted_registry_v2` dataset) | New catalog key + rewiring both pipelines | Two JSON datasets to keep in sync; scoring would need to know which to read → diamond coupling | Rejected — catalog collision (Contract 1) |

## 2.4 Trade-off record

- **Decision:** Option A. **Gain:** v1 artifacts keep scoring; v2 writers land behind the flag. **Give up:** reader carries two code paths until v1 is retired (retirement = separate ADR + deprecation window ≥ 1 P-phase). **Reverse if:** v2 step shape proves incompatible in shadow tests — then freeze v1 writer and re-propose.

## 2.5 Migration / dual-read semantics

```
read_registry(obj, require_v2=False) -> NormalizedRegistry
  normalize to {version, schema, meta, steps}          # accepts v1 flat OR v2 nested/steps
  if require_v2 and version != 2: raise ValueError("requires fitted_registry v2")
  if version == 1: apply v1 steps verbatim (no re-fit, no statistic recompute)
  if version == 2: apply v2 steps verbatim
  unknown version/schema (e.g. 3): raise ValueError (fail loud, never silent identity)
```

- **`_check_registry_version` update (H2, internal consistency):** change from "raise unless exactly v1" to "raise unless in `{1, 2}` (or a subset when `require_v2`), **and** internally consistent". Present `version`/`schema`/`__meta__` values must all agree with each other: collect every present `version`-kind value and every present `schema`-kind value; if version-kind values are present they must be a single version `V ∈ {1, 2}` and every schema-kind value present must equal `fitted_registry.vV` for that same `V` (so a v2 registry must stamp `schema='fitted_registry.v2'` on **every** present schema key — top-level `schema`/`__schema__` and `__meta__` schema alike — not per-key `{1, 2}` membership). `version=2` + `schema='fitted_registry.v1'` (or `version=1` + v2 schema, or two different versions present) → `ValueError('fitted_registry schema mismatch')`. This preserves the existing fail-loud precedent (`fitted.py:259-284`) while admitting exactly one consistent version per document.
- **H1 — exact `data_prep` node wiring (chosen: wire params, not scoring-only).** `data_prep/pipeline.py:91-96` `apply_node` today takes `inputs=['test_raw', 'fitted_registry']` with `func=apply_preprocess` (`data_prep/nodes.py:836-855`, current signature `apply_preprocess(df, registry)`). P1 changes it to **exactly**:

  ```python
  # data_prep/pipeline.py apply_node (H1)
  apply_node = node(
    func=apply_preprocess,
    inputs=['test_raw', 'fitted_registry', 'params:modeling_params'],
    outputs=['test_preprocessed_data'],
    name='apply_preprocess',
  )
  # data_prep/nodes.py
  def apply_preprocess(df, registry, params=None): ...
    # require_v2 = bool((params or {}).get('fitted_registry_v2', False))
    # return _fitted_apply_preprocess(df, registry, require_v2=require_v2)
  # data_prep/fitted.py
  def apply_preprocess(df, registry, require_v2=False): ...
    # steps = _registry_steps(registry, require_v2=require_v2)
  ```

  `params:modeling_params` is the same params dataset this pipeline already consumes (`split_node`, `data_prep/pipeline.py:77-83`), so no new pipeline dependency is introduced. `scoring/pipeline.py:247-251` already wires `params:scoring_params` into `score_data_prep`, which reads `scoring_params.fitted_registry_v2`. Both paths now honour **their own** flag under the both-flags procedure in §2.2 (operator sets both; neither node reads a top-level key, and neither reads the other pipeline's params); the scoring-only narrowing is **not** chosen, and a single shared location is **not** chosen (it would force one pipeline to consume the other's params dataset).
- **No statistic change:** v2 does not alter fit math in P0 (seven steps, same `_FIT`/`_APPLY`). Any future step-shape change is a separate `fitted_registry.v3` proposal.
- **Backfill:** `registry_from_json`/`registry_to_json` (`fitted.py:978-985`) pass v1 payloads through unchanged; no P0 backfill script and no migration-echo stamping. Scoring never backfills in-line.

## 2.6 Catalog / serving impact

- Catalog key `fitted_registry` unchanged (still `JSONDataset`). No new dataset; `metadata` gains `schema: fitted_registry.v2` annotation only in the grouped catalog (Contract 1) — not required for reading.
- Serving impact: none directly; batch scoring (`scoring/pipeline.py:185`) is the consumer. Online serving (`serving/app.py`) does not read `fitted_registry` and must continue not to.

## 2.7 Failure modes

| Failure | Observable behavior | Handling |
|---|---|---|
| v1 registry scored with `require_v2=True` | `ValueError: requires fitted_registry v2` (names the flag) | Caller must refit or clear the flag; never silent identity |
| v2 registry scored with default flag | Scores identically to v1 writer output for same steps | Regression test: v1/v2 fixture parity on all seven steps |
| Mixed meta (`version=2`, `schema=v1`) | `ValueError: fitted_registry schema mismatch` | Fail loud; writer bug, not data bug |
| Unknown version `3` | `ValueError` (unknown registry version) | Must not attempt partial apply |
| Empty registry + fitted steps declared | Existing `ValueError` preserved (`scoring/pipeline.py:220-226`) | Unchanged |

## 2.8 Test strategy (observable behavior, TDD, ≥85% coverage)

- **RED first:** fixture pair `fitted_registry_v1.json` / `fitted_registry_v2.json` (same step states, different envelope) in `tests/fixtures/`. Tests assert: `apply_preprocess(df, v1) == apply_preprocess(df, v2)` (frame equality); `require_v2` rejects v1; mixed meta raises; v3 raises; `score_data_prep` identity-vs-error matrix unchanged. All on DataFrame behavior.
- **Node-level wiring tests (H1, FIX3 — both nodes, not just the reader):** (1) `data_prep` node test: build the `apply_preprocess` node from `data_prep/pipeline.py` and assert its inputs are exactly `['test_raw', 'fitted_registry', 'params:modeling_params']`, then run the node function with `modeling_params={'fitted_registry_v2': True}` + v1 registry → `ValueError`, and with `False` → scores; (2) `scoring` node test: build the `score_data_prep` node from `scoring/pipeline.py:247-251`, assert its params input reads `scoring_params`, then run with `scoring_params={'fitted_registry_v2': True}` + v1 registry → `ValueError`, and with `False` → scores. Both tests must fail if either node's params wiring is dropped, even when the reader function itself is correct.
- **GREEN:** reader + flag + `_check_registry_version` update. **REFACTOR:** no per-caller branches.
- **Coverage:** ≥85% of `fitted.py` reader/version paths (new lines); full-file gate via `pytest --cov=forecast_ml_framework.pipelines.data_prep.fitted --cov-fail-under=85` in CI.

**P1 handoff (pseudocode only):** `read_registry(raw, require_v2=False) -> steps_dict` // normalize v1|v2, validate (H2 consistency), return steps; `fit_preprocess` stamps v2 envelope + nested `steps` only (H2); `data_prep` `apply_node` inputs `['test_raw', 'fitted_registry', 'params:modeling_params']` and `apply_preprocess(df, registry, params=None)` forwards `require_v2=bool(params.get('fitted_registry_v2', False))` (H1, modeling_params); `score_data_prep` forwards `require_v2=bool(scoring_params.get('fitted_registry_v2', False))` (H1, scoring_params); operator sets both flags together (§2.2).

---

# Contract 3 — `best_model` v2 (dict, singular choice; `_predict_proba` shape, loader dispatch, pyfunc gate)

**ADR:** D-BEST-MODEL-V2-1 (BLOCKING) · **Owner:** `solution-architect` · **Status:** `frozen` (FIX4, 2026-11-08)

## 3.1 Current-state evidence (file:line)

| Claim | Evidence |
|---|---|
| `best_model` catalog entry is a single pickle | `conf/base/catalog.yml:565-567` `RestrictedPickleDataset → artifacts/models/best_model.pkl`; `best_model_name` sibling `catalog.yml:571-573` |
| `train_models` returns a bare adapter | `src/forecast_ml_framework/pipelines/modeling/adapters.py:1012-1048` → `return fitted[best], per_model, best` (single fitted adapter) |
| Modeling pipeline wires bare adapter through calibration | `src/forecast_ml_framework/pipelines/modeling/pipeline.py:442-452` `train_models → [best_model, …]`; `calibrate_model: inputs=[best_model, x_train, y_train, params] → [calibrated_model, calibration_meta]` |
| `calibrate_model` takes a model, returns wrapper + meta | `src/forecast_ml_framework/pipelines/modeling/calibration.py:237-328` `calibrate_model(model, x_train, y_train, params) -> (wrapped, {calibrated, method, requested_method, fallback})`; `_BetaCalWrapper` (`calibration.py:18-32`) exposes `predict_proba`/`predict` |
| `_predict_proba` is binary-assumed, `None` for regression | `pipeline.py:244-248` returns `model.predict_proba(x_test)` for classification, `None` otherwise; downstream `build_lift_table`/`select_model_threshold` (`pipeline.py:251-284`, `380-400`) take column 1 when 2-D |
| Probes / adapters expose `predict_proba` passthrough | `adapters.py:528-530` `predict_proba → model.predict_proba`; `adapters.py:766-782` H2O variant requires `predict_proba` on the wrapped model |
| Loader is pickle-restricted, pyfunc gated | `src/forecast_ml_framework/serving/model_loader.py:4-25` (module docstring: never `mlflow.sklearn` entry point on registry bytes; pyfunc NOT default); `model_loader.py:277-357` `FileModelLoader` → `restricted_loads`; `model_loader.py:360-601` `RegistryModelLoader(allow_pyfunc=False)`; `model_loader.py:553-599` pyfunc only when `allow_pyfunc=True` and restricted load fails; `model_loader.py:432-436` startup warning |

## 3.2 Proposed v2 — singular choice: dict

**`best_model` v2 IS a dict with exactly these four keys** (no list alternative, no `either/or`):

```python
best_model_v2 = {
  "model":         <fitted adapter / estimator object>,   # same object as v1 single pickle
  "label_set":     <list[str]> | None,            # see §3.3; None only for regression
  "registry_ref":  {"name": str, "alias": str, "version": str | None} | None,
  "onnx_sidecar":  {"path": str | None, "opset": int | None, "sha256": str | None} | None,
}
```

| Key | Type | Invariants |
|---|---|---|
| `model` | **raw estimator** — the fitted sklearn estimator object wrapped in `adapter.model` (`adapters.py:532-535` `save()` dumps `self.model`, never the adapter) or, after calibration, the `CalibratedClassifierCV` instance or `_BetaCalWrapper` base estimator | Required; `predict` always; `predict_proba` required iff classification. **Not** the `ModelAdapter` itself (see C2 below). Calibrated cases additionally require the 7 `_SAFE_GLOBALS` additions in the C2 paragraph below (they are **not** in the current allowlist) |
| `label_set` | `list[str]` (ordered) or `None` | Ordered, unique, length == proba width. `None` ⇔ `task_type == 'regression'` (proba is `None`, never `shape (n,0)`) |
| `registry_ref` | dict or `None` | `None` = file-only model; otherwise `name`+`alias` required, `version` is the resolved MLflow version string **from `loader.resolved_version`** (stored by `RegistryModelLoader` at `model_loader.py:575`, §3.2 — never parsed from the log line) |
| `onnx_sidecar` | dict or `None` | `None` = pickle-only. Otherwise `path` required (`.onnx`), `opset` required int, `sha256` optional pin recorded only in P0 (no session use — see §3.4) |

**`unwrap_envelope` (C1, normative):** every consumer of a possibly-enveloped model object must call the single helper:

```python
def unwrap_envelope(obj):
  # bare (v1 adapter/estimator) -> itself; v2 dict envelope -> obj['model']
  if isinstance(obj, dict) and 'model' in obj:
    return obj['model']
  return obj
```

This is the only unwrapping mechanism in P1 — no per-caller `obj['model'] if isinstance(obj, dict)` branches. Writers call `wrap_envelope(model, ...)` to build the dict; all readers call `unwrap_envelope(obj)` before invoking `predict`/`predict_proba`/`classes_`.

**What `envelope['model']` is (C2, normative):** the **raw estimator**, not the adapter. Evidence: `ModelAdapter.save()` persists `self.model` (`adapters.py:532-535` `pickle.dump(self.model, handle)`); `calibrate_model` already unwraps with `base = getattr(model, 'model', model)` (`calibration.py:296`) before fitting and re-wraps via `_wrap(calibrated)` (`calibration.py:35-39`, `calibration.py:324-327`) putting the calibrated estimator into `adapter.model`. Envelope writers must therefore store `adapter.model` (the estimator inside the adapter), never the adapter instance. Storing adapters would force additional allowlist entries and is forbidden in P0. `save()` callers that persist the envelope pickling the dict directly therefore persist estimator-bytes exactly as v1 `save()` does — **except** for the calibrated case, which needs the additions below.
- **Calibrated `_SAFE_GLOBALS` additions (C2-critical, FIX3 — the prior "no additions" claim is deleted):** neither calibrated form is in the current allowlist (`artifact_store.py:47-338` grep confirms zero entries for any of these classes): `envelope['model']` is a `CalibratedClassifierCV` (`calibration.py:322-327`, sigmoid/isotonic) or a `_BetaCalWrapper` wrapping `betacal.BetaCalibration` (`calibration.py:317-318`, betacal). P1 **extends** `_SAFE_GLOBALS` with **exactly** these 7 `(module, name)` tuples (no prefix/module-wide entries, no loaders):
  ```python
  _SAFE_GLOBALS_ADDITIONS_CALIBRATED = {
    ('sklearn.calibration', 'CalibratedClassifierCV'),        # sklearn/calibration.py:74
    ('sklearn.calibration', '_CalibratedClassifier'),        # sklearn/calibration.py:754 (per-fold wrapper inside CCC.calibrated_classifiers_)
    ('sklearn.calibration', '_SigmoidCalibration'),          # sklearn/calibration.py:1015 (sigmoid calibrator inside _CalibratedClassifier.calibrators)
    ('sklearn.isotonic', 'IsotonicRegression'),              # sklearn/isotonic.py:181 (isotonic calibrator; imported at sklearn/calibration.py:30)
    ('forecast_ml_framework.pipelines.modeling.calibration', '_BetaCalWrapper'),  # calibration.py:18
    ('betacal', 'BetaCalibration'),                          # betacal/__init__.py (class defined in package __init__, so __module__ == 'betacal')
    ('betacal.beta_calibration', '_BetaCal'),                # betacal/beta_calibration.py (default parameters='abm' calibrator_ inside fitted BetaCalibration)
  }
  ```
  **Verification method (pickletools, P1 gate):** fit (a) `CalibratedClassifierCV(LogisticRegression, method='sigmoid')` and `method='isotonic'`, (b) `_BetaCalWrapper(base, BetaCalibration().fit(oof, y))` (betacal present), then `pickle.dumps` each and scan `pickletools.genops` for `GLOBAL`/`STACK_GLOBAL` `(module, name)` pairs; subtract the already-allowlisted globals — the residual must be **exactly** the 7 tuples above (base-estimator/numpy/pandas globals are already listed). `_TemperatureScaling` (`sklearn/calibration.py:1070`) and the non-default betacal calibrators (`_BetaAMCal`/`_BetaABCal`/`_BetaACal`) are **not** added in P0: `calibrate_model` only allows sigmoid/isotonic/betacal (`_CALIBRATION_METHODS`, `calibration.py:81`) with default `BetaCalibration()`, so a temperature/other-variant pickle must fail loud via `restricted_loads` rejection (never silently allowlisted). Calibrated envelope storage is **unchanged** (still a pickled dict via `RestrictedPickleDataset`) — the fix is the allowlist extension above, not a storage change and not a `pickle.load` bypass.

**MLflow `log_model` (C2):** `model_registry.py:785-787` calls `sklearn.log_model(sk_model=model, ...)` — `sk_model` must be a sklearn estimator, never the v2 dict and never the adapter. P1 registry writer must unwrap first: `sk_model = unwrap_envelope(model)` (dict → estimator; bare → itself). Passing the envelope dict to `log_model` is a contract violation.

**`RegistryModelLoader` reconstruction (C2, FIX3):** the artifact returned by `model_loader.py:360-601` `RegistryModelLoader._load_registry` is the bare estimator (`restricted_loads(raw)` at line 588 — MLflow logged only the estimator). `label_set`/`registry_ref` are therefore **not** inside the artifact; the loader itself returns the bare estimator, and envelope metadata is reconstructed **outside** it by the P1 caller via this exact mechanism:
  1. **`resolved_version` stored on the loader (not log parsing):** P1 adds `self.resolved_version = None` in `RegistryModelLoader.__init__` (`model_loader.py:386-438`) and sets `self.resolved_version = str(version)` immediately after the existing local `version = self._version_field(version_obj, 'version')` at `model_loader.py:575` (before the `:576-578` log line, which stays as a log only). The caller obtains the version as `loader.resolved_version` after `load()`/`__call__` — never by parsing logs. `registry_ref = {"name": loader.model_name, "alias": loader.alias, "version": loader.resolved_version}`.
  2. **`label_set` from MLflow version tags (exact API):** `version_obj = MlflowClient.get_model_version_by_alias(name, alias)` (the same call already at `model_loader.py:565-566` / `model_registry.py:856-857`), then `label_set = json.loads(version_obj.tags['label_set'])` and `task_type = version_obj.tags['task_type']`. Tags are set by the P1 `model_registry` writer **via `sklearn.log_model(..., tags=...)` at `model_registry.py:785-787`** (`tags={'label_set': json.dumps(label_set), 'task_type': task_type}` merged into the existing `tags` dict passed there) — **not** via the run-level `mlflow.set_tags(tags)` at `model_registry.py:783-784` (that writes run tags, which `get_model_version_by_alias(...).tags` does not return; relying on it is a contract violation). P1 must pass `label_set`/`task_type` in the `log_model(tags=...)` argument on every register.
  3. **Fallback if the tag is absent:** `estimator` is the **return value** of `_load_registry` (i.e. `restricted_loads(raw)` at `model_loader.py:588`), **not** the raw bytes — derive `label_set = [str(c) for c in model_classes(estimator)]` (C3, §3.3) and `task_type` from `params.modeling_params.task_type`, failing with `ModelCorruptError` if `model_classes(estimator)` is `None` and no tag exists. Applying `model_classes` to raw bytes is a contract violation.
  File-loaded envelopes (catalog pickle) preserve the dict directly, so steps (2)–(3) apply only to registry-loaded bare estimators.

**Versioning:** the dict itself carries `model_version`? No — deliberate omission: `registry_ref.version` is the version of record when registered; for file-only models the catalog `filepath` + `sha256` (when pinned) is the version. A separate `meta` is not added — calibration meta stays in `calibration_meta` (`catalog.yml:589-591`).

**Why not a list:** a `list` (`[model, label_set, registry_ref, onnx_sidecar]`) would make every consumer do positional unpacking; adding a fifth key later would silently shift meanings. The dict is self-describing, forward-compatible (new keys ignored with a warning), and matches the `calibration_meta` / `split_meta` dict precedent in this codebase.

## 3.3 `_predict_proba` shape per `label_set`; `model_classes` (C3); `calibrate_model`

**`model_classes` (C3, normative):** the single helper for reading class labels off any fitted object. It unwraps in order — envelope → `ModelAdapter` → calibration wrappers — and never calls `predict`:

```python
def model_classes(model):
  m = unwrap_envelope(model)                    # envelope -> estimator (C1)
  if hasattr(m, 'model'): m = m.model           # ModelAdapter (adapters.py:515) -> self.model
  if type(m).__name__ == '_BetaCalWrapper':     # calibration.py:18-32 (no classes_ of its own)
    m = m.base                                  # -> self.base.classes_ (calibration.py:32)
  return getattr(m, 'classes_', None)           # CalibratedClassifierCV exposes classes_ after fit
```

Precisely: `ModelAdapter` delegates to `self.model` (`adapters.py:515, 526-530`); `_BetaCalWrapper` (`calibration.py:18-32`) exposes no `classes_` of its own — its labels are `self.base.classes_` (`calibration.py:32` `self.base.classes_[...]`); `CalibratedClassifierCV` (`calibration.py:322-325`) exposes `classes_` after fit, delegating to its base estimators; `_wrap` (`calibration.py:35-39`) always parks the calibrated estimator in `adapter.model`, so the `getattr(m, 'model', m)` step already resolves `_wrap`-produced adapters. `model_classes` returns the numpy `classes_` array or `None` for regression estimators that have none; `validate_envelope` compares `label_set` to `[str(c) for c in model_classes(model)]`.

- `task_type == 'classification'`: `predict_proba(x) -> np.ndarray shape (n, len(label_set))`, rows sum to 1.0 (±1e-6). `label_set` column order == proba column order. Binary: `label_set == ['0','1']`-ordered positive last (column 1 is positive — preserves `pipeline.py:281-282` behavior).
- `task_type == 'regression'`: `predict_proba` is not called; pipeline `_predict_proba` yields `None` (`pipeline.py:244-248` preserved). `label_set is None` is the signal; a non-`None` `label_set` with regression is a `ValueError` at envelope validation.
- `label_set` vs `model_classes`: when the fitted estimator exposes `classes_`, envelope validation requires `label_set == [str(c) for c in classes_]` (order-sensitive, via `model_classes` above so adapters and calibration wrappers compare correctly). Mismatch → `ValueError('label_set does not match model classes_')` at training time, never at serving time.
- **Downstream multiclass behavior (C3, explicit):** `build_lift_table` (`pipeline.py:251-284`) and `select_model_threshold` (`pipeline.py:380-400`) take `proba[:, 1]` whenever `proba.ndim == 2 and shape[1] > 1` (`pipeline.py:281-282`). For **binary** (`len(label_set) == 2`) column 1 is the positive class under the envelope invariant above, so the slicing is semantically correct. For **multiclass** (`len(label_set) > 2`) `proba` is `(n, k)` in `label_set` order and `proba[:, 1]` selects the **second label's** score (`label_set[1]`), not a "positive class" — v1 slicing is preserved verbatim for back-compat, but multiclass lift/threshold output is a binary reduction on `label_set[1]` and must not be read as pos-vs-rest. P0 freezes exactly this behavior; any multiclass lift/threshold semantics beyond "column 1 as-is" require an explicit `label_set` index parameter and a new ADR before production use (P5/P11 gate must reject multiclass threshold promotion that relies on the bare `[:,1]` slice).
- Downstream (`build_lift_table`, `select_model_threshold`, `evaluate_model`) consume the **unwrapped** model only (C1); proba shapes are unchanged from v1 so their `proba[:,1]` extraction keeps working.

**`calibrate_model` v2 signature** (behavior addition, not a rename):

```python
def calibrate_model(model, x_train, y_train, params) -> (calibrated_envelope, meta)
# where calibrated_envelope is the v2 dict with:
#   model = calibrated wrapper, label_set = input label_set (validated unchanged),
#   registry_ref/onnx_sidecar carried through (onnx_sidecar is NOT auto-regenerated).
```

- The input `model` argument is unwrapped with `unwrap_envelope` first (C1). If it arrives as a bare adapter (v1 path), the function wraps the result into a v2 envelope with `label_set` derived from `model_classes(unwrapped)` (or `None` for regression) and `registry_ref/onnx_sidecar=None`. The **regression / `method='none'` passthrough must also return an envelope, never the bare input**: today it returns `(model, {...})` unwrapped (`calibration.py:282-285`); in v2 that branch returns `(wrap_envelope(unwrap_envelope(model), label_set=<derived-or-carried>), meta)` so the return type is uniform for all methods. Both paths (calibrated / passthrough) return the same dict type.
- Multiclass vs binary: binary `_oof_proba` (`calibration.py:181-227`) already requires `(n,2)`; v2 preserves that. `calibrate_probability` (`calibration.py:84-178`) is unchanged (1-D/(n,1)/(n,2) only).

## 3.4 Model loader flavor dispatch (pickle-restricted in P0; ONNX session deferred to P10)

> **ONNX session deferred to P10** — full ONNX InferenceSession/OnnxAdapter is deferred to P10 and is not frozen or built in P0. P0 freezes only (i) the dict-key semantics in §3.2 (including the `onnx_sidecar` key shape, recorded but unused) and (ii) the `.onnx` rejection rule below.

```
load_model(envelope_or_path, *, allow_pyfunc=False, ...) -> fitted_model
  if envelope is dict (v2):
      return unwrap_envelope(envelope)          # already in-memory; C1 single mechanism
      # onnx_sidecar present: recorded only; never used to load in P0
  elif path/URI:
      if suffix == ".onnx": ModelCorruptError  # P0 rejection rule: always, regardless of flags (session deferred to P10)
      elif suffix in (".pkl", ".pickle"): restricted_loads (existing FileModelLoader/RegistryModelLoader)
      else: ModelCorruptError
  pyfunc: only when allow_pyfunc=True AND restricted_loads failed (existing gate, preserved verbatim)
```

- **P0 rejection rule (frozen):** any `.onnx` path load raises `ModelCorruptError('model artifact corrupt')` with no URI leak. There is no `allow_onnx` flag and no ONNX session in P0; P10 adds the session dispatch behind a new gate.
- **Pickle path is unchanged:** `restricted_loads` allowlist (`core/artifact_store.py`), never `pickle.load`.
- **Pyfunc remains gated:** `allow_pyfunc=False` default; when `True`, startup warning (`model_loader.py:432-436`) + fallback only after restricted failure (`model_loader.py:590-599`). `.onnx` rejection does **not** fall back to pyfunc — it errors (pyfunc executes arbitrary code).

## 3.5 Alternatives considered

| Option | Cost | Failure mode | Verdict |
|---|---|---|---|
| **A. Dict (chosen)** | Small: one envelope builder + validator | A caller mutating `envelope["model"]` in place could desync `label_set` → validator runs at train + load boundaries only | **Chosen** — self-describing, matches existing `_meta` dict precedent |
| B. List `[model, label_set, registry_ref, onnx_sidecar]` | Smaller diff today | Positional coupling; fifth key breaks all unpacking sites; no deprecation path | Rejected — violates additive-extension rule; singular choice is dict, list is not supported |
| C. Keep bare object + parallel datasets (`best_model_labels`, `best_model_ref`) | Zero envelope code | N artifacts to keep consistent; serving must reconstruct state from 4 catalog reads; GDPR erasure must touch 4 places (Contract 5) | Rejected — replicates catalog collision at the artifact level |

## 3.6 Migration / dual-read / rollout

- **Read:** `load_best_model(obj)` accepts bare adapter (v1) or v2 dict. Bare → wrap into v2 (`label_set` from `model_classes(...)` or `None`, others `None`). Dict → validate keys + `label_set`/`classes_` consistency. Both yield the same downstream type. Every internal consumer unwraps with `unwrap_envelope` (C1); no consumer indexes `['model']` directly.
- **Write:** `train_models` + `calibrate_model` emit v2 dict whose `model` is the raw estimator (`adapter.model`, C2). The catalog `filepath` still points at the pickle bytes of `envelope["model"]` (Kedro `RestrictedPickleDataset` serializes the dict; `restricted_loads` returns the dict). No `filepath` rename in P0. Calibrated envelopes additionally require the 7 `_SAFE_GLOBALS` additions in §3.2 — P1 lands those tuples in the same change as the calibrated round-trip test, or calibrated envelopes remain unloadable.
- **Serving:** `FileModelLoader`/`RegistryModelLoader` gain only the `.onnx` rejection branch above **plus** `RegistryModelLoader.resolved_version` storage (`model_loader.py:575`, §3.2); existing `model_loader(path)` calls without flags behave identically to v1. Registry-loaded bare estimators rebuild `label_set`/`registry_ref` by the `get_model_version_by_alias(...).tags` mechanism in §3.2 (C2).

## 3.10 `unwrap_envelope` consumers (C1, mandatory — exhaustive)
Every site that can receive a `best_model`/`calibrated_model` object calls `unwrap_envelope` exactly once at entry. No site may duplicate the `isinstance(..., dict)` test.

| Consumer | Change in P1 |
|---|---|
| `calibrate_model` (`calibration.py:237`) | `model = unwrap_envelope(model)` first line; regression/`none` passthrough returns `wrap_envelope(...)` (§3.3) |
| `_predict` (`pipeline.py:239-241`) | `return unwrap_envelope(model).predict(x_test)` |
| `_predict_proba` (`pipeline.py:244-248`) | `return unwrap_envelope(model).predict_proba(x_test)` (classification only; `None` for regression) |
| `explain_model` (`pipeline.py:319`) | `model = unwrap_envelope(model)` at entry before `shap_explain(model, ...)` |
| `make_charts` wiring (`pipeline.py:485-489` node) | `make_charts(y_true, y_pred, y_proba, x_test, params)` (`charts.py:1743`) never receives the envelope itself — its model dependency arrives as `y_pred_proba` from `_predict_proba`, which unwraps (C1). The wiring change required is therefore in `_predict_proba`/`_predict` only; `make_charts` must not be passed `calibrated_model` directly. If a future chart needs the estimator it must call `unwrap_envelope` explicitly |
| Loader (`serving/model_loader.py`, `load_best_model`/`load_model`) | dict envelope → `unwrap_envelope(envelope)`; file/registry bytes → `restricted_loads` then envelope-wrap or bare-estimator tag reconstruction per §3.2 (C2) |

## 3.7 Catalog / serving impact

- Catalog v2 adds no ONNX entry in P0 (Contract 1) + no `metadata.model_flavour` dispatch for `.onnx` (P0 rejection lives in the loader, not catalog metadata). The existing `best_model` entry gains `metadata: {schema: best_model.v2}` once writers emit dicts (annotation only; readers accept both).
- Serving `app.py` `_resolve_loader` (`serving/app.py:551-586`) unchanged in shape: it returns a callable; P0 rejection lives inside the loader, not the app.

## 3.8 Failure modes

| Failure | Observable behavior | Handling |
|---|---|---|
| `label_set` length ≠ proba width | `ValueError` at envelope validation (train-time) / `ModelCorruptError` (load-time) | Never serve mislabeled proba |
| `label_set` order ≠ `classes_` | `ValueError` at train-time | Fail before artifact write |
| ONNX sidecar present in envelope | Pickle path used; sidecar recorded only (P0) | No session use until P10 |
| `.onnx` path load | `ModelCorruptError` (P0 rejection rule, always) | Explicit, not silent; no pyfunc fallback |
| Dict missing `model` key | `ValueError('best_model envelope requires model')` | Fail loud at boundary |

## 3.9 Test strategy (observable behavior, TDD, ≥85% coverage)

- **RED first:** `tests/test_best_model_v2.py` — (1) train on tiny fixture → envelope is dict with 4 keys and `envelope['model']` is the raw estimator (not the adapter; uncalibrated `envelope['model']` round-trips through `restricted_loads`); (2) `calibrate_model` returns envelope with unchanged `label_set`, regression passthrough also returns envelope; (3) `_predict_proba` shapes: binary `(n,2)`, 3-class `(n,3)`, regression `None`; (4) loader: pickle-via-restricted works, `.onnx` path raises `ModelCorruptError`, pyfunc still gated (mock `mlflow.pyfunc.load_model` never called when `allow_pyfunc=False`); (5) `model_classes` on adapter / `CalibratedClassifierCV` / `_BetaCalWrapper` fixtures; (6) `mlflow.sklearn.log_model` receives the unwrapped estimator. No ONNX session test in P0 (deferred to P10). All on `predict`/`predict_proba` outputs.
- **Calibrated-envelope round-trip RED test (C2-critical, FIX3 — separate from the uncalibrated fixture above):** `test_calibrated_envelope_restricted_roundtrip` in `tests/test_best_model_v2.py` (or `tests/test_core.py` allowlist suite): (a) fit `CalibratedClassifierCV` sigmoid **and** isotonic on a tiny fixture, wrap via `calibrate_model`/envelope, `pickle.dumps` the envelope, assert `restricted_loads(bytes)` succeeds and the restored `unwrap_envelope(restored).predict_proba(x)` equals the pre-pickle proba (allclose); (b) betacal case (guarded `importorskip('betacal')`): `_BetaCalWrapper` + fitted `BetaCalibration` envelope round-trips identically. This test **fails before** the 7 additions in §3.2 and passes after — an uncalibrated-only round-trip does not satisfy it. A pickletools assertion in the same test checks the residual globals are exactly the 7 tuples.
- **Mocked-client loader test (C2-high, FIX3):** `test_registry_loader_resolved_version_and_tags` with a mocked `MlflowClient`: `get_model_version_by_alias(name, alias)` returns a fake version object (`version='7'`, `tags={'label_set': '["0", "1"]', 'task_type': 'classification'}`), `_download_bytes` returns `pickle.dumps(fitted_estimator)`; assert (i) `loader.load()` returns the estimator, (ii) `loader.resolved_version == '7'` (no log parsing), (iii) the caller reconstruction `json.loads(version_obj.tags['label_set'])` yields `['0','1']` and `registry_ref['version'] == loader.resolved_version`, (iv) fallback path: with `tags={}`, `model_classes(estimator)` (estimator = `restricted_loads` return, not raw bytes) yields the same `label_set`. Also assert the writer side passes `label_set`/`task_type` via `log_model(tags=...)` (mock `sklearn.log_model` captures `tags`), not only via run `set_tags`.
- **GREEN/REFACTOR:** envelope builder/validator + `.onnx` rejection branch + 7 `_SAFE_GLOBALS` additions + `resolved_version` storage. **Coverage:** ≥85% of new envelope + rejection code; `model_loader.py` total coverage must not drop.

**P1 handoff (pseudocode only):**
```python
def unwrap_envelope(obj) -> estimator:
  # C1 single mechanism: dict envelope -> obj['model']; bare -> obj itself
  if isinstance(obj, dict) and 'model' in obj: return obj['model']
  return obj
def wrap_envelope(model, label_set, registry_ref=None, onnx_sidecar=None) -> dict: ...
  # C2: model must be adapter.model (raw estimator); never the adapter itself
def model_classes(model) -> array | None: ...  # C3: envelope -> adapter.model -> classes_; _BetaCalWrapper -> base.classes_
def validate_envelope(env, task_type) -> None: ...  # label_set vs model_classes, proba width; erasure field if present (§5.2(d))
def load_model(obj, allow_pyfunc=False): ...  # pickle-restricted; .onnx -> ModelCorruptError (P0); pyfunc gated; unwrap via unwrap_envelope
# modeling/pipeline.py nodes (C1 call sites):
#   train_models_node: outputs best_model envelope (wrap_envelope)
#   calibrate_model_node: calibrated_envelope = wrap_envelope(calibrate(unwrap_envelope(best_model)))
#   _predict node: unwrap_envelope(calibrated_model).predict(x_test)
#   _predict_proba node: unwrap_envelope(calibrated_model).predict_proba(x_test) | None
#   explain_model node: unwrap_envelope(calibrated_model) before shap_explain
#   make_charts node: inputs [y_test, y_pred, y_pred_proba, x_test, params] only (no envelope input; y_pred_proba already unwrapped upstream)
# registry (C2): mlflow.sklearn.log_model(sk_model=unwrap_envelope(model), ..., tags={'label_set': json.dumps(label_set), 'task_type': task_type})  # version tags via log_model(tags=...) at model_registry.py:785-787, NOT run set_tags at :783-784
# loader (C2): RegistryModelLoader.resolved_version set from model_loader.py:575 local; returns bare estimator (restricted_loads return at :588);
#   caller: version_obj = MlflowClient.get_model_version_by_alias(name, alias); label_set = json.loads(version_obj.tags['label_set']) or fallback model_classes(estimator)
#   registry_ref = {name: loader.model_name, alias: loader.alias, version: loader.resolved_version}
# _SAFE_GLOBALS (C2-critical): add exactly the 7 calibrated tuples in §3.2; calibrated envelope round-trip test must pass
```

---

# Contract 4 — Temporal split API (`temporal_split` + `assert_no_leakage` frozen; purged CV deferred)

**ADR:** D-TEMPORAL-SPLIT-1 · **Owner:** `solution-architect` · **Status:** `frozen` (FIX4, 2026-11-08) · **GATE: Blocks P3 entry; P5 inherits via P3→P4→P5 chain (P5 entry = P4 exit) (see §4.8)**

## 4.1 Current-state evidence (file:line)

| Claim | Evidence |
|---|---|
| Three spellings collapse to one ordered split | `src/forecast_ml_framework/pipelines/modeling/data_split.py:35-62` `_resolve_split_options`: `('time','ordered','temporal') → 'time'`, `shuffle=False`; `data_split.py:176-184` `_ordered_split` (first rows train, last rows test, sorted by `time_col` if given) |
| `split_data` meta is flat and has no purge/embargo | `data_split.py:221-227` `meta = {task_type, test_size, random_state, sampling_method, requested_method, shuffle, time_col, n_train, n_test}` |
| Data-prep split is a thin wrapper | `src/forecast_ml_framework/pipelines/data_prep/nodes.py:858-913` `split_raw_data_with_meta` calls `split_data` with `feature_cols=[all non-target]`, returns `(train_raw, test_raw, split_meta_raw)`; `nodes.py:916-941` `split_raw_data` discards meta (compat wrapper) |
| Modeling presplit merges raw meta as `raw_*` | `src/forecast_ml_framework/pipelines/modeling/pipeline.py:133-236` `split_presplit_data` — `sampling_method='presplit'`, `shuffle=False`, merges `raw_sampling_method/raw_requested_method/raw_shuffle/raw_time_col` (`pipeline.py:231-234`); pipeline wiring `modeling/pipeline.py:436-441` |
| Data-prep pipeline wiring | `src/forecast_ml_framework/pipelines/data_prep/pipeline.py:77-83` `split_raw_data_with_meta: [validated_data, params:modeling_params, variables] → [train_raw, test_raw, split_meta_raw]` |
| No purged CV, no embargo anywhere | `grep -i purged|embargo` over `src/` → zero matches (verified at freeze); `data_split.py` has no cross-validation splitter at all |
| PIT leakage guard exists only in feature store | `src/forecast_ml_framework/feature_store/offline.py:243-312` `get_features` (strict_pit / `as_of <= request` filtering, `offline.py:202-241` `_combined_frame`); `online.py:234-304` same for online. No equivalent in the split path |

## 4.2 Proposed v2 — temporal API

**Single temporal entry point frozen in P0** (legacy `split_data` delegates, never duplicated). `purged_cv_folds` is **not** frozen in P0 — moved to P5/P7 charter (B4(c)):

```python
def temporal_split(df, params) -> (train_raw, test_raw, split_meta)
# params (all optional with defaults):
#   target_col: str = 'target'
#   time_col: str (required for method='temporal'; raises if missing)
#   method: 'presplit' | 'temporal' = 'temporal'   # 'presplit' = today's data_prep behavior (no re-split)
#   test_size: float in (0,1) = 0.2                 # FINAL fraction by ROW COUNT, POST-embargo (see fixed-point assignment below: test_start is frozen at N - n_test_target, so the returned n_test/N matches test_size)
#   embargo: timedelta | int (days) = 0               # gap dropped between train/test
#   random_state: int | None = 42                    # ignored for temporal/presplit (determinism)

def assert_no_leakage(meta) -> None: ...
# precise invariant (single formula, C5(b)):
#   max_time_train < min_time_test - embargo     (strict; gap := min_time_test - max_time_train must be > embargo)
# DEFERRED to P5/P7 charter (not P0): purged_cv_folds + calibrate_model prohibition
```

- **`method='presplit'`** = today's two-stage chain (`data_prep.split_raw_data_with_meta` + `modeling.split_presplit_data`) re-expressed as one contract: the caller already split; `temporal_split` validates + annotates `split_meta` without moving rows.
- **`method='temporal'`** = order by `time_col` (UTC-normalized, stable sort), then cut with the **fixed-point assignment** so the **final** test fraction equals `test_size` (FIX3-medium; this reconciles the signature comment above with this bullet — `test_size` is post-embargo, not pre-embargo):
  ```python
  N = len(sorted_df)
  n_test_target = ceil(test_size * N)                 # final, post-embargo test rows
  test_start = N - n_test_target                      # frozen; never moves (no loop, no cumulative cut)
  embargo_td = <embargo normalized once to time_col unit: int days -> Timedelta(days=embargo) for datetime time_col; int time-units unchanged for numeric time_col>
  # Fixed-point assignment: train horizon is derived from the frozen test_start,
  # not from a moving cut. Any count helper, if retained, must take the train
  # horizon max_time_train = time[train_end] explicitly — never a moving candidate cut.
  train_end = last index i in [0, test_start) with time[i] < time[test_start] - embargo_td
  #   (strict <; if no such i exists: raise ValueError('temporal_split: embargo consumes <test_size> fraction'))
  train = sorted_df[:train_end + 1]
  test = sorted_df[test_start:]
  embargoed = sorted_df[train_end + 1:test_start]     # gap rows between train_end+1 and test_start-1 (dropped_final)
  dropped_final = len(embargoed)
  n_embargoed = dropped_final                         # final dropped count only
  # invariant: len(test) == n_test_target, so len(test)/N == test_size within 1/N rounding;
  #   len(train) + n_embargoed + len(test) == N
  # TRACE (§4.9 gate): N=100 daily rows t=0..99, test_size=0.2, embargo=7d ->
  #   n_test_target=ceil(0.2*100)=20, test_start=80 (t=80), horizon t[80]-7d = t73,
  #   train_end = last i with t[i] < t73 = 72, train=[:73] -> n_train=73,
  #   test=[80:] -> len(test)=20, embargoed=[73:80] -> n_embargoed=7. 73+7+20=100. QED vs §4.9.
  ```
  Embargo gap rows are the rows **strictly between train and test** (`train_end+1 .. test_start-1`; neither train nor test), in time units not row counts when `time_col` is datetime — because `test_start` is frozen, embargo never shrinks the final test. The strict-`<` in the `train_end` definition inherently moves `train_end` earlier past duplicate timestamps at the horizon (next bullet). No shuffle ever. `split_meta['test_size']` records the requested post-embargo fraction; `n_test`/`n_train` record the final post-embargo counts.
- **Single strictness formula (C5(b)):** every temporal return satisfies `max_time_train < min_time_test - embargo`. With `embargo=0` this reduces to `max_time_train < min_time_test` strictly (equal boundary timestamps are leakage, not allowed). On dense data with duplicate timestamps at the horizon, the frozen-assignment definition of `train_end` (last index with `time < time[test_start] - embargo`) moves `train_end` earlier past all rows at or after the horizon until the strict inequality holds (`test_start` stays frozen); if every row shares one timestamp the split raises `ValueError` (frozen — never silently returned leaky or empty).
- **Timezone normalization (C5(a), series-level):** the scalar `core.params.parse_datetime` (`params.py:70-98`, naive → UTC) is not sufficient row-wise. `temporal_split` first runs the named check `_check_no_mixed_tz(time_series)` on the whole `time_col` **before** any conversion: if any value is naive and any value is aware → `ValueError('temporal_split: mixed naive/aware time_col ...')` (named check raising `ValueError`, never `TypeError` from comparing naive to aware). Otherwise the entire series is vectorized with `pd.to_datetime(time_series, utc=True, errors='raise')` (all values normalized to UTC in one pass); meta ISO strings are emitted from the normalized series. Mixed naive/aware never reaches the sort or the leakage comparison.
- **Deferred (B4(c)):** purged-CV folds and any prohibition on combining `calibrate_model` with temporal CV are moved to the P5/P7 charter. P0 freezes no CV helper, no fold return fields (any `folds`/`fold_train_test` field is undefined in P0 and must not be emitted), and no such prohibition; existing stratified CV in calibration is unchanged and out of P0 scope.
- **Named calibrate+temporal detection param (C5(d)):** P0 names — but does not enforce — the detection pair the P5/P7 charter will enforce: `params['method']` (or `split_meta['sampling_method']`) `== 'temporal'` **together with** `params['calibration']['method'] != 'none'` (the argument read at `calibration.py:263-264`). P0 workers record both values honestly in `split_meta` (`sampling_method`, `requested_method`) so P5 can detect the combination without new instrumentation; P0 raises nothing for the combination (enforcement deferred).

## 4.3 `split_meta` v2 contract

```python
split_meta = {
  "sampling_method": "presplit" | "temporal",   # actual method used (honest, as today)
  "requested_method": str,                      # what the caller asked for
  "shuffle": False,                             # always False for temporal/presplit
  "time_col": str | None,
  "test_size": float,
  "random_state": int | None,                   # None recorded for temporal/presplit determinism
  "n_train": int, "n_test": int,
  "n_embargoed": int,                           # rows dropped by embargo (0 for presplit/back-compat)
  "embargo": str | None,                        # normalized duration string, e.g. "P7D"
  "max_time_train": str | None,                 # ISO-8601 UTC max time_col in train
  "min_time_test":  str | None,                 # ISO-8601 UTC min time_col in test
  "leakage_checked": True,                      # see §4.5; False is invalid for temporal
  # back-compat keys preserved from data_split.py:221-227 and pipeline.py:220-235:
  # task_type, raw_* merge keys (raw_sampling_method etc.) still emitted by modeling presplit node
}
```

- `max_time_train < min_time_test - embargo` must hold for `method='temporal'` (C5(b) single formula); else `ValueError('temporal leakage: train/test overlap')`. For `presplit`, the check runs on the merged raw meta and records the result (`leakage_checked`, gap) without moving rows.

## 4.4 Alternatives considered

| Option | Cost | Failure mode | Verdict |
|---|---|---|---|
| **A. One API frozen now; purged-CV helper deferred (chosen)** | Small: delegates to `_ordered_split`; embargo = head-of-test row filter | UTC normalization edge cases (naive vs aware) → series-level `_check_no_mixed_tz` + `pd.to_datetime(..., utc=True)` vectorized pass (C5(a)); scalar `core.params.parse_datetime` (`params.py:70-98`) is for meta values only | **Chosen** — single leakage definition for all callers; CV deferred to P5/P7 charter (B4(c)) |
| B. Keep `split_data` + add `embargo` kwarg only | Smaller diff | CV still stratified/shuffled; temporal callers diverge on purge semantics | Rejected — does not close the leakage hole for the P3→P4→P5 chain |
| C. Require callers to use `feature_store` PIT for all splits | Large: feature store is point-in-time feature lookup, not a splitter | Splitting becomes a feature-store query; no fold concept | Rejected — wrong abstraction |

## 4.5 Leakage guard (binding)

- Every `method='temporal'` return passes `assert_no_leakage(meta)` implementing the **single** formula `max_time_train < min_time_test - embargo` (gap `> embargo`). Failure raises `ValueError` naming the gap; the split is never returned in a leaky state. The `<=` / `>=` variants are not alternative readings — strict `<` is the frozen rule (C5(b)).
- `presplit` validates the caller-supplied cut with the same function and annotates meta; it never silently reorders.
- Timezone rule: the whole `time_col` series must pass `_check_no_mixed_tz` (C5(a)) — mixed naive/aware raises `ValueError` before normalization; uniform naive values are interpreted as UTC (documented, matches `feature_store._naive_utc`) and uniform aware values are vectorized to UTC via `pd.to_datetime(..., utc=True)`. No scalar-per-row parsing in the split path.

## 4.6 Migration / dual-read / rollout

- New `temporal_split` is additive (`purged_cv_folds` is not added in P0 — deferred). `split_data(df, params)` with `method ∈ {time, ordered, temporal}` delegates to `temporal_split(..., method='temporal')` and keeps its current return shape + meta keys (plus new `n_embargoed`/`embargo` keys defaulting to `0`/`None`) — existing callers and tests keep passing.
- `split_presplit_data` keeps its `sampling_method='presplit'` behavior; workers add the leakage annotation + `n_embargoed: 0` to its meta (additive key).
- No catalog rename: `split_meta_raw` (`catalog.yml:549-550`) and `split_meta` (`catalog.yml:541-542`) keep their `MemoryDataset` types.

## 4.7 Catalog / serving impact

- No new catalog datasets. `split_meta_raw`/`split_meta` values gain fields; consumers reading only known keys are unaffected (additive dict — same pattern as `fitted_registry` v2).
- Serving impact: none (splitting is training-only).

## 4.8 **P3-entry gate (explicit, reconciled — B1)**

> **Blocks P3 entry; P5 inherits via P3→P4→P5 chain (P5 entry = P4 exit).** P3 (time-split entry work) must not start until: (1) this ADR is flipped to `frozen` with anti-team `APPROVED`, (2) `temporal_split` + `assert_no_leakage` are implemented (TDD), and (3) the leakage-guard test suite (§4.9) passes with ≥85% coverage on the new code. A P3/P4/P5 plan that embeds its own split logic instead of calling `temporal_split` is a gate failure — the Lead must reject it. `split_data(df, params)` with a temporal method counts as calling `temporal_split` (delegation), stratified/shuffled splits do not. This is the single gate statement for this contract.

## 4.9 Test strategy (observable behavior, TDD, ≥85% coverage)

- **RED first:** `tests/test_temporal_split.py` — (1) temporal split on time-ordered fixture: `max_time_train < min_time_test - embargo`; (2) embargo drops N rows from the test head only and `n_embargoed == N`; (3) overlapping fixture raises `ValueError` (leakage, via `assert_no_leakage`); (4) `presplit` leaves rows untouched but annotates meta; (5) mixed naive/aware `time_col` raises `ValueError` (not `TypeError`) naming the tz check; (6) dense duplicate-timestamp cut advances past the boundary (strict `<`) or raises when all timestamps coincide. No purged-CV or calibrate-combination enforcement test in P0 (deferred to P5/P7 charter; detection values recorded only). All on `(train, test, meta)` behavior.
- **Post-embargo fraction test (FIX3-medium):** `test_temporal_split_post_embargo_fraction`: on a 100-row daily fixture with `test_size=0.2` and `embargo=timedelta(days=7)` (7 rows dropped), assert `len(test) == 20` and `len(test)/len(df) == 0.2` (within 1/N), `n_embargoed == 7`, and `n_train == 73` (overshoot moved the cut earlier by 7) — i.e. the final fraction is post-embargo, not the pre-embargo 27/100 a naive cut-then-drop would return. Same assertion with `embargo=0` as control (`len(test) == 20`).
- **GREEN/REFACTOR:** `temporal_split` + `_check_no_mixed_tz` + `assert_no_leakage`. **Coverage:** ≥85% of new module; `data_split.py` delegation lines 100% covered.

**P1 handoff (pseudocode only):**
```python
def _check_no_mixed_tz(series) -> None: ...          # C5(a): whole-column check; ValueError on mixed naive/aware
def temporal_split(df, params) -> (train, test, split_meta): ...  # method presplit|temporal, head-of-test embargo with post-embargo overshoot (final len(test)/N == test_size), strict max_train < min_test - embargo
def assert_no_leakage(meta) -> None: ...                           # single formula: max_train < min_test - embargo
# DEFERRED to P5/P7 charter: purged_cv_folds (no return fields in P0) + calibrate+temporal prohibition (detection pair: method=='temporal' and calibration.method != 'none')
```

---

# Contract 5 — Secrets / retention / PII / GDPR contract

**ADR:** D-SECRETS-PII-GDPR-1 · **Owner:** `solution-architect` + `security` · **Status:** `frozen` (FIX4, 2026-11-08)

## 5.1 Current-state evidence (file:line)

| Claim | Evidence |
|---|---|
| `scrub.py` is the sole stdlib-only leaf | `src/forecast_ml_framework/governance/scrub.py:4-60` (docstring: "stdlib-only, no telemetry or governance imports… this leaf exists so `telemetry.exceptions` can scrub without importing `governance.audit`"); `scrub.py:66` `REDACTED = '***REDACTED***'` |
| Audit + telemetry both import the leaf (no second scrubber) | `src/forecast_ml_framework/governance/audit.py:12-18` (imports `REDACTED`, `scrub_details`, `scrub_text`…); `src/forecast_ml_framework/telemetry/exceptions.py:58-69` (imports `scrub_details`, `scrub_type_name`); `src/forecast_ml_framework/governance/__init__.py:38` (`_SUBMODULES = {audit, lineage, retention, scrub}`) |
| `audit_log` requires actor, scrubs details | `audit.py:78-116` (actor required, `details` via `scrub_details`, ISO timestamp) |
| Retention is a pure keep/delete split (no enforcement) | `src/forecast_ml_framework/governance/retention.py:4-11` ("Pure functions only: no persistence and no deletion happens here"); `retention.py:109-147` `check_retention → {'keep':…, 'delete':…}`; strictly-greater-than (`retention.py:150-153` `_is_kept`: `age is None or age <= days`) |
| No erasure reaching artifacts | `grep -i gdpr|erasure|pii` over `src/` → zero matches (verified at freeze); `feature_store/offline.py:56-91` `materialize` + `online.py:163-183` `set_features` have no retention/erasure hooks |
| Feature stores are the PII-bearing surfaces | `offline.py:17-54` (versioned artifact snapshots by `as_of`); `online.py:24-56` (memory + lazy Redis, `_records_for` merge) — both keyed by `entity_id`, features are arbitrary dicts |
| Catalog credentials never in YAML | `catalog.yml:2-4` (credentials from env); `catalog.yml:68-77` SQL is trusted config (Spark BQ dataset enforces single-statement read-only) |

## 5.2 Proposed contract

**a) Secrets — `scrub.py` is the sole leaf (no second implementation):**

- Every layer that touches log/audit/telemetry context imports from `governance.scrub` (`scrub_details`, `scrub_text`, `scrub_type_name`, `REDACTED`). New code must not implement its own regex/pattern scrubbing; a second scrubber is a review failure.
- Public aliases at the end of `scrub.py` (`safe_str`, `scrub_text`, …) are the cross-layer API; `_`-prefixed names are identical aliases, not a second API.
- **Sole sha256 leaf (C6):** erasure hashing lives only in `governance/scrub.py` as `subject_hash`:

  ```python
  def subject_hash(subject_id: str) -> str:
    # stdlib-only (hashlib), 64-hex lowercase; the ONLY sha256 subject-hash in the codebase
    ...
  ```

  No second implementation may appear in `fitted.py`, `calibration.py`, `model_registry.py`, or any erasure caller — every marker writer calls `scrub.subject_hash(subject_id)`. This keeps the leaf stdlib-only and dependency-free per `scrub.py:4-60`.
- Secrets come from env only: `GOOGLE_APPLICATION_CREDENTIALS` (GCP), `AWS_PROFILE`/AWS env (AWS), `MODEL_SHA256`/`FORECAST_MODEL_SHA256` (artifact pin) — same sources as today (`model_loader.py:104-156`). No secret value may appear in catalog YAML, parameters, or IaC (P2 gate).

**b) PII tagging — catalog metadata (additive):**

```yaml
# per-dataset, additive metadata on every catalog entry:
metadata: { pii_classes: [email, user_id], retention_days: 365, erasure_key: entity_id }
```

- `pii_classes` is a closed vocabulary maintained in one place (`governance/pii.py` at P1: `email, user_id, device_id, location, financial, health, none`); unknown class → config error. `none` is explicit (datasets with no PII state it; missing `pii_classes` means "unclassified" and blocks promotion to prod — P11 gate).
- Feature-store writes (`OfflineFeatureStore.materialize`, `OnlineFeatureStore.set_features`/`materialize`) **require** the frame's `pii_classes` to be declared (parameter or store-level) and record it in snapshot metadata. Writes with undeclared PII raise `ValueError` (fail-closed at the boundary, matching `audit_log` actor-required precedent).

**c) Retention enforcement — online + offline (caller deletes, policy decides):**

- `check_retention` stays pure (`retention.py:109-147` unchanged in signature). Enforcement is a new **caller** function that both stores invoke:
```
enforce_retention(store, params) -> {keep, delete, audit_entry}
  plan = check_retention(list_snapshots(store), params)
  for key in plan['delete']: store.delete(key)        # store implements delete; refusal surfaces as error
  audit_log('retention_sweep', actor, {swept: n_deleted, kept: n_kept})  # via scrub leaf
```
- Covers **both** surfaces: offline snapshots (`offline.py` keys) and online entity records (`online.py:_store` + Redis `feature:{entity}`). Online enforcement deletes the *record* (entity + timestamps), not just the Redis key — memory is always purged and Redis is purged whenever a client exists (`_client() is not None`, conditional rule below; `_records_for` merge source list is the checklist).
- Future-dated artifacts (age < 0) are kept and logged; unknown age is kept (existing conservative rule, `retention.py:150-153` preserved).
- **Delete surface (C6, exact signatures; P1 implements all):**

  ```python
  # core/artifact_store.py
  class ArtifactStore(abc.ABC):            # artifact_store.py:861
    def delete(self, path): ...            # abstract: delete object at path (file key or object key)

  class LocalArtifactStore(ArtifactStore):
    def delete(self, path): ...            # _full_path(path); os.remove if file exists, else no-op (idempotent)

  class GCSArtifactStore(ArtifactStore):
    def delete(self, path): ...            # _validate_cloud_key(path); blob.delete()

  class S3ArtifactStore(ArtifactStore):
    def delete(self, path): ...            # _validate_cloud_key(path); s3.delete_object(Bucket=bucket, Key=key)

  # feature_store/online.py
  class OnlineFeatureStore(FeatureStore):
    def delete(self, entity_id): ...       # see conditional-Redis rule below
  ```

  **Conditional Redis delete (FIX3-medium):** `OnlineFeatureStore.delete(entity_id)` is exactly:
  ```python
  def delete(self, entity_id):
    self._store.pop(entity_id, None)          # memory always purged
    c = self._client()                        # online.py:59-82: returns the supplied client, lazily builds from redis_url, or None
    if c is not None:                         # client may be None (no url/client, redis package missing, from_url failed) — never call delete unconditionally
      c.delete(f'feature:{entity_id}')        # Redis only when a client exists
  ```
  The client's `get`/`set` are the only methods the existing lazy-Redis path requires (`online.py:24-30`, `_redis_get`/`_redis_put`); a `delete` call on a `None` client would crash memory-only stores, so the `is not None` guard is normative. Fake Redis clients in tests (and any P1-supplied client used with `delete`) must implement `delete`; memory-only stores (`_client() is None`) must purge without error.

  `delete` is idempotent everywhere (missing key is not an error — retention/erasure sweeps must be retryable). Offline snapshot deletes route through `ArtifactStore.delete` on the snapshot key; online entity purges route through `OnlineFeatureStore.delete(entity_id)` above (memory always; Redis only when `_client()` is not `None`).

**d) GDPR erasure — reaches `fitted_registry` + `best_model` artifacts:**

```python
def erase_subject(subject_id, *, stores, artifacts, actor) -> ErasureReport
# stores: feature stores (offline + online), catalog datasets carrying PII
# artifacts: fitted_registry (JSON), best_model envelope (pickle dict), predictions/monitoring outputs
# Returns: {subject_id, removed: [...], retained_legal_hold: [...], audit_id}
```

- **Feature stores:** all snapshots containing `subject_id` as `entity_id` (or any `pii_classes`-tagged column value) are removed from offline snapshots (row-level removal + snapshot rewrite; empty snapshot → key deleted) and online records (memory always + Redis entry when `_client()` is not `None`, per the conditional rule in §5.2(c)), then verified by re-read (`get_features` returns no row for the subject).
- **`fitted_registry` handling (Contract 2):** if the registry's training data included the subject, the registry cannot be "row-deleted" (it holds statistics, not rows). Erasure records a **legal-hold/retrain-required** marker in the single exact field `__meta__.erasure = {"subject_hash": sha256(subject_id), "requested_at": iso, "action": "retrain_required"}` (hashed via `scrub.subject_hash(subject_id)` — never clear, full IDs never logged). The subject hash is computed only by `subject_hash` (§5.2(a)). Validator: `read_registry` / `_check_registry_version` (Contract 2) — if `__meta__.erasure` is present it must be a dict with exactly `subject_hash` (str, 64 hex), `requested_at` (ISO-8601 str), `action == "retrain_required"`, else `ValueError`; the marker is excluded from steps (`fitted.py:51-53`) so scoring is unaffected. The registry's fitted statistics are **not** silently edited; erasure is verified by flagging the artifact for retrain (data deletion triggers retraining pipeline, `pipelines/retraining/`). P1 worker implements the marker writer; P0 freezes the semantics.
- **`best_model` envelope (Contract 3):** exact field is the single top-level envelope key `erasure = {"subject_hash": sha256(subject_id), "requested_at": iso, "action": "retrain_required"}` (not `__meta__`, not registry-adjacent — this is the single exact name, reconciled with the B-minor exact-field rule: exactly one marker key per artifact, no duplicates or aliases). It is the only allowed 5th key on the v2 dict. Validator: `validate_envelope(env, task_type)` (Contract 3 §3.9) — if `erasure` is present it must match the shape above exactly (hash via `subject_hash`), else `ValueError`; the model object is not mutated. Erasure report lists the envelope path as `retrain_required`.
- **Re-save path (C6; P1 implements):** `erase_subject` writes the marker and re-saves both artifacts through their existing catalog/dataset writers (no new persistence path):
  1. Registry: `registry['__meta__']['erasure'] = {subject_hash, requested_at, action}` (in place on the loaded dict) → re-serialize via the `fitted_registry` `JSONDataset` (catalog `fitted_registry`) exactly as `fit_preprocess` output is stored (same `registry_to_json` / save path, v2 nested steps unchanged).
  2. Envelope: `envelope['erasure'] = {subject_hash, requested_at, action}` (top-level key, in place) → re-pickle via the `best_model` `RestrictedPickleDataset` exactly as `RestrictedPickleDataset.save` stores it (`core/datasets.py:471` equivalent of `pickle.dumps`), never via a new store.
  3. Feature rows: `OnlineFeatureStore.delete(entity_id)` / offline snapshot `ArtifactStore.delete(key)` (C6) — deletion, not marker, for row data.
  A second erasure field name, a second hash helper, or a new save path is a review failure.
- **Predictions / monitoring / audit outputs:** rows for the subject are deleted where row-keyed; `audit_log` entries themselves are never deleted (audit trail integrity) — erasure appends an `erasure` audit entry referencing the subject hash.
- **Verification (observable):** after `erase_subject`, `get_features([subject_id])` → empty, registry `__meta__.erasure` and envelope `erasure` pass their validators, `apply_preprocess` smoke on the flagged registry still runs (marker is metadata, not a step), and `ErasureReport` + audit entry exist. Report + audit entry are the proof.

**e) Rotation + residency dispositions (B3, explicit — D-SECRETS-PII-GDPR-1 remains freezable):**

- **Rotation:** out-of-band via cloud, no in-repo rotation in P0. Secret rotation is performed by the cloud secret manager / IAM outside this repo; P0 adds no rotation code, no rotation schedule, and no in-repo key material. Any future in-repo rotation helper requires a new ADR.
- **Residency:** region-agnostic; decision at P2 IaC before P11. P0 pins no region and makes no residency claim; the P2 IaC task must record the region/residency decision before the P11 production gate. Artifacts created before that decision carry no residency guarantee.

## 5.3 Alternatives considered

| Option | Cost | Failure mode | Verdict |
|---|---|---|---|
| **A. Leaf + tags + enforce caller + marker (chosen)** | Small: one tag vocabulary, one enforce function, one marker | Marker could be ignored by a lazy consumer → P11 gate checks marker presence on PII artifacts | **Chosen** — minimal, uses existing pure `check_retention` + scrub leaf |
| B. New scrubber per layer | Duplicates evasion logic (filler/confusable handling in `scrub.py:92-259`) | Two scrubbers drift; a marker caught by one leaks via the other | Rejected — `scrub.py` docstring already forbids this (`scrub.py:54-56` cycle rationale) |
| C. Tombstone rows in place (soft delete) | Easy at write time | Tombstones still carry PII values; `get_features` without tombstone filter leaks; GDPR requires actual erasure | Rejected — soft delete ≠ erasure |

## 5.4 Trade-off record

- **Decision:** Option A. **Gain:** one scrub implementation, honest erasure semantics (flag-for-retrain instead of fake "edited statistics"). **Give up:** erasure does not instantly purge information from already-fitted statistics — that requires retraining (explicitly reported, never hidden). **Reverse if:** legal requires statistical erasure proof without retraining — then Contract 2 needs differential-privacy bounds (out of P0 scope, new ADR).

## 5.5 Migration / dual-read / rollout

- Additive: existing datasets without `pii_classes` read as `unclassified` (allowed in dev, blocks P11 prod gate). `check_retention` signature unchanged. `erase_subject` is a new function (no v1 behavior change).
- Rollout: (1) tag catalog entries, (2) add enforce caller + wire cron/job in P2 IaC, (3) implement marker writer + `erase_subject` in P1. Retention windows (`retention_days` per dataset) are set at tag time; a global default (365d) applies only where a dataset omits it explicitly — omission is logged.

## 5.6 Catalog / serving impact

- Catalog v2 gains `metadata: {pii_classes, retention_days, erasure_key}` per entry (Contract 1 grouped catalog carries these). Serving impact: serving responses must not echo PII fields from feature lookups — `OnlineFeatureStore.get_features` callers at the serving edge pass `feature_names` explicitly; the contract requires the serving layer to allowlist response fields (P10 gate).

## 5.7 Failure modes

| Failure | Observable behavior | Handling |
|---|---|---|
| Feature write without `pii_classes` | `ValueError` | Fail-closed; no silent write |
| Retention delete fails (store error) | Error surfaces, sweep aborts, audit entry records partial | Never mark swept on failure |
| Erasure finds no subject | `ErasureReport(removed=[], …)` + audit entry (proof of check) | Not an error |
| Registry marker write conflicts with dual-read | Marker lives in `__meta__` (excluded from steps per `fitted.py:51-53`) | Scoring unaffected; verified by smoke test |
| Second scrub implementation proposed | Review blocker | Single leaf rule (`scrub.py:4-60`) |

## 5.8 Module boundaries

- **Owner:** `solution-architect` + `security` (contract) → workers (tags, enforce caller, erasure). **Must never depend on:** `telemetry` (leaf rule: `governance.scrub` imports nothing from `telemetry`/`audit` — the dependency arrow points one way).

## 5.9 Test strategy (observable behavior, TDD, ≥85% coverage)

- **RED first:** `tests/test_governance_v2.py` — (1) `check_retention` + `enforce_retention` on fake offline/online stores: expired snapshot/record deleted, kept + unknown-age retained, audit entry appended; (2) undeclared-PII write raises; (3) `erase_subject` on fake stores + registry + envelope fixtures: feature rows gone (re-read empty), registry/envelope carry hashed `erasure` marker, report returned, clear subject ID never appears in report/audit (string-search assertion via scrub leaf); (4) evaluation: no second `scrub_*` implementation exists (`grep` guard test — import surface only, not regex). All on store reads + report contents.
- **Fake-client online-delete test (FIX3-medium):** `test_online_delete_conditional_redis`: (a) fake Redis client implementing `get`/`set`/`delete` (records `delete` keys) supplied to `OnlineFeatureStore` → `delete(entity)` purges memory **and** calls `fake.delete('feature:<entity>')` exactly once; (b) memory-only store (no client, no `redis_url` → `_client()` returns `None`) → `delete(entity)` purges memory with **no** exception and no Redis call. Both must pass; an unconditional `client.delete(...)` fails (b).
- **GREEN/REFACTOR:** tag vocabulary, enforce caller, `erase_subject`. **Coverage:** ≥85% of new governance code; `retention.py` pure-function lines stay 100% (existing tests unchanged).

**P1 handoff (pseudocode only):**
```python
def subject_hash(subject_id: str) -> str: ...        # C6: sole sha256 leaf in governance/scrub.py (hashlib, 64-hex)
def enforce_retention(store, params, actor) -> dict: ...          # check_retention + delete + audit
def erase_subject(subject_id, stores, artifacts, actor) -> dict: ...  # remove rows (delete), flag registry/envelope (subject_hash), re-save both artifacts, audit
# C6 delete surface (exact signatures):
#   ArtifactStore.delete(self, path) [ABC] -> LocalArtifactStore.delete(self, path) | GCSArtifactStore.delete(self, path) | S3ArtifactStore.delete(self, path)
#   OnlineFeatureStore.delete(self, entity_id)  # memory _store.pop always; c = self._client(); if c is not None: c.delete(f'feature:{entity_id}')
PII_CLASSES = (email, user_id, device_id, location, financial, health, none)
```

---

# Cross-cutting contracts

## Module boundaries (one direction only)

```
conf (YAML)  ──►  pipelines (data_prep / modeling / scoring / …)  ──►  catalog datasets
        fitted.py (reader/writer) ◄── score_data_prep, apply_preprocess nodes
        calibration.py (envelope in/out) ◄── train_models (envelope out)
governance.scrub (leaf, stdlib-only) ◄── audit / telemetry / retention / erasure / serving loader
serving/model_loader (pickle + .onnx rejection) ◄── catalog metadata · envelopes · ONNX sidecar (recorded only, P0)
feature_store (offline/online) ◄── retention enforcer · erasure (PII-tagged writes)
```

- Nothing imports "up": `scrub.py` imports nothing from the framework; `fitted.py` imports only step helpers; loaders import `artifact_store` allowlist only.
- Every boundary validates (contract-first, `api-and-interface-design`): envelope validation at train+load, registry validation at read, catalog load validation (grouping) at load, PII declaration at write, leakage assert at split.

## Failure-mode summary (all contracts)

| Mode | Where | Rule |
|---|---|---|
| Version/schema mismatch | `fitted_registry` | Fail loud `ValueError`, never silent identity |
| Flavor mismatch | `best_model` loader | `ModelCorruptError`, no URI/path leak, no pyfunc fallback for `.onnx` (P0 rejection) |
| Catalog collision | deferred fragments (future migration) | `ValueError` from custom `check_catalog_duplicates` (`yaml.compose` + `file:line`, §1.2(b)) naming both locations (when migration lands; P0 single-file grouping has no merge mechanism or helper) |
| Temporal leakage | split | `ValueError`, split never returned leaky |
| PII undeclared / secret in log | governance | Fail-closed write / scrubbed log (single leaf) |
| Retention delete failure | enforce | Abort sweep, audit partial, never claim swept |

## Scaling / security boundaries

- **Scaling lever:** single grouped catalog stays file-based (no runtime service); pickle serving scales by replica; feature-store retention scales by snapshot key prefix (`offline.py:96-113` `_all_keys`); temporal split is caller-invoked (no cluster service). ONNX session scaling is deferred to P10.
- **Trust boundaries:** catalog YAML + parameters = trusted config (never secrets); pickle bytes = untrusted until `restricted_loads`; registry URIs = trusted, bytes = not (per `model_loader.py:361-366`); PII fields = only cross the feature-store → serving boundary via explicit `feature_names` allowlist. (`.onnx` bytes are rejected in P0, never loaded.)

---

# Acceptance trace (task P0-2, as revised by P0-2-FIX)

- [x] 5 contracts, each with evidence `file:line` + migration + tests — §§1–5.
- [x] `best_model` choice is **singular** (dict, §3.2; list explicitly rejected, §3.5).
- [x] Temporal API gate is **Blocks P3 entry; P5 inherits via P3→P4→P5 chain (P5 entry = P4 exit)** (§4.8, single gate statement — B1).
- [x] Catalog ADR orphan fixed: D-CATALOG-V2-1 proposal at top, owner `solution-architect`, acceptance = fragment merge rules + ONNX sidecar entry + per-task split; planner adds row to `decisions.md` on freeze (B2).
- [x] Secrets dispositions: rotation out-of-band via cloud, no in-repo rotation in P0; residency region-agnostic, decision at P2 IaC before P11 (§5.2(e) — B3).
- [x] YAGNI cuts: ONNX session deferred to P10 (dict keys + `.onnx` rejection only, §3.4); ONE-catalog grouping now, 6-fragment migration deferred (§1.2(b)); only `temporal_split` + `assert_no_leakage` frozen, purged CV + calibrate prohibition to P5/P7 charter (§4.2); migration echo deleted (§2.2) (B4(a)–(d)).
- [x] Minor: `All paths relative to forecast_ml_framework/` header; exact erasure fields (`__meta__.erasure` + envelope `erasure`) with validators (§5.2(d)); typos `opset`/`Kedro` fixed.
- [x] P0-2-FIX2 (C1–C6, H1–H2): `unwrap_envelope` single mechanism + all consumers + modeling/pipeline.py pseudocode (§3.2/§3.10); `envelope['model']` = raw `adapter.model` estimator, `log_model` unwrap, loader tag reconstruction (§3.2) — **the FIX2 "no `_SAFE_GLOBALS` additions" sub-claim is superseded by FIX3 below (calibrated case)**; `model_classes` + multiclass `proba[:,1]` behavior (§3.3); no P0 `best_model_onnx` Parquet entry, no fragment-merge/`CatalogMergeError` claim, ONE `catalog.yml` (§1, D-CATALOG-V2-1); series tz check `_check_no_mixed_tz` (`ValueError`), head-of-test embargo, single `max_train < min_test - embargo` formula, purged folds fully deferred, named calibrate+temporal detection pair (§4.2/§4.5); `ArtifactStore.delete` ABC + Local/GCS/S3 + `OnlineFeatureStore.delete`, sole `scrub.subject_hash` leaf, single marker keys + re-save paths (§5.2); `data_prep` `apply_preprocess` params wiring (§2.5 H1); nested-only v2 writer + consistency check (§2.2/§2.5 H2). B1–B4 intact.
- [x] P0-2-FIX3 (re-gate): exactly 7 calibrated `_SAFE_GLOBALS` tuples + calibrated round-trip RED test, "no additions" claim deleted (§3.2/§3.9); `resolved_version` stored/exposed, `get_model_version_by_alias(...).tags` + `log_model(tags=...)` named, `model_classes(estimator)` fallback, mocked-client test (§3.2/§3.9); both-flags H1 procedure + node wiring tests for both nodes (§2.2/§2.5/§2.8); post-embargo `test_size` + overshoot formula + fraction test (§4.2/§4.9); `yaml.compose` duplicate-guard mechanism (§1.2(b)/§1.7); conditional Redis delete + fake-client test (§5.2(c)/§5.9); `RestrictedPickleDataset` typo fixed (§5.2(d)). B1–B4/C1/C3/C4/C6/H2 intact.
- [x] No `src/` modified (this file only; pseudocode + signatures only).

## Worker readiness (anti-team gate)

This design is ready for the anti-team arch gate. On `APPROVED`, the planner flips D-BEST-MODEL-V2-1 / D-FITTED-REG-V2-1 / D-TEMPORAL-SPLIT-1 / D-SECRETS-PII-GDPR-1 / D-CATALOG-V2-1 to `frozen` (planner owns `decisions.md`, adds D-CATALOG-V2-1 row). P1 workers build strictly from the §-handoff pseudocode; any field/semantics change requires a new ADR + gate cycle.
