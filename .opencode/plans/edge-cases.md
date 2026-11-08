# Edge-Case Taxonomy (code-real) — corrected

> **Owner:** `docs-engineer` · **Status:** corrected draft for anti-team gate
> **Scope:** test backlog input per phase (P0 contracts → P11 gate). Docs-only; no `src/`, README, `contracts.md`, or `decisions.md` edits.
> **Method (binding):** every handling claim below was grepped/read to a `file:line` under `forecast_ml_framework/src/forecast_ml_framework/` (written as `path:line` relative to that root). **If code contradicts a prior claim, code wins and the doc says so.** Absence claims name the closest code plus the grep that found no handler.
> **How to use:** each item is one backlog row: edge case → true handling → severity → observable test. Phase tags reuse `decisions.md` vocabulary (P0 contracts, P1 contract implementation, P2 platform/IaC, P3 time-split, P9 MLOps, P10 serving, P11 release gate).

---

## Part 0 — Corrections to the rejected F1–F13 claims (what was false/missing, and what is true)

### F1 — Empty input raises; <10 rows scores in-sample; ordered split clamps to min 1 test row
- **Edge case:** empty frame; train frame with <10 rows; 1-row ordered split.
- **Evidence:** `pipelines/modeling/data_split.py:151-152` (`len(df)==0` → `ValueError('split_data requires a non-empty dataframe')`); `pipelines/modeling/pipeline.py:181-188` (presplit empty → `ValueError`); `pipelines/modeling/adapters.py:954-956` (`len(y)<10` → return `None`); `pipelines/modeling/adapters.py:995-998` (`split is None` → fit then `_score_on(x_train)` in-sample); `pipelines/modeling/data_split.py:27-30` (`n_test=max(1, round(n*test_size))`, capped to `n-1`).
- **Current handling (true):** Empty raises, does not return empty artifacts. With <10 rows there is deliberately **no** held-out selection slice: selection scores on the full training set (in-sample, optimistic). Ordered/time split never produces an empty test side — at least 1 test row is forced.
- **Severity:** High (small-data in-sample selection silently inflates the selection metric).
- **Proposed test:** (a) `split_data` on 0-row frame raises `ValueError`; (b) `train_models` on 9 rows reports per-model metrics equal to in-sample scores (recompute on `x_train` and assert equality); (c) 2-row ordered split yields `n_test==1`, `n_train==1`.

### F2 — Serving never checks key order; reordered keys succeed
- **Edge case:** client sends feature dict keys in different order than training.
- **Evidence:** `serving/app.py:487-494` (required order from `feature_names_in_` when present; fallback `serving/app.py:488-489` `required = keys` — the first record's keys order — when `feature_names_in_` is absent); `serving/app.py:498-503` (unknown-key set check only); `serving/app.py:504-526` (rows built iterating `required`, not input order).
- **Current handling (true):** No order validation. **Only when the model bears `feature_names_in_`** are rows assembled in model-fitted order, so a reordered-but-complete dict produces the identical row list. When `feature_names_in_` is absent, assembly follows the first record's keys order (`serving/app.py:488-489`), not a fitted order — no fitted-order guarantee exists in that case.
- **Severity:** Low (safe behavior for `feature_names_in_`-bearing models; test guards against a future regression that restores positional order).
- **Proposed test:** Scoped to a `feature_names_in_`-bearing model only: POST two records with same keys in reversed order → identical predictions byte-for-byte. Do not assert the same fitted-order guarantee for a `feature_names_in_`-less model; there `required` follows the first record's keys order (`serving/app.py:488-489`).

### F3 — OHE is a hard-fail over `max_cols`, not a cap/truncation; binning `300` is a min-count keep threshold
- **Edge case:** categorical expansion exceeds `max_cols`; high-cardinality column with counts around 300.
- **Evidence:** `pipelines/data_prep/nodes.py:442-457` (pre-expansion raise), `pipelines/data_prep/nodes.py:487-491` (post-expansion raise); `pipelines/scoring/nodes.py:622-643` (same hard-fail with per-column distinct check); `pipelines/data_prep/nodes.py:408-418` (`val_count` default 300, `keep = counts[counts>=val_count]`); fitted twin `pipelines/data_prep/fitted.py:567-578`.
- **Current handling (true):** Expansion over the cap **raises `ValueError`**; nothing is truncated or silently bucketed. `300` is the minimum count for a category to be kept as itself; rarer categories become `'Other'`. It is not a maximum-cardinality cap and not a row count.
- **Severity:** Medium (fail-loud is correct, but operators expecting truncation will hit hard failures).
- **Proposed test:** (a) OHE with expansion > cap raises `ValueError` naming the column; (b) category with count 299 → `'Other'`, count 300 → kept, count 301 → kept.

### F4 — Variance/correlation/L1 (and missingness/kbest/tree) selection are all conditional; none runs by default
- **Edge case:** caller expects variance/correlation/L1 pruning without enabling it.
- **Evidence:** `pipelines/data_prep/nodes.py:1003-1010` (variance only `if vt is not None`); `pipelines/data_prep/nodes.py:1017-1029` (correlation only `if corr_cfg.get('enabled')`); `pipelines/data_prep/nodes.py:1053-1070` (L1 only `if l1.get('enabled')`); kbest/tree same pattern `pipelines/data_prep/nodes.py:1030-1052,1071-1086`.
- **Current handling (true):** Without the enabling keys, `select_features` keeps all input feature candidates (subject only to the core merge at `pipelines/data_prep/nodes.py:1087-1115`). No filter is a default.
- **Severity:** Medium (silent no-op config is a common misread).
- **Proposed test:** call `select_features` with no selection keys on a frame containing a zero-variance and two perfectly correlated columns → selected list equals input list, report has no `variance`/`correlation_drop`/`l1` entries.

### F5 — Fitted preprocessing is train-only and stateful (the "fit on everything" claim is false)
- **Edge case:** leakage via fitting preprocessing on train+test, or assuming scoring re-fits.
- **Evidence:** `pipelines/data_prep/fitted.py:16-24` ("fitted on the train split only"); `pipelines/data_prep/fitted.py:898-949` (`fit_preprocess` fits state, returns JSON-serializable registry); `pipelines/data_prep/nodes.py:805-833` (wrapper documents same-fit train frame + persisted registry); `pipelines/data_prep/nodes.py:836-855` + `pipelines/data_prep/fitted.py:952-975` (`apply_preprocess` replays stored state verbatim); `pipelines/data_prep/pipeline.py:88-93` (test applies stored `fitted_registry`).
- **Current handling (true):** Seven steps (Imputations, Normalizations, QuantileDiscretizer, HighLabelBinning, CategoricalEncoding, TargetEncoding, CapOutliers — `pipelines/data_prep/fitted.py:874-892`) fit once on train; test/score only replay. TargetEncoding additionally splits OOF-for-train vs full-train-mapping-for-apply (`pipelines/data_prep/fitted.py:930-936`). Scoring never re-fits.
- **Severity:** Critical if violated (leakage); the code is correct — the test freezes the contract.
- **Proposed test:** fit registry on train, then apply to a test frame with shifted column means → output uses train means exactly (assert equality with manual replay), registry bytes unchanged.

### F6 — All-NaN fit raises `ValueError` in the 4 numeric steps only; the 3 categorical steps silently record an empty state
- **Edge case:** a column that is entirely NaN at fit time (numeric and categorical cases differ).
- **Evidence (raises — 4 numeric steps only):** `pipelines/data_prep/fitted.py:366-379` (Imputations mean/median/mode), `pipelines/data_prep/fitted.py:435-437` (Normalizations), `pipelines/data_prep/fitted.py:511-513` (QuantileDiscretizer), `pipelines/data_prep/fitted.py:829-846` (CapOutliers incl. NaN-bound re-raise).
- **Evidence (silent — 3 categorical steps):** HighLabelBinning `pipelines/data_prep/fitted.py:575-577` (`keep = [k for k in counts[counts >= val_count].index]` → `keep=[]` when no category reaches `val_count`, e.g. all-NaN); CategoricalEncoding `pipelines/data_prep/fitted.py:599` (`cats = list(pd.factorize(df[col])[1])` → `categories=[]` / `cats=[nan]` for all-NaN, with no `_assert_finite_state` / finite-state check in `_fit_encode_cat`); TargetEncoding `pipelines/data_prep/fitted.py:716-735` (all-NaN feature column yields an empty `mapping=[]`; only an all-NaN **target** raises, `pipelines/data_prep/fitted.py:709-712`). Related silent-skip: Imputations `strategy='none'` (default) `pipelines/data_prep/fitted.py:357` silently skips — `fill` stays `None` and no state is recorded, with no `ValueError`.
- **Current handling (true):** The 4 numeric steps (Imputations mean/median/mode, Normalizations, QuantileDiscretizer, CapOutliers) fail loud with `ValueError('... cannot fit all-NaN column ...')` — no empty-state, no zero-fill, no skip path. The 3 categorical steps (HighLabelBinning, CategoricalEncoding, TargetEncoding) instead silently record an empty state (`keep=[]`, `categories=[]` / `cats=[nan]`, `mapping=[]` respectively) and fit succeeds. Note: columns absent from the frame are skipped (`if col not in df.columns: continue`, e.g. `pipelines/data_prep/fitted.py:362-363`) — absent ≠ all-NaN. **Do not "fix" the categorical steps to raise:** that would break the shipped CategoricalEncoding config (`conf/base/parameters/preprocessing.yml:73-85`, `skip: false`, `ohe_flag: true`). This doc recommends no code change; it records current behavior only.
- **Severity:** High for the 4 numeric steps (whole fit aborts; upstream must drop the column before fit). Medium for the 3 categorical steps (silent empty state can mask a dead column, but that is the shipped contract to document, not to change here).
- **Proposed test:** (a) for each of the **4 numeric steps only** (Imputations mean/median/mode, Normalizations, QuantileDiscretizer, CapOutliers), fit a frame with one all-NaN column → `ValueError` matching `all-NaN`; (b) for the **3 categorical steps, assert the current silent behavior, not `ValueError`** — HighLabelBinning fits with `keep=[]`, CategoricalEncoding fits with `categories=[]` (or `cats=[nan]`), TargetEncoding fits with `mapping=[]`; (c) Imputations with `strategy='none'` on an all-NaN column fits with no state recorded (silent skip), not `ValueError`.

### F7 — H2O `predict_proba` raises `NotImplementedError` and no upstream path catches it → run crashes
- **Edge case:** classification run where the best model is `h2o_automl` and proba is required.
- **Evidence:** `pipelines/modeling/adapters.py:766-780` (`raise NotImplementedError` when the fitted model lacks callable `predict_proba`); `pipelines/modeling/adapters.py:974-983` (`_score_on` calls `adapter.predict_proba(x)` unguarded, no `try`); `pipelines/modeling/pipeline.py:244-248` (`_predict_proba` calls `model.predict_proba(x_test)` unguarded); `pipelines/scoring/nodes.py:756-759` (proba path calls `predict_proba` with only `hasattr` guard, no `NotImplementedError` catch).
- **Current handling (true):** Crash with `NotImplementedError`, not a graceful fallback to labels or a clean `ValueError`. There is no adapter-level "proba unavailable" contract.
- **Severity:** High (zoo selection can produce a best model that then crashes evaluation/scoring).
- **Proposed test:** classification with `models_to_use=['h2o_automl']` and a proba-less fitted stub → assert `NotImplementedError` propagates from `train_models`/pipeline (observable crash, documented as such until a contract lands).

### F8 — Multiclass proba has three different behaviors (warned vs silent `[:,1]`)
- **Edge case:** `predict_proba` returns `(n,k>2)`.
- **Evidence (warned):** `pipelines/modeling/threshold.py:123-128` (warns, returns 0.5); `pipelines/scoring/nodes.py:479-483` (warns, falls back to row-wise max).
- **Evidence (silent):** `pipelines/modeling/pipeline.py:280-282` (`proba[:,1]`) and `pipelines/modeling/pipeline.py:396-398` (same); `pipelines/modeling/evaluate.py:89-91` (`arr[:,1]`).
- **Current handling (true):** `select_threshold` and scoring warn; the pipeline/evaluate `[:,1]` paths silently trust column 1 as the positive class, so multiclass input yields a binary decision without any warning. Column 1's meaning (which class) is nowhere recorded.
- **Severity:** High.
- **Proposed test:** feed `(n,3)` proba + binary `y_test` into `build_lift_table`/`select_model_threshold`/`evaluate_model` → assert result equals the `[:,1]`-only computation **and** record that no `UserWarning` is emitted (documents the silent path as the behavior to fix).

### F9 — `select_threshold` 0.5 fallbacks: three silent paths (no warning)
- **Edge case:** single-class `y_true`; non-finite proba; out-of-range proba.
- **Evidence (silent):** `pipelines/modeling/threshold.py:147-148` (single-class → `return 0.5`, no warn); `pipelines/modeling/threshold.py:154-155` (non-finite proba → `return 0.5`, no warn); `pipelines/modeling/threshold.py:156-157` (proba outside [0,1] → `return 0.5`, no warn).
- **Evidence (warned, for contrast):** non-numeric proba `threshold.py:97-101`; bad `y_true` shape `threshold.py:111-115`; multiclass proba `threshold.py:123-128`; string `y_true` `threshold.py:134-140`; non-finite `y_true` `threshold.py:141-145`; multiclass `y_true` `threshold.py:149-153`; unknown/failed method `threshold.py:162-166,219-222`.
- **Current handling (true):** All three silent paths return 0.5 indistinguishable from a legitimately selected 0.5. Callers cannot tell "selected" from "gave up".
- **Severity:** High.
- **Proposed test:** each silent input returns 0.5 with **zero** warnings (assert via `warnings.catch_warnings` record); contrast case (multiclass) emits exactly one warning.

### F10 — Unknown `objective` passes through; unknown `loss_function` raises (asymmetric by design)
- **Edge case:** typo'd `objective='binay:logistic'` vs typo'd `loss_function='loglos'`.
- **Evidence:** `pipelines/modeling/adapters.py:317-324` (objective: `try get_loss … except ValueError: pass` — "unrecognised native objective passes through unchanged"); `pipelines/modeling/adapters.py:325-328` (`loss_function` → `get_loss`, unguarded); `pipelines/modeling/adapters.py:286-293` (unknown loss → `ValueError` listing valid losses); `pipelines/modeling/adapters.py:264-271` (non-string/empty → `ValueError`).
- **Current handling (true):** `objective` is back-compat-tolerant (unknown natives reach the estimator, which may then fail or may accept a legit native not in the registry). `loss_function` is strict. Same evidence: a dotted `objective` (contains `.`) skips normalization entirely (`adapters.py:318`).
- **Severity:** Medium (typo in `objective` fails late inside the estimator, if at all).
- **Proposed test:** `objective='not_a_loss'` → estimator receives it verbatim (assert via stub adapter capturing kwargs); `loss_function='not_a_loss'` → `ValueError` before construction.

### F11 — Scrubbed-detail logging is a server-side-only surface; clients get generic messages
- **Edge case:** validation failure or model exception carrying secrets/paths in its detail.
- **Evidence:** `serving/app.py:1026-1027` (client 422 detail is literally `'validation failed'`); `serving/app.py:290-310` (`_log_validation_error` scrubs via `scrub_text` server-side, strips control chars); `serving/app.py:177-188` (`_log_exception` logs event code + type only — "The exception message is never logged"); `serving/app.py:33-35` (docstring contract); access log carries only HMAC key-id/outcome/latency/records `serving/app.py:241-260,270-285`.
- **Current handling (true):** Two distinct surfaces: (a) client — generic 422; (b) operator log — scrubbed field names. Exception messages never reach either log (code+type only). This is intentional and must be preserved: "more detail to client" is a security regression, not an improvement.
- **Severity:** High if broken in either direction (leak vs blind ops).
- **Proposed test:** POST with unknown feature containing a fake secret string → response body is exactly `{'detail':'validation failed'}` **and** server log contains the scrubbed field name with the secret redacted **and** raw key/payload absent.

### F12 — Training has partial schema enforcement; serving parity is a gap, not "none"
- **Edge case:** inf/NaN/typed inputs entering training vs serving via different gates.
- **Evidence (training):** inf rejection `pipelines/data_prep/fitted.py:168-181` (`_reject_inf_numeric`, called from `_fit_impute:359`, `_fit_normalize:430`, `_fit_discretize:505`, `_fit_target_encode:707`, `_fit_cap:821`); numeric-state check `_assert_finite_state` `pipelines/data_prep/fitted.py:184-201`; Spark type gate `pipelines/data_prep/nodes.py:784-787` + `pipelines/data_prep/nodes.py:825-828` (`TypeError` when Spark input but `convert_to_pandas=False`).
- **Evidence (serving/scoring):** per-record finite-number check `serving/app.py:510-523`; post-OHE non-finite check `pipelines/scoring/nodes.py:599-603`.
- **Current handling (true):** Enforcement exists in both paths but is **not shared**: training distinguishes inf (reject) from NaN (imputation steps own NaN) and validates fitted *state*; serving validates wire values only and has no fitted-state analogue. Training's gate covers only the 7 fitted steps' listed numeric columns — e.g. inf in a non-step column is not training-rejected. The gap is parity/ownership, not absence.
- **Severity:** Medium.
- **Proposed test:** (a) inf in a fitted numeric column → training `ValueError`; (b) inf in a non-listed column → training does **not** raise (documents partial coverage); (c) same inf value via serving → 422 `validation failed`. All three behaviors asserted, not assumed identical.

### F13 — Isotonic calibration has no degenerate-data guard (not a clean break)
- **Edge case:** isotonic requested with a too-small / single-class / constant-proba training set.
- **Evidence:** cv validation `pipelines/modeling/calibration.py:59-78` (only `2 <= cv <= min class count`); `pipelines/modeling/calibration.py:322-325` (`CalibratedClassifierCV(base, method=method, cv=cv).fit(...)` — no isotonic-specific minimum, no `try/except`, no fallback); contrast betacal missing-library fallback `pipelines/modeling/calibration.py:305-313` (warns → sigmoid).
- **Current handling (true):** Isotonic gets no framework-level degenerate guard or fallback. Whatever sklearn raises (or mis-fits) propagates raw from `calibrated.fit` — there is no framework `ValueError` naming isotonic degeneracy, and no warning+fallback as betacal has.
- **Severity:** Medium.
- **Proposed test:** isotonic + `cv=2` on a minimal 2-per-class set → assert the propagating exception type/message (whatever sklearn emits) is recorded verbatim in the test; any change to framework error surfacing must update the test.

---

## Part 1 — Missing cases M1–M16 (adversarial / robustness / operations)

### M1 — Adversarial serving inputs (in-distribution-looking attacks)
- **Evidence:** `serving/app.py:468-526` validates only key-set, type, finiteness — no range/distribution/anomaly check; grep over `serving/` finds no adversarial/anomaly handler.
- **Handling:** any finite floats within key-set are scored.
- **Severity:** High. **Test:** extreme-but-finite values (e.g. 1e308) → 200 with predictions (documents no adversarial gate); decide per phase whether ranges belong in schema (P10).

### M2 — Training-data poisoning
- **Evidence:** narrow gates only: inf rejection `pipelines/data_prep/fitted.py:168-181`; handle allowlist `pipelines/modeling/adapters.py:145-186` (prevents RCE, not data attacks); grep finds no provenance/poisoning defense.
- **Handling:** poisoned-but-finite rows are fitted normally.
- **Severity:** High. **Test:** flip 5% of labels in train → fit succeeds and selection metric moves (documents no poisoning resistance; motivates provenance policy P9).

### M3 — Model extraction (600/min, full-precision, shared key)
- **Evidence:** rate `serving/app.py:84` (600/min) + token bucket `serving/app.py:357-385`; full-precision list output `serving/app.py:529-548`; single shared key, single-process limits `serving/app.py:41-46`.
- **Handling:** an authenticated caller can query 600/min and get exact float predictions; one shared key means extraction attribution is all-or-nothing.
- **Severity:** High. **Test:** 30 queries at full precision → assert predictions are exact floats (no rounding anywhere in `_predict`), extraction surface documented for P10 threat model.

### M4 — Cost/DoS via expensive models (no timeout/semaphore; zoo trains all incl. H2O, no budget)
- **Evidence:** no timeout/semaphore in predict path `serving/app.py:529-548` (grep: no `timeout`/`Semaphore` in `serving/`); zoo trains every name `pipelines/modeling/adapters.py:1041-1046`; H2O trains full AutoML `pipelines/modeling/adapters.py:738-759` (no budget kwarg beyond loss); only bound is count `pipelines/modeling/adapters.py:16-17,904-909` (`MAX_MODELS=10`, counts models, not compute).
- **Handling:** 10 expensive models (incl. H2O AutoML) fit sequentially with no wall-clock/compute budget.
- **Severity:** High. **Test:** stub adapter sleeping 60s × `models_to_use` of 3 → assert no timeout/abort (documents unbounded cost); P2/P9 decision: budgets vs fail-loud.

### M5 — Concurrency races on unlocked audit/health surfaces
- **Evidence:** audit append unlocked `governance/audit.py:114-115`; health cache plain-dict read/write `serving/app.py:920-941`; contrast locked surfaces: token bucket `serving/app.py:375`, loaders `serving/model_loader.py:308,438`.
- **Handling:** concurrent `/predict` audit appends and health-cache refreshes race (GIL makes list-append benign in CPython, but ordering/loss under eviction is unspecified).
- **Severity:** Medium. **Test:** 1000 concurrent `_audit_predict` calls → assert no exception and `len <= MAX_AUDIT_ENTRIES` (documents current guarantee level: no crash, no ordering promise).

### M6 — Fitted-registry vs model version skew (undetected)
- **Evidence:** registry carries own version `pipelines/data_prep/fitted.py:46-49,937-949`; apply checks only registry version `pipelines/data_prep/fitted.py:259-284`; model identity is separate (`pipelines/scoring/pipeline.py:185-227` applies registry without naming the model it was fitted with).
- **Handling:** a registry fitted for model A silently scores model B as long as columns align.
- **Severity:** High. **Test:** pair registry-from-A with model-B (same schema) → scoring succeeds (documents no binding); P1 contract decision: bind registry↔model version.

### M7 — Registry alias race (resolved once, cached forever)
- **Evidence:** `serving/model_loader.py:440-454` (`load` caches `self._model` on first call, double-checked lock); alias resolved once `serving/model_loader.py:543-566`; no invalidation/TTL (grep: no `invalidate`/`reload` on loader). Contrast health TTL `serving/app.py:84-87` (health only).
- **Handling:** repointing `production` alias in the registry has no effect on a running process.
- **Severity:** High. **Test:** load via stub registry v1, repoint alias to v2, call `load` again → still v1 (documents staleness); P10 decision: reload/TTL vs restart-required.

### M8 — Concept drift undetectable in the small-window case (NaN → no alert)
- **Evidence:** small/empty → NaN stats + warning `pipelines/monitoring/nodes.py:1-10,245-260,427-433`; "NaN values never alert" `pipelines/monitoring/nodes.py:737-741,786-789`.
- **Handling:** below `min_rows` (default 10) or on non-finite inputs, drift yields NaN, and NaN yields **zero** alerts. A drifting stream with tiny windows is indistinguishable from no drift.
- **Severity:** Critical. **Test:** 5-row window with maximal shift vs reference → `evaluate_alerts` returns `[]` (documents the silent hole; motivates "insufficient data" alert, P9).

### M9 — Feedback loops (model outputs re-enter training/serving features)
- **Evidence:** grep for `feedback` across `src/` finds no handler; scoring writes predictions `pipelines/scoring/nodes.py:1093-1138` with no loop/lineage guard; retraining consumes monitoring report `pipelines/retraining/pipeline.py:10-20` with no provenance check that report features aren't model outputs.
- **Handling:** no detection or break of output→input loops.
- **Severity:** Medium. **Test:** document as missing (grep-verified); test when a guard lands: predictions-derived feature column flagged in EDA leakage screen.

### M10 — Cold start (entity never seen)
- **Evidence:** `feature_store/online.py:288-304` (`chosen is None → continue`; result contains only found entities; docstring `feature_store/online.py:252-254` "Future-only entities are omitted").
- **Handling:** unknown entities are silently absent from the result dict (no default feature vector, no error). Downstream that zips by position gets misalignment; by key it works.
- **Severity:** Medium. **Test:** request `[known, unknown]` → result keys are exactly `{known}` (assert the omission, so consumers never assume presence).

### M11 — Sparse categories (unseen/rare values at score time)
- **Evidence:** binning unseen → `'Other'` `pipelines/data_prep/fitted.py:581-587`; OHE unseen → all-zero dummies `pipelines/data_prep/fitted.py:671-676`; factorize unseen → `-1` `pipelines/data_prep/fitted.py:682-683`; target-encode unseen → global mean `pipelines/data_prep/fitted.py:767-800`.
- **Handling:** three different unseen-value semantics across steps (Other vs zero-dummies vs −1) — all silent.
- **Severity:** Medium. **Test:** score frame with one unseen category per step type → assert the three distinct encodings byte-exactly.

### M12 — Categorical appearance-order: (a) fitted factorize is stable, (b) scoring OHE dummy order is appearance-order unless `feature_names_in_` is present
- **Evidence (a — fitted factorize, stable):** fitted categories recorded in factorize order `pipelines/data_prep/fitted.py:599`, replayed verbatim at apply via the stored state `pipelines/data_prep/fitted.py:669`; code mapping is built from that fitted state `pipelines/data_prep/fitted.py:682-683` (`mapping = {c: i for i, c in enumerate(cats)}`), so fitted codes are **stable** (fitted-order mapping, no issue).
- **Evidence (b — scoring OHE):** scoring dummy columns come from `pd.get_dummies` (appearance order) `pipelines/scoring/nodes.py:666` in `_ohe_encode` (`pipelines/scoring/nodes.py:607`); reindexing to fitted order happens only when `feature_names_in_` is present — `_align_to_fitted` `pipelines/scoring/nodes.py:680-704` returns `x` unchanged when `feature_names_in_` is absent (`pipelines/scoring/nodes.py:696-704`).
- **Handling:** (a) No issue — fitted factorize codes are stable by construction. (b) Dummy column order is scoring-appearance order via `get_dummies`, reindexed to fitted order **only if** `feature_names_in_` is present; a model without `feature_names_in_` keeps scoring-appearance order, so the same category can occupy a different dummy column at score time if row order differs from training.
- **Severity:** None for (a). High for (b) when `feature_names_in_` absent. **Test:** do **not** assert on fitted codes (stable, see (a)). Assert dummy order / prediction via `_ohe_encode`: encode frames with rows ordered [b,a] vs [a,b] through `_ohe_encode` (`pipelines/scoring/nodes.py:607,666`) with a `feature_names_in_`-less stub → dummy column order / downstream prediction differs (documents order-dependence in (b)); contrast: with `feature_names_in_` present, `_align_to_fitted` (`pipelines/scoring/nodes.py:680-704`) reindexes dummy columns to fitted order.

### M13 — Target skew / zero-inflation / censoring
- **Evidence:** imbalance validates labels only `pipelines/modeling/imbalance.py:44-80`; grep finds no zero-inflation/censoring handling; target encoding uses plain group means `pipelines/data_prep/fitted.py:700-738`; imputation fills NaN by mean/median/mode `pipelines/data_prep/fitted.py:354-397`.
- **Handling:** heavy skew, zero-inflated, and censored targets all flow through mean-based encoding/imputation unadjusted.
- **Severity:** Medium. **Test:** 95%-zero target → global mean ≈ 0 and encodings collapse near 0 (observable), assert per-model metric degradation is surfaced by evaluation (documents no adjustment).

### M14 — Multivariate / correlation / label shift (monitoring is univariate only)
- **Evidence:** drift loop is per-column `pipelines/monitoring/nodes.py:389-398`; prediction drift is a single series `pipelines/monitoring/nodes.py:486-519`; grep finds no multivariate/joint drift.
- **Handling:** joint distribution shifts that preserve every marginal (and label shift without prediction-distribution change) are invisible to PSI/KS.
- **Severity:** High. **Test:** permute two features jointly so marginals are identical but correlation flips → drift report shows no alert (documents the blind spot).

### M15 — Calibration under prior shift (corrects only with explicit rates)
- **Evidence:** odds-adjustment exists `pipelines/modeling/calibration.py:84-178`; `calibrate_model` fits on train CV `pipelines/modeling/calibration.py:237-325` with no shift measurement; scoring correction applies only when both rates present `pipelines/scoring/nodes.py:788-791`.
- **Handling:** under shift without `real_rate`+`sample_rate`, calibration silently serves train-prior probabilities.
- **Severity:** High. **Test:** shift prevalence 0.5→0.1 without rates → scored proba unchanged (assert identity, documents no auto-correction).

### M16 — Learning-curve chart with no gate (chart exists, nothing consumes it)
- **Evidence:** chart function `pipelines/modeling/charts.py:957-976` (returns `None` if <2 points); included via `_safe` `pipelines/modeling/charts.py:1128`; promotion reads only `selected_ntile` vs `old_score` `pipelines/modeling/pipeline.py:287-316`.
- **Handling:** a diverging train/val gap never blocks promotion; the chart is decorative.
- **Severity:** Medium. **Test:** divergent synthetic curves → promotion decision unchanged (assert gate ignores the chart; documents the missing gate for P9).

---

## Part 2 — Re-ranked Top 5 (with rationale; prior misranking corrected)

| Rank | Edge case | Why this rank (evidence) |
|---|---|---|
| **#1** | **Silent drift absence + concept drift** (M8, M14) | The safety net itself fails silently: NaN never alerts (`pipelines/monitoring/nodes.py:737-741`), small windows return NaN (`:245-260`), and joint/label shift is invisible (`:389-398`). A broken model in production produces *zero* signal. Nothing else matters if you cannot see failure. |
| **#2** | **Temporal leakage on the default path** | Default split is stratified+shuffled (`pipelines/modeling/data_split.py:49-62`); time split requires explicit `method='time'` or `time_col`+`shuffle=False` (`:54-57`). Every unspecified config is a leakage config. |
| **#3** | **Entity leakage (no group split exists)** | `split_data` has no group/entity parameter at all (options in `data_split.py:35-62`, signature `data_split.py:120-147`); only a post-hoc batch warning exists for point-in-time (`feature_store/store.py:417-424`). Same entity in train+test inflates every metric. |
| **#4** | **All-zoo NaN → silent first-model fallback** | `_pick_best` skips NaN/None (`pipelines/modeling/adapters.py:894-901`), returns `None`; caller falls back to `names[0]` (`adapters.py:1047`). If every model fails to produce the selection metric, an arbitrary (first-listed) model ships with no error. |
| **#5** | **Version skew / alias race** (M6+M7) | Registry↔model unbound (`pipelines/data_prep/fitted.py:259-284` checks registry only); alias cached forever (`serving/model_loader.py:440-454`). Stale/mismatched artifacts serve indefinitely with clean health. |

**Audit eviction is Medium, not Top-5 (corrected):** the bounded audit list drops oldest entries beyond 1000 (`serving/app.py:89-90,241-267`). Losing audit history under burst load is an observability cost, not a wrong-prediction or silent-failure risk — unlike #1–#5 it cannot corrupt scores, leak data, or ship the wrong model. Revisit only if audit eviction can hide a security event (see F11/M3 interplay).

---

## Part 3 — P0 contract-breakage surface (must freeze before P1/P2)

### P0-A — `best_model` as dict breaks catalog, calibration, serving schema, and model_loader
- **Evidence (current contract):** `train_models` returns a fitted adapter object (`pipelines/modeling/adapters.py:1047-1048`); pipeline wires `best_model → calibrate_model → _predict/_predict_proba` (`pipelines/modeling/pipeline.py:442-461`); calibration expects an adapter/estimator (`pipelines/modeling/calibration.py:237-296`, `getattr(model,'model',model)` at `:296`); serving expects `predict`/`predict_proba` objects (`serving/app.py:529-548`); scoring resolves loaders to `predict`-bearing models (`pipelines/scoring/nodes.py:351-397`).
- **Breakage:** a dict `{model,label_set,registry_ref,onnx_sidecar}` is not callable, has no `predict`, and cannot be passed to `calibrate_model` or the ONNX/serving schema without a new unwrapping layer. The Kedro catalog node `train_models_node` outputs `best_model` as a single dataset (`pipelines/modeling/pipeline.py:442-447`) — splitting into four outputs changes every downstream input.
- **Severity:** Blocking (P1 entry gate). **Test:** propose-then-refuse: a dict-shaped `best_model` must fail fast at the first consumer with a named error until the unwrapping contract is frozen.

### P0-B — `fitted_registry` v2 hard-rejects (there is no dual-read)
- **Evidence:** schema is v1-only (`pipelines/data_prep/fitted.py:46-49`); any present version/schema mismatch raises (`pipelines/data_prep/fitted.py:259-284`); non-dict registry raises (`:290-291`); unknown steps raise at apply (`:968-973`); legacy typed-registry guard raises instead of mapping (`:325-351`).
- **Breakage:** a v2 registry fed to current `apply_preprocess` raises `ValueError` — no fallback, no shadow read, no migration hook. Existing v1 consumers gain nothing from v2 until a dual-read branch lands.
- **Severity:** Blocking (P1 entry gate). **Test:** v2-shaped dict → current code raises `ValueError` matching `fitted_registry .* mismatch` (observable refusal; dual-read must invert this test).

### P0-C — Presplit bypasses the temporal split entirely
- **Evidence:** modeling wires `split_presplit_data` with `sampling_method='presplit'` and no shuffle (`pipelines/modeling/pipeline.py:133-161,210-236,436-441`); it only validates non-empty + target present (`:181-195`) and slices the two frames — it never inspects time ordering. The actual split happens in data_prep via `split_data` (`pipelines/data_prep/nodes.py:858-913`), whose default is stratified+shuffled (F1/Top-5 #2).
- **Breakage:** once data_prep has split (default: non-temporal), modeling's presplit node can never recover temporal correctness — it reports `raw_sampling_method`/`raw_time_col` in meta (`pipelines/modeling/pipeline.py:218-234`) but does not enforce them. Any contract that promises "temporal split" is void if data_prep's split isn't temporal.
- **Severity:** Blocking (P3 design). **Test:** feed presplit frames split randomly (future rows in train) with `method='time'` expected → assert modeling **succeeds** and meta shows `sampling_method='presplit'` (documents the bypass; temporal enforcement must be added at the data_prep split or the presplit gate).

---

## Verification appendix (greps run)

- `grep -rn "NotImplementedError|H2O" src/` → adapters H2O block; no catch (F7).
- `grep -rn "isotonic" src/` → no degenerate guard; only `calibrate_model` passthrough (F13).
- `grep -rn "feedback|cold|zero.inflate|skew|censor" src/` → no feedback/cold/zero-inflation/censoring handlers (M9/M13; only residual-skew metric + EDA joint counts).
- `grep -rn "best_model|fitted_registry|presplit" src/` → P0-A/B/C consumer lists above.
- `grep -rn "broadcast|poison|adversar"` → no handlers (M1/M2).
- `grep -rn "timeout|Semaphore|Lock" serving/` → locks only on bucket+loaders; no timeout/semaphore (M4/M5).

*Unverified / code-wins note: where sklearn-internal error text (F13) and exact audit-eviction ordering under threads (M5) could not be determined from this repo alone, the doc records the observable surface (propagating exception; no-crash + bound) instead of inventing a guarantee.*
