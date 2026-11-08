# Handoff P3-D — Time-series split integration (worker-backend)

## TDD trail

### RED (test first, no production change)

Wrote `forecast_ml_framework/tests/test_p3_pipeline_integration.py`
(8 tests, P3-D acceptance items) before any production edit.

Command:
`python -m pytest forecast_ml_framework/tests/test_p3_pipeline_integration.py -q`

Output (truncated, 4 failed 4 passed):

```
4 failed, 4 passed in 1.71s
FAILED ...test_end_to_end_temporal_73_20_7_real_shapes
  assert meta['n_embargoed'] == 7  ->  assert 0 == 7
FAILED ...test_leaky_presplit_raises (fixed during RED to computed
  path; clean-raw passed through, so empty meta proves frame check)
FAILED ...test_calibrate_node_wires_split_meta
  Right contains one more item: 'split_meta'
FAILED ...test_modeling_yml_exact_paths
  KeyError: 'time_col'
```

RED fails because the wiring is missing (presplit embargo
count hardcoded 0, calibrate node has 4 inputs, yaml lacks P3
keys), not because of test typos. `split_raw_data_with_meta`
73/20/7, `temporal`-without-`time_col` raise, mixed-tz
`ValueError`, prohibition raise, and no-`sort_values` passed
already (delegation via frozen `split_data` was already
honest).

### GREEN (minimal wiring)

- `modeling/pipeline.py`: import `assert_no_overlap`
  (read-only from `data_split`); propagate
  `n_embargoed = int(raw.get('n_embargoed', 0) or 0)`
  (was hardcoded `0`); call `assert_no_overlap` on the exact
  1-fold wrapper `[{'fold_index': 0, 'test_df': test_df,
  'fold_meta': meta}]` when `time_col_presplit` is not `None`
  **and** `leakage_checked_presplit` (see Assumptions); add
  `'split_meta'` as the calibrate node 5th input
  (`['best_model','x_train','y_train',
  'params:modeling_params','split_meta']`).
- `conf/base/parameters/modeling.yml`: exact P3-D schema
  (`sampling_params` extras, `calibration.allow_temporal:
  false`, new `purged_cv`/`backtest` blocks under
  `modeling_params` only; `method` stays `'stratified'`).
- `data_prep/nodes.py` and `data_prep/pipeline.py`: no edit.
  Verification showed `split_raw_data_with_meta` already
  delegates via `split_data` and returns the temporal meta
  verbatim (`n_train` 73, `n_test` 20, `n_embargoed` 7,
  `embargo` `P7D`, `leakage_checked` True), and the data_prep
  pipeline already wires it to `split_meta_raw`. No second
  `sort_values` anywhere in the three owned Python files.
- No edits to `data_split.py` / `purged_cv.py` /
  `calibration.py` (read-only imports only). `.pylintrc`
  untouched (`git diff -- .pylintrc` empty).

## Verification (GREEN)

- `python -m pytest forecast_ml_framework/tests/test_p3_pipeline_integration.py -q`
  → `8 passed in 2.77s`
- Full suite (`cd forecast_ml_framework && uv run --frozen pytest -q`)
  → `3305 passed, 4 skipped` (no regressions; P1 golden,
  layering presplit merge, modeling pipeline end-to-end
  via 4-arg node call still green).
- Pylint gate (pinned 4.1.3, root `.pylintrc`):
  `pylint --rcfile=.pylintrc --fail-under=9.0 forecast_ml_framework/src`
  → `9.68/10` (touched files alone `9.43/10`, above 9.0).
- Line-length gate: `awk 'length>80'` on the new test file
  and `modeling/pipeline.py` → `0` lines. No double quotes
  in the new test file. 2-space indent throughout.
- Grep guard (also asserted in `test_no_second_sort_in_owned_files`):
  `sort_values` in `data_prep/nodes.py`,
  `data_prep/pipeline.py`, `modeling/pipeline.py` → empty.
- Yaml check (also in `test_modeling_yml_exact_paths`):
  parsed equality for every P3-D path plus negatives
  (`purged_cv`/`backtest` absent at root and under
  `modeling_reg_params`, `allow_temporal` absent under
  `modeling_reg_params.calibration` and `calibration_params`,
  `sampling_params.method == 'stratified'`).

## Coverage (per-module, P3.md Global Constraints)

- `data_prep.nodes`: literal `--cov=forecast_ml_framework.pipelines.data_prep.nodes`
  cannot collect in this dev box (pre-existing env bug:
  `data_prep/__init__` eagerly imports `pipeline`→`nodes`→
  numpy during coverage source resolution; Python 3.14
  raises `ImportError: cannot load module more than once
  per process` for **every** test file, including
  pre-existing `test_data_prep.py`). Equivalent
  whole-package run on the gate's files
  (`test_p3_pipeline_integration`, `test_data_prep`,
  `test_splitfirst`, `68 passed`) →
  `data_prep/nodes.py 560 stmts 64 miss 89%` (≥85%).
- `data_prep.pipeline`: same env bug for the literal
  import-path form; equivalent run
  (`test_p3_pipeline_integration`, `test_data_prep`,
  `test_pipeline_registry`, `64 passed`) →
  `data_prep/pipeline.py 10 stmts 0 miss 100%` (≥85%).
- `modeling.pipeline`: literal command runs but fails the
  gate: `--cov=forecast_ml_framework.pipelines.modeling.pipeline
  --cov-fail-under=85 test_p3_pipeline_integration
  test_modeling_pipeline test_modeling_split_calibrate`
  → `32 passed` but `modeling/pipeline.py 154 stmts 57 miss
  63%` (missing: split_presplit edge paths + `build_lift_table`
  336-348, `build_promotion` 366-373, `_threshold_method`
  429-443, `select_model_threshold` 460-465,
  `build_tuning_report` 484-488 — all outside P3-D wiring
  scope; adding lift/promotion/threshold tests here would
  be YAGNI scope creep and duplicate `test_parity_pipeline`).
  With the suites that actually cover frozen lines
  (`+ test_parity_pipeline + test_interpretability`,
  `74 passed`) the file is `154 stmts 16 miss 90%`.

## Pinned behavior proved

- 73/20/7 through the real pipeline: nested
  `sampling_params.method='temporal'` (+ `time_col`,
  `test_size` 0.2, `embargo` 7 days, no top-level `method`)
  → `split_meta_raw` 73/20/7 and presplit `split_meta`
  73/20/7, `sampling_method` `presplit`,
  `raw_sampling_method`/`raw_requested_method` `time`.
- Nested temporal without `time_col` raises `ValueError`
  (not stratified fallback); nested temporal with mixed
  naive/aware raises `ValueError` (not `TypeError`).
- Leaky presplit (future rows in train, computed frames)
  raises `ValueError` via `split_presplit_data` (P0-C spoof).
- Calibrate node inputs are exactly
  `['best_model','x_train','y_train','params:modeling_params',
  'split_meta']`.
- Prohibition end-to-end on the real presplit shape:
  `calibrate_model(base, x_train, y_train, params,
  split_meta)` with sigmoid raises
  `ValueError` matching `calibrate_model prohibited with
  temporal split`; with `allow_temporal=True` proceeds and
  `meta['temporal_calibration_allowed'] is True` (bool).

## Assumptions

- `assert_no_overlap` on the presplit 1-fold wrapper runs
  only when `time_col_presplit is not None` **and**
  `leakage_checked_presplit` (max/min present). A bare
  `time_col present` gate would break the frozen layering
  test (`raw_time_col='ts'` with no max/min must stay
  `leakage_checked=False`, never raise). The spec's intent
  (time-bound wrapper) is preserved; temporal presplits have
  both.
- `n_embargoed` merges verbatim from `split_meta_raw`
  (`raw.get('n_embargoed', 0)`); without raw meta it stays
  `0`, preserving frozen `test_split_presplit_data_clean_annnotates`.
- Prohibition `x_train`/`y_train` come from a second
  `split_presplit_data` call adding `feature_cols=['f1']` so
  sklearn sees numeric features (`time` is datetime and
  cannot be fitted). Counts/meta are identical to the
  `73/20/7` run; `data_prep` ignores `feature_cols` by
  design, only modeling slices by it.
- `'time'`/`'ordered'` aliases without `time_col` still take
  the frozen `_ordered_split` fallback in `split_data`
  (read-only `data_split.py`); only `'temporal'` raises,
  which is what P3-D acceptance pins.
- Test file name is `test_p3_pipeline_integration.py`
  (P3.md File footprint matrix + Acceptance, normative),
  not the shortened `test_p3_integration.py` in the
  dispatch shorthand.

## Blockers

BLOCKED (coverage-gate arithmetic, needs Lead decision):
literal `modeling.pipeline` per-module gate is `63%` with
P3.md's named suites because those suites never exercise
`build_lift_table`/`build_promotion`/threshold/tuning
functions (owned by parity suites). Options: (1) add
`test_parity_pipeline.py` + `test_interpretability.py` to
the gate (reaches `90%`), (2) accept whole-suite `85%`
total, or (3) authorize out-of-scope tests in the P3-D
file. Literal `data_prep.*` gates also cannot run
in this dev box (numpy double-load, pre-existing); CI
env may differ. Awaiting `anti-team-arch` gate + Lead
ruling on the coverage gate.
