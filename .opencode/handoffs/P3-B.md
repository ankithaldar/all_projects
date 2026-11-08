# Handoff P3-B — Backtesting harness (worker-backend)

## TDD trail

### RED (test first, no production code)

Wrote `forecast_ml_framework/tests/test_p3_backtest.py` (26 tests)
before any `backtest.py` existed.

Command: `python -m pytest forecast_ml_framework/tests/test_p3_backtest.py -q`

Output (truncated, 26 failed):

```
FFFFFFFFFFFFFFFFFFFFFFFFFF                                               [100%]
...
___ test_exact_scores_regression_train_mean ___
    from forecast_ml_framework.pipelines.modeling.backtest import (...)
E   ModuleNotFoundError: No module named 'forecast_ml_framework.pipelines.modeling.backtest'
...
26 failed in 0.64s
```

RED fails because the feature is missing (module absent), not
because of a typo — every failure is the same `ModuleNotFoundError`
on the new `backtest` module.

### GREEN (minimal implementation)

Implemented
`forecast_ml_framework/src/forecast_ml_framework/pipelines/modeling/backtest.py`
(`run_backtest` + `BacktestResult` + `default_metric`) importing
only `purged_cv_folds` and `fold_leakage_check` from `purged_cv`
(read-only). No edits to `data_split.py` / `purged_cv.py`.

No REFACTOR change to behavior: only test-helper cleanup
(unused-arg prefixes, one docstring typo, `assert not calls`).

## Verification (GREEN)

- `python -m pytest forecast_ml_framework/tests/test_p3_backtest.py -q`
  → `26 passed in 1.51s`
- `python -m pytest --cov=forecast_ml_framework.pipelines.modeling.backtest --cov-fail-under=85 forecast_ml_framework/tests/test_p3_backtest.py -q`
  → `forecast_ml_framework/src/forecast_ml_framework/pipelines/modeling/backtest.py 62 0 100%`
  → `Required test coverage of 85% reached. Total coverage: 100.00%`
- P1 + all P3 existing suites:
  `python -m pytest test_p1_temporal_split.py test_p3_purged_cv.py test_p3_backtest.py test_p3_guards_calibrate.py test_p3_pipeline_integration.py -q`
  → `133 passed`
- Full suite (`cd forecast_ml_framework && python -m pytest -q`):
  → `3305 passed, 4 skipped`
- Pylint gate (pinned 4.1.3, root `.pylintrc`):
  `pylint --rcfile=.pylintrc --fail-under=9.0 .../modeling/backtest.py`
  → `Your code has been rated at 10.00/10`
  Src-wide gate → `9.68/10` (above 9.0).
- Line-length gate: `awk 'length>80'` on both owned `.py`
  files → empty. `.pylintrc` unchanged (`git diff -- .pylintrc` empty).
- Grep guard (also asserted in `test_grep_guard_single_builder`):
  `backtest.py` contains `purged_cv_folds` and
  `fold_leakage_check`, and contains none of `sort_values`,
  `temporal_split`, `_temporal_fold_assignment`,
  `_parse_embargo`, `train_end`, `Timedelta`, nor
  `return []` / `return {}` / `except: pass`.

## Pinned behavior proved

- Exact fixture: `model_fn` call lengths `[53, 73]`, scores
  `-43.5` / `-53.5`, aggregate `-48.5`, `n_folds == 2`,
  `n_train` 53/73, `n_test` 20/20, `leakage_checked is True`.
- 1-fold harness golden: `n_train 73`, `n_test 20`,
  `n_embargoed 7`.
- Fail-loud: `model_fn` None / non-callable (before folds),
  `model_fn` returning `None` naming the fold, missing
  `predict`, `predict` returning `None`, wrong length, scalar,
  and 2-D (all `ValueError` naming the fold), empty folds,
  leaking fold, empty/`None` df, missing `target_col` /
  `time_col` (including `params=None`).
- `metric_fn` raising and `predict` raising propagate as the
  original `RuntimeError` (never caught).
- `default_metric`: classification accuracy (0.75 probe),
  regression negated MAE (-1.0 probe), unknown `ValueError`.
- Determinism: two runs with exact `==` equality.
- Custom `metric_fn` and explicit `feature_cols` paths.
- Classification backtest: zero estimator scores 0.5/0.5/0.5.

## Assumptions

- `default_metric` matches task names case-insensitively via
  `str(task_type).strip().lower()`; any other name raises
  `ValueError`. Exact `'classification'` / `'regression'` pass.
- `BacktestResult.per_fold` `n_train` / `n_test` use
  `len(train_df)` / `len(test_df)` (equal to `fold_meta`).
- `y_true` passed as `test_df[target_col].to_numpy()`;
  estimator receives the `test_df[feature_cols]` frame.
- Missing `feature_cols` entries are not pre-validated;
  pandas would raise `KeyError` (unspecified by the spec).
- `params` forwarded verbatim to `purged_cv_folds`, which
  ignores the extra `target_col` / `task_type` / `feature_cols`
  keys (verified: no unknown-key rejection in P3-A).

## Blockers

None. Awaiting `anti-team-arch` gate.
