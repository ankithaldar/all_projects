# Forecast ML Framework

Config-driven Kedro ML framework for forecasting and tabular prediction. One
repository runs the full lifecycle — ingest, validate, EDA, preprocess,
select features, train a model zoo, calibrate, evaluate, chart, explain
with SHAP/PDP/LIME, register and serve models, batch-score, monitor drift,
plan retraining, track cost, and govern artifacts — almost entirely from
configuration (`variables.yml`, the split `parameters/*.yml` buckets,
`catalog.yml`) plus per-environment overlays. Changing a dataset, a model,
or a loss function is a config edit, not a code edit.

Phases 1–6 are landed (1160 tests pass, see Tests; count supplied by
the Phase 6 teams and checked against `pipeline_registry.py` — 12
registered pipelines — in this docs pass; the suite was not re-run
here). Phase 6 (parity) added ntile eval-metric grammar,
`selected_ntile`/`by_centile`, descriptive stats, the confusion rates
table, lift table, threshold selection, promotion gating, scoring
ntile columns, pipeline wiring, catalog anchors, and `eval_metric`
config. Phase 2 added the MLflow model registry, batch scoring,
Terraform IaC (`infra/`), and the static dashboard (`web/`). Phase 3
added EDA/leakage, monitoring/retraining, the feature store, A/B
shadow serving, cost tracking, governance, interpretability, metrics
v2, the full chart catalog, and Spark scaling. Phase 4 (hardening,
final) registered the Phase 3 pipelines, moved SHAP explanation to
`interpretability.nodes.shap_explain`, and tightened serving/registry
validation. Phase 5 (clean-ux) added Kedro project markers + CLI
introspection (`list-pipelines` / `validate-config` / `dry-run`),
degraded serving health, typed serving errors, sample data, EDA
params, a smoke local profile, and a strict config schema — see the
[Change Log](#change-log).

> **Current-state note (Phase 6 parity).** This README documents the
> code *as found on disk*, verified against `pipeline_registry.py`
> (`_PIPELINE_MODULES`, 12 entries), every `pipelines/*/pipeline.py`,
> `conf/`, `cli.py`, and `serving/app.py`. All eight pipeline modules are **registered** for
> `kedro run --pipeline <name>` (12 display names; former Known
> Issue #12 is fixed).

## Features

- **Config-driven.** `conf/base/variables.yml` (column registry),
  `conf/base/parameters/*.yml` (7 buckets — see
  [Parameters split](#parameters-split-7-buckets)), and
  `conf/base/catalog.yml` (Kedro Data Catalog). No code edits are needed to
  point the framework at a new dataset.
- **Kedro pipelines.** Eight pipeline modules on disk — `data_prep`,
  `modeling`, `scoring`, `eda`, `monitoring`, `retraining`, `cost_tracking`,
  `interpretability` — each with `create_pipeline()` in
  `pipelines/<name>/pipeline.py`. `pipeline_registry.register_pipelines()`
  exposes 12 display names: `data`, `preprocessing`, `feature_selection`
  (`data_prep`), `modeling`, `evaluation`, `predict` (`modeling`),
  `scoring` (`scoring`), `eda`, `monitoring`, `retraining`,
  `cost_tracking`, and `interpretability` (Phase 4 registration;
  former Known Issue #12 fixed).
- **EDA + leakage** (`pipelines/eda/`): `eda_summary` → `eda_report`
  (summary statistics, `eda_quality_checks`, `detect_leakage` — correlation,
  AUC, near-duplicate, near-constant, post-outcome name keywords — wired
  into the report). See [Phase 3 pipelines](#phase-3-pipelines).
- **Monitoring + retraining** (`pipelines/monitoring/`, `pipelines/retraining/`):
  PSI/KS drift (`compute_drift` → `build_monitoring_report` →
  `evaluate_alerts`, severity-graded) and plan-only retraining triggers
  (`should_retrain` → `build_retrain_plan`; never executes Kedro).
- **Feature store, governance, cost** (`feature_store/`, `governance/`,
  `pipelines/cost_tracking/`): offline/online stores with point-in-time
  consistency checks; actor-required secret-scrubbed audit, lineage
  (config hash), retention keep/delete; offline cost estimation from
  `run_info` + config unit prices (no cloud API calls).
- **A/B + shadow serving** (`serving/ab_testing.py`): `ABRouter` with
  stable-hash bucketing, shadow/ab modes, `compare_variants` aggregation
  (mean prediction, latency p50/p95, count).
- **Interpretability pipeline** (`pipelines/interpretability/`): SHAP +
  permutation (`explain_model`), PDP + LIME rows (`interpret_features`),
  diagnostics (`run_diagnostics`), and 27 charts (12 interpretability +
  15 diagnostics in `interpretability/charts.py`).
- **Metrics v2 + full charts.** 144-name `METRIC_REGISTRY`
  (`pipelines/modeling/metrics.py`), selectable via
  `metrics`/`metrics_to_calculate` (`'all'`, a list, or absent = curated
  default subset); chart groups I–VII in `pipelines/modeling/charts.py`
  (data, classification, regression, training, interpretability,
  comparison, specialized). See [Metrics + charts catalog (v2)](#metrics--charts-catalog-v2).
- **Scaling.** Spark partitioned scoring (`scoring_params.spark_scoring`,
  `mapPartitions`, no `toPandas()` on the driver), pandas 5M-row guard
  (`pandas_conversion_params.max_rows_for_pandas`), Dataproc autoscaling
  defaults in `spark_params` (dynamic allocation + AQE). See
  [Scaling: 100M-row guidance](#scaling-100m-row-guidance).
- **Model zoo** (`pipelines/modeling/adapters.py`): H2O AutoML, XGBoost,
  LightGBM, CatBoost, scikit-learn, and any estimator named by a dotted
  `model_handle` (e.g. `xgboost.XGBClassifier`). Train several at once,
  pick the best by `selection_metric`.
- **Calibration.** Classification only: `sigmoid`, `isotonic`, `betacal`, or
  `none` (with `betacal.BetaCalibration` fallback to sigmoid when the
  `betacal` package is absent). Regression calibration is a no-op.
- **SHAP explainability.** `shap.Explainer` with permutation-importance
  fallback (`pipelines/interpretability/nodes.py:shap_explain`); the
  legacy `pipelines/modeling/pipeline.py:explain_model` node wraps it
  for the modeling pipeline. `method: auto` tries SHAP then falls
  back, `method: shap` raises if SHAP is unavailable.
- **Metrics and charts, global and per-bucket/per-decile.** Full metric
  catalogues for classification and regression (below), evaluated globally,
  by `bucket_segment_col`, and by prediction decile
  (`pipelines/modeling/evaluate.py`). Matplotlib charts for both task types
  (`pipelines/modeling/charts.py`).
- **Feature selection plus business/override rules.** Variance, missingness,
  correlation, SelectKBest, L1, and tree-importance filters, then a merge
  rule with `business_*`, `override_*`, and `exclude_*` lists
  (`core/config_loader.py:ConfigLoader.merged_feature_list`,
  `pipelines/data_prep/nodes.py:select_features`).
- **Configurable loss functions per model.** A top-level `loss_function`
  default with per-model overrides, mapped per adapter through a loss
  registry (aliases accepted; exact-allowlist callables only —
  `forecast_ml_framework.losses.huber_loss` and
  `sklearn.metrics.mean_absolute_error` and other names in
  `losses.LOSS_CALLABLES`); any other dotted name raises `ValueError`
  listing valid losses
  (`pipelines/modeling/adapters.py:get_loss`,
  `LOSS_REGISTRY`, `LOSS_KWARGS`; see
  [Loss functions per model](#loss-functions-per-model)).
- **FastAPI serving + registry.** `GET /health`, `POST /predict` with strict
  input validation and payload caps (`serving/app.py`); `create_app()`
  accepts an injected loader or registry config (env `MODEL_REGISTRY_URI` /
  `MODEL_NAME` / `MODEL_ALIAS`), and `RegistryModelLoader`
  (`serving/model_loader.py`) loads via the MLflow registry
  (`registry/model_registry.py`, alias promotion). Serving validation
  uses the model's `feature_names_in_` when present (required feature
  set + order), otherwise the first record's keys; unknown/missing
  features return 422. Registration uses
  `mlflow.sklearn.log_model` (sklearn flavor only, no pyfunc flavor);
  loading defaults to `mlflow.sklearn.load_model`, with the pyfunc
  fallback gated behind `allow_pyfunc=True` (opt-in only), and there
  is still no Docker. Serving auth is optional: when `FORECAST_API_KEY`
  is set, `/predict` requires it (see
  [Serving API](#serving-api)).
- **MLflow / DVC.** `mlflow_params` / `dvc_params` in `parameters/platform.yml`
  are validated by the schema; MLflow tracking/registry runs through
  `registry/model_registry.py` (lazy import — importing the module never
  imports mlflow). DVC remains parameters/config only (no execution code).
- **Spark / BigQuery / GCS / S3 abstraction.** `SparkSessionFactory` with
  graceful local fallback (`core/spark_factory.py`), `SQLQueryDataSet` and
  `SparkSQLDataSet` Kedro datasets (`core/datasets.py`), and
  local/GCS/S3 artifact stores selected by `artifact_store_params.type`
  (`core/artifact_store.py:create_artifact_store`).
- **Pydantic + OmegaConf config validation.** `core/config_schema.py`
  validates every shipped section with `extra='allow'` (unknown keys pass
  through); `load_config()` merges env overlays and resolves `${...}`
  interpolations to plain dicts. `ConfigLoader` accepts plain dicts,
  OmegaConf objects, or pydantic models.

## Quick start

### Prerequisites

- Python **3.14** (`requires-python = '>=3.14'` in `pyproject.toml`).
  Python is not downgraded; libraries that fail on 3.14 move to the
  `extras-314-failures` optional group instead. The compatibility spike
  found all 17 libraries OK (see
  [Python 3.14 compatibility spike](#python-314-compatibility-spike)).
- [uv](https://docs.astral.sh/uv/) (package manager and runner).

### Install

```bash
cd forecast_ml_framework
uv sync
```

This creates `.venv/` inside `forecast_ml_framework/` (never commit it).

### Quickstart: train smoke (under 2 minutes)

From `forecast_ml_framework/` (verified against `cli.py:253`):

```bash
cd forecast_ml_framework
uv sync
uv run forecast-ml validate-config --env local   # prints "config ok (local)"
uv run forecast-ml train --env local             # kedro run --pipeline modeling --env local
```

The `local` overlay is the smoke profile: `models_to_use: ['sklearn']`
only (`conf/local/parameters.yml:60-62`), local MLflow URI
(`./mlruns`), DVC off, `validation_params.fail_fast: true`,
`artifact_store_params.type: local`. Sample data ships at
`data/raw/raw_data.csv` (columns `id,f1,f2,f3,f4,cat,seq,target`,
100 rows, `synthetic_data` shape), so a fresh checkout trains with
no downloads. Kedro discovery needs the project markers in
`pyproject.toml:77-80` (`[tool.kedro]` with `package_name` /
`project_name` / `kedro_init_version` only — no `project_version`)
plus `<package>.cli:cli` (`cli.py:23-51`), which re-exports Kedro's
top-level Click group; without the marker Kedro 1.7 exits with
"unexpected keys" or fails to load project commands.
`forecast-ml` also exposes the alias `forecast-ml-framework`
(`pyproject.toml:75`) alongside `forecast-ml`.

```bash
uv run forecast-ml list-pipelines --env local    # 12 lines: "<name> <module>"
uv run forecast-ml validate-config --env local   # "config ok (local)"
uv run forecast-ml dry-run modeling --env local  # planned command + node names
uv run forecast-ml dry-run eda --env local
```

`list-pipelines` prints one `<name> <module>` line per registered
pipeline (`cli.py:195-208`). `validate-config` loads the env config
and validates parameters, printing `config ok (<env>)` on success
(`cli.py:211-230`). `dry-run <pipeline>` prints the planned
`kedro run --pipeline <name> --env <env>` command plus node names,
without running anything (`cli.py:233-250`). `--env` accepts
`base|local|gcp|aws` only; anything else exits 2 (`cli.py:63-68`).

### Environment variables

| Variable | Use | Default |
|---|---|---|
| `MODEL_REGISTRY_URI` | MLflow registry URI for `RegistryModelLoader` (`serving/app.py:231`) | unset |
| `MODEL_NAME` | Registered model name | unset → degraded mode |
| `MODEL_ALIAS` | Registry alias | `production` |
| `MODEL_ARTIFACT_PATH` | File-loader artifact path (`serving/app.py:224`) | unset |
| `FORECAST_API_KEY` | When set, `/predict` requires it (`X-API-Key` or `Authorization: Bearer ...`); `/health` stays public (401 otherwise) | off |
| `GOOGLE_APPLICATION_CREDENTIALS` | GCP credentials for GCS/BigQuery | unset |
| `AWS_PROFILE` / AWS env vars | AWS credentials for S3/Athena | unset |
| `MLFLOW_DISABLE_AGENT_HINT` | Silence MLflow import banner | unset |

### Tests

Tests live in the top-level `tests/` directory (see the
[Testpaths note](#testpaths) in Known Issues for why the explicit path
below is required):

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

Uv-run variants (same suite, inside the uv environment):

```bash
uv run python -m pytest tests/ -q
uv run pytest tests/ -q
```

Both forms were used to verify this README's commands. Expected result on
success: `pytest` exits 0 with all tests passing — **1160 tests** at Phase 6
land (48 test files in `tests/`, including `test_eda`, `test_monitoring`,
`test_retraining`, `test_cost_tracking`, `test_interpretability`,
`test_feature_store`, `test_governance`, `test_ab_testing`, `test_scaling`,
`test_phase3_config`, `test_pipeline_registry`, `test_demo_run`,
`test_kedro_markers`, `test_smoke_defaults`, `test_eda_params`,
`test_schema_strictness`, `test_validation_profile`, `test_cli`,
`test_main_entrypoint`, `test_catalog_gaps`). Note: the real-MLflow roundtrip test carries
`@pytest.mark.serial` and resets global mlflow state (Known Issue #17) —
run the full suite serially; do not parallelise it. `pyproject.toml` now
sets `testpaths = ['tests']` and `pythonpath = ['src']`, so a bare
`pytest` also works. Environment-dependent tests (Spark, cloud SDKs) are
mocked or skipped when the dependency or service is absent; `httpx` is a
dev-only dependency so the FastAPI `TestClient` works without polluting
production dependencies.

### Run the pipelines

```bash
uv run kedro run --env local
```

`--env local` selects the `conf/local/` overlay (local paths, local
artifact store, local MLflow tracking URI, DVC disabled). `--env gcp` and
`--env aws` select the GCS/BigQuery and S3/Athena overlays. Run a single
pipeline family with `--pipeline <name>` (registered names, 12 total:
`data`, `preprocessing`, `feature_selection`, `modeling`, `evaluation`,
`predict`, `scoring`, `eda`, `monitoring`, `retraining`, `cost_tracking`,
`interpretability`).

### Console usage

The console script is `forecast-ml` (declared in `pyproject.toml`,
entry point `forecast_ml_framework.__main__:main`, which delegates to
`forecast_ml_framework.cli:main`):

```bash
uv run forecast-ml --help
uv run forecast-ml train --help
uv run forecast-ml predict --help
uv run forecast-ml serve --help
uv run forecast-ml run --help
uv run forecast-ml list-pipelines --env local
uv run forecast-ml validate-config --env local
uv run forecast-ml dry-run modeling --env local
```

Subcommands: `train`, `predict`, `serve`, `run`, each accepting
`--env` (default `local`), `--artifact`, `--host` (default `127.0.0.1`),
and `--port` (default `8000`), plus `list-pipelines`, `validate-config`,
and `dry-run <pipeline>` (accept `--env` only; see Quickstart). `train`
delegates to `kedro run --pipeline modeling --env <env>`, `predict` to
`--pipeline scoring`, `run` to the default pipeline (no `--pipeline`),
`serve` to `uvicorn forecast_ml_framework.serving.app:app --host <host>
--port <port>` with the artifact passed via
`MODEL_ARTIFACT_PATH` (`cli.py:56-99`). When `kedro`/`uvicorn` is not on
`PATH`, the CLI prints the planned command instead; when the Kedro
project root cannot be found it exits 2 (`cli.py:156-192`).

### Synthetic demo walkthrough

`pipelines/data_prep/nodes.py:synthetic_data()` generates a reproducible
demo frame (`n=100`, `seed=42`, `task='classification'` by default) with
columns `id`, `f1`–`f4` (numeric), `cat` (categorical `a`/`b`/`c`), `seq`
(list of floats), and `target` (binary). Walk the nodes in Python:

```python
import numpy as np  # noqa: F401 - available via the framework anyway

from forecast_ml_framework.pipelines.data_prep.nodes import (
    preprocess,
    select_features,
    synthetic_data,
    validate_data,
)

df = synthetic_data(n=100, task='classification', seed=42)
variables = {
    'target_col': 'target',
    'feature_cols': ['f1', 'f2', 'f3', 'f4'],
    'num_cols': ['f1', 'f2', 'f3', 'f4'],
    'cat_cols': ['cat'],
    'seq_cols': ['seq'],
}
params = {
    'validation_params': {'fail_fast': False, 'max_null_ratio': 0.5},
    'data_prep_params': [
        {'Imputations': {'skip': False, 'args': {
            'strategy': 'median',
            'cols': {'include_cols': ['f1', 'f2', 'f3', 'f4']}}}},
    ],
    'feature_selection_params': {'missing_threshold': 0.5},
}
df, report = validate_data(df, params, variables)
df = preprocess(df, params['data_prep_params'], variables)
df, selected, report = select_features(df, params, variables)
print(selected)  # merged feature list after business/override rules
```

For `task='regression'`, `target` becomes `2*f1 + 0.5*f2 + noise` and the
modeling pipeline consumes it as `modeling_reg_params`.

## Architecture

> **Diagrams.** All diagrams in this README are Mermaid fenced
> blocks and render natively on GitHub. Node names
> match `pipeline_registry.py` and the `name=` values in
> `pipelines/*/pipeline.py`. All eight modules are registered
> (12 display names, Phase 4 final).

### System architecture

```mermaid
flowchart TD
  USER["User / console<br/>uv run forecast-ml train|predict|serve"] --> CLI["forecast-ml<br/>__main__ -> cli.main"]
  CLI --> KEDRO["Kedro CLI<br/>kedro run --env local|gcp|aws"]
  KEDRO --> REG["pipeline_registry.register_pipelines()"]
  REG --> DP["data_prep pipeline<br/>data / preprocessing / feature_selection"]
  REG --> MP["modeling pipeline<br/>modeling / evaluation / predict"]
  REG --> SC["scoring pipeline<br/>scoring<br/>score_data -> predict -> write_predictions"]
  DP --> ART["Artifacts<br/>catalog datasets"]
  MP --> ART
  SC --> ART
  MP --> REGISTRY["MLflow registry<br/>registry/model_registry.py<br/>alias promotion"]
  MP --> SERVE["FastAPI serving<br/>GET /health POST /predict<br/>serving/app.py + model_loader.py"]
  SERVE --> AB["A/B + shadow<br/>serving/ab_testing.py<br/>ABRouter"]
  SERVE --> WEB["Web dashboard<br/>web/ static SPA<br/>python3 -m http.server"]
  ART --> LOCAL["local overlay<br/>./data ./artifacts"]
  ART --> GCP["gcp overlay<br/>BigQuery + GCS<br/>Dataproc via Spark"]
  ART --> AWS["aws overlay<br/>Athena + S3<br/>EMR via Spark"]
  REG --> EDA["eda pipeline<br/>eda_summary -> eda_report"]
  EDA -.-> DP
  REG --> MON["monitoring pipeline<br/>compute_drift -> build_monitoring_report<br/>-> evaluate_alerts"]
  MON -.-> RET["retraining pipeline<br/>should_retrain -> build_retrain_plan"]
  REG --> COST["cost_tracking pipeline<br/>estimate_cost -> track_run_cost<br/>-> summarize_costs"]
  COST -.-> ART
  REG --> INTERP["interpretability pipeline<br/>explain_model -> interpret_features<br/>-> run_diagnostics + charts"]
  INTERP -.-> ART
  FS["feature_store/<br/>governance/<br/>offline+online, audit/lineage"] -.-> DP
  FS -.-> SERVE
```

### Folder structure

```text
forecast_ml_framework/
├── pyproject.toml              # deps, console script, pytest/ruff config
├── README.md                   # this file
├── uv.lock                     # resolved dependency lockfile
├── conf/                       # Kedro config (base + env overlays)
│   ├── base/
│   │   ├── variables.yml       # column registry (target, id/cat/num/date/seq/ind/drop)
│   │   ├── parameters/         # split parameters buckets (7 files, see below)
│   │   │   ├── global.yml      # runtime keys + global_params
│   │   │   ├── data.yml        # data sources, sampling, split, validation, pandas guard
│   │   │   ├── preprocessing.yml # FE registry + feature selection
│   │   │   ├── modeling.yml    # models, H2O, modeling(_reg), calibration, cv
│   │   │   ├── evaluation.yml  # evaluation_params incl. metrics_to_calculate
│   │   │   └── platform.yml    # spark, mlflow, dvc, shap, registry, scoring, infra, monitoring,
│   │   │                       #   retraining, feature_store, ab, cost, governance
│   │   └── catalog.yml         # Kedro Data Catalog entries (+ Phase 3: monitoring_report,
│   │                           #   alerts, cost_log, lineage, audit_log)
│   ├── local/                  # local overlay: catalog + parameters
│   ├── gcp/                    # GCS + BigQuery overlay: catalog
│   └── aws/                    # S3 + Athena overlay: catalog
├── infra/                      # Terraform IaC (Phase 2)
│   ├── gcp/                    # GCS, BigQuery, Dataproc (off by default), scheduler, Pub/Sub alerts
│   ├── aws/                    # S3, Athena, EMR stub (off), EventBridge stub
│   └── README.md               # apply order, state, costs, 100M-row sizing
├── web/                        # static evaluation dashboard (Phase 2, no build step)
│   ├── index.html / app.js / styles.css
│   └── sample/                 # sample metrics.json + chart PNGs
├── src/forecast_ml_framework/  # importable package (PYTHONPATH=src)
│   ├── __init__.py             # __version__ = '0.1.0'
│   ├── __main__.py             # forecast-ml console entry point
│   ├── cli.py                  # train|predict|serve|run argparse CLI
│   ├── settings.py             # Kedro settings: CONF_SOURCE, HOOKS=()
│   ├── pipeline_registry.py    # register_pipelines(), lazy imports (12 display names, 8 modules)
│   ├── core/
│   │   ├── artifact_store.py   # local/GCS/S3 stores, restricted unpickler
│   │   ├── config_loader.py    # ConfigLoader: lookup, task_type, feature merge
│   │   ├── config_schema.py    # Pydantic models + OmegaConf load_config()
│   │   ├── datasets.py         # SQLQueryDataSet, SparkSQLDataSet
│   │   └── spark_factory.py    # SparkSessionFactory, to_pandas(), pandas 5M guard, autoscaling defaults
│   ├── pipelines/
│   │   ├── data_prep/          # pipeline.py, nodes.py
│   │   ├── modeling/           # pipeline.py, adapters.py, calibration.py,
│   │   │                       # data_split.py, evaluate.py, metrics.py (144 registry),
│   │   │                       # charts.py (groups I-VII); SHAP via interpretability.nodes.shap_explain;
│   │   │                       # ntile.py, lift_table.py, threshold.py, validation.py (Phase 6 parity)
│   │   ├── scoring/            # pipeline.py, nodes.py (batch scoring, Spark partitioned)
│   │   ├── eda/                # pipeline.py, nodes.py (summary + leakage)
│   │   ├── monitoring/         # pipeline.py, nodes.py (PSI/KS drift, alerts)
│   │   ├── retraining/         # pipeline.py, nodes.py (plan-only triggers)
│   │   ├── cost_tracking/      # pipeline.py, nodes.py (offline cost estimation)
│   │   └── interpretability/   # pipeline.py, nodes.py, charts.py (SHAP/PDP/LIME + 27 charts)
│   ├── feature_store/          # store.py, offline.py, online.py (point-in-time consistency)
│   ├── governance/             # audit.py, lineage.py, retention.py (pure functions, no persistence)
│   ├── registry/               # model_registry.py (MLflow registry, alias promotion, lazy import)
│   └── serving/                # app.py (FastAPI), model_loader.py, ab_testing.py
├── tests/                      # pytest suite (48 test files, 1160 tests)
└── .venv/                      # created by `uv sync`, never commit
```

### Pipeline list and nodes

| Display name | Real module | Nodes (in order) |
|---|---|---|
| `data`, `preprocessing`, `feature_selection` | `pipelines/data_prep/pipeline.py` | `data_ingestion_validate` → `preprocessing` → `feature_selection` |
| `modeling`, `evaluation`, `predict` | `pipelines/modeling/pipeline.py` | `split_data_node` → `train_models_node` → `calibrate_model_node` → `predict_node` → `predict_proba_node` → `evaluate_model_node` → `build_lift_table_node` → `build_promotion_node` → `make_charts_node` → `explain_model_node` |
| `scoring` | `pipelines/scoring/pipeline.py` | `score_data` (`load_scoring_data`) → `predict` → `write_predictions` |
| `eda` | `pipelines/eda/pipeline.py` | `eda_summary` → `eda_report` |
| `monitoring` | `pipelines/monitoring/pipeline.py` | `compute_drift` → `build_monitoring_report` → `evaluate_alerts` |
| `retraining` | `pipelines/retraining/pipeline.py` | `should_retrain` → `build_retrain_plan` |
| `cost_tracking` | `pipelines/cost_tracking/pipeline.py` | `estimate_cost` → `track_run_cost` → `summarize_costs` |
| `interpretability` | `pipelines/interpretability/pipeline.py` | `explain_model` → `interpret_features` → `run_diagnostics` → `make_interpretability_charts` → `make_diagnostics_charts` |

`pipeline_registry.register_pipelines()` maps each of the 12 display
names to one cached pipeline object (aliases for the same module share
one object) and imports lazily, so an unavailable pipeline module is
skipped instead of breaking the registry. `_PIPELINE_MODULES`
(`pipeline_registry.py:19`) covers all eight modules; `kedro run
--pipeline eda` (etc.) works (Phase 4; former Known Issue #12 fixed).

### How data flows

```text
raw_data (catalog)
  │  pipelines/data_prep
  ▼
validated_data ── preprocess (FE registry, in order) ──► preprocessed_data
  │                                                     │
  │                                                     ▼
  │                                              select_features
  │                                                     │
  │                                                     ▼
  └────────────────────────────►  train_data_selected ─┘
                                                        │
                                                        ▼  modeling pipeline
  modeling_input ── split_data ──► x_train / x_test / y_train / y_test
                                        │
                                        ▼ train_models (zoo, best by selection_metric)
                                   best_model ── calibrate_model ──► calibrated_model
                                        │                              │
                                        │                              ▼
                                        │                        y_pred / y_pred_proba
                                        │                              │
                                        ▼                              ▼
                                   per_model_metrics           evaluate_model ──► evaluation_metrics
                                                                  (global / by_bucket / by_decile)
                                        │                              │
                                        ▼                              ▼
                                   explanations              make_charts ──► charts
                                   (SHAP/permutation)        explain_model ──► shap_artifact
```

In practice the catalog wires `train_data` into the data-prep pipeline
and the modeling pipeline starts from `modeling_input`; `model`,
`calibration_model`, and per-adapter model datasets persist the fitted
models, `metrics`/`metrics_csv` persist evaluation output, `predictions`
persists predictions, and `shap_values`/`shap_summary` persist the
explanation artifacts.

### Data flow (Mermaid)

```mermaid
sequenceDiagram
  participant RAW as raw SQL / files<br/>catalog raw_data
  participant VAL as data_ingestion_validate
  participant FE as preprocessing<br/>FE registry in order
  participant FS as feature_selection<br/>business/override rule
  participant SPLIT as split_data_node
  participant TRAIN as train_models_node<br/>model zoo
  participant CAL as calibrate_model_node
  participant EVAL as evaluate_model_node
  participant OUT as charts + SHAP<br/>make_charts / explain_model
  participant STORE as catalog datasets<br/>model / metrics / predictions
  participant API as FastAPI serving<br/>/health /predict
  RAW->>VAL: train_data + validation_params
  VAL->>FE: validated_data
  FE->>FS: preprocessed_data
  FS->>SPLIT: train_data_selected
  SPLIT->>TRAIN: x_train y_train x_test y_test
  Note over TRAIN: H2O / XGBoost / LightGBM / CatBoost / sklearn<br/>best by selection_metric
  TRAIN->>CAL: best_model
  CAL->>EVAL: calibrated_model y_pred y_pred_proba
  EVAL->>OUT: evaluation_metrics<br/>global / by_bucket / by_decile
  OUT->>STORE: charts + shap_artifact
  TRAIN->>STORE: model + per_model_metrics
  STORE->>API: FileModelLoader artifact
```

```mermaid
flowchart LR
  A["raw SQL / files"] --> B["validation<br/>data_ingestion_validate"]
  B --> C["FE registry<br/>preprocessing"]
  C --> D["feature selection<br/>business/override rule"]
  D --> E["split / sample<br/>split_data_node"]
  E --> F["model zoo<br/>H2O XGB LGBM CatBoost sklearn"]
  F --> G["calibration<br/>calibrate_model_node"]
  G --> H["evaluation<br/>metrics + charts + SHAP"]
  H --> I["registry / serving<br/>catalog + FastAPI"]
```

## Phase 3 pipelines

All five Phase 3 pipelines are plain pandas/numpy modules under
`src/forecast_ml_framework/pipelines/<name>/` with `create_pipeline()` +
`nodes.py`. All five are **registered** in `pipeline_registry.py`
(Phase 4) and run via `kedro run --pipeline <name>`.
Params and catalog datasets are documented in each `pipeline.py` docstring.

### EDA + leakage — `pipelines/eda/`

`eda_summary(raw_data, params:eda_params, variables)` → `eda_report(...)`.
`eda_summary` computes standalone summary statistics (shape, dtypes,
missing rates, numeric descriptives, correlations, top categorical values).
`eda_report` adds `eda_quality_checks` findings (`high_missing`,
`cardinality_explosion`, `class_imbalance`, `constant_cols`,
`duplicates`) and `detect_leakage` findings — feature-target correlation
(`leakage_threshold`, default 0.95), single-feature AUC
(`leakage_auc_threshold`, 0.99), near-duplicate features (0.99),
near-constant features (0.95), and post-outcome name keywords
(`leakage_name_keywords`: `target`, `attrition`, `label`, `outcome`,
`result`, `after`, `post`). Outputs: `eda_summary`, `eda_report`
(summary + findings + recommendations). Catalog: `raw_data` in,
`variables` in (target column wins over `eda_params.target_col`).

### Monitoring / drift + alerts — `pipelines/monitoring/`

`compute_drift(reference_data, current_data, params:monitoring_params)` →
`build_monitoring_report(drift_report, reference_predictions,
current_predictions, current_metrics, ...)` → `evaluate_alerts(...)`.
Per-feature **PSI** (quantile bins for numeric, union-of-categories for
categorical; `bins` default 10, `epsilon` 1e-4 smoothing) and **KS**
(two-sample for numeric, total-variation analogue for categorical), plus
missing-rate deltas; `prediction_drift` and `performance_decay` cover
prediction and metric drift. Empty/small inputs (`min_rows` default 10)
return NaN stats + a warning, never raise. `evaluate_alerts` grades
severity from `thresholds` (defaults: psi 0.1/0.25, ks 0.1/0.25,
missing_rate_delta 0.05/0.1 for warning/critical). Outputs:
`drift_report`, `monitoring_report`, `alerts`. Catalog Phase 3 entries:
`monitoring_report`, `alerts` (`conf/base/catalog.yml`).

### Retraining triggers — `pipelines/retraining/`

`should_retrain(monitoring_report, last_train_info,
params:retraining_params)` → `build_retrain_plan(retrain_decision, ...)`.
**Plan only** — `build_retrain_plan` returns a plain dict describing a
would-be `modeling` run for `env`/`pipeline_name`; it never executes
Kedro, scores, or writes a catalog. Triggers (any fires → retrain; `null`
disables): drift-alert count ≥ `drift_alert_threshold` (default 1),
performance decay > `performance_decay_threshold` (0.05), schedule
elapsed ≥ `schedule_days` (30; `platform.yml` ships 7), data-volume
growth ≥ `data_volume_growth_threshold` (0.20). Inject `now` in params
for deterministic schedule checks in tests.

### Cost tracking — `pipelines/cost_tracking/`

`estimate_cost(run_info, params:cost_tracking_params)` → `track_run_cost`
→ `summarize_costs(cost_log)`. Offline arithmetic only — **no cloud API
calls**; usage numbers arrive as plain dict data in `run_info`
(`runtime_hours`, `bytes_scanned`, `artifact_gb`, model/env/month) and
unit prices come from config (`compute_price_per_hour`,
`data_scan_price_per_gb`, `storage_price_per_gb_month`, `currency`;
flat, `unit_prices`, `cost_tracking_params`, or `cost_params` nesting;
`platform.yml` ships `cost_params`). Missing/negative values warn and
count as 0, never crash. Outputs: `cost_estimate`, `cost_record`,
`cost_summary` (totals by model/env/month). Catalog: `cost_log`.

### Interpretability pipeline — `pipelines/interpretability/`

`explain_model(model, X, y, params:interpretability_params)` (SHAP via
`shap.Explainer`, permutation/occlusion surrogate fallback; `method`
records which ran) → `interpret_features(model, X, ...)` (PDP per
feature + LIME rows, LIME/permutation surrogate fallback) →
`run_diagnostics(model, X, y, ...)` (calibration, residuals, segments) →
`make_interpretability_charts` + `make_diagnostics_charts`. Params:
`max_rows` (20), `top_k` (10), `n_repeats` (5), `random_state`/`seed`
(42), `n_grid` (20), `pdp_features`, `n_calibration_bins` (10),
`segment_col`, `scorer`. **Chart catalog (27 charts)** in
`interpretability/charts.py` — interpretability (12): `shap_beeswarm`,
`shap_bar`, `shap_heatmap`, `shap_dependence`, `shap_waterfall`,
`shap_force`, `pdp_1d_chart` (+ICE), `pdp_2d_heatmap`,
`permutation_importance_bar`, `lime_bar`, `decision_boundary_chart`,
`tree_plot`; diagnostics (15): `calibration_curve`,
`residual_vs_fitted`, `residual_hist_qq`, `scale_location`,
`error_by_segment_bar`, `prediction_distribution_by_class`,
`confidence_histogram`, `residuals_over_time_chart`, `cooks_distance`,
`rec_curve`, `learning_curve`, `validation_curve`, `loss_vs_epoch`,
`cv_box`, `overfitting_gap`. Every chart returns a closed `Figure` or
`None` on insufficient data, never raises.

### Feature store, governance, A/B shadow

- **Feature store** (`feature_store/`): `FeatureStore` ABC with
  `OfflineFeatureStore` / `OnlineFeatureStore`; frames are normalised to
  canonical `entity_id` + `as_of` columns; `check_consistency(offline,
  online, ...)` verifies offline/online agreement at a point in time.
  Config: `feature_store_params` (`offline_path`, `online_backend`,
  `entity_key`) in `platform.yml`. Entity-key mismatch between offline
  and online causes train/serve skew — keep `entity_key` in sync.
- **Governance** (`governance/`): pure functions, **no persistence** —
  the caller (or catalog) saves the returned dicts. `audit.py`:
  `audit_log(event, actor, details, ...)` requires a non-blank actor
  and deep-scrubs secret-like keys (`password`/`secret`/`token`/
  `api_key`/…) to `***REDACTED***`. `lineage.py`: run lineage with
  `config_hash` = sha256 of canonical JSON of `variables`+`parameters`.
  `retention.py`: `check_retention` splits artifacts into keep/delete
  by age (delete only when age **strictly greater** than
  `retention_days`; unusable ages are kept). Config: `governance_params`
  (`retention_days: 365`, `audit_actor_required: true`). Catalog:
  `lineage`, `audit_log`.
- **A/B + shadow** (`serving/ab_testing.py`): `ABRouter(champion_loader,
  challenger_loader, traffic_split=..., mode='shadow'|'ab')` buckets
  `request_id` by stable SHA-256 hash; shadow mode returns the champion
  prediction while logging the challenger, `ab` mode routes a fraction
  to the challenger. `compare_variants(logs)` aggregates per variant
  (mean prediction, latency p50/p95, count). Loaders are injected
  callables (e.g. `FileModelLoader` / `RegistryModelLoader`) — no MLflow
  access in this module. Config: `ab_params` (`traffic_split: 0.1`,
  `mode: 'shadow'`) in `platform.yml`.

## Configuration guide

Kedro loads `conf/base/` and merges the selected env overlay
(`conf/local/`, `conf/gcp/`, `conf/aws/`) over it — overlay entries
override base entries of the same name, everything else is inherited.
`settings.py` sets `CONF_SOURCE = 'conf'` and `HOOKS = ()` (the seam for
future MLflow/DVC hooks).

Two loading paths exist:

- **Kedro runtime** — Kedro's `OmegaConfigLoader` reads the same files.
- **Programmatic** — `core/config_schema.py:load_config(env_dir)` merges
  base + overlay with OmegaConf, resolves interpolations, and returns
  `{'variables', 'parameters', 'catalog'}` as plain dicts.
  `validate_variables()` / `validate_parameters()` validate the result
  against the pydantic models.

### Config flow (Mermaid)

```mermaid
flowchart TD
  V["variables.yml<br/>column registry"] --> OC["OmegaConf merge<br/>base + env overlay"]
  P["parameters/*.yml<br/>7 buckets: global data preprocessing<br/>modeling evaluation eda platform"] --> OC
  C["catalog.yml<br/>Kedro datasets"] --> OC
  OC --> RES["OmegaConf.resolve()<br/>interpolations ${...}"]
  RES --> PM["Pydantic validation<br/>config_schema.py<br/>extra=allow"]
  PM --> CL["ConfigLoader<br/>config_loader.py"]
  CL --> DP["data_prep pipeline<br/>data preprocessing feature_selection"]
  CL --> MP["modeling pipeline<br/>modeling evaluation predict"]
  CL --> SC["scoring pipeline<br/>scoring"]
  CL --> P3["Phase 3 pipelines<br/>eda monitoring retraining<br/>cost_tracking interpretability<br/>registered (Phase 4)"]
  subgraph ENV["env overlays - override base by name"]
    L["conf/local<br/>catalog + parameters"]
    G["conf/gcp<br/>catalog - GCS BigQuery"]
    A["conf/aws<br/>catalog - S3 Athena"]
  end
  L -. "merge over base" .-> OC
  G -. "merge over base" .-> OC
  A -. "merge over base" .-> OC
```

### `variables.yml` — the column registry

Delivered schema (synthetic demo: one row per customer snapshot,
classification by default):

```yaml
target_col: 'target'            # target column name
id_cols: ['id']                 # identifiers: never features
cat_cols: ['region', 'plan_type', 'channel']   # categorical → encoded
num_cols: ['age', 'income', 'tenure_days', 'purchase_count']  # numeric
date_cols: ['signup_date', 'last_purchase_date']               # dates
seq_cols: ['monthly_spend_seq'] # sequences → Sequence* FE steps
ind_cols: ['is_premium', 'has_support_ticket']                 # 0/1 indicators
drop_cols: ['email_hash']       # dropped before modelling (leakage/PII)
bucket_segment_col: ${.cat_cols[0]}   # segment for per-segment metrics
business_relevant_features: []  # always kept through selection
override_features: []           # forced in even if selection drops them
exclude_features: []            # always excluded
```

For a regression target, swap `target_col` to a continuous column and
point `data_params.train_data` at `raw_data_regression` (commented
example at the bottom of the file). `bucket_segment_col` uses an
OmegaConf interpolation (`${.cat_cols[0]}`).

Validation model: `VariablesConfig` (`core/config_schema.py`).

### Parameters split (7 buckets)

The former single `conf/base/parameters.yml` **no longer exists**. Its
sections were split into 7 files under `conf/base/parameters/`; the
OmegaConf/Kedro loader merges every `*.yml` in that directory, so all
top-level section names are unchanged and existing `params:<section>`
references keep working. **Migration (Known Issue #13):** any tooling,
script, or checkout that reads/writes `conf/base/parameters.yml` directly
must be pointed at the bucket files below — edit the bucket that owns
your section.

| Bucket file | Owns (top-level sections) |
|---|---|
| `conf/base/parameters/global.yml` | Flat runtime keys (`score_model_id`, `run_date`, …) + `global_params` |
| `conf/base/parameters/data.yml` | `data_params`, `sampling_params`, `sampling_reg_params`, `pandas_conversion_params` (incl. `max_rows_for_pandas: 5000000`), `train_test_split_params`, `validation_params` |
| `conf/base/parameters/preprocessing.yml` | `data_prep_params` (FE registry), `feature_selection_params` |
| `conf/base/parameters/modeling.yml` | `models_to_use`, `h2o_automl_params`, `modeling_params`, `modeling_reg_params`, `model_params_by_name`, `calibration_params`, `cv_params` |
| `conf/base/parameters/evaluation.yml` | `evaluation_params` incl. `metrics_to_calculate` |
| `conf/base/parameters/eda.yml` | `eda_params` (EDA + leakage thresholds) |
| `conf/base/parameters/platform.yml` | `spark_params`, `artifact_store_params`, `mlflow_params`, `dvc_params`, `shap_params`, `chart_params`, `registry_params`, `scoring_params`, `infra_params`, `monitoring_params`, `retraining_params`, `feature_store_params`, `ab_params`, `cost_params`, `governance_params` |

`conf/local/parameters.yml` still exists and overrides only
`artifact_store_params`, `mlflow_params`, `dvc_params`, `spark_params`,
`validation_params` (fail_fast smoke), and `models_to_use` (sklearn-only
smoke). `conf/gcp/` and `conf/aws/` ship `catalog.yml` only, so all their
parameters come from base.

### `parameters/*.yml` — every section

Base defaults below (section → bucket mapping in the table above).
`eda_params` lives in its own bucket `conf/base/parameters/eda.yml`
(Phase 5): typed `EdaParams` in `core/config_schema.py:269-291`
field-for-field with `pipelines/eda/nodes.py` defaults —
`target_col`, `missing_threshold` (0.2), `cardinality_threshold` (50),
`leakage_threshold`/`leakage_corr_threshold` (0.95),
`duplicate_corr_threshold` (0.99), `leakage_auc_threshold` (0.99),
`near_constant_threshold` (0.95), `leakage_name_keywords` (7 keywords),
`imbalance_ratio_threshold` (5.0), `zscore_threshold` (3.0),
`top_correlations` (10), `top_values` (3). Wrong types raise
`ValidationError` (`tests/test_schema_strictness.py:21-34`);
`ConfigLoader.task_type` defaults to `'classification'` everywhere
(single default, `config_loader.py`).

#### Runtime params (catalog interpolations)

Flat keys referenced by `catalog.yml` as `${params:<key>}` (Kedro params
namespace), mostly interpolations of `global_params`:

```yaml
score_model_id: ${.global_params.score_model_id}
output_gcs_location: 'artifacts'
model_id_prefix: 'demo_model_v1'
suffix: 'v1'
run_date: ${.global_params.run_date}
train_run_dt: ${.global_params.run_date}
score_run_dt: ${.global_params.run_date}
# data_project, db_staging_data, etl_project, etl_dataset, etl_table,
# external_project, tmp_bucket, brand: '' ; is_external_platform: false
```

#### Global / run

```yaml
global_params:
  is_regression: false          # true → regression defaults
  is_external_platform: false
  score_model_id: 'demo_model_v1'
  run_date: '2026-11-05'
  model_type: 'sklearn'
  random_state: 42
  target_col: ${variables.target_col}   # interpolation into variables.yml
  key_cols_list: ['id']
```

#### Data sources

```yaml
data_params:
  train_data: 'raw_data'        # catalog dataset name (catalog.yml)
  test_data: 'raw_data'
```

#### Sampling (classification and regression)

```yaml
sampling_params:                # contract: {method, test_size, stratify_col}
  method: 'stratified'          # 'stratified' | 'random'
  test_size: 0.2
  stratify_col: ${..global_params.target_col}
sampling_reg_params:            # same shape, method 'random'
  method: 'random'
  test_size: 0.2
  stratify_col: ${..global_params.target_col}
```

#### Spark → pandas conversion

```yaml
pandas_conversion_params:
  convert_to_pandas: true       # false + Spark input → TypeError
  category_cols: []
```

#### Evaluation

```yaml
evaluation_params:
  multiclass_flag: false
  prediction_col: 'xgboost_pred'
  prediction_probability_col: 'score_value'
  metrics:
    classification: ['roc_auc', 'average_precision', 'f1']
    regression: ['rmse', 'mae', 'r2']
```

#### Train/test split

```yaml
train_test_split_params:        # {training_split_ratio, test_split_ratio, ...}
  training_split_ratio: 0.8
  test_split_ratio: 0.2
  stratify_col: ${..global_params.target_col}
  shuffle: true
  random_state: 42
```

#### Data preparation: the ordered FE registry

`data_prep_params` is a list of steps that run **in list order**. Each
step is `{StepName: {skip: false, args: {...}}}`; `skip: true` disables
it. Steps with a single importable class carry `args.handle` (a dotted
importable string resolved via `resolve_handle()`); steps with no single
importable class are implemented directly in `data_prep/nodes.py`
(`CapOutliers`, `HighLabelBinning`, `FixedLength`, `Slope`,
`Similarity`, `Sequence*` — see the `# Note:` comments in the file).
`args.cols` uses `{exclude_cols, include_cols}`; an empty `include_cols`
means "no-op / all, per implementation".

Shipped steps (enabled in base): `Imputations` (numeric, median,
`sklearn.impute.SimpleImputer`), `Imputations` (categorical, mode),
`CategoricalEncoding` (label via `pyspark.ml.feature.StringIndexer`,
one-hot via `pyspark.ml.feature.OneHotEncoder`, `ohe_flag: true`).
Disabled in base: `CapOutliers`, `QuantileDiscretizer`,
`Normalizations`, `HighLabelBinning`, `TargetEncoding` (skipped
`sklearn.preprocessing.TargetEncoder`), `FixedLength`, `Slope`,
`Similarity`, `SequenceAverage`, `SequenceMin`, `SequenceMax`,
`SequenceLast`, `SequenceDelta`, `SequenceNormalize`.

```yaml
# Enabled example (numeric imputation):
- Imputations:
    skip: false
    args:
      handle: 'sklearn.impute.SimpleImputer'
      strategy: 'median'
      custom_bounds: {}
      cols:
        exclude_cols: []
        include_cols: ['age', 'income', 'tenure_days', 'purchase_count']
# Disabled example:
- CapOutliers:
    skip: true
    args:
      method: 'bound'
      bound: {lower: 0.01, upper: 0.99}
      cols: {exclude_cols: [], include_cols: []}
```

Unknown step names do not crash: `preprocess()` logs a warning and
records them in the report (`unknown_steps`).

#### Feature selection

```yaml
feature_selection_params:       # code contract in data_prep.nodes.select_features
  enabled: true
  random_state: 42
  is_regression: false
  variance_threshold: 0.0       # drop numeric cols with var <= threshold
  missing_threshold: 0.5        # drop numeric cols with null ratio > threshold
  correlation: {enabled: false, threshold: 0.95}
  kbest: {enabled: false, k: 20, score_func: 'f_classif'}
  l1: {enabled: false, alpha: 0.01, C: 1.0}
  tree: {enabled: false, n_estimators: 200, threshold: 0.01}
  business_features: []         # aliases honoured: business_relevant_features,
  override_features: []         #   feature_override, features_override, ...
  exclude_features: []
  output: 'selected_features'
```

The merge rule is documented
[below](#feature-selection--businessoverride-rule). `kbest.score_func`
accepts `f_classif`, `f_regression`, `mutual_info_classif`,
`mutual_info_regression`.

#### Models to use

```yaml
models_to_use:
  - 'h2o_automl'
  - 'xgboost'
  - 'lightgbm'
  - 'catboost'
  - 'sklearn'
```

(Mirrored inside `modeling_params.models_to_use` /
`modeling_reg_params.models_to_use`.) Up to `MAX_MODELS = 10` distinct
models are fitted per `train_models` call; a missing optional library
(xgboost, lightgbm, catboost, h2o) raises an error naming the package
and telling you to remove the entry from `models_to_use`.

#### H2O AutoML

```yaml
h2o_automl_params:              # also mirrored as modeling_params.model_params.h2o_automl
  max_models: 10
  max_runtime_secs: 600
  seed: 42
  nfolds: 5
  sort_by: 'AUC'
  include_algos: ['GBM', 'XGBoost', 'DeepLearning', 'GLM', 'RandomForest']
  exclude_algos: []
  stopping_rounds: 5
  verbosity: 'info'
```

#### Modelling (classification and regression)

`modeling_params` (classification) and `modeling_reg_params`
(regression) share one shape:

```yaml
modeling_params:
  task_type: 'classification'   # 'classification' | 'regression'
  target_col: 'target'
  feature_cols: null            # null = every non-target column
  models_to_use: ['h2o_automl', 'xgboost', 'lightgbm', 'catboost', 'sklearn']
  selection_metric: 'f1_weighted'  # regression: 'rmse'
  random_state: 42
  test_size: 0.2
  sampling_params: {method: 'stratified', test_size: 0.2, stratify_col: 'target'}
  calibration: {method: 'betacal', cv: 5}   # sigmoid|isotonic|betacal|none
  model_handle: 'xgboost.XGBClassifier'     # dotted importable, or adapter name
  calibration_handle: 'sklearn.calibration.CalibratedClassifierCV'
  tuning_handle: 'sklearn.model_selection.GridSearchCV'
  loss_function: 'Logloss'                  # top-level default (see below)
  model_params:                             # per-adapter hyperparameters
    xgboost:
      model_handle: 'xgboost.XGBClassifier'
      n_estimators: 300
      max_depth: 5
      learning_rate: 0.05
      objective: 'binary:logistic'
      loss_function: 'binary:logistic'
      tree_method: 'hist'
      random_state: 42
    # lightgbm, catboost, sklearn, h2o_automl — same pattern
  bucket_segment_col: 'region'
  shap_params: {method: 'shap', n_repeats: 5, sample_rows: 100, random_state: 42}
```

The regression twin uses `task_type: 'regression'`,
`modeling_reg_params`, `XGBRegressor`/`LGBMRegressor`/`CatBoostRegressor`
/`RandomForestRegressor` handles, `selection_metric: 'rmse'`,
`calibration: {method: 'none'}`, `loss_function: 'RMSE'`, and
`sort_by: 'RMSE'` for H2O.

##### Loss functions per model

`loss_function` is a plain string. It is first normalised through the
per-adapter `LOSS_REGISTRY` in `pipelines/modeling/adapters.py`
(`get_loss()`) — aliases (case-insensitive) and native values map to
the adapter's native kwarg (`LOSS_KWARGS`: `objective` for xgboost /
lightgbm, `loss_function` for catboost / h2o_automl, `loss` for
sklearn). The top-level `loss_function` is the default for every
model; `model_params.<model>.loss_function` overrides it. XGBoost and
LightGBM also ship a native `objective` key; an explicit
`loss_function` overrides `objective`/`loss`, a recognised `objective`
is normalised, and an unrecognised native objective passes through
unchanged. No `loss_function` means estimator defaults are unchanged.
Unknown losses raise `ValueError` listing the valid losses for the
adapter.

**Exact-allowlist callables.** In addition to aliases, an exact dotted
name from the allowlist in `losses.py` (`LOSS_CALLABLES`, resolved by
`losses.get_loss_callable()`) is accepted and passed as the callable
itself. The allowlist is exactly:

- `sklearn.metrics.{mean_squared_error, mean_absolute_error,
  mean_squared_log_error, root_mean_squared_error,
  root_mean_squared_log_error, median_absolute_error,
  mean_absolute_percentage_error, log_loss, hinge_loss,
  brier_score_loss, mean_poisson_deviance, mean_gamma_deviance,
  mean_tweedie_deviance, mean_pinball_loss, zero_one_loss}` —
  e.g. `sklearn.metrics.mean_absolute_error`;
- `forecast_ml_framework.losses.huber_loss`,
  `forecast_ml_framework.losses.quantile_loss`,
  `forecast_ml_framework.losses.pinball_loss`.

Any other dotted name — including an arbitrary callable from an
allowlisted top-level module such as `forecast_ml_framework.cli` or
`sklearn.metrics.accuracy_score` — is rejected with `ValueError`
listing the valid callables. Config strings cannot register a new
loss: trusted code may add one only by calling
`forecast_ml_framework.losses.register_loss('forecast_ml_framework.losses.<name>', func)`
in code (name must be under that prefix, be callable, and not already
registered); that is a code-only extension point, not a config edit.

Valid built-in examples per adapter:

| Adapter | Native kwarg | Examples |
|---|---|---|
| `xgboost` | `objective` | `binary:logistic`, `binary:logitraw`, `reg:squarederror`, `reg:absoluteerror`, `count:poisson` |
| `lightgbm` | `objective` | `binary`, `multiclass`, `regression`, `regression_l1`, `huber` |
| `catboost` | `loss_function` | `Logloss`, `CrossEntropy`, `RMSE`, `MAE`, `Poisson` |
| `sklearn` | `loss` | `log_loss`, `squared_error`, `absolute_error`, `huber` |
| `h2o_automl` | `loss_function` | `cross_entropy`, `binomial_exp_deviance`, `mse`, `mae`, `huber` |

Aliases are accepted case-insensitively (`logloss`, `squared_error`/`mse`/`rmse`,
`mae`, `poisson`, …) and native values map to themselves. The full
registry is `LOSS_REGISTRY` in `pipelines/modeling/adapters.py` and
the callable allowlist is `LOSS_CALLABLES` in `losses.py`.

```yaml
# Change the optimisation objective for one model only:
modeling_params.model_params.xgboost.loss_function: 'binary:logitraw'
# Or use an allowlisted callable (exact dotted name only):
modeling_params.model_params.sklearn.loss_function: 'sklearn.metrics.mean_absolute_error'
# Framework custom loss (also exact-allowlisted):
modeling_params.model_params.sklearn.loss_function: 'forecast_ml_framework.losses.huber_loss'
```

#### Per-model params (multi-model fallback)

```yaml
model_params_by_name:           # used when models_to_use has >1 entry
  xgboost: {model_handle: 'xgboost.XGBClassifier', n_estimators: 300, ...}
  lightgbm: {model_handle: 'lightgbm.LGBMClassifier', ...}
  catboost: {model_handle: 'catboost.CatBoostClassifier', ...}
  sklearn: {model_handle: 'sklearn.ensemble.RandomForestClassifier', ...}
```

#### Calibration

```yaml
calibration_params:             # standalone equivalent of modeling_params.calibration
  enabled: true
  method: 'betacal'             # 'sigmoid' | 'isotonic' | 'betacal' | 'none'
  cv: 5
  handle: 'betacal.BetaCalibration'
  params: {parameters: 'abm'}
```

#### Cross-validation

```yaml
cv_params:
  n_splits: 5
  shuffle: true
  random_state: 42
  scoring: 'roc_auc'
  strategy: 'stratified_kfold'
```

#### Spark

```yaml
spark_params:
  master: 'local[*]'
  app_name: 'forecast-ml-framework'
  spark:
    spark.sql.shuffle.partitions: 200
    spark.driver.memory: '4g'
  jar_packages: []
```

Spark is **opt-in**: `SparkSessionFactory.get_spark()` returns `None`
unless the config explicitly enables it (`enabled: true` / truthy
`spark_enabled` / `use_spark`, or a non-empty Spark section carrying
keys), and returns `None` gracefully when `pyspark` is missing, so
local pandas execution is never broken by missing Spark.

#### Artifact store

```yaml
artifact_store_params:
  type: 'local'                 # 'local' | 'gcs' | 's3' (overridden per env)
  base_path: 'artifacts'
  bucket: ''                    # gcp: gs://forecast-ml-artifacts | aws: s3://forecast-ml-artifacts
  versioning: true
```

#### MLflow

```yaml
mlflow_params:
  tracking_uri: 'mlruns'
  experiment_name: 'forecast-ml-framework'
  registry_uri: ''              # empty = no registry; set MODEL_REGISTRY_URI for registry loading
  log_params: true
  log_metrics: true
  log_artifacts: true
```

#### DVC

```yaml
dvc_params:
  enabled: true
  remote: 'local'
  remote_url: ''
  push_on_run: false
```

#### SHAP

```yaml
shap_params:                    # standalone (pipeline-level: modeling_params.shap_params)
  enabled: true
  explainer: 'TreeExplainer'
  background_size: 100
  top_k: 20
  output_dir: 'artifacts/shap'
```

#### Charts

```yaml
chart_params:
  enabled: true
  output_dir: 'artifacts/charts'
  figure_size: [10, 6]
  dpi: 150
  charts: ['confusion_matrix', 'roc_curve', 'feature_importance', 'calibration_curve']
```

#### Validation

```yaml
validation_params:              # contract in data_prep.nodes.validate_data
  fail_fast: false              # true → missing required cols raise ValueError
  required_cols: []
  max_null_ratio: 0.5
```

### `catalog.yml` — Kedro Data Catalog

Base entries (env overlays re-declare the same names with env-specific
filepaths; Kedro merges by dataset name):

| Dataset | Type | Notes |
|---|---|---|
| `raw_data` | `pandas.CSVDataset` | Raw CSV input |
| `raw_data_parquet` | `pandas.ParquetDataset` | Large-file alternative (same key name) |
| `raw_data_regression` | `pandas.CSVDataset` | Regression raw CSV |
| `processed_data`, `train_data`, `test_data` | `pandas.ParquetDataset` | Split/processed data |
| `selected_features` | `MemoryDataset` | Persisted copy as `JSONDataset` (commented) |
| `model`, `calibration_model`, `h2o_automl_model`, `xgboost_model`, `lightgbm_model`, `catboost_model`, `sklearn_model` | `PickleDataset` | Fitted models (`artifacts/models/*.pkl`) |
| `model_memory`, `calibration_model_memory` | `MemoryDataset` | In-memory pipeline wiring |
| `metrics` | `JSONDataset` | Evaluation metrics (`artifacts/metrics/metrics.json`) |
| `metrics_csv` | `pandas.CSVDataset` | Metrics CSV |
| `predictions` | `pandas.ParquetDataset` | Predictions (`artifacts/predictions/`) |
| `predictions_csv` | `pandas.CSVDataset` | Predictions CSV |
| `shap_values` | `pandas.ParquetDataset` | SHAP values (`artifacts/shap/`) |
| `shap_summary` | `JSONDataset` | SHAP summary |
| `charts` | `MemoryDataset` | Persisted chart files use `XLSXDataset`/`ImageDataset` per impl |
| `scoring_input` / `scoring_output` | `pandas.ParquetDataset` | Batch scoring input / legacy writable alias (pipeline output is `predictions`) |
| `monitoring_report` / `alerts` | `pandas.ParquetDataset` | Phase 3: drift report + severity-graded alerts (`artifacts/monitoring/`) |
| `cost_log` | `pandas.ParquetDataset` | Phase 3: per-run cost rows (`artifacts/cost/`) |
| `lineage` / `audit_log` | `pandas.ParquetDataset` | Phase 3: governance lineage + audit trail (`artifacts/governance/`) |
| `registry_model` | `PickleDataset` | File-backed handle for the MLflow registry artifact (Phase 2) |

SQL placeholders are commented in the file (`sql.SQLTableDataset` for
BigQuery/Athena). **No credentials in this file** — SQL/cloud credentials
come from the environment (`GOOGLE_APPLICATION_CREDENTIALS`, `AWS_PROFILE`,
etc.). Two extra datasets ship in code
(`core/datasets.py`): `SQLQueryDataSet` (read-only SQL → pandas) and
`SparkSQLDataSet` (Spark SQL stub for BigQuery/Athena via Spark JDBC).

#### The `anchors` / `*_params` pattern

Catalog entries delegate their arguments to parameter sections so env
overlays can re-target storage without touching base:

```yaml
# pattern used across the catalog:
train_data:
  type: 'pandas.ParquetDataset'
  filepath: 'data/processed/train_data.parquet'   # overridden per env
```

`params:<section>` inputs in the pipelines resolve to nested sections of
`parameters.yml` (e.g. `params:data_prep_params`,
`params:modeling_params.task_type`), and flat runtime keys are referenced
as `${params:<key>}` from the catalog. Define a new section in
`parameters.yml`, reference it as `params:<your_section>`, and add its
pydantic model to `core/config_schema.py` (with `extra='allow'`, unknown
keys pass through, so shipped conf always validates).

### Env overlays: `local` / `gcp` / `aws`

| Env | Ships | Overrides |
|---|---|---|
| `local` | `catalog.yml`, `parameters.yml` | Local `./data`, `./artifacts` filepaths; `artifact_store_params.type: local`, local MLflow tracking URI, DVC off, `local[*]` Spark |
| `gcp` | `catalog.yml` | `gs://forecast-ml-artifacts/...` filepaths (GCS + BigQuery); credentials via `GOOGLE_APPLICATION_CREDENTIALS` |
| `aws` | `catalog.yml` | `s3://forecast-ml-artifacts/...` filepaths (S3 + Athena); credentials via `AWS_PROFILE` / AWS env vars |

Only filepaths/URIs (and, for `local`, the minimal parameter overrides)
are overridden; everything else is inherited from `base/`.

### OmegaConf interpolations

- `target_col: ${variables.target_col}` (in `global_params`) pulls the
  target name from `variables.yml`.
- `stratify_col: ${..global_params.target_col}` (relative reference)
  re-uses the global target as the stratification column.
- `score_model_id: ${.global_params.score_model_id}`,
  `run_date: ${.global_params.run_date}` (and `train_run_dt` /
  `score_run_dt`) fan the global values out to flat runtime keys.
- `bucket_segment_col: ${.cat_cols[0]}` (in `variables.yml`) defaults
  the bucket segment to the first categorical column.

`load_config()` calls `OmegaConf.resolve()` before returning, so all
interpolations arrive resolved. Kedro's `OmegaConfigLoader` resolves the
same syntax at runtime.

### Pydantic validation

`core/config_schema.py` defines one model per section
(`VariablesConfig`, `GlobalParams`, `DataParams`, `SamplingParams`,
`PandasConversionParams`, `EvaluationParams`, `TrainTestSplitParams`,
`FeatureSelectionParams`, `ModelingParams`, `ModelingRegParams`,
`CalibrationParams`, `CvParams`, `SparkParams`, `ArtifactStoreParams`,
`MlflowParams`, `DvcParams`, `ShapParams`, `ChartParams`,
`ValidationParams`, `EdaParams`, and the top-level `ParametersConfig`), all with
`extra='allow'` so unknown sections/keys pass through and shipped conf
always validates. Strictness (Phase 5) is typed fields + defaults +
`ValidationError` on wrong types (e.g. `EdaParams` rejects
non-numeric `missing_threshold`), not `extra='forbid'`. `ConfigLoader` (core/config_loader.py) normalises
dict / OmegaConf / pydantic inputs to plain dicts on construction.

### How to add a dataset / model / loss

**Add a dataset (config only).** Add an entry to
`conf/base/catalog.yml`, point `data_params.train_data`/`test_data` at
it, and list its columns in `conf/base/variables.yml`. For SQL: add a
`SQLQueryDataSet` entry with a `connection` (URL/engine/config dict) and
a single-statement `query`; credentials stay in the environment. For
Spark SQL: `SparkSQLDataSet` with a live session or JDBC config.

**Add a model (config only).** Add the name to `models_to_use` (and
`modeling_params.models_to_use`), then add
`model_params.<name>` with `model_handle` (dotted importable) plus
hyperparameters. A dotted name in `models_to_use` itself also works —
it becomes a `HandleAdapter`. Loss: set
`model_params.<name>.loss_function` (plain name, alias, or dotted
callable).

**Add a loss function.** Either use an existing alias/native value from
`LOSS_REGISTRY` or an exact dotted name from the allowlist
`LOSS_CALLABLES` in `losses.py` (see the
[loss table](#loss-functions-per-model)) — e.g.
`forecast_ml_framework.losses.huber_loss` or
`sklearn.metrics.mean_absolute_error`. Any other dotted name is
rejected. To add a genuinely new callable, trusted code calls
`forecast_ml_framework.losses.register_loss('forecast_ml_framework.losses.<name>', func)`
in code (code-only extension; config strings cannot register a loss).

## Feature selection + business/override rule

`ConfigLoader.merged_feature_list(selected)` implements one rule, in
three steps:

1. **Override wins alone.** A non-empty `override_features` (aliases:
   `feature_override`, `features_override`, `feature_list_override`) is
   returned deduplicated, ignoring everything else. A string override is
   one feature name, not a sequence of characters. Override length is
   capped at 10,000 entries (`ValueError` beyond that, or when the
   override is not a list/tuple).
2. **Otherwise merge business features.** The result is
   `selected + business_features` deduplicated in first-seen order
   (aliases: `business_feature_list`, `business_cols`,
   `business_columns`, `additional_features`; `variables.yml`'s
   `business_relevant_features` is translated to `business_features` by
   `select_features()` because core only knows `business_features`).
3. **Subtract exclusions.** `exclude_features` (aliases:
   `excluded_features`, `exclude`, `exclude_columns`,
   `dropped_features`) are removed last.

Existence of named columns is the caller's responsibility: non-existent
names pass through the merge unchanged; `select_features()` records them
as warnings (`'selected feature not in data: ...'`) and filters the final
list to columns actually present in the frame.

### Feature-selection decision (Mermaid)

```mermaid
flowchart TD
  S["selected features<br/>variance / missing / correlation<br/>kbest / l1 / tree"] --> Q{"override_features<br/>non-empty?"}
  Q -- "yes" --> O["return override<br/>deduplicated - wins alone"]
  Q -- "no" --> M["selected + business_features<br/>deduplicated, first-seen order"]
  M --> X["minus exclude_features"]
  X --> OUT["merged_feature_list()<br/>select_features keeps<br/>columns present in frame"]
```

```yaml
# variables.yml — business features always kept:
business_relevant_features: ['tenure_days']
override_features: []           # non-empty → wins alone (see caveat)
exclude_features: []            # always excluded
# feature_selection_params.business_features etc. — same meaning
```

> **Caveat.** `override_features` winning *alone* (dropping business
> features and selected features) is the documented behaviour of
> `merged_feature_list` (`core/config_loader.py`). If that rule ever
> changes to union-with-override, this section must be updated in the
> same change.

## Metrics + charts catalog (v2)

`pipelines/modeling/metrics.py:METRIC_REGISTRY` maps **144 metric names**
(`CLASSIFICATION_METRICS` + `REGRESSION_METRICS`,
`metrics.py:1247`) to callables `fn(y_true, y_pred, y_proba) -> {name:
value}`; `compute_metrics(y_true, y_pred, y_proba, task_type,
selected=...)` computes only the selected ones, sharing intermediates
(confusion matrix, MAE, ROC-AUC, …) once and skipping unselected
families. `evaluate.py:evaluate_model()` returns `{'global': ...,
'by_bucket': ..., 'by_decile': ..., 'metrics_calculated': [...],
'available_metrics': 144}` where `by_bucket` groups by
`bucket_segment_col` and `by_decile` groups by prediction decile.
Metrics never crash the pipeline: individual failures become `nan`.
`LOWER_BETTER` is used to infer `selection_metric_higher_better` when
picking the best model.

### Metrics selection (`metrics_to_calculate`)

```yaml
# evaluation.yml — shipped default: curated 16-name subset
# (11 classification + 5 regression), NOT all metrics:
evaluation_params:
  metrics_to_calculate: ['accuracy', 'balanced_accuracy',
    'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc',
    'pr_auc', 'log_loss', 'brier_score', 'cohen_kappa',
    'matthews_corrcoef', 'mae', 'rmse', 'r2', 'mape', 'smape']
  # metrics_to_calculate: 'all'          # every registry metric (~144, slower/noisier)
  # metrics_to_calculate: ['f1_weighted', 'roc_auc', 'mae']  # custom list only
```

Semantics (`metrics.py:resolve_selected_metrics`, `evaluate.py:106`):
absent → `DEFAULT_METRICS` (task-filtered curated subset, 11
classification / 5 regression in code); `'all'` → every registry metric
for the task; a list → only those names (unknown name → `ValueError`
listing valid names). **Wiring caveat (Known Issue #14):**
`evaluate_model_node` receives `params:modeling_params`, not
`evaluation_params` — so to select metrics for a Kedro modeling run, set
`metrics` (or `metrics_to_calculate`) inside `modeling_params`; the
`evaluation_params.metrics_to_calculate` in `evaluation.yml` documents
the selection but is **not fed into the node** by
`pipelines/modeling/pipeline.py`. `y_proba` is required for
classification metrics that need probabilities.

### Classification / regression highlights

Classification includes `accuracy`, `balanced_accuracy`,
`precision/recall/f1_{macro,micro,weighted}`, `per_class_f1`,
`roc_auc`, `pr_auc`, `log_loss`, `brier`, `kappa`, `mcc`,
`sensitivity`, `specificity`, `ks`, `confusion_matrix`, `ece`, plus
registry-only names (`f_beta`, `ndcg`, `kl_divergence`, `top_k_accuracy_*`,
`youden_j`, …). Regression includes `mae`, `medae`, `rmse`, `rmsle`,
`mse`, `r2`, `adj_r2`, `mape`, `smape`, `explained_variance`,
`max_error`, `bias`, plus `huber_loss`, `mase`, `pinball_*`,
`quantile_loss_*`, `pearson`/`spearman`, … Full lists:
`CLASSIFICATION_METRICS` / `REGRESSION_METRICS` (`metrics.py:1153`).

### Charts (groups I–VII)

`pipelines/modeling/charts.py:make_charts()` returns `{name: matplotlib
Figure}`; charts whose data are insufficient are skipped gracefully.
**Every figure is closed before returning** — save images inside the
call; callers receive closed figures. Charts are built with the `Agg`
backend. Phase 3 added groups I–VII on top of the original evaluation
charts:

| Group | Entry point | Charts |
|---|---|---|
| Original classification | `make_charts` | `roc`, `pr`, `calibration_curve`, `threshold_curve`, `lift_gain`, `ks_plot`, `prob_distribution`, `cost_curve` (stub), `confusion_heatmap`, `per_class_bars` |
| Original regression | `make_charts` | `pred_vs_actual`, `residuals_vs_fitted`, `residuals_vs_actual`, `residual_hist_qq` (hist + QQ), `scale_location`, `error_vs_actual`, `error_by_bucket`, `segment_curves`, `feature_vs_residual` |
| I. Data understanding | `make_data_charts` | target/feature distributions, correlation heatmap, pair plot, missing heatmap, outlier box, PCA scatter |
| II. Classification | `make_charts` | DET curve, confusion wheel, top-k accuracy, balanced accuracy, PR gain curve |
| III. Regression | `make_charts` | residual autocorrelation, Cook's distance, REC curve, prediction-interval coverage, residuals over time, forecast vs actual |
| IV. Training | `make_training_charts` | learning curve, validation curve, loss vs epoch, CV scores box, overfitting gap, bias-variance |
| V. Interpretability | `make_charts` / interpretability pipeline | SHAP dependence/waterfall/force, PDP 1D, ICE, LIME bar, decision boundary, tree plot |
| VI. Comparison | `make_comparison_charts` | model comparison box, Bland-Altman, ensemble diversity, ROC/PR overlays, error correlation heatmap |
| VII. Specialized | `make_charts` | quantile coverage, prediction interval vs actual |

For the 27-chart interpretability/diagnostics catalog, see
[Interpretability pipeline](#interpretability-pipeline--pipelinesinterpretability).

## Metrics Parity (Phase 6, parity)

Parity helpers live in `pipelines/modeling/ntile.py`,
`pipelines/modeling/lift_table.py`, `pipelines/modeling/threshold.py`,
and `pipelines/modeling/validation.py`, wired into
`pipelines/modeling/pipeline.py` (`build_lift_table_node` →
`lift_table`, `build_promotion_node` → `promotion`) and
`pipelines/modeling/evaluate.py:evaluate_model`.

### `eval_metric` grammar

Set top-level `eval_metric` in `modeling_params` /
`modeling_reg_params` (commented examples in
`conf/base/parameters/modeling.yml:86-91,174-178`; distinct from the
per-model `model_params.xgboost.eval_metric`). Parsed by
`ntile.parse_eval_metric(metric_name, task)`
(`ntile.py:89-164`) into `(metric, kind, n)`:

```yaml
modeling_params:            # classification
  eval_metric: 'LIFT_DECILE_10'   # or 'LIFT_CENTILE_100', 'AUC_DECILE_10'
modeling_reg_params:        # regression
  eval_metric: 'RMSE_CENTILE_100' # or 'RMSE', 'MAE_QUINTILE_5'
```

Grammar: `<M>_<DECILE|CENTILE|QUINTILE|QUARTILE>_<n>` where `M` is a
selectable metric — classification `LIFT|PRECISION|RECALL|ACCURACY|F1|
AUC|BRIER_SCORE|BRIER` (`ntile.py:35-39`), regression
`MAE|MAPE|MSE|MSLE|RMSE|RMSLE|ADJ_R2|R2` (`ntile.py:40-44`) — plus
forecast aliases (`lift_top_decile`, `roc_auc`, `f1_weighted`,
`root_mean_squared_error`, …; `ntile.py:47-57`). Bare selectable names
use the task default kind; names containing
`decile|quintile|quartile|centile` select that kind/count
(10/5/4/100); `overall|global` selects kind `overall`, `n=0`. Defaults
when omitted/empty: classification `('lift','decile',10)`, regression
`('rmse','overall',0)`. Unknown names warn and fall back to the task
default — they never raise.

### `selected_ntile` and `by_centile`

`evaluate_model` sets `result['selected_ntile'] = None` when
`eval_metric` is absent, else `_selected_ntile(...)`
(`evaluate.py:163-168,171-223`) returns
`{'eval_metric', 'metric', 'score', 'ntile'}`. The requested 1-based
ntile `n` maps to 0-based group label `str(n-1)` (`evaluate.py:174-178`);
if that label is absent (qcut collapsed), it falls back to the largest
available label.
`kind == 'centile'` recomputes 100 groups and also sets
`result['by_centile']`; `kind == 'decile'` recomputes decile groups
with the parsed metric included even when it is not in the user's
selected list; `kind == 'overall'` reads the global score
(recomputing if needed). `score` falls back to the global metric,
then `nan` — never raises.

### Descriptive stats (regression) and confusion rates table

Descriptive stats (regression) are computed only when selected
(`metrics.py:1009-1030`): `target_sum|min|max|mean`,
`pred_sum|min|max|mean`, `total_count`, and `max_pred_error`
(signed `max(y_true - y_pred)`, not max absolute error). The
parity-shaped confusion table is `metrics.confusion_rates_table(y_true,
y_pred)` (`metrics.py:1196-1229`): two rows (`Actual Positive` /
`Actual Negative`, each with `label` and `Actual` keys) carrying
`tpr|fpr|fnr|tnr` floats; a failed/non-2×2 confusion matrix yields
`nan` rates, never a crash.

### Lift table

`lift_table.compute_lift_table(y_true, y_pred, proba, n=10,
threshold=None)` (`lift_table.py:39-184`) returns a DataFrame with
exactly the `COLUMNS` list (`lift_table.py:13-26`): `decile`,
`min_prob`, `max_prob`, `avg_prob`, `decile_count`,
`non_target_count`, `target_count`, `target_percentage`,
`non_target_percentage`, `cum_target_percentage`,
`cum_non_target_percentage`, `ks`, `lift`, `decile_disconnect_rate`,
`predicted_positive`, `predicted_negative`, `tp`, `fp`, `tn`, `fn`,
`precision`, `recall`, `customers_at_or_above_threshold`,
`true_positive_at_or_above_threshold`,
`precision_at_or_above_threshold`, `recall_at_or_above_threshold`,
`tp_above_threshold`, `fp_above_threshold`, `tn_above_threshold`,
`fn_above_threshold`, `precision_above_threshold`,
`recall_above_threshold`. Rows are deciles 1..n (1 = lowest proba)
sorted descending; `threshold` is wired, not reserved: when given
(in [0, 1], validated at `lift_table.py:84-92`), the
above-threshold columns are cumulative-from-top sums over the
subset `proba >= threshold`; when `None` (default), they are
cumulative from the top decile down over all rows
(`lift_table.py:39-55,152-184`). `pipeline.build_lift_table`
(`pipeline.py:135-168`) passes the threshold from
`modeling_params` (`lift_threshold`, alias `model_threshold`)
through to `compute_lift_table`, extracts the positive-class
proba, and returns the empty frame when `y_proba is None`.

### Threshold methods

`threshold.select_threshold(y_true, proba, method)`
(`threshold.py:49-111`) supports `pr_curve` (max F-score),
`roc_curve` (max G-mean), `mcc_curve` (max Matthews correlation,
0.01 → 0.0001 → 0.000001 step grids), and `ks_statistics` (max
TPR−FPR). Unknown methods, empty, or single-class input return
`0.5`. Results are rounded per parity: `<0.0001` → 6dp, `<0.01` → 4dp,
else 2dp (`threshold.py:16-22`).

### Promotion thresholds

`validation.should_promote(new_score, old_score, task, metric='')`
(`validation.py:30-74`) returns `(promote, deviation_pct)` with
`deviation = (new - old) / old * 100`. Classification (and
higher-better regression `r2`/`adj_r2`) promotes at
`deviation >= -5%`; lower-better regression promotes at
`deviation < +5%`. `old_score` of `0`, `None`, `nan`, or an
unparseable score never promotes (`(False, nan)`).
`pipeline.build_promotion` compares `selected_ntile.score` against
`params.old_score` and returns `{'promote', 'deviation_pct',
'new_score', 'old_score', 'eval_metric', 'metric', 'task_type'}`
(`pipeline.py:171-200`); missing `old_score`/`selected_ntile` never
promotes. `should_promote` only returns the decision — registry
promotion stays with the caller.

### Scoring ntile columns

`pipelines/scoring/nodes.py:_score_pandas_chunk` (`scoring/nodes.py:564-666`)
adds `score_decile` (10 tiles) and `score_centile` (100 tiles)
(`scoring/nodes.py:652-653`) via `_ntile_labels`
(`scoring/nodes.py:438-467`) → `ntile.assign_ntile` (`ntile.py:167-198`;
`assign_ntile` is 1-based, lowest scores in tile 1, `overall` → all
zeros). When a calibrated source is present (`_calib_values`,
`scoring/nodes.py:647`), it also adds `calib_decile` / `calib_centile`
(`scoring/nodes.py:654-656`). Classification rows also get
`high_score_ind` (`'Y'` when `proba > model_threshold` (default 0.5)
and `score_decile >= 5`, else `'N'`, `scoring/nodes.py:657-661`).

## Scaling: 100M-row guidance

The local pandas path is capped by design; the cloud path never
materialises the full frame on the driver.

| Lever | Local default | 100M-row setting |
|---|---|---|
| Pandas guard | `pandas_conversion_params.max_rows_for_pandas: 5000000` (`data.yml`); `check_pandas_limits` (`core/spark_factory.py`) raises above it, directing to Spark-native FE / distributed training | Keep the guard; route oversized frames to Spark |
| Spark scoring | `scoring_params.spark_scoring.enabled: false` (pandas, safe) | `enabled: true` — partitioned scoring via `mapPartitions` (`pipelines/scoring/`), `partition_batch_size: 10000` rows per partition sub-batch, `scoring_params.max_rows: null` (unbounded on Spark; pandas path is still capped) |
| Spark conf | `spark_params` in `platform.yml`: `spark.sql.shuffle.partitions: 1000` (~100k rows/partition at 100M rows), driver `4g` | Dynamic allocation on (min 2 / max 20 executors), AQE + coalesce on — autoscaling defaults in `spark_factory.py:_AUTOSCALING_DEFAULTS`; Dataproc: `enable_dataproc = true` + `autoscaling_enabled = true` (see `infra/README.md`) |
| Cluster submission | None — `get_spark` only builds the session config | Submit on Dataproc (`master=yarn`) / EMR yourself; **no submit automation or credentials in this repo** |

> **Known limitation (Known Issue #15).** Spark partitioned scoring
> closes over the model object — for real Spark the model must be
> **broadcast/picklable**; an unpicklable model fails on executors, not
> on the driver. Keep `batch_size` / `partition_batch_size` small enough
> for per-task memory at >500 features.

## Serving API

`serving/app.py:create_app(model_loader=None, registry_uri=None,
model_name=None, alias=None)` builds the FastAPI app
(`title='forecast-ml serving'`). `model_loader` is a callable returning
a model or the model itself; wire it to
`serving/model_loader.py:FileModelLoader(artifact_path)` for file-based
loading, or inject a stub in tests. Alternatively pass registry config
(`registry_uri`/`model_name`/`alias`, or env `MODEL_REGISTRY_URI` /
`MODEL_NAME` / `MODEL_ALIAS`) — `create_app` then builds a
`RegistryModelLoader` (MLflow registry, `models/<name>@<alias>`) via
`registry/model_registry.py`. `create_app_from_registry(...)` is the
registry-only shortcut. `RegistryModelLoader.load()` caches the model
and raises `RuntimeError` (no URI leaked) when the registry model is
not found. A/B + shadow routing lives in `serving/ab_testing.py`
(`ABRouter`) — see [Phase 3 pipelines](#phase-3-pipelines).

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
# → {"status": "ok", "model_loaded": true}
```

`HealthResponse` carries `status` and `model_loaded`
(`serving/app.py:46-50`). The first probe loads the model: on success
`status: ok, model_loaded: true`; on any load failure `status:
degraded, model_loaded: false` (`serving/app.py:431-444`). The result
is cached on the app instance. With no model configured
(`DegradedLoader` — no injected loader, no `MODEL_ARTIFACT_PATH`,
no `MODEL_NAME`), `/predict` returns 503 `model not configured` and
`/health` reports `degraded`.

`/health` is always public (no API key required), for probes and
load-balancer health checks.

### `POST /predict`

Request schema: `{"features": {name: number, ...}}` (single record) or
`{"records": [{...}, {...}]}` (batch). At least one must be present.
Validation is strict: every feature must be a finite number (booleans,
nested values, non-finite floats, missing features → HTTP 422 naming
the offending field); payload caps (`MAX_RECORDS = 10000`,
`MAX_FEATURE_KEYS = 1000`) → HTTP 413; request bodies are capped at
**10 MB** by a streaming byte counter — chunked bodies without
`Content-Length` are rejected mid-stream with HTTP 413 as soon as the
running total passes `MAX_BODY_BYTES = 10 * 1024 * 1024`, not only by
header check; model load failures → HTTP 500
without leaking the server filesystem path. Feature-name contract
(Phase 4): when the model carries `feature_names_in_` (e.g. a fitted
sklearn estimator), that list is the required feature set and order;
otherwise the first record's keys are used. Unknown features → 422.

**Typed errors (`model_loader.py:69-90`).** Typed loader errors map to:
`ModelNotConfiguredError` → 503 `model not configured`; missing
artifact/version → `ModelNotFoundError` → 503 `model artifact not
found`; malformed artifact or registry flavour mismatch →
`ModelCorruptError` → 500 `model failed to load`. Validation
`ValueError` → 422 naming the field; model execution errors → generic
500 `prediction failed`. Messages never contain the server filesystem
path or registry URI.

**Authentication (optional).** Auth is off by default. When the
`FORECAST_API_KEY` environment variable is set, every path except
`/health` requires the matching key, else the app returns
`401 {"detail": "unauthorized"}`:

```bash
export FORECAST_API_KEY='...'   # set before starting uvicorn
curl http://127.0.0.1:8000/health          # public, no key
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' -H 'X-API-Key: ...' -d '...'
# or: -H 'Authorization: Bearer ...'
```

The key is accepted via `X-API-Key` or `Authorization: Bearer ...`
and compared in constant time. A missing or wrong key gets 401.

**Security headers.** Every response (including 401/413 errors)
carries `x-content-type-options: nosniff`,
`x-frame-options: DENY`, `referrer-policy: no-referrer`,
`cache-control: no-store`, a restrictive
`content-security-policy` (`default-src 'none'; frame-ancestors 'none'`),
and `x-permitted-cross-domain-policies: none`.

Registry note (Phase 4 final, verified in
`registry/model_registry.py`): registration uses
`mlflow.sklearn.log_model` with `registered_model_name` + alias
promotion — sklearn flavor only, no pyfunc flavor, no model
signature/input-example wiring. Loading defaults to
`mlflow.sklearn.load_model`; the pyfunc fallback is opt-in only via
`RegistryModelLoader(..., allow_pyfunc=True)` (a startup warning is
logged; pyfunc executes arbitrary code shipped with the model, so
only enable it for fully trusted registries). No Docker is provided.

```bash
# Single record
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": {"f1": 0.1, "f2": 0.2, "f3": 0.3, "f4": 0.4}}'

# Batch
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"records": [{"f1": 0.1, "f2": 0.2, "f3": 0.3, "f4": 0.4},
                     {"f1": 0.5, "f2": 0.6, "f3": 0.7, "f4": 0.8}]}'
# → {"predictions": [0, 1]}
```

```python
import httpx

r = httpx.post(
    'http://127.0.0.1:8000/predict',
    json={'features': {'f1': 0.1, 'f2': 0.2, 'f3': 0.3, 'f4': 0.4}},
)
print(r.json())  # {'predictions': [...]}
```

> **Note.** The `forecast-ml serve` CLI subcommand delegates to
> `uvicorn forecast_ml_framework.serving.app:app --host <host> --port
> <port>` with the artifact passed via `MODEL_ARTIFACT_PATH`
> (`cli.py:90-95,188-190`, verified Phase 5). The programmatic path
> `create_app(FileModelLoader(path))` or `create_app_from_registry(...)`
> also works under uvicorn.

## Known Issues & Fixes

Each issue: symptom → cause → fix/workaround **now** in the code.
Status column: **Fixed** = verified in current code; **Workaround** =
live with it as documented.

| # | Issue | Status | Symptom / cause | Fix / workaround now |
|---|---|---|---|---|
| 1 | Pickle deserialization trust boundary | **Fixed** | `pickle.load` on an artifact executes `__reduce__` code | Restricted unpickler `restricted_loads()` / `_RestrictedUnpickler` (`core/artifact_store.py`) allowlists only sklearn/numpy/pandas + safe builtin types; anything else raises `ValueError`. `FileModelLoader` (`serving/model_loader.py`) uses it. Artifacts are trusted only when produced by this framework; the unpickler is defence in depth |
| 2 | Path traversal jail | **Fixed** | A key like `../../etc/passwd` escapes the store root | `LocalArtifactStore._full_path()` resolves and jail-checks every path against the base; cloud keys validated by `_validate_cloud_key()` (no absolute keys, no `..`). Absolute paths and escaping keys raise `ValueError` |
| 3 | SQL config-trust | **Fixed (documented)** | Arbitrary SQL string → arbitrary SQL execution | `SQLQueryDataSet` (`core/datasets.py`) validates intent: single statement, `SELECT`/`WITH` only (`_validate_read_only_query`). Documented trust boundary: connection and query come from trusted config (catalog/params), **never** untrusted user input — this is intent-checking, not a SQL sandbox. Only trusted queries may be used |
| 4 | Matplotlib figure lifecycle | **Fixed** | Unclosed figures leak / "too many open figures" in long runs | `make_charts()` (`pipelines/modeling/charts.py`) closes figures on failure (`_safe`) and closes **every** figure in a `finally` block before returning. Callers must save images inside the call; figures are returned closed |
| 5 | Env-dependent tests | **Fixed** | Serving tests fail without `httpx`; Spark/cloud tests fail without services | `httpx` is a dev dependency (`[dependency-groups] dev` in `pyproject.toml`) so `fastapi.testclient` works; env-dependent behaviour is mocked/skipped (Spark returns `None` when disabled/missing; cloud SDKs imported lazily). Run `PYTHONPATH=src python -m pytest tests/ -q` |
| 6 | Python 3.14 lib compatibility | **Fixed (spike)** | Third-party libs may not import on 3.14 | Full spike below: all 17 libraries OK, nothing moved to `extras-314-failures`. Caveats: MLflow prints an agent-hint banner on import — silence with `MLFLOW_DISABLE_AGENT_HINT=1`. PySpark 4.3 on pandas >= 3 emits a `FutureWarning` — expected and harmless |
| 7 | Config drift | **Fixed** | `parameters.yml` shapes diverged from code contracts | Config aligned to code; bucket-file headers state the code contract per section (`sampling_params {method,test_size,stratify_col}`, `validation_params {fail_fast,required_cols,max_null_ratio}`, `modeling_params`, `feature_selection_params` flat keys, etc.) and `config_schema.py` models match |
| 8 | Testpaths | **Fixed** | Bare `pytest` discovered nothing: `testpaths = ['src']` while tests lived in `tests/` | `pyproject.toml` now sets `testpaths = ['tests']` + `pythonpath = ['src']`, so bare `pytest` works; the explicit `PYTHONPATH=src python -m pytest tests/ -q` form still works |
| 9 | H2O `predict_proba` | **Fixed** | H2O `predict()` returns labels, not probabilities; substituting them as proba silently corrupts metrics | `H2OAutoMLAdapter.predict_proba()` (`pipelines/modeling/adapters.py`) raises `NotImplementedError` when the fitted model has no callable `predict_proba`; labels are never used as a proba substitute. Use `predict()` for labels |
| 10 | Params mutation | **Fixed** | Mutating shared param dicts across models in the zoo | Adapters never mutate input params: `_loss_kwargs()` / `_estimator_kwargs()` build fresh kwargs, and `train_models()` copies each model's params (`mp = dict(source)`). `ConfigLoader` normalises inputs to plain dicts on construction |
| 11 | Console script name | **Fixed** | Old scaffold README invoked a non-existent `forecastin-ml` script | Only `forecast-ml` is declared (`pyproject.toml`); `__main__.py` delegates to `cli.main`. Use `uv run forecast-ml ...` |
| 12 | Phase 3 pipelines not registered | **Fixed (Phase 4)** | `kedro run --pipeline eda|monitoring|retraining|cost_tracking|interpretability` failed — "pipeline not found" | `pipeline_registry._PIPELINE_MODULES` (`pipeline_registry.py:19`) now maps all 12 display names across 8 modules; verified by `tests/test_pipeline_registry.py` (12 keys) |
| 13 | `parameters.yml` split migration | **Workaround** | Code/scripts reading `conf/base/parameters.yml` break — the file no longer exists | Edit the bucket that owns your section under `conf/base/parameters/` (see [Parameters split](#parameters-split-7-buckets)); Kedro/OmegaConf merges all `*.yml` there, so `params:<section>` references are unchanged. `conf/local/parameters.yml` still exists |
| 14 | Metrics default-subset semantics | **Workaround (documented)** | Setting only `evaluation_params.metrics_to_calculate` in `evaluation.yml` has no effect on a Kedro `modeling` run — `evaluate_model_node` receives `params:modeling_params`, not `evaluation_params` (`pipelines/modeling/pipeline.py:151`) | Put `metrics`/`metrics_to_calculate` inside `modeling_params` for Kedro runs (absent → curated `DEFAULT_METRICS` subset; `'all'` → all 144; list → only those). See [Metrics selection](#metrics-selection-metrics_to_calculate) |
| 15 | Spark scoring model broadcast | **Workaround** | Partitioned Spark scoring fails on executors with a pickling error for an unpicklable model object | The partition function closes over / broadcasts the model (`pipelines/scoring/pipeline.py:82`) — use a picklable sklearn/XGBoost/LightGBM/CatBoost estimator; keep batch sizes bounded |
| 16 | `terraform validate` not run locally | **Workaround** | `infra/` Terraform was not validated with a local `terraform validate` in this environment | Before applying, run `terraform init && terraform validate` inside `infra/gcp` and `infra/aws` (see `infra/README.md` Validation); review `terraform plan` before `apply` |
| 17 | MLflow serial marker | **Workaround** | Running the real-MLflow roundtrip test in parallel corrupts global mlflow state (tracking/registry URIs) | The test carries `@pytest.mark.serial` + a state-reset fixture (`tests/test_registry.py:449`) — run the full suite serially; do not parallelise with `pytest-xdist -n auto` |

### Known-issues audit — 2026-11-06 (post-Phase 6, 1160 tests)

Each item below was re-verified against current code on disk in this
docs pass (File:Line cites the code that proves it). No code was
changed. Items #3 and #16 already appear above; they are re-verified
here as #20 and #25 so the audit list is complete in one place.

| # | Issue | Symptom | Impact | Workaround / config |
|---|---|---|---|---|
| 18 | H2O pickle serving rejected (no cluster) | Loading an H2O AutoML pickle via `ModelAdapter.load()` / `FileModelLoader` raises `ValueError` ("Forbidden pickle global") — H2O globals are absent from the exact allowlist (`core/artifact_store.py:309-323` has xgboost/lightgbm/catboost only, no `h2o` entries; `pipelines/modeling/adapters.py:403-416` states zoo pickles referencing non-allowlisted globals are rejected by design). Training is in-process only: `H2OAutoMLAdapter.fit` calls `h2o.init(disable_server=True)` (`pipelines/modeling/adapters.py:544`); there is no H2O cluster connect/load path (`h2o.connect` / `h2o.import_model` do not appear in `src/`) | H2O models train in the modeling pipeline but cannot be re-loaded for serving/scoring in this repo; serving an H2O model fails at load (`ModelCorruptError` via `serving/model_loader.py:141`) | Serve sklearn/XGBoost/LightGBM/CatBoost pickles (allowlisted), or export the H2O model to MOJO/POJO outside this repo. Widening the allowlist with H2O globals requires a reviewed change — `pipelines/modeling/adapters.py:414-416` forbids widening it unilaterally |
| 19 | pyfunc opt-in, trusted registry only | `RegistryModelLoader` loads the sklearn flavour only by default; a flavour mismatch raises `ModelCorruptError` instead of falling back (`serving/model_loader.py:250-255`). The pyfunc fallback executes arbitrary code shipped with the model | A non-sklearn-flavour model cannot serve by default; enabling pyfunc grants code execution to whoever controls the registry | Keep `allow_pyfunc=False` (default, `serving/model_loader.py:180`). Only pass `allow_pyfunc=True` for a fully trusted registry — a startup warning is logged (`serving/model_loader.py:190-194`). Trust boundary: `serving/model_loader.py:10-18` |
| 20 | SQL trusted-config, not a sandbox (re-verifies #3) | `SQLQueryDataSet` / `SparkSQLDataSet` accept only a single `SELECT`/`WITH` statement with denylisted keywords (`core/datasets.py:71-119`), but a `SELECT` against any readable table still executes | Any trusted-config query can read whatever the DB account can read; this does not make untrusted SQL safe | Only put trusted catalog/params queries in config, never user input; use a read-only DB account (`core/datasets.py:87-92,125-131`) |
| 21 | No rate limiting | `serving/app.py` caps payload size only (`MAX_RECORDS=10000`, `MAX_BODY_BYTES=10MB`, `serving/app.py:39-41`; `_check_caps`, `serving/app.py:96-111`) and offers optional API-key auth (`FORECAST_API_KEY`, `serving/app.py:11-18`). A grep of `serving/` finds no rate-limiter, throttle, or 429 handling | An unauthenticated or authenticated client can call `/predict` as fast as the host allows; no per-client quota | Put the app behind a gateway/proxy with rate limiting (e.g. nginx, ALB, API Gateway), and set `FORECAST_API_KEY` in production |
| 22 | DNS rebinding: literal-IP check only | `_validate_uri_host` rejects non-public **literal** IPs via `ipaddress.ip_address(host)` (`registry/model_registry.py:140-155`) and blocks metadata/encoded forms (`registry/model_registry.py:120-137`), but a hostname that is not a literal IP passes without any DNS resolution — the check never resolves the name | A hostname (e.g. attacker-controlled DNS) that resolves to a private/metadata IP at connect time is not caught by this validator | Use an allowlisted registry hostname you control, resolve and re-check the IP before connecting, or egress through a proxy that validates resolved addresses |
| 23 | Pickle native-booster trust | The restricted unpickler allowlists native `Booster` constructors for XGBoost/LightGBM (`core/artifact_store.py:315-320`) and CatBoost wrappers (`core/artifact_store.py:321-322`), exact-match only (`core/artifact_store.py:326-349`). Unpickling a `Booster` invokes native code; `restricted_loads` even notes `Booster(model_file=...)` errors may carry filesystem paths (`core/artifact_store.py:374-377`) | A malicious pickle that reaches an allowlisted `Booster` global runs native deserialization code — the allowlist narrows, but does not eliminate, native-code risk | Load only artifacts produced by this framework from a fully trusted registry/store (`serving/model_loader.py:10-25`, `core/artifact_store.py:352-358`); treat any other pickle source as untrusted |
| 24 | Dataproc / EMR stubs off by default | GCP Dataproc defaults `enable_dataproc = false`, `count = 0` (`infra/gcp/variables.tf:47-51`, `infra/gcp/main.tf:88`); AWS EMR defaults `enable_emr = false`, `count = 0` (`infra/aws/variables.tf:31-35`, `infra/aws/main.tf:83-85`). EMR autoscaling and the EventBridge scheduler ship commented-out | No cloud Spark compute exists until explicitly enabled; 100M-row cloud runs need manual provisioning | Set `-var="enable_dataproc=true"` (+ `autoscaling_enabled=true`) or `-var="enable_emr=true"` (+ `emr_subnet_id`), plan/apply, and destroy when idle (see `infra/README.md`) |
| 25 | `terraform validate` not run locally (re-verifies #16) | Local checks run only `terraform fmt -check` when terraform is present (`ci/run_checks.sh:37-42`); no `terraform validate` / `plan` step runs in this environment | Terraform syntax/plan errors surface only at first real apply | Before applying, run `terraform init && terraform validate` in `infra/gcp` and `infra/aws`, then review `terraform plan` before `apply` |
| 26 | System-Anaconda skips | The uv venv is canonical; the system Anaconda env is explicitly not canonical and skips for missing deps (e.g. mlflow extras) are expected there (`pyproject.toml:1-4`, `ci/run_checks.sh:4-7`) | Running pytest under system Anaconda yields skips/false failures that do not reflect the real suite | Always use the uv environment: `uv sync`, then `uv run python -m pytest tests/ -q` (1160 tests). Do not treat Anaconda skips as regressions |
| 27 | Metrics / charts size | `METRIC_REGISTRY` holds the full 144-name catalog (`pipelines/modeling/metrics.py:1284-1335` lists, `1380-1384` registry build); `make_charts` builds every chart for the task in one call (`pipelines/modeling/charts.py:1690-1759`) and returns them closed. `'all'` selects every registry metric (`pipelines/modeling/metrics.py:1416-1418`) | Selecting `'all'` metrics and/or all chart groups on large eval frames inflates time, memory, and artifact size | Keep metrics at the curated default (absent → `DEFAULT_METRICS`, `pipelines/modeling/metrics.py:1339-1350`) unless you need more; select an explicit list, and save figures inside `make_charts` (they return closed) |
| 28 | Health cache staleness | `/health` caches its first probe result in `_health_cache` and returns it forever after (`serving/app.py:425-444`); there is no TTL or invalidation | If the model artifact appears/fails after the first probe, `/health` keeps reporting the stale `ok`/`degraded` status until the app process restarts | Recreate the app (restart the service) to force a re-probe; do not rely on `/health` changes alone to detect a model swapped at runtime |
| 29 | Localhost allowed by design | Canonical `localhost`, `127.0.0.1`, and `::1` are explicitly allowed as registry hosts (`registry/model_registry.py:63-64,95-99,138-139,147-152`); subdomains of `localhost` and encoded/mapped loopback forms are rejected (`registry/model_registry.py:129-137`) | Loopback registry URIs work by design — intended for local dev, but a misconfigured non-loopback deployment could still reach a local registry | Bind the CLI to `127.0.0.1` by default (`cli.py:266`) and only use loopback registry URIs for local development; do not expose the service with a loopback registry in production |
| 30 | Dashboard sample data | The dashboard defaults to `sample/metrics.json` + `sample/charts/` (`web/app.js:6-7`), and that file self-identifies as sample/synthetic (`web/sample/metrics.json:12`, `generated_by: ... (sample/synthetic)`; `web/README.md:26-27`) | Opening the dashboard shows demo numbers, not your evaluation output — easy to mistake for real results | Point `CONFIG.metricsUrl` / `chartsBase` at real artifacts or copy them into `web/sample/` (`web/README.md:31,87`); serve via `python3 -m http.server` (fetch is blocked under `file://`) |

### Python 3.14 compatibility spike

`uv sync` resolution + import probe run on CPython 3.14.10, uv 0.13.7:

| Library | Result | Version |
|---|---|---|
| kedro | OK | 1.7.0 |
| h2o | OK | 3.46.0.12 |
| pyspark | OK | 4.3.0 |
| xgboost | OK | 3.4.1 |
| lightgbm | OK | 4.7.0 |
| catboost | OK | 1.2.10 |
| mlflow | OK | 3.18.0 |
| shap | OK | 0.54.1 |
| dvc | OK | 3.67.1 |
| fastapi | OK | 0.162.0 |
| scikit-learn | OK | 1.9.1 |
| pandas | OK | 3.1.1 |
| polars | OK | 2.1.0 |
| pyyaml | OK | 6.0.3 |
| matplotlib | OK | 3.11.2 |
| seaborn | OK | 0.13.2 |
| uvicorn | OK | 0.54.0 |

All 17 libraries import on CPython 3.14.10, so nothing was moved to the
`extras-314-failures` optional group and `uv sync` succeeds with every
dependency in the core set.

## Troubleshooting FAQ

| Question | Answer |
|---|---|
| `pytest` collects no tests | `testpaths = ['tests']` is now set, so bare `pytest` works; the explicit `PYTHONPATH=src python -m pytest tests/ -q` form also works (Known Issue #8, fixed) |
| `kedro run --pipeline eda` (etc.) fails | All 12 names are registered (Phase 4); check the name against `pipeline_registry._PIPELINE_MODULES` — only those 12 work (former Known Issue #12, fixed) |
| FileNotFound: `conf/base/parameters.yml` | The file was split into `conf/base/parameters/*.yml` (7 buckets) — edit the owning bucket; `params:<section>` references are unchanged (Known Issue #13) |
| My `metrics_to_calculate` had no effect | `evaluate_model_node` reads `modeling_params`, not `evaluation_params` — set `metrics`/`metrics_to_calculate` in `modeling_params` (Known Issue #14) |
| Spark scoring fails with a pickling error | The model must be broadcast/picklable for partitioned scoring (Known Issue #15) |
| `check_pandas_limits` / conversion raises on a large frame | Frame exceeds `max_rows_for_pandas` (default 5,000,000) — enable `scoring_params.spark_scoring` for the Spark path (see Scaling) |
| Real-MLflow test fails / state leaks when run in parallel | It is `@pytest.mark.serial` — run the suite serially (Known Issue #17) |
| `terraform validate` / first apply | Run `terraform init && terraform validate` in each stack first; it was not run in the docs environment (Known Issue #16) |
| Dashboard shows a fallback message | Browsers block `fetch()` under `file://` — serve it: `cd web && python3 -m http.server 8000` (see `web/README.md`) |
| `ImportError: ... requires the 'xgboost' library` | Remove `xgboost` from `models_to_use` / `modeling_params.models_to_use`, or install the library |
| `ValueError: Unknown loss_function ...` | The loss name is not in `LOSS_REGISTRY` for that adapter (or, for a dotted callable, not in the exact allowlist `LOSS_CALLABLES` in `losses.py` — e.g. use `'forecast_ml_framework.losses.huber_loss'` or `'sklearn.metrics.mean_absolute_error'`). The error lists valid losses |
| MLflow prints a banner on import | Harmless. Silence with `MLFLOW_DISABLE_AGENT_HINT=1` |
| PySpark `FutureWarning` on pandas >= 3 | Expected with PySpark 4.3 + pandas 3.x; harmless and ignored by the pipeline |
| `SQLQueryDataSet` rejects my query | Only a single `SELECT`/`WITH` statement is allowed. Multi-statement queries or writes raise `ValueError`. The query must come from trusted config |
| `SparkSQLDataSet` won't load | A live Spark session is required (passed in, or creatable via `SparkSessionFactory` config). Without Spark it raises `RuntimeError` by design |
| No charts returned / empty `charts` dict | Charts with insufficient data are skipped gracefully; check that `y_proba` is present for classification. Remember figures are returned closed — save inside the call |
| MLflow registry / DVC does nothing | MLflow registry loads via `RegistryModelLoader` / `registry/model_registry.py` (needs `tracking_uri`/`registry_uri` or `MODEL_*` env vars). DVC remains parameters/config only — no execution code |
| `forecast-ml serve` doesn't start the API | It delegates to `uvicorn forecast_ml_framework.serving.app:app` with `MODEL_ARTIFACT_PATH` (verified Phase 5); check the artifact path exists and `MODEL_NAME`/`MODEL_REGISTRY_URI` for registry mode. Without a model, `/health` reports `degraded` and `/predict` returns 503 |
| Override feature list was ignored / merged unexpectedly | A non-empty override wins alone by design. Empty it to get `selected + business − exclude` |
| Keys in my new config section are rejected | They aren't: all pydantic models use `extra='allow'`. Add a model for the section in `config_schema.py` for typed validation, or rely on the passthrough |
| First Kedro run fails on missing data files | `conf/base/catalog.yml` points at `data/raw/raw_data.csv` (or the env overlay path). Place your CSV/Parquet there, or re-point `data_params.train_data` at your catalog entry |

## Phase 2 + Phase 3: landed

```mermaid
flowchart LR
  P1["Phase 1 - done<br/>scaffold configs pipelines serving<br/>2026-11-05"] --> P2["Phase 2 - done<br/>MLflow registry<br/>batch scoring (scoring)<br/>Terraform IaC (infra/)<br/>web dashboard (web/)"]
  P2 --> P3["Phase 3 - done<br/>eda + leakage<br/>monitoring + retraining<br/>feature store + governance<br/>A/B shadow + cost<br/>interpretability + metrics v2<br/>scaling (Spark, Dataproc)"]
  P3 --> P4["Phase 4 - done<br/>register Phase 3 pipelines (12 names)<br/>SHAP -> interpretability.nodes<br/>serving feature_names + registry hardening<br/>821 tests"]
  P4 --> P5["Phase 5 - done (clean-ux)<br/>kedro markers + CLI list-pipelines / validate-config / dry-run<br/>degraded health + typed errors<br/>sample data + eda params + smoke local profile<br/>strict schema<br/>955 tests"]
  P5 --> REM["Remaining<br/>Docker / DVC exec<br/>Kedro MLflow/DVC hooks<br/>pyfunc registration"]
```

| Area | Status on disk |
|---|---|
| Model registry | **Done (Phase 2):** `registry/model_registry.py` (alias promotion, lazy mlflow import), `RegistryModelLoader.load()` works via `models/<name>@<alias>`; `create_app_from_registry` |
| MLflow integration | Tracking/registry via `registry/` + `registry_params`; `settings.HOOKS` still `()` (no Kedro hooks) |
| DVC integration | Parameters/config only; no execution code |
| IaC | **Done (Phase 2):** Terraform in `infra/gcp` + `infra/aws` (Dataproc/EMR off by default, retraining scheduler + alerts Pub/Sub; AWS EventBridge stub commented). `terraform validate` not run locally (Known Issue #16) |
| Monitoring / observability | **Done (Phase 3):** PSI/KS drift + alerts (`pipelines/monitoring`), plan-only retraining, cost tracking; static dashboard in `web/` |
| Serving | FastAPI + optional API-key auth (`FORECAST_API_KEY`) + registry + A/B shadow (`ABRouter`); `serve` CLI wired to uvicorn (`cli.py:90-95`); degraded health when no model; no Docker |
| Phase 3 pipelines | **Registered (Phase 4):** all 12 display names in `pipeline_registry.py`; `tests/test_pipeline_registry.py` asserts 12 keys (former Known Issue #12 fixed) |
| Phase 4 hardening | SHAP moved to `interpretability.nodes.shap_explain` (legacy `modeling.pipeline.explain_model` wraps it); serving validates against `feature_names_in_`; registry verified unchanged (sklearn flavor registration; pyfunc loading is opt-in only via `allow_pyfunc=True`) — Phase 4 final |

## Change Log

### 2026-11-06 — Known-issues documentation pass

- Docs only (no code changes): added the post-Phase 6 known-issues
  audit (#18–#30) under Known Issues & Fixes, each verified against
  current code with File:Line cites — H2O pickle serving rejected
  (no cluster), pyfunc opt-in / trusted registry, SQL
  trusted-config not a sandbox, no rate limiting, DNS rebinding
  (literal-IP check only), pickle native-Booster trust, Dataproc/EMR
  stubs off, `terraform validate` not run locally, system-Anaconda
  skips, metrics/charts size, health-cache staleness, localhost
  allowed by design, and dashboard sample data. #3 and #16 were
  re-verified in place as #20 and #25.
- Test count confirmed at **1075 tests** (Phase 6 land; supplied by
  the Phase 6 teams — the suite was not re-run in this docs pass).

### 2026-11-06 — Phase 6 parity

- Ntile eval-metric grammar: `ntile.parse_eval_metric` (Parity
  `<M>_<KIND>_<n>`, selectable + forecast aliases, task defaults
  `lift/decile/10` and `rmse/overall/0`, warn-and-fallback on unknown)
  behind top-level `eval_metric` in `modeling_params` /
  `modeling_reg_params` (`conf/base/parameters/modeling.yml`).
- `selected_ntile` (`{eval_metric, metric, score, ntile}`, 1-based
  ntile → 0-based group with largest-label fallback) and
  `by_centile` in `evaluate_model`; regression descriptive stats
  (`target_*/pred_*`, `total_count`, signed `max_pred_error`) and
  `confusion_rates_table` (`tpr/fpr/fnr/tnr`, `Actual Positive` /
  `Actual Negative` rows) in `metrics.py`.
- Lift table (`lift_table.COLUMNS`, 32 parity columns, decile-descending),
  threshold selection (`pr_curve`/`roc_curve`/`mcc_curve`/
  `ks_statistics`, parity rounding, unknown → 0.5), promotion gate
  (`should_promote`: classification/higher-better ≥ −5%,
  lower-better < +5%, `old_score` 0/None never promotes), scoring
  ntile columns (`score_decile`/`score_centile`,
  `calib_decile`/`calib_centile`, `high_score_ind`), modeling
  pipeline wiring (`build_lift_table` → `lift_table`,
  `build_promotion` → `promotion`), catalog anchors, and
  `eval_metric` config.
- Tests: 1075 pass (count supplied by Phase 6 teams; docs pass
  verified parity code against `ntile.py`, `evaluate.py`,
  `lift_table.py`, `threshold.py`, `validation.py`,
  `metrics.py`, `scoring/nodes.py`, and `modeling.yml`; the suite
  was not re-run here).
- Docs: test count 955 → 1075; new Metrics Parity (Phase 6, parity)
  section with grammar examples, `selected_ntile`, lift columns,
  threshold methods, promotion thresholds, scoring columns, and
  descriptive stats.

### 2026-11-06 — Phase 5 clean-ux

- Kedro project markers (`pyproject.toml:[tool.kedro]` with only
  `package_name` / `project_name` / `kedro_init_version`; alias
  `forecast-ml-framework` added) + CLI introspection verified
  against `cli.py:253-310`: `list-pipelines` (12 lines
  `<name> <module>`), `validate-config --env <env>` prints
  `config ok (<env>)`, `dry-run <pipeline> --env <env>` prints the
  planned command + node names; `--env` whitelist
  `base|local|gcp|aws` rejects traversal with exit 2.
- Serving: degraded health (`HealthResponse status: ok|degraded`,
  `model_loaded`, cached probe, `serving/app.py:431-444`);
  `serve` CLI wired to uvicorn with `MODEL_ARTIFACT_PATH`
  (`cli.py:90-95`); typed errors (`ModelNotConfiguredError` → 503,
  `ModelNotFoundError` → 503, `ModelCorruptError` → 500;
  validation → 422; execution → generic 500, no path/URI leaks,
  `serving/model_loader.py:69-90`, `serving/app.py:446-489`).
- Smoke local profile: `conf/local/parameters.yml` pins
  `models_to_use: ['sklearn']`, `validation_params.fail_fast: true`,
  local MLflow URI, DVC off; sample data at
  `data/raw/raw_data.csv` (100 rows, `id,f1-f4,cat,seq,target`)
  trains with no downloads (<2min sklearn local).
- EDA params: `conf/base/parameters/eda.yml` owns typed `eda_params`
  (`core/config_schema.py:269-291`, `EdaParams`); strict schema rejects
  wrong types (`tests/test_schema_strictness.py`); single
  `task_type` default `classification`.
- Tests: 955 pass (count supplied by Phase 5 teams; docs pass
  verified CLI/serving/config code, not a full suite re-run).
- Docs: test count 821 → 955, bucket count 6 → 7, stale serve and
  registry-stub claims removed.

### 2026-11-06 — Phase 4 hardening (final, all-teams pass)

- Pipelines registered: `pipeline_registry._PIPELINE_MODULES` now
  covers 12 display names across 8 modules (`data_prep`, `modeling`,
  `scoring`, `eda`, `monitoring`, `retraining`, `cost_tracking`,
  `interpretability`); former Known Issue #12 dropped as fixed.
  Verified against `pipeline_registry.py:19` and
  `tests/test_pipeline_registry.py` (asserts 12 keys).
- SHAP path corrected: `pipelines/interpretability/nodes.py:shap_explain`
  is the implementation; `pipelines/modeling/pipeline.py:explain_model`
  remains as the legacy modeling-pipeline wrapper node.
- Serving: `feature_names_in_` contract documented — required feature
  set/order comes from the model when present, else the first record's
  keys; unknown/missing features → 422 (`serving/app.py:_validate_records`).
- Registry (Phase 4 final, verified unchanged by docs pass):
  `mlflow.sklearn.log_model` + alias promotion, no pyfunc flavor;
  pyfunc loading is opt-in only via `RegistryModelLoader(allow_pyfunc=True)`.
  No Docker added.
- Docs: stale 549-test count corrected to 821, duplicated Parameters
  override paragraph removed, all "not registered" diagram/table text
  updated. Tests: 821 pass (count from Phase 4 teams; docs pass
  verified the 12-pipeline registry, not a full suite re-run).
- Dead code reported to Lead only (not fixed in this docs pass):
  `pipelines/eda/nodes.py:_try_data_charts` (return value ignored at
  call site), `__main__.py:_build_parser` duplicating `cli.py` parser,
  scipy `spearmanr` import in `pipelines/modeling/metrics.py:_spearman`
  (manual fallback exists), `feature_store/` offline/online
  copy-paste, and the `TODO parameters.yml` docstring in
  `pipelines/data_prep/pipeline.py:6`.

### 2026-11-05 — Phase 3

- Pipelines: `pipelines/eda` (`eda_summary` → `eda_report`, leakage
  detection), `pipelines/monitoring` (PSI/KS drift + severity alerts),
  `pipelines/retraining` (plan-only triggers), `pipelines/cost_tracking`
  (offline cost estimation, no cloud APIs), `pipelines/interpretability`
  (SHAP/PDP/LIME + 27 charts), plus Phase 2 `pipelines/scoring` (batch
  scoring, Spark partitioned). **Not registered** in
  `pipeline_registry.py` yet (Known Issue #12).
- Feature store (`feature_store/`: offline/online, point-in-time
  consistency), governance (`governance/`: secret-scrubbed audit,
  lineage with config hash, retention), A/B shadow serving
  (`serving/ab_testing.py:ABRouter`, `compare_variants`).
- Metrics v2: 144-name `METRIC_REGISTRY` with selectable
  `metrics`/`metrics_to_calculate` (`'all'` / list / absent = curated
  default subset); full chart catalog groups I–VII in
  `pipelines/modeling/charts.py`.
- Config split (breaking): `conf/base/parameters.yml` →
  `conf/base/parameters/*.yml` (6 buckets: global, data, preprocessing,
  modeling, evaluation, platform); section names unchanged, loader
  merges all `*.yml` (Known Issue #13). `evaluation.yml` ships a
  16-name `metrics_to_calculate` default (Known Issue #14 wiring note).
- Scaling: Spark partitioned scoring (`spark_scoring.enabled`,
  `mapPartitions`), pandas 5M-row guard
  (`pandas_conversion_params.max_rows_for_pandas`), Dataproc
  autoscaling defaults (`spark_params`: dynamic allocation + AQE,
  `shuffle.partitions: 1000`); 100M-row guidance in this README and
  `infra/README.md`.
- Dashboard + IaC (Phase 2, now documented): static evaluation
  dashboard in `web/` (serve via `http.server`, not `file://`),
  Terraform in `infra/gcp` + `infra/aws`; MLflow registry
  (`registry/model_registry.py`, `RegistryModelLoader.load()` works);
  catalog Phase 3 datasets (`monitoring_report`, `alerts`, `cost_log`,
  `lineage`, `audit_log`).
- Tests: 549 tests pass across 28 files in `tests/` (new: `test_eda`,
  `test_monitoring`, `test_retraining`, `test_cost_tracking`,
  `test_interpretability`, `test_feature_store`, `test_governance`,
  `test_ab_testing`, `test_scaling`, `test_phase3_config`, plus Phase 2
  `test_registry`, `test_scoring`, `test_phase2_config`).
  `pyproject.toml` `testpaths` fixed to `['tests']` (Known Issue #8);
  real-MLflow test is `@pytest.mark.serial` (Known Issue #17).
- Known Issues added: #12 Phase 3 pipelines not registered, #13
  `parameters.yml` split migration, #14 metrics default-subset
  semantics, #15 Spark scoring model broadcast, #16 terraform validate
  not run locally, #17 MLflow serial marker. Stale claims removed
  (Phase 1-only header, "dashboard NOT implemented", registry stub).

### 2026-11-05 — Phase 1 initial

- Scaffold: `pyproject.toml` (Python >= 3.14, uv, hatchling build,
  `forecast-ml` console script, pytest/ruff config), `.venv` inside
  `forecast_ml_framework/`, `src/forecast_ml_framework/__init__.py`
  (`__version__ = '0.1.0'`).
- Configs: `conf/base/{variables,parameters,catalog}.yml` plus
  `conf/local/`, `conf/gcp/`, `conf/aws/` overlays; flat runtime params
  for catalog interpolations; regression and classification modelling
  sections; per-env artifact/mlflow/dvc/spark overrides.
- Pipelines: `pipelines/data_prep` (validate → preprocess → select) and
  `pipelines/modeling` (split → train zoo → calibrate → predict →
  evaluate → charts → SHAP); `pipeline_registry.register_pipelines()`
  with lazy imports and alias names; `settings.py` (`CONF_SOURCE`,
  empty `HOOKS`).
- Serving: FastAPI app (`/health`, `/predict`) with strict validation
  and payload caps; `FileModelLoader` with the restricted unpickler;
  `RegistryModelLoader` Phase 2 stub.
- Core: artifact stores (local/GCS/S3) with path jail and restricted
  unpickler; `ConfigLoader` with business/override/exclude merge rule;
  `SparkSessionFactory` with graceful fallback; `SQLQueryDataSet` /
  `SparkSQLDataSet` Kedro datasets.
- Reviewer / adversarial fixes: restricted pickle unpickler, artifact
  path jail, SQL single-statement `SELECT`/`WITH` validation, matplotlib
  figure lifecycle (close-on-failure + close-all), H2O `predict_proba`
  guard (no label substitution), params non-mutation, console script
  corrected to `forecast-ml`, `httpx` dev dependency for `TestClient`.
- Importable strings: dotted `model_handle` / `calibration_handle` /
  `tuning_handle` / `handle` resolution via `resolve_handle()`
  (`data_prep/nodes.py`, `modeling/adapters.py`); `HandleAdapter` for
  any estimator; loss-function dotted callables.
- Loss functions: `LOSS_REGISTRY` + `LOSS_KWARGS` per adapter;
  top-level default with per-model overrides; aliases; `ValueError`
  listing valid losses on unknown losses.
- Pydantic + OmegaConf: `core/config_schema.py` (per-section models,
  `extra='allow'`, `validate_*`, merged `load_config()` with resolved
  interpolations); `ConfigLoader` accepts dict / OmegaConf / pydantic.
- Kedro datasets: `pandas.CSVDataset` / `pandas.ParquetDataset` /
  `PickleDataset` / `JSONDataset` / `MemoryDataset` catalog entries,
  `anchors`/`*_params` pattern, SQL/Spark datasets in code.
- Tests: 16 test files in top-level `tests/` (`test_app`, `test_catalog_params`,
  `test_cli`, `test_conf_omegaconf`, `test_config_schema`, `test_core`,
  `test_data_prep`, `test_handles`, `test_loss_params`,
  `test_model_loader`, `test_modeling_adapters`,
  `test_modeling_evaluate_charts`, `test_modeling_loss`,
  `test_modeling_pipeline`, `test_modeling_split_calibrate`).
- Python 3.14 compatibility spike: all 17 libraries OK (table in
  Known Issues); `MLFLOW_DISABLE_AGENT_HINT=1` documented; PySpark
  pandas >= 3 `FutureWarning` documented.
- README rewrite: comprehensive Phase 1 documentation (this file) —
  features, quickstart, architecture, full configuration guide,
  feature-selection rule, metrics/charts catalog, serving API, Known
  Issues & Fixes, Troubleshooting FAQ, Phase 2 roadmap, Change Log.

### 2026-11-05 — Mermaid diagrams

- Added Mermaid fenced diagrams (GitHub-renderable) to this
  README: system architecture flowchart (console → Kedro → pipelines
  → artifacts/MLflow/DVC → FastAPI, with local/GCP/AWS env overlays),
  data-flow sequence + flowchart (raw → validation → FE → selection →
  split → model zoo → calibration → evaluation/charts/SHAP →
  registry/serving), config-flow diagram (variables + parameters +
  catalog → OmegaConf merge → Pydantic → ConfigLoader → pipelines),
  feature-selection decision flowchart (override wins, else
  selected + business − exclude), and phase roadmap flowchart
  (Phase 1 → Phase 2 → Phase 3 with red-team acceptance + commit
  gates). Node names verified against `pipeline_registry.py` and
  `pipelines/*/pipeline.py`; dashboard and Phase 2 execution boxes
  marked not yet implemented.

### YYYY-MM-DD — <version / milestone name>

```
- Added: ...
- Fixed: ...
- Changed (breaking): ...
```
