# Configuration guide — `conf/`

Three files per environment (`conf/base/`, plus `local` / `gcp` /
`aws` overlays that override only what they must):

| File | What it holds | Who reads it |
|---|---|---|
| `variables.yml` | Column registry: target, id/cat/num/date/seq/indicator columns, drop list, feature-selection rules | `pipelines/data_prep` nodes, `core.config_loader` |
| `parameters.yml` | Everything else: FE registry, models, loss functions, calibration, splits, Spark, artifact store, MLflow/DVC/SHAP, runtime keys for the catalog | Kedro `params:*` inputs to every node |
| `catalog.yml` | Dataset registry: every dataset name -> importable dataset class + filepath/connection | Kedro Data Catalog |

## How the files relate

```
variables.yml ── target_col, column groups, business/override/exclude
parameters.yml ── FE steps, models, loss, calibration, splits, runtime keys
catalog.yml ───── dataset name -> type (importable string) + filepath/sql
```

- **OmegaConf interpolations.** `parameters.yml` references the single
  target definition as `target_col: ${variables.target_col}` and
  stratify columns as `${..global_params.target_col}`; `variables.yml`
  uses `${.cat_cols[0]}` for `bucket_segment_col`. Load both files
  merged (`OmegaConf.merge({'variables': ..., 'parameters': ...})`)
  and call `OmegaConf.resolve(cfg)` — see `tests/test_conf_omegaconf.py`.
- **Catalog interpolations use ONE namespace:** `${params:<key>}`
  (Kedro params namespace) for runtime keys defined at the top of
  `parameters.yml` (`score_model_id`, `output_gcs_location`,
  `model_id_prefix`, `suffix`, `data_project`, `db_staging_data`,
  `etl_project`, `etl_dataset`, `etl_table`, `external_project`,
  `tmp_bucket`, `is_domino`, `run_date`, `train_run_dt`,
  `score_run_dt`, `brand`). These are **new keys** — the pydantic
  config schema must allow them (`extra='allow'`).
- **Pydantic validation.** Every shipped key must be accepted by the
  core config-schema models (`extra` allow). New keys introduced by
  this work: the runtime keys above, `loss_function` / `objective`
  (top-level and per-model), `tuning_handle`, per-model
  `model_handle` entries, and FE `handle:` keys. If you add a key,
  add it to the schema or keep `extra='allow'`.

## Switch environment / task / model / loss / feature selection

- **Environment:** run Kedro with `--env local|gcp|aws`. Overlays only
  change filepaths (URIs) and credentials come from the environment
  (`GOOGLE_APPLICATION_CREDENTIALS`, `AWS_PROFILE`) — never from YAML.
- **Task (classification <-> regression):** set
  `modeling_params.task_type` / `modeling_reg_params.task_type` and
  the matching `model_handle` (`XGBClassifier` vs `XGBRegressor`).
  `core.config_loader.ConfigLoader.task_type()` auto-detects from
  `target_type` when not set explicitly.
- **Model:** `models_to_use: [xgboost, sklearn]` trains only those.
  `model_handle` is a dotted importable string
  (`xgboost.XGBClassifier`, `sklearn.ensemble.RandomForestClassifier`,
  `lightgbm.LGBMClassifier`, `catboost.CatBoostClassifier`).
  Per-model hyperparameters live in
  `modeling_params.model_params.<name>`.
- **Loss function:** `modeling_params.loss_function` (top-level
  default, e.g. `logloss`) with per-model
  `model_params.<name>.loss_function` overrides. Valid sets and
  aliases are listed in the comment block above `modeling_params`
  in `parameters.yml`. Custom loss: use a dotted importable string
  (`'my_losses.my_loss'`). No `loss_function` = library defaults.
- **Feature selection:** toggle methods in
  `feature_selection_params.methods` (`enabled: true|false`), or
  force features via `variables.yml`:
  - `business_relevant_features` — always kept.
  - `override_features` — when non-empty, ONLY these are used.
  - `exclude_features` — always removed.

## Add a new dataset / model handle (importable string)

1. **Dataset:** add an entry to `conf/base/catalog.yml` with
   `type:` as a dotted importable string. Prefer `kedro_datasets.*`
   (`kedro_datasets.pandas.ParquetDataset`,
   `kedro_datasets.pandas.CSVDataset`,
   `kedro_datasets.spark.SparkDataset`,
   `kedro_datasets.pickle.PickleDataset`,
   `kedro_datasets.json.JSONDataset`); for SQL/BigQuery use only
   `forecast_ml_framework.core.datasets.SQLQueryDataSet` /
   `SparkSQLDataSet`. Add a one-line comment naming the dataset
   class purpose. Filepath keys must match that class's signature
   (`filepath`, `file_format`, `load_args`, `save_args`, ...).
2. **Model / FE handle:** add `handle:` / `model_handle:` /
   `label_encode_handle:` / `ohe_handle:` / `calibration_handle:` /
   `tuning_handle:` with a dotted importable string
   (`sklearn.impute.SimpleImputer`,
   `pyspark.ml.feature.StringIndexer`,
   `sklearn.calibration.CalibratedClassifierCV`,
   `sklearn.model_selection.GridSearchCV`). Code resolves these via
   `importlib.import_module` + `getattr`
   (`pipelines.data_prep.nodes.resolve_handle`,
   `pipelines.modeling.adapters.resolve_handle`). Verify with
   `tests/test_handles.py` / `tests/test_catalog_params.py`.

## Common edits walkthrough

- **Change the target:** edit `target_col` in `variables.yml` only
  (everything else interpolates it). Make sure the column exists in
  the raw data and is not in `drop_cols`.
- **Add a business feature:** append the column name to
  `business_relevant_features` in `variables.yml`.
- **Override features:** set `override_features: ['f1', 'f2']` in
  `variables.yml` (only these features are then used; exclude rules
  do not apply to overrides).
- **Change the loss:** edit `loss_function` in
  `modeling_params` (all models) or
  `modeling_params.model_params.<name>.loss_function` (one model).
  Check the valid set in the comment block first.
- **Change the split:** edit `train_test_split_params`
  (`training_split_ratio` / `test_split_ratio`,
  `stratify_col`, `random_state`).

## Model registry (`registry_params`)

Model registry hooks live in base `parameters.yml` under
`registry_params`. Every key is commented in place (strict per-key
comments for every new key):

- `tracking_uri` — MLflow tracking URI (`${oc.env:MLFLOW_TRACKING_URI,mlruns}`).
- `registry_uri` — MLflow registry URI (`''` = no registry,
  `${oc.env:MLFLOW_REGISTRY_URI,}`).
- `model_name` — registered model name (e.g. `forecast_model`).
- `alias` — MLflow alias to promote under (e.g. `champion`).
- `promote_on_metric` — metric that must improve to promote
  (e.g. `f1_weighted`; `''` = promote unconditionally).

WARNING: promoting overwrites the alias target for consumers.
The catalog placeholder `registry_model`
(`kedro_datasets.pickle.PickleDataset`,
`artifacts/models/registry_model.pkl`) gives Kedro a file-backed
handle; the real backend is MLflow via the URIs above.

## Batch scoring (`scoring_params`)

Batch scoring lives under `scoring_params` (contract:
`pipelines/scoring/pipeline.py` + `nodes.py`):

- `input_dataset` — catalog input name (`scoring_input`).
- `output_dataset` — catalog output name (`predictions`). **Chosen
  wiring:** `predictions` to match the pipeline default — the
  scoring pipeline hardcodes `outputs='predictions'` for
  `write_predictions`, so `output_dataset: predictions` keeps
  config and pipeline in sync. (Alternative `scoring_output` was
  rejected: it would diverge from the pipeline's catalog write.)
- `score_model_id` — model id to score with
  (`${..global_params.score_model_id}`, single definition).
- `batch_size` — rows per scoring batch (positive int, e.g. `1000`;
  too large can OOM on wide feature sets).
- `max_rows` — hard cap on rows scored per run (positive int, e.g.
  `1000000`).
- `max_cols` — hard cap on feature columns scored (positive int, e.g.
  `1000`; guards against runaway one-hot expansions).
- `model_path` — pickle path loaded via `FileModelLoader`
  (e.g. `artifacts/models/model.pkl`; `''` = none).
- `model_name` / `alias` — registry model name / MLflow alias for
  registry load (e.g. `forecast_model` / `champion`; mirror
  `registry_params.model_name` / `registry_params.alias`).
- `id_cols` — identifier columns carried into the output
  (e.g. `['id']`).
- `feature_cols` — feature columns to score on (`null` = all except
  id/target, or a list).
- `task_type` — `classification` | `regression` (controls `proba`).

Catalog entries: `scoring_input` / `predictions` (canonical, writable)
plus legacy writable `scoring_output`
(`kedro_datasets.pandas.ParquetDataset`, `artifacts/scoring/*.parquet`).
SQL **inputs only** are read-only — `scoring_input_bq` /
`scoring_input_athena` use only our
`forecast_ml_framework.core.datasets.SQLQueryDataSet` /
`SparkSQLDataSet` types with `${params:...}` interpolations, and their
`save` raises `NotImplementedError` by design. The former
`scoring_output_bq` / `scoring_output_athena` SQL entries were
**removed as write targets** and replaced with writable
`kedro_datasets.pandas.ParquetDataset` entries
(`artifacts/scoring/output_{bq,athena}.parquet`); export to
BigQuery/Athena in a separate writable step. No extra/swallowed keys
are present on the SQL input entries — only documented keys
(`type`, `project`, `dataset`, `table_name`, `sql`, `database`,
plus the dataset's own documented params).

## Infra hooks (`infra_params`)

CI/infra hooks live under `infra_params`. Every leaf is an
OmegaConf env-var placeholder (`${oc.env:...}`) — no bucket,
project, dataset, cluster or database literal ever ships in YAML:

- `gcp.project` (`GCP_PROJECT`), `gcp.bucket` (`GCP_BUCKET`),
  `gcp.dataset` (`GCP_DATASET`), `gcp.cluster` (`GCP_CLUSTER`).
- `aws.bucket` (`AWS_BUCKET`), `aws.database` (`AWS_DATABASE`).

Set the env vars at runtime; credentials come from the cloud SDK
environment (`GOOGLE_APPLICATION_CREDENTIALS` / `AWS_PROFILE`),
never from YAML. Empty defaults mean unset — hooks must fail fast
when a required env var is missing. All keys are Pydantic-compatible
(`extra='allow'`) and OmegaConf-resolvable
(`OmegaConf.resolve` leaves no `${...}` in parameters).

## Tests

```bash
cd forecast_ml_framework
PYTHONPATH=src python -m pytest tests/test_conf_omegaconf.py \
  tests/test_catalog_params.py tests/test_loss_params.py \
  tests/test_phase2_config.py -q
```

`test_conf_omegaconf.py` loads base conf via OmegaConf and asserts
interpolations resolve; `test_catalog_params.py` asserts every
catalog `type:` string is importable (skipping optional libraries
that are not installed).

## Environments: file x env matrix

| File | `base` | `local` | `gcp` | `aws` |
|---|---|---|---|---|
| `variables.yml` | column registry (the definition) | — (inherits base) | — (inherits base) | — (inherits base) |
| `parameters.yml` | all parameters (the definition) incl. `registry_params` / `scoring_params` / `infra_params` | minimal overrides (artifact store, MLflow, DVC, Spark) | — (inherits base incl. new sections) | — (inherits base incl. new sections) |
| `catalog.yml` | all datasets (the definition) incl. `scoring_input` / `scoring_output` / `scoring_*_bq` / `scoring_*_athena` / `registry_model` | `filepath` only -> `./data`, `./artifacts` | `filepath` only -> `gs://forecast-ml-artifacts/...` | `filepath` only -> `s3://forecast-ml-artifacts/...` |

**Why only `catalog.yml` is overridden per env (gcp/aws):** the only
thing that differs between environments is *where bytes live*
(filepaths/URIs) and *how credentials are found* (environment
variables, never YAML). Column names, models, losses and splits are
identical across envs, so duplicating `variables.yml` /
`parameters.yml` per env would only create drift. `local` additionally
overrides a few `parameters.yml` keys (local artifact store, local
MLflow folder, DVC off, `local[*]` Spark) because those are
genuinely machine-local choices.

**Per-env files are documented per key:** every overridden
`filepath` in `conf/{local,gcp,aws}/catalog.yml` carries a comment
(what changes vs base, valid values, example, warning), and every
key in `conf/local/parameters.yml` is commented the same way.

### How to add a new environment

1. Copy `conf/local/` to `conf/<new-env>/`.
2. In `<new-env>/catalog.yml`, change only the `filepath` values
   (keep dataset names and `type:` strings identical to base).
3. In `<new-env>/parameters.yml` (optional), override only the
   machine-specific keys you need (artifact store, Spark, MLflow).
   Do not copy base `parameters.yml` — inherit it.
4. Run Kedro with `--env <new-env>` and confirm credentials come
   from the environment, never from YAML.
