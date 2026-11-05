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

## Tests

```bash
cd forecast_ml_framework
PYTHONPATH=src python -m pytest tests/test_conf_omegaconf.py \
  tests/test_catalog_params.py tests/test_loss_params.py -q
```

`test_conf_omegaconf.py` loads base conf via OmegaConf and asserts
interpolations resolve; `test_catalog_params.py` asserts every
catalog `type:` string is importable (skipping optional libraries
that are not installed).

## Environments: file x env matrix

| File | `base` | `local` | `gcp` | `aws` |
|---|---|---|---|---|
| `variables.yml` | column registry (the definition) | — (inherits base) | — (inherits base) | — (inherits base) |
| `parameters.yml` | all parameters (the definition) | minimal overrides (artifact store, MLflow, DVC, Spark) | — (inherits base) | — (inherits base) |
| `catalog.yml` | all datasets (the definition) | `filepath` only -> `./data`, `./artifacts` | `filepath` only -> `gs://forecast-ml-artifacts/...` | `filepath` only -> `s3://forecast-ml-artifacts/...` |

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
