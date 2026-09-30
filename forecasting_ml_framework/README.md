# Forecasting ML Framework

A configuration-driven framework for propensity, retention and regression
models. The unit of reuse is not a model — it is a **model program**: a
declarative description of which SQL builds the training frame and derives the
label, which feature transforms run in what order over which columns, which
estimator and search to fit, which metric and decile-lift tables to emit, and how
the fitted model is promoted to *current* in the enterprise model registry.

A new production model is stood up by **authoring configuration, not writing
Python**. The framework contains no model-specific code: class names appear as
strings in YAML and are resolved reflectively at fit time.

```yaml
modeling_params:
  model_handle: 'xgboost.XGBClassifier'   # a string, resolved at fit time
  model_params: {n_estimators: 300, max_depth: 5}
  threshold_selection_method: 'mcc_curve'
  eval_metric: 'LIFT_DECILE_10'
```

Built on Kedro 1.0, PySpark 3.5 and Google BigQuery, with a local execution
harness that runs the whole pipeline on a workstation with no cloud account.

---

## Contents

1. [Quick start](#quick-start)
2. [Requirements](#requirements)
3. [How to run it](#how-to-run-it)
4. [Repository layout](#repository-layout)
5. [Architecture](#architecture)
6. [Core abstractions](#core-abstractions)
7. [Configuration](#configuration)
8. [The pipeline graph](#the-pipeline-graph)
9. [Preprocessing library](#preprocessing-library)
10. [Modelling](#modelling)
11. [Scoring contract](#scoring-contract)
12. [Promotion protocol](#promotion-protocol)
13. [Persistence](#persistence)
14. [Observability](#observability)
15. [The local execution harness](#the-local-execution-harness)
16. [Error taxonomy](#error-taxonomy)
17. [Tests](#tests)
18. [Extending the framework](#extending-the-framework)
19. [Known defects](#known-defects)
20. [Operations notes](#operations-notes)

---

## Quick start

```bash
cd forecasting_ml_framework

uv venv --python 3.11
uv pip install -e '.[dev]'

pytest                                       # 353 tests, no cloud required

# Inspect the transform catalogue and the registered graph
forecasting-ml version
python main.py pipeline
python main.py routines

# Run the local end-to-end pipeline (SQLite + local disk + local Spark)
PYTHONPATH=src .venv/bin/python -m forecasting_ml_framework.local.run_nba
```

## Requirements

| Requirement | Version | Why |
|---|---|---|
| Python | `>=3.11,<3.12` | The classifiers use `X \| None` syntax and `match`-adjacent idioms throughout |
| Java | 8, 11 or **17** | Spark 3.5 does not start on a newer JDK — see [Known defects](#known-defects) |
| `uv` | any recent | Dependency and virtualenv management |
| Google Cloud (production only) | — | BigQuery, GCS, service-account credentials |

The `transformer` extra (`torch`, `pytorch-lightning`, `torchmetrics`) is
optional. The deep-learning tier is imported lazily, so the framework runs
without it until the tier is enabled:

```bash
uv pip install -e '.[transformer]'
```

> **Installing the pinned dependency set today fails.** `optbinning>=0.25.0`
> resolves to 1.0.0, which requires `scikit-learn>=1.6.0`, while `pyproject.toml`
> pins `scikit-learn>=1.4.0,<1.6.0`. Either relax the scikit-learn ceiling or
> pin `optbinning<0.25.0`. See [Known defects](#known-defects).

## How to run it

### The local end-to-end run

This is the primary way to exercise the framework. It runs the framework's own
node functions — `fit_registry`, `apply_registry`, `model_training`,
`model_evaluation`, `lift_calculation`, `model_scoring` — against a SQLite
warehouse and a local object store.

```bash
PYTHONPATH=src .venv/bin/python -m forecasting_ml_framework.local.run_nba
```

| Variable | Meaning | Default |
|---|---|---|
| `LOCAL_WORKSPACE` | Root for the warehouse, object store and dataset | `/tmp/forecasting_ml_local` |
| `FORECASTING_ML_JAVA_HOME` | Path to a Java 8/11/17 runtime for Spark | auto-detected (see below) |
| `FORECASTING_ML_LOG_LEVEL` | Framework log verbosity | `INFO` |

The run resets the workspace on every invocation, so it always starts from a
known state — reusing the last run's warehouse is how a run appears to succeed
while reading stale output.

### The pipeline graph (production)

The graph is not statically declared. It is a function of five environment
variables read at registration time, which is what lets one codebase serve an
unbounded number of model instances. Inspect it before running anything:

```bash
python main.py pipeline                  # every node, its inputs and its outputs
python main.py pipeline model_training   # one pipeline only
```

### Running a pipeline

The `run` command is the orchestrator's entry point. The scheduler invokes the
bootstrap module as the Spark driver's main file:

```bash
python src/forecasting_ml_framework/__main__.py run \
  --pipeline=fe_training \
  --env=base \
  --params=model_key:demo_retention,run_date:2026-09-29,train_run_dt:2026-06-30
```

The bootstrap rewrites the configuration documents on disk so runtime parameters
are resolvable *before* Kedro starts, then hands over to `kedro run`. See
[Configuration precedence](#configuration-precedence).

Locally, the same run is available through the console script:

```bash
forecasting-ml run --pipeline=fe_training --env=local
forecasting-ml run --pipeline=model_training --env=local --params=train_mode:train_compareMODEL_updateMETADATA
forecasting-ml run --pipeline=__default__ --env=local
```

`run` options:

| Option | Meaning |
|---|---|
| `-p, --pipeline` | Pipeline name: `fe_training`, `model_training`, `fe_scoring`, `model_scoring`, `__default__` |
| `-e, --env` | Configuration environment (`base`, `local`) |
| `-P, --params` | Runtime parameters as `key:value` pairs, comma-separated |
| `-n, --nodes` / `-f, --from-nodes` / `-t, --to-nodes` | Partial runs |
| `-r, --runner` | `SequentialRunner` (default) or `ThreadRunner` |
| `--parallel` | Run nodes asynchronously |
| `-c, --conf-source` | Use a configuration directory other than `conf/` |
| `--runner-args` | JSON object passed to the runner |

### Utility commands

| Command | Purpose |
|---|---|
| `forecasting-ml version` | Framework version and artefact format version |
| `python main.py pipeline` | The registered graph, with every node and dataset |
| `python main.py routines` | The transform catalogue with each routine's parameter surface |
| `pytest tests/test_local.py` | The SQL translator and warehouse, no Spark and no network |

> `forecasting-ml routines` and `forecasting-ml pipeline` **do not work** — the
> subcommands live on the `project` group, which is not wired to a console
> script. Use `python main.py …`. See [Known defects](#known-defects).

### Code quality

```bash
ruff check .        # the project's single linter
ruff format .       # the formatter; two-space indent, single quotes
pytest --cov        # coverage over src/forecasting_ml_framework
```

## Repository layout

```
forecasting_ml_framework/
├── main.py                     developer entry point (pipeline / routines / version)
├── pyproject.toml              packaging, entry points, ruff + pytest + coverage config
├── requirements.txt            runtime manifest, kept in step with pyproject.toml
├── conf/
│   ├── base/                   parameters, globals, catalog, spark tuning, logging
│   └── local/                  local overrides pointing every coordinate at a local sink
├── src/forecasting_ml_framework/
│   ├── __main__.py             bootstrap entry point (the Spark driver's main file)
│   ├── cli.py                  the custom `run` command and the `project` command
│   ├── constants.py            every string that appears in more than one module
│   ├── exceptions.py           the exception taxonomy
│   ├── hooks.py                lifecycle hooks: topology export, provenance, tracking
│   ├── settings.py             how the framework's own Kedro project is wired
│   ├── core/                   Metadata, the routine contract, Registry
│   ├── preprocessing/          27 transforms + the pure sequence helpers
│   ├── modeling/               trainers, scoring, calibration, metrics, promotion, transformer
│   ├── nodes/                  feature / model / validation node functions
│   ├── pipelines/              the registry and the eight pipeline factories
│   ├── platform/               Spark lifecycle, bootstrap rewriter, config loader, context
│   ├── persistence/            five Kedro dataset adapters
│   ├── observability/          logging, data-quality assertions, MLflow tracking
│   ├── utils/                  reflection, storage, text/parameter parsing
│   └── local/                  the workstation harness (SQLite, local Spark, NBA data)
└── tests/                      353 tests
```

~17,100 lines of source, ~3,400 lines of tests.

## Architecture

Strictly one-way dependencies — an upper layer depends on lower layers, never the
reverse. The import graph is asserted by the test suite, because a layered
design whose only enforcement is convention is a layered design that drifts.

| Layer | Contents | Key files |
|---|---|---|
| L6 external orchestration | scheduler DAGs, deployment scripts, egress SQL | *not in this repo — coupling is by string only* |
| L5 configuration | parameters, globals, catalog, Spark tuning, logging, hooks, settings | `conf/`, `settings.py`, `hooks.py` |
| L4 orchestration | pipeline registry and the eight pipeline factories | `pipelines/` |
| L3 nodes | feature-engineering, modelling, validation nodes | `nodes/` |
| L2 core | metadata, transform contract, routine library, registry, trainers, scoring, metrics, promotion, sequence tier | `core/`, `preprocessing/`, `modeling/` |
| L1 platform + persistence | Spark session, bootstrap rewriter, dataset adapters | `platform/`, `persistence/`, `utils/` |

The base of the graph is `observability.logging`, `constants` and `exceptions`;
nothing imports upward from there.

## Core abstractions

| Abstraction | Role | Cardinality |
|---|---|---|
| `Metadata` | the schema contract; its `feature_resolution` is the single funnel every feature list passes through | 1 per dataset |
| `PreprocessRoutine` | the two-phase `fit`/`apply` transform contract | 27 implementations |
| `Registry` | the ordered, composable, serialisable transform chain | 1 per model run |
| `ModelTraining` | estimator lifecycle; the instance **is** the model artefact | 3 tiers |
| `get_spark_session` | process-wide compute resource | 1 per process |

### The feature-resolution funnel

`Metadata.feature_resolution` is the single place a feature list is produced, and
it is **double-gated**: a column must appear both in the requested role *and* in
`feature_cols`. The two failure modes this prevents are the ones that make a
mis-registered transform silently invisible:

* a routine that appends to a role list without appending to `feature_cols`
  produces an invisible column;
* a routine that appends to `feature_cols` for a column it did not emit produces
  a `KeyError` deep inside the trainer.

The concatenation order `NUM_COL + CAT_COL + IND_COL` is therefore the model's
feature-vector order **for the life of the persisted artefact**. Reordering the
column lists silently changes the meaning of every existing model.

## Configuration

Three documents, three precedence tiers:

| Document | File | Scope |
|---|---|---|
| `globals` | `conf/<env>/globals.yml` | per-model column contract, version-controlled |
| `parameters` | `conf/<env>/parameters.yml` | the model program: roles, transforms, estimator, search |
| catalog | `conf/<env>/catalog.yml` | every persisted dataset |
| runtime params | `--params=key:value` | per-run, from the scheduler — highest |

### Configuration precedence

The bootstrap rewrites the documents on disk before the orchestrator starts,
routing each key to exactly one channel:

```
globals document    (per-model,  version-controlled)     lowest
parameters document (per-project defaults)
runtime parameters  (per-run,     from the scheduler)    highest
```

For each key in the globals document, exactly one of two mutually exclusive
branches applies, so a value can never resolve from both channels:

* the key **was** supplied at runtime → every `${KEY` becomes `${runtime_params:KEY`;
* the key was **not** supplied → every `${KEY` becomes `${globals:KEY`.

The suffix namespace is resolved separately and asymmetrically, and deliberately
so: the parameters document is rewritten to the *default* suffix token so the
node layer can look configuration up by a stable name, while the catalog is
repointed at the *actual* suffix so the graph and the catalog agree.

A rewrite failure raises `ConfigurationRewriteError` rather than being logged —
a run against partially-rewritten configuration surfaces minutes later as an
unresolvable interpolation error, separated from its cause by the whole startup
sequence.

### Column roles

| Key | Meaning |
|---|---|
| `NUM_COL` | numeric features |
| `CAT_COL` | categorical features |
| `IND_COL` | indicator features, cast to float at load |
| `SEQ_COL` | `array<float>` behavioural sequences |
| `ID_COL` | identifiers used for projection and join keys |

Two invariants are enforced by the framework, not by convention: **no column may
occupy in two role lists**, and a sequence column must not also be declared
scalar. `Metadata.detach_sequence_features()` removes arrays from the model
feature list at construction, because the sequence family registers its *scalar
summaries* as features.

### The transform program

`data_prep_params` is a **list of single-key dictionaries**, and that shape is the
enabling decision — a mapping could express neither ordering nor repetition, so a
list is what allows two `Imputations` blocks with different scopes, and what
makes list position *be* execution order.

```yaml
data_prep_params:
  - Imputations:
      skip: False
      args:
        strategy: "none"
        custom_bounds: ${globals:fill_missing_dict}
        cols: {exclude_cols: [], include_cols: ${globals:CAT_COL}}

  - CategoricalEncoding:
      skip: False
      args:
        label_encode_handle: "pyspark.ml.feature.StringIndexer"
        ohe_flag: True
        cols: {exclude_cols: [], include_cols: ${globals:CAT_COL}}
```

`skip: false` activates a block; a block that omits `skip` entirely is excluded,
so the key is load-bearing. `modules` optionally extends where a routine is
searched, so a routine outside this framework's module is reachable without a
code change.

### The estimator block

The block is named `modeling_{suffix}` — a run with `suffix: tf` expects
`modeling_tf`. The lookup is by **prefix** and is strict: zero matches and
multiple matches are both errors, because silently picking the first of two would
train the wrong model.

```yaml
modeling_params:
  model_handle: "xgboost.XGBClassifier"     # or lightgbm / catboost / sklearn
  model_params: {n_estimators: 300, max_depth: 5}
  eval_size: 0.20                           # the holdout fraction; 0.0 trains on everything
  threshold_selection_method: "mcc_curve"
  balance_class_weight: True
  calibration_handle: "betacal.BetaCalibration"   # optional
  tuning_handle: "sklearn.model_selection.HalvingGridSearchCV"
  model_initial_params: {...}               # the BASE the search explores around
  tuning_params: {param_grid: {...}, refit: True}
  eval_metric: "LIFT_DECILE_10"             # the promotion gate's metric
```

> `model_initial_params` builds the base estimator a search explores around;
> `model_params` is used when **no** search is configured. That distinction is
> non-obvious and is a recurring source of confusion when adding a search to an
> existing model.

## The pipeline graph

Four independently runnable stages, plus `__default__` — their sum.

| Pipeline | Purpose |
|---|---|
| `fe_training` | load → [split] → sample → build registry → fit registry |
| `model_training` | spark→pandas → train → promote → score/evaluate/lift each slice |
| `fe_scoring` | load score input → apply the **persisted** registry |
| `model_scoring` | spark→pandas → score → write the score table |

### The five specialisation axes

The graph's shape is a function of five variables, read from the process
environment at registration time and written there by a lifecycle hook that reads
the run's runtime parameters.

| Variable | Values | Effect |
|---|---|---|
| `suffix` | any string | namespaces every dataset name — many models, one codebase |
| `split` | `True`/`False` | inserts the Spark-level split node |
| `is_regression` | `True`/`False` | selects the regression node variants |
| `run_mode` | `prod` / `eval` | `prod` withholds the target from the scoreset; `eval` adds the test-evaluation nodes |
| `train_mode` | `train_only` / `train_updateMETADATA` / `train_compareMODEL_updateMETADATA` / `train_compare_update` | which promotion mode is active |

Measured node counts:

| Topology | `fe_training` | `model_training` | `fe_scoring` | `model_scoring` | `__default__` |
|---|---|---|---|---|---|
| classification / prod / train_only | 4 | 13 | 2 | 2 | 21 |
| classification / prod / compare | 4 | 13 | 2 | 2 | 21 |
| classification / eval / compare / split | 5 | 13 | 1 | 6 | 25 |
| regression / prod / train_only | 4 | 9 | 2 | 2 | 17 |
| regression / eval / compare | 4 | 9 | 2 | 4 | 19 |

### Sequencing tokens

Two evaluation nodes consume the same prediction frame, so the topological sort
leaves them unordered and eligible for parallel execution. A deterministic order
is forced with throwaway datasets — `sequence_ind1` … `sequence_ind15` — threaded
as a trailing input and output through the chain, each node returning the token
it received. This is a **dependency-token idiom**: a synthetic edge inserted
purely to constrain execution order. The regression pipelines need no tokens,
because they have no calibration dimension and therefore no fan-out.

The token budget is declared (`TRAINING_SEQUENCE_TOKENS = 10`,
`SCORING_SEQUENCE_TOKENS = 15`) and exceeding it raises `PipelineTopologyError`.

### Promotion-aware routing

Exactly one node writes the production-named model trio, and it always takes the
candidate under a *different* dataset name. Kedro rejects a node that lists the
same dataset as both an input and an output, so routing the candidate through
`model_handle_candidate_{suffix}` makes the promotion explicit, keeps the output
contract identical in every mode, and makes it impossible for a node to read and
write the same artefact.

> The two generations **must** resolve to different paths. If
> `model_handle_{suffix}` and `model_handle_previous_{suffix}` pointed at the same
> file, the runner would save the challenger before the validation node read the
> incumbent, the comparison would resolve to the challenger against itself, the
> deviation would be approximately zero, the gate would always pass, and the
> champion/challenger protocol would be a no-op that appears to work.

## Preprocessing library

27 routines in `preprocessing/routines.py`, all reachable by short name from
configuration.

### Bases

| Base | For |
|---|---|
| `PreprocessRoutine` | transforms whose fitted state is small and picklable in place |
| `SparkMlRoutine` | transforms whose state is a Spark ML `PipelineModel` |
| `UdfRoutine` | transforms delegating their arithmetic to a pure helper through a UDF |

### Scalar

| Routine | Purpose | Key parameters |
|---|---|---|
| `Imputations` | fill missing values from a learned or configured bound | `strategy` (`median`/`mean`/`min`/`max`/`mode`/`none`), `custom_bounds` |
| `KNNImputation` | impute from a scikit-learn nearest-neighbour model | `handle`, `params` |
| `TreeImputation` | reconstruct each column from a predictive model | `params.regressor`, `params.classifier` |
| `CapOutliers` | clamp to a learned fence (winsorise, not filter) | `method` (`std`/`iqr`/`bound`), `bound` |
| `Normalizations` | scale with a Spark ML vector assembler and scaler | `handle`, `round_off` |
| `QuantileDiscretizer` | discretise into quantile buckets | `params.numBuckets` |
| `DomainNormalizations` | divide by a reference column, so the feature is a rate | `normalize_by_col` |

`strategy: 'none'` is a genuine no-op, not an alias for "fill with zero" —
zero-filling a *categorical string* column inserts the literal `'0.0'` as a new
category downstream, which is the opposite of what the strategy name says and
invisible in every aggregate metric.

### Categorical and cardinality

| Routine | Purpose | Key parameters |
|---|---|---|
| `CategoricalEncoding` | `StringIndexer` + optional one-hot | `ohe_flag`, `label_encode_handle` |
| `TargetEncoding` | frozen out-of-fold category→target-mean map | `handle`, `params.cv` |
| `OptimalBinning` | target-supervised split edges (`optbinning`) | `dtype`, `params.max_n_bins` |
| `HighLabelBinning` | collapse low-frequency categories | `val_count`, `bin_count_by_col` |

`TargetEncoding` falls back to the **global target mean** for unseen categories,
never to zero: a zero fallback for a retention model asserts the category never
attrits — catastrophic and entirely invisible.

### Sequence (16 routines)

Array-shape transforms (`FixedLength`, `SequenceFillNa`, `SequenceNormalize`,
`SequenceDomainNormalizations`), aggregations (`SequenceAverage`, `SequenceSum`,
`SequenceMin`, `SequenceMax`, `SequenceLast`, `SequenceDelta`,
`SequenceDateDelta`, `SequenceRecency`) and descriptors (`Slope`, `Similarity`,
`ChangeRatio`, `CoefficientOfVariation`).

Summarising routines write a *new*, suffixed column and register it in both
`feature_cols` and `numerical_cols`; in-place routines overwrite the source
array. Either way the source array survives, so several summaries of one column
coexist and no routine can silently destroy a sequence.

`get_fixed_length` pads *and* slices, so the result is always exactly the
configured length. A dense tensor cannot be built from a ragged batch, and the
customers with the most history are the ones that would break it.

## Modelling

### Three trainer tiers

| Tier | Class | Notes |
|---|---|---|
| pandas / scikit-learn | `ModelTraining` | the primary abstraction |
| Spark ML | `SparkModelTraining` | assembles a `VectorAssembler` + estimator pipeline |
| transformer | `TransformerTraining` | opt-in, needs the `transformer` extra |

All three `train` methods return the **same five-tuple**
`(self, fitted_model, calibrated_model, train_frame, holdout_frame)`. The arity
is part of the contract, not a detail of one tier: a trainer returning a
different number of values is still wired and dispatched to, then fails to
unpack at every call site.

### The self-describing handle

After `train`, a `ModelTraining` instance carries the ordered feature vector, the
resolved target, the label set, **both** operating-point thresholds and **both**
rate statistics. Pickling it yields a model artefact that is sufficient to score
on its own — no external feature list, no separate scaler, no separate threshold.

### Hyperparameter search

The searcher is invoked as `tuning_handle(**tuning_params)` with `estimator`
injected, so anything honouring that contract works: `GridSearchCV`,
`RandomizedSearchCV`, `HalvingGridSearchCV` and `OptunaSearchCV` all work without
a framework change. `refit: True` is required for `best_estimator_` to be fitted
rather than an unfitted prototype.

`tuning_handle: "optuna.study"` makes the resolved object a **module** whose
`create_study` is called, so the framework never imports Optuna itself.

Prefer `HalvingGridSearchCV` to `GridSearchCV` for anything large: a plain grid
over the seven parameters in `conf/base/parameters.yml` expands to 43,200
combinations, while the halving variant needs roughly a thousand evaluations.

### Threshold selection

| Method | Optimises |
|---|---|
| `pr_curve` | max F1 |
| `roc_curve` | max G-mean |
| `mcc_curve` | max Matthews correlation — the right default under extreme imbalance |
| `ks_statistics` | max KS separation |

`mcc_curve` escalates through three grids (`0.01`, `0.0001`, `0.000001`), each
entered **only** if the previous one selected its own leftmost point. For a
retention model with a 1 % positive rate the MCC optimum is routinely below
0.01, and a single coarse grid reports `0` because its resolution cannot express
the answer. The zoom costs nothing when the coarse answer is already adequate.

### Calibration

Calibration is **arithmetic, not refitting** by default. The framework measures
the population positive rate before reweighting (`real_rate`) and the effective
training rate after (`sample_rate`), then inverts the shift exactly with the
Saerens-Latinne-Decaestecker correction:

```
p_adj = (p · π_true/π_sample) / ( (1-p) · (1-π_true)/(1-π_sample) + p · π_true/π_sample )
```

The direction is the whole point and is easy to invert: a model trained on a
sample enriched with positives **overstates** the true probability, so
`π_true < π_sample` must *reduce* the odds. An implementation that multiplies by
`π_sample/π_true` produces a score moving the wrong way — worse than no
calibration, because the output still looks like a probability.

When no reweighting was applied, `sample_rate == real_rate` and the formula
reduces to the identity, which is why the correction ships unconditionally. A
calibrator is fitted only when one is configured:
`CalibratedClassifierCV`, `LogisticRegression`, `BetaCalibration` or
`IsotonicRegression`. An unrecognised handle is logged and falls back to
prior-shift.

### Metrics

`METRIC_REGISTRY` is the single source of truth for metric vocabulary,
spellings and direction.

| Kind | Metrics |
|---|---|
| classification, higher is better | `LIFT`, `PRECISION`, `RECALL`, `ACCURACY`, `F1`, `AUC` |
| classification, lower is better | `BRIER_SCORE` |
| regression, higher is better | `R2`, `ADJ_R2` |
| regression, lower is better | `MAE`, `MAPE`, `MSE`, `MSLE`, `RMSE`, `RMSLE` |

> `KSI` appears in `HIGHER_IS_BETTER` and in `METRIC_GRAMMAR_HELP`, but is
> explicitly subtracted from the registry, so `parse_metric('KSI')` raises
> `MetricConfigurationError` — the help text advertises a name the parser
> rejects. See [Known defects](#known-defects).

The grammar is `<METRIC>[_DECILE|_CENTILE][_n]`. `LIFT_DECILE_10` is evaluated
within the top decile; an unbanded name is evaluated over the whole population.

**Degradation that changes a number is never silent:**

* an unrecognised `eval_metric` **fails the promotion** rather than substituting
  the default — deciding a production promotion on a metric the operator did not
  choose is a governance failure, not a usability convenience;
* a requested band with no rows raises `MetricBandEmptyError` rather than
  falling back to the top band, because reporting decile 10's lift under a
  request for `LIFT_DECILE_7` presents a materially different quantity as a
  successful measurement;
* a band fallback is available via `allow_band_fallback: true`, and the
  substitution is recorded in the promotion audit trail.

## Scoring contract

Two score columns are produced simultaneously, deliberately:

| Column | Meaning |
|---|---|
| `orig_score_value` | the raw model probability — discrimination, model comparison, threshold semantics, decile banding |
| `score_value` | the calibrated probability — the published expected event probability |

Keeping both lets a consumer verify that calibration has not degraded ranking,
supports an A/B comparison of calibration strategies without re-scoring, and lets
a monitor watch the two distributions independently.

The score-table columns are `SCORE_COLUMNS` plus the banding, and a single
`project_score_frame` produces both variants. Two overlapping `select` calls kept
in sync by hand is a maintenance hazard with nothing to catch a miss.

`high_score_ind` is `Y` only when the score is **both** above the threshold and in
the top half of the distribution (decile ≥ 5).

Bands are rank-based (`pd.qcut`), so they are equal-volume by construction, then
**densely re-ranked** — `duplicates='drop'` leaves gaps in the label sequence on a
tied score, and a consumer asking for `score_decile = 2` would find no rows.

> For `K > 2` the hard prediction is the estimator's own argmax and the learned
> thresholds are **not** applied, because there is no single positive class to
> gate on. This is logged at scoring time rather than left silent. A multiclass
> model must be configured with an explicit strategy this tier does not yet
> support.

## Promotion protocol

The most consequential business logic in the framework. Two gates apply, in order.

**Gate 1 — the non-regression tolerance.** The challenger is scored on the same
held-out slice as the incumbent, at each model's own operating point, using each
model's own feature list, and is accepted if its relative deviation is within
±5 %. A challenger that is *worse but by less than the tolerance* is promoted.

That is deliberate. Retrained models on rolling windows are routinely
statistically indistinguishable, so an improvement gate would reject near-ties on
sampling noise and stall the lifecycle indefinitely. The sign is derived from the
metric's declared direction, so it is correct for losses as well as scores — a
regression gate testing `deviation < 5` accepts a challenger up to 5 % *worse*
and rejects one 4 % *better*.

**Gate 2 — temporal validity.** A challenger may only supersede an incumbent if it
was evaluated against a *strictly later* observation window (`perf_base_date`).
*A model may only supersede a better-informed one if it was itself better
informed.* This also makes re-running a day idempotent: an already-recorded
window is a `NO_OP`.

| Decision | Meaning |
|---|---|
| `PROMOTE` | strictly later evidence window; install the challenger |
| `NO_OP` | evidence window already recorded; idempotent replay protection |
| `RECORD_ONLY` | older window; retained for the audit trail, cannot supersede |

**The promotion is atomic.** The audit row, the incumbent's demotion and the
challenger's install are issued as one BigQuery multi-statement transaction, so a
failure at any point leaves the registry exactly as it was. Three separate
operations leave a window in which the incumbent is demoted and no challenger is
installed — and the drift monitor's join against the registry would then return
nothing, so it would silently *pass*.

The registry doubles as a validity predicate: the downstream drift monitor joins
the score table on `report_period >= base_date`, so its baseline covers only the
period the current version was actually serving.

**The framework cannot bootstrap its own registry.** Onboarding a new
`model_key` requires an out-of-band insert of an initial row with
`status_ind = 'c'` and a `model_perf_validation_base_date`. That dependency is
stated in the `RegistryEntryMissingError` message rather than left implicit.

Values interpolated into registry SQL are **validated, not escaped** — anything
outside `[A-Za-z0-9_.+-%:/\s]` is rejected. No legitimate model key, table name,
metric name or rationale contains a quote, so a value that does is a
configuration error that should be visible.

## Persistence

Five Kedro dataset adapters, all satisfying the same contract: validate
configuration eagerly with no I/O, return every key from `_describe`, and refuse
serialisation when holding a live engine object.

| Adapter | Carries | Notes |
|---|---|---|
| `CustomSparkBQDataSet` | Spark DataFrames ↔ BigQuery | 5 write modes; the score input's label lives here |
| `BQTableDataSet` | pandas ↔ BigQuery table | load-job API; partition decorators; carries all nine metric and lift tables |
| `BQQueryDataSet` | parameterised query with `{param}` templating | read-only |
| `GBQTableDataSet` | pandas ↔ BigQuery via `pandas-gbq` | `if_exists: replace` is what makes scoring idempotent |
| `ParquetDataSet` | versioned parquet over any `fsspec` filesystem | the version segment is what makes a rerun reversible |

### Write modes

`insert`, `upsert`, `overwrite`, `insert_overwrite`, `create_replace` — selected
by a bound-method dispatch table, so adding a mode is one entry rather than one
branch. An empty frame is always a no-op, which is what lets an empty evaluation
slice produce an empty table rather than a failed job.

`upsert` stages the payload into a **scratch** dataset rather than the target
dataset, and drops the staging table in a `finally` block whether the body
succeeded or raised.

## Observability

### Logging

The framework's logging facade is enabled by default at `INFO` with no
environment variable required, and every record carries a run identifier for
correlation.

| Variable | Effect |
|---|---|
| `FORECASTING_ML_LOG_LEVEL` | raises the root framework logger (`DEBUG`, `INFO`, …) |
| `FORECASTING_ML_RUN_NAME` | the run identifier stamped on every record |
| `KEDRO_LOGGING_CONFIG` | shape Kedro's own handlers from `conf/base/logging.yml` |

Propagation is deliberately left enabled, so a Kedro `dictConfig` — which is
exactly what `KEDRO_LOGGING_CONFIG` supplies — sees every framework record.

### Data-quality assertions

Observation is not enforcement. Every stage asserts as well as reports:

| Assertion | Failure |
|---|---|
| `assert_row_count` | a stage changed the row count beyond tolerance |
| `assert_target_cardinality` | the target has fewer than two distinct values |
| `assert_column_roles_disjoint` | a column occupies two mutually exclusive roles |
| `assert_columns_present` | a required column is absent |
| `assert_null_rate` | a validated column exceeds the permitted null rate |

A transform that silently dropped 40 % of a frame is a failure, not a plausible
model.

### Provenance hooks

For every node the hooks report the resolved object-storage path of every
pickle-backed input and output — the model handles, the fitted registries and the
metadata objects. Given the framework's reliance on path-suffixed artefacts, a run
log naming exactly which file each handle was loaded from and saved to is the
primary tool for diagnosing artefact-resolution problems.

### Experiment tracking

MLflow registration is inert unless `MLFLOW_TRACKING_URI` is set, so a developer
needs no tracking server to run the framework and a missing backend can never be
the reason a production run fails. `mlflow` and `jwt` are imported lazily.

| Variable | Purpose |
|---|---|
| `MLFLOW_TRACKING_URI` | enables tracking |
| `HOSTED_PLATFORM_USER_API_KEY` | request-header authentication |
| `HOSTED_PLATFORM_RUN_ID` | execution attribution claim |

In comparison mode the hook registers the **validation node's** output — the
winning model — rather than the training node's rejected candidate.

## The local execution harness

`src/forecasting_ml_framework/local/` runs the framework's own node functions
against a SQLite warehouse and a local disk. Nothing in the framework's own
layers imports this package; the dependency runs one way, from here into the
framework, which is what keeps a local harness from becoming a second
implementation of the thing it is testing.

```bash
PYTHONPATH=src .venv/bin/python -m forecasting_ml_framework.local.run_nba
```

### What is substituted

| Production | Local | Faithful? |
|---|---|---|
| BigQuery tables and queries | SQLite file | query semantics via a dialect translator |
| Object storage | local directory | yes |
| Cluster Spark session | `local[2]` | behaviourally |
| `pandas-gbq` score table | SQLite table | write dispositions yes, types no |
| Label derived in SQL | label derived in SQL | **identical query** |

The label-derivation query is *translated, not replaced*. The business rules it
encodes — the 58-day outcome window, the 88-day maturity horizon, the narrowing
of the positive class — are the rules the local run applies. A second set of
queries written for the test would be a second description of the same rules, and
the two would diverge the first time a rule changed.

### The dataset

Public binary-classification data from
[rt-datasets-binary-classification](https://github.com/readytensor/rt-datasets-binary-classification):
1,340 NBA players, 19 numeric statistics, and a binary label recording whether
the player was still in the league five years later.

The label is **re-derived** rather than copied. Source rows are split into a
`nba_features` table carrying no outcome and an `nba_deactivations` table
carrying a deactivation date, so a leak between the feature and label paths is
structurally impossible and the date logic is genuinely exercised: 462 of 831
deactivations fall inside the observation window, giving a ~12 % positive rate
rather than the source's 62 %.

> **`Name` is not unique in this dataset** — 29 names belong to more than one
> player, and "Charles Smith" appears nine times. Used as a join key it multiplies
> a 1,340-row feed to 1,526 rows, and the run still trains, still scores, and
> reports metrics computed over duplicated entities. Nothing errors, because a
> fan-out join is a legitimate operation. The seeder therefore derives a synthetic
> unique key and *asserts* uniqueness rather than assuming it.

### Module map

| Module | Role |
|---|---|
| `sql_translate` | warehouse SQL → SQLite; refuses constructs it will not guess at |
| `sqlite_warehouse` | tables, declared types, write dispositions |
| `datasets` | Kedro dataset adapters over the above |
| `spark_session` | the two environmental facts, and the session |
| `nba` | download, normalise, seed |
| `run_nba` | the pipeline driver and its assertions |

### What the run asserts

* **Train/serve parity** — the registry is pickled, reloaded from disk, and
  applied to a frame it has never seen.
* **Row-count preservation** at every transform. A fan-out join is silent; this
  makes it loud.
* **Self-describing model artefacts** — the handle, estimator and calibrator are
  all pickled, dropped and re-read, and everything downstream uses only the
  re-read copies.
* **Confusion counts against the scalars** — `tp / (tp + fp)` equals the reported
  precision and `tp / (tp + fn)` equals the reported recall. A transposed confusion
  matrix still sums to the population, so reconciliation alone cannot detect it;
  only disagreement with the scalars can.

### Two environmental facts that stop Spark on a workstation

**Java version.** Spark 3.5 runs on Java 8, 11 and 17. A newer JDK fails at
startup with a stack trace that names none of the three technologies involved:
`UserGroupInformation.getCurrentUser` calls `Subject.getSubject`, removed in JDK
24, so the driver dies inside its own initialisation. `spark_session` sets
`JAVA_HOME` before the session is created, so the framework's own session builder
is neither modified nor bypassed.

**The worker's Python.** A Spark executor launches a *separate* Python process
picked from the environment, not from the virtualenv the driver came from. When
the two minor versions differ, PySpark refuses to run — but only after the driver
has started. `PYSPARK_PYTHON` is set to `sys.executable`, which is correct
however the run was invoked.

### Why the translator refuses rather than guesses

A mistranslated query runs and returns numbers. `EXTERNAL_QUERY`, `EXPORT DATA`,
`ML.PREDICT` and `CREATE TEMP FUNCTION` therefore raise, and so does an interval
unit with no SQLite equivalent.

The subtle one is unit scaling: a translator that maps every unit to a suffix
renders `INTERVAL 2 WEEK` as two *days* — a query that runs and returns the wrong
date. Weeks scale by seven and quarters by three, and a test asserts the
magnitudes rather than the suffix.

## Error taxonomy

A small, explicit hierarchy rooted at `FrameworkError`, so a failure is
classifiable both programmatically and in the run log. Every type takes arbitrary
keyword context and renders it into the message, so a failure names the run, the
model and the stage without reading the traceback.

```
FrameworkError
├── ConfigurationError
│   ├── MissingParameterError
│   └── ConfigurationRewriteError
├── DataContractError
│   ├── TargetCardinalityError
│   ├── ColumnRoleConflictError
│   └── RowCountAssertionError
├── RoutineError
│   └── RoutineNotFoundError
├── ModelError
│   ├── HoldoutRequiredError
│   ├── MetricConfigurationError
│   └── MetricBandEmptyError
├── PromotionError
│   └── RegistryEntryMissingError
├── DatasetError  (also subclasses Kedro's DatasetError)
│   ├── InvalidWriteModeError
│   └── PartitionValueMissingError
├── SparkSessionError
│   └── SparkTuningConfigNotFoundError
├── PipelineTopologyError
└── InvalidSearchSpaceError  (also a ConfigurationError)
```

`DatasetError` subclasses both `FrameworkError` and Kedro's own `DatasetError`,
so a consumer catching either sees it — which matters because the catalog itself
catches Kedro's type when reporting a dataset failure.

## Tests

353 tests, none of which require a warehouse, a cloud credential or a Spark
session.

| File | Tests | Covers |
|---|---|---|
| `test_smoke.py` | 73 | every module imports, every routine constructs, every dataset configures, every write mode is dispatchable, no module can be added without a test |
| `test_local.py` | 52 | the SQL translator, the SQLite warehouse, interval scaling, supported/unsupported constructs |
| `test_modeling.py` | 48 | prior-shift calibration against its closed form, all four threshold criteria including the progressive zoom, the metric grammar, the acceptance gate in both directions |
| `test_reflection.py` | 46 | every rejection case is an attack shape; the keyword-argument channel, comprehensions, `**kwargs`, builtin escapes |
| `test_sequence_ops.py` | 34 | the pure helpers, including every empty-input and ragged-window case |
| `test_core.py` | 29 | the feature-resolution funnel, the metadata protocol, the transform contract, two-tier precedence, data-quality assertions |
| `test_pipelines.py` | 27 | all six reachable topologies build as valid DAGs, no node reads and writes the same dataset, the sequencing-token chains are unbroken |
| `test_integration.py` | 23 | a real trainer over a real frame: the self-describing handle, a pickle round trip, the score-table contract, equal-volume banding, the calibrator lifecycle, a real grid search |
| `test_slice_metrics.py` | 21 | the confusion counts cross-checked against precision and recall |

```bash
pytest                              # all 353
pytest tests/test_local.py          # no Spark, no network
pytest -m spark                     # requires a live Spark session
pytest -m bigquery                  # requires BigQuery credentials
pytest --cov --cov-report=term-missing
```

The suite is a smoke test first: it imports every module, constructs every routine
and every dataset adapter, and asserts that no module can be added without a test
covering it. That layer alone catches a routine whose constructor references a
name it never bound, an adapter that dereferences an attribute `__init__` never
assigned, and a helper that raises on a realistic input — the class of fault that
otherwise survives review and fails on first production use.

The integration layer then trains and scores for real, because a framework whose
central claim is train/serve parity is not verified by imports.

## Extending the framework

| Goal | Effort | Code change |
|---|---|---|
| new estimator (LightGBM, CatBoost, in-house) | configuration only | none |
| new hyperparameter search | configuration only | none |
| new calibrator | configuration only | none |
| new preprocessing routine | one class in `preprocessing/`, one entry in `AVAILABLE_ROUTINES` | additive |
| new persistence backend | one class + a catalog `type:` | additive |
| new metric for the promotion gate | one entry in `METRIC_REGISTRY` | none |
| new pipeline stage | one factory in `pipelines/` | additive |

The reflective resolver is the single mechanism behind the whole extension
surface. Configuration carries a *string*; the constructor accepts either a
string or an already-resolved object. Resolution is deferred to first use, so the
framework does not need XGBoost installed to run a logistic-regression model.

Search-space distributions are evaluated against an explicit allow-list of roots
(`numpy`, `np`, `scipy`, `sklearn`) through an AST walk, never bare `eval`. A
configuration document is data, and data must never be able to name an arbitrary
callable. The literal-only rule covers positional *and* keyword arguments, and
`**kwargs` is rejected — a guard inspecting only positional arguments leaves the
entire keyword channel open.

## Known defects

Verified against the current tree, with the reproduction for each.

### 1. `get_spark_session(apply_tuning=True)` cannot start Spark

`platform/spark.py` passes a plain `dict` to `SparkSession.builder.config(conf=…)`.
PySpark 3.5 requires a `SparkConf` for `conf=`; a dict belongs in `map=`.

```python
AttributeError: 'dict' object has no attribute 'getAll'
```

This breaks **every** path that applies tuning — the `run` command, the
Spark-aware context and `init_spark_session` — which is to say the entire
production entry point. It does not affect the local harness, which builds its
session directly.

*Fix:* have `_load_spark_conf` return a `SparkConf`, or pass the mapping as
`map=`.

### 2. Java auto-detection cannot identify a supported runtime

`local/spark_session.py::_java_major` joins every digit of the version string, so
`"17.0.9"` yields `1709` and `"11.0.21"` yields `11021` — neither is in
`(8, 11, 17)`.

```python
_java_major('/path/to/jdk-17')   # -> 1709, expected 17
```

The result is that auto-detection never succeeds, and the local run falls back to
whatever JDK the machine has. `FORECASTING_ML_JAVA_HOME` is honoured correctly
and is currently the only working path.

*Fix:* split on `.` and take the first component, honouring the `1.8` form.

### 3. The dependency set does not resolve

`uv pip install -e '.[dev]'` fails: `optbinning>=0.25.0` resolves to 1.0.0, which
requires `scikit-learn>=1.6.0`, while the project pins `scikit-learn<1.6.0`.

*Fix:* pin `optbinning<0.25.0` or raise the scikit-learn ceiling. The venv in
this checkout was provisioned out-of-band, which is why the suite runs.

### 4. `kedro run` is unavailable in this checkout

`pyproject.toml` declares no `[tool.kedro]` section, which is what
`kedro.utils.is_kedro_project` looks for, so Kedro reports "Kedro project not
found in this directory". Adding the section then fails differently: Kedro 1.0
requires the project package to expose a `cli` attribute, and
`forecasting_ml_framework.cli` exports `commands` and `project_commands` instead.

*Workaround:* use the `forecasting-ml run …` console script, which bypasses Kedro's
project detection entirely. (It then hits defect 1.)

### 5. `forecasting-ml routines` and `forecasting-ml pipeline` do not exist

Both subcommands are registered on the `project` click group, but
`[project.scripts] forecasting-ml` points at `cli:main`, which invokes the `run`
group. The `kedro.project_commands` entry point declares
`forecasting_ml_framework.cli:project`, and that attribute does not exist either —
the module exports `project_commands`.

*Workaround:* `python main.py routines` and `python main.py pipeline`, which
dispatch directly.

### 6. `python main.py routines` raises `TypeError`

`main.py` constructs each routine as `routine()` with no arguments, but every
routine's `__init__` requires `(global_params, params)`. The click implementation
in `cli.py::routine_catalogue` passes both correctly.

*Fix:* call `routine(global_params=None, params={})` in `main.py`.

### 7. `KSI` is advertised but unreachable

`METRIC_GRAMMAR_HELP` — the text an operator reads when a metric name is
rejected — lists `KSI`, and `HIGHER_IS_BETTER` includes it, but the registry
construction subtracts it:

```python
METRIC_REGISTRY = {
  name: {...} for name in HIGHER_IS_BETTER - {'R2', 'ADJ_R2', 'KSI'}
}
```

So `parse_metric('KSI')` raises `MetricConfigurationError` listing `KSI` among
the supported names. An operator reading the error is told to use a metric that
will be rejected on the next attempt.

*Fix:* either add `KSI` to the registry, or remove it from `HIGHER_IS_BETTER` and
from `METRIC_GRAMMAR_HELP` so the help and the parser agree.

### 8. Lint and format are not clean

`ruff check .` reports 129 findings and `ruff format --check .` would reformat 27
files. The dominant categories are `UP009` (61 redundant `# -*- coding: utf-8 -*-`
declarations), `D417` (18 undocumented parameters) and `I001` (10 unsorted
imports) — none of them behavioural.

## Operations notes

- **The container image is authoritative** in production. Pin it to a digest or an
  immutable tag: a floating tag can change the library set underneath a persisted
  pickle, and nothing version-checks it.
- **Model handles are pickles.** They carry a format stamp
  (`ARTEFACT_FORMAT_VERSION`), so a mismatch is detected on load rather than
  surfacing as an unreadable traceback deep inside `pickle`. `forecasting-ml
  version` prints the pairing.
- **The scoring half is idempotent.** `create_replace` on the scoreset and
  `if_exists: replace` on the score table mean any day can be re-run safely.
- **The target is withheld in production.** `run_mode: prod` excludes the label
  from the scoreset materialisation, so a production scoring frame never carries
  an outcome — a data-governance control, not an optimisation.
- **Label windows govern statistical validity.** `OUTCOME_WINDOW_DAYS = 58` and
  `MATURITY_WINDOW_DAYS = 88` (58 + a 30-day settlement lag). A model predicting
  a 7-day outcome and scored on a 58-day label looks far worse than it is.
  Changing either silently changes what the label means.
- **Logging is on by default** at `INFO`. The rotating file handler in
  `conf/base/logging.yml` writes to the ephemeral driver disk, so prefer the
  console handlers and let the platform's log collector do retention.
- **One pipeline per process.** The topology is passed through `os.environ`, a
  global mutable channel; two concurrent runs in one process would clobber each
  other. This is enforced externally by the scheduler. `describe_pipelines`
  prints the resolved topology at registration so a mismatch is visible
  immediately.
