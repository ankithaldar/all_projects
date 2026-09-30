# Local execution harness

Runs the framework's own pipeline against a SQLite warehouse and a local disk,
so the node functions, the catalog queries and the trainer can be exercised on a
workstation with no cloud account, no credentials and no cluster.

Nothing in the framework's own layers imports this package. The dependency runs
one way, from here into the framework, which is what keeps a local harness from
becoming a second implementation of the thing it is testing.

## What is substituted, and what is not

| Production | Local | Faithful? |
|---|---|---|
| BigQuery tables and queries | SQLite file | Query semantics via a dialect translator |
| Object storage | Local directory | Yes |
| Cluster Spark session | `local[2]` | Yes, behaviourally |
| `pandas-gbq` score table | SQLite table | Write dispositions yes, types no |
| Label derived in SQL | Label derived in SQL | **Identical query** |

The label-derivation query is *translated, not replaced*. That is the point of
the harness: the business rules encoded in it -- the 58-day outcome window, the
88-day maturity horizon, the narrowing of the positive class to deliberate
attrition -- are the rules the local run applies. A second set of queries written
for the test would be a second description of the same rules, and the two would
diverge the first time a rule changed.

## Running it

```bash
cd forecasting_ml_framework
PYTHONPATH=src .venv/bin/python -m forecasting_ml_framework.local.run_nba
```

Environment:

| Variable | Meaning | Default |
|---|---|---|
| `LOCAL_WORKSPACE` | Root for the warehouse, objects and data | `/tmp/forecasting_ml_local` |
| `FORECASTING_ML_JAVA_HOME` | A Java 8/11/17 runtime for Spark | auto-detected |

## Two environmental facts that stop Spark on a workstation

**Java version.** Spark 3.5 runs on Java 8, 11 and 17. A newer JDK fails at
startup with a stack trace that names none of the three technologies involved:
`UserGroupInformation.getCurrentUser` calls `Subject.getSubject`, removed in JDK
24, so the driver dies inside its own initialisation. `spark_session` sets
`JAVA_HOME` before the session is created, so the framework's own session
builder is not modified or bypassed.

**The worker's Python.** A Spark executor launches a *separate* Python process
picked from the environment, not from the virtualenv the driver came from. When
the two minor versions differ, PySpark refuses to run -- but only after the
driver has started. `PYSPARK_PYTHON` is set to `sys.executable`, which is
correct however the run was invoked.

## The dataset

Public binary-classification data: 1,340 NBA players, 19 numeric statistics, and
a binary label recording whether the player was still in the league five years
later. A real propensity problem with a real class imbalance, not a synthetic
one.

The label is **re-derived** rather than copied. The source rows are split into a
features table carrying no outcome and an events table carrying a deactivation
date; the label is then constructed by the catalog query joining the two. So a
leak between the feature and label paths is structurally impossible, and the
date logic is genuinely exercised -- 462 of 831 deactivations fall inside the
observation window, giving a 12% positive rate rather than the source's 62%.

### One trap worth naming

`Name` is **not unique** in this dataset: 29 names belong to more than one
player, and "Charles Smith" appears nine times. Used as a join key it multiplies
a 1,340-row feed to 1,526 rows, and the run still trains, still scores, and
reports metrics computed over duplicated entities. Nothing errors, because a
fan-out join is a legitimate operation -- it is only illegitimate against a key
supposed to identify one entity.

So the seeder derives a synthetic unique key and *asserts* uniqueness rather than
assuming it, and the driver asserts row-count preservation at every stage.

## What the run checks

Beyond executing the pipeline, the run asserts the framework's structural claims:

* **Train/serve parity.** The registry is pickled, reloaded from disk, and
  applied to a frame it has never seen. The scoring frame must not gain a column
  the training frame lacks, and must carry every feature.
* **Row-count preservation.** Asserted at every transform. A fan-out join is
  silent; this makes it loud.
* **Self-describing model artefacts.** The handle, estimator and calibrator are
  all pickled, dropped, and re-read. Everything downstream uses only the re-read
  copies, so the test crosses a serialisation boundary rather than passing
  against an in-memory object.
* **Confusion counts against the scalars.** `tn + fp + fn + tp` reconciles, and
  -- the check that matters -- `tp / (tp + fp)` equals the reported precision and
  `tp / (tp + fn)` equals the reported recall. A transposed confusion matrix
  still sums to the population, so reconciliation alone cannot detect it; only
  disagreement with the scalars can.

## Module map

| Module | Role |
|---|---|
| `sql_translate` | Warehouse SQL to SQLite. Refuses constructs it will not guess at. |
| `sqlite_warehouse` | Tables, declared types, write dispositions. |
| `datasets` | Kedro dataset adapters over the above. |
| `spark_session` | The two environmental facts, and the session. |
| `nba` | Download, normalise, seed. |
| `run_nba` | The pipeline driver and its assertions. |

## Why the translator refuses rather than guesses

A mistranslated query runs and returns numbers. A refusal fails at translation
time and names what must be rewritten. `EXTERNAL_QUERY`, `EXPORT DATA`,
`ML.PREDICT` and `CREATE TEMP FUNCTION` therefore raise, and so does an interval
unit with no SQLite equivalent.

The subtle one is unit scaling. A translator that maps every unit to a suffix
renders `INTERVAL 2 WEEK` as two *days* -- a query that runs and returns the
wrong date. Weeks scale by seven and quarters by three, and a test asserts the
magnitudes rather than the suffix.

## Tests

`tests/test_local.py` covers the translator and the warehouse directly, with no
Spark and no network, because both are correctness surfaces rather than
scaffolding. The end-to-end run is the integration test, and it reports its own
failures by stage name.
