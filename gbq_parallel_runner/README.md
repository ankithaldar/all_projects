# BigQuery Parallel Runner

Runs a set of BigQuery SQL scripts as a staged, parallel ETL. The scripts are
grouped into dependency **levels**, and each level runs only after the previous
one has finished successfully. Within a level, every SQL file runs against every
**run configuration** at the same time, each as an independently retried job,
while a live table on stdout shows state, bytes processed, slot milliseconds and
runtime for each one.

The point of the shape is that you describe *what* has to happen in one config
file and *which* slices you want it for as a list of overrides, and the runner
handles the ordering, the concurrency, the retries and the scheduling.

## What is in this folder

| File | Role |
|---|---|
| `run_etl_in_levels.py` | The entire engine. Single file, no local package to install. |
| `config.yml` | The config contract: base config, run configs, level → SQL map. |
| `run_etl.ipynb` | Driver notebook: installs deps, checks auth, then invokes the script. |
| `requirements.txt` | `google-cloud-bigquery`, `PyYAML`, `structlog`, `rich`. |
| `test_gbq_parallel_runner.py` | Regression tests; third-party deps are stubbed. |
| `sqls/` | The SQL to run: `drops.sql` and `etl.sql`, wired into `sql_level_maps`. |

`run_etl_in_levels.py` is the whole implementation — there is no module layout to
learn. Read it top to bottom and you have seen the tool: types, logging,
`Config`/`deep_merge`/`ConfigResolver`, placeholder validation, `BQRunner`,
`ProgressTracker`, `Orchestrator`, CLI.

## The config file

`config.yml` has exactly three top-level keys, and all three are required:

```yaml
base_config:      # things that never change
  billing_project: my-billing-project
  project_id:      my-source-project
  dataset_id:      my_dataset
  max_concurrency: 6
  max_retries: 3
  max_backoff_seconds: 30

sql_level_maps:   # dependency order: level -> SQL files
  0:
    - sqls/drops.sql
  1:
    - sqls/etl.sql
  2:
    - sqls/drops.sql

run_configs:      # one entry per parallel ETL run
  - scheduled_time: "2026-01-01T02:00:00Z"
    key_01: alpha
    key_02: 2026-01-01
  - scheduled_time: "2026-01-01T02:00:00Z"
    key_01: beta
    key_02: 2026-01-01
```

### `base_config`

Everything that is shared by all runs: the billing project the queries are
charged to, the source `project_id` and target `dataset_id`, and the retry and
concurrency defaults. It may itself contain `${...}` placeholders, which are
resolved against the merged config.

### `run_configs`

A **list**, and that is what makes the runner parallel. Each entry is a dict of
overrides layered on top of `base_config` via a deep merge. Dicts merge
recursively; lists are **concatenated**, not replaced; a `null` override is
ignored. With two run configs and two SQL files in a level, four BigQuery jobs
start at once.

Every run config may carry its own `scheduled_time`. That is the hook for the
common pattern of "rebuild every region from yesterday's data, each a few hours
apart" — you list the times, and the runner waits for each one independently.

`scheduled_time` is an ISO 8601 string, `YYYY-MM-DDTHH:MM:SSZ`. A time with no
timezone offset is assumed UTC. If the time has already passed today, the runner
logs a warning and runs immediately rather than skipping the run.

### `sql_level_maps`

An integer level → list of SQL paths, resolved relative to where you ran the
command from. Levels are executed in ascending numeric order and the whole run
stops at the first level that fails: level *N+1* never starts if any job in level
*N* failed, because the level below it usually built the tables the next one
consumes. The example above is `drops.sql` → `etl.sql` → `drops.sql`: clear the
target, build it, drop the staging tables.

### Placeholders

There are two placeholder styles, and they are not interchangeable.

- **`${var}` in the YAML.** Resolved by `ConfigResolver` against the merged
  config itself, treated as a flat namespace. Every placeholder must resolve to
  a string. The resolver iterates until the value stops changing (max 10 passes)
  and raises on anything unresolved, so a typo surfaces as a config error before
  a single job is submitted. The classic use is composing an id from parts, e.g.
  `run_id: '${placeholder_01}_${placeholder_n}'`.

- **`{var}` in the SQL.** Plain `str.format`, resolved from the same merged
  config. So `sqls/etl.sql` containing

  ```sql
  SELECT * FROM `{project_id}.{dataset_id}.source`
  WHERE dt = '{key_02}' AND region = '{key_01}'
  ```

  gets `project_id`, `dataset_id`, `key_01` and `key_02` substituted per run.

  **Every key a SQL file references must exist in the config**, or validation
  exits 1 before anything is submitted. And **literal braces do not work** —
  the scanner is a plain regex for "brace, word, brace", so `{{foo}}` matches
  `foo` exactly like `{foo}` does, and `str.format` then escapes it into the
  literal text `{foo}` rather than substituting anything. There is no working
  escape: a file needing a literal `{` or `}` cannot use this runner as written.
  Keep brace-shaped text, including brace-shaped examples in SQL comments, out
  of the file.

Before anything is submitted, every SQL file referenced by the level map is read
and its placeholders are checked against the config. A missing key logs every
offending file and placeholder and exits `1`. This validation is what makes
`--dry_run` cheap: config errors cost you nothing in slots.

## Running it from bash

```bash
pip install -r requirements.txt
gcloud auth login --update-adc --no-launch-browser

python run_etl_in_levels.py --all_configs config.yml
```

The flags:

| Flag | Meaning |
|---|---|
| `--all_configs PATH` | **Required.** Path to the YAML config. |
| `--dry_run` | Validate config and placeholders, submit nothing. |
| `--max-concurrency N` | Cap on concurrent jobs per level. |
| `--time_set "ISO8601"` | Sleep until this time, then start the whole run. |

`--dry_run` is the one to use first on any new config. It runs the identical
resolution and validation path, then sleeps 100ms per job instead of submitting
it, and marks each one `DRY_SUCCESS` in the progress table. If that table fills up
with `DRY_SUCCESS` rows, the real run will submit cleanly.

`--max-concurrency` is per level. Be aware that the level gate is strict but the
global footprint is not: it is capped independently (see *Known rough edges*
below), so raising this past that ceiling changes nothing.

`--time_set` is the global counterpart of a run config's `scheduled_time` and is
useful for "start the whole thing at 02:00": it sleeps in five-second chunks so
it stays responsive to Ctrl-C, then hands over to normal execution.

The process exits `0` if every level succeeded and `1` on any failure,
validation error, or shutdown.

## Running it from the notebook

`run_etl.ipynb` is a thin wrapper over the same script, meant for a Colab or
local Jupyter session where you are iterating on the config and would rather not
leave the browser. It has seven cells and only one does real work:

1. **`%%time` install cell** — `pip install --upgrade pip`, then
   `pip install -r requirements.txt`, then `pip cache purge`.
2. **`%load_ext autoreload` / `%autoreload 2`** — so edits to the script are
   picked up without restarting the kernel. This is the reason to use the
   notebook at all: you can change the retry logic and re-run without a reload.
3. **Auth** — a commented-out `gcloud auth login --update-adc --no-launch-browser`.
   Uncomment and run it once per session. On Colab you would use
   `auth.authenticate_user()` instead; anywhere with `GOOGLE_APPLICATION_CREDENTIALS`
   set in the environment, do nothing.
4. **Markdown** — reminds you which config goes where: the non-changing
   parameters in the base config, the per-run parameters in the run configs.
5. **`%%bash` run cell** — the actual invocation:

   ```bash
   python run_etl_in_levels.py \
     --all_configs base_config.yml \
     --max-concurrency 100
   ```

6. **Markdown** — the same thing again as a copy-pasteable bash block, with
   `--max-concurrency 25`.

Two things to know about the notebook path. First, the `%%bash` cell runs with
the notebook's directory as the working directory, so `--all_configs config.yml`
and the `sqls/` paths in `sql_level_maps` are both resolved relative to the
notebook, not to wherever you launched Jupyter. Second, **the run cell as written
references `base_config.yml`, which is not the name of the file in this
folder** — the committed config is `config.yml`. Edit the cell to
`--all_configs config.yml` before running it, or the script exits immediately
with a file-not-found.

The `rich` live table renders fine inside a standard Jupyter cell, but it does
not render in a plain `%%bash` output cell the way it does in a terminal, so if
you see no progress table, check the script's structlog INFO lines for
`Starting level N` and job state changes instead.

## Concurrency

Two limits stack:

- **Per level**, `max_concurrency` jobs. In code this defaults to
  `base_config.max_concurrency` (or 3), falling back to the number of run
  configs.
- **Globally**, a single semaphore shared across all levels and all run configs,
  hardcoded in `Orchestrator`.

Each job reads its SQL, interpolates placeholders, optionally waits for its
`scheduled_time` **before** acquiring a semaphore, so a run scheduled for 09:00
does not occupy a concurrency slot for the three hours before it fires. Then it
submits through the blocking BigQuery client pushed onto a worker thread with
`asyncio.to_thread` — that is the entire concurrency mechanism, since the SDK has
no async API. One `asyncio` event loop drives all the waiting and all the
retries.

## Retries

`_run_with_retry` loops until success or `max_retries` (default 20 in code, set
to 3 in `config.yml`). Backoff is `min(2 ** attempt, max_backoff_seconds)` — 1s,
2s, 4s … capped at 30s in the example config. Each attempt gets a **fresh
`job_id`**, formatted as
`bq_parallel_run__{run_id}__{sql_stem}__{level}__{epoch_ms}`, so retries are
distinguishable in the BigQuery console rather than looking like one
indeterminate job. When retries are exhausted the exception is re-raised, the
task is recorded as `FAILED`, the level is reported as failed, and the run stops.

## The progress table

`ProgressTracker` holds one row per `(run_id, sql_path)` pair behind an
`asyncio.Lock` and renders it once a second through `rich.Live`:

| Level | Run-ID | SQL File | Job ID | State | Bytes | Slot ms | Runtime s |
|---|---|---|---|---|---|---|---|

`Bytes` is `total_bytes_processed`, `Slot ms` is `slot_millis` — the two numbers
that tell you what a run actually cost. Every terminal state renders, including
`DRY_SUCCESS` from `--dry_run` and `FAILED` after retries are exhausted. Rows are
sorted by SQL file then run id, so the same script across all your slices sits
together, which makes it easy to eyeball whether one region is lagging the rest.

## Shutting down

`SIGINT` and `SIGTERM` install handlers that set a shutdown flag, then cancel
every outstanding asyncio task *and* every BigQuery job those tasks submitted —
the runner tracks its live jobs for exactly this reason, because a cancelled
asyncio task still leaves the job running and billing on the BigQuery side.
Results are gathered and the process exits `1`. If you hit Ctrl-C mid-run you get
a clean stop and a non-zero exit code rather than a stack trace.

## Known rough edges

These are live in the code, not hypothetical:

- **`--max-concurrency` is mostly ignored.** `max_concurrency` is resolved as
  `len(run_cfgs) or args.max_concurrency or base_config.max_concurrency`, so
  whenever there is at least one run config the CLI flag and the config value are
  both discarded and you get one job per run config instead. `--max-concurrency
  100` in the notebook cell therefore does not do what it looks like.
- **The global concurrency cap is hardcoded** — currently 10 — inside
  `Orchestrator`, and is separate from the per-level cap. So no matter what you
  pass to `--max-concurrency`, more than 10 jobs never run at once. Raise it in
  `Orchestrator.__init__` if you actually want wide fan-out.
- **The notebook's run cell points at the wrong file.** It invokes
  `--all_configs base_config.yml`, but the config committed here is
  `config.yml`. Edit the cell before the first run.
- **Ctrl-C cancels the BigQuery jobs, but only the ones still tracked.** The
  runner registers a job when it submits it and forgets it once it succeeds,
  so shutdown cancels live jobs; a job whose retries were exhausted has already
  been forgotten and is cancelled by no one. That case ends the run anyway, so
  it rarely bites.
- **Literal braces cannot be escaped.** `collect_placeholders` matches the inner
  `{foo}` of `{{foo}}` too, so an escape both demands a config key you never
  meant to supply and still comes out as literal `{foo}` after `str.format`. The
  sample SQL in `sqls/` therefore avoids literal braces entirely, comments
  included.
- **`sql_level_maps` runs `drops.sql` at level 2 as well as level 0**, which drops
  the table `etl.sql` just built. Point level 2 at a different file if you want
  the result kept.
- **`project_id` and `dataset_id` ship as empty strings.** The sample SQL builds
  valid table references only once you fill them in.
- **The retry loop catches `Exception` broadly.** Transient BigQuery, network
  and quota errors all arrive as different types and retrying on any of them is
  the point, so the catch is deliberately wide and carries a `pylint: disable`
  rather than a narrower type list. The cost is that a genuine bug — a bad key
  name, say — is retried `max_retries` times before surfacing.

## Linting and tests

```bash
pylint --rcfile=../.pylintrc run_etl_in_levels.py   # 10.00/10
python -m pytest test_gbq_parallel_runner.py        # 11 passed
```

`test_gbq_parallel_runner.py` covers the retry, cancellation, scheduling and
placeholder paths, and pins the bugs that were fixed in this pass: the dropped
`await` on the `DRY_SUCCESS` and `FAILED` progress transitions, shutdown not
reaching the BigQuery jobs, `ConfigResolver` labelling every error as a
run-config error, and the missing `typing` imports. The third-party
dependencies are stubbed in the test module, so it runs without
google-cloud-bigquery, PyYAML, structlog or rich installed.

## Dependencies

```
google-cloud-bigquery   # the blocking client, run on worker threads
PyYAML                  # config loading
structlog               # structured logging
rich                    # live progress table
```

Python 3.10+, which is what the module's docstring asks for. The type hints
themselves are all lazy (`from __future__ import annotations`), so nothing is
evaluated at import time.
