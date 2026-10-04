# Local Testing Guide

How to get `stock_rl` running, tested and linted on your own machine.

This guide is standalone. You do not need to read
[NIFTY-RL-Technical-Doc.md](NIFTY-RL-Technical-Doc.md) first, and it does
not assume any of the deployment architecture described there.

Everything below was executed against the repository as committed. Where
this guide quotes an exit code or a message, that is what actually
happened.

---

## 1. Prerequisites

| Requirement | Pinned in | Verified |
|---|---|---|
| **Python >= 3.14** | `pyproject.toml`, `requires-python` | 3.14.7 |
| **uv** | used by every `make` target | 0.11.16 |

`.python-version` contains `3.14`, so `uv` will fetch and pin the right
interpreter for you. **You do not need a system-wide Python 3.14.** `uv`
manages the interpreter and the virtualenv in `.venv/`.

Check both:

```bash
uv --version
cat .python-version
```

If `uv` is not installed, the canonical installer is
<https://docs.astral.sh/uv/getting-started/installation/>. Do not
hand-roll a `venv` — the `make` targets assume `uv run`, and `uv run`
is also what guarantees the dev tools are present.

---

## 2. Why there are zero runtime dependencies

`pyproject.toml` declares:

```toml
dependencies = []
```

Confirmed at runtime — the installed distribution requires nothing:

```bash
uv run python -c \
  "import importlib.metadata as m; print(m.requires('stock-rl'))"
# -> None
```

The dev-only extras (`pytest`, `pytest-cov`, `pylint`) live under
`[dependency-groups]`, so they are installed by `uv sync` but are never
importable by the library itself.

### Why this matters for your setup

1. **Setup cannot fail on a resolver conflict.** There is no transitive
   tree to disagree with anything already on the machine. `uv sync`
   resolves 17 packages total, all of them tooling.
2. **No third-party code in the trading or compliance path.** The cost
   model (`costs.py`), the split logic (`splits.py`), the risk engine
   (`risk/`) and the audit trail (`execution/audit.py`) are all standard
   library. There is no library to be compromised between you and an
   order.
3. **Backtests are reproducible.** No version drift can silently change
   a number you already reported. `metrics.py` and `engine.py` have no
   NumPy to disagree about.
4. **The NSE charge model exists in exactly one place.** This is the
   one that costs you money if it goes wrong: three implementations of
   Indian transaction costs that drift apart is how a backtest stops
   matching live fills. `costs.py` is a single dataclass, and nothing
   can shadow it.
5. **The reasoning is preserved in the repo, not a wiki.** The
   docstrings in `src/stock_rl/__init__.py`, `rl/train.py` and
   `risk/var.py` all state the ceiling and the upgrade path for the
   standard-library choice, so the trade-off is auditable at the point
   someone is tempted to break it.

If you add a runtime dependency, that trade-off is no longer free and
should be argued in the commit message, not silently.

---

## 3. Install

From the repository root, in the `stock_rl/` directory that contains
`pyproject.toml`:

```bash
cd stock_rl
make sync
```

That runs `uv sync`. It is idempotent and takes under a second on a warm
cache. Verified output:

```
Resolved 17 packages in 2ms
Checked 16 packages in 0.64ms
```

You can also just run `uv run <anything>` — it syncs implicitly — but
running `make sync` once up front means later commands are quieter.

Confirm the environment:

```bash
uv run python -V
uv run python -c "import stock_rl; print(stock_rl.__version__)"
```

Verified output: `Python 3.14.7` and `0.1.0`.

---

## 4. The make targets

These are the only targets that exist. `make help` prints the same list.

| Target | Expands to | What it does |
|---|---|---|
| `help` | — | prints the target list |
| `sync` | `uv sync` | install / resolve |
| `test` | `uv run pytest --cov` | suite **+ coverage gate** |
| `lint` | `uv run pylint --rcfile=../.pylintrc ...` | the linter |
| `check` | `lint` then `test` | **before every push** |
| `clean` | — | remove caches and build output |

`clean` removes `.pytest_cache`, `.coverage`, `htmlcov`, `dist`, `build`
and every `__pycache__`:

```bash
make clean
```

---

## 5. Running the tests

### The full suite, with the coverage gate

```bash
make test
```

Both numbers below are a snapshot and both move as the suite grows. Get
the current ones with:

```bash
uv run pytest --collect-only | tail -1
```

A full `make test` on the tree as of 2026-10-05 collected **~1,900 tests
across 24 files** and took roughly 3 minutes.

`make test` is `uv run pytest --cov`, and pytest is configured entirely
in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ['tests']
addopts = '-q --strict-markers'
markers = ['network: tests that perform real outbound HTTP; deselect with
           -m "not network"']
```

Three consequences worth internalising:

- **`testpaths = ['tests']`** — bare `pytest` from `stock_rl/` already
  collects the right thing. You never pass a path.
- **`--strict-markers`** — an unregistered `@pytest.mark.foo` is an
  error, not a warning. A typo cannot silently disable a test.
- **A `network` marker exists** for tests that make real outbound HTTP.
  Deselect them with the exact string in the marker description:

  ```bash
  uv run pytest -m "not network"
  ```

  At the time of writing, no test in the tree carries the marker yet, so
  this currently changes nothing. It is wired up before the first test
  that needs it, so you do not have to remember the convention then.

### One file, or one test

```bash
uv run pytest tests/test_splits.py
uv run pytest tests/test_metrics.py -k dsr
```

### The trap: do not combine a subset run with `--cov`

`pytest-cov` is on by default only through `make test`. If you ask for
coverage on a *subset*, the gate fires and fails the run:

```bash
uv run pytest --cov tests/test_splits.py
```

Verified result:

```
FAIL Required test coverage of 90.0% not reached. Total coverage: 1.24%
$ echo $?
1
```

Nothing is wrong with your code. You measured 1.24% of the package while
running 35 tests. **Drop `--cov` for any run that is not the full
suite:**

```bash
uv run pytest tests/test_splits.py        # exit 0
```

---

## 6. The coverage gate

Configured under `[tool.coverage]` in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ['src/stock_rl']
branch = true

[tool.coverage.report]
show_missing = true
skip_covered = true
fail_under = 90
```

| Setting | Meaning |
|---|---|
| `source = ['src/stock_rl']` | only the library is measured, not tests |
| `branch = true` | partial branches count against you too |
| `show_missing = true` | print the exact uncovered line numbers |
| `skip_covered = true` | hide 100%-covered files; show only gaps |
| **`fail_under = 90`** | **hard gate; below 90% exits non-zero** |

`skip_covered` is why a clean run prints a short table. When the report
looks suspiciously short, that is the setting working, not coverage
having collapsed.

On a passing run the tail looks like this (verified 2026-10-05 on the
suite as it stood):

```
Required test coverage of 90.0% reached. Total coverage: 93.75%
1666 passed in 166.81s (0:02:46)
```

---

## 7. The linter, and the one thing that will bite you

```bash
make lint
```

which is:

```
uv run pylint --rcfile=../.pylintrc src/stock_rl tests
```

### Read this before you "fix" anything

**`--rcfile=../.pylintrc` is mandatory. Omitting it will make pylint
report a wall of errors that are not errors.**

pylint looks for its config in the current directory only. It does **not**
search parent directories. This repository puts `.pylintrc` at the
repository root, one level **above** `stock_rl/` — precisely the case
pylint will not walk up into.

The project indents Python with **two spaces**, and the config says so:

```
indent-string='  '
max-line-length=80
```

Without `--rcfile`, pylint falls back to its own built-in default of
**four**-space indentation and flags every correctly indented line in the
codebase. Verified, on a single untouched file:

```bash
uv run pylint src/stock_rl/compliance/retention.py
```

```
src/stock_rl/compliance/retention.py:75:0: W0311: Bad indentation.
    Found 2 spaces, expected 4 (bad-indentation)
src/stock_rl/compliance/retention.py:99:0: W0311: Bad indentation.
    Found 2 spaces, expected 4 (bad-indentation)
src/stock_rl/compliance/retention.py:110:0: W0311: Bad indentation.
    Found 2 spaces, expected 4 (bad-indentation)
... one W0311 per line of the file ...
$ echo $?
20
```

(long lines wrapped here for the 80-column limit; pylint does not)

With the flag:

```bash
uv run pylint --rcfile=../.pylintrc src/stock_rl/compliance/retention.py
```

```
-------------------------------------------------------------------
Your code has been rated at 10.00/10
$ echo $?
0
```

**64 lines of output without `--rcfile`, 3 with it — on a file nobody
had touched.** That ratio is the signature to remember.

### How to recognise the symptom

**Hundreds of `W0311: Bad indentation. Found 2 spaces, expected 4`,
roughly one per line of the file.**
Missing `--rcfile`. Do not reindent anything — the code is right and the
config is absent.

**`Your code has been rated at 10.00/10` sitting next to a list of
findings.**
Normal. The rating is a floor, not a total.

**`The config file .pylintrc doesn't exist!` — and nothing else is
printed.**
A wrong path. pylint aborts before analysing anything, so the output
contains **no findings at all**, which can read as a clean bill of
health if you only skim the output. It does exit non-zero (32, its
usage-error code), so `make lint` still fails — but the message does not
tell you the path was the problem. Verified:

```bash
uv run pylint --rcfile=.pylintrc src/stock_rl/compliance/retention.py
# The config file .pylintrc doesn't exist!
# exit 32
```

**Rule: always run `make lint`, and read the exit code, not the
output.**

Two exit codes to keep straight:

| Exit | Meaning |
|---|---|
| **0** | No message of any severity |
| **20** | Messages emitted (`error`, `warning` or convention). Read them |
| **32** | pylint could not run — usage error, e.g. a bad `--rcfile` path |

So `make lint` returning non-zero means "read the finding, or read the
error", not "the tooling broke".

---

## 8. Before you push

```bash
make check
```

That is `lint` followed by `test`. Nothing else — no separate build, no
type-check step, because the project has no runtime dependencies to
resolve and no compiled artefacts.

Commit style is enforced by convention rather than a tool: two-space
indent, 80-column limit, Google-style docstrings on every public symbol.

---

## 9. A smoke test you can paste

Confirms the interpreter, the install and the metrics module all work
end to end. Verified output shown:

```bash
uv run python -c "
from stock_rl import metrics as m
r = [0.01, -0.004, 0.013, 0.002, -0.006, 0.009, 0.011, -0.002]
print('sharpe       ', round(m.sharpe_ratio(r), 3))
print('dsr @20 tries',
      round(m.deflated_sharpe(r, trials=20,
                              trial_sharpes=[1.0, 1.2, 0.8]), 4))
print('min history  ',
      round(m.minimum_backtest_length(20, 1.0), 2),
      'years for SR 1.0 over 20 trials')
"
```

```
sharpe         8.719
dsr @20 tries 0.6666
min history   3.61 years for SR 1.0 over 20 trials
```

The third line is the one that matters. Twenty trials against a Sharpe
of 1.0 needs 3.61 years of history before any Sharpe is worth
reporting. If your dataset cannot cover that, the honest answer is no
Sharpe — not a flattering one.

---

## 10. The pre-registered experiment harness

`src/stock_rl/experiment/` exists. It answers one question: **does
adding the fused context vector to the state make the strategy better
than the same strategy on price alone?** It is a harness, not a
strategy, and it is the Phase 0 measurement from
[NIFTY-RL-Technical-Doc.md §14](NIFTY-RL-Technical-Doc.md#14-roadmap).

Three arms, run over the same panels, history, cadence, cost model and
folds:

| Arm | State vector | What it answers |
|---|---|---|
| `Arm.CONTROL` | price features only | The control. Byte-identical to the no-context path |
| `Arm.CONTEXT` | A + fused context vector | Does context add anything at all |
| `Arm.PRICE_CONTEXT` | B + one injected LLM scalar | Is the extra call worth its cost |

### Run it

```bash
uv run pytest tests/test_experiment.py
```

Verified: 43 tests pass, and `pylint` rates
`src/stock_rl/experiment` 10.00/10.

Against real panels, the call is:

```python
from stock_rl.experiment import ArmConfig, run_experiment

report = run_experiment(
    panels,                                # {symbol: [Bar, ...]}
    readings=book,                         # optional sentiment
    llm_scalars={s: 0.1 for s in panels},  # required by arm C
    config=ArmConfig(seeds=(1, 2, 3), panel_kind='real'),
)
print(report.verdict, report.kill)
```

Three properties worth knowing before you trust its output:

1. **The thresholds live in `context/fuse.py` as `KillCriteria`,** not
   in the harness. The harness only feeds measurements in and reports
   the verdict verbatim. The defaults are `alpha_pvalue=0.1`,
   `sharpe_pvalue=0.1`, `min_abs_t=3.0`, `folds_passed=4` of
   `folds_total=5`, `max_monthly_one_sided_turnover=0.5`.
2. **The length gate runs before the kill criteria.** If
   `minimum_backtest_length` refuses the claimed Sharpe, the verdict is
   `INCONCLUSIVE` and **no claim may be made in either direction**. That
   is an absence of a claim, not a softer `KILL`.
3. **`context_enabled` is `False` and stays `False` regardless of the
   verdict.** A verdict is a report, not a switch. Nothing in the
   package flips the gate.

You can read the criteria directly, without running anything:

```bash
uv run python -c "
from stock_rl.context.fuse import context_enabled, KillCriteria
c = KillCriteria()
print('gate', context_enabled)
print('alpha_p', c.alpha_pvalue, 'sharpe_p', c.sharpe_pvalue)
print('min_abs_t', c.min_abs_t, 'folds', c.folds_passed, '/', c.folds_total)
"
```

```
gate False
alpha_p 0.1 sharpe_p 0.1
min_abs_t 3.0 folds 4 / 5
```

Always pass `panel_kind`. On generated panels pass
`panel_kind='synthetic'`, which is what stops a demo verdict from being
quoted as a finding.

Verified output on five synthetic panels:

```
verdict          kill
gate             1.03 years available against 0.21 required
benchmark sharpe 1.881
  context_minus_a:  dSharpe +0.000 p=1.000 |t|=0.00 folds 0/5
  price_context_minus_a:
                    dSharpe +0.000 p=1.000 |t|=0.00 folds 0/5
```

`report.kill` names every unmet threshold rather than summarising, so a
verdict cannot be reinterpreted after you have seen it:

```
kill KillCriteriaResult(passed=False, failures=(
  'alpha p=1.000 > 0.1: delete all numeric context work',
  'sharpe p=1.000 > 0.1: numeric context did not beat price-only',
  '|t|=0.00 <= 3.0: conventional t>2.0 is inadequate after data mining',
  '0 of 5 folds passed, need 4'))
```

`kill` is the **expected** verdict. The prior from
[research/agents/llm-agents-in-finance.md](research/agents/llm-agents-in-finance.md)
is that it ends in `KILL`, which is precisely why it is cheap to run
before building the data fabric rather than after.

---

## 11. What you do not need

Because there are no runtime dependencies, none of this is required:

- no Docker
- no Kubernetes, kind, Helm, Kafka, Redis, Neo4j, Qdrant or LocalStack
- no GPU
- no broker account or live API credentials
- no network access

`make sync && make check` is the entire local story. Anything that asks
you to install an infrastructure stack before running a unit test is
building a tax you have not agreed to pay.

---

## 12. Related documents

- [research/README.md](research/README.md) — index of the research corpus
- [research/agents/llm-agents-in-finance.md](research/agents/llm-agents-in-finance.md)
  — the evidence the experiment harness is built against
- [research/corrections-to-design-docs.md](research/corrections-to-design-docs.md)
  — claims struck from the design docs, and why
- [research/architecture/project-structure.md](research/architecture/project-structure.md)
  — why this is a monolith and not eleven services
- [research/methodology/deflated-sharpe.md](research/methodology/deflated-sharpe.md)
  — why `metrics.py` is written the way it is
