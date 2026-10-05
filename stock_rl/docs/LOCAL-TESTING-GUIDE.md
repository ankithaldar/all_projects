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

## 11. The command line: `python -m stock_rl`

Thirty-odd library modules, one entry point. Five subcommands, each a thin
argument parser over work that already exists, and **the exit code is the
product** — the harness's verdict becomes a process status a script can
branch on.

Every output block below is what the command actually printed. Free-text
lines wider than 80 columns are wrapped, and stderr is shown separately from
stdout where the two differ. The fixed-width tables (`health`, `baselines`)
are left at their natural width so their columns still line up.

```bash
uv run python -m stock_rl --help
```

```
usage: python -m stock_rl [-h] [--version] COMMAND ...

Command-line entry point for the stock_rl backtest core. Plain text on
stdout, diagnostics on stderr, no colour and no progress bars.

positional arguments:
  COMMAND
    serve       run the read-only HTTP API on loopback
    experiment  run the three-arm A/B experiment and print its verdict
    baselines   run all five baselines over one panel and print a table
    health      print version, dependency status and available data
    skills      list shipped skill files, their tier and their gate status

options:
  -h, --help    show this help message and exit
  --version     print the package version and exit

exit codes:
  0  KEEP, every pre-registered kill criterion cleared
  1  KILL, at least one criterion failed (failures printed verbatim)
  2  INCONCLUSIVE, the length gate refused to evaluate anything;
     also argparse usage errors, which argparse exits 2 on
  3  nothing was run: no data, unusable data or arguments, or a
     --synthetic panel, whose verdict is never evidence
```

Verified: exit 0. `uv run python -m stock_rl --version` prints
`stock_rl 0.1.0`, exit 0. Invoking it with no subcommand is a usage error,
exit 2, not a silent success.

### The five subcommands

| Subcommand | What it does |
|---|---|
| `serve` | the read-only HTTP API, loopback only |
| `experiment` | the three-arm A/B test, verdict and failures verbatim |
| `baselines` | the five providers in `baselines.py`, as a table |
| `health` | version, dependencies, price panels on disk |
| `skills` | each shipped `.md`, its tier, its gate status |

`serve` and `experiment` are covered below. The other three take few or no
flags, so read them straight out of the parser:

```bash
uv run python -m stock_rl health --help
```

```
usage: python -m stock_rl health [-h] [--data-dir DIR] [--json]

Report what this installation actually is: its version, whether it carries
runtime dependencies, and which price panels are on disk. Reports on files, so
it cannot fabricate a result and exits 0.

options:
  -h, --help      show this help message and exit
  --data-dir DIR  directory of one CSV per symbol to report on; without it the
                  available-data section says so rather than looking for data
                  somewhere the operator did not name
  --json          print the report as JSON instead of text
```

`baselines`, `health` and `skills` all exit 0 on a run. They report on
files, which cannot be fabricated, so a non-zero status from them means the
command itself failed — `skills` lets a broken skill file raise rather than
dropping it, which is the one case where non-zero is the right answer.

### The exit-code table

This is the whole point, so it is worth being precise about which code means
what. Rows 1, 2 and 3 were each executed against this tree and are
reproduced below; row 0 was not, because `context_enabled` is `False` and the
gate has never recorded a pass, so no KEEP exists to show.

| Exit | Meaning |
|---|---|
| **0** | `KEEP` — every pre-registered criterion cleared |
| **1** | `KILL` — at least one failed, **every failure printed verbatim** |
| **2** | `INCONCLUSIVE` — the gate refused to evaluate. Or an argparse error |
| **3** | nothing ran: no data, unusable data or arguments, or `--synthetic` |

Reproduced further down: the KILL at
[Running the experiment on real data](#running-the-experiment-on-real-data);
`INCONCLUSIVE`, the usage error and three separate flavours of 3 at
[The other three exits](#the-other-three-exits-reproduced) and the sections
above it.

`2` carries two meanings on purpose. argparse already exits 2 on a usage
error, so mapping `INCONCLUSIVE` onto the same code means a script needs
exactly one branch for *"you asked wrongly, or there is not enough history"*
and one for *"the measurement says stop"*. Anything that could not be
measured is deliberately **not** 0: a CI step that ran the harness on 200
bars of history must fail, because the harness never evaluated a criterion.

Two more codes exist that the table does not list. A `--host` outside
loopback exits **2** — it is a usage error, and no socket was ever opened:

```bash
uv run python -m stock_rl serve --host 0.0.0.0
```

```
error: refusing to bind '0.0.0.0': this API has no authentication and serves
trading decisions and audit state, so it binds loopback only (127.0.0.1).
Put a TLS-terminating reverse proxy with a client certificate in front of it
instead of widening this.
```

Exit 2. So does an unreadable `--data-dir` on `serve`:

```
error: --data-dir /tmp/srl-nope could not be loaded (ValueError:
/tmp/srl-nope is not a directory)
```

On `experiment` the same mistake is 3 rather than 2, because there the
directory is the panel rather than an argument. Either way it is non-zero,
which is the part that matters.

And truncating a report with `head` exits **141** (128 + SIGPIPE), which is
the number a shell reports for a process killed by that signal:

```bash
uv run python -m stock_rl experiment --synthetic | head -3
```

```
experiment  three-arm A/B, pre-registered kill criteria
panel_kind  synthetic
            GENERATED PRICES. Nothing below is a finding about the market.
```

Exit 141. A pipeline that cut the report short must not be mistaken for a run
that produced nothing.

### `--synthetic` always exits 3, whatever the verdict

**This is the single most important thing to know about the CLI.** If the
panel is generated, the exit status is 3 no matter what the harness decided:

```bash
uv run python -m stock_rl experiment --synthetic
```

```
experiment  three-arm A/B, pre-registered kill criteria
panel_kind  synthetic
            GENERATED PRICES. Nothing below is a finding about the market.
symbols     3
run_at      2026-10-05T02:49:03Z
seeds       1, 2, 3
settings    capital=10000000 rebalance_days=21 holdings=5 max_weight=0.1
            history=60
arms        sharpe mean +/- stdev over seeds, growth multiple
  control        sharpe +0.891 +/- 0.000   growth 1.081
  context        sharpe +0.891 +/- 0.000   growth 1.081
  price_context  sharpe +0.891 +/- 0.000   growth 1.081
benchmark   equal weight 1/N sharpe +0.891
comparisons treatment minus the price-only control
  context_minus_a        dSharpe +0.000  alpha +0.00000  alpha_p 1.000
                         |t| 0.00  folds 0/5  turnover/mo 0.003
  price_context_minus_a  dSharpe +0.000  alpha +0.00000  alpha_p 1.000
                         |t| 0.00  folds 0/5  turnover/mo 0.003
length gate 2.54 years available against 0.92 required
verdict     KILL
exit        1
kill        FAILED. The strings below are the harness's own, verbatim:
  alpha p=1.000 > 0.1: delete all numeric context work
  sharpe p=1.000 > 0.1: numeric context did not beat price-only
  |t|=0.00 <= 3.0: conventional t>2.0 is inadequate after data mining
  0 of 5 folds passed, need 4
panel_kind=synthetic: this verdict is not evidence, so the exit status is
3 regardless of what it says
```

The process exit was **3**. Note the report's own `exit 1` line: the verdict
*is* KILL, the harness's mapping *would* be 1, and the CLI overrides it. That
override is the feature, and it is the only place in the CLI where a printed
exit line and the process status deliberately differ.

The reasoning, in one paragraph. The generated panel is a geometric random
walk, so it has no cross-sectional structure for the context block to
explain, and `--llm-scalar` is one constant for every symbol. All three arms
therefore rank identically, the paired difference is exactly zero, and
`KILL` is the expected verdict. **A KILL on a random walk proves the harness
discriminates** — it separates a treatment from a control when there is
nothing to separate. A KEEP on the same panel would prove nothing at all,
and a CI step that could read one as a pass is exactly the failure mode this
project's research is about: a number that cannot fail being read as a
result. So the demo panel is stamped `panel_kind=synthetic`, the report
prints "GENERATED PRICES" on its third line, and the status is 3 regardless.

The same rule covers `baselines`. A table measured on generated bars is a
demonstration of the backtester, not a comparison:

```bash
uv run python -m stock_rl baselines --synthetic
```

```
baselines  one backtester, one cost model, one panel
panel_kind synthetic
           GENERATED PRICES. Nothing below is a finding about the market.
settings   capital=10000000 rebalance_days=21 max_weight=0.1 history=60
           costs=DELIVERY
symbols    3
strategy                     sharpe   growth  drawdown  turnover       cost Rs  status
buy_and_hold                  0.891    1.081     0.023     0.695        12,464  ok
equal_weight                  0.891    1.081     0.023     0.695        12,464  ok
low_volatility                0.891    1.081     0.023     0.695        12,464  ok
momentum_ranked               0.891    1.081     0.023     0.695        12,464  ok
trend_filtered_momentum *     0.919    1.062     0.021     3.765        63,171  ok

* highest Sharpe under these settings: trend_filtered_momentum. DeMiguel
et al. (2009) is the bar to clear, and equal weight is in this table.
Research output from a backtested model. Not financial advice, not a
recommendation, and not a solicitation to buy or sell anything.
```

then on stderr:

```
panel_kind=synthetic: these Sharpes are a demonstration of the backtester,
so the exit status is 3 regardless
```

Exit 3. And note `trend_filtered_momentum` winning on a random walk is
exactly why the table is not a comparison: there is no signal in that panel
to rank.

### Pointing `--data-dir` at a directory of one CSV per symbol

There is no data in this repository, so if you want to see the CLI run
without a vendor export, generate a panel yourself. **Everything from here to
the end of the section is generated prices, not market data, and no number in
it is a finding about the market.** The generator is here so the commands are
copy-pasteable; on real vendor data you would skip it and point `--data-dir`
at your own directory.

```bash
uv run python - <<'PY'
import random
from datetime import date, timedelta
from pathlib import Path

FIRST = date(2024, 1, 1)
SYMBOLS = ('AAA', 'BBB', 'CCC', 'DDD', 'EEE')


def write(directory, bars, drift, volatility, seed):
  """Write one CSV per symbol. Generated prices, never market data."""
  directory.mkdir(parents=True, exist_ok=True)
  for index, symbol in enumerate(SYMBOLS):
    source = random.Random(seed + index)
    close = 100.0 + 25.0 * index
    rows = ['date,open,high,low,close,volume']
    previous = close
    for step in range(bars):
      close = round(close * (1.0 + drift + source.gauss(0.0, volatility)), 4)
      day = (FIRST + timedelta(days=step)).isoformat()
      rows.append(f'{day},{previous:.4f},{max(previous, close):.4f},'
                  f'{min(previous, close):.4f},{close:.4f},1000000')
      previous = close
    (directory / f'{symbol}.csv').write_text('\n'.join(rows) + '\n',
                                            encoding='utf-8')
  print(directory, sorted(p.name for p in directory.glob('*.csv')))


write(Path('/tmp/srl-demo-panel'), 700, 0.0004, 0.013, 100)
write(Path('/tmp/srl-short-panel'), 100, 0.0007, 0.009, 91)
PY
```

```
/tmp/srl-demo-panel ['AAA.csv', 'BBB.csv', 'CCC.csv', 'DDD.csv', 'EEE.csv']
/tmp/srl-short-panel ['AAA.csv', 'BBB.csv', 'CCC.csv', 'DDD.csv', 'EEE.csv']
```

Exit 0. The CSV format is whatever `stock_rl.bars.load_csv` accepts: a
header row plus one bar per line, column names matched case-insensitively
against an alias table (`date`/`timestamp`/`trade_date`, `open`/`o`,
`close`/`c`/`ltp`, and so on), timestamps strictly ascending. `volume` is
optional and defaults to 0. One file per symbol, named after the symbol —
`load_panels` keys the panel on the file stem.

Check what you have before you trust it:

```bash
uv run python -m stock_rl health --data-dir /tmp/srl-demo-panel
```

```
health
  version            0.1.0
  python             3.14.7
  runtime_deps       none declared; imports are stdlib only
  run_at             2026-10-05T02:49:46Z
  data
    directory         /tmp/srl-demo-panel
    symbols           5 (3500 bars total)
      AAA             700 bars  2023-12-31T18:30:00Z (read as IST) .. 2025-11-29T18:30:00Z (read as IST)
      BBB             700 bars  2023-12-31T18:30:00Z (read as IST) .. 2025-11-29T18:30:00Z (read as IST)
      CCC             700 bars  2023-12-31T18:30:00Z (read as IST) .. 2025-11-29T18:30:00Z (read as IST)
      DDD             700 bars  2023-12-31T18:30:00Z (read as IST) .. 2025-11-29T18:30:00Z (read as IST)
      EEE             700 bars  2023-12-31T18:30:00Z (read as IST) .. 2025-11-29T18:30:00Z (read as IST)
    calendar          no exchange holiday list is bundled in this project, so a date cannot be declared a closure here
  skills             6 shipped markdown files
```

Exit 0. The per-symbol rows are the reason a misaligned panel is visible
before you spend an afternoon on it: `health` reports symbols whose first bar
is not the panel's earliest. The `(read as IST)` suffix is the loader telling
you it assumed a naive vendor timestamp was exchange local time, rather than
printing a naive stamp you would later misread.

To see the misalignment check fire, start two of the three symbols 30 bars
late:

```bash
uv run python - <<'PY'
from datetime import date, timedelta
from pathlib import Path

out = Path('/tmp/srl-misaligned')
out.mkdir(parents=True, exist_ok=True)
for index, (symbol, offset) in enumerate((('AAA', 0), ('BBB', 30),
                                          ('CCC', 30))):
    rows = ['date,open,high,low,close,volume']
    day = date(2024, 1, 1) + timedelta(days=offset)
    for step in range(400 - offset):
        stamp = (day + timedelta(days=step)).isoformat()
        rows.append(f'{stamp},100.0,101.0,99.0,100.5,1000000')
    (out / f'{symbol}.csv').write_text('\n'.join(rows) + '\n',
                                       encoding='utf-8')
PY
uv run python -m stock_rl health --data-dir /tmp/srl-misaligned
```

```
...
      AAA             400 bars  2023-12-31T18:30:00Z (read as IST) .. 2025-02-02T18:30:00Z (read as IST)
      BBB             370 bars  2024-01-30T18:30:00Z (read as IST) .. 2025-02-02T18:30:00Z (read as IST)
      CCC             370 bars  2024-01-30T18:30:00Z (read as IST) .. 2025-02-02T18:30:00Z (read as IST)
    calendar          no exchange holiday list is bundled in this project, so a date cannot be declared a closure here
    first bars differ across symbols: ['BBB', 'CCC']. run_portfolio refuses
    a misaligned panel rather than forward-filling bars that did not trade.
```

`health` still exits 0, because a misalignment is a fact about your files
rather than a verdict. `experiment` and `baselines` will not measure it:

```
error: the harness refused these panels: panels must be aligned, got
[370, 400]
```

Exit 3 for `experiment`. `baselines` prints the reason on every row instead
of dropping them, and also exits 3:

```
trend_filtered_momentum           -        -         -         -             -  skipped: panels must be aligned, got [370, 400]
```

Neither forward-fills bars that did not trade.

And a `--data-dir` which cannot be read is reported the same way, on a line
saying `unusable`, still on exit 0:

```
uv run python -m stock_rl health --data-dir /tmp/srl-nope
```

```
    unusable          ValueError: /tmp/srl-nope is not a directory
```

To serve the dashboard over the same directory, from `stock_rl/`:

```bash
uv run python -m stock_rl serve --data-dir /tmp/srl-demo-panel --port 8793
```

```
loaded 5 symbol(s) from /tmp/srl-demo-panel
binding 127.0.0.1:8793 -- loopback only, no authentication, not
financial advice
```

Those two lines are on stderr; the server then serves until interrupted.
From another shell:

```bash
curl -s -o /dev/null -w 'health_http=%{http_code}\n' \
  http://127.0.0.1:8793/api/health
```

```
health_http=200
```

`/api/health` then reports `"bars": 3500, "symbols": 5`. Ctrl-C, or `SIGINT`,
shuts the server down cleanly and the process exits **0**. Without
`--data-dir` the dashboard starts empty and says so on `/api/health` rather
than looking for data you did not name.

### Running the experiment on real data

Substitute your own vendor directory. The panel below is the one generated in
the previous section, run through `--data-dir` so the exit code comes from the
verdict rather than from the synthetic override. **The numbers are still
generated prices**, so treat the verdict as a demonstration of the mechanism,
not as a finding about the market:

```bash
uv run python -m stock_rl experiment --data-dir /tmp/srl-demo-panel
```

```
experiment  three-arm A/B, pre-registered kill criteria
panel_kind  vendor-csv/replay
symbols     5
run_at      2026-10-05T02:51:30Z
seeds       1, 2, 3
settings    capital=10000000 rebalance_days=21 holdings=5 max_weight=0.1
            history=60
arms        sharpe mean +/- stdev over seeds, growth multiple
  control        sharpe +0.764 +/- 0.000   growth 1.091
  context        sharpe +0.764 +/- 0.000   growth 1.091
  price_context  sharpe +0.764 +/- 0.000   growth 1.091
benchmark   equal weight 1/N sharpe +0.764
comparisons treatment minus the price-only control
  context_minus_a        dSharpe +0.000  alpha +0.00000  alpha_p 1.000
                         |t| 0.00  folds 0/5  turnover/mo 0.003
  price_context_minus_a  dSharpe +0.000  alpha +0.00000  alpha_p 1.000
                         |t| 0.00  folds 0/5  turnover/mo 0.003
length gate 2.54 years available against 1.25 required
verdict     KILL
exit        1
kill        FAILED. The strings below are the harness's own, verbatim:
  alpha p=1.000 > 0.1: delete all numeric context work
  sharpe p=1.000 > 0.1: numeric context did not beat price-only
  |t|=0.00 <= 3.0: conventional t>2.0 is inadequate after data mining
  0 of 5 folds passed, need 4
$ echo $?
1
```

**Worked example, read the exit code.** `verdict KILL` was printed, the
report's own `exit` line says 1, and the process exited **1**. All three
agree, which is the property the mapping exists to guarantee.

The three arm Sharpes are identical because a random walk gives the context
block nothing to explain, so the paired difference is exactly zero and all
four thresholds fail at once. On real vendor data the arm Sharpes will
differ; the *shape* of the report and the exit-code mapping do not.

The four failure strings are the harness's own, one per line, with no
prefix, no count and no rounding. No "3 of 4 thresholds failed". That is
deliberate. The pre-registered thresholds are the entire point of the
experiment, and a CLI that summarised them could make a near-miss readable as
a pass — the exact failure the harness was built to prevent.

`--data-dir` is mutually exclusive with `--synthetic`, and the CLI refuses
rather than picking one silently:

```bash
uv run python -m stock_rl experiment --synthetic \
  --data-dir /tmp/srl-demo-panel
```

```
error: --synthetic and --data-dir are mutually exclusive: a generated panel and
a vendor export are different claims about where the prices came from, and
--synthetic wins silently is exactly the kind of quiet this project refuses
```

Exit 3.

Flags worth knowing for a real run, all read from `--help`:

- `--shock NODE_KEY` traverses the Nifty-50 dependency graph from a node,
  e.g. `--shock commodity:crude`, and feeds the depth into the context block.
  Without it the graph half of the context is zeros. Verified with
  `commodity:crude`, exit 1 on the generated panel.
- `--seeds 1 2 3` (the default) — each threshold is judged on the *least
  favourable* seed, so a borderline result does not pass on a lucky one.
- `--free-costs` zeroes every charge so a turnover difference cannot be
  blamed on the cost model. DELIVERY costs are the default because that is
  what a real trade pays.
- `--json` prints the report as JSON with the verdict under `"verdict"` and
  the verbatim failures under `"failures"`; `--out PATH` also writes it.
  Both can be combined — that combination exits 1 on the panel above.
- `--bars N` applies to `--synthetic` only, and is ignored when a
  `--data-dir` supplies the panel.
- `--symbols NAME ...` restricts the panel. A symbol the directory does not
  hold exits 3 rather than being dropped:

  ```
  error: no panel available for ['NOPE']; the panel holds ['AAA', 'BBB',
  'CCC', 'DDD', 'EEE']
  ```

### The other three exits, reproduced

Nothing was run at all — no data:

```bash
uv run python -m stock_rl experiment
```

```
error: no panel available: pass --data-dir with one CSV per symbol, or
--synthetic to generate a labelled demo panel. This command will not invent
prices, because a backtest on invented prices is a test of nothing
```

Exit **3**. The CLI will not invent prices.

`INCONCLUSIVE`, exit **2** — the length gate refusing to evaluate anything,
which is not a softer `KILL` but an absence of a claim. The 100-bar panel,
with a warm-up and cadence it can support, so it runs and then declines to
grade the result:

```bash
uv run python -m stock_rl experiment --data-dir /tmp/srl-short-panel \
  --history 20 --rebalance-days 5
```

```
length gate 3 arms claiming Sharpe 0.651 need 1.72 years of returns, only 0.32
are available, so no kill criterion was evaluated
verdict     INCONCLUSIVE
exit        2
kill        no criterion was evaluated, so no claim may be made in either
            direction
```

Exit 2. Compare the `length gate` lines. In the worked example above,
`2.54 years available against 1.25 required` clears the gate and the criteria
get evaluated. Here, three arms claiming Sharpe 0.651 need 1.72 years and only
0.32 exist, so no criterion was evaluated and the report says exactly that
rather than guessing at a verdict.

Two things follow. A shorter warm-up makes it *worse*, not better — the gate
measures return history, so cutting `--history` to 10 and `--rebalance-days`
to 2 asks for 26.93 years on 0.36 available:

```
length gate 3 arms claiming Sharpe 0.164 need 26.93 years of returns, only
0.36 are available, so no kill criterion was evaluated
verdict     INCONCLUSIVE
exit        2
```

Still exit 2. And the same panel at the default `--history 60` never reaches
the gate at all:

```bash
uv run python -m stock_rl experiment --data-dir /tmp/srl-short-panel
```

```
error: the harness refused these panels: series of 40 bars is too short for
min_train=60, horizon=1, embargo=1; need at least 63
```

Exit 3 — too short to measure is not the same as too short to believe.

Unusable arguments, exit **2**. An unknown flag:

```bash
uv run python -m stock_rl experiment --nope
```

```
usage: python -m stock_rl [-h] [--version] COMMAND ...
python -m stock_rl: error: unrecognized arguments: --nope
```

And an unknown subcommand:

```bash
uv run python -m stock_rl bogus
```

```
usage: python -m stock_rl [-h] [--version] COMMAND ...
python -m stock_rl: error: argument COMMAND: invalid choice: 'bogus' (choose
from 'serve', 'experiment', 'baselines', 'health', 'skills')
```

Both exit 2, which is argparse's own usage-error code and the reason
`INCONCLUSIVE` was mapped onto it. One `case 2` branch in a script covers
"you asked wrongly" and "there is not enough history" together.

### A CI step that cannot be fooled

```bash
export PANEL_DIR=/path/to/vendor/csv
uv run python -m stock_rl experiment --data-dir "$PANEL_DIR" \
  --out artifacts/experiment.json
```

`--out` creates `artifacts/` if it does not exist and writes the JSON report
there. On the panel above it exited 1 and left a 3,475-byte
`artifacts/experiment.json`, whose `verdict`, `panel_kind` and `failures` keys
carry the same three facts the text report prints.

The exit codes are the whole contract: 0 only on a KEEP, 1 on a KILL with the
failures printed verbatim, 2 when the series is too short to judge, 3 when
there was no panel — or when someone swapped in `--synthetic` to make it
green. There is no flag that turns a generated panel into a pass, so a
pipeline that goes green has either run a real experiment and kept the
context layer, or it has been changed by somebody who did not read this.

---

## 12. What you do not need

Because there are no runtime dependencies, none of this is required:

- no Docker
- no Kubernetes, kind, Helm, Kafka, Redis, Neo4j, Qdrant or LocalStack
- no GPU
- no broker account or live API credentials
- no network access

`make sync && make check` is the entire local story for testing. Anything that
asks you to install an infrastructure stack before running a unit test is
building a tax you have not agreed to pay.

---

## 13. Related documents

- [research/README.md](research/README.md) — index of the research corpus
- [research/agents/llm-agents-in-finance.md](research/agents/llm-agents-in-finance.md)
  — the evidence the experiment harness is built against
- [research/corrections-to-design-docs.md](research/corrections-to-design-docs.md)
  — claims struck from the design docs, and why
- [research/architecture/project-structure.md](research/architecture/project-structure.md)
  — why this is a monolith and not eleven services
- [research/methodology/deflated-sharpe.md](research/methodology/deflated-sharpe.md)
  — why `metrics.py` is written the way it is
