# stock-rl

Backtest core, RL environments and compliance harness for NSE-listed
equities.

**Status: research code.** Nothing here has traded real money. The backtest
engines are tested and the performance claims are constrained, but no
result from this repository has been validated against live execution.

---

## Read this before anything else

The single most important finding from building this is negative.

On Nifty-50 panel data, **the simple baselines beat the reinforcement
learning agents.** Equal-weight allocation beat the SAC-style portfolio
agent, and HRP beat all four DRL agents tried. That is consistent with
DeMiguel, Garlappi & Uppal (2009), who showed 1/N needs roughly 500 years
of data to be beaten by an estimated optimal portfolio.

FINSABER (KDD 2026) found buy-and-hold beats LLM trading agents at
p ≤ 0.006 across 63–91 unbiased symbols, with no significant alpha
(p > 0.34 for all). Stanford measured multi-agent LLM debate losing to a
plain equal-weight aggregate 67% of the time; the 4-agent/5-round
configuration returned −0.27% at the *highest* reasoning quality setting.

Consequently `CONTEXT_ENABLED = False` and the sentiment/context layer is
gated behind a pre-registered experiment rather than switched on. See
[the agent research][agents].

**Recommendation: run the three-arm A/B test before building the data
fabric.** It costs about two weeks and $40. The alternative is building
the expensive layer first and then having no honest way to conclude that it
did not work.

---

## Zero runtime dependencies

`dependencies = []` in `pyproject.toml`, and that is enforced rather than
merely intended. `tests/test_packaging.py` parses every module in the
package with `ast`, collects every import including ones nested inside
function bodies, and classifies each top-level name against the
interpreter's `sys.stdlib_module_names`.

This replaces what would normally be FastAPI, Pydantic, gymnasium, Stable
Baselines3, NumPy, pandas and an ONNX runtime. That is a real
consequence: the RL agents here are written out longhand, and vectorised
numerics do not exist. It was chosen because the alternative was a
dependency tree for a backtest engine that has to be auditable and has to
produce the same number twice.

### Two things this test suite got wrong first

Both are recorded because a check that cannot fail is worse than no check,
since it gets read as evidence.

- **Classifying by `sysconfig` path.** On Anaconda, Debian and several
  other layouts `site-packages` sits *inside* the directory the interpreter
  reports as its standard library. Every third-party package was
  classified as stdlib and the suite passed unconditionally.
- **Diffing `sys.modules` around a reload.** This depends on import order
  across the whole test session. A module containing only
  `import pylint.lint` went undetected, because pylint was already loaded
  by an earlier test so the set difference was empty.

Static analysis has neither problem. Both approaches are caught by
`TestClassifierIsNotVacuous`, which asserts that `pylint`, `numpy`,
`pandas` and `fastapi` are rejected.

---

## Quickstart

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```sh
cd stock_rl
make sync     # uv sync
make test     # pytest with coverage, gate is 90%
make lint     # pylint against ../.pylintrc
make check    # lint + test. Run this before every push.
```

### Running it

`python -m stock_rl` reaches every module: `serve`, `experiment`,
`baselines`, `health`, `skills`.

```sh
uv run python -m stock_rl --help
uv run python -m stock_rl health
uv run python -m stock_rl experiment --data-dir /path/to/vendor/csv
```

The **exit code is the product** — the harness verdict becomes a process
status:

| Exit | Meaning |
|---|---|
| 0 | `KEEP`, every pre-registered criterion cleared |
| 1 | `KILL`, at least one failed; failures printed verbatim |
| 2 | `INCONCLUSIVE`, or an argparse usage error |
| 3 | nothing ran: no data, or a `--synthetic` panel |

`--synthetic` **always exits 3**, whatever the verdict: a KILL on generated
prices shows the harness discriminates, and a KEEP would prove nothing. A CI
step cannot read a synthetic panel as a pass. Details and worked output in
[the testing guide](docs/LOCAL-TESTING-GUIDE.md).

### The one trap that will waste your afternoon

**pylint does not search parent directories for a configuration file.**

`.pylintrc` lives at the repository root, one level above `stock_rl/`. If
you run pylint without `--rcfile=../.pylintrc`, it silently falls back to
its own defaults, which assume 4-space indentation. This codebase is
2-space, so pylint then flags *every correctly indented line in the
project* as `bad-indentation`. The errors look like the code is wrong.
The code is fine; the config was missing.

`make lint` passes the flag for you. If you invoke pylint by hand, copy
the flag out of the `Makefile`.

Two more conventions the root `.pylintrc` enforces:

- `variable-rgx=^[a-z][a-z0-9_]*$` rejects UPPERCASE for ordinary names, so
  **functions and variables are lowercase**. Two conventions are deliberate
  and permitted by `const-rgx`: **module constants are UPPERCASE**
  (`costs.DELIVERY`, `metrics.TRADING_DAYS_PER_YEAR`,
  `context.CONTEXT_ENABLED`) and **type aliases are CamelCase**
  (`gym.Obs`, `gym.Info`, `killswitch.Clock`, `policy.Action`). An earlier
  version of this file claimed module constants were lowercase, which was
  simply wrong — it described a convention the code never used. Enum
  *members* are uppercase because `class-const-rgx` demands it
  (`costs.Side.BUY`).
- `bad-indentation` is disabled precisely because the built-in check
  assumes 4-space blocks and cannot be configured for 2.

---

## Layout

| Module | What it is |
|---|---|
| `bars.py`, `engine.py` | Bar series and the single-assect backtest loop |
| `costs.py` | Indian transaction costs, delivery and intraday |
| `metrics.py` | Sharpe, Sortino, drawdown, Deflated Sharpe |
| `splits.py` | Purged walk-forward folds with an embargo |
| `indicators.py` | RSI, ATR, moving averages, with warm-up alignment |
| `portfolio.py` | Cross-sectional backtester; every strategy is measured here |
| `baselines.py` | 1/N, buy-and-hold, 12-1 momentum, inverse-vol |
| `weights.py` | Weight clamping and the affordability constraint |
| `env/` | `gym`-style Protocols and the NSE portfolio environment |
| `rl/` | Policies, portfolio and hedging environments, training harness |
| `risk/` | NSE para 11.1 pre-trade checks, circuits, kill switch, VaR/CVaR |
| `execution/` | Position sizing, broker abstraction, hash-chained audit log |
| `compliance/` | SEBI/NSE retention, rate limits, robots, algo tagging |
| `data/` | Trading calendar and market-data vendors |
| `graph/`, `context/`, `sentiment/` | Dependency graph, context, sentiment |
| `skills/` | Agents as gated markdown files |
| `api.py`, `web/` | Stdlib HTTP API and a dependency-free dashboard |
| `experiment/` | Three-arm A/B harness with pre-registered kill criteria |
| `decisions.py` | BUY/HOLD/SELL with timeframe and ordered reasons |

Measured at the last full run: **1,622 tests**, 51 modules, pylint 10.00/10.

---

## Design decisions that contradict the original design docs

Research overrode the design docs wherever they conflicted. Full record in
[the corrections log][corrections].

**Fabricated sources removed.** `InvestingNote` and `Indian Express v.
Zeal` do not exist. "7 years per SEBI" is wrong: the real figures are 5
years (NSE para 10.3) and 8 years (SEBI Stock Brokers Regulations 2026
reg. 16).

**Ten microservices on EKS rejected.** The stated cost was roughly $15,000
per month; the floor is about **$1,680 per month**, and EKS cannot scale to
zero, so that floor is permanent. This is one VM and a modular monolith.

**Multi-agent LLM debate rejected as harmful.** See the top of this file.

**SAC cannot take discrete actions,** so the hedge action space is
continuous coverage ratios over a predefined instrument set.

**Constraints live outside the policy.** A cvxpy projection inside an actor
is non-differentiable and silently corrupts the policy gradient.

**Per-step drawdown penalty is off by default.** It rewards the doubling
strategy.

**Deflated Sharpe was wrong three ways** and now reproduces Bailey &
López de Prado's published values (0.900397 / 0.950502 / 0.950491). The
denominator is `1 − skew·sr + ((kurtosis−1)/4)·sr²`; the Euler-Mascheroni
constant does not belong there, and `sqrt(T−1)` counts observations, not
years.

**Per-decision rationale is not a SEBI requirement.** The actual driver of
documentation is NSE para 9.1/9.9: re-registration is required when
strategy *logic* changes. See
[the attribution research][attribution].

---

## What is deliberately not built

Naming these is the point; an unstated omission reads as an oversight.

- **No real broker integration, credentials or network calls.**
- **No authentication on the API.** It binds to 127.0.0.1 by default. The
  intended replacement is a TLS-terminating proxy plus client certificate,
  alongside the existing Algo-ID header. Documented in the module docstring.
- **No Neo4j, Qdrant, embeddings or vector store.** ~300–2000 static edges
  justify a JSON graph.
- **No LangGraph, no orchestrator, no LLM judge.** An LLM judge is
  strictly worse than a plain mean, per the same research.
- **No sentiment model.** FinBERT-class signals measured *inverted*
  standalone (−0.37% next-day). Sentiment is an injected input requiring a
  minimum of two sources, not a model.
- **No per-client MWPL checks.** Needs market-cap reference data this
  project does not hold.
- **2 of the 16 para 11.1 checks omitted** — net-position-vs-margin is the
  clearing member's job in production, and efficient-price-discovery is a
  commodity-segment control. Both named in the module docstring.

---

## The cost, honestly

Where the design docs quote a figure, these are the real ones:

| Item | Design doc claim | Actual |
|---|---|---|
| 4× A100 nightly training | ₹45,000/mo | **≈ $6,785/mo (₹5.7 lakh)** |
| EKS + infrastructure | ₹15,000/mo | **≈ $1,680/mo**, cannot scale to zero |
| Full plan | — | **≈ ₹7 lakh/month** |

For a business decision, the relevant comparison is that the entire
three-arm experiment costs about **$40**. It should be run first.

---

## Bugs this repository found in itself

Recorded because a backtest that is wrong in your favour is worse than one
that crashes. All were caught by tests, not by inspection.

- **Momentum computed the negative of momentum.** The 12-1 score read
  `old/recent` instead of `recent/old`, so the "strongest anomaly family in
  Indian data" baseline bought the biggest loser in the universe. This
  contaminated `MomentumPolicy` and `trend_filtered_momentum`, so every RL
  comparison against them was testing the opposite hypothesis to the one
  documented.
- **The embargo was validated and then never applied.** Every value
  produced identical folds, and serial correlation leaked across every
  boundary — the one thing the parameter exists to prevent.
- **The hedging error was offset by exactly +1.0** for every policy, which
  collapsed `semi_rmse` to zero and handed the *never-hedged* book a
  better terminal reward than a perfect hedge.
- **The RL reward paid for not hedging.** `cvar` is signed and negative
  while `semi_rmse` is a non-negative semideviation; a single subtract
  applied to both rewarded whichever was negative. Under `cvar` a book that
  never hedged scored best of all.
- **A NaN weight became a maximum-weight position,** because
  `max(0.0, min(0.1, nan))` returns `0.1`. A policy that divided by zero
  silently received the largest permitted position.
- **Cash went negative** in all three engines, which is unpriced borrowing:
  invisible on a flat series, free leverage on a rising one.
- **The DP charge was levied at zero notional**, and the exchange rate
  went stale within a day.

`tests/test_review_findings.py` is the adversarial review that found these.

[agents]: docs/research/agents/llm-agents-in-finance.md
[corrections]: docs/research/corrections-to-design-docs.md
[attribution]: docs/research/compliance/signal-attribution-and-audit-trail.md

