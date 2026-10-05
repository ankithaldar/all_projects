# MEMORY.md

State of `stock_rl` as of commit `23c23f6` on `origin/stock-rl/main`.
**`make check` is FULLY GREEN: 2100 passed, pylint 10.00/10, coverage 96.94%.**

## What this project is

A backtest core, RL environments and compliance harness for NSE-listed
equities. **Research code — nothing here has traded real money.**

## THE HEADLINE FINDING (do not lose this)

On this data **the simple baselines beat the RL agents.** Equal-weight beat
the SAC-style agent; HRP beat all four DRL agents. FINSABER (KDD 2026) found
buy-and-hold beats LLM agents at p<=0.006. Stanford measured multi-agent
debate losing to equal-weight 67% of the time.

Therefore `CONTEXT_ENABLED = False` and the sentiment/context layer is gated
behind a pre-registered experiment, not switched on.

**Recommendation to the user: run the three-arm A/B test before building the
data fabric.** It costs ~2 weeks and ~$40.

## Hard constraints (violating these breaks the build)

- ZERO runtime dependencies. `dependencies = []` in pyproject.toml, enforced
  by `tests/test_packaging.py` (AST scan, classifies against
  `sys.stdlib_module_names`).
- Line 1 exactly `#!/usr/bin/env python`, line 2 exactly
  `# -*- coding: utf-8 -*-`. The `#` is required.
- 2-space indent, NEVER tabs or 4 spaces. Single quotes. Google docstrings.
  Max 80 cols. Python 3.14+.
- ALL names lowercase_with_underscores **including module constants** —
  `.pylintrc` has `variable-rgx=^[a-z][a-z0-9_]*$`. Enum members excepted.
- **pylint does not search parent dirs.** `make lint` must pass
  `--rcfile=../.pylintrc` (the Makefile does) or it assumes 4-space indent and
  flags every correct line. A wrong rcfile path exits 32, loudly.
- Every file ends with a newline. The write tool strips it; pylint C0304.
- Deliberate simplifications carry a `# ponytail:` comment naming the ceiling
  and the upgrade path.

## Commands

```sh
cd stock_rl
make check          # lint + test. Run before every push.
make lint           # pylint --rcfile=../.pylintrc src/stock_rl tests
make test           # pytest --cov, gate is 90%
uv run python -m stock_rl --help    # CLI exists as of 4e3e198
```

CLI exit codes: `0` KEEP, `1` KILL (failures printed verbatim), `2`
INCONCLUSIVE, `3` nothing ran / `--synthetic` (a synthetic verdict is never
evidence, whatever it says).

## Architecture decisions that contradict the original design docs

Full record: `docs/research/corrections-to-design-docs.md`.

- Research OVERRODE the design docs wherever they conflicted.
- 10 microservices on EKS rejected: ~$1,680/mo floor (not $15k), and EKS
  cannot scale to zero. One VM, modular monolith.
- Multi-agent LLM debate rejected as harmful.
- SAC cannot take Discrete actions; hedge space is continuous coverage ratios.
- Constraints enforced OUTSIDE the policy (cvxpy inside an actor is
  non-differentiable and silently corrupts the gradient).
- Per-step drawdown penalty OFF by default — it rewards doubling.
- Fabricated sources removed: "InvestingNote", "Indian Express v. Zeal",
  a 2021 SEBI algo circular. Retention is 8yr (SEBI Reg.16) / 5yr (NSE
  10.3), **not** "7 years per SEBI".
- Per-decision rationale is NOT a SEBI requirement. The real driver is NSE
  paras 9.1/9.9 re-registration on logic change.
- Real cost: ~₹7 lakh/month total (4×A100 ≈ $6,785/mo). Not ₹45k.

## Design decisions taken on the user's instruction

**Entity resolution is a TABLE LOOKUP, not inference** (commit `30df4ad`).
The registry is always a mapping (`SymbolRecord('AXISBANK', 'Axis Bank',
('Axis',))`); the old code ignored it and derived identity from strings by
tokenising + prefix rules, which resolved `AXISBANKING`→AXISBANK,
`ITC-INFRA`→ITC, `inf`→INFY. Now: index every written form at construction,
resolve with one dict hit. `'ongc oil'` and `'Tata Consultancy'` are now
explicit table rows, not inferred. Behaviour changes: `'RELI'` is UNKNOWN
(not AMBIGUOUS); `'RE'` has no prefix floor; a shared form is AMBIGUOUS at
resolve time.

## Bugs this repo found in itself

All caught by tests. A clean pylint score and high coverage are NOT evidence —
`fuse.py` sat at 100% statement AND 100% branch coverage while
`evaluate_kill_criteria` returned `passed=True` for all-NaN input.

- Momentum computed the **negative** of momentum — bought the biggest loser.
- The embargo was validated then never applied.
- Hedging error offset by exactly +1.0; RL reward **paid for not hedging**.
- A NaN weight became a maximum-weight position (`min(cap, nan)` == cap).
- Cash went negative in all three engines.
- The **dashboard rendered nothing**: `/style.css` and `/app.js` had no route.
  173 tests, 99% coverage, 10.00/10 pylint.
- `cycles()` hung 342s on a 333-node **acyclic** graph inside its own budget.
- An empty book reported Sharpe 0.0 / growth 1.0 — byte-identical to a real
  backtest that returned nothing.
- A scheduled `reset('ack')` cleared a live drawdown halt.
- The sizer overspent the account (third engine with this bug).

## Current state

- **`make check` green at 3291 passed + 31 subtests**, pylint 10.00/10,
  coverage 95.91-97.50%, `dependencies = []`.
- 47 indicator functions in 4 modules (`indicators_trend/oscillators/
  volatility/volume.py`) + `strategies.py`. All length-preserving, all
  proven look-ahead-free by TRUNCATION, 100% branch coverage on each.
- `api.py` (2892 lines) split into `api/` — 13 modules, largest 595 lines,
  public surface byte-identical, all 81 definitions AST-identical.
- `logging_support.py` landed (897 lines, 130 tests) but is **NOT yet wired
  into anything**. See the follow-up list.
- CLI works: `python -m stock_rl`.

## Naming conventions - CORRECTED, the README was wrong

Module CONSTANTS are **UPPERCASE** (`costs.DELIVERY`,
`metrics.TRADING_DAYS_PER_YEAR`, `context.CONTEXT_ENABLED`) and type ALIASES
are **CamelCase** (`gym.Obs`, `gym.Info`, `killswitch.Clock`,
`policy.Action`). `const-rgx` accepts both; the build was always green.
An earlier README claimed constants were lowercase and an audit test
enforced that invented rule - both now pin what the code actually does.

## Three bugs found by the api refactor, reported NOT fixed (do next)

Found while splitting `api.py`. All three are pre-existing; the refactor was
deliberately behaviour-preserving so they are documented, not patched.

1. **`GET /api/baselines?capital=0` returns 200 with the defaults instead of
   400.** In `_control_of` the parsed value is combined with `or`, and `0` /
   `0.0` is falsy, so the validator never sees it. `?max_weight=0`,
   `?rebalance_days=0`, `?history=0` behave the same way. The POST body path
   refuses correctly (`{"capital": 0}` -> 400). This is exactly the
   disagreement `_control_of`'s own docstring claims the two endpoints
   cannot have. Now `api/validate.py:_control_of`.
2. **`ApiService.positions` treats a supplied `cash == 0.0` as absent**:
   `state.cash if state.cash else total - held_value`. A fully-invested book
   with a real zero balance gets a synthetic residual. Now `api/service.py`.
3. **`serve()` logs an unusable IPv6 URL**: no brackets, so an IPv6 bind
   logs `http://::1:8765/`. `bound_address()` unpacks IPv6 correctly one
   function above. Now `api/server.py`.

## Fixed, do not re-open

- **Partial weight mappings.** `run_portfolio` read an omitted symbol as
  0.0 (exit), `WeightAllocationEnv._target_weights` read it as hold. Both now
  read one shared constant `portfolio.omitted_weight`. **Two** providers
  could emit partials, not one - `low_volatility` was the second, dropping
  symbols with zero realised volatility. The 2.6x cost gap on
  `trend_filtered_momentum` between backtester and env closed to zero.
- **`max_weight` never reached the baseline.** Providers imposed their own
  0.10 default, so 0.10 and 0.30 gave byte-identical Sharpe (`pipeline`
  2.368442787274592 twice, `cli` 3.3955 twice). Fixed at all three seams -
  `api`, `pipeline`, `cli` - by binding the cap BY KEYWORD and reporting
  `applied_max_weight` measured off the book. Shared helpers now live in
  `weights.py`: `bind_weight_cap`, `accepts_weight_cap`,
  `applied_weight_cap`, `CAP_KEYWORD`.
  **CORRECTION: the commit message for `5a38221` says "61 tests". It is 121**
  (`tests/test_cap_binding.py`, verified by `--collect-only`). The number was
  copied from the wrong agent's report. Do not trust that commit message on
  this detail.
  **The cap is a CEILING, not a target.** `applied <= reported` always, and
  above 1/N the construction binds so applied drops strictly below
  reported: at cap 0.30 with 4 symbols, four of five baselines report 0.30
  but build at 0.25, while `low_volatility` does reach 0.30. A test asserting
  reported == applied would be WRONG - it would encode target semantics.
- **The Rink/India doc error.** `technical-indicators-nse.md` said twice
  that Rink (2023) excluded India. His Table 3 Panel B lists `IND`, BSE
  Sensex, 1979-2016. Corrected both places; the burden of proof is now
  HEAVIER, not lighter.

## Process lessons (expensive, do not relearn)

1. **Never `git checkout --` a file you have edited but not committed.** I
   lost the entire entity-linker lookup that way and both mis-resolution bugs
   came straight back.
2. **Do not dispatch two agents to the same files.** I assigned
   `pipeline.py` and `cli.py` twice without checking whether the first agent
   was still running. One agent's work was overwritten twice and it correctly
   stopped rather than fight. Check for in-flight agents first.
3. **Oversized prompts kill agents silently.** Three agents returned no text
   and wrote nothing. Scope each to one or two files.
4. "Agent completed without a response" does NOT mean it did nothing — one
   wrote its file but never reported. Check `git status` after each.
5. A defect test that asserts the CORRECT behaviour passes while the bug is
   present. Assert on the docstring TEXT with the live computation as oracle.
6. Use `.venv/bin/python`, not `python3`, for AST sweeps — 3.11 misreads PEP
   701 nested same-quote f-strings as IndentationError.

## Next steps

1. **Fix the three api bugs** above (falsy-`or` validation, `cash == 0.0`,
   IPv6 log line). Small and now documented.
2. **Wire `logging_support` into `cli.py`.** The facility exists and nothing
   calls it. The valuable call sites are around `run_portfolio` and
   `run_experiment` — that is where a wrong Sharpe currently gives no clue.
   Leave `_fail()` as stderr prose: a CLI error must reach the user
   regardless of level.
3. **Deploy scripts for AWS and GCP** (user-requested, never started).
4. **AWS/GCP deployment docs on README.md** (user-requested, never started).
5. **Remaining monoliths**: `pipeline.py` (1722), `experiment/harness.py`
   (1380), `decisions.py` (1141). `api.py` is done and shows the pattern:
   record `__all__` and AST-compare definitions before and after.
6. **Frontend dependency graph with news** — `graph/news.py` and
   `web/graph.js` are landing; the join maps a commodity node to news via
   its DEPENDENTS, never by fuzzy-matching a headline to a commodity name.
7. **Real work still undone**: no licensed NSE data wired up; the 3-arm A/B
   experiment has only ever run on synthetic panels, so **no conclusion
   about the strategy is supported yet**. Crash simulation for RL training
   is under research.

## The live negative result, and why it matters

Four independent lines now agree that this project's edge is not visible in
any test available here:

| Source | Finding |
|---|---|
| backtest | equal weight beat the SAC agent; HRP beat all four DRL agents |
| research (trend) | momentum strong GROSS of costs, absent NET |
| research (reversion) | RSI 30/70 never Indian-validated; the one corrected test FAILED it at p=0.1392 |
| strategies | 4/5 LOSE to equal weight once drift is removed; 15-seed sweep is a coin flip (4/15, 5/15, 7/15, 7/15) |

Likely reason: liquid momentum returns 8.51% net against the Nifty 50's own
10.41%. The alpha is in illiquid names; this book trades liquid constituents.
Also: SEBI replaced the close with a closing auction on 3 Aug 2026, so any
backtest spanning it mixes two close definitions.


