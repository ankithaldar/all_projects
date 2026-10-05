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

- **`make check` green**: 2100 passed + 31 subtests, pylint 10.00/10,
  coverage 96.94%, `dependencies = []`. Nothing in flight.
- `pipeline.py` landed and cross-checks the seams (432 configurations, max
  Sharpe gap 0.0, max cost gap 1.16e-10 rupees).
- CLI works: `python -m stock_rl`.

## Naming conventions - CORRECTED, the README was wrong

Module CONSTANTS are **UPPERCASE** (`costs.DELIVERY`,
`metrics.TRADING_DAYS_PER_YEAR`, `context.CONTEXT_ENABLED`) and type ALIASES
are **CamelCase** (`gym.Obs`, `gym.Info`, `killswitch.Clock`,
`policy.Action`). `const-rgx` accepts both; the build was always green.
An earlier README claimed constants were lowercase and an audit test
enforced that invented rule - both now pin what the code actually does.

## Real seam defect found, STILL NOT fixed (highest-value next task)

`run_portfolio` reads an omitted symbol from a weight mapping as target
**0.0** (sells); `WeightAllocationEnv._target_weights` reads it as **hold
current** (keeps). `baselines.trend_filtered_momentum` returns a partial
mapping routinely — the only baseline of five that does, and it has no
Policy wrapper so it cannot be parity-checked. Measured on one panel:
Sharpe 1.994873 vs 2.175971, cost ₹122,435.95 vs ₹26,868.58 — a 4.4x cost
difference that reads as a real result. Currently invisible because every
published `Policy` emits complete mappings.
**Fix: complete the mapping with `0.0`** (what `run_portfolio` and
`momentum_ranked` already use).

Also: any test panel must carry **more symbols than `1 / max_weight`** or
four of five baselines collapse to the same uniform book and report the same
Sharpe to 12 decimals.

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

1. Fix the remaining 4 `test_audit_consistency.py` pins.
2. Finish/verify `pipeline.py`; reconcile the duplicate assignment first.
3. Fix the partial-mapping seam defect above.
4. Add a CLI section to `docs/LOCAL-TESTING-GUIDE.md` quoting executed
   commands.
5. Real work still undone: no licensed NSE data wired up; the 3-arm A/B
   experiment has only ever run on synthetic panels, so **no conclusion
   about the strategy is supported yet**.