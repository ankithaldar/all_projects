# Format Loop Report — cat_game_rl

Date: 2026-11-06
Project: `cat_game_rl/`
Status: Loop complete. Final QA (cycle 3): PASS — 0 issues.
Owner of this report: docs-engineer (report-only; no code, `.opencode/`,
or `.pylintrc` edits).

This file is the committed record of the format loop. The working
blackboard (`.opencode/plans/`, `.opencode/critique-log/`,
`.opencode/flaw-reports/`, `.opencode/context/decisions.md`) is
gitignored (`/.opencode/` in `.gitignore:209`), so it will not commit.
Everything needed to re-verify the loop is reproduced or cited here.

## 1. What cat_game_rl does

cat_game_rl is a reinforcement-learning agent plus a genetic-algorithm
(GA) baseline for optimizing batch production schedules in a 23-item
crafting game. Source: `cat_game_rl/README.md:1-131`.

- **Domain core** (`cat_game_rl/src/core/`): 23 items (4 base:
  cotton/tree/rock/quartz + 19 craftable), inventory/stash, coin
  economy (210 coins per 5-minute tick), batch cost
  `init_cost * (1 + 0.5*(n-1))`, one exclusive manufacturing slot per
  craftable item, slot scheduling, target provider. Goal: meet level
  targets within a 30-day horizon. See `cat_game_rl/README.md:121-131`
  (game rules) and `cat_game_rl/README.md:106-119` (project structure).
- **Environment + agent** (`cat_game_rl/src/cat_game_env/`,
  `cat_game_rl/src/agent/`): Gymnasium crafting environment,
  observation builder, reward shaper, action handler/mask, frame
  skipper, multi-agent wrapper; MaskablePPO agent with action masking
  and callbacks.
- **GA baseline** (`cat_game_rl/src/ga/`): NSGA-II genetic algorithm
  (chromosome, operators, GA scheduler) for multi-objective
  comparison against RL (cost vs. utilization vs. target completion).
- **Dashboard + logging** (`cat_game_rl/src/dashboard/`,
  `cat_game_rl/src/logging_util/`): Streamlit dashboard (simulation
  scrubber, bottleneck heatmaps, RL vs GA comparison, Pareto front,
  target editor, CSV replay export); rotating gzip-compressed
  inventory logger.
- **Scripts + tests** (`cat_game_rl/scripts/`, `cat_game_rl/tests/`):
  entry points `train.py`, `train_multi_agent.py`, `evaluate.py`,
  `run_ga.py`, `run_baselines.py`, `launch_dashboard.py`; pytest
  suite covering core, env, GA, agent, dashboard, and logging units.

This loop did **not** change game logic, RL/GA behavior, APIs, or
config semantics. It brought every Python file to the strict
formatting spec while keeping behavior equivalent.

## 2. Strict formatting spec (binding)

Derived from the user spec plus `.pylintrc` (untouched, see §9).
Normative rules:

1. **File header (every `.py` file, in order):**
   - Line 1: shebang — final standard `#!/usr/bin/env python`
     (see ADR-001 note in §7; the original plan draft said `python3`,
     the accepted ADR records `python`).
   - Line 2: `# -*- coding: utf-8 -*-`
   - Immediately after: a module docstring using triple single
     quotes `'''...'''` (not `"""`), Google style. Verified example:
     `cat_game_rl/src/core/items.py:1-4`,
     `cat_game_rl/src/ga/ga_scheduler.py:1-4`.
2. **Google docstrings:** module, class, and public-function
   docstrings in Google style (summary line, blank line,
   `Args:` / `Returns:` / `Raises:` sections as applicable). Private
   helpers get a one-line summary. No NumPy/Sphinx style. Respect
   `.pylintrc` `docstring-min-length=12` (`.pylintrc:211`) and
   `no-docstring-rgx` exemptions (`.pylintrc:207`).
3. **Indentation:** 2 spaces per level (`.pylintrc:262`
   `indent-string='  '`), `indent-after-paren=4` (`.pylintrc:265`).
   No tabs. No 4-space blocks.
4. **Quotes:** single quotes for all strings, including docstrings
   (`'''`). `.pylintrc:281` `check-quote-consistency=yes`. Escape
   apostrophes (`\'`) rather than switching delimiter; no f-string
   prefix changes beyond quotes.
5. **`.pylintrc` untouched:** zero edits (hash must not change).
   Compliance by fixing code, never by disabling checks in config.
6. **SOLID:** no behavior change; reformat must not merge classes,
   move boundaries, or introduce god objects / circular imports.
7. **TDD:** run covering pytest first (RED baseline), reformat,
   re-run (GREEN). Full-suite GREEN at the end of each task.
8. **Pylint target:** `pylint` with repo `.pylintrc` reports
   **0 issues** (no C/R/W/E/F) and 10.00/10 on the project. Line
   length 80 (`.pylintrc:241` `max-line-length=80`). `single-line-if-stmt=yes`
   (`.pylintrc:253`) is allowed by config.

## 3. Baseline (start of loop)

Per Lead handoff and the plan
(`.opencode/plans/cat_game_rl-format-loop.md:82-87`):

- pylint score: **9.03/10**.
- Every `.py` file: **NOSHEBANG** (no shebang line).
- **Double quotes prevalent**; module docstrings missing or
  `"""`-delimited.
- **6 GA failures** in the pre-loop test run (Lead handoff; root
  cause addressed by the debugger fix in §5).
- Coverage **59.67%** at baseline (Lead handoff), below the 90%
  gate in `cat_game_rl/pyproject.toml:31`
  (`--cov-fail-under=90`).

Note on verification: the 9.03 / NOSHEBANG / double-quote items are
recorded in the plan file; the 6 GA failures and 59.67% coverage
figures are Lead-handoff baseline values. Docs-engineer did not
re-run the pre-loop baseline (no shell available in this session);
they are reported as handoff facts, not independently re-derived.

## 4. Worker tasks (4, file-disjoint, parallel-ready)

Each task owned ONLY its listed paths; no agent edited outside its
Owns. Source: plan §4 (`.opencode/plans/cat_game_rl-format-loop.md:89-158`).

| Task | Owner | Owns | Tests run (target GREEN) |
|---|---|---|---|
| A — core | worker-core | `cat_game_rl/src/core/**` (7 files: `__init__.py`, `coin_generator.py`, `cost_calculator.py`, `inventory.py`, `items.py`, `slot_scheduler.py`, `target_provider.py`) | `test_items.py`, `test_inventory.py`, `test_cost_calculator.py`, `test_coin_generator.py`, `test_slot_scheduler.py`, `test_target_provider.py` |
| B — env + agent | worker-backend | `cat_game_rl/src/cat_game_env/**` + `cat_game_rl/src/agent/**` (11 files) | `test_crafting_env.py`, `test_reward_shaper.py`, `test_observation_builder.py`, `test_action_handler.py`, `test_action_mask.py`, `test_frame_skipper.py` |
| C — GA + dashboard + logging | worker-ga-dash | `cat_game_rl/src/ga/**`, `cat_game_rl/src/dashboard/**`, `cat_game_rl/src/logging_util/**` | `test_chromosome.py`, `test_ga_scheduler.py`, `test_inventory_logger.py` |
| D — scripts + tests | worker-scripts-tests | `cat_game_rl/scripts/**` + `cat_game_rl/tests/**` (6 scripts, 20 `test_*.py` files plus `__init__.py`/`conftest.py`) | full suite |

Out of scope for all tasks: `python_proj_template/`, `.venv/`,
`output/`, `config/`, `.pylintrc`, any non-`.py` file.

Handoff evidence for Task B (file-disjoint scope kept):
`.opencode/handoffs/fix-env-agent.md` records 11 files changed in
`cat_game_rl/src/cat_game_env` + `cat_game_rl/src/agent` only,
pylint 10.00/10, named pytest 90 passed, with 6 failures in
GA tests outside its owned dirs during parallel work (later closed
by the debugger fix, §5).

## 5. Debugger fix (GA failures)

The 6 baseline GA failures were root-caused and fixed by the
debugger (Lead handoff summary; final-state locations verified in
current code below). Four fix themes:

1. **Selection operator:** `selTournamentDCD` → `selNSGA2`.
   Final state verified: `cat_game_rl/src/ga/ga_scheduler.py:96`
   registers `tools.selNSGA2` as `select`. No `selTournamentDCD`
   remains in `cat_game_rl/src`/`scripts`/`tests` (only a research
   note mentions it: `cat_game_rl/agents/research/SKILLS.md:16`).
2. **ParetoFront:** constructed with a `similar=` comparator.
   Final state verified: `cat_game_rl/src/ga/ga_scheduler.py:146`
   `tools.ParetoFront(similar=_individuals_equal)`, and the
   `_evaluate` → `Chromosome.evaluate` path at
   `cat_game_rl/src/ga/ga_scheduler.py:109-112`.
3. **Waste test / `initial_coins`:** waste/evaluate assertions now
   pass an explicit `initial_coins` value. Final state verified:
   `Chromosome.evaluate(..., initial_coins: int = 0, ...)` at
   `cat_game_rl/src/ga/chromosome.py:67-92`, with `CoinGenerator(initial_coins)`
   at `cat_game_rl/src/ga/chromosome.py:92` and the GA scheduler
   forwarding `self._initial_coins` at
   `cat_game_rl/src/ga/ga_scheduler.py:111`.
4. **Shape / `MAX_TICKS`:** gene-array shape assertions use the
   shared constant `MAX_TICKS = 8064`
   (`cat_game_rl/src/ga/chromosome.py:22`). Final state verified:
   `genes.shape == (MAX_TICKS, 19)` in
   `cat_game_rl/tests/test_chromosome.py:24`, `(MAX_TICKS, NUM_CRAFTABLE)`
   in `cat_game_rl/tests/test_ga_operators.py:45-46`, and
   `best.shape == (MAX_TICKS, 19)` in
   `cat_game_rl/tests/test_ga_scheduler.py:114`.

## 6. Tester work (new tests, coverage)

Tester added new test files (Lead handoff: **5 files**). Current
`cat_game_rl/tests/` contains 20 `test_*.py` files; the
coverage-driving additions visible in the final tree include:

- `cat_game_rl/tests/test_coverage_extra.py` (uses `MAX_TICKS`
  shape + `initial_coins=100000` evaluation paths, e.g. lines
  36, 65-70)
- `cat_game_rl/tests/test_dashboard_pages.py`
- `cat_game_rl/tests/test_dashboard_utils.py`
- `cat_game_rl/tests/test_agent_callbacks.py`
- `cat_game_rl/tests/test_ga_operators.py`

Coverage rose from the 59.67% baseline to **97.26%**
(final QA, §8), above the 90% gate
(`cat_game_rl/pyproject.toml:31`). Lowest per-file coverage at
final QA: `src/cat_game_env/multi_agent.py` 91% and
`src/dashboard/app.py` 91% — both still above the gate in aggregate.

Note on verification: the exact 5-file list is a Lead-handoff
fact; docs-engineer verified the 20-file final tree and the
coverage result, not the per-file diff history.

## 7. Gate cycles

### Cycle 1 — REJECTED (arch, reviewer) / APPROVED with mediums (security)

- **Arch REJECTED** — two drivers
  (`.opencode/critique-log/cat_game_rl-cycle-1-arch.md`):
  - A1 shebang split: mixed `#!/usr/bin/env python` and
    `#!/usr/bin/env python3` across the repo.
  - A2 pylint suppressions: Lead-reported 10 disables across
    `scripts/`, `src/dashboard/app.py`, `tests/`, and
    `src/cat_game_env/multi_agent.py`.
- **Reviewer REJECTED** — findings R1–R4, all pre-existing
  (`.opencode/critique-log/cat_game_rl-cycle-1-reviewer.md`).
  Deferred with no code change (see §9 below).
- **Security APPROVED (with mediums)** — S1–S5 mediums
  (`.opencode/critique-log/cat_game_rl-cycle-1-security.md`).
  Deferred under the local trusted-operator threat model (see
  §9 below).
- Dispositions recorded as ADR-001–ADR-005 in
  `.opencode/context/decisions.md` (shebang standard; suppressions
  disclosed; reviewer/security deferred, flag-only; 0-issue
  definition). Key ADRs:
  - **ADR-001:** standard shebang `#!/usr/bin/env python` for all
    Python sources, scripts, and tests.
  - **ADR-002:** pylint suppression inventory (disclosed, counted).
  - **ADR-005:** 0-issue = pylint 10/10 WITH disclosed suppressions
    + tests green + coverage >= 90%.

### Cycle 2 — fixes applied, gates APPROVED, 3 minor QA fails

Cycle-2 fixes (per gate logs):
- **Shebang normalized:** 62/62 files identical
  `#!/usr/bin/env python` (was split python/python3).
- **`'''` docstrings:** `"""` count 0 (all module docstrings now
  triple-single-quote).
- **2-space indent + single quotes:** 0 tabs, 0 double-quote
  lines, 1789 single-quote lines.

Gate verdicts:
- **Arch APPROVED** (cycle 2)
  (`.opencode/critique-log/cat_game_rl-cycle-2-arch.md`): shebang
  62/62, `"""` count 0, suppressions 15 lines inventoried,
  boundaries intact, 4 low ADR-sync nits (non-blocking; no
  nit-by-nit text was supplied, none invented here).
- **Reviewer APPROVED** (cycle 2)
  (`.opencode/critique-log/cat_game_rl-cycle-2-reviewer.md`): no
  new YAGNI; R1–R4 remain deferred per ADR-003.
- **QA final-qa-c2** (`.opencode/flaw-reports/cat_game_rl-flaws.md`,
  cycle-2 section): pylint 10.00/10 exit 0; 318 passed; coverage
  97.26%; spec greps PASS except **3 minor fails** (F1–F3):
  - F1 `tests/test_crafting_env.py:236` — 100 chars (line split
    needed).
  - F2 `tests/test_agent_callbacks.py:6` — 149 chars (long
    `pylint: disable` header).
  - F3 `scripts/run_baselines.py:164` — 3 leading spaces
    (continuation-line alignment in `StrategyFn = Callable[...]`).

### Cycle 3 — 3 minor lines fixed, all checks PASS

Cycle-3 fixes (verified in current code):
- `tests/test_crafting_env.py:236-237` split across two lines
  (F1 fixed).
- `tests/test_agent_callbacks.py:6-8` now 3 short
  `# pylint: disable=...` lines (F2 fixed; verified at
  `cat_game_rl/tests/test_agent_callbacks.py:6-8`).
- `scripts/run_baselines.py:164-165` now 2-space indent
  (F3 fixed; verified at `cat_game_rl/scripts/run_baselines.py:164`).

Final QA (final-qa-c3, `.opencode/flaw-reports/cat_game_rl-flaws.md`
cycle-3 section): **PASS — 0 issues.** Suppression line count
rose 15 → 17 only because the F2 fix split one long header into
3 lines; no new checks suppressed.

## 8. Final evidence

All commands run from `cat_game_rl/` (working dir per the flaw
report). Source: `.opencode/flaw-reports/cat_game_rl-flaws.md`
(final-qa-c3 §C3-1–C3-4).

| Check | Command | Result |
|---|---|---|
| pylint | `python3 -m pylint --rcfile=../.pylintrc src scripts tests` | **PASS** — 10.00/10, exit 0 |
| Suppressions | `grep -rn 'pylint: disable' src scripts tests --include='*.py' \| wc -l` | **17** (includes 1 `disable-next`; see below) |
| pytest (no cov) | `uv run pytest tests/ -q --no-cov` | **PASS** — 318 passed in 25.10s, exit 0 |
| Coverage gate | `uv run pytest -q` | **PASS** — 318 passed in 25.61s, **97.26%** ≥ 90%, exit 0 |
| Shebang uniformity | grep line 1 across 62 py files in `src scripts tests` | **PASS** — 62/62 `#!/usr/bin/env python` |
| `"""` count | `grep -rn '"""' src scripts tests --include='*.py' \| wc -l` | **PASS** — 0 |
| Double quotes | `grep -rn '"' ... \| wc -l` | **PASS** — 0 double-quote lines, 1789 single-quote lines |
| Tabs | `grep -rPn '\t' ... \| wc -l` | **PASS** — 0 |
| Lines >80 | `awk 'length>80{...}'` over 62 files | **PASS** — 0 lines |
| Indent anomalies | leading-space distribution | **PASS** — 0 anomalies (all indents even / 2-space) |

**Suppression inventory (17 lines, cycle-3 count):** full list in
the flaw report §C3-1b — `src/cat_game_env/multi_agent.py:465`,
`src/dashboard/app.py:6`,
`scripts/{evaluate,run_baselines,run_ga,train,train_multi_agent}.py`,
and 10 sites in `tests/` (including
`tests/test_dashboard_pages.py:340` `disable-next=unused-argument`).
Cycle-2 count was 15 lines; both counts are disclosed per
ADR-002/ADR-005 (pylint still 10.00/10). The ADR-002 inventory
groups the same sites into 10 disclosed disable groups with
per-site justification (`sys.path` bootstrap; white-box tests;
`multi_agent` protected-access). Any new suppression requires a
new ADR row.

**Loop exit criteria met:** pylint 10.00/10 (0 issues) + 318
passed + 97.26% coverage + spec greps all PASS + gates
arch APPROVED (c2), reviewer APPROVED (c2), security APPROVED
(c1, with mediums — §9).

## 9. Deferred findings (not resolved this loop)

### Reviewer R1–R4 — DEFERRED, flag-only (ADR-003)

No code change. All pre-existing, introduced before this loop.

- **R1** `MaskedAgent` / `TierAgent` duplication —
  `cat_game_rl/src/agent/masked_agent.py:25` vs `:111`
  (near-identical `__init__`/`predict`/`save`/`load`).
- **R2** Hand-rolled `CheckpointCallback` —
  `cat_game_rl/src/agent/callbacks.py:55`, used at
  `cat_game_rl/src/agent/masked_agent.py:65-68`.
- **R3** Dead `action_mask_fn` —
  `cat_game_rl/src/agent/action_mask_fn.py`, re-exported at
  `cat_game_rl/src/agent/__init__.py:7`; production paths use
  local `_mask_fn`/`_tier_mask_fn` instead.
- **R4** `TierAgent.load` dead —
  `cat_game_rl/src/agent/masked_agent.py:163-173`, no production
  caller.

Revisit when a dedicated refactor task is scheduled, with parity
tests proving `MaskedAgent`/`TierAgent` parity before any merge.

### Security mediums S1–S5 — DEFERRED (ADR-004)

Threat model: **local, trusted operator** — dashboard and scripts
run only by the repo owner on their own machine, own config/output
files, no untrusted remote user, no network exposure of Streamlit,
no multi-tenant data.

- **S1** Dashboard `_is_safe_path` symlink bypass —
  `cat_game_rl/src/dashboard/app.py:87-100` (prefix check without
  `realpath`; callers at lines 104, 112, 121, 128).
- **S2** `yaml` None guards missing — `yaml.safe_load` callers
  subscript/use result without a guard, e.g.
  `cat_game_rl/scripts/run_ga.py:39,42,50`,
  `cat_game_rl/scripts/run_baselines.py:518,524`,
  `cat_game_rl/scripts/train.py:39`,
  `cat_game_rl/scripts/train_multi_agent.py:70`,
  `cat_game_rl/scripts/evaluate.py:38`,
  `cat_game_rl/src/dashboard/app.py:47-49`.
- **S3** Streamlit no-auth — `cat_game_rl/src/dashboard/app.py:33-40`.
- **S4** Markdown name injection —
  `cat_game_rl/src/dashboard/pages/simulation.py:122`,
  `cat_game_rl/src/dashboard/pages/targets.py:85`.
- **S5** YAML schema validation absent (same load sites as S2).

If Streamlit is ever bound to a non-localhost interface or shared,
ADR-004 must be revisited before exposure: add auth, `realpath`
symlink resolution, `None` guards, markdown escaping, and config
schema validation.

### Other deferred / flagged items

- 4 low ADR-sync nits from arch cycle 2 (non-blocking; no
  nit-by-nit text supplied — see §7).
- ADR-002 follow-ups: revisit script-suppression sites (#1–#6)
  when entrypoints become installed packages; white-box test
  sites (#7–#9) when refactored to public APIs; `multi_agent`
  site (#10) when it gets a public accessor.

## 10. How to re-run verification

From the repo root, with the project venv active
(`cat_game_rl/.venv`; see `cat_game_rl/README.md:5-17`):

```bash
cd cat_game_rl

# 1. pylint — expect 10.00/10, exit 0
python3 -m pylint --rcfile=../.pylintrc src scripts tests

# 2. suppression count — expect 17 (disclose any change)
grep -rn 'pylint: disable' src scripts tests --include='*.py' | wc -l

# 3. pytest without coverage — expect 318 passed, exit 0
uv run pytest tests/ -q --no-cov

# 4. coverage gate — expect 318 passed, 97.26% (>= 90%), exit 0
uv run pytest -q

# 5. spec greps — expect all zeros / 62 of 62
find src scripts tests -name '*.py' | wc -l            # 62
head -1 src/core/items.py                              # #!/usr/bin/env python
grep -rn '"""' src scripts tests --include='*.py' | wc -l   # 0
grep -rn '"'  src scripts tests --include='*.py' | wc -l   # 0
grep -rPn '\t' src scripts tests --include='*.py' | wc -l  # 0
awk 'length>80{print FILENAME":"FNR}' $(find src scripts tests -name '*.py')  # (none)
```

Also confirm `.pylintrc` is untouched:

```bash
git diff -- .pylintrc   # expect empty
```

Gate verdicts to compare against: arch APPROVED (c2), reviewer
APPROVED (c2), security APPROVED with mediums (c1), QA PASS (c3).
If any check above differs, the loop is not in its exit state —
report the diff to the Lead; do not edit code from docs.

## 11. What was not verified here (by docs-engineer)

- Pre-loop baseline numbers beyond the plan file (6 GA failures,
  59.67% coverage) — Lead-handoff facts, not re-run (§3).
- Runtime behavior beyond pytest (no training/eval runs, no
  dashboard smoke test).
- Ruff check (not in loop scope; note `pyproject.toml` sets
  `line-length = 80`).
- The exact 5-file tester diff list and debugger per-change diffs
  beyond the final-state File:Line evidence in §5–§6.
- The 4 low ADR-sync nits from arch cycle 2 (count/severity only).

## 12. References

- `cat_game_rl/README.md:1-131` — project purpose, run/test
  commands, structure, game rules.
- `.pylintrc:238-268` — FORMAT (max-line-length=80,
  indent-string 2 spaces, single-line-if-stmt=yes).
- `.pylintrc:277-282` — STRING check-quote-consistency=yes.
- `cat_game_rl/pyproject.toml:29-38` — pytest addopts
  (90% coverage gate), ruff line-length 80.
- `.gitignore:209` — `/.opencode/` (why this committed report
  exists).
- Blackboard (gitignored, cited for traceability):
  `.opencode/plans/cat_game_rl-format-loop.md`,
  `.opencode/critique-log/cat_game_rl-cycle-1-{arch,reviewer,security}.md`,
  `.opencode/critique-log/cat_game_rl-cycle-2-{arch,reviewer}.md`,
  `.opencode/flaw-reports/cat_game_rl-flaws.md`,
  `.opencode/context/decisions.md` (ADR-001–ADR-005),
  `.opencode/handoffs/fix-env-agent.md`.
