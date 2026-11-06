# cat_game_rl Format Loop — Final Report

**Date:** 2026-11-06
**Verdict: PASS** — all gates Cycle 4 APPROVED/PASS, 0 blocking issues
(Cycle 3 verdict below superseded; Cycle 4 evidence in §13).

## 1. What cat_game_rl does
cat_game_rl is a crafting-game RL/GA experiment (`cat_game_rl/docs/architecture.md`): a Gymnasium `CraftingEnv` (obs via `ObservationBuilder`, actions via `ActionHandler`, 7-component `RewardShaper`) trains a MaskablePPO agent (`src/agent/`), a DEAP GA scheduler (`src/ga/`, NSGA-II, dense `(8064,19)` chromosome) optimizes the same core domain (`src/core/`: CraftingTree, Stash, CoinGenerator, SlotScheduler, CostCalculator, TargetProvider — numpy + pyyaml only), and a Streamlit dashboard (`src/dashboard/`) re-simulates schedules deterministically. Entry points: `scripts/train.py`, `scripts/evaluate.py`, `scripts/run_ga.py`, `scripts/launch_dashboard.py`. Multi-agent tier system (Tiers 1–8 via OrderBoard) is in `src/cat_game_env/multi_agent.py`.

## 2. Strict spec
- Pylint 10.00/10 with root `.pylintrc`, whole tree (`src` + `scripts` + `tests`), exit 0, 0 messages.
- Pytest GREEN, coverage ≥90%.
- Strict format greps: lines ≤80 chars, no trailing whitespace / tabs / CRLF, EOF newline, every `.py` starts `#!/usr/bin/env python`, single quotes only (legitimate JSON payloads inside single-quoted strings excepted), no TODO/FIXME/XXX.
- `.pylintrc` must not be edited to reach 10/10.

## 3. Baseline
- Pylint **9.03/10**.
- NOSHEBANG on all files.
- Double quotes prevalent.
- 318 tests at baseline with 7 Colab-only failures (notebook environment, not local suite).

## 4. Cycle 1 — flaws and gate verdicts
Tester PASS (318 passed, pylint 10/10 claimed, 62/62 format), anti-team-security APPROVED (0 blocking, mediums accepted). Rejected:
- Anti-team-arch REJECTED: pylint unverified in arch env; magic `9999` ×3 (`src/env/inventory.py:97,104` + third instance), duplicated replenish (`src/env/crafting_env.py:130-135` vs `src/env/multi_agent.py:490-496`), sys.path / inline yaml lows.
- Reviewer REJECTED (blockers): dead `action_mask_fn.py:13-15`, `TierAgent` dup `masked_agent.py:111-173` (n_steps 2048 vs 512), stub docstrings, print/logging, magic range.
- Security REJECTED: SEC-01 High pickle load (`masked_agent.py:102,172`, `evaluate.py:56`, SB3 2.3.2 `weights_only=False`), SEC-02 Medium dashboard no auth, SEC-03/04 Low, SEC-05 Medium supply chain, SEC-06/07 Info.
- Adversarial REJECTED: F1 symlink traversal, F2 negative craft_time, F3 negative targets, F4 negative ingredient qty, F5 crash class (unknown items, negative coin/stash, wrong-shape genes, malformed schedule), F6 unbounded DoS (`max_ticks`, `max_batch`).

## 5. Cycle 2 — fixes and verdicts
Fixes: single-sourced `MAX_AFFORDABLE_BATCH = 9999` (`src/core/items.py:46`, reused via `src/cat_game_env/replenish.py` + env modules), shared replenish, F1–F6 input validation + caps (`MAX_SIMULATE_TICKS=100000`, `MAX_PARSE_FILE_BYTES=10000000`, `max_batch` cap 9999), ADR-001 mitigation (trusted-only warning + SHA256-at-save + docs), ADR-002 local threat model, notebook with pinned fallback + verify-branch note (15 cells, valid JSON), shebangs + single quotes everywhere.
Gate verdicts Cycle 2: arch **REJECTED** (3 mediums obs/chromosome/baselines 9999+replenish, pylint unverified), anti-sec APPROVED, reviewer APPROVED (minor items range), security APPROVED, adversarial APPROVED (22/22 PoCs blocked), tester **FAIL-narrow** (372 passed; whole-tree pylint 9.99/10 exit 20 — W1514 ×3 `tests/test_agent_callbacks.py:591,616,631` + C0415 `tests/test_chromosome.py:142`; coverage 95.79% vs 97% claimed). See `.opencode/flaw-reports/cat_game_rl-flaws-cycle2.md`.

## 6. Fix3 / Fix4
- **Fix3:** resolve tester D1/D2 — add `encoding=` to the 3 `open()` calls in `test_agent_callbacks.py`, move the line-142 import in `test_chromosome.py` to top level, add tests for `src/ga/chromosome.py` (75% → 95%) to genuinely reach 97% coverage.
- **Fix4:** resolve arch Cycle 2 mediums (obs/chromosome/baselines 9999+replenish single-sourcing) and the adversarial residual (null `targets:` → `ValueError` at `src/core/target_provider.py:41`); re-run all gates.

## 7. Cycle 3 — final evidence (verbatim commands + outputs)
From repo root unless stated:

```bash
cd cat_game_rl
python3 -m pytest tests/ -q --cov=src --cov-report=term-missing --cov-fail-under=90
# 377 passed, 1 warning in 36.18s, 0 failures/errors
# TOTAL 1877 stmts, 56 miss, 97.02% — Required test coverage of 90% reached. Total coverage: 97.02%

python3 -m pylint --rcfile=.pylintrc cat_game_rl/src cat_game_rl/scripts cat_game_rl/tests
# 10.00/10, exit 0, 0 messages

git diff -- .pylintrc
# (empty — .pylintrc untouched)

# strict greps over cat_game_rl/src scripts tests --include='*.py'
# Lines >80 chars: 0; trailing whitespace: 0; tabs: 0; CRLF: 0; missing EOF newline: 0
# Files without shebang: 0
# Double-quote lines: 2 only — tests/test_dashboard_pages.py:454,456 ('{"gen": 0}\n' JSON payloads)
# TODO/FIXME/XXX: 0

python3 -m json.tool cat_game_rl/notebooks/cat_game_rl_colab.ipynb > /dev/null
# valid JSON, nbformat 4, 15 cells

pytest test_items test_target_provider test_dashboard_utils test_chromosome test_action_handler test_inventory test_coin_generator test_ga_scheduler -q
# 129 passed (adversarial Cycle 3 regression subset)
```

Gate verdicts Cycle 3: arch APPROVED (lows only), anti-sec APPROVED, reviewer APPROVED, security APPROVED, adversarial APPROVED, tester PASS (377 / 97.02% / 10.00). Verdict files: `.opencode/critique-log/cat_game_rl-cycle-3-{arch,security}.md`, `.opencode/flaw-reports/cat_game_rl-{reviewer,security,adversarial,flaws}-cycle3.md`.

## 8. Notebook guide
`cat_game_rl/notebooks/cat_game_rl_colab.ipynb` (15 cells, nbformat 4): cell 1 carries the SEC-05 supply-chain / verify-branch note, cell 4 documents the pinned `requirements.txt` fallback, no `pip install` lines. Referenced entry points exist: `scripts/train.py`, `scripts/evaluate.py`, `scripts/run_ga.py`, `scripts/launch_dashboard.py`. Cells were checked statically; training cells were not executed (long/GPU-dependent).

## 9. ADRs
- **ADR-001** (`.opencode/context/decisions.md`): SEC-01 conflict resolution — trusted-only model load (warning + SHA256 + doc), no `weights_only` upgrade (unavailable in SB3 2.3.2).
- **ADR-002**: Threat model — local single-operator. Any untrusted/shared exposure supersedes this ADR and requires re-gating.

## 10. Residuals (non-blocking, tracked)
- SEC-06 / SEC-07 backlog (security Info items, non-blocking).
- CheckpointCallback sidecar (known sidecar behaviour, non-blocking).
- README warning (present, per ADR-001).
- `sys.path` hack low (scripts + `src/dashboard/app.py:17`).
- `NUM_ITEMS` literal low.
- `targets_total` clip low.
- Sibling malformed `targets:` shapes (e.g. `targets: 5`, non-int counts) still raise `AttributeError`/`TypeError` instead of `ValueError` — rejected, no silent acceptance (adversarial Cycle 3 residual).

## 11. How to re-run
```bash
cd cat_game_rl
python3 -m pytest tests/ -q --cov=src --cov-report=term-missing --cov-fail-under=90
python3 -m pylint --rcfile=.pylintrc cat_game_rl/src cat_game_rl/scripts cat_game_rl/tests
git diff -- .pylintrc   # must be empty
```
Colab: open `cat_game_rl/notebooks/cat_game_rl_colab.ipynb`, follow cell 1 verify-branch note and cell 4 pinned fallback; do not run unpinned `pip install` from an arbitrary branch (SEC-05).

## 12. Blind spots / assumptions
- Evidence above is as reported by the Cycle 3 tester/adversarial gates and task context; this docs session did not re-run pytest/pylint/git (no shell access) and `.pylintrc` emptiness is taken from task context.
- System python 3.14 / pytest 9.1.1 was used by tester (`.venv` has no pytest); pinned pytest 8.1.1 results may differ marginally.

## 13. Cycle 4 — Colab fixes and final evidence

**Verdict: PASS** — all gates Cycle 4 APPROVED/PASS, 0 blocking
issues. Status: `.opencode/status.md` (Final PASS cycle4).

### 13.1 Colab 11 failures — root causes

The Colab run failed 11 tests that passed locally. Two root causes,
both (test ↔ environment) mismatches, neither a production bug:

1. **Real SB3 vs stub (`tests/test_agent_callbacks.py`).** Locally the
   suite stubs `stable_baselines3` in `sys.modules` before importing
   `masked_agent`; Colab runs the real SB3 2.3.2 stack, whose
   constructor/`predict` wiring differs from the stub. Tests written
   against the stub therefore failed under real SB3.
2. **Remote config drift (`tests/test_target_provider.py`).**
   Colab clones the remote branch, where `config/targets.yaml` on
   commit `42c66e8` carries `gold: 13`. That config value is
   intentional (user commit), but the test asserted against the
   local config value, so it failed on the remote config. The
   adversarial gate records the same baseline:
   `git rev-parse --short HEAD` = `42c66e8`
   (`.opencode/flaw-reports/cat_game_rl-adversarial-cycle4.md:2`).
3. **Notebook cell 4 was a smoke pytest run**, not the full suite —
   it surfaced the above failures as a subset, not a separate
   notebook defect.

### 13.2 Fixes (test-only)

- `tests/test_agent_callbacks.py`: mock-based fakes
  (`_RecordingMaskablePPO`, `_RecordingActionMasker`) replace only
  the SB3 boundary; real `BaseMaskedAgent` code executes. New
  assertions pin real wiring — `args[0]=='MultiInputPolicy'`,
  `args[1] is wrapper`, `wrapped.env is env`,
  `wrapped.mask_fn is compute_action_mask`,
  `kwargs['n_steps']==2048/512`, `verbose==1/0`, predict forwarding
  `call['obs'] is obs`, `deterministic is True`,
  `action_masks is None` — replacing the old
  `getattr(..., 'mask_fn', None) or getattr(..., '_action_mask_fn', None)`
  fallback that could false-green. Diff: +111/-38.
  (`.opencode/flaw-reports/cat_game_rl-adversarial-cycle4.md:18`)
- `tests/test_target_provider.py`: config-independent fixture —
  writes its own `KNOWN_TARGETS` YAML (gold 14, amethyst 65,
  necklace 3, artifact 6) via `tmp_path`, decoupled from the real
  `config/targets.yaml` (file-top `# pylint:
  disable=redefined-outer-name` at line 6 allows the local
  `target_provider` fixture to shadow `conftest.py:52`). Diff:
  +24/-0. (ibid. `:18,22`)
- No production, config, or dependency change:
  `git diff -- src/ config/ pyproject.toml requirements.txt` = 0
  lines; only the 2 test files changed. (ibid. `:6`)

### 13.3 Gate verdicts Cycle 4

| Gate | Verdict | File |
|---|---|---|
| anti-team-arch | APPROVED (lows A1/A2/T1) | `.opencode/critique-log/cat_game_rl-cycle-4-arch.md` |
| anti-team-security | APPROVED | `.opencode/critique-log/cat_game_rl-cycle-4-security.md` |
| reviewer | APPROVED (3 nits) | `.opencode/flaw-reports/cat_game_rl-reviewer-cycle4.md` |
| tester | PASS (377 / 97.02% / 10.00) | `.opencode/flaw-reports/cat_game_rl-flaws-cycle4.md` |
| security | APPROVED (3 Info) | `.opencode/flaw-reports/cat_game_rl-security-cycle4.md` |
| adversarial | APPROVED (55 tests, test-only) | `.opencode/flaw-reports/cat_game_rl-adversarial-cycle4.md` |

### 13.4 Final evidence (verbatim, from gate files)

```bash
cd cat_game_rl
uv run pytest tests/ -q --cov=src --cov-report=term-missing --cov-fail-under=90
# 377 passed, 1 warning in 63.94s, 0 failures/errors
# TOTAL 1877 stmts, 56 miss, 97.02% — Required test coverage of 90% reached.

python -m pytest tests/test_agent_callbacks.py tests/test_target_provider.py -q -p no:cacheprovider --no-cov
# 55 passed in 1.68s (adversarial Cycle 4 subset)

uv run --project cat_game_rl pylint --rcfile=.pylintrc cat_game_rl/src cat_game_rl/scripts cat_game_rl/tests
# Your code has been rated at 10.00/10, exit 0, 0 messages
```

Sources: `.opencode/flaw-reports/cat_game_rl-flaws-cycle4.md:12-39`
(tester), `.opencode/flaw-reports/cat_game_rl-adversarial-cycle4.md:7`
(adversarial). The 1 warning is pre-existing and unrelated
(`test_dashboard_utils.py` unknown-item schedule line).

### 13.5 Push note (Colab)

Colab clones the remote branch — it does not see local-only fixes.
**Push the Cycle 4 test fixes before re-running the notebook**,
otherwise Colab will reproduce the same 11 failures against the
pre-fix remote (including the intentional `gold: 13` config on
`42c66e8`).

### 13.6 Assumptions / blind spots (Cycle 4 docs)

- This docs session persisted verdicts from the task handoff summary
  plus the two pre-existing gate files
  (`.opencode/flaw-reports/cat_game_rl-{flaws,adversarial}-cycle4.md`,
  preserved verbatim); it did not re-run pytest/pylint/git and did
  not edit code/config/`.pylintrc`.
- Reviewer 3 nits and security 3 Info items: counts and non-blocking
  status are as reported in the handoff; per-item text was not in
  the handoff and is recorded as tracked (not invented) in
  `.opencode/flaw-reports/cat_game_rl-{reviewer,security}-cycle4.md`.
- Anti-arch lows A1/A2/T1 and the Colab root-cause narrative are as
  reported in the handoff, cross-checked against the adversarial
  gate file (baseline `42c66e8`, test-only diff, 55 passed).
