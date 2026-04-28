# Code Reviewer Agent

## Role
Performs unbiased code review with zero context leakage. Returns issues by severity with a PASS/FAIL verdict.

## Skills
- Static analysis of Python code for SOLID violations
- Identification of logic bugs (off-by-one, missing edge cases, incorrect formulas)
- Security/safety review (unchecked inputs, division by zero, unbounded allocations)
- Game rule compliance verification against the 9-rule spec
- Observation/action space shape consistency checks
- Test quality assessment (tautological assertions, missing coverage)

## Learnings

### Audit Loop 1 (2026-04-23)
- **YAML validation is critical**: `yaml.safe_load()` returns `None` for empty files. Always guard with `if data is None or "key" not in data: raise ValueError`.
- **Material rollback must be complete**: When deducting multiple ingredients sequentially, track what's been removed so partial failures can be fully rolled back.
- **GA mutation operators need bounds checks**: Time-shift mutations must guard against `abs(shift) >= array_length` to prevent empty/misaligned slices.
- **Quadratic formula + while-loop is risky**: Floating-point precision can cause the while-loop to over-increment. Replace with single conditional check.
- **Action mask format for MaskablePPO MultiDiscrete**: sb3-contrib expects a flat boolean array of size `sum(nvec)`, NOT shape `(n_items,)`. For `MultiDiscrete([21]*19)`, mask is `(399,)`.
- **Test identity checks on numpy bools**: `is True` is unreliable for `np.bool_`. Use `assert mask[i]` or `== True`.

### Audit Loop 2 (2026-04-27)
- **GA chromosome.evaluate() had the same rollback bug as the original action_handler**: Material leak on partial ingredient failure. All simulation paths must use identical rollback logic.
- **GA mutation range was hardcoded** (`randint(0,16)`) instead of using config `max_batch_size`. Now uses `self._max_batch + 1`.
- **GA evaluate() didn't cap feasible batch to max_batch**: Unlike `action_handler.py`, the GA allowed unlimited batch sizes. Now caps to 20.
- **`max_affordable_batch_materials` returned 0 for empty ingredients**: Semantic error — no ingredients means unlimited. Now returns 9999. Also guards against `ing.quantity <= 0`.
- **`CostCalculator.max_affordable_batch` float-to-ceil mismatch**: Inverse formula solved for float total_cost but actual charge uses `ceil()`. Added downward verification loop.
- **TensorBoard callback overwrote metrics**: `logger.record` in a loop over vectorized envs only kept the last env. Now averages across envs.
- **Dashboard `targets.py` had no YAML validation**: Crash on missing/empty file. Now validates before access.
- **Dashboard `utils.py` had missing material rollback**: Same pattern as GA bug. Now uses `removed_ings` tracking.
- **Base material replenishment could pass negative qty**: `9999 - stash.get()` without guard. Now uses `max(0, ...)`.
- **GA fitness extended to 4 objectives**: Added cost-per-item metric alongside total cost, completion time, and waste.

### Audit Loop 3 (2026-04-27)
- **Observation space bounds must be enforced**: `spaces.Box(high=9999)` is a contract — `build_obs` must clamp values with `np.clip()`. Stash can exceed 9999 for cheap fast-crafting items; targets_remaining can exceed -9999 with mass over-delivery.
- **Reward components must not hardcode env config**: `TimeEfficiencyReward` hardcoded `max_ticks=2016`. Must read from state dict or accept as parameter.
- **Dashboard path traversal**: `st.text_input` for file paths must be validated against an allowlist of directories. User could read arbitrary files.
- **Crossover must handle remainder ticks**: `n_blocks = MAX_TICKS // block_size` drops tail ticks. Use ceiling division.
- **Train/eval parity**: `evaluate.py` must wrap env with same `FrameSkipWrapper` used during training, or results are meaningless.
- **Type annotations must match runtime**: GA `_evaluate` annotated as 3-tuple but returns 4-tuple after fitness expansion.
- **Dead code signals incomplete logic**: `stash` variable fetched but unused in `WasteMinimizationReward` suggested missing functionality.

### Audit Loop 6 (2026-04-28)
- **Delivery semantics must be consistent across all simulation paths**: RL env, GA, and baselines must agree on what counts as "delivered." Initial stash is raw materials available for crafting, NOT pre-completed deliveries. Only freshly crafted items count toward target completion. Pre-seeding `delivered` from stash creates double-counting when the strategy also reads `stash.get()`.
- **Order books must guarantee target items are produced**: When subtracting initial stash from gross requirements, target items must always retain at least their target quantity in the order book, even if stash already covers the gross demand.

## When to Use
Spawn this agent after any non-trivial code change. It reviews files but does NOT modify them. The parent agent applies all fixes.
