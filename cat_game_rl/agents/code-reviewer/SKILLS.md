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

## When to Use
Spawn this agent after any non-trivial code change. It reviews files but does NOT modify them. The parent agent applies all fixes.
