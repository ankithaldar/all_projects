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

## When to Use
Spawn this agent after any non-trivial code change. It reviews files but does NOT modify them. The parent agent applies all fixes.
