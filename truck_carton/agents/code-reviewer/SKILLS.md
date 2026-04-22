# Code Reviewer Agent

## Purpose

Unbiased code review with zero parent context.
Evaluates code changes for correctness, style
compliance, and potential issues.

## Capabilities

- Reviews Python source files for bugs and logic
  errors
- Checks compliance with Google Python Style Guide
  (`.pylintrc`)
- Validates type annotations and dataclass usage
- Identifies potential performance issues in NumPy
  operations
- Verifies Gymnasium environment contract compliance

## Usage

Spawn as a subagent with the changed file paths.
Returns issues by severity with a PASS/FAIL verdict.

## Review Checklist

1. **Correctness** — Logic errors, off-by-one,
   boundary conditions
2. **Style** — 4-space indent, 80-char lines, single
   quotes, snake_case (per .pylintrc)
3. **Types** — Proper annotations, Protocol usage
4. **NumPy** — Vectorization opportunities, dtype
   consistency
5. **Gym contract** — observation space shape matches,
   action mask correctness

## Known Bug Patterns (from audit history)

- **Uninitialized lookup dicts**: Methods that depend
  on `_carton_lookup` must populate it before use.
  Check that `validate_placement()` receives and
  builds the lookup from `all_cartons`.
- **Observation bounds**: All features MUST be in
  [0, 1]. Watch for hardcoded divisors (e.g., `/50.0`)
  that break when config values change. Use config
  fields instead.
- **Reward bounds**: Each RewardComponent.compute()
  must return [0, 1]. The RewardCalculator handles
  sign via weights.
- **Cumulative vs step reward**: episode info `r` must
  be the cumulative sum, not the final step's reward.
- **Carton skip on failed placement**: If action
  decodes to None, the environment must NOT advance
  to the next carton.
- **Route-unaware displacement**: Cartons assigned to
  stores not on a truck's route must be excluded from
  displacement calculations.
- **Unbounded per-truck accumulation**: When summing
  a per-truck metric (weight ratio, utilization),
  cap each truck's contribution to 1.0 before
  averaging. Otherwise overloaded trucks produce
  values > 1.0.
- **Stale test assertions**: When a component's return
  range changes, grep for all tests asserting on the
  old range and update them.
