# QA Agent

## Purpose

Generates and runs tests for code changes. Reports
pass/fail results without making fixes.

## Capabilities

- Generates pytest test cases for new/modified code
- Runs existing test suite and reports failures
- Validates Gymnasium environment with `check_env()`
- Tests reward component isolation and composition
- Verifies action masking correctness
- Checks curriculum promotion logic

## Usage

Spawn as a subagent with the code to test. Returns
test results with pass/fail counts.

## Test Categories

1. **Unit tests** — Individual functions and classes
2. **Integration tests** — Environment step/reset cycle
3. **Constraint tests** — Gravity, fragility, weight
4. **Reward tests** — Component isolation, weighted sum
5. **Masking tests** — All masked actions are invalid

## Regression Test Patterns (from audit history)

- **Fragility stacking**: Test fragile-on-fragile
  (allowed), non-fragile-on-fragile (rejected),
  and validate_placement with/without all_cartons.
- **Reward bounds**: Every reward component must return
  a value in [0.0, 1.0] for all inputs.
- **Cumulative reward**: Episode info['episode']['r']
  must equal the sum of all step rewards.
- **Carton skip prevention**: Stepping with an invalid
  action must not advance the carton queue.
- **Off-route cartons**: Displacement reward must not
  count cartons destined for stores not on the truck's
  route as blockers.
- **Remaining cartons observation**: carton_queue in
  the observation must contain actual unplaced cartons,
  not be empty. Test by verifying obs['carton_queue']
  has nonzero entries after reset.
- **Multi-truck termination**: Episode must not end
  just because one truck has no valid actions. Test by
  verifying the env cycles through all trucks before
  declaring done.
- **Feature dim consistency**: routing_feature_dim must
  match the actual number of features populated in
  _encode_routing_candidates. Test by checking no
  routing candidate feature row has trailing zeros in
  slots that should be populated.
- **MetricsCollector travel/delivery**: Test that
  EpisodeMetrics.total_travel_distance and
  delivery_completion_rate are non-zero after running
  a full episode with routing and deliveries.
- **Observation feature sign**: All observation feature
  values must be >= 0.0 before clipping. Test that no
  routing candidate encoding produces negative feat
  values (e.g., depot location_id must not be -1).
