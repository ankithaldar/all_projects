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
- **Wrong type annotations**: Check that method params
  match the actual objects passed. `truck: Carton` when
  it should be `truck: Truck` causes runtime errors.
- **Missing imports in tests**: New test code that uses
  numpy must import it explicitly.
- **SOLID violations to watch**: God classes (>300 LOC),
  methods with >5 params (use a context dataclass),
  thin wrappers that add no abstraction value.
- **Feature dim drift**: When max_trucks changes, the
  candidate_feature_dim must be max_trucks + 8 (onehot
  + 8 position/weight features). Check this invariant
  after any config change.
- **Component count drift**: When a reward component is
  added, update the test asserting len(breakdown)==N.
- **Stale generation tests**: When domain model fields
  change (e.g. truck.route becomes dynamic), tests
  asserting on generated data must be updated.
- **Grid capacity overflow**: When facilities exceed
  available grid cells, cap the count to avoid
  infinite placement loops.
- **State machine deadlocks**: If a truck is LOADING
  but no cartons fit (too large), it must transition
  to ROUTING. Check _advance_truck_state handles the
  "queue non-empty but no valid placements" case.
- **Observation value range**: All obs must be [0,1].
  Watch stage_index normalization — dividing by a
  hardcoded constant breaks when curriculum stages
  are added. Use max(num_stages-1, 1) or clamp.
- **Encapsulation**: Scripts must not access env._*
  private attributes. Use public properties instead.
  Add @property accessors for commonly-needed state.
- **Reward output clipping**: All reward components
  MUST return min(..., 1.0). Violation-count-based
  rewards (fragility, support) can theoretically
  exceed 1.0 even though violations <= total. Always
  add a safety clamp.
- **Unbounded frame accumulation**: GridRenderer
  capture_frame() must have a max_frames limit or
  LRU eviction. Without it, long training episodes
  cause OOM.
- **Logger thread safety**: JSON formatters using
  json.dumps(default=str) should wrap in try/except
  for non-serializable edge cases (circular refs,
  custom objects).
- **Dashboard state consistency**: Streamlit session
  state mutations during reruns can cause race-like
  behavior. Use explicit episode IDs and guard all
  reads with key-existence checks.
- **Heuristic agent encapsulation**: Baseline agents
  must use public @property accessors (action_manager,
  current_carton, carton_lookup, active_truck_idx,
  warehouse_cartons, truck_cargo) instead of env._*
  private attributes. Add new @property to env when
  needed rather than violating encapsulation.
- **None carton guard**: When scoring packing
  candidates, always check that current_carton is not
  None before accessing its fields. The invariant
  (candidates imply carton) is implicit and fragile.
- **Test env mocking**: Dont use bare DummyEnv objects
  without required attributes. Use a real env instance
  with reset() called, or properly mock all accessed
  properties.
- **Empty observation context fields**: When building
  ObservationContext, all list fields must be populated
  with actual data, not empty lists. An empty
  remaining_cartons field means carton_queue obs is
  always zeros — a silent information loss that cripples
  the agent's ability to plan.
- **Premature episode termination**: After switching
  the active truck, if that truck has no valid actions,
  cycle through other non-depot trucks before declaring
  the episode done. Only terminate when ALL trucks lack
  valid actions.
- **Observation feature slot waste**: If a feature dim
  config (e.g., routing_feature_dim) reserves N slots
  but only N-1 are populated, the last slot is always
  zero and wastes model capacity. Ensure dim matches
  actual feature count.
- **Hardcoded normalization constants**: Stage index
  normalization must use config-derived values (e.g.,
  num_curriculum_stages - 1), not hardcoded numbers
  like 4.0 that break when stages are added/removed.
- **Magic number enum comparisons**: Use CellType enum
  values (CellType.DEPOT, CellType.WAREHOUSE) instead
  of raw integers (2, 3, 4) when comparing location
  types in observation encoding.
