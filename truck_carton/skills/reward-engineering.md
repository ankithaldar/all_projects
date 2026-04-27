# Reward Engineering

## Key Learnings

### Shaped Reward Design

- 9 independent components with configurable weights
- Each component returns a normalized scalar in a
  predictable range
- Negative weights for constraint violations, positive
  for desired behaviors
- Completion reward (eta=10.0) dominates to incentivize
  placing all cartons

### Component Independence

- Each component uses a Protocol interface:
  `compute(state) -> float`
- Components only read from `EnvironmentState`, never
  modify it
- Lazy imports in `RewardCalculator.__init__` break
  circular dependencies

### Displacement Calculation

- Truck door is at x=0 — lower X means more accessible
- At each store stop, count cartons from later stores
  that block access
- Simulate unloading in route order, tracking which
  cartons are already removed
- Normalize by total cartons per truck

### Fragility Rules

- Non-fragile above fragile = violation (checked by
  scanning unique IDs below)
- Fragile cartons: only check nothing is above them
  (vacuously true at placement time)
- The fragility check runs both at validation time
  (placement) and at reward time (verification)

### Weight vs Volume Tradeoff

- Utilization reward averages volumetric and weight
  utilization
- Weight penalty only triggers when max_weight is
  exceeded (hard constraint)
- This allows the agent to prioritize volume fill while
  respecting weight limits

### Audit Learnings (Loop 1)

- All reward components must return values in [0, 1].
  The RewardCalculator weights handle sign/magnitude.
- Completion reward originally returned up to 2.0
  (fraction + bonus). Fixed to return fraction*0.5
  normally, 1.0 on full completion.
- Displacement must exclude cartons whose destination
  store is not on the truck's route, otherwise they
  become permanent blockers inflating the metric.
- Priority reward normalization can produce negative
  values; must clamp output to [0, 1] with np.clip.
- Reward components are RATES not REWARDS: fragility
  and support return violation rates (0.0 = good),
  which the negative config weight converts to
  penalties. Don't invert them inside the component.

### Audit Learnings (Loop 2)

- Every per-truck accumulation must be capped per truck
  before averaging. Weight utilization and weight
  penalty can both exceed 1.0 per truck if the truck
  is overloaded beyond 2x capacity.
- Pattern: `min(per_truck_value, 1.0)` before summing,
  then `min(result, 1.0)` on the final output.
- When changing a reward component's return range,
  update ALL tests that assert on the old range.
  The test_completion_all_placed assertion was stale
  after capping completion to [0, 1].
- When adding a new reward component, update BOTH the
  RewardCalculator import/registration AND the test
  that asserts len(breakdown)==N.
