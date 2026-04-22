# Reward Engineering

## Key Learnings

### Shaped Reward Design

- 8 independent components with configurable weights
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
