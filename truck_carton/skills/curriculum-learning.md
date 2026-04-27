# Curriculum Learning

## Key Learnings

### Stage Design

- 5 stages: Tiny (2/2/10), Small (3/3/20),
  Medium (5/4/40), Large (8/6/80), Full (12/8/120)
- Each stage increases trucks, stores, cartons,
  warehouses, and grid size
- The agent transfers learned packing and routing
  strategies to harder problems

### Promotion Mechanism

- Track episode rewards in a sliding window deque
- Promote when mean reward exceeds threshold for the
  full window length
- Clear reward history on promotion to prevent
  immediate re-promotion
- Final stage has no promotion — training continues
  until timestep budget

### Fixed Observation Space

- Observation dimensions are always sized for the
  maximum stage (12 trucks, 120 cartons, 500 packing
  candidates, 100 routing candidates)
- Earlier stages use more zero-padding
- This means the same neural network architecture
  works across all stages without modification
- The `is_active` flag in truck_meta (slot 6 = 1.0)
  distinguishes real trucks from padding

### Integration with SB3

- `CurriculumCallback` monitors `info['episode']`
  dicts in the rollout buffer
- On promotion, it unwraps the environment to call
  `set_curriculum_stage()`
- The next `reset()` automatically generates data for
  the new stage
- Model weights are preserved — transfer learning

### Tuning Tips

- Start with a generous promotion threshold (0.5-0.7)
- Use a window of 50-200 episodes to filter noise
- If the agent never promotes, the toy stage may be
  too hard — check reward weights
- Monitor `curriculum/stage` in TensorBoard to verify
  transitions
