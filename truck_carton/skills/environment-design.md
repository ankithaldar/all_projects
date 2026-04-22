# Environment Design

## Key Learnings

### Observation Space

- Use `gymnasium.spaces.Dict` with `Box` components
  for heterogeneous data
- `MultiInputPolicy` in sb3-contrib handles Dict
  observations via `CombinedExtractor`
- All observations must be fixed-shape; use zero-padding
  for variable episode sizes
- Normalize everything to [0, 1] for stable training

### Action Space

- `Discrete(K)` with candidate-index selection is
  cleaner than raw `MultiDiscrete`
- Pre-computing valid candidates removes illegal
  actions entirely
- Fixed carton ordering (by logistics constraints)
  removes one decision dimension
- K=500 candidates is sufficient for trucks up to
  16x8x8 with 5 trucks

### Episode Lifecycle

- `reset()` generates fresh random data per episode
- `step()` places one carton per step
- Episode terminates on: all placed, no valid
  candidates, or max steps
- The `action_masks()` method is called automatically
  by MaskablePPO

### Common Pitfalls

- Observation space shape must be constant across
  curriculum stages
- `info['episode']` dict is required for SB3's Monitor
  wrapper to detect episode boundaries
- Height map must be recomputed after every placement

### Audit Learnings (Loop 1)

- info['episode']['r'] must be CUMULATIVE episode
  reward, not the single-step reward from the final
  step. SB3's Monitor records this value as the total
  episode return.
- All observation features must be normalized to [0,1].
  Hardcoded 1.0 placeholders in truck_meta waste
  observation capacity. Use actual normalized values
  (e.g., truck.max_weight / max_weight_capacity).
- Carton weight normalization must use a config value
  (max_carton_weight) not a hardcoded 50.0.
- EnvironmentConfig should include max_weight_capacity
  and max_carton_weight so the observation builder can
  normalize without accessing TruckConfig/CartonConfig.

### Grid World Integration

- The grid is procedurally generated per episode via
  DataGenerator._generate_grid_world(). Uses MST on
  facility positions for guaranteed connectivity, plus
  random extra edges for redundancy.
- L-shaped road segments (inspired by the reference
  supply-chain environment's build_railroad) connect
  facilities with random bend points.
- networkx computes all-pairs shortest paths at episode
  start. Paths are cached in GridWorld.path_cache.
- Trucks use a state machine: ROUTING (pick destination)
  -> LOADING (pack at warehouse) -> ROUTING -> ... ->
  AT_DEPOT (finished).
- The unified Discrete(600) action space keeps packing
  (0..499) and routing (500..599) mutually exclusive
  via the action mask.
- When a truck arrives at a store, cargo matching that
  store is auto-unloaded (no explicit unloading action
  needed).

### PIL-Based Grid Rendering

- GridRenderer uses PIL/Pillow for composite images
  with ASCII grid, Unicode road sprites, facility
  labels, truck markers, and a status dashboard.
- Road sprites use 4-neighbor connectivity checks to
  select from Unicode box-drawing characters:
  ─ │ ┌ ┐ └ ┘ ├ ┤ ┬ ┴ ┼
- The env provides snapshots via get_render_snapshot()
  returning a dict decoupled from the env object.
- capture_frame() + save_gif() enables animated GIF
  export of full episodes.
- play_animation() uses matplotlib.animation for
  Jupyter HTML5 video playback.
