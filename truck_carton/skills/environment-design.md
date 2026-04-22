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

### Scaling to Large Grids (160x160)

- Direct grid encoding (grid_map) doesn't scale —
  160x160 = 25,600 cells is too large for observation.
- Use graph-distance features instead: pairwise
  facility distances in a fixed-size matrix
  (max_locations x max_locations = 15x15 = 225 floats).
- Truck positions are normalized (row/grid_rows,
  col/grid_cols) so they stay [0,1] at any grid size.
- Facility spacing must scale with grid size:
  spacing = max(base, grid_dim * scale_factor).
- Curriculum stages scale gradually: 4x4 -> 16x16 ->
  40x40 -> 80x80 -> 160x160 to prevent the agent from
  being overwhelmed by large grids early.

### Runtime Edge Cases

- Grid generation must cap facility count to available
  interior cells (rows-2)*(cols-2)/3 per type to avoid
  infinite placement loops on small grids.
- State machine must handle "queue has cartons but none
  fit in truck" by transitioning LOADING -> ROUTING.
  Otherwise the truck is stuck forever.
- Adjacent facilities (Manhattan dist 1) don't need
  road segments — they're directly connected as graph
  nodes since all non-TERRAIN cells are traversable.
- Observation stage_index must be normalized by the
  actual max stage count, not a hardcoded constant.
  With 5 stages (indices 0-4), dividing by 4.0 and
  clamping to 1.0 is correct.
- Scripts should use public @property accessors
  (episode_data, spaces, placed_cartons, num_delivered,
  step_count) instead of env._* private attributes.

### Logging & Observability

- EpisodeLogger accumulates per-step structured data
  (action, type, reward breakdown, truck states and
  positions) and emits a complete summary at episode
  end via end_episode().
- JSON log output (one object per line) feeds the
  Streamlit dashboard and external monitoring tools.
- Hierarchical loggers: truck_carton.{env, episode,
  training, curriculum, packing, reward}.
- Logger integration is non-blocking: DEBUG-level
  step logs, INFO-level episode/promotion events.
- setup_logging() is idempotent — safe to call
  multiple times without duplicating handlers.

### Live Dashboard

- Streamlit dashboard (scripts/dashboard.py) renders
  the grid world map live during simulation using
  Plotly go.Image + go.Scatter overlays.
- Trucks shown as colored diamonds with state-based
  colors (ROUTING=orange, LOADING=cyan, AT_DEPOT=gray)
- Per-truck cargo bar chart, reward polar breakdown,
  cumulative timeline, and episode history trends.
- Dashboard uses st.session_state for cross-rerun
  persistence; guard all state reads with defaults.
