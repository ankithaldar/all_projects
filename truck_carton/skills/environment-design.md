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
