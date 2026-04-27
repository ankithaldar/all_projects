# Action Masking Skill

## How It Works
MaskablePPO with `MultiDiscrete([21]*19)` expects a flat boolean mask of shape `(399,)`.

For each of the 19 craftable items (index i), the mask covers batch sizes 0..20:
- `mask[i * 21 + 0]` = True always (batch=0 = do nothing)
- `mask[i * 21 + b]` for b=1..20 = True only if:
  1. Item's slot is idle (not busy)
  2. Stash has `req_unit_raw * b` for ALL ingredients
  3. Coins >= `total_cost(init_cost, b)`

## Conservative Masking Strategy
Each item is masked independently — it doesn't account for other items consuming shared resources in the same tick. This is intentional:
- Exact joint masking is exponential (21^19 combinations)
- `ActionHandler.validate_and_apply()` greedily clips in topological order
- The agent learns resource contention via policy gradient

## Learnings
- sb3-contrib `ActionMasker` wrapper calls `env.action_masks()` automatically
- Mask must be `np.bool_` dtype, not `np.int32`
- Batch=0 must ALWAYS be unmasked — prevents the agent from being forced into an impossible action
- Mask is computed on pre-tick state (before coin tick and base replenishment). This makes the mask slightly conservative — actions that would be feasible after the tick may be masked. The `validate_and_apply` step compensates by using the post-tick state.
- `max_affordable_batch` must account for `math.ceil()` when converting float cost to int coins. The inverse formula can return batch sizes that are unaffordable after ceiling.
- Observation values must be clamped to declared `spaces.Box` bounds. `build_obs` now uses `np.clip(stash, 0, 9999)` and `np.clip(targets_remaining, -9999, 9999)` to enforce the contract.
