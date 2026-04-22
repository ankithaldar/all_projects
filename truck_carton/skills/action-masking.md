# Action Masking

## Key Learnings

### Why Action Masking Is Essential

- Raw action space has ~600K combinations, >99% invalid
- Without masking, the agent wastes exploration on
  illegal moves and receives only penalties
- MaskablePPO zeros out logits for masked actions
  before sampling

### Candidate-Index Approach

- Pre-compute all valid placements for the current
  carton across all trucks
- Store as a list of `PlacementCandidate` objects
- Pad to fixed size K=500 with boolean mask
- Agent selects index into this list
- Every selectable action is guaranteed valid

### Height Map Acceleration

- Instead of checking all (x, y, z) positions, use
  the 2D height map
- For each (x, y) and rotation, the only valid z is
  `max(height_map[footprint])`
- Full support requires all height map values in the
  footprint to be equal
- Reduces search from O(L*W*H*R) to O(L*W*R)

### Fragility in Masking

- Non-fragile cartons: check no fragile carton exists
  below in the same columns
- This requires a carton lookup dict to check
  `is_fragile` for IDs found in the grid
- The lookup is rebuilt each time `find_valid_positions`
  is called

### Candidate Overflow

- If valid positions exceed K=500, candidates are
  truncated
- Future improvement: sample strategically (spread
  across trucks, prefer corners, prefer low-z)

### Audit Learnings (Loop 1)

- PlacementValidator._carton_lookup must be populated
  before any fragility check. validate_placement()
  originally missed this — fragility checks silently
  passed because the lookup was empty.
- _check_no_nonfragile_above originally rejected ANY
  carton above a fragile one (np.all(above == 0)).
  Fixed to allow fragile-above-fragile by checking
  each carton's is_fragile flag individually.
- The environment must not advance to the next carton
  when placement fails (action decodes to None).
  Otherwise cartons are silently skipped.
