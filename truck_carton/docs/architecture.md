# Architecture

## Overview

The system optimizes 3D carton loading into delivery
trucks using reinforcement learning. It handles
multi-truck fleets with multi-store delivery routes,
respecting physical constraints (gravity, weight,
fragility) and logistics constraints (delivery order,
store grouping, priority accessibility).

## High-Level Architecture

```
┌─────────────────────────────────────────────┐
│              Training Pipeline               │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
│  │ Trainer  │──│MaskablePPO│──│ Callbacks │ │
│  └──────────┘  └──────────┘  └───────────┘ │
│                     │                        │
│              ┌──────┴──────┐                 │
│              │  Environment │                │
│              └──────┬──────┘                 │
└─────────────────────┼───────────────────────┘
                      │
    ┌─────────────────┼──────────────────┐
    │                 │                  │
┌───┴───┐    ┌───────┴──────┐   ┌───────┴──────┐
│Packing│    │    Reward    │   │  Observation  │
│Engine │    │  Calculator  │   │   Builder     │
└───────┘    └──────────────┘   └──────────────┘
```

## Module Dependency Graph

```
config.py
    │
    ├── domain/models.py
    │       │
    │       └── domain/data_generator.py
    │
    ├── packing/rotation.py
    │       │
    │       ├── packing/space3d.py
    │       │
    │       └── packing/placement.py
    │               │
    │               └── env/action.py
    │
    ├── reward/calculator.py
    │       │
    │       └── reward/{utilization,displacement,
    │           grouping,fragility,support,weight,
    │           completion,priority}.py
    │
    ├── env/observation.py
    │
    ├── env/masking.py
    │
    ├── env/packing_env.py  (integrates all above)
    │
    ├── curriculum/manager.py
    │
    └── training/{trainer,callbacks}.py
```

## Key Design Decisions

### 1. Action Space: Candidate-Index Selection

The raw action tuple (carton, truck, x, y, z,
rotation) has ~600K combinations. Instead:

- Carton order is fixed by logistics constraints
  (reverse delivery route, then priority ascending)
- At each step, all valid placements are pre-computed
  across all trucks
- Action space is `Discrete(500)` — the agent selects
  a candidate index
- Action masking ensures only valid candidates are
  selectable

This reduces the problem from "find a valid placement"
to "choose the best among valid placements."

### 2. Gravity via Height Map

Each `Space3D` maintains a 2D height map `(L, W)` where
each cell tracks the highest occupied Z+1. Valid
placement z for a carton footprint is
`max(height_map[footprint])`, and full base support
requires `min(height_map[footprint]) == max(...)`.

This reduces candidate search from `O(L*W*H*R)` to
`O(L*W*R)`.

### 3. Observation Space: Dict of Box Spaces

Using `MultiInputPolicy`, the observation is a Dict
with components:

| Key             | Shape           | Content              |
|-----------------|-----------------|----------------------|
| truck_grids     | (5, 1024)       | Flattened 3D grids   |
| truck_meta      | (5, 7)          | Dimensions, weights  |
| truck_routes    | (5, 4)          | Store visit matrix   |
| height_maps     | (5, 16, 8)     | 2D height projection |
| current_carton  | (7,)            | Current carton feats |
| carton_queue    | (40, 7)         | Remaining cartons    |
| candidates      | (500, 18)       | Candidate features   |
| global_info     | (3,)            | Progress indicators  |

All values normalized to [0, 1]. Variable-size episodes
use zero-padding up to maximum dimensions.

### 4. Multi-Truck Handling

All trucks are considered simultaneously through the
candidate list. Candidates include truck-identifying
features (one-hot ID, remaining capacity, occupancy).
The agent learns to distribute cartons across the fleet.

### 5. Reward Decomposition

Eight independent components, each returning a scalar:

| Component   | Weight | Signal                      |
|-------------|--------|-----------------------------|
| Utilization | +1.0   | Volume + weight fill ratio  |
| Displacement| -2.0   | Cartons blocking unloading  |
| Grouping    | +1.5   | Same-store bbox tightness   |
| Fragility   | -5.0   | Non-fragile above fragile   |
| Support     | -5.0   | Unsupported carton cells    |
| Weight      | -3.0   | Excess weight ratio         |
| Completion  | +10.0  | Fraction placed + bonus     |
| Priority    | +1.0   | High-priority accessibility |

### 6. Curriculum Learning

Three stages with increasing complexity. Promotion is
triggered when mean reward exceeds a configurable
threshold over a sustained window. The observation space
is fixed to maximum dimensions — only the amount of
zero-padding changes between stages.

## Data Flow

```
reset()
  → DataGenerator.generate() → EpisodeData
  → Space3D per truck
  → Sort cartons (reverse route, priority)
  → Compute candidates for first carton
  → Encode observation → return (obs, info)

step(action)
  → Decode candidate from action index
  → Place carton into Space3D
  → Compute reward via RewardCalculator
  → Advance to next carton in queue
  → Compute new candidates
  → Encode new observation
  → return (obs, reward, terminated, truncated, info)

Episode ends when:
  - All cartons placed
  - Current carton has 0 valid candidates
  - Max steps exceeded
```

## Algorithm Choice: MaskablePPO

- **Action masking** prevents illegal placements
- **PPO** is stable and works well with short episodes
  (10-40 steps)
- **MultiInputPolicy** handles Dict observation spaces
- **On-policy** makes it practical for frequently
  resetting environments

Alternatives considered:
- DQN: no Dict observation support in SB3
- SAC: designed for continuous action spaces
- A2C: less stable than PPO's clipping mechanism
