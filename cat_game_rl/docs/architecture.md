# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Entry Points                              │
│  scripts/train.py  evaluate.py  run_ga.py  launch_dashboard │
└──────┬───────────────┬────────────┬──────────────┬──────────┘
       │               │            │              │
       ▼               ▼            ▼              ▼
┌──────────┐  ┌──────────┐  ┌───────────┐  ┌──────────────┐
│ MaskedAgent│ │  Evaluate │ │GaScheduler│  │  Streamlit   │
│ (sb3-PPO) │ │  + Export │ │  (DEAP)   │  │  Dashboard   │
└─────┬──────┘ └─────┬────┘ └─────┬─────┘  └──────┬───────┘
      │              │            │                │
      ▼              ▼            ▼                ▼
┌────────────────────────────────────────────────────────────┐
│                   CraftingEnv (Gymnasium)                    │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────────┐  │
│  │ ObsBuilder   │ │ActionHandler │ │  RewardShaper      │  │
│  │ (Dict space) │ │ (validate+   │ │  (7 components)    │  │
│  │              │ │  apply)      │ │                    │  │
│  └──────────────┘ └──────────────┘ └────────────────────┘  │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│                    Core Domain                              │
│  ┌──────────┐ ┌──────┐ ┌─────────────┐ ┌───────────────┐  │
│  │CraftingTree│ │Stash │ │CoinGenerator│ │SlotScheduler  │  │
│  │ (recipes) │ │(inv) │ │ (210/tick)  │ │ (19 slots)    │  │
│  └──────────┘ └──────┘ └─────────────┘ └───────────────┘  │
│  ┌──────────────┐ ┌───────────────┐                        │
│  │CostCalculator│ │TargetProvider │                        │
│  │ (batch cost) │ │ (YAML targets)│                        │
│  └──────────────┘ └───────────────┘                        │
└────────────────────────────────────────────────────────────┘
```

## SOLID Compliance

### Single Responsibility
- `CostCalculator` — only batch cost math
- `Stash` — only inventory counts
- `CoinGenerator` — only coin balance
- `SlotScheduler` — only manufacturing slot state
- `TargetProvider` — only target tracking
- `ObservationBuilder` — only obs dict construction
- `ActionHandler` — only action decode/validate/apply
- `RewardShaper` — only reward computation

### Open/Closed
- `RewardComponent` ABC allows adding new reward terms without modifying `RewardShaper`
- `CraftingTree.from_yaml()` loads any item set from YAML
- Training algorithm is configurable via YAML (`training.yaml`)

### Liskov Substitution
- `FrameSkipWrapper` extends `gymnasium.Wrapper` — substitutable anywhere `Env` is expected
- All `RewardComponent` subclasses implement the same `compute()` interface

### Interface Segregation
- Core domain has no ML dependencies (numpy + pyyaml only)
- Dashboard doesn't import RL/GA code — it re-simulates from schedule files
- GA evaluator uses core domain directly, not the Gym env

### Dependency Inversion
- `CraftingEnv` depends on abstractions (CraftingTree, Stash, etc.), not concretions
- Configuration injected via dicts from YAML, not hardcoded

## Data Flow

### Training
```
YAML configs → CraftingEnv → MaskablePPO.learn()
                  ↕                    ↓
            obs, reward, mask     model checkpoints
                                       ↓
                              output/models/final_model.zip
```

### Evaluation
```
model.zip + CraftingEnv → predict loop → batch_schedule.txt
```

### GA Baseline
```
ga.yaml → GaScheduler → NSGA-II loop → Pareto front
              ↓                              ↓
        Core domain sim            ga_population_log.jsonl
              ↓                    ga_batch_schedule.txt
        (8064 ticks/individual)
```

### Dashboard
```
batch_schedule.txt ──┐
ga_population_log.jsonl ──┼→ Streamlit app → browser
targets.yaml ────────┘         ↓
                          simulate_schedule()
                          (deterministic re-sim)
```

## Key Design Decisions

### MultiDiscrete Action Space
19 items x 21 batch sizes = `MultiDiscrete([21]*19)`. MaskablePPO masks per sub-space. Total mask dimension: 399 bools.

### Conservative Masking + Greedy Validation
Masks each item independently (ignoring other items' demands this tick). `ActionHandler.validate_and_apply()` then applies greedily in topological order, clipping any that can't be satisfied. Agent learns contention via policy gradient. Mask is computed on pre-tick state (210 coins conservative). Observations are clamped to declared `spaces.Box` bounds.

### GA Chromosome: Dense (8064, 19)
Fixed-size enables simple crossover (time-block swaps) and numpy-friendly evaluation. 95% zeros at init keeps it effectively sparse.

### Optimization Objectives
Both RL and GA optimize for minimum coins used and lowest completion time with efficient batching:

**RL Reward Components (7 total):**
1. `SlotUtilizationReward` — maximize active manufacturing slots
2. `WasteMinimizationReward` — penalize over-production beyond targets
3. `TargetCompletionReward` — reward progress + completion bonus
4. `ExcessInventoryPenalty` — penalize large stash accumulation
5. `CoinEfficiencyReward` — penalize high cost-per-item produced
6. `TimeEfficiencyReward` — bonus for early target completion
7. `BatchOptimizationReward` — reward larger, more efficient batches

**GA Fitness (4 objectives, all minimized via NSGA-II):**
1. Total coin cost
2. Completion tick (time to meet all targets)
3. Waste (excess production beyond targets)
4. Cost per item produced (batch efficiency)

### Shared Simulation Logic
GA's `Chromosome.evaluate()` and `CraftingEnv.step()` both use the same core domain classes (Stash, CoinGenerator, SlotScheduler, CostCalculator). No duplicate game logic.

### Multi-Agent Tier System
Independent envs per tier connected via an OrderBoard message bus:

```
OrderBoard (demand ↓, supply ↑)
    │
    ├─ Tier 8 Agent: artifact (1 item)
    ├─ Tier 7 Agent: elementstone (1 item)
    ├─ Tier 6 Agent: waterstone, firestone (2 items)
    ├─ Tier 5 Agent: necklace, fire (2 items)
    ├─ Tier 4 Agent: gold, pendant, water (3 items)
    ├─ Tier 3 Agent: sparkles, silver (2 items)
    ├─ Tier 2 Agent: ribbon, needles (2 items)
    └─ Tier 1 Agent: string, wood, metal, bronze, amethyst, orb (6 items)
```

**Communication**: Orders propagate top-down (high tier demands from low). Fulfillments propagate bottom-up (completed items available to higher tiers). Shared coin pool across all tiers.

**Per-tier env**: Each TierEnv has local obs (own stash, slots, ingredients, orders), local action masking, and mixed reward (40% global target progress + 60% local efficiency).

**Orchestrator**: `MultiAgentOrchestrator` coordinates tick loop — coins, base refill, agent actions in topo order, slot completion, order propagation.

## Module Dependencies

```
config/*.yaml
    └→ src/core/items.py (CraftingTree.from_yaml)
    └→ src/core/target_provider.py (TargetProvider)

src/core/ (zero ML deps: numpy + pyyaml)
    └→ src/cat_game_env/ (adds gymnasium)
        └→ src/agent/ (adds sb3-contrib, torch)
    └→ src/ga/ (adds deap)
    └→ src/dashboard/ (adds streamlit, plotly, pandas)

src/logging_util/ (stdlib logging only)
```
