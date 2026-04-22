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
│  │ (Dict space) │ │ (validate+   │ │  (4 components)    │  │
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
        (2016 ticks/individual)
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
Masks each item independently (ignoring other items' demands this tick). `ActionHandler.validate_and_apply()` then applies greedily in topological order, clipping any that can't be satisfied. Agent learns contention via policy gradient.

### GA Chromosome: Dense (2016, 19)
Fixed-size enables simple crossover (time-block swaps) and numpy-friendly evaluation. 95% zeros at init keeps it effectively sparse.

### Shared Simulation Logic
GA's `Chromosome.evaluate()` and `CraftingEnv.step()` both use the same core domain classes (Stash, CoinGenerator, SlotScheduler, CostCalculator). No duplicate game logic.

## Module Dependencies

```
config/*.yaml
    └→ src/core/items.py (CraftingTree.from_yaml)
    └→ src/core/target_provider.py (TargetProvider)

src/core/ (zero ML deps: numpy + pyyaml)
    └→ src/env/ (adds gymnasium)
        └→ src/agent/ (adds sb3-contrib, torch)
    └→ src/ga/ (adds deap)
    └→ src/dashboard/ (adds streamlit, plotly, pandas)

src/logging_util/ (stdlib logging only)
```
