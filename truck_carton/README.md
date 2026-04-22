# Truck-Carton: RL-Based 3D Bin Packing

Reinforcement learning system for optimizing 3D carton
loading into delivery trucks. Uses MaskablePPO with
curriculum learning to handle multi-truck, multi-store
delivery optimization.

## Features

- **3D bin packing** with gravity, fragility, and weight
  constraints
- **Multi-truck fleet optimization** with delivery route
  awareness
- **Action masking** via pre-computed valid placement
  candidates
- **Curriculum learning** across 3 difficulty stages
- **8-component shaped reward** covering utilization,
  displacement, grouping, fragility, support, weight,
  completion, and priority
- **3D visualization** of packing results

## Prerequisites

- Python 3.10+
- [UV](https://docs.astral.sh/uv/) package manager

## Setup

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone <repo-url> truck_carton
cd truck_carton
uv sync --dev
```

## Usage

### Train the agent

```bash
uv run python scripts/train.py
uv run python scripts/train.py --timesteps 500000
uv run python scripts/train.py --seed 123 --output ./my_run
```

Training outputs to `./output/` by default:
- `models/` — saved model checkpoints
- `tb_logs/` — TensorBoard logs
- `logs/` — evaluation logs

### Monitor training

```bash
uv run tensorboard --logdir ./output/tb_logs
```

### Evaluate a trained model

```bash
uv run python scripts/evaluate.py \
    --model ./output/models/best_model \
    --episodes 50 \
    --stage 0
```

### Visualize packing

```bash
# Random placements (no model)
uv run python scripts/visualize.py

# With a trained model
uv run python scripts/visualize.py \
    --model ./output/models/best_model
```

### Run tests

```bash
uv run pytest
```

## Project Structure

```
src/truck_carton/
├── config.py           # All tunable parameters
├── domain/             # Truck, Store, Carton models
│   ├── models.py
│   └── data_generator.py
├── packing/            # 3D spatial logic
│   ├── rotation.py     # 6 orientations, fragile filtering
│   ├── space3d.py      # Occupancy grid + height map
│   └── placement.py    # Validation + candidate enumeration
├── reward/             # 8-component reward system
│   ├── calculator.py
│   ├── utilization.py
│   ├── displacement.py
│   ├── grouping.py
│   ├── fragility.py
│   ├── support.py
│   ├── weight.py
│   ├── completion.py
│   └── priority.py
├── env/                # Gymnasium environment
│   ├── observation.py  # Dict observation space
│   ├── action.py       # Candidate-index actions
│   ├── masking.py      # Action mask provider
│   └── packing_env.py  # Main environment
├── curriculum/
│   └── manager.py      # Stage promotion logic
├── training/
│   ├── trainer.py      # MaskablePPO pipeline
│   └── callbacks.py    # Curriculum + metrics logging
└── evaluation/
    ├── metrics.py      # 9 evaluation metrics
    └── visualizer.py   # 3D matplotlib rendering
```

## Configuration

All parameters are in `src/truck_carton/config.py`:

- **TruckConfig** — dimension and weight ranges
- **CartonConfig** — dimension, weight, fragility prob
- **RewardWeights** — 8 reward component weights (alpha
  through theta)
- **CurriculumConfig** — 3 stages with promotion
  thresholds
- **TrainingConfig** — PPO hyperparameters

## Curriculum Stages

| Stage  | Trucks | Stores | Cartons |
|--------|--------|--------|---------|
| Toy    | 2      | 2      | 10      |
| Small  | 3      | 3      | 20      |
| Medium | 5      | 4      | 40      |

Promotion occurs when mean reward exceeds the threshold
over a sustained window of episodes.

## Algorithm

**MaskablePPO** (sb3-contrib) with:
- `Discrete(500)` action space — agent selects from
  pre-validated placement candidates
- `Dict` observation space via `MultiInputPolicy`
- Carton ordering determined by logistics (reverse
  delivery route, then priority)
- All candidates pre-validated for gravity, overlap,
  weight, and fragility

## Evaluation Metrics

- Volumetric utilization (per truck + fleet)
- Weight utilization per truck
- Average displacement per store stop
- Store-grouping compliance rate
- Fragility violation rate
- Physical support violation rate
- Weight constraint violation rate
- Priority accessibility score
- Total reward per episode
