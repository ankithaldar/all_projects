# Crafting Game RL Simulation

A reinforcement-learning agent and genetic-algorithm baseline for optimizing batch production schedules in a 23-item crafting game.

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
make test
```

## Training the RL Agent

Train the MaskablePPO agent (~30 min on 8-CPU laptop with frame-skip=5):

```bash
# Default training (2M timesteps, frame_skip=1)
make train

# Faster training with frame-skip
python scripts/train.py --config config/training.yaml --frame-skip 5 --timesteps 500000
```

The trained model is saved to `output/models/` with a timestamp.

### Custom Targets

Edit `config/targets.yaml` with your level's target counts, then retrain:

```yaml
targets:
  gold: 14
  amethyst: 65
  necklace: 3
  artifact: 6
```

## Evaluating and Exporting

```bash
# Evaluate a trained model and export batch_schedule.txt
python scripts/evaluate.py --model output/models/final_model_20260423_120000.zip

# Output: output/batch_schedule_<timestamp>.txt
```

## GA Baseline

Run the NSGA-II genetic algorithm for comparison:

```bash
make baseline-ga

# Or with custom generations
python scripts/run_ga.py --config config/ga.yaml --generations 100
```

GA outputs:
- `output/ga_results/ga_batch_schedule_<timestamp>.txt` - Best schedule
- `output/ga_results/ga_population_log.jsonl` - Full population log per generation

## Streamlit Dashboard

```bash
make dashboard
# Opens browser at http://localhost:8501
```

Dashboard features:
- **Simulation**: Time slider to scrub through 30-day horizon, stash/coin/slot visualization
- **Bottleneck**: Material availability heatmaps, slot idle time, coin pressure analysis
- **RL vs GA**: Side-by-side metrics comparison, utilization and coin curves
- **Pareto Front**: 2D and 3D Pareto plots from GA multi-objective optimization
- **Target Editor**: Edit targets on-the-fly and re-simulate deterministically
- **Download Replay**: Export stash/coin/slot traces as CSV

## Understanding batch_schedule.txt

Each row represents one manufacturing decision:

```
tick_index  elapsed_minutes   item_name           batch_size_decided
0           0                 string              5
0           0                 wood                3
1           5                 metal               2
```

- `tick_index`: 5-minute game tick (0-2015 for 30-day horizon)
- `elapsed_minutes`: tick_index * 5
- `item_name`: which item to manufacture
- `batch_size_decided`: how many units to produce in this batch

### Spotting Bottlenecks

1. Look for items with long gaps between batches - they may be starved of materials
2. Cross-reference with coin balance in the dashboard - frequent zero-coin periods indicate over-spending
3. Compare RL vs GA schedules in the dashboard to identify where the RL agent differs from the evolutionary baseline

## Project Structure

```
config/              YAML configuration files
src/core/            Domain model (items, inventory, costs, slots, targets)
src/cat_game_env/    Gymnasium environment, reward shaper, frame skipper
src/agent/           MaskablePPO agent wrapper
src/ga/              NSGA-II genetic algorithm baseline
src/logging_util/    Rotating gzip-compressed logger
src/dashboard/       Streamlit visualization dashboard
scripts/             Entry-point scripts
tests/               Pytest test suite
output/              Runtime artifacts (models, logs, schedules)
```

## Game Rules

1. 23 items: 4 base (cotton, tree, rock, quartz) + 19 craftable
2. One manufacturing slot per craftable item (exclusive)
3. Batch cost formula: `init_cost * (1 + 0.5*(n-1))` per unit n
4. Materials consumed: `req_unit_raw * batch_size` from stash
5. Crafting completes after `craft_time` minutes
6. Can't start if materials insufficient
7. All items can manufacture in parallel
8. Coins generated: 210 per 5-minute tick
9. Goal: meet level targets within 30-day horizon
