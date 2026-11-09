# World of Supply

Multi-agent supply-chain simulation and reinforcement-learning sandbox
(Ray RLlib 2.x, PyTorch). Steel and lumber factories feed toy factories;
warehouses ship to retailers, which sell into a price-sensitive market over
a railroad grid. The env allocates producer+consumer ids per facility
(18/9) — a _producer_ (price + production rate) and a _consumer_ (what to
order, from whom, how much) — but raw-material consumer actions are
no-ops: no ConsumerUnit installed (`src/world_of_supply/facility.py:434-448`,
`n_sources=0` in `src/world_of_supply/rl/actions.py:78-109`). The pure
simulation runs without heavy RL dependencies (lazy `rl` import).

## What it is

- Package entry point and public exports: `world_of_supply/__init__.py`
  (`src/world_of_supply/__init__.py:52-76`); lazy `rl` import via
  `__getattr__` so the simulation runs without Ray/Gymnasium
  (`src/world_of_supply/__init__.py:79-95`; lazy `FacilityNet` in
  `src/world_of_supply/rl/__init__.py:50-67`).
- Money primitive: `BalanceSheet` with `total() = profit + loss`
  (`src/world_of_supply/economy.py:16-34`, add/sub/sum in `:36-76`).
- Grid primitives: `Cell` / `TerrainCell` / `RailroadCell`
  (`src/world_of_supply/geography.py:11-56`); A* via
  `build_traversable_graph` / `shortest_path` over plain sizes + predicate
  (`src/world_of_supply/routing.py:21-73`).
- Units: `StorageUnit` (`src/world_of_supply/storage.py:40-148`),
  `ManufacturingUnit` + `BillOfMaterials`
  (`src/world_of_supply/manufacturing.py:24-138`), `DistributionUnit` + FIFO
  `Order` queue (`src/world_of_supply/distribution.py:21-162`),
  `ConsumerUnit` (`src/world_of_supply/consumer.py:37-142`),
  `SellerUnit` linear demand `intercept - slope * price`
  (`src/world_of_supply/seller.py:29-93`), `Transport` truck cycle
  load -> move -> unload -> return
  (`src/world_of_supply/transport.py:66-258`).

## Quickstart

All commands in this section run from inside `world_of_supply/`, not
from the repo root (there is no `main.py` or `pyproject.toml` at the
repo root). The Testing section below is the exception: it runs from
the repo root.

Install (Python >=3.10,<3.14 per
`world_of_supply/pyproject.toml:10`):

```bash
cd world_of_supply
pip install -e .
pip install -e '.[dev]'
```

Dependencies are unpinned ranges in `pyproject.toml`
(`world_of_supply/pyproject.toml:29-39`: e.g. `numpy>=1.26.0,<3.0.0`,
`networkx>=3.3`, `ray[rllib]>=2.40.0,<3.0.0`). There is no lockfile and
no reproducible-install claim.

Entry points: `main.py` is a shim that puts `src` on `sys.path` and
delegates to `world_of_supply.cli.main`
(`world_of_supply/main.py:15-19`). The installed console script is
`world-of-supply = world_of_supply.cli:main`
(`world_of_supply/pyproject.toml:51-52`). All runs below use the shim:

```bash
# simulate the hand-coded supply chain (defaults --ticks 60)
python main.py simulate --ticks 60 --seed 42 --render-dir out/

# score the scripted agents inside the RL environment
python main.py baseline --episodes 3 --seed 42

# train PPO (requires ray)
python main.py train --iterations 5 --toy-only
```

CLI flags (citations):

- `simulate`: `--ticks` (int, default 60), `--seed` (int|None),
  `--render-dir` (str|None); runs `run_simulation`
  (`src/world_of_supply/cli.py:271-286`, `117-138`).
- `baseline`: `--episodes` (int, default 3), `--seed` (int, default 42);
  runs `run_baseline` via `evaluate_scripted`
  (`src/world_of_supply/cli.py:289-303`, `157-173`).
- `train`: `--iterations` (int, default 5), `--toy-only`,
  `--curriculum-warehouses`; runs `run_training`
  (`src/world_of_supply/cli.py:306-321`, `237-268`).
- `run_simulation` builds with `WorldBuilder.build(seed=seed)`
  (`src/world_of_supply/cli.py:129`); `baseline` builds
  `WorldOfSupplyEnv(EnvConfig(episode_duration=200,
  downsampling_rate=20, seed=seed))`
  (`src/world_of_supply/cli.py:141-154`).

Programmatic use (verified signatures): run from `world_of_supply/`
after the install above (`pip install -e .`), or, without installing,
with `PYTHONPATH=src` from `world_of_supply/` (equivalently
`PYTHONPATH=world_of_supply/src` from the repo root). Without an
install or `PYTHONPATH`, plain `python` cannot import the src-layout
package (`world_of_supply/pyproject.toml:54-55`) — the `main.py` shim
(`world_of_supply/main.py:15-19`) avoids this by putting `src` on
`sys.path`, but a bare snippet does not:

```python
from world_of_supply import ScenarioConfig, WorldBuilder
from world_of_supply.policies import ScriptedSupplyChainPolicy
from world_of_supply.world import Control

world = WorldBuilder.build(ScenarioConfig(), seed=0)
policy = ScriptedSupplyChainPolicy(seed=0)
for _ in range(100):
    world.act(policy.compute_control(world))
print(world.economy.global_balance())
```

`WorldBuilder.build(config=None, seed=None)`
(`src/world_of_supply/scenario.py:382-411`); `World.act(control)`
(`src/world_of_supply/world.py:345-360`); `Control` wraps
`facility_controls` (`src/world_of_supply/world.py:171-180`);
`ScriptedSupplyChainPolicy.compute_control(world)`
(`src/world_of_supply/policies.py:81-102`).

## Architecture (after refactor)

### Supply-chain tiers

`PRODUCT_IDS = (lumber, steel, toy_car)`
(`src/world_of_supply/scenario.py:29-32`). Default map is 80x16
(`src/world_of_supply/scenario.py:61-62`): steel@(10,6) / lumber@(10,10)
(`:298-305`), toy factories at x=35 (`:321-325`), warehouses at x=50
(`:342-346`), retailers at x=70 (`:363-367`). Counts: 3 toy, 2 warehouse,
2 retailer plus 2 raw-materials = 9 facilities total
(`src/world_of_supply/scenario.py:76-78`); costs/capacities/balances in
`ScenarioConfig` (`:61-79`) and BOMs in `_bills_of_materials`
(`:188-199`).

### World / FacilityRegistry / WorldRouter

`World` is the grid step engine (`src/world_of_supply/world.py:194-360`).
Identity and lookup live in `FacilityRegistry`
(`src/world_of_supply/world.py:19-72`); cached routing lives in
`WorldRouter` (`src/world_of_supply/world.py:75-137`). `World` exposes
the registry dict via `facilities` (`:228-247`), `register_facility`
(`:257-266`), `place_cell` invalidates the router cache (`:268-279`),
`find_path(start, goal)` delegates to the router (`:318-332`),
`act(control)` fans out per facility and sums sheets (`:345-360`).
`WorldEconomy.global_balance()` sums facilities (`:140-168`).

### Scenario builders

`RailroadBuilder` lays L-shaped railroads with jittered elbows
(`src/world_of_supply/scenario.py:82-185`). `TierBuilder` builds one tier
at a time (`:202-368`). `WorldBuilder.build` coordinates them
(`:371-411`) with legacy shims `_connect` / `_build_railroad`
(`:413-447`).

### Facility composition (unit order)

`FacilityCell` is a Template Method: `_install_units(config)` hook
(`src/world_of_supply/facility.py:342-354`); `act(control)` fans one
`FacilityControl` out to every installed unit and deposits sheets
(`:412-422`). Unit execution order is Storage > Consumer > Manufacturing
> Distribution > Seller (`src/world_of_supply/facility.py:356-374`).
Concretely: `RawMaterialsFactoryCell` installs storage + manufacturing +
distribution (`:434-448`); `ValueAddFactoryCell` (parent of
`ToyFactoryCell`) installs storage + consumer + manufacturing +
distribution (`:481-496`); `WarehouseCell` installs storage + consumer +
distribution (`:519-532`); `RetailerCell` installs storage + consumer +
seller, no trucks (`:544-561`). Shared control fields are all-optional
(`:38-57`).

### Narrow protocols (SOLID seams)

`base.Agent.act` (`src/world_of_supply/base.py:22-48`),
`StorageUnitLike` (`:51-133`), `ConsumerUnitLike` (`:136-166`),
`DistributionUnitLike` (`:169-188`), `FacilityLike` (`:191-218`) let
units depend on narrow interfaces, not the `FacilityCell` god class.
`FacilityConfig` fragments into `StorageConfig` / `DistributionConfig` /
`ManufacturingConfig` / `SellerConfig`, each with
`from_facility_config` (`src/world_of_supply/facility.py:96-206`), with
factories `_storage_unit` / `_manufacturing_unit` /
`_distribution_unit` (`:243-291`).

### RL environment

Two agent ids per facility — producer (suffix `p`) and consumer (suffix
`c`) (`src/world_of_supply/rl/agents.py:13-47`), built per facility in
`_build_agent_ids` (`src/world_of_supply/rl/env.py:135-148`). The env
allocates producer+consumer ids per facility (18/9) — with 9 facilities
that is 18 agents — but raw-material consumer actions are no-ops: no
ConsumerUnit installed (`src/world_of_supply/facility.py:434-448`,
`n_sources=0` in `src/world_of_supply/rl/actions.py:78-109`). `EnvConfig`
defaults: `episode_duration
= 1000`, `downsampling_rate = 20` (50 decisions/episode), global weights
0.9 (`src/world_of_supply/rl/env.py:33-51`). `WorldOfSupplyEnv` follows
the Gymnasium API with `reset(seed, options)` and 5-tuple `step`
(`src/world_of_supply/rl/env.py:248-293`); one decision applies controls
then `downsampling_rate - 1` untouched ticks
(`src/world_of_supply/rl/env.py:275-279`). Spaces: producer
`MultiDiscrete[8, 6]`, consumer `MultiDiscrete[3, max_sources, 6]` built
in `_build_spaces` (`src/world_of_supply/rl/env.py:151-177`) from
`PRICE_LEVELS` / `RATE_LEVELS` / `QUANTITY_LEVELS`
(`src/world_of_supply/rl/actions.py:16-18`); `ActionDecoder.decode`
merges roles into `FacilityControl`
(`src/world_of_supply/rl/actions.py:42-110`). Observations: raw feature
dict plus min-max normalized flat vector
(`src/world_of_supply/rl/observations.py:23-299`, `normalize` `:285-299`).
Rewards: `RetailerProfitRewardShaper` blends mean retailer profit and
mean total profit with curriculum weight ramping to 0.8
(`src/world_of_supply/rl/rewards.py:47-137`). Model: `FacilityNet`
PyTorch MLP/LSTM with multi-discrete logits + value head
(`src/world_of_supply/rl/models.py:37-137`); `FacilityRLModule` wrapper
for RLLib new API stack (`src/world_of_supply/rl/rl_modules.py:94-247`).
Training: `build_ppo_algorithm` / `train` / `describe_model` /
`evaluate_scripted` (`src/world_of_supply/rl/training.py:201-452`);
heuristics `ScriptedAgentController` at action-vector level
(`src/world_of_supply/rl/heuristics.py:129-171`).

### Rendering / CLI

`AsciiWorldRenderer` + `NotebookAnimator`
(`src/world_of_supply/rendering/renderer.py:352-501`),
`WorldStatusFormatter` (`src/world_of_supply/rendering/status.py:46-353`),
`railroad_glyph` (`src/world_of_supply/rendering/sprites.py:30-58`).
CLI (`src/world_of_supply/cli.py:324-357`) wires simulation, baseline,
and training.

### Analytics (legacy-notebook only)

`SimulationTracker` episode metrics and plots
(`src/world_of_supply/analytics/tracker.py:79-161`) and
`print_hardware_status` CPU/memory/GPU probe
(`src/world_of_supply/analytics/hardware.py:157-171`) are legacy-notebook
only. The CLI does not wire them: `cli.py` imports only `policies`,
`rendering`, and `scenario` (`src/world_of_supply/cli.py:18-21`).

## Strict format rules

- Pylint style (`max-line-length=80`, `indent-string='  '` 2-space) is authoritative per user spec, not CI-enforced; ruff line-length 110 is not enforced.
- `pyproject.toml` declares `line-length = 110`, `indent-width = 2`,
  single-quote, 2-space format (`world_of_supply/pyproject.toml:63-73`).
- ruff is a known-failing non-gate.

## Project structure

```
world_of_supply/
├── src/world_of_supply/
│   ├── __init__.py        exports + lazy rl
│   ├── world.py           World, FacilityRegistry, WorldRouter
│   ├── scenario.py        ScenarioConfig, WorldBuilder, RailroadBuilder, TierBuilder
│   ├── facility.py        FacilityCell + 7 subclasses (RawMaterials/Steel/Lumber/ValueAdd/Toy/Warehouse/Retailer)
│   ├── base.py            narrow Protocols (Agent, FacilityLike, ...)
│   ├── economy.py geography.py routing.py
│   ├── storage.py manufacturing.py distribution.py transport.py
│   ├── consumer.py seller.py policies.py cli.py
│   ├── rl/                env, actions, observations, rewards, models,
│   │                      rl_modules, training, heuristics, agents
│   ├── rendering/         sprites, status, renderer
│   ├── analytics/         tracker, hardware
│   └── assets/            fonts + logo
├── tests/                 pytest suite (86 tests: 52 original + 34 seam)
├── src/legacy/            DEPRECATED - legacy ray.init() at import time, do not import
├── main.py                shim delegating to world_of_supply.cli
├── pyproject.toml         package metadata + dependency ranges
├── requirements.txt       dependency ranges mirror
├── colab_run.sh           Colab bootstrap (see Safety warnings)
└── BLUEPRINT.md           blueprint (stale test counts - see below)
```

`src/legacy/` holds 6 flat reference modules plus `resources/` and a
notebook (`src/legacy/README.md:8-19`); it is not installed by the
package (`src/legacy/README.md:3-6`). The legacy training module calls
`ray.init()` at module top level (`src/legacy/world_of_supply_rllib_training.py:33-34`)
and the legacy tools module calls `ray.init(num_gpus=1)`
(`src/legacy/world_of_supply_tools.py:51`): legacy `ray.init()` at import
time, do not import.

## Testing

86 tests: 52 original + 34 seam. Run from the repo root (not from
inside `world_of_supply/` as in Quickstart — if you are still there,
run `cd ..` first):

```bash
python -m pytest world_of_supply/tests/ -q
```

Testpaths are configured in `pyproject.toml`
(`world_of_supply/pyproject.toml:60-61`). See `BLUEPRINT.md` for a
guided tour of modules, codeflows (order lifecycle, reward shaping,
training loop), and design notes — but for test counts use this README
(see drift note below).

## BLUEPRINT drift note

`BLUEPRINT.md` claims 43 / 51 tests and is stale: section 5 is titled
"Test suite (51 tests)" (`world_of_supply/BLUEPRINT.md:360-373`) and
section 10 reports "43 passed" (`world_of_supply/BLUEPRINT.md:488-496`).
It also repeats the stale "6" facility count: "FacilityCell + 6 concrete
types" (`world_of_supply/BLUEPRINT.md:33`), corrected to 7 subclasses
above. This README is the primary guide; the actual suite is 86 tests (52
original + 34 seam).

## Safety warnings (Google Colab bootstrap)

`colab_run.sh` is an unpatched bootstrap script
(`world_of_supply/colab_run.sh:1-28`). Read the warnings before running
it.

- **Destructive delete.** colab_run.sh runs `rm -rf $PROJECT_DIR` before cloning (the script runs `rm -rf "$PROJECT_DIR"`, `world_of_supply/colab_run.sh:66`). NEVER set PROJECT_DIR to /, ~, empty, or a non-empty dir. Use a fresh, empty target dir (default `/content/all_projects`). A bad value deletes user data.
- **Google Drive mount.** With `MOUNT_DRIVE=1` the script runs `drive.mount('/content/drive')` and sets `OUT_DIR` under Drive (`world_of_supply/colab_run.sh:105-110`). This mounts your Google Drive - never use MOUNT_DRIVE=1 without understanding scope. Do not enable it blindly.
- **Accepted risk.** The script is unpatched (docs warning only, accepted risk). The `rm -rf` and Drive mount behavior above is documented here instead of patched.

Defaults in the script header (`world_of_supply/colab_run.sh:17-25`):
`GIT_URL` (already-public repo default), `GIT_BRANCH`
(`world_of_supply/main`), `PROJECT_DIR` (`/content/all_projects`),
`PROJECT_SUBDIR` (`world_of_supply`), `OUT_DIR`
(`/content/wos_outputs`). The script picks Python 3.10-3.13
(`world_of_supply/colab_run.sh:41-55`), installs with
`pip install -e .` (`:92-94`), runs tests (`:102`), demo + baseline
(`:114-117`), and training (`:120`).

## Source notes

| Claim | Source |
|---|---|
| Exports + lazy `rl` | `src/world_of_supply/__init__.py:52-95` |
| `main.py` shim | `world_of_supply/main.py:15-19` |
| `simulate --ticks/--seed/--render-dir` | `src/world_of_supply/cli.py:271-286`, `:117-138` |
| `baseline --episodes/--seed` | `src/world_of_supply/cli.py:289-303`, `:157-173` |
| `train --iterations/--toy-only/--curriculum-warehouses` | `src/world_of_supply/cli.py:306-321`, `:237-268` |
| Console script | `world_of_supply/pyproject.toml:51-52` |
| `World` / registry / router | `src/world_of_supply/world.py:19-137`, `:194-360` |
| Tiers + `WorldBuilder.build` | `src/world_of_supply/scenario.py:202-411` |
| Railroad / tier builders | `src/world_of_supply/scenario.py:82-185`, `:202-368` |
| Unit order Storage > Consumer > Manufacturing > Distribution > Seller | `src/world_of_supply/facility.py:356-374` |
| Narrow protocols | `src/world_of_supply/base.py:22-218` |
| Narrow configs | `src/world_of_supply/facility.py:96-206` |
| 18 agents / 1000 ticks / 50 decisions | `src/world_of_supply/rl/env.py:33-51`, `:135-148` |
| `WorldOfSupplyEnv.reset/step` | `src/world_of_supply/rl/env.py:248-293` |
| Action spaces / levels | `src/world_of_supply/rl/actions.py:16-18`, `src/world_of_supply/rl/env.py:151-177` |
| Observations / rewards | `src/world_of_supply/rl/observations.py:23-299`, `src/world_of_supply/rl/rewards.py:47-137` |
| `FacilityNet` / `FacilityRLModule` | `src/world_of_supply/rl/models.py:37-137`, `src/world_of_supply/rl/rl_modules.py:94-247` |
| Training / heuristics | `src/world_of_supply/rl/training.py:201-452`, `src/world_of_supply/rl/heuristics.py:129-171` |
| Rendering | `src/world_of_supply/rendering/renderer.py:352-501`, `src/world_of_supply/rendering/status.py:46-353`, `src/world_of_supply/rendering/sprites.py:30-58` |
| Analytics (legacy-notebook only) | `src/world_of_supply/analytics/tracker.py:79-161`, `src/world_of_supply/analytics/hardware.py:157-171`; CLI imports `src/world_of_supply/cli.py:18-21` |
| Dependency ranges (no lockfile) | `world_of_supply/pyproject.toml:29-39` |
| Ruff / pylint format | `world_of_supply/pyproject.toml:63-73`; parent `.pylintrc:241,262` |
| `rm -rf $PROJECT_DIR` / `drive.mount` / `MOUNT_DRIVE` | `world_of_supply/colab_run.sh:66`, `:105-110` |
| Legacy `ray.init()` at import time | `src/legacy/world_of_supply_rllib_training.py:33-34`, `src/legacy/world_of_supply_tools.py:51` |
| BLUEPRINT drift | `world_of_supply/BLUEPRINT.md:360`, `:488-493` |
