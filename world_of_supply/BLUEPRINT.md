# World of Supply — Code Blueprint

A multi-agent reinforcement learning (MARL) **supply-chain simulator**.
A railroad grid hosts steel/lumber factories → toy factories → warehouses →
retailers. Trucks ship products along A* paths; every facility keeps books
(`BalanceSheet`). Each facility is controlled by **two agents** — a *producer*
(price + production rate) and a *consumer* (what/how much to order) — trained
with PPO on Ray RLlib 2.x (PyTorch, new API stack), with hand-coded
heuristics as the behavioral baseline.

```
steel ─┐                                  toy_car ─┐                        toy_car ─┐
       ├─► ToyFactory(3) ── toy_car ──► Warehouse(2) ├─ toy_car ──► Retailer(2) ├──► market demand
lumber ─┘   make toy_car                 ship        $1400           sell      $1800·demand
```

---

## 1. Package layout

```
src/world_of_supply/
├── economy.py          BalanceSheet (profit/loss ledger arithmetic)
├── base.py             Agent protocol (act(control) -> BalanceSheet)
├── geography.py        Cell ABC + TerrainCell + RailroadCell
├── routing.py          graph building + A* shortest path (networkx)
├── storage.py          StorageUnit: bounded inventory, holding costs
├── manufacturing.py    BillOfMaterials + ManufacturingUnit
├── transport.py        Transport truck FSM + freight costs
├── distribution.py     DistributionUnit: order queue + fleet dispatch
├── consumer.py         ConsumerUnit: ordering + open-order bookkeeping
├── seller.py           SellerUnit: linear demand curve sales
├── facility.py         FacilityControl/Config, FacilityCell + 6 concrete types
├── world.py            World step engine, cached routing, global economy
├── scenario.py         ScenarioConfig + WorldBuilder (default map)
├── policies.py         ScriptedSupplyChainPolicy (hand-coded baseline)
├── cli.py              argparse CLI (simulate / baseline / train)
├── rendering/
│   ├── sprites.py      box-drawing glyph selection for railroads
│   ├── status.py       singledispatch status formatter + progress bars
│   └── renderer.py     layered PIL image renderer + notebook animator
├── analytics/
│   ├── tracker.py      SimulationTracker episode metrics + plots
│   └── hardware.py     CPU/RAM/GPU probe (torch/ray based)
├── rl/
│   ├── agents.py       producer/consumer agent-id codec
│   ├── observations.py ObservationEncoder (raw features + normalized vectors)
│   ├── actions.py      ActionDecoder + price/rate/quantity level tables
│   ├── rewards.py      RewardShaper protocol + curriculum reward shaping
│   ├── env.py          WorldOfSupplyEnv (Gymnasium/RLLib MultiAgentEnv)
│   ├── models.py       FacilityNet (PyTorch MLP/LSTM policy+value net)
│   ├── rl_modules.py   FacilityRLModule wrapper for RLLib new API stack
│   ├── heuristics.py   scripted producers/consumers at action-vector level
│   └── training.py     PPO config builder, train loop, baseline evaluation
├── assets/             fonts + logo used by the renderer
tests/                  pytest suite (simulation, RL env, models, rendering)
main.py                 shim delegating to world_of_supply.cli
```

Dependency direction (clean layering — no cycles):

```
rl/training ─► rl/env ─► rl/{observations,actions,rewards} ─► world/facility ─► units ─► economy/geography
     │              │                                              ▲
     └─► rl/rl_modules ─► rl/models (torch only)                   │
                    ▲                                              │
             rendering / analytics / policies ─────────────────────┘
```

Ray/torch imports are confined to `rl/`; the pure simulation and rendering
run without them.

---

## 2. Module deep-dive

### Simulation core

**`economy.py`** — `BalanceSheet(profit, loss)` with `total() = profit + loss`
(profit ≥ 0, loss ≤ 0), component-wise `+`/`-`, `sum()` support via `__radd__`.

**`geography.py`** — `Cell(x, y)` base; `TerrainCell` impassable filler,
`RailroadCell` traversable road.

**`routing.py`** — Strategy-style functions over plain grids:
`build_traversable_graph(size_x, size_y, is_traversable)` builds a networkx
graph; `shortest_path(...)` runs A*. The *world* owns caching.

**`storage.py`** — `StorageUnit`: capacity-checked `try_add_units`
(all-or-nothing or partial), atomic `try_take_units`, partial
`take_available`; books `-used_capacity × unit_storage_cost` per tick.

**`manufacturing.py`** — `BillOfMaterials(inputs Counter, output_product_id,
output_lot_size)`; `ManufacturingUnit.act` runs up to `control.production_rate`
lots, each requiring free space ≥ `lot_size − input_units_per_lot` plus full
BOM availability; books production cost.

**`transport.py`** — one truck with an explicit state machine
(`TransportState.IDLE/LOADING/EN_ROUTE/UNLOADING/RETURNING`) derived from
`step` (+1/-1/0) and `location_pointer`. `schedule()` precomputes the route;
`try_unloading` delivers what fits (rest is lost) and notifies the buyer's
consumer unit. Freight cost per tick = `payload × cost × |step|`.

**`distribution.py`** — `DistributionUnit`: FIFO `order_queue`;
`place_order` returns the buyer payment immediately (prepayment semantics):
valid products join the queue at the current `unit_price`; wrong products
incur the $500/unit penalty (stats). Queued orders cost $4/order/tick.
Idle trucks pop orders one per tick. Revenue is flushed from `order_checkin`.

**`consumer.py`** — `ConsumerUnit.act` orders from
`sources[control.consumer_source_id]`, books `-payment`, tracks
per-source/per-product `open_orders`; `on_order_reception` decrements them.

**`seller.py`** — `SellerUnit.act`: demand = `intercept − slope·price`
(defaults 50 / 0.005); sells as much stock as demanded at the offered price.

**`facility.py`** — Template Method composition:
`FacilityCell.__init__` assigns ids/balance then calls `_install_units(config)`
which each concrete type overrides:

| Facility | Units installed |
|---|---|
| SteelFactoryCell, LumberFactoryCell | storage + manufacturing + distribution |
| ToyFactoryCell | storage + consumer + manufacturing + distribution |
| WarehouseCell | storage + consumer + distribution |
| RetailerCell | storage + consumer + seller |

`act(control)` fans one shared `FacilityControl` (price, rate, product,
source, quantity — all optional) out to every unit, sums sheets, deposits.

**`world.py`** — `World.act(Control)` iterates facilities in registration
order (missing controls ⇒ no-op ticks), collects per-facility sheets into a
`StepOutcome`. Routing graph is cached and invalidated on cell changes.
`WorldEconomy.global_balance()` sums all facilities.

**`scenario.py`** — `ScenarioConfig` dataclass (sizes, costs, counts,
balances) + `WorldBuilder.build(config, seed)`: steel@(10,6)/lumber@(10,10),
toys@x=35, warehouses@x=50, retailers@x=70 on an 80×16 grid; L-shaped
railroads with jittered elbows (seeded rng; always defined, unlike legacy).

**`policies.py`** — `ScriptedSupplyChainPolicy` (Strategy): fixed prices per
class ($400/$1000/$1400/$1800), rate 4 lots, reorder quantity 8; stops
ordering when broke or booked stock exceeds capacity; picks the most
under-stocked BOM input from a random exporting supplier.

### Rendering & analytics

- **`rendering/sprites.py`**: neighbor-aware box-drawing glyphs (══║╔╬…).
- **`rendering/status.py`**: `WorldStatusFormatter` uses
  `singledispatchmethod` to format worlds/facilities/trucks/storages into
  YAML-safe nested lists; ASCII progress bars.
- **`rendering/renderer.py`**: three text layers (railroads, en-route trucks
  `*`, facility letters S/L/T/W/R) composited onto a dark PIL canvas with the
  logo and 3-column YAML status panels; fonts/logo load from package assets
  via `importlib.resources` (modern Pillow APIs: `multiline_textbbox`,
  `Image.LANCZOS`). `NotebookAnimator` plays frame sequences as HTML5 video.
- **`analytics/tracker.py`**: matrices of global/per-facility balances across
  episodes + 3-panel matplotlib/seaborn plot.
- **`analytics/hardware.py`**: CPU/mem/GPU probe using torch and ray when present.

### RL layer

**Agent model** (`rl/agents.py`): every facility gets `<FacilityId>p`
(producer) and `<FacilityId>c` (consumer) — 18 agents in the default scenario.

**Actions** (`rl/actions.py`):

| Role | Space | Meaning |
|---|---|---|
| Producer | `MultiDiscrete[8, 6]` | price ∈ {400…2000}, lots/tick ∈ {0,2,4,6,8,10} |
| Consumer | `MultiDiscrete[3, max_sources, 6]` | product index, source index (clamped), qty ∈ {0,2,4,6,8,10} |

**Observations** (`rl/observations.py`): per-facility raw feature dict
(facility-type/id one-hots, balance flag, BOM inputs/outputs, storage state,
own outbound queue, supplier export mask `[source][product]` row-major — the
legacy stride bug is fixed — open orders, global time/utilization). The flat
vector min-max normalizes the concatenation into `[0,1]` (zero-range safe);
raw dicts travel through `info` and feed the heuristics.

**Rewards** (`rl/rewards.py`): `RetailerProfitRewardShaper` blends mean
retailer revenue and mean total profit with a curriculum weight ramping to
0.8 over training; each agent mixes the global signal with its own facility
result at role-specific weights (default 0.9).

**Environment** (`rl/env.py`): `WorldOfSupplyEnv(MultiAgentEnv)` follows the
Gymnasium API (`reset(seed, options)` → `(obs, info)`;
`step` → 5-tuple with `terminated`/`truncated` including `'__all__'`). One
decision applies controls then runs `downsampling_rate−1` untouched ticks so
logistics progress between decisions (episode = 1000 ticks / 20 = 50
decisions). Exposes `possible_agents`/`agents` for RLLib's checker. Falls
back to plain `gymnasium.Env` when Ray is absent.

**Model** (`rl/models.py`): `FacilityNet(nn.Module)` — PyTorch port of the
legacy TF `FacilityNet`: Dense trunk → optional LSTM → concatenated
MultiDiscrete logits + scalar value head; `initial_state()` support.

**RLModule** (`rl/rl_modules.py`): `FacilityRLModule(TorchRLModule,
ValueFunctionAPI)` wires FacilityNet into RLLib's new API stack: emits
`action_dist_inputs` in all forward modes, `vf_preds` + `compute_values()`
for GAE during training, binds a per-module `TorchMultiCategorical`
distribution class, and resolves config across Ray versions (`cfg`/`config`).

**Training** (`rl/training.py`): `build_ppo_algorithm()` assembles PPO via
the modern builder (torch framework; version-tolerant scalar settings;
`env_runners`; per-policy `MultiRLModuleSpec`). Policies: trainable
`ppo_producer/ppo_consumer` + frozen clones; `make_policy_mapping_fn(
train_toy_factories_only=True)` mirrors the legacy setup where only toy
factories learn. `train()` pushes iteration counters into env workers (drives
the reward curriculum). `evaluate_scripted()` scores heuristics headlessly.

**Heuristics** (`rl/heuristics.py`): `ScriptedAgentController` produces a
full action dict from raw features (fulfillment-ratio reordering) usable
directly with `env.step()` — no Ray required.

---

## 3. Codeflows

### One decision step

```
env.step(action_dict)
 ├─ ActionDecoder.decode → Control{facility_id: FacilityControl}
 ├─ World.act(control)                       # 1 decision tick (see below)
 ├─ (downsampling_rate−1) × World.act({})    # sheets summed into outcome
 ├─ RewardShaper.shape(sheets, iter, n_iter) # blended per-agent rewards
 ├─ ObservationEncoder.encode_world          # normalized obs + raw infos
 └─ done when time_step ≥ episode_duration   # truncated['__all__']
```

### Inside `World.act`

```
for facility in world.facilities:
  facility.act(control.get(id))
    ├─ StorageUnit       → −capacity_used·$1
    ├─ ConsumerUnit      → order upstream, pay now, open_orders ↑
    ├─ ManufacturingUnit → BOM lot in → output lot out, −$100/unit
    ├─ DistributionUnit  → dispatch idle trucks, −$4/queued order, flush revenue
    └─ SellerUnit        → sell min(stock, demand(price)), +revenue
```

### Order lifecycle

```
ConsumerUnit ──place_order──► DistributionUnit.queue   (buyer pays now)
   valid: queued at unit_price (seller revenue booked now)
   wrong product: −$500/unit penalty both sides (seller stats only)
idle truck pops order → schedule(A*) → LOAD → MOVE($1/unit/cell) → UNLOAD
   fits? delivered + buyer.on_order_reception (open_orders ↓)
   else leftover units lost. Truck returns home → IDLE.
```

### PPO training

```
build_ppo_algorithm → PPOConfig(torch).environment.env_runners.multi_agent.rl_module
  policies: ppo_producer/consumer (trained) + frozen clones (not trained)
  mapping_fn: agent_id → policy (toy-only mode freezes everything else)
  modules: FacilityRLModule(FacilityNet) per policy w/ MultiCategorical dist
train(algo, n):
  algo.workers.foreach_env(set_iteration(i, n))   # reward curriculum ramp
  result = algo.train()                           # sample + GAE + PPO update
```

---

## 4. Reference tables

**Products:** `['lumber', 'steel', 'toy_car']` · **Agents:** 18 (9 facilities × 2 roles).

**Scenario defaults:** capacity 20 (retailers 10), storage $1/unit/tick,
freight $1/unit/cell, manufacturing $100/unit, wrong-order penalty $500/unit,
pending fee $4/order/tick, demand `50 − 0.005·price`, starting balances
$1000/$2000/$3000 by tier, episode 1000 ticks @ downsampling 20 → 50 decisions.

**Reward:** `reward = 0.9·global_blend + 0.1·own_total`;
`global_blend = (1−w)·mean_retail_profit + w·mean_all_profit`; `w: 0→0.8`.

## 5. Migration notes (vs. legacy flat scripts)

- TensorFlow replaced by PyTorch everywhere; RLlib new API stack replaces
  `PPOTFPolicy`/`build_trainer`; `gymnasium` replaces `gym`.
- Fixed legacy bugs: consumer-side source-mask stride, `_safe_div` dead code,
  distribution sign error on freight costs, unbound railroad elbow variable,
  zero-range NaN normalization, lru-cache retaining stale worlds.
- Sign convention cleaned: buyers book losses directly; penalties tracked
  explicitly (`wrong_order_penalty` renamed from typo `penatly`).
- Hand-coded baselines now run at action-vector level against the same env;
  mixed hand-coded-in-trainer policies are approximated by frozen PPO-policy
  clones excluded from updates.

## 6. Getting started

```bash
pip install -e '.[dev]'
pytest tests/
python main.py simulate --ticks 60 --render-dir out/
python main.py baseline --episodes 3
python main.py train --iterations 20 --toy-only
```
