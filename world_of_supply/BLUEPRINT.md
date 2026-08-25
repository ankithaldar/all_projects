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
main.py                 shim delegating to world_of_supply.cli (single-process use)
.gitignore              render artifacts (out/), caches, egg-info
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

### Design-pattern map

| Pattern | Where | Purpose |
|---|---|---|
| Strategy | `routing.py` (traversable predicate), `rl/rewards.py` (`RewardShaper`), `policies.py` (`ControlPolicy`) | swappable path-finding, reward shaping, and control behavior |
| Template Method | `facility.py` (`FacilityCell._install_units`) | subclasses install only their units; base owns id/books/act fan-out |
| Builder | `scenario.py` (`WorldBuilder`) | assembles the full wired world from `ScenarioConfig` |
| Protocol / duck typing | `base.py` (`Agent`), `rl/rewards.py`, `policies.py` | structural interfaces, no inheritance coupling |
| Factory Method | `facility.py` (`_distribution_unit`, concrete cells) | unit assembly from one shared `FacilityConfig` |
| Dispatch table | `rendering/status.py` (`singledispatchmethod`), `rendering/sprites.py` | type-driven formatting and sprite selection |
| Facade + lazy import | `__init__.py` (PEP 562 `__getattr__`), `rl/__init__.py` | single import surface; torch/ray load only on demand |
| Value objects | `economy.py`, all `*Config`/`*Economy` dataclasses | immutable-ish money and parameter carriers |
| Composition | `facility.py` units (storage/consumer/…/seller) | facilities are aggregations of independent agents |

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
`try_unloading` delivers what fits and **zeroes the payload regardless —
units that do not fit are lost** (legacy semantics, verified by test) — and
notifies the buyer's consumer unit for the delivered volume. Freight cost
per tick = `payload × cost × |step|`.

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
ordering when broke or when **total storage occupancy plus all open orders**
exceeds capacity (legacy booked-inventory rule); picks the most
under-stocked BOM input from a random exporting supplier.

### Rendering & analytics

- **`rendering/sprites.py`**: neighbor-aware box-drawing glyphs (══║╔╬…).
- **`rendering/status.py`**: `WorldStatusFormatter` uses
  `singledispatchmethod` to format worlds/facilities/trucks/storages into
  YAML-safe nested lists; trucks use the legacy LOAD/MOVE/UNLD/BACK wording
  with route progress bars in fleet listings.
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
distribution class, resolves config across Ray versions (`cfg`/`config`),
and — when recurrent — feeds RLLib's `state_in` tensors into the LSTM
(normalizing batch-first to layer-first) so state carries across timesteps.

**Training** (`rl/training.py`): `build_ppo_algorithm()` assembles PPO via
the modern builder (torch framework; `_configure()` applies scalar settings
version-tolerantly — `lr`/`gamma`/`train_batch_size` are plain attributes on
new RLLib, `training()` kwargs on older; runner counts go through
`env_runners` with a `rollout` fallback). Per-policy networks match the
legacy widths (`POLICY_MODEL_CONFIGS`: producer 128×2, consumer 256×2).
Policies: trainable `ppo_producer/ppo_consumer` + frozen clones;
`make_policy_mapping_fn()` returns a mapping plus a mutable state dict so
`apply_curriculum()` can promote facility prefixes mid-run (restores the
legacy `update_policy_map`; CLI flag `--curriculum-warehouses` promotes
warehouses at 25%/35%). `train()` pushes curriculum counters into every
worker env, resolving the runner group across RLLib generations and
unwrapping Gymnasium wrapper chains (see §3). `describe_model()` prints the
per-role architectures (legacy `print_model_summaries`).
`evaluate_scripted()` scores the heuristics headlessly — no Ray required.

**Heuristics** (`rl/heuristics.py`): `ScriptedAgentController` produces a
full action dict from raw features (fulfillment-ratio reordering) usable
directly with `env.step()` — no Ray required.

---

## 3. Codeflows

### Episode lifecycle

```
reset(seed) → WorldBuilder.build(scenario, seed) → encode obs for all 18 agents
loop up to episode_duration / downsampling_rate decisions (default 50):
    agent actions → decode → world ticks (1 + 19 no-ops) → rewards → obs
until time_step ≥ episode_duration → truncated['__all__'] = True
info dicts carry raw per-agent features on every step (feeds heuristics/debug)
```

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
train(algo, n, curriculum=()):                   # each iteration
  apply_curriculum(state, i, n, curriculum)      # promote prefixes at fractions
  worker_set = algo.env_runner_group             # new Ray
             | algo.workers / algo.workers()     # legacy fallbacks
  for runner in worker_set:                      # foreach_env_runner / foreach_worker
    for env in unwrap(runner.env):               # VectorEnv.envs → OrderEnforcing →
      env.set_iteration(i, n)                    #   .unwrapped; skip None placeholders
  result = algo.train()                          # sample + GAE + PPO update
  log: reward  ← episode_reward_mean | env_runners.episode_return_mean
       steps   ← timesteps_total         | env_runners.num_env_steps_sampled_lifetime
```

Version-tolerance notes (Ray 2.5x verified):
- `Algorithm.workers` is a deprecated callable that *raises* on access in
  recent releases — `env_runner_group` is probed first, never touched after.
- Runner envs arrive wrapped (`SyncVectorMultiAgentEnv` → `OrderEnforcing`
  → `WorldOfSupplyEnv`); `_apply_to_envs()` unwraps via `.unwrapped` and
  skips `None` placeholder slots (local runners hold no env when remote
  workers exist).
- Result dicts no longer carry top-level `episode_reward_mean` /
  `timesteps_total`; the CLI log falls back to the nested `env_runners`
  schema.

---

## 4. Reference tables

**Products:** `['lumber', 'steel', 'toy_car']` · **Agents:** 18 (9 facilities × 2 roles).

**Scenario defaults:** capacity 20 (retailers 10), storage $1/unit/tick,
freight $1/unit/cell, manufacturing $100/unit, wrong-order penalty $500/unit,
pending fee $4/order/tick, demand `50 − 0.005·price`, starting balances
$1000/$2000/$3000 by tier, episode 1000 ticks @ downsampling 20 → 50 decisions.

**Reward:** `reward = 0.9·global_blend + 0.1·own_total`;
`global_blend = (1−w)·mean_retail_profit + w·mean_all_profit`; `w: 0→0.8`.

**Money ledger** — who books what:

| Event | Buyer/owner ledger | Counterparty |
|---|---|---|
| Order placement (valid) | loss `−price·qty` (prepayment) | seller revenue `+price·qty` same tick |
| Wrong-product order | loss `−$500·qty` | seller stats only (`total_wrong_order_penalties`) |
| Queued order, per tick | — | seller loss `−$4` per queued order |
| Production lot | — | owner loss `−$100·units` |
| Storage holding, per tick | — | owner loss `−$1·stored unit` |
| Freight, per moving tick | — | owner loss `−$1·payload·cell` |
| Retail sale | — | owner profit `+price·sold` (demand-capped) |
| Initial capital | profit `+$1000/$2000/$3000` by tier | — |

**`EnvConfig` defaults** (`rl/env.py`): `episode_duration=1000`,
`downsampling_rate=20` (→ 50 decisions/episode),
`global_reward_weight_producer=0.9`, `global_reward_weight_consumer=0.9`,
`scenario=ScenarioConfig()`, `seed=None`. Pass as
`build_ppo_algorithm({'env': EnvConfig(...)})`; RLLib forwards the dict to
worker env constructors, where `coerce_env_config()` accepts a wrapped
`{'env': EnvConfig}`, a bare `EnvConfig`, a dict of EnvConfig fields, or
`None`. Custom reward shapers inject via `env_class=MyEnv` on
`build_ppo_algorithm`.

## 5. Test suite (51 tests)

| File | # | Covers |
|---|---|---|
| `test_economy.py` | 5 | BalanceSheet arithmetic, `sum()`, repr |
| `test_storage_manufacturing.py` | 8 | capacity all-or-nothing/partial, atomic take, holding cost, BOM consumption + space rule |
| `test_transport_distribution.py` | 7 | valid/wrong orders, payment booking, full delivery cycle, truck FSM states, unfitting units lost |
| `test_consumer_seller.py` | 4 | demand curve, sell-out, open-order pruning, booked-capacity rule counts all stock |
| `test_world_scenario.py` | 5 | 9 facilities registered, supplier connectivity, act() sheets, balance conservation, seeded layout determinism |
| `test_policies.py` | 3 | full control coverage, retail revenue within 80 ticks |
| `test_rl_env.py` | 5 | agent/space coverage, normalized obs bounds, downsampling timing, truncation at horizon, curriculum shaping |
| `test_heuristics_rendering.py` | 6 | scripted actions in-bounds, scripted episode to horizon, status formatter, legacy truck wording, crossing sprite, PNG render |
| `test_models.py` | 4 | FacilityNet forward shapes, LSTM state, gradient flow, legacy-width 2-layer trunk (skipped without torch) |
| `test_training.py` | 4 | role mapping, toy-only freezing, curriculum promotion at thresholds, idempotence |

## 6. Extension guide

**Add a facility type** (e.g. a battery factory):
1. `facility.py`: subclass the unit mix you need and override `_install_units`.
2. `scenario.py`: instantiate it inside `WorldBuilder.build` and connect roads.
3. `rendering/renderer.py`: add a letter to `_FACILITY_GLYPHS`.
4. Scripted baselines: add a price to `policies.py` `_DEFAULT_PRICES` and
   `rl/heuristics.py` `_CLASS_PRICE_INDEX`.
Observation one-hots, action spaces, and encoder dims adapt automatically
(facility types and products are discovered from the reference world).

**Add a product**: declare it in BOMs in `scenario.py` (producer output +
consumer inputs); the sorted product list — and therefore observation blocks
and consumer action width — updates itself. Update the `PRODUCT_IDS`
informational constant to match.

**Change action levels**: edit `PRICE_LEVELS`/`RATE_LEVELS`/`QUANTITY_LEVELS`
in `rl/actions.py`. Note the env builds `MultiDiscrete([len(PRICE_LEVELS), 6])`
— keep the rate/quantity tuple length and the env space in sync.

**Custom rewards**: implement the `RewardShaper` protocol
(`shape(step_sheets, iteration, n_iterations) -> {agent_id: float}`) and
inject it by subclassing the env — remote workers rebuild envs from the
class, so subclassing (not attribute patching) is the reliable path:

```python
class MyEnv(WorldOfSupplyEnv):
  def __init__(self, config=None):
    super().__init__(config)
    self.reward_shaper = MyRewardShaper()

algo = build_ppo_algorithm({'env': EnvConfig()}, env_class=MyEnv)
```

**Scenario variants**: any `ScenarioConfig(...)` overrides (tier counts,
balances, capacities, costs, grid size) flow through `WorldBuilder.build`.
Default-layout tests assert 9 facilities — update them if you change tiers.

## 7. Glossary

| Term | Meaning |
|---|---|
| Simulation tick | One `World.act` — every unit steps once |
| Decision tick | One `env.step` — applies controls, then `downsampling_rate−1` untouched ticks |
| Open order | Units ordered but not yet received, tracked per buyer per source per product |
| Booked inventory | Storage occupancy plus all open orders; the reorder stop-limit |
| Prepayment | Buyers pay `price·qty` at order placement; seller books revenue the same tick |
| Wrong order | Ordering a product the supplier does not output → $500/unit penalty |
| Frozen policy | A policy clone excluded from `policies_to_train`; samples actions, never learns |
| Curriculum promotion | Moving a facility prefix from frozen to trainable PPO policies at an iteration threshold |
| Global blend | Curriculum-weighted mix of mean retailer revenue and mean facility profit in every reward |

## 8. Troubleshooting

- **`ModuleNotFoundError: No module named 'world_of_supply'` in Ray workers**
  → the package is not installed; Ray workers are separate processes and do
  not inherit `sys.path` shims. Run `pip install -e .` before `train`.
- **`ValueError: 'workers' has been deprecated`** → old code path; the
  current `train()` resolves `env_runner_group` first (fixed in `d430f2a`).
- **Metrics print `None`** → worker crash swallowed by RLLib (usually the
  wrapper-unwrap issue above) or legacy result schema; both handled, verify
  you are on the latest `rl/training.py` + `cli.py`.
- **`plt.show()` does nothing / animation fails headless** → expected off
  notebooks; use `--render-dir` PNGs or set `MPLBACKEND=Agg`.
- **`Install gputil for GPU system monitoring`** → harmless Ray warning;
  `pip install gputil` silences it.

## 9. Migration notes (vs. legacy flat scripts)

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
- Ray 2.5x compatibility shims (see §3 training flow): runner-group
  accessors, Gymnasium wrapper unwrapping, nested result-schema metric
  fallbacks, and attribute-vs-kwargs algorithm settings.
- Training config propagation: `env_config` is now attached to the RLLib
  config and worker envs normalize wrapped/field-dict shapes via
  `coerce_env_config()` (previously workers silently used default
  `EnvConfig`, ignoring episode duration/downsampling/seed overrides).
- Second (semantic) parity audit fixes: truck leftovers are lost on unload
  (payload zeroed, matching legacy — an earlier refactor wrongly retained
  them); scripted-policy booking counts total storage occupancy, not just
  BOM-related products; fleet status strings restored to legacy
  LOAD/MOVE/UNLD/BACK format with route progress bars; `lspci` GPU probe
  returned to `analytics/hardware`.
- Third (tick-exact) audit fix: legacy runs the arrival flip and the unload
  in the SAME `Transport.act` call (two sequential `if` blocks); the port
  had used `elif`, delaying delivery by one tick and one freight charge.
  Now tick-exact, verified by a forced-path trace comparison.
- Empirical parity harness: legacy vs refactored are compared side-by-side
  (layout, sheets, storage/manufacturing/seller/consumer sequences,
  transport trace, scripted policy, reward formula, observation features,
  action decoding) — legacy sources live in `src/legacy/`.

### Legacy file → new module map

| Legacy flat script (deleted) | New modules |
|---|---|
| `world_of_supply_environment.py` | `economy`, `base`, `geography`, `routing`, `storage`, `manufacturing`, `transport`, `distribution`, `consumer`, `seller`, `facility`, `world`, `scenario`, `policies` |
| `world_of_supply_rllib.py` | `rl/agents`, `rl/observations`, `rl/actions`, `rl/rewards`, `rl/env`, `rl/heuristics` |
| `world_of_supply_rllib_models.py` (TF) | `rl/models` (PyTorch) + `rl/rl_modules` (RLLib wrapper) |
| `world_of_supply_rllib_training.py` | `rl/training` + `cli.py` |
| `world_of_supply_renderer.py` | `rendering/sprites`, `rendering/status`, `rendering/renderer` |
| `world_of_supply_tools.py` | `analytics/tracker`, `analytics/hardware` |
| `world-of-supply.ipynb` | replaced by `main.py` CLI + `tests/` |

## 10. End-to-end validation (verified run)

| Stage | Command | Observed result |
|---|---|---|
| Tests | `pytest tests/` | 43 passed |
| Simulation + render | `simulate --ticks 120 --render-dir out/frames` | 15 PNG frames; balance oscillating around +$3k…+$14k |
| Chain flow | scripted policy, 400 ticks | retailers sold 40 units by tick 400; global balance **+$28,528** |
| RL baseline | `baseline --episodes 3` | −4,409 / +234,720 / +234,704 total reward |
| PPO training | `train --iterations 8 --toy-only` | 16k env steps sampled; curriculum ramping (w = 0.8·i/n reaches 0.7) |
| Curriculum + summaries | `train --iterations 4 --toy-only --curriculum-warehouses` | warehouses promoted at iteration 1 (`int(0.25·4)`); producer 128×2 / consumer 256×2 summaries printed |

Training rewards start deeply negative and fall further early on — expected
dynamics: untrained consumers order random products, and ~1/3 of those are
wrong-product orders penalized at $500/unit, amplified by the curriculum
blending global (penalty-laden) profit into every agent's reward. The
scripted baseline (+234k/episode) is the reference the agents must beat.

## 11. Getting started

```bash
pip install -e '.[dev]'    # REQUIRED for training: Ray workers import the
                           # installed package, sys.path shims don't propagate
pytest tests/
python main.py simulate --ticks 120 --seed 42 --render-dir out/frames
python main.py baseline --episodes 3 --seed 7
python main.py train --iterations 8 --toy-only
python main.py train --iterations 20 --toy-only --curriculum-warehouses
```

CLI subcommands: `simulate` (`--ticks --seed --render-dir`), `baseline`
(`--episodes --seed`), `train` (`--iterations --toy-only`). A
`world-of-supply` console script is installed alongside the package.
Render artifacts land in `out/` (git-ignored).

**Google Colab:** `colab_run.sh` bootstraps everything (interpreter pick
3.10–3.13, `pip install -e .`, tests, demo, baseline, short training) —
see the usage header in the script; artifacts go to `/content/wos_outputs`
or Google Drive with `MOUNT_DRIVE=1`.

## Appendix A — Legacy → refactored API map (1:1)

Legacy sources: `src/legacy/` (reference-only, TensorFlow/Ray-1.x era).
Every legacy symbol and where it lives now; ✱ = intentional behavior fix.

### `world_of_supply_environment.py`

| Legacy | Refactored | Notes |
|---|---|---|
| `BalanceSheet` (+ `total/__add__/__sub__/__radd__/__repr__`) | `economy.BalanceSheet` | identical API |
| `Cell`, `TerrainCell`, `RailroadCell` | `geography.Cell/TerrainCell/RailroadCell` | identical |
| `Agent` (ABC, `act`) | `base.Agent` (Protocol) | structural typing |
| `Transport` attrs: `source, destination, path, location_pointer, step, payload, product_id, requested_quantity, economy` | `transport.Transport` (same names) | + derived `state` property |
| `Transport.schedule/path_len/is_enroute/current_location/try_loading/try_unloading/act` | same methods | `try_unloading` returns delivered count; leftovers lost ✱ |
| `Transport.Economy.step_balance_sheet(transport)` | `TransportEconomy.step_balance_sheet(payload, step)` | args flattened |
| `BillOfMaterials` (+ `input_units_per_lot`) | `manufacturing.BillOfMaterials` | `inputs` has `default_factory` |
| `StorageUnit.Economy.step_balance_sheet(storage)` | `StorageEconomy.step_balance_sheet(used_capacity)` | |
| `StorageUnit.Config` | fields folded into `FacilityConfig` | |
| `StorageUnit.used/available_capacity/try_add_units/try_take_units/take_available/act` | `storage.StorageUnit` same methods | |
| `DistributionUnit.Economy` (`unit_price, wrong_order_penatly, pending_order_penalty, order_checkin, total_*`; `profit`) | `DistributionEconomy` | typo fixed: `wrong_order_penalty`; penalty tallies stay negative ✱ |
| `DistributionUnit.Config` | fields folded into `FacilityConfig` | |
| `DistributionUnit.Control(unit_price)` | `FacilityControl.unit_price` | one shared control object |
| `DistributionUnit.Order` | `distribution.Order` | |
| `DistributionUnit.place_order/act`, attrs `fleet, order_queue` | `distribution.DistributionUnit` same | freight costs booked as losses ✱ (legacy sign bug) |
| `ManufacturingUnit.Economy.cost/step_balance_sheet` | `ManufacturingEconomy.step_balance_sheet` | `cost` folded in |
| `ManufacturingUnit.act`, `Config.unit_manufacturing_cost` | `manufacturing.ManufacturingUnit.act`, `FacilityConfig` field | |
| `ConsumerUnit.Economy` (purchase/received totals) | `consumer.ConsumerEconomy` | |
| `ConsumerUnit.Control` (3 fields) | `FacilityControl.consumer_*` | |
| `ConsumerUnit.act/on_order_reception/_update_open_orders` | `consumer.ConsumerUnit` (`_shift_open_order`) | setdefault-prune: tolerates unsolicited deliveries ✱ |
| `SellerUnit.Economy` (+ `market_demand/profit/step_balance_sheet`) | `seller.SellerEconomy` | `profit`/sheet folded into `SellerUnit.act` |
| `FacilityCell.Config` (5-way dataclass inheritance) | `facility.FacilityConfig` (flat dataclass) | |
| `FacilityCell.EconomyConfig` | `FacilityConfig.initial_balance` | |
| `FacilityCell.Economy.deposit` | `FacilityEconomy.deposit` | |
| `FacilityCell.Control` (multi-inheritance) | `facility.FacilityControl` | all-optional fields |
| `FacilityCell.__init__(x, y, world, config, economy_config)` | `FacilityCell.__init__(x, y, world, config)` | configs merged |
| per-subclass `__init__` overrides | `_install_units` (Template Method) | |
| `create_distribution_unit` | `facility._distribution_unit` | |
| `RawMaterials/Steel/Lumber/ValueAdd/Toy/Warehouse/RetailerCell` | same class names in `facility.py` | |
| `World.Economy.global_balance` | `world.WorldEconomy.global_balance` | |
| `World.Control/StepOutcome` | `world.Control/StepOutcome` | |
| `World.generate_id/act/create_cell/place_cell/is_railroad/is_traversable/get_facilities` | `world.World` same methods | `act` iterates registry; + `register_facility` |
| `World.c_tostring/map_to_graph` | `routing.build_traversable_graph` | int node encoding; cache invalidated on cell change ✱ |
| `World.find_path(x1, y1, x2, y2)` (lru_cache on method) | `World.find_path(start, goal)` | tuple args; stale-world cache leak fixed ✱ |
| `WorldBuilder.create(x, y)` | `WorldBuilder.build(config, seed)` | + seeded railroad jitter; elbow always defined ✱ |
| `WorldBuilder.default_facility_config/default_economy_config` | `FacilityConfig` + `ScenarioConfig` | |
| `WorldBuilder.connect_cells/build_railroad` | `WorldBuilder._connect/_build_railroad` | unbound `xi` NameError fixed ✱ |
| `SimpleControlPolicy.compute_control` | `policies.ScriptedSupplyChainPolicy.compute_control` | |
| `SimpleControlPolicy._find_source` | `._select_source` | booked = full occupancy ✱ |
| `SimpleControlPolicy.find_exporting_sources` | static method, same name | |
| `default_facility_control` (inner fn) | inlined `FacilityControl(...)` | |

### `world_of_supply_rllib.py`

| Legacy | Refactored | Notes |
|---|---|---|
| `Utils.agentid_producer/consumer` | `rl.agents.producer_agent_id/consumer_agent_id` | |
| `Utils.is_producer_agent/is_consumer_agent` | `rl.agents.is_producer/is_consumer` | |
| `Utils.agentid_to_fid` | `rl.agents.facility_id_of` | |
| `RewardCalculator.calculate_reward/_retailer_profit` | `rl.rewards.RetailerProfitRewardShaper.shape` (+ `_curriculum_weight`) | weights via constructor, not env_config; empty-mean guard ✱ |
| `StateCalculator.world_to_state` | `ObservationEncoder.encode_world` | returns (vectors, raws) same shape |
| `StateCalculator._state` | `ObservationEncoder.encode_facility` | |
| `StateCalculator._add_global/_add_bom/_add_distributor/_add_consumer_features` | folded into `encode_facility` | export-mask stride fixed ✱ (`[source][product]` row-major) |
| `StateCalculator._serialize_state` | `ObservationEncoder._flatten` + `normalize` | zero-range NaN guard ✱ |
| `StateCalculator._safe_div/_balance_norm` | dropped | dead/commented code |
| `ActionCalculator.action_dictionary_to_control` | `rl.actions.ActionDecoder.decode` | |
| `ActionCalculator._actions_to_control` + `get_or_zero` | `ActionDecoder._decode_facility` + `element()` | source index clamped ≥0 ✱; product index mod len ✱ |
| price/rate/small-control dicts | `PRICE_LEVELS/RATE_LEVELS/QUANTITY_LEVELS` | module constants |
| `WorldOfSupplyEnv.__init__/reset/step/set_iteration` | `rl.env.WorldOfSupplyEnv` | Gymnasium API: `reset(seed, options)`, 5-tuple step, `terminated`+`truncated` with `__all__` |
| `WorldOfSupplyEnv.agent_ids()` (method) | `agent_ids`/`possible_agents`/`agents` attributes | RLLib checker requirement |
| `WorldOfSupplyEnv.n_products/_product_ids` | `product_ids` attribute | sorted → deterministic ✱ (legacy `list(set)`) |
| — | `coerce_env_config`, `scripted_actions()` | new: dict-config normalization; scripted-action helper |
| `SimplePolicy` (+ `compute_actions/learn_on_batch/get/set_weights/get_config_from_env`) | dropped as RLLib policies | see `Producer/ConsumerSimplePolicy` below |
| `ProducerSimplePolicy._action` | `rl.heuristics.ScriptedProducer.action` | class→price-index table identical |
| `ConsumerSimplePolicy._action/_find_source` | `rl.heuristics.ScriptedConsumer.action` | same fulfillment-ratio logic on raw features |
| — | `rl.heuristics.ScriptedAgentController` | full action dict per world (replaces in-trainer hand-coded policies) |

### `world_of_supply_rllib_models.py`

| Legacy (TensorFlow) | Refactored (PyTorch) | Notes |
|---|---|---|
| `FacilityNet.__init__(hiddens_size=256, cell_size=64)` | `rl.models.FacilityNet(hidden_size=256, hidden_layers=1, lstm_cell_size=64, use_lstm=False)` | MLP default (legacy ran MLPs too); LSTM optional |
| keras `Input/Dense/LSTM/Dense` graph | `torch.nn.Sequential` trunk + `policy_head` + `value_head` | logits concat for MultiDiscrete |
| `forward_rnn(inputs, state, seq_lens)` | `forward(observations, state)` | + `_normalize_lstm_state` batch→layer first |
| `get_initial_state` (list of np zeros) | `FacilityNet.initial_state(batch)` list; RLModule `get_initial_state` dict `{h, c}` | new-stack contract |
| `value_function()` | `value_head` + `FacilityRLModule.compute_values` | ValueFunctionAPI for GAE |
| — | `rl.rl_modules.FacilityRLModule` | `action_dist_inputs`/`vf_preds` keys, bound `TorchMultiCategorical`, `cfg`/`config` resolution |

### `world_of_supply_rllib_training.py`

| Legacy | Refactored | Notes |
|---|---|---|
| `env_config` dict (`episod_duration`, weights, `downsampling_rate`) | `rl.env.EnvConfig` dataclass (`episode_duration` — typo fixed) | passed as `{'env': EnvConfig}`; coerced in env |
| `policies` dict + `policy_mapping_global` | `TRAINABLE_POLICIES`/`FROZEN_POLICIES` + `PolicySpec`s | frozen clones replace in-trainer hand-coded policies |
| `create_policy_mapping_fn/mapping_fn` | `make_policy_mapping_fn` | returns `(fn, mutable_state)` |
| `update_policy_map` (disabled stub) | `apply_curriculum` + `train(curriculum=...)` | functional again; CLI `--curriculum-warehouses` |
| `print_model_summaries` | `describe_model()` | torch reprs, legacy widths 128×2 / 256×2 |
| `print_training_results` | `cli.run_training.log` | nested-schema fallbacks |
| `play_baseline` (no-learning trainer) | `evaluate_scripted` (+ `cli baseline`) | env-direct, no RLLib needed |
| `train_ppo` | `build_ppo_algorithm` + `train` (+ `cli train`) | PPOConfig builder; version-tolerant settings/runner access |
| `filter_keys` | inline dict comprehension | |

### `world_of_supply_renderer.py`

| Legacy | Refactored | Notes |
|---|---|---|
| `Utils.ascii_progress_bar(done, limit, bar_lenght_char)` | `rendering.status.ascii_progress_bar(..., bar_length)` | typo fixed |
| `WorldRenderer.plot_sequence_images` | `rendering.renderer.NotebookAnimator.plot_sequence_images` | |
| `AsciiWorldStatusPrinter.status(world/facility/transport/storage)` (multipledispatch) | `WorldStatusFormatter.status` (`singledispatchmethod`) | stdlib dispatch; legacy LOAD/MOVE/UNLD/BACK words kept |
| `AsciiWorldStatusPrinter.cell_status` | dropped | unused |
| `AsciiWorldStatusPrinter.counter` | `counter_to_dict` | |
| `AsciiWorldRenderer.render` + inner `new_layer` | `AsciiWorldRenderer.render` + `_ascii_layers` | |
| `to_yaml` | inline `yaml.dump` | |
| `railroad_sprite` | `rendering.sprites.railroad_glyph` + `renderer._railroad_sprite` | table-driven |
| `multiline_textsize` / `Image.ANTIALIAS` | `multiline_textbbox` / `Image.LANCZOS` | Pillow ≥10 |
| `resources/` cwd-relative fonts | `world_of_supply/assets/` via `importlib.resources` | |

### `world_of_supply_tools.py`

| Legacy | Refactored | Notes |
|---|---|---|
| `SimulationTracker(eposod_len, n_episod, ...)` | `analytics.tracker.SimulationTracker(episode_length, n_episodes, ...)` | typos fixed; empty-name guard |
| `SimulationTracker.add_sample/render` | same methods | same 3-panel plot |
| `print_hardware_status` (TF `device_lib`, `ray.init(num_gpus=1)`) | `analytics.hardware.print_hardware_status` | torch CUDA probe + `lspci`; GPU-optional ray init |
