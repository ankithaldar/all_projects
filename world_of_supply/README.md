# World of Supply

A multi-agent **supply-chain simulation** and reinforcement-learning sandbox.

Steel and lumber factories feed toy factories; warehouses ship to retailers,
which sell into a price-sensitive market over a railroad grid. Every facility
is controlled by two agents — a *producer* (price + production rate) and a
*consumer* (what to order, from whom, how much) — trainable with PPO
(Ray RLlib 2.x, PyTorch) or replaceable with hand-coded heuristics.

## Layout

```
src/world_of_supply/
├── economy.py            BalanceSheet money primitive
├── base.py               Agent protocol
├── geography.py          grid cells (terrain / railroad / facility base)
├── routing.py            networkx A* path finding
├── storage.py            capacity-bounded inventories
├── manufacturing.py      BOM production units
├── transport.py          trucks: load → move → unload → return
├── distribution.py       order queues + truck fleets
├── consumer.py           upstream ordering + open-order bookkeeping
├── seller.py             linear demand-curve sales
├── facility.py           facilities composed of the units above
├── world.py              step engine + cached routing
├── scenario.py           default 80×16 supply-chain builder
├── policies.py           scripted control baseline
├── cli.py                command-line interface
├── rendering/            sprites, status formatter, PIL renderer, animator
├── analytics/            episode tracker, hardware probe
└── rl/                   Gymnasium/RLLib env, encoders, FacilityNet (PyTorch),
                          RLModule wrapper, heuristics, PPO training setup
```

## Install

```bash
pip install -e .          # core + RL stack (torch, ray[rllib], gymnasium)
pip install -e '.[dev]'   # + pytest/ruff
```

Requires Python ≥ 3.10.

## Quick start

```bash
# simulate the hand-coded supply chain, print status, save frames
python main.py simulate --ticks 60 --render-dir out/

# score the scripted agents inside the RL environment
python main.py baseline --episodes 3

# train PPO (only toy-factory agents trainable by default)
python main.py train --iterations 20 --toy-only
```

Or programmatically:

```python
from world_of_supply import ScenarioConfig, WorldBuilder, ScriptedSupplyChainPolicy

world = WorldBuilder.build(ScenarioConfig(), seed=0)
policy = ScriptedSupplyChainPolicy()
for _ in range(100):
    world.act(policy.compute_control(world))
print(world.economy.global_balance())
```

## Tests

```bash
pytest tests/
```

See `BLUEPRINT.md` for a guided tour of every module, the codeflows
(order lifecycle, reward shaping, training loop), and design notes.
