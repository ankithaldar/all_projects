# Legacy sources (pre-refactor)

The original flat-module implementation of World of Supply, kept for
reference. It targets the **TensorFlow + Ray RLlib 1.x + gym** stack and is
**not** installed by the package — the maintained code lives in
`../world_of_supply/`.

## Contents

| File | Was |
|---|---|
| `world_of_supply_environment.py` | simulation core (cells, units, world, builder) |
| `world_of_supply_rllib.py` | RLlib 1.x multi-agent env + policies |
| `world_of_supply_rllib_models.py` | TensorFlow `FacilityNet` (LSTM) |
| `world_of_supply_rllib_training.py` | PPO/baseline training driver |
| `world_of_supply_renderer.py` | ASCII/PIL renderer |
| `world_of_supply_tools.py` | tracker + hardware probe |
| `world-of-supply.ipynb` | exploration notebook |
| `resources/` | fonts + logo (also copied into the package assets) |

## Running

The environment module works on modern Python; the RLlib modules need the
legacy pinned stack (Ray `<2`, TensorFlow `2.4-2.5`, `gym<0.22`,
`Pillow<10` — see the pins in git history at `6bc395e~1:requirements.txt`).

```bash
cd legacy
PYTHONPATH=. python -c "
import world_of_supply_environment as ws
world = ws.WorldBuilder.create(80, 16)
print(world.economy.global_balance())
"
```

See `BLUEPRINT.md` section 9 in the project root for the legacy-to-new
module map and the list of intentional behavior changes.
