# Cat Game NEAT

## What it is

A static browser app that searches for a low-cost crafting schedule for a 23-item
crafting game using 19 evolved NEAT policy networks. It is HTML + CSS + ES modules
only. There is no build step, no Python application code, and no backend; the
simulation and the NEAT engine run entirely in the browser. `index.html` is the
production entry point (`index.html:95`).

## How to run

Use the launcher (recommended). It serves only this project directory from a
static server bound to `127.0.0.1`, picks a locally generated, validated port,
and opens `http://127.0.0.1:<port>/index.html` in your default browser:

```
# POSIX (Linux/macOS) — requires python3
./start.sh
# Windows
start.bat
```

The server keeps running until Ctrl-C. Direct `file://` opening is unsupported
by browsers; use the launcher instead (`start.sh:1`, `start.bat:1`).

Or serve the directory over `http://` yourself and open `index.html`:

```
cd random_codes/0004_cat_game_neat
python3 -m http.server 8000
# open http://localhost:8000/index.html
```

Any static server works (for example VS Code Live Server). `http://` is required:
ES-module loading and the training Web Worker need it. Over `file://` the app
falls back to the embedded config, disables training, and shows only a labelled
greedy preview (`src/config.js:59`, `src/main.js:36`, `src/main.js:424`).

Tests are at `http://localhost:8000/tests.html`.

## Game rules (as implemented)

- **Items:** 23 total — 4 unlimited tier-0 bases (`cotton`, `tree`, `rock`,
  `quartz`) and 19 craftables, each with one craft slot (`config/crafting_tree.yaml`,
  `src/sim/engine.js:64`).
- **Targets:** 11 from `config/targets.yaml`, evaluated against the **current**
  stash each tick (`src/sim/engine.js:249`); consuming a target below its count
  emits `TARGET_UNMET` (`src/sim/engine.js:124`).
- **Batch cost:** `init_cost * b * (1 + 0.25 * (b - 1))` for a batch of `b`; the
  n-th unit costs `init_cost * (1 + 0.5 * (n - 1))` (`src/sim/cost.js:21-30`).
  Batches are capped at `MAX_BATCH = 20` (`src/config.js:12`).
- **Craft time:** `ceil(craft_time_minutes / 5)` ticks, clamped to >= 1
  (`src/config.js:30`).
- **Income:** +210 coins every 5-minute tick (`src/config.js:17-19`).
- **Horizon:** 8,064 ticks = 28 days (`src/config.js:17`, `src/ui/format.js:56`).
- **Starter modes** (`src/sim/demand.js:47`, `src/sim/engine.js:39-42`):
  - `PREDELIVERED_RESERVED` (default, the `reuse` preset): starter stash counts
    toward targets but cannot be consumed. The preset starts with 5,998,996
    coins and the stash in `src/config.js:38-50`.
  - `CONSUMABLE`: starter stash counts toward targets and can be consumed.
    Implemented in the engine/demand but **not selectable in the UI**.
  - `FREE_INGREDIENTS` (the `empty` preset): starter stash is consumable and does
    not count toward targets; starts with 0 coins and an empty stash
    (`src/config.js:51-56`).

## Manufacturing cap (exact requirement)

The engine enforces a hard cap so no craftable item manufactures more than the
gross quantity needed to meet the 11 targets **and** replace every unit those
targets consume downstream. The cap uses a NEW basis, `manufactureRequired`,
computed by `expandManufactureRequired` (`src/sim/demand.js:96`) in reverse
topological order. The shipped `required` map from `expandDemand` is left exactly
as-is (tests, tooltips and the NEAT policies depend on it); the two must not be
confused:

- `required` credits starter stash against downstream demand. In
  `PREDELIVERED_RESERVED` (the default `reuse` preset) starter units **cannot be
  consumed as ingredients**, so that credit creates holes — capping at `required`
  would leave the default preset unable to finish.
- `manufactureRequired` is the gross number of units a craftable must actually be
  manufactured. It is keyed by craftable id only; the four base items
  (`cotton`, `tree`, `rock`, `quartz`) are unlimited and never capped, and are
  absent from the map.

Per-item mode rules (`src/sim/demand.js:111-140`), where `st` is the item's
starter stash and `ingredientNeed` is demand pushed down from downstream
products:

- `PREDELIVERED_RESERVED` (starter counts toward the target but is not
  consumable): a target gets `max(0, targetNeed - st) + ingredientNeed`; a
  non-target gets `ingredientNeed` (starter earns no ingredient credit).
- `CONSUMABLE` (starter counts toward the target and is consumable): a target
  uses `min(st, targetNeed)` toward its target and offsets ingredient need with
  the leftover; a non-target gets `max(0, ingredientNeed - st)`.
- `FREE_INGREDIENTS` (starter is consumable and does not count toward the
  target): every item gets `targetNeed + max(0, ingredientNeed - st)`.

For the shipped `reuse` preset the golden values are bronze 867 (the old
`required` was 807), string 3932 (old 3576), amethyst 378 (old 180), and
silver 289.

**Engine enforcement.** `produced[id]` counts completed units and
`committed[id]` counts in-flight units (one slot per craftable). At each
free-slot decision (`src/sim/engine.js:195-210`) the engine computes
`remaining = max(0, manufactureRequired - produced - committed)` and clips the
batch to `min(materialsLimit, coinAffordableLimit, MAX_BATCH, remaining)`. This
holds the invariant `produced + committed <= manufactureRequired` at every tick
in all three starter modes. When `remaining === 0` the cap is evaluated
**before** policy/material/coin checks: the item is skipped with a
`DECISION_REJECTED` carrying the single frozen reason literal `REQUIRED_MET`,
both starvation flags are cleared, and no `STARVED_*` event is emitted.

**Target status and refill.** Target status is still decided each tick against
the **current** stash (`src/sim/engine.js:269-281`); the cap does not change that.
Because `manufactureRequired` counts gross manufacture (the target itself plus
everything downstream consumes), a target item that is spent as an ingredient
is manufactured again to refill it. The trade-off is deliberate: once the gross
total has been manufactured the cap blocks further batches, so the engine will
not re-manufacture beyond that total.

## Controls

- **Start / Pause / Reset** (`src/ui/controls.js:55-57`, `src/main.js:322-384`).
- **Speed:** `1x` = 500 ms/tick, `10x` = 50 ms/tick, `fast` = no delay
  (`src/main.js:215`).
- **Seed:** uint32, 0..4294967295 (`index.html:31`). Same seed + config reproduce
  the same genomes and schedule.
- **Preset:** `reuse` (pre-delivered, reserved) or `empty` (`index.html:33-38`).
- **Pop size:** input allows 1..500 (`index.html:40`), but training requires >= 2.
  `main.js` clamps the value to at least 2 before use; `validate.js` allows 1 and
  the controller also enforces >= 2 (`src/main.js:92-93`, `src/validate.js:363`,
  `src/neat/controller.js:310`).
- **Generations:** 1..10000 (`index.html:43`).
- **Logger:** substring filter on event type or item, and Clear log
  (`src/ui/controls.js:59-77`).

Training-budget defaults in the controller/worker are **10 generations, pop 20,
and a 180,000 ms wall clock** (`src/neat/controller.js:33-35`). The wall budget is
not exposed in the UI. The Generations input ships with 100 selected
(`index.html:43`), so an unmodified Start runs 100 generations. The worker always
emits a terminal `DONE` message, including on error and on wall-clock expiry
(`src/worker/train-worker.js:425`, `src/worker/train-worker.js:480-484`).

## Editable initial state

A fresh load / refresh always starts from the `empty` preset: 0 coins, an empty
19-item starter stash and the 11 targets from `config/targets.yaml`; no browser
storage is read and no YAML file is written (`src/config.js:151-158`,
`src/main.js:765-768`). The "Initial state" panel (`#initial-state-panel`) has one
numeric input for starting coins plus 11 target inputs and 19 starter-stash
inputs. The 19 stash inputs are listed in crafting-tree topological order
(`string, wood, metal, bronze, amethyst, orb, ribbon, needles, sparkles, silver,
gold, pendant, water, necklace, fire, waterstone, firestone, elementstone,
artifact`), and every input carries a `title` tooltip stating its meaning and
integer range (`index.html:51-101`).

The panel has three buttons. **Apply** is the panel's only run trigger: it reads
and validates the form and starts the run with the edited values through the same
`startAll` path as Start (`src/ui/initial-state.js:135-137`,
`src/main.js:752-756`). **Reset** stops the run and restores the active preset's
coins, targets and stash (`src/main.js:627-653`). **Reset to 0** is form-only: it
zeros the coins and all 19 stash inputs, leaves the 11 target inputs unchanged and
does not start a run (`src/ui/initial-state.js:141-146`,
`src/tests/ui.test.js:838`). None of these buttons writes to browser storage or
to the YAML files.

Values are read with strict integer coercion — empty, fractional, negative,
over-cap or junk-suffixed input falls back to the prefill value and logs only the
stable `CONFIG_INPUT_INVALID` code, never the raw input — then passed through
`validateInitialState`, the single authoritative schema, with bounds targets
`0..1e7`, coins `0..1e9` and stash `0..1e9` (`src/ui/initial-state.js:18-21`,
`src/validate.js:55-58`, `src/validate.js:194-215`). The validated null-prototype
maps are assigned by reference into the module config and then to the worker
START message and engine (`src/config.js:184-204`, `src/main.js:322-341`).

## Sim time

A simulated-time readout sits at the top of the page (`#wall-clock`,
`role="timer"`, `aria-live="off"`). It is derived from engine ticks, not real
elapsed time: `N min = tick × 5` minutes (`TICK_MINUTES = 5`), rendered with the
label `Sim time: N min`. It updates with each display snapshot, freezes on the
final value when the display run completes, and resets on Start/Apply. It measures
simulated game time only — never real `Date.now()` elapsed minutes — and is
written with `textContent` after `capString(...,120)`.

## UI

- **Train / DAG tab:** left-to-right SVG graph laid out by **computed dependency
  depth 0..10**, not the raw YAML `tier`, which is metadata only and inconsistent
  for 14 of 19 craftables (`src/sim/items.js:3-9`). Examples: `needles` has raw
  tier 2 but depth 3; `artifact` has raw tier 8 but depth 10. Each node shows its
  stash count, a live countdown, and a `×N` batch badge while crafting. Idle nodes
  are highlighted amber when material-starved, red when coin-starved, and
  dashed red when both; busy nodes show a blue busy style. A legend is in the DAG
  header (`index.html:66-72`, `styles.css:108-138`).
- **Manufactured-count badge:** each craftable node carries its own SVG `<text>`
  badge showing `produced/manufactureRequired`; base nodes show `n/a`. Values
  are read only from own keys of the engine-local cap maps, normalized to a
  non-negative integer, `capString`-limited to `MAX_STR = 120`, and written with
  `textContent` only (`src/ui/dag.js:28-45`, `src/ui/dag.js:274-292`). A craftable
  is styled `COMPLETE` (green ring + green badge) only when both values are
  finite and `produced >= manufactureRequired`; `COMPLETE` outranks the
  stash-based `target-met` style (`styles.css:147-152`). Node tooltips show
  separately labelled stash, `required`, `manufactureRequired`, `produced` and
  `committed` values, each capped at 120 (`src/ui/dag.js:310-320`).
- **Click a node** to highlight its full ancestry down to depth-0 bases and dim
  all other nodes/edges; click again or click the background to clear
  (`src/ui/dag.js:103-107`, `src/ui/dag.js:155-202`).
- **Tooltips** are present on DAG nodes and edges, stash rows, the coin/spent
  balance, logger rows, and controls (`src/ui/dag.js:94`, `src/ui/dag.js:132`,
  `src/ui/hud.js:31-39`, `src/ui/logger.js:37`, `index.html` control titles).
- **Stash panel** (top left) lists every item with live counts and shows the coin
  balance and cumulative `Spent` (`src/ui/hud.js`).
- **Logger tab:** 5,000-entry ring buffer with at most 600 DOM rows
  (`src/ui/logger.js:5`, `src/ui/logger.js:72-78`).
- **Results tab:** best (trained) schedule, totals, target status, feasibility
  verdict, and per-genome telemetry with per-item coin/time/batch/unit breakdown
  (`src/ui/results.js:37-113`).
- **Tab switching:** Train / DAG, Logger and Results all switch correctly:
  clicking a tab shows only its panel and keeps `is-active` / `aria-selected` in
  sync (`src/main.js:504-551`). The top-left stash panel, live coin counter, DAG
  `produced/manufactureRequired` badge and `COMPLETE` styling above are
  unchanged.

## Architecture

- `config/` holds the two YAML files. `src/config.js` fetches them (256 KiB cap),
  parses with js-yaml 4.1.1 (`CORE_SCHEMA`, `json: true`) loaded via CDN with an
  SRI pin in `index.html:9`, validates them with an allowlist, and falls back to
  an embedded JavaScript copy on any failure.
- `src/sim/*` is a DOM-free deterministic engine. It uses an explicit
  `totalSpent` accumulator and per-item coin/time attribution, so
  `sum(perItem.coinsSpent) === totalSpent` (`src/sim/engine.js:199-201`,
  `src/sim/engine.js:265-291`).
- `src/neat/*` evolves 19 separate in-house canonical NEAT genomes in dependency
  topological order. During evaluation, earlier items use frozen evolved genomes,
  the current item uses its candidate, and later items use a greedy policy
  (`src/neat/controller.js:140-153`). Fitness is lexicographic:
  `-(totalSpent * 100000 + completionTick)` when feasible, and an infeasible
  floor of `-1e12 + progressBonus` otherwise (`src/neat/fitness.js`).
- `src/neat/batch-planner.js`, `src/neat/priority.js` and
  `src/neat/policy-caps.js` compute offline batch caps and apply them through a
  single shared helper in both training and display (see "Solver improvements").
- `src/neat/batch-quantizer.js` and `src/neat/time-aligner.js` compute per-item
  batch quantums and per-product lead-time alignments; `effectiveCap` is the
  single cap authority shared by the planner and the engine.
- `src/ui/initial-state.js` renders and reads the editable Initial state panel;
  `src/main.js` owns the sim-time display and the config/preset/Apply/Reset
  propagation path (see "Editable initial state" and "Sim time").
- `src/ui/diagnostics.js` renders main-thread-only solver diagnostics.
- `src/worker/train-worker.js` performs silent, bounded rollouts and posts
  `PROGRESS | BEST | GENOME_RESULT | DONE | ERROR`. `src/main.js` runs the visible
  display and, once trained genomes arrive, a **second** main-thread display run
  driven by those genomes; the greedy idle preview does not write Results
  (`src/main.js:269-317`). Worker messages and UI controls are validated in
  `src/validate.js`; all message and ring buffers are bounded.

## Solver improvements (game rules unchanged)

These changes optimise the search for a schedule; no game rule changed (items,
recipes, cost formula, +210/tick, 8,064-tick horizon, targets, one slot per
item). See `docs/architecture.md` for contracts and File:Line citations.

### Batch planner

`src/neat/batch-planner.js` computes per-item batch caps offline from the exact
dependency critical path and the real superlinear cost (`src/sim/cost.js`).
Items on the critical chain (artifact/elementstone) get larger caps — fewer
batches, less time — while less-critical items get smaller caps — cheaper per
unit, more time. Caps are clamped to `[1, MAX_BATCH]` and the resulting critical
path must fit the 8,064-tick horizon. One shared helper, `applyBatchCaps`
(`src/neat/policy-caps.js`), applies the caps in both training and the display
runner, so both use the same limits; the engine's own material/coin/MAX_BATCH
clipping is unchanged.

Offline criticality (`src/neat/priority.js`) feeds only the planner. It is a
null-prototype map keyed by validated craftable ids; it never reaches a worker
message and never throttles coins at runtime — all 19 items feed target paths
and none is starved.

On the shipped `reuse` preset the computed plan estimates `estTicks = 6203`
(<= 8064) and `estimatedCost = 4,794,750`. Applied through the shared helper in
a real-engine run, the schedule is feasible at completion tick **2517** with
`totalSpent = 3,947,900`, versus the prior greedy feasible baseline of tick
6210 and `totalSpent = 7,111,975` (`src/tests/batch-planner.test.js:109`).

### Batch quantization and lead-time alignment

`src/neat/batch-quantizer.js` and `src/neat/time-aligner.js` add a post-clip
quantization pass over the same validated config and `manufactureRequired` basis;
no game rule changes.

- **Per-item quantum table.** `computeBatchQuantums(items)` returns a
  null-prototype map keyed only by validated craftable ids: each item's step is
  the greatest common divisor of every per-unit quantity with which downstream
  recipes consume it; a terminal item defaults to `1`. On the shipped items the
  steps are `string 2, wood 2, metal 4, bronze 3, amethyst 7, orb 2, ribbon 1,
  needles 2, sparkles 1, silver 1, gold 1, pendant 3, water 2, necklace 1,
  fire 1, waterstone 1, firestone 1, elementstone 1, artifact 1`
  (`src/neat/batch-quantizer.js:33-69`, `src/tests/batch-quantizer.test.js:9-20`).
- **Snapped to multiples, exact remainders.** A non-aligned accepted batch is
  rounded down to a positive multiple of its step; a final remainder smaller than
  a full quantum is kept exactly rather than snapped up. A sub-quantum clip with
  quantum headroom still available is rejected instead of snapped up
  (`src/neat/batch-quantizer.js:92-133`).
- **Post-clip quantization in the engine.** After all existing clips (`want`,
  materials, coins, `MAX_BATCH`, `manufactureRequired` remaining), the engine calls
  `quantizeBatch` before spending. A rejection emits the frozen
  `DECISION_REJECTED` literal `BELOW_QUANTUM`, latches a `blocked` telemetry flag,
  clears starvation flags, emits no `STARVED_*` event and leaves state untouched
  (`src/sim/engine.js:324-338`). `blocked` is exposed on the policy context and the
  snapshot DTO (`src/sim/engine.js:277`, `src/sim/engine.js:442-443`,
  `src/sim/engine.js:472-474`).
- **Per-product lead-time alignment.** `computeAlignment` works per product unit:
  over a product's craftable ingredients it takes the least common multiple of
  their craft ticks, sets `k = lcm / craftTicks(ingredient)` and
  `alignCap = ceil(q / k)`, and keeps a recipe only when `k <= q`,
  `k * craftTicks <= MAX_TICKS` and `alignCap <= MAX_BATCH`. An item feeding
  several recipes keeps the minimum feasible `alignCap`. The shipped
  `waterstone`, `firestone` and `artifact` recipes are reported infeasible and
  fall back to `k=1`; the six feasible feeders are bronze, amethyst, orb,
  sparkles, ribbon and pendant (`src/neat/time-aligner.js:69-115`,
  `src/tests/time-aligner.test.js:13-49`).
- **Relaxation fallback.** `planBatchCaps` takes quantums/alignments as explicit
  inputs, repairs the exact critical path and, if strict alignment overruns the
  horizon, relaxes the worst aligned contributors to `alignCap=0` until the path
  fits, reporting each drop as `ALIGNMENT_RELAXED` (`src/neat/batch-planner.js:177-255`).
  `effectiveCap` is the single cap authority shared by planner and engine:
  `alignCap > 0 -> min(plannerCap, alignCap)`; `alignCap = 0 ->` snap `plannerCap`
  down to a positive step multiple; result clamped to `[1, MAX_BATCH]`
  (`src/neat/batch-quantizer.js:77-90`).
- **Measured reuse result.** With quantization and alignment enabled, the shipped
  `reuse` preset run is feasible at completion tick **2759** with
  `totalSpent = 4,156,250`.

### Training coverage and speed

Training stays item-major (one population per craftable in dependency order,
preserving per-item RNG lineage). Each item gets a reserved time slice
(`reserveMs = (maxWallMs / 19) / generations`), and generation 0 is never
skipped by the shared wall-clock deadline, so all 19 networks reach generation
0 under the default budget. If a slice expires, the best genome found so far is
kept (or a null genome falls back to greedy). Before a best genome is frozen for
downstream items it is re-evaluated against the final upstream frozen policies;
early stopping happens only after that re-evaluation is feasible
(`src/neat/controller.js:343`, `src/neat/controller.js:380`).

Curriculum generations 0-1 truncate the **rollout** to 2,000 ticks for speed,
but the policy **observation** always sees the full 8,064-tick horizon
(`src/neat/controller.js:46-49`, `src/neat/controller.js:140`). Selection uses a
separate bounded shaped signal, `trainingFitness`
(`src/neat/fitness.js:114`): manufactured progress, target progress and time
progress are each clamped to `[0,1]` and finite/clamp-checked, and the score is
strictly below the smallest feasible score. Telemetry `GENOME_RESULT.fitness`
and the final reported score remain the shipped lexicographic
`-(totalSpent * 100000 + completionTick)` (coins first, time second).

### Lazy stash view and diagnostics

`src/sim/engine.js:155` replaces the per-tick stash object passed to policies
with one memoised, read-only Proxy. It computes each item's scalar stash value
on demand from current engine state, exposes the same id keys to property
reads, `in`, `Object.keys` and object spread, and rejects every mutation. This
keeps rollouts faster without changing values or the policy contract.

Criticality and cap diagnostics are computed on the **main thread only**
(`src/main.js:244`) and rendered, when a `#solver-diagnostics` host exists, by
`src/ui/diagnostics.js` using `createElement`/`textContent` and a 120-character
cap. They are never added to a worker message: the START message is frozen to
`type`/`config`/`seed`/`options`, and `main.js` binds only `worker.onmessage`
(no `window.addEventListener('message')`). The worker message contract is
unchanged.

## NEAT engine provenance

The vendored `@stsoftware/neat-ai@7.1.22` bundle is retained in
`src/vendor/` (with `src/vendor/manifest.json`) as provenance only. It hangs during
top-level initialization over `http://`, so it is **not imported** at runtime.
The active engine is the in-house `src/neat/inhouse-neat.js`, which passed the
Phase 0 spike 5/5 (`docs/neat-spike.md`).

## Best schedule and saved initial state (data/)

- `data/best-schedule.csv` — the app's best schedule: the pivot of `result().schedule` of the trained
  display run (`src/ui/results.js:42-47`). Rows = ticks 0..3348 (3,349 rows); cells = batch produced
  per craftable per tick (0 when idle). Reuse-preset run (feasible; see `docs/best-schedule.md`):
  completion tick 3348, `totalSpent: 2,823,250`, 5,883 events.
- `data/initial-state.json` — the saved, locally editable initial state (`{targets, initialCoins,
  initialStash}`); app-valid (`validateInitialState`, `src/validate.js:201`). The app's **Reuse**
  preset loads it over `http://` (`loadConfig('reuse')`, `src/config.js:164-216`: bounded fetch,
  explicit presence + `isPlainObject` checks, null-proto maps, validation; missing/invalid file or
  `file://` → hardcoded preset) — so does the generator. It is currently a custom snapshot
  (`initialCoins: 6383201`; stash values in the file); the CSV above was generated earlier from the
  `reuse` preset snapshot (`initialCoins: 5998996`) — re-run the generator to regenerate the schedule
  from the currently saved state.
- `tools/best-schedule.mjs` — generator: `node tools/best-schedule.mjs [reuse|empty]` (from
  `random_codes/0004_cat_game_neat`). Reads `data/initial-state.json` when present — the simulation
  runs on the saved state — and writes `data/best-schedule.csv` via the app-faithful pipeline
  (`trainAll` → `createDisplayPolicyBundle` → `planBatchCaps`/`applyBatchCaps` → `createEngine`).

`data/` is not committed (user request — "it's for the current use to see only").

## Tests

Open `tests.html` over `http://`. It runs 280 in-browser tests across 15 test
files through `src/tests/harness.js`. Results render in `#test-results` (PASS/FAIL rows plus a
`TOTAL passed=... failed=...` line) and are also published as
`window.__TEST_RESULTS__ = { passed, failed, failures }`. Coverage includes
cost/demand/engine/fitness/inputs/defaults logic, config/YAML-parity and 256 KiB
cap checks, validator and XSS checks, termination and wall-deadline behavior, UI
rendering and tooltips, and a real module-worker end-to-end test
(`src/tests/worker-bridge.test.js:406`).

The suite also covers the new cap: exact `manufactureRequired` arithmetic and
all three starter modes (`src/tests/demand.test.js`); the
`produced + committed <= manufactureRequired` invariant including an in-flight
batch, the exact `REQUIRED_MET` rejection with no starvation flags and no
`STARVED_*` event, target refill, and baseline non-regression (byte-identical
`result()` when the cap is non-binding; greedy presets no worse than baseline;
`reuse` reaching `feasible: true` with an adequate policy)
(`src/tests/engine.test.js`); and badge visibility/computed styles, base-node
`n/a`, `COMPLETE` precedence, and forbidden-key/payload-string safety
(`src/tests/ui.test.js`).

It also covers the solver improvements: the batch-planner contract (exact
clamped-cap critical path, superlinear cost, critical-chain allocation and a
real-engine `reuse` run that is feasible within 8,064 ticks below the 7,111,975
spend baseline), null-prototype criticality maps and forbidden-key rejection,
bounded `trainingFitness`, the reserved gen-0 guarantee for all 19 networks with
an injectable clock, curriculum observation byte-identity, the lazy read-only
stash Proxy (eager-equivalent values and rejected mutation), main-thread
diagnostics rendered with `textContent` only and XSS-safe, Logger/Results tab
switching, and a source scan that keeps solver fields out of worker messages
(`src/tests/batch-planner.test.js`, `src/tests/priority.test.js`,
`src/tests/fitness.test.js`, `src/tests/controller.test.js`,
`src/tests/engine.test.js`, `src/tests/worker-bridge.test.js`). The launcher
scripts are shell/batch programs and are not executed by the in-browser harness;
their loopback-only bind, validated local port and fixed loopback URL are
enforced at runtime by the scripts themselves.

It further covers batch quantization and alignment: the exact 19-item quantum
table, `effectiveCap` snapping and single-authority idempotence, `quantizeBatch`
snapping/exact-remainder/`BELOW_QUANTUM` behaviour and aligned split sizes, the
worked alignment table, infeasible `waterstone`/`firestone`/`artifact` recipes
reported with `ALIGNMENT_INFEASIBLE`, forced-alignment `ALIGNMENT_RELAXED`
fallback, and prototype-safe null-prototype maps. It covers the editable initial
state too: forced override maps committed by reference, null-prototype/XSS-safe
reads that create no element or handler, refresh defaults to the `empty` preset,
Reset-to-0 form-only behaviour, Apply/Reset wiring, and validation bounds
(`src/tests/batch-quantizer.test.js`, `src/tests/time-aligner.test.js`,
`src/tests/config.test.js`, `src/tests/security.test.js`, `src/tests/ui.test.js`,
`src/tests/worker-bridge.test.js`). The sim-time readout is checked with
injected tick snapshots: it renders `tick × 5` minutes with the `Sim time:` label,
follows display snapshots, freezes on completion and resets on Start/Apply — no
real timers (`src/tests/ui.test.js`).

## Dev-only artifacts

`spike.html` and `src/ui/preview.html` with `src/ui/mock*.js` are development-only
layout/spike tools. They are not loaded by the production `index.html`, which
loads only `src/main.js`.

## Known limitations

- Networks are feed-forward only (`docs/neat-spike.md` §4).
- `http://` is required for ES modules and training; `file://` uses the embedded
  config with training disabled, where the greedy preview is the only result.
- The greedy idle preview before training is labelled and does not write to
  Results, except under `file://` where training is unavailable.
- Dev-only artifacts are present (see above).
- A zero wall-clock budget starts no population, so every item falls back to the
  greedy policy in the display run; at any positive budget generation 0 is
  reserved for all 19 items (`src/neat/controller.js:343-346`,
  `src/neat/controller.js:380-388`).

## Layout

- `index.html` — production entry point; `tests.html` — test runner;
  `styles.css` — styles.
- `start.sh`, `start.bat` — loopback-only static-server launchers (see "How to
  run").
- `config/` — crafting tree and target YAML.
- `src/sim/` — engine, state, items, cost, demand.
- `src/neat/` — controller, adapter, in-house NEAT, fitness, inputs, defaults,
  batch-planner, priority, policy-caps, batch-quantizer, time-aligner.
- `src/worker/` — training Web Worker.
- `src/ui/` — DAG, HUD, logger, controls, results, formatting, item registry,
  diagnostics, initial-state (plus dev-only preview/mock files).
- `src/validate.js`, `src/config.js` — validation and config loading.
- `src/tests/` — harness and 15 test modules.
- `src/vendor/` — vendored NEAT bundle and manifest (provenance only).
- `docs/architecture.md` — module boundaries and contracts;
  `docs/neat-spike.md` — Phase 0 capability-spike record.
