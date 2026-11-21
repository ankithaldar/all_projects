# Architecture (implemented)

This document describes the code as it exists. It complements `README.md` with
module boundaries, message types, and data contracts. No build step, backend, or
Python code is involved; everything runs in the browser.

## Module boundaries

| Area | Files | Responsibility |
|---|---|---|
| Config | `src/config.js`, `src/validate.js`, `config/*.yaml` | Fetch (256 KiB cap), js-yaml 4.1.1 parse (`CORE_SCHEMA`, `json: true`), allowlist validation, embedded fallback. |
| Simulation | `src/sim/{engine,state,items,cost,demand}.js` | DOM-free deterministic tick engine and canonical item/DAG metadata. |
| Evolution | `src/neat/{controller,adapter,inhouse-neat,fitness,inputs,defaults,batch-planner,priority,policy-caps,batch-quantizer,time-aligner}.js` | 19 NEAT populations, rollout orchestration, genome engine, observation/decoding, offline batch-cap planning and capping, batch quantums and per-product lead-time alignment. |
| Worker | `src/worker/train-worker.js` | Silent bounded rollouts; emits `PROGRESS`/`BEST`/`GENOME_RESULT`/`DONE`/`ERROR`. |
| UI | `src/ui/*.js` | DAG, HUD, logger, controls, results, formatting, item registry, main-thread diagnostics, editable initial-state panel. |
| Bootstrap | `src/main.js` | Wires UI, validates worker messages, runs displays, computes main-thread diagnostics, owns the sim-time display and config/preset/Apply/Reset propagation. |
| Launcher | `start.sh`, `start.bat` | Loopback-only static-server launchers with a locally generated, validated port. |

`src/config.js` is the single owner of `MAX_BATCH = 20` and the plan-locked
constants: `MAX_TICKS = 8064`, `TICK_MINUTES = 5`, `COINS_PER_TICK = 210`,
`BIG = 100000`, `LOG_CAP = 5000` (`src/config.js:12-23`). `src/validate.js` is the
single owner of config/control/worker-message validators and the text sanitizer
used at every DOM boundary.

## Config contract

`loadConfig(presetId)` returns
`{ items, targets, initialCoins, initialStash, starterMode }` (`src/config.js:154`).
It tries the two YAML files through `fetchCapped` (declared `content-length` and
actual byte length both capped at 256 KiB) and `parseYaml`, then calls
`validateConfig`; any failure falls back to `EMBEDDED_RAW`. Preset lookup is
strict (`reuse` or `empty`) to avoid prototype-chain resolution. `normalizeItems`
drops `require_crafted` and maps YAML keys to the canonical shape
`{ id, name, tier, type, ingredients, coinCost, craftTimeMinutes }`.

The default argument is `presetId = 'empty'`, so a fresh load/refresh returns 0
coins, an empty stash and the YAML targets with no persisted edits and no storage
reads (`src/config.js:151-158`). Two further helpers support editable initial
state: `prefillInitialState(config)` returns fresh null-prototype maps keyed by the
11 target ids and 19 craftable ids with missing values defaulting to 0
(`src/config.js:206-229`), and `applyOverrides(config, overrides)` shallow-copies
the config and assigns the validated `targets`/`initialCoins`/`initialStash` **by
reference** so their null prototype survives into the START message and engine.
It validates the overrides before and the exact references after assignment, and
never mutates the loaded config (`src/config.js:184-204`).

## Simulation contract

`createEngine({ config, policyForItem, quiet })` returns `{ step, snapshot,
result }` plus getters (`tick`, `coins`, `totalSpent`, `done`, `stash`, `slots`,
`starvation`, `status`, `required`, plus the cap maps `manufactureRequired`,
`produced`, `committed`) (`src/sim/engine.js:34`, `src/sim/engine.js:346-362`).
Per tick (`src/sim/engine.js:172`):

1. Add 210 coins.
2. For each free craftable slot in topological order, ask the policy and clip the
   batch to `min(materialsLimit, coinAffordableLimit, MAX_BATCH, remaining)`,
   where `remaining = max(0, manufactureRequired - produced - committed)`. That
   clipped size is then passed through `quantizeBatch(requested, step, remaining,
   batchCap, aligned)` (post-clip quantization) before any spend; see "Batch
   quantizer and time aligner". A positive post-quantization batch spends coins,
   consumes non-base ingredients, starts the slot and increments `committed`; a
   zero batch sets material/coin starvation flags. A sub-quantum rejection with
   quantum headroom still available emits the frozen `DECISION_REJECTED` reason
   literal `BELOW_QUANTUM`, latches the `blocked` telemetry flag, emits no
   `STARVED_*` event and leaves state untouched (`src/sim/engine.js:324-338`).
   When `remaining === 0` the cap is evaluated first: the item is skipped with a
   `DECISION_REJECTED` whose reason is the frozen literal `REQUIRED_MET`, both
   starvation flags are cleared, and no `STARVED_*` event is emitted. Completed
   batches move `committed` back into `produced`, keeping
   `produced + committed <= manufactureRequired` true at every tick.
3. Advance timers and count busy ticks against the item.
4. Completed slots add the batch to the stash.
5. Re-evaluate targets against the current stash; stop when all are met or at
   tick 8063.

`quiet: true` skips per-tick event allocation during fitness rollouts without
changing state or `result()` (`src/sim/engine.js:103`). `result()` returns
`{ feasible, completionTick, elapsedTicks, totalSpent, produced, schedule,
shortfall, perItem }`; `makeSnapshot` builds the structured-clone-safe DTO
consumed by the UI (`src/sim/state.js`). It now also exposes three craftable-only
maps, `manufactureRequired`, `produced` and `committed` (base ids absent; keys
drawn from validated `craftIds`, never from `Object.keys` of the maps);
`required` stays as the unchanged exact-demand map for tooltips.

`policyContext(id)` (`src/sim/engine.js:213`) passes policies a **lazy, read-only
stash view**: a single memoised throwing Proxy (`src/sim/engine.js:155`) over a
null-prototype target. Each item's scalar value is computed on demand from the
current engine state; property reads, `in`, `Object.keys` and object spread
expose exactly the same id keys and scalars as the old eager object, while
`set`, `defineProperty`, `deleteProperty`, `setPrototypeOf` and
`preventExtensions` all throw. No `eval`/`Function` is used. This removes the
per-tick, per-item object rebuild so rollouts are faster without changing values
or the policy contract. The `required` and `remaining` keys on `policyContext`
are retained for legacy UI/tooltips but are deliberately NOT read by the
controller's `buildContext()`, which uses the `manufactureRequired` basis
(`src/sim/engine.js:164`, `src/neat/controller.js:100`).

Cost and demand contracts: `totalCost(init, b) = init*b*(1+0.25*(b-1))`
(`src/sim/cost.js:27`); `expandDemand` computes `required[name]` in reverse
topological order, crediting starter stash except in `FREE_INGREDIENTS`
(`src/sim/demand.js:51`).

`expandManufactureRequired(items, targets, starterStash, starterMode)` returns a
separate `manufactureRequired` map (craftable ids only) used solely as the engine
hard-cap basis; `expandDemand().required` is untouched. It walks the reverse
topological order and folds `ingredientNeed` (demand pushed down from downstream
products) into a per-item cap using mode-correct starter rules:

- `PREDELIVERED_RESERVED` — starter counts toward a target but is not consumable:
  a target gets `max(0, targetNeed - st) + ingredientNeed`, a non-target gets
  `ingredientNeed`.
- `CONSUMABLE` — starter counts toward the target and is consumable: a target
  uses `min(st, targetNeed)` toward the target and offsets ingredient need with
  its leftover, while a non-target gets `max(0, ingredientNeed - st)`.
- `FREE_INGREDIENTS` — starter does not count toward the target: every item gets
  `targetNeed + max(0, ingredientNeed - st)`.

Each cap is then propagated to craftable ingredients as `cap * quantity`; bases
are unlimited and are never propagated or capped. Because the cap is gross
manufacture (target plus downstream consumption), a target consumed as an
ingredient is manufactured again; status is still evaluated against the current
stash, and a deliberate cap therefore prevents re-manufacture beyond that gross
total. `REQUIRED_MET` is the exact, frozen reason literal emitted when the cap
alone forces a rejection.

## DAG depth vs YAML tier

`src/sim/items.js` builds `DAG_EDGES` from recipes and computes `DEPTH_BY_ID` as
the longest path from a base (bases = 0) via topological relaxation. The raw YAML
`tier` is kept only for metadata/display; `effectiveTier(id)` exposes the computed
depth used for left-to-right graph columns (0..10).

## NEAT contract

`src/neat/inhouse-neat.js` implements canonical NEAT over plain, structured-clone-safe
genomes (`nodes`, `genes`, `frozen`, `fitness`, `adjustedFitness`, `speciesId`), a
seeded mulberry32 PRNG, speciation with fitness sharing, innovation-aligned
crossover, and mutation. Topologies are feed-forward: a connection is legal only
when `depth(from) < depth(to)`. The vendored bundle under `src/vendor/` is not
imported at runtime.

`src/neat/adapter.js` is the stable seam exposing `evaluatePopulation`,
`getBestGenome`, `exportGenome`, `importGenome`, `cloneFrozen`, `activate`, and
`reset`. `src/neat/controller.js::trainAll` evolves the 19 craftables in
`CRAFTABLE_IDS` topological order, each with a seed derived from the master seed.
For item `k`, earlier items use `cloneFrozen` genomes, item `k` uses the candidate,
and later items use `greedyBatch` from `src/neat/defaults.js`. Each candidate is
scored by a quiet engine rollout; fitness is `-(totalSpent*100000+completionTick)`
when feasible, else `-1e12 + progressBonus` (`src/neat/fitness.js`). The policy
observation is a fixed 5-element vector
`[remainingDemand, coins, ticksRemaining, materialAvailability, slotActive]`
(`src/neat/inputs.js:16`); output is decoded to a batch in 1..20.

## Batch planner and policy caps

`src/neat/batch-planner.js` implements the frozen planner contract:

`planBatchCaps(items, mfr, maxTicks, maxBatch = MAX_BATCH, safety = 0.8)`
returns `{ caps, estimatedCost, estimatedTicks }`. `caps` is a null-prototype
map keyed only by validated craftable ids, each clamped to `[1, maxBatch]`.
`estimatedTicks` is the **exact critical path** — the maximum over dependency
chains of `Σ ceil(mfr[id] / cap[id]) × craftTicks(id)` — computed with the clamped
caps and required to be `<= maxTicks`; the safety factor (default `0.8`) targets
`maxTicks * safety` so a bounded repair loop leaves headroom. `estimatedCost` is
the sum of the real superlinear `totalCost(init, b)` per item
(`src/sim/cost.js`), with no item-count shortcut.

`src/neat/priority.js::computeCriticality(items, targets)` returns a
null-prototype `{ [validated craftable id]: number in [0,1] }`, computed as
longest-path depth for items that feed a target (`src/neat/priority.js:77`).
Higher scores (the artifact/elementstone chain) get larger caps in the planner;
lower scores get smaller caps. It is **offline only**: never put on a worker
message, never written to telemetry, and never used to throttle coins at
runtime — all 19 items feed target paths and none may be starved
(`src/neat/priority.js:1`).

`src/neat/policy-caps.js::applyBatchCaps(policies, caps)` is the **single shared
implementation** used by both `src/neat/controller.js` (training) and
`src/main.js` (display). It returns a null-prototype map of validated-craftable
policies whose batches are clamped to each item's cap
(`src/neat/policy-caps.js:56`). Capping is at the policy layer only: the engine's
material/coin/`MAX_BATCH` clipping is untouched, and the helper is idempotent.

## Batch quantizer and time aligner

`src/neat/batch-quantizer.js` is a frozen, pure, DOM-free module returning only
null-prototype maps keyed by validated craftable ids (so inherited keys can never
be mistaken for items). Contracts (`src/neat/batch-quantizer.js:33-133`):

- `computeBatchQuantums(items) -> { [id]: step }` — each craftable's step is the
  GCD of every per-unit quantity with which downstream recipes consume it; a
  terminal item gets `1`.
- `effectiveCap(plannerCap, step, alignCap)` — the **single cap authority** used
  by both the planner and the engine. `alignCap > 0` selects
  `min(plannerCap, alignCap)` (quantum rule relaxed); `alignCap = 0` snaps
  `plannerCap` down to a positive multiple of `step`; the result is clamped to
  `[1, MAX_BATCH]` and is idempotent.
- `quantizeBatch(requested, step, remainingCap, batchCap, aligned)` — returns the
  exact batch size or `0` (reject). With `n = min(requested, remainingCap,
  batchCap)`: `n >= step` snaps down to a step multiple; a final remainder below a
  full quantum is preserved exactly; aligned items use `n` directly as the
  per-product split; `0 < n < step` with headroom returns `0` and never snaps up.

`src/neat/time-aligner.js::computeAlignment(items, mfr, quantums)` returns
`{ alignments, infeasibleRecipes }` (both null-prototype). Alignment is computed
per product unit, never from total `mfr`: for a product's craftable ingredients
it takes `totalTicks = lcm(craftTicks...)` and, per ingredient,
`k = totalTicks / craftTicks`, `alignCap = ceil(q / k)`. A candidate is feasible
only when `k <= q`, `k * craftTicks <= MAX_TICKS` and `alignCap <= MAX_BATCH`; if
any ingredient fails, the whole recipe is `ALIGNMENT_INFEASIBLE` and falls back to
`k=1` (`alignCap=0`). An item feeding several recipes keeps the minimum feasible
`alignCap`. On the shipped config the six active feeders are bronze (k=2,
alignCap=2), amethyst (2,4), orb (2,1), sparkles (2,1), ribbon (2,1) and pendant
(2,2), while `waterstone`, `firestone` and `artifact` are reported infeasible.
`ALIGNMENT_STATUS` freezes `ALIGNMENT_FEASIBLE`, `ALIGNMENT_INFEASIBLE` and
`ALIGNMENT_RELAXED` (`src/neat/time-aligner.js:28-115`,
`src/tests/time-aligner.test.js:13-49`).

The engine derives quantums, alignments and the offline plan itself from its
validated `config.items` and `manufactureRequired` basis; it owns no cap config
field. The planner returns `relaxed` reports for alignments dropped to restore the
hard horizon: `planBatchCaps` accepts explicit `quantums`/`alignments`, repairs
the exact critical path, and on overrun relaxes the worst aligned contributors to
`alignCap=0` until the path fits, marking each as `ALIGNMENT_RELAXED`
(`src/neat/batch-planner.js:58-262`). Effective caps come only from `effectiveCap`,
so planner and engine agree.

## Initial-state schema and override propagation

`src/validate.js::validateInitialState(overrides)` is the mandatory full schema
for user-supplied initial state (`src/validate.js:194-215`). It requires three
fields: null-prototype `targets` (the 11 ids in `KNOWN_TARGET_IDS`), a bounded
`initialCoins`, and null-prototype `initialStash` (the 19 ids in
`KNOWN_CRAFTABLE_IDS`). Each value is a bounded integer with targets `0..1e7` and
coins/stash `0..1e9`; forbidden prototype keys and unknown ids are rejected and
own property names are checked so `__proto__` cannot hide from the allowlist.
`src/ui/initial-state.js` mirrors those bounds for form coercion, falls back to
the prefill value on bad input and calls back with the stable
`CONFIG_INPUT_INVALID` code only (never raw input). Its DOM writes are
`createElement`/`textContent`/`setAttribute` only, with `capString(...,120)` on
every emitted string; the produced null-prototype maps are assigned by reference
downstream (`src/ui/initial-state.js:1-150`).

`src/main.js::applyOverridesAndStart(base, overrides, hooks)` is the single
propagation seam: it re-validates the overrides, stores the pristine base in
`baseConfig`, commits the by-reference result to the module-level `config`, and
passes that exact reference to the START message, engine and diagnostics
(`src/main.js:322-341`). `startAll()` reads the panel only after the config
resolves, so prefill fallbacks use the active preset; **Apply** delegates to the
same `startAll()` path. `resetAll()` restores `config = baseConfig` and re-prefills
the panel from it; `onPresetChange()` loads the new preset, refreshes `baseConfig`
and re-prefills the panel (`src/main.js:560-685`, `src/main.js:738-778`). The
**Reset to 0** button is form-only: it zeroes coins and the 19 stash inputs,
leaves the 11 targets untouched and calls neither Apply nor Reset
(`src/ui/initial-state.js:141-146`). No path writes browser storage or the YAML
files. Because the default preset is `empty`, a fresh load/refresh shows 0 coins,
an empty stash and the YAML targets.

## Sim time

The top-of-page readout (`#wall-clock`, `role="timer"`, `aria-live="off"`) shows
simulated game time, not real elapsed time. It is derived from engine ticks:
`N min = tick × TICK_MINUTES` with `TICK_MINUTES = 5`, rendered as
`Sim time: N min`. It is refreshed from each display snapshot, frozen on the final
snapshot when the run completes, and reset on Start/Apply. No `Date.now()` elapsed
minutes or periodic real-time timer are involved. The host element is written only
with `textContent` after `capString(...,120)`. Tests drive it with injected tick
snapshots rather than real timers.

## Training orchestration and curriculum

Training remains item-major (`CRAFTABLE_IDS` topological order, per-item RNG
lineage). Each item gets a reserved wall slice
`reserveMs = (maxWallMs / 19) / generations`; generation 0 is never gated by the
shared deadline, so all 19 networks reach generation 0 at the default budget. If
an item's slice expires, the best genome found so far is kept (a null best falls
back to greedy); later generations use the remaining budget
(`src/neat/controller.js:343-346`, `src/neat/controller.js:380-388`).

Before a best genome is frozen for downstream items it is re-evaluated once in
the **final** upstream frozen-policy context (`finalCheck`,
`src/neat/controller.js:437-444`); early stopping happens only after that
re-evaluation is feasible (`src/neat/controller.js:428-434`). No runtime coin
throttling is applied.

Curriculum: generations 0-1 (`CURRICULUM_GENS = 2`) truncate only the **rollout**
to `maxTicks = 2000` (`CURRICULUM_MAX_TICKS`) for speed. The policy observation
always uses the full 8,064-tick horizon through `buildContext()`
(`src/neat/controller.js:46-49`, `src/neat/controller.js:140-144`).

Fitness has two signals (`src/neat/fitness.js`):

- `fitness(result)` is the **shipped** score,
  `-(totalSpent * BIG + completionTick)` when feasible and `-1e12 +
  progressBonus` otherwise. It stays the value in telemetry
  `GENOME_RESULT.fitness` and the final reported score.
- `trainingFitness(result, phase, mfr, maxTicks)` is a bounded selection signal
  used only while a rollout is infeasible: `INFEASIBLE_BASE +
  1000*manufacturedProgress + 100*targetProgress + 10*timeProgress`. Each
  progress term is clamped to `[0,1]` with guarded divisions (`qty > 0`), the
  sum is `Number.isFinite`-checked and clamped to `[-1e15, 1e15]`; its maximum
  (`INFEASIBLE_BASE + 1110`) is strictly below the smallest feasible score.
  Once feasible it returns the shipped score immediately
  (`src/neat/fitness.js:114`). `enrichResult()` supplies targets, `mfr` and
  `maxTicks` to both signals without mutating the source result.

## Worker message contract

START (main → worker): `{ type: 'START', config, seed,
options: { populationSize, generations } }` (`src/main.js:205`).

Worker → main message types (`src/worker/train-worker.js`, validated by
`validateWorkerMessage` in `src/validate.js`):

| Type | Payload (notable fields) |
|---|---|
| `PROGRESS` | `itemId`, `generation`, `completed`, `total`, `phase`, `message` (budget pings also carry `elapsedMs`, `budgetMs`, `remainingMs`). |
| `BEST` | `itemId`, `generation`, `fitness`, `feasible`, `genome` (validated). |
| `GENOME_RESULT` | `itemId`, `generation`, `genomeIndex`, `fitness`, `feasible`, `completionTick`, `elapsedTicks`, `totalSpent`, `perItem` (per-item coin/time/batch/unit rows). |
| `DONE` | `feasible`, `completionTick`, `totalSpent`, `genomeCount`, `telemetryCount`, `genomes` (bounded map, max 19). |
| `ERROR` | `code`, `message` (no stack traces). |

Bounded messaging: a hard cap of 20 messages per `(itemId, generation)` per
`PROGRESS`/`GENOME_RESULT` kind, a global cap of 50,000 posted messages, a 5,000
entry telemetry ring, and exactly one terminal `DONE` (emitted by normal finish or
by the armed wall-clock deadline). `main.js` binds only `worker.onmessage`; there
is no `window.addEventListener('message')` anywhere.

The manufacturing cap is engine-local: snapshots are never posted to the worker,
so `WORKER_TYPES`, the five outbound message shapes and `src/validate.js` remain
unchanged. The policy observation context is built from the `manufactureRequired`
basis (`src/neat/controller.js:104`, `src/neat/controller.js:332-333`); the
engine's `policyContext.required` / `remaining` keys stay only for legacy
UI/tooltips and are not read by the controller.

Criticality and batch-cap diagnostics are computed on the **main thread only**
(`buildSolverDiagnostics`, `src/main.js:244`) and are never added to a worker
message: `buildStartMessage()` emits only `type`/`config`/`seed`/`options`
(`src/main.js:228`), `main.js` binds only `worker.onmessage`, and there is no
`window.addEventListener('message')` anywhere. The worker message contract is
frozen.

## Display runs

`main.js` runs a labelled greedy preview immediately. On `BEST`/`DONE`, trained
genomes are collected and `createDisplayPolicyBundle` builds per-item policies
(genome policy when present, greedy otherwise) for a **second** display run; that
run is authoritative and the only one that writes to Results. Under `file://` no
worker is started and the greedy preview writes the result instead
(`src/main.js:284-359`).

The trained display run also replays the same capped policies as training:
`planBatchCaps` produces caps on the main thread and `applyBatchCaps` wraps the
display policies through the single shared helper (`src/main.js:388-429`).

## UI tab wiring

`src/main.js` owns tab switching: `wireTabs()` binds click handlers for
`tab-train`, `tab-logger` and `tab-results`, and `applyTabSelection()` toggles
the `is-active` class, keeps `aria-selected` in sync and removes `hidden` from
exactly one `#panel-<name>` (`src/main.js:504-551`). Logger and Results now
switch correctly. The top-left stash panel, live coin counter and DAG
`produced/manufactureRequired` badge with `COMPLETE` styling are unchanged.

## Launcher scripts

`start.sh` (POSIX) and `start.bat` (Windows) start a static server from the
project directory itself (`cd "$(dirname "$0")"` / `cd /d "%~dp0"`), bind
explicitly to `127.0.0.1`, derive a validated integer port locally (Python
socket bind-to-port-0), and interpolate it only into the fixed
`http://127.0.0.1:<port>/index.html` literal (`start.sh:1-154`,
`start.bat:1-90`). Startup checks confirm the server answers on loopback and is
NOT reachable on a local non-loopback address. Both scripts open the fixed URL
in the default browser and keep the server running until Ctrl-C. Direct
`file://` opening is unsupported by browsers, so the launcher is the supported
way to play; it changes no app code or worker contract.

## Verification

`tests.html` imports 15 test modules and `src/tests/harness.js`, then calls
`runTests()`. The harness writes PASS/FAIL rows and a total into `#test-results`
and publishes `window.__TEST_RESULTS__`. The current suite is 280/280 green. All
dynamic DOM text is written with `textContent` after `capString`/`sanitizeText`;
no dynamic HTML is used.

The suite additionally covers the batch-planner contract (exact clamped-cap
critical path, superlinear cost and a real-engine `reuse` run that is feasible
within 8,064 ticks below the 7,111,975 spend baseline), null-prototype
criticality maps with forbidden-key rejection, bounded `trainingFitness`, the
reserved gen-0 guarantee for all 19 networks, curriculum observation
byte-identity, the lazy read-only stash Proxy (eager-equivalent values and
rejected mutation), main-thread diagnostics that are `textContent`-only and
XSS-safe, Logger/Results tab switching, and a source scan that keeps solver
fields out of worker messages. The launcher scripts are shell/batch programs and
are not executed by the in-browser harness; their loopback-only bind, validated
local port and fixed loopback URL are enforced at runtime by the scripts
themselves.

The suite also covers the quantizer/aligner contracts (exact per-item quantum
table, `effectiveCap` snapping and single-authority idempotence,
`quantizeBatch` snapping/exact-remainder/`BELOW_QUANTUM` behaviour, the worked
alignment table, infeasible `waterstone`/`firestone`/`artifact` recipes with
`ALIGNMENT_INFEASIBLE`, forced-alignment `ALIGNMENT_RELAXED` fallback and
prototype-safe maps), the editable initial state (forced overrides committed by
reference, null-prototype/XSS-safe reads that create no element or handler,
refresh defaults to the `empty` preset, form-only Reset-to-0, Apply/Reset wiring
and the validation bounds), and the sim-time readout driven by injected tick
snapshots rather than real timers (`src/tests/batch-quantizer.test.js`,
`src/tests/time-aligner.test.js`, `src/tests/config.test.js`,
`src/tests/security.test.js`, `src/tests/ui.test.js`,
`src/tests/worker-bridge.test.js`).
