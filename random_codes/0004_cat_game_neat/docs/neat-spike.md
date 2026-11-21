# Phase 0 — NEAT capability spike (BLOCKING gate)

**Verdict: PASS via the pre-approved in-house fallback.** The vendored
`@stsoftware/neat-ai@7.1.22` esm.sh bundle was downloaded and hash-pinned, but
**fails capability (e)**: its top-level initialization never settles in a real
browser over `http://`. All five required capabilities PASS on the self-contained
canonical NEAT module at `src/neat/inhouse-neat.js`. **App code must import the
in-house module, not the vendored bundle.** The vendored file is retained only
for provenance/reproducibility of this decision.

---

## 1. Vendored library attempt (FAILED)

### 1.1 What was vendored

| Field | Value |
|---|---|
| Package | `@stsoftware/neat-ai` (JSR: `@stsoftware/neat-ai@7.1.22`) |
| URL | `https://esm.sh/@jsr/stsoftware__neat-ai@7.1.22/es2022/stsoftware__neat-ai.bundle.mjs` |
| File | `src/vendor/neat-ai/neat-ai.bundle.mjs` |
| Bytes | 1,178,777 |
| SHA-256 | `86337d1c26d30870c8b2685d5aace197ea2efbd1abe8273c34d2932297dd198d` |
| Retrieved | 2026-11-18T17:25:48Z |
| Manifest | `src/vendor/manifest.json` |

The alternate candidate `https://esm.sh/jsr/@stsoftware/neat-ai@7.1.22?bundle`
was rejected: it is a 135-byte re-export stub containing an external import,
not a self-contained bundle.

The chosen bundle is a genuine ES module (large `export{...}` list including
`Creature`, `EpisodeAdapter`, `createSeededRng`, `setRandomNumberGenerator`,
`exportCheckpoint`/`importCheckpoint`) with **no external import statements**.
File integrity was re-verified after vendoring:

```
$ sha256sum src/vendor/neat-ai/neat-ai.bundle.mjs
86337d1c26d30870c8b2685d5aace197ea2efbd1abe8273c34d2932297dd198d  src/vendor/neat-ai/neat-ai.bundle.mjs
# equals manifest.json sha256 -> HASH_MATCH=PASS
```

### 1.2 Failure evidence (real headless Chrome, real time)

Probe page dynamically imported the bundle inside a `Promise.race` with a
25-second wall-clock timeout; the page load event was held open by a deliberately
slow local subresource so `--dump-dom` captured the result in real time (no
virtual clock, which can fast-forward timers while real work stays pending).

```
$ google-chrome --headless --disable-gpu --no-sandbox --disable-dev-shm-usage \
    --dump-dom http://127.0.0.1:8137/spike.html
EXIT=0 REAL_SECONDS=31
<pre id="results">SCRIPT_STARTED
IMPORT=ERR Error: TIMEOUT_IMPORT_25S_REAL
PROBE_DONE=true</pre>
```

Shorter 40s-virtual-time and 6s-real-time runs produced the same result.
No `window.onerror` and no `unhandledrejection` fired: the dynamic-import
promise simply never settles. Node 26 produced the same symptom
(`Detected unsettled top-level await`). Root cause is browser-incompatible
top-level initialization: the bundle awaits a deep Deno/WASM init chain and
constructs workers from relative URLs that do not exist in the vendored artifact
(e.g. `new Worker(new URL("./EventLoopWatchdogWorker.ts", import.meta.url))`).
Result: **capability (e) FAILS, and (a)–(d) cannot be exercised at all.**

Consequence for the plan: Decision 8's "in-house canonical NEAT allowed
automatically if Phase 0 spike fails" branch is now active.

---

## 2. In-house fallback capabilities (PASS 5/5)

Command (local static server on `127.0.0.1:8137`, root = the project dir):

```
$ google-chrome --headless --disable-gpu --no-sandbox --disable-dev-shm-usage \
    --dump-dom http://127.0.0.1:8137/spike.html
EXIT=0 REAL_SECONDS=9
<p id="verdict">VERDICT=PASS (5/5 capabilities)</p>
<pre id="results">MODULE_IMPORT=OK exports=DEFAULT_CONFIG,createInhouseNeat,createSeededRng
PASS a_blackbox_fitness :: fitnessFn called; gen 0-&gt;1; bestFitness=935.000; constant-fitness control bestFitness=42.000
PASS b_two_populations_frozen :: frozen byte-identical after evolution=true; populations independent=true; genes e1=2 e2=2
PASS c_seeded_determinism :: same seed identical=true; different seed differs=true; genomeLen=828
PASS d_export_import :: roundtrip byte-identical=true; activate identical=true; bytes=1049
PASS e_browser_clean :: module loaded over http://; no thrown errors; browser-only globals absent check=ok
VERDICT=PASS (5/5 capabilities)
SPIKE_DONE=true</pre>
```

What each capability proves:

| # | Capability | Evidence |
|---|---|---|
| a | User-supplied black-box fitness over a simulated rollout (not XOR/supervised) | `evaluatePopulation(fitnessFn)` called the supplied callback each generation; a 10-tick stash/spend rollout drove `bestFitness=935`; a constant-fitness control returned exactly `42`, proving the callback controls selection. |
| b | ≥2 independent populations + frozen genome | Two engines with different seeds evolved independently; a `cloneFrozen` copy stayed byte-identical after both runs. |
| c | Deterministic from explicit seed | Two independent engines with seed `123456` produced byte-identical best-genome JSON after 4 generations; a different seed (`999`) produced a different genome. |
| d | Genome export/import | `import(JSON.parse(export(g)))` re-exported byte-identically and produced identical `activate()` output. |
| e | No Node-only globals / no uncaught errors over http:// | Module loaded from `http://127.0.0.1`; no `window.error`, no unhandled rejection; `process`/`Buffer`/`Deno`/`require` absent. |

---

## 3. EXACT API the adapter worker must use

Module: `src/neat/inhouse-neat.js` (ES module, no build step).

```js
import { createInhouseNeat, createSeededRng, DEFAULT_CONFIG } from './inhouse-neat.js';
```

### 3.1 Factory

```js
const engine = createInhouseNeat({
  inputs: number,       // required, > 0, int
  outputs: number,      // required, > 0, int
  seed: number,         // required; coerced with >>> 0 (uint32)
  populationSize?: number, // default 50, must be >= 2
  config?: object       // partial override of DEFAULT_CONFIG
});
```

Engine properties: `inputs`, `outputs`, `seed`, `config` (frozen resolved config).

### 3.2 Engine methods (signatures)

| Method | Signature | Notes |
|---|---|---|
| `createGenome` | `createGenome() -> genome` | Fresh random genome (input→output seed genes). |
| `activate` | `activate(genome, inputsArray) -> number[]` | Deterministic forward pass; returns exactly `outputs` values; inputs are numbers (non-finite → 0). Use this inside rollouts. |
| `evaluatePopulation` | `evaluatePopulation(fitnessFn) -> summary` | `fitnessFn(genome, engine) -> number` (higher = better; non-finite/thrown → `-Infinity`). Runs one generation then breeds the next. |
| `getBestGenome` | `getBestGenome() -> genome \| null` | All-time best by fitness. **Returns `null` until the first `evaluatePopulation()` call.** Returns a clone. |
| `getGeneration` | `getGeneration() -> number` | Number of completed generations. |
| `getPopulation` | `getPopulation() -> genome[]` | Clones of the current (next-generation) population. |
| `exportGenome` | `exportGenome(genome) -> plainObject` | JSON/structured-clone-safe deep copy (no methods). |
| `importGenome` | `importGenome(plainObject) -> genome` | Validates + deep-copies; advances internal innovation/node counters to avoid id collisions. |
| `cloneFrozen` | `cloneFrozen(genome) -> genome` | Deep clone with `genome.frozen = true` and every node/gene `frozen = true`; mutation skips frozen genomes/genes. |
| `cloneGenome` | `cloneGenome(genome) -> genome` | Plain deep copy (frozen flags preserved). |
| `compatibility` | `compatibility(a, b) -> number` | NEAT compatibility distance; useful for diagnostics. |
| `reset` | `reset(newSeed?) -> engine` | Restores seed, RNG, innovation counters, generation 0 and a fresh population. Returns the engine for chaining. |
| `rngState` | `rngState() -> uint32` | Current PRNG state (diagnostics / checkpointing). |

`evaluatePopulation` summary:
```
{ generation:number, bestFitness:number, generationBestFitness:number,
  speciesCount:number, populationSize:number, bestGenome:genome }
```

### 3.3 Genome object shape (plain data — safe across worker.postMessage)

```js
{
  nodes: [
    { id:number, type:'input'|'output'|'hidden',
      activation:'LOGISTIC'|'TANH'|'RELU'|'IDENTITY',
      bias:number, depth:number, frozen:boolean }
  ],
  genes: [
    { innovation:number, from:number, to:number,
      weight:number, enabled:boolean, frozen:boolean }
  ],
  frozen:boolean,
  fitness:number,
  adjustedFitness:number,
  speciesId:number
}
```

Topology is feed-forward by construction: a connection is only legal when
`depth(from) < depth(to)`; activation visits nodes in `(depth, id)` order.

### 3.4 Seed / reproducibility rules

- One engine owns one mulberry32 PRNG seeded from `seed >>> 0`. There are no
  other sources of randomness (`Math.random`, `Date.now`, workers) in the module.
- Innovation numbers and hidden-node ids come from engine-level counters that
  `reset()` restores, so a re-run with the same seed reproduces the exact
  structural-innovation sequence.
- **Contract:** same `seed` + same `config` + same deterministic `fitnessFn`
  (which must use only `engine.activate` and its own closed-over state) ⇒
  byte-identical best genome. Verified over 4 generations in capability (c).
- The controller must give each of the 19 item populations its own engine with
  an explicit derived seed, and must call `evaluatePopulation()` before
  `getBestGenome()`.

### 3.5 Population / clone / freeze behavior

- Default `populationSize = 50`; elitism carries the single global best
  unchanged; speciation uses compatibility distance
  `c1·(disjoint+excess)/N + c3·meanMatchingWeightDelta` with threshold `2.6`;
  adjusted fitness uses fitness sharing with a positive shift so the app's
  **negative** coins-first fitness ranks correctly.
- Parents are selected from each species' top 2; offspring are innovation-aligned
  crossover (75%) or a clone, then mutated.
- Frozen genomes are for the "earlier items frozen" policy: obtain an evolved
  genome with `getBestGenome()`, make an immutable copy with `cloneFrozen(g)`,
  and feed it into later rollouts. A frozen genome is never silently added to a
  later population — the controller decides that explicitly (e.g. via
  `importGenome`). `cloneFrozen` round-trips byte-identically (capability b).
- `exportGenome`/`importGenome` are the serialization boundary for worker
  messages, results export and checkpointing.

### 3.6 Default config values (`DEFAULT_CONFIG`)

```
populationSize 50 | weightRandomRate 0.10 | weightPerturbRate 0.75
perturbStrength 0.35 | weightRange 3 | addConnectionRate 0.06
addNodeRate 0.04 | activationRate 0.01 | maxNodes 128 | maxGenes 512
c1 1.0 | c3 0.45 | speciesThreshold 2.6 | elitism 1 | crossoverRate 0.75
```

### 3.7 Mapping onto the plan's `neat/adapter.js` contract

The plan requires:
`createNeatAdapter({inputs,outputs,seed,config}) -> {evaluatePopulation(fitnessFn), getBestGenome(), exportGenome(g), importGenome(g), cloneFrozen(g)}`.
The in-house engine already implements those five methods verbatim. A thin
adapter is therefore all that is needed (written by the adapter worker, **not**
by this spike task):

```js
import { createInhouseNeat } from './inhouse-neat.js';

export function createNeatAdapter({ inputs, outputs, seed, config }) {
  const eng = createInhouseNeat({ inputs, outputs, seed, config });
  return {
    evaluatePopulation: (fitnessFn) => eng.evaluatePopulation(fitnessFn),
    getBestGenome:      () => eng.getBestGenome(),
    exportGenome:       (g) => eng.exportGenome(g),
    importGenome:       (g) => eng.importGenome(g),
    cloneFrozen:        (g) => eng.cloneFrozen(g),
    // REQUIRED IN ADDITION for fitness.js rollouts:
    activate:           (g, obs) => eng.activate(g, obs),
    reset:              (s) => eng.reset(s)
  };
}
```

`activate` is listed here because the plan's `fitness.js` rollout must query the
candidate network; without it the adapter cannot drive a simulation. `rng.js`
from the plan is unnecessary — the engine owns the seeded PRNG (see 3.4).

---

## 4. Known limitations

1. **Feed-forward only.** No recurrent/self connections. Sufficient for the
   per-item batch decision policy; do not expect memory across ticks beyond what
   `inputs.js` exposes in the observation.
2. **Small activation set** (`LOGISTIC`, `TANH`, `RELU`, `IDENTITY`).
3. **Structural caps** (`maxNodes=128`, `maxGenes=512`) bound rollout cost;
   override via `config` if needed.
4. **`getBestGenome()` is `null` before the first generation.** Call
   `evaluatePopulation()` first.
5. **Determinism is conditional on a pure `fitnessFn`.** Any `Math.random`,
   clock read or external mutable state inside the rollout breaks the same-seed
   guarantee.
6. **`activate` is O(nodes × genes) per call.** Acceptable for the spike; if
   heavy rollouts become a bottleneck, cache per-node incoming genes in the
   adapter (keep the genome itself plain JSON).
7. **`speciesId` is not a stable cross-run key** (species are discovered in
   population order each generation). Use genome JSON, not species id, for
   comparisons.
8. **No persistence or parallelism in this module.** All state is in memory;
   checkpointing is the caller's job via `exportGenome`/`importGenome`. The
   vendored bundle must NOT be dynamically imported at runtime — it hangs.
9. Vendored bundle remains in `src/vendor/` with a verified manifest for
   provenance only; it is dead weight for runtime and can be removed in a later
   cleanup task if Lead approves (not done here — outside this task's scope).

---

## 5. Ownership / scope note

Files created by this task, all inside the task's `Owns:` field:
`src/vendor/manifest.json`, `src/vendor/neat-ai/neat-ai.bundle.mjs`,
`spike.html`, `src/neat/inhouse-neat.js`, `docs/neat-spike.md`.
No `src/neat/{adapter,controller,fitness,inputs,defaults,rng}.js`, no UI file,
no app code, and no file outside `random_codes/0004_cat_game_neat/` was written.
Pre-existing unrelated modifications in `random_codes/0003_car_racer/` and
`.gitignore` were already present in the working tree and were not touched.
