// Tests for src/neat/adapter.js and src/neat/controller.js.
// Fakes are injected through adapterFactory/engineFactory; the real in-house
// adapter is also exercised directly for the exact spike API.
import { test, assert, assertEqual, assertDeepEqual, assertThrows } from './harness.js';
import * as controllerModule from '../neat/controller.js';
import { createNeatAdapter } from '../neat/adapter.js';
import {
  CRAFTABLE_IDS, TELEMETRY_CAP, OBSERVATION_SIZE, deriveSeed, createRingBuffer,
  createDisplayPolicyBundle, trainAll
} from '../neat/controller.js';
import { greedyBatch } from '../neat/defaults.js';
import { planBatchCaps } from '../neat/batch-planner.js';
import { fitness, trainingFitness, enrichResult } from '../neat/fitness.js';
import { CRAFTABLE_IDS as SIM_CRAFTABLE_IDS } from '../sim/items.js';
import { expandDemand, expandManufactureRequired } from '../sim/demand.js';
import { createEngine } from '../sim/engine.js';
import { EMBEDDED_CONFIG } from '../config.js';

function deepFreeze(o) {
  if (!o || typeof o !== 'object' || Object.isFrozen(o)) return o;
  for (const k of Object.keys(o)) deepFreeze(o[k]);
  return Object.freeze(o);
}

// ---- Fake adapter: deterministic output derived from the engine seed ----
function makeSeededAdapterFactory(outputForSeed, registry) {
  return ({ seed, inputs, outputs, config }) => {
    const s = seed >>> 0;
    const output = outputForSeed(s);
    let best = null;
    let gen = 0;
    const makeGenome = () => ({ output, seed: s, frozen: false, fitness: 0 });
    const api = {
      seed: s, inputs, outputs, config,
      evaluatePopulation(fn) {
        // One genome per generation keeps orchestration assertions focused;
        // population-size plumbing is verified separately via adapterFactory.
        const g = makeGenome();
        g.fitness = fn(g, api);
        best = g;
        gen++;
        return {
          generation: gen, bestFitness: best.fitness, generationBestFitness: best.fitness,
          speciesCount: 1, populationSize: 1, bestGenome: best
        };
      },
      getBestGenome() { return best ? JSON.parse(JSON.stringify(best)) : null; },
      exportGenome(g) { return JSON.parse(JSON.stringify(g)); },
      importGenome(g) { return JSON.parse(JSON.stringify(g)); },
      cloneFrozen(g) {
        const c = JSON.parse(JSON.stringify(g));
        c.frozen = true;
        deepFreeze(c);
        if (registry) registry.push({ ref: c, before: JSON.stringify(c) });
        return c;
      },
      activate(g, obs) {
        if (registry && g && g.frozen) registry.activations.push(obs.slice());
        return [g.output];
      },
      reset() { gen = 0; best = null; return api; }
    };
    if (registry) registry.activations = registry.activations || [];
    return api;
  };
}

// ---- Fake engine: batch-dependent, deterministic rollout result ----
function makeRecordingEngineFactory(state) {
  state.resultCalls = 0;
  state.allBatches = [];
  return ({ policyForItem }) => {
    let tick = 0;
    return {
      step() {
        const batches = [];
        for (const id of Object.keys(policyForItem)) {
          const b = policyForItem[id]({
            tick, coins: 1000, stash: {}, slots: {}, starvation: {},
            materialsAvailable: 20, slotActive: false
          });
          batches.push([id, b]);
        }
        state.allBatches.push(batches);
        tick++;
      },
      snapshot() {
        return { tick, done: state.allBatches.length > 0, coins: 1000, stash: {}, slots: {}, starvation: {} };
      },
      result() {
        state.resultCalls++;
        const last = state.allBatches[state.allBatches.length - 1] || [];
        let totalSpent = 0;
        const perItem = {};
        for (const [id, b] of last) {
          const spend = b * 50;
          totalSpent += spend;
          perItem[id] = {
            coinsSpent: spend, timeMinutes: b * 5, busyTicks: b,
            batches: b ? 1 : 0, units: b, firstTick: tick, lastTick: tick
          };
        }
        return {
          feasible: state.feasible !== false,
          completionTick: 8064 - totalSpent / 50,
          elapsedTicks: 8064 - totalSpent / 50,
          totalSpent, produced: {}, schedule: [], shortfall: {}, perItem
        };
      }
    };
  };
}

// ---- Fake engine that is feasible on every rollout (early-stop probe) ----
function makeFeasibleEngineFactory() {
  return () => ({
    step() {},
    snapshot() { return { tick: 1, done: true, stash: {}, slots: {}, starvation: {}, coins: 0 }; },
    result() {
      return {
        feasible: true, completionTick: 10, elapsedTicks: 11, totalSpent: 100,
        produced: {}, schedule: [], shortfall: {}, perItem: {}
      };
    }
  });
}

// ---------------------------------------------------------------------------
// Adapter — exact spike §3.7 API over the real in-house engine
// ---------------------------------------------------------------------------

test('adapter: getBestGenome is null before the first evaluatePopulation', async () => {
  const a = createNeatAdapter({ inputs: OBSERVATION_SIZE, outputs: 1, seed: 42 });
  assertEqual(a.getBestGenome(), null);
  a.evaluatePopulation(() => 1);
  assert(a.getBestGenome() !== null, 'best genome must exist after one generation');
});

test('adapter: evaluatePopulation invokes the black-box fitness for the whole population', async () => {
  const a = createNeatAdapter({ inputs: OBSERVATION_SIZE, outputs: 1, seed: 7, config: { populationSize: 4 } });
  let calls = 0;
  a.evaluatePopulation(() => { calls++; return 5; });
  assertEqual(calls, 4);
  assertEqual(a.getBestGenome().fitness, 5);
});

test('adapter: activate is deterministic and returns exactly outputs values', async () => {
  const a = createNeatAdapter({ inputs: OBSERVATION_SIZE, outputs: 1, seed: 9 });
  a.evaluatePopulation(() => 3);
  const g = a.getBestGenome();
  const obs = [0.5, 0.1, 0.9, 0.2, 0];
  const o1 = a.activate(g, obs);
  const o2 = a.activate(g, obs);
  assertEqual(o1.length, 1);
  assertDeepEqual(o1, o2);
});

test('adapter: export/import round-trips byte-identically and preserves activation', async () => {
  const a = createNeatAdapter({ inputs: OBSERVATION_SIZE, outputs: 1, seed: 11 });
  a.evaluatePopulation(() => 8);
  const g = a.getBestGenome();
  const exported = a.exportGenome(g);
  const imported = a.importGenome(exported);
  assertDeepEqual(a.exportGenome(imported), exported);
  const obs = [0.2, 0.4, 0.6, 0.8, 1];
  assertDeepEqual(a.activate(imported, obs), a.activate(g, obs));
});

test('adapter: cloneFrozen marks genome/nodes/genes frozen', async () => {
  const a = createNeatAdapter({ inputs: OBSERVATION_SIZE, outputs: 1, seed: 13 });
  a.evaluatePopulation(() => 2);
  const f = a.cloneFrozen(a.getBestGenome());
  assertEqual(f.frozen, true);
  assert(f.nodes.every((n) => n.frozen === true));
  assert(f.genes.every((x) => x.frozen === true));
});

test('adapter: reset(seed) restores a reproducible population', async () => {
  const a = createNeatAdapter({ inputs: OBSERVATION_SIZE, outputs: 1, seed: 17, config: { populationSize: 4 } });
  a.evaluatePopulation(() => 4);
  const first = a.exportGenome(a.getBestGenome());
  a.reset(17);
  a.evaluatePopulation(() => 4);
  assertDeepEqual(a.exportGenome(a.getBestGenome()), first);
});

// ---------------------------------------------------------------------------
// Controller constants / helpers
// ---------------------------------------------------------------------------

test('controller: CRAFTABLE_IDS is statically re-exported from sim/items.js (no local duplicate)', async () => {
  assert(CRAFTABLE_IDS === SIM_CRAFTABLE_IDS, 'controller must re-export the sim core order directly');
  assertDeepEqual(CRAFTABLE_IDS, SIM_CRAFTABLE_IDS);
  assertEqual(CRAFTABLE_IDS.length, 19);
  assertEqual(OBSERVATION_SIZE, 5);
  assertEqual(TELEMETRY_CAP, 5000);
});

test('controller: deriveSeed is deterministic, uint32, and distinct per item', async () => {
  assertEqual(deriveSeed(123, 0), deriveSeed(123, 0));
  const seeds = CRAFTABLE_IDS.map((_, k) => deriveSeed(123, k));
  assertEqual(new Set(seeds).size, 19, 'each item population needs its own seed');
  for (const s of seeds) {
    assert(Number.isInteger(s) && s >= 0 && s <= 0xffffffff, `invalid derived seed ${s}`);
  }
  assert(deriveSeed(123, 0) !== deriveSeed(124, 0), 'master seed must change derived seeds');
});

test('controller: createRingBuffer evicts oldest entries at capacity', async () => {
  const rb = createRingBuffer(3);
  rb.push('a'); rb.push('b'); rb.push('c'); rb.push('d'); rb.push('e');
  assertEqual(rb.length, 3);
  assertEqual(rb.capacity, 3);
  assertDeepEqual(rb.toArray(), ['c', 'd', 'e']);
  assertEqual(createRingBuffer().capacity, 5000);
});

test('controller: rejects a missing engineFactory', async () => {
  let threw = false;
  try { await trainAll({ config: {}, seed: 1 }); } catch (_err) { threw = true; }
  assert(threw);
});

// ---------------------------------------------------------------------------
// Controller orchestration with injected fakes
// ---------------------------------------------------------------------------

test('controller: every genome evaluation calls engine.result() and emits GENOME_RESULT telemetry', async () => {
  const state = {};
  const { telemetry } = await trainAll({
    adapterFactory: makeSeededAdapterFactory(() => 0.5, null),
    engineFactory: makeRecordingEngineFactory(state),
    config: { generations: 1 },
    seed: 100
  });
  const events = telemetry.toArray();
  assertEqual(events.length, 19, 'one evaluation per item with a single-genome population');
  assert(state.resultCalls >= events.length, 'engine.result() must be called for every evaluation');
  const requiredFields = ['type', 'itemId', 'generation', 'genomeIndex', 'fitness', 'feasible', 'completionTick', 'elapsedTicks', 'totalSpent', 'perItem'];
  for (const e of events) {
    assertEqual(e.type, 'GENOME_RESULT');
    for (const f of requiredFields) assert(f in e, `telemetry missing field ${f}`);
    assertDeepEqual(CRAFTABLE_IDS.includes(e.itemId), true);
    assert(Number.isFinite(e.fitness), 'fitness must be finite');
  }
});

test('controller: telemetry per-item coins sum to totalSpent (attribution)', async () => {
  const state = {};
  const { telemetry } = await trainAll({
    adapterFactory: makeSeededAdapterFactory(() => 0.5, null),
    engineFactory: makeRecordingEngineFactory(state),
    config: { generations: 1 }, seed: 101
  });
  const events = telemetry.toArray();
  assertEqual(events.length, 19);
  for (const e of events) {
    const sum = Object.values(e.perItem).reduce((a, row) => a + row.coinsSpent, 0);
    assertEqual(sum, e.totalSpent);
  }
});

test('controller: candidate batch change changes fitness (only the candidate varies)', async () => {
  const MASTER = 202;
  const artifactSeed = deriveSeed(MASTER, CRAFTABLE_IDS.length - 1);
  const outputFor = (candidateOutput) => (seed) => seed === artifactSeed ? candidateOutput : 0.5;

  const runA = await trainAll({
    adapterFactory: makeSeededAdapterFactory(outputFor(1.0), null),
    engineFactory: makeRecordingEngineFactory({}),
    config: { generations: 1 }, seed: MASTER
  });
  const runB = await trainAll({
    adapterFactory: makeSeededAdapterFactory(outputFor(0.0), null),
    engineFactory: makeRecordingEngineFactory({}),
    config: { generations: 1 }, seed: MASTER
  });
  const evA = runA.telemetry.toArray();
  const evB = runB.telemetry.toArray();
  // Earlier items are identical; the artifact (current candidate) differs.
  assertDeepEqual(evA.slice(0, -1).map((e) => e.fitness), evB.slice(0, -1).map((e) => e.fitness));
  assert(evA[18].fitness !== evB[18].fitness,
    `artifact candidate output must change fitness (${evA[18].fitness} vs ${evB[18].fitness})`);
  assert(evA[18].totalSpent > evB[18].totalSpent, 'batch 20 candidate must spend more than batch 1');
});

test('controller: frozen genomes stay byte-identical after later training and are reused', async () => {
  const registry = [];
  await trainAll({
    adapterFactory: makeSeededAdapterFactory(() => 0.5, registry),
    engineFactory: makeRecordingEngineFactory({}),
    config: { generations: 1 }, seed: 303
  });
  assertEqual(registry.length, 19, 'one frozen genome per item');
  for (const entry of registry) {
    assertEqual(JSON.stringify(entry.ref), entry.before, 'frozen genome mutated during later training');
  }
  assert(registry.activations.length > 0, 'frozen genomes must be activated in later rollouts');
});

test('controller: later items chase exact downstream manufactureRequired, not raw targets', async () => {
  const items = EMBEDDED_CONFIG.items;
  const targets = { artifact: 3 };
  const starterStash = {};
  const starterMode = 'FREE_INGREDIENTS';
  // Policies/observations now use the manufactureRequired basis (the engine
  // hard cap), while required remains untouched for legacy tooltips.
  const expected = expandManufactureRequired(items, targets, starterStash, starterMode).manufactureRequired;
  const caps = planBatchCaps(items, expected, 8064, 20).caps;
  const state = {};
  await trainAll({
    adapterFactory: makeSeededAdapterFactory(() => 0.0, null),
    engineFactory: makeRecordingEngineFactory(state),
    config: { items, targets, initialStash: starterStash, starterMode, generations: 1 }, seed: 404
  });
  const firstEval = state.allBatches[0];
  const batchOf = (id) => (firstEval.find(([x]) => x === id) || [null, null])[1];
  assertEqual(batchOf('string'), 1, 'current candidate with output 0 -> batch 1');
  assert(expected.ribbon > 20, 'ribbon inherits downstream artifact demand');
  assertEqual(batchOf('ribbon'), Math.min(caps.ribbon, 20, expected.ribbon),
    'later item chases expanded mfr demand, clamped by its planner cap');
  assertEqual(batchOf('artifact'), Math.min(caps.artifact, 20, expected.artifact), 'terminal target demand');
});

test('controller: feasible fitness on telemetry follows -(totalSpent*BIG + completionTick)', async () => {
  const { telemetry } = await trainAll({
    adapterFactory: makeSeededAdapterFactory(() => 0.5, null),
    engineFactory: makeRecordingEngineFactory({}),
    config: { generations: 1 }, seed: 505
  });
  const e = telemetry.toArray()[0];
  assertEqual(e.fitness, -(e.totalSpent * 100000 + e.completionTick));
  assertEqual(e.feasible, true);
});

test('controller: same seed and config produce byte-identical genomes and telemetry with real NEAT', async () => {
  const config = {
    generations: 2,
    adapterConfig: { populationSize: 4 },
    items: EMBEDDED_CONFIG.items,
    targets: { orb: 4, artifact: 2 },
    initialStash: {},
    starterMode: 'FREE_INGREDIENTS'
  };
  const r1 = await trainAll({
    adapterFactory: createNeatAdapter,
    engineFactory: makeRecordingEngineFactory({ feasible: false }),
    config, seed: 606
  });
  const r2 = await trainAll({
    adapterFactory: createNeatAdapter,
    engineFactory: makeRecordingEngineFactory({ feasible: false }),
    config, seed: 606
  });
  assertEqual(Object.keys(r1.genomes).length, 19, 'all 19 genomes must be present');
  assertDeepEqual(JSON.parse(JSON.stringify(r1.genomes)), JSON.parse(JSON.stringify(r2.genomes)));
  assertEqual(r1.telemetry.length, 19 * 2 * 4, 'one event per genome evaluation');
  assertDeepEqual(r1.telemetry.toArray(), r2.telemetry.toArray());
});

test('controller: exact reverse-topo demand is authoritative and passed to the engine factory', async () => {
  const items = EMBEDDED_CONFIG.items;
  const targets = { artifact: 2 };
  const initialStash = {};
  const starterMode = 'FREE_INGREDIENTS';
  const expected = expandDemand(items, targets, initialStash, starterMode).required;
  const received = {};
  await trainAll({
    adapterFactory: makeSeededAdapterFactory(() => 0, null),
    engineFactory: (opts) => {
      received.required = opts.required;
      return makeRecordingEngineFactory(received)(opts);
    },
    config: { items, targets, initialStash, starterMode, generations: 1 }, seed: 707
  });
  assertDeepEqual(received.required, expected, 'controller must pass expandDemand().required');
  assert(received.required.necklace > 0, 'downstream necklace demand must be expanded');
  assert(received.required.elementstone > 0, 'downstream elementstone demand must be expanded');
  assert(Object.keys(received.required).length > Object.keys(targets).length,
    'exact demand must be richer than raw targets (no targets fallback)');
});

test('controller: forwards each GENOME_RESULT to onProgress from the evaluation callback', async () => {
  const seen = [];
  const { telemetry } = await trainAll({
    adapterFactory: makeSeededAdapterFactory(() => 0.5, null),
    engineFactory: makeRecordingEngineFactory({}),
    config: {
      items: EMBEDDED_CONFIG.items, targets: { orb: 1 },
      initialStash: {}, starterMode: 'FREE_INGREDIENTS', generations: 1
    },
    seed: 708,
    onProgress: (ev) => seen.push(ev)
  });
  const forwarded = seen.filter((e) => e.type === 'GENOME_RESULT');
  assertEqual(forwarded.length, 19, 'one GENOME_RESULT per genome evaluation');
  assertDeepEqual(forwarded, telemetry.toArray(), 'forwarded events match the ring buffer');
  assertEqual(seen[0].type, 'GENOME_RESULT', 'telemetry emitted inside the evaluation callback');
  assert(seen.some((e) => e.type === 'PROGRESS'));
  assert(seen.some((e) => e.type === 'BEST'));
});

test('controller: real engine yields byte-identical schedules for the same seed + config', async () => {
  const config = {
    items: EMBEDDED_CONFIG.items,
    targets: { orb: 2, silver: 1 },
    initialCoins: 100000,
    initialStash: {},
    starterMode: 'FREE_INGREDIENTS',
    generations: 1,
    adapterConfig: { populationSize: 3 },
    maxTicks: 40
  };
  const run = async () => {
    const schedules = [];
    const engineFactory = (opts) => {
      const eng = createEngine(opts);
      const originalResult = eng.result;
      eng.result = () => {
        const r = originalResult();
        schedules.push(r.schedule);
        return r;
      };
      return eng;
    };
    const result = await trainAll({ adapterFactory: createNeatAdapter, engineFactory, config, seed: 709 });
    return { schedules, result };
  };
  const a = await run();
  const b = await run();
  assert(a.schedules.length > 0, 'engine.result() must be recorded');
  assert(a.schedules.some((s) => s.length > 0), 'at least one non-empty schedule');
  assertDeepEqual(a.schedules, b.schedules, 'schedules must be byte-identical across runs');
  assertDeepEqual(a.result.telemetry.toArray(), b.result.telemetry.toArray());
  assertDeepEqual(JSON.parse(JSON.stringify(a.result.genomes)), JSON.parse(JSON.stringify(b.result.genomes)));
});

// ---------------------------------------------------------------------------
// F1 — termination budget / quiet rollouts; F3 — population size control
// ---------------------------------------------------------------------------

test('controller: passes quiet:true to engineFactory for every fitness rollout', async () => {
  const flags = [];
  await trainAll({
    adapterFactory: makeSeededAdapterFactory(() => 0.5, null),
    engineFactory: (opts) => {
      flags.push(opts.quiet);
      return makeRecordingEngineFactory({})(opts);
    },
    config: { generations: 1 }, seed: 710
  });
  assert(flags.length >= 19, 'one rollout per genome plus the final best re-evaluation');
  assert(flags.every((v) => v === true), `all rollouts must be quiet: ${JSON.stringify(flags)}`);
});

test('controller: exports budget constants and caps generations via maxGenerations', async () => {
  assert('DEFAULT_GENS' in controllerModule, 'DEFAULT_GENS must be exported');
  assert('DEFAULT_POP' in controllerModule, 'DEFAULT_POP must be exported');
  assert('DEFAULT_WALL_MS' in controllerModule, 'DEFAULT_WALL_MS must be exported');
  const { DEFAULT_GENS, DEFAULT_POP, DEFAULT_WALL_MS } = controllerModule;
  assert(DEFAULT_GENS >= 1 && DEFAULT_GENS <= 10);
  assert(DEFAULT_POP >= 2, 'inhouse-neat requires populationSize >= 2');
  assert(DEFAULT_WALL_MS > 0);

  const { genomes, telemetry } = await trainAll({
    adapterFactory: makeSeededAdapterFactory(() => 0.5, null),
    engineFactory: makeRecordingEngineFactory({ feasible: false }),
    config: { generations: 100, maxGenerations: 2, adapterConfig: { populationSize: 2 } }, seed: 712
  });
  assertEqual(Object.keys(genomes).length, 19, 'all 19 real best genomes must exist');
  assert(genomes.string && typeof genomes.string === 'object');
  assertEqual(telemetry.length, 19 * 2, 'maxGenerations caps the generation loop');
});

test('controller: maxWallMs=0 stops before the first population (DONE-safe)', async () => {
  const { genomes, telemetry } = await trainAll({
    adapterFactory: makeSeededAdapterFactory(() => 0.5, null),
    engineFactory: makeRecordingEngineFactory({ feasible: false }),
    config: { generations: 5, maxWallMs: 0, adapterConfig: { populationSize: 2 } }, seed: 714
  });
  assertEqual(Object.keys(genomes).length, 0, 'an expired deadline starts no population');
  assertEqual(telemetry.length, 0);
});

test('controller: early-stops an item once a feasible genome is found', async () => {
  const { telemetry } = await trainAll({
    adapterFactory: makeSeededAdapterFactory(() => 0.5, null),
    engineFactory: makeFeasibleEngineFactory(),
    config: { generations: 10, adapterConfig: { populationSize: 1 } }, seed: 713
  });
  assertEqual(telemetry.length, 19, 'one generation per item after feasibility');
});

test('controller: sanitizes adapterConfig.populationSize (>=2) before adapter creation', async () => {
  const seen = [];
  const capture = (opts) => {
    seen.push(opts.config);
    return makeSeededAdapterFactory(() => 0.5, null)(opts);
  };
  await trainAll({
    adapterFactory: capture, engineFactory: makeRecordingEngineFactory({}),
    config: { generations: 1, adapterConfig: { populationSize: 7 } }, seed: 715
  });
  assertEqual(seen[0].populationSize, 7, 'valid populationSize is honoured');

  const seen2 = [];
  const capture2 = (opts) => {
    seen2.push(opts.config);
    return makeSeededAdapterFactory(() => 0.5, null)(opts);
  };
  await trainAll({
    adapterFactory: capture2, engineFactory: makeRecordingEngineFactory({}),
    config: { generations: 1, adapterConfig: { populationSize: 1 } }, seed: 716
  });
  assert(seen2[0].populationSize >= 2, 'inhouse-neat throws when populationSize < 2');
  assertEqual(seen2[0].populationSize, 2, 'out-of-range size is clamped to the minimum');
});

test('controller: DONE still has real best genomes under a normal wall budget', async () => {
  const { genomes, telemetry } = await trainAll({
    adapterFactory: createNeatAdapter,
    engineFactory: makeFeasibleEngineFactory(),
    config: { generations: 100, maxWallMs: 5000, adapterConfig: { populationSize: 2 } }, seed: 717
  });
  assertEqual(Object.keys(genomes).length, 19);
  const best = genomes.artifact;
  assert(best && Array.isArray(best.nodes) && Array.isArray(best.genes), 'best genome must be a real genome');
  assert(telemetry.length > 0, 'at least one GENOME_RESULT was emitted');
});

// ---------------------------------------------------------------------------
// Display-policy bundle (trained genomes drive the main-thread display run)
// ---------------------------------------------------------------------------

test('controller: display bundle uses genome policy when present, greedy otherwise', async () => {
  const required = { orb: 3 };
  const bundle = createDisplayPolicyBundle({
    genomes: { string: { output: 1.0 } }, // fake adapter maps this to batch 20
    config: { items: EMBEDDED_CONFIG.items },
    required,
    manufactureRequired: required,
    adapterFactory: makeSeededAdapterFactory(() => 1.0, null),
    seed: 900,
    maxTicks: 40
  });
  const ctx = { itemId: 'string', stash: {}, produced: 0, coins: 0, tick: 0 };
  assertEqual(bundle.policyForItem.string(ctx), 20, 'genome output 1.0 decodes to MAX_BATCH');
  assertEqual(bundle.policyForItem.orb(ctx), greedyBatch('orb', ctx, required),
    'item without a genome uses the greedy default');
  assertEqual(typeof bundle.setEngine, 'function');
});

test('controller: display bundle setEngine changes the live observation', async () => {
  const obsLog = [];
  const spyFactory = (opts) => {
    const a = createNeatAdapter(opts);
    return {
      ...a,
      activate: (g, obs) => { obsLog.push(obs.slice()); return a.activate(g, obs); }
    };
  };
  const id = CRAFTABLE_IDS[0];
  const bundle = createDisplayPolicyBundle({
    genomes: { [id]: { nodes: [], genes: [] } }, // validated (or rejected) by importGenome
    config: { items: EMBEDDED_CONFIG.items },
    required: { [id]: 1 },
    adapterFactory: spyFactory,
    seed: 901,
    maxTicks: 40
  });
  bundle.policyForItem[id]({});
  const before = obsLog[obsLog.length - 1].slice();
  bundle.setEngine({
    snapshot: () => ({
      tick: 1234, coins: 777777, stash: { [id]: 5 },
      slots: {}, starvation: {}, done: false
    })
  });
  bundle.policyForItem[id]({});
  const after = obsLog[obsLog.length - 1].slice();
  assert(JSON.stringify(before) !== JSON.stringify(after),
    `setEngine must change observations: ${JSON.stringify(before)} vs ${JSON.stringify(after)}`);
});

test('controller: real evolved genome changes at least one display batch vs greedy', async () => {
  const config = {
    items: EMBEDDED_CONFIG.items,
    targets: { orb: 2 },
    initialStash: {},
    starterMode: 'FREE_INGREDIENTS',
    generations: 1,
    adapterConfig: { populationSize: 2 },
    maxTicks: 40
  };
  const { genomes } = await trainAll({
    adapterFactory: createNeatAdapter,
    engineFactory: makeRecordingEngineFactory({ feasible: false }),
    config, seed: 902
  });
  assertEqual(Object.keys(genomes).length, 19, 'real evolved genomes available');

  const required = {};
  for (const id of CRAFTABLE_IDS) required[id] = 1; // greedy always asks for 1
  const bundle = createDisplayPolicyBundle({ genomes, config, required, seed: 902, maxTicks: 40 });
  const ctx = { stash: {}, produced: 0, coins: 0, tick: 0 };
  let different = 0;
  for (const id of CRAFTABLE_IDS) {
    const genomeBatch = bundle.policyForItem[id](ctx);
    const greedy = greedyBatch(id, ctx, required);
    if (genomeBatch !== greedy) different += 1;
  }
  assert(different > 0, 'a trained genome must change at least one batch decision vs greedy');
});

// ---------------------------------------------------------------------------
// Wall deadline must actually bound wall-clock time (adversarial U1)
// ---------------------------------------------------------------------------

test('controller: reserved gen-0 slice guarantees generation 0 for all 19 (injectable clock)', async () => {
  // FAKE CLOCK: no busy-wait, no real-time dependency. Each tick costs 1 ms.
  // With a tight 19 ms budget and 10 generations, reserveMs = 19/19/10 = 0.1 ms
  // per item's gen 0. The OLD outer `now() >= deadline` break would skip items
  // 1..18 entirely; the structural reservation must still give all 19 a genome.
  let virtualMs = 0;
  let steps = 0;
  const now = () => virtualMs;
  const slowEngineFactory = () => ({
    step() { steps += 1; virtualMs += 1; },
    snapshot() { return { tick: steps, done: false, stash: {}, slots: {}, starvation: {}, coins: 0 }; },
    result() {
      return {
        feasible: false, completionTick: null, elapsedTicks: steps, totalSpent: 0,
        produced: {}, schedule: [], shortfall: {}, perItem: {}
      };
    }
  });
  const { genomes, telemetry } = await trainAll({
    adapterFactory: (opts) => makeSeededAdapterFactory(() => 0.5, null)(opts),
    engineFactory: slowEngineFactory,
    config: { generations: 10, maxTicks: 1000, adapterConfig: { populationSize: 2 } },
    maxWallMs: 19,
    now,
    seed: 903
  });
  assertEqual(Object.keys(genomes).length, 19, 'all 19 items must have a genome (gen 0 never skipped)');
  const events = telemetry.toArray();
  for (const id of CRAFTABLE_IDS) {
    assert(events.some((e) => e.itemId === id && e.generation === 1),
      `${id}: generation 0 must have been evaluated`);
  }
  // gen-0 slices are one tick each; the final best re-eval is bounded by the
  // global deadline, so total clock time stays within budget + one budget.
  assert(virtualMs <= 38, `reserved slices must stay bounded (clock=${virtualMs})`);
  assert(telemetry.length >= 19, 'at least one GENOME_RESULT per item');
});

// ---------------------------------------------------------------------------
// Curriculum truncation, telemetry fitness, re-evaluation, mfr fallback
// ---------------------------------------------------------------------------

test('controller: generations 0-1 truncate the rollout to 2000 ticks only; later use full maxTicks', async () => {
  const config = {
    items: EMBEDDED_CONFIG.items,
    targets: { orb: 2 },
    initialStash: {},
    starterMode: 'FREE_INGREDIENTS',
    generations: 3,
    maxTicks: 8064,
    adapterConfig: { populationSize: 1 }
  };
  const { telemetry } = await trainAll({
    adapterFactory: makeSeededAdapterFactory(() => 0.5, null),
    engineFactory: () => {
      let steps = 0;
      return {
        step() { steps += 1; },
        snapshot() { return { done: false, stash: {}, slots: {}, starvation: {}, coins: 0 }; },
        result() {
          return {
            feasible: false, completionTick: null, elapsedTicks: steps, totalSpent: 0,
            produced: {}, schedule: [], shortfall: {}, perItem: {}
          };
        }
      };
    },
    config, seed: 904
  });
  const e = telemetry.toArray();
  assertEqual(e[0].elapsedTicks, 2000, 'gen 0 rollout truncated to 2000');
  assertEqual(e[1].elapsedTicks, 2000, 'gen 1 rollout truncated to 2000');
  assertEqual(e[2].elapsedTicks, 8064, 'gen 2 uses the full horizon');
});

test('controller: observation is byte-identical across curriculum phases (maxTicks fixed 8064)', async () => {
  const observed = [];
  const id = CRAFTABLE_IDS[0];
  const spyFactory = (opts) => {
    const a = makeSeededAdapterFactory(() => 0.5, null)(opts);
    return { ...a, activate: (g, obs) => { observed.push(obs.slice()); return a.activate(g, obs); } };
  };
  await trainAll({
    adapterFactory: spyFactory,
    engineFactory: makeRecordingEngineFactory({ feasible: false }),
    config: {
      items: EMBEDDED_CONFIG.items, targets: { orb: 2 }, initialStash: {},
      starterMode: 'FREE_INGREDIENTS', generations: 3, maxTicks: 8064,
      adapterConfig: { populationSize: 1 }
    },
    seed: 905
  });
  assert(observed.length >= 3, 'observations from first item across 3 generations');
  const first = JSON.stringify(observed[0]);
  for (let i = 1; i < 3; i++) {
    assertEqual(JSON.stringify(observed[i]), first, `observation changed across curriculum gen ${i}`);
  }
  // The fixed maxTicks is 8064: a tick-0, fresh-stash item sees 1.0 remaining.
  assertEqual(observed[0][2], 1, 'ticksRemaining uses the full 8064 horizon');
});

test('controller: telemetry fitness stays the shipped fitness while selection uses trainingFitness', async () => {
  const targets = { orb: 4 };
  let rawResult = null;
  const engineFactory = () => ({
    step() {},
    snapshot() { return { done: true, stash: {}, slots: {}, starvation: {}, coins: 0 }; },
    result() {
      rawResult = {
        feasible: false, completionTick: null, elapsedTicks: 100, totalSpent: 0,
        produced: { orb: 2 }, schedule: [], shortfall: {}, perItem: {}
      };
      return JSON.parse(JSON.stringify(rawResult));
    }
  });
  const { telemetry } = await trainAll({
    adapterFactory: makeSeededAdapterFactory(() => 0.5, null),
    engineFactory,
    config: {
      items: EMBEDDED_CONFIG.items, targets, initialStash: {},
      starterMode: 'FREE_INGREDIENTS', generations: 1, maxTicks: 8064,
      adapterConfig: { populationSize: 1 }
    },
    seed: 906
  });
  const ev = telemetry.toArray()[0];
  const mfr = expandManufactureRequired(
    EMBEDDED_CONFIG.items, targets, {}, 'FREE_INGREDIENTS'
  ).manufactureRequired;
  const expectedShipped = fitness(enrichResult(rawResult, targets, mfr, 8064));
  assertEqual(ev.fitness, expectedShipped, 'GENOME_RESULT.fitness is the shipped fitness()');
  const shaped = trainingFitness(enrichResult(rawResult, targets, mfr, 8064), 'pre', mfr, 8064);
  assert(Number.isFinite(shaped), 'selection signal must be finite');
});

test('controller: re-evaluates the best genome before freezing (extra rollouts per item)', async () => {
  let engineCalls = 0;
  await trainAll({
    adapterFactory: makeSeededAdapterFactory(() => 0.5, null),
    engineFactory: (opts) => {
      engineCalls += 1;
      return makeFeasibleEngineFactory()(opts);
    },
    config: { generations: 5, adapterConfig: { populationSize: 1 } }, seed: 907
  });
  // gen0 rollout + in-loop recheck + final re-check => 3 engine rollouts/item.
  assert(engineCalls >= 19 * 2, `best genome must be re-evaluated before freeze (calls=${engineCalls})`);
});

test('controller: display bundle fallback uses expandManufactureRequired, not expandDemand.required', () => {
  const items = EMBEDDED_CONFIG.items;
  const targets = { artifact: 2 };
  const stash = { ribbon: 5 }; // PREDELIVERED_RESERVED: mfr != required here
  const mode = 'PREDELIVERED_RESERVED';
  const config = { items, targets, initialStash: stash, starterMode: mode };
  const mfr = expandManufactureRequired(items, targets, stash, mode).manufactureRequired;
  const required = expandDemand(items, targets, stash, mode).required;
  const bundle = createDisplayPolicyBundle({
    config, // no manufactureRequired option -> falls back to expandManufactureRequired
    adapterFactory: makeSeededAdapterFactory(() => 0.5, null),
    seed: 908, maxTicks: 40
  });
  // Pick an item where mfr differs from required to prove the basis switched.
  const id = CRAFTABLE_IDS.find((x) => mfr[x] !== required[x]);
  assert(id !== undefined, 'there is an item where mfr and required differ');
  const ctx = { produced: 0, stash: {}, coins: 0, tick: 0 };
  assertEqual(bundle.policyForItem[id](ctx), greedyBatch(id, ctx, mfr), 'fallback uses mfr basis');
});
