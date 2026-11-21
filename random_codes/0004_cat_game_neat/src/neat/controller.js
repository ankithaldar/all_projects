/**
 * controller.js — 19 sequential, dependency-topo-ordered NEAT populations.
 *
 * Policy at step k (plan "Training Orchestration"):
 *   - items j < k : frozen evolved genomes (cloneFrozen, never mutated)
 *   - item  k    : the candidate genome under evaluation
 *   - items j > k : greedy default from defaults.js (exact required[item] goal)
 *
 * Every genome evaluation drives a DOM-free engine rollout via engineFactory,
 * calls engine.result(), computes the shipped fitness for telemetry AND the
 * bounded trainingFitness for selection, and BOTH stores a bounded
 * GENOME_RESULT in the ring buffer and forwards it through onProgress so the
 * worker can post it. Rollouts are quiet, generation count and wall time are
 * explicitly budgeted, an item stops evolving only after its best genome is
 * re-evaluated feasible in the final upstream context, and generation 0 is
 * structurally reserved for all 19 items.
 *
 * Policy/observation basis is `manufactureRequired` (the hard engine cap).
 * The engine's policyContext.required / remaining keys are retained for legacy
 * UI/tooltips but are deliberately NOT read here.
 */
import { createNeatAdapter } from './adapter.js';
import { fitness, trainingFitness, enrichResult } from './fitness.js';
import { makeObservation, decodeBatch, MAX_BATCH, OBSERVATION_SIZE as OBS_SIZE } from './inputs.js';
import { greedyBatch } from './defaults.js';
import { applyBatchCaps } from './policy-caps.js';
import { planBatchCaps } from './batch-planner.js';
import { createSeededRng } from './inhouse-neat.js';
import { CRAFTABLE_IDS } from '../sim/items.js';
import { expandDemand, expandManufactureRequired } from '../sim/demand.js';

// The sim core is the single source of truth for the dependency-respecting
// topological order (YAML `tier` labels are inconsistent with dependencies).
export { CRAFTABLE_IDS };

/** Bounded telemetry ring buffer shared with the logger (plan LOG_CAP). */
export const TELEMETRY_CAP = 5000;
/** Observation vector length consumed by every per-item network. */
export const OBSERVATION_SIZE = OBS_SIZE;

/** Budget defaults exported so worker/UI controls agree on the same numbers. */
export const DEFAULT_GENS = 10;
export const DEFAULT_POP = 20;
export const DEFAULT_WALL_MS = 180000;

const DEFAULT_MAX_TICKS = 8064;
/** Generations 0-1 truncate the ROLLOUT only; observation maxTicks stays 8064. */
const CURRICULUM_GENS = 2;
const CURRICULUM_MAX_TICKS = 2000;

function intInRange(value, fallback, min, max) {
  if (value === undefined || value === null) return fallback;
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, Math.floor(n)));
}

const isObj = (v) => !!v && typeof v === 'object' && !Array.isArray(v);

/**
 * Derive an item's population seed deterministically from the master seed.
 * Integer mix (murmur-style finalizer) -> uint32, unique across item indices.
 */
export function deriveSeed(masterSeed, index) {
  let x = (Number(masterSeed) >>> 0) ^ Math.imul(((Number(index) >>> 0) + 1), 0x9E3779B1);
  x = Math.imul(x ^ (x >>> 16), 0x85EBCA6B);
  x = Math.imul(x ^ (x >>> 13), 0xC2B2AE35);
  return (x ^ (x >>> 16)) >>> 0;
}

/** Fixed-capacity FIFO ring buffer; oldest entries are evicted at capacity. */
export function createRingBuffer(capacity = TELEMETRY_CAP) {
  let cap = Math.floor(Number(capacity));
  if (!Number.isFinite(cap) || cap < 1) cap = TELEMETRY_CAP;
  const buf = new Array(cap);
  let start = 0;
  let size = 0;
  return {
    capacity: cap,
    get length() { return size; },
    push(value) {
      const idx = (start + size) % cap;
      buf[idx] = value;
      if (size < cap) size++; else start = (start + 1) % cap;
    },
    at(i) {
      if (!Number.isInteger(i) || i < 0 || i >= size) return undefined;
      return buf[(start + i) % cap];
    },
    toArray() {
      const out = new Array(size);
      for (let i = 0; i < size; i++) out[i] = buf[(start + i) % cap];
      return out;
    },
    clear() { start = 0; size = 0; }
  };
}

/**
 * Merge the (optional, quiet-mode-safe) engine snapshot with policy context.
 * `mfr` is the manufactureRequired basis; the engine's own required/remaining
 * are intentionally ignored here.
 */
function buildContext(holder, context, itemId, mfr, maxTicks) {
  const engine = holder.engine;
  let snap = null;
  if (engine && typeof engine.snapshot === 'function') {
    try { snap = engine.snapshot(); } catch (_err) { snap = null; }
  }
  const snapshot = (snap && typeof snap === 'object') ? snap : {};
  const extra = isObj(context) ? context : {};
  const merged = { ...snapshot, ...extra };

  const stash = (merged.stash && typeof merged.stash === 'object') ? merged.stash : {};
  const slots = (merged.slots && typeof merged.slots === 'object') ? merged.slots : {};
  const slot = slots[itemId] || {};
  const starvation = (merged.starvation && typeof merged.starvation === 'object') ? merged.starvation : {};
  const starve = starvation[itemId] || {};
  const reqMap = isObj(mfr) ? mfr : {};
  const req = reqMap[itemId] != null ? reqMap[itemId] : 0;

  // The engine supplies `produced` as a scalar for this item; older contexts
  // may supply a per-item map. Accept both, then fall back to stash.
  let produced = merged.produced;
  if (produced != null && typeof produced === 'object') produced = produced[itemId];
  if (produced == null) produced = stash[itemId];

  let materials = merged.materialsAvailable;
  if (materials == null) materials = starve.material === true ? 0 : MAX_BATCH;

  const slotActive = slot.state === 'ACTIVE' || merged.slotActive === true || extra.slotActive === true;

  return {
    itemId,
    required: req,
    produced: produced != null ? produced : 0,
    stash,
    coins: merged.coins != null ? merged.coins : 0,
    tick: merged.tick != null ? merged.tick : 0,
    // Observation maxTicks is ALWAYS the full horizon, even in curriculum gens.
    maxTicks: merged.maxTicks != null ? merged.maxTicks : maxTicks,
    materialsAvailable: materials,
    slotActive
  };
}

function makeGenomePolicy(adapter, genome, itemId, mfr, holder, maxTicks) {
  if (!genome) return () => 0;
  return (context) => {
    const obs = makeObservation(buildContext(holder, context, itemId, mfr, maxTicks));
    return decodeBatch(adapter.activate(genome, obs));
  };
}

function makeGreedyPolicy(itemId, mfr, holder, maxTicks) {
  return (context) => greedyBatch(itemId, buildContext(holder, context, itemId, mfr, maxTicks), mfr);
}

function buildPolicies(adapter, candidate, k, mfr, frozen, holder, maxTicks) {
  const policyForItem = {};
  for (let j = 0; j < CRAFTABLE_IDS.length; j++) {
    const id = CRAFTABLE_IDS[j];
    if (j < k && frozen[id]) {
      policyForItem[id] = makeGenomePolicy(adapter, frozen[id], id, mfr, holder, maxTicks);
    } else if (j === k && candidate) {
      policyForItem[id] = makeGenomePolicy(adapter, candidate, id, mfr, holder, maxTicks);
    } else {
      // No candidate yet (null) -> greedy fallback, never a zero policy.
      policyForItem[id] = makeGreedyPolicy(id, mfr, holder, maxTicks);
    }
  }
  return policyForItem;
}

/** Poor but FINITE fitness for genomes skipped after the wall deadline. */
const BUDGET_EXCEEDED_FITNESS = -1e15;

/**
 * Step a DOM-free QUIET engine to completion (or rolloutMaxTicks/deadline),
 * then result(). The deadline is checked every tick (Date.now() is far cheaper
 * than a step) so a wall budget bounded in milliseconds cannot be overshot by a
 * long rollout. `rolloutMaxTicks` is the curriculum truncation; the observation
 * always sees the full maxTicks via buildContext.
 */
function runRollout(engine, rolloutMaxTicks, deadline, now) {
  let ticks = 0;
  while (ticks < rolloutMaxTicks) {
    if (engine && engine.done === true) break;
    if (deadline !== undefined && now() >= deadline) break;
    engine.step();
    ticks++;
    if (engine && engine.done === true) break;
    if (deadline !== undefined && now() >= deadline) break;
    if (typeof engine.snapshot === 'function') {
      let s = null;
      try { s = engine.snapshot(); } catch (_err) { s = null; }
      if (s && s.done === true) break;
    }
  }
  return typeof engine.result === 'function' ? engine.result() : {};
}

function makeTelemetryEvent(itemId, generation, genomeIndex, fit, result) {
  let perItem = {};
  try {
    perItem = JSON.parse(JSON.stringify((result.perItem && typeof result.perItem === 'object') ? result.perItem : {}));
  } catch (_err) { perItem = {}; }
  return {
    type: 'GENOME_RESULT',
    itemId,
    generation,
    genomeIndex,
    fitness: fit,
    feasible: result.feasible === true,
    completionTick: Number.isFinite(result.completionTick) ? result.completionTick : null,
    elapsedTicks: Number.isFinite(result.elapsedTicks) ? result.elapsedTicks : null,
    totalSpent: Number.isFinite(result.totalSpent) ? result.totalSpent : 0,
    perItem
  };
}

/**
 * Build the policy bundle for the main-thread DISPLAY run: every evolved
 * genome drives its item, and items without a genome fall back to the greedy
 * default. Reuses makeGenomePolicy / makeGreedyPolicy, so the observation,
 * batch decoding and activation paths are identical to training.
 *
 * `manufactureRequired` is the policy/observation basis. `required` is accepted
 * only for legacy UI-tooltip compatibility and is not read here. When no mfr
 * map is supplied the fallback is expandManufactureRequired (never the old
 * expandDemand().required fallthrough).
 *
 * @param {{genomes?:Object, config?:object, required?:Object,
 *          manufactureRequired?:Object, adapterFactory?:Function,
 *          seed?:number, maxTicks?:number}} opts
 * @returns {{policyForItem:Object, setEngine:Function}}
 */
export function createDisplayPolicyBundle({
  genomes,
  config,
  required,
  manufactureRequired,
  adapterFactory = createNeatAdapter,
  seed = 1,
  maxTicks
} = {}) {
  const cfg = (config && typeof config === 'object') ? config : {};
  const maxTicksNum = Number(maxTicks);
  const resolvedMaxTicks = Number.isFinite(maxTicksNum) && maxTicksNum > 0 ? maxTicksNum : DEFAULT_MAX_TICKS;

  const items = Array.isArray(cfg.items) ? cfg.items : [];
  const targets = (cfg.targets && typeof cfg.targets === 'object') ? cfg.targets : {};
  const starterStash = (cfg.initialStash && typeof cfg.initialStash === 'object') ? cfg.initialStash : {};
  const mfr = isObj(manufactureRequired) ? manufactureRequired
    : expandManufactureRequired(items, targets, starterStash, cfg.starterMode).manufactureRequired;

  // One shared, stateless adapter for every item (activate() is pure).
  const adapter = adapterFactory({
    inputs: OBSERVATION_SIZE,
    outputs: 1,
    seed: Number(seed) >>> 0,
    config: { populationSize: DEFAULT_POP }
  });
  const holder = { engine: null, maxTicks: resolvedMaxTicks };

  const policyForItem = {};
  for (const id of CRAFTABLE_IDS) {
    const raw = (genomes && typeof genomes === 'object') ? genomes[id] : null;
    let genome = null;
    if (raw) {
      try { genome = adapter.importGenome(raw); } catch (_err) { genome = null; }
    }
    policyForItem[id] = genome
      ? makeGenomePolicy(adapter, genome, id, mfr, holder, resolvedMaxTicks)
      : makeGreedyPolicy(id, mfr, holder, resolvedMaxTicks);
  }

  return {
    policyForItem,
    /** Bind the live display engine so buildContext sees stash/slots/starvation. */
    setEngine(engine) { holder.engine = engine; return engine; }
  };
}

/**
 * Evolve the 19 item populations in dependency topo order under an explicit
 * generation + wall-clock budget. Item-major order and per-item RNG lineage are
 * preserved; generation 0 is structurally reserved for every item.
 * @param {{adapterFactory?:Function, engineFactory:Function, config?:object,
 *          seed?:number, onProgress?:Function, maxGenerations?:number,
 *          maxWallMs?:number, now?:Function}} opts
 * @returns {Promise<{genomes:Object, telemetry:ReturnType<typeof createRingBuffer>}>}
 */
export async function trainAll(opts = {}) {
  const {
    adapterFactory = createNeatAdapter,
    engineFactory,
    config,
    seed = 1,
    onProgress = null,
    maxGenerations: genBudgetOpt,
    maxWallMs: wallBudgetOpt,
    now = Date.now
  } = opts;

  if (typeof adapterFactory !== 'function') throw new TypeError('trainAll: adapterFactory must be a function');
  if (typeof engineFactory !== 'function') throw new TypeError('trainAll: engineFactory must be a function');

  const cfg = (config && typeof config === 'object') ? config : {};
  const masterSeed = Number(seed) >>> 0;
  const maxTicksRaw = Number(cfg.maxTicks);
  const maxTicks = Number.isFinite(maxTicksRaw) && maxTicksRaw > 0 ? maxTicksRaw : DEFAULT_MAX_TICKS;

  const maxGenerations = intInRange(
    genBudgetOpt !== undefined ? genBudgetOpt : cfg.maxGenerations, DEFAULT_GENS, 1, 10000
  );
  const generations = Math.min(intInRange(cfg.generations, DEFAULT_GENS, 1, 10000), maxGenerations);
  const wallRaw = wallBudgetOpt !== undefined ? wallBudgetOpt : cfg.maxWallMs;
  const maxWallMs = (wallRaw === undefined || wallRaw === null || !Number.isFinite(Number(wallRaw)) || Number(wallRaw) < 0)
    ? DEFAULT_WALL_MS : Number(wallRaw);

  const adapterConfig = (cfg.adapterConfig && typeof cfg.adapterConfig === 'object') ? cfg.adapterConfig
    : ((cfg.neatConfig && typeof cfg.neatConfig === 'object') ? cfg.neatConfig : {});
  const popRaw = adapterConfig.populationSize !== undefined ? adapterConfig.populationSize : cfg.populationSize;
  const populationSize = intInRange(popRaw, DEFAULT_POP, 2, 500);
  const resolvedAdapterConfig = { ...adapterConfig, populationSize };

  const items = Array.isArray(cfg.items) ? cfg.items : [];
  const targets = (cfg.targets && typeof cfg.targets === 'object') ? cfg.targets : {};
  const starterStash = (cfg.initialStash && typeof cfg.initialStash === 'object') ? cfg.initialStash : {};
  const starterMode = cfg.starterMode;
  const required = expandDemand(items, targets, starterStash, starterMode).required;
  const mfr = expandManufactureRequired(items, targets, starterStash, starterMode).manufactureRequired;

  const telemetry = createRingBuffer(TELEMETRY_CAP);
  const genomes = {};
  const frozen = {};
  const holder = { engine: null, maxTicks };
  const startedAt = now();
  const deadline = startedAt + maxWallMs;
  const itemCount = CRAFTABLE_IDS.length;

  // Per-item wall slice; gen 0 always gets a reserved reserveMs so the shared
  // deadline can never skip an item's first population.
  const sliceMs = maxWallMs / itemCount;
  const reserveMs = sliceMs / generations;

  // Single shared batch-cap allocation (offline, pure) applied to every
  // training rollout via the SAME helper the display runner uses.
  const caps = planBatchCaps(items, mfr, maxTicks, MAX_BATCH).caps;

  /** Evaluate one genome once in the CURRENT frozen-upstream context (no telemetry). */
  function evaluateGenomeOnce(adapter, genome, k, itemSeed, rolloutDeadline, rolloutTicks) {
    const policyForItem = applyBatchCaps(
      buildPolicies(adapter, genome, k, mfr, frozen, holder, maxTicks), caps
    );
    const rng = createSeededRng(deriveSeed(itemSeed, 0x5bd1));
    const engine = engineFactory({ config: cfg, policyForItem, rng, required, quiet: true });
    holder.engine = engine;
    let result;
    try {
      result = runRollout(engine, rolloutTicks, rolloutDeadline, now);
    } finally {
      holder.engine = null;
    }
    return enrichResult(result, targets, mfr, maxTicks);
  }

  // A zero wall budget starts no population (DONE-safe), preserving the
  // deadline guard exactly as before.
  if (maxWallMs <= 0) return { genomes, telemetry };

  for (let k = 0; k < itemCount; k++) {
    const itemId = CRAFTABLE_IDS[k];
    const itemSeed = deriveSeed(masterSeed, k);
    const adapter = adapterFactory({ inputs: OBSERVATION_SIZE, outputs: 1, seed: itemSeed, config: resolvedAdapterConfig });
    const itemStart = now();
    let itemFeasible = false;

    for (let gen = 0; gen < generations; gen++) {
      // gen 0 is NEVER gated by the deadline; later generations are.
      if (gen > 0 && now() >= deadline) break;

      const rolloutDeadline = gen === 0
        ? itemStart + reserveMs
        : Math.min(deadline, itemStart + sliceMs);
      // Curriculum: only the ROLLOUT is truncated for the first two gens.
      const rolloutTicks = gen < CURRICULUM_GENS ? Math.min(CURRICULUM_MAX_TICKS, maxTicks) : maxTicks;

      let genomeIndex = 0;
      const generationNo = gen + 1;
      const summary = adapter.evaluatePopulation((candidate) => {
        const idx = genomeIndex++;
        // Skip this genome only once its own rollout slice is spent.
        if (now() >= rolloutDeadline) return BUDGET_EXCEEDED_FITNESS;
        const policyForItem = applyBatchCaps(
          buildPolicies(adapter, candidate, k, mfr, frozen, holder, maxTicks), caps
        );
        const rng = createSeededRng(deriveSeed(itemSeed, 0x5bd1));
        const engine = engineFactory({ config: cfg, policyForItem, rng, required, quiet: true });
        holder.engine = engine;
        let result;
        try {
          result = runRollout(engine, rolloutTicks, rolloutDeadline, now);
        } finally {
          holder.engine = null;
        }
        const enriched = enrichResult(result, targets, mfr, maxTicks);
        if (enriched.feasible === true) itemFeasible = true;
        // Telemetry keeps the SHIPPED fitness; selection uses bounded shaping.
        const shipped = fitness(enriched);
        const phase = enriched.feasible === true ? 'feasible' : 'pre';
        const event = makeTelemetryEvent(itemId, generationNo, idx, shipped, enriched);
        telemetry.push(event);
        // Forward from inside the evaluation callback so the worker can post it.
        if (typeof onProgress === 'function') onProgress(event);
        return trainingFitness(enriched, phase, mfr, maxTicks);
      });

      if (typeof onProgress === 'function') {
        onProgress({
          type: 'PROGRESS', itemId, itemIndex: k,
          generation: generationNo, generations,
          bestFitness: (summary && Number.isFinite(summary.bestFitness)) ? summary.bestFitness : null
        });
      }

      // Early stop ONLY after the current best is re-evaluated feasible in the
      // final upstream context (a feasible candidate alone is not enough).
      if (itemFeasible) {
        const bestNow = adapter.getBestGenome();
        const recheck = evaluateGenomeOnce(adapter, bestNow, k, itemSeed, rolloutDeadline, maxTicks);
        if (recheck.feasible === true) break;
      }
    }

    const best = adapter.getBestGenome(); // only after >=1 evaluatePopulation()
    // Authoritative re-evaluation of the stored best against the FINAL frozen
    // upstream policies before freezing it for downstream items.
    const finalCheck = evaluateGenomeOnce(adapter, best, k, itemSeed, deadline, maxTicks);
    itemFeasible = finalCheck.feasible === true;

    genomes[itemId] = best;
    frozen[itemId] = best ? adapter.cloneFrozen(best) : null;

    if (typeof onProgress === 'function') {
      onProgress({ type: 'BEST', itemId, itemIndex: k, genome: best ? adapter.exportGenome(best) : null });
    }

    // Let the worker's timers / DONE messages fire during long training runs.
    await new Promise((resolve) => setTimeout(resolve, 0));
  }

  return { genomes, telemetry };
}

export default trainAll;
