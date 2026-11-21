/**
 * train-worker.js — silent NEAT evaluation rollouts (plan Interfaces).
 *
 * Contract:
 *  - ES module worker: main creates it with { type: 'module' }.
 *  - Imports controller.js (trainAll), adapter.js and sim/engine.js.
 *  - Outbound messages ONLY: PROGRESS | BEST | GENOME_RESULT | DONE | ERROR.
 *  - BEST carries the validated best genome; DONE carries a bounded genomes
 *    map so main can drive a second display run with the TRAINED policies.
 *  - GENOME_RESULT carries total time, total coins and a per-item breakdown.
 *  - Per-(itemId|generation) caps, separate for PROGRESS and GENOME_RESULT.
 *
 * Config arrives from main via a trusted START postMessage (config.js relies on
 * a js-yaml UMD global that is not present inside a worker, so main loads it).
 *
 * Pure helpers are exported for src/tests/worker-bridge.test.js, which exercises
 * the exact production logic on the main thread.
 */

import * as controllerModule from '../neat/controller.js';
import { createNeatAdapter } from '../neat/adapter.js';
import { createEngine } from '../sim/engine.js';

const { trainAll } = controllerModule;

// Locked constants from the plan (kept local so the worker has no fetch/jsyaml dependency).
const MAX_TICKS = 8064;
const BIG = 100000;
const MAX_PER_GEN = 20;      // hard cap per (itemId|generation) per message kind
const GLOBAL_CAP = 50000;    // hard cap on total outbound messages
const STR_CAP = 120;
const MAX_GENOME_NODES = 128;
const MAX_GENOME_GENES = 512;
const MAX_DONE_GENOMES = 19;

// Mirror the controller's budget exports (names reported in the handoff report).
export const DEFAULT_GENS = Number.isFinite(controllerModule.DEFAULT_GENERATIONS)
  ? controllerModule.DEFAULT_GENERATIONS
  : Number.isFinite(controllerModule.DEFAULT_GENS) ? controllerModule.DEFAULT_GENS : 10;
export const DEFAULT_POP = Number.isFinite(controllerModule.DEFAULT_POPULATION_SIZE)
  ? controllerModule.DEFAULT_POPULATION_SIZE
  : Number.isFinite(controllerModule.DEFAULT_POP) ? controllerModule.DEFAULT_POP : 20;
export const DEFAULT_MAX_WALL_MS = Number.isFinite(controllerModule.DEFAULT_WALL_MS)
  ? controllerModule.DEFAULT_WALL_MS : 180000;
const PROGRESS_PING_MS = 1000;

let started = false;
let totalPosted = 0;
let genomeResultsEmitted = 0;
let lastBudgetPing = 0;
// Partial results collected from progress events so a hard-deadline DONE is
// still built from real genomes / telemetry if trainAll has not returned yet.
let partialGenomes = {};
let partialTelemetry = [];

/* ----------------------------- local sanitizers ----------------------------- */

function num(v, fallback, min, max) {
  if (typeof v !== 'number' || !Number.isFinite(v)) return fallback;
  return Math.min(max, Math.max(min, v));
}
function int(v, fallback, min, max) {
  return Math.round(num(v, fallback, min, max));
}
function str(v, fallback = '', cap = STR_CAP) {
  return typeof v === 'string' ? v.slice(0, cap) : fallback;
}
function bool(v) { return !!v; }
function isObj(v) { return !!v && typeof v === 'object' && !Array.isArray(v); }

function sanitizePerItem(pi) {
  const out = {};
  if (!isObj(pi)) return out;
  const keys = Object.keys(pi).slice(0, 23); // 23 game items
  for (const key of keys) {
    if (!/^[A-Za-z0-9_-]{1,40}$/.test(key)) continue;
    const row = pi[key];
    if (!isObj(row)) continue;
    out[key] = {
      coinsSpent: num(row.coinsSpent, 0, 0, 1e12),
      timeMinutes: num(row.timeMinutes, 0, 0, 1e9),
      busyTicks: int(row.busyTicks, 0, 0, MAX_TICKS),
      batches: int(row.batches, 0, 0, 1e6),
      units: int(row.units, 0, 0, 1e9),
      firstTick: int(row.firstTick, 0, 0, MAX_TICKS),
      lastTick: int(row.lastTick, 0, 0, MAX_TICKS)
    };
  }
  return out;
}

const NODE_TYPES = new Set(['input', 'output', 'hidden']);
const ACTIVATIONS = new Set(['LOGISTIC', 'TANH', 'RELU', 'IDENTITY']);

/** Bounded, structured-clone-safe genome (plain data, no methods). */
export function sanitizeGenome(g) {
  if (!isObj(g)) return null;
  const nodes = [];
  if (Array.isArray(g.nodes)) {
    for (const n of g.nodes.slice(0, MAX_GENOME_NODES)) {
      if (!isObj(n)) continue;
      nodes.push({
        id: int(n.id, 0, 0, 1e6),
        type: NODE_TYPES.has(n.type) ? n.type : 'hidden',
        activation: ACTIVATIONS.has(n.activation) ? n.activation : 'LOGISTIC',
        bias: num(n.bias, 0, -1e6, 1e6),
        // Hidden nodes use (from.depth + to.depth)/2 (e.g. 0.5). Rounding would
        // reorder activate()'s (depth,id) sort and change behaviour in replay.
        depth: num(n.depth, 0, 0, 1e6),
        frozen: bool(n.frozen)
      });
    }
  }
  const genes = [];
  if (Array.isArray(g.genes)) {
    for (const gn of g.genes.slice(0, MAX_GENOME_GENES)) {
      if (!isObj(gn)) continue;
      genes.push({
        innovation: int(gn.innovation, 0, 0, 1e9),
        from: int(gn.from, 0, 0, 1e6),
        to: int(gn.to, 0, 0, 1e6),
        weight: num(gn.weight, 0, -1e6, 1e6),
        enabled: gn.enabled !== false,
        frozen: bool(gn.frozen)
      });
    }
  }
  return {
    nodes,
    genes,
    frozen: bool(g.frozen),
    fitness: num(g.fitness, 0, -1e15, 1e15),
    adjustedFitness: num(g.adjustedFitness, 0, -1e15, 1e15),
    speciesId: int(g.speciesId, 0, 0, 1e6)
  };
}

/** Bounded itemId -> sanitized genome map for DONE. */
export function sanitizeGenomes(map) {
  const out = {};
  if (!isObj(map)) return out;
  for (const key of Object.keys(map).slice(0, MAX_DONE_GENOMES)) {
    if (!/^[a-z][a-z0-9_]{0,39}$/.test(key)) continue; // 19 known lowercase item ids
    const g = sanitizeGenome(map[key]);
    if (g) out[key] = g;
  }
  return out;
}

/* --------------------------- pure result helpers --------------------------- */

/**
 * Extract the controller's telemetry as a plain array. The controller returns a
 * bounded ring-buffer OBJECT with toArray(); older producers may return an array.
 */
export function collectTelemetry(result) {
  const t = result && result.telemetry;
  if (!t) return [];
  if (typeof t.toArray === 'function') {
    const arr = t.toArray();
    return Array.isArray(arr) ? arr : [];
  }
  return Array.isArray(t) ? t.slice() : [];
}

export function isGenomeResultEvent(ev) {
  return isObj(ev) && !!(ev.type === 'GENOME_RESULT' || (ev.perItem && typeof ev.fitness === 'number'));
}

export function selectBestResult(events) {
  const list = Array.isArray(events) ? events : collectTelemetry(events);
  let best = null;
  let bestScore = -Infinity;
  for (const ev of list) {
    if (!isGenomeResultEvent(ev) || ev.feasible !== true) continue;
    const tick = Number.isFinite(ev.completionTick) ? ev.completionTick : MAX_TICKS;
    const spend = Number.isFinite(ev.totalSpent) ? ev.totalSpent : 0;
    const score = -(spend * BIG + tick);
    if (score > bestScore) { best = ev; bestScore = score; }
  }
  if (best) return best;

  let fallback = null;
  let fallbackFitness = -Infinity;
  for (const ev of list) {
    if (!isGenomeResultEvent(ev) || !Number.isFinite(ev.fitness)) continue;
    if (ev.fitness > fallbackFitness) { fallback = ev; fallbackFitness = ev.fitness; }
  }
  return fallback;
}

export function buildDoneMessage(result) {
  const events = collectTelemetry(result);
  const best = selectBestResult(events);
  const genomes = sanitizeGenomes(result && result.genomes);
  return {
    type: 'DONE',
    feasible: best ? best.feasible === true : false,
    completionTick: best ? int(best.completionTick, MAX_TICKS, 0, MAX_TICKS) : MAX_TICKS,
    totalSpent: best ? num(best.totalSpent, 0, 0, 1e12) : 0,
    genomeCount: Object.keys(genomes).length,
    telemetryCount: int(events.length, 0, 0, 5000),
    // Bounded genomes map: main uses these to drive the trained display run.
    genomes
  };
}

export function sanitizeGenomeResult(ev) {
  if (!isGenomeResultEvent(ev)) return null;
  return {
    type: 'GENOME_RESULT',
    itemId: str(ev.itemId, 'unknown', 40),
    generation: int(ev.generation !== undefined ? ev.generation : ev.gen, 0, 0, 100000),
    genomeIndex: int(ev.genomeIndex, 0, 0, 1e6),
    fitness: num(ev.fitness, -1e12, -1e15, 1e15),
    feasible: bool(ev.feasible),
    completionTick: int(ev.completionTick, MAX_TICKS, 0, MAX_TICKS),
    elapsedTicks: int(ev.elapsedTicks, MAX_TICKS, 0, MAX_TICKS),
    totalSpent: num(ev.totalSpent, 0, 0, 1e12),
    perItem: sanitizePerItem(ev.perItem)
  };
}

export function createCapState() {
  return {
    PROGRESS: { key: null, count: 0 },
    GENOME_RESULT: { key: null, count: 0 }
  };
}

/** Cap key includes the item, because all 19 items restart at generation 1. */
function withinCap(kind, itemId, gen, caps) {
  const state = caps && caps[kind];
  if (!state) return false;
  const key = `${str(itemId, 'unknown', 40)}|${int(gen, 0, 0, 100000)}`;
  if (state.key !== key) { state.key = key; state.count = 0; }
  if (state.count >= MAX_PER_GEN) return false;
  state.count += 1;
  return true;
}

export function buildReplayMessages(result, maxPerGen = MAX_PER_GEN) {
  const events = collectTelemetry(result);
  const caps = createCapState();
  const out = [];
  for (const ev of events) {
    if (!isGenomeResultEvent(ev)) continue;
    const gen = int(ev.generation !== undefined ? ev.generation : ev.gen, 0, 0, 100000);
    if (!withinCap('GENOME_RESULT', ev.itemId, gen, caps)) continue;
    const msg = sanitizeGenomeResult(ev);
    if (msg) out.push(msg);
  }
  return out;
}

export function buildTrainOptions(data) {
  const cfg = (data && isObj(data.config)) ? data.config : null;
  const opts = (data && isObj(data.options)) ? data.options : {};
  return {
    config: cfg,
    seed: int(data && data.seed, 1, 0, 4294967295),
    maxGenerations: int(opts.generations, DEFAULT_GENS, 1, 10000),
    maxWallMs: int(opts.maxWallMs, DEFAULT_MAX_WALL_MS, 1000, 15 * 60 * 1000),
    adapterConfig: { populationSize: int(opts.populationSize, DEFAULT_POP, 2, 500) },
    quiet: true
  };
}

/**
 * Hard-deadline guard. Exactly one DONE is posted: either by finish() on the
 * normal path or by the armed timer when the wall budget expires. Timer and
 * clock are injectable so worker-bridge.test.js can prove the behaviour with a
 * stubbed clock instead of waiting in real time.
 */
export function createDeadline({
  maxWallMs = DEFAULT_MAX_WALL_MS,
  sink = (msg) => post(msg),
  makeTimer = (cb, ms) => setTimeout(cb, ms),
  cancelTimer = (id) => clearTimeout(id),
  getPartial = () => null
} = {}) {
  let finished = false;
  let timer = null;

  function finish(result) {
    if (finished) return false; // no duplicate DONE
    finished = true;
    if (timer !== null) { cancelTimer(timer); timer = null; }
    sink(buildDoneMessage(result !== undefined ? result : getPartial()));
    return true;
  }

  function arm() {
    if (timer !== null) cancelTimer(timer);
    timer = makeTimer(() => {
      const firedId = timer;
      timer = null;
      if (finished) return;
      finished = true;
      if (firedId !== null) cancelTimer(firedId); // cleanup: no re-armed timer left behind
      sink(buildDoneMessage(getPartial()));
    }, maxWallMs);
    return timer;
  }

  return {
    arm,
    finish,
    isFinished: () => finished,
    isArmed: () => timer !== null
  };
}

/* ------------------------ message builders & dispatch ---------------------- */

function buildProgressMessage(ev, gen) {
  return {
    type: 'PROGRESS',
    itemId: str(ev.itemId, 'unknown', 40),
    generation: int(gen, 0, 0, 100000),
    completed: int(ev.completed, 0, 0, 1e9),
    total: int(ev.total, 0, 0, 1e9),
    phase: str(ev.phase, '', 40),
    message: str(ev.message, '', STR_CAP)
  };
}

function buildBestMessage(ev, gen) {
  return {
    type: 'BEST',
    itemId: str(ev.itemId, 'unknown', 40),
    generation: int(gen, 0, 0, 100000),
    fitness: num(ev.fitness !== undefined ? ev.fitness : ev.bestFitness, 0, -1e15, 1e15),
    feasible: bool(ev.feasible),
    // Validated genome so main can replay the trained policy in a display run.
    genome: sanitizeGenome(ev.genome)
  };
}

/**
 * Dispatch a controller progress event to a sink, keyed on explicit `type`
 * FIRST. The controller's PROGRESS events also carry bestFitness, so any
 * heuristic based on bestFitness would misroute them as BEST.
 */
export function dispatchProgress(ev, sink, caps) {
  if (!isObj(ev)) return null;
  const gen = int(ev.generation !== undefined ? ev.generation : ev.gen, 0, 0, 100000);
  const kind = str(ev.type, '', 40);
  let msg = null;

  if (kind === 'PROGRESS') {
    if (withinCap('PROGRESS', ev.itemId, gen, caps)) { msg = buildProgressMessage(ev, gen); sink(msg); }
  } else if (kind === 'GENOME_RESULT') {
    if (withinCap('GENOME_RESULT', ev.itemId, gen, caps)) { msg = sanitizeGenomeResult(ev); if (msg) sink(msg); }
  } else if (kind === 'BEST') {
    msg = buildBestMessage(ev, gen); sink(msg);
  } else if (isGenomeResultEvent(ev)) {
    if (withinCap('GENOME_RESULT', ev.itemId, gen, caps)) { msg = sanitizeGenomeResult(ev); if (msg) sink(msg); }
  } else if (withinCap('PROGRESS', ev.itemId, gen, caps)) {
    msg = buildProgressMessage(ev, gen); sink(msg);
  }
  return msg;
}

/* ------------------------------- emitting ---------------------------------- */

const caps = createCapState();

function post(msg) {
  if (totalPosted >= GLOBAL_CAP) return false;
  totalPosted += 1;
  self.postMessage(msg);
  return true;
}

/** Pure budget ping builder (elapsed / budget / remaining are explicit). */
export function buildBudgetProgressMessage(startedAt, maxWallMs, generation, now = Date.now()) {
  const elapsedMs = Math.max(0, now - startedAt);
  return {
    type: 'PROGRESS',
    itemId: 'budget',
    generation: int(generation, 0, 0, 100000),
    completed: int(elapsedMs, 0, 0, 1e9),
    total: int(maxWallMs, 0, 0, 1e9),
    // Explicit deadline-adherence fields for the UI / log.
    elapsedMs: int(elapsedMs, 0, 0, 1e9),
    budgetMs: int(maxWallMs, 0, 0, 1e9),
    remainingMs: int(Math.max(0, maxWallMs - elapsedMs), 0, 0, 1e9),
    phase: 'budget',
    message: `elapsed ${elapsedMs}ms remaining ${Math.max(0, maxWallMs - elapsedMs)}ms`
  };
}

function emitBudgetProgress(startedAt, maxWallMs, generation) {
  const now = Date.now();
  if (now - lastBudgetPing < PROGRESS_PING_MS) return;
  lastBudgetPing = now;
  post(buildBudgetProgressMessage(startedAt, maxWallMs, generation, now));
}

function emitError(code, message) {
  post({ type: 'ERROR', code: str(code, 'TRAIN_ERROR', 40), message: str(message, '', STR_CAP) });
}

function onProgress(ev) {
  const msg = dispatchProgress(ev, post, caps);
  if (!msg) return;
  if (msg.type === 'GENOME_RESULT') {
    genomeResultsEmitted += 1;
    // Partial telemetry for a deadline-triggered DONE (bounded ring).
    partialTelemetry.push(msg);
    if (partialTelemetry.length > 5000) partialTelemetry.shift();
  } else if (msg.type === 'BEST' && msg.genome) {
    partialGenomes[msg.itemId] = msg.genome;
  }
}

/* --------------------------------- start ---------------------------------- */

async function start(data) {
  if (started) return;
  const config = data && data.config;
  if (!config || !isObj(config) || !Array.isArray(config.items) || !config.targets) {
    emitError('BAD_CONFIG', 'START message missing valid config');
    post(buildDoneMessage(null)); // guaranteed terminal message
    return;
  }
  started = true;
  partialGenomes = {};
  partialTelemetry = [];

  const trainOpts = buildTrainOptions(data);
  // The controller reads adapterConfig / generations from `config`, so merge
  // the sanitized control values into it before calling trainAll (F3).
  const runConfig = Object.assign({}, config, {
    populationSize: trainOpts.adapterConfig.populationSize,
    adapterConfig: trainOpts.adapterConfig,
    generations: trainOpts.maxGenerations,
    maxGenerations: trainOpts.maxGenerations,
    maxWallMs: trainOpts.maxWallMs
  });
  const startedAt = Date.now();
  let generation = 0;

  dispatchProgress({ type: 'PROGRESS', phase: 'start', message: 'training started', completed: 0, total: trainOpts.maxWallMs }, post, caps);

  const budgetTimer = setInterval(() => emitBudgetProgress(startedAt, trainOpts.maxWallMs, generation), PROGRESS_PING_MS);
  // Hard deadline: if trainAll has not returned by maxWallMs, post DONE from
  // whatever genomes / telemetry have already been collected.
  const deadline = createDeadline({
    maxWallMs: trainOpts.maxWallMs,
    sink: post,
    getPartial: () => ({ genomes: partialGenomes, telemetry: partialTelemetry })
  });
  deadline.arm();
  let result = null;

  try {
    const maybe = trainAll({
      adapterFactory: (opts) => createNeatAdapter(opts),
      engineFactory: (opts) => createEngine(opts),
      config: runConfig,
      seed: trainOpts.seed,
      maxGenerations: trainOpts.maxGenerations,
      maxWallMs: trainOpts.maxWallMs,
      quiet: trainOpts.quiet,
      onProgress: (ev) => {
        if (ev && Number.isFinite(ev.generation)) generation = ev.generation;
        onProgress(ev);
        emitBudgetProgress(startedAt, trainOpts.maxWallMs, generation);
      }
    });
    result = (maybe && typeof maybe.then === 'function') ? await maybe : maybe;

    if (genomeResultsEmitted === 0) {
      for (const msg of buildReplayMessages(result)) post(msg);
    }
  } catch (err) {
    emitError('TRAIN_FAILED', (err && err.message) ? err.message : String(err));
  } finally {
    clearInterval(budgetTimer);
    // finish() clears the deadline timer and is a no-op if it already fired.
    deadline.finish(result);
  }
}

/* Only bind the real worker global. Importing this module on the main thread
 * (as worker-bridge.test.js does) must NOT install a window message listener. */
// A module worker has no `document` and owns postMessage; `WorkerGlobalScope`
// is not exposed as a constructor in every browser, so detect by capability.
const isWorkerScope = typeof document === 'undefined' &&
  typeof self !== 'undefined' && typeof self.postMessage === 'function';
if (isWorkerScope) {
  self.onmessage = (event) => {
    const data = event && event.data;
    if (!data || typeof data !== 'object') return;
    if (data.type === 'START') void start(data);
  };
}
