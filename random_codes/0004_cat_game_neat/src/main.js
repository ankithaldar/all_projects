/**
 * main.js — bootstrap, UI wiring, validated worker bridge and main-thread
 * display run (plan Interfaces / Security Controls).
 *
 * Security invariants implemented here:
 *  - Worker created with { type: 'module' }; ONLY worker.onmessage is bound.
 *    There is no window.addEventListener('message') anywhere in this file.
 *  - Every inbound worker message is validated by the single owner
 *    validate.js::validateWorkerMessage before use; invalid payloads are
 *    rejected and logged, never applied.
 *  - UI control values are validated by validate.js::validateControl.
 *  - All dynamic text is produced by the UI modules via textContent.
 *  - file:// => embedded config only, training disabled, no worker started.
 */

import { loadConfig, deriveConstants, applyOverrides } from './config.js';
import { validateWorkerMessage, validateControl, validateInitialState, ERROR_CODES } from './validate.js';
import { createEngine } from './sim/engine.js';
import { expandDemand, expandManufactureRequired } from './sim/demand.js';
import { MAX_BATCH } from './sim/items.js';
import { createSeededRng } from './neat/inhouse-neat.js';
import { createNeatAdapter } from './neat/adapter.js';
import * as controllerModule from './neat/controller.js';
import { createDag } from './ui/dag.js';
import { createHud } from './ui/hud.js';
import { createLogger } from './ui/logger.js';
import { createControls } from './ui/controls.js';
import { createResults } from './ui/results.js';
import { createInitialStatePanel, readInitialState } from './ui/initial-state.js';
import { capString, TICK_MINUTES } from './ui/format.js';
import { createDiagnostics } from './ui/diagnostics.js';
import { DEFAULT_GENS, DEFAULT_POP } from './worker/train-worker.js';

const C = deriveConstants() || {};
const MAX_TICKS = Number.isFinite(C.MAX_TICKS) ? C.MAX_TICKS : 8064;
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const isFileProtocol = (typeof location !== 'undefined' && location.protocol === 'file:');

let config = null;
// Pristine preset snapshot. startAll refreshes it on every load; resetAll
// restores it verbatim. Never mutated by applyOverrides (which shallow-copies).
let baseConfig = null;
let initialState = null; // the editable Initial state panel
let simClock = null;     // tick-driven simulated-time clock in the header
let seed = 1;
let speed = '1x';
let worker = null;
let running = false;
let paused = false;
let starting = false;
let engine = null;
let runToken = 0;
// Best genome per item, collected from validated BEST / DONE worker messages.
let bestGenomes = {};

let dag = null;
let hud = null;
let logger = null;
let controls = null;
let results = null;
let diagnostics = null;        // main-thread-only diagnostics view (optional)
let solverDiagnostics = null;  // last computed main-thread diagnostics object

/* ----------------------------- UI call wrappers ---------------------------- */

// The UI wrappers are guarded so the exported test seams (startAll/resetAll/
// onPresetChange) can run in tests.html, which has no #dag-svg and no bootstrap.
function logEvent(ev) { if (logger) logger.append(ev); }
function updateDag(snap) { if (dag) dag.update(snap); }
function updateHud(snap) { if (hud) hud.update(snap); }
function setStatus(s) { if (controls) controls.setStatus(s); }
function showResult(r) { if (results) results.show(r); }
function showGenome(ev) { if (results) results.showGenomeTelemetry(ev); }

/** Stable rejected-input log: only the constant, never the raw input. */
function logInvalidInput() {
  const code = ERROR_CODES.CONFIG_INPUT_INVALID || 'CONFIG_INPUT_INVALID';
  logEvent({ tick: 0, type: code, item: '', delta: 0, reason: code });
}

/* ----------------------- simulated-time clock (header) --------------------- */

// Stable DOM id kept to avoid markup churn; the visible label is simulated time.
export const SIM_CLOCK_ID = 'wall-clock';
export const SIM_TICK_MINUTES = TICK_MINUTES; // one engine tick === 5 simulated minutes

/** Normalize any tick value to a non-negative integer (0 for hostile input). */
function normalizeTick(tick) {
  return Number.isFinite(tick) && tick > 0 ? Math.floor(tick) : 0;
}

/**
 * Format an engine tick count as total SIMULATED time: "Sim time: N min" where
 * N = tick * TICK_MINUTES. There is no wall-clock / Date.now arithmetic here.
 */
export function formatSimClock(tick, tickMinutes = SIM_TICK_MINUTES) {
  const per = Number.isFinite(tickMinutes) && tickMinutes > 0 ? tickMinutes : SIM_TICK_MINUTES;
  return 'Sim time: ' + (normalizeTick(tick) * per) + ' min';
}

/**
 * Tick-driven clock controller. No timer and no Date.now: the display loop
 * calls update(tick) once per snapshot. Tests inject tick values directly, so
 * no real waiting or virtual time is needed. The host element is written only
 * via textContent + capString(...,120); aria is kept in sync as plain text.
 */
export function createSimClock(deps = {}, doc = (typeof document !== 'undefined' ? document : null)) {
  const per = Number.isFinite(deps.tickMinutes) && deps.tickMinutes > 0
    ? deps.tickMinutes : SIM_TICK_MINUTES;
  const el = doc && typeof doc.getElementById === 'function'
    ? doc.getElementById(SIM_CLOCK_ID) : null;
  let currentTick = 0;
  let frozen = false;

  function render() {
    if (!el) return;
    const minutes = currentTick * per;
    if (typeof el.setAttribute === 'function') {
      el.setAttribute('aria-label', 'Simulated time: ' + minutes + ' minutes');
    }
    el.textContent = capString(formatSimClock(currentTick, per), 120);
  }
  render(); // establish the initial "Sim time: 0 min" text immediately

  return {
    /** Apply one display snapshot's tick. Ignored once the run is frozen. */
    update(tick) {
      if (frozen) return;
      currentTick = normalizeTick(tick);
      render();
    },
    /** Reset to tick 0 and unfreeze (Start/Apply and Reset paths). */
    reset() {
      currentTick = 0;
      frozen = false;
      render();
    },
    /** Freeze on completion; optionally pin the final tick from result().completionTick. */
    freeze(tick) {
      if (tick !== undefined && tick !== null && Number.isFinite(tick) && tick >= 0) {
        currentTick = Math.floor(tick);
      }
      frozen = true;
      render();
    },
    getMinutes: () => currentTick * per,
    getTick: () => currentTick,
    isFrozen: () => frozen,
    render
  };
}

/** Apply a single display snapshot to a sim clock (exported test seam). */
export function updateSimClockFromSnapshot(snap, clock = simClock) {
  if (!clock || !snap) return;
  clock.update(normalizeTick(snap.tick));
}

/* --------------------------- control value reading ------------------------- */

function normalizeControlValue(name, raw) {
  if (name === 'seed' || name === 'pop-size' || name === 'generations') {
    const n = Number(raw);
    return Number.isFinite(n) ? Math.round(n) : raw; // validator rejects NaN
  }
  return typeof raw === 'string' ? raw.trim() : raw;
}

/** Read one DOM control through the single-owner validator; fall back on rejection. */
function readValidated(name, fallback) {
  const el = $(name);
  if (!el || el.value === undefined || el.value === null || el.value === '') return fallback;
  try {
    return validateControl(name, normalizeControlValue(name, el.value));
  } catch (_e) {
    return fallback;
  }
}

/** preset is validated only against reuse|empty (validate.js is the allowlist owner). */
function readControls() {
  return {
    speed: readValidated('speed', '1x'),
    seed: (readValidated('seed', 1) >>> 0),
    preset: readValidated('preset', 'empty'),
    // Controller requires population >= 2; validate.js allows 1, so clamp here.
    populationSize: Math.max(2, Math.min(500, readValidated('pop-size', DEFAULT_POP))),
    generations: readValidated('generations', DEFAULT_GENS)
  };
}

/* ------------------------ worker message validation ------------------------ */

/**
 * Pure collector for trained genomes carried by validated BEST / DONE messages.
 * Kept separate from the validator so it can be tested directly even while
 * validate.js is being extended to accept genome payloads.
 */
export function collectGenomePayload(msg, store = {}) {
  if (!msg || typeof msg !== 'object') return store;
  if (msg.type === 'BEST' && typeof msg.itemId === 'string' && msg.genome && typeof msg.genome === 'object') {
    store[msg.itemId] = msg.genome;
  }
  if (msg.type === 'DONE' && msg.genomes && typeof msg.genomes === 'object') {
    for (const id of Object.keys(msg.genomes)) {
      const g = msg.genomes[id];
      if (g && typeof g === 'object') store[id] = g;
    }
  }
  return store;
}

export function rejectWorkerMessage() {
  return {
    tick: 0, type: 'WORKER_MSG_REJECTED', item: '', delta: 0,
    reason: ERROR_CODES.MESSAGE_INVALID || 'MESSAGE_INVALID'
  };
}

/**
 * LIVE runtime message path. Every inbound worker message goes through the
 * single-owner validator here before any handler is called. Exported so
 * worker-bridge.test.js can exercise the real dispatch path (F4), not just the
 * validator library in isolation.
 */
export function dispatchWorkerMessage(msg, h) {
  let cleaned;
  try {
    cleaned = validateWorkerMessage(msg);
  } catch (_e) {
    h.logEvent(rejectWorkerMessage());
    h.reject();
    return;
  }
  if (!cleaned) { h.logEvent(rejectWorkerMessage()); h.reject(); return; }

  switch (cleaned.type) {
    case 'PROGRESS':
      h.setStatus(`training ${cleaned.itemId} gen ${cleaned.generation}`);
      h.logEvent({ tick: 0, type: 'TRAIN_PROGRESS', item: cleaned.itemId, delta: 0, reason: cleaned.message || cleaned.phase });
      break;
    case 'BEST':
      h.onBest(cleaned);
      h.setStatus(`best ${cleaned.itemId}: ${cleaned.fitness}`);
      h.logEvent({ tick: 0, type: 'TRAIN_BEST', item: cleaned.itemId, delta: cleaned.fitness, reason: cleaned.feasible ? 'feasible' : 'infeasible' });
      break;
    case 'GENOME_RESULT':
      h.showGenome(cleaned);
      h.logEvent({
        tick: 0, type: 'GENOME_RESULT', item: cleaned.itemId, delta: cleaned.totalSpent,
        reason: `gen ${cleaned.generation} spend=${cleaned.totalSpent} ticks=${cleaned.elapsedTicks}`
      });
      break;
    case 'DONE':
      h.onDone(cleaned);
      h.setStatus(cleaned.feasible ? 'training complete (feasible)' : 'training complete (infeasible)');
      h.logEvent({ tick: 0, type: 'TRAIN_DONE', item: '', delta: cleaned.totalSpent, reason: `genomes=${cleaned.genomeCount}` });
      break;
    case 'ERROR':
      // Stable code to the UI log; no stack is ever sent through messages.
      h.setStatus(`training error: ${cleaned.code}`);
      h.logEvent({ tick: 0, type: 'TRAIN_ERROR', item: '', delta: 0, reason: `${cleaned.code}: ${cleaned.message}` });
      console.error('[train-worker]', cleaned.code, cleaned.message);
      break;
    default:
      h.logEvent(rejectWorkerMessage());
      h.reject();
      break;
  }
}

function onWorkerMessage(event) {
  dispatchWorkerMessage(event && event.data, {
    logEvent, setStatus, showGenome,
    onBest: (msg) => { collectGenomePayload(msg, bestGenomes); },
    onDone: (msg) => {
      collectGenomePayload(msg, bestGenomes);
      if (Object.keys(bestGenomes).length === 0) {
        logEvent({ tick: 0, type: 'TRAIN_NO_GENOMES', item: '', delta: 0, reason: 'PREVIEW ONLY — no trained genomes were delivered' });
        return;
      }
      void runTrainedDisplay();
    },
    reject: () => { setStatus('invalid worker message'); }
  });
}

/* -------------------------- solver helper loading -------------------------- */

/**
 * The batch planner / policy-cap helpers are authored by the NEAT worker in
 * src/neat/*.js. They are imported dynamically so this module stays loadable
 * while those files land; until then the display degrades to the shipped
 * uncapped behaviour. This is still a SINGLE implementation of applyBatchCaps
 * (the controller imports the same module statically) - no local duplicate.
 */
let solverHelpersPromise = null;
function loadSolverHelpers() {
  if (solverHelpersPromise === null) {
    solverHelpersPromise = Promise.all([
      import('./neat/policy-caps.js'),
      import('./neat/batch-planner.js'),
      import('./neat/priority.js')
    ]).then(([caps, planner, priority]) => ({
      applyBatchCaps: caps && caps.applyBatchCaps,
      planBatchCaps: planner && planner.planBatchCaps,
      computeCriticality: (priority && priority.computeCriticality)
        || (planner && planner.computeCriticality)
    })).catch(() => null);
  }
  return solverHelpersPromise;
}

/**
 * Pure builder for the START message. The worker message contract is FROZEN:
 * the only fields are type/config/seed/options. Solver diagnostics are
 * computed on the main thread and are NEVER added to a worker message.
 */
export function buildStartMessage(cfg, seedValue, options) {
  const opts = (options && typeof options === 'object') ? options : {};
  return {
    type: 'START',
    config: cfg,
    seed: seedValue,
    options: { populationSize: opts.populationSize, generations: opts.generations }
  };
}

/** Exposed read-only seams for the integration tests (no bootstrap required). */
export function getConfig() { return config; }
export function getBaseConfig() { return baseConfig; }
export function getRunToken() { return runToken; }

/**
 * THE propagation seam. Commit a freshly loaded base config plus validated
 * user overrides to the MODULE-LEVEL `config`, then hand that exact reference
 * to START, the engine and diagnostics. There is no separate local config
 * path. `hooks` only inject observers/options for tests; production startAll
 * supplies none. validate.js gates the overrides before any consumer runs.
 */
export function applyOverridesAndStart(base, overrides, hooks = {}) {
  validateInitialState(overrides);          // mandatory single-owner gate
  baseConfig = base;                        // pristine, before applying overrides
  config = applyOverrides(base, overrides); // module config, assigned by reference
  const startMessage = buildStartMessage(
    config,
    hooks.seed === undefined ? seed : hooks.seed,
    hooks.options || {}
  );
  if (typeof hooks.onCreateEngine === 'function') hooks.onCreateEngine(config);
  if (typeof hooks.onStart === 'function') hooks.onStart(startMessage, config);
  return { config, startMessage };
}

/**
 * Compute criticality / batch-cap diagnostics on the MAIN THREAD from the
 * validated config + targets. `helpers` is injectable for tests; by default
 * the shared planner/priority modules are loaded lazily. The returned maps are
 * plain (bounded upstream by the UI sanitizer); no value is sent to a worker.
 */
export async function buildSolverDiagnostics(config, helpers = null) {
  const h = helpers || await loadSolverHelpers();
  const items = (config && Array.isArray(config.items)) ? config.items : [];
  const targets = (config && config.targets && typeof config.targets === 'object') ? config.targets : {};
  const starterStash = (config && config.initialStash && typeof config.initialStash === 'object')
    ? config.initialStash : {};
  const starterMode = (config && config.starterMode) || 'PREDELIVERED_RESERVED';

  let manufactureRequired = {};
  try {
    manufactureRequired = expandManufactureRequired(items, targets, starterStash, starterMode)
      .manufactureRequired || {};
  } catch (_e) { manufactureRequired = {}; }

  let rawCaps = {};
  let estimatedTicks = 0;
  let estimatedCost = 0;
  if (h && typeof h.planBatchCaps === 'function') {
    try {
      const planned = h.planBatchCaps(items, manufactureRequired, MAX_TICKS, MAX_BATCH);
      if (planned && typeof planned === 'object') {
        rawCaps = (planned.caps && typeof planned.caps === 'object') ? planned.caps : {};
        estimatedTicks = Number(planned.estimatedTicks) || 0;
        estimatedCost = Number(planned.estimatedCost) || 0;
      }
    } catch (_e) { /* best-effort diagnostics only */ }
  }

  let rawCriticality = {};
  if (h && typeof h.computeCriticality === 'function') {
    try {
      rawCriticality = h.computeCriticality(items, targets) || {};
    } catch (_e) { rawCriticality = {}; }
  }

  return { criticality: rawCriticality, caps: rawCaps, estimatedTicks, estimatedCost };
}

async function refreshDiagnostics() {
  if (!config) return;
  try {
    solverDiagnostics = await buildSolverDiagnostics(config);
    const host = typeof document !== 'undefined' ? document.getElementById('solver-diagnostics') : null;
    if (diagnostics && host) diagnostics.render(solverDiagnostics);
  } catch (_e) { /* diagnostics are best-effort; never break the display */ }
}

/* -------------------------------- worker start ---------------------------- */

function startWorkerTraining(cfg) {
  if (isFileProtocol) return; // file:// => no worker, no training
  if (worker) {
    worker.terminate();
    worker = null;
  }
  worker = new Worker(new URL('./worker/train-worker.js', import.meta.url), { type: 'module' });
  // ONLY worker.onmessage is bound. No window message listener is installed.
  worker.onmessage = onWorkerMessage;
  worker.postMessage(buildStartMessage(config, seed, cfg));
}

/* ------------------------------ display run ------------------------------- */

function delayForSpeed(sp) {
  return sp === 'fast' ? 0 : sp === '10x' ? 50 : 500; // 1x = 500ms
}

/** Greedy fallback used only for the idle preview / initial snapshot. */
function greedyPolicyMap() {
  const map = {};
  for (const id of (controllerModule.CRAFTABLE_IDS || [])) map[id] = () => MAX_BATCH;
  return map;
}

export function buildEngine(policyForItem = () => MAX_BATCH) {
  return createEngine({ config, policyForItem, rng: createSeededRng(seed) });
}

function extractEvents(stepResult) {
  if (Array.isArray(stepResult)) return stepResult;
  if (stepResult && typeof stepResult === 'object' && !Array.isArray(stepResult)) {
    if (Array.isArray(stepResult.events)) return stepResult.events;
    return [stepResult];
  }
  return [];
}

export async function runDisplay(token, opts = {}) {
  const policyForItem = opts.policyForItem || (() => MAX_BATCH);
  const setEngine = typeof opts.setEngine === 'function' ? opts.setEngine : () => {};
  const showResults = !!opts.showResults;
  const label = opts.label || 'display';
  const clock = opts.simClock || simClock;
  const nap = typeof opts.sleep === 'function' ? opts.sleep : sleep;
  const build = typeof opts.buildEngine === 'function' ? opts.buildEngine : buildEngine;

  engine = build(policyForItem);
  running = true;
  paused = false;
  setStatus(`${label} running (${speed})`);
  const delay = delayForSpeed(speed);
  let lastSnapTick = 0;

  try {
    for (let i = 0; i <= MAX_TICKS + 8 && running && token === runToken; i += 1) {
      if (paused) { await nap(50); continue; }

      // Bind the live engine so controller policies see current stash/slots.
      setEngine(engine);

      const events = extractEvents(engine.step());
      if (events.length) logEvent(events);

      const snap = engine.snapshot();
      updateDag(snap);
      updateHud(snap);
      // Sim time is derived from the engine tick on EVERY display snapshot.
      updateSimClockFromSnapshot(snap, clock);
      if (snap && Number.isFinite(snap.tick)) lastSnapTick = snap.tick;
      if (snap && snap.done) break;

      await nap(delay); // delay === 0 => yields to the event loop in fast mode
    }

    if (token === runToken && engine) {
      // Freeze on the final simulated tick. result().completionTick is the
      // authoritative final tick when available; otherwise use the last snapshot.
      const result = engine.result();
      const finalTick = (result && Number.isFinite(result.completionTick))
        ? result.completionTick : lastSnapTick;
      if (clock) clock.freeze(finalTick);
      // The idle greedy preview never writes results: only the trained run does.
      if (showResults) showResult(result);
      setStatus(`${label} complete`);
    }
  } catch (err) {
    const code = ERROR_CODES.DISPLAY_FAILED || 'DISPLAY_FAILED';
    logEvent({ tick: 0, type: code, item: '', delta: 0, reason: code });
    console.error('[main] ' + code, err);
    setStatus(code);
  } finally {
    if (token === runToken) {
      running = false;
      if (clock && typeof clock.isFrozen === 'function' && !clock.isFrozen()) {
        clock.freeze(lastSnapTick);
      }
    }
  }
}

/** Second display run driven by the evolved genomes delivered by the worker. */
async function runTrainedDisplay() {
  if (isFileProtocol || !config) return;
  // `required` stays only for legacy UI-tooltip compatibility. Policies and
  // observations use the manufactureRequired basis (plan section 1).
  const required = expandDemand(
    config.items, config.targets, config.initialStash, config.starterMode
  ).required;
  const manufactureRequired = expandManufactureRequired(
    config.items, config.targets, config.initialStash, config.starterMode
  ).manufactureRequired;

  // Compute criticality/caps diagnostics on this (main) thread only. Nothing
  // here is ever written into a worker message.
  void refreshDiagnostics();

  const helpers = await loadSolverHelpers();
  let caps = null;
  if (helpers && typeof helpers.planBatchCaps === 'function') {
    try {
      const planned = helpers.planBatchCaps(config.items, manufactureRequired, MAX_TICKS, MAX_BATCH);
      caps = (planned && planned.caps && typeof planned.caps === 'object') ? planned.caps : null;
    } catch (_e) { caps = null; }
  }

  let bundle = null;
  try {
    const factory = controllerModule.createDisplayPolicyBundle;
    if (typeof factory !== 'function') throw new Error('createDisplayPolicyBundle unavailable');
    bundle = factory({
      genomes: bestGenomes,
      config,
      required,
      manufactureRequired,
      adapterFactory: (opts) => createNeatAdapter(opts),
      seed,
      maxTicks: MAX_TICKS
    });
  } catch (err) {
    const code = ERROR_CODES.DISPLAY_FAILED || 'DISPLAY_FAILED';
    logEvent({ tick: 0, type: code, item: '', delta: 0, reason: code });
    console.error('[main] ' + code, err);
    setStatus(code);
    return;
  }

  let policyForItem = (bundle && bundle.policyForItem) || greedyPolicyMap();
  // Apply the shared, single-implementation cap helper to the display policies.
  if (helpers && typeof helpers.applyBatchCaps === 'function' && caps) {
    try {
      const capped = helpers.applyBatchCaps(policyForItem, caps);
      if (capped) policyForItem = capped;
    } catch (_e) { /* keep uncapped policies if the helper rejects the shape */ }
  }
  const setEngine = (bundle && typeof bundle.setEngine === 'function') ? bundle.setEngine : () => {};
  // Cancel the idle preview; the trained run becomes the authoritative result.
  runToken += 1;
  const token = runToken;
  logEvent({ tick: 0, type: 'TRAINED_DISPLAY', item: '', delta: 0, reason: 'replaying trained genomes' });
  void runDisplay(token, { policyForItem, setEngine, showResults: true, label: 'trained display' });
}

/* ------------------------------- controls --------------------------------- */

/**
 * Start (or resume) the run. `deps` injects I/O for the integration seam so
 * tests can drive the REAL module-config propagation without bootstrap or
 * #dag-svg; production supplies no deps and uses the live DOM/worker/display.
 */
export async function startAll(deps = {}) {
  if (starting || (running && !paused)) return;
  if (running && paused) { // resume
    paused = false;
    setStatus(`display running (${speed})`);
    return;
  }

  const cfg = deps.controls || readControls();
  seed = cfg.seed;
  speed = cfg.speed;
  bestGenomes = {};
  runToken += 1;
  const token = runToken;
  starting = true;
  // Start/Apply resets the simulated clock to tick 0. No real time is measured.
  const clock = deps.simClock || simClock;

  try {
    if (clock) clock.reset();
    const load = typeof deps.loadConfig === 'function' ? deps.loadConfig : loadConfig;
    const base = await load(cfg.preset);
    if (token !== runToken) { starting = false; return; }

    // Read AFTER the config resolves so prefill fallbacks use the active
    // preset. The same committed module config feeds worker, engine and diag.
    const overrides = typeof deps.readInitialState === 'function'
      ? deps.readInitialState(base)
      : readInitialState(base, undefined, logInvalidInput);
    const effective = applyOverridesAndStart(base, overrides, {
      seed,
      options: cfg,
      onCreateEngine: deps.onCreateEngine,
      onStart: deps.onStart
    });

    const panel = deps.panel || initialState;
    if (panel && typeof panel.prefill === 'function') panel.prefill(effective.config);

    const start = typeof deps.startWorkerTraining === 'function'
      ? deps.startWorkerTraining : startWorkerTraining;
    start(cfg);

    // Under file:// there is no worker, so the labelled greedy preview is the
    // only available result. Over http it stays an idle preview; only the
    // trained run writes to the Results tab.
    if (typeof deps.runDisplay === 'function') deps.runDisplay(token, effective, cfg);
    else void runDisplay(token, {
      showResults: isFileProtocol,
      label: isFileProtocol ? 'file:// preview' : 'idle preview'
    });

    const refresh = typeof deps.refreshDiagnostics === 'function'
      ? deps.refreshDiagnostics : refreshDiagnostics;
    void refresh();
    starting = false;
    return effective.config; // the exact committed module config
  } catch (err) {
    const code = ERROR_CODES.CONFIG_FAILED || 'CONFIG_FAILED';
    logEvent({ tick: 0, type: code, item: '', delta: 0, reason: code });
    console.error('[main] ' + code, err);
    setStatus(code);
    starting = false;
    return null;
  }
}

function pauseAll() {
  if (!running || paused) return;
  paused = true;
  setStatus('paused');
}

/** Reset to the pristine preset: restore module config and clear overrides
 * (re-prefill the panel from baseConfig) BEFORE rebuilding the idle snapshot. */
export function resetAll(deps = {}) {
  runToken += 1;
  const clock = deps.simClock || simClock;
  if (clock) clock.reset();
  running = false;
  paused = false;
  engine = null;
  bestGenomes = {};
  if (worker) {
    worker.terminate();
    worker = null;
  }
  config = baseConfig;
  const panel = deps.panel || initialState;
  if (panel && typeof panel.prefill === 'function') {
    panel.prefill(baseConfig || config || {});
  }
  if (logger && typeof logger.clear === 'function') logger.clear();
  try {
    const initial = buildEngine().snapshot();
    updateDag(initial);
    updateHud(initial);
  } catch (_e) { /* config may not be loaded yet */ }
  setStatus('idle — press Start');
}

/**
 * Preset-change path: load the new preset, refresh `baseConfig`, re-prefill
 * the panel, and rebuild diagnostics. Injected I/O keeps this testable without
 * bootstrap. Does not touch a display already in flight.
 */
export async function onPresetChange(presetId, deps = {}) {
  const load = typeof deps.loadConfig === 'function' ? deps.loadConfig : loadConfig;
  let base;
  try {
    base = await load(presetId);
  } catch (err) {
    const code = ERROR_CODES.CONFIG_FAILED || 'CONFIG_FAILED';
    logEvent({ tick: 0, type: code, item: '', delta: 0, reason: code });
    console.error('[main] ' + code, err);
    setStatus(code);
    return null;
  }
  baseConfig = base;
  if (!running) config = base;
  const panel = deps.panel || initialState;
  if (panel && typeof panel.prefill === 'function') panel.prefill(base);
  const refresh = typeof deps.onRefresh === 'function' ? deps.onRefresh : refreshDiagnostics;
  void refresh(base);
  try {
    const initial = buildEngine().snapshot();
    updateDag(initial);
    updateHud(initial);
  } catch (_e) { /* config may not be usable yet */ }
  setStatus(`preset ${presetId} — initial state refreshed`);
  return base;
}

/* --------------------------------- tabs ----------------------------------- */

/** Tab ids, in markup order. Panel id = 'panel-' + id without the 'tab-' prefix. */
export const TAB_IDS = ['tab-train', 'tab-logger', 'tab-results'];

/**
 * Apply a tab selection to a document. Pure of any module state: the tab gets
 * the shipped `is-active` class (CSS uses `.tab.is-active`), aria-selected is
 * kept in sync, and exactly one `#panel-<name>` has its `hidden` attribute
 * removed (CSS hides via `.panel[hidden]{display:none}`).
 */
export function applyTabSelection(doc, selectedId) {
  if (!doc) return selectedId;
  for (const id of TAB_IDS) {
    const selected = id === selectedId;
    const tab = doc.getElementById(id);
    if (tab) {
      if (tab.classList && typeof tab.classList.toggle === 'function') {
        tab.classList.toggle('is-active', selected);
      }
      if (typeof tab.setAttribute === 'function') {
        tab.setAttribute('aria-selected', selected ? 'true' : 'false');
      }
    }
    const panel = doc.getElementById('panel-' + id.slice('tab-'.length));
    if (panel) {
      if (typeof panel.removeAttribute !== 'function' || typeof panel.setAttribute !== 'function') continue;
      if (selected) panel.removeAttribute('hidden');
      else panel.setAttribute('hidden', '');
    }
  }
  return selectedId;
}

/**
 * Bind tab clicks. `doc` is injectable so the wiring is testable with a DOM
 * stub; defaults to the live document. The initial call forces the state to
 * match the shipped markup (Train active, Logger/Results hidden).
 */
export function wireTabs(doc = (typeof document !== 'undefined' ? document : null)) {
  if (!doc) return;
  for (const id of TAB_IDS) {
    const el = doc.getElementById(id);
    if (!el || typeof el.addEventListener !== 'function') continue;
    el.addEventListener('click', () => applyTabSelection(doc, id));
  }
  applyTabSelection(doc, 'tab-train');
}

/* ------------------------------- bootstrap -------------------------------- */

async function bootstrap() {
  dag = createDag($('dag-svg'));
  hud = createHud({ stashEl: $('stash-panel'), coinEl: $('coin-counter'), spentEl: $('coin-spent') });
  logger = createLogger($('log-list'));
  results = createResults($('results-body'));
  controls = createControls({
    onStart: startAll,
    onPause: pauseAll,
    onReset: resetAll,
    onPresetChange: (cfg) => { void onPresetChange(cfg.preset); }
  });
  // The editable Initial state panel is pure over an injected snapshot; it is
  // (re)prefilled as soon as a config resolves below. Apply/Reset delegate to
  // the same start/reset paths; rejected input logs only CONFIG_INPUT_INVALID.
  initialState = createInitialStatePanel({}, {
    onApply: () => { void startAll(); },
    onReset: () => resetAll(),
    onInvalid: logInvalidInput
  });
  simClock = createSimClock(); // renders "Sim time: 0 min" at load
  // Diagnostics view is optional: it only mounts when a host element exists.
  const diagHost = $('solver-diagnostics');
  if (diagHost) diagnostics = createDiagnostics(diagHost);

  // Button wiring is owned by controls.js: no duplicate listeners here.
  wireTabs();

  try {
    // Fresh load / refresh always starts from the `empty` preset: 0 coins,
    // empty starter stash and the YAML targets. No browser storage is read.
    config = await loadConfig('empty');
  } catch (err) {
    const code = ERROR_CODES.CONFIG_FAILED || 'CONFIG_FAILED';
    logEvent({ tick: 0, type: code, item: '', delta: 0, reason: code });
    console.error('[main] ' + code, err);
    setStatus(code);
    return;
  }
  // Prefill only AFTER the config resolves (same in startAll / preset change).
  baseConfig = config;
  if (initialState) initialState.prefill(config);
  // Criticality/caps are computed and rendered on the main thread only.
  void refreshDiagnostics();

  if (isFileProtocol) {
    setStatus('file:// — training disabled');
    logEvent({
      tick: 0, type: 'INFO', item: '', delta: 0,
      reason: 'file:// detected: embedded config only; training disabled; worker not started.'
    });
  }

  // Show the initial (tick 0) state in DAG + HUD.
  try {
    const initial = buildEngine().snapshot();
    updateDag(initial);
    updateHud(initial);
  } catch (_e) { /* leave UI empty if the initial snapshot cannot be built */ }

  setStatus(isFileProtocol ? 'file:// — press Start for display run' : 'idle — press Start');
}

// tests.html has no #dag-svg: importing this module there must not bootstrap
// the app, but the exported live dispatch path remains testable.
if (typeof document !== 'undefined' && document.getElementById('dag-svg')) {
  void bootstrap();
}
