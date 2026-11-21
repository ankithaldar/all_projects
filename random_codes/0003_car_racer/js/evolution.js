// js/evolution.js — The evolution engine for the 0003_car_racer NN drivers.
//
// Script position (documented, LOAD-BEARING): loaded AFTER nn.js, track.js,
// car.js, sensors.js and js/driver-sim.js, and BEFORE Db / Trainer / Replay /
// Main. Offline use in the browser rides on the plain <script> stack; the
// Web Worker imports it via importScripts AFTER the five (nn → track → car →
// sensors → driver-sim) it needs.
//
// Evolution owns THE mulberry32 instance for the whole lineage (see plan
// EVOLUTION + DETERMINISM). All randomness flows through it:
//   * gen 1 — 20 nets via NeuralNetwork.randomInit(this.rng)
//   * gen 51 — ONE-TIME full restart ONLY: all 20 children are FRESH
//     new NeuralNetwork(this.rng) nets from the CURRENT stream (20 × 800 =
//     16,000 fixed draws). The elite clone is DELIBERATELY DROPPED at this one
//     generation — the single escape from the oval-specialist local optimum.
//   * EVERY other generation — explicitly INCLUDING 101, 151 and 201, which
//     are NOT restart boundaries — child0 = best.clone() EXACT (elitism /
//     replay anchor, zero draws) and children 1..19 =
//     best.clone().mutate(MUTATION_RATE, MUTATION_SCALE, rng). No re-diversification.
// Resume byte-identity still holds via the validated rngState cursor (see
// resumeFromBest / STATE PORTABILITY below).
// No Math.random anywhere in this file (or anywhere in the train path).
//
// DETERMINISM CONTRACT: same seed + same track ⇒ byte-identical populations
// and best-weights for every generation, on the same JS engine (v1 target);
// cross-engine last-ulp divergence is real (see M1 F-12). Achieved by (a) the
// fixed draw order in randomInit/mutate (nn.js), (b) all constants below being
// literal, (c) the eval loop never reading wall-clock/Date.now for liveness,
// caps or rng decisions. The only "human" input is the seed / rngState.
//
// STATE PORTABILITY: the Evolution instance is the single owner of the stream.
//   this.rngState()          → current uint32 cursor (for DB meta.rng_state and
//                              the worker's per-generation payload)
//   new Evolution(seed, { rngState })   → re-seed a restored stream cursor
// The mutation draws for generation N+1 are made with the cursor captured at
// the END of generation N (after gen N's children were spawned), which is what
// makes resume byte-identical to an uninterrupted run (see
// Evolution.resumeFromBest()).
//
// SIM SECONDS: SIM_DT, EVAL_CAP_S and STUCK_TIMEOUT_S are SIMULATED seconds
// (ticks = sim-seconds × 120). The eval loop never consults the clock. The
// four eval constants are owned by js/driver-sim.js (the single source); this
// file reads them from globalThis and re-exports them for node tests.
//
// LIVENESS / STUCK KILL (plan EVOLUTION + DETERMINISM, STUCK_TIMEOUT_S=12,
// PROGRESS_ADVANCE_PX=20): the arcPos of track.nearestCenterline() is tracked
// as an UNWRAPPED cumulative progress value `cumArc` (the wrap delta is
// corrected by ±track.length so a closed-loop lap does NOT look like a
// regression). Only forward motion accumulates
//   if (cumArc - lastResetCumArc >= PROGRESS_ADVANCE_PX) → reset stagnation
//   else                                                  → stagnation += dt
// A stationary car, a slowly oscillating car and a car doing tight circles all
// die at exactly STUCK_TIMEOUT_S simulated seconds without +20 px of net
// forward centerline progress. Odometer is net signed px and (with the lap
// bonus) composes fitness; arcPos gates liveness only. The tick lives in
// js/driver-sim.js (DriverSim) — EVERY eval path (headless
// trainer, replay, spectate) runs the same one.
//
// Plain classic script. Browser: exposes globals on globalThis (Evolution,
// POPULATION, …). Node: guarded module.exports at the bottom (tests). The
// module references its dependencies ONLY through globalThis (same pattern as
// car.js), and fails loudly at load time if one is missing — in node, require
// track.js → car.js → sensors.js → nn.js → driver-sim.js → evolution.js IN
// THAT ORDER.
//
// DOM-free by design: no document/window/canvas/innerHTML/fetch references.

'use strict';

// Shared ValidationError — see nn.js for the rationale (script order safety).
if (typeof globalThis.ValidationError !== 'function') {
  globalThis.ValidationError = class ValidationError extends Error {
    constructor(message) {
      super(message);
      this.name = 'ValidationError';
    }
  };
}
var ValidationError = globalThis.ValidationError;

// The dependencies this file needs from earlier scripts. Read them once at
// load time and fail loudly instead of half-working mid-run (same policy as
// car.js → track.js).
var _mulberry32 = (typeof globalThis !== 'undefined' && globalThis.mulberry32) || null;
var _NeuralNetwork = (typeof globalThis !== 'undefined' && globalThis.NeuralNetwork) || null;
var _Car = (typeof globalThis !== 'undefined' && globalThis.Car) || null;
var _SensorFan = (typeof globalThis !== 'undefined' && globalThis.SensorFan) || null;
var _DriverSim = (typeof globalThis !== 'undefined' && globalThis.DriverSim) || null;
var _SIM_DT = (typeof globalThis !== 'undefined' && globalThis.SIM_DT);
var _EVAL_CAP_S = (typeof globalThis !== 'undefined' && globalThis.EVAL_CAP_S);
var _STUCK_TIMEOUT_S = (typeof globalThis !== 'undefined' && globalThis.STUCK_TIMEOUT_S);
var _PROGRESS_ADVANCE_PX = (typeof globalThis !== 'undefined' && globalThis.PROGRESS_ADVANCE_PX);
var _MAX_SPEED = (typeof globalThis !== 'undefined' && globalThis.MAX_SPEED) || null;
var _SENSOR_MAX = (typeof globalThis !== 'undefined' && globalThis.SENSOR_MAX) || null;
var _strictUint32 = (typeof globalThis !== 'undefined' && globalThis.strictUint32) || null;
if (typeof _mulberry32 !== 'function' ||
    typeof _NeuralNetwork !== 'function' ||
    typeof _Car !== 'function' ||
    typeof _SensorFan !== 'function' ||
    typeof _DriverSim !== 'function' ||
    typeof _SIM_DT !== 'number' ||
    typeof _EVAL_CAP_S !== 'number' ||
    typeof _STUCK_TIMEOUT_S !== 'number' ||
    typeof _PROGRESS_ADVANCE_PX !== 'number' ||
    typeof _MAX_SPEED !== 'number' ||
    typeof _SENSOR_MAX !== 'number' ||
    typeof _strictUint32 !== 'function') {
  throw new ValidationError(
    'evolution.js requires nn.js, track.js, car.js, sensors.js and driver-sim.js to load first ' +
    '(globalThis.mulberry32 / NeuralNetwork / Car / SensorFan / DriverSim / SIM_DT / EVAL_CAP_S / ' +
    'STUCK_TIMEOUT_S / PROGRESS_ADVANCE_PX / MAX_SPEED / SENSOR_MAX / strictUint32)'
  );
}

// ---------------------------------------------------------------------------
// CONSTANTS — POPULATION / MAX_GENERATIONS / MUTATION_* are evolution's own
// (mirrors the plan's CONSTANTS). The four EVAL CONSTANTS are owned by
// js/driver-sim.js and read from globalThis here (no literal twins).
// ---------------------------------------------------------------------------
var POPULATION = 20;
var MAX_GENERATIONS = 5000;
var SIM_DT = _SIM_DT;            // s — one physics tick in SIMULATED seconds (driver-sim.js)
var EVAL_CAP_S = _EVAL_CAP_S;    // s — max sim-seconds a driver may live (driver-sim.js)
var STUCK_TIMEOUT_S = _STUCK_TIMEOUT_S;     // s — sim-seconds without net progress → kill (driver-sim.js)
var PROGRESS_ADVANCE_PX = _PROGRESS_ADVANCE_PX; // px — net centerline progress that resets the timer (driver-sim.js)
var MUTATION_RATE = 0.25;        // per-weight probability of a Gaussian nudge
var MUTATION_SCALE = 0.30;       // Gaussian sigma of the nudge

// ---------------------------------------------------------------------------
// wrapToPi — normalize an angle to [−π, π). The plan's NN-input direction is
// wrapToPi(heading − tangentAngle)/π ∈ [−1,1]; canvas convention (y-down ⇒
// positive angle = clockwise; steer +1 = right = clockwise heading increase).
// ---------------------------------------------------------------------------
function wrapToPi(a) {
  return Math.atan2(Math.sin(a), Math.cos(a));
}

// ---------------------------------------------------------------------------
// Small helpers.
// ---------------------------------------------------------------------------
function assertFiniteNumber(value, name) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ValidationError(name + ' must be a finite number, got ' + String(value));
  }
  return value;
}

// assertUint32 — thin name-bound wrapper around nn.js's strictUint32 (the ONE
// strict rule for uint32 state/cursors; see nn.js). Kept as a local name so
// call sites read naturally; carries the strict contract unchanged.
function assertUint32(value, name) {
  return _strictUint32(value, name);
}

function assertResultsArray(results) {
  if (!Array.isArray(results) || results.length !== POPULATION) {
    throw new ValidationError('results must be an array of exactly ' + POPULATION + ' per-driver result objects');
  }
  for (var i = 0; i < results.length; i++) {
    var r = results[i];
    if (r === null || typeof r !== 'object') {
      throw new ValidationError('results[' + i + '] must be an object { distance, topSpeed, timeAlive }');
    }
    assertFiniteNumber(r.distance, 'results[' + i + '].distance');
    assertFiniteNumber(r.topSpeed, 'results[' + i + '].topSpeed');
    assertFiniteNumber(r.timeAlive, 'results[' + i + '].timeAlive');
  }
  return results;
}

// ---------------------------------------------------------------------------
// Evolution — the population and the RNG stream, nothing else.
//
// constructor(seed, opts)
//   seed: uint32 (number or decimal string). Required unless opts.rngState is
//         provided (resume path). Seed 4242 ⇒ fully reproducible lineage.
//   opts (optional):
//     rngState — uint32 cursor previously read from .rngState(); when present
//                the fresh stream is fast-forwarded/restored to exactly that
//                cursor (setState) and the seed is ignored. Used by the
//                resume path together with bestWeights (see resumeFromBest).
// ---------------------------------------------------------------------------
class Evolution {
  constructor(seed, opts) {
    opts = opts || {};
    var hasResumeState = opts.rngState !== undefined && opts.rngState !== null;
    if (!hasResumeState) {
      if (seed === undefined || seed === null) {
        throw new ValidationError('Evolution requires a uint32 seed (or a restored opts.rngState)');
      }
      this.seed = assertUint32(seed, 'Evolution seed');
      this.rng = _mulberry32(this.seed);
    } else {
      var restored = assertUint32(opts.rngState, 'Evolution opts.rngState');
      this.seed = seed === undefined || seed === null ? 0 : assertUint32(seed, 'Evolution seed');
      this.rng = _mulberry32(this.seed);
      this.rng.setState(restored);
    }
    this.population = null;   // array of POPULATION NeuralNetwork instances
    this.generation = 0;      // 1 after randomPopulation(); +1 per spawnNextGeneration()
    this.bestIndex = -1;      // index into population; -1 until an evaluation is recorded
    this.lastResults = null;  // per-driver results of the most recent evaluation
  }

  // ---- gen 1: 20 random drivers; consumes the FIRST rng draws of the lineage.
  randomPopulation() {
    var pop = new Array(POPULATION);
    for (var i = 0; i < POPULATION; i++) {
      pop[i] = _NeuralNetwork.randomInit(this.rng);
    }
    this.population = pop;
    this.generation = 1;
    this.bestIndex = -1;
    this.lastResults = null;
    return pop;
  }

  // ---- Static evaluator. Thin wrapper since the DriverSim extraction: the ONE
  //      per-car tick lives in js/driver-sim.js (shared by this eval path,
  //      replay and spectate). The returned payload is unchanged. The legacy
  //      positional `config` slot was DELETED (FINAL-BATCH-3B deviation #1) —
  //      the signature is now (track, net, policyOverride) and any third
  //      argument IS the policyOverride.
  //
  // Evolution.evaluateDriver(track, net, policyOverride)
  //   track:          a Track instance (startPose / nearestCenterline / segments)
  //   net:            a NeuralNetwork instance OR null when policyOverride is given
  //   policyOverride: optional (inputs7) => [gas, steer]. TEST HOOK. When set it
  //                    replaces net.forward entirely. There is NO config slot —
  //                    the eval constants are frozen (DRIVER_SIM_CONFIG in
  //                    js/driver-sim.js) and any third argument IS the
  //                    policyOverride.
  //
  // Returns { distance, topSpeed, timeAlive, crashed, stuck }:
  //   distance  — FITNESS: car.odometer + laps × LAP_BONUS_TRACK_LENGTHS × track.length
  //               (net signed px + lap bonus; may be negative) — the selection currency
  //   topSpeed  — max forward speed seen, px/s (reported only; not used by selectBest)
  //   timeAlive — simulated seconds lived (exact-distance tiebreak: HIGHER wins)
  //   crashed   — true if a wall collision ended the run (exact-distance tiebreak: non-crashed wins)
  //   stuck     — true if the stagnation timer ended the run
  // (Per tick — order, liveness unwrap, kill caps, 'finished' terminal — see js/driver-sim.js;
  //  the raw odometer and the lap counter are NOT surfaced here — fitness() is the currency.)
  static evaluateDriver(track, net, policyOverride) {
    var sim = new _DriverSim(track, net, typeof policyOverride === 'function' ? policyOverride : null);
    for (;;) {
      if (sim.step().done) break;
    }
    return {
      distance: sim.fitness(),    // bonus-inclusive fitness (net signed + lap bonus; may be negative)
      topSpeed: sim.topSpeed(), // px/s
      timeAlive: sim.timeAlive, // simulated seconds
      crashed: sim.crashed,
      stuck: sim.stuck,
    };
  }

  // recordEvaluation(results) — validate + remember the outcome of one
  // generation. Shared by evaluatePopulation (sync path) and the worker's
  // yielding driver loop (which calls evaluateDriver per driver and reports
  // here), so both paths produce IDENTICAL selection results.
  recordEvaluation(results) {
    var r = assertResultsArray(results);
    var bestIndex = Evolution.selectBest(r);
    this.bestIndex = bestIndex;
    this.lastResults = r;
    return bestIndex;
  }

  // evaluatePopulation(track) — synchronous evaluation of all
  // POPULATION drivers (the node / determinism path). Returns
  // { results, bestIndex }.
  evaluatePopulation(track) {
    if (!this.population) {
      throw new ValidationError('evaluatePopulation before any population exists — call randomPopulation() first');
    }
    var results = new Array(POPULATION);
    for (var i = 0; i < POPULATION; i++) {
      results[i] = Evolution.evaluateDriver(track, this.population[i]);
    }
    var bestIndex = this.recordEvaluation(results);
    return { results: results, bestIndex: bestIndex };
  }

  // selectBest(results) — fitness = distance (net signed px + lap bonus; may
  // be negative). PRIMARY ordering is by distance ONLY and is never changed:
  // bestIndex is always a max-distance driver (main.js:1238 stays truthful).
  // On an EXACT distance tie: prefer a non-crashed driver (null-safe `!crashed`
  // so false/0/undefined all count as "not crashed"), then prefer HIGHER
  // timeAlive. Exact float comparisons (deterministic). Static.
  static selectBest(results) {
    var r = assertResultsArray(results);
    var best = 0;
    for (var i = 1; i < r.length; i++) {
      var a = r[i], b = r[best];
      if (a.distance > b.distance) { best = i; continue; }
      if (a.distance === b.distance) {
        var aAlive = !a.crashed, bAlive = !b.crashed;
        if (aAlive !== bAlive) {
          if (aAlive) best = i;        // non-crashed beats crashed
        } else if (a.timeAlive > b.timeAlive) {
          best = i;                    // equal crash status: longer life wins
        }
      }
    }
    return best;
  }

  // ---- spawn the next generation from the recorded best index.
  // For every target generation the routine is uniform (see _spawnChildren):
  // child0 = best.clone() EXACT (elitism / replay anchor) and children 1..19
  // are Gaussian mutants — EXCEPT the one-time full restart at generation 51,
  // where all 20 children are fresh nets and the elite clone is dropped.
  // 101/151/201 are ordinary mutant generations, NOT restart boundaries.
  // The TARGET generation is passed EXPLICITLY (this.generation + 1) BEFORE
  // the counter is incremented. Resume byte-identity is preserved because the
  // rng cursor at that point is exactly the validated rngState handshake.
  spawnNextGeneration() {
    if (!this.population || this.bestIndex < 0) {
      throw new ValidationError('spawnNextGeneration requires a recorded evaluation — bestIndex unknown');
    }
    var parent = this.population[this.bestIndex];
    this.population = this._spawnChildren(parent, this.generation + 1);
    this.generation += 1;
    this.bestIndex = -1;
    this.lastResults = null;
    return this.population;
  }

  // _spawnChildren(parent, nextGeneration) — the single spawn routine used by
  // BOTH the normal path (parent = current best) and the resume path (parent =
  // best weights loaded from the DB).
  //
  // ONE-TIME FULL RESTART at nextGeneration === 51 (and ONLY 51): every child
  // 0..19 is a FRESH new NeuralNetwork(this.rng) from the CURRENT stream — the
  // same constructor path as randomInit, 800 fixed draws each (20 × 800 =
  // 16,000 draws), biases zero-filled. The elite clone is DELIBERATELY DROPPED
  // at this generation. This is the single escape from the oval-specialist
  // local optimum.
  //
  // EVERY OTHER generation — explicitly INCLUDING 101, 151 and 201 — is an
  // ordinary mutant spawn: child0 = parent.clone() EXACT (elitism / replay
  // anchor, zero RNG draws), children 1..19 = parent.clone().mutate(
  // MUTATION_RATE, MUTATION_SCALE, this.rng). There is NO re-diversification
  // at 101/151/201; they are not boundaries. Population stays exactly 20 and
  // the draw count is a pure function of nextGeneration + the rng cursor.
  // Resume byte-identity still holds: the caller's validated rngState cursor
  // reproduces this stream exactly (see resumeFromBest).
  _spawnChildren(parent, nextGeneration) {
    var children = new Array(POPULATION);
    if (nextGeneration === 51) {
      // One-time full restart: all 20 fresh nets, elite clone deliberately dropped.
      for (var f = 0; f < POPULATION; f++) {
        children[f] = new _NeuralNetwork(this.rng);
      }
      return children;
    }
    // Every other generation (incl. 101/151/201): exact elite clone + 19 mutants.
    children[0] = parent.clone(); // elitism anchor — NO rng draws
    for (var i = 1; i < POPULATION; i++) {
      children[i] = parent.clone().mutate(MUTATION_RATE, MUTATION_SCALE, this.rng);
    }
    return children;
  }

  // Evolution.resumeFromBest(bestWeights, completedGeneration, rngState) —
  // the resume constructor. `bestWeights` is validated by NeuralNetwork.fromJSON
  // (architecture + exactly 434 finite floats + |v| ≤ 1e6); `rngState` is the
  // uint32 cursor recorded AFTER `completedGeneration` was evaluated (i.e. the
  // state an uninterrupted run would hold right before spawning generation
  // N+1). Returns an Evolution whose population IS generation N+1 and whose
  // .generation is N+1 — byte-identical children to an uninterrupted run.
  static resumeFromBest(bestWeights, completedGeneration, rngState) {
    if (!Number.isInteger(completedGeneration) || completedGeneration < 1) {
      throw new ValidationError('resumeFromBest completedGeneration must be an integer >= 1, got ' + String(completedGeneration));
    }
    var state = assertUint32(rngState, 'resumeFromBest rngState');
    var parent = _NeuralNetwork.fromJSON(bestWeights); // full weights validation
    var evo = new Evolution(0, { rngState: state });
    // Pass the TARGET generation explicitly BEFORE setting evo.generation —
    // evo.generation is still 0 here, so reading it would compute the wrong
    // target. completedGeneration + 1 is the generation being resumed into
    // (e.g. 51 for the one-time full restart); the validated rngState cursor
    // restores the exact stream so the spawn is byte-identical to an
    // uninterrupted run.
    evo.population = evo._spawnChildren(parent, completedGeneration + 1);
    evo.generation = completedGeneration + 1;
    evo.bestIndex = -1;
    evo.lastResults = null;
    return evo;
  }

  // rngState() — current uint32 cursor of THE stream. Captured once per
  // generation for the DB (meta.rng_state, authoritative) and logged with
  // every generation row (generations.rng_state, audit-only).
  rngState() {
    return this.rng.state();
  }

  // bestWeights() — the full weight payload of the CURRENT best net. Call it
  // BEFORE spawnNextGeneration() mutates anything; this is the read the
  // logger/replay/resume wants. Paired with rngState() (the validated cursor),
  // it lets resumeFromBest() rebuild the exact next population — at the one-time
  // gen-51 restart AND at every ordinary mutant generation — byte-identically.
  // Throws until an evaluation has been recorded.
  bestWeights() {
    if (!this.population || this.bestIndex < 0) {
      throw new ValidationError('bestWeights() requires a recorded evaluation — call evaluatePopulation() first');
    }
    return this.population[this.bestIndex].getWeights();
  }
}

// Browser-global exposure + guarded CommonJS export (node tests).
if (typeof globalThis !== 'undefined') {
  globalThis.Evolution = Evolution;
  globalThis.POPULATION = POPULATION;
  globalThis.MAX_GENERATIONS = MAX_GENERATIONS;
  globalThis.MUTATION_RATE = MUTATION_RATE;
  globalThis.MUTATION_SCALE = MUTATION_SCALE;
  globalThis.wrapToPi = wrapToPi;
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    Evolution: Evolution,
    ValidationError: ValidationError,
    POPULATION: POPULATION,
    MAX_GENERATIONS: MAX_GENERATIONS,
    SIM_DT: SIM_DT,
    EVAL_CAP_S: EVAL_CAP_S,
    STUCK_TIMEOUT_S: STUCK_TIMEOUT_S,
    PROGRESS_ADVANCE_PX: PROGRESS_ADVANCE_PX,
    MUTATION_RATE: MUTATION_RATE,
    MUTATION_SCALE: MUTATION_SCALE,
    wrapToPi: wrapToPi,
  };
}
