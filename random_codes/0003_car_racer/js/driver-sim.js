// js/driver-sim.js — ONE stateful driver simulation for the 0003_car_racer
// evolutionary NN car racer. Plain classic script, browser-global `DriverSim`
// on globalThis. Guarded CommonJS export at the bottom for node checks.
//
// WHAT THIS FILE IS
//   The single owner of the per-car simulation tick previously DUPLICATED in
//   Evolution.evaluateDriver (headless eval), js/replay.js (replay tick) and
//   js/main.js (spectate per-car step). One DriverSim instance owns a Car, a
//   SensorFan and the liveness state (arcPos seam-unwrapped cumulative
//   progress + stagnation timer). Advance it with .step().
//
// TICK ORDER (EXACTLY the training eval path's order):
//   1. read the 5 sensors (distances / sensorMax, [0,1]; 1 = no hit)
//   2. track.nearestCenterline() ONCE — feeds BOTH the NN input and liveness
//      (arcPos delta seam-unwrapped by ±track.length; cumArc += delta)
//   3. inputs = [s0..s4, car.speed / MAX_SPEED, wrapToPi(heading − tangent) / π]
//   4. gas/steer = policyOverride(inputs) ?? net.forward(inputs)
//   5. car.update(simDt, { gas, steer })
//   6. timeAlive += simDt; topSpeed = max(topSpeed, car.speed)
//   7. stagnation: cumArc − lastReset ≥ progressAdvancePx → reset, else += simDt
//   8. lap accounting: maxCumArc = high-water ratchet of cumArc;
//      laps = min(floor(maxCumArc / track.length), LAP_BONUS_LAPS)
//   9. kill on crash / finished (laps == LAP_BONUS_LAPS) / stagnation
//      ≥ stuckTimeoutS / tick cap (evalCapS / simDt) — kill order is
//      crash > finished > stuck > time-cap
//
// CANONICAL GLOSSARY (see PLAN Appendix B — normative for this file):
//   1. car.odometer — NET SIGNED px travelled: Σ(speed·dt); can be negative
//      (an all-reverse run). Display/debug only; never the selection currency
//      by itself.
//   2. liveness cumArc — seam-unwrapped signed centerline arcPos progress;
//      drives the stagnation kill. Unchanged.
//   3. cumArc high-water (maxCumArc) — ratcheted maximum of cumArc;
//      laps := floor(maxCumArc / track.length), capped at LAP_BONUS_LAPS.
//   4. fitness — car.odometer + laps × LAP_BONUS_TRACK_LENGTHS × track.length.
//      THE selection/logging/persistence currency. May be negative.
//   5. terminal outcomes — 'crashed' (wall touch), 'stuck' (stagnation kill),
//      'time-cap' (EVAL_CAP_S), 'finished' (laps == LAP_BONUS_LAPS).
//   6. kill order (deterministic) — crash > finished > stuck > time-cap.
//      Stagnation never preempts 'finished'.
//   7. fitness may be negative — stored/rendered as-is; selection handles
//      signs correctly.
//
// LOAD POSITION
//   This file loads FIRST (before track/car/sensors/nn/evolution) — it cannot
//   read its sibling globals at parse time, so every dependency is resolved
//   LAZILY at DriverSim construction time and fails CLOSED with a typed
//   ValidationError. The EVAL + LAP-BONUS CONSTANTS are owned HERE (the
//   single source) and mirrored onto globalThis + module.exports, so later
//   scripts read them directly with no fallback twins. js/evolution.js reads
//   the EVAL constants from globalThis at load (it loads after this file) and
//   re-exports them for node test consumers.
//
// API
//   new DriverSim(track, net, policyOverride?)
//     track         — a Track-like object (startPose / nearestCenterline / length).
//     net           — a NeuralNetwork-like object with forward(inputs7) → [gas, steer].
//                     May be null when policyOverride is provided.
//     policyOverride — TEST-HOOK ONLY: (inputs7) => [gas, steer]. Replaces
//                     net.forward entirely. Production callers pass nothing;
//                     js/evolution.js forwards its own test hook through it.
//                     The eval constants are FROZEN (DRIVER_SIM_CONFIG below)
//                     — there is no opts bag and no per-key override.
//   sim.step()     → { done, outcome } — outcome ∈ 'crashed' | 'stuck' | 'time-cap' | 'finished'
//                     (idempotent once done: replays the recorded outcome).
//                     Throws ValidationError when the policy emits non-finite
//                     [gas, steer] (the historical net.forward contract).
//   sim.car        → the underlying Car instance.
//   sim.fitness()  → car.odometer + laps × LAP_BONUS_TRACK_LENGTHS × track.length
//                     (THE selection/logging/persistence currency; signed).
//   sim.laps       → laps credited via the maxCumArc high-water ratchet
//                     (floor(maxCumArc / track.length), capped at LAP_BONUS_LAPS).
//   sim.maxCumArc  → the ratcheted high-water mark of cumArc (lap accounting).
//   sim.topSpeed() → max forward speed seen (px/s).
//   sim.timeAlive  → simulated seconds lived (a plain number).
//   sim.outcome    → null until done, then 'crashed' | 'stuck' | 'time-cap' | 'finished'.
//   sim.crashed / sim.stuck → booleans (false until the sim ends; 'finished' leaves both false).
//   sim.simDt      → the per-tick quantum in simulated seconds (replay's
//                     rAF accumulator sizes its steps from this).
//   sim.maxTicks   → the hard tick cap (Math.ceil(evalCapS / simDt)).
//
// DOM-free by design: no document/window/canvas/innerHTML/fetch references.

'use strict';

// Shared ValidationError — see nn.js for the rationale (script order safety;
// this file loads FIRST, before nn.js, so it installs its own when absent).
if (typeof globalThis.ValidationError !== 'function') {
  globalThis.ValidationError = class ValidationError extends Error {
    constructor(message) {
      super(message);
      this.name = 'ValidationError';
    }
  };
}
var ValidationError = globalThis.ValidationError;

// ---------------------------------------------------------------------------
// CONSTANTS — the single source of truth for the eval loop (mirrors the plan's
// CONSTANTS). Owned HERE; everything else (evolution.js re-exports, replay.js
// accumulator, main.js SIM_DT) reads them from globalThis — no literal twins.
// ---------------------------------------------------------------------------
var SIM_DT = 1 / 120;            // s — one physics tick in SIMULATED seconds
var EVAL_CAP_S = 45;             // s — max sim-seconds a driver may live
var STUCK_TIMEOUT_S = 12;        // s — sim-seconds without net progress → kill
var PROGRESS_ADVANCE_PX = 20;    // px — net centerline progress that resets the timer
var LAP_BONUS_TRACK_LENGTHS = 2; // laps × 2 × track.length added to fitness per lap
var LAP_BONUS_LAPS = 100;        // cap on the lap bonus (and the 'finished' trigger)

// DRIVER_SIM_CONFIG — the immutable default set. Frozen; each instance copies
// these + the sensors.js SENSOR_MAX global into a per-instance cfg. Reads in the
// engine go through this._cfg (never the bare literals) so the frozen set is the
// single source. Never mutate this object.
var DRIVER_SIM_CONFIG = Object.freeze({
  simDt: SIM_DT,
  evalCapS: EVAL_CAP_S,
  stuckTimeoutS: STUCK_TIMEOUT_S,
  progressAdvancePx: PROGRESS_ADVANCE_PX,
  lapBonusTrackLengths: LAP_BONUS_TRACK_LENGTHS,
  lapBonusLaps: LAP_BONUS_LAPS,
});

// resolveDeps() — lazy dependency lookup: this file loads FIRST, so the
// sibling globals do not exist at parse time. Fail CLOSED with a typed,
// readable ValidationError at the point of USE (construction), never at load
// time and never with a half-working sim.
function resolveDeps() {
  var g = globalThis;
  var Car = g.Car;
  var SensorFan = g.SensorFan;
  var MAX_SPEED = g.MAX_SPEED;
  var SENSOR_MAX = g.SENSOR_MAX;
  var wrapToPi = g.wrapToPi;
  if (typeof Car !== 'function' || typeof SensorFan !== 'function') {
    throw new ValidationError('driver-sim.js requires js/car.js and js/sensors.js to be loaded (globalThis.Car / SensorFan missing) — check the script order');
  }
  if (typeof MAX_SPEED !== 'number' || !Number.isFinite(MAX_SPEED) || MAX_SPEED <= 0) {
    throw new ValidationError('driver-sim.js requires js/car.js to be loaded (globalThis.MAX_SPEED missing — check the script order)');
  }
  if (typeof SENSOR_MAX !== 'number' || !Number.isFinite(SENSOR_MAX) || SENSOR_MAX <= 0) {
    throw new ValidationError('driver-sim.js requires js/sensors.js to be loaded (globalThis.SENSOR_MAX missing — check the script order)');
  }
  if (typeof wrapToPi !== 'function') {
    throw new ValidationError('driver-sim.js requires js/evolution.js to be loaded (globalThis.wrapToPi missing — check the script order)');
  }
  return { Car: Car, SensorFan: SensorFan, MAX_SPEED: MAX_SPEED, SENSOR_MAX: SENSOR_MAX, wrapToPi: wrapToPi };
}

// ---------------------------------------------------------------------------
// DriverSim — one stateful driver. See the header for the contract.
// ---------------------------------------------------------------------------
class DriverSim {
  constructor(track, net, policyOverride) {
    if (!track || typeof track.startPose !== 'function' || typeof track.nearestCenterline !== 'function') {
      throw new ValidationError('DriverSim requires a Track-like object (startPose(), nearestCenterline())');
    }
    var deps = resolveDeps();
    if (policyOverride !== undefined && policyOverride !== null && typeof policyOverride !== 'function') {
      throw new ValidationError('DriverSim policyOverride must be a function, got ' + typeof policyOverride);
    }
    if (typeof policyOverride !== 'function') {
      if (!net || typeof net.forward !== 'function') {
        throw new ValidationError('DriverSim requires a NeuralNetwork-like net (or a policyOverride)');
      }
    }
    // The eval constants are FROZEN (DRIVER_SIM_CONFIG) — no per-key overrides.
    var cfg = {
      simDt: DRIVER_SIM_CONFIG.simDt,
      evalCapS: DRIVER_SIM_CONFIG.evalCapS,
      stuckTimeoutS: DRIVER_SIM_CONFIG.stuckTimeoutS,
      progressAdvancePx: DRIVER_SIM_CONFIG.progressAdvancePx,
      lapBonusTrackLengths: DRIVER_SIM_CONFIG.lapBonusTrackLengths,
      lapBonusLaps: DRIVER_SIM_CONFIG.lapBonusLaps,
      sensorMax: deps.SENSOR_MAX,
    };

    this.track = track;
    this.net = typeof policyOverride === 'function' ? null : net;
    this._policyOverride = typeof policyOverride === 'function' ? policyOverride : null;
    this._deps = deps;
    this._cfg = cfg;
    this.simDt = cfg.simDt;
    this.maxTicks = Math.ceil(cfg.evalCapS / cfg.simDt);
    this.trackLength = track.length; // public — the fitness bonus term reads it

    var pose = track.startPose();
    this.car = new deps.Car(track, pose.x, pose.y, pose.heading);
    this.sensors = new deps.SensorFan(this.car, track, this._cfg.sensorMax);

    // ---- liveness state (arcPos is per-sample; unwrap across the lap seam) --
    this._trackLength = track.length;
    var first = track.nearestCenterline(this.car.x, this.car.y);
    this._lastArcPos = first.arcPos;
    this._cumArc = 0;           // cumulative UNWRAPPED signed centerline progress
    this._lastResetCumArc = 0;  // cumArc value at the last stagnation reset
    this._stagnation = 0;       // simulated seconds accumulated without +progressAdvancePx net
    this.maxCumArc = 0;         // high-water ratchet of cumArc (lap accounting, Appendix B)
    this.laps = 0;              // min(floor(maxCumArc / track.length), cfg.lapBonusLaps)

    this._ticks = 0;
    this.timeAlive = 0;
    this._topSpeed = 0;
    this.outcome = null;        // 'crashed' | 'stuck' | 'time-cap' | 'finished' once done
    this.crashed = false;
    this.stuck = false;
  }

  // fitness() — THE selection/logging/persistence currency (Appendix B):
  //   car.odometer + laps × LAP_BONUS_TRACK_LENGTHS × track.length
  // The odometer is net signed px (may be negative); the lap bonus is paid once
  // per lap boundary via the maxCumArc high-water ratchet (an oscillator or
  // infield flip-flop whose cumArc never grows pays nothing; an honest forward
  // driver pays exactly 2 × track.length per lap, capped at LAP_BONUS_LAPS).
  // May be negative (an all-reverse run). Stored/rendered/logged as-is.
  fitness() {
    return this.car.odometer + this.laps * this._cfg.lapBonusTrackLengths * this.trackLength;
  }

  // topSpeed() — max forward speed seen, px/s (selection tiebreak #1).
  topSpeed() {
    return this._topSpeed;
  }

  // step() — advance the simulation by ONE simDt tick. Returns
  // { done, outcome }; once done, repeated calls return the recorded outcome.
  step() {
    if (this.outcome !== null) return { done: true, outcome: this.outcome };

    var deps = this._deps;
    var car = this.car;
    var cfg = this._cfg;

    // 1. sensors
    var reading = this.sensors.read();
    // 2. centerline (one call per tick — feeds BOTH the NN input and liveness)
    var nc = this.track.nearestCenterline(car.x, car.y);
    var trackLength = this._trackLength;
    var delta = nc.arcPos - this._lastArcPos;
    if (delta > trackLength / 2) delta -= trackLength;        // wrapped forward across the seam
    else if (delta < -trackLength / 2) delta += trackLength;  // wrapped backward across the seam
    this._lastArcPos = nc.arcPos;
    this._cumArc += delta;
    if (this._cumArc > this.maxCumArc) this.maxCumArc = this._cumArc;
    this.laps = Math.min(Math.floor(this.maxCumArc / trackLength), cfg.lapBonusLaps);

    // 3. inputs (EXACT plan order: 5 sensor norms, speed/MAX_SPEED, direction/π)
    var direction = deps.wrapToPi(car.heading - nc.tangentAngle) / Math.PI;
    var inputs = [
      reading.norm[0], reading.norm[1], reading.norm[2], reading.norm[3], reading.norm[4],
      car.speed / deps.MAX_SPEED,
      direction,
    ];

    // 4. policy (net or test hook)
    var out = this._policyOverride ? this._policyOverride(inputs) : this.net.forward(inputs);
    var gas = Array.isArray(out) || (out && typeof out.length === 'number') ? out[0] : NaN;
    var steer = Array.isArray(out) || (out && typeof out.length === 'number') ? out[1] : NaN;
    if (!Number.isFinite(gas) || !Number.isFinite(steer)) {
      throw new ValidationError(
        'policyOverride/net.forward must produce [gas, steer] finite numbers, got ' +
        JSON.stringify([gas, steer])
      );
    }

    // 5. physics
    car.update(cfg.simDt, { gas: gas, steer: steer });

    // 6. bookkeeping
    this._ticks += 1;
    this.timeAlive += cfg.simDt;
    if (car.speed > this._topSpeed) this._topSpeed = car.speed;

    // 7. stagnation (arcPos gates liveness only)
    if (this._cumArc - this._lastResetCumArc >= cfg.progressAdvancePx) {
      this._lastResetCumArc = this._cumArc;
      this._stagnation = 0;
    } else {
      this._stagnation += cfg.simDt;
    }

    // 8. kill conditions — deterministic kill order: crash > finished > stuck > time-cap.
    //    'finished' (laps == LAP_BONUS_LAPS) fires as soon as the lap counter
    //    crosses the cap; stagnation never preempts it.
    if (car.crashed) this.outcome = 'crashed';
    else if (this.laps >= cfg.lapBonusLaps) this.outcome = 'finished';
    else if (this._stagnation >= cfg.stuckTimeoutS) this.outcome = 'stuck';
    else if (this._ticks >= this.maxTicks) this.outcome = 'time-cap';

    if (this.outcome !== null) {
      if (this.outcome === 'crashed') this.crashed = true;
      else if (this.outcome === 'stuck') this.stuck = true;
      // 'finished' leaves crashed/stuck false — train-worker posts the flags as-is.
      return { done: true, outcome: this.outcome };
    }
    return { done: false, outcome: null };
  }
}

// Browser-global exposure + guarded CommonJS export (node tests).
if (typeof globalThis !== 'undefined') {
  globalThis.DriverSim = DriverSim;
  globalThis.SIM_DT = SIM_DT;
  globalThis.EVAL_CAP_S = EVAL_CAP_S;
  globalThis.STUCK_TIMEOUT_S = STUCK_TIMEOUT_S;
  globalThis.PROGRESS_ADVANCE_PX = PROGRESS_ADVANCE_PX;
  globalThis.LAP_BONUS_TRACK_LENGTHS = LAP_BONUS_TRACK_LENGTHS;
  globalThis.LAP_BONUS_LAPS = LAP_BONUS_LAPS;
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    DriverSim: DriverSim,
    ValidationError: ValidationError,
    SIM_DT: SIM_DT,
    EVAL_CAP_S: EVAL_CAP_S,
    STUCK_TIMEOUT_S: STUCK_TIMEOUT_S,
    PROGRESS_ADVANCE_PX: PROGRESS_ADVANCE_PX,
    LAP_BONUS_TRACK_LENGTHS: LAP_BONUS_TRACK_LENGTHS,
    LAP_BONUS_LAPS: LAP_BONUS_LAPS,
    DRIVER_SIM_CONFIG: DRIVER_SIM_CONFIG,
  };
}
