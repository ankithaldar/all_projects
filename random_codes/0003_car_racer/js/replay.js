// js/replay.js — Manual replay of a logged generation (the "verify selection"
// path). Plain classic script, browser-global `Replay` on globalThis (the M4
// plan: script order Track → Car → Sensors → NN → Evolution → Db → Trainer →
// Replay → Main). Guarded CommonJS export at the bottom for node checks.
//
// WHAT THIS FILE DOES
//   1. Replay.loadGeneration(db, n) — reads generation n from the LINEAGE
//      FILE (never the active/editor track): rebuilds the Track from the
//      generations.track_json SNAPSHOT through Track.fromJSON (FULL
//      validation), cross-checks the rebuilt track's hash against the
//      generations.track_hash pin (mismatch ⇒ typed ValidationError), loads
//      the saved best weights through db.getBestWeights(n) (which runs
//      NeuralNetwork.fromJSON validation) and returns the plain VALIDATED
//      weights object. Also returns the per-driver results and a stats
//      summary for the HUD/Logs panel.
//   2. Replay.createRun(opts) — rebuilds the run HEADLESSLY (no DOM, no
//      canvas): a DriverSim at track.startPose(), advanced one SIM_DT tick at
//      a time (a NeuralNetwork from `weights` via fromJSON validation — a
//      wrong-count payload throws ValidationError). The tick order, liveness
//      rules, arcPos seam unwrap and kill caps are exactly the DriverSim
//      contract (js/driver-sim.js — ONE implementation shared with the
//      headless eval path and spectate). SIM_DT = 1/120 is a SIMULATED second;
//      EVAL_CAP_S = 45, STUCK_TIMEOUT_S = 12, PROGRESS_ADVANCE_PX = 20 —
//      owned by js/driver-sim.js. The loop normally runs off a rAF
//      accumulator in the browser; in headless/node contexts rAF does not
//      exist, so `step(dtSimMs)` advances the simulation manually.
//
// WHAT THIS FILE DOES NOT DO
//   * No DOM access at all — no innerHTML, no document.*, no canvas. The
//     caller (js/main.js) draws; this file reports plain objects through
//     onFrame / onFinish / onBanner.
//   * No fetch / network / eval / new Function. No file I/O.
//   * No writes to the database. Replay is READ-ONLY against Db.
//
// TRACK SNAPSHOT vs ACTIVE TRACK (plan §LAYOUT replay.js, §LINEAGE PIN)
//   Replay ALWAYS defaults to the track snapshot stored in the generation row.
//   The active/editor track is used ONLY when the caller passes the explicit
//   labeled toggle ("Use active track instead — results will differ") as
//   opts.useActiveTrack === true, together with the Track instance itself
//   (opts.activeTrack for loadGeneration, opts.track for createRun). In that
//   case the run is marked useActiveTrack = true (on the run object AND in
//   run.state), trackSource becomes 'active-track', and start() fires one
//   onBanner payload so the HUD/status can show the warning badge.
//
// ENVIRONMENT GUARDS
//   `typeof requestAnimationFrame !== 'undefined'` gates the browser loop;
//   `typeof module !== 'undefined'` guards the node export. Nothing here
//   assumes a window/document exists.

'use strict';

// ---------------------------------------------------------------------------
// ValidationError — shared, order-independent (same pattern as the other
// pure-logic files: install on globalThis if absent, then use the global).
// ---------------------------------------------------------------------------
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
// Constants — owned by js/driver-sim.js (the single source) and read from
// globalThis; the documented script order loads it first. No literal twins —
// a missing driver-sim.js is a typed, loud ValidationError at createRun()
// time (see requireDriverSim), never a silently divergent replay.
// ---------------------------------------------------------------------------
var MAX_FRAME_DELTA_MS = 250;                   // clamp tab-switch gaps; never spiral

// ---------------------------------------------------------------------------
// Dependency accessors — lazy lookups so a missing sibling file fails CLOSED
// with a typed, readable error at the point of use (never at load time, so
// partial environments can still be inspected without crashing the boot).
// ---------------------------------------------------------------------------
function requireDriverSim() {
  var DriverSim = globalThis.DriverSim;
  if (typeof DriverSim !== 'function') {
    throw new ValidationError('js/replay.js requires js/driver-sim.js (globalThis.DriverSim is missing) — load driver-sim.js before replay.js');
  }
  return DriverSim;
}
function requireReplayTrack() {
  var Track = globalThis.Track;
  if (!Track || typeof Track.fromJSON !== 'function') {
    throw new ValidationError('js/replay.js requires js/track.js (globalThis.Track.fromJSON is missing) — load track.js before replay.js');
  }
  return Track;
}
function requireReplayNeuralNetwork() {
  var NeuralNetwork = globalThis.NeuralNetwork;
  if (!NeuralNetwork || typeof NeuralNetwork.fromJSON !== 'function') {
    throw new ValidationError('js/replay.js requires js/nn.js (globalThis.NeuralNetwork.fromJSON is missing) — load nn.js before replay.js');
  }
  return NeuralNetwork;
}
// isTrackLike — a Track INSTANCE or an object with the full Track contract
// (startPose / nearestCenterline / raycast / segments / length). Used for the
// active-track override and for createRun's track argument.
function isTrackLike(track) {
  return !!track &&
    typeof track.startPose === 'function' &&
    typeof track.nearestCenterline === 'function' &&
    typeof track.raycast === 'function' &&
    Array.isArray(track.segments) &&
    typeof track.length === 'number' && Number.isFinite(track.length) && track.length > 0;
}

// safeHook(fn, payload, label) — a throwing UI hook must never break the
// simulation loop or lose the finish signal (house pattern, see db.js).
function safeHook(fn, payload, label) {
  if (typeof fn !== 'function') return;
  try {
    fn(payload);
  } catch (err) {
    if (typeof console !== 'undefined' && console && typeof console.error === 'function') {
      console.error('[replay] ' + label + ' hook threw (run continues):', err);
    }
  }
}

// ---------------------------------------------------------------------------
// loadGeneration(db, n, opts?)
//   db:    a Db-like object exposing getGeneration(n) / getDriverResults(n) /
//          getBestWeights(n). Not initialized / missing generation ⇒ typed throw.
//   n:     generation number, integer >= 1.
//   opts:  optional { useActiveTrack?: boolean (default false),
//                     activeTrack?: Track } — the LABELED OVERRIDE. Only when
//                     useActiveTrack === true do we use opts.activeTrack instead
//                     of the generations.track_json snapshot; the returned track
//                     and run are then marked useActiveTrack/trackSource='active-track'.
//                     The snapshot path NEVER silently falls back to the active
//                     track and vice versa.
// Returns { generation, results, bestWeights, track, stats, useActiveTrack, trackSource }.
// ---------------------------------------------------------------------------
function loadGeneration(db, n, opts) {
  if (!db ||
      typeof db.getGeneration !== 'function' ||
      typeof db.getDriverResults !== 'function' ||
      typeof db.getBestWeights !== 'function') {
    throw new ValidationError('loadGeneration requires a Db-like object with getGeneration(n) / getDriverResults(n) / getBestWeights(n)');
  }
  if (!Number.isInteger(n) || n < 1) {
    throw new ValidationError('loadGeneration: generation must be an integer >= 1, got ' + JSON.stringify(n));
  }
  var options = opts || {};
  if (options !== null && typeof options !== 'object') {
    throw new ValidationError('loadGeneration opts must be an object, got ' + typeof options);
  }
  var useActiveTrack = !!(options && options.useActiveTrack === true);

  var generation = db.getGeneration(n);
  if (!generation || typeof generation !== 'object') {
    throw new ValidationError('loadGeneration: generation ' + n + ' not found in the database');
  }

  var Track = requireReplayTrack();
  var NeuralNetwork = requireReplayNeuralNetwork();

  // ---- Track: SNAPSHOT by default; the active track ONLY on the labeled toggle.
  var track;
  var trackSource;
  if (useActiveTrack) {
    var activeTrack = options.activeTrack;
    if (!isTrackLike(activeTrack)) {
      throw new ValidationError('loadGeneration: opts.useActiveTrack is true but opts.activeTrack is not a Track-like object (startPose/nearestCenterline/raycast/segments/length) — refusing to guess');
    }
    track = activeTrack;
    trackSource = 'active-track';
  } else {
    var trackJson = generation.track_json;
    if (typeof trackJson !== 'string' || trackJson.length === 0) {
      throw new ValidationError('loadGeneration: generation ' + n + ' has a missing/empty track_json snapshot — refusing to replay');
    }
    try {
      track = Track.fromJSON(trackJson); // FULL validation (length cap, JSON parse, points, halfWidth)
    } catch (err) {
      if (err && err.name === 'ValidationError') {
        throw new ValidationError('loadGeneration: generation ' + n + ' track_json failed validation — ' + (err && err.message));
      }
      throw err;
    }
    // Cross-check the rebuilt snapshot against the generations.track_hash pin.
    // Both sides come from the same committed row, so a mismatch means the row
    // is corrupt (or the snapshot was hand-edited) — typed ValidationError.
    if (typeof generation.track_hash === 'string' && generation.track_hash.length > 0) {
      var actualHash = null;
      try {
        actualHash = track.hash();
      } catch (hashErr) {
        throw new ValidationError('loadGeneration: generation ' + n + ' track rebuilt but its hash could not be computed — ' + (hashErr && hashErr.message));
      }
      if (actualHash !== generation.track_hash) {
        throw new ValidationError('loadGeneration: generation ' + n + ' track_json does not match the committed track_hash pin (stored ' + JSON.stringify(generation.track_hash) + ', rebuilt ' + JSON.stringify(actualHash) + ') — the generation row is corrupt or was hand-edited; refusing to replay');
      }
    }
    trackSource = 'generation-snapshot';
  }

  // ---- Best weights: db.getBestWeights returns a VALIDATED NeuralNetwork
  //      instance (or throws ValidationError / returns null). We reduce it to
  //      the plain VALIDATED payload the caller can hand to createRun.
  var net = db.getBestWeights(n); // corrupt payload ⇒ typed ValidationError propagates
  if (!net || typeof net.getWeights !== 'function') {
    throw new ValidationError('loadGeneration: best_weights are missing for generation ' + n + ' — the generation cannot be replayed honestly');
  }
  var bestWeights = net.getWeights();
  // Belt and braces: re-validate the PLAIN object (wrong-count / non-finite /
  // architecture payload ⇒ ValidationError here, before any car is built).
  try {
    NeuralNetwork.fromJSON(bestWeights);
  } catch (weightsErr) {
    if (weightsErr && weightsErr.name === 'ValidationError') {
      throw new ValidationError('loadGeneration: generation ' + n + ' best weights failed validation — ' + (weightsErr && weightsErr.message));
    }
    throw weightsErr;
  }

  // ---- Per-driver results (may be an empty array for an unusual row).
  var results = db.getDriverResults(n);
  if (!Array.isArray(results)) results = [];

  // ---- Stats summary for the HUD/Logs panel. Distances/times are FITNESS
  //      (net signed odometer + lap bonus; may be negative) and simulated
  //      seconds, exactly as logged.
  var sumDistance = 0, sumTimeAlive = 0;
  var bestDistanceOverall = -Infinity, bestTimeAliveOverall = 0, bestTopSpeedOverall = 0;
  for (var i = 0; i < results.length; i++) {
    var r = results[i] || {};
    var d = Number(r.distance);
    var t = Number(r.time_alive);
    var ts = Number(r.top_speed);
    if (Number.isFinite(d)) {
      sumDistance += d;
      if (d > bestDistanceOverall) bestDistanceOverall = d;
    }
    if (Number.isFinite(t)) {
      sumTimeAlive += t;
      if (t > bestTimeAliveOverall) bestTimeAliveOverall = t;
    }
    if (Number.isFinite(ts) && ts > bestTopSpeedOverall) bestTopSpeedOverall = ts;
  }
  var resultCount = results.length;
  var bestDriverIndex = Number.isInteger(generation.best_driver_index) ? generation.best_driver_index : -1;
  var bestDriverResult = (bestDriverIndex >= 0 && bestDriverIndex < resultCount) ? results[bestDriverIndex] : null;
  var bestDistance = bestDriverResult && Number.isFinite(Number(bestDriverResult.distance))
    ? Number(bestDriverResult.distance)
    : bestDistanceOverall;
  var bestTimeAlive = bestDriverResult && Number.isFinite(Number(bestDriverResult.time_alive))
    ? Number(bestDriverResult.time_alive)
    : bestTimeAliveOverall;
  var bestTopSpeed = bestDriverResult && Number.isFinite(Number(bestDriverResult.top_speed))
    ? Number(bestDriverResult.top_speed)
    : bestTopSpeedOverall;

  var stats = {
    distance: bestDistance,                       // FITNESS of the SELECTED best driver (net signed + lap bonus; may be negative)
    bestDistance: bestDistance,
    avgDistance: resultCount > 0 ? sumDistance / resultCount : null,
    avgTimeAlive: resultCount > 0 ? sumTimeAlive / resultCount : 0,
    bestTimeAlive: bestTimeAlive,                 // simulated seconds lived by the best driver
    bestTopSpeed: bestTopSpeed,                   // px/s
    // fitness fields — Number.isFinite guards (fitness may be NEGATIVE; ||0 would mask both NaN and a negative)
    bestFitness: Number.isFinite(Number(generation.best_fitness)) ? Number(generation.best_fitness) : null,
    avgFitness: Number.isFinite(Number(generation.avg_fitness)) ? Number(generation.avg_fitness) : null,
    worstFitness: Number.isFinite(Number(generation.worst_fitness)) ? Number(generation.worst_fitness) : null,
    bestDriverIndex: bestDriverIndex,
    driversEvaluated: Number.isInteger(generation.drivers_evaluated) ? generation.drivers_evaluated : resultCount,
    simSeconds: Number.isFinite(Number(generation.sim_seconds)) ? Number(generation.sim_seconds) : 0,
    mutationRate: Number.isFinite(Number(generation.mutation_rate)) ? Number(generation.mutation_rate) : 0,
    mutationScale: Number.isFinite(Number(generation.mutation_scale)) ? Number(generation.mutation_scale) : 0,
    trackHash: typeof generation.track_hash === 'string' ? generation.track_hash : null,
    createdAt: generation.created_at != null ? String(generation.created_at) : null,
  };

  return {
    generation: generation,
    results: results,
    bestWeights: bestWeights,   // plain validated { architecture, w1..b3 } — JSON-safe
    track: track,               // Track built from the generation SNAPSHOT (or the active track on the labeled toggle)
    stats: stats,
    useActiveTrack: useActiveTrack,
    trackSource: trackSource,
  };
}

// ---------------------------------------------------------------------------
// createRun(opts)
//   opts: { track, weights, onFrame?, onFinish?, onBanner?,
//           useActiveTrack?: boolean (default false),
//           generation?: number|null,
//           trackSource?: string }
//     track:   Track-like object (usually the snapshot from loadGeneration).
//     weights: plain validated weights object OR a JSON string — parsed and
//              validated through NeuralNetwork.fromJSON. A wrong-count payload
//              throws ValidationError HERE, before any car exists.
//   Returns { start(), stop(), step(dtSimMs), state, useActiveTrack, trackSource }.
//     state is a live object: { status: 'ready'|'running'|'finished'|'stopped',
//       ticks, simSeconds, timeAlive, odometer, topSpeed, outcome,
//       useActiveTrack, trackSource, generation }.
//     (state.odometer is the RAW net signed odometer, not fitness — the HUD
//       shows net px, labeled "Odometer (net)".)
//     step(dtSimMs) — advance the simulation by dtSimMs MILLISECONDS of SIM
//       time, in SIM_DT quanta; used by headless/node checks (rAF absent). It
//       also renders one frame via onFrame at the end of the step.
//     stop() — halt immediately; cancels any pending rAF; does NOT fire
//       onFinish; the run cannot be restarted (create a new run instead).
// ---------------------------------------------------------------------------
function createRun(opts) {
  var options = opts || {};
  if (typeof options !== 'object' || Array.isArray(options)) {
    throw new ValidationError('createRun requires an options object, got ' + (opts === null ? 'null' : typeof opts));
  }

  var track = options.track;
  if (!isTrackLike(track)) {
    throw new ValidationError('createRun requires a Track-like object (startPose/nearestCenterline/raycast/segments/length) as opts.track');
  }
  var weights = options.weights;
  if (weights === undefined || weights === null) {
    throw new ValidationError('createRun requires opts.weights (plain weights object or JSON string)');
  }
  var NeuralNetwork = requireReplayNeuralNetwork();
  var DriverSim = requireDriverSim();

  // Validation happens HERE — a wrong-weight-count payload throws before any
  // DriverSim exists. fromJSON handles both the plain object and the
  // JSON-string forms (the string length cap included).
  var net = NeuralNetwork.fromJSON(weights);

  var onFrame = typeof options.onFrame === 'function' ? options.onFrame : null;
  var onFinish = typeof options.onFinish === 'function' ? options.onFinish : null;
  var onBanner = typeof options.onBanner === 'function' ? options.onBanner : null;

  var useActiveTrack = options.useActiveTrack === true; // DEFAULT FALSE — labeled toggle only
  var trackSource = useActiveTrack
    ? 'active-track'
    : (typeof options.trackSource === 'string' && options.trackSource ? options.trackSource : 'generation-snapshot');
  var generationNumber = Number.isInteger(options.generation) ? options.generation : null;

  // One DriverSim instance owns the Car, the SensorFan and the liveness
  // state — the tick order, the arcPos seam unwrap, the stagnation gate and
  // the kill caps are exactly the DriverSim contract (js/driver-sim.js).
  var sim = new DriverSim(track, net, null);
  var car = sim.car;
  var simDt = sim.simDt;

  var ticks = 0;
  var finished = false;
  var finishSent = false;
  var accumulatorSeconds = 0;
  var lastTimestamp = null;
  var rafHandle = null;

  var state = {
    status: 'ready', // 'ready' | 'running' | 'finished' | 'stopped'
    ticks: 0,
    simSeconds: 0,
    timeAlive: 0,
    odometer: 0,
    topSpeed: 0,
    outcome: null,   // 'crashed' | 'stuck' | 'time-cap' | 'finished' | 'error' | null
    useActiveTrack: useActiveTrack,
    trackSource: trackSource,
    generation: generationNumber,
  };

  function snapshotState() {
    state.status = finished ? 'finished' : state.status;
    state.ticks = ticks;
    state.simSeconds = sim.timeAlive;
    state.timeAlive = sim.timeAlive;
    state.odometer = car.odometer;
    state.topSpeed = sim.topSpeed();
    return state;
  }

  function renderFrame() {
    if (!onFrame) return;
    safeHook(onFrame, {
      x: car.x,
      y: car.y,
      heading: car.heading,
      speed: car.speed,
      odometer: car.odometer,
    }, 'onFrame');
  }

  function finish(outcome) {
    if (finishSent) return;
    finished = true;
    finishSent = true;
    state.status = 'finished';
    state.outcome = outcome;
    snapshotState();
    safeHook(onFinish, {
      distance: sim.fitness(),     // fitness — net signed odometer + lap bonus; may be negative
      timeAlive: sim.timeAlive,
      outcome: outcome,             // 'crashed' | 'stuck' | 'time-cap' | 'finished' | 'error'
      topSpeed: sim.topSpeed(),
      ticks: ticks,
      useActiveTrack: useActiveTrack,
      trackSource: trackSource,
      generation: generationNumber,
    }, 'onFinish');
  }

  // One SIM_DT tick — delegated to the DriverSim (the exact tick order,
  // integration order and arcPos seam unwrap of the training eval path;
  // kill caps included). DriverSim.step() is idempotent once terminal.
  // A non-finite [gas, steer] throws ValidationError out of sim.step() —
  // callers route it into the 'error' outcome (see drainAccumulator / frame).
  function tick() {
    var res = sim.step();
    ticks += 1;
    if (res.done) { finish(res.outcome); return; }
    snapshotState();
  }

  function drainAccumulator() {
    // Drain in simDt quanta; never spiral (accumulator is clamped by the
    // frame-delta clamp in _frame; step() adds caller-controlled amounts).
    while (accumulatorSeconds >= simDt && !finished) {
      accumulatorSeconds -= simDt;
      try {
        tick();
      } catch (err) {
        // A tick fault (e.g. the network produced non-finite [gas, steer])
        // must not escape as an uncaught rAF error (fail-loud in the console,
        // dead run, nothing painted). Report through onBanner — the HUD shows
        // it — and finish with outcome 'error'; renderFrame() then paints the
        // frozen pose.
        safeHook(onBanner, { type: 'error', message: String(err && err.message || err) }, 'onBanner');
        finish('error');
        return;
      }
      if (finished) break;
    }
    if (finished) {
      // Once finished the sub-tick remainder is meaningless.
      accumulatorSeconds = 0;
    }
  }

  function hasRaf() {
    return typeof requestAnimationFrame === 'function' && typeof cancelAnimationFrame === 'function';
  }

  function frame(timestamp) {
    rafHandle = null;
    if (finished || state.status !== 'running') return;
    var t = (typeof timestamp === 'number' && Number.isFinite(timestamp)) ? timestamp : null;
    if (t === null) {
      t = (typeof performance !== 'undefined' && performance && typeof performance.now === 'function')
        ? performance.now()
        : 0;
    }
    if (lastTimestamp === null) lastTimestamp = t;
    var deltaMs = t - lastTimestamp;
    lastTimestamp = t;
    if (!(deltaMs > 0)) deltaMs = 0;
    if (deltaMs > MAX_FRAME_DELTA_MS) deltaMs = MAX_FRAME_DELTA_MS;
    accumulatorSeconds += deltaMs / 1000;
    drainAccumulator(); // faults surface here are already routed to the 'error' outcome
    renderFrame();
    if (!finished && state.status === 'running' && hasRaf()) {
      rafHandle = requestAnimationFrame(frame);
    }
  }

  function start() {
    if (state.status === 'finished' || state.status === 'stopped') return run; // terminal states are sticky
    if (state.status === 'running') return run;                                // idempotent
    state.status = 'running';
    if (useActiveTrack) {
      safeHook(onBanner, {
        type: 'use-active-track',
        persistent: true,
        useActiveTrack: true,
        trackSource: trackSource,
        message: 'Using active track instead of the generation snapshot — results will differ',
      }, 'onBanner');
    }
    snapshotState();
    renderFrame(); // paint the spawn pose even before the first sim tick
    if (hasRaf()) {
      rafHandle = requestAnimationFrame(frame);
    }
    return run;
  }

  function stop() {
    if (finished) return run;
    if (rafHandle !== null && typeof cancelAnimationFrame === 'function') {
      try { cancelAnimationFrame(rafHandle); } catch (err) { /* nothing useful to do */ }
    }
    rafHandle = null;
    state.status = 'stopped';
    snapshotState();
    return run;
  }

  // step(dtSimMs) — advance the simulation by dtSimMs MILLISECONDS of SIM
  // time (e.g. step(1000 / 120), or step(45000) to fast-forward a whole
  // 45 s cap). Used by headless/node checks where rAF does not exist; works
  // in the browser too. Fires onFrame once at the end of the step.
  function step(dtSimMs) {
    if (typeof dtSimMs !== 'number' || !Number.isFinite(dtSimMs) || dtSimMs < 0) {
      throw new ValidationError('step(dtSimMs) requires a finite number >= 0 (milliseconds of SIM time), got ' + String(dtSimMs));
    }
    if (finished || state.status === 'stopped') {
      snapshotState();
      return state;
    }
    if (state.status === 'ready') state.status = 'running'; // stepping implies running (headless driver)
    accumulatorSeconds += dtSimMs / 1000;
    drainAccumulator();
    renderFrame();
    snapshotState();
    return state;
  }

  var run = {
    start: start,
    stop: stop,
    step: step,
    state: state,
    useActiveTrack: useActiveTrack,
    trackSource: trackSource,
  };
  return run;
}

// ---------------------------------------------------------------------------
// Exports — browser global `Replay` on globalThis only (no window mirror);
// guarded CommonJS export for node checks (tests/smoke convention). The
// plan-pinned constants are owned by js/driver-sim.js — nothing outside
// driver-sim.js consumes them via the Replay namespace (zero users tree-wide).
// ---------------------------------------------------------------------------
var Replay = {
  loadGeneration: loadGeneration,
  createRun: createRun,
};

if (typeof globalThis !== 'undefined') {
  globalThis.Replay = Replay;
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    Replay: Replay,
    ValidationError: ValidationError,
  };
}
