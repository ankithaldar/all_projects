// js/train-worker.js — Headless evaluation Web Worker (the ONLY file in the
// project permitted to call importScripts; see plan L-NEW8/SEC-11).
//
// importScripts ARGS ARE './'-RELATIVE LITERALS, nothing else — no remote
// URLs, no absolute paths, no variables. Grep-gates pin this. The seven-file
// importScripts list is fixed; three load-time constraints govern the order:
//   ./driver-sim.js (loads FIRST — owns the eval constants (SIM_DT/EVAL_CAP_S/
//                  STUCK_TIMEOUT_S/PROGRESS_ADVANCE_PX) and the DriverSim;
//                  resolves its sibling deps lazily at construction, not at
//                  load time)
//   ./nn.js       (loads early; no dependencies)
//   ./track.js    (loads before car.js — car.js requires globalThis.Track at LOAD time)
//   ./rotation.js(no load-time requirements — pure logic with guarded
//                  CommonJS export; resolves canonicalJSON/cyrb53 by global
//                  lookup at call time, i.e. after track.js loads)
//   ./car.js      (requires track.js)
//   ./sensors.js  (no load-time requirements; loads before evolution.js)
//   ./evolution.js(requires mulberry32 / NeuralNetwork / Car / SensorFan /
//                  DriverSim / strictUint32 / the driver-sim.js constants at
//                  LOAD time — nn.js, car.js and driver-sim.js have set them)
//
// SECURITY / SCOPE: no DOM (none exists in a worker anyway), no innerHTML, no
// eval / new Function, no fetch/XMLHttpRequest/WebSocket/import() — the worker
// computes physics and ships results. Persistence and logging are the MAIN
// THREAD's job (trainer.js) — the worker never touches the DB.
//
// MESSAGE PROTOCOL ( structured clone; all messages are plain objects )
//   IN  { type:'init',   controlPoints, halfWidth, seed, rotationSeed }
//         → builds the Track (full validation), spawns gen 1 from `seed`,
//           re-pins the block cache to rotation 0 (rotation 0 is the pinned
//           track; rotation >= 1 is generated from (rotationSeed, ri) once
//           per 50-generation block by trackForGeneration())
//         OUT { type:'ready', generation: 1 }
//   IN  { type:'resume', controlPoints, halfWidth, bestWeights, rngState, generation, rotationSeed }
//         → builds the Track; loads `bestWeights` through NeuralNetwork.fromJSON
//           (FULL validation); constructs gen N+1 via spawnNextGeneration
//           semantics from the restored rngState (see Evolution.resumeFromBest);
//           the caller passes the COMPLETED generation N — the worker produces
//           gen N+1. rotationSeed is REQUIRED (uint32) — a missing/zero invented
//           seed never silently passes (D-11). OUT { type:'ready', generation: N + 1 }
//   IN  { type:'start',  maxGenerations }
//         → evaluates the current generation, driver by driver, YIELDING to the
//           worker event loop BETWEEN drivers (setTimeout(0)) so a queued
//           'stop' is observed mid-generation. After all 20 drivers:
//         OUT { type:'generation', payload:{ generation, results, bestIndex,
//                                            bestWeights, rngState,
//                                            rotationIndex, difficulty, trackHash } }
//         …then WAITS for IN {type:'next'} (main thread logs to the DB first)
//         …or IN {type:'stop'}. If evo.generation >= maxGenerations the 'next'
//         posts OUT { type:'done', generation } instead of spawning again.
//   IN  { type:'next' } → spawn the next generation (unless stopRequested or
//                         maxGenerations reached → 'done'), evaluate, post
//                         'generation' again. `next` while no generation is in
//                         flight → { type:'error', … }.
//   IN  { type:'stop' } → sticky stopRequested flag; observed BETWEEN DRIVERS
//                         (mid-generation abort) and BETWEEN GENERATIONS
//                         (in 'next' / 'stop' at waitingForNext). A generation
//                         aborted this way is DISCARDED — it was never posted,
//                         so the main thread never logs it; the committed
//                         lineage prefix stays consistent and resume recomputes
//                         the aborted generation identically.
//         OUT { type:'stopped', generation }  (generation = evo.generation at
//                         abort time — the discarded or the last completed one)
//   Any handler throw → OUT { type:'error', message } — the worker survives.
//
// FIT-V3 — ROTATION (per G1): handleStart/handleNext RE-DERIVE trackFor(generation)
// PER GENERATION (never cache state.track at function entry): rotation 0 is the
// pinned starting track carried by the handshake; rotation >= 1 is built once
// per block from resolveGeneratedTrack(rotationSeed, ri) (never re-built within
// a block, never stale across a boundary). Progress payloads carry
// { rotationIndex, difficulty, trackHash } — small numbers/strings only; the
// main thread re-derives the BYTES (same engine + inputs ⇒ identical bytes).
//
// rngState in the payload is captured at the END of generation N's spawn
// (i.e. the cursor an uninterrupted run holds right before gen N+1's
// mutations) — exactly what resume needs to re-derive gen N+1 byte-identically.

'use strict';

importScripts('./driver-sim.js', './nn.js', './track.js', './rotation.js', './car.js', './sensors.js', './evolution.js');

// ---------------------------------------------------------------------------
// Worker-local state. Exactly one track + one Evolution lineage per worker.
// ---------------------------------------------------------------------------
var state = {
  track: null,            // the CURRENT BLOCK's Track (re-pinned per block — see trackForGeneration)
  pinnedTrack: null,      // the handshake-built Track — rotation 0 is the lineage's pinned starting track
  rotationSeed: 0,        // from the init/resume handshake — written once, never rewritten
  generatorVersion: null, // FIT-V4 — canonical integer version from the strict init/resume handshake; REQUIRED, never defaulted
  currentRotationIndex: null, // the block the current state.track belongs to (null = force a rebuild)
  evolution: null,
  running: false,          // a generation loop is in flight
  waitingForNext: false,   // 'generation' posted, awaiting 'next' or 'stop'
  stopRequested: false,    // sticky until the next init/resume/start
  maxGenerations: 0,
};

function post(msg) { postMessage(msg); }
function errText(err) { return String((err && err.message) || err); }

// strict(value, name) — delegate to nn.js's strictUint32 (the ONE strict rule
// for uint32 state/cursors; see nn.js). Refusing to run with a clear error is
// better than a silently coerced resume cursor.
function strict(value, name) {
  var G = typeof globalThis !== 'undefined' ? globalThis : self;
  if (typeof G.strictUint32 !== 'function') {
    throw new ValidationError('strictUint32 is not available — nn.js must load before train-worker.js');
  }
  return G.strictUint32(value, name);
}

function rotationApi() {
  var G = typeof globalThis !== 'undefined' ? globalThis : self;
  if (!G.Rotation || typeof G.Rotation.rotationForGeneration !== 'function') {
    throw new ValidationError('Rotation is not available — js/rotation.js must load before train-worker.js runs');
  }
  return G.Rotation;
}

// requireVersion(raw) — the message/state boundary gate. The canonical strict
// parser lives in rotation.js (Task A); a missing or empty handshake field is
// malformed and throws, NEVER a default. Called in BOTH init and resume before
// any worker state is committed.
function requireVersion(raw) {
  var R = rotationApi();
  if (typeof R.requireGeneratorVersion !== 'function') {
    throw new ValidationError('Rotation.requireGeneratorVersion is missing — js/rotation.js version contract not loaded');
  }
  return R.requireGeneratorVersion(raw);
}

// trackForGeneration(generation) — the CURRENT BLOCK's Track. Rebuilt at
// most once per 50-generation block (cached on state.currentRotationIndex),
// NEVER re-built within a block, NEVER stale across a boundary. Rotation 0 is
// the lineage's pinned starting track (the handshake track); rotation >= 1 is
// the generated curriculum track resolved from (rotationSeed, ri) — byte-
// identical to the pre-materialized tracks rows because both run the same
// resolver on the same inputs.
function trackForGeneration(generation) {
  var R = rotationApi();
  var ri = R.rotationForGeneration(generation);
  if (state.currentRotationIndex !== ri) {
    if (ri === 0) {
      state.track = state.pinnedTrack;
    } else {
      var resolved = R.resolveGeneratedTrack(state.rotationSeed, ri, state.generatorVersion);
      state.track = new Track(resolved.controlPoints, resolved.halfWidth);
    }
    state.currentRotationIndex = ri;
  }
  return state.track;
}

function handleInit(msg) {
  var seed = strict(msg.seed, 'init seed'); // strict rule (nn.js) — hex/exponent strings rejected
  var rotationSeed = strict(msg.rotationSeed, 'init rotationSeed'); // FIT-V3 — uint32, never an invented zero
  var generatorVersion = requireVersion(msg.generatorVersion); // FIT-V4 — strict; missing/empty throws, never defaults
  var track = new Track(msg.controlPoints, msg.halfWidth); // full validation
  var evo = new Evolution(seed);
  evo.randomPopulation();
  state.pinnedTrack = track;
  state.track = track;
  state.rotationSeed = rotationSeed;
  state.generatorVersion = generatorVersion;
  state.currentRotationIndex = null; // force a block rebuild at the first generation
  state.evolution = evo;
  state.running = false;
  state.waitingForNext = false;
  state.stopRequested = false;
  state.maxGenerations = 0;
  post({ type: 'ready', generation: evo.generation }); // = 1
}

function handleResume(msg) {
  if (msg.bestWeights === undefined || msg.bestWeights === null) {
    throw new ValidationError('resume requires bestWeights (payload object or JSON string)');
  }
  if (!Number.isInteger(msg.generation) || msg.generation < 1) {
    throw new ValidationError('resume requires generation: integer >= 1 (the COMPLETED generation), got ' + String(msg.generation));
  }
  // H-1 strict gate FIRST, before any construction: only integer numbers in
  // [0, 4294967295] or decimal TEXT /^(0|[1-9][0-9]{0,9})$/ ≤ 4294967295 are
  // accepted — null, booleans, '', '0x10', '1e3', NaN, floats, hex/exponent
  // strings, out-of-range text, arrays and objects all throw here.
  var rngState = strict(msg.rngState, 'resume rngState');
  var rotationSeed = strict(msg.rotationSeed, 'resume rotationSeed'); // FIT-V3 — uint32, never an invented zero
  var generatorVersion = requireVersion(msg.generatorVersion); // FIT-V4 — strict; missing/empty throws, never defaults
  var track = new Track(msg.controlPoints, msg.halfWidth); // full validation
  // resumeFromBest re-validates bestWeights via NeuralNetwork.fromJSON and
  // rngState via the same strict rule before the population exists — a bad
  // payload leaves the worker with NO state instead of a corrupt one.
  var evo = Evolution.resumeFromBest(msg.bestWeights, msg.generation, rngState);
  state.pinnedTrack = track;
  state.track = track;
  state.rotationSeed = rotationSeed;
  state.generatorVersion = generatorVersion;
  state.currentRotationIndex = null; // force a block rebuild at the first generation
  state.evolution = evo;
  state.running = false;
  state.waitingForNext = false;
  state.stopRequested = false;
  state.maxGenerations = 0;
  post({ type: 'ready', generation: evo.generation }); // = N + 1
}

function finishStopped() {
  state.running = false;
  state.waitingForNext = false;
  state.stopRequested = false;
  post({ type: 'stopped', generation: state.evolution ? state.evolution.generation : 0 });
}

// ---- one generation, driver by driver, yielding between drivers ------------
function startGenerationEvaluation() {
  var evo = state.evolution;
  // RE-DERIVE the block's Track PER GENERATION (G1 — never cache state.track
  // at function entry across a boundary): gens 1..50 ride the pinned track,
  // gens >= 51 ride the generated curriculum track. Rebuilt at most once per
  // block; never within a block.
  var generation = evo.generation;
  var track = trackForGeneration(generation);
  var population = evo.population;
  var results = new Array(population.length);
  var i = 0;

  function step() {
    try {
      // Stop checked BETWEEN drivers (see header). A mid-generation abort
      // discards the whole generation: nothing is posted, so the main thread
      // never logs it and the lineage stays a consistent prefix.
      if (state.stopRequested) { finishStopped(); return; }
      results[i] = Evolution.evaluateDriver(track, population[i]);
      i += 1;
      if (i < population.length) { setTimeout(step, 0); return; }
      var bestIndex = evo.recordEvaluation(results);
      state.waitingForNext = true;
      var R = rotationApi();
      var ri = R.rotationForGeneration(evo.generation);
      post({
        type: 'generation',
        payload: {
          generation: evo.generation,
          results: results,
          bestIndex: bestIndex,
          bestWeights: evo.population[bestIndex].getWeights(),
          rngState: evo.rngState(),
          rotationIndex: ri,
          difficulty: R.difficultyForRotation(ri),
          trackHash: track.hash(),
        },
      });
    } catch (err) {
      state.running = false;
      state.waitingForNext = false;
      state.stopRequested = false;
      post({ type: 'error', message: errText(err) });
    }
  }
  setTimeout(step, 0);
}

function handleStart(msg) {
  if (!state.evolution || !state.track) {
    post({ type: 'error', message: 'start requires a prior init or resume' });
    return;
  }
  var maxGen = msg.maxGenerations;
  if (typeof maxGen !== 'number' || !Number.isFinite(maxGen) || Math.floor(maxGen) !== maxGen || maxGen < 1) {
    post({ type: 'error', message: 'start requires maxGenerations: integer >= 1, got ' + String(maxGen) });
    return;
  }
  if (state.running || state.waitingForNext) {
    post({ type: 'error', message: 'start ignored — a generation loop is already in flight' });
    return;
  }
  state.maxGenerations = maxGen;
  state.stopRequested = false;
  if (state.evolution.generation > maxGen) {
    // Nothing left to compute at this cap (e.g. start after done) — honor it.
    post({ type: 'done', generation: state.evolution.generation });
    return;
  }
  state.running = true;
  startGenerationEvaluation();
}

function handleNext() {
  if (!state.evolution) { post({ type: 'error', message: 'next requires a prior init or resume' }); return; }
  if (!state.waitingForNext) { post({ type: 'error', message: 'next received but no generation is awaiting the logging handshake' }); return; }
  // Stop checked BETWEEN generations.
  if (state.stopRequested) { finishStopped(); return; }
  state.waitingForNext = false;
  if (state.evolution.generation >= state.maxGenerations) {
    state.running = false;
    post({ type: 'done', generation: state.evolution.generation });
    return;
  }
  state.evolution.spawnNextGeneration(); // evo.generation += 1
  startGenerationEvaluation();
}

self.onmessage = function (event) {
  var msg = event && event.data ? event.data : {};
  try {
    switch (msg.type) {
      case 'init':   handleInit(msg); break;
      case 'resume': handleResume(msg); break;
      case 'start':  handleStart(msg); break;
      case 'next':   handleNext(); break;
      case 'stop':
        state.stopRequested = true;
        if (state.waitingForNext || !state.running) {
          // Between generations (or idle): stop right here, idempotently.
          // Mid-generation: the step() loop observes the flag on the NEXT
          // driver boundary — messages cannot preempt a synchronous tick.
          finishStopped();
        }
        break;
      default:
        post({ type: 'error', message: 'unknown message type: ' + String(msg.type) });
    }
  } catch (err) {
    // Protocol errors (bad payloads, validation failures) must not kill the
    // worker — report and keep serving.
    post({ type: 'error', message: errText(err) });
  }
};
