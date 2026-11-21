// js/trainer.js — Trainer: main-thread controller for headless training
// (0003_car_racer, M2 scope: headless only; spectate/render lands in M4).
//
// Responsibilities:
//   * spawn the evaluation backend — a real Web Worker running
//     js/train-worker.js, OR a MessageChannel self-rescheduling fallback that
//     runs the SAME protocol in-process, yielding between drivers;
//   * run the init/resume → generation×N handshake;
//   * call opts.logger.logGeneration(payload) once per generation, inside the
//     LOG-FAILURE degrade policy (see below);
//   * invoke the optional opts.onPersist hook per logged generation (M3 wires
//     Db.persist here; during a logging-disabled session it is SKIPPED with a
//     distinct banner, per plan SEC-26);
//   * report progress via opts.onProgress; surface banners via opts.onBanner;
//   * enforce the track_hash lineage guard at start (M3 wires the real hash
//     source; M2 exposes the hook);
//
// LOG-FAILURE DEGRADE (Lead decision; SEC-24 carve-out — see plan LOG-FAILURE):
//   logger.logGeneration() throw:
//     * UNIQUE/PRIMARY-KEY lineage-integrity conflict
//         → transaction ROLLED BACK (the Db does this), red banner
//           "LINEAGE INTEGRITY ERROR — RUN HALTED — committed lineage untouched",
//           the run HALTS (state 'halted'), the worker is stopped. Committed
//           lineage is NEVER rewritten.
//     * any other throw (disk, quota, non-lineage constraint)
//         → rolled back; LOGGING DISABLED for the REST OF THIS SESSION; red
//           banner "GENERATION N NOT LOGGED — … — LOGGING DISABLED THIS
//           SESSION — …"; training CONTINUES unlogged; persist hook skipped.
//   isUniqueViolation(err) is exported so M3's resume-UNIQUE-conflict path and
//   the adversarial tests classify identically.
//
// PERSIST HOOK: opts.onPersist(payload) is called once per LOGGED generation
// (never inside a logging-disabled session — the skip is deliberate and the
// banner says so). A THROW from onPersist follows the LOG-FAILURE policy
// exactly like a logger throw: logging/persist are disabled for the rest of
// the session and the banner carries the same plan wording. In M2 there is no
// DB, so the hook is only fired when a caller supplies it (M3 passes
// Db.persist-bound closure here).
//
// BACKEND FAILURE MATRIX (who reacts, in which order — see _onWorkerError):
//   1. sync spawn throw     — `new Worker(url)` throws at construction
//      (file:// origin, sandbox, CSP worker-src) — or, under the test seam,
//      `opts.workerFactory(url)` throws/returns falsy. The MessageChannel
//      fallback is created transparently and the session starts on it;
//      construction throws only if BOTH backends are unavailable. No banner
//      for the normal fallback (it is a supported configuration) —
//      `this._fallbackReason` records why.
//   2. async onerror pre-first-commit — the Worker errors before ANY
//      generation was committed this session (generationsLogged === 0), even
//      if a 'ready' WAS received. Reaction: terminate() the broken worker
//      FIRST, then tear down and respawn the lineage EXACTLY ONCE on the
//      MessageChannel fallback, replay the stored init/resume + start
//      handshake, and fire onBanner({ type:'worker-failed-fallback' }). The
//      fresh fallback re-derives the lineage from the same seed/rngState —
//      byte-identical by construction. (Amended M3-fix, arch F-4: post-ready
//      pre-first-commit ⇒ respawn, was: halt — integrity is unchanged because
//      no commit happened.)
//   3. async onerror post-first-commit — the Worker errors after a generation
//      WAS logged (generationsLogged > 0). Reaction: terminate() FIRST,
//      then state 'halted' and the extended LINEAGE banner
//      ("LINEAGE INTEGRITY ERROR — RUN HALTED — committed lineage untouched
//       — reload and resume recomputes the unlogged generation identically").
//      The lineage is NEVER re-run from scratch after a partial commit.
//   4. fallback failure      — the fallback transport cannot be built (no
//      MessageChannel/setTimeout) or the respawn attempt itself throws.
//      Reaction: state 'halted' and the SAME extended LINEAGE banner as (3).
//   Any { type:'error' } message from a backend follows row (4)'s handling —
//   it is the async failure surface for protocol/runtime throws inside the
//   backend, and it is treated as terminal.
//   onBanner / onProgress throwing can never corrupt this flow: both are
//   try/caught at the call site and logged via console.error (see _banner /
//   _emitProgress) — the protocol continues, and a throwing UI can never
//   swallow the 'next' post.
//
// HALTED is terminal: start() refuses to run (throws), 'stopped'/'done'
// backend acks cannot move a halted trainer out of 'halted'.
//
// DOM-free (no document/window/canvas/innerHTML). References Worker,
// MessageChannel, crypto — all guarded; never required to exist at load time.
// Plain classic script; browser: window.Trainer; node: guarded module.exports
// (createFallbackCore is exported as a test seam for the smoke suite; it is
// NOT a public API and may change without notice).

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

// ---------------------------------------------------------------------------
// isUniqueViolation(err) — classify a logger/DB throw. DELEGATES to the
// shared lineage-integrity classifier exported by js/db.js
// (globalThis.isLineageIntegrityFault — arch F-2 / security M2) so the two
// never drift. The inline copy below is the fallback for sessions that load
// trainer.js WITHOUT db.js (M2 loads no db.js); it is kept in lockstep with
// db.js's classifier and tests/smoke.mjs leg 28 probes BOTH paths against
// the same battery of messages. sql.js (the vendored sqlite in M3) surfaces
// constraint violations as plain Error with a message like
// "UNIQUE constraint failed: generations.generation" (and, depending on
// the offending table, "PRIMARY KEY constraint failed: …" for rowid-alias
// PKs). Both mean the SAME thing for this codebase: a lineage-integrity
// conflict — never eligible for continue-unlogged. M3's injection tests pin
// these strings against the REAL vendored sqlite.
// ---------------------------------------------------------------------------
function isUniqueViolation(err) {
  var shared = typeof globalThis !== 'undefined' ? globalThis.isLineageIntegrityFault : undefined;
  if (typeof shared === 'function') return shared(err);
  if (!err) return false;
  if (err.code === 'SQLITE_CONSTRAINT_UNIQUE' || err.code === 'SQLITE_CONSTRAINT_PRIMARYKEY') return true;
  if (err.code === 'FIT_V3_ROTATION_DIVERGENCE') return true; // lockstep with js/db.js's classifier — the M2 fallback keeps parity
  var message = String((err && err.message) || err);
  return /\b(unique|primary key)\s+constraint\s+failed/i.test(message);
}

// FIT-V3 — the rotationSeed gate for start(). BOTH lineage kinds require an
// explicit uint32 rotationSeed (D-11): for fresh, the seed passed at lineage
// creation (the same value ensureLineage persists as meta.rotation_seed); for
// resume, resumeState().rotationSeed. A caller that passes neither — or a
// garbage value — throws a typed ValidationError BEFORE any backend handshake
// goes out. Never an invented zero seed.
function strictLineageSeed(value, name) {
  var G = typeof globalThis !== 'undefined' ? globalThis : {};
  if (typeof G.strictUint32 !== 'function') {
    throw new ValidationError('strictUint32 is not available — js/nn.js must load before the trainer starts');
  }
  return G.strictUint32(value, name);
}

// FIT-V4 — the generator version is REQUIRED on every lineage handed to
// start(). The ONE canonical strict parser lives in rotation.js (Task A); this
// is a thin guarded reference, never a redefinition or a silent default. A
// missing/empty field throws a typed ValidationError before any handshake.
function strictLineageVersion(value) {
  var G = typeof globalThis !== 'undefined' ? globalThis : {};
  if (!G.Rotation || typeof G.Rotation.requireGeneratorVersion !== 'function') {
    throw new ValidationError('Rotation.requireGeneratorVersion is not available — js/rotation.js must load before the trainer starts');
  }
  return G.Rotation.requireGeneratorVersion(value);
}

// ---------------------------------------------------------------------------
// createFallbackCore() — the MessageChannel self-rescheduling fallback.
//
// When `new Worker('js/train-worker.js')` throws — file:// origins (Chromium
// refuses workers from origin 'null'), offline CI without web support, or an
// environment with no Worker at all — the Trainer transparently falls back to
// running the SAME protocol in-process, chunked by setTimeout(0) so the event
// loop keeps breathing between drivers. The rescheduling loop never uses
// rAF (training must proceed when the tab is backgrounded — same contract as
// the Worker path).
//
// This is a DELIBERATE, commented duplication of js/train-worker.js's core:
// the worker cannot import the main page's scripts (and vice versa) without
// eval/import tricks that the plan forbids. Both paths implement the same
// protocol and the same driver-loop/yield discipline; keep them in sync —
// M4 exercises the real Worker path in an actual browser.
// ---------------------------------------------------------------------------
function createFallbackCore() {
  var G = typeof globalThis !== 'undefined' ? globalThis : {};
  var Evolution = G.Evolution;
  var Track = G.Track;
  var ValidationErrorG = G.ValidationError || ValidationError;

  var state = {
    track: null,
    pinnedTrack: null,
    rotationSeed: 0,
    generatorVersion: null, // FIT-V4 — canonical integer from the strict init/resume handshake; REQUIRED
    currentRotationIndex: null,
    evolution: null,
    running: false,
    waitingForNext: false,
    stopRequested: false,
    maxGenerations: 0,
  };

  // Same strict-uint32 gate as the worker (nn.js strictUint32) — both cores
  // enforce the identical rule; see the failure matrix in the trainer header.
  function strictFn(value, name) {
    if (typeof G.strictUint32 !== 'function') {
      throw new ValidationErrorG('strictUint32 is not available — nn.js must load before the fallback core runs');
    }
    return G.strictUint32(value, name);
  }

  // Rotation resolution mirrors db.js's requireRotation(): the browser path
  // reads globalThis.Rotation (script order pins rotation.js before trainer.js);
  // in node the suite may not have required js/rotation.js yet when the Trainer
  // legs run, so a guarded CommonJS fallback loads the sibling file. NEVER a
  // silent default — absence throws.
  var _rotationModuleCache = null;
  function rotationApi() {
    if (_rotationModuleCache) return _rotationModuleCache;
    var g = (typeof globalThis !== 'undefined') ? globalThis : {};
    if (g.Rotation && typeof g.Rotation.rotationForGeneration === 'function') {
      _rotationModuleCache = g.Rotation;
      return _rotationModuleCache;
    }
    if (typeof require === 'function' && typeof __filename === 'string') {
      var path = require('node:path');
      var rotationModule = require(path.join(path.dirname(__filename), 'rotation.js'));
      if (rotationModule && rotationModule.Rotation && typeof rotationModule.Rotation.rotationForGeneration === 'function') {
        _rotationModuleCache = rotationModule.Rotation;
        return _rotationModuleCache;
      }
    }
    throw new ValidationErrorG('Rotation is not available — js/rotation.js must load before the fallback core runs');
  }

  // requireVersion(raw) — the message/state boundary gate for this core. The
  // canonical strict parser lives in rotation.js; a missing/empty handshake
  // field throws, never a default. Kept in lockstep with train-worker.js.
  function requireVersion(raw) {
    var R = rotationApi();
    if (typeof R.requireGeneratorVersion !== 'function') {
      throw new ValidationErrorG('Rotation.requireGeneratorVersion is missing — js/rotation.js version contract not loaded');
    }
    return R.requireGeneratorVersion(raw);
  }

  // trackForGeneration(generation) — the CURRENT BLOCK's Track. Identical
  // logic to the worker: rotation 0 is the lineage's pinned starting track
  // (the handshake track); rotation >= 1 is built once per block from
  // resolveGeneratedTrack(rotationSeed, ri). Never rebuilt within a block,
  // never stale across a boundary.
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

  function finishStopped(post) {
    state.running = false;
    state.waitingForNext = false;
    state.stopRequested = false;
    post({ type: 'stopped', generation: state.evolution ? state.evolution.generation : 0 });
  }

  function startGenerationEvaluation(post) {
    var evo = state.evolution;
    // RE-DERIVE the block's Track PER GENERATION (G1 — never cache state.track
    // at function entry across a boundary).
    var generation = evo.generation;
    var track = trackForGeneration(generation);
    var population = evo.population;
    var results = new Array(population.length);
    var i = 0;
    function step() {
      try {
        if (state.stopRequested) { finishStopped(post); return; }
        results[i] = Evolution.evaluateDriver(track, population[i]);
        i += 1;
        if (i < population.length) { setTimeout(step, 0); return; }
        var bestIndex = evo.recordEvaluation(results);
        state.waitingForNext = true;
        state.running = true;
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
        post({ type: 'error', message: String((err && err.message) || err) });
      }
    }
    setTimeout(step, 0);
  }

  return function handleMessage(msg, post) {
    try {
      switch (msg.type) {
        case 'init': {
          var seed = strictFn(msg.seed, 'init seed');
          var initRotationSeed = strictFn(msg.rotationSeed, 'init rotationSeed');
          var initGeneratorVersion = requireVersion(msg.generatorVersion); // FIT-V4 — strict, never defaults
          var track = new Track(msg.controlPoints, msg.halfWidth);
          var evo = new Evolution(seed);
          evo.randomPopulation();
          state.pinnedTrack = track;
          state.track = track;
          state.rotationSeed = initRotationSeed;
          state.generatorVersion = initGeneratorVersion;
          state.currentRotationIndex = null;
          state.evolution = evo;
          state.running = false;
          state.waitingForNext = false;
          state.stopRequested = false;
          state.maxGenerations = 0;
          post({ type: 'ready', generation: evo.generation });
          break;
        }
        case 'resume': {
          if (!Number.isInteger(msg.generation) || msg.generation < 1) {
            throw new ValidationErrorG('resume requires generation: integer >= 1, got ' + String(msg.generation));
          }
          // H-1 strict gate FIRST, before any construction (see worker).
          var rngState = strictFn(msg.rngState, 'resume rngState');
          var resumeRotationSeed = strictFn(msg.rotationSeed, 'resume rotationSeed');
          var resumeGeneratorVersion = requireVersion(msg.generatorVersion); // FIT-V4 — strict, never defaults
          var trackR = new Track(msg.controlPoints, msg.halfWidth);
          var evoR = Evolution.resumeFromBest(msg.bestWeights, msg.generation, rngState);
          state.pinnedTrack = trackR;
          state.track = trackR;
          state.rotationSeed = resumeRotationSeed;
          state.generatorVersion = resumeGeneratorVersion;
          state.currentRotationIndex = null;
          state.evolution = evoR;
          state.running = false;
          state.waitingForNext = false;
          state.stopRequested = false;
          state.maxGenerations = 0;
          post({ type: 'ready', generation: evoR.generation });
          break;
        }
        case 'start': {
          if (!state.evolution || !state.track) { post({ type: 'error', message: 'start requires a prior init or resume' }); break; }
          var maxGen = msg.maxGenerations;
          if (typeof maxGen !== 'number' || !Number.isFinite(maxGen) || Math.floor(maxGen) !== maxGen || maxGen < 1) {
            post({ type: 'error', message: 'start requires maxGenerations: integer >= 1, got ' + String(maxGen) });
            break;
          }
          if (state.running || state.waitingForNext) { post({ type: 'error', message: 'start ignored — a generation loop is already in flight' }); break; }
          state.maxGenerations = maxGen;
          state.stopRequested = false;
          if (state.evolution.generation > maxGen) { post({ type: 'done', generation: state.evolution.generation }); break; }
          state.running = true;
          startGenerationEvaluation(post);
          break;
        }
        case 'next': {
          if (!state.evolution) { post({ type: 'error', message: 'next requires a prior init or resume' }); break; }
          if (!state.waitingForNext) { post({ type: 'error', message: 'next received but no generation is awaiting the logging handshake' }); break; }
          if (state.stopRequested) { finishStopped(post); break; }
          state.waitingForNext = false;
          if (state.evolution.generation >= state.maxGenerations) {
            state.running = false;
            post({ type: 'done', generation: state.evolution.generation });
            break;
          }
          state.evolution.spawnNextGeneration();
          startGenerationEvaluation(post);
          break;
        }
        case 'stop':
          state.stopRequested = true;
          if (state.waitingForNext || !state.running) { finishStopped(post); }
          break;
        default:
          post({ type: 'error', message: 'unknown message type: ' + String(msg.type) });
      }
    } catch (err) {
      post({ type: 'error', message: String((err && err.message) || err) });
    }
  };
}

// ---------------------------------------------------------------------------
// Trainer — the public class. M2 headless scope.
//
//   new Trainer(track, {
//     logger,             // object with logGeneration(payload) — M3 passes Db
//     onProgress,         // fn({ generation, bestFitness, avgFitness, worstFitness, bestIndex, results, rotationIndex, difficulty, trackHash })
//     onBanner,           // fn({ type, ... }) — M4 renders the banner UI
//     onPersist,          // fn(payload) — M3 passes Db.persist(); skipped once logging is disabled
//     lineage,            // { seed: uint32, rotationSeed: uint32 } (fresh) | { resume: { bestWeights, rngState, generation }, rotationSeed: uint32 }
//                         // rotationSeed is REQUIRED for BOTH kinds (uint32) —
//                         // a missing or invented seed throws at start() (D-11)
//     expectedTrackHash,  // lineage guard (M3 wires the real source; M2 hook)
//     trackHash,          //   lineage-level pair is ATOMIC: any lineage-level hash makes BOTH sides come from lineage
//     workerUrl,          // default 'js/train-worker.js'
//     workerFactory,      // TEST SEAM (not public API): fn(url) replacing `new Worker(url)`; a factory
//                         // throw is handled EXACTLY like a sync Worker constructor failure
//   })
//
// TEST SEAMS (used by the smoke suite; not public API):
//   opts.workerFactory(url) — replaces the real Worker with an injected
//     backend (e.g. a FakeWorker). A factory that throws (or returns a
//     falsy backend) is treated exactly like a sync Worker constructor
//     failure: transparent MessageChannel fallback, _fallbackReason recorded,
//     NO banner, run completes.
//   MessageChannel shadowing — the fallback constructor checks
//     `typeof MessageChannel` at CALL time, so a test may set
//     globalThis.MessageChannel = undefined to force the fallback-failure
//     path (and restore it in a finally).
//   createFallbackCore (module.exports) — drives the fallback directly.
//
// States: 'idle' → 'running' → 'done' | 'stopped' | 'halted'.
//         'halted' is terminal (UNIQUE lineage-integrity conflict); the only
//         exits are reload + resume, or re-creating the Trainer.
// ---------------------------------------------------------------------------
class Trainer {
  constructor(track, opts) {
    if (!track || typeof track.startPose !== 'function' || !Array.isArray(track.controlPoints)) {
      throw new ValidationError('Trainer requires a Track-like object (startPose(), controlPoints[])');
    }
    opts = opts || {};
    var assertFn = function (value, name) {
      if (value !== undefined && value !== null && typeof value !== 'function') {
        throw new ValidationError('Trainer opts.' + name + ' must be a function when provided, got ' + typeof value);
      }
    };
    assertFn(opts.onProgress, 'onProgress');
    assertFn(opts.onBanner, 'onBanner');
    assertFn(opts.onPersist, 'onPersist');
    if (opts.workerFactory !== undefined && typeof opts.workerFactory !== 'function') {
      throw new ValidationError('Trainer opts.workerFactory must be a function when provided, got ' + typeof opts.workerFactory);
    }
    if (opts.logger !== undefined && opts.logger !== null) {
      if (typeof opts.logger !== 'object' || typeof opts.logger.logGeneration !== 'function') {
        throw new ValidationError('Trainer opts.logger must be an object with logGeneration(payload), got ' + typeof opts.logger);
      }
    }

    var lineage = opts.lineage || null;

    this.track = track;
    this.opts = opts;
    this.logger = opts.logger || null;
    this.onProgress = opts.onProgress || null;
    this.onBanner = opts.onBanner || null;
    this.onPersist = opts.onPersist || null;

    // Lineage-guard hook (M2/C2-F2). The lineage-level pair is ATOMIC: if the
    // caller provides ANY lineage-level hash, BOTH sides of the effective pair
    // come from lineage (a partial lineage pair is NEVER completed by the root
    // pair). Otherwise the opts-root pair is used. The EFFECTIVE pair must be
    // complete or empty — exactly one side set from either source is a caller
    // bug and throws at construction. start() additionally refuses to run when
    // both sides are set and differ.
    var lineageAttempted = !!(lineage &&
      (typeof lineage.expectedTrackHash === 'string' || typeof lineage.trackHash === 'string'));
    var guardExpected, guardActual;
    if (lineageAttempted) {
      guardExpected = typeof lineage.expectedTrackHash === 'string' ? lineage.expectedTrackHash : null;
      guardActual = typeof lineage.trackHash === 'string' ? lineage.trackHash : null;
    } else {
      guardExpected = typeof opts.expectedTrackHash === 'string' ? opts.expectedTrackHash : null;
      guardActual = typeof opts.trackHash === 'string' ? opts.trackHash : null;
    }
    if ((guardExpected === null) !== (guardActual === null)) {
      throw new ValidationError('Trainer: lineage guard requires BOTH hashes or neither (effective pair)');
    }
    this.expectedTrackHash = guardExpected;
    this.trackHash = guardActual;

    this.state = 'idle';
    this.generation = 0;        // last fully evaluated generation
    this.currentResults = null; // results array of that generation

    // Session-scoped degrade flag (LOG-FAILURE). NEVER reset by stop()/start()
    // within one session — "for the REST OF THIS SESSION" means the whole tab.
    this.loggingDisabled = false;
    this._persistSkipBannerShown = false;
    this._stopRequested = false;

    // Backend bookkeeping for the failure matrix (see header).
    this.readyReceived = false;      // set on a 'ready' from the backend
    this.generationsLogged = 0;      // SESSION-CUMULATIVE — incremented on a SUCCESSFUL logGeneration; never reset by start() (gates the M-4 onerror respawn)
    this._fallbackRetried = false;   // one MessageChannel respawn per start()
    this._lastStart = null;          // stored handshake for onerror replay

    this._usingFallback = false;
    this._workerFactory = typeof opts.workerFactory === 'function' ? opts.workerFactory : null;
    this._worker = null;
    this._localPort = null;   // fallback: the "main thread" end of the channel
    this._remotePort = null;  // fallback: the "worker" end — runs the core
    this._spawnBackend(opts.workerUrl || 'js/train-worker.js');
  }

  // ---- fallback transport factory (shared by construction and onerror respawn)
  // Creates the MessageChannel pair and wires the in-process core exactly the
  // way a Worker would be wired. Throws if MessageChannel/setTimeout is
  // unavailable — callers decide what that means (constructor throw vs halt).
  _createFallbackTransport() {
    if (typeof MessageChannel === 'undefined' || typeof setTimeout === 'undefined') {
      throw new Error('MessageChannel (or setTimeout) is not available in this environment');
    }
    var self = this;
    var channel = new MessageChannel();
    var localPort = channel.port1;   // main-thread side: posts INTO the core, receives core's replies
    var remotePort = channel.port2;  // "worker" side: the core receives on this port, replies on this port
    var core = createFallbackCore();
    remotePort.onmessage = function (event) {
      core(event.data, function (out) { remotePort.postMessage(out); });
    };
    localPort.onmessage = function (event) { self._onBackendMessage(event.data); };
    if (typeof localPort.start === 'function') localPort.start();
    if (typeof remotePort.start === 'function') remotePort.start();
    return { localPort: localPort, remotePort: remotePort };
  }

  // ---- backend spawn: Worker preferred, MessageChannel fallback, else throw --
  _spawnBackend(workerUrl) {
    this._fallbackReason = null;
    var self = this;

    // Preferred path: the injected test/diagnostic backend (opts.workerFactory),
    // else a real Web Worker, off the main thread entirely.
    try {
      var worker = null;
      if (this._workerFactory) {
        // Test seam — the factory REPLACES `new Worker(url)`. A factory throw
        // or falsy return is handled EXACTLY like a sync Worker constructor
        // failure: transparent MessageChannel fallback, no banner.
        worker = this._workerFactory(workerUrl) || null;
      } else if (typeof Worker !== 'undefined') {
        worker = new Worker(workerUrl);
      } else {
        this._fallbackReason = 'Worker is not available in this environment';
      }
      if (worker) {
        worker.onmessage = function (event) { self._onBackendMessage(event.data); };
        worker.onerror = function (event) { self._onWorkerError(event); };
        this._worker = worker;
        return;
      }
      if (this._workerFactory && !this._fallbackReason) {
        this._fallbackReason = 'workerFactory returned falsy for ' + workerUrl;
      }
    } catch (err) {
      // workerFactory threw, or new Worker threw (file:// origin, sandbox, CSP…).
      this._fallbackReason = String((err && err.message) || err);
    }

    // Fallback path: MessageChannel self-rescheduling loop on this thread.
    try {
      var t = this._createFallbackTransport();
      this._usingFallback = true;
      this._localPort = t.localPort;
      this._remotePort = t.remotePort;
      return;
    } catch (fallbackErr) {
      // Both paths unavailable — no headless training is possible. Throw LOUDLY;
      // never silently degrade into unlogged/unverified execution.
      throw new Error(
        'Trainer: neither Web Worker nor MessageChannel fallback could start — cannot run headless training. ' +
        (this._fallbackReason ? 'Worker failure: ' + this._fallbackReason + '. ' : '') +
        'Fallback failure: ' + String((fallbackErr && fallbackErr.message) || fallbackErr)
      );
    }
  }

  _postToBackend(msg) {
    if (this._usingFallback) { this._localPort.postMessage(msg); return; } // port1 posts → port2 (core) receives
    this._worker.postMessage(msg);
  }

  // L-5: a throwing UI can never break the protocol. onBanner failures are
  // logged to console.error (stderr in node) and SWALLOWED — the caller keeps
  // going, and no 'next' is ever lost because a banner blew up.
  _banner(payload) {
    if (!this.onBanner) return;
    try {
      this.onBanner(payload);
    } catch (err) {
      console.error('[trainer] onBanner handler threw (protocol continues):', err);
    }
  }

  // The plan's LOG-FAILURE wording — shared verbatim by the logger-throw path
  // and the persist-throw path (M1/SEC-26).
  _logFailureWording(generation) {
    return 'GENERATION ' + generation + ' NOT LOGGED — this tab\'s committed prefix intact through ' +
      'generation ' + (generation - 1) + ' — LOGGING DISABLED THIS SESSION — TRAINING CONTINUES ' +
      'WITHOUT LOGGING — resume later to recompute generation ' + generation + ' identically';
  }

  // Extended LINEAGE banner for backend-death halts (M-4): the plan's UNIQUE
  // halt wording PLUS the resume-identity sentence. Used for async onerror
  // post-logging and for any backend failure surface that falls through to
  // halt. The committed prefix is never rewritten; a resume recomputes the
  // unlogged generation identically.
  _haltedLineageBanner(prefix) {
    return prefix + 'LINEAGE INTEGRITY ERROR — RUN HALTED — committed lineage untouched — ' +
      'reload and resume recomputes the unlogged generation identically';
  }

  _controlPoints() {
    var pts = this.track.controlPoints;
    var out = new Array(pts.length);
    for (var i = 0; i < pts.length; i++) out[i] = { x: pts[i].x, y: pts[i].y };
    return out;
  }

  // ---- start(maxGenerations) — lineage guard FIRST, then handshake --------
  start(maxGenerations) {
    // Lineage guard (M2 hook): both hashes must be present AND differ to refuse.
    if (this.expectedTrackHash && this.trackHash && this.expectedTrackHash !== this.trackHash) {
      this._banner({
        type: 'lineage-mismatch',
        expected: this.expectedTrackHash,
        actual: this.trackHash,
        message: 'LINEAGE GUARD — active track does not match the DB lineage track_hash; refusing to train. Re-point the lineage (M3: Reconcile / Fork flow).',
      });
      return false;
    }
    var maxGen = maxGenerations;
    if (typeof maxGen !== 'number' || !Number.isFinite(maxGen) || Math.floor(maxGen) !== maxGen || maxGen < 1) {
      throw new ValidationError('Trainer.start(maxGenerations) requires an integer >= 1, got ' + String(maxGenerations));
    }
    if (this.state === 'halted') {
      // Defense-in-depth (M-2): a halted trainer never restarts on its own.
      throw new Error('Trainer is halted (lineage-integrity or backend failure) — start() refused; reload the page and resume');
    }
    if (this.state === 'running') {
      throw new Error('Trainer.start() called while a run is already in flight');
    }

    this._stopRequested = false;
    this._fallbackRetried = false;   // one respawn per start()
    this.readyReceived = false;      // fresh backend is armed for a new 'ready'
    this.state = 'running';

    var lineage = this.opts.lineage || null;
    // FIT-V3 (D-11) — rotationSeed is REQUIRED for BOTH lineage kinds: the
    // handshake carries it to the backend, which re-derives the rotation
    // curriculum per generation. A missing seed throws a typed ValidationError
    // BEFORE any handshake goes out — never an invented zero.
    var rotationSeed = null;
    if (lineage && lineage.rotationSeed !== undefined && lineage.rotationSeed !== null) {
      try {
        rotationSeed = strictLineageSeed(lineage.rotationSeed, 'opts.lineage.rotationSeed');
      } catch (err) {
        this.state = 'idle';
        throw err;
      }
    }
    if (rotationSeed === null) {
      this.state = 'idle';
      throw new ValidationError(
        'Trainer.start: opts.lineage.rotationSeed (uint32) is required for both fresh and resume lineages — ' +
        'the lineage persists it as meta.rotation_seed (D-11); refusing to run with a missing or invented seed'
      );
    }
    // FIT-V4 — the canonical integer generator version is REQUIRED on BOTH
    // lineage kinds. Strict parser via the one rotation.js contract: a missing
    // or empty field throws a typed ValidationError BEFORE any handshake goes
    // out, never a default. Carried in `shared` into init/resume/replay.
    var generatorVersion;
    try {
      generatorVersion = strictLineageVersion(lineage ? lineage.generatorVersion : undefined);
    } catch (err) {
      this.state = 'idle';
      throw err;
    }
    var resume = lineage && lineage.resume ? lineage.resume : null;
    var shared = { controlPoints: this._controlPoints(), halfWidth: this.track.halfWidth, rotationSeed: rotationSeed, generatorVersion: generatorVersion };
    var handshake;
    if (resume) {
      // Resume: caller passes the COMPLETED generation N; the backend spawns
      // gen N+1 (RNG state restored from the authoritative cursor).
      handshake = Object.assign({
        type: 'resume',
        bestWeights: resume.bestWeights,
        rngState: resume.rngState,
        generation: resume.generation,
      }, shared);
    } else {
      var seed = lineage && typeof lineage.seed === 'number' ? lineage.seed : null;
      if (seed === null) {
        this.state = 'idle';
        throw new ValidationError(
          'Trainer.start: opts.lineage.seed (uint32) is required for a fresh lineage. ' +
          '(crypto.getRandomValues-seeded lineage creation lands with the DB in M3.)'
        );
      }
      handshake = Object.assign({ type: 'init', seed: seed }, shared);
    }
    // Remember the exact handshake so an async pre-ready onerror can replay
    // the lineage ONCE on the fallback (M-4).
    this._lastStart = { handshake: handshake, maxGenerations: maxGen, lineage: lineage, shared: shared };
    this._postToBackend(handshake);
    // FIFO: 'start' is processed AFTER the init/resume handshake by both the
    // Worker and the MessageChannel fallback.
    this._postToBackend({ type: 'start', maxGenerations: maxGen });
    return true;
  }

  // ---- stop() — idempotent; safe to call from any state ------------------
  stop() {
    if (this.state === 'stopped' || this.state === 'done' || this.state === 'halted') return;
    this._stopRequested = true;
    this._postToBackend({ type: 'stop' });
    // Optimistic — the backend acks 'stopped' (possibly later, mid-generation);
    // _onBackendMessage re-sets 'stopped' idempotently.
    this.state = 'stopped';
  }

  // ---- M-4: async Worker failure surface ----------------------------------
  // See the failure matrix in the file header. Order of operations is
  // load-bearing: terminate() FIRST, decide SECOND, respawn/halt THIRD.
  _onWorkerError(event) {
    if (this._worker) {
      try { this._worker.terminate(); } catch (_) { /* already dead — fine */ }
    }
    this._worker = null;
    var msg = String(
      (event && event.message) ||
      (event && event.error && event.error && event.error.message) ||
      'unknown worker error'
    );

    // If a generation was LOGGED this session, a partial commit happened —
    // halt, never re-run. If NOTHING was committed (even if a 'ready' was
    // received), the lineage was never persisted anywhere — a one-shot
    // respawn on the fallback re-derives it byte-identically (arch F-4: the
    // readyReceived conjunct was dropped — ANY pre-first-commit failure
    // respawns; integrity unchanged — no commit happened).
    var nothingTouched = this.state === 'idle' || this.generationsLogged === 0;
    if (!this._fallbackRetried && nothingTouched) {
      this._fallbackRetried = true;
      try {
        var t = this._createFallbackTransport();
        this._usingFallback = true;
        this._localPort = t.localPort;
        this._remotePort = t.remotePort;
        this.readyReceived = false;   // the fresh fallback will re-ready
        this._banner({
          type: 'worker-failed-fallback',
          message: 'worker failed before producing anything (' + msg + ') — falling back to the in-thread MessageChannel backend; the lineage replays identically from the stored handshake',
        });
        this._replayHandshake();
        return;
      } catch (fallbackErr) {
        // Row 4 of the failure matrix — the fallback ALSO failed: halt, extend the banner.
        var fmsg = String((fallbackErr && fallbackErr.message) || fallbackErr);
        this.state = 'halted';
        this._banner({
          type: 'worker-error',
          message: this._haltedLineageBanner('Worker backend failed and the MessageChannel fallback failed too (' + fmsg + ') — '),
        });
        return;
      }
    }

    // Row 3 of the failure matrix — progress was made or generations were
    // logged: the lineage is NEVER re-run from scratch after a partial commit.
    this.state = 'halted';
    this._banner({
      type: 'worker-error',
      message: this._haltedLineageBanner('Worker backend failed (' + msg + ') — '),
    });
  }

  // Replay the stored handshake (init/resume + start) on a fresh fallback
  // backend after an async pre-first-commit onerror (M-4 row 2). No-op when
  // nothing was ever started (state 'idle') — the fresh backend simply
  // waits for the next start().
  _replayHandshake() {
    var last = this._lastStart;
    if (!last || this.state !== 'running') return;
    this._postToBackend(last.handshake);
    this._postToBackend({ type: 'start', maxGenerations: last.maxGenerations });
  }

  // ---- backend → main-thread message handling ----------------------------
  // Top-level guard (L-5): NOTHING thrown inside a backend message handler
  // ever propagates out of it. UI/banner failures are already contained by
  // _banner/_emitProgress; anything left (a corrupt payload, a bug in this
  // switch) is logged and treated as terminal for the run — never silent,
  // never a half-processed generation.
  _onBackendMessage(msg) {
    if (!msg || typeof msg !== 'object') return;
    try {
      this._dispatchBackendMessage(msg);
    } catch (err) {
      console.error('[trainer] exception inside backend message handler (run halted):', err);
      this.state = 'halted';
      this._banner({
        type: 'worker-error',
        message: this._haltedLineageBanner('Trainer backend handler failed (' + String((err && err.message) || err) + ') — '),
      });
    }
  }

  _dispatchBackendMessage(msg) {
    switch (msg.type) {
      case 'ready':
        this.readyReceived = true;
        this.generation = msg.generation;
        return;
      case 'generation':
        this._onGenerationMessage(msg.payload);
        return;
      case 'stopped':
        if (this.state === 'halted') return; // M-2 — a halted trainer stays halted
        this.state = 'stopped';
        if (typeof msg.generation === 'number') this.generation = msg.generation;
        return;
      case 'done':
        if (this.state === 'halted') return; // M-2 — a halted trainer stays halted
        this.state = 'done';
        if (typeof msg.generation === 'number') this.generation = msg.generation;
        return;
      case 'error':
        // Async failure surface of the backend — terminal (see failure matrix).
        this.state = 'halted';
        this._banner({
          type: 'worker-error',
          message: this._haltedLineageBanner('Backend reported error (' + String(msg.message || 'unknown backend error') + ') — '),
        });
        return;
      default:
        this._banner({ type: 'worker-error', message: 'unknown backend message: ' + String(msg.type) });
    }
  }

  _onGenerationMessage(payload) {
    // Late in-flight generation after stop(): DO NOT log it, DO NOT advance.
    // The committed prefix stays put; a resume recomputes it identically.
    if (this._stopRequested) { this.state = 'stopped'; return; }
    this.generation = payload.generation;
    this.currentResults = payload.results;

    var mayContinue = true;
    if (!this.loggingDisabled && this.logger) {
      try {
        this.logger.logGeneration(payload);
        this.generationsLogged += 1;   // session-cumulative — gates M-4 onerror respawn
      } catch (err) {
        var unique = isUniqueViolation(err);
        if (unique) {
          // SEC-24 carve-out: lineage-integrity conflict — never continue-unlogged.
          this.state = 'halted';
          this._postToBackend({ type: 'stop' });
          this._banner({
            type: 'log-failed',
            generation: payload.generation,
            unique: true,
            message: 'LINEAGE INTEGRITY ERROR — RUN HALTED — committed lineage untouched',
          });
          mayContinue = false;
        } else {
          // Session-disable policy: training continues, logging stops HERE.
          this.loggingDisabled = true;
          this._banner({
            type: 'log-failed',
            generation: payload.generation,
            unique: false,
            message: this._logFailureWording(payload.generation),
          });
        }
      }
    }

    if (mayContinue) {
      // Persist hook — M3 wires Db.persist here. SEC-26: SKIPPED for the rest
      // of a logging-disabled session (distinct banner, shown ONCE).
      if (this.onPersist) {
        if (this.loggingDisabled) {
          if (!this._persistSkipBannerShown) {
            this._persistSkipBannerShown = true;
            this._banner({
              type: 'persist-skipped',
              generation: payload.generation,
              message: 'LOGGING DISABLED — persist skipped',
            });
          }
        } else {
          try {
            this.onPersist(payload);
          } catch (err) {
            // M-1: a THROWING persist hook takes the same LOG-FAILURE latch as
            // a logger throw — for the rest of this session persist is skipped
            // silently through the SEC-26 path, and the banner carries the
            // plan's LOG-FAILURE wording. Training CONTINUES.
            this.loggingDisabled = true;
            this._banner({
              type: 'log-failed',
              generation: payload.generation,
              unique: false,
              message: this._logFailureWording(payload.generation),
            });
          }
        }
      }
      this._emitProgress(payload);
      if (this.state === 'running') this._postToBackend({ type: 'next' });
    }
  }

  // L-5: a throwing onProgress can never break the protocol either — the
  // error is logged to console.error (stderr) and swallowed; the 'next' post
  // in the caller is never dropped because UI code threw.
  // NOTE (fitness-signed semantics): payload.results[].distance carries
  // FITNESS (net signed odometer + lap bonus; may be negative) — sum/worst
  // are computed from it as-is; bestFitness is the SELECTED driver's fitness.
  _emitProgress(payload) {
    if (!this.onProgress) return;
    var d = payload.results;
    var sum = 0, worst = Infinity;
    for (var i = 0; i < d.length; i++) {
      sum += d[i].distance;
      if (d[i].distance < worst) worst = d[i].distance;
    }
    try {
      this.onProgress({
        generation: payload.generation,
        bestFitness: d[payload.bestIndex].distance,
        avgFitness: sum / d.length,
        worstFitness: worst,
        bestIndex: payload.bestIndex,
        results: payload.results,
        rotationIndex: payload.rotationIndex,
        difficulty: payload.difficulty,
        trackHash: payload.trackHash,
      });
    } catch (err) {
      console.error('[trainer] onProgress handler threw (protocol continues):', err);
    }
  }

}

// Browser-global exposure + guarded CommonJS export (node tests).
// createFallbackCore is exported as a smoke-test seam (legs 14/15/17 drive
// the fallback directly); it is NOT a public API.
if (typeof globalThis !== 'undefined') {
  globalThis.Trainer = Trainer;
  globalThis.isUniqueViolation = isUniqueViolation;
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    Trainer: Trainer,
    isUniqueViolation: isUniqueViolation,
    ValidationError: ValidationError,
    createFallbackCore: createFallbackCore,
  };
}
