// tests/smoke.mjs — node smoke suite for the 0003_car_racer pure-logic core
// (M1), the M2 evolution/eval/Trainer legs (legs 10–18, incl. the M2-fix
// legs: mid-run stop discard, two-core byte-equality, strict uint32 gate,
// persist-throw latch, backend-death matrix + effective-pair lineage guard),
// and the M3 DB legs (legs 19–33: vendor SHA-256 manifest gate + embedded
// sql-wasm-binary.js round-trip + index.html script order, schema init,
// FK cascade, transaction rollback, UNIQUE lineage-integrity halt, resume
// byte-identity (incl. the Trainer-handshake resume + per-generation rng_state
// audit compare), resume end-to-end persist (injected handle), the ADOPTION
// BATTERY incl. crafted trigger/view/extra-object rejection + user_version +
// best_weights-for-MAX(generation) refusal, the persist step-2 verify-failure
// path, the measured per-persist cost, leg 28 (the code-only grep gates +
// raw-source backstop + planted-case self-tests), leg 29 (persist
// end-to-end incl. the stubbed navigator.locks variant), leg 30 (SEC-22
// node-side adoption paths + handle-file size gate), leg 31 (fingerprint
// fault injection), leg 32 (the security-M4 combined fork/snapshot/import
// refusal battery), leg 33 (read-path result-set shapes: getDriverResults
// end-to-end + the array-vs-set sibling read paths), leg 34
// (restoreFromLastKnownGood: no-cache / battery-valid adoption / trigger /
// truncation, node), legs 35–41 (FINAL-BATCH-2: the battery's semantic arm —
// FK-missing / +Inf best_fitness / best_driver_index=99 / crashed=7 crafted
// files all refused; the Track sample-budget refusal; the LSG no-regress
// matrix; the snapshot-bytes coherence tag), legs 42–44 (FINAL-BATCH-3B:
// boot-fallback batteries through DbSession.restoreIfPossible against the
// real code with an in-memory IndexedDB stand-in; the persist()-level LSG
// NO-REGRESS arms — strictly-newer overwrite + equal-but-divergent skip +
// the cache-ahead banner — through the real file path and IDB-fallback mode),
// legs 45–53 (FITNESS-V2: fresh v2 schema assertion + DDL twin string, the
// v1 → v2 adoption-migration — rows preserved byte-for-byte, FKs re-declared
// CASCADE, FK cascade re-verified — the injected-migration-failure rollback
// arm, negative-fitness log → export → battery re-adopt round trip, the R1
// FK-order spot check, fresh-v2 fork + re-log, the four terminal outcomes
// (crashed/stuck/time-cap/finished) via stubbed cars, the R5 net-negative
// fitness-pipeline fixture — bonus-inclusive end-to-end — the R6 grep gate,
// and the log → export → re-adopt → resumeState end-to-end round trip; with
// the 46b arm this is 44 pre-existing legs + 11 new legs (legs 45-54) = 55 legs;
// the FIT-V3 M6a procedural-track legs (56/57/65/69) = 59 legs; the FIT-V3 M6b
// DB-v3 legs (59–63, 67, 68, 70–73) bring the total to 70 legs; the FIT-V3
// M6c training-runtime legs (58, 58b, 64, 66) bring the total to 74 legs; the
// fix-wave legs (75, 76) bring the total to 76 legs).
// Run with:  node tests/smoke.mjs
// Also:      node --check tests/smoke.mjs
//
// The pure-logic modules in js/ are plain classic scripts with a guarded
// `module.exports = ...` at the end for exactly this kind of node consumer,
// so we load them via createRequire here. No npm dependencies are used.
//
// Exit code: 0 when every leg passes, 1 otherwise. There are NO silent
// skips — every leg runs and failures are reported loudly with the
// offending assertion message.

import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);

const REPO_ROOT = fileURLToPath(new URL('..', import.meta.url));

const { NeuralNetwork, ValidationError, mulberry32, NN_ARCH } = require('../js/nn.js');
const {
  Track,
  DEFAULT_CONTROL_POINTS,
  canonicalJSON,
  cyrb53,
} = require('../js/track.js');
const { Car, MAX_SPEED, CAR_RADIUS } = require('../js/car.js');
const { SensorFan, SENSOR_ANGLES_DEG, SENSOR_MAX } = require('../js/sensors.js');
const { DriverSim, LAP_BONUS_TRACK_LENGTHS, LAP_BONUS_LAPS } = require('../js/driver-sim.js');
const { Replay } = require('../js/replay.js');
const {
  Evolution,
  POPULATION,
  SIM_DT,
  EVAL_CAP_S,
  STUCK_TIMEOUT_S,
  PROGRESS_ADVANCE_PX,
  MUTATION_RATE,
  MUTATION_SCALE,
} = require('../js/evolution.js');
// FIT-V4 — rotation.js owns the single canonical generator-version parser
// (globalThis.Rotation.requireGeneratorVersion). Trainer.start() and both
// backend cores consult it at RUN time, so rotation.js MUST be loaded before
// the first Trainer leg (the lazy requireRotation() below is defined at the
// M6a section, which is too late for legs 14–18/24/34). Load order mirrors
// tests/version-flow.mjs: rotation before trainer. require() is idempotent,
// so the later requireRotation() simply reuses this cached load.
require('../js/rotation.js');
const { Trainer, createFallbackCore } = require('../js/trainer.js');

// Async helpers — used by the Trainer/core legs to await backend handshakes.
// (Date.now is fine here — it is test orchestration, not the eval loop.)
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
async function waitFor(condition, label, timeoutMs) {
  const t0 = Date.now();
  const limit = typeof timeoutMs === 'number' ? timeoutMs : 60000;
  while (Date.now() - t0 < limit) {
    if (condition()) return;
    await sleep(2);
  }
  throw new Error('timeout waiting for ' + label);
}

// -----------------------------------------------------------------------------
// Tiny test harness — counts legs, prints one line per leg, never throws.
// Legs may be sync or async (the Trainer/core legs await backend handshakes).
// -----------------------------------------------------------------------------
let legsPassed = 0;
const failures = [];

async function leg(name, fn) {
  try {
    await fn();
    legsPassed += 1;
    console.log('  ok   — ' + name);
  } catch (err) {
    failures.push({ name, err });
    const detail = String((err && err.stack) || err)
      .split('\n')
      .map((line) => '         ' + line)
      .join('\n');
    console.log('  FAIL — ' + name + '\n' + detail);
  }
}

function assert(condition, message) {
  if (!condition) throw new Error('assert failed: ' + message);
}

function assertValidationError(fn, message) {
  let threw = null;
  try {
    fn();
  } catch (err) {
    threw = err;
  }
  assert(threw !== null, message + ' — expected ValidationError, but nothing was thrown');
  assert(
    threw.name === 'ValidationError',
    message + ' — expected ValidationError, got ' + threw.name + ': ' + threw.message
  );
}

// wrapToPi: normalize an angle to (-π, π]. Needed by the leg-5 pure-pursuit
// follower; inlined here so the suite has no hidden helpers outside the legs.
function wrapToPi(a) {
  return Math.atan2(Math.sin(a), Math.cos(a));
}

// Binary search: sample nearest to arcPos (samples are sorted by arcPos and
// arc length is cumulative). Used by the leg-5 follower for a fixed
// look-ahead distance instead of naive nearest-point pursuit.
function sampleNearArcPos(track, arcPos) {
  const L = track.length;
  const a = ((arcPos % L) + L) % L;
  const s = track.samples;
  let lo = 0, hi = s.length - 1, best = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (s[mid].arcPos <= a) { best = mid; lo = mid + 1; } else { hi = mid - 1; }
  }
  return s[best];
}

// Pure-pursuit centerline follower used ONLY by the driving leg: aim at a
// point ~90 px ahead along the road centerline and brake when the road ahead
// bends away from the nose. This is the same shape of controller the
// evolutionary drivers will learn; a plain proportional controller to the
// nearest point cannot clear the corner radii at full chat.
function driveAlongTrack(track, car) {
  const nc = track.nearestCenterline(car.x, car.y);
  const look = sampleNearArcPos(track, nc.arcPos + 90);
  const targetAngle = Math.atan2(look.y - car.y, look.x - car.x);
  const err = wrapToPi(targetAngle - car.heading);
  const steer = Math.max(-1, Math.min(1, err * 2.2));
  const gas = Math.abs(err) > 0.55 ? -0.35 : 1; // lift/brake for curves
  car.update(1 / 120, { gas, steer });
  return nc;
}

// =============================================================================
// FITNESS-V2 stub cars — STUB DRIVER CARS for the outcome/bonus legs (50/51).
// Swap onto globalThis.Car for the duration of a leg and restore in `finally`
// (DriverSim resolves its Car dependency LAZILY at construction, so this is
// race-free within a single-threaded leg). The stubs telescope along the
// centerline by a fixed arc per tick — they NEVER crash (they ride the
// centerline), never consult the wall geometry, and reset their stagnation
// window via the standard cumArc gate when stepArc >= PROGRESS_ADVANCE_PX.
// =============================================================================
function makeAdvancingStubCar(stepArc) {
  return class StubCar {
    constructor(track, x, y, heading) {
      this.track = track;
      this.x = x;
      this.y = y;
      this.heading = heading;
      this.speed = 600;
      this.odometer = 0;
      this.crashed = false;
      this._arc = 0;
    }
    driverSeat() { return { x: this.x, y: this.y }; }
    update(dt) {
      const length = this.track.length;
      this._arc = (this._arc + stepArc) % length;
      const s = sampleNearArcPos(this.track, this._arc);
      this.x = s.x;
      this.y = s.y;
      this.heading = s.tangent;
      this.speed = 600;
      this.odometer += this.speed * dt;
    }
  };
}
// Crash-on-first-tick stub — folds the 'crashed' arm without touching walls.
class StubCrashCar {
  constructor(track, x, y, heading) {
    this.track = track; this.x = x; this.y = y; this.heading = heading;
    this.speed = 0; this.odometer = 0; this.crashed = false;
  }
  driverSeat() { return { x: this.x, y: this.y }; }
  update() { this.crashed = true; this.speed = 0; }
}
// Stationary stub (speed stays 0) — the stagnation timer fires 'stuck'.
class StubStationaryCar {
  constructor(track, x, y, heading) {
    this.track = track; this.x = x; this.y = y; this.heading = heading;
    this.speed = 0; this.odometer = 0; this.crashed = false;
  }
  driverSeat() { return { x: this.x, y: this.y }; }
  update() { /* never moves */ }
}
// Crash-on-finish-line stub — teleports 0.45 × length per tick like
// makeAdvancingStubCar (the sim's maxCumArc ratchet lags the car position by
// exactly one tick, so laps == LAP_BONUS_LAPS first becomes true on the tick
// AFTER the teleport crosses the line). Sets crashed=true on that very same
// tick: both conditions co-fire at the kill check and the outcome MUST be
// 'crashed' (crash beats finished). Every cruise tick resets stagnation, so
// the stagnation window never fires — this fixture pins the precedence,
// never the stagnation timer.
class StubCrashOnFinishCar {
  constructor(track, x, y, heading) {
    this.track = track; this.x = x; this.y = y; this.heading = heading;
    this.speed = 600; this.odometer = 0; this.crashed = false;
    this._arc = 0;
    this._crossed = false;
  }
  driverSeat() { return { x: this.x, y: this.y }; }
  update(dt) {
    var L = this.track.length;
    if (this._crossed) {
      // The update AFTER the teleport crossed the 100-lap line — exactly the
      // tick whose step-2 ratchet reads 100 laps off the previous position.
      this.crashed = true;
    }
    this._arc += 0.45 * L;
    if (this._arc >= 100 * L) this._crossed = true;
    const s = sampleNearArcPos(this.track, this._arc);
    this.x = s.x; this.y = s.y; this.heading = s.tangent;
    this.speed = 600;
    this.odometer += this.speed * dt;
  }
}
// Grind-to-the-cap stub — the ONLY construction in which 'finished', stuck and
// time-cap all co-fire on ONE tick (tick 5400 == maxTicks):
//   sim ticks 1..222   — cruise 0.45 × length per tick (arc -> 99.9 laps),
//   sim ticks 223..3492 — +22 px nudges every 467 ticks at exactly
//                         223, 690, 1157, 1624, 2091, 2558, 3025, 3492
//                         (each ≥ PROGRESS_ADVANCE_PX, resetting stagnation),
//   sim tick 3958       — nudge #9 teleports arc to exactly 100L − 8 px (the
//                         LAST ≥ 20 px advance the run ever sees; its step-2
//                         delta lands the stagnation reset on tick 3959),
//   sim ticks 3960..    — flat stall; stagnation accumulates 1/120 per tick.
//                         At tick 5399 the accrual is 1440 × (1/120) = 11.9999…
//                         99999996 < 12 (the stuck gate does NOT trip early,
//                         laps is still 99). At tick 5400 it is 1441 × (1/120)
//                         = 12.0083… ≥ 12 — the stuck gate IS due then,
//   sim tick 5399       — creep to arc 100L + 7 px (+15 px < PROGRESS_ADVANCE_PX,
//                         so stagnation does NOT reset); the sim's step-2
//                         ratchet then reads cumArc ≥ 100L on tick 5400 —
//                         the same tick _ticks reaches maxTicks and stagnation
//                         is ≥ STUCK_TIMEOUT_S. All three kill conditions
//                         hold together and 'finished' MUST win (crash >
//                         finished > stuck > time-cap).
class StubGrindFinishCar {
  constructor(track, x, y, heading) {
    this.track = track; this.x = x; this.y = y; this.heading = heading;
    this.speed = 600; this.odometer = 0; this.crashed = false;
    this._arc = 0;
    this._updates = 0;
  }
  driverSeat() { return { x: this.x, y: this.y }; }
  update(dt) {
    var L = this.track.length;
    this._updates += 1;
    var tick = this._updates;
    if (tick <= 222) {
      this._arc += 0.45 * L;                 // cruise — banks ~99.9 laps
      this.speed = 600;
    } else if (tick === 3958) {
      this._arc = 100 * L - 8;               // nudge #9 — lands the stall at 100L−8 px; its step-2 delta resets stagnation on tick 3959
      this.speed = 0;
    } else if (tick < 5400 && (tick - 223) % 467 === 0 && tick <= 3492) {
      this._arc += 22;                       // nudges #1..#8 — each ≥ PROGRESS_ADVANCE_PX resets stagnation
      this.speed = 0;
    } else if (tick === 5399) {
      this._arc = 100 * L + 7;               // +15 px creep — the crossing the ratchet reads on tick 5400
      this.speed = 0;
    } else {
      this.speed = 0;                        // flat stall — no movement, no stagnation reset
    }
    const s = sampleNearArcPos(this.track, this._arc);
    this.x = s.x; this.y = s.y; this.heading = s.tangent;
  }
}

// Run `fn` with globalThis.Car replaced by `stubClass`; ALWAYS restores.
// Async-aware: awaits fn() so the swap spans the whole (possibly async) body.
async function withStubbedCar(stubClass, fn) {
  const realCar = globalThis.Car;
  globalThis.Car = stubClass;
  try {
    return await fn();
  } finally {
    globalThis.Car = realCar;
  }
}

console.log('car_racer smoke suite (M1+M2+M3)');
console.log('------------------------');

// =============================================================================
// M1 leg 1 — NN forward shape and tanh range.
// =============================================================================
await leg('leg 1 — NN forward shape + outputs in [-1,1]', () => {
  const net = new NeuralNetwork(mulberry32(123));
  const rng = mulberry32(999);
  // 200 random input vectors across the full advertised input range:
  // [s1..s5 ∈ (0,1), speed/MAX_SPEED ∈ (-1,1), direction/π ∈ (-1,1)].
  for (let i = 0; i < 200; i++) {
    const input = [
      (rng() + 0.5) / 4294967296,
      (rng() + 0.5) / 4294967296,
      (rng() + 0.5) / 4294967296,
      (rng() + 0.5) / 4294967296,
      (rng() + 0.5) / 4294967296,
      ((rng() + 0.5) / 4294967296) * 2 - 1,
      ((rng() + 0.5) / 4294967296) * 2 - 1,
    ];
    const out = net.forward(input);
    assert(Array.isArray(out), 'forward returns a plain array');
    assert(out.length === 2, 'forward returns exactly 2 values, got ' + out.length);
    assert(Number.isFinite(out[0]) && Number.isFinite(out[1]), 'outputs are finite');
    // tanh saturates to exactly ±1 for |z| >= ~19 — outputs may reach ±1.
    assert(Math.abs(out[0]) <= 1, 'gas in [-1,1] inclusive, got ' + out[0]);
    assert(Math.abs(out[1]) <= 1, 'steer in [-1,1] inclusive, got ' + out[1]);
  }
  // Input contract: exactly 7 finite numbers.
  assertValidationError(() => net.forward([0, 0, 0, 0, 0, 0, 0, 0]), 'forward rejects length 8');
  assertValidationError(() => net.forward([0, 0, 0]), 'forward rejects length 3');
  assertValidationError(() => net.forward([0, 0, 0, 0, 0, 0, NaN]), 'forward rejects NaN');
  assertValidationError(() => net.forward([0, 0, 0, 0, 0, 0, Infinity]), 'forward rejects Infinity');
  assertValidationError(() => net.forward(null), 'forward rejects null');
});

// =============================================================================
// M1 leg 2 — weights validation battery + round-trips + setWeights atomicity.
// =============================================================================
await leg('leg 2 — weights validation battery, roundtrip, setWeights atomicity', () => {
  const seedNet = new NeuralNetwork(mulberry32(7));
  const w = seedNet.getWeights();

  // Structure: exactly 434 floats in the canonical order.
  const total =
    w.w1.length + w.b1.length + w.w2.length + w.b2.length + w.w3.length + w.b3.length;
  assert(total === 434, 'exactly 434 floats, got ' + total);
  assert(w.w1.length === 112 && w.b1.length === 16, 'w1 is 7x16 row-major, b1 is 16');
  assert(w.w2.length === 256 && w.b2.length === 16, 'w2 is 16x16, b2 is 16');
  assert(w.w3.length === 32 && w.b3.length === 2, 'w3 is 16x2, b3 is 2');
  assert(w.architecture === NN_ARCH, 'architecture string is ' + NN_ARCH);

  // getWeights() must hand back deep copies, never internal references.
  const snapshotJson = JSON.stringify(w);
  w.w1[0] = 123456;
  assert(seedNet.getWeights().w1[0] !== 123456, 'getWeights returns deep copies');
  w.w1[0] = seedNet.getWeights().w1[0]; // restore for downstream checks

  // Valid roundtrip through plain-object construction.
  const input = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7];
  const net2 = new NeuralNetwork(JSON.parse(snapshotJson));
  const o1 = seedNet.forward(input);
  const o2 = net2.forward(input);
  assert(o1[0] === o2[0] && o1[1] === o2[1], 'roundtrip net produces identical outputs');
  assert(
    JSON.stringify(net2.getWeights()) === snapshotJson,
    'roundtrip net has byte-identical weights'
  );

  // Valid roundtrip through a JSON string (NeuralNetwork.fromJSON).
  const netStr = NeuralNetwork.fromJSON(snapshotJson);
  assert(
    JSON.stringify(netStr.getWeights()) === snapshotJson,
    'fromJSON(string) roundtrip is byte-identical'
  );

  // setWeights happy path changes the network…
  const shifted = JSON.parse(snapshotJson);
  shifted.b3 = shifted.b3.map((v) => v + 0.5);
  seedNet.setWeights(shifted);
  assert(
    JSON.stringify(seedNet.getWeights()) === JSON.stringify(shifted),
    'setWeights applies a valid payload'
  );
  seedNet.setWeights(JSON.parse(snapshotJson)); // put it back

  // …but a failing setWeights must leave the network UNCHANGED (validate-then-commit).
  const beforeBadSet = JSON.stringify(seedNet.getWeights());
  assertValidationError(
    () => seedNet.setWeights(Object.assign({}, JSON.parse(snapshotJson), { architecture: '7-16-16-1' })),
    'setWeights rejects wrong architecture'
  );
  assert(JSON.stringify(seedNet.getWeights()) === beforeBadSet, 'failed setWeights left network unchanged');

  const wrongLength = JSON.parse(snapshotJson);
  wrongLength.w1 = wrongLength.w1.slice(0, 111); // 433 total floats
  assertValidationError(() => seedNet.setWeights(wrongLength), 'setWeights rejects 433 floats');
  assert(
    JSON.stringify(seedNet.getWeights()) === beforeBadSet,
    'failed setWeights (short w1) left network unchanged'
  );

  // ---- the throwing battery: every entry must throw ValidationError ----
  const base = JSON.parse(snapshotJson);
  const cases = [
    ['architecture 7-16-16-1', Object.assign({}, base, { architecture: '7-16-16-1' })],
    ['architecture missing', ((o) => { delete o.architecture; return o; })(JSON.parse(snapshotJson))],
    ['architecture null', Object.assign({}, base, { architecture: null })],
    ['w1 111 (433 floats)', (() => { const o = JSON.parse(snapshotJson); o.w1 = o.w1.slice(0, 111); return o; })()],
    ['w1 113 (435 floats)', (() => { const o = JSON.parse(snapshotJson); o.w1 = o.w1.concat([0]); return o; })()],
    ['w2 short', (() => { const o = JSON.parse(snapshotJson); o.w2 = o.w2.slice(1); return o; })()],
    ['b3 long', (() => { const o = JSON.parse(snapshotJson); o.b3 = o.b3.concat([0]); return o; })()],
    ['w3 NaN', (() => { const o = JSON.parse(snapshotJson); o.w3[5] = NaN; return o; })()],
    ['b2 Infinity', (() => { const o = JSON.parse(snapshotJson); o.b2[3] = Infinity; return o; })()],
    ['w1 -Infinity', (() => { const o = JSON.parse(snapshotJson); o.w1[0] = -Infinity; return o; })()],
    ['w3 exceeds 1e6', (() => { const o = JSON.parse(snapshotJson); o.w3[5] = 1e6 + 1; return o; })()],
    ['b1 -1e6 - 1', (() => { const o = JSON.parse(snapshotJson); o.b1[0] = -(1e6 + 1); return o; })()],
    ['w1 not an array', Object.assign({}, base, { w1: 'nope' })],
    ['top-level array', []],
    ['top-level null', null],
  ];
  for (let i = 0; i < cases.length; i++) {
    assertValidationError(() => NeuralNetwork.fromJSON(cases[i][1]), 'fromJSON case: ' + cases[i][0]);
    assertValidationError(() => new NeuralNetwork(cases[i][1]), 'constructor case: ' + cases[i][0]);
  }

  // 1e999 is valid JSON syntax but parses to Infinity — must be rejected, and
  // it must travel through the real string path.
  const json = JSON.stringify(JSON.parse(snapshotJson));
  const keyIdx = json.indexOf('"w1":[') + '"w1":['.length;
  const commaIdx = json.indexOf(',', keyIdx);
  const hacked = json.slice(0, keyIdx) + '1e999' + json.slice(commaIdx);
  assert(JSON.parse(hacked).w1[0] === Infinity, '1e999 parses to Infinity (test sanity)');
  assertValidationError(() => NeuralNetwork.fromJSON(hacked), 'fromJSON rejects 1e999');

  // Oversized weights_json string rejected before parse.
  assertValidationError(
    () => NeuralNetwork.fromJSON(' '.repeat(16385)),
    'fromJSON rejects weights_json longer than 16384 chars'
  );

  // Constructor with no arguments must throw — rng is mandatory, no silent built-in-RNG fallback.
  assertValidationError(() => new NeuralNetwork(), 'constructor without rng throws');

  // Boundary: |v| == 1e6 exactly is ALLOWED (spec: |v| <= 1e6).
  const bound = JSON.parse(snapshotJson);
  bound.w1[0] = 1e6;
  bound.w1[1] = -1e6;
  new NeuralNetwork(bound); // must not throw

  // clone() is a deep copy: mutating the clone must not touch the original.
  const clone = seedNet.clone();
  assert(
    JSON.stringify(clone.getWeights()) === JSON.stringify(seedNet.getWeights()),
    'clone has identical weights'
  );
  clone.setWeights((() => { const o = JSON.parse(snapshotJson); o.w1 = o.w1.map((v) => v + 1); return o; })());
  assert(
    JSON.stringify(clone.getWeights()) !== JSON.stringify(seedNet.getWeights()),
    'clone is a deep copy (independent mutability)'
  );

  // mutate(): rate 1 perturbs every weight; rate 0 perturbs none; deterministic under seeded rng.
  const perturbed = seedNet.clone();
  perturbed.mutate(1, 0.3, mulberry32(11));
  assert(
    JSON.stringify(perturbed.getWeights()) !== JSON.stringify(seedNet.getWeights()),
    'mutate(rate=1) changes weights'
  );
  const untouched = seedNet.clone();
  untouched.mutate(0, 5, mulberry32(11));
  assert(
    JSON.stringify(untouched.getWeights()) === JSON.stringify(seedNet.getWeights()),
    'mutate(rate=0) changes nothing'
  );
  const p1 = seedNet.clone(); p1.mutate(0.25, 0.3, mulberry32(77));
  const p2 = seedNet.clone(); p2.mutate(0.25, 0.3, mulberry32(77));
  assert(
    JSON.stringify(p1.getWeights()) === JSON.stringify(p2.getWeights()),
    'mutate is deterministic under a seeded rng'
  );
});

// =============================================================================
// M1 leg 3 — track build from DEFAULT_CONTROL_POINTS.
// =============================================================================
await leg('leg 3 — track builds; samples, walls, startPose valid', () => {
  const track = new Track(DEFAULT_CONTROL_POINTS, 70);
  assert(Array.isArray(track.samples) && track.samples.length > 100, 'samples built: ' + track.samples.length);
  assert(Array.isArray(track.segments) && track.segments.length > 100, 'wall segments built: ' + track.segments.length);
  assert(Number.isFinite(track.length) && track.length > 1000, 'arc length plausible: ' + track.length.toFixed(1) + ' px');

  // Sample spacing must be ~5 px or tighter (spec) — measure including wrap-around.
  let maxGap = 0;
  for (let i = 1; i < track.samples.length; i++) {
    const dx = track.samples[i].x - track.samples[i - 1].x;
    const dy = track.samples[i].y - track.samples[i - 1].y;
    if (Math.hypot(dx, dy) > maxGap) maxGap = Math.hypot(dx, dy);
  }
  const sFirst = track.samples[0];
  const sLast = track.samples[track.samples.length - 1];
  const wrapGap = Math.hypot(sFirst.x - sLast.x, sFirst.y - sLast.y);
  if (wrapGap > maxGap) maxGap = wrapGap;
  assert(maxGap <= 5.0, 'sample spacing <= 5.0 px (max gap ' + maxGap.toFixed(3) + ' px)');

  const pose = track.startPose();
  assert(Number.isFinite(pose.x) && Number.isFinite(pose.y) && Number.isFinite(pose.heading), 'startPose finite');
  const nc = track.nearestCenterline(pose.x, pose.y);
  assert(
    nc.dist < track.halfWidth,
    'startPose is on the road (dist ' + nc.dist.toFixed(3) + ' < halfWidth ' + track.halfWidth + ')'
  );
  assert(Number.isFinite(nc.arcPos) && Number.isFinite(nc.tangentAngle), 'nearestCenterline returns finite values');
  // arcPos is cumulative arc length — must be within [0, track.length].
  assert(nc.arcPos >= 0 && nc.arcPos <= track.length, 'arcPos within track length');
});

// =============================================================================
// M1 leg 4 — grid raycast is EXACTLY equal to brute-force raycast.
// 10000 seeded random rays; every 5th ray (20%) uses an origin flung off the
// road (jitter ±400 px around the sample point — negative coords included),
// the rest stay near the corridor. Zero mismatches tolerated.
// =============================================================================
await leg('leg 4 — grid raycast === brute-force raycast (10000 rays, 20% off-road)', () => {
  const track = new Track(DEFAULT_CONTROL_POINTS, 70);
  const rng = mulberry32(2024);
  const origins = track.samples.length;
  const RAY_COUNT = 10000;
  const MAX_DIST = 400;
  let mismatches = 0;
  let firstMismatch = '';
  let offRoadCount = 0;
  for (let i = 0; i < RAY_COUNT; i++) {
    const sample = track.samples[Math.floor(((rng() + 0.5) / 4294967296) * origins)];
    const offRoad = i % 5 === 0; // 20% of rays originate off-track
    if (offRoad) offRoadCount++;
    const jitter = offRoad ? 800 : 60; // off-road: ±400 px, can push coords negative
    const jx = (((rng() + 0.5) / 4294967296) - 0.5) * jitter;
    const jy = (((rng() + 0.5) / 4294967296) - 0.5) * jitter;
    const ox = sample.x + jx;
    const oy = sample.y + jy;
    const angle = ((rng() + 0.5) / 4294967296) * Math.PI * 2;
    const grid = track.raycast(ox, oy, angle, MAX_DIST);
    const brute = track._raycastBrute(ox, oy, angle, MAX_DIST);
    if (grid !== brute) {
      mismatches++;
      if (!firstMismatch) {
        firstMismatch =
          'ray ' + i + ' at origin (' + ox.toFixed(2) + ', ' + oy.toFixed(2) +
          ') angle ' + angle.toFixed(4) + ': grid=' + grid + ' brute=' + brute;
      }
    }
    assert(grid >= 0 && grid <= MAX_DIST, 'ray distance within [0, maxDist]');
  }
  assert(offRoadCount === RAY_COUNT / 5, 'off-road share is exactly 20%');
  assert(mismatches === 0, 'grid vs brute-force mismatches: ' + mismatches + (firstMismatch ? ' — first: ' + firstMismatch : ''));
});

// =============================================================================
// M1 leg 5 — car odometer SIGNED integral (Σ speed·dt — net px, may be
// negative), near-MAX_SPEED, collision, reset. FITNESS-V2: the odometer is no
// longer an always-growing counter — reverse driving subtracts from it. Forward
// throttle leg, reverse leg, crash-frozen leg, reset leg.
// Uses a dedicated physics test track (two long straights, ~4000 px x-span) so
// top speed is measurable without relying on steering sophistication. The
// default game track exercises the same physics via the centerline follower.
// =============================================================================
await leg('leg 5 — car odometer signed integral, near MAX_SPEED, collision crash, reset', () => {
  // Extra-long stadium-shaped loop (canvas-agnostic, physics only). The end
  // semicircles have radius ~380 px — comfortably above the car's minimum
  // turn radius at MAX_SPEED (~264 px), so a competent follower can drive it
  // flat-out-ish. Facade straights are ~2100 px each: top speed measurable.
  const physicsTrack = new Track([
    { x: 200, y: 300 }, { x: 800, y: 140 }, { x: 1700, y: 110 },
    { x: 3000, y: 110 }, { x: 3900, y: 140 }, { x: 4550, y: 260 },
    { x: 4700, y: 520 }, { x: 4550, y: 780 }, { x: 3900, y: 850 },
    { x: 3000, y: 880 }, { x: 1700, y: 880 }, { x: 800, y: 850 },
    { x: 200, y: 720 }, { x: 50, y: 500 },
  ], 70);

  const pose = physicsTrack.startPose();

  // Signed integral, arm 1 — forward throttle GROWS the odometer (net px > 0).
  // (The old per-tick `odometer >= prev` assert was DELETED — vacuous by physics:
  // speed is signed and the odometer is the signed integral of it.)
  const fwdCar = new Car(physicsTrack, pose.x, pose.y, pose.heading);
  for (let i = 0; i < 600; i++) {
    fwdCar.update(1 / 120, { gas: 1, steer: 0 });
  }
  assert(fwdCar.odometer > 0, 'forward throttle yields a POSITIVE odometer (got ' + fwdCar.odometer.toFixed(1) + ' px)');

  // Signed integral, arm 2 — reverse throttle drives the car BACKWARDS
  // (speed < 0) and drags the odometer NEGATIVE. A steer=0 car runs off the
  // racing line and crashes into a wall sooner or later — physics is frozen
  // after crash, which does not change the sign conclusion (odometer stays
  // whatever negative value it had).
  const revCar = new Car(physicsTrack, pose.x, pose.y, pose.heading);
  let minSpeed = Infinity;
  for (let i = 0; i < 240; i++) {
    revCar.update(1 / 120, { gas: -1, steer: 0 });
    if (revCar.speed < minSpeed) minSpeed = revCar.speed;
  }
  assert(minSpeed < 0, 'reverse throttle drives the car backwards (min speed ' + minSpeed.toFixed(2) + ' px/s)');
  assert(revCar.odometer < 0, 'reverse run yields a NEGATIVE odometer (got ' + revCar.odometer.toFixed(1) + ' px)');

  // Full-follow drive — the odometer stays large and positive while the car
  // drives the loop forward (signed: mostly-forward motion dominates).
  const car = new Car(physicsTrack, pose.x, pose.y, pose.heading);
  let maxSpeedSeen = 0;
  // 3600 ticks at SIM_DT = 1/120 s → 30 simulated seconds, ~3 laps.
  for (let i = 0; i < 3600; i++) {
    driveAlongTrack(physicsTrack, car);
    if (car.speed > maxSpeedSeen) maxSpeedSeen = car.speed;
  }
  assert(car.odometer > 6000, 'odometer accumulated over 30 s of forward driving (got ' + car.odometer.toFixed(1) + ' px)');
  assert(maxSpeedSeen > 500, 'car reaches near MAX_SPEED (max ' + maxSpeedSeen.toFixed(1) + ' px/s)');
  assert(car.speed <= MAX_SPEED + 1e-9, 'speed never exceeds MAX_SPEED');

  // Collision: center the car exactly on a wall segment midpoint — distance 0 < CAR_RADIUS.
  const seg = physicsTrack.segments[0];
  const wallCar = new Car(physicsTrack, (seg.x1 + seg.x2) / 2, (seg.y1 + seg.y2) / 2, 0);
  wallCar.update(1 / 120, { gas: 0, steer: 0 });
  assert(wallCar.crashed, 'car overlapping a wall is flagged crashed');
  assert(wallCar.speed === 0, 'crashed car speed is zeroed');
  assert(wallCar.odometer === 0, 'a crashed car never accumulates odometer (update() is a no-op)');
  wallCar.update(1 / 120, { gas: 1, steer: 0 });
  assert(wallCar.odometer === 0, 'a crashed car stays at zero odometer even with gas applied');

  // reset() must restore the start pose and clear crash state.
  wallCar.reset();
  const poseAfter = wallCar;
  assert(poseAfter.crashed === false, 'reset() clears crashed');
  assert(poseAfter.speed === 0 && poseAfter.odometer === 0, 'reset() zeroes speed and odometer');

  // Driver seat is defined (default = car position, documented in car.js).
  const seat = car.driverSeat();
  assert(Number.isFinite(seat.x) && Number.isFinite(seat.y), 'driverSeat returns finite coords');
});

// =============================================================================
// M1 leg 6 — track hash: canonicalJSON -> cyrb53 stability + tamper detection.
// =============================================================================
await leg('leg 6 — track hash stable across roundtrips; tamper detected', () => {
  const track = new Track(DEFAULT_CONTROL_POINTS, 70);
  const h1 = track.hash();
  assert(typeof h1 === 'string' && h1.length === 16, 'hash is a 16-char hex string, got ' + h1);

  const round1 = Track.fromJSON(track.toJSON());
  assert(round1.hash() === h1, 'hash stable after fromJSON(toJSON())');

  const round2 = Track.fromJSON(round1.toJSON());
  assert(round2.hash() === h1, 'hash stable after a double roundtrip');

  // Tamper with one coordinate by 1e-7 px — far above the 9-decimal canonical
  // rounding — and the hash MUST change.
  const tampered = JSON.parse(track.toJSON());
  tampered.points[0].x = tampered.points[0].x + 1e-7;
  assert(
    cyrb53(canonicalJSON(tampered)) !== h1,
    'tampered coordinate produces a different hash'
  );

  // cyrb53 is deterministic.
  assert(cyrb53('car-racer') === cyrb53('car-racer'), 'cyrb53 deterministic');
  assert(cyrb53('car-racer') !== cyrb53('car-racep'), 'cyrb53 avalanche on single char');
});

// =============================================================================
// M1 leg 7 — save/load roundtrip logic (the former localStorage leg):
// the exact serialization the editor will persist is exercised end-to-end here.
// =============================================================================
await leg('leg 7 — toJSON/fromJSON roundtrip logic + key order + validation', () => {
  const track = new Track(DEFAULT_CONTROL_POINTS, 70);
  const json = track.toJSON();

  const t2 = Track.fromJSON(json);
  assert(
    JSON.stringify(t2.controlPoints) === JSON.stringify(track.controlPoints),
    'control points survive toJSON/fromJSON'
  );
  assert(t2.halfWidth === track.halfWidth, 'halfWidth survives roundtrip');
  assert(
    Math.abs(t2.length - track.length) < 1e-9,
    'arc length survives roundtrip (' + t2.length.toFixed(6) + ' vs ' + track.length.toFixed(6) + ')'
  );

  // Fixed key order in the canonical serialization: points, then halfWidth.
  const parsed = JSON.parse(json);
  assert(Object.keys(parsed)[0] === 'points' && Object.keys(parsed)[1] === 'halfWidth', 'top-level key order points,halfWidth');
  assert(Object.keys(parsed.points[0])[0] === 'x' && Object.keys(parsed.points[0])[1] === 'y', 'point key order x,y');

  // fromJSON validation battery.
  assertValidationError(() => Track.fromJSON('not json {'), 'fromJSON rejects invalid JSON');
  assertValidationError(() => Track.fromJSON('[1,2,3]'), 'fromJSON rejects a top-level array');
  assertValidationError(() => Track.fromJSON('null'), 'fromJSON rejects null');
  assertValidationError(
    () => Track.fromJSON(JSON.stringify({ points: [{ x: 0, y: 0 }, { x: 10, y: 10 }], halfWidth: 70 })),
    'fromJSON rejects fewer than 3 points'
  );
  const pts513 = Array.from({ length: 513 }, (_, i) => ({ x: i, y: (i * 37) % 500 }));
  assertValidationError(
    () => Track.fromJSON(JSON.stringify({ points: pts513, halfWidth: 70 })),
    'fromJSON rejects 513 points'
  );
  assertValidationError(
    () => Track.fromJSON(JSON.stringify({ points: [{ x: 0, y: 0 }, { x: 1e6 + 1, y: 1 }, { x: 2, y: 2 }], halfWidth: 70 })),
    'fromJSON rejects |coord| > 1e6'
  );
  assertValidationError(
    () => Track.fromJSON(JSON.stringify({ points: [{ x: 0, y: 0 }, { x: 1, y: 1 }, { x: 2, y: 2 }], halfWidth: 0 })),
    'fromJSON rejects halfWidth < 1'
  );
  assertValidationError(
    () => Track.fromJSON(JSON.stringify({ points: [{ x: 0, y: 0 }, { x: 1, y: 1 }, { x: 2, y: 2 }], halfWidth: 2001 })),
    'fromJSON rejects halfWidth > 2000'
  );
  assertValidationError(() => Track.fromJSON(' '.repeat(262145)), 'fromJSON rejects payloads > 262144 chars');
  // Strict boundary: halfWidth must be a JSON number, never a numeric string.
  assertValidationError(
    () => Track.fromJSON(JSON.stringify({ points: [{ x: 0, y: 0 }, { x: 1, y: 1 }, { x: 2, y: 2 }], halfWidth: '70' })),
    'fromJSON rejects string halfWidth'
  );
  assertValidationError(
    () => new Track([{ x: 0, y: 0 }, { x: Infinity, y: 1 }, { x: 2, y: 2 }], 70),
    'constructor rejects non-finite coordinates'
  );
});

// =============================================================================
// M1 leg 8 — NN determinism: same seed → byte-identical weights.
// =============================================================================
await leg('leg 8 — NN determinism: same seed → byte-identical getWeights()', () => {
  const n1 = new NeuralNetwork(mulberry32(4242));
  const n2 = new NeuralNetwork(mulberry32(4242));
  const n3 = new NeuralNetwork(mulberry32(4243));
  assert(
    JSON.stringify(n1.getWeights()) === JSON.stringify(n2.getWeights()),
    'two nets seeded with mulberry32(4242) are byte-identical'
  );
  assert(
    JSON.stringify(n1.getWeights()) !== JSON.stringify(n3.getWeights()),
    'different seeds produce different weights'
  );
  // F-2: the plan's `nn.randomInit(rng)` API behaves exactly like new NeuralNetwork(rng).
  assert(typeof NeuralNetwork.randomInit === 'function', 'NeuralNetwork.randomInit exists as a static');
  const viaStatic = NeuralNetwork.randomInit(mulberry32(4242));
  assert(
    JSON.stringify(viaStatic.getWeights()) === JSON.stringify(n1.getWeights()),
    'NeuralNetwork.randomInit(rng) matches new NeuralNetwork(rng) byte-for-byte'
  );
});

// =============================================================================
// M1 leg 9 — SensorFan contract: 5 rays, offsets [-70,-35,0,35,70], capped at SENSOR_MAX.
// (Beyond the mandated 8 — sensors.js is in scope and untested otherwise.)
// =============================================================================
await leg('leg 9 — SensorFan: 5 readings, offsets, norm in [0,1]; FIT-V6 controller/sensor hairpin falsifier', () => {
  assert(
    JSON.stringify(SENSOR_ANGLES_DEG) === JSON.stringify([-70, -35, 0, 35, 70]),
    'SENSOR_ANGLES_DEG is exactly [-70,-35,0,35,70]'
  );
  const track = new Track(DEFAULT_CONTROL_POINTS, 70);
  const pose = track.startPose();
  const car = new Car(track, pose.x, pose.y, pose.heading);
  const sensors = new SensorFan(car, track, SENSOR_MAX);
  assert(SENSOR_MAX === 300, 'SENSOR_MAX is 300 px');

  const reading = sensors.read();
  assert(reading.norm.length === 5 && reading.px.length === 5, 'exactly 5 sensor readings');
  for (let i = 0; i < 5; i++) {
    assert(Number.isFinite(reading.px[i]), 'px[' + i + '] finite');
    assert(reading.px[i] >= 0 && reading.px[i] <= SENSOR_MAX, 'px[' + i + '] within [0, SENSOR_MAX]');
    assert(reading.norm[i] >= 0 && reading.norm[i] <= 1, 'norm[' + i + '] within [0, 1]');
    assert(
      Math.abs(reading.norm[i] * SENSOR_MAX - reading.px[i]) < 1e-3 * SENSOR_MAX,
      'norm[' + i + '] consistent with px[' + i + '] / SENSOR_MAX'
    );
  }
  // Center ray (index 2, offset 0) must equal a direct raycast along the heading.
  const direct = track.raycast(car.x, car.y, car.heading, SENSOR_MAX);
  assert(
    Math.abs(reading.px[2] - direct) < 1e-9,
    'center sensor matches raycast along heading (' + reading.px[2] + ' vs ' + direct + ')'
  );

  // ---- FIT-V6 controller/sensor hairpin falsifier ---------------------------
  // The walls are the sole authoritative collision/sensor geometry. The six
  // cycle-3 arc-built F1 templates (91-96 control points, one genuine >=150°
  // hairpin each) clear the clipped inner wall by >=18px at EVERY centreline
  // sample. Place the car at EVERY centreline sample across the whole loop
  // (apex included) — no skips — and assert: min centreline-to-CLIPPED-wall
  // clearance >= CAR_RADIUS + 4 (18px), no phantom collision, the car stays
  // inside the road, all five sensors are finite, and each sensor equals the
  // brute-force ray (grid == brute — no phantom SENSOR_MAX reading where a
  // wall exists). A short closed-loop pure-pursuit drive then confirms the
  // controller can move off the standing start without colliding or
  // desynchronising the sensor fan. (The 10,000-ray grid==brute and wall-byte
  // golden corpus live in tests/track-clipped.mjs; the generator feasibility
  // gates live in tests/rotation-v6.mjs.)
  const Rotation = require('../js/rotation.js').Rotation;
  const templates = Rotation.V5_TEMPLATES;
  assert(Array.isArray(templates) && templates.length === 6, 'six arc-built hairpin templates exported');
  const angles = SENSOR_ANGLES_DEG.map((d) => d * Math.PI / 180);
  let leg9MinClearance = Infinity;
  let leg9SamplesChecked = 0;
  for (let t = 0; t < templates.length; t++) {
    for (const d of [0, 50, 100]) {
      const hw = Rotation.halfWidthForDifficulty(d);
      const pts = templates[t].map((p) => ({ x: p[0], y: p[1] }));
      const hairpin = new Track(pts, hw);
      assert(Rotation.v6ApexRadius(pts) >= 75, 'template ' + t + ' apex radius >= 75 (car min turning radius)');
      const S = hairpin.samples;
      // Cycle-3 gate: check EVERY centreline sample (stride 1, no skip). With
      // the arc-built templates the minimum clearance is ~25px, so the
      // un-skipped leg passes; any regression back toward a cusp (clearance
      // < 18px) fails here instead of being silently skipped.
      for (let i = 0; i < S.length; i++) {
        const s = S[i];
        let clearance = Infinity;
        for (const seg of hairpin.segments) {
          const dx = seg.x2 - seg.x1, dy = seg.y2 - seg.y1;
          const u = Math.max(0, Math.min(1, ((s.x - seg.x1) * dx + (s.y - seg.y1) * dy) / (dx * dx + dy * dy || 1)));
          clearance = Math.min(clearance, Math.hypot(s.x - (seg.x1 + u * dx), s.y - (seg.y1 + u * dy)));
        }
        leg9MinClearance = Math.min(leg9MinClearance, clearance);
        leg9SamplesChecked++;
        assert(clearance >= CAR_RADIUS + 4,
          't' + t + ' d' + d + ' sample ' + i + ' — centreline-to-CLIPPED-wall clearance ' + clearance.toFixed(3) + 'px >= ' + (CAR_RADIUS + 4) + 'px (ALL samples, no skips)');
        const probeCar = new Car(hairpin, s.x, s.y, s.tangent);
        probeCar.update(1 / 120, { gas: 0, steer: 0 }); // zero-input tick: collision only
        assert(!probeCar.crashed, 't' + t + ' d' + d + ' sample ' + i + ' — no phantom collision on centerline');
        const nc = hairpin.nearestCenterline(probeCar.x, probeCar.y);
        assert(nc.dist <= hw, 't' + t + ' d' + d + ' sample ' + i + ' — car stays inside the road');
        const fan = new SensorFan(probeCar, hairpin, SENSOR_MAX).read();
        assert(fan.px.length === 5 && fan.norm.length === 5, 'five sensor readings on the hairpin');
        for (let k = 0; k < 5; k++) {
          const brute = hairpin._raycastBrute(probeCar.x, probeCar.y, probeCar.heading + angles[k], SENSOR_MAX);
          assert(Number.isFinite(fan.px[k]) && fan.px[k] >= 0 && fan.px[k] <= SENSOR_MAX,
            't' + t + ' d' + d + ' sample ' + i + ' sensor ' + k + ' finite/in-range');
          assert(Math.abs(fan.px[k] - brute) <= 1e-6,
            't' + t + ' d' + d + ' sample ' + i + ' sensor ' + k + ' grid==brute (no phantom ' + SENSOR_MAX + 'px)');
        }
      }
    }
    // Closed-loop drive: short, low-speed pure pursuit from a clear centerline
    // sample — must not collide and sensors must keep matching brute each step.
    const hw = Rotation.halfWidthForDifficulty(50);
    const circuit = new Track(templates[t].map((p) => ({ x: p[0], y: p[1] })), hw);
    let start = 0, bestClear = -1;
    for (let i = 0; i < circuit.samples.length; i += 3) {
      const s = circuit.samples[i];
      let clear = Infinity;
      for (const seg of circuit.segments) {
        const dx = seg.x2 - seg.x1, dy = seg.y2 - seg.y1;
        const u = Math.max(0, Math.min(1, ((s.x - seg.x1) * dx + (s.y - seg.y1) * dy) / (dx * dx + dy * dy || 1)));
        clear = Math.min(clear, Math.hypot(s.x - (seg.x1 + u * dx), s.y - (seg.y1 + u * dy)));
      }
      if (clear > bestClear) { bestClear = clear; start = i; }
    }
    const st = circuit.samples[start];
    const driver = new Car(circuit, st.x, st.y, st.tangent);
    const fan = new SensorFan(driver, circuit, SENSOR_MAX);
    let target = (start + 8) % circuit.samples.length;
    for (let step = 0; step < 120 && !driver.crashed; step++) {
      while (Math.hypot(driver.x - circuit.samples[target].x, driver.y - circuit.samples[target].y) < 25) {
        target = (target + 1) % circuit.samples.length;
      }
      const err = wrapToPi(Math.atan2(circuit.samples[target].y - driver.y, circuit.samples[target].x - driver.x) - driver.heading);
      driver.update(1 / 120, { gas: driver.speed < 25 ? 0.4 : 0, steer: Math.max(-1, Math.min(1, err * 2)) });
      if (step % 10 === 0) {
        const r = fan.read();
        for (let k = 0; k < 5; k++) {
          const brute = circuit._raycastBrute(driver.x, driver.y, driver.heading + angles[k], SENSOR_MAX);
          assert(Math.abs(r.px[k] - brute) <= 1e-6, 't' + t + ' drive step ' + step + ' sensor ' + k + ' grid==brute');
        }
      }
    }
    assert(!driver.crashed, 't' + t + ' closed-loop drive — no collision from standing start');
    assert(Math.abs(driver.odometer) > 10, 't' + t + ' closed-loop drive — the car actually moved');
  }
  // Cycle-3 gate (verbatim evidence): every centreline sample of every
  // template × difficulty was probed (no skips) and met the >=18px floor.
  console.log('       (leg 9 — ' + leg9SamplesChecked + ' centreline samples, 0 skipped; min clipped-wall clearance '
    + leg9MinClearance.toFixed(3) + 'px >= ' + (CAR_RADIUS + 4) + 'px across ALL samples)');
  assert(leg9MinClearance >= CAR_RADIUS + 4,
    'leg 9 — min clipped-wall clearance ' + leg9MinClearance.toFixed(3) + 'px >= ' + (CAR_RADIUS + 4) + 'px across ALL ' + leg9SamplesChecked + ' samples (no skips)');
});

// =============================================================================
// M2 leg 10 — oscillating car killed by stuck-kill (progress-based stagnation).
// Two mandated static policies die at exactly STUCK_TIMEOUT_S simulated
// seconds; a third, genuinely oscillating policy keeps the run alive (net
// centerline progress < PROGRESS_ADVANCE_PX over the timer window) until the
// stagnation timer trips it. FITNESS-V2: distance (fitness) is SIGNED — an
// oscillator's net can be either sign — so the kill-time assertion pins
// finite fitness + stuck (NOT a positive distance).
//
// Note: STUCK_TIMEOUT_S is a sim-seconds constraint, and the eval loop never
// reads the wall clock — timeAlive here is SIMULATED seconds.
// =============================================================================
await leg('leg 10 — stuck-kill: oscillating and stationary cars die at STUCK_TIMEOUT_S', () => {
  const track = new Track(DEFAULT_CONTROL_POINTS, 70);
  const slack = 2 * SIM_DT + 1e-6;

  // Mandated case 1 — gas 0 / steer 1 (stationary in this physics model: gas
  // 0 ⇒ zero throttle force, speed stays 0, odometer never grows).
  const r1 = Evolution.evaluateDriver(track, null, () => [0, 1]);
  assert(r1.stuck === true, 'gas=0/steer=1 policy is killed by the stuck-kill');
  assert(r1.crashed === false, 'gas=0/steer=1 policy is not reported as crashed');
  assert(
    r1.timeAlive <= STUCK_TIMEOUT_S + slack,
    'gas=0/steer=1 killed within STUCK_TIMEOUT_S + 2*SIM_DT (timeAlive ' + r1.timeAlive.toFixed(4) + ')'
  );

  // Mandated case 2 — gas 0 / steer 0 (stationary).
  const r0 = Evolution.evaluateDriver(track, null, () => [0, 0]);
  assert(r0.stuck === true, 'stationary policy is killed by the stuck-kill');
  assert(r0.crashed === false, 'stationary policy is not reported as crashed');
  assert(
    r0.timeAlive <= STUCK_TIMEOUT_S + slack,
    'stationary policy killed within STUCK_TIMEOUT_S + 2*SIM_DT (timeAlive ' + r0.timeAlive.toFixed(4) + ')'
  );

  // Oscillating car — gas and steer swing opposite phases, so the car weaves
  // forward/backward along the start straight: the signed odometer keeps
  // moving (either sign), but NET centerline arc progress stays
  // below PROGRESS_ADVANCE_PX long enough that the stagnation timer trips.
  // The tight STUCK_TIMEOUT_S bound does NOT apply here — this car does make
  // occasional +20 px net-progress resets, so timeAlive may exceed 12 s; it
  // must still be killed well before the EVAL_CAP_S cap.
  let k = 0;
  const rOsc = Evolution.evaluateDriver(track, null, () => {
    k += 1;
    return [Math.sin(k * 0.11) * 0.6, Math.sin(k * 0.07 + 1) * 0.6];
  });
  assert(rOsc.stuck === true, 'oscillating policy is killed by the stuck-kill');
  assert(rOsc.crashed === false, 'oscillating policy dies to the timer, not to a wall');
  assert(Number.isFinite(rOsc.distance) && rOsc.stuck === true, 'oscillating policy: distance (fitness) is finite at kill time (distance ' + rOsc.distance.toFixed(1) + ')');
  assert(rOsc.timeAlive < EVAL_CAP_S, 'oscillating car killed by the timer, not by the EVAL_CAP_S cap (timeAlive ' + rOsc.timeAlive.toFixed(2) + ' s)');

  // Stagnation must be measured on UNWRAPPED arc progress — sanity-check the
  // track length so a closed lap cannot reset the timer by aliasing arcPos.
  assert(track.length > 1000, 'sanity: default track length ' + track.length.toFixed(1) + ' px');
  assert(PROGRESS_ADVANCE_PX >= 1, 'sanity: PROGRESS_ADVANCE_PX is sane (' + PROGRESS_ADVANCE_PX + ' px)');
});

// =============================================================================
// M2 leg 11 — gen-1 population: 20 nets, all 7-16-16-2, weights finite.
// =============================================================================
await leg('leg 11 — gen-1 population: 20 nets, 7-16-16-2, finite weights', () => {
  const evo = new Evolution(4242);
  const pop = evo.randomPopulation();
  assert(evo.generation === 1, 'randomPopulation() sets generation 1, got ' + evo.generation);
  assert(pop.length === POPULATION, 'population has exactly ' + POPULATION + ' nets, got ' + pop.length);
  assert(POPULATION === 20, 'POPULATION constant is 20');

  const totals = [112, 16, 256, 16, 32, 2];
  for (let n = 0; n < pop.length; n++) {
    const w = pop[n].getWeights();
    assert(w.architecture === NN_ARCH, 'net ' + n + ' architecture is ' + NN_ARCH + ', got ' + JSON.stringify(w.architecture));
    const lens = [w.w1.length, w.b1.length, w.w2.length, w.b2.length, w.w3.length, w.b3.length];
    assert(
      lens.every((len, i) => len === totals[i]),
      'net ' + n + ' tensor shapes are 112/16/256/16/32/2, got ' + lens.join('/')
    );
    const all = [].concat(w.w1, w.b1, w.w2, w.b2, w.w3, w.b3);
    assert(all.length === 434, 'net ' + n + ' holds exactly 434 floats, got ' + all.length);
    for (let i = 0; i < all.length; i++) {
      assert(Number.isFinite(all[i]), 'net ' + n + ' weight ' + i + ' is finite, got ' + all[i]);
    }
  }
});

// =============================================================================
// M2 leg 12 — selection + spawnNextGeneration. Fabricated results put the
// highest distance on driver #7; after recordEvaluation + spawn:
//   child0 is a BYTE-IDENTICAL clone of the parent's best weights (elitism /
//   replay anchor), children 1..19 keep the 7-16-16-2 shapes, the population
//   stays 20 long, and the generation counter increments 1 → 2.
// =============================================================================
await leg('leg 12 — selection + spawnNextGeneration: child0 exact clone, 1 → 2', () => {
  const evo = new Evolution(7);
  evo.randomPopulation();
  const parentBest = evo.population[7];

  // Fabricate a results table where driver #7 wins on distance, unambiguously.
  const results = [];
  for (let i = 0; i < POPULATION; i++) {
    results.push({
      distance: i === 7 ? 10000 : 100 + i,
      topSpeed: i === 7 ? 500 : 100,
      timeAlive: i === 7 ? 45 : 30,
      crashed: false,
      stuck: false,
    });
  }
  const picked = evo.recordEvaluation(results);
  assert(picked === 7, 'selectBest picks fabricated winner #7, got ' + picked);
  assert(Evolution.selectBest(results) === 7, 'static selectBest agrees on #7');

  const bestJson = JSON.stringify(parentBest.getWeights());
  evo.spawnNextGeneration();

  assert(evo.population.length === POPULATION, 'population length stays 20, got ' + evo.population.length);
  assert(evo.generation === 2, 'generation increments 1 → 2, got ' + evo.generation);
  assert(evo.population[0] !== parentBest, 'child0 is a clone, not the parent object');
  assert(
    JSON.stringify(evo.population[0].getWeights()) === bestJson,
    'child0 weights are byte-identical to the parent best weights'
  );
  for (let i = 1; i < evo.population.length; i++) {
    const w = evo.population[i].getWeights();
    assert(w.architecture === NN_ARCH, 'child ' + i + ' architecture is ' + NN_ARCH);
    assert(
      w.w1.length === 112 && w.w2.length === 256 && w.w3.length === 32 &&
      w.b1.length === 16 && w.b2.length === 16 && w.b3.length === 2,
      'child ' + i + ' keeps the 7-16-16-2 tensor shapes'
    );
    const all = [].concat(w.w1, w.b1, w.w2, w.b2, w.w3, w.b3);
    for (let j = 0; j < all.length; j++) {
      assert(Number.isFinite(all[j]), 'child ' + i + ' weight ' + j + ' is finite');
    }
  }
  // Mutation is real (overwhelmingly certain at rate 0.25 over 434 floats) and deterministic.
  const changed = evo.population.slice(1).some(
    (child) => JSON.stringify(child.getWeights()) !== bestJson
  );
  assert(changed, 'at least one of children 1..19 actually mutated');
  assert(MUTATION_RATE === 0.25 && MUTATION_SCALE === 0.30, 'mutation constants are 0.25 / 0.30');
});

// =============================================================================
// M2 leg 13 — determinism DOUBLE-RUN: tests/determinism.mjs spawned as TWO
// fresh node child processes; their stdout payloads (gens 1..3 bestWeights,
// seed 4242) must be byte-identical. This is the M2 "same seed, two fresh
// processes, byte-identical weights_json for gens 1..3" gate.
// =============================================================================
await leg('leg 13 — determinism double-run: two fresh processes, byte-identical stdout', () => {
  const runOnce = () =>
    execFileSync(process.execPath, ['tests/determinism.mjs'], {
      cwd: REPO_ROOT,
      maxBuffer: 32 * 1024 * 1024,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

  const out1 = runOnce();
  const out2 = runOnce();
  const len1 = out1.length;
  const len2 = out2.length;
  console.log('       double-run compared ' + len1 + ' bytes vs ' + len2 + ' bytes — byte-for-byte');
  assert(len1 > 0, 'determinism probe produced non-empty output (got ' + len1 + ' bytes)');
  assert(
    out1.equals(out2),
    'two fresh processes with seed 4242 produced identical gens 1..3 bestWeights (' + len1 + ' vs ' + len2 + ' bytes)'
  );
});

// =============================================================================
// M2 leg 14 — MID-RUN stop on the MessageChannel fallback: stop after the
// FIRST driver finishes, before the generation completes. The in-flight
// generation must be DISCARDED: no 'generation' payload, nothing logged, no
// onProgress, 'stopped' acked, Trainer.state === 'stopped'. Instrumentation is
// test-scoped (evaluateDriver is wrapped and restored in finally).
// =============================================================================
await leg('leg 14 — fallback core: mid-run stop discards the generation (no payload, acked, stopped)', async () => {
  if (typeof Worker !== 'undefined') {
    throw new Error('leg 14 requires node (no global Worker) — it exercises the in-process fallback path');
  }
  const track = new Track(DEFAULT_CONTROL_POINTS, 70);
  const backendMessages = [];
  const loggerCalls = [];
  const progressCalls = [];

  const originalEvaluateDriver = Evolution.evaluateDriver;
  let driversSeen = 0;
  let trainer = null;
  // Wrap the static evaluator: on the FIRST driver, call Trainer.stop() from
  // INSIDE the evaluation — the stop therefore lands after driver 1 completes
  // and before drivers 2..20 run (the fallback checks stopRequested between
  // drivers).
  Evolution.evaluateDriver = function (...args) {
    driversSeen += 1;
    const result = originalEvaluateDriver.apply(this, args);
    if (driversSeen === 1 && trainer) trainer.stop();
    return result;
  };

  try {
    trainer = new Trainer(track, {
      lineage: { seed: 4242, rotationSeed: 4242, generatorVersion: 1 },
      onBanner() {},
      onProgress(p) { progressCalls.push(p); },
      logger: { logGeneration(payload) { loggerCalls.push(payload.generation); } },
    });
    const originalOnBackendMessage = trainer._onBackendMessage.bind(trainer);
    trainer._onBackendMessage = (msg) => {
      if (msg && msg.type) backendMessages.push(msg.type);
      return originalOnBackendMessage(msg);
    };
    trainer.start(1);
    await waitFor(() => backendMessages.includes('stopped'), "stopped ack from the fallback core", 60000);
  } finally {
    Evolution.evaluateDriver = originalEvaluateDriver;
  }

  assert(driversSeen === 1, 'exactly one driver was evaluated before the stop landed, got ' + driversSeen);
  assert(!backendMessages.includes('generation'), 'NO generation payload was emitted for the discarded generation');
  assert(backendMessages.includes('stopped'), 'stopped ack was emitted');
  assert(loggerCalls.length === 0, 'the discarded generation was never logged');
  assert(progressCalls.length === 0, 'onProgress was never called for a discarded generation');
  assert(trainer.state === 'stopped', "Trainer.state === 'stopped', got " + trainer.state);
  assert(trainer.currentResults === null && trainer.generationsLogged === 0, 'no generation was recorded or logged');
});

// =============================================================================
// M2 leg 15 — two-core equivalence: the REAL js/train-worker.js (executed
// in-process through a stubbed importScripts/self/postMessage harness) and
// the MessageChannel fallback core both run ONE identical scripted protocol
// (init → start(2) → next → gen 2 → stop). Every emitted 'generation' payload
// must be BYTE-IDENTICAL between the two cores (same seed, same JS engine).
// =============================================================================
await leg('leg 15 — two-core equivalence: Worker harness vs MessageChannel fallback, byte-identical payloads', async () => {
  // FIT-V4 — both cores now require an explicit generator version on the
  // init/resume handshake (strict parser, never a default). This is a legacy
  // V1 lineage fixture, so supply the canonical integer 1.
  const shared = { controlPoints: JSON.parse(JSON.stringify(DEFAULT_CONTROL_POINTS)), halfWidth: 70, rotationSeed: 4242, generatorVersion: 1 };
  const SEED = 4242;

  // ---- core A: js/train-worker.js through the stubbed worker global surface
  const workerMessages = [];
  const savedSelf = globalThis.self;
  const savedPostMessage = globalThis.postMessage;
  const savedImportScripts = globalThis.importScripts;
  const savedOnmessage = globalThis.onmessage;
  try {
    globalThis.self = globalThis;
    globalThis.postMessage = (msg) => workerMessages.push(msg);
    globalThis.importScripts = (...urls) => {
      for (const u of urls) {
        require(fileURLToPath(new URL('../js/' + u.replace(/^\.\//, ''), import.meta.url)));
      }
    };
    require(fileURLToPath(new URL('../js/train-worker.js', import.meta.url)));
    const sendWorker = (msg) => globalThis.self.onmessage({ data: msg });
    sendWorker({ type: 'init', ...shared, seed: SEED });
    sendWorker({ type: 'start', maxGenerations: 2 });
    await waitFor(() => workerMessages.filter((m) => m.type === 'generation').length === 1, 'worker gen 1 payload');
    sendWorker({ type: 'next' });
    await waitFor(() => workerMessages.filter((m) => m.type === 'generation').length === 2, 'worker gen 2 payload');
    sendWorker({ type: 'stop' });
    await waitFor(() => workerMessages.some((m) => m.type === 'stopped'), 'worker stopped ack');
  } finally {
    if (savedSelf === undefined) delete globalThis.self; else globalThis.self = savedSelf;
    if (savedPostMessage === undefined) delete globalThis.postMessage; else globalThis.postMessage = savedPostMessage;
    if (savedImportScripts === undefined) delete globalThis.importScripts; else globalThis.importScripts = savedImportScripts;
    if (savedOnmessage === undefined) delete globalThis.onmessage; else globalThis.onmessage = savedOnmessage;
  }

  // ---- core B: the MessageChannel fallback core, driven through a real channel
  const fallbackMessages = [];
  const channel = new MessageChannel();
  const core = createFallbackCore();
  channel.port2.onmessage = (event) => core(event.data, (out) => channel.port2.postMessage(out));
  channel.port1.onmessage = (event) => fallbackMessages.push(event.data);
  if (typeof channel.port1.start === 'function') channel.port1.start();
  if (typeof channel.port2.start === 'function') channel.port2.start();
  const sendFallback = (msg) => channel.port1.postMessage(msg);
  sendFallback({ type: 'init', ...shared, seed: SEED });
  sendFallback({ type: 'start', maxGenerations: 2 });
  await waitFor(() => fallbackMessages.filter((m) => m.type === 'generation').length === 1, 'fallback gen 1 payload');
  sendFallback({ type: 'next' });
  await waitFor(() => fallbackMessages.filter((m) => m.type === 'generation').length === 2, 'fallback gen 2 payload');
  sendFallback({ type: 'stop' });
  await waitFor(() => fallbackMessages.some((m) => m.type === 'stopped'), 'fallback stopped ack');

  const workerGens = workerMessages.filter((m) => m.type === 'generation');
  const fbGens = fallbackMessages.filter((m) => m.type === 'generation');

  // Non-vacuous structural assertions on BOTH cores.
  for (const m of [...workerGens, ...fbGens]) {
    assert(Array.isArray(m.payload.results) && m.payload.results.length === POPULATION, 'generation payload carries 20 results');
    assert(
      m.payload.bestWeights && m.payload.bestWeights.w1 && m.payload.bestWeights.w1.length === 112,
      'payload carries the full bestWeights tensor set (w1 length ' + (m.payload.bestWeights && m.payload.bestWeights.w1 && m.payload.bestWeights.w1.length) + ')'
    );
    assert(Number.isInteger(m.payload.rngState) && m.payload.rngState >= 0 && m.payload.rngState <= 4294967295, 'rngState is a uint32');
    assert(Number.isFinite(m.payload.results[m.payload.bestIndex].distance), 'best distance is finite');
    assert(Number.isInteger(m.payload.rotationIndex) && m.payload.rotationIndex >= 0, 'payload carries a rotationIndex');
    assert(Number.isInteger(m.payload.difficulty) && m.payload.difficulty >= 0 && m.payload.difficulty <= 100, 'payload carries a difficulty');
    assert(typeof m.payload.trackHash === 'string' && /^[0-9a-f]{16}$/.test(m.payload.trackHash), 'payload carries a 16-hex trackHash, got ' + m.payload.trackHash);
  }
  // DEFAULT_TRACK_HASH is computed lazily later in the file (M3 section) —
  // derive the pin hash here from the same eager globals.
  const pinHash = new Track(DEFAULT_CONTROL_POINTS, 70).hash();
  assert(workerGens[0].payload.rotationIndex === 0 && workerGens[0].payload.difficulty === 0 && workerGens[0].payload.trackHash === pinHash, 'gen 1 (rotation 0) carries the pin hash / rotation 0 / difficulty 0');
  assert(workerGens.length === 2 && fbGens.length === 2, 'both cores emitted exactly 2 generation payloads (worker ' + workerGens.length + ' / fallback ' + fbGens.length + ')');
  assert(workerGens.map((m) => m.payload.generation).join() === '1,2', 'worker emitted generations 1,2 in order');
  assert(fbGens.map((m) => m.payload.generation).join() === '1,2', 'fallback emitted generations 1,2 in order');

  // Byte-equality — same seed, same JS engine: the two cores must be
  // indistinguishable in every emitted generation payload.
  for (let i = 0; i < 2; i++) {
    assert(
      JSON.stringify(workerGens[i].payload) === JSON.stringify(fbGens[i].payload),
      'generation ' + (i + 1) + ' payload byte-identical across the two cores'
    );
  }
});

// =============================================================================
// M2 leg 16 — STRICT uint32 gate (H-1): resumeFromBest / setState accept only
// integer numbers in [0, 4294967295] or decimal TEXT /^(0|[1-9][0-9]{0,9})$/
// ≤ 4294967295; everything else throws ValidationError BEFORE any state is
// committed. Positive case: decimal TEXT '3894656591' round-trips through
// setState and resume produces byte-identical children for TEXT vs number.
// =============================================================================
await leg('leg 16 — strict uint32 gate: resumeFromBest/setState accept only uint32 (H-1)', () => {
  const weights = new NeuralNetwork(mulberry32(1)).getWeights();
  const badStates = [null, false, '', '0x10', '1e3', NaN, 1.5, '4294967296', {}];
  for (const bad of badStates) {
    assertValidationError(() => Evolution.resumeFromBest(weights, 1, bad), 'resumeFromBest rejects rngState ' + JSON.stringify(bad));
  }
  assertValidationError(() => mulberry32(0).setState(null), 'setState rejects null');
  assertValidationError(() => mulberry32(0).setState('  '), 'setState rejects whitespace string');
  assertValidationError(() => mulberry32(0).setState('00'), 'setState rejects leading zeros');
  assertValidationError(() => mulberry32(0).setState(4294967296), 'setState rejects 4294967296');
  assertValidationError(() => mulberry32(0).setState(new Number(7)), 'setState rejects Number objects');
  assertValidationError(() => new Evolution(1, { rngState: '0x10' }), 'Evolution constructor rejects hex rngState');

  // Positive case — the DB TEXT round-trip: '3894656591' (< 4294967295).
  const rng = mulberry32(0);
  rng.setState('3894656591');
  assert(rng.state() === 3894656591, 'decimal TEXT "3894656591" round-trips into setState (got ' + rng.state() + ')');
  const evoA = Evolution.resumeFromBest(weights, 1, '3894656591');
  const evoB = Evolution.resumeFromBest(weights, 1, 3894656591);
  assert(evoA.population.length === POPULATION && evoB.population.length === POPULATION, 'resume populations are POPULATION long');
  assert(evoA.generation === 2 && evoB.generation === 2, 'resume generation is completed + 1');
  for (let i = 0; i < evoA.population.length; i++) {
    assert(
      JSON.stringify(evoA.population[i].getWeights()) === JSON.stringify(evoB.population[i].getWeights()),
      'child ' + i + ' byte-identical for TEXT vs numeric rngState'
    );
  }
});

// =============================================================================
// M2 leg 17 — persist-throw latch (M-1): an onPersist throw fires the
// log-failed banner with the plan's LOG-FAILURE wording, latches
// loggingDisabled for the rest of the session, and subsequent generations
// skip the persist hook entirely (SEC-26 skip path) — the hook is attempted
// EXACTLY ONCE.
// =============================================================================
await leg('leg 17 — onPersist throw latches loggingDisabled; persist attempted exactly once (call count == 1)', async () => {
  const track = new Track(DEFAULT_CONTROL_POINTS, 70);
  const banners = [];
  let persistCalls = 0;
  const trainer = new Trainer(track, {
    lineage: { seed: 4242, rotationSeed: 4242, generatorVersion: 1 },
    onBanner(b) { banners.push(b); },
    onPersist() {
      persistCalls += 1;
      throw new Error('injected persist failure');
    },
  });
  trainer.start(2);
  await waitFor(() => trainer.state === 'done', 'persist-latch run to complete', 120000);

  assert(persistCalls === 1, 'persist hook attempted exactly once (gen 1), got ' + persistCalls);
  assert(trainer.loggingDisabled === true, 'persist throw latches loggingDisabled for the rest of the session');
  const logFailed = banners.filter((b) => b.type === 'log-failed');
  assert(logFailed.length === 1, 'exactly one log-failed banner from the persist path, got ' + logFailed.length);
  assert(logFailed[0].generation === 1 && logFailed[0].unique === false, 'banner carries generation 1 and unique=false');
  assert(String(logFailed[0].message).includes('LOGGING DISABLED THIS SESSION'), 'banner carries the plan LOG-FAILURE wording');
  const skipped = banners.filter((b) => b.type === 'persist-skipped');
  assert(skipped.length === 1, 'SEC-26 persist-skipped banner fired exactly once (gen 2), got ' + skipped.length);
});

// =============================================================================
// M2 leg 18 — backend-death matrix + effective-pair lineage guard (C2-F1/C2-F2).
// Pure node, deterministic, no browser, no network. The six backend-death
// scenarios drive an INJECTED fake backend (the opts.workerFactory test seam);
// the C2-F2 effective-pair verifications run constructor-first. The real
// MessageChannel fallback is used for the completion paths, so what is proven
// here is the REAL production flow with only the Worker replaced.
// =============================================================================
await leg('leg 18 — backend-death matrix + effective-pair lineage guard (C2-F1/C2-F2)', async () => {
  const track = new Track(DEFAULT_CONTROL_POINTS, 70);

  // A genuine generation-1 payload from a scratch lineage — used by the
  // post-ready scenarios (the trainer trusts backend payloads; it cannot be
  // driven to gen 1 without a fabricated or real payload).
  const scratch = new Evolution(4242);
  scratch.randomPopulation();
  scratch.evaluatePopulation(track);
  const gen1Payload = {
    generation: 1,
    results: scratch.lastResults,
    bestIndex: scratch.bestIndex,
    bestWeights: scratch.bestWeights(),
    rngState: scratch.rngState(),
  };
  const validResume = { bestWeights: gen1Payload.bestWeights, rngState: gen1Payload.rngState, generation: 1 };

  // FakeWorker: records every message (including __terminated__ markers) and
  // lets the test drive onmessage/onerror explicitly. Injected per-test via
  // opts.workerFactory.
  function makeFakeWorker() {
    const instances = [];
    class FakeWorker {
      constructor(url) {
        this.url = url;
        this.onmessage = null;
        this.onerror = null;
        this.terminated = false;
        this.messages = [];
        instances.push(this);
      }
      postMessage(msg) { this.messages.push(msg); }
      terminate() { this.terminated = true; this.messages.push({ type: '__terminated__' }); }
      emit(type, extra) { if (this.onmessage) this.onmessage({ data: Object.assign({ type }, extra || {}) }); }
      error(message) { if (this.onerror) this.onerror({ message }); }
    }
    return { FakeWorker, instances };
  }

  // ---------- C2-F2: effective-pair lineage guard (construction time) ----------
  // (1) fresh lineage {seed} + one-sided ROOT pair → throws
  assertValidationError(
    () => new Trainer(track, { lineage: { seed: 42, rotationSeed: 42 }, trackHash: 'x' }),
    'fresh lineage + one-sided root pair throws'
  );
  // (2) resume lineage + one-sided ROOT pair → throws
  assertValidationError(
    () => new Trainer(track, { lineage: { resume: validResume }, expectedTrackHash: 'e' }),
    'resume + one-sided root pair throws'
  );
  // (3) both-empty ⇒ ok
  {
    const t = new Trainer(track, { lineage: { seed: 42 } });
    assert(t.state === 'idle', 'both-empty effective pair constructs ok');
    assert(t.expectedTrackHash === null && t.trackHash === null, 'both-empty effective pair');
  }
  // (4) both-present ⇒ ok
  {
    const t = new Trainer(track, { lineage: { seed: 42, rotationSeed: 42 }, expectedTrackHash: 'e', trackHash: 'a' });
    assert(t.expectedTrackHash === 'e' && t.trackHash === 'a', 'both-present effective pair (root)');
  }
  // (5) effective mixed after precedence ⇒ throws — lineage one-sided pair is
  // atomic (never completed from root), root's opposite side ignored → XOR
  assertValidationError(
    () => new Trainer(track, { lineage: { resume: validResume, trackHash: 'x' }, expectedTrackHash: 'e' }),
    'effective mixed after precedence throws'
  );
  // (6) lineage-level one-sided with an empty root pair → throws
  assertValidationError(
    () => new Trainer(track, { lineage: { resume: validResume, trackHash: 'x' } }),
    'lineage-level one-sided pair throws'
  );
  // (7) lineage-level pair complete ⇒ lineage pair wins wholesale, root ignored
  {
    const t = new Trainer(track, {
      lineage: { resume: validResume, expectedTrackHash: 'll', trackHash: 'lr' },
      expectedTrackHash: 're', trackHash: 'ra',
    });
    assert(t.expectedTrackHash === 'll' && t.trackHash === 'lr', 'lineage-level pair is atomic and wins wholesale');
  }

  // ---------- (i) pre-ready onerror ⇒ terminate FIRST, exactly one respawn,
  //              handshake replayed verbatim, lineage completes via fallback
  {
    const { FakeWorker, instances } = makeFakeWorker();
    const events = [];
    const outbound = [];
    const logged = [];
    const banners = [];
    const tr = new Trainer(track, {
      lineage: { seed: 4242, rotationSeed: 4242, generatorVersion: 1 },
      workerFactory: (url) => new FakeWorker(url),
      logger: { logGeneration(p) { logged.push(p.generation); } },
      onBanner(b) { banners.push(b); },
    });
    const origCreate = tr._createFallbackTransport.bind(tr);
    tr._createFallbackTransport = function () { events.push('fallback-create'); return origCreate(); };
    const origPost = tr._postToBackend.bind(tr);
    tr._postToBackend = function (msg) { outbound.push(msg); return origPost(msg); };
    const fake = instances[0];
    const origTerminate = fake.terminate.bind(fake);
    fake.terminate = function () { events.push('terminate'); return origTerminate(); };

    tr.start(2);
    fake.error('boom: simulated pre-ready worker failure'); // no 'ready' was ever emitted
    await waitFor(() => tr.state === 'done', 'replayed lineage completes through the fallback', 120000);

    const ti = events.indexOf('terminate');
    const fi = events.indexOf('fallback-create');
    assert(ti !== -1 && fi !== -1 && ti < fi, 'terminate() was called BEFORE the fallback respawn (order: ' + events.join(' → ') + ')');
    assert(events.filter((e) => e === 'fallback-create').length === 1, 'exactly one fallback respawn, got ' + events.filter((e) => e === 'fallback-create').length);
    assert(logged.join() === '1,2', 'two generations logged through the fallback, got ' + logged.join());
    assert(banners.filter((b) => b.type === 'worker-failed-fallback').length === 1, 'worker-failed-fallback banner posted exactly once');
    assert(!banners.some((b) => b.type === 'worker-error'), 'no worker-error banner after a successful respawn');
    assert(outbound.length >= 4, 'init, start, replay init, replay start were all posted, got ' + outbound.length);
    assert(outbound[0].type === 'init' && outbound[2].type === 'init', 'first and replayed messages are init');
    assert(JSON.stringify(outbound[0]) === JSON.stringify(outbound[2]), 'replayed init handshake is byte-identical to the first attempt');
    assert(outbound[1].type === 'start' && outbound[3].type === 'start', 'first and replayed messages are start');
    assert(JSON.stringify(outbound[1]) === JSON.stringify(outbound[3]), 'replayed start is byte-identical to the first attempt');
    assert(tr.state === 'done', "lineage completed — state 'done', got " + tr.state);
  }

  // ---------- (ii) post-ready onerror (after gen 1 was logged) ⇒ NO respawn,
  //               halted, extended LINEAGE banner, start() throws
  {
    const { FakeWorker, instances } = makeFakeWorker();
    const logged = [];
    const banners = [];
    const tr = new Trainer(track, {
      lineage: { seed: 4242, rotationSeed: 4242, generatorVersion: 1 },
      workerFactory: (url) => new FakeWorker(url),
      logger: { logGeneration(p) { logged.push(p.generation); } },
      onBanner(b) { banners.push(b); },
    });
    const origCreate = tr._createFallbackTransport.bind(tr);
    let respawns = 0;
    tr._createFallbackTransport = function () { respawns += 1; return origCreate(); };

    tr.start(2);
    const fake = instances[0];
    fake.emit('ready', { generation: 1 });
    fake.emit('generation', { payload: gen1Payload }); // gen 1 gets logged through the fake
    assert(logged.join() === '1', 'gen 1 logged before the failure, got ' + logged.join());
    fake.error('boom: worker died after gen 1 was logged');

    assert(respawns === 0, 'NO fallback respawn after progress was logged, got ' + respawns);
    assert(instances.length === 1, 'no new backend was constructed after the halt');
    assert(tr.state === 'halted', "state === 'halted' after post-ready onerror, got " + tr.state);
    const banner = banners.find((b) => b.type === 'worker-error');
    assert(banner && /LINEAGE INTEGRITY ERROR/.test(banner.message), 'LINEAGE-class banner posted');
    assert(banner && /reload and resume recomputes/.test(banner.message), 'extended banner carries the resume sentence');
    let threw = null;
    try { tr.start(3); } catch (e) { threw = e; }
    assert(threw && /halted/.test(threw.message), 'start() throws after halt');
  }

  // ---------- (iii) pre-ready onerror with MessageChannel UNAVAILABLE ⇒
  //               halted, LINEAGE-class banner, no respawn attempt succeeds
  {
    const { FakeWorker, instances } = makeFakeWorker();
    const banners = [];
    const tr = new Trainer(track, {
      lineage: { seed: 4242, rotationSeed: 4242, generatorVersion: 1 },
      workerFactory: (url) => new FakeWorker(url),
      onBanner(b) { banners.push(b); },
    });
    const origCreate = tr._createFallbackTransport.bind(tr);
    let respawnAttempts = 0;
    tr._createFallbackTransport = function () { respawnAttempts += 1; return origCreate(); };
    const savedMC = globalThis.MessageChannel;
    globalThis.MessageChannel = undefined; // monkey-patched OUT for this scenario
    try {
      tr.start(1);
      instances[0].error('boom: pre-ready with MessageChannel unavailable');
    } finally {
      globalThis.MessageChannel = savedMC;
    }
    assert(tr.state === 'halted', "state === 'halted' when the fallback is unavailable, got " + tr.state);
    assert(respawnAttempts === 1, 'exactly one respawn attempt was made and it failed, got ' + respawnAttempts);
    // With globalThis.MessageChannel === undefined the fallback constructor's
    // typeof-guard throws before any construction can happen — zero channels
    // can be constructed, by construction of the seam.
    const banner = banners.find((b) => b.type === 'worker-error');
    assert(banner && /fallback failed too/.test(banner.message), 'banner reports the fallback failure');
    assert(banner && /MessageChannel/.test(banner.message), 'banner carries the MessageChannel failure detail');
    assert(banner && /LINEAGE INTEGRITY ERROR/.test(banner.message), 'banner is LINEAGE-class');
    assert(!banners.some((b) => b.type === 'worker-failed-fallback'), 'no worker-failed-fallback banner on a FAILED respawn');
    let threw = null;
    try { tr.start(2); } catch (e) { threw = e; }
    assert(threw && /halted/.test(threw.message), 'start() throws after halt');
  }

  // ---------- (iv) sync Worker constructor throw ⇒ transparent fallback,
  //               NO banner, run completes
  {
    const banners = [];
    const logged = [];
    const tr = new Trainer(track, {
      lineage: { seed: 42, rotationSeed: 42, generatorVersion: 1 },
      workerFactory: () => { throw new Error('SecurityError: The operation is insecure.'); },
      logger: { logGeneration(p) { logged.push(p.generation); } },
      onBanner(b) { banners.push(b); },
    });
    tr.start(1);
    await waitFor(() => tr.state === 'done', 'transparent fallback completes the run', 120000);
    assert(logged.join() === '1', 'generation logged through the transparent fallback, got ' + logged.join());
    assert(!banners.some((b) => b.type === 'worker-error' || b.type === 'worker-failed-fallback'), 'no failure banner for a supported fallback path');
  }

  // ---------- (v) UNIQUE-throwing logger + {type:'stopped'} backend ack ⇒
  //               state stays 'halted', start() throws
  {
    const { FakeWorker, instances } = makeFakeWorker();
    const banners = [];
    const tr = new Trainer(track, {
      lineage: { seed: 4242, rotationSeed: 4242, generatorVersion: 1 },
      workerFactory: (url) => new FakeWorker(url),
      logger: { logGeneration() { const e = new Error('UNIQUE constraint failed: generations.generation'); throw e; } },
      onBanner(b) { banners.push(b); },
    });
    tr.start(2);
    const fake = instances[0];
    fake.emit('ready', { generation: 1 });
    fake.emit('generation', { payload: gen1Payload }); // UNIQUE throw → halt + stop posted to the backend
    assert(tr.state === 'halted', "UNIQUE logger throw halts the run, got " + tr.state);
    assert(fake.messages.some((m) => m.type === 'stop'), 'the halt posted stop to the backend');
    fake.emit('stopped', { generation: 1 }); // the backend acks the stop — must NOT downgrade halted
    assert(tr.state === 'halted', "a later 'stopped' ack must not downgrade 'halted', got " + tr.state);
    let threw = null;
    try { tr.start(3); } catch (e) { threw = e; }
    assert(threw && /halted/.test(threw.message), 'start() throws after halt');
    assert(banners.some((b) => b.type === 'log-failed' && b.unique === true), 'UNIQUE halt banner was posted');
  }

  // ---------- (vi) post-READY onerror with generationsLogged === 0 ⇒ ONE
  //               fallback respawn + lineage completes (M3-fix F-4). The
  //               failure surfaced AFTER 'ready' but BEFORE the first commit —
  //               no partial commit exists, so the lineage is re-derived
  //               byte-identically; a halt here would wedge a run that never
  //               persisted anything. (Was: halt — row 3 of the failure matrix.)
  {
    const { FakeWorker, instances } = makeFakeWorker();
    const logged = [];
    const banners = [];
    const tr = new Trainer(track, {
      lineage: { seed: 4242, rotationSeed: 4242, generatorVersion: 1 },
      workerFactory: (url) => new FakeWorker(url),
      logger: { logGeneration(p) { logged.push(p.generation); } },
      onBanner(b) { banners.push(b); },
    });
    const origCreate = tr._createFallbackTransport.bind(tr);
    let respawns = 0;
    tr._createFallbackTransport = function () { respawns += 1; return origCreate(); };

    tr.start(2);
    const fake = instances[0];
    fake.emit('ready', { generation: 1 }); // POST-ready — but nothing was logged yet
    fake.error('boom: worker died post-ready before the first commit');
    await waitFor(() => tr.state === 'done', 'post-ready pre-first-commit respawn completes the lineage', 120000);

    assert(respawns === 1, 'exactly one fallback respawn after post-ready pre-commit failure, got ' + respawns);
    assert(logged.join() === '1,2', 'the lineage re-derived and logged gens 1,2, got ' + logged.join());
    assert(banners.filter((b) => b.type === 'worker-failed-fallback').length === 1, 'worker-failed-fallback banner posted exactly once');
    assert(!banners.some((b) => b.type === 'worker-error'), 'no worker-error halt banner — the respawn succeeded');
    assert(tr.state === 'done', "state 'done' after respawn, got " + tr.state);
  }
});

// =============================================================================
// M3 legs — js/db.js (sql.js persistence), the vendored WASM vendor hash gate,
// the ADOPTION BATTERY (every bytes-adoption path), the persist step-2 failure
// path, resume byte-identity through a real export/adopt cycle, and the
// measured per-persist cost. These legs are HARD failures: any regression
// fails the suite.
//
// js/db.js is a plain classic script with guarded `module.exports`; it is
// required LAZILY inside each leg so a missing/broken module fails the M3
// legs loudly instead of crashing the whole suite at import time.
// =============================================================================
const VENDOR_DIR = new URL('../vendor/', import.meta.url).pathname; // trailing slash; path.join handles it
// Pinned at vendoring time (see plan TECH STACK + README threat model).
// A vendor file whose SHA-256 does not match these constants is a HARD FAIL —
// any vendor hash change requires a new anti-team-security cycle (SEC-19).
const VENDOR_SQL_WASM_JS_SHA256 = '9b96118841c6edfdf53c6f388c1419d1f473cf469df6a6fc2864b386d60bdb5e';
const VENDOR_SQL_WASM_WASM_SHA256 = '43bbcdd44287e5a2db3b6b5af34b1a3485a7a441285d247976c1ebe3c61f8dbf';
const VENDOR_MAX_BYTES = 64 * 1024 * 1024;

const defaultTrack = new Track(DEFAULT_CONTROL_POINTS, 70);
const DEFAULT_TRACK_JSON = defaultTrack.toJSON();
const DEFAULT_TRACK_HASH = defaultTrack.hash();

const requireDb = () => require('../js/db.js');
// js/rotation.js (FIT-V3 M6a — the procedural track generator + rotation
// ladder) is required here for the same reason: a missing/broken module
// fails only the M6a legs, never the whole suite.
const requireRotation = () => require('../js/rotation.js');

async function freshDb() {
  const { Db } = requireDb();
  await Db.loadNode(VENDOR_DIR);
  const db = new Db();
  await db.init();
  return db;
}

function countRows(db, table) {
  const rows = db.db.exec('SELECT COUNT(*) FROM ' + table);
  return rows && rows[0] ? rows[0].values[0][0] : NaN;
}

function makeValidResults() {
  const results = [];
  for (let i = 0; i < POPULATION; i++) {
    results.push({
      distance: 1000 - i * 10,
      topSpeed: 400 + i,
      timeAlive: 5.25,
      crashed: i % 4 === 0 ? 1 : 0,
      stuck: 0,
    });
  }
  return results;
}

function makeValidWeightsString() {
  return JSON.stringify(new NeuralNetwork(mulberry32(777)).getWeights());
}

function makeValidGen(overrides) {
  const gen = {
    generation: 1,
    driversEvaluated: POPULATION,
    bestFitness: 1000,
    avgFitness: 900,
    worstFitness: 800,
    bestDriverIndex: 0,
    mutationRate: MUTATION_RATE,
    mutationScale: MUTATION_SCALE,
    trackJson: DEFAULT_TRACK_JSON,
    trackHash: DEFAULT_TRACK_HASH,
    simSeconds: 105,
    rngState: 123456789,
    results: makeValidResults(),
    bestWeightsJson: makeValidWeightsString(),
  };
  return Object.assign(gen, overrides);
}

// Idempotent: same values -> no-op; contradictory values -> throw (lineage).
// M6b: ensureLineage gains the REQUIRED trackJson third argument (the v3
// curriculum is materialized from the pinned track's bytes at pin time).
function ensureLineageFor(db) {
  db.ensureLineage(DEFAULT_TRACK_HASH, 123456789, DEFAULT_TRACK_JSON);
}

function logGen(db, n, overrides) {
  const gen = makeValidGen(Object.assign({ generation: n, rngState: 1000 + n }, overrides));
  db.logGeneration(gen);
  return gen;
}

function assertBatteryReject(Db, bytes, pattern, label) {
  let threw = null;
  try {
    Db.validateAdoptionBytes(bytes);
  } catch (err) {
    threw = err;
  }
  assert(threw !== null, (label || 'battery') + ' — expected a rejection, got a pass');
  assert(threw.name === 'ValidationError', (label || 'battery') + ' — expected ValidationError, got ' + threw.name + ': ' + threw.message);
  assert(pattern.test(String(threw.message)), (label || 'battery') + ' — error ' + JSON.stringify(threw.message) + ' does not match ' + pattern);
}

// FIT-V6: any pre-v4 byte (uv∈{1,2,3}) is refused at battery step 3b with a
// typed WallModelChangedError carrying the refused bytes verbatim. This helper
// pins that contract for the migrated legacy-fixture legs (no silent skip, and
// strictly checks the typed name + message + byte-preservation).
function assertWallModelRefusal(Db, bytes, label) {
  const before = new Uint8Array(bytes);
  let threw = null;
  try { Db.validateAdoptionBytes(bytes); } catch (err) { threw = err; }
  assert(threw !== null, (label || 'battery') + ' — expected a wall-model refusal, got a pass');
  assert(threw.name === 'WallModelChangedError', (label || 'battery') + ' — expected WallModelChangedError, got ' + threw.name + ': ' + threw.message);
  assert(/wall model/i.test(threw.message), (label || 'battery') + ' — message must name the wall model: ' + threw.message);
  assert(/untouched/i.test(threw.message), (label || 'battery') + ' — message must state the original was untouched: ' + threw.message);
  assert(threw.refusedBytes instanceof Uint8Array && sameBytes(threw.refusedBytes, before), (label || 'battery') + ' — typed error must carry the refused bytes verbatim');
  assert(sameBytes(bytes, before), (label || 'battery') + ' — refusal must not mutate the bytes');
  return threw;
}
function sameBytes(a, b) {
  if (!(a instanceof Uint8Array) || !(b instanceof Uint8Array)) return false;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// =============================================================================
// M3 leg 19 — vendor SHA-256 hash leg (HARD FAIL on mismatch). Also pins the
// embedded base64 wasm (vendor/sql-wasm-binary.js — M4-fix FIX-FILE-URL for
// file://): the base64 is extracted with the EXACT pinned regex, decoded,
// and its SHA-256 must equal the pinned sql-wasm.wasm hash. And index.html
// must load sql-wasm-binary.js BEFORE sql-wasm.js (script order is
// LOAD-BEARING — file:// works with zero fetch iff the base64 global exists
// by the time db.js's createBrowser() runs).
// =============================================================================
await leg('leg 19 — vendor pins: sql-wasm.js/sql-wasm.wasm SHA-256 + embedded sql-wasm-binary.js round-trips + index.html script order (hard-fail)', () => {
  const { createHash } = require('node:crypto');
  const { readFileSync } = require('node:fs');
  const path = require('node:path');
  const jsHash = createHash('sha256').update(readFileSync(path.join(VENDOR_DIR, 'sql-wasm.js'))).digest('hex');
  const wasmHash = createHash('sha256').update(readFileSync(path.join(VENDOR_DIR, 'sql-wasm.wasm'))).digest('hex');
  assert(jsHash === VENDOR_SQL_WASM_JS_SHA256, 'vendor/sql-wasm.js SHA-256 mismatch: got ' + jsHash + ' (pinned ' + VENDOR_SQL_WASM_JS_SHA256 + ') — any vendor hash change = new anti-team-security cycle (SEC-19)');
  assert(wasmHash === VENDOR_SQL_WASM_WASM_SHA256, 'vendor/sql-wasm.wasm SHA-256 mismatch: got ' + wasmHash + ' (pinned ' + VENDOR_SQL_WASM_WASM_SHA256 + ') — any vendor hash change = new anti-team-security cycle (SEC-19)');
  // M4-fix FIX-FILE-URL — the embedded base64 wasm round-trips to the pinned hash.
  const binarySrc = readFileSync(path.join(VENDOR_DIR, 'sql-wasm-binary.js'), 'utf8');
  // SEC L1 — the extraction regex is END-ANCHORED (a trailing same-line payload
  // fails here, before any decode) and the WHOLE file must match the whitelist:
  // one leading comment + exactly one globalThis.SQL_WASM_BINARY_BASE64
  // assignment + nothing else. A trailing same-line or later-line payload
  // breaks the whitelist and fails the leg loudly.
  assert(/^\/\/[^\n]*\nglobalThis\.SQL_WASM_BINARY_BASE64="[A-Za-z0-9+/=]+";\n?$/.test(binarySrc), 'vendor/sql-wasm-binary.js must contain ONLY a comment followed by exactly one globalThis.SQL_WASM_BINARY_BASE64 assignment — nothing else (single-line comment + single global whitelist), got prefix: ' + binarySrc.slice(0, 120));
  const b64Match = binarySrc.match(/globalThis\.SQL_WASM_BINARY_BASE64="([A-Za-z0-9+/=]+)";\s*$/);
  assert(b64Match !== null, 'vendor/sql-wasm-binary.js must define globalThis.SQL_WASM_BINARY_BASE64 with the exact pinned shape (one line, no whitespace), got prefix: ' + binarySrc.slice(0, 120));
  const embeddedHash = createHash('sha256').update(Buffer.from(b64Match[1], 'base64')).digest('hex');
  assert(embeddedHash === VENDOR_SQL_WASM_WASM_SHA256, 'embedded base64 wasm SHA-256 mismatch: got ' + embeddedHash + ' (pinned ' + VENDOR_SQL_WASM_WASM_SHA256 + ')');
  const definitions = binarySrc.split('globalThis.SQL_WASM_BINARY_BASE64=').length - 1;
  assert(definitions === 1, 'SQL_WASM_BINARY_BASE64 defined exactly once, got ' + definitions);
  // Script order is LOAD-BEARING — binary BEFORE sql-wasm.js.
  const indexHtml = readFileSync(path.join(REPO_ROOT, 'index.html'), 'utf8');
  const BIN_TAG = '<script src="vendor/sql-wasm-binary.js"></script>';
  const JS_TAG = '<script src="vendor/sql-wasm.js"></script>';
  const binIdx = indexHtml.indexOf(BIN_TAG);
  const jsIdx = indexHtml.indexOf(JS_TAG);
  assert(binIdx !== -1, 'index.html loads vendor/sql-wasm-binary.js (expected tag: ' + BIN_TAG + ')');
  assert(jsIdx !== -1, 'index.html loads vendor/sql-wasm.js (expected tag: ' + JS_TAG + ')');
  assert(binIdx < jsIdx, 'index.html loads sql-wasm-binary.js BEFORE sql-wasm.js (binary at ' + binIdx + ', js at ' + jsIdx + ')');
});

// =============================================================================
// M3 leg 20 — fresh in-memory schema: PRAGMA user_version = 4, foreign_keys = 1
// (asserted at init, re-read here), sqlite_master EXACTLY the five tables (+
// sqlite_autoindex entries), zero triggers/views, meta seed rows present.
// FIT-V6: fresh DBs are v4 (schema_version='4', wall_model='v6', app_version
// '1.1.0', generations carries REFERENCES tracks(track_hash); the v3 DDL twins
// are reused; the dedicated tracks table exists).
// =============================================================================
await leg('leg 20 — fresh schema: user_version=4, foreign_keys=1, EXACTLY the five tables (+autoindex)', async () => {
  const db = await freshDb();
  const uv = db.db.exec('PRAGMA user_version');
  assert(uv && uv[0] && uv[0].values[0][0] === 4, 'PRAGMA user_version must be 4, got ' + JSON.stringify(uv));
  const fk = db.db.exec('PRAGMA foreign_keys');
  assert(fk && fk[0] && fk[0].values[0][0] === 1, 'PRAGMA foreign_keys must be 1, got ' + JSON.stringify(fk));
  const objects = db.db.exec('SELECT type, name FROM sqlite_master ORDER BY type, name');
  const rows = objects && objects[0] ? objects[0].values : [];
  const EXPECTED_TABLES = ['best_weights', 'driver_results', 'generations', 'meta', 'tracks'];
  const tableNames = rows.filter((r) => r[0] === 'table').map((r) => r[1]).sort();
  assert(JSON.stringify(tableNames) === JSON.stringify(EXPECTED_TABLES), 'EXACTLY the five tables expected, got ' + JSON.stringify(tableNames));
  for (const [type, name] of rows) {
    if (type === 'table') continue;
    assert(type === 'index' && String(name).startsWith('sqlite_autoindex_'), 'unexpected sqlite_master object type=' + type + ' name=' + name + ' (only sqlite_autoindex_* entries permitted)');
  }
  assert(!rows.some((r) => r[0] === 'trigger' || r[0] === 'view'), 'zero triggers, zero views');
  const metaRows = db.db.exec('SELECT key, value FROM meta');
  const metaObj = {};
  for (const [k, v] of (metaRows && metaRows[0] ? metaRows[0].values : [])) metaObj[k] = v;
  assert(metaObj.schema_version === '4', 'meta.schema_version must be 4, got ' + metaObj.schema_version);
  assert(metaObj.wall_model === 'v6', 'meta.wall_model must be v6, got ' + metaObj.wall_model);
  assert(metaObj.architecture === '7-16-16-2', 'meta.architecture must be 7-16-16-2, got ' + metaObj.architecture);
  assert(metaObj.app_version === '1.1.0', 'meta.app_version must be 1.1.0, got ' + metaObj.app_version);
  assert(typeof metaObj.created_at === 'string' && metaObj.created_at.length > 0, 'meta.created_at must be present');
  db.db.close();
});

// =============================================================================
// M3 leg 21 — FK cascade: deleting generation 1 removes its driver_results and
// best_weights rows; generation 2 is untouched.
// =============================================================================
await leg('leg 21 — FK cascade: delete generation 1 -> driver_results/best_weights rows gone, generation 2 intact', async () => {
  const db = await freshDb();
  ensureLineageFor(db);
  logGen(db, 1);
  logGen(db, 2);
  db.db.exec('DELETE FROM generations WHERE generation = 1');
  assert(countRows(db, 'generations') === 1, 'generations count after delete must be 1');
  assert(countRows(db, 'driver_results') === POPULATION, 'driver_results must hold only generation 2 (20 rows), got ' + countRows(db, 'driver_results'));
  assert(countRows(db, 'best_weights') === 1, 'best_weights must hold only generation 2, got ' + countRows(db, 'best_weights'));
  const gen1 = db.db.exec('SELECT COUNT(*) FROM driver_results WHERE generation = 1')[0].values[0][0];
  const bw1 = db.db.exec('SELECT COUNT(*) FROM best_weights WHERE generation = 1')[0].values[0][0];
  assert(gen1 === 0 && bw1 === 0, 'generation 1 children must be cascade-deleted (driver_results=' + gen1 + ', best_weights=' + bw1 + ')');
  assert(db.listGenerations().join() === '2', 'listGenerations must return [2], got ' + db.listGenerations());
  db.db.close();
});

// =============================================================================
// M3 leg 22 — logGeneration rejects a NaN fitness BEFORE any write (arm A,
// no lineage ensured) and rolls back cleanly after ensureLineage (arm B):
// generations UNCHANGED, no driver_results rows, no best_weights row,
// meta.rng_state absent (arm A) / unchanged (arm B). Then a valid gen 1 log.
// Arm C (M3-fix F-11): driversEvaluated !== results.length ⇒ pre-BEGIN throw,
// prefix intact. Arm D (M3-fix F-13): injected NON-UNIQUE mid-transaction
// failure ⇒ ROLLBACK + raw rethrow + zero rows written.
// =============================================================================
await leg('leg 22 — logGeneration transaction + rollback on injected failure (NaN distance, driversEvaluated lie, DiskFullError mid-txn)', async () => {
  // Arm A — no ensureLineage: pre-condition failure, nothing committed.
  const dbA = await freshDb();
  const threwA = (() => { try { dbA.logGeneration(makeValidGen({})); return null; } catch (e) { return e; } })();
  assert(threwA !== null, 'arm A — logGeneration without meta.rng_state must throw');
  assert(/rng_state/i.test(String(threwA.message)), 'arm A — error must mention rng_state, got: ' + threwA.message);
  assert(countRows(dbA, 'generations') === 0 && countRows(dbA, 'driver_results') === 0 && countRows(dbA, 'best_weights') === 0, 'arm A — zero rows written');
  const metaA = dbA.db.exec("SELECT value FROM meta WHERE key = 'rng_state'");
  assert(!metaA || metaA.length === 0 || metaA[0].values.length === 0, 'arm A — meta.rng_state must be ABSENT');
  dbA.db.close();

  // Arm B — lineage ensured; injected NaN distance mid-payload.
  const dbB = await freshDb();
  ensureLineageFor(dbB);
  const results = makeValidResults();
  results[3] = { distance: NaN, topSpeed: 1, timeAlive: 1, crashed: 0, stuck: 0 };
  let threwB = null;
  try { dbB.logGeneration(makeValidGen({ results })); } catch (err) { threwB = err; }
  assert(threwB !== null, 'arm B — NaN distance must throw');
  assert(threwB.name === 'ValidationError', 'arm B — expected ValidationError, got ' + threwB.name + ': ' + threwB.message);
  assert(countRows(dbB, 'generations') === 0 && countRows(dbB, 'driver_results') === 0 && countRows(dbB, 'best_weights') === 0, 'arm B — the transaction wrote nothing (counts: generations=' + countRows(dbB, 'generations') + ' driver_results=' + countRows(dbB, 'driver_results') + ' best_weights=' + countRows(dbB, 'best_weights') + ')');
  const metaB = dbB.db.exec("SELECT value FROM meta WHERE key = 'rng_state'")[0].values[0][0];
  assert(metaB === '123456789', 'arm B — meta.rng_state UNCHANGED (got ' + metaB + ')');
  // A valid full log of generation 1 succeeds afterwards.
  logGen(dbB, 1);
  assert(countRows(dbB, 'generations') === 1 && countRows(dbB, 'driver_results') === POPULATION && countRows(dbB, 'best_weights') === 1, 'valid gen 1 log commits fully');
  dbB.db.close();

  // Arm C — driversEvaluated !== results.length ⇒ pre-BEGIN ValidationError,
  // prefix intact (M3-fix F-11).
  const dbC = await freshDb();
  ensureLineageFor(dbC);
  logGen(dbC, 1);
  const beforeC = dbC.db.exec("SELECT value FROM meta WHERE key = 'rng_state'")[0].values[0][0];
  let threwC = null;
  try { dbC.logGeneration(makeValidGen({ generation: 2, driversEvaluated: 7 })); } catch (err) { threwC = err; }
  assert(threwC !== null && threwC.name === 'ValidationError' && /driversEvaluated/.test(String(threwC.message)), 'arm C — driversEvaluated mismatch throws ValidationError, got ' + (threwC && threwC.name + ': ' + threwC.message));
  assert(dbC.listGenerations().join() === '1', 'arm C — committed prefix intact (gens ' + dbC.listGenerations() + ')');
  assert(countRows(dbC, 'driver_results') === POPULATION && countRows(dbC, 'best_weights') === 1, 'arm C — zero rows written');
  const afterC = dbC.db.exec("SELECT value FROM meta WHERE key = 'rng_state'")[0].values[0][0];
  assert(afterC === beforeC, 'arm C — meta.rng_state unchanged (pre-BEGIN throw)');
  dbC.db.close();

  // Arm D — injected NON-UNIQUE failure MID-TRANSACTION ⇒ ROLLBACK + the RAW
  // error rethrown (never a LineageIntegrityError) + zero rows written +
  // generations empty (M3-fix F-13). Seam style mirrors leg 26b's documented
  // instance-method override: persist()/logGeneration()'s real code paths are
  // under test; only the INSERT call is sabotaged.
  const dbD = await freshDb();
  ensureLineageFor(dbD);
  const realRun = dbD.db.run.bind(dbD.db);
  let injected = false;
  dbD.db.run = function (sql, params) {
    if (/INSERT INTO driver_results/.test(sql) && !injected) {
      injected = true;
      const e = new Error('disk I/O error (injected DiskFullError)');
      e.name = 'DiskFullError';
      throw e;
    }
    return realRun(sql, params);
  };
  let threwD = null;
  try { dbD.logGeneration(makeValidGen({ generation: 1 })); } catch (err) { threwD = err; }
  assert(threwD !== null && threwD.name === 'DiskFullError', 'arm D — the RAW DiskFullError is rethrown, got ' + (threwD && threwD.name));
  assert(threwD && threwD.name !== 'LineageIntegrityError', 'arm D — a non-UNIQUE fault is NEVER wrapped in LineageIntegrityError');
  assert(injected === true, 'arm D — the mid-transaction INSERT was actually reached');
  assert(countRows(dbD, 'generations') === 0 && countRows(dbD, 'driver_results') === 0 && countRows(dbD, 'best_weights') === 0, 'arm D — ROLLBACK: gens=' + countRows(dbD, 'generations') + ' driver_results=' + countRows(dbD, 'driver_results') + ' best_weights=' + countRows(dbD, 'best_weights'));
  const metaD = dbD.db.exec("SELECT value FROM meta WHERE key = 'rng_state'")[0].values[0][0];
  assert(metaD === '123456789', 'arm D — meta.rng_state rolled back to the ensured value (got ' + metaD + ')');
  dbD.db.close();
});

// =============================================================================
// M3 leg 23 — UNIQUE conflict on generations.generation: second log of gen 1
// throws LineageIntegrityError (message contains UNIQUE); the committed
// lineage is untouched — gen 1 keeps the FIRST payload's weights, count stays 1.
// Arm (c) (M3-fix F-2): a PRIMARY-KEY conflict raised by the FIRST
// driver_results INSERT (pre-seeded crafted bytes) wraps identically — probe
// arms pin the REAL vendored sqlite messages against the shared classifier.
// =============================================================================
await leg('leg 23 — UNIQUE conflict => LineageIntegrityError, ROLLBACK, committed lineage intact', async () => {
  const db = await freshDb();
  ensureLineageFor(db);
  const firstWeights = new NeuralNetwork(mulberry32(555)).getWeights();
  db.logGeneration(makeValidGen({ generation: 1, rngState: 111111, bestWeightsJson: JSON.stringify(firstWeights) }));
  const weightsBefore = db.getBestWeights(1).getWeights();
  const metaBefore = db.db.exec("SELECT value FROM meta WHERE key = 'rng_state'")[0].values[0][0];
  assert(metaBefore === '111111', 'rng_state committed as 111111, got ' + metaBefore);

  const hostile = makeValidGen({ generation: 1, bestFitness: 999999, rngState: 222222 });
  hostile.results = makeValidResults().map((r) => Object.assign({}, r, { distance: 999999 }));
  hostile.bestWeightsJson = JSON.stringify(new NeuralNetwork(mulberry32(999)).getWeights());
  let threw = null;
  try { db.logGeneration(hostile); } catch (err) { threw = err; }
  assert(threw !== null, 'second log of generation 1 must throw');
  assert(threw.name === 'LineageIntegrityError', 'expected LineageIntegrityError, got ' + threw.name + ': ' + threw.message);
  assert(/UNIQUE/.test(String(threw.message)), 'message must contain UNIQUE, got: ' + threw.message);
  assert(countRows(db, 'generations') === 1, 'generations count still 1');
  assert(countRows(db, 'driver_results') === POPULATION, 'driver_results still 20');
  assert(countRows(db, 'best_weights') === 1, 'best_weights still 1');
  const metaAfter = db.db.exec("SELECT value FROM meta WHERE key = 'rng_state'")[0].values[0][0];
  assert(metaAfter === '111111', 'meta.rng_state NOT bumped to 222222 (ROLLBACK), got ' + metaAfter);
  const weightsAfter = db.getBestWeights(1).getWeights();
  assert(JSON.stringify(weightsAfter) === JSON.stringify(weightsBefore), 'committed gen 1 weights byte-identical to the FIRST payload');
  assert(JSON.stringify(weightsAfter.w1) === JSON.stringify(firstWeights.w1), 'gen 1 w1 still the first payload');
  db.db.close();

  // Arm (c) — a UNIQUE/PRIMARY-KEY conflict raised by the FIRST driver_results
  // INSERT (i.e. NOT at the generations row) also wraps to LineageIntegrityError
  // and ROLLBACKs cleanly (M3-fix F-2). The attack bytes are crafted raw:
  // driver_results rows for generation 1 are pre-seeded with NO generations
  // row (FK checks are disabled on the crafting connection — foreign_keys is
  // per-connection and not persisted, so the EXPORTED file carries no pragma).
  // Vendored sql-wasm OBSERVED (see js/db.js isLineageIntegrityFault comment):
  //   "UNIQUE constraint failed: generations.generation"                     (integer PK)
  //   "UNIQUE constraint failed: driver_results.generation, driver_results.driver_index"  (composite PK)
  //   "UNIQUE constraint failed: t.b"                                        (explicit UNIQUE)
  // Every observed error carries code === undefined (plain Error); the regex
  // half of the classifier is the load-bearing half with this vendor.
  {
    const { Db, isLineageIntegrityFault } = requireDb();
    const SQL = globalThis.SQL;
    assert(SQL && typeof SQL.Database === 'function', 'SQL namespace available (loadNode ran)');
    const cleanDb = await freshDb();
    const cleanBytes = cleanDb.db.export();
    cleanDb.db.close();

    // Probe the vendored build for the exact message strings — the raw
    // message MUST match the classifier regex (pinned assertions, so a vendor
    // swap that changes the wording fails HERE, not silently).
    const probe = new SQL.Database(cleanBytes);
    probe.run("INSERT INTO generations (generation, drivers_evaluated, best_fitness, avg_fitness, worst_fitness, best_driver_index, mutation_rate, mutation_scale, track_json, track_hash, sim_seconds, rng_state) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
      [1, POPULATION, 1, 1, 1, 0, 0.25, 0.3, DEFAULT_TRACK_JSON, DEFAULT_TRACK_HASH, 1, 1]);
    let rawUniqueMsg = '';
    try {
      probe.run("INSERT INTO generations (generation, drivers_evaluated, best_fitness, avg_fitness, worst_fitness, best_driver_index, mutation_rate, mutation_scale, track_json, track_hash, sim_seconds, rng_state) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [1, POPULATION, 1, 1, 1, 0, 0.25, 0.3, DEFAULT_TRACK_JSON, DEFAULT_TRACK_HASH, 1, 1]);
    } catch (err) { rawUniqueMsg = String((err && err.message) || err); }
    probe.run('PRAGMA foreign_keys = OFF');
    probe.run('INSERT INTO driver_results (generation, driver_index, distance, top_speed, time_alive, crashed, stuck) VALUES (?,?,?,?,?,?,?)', [1, 0, 1, 1, 1, 0, 0]);
    let rawPkMsg = '';
    try {
      probe.run('INSERT INTO driver_results (generation, driver_index, distance, top_speed, time_alive, crashed, stuck) VALUES (?,?,?,?,?,?,?)', [1, 0, 1, 1, 1, 0, 0]);
    } catch (err) { rawPkMsg = String((err && err.message) || err); }
    probe.close();
    assert(rawUniqueMsg.length > 0 && rawPkMsg.length > 0, 'probe captured both raw messages (UNIQUE=' + JSON.stringify(rawUniqueMsg) + ', PK=' + JSON.stringify(rawPkMsg) + ')');
    assert(/\b(unique|primary key)\s+constraint\s+failed/i.test(rawUniqueMsg), 'raw UNIQUE message matches the classifier regex, got: ' + rawUniqueMsg);
    assert(/\b(unique|primary key)\s+constraint\s+failed/i.test(rawPkMsg), 'raw composite-PK message matches the classifier regex, got: ' + rawPkMsg);
    assert(isLineageIntegrityFault({ message: rawUniqueMsg }) === true, 'classifier accepts the raw UNIQUE message');
    assert(isLineageIntegrityFault({ message: rawPkMsg }) === true, 'classifier accepts the raw composite-PK message');
    assert(isLineageIntegrityFault({ message: 'disk I/O error' }) === false, 'classifier rejects non-integrity text');
    assert(isLineageIntegrityFault(null) === false, 'classifier rejects null');
    assert(isLineageIntegrityFault({ code: 'SQLITE_CONSTRAINT_UNIQUE' }) === true, 'classifier accepts SQLITE_CONSTRAINT_UNIQUE code');
    assert(isLineageIntegrityFault({ code: 'SQLITE_CONSTRAINT_PRIMARYKEY' }) === true, 'classifier accepts SQLITE_CONSTRAINT_PRIMARYKEY code');

    // FIT-V6: the arm (c) UNIQUE-conflict guard is exercised on a FRESH uv4
    // lineage with TWO committed generations (pre-v1 fixtures are typed-refused
    // at step 3b and can no longer be adopted/migrated). The guard itself —
    // UNIQUE wraps to LineageIntegrityError and ROLLBACKs cleanly — is unchanged.
    const craftedDb = await freshDb();
    ensureLineageFor(craftedDb);
    logGen(craftedDb, 1);
    logGen(craftedDb, 2);
    craftedDb.db.run("UPDATE meta SET value = '123456789' WHERE key = 'rng_state'");
    let threw2 = null;
    try { logGen(craftedDb, 1, { trackHash: DEFAULT_TRACK_HASH, trackJson: DEFAULT_TRACK_JSON }); } catch (err) { threw2 = err; }
    assert(threw2 !== null, 'arm (c) — logGeneration for an existing generation must throw');
    assert(threw2.name === 'LineageIntegrityError', 'arm (c) — expected LineageIntegrityError, got ' + threw2.name + ': ' + threw2.message);
    assert(/UNIQUE/.test(String(threw2.message)), 'arm (c) — message must contain UNIQUE, got: ' + threw2.message);
    assert(countRows(craftedDb, 'generations') === 2, 'arm (c) — ROLLBACK: the migrated generations survive, got ' + countRows(craftedDb, 'generations'));
    assert(countRows(craftedDb, 'driver_results') === 2 * POPULATION, 'arm (c) — the migrated driver_results survive, got ' + countRows(craftedDb, 'driver_results'));
    assert(countRows(craftedDb, 'best_weights') === 2, 'arm (c) — the migrated best_weights survive, got ' + countRows(craftedDb, 'best_weights'));
    const metaC = craftedDb.db.exec("SELECT value FROM meta WHERE key = 'rng_state'")[0].values[0][0];
    assert(metaC === '123456789', 'arm (c) — meta.rng_state unchanged (ROLLBACK), got ' + metaC);
    craftedDb.db.close();
  }
});

// =============================================================================
// M3 leg 24 — resume byte-identity. Reference: ONE uninterrupted Trainer
// (MessageChannel fallback) run, gens 1..5, logged into refDb through the
// same logger-wiring shape M4 will use. Then a prefix lineage (gens 1..2) is
// exported, re-adopted by a fresh Db, resumed via resumeState() +
// Evolution.resumeFromBest, and generation 3 is recomputed — byte-identical
// to the reference. Kill-mid-log: a sabotaged gen-3 log into the adopted
// prefix throws, leaving gens 1..2; resumeState reports maxGeneration=2;
// recomputing gen 3 is byte-identical again. Trainer-handshake (M3-fix F-13):
// a NEW Trainer with lineage.resume over the ADOPTED prefix bytes re-derives
// gens 3..5 byte-identically to the uninterrupted reference (bestWeights
// compared for gens 4 and 5). Per-generation rng_state INTEGER audit column
// equals the reference (M3-fix F-18). resumeState() with the tip generation's
// best_weights row DELETEd post-battery throws ValidationError, never a
// TypeError (M3-fix F-1ii).
// =============================================================================
await leg('leg 24 — resume from max(generation) reproduces gens 3,4,5 byte-identically (incl. kill-mid-log prefix, Trainer handshake, rng audit)', async () => {
  const track = defaultTrack;
  const trackJson = DEFAULT_TRACK_JSON;
  const trackHash = DEFAULT_TRACK_HASH;

  // M4-wiring-shaped adapter: Trainer payload -> Db.logGeneration argument.
  const makeTrainerLogger = (db) => ({
    logGeneration(payload) {
      const results = payload.results.map((r) => ({
        distance: r.distance,
        topSpeed: r.topSpeed,
        timeAlive: r.timeAlive,
        crashed: r.crashed ? 1 : 0,
        stuck: r.stuck ? 1 : 0,
      }));
      let worst = Infinity, distSum = 0, timeSum = 0;
      for (const r of results) {
        if (r.distance < worst) worst = r.distance;
        distSum += r.distance;
        timeSum += r.timeAlive;
      }
      db.logGeneration({
        generation: payload.generation,
        driversEvaluated: results.length,
        bestFitness: results[payload.bestIndex].distance,
        avgFitness: distSum / results.length,
        worstFitness: worst,
        bestDriverIndex: payload.bestIndex,
        mutationRate: MUTATION_RATE,
        mutationScale: MUTATION_SCALE,
        trackJson,
        trackHash,
        simSeconds: timeSum,
        rngState: payload.rngState,
        results,
        bestWeightsJson: payload.bestWeights,
      });
    },
  });

  // ---- Reference: uninterrupted run, gens 1..5.
  const refDb = await freshDb();
  refDb.ensureLineage(trackHash, 4242, trackJson);
  const refTrainer = new Trainer(track, {
    lineage: { seed: 4242, rotationSeed: 4242, generatorVersion: 1 },
    logger: makeTrainerLogger(refDb),
    onBanner() {},
  });
  refTrainer.start(5);
  await waitFor(() => refTrainer.state === 'done', 'reference run reaches done', 300000);
  assert(refDb.listGenerations().join() === '1,2,3,4,5', 'reference DB holds gens 1,2,3,4,5, got ' + refDb.listGenerations());
  const referenceGen3 = JSON.stringify(refDb.getBestWeights(3).getWeights());
  const referenceGen4 = JSON.stringify(refDb.getBestWeights(4).getWeights());
  const referenceGen5 = JSON.stringify(refDb.getBestWeights(5).getWeights());

  // ---- Prefix: fresh lineage, SAME seed, gens 1..2 only.
  const prefixDb = await freshDb();
  prefixDb.ensureLineage(trackHash, 4242, trackJson);
  const prefixTrainer = new Trainer(track, {
    lineage: { seed: 4242, rotationSeed: 4242, generatorVersion: 1 },
    logger: makeTrainerLogger(prefixDb),
    onBanner() {},
  });
  prefixTrainer.start(2);
  await waitFor(() => prefixTrainer.state === 'done', 'prefix run reaches done', 180000);
  assert(prefixDb.listGenerations().join() === '1,2', 'prefix DB holds gens 1,2, got ' + prefixDb.listGenerations());
  const exported = prefixDb.exportBytes();
  prefixDb.db.close();

  // ---- Adopt the prefix bytes in a FRESH Db; resume at gen 2; gen 3 must be
  //      byte-identical to the reference gen 3.
  const { Db } = requireDb();
  const adopted = new Db();
  await adopted.init(exported);
  const st = adopted.resumeState();
  assert(st.maxGeneration === 2, 'resumeState.maxGeneration is 2, got ' + st.maxGeneration);
  assert(typeof st.rngState === 'number' && Number.isInteger(st.rngState) && st.rngState >= 0 && st.rngState <= 4294967295, 'resumeState.rngState is a uint32 number, got ' + st.rngState);
  assert(st.trackHash === trackHash, 'resumeState.trackHash round-trips, got ' + st.trackHash);
  const evo = Evolution.resumeFromBest(st.bestWeights, st.maxGeneration, st.rngState);
  evo.evaluatePopulation(track);
  assert(JSON.stringify(evo.bestWeights()) === referenceGen3, 'recomputed gen 3 (resume from adopted bytes) is byte-identical to the reference');

  // ---- Kill-mid-log: fresh adoption of the same prefix; a sabotaged gen-3
  //      log (NaN distance) throws; gens 1..2 survive; resumeState still
  //      reports maxGeneration=2; gen 3 recomputed byte-identically again.
  const killed = new Db();
  await killed.init(exported);
  const stKilled = killed.resumeState();
  const sabotaged = makeValidGen({ generation: 3, rngState: stKilled.rngState });
  sabotaged.results = makeValidResults();
  sabotaged.results[5] = { distance: NaN, topSpeed: 1, timeAlive: 1, crashed: 0, stuck: 0 };
  let threw = null;
  try { killed.logGeneration(sabotaged); } catch (err) { threw = err; }
  assert(threw !== null, 'sabotaged gen 3 log must throw');
  assert(killed.listGenerations().join() === '1,2', 'kill-mid-log leaves ONLY gens 1,2, got ' + killed.listGenerations());
  const st2 = killed.resumeState();
  assert(st2.maxGeneration === 2, 'resumeState after kill-mid-log reports maxGeneration=2, got ' + st2.maxGeneration);
  const evo2 = Evolution.resumeFromBest(st2.bestWeights, st2.maxGeneration, st2.rngState);
  evo2.evaluatePopulation(track);
  assert(JSON.stringify(evo2.bestWeights()) === referenceGen3, 'recomputed gen 3 (post kill-mid-log) is byte-identical to the reference');

  // ---- Per-generation rng_state INTEGER audit column equals the reference
  //      for the ADOPTED prefix (gens 1..2) — the numbers committed by the
  //      prefix run (M3-fix F-18).
  const refRngRows = refDb.db.exec('SELECT generation, rng_state FROM generations ORDER BY generation')[0].values;
  const refRng = new Map(refRngRows.map((r) => [r[0], r[1]]));
  const auditRng = (instance, label, upto) => {
    const rows = instance.db.exec('SELECT generation, rng_state FROM generations ORDER BY generation')[0].values;
    for (const [g, s] of rows) {
      if (g <= upto) {
        assert(refRng.get(g) === s, label + ' gen ' + g + ' rng_state audit column equals the reference (got ' + s + ' vs ' + refRng.get(g) + ')');
      }
    }
  };
  auditRng(adopted, 'adopted-prefix', 2);

  // ---- Trainer-handshake resume over the ADOPTED prefix bytes (M3-fix F-13):
  //      a NEW Trainer with lineage.resume = resumeState() re-derives gens
  //      3..5 through the REAL M4-wiring-shaped path (Trainer + fallback
  //      core + the logger adapter) — byte-identical to the uninterrupted
  //      reference, gens 4 and 5 bestWeights in particular.
  const adopted2 = new Db();
  await adopted2.init(exported);
  const stHandshake = adopted2.resumeState();
  assert(stHandshake.maxGeneration === 2, 'handshake resumeState.maxGeneration is 2, got ' + stHandshake.maxGeneration);
  const resumedTrainer = new Trainer(track, {
    lineage: {
      resume: {
        bestWeights: stHandshake.bestWeights,
        rngState: stHandshake.rngState,
        generation: stHandshake.maxGeneration,
      },
      rotationSeed: 4242,
      generatorVersion: 1,
    },
    logger: makeTrainerLogger(adopted2),
    onBanner() {},
  });
  resumedTrainer.start(5);
  await waitFor(() => resumedTrainer.state === 'done', 'trainer-handshake resume reaches done', 300000);
  assert(adopted2.listGenerations().join() === '1,2,3,4,5', 'trainer-handshake resumed DB holds gens 1,2,3,4,5, got ' + adopted2.listGenerations());
  assert(JSON.stringify(adopted2.getBestWeights(3).getWeights()) === referenceGen3, 'handshake gen 3 bestWeights byte-identical to the reference');
  assert(JSON.stringify(adopted2.getBestWeights(4).getWeights()) === referenceGen4, 'handshake gen 4 bestWeights byte-identical to the reference');
  assert(JSON.stringify(adopted2.getBestWeights(5).getWeights()) === referenceGen5, 'handshake gen 5 bestWeights byte-identical to the reference');
  auditRng(adopted2, 'handshake-resumed', 5);
  adopted2.db.close();

  // ---- resumeState() refuses a max(generation) with NO best_weights row
  //      (M3-fix F-1ii): battery-passed instance, then the tip weights row is
  //      DELETEd by direct SQL — resumeState() must throw ValidationError,
  //      NEVER a TypeError.
  const stDel = adopted.resumeState();
  assert(stDel.maxGeneration === 2, 'precondition: adopted prefix tip is gen 2');
  adopted.db.run('DELETE FROM best_weights WHERE generation = ?', [stDel.maxGeneration]);
  let threwRm = null;
  try { adopted.resumeState(); } catch (err) { threwRm = err; }
  assert(threwRm !== null, 'resumeState() after DELETEing tip best_weights must throw');
  assert(threwRm.name === 'ValidationError', 'resumeState() must throw ValidationError, got ' + threwRm.name + ': ' + threwRm.message);
  assert(!(threwRm instanceof TypeError), 'resumeState() must NEVER throw a TypeError');
  assert(/best_weights/.test(String(threwRm.message)), 'resumeState() error names the missing best_weights row, got: ' + threwRm.message);

  adopted.db.close();
  killed.db.close();
  refDb.db.close();
});

// =============================================================================
// M3 leg 25 — the ADOPTION BATTERY. Arm (a): valid 2-gen export accepted with
// the coherence summary. Arms (b)-(m): crafted/adversarial bytes rejected
// (truncated prefix, >64 MB pre-parse, trigger, view, track_json cap,
// weights_json cap, extra table, weights payload layer, missing tip
// best_weights, non-tip best_weights only, user_version=999, lookalike
// autoindex). Arm (j): importBytes confirm gate — applies ONLY with
// { confirm: () => true }.
// =============================================================================
await leg('leg 25 — ADOPTION BATTERY: accept + crafted-rejection arms (incl. tip best_weights, user_version, autoindex) + importBytes confirm gate', async () => {
  const { Db } = requireDb();
  const SQL = globalThis.SQL;
  assert(SQL && typeof SQL.Database === 'function', 'SQL namespace available (loadNode ran)');

  // Arm (a) — valid 2-gen export.
  const good = await freshDb();
  ensureLineageFor(good);
  logGen(good, 1);
  logGen(good, 2);
  const goodBytes = good.exportBytes();
  const summary = Db.validateAdoptionBytes(goodBytes);
  assert(summary.generation === 2, 'arm (a) — summary.generation is 2, got ' + summary.generation);
  assert(summary.architecture === '7-16-16-2', 'arm (a) — summary.architecture is 7-16-16-2');
  assert(summary.trackHash === DEFAULT_TRACK_HASH, 'arm (a) — summary.trackHash round-trips, got ' + summary.trackHash);
  assert(typeof summary.rngState === 'number' && Number.isInteger(summary.rngState) && summary.rngState >= 0 && summary.rngState <= 4294967295, 'arm (a) — summary.rngState is a uint32');

  // Shared building block: the EXACT empty schema bytes from a fresh Db, then
  // poison it with raw DDL/DML. The battery must reject every arm.
  const cleanDb = await freshDb();
  const cleanBytes = cleanDb.db.export();
  cleanDb.db.close();
  const buildCraftedBytes = (poison) => {
    const raw = new SQL.Database(cleanBytes);
    raw.run(poison);
    const bytes = raw.export();
    raw.close();
    return bytes;
  };
  const insertGeneration = (raw, trackJson, trackHash) => {
    raw.run('INSERT INTO generations (generation, drivers_evaluated, best_fitness, avg_fitness, worst_fitness, best_driver_index, mutation_rate, mutation_scale, track_json, track_hash, sim_seconds, rng_state) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
      [1, POPULATION, 1, 1, 1, 0, 0.25, 0.3, trackJson, trackHash === undefined ? 'h' : trackHash, 1, 1]);
  };
  // A consistency-keeping tracks pin row for DEFAULT_TRACK_HASH (FK-honest
  // files that must pass the battery's 3a/10b arms before the arm under test
  // fires — the 10b step-10b count arm is belt-and-braces behind 3a).
  const insertDefaultTrackRow = (raw) => {
    raw.run("INSERT INTO tracks (track_hash, source, control_points, half_width, difficulty, created_at) VALUES (?, 'default', ?, 70, 0, '2026-01-01 00:00:00')",
      [DEFAULT_TRACK_HASH, JSON.stringify(DEFAULT_CONTROL_POINTS.map((p) => ({ x: p.x, y: p.y })))]);
  };

  // Arm (d) — crafted BEFORE INSERT trigger.
  assertBatteryReject(Db, buildCraftedBytes('CREATE TRIGGER t BEFORE INSERT ON generations BEGIN UPDATE driver_results SET distance = -distance; END'), /trigger/i, 'arm (d) crafted trigger');
  // Arm (e) — crafted view.
  assertBatteryReject(Db, buildCraftedBytes('CREATE VIEW v AS SELECT * FROM generations'), /view/i, 'arm (e) crafted view');
  // Arm (f) — generations row with track_json over the cap (raw SQL, bypassing
  // logGeneration's validation — exactly the attack shape the battery exists for).
  // A VALID best_weights row for gen 1 is included so the tip-weights refusal
  // (step 7b) does not fire before the track_json cap check.
  const fBytes = (() => {
    const raw = new SQL.Database(cleanBytes);
    insertDefaultTrackRow(raw);
    insertGeneration(raw, 'x'.repeat(300000), DEFAULT_TRACK_HASH);
    raw.run('INSERT INTO best_weights (generation, architecture, weights_json) VALUES (?,?,?)', [1, '7-16-16-2', makeValidWeightsString()]);
    const b = raw.export();
    raw.close();
    return b;
  })();
  assertBatteryReject(Db, fBytes, /track_json/i, 'arm (f) track_json over 262144');
  // Arm (g) — best_weights row with weights_json over the 16 KB cap.
  const gBytes = (() => {
    const raw = new SQL.Database(cleanBytes);
    insertDefaultTrackRow(raw);
    insertGeneration(raw, DEFAULT_TRACK_JSON, DEFAULT_TRACK_HASH);
    raw.run('INSERT INTO best_weights (generation, architecture, weights_json) VALUES (?,?,?)', [1, '7-16-16-2', 'x'.repeat(20000)]);
    const b = raw.export();
    raw.close();
    return b;
  })();
  assertBatteryReject(Db, gBytes, /weights_json/i, 'arm (g) weights_json over 16384');
  // Arm (h) — extra table (well within the schema otherwise).
  assertBatteryReject(Db, buildCraftedBytes('CREATE TABLE evil (x TEXT)'), /sqlite_master|unexpected/i, 'arm (h) extra table');
  // Arm (i) — weights_json with 433 floats (one short): the NeuralNetwork
  // fromJSON validation layer must propagate through the battery.
  const shortWeights = new NeuralNetwork(mulberry32(31337)).getWeights();
  shortWeights.b3 = [shortWeights.b3[0]];
  const iBytes = (() => {
    const raw = new SQL.Database(cleanBytes);
    insertDefaultTrackRow(raw);
    insertGeneration(raw, DEFAULT_TRACK_JSON, DEFAULT_TRACK_HASH);
    raw.run('INSERT INTO best_weights (generation, architecture, weights_json) VALUES (?,?,?)', [1, '7-16-16-2', JSON.stringify(shortWeights)]);
    const b = raw.export();
    raw.close();
    return b;
  })();
  assertBatteryReject(Db, iBytes, /weights/i, 'arm (i) 433-float weights_json');
  // Arm (b) — truncated export: first 100 KB of a REAL export (25 logged
  // generations push the file well past 100 KB).
  const fat = await freshDb();
  ensureLineageFor(fat);
  for (let n = 1; n <= 25; n++) logGen(fat, n);
  const fatBytes = fat.exportBytes();
  assert(fatBytes.length > 100 * 1024, 'fat export exceeds 100 KB (got ' + fatBytes.length + ')');
  assertBatteryReject(Db, fatBytes.subarray(0, 100 * 1024), /battery/i, 'arm (b) truncated first-100KB prefix');
  // Arm (c) — >64 MB refused PRE-parse, fast, without opening sqlite at all.
  const t0 = Date.now();
  assertBatteryReject(Db, new Uint8Array(VENDOR_MAX_BYTES + 1), /64/, 'arm (c) >64 MB pre-parse refusal');
  const elapsed = Date.now() - t0;
  assert(elapsed < 5000, 'arm (c) refused in ' + elapsed + ' ms (< 5000 ms — pre-parse, no sqlite open)');
  // Arm (j) — importBytes confirm gate. Without confirm: refusal, no swap.
  const target = await freshDb();
  ensureLineageFor(target);
  logGen(target, 1);
  let threwConfirm = null;
  try { await target.importBytes(goodBytes); } catch (err) { threwConfirm = err; }
  assert(threwConfirm !== null, 'arm (j) — importBytes WITHOUT a confirm callback must refuse');
  assert(threwConfirm.name === 'ValidationError' && /confirm/i.test(String(threwConfirm.message)), 'arm (j) — refusal must be an explicit confirm-gate ValidationError, got ' + threwConfirm.name + ': ' + threwConfirm.message);
  assert(target.listGenerations().join() === '1', 'arm (j) — refusal leaves the working DB untouched (gens ' + target.listGenerations() + ')');
  const declined = await target.importBytes(goodBytes, { confirm: () => false });
  assert(declined && declined.applied === false, 'arm (j) — confirm()===false → applied:false, got ' + JSON.stringify(declined));
  assert(target.listGenerations().join() === '1', 'arm (j) — declined confirm leaves the working DB untouched');
  const accepted = await target.importBytes(goodBytes, { confirm: (gen) => gen === 2 });
  assert(accepted && accepted.applied === true && accepted.generation === 2, 'arm (j) — confirm()===true applies, got ' + JSON.stringify(accepted));
  assert(target.listGenerations().join() === '1,2', 'arm (j) — after adoption the working DB IS the adopted bytes (gens ' + target.listGenerations() + ')');
  // Arm (k) — generations present but NO best_weights row for MAX(generation)
  //          ⇒ battery rejects (M3-fix F-1i / plan amendment: the battery
  //          requires a best_weights row for MAX(generation) whenever
  //          generations is non-empty).
  const kBytes = (() => {
    const raw = new SQL.Database(cleanBytes);
    insertDefaultTrackRow(raw);
    insertGeneration(raw, DEFAULT_TRACK_JSON, DEFAULT_TRACK_HASH); // gen 1, deliberately NO best_weights row
    const b = raw.export();
    raw.close();
    return b;
  })();
  assertBatteryReject(Db, kBytes, /best_weights/i, 'arm (k) generations without tip best_weights rejected');
  // Arm (k2) — best_weights covers a NON-tip generation only (gen 1 of 1,2)
  //           ⇒ battery still rejects (the tip is what resume needs).
  const k2Bytes = (() => {
    const raw = new SQL.Database(cleanBytes);
    insertDefaultTrackRow(raw);
    insertGeneration(raw, DEFAULT_TRACK_JSON, DEFAULT_TRACK_HASH); // gen 1
    raw.run('INSERT INTO generations (generation, drivers_evaluated, best_fitness, avg_fitness, worst_fitness, best_driver_index, mutation_rate, mutation_scale, track_json, track_hash, sim_seconds, rng_state) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
      [2, POPULATION, 1, 1, 1, 0, 0.25, 0.3, DEFAULT_TRACK_JSON, DEFAULT_TRACK_HASH, 1, 2]);
    raw.run('INSERT INTO best_weights (generation, architecture, weights_json) VALUES (?,?,?)',
      [1, '7-16-16-2', makeValidWeightsString()]);
    const b = raw.export();
    raw.close();
    return b;
  })();
  assertBatteryReject(Db, k2Bytes, /best_weights/i, 'arm (k2) non-tip best_weights only — rejected');
  // Arm (l) — PRAGMA user_version = 999 ⇒ battery rejects (M3-fix F-17:
  //          user_version ∈ {1,2,3}).
  const lBytes = (() => {
    const raw = new SQL.Database(cleanBytes);
    raw.run('PRAGMA user_version = 999');
    const b = raw.export();
    raw.close();
    return b;
  })();
  assertBatteryReject(Db, lBytes, /user_version/i, 'arm (l) user_version=999 rejected');
  // Arm (m) — a lookalike index named sqlite_autoindex_* that does NOT match
  //          the tightened /^sqlite_autoindex_[A-Za-z0-9_]+_[0-9]+$/ shape is
  //          rejected by the scrub like any foreign object (M3-fix F-15).
  //          sqlite itself REFUSES to CREATE a sqlite_*-prefixed index, but a
  //          file crafted outside sqlite's DDL checker can carry one in
  //          sqlite_master (that is exactly the byte-craft the scrub exists
  //          for) — reproduced here via PRAGMA writable_schema.
  const mBytes = (() => {
    const raw = new SQL.Database(cleanBytes);
    raw.run('CREATE INDEX evil_idx ON generations(generation)');
    raw.run('PRAGMA writable_schema = ON');
    raw.run("UPDATE sqlite_master SET name = 'sqlite_autoindex_evil' WHERE name = 'evil_idx'");
    raw.run('PRAGMA writable_schema = OFF');
    const b = raw.export();
    raw.close();
    return b;
  })();
  assertBatteryReject(Db, mBytes, /sqlite_master|sqlite_autoindex|unexpected/i, 'arm (m) lookalike autoindex rejected');
  good.db.close(); fat.db.close(); target.db.close();
});

// =============================================================================
// M3 leg 26 — persist step-2 (throwaway verify) failure: _verifyForPersist
// rejects corrupt bytes; the failure path returns BEFORE any IDB/LSG write
// (spy on the _idbPut hook) and BEFORE any createWritable — the freshness
// baseline and lastKnownGood are untouched.
// =============================================================================
await leg('leg 26 — persist step-2 verify failure: lastKnownGood/IDB hook NOT called, zero writes', async () => {
  const { Db } = requireDb();
  const db = await freshDb();
  ensureLineageFor(db);
  logGen(db, 1);
  // (26a) the exposed throwaway check rejects both garbage and a sliced real DB.
  let garbageThrew = null;
  try { db._verifyForPersist(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8])); } catch (err) { garbageThrew = err; }
  assert(garbageThrew !== null, '26a — garbage bytes must throw');
  let slicedThrew = null;
  const real = db.exportBytes();
  try { db._verifyForPersist(real.subarray(0, Math.floor(real.length / 2))); } catch (err) { slicedThrew = err; }
  assert(slicedThrew !== null, '26a — a truncated real export must throw');

  // (26b) the persist() failure path itself. NOTE: persist() verifies the
  // FRESHLY EXPORTED bytes of the live in-memory DB — a healthy live DB
  // cannot produce a corrupt export, so the step-2 failure is reached here
  // through the documented test seam (Db.prototype._verifyForPersist is
  // exposed for exactly this): it injects the throw. The production code
  // path under test is unchanged — export → verify throws → banner + return
  // { ok:false, reason:'verify' } — and the assertions below pin that the
  // path returns BEFORE any IDB/LSG write and BEFORE any file write.
  const realExport = real.slice();
  db._verifyForPersist = function (bytes) {
    assert(bytes instanceof Uint8Array && bytes.length === realExport.length, '26b — persist verifies THE EXPORTED BYTES');
    const err = new Error('injected step-2 verify failure (test seam)');
    err.name = 'ValidationError';
    throw err;
  };
  let idbPutCalls = 0;
  db._idbPut = async () => { idbPutCalls += 1; return { ok: true }; };
  const createdWritables = [];
  db._fileHandle = {
    async getFile() { return { size: realExport.length, arrayBuffer: async () => realExport.slice().buffer }; },
    async createWritable() { createdWritables.push('createWritable'); throw new Error('createWritable must not be reached on verify failure'); },
  };
  const banners = [];
  db.setPersistenceHooks({ onBanner(b) { banners.push(b); }, persistSkipped: () => false });
  const result = await db.persist();
  assert(result && result.ok === false && result.reason === 'verify', '26b — persist returns {ok:false, reason:"verify"}, got ' + JSON.stringify(result));
  assert(idbPutCalls === 0, '26b — lastKnownGood/IDB write NOT called on verify failure (got ' + idbPutCalls + ')');
  assert(createdWritables.length === 0, '26b — createWritable NOT called on verify failure');
  assert(banners.some((b) => b.type === 'db-verify-failed'), '26b — persistent verify-failure banner fired, got ' + JSON.stringify(banners));
  db.db.close();
});

// =============================================================================
// M3 leg 27 — measured per-persist cost, printed for the record (never
// threshold-gated). A second leg-style line group prints the numbers; the
// 5000-generation wall-clock extrapolation MUST cite these (status.md).
// =============================================================================
await leg('leg 27 — measured per-persist cost at gens 1, 2 and 10 (export / throwaway verify / fs write / total) — report only', async () => {
  for (const genCount of [1, 2, 10]) {
    const db = await freshDb();
    ensureLineageFor(db);
    for (let n = 1; n <= genCount; n++) logGen(db, n);
    const cost = await db.getPersistCost();
    console.log('       === per-persist measurement (' + genCount + '-generation DB, ' + cost.bytes + ' bytes) ===');
    console.log('       export-MS            : ' + cost.exportMs.toFixed(3) + ' ms');
    console.log('       verify-MS            : ' + cost.verifyMs.toFixed(3) + ' ms   (throwaway integrity_check + architecture + coherence)');
    console.log('       fsWrite-MS           : ' + (cost.fsWriteMs === null ? 'n/a (not node)' : cost.fsWriteMs.toFixed(3) + ' ms') + '   (fs.writeFile to os.tmpdir — node proxy for the file write)');
    console.log('       TOTAL-MS             : ' + cost.totalMs.toFixed(3) + ' ms');
    assert(Number.isFinite(cost.exportMs) && Number.isFinite(cost.verifyMs) && Number.isFinite(cost.totalMs), 'all ms values finite (gens=' + genCount + ')');
    assert(cost.bytes > 0, 'export produced bytes (gens=' + genCount + ')');
    db.db.close();
  }
  console.log('       note: O(file size) — a full 5000-gen lineage rewrites a ~25-80 MB file per persist;');
  console.log('       fsWriteMs is a node proxy — a browser persist adds one getFile+arrayBuffer read (tripwire) + one (post-close verify)');
});

// =============================================================================
// M3 leg 28 — grep gates over js/ (comment-stripped): no INSERT-OR spelling,
// no REPLACE-INTO / ON-CONFLICT spelling, no template-literal SQL interpolation
// (a [^`]* gap — MUST fire on quoted shapes), no innerHTML, no eval(, no
// new Function(; importScripts is train-worker.js only with literal ./-relative
// args. RAW-source backstop (M3-fix L1): the exact /INSERT\s+OR\s+/i and
// /(REPLACE\s+INTO|ON\s+CONFLICT)/i regexes are ALSO scanned against the RAW
// file text — a comment tripping them is an honest fail (fix the wording,
// never the gate). Planted-case self-tests pin the matcher as a pure function
// (never false-negative on interpolated shapes). stripComments carries a
// documented limitation fixture (quote inside a regex literal). The shared
// lineage-integrity classifier (db.js) and the Trainer's isUniqueViolation
// delegation are probed for EXACT agreement on the same message battery.
// (CYCLE-5 N5-3 note: the stripped gates scan the code, not the comments.)
// =============================================================================
function stripComments(src) {
  let out = '';
  let i = 0;
  const n = src.length;
  let state = 'code';
  while (i < n) {
    const ch = src[i];
    const next = src[i + 1];
    if (state === 'code') {
      if (ch === '/' && next === '/') { state = 'line'; i += 2; continue; }
      if (ch === '/' && next === '*') { state = 'block'; i += 2; continue; }
      if (ch === "'") { state = 'sq'; out += ch; i += 1; continue; }
      if (ch === '"') { state = 'dq'; out += ch; i += 1; continue; }
      if (ch === '`') { state = 'tpl'; out += ch; i += 1; continue; }
      out += ch; i += 1; continue;
    }
    if (state === 'line') { if (ch === '\n') { state = 'code'; out += '\n'; } i += 1; continue; }
    if (state === 'block') { if (ch === '*' && next === '/') { state = 'code'; i += 2; continue; } i += 1; continue; }
    if (ch === '\\') { out += ch + (src[i + 1] || ''); i += 2; continue; }
    out += ch;
    if (state === 'sq' && ch === "'") state = 'code';
    else if (state === 'dq' && ch === '"') state = 'code';
    else if (state === 'tpl' && ch === '`') state = 'code';
    i += 1;
  }
  return out;
}

// The ONE interpolation-gate pattern (M-NEW6). The gap class is [^`]* — it
// MUST fire on quoted SQL shapes (e.g. '…VALUES ("${z}")'), not only on
// bare ones. Planted cases below pin this.
const SQL_INTERPOLATION_GATE = /(INSERT|UPDATE|SELECT|DELETE)[^`]*\$\{/;

await leg('leg 28 — grep gates: plain-INSERT only, no SQL template interpolation, no DOM sinks/eval in js/ (+ raw backstop + planted self-tests)', () => {
  const { readdirSync, readFileSync } = require('node:fs');
  const path = require('node:path');
  const jsDir = path.join(REPO_ROOT, 'js');
  const files = readdirSync(jsDir).filter((f) => f.endsWith('.js')).sort();
  assert(files.length > 0, 'js/ is non-empty');

  // stripComments DOCUMENTS a limitation (M3-fix F-14): a regex literal that
  // CONTAINS a quote is NOT handled — the quote inside /…/ starts a string
  // state and swallows everything to the next quote (the trailing comment is
  // NOT stripped). No js/ file writes a quote inside a regex literal ([^`]
  // classes are used instead). This fixture pins the behavior so a future
  // regex-with-quote fails HERE, not inside a gate.
  const regexSrc = "var re = /don't/; // trailing comment";
  assert(stripComments(regexSrc) === regexSrc, 'stripComments must leave a quote-containing regex literal verbatim (documented limitation), got: ' + JSON.stringify(stripComments(regexSrc)));

  // PLANTED-CASE self-tests of the matcher (fail-safe: NEVER false-negative on
  // an interpolated shape; the fully bound form must NOT match).
  const mustFire = [
    "INSERT INTO t(k,v) VALUES ('${x}', '${y}')",
    'INSERT INTO t VALUES ("${z}")',
    "UPDATE meta SET v = '${w}'",
    'UPDATE meta SET value = "${w}" WHERE key = "${k}"',
    'DELETE FROM generations WHERE generation = ${g}',
    'SELECT value FROM meta WHERE key = ${k}',
  ];
  for (const s of mustFire) {
    assert(SQL_INTERPOLATION_GATE.test(s), 'planted interpolation MUST match: ' + s);
  }
  const mustNotFire = [
    "db.run('INSERT INTO t VALUES (?)', [v])",
    'db.run("UPDATE meta SET value = ? WHERE key = ?", [value, key])',
    'db.exec(\'SELECT * FROM generations WHERE generation = ?\', [n])',
  ];
  for (const s of mustNotFire) {
    assert(!SQL_INTERPOLATION_GATE.test(s), 'bound-parameter form must NOT match: ' + s);
  }

  const codeByFile = new Map(files.map((f) => [f, stripComments(readFileSync(path.join(jsDir, f), 'utf8'))]));
  const rawByFile = new Map(files.map((f) => [f, readFileSync(path.join(jsDir, f), 'utf8')]));
  for (const [f, code] of codeByFile) {
    assert(!/INSERT\s+OR\s+/i.test(code), f + ': /INSERT\\s+OR\\s+/i gate (plain INSERT only)');
    assert(!/(REPLACE\s+INTO|ON\s+CONFLICT)/i.test(code), f + ': /(REPLACE\\s+INTO|ON\\s+CONFLICT)/i gate (M-NEW6)');
    assert(!SQL_INTERPOLATION_GATE.test(code), f + ': /(INSERT|UPDATE|SELECT|DELETE)[^`]*\\$\\{/ gate (M-NEW6)');
    assert(!/innerHTML/i.test(code), f + ': innerHTML gate');
    assert(!/\beval\s*\(/.test(code), f + ': eval( gate');
    assert(!/new\s+Function\s*\(/.test(code), f + ': new Function( gate');
  }
  // RAW backstop (M3-fix L1): the exact regexes must ALSO match nothing in
  // the raw source. A comment tripping this is an honest fail — fix the
  // comment wording, never the gate.
  for (const [f, raw] of rawByFile) {
    assert(!/INSERT\s+OR\s+/i.test(raw), f + ': raw /INSERT\\s+OR\\s+/i backstop gate');
    assert(!/(REPLACE\s+INTO|ON\s+CONFLICT)/i.test(raw), f + ': raw /(REPLACE\\s+INTO|ON\\s+CONFLICT)/i backstop gate');
  }
  // Shared lineage-integrity classifier (M3-fix F-2): Trainer's
  // isUniqueViolation DELEGATES to db.js's isLineageIntegrityFault. Probe
  // BOTH — including trainer.js's inline fallback with the global deleted —
  // on the same message battery so the two paths can never drift.
  const { isLineageIntegrityFault } = requireDb();
  const { isUniqueViolation } = require('../js/trainer.js');
  const probes = [
    { message: 'UNIQUE constraint failed: generations.generation' },
    { message: 'UNIQUE constraint failed: driver_results.generation, driver_results.driver_index' },
    { message: 'UNIQUE constraint failed: t.b' },
    { message: 'disk I/O error' },
    { message: 'too many terms in compound SELECT' },
    { message: 'FOREIGN KEY constraint failed: driver_results.generation', code: 'SQLITE_CONSTRAINT_FOREIGNKEY' },
    { code: 'SQLITE_CONSTRAINT_UNIQUE', message: 'x' },
    { code: 'SQLITE_CONSTRAINT_PRIMARYKEY', message: 'y' },
    { code: 'FIT_V3_ROTATION_DIVERGENCE', message: 'rotation divergence — never eligible for continue-unlogged' },
    'UNIQUE constraint failed: raw string passed as the error itself',
    null,
    undefined,
    {},
    42,
  ];
  assert(typeof globalThis.isLineageIntegrityFault === 'function', 'db.js installs globalThis.isLineageIntegrityFault');
  for (const p of probes) {
    assert(isUniqueViolation(p) === isLineageIntegrityFault(p), 'isUniqueViolation and isLineageIntegrityFault disagree on ' + JSON.stringify(p));
  }
  const savedClassifier = globalThis.isLineageIntegrityFault;
  delete globalThis.isLineageIntegrityFault; // db.js "absent" — trainer.js inline fallback must classify identically
  try {
    for (const p of probes) {
      assert(isUniqueViolation(p) === isLineageIntegrityFault(p), 'trainer.js fallback disagrees on ' + JSON.stringify(p));
    }
  } finally {
    if (savedClassifier !== undefined) globalThis.isLineageIntegrityFault = savedClassifier;
  }
  assert(globalThis.isLineageIntegrityFault === savedClassifier, 'globalThis.isLineageIntegrityFault restored after the fallback probe');
  for (const [f, code] of codeByFile) {
    const hasImportScripts = /importScripts\s*\(/.test(code);
    if (f === 'train-worker.js') {
      assert(hasImportScripts, 'train-worker.js still uses importScripts with a whitelist');
      const calls = code.match(/importScripts\([^)]*\)/g) || [];
      for (const call of calls) {
        const args = call.slice(call.indexOf('(') + 1, -1);
        assert(/^(\s*'\.\/[A-Za-z0-9_.-]+'\s*,)*\s*'\.\/[A-Za-z0-9_.-]+'\s*$/.test(args), 'train-worker.js importScripts args must be literal ./-relative strings: ' + call);
      }
    } else {
      assert(!hasImportScripts, f + ': importScripts is train-worker.js only');
    }
  }
});

// =============================================================================
// M3 leg 29 — persist() END-TO-END in node (injected handle): battery-passed
// adoption → baseline recorded FROM THE ADOPTED BYTES → tripwire passes →
// lastKnownGood written (AFTER verify+tripwire) → exclusive-write seam →
// close → post-close size+SHA-256 assert → baseline refreshed +
// lineageSnapshot refreshed. Then: (2) a file rewritten outside the tab ⇒
// DO NOT WRITE (Reconcile banner, zero LSG/writable calls); (3) a stale file
// + a lying close() ⇒ post-close verification fails ⇒ "DB NOT PERSISTING",
// baseline NOT refreshed; (4) the SAME pipeline under a STUBBED
// navigator.locks (car_racer-db / exclusive) — the callback runs exactly
// once and the outcome is ok (M3-fix F-12; save/restored in a finally).
// =============================================================================
await leg('leg 29 — persist end-to-end (node, injected handle): baseline → LSG → write → post-close; tripwire + post-close failure paths + stubbed navigator.locks', async () => {
  const { Db } = requireDb();
  // Seed a 1-gen lineage in a scratch Db; export its bytes.
  const seed = await freshDb();
  ensureLineageFor(seed);
  logGen(seed, 1);
  const adoptedBytes = seed.exportBytes();
  seed.db.close();
  // Adopt the bytes in a fresh Db (ADOPTION BATTERY runs; baseline recorded
  // FROM THE ADOPTED BYTES) and wire a stub file holding exactly those bytes.
  const db = new Db();
  await db.init(adoptedBytes);
  let fileState = { bytes: adoptedBytes.slice() };
  const banners = [];
  const idbCalls = [];
  db._idbPut = async (key, value) => {
    idbCalls.push({ key, maxGeneration: value && value.tag ? value.tag.maxGeneration : null });
    return { ok: true };
  };
  db.setPersistenceHooks({ onBanner(b) { banners.push(b); }, persistSkipped: () => false });
  const writes = [];
  db._createWritable = async () => {
    const w = {
      written: null,
      closed: false,
      async write(b) { w.written = new Uint8Array(b); },
      async close() { fileState.bytes = w.written; w.closed = true; },
      async abort() { w.aborted = true; },
    };
    writes.push(w);
    return w;
  };
  db._fileHandle = {
    async getFile() { return { size: fileState.bytes.length, arrayBuffer: async () => fileState.bytes.slice().buffer }; },
  };

  // (1) Untampered adopted file → full pipeline succeeds.
  const okResult = await db.persist();
  assert(okResult && okResult.ok === true, '29(1) — persist succeeds on an untampered adopted file, got ' + JSON.stringify(okResult));
  assert(okResult.bytesWritten === fileState.bytes.length, '29(1) — success object carries bytesWritten === byte count, got ' + JSON.stringify(okResult.bytesWritten));
  assert(!Object.prototype.hasOwnProperty.call(okResult, 'bytes'), '29(1) — the success object carries NO bare `bytes` key (M3-fix L2)');
  assert(writes.length === 1 && writes[0].closed && writes[0].written && writes[0].written.length === fileState.bytes.length, '29(1) — the verified bytes were written and closed');
  for (let i = 0; i < fileState.bytes.length; i++) {
    if (fileState.bytes[i] !== writes[0].written[i]) throw new Error('29(1) — the file does not hold the freshly exported bytes');
  }
  assert(idbCalls.some((c) => c.key === 'car_racer.lastKnownGood'), '29(1) — lastKnownGood written AFTER verify+tripwire');
  assert(idbCalls.some((c) => c.key === 'car_racer.lineageSnapshot'), '29(1) — lineageSnapshot refreshed on success');
  const goodFileBytes = fileState.bytes.slice();
  assert(db._freshnessAtOpen && db._freshnessAtOpen.size === goodFileBytes.length, '29(1) — freshness baseline refreshed after success');

  // (2) File rewritten OUTSIDE the tab → tripwire must refuse to write.
  const tampered = fileState.bytes.slice();
  tampered[tampered.length - 1] = tampered[tampered.length - 1] ^ 0xff;
  fileState.bytes = tampered;
  const beforeLsg = idbCalls.filter((c) => c.key === 'car_racer.lastKnownGood').length;
  const beforeWrites = writes.length;
  const r2 = await db.persist();
  assert(r2.ok === false && r2.reason === 'freshness', '29(2) — tampered file ⇒ Reconcile required, got ' + JSON.stringify(r2));
  assert(writes.length === beforeWrites, '29(2) — NO writable created after tamper');
  assert(idbCalls.filter((c) => c.key === 'car_racer.lastKnownGood').length === beforeLsg, '29(2) — NO LSG write after tamper');
  assert(banners.some((b) => b.type === 'reconcile-required'), '29(2) — Reconcile banner fired, got ' + JSON.stringify(banners));

  // (3) DB changed (gen 2 logged), file kept EXACTLY at the baseline (so the
  // tripwire passes), but the writable's close() does not actually replace
  // the file bytes → post-close verification MUST fail and the baseline
  // must NOT be refreshed.
  fileState.bytes = goodFileBytes.slice(); // restore the baseline bytes
  logGen(db, 2);                            // the export now DIFFERS from goodFileBytes
  db._createWritable = async () => {
    const w = {
      written: null,
      async write(b) { w.written = new Uint8Array(b); },
      async close() { w.closed = true; /* fileState deliberately NOT updated — lying close() */ },
      async abort() { w.aborted = true; },
    };
    writes.push(w);
    return w;
  };
  const baselineSize = db._freshnessAtOpen.size;
  const r3 = await db.persist();
  assert(r3.ok === false && r3.reason === 'post-close', '29(3) — lying close() ⇒ post-close verification failure, got ' + JSON.stringify(r3));
  assert(banners.some((b) => b.type === 'db-not-persisting' && b.detail && /post-close/.test(b.detail)), '29(3) — post-close banner fired, got ' + JSON.stringify(banners));
  assert(db._freshnessAtOpen && db._freshnessAtOpen.size === baselineSize, '29(3) — freshness baseline NOT refreshed after post-close failure');

  // (4) The SAME end-to-end pipeline under a STUBBED Web Locks API (M3-fix
  // F-12): steps 3–6 run inside ONE exclusive 'car_racer-db' lock. Node has
  // no navigator.locks, so the lock path is untested without the stub; the
  // stub asserts the lock name + exclusive mode and runs the callback.
  // NOTE: node defines globalThis.navigator as a GETTER — the stub is
  // installed via defineProperty and the ORIGINAL property DESCRIPTOR is
  // restored in the finally (never a plain assignment).
  const lockCalls = [];
  const savedDesc = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {
    value: {
      locks: {
        async request(name, lockOpts, cb) {
          assert(name === 'car_racer-db' && lockOpts && lockOpts.mode === 'exclusive', '29(4) — lock request is (car_racer-db, exclusive), got ' + JSON.stringify([name, lockOpts]));
          lockCalls.push(name);
          return cb();
        },
      },
    },
    configurable: true,
    writable: true,
    enumerable: true,
  });
  try {
    const db4 = new Db();
    await db4.init(adoptedBytes);
    const fileState4 = { bytes: adoptedBytes.slice() };
    const banners4 = [];
    db4._idbPut = async () => ({ ok: true });
    db4.setPersistenceHooks({ onBanner(b) { banners4.push(b); }, persistSkipped: () => false });
    db4._createWritable = async () => {
      const w = {
        written: null,
        async write(b) { w.written = new Uint8Array(b); },
        async close() { fileState4.bytes = w.written; },
        async abort() { w.aborted = true; },
      };
      return w;
    };
    db4._fileHandle = {
      async getFile() { return { size: fileState4.bytes.length, arrayBuffer: async () => fileState4.bytes.slice().buffer }; },
    };
    const r4 = await db4.persist();
    assert(r4 && r4.ok === true, '29(4) — persist under stubbed navigator.locks succeeds, got ' + JSON.stringify(r4));
    assert(r4.bytesWritten === fileState4.bytes.length, '29(4) — bytesWritten is the byte count, got ' + JSON.stringify(r4.bytesWritten));
    assert(lockCalls.length === 1 && lockCalls[0] === 'car_racer-db', '29(4) — the lock callback ran exactly once, got ' + JSON.stringify(lockCalls));
    db4.db.close();
  } finally {
    // Node must not leak the stub into the other legs — restore the ORIGINAL
    // property descriptor (node's navigator is a getter), or delete ours.
    if (savedDesc) {
      Object.defineProperty(globalThis, 'navigator', savedDesc);
    } else {
      delete globalThis.navigator;
    }
  }
  db.db.close();
});

// =============================================================================
// M3 leg 30 — SEC-22 adoption paths, node-side halves: (i) open of a
// crafted BEFORE-INSERT-trigger .sqlite is REFUSED at openExisting — zero
// adoption, the working DB is untouched (M4 maps this banner to
// Manual-only boot); (ii) a trigger-poisoned cache-offer (what a poisoned
// car_racer.lastKnownGood/lineageSnapshot would be) is refused by the SAME
// battery the M4 reconcile flow must run before offering it — it is never
// offered; valid bytes DO adopt, with the baseline re-recorded from the
// adopted bytes; (iii) adopted bytes; (iv)–(vi) the handle/pick size gate —
// files larger than MAX_DB_BYTES are refused BEFORE arrayBuffer() runs,
// and a rejecting arrayBuffer() surfaces as a typed ValidationError.
// =============================================================================
await leg('leg 30 — SEC-22 adoption paths: trigger bytes refused at openExisting + at the cache-offer gate; valid bytes adopt + handle-file size gate', async () => {
  const { Db, MAX_DB_BYTES } = requireDb();
  const SQL = globalThis.SQL;
  const cleanDb = await freshDb();
  const cleanBytes = cleanDb.db.export();
  cleanDb.db.close();
  // Craft a trigger-bearing snapshot — the attack shape a poisoned cache holds.
  const raw = new SQL.Database(cleanBytes);
  raw.run('CREATE TRIGGER t BEFORE INSERT ON generations BEGIN UPDATE driver_results SET distance = -distance; END');
  const poisoned = raw.export();
  raw.close();

  // (i) open path — refused, ZERO adoption, working DB untouched.
  const working = await freshDb();
  ensureLineageFor(working);
  logGen(working, 1);
  const gensBefore = working.listGenerations().join();
  let threwOpen = null;
  try { await working.openExisting({ bytes: poisoned }); } catch (err) { threwOpen = err; }
  assert(threwOpen !== null, '(i) — crafted trigger bytes refused at openExisting');
  assert(threwOpen.name === 'ValidationError', '(i) — refusal is a ValidationError, got ' + threwOpen.name + ': ' + threwOpen.message);
  assert(/trigger|battery/i.test(String(threwOpen.message)), '(i) — refusal message names the battery/trigger, got: ' + threwOpen.message);
  assert(working.listGenerations().join() === gensBefore, '(i) — working DB UNTOUCHED after refused open');

  // (ii) reconcile cache-offer path — the SAME single choke point refuses the
  // poison before it could be offered to the user.
  assertBatteryReject(Db, poisoned, /trigger/i, '(ii) poisoned lastKnownGood/lineageSnapshot bytes refused at the cache-offer battery');

  // (iii) battery-passed adoption — and the freshness baseline is re-recorded
  // FROM THE ADOPTED BYTES (a subsequent verified persist does not tripwire).
  const adopted = new Db();
  const adoption = await adopted.openExisting({ bytes: cleanBytes });
  assert(adoption && adoption.ok === true && adoption.generation === 0, '(iii) — clean bytes adopt, got ' + JSON.stringify(adoption));
  assert(adopted._freshnessAtOpen && adopted._freshnessAtOpen.size === cleanBytes.length, '(iii) — baseline recorded FROM THE ADOPTED BYTES');
  adopted.db.close();
  // (iv) oversized HANDLE file ⇒ ValidationError BEFORE arrayBuffer() is ever
  // called (security M3). 64 MiB + 1 byte announced by getFile() — the reader
  // must NOT run.
  let arrayBufferCalls = 0;
  const oversize = MAX_DB_BYTES + 1;
  const bigHandle = {
    async getFile() {
      return {
        size: oversize,
        async arrayBuffer() { arrayBufferCalls += 1; throw new Error('arrayBuffer must not be called'); },
      };
    },
  };
  let threwBig = null;
  try { await working.openExisting({ handle: bigHandle }); } catch (err) { threwBig = err; }
  assert(threwBig !== null && threwBig.name === 'ValidationError' && /too large/.test(String(threwBig.message)), '(iv) — oversize handle ⇒ ValidationError(too large), got ' + (threwBig && threwBig.name + ': ' + threwBig.message));
  assert(/file too large \(67108865 bytes\)/.test(String(threwBig.message)), '(iv) — message names the byte count, got: ' + threwBig.message);
  assert(arrayBufferCalls === 0, '(iv) — arrayBuffer NOT called for an oversize file, got ' + arrayBufferCalls);
  assert(working.listGenerations().join() === '1', '(iv) — working DB untouched after refused big open');
  // (v) getFile().arrayBuffer() REJECTS ⇒ typed ValidationError (never a raw rejection).
  const readFailHandle = {
    async getFile() {
      return { size: 4096, async arrayBuffer() { throw new Error('simulated I/O failure mid-read'); } };
    },
  };
  let threwRead = null;
  try { await working.openExisting({ handle: readFailHandle }); } catch (err) { threwRead = err; }
  assert(threwRead !== null && threwRead.name === 'ValidationError', '(v) — arrayBuffer rejection ⇒ typed ValidationError, got ' + (threwRead && threwRead.name));
  assert(working.listGenerations().join() === '1', '(v) — working DB untouched after failed read');
  // (vi) the { pick: true } browser branch enforces the SAME gates
  // (showOpenFilePicker stubbed; save/restore).
  const savedPicker = globalThis.showOpenFilePicker;
  const hadPicker = Object.prototype.hasOwnProperty.call(globalThis, 'showOpenFilePicker');
  globalThis.showOpenFilePicker = async () => [bigHandle];
  try {
    let threwPick = null;
    try { await working.openExisting({ pick: true }); } catch (err) { threwPick = err; }
    assert(threwPick !== null && threwPick.name === 'ValidationError' && /too large/.test(String(threwPick.message)), '(vi) — pick branch refuses an oversize file, got ' + (threwPick && threwPick.name + ': ' + threwPick.message));
    assert(arrayBufferCalls === 0, '(vi) — arrayBuffer NOT called on the pick branch either, got ' + arrayBufferCalls);
    assert(working.listGenerations().join() === '1', '(vi) — working DB untouched after refused pick');
  } finally {
    if (hadPicker) { globalThis.showOpenFilePicker = savedPicker; } else { delete globalThis.showOpenFilePicker; }
  }
  working.db.close();
});

// =============================================================================
// M3 leg 31 — fingerprint fault injection (arch F-5 / security M1). The
// crypto.subtle digest can reject (permission/CSP/quota). persist() must NOT
// reject: it fires the persistent banner and RESOLVES { ok:false,
// reason:'fingerprint' }; training continues. init(bytes) and openExisting()
// throw a typed ValidationError with the working DB untouched. The
// _fingerprintBytes seam is overridden in each case and RESTORED afterwards
// — the restored hook is verified to run the real implementation.
// =============================================================================
await leg('leg 31 — fingerprint guards: persist ⇒ {ok:false, reason:"fingerprint"} + banner (never a rejection); init(bytes)/openExisting ⇒ ValidationError', async () => {
  const { Db } = requireDb();
  const db = await freshDb();
  ensureLineageFor(db);
  logGen(db, 1);
  const realFp = db._fingerprintBytes; // the prototype method (bound on use)
  const banners = [];
  db.setPersistenceHooks({ onBanner(b) { banners.push(b); }, persistSkipped: () => false });

  // (a) persist() — forced rejection of the fingerprint hook.
  let calls = 0;
  db._fingerprintBytes = async function () { calls += 1; throw new Error('injected crypto.subtle rejection'); };
  let persistThrew = null;
  let persistResult = null;
  try {
    persistResult = await db.persist();
  } catch (err) {
    persistThrew = err;
  }
  assert(persistThrew === null, 'persist() must NOT reject on a fingerprint failure, got: ' + String(persistThrew && persistThrew.message));
  assert(persistResult && persistResult.ok === false && persistResult.reason === 'fingerprint', 'persist() resolves {ok:false, reason:"fingerprint"}, got ' + JSON.stringify(persistResult));
  assert(persistResult.error && /crypto|fingerprint/i.test(String(persistResult.error)), 'persist() result carries the fingerprint error detail, got ' + JSON.stringify(persistResult));
  assert(banners.some((b) => b.type === 'db-not-persisting' && b.detail && /fingerprint/.test(b.detail)), 'persistent fingerprint banner fired, got ' + JSON.stringify(banners));
  assert(calls === 1, 'the fingerprint hook was attempted exactly once, got ' + calls);
  db._fingerprintBytes = realFp; // restore
  // The restored hook runs the REAL implementation.
  const realDigest = await db._fingerprintBytes(new Uint8Array([1, 2, 3, 4]));
  assert(realDigest && realDigest.size === 4, 'restored fingerprint hook runs the real implementation (size ' + (realDigest && realDigest.size) + ')');
  db.db.close();

  // (b) init(bytes) — forced rejection ⇒ typed ValidationError; ZERO adoption.
  const seedDbB = await freshDb();
  ensureLineageFor(seedDbB);
  logGen(seedDbB, 1);
  const validBytes = seedDbB.exportBytes();
  seedDbB.db.close();
  const freshInit = new Db();
  freshInit._fingerprintBytes = async function () { throw new Error('injected fingerprint rejection'); };
  let threwInit = null;
  try {
    await freshInit.init(validBytes);
  } catch (err) {
    threwInit = err;
  }
  assert(threwInit !== null && threwInit.name === 'ValidationError', 'init(bytes) ⇒ ValidationError, got ' + (threwInit && threwInit.name + ': ' + threwInit.message));
  assert(/fingerprint/i.test(String(threwInit.message)), 'init(bytes) error names the fingerprint failure, got: ' + threwInit.message);
  assert(freshInit.db === null, 'init(bytes) — ZERO adoption (working DB untouched)');

  // (c) openExisting({ handle }) — _activateBytes fingerprint failure ⇒
  // typed ValidationError; the working DB is untouched.
  const working = await freshDb();
  ensureLineageFor(working);
  logGen(working, 1);
  const gensBefore = working.listGenerations().join();
  working._fingerprintBytes = async function () { throw new Error('injected fingerprint rejection'); };
  const fakeHandle = {
    async getFile() {
      return { size: validBytes.length, async arrayBuffer() { return validBytes.slice().buffer; } };
    },
  };
  let threwOpen = null;
  try {
    await working.openExisting({ handle: fakeHandle });
  } catch (err) {
    threwOpen = err;
  }
  assert(threwOpen !== null && threwOpen.name === 'ValidationError', 'openExisting ⇒ ValidationError, got ' + (threwOpen && threwOpen.name + ': ' + threwOpen.message));
  assert(/fingerprint/i.test(String(threwOpen.message)), 'openExisting error names the fingerprint failure, got: ' + threwOpen.message);
  assert(working.listGenerations().join() === gensBefore, 'openExisting — working DB untouched (gens ' + working.listGenerations() + ')');
  working.db.close();
  if (freshInit.db) freshInit.db.close();
});

// =============================================================================
// M3 leg 32 — the security-M4 (combined) battery: (a) forkLineage wipe
// semantics — generations/driver_results/best_weights all empty, meta
// re-pinned, sqlite_master scrub-clean (verified via the adoption battery,
// whose step 9 IS the scrub); (b) resetLineageSnapshot() in node ⇒
// { ok:false, reason:'idb-unavailable' }; (c) init(bytes) with crafted
// trigger bytes ⇒ throws (battery), ZERO adoption; (d) importBytes with a
// confirm spy ⇒ the spy is NEVER called and the working DB is unchanged;
// (e) a battery-LEGAL DB with meta.rng_state DELETEd ⇒ resumeState() throws
// ValidationError.
// =============================================================================
await leg('leg 32 — security M4 combined: forkLineage wipe, resetLineageSnapshot node no-op, trigger-init refusal, confirm-spy never called, deleted rng_state pin', async () => {
  const { Db } = requireDb();
  const SQL = globalThis.SQL;

  // (a) forkLineage — log 2 gens, fork, everything wiped, meta re-pinned.
  const forked = await freshDb();
  ensureLineageFor(forked);
  logGen(forked, 1);
  logGen(forked, 2);
  const otherPoints = DEFAULT_CONTROL_POINTS.map((p) => ({ x: p.x + 5, y: p.y + 5 }));
  const otherTrack = new Track(otherPoints, 80);
  const otherJson = otherTrack.toJSON();
  const otherHash = otherTrack.hash();
  forked.forkLineage(otherHash, 1000, otherJson);
  assert(countRows(forked, 'generations') === 0 && countRows(forked, 'driver_results') === 0 && countRows(forked, 'best_weights') === 0,
    '(a) — forkLineage wipes generations/driver_results/best_weights (got ' + countRows(forked, 'generations') + '/' + countRows(forked, 'driver_results') + '/' + countRows(forked, 'best_weights') + ')');
  const rngAfterFork = forked.db.exec("SELECT value FROM meta WHERE key = 'rng_state'")[0].values[0][0];
  const hashAfterFork = forked.db.exec("SELECT value FROM meta WHERE key = 'track_hash'")[0].values[0][0];
  assert(rngAfterFork === '1000' && hashAfterFork === otherHash, '(a) — meta re-pinned to rng_state=1000, track_hash=<other track>, got ' + rngAfterFork + '/' + hashAfterFork);
  // scrubSqliteMaster is module-private; the battery IS the scrub's
  // enforcement point — the forked DB must pass it untouched.
  let forkBatteryThrew = null;
  try { Db.validateAdoptionBytes(forked.exportBytes()); } catch (err) { forkBatteryThrew = err; }
  assert(forkBatteryThrew === null, '(a) — forked DB passes the adoption battery (sqlite_master scrub), got: ' + (forkBatteryThrew && forkBatteryThrew.message));
  // (b) resetLineageSnapshot in node (no IndexedDB) ⇒ typed no-op.
  const snap = await forked.resetLineageSnapshot();
  assert(snap && snap.ok === false && snap.reason === 'idb-unavailable', '(b) — resetLineageSnapshot ⇒ {ok:false, reason:"idb-unavailable"} in node, got ' + JSON.stringify(snap));
  forked.db.close();

  // (c) init(bytes) with crafted trigger bytes ⇒ throws (battery), zero adoption.
  const cleanDb = await freshDb();
  const cleanBytes = cleanDb.db.export();
  cleanDb.db.close();
  const rawTrigger = new SQL.Database(cleanBytes);
  rawTrigger.run('CREATE TRIGGER t BEFORE INSERT ON generations BEGIN UPDATE driver_results SET distance = -distance; END');
  const triggerBytes = rawTrigger.export();
  rawTrigger.close();
  const triggerInit = new Db();
  let threwC = null;
  try { await triggerInit.init(triggerBytes); } catch (err) { threwC = err; }
  assert(threwC !== null && threwC.name === 'ValidationError', '(c) — trigger bytes refused at init(bytes), got ' + (threwC && threwC.name + ': ' + threwC.message));
  assert(/trigger|sqlite_master/i.test(String(threwC.message)), '(c) — refusal names the trigger/sqlite_master violation, got: ' + threwC.message);
  assert(triggerInit.db === null, '(c) — ZERO adoption (working DB untouched)');

  // (d) importBytes(poisoned, { confirm: spy }) ⇒ confirm NEVER called, throws;
  // working DB unchanged.
  const working = await freshDb();
  ensureLineageFor(working);
  logGen(working, 1);
  let spyCalls = 0;
  let threwD = null;
  try {
    await working.importBytes(triggerBytes, { confirm: () => { spyCalls += 1; return true; } });
  } catch (err) {
    threwD = err;
  }
  assert(spyCalls === 0, '(d) — the confirm callback was NEVER called on battery failure, got ' + spyCalls);
  assert(threwD !== null && /sqlite_master|trigger|import/i.test(String(threwD.message)), '(d) — refusal matches /sqlite_master|trigger|import/i, got: ' + (threwD && threwD.name + ': ' + threwD.message));
  assert(working.listGenerations().join() === '1', '(d) — working DB unchanged (gens ' + working.listGenerations() + ')');

  // (e) battery-LEGAL DB with meta.rng_state DELETEd ⇒ resumeState() throws
  // ValidationError. Construct: valid 2-gen export, then DELETE the pin.
  const good = await freshDb();
  ensureLineageFor(good);
  logGen(good, 1);
  logGen(good, 2);
  const goodBytes = good.exportBytes();
  good.db.close();
  const rawE = new SQL.Database(goodBytes);
  rawE.run("DELETE FROM meta WHERE key = 'rng_state'");
  const eBytes = rawE.export();
  rawE.close();
  // battery allows it (rng_state is not required by the battery)…
  let batteryThrew = null;
  try { Db.validateAdoptionBytes(eBytes); } catch (err) { batteryThrew = err; }
  assert(batteryThrew === null, '(e) — battery admits a missing meta.rng_state (the resume path is where it refuses), got: ' + (batteryThrew && batteryThrew.message));
  const eDb = new Db();
  await eDb.init(eBytes);
  let threwE = null;
  try { eDb.resumeState(); } catch (err) { threwE = err; }
  assert(threwE !== null && threwE.name === 'ValidationError', '(e) — resumeState() ⇒ ValidationError, got ' + (threwE && threwE.name + ': ' + threwE.message));
  assert(/rng_state/i.test(String(threwE.message)), '(e) — error names meta.rng_state, got: ' + threwE.message);
  eDb.db.close();
  working.db.close();
});

// =============================================================================
// M3 leg 33 — read-path result-set shapes (DB-FIX): sql.js exec() returns an
// ARRAY of result sets; db.getDriverResults() must unwrap rows[0] before
// rowObjects(). Silently broken until this fix: after one logged generation
// with 20 results it returned [] (a full UI dependency — Logs tab expand +
// replay stats). One leg pins getDriverResults end-to-end AND keeps the
// array-vs-set siblings pinned so they cannot regress into the same class:
// getGeneration / listGenerations / stats / resumeState, on BOTH a populated
// and an empty database.
// =============================================================================
await leg('leg 33 — read paths: getDriverResults(1) returns 20 typed rows + getGeneration/listGenerations/stats/resumeState shapes (populated + empty DB)', async () => {
  const db = await freshDb();
  ensureLineageFor(db);
  logGen(db, 1);
  // The silently-broken path:
  const results = db.getDriverResults(1);
  assert(Array.isArray(results), 'getDriverResults returns an array');
  assert(results.length === POPULATION, 'getDriverResults(1).length === 20, got ' + results.length);
  for (let i = 0; i < POPULATION; i++) {
    assert(results[i].driver_index === i, 'results[' + i + '].driver_index === ' + i + ' (got ' + results[i].driver_index + ')');
    assert(typeof results[i].distance === 'number' && typeof results[i].top_speed === 'number' && typeof results[i].time_alive === 'number', 'results[' + i + '] distance/top_speed/time_alive are numbers');
    assert((results[i].crashed === 0 || results[i].crashed === 1) && (results[i].stuck === 0 || results[i].stuck === 1), 'results[' + i + '] crashed/stuck are stored 0/1 (got ' + results[i].crashed + '/' + results[i].stuck + ')');
  }
  // Sibling array-vs-set read paths — verified correct in the audit, pinned here:
  const gen = db.getGeneration(1);
  assert(gen && gen.generation === 1 && typeof gen.best_fitness === 'number' && typeof gen.track_hash === 'string', 'getGeneration(1) returns the generation row object, got ' + JSON.stringify(gen));
  assert(db.getGeneration(9999) === null, 'getGeneration(9999) === null (fail closed on missing rows)');
  assert(db.listGenerations().join() === '1', 'listGenerations() returns [1], got ' + db.listGenerations());
  const stats = db.stats();
  assert(stats.generationCount === 1 && stats.latestGeneration === 1 && stats.bestFitnessEver === 1000 && stats.avgOfLatest === 900, 'stats() reports generationCount=1, latestGeneration=1, bestFitnessEver=1000, avgOfLatest=900 — got ' + JSON.stringify(stats));
  const st = db.resumeState();
  assert(st.maxGeneration === 1 && st.bestWeights && typeof st.rngState === 'number' && st.trackHash === DEFAULT_TRACK_HASH, 'resumeState() returns the documented shape, got maxGeneration=' + st.maxGeneration);
  // Empty database — every read path keeps its documented empty values:
  const empty = await freshDb();
  assert(empty.getDriverResults(1).length === 0, 'getDriverResults on an empty DB returns []');
  assert(empty.listGenerations().length === 0, 'listGenerations on an empty DB returns []');
  assert(empty.getGeneration(1) === null, 'getGeneration on an empty DB returns null');
  const emptyStats = empty.stats();
  assert(emptyStats.generationCount === 0 && emptyStats.latestGeneration === 0 && emptyStats.bestFitnessEver === null && emptyStats.avgOfLatest === null, 'stats on an empty DB reports 0/0/null/null — got ' + JSON.stringify(emptyStats));
  const emptyResume = empty.resumeState();
  assert(emptyResume.maxGeneration === 0 && emptyResume.bestWeights === null && emptyResume.rngState === null && emptyResume.trackHash === null, 'resumeState on an empty DB returns the documented zeros — got ' + JSON.stringify(emptyResume));
  empty.db.close();
  db.db.close();
});

// =============================================================================
// M4-fix leg 34 — restoreFromLastKnownGood (the battery-gated recovery core,
// node-testable via the opts.read stub — no IDB involved). Arms:
//   (a) absent/empty/garbage reader payloads ⇒ {ok:false, reason:'no-cache'}
//   (b) valid battery-passed bytes built from the 4242 reference lineage
//       (2 generations) ⇒ {ok:true} · summary.generation===2 · the restored
//       scratch Db's resumeState() is byte-equal to the source (rng uint32)
//   (c) poisoned (trigger-bearing) bytes ⇒ {ok:false, reason:'validation'},
//       NO db key — the working DB is never adopted/touched
//   (d) truncated bytes ⇒ {ok:false, reason:'validation'}, NO db key
// =============================================================================
await leg('leg 34 — restoreFromLastKnownGood: no-cache / 4242-lineage adoption / trigger validation / truncation (node)', async () => {
  const { Db } = requireDb();
  // (a) reader yields null / zero-length / a non-Uint8Array — all ⇒ no-cache
  const resNull = await Db.restoreFromLastKnownGood({ read: async () => null });
  assert(resNull && resNull.ok === false && resNull.reason === 'no-cache', '(a) null bytes ⇒ {ok:false, reason:"no-cache"}, got ' + JSON.stringify(resNull));
  const resEmpty = await Db.restoreFromLastKnownGood({ read: async () => new Uint8Array(0) });
  assert(resEmpty && resEmpty.ok === false && resEmpty.reason === 'no-cache', '(a) zero-length bytes ⇒ {ok:false, reason:"no-cache"}, got ' + JSON.stringify(resEmpty));
  const resGarbage = await Db.restoreFromLastKnownGood({ read: async () => ({}) });
  assert(resGarbage && resGarbage.ok === false && resGarbage.reason === 'no-cache', '(a) a non-Uint8Array reader payload ⇒ {ok:false, reason:"no-cache"} (the pre-check fires before the battery), got ' + JSON.stringify(resGarbage));

  // (b) valid battery-passed bytes — the 4242 reference lineage, 2 generations
  const makeTrainerLogger = (db) => ({
    logGeneration(payload) {
      const results = payload.results.map((r) => ({
        distance: r.distance,
        topSpeed: r.topSpeed,
        timeAlive: r.timeAlive,
        crashed: r.crashed ? 1 : 0,
        stuck: r.stuck ? 1 : 0,
      }));
      let worst = Infinity, distSum = 0, timeSum = 0;
      for (const r of results) {
        if (r.distance < worst) worst = r.distance;
        distSum += r.distance;
        timeSum += r.timeAlive;
      }
      db.logGeneration({
        generation: payload.generation,
        driversEvaluated: results.length,
        bestFitness: results[payload.bestIndex].distance,
        avgFitness: distSum / results.length,
        worstFitness: worst,
        bestDriverIndex: payload.bestIndex,
        mutationRate: MUTATION_RATE,
        mutationScale: MUTATION_SCALE,
        trackJson: DEFAULT_TRACK_JSON,
        trackHash: DEFAULT_TRACK_HASH,
        simSeconds: timeSum,
        rngState: payload.rngState,
        results,
        bestWeightsJson: payload.bestWeights,
      });
    },
  });
  const prefixDb = await freshDb();
  prefixDb.ensureLineage(DEFAULT_TRACK_HASH, 4242, DEFAULT_TRACK_JSON);
  const prefixTrainer = new Trainer(defaultTrack, {
    lineage: { seed: 4242, rotationSeed: 4242, generatorVersion: 1 },
    logger: makeTrainerLogger(prefixDb),
    onBanner() {},
  });
  prefixTrainer.start(2);
  await waitFor(() => prefixTrainer.state === 'done', 'leg 34 prefix run reaches done', 180000);
  assert(prefixDb.listGenerations().join() === '1,2', 'leg 34 prefix DB holds gens 1,2, got ' + prefixDb.listGenerations());
  const twoGenBytes = prefixDb.exportBytes();
  const sourceRng = prefixDb.resumeState().rngState;
  prefixDb.db.close();

  const resValid = await Db.restoreFromLastKnownGood({ read: async () => twoGenBytes });
  assert(resValid && resValid.ok === true, '(b) battery-passed bytes restore ⇒ {ok:true}, got ' + JSON.stringify(resValid && { ok: resValid.ok, reason: resValid.reason, detail: resValid.detail }));
  assert(resValid.summary && resValid.summary.generation === 2, '(b) summary.generation === 2, got ' + JSON.stringify(resValid.summary));
  assert(resValid.summary.trackHash === DEFAULT_TRACK_HASH, '(b) summary.trackHash is the committed pin, got ' + JSON.stringify(resValid.summary.trackHash));
  assert(typeof resValid.summary.rngState === 'number' && Number.isInteger(resValid.summary.rngState), '(b) summary.rngState is a uint32 number, got ' + JSON.stringify(resValid.summary.rngState));
  const restoredState = resValid.db.resumeState();
  assert(restoredState.rngState === sourceRng, '(b) restored resumeState rngState equals the source (got ' + restoredState.rngState + ' vs ' + sourceRng + ')');
  assert(restoredState.maxGeneration === 2, '(b) restored resumeState maxGeneration === 2, got ' + restoredState.maxGeneration);
  assert(resValid.db.listGenerations().join() === '1,2', '(b) restored DB holds gens 1,2, got ' + resValid.db.listGenerations());
  resValid.db.db.close();

  // (c) poisoned (trigger-bearing) bytes ⇒ refused, NO db key, zero adoption
  const SQL = globalThis.SQL;
  const raw = new SQL.Database(twoGenBytes);
  raw.run('CREATE TRIGGER t BEFORE INSERT ON generations BEGIN UPDATE driver_results SET distance = -distance; END');
  const poisoned = raw.export();
  raw.close();
  const resPoison = await Db.restoreFromLastKnownGood({ read: async () => poisoned });
  assert(resPoison && resPoison.ok === false && resPoison.reason === 'validation', '(c) crafted trigger bytes ⇒ {ok:false, reason:"validation"}, got ' + JSON.stringify(resPoison && { ok: resPoison.ok, reason: resPoison.reason, detail: resPoison.detail }));
  assert(typeof resPoison.detail === 'string' && resPoison.detail.length > 0, '(c) refusal carries a detail message, got ' + JSON.stringify(resPoison.detail));
  assert(!('db' in resPoison), '(c) refused restore carries NO db key — the working DB is never adopted/touched');

  // (d) truncated bytes ⇒ refused, NO db key
  const truncated = twoGenBytes.subarray(0, Math.floor(twoGenBytes.length / 2));
  const resTrunc = await Db.restoreFromLastKnownGood({ read: async () => truncated });
  assert(resTrunc && resTrunc.ok === false && resTrunc.reason === 'validation', '(d) truncated bytes ⇒ {ok:false, reason:"validation"}, got ' + JSON.stringify(resTrunc && { ok: resTrunc.ok, reason: resTrunc.reason, detail: resTrunc.detail }));
  assert(!('db' in resTrunc), '(d) refused restore carries NO db key — the working DB is never adopted/touched');
});

// =============================================================================
// FINAL-BATCH-2 legs 35–41 — the adversarial-tester's REJECTED battery arms,
// now enforced: (35) a hand-crafted FK-less file (exact column NAME sets, no
// foreign keys) is REFUSED by the ADOPTION BATTERY and the refusal names the
// missing driver_results FK (F-1); (36) +Inf best_fitness is refused (F-8);
// (37) best_driver_index=99 refused (F-8); (38) crashed=7 refused (F-8);
// (39) the 3-point bomb track (points at 1e6-coordinate scale) is refused at
// Track.fromJSON by the sample budget (F-2); (40) the LSG no-regress rule
// matrix (F-3) — _lsgShouldOverwrite true/false incl. missing-stored and
// snake_case-tolerant arms; (41) the LSG coherence tag is computed FROM THE
// SNAPSHOT BYTES (fresh-initialized scratch over the exported buffer) and
// reads generation 2 after logging gens 1+2 (F-4).
// Every probe file is a scratch SQL Database built from DDL that reproduces
// the hand-crafted-file shape (no CHECK constraints; the FK arm is the ONLY
// thing WITH_FK passes that ATTACKER_DDL misses). Rows are inserted with raw
// DML via bound parameters — exactly the attacker's write path.
// =============================================================================
const ATTACKER_DDL_NO_FK = [
  'CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);',
  'CREATE TABLE generations(generation INTEGER PRIMARY KEY, created_at TEXT NOT NULL DEFAULT (datetime(\'now\')), drivers_evaluated INTEGER NOT NULL, best_fitness REAL NOT NULL, avg_fitness REAL NOT NULL, worst_fitness REAL NOT NULL, best_driver_index INTEGER NOT NULL, mutation_rate REAL NOT NULL, mutation_scale REAL NOT NULL, track_json TEXT NOT NULL, track_hash TEXT NOT NULL, sim_seconds REAL NOT NULL, rng_state INTEGER NOT NULL);',
  'CREATE TABLE driver_results(generation INTEGER NOT NULL, driver_index INTEGER NOT NULL, distance REAL NOT NULL, top_speed REAL NOT NULL, time_alive REAL NOT NULL, crashed INTEGER NOT NULL, stuck INTEGER NOT NULL, PRIMARY KEY (generation, driver_index));',
  'CREATE TABLE best_weights(generation INTEGER PRIMARY KEY, architecture TEXT NOT NULL, weights_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime(\'now\')));',
];
// Same shape but WITH the declared FKs (no CHECKs) — isolates the value-domain
// arm from the FK arm (legs 36–38 must pass the FK arm and fail value-domain).
const ATTACKER_DDL_WITH_FK = [
  'CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);',
  'CREATE TABLE generations(generation INTEGER PRIMARY KEY, created_at TEXT NOT NULL DEFAULT (datetime(\'now\')), drivers_evaluated INTEGER NOT NULL, best_fitness REAL NOT NULL, avg_fitness REAL NOT NULL, worst_fitness REAL NOT NULL, best_driver_index INTEGER NOT NULL, mutation_rate REAL NOT NULL, mutation_scale REAL NOT NULL, track_json TEXT NOT NULL, track_hash TEXT NOT NULL, sim_seconds REAL NOT NULL, rng_state INTEGER NOT NULL);',
  'CREATE TABLE driver_results(generation INTEGER NOT NULL REFERENCES generations(generation) ON DELETE CASCADE, driver_index INTEGER NOT NULL, distance REAL NOT NULL, top_speed REAL NOT NULL, time_alive REAL NOT NULL, crashed INTEGER NOT NULL, stuck INTEGER NOT NULL, PRIMARY KEY (generation, driver_index));',
  'CREATE TABLE best_weights(generation INTEGER PRIMARY KEY REFERENCES generations(generation) ON DELETE CASCADE, architecture TEXT NOT NULL, weights_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime(\'now\')));',
];

function buildSemanticProbe({ noFk, generation, results, orphan }) {
  const SQL = globalThis.SQL;
  const raw = new SQL.Database();
  const ddl = noFk ? ATTACKER_DDL_NO_FK : ATTACKER_DDL_WITH_FK;
  for (const stmt of ddl) raw.run(stmt);
  raw.run('PRAGMA user_version = 1'); // fitness-v2: these probe files must pass battery step 3b ({1,2,3}) so the FK/value-domain arms are what refuse them
  raw.run('INSERT INTO meta(key, value) VALUES (?, ?)', ['schema_version', '1']);
  raw.run('INSERT INTO meta(key, value) VALUES (?, ?)', ['architecture', '7-16-16-2']);
  raw.run('INSERT INTO meta(key, value) VALUES (?, ?)', ['app_version', '1.0.0']);
  raw.run("INSERT INTO meta(key, value) VALUES (?, datetime('now'))", ['created_at']);
  raw.run('INSERT INTO meta(key, value) VALUES (?, ?)', ['track_hash', DEFAULT_TRACK_HASH]);
  raw.run('INSERT INTO meta(key, value) VALUES (?, ?)', ['rng_state', '123456789']);
  const gen = Object.assign({
    generation: 1,
    driversEvaluated: POPULATION,
    bestFitness: 1000,
    avgFitness: 900,
    worstFitness: 800,
    bestDriverIndex: 0,
    mutationRate: MUTATION_RATE,
    mutationScale: MUTATION_SCALE,
    trackJson: DEFAULT_TRACK_JSON,
    trackHash: DEFAULT_TRACK_HASH,
    simSeconds: 105,
    rngState: 1001,
  }, generation || {});
  raw.run('INSERT INTO generations (generation, drivers_evaluated, best_fitness, avg_fitness, worst_fitness, best_driver_index, mutation_rate, mutation_scale, track_json, track_hash, sim_seconds, rng_state) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
    [gen.generation, gen.driversEvaluated, gen.bestFitness, gen.avgFitness, gen.worstFitness, gen.bestDriverIndex, gen.mutationRate, gen.mutationScale, gen.trackJson, gen.trackHash, gen.simSeconds, gen.rngState]);
  const base = makeValidResults();
  for (let i = 0; i < base.length; i++) {
    const rr = Object.assign({}, base[i], results || {});
    raw.run('INSERT INTO driver_results (generation, driver_index, distance, top_speed, time_alive, crashed, stuck) VALUES (?,?,?,?,?,?,?)',
      [gen.generation, i, rr.distance, rr.topSpeed, rr.timeAlive, rr.crashed, rr.stuck]);
  }
  if (orphan) {
    raw.run('INSERT INTO driver_results (generation, driver_index, distance, top_speed, time_alive, crashed, stuck) VALUES (99, 0, 1, 1, 1, 0, 0)');
  }
  raw.run('INSERT INTO best_weights (generation, architecture, weights_json) VALUES (?,?,?)', [gen.generation, '7-16-16-2', makeValidWeightsString()]);
  const bytes = raw.export();
  raw.close();
  return bytes;
}

// FIT-V6 (legs 35–38): the semantic probes below are pre-v4 (uv1) fixtures.
// Under the shipped contract pre-v4 bytes are typed-refused at step 3b BEFORE
// the FK / value-domain arms run. The arms themselves remain enforced at uv4 by
// legs 54/62/71/72 (which craft the same defect classes on uv4 bytes). Each leg
// below is retained and upgraded to pin the typed wall-model refusal contract.
await leg('leg 35 — FINAL-BATCH-2 F-1: pre-v4 FK-less crafted file ⇒ typed wall-model refusal (FK/value arms enforced at uv4 by legs 54/62)', () => {
  const { Db } = requireDb();
  assertWallModelRefusal(Db, buildSemanticProbe({ noFk: true }), 'leg 35');
});

await leg('leg 36 — FINAL-BATCH-2 F-8: pre-v4 +Inf best_fitness file ⇒ typed wall-model refusal (value-domain arm at uv4 by leg 54)', () => {
  const { Db } = requireDb();
  assertWallModelRefusal(Db, buildSemanticProbe({ noFk: false, generation: { bestFitness: Infinity } }), 'leg 36');
});

await leg('leg 37 — FINAL-BATCH-2 F-8: pre-v4 best_driver_index=99 file ⇒ typed wall-model refusal (value-domain arm at uv4 by leg 54)', () => {
  const { Db } = requireDb();
  assertWallModelRefusal(Db, buildSemanticProbe({ noFk: false, generation: { bestDriverIndex: 99 } }), 'leg 37');
});

await leg('leg 38 — FINAL-BATCH-2 F-8: pre-v4 crashed=7 file ⇒ typed wall-model refusal (value-domain arm at uv4 by leg 54)', () => {
  const { Db } = requireDb();
  assertWallModelRefusal(Db, buildSemanticProbe({ noFk: false, results: { crashed: 7 } }), 'leg 38');
});

await leg('leg 39 — FINAL-BATCH-2 F-2: the 3-point bomb track ⇒ Track.fromJSON throws (sample budget)', () => {
  const bomb = JSON.stringify({ points: [{ x: 1e6, y: 1e6 }, { x: -1e6, y: 1e6 }, { x: 0, y: -1e6 }], halfWidth: 70 });
  assertValidationError(() => Track.fromJSON(bomb), 'leg 39 — bomb track refused at fromJSON');
  let msg = '';
  try { Track.fromJSON(bomb); } catch (err) { msg = String(err && err.message); }
  assert(/sample budget/i.test(msg), 'leg 39 — refusal names the sample budget, got: ' + msg);
  // Non-vacuous companion: the default circuit still builds fine.
  const okTrack = new Track(DEFAULT_CONTROL_POINTS, 70);
  assert(okTrack.samples.length > 0, 'leg 39 — default circuit still builds (sample budget does not catch sane tracks)');
});

await leg('leg 40 — FINAL-BATCH-2 F-3: _lsgShouldOverwrite matrix (incl. missing-stored case)', () => {
  const { Db } = requireDb();
  const T = (maxGeneration, trackHash) => ({ maxGeneration, trackHash });
  assert(Db._lsgShouldOverwrite(null, T(2, 'h')) === true, 'missing stored tag ⇒ overwrite');
  assert(Db._lsgShouldOverwrite(undefined, T(0, 'h')) === true, 'undefined stored tag ⇒ overwrite');
  assert(Db._lsgShouldOverwrite(T(2, 'h'), T(2, 'h')) === false, 'equal gen + same track ⇒ cache AHEAD — do NOT overwrite');
  assert(Db._lsgShouldOverwrite(T(5, 'h'), T(3, 'h')) === false, 'stored strictly ahead ⇒ do NOT overwrite');
  assert(Db._lsgShouldOverwrite(T(3, 'h'), T(5, 'h')) === true, 'incoming strictly newer ⇒ overwrite');
  assert(Db._lsgShouldOverwrite(T(2, 'h'), T(2, 'other')) === false, 'same gen, differing hash ⇒ keep the cache (equal generation) — the M6c banner owns the divergence notice; db.js returns the boolean (M6b re-spec)');
  assert(Db._lsgShouldOverwrite(T(2, 'h'), T(1, 'other')) === false, 'older gen, other track ⇒ stored is ahead ⇒ do NOT overwrite');
  assert(Db._lsgShouldOverwrite({ maxGeneration: 'bogus', trackHash: 'h' }, T(2, 'h')) === true, 'unreadable stored maxGeneration ⇒ overwrite');
  assert(Db._lsgShouldOverwrite({ maxGeneration: 2 }, T(2, 'h')) === false, 'equal generation, stored tag without trackHash ⇒ keep the cache (equal generation)');
  assert(Db._lsgShouldOverwrite({ maxGeneration: 2, track_hash: 'h' }, T(2, 'h')) === false, 'legacy snake_case stored track_hash equal ⇒ do NOT overwrite');
});

await leg('leg 41 — FINAL-BATCH-2 F-4: LSG coherence tag computed FROM THE SNAPSHOT BYTES reads generation 2 (gens 1+2 logged)', async () => {
  const { Db } = requireDb();
  const db = await freshDb();
  ensureLineageFor(db);
  logGen(db, 1);
  logGen(db, 2);
  const bytes = db.exportBytes(); // the exact buffer persist() verifies/writes/caches
  const scratch = new Db();
  await scratch.init(bytes); // fresh-initialized scratch — the coherence tag must describe the BYTES
  const tag = scratch._coherenceTag(bytes);
  assert(tag && tag.maxGeneration === 2, 'coherence from the snapshotted bytes ⇒ maxGeneration 2, got ' + JSON.stringify(tag));
  assert(tag.trackHash === DEFAULT_TRACK_HASH, 'coherence trackHash round-trips, got ' + JSON.stringify(tag.trackHash));
  assert(Number.isInteger(tag.rngState) && tag.rngState >= 0 && tag.rngState <= 4294967295, 'coherence rngState is a uint32, got ' + JSON.stringify(tag.rngState));
  scratch.db.close();
  db.db.close();
});

// FINAL-BATCH-3B leg 42 — boot-fallback batteries (F-6) through the REAL
// js/db-session.js restoreIfPossible() with an in-memory IndexedDB stand-in
// installed on globalThis (Map-backed kv stores — node has no indexedDB; the
// browser path is Lead-verified). Mimics BOTH databases the boot path opens:
// 'car_racer-sessions' v2 ('handles'/'meta' stores) and db.js's 'car_racer'
// v1 ('kv'). Values pass by reference (no structured clone) — Uint8Array
// rides through normalizeClonedBytes exactly as in the browser after IDB's
// own clone. Scenarios: (a) battery-refused FILE handle + valid LSG ⇒ the
// cache is adopted, slot + generation named, file NEVER adopted, handle
// cleared; (b) BOTH refuse ⇒ fresh in-memory session, dual-reason message
// carrying BOTH refusal strings, poisoned LSG NEVER adopted + the slot is
// CLEARED by the both-refuse branch (FINAL-CLEANUP-1), handle cleared.
// =============================================================================
function makeFakeIdb() {
  const databases = new Map(); // name -> { version, stores: Map(storeName, Map(key, value)) }
  const makeReq = () => ({ result: undefined, error: null, onsuccess: null, onerror: null, onupgradeneeded: null, onblocked: null });
  function deliver(req, fn) {
    // IDB delivers on a later task; Promise microtasks keep handler
    // registration (synchronous, right after open()) ahead of delivery.
    Promise.resolve().then(() => {
      try { fn(); } catch (err) {
        req.error = err;
        if (typeof req.onerror === 'function') req.onerror();
      }
    });
  }
  function makeTx(rec, storeName) {
    const tx = { oncomplete: null, onerror: null, onabort: null, error: null };
    let pending = 0;
    let completed = false;
    function settle() {
      pending -= 1;
      if (pending === 0 && !completed) {
        completed = true;
        Promise.resolve().then(() => { if (typeof tx.oncomplete === 'function') tx.oncomplete(); });
      }
    }
    tx.objectStore = function (name) {
      const map = rec.stores.get(name);
      if (!map) throw new Error("IDB object store '" + name + "' does not exist");
      return {
        get(key) {
          const req = makeReq();
          deliver(req, () => {
            req.result = map.has(key) ? map.get(key) : undefined;
            if (req.onsuccess) req.onsuccess();
          });
          return req;
        },
        put(value, key) {
          const req = makeReq();
          pending += 1;
          deliver(req, () => {
            map.set(key, value);
            if (req.onsuccess) req.onsuccess();
            settle();
          });
          return req;
        },
        delete(key) {
          const req = makeReq();
          pending += 1;
          deliver(req, () => {
            map.delete(key);
            if (req.onsuccess) req.onsuccess();
            settle();
          });
          return req;
        },
      };
    };
    return tx;
  }
  function makeIdb(rec) {
    return {
      objectStoreNames: { contains: (n) => rec.stores.has(n) },
      createObjectStore(name) { rec.stores.set(name, new Map()); return {}; },
      deleteObjectStore(name) { rec.stores.delete(name); },
      transaction(storeName) {
        if (!rec.stores.has(storeName)) throw new Error("IDB transaction: store '" + storeName + "' not found");
        return makeTx(rec, storeName);
      },
      close() {},
    };
  }
  const indexedDB = {
    open(name, version) {
      const req = makeReq();
      Promise.resolve().then(() => {
        let rec = databases.get(name);
        if (!rec) {
          rec = { version: 0, stores: new Map() };
          databases.set(name, rec);
        }
        const idb = makeIdb(rec);
        req.result = idb;
        if (rec.version < version) {
          rec.version = version;
          if (typeof req.onupgradeneeded === 'function') req.onupgradeneeded();
        }
        if (typeof req.onsuccess === 'function') req.onsuccess();
      });
      return req;
    },
  };
  return { indexedDB, databases };
}

await leg('leg 42 — FINAL-BATCH-3B: boot fallback — LSG-valid + refusing file ⇒ adopts cache, names slot, file NEVER adopted; both-refuse ⇒ dual-reason message', async () => {
  const { Db } = requireDb();
  const { DbSession } = require('../js/db-session.js');

  // (seeded) a valid 2-generation lineage — the same helper family legs 29/34 use
  const seed = await freshDb();
  ensureLineageFor(seed);
  logGen(seed, 1);
  logGen(seed, 2);
  const goodTwoGen = seed.exportBytes();
  seed.db.close();

  // The refusing FILE: trigger-bearing bytes crafted from the SAME lineage
  // (the leg-30 attack shape) — the ADOPTION BATTERY refuses it at openExisting
  const SQL = globalThis.SQL;
  const raw = new SQL.Database(goodTwoGen);
  raw.run('CREATE TRIGGER t BEFORE INSERT ON generations BEGIN UPDATE driver_results SET distance = -distance; END');
  const poisonedFileBytes = raw.export();
  raw.close();

  // The refusing CACHE for scenario (b): FIT-V6 pre-v4 bytes (the leg-36 probe)
  // — their typed refusal names 'wall model', distinct from the file's
  // 'trigger', so the dual-reason message is verifiable by content, not shape.
  const poisonedLsgBytes = buildSemanticProbe({ noFk: false, generation: { bestFitness: Infinity } });

  const makeRefusingHandle = () => ({
    name: 'refusing-battery.sqlite',
    queryPermission: async () => 'granted',
    getFile: async () => ({
      size: poisonedFileBytes.length,
      arrayBuffer: async () => poisonedFileBytes.slice().buffer,
    }),
  });

  const fakeIdb = makeFakeIdb();
  const savedIdb = Object.getOwnPropertyDescriptor(globalThis, 'indexedDB');
  const savedDbGlobal = Object.getOwnPropertyDescriptor(globalThis, 'Db');
  Object.defineProperty(globalThis, 'indexedDB', { value: fakeIdb.indexedDB, configurable: true, writable: true });
  // Read the handles store straight off the fake — DbSession does not export
  // loadHandle, and this file must not touch db-session.js beyond the N-3 fix.
  const storedHandle = () => {
    const rec = fakeIdb.databases.get('car_racer-sessions');
    const store = rec && rec.stores.get('handles');
    return store ? (store.has('current') ? store.get('current') : null) : null;
  };
  globalThis.Db = Db; // db-session.js resolves the recovery core off globalThis.Db
  try {
    const probe = new Db(); // scratch writer for the 'car_racer' kv slot — REAL code path behind the fake IDB
    // ---- (a) battery-refused file + valid LSG ⇒ cache adopted; file NEVER adopted
    await probe._idbPut('car_racer.lastKnownGood', {
      bytes: goodTwoGen,
      tag: { maxGeneration: 2, trackHash: DEFAULT_TRACK_HASH, rngState: 1002 },
      savedAt: new Date().toISOString(),
    });
    const saveA = await DbSession.saveHandle(makeRefusingHandle());
    assert(saveA && saveA.ok === true, '(a) the refusing handle was seeded into the fake handles store, got ' + JSON.stringify(saveA));
    const freshA = new Db(); // deliberately NOT initialized
    const resA = await DbSession.restoreIfPossible(freshA);
    assert(resA && resA.state === 'cache', '(a) refusing file + valid LSG ⇒ state "cache", got ' + JSON.stringify(resA && resA.state));
    assert(resA.db && resA.db !== freshA, '(a) the adopted database is the restore-built scratch instance, not the passed-in fresh Db');
    assert(freshA.db === null, '(a) the passed-in fresh Db was NEVER initialized — the file bytes were never adopted');
    assert(resA.generation === 2, '(a) restored generation 2, got ' + JSON.stringify(resA.generation));
    assert(resA.db.resumeState().maxGeneration === 2, '(a) the adopted cache holds generation 2, got ' + JSON.stringify(resA.db.resumeState()));
    assert(/lastKnownGood/.test(resA.message), '(a) the message names the winning slot, got: ' + resA.message);
    assert(/generation 2/.test(resA.message), '(a) the message names the generation, got: ' + resA.message);
    assert(/trigger/i.test(resA.message), '(a) the message carries the FILE refusal reason, got: ' + resA.message);
    assert(storedHandle() === null, '(a) the refusing handle was cleared (no retry loop on next boot)');
    resA.db.db.close();

    // ---- (b) BOTH refuse ⇒ fresh in-memory session; dual-reason message
    await probe._idbPut('car_racer.lastKnownGood', {
      bytes: poisonedLsgBytes,
      tag: { maxGeneration: 1, trackHash: DEFAULT_TRACK_HASH, rngState: 7 },
      savedAt: new Date().toISOString(),
    }); // overwrite LSG with battery-refused bytes (the leg-36 shape)
    const saveB = await DbSession.saveHandle(makeRefusingHandle());
    assert(saveB && saveB.ok === true, '(b) the refusing handle was re-seeded, got ' + JSON.stringify(saveB));
    const freshB = new Db();
    const resB = await DbSession.restoreIfPossible(freshB);
    assert(resB && resB.state === 'memory', '(b) file refused + cache refused ⇒ state "memory", got ' + JSON.stringify(resB && resB.state));
    assert(!('db' in resB), '(b) no db key — nothing was adopted');
    assert(freshB.db !== null, '(b) a fresh in-memory database was initialized for the session');
    assert(/trigger/i.test(resB.message), '(b) the message carries the FILE refusal reason (trigger), got: ' + resB.message);
    assert(/wall model/i.test(resB.message), '(b) the message carries the CACHE refusal reason (FIT-V6 wall model), got: ' + resB.message);
    // FINAL-CLEANUP-1: the handle-refusing both-refuse path now clears the
    // poisoned LSG slot too (mirrors path 2's validation-failure arm) — the
    // slot held a battery-refused copy; nothing is ever adopted.
    assert(freshB.listGenerations().length === 0, '(b) the poisoned LSG was NOT adopted — zero generations in the session database');
    const lsgAfterB = (() => {
      const rec = fakeIdb.databases.get('car_racer');
      const store = rec && rec.stores.get('kv');
      return store && store.has('car_racer.lastKnownGood') ? store.get('car_racer.lastKnownGood') : null;
    })();
    assert(lsgAfterB === null, '(b) the poisoned LSG slot was cleared by the both-refuse branch, got: ' + JSON.stringify(lsgAfterB && lsgAfterB.tag));
    assert(storedHandle() === null, '(b) the refusing handle was cleared');
    freshB.db.close();
  } finally {
    if (savedIdb) Object.defineProperty(globalThis, 'indexedDB', savedIdb);
    else { try { delete globalThis.indexedDB; } catch (e) { /* best effort */ } }
    if (savedDbGlobal) Object.defineProperty(globalThis, 'Db', savedDbGlobal);
    else { try { delete globalThis.Db; } catch (e) { /* best effort */ } }
  }
});

// =============================================================================
// FINAL-BATCH-3B legs 43–44 — the LSG no-regress rule END-TO-END through the
// REAL persist() path (leg 29's node harness: injected file handle / stub
// writable / hooked banners; here the IDB layer is the leg-29-style per-instance
// _idbGet/_idbPut/_idbDelete override over an in-memory map). (43) file mode:
// a STALE stored tag on the same lineage is OVERWRITTEN by a strictly-newer
// persist (no cache-ahead banner, file proceeds, LSG retains the NEW tag);
// then an exactly-equal-but-divergent tag re-persist SKIPS the LSG write —
// the cache-ahead banner fires, NO file-path banner, file still written, LSG
// unchanged. (44) IDB-fallback mode (no handle): the exactly-equal divergent
// case from the leg-40 matrix fires cache-ahead and getLSGTag() is UNCHANGED;
// the strictly-newer case fires NO banner and overwrites the LSG.
// =============================================================================
await leg('leg 43 — FINAL-BATCH-3B: LSG no-regress end-to-end through persist() (file mode) — stale tag overwritten strictly-newer; equal-divergent re-persist skips LSG + cache-ahead banner', async () => {
  const { Db } = requireDb();
  const seed = await freshDb();
  ensureLineageFor(seed);
  logGen(seed, 1);
  const gen1Bytes = seed.exportBytes();
  seed.db.close();
  // Adopt the gen-1 bytes (leg-29 harness) and wire a stub file holding them
  const db = new Db();
  await db.init(gen1Bytes);
  let fileState = { bytes: gen1Bytes.slice() };
  const banners = [];
  const idbMap = new Map();
  const idbCalls = [];
  db._idbGet = async (key) => ({ ok: true, value: idbMap.has(key) ? idbMap.get(key) : null });
  db._idbPut = async (key, value) => {
    idbCalls.push({ key: key, tag: value && value.tag ? Object.assign({}, value.tag) : null });
    idbMap.set(key, value);
    return { ok: true };
  };
  db._idbDelete = async (key) => { idbMap.delete(key); return { ok: true }; };
  db.setPersistenceHooks({ onBanner(b) { banners.push(b); }, persistSkipped: () => false });
  const writes = [];
  db._createWritable = async () => {
    const w = {
      written: null,
      closed: false,
      async write(b) { w.written = new Uint8Array(b); },
      async close() { fileState.bytes = w.written; w.closed = true; },
      async abort() { w.aborted = true; },
    };
    writes.push(w);
    return w;
  };
  db._fileHandle = {
    async getFile() { return { size: fileState.bytes.length, arrayBuffer: async () => fileState.bytes.slice().buffer }; },
  };
  const getLSGTag = () => {
    const rec = idbMap.get('car_racer.lastKnownGood');
    return rec && rec.tag ? rec.tag : null;
  };
  // STALE tag in the LSG slot — same lineage, gen 1 (a pre-persist remnant)
  await db._idbPut('car_racer.lastKnownGood', {
    bytes: gen1Bytes,
    tag: { maxGeneration: 1, trackHash: DEFAULT_TRACK_HASH, rngState: 111 },
    savedAt: new Date().toISOString(),
  });
  logGen(db, 2); // strictly newer than the stale tag — the no-regress rule must NOT fire
  const r1 = await db.persist();
  assert(r1 && r1.ok === true, '43(1) — persist at gen 2 succeeds, got ' + JSON.stringify(r1));
  assert(r1.tag && r1.tag.maxGeneration === 2, '43(1) — the persist return tag names gen 2, got ' + JSON.stringify(r1.tag));
  assert(writes.length === 1 && writes[0].closed, '43(1) — the verified bytes were written and closed');
  assert(!banners.some((b) => b.type === 'cache-ahead'), '43(1) — NO cache-ahead banner on the strictly-newer persist, got ' + JSON.stringify(banners));
  assert(!banners.some((b) => b.type === 'reconcile-required' || b.type === 'db-not-persisting' || b.type === 'db-verify-failed'), '43(1) — NO file-path banner fired, got ' + JSON.stringify(banners));
  assert(getLSGTag() && getLSGTag().maxGeneration === 2 && getLSGTag().rngState !== 111, '43(1) — the LSG retains the NEW (gen-2) copy, got ' + JSON.stringify(getLSGTag()));

  // Exactly-equal-but-divergent re-persist (the leg-40 equal case, through the
  // REAL persist): stored tag names gen 2 on this lineage but DIVERGES
  // (rngState 999) — the LSG write must SKIP, the cache-ahead banner fires,
  // and the file path still completes end-to-end.
  idbMap.set('car_racer.lastKnownGood', {
    bytes: gen1Bytes, // stale bytes — only the tag is compared by the no-regress rule
    tag: { maxGeneration: 2, trackHash: DEFAULT_TRACK_HASH, rngState: 999 },
    savedAt: new Date().toISOString(),
  });
  const bannersBefore = banners.length;
  const writesBefore = writes.length;
  const putsBefore = idbCalls.filter((c) => c.key === 'car_racer.lastKnownGood').length;
  const r2 = await db.persist();
  assert(r2 && r2.ok === true, '43(2) — the equal-generation persist still succeeds (the skip never blocks the file), got ' + JSON.stringify(r2));
  assert(writes.length === writesBefore + 1, '43(2) — the file was written anyway (the skip is LSG-only)');
  assert(banners.slice(bannersBefore).some((b) => b.type === 'cache-ahead'), '43(2) — the cache-ahead banner FIRED for the exactly-equal divergent tag, got ' + JSON.stringify(banners.slice(bannersBefore)));
  assert(!banners.slice(bannersBefore).some((b) => b.type === 'reconcile-required' || b.type === 'db-not-persisting'), '43(2) — NO file-path banner fired on the skip, got ' + JSON.stringify(banners.slice(bannersBefore)));
  assert(idbCalls.filter((c) => c.key === 'car_racer.lastKnownGood').length === putsBefore, '43(2) — the LSG write was SKIPPED (no overwrite)');
  assert(getLSGTag() && getLSGTag().rngState === 999, '43(2) — the LSG retains the stale/equal copy (reports skip), got ' + JSON.stringify(getLSGTag()));
  db.db.close();
});

await leg('leg 44 — FINAL-BATCH-3B: persist() LSG skip + cache-ahead banner (IDB-fallback) — exactly-equal divergent ⇒ getLSGTag() unchanged + banner; strictly-newer ⇒ NO banner + overwrite', async () => {
  const { Db } = requireDb();
  const db = await freshDb();
  ensureLineageFor(db);
  logGen(db, 1);
  logGen(db, 2);
  const banners = [];
  const idbMap = new Map();
  db._idbGet = async (key) => ({ ok: true, value: idbMap.has(key) ? idbMap.get(key) : null });
  db._idbPut = async (key, value) => { idbMap.set(key, value); return { ok: true }; };
  db._idbDelete = async (key) => { idbMap.delete(key); return { ok: true }; };
  db.setPersistenceHooks({ onBanner(b) { banners.push(b); }, persistSkipped: () => false });
  const getLSGTag = () => {
    const rec = idbMap.get('car_racer.lastKnownGood');
    return rec && rec.tag ? rec.tag : null;
  };
  // NO file handle — IDB-fallback mode (§PERSIST step 8): the LSG slot IS the cache
  assert(!db._fileHandle, '44 — harness sanity: no file handle (IDB-fallback mode)');
  // The exactly-equal DIVERGENT case (the leg-40 equal-gen/same-track row): the
  // stored tag names gen 2 on this lineage but DIVERGES (rngState 424242 vs the
  // committed tag) — the no-regress rule keeps the slot and fires cache-ahead.
  idbMap.set('car_racer.lastKnownGood', {
    bytes: db.exportBytes(),
    tag: { maxGeneration: 2, trackHash: DEFAULT_TRACK_HASH, rngState: 424242 },
    savedAt: new Date().toISOString(),
  });
  const r1 = await db.persist();
  assert(r1 && r1.ok === true, '44(1) — IDB-fallback persist succeeds, got ' + JSON.stringify(r1));
  assert(r1.tag && r1.tag.maxGeneration === 2, '44(1) — the persist return tag names gen 2, got ' + JSON.stringify(r1.tag));
  const ahead = banners.filter((b) => b.type === 'cache-ahead');
  assert(ahead.length === 1, '44(1) — exactly one cache-ahead banner fired, got ' + JSON.stringify(banners));
  assert(/gen 2/.test(ahead[0].message), '44(1) — the banner names generation 2, got: ' + ahead[0].message);
  assert(!banners.some((b) => b.type === 'reconcile-required' || b.type === 'db-not-persisting'), '44(1) — no file-path banner on a skip');
  assert(getLSGTag() && getLSGTag().maxGeneration === 2 && getLSGTag().rngState === 424242, '44(1) — getLSGTag() UNCHANGED (the divergent copy was kept), got ' + JSON.stringify(getLSGTag()));
  // Strictly-newer case: logging gen 3 makes the incoming persist STRICTLY
  // NEWER than the stored (424242) tag — overwrite, NO banner.
  logGen(db, 3);
  const bannersBefore = banners.length;
  const lsgBefore = getLSGTag();
  const r2 = await db.persist();
  assert(r2 && r2.ok === true, '44(2) — the strictly-newer persist succeeds, got ' + JSON.stringify(r2));
  assert(r2.tag && r2.tag.maxGeneration === 3, '44(2) — the persist return tag names gen 3, got ' + JSON.stringify(r2.tag));
  assert(!banners.slice(bannersBefore).some((b) => b.type === 'cache-ahead'), '44(2) — NO cache-ahead banner in the strictly-newer case, got ' + JSON.stringify(banners.slice(bannersBefore)));
  assert(getLSGTag() && getLSGTag().maxGeneration === 3 && getLSGTag().rngState !== 424242, '44(2) — the LSG was OVERWRITTEN with the new (gen-3) tag, got ' + JSON.stringify(getLSGTag()));
  assert(lsgBefore && lsgBefore.rngState === 424242, '44(2) — sanity: the pre-overwrite LSG tag was the divergent 424242 one');
  db.db.close();
});

// =============================================================================
// FITNESS-V2 legs 45–53 — schema v2 migration, signed fitness, outcome enum.
// =============================================================================

// clearRows(db) — dump the four lineage tables as comparable JSON for the
// migration byte-compare (rows must survive v1 → v2 EXACTLY — created_at
// included).
function snapshotLineageRows(db) {
  const q = (sql) => {
    const res = db.db.exec(sql);
    return res && res[0] ? res[0].values : [];
  };
  return {
    generations: q('SELECT generation, created_at, drivers_evaluated, best_fitness, avg_fitness, worst_fitness, best_driver_index, mutation_rate, mutation_scale, track_json, track_hash, sim_seconds, rng_state FROM generations ORDER BY generation'),
    driverResults: q('SELECT generation, driver_index, distance, top_speed, time_alive, crashed, stuck FROM driver_results ORDER BY generation, driver_index'),
    bestWeights: q('SELECT generation, architecture, weights_json, created_at FROM best_weights ORDER BY generation'),
  };
}

// FK-list normalization — foreign_key_list output varies by VM build (the `to`
// column may arrive QUOTED, e.g. '"generation"'); normalize strip-quotes +
// lowercase before comparing. Mirrors db.js's checkSingleFk contract.
function readChildFk(db, table) {
  const res = db.db.exec('PRAGMA foreign_key_list(' + table + ')');
  const rows = (res && res[0] && res[0].values) || [];
  return rows.map((row) => ({
    id: row[0],
    seq: row[1],
    table: String(row[2]).toLowerCase(),
    from: String(row[3]).toLowerCase(),
    to: String(row[4] || '').replace(/^"(.*)"$/, '$1').toLowerCase(),
    onDelete: String(row[6] == null ? '' : row[6]).trim().toLowerCase(),
  }));
}

// buildV1LineageBytes() — a FAITHFUL v1 fixture: the v1 DDL exactly as the old
// build shipped it (generations carries `CHECK (best_fitness >= 0)` — values
// MUST be non-negative here; a v1 file can never hold negative fitness), two
// logged generations, both best_weights rows, meta pinned. Adopting these
// bytes MUST migrate v1 → v2 → v3 in place, preserving every row.
// (FIT-V3: the (track_json, track_hash) pair MUST recombine — the battery's
// self-hash arm refuses divergent fixtures. DEFAULT_TRACK_JSON/DEFAULT_TRACK_HASH
// are used throughout.)
function buildV1LineageBytes() {
  const SQL = globalThis.SQL;
  const V1_DDL_GENERATIONS = "CREATE TABLE generations(generation INTEGER PRIMARY KEY CHECK (generation >= 1), created_at TEXT NOT NULL DEFAULT (datetime('now')), drivers_evaluated INTEGER NOT NULL CHECK (drivers_evaluated > 0), best_fitness REAL NOT NULL CHECK (best_fitness >= 0), avg_fitness REAL NOT NULL, worst_fitness REAL NOT NULL, best_driver_index INTEGER NOT NULL CHECK (best_driver_index BETWEEN 0 AND 19), mutation_rate REAL NOT NULL, mutation_scale REAL NOT NULL, track_json TEXT NOT NULL, track_hash TEXT NOT NULL, sim_seconds REAL NOT NULL, rng_state INTEGER NOT NULL CHECK (rng_state BETWEEN 0 AND 4294967295));";
  const raw = new SQL.Database();
  raw.run('PRAGMA foreign_keys = ON');
  raw.run('CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);');
  raw.run(V1_DDL_GENERATIONS);
  raw.run("CREATE TABLE driver_results(generation INTEGER NOT NULL REFERENCES generations(generation) ON DELETE CASCADE, driver_index INTEGER NOT NULL CHECK (driver_index BETWEEN 0 AND 19), distance REAL NOT NULL, top_speed REAL NOT NULL, time_alive REAL NOT NULL, crashed INTEGER NOT NULL CHECK (crashed IN (0,1)), stuck INTEGER NOT NULL CHECK (stuck IN (0,1)), PRIMARY KEY (generation, driver_index));");
  raw.run("CREATE TABLE best_weights(generation INTEGER PRIMARY KEY REFERENCES generations(generation) ON DELETE CASCADE, architecture TEXT NOT NULL CHECK (architecture = '7-16-16-2'), weights_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now')));");
  raw.run('PRAGMA user_version = 1');
  raw.run("INSERT INTO meta(key, value) VALUES ('schema_version', '1')");
  raw.run("INSERT INTO meta(key, value) VALUES ('architecture', '7-16-16-2')");
  raw.run("INSERT INTO meta(key, value) VALUES ('app_version', '1.0.0')");
  raw.run("INSERT INTO meta(key, value) VALUES ('created_at', '2026-11-01 00:00:00')");
  raw.run("INSERT INTO meta(key, value) VALUES ('track_hash', ?)", [DEFAULT_TRACK_HASH]);
  raw.run("INSERT INTO meta(key, value) VALUES ('rng_state', '1001')");
  const weights = makeValidWeightsString();
  for (let g = 1; g <= 2; g++) {
    raw.run('INSERT INTO generations (generation, drivers_evaluated, best_fitness, avg_fitness, worst_fitness, best_driver_index, mutation_rate, mutation_scale, track_json, track_hash, sim_seconds, rng_state) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
      [g, POPULATION, 1000 + g, 900 + g, 800 + g, 0, MUTATION_RATE, MUTATION_SCALE, DEFAULT_TRACK_JSON, DEFAULT_TRACK_HASH, 100 + g, 1000 + g]);
    for (let i = 0; i < POPULATION; i++) {
      raw.run('INSERT INTO driver_results (generation, driver_index, distance, top_speed, time_alive, crashed, stuck) VALUES (?,?,?,?,?,?,?)',
        [g, i, 500 + i, 300 + i, 10 + i, i % 4 === 0 ? 1 : 0, 0]);
    }
    raw.run("INSERT INTO best_weights (generation, architecture, weights_json) VALUES (?,?,?)", [g, '7-16-16-2', weights]);
  }
  const bytes = raw.export();
  raw.close();
  return bytes;
}

// buildV2LineageBytes(pin) — a FAITHFUL uv==2 fixture (FIT-V2 bytes, zero tracks
// table): fresh v3 output would not do (freshDb() is v3 now). generations 1–2
// committed on DEFAULT_TRACK_JSON/DEFAULT_TRACK_HASH by default; meta pinned
// with rng_state='123456789'; schema_version='2'; app_version='1.0.0'. Adopting
// these bytes MUST migrate v2 → v3 in the same call. An optional pin
// { trackJson, trackHash } swaps the pinned geometry (e.g. the editor-pin
// retrofit arm of leg 66) — generations 1..2 then carry the editor track.
function buildV2LineageBytes(pin) {
  const SQL = globalThis.SQL;
  const { DDL_GENERATIONS_V2, DDL_DRIVER_RESULTS, DDL_BEST_WEIGHTS, DDL_META } = requireDb();
  const pinTrackJson = (pin && pin.trackJson) || DEFAULT_TRACK_JSON;
  const pinTrackHash = (pin && pin.trackHash) || DEFAULT_TRACK_HASH;
  const raw = new SQL.Database();
  raw.run('PRAGMA foreign_keys = ON');
  raw.run(DDL_GENERATIONS_V2);
  raw.run(DDL_DRIVER_RESULTS);
  raw.run(DDL_BEST_WEIGHTS);
  raw.run(DDL_META);
  raw.run('PRAGMA user_version = 2');
  raw.run("INSERT INTO meta(key, value) VALUES ('schema_version', '2')");
  raw.run("INSERT INTO meta(key, value) VALUES ('architecture', '7-16-16-2')");
  raw.run("INSERT INTO meta(key, value) VALUES ('app_version', '1.0.0')");
  raw.run("INSERT INTO meta(key, value) VALUES ('created_at', '2026-11-01 00:00:00')");
  raw.run("INSERT INTO meta(key, value) VALUES ('track_hash', ?)", [pinTrackHash]);
  raw.run("INSERT INTO meta(key, value) VALUES ('rng_state', '123456789')");
  const weights = makeValidWeightsString();
  for (let g = 1; g <= 2; g++) {
    raw.run('INSERT INTO generations (generation, drivers_evaluated, best_fitness, avg_fitness, worst_fitness, best_driver_index, mutation_rate, mutation_scale, track_json, track_hash, sim_seconds, rng_state) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
      [g, POPULATION, 1000 + g, 900 + g, 800 + g, 0, MUTATION_RATE, MUTATION_SCALE, pinTrackJson, pinTrackHash, 100 + g, 1000 + g]);
    for (let i = 0; i < POPULATION; i++) {
      raw.run('INSERT INTO driver_results (generation, driver_index, distance, top_speed, time_alive, crashed, stuck) VALUES (?,?,?,?,?,?,?)',
        [g, i, 500 + i, 300 + i, 10 + i, i % 4 === 0 ? 1 : 0, 0]);
    }
  }
  raw.run("INSERT INTO best_weights (generation, architecture, weights_json) VALUES (?,?,?)", [2, '7-16-16-2', weights]);
  const bytes = raw.export();
  raw.close();
  return bytes;
}

// craftV3Bytes(mutate) — async: builds the v3 base bytes (fresh lineage, one
// committed generation), then applies raw SQL against a THROWAWAY connection
// over those bytes and returns the re-exported, poisoned file. CHECK
// constraints are NOT enforced on this connection when the mutator enables
// `PRAGMA ignore_check_constraints = ON` first — exactly the byte-crafting
// class the battery's typeof-strict arms defend.
async function craftV3Bytes(mutate) {
  const db = await freshDb();
  ensureLineageFor(db);
  logGen(db, 1);
  const base = db.exportBytes();
  db.db.close();
  const SQL = globalThis.SQL;
  const raw = new SQL.Database(base);
  mutate(raw);
  const bytes = raw.export();
  raw.close();
  return bytes;
}

// buildV3ProbeBytes(over) — an EXACT-twin uv==3 probe file: tracks is created
// with DDL_TRACKS verbatim; generations carries DDL_GENERATIONS (v3, with the
// REFERENCES clause) by default. over:
//   generationsDdl  — literal to use for the generations table
//   tracksDdl       — literal to use for the tracks table
//   uv              — user_version to stamp (default 3)
//   schemaVersion   — '3' (default) | null to OMIT the meta row | a literal
//   appVersion      — '1.1.0' (default)
function buildV3ProbeBytes(over) {
  over = over || {};
  const SQL = globalThis.SQL;
  const { DDL_GENERATIONS, DDL_TRACKS, DDL_DRIVER_RESULTS, DDL_BEST_WEIGHTS, DDL_META } = requireDb();
  const generationsDdl = over.generationsDdl === undefined ? DDL_GENERATIONS : over.generationsDdl;
  const tracksDdl = over.tracksDdl === undefined ? DDL_TRACKS : over.tracksDdl;
  const raw = new SQL.Database();
  raw.run('PRAGMA foreign_keys = OFF');
  raw.run(tracksDdl);
  raw.run(generationsDdl);
  raw.run(DDL_DRIVER_RESULTS);
  raw.run(DDL_BEST_WEIGHTS);
  raw.run(DDL_META);
  raw.run('PRAGMA user_version = ' + (over.uv === undefined ? 3 : over.uv));
  if (over.schemaVersion !== null) {
    raw.run("INSERT INTO meta(key, value) VALUES ('schema_version', ?)", [over.schemaVersion === undefined ? '3' : over.schemaVersion]);
  }
  raw.run("INSERT INTO meta(key, value) VALUES ('architecture', '7-16-16-2')");
  raw.run("INSERT INTO meta(key, value) VALUES ('app_version', ?)", [over.appVersion === undefined ? '1.1.0' : over.appVersion]);
  raw.run("INSERT INTO meta(key, value) VALUES ('created_at', '2026-11-01 00:00:00')");
  raw.run("INSERT INTO meta(key, value) VALUES ('track_hash', ?)", [DEFAULT_TRACK_HASH]);
  raw.run("INSERT INTO meta(key, value) VALUES ('rng_state', '123456789')");
  const weights = makeValidWeightsString();
  raw.run('INSERT INTO generations (generation, drivers_evaluated, best_fitness, avg_fitness, worst_fitness, best_driver_index, mutation_rate, mutation_scale, track_json, track_hash, sim_seconds, rng_state) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
    [1, POPULATION, 1000, 900, 800, 0, MUTATION_RATE, MUTATION_SCALE, DEFAULT_TRACK_JSON, DEFAULT_TRACK_HASH, 105, 1001]);
  for (let i = 0; i < POPULATION; i++) {
    raw.run('INSERT INTO driver_results (generation, driver_index, distance, top_speed, time_alive, crashed, stuck) VALUES (?,?,?,?,?,?,?)',
      [1, i, 500 + i, 300 + i, 10 + i, i % 4 === 0 ? 1 : 0, 0]);
  }
  raw.run('INSERT INTO best_weights (generation, architecture, weights_json) VALUES (?,?,?)', [1, '7-16-16-2', weights]);
  const defaultPoints = JSON.stringify(DEFAULT_CONTROL_POINTS.map((p) => ({ x: p.x, y: p.y })));
  raw.run("INSERT INTO tracks (track_hash, source, control_points, half_width, difficulty, created_at) VALUES (?, 'default', ?, 70, 0, '2026-01-01 00:00:00')", [DEFAULT_TRACK_HASH, defaultPoints]);
  const bytes = raw.export();
  raw.close();
  return bytes;
}

// =============================================================================
// leg 45 — fresh DBs are v4 (FIT-V6): user_version == 4, meta.schema_version ==
// '4', meta.wall_model == 'v6', FK ON; DDL_GENERATIONS carries the REFERENCES
// tracks clause (FIT-V3 twin, reused); the migration twin DDL_GENERATIONS_NEW
// differs from DDL_GENERATIONS ONLY by the table name.
// =============================================================================
await leg('leg 45 — fresh v4 DBs: user_version=4, meta.schema_version=4, wall_model=v6, REFERENCES clause present', async () => {
  const { DDL_GENERATIONS, DDL_GENERATIONS_NEW } = requireDb();
  const db = await freshDb();
  const uv = db.db.exec('SELECT COALESCE(user_version, 0) FROM pragma_user_version');
  assert(uv && uv[0] && Number(uv[0].values[0][0]) === 4, 'fresh DB must carry user_version == 4, got ' + JSON.stringify(uv));
  const sv = db.db.exec("SELECT value FROM meta WHERE key = 'schema_version'");
  assert(sv && sv[0] && sv[0].values[0][0] === '4', "fresh DB must carry meta.schema_version == '4', got " + JSON.stringify(sv && sv[0] && sv[0].values));
  const wm = db.db.exec("SELECT value FROM meta WHERE key = 'wall_model'");
  assert(wm && wm[0] && wm[0].values[0][0] === 'v6', "fresh DB must carry meta.wall_model == 'v6', got " + JSON.stringify(wm && wm[0] && wm[0].values));
  assert(!DDL_GENERATIONS.includes('CHECK (best_fitness >= 0)'), 'DDL_GENERATIONS must NOT contain the fitness-non-negative CHECK');
  assert(DDL_GENERATIONS.includes('best_fitness REAL NOT NULL'), 'DDL_GENERATIONS keeps best_fitness REAL NOT NULL');
  assert(DDL_GENERATIONS.includes('REFERENCES tracks(track_hash)'), 'DDL_GENERATIONS carries the v3 REFERENCES tracks(track_hash) clause');
  assert(DDL_GENERATIONS_NEW === DDL_GENERATIONS.replace('CREATE TABLE generations(', 'CREATE TABLE generations_new('), 'DDL_GENERATIONS_NEW must be byte-identical to DDL_GENERATIONS except the table name');
  db.db.close();
});

// =============================================================================
// leg 46 / 46b — v1 fixture adoption: the ADOPTION BATTERY passes the v1
// bytes, then the v1 → v2 migration runs IN THE SAME CALL and preserves
// every row byte-for-byte (created_at included), re-declares the children's
// FKs (exactly one each: generations, ON DELETE CASCADE, quote-tolerant) and
// empties foreign_key_check. 46b — an INJECTED mid-migration failure rolls
// the transaction back, discards the instance (zero adoption), leaves the
// caller's bytes untouched, and a re-adoption afterwards succeeds cleanly.
// =============================================================================
await leg('leg 46 — pre-v4 lineage refused on adopt: typed WallModelChangedError, zero adoption, byte-preserved', async () => {
  const { Db } = requireDb();
  const v1Bytes = buildV1LineageBytes();
  assertWallModelRefusal(Db, v1Bytes, 'leg 46');
  const db = new Db();
  let threw = null;
  try { await db.init(v1Bytes); } catch (err) { threw = err; }
  assert(threw !== null && threw.name === 'WallModelChangedError', 'leg 46 — init() typed refusal, got ' + (threw && threw.name));
  assert(db.db === null, 'leg 46 — zero adoption on refusal');
  assert(sameBytes(v1Bytes, new Uint8Array(v1Bytes)), 'leg 46 — caller bytes untouched');
});
await leg('leg 46b — pre-v4 refusal is read-only: bytes untouched, no migration fault reachable, clean re-adoption of uv4', async () => {
  const { Db } = requireDb();
  const v1Bytes = buildV1LineageBytes();
  const before = Buffer.from(v1Bytes);
  assertWallModelRefusal(Db, v1Bytes, 'leg 46b');
  const db = new Db();
  let threw = null;
  try { await db.init(v1Bytes); } catch (err) { threw = err; }
  assert(threw !== null && threw.name === 'WallModelChangedError', 'leg 46b — typed refusal, got ' + (threw && threw.name));
  assert(db.db === null, 'leg 46b — instance discarded (zero adoption)');
  assert(Buffer.from(v1Bytes).equals(before), 'leg 46b — caller bytes UNCHANGED');
  // The refusal is deterministic and repeatable; no migration code was reached.
  assertWallModelRefusal(Db, v1Bytes, 'leg 46b repeat');
  // A clean uv4 lineage still adopts normally (no over-refusal).
  const clean = new Db();
  await clean.init();
  assert(clean.db !== null && Number(clean.db.exec('PRAGMA user_version')[0].values[0][0]) === 4, 'leg 46b — clean uv4 init succeeds');
  clean.db.close();
});
await leg('leg 47 — negative-fitness generation: log → export → battery re-adopt round trip', async () => {
  const { Db } = requireDb();
  const track = new Track(DEFAULT_CONTROL_POINTS, 70);
  // All-reverse policy: strictly negative speed ⇒ strictly negative odometer;
  // maxCumArc never grows ⇒ laps == 0 ⇒ fitness == odometer < 0.
  const sim = new DriverSim(track, null, () => [-1, 0]);
  let guard = 0;
  while (!sim.step().done && guard < 100000) guard += 1;
  assert(sim.fitness() < 0, 'expected a negative fitness, got ' + sim.fitness());
  assert(sim.laps === 0, 'reverse run credits zero laps, got ' + sim.laps);
  const neg = sim.fitness();
  // A deliberately net-negative population derived from that run.
  const results = [];
  for (let i = 0; i < POPULATION; i++) {
    results.push({ distance: neg - i, topSpeed: 100 + i, timeAlive: sim.timeAlive, crashed: 0, stuck: sim.stuck ? 1 : 0 });
  }
  const avg = results.reduce((s, r) => s + r.distance, 0) / results.length;
  const db = await freshDb();
  ensureLineageFor(db);
  logGen(db, 1, {
    results,
    bestFitness: neg,
    avgFitness: avg,
    worstFitness: neg - (POPULATION - 1),
    bestDriverIndex: 0,
    rngState: 4242,
  });
  assert(db.getGeneration(1).best_fitness === neg, 'logged best_fitness is the negative value, got ' + db.getGeneration(1).best_fitness);
  const bytes = db.exportBytes();
  const summary = Db.validateAdoptionBytes(bytes);
  assert(summary.generation === 1, 'exported v2 bytes pass the battery');
  const db2 = new Db();
  await db2.init(bytes);
  assert(db2.getGeneration(1).best_fitness === neg, 're-adopted best_fitness is preserved, got ' + db2.getGeneration(1).best_fitness);
  assert(db2.getDriverResults(1)[0].distance === neg, 're-adopted driver_results[0].distance is preserved, got ' + db2.getDriverResults(1)[0].distance);
  assert(db2.stats().bestFitnessEver === neg, 're-adopted stats().bestFitnessEver is the negative value, got ' + db2.stats().bestFitnessEver);
  const rs = db2.resumeState();
  assert(rs.maxGeneration === 1, 'resumeState maxGeneration === 1, got ' + rs.maxGeneration);
  assert(rs.rngState === 4242, 'resumeState rng_state matches the logged cursor (4242), got ' + rs.rngState);
  db.db.close();
  db2.db.close();
});

// =============================================================================
// leg 48 — R1 migration-order spot check: after the v1 → v2 adoption, the
// children's declared FKs report table=='generations', on_delete=='CASCADE'
// (quote-tolerant normalization, case-insensitive) — the rename-first order
// of the migration is what makes this hold.
// =============================================================================
await leg('leg 48 — R1 FK declaration spot check on a fresh uv4 lineage: children FK -> generations, ON DELETE CASCADE', async () => {
  const db = await freshDb();
  ensureLineageFor(db);
  for (const table of ['driver_results', 'best_weights']) {
    const fks = readChildFk(db, table);
    assert(fks.length === 1, table + ' declares exactly one FK, got ' + JSON.stringify(fks));
    assert(fks[0].table === 'generations', table + ' FK targets generations, got ' + JSON.stringify(fks[0]));
    assert(fks[0].from === 'generation' && fks[0].to === 'generation', table + ' FK maps generation -> generation');
    assert(fks[0].onDelete === 'cascade', table + ' FK is ON DELETE CASCADE');
  }
  const fkc = db.db.exec('PRAGMA foreign_key_check');
  assert(!fkc || !fkc[0] || !fkc[0].values || fkc[0].values.length === 0, 'foreign_key_check empty');
  db.db.close();
});
await leg('leg 49 — fresh fork on v3: forkLineage wipe, re-log gens 1+2, FK cascade intact', async () => {
  const db = await freshDb();
  ensureLineageFor(db);
  logGen(db, 1);
  logGen(db, 2);
  const wiped = db.forkLineage(DEFAULT_TRACK_HASH, 999, DEFAULT_TRACK_JSON);
  assert(wiped && wiped.ok === true, 'forkLineage reports ok, got ' + JSON.stringify(wiped));
  assert(countRows(db, 'generations') === 0 && countRows(db, 'driver_results') === 0 && countRows(db, 'best_weights') === 0, 'forkLineage wipes generations + children');
  logGen(db, 1);
  logGen(db, 2);
  assert(db.listGenerations().join() === '1,2', 're-logged generations present after fork, got ' + db.listGenerations());
  assert(countRows(db, 'driver_results') === POPULATION * 2, 're-logged driver_results rows present');
  const fks = readChildFk(db, 'driver_results');
  assert(fks.length === 1 && fks[0].table === 'generations' && fks[0].onDelete === 'cascade', 'driver_results FK intact post-fork, got ' + JSON.stringify(fks));
  db.db.close();
});

// =============================================================================
// leg 50 — terminal outcomes: 'crashed' | 'stuck' | 'time-cap' | 'finished' —
// via stubbed Car (swapped onto globalThis.Car, restored in `finally`). The
// 'finished' arm teleports the car along the centerline 0.45 × track.length
// per tick — laps hits LAP_BONUS_LAPS (100) and the run ends 'finished' before
// any other kill condition fires. Arms (e) and (f) pin the KILL ORDER: (e) the
// crash-on-finish-line arm co-fires crash with laps == LAP_BONUS_LAPS — crash
// must win; (f) the grinder arm co-fires laps == LAP_BONUS_LAPS with the
// stagnation timer at/past STUCK_TIMEOUT_S AND _ticks == maxTicks — finished
// must win (finished beats stuck and beats time-cap). The stub-based arms
// never touch real physics/walls; the stuck/crash arms pin the flags.
// =============================================================================
await leg('leg 50 — outcomes crashed / stuck / time-cap / finished (stubbed car)', async () => {
  const track = new Track(DEFAULT_CONTROL_POINTS, 70);
  const pose = track.startPose();
  const L = track.length;

  // (a) crashed — stub crashes on the very first tick.
  await withStubbedCar(StubCrashCar, async () => {
    const sim = new DriverSim(track, null, () => [1, 0]);
    const res = sim.step();
    assert(res.done === true && res.outcome === 'crashed', "crash arm: outcome must be 'crashed', got " + JSON.stringify(res));
    assert(sim.crashed === true && sim.stuck === false, 'crash arm: crashed flag set, stuck clear');
    assert(sim.laps === 0 && sim.outcome !== 'finished', 'crash arm: no laps credited, not finished');
  });

  // (b) stuck — stationary stub: stagnation timer (12 s) fires before anything else.
  await withStubbedCar(StubStationaryCar, async () => {
    const sim = new DriverSim(track, null, () => [0, 0]);
    let res = null;
    for (;;) {
      res = sim.step();
      if (res.done) break;
      if (sim.timeAlive > STUCK_TIMEOUT_S + 1) break;
    }
    assert(res && res.outcome === 'stuck', "stuck arm: outcome must be 'stuck', got " + JSON.stringify(res));
    assert(sim.stuck === true && sim.crashed === false, 'stuck arm: stuck flag set, crashed clear');
    assert(sim.timeAlive >= STUCK_TIMEOUT_S - 2 * SIM_DT && sim.timeAlive <= STUCK_TIMEOUT_S + 2 * SIM_DT + 1e-9, 'stuck arm: killed at STUCK_TIMEOUT_S, got ' + sim.timeAlive);
  });

  // (c) time-cap — stub advances only PROGRESS_ADVANCE_PX + ε per tick, so the
  // stagnation window never trips mid-run; it reaches EVAL_CAP_S (45 s) long
  // before lapping LAP_BONUS_LAPS (needs ~100 laps — impossible here).
  const slowArc = PROGRESS_ADVANCE_PX + 0.5;
  await withStubbedCar(makeAdvancingStubCar(slowArc), async () => {
    const sim = new DriverSim(track, null, () => [1, 0]);
    let res = null;
    for (;;) {
      res = sim.step();
      if (res.done) break;
    }
    assert(res && res.outcome === 'time-cap', "time-cap arm: outcome must be 'time-cap', got " + JSON.stringify(res));
    assert(sim.laps >= 1 && sim.laps < LAP_BONUS_LAPS, 'time-cap arm: some laps banked but the 100-lap cap not reached, got ' + sim.laps);
    assert(sim.crashed === false && sim.stuck === false, 'time-cap arm: neither crash nor stuck involved');
  });

  // (d) finished — the 0.45 × length teleport reaches LAP_BONUS_LAPS in
  // ~223 ticks; kill order crash > finished > stuck > time-cap means the
  // 'finished' outcome must fire the moment laps hits the cap.
  const fastArc = 0.45 * L;
  await withStubbedCar(makeAdvancingStubCar(fastArc), async () => {
    const sim = new DriverSim(track, null, () => [1, 0]);
    let res = null;
    for (let stepN = 0; stepN < 10000; stepN++) {
      res = sim.step();
      if (res.done) break;
    }
    assert(res && res.outcome === 'finished', "finished arm: outcome must be 'finished', got " + JSON.stringify(res));
    assert(sim.laps === LAP_BONUS_LAPS, 'finished arm: laps capped at LAP_BONUS_LAPS, got ' + sim.laps);
    assert(LAP_BONUS_LAPS === 100, 'LAP_BONUS_LAPS is 100');
    assert(sim.crashed === false && sim.stuck === false, "finished arm: 'finished' leaves crashed/stuck false (R9: train-worker posts the flags unchanged)");
    const expected = sim.car.odometer + sim.laps * LAP_BONUS_TRACK_LENGTHS * L;
    assert(sim.fitness() === expected, 'finished arm: fitness === odometer + laps × LAP_BONUS_TRACK_LENGTHS × track.length');
    assert(Number.isFinite(sim.fitness()), 'finished arm: fitness is finite');
  });

  // (e) crash ON the finish line — the stub telescopes 0.45 × length per tick;
  // the teleport that crosses the 100-lap line happens during one update(), but
  // the sim's maxCumArc ratchet only reads the position one tick LATER — so the
  // crash flag set in THAT later update() and laps == LAP_BONUS_LAPS co-fire at
  // the kill check. The outcome MUST be 'crashed' (crash beats finished).
  // Stagnation never fires here (every cruise tick resets it); this arm pins
  // the precedence, not the stagnation window.
  await withStubbedCar(StubCrashOnFinishCar, async () => {
    const sim = new DriverSim(track, null, () => [1, 0]);
    let res = null;
    for (let stepN = 0; stepN < 10000; stepN++) {
      res = sim.step();
      if (res.done) break;
    }
    assert(res && res.outcome === 'crashed', "crash-on-finish-line arm: outcome must be 'crashed' (crash beats finished), got " + JSON.stringify(res));
    assert(sim.laps === LAP_BONUS_LAPS, 'crash-on-finish-line arm: laps reached the LAP_BONUS_LAPS cap on the crash tick, got ' + sim.laps);
    assert(sim.crashed === true && sim.stuck === false, 'crash-on-finish-line arm: crashed flag set, stuck clear');
  });

  // (f) finished BEATS stuck AND time-cap — the grinder stub reaches laps ==
  // LAP_BONUS_LAPS exactly on tick 5400 == maxTicks, with the stagnation timer
  // at 12 s+ (the last ≥ 20 px advance landed at tick 3958, landing the
  // stagnation reset on tick 3959; the stall then accumulates past STUCK_TIMEOUT_S).
  // All three kill conditions co-fire at tick 5400; 'finished' must win.
  await withStubbedCar(StubGrindFinishCar, async () => {
    const sim = new DriverSim(track, null, () => [1, 0]);
    assert(sim.maxTicks === Math.ceil(EVAL_CAP_S / SIM_DT), 'grinder sanity: maxTicks is the EVAL_CAP_S tick cap, got ' + sim.maxTicks);
    let res = null;
    for (let stepN = 0; stepN < 10000; stepN++) {
      res = sim.step();
      if (res.done) break;
    }
    assert(res && res.outcome === 'finished', "finished-precedence arm: outcome must be 'finished' (finished beats stuck and time-cap), got " + JSON.stringify(res));
    assert(sim.laps === LAP_BONUS_LAPS, 'finished-precedence arm: laps capped at LAP_BONUS_LAPS on the co-fire tick, got ' + sim.laps);
    assert(sim.timeAlive >= EVAL_CAP_S - 1e-9, 'finished-precedence arm: simulation time is at the EVAL_CAP_S boundary (within FP epsilon), got ' + sim.timeAlive);
    assert(sim._stagnation >= STUCK_TIMEOUT_S, 'finished-precedence arm: stagnation timer is at/past the STUCK_TIMEOUT_S boundary on the co-fire tick, got ' + sim._stagnation);
    assert(sim.crashed === false && sim.stuck === false, "finished-precedence arm: 'finished' leaves crashed/stuck false");
  });

  // sanity — globalThis.Car is the real Car again (restored in finally).
  assert(globalThis.Car === Car, 'globalThis.Car restored to the real Car');
});

// =============================================================================
// leg 51 — R5 fitness-pipeline fixture: a deliberately NET-NEGATIVE
// population (19 reverse rows derived from one measured all-reverse run).
// Logged generation → the logged fields (best/avg/worst/driver distances),
// the replay stats, the stats()/summaries read-paths and the replayed
// onFinish payload ALL agree on the bonus-INCLUSIVE value:
//   fitness === odometer + laps × LAP_BONUS_TRACK_LENGTHS × track.length
// =============================================================================
await leg('leg 51 — R5 pipeline fixture: net-negative population, bonus-inclusive end-to-end', async () => {
  const { Db } = requireDb();
  const track = new Track(DEFAULT_CONTROL_POINTS, 70);

  // Fitness identity, odometer side (laps == 0 reverse run).
  const rev = new DriverSim(track, null, () => [-1, 0]);
  while (!rev.step().done) {}
  assert(rev.fitness() === rev.car.odometer + rev.laps * LAP_BONUS_TRACK_LENGTHS * track.length, 'identity (reverse): fitness === odometer + laps × bonus');

  // Fitness identity with the bonus term NON-trivial — stub-forward at 0.45×L:
  // reaches the 100-lap cap; odometer is fresh-signed positive; the bonus
  // term dominates (100 × 2 × 2069.6 ≈ 413,930 px).
  const fastArc = 0.45 * track.length;
  await withStubbedCar(makeAdvancingStubCar(fastArc), async () => {
    const fwd = new DriverSim(track, null, () => [1, 0]);
    for (;;) { if (fwd.step().done) break; }
    assert(fwd.laps === LAP_BONUS_LAPS, 'bonus fixture: stub reaches the lap cap, got ' + fwd.laps);
    const expected = fwd.car.odometer + fwd.laps * LAP_BONUS_TRACK_LENGTHS * track.length;
    assert(fwd.fitness() === expected, 'identity (laps=100): fitness === odometer + laps × bonus');
    assert(fwd.fitness() - fwd.car.odometer === 100 * LAP_BONUS_TRACK_LENGTHS * track.length, 'the bonus term is exactly 2 × 100 laps of track.length');
  });

  // The seeded net-negative population.
  const neg = rev.fitness();
  const results = [];
  for (let i = 0; i < POPULATION; i++) {
    results.push({ distance: neg - i, topSpeed: 100 + i, timeAlive: rev.timeAlive, crashed: 0, stuck: 0 });
  }
  const sum = results.reduce((s, r) => s + r.distance, 0);
  const best = neg;               // results[0] is the largest
  const worst = neg - (POPULATION - 1);
  const avg = sum / results.length;
  assert(Evolution.selectBest(results) === 0, 'selectBest picks the least-negative row (fitness-aware comparison works on negatives)');

  const db = await freshDb();
  ensureLineageFor(db);
  logGen(db, 1, {
    results,
    bestFitness: best,
    avgFitness: avg,
    worstFitness: worst,
    bestDriverIndex: 0,
    rngState: 4242,
  });

  // Logged row agrees, including the odometer-vs-fitness distinction.
  const gen = db.getGeneration(1);
  assert(gen.best_fitness === best && gen.avg_fitness === avg && gen.worst_fitness === worst, 'logged best/avg/worst are the bonus-inclusive fitness');
  const dr = db.getDriverResults(1);
  assert(dr.length === POPULATION && dr[0].distance === neg, 'driver_results[0].distance is the fitness');

  // getBestWeights → weights for the elitism anchor (replay input).
  const net = db.getBestWeights(1);
  assert(net && net.getWeights, 'getBestWeights(1) returns the validated net');

  // Replay stats + onFinish payload agree on the fitness value.
  const loaded = Replay.loadGeneration(db, 1);
  assert(loaded.stats.distance === best, 'replay stats.distance is the SELECTED best driver fitness (bonus-inclusive), got ' + loaded.stats.distance);
  assert(loaded.stats.bestFitness === best && loaded.stats.avgFitness === avg && loaded.stats.worstFitness === worst, 'replay stats best/avg/worst fitness agree');
  const driven = new DriverSim(loaded.track, NeuralNetwork.fromJSON(loaded.bestWeights), null);
  while (!driven.step().done) {}
  let finishPayload = null;
  const run = Replay.createRun({ track: loaded.track, weights: loaded.bestWeights, onFinish: (p) => { finishPayload = p; } });
  run.step(60000);
  assert(finishPayload !== null, 'replay fires onFinish');
  assert(Number.isFinite(finishPayload.distance), 'replay onFinish.distance is finite');
  assert(Math.abs(finishPayload.distance - driven.fitness()) < 1e-9, 'replay onFinish.distance === bonus-inclusive fitness of the same deterministic run, got ' + finishPayload.distance + ' vs ' + driven.fitness());
  assert(finishPayload.distance === driven.car.odometer + driven.laps * LAP_BONUS_TRACK_LENGTHS * loaded.track.length, 'onFinish.distance equals odometer + laps × LAP_BONUS_TRACK_LENGTHS × track.length');
  assert(['crashed', 'stuck', 'time-cap', 'finished'].includes(finishPayload.outcome), "replay onFinish.outcome is one of the enumerated outcomes, got '" + finishPayload.outcome + "'");

  // stats() and the list/summaries read-paths agree.
  const stats = db.stats();
  assert(stats.generationCount === 1 && stats.bestFitnessEver === best && stats.avgOfLatest === avg, 'stats() agrees on the negative fitness, got ' + JSON.stringify(stats));
  const summaries = db.listGenerationSummaries(5, 0);
  assert(summaries.length === 1 && summaries[0].best_fitness === best && summaries[0].avg_fitness === avg && summaries[0].worst_fitness === worst, 'listGenerationSummaries agrees on the fitness values');
  db.db.close();
});

// =============================================================================
// leg 52 — R6 grep gate: js/ and tests/ must carry ZERO occurrences of the
// three banned legacy claims (always-growing odometer claim; never-decreases
// odometer assertion; the absolute-distance formula).
// (Patterns are assembled by concatenation so THIS leg's source does not
// itself contain them.)
// =============================================================================
await leg('leg 52 — R6 grep gate: signed-odometer incompatible claims are absent from js/ + tests/', () => {
  const { readdirSync, readFileSync } = require('node:fs');
  const path = require('node:path');
  const gates = [
    { name: 'always-growing odometer claim', re: new RegExp('mono' + 'tone(?!.*net)') },
    { name: 'never-decreases odometer assertion', re: new RegExp('odome' + 'ter >= prev' + 'Odom') },
    { name: 'absolute-distance formula', re: new RegExp('cumula' + 'tive \\|dista' + 'nce\\|') },
  ];
  const dirs = [path.join(REPO_ROOT, 'js'), path.join(REPO_ROOT, 'tests')];
  const files = [];
  for (const dir of dirs) {
    for (const name of readdirSync(dir)) {
      if (/\.(js|mjs)$/.test(name)) files.push(path.join(dir, name));
    }
  }
  assert(files.length >= 15, 'the gate sweeps the full js/ + tests/ tree (files=' + files.length + ')');
  const hits = [];
  for (const file of files) {
    const text = readFileSync(file, 'utf8');
    const lines = text.split('\n');
    for (let i = 0; i < lines.length; i++) {
      for (const gate of gates) {
        if (gate.re.test(lines[i])) {
          hits.push(path.relative(REPO_ROOT, file) + ':' + (i + 1) + ' — ' + gate.name + ': ' + lines[i].trim());
        }
      }
    }
  }
  assert(hits.length === 0, 'R6 gate FAILED, ' + hits.length + ' hit(s):\n         ' + hits.join('\n         '));
});

// =============================================================================
// leg 53 — fitness pipeline end-to-end: log gen 1 → export bytes → RE-ADOPT
// through the full battery → resumeState() reports the saved rng_state and
// maxGeneration → the best weights survive the round trip exactly, and a
// post-adoption log of generation 2 lands cleanly in the re-adopted lineage.
// =============================================================================
await leg('leg 53 — fitness pipeline end-to-end: log → export → battery → resumeState', async () => {
  const { Db } = requireDb();
  const db = await freshDb();
  ensureLineageFor(db);
  logGen(db, 1, { rngState: 1001 });
  const weightsBefore = JSON.stringify(db.getBestWeights(1).getWeights());
  const bytes = db.exportBytes();
  const summary = Db.validateAdoptionBytes(bytes);
  assert(summary.generation === 1 && summary.rngState === 1001, 'battery round trip: maxGen 1, rngState 1001, got ' + JSON.stringify(summary));
  const db2 = new Db();
  await db2.init(bytes);
  const rs = db2.resumeState();
  assert(rs.maxGeneration === 1, 'resumeState maxGeneration === 1, got ' + rs.maxGeneration);
  assert(rs.rngState === 1001, 'resumeState rng_state matches the saved cursor (1001), got ' + rs.rngState);
  const weightsAfter = JSON.stringify(db2.getBestWeights(1).getWeights());
  assert(weightsAfter === weightsBefore, 'best weights survive the export → adopt round trip byte-identically');
  const genBefore = db.getGeneration(1);
  const genAfter = db2.getGeneration(1);
  assert(genBefore.best_fitness === genAfter.best_fitness && genBefore.avg_fitness === genAfter.avg_fitness && genBefore.worst_fitness === genAfter.worst_fitness, 'logged fitness values round-trip exactly');
  logGen(db2, 2);
  assert(db2.listGenerations().join() === '1,2', 'post-adoption log lands in the re-adopted lineage');
  const rs2 = db2.resumeState();
  assert(rs2.maxGeneration === 2 && rs2.rngState === 1002, 'post-adoption log advances the rng_state cursor to 1002, got ' + rs2.rngState);
  db.db.close();
  db2.db.close();
});

// buildExactTwinV2Bytes(over) — a uv==2, schema_version=='2' file whose tables
// are created with the EXACT FIT-V2 DDL this build ships (sqlite_master will
// then hold it verbatim, modulo case/whitespace which the battery folds before
// comparing).
// over:
//   ddl            — verbatim CREATE TABLE generations(...) string to use
//                    (absent ⇒ DDL_GENERATIONS_V2 itself; used to craft the
//                    whitespace/case twin and the v1-CHECK-in-uv2 relabel)
//   schemaVersion  — '2' (default) | null to OMIT the meta row | a literal value
//   orphan         — true to add a driver_results row for generation 99 (an
//                    orphan no battery should tolerate, even uv==2)
//   trackJson      — DEFAULT_TRACK_JSON (default) | a literal track_json payload
//                    committed in the generations row (leg 67 forges a stale-hash
//                    row this way — the row's track_hash stays the default and
//                    the battery's self-hash arm must refuse)
// NOTE: the fixture pins Steps 3a/3c only — the migration path is leg 46/46b.
function buildExactTwinV2Bytes(over) {
  over = over || {};
  const SQL = globalThis.SQL;
  // The uv==2 file must carry the EXACT FIT-V2 generations DDL (the battery's
  // uv==2 twin arm compares against it — db.js keeps it as DDL_GENERATIONS_V2,
  // exported for exactly this fixture).
  const { DDL_GENERATIONS_V2 } = requireDb();
  // DDL literals mirrored from js/db.js (private, not exported). They MUST stay
  // byte-identical to js/db.js's DDL_DRIVER_RESULTS / DDL_BEST_WEIGHTS / DDL_META
  // or this fixture tests nothing; verified once at author time.
  const DDL_DRIVER_RESULTS = 'CREATE TABLE driver_results(generation INTEGER NOT NULL REFERENCES generations(generation) ON DELETE CASCADE, driver_index INTEGER NOT NULL CHECK (driver_index BETWEEN 0 AND 19), distance REAL NOT NULL, top_speed REAL NOT NULL, time_alive REAL NOT NULL, crashed INTEGER NOT NULL CHECK (crashed IN (0,1)), stuck INTEGER NOT NULL CHECK (stuck IN (0,1)), PRIMARY KEY (generation, driver_index));';
  const DDL_BEST_WEIGHTS = "CREATE TABLE best_weights(generation INTEGER PRIMARY KEY REFERENCES generations(generation) ON DELETE CASCADE, architecture TEXT NOT NULL CHECK (architecture = '7-16-16-2'), weights_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now')));";
  const DDL_META = 'CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);';
  const raw = new SQL.Database();
  raw.run('PRAGMA foreign_keys = ON');
  raw.run(over.ddl || DDL_GENERATIONS_V2);
  raw.run(DDL_DRIVER_RESULTS);
  raw.run(DDL_BEST_WEIGHTS);
  raw.run(DDL_META);
  raw.run('PRAGMA user_version = 2');
  if (over.schemaVersion !== null) {
    raw.run("INSERT INTO meta(key, value) VALUES ('schema_version', ?)", [over.schemaVersion === undefined ? '2' : over.schemaVersion]);
  }
  raw.run("INSERT INTO meta(key, value) VALUES ('architecture', '7-16-16-2')");
  raw.run("INSERT INTO meta(key, value) VALUES ('app_version', '1.0.0')");
  raw.run("INSERT INTO meta(key, value) VALUES ('created_at', '2026-11-01 00:00:00')");
  raw.run("INSERT INTO meta(key, value) VALUES ('track_hash', ?)", [DEFAULT_TRACK_HASH]);
  raw.run("INSERT INTO meta(key, value) VALUES ('rng_state', '123456789')");
  const weights = makeValidWeightsString();
  const trackJson = over.trackJson === undefined ? DEFAULT_TRACK_JSON : over.trackJson;
  raw.run('INSERT INTO generations (generation, drivers_evaluated, best_fitness, avg_fitness, worst_fitness, best_driver_index, mutation_rate, mutation_scale, track_json, track_hash, sim_seconds, rng_state) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
    [1, POPULATION, 1000, 900, 800, 0, MUTATION_RATE, MUTATION_SCALE, trackJson, DEFAULT_TRACK_HASH, 105, 1001]);
  for (let i = 0; i < POPULATION; i++) {
    raw.run('INSERT INTO driver_results (generation, driver_index, distance, top_speed, time_alive, crashed, stuck) VALUES (?,?,?,?,?,?,?)',
      [1, i, 500 + i, 300 + i, 10 + i, i % 4 === 0 ? 1 : 0, 0]);
  }
  if (over.orphan) {
    raw.run('PRAGMA foreign_keys = OFF');
    raw.run('INSERT INTO driver_results (generation, driver_index, distance, top_speed, time_alive, crashed, stuck) VALUES (99, 0, 1, 1, 1, 0, 0)');
    raw.run('PRAGMA foreign_keys = ON');
  }
  raw.run("INSERT INTO best_weights (generation, architecture, weights_json) VALUES (?,?,?)", [1, '7-16-16-2', weights]);
  const bytes = raw.export();
  raw.close();
  return bytes;
}

// =============================================================================
// leg 54 — battery arms PINNED by executable probes (security cycle-2, Finding
// 1): the two-sided DBL_MAX fitness arms, generation >= 1, mutation_rate ∈
// [0,1], finite mutation_scale ≥ 0, driver_index ∈ [0,POPULATION), created_at
// type 'text', Step 3a foreign_key_check for BOTH user_version arms, Step 3c
// schema_version readback (lying/missing) and DDL-twin compare (v1-DDL relabel),
// plus the anti-over-refusal whitespace/case twin. Fixtures mirror the probe
// classes that broke cycle 1; without these legs a silent deletion of any of
// the above arms keeps the rest of the suite green.
// =============================================================================
await leg('leg 54 — adoption battery arms pinned: pre-v4 fixtures typed-refused; value-domain/twin arms enforced on uv4 by legs 62/71/72', async () => {
  const { Db } = requireDb();
  // FIT-V6: every fixture in this leg is pre-v4, so step 3b fires first. The
  // arms below remain enforced on uv4 bytes by legs 62/71/72; here we pin that
  // the whole pre-v4 class is refused TYPED and byte-preserved.
  const check = (bytes, pattern, label) => { assertWallModelRefusal(Db, bytes, label); };

  // (a) two-sided fitness arms — -Infinity, +Infinity. (SQLite stores NaN as
  // NULL, which cannot satisfy a NOT NULL column via DML — NaN is unreachable
  // at rest through any INSERT/UPDATE path; the arms' `NOT (x = x)` disjuncts
  // remain as defense-in-depth for hypothetical future drift and are not
  // fixture-testable here.)
  check(buildSemanticProbe({ noFk: false, generation: { bestFitness: -Infinity } }), /best_fitness/i, 'arm (a) — -Infinity best_fitness refused');
  check(buildSemanticProbe({ noFk: false, generation: { avgFitness: Infinity } }), /avg_fitness/i, 'arm (a) — +Infinity avg_fitness refused');
  check(buildSemanticProbe({ noFk: false, generation: { worstFitness: -Infinity } }), /worst_fitness/i, 'arm (a) — -Infinity worst_fitness refused');
  check(buildSemanticProbe({ noFk: false, results: { distance: -Infinity } }), /value domain/i, 'arm (a) — -Infinity driver_results.distance refused');

  // (a') mutation_rate ∈ [0,1] — negative, >1, non-numeric (NaN unreachable at
  // rest — see (a)).
  check(buildSemanticProbe({ noFk: false, generation: { mutationRate: -0.1 } }), /mutation_rate/i, 'arm (a\') — mutation_rate −0.1 refused');
  check(buildSemanticProbe({ noFk: false, generation: { mutationRate: 2 } }), /mutation_rate/i, 'arm (a\') — mutation_rate 2 refused');
  check(buildSemanticProbe({ noFk: false, generation: { mutationRate: 'abc' } }), /mutation_rate/i, 'arm (a\') — mutation_rate text refused');

  // (a'') mutation_scale — negative, +Infinity (> DBL_MAX; NaN unreachable).
  check(buildSemanticProbe({ noFk: false, generation: { mutationScale: -1 } }), /mutation_scale/i, 'arm (a\'\') — mutation_scale −1 refused');
  check(buildSemanticProbe({ noFk: false, generation: { mutationScale: Infinity } }), /mutation_scale/i, 'arm (a\'\') — mutation_scale +Infinity refused');

  // (a''') generation >= 1.
  check(buildSemanticProbe({ noFk: false, generation: { generation: 0 } }), /generation/i, 'arm (a\'\'\') — generation=0 refused');

  // (a'''') driver_index must be an integer in [0, 19] — attacker DDL has no
  // CHECK, so a row with driver_index=99 is what the new arm refuses.
  (() => {
    const SQL = globalThis.SQL;
    const raw = new SQL.Database();
    for (const stmt of ATTACKER_DDL_WITH_FK) raw.run(stmt);
    raw.run('PRAGMA user_version = 1');
    raw.run("INSERT INTO meta(key, value) VALUES ('schema_version', '1')");
    raw.run("INSERT INTO meta(key, value) VALUES ('architecture', '7-16-16-2')");
    raw.run("INSERT INTO meta(key, value) VALUES ('app_version', '1.0.0')");
    raw.run("INSERT INTO meta(key, value) VALUES ('created_at', '2026-11-01 00:00:00')");
    raw.run("INSERT INTO meta(key, value) VALUES ('track_hash', ?)", [DEFAULT_TRACK_HASH]);
    raw.run("INSERT INTO meta(key, value) VALUES ('rng_state', '123456789')");
    raw.run('INSERT INTO generations (generation, drivers_evaluated, best_fitness, avg_fitness, worst_fitness, best_driver_index, mutation_rate, mutation_scale, track_json, track_hash, sim_seconds, rng_state) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
      [1, POPULATION, 1000, 900, 800, 0, MUTATION_RATE, MUTATION_SCALE, DEFAULT_TRACK_JSON, DEFAULT_TRACK_HASH, 105, 1001]);
    for (let i = 0; i < POPULATION - 1; i++) {
      raw.run('INSERT INTO driver_results (generation, driver_index, distance, top_speed, time_alive, crashed, stuck) VALUES (?,?,?,?,?,?,?)',
        [1, i, 500 + i, 300 + i, 10 + i, 0, 0]);
    }
    raw.run('INSERT INTO driver_results (generation, driver_index, distance, top_speed, time_alive, crashed, stuck) VALUES (?,?,?,?,?,?,?)',
      [1, 99, 1, 1, 1, 0, 0]);
    raw.run("INSERT INTO best_weights (generation, architecture, weights_json) VALUES (?,?,?)", [1, '7-16-16-2', makeValidWeightsString()]);
    const bytes = raw.export();
    raw.close();
    check(bytes, /driver_index/i, "arm (a'''') — driver_index=99 refused");
  })();

  // (b) Step 3a — orphan driver_results refused for uv==1.
  // Step 3a (foreign_key_check) runs BEFORE step 3b, so an orphan is refused
  // with the generic ValidationError even on pre-v4 bytes.
  assertBatteryReject(Db, buildSemanticProbe({ noFk: false, orphan: true }), /foreign_key_check|orphan/i, 'arm (b) — uv=1 orphan driver_results refused at Step 3a');
  // (b') Step 3a — orphan refused for uv==2 (the exact-twin file is otherwise v2-clean).
  assertBatteryReject(Db, buildExactTwinV2Bytes({ orphan: true }), /foreign_key_check|orphan/i, "arm (b') — uv=2 orphan driver_results refused at Step 3a");

  // (b'') Step 3c — uv==2 readback: lying schema_version, missing schema_version.
  check(buildExactTwinV2Bytes({ schemaVersion: '1' }), /schema_version/i, "arm (b'') — uv==2 with lying schema_version='1' refused at 3c");
  check(buildExactTwinV2Bytes({ schemaVersion: null }), /schema_version/i, "arm (b'') — uv==2 with missing schema_version refused at 3c");
  // (b''') Step 3c — DDL twin compare: v1-DDL generations table in uv==2 clothing.
  const v1Ddl = "CREATE TABLE generations(generation INTEGER PRIMARY KEY CHECK (generation >= 1), created_at TEXT NOT NULL DEFAULT (datetime('now')), drivers_evaluated INTEGER NOT NULL CHECK (drivers_evaluated > 0), best_fitness REAL NOT NULL CHECK (best_fitness >= 0), avg_fitness REAL NOT NULL, worst_fitness REAL NOT NULL, best_driver_index INTEGER NOT NULL CHECK (best_driver_index BETWEEN 0 AND 19), mutation_rate REAL NOT NULL, mutation_scale REAL NOT NULL, track_json TEXT NOT NULL, track_hash TEXT NOT NULL, sim_seconds REAL NOT NULL, rng_state INTEGER NOT NULL CHECK (rng_state BETWEEN 0 AND 4294967295));";
  check(buildExactTwinV2Bytes({ ddl: v1Ddl }), /DDL|v2 schema|does not match/i, "arm (b''') — uv==2 with v1-DDL generations refused at 3c");

  // (c) even a whitespace/case twin of the OLD v2 DDL is pre-v4 ⇒ typed refusal.
  const { DDL_GENERATIONS_V2 } = requireDb();
  const foldedDdl = DDL_GENERATIONS_V2.toLowerCase().replace(/\s+/g, '   ');
  const twinBytes = buildExactTwinV2Bytes({ ddl: foldedDdl });
  assertWallModelRefusal(Db, twinBytes, "arm (c) — whitespace/case twin of the v2 DDL is still pre-v4");

  // (d) created_at stays TEXT — a BLOB smuggled into the TEXT column is refused.
  (() => {
    const SQL = globalThis.SQL;
    const raw = new SQL.Database();
    for (const stmt of ATTACKER_DDL_WITH_FK) raw.run(stmt);
    raw.run('PRAGMA user_version = 1');
    raw.run("INSERT INTO meta(key, value) VALUES ('schema_version', '1')");
    raw.run("INSERT INTO meta(key, value) VALUES ('architecture', '7-16-16-2')");
    raw.run("INSERT INTO meta(key, value) VALUES ('app_version', '1.0.0')");
    raw.run("INSERT INTO meta(key, value) VALUES ('created_at', '2026-11-01 00:00:00')");
    raw.run("INSERT INTO meta(key, value) VALUES ('track_hash', ?)", [DEFAULT_TRACK_HASH]);
    raw.run("INSERT INTO meta(key, value) VALUES ('rng_state', '123456789')");
    raw.run('INSERT INTO generations (generation, drivers_evaluated, best_fitness, avg_fitness, worst_fitness, best_driver_index, mutation_rate, mutation_scale, track_json, track_hash, sim_seconds, rng_state) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
      [1, POPULATION, 1000, 900, 800, 0, MUTATION_RATE, MUTATION_SCALE, DEFAULT_TRACK_JSON, DEFAULT_TRACK_HASH, 105, 1001]);
    raw.run('UPDATE generations SET created_at = X\'0102FF\' WHERE generation = 1');
    for (let i = 0; i < POPULATION; i++) {
      raw.run('INSERT INTO driver_results (generation, driver_index, distance, top_speed, time_alive, crashed, stuck) VALUES (?,?,?,?,?,?,?)',
        [1, i, 500 + i, 300 + i, 10 + i, 0, 0]);
    }
    raw.run("INSERT INTO best_weights (generation, architecture, weights_json) VALUES (?,?,?)", [1, '7-16-16-2', makeValidWeightsString()]);
    const bytes = raw.export();
    raw.close();
    check(bytes, /created_at/i, "arm (d) — created_at BLOB refused (type != 'text')");
  })();

  // (e) canonical sanity — a pristine export of a fresh lineage still ADOPTS.
  const pristine = await freshDb();
  ensureLineageFor(pristine);
  logGen(pristine, 1);
  const okSummary = Db.validateAdoptionBytes(pristine.exportBytes());
  assert(okSummary && okSummary.generation === 1, 'arm (e) — pristine v2 lineage ADOPTS (no over-refusal), got ' + JSON.stringify(okSummary));
  pristine.db.close();
});

// =============================================================================
// FIT-V3 M6a legs 56/57/65/69 — the procedural track generator + rotation
// model (plan FIT-V3 §3/§4; pure logic in js/rotation.js — no DB, no DOM).
// Legs 58/58b/64/66 (and 59–63/67/68/70–73) belong to M6b/M6c and are NOT
// part of this milestone.
// =============================================================================
await leg('leg 56 — generator sanity: rotation/difficulty boundaries, HALFWIDTH/HARMONICS/AMPLITUDE ladders, byte-identical double-run, Track round-trip', () => {
  const { Rotation } = requireRotation();

  // rotationForGeneration — plan D-1 boundary math: gens 1-50 -> 0; every
  // 50-generation block increments; gens 4951-5000 -> 99.
  assert(Rotation.rotationForGeneration(1) === 0, 'rotationForGeneration(1) === 0, got ' + Rotation.rotationForGeneration(1));
  assert(Rotation.rotationForGeneration(50) === 0, 'rotationForGeneration(50) === 0 (end of block 0), got ' + Rotation.rotationForGeneration(50));
  assert(Rotation.rotationForGeneration(51) === 1, 'rotationForGeneration(51) === 1 (first boundary), got ' + Rotation.rotationForGeneration(51));
  assert(Rotation.rotationForGeneration(5000) === 99, 'rotationForGeneration(5000) === 99 (the 5000-generation cap spans 100 blocks), got ' + Rotation.rotationForGeneration(5000));

  // FIT-V6 HALFWIDTH(d) = 50 − 0.25 × d  // 50.0 (d=0) → 25.0 (d=100)
  assert(Rotation.halfWidthForDifficulty(0) === 50, 'halfWidth(0) === 50, got ' + Rotation.halfWidthForDifficulty(0));
  assert(Rotation.halfWidthForDifficulty(20) === 45, 'halfWidth(20) === 45, got ' + Rotation.halfWidthForDifficulty(20));
  assert(Rotation.halfWidthForDifficulty(50) === 37.5, 'halfWidth(50) === 37.5, got ' + Rotation.halfWidthForDifficulty(50));
  assert(Rotation.halfWidthForDifficulty(100) === 25, 'halfWidth(100) === 25, got ' + Rotation.halfWidthForDifficulty(100));

  // HARMONICS(d) = 2 + floor(d/25)  // 2 (0-24) → 6 (≥100)
  assert(Rotation.harmonicsForDifficulty(0) === 2, 'HARMONICS(0) === 2, got ' + Rotation.harmonicsForDifficulty(0));
  assert(Rotation.harmonicsForDifficulty(30) === 3, 'HARMONICS(30) === 3, got ' + Rotation.harmonicsForDifficulty(30));
  assert(Rotation.harmonicsForDifficulty(50) === 4, 'HARMONICS(50) === 4, got ' + Rotation.harmonicsForDifficulty(50));
  assert(Rotation.harmonicsForDifficulty(100) === 6, 'HARMONICS(100) === 6, got ' + Rotation.harmonicsForDifficulty(100));

  // AMPLITUDE(d) = 0.08 + 0.34 × d/100  // 0.08 (d=0) → 0.42 (d=100)
  assert(Rotation.amplitudeForDifficulty(0) === 0.08, 'AMPLITUDE(0) === 0.08, got ' + Rotation.amplitudeForDifficulty(0));
  assert(Rotation.amplitudeForDifficulty(100) === 0.42, 'AMPLITUDE(100) === 0.42, got ' + Rotation.amplitudeForDifficulty(100));

  // Two independent construction runs are byte-identical.
  for (const [seed, rotation] of [[4242, 0], [4242, 1], [7, 37]]) {
    const a = Rotation.generateTrack(seed, rotation);
    const b = Rotation.generateTrack(seed, rotation);
    assert(JSON.stringify(a.points) === JSON.stringify(b.points), 'double-run byte-identical points for seed=' + seed + ' rotation=' + rotation);
    assert(a.halfWidth === b.halfWidth && a.difficulty === b.difficulty, 'double-run halfWidth/difficulty identical for seed=' + seed + ' rotation=' + rotation);
  }

  // Track round-trip: every generated track builds through Track.fromJSON.
  for (let r = 0; r <= 99; r++) {
    const gen = Rotation.generateTrack(4242, r);
    const json = JSON.stringify({ points: gen.points, halfWidth: gen.halfWidth });
    const track = Track.fromJSON(json);
    assert(track.controlPoints.length === 14, 'Track.fromJSON round-trip kept 14 control points (rotation ' + r + ')');
    assert(track.halfWidth === gen.halfWidth, 'Track.fromJSON round-trip kept halfWidth (rotation ' + r + ')');
    assert(track.samples.length > 0, 'Track.fromJSON round-trip produced samples (rotation ' + r + ')');
  }
});

// =============================================================================
// FIT-V3 M6a leg 57 — rotation boundary math + hash pinning: round-trip
// resolveGeneratedTrack for a fixed seed across rotations 0..3 — derivation
// pinned, consecutive rotations produce different hashes, hash stable across
// two invocations, resolver errors throw ValidationError.
// =============================================================================
await leg('leg 57 — rotation boundary math + hash pinning: fixed-seed resolver round-trip (rotations 0..3), hash stability/distinctness, ValidationError arms', () => {
  const { Rotation } = requireRotation();
  const seed = 4242;

  // Derivation pinning: generation -> rotation -> difficulty (plan D-1/D-2).
  const pinned = [
    [1, 0, 0], [50, 0, 0], [51, 1, 1], [100, 1, 1], [101, 2, 2], [151, 3, 3],
  ];
  for (const [g, r, d] of pinned) {
    assert(Rotation.rotationForGeneration(g) === r, 'generation ' + g + ' -> rotation ' + r + ', got ' + Rotation.rotationForGeneration(g));
    assert(Rotation.difficultyForRotation(Rotation.rotationForGeneration(g)) === d, 'generation ' + g + ' -> difficulty ' + d + ', got ' + Rotation.difficultyForRotation(Rotation.rotationForGeneration(g)));
  }

  // Round-trip across rotations 0..3.
  const resolved = [];
  for (let r = 0; r <= 3; r++) {
    const first = Rotation.resolveGeneratedTrack(seed, r);
    const second = Rotation.resolveGeneratedTrack(seed, r);
    assert(Array.isArray(first.controlPoints) && first.controlPoints.length === 14, 'rotation ' + r + ' — 14 control points');
    assert(first.halfWidth === Rotation.halfWidthForDifficulty(Math.min(r, 100)), 'rotation ' + r + ' — halfWidth pinned to HALFWIDTH(d), got ' + first.halfWidth);
    assert(first.difficulty === Math.min(r, 100), 'rotation ' + r + ' — difficulty === min(rotationIndex, 100), got ' + first.difficulty);
    assert(typeof first.trackHash === 'string' && /^[0-9a-f]{16}$/.test(first.trackHash), 'rotation ' + r + ' — trackHash is 16 lowercase hex, got ' + first.trackHash);
    assert(first.trackHash === second.trackHash, 'rotation ' + r + ' — hash stable across two invocations');
    // The hash is pinned to the exact bytes, three ways: the resolver's own
    // value, the shared canonicalJSON/cyrb53 path, and a Track built from
    // the bytes.
    const fromBytes = cyrb53(canonicalJSON({ points: first.controlPoints, halfWidth: first.halfWidth }));
    assert(first.trackHash === fromBytes, 'rotation ' + r + ' — resolver hash === cyrb53(canonicalJSON(bytes))');
    const built = Track.fromJSON(JSON.stringify({ points: first.controlPoints, halfWidth: first.halfWidth }));
    assert(built.hash() === first.trackHash, 'rotation ' + r + ' — Track built from the bytes hashes the same');
    resolved.push(first);
  }

  // Consecutive rotations must not share a hash.
  for (let r = 1; r < resolved.length; r++) {
    assert(resolved[r].trackHash !== resolved[r - 1].trackHash, 'rotation ' + r + ' hash differs from rotation ' + (r - 1));
  }
  assert(new Set(resolved.map((entry) => entry.trackHash)).size === resolved.length, 'all four rotation hashes are distinct');

  // Input errors — ValidationError.
  assertValidationError(() => Rotation.resolveGeneratedTrack(seed, -1), 'resolveGeneratedTrack(rotationIndex=-1) refused');
  assertValidationError(() => Rotation.resolveGeneratedTrack(seed, 1.5), 'resolveGeneratedTrack(rotationIndex=1.5) refused');
  assertValidationError(() => Rotation.resolveGeneratedTrack(seed, NaN), 'resolveGeneratedTrack(rotationIndex=NaN) refused');
  assertValidationError(() => Rotation.generateTrack(seed, -1), 'generateTrack(rotationIndex=-1) refused');
  assertValidationError(() => Rotation.generateTrack(seed, 1.5), 'generateTrack(rotationIndex=1.5) refused');
  assertValidationError(() => Rotation.rotationForGeneration(0), 'rotationForGeneration(0) refused');
  assertValidationError(() => Rotation.rotationForGeneration(2.5), 'rotationForGeneration(2.5) refused');
  assertValidationError(() => Rotation.difficultyForRotation(-1), 'difficultyForRotation(-1) refused');
  assertValidationError(() => Rotation.difficultyForRotation(1.5), 'difficultyForRotation(1.5) refused');
  assertValidationError(() => Rotation.resolveGeneratedTrack(4294967296, 0), 'resolveGeneratedTrack(rotationSeed=2^32) refused');
  assertValidationError(() => Rotation.resolveGeneratedTrack(-1, 0), 'resolveGeneratedTrack(rotationSeed=-1) refused');
  assertValidationError(() => Rotation.resolveGeneratedTrack(1.5, 0), 'resolveGeneratedTrack(rotationSeed=1.5) refused');
});

// =============================================================================
// FIT-V3 M6a leg 65 — dimension sweep: 3 fixed seeds × rotations 1..99
// (99 tracks per seed). Every track builds, the control polygon does not
// self-intersect, zero NaN anywhere, the max radius equals TRACK_MAX_RADIUS
// unconditionally (within the 3-decimal rounding bound), and the resolver's
// hash agrees with Track.hash() and cyrb53(canonicalJSON(bytes)). Difficulty
// is asserted non-decreasing per rotation.
// =============================================================================
await leg('leg 65 — dimension sweep: 3 seeds × rotations 1..99 — every track builds, no self-intersection on the control polygon, zero NaN, max radius == TRACK_MAX_RADIUS (unconditionally), hash agreement, difficulty non-decreasing per rotation', () => {
  const { Rotation } = requireRotation();
  const seeds = [4242, 12648430, 1];
  const CENTER = Rotation.TRACK_CENTER;
  // Independent self-intersection oracle for the control polygon (proper
  // crossings only; adjacent closed pairs excluded — mirrors the generator's
  // contract, written separately so a broken generator check fails here).
  const orient = (ax, ay, bx, by, cx, cy) => (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  const segmentsCross = (ax, ay, bx, by, cx, cy, dx, dy) => {
    const o1 = orient(ax, ay, bx, by, cx, cy);
    const o2 = orient(ax, ay, bx, by, dx, dy);
    const o3 = orient(cx, cy, dx, dy, ax, ay);
    const o4 = orient(cx, cy, dx, dy, bx, by);
    return ((o1 > 0 && o2 < 0) || (o1 < 0 && o2 > 0)) && ((o3 > 0 && o4 < 0) || (o3 < 0 && o4 > 0));
  };
  const polygonSelfIntersects = (points) => {
    const n = points.length;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (j === i + 1) continue; // shared vertex in the closed loop
        if (i === 0 && j === n - 1) continue; // wrap-around adjacency
        const a = points[i], b = points[(i + 1) % n], c = points[j], d = points[(j + 1) % n];
        if (segmentsCross(a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y)) return true;
      }
    }
    return false;
  };
  let builds = 0;
  for (const seed of seeds) {
    let previousDifficulty = -1;
    for (let r = 1; r <= 99; r++) {
      const gen = Rotation.generateTrack(seed, r);
      assert(gen.points.length === 14, 'seed ' + seed + ' rotation ' + r + ' — 14 points');
      let maxDist = 0;
      for (const p of gen.points) {
        assert(Number.isFinite(p.x) && Number.isFinite(p.y), 'seed ' + seed + ' rotation ' + r + ' — finite point');
        assert(Math.abs(p.x) <= 1e6 && Math.abs(p.y) <= 1e6, 'seed ' + seed + ' rotation ' + r + ' — |coordinate| within bounds');
        const d = Math.hypot(p.x - CENTER.x, p.y - CENTER.y);
        if (d > maxDist) maxDist = d;
      }
      // max(r_i) == TRACK_MAX_RADIUS unconditionally: R0 is clamped so the
      // OUTER radius is exactly 245 pre-rounding; the returned points are
      // rounded to 3 decimals, which perturbs any radius by at most
      // sqrt(2)*5e-4 ≈ 7.1e-4. The 1e-3 bound carries exactly that.
      assert(maxDist <= Rotation.TRACK_MAX_RADIUS + 1e-3, 'seed ' + seed + ' rotation ' + r + ' — every point stays within TRACK_MAX_RADIUS + rounding, got ' + maxDist);
      assert(Math.abs(maxDist - Rotation.TRACK_MAX_RADIUS) <= 1e-3, 'seed ' + seed + ' rotation ' + r + ' — max radius == TRACK_MAX_RADIUS within rounding, got ' + maxDist);
      assert(!polygonSelfIntersects(gen.points), 'seed ' + seed + ' rotation ' + r + ' — control polygon self-intersects');
      assert(gen.difficulty === Math.min(r, 100), 'seed ' + seed + ' rotation ' + r + ' — difficulty === min(rotationIndex, 100), got ' + gen.difficulty);
      assert(gen.difficulty >= previousDifficulty, 'seed ' + seed + ' rotation ' + r + ' — difficulty is non-decreasing per rotation (previous ' + previousDifficulty + ')');
      previousDifficulty = gen.difficulty;
      assert(gen.halfWidth === Rotation.halfWidthForDifficulty(gen.difficulty), 'seed ' + seed + ' rotation ' + r + ' — halfWidth === HALFWIDTH(difficulty)');
      // Builds through Track; zero NaN in the sampled spline.
      const track = new Track(gen.points, gen.halfWidth);
      for (const s of track.samples) {
        assert(Number.isFinite(s.x) && Number.isFinite(s.y) && Number.isFinite(s.tangent) && Number.isFinite(s.arcPos), 'seed ' + seed + ' rotation ' + r + ' — NaN/Infinity in track samples');
      }
      // Hash agreement, three ways; stable across two runs.
      const resolved = Rotation.resolveGeneratedTrack(seed, r);
      const direct = cyrb53(canonicalJSON({ points: gen.points, halfWidth: gen.halfWidth }));
      assert(resolved.trackHash === direct, 'seed ' + seed + ' rotation ' + r + ' — resolver hash === cyrb53(canonicalJSON(bytes))');
      assert(track.hash() === direct, 'seed ' + seed + ' rotation ' + r + ' — Track built from the bytes hashes the same');
      assert(Rotation.resolveGeneratedTrack(seed, r).trackHash === resolved.trackHash, 'seed ' + seed + ' rotation ' + r + ' — hash stable across two runs');
      builds += 1;
    }
  }
  console.log('       (leg 65 — ' + builds + ' generated tracks verified across 3 seeds x 99 rotations)');
});

// =============================================================================
// FIT-V3 M6a leg 69 — gen-boundary simulation smoke (no DB): a synthetic
// lineage loop feeds generations 1..60 through the rotation ladder — rotation
// 0 for gens 1..50 (the pinned starting track), rotation 1 for gens 51..60 —
// and pins the acceptance assert the logGeneration boundary arm uses (a
// re-pin is legal only at MAX(generation) ≡ 0 mod 50, with the difficulty
// the rotation ladder dictates).
// =============================================================================
await leg('leg 69 — gen-boundary simulation smoke: synthetic lineage over generations 1..60 — r0 for 1..50, r1 for 51..60; the pin-move acceptance assert', () => {
  const { Rotation } = requireRotation();

  // §7 pin-move acceptance test, distilled to its pure predicates: a re-pin
  // to a different track is legal iff MAX(generation) ≡ 0 (mod 50) or 0
  // AND a tracks row EXISTS for the new hash (always true post-
  // pre-materialization) AND that row's difficulty ==
  // difficultyForRotation(rotationForGeneration(MAX + 1)) — integer equality
  // pins the rotation. (A missing row ⇒ typed refusal ⇒ the caller-toolchain
  // bug arm; the row existing is the DB layer's concern, not this stub's.)
  const pinMoveIsLegal = (maxGeneration, newDifficulty) => {
    if (!(maxGeneration === 0 || maxGeneration % 50 === 0)) return false; // boundary arm
    return newDifficulty === Rotation.difficultyForRotation(Rotation.rotationForGeneration(maxGeneration + 1));
  };

  const seed = 4242;
  // Rotation 0 stands in for the lineage's pinned starting track (the DB
  // layer owns the real pin; M6a is pure). The fake lineage records
  // (generation, rotationIndex, difficulty, trackHash) for 1..60.
  const pin = Rotation.resolveGeneratedTrack(seed, 0);
  const used = [];
  for (let g = 1; g <= 60; g++) {
    const rotation = Rotation.rotationForGeneration(g);
    const resolved = rotation === 0
      ? { controlPoints: pin.controlPoints, halfWidth: pin.halfWidth, difficulty: 0, trackHash: pin.trackHash }
      : Rotation.resolveGeneratedTrack(seed, rotation);
    used.push({ generation: g, rotationIndex: rotation, difficulty: Rotation.difficultyForRotation(rotation), trackHash: resolved.trackHash });
  }

  // Rotation 0 for 1..50; rotation 1 for 51..60.
  for (const entry of used) {
    const expectedRotation = entry.generation <= 50 ? 0 : 1;
    assert(entry.rotationIndex === expectedRotation, 'generation ' + entry.generation + ' uses rotation ' + expectedRotation + ', got ' + entry.rotationIndex);
    assert(entry.difficulty === (entry.generation <= 50 ? 0 : 1), 'generation ' + entry.generation + ' difficulty derived from the rotation ladder');
  }
  // hash(gen 1..50) identical — the pinned starting track is never re-rolled;
  // hash(gen 51..60) identical; the boundary flips the hash (the boundary event).
  const hashAt = (g) => used[g - 1].trackHash;
  for (let g = 2; g <= 50; g++) assert(hashAt(g) === hashAt(1), 'hash identical across 1..50 at generation ' + g);
  for (let g = 52; g <= 60; g++) assert(hashAt(g) === hashAt(51), 'hash identical across 51..60 at generation ' + g);
  assert(hashAt(51) !== hashAt(50), 'hash(gen 51) differs from hash(gen 50) — the boundary event');

  // The pin-move assert: legal at MAX == 50 (gen 51's move), refused elsewhere.
  assert(pinMoveIsLegal(0, 0) === true, 'pin-move at MAX == 0 with difficulty 0 is legal (lineage start)');
  assert(pinMoveIsLegal(50, 1) === true, 'pin-move at MAX ≡ 0 (mod 50) with difficulty 1 is legal (gen 51)');
  assert(pinMoveIsLegal(20, 1) === false, 'pin-move at MAX = 20 (not a boundary) is refused');
  assert(pinMoveIsLegal(49, 1) === false, 'pin-move at MAX = 49 is refused');
  assert(pinMoveIsLegal(50, 2) === false, 'pin-move at MAX = 50 with difficulty 2 (not the rotation ladder) is refused');
  assert(pinMoveIsLegal(50, 0) === false, 'pin-move at MAX = 50 with difficulty 0 (wrong rung) is refused');

  // The full 1..5000 map: floor((g-1)/50) everywhere; the boundary events are
  // exactly the generations g ≡ 1 (mod 50) with g ≥ 51; rotation(5000) == 99;
  // difficulty caps at 100.
  for (let g = 1; g <= 5000; g++) {
    const r = Rotation.rotationForGeneration(g);
    assert(r === Math.floor((g - 1) / 50), 'rotationForGeneration(' + g + ') === floor((g-1)/50)');
    if (g >= 51 && g % 50 === 1) {
      assert(Rotation.rotationForGeneration(g) === Rotation.rotationForGeneration(g - 1) + 1, 'rotation advances exactly at generation ' + g);
    }
  }
  assert(Rotation.rotationForGeneration(5000) === 99, 'rotationForGeneration(5000) === 99 — the cap spans 100 blocks');
  assert(Rotation.difficultyForRotation(100) === 100, 'difficultyForRotation(100) === 100 — difficulty caps at 100');
  assert(Rotation.difficultyForRotation(150) === 100, 'difficultyForRotation(150) === 100 — difficulty caps at 100');
});

// =============================================================================
// FIT-V3 M6b legs 59–63, 67, 68, 70–73 — the v3 DB work: schema v3 fresh init,
// the v2 → v3 retrofit migration (same-call chain from v1), the battery's v3
// arms (self-hash 8b, tracks 10b, five-table scrub, DDL twins), the verified
// set's v3 arms, ensureLineage/forkLineage curriculum pre-materialization, and
// the §7 pin-move boundary arms.
// =============================================================================
await leg('leg 59 — pre-v4 v2 lineage refused on adopt: typed wall-model refusal, byte-preserved, zero migration', async () => {
  const { Db } = requireDb();
  const v2Bytes = buildV2LineageBytes();
  const before = Buffer.from(v2Bytes);
  assertWallModelRefusal(Db, v2Bytes, 'leg 59');
  const db = new Db();
  let threw = null;
  try { await db.init(v2Bytes); } catch (err) { threw = err; }
  assert(threw !== null && threw.name === 'WallModelChangedError', 'leg 59 — init typed refusal, got ' + (threw && threw.name));
  assert(db.db === null, 'leg 59 — zero adoption');
  assert(Buffer.from(v2Bytes).equals(before), 'leg 59 — bytes untouched by the refusal');
});
await leg('leg 60 — pre-v4 chain fixtures refused typed: no v1->v2->v3 chain runs on adoption; bytes untouched', async () => {
  const { Db } = requireDb();
  for (const [label, bytes] of [['v1', buildV1LineageBytes()], ['v2', buildV2LineageBytes()]]) {
    const before = Buffer.from(bytes);
    const err = assertWallModelRefusal(Db, bytes, 'leg 60 ' + label);
    assert(/wall model/i.test(err.message), 'leg 60 ' + label + ' — typed wall-model message');
    assert(Buffer.from(bytes).equals(before), 'leg 60 ' + label + ' — bytes untouched');
  }
});
await leg('leg 61 — v3 clothing: pre-v4 bytes typed-refused; uv==9 generically refused; uv4 shapes adopt', async () => {
  const { Db } = requireDb();
  const check = (bytes, pattern, label) => assertWallModelRefusal(Db, bytes, label);
  const { DDL_GENERATIONS_V2 } = requireDb();
  // (i) uv==3 + schema_version=='3' + the v2 (FIT-V2) generations DDL ⇒ refused at 3c.
  check(buildV3ProbeBytes({ generationsDdl: DDL_GENERATIONS_V2 }), /DDL|does not match/i, 'leg 61(i) — uv==3 with v2-DDL generations refused at 3c');
  // (ii) uv==3 + the FIT-V1 (v1-CHECK) generations DDL ⇒ refused at 3c.
  const v1Ddl = "CREATE TABLE generations(generation INTEGER PRIMARY KEY CHECK (generation >= 1), created_at TEXT NOT NULL DEFAULT (datetime('now')), drivers_evaluated INTEGER NOT NULL CHECK (drivers_evaluated > 0), best_fitness REAL NOT NULL CHECK (best_fitness >= 0), avg_fitness REAL NOT NULL, worst_fitness REAL NOT NULL, best_driver_index INTEGER NOT NULL CHECK (best_driver_index BETWEEN 0 AND 19), mutation_rate REAL NOT NULL, mutation_scale REAL NOT NULL, track_json TEXT NOT NULL, track_hash TEXT NOT NULL, sim_seconds REAL NOT NULL, rng_state INTEGER NOT NULL CHECK (rng_state BETWEEN 0 AND 4294967295));";
  check(buildV3ProbeBytes({ generationsDdl: v1Ddl }), /DDL|does not match/i, 'leg 61(ii) — uv==3 with v1-DDL generations refused at 3c');
  // (iii) uv==3 + schema_version='2' (legacy marker) ⇒ refused at the 3c readback.
  check(buildV3ProbeBytes({ schemaVersion: '2' }), /schema_version/i, "leg 61(iii) — uv==3 with schema_version='2' refused at 3c");
  // (iv) uv==2 + schema_version='1' ⇒ refused at 3c (existing arm, pinned here in the uv matrix).
  check(buildExactTwinV2Bytes({ schemaVersion: '1' }), /schema_version/i, "leg 61(iv) — uv==2 with schema_version='1' refused at 3c");
  // (v) uv==9 ⇒ GENERIC ValidationError (not the typed wall-model break).
  assertBatteryReject(Db, buildV3ProbeBytes({ uv: 9 }), /user_version/i, 'leg 61(v) — uv==9 refused at 3b');
  // (vi) FIT-V6: uv∈{1,2,3} shapes are ALL typed-refused; uv4 adopts.
  for (const [label, bytes] of [
    ['uv1', buildV1LineageBytes()], ['uv2', buildV2LineageBytes()], ['uv3', buildV3ProbeBytes()],
  ]) {
    assertWallModelRefusal(Db, bytes, 'leg 61(vi) ' + label);
  }
  const fresh = await freshDb();
  ensureLineageFor(fresh);
  logGen(fresh, 1);
  const ok4 = Db.validateAdoptionBytes(fresh.exportBytes());
  assert(ok4 && ok4.generation === 1, 'leg 61(vi) — uv4 shape ADOPTS, got ' + JSON.stringify(ok4));
  fresh.db.close();
});

await leg('leg 62 — five-table scrub (uv==3): trigger/view/extra table refused; dangling generations.track_hash refused; difficulty/half_width/source/control_points/track_hash arms refused (craft via ignore_check_constraints — the battery never trusts the .sql)', async () => {
  const { Db } = requireDb();
  const check = (bytes, pattern, label) => assertBatteryReject(Db, bytes, pattern, label);
  // (a) a trigger in a uv==3 file ⇒ refused at the five-table scrub.
  check(await craftV3Bytes((raw) => {
    raw.run('CREATE TRIGGER evil AFTER INSERT ON generations BEGIN SELECT 1; END;');
  }), /trigger/i, 'leg 62(a) — uv==3 file with a trigger refused at the five-table scrub');
  // (b) a view in a uv==3 file ⇒ refused at the five-table scrub.
  check(await craftV3Bytes((raw) => {
    raw.run('CREATE VIEW evil AS SELECT * FROM meta');
  }), /view/i, 'leg 62(b) — uv==3 file with a view refused at the five-table scrub');
  // (c) an extra table in a uv==3 file ⇒ refused at the five-table scrub.
  check(await craftV3Bytes((raw) => {
    raw.run('CREATE TABLE evil(x INTEGER)');
  }), /unexpected|exactly|tables/i, 'leg 62(c) — uv==3 file with an extra table refused at the five-table scrub');
  // (d) dangling generations.track_hash — the generation's pin row is gone.
  // foreign_key_check catches this at Step 3a FIRST (the 10b count arm is
  // belt-and-braces under it — no crafting route on a 3c-honest file reaches
  // it before 3a does); the file is refused either way.
  check(await craftV3Bytes((raw) => {
    raw.run('PRAGMA foreign_keys = OFF');
    raw.run('DELETE FROM tracks WHERE track_hash = ?', [DEFAULT_TRACK_HASH]);
  }), /foreign_key_check|orphan|dangling/i, 'leg 62(d) — generations.track_hash with NO tracks row refused (Step 3a foreign_key_check; the count arm is belt-and-braces)');
  // (e) difficulty = 99.9 (REAL) ⇒ 10b arm 1.
  check(await craftV3Bytes((raw) => {
    raw.run('PRAGMA ignore_check_constraints = ON');
    raw.run('UPDATE tracks SET difficulty = 99.9 WHERE track_hash = ?', [DEFAULT_TRACK_HASH]);
  }), /difficulty/i, 'leg 62(e) — difficulty=99.9 (REAL) refused');
  // (f) half_width = BLOB — INTEGER-vs-BLOB comparisons are always FALSE,
  // so the CHECK fires and this sql.js build's integrity_check reports
  // "CHECK constraint failed in tracks" at step 3 (the arm-2 typeof arm is
  // belt-and-braces for files/builds where IC does not flag it).
  check(await craftV3Bytes((raw) => {
    raw.run('PRAGMA ignore_check_constraints = ON');
    raw.run("UPDATE tracks SET half_width = X'0102' WHERE track_hash = ?", [DEFAULT_TRACK_HASH]);
  }), /CHECK constraint failed in tracks|half_width/i, 'leg 62(f) — half_width BLOB refused');
  // (g) source = 'weird' — TEXT > integer always makes the CHECK fail; this
  // build's integrity_check reports "CHECK constraint failed in tracks" at
  // step 3 (arm 3 is the belt-and-braces twin).
  check(await craftV3Bytes((raw) => {
    raw.run('PRAGMA ignore_check_constraints = ON');
    raw.run("UPDATE tracks SET source = 'weird' WHERE track_hash = ?", [DEFAULT_TRACK_HASH]);
  }), /CHECK constraint failed in tracks|source/i, 'leg 62(g) — source=weird refused');
  // (h) control_points with non-numeric coordinates ⇒ 10b arm 5.
  check(await craftV3Bytes((raw) => {
    raw.run('PRAGMA ignore_check_constraints = ON');
    raw.run("UPDATE tracks SET control_points = '[{\"x\": \"a\"},{\"x\": 1, \"y\": 1},{\"x\": 2, \"y\": 2}]' WHERE track_hash = ?", [DEFAULT_TRACK_HASH]);
  }), /control_points|JSON|point/i, 'leg 62(h) — control_points with non-numeric coordinates refused');
  // (k) control_points with fewer than 3 points ⇒ 10b arm 5.
  check(await craftV3Bytes((raw) => {
    raw.run('PRAGMA ignore_check_constraints = ON');
    raw.run("UPDATE tracks SET control_points = '[{\"x\": 1, \"y\": 1},{\"x\": 2, \"y\": 2}]' WHERE track_hash = ?", [DEFAULT_TRACK_HASH]);
  }), /control_points|point/i, 'leg 62(k) — control_points with fewer than 3 points refused');
  // (i) control_points over 262144 chars ⇒ 10b arm 4.
  const huge = '[' + Array(40000).fill('{"x":1,"y":1}').join(',') + ']';
  assert(huge.length > 262144, 'the huge control_points payload exceeds the 262144 cap');
  check(await craftV3Bytes((raw) => {
    raw.run('PRAGMA ignore_check_constraints = ON');
    raw.run('UPDATE tracks SET control_points = ? WHERE track_hash = ?', [huge, DEFAULT_TRACK_HASH]);
  }), /control_points/i, 'leg 62(i) — control_points over 262144 characters refused');
  // (j) track_hash length > 64 ⇒ 10b arm 6; (j2) blob track_hash ⇒ arm 6.
  // (Generations' track_hash is renamed in lock-step so the FK stays
  // consistent and the file reaches arm 6 instead of dying at Step 3a.)
  check(await craftV3Bytes((raw) => {
    raw.run('PRAGMA ignore_check_constraints = ON');
    raw.run('UPDATE tracks SET track_hash = ? WHERE track_hash = ?', ['f'.repeat(65), DEFAULT_TRACK_HASH]);
    raw.run('UPDATE generations SET track_hash = ? WHERE track_hash = ?', ['f'.repeat(65), DEFAULT_TRACK_HASH]);
  }), /track_hash/i, 'leg 62(j) — track_hash length 65 refused');
  check(await craftV3Bytes((raw) => {
    raw.run('PRAGMA ignore_check_constraints = ON');
    raw.run("UPDATE tracks SET track_hash = X'41424344' WHERE track_hash = ?", [DEFAULT_TRACK_HASH]);
    raw.run("UPDATE generations SET track_hash = X'41424344' WHERE track_hash = ?", [DEFAULT_TRACK_HASH]);
  }), /track_hash/i, 'leg 62(j2) — blob track_hash refused');
  // (j3) track_hash = 42 — TEXT affinity converts it to '42' (typeof text); the
  // row then fails the whole-row recombination arm instead — STILL refused.
  check(await craftV3Bytes((raw) => {
    raw.run('PRAGMA ignore_check_constraints = ON');
    raw.run('UPDATE tracks SET track_hash = 42 WHERE track_hash = ?', [DEFAULT_TRACK_HASH]);
    raw.run('UPDATE generations SET track_hash = 42 WHERE track_hash = ?', [DEFAULT_TRACK_HASH]);
  }), /track_hash|recombin/i, 'leg 62(j3) — track_hash=42 refused (affinity-converted; whole-row recombination arm)');
  // (l) whole-row recombination arm — half_width flipped to 80 (valid range) ⇒ recombined hash ≠ track_hash.
  check(await craftV3Bytes((raw) => {
    raw.run('PRAGMA ignore_check_constraints = ON');
    raw.run('UPDATE tracks SET half_width = 80 WHERE track_hash = ?', [DEFAULT_TRACK_HASH]);
  }), /track_hash|recombin/i, 'leg 62(l) — whole-row recombination arm refused (hash/bytes divergence)');
});

await leg('leg 63 — fresh v4 pin: uv==4, schema_version==4, wall_model=v6, DDL exports by reference, EXACTLY one pin row + 99 generated curriculum rows with difficulty == rotation index', async () => {
  const { Db, DDL_GENERATIONS, DDL_GENERATIONS_NEW, DDL_TRACKS } = requireDb();
  const db = await freshDb();
  ensureLineageFor(db);
  const uv = db.db.exec('SELECT COALESCE(user_version, 0) FROM pragma_user_version');
  assert(Number(uv[0].values[0][0]) === 4, 'fresh DB must carry user_version == 4, got ' + JSON.stringify(uv[0].values));
  const sv = db.db.exec("SELECT value FROM meta WHERE key = 'schema_version'");
  assert(sv[0].values[0][0] === '4', "fresh DB must carry meta.schema_version == '4'");
  const wm = db.db.exec("SELECT value FROM meta WHERE key = 'wall_model'");
  assert(wm[0].values[0][0] === 'v6', "fresh DB must carry meta.wall_model == 'v6'");
  assert(DDL_GENERATIONS.includes('REFERENCES tracks(track_hash)'), 'DDL_GENERATIONS carries the v3 REFERENCES tracks(track_hash) clause');
  assert(DDL_GENERATIONS_NEW === DDL_GENERATIONS.replace('generations', 'generations_new'), 'DDL_GENERATIONS_NEW === DDL_GENERATIONS with generations → generations_new');
  assert(DDL_TRACKS === "CREATE TABLE tracks(track_hash TEXT PRIMARY KEY, source TEXT NOT NULL CHECK (source IN ('default','generated','editor')), control_points TEXT NOT NULL, half_width REAL NOT NULL CHECK (half_width >= 1 AND half_width <= 2000), difficulty INTEGER NOT NULL CHECK (difficulty BETWEEN 0 AND 100), created_at TEXT NOT NULL DEFAULT (datetime('now')));", 'DDL_TRACKS verbatim (Appendix A)');
  assert(countRows(db, 'tracks') === 100, 'fresh lineage holds EXACTLY 100 tracks rows (1 pin + 99 generated), got ' + countRows(db, 'tracks'));
  const pinRows = db.db.exec("SELECT COUNT(*) FROM tracks WHERE source = 'default'")[0].values[0][0];
  assert(pinRows === 1, 'EXACTLY ONE pin row (source=default), got ' + pinRows);
  const pin = db.db.exec("SELECT track_hash, difficulty, half_width FROM tracks WHERE source = 'default'")[0].values[0];
  assert(pin[0] === DEFAULT_TRACK_HASH && pin[1] === 0 && Number(pin[2]) === 70, 'the pin row is the DEFAULT track, difficulty 0, half_width 70, got ' + JSON.stringify(pin));
  const { Rotation } = requireRotation();
  // FIT-V4 — ensureLineageFor() pins a FRESH lineage, which writes
  // generator_version=2 once and pre-materializes the V2 curriculum. The
  // expected hashes must therefore be derived with the explicit V2 resolver.
  for (let r = 1; r <= 99; r++) {
    const resolved = Rotation.resolveGeneratedTrack(123456789, r, 2);
    const row = db.db.exec('SELECT source, difficulty FROM tracks WHERE track_hash = ?', [resolved.trackHash]);
    const values = row && row[0] && row[0].values && row[0].values[0];
    assert(values && values[0] === 'generated' && values[1] === r, 'rotation ' + r + ' row: source=generated, difficulty==' + r + ', got ' + JSON.stringify(values));
  }
  const genCount = db.db.exec("SELECT COUNT(*) FROM tracks WHERE source = 'generated'")[0].values[0][0];
  assert(genCount === 99, 'exactly 99 generated rows, got ' + genCount);
  db.db.close();
});

await leg('leg 67 — pre-v4 stale-hash fixture ⇒ typed wall-model refusal (stale-hash arm enforced on uv4 by legs 62/68)', async () => {
  const { Db } = requireDb();
  const parsed = JSON.parse(DEFAULT_TRACK_JSON);
  parsed.points[0].y += 1;
  const forged = buildExactTwinV2Bytes({ trackJson: JSON.stringify(parsed) });
  assertWallModelRefusal(Db, forged, 'leg 67');
});
await leg('leg 68 — battery refuses a flipped tracks byte at adoption: flipped half_width ⇒ refused at the battery (never re-reaches the verified set), zero adoption; verified-set arms covered by leg 76', async () => {
  const { Db } = requireDb();
  // FIT-V6: source lineage is uv4 (pre-v4 bytes are typed-refused at step 3b
  // before the recombination arm could run).
  const sourceDb = await freshDb();
  ensureLineageFor(sourceDb);
  logGen(sourceDb, 1);
  const persisted = sourceDb.exportBytes();
  sourceDb.db.close();
  const captured = Uint8Array.from(persisted);
  // Flip the pin row's half_width 70 → 80 (a VALID in-range REAL — no arm but
  // the recombination arm can catch this) on a scratch connection.
  const SQL = globalThis.SQL;
  const raw = new SQL.Database(persisted);
  raw.run('UPDATE tracks SET half_width = 80 WHERE track_hash = ?', [DEFAULT_TRACK_HASH]);
  const flipped = raw.export();
  raw.close();
  assert(Buffer.from(persisted).equals(Buffer.from(captured)), 'the captured bytes were not mutated by the flip');
  // Re-adoption of the flipped bytes ⇒ refused at the battery (count/recombination arms) — never reaches the verified set.
  assertBatteryReject(Db, flipped, /track_hash|recombination|recombine|track/i, 'leg 68 — flipped half_width refused at adoption');
  const db2 = new Db();
  let threw = null;
  try {
    await db2.init(flipped);
  } catch (err) {
    threw = err;
  }
  assert(threw !== null, 'leg 68 — init(flipped) must throw');
  assert(db2.db === null, 'leg 68 — ZERO adoption: the failed instance is discarded');
});

await leg('leg 70 — pre-v4 fixture typed-refused; uv4 file WITHOUT a tracks table refused (3c tracks-DDL twin arm); uv4 dangling generations.track_hash refused (count arm)', async () => {
  const { Db } = requireDb();
  // (a) FIT-V6: uv2 bytes are typed-refused before any structural arm runs.
  assertWallModelRefusal(Db, buildV2LineageBytes(), 'leg 70(a)');
  // (b) uv4 file with NO tracks table ⇒ refused (the 3c tracks twin readback fires first).
  const noTracks = await craftV3Bytes((raw) => {
    raw.run('PRAGMA foreign_keys = OFF');
    raw.run('DROP TABLE tracks');
  });
  assertBatteryReject(Db, noTracks, /tracks|DDL|does not match/i, 'leg 70(b) — uv==3 file with no tracks table refused');
  // (c) uv==3 file with dangling generations.track_hash ⇒ refused (foreign_key_check at Step 3a fires first — the count arm is belt-and-braces).
  const dangling = await craftV3Bytes((raw) => {
    raw.run('PRAGMA foreign_keys = OFF');
    raw.run('DELETE FROM tracks WHERE track_hash = ?', [DEFAULT_TRACK_HASH]);
  });
  assertBatteryReject(Db, dangling, /foreign_key_check|orphan|dangling/i, 'leg 70(c) — dangling generations.track_hash refused (Step 3a foreign_key_check)');
});

await leg('leg 71 — uv==3 tracks DDL-twin refusals: PK-less tracks (3a foreign_key_check mismatch), re-typed difficulty, reordered columns, annotated twin, reordered CHECKs — all refused', async () => {
  const { Db } = requireDb();
  // (i) PK-less tracks — the generations FK target is not unique; foreign_key_check throws.
  const noPk = await craftV3Bytes((raw) => {
    raw.run('PRAGMA foreign_keys = OFF');
    raw.run('DROP TABLE tracks');
    raw.run("CREATE TABLE tracks(track_hash TEXT NOT NULL, source TEXT NOT NULL CHECK (source IN ('default','generated','editor')), control_points TEXT NOT NULL, half_width REAL NOT NULL CHECK (half_width >= 1 AND half_width <= 2000), difficulty INTEGER NOT NULL CHECK (difficulty BETWEEN 0 AND 100), created_at TEXT NOT NULL DEFAULT (datetime('now')));");
  });
  assertBatteryReject(Db, noPk, /mismatch|tracks|DDL/i, 'leg 71(i) — PK-less tracks refused');
  // (ii) difficulty re-typed TEXT (PK kept) ⇒ 3c twin mismatch.
  const retyped = await craftV3Bytes((raw) => {
    raw.run('PRAGMA foreign_keys = OFF');
    raw.run('DROP TABLE tracks');
    raw.run("CREATE TABLE tracks(track_hash TEXT PRIMARY KEY, source TEXT NOT NULL CHECK (source IN ('default','generated','editor')), control_points TEXT NOT NULL, half_width REAL NOT NULL CHECK (half_width >= 1 AND half_width <= 2000), difficulty TEXT NOT NULL CHECK (difficulty BETWEEN 0 AND 100), created_at TEXT NOT NULL DEFAULT (datetime('now')));");
  });
  assertBatteryReject(Db, retyped, /tracks|DDL|does not match/i, 'leg 71(ii) — difficulty TEXT refused at 3c');
  // (iii) nominal columns, reordered (created_at before difficulty) ⇒ 3c twin mismatch.
  const reordered = await craftV3Bytes((raw) => {
    raw.run('PRAGMA foreign_keys = OFF');
    raw.run('DROP TABLE tracks');
    raw.run("CREATE TABLE tracks(track_hash TEXT PRIMARY KEY, source TEXT NOT NULL CHECK (source IN ('default','generated','editor')), control_points TEXT NOT NULL, half_width REAL NOT NULL CHECK (half_width >= 1 AND half_width <= 2000), created_at TEXT NOT NULL DEFAULT (datetime('now')), difficulty INTEGER NOT NULL CHECK (difficulty BETWEEN 0 AND 100));");
  });
  assertBatteryReject(Db, reordered, /tracks|DDL|does not match/i, 'leg 71(iii) — reordered columns refused at 3c');
  // (iv) annotated twin (comment inside the CREATE TABLE) ⇒ 3c twin mismatch.
  const annotated = await craftV3Bytes((raw) => {
    raw.run('PRAGMA foreign_keys = OFF');
    raw.run('DROP TABLE tracks');
    raw.run("CREATE TABLE tracks(track_hash TEXT PRIMARY KEY, /* annotation */ source TEXT NOT NULL CHECK (source IN ('default','generated','editor')), control_points TEXT NOT NULL, half_width REAL NOT NULL CHECK (half_width >= 1 AND half_width <= 2000), difficulty INTEGER NOT NULL CHECK (difficulty BETWEEN 0 AND 100), created_at TEXT NOT NULL DEFAULT (datetime('now')));");
  });
  assertBatteryReject(Db, annotated, /tracks|DDL|does not match/i, 'leg 71(iv) — annotated tracks DDL refused at 3c');
  // (v) reordered columns/CHECKs (difficulty first) ⇒ 3c twin mismatch.
  const checkReordered = await craftV3Bytes((raw) => {
    raw.run('PRAGMA foreign_keys = OFF');
    raw.run('DROP TABLE tracks');
    raw.run("CREATE TABLE tracks(track_hash TEXT PRIMARY KEY, difficulty INTEGER NOT NULL CHECK (difficulty BETWEEN 0 AND 100), source TEXT NOT NULL CHECK (source IN ('default','generated','editor')), control_points TEXT NOT NULL, half_width REAL NOT NULL CHECK (half_width >= 1 AND half_width <= 2000), created_at TEXT NOT NULL DEFAULT (datetime('now')));");
  });
  assertBatteryReject(Db, checkReordered, /tracks|DDL|does not match/i, 'leg 71(v) — reordered CHECKs refused at 3c');
});

await leg('leg 72 — difficulty/half_width arms: 99.9 refused; 2147483647 refused; TEXT difficulties refused; honest half_width=81 pin adopts green; half_width BLOB refused', async () => {
  const { Db } = requireDb();
  const check = (bytes, pattern, label) => assertBatteryReject(Db, bytes, pattern, label);
  // (a) difficulty = 99.9 (REAL — satisfies the BETWEEN check, fails typeof-strict) ⇒ arm 1.
  check(await craftV3Bytes((raw) => {
    raw.run('PRAGMA ignore_check_constraints = ON');
    raw.run('UPDATE tracks SET difficulty = 99.9 WHERE track_hash = ?', [DEFAULT_TRACK_HASH]);
  }), /difficulty/i, 'leg 72(a) — difficulty=99.9 (REAL) refused');
  // (b) difficulty = 2147483647 — integer TEXT vs integer comparisons always
  // violate the BETWEEN check (any value > 100 does), so this build's
  // integrity_check reports "CHECK constraint failed in tracks" at step 3 —
  // the file is refused either way; arm 1 pins the typeof-integer-and-range
  // half of the same rule for CHECK-satisfying reals.
  check(await craftV3Bytes((raw) => {
    raw.run('PRAGMA ignore_check_constraints = ON');
    raw.run('UPDATE tracks SET difficulty = 2147483647 WHERE track_hash = ?', [DEFAULT_TRACK_HASH]);
  }), /CHECK constraint failed in tracks|difficulty/i, 'leg 72(b) — difficulty=2147483647 refused');
  // (c) difficulty as TEXT — 'abc' violates the BETWEEN check (TEXT > any
  // integer makes the upper comparison FALSE) ⇒ refused at integrity_check;
  // '50.5' converts to REAL via INTEGER affinity ⇒ refused at arm 1.
  check(await craftV3Bytes((raw) => {
    raw.run('PRAGMA ignore_check_constraints = ON');
    raw.run("UPDATE tracks SET difficulty = 'abc' WHERE track_hash = ?", [DEFAULT_TRACK_HASH]);
  }), /CHECK constraint failed in tracks|difficulty/i, "leg 72(c) — difficulty='abc' (TEXT) refused");
  check(await craftV3Bytes((raw) => {
    raw.run('PRAGMA ignore_check_constraints = ON');
    raw.run("UPDATE tracks SET difficulty = '50.5' WHERE track_hash = ?", [DEFAULT_TRACK_HASH]);
  }), /difficulty/i, "leg 72(c) — difficulty='50.5' (TEXT→REAL) refused");
  // (d) honest half_width=81 — an 81-wide track forked as the pin adopts green (the
  // row is type-REAL 81.0, integral and in range — the arm accepts integer|real).
  const widePoints = DEFAULT_CONTROL_POINTS.map((p) => ({ x: p.x + 5, y: p.y + 5 }));
  const wideTrack = new Track(widePoints, 81);
  const wideJson = wideTrack.toJSON();
  const wideHash = wideTrack.hash();
  const wide = await freshDb();
  wide.forkLineage(wideHash, 4242, wideJson);
  const wideBytes = wide.exportBytes();
  wide.db.close();
  const summary = Db.validateAdoptionBytes(wideBytes);
  assert(summary && summary.trackHash === wideHash, 'leg 72(d) — the honest half_width=81 pin ADOPTS green, got ' + JSON.stringify(summary));
  const wideRaw = new globalThis.SQL.Database(wideBytes);
  const hw = wideRaw.exec('SELECT half_width FROM tracks WHERE track_hash = ?', [wideHash])[0].values[0][0];
  wideRaw.close();
  assert(Number(hw) === 81, 'leg 72(d) — the pin row carries half_width 81, got ' + JSON.stringify(hw));
  // (e) half_width = BLOB — the BETWEEN check compares integer<->BLOB as
  // FALSE, so this build's integrity_check reports "CHECK constraint failed
  // in tracks" at step 3; arm 2 is the belt-and-braces typeof twin.
  check(await craftV3Bytes((raw) => {
    raw.run('PRAGMA ignore_check_constraints = ON');
    raw.run("UPDATE tracks SET half_width = X'0102' WHERE track_hash = ?", [DEFAULT_TRACK_HASH]);
  }), /CHECK constraint failed in tracks|half_width/i, 'leg 72(e) — half_width BLOB refused');
});

await leg('leg 73 — pin-move arms: off-boundary re-pin refused (no ensureLineage call); recombination arm at the log path; the gen 50 → 51 boundary commit moves meta.track_hash INSIDE the single logging transaction', async () => {
  const { Db } = requireDb();
  const { Rotation } = requireRotation();
  const rot1 = Rotation.resolveGeneratedTrack(123456789, 1, 2); // fresh uv4 lineage is generator_version=2
  const rot1Json = JSON.stringify({ points: rot1.controlPoints, halfWidth: rot1.halfWidth });
  // (a) off-boundary re-pin — meta.track_hash != payload.track_hash at gen 4 (MAX=3).
  const db = await freshDb();
  ensureLineageFor(db);
  logGen(db, 1);
  logGen(db, 2);
  logGen(db, 3);
  let threwA = null;
  try {
    logGen(db, 4, { trackJson: rot1Json, trackHash: rot1.trackHash });
  } catch (err) {
    threwA = err;
  }
  assert(threwA && threwA.name === 'ValidationError', '(a) — off-boundary re-pin refused with ValidationError, got ' + (threwA && (threwA.name + ': ' + threwA.message)));
  assert(/boundary|mismatch/i.test(threwA.message), '(a) — the refusal names the boundary rule, got: ' + threwA.message);
  assert(db.listGenerations().join() === '1,2,3', '(a) — NOTHING was logged (refused before BEGIN)');
  // (b) recombination arm — track_json whose recombined hash != payload.track_hash refused.
  const parsed = JSON.parse(DEFAULT_TRACK_JSON);
  parsed.points[0].y += 1;
  const tamperedJson = JSON.stringify(parsed);
  let threwB = null;
  try {
    logGen(db, 4, { trackJson: tamperedJson, trackHash: DEFAULT_TRACK_HASH });
  } catch (err) {
    threwB = err;
  }
  assert(threwB && threwB.name === 'ValidationError', '(b) — hash/bytes divergence refused at the log path, got ' + (threwB && threwB.name));
  assert(/recombin|recompute|hash/i.test(threwB.message), '(b) — the refusal names the recombination arm, got: ' + threwB.message);
  assert(db.listGenerations().join() === '1,2,3', '(b) — NOTHING was logged (refused before BEGIN)');
  db.db.close();
  // (c) in-transaction pin-move at the gen 50 → 51 boundary.
  // FIT-V6: boundary guard lives on a fresh uv4 lineage (pre-v4 fixtures are
  // typed-refused at step 3b and never reach the log path).
  const dbC = await freshDb();
  ensureLineageFor(dbC); // uv4; curriculum pre-materialized; pin on the default track
  logGen(dbC, 1);
  logGen(dbC, 2);
  for (let g = 3; g <= 50; g++) logGen(dbC, g);
  const hashAtPin = dbC._metaValue('track_hash');
  assert(hashAtPin === DEFAULT_TRACK_HASH, '(c) — generations 1..50 stay on the pinned track, got ' + hashAtPin);
  const uvC = Number(dbC.db.exec('SELECT COALESCE(user_version, 0) FROM pragma_user_version')[0].values[0][0]);
  assert(uvC === 4, '(c) — the lineage is uv4, got uv=' + uvC);
  // Boundary generation — the rotation-1 payload is legal iff the tracks row exists and difficulty == 1.
  logGen(dbC, 51, { trackJson: rot1Json, trackHash: rot1.trackHash, rngState: 9001 });
  const metaHash = dbC._metaValue('track_hash');
  assert(metaHash === rot1.trackHash, '(c) — meta.track_hash moved to the rotation-1 hash INSIDE the logging transaction, got ' + metaHash);
  const tip = dbC.db.exec('SELECT track_hash FROM generations WHERE generation = 51')[0].values[0][0];
  assert(tip === rot1.trackHash, '(c) — generations(MAX).track_hash == meta.track_hash at the boundary, got ' + tip);
  const rs = dbC.resumeState();
  assert(rs.maxGeneration === 51 && rs.trackHash === rot1.trackHash, '(c) — resumeState pins the boundary hash, got ' + JSON.stringify({ maxGeneration: rs.maxGeneration, trackHash: rs.trackHash }));
  assert(rs.rotationIndex === 1 && rs.difficulty === 1 && rs.trackSource === 'generated', '(c) — resumeState gains {rotationIndex:1, difficulty:1, trackSource:\'generated\'}, got ' + JSON.stringify({ rotationIndex: rs.rotationIndex, difficulty: rs.difficulty, trackSource: rs.trackSource }));
  assert(rs.rotationSeed === 123456789, '(c) — resumeState.rotationSeed round-trips the adoption-time cursor, got ' + JSON.stringify(rs.rotationSeed));

  // (d) strengthened pin-move arm — a tracks row of the right difficulty but
  // WRONG hash (a stale-seed row from another lineage's fork retention) is
  // REFUSED at the boundary. craft: a resolveGeneratedTrack(999999, 1) row
  // (difficulty 1, source 'generated') sits in the table; at gen 51 the
  // resolver-enforced arm must refuse (row hash ≠ resolver output).
  const decoy = Rotation.resolveGeneratedTrack(999999, 1);
  const dbD = await freshDb();
  ensureLineageFor(dbD);
  for (let g = 1; g <= 50; g++) logGen(dbD, g);
  const decoyBytes = JSON.stringify({ points: decoy.controlPoints, halfWidth: decoy.halfWidth });
  dbD.db.run("INSERT INTO tracks (track_hash, source, control_points, half_width, difficulty, created_at) VALUES (?, ?, ?, ?, ?, ?)",
    [decoy.trackHash, 'generated', JSON.stringify(decoy.controlPoints), decoy.halfWidth, 1, '2026-01-01 00:00:00']);
  let threwD = null;
  try {
    logGen(dbD, 51, { trackJson: decoyBytes, trackHash: decoy.trackHash, rngState: 9002 });
  } catch (err) {
    threwD = err;
  }
  assert(threwD && threwD.name === 'ValidationError', '(d) — same-difficulty stale-seed row REFUSED at the boundary (resolver equality), got ' + (threwD && (threwD.name + ': ' + threwD.message)));
  assert(/resolver|byte-verified/i.test(threwD.message), '(d) — refusal names the resolver equality arm, got: ' + threwD.message);
  assert(dbD._metaValue('track_hash') === DEFAULT_TRACK_HASH, '(d) — meta.track_hash unchanged (refused BEFORE BEGIN, no re-pin)');
  assert(dbD.listGenerations().length === 50, '(d) — gen 51 was NOT logged (nothing moved)');
  dbD.db.close();

  // (e) boundary-bypass regression — a payload carrying the CURRENT PIN at a
  // boundary is REFUSED. (The old arm required a MISMATCH; the pin passed
  // silently there, stalling the curriculum forever — adversarial cycle-2 F-3.)
  const dbE = await freshDb();
  ensureLineageFor(dbE);
  for (let g = 1; g <= 50; g++) logGen(dbE, g);
  const pinHash = dbE._metaValue('track_hash');
  let threwE = null;
  try {
    logGen(dbE, 51, { trackJson: DEFAULT_TRACK_JSON, trackHash: pinHash, rngState: 7001 });
  } catch (err) {
    threwE = err;
  }
  assert(threwE && threwE.name === 'ValidationError', '(e) — the CURRENT PIN payload at gen 51 throws ValidationError, got ' + (threwE && (threwE.name + ': ' + threwE.message)));
  assert(/resolver|boundary/i.test(threwE.message), '(e) — the refusal names the resolver/boundary rule, got: ' + threwE.message);
  assert(dbE._metaValue('track_hash') === pinHash, '(e) — meta.track_hash unchanged (no re-pin, no stall)');
  assert(dbE.listGenerations().length === 50, '(e) — gen 51 was NOT logged');
  dbE.db.close();
  dbC.db.close();
});

// =============================================================================
// FIT-V3 M6c legs 58 / 58b / 64 / 66 — the training runtime + UI plumbing:
// resume fidelity across the rotation boundary (fresh + EDITOR-pinned), the
// cross-boundary-after-resume swim (LSG tags carry rotationIndex), fork
// under rotation, and the editor-fork legacy retrofit preserving its pinned
// track with rotation firing at 51. Shared local helpers live in the legs
// that use them (the suite keeps no hidden helpers).
// =============================================================================
await leg('leg 58 — resume fidelity across the rotation boundary: gen 51 pin-moves meta INSIDE the logging transaction; resumeState pins {rotationIndex:1, difficulty:1, trackSource:generated, rotationSeed}; the stored tracks row rebuilds byte-identically; a resumed Trainer evaluates generation 52 on the rotation-1 track; EDITOR-pinned lineage identical; the missing-pin adoption arm refuses (validateAdoptionBytes)', async () => {
  const { Db } = requireDb();
  const { Rotation } = requireRotation();
  const seed = 123456789; // the pin seed ensureLineageFor() uses
  // FIT-V4 — this is a FRESH lineage (ensureLineageFor/ensureLineage), so it
  // writes generator_version=2 and pre-materializes V2 curriculum rows. All
  // expected rotation hashes/payloads below must use the explicit V2 resolver.
  const rotPayloadJson = (r) => {
    const resolved = Rotation.resolveGeneratedTrack(seed, r, 2);
    return JSON.stringify({ points: resolved.controlPoints, halfWidth: resolved.halfWidth });
  };
  const rot1 = Rotation.resolveGeneratedTrack(seed, 1, 2);

  // ---- (a) fresh default-pin lineage: gens 1..50 on the pin, gen 51 across the boundary.
  const db = await freshDb();
  ensureLineageFor(db); // pins DEFAULT_TRACK_HASH with rng_state/rotation_seed = 123456789
  for (let g = 1; g <= 50; g++) logGen(db, g);
  logGen(db, 51, { trackJson: rotPayloadJson(1), trackHash: rot1.trackHash, rngState: 5051 });
  const rs = db.resumeState();
  assert(rs.maxGeneration === 51, '(a) — maxGeneration is the tip (51), got ' + rs.maxGeneration);
  assert(rs.rotationIndex === 1, '(a) — rotationIndex === 1 at gen 51, got ' + rs.rotationIndex);
  assert(rs.difficulty === 1, '(a) — difficulty === 1 at the first rung, got ' + rs.difficulty);
  assert(rs.trackSource === 'generated', "(a) — trackSource === 'generated' at gen 51, got " + rs.trackSource);
  assert(rs.rotationSeed === seed, '(a) — rotationSeed round-trips the pin seed, got ' + JSON.stringify(rs.rotationSeed));
  assert(rs.generatorVersion === 2, '(a) — a FRESH lineage carries generator version 2, got ' + JSON.stringify(rs.generatorVersion));
  assert(rs.trackHash === rot1.trackHash, '(a) — resumeState pins the rotation-1 hash');
  assert(db._metaValue('track_hash') === rot1.trackHash, '(a) — meta.track_hash moved at the gen 50 → 51 boundary (the pin-move invariant)');
  const tip = db.db.exec('SELECT track_hash FROM generations WHERE generation = 51')[0].values[0][0];
  assert(tip === rot1.trackHash, '(a) — meta.track_hash == generations(MAX).track_hash at the boundary');
  for (const g of [1, 25, 50]) {
    const h = db.db.exec('SELECT track_hash FROM generations WHERE generation = ?', [g])[0].values[0][0];
    assert(h === DEFAULT_TRACK_HASH, '(a) — generation ' + g + ' logged on the pinned track');
  }
  // The stored tracks row rebuilds the pin byte-identically (D-5).
  const stored = db.db.exec('SELECT control_points, half_width, difficulty, source FROM tracks WHERE track_hash = ?', [rot1.trackHash]);
  assert(stored && stored[0] && stored[0].values.length === 1, '(a) — the rotation-1 tracks row exists (pre-materialized at ensureLineage)');
  const pinnedTrack = Track.fromJSON(JSON.stringify({ points: JSON.parse(stored[0].values[0][0]), halfWidth: stored[0].values[0][1] }));
  assert(pinnedTrack.hash() === rs.trackHash, '(a) — Track.fromJSON(stored row) re-hashes to rs.trackHash');
  assert(Number(stored[0].values[0][2]) === 1 && String(stored[0].values[0][3]) === 'generated', '(a) — stored row carries difficulty 1 / source generated');

  // ---- (b) end-to-end: a Trainer resumed at gen 51 evaluates generation 52 on the
  //          rotation-1 track (byte-identical re-derivation from the handshake seed).
  const captured = [];
  const makeRotationLogger = (database) => ({
    logGeneration(payload) {
      const results = payload.results.map((r) => ({
        distance: r.distance,
        topSpeed: r.topSpeed,
        timeAlive: r.timeAlive,
        crashed: r.crashed ? 1 : 0,
        stuck: r.stuck ? 1 : 0,
      }));
      let worst = Infinity, distSum = 0, timeSum = 0;
      for (const r of results) {
        if (r.distance < worst) worst = r.distance;
        distSum += r.distance;
        timeSum += r.timeAlive;
      }
      const ri = Rotation.rotationForGeneration(payload.generation);
      const stamp = ri === 0
        ? { trackJson: DEFAULT_TRACK_JSON, trackHash: DEFAULT_TRACK_HASH }
        : (() => {
            const resolved = Rotation.resolveGeneratedTrack(seed, ri, 2);
            return { trackJson: JSON.stringify({ points: resolved.controlPoints, halfWidth: resolved.halfWidth }), trackHash: resolved.trackHash };
          })();
      database.logGeneration({
        generation: payload.generation,
        driversEvaluated: results.length,
        bestFitness: results[payload.bestIndex].distance,
        avgFitness: distSum / results.length,
        worstFitness: worst,
        bestDriverIndex: payload.bestIndex,
        mutationRate: MUTATION_RATE,
        mutationScale: MUTATION_SCALE,
        trackJson: stamp.trackJson,
        trackHash: stamp.trackHash,
        simSeconds: timeSum,
        rngState: payload.rngState,
        results,
        bestWeightsJson: payload.bestWeights,
      });
      captured.push(payload);
    },
  });
  const trainer = new Trainer(pinnedTrack, {
    lineage: {
      resume: { bestWeights: rs.bestWeights, rngState: rs.rngState, generation: 51 },
      rotationSeed: rs.rotationSeed,
      generatorVersion: rs.generatorVersion,
      trackHash: rs.trackHash,
      expectedTrackHash: rs.trackHash,
    },
    logger: makeRotationLogger(db),
    onBanner() {},
  });
  trainer.start(52);
  await waitFor(() => trainer.state === 'done', 'resumed run reaches done', 180000);
  assert(captured.length === 1 && captured[0].generation === 52, '(b) — exactly generation 52 was logged, got ' + JSON.stringify(captured.map((p) => p.generation)));
  assert(captured[0].rotationIndex === 1, '(b) — generation 52 evaluates on rotation 1, got ' + JSON.stringify(captured[0].rotationIndex));
  assert(captured[0].difficulty === 1, '(b) — generation 52 carries difficulty 1, got ' + JSON.stringify(captured[0].difficulty));
  assert(captured[0].trackHash === rot1.trackHash, '(b) — generation 52 evaluates on the rotation-1 track (the stored-row identity), got ' + JSON.stringify(captured[0].trackHash));
  assert(db._metaValue('track_hash') === rot1.trackHash, '(b) — gen 52 (rotation 1) stays on the pin — no boundary move inside the block');
  db.db.close();

  // ---- (c) EDITOR-pinned lineage behaves identically (gens 1..50 on the editor pin).
  const editorPoints = DEFAULT_CONTROL_POINTS.map((p) => ({ x: p.x + 13.5, y: p.y - 7.25 }));
  const editorTrack = new Track(editorPoints, 70);
  const editorJson = editorTrack.toJSON();
  const editorHash = editorTrack.hash();
  assert(editorHash !== DEFAULT_TRACK_HASH, '(c) — editor fixture geometry is distinct from the default pin');
  const db2 = await freshDb();
  db2.ensureLineage(editorHash, seed, editorJson);
  for (let g = 1; g <= 50; g++) logGen(db2, g, { trackJson: editorJson, trackHash: editorHash });
  logGen(db2, 51, { trackJson: rotPayloadJson(1), trackHash: rot1.trackHash, rngState: 5051 });
  const rs2 = db2.resumeState();
  assert(rs2.rotationIndex === 1 && rs2.difficulty === 1 && rs2.trackSource === 'generated' && rs2.rotationSeed === seed, '(c) — the editor-pinned lineage resumes at rotation 1 / difficulty 1 / generated / the pin seed, got ' + JSON.stringify({ rotationIndex: rs2.rotationIndex, difficulty: rs2.difficulty, trackSource: rs2.trackSource, rotationSeed: rs2.rotationSeed }));
  assert(rs2.trackHash === rot1.trackHash, '(c) — the editor-pinned lineage pins the rotation-1 hash at gen 51');
  for (const g of [1, 25, 50]) {
    const h = db2.db.exec('SELECT track_hash FROM generations WHERE generation = ?', [g])[0].values[0][0];
    assert(h === editorHash, '(c) — generation ' + g + ' logged on the editor pin');
  }
  const pinRow = db2.db.exec("SELECT source, difficulty FROM tracks WHERE track_hash = ?", [editorHash])[0].values[0];
  assert(String(pinRow[0]) === 'editor' && Number(pinRow[1]) === 0, '(c) — the editor pin row carries source=editor / difficulty 0');
  db2.db.close();

  // ---- (d) RESUME-REFUSE arm — a synthetic file with the pin's tracks row REMOVED
  //          never reaches the resume path (adoption refuses first).
  const poisoned = await craftV3Bytes((raw) => {
    raw.run('PRAGMA ignore_check_constraints = ON');
    raw.run('DELETE FROM tracks WHERE track_hash = ?', [DEFAULT_TRACK_HASH]);
  });
  let refused = null;
  try {
    Db.validateAdoptionBytes(poisoned);
  } catch (err) {
    refused = err;
  }
  assert(refused && refused.name === 'ValidationError', '(d) — the battery refused the dangling-pin file, got ' + (refused ? refused.message : 'NOTHING (adoption succeeded)'));
  assert(/foreign_key_check|NO tracks row/i.test(refused.message), '(d) — the refusal names the dangling pin (a child references a missing tracks row), got: ' + refused.message);
});

await leg('leg 58b — cross-boundary-after-resume: the swim from gen 1 crosses 50→51 and 100→101 in-process; hash(gen 101) === resolveGeneratedTrack(seed, 2); resumeState at 101 yields rotationIndex 2; listGenerationSummaries carries rotationIndex; LSG + coherence tags carry rotationIndex at both boundaries; gens 102..110 stay on rotation 2', async () => {
  const { Db } = requireDb();
  const { Rotation } = requireRotation();
  const seed = 123456789;
  // FIT-V4 — FRESH lineage (ensureLineageFor) ⇒ generator_version=2 with V2
  // curriculum rows; derive every expected boundary hash with the V2 resolver.
  const rot1 = Rotation.resolveGeneratedTrack(seed, 1, 2);
  const rot2 = Rotation.resolveGeneratedTrack(seed, 2, 2);
  const rotPayloadJson = (r) => {
    const resolved = Rotation.resolveGeneratedTrack(seed, r, 2);
    return JSON.stringify({ points: resolved.controlPoints, halfWidth: resolved.halfWidth });
  };
  const db = await freshDb();
  ensureLineageFor(db);
  let bytes51 = null;
  for (let g = 1; g <= 101; g++) {
    if (g <= 50) {
      logGen(db, g);
    } else {
      const ri = Rotation.rotationForGeneration(g);
      logGen(db, g, { trackJson: rotPayloadJson(ri), trackHash: Rotation.resolveGeneratedTrack(seed, ri, 2).trackHash, rngState: 1000 + g });
    }
    if (g === 51) bytes51 = db.exportBytes(); // mid-swim snapshot for the first-boundary tag check
  }
  const gen50 = db.db.exec('SELECT track_hash FROM generations WHERE generation = 50')[0].values[0][0];
  const gen51 = db.db.exec('SELECT track_hash FROM generations WHERE generation = 51')[0].values[0][0];
  const gen100 = db.db.exec('SELECT track_hash FROM generations WHERE generation = 100')[0].values[0][0];
  const gen101 = db.db.exec('SELECT track_hash FROM generations WHERE generation = 101')[0].values[0][0];
  assert(gen50 === DEFAULT_TRACK_HASH, '58b — gen 50 rides the default pin');
  assert(gen51 === rot1.trackHash && gen51 !== gen50, '58b — gen 51 crossed into rotation 1');
  assert(gen100 === rot1.trackHash, '58b — gen 100 (last of rotation 1) still hashes rotation 1');
  assert(gen101 === rot2.trackHash, '58b — hash(gen 101) === resolveGeneratedTrack(seed, 2).trackHash, got ' + gen101);
  assert(db._metaValue('track_hash') === rot2.trackHash, '58b — meta.track_hash moved to rotation 2 at the 100 → 101 boundary (pin-move INSIDE the logging transaction)');
  // resumeState at the tip (maxGen = 101) — floor((101 - 1) / 50) === 2.
  const rs101 = db.resumeState();
  assert(rs101.maxGeneration === 101, '58b — maxGeneration is 101, got ' + rs101.maxGeneration);
  assert(rs101.rotationIndex === 2, '58b — resumeState at 101 yields rotationIndex 2, got ' + rs101.rotationIndex);
  assert(rs101.difficulty === 2, '58b — resumeState at 101 yields difficulty 2, got ' + rs101.difficulty);
  assert(rs101.trackSource === 'generated' && rs101.rotationSeed === seed, '58b — resumeState at 101 yields trackSource generated / the pin seed');
  assert(rs101.trackHash === rot2.trackHash, '58b — resumeState at 101 pins the rotation-2 hash');
  // listGenerationSummaries carries rotationIndex on the joined rows.
  const summaries = db.listGenerationSummaries(3, 0);
  assert(summaries.length === 3 && summaries.map((s) => s.generation).join(',') === '101,100,99', '58b — listGenerationSummaries(3, 0) returns gens 101,100,99, got ' + JSON.stringify(summaries.map((s) => s.generation)));
  assert(summaries[0].rotationIndex === 2 && Number(summaries[0].difficulty) === 2, '58b — gen 101 summary carries rotationIndex 2 / difficulty 2, got ' + JSON.stringify({ rotationIndex: summaries[0].rotationIndex, difficulty: summaries[0].difficulty }));
  assert(summaries[1].rotationIndex === 1 && Number(summaries[1].difficulty) === 1, '58b — gen 100 summary carries rotationIndex 1 / difficulty 1, got ' + JSON.stringify({ rotationIndex: summaries[1].rotationIndex, difficulty: summaries[1].difficulty }));
  // Coherence tags carry rotationIndex at BOTH boundaries.
  assert(db._coherenceTag(bytes51).rotationIndex === 1, '58b — the coherence tag at the 50 → 51 boundary carries rotationIndex 1');
  const bytes101 = db.exportBytes();
  assert(db._coherenceTag(bytes101).rotationIndex === 2, '58b — the coherence tag at the 100 → 101 boundary carries rotationIndex 2');
  // LSG tag for boundary generations carries rotationIndex — through the real persist().
  const banners = [];
  const idbMap = new Map();
  db._idbGet = async (key) => ({ ok: true, value: idbMap.has(key) ? idbMap.get(key) : null });
  db._idbPut = async (key, value) => { idbMap.set(key, value); return { ok: true }; };
  db._idbDelete = async (key) => { idbMap.delete(key); return { ok: true }; };
  db.setPersistenceHooks({ onBanner(b) { banners.push(b); }, persistSkipped: () => false });
  const persisted = await db.persist();
  assert(persisted && persisted.ok === true, '58b — persist at gen 101 succeeds (IDB-fallback, no file handle), got ' + JSON.stringify(persisted));
  const lsg = idbMap.get('car_racer.lastKnownGood');
  assert(lsg && lsg.tag && lsg.tag.maxGeneration === 101, '58b — the LSG tag names gen 101, got ' + JSON.stringify(lsg && lsg.tag));
  assert(lsg && lsg.tag && lsg.tag.rotationIndex === 2, '58b — the LSG tag at the boundary carries rotationIndex 2, got ' + JSON.stringify(lsg && lsg.tag));
  assert(!banners.some((b) => b.type === 'db-not-persisting' || b.type === 'db-verify-failed'), '58b — no file-path failure banner, got ' + JSON.stringify(banners));
  // Cross into the next block: gens 102..110 are still rotation 2 (same block).
  for (let g = 102; g <= 110; g++) {
    logGen(db, g, { trackJson: rotPayloadJson(2), trackHash: rot2.trackHash, rngState: 1000 + g });
  }
  const rs110 = db.resumeState();
  assert(rs110.maxGeneration === 110 && rs110.rotationIndex === 2 && rs110.trackHash === rot2.trackHash, '58b — gen 110 stays on rotation 2 (the block runs to gen 100-aligned boundaries only), got ' + JSON.stringify({ maxGeneration: rs110.maxGeneration, rotationIndex: rs110.rotationIndex }));
  const tip110 = db.db.exec('SELECT track_hash FROM generations WHERE generation = 110')[0].values[0][0];
  assert(tip110 === rot2.trackHash, '58b — gen 110 still hashes rotation 2');
  db.db.close();
});

await leg('leg 64 — fork under rotation: forkLineage wipes generations, retains + extends the tracks rows, re-pins an EDITOR track as rotation 0 (difficulty 0) and pre-materializes the fork curriculum from the FORK seed; generation 1 on the editor pin; the fork rotates to rotation 1 at gen 51', async () => {
  const { Db } = requireDb();
  const { Rotation } = requireRotation();
  const seed = 123456789;
  const forkSeed = 987654321; // fixed fork seed — deterministic across runs
  // Distinct editor geometry: the default loop translated by a nonzero offset (valid loop, distinct hash).
  const editorPoints = DEFAULT_CONTROL_POINTS.map((p) => ({ x: p.x + 13.5, y: p.y - 7.25 }));
  const editorTrack = new Track(editorPoints, 70);
  const editorJson = editorTrack.toJSON();
  const editorHash = editorTrack.hash();
  assert(editorHash !== DEFAULT_TRACK_HASH, '64 — editor fixture geometry is distinct from the default pin');

  const db = await freshDb();
  ensureLineageFor(db); // 100 tracks rows: 1 default pin + 99 generated (seed 123456789)
  assert(countRows(db, 'tracks') === 100, '64 — the fresh lineage holds exactly the 100 curriculum rows, got ' + countRows(db, 'tracks'));
  logGen(db, 1);
  logGen(db, 2);
  logGen(db, 3);
  const countBefore = countRows(db, 'tracks');
  db.forkLineage(editorHash, forkSeed, editorJson);
  assert(countRows(db, 'generations') === 0, '64 — the fork wiped every generation');
  const countAfter = countRows(db, 'tracks');
  assert(countAfter >= countBefore + 1, '64 — the fork retained the old curriculum and added the editor pin (+ the fork curriculum), got ' + countAfter);
  // Exact-total arm: the pre-fork lineage holds 100 curriculum rows (1 default
  // pin + 99 generated rows for the original seed). The fork adds its pin row
  // plus 99 generated rows for the FORK seed (SELECT-then-split in the same
  // transaction). Nothing is deleted, so the total is exactly 200.
  assert(countAfter === 200, '64 — after the fork, tracks holds exactly 100 (pre-fork) + 100 (fork) = 200 rows, got ' + countAfter);
  // The fork's OWN curriculum is fully materialized: for every r in 1..99 a
  // source='generated' row exists whose track_hash == resolveGeneratedTrack(forkSeed, r)
  // and whose difficulty == r (the ladder is the mandate — kill difficulty/swap bugs).
  const generatedRows = new Map();
  const generatedResult = db.db.exec("SELECT track_hash, difficulty FROM tracks WHERE source = 'generated'");
  for (const row of (generatedResult && generatedResult[0] ? generatedResult[0].values : [])) {
    generatedRows.set(String(row[0]), Number(row[1]));
  }
  // FIT-V4 — forkLineage pins a FRESH fork lineage: it writes
  // generator_version=2 and pre-materializes V2 rows from the FORK seed. The
  // retained pre-fork rows were also V2 (fresh ensureLineageFor above). Both
  // expectation loops therefore use the explicit V2 resolver.
  for (let r = 1; r <= 99; r++) {
    const resolved = Rotation.resolveGeneratedTrack(forkSeed, r, 2);
    assert(generatedRows.get(resolved.trackHash) === r, '64 — the fork curriculum row for rotation ' + r + ' hashes resolveGeneratedTrack(forkSeed, ' + r + ') at difficulty ' + r + ', got ' + JSON.stringify(generatedRows.get(resolved.trackHash)));
  }
  // The ORIGINAL seed's curriculum survived the fork (retained per D-4/fork semantics).
  for (const r of [1, 50, 99]) {
    const resolved = Rotation.resolveGeneratedTrack(seed, r, 2);
    assert(generatedRows.get(resolved.trackHash) === r, '64 — the pre-fork curriculum row for rotation ' + r + ' (original seed) survives the fork');
  }
  // Exactly one editor-source row exists and it hashes the editor pin at difficulty 0.
  const editorRows = db.db.exec("SELECT track_hash, difficulty FROM tracks WHERE source = 'editor'");
  assert(editorRows && editorRows[0] && editorRows[0].values.length === 1, '64 — exactly one editor-source tracks row survives the fork, got ' + JSON.stringify(editorRows && editorRows[0] && editorRows[0].values));
  assert(editorRows[0].values[0][0] === editorHash && Number(editorRows[0].values[0][1]) === 0, '64 — the editor pin row hashes the editor track at difficulty 0');
  // Generation 1 on the editor pin — difficulty 0, the fork seed, no new tracks rows.
  logGen(db, 1, { trackJson: editorJson, trackHash: editorHash });
  assert(countRows(db, 'tracks') === countAfter, '64 — logging generation 1 on the editor pin inserts no new tracks rows (the pin is pre-materialized)');
  const rs = db.resumeState();
  assert(rs.rotationIndex === 0, '64 — the fork starts at rotation 0, got ' + rs.rotationIndex);
  assert(rs.difficulty === 0, '64 — the editor pin is difficulty 0, got ' + rs.difficulty);
  assert(rs.trackSource === 'editor', "64 — the fork's rotation-0 source is 'editor', got " + rs.trackSource);
  assert(rs.rotationSeed === forkSeed, '64 — the fork carries its OWN rotation_seed (not the pre-fork one), got ' + JSON.stringify(rs.rotationSeed));
  assert(rs.trackHash === editorHash, '64 — the fork pins the editor hash');
  // The fork trains on the editor pin through gen 50; rotation 1 fires at gen 51 from the FORK seed.
  for (let g = 2; g <= 50; g++) logGen(db, g, { trackJson: editorJson, trackHash: editorHash });
  const tip50 = db.db.exec('SELECT track_hash FROM generations WHERE generation = 50')[0].values[0][0];
  assert(tip50 === editorHash, '64 — generations 1..50 of the fork ride the editor pin');
  const rot1Fork = Rotation.resolveGeneratedTrack(forkSeed, 1, 2);
  logGen(db, 51, { trackJson: JSON.stringify({ points: rot1Fork.controlPoints, halfWidth: rot1Fork.halfWidth }), trackHash: rot1Fork.trackHash });
  assert(db._metaValue('track_hash') === rot1Fork.trackHash, '64 — the forked lineage rotates to rotation 1 at gen 51 (the editor pin is rotation 0 only)');
  const g51 = db.db.exec('SELECT track_hash FROM generations WHERE generation = 51')[0].values[0][0];
  assert(g51 === rot1Fork.trackHash, '64 — gen 51 of the fork hashes the fork-seed rotation-1 track');
  const rs51 = db.resumeState();
  assert(rs51.rotationIndex === 1 && rs51.difficulty === 1 && rs51.trackSource === 'generated' && rs51.rotationSeed === forkSeed, '64 — the fork resumes at rotation 1 / difficulty 1 / generated / the fork seed, got ' + JSON.stringify({ rotationIndex: rs51.rotationIndex, difficulty: rs51.difficulty, trackSource: rs51.trackSource, rotationSeed: rs51.rotationSeed }));
  db.db.close();
});

await leg('leg 66 — pre-v4 editor/default-pin fixtures refused typed: retrofit migration retired, byte-preserved refusal', async () => {
  const { Db } = requireDb();
  const editor = new Track(DEFAULT_CONTROL_POINTS.map((p, i) => ({ x: p.x + (i % 2 ? 1 : 0), y: p.y })), 70);
  for (const [label, bytes] of [
    ['default-pin', buildV2LineageBytes()],
    ['editor-pin', buildV2LineageBytes({ trackJson: editor.toJSON(), trackHash: editor.hash() })],
  ]) {
    const before = Buffer.from(bytes);
    assertWallModelRefusal(Db, bytes, 'leg 66 ' + label);
    const db = new Db();
    let threw = null;
    try { await db.init(bytes); } catch (err) { threw = err; }
    assert(threw !== null && threw.name === 'WallModelChangedError', 'leg 66 ' + label + ' — init typed refusal');
    assert(db.db === null, 'leg 66 ' + label + ' — zero adoption');
    assert(Buffer.from(bytes).equals(before), 'leg 66 ' + label + ' — bytes untouched');
  }
});
await leg('leg 75 — pre-v4 no-op-stage fixture refused typed before any chain logic: no migration, bytes untouched', async () => {
  const { Db } = requireDb();
  const beforeRaw = buildV1LineageBytes();
  const beforeBuf = Buffer.from(beforeRaw);
  assertWallModelRefusal(Db, beforeRaw, 'leg 75');
  const db = new Db();
  let threw = null;
  try { await db.init(beforeRaw); } catch (err) { threw = err; }
  assert(threw !== null && threw.name === 'WallModelChangedError', 'leg 75 — typed refusal, got ' + (threw && threw.name));
  assert(db.db === null, 'leg 75 — instance discarded (zero adoption)');
  assert(Buffer.from(beforeRaw).equals(beforeBuf), 'leg 75 — caller bytes untouched');
});
await leg('leg 76 — verified-set driven to FAILURE: corrupted uv4 instance discarded; fresh-init and uv4-adoption faults discard instance', async () => {
  const { Db } = requireDb();
  // Arm A — corrupt a scratch IN-ADOPTION post-migration instance — flip ONE
  // byte of a tracks row's control_points — then call the verified set
  // directly and assert it THROWS with the recombination arm naming.
  // FIT-V6: the verified-set corruption arms run on a fresh uv4 lineage.
  const dbA = await freshDb();
  ensureLineageFor(dbA);
  assert(dbA.db !== null, '(A) — uv4 instance is healthy before corruption');
  const target = dbA.db;
  const rows = target.exec("SELECT track_hash, control_points FROM tracks WHERE difficulty = 1 AND source = 'generated'");
  const row = rows && rows[0] && rows[0].values && rows[0].values[0];
  assert(!!row, '(A) — the difficulty-1 generated row exists for the corruption arm');
  const parsedPts = JSON.parse(String(row[1]));
  parsedPts[0].y = parsedPts[0].y + 1; // one-point drift — the recomputed hash no longer matches the row's pin
  target.run('UPDATE tracks SET control_points = ? WHERE track_hash = ?', [JSON.stringify(parsedPts), row[0]]);
  let threwA = null;
  try {
    dbA._assertMigratedState(target, 'injection (v3)');
  } catch (err) {
    threwA = err;
  }
  assert(threwA !== null, '(A) — the corrupted instance must FAIL the verified set');
  assert(/recombin|recompute|whole-row|hash/i.test(String(threwA.message || '')) || threwA.name === 'MigrationError', '(A) — the failure names the recombination arm (or the whole-class v3 arm normalize), got ' + threwA.name + ': ' + threwA.message);
  dbA.db.close();
  // Arm B — fresh-init-time verified-set throw discards the instance.
  const dbB = new Db();
  const origFresh = Db.prototype._assertMigratedState;
  Db.prototype._assertMigratedState = function injectedThrow() {
    throw new globalThis.MigrationError('injected post-commit verified-set fault');
  };
  let threwB = null;
  try {
    await dbB.init();
  } catch (err) {
    threwB = err;
  } finally {
    Db.prototype._assertMigratedState = origFresh;
  }
  assert(threwB !== null, '(B) — fresh init must REFUSE after the injected verified-set fault');
  assert(dbB.db === null, '(B) — the fresh instance is discarded (zero adoption, no half-trusted DB)');
  // (A2) arm (l) — value domains: flip a tracks row's difficulty OUT of [0,100]
  // via ignore_check_constraints; the hash-covers-points/halfWidth arm fires
  // FIRST, so (k) passes; (l) refuses difficulty=99.9 (non-integer arm),
  // naming the label.
  const dbA2 = await freshDb();
  ensureLineageFor(dbA2);
  const tA2 = dbA2.db;
  tA2.run('PRAGMA ignore_check_constraints = ON');
  tA2.run("UPDATE tracks SET difficulty = 99.9 WHERE track_hash = (SELECT track_hash FROM tracks WHERE source = 'generated' AND difficulty = 1 LIMIT 1)");
  tA2.run('PRAGMA ignore_check_constraints = OFF');
  let threwA2 = null;
  try { dbA2._assertMigratedState(tA2, 'inj-l (v3)'); } catch (err) { threwA2 = err; }
  assert(threwA2 !== null, '(A2) — the difficulty arm (l) must FIRE');
  assert(threwA2.name === 'MigrationError', '(A2) — the failure surfaces as MigrationError, got ' + threwA2.name + ': ' + threwA2.message);
  assert(/difficulty/i.test(String(threwA2.message || '')), '(A2) — the failure names the difficulty arm, got: ' + threwA2.message);
  dbA2.db.close();

  // (A3) arm (m) — adoption semantics: flip a generations row's best_driver_index
  // OUT of [0,19] via ignore_check_constraints; (l) passes (tracks only), then
  // (m) fires on the 17 generations rails.
  const dbA3 = await freshDb();
  ensureLineageFor(dbA3);
  logGen(dbA3, 1);
  const tA3 = dbA3.db;
  tA3.run('PRAGMA ignore_check_constraints = ON');
  tA3.run('UPDATE generations SET best_driver_index = 99 WHERE generation = 1');
  tA3.run('PRAGMA ignore_check_constraints = OFF');
  let threwA3 = null;
  try { dbA3._assertMigratedState(tA3, 'inj-m (v3)'); } catch (err) { threwA3 = err; }
  assert(threwA3 !== null, '(A3) — the adoption-semantics arm (m) / integrity arm must FIRE');
  assert(threwA3.name === 'MigrationError', '(A3) — the failure surfaces as MigrationError, got ' + threwA3.name + ': ' + threwA3.message);
  assert(/CHECK|integrity|value domain|best_driver_index/i.test(String(threwA3.message || '')), '(A3) — the failure names a v3 verified-set arm (CHECK integrity or generations value domain), got: ' + threwA3.message);
  dbA3.db.close();

  // (D) migration-path verified-set fault end-to-end: monkey-patch the
  // verified set to throw INSIDE adoption (a v2 fixture that migrates into
  // v3); assert the instance is discarded and the caller bytes untouched.
  // FIT-V6: a verified-set throw on the uv4 adoption path discards the instance.
  const srcD = await freshDb();
  ensureLineageFor(srcD);
  const goodD = srcD.exportBytes();
  srcD.db.close();
  const dbD2 = new Db();
  const snapshotD = Buffer.from(goodD);
  // FIT-V6: uv4 adoption runs the battery (not the migration verified set), so
  // inject the fault at the battery itself to prove fail-closed adoption.
  const origD2 = Db.validateAdoptionBytes;
  Db.validateAdoptionBytes = function injectedThrowAdoption() {
    throw new globalThis.MigrationError('injected adoption-path verified-set fault');
  };
  let threwD2 = null;
  try {
    await dbD2.init(goodD);
  } catch (err) {
    threwD2 = err;
  } finally {
    Db.validateAdoptionBytes = origD2;
  }
  assert(threwD2 !== null, '(D) — adoption must REFUSE after the injected verified-set fault');
  assert(dbD2.db === null, '(D) — the adopted instance is discarded (zero adoption; bytes untouched)');
  assert(Buffer.compare(snapshotD, Buffer.from(goodD)) === 0, '(D) — the caller bytes are untouched');

  const dbC = new Db();
  await dbC.init();
  assert(dbC.db !== null, '(C) — a fresh Db() is healthy once the injection is removed (no cross-instance poison)');
  dbC.db.close();
});

// =============================================================================
// Summary — hard fail with a non-zero exit code if any leg failed.
// =============================================================================
console.log('------------------------');
const total = legsPassed + failures.length;
if (failures.length > 0) {
  console.log('SMOKE FAILED — ' + failures.length + ' of ' + total + ' legs failed:');
  for (const f of failures) console.log('  - ' + f.name);
  process.exit(1);
} else {
  console.log('SMOKE OK — ' + legsPassed + ' legs passed');
  process.exit(0);
}
