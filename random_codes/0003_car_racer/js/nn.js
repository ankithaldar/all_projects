// js/nn.js — Neural network for the 0003_car_racer evolutionary drivers.
//
// Architecture (FIXED forever, single source of truth: NN_ARCH):
//
//     7 inputs ──► 16 tanh ──► 16 tanh ──► 2 tanh outputs  →  [gas, steer]
//
// Inputs (EXACT, ordered — see plan NN INPUTS):
//     [ s1..s5 (sensor distance / SENSOR_MAX, [0,1], 1 when no hit),
//       speed / MAX_SPEED,
//       direction / π ]          →  direction = wrapToPi(heading − tangentAngle)/π
//
// Outputs: gas ∈ [−1,1] (−1 full brake … +1 full throttle),
//          steer ∈ [−1,1] (−1 full left  … +1 full right).
// tanh everywhere ⇒ outputs approach ±1 asymptotically; Math.tanh saturates
// to exactly ±1 for |z| ≳ 19, so consumers must tolerate [−1, 1] inclusive.
//
// This file is a plain classic script with NO imports and NO build step.
// It works both in a browser (declares globals on `globalThis`: NeuralNetwork,
// ValidationError, mulberry32, NN_ARCH, …) and in node (guarded module.exports
// at the very bottom, for tests/smoke.mjs).
//
// DETERMINISM: the built-in randomness source is NEVER used. All randomness
// flows through an explicit mulberry32-like rng: a function returning uint32
// in [0, 2^32−1]. Convert a raw draw to (0,1) with (rng() + 0.5) / 2^32.

'use strict';

// ---------------------------------------------------------------------------
// Constants (single source of truth for the network shape).
// ---------------------------------------------------------------------------
var NN_ARCH = '7-16-16-2';
var INPUT_SIZE = 7;
var HIDDEN_SIZE_1 = 16;
var HIDDEN_SIZE_2 = 16;
var OUTPUT_SIZE = 2;

// 7*16 + 16 + 16*16 + 16 + 16*2 + 2 = 112 + 16 + 256 + 16 + 32 + 2 = 434 floats.
var WEIGHT_COUNT =
  INPUT_SIZE * HIDDEN_SIZE_1 + HIDDEN_SIZE_1 +
  HIDDEN_SIZE_1 * HIDDEN_SIZE_2 + HIDDEN_SIZE_2 +
  HIDDEN_SIZE_2 * OUTPUT_SIZE + OUTPUT_SIZE; // 434

// Byte caps for payloads that travel through files/DBs (see plan: weights_json
// column is capped at 16 KB in SQLite).
var MAX_WEIGHTS_JSON_LENGTH = 16384;

// Absolute magnitude cap for every weight/bias value (plan: |v| <= 1e6).
var MAX_ABS_WEIGHT = 1e6;

// ---------------------------------------------------------------------------
// strictUint32(value, name) — the ONE strict rule for uint32 state/cursors,
// shared by mulberry32.setState, evolution.js (seed / rngState) and the
// worker's resume handler. ACCEPTS ONLY:
//   (a) typeof 'number' that is an integer in [0, 4294967295], or
//   (b) typeof 'string' matching /^(0|[1-9][0-9]{0,9})$/ whose Number() is
//       ≤ 4294967295.
// EVERYTHING ELSE throws ValidationError — null, false, true, NaN, Infinity,
// floats, '', '  ', '0x10' (hex), '1e3' (exponent), '00' / '000' (leading
// zeros), '4294967296' (out of range), arrays, plain objects — before any
// state is committed. (Earlier revisions coerced with Number(), which silently
// accepted null→0 and '0x10'→16; that leniency is gone by design.)
// Returns the value as a plain JS number (already a safe uint32).
// ---------------------------------------------------------------------------
function strictUint32(value, name) {
  var label = name || 'value';
  if (typeof value === 'number') {
    if (!Number.isInteger(value) || value < 0 || value > 4294967295) {
      throw new ValidationError(
        label + ' must be an integer uint32 [0, 4294967295], got ' + String(value)
      );
    }
    return value;
  }
  if (typeof value === 'string') {
    if (!/^(0|[1-9][0-9]{0,9})$/.test(value)) {
      throw new ValidationError(
        label + ' must be a decimal uint32 string (/^(0|[1-9][0-9]{0,9})$/), got ' + JSON.stringify(value)
      );
    }
    var n = Number(value); // ≤ 9999999999 — exactly representable
    if (!(n <= 4294967295)) {
      throw new ValidationError(
        label + ' must be ≤ 4294967295, got ' + JSON.stringify(value)
      );
    }
    return n;
  }
  throw new ValidationError(
    label + ' must be a uint32 (integer number or decimal string), got ' +
      (value === null ? 'null' : Array.isArray(value) ? 'array' : typeof value)
  );
}

// ---------------------------------------------------------------------------
// ValidationError — shared, order-independent across the pure-logic files.
//
// Track is loaded BEFORE NeuralNetwork in the documented script order, so we
// cannot assume nn.js defined it first. Each pure-logic file installs the
// class on globalThis if nobody did yet, then uses the (now existing) global.
// In node each require() shares one globalThis, so all modules agree on the
// same class object. The pure-logic files never redeclare the class lexically
// — a top-level class/const would collide across classic <script> tags that
// share one page and one global lexical environment.
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
// mulberry32 — tiny deterministic PRNG factory (public domain pattern).
//
// mulberry32(seed) returns a function () => uint32 in [0, 2^32−1].
// The returned function carries its whole state inside a closure, which makes
// it portable: Evolution can serialize/restore `state` later by just re-seeding.
//
// The returned function exposes two extra own-properties (M2 addition — the
// call algorithm itself is bit-identical to the original closure):
//   .state()          → the CURRENT uint32 state (i.e. what the next call will
//                       add 0x6D2B79F5 to). This is the value the plan calls
//                       `rng_state` — meta.rng_state in the DB, the audit
//                       column in generations, and the 'rngState' field of the
//                       worker's 'generation' payload.
//   .setState(s)      → restore a previously captured state via the STRICT
//                       rule in strictUint32 (above): ONLY integer numbers in
//                       [0, 4294967295], or decimal TEXT /^(0|[1-9][0-9]{0,9})$/
//                       with value ≤ 4294967295. EVERYTHING ELSE — null,
//                       false, '', '0x10', '1e3', '00', NaN, 1.5, out-of-range
//                       text, arrays, objects — throws ValidationError BEFORE
//                       any state is committed. The DB stores rng_state as
//                       TEXT (decimal), so meta.rng_state round-trips directly.
//
// Usage:  const rng = mulberry32(1234);  rng()  →  3123… (uint32)
//         To get a float in (0,1): (rng() + 0.5) / 4294967296
//         const s = rng.state();  …pass s to a fresh mulberry32(0).setState(s)
//         to resume the stream exactly where it left off.
// ---------------------------------------------------------------------------
function mulberry32(seed) {
  // Coerce seed to uint32 — seeds may arrive as decimal strings from the DB
  // (meta.rng_state is TEXT at the edges, e.g. '4294967295').
  var state = (typeof seed === 'number' ? seed : Number(seed)) >>> 0;
  var next = function next() {
    state = (state + 0x6d2b79f5) >>> 0;
    var t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (t ^ (t >>> 14)) >>> 0;
  };
  // .state() — the current uint32 cursor. Non-enumerable is unnecessary; a
  // plain own property on the function object is the simplest stable shape.
  next.state = function stateGetter() {
    return state >>> 0;
  };
  // .setState(s) — restore a cursor previously read from .state(). STRICT
  // rule (see strictUint32 above): accepted only as integer number or decimal
  // TEXT; everything else throws ValidationError BEFORE touching the closure
  // variable (validate-then-commit).
  next.setState = function stateSetter(s) {
    state = strictUint32(s, 'mulberry32.setState');
  };
  return next;
}

// Draw a standard-normal sample from a mulberry32-like rng via Box–Muller.
// u1' = 1 − u1 is used as the log argument so we NEVER compute ln(0) — the
// raw (rng() + 0.5) / 2^32 guarantees u1 ∈ (0,1) exclusive, and 1 − u1 ∈ (0,1].
function gaussianFromRng(rng) {
  var u1 = (rng() + 0.5) / 4294967296; // (0,1)
  var u2 = (rng() + 0.5) / 4294967296; // (0,1)
  return Math.sqrt(-2 * Math.log(1 - u1)) * Math.cos(2 * Math.PI * u2);
}

// ---------------------------------------------------------------------------
// Weight payload validation — shared by the constructor, fromJSON() and
// setWeights(). Validates into FRESH arrays (deep copies); the caller decides
// whether to commit them. Never mutates anything on failure.
// ---------------------------------------------------------------------------
function validateWeightsPayload(obj) {
  if (obj === null || typeof obj !== 'object' || Array.isArray(obj)) {
    throw new ValidationError('weights payload must be a plain object, got ' + (obj === null ? 'null' : Array.isArray(obj) ? 'array' : typeof obj));
  }
  if (obj.architecture !== NN_ARCH) {
    throw new ValidationError('architecture must be exactly "' + NN_ARCH + '", got ' + JSON.stringify(obj.architecture));
  }

  // Canonical order of tensors. Each entry: [payload key, expected length].
  var spec = [
    ['w1', INPUT_SIZE * HIDDEN_SIZE_1],   // 112
    ['b1', HIDDEN_SIZE_1],                //  16
    ['w2', HIDDEN_SIZE_1 * HIDDEN_SIZE_2],// 256
    ['b2', HIDDEN_SIZE_2],                //  16
    ['w3', HIDDEN_SIZE_2 * OUTPUT_SIZE],  //  32
    ['b3', OUTPUT_SIZE],                  //   2
  ];

  var out = {};
  var total = 0;
  for (var i = 0; i < spec.length; i++) {
    var key = spec[i][0];
    var expected = spec[i][1];
    var arr = obj[key];
    if (!Array.isArray(arr)) {
      throw new ValidationError('weights.' + key + ' must be an array of ' + expected + ' numbers');
    }
    if (arr.length !== expected) {
      throw new ValidationError('weights.' + key + ' must contain exactly ' + expected + ' numbers, got ' + arr.length);
    }
    var clean = new Array(expected);
    for (var j = 0; j < expected; j++) {
      var v = arr[j];
      if (typeof v !== 'number' || !Number.isFinite(v)) {
        throw new ValidationError('weights.' + key + '[' + j + '] must be a finite number, got ' + String(v));
      }
      if (Math.abs(v) > MAX_ABS_WEIGHT) {
        throw new ValidationError('weights.' + key + '[' + j + '] exceeds |v| <= ' + MAX_ABS_WEIGHT + ' (got ' + v + ')');
      }
      clean[j] = v;
    }
    out[key] = clean;
    total += expected;
  }
  if (total !== WEIGHT_COUNT) {
    // Unreachable given the exact per-tensor lengths above, but the plan
    // demands an explicit "exactly 434 floats" guarantee — keep the check.
    throw new ValidationError('weights payload must contain exactly ' + WEIGHT_COUNT + ' floats, got ' + total);
  }
  return out;
}

// ---------------------------------------------------------------------------
// NeuralNetwork — the driver's brain.
//
// Construction:
//   new NeuralNetwork(rng)      — random init (Xavier-like fan-in sigma scale,
//                                 biases = 0). `rng` is a mulberry32-like
//                                 function. It is MANDATORY — there is no
//                                 silent built-in-RNG fallback.
//   new NeuralNetwork(weightsObj)
//   NeuralNetwork.fromJSON(objOrString) — validated load (see below).
//
// Internals: flat row-major arrays w1[i*16+j], w2[i*16+j], w3[i*2+j].
// Hidden activations use preallocated scratch arrays so forward() performs no
// allocation beyond the returned [gas, steer] pair. Draw order for random
// init and for mutate() is FIXED: w1 → b1 → w2 → b2 → w3 → b3, row-major.
// ---------------------------------------------------------------------------
function NeuralNetwork(arg) {
  if (typeof arg === 'function') {
    // Random init path — rng is mandatory and must quack like mulberry32:
    // () => uint32 in [0, 2^32−1]. We deliberately do NOT call rng() here to
    // "validate" it — that would consume a draw and silently shift every
    // downstream random decision for callers sharing the rng instance.
    var rng = arg;
    this.w1 = NeuralNetwork._randomTensor(INPUT_SIZE * HIDDEN_SIZE_1, rng, INPUT_SIZE);
    this.b1 = new Array(HIDDEN_SIZE_1).fill(0);
    this.w2 = NeuralNetwork._randomTensor(HIDDEN_SIZE_1 * HIDDEN_SIZE_2, rng, HIDDEN_SIZE_1);
    this.b2 = new Array(HIDDEN_SIZE_2).fill(0);
    this.w3 = NeuralNetwork._randomTensor(HIDDEN_SIZE_2 * OUTPUT_SIZE, rng, HIDDEN_SIZE_2);
    this.b3 = new Array(OUTPUT_SIZE).fill(0);
  } else if (arg !== null && typeof arg === 'object') {
    // Weights-object path — fully validated, then committed.
    var clean = validateWeightsPayload(arg);
    this.w1 = clean.w1;
    this.b1 = clean.b1;
    this.w2 = clean.w2;
    this.b2 = clean.b2;
    this.w3 = clean.w3;
    this.b3 = clean.b3;
  } else {
    throw new ValidationError(
      'NeuralNetwork requires a mulberry32 rng function (random init) or a weights object/fromJSON payload'
    );
  }

  // Preallocated hidden-layer scratch (avoids per-forward array allocation).
  this._h1 = new Array(HIDDEN_SIZE_1);
  this._h2 = new Array(HIDDEN_SIZE_2);
}

// Fill a tensor with N(0, 1/sqrt(fanIn)) samples — Xavier-like fan-in scaling
// (sigma = 1/sqrt(fanIn), i.e. ≈0.378 for the 7-input layer), deterministic
// draw order (row-major, single global order w1 → b1 → w2 → b2 → w3 → b3).
NeuralNetwork._randomTensor = function _randomTensor(count, rng, fanIn) {
  var out = new Array(count);
  var sigma = 1 / Math.sqrt(fanIn);
  for (var i = 0; i < count; i++) out[i] = gaussianFromRng(rng) * sigma;
  return out;
};

NeuralNetwork.ARCH_STRING = NN_ARCH;
NeuralNetwork.INPUT_SIZE = INPUT_SIZE;
NeuralNetwork.HIDDEN_SIZE_1 = HIDDEN_SIZE_1;
NeuralNetwork.HIDDEN_SIZE_2 = HIDDEN_SIZE_2;
NeuralNetwork.OUTPUT_SIZE = OUTPUT_SIZE;
NeuralNetwork.WEIGHT_COUNT = WEIGHT_COUNT;

// Load from a weights payload: plain object OR a JSON string.
// When a string is passed it must be ≤ 16384 chars (SQLite column cap) and
// must parse as JSON; every other violation throws ValidationError.
NeuralNetwork.fromJSON = function fromJSON(input) {
  var obj = input;
  if (typeof input === 'string') {
    if (input.length > MAX_WEIGHTS_JSON_LENGTH) {
      throw new ValidationError(
        'weights_json exceeds ' + MAX_WEIGHTS_JSON_LENGTH + ' characters (got ' + input.length + ')'
      );
    }
    try {
      obj = JSON.parse(input);
    } catch (err) {
      throw new ValidationError('weights_json is not valid JSON: ' + (err && err.message));
    }
  }
  return new NeuralNetwork(obj); // constructor validates everything
};

// Static randomInit — the plan's `nn.randomInit(rng)` API. Delegates to the
// constructor's random-init path so `nn.randomInit(rng)` and
// `new NeuralNetwork(rng)` are interchangeable (evolution owns the shared
// rng instance and calls one of them).
NeuralNetwork.randomInit = function randomInit(rng) {
  return new NeuralNetwork(rng);
};

// forward(inputs7) → [gas, steer]
// Validates input length 7 and finiteness; throws ValidationError otherwise.
// Hot path: plain loops, no closures, no object allocation beyond [gas, steer].
NeuralNetwork.prototype.forward = function forward(inputs) {
  if (!inputs || inputs.length !== INPUT_SIZE) {
    throw new ValidationError('forward expects exactly ' + INPUT_SIZE + ' inputs, got ' + (inputs && inputs.length));
  }
  var x0 = inputs[0], x1 = inputs[1], x2 = inputs[2], x3 = inputs[3], x4 = inputs[4], x5 = inputs[5], x6 = inputs[6];
  if (
    !Number.isFinite(x0) || !Number.isFinite(x1) || !Number.isFinite(x2) ||
    !Number.isFinite(x3) || !Number.isFinite(x4) || !Number.isFinite(x5) || !Number.isFinite(x6)
  ) {
    throw new ValidationError('forward inputs must all be finite numbers');
  }

  var w1 = this.w1, b1 = this.b1, w2 = this.w2, b2 = this.b2, w3 = this.w3, b3 = this.b3;
  var h1 = this._h1, h2 = this._h2;

  // Hidden layer 1: h1[j] = tanh(b1[j] + Σ_i x[i] * w1[i*16 + j])
  for (var j = 0; j < HIDDEN_SIZE_1; j++) {
    h1[j] = Math.tanh(
      b1[j] +
      w1[j] * x0 + w1[HIDDEN_SIZE_1 + j] * x1 + w1[2 * HIDDEN_SIZE_1 + j] * x2 +
      w1[3 * HIDDEN_SIZE_1 + j] * x3 + w1[4 * HIDDEN_SIZE_1 + j] * x4 +
      w1[5 * HIDDEN_SIZE_1 + j] * x5 + w1[6 * HIDDEN_SIZE_1 + j] * x6
    );
  }

  // Hidden layer 2: h2[j] = tanh(b2[j] + Σ_i h1[i] * w2[i*16 + j])
  for (var k = 0; k < HIDDEN_SIZE_2; k++) {
    var acc = b2[k];
    for (var i = 0; i < HIDDEN_SIZE_1; i++) acc += h1[i] * w2[i * HIDDEN_SIZE_2 + k];
    h2[k] = Math.tanh(acc);
  }

  // Output layer: out[j] = tanh(b3[j] + Σ_i h2[i] * w3[i*2 + j])
  var gas = b3[0], steer = b3[1];
  for (var m = 0; m < HIDDEN_SIZE_2; m++) {
    gas += h2[m] * w3[m * OUTPUT_SIZE + 0];
    steer += h2[m] * w3[m * OUTPUT_SIZE + 1];
  }
  return [Math.tanh(gas), Math.tanh(steer)];
};

// getWeights() → fresh deep-copy object. Safe to mutate/JSON.stringify.
NeuralNetwork.prototype.getWeights = function getWeights() {
  return {
    architecture: NN_ARCH,
    w1: this.w1.slice(),
    b1: this.b1.slice(),
    w2: this.w2.slice(),
    b2: this.b2.slice(),
    w3: this.w3.slice(),
    b3: this.b3.slice(),
  };
};

// setWeights(obj) — same validation as fromJSON, but validate-then-commit:
// on ANY violation the network is left completely UNCHANGED.
NeuralNetwork.prototype.setWeights = function setWeights(obj) {
  var clean = validateWeightsPayload(obj); // throws before touching this.*
  this.w1 = clean.w1;
  this.b1 = clean.b1;
  this.w2 = clean.w2;
  this.b2 = clean.b2;
  this.w3 = clean.w3;
  this.b3 = clean.b3;
  return this;
};

// clone() — independent deep copy (used by evolution: child0 = exact clone).
NeuralNetwork.prototype.clone = function clone() {
  return new NeuralNetwork(this.getWeights()); // revalidates, deep-copies
};

// mutate(rate, scale, rng) — in-place Gaussian perturbation.
//   rate  ∈ [0,1]: per-weight probability of being perturbed (biases included)
//   scale ≥ 0    : standard deviation of the perturbation (Gaussian sigma)
//   rng          : mulberry32-like function () => uint32
// Draw order is FIXED (w1 → b1 → w2 → b2 → w3 → b3, row-major); per weight,
// one u ∈ (0,1) is drawn for the rate gate, then u1,u2 ∈ (0,1) for Box–Muller.
// Box–Muller with u1' = 1 − u1 (never ln 0). Draw order is part of the
// deterministic contract — do not reorder without a new lineage.
NeuralNetwork.prototype.mutate = function mutate(rate, scale, rng) {
  if (!Number.isFinite(rate) || rate < 0 || rate > 1) {
    throw new ValidationError('mutate rate must be a finite number in [0,1], got ' + rate);
  }
  if (!Number.isFinite(scale) || scale < 0) {
    throw new ValidationError('mutate scale must be a finite number >= 0, got ' + scale);
  }
  if (typeof rng !== 'function') {
    throw new ValidationError('mutate requires an rng function () => uint32');
  }
  var tensors = [this.w1, this.b1, this.w2, this.b2, this.w3, this.b3];
  for (var t = 0; t < tensors.length; t++) {
    var tensor = tensors[t];
    for (var i = 0; i < tensor.length; i++) {
      var gate = (rng() + 0.5) / 4294967296;
      if (gate < rate) tensor[i] += gaussianFromRng(rng) * scale;
    }
  }
  return this;
};

// Browser-global exposure + guarded CommonJS export (node tests).
if (typeof globalThis !== 'undefined') {
  globalThis.NeuralNetwork = NeuralNetwork;
  globalThis.ValidationError = ValidationError;
  globalThis.mulberry32 = mulberry32;
  globalThis.NN_ARCH = NN_ARCH;
  globalThis.strictUint32 = strictUint32;
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    NeuralNetwork: NeuralNetwork,
    ValidationError: ValidationError,
    mulberry32: mulberry32,
    NN_ARCH: NN_ARCH,
    WEIGHT_COUNT: WEIGHT_COUNT,
    MAX_WEIGHTS_JSON_LENGTH: MAX_WEIGHTS_JSON_LENGTH,
    strictUint32: strictUint32,
  };
}
