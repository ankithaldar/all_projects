// js/rotation.js — Procedural track rotation (FIT-V3) for the 0003_car_racer
// game.
//
// Pure logic: no DOM, no database, no wall-clock inputs and no unseeded
// randomness.
// Deterministic per (rotation_seed, rotationIndex) — the generator draws its
// phases only from mulberry32((rotationSeed ^ Math.imul(rotationIndex + 1,
// 0x85EBCA6B)) >>> 0), so the same seed + index always produces the same
// bytes on the same engine. The default circuit is deliberately NOT read or
// reused here; rotation 0 is the caller's pinned track, not a generated one.
//
// Plain classic script; browser globals on globalThis, guarded CommonJS
// export at the bottom for node tests.
//
// Load order: AFTER js/track.js (the classic <script> list in index.html and
// the worker importScripts list) — trackHash resolves canonicalJSON/cyrb53 by
// global lookup at CALL time. FIT-V5/V6 additionally resolves globalThis.Track
// at CALL time for the V2 canonical clipped-geometry gate; V1 and both version
// parsers still depend on nothing from track.js.

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
// Frozen tuning constants (plan FIT-V3 §4 — nothing here is a knob).
// ---------------------------------------------------------------------------
var ROTATE_EVERY = 50;            // generations per rotation block (D-1)
var TRACK_POINTS = 14;            // control points per generated track
var TRACK_CENTER_X = 550;
var TRACK_CENTER_Y = 325;
var TRACK_MAX_RADIUS = 245;       // hard outer bound — r_i <= 245 by construction (pre-rounding)
var TRACK_FALLBACK_RADIUS = 170;  // final fallback: regular 14-gon radius
var TRACK_MAX_REJECTS = 200;      // self-intersection is the ONLY reject path
var TAU = Math.PI * 2;
var TWO_POW_NEG_32 = 1 / 4294967296; // 2^-32 — u_k = s() * 2^-32 (raw-uint32 semantics)
var U32_MAX = 4294967295;

// HALFWIDTH(d) = 50 − 0.25 × d     // 50.0 (d=0) → 25.0 (d=100)
// FIT-V6 (plan §Half-width & hairpin feasibility): the shared ladder is
// narrowed so the clipped inner wall never merges/vanishes at a hairpin.
// With max hw=50 and the generator's apex-radius gate R>=75, arm separation
// 2R >= 150 > 2*50+20 = 120 by construction. This is a SHARED ladder used by
// V1 and V2 alike (the plan accepts the global break to both golden corpora);
// the V1 generator algorithm itself is left literally unchanged.
// The result is normalized through toFixed(9) (the same rounding
// canonicalJSON applies) so exact values come out exact — e.g. 50 − 0.25*100
// === 25, not 24.99999999999999.
function halfWidthForDifficulty(d) {
  validateDifficultyInput(d);
  return Number((50 - 0.25 * d).toFixed(9));
}

// HARMONICS(d) = 2 + floor(d/25)   // 2 (0-24) → 6 (≥100)
function harmonicsForDifficulty(d) {
  validateDifficultyInput(d);
  return 2 + Math.floor(d / 25);
}

// AMPLITUDE(d) = 0.08 + 0.34 × d/100  // 0.08 (d=0) → 0.42 (d=100)
// Normalized through toFixed(9) like HALFWIDTH (0.08 + 0.34 otherwise yields
// 0.42000000000000004).
function amplitudeForDifficulty(d) {
  validateDifficultyInput(d);
  return Number((0.08 + 0.34 * (d / 100)).toFixed(9));
}

// Difficulty inputs are finite integers in [0, 100] — the difficulty ladder
// is integer by construction (D-3) and difficultyForRotation caps at 100.
function validateDifficultyInput(d) {
  if (!Number.isFinite(d) || d < 0 || d > 100) {
    throw new ValidationError('difficulty must be a finite number in [0, 100], got ' + String(d));
  }
}

// rotationForGeneration(generation) — D-1: floor((N − 1) / ROTATE_EVERY).
// Gens 1–50 → rotation 0; gens 4951–5000 → rotation 99. The switch happens
// exactly at generation N with N ≡ 1 (mod 50), N ≥ 51.
function rotationForGeneration(generation) {
  if (!Number.isInteger(generation) || generation < 1) {
    throw new ValidationError('rotationForGeneration expects an integer generation >= 1, got ' + String(generation));
  }
  return Math.floor((generation - 1) / ROTATE_EVERY);
}

// difficultyForRotation(rotationIndex) — D-2: Math.min(rotationIndex, 100).
function difficultyForRotation(rotationIndex) {
  if (!Number.isInteger(rotationIndex) || rotationIndex < 0) {
    throw new ValidationError('difficultyForRotation expects an integer rotationIndex >= 0, got ' + String(rotationIndex));
  }
  return Math.min(rotationIndex, 100);
}

function validateRotationSeed(rotationSeed) {
  if (!Number.isInteger(rotationSeed) || rotationSeed < 0 || rotationSeed > U32_MAX) {
    throw new ValidationError('rotationSeed must be a uint32 (integer in [0, 4294967295]), got ' + String(rotationSeed));
  }
}

function validateRotationIndex(rotationIndex) {
  if (!Number.isInteger(rotationIndex) || rotationIndex < 0) {
    throw new ValidationError('rotationIndex must be an integer >= 0, got ' + String(rotationIndex));
  }
}

// rotationMulberry32 — uint32 stream (the same algorithm js/nn.js exposes as
// the global mulberry32 — re-stated here, PRIVATE on purpose, so this module
// never depends on load order AND never clobbers the global nn.js copy the
// Evolution layer captures at load time; s() returns a uint32 and u_k = s() ×
// 2⁻³² is uniform in [0,1) — the raw-uint32 × 2⁻³² semantics pinned in
// plan §4).
function rotationMulberry32(seed) {
  var state = (typeof seed === 'number' ? seed : Number(seed)) >>> 0;
  return function () {
    state = (state + 0x6d2b79f5) >>> 0;
    var t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (t ^ (t >>> 14)) >>> 0;
  };
}

// Plan §4, verbatim, for the record:
//   φ_k = 2π·u_k,  u_k = s() × 2⁻³²   (pinned: u_k uniform in [0,1) —
//                                    raw-uint32 semantics × 2⁻³²)
//   w_k = 1 / k^(3/2);  a_k = AMPLITUDE × w_k / Σ_{j≤K} w_j
//   θ_i = 2π · i / 14                   (fixed, no jitter)
//   ρ_i = 1 + Σ_k a_k · sin(k·θ_i + φ_k)
//   R0  = TRACK_MAX_RADIUS / Math.max(1, max(ρ_i))  (clamp; r_i ≤ 245)
//   points = r_i at θ_i + TRACK_CENTER, rounded to 3 decimals
var THETAS = new Array(TRACK_POINTS);
for (var thetaInit = 0; thetaInit < TRACK_POINTS; thetaInit++) {
  THETAS[thetaInit] = TAU * thetaInit / TRACK_POINTS;
}

function round3(v) {
  return Number(v.toFixed(3));
}

function orientationSign(ax, ay, bx, by, cx, cy) {
  return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

// Proper crossing of two closed segments — collinear/touching cases report
// no crossing (measure-zero here).
function segmentsCross(ax, ay, bx, by, cx, cy, dx, dy) {
  var o1 = orientationSign(ax, ay, bx, by, cx, cy);
  var o2 = orientationSign(ax, ay, bx, by, dx, dy);
  var o3 = orientationSign(cx, cy, dx, dy, ax, ay);
  var o4 = orientationSign(cx, cy, dx, dy, bx, by);
  return ((o1 > 0 && o2 < 0) || (o1 < 0 && o2 > 0)) &&
         ((o3 > 0 && o4 < 0) || (o3 < 0 && o4 > 0));
}

// Self-intersection over the control polygon — non-adjacent closed-segment
// pairs only (14 segments ⇒ ≤ 91 pair-tests). Adjacent pairs (including the
// wrap-around pair 0 ↔ 13) share a vertex and are excluded.
function controlPolygonSelfIntersects(points) {
  var n = points.length;
  for (var i = 0; i < n; i++) {
    for (var j = i + 1; j < n; j++) {
      if (j === i + 1) continue;
      if (i === 0 && j === n - 1) continue;
      var a = points[i], b = points[(i + 1) % n];
      var c = points[j], d = points[(j + 1) % n];
      if (segmentsCross(a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y)) return true;
    }
  }
  return false;
}

// generateTrack(rotationSeed, rotationIndex)
//   → { points: [{x,y} × 14], halfWidth, difficulty }
// Deterministic; runs once per rotation at pre-materialization time and
// never in a hot path. Self-intersection of the control polygon is the ONLY
// reject path — a reject releases the stream (state has advanced) and draws
// again, up to TRACK_MAX_REJECTS attempts; then the regular 14-gon fallback
// (valid by construction).
function generateTrack(rotationSeed, rotationIndex) {
  validateRotationSeed(rotationSeed);
  validateRotationIndex(rotationIndex);
  var difficulty = difficultyForRotation(rotationIndex);
  var halfWidth = halfWidthForDifficulty(difficulty);
  var harmonics = harmonicsForDifficulty(difficulty);
  var amplitude = amplitudeForDifficulty(difficulty);
  var stream = rotationMulberry32((rotationSeed ^ Math.imul(rotationIndex + 1, 0x85EBCA6B)) >>> 0);

  // w_k = 1 / k^(3/2);  a_k = AMPLITUDE × w_k / Σ_{j≤K} w_j
  var wSum = 0;
  for (var k = 1; k <= harmonics; k++) wSum += 1 / Math.pow(k, 1.5);
  var amplitudes = new Array(harmonics);
  for (var a = 0; a < harmonics; a++) {
    amplitudes[a] = amplitude * (1 / Math.pow(a + 1, 1.5)) / wSum;
  }

  for (var attempt = 0; attempt < TRACK_MAX_REJECTS; attempt++) {
    // φ_k = 2π·u_k, u_k = s() × 2⁻³² — drawn fresh per attempt so a rejected
    // draw never recurs.
    var phases = new Array(harmonics);
    for (var p = 0; p < harmonics; p++) phases[p] = TAU * (stream() * TWO_POW_NEG_32);

    var rhoValues = new Array(TRACK_POINTS);
    var maxRho = -Infinity;
    for (var i = 0; i < TRACK_POINTS; i++) {
      var theta = THETAS[i];
      var acc = 0;
      for (var a2 = 0; a2 < harmonics; a2++) {
        acc += amplitudes[a2] * Math.sin((a2 + 1) * theta + phases[a2]);
      }
      var rho = 1 + acc;
      rhoValues[i] = rho;
      if (rho > maxRho) maxRho = rho;
    }
    // R0 clamp — belt-and-braces for a case the harmonic sum proves
    // impossible (mean(ρ)=1 over the 14 control points ⇒ max(ρ) ≥ 1 always);
    // keeps r_i ≤ 245 when active. NO behavior change.
    var r0 = TRACK_MAX_RADIUS / Math.max(1, maxRho);
    var points = new Array(TRACK_POINTS);
    for (var i2 = 0; i2 < TRACK_POINTS; i2++) {
      var radius = r0 * rhoValues[i2];
      points[i2] = {
        x: round3(TRACK_CENTER_X + radius * Math.cos(THETAS[i2])),
        y: round3(TRACK_CENTER_Y + radius * Math.sin(THETAS[i2])),
      };
    }
    if (!controlPolygonSelfIntersects(points)) {
      return { points: points, halfWidth: halfWidth, difficulty: difficulty };
    }
  }

  // Final fallback: regular 14-gon of radius TRACK_FALLBACK_RADIUS (valid by
  // construction; the sweep leg verifies it is never approached in
  // practice). Never a crash.
  var fallback = new Array(TRACK_POINTS);
  for (var i3 = 0; i3 < TRACK_POINTS; i3++) {
    fallback[i3] = {
      x: round3(TRACK_CENTER_X + TRACK_FALLBACK_RADIUS * Math.cos(THETAS[i3])),
      y: round3(TRACK_CENTER_Y + TRACK_FALLBACK_RADIUS * Math.sin(THETAS[i3])),
    };
  }
  return { points: fallback, halfWidth: halfWidth, difficulty: difficulty };
}

// ---------------------------------------------------------------------------
// FIT-V4 V2 generator (plan §Frozen V2 formula) — ADDITIVE ONLY. Every V1
// constant and function above is left literally unchanged. V2 gets its own
// THETAS36, harmonic/amplitude ladders and fallback, and REUSES the V1
// mulberry32 seed derivation, round3, center, radius cap, reject rule and the
// fixed 200-attempt cap. V2 hashes are persisted at pin time, so the formula
// is frozen: any change requires a new red-team gate.
// ---------------------------------------------------------------------------
var TRACK_POINTS_V2 = 36;

var THETAS36 = new Array(TRACK_POINTS_V2);
for (var theta36Init = 0; theta36Init < TRACK_POINTS_V2; theta36Init++) {
  THETAS36[theta36Init] = TAU * theta36Init / TRACK_POINTS_V2;
}

// K2(d) = 4 + floor(d/16)  // 4 (d=0-15) → 10 (d≥96)
function harmonicsV2ForDifficulty(d) {
  validateDifficultyInput(d);
  return 4 + Math.floor(d / 16);
}

// A2(d) = 0.12 + 0.33·d/100  // 0.12 (d=0) → 0.45 (d=100); ≤0.45 < 1 ⇒ ρ ≥ 0.55
// Normalized through toFixed(9) exactly like the V1 amplitude ladder so the
// result is deterministic across engines.
function amplitudeV2ForDifficulty(d) {
  validateDifficultyInput(d);
  return Number((0.12 + 0.33 * (d / 100)).toFixed(9));
}

// Regular 36-gon fallback (radius 170) — valid by construction. Exported on
// Rotation so the guaranteed-valid fallback geometry can be verified.
function generateTrackV2Fallback() {
  var fallback = new Array(TRACK_POINTS_V2);
  for (var i = 0; i < TRACK_POINTS_V2; i++) {
    fallback[i] = {
      x: round3(TRACK_CENTER_X + TRACK_FALLBACK_RADIUS * Math.cos(THETAS36[i])),
      y: round3(TRACK_CENTER_Y + TRACK_FALLBACK_RADIUS * Math.sin(THETAS36[i])),
    };
  }
  return fallback;
}

function pointsAreFinite(points) {
  for (var i = 0; i < points.length; i++) {
    if (!Number.isFinite(points[i].x) || !Number.isFinite(points[i].y)) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// FIT-V5 V2 — frozen template-family generator (plan §Frozen generator
// algorithm, steps 1-9). V1 above and both version parsers below are left
// literally unchanged; only the BODY of generateTrackV2 is replaced. The six
// coordinate arrays are verbatim frozen data from the approved spike.
// ---------------------------------------------------------------------------
// FIT-V6 cycle-3 S1: six arc-built F1 circuits (91-96 control points). Each
// is a closed sequence of straight segments and circular arcs of radius
// ~130px with ONE genuine >=150-degree U-turn hairpin per track, plus long
// straights and asymmetric turn sequences (C1,C2,C3,B2,B3,A2). Verified
// against the real Track + Car at hw 50/49.75/37.25/25: 0 crashes, min
// centreline-to-clipped-wall clearance >= 18px over samples x segments.
var V5_TEMPLATES = Object.freeze([
  Object.freeze([[307.838,127.95],[325.044,127.95],[362.897,127.95],[446.175,127.95],[773.785,127.95],[857.063,127.95],[894.917,127.95],[912.123,127.95],[926.461,127.95],[936.728,128.46],[946.893,129.985],[956.857,132.511],[966.522,136.012],[975.792,140.454],[984.575,145.793],[992.787,151.977],[1000.344,158.944],[1007.174,166.627],[1013.208,174.948],[1018.388,183.827],[1022.662,193.175],[1025.988,202.902],[1028.334,212.91],[1029.675,223.101],[1030,233.375],[1029.305,243.631],[1027.596,253.767],[1024.891,263.684],[1021.216,273.283],[1016.607,282.471],[1011.11,291.157],[1004.779,299.256],[997.676,306.686],[989.872,313.376],[981.443,319.259],[972.472,324.278],[963.048,328.382],[953.263,331.532],[939.414,335.243],[922.794,339.697],[886.23,349.494],[805.79,371.048],[628.821,418.466],[579.72,431.623],[402.751,479.042],[322.311,500.596],[285.747,510.393],[269.127,514.846],[255.277,518.557],[245.567,520.666],[235.699,521.834],[225.764,522.05],[215.855,521.314],[206.062,519.631],[196.475,517.017],[187.182,513.497],[178.27,509.102],[169.821,503.873],[161.911,497.858],[150.927,488.642],[117.975,460.992],[106.991,451.775],[99.624,444.959],[92.952,437.461],[87.036,429.352],[81.934,420.709],[77.692,411.612],[74.35,402.147],[71.941,392.404],[70.486,382.473],[70,372.447],[70,358.109],[70,315.093],[70,300.755],[70.499,290.605],[71.99,280.552],[74.459,270.694],[77.883,261.126],[82.228,251.939],[87.452,243.222],[93.506,235.06],[100.331,227.53],[110.47,217.391],[140.886,186.975],[151.025,176.836],[159.08,169.583],[167.849,163.212],[177.237,157.792],[187.139,153.383],[197.448,150.034],[211.298,146.323],[252.847,135.189],[266.697,131.478],[275.517,129.523],[284.474,128.344],[293.499,127.95]]),
  Object.freeze([[366.404,80.546],[387.299,80.546],[433.267,80.546],[534.398,80.546],[718.878,80.546],[820.009,80.546],[865.977,80.546],[886.872,80.546],[904.284,80.546],[916.805,81.171],[929.202,83.039],[941.351,86.133],[953.132,90.421],[964.427,95.86],[975.124,102.397],[985.118,109.967],[994.308,118.494],[1002.603,127.894],[1009.922,138.073],[1016.19,148.93],[1021.346,160.358],[1025.339,172.241],[1028.129,184.464],[1029.687,196.903],[1030,209.436],[1029.063,221.938],[1026.886,234.284],[1023.491,246.352],[1018.91,258.023],[1013.191,269.179],[1006.389,279.71],[998.573,289.511],[989.819,298.486],[980.215,306.545],[969.857,313.607],[958.847,319.603],[947.295,324.472],[930.933,330.428],[911.298,337.574],[868.102,353.296],[773.07,387.885],[468.945,498.577],[373.913,533.166],[330.717,548.888],[311.083,556.035],[294.72,561.99],[282.487,565.749],[269.936,568.246],[257.195,569.454],[244.398,569.361],[231.676,567.968],[219.162,565.289],[206.985,561.352],[195.271,556.198],[184.141,549.88],[173.712,542.463],[164.09,534.025],[155.376,524.653],[144.183,511.314],[110.606,471.298],[99.414,457.96],[91.954,448.128],[85.495,437.613],[80.097,426.515],[75.813,414.941],[72.684,403.003],[70.74,390.816],[70,378.498],[70.471,366.166],[71.989,348.82],[76.541,296.782],[78.059,279.436],[79.773,267.016],[82.715,254.83],[86.855,242.997],[92.154,231.634],[98.557,220.856],[106.001,210.769],[114.413,201.473],[126.726,189.161],[163.663,152.224],[175.975,139.912],[185.757,131.104],[196.406,123.367],[207.806,116.785],[219.831,111.432],[232.349,107.364],[249.168,102.857],[299.625,89.337],[316.444,84.831],[327.155,82.456],[338.032,81.024],[348.992,80.546]]),
  Object.freeze([[456.712,70],[479.616,70],[530.007,70],[745.472,70],[795.862,70],[818.767,70],[837.854,70],[851.643,70.691],[865.294,72.759],[878.669,76.181],[891.635,80.924],[904.062,86.94],[915.824,94.169],[926.805,102.538],[936.893,111.964],[945.987,122.351],[953.997,133.596],[960.843,145.586],[966.454,158.201],[970.775,171.313],[973.764,184.792],[975.389,198.503],[975.634,212.307],[974.497,226.066],[971.99,239.643],[968.138,252.901],[962.978,265.707],[956.564,277.932],[948.958,289.455],[940.238,300.159],[930.492,309.937],[919.816,318.692],[908.318,326.334],[896.113,332.788],[878.814,340.855],[858.055,350.535],[812.386,371.831],[711.915,418.681],[577.778,481.23],[477.306,528.081],[431.637,549.377],[410.878,559.057],[393.58,567.123],[380.408,572.457],[366.757,576.407],[352.772,578.93],[338.601,580],[324.396,579.605],[310.307,577.751],[296.483,574.455],[283.073,569.754],[270.217,563.697],[258.054,556.349],[246.711,547.788],[232.09,535.519],[188.225,498.712],[173.603,486.443],[163.89,477.464],[155.082,467.595],[147.26,456.928],[140.497,445.56],[134.854,433.596],[130.384,421.146],[127.128,408.325],[125.116,395.252],[124.366,382.045],[124.885,368.827],[126.549,349.813],[131.539,292.769],[133.203,273.755],[135.042,260.352],[138.186,247.194],[142.604,234.408],[148.254,222.116],[155.082,210.438],[163.022,199.485],[171.997,189.362],[181.921,180.169],[196.542,167.9],[240.407,131.093],[255.029,118.824],[266.553,110.14],[278.921,102.708],[291.998,96.61],[305.641,91.912],[319.701,88.667],[338.498,85.352],[394.89,75.409],[413.687,72.094],[425.61,70.525],[437.625,70]]),
  Object.freeze([[320.219,89.775],[339.906,89.775],[383.218,89.775],[478.505,89.775],[736.855,89.775],[832.142,89.775],[875.455,89.775],[895.142,89.775],[911.548,89.775],[923.346,90.364],[935.026,92.125],[946.473,95.04],[957.573,99.08],[968.216,104.205],[978.295,110.364],[987.711,117.497],[996.37,125.531],[1004.186,134.388],[1011.082,143.979],[1016.988,154.208],[1021.846,164.975],[1025.608,176.173],[1028.237,187.689],[1029.706,199.41],[1030,211.218],[1029.117,222.998],[1027.066,234.63],[1023.867,246.001],[1019.551,256.997],[1014.162,267.509],[1007.754,277.431],[1000.389,286.667],[992.141,295.123],[983.092,302.716],[973.332,309.37],[962.958,315.019],[952.074,319.607],[936.657,325.219],[918.157,331.952],[877.457,346.766],[787.916,379.356],[474.575,493.403],[385.035,525.993],[344.334,540.806],[325.834,547.54],[310.418,553.151],[299.559,556.523],[288.428,558.838],[277.126,560.076],[265.757,560.225],[254.426,559.283],[243.238,557.26],[232.295,554.174],[221.698,550.053],[211.545,544.935],[201.93,538.868],[188.491,529.458],[148.173,501.228],[134.734,491.817],[125.548,484.689],[117.104,476.694],[109.485,467.91],[102.764,458.421],[97.005,448.319],[92.264,437.702],[88.586,426.671],[86.008,415.332],[83.159,399.176],[74.612,350.705],[71.763,334.548],[70.294,322.827],[70,311.019],[70.883,299.239],[72.934,287.606],[76.133,276.235],[80.449,265.24],[85.838,254.728],[94.041,240.52],[118.65,197.896],[126.853,183.688],[133.406,173.565],[140.953,164.161],[149.417,155.573],[158.711,147.889],[168.737,141.19],[179.392,135.545],[194.261,128.611],[238.868,107.81],[253.737,100.877],[265.725,96.064],[278.167,92.584],[290.913,90.48],[303.812,89.775]]),
  Object.freeze([[522.448,70],[545.422,70],[595.965,70],[693.434,70],[743.977,70],[766.951,70],[786.096,70],[799.995,70.7],[813.754,72.794],[827.232,76.261],[840.293,81.064],[852.806,87.157],[864.642,94.476],[875.683,102.948],[885.817,112.487],[894.94,122.996],[902.96,134.369],[909.797,146.491],[915.381,159.239],[919.655,172.483],[922.576,186.09],[924.114,199.922],[924.254,213.838],[922.994,227.697],[920.348,241.36],[916.342,254.688],[911.016,267.546],[904.425,279.803],[896.635,291.335],[887.725,302.026],[877.786,311.767],[866.918,320.459],[855.231,328.016],[838.651,337.588],[818.755,349.075],[774.983,374.347],[678.686,429.944],[627.364,459.575],[531.066,515.172],[487.295,540.444],[467.398,551.931],[450.818,561.503],[438.431,567.846],[425.469,572.912],[412.063,576.649],[398.35,579.02],[384.468,580],[370.557,579.58],[356.759,577.763],[343.214,574.569],[330.059,570.029],[317.426,564.19],[305.444,557.111],[294.235,548.863],[283.912,539.53],[270.374,525.992],[229.761,485.379],[216.223,471.842],[207.302,462.02],[199.364,451.389],[192.481,440.047],[186.717,428.096],[182.125,415.648],[178.747,402.818],[176.614,389.722],[175.746,376.483],[176.151,363.221],[177.826,350.059],[181.15,331.205],[191.124,274.642],[194.448,255.788],[197.457,242.557],[201.749,229.684],[207.281,217.294],[214.002,205.506],[221.845,194.433],[230.736,184.183],[240.589,174.853],[251.309,166.534],[266.992,155.553],[314.04,122.61],[329.723,111.628],[342.525,103.668],[356.118,97.148],[370.339,92.146],[385.021,88.723],[403.875,85.399],[460.438,75.425],[479.292,72.101],[491.252,70.526],[503.303,70]]),
  Object.freeze([[396.413,70],[418.782,70],[467.994,70],[576.261,70],[685.106,70],[793.373,70],[842.585,70],[864.954,70],[883.595,70],[896.833,70.652],[909.942,72.603],[922.796,75.834],[935.27,80.312],[947.244,85.996],[958.6,92.829],[969.23,100.746],[979.029,109.67],[987.904,119.514],[995.768,130.182],[1002.544,141.573],[1008.168,153.574],[1012.584,166.071],[1015.751,178.941],[1017.636,192.06],[1018.223,205.301],[1017.504,218.535],[1015.488,231.634],[1012.194,244.472],[1007.653,256.924],[1001.91,268.869],[995.02,280.191],[987.05,290.781],[978.078,300.536],[968.189,309.361],[957.482,317.172],[946.057,323.891],[934.028,329.455],[916.744,336.438],[896.004,344.818],[850.375,363.253],[749.992,403.811],[522.192,495.848],[421.809,536.405],[376.181,554.84],[355.44,563.22],[338.157,570.203],[325.653,574.553],[312.783,577.653],[299.67,579.475],[286.442,580],[273.226,579.223],[260.151,577.152],[247.342,573.807],[234.923,569.22],[223.015,563.436],[211.733,556.51],[201.186,548.509],[186.906,536.527],[144.066,500.58],[129.787,488.598],[119.844,479.362],[110.881,469.172],[102.99,458.132],[96.251,446.354],[90.731,433.956],[86.488,421.067],[83.563,407.815],[81.988,394.336],[81.777,380.767],[82.428,362.138],[84.38,306.249],[85.03,287.619],[86.139,274.454],[88.532,261.461],[92.188,248.765],[97.07,236.488],[103.132,224.749],[110.316,213.661],[118.552,203.33],[127.76,193.856],[141.394,181.143],[182.293,143.004],[195.926,130.291],[205.973,121.801],[216.799,114.332],[228.304,107.955],[240.375,102.731],[252.898,98.711],[270.904,93.886],[324.922,79.412],[342.927,74.587],[354.394,72.045],[366.038,70.512],[377.772,70]])
]);

var V5_MAX_ATTEMPTS = 200;         // fixed redraw cap (19,800 cumulative ceiling unchanged)
var V5_TANGENT_TOL_DEG = 1.0;       // tangent-triple collinearity gate (degrees)
var V5_PRE_FILTER_DEG = 1.05;       // relaxed pre-filter: cannot drop a <=1.0° rounded candidate
var V5_PRE_FILTER_TAN = Math.tan(V5_PRE_FILTER_DEG * Math.PI / 180);
var V5_SPATIAL_CELL = 64;           // spatial-hash cell for simplicity sweeps (px)
var V5_BOUNDS_EPS = 1e-3;           // rounding slack on the canvas-margin gate

// FIT-V6 feasibility gates (plan §Half-width & hairpin feasibility). The
// apex-radius metric is intentionally NOT the single-segment spline curvature
// (the spike hairpins have a near-cusp there); it is the minimum circumradius
// of the control-polygon triple spanning three segments around each vertex —
// the effective turning radius a car follows across the hairpin. The plan's
// red-team note explicitly leaves this metric unpinned; the controller/sensor
// falsifier and the G1 wall-byte corpus bound it. At three steps every sharp
// spike template clears 75px (the car's wheelbase/tan(steer) minimum ~70px),
// so 2R >= 150 > 2*hw+20 for the whole ladder.
var V6_APEX_MIN_RADIUS = 75;        // px — must clear the car's min turn radius
var V6_APEX_WINDOW = 3;             // control-point steps in the apex triple
var V6_ARM_CLEARANCE = 20;          // px — inner-wall merge margin

// Circumradius of three points (Infinity for a collinear/degenerate triple —
// a straight is never a feasibility failure). Toroidal indexing is the
// caller's responsibility.
function v6CircumRadius(a, b, c) {
  var abx = b.x - a.x, aby = b.y - a.y;
  var bcx = c.x - b.x, bcy = c.y - b.y;
  var cax = a.x - c.x, cay = a.y - c.y;
  var cross = abx * (c.y - a.y) - aby * (c.x - a.x);
  if (Math.abs(cross) < 1e-12) return Infinity;
  var ab = Math.sqrt(abx * abx + aby * aby);
  var bc = Math.sqrt(bcx * bcx + bcy * bcy);
  var ca = Math.sqrt(cax * cax + cay * cay);
  return ab * bc * ca / (2 * Math.abs(cross));
}

// Minimum effective apex radius over every control vertex, measured on the
// three-step window. Exported so the shape gate is independently testable.
function v6ApexRadius(points) {
  var n = points.length;
  var min = Infinity;
  for (var i = 0; i < n; i++) {
    var r = v6CircumRadius(
      points[(i - V6_APEX_WINDOW + n) % n],
      points[i],
      points[(i + V6_APEX_WINDOW) % n]
    );
    if (r < min) min = r;
  }
  return min;
}

// Hairpin arm separation — by construction 2R (plan §Half-width: "arm
// separation 2R > 2*halfWidth + 20"). Must exceed 2*hw + 20 so the inner wall
// cannot merge/vanish at the widest ladder step.
function v6ArmSeparation(points) {
  return 2 * v6ApexRadius(points);
}

function v5Amplitude(d) {
  // amp(d) = 0.02 + 0.06·d/100  (0.02 → 0.08); plain formula per frozen spec.
  return 0.02 + 0.06 * (d / 100);
}

function v5Centroid(template) {
  var sx = 0, sy = 0;
  for (var i = 0; i < template.length; i++) {
    sx += template[i][0];
    sy += template[i][1];
  }
  return { x: sx / template.length, y: sy / template.length };
}

// Draw order is the caller's responsibility: theta is drawn FIRST, then two
// uint32s dx,dy per point in array order. When `perturbed` is false (fallback)
// the stream is not touched at all (zero perturbation).
function v5Transform(template, cx, cy, theta, amplitude, stream, perturbed, out) {
  var n = template.length;
  var cs = Math.cos(theta), sn = Math.sin(theta);
  for (var i = 0; i < n; i++) {
    var px = template[i][0], py = template[i][1];
    if (perturbed) {
      var dx = stream(), dy = stream();
      var ox0 = px - cx, oy0 = py - cy;
      var dist = Math.sqrt(ox0 * ox0 + oy0 * oy0);
      px = px + (dx * TWO_POW_NEG_32 - 0.5) * 2 * amplitude * dist;
      py = py + (dy * TWO_POW_NEG_32 - 0.5) * 2 * amplitude * dist;
    }
    var ox = px - cx, oy = py - cy;
    var slot = out[i] || (out[i] = { x: 0, y: 0 });
    slot.x = cx + ox * cs - oy * sn;
    slot.y = cy + ox * sn + oy * cs;
  }
  return out;
}

// Uniform scale to fit + translate centroid to canvas centre. `doRound=false`
// is used by the cheap attempt pre-filter so a rejected draw never pays for
// 2·n toFixed() calls; the accepted candidate is rounded before hashing.
function v5ScaleTranslate(raw, cx, cy, halfWidth, doRound) {
  var n = raw.length;
  var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (var i = 0; i < n; i++) {
    if (raw[i].x < minX) minX = raw[i].x;
    if (raw[i].x > maxX) maxX = raw[i].x;
    if (raw[i].y < minY) minY = raw[i].y;
    if (raw[i].y > maxY) maxY = raw[i].y;
  }
  var margin = halfWidth + 20;
  var sx = (1100 - 2 * margin) / (maxX - minX);
  var sy = (650 - 2 * margin) / (maxY - minY);
  var k = Math.min(sx, sy, 1.0);
  // FIT-V6: translate the BBOX centre, not the centroid. Rotation by up to
  // 0.10*TAU makes an asymmetric template's centroid off-centre from its
  // bounding box; centroid translation could push a rotated point outside the
  // canvas (the old fallback then failed its own bounds gate and threw).
  // Bbox-centred translation keeps the scaled polyline inside the margin band
  // by construction (k bounds its size) while leaving angles/scale untouched.
  var boxCX = (minX + maxX) / 2;
  var boxCY = (minY + maxY) / 2;
  var pts = new Array(n);
  for (var j = 0; j < n; j++) {
    var x = TRACK_CENTER_X + (raw[j].x - boxCX) * k;
    var y = TRACK_CENTER_Y + (raw[j].y - boxCY) * k;
    pts[j] = { x: doRound ? round3(x) : x, y: doRound ? round3(y) : y };
  }
  return pts;
}

// atan2-free version of the relaxed pre-filter: returns true only when the
// triple deviation is <= V5_PRE_FILTER_DEG (dot <= 0 => >=90deg => reject).
function v5TangentWithinPreFilter(points) {
  var n = points.length;
  var a = points[n - 1], b = points[0], c = points[1];
  var v1x = b.x - a.x, v1y = b.y - a.y, v2x = c.x - b.x, v2y = c.y - b.y;
  var dot = v1x * v2x + v1y * v2y;
  if (dot <= 0) return false;
  var cr = Math.abs(v1x * v2y - v1y * v2x);
  return cr <= V5_PRE_FILTER_TAN * dot;
}

// Angle (degrees) between the two chords of the triple pts[n-1], pts[0],
// pts[1] — the triple that sets the arcPos-0 tangent in track.js:111-115.
function v5TangentDeviation(points) {
  var n = points.length;
  var a = points[n - 1], b = points[0], c = points[1];
  var v1x = b.x - a.x, v1y = b.y - a.y, v2x = c.x - b.x, v2y = c.y - b.y;
  var dot = v1x * v2x + v1y * v2y;
  var cr = Math.abs(v1x * v2y - v1y * v2x);
  return Math.atan2(cr, dot) * 180 / Math.PI;
}

// Numeric cell key used by the wall-soundness sweep below (cell = 64px).
function v5CellKey(cx, cy) {
  return (cx + 64) * 256 + (cy + 64);
}

function v5TrackConstructor() {
  var T = (typeof globalThis !== 'undefined') ? globalThis.Track : undefined;
  if (typeof T !== 'function') {
    throw new Error('rotation.js V2 validation requires js/track.js to be loaded first (Track)');
  }
  return T;
}

// Offset-wall soundness (spatial-hash sweep, never O(n^2) over all pairs).
// The frozen F1 templates deliberately include tight hairpins: their inner
// offset wall folds back over itself at the road widths of the difficulty
// ladder (numerically verified for all six templates at every d, including
// each template's hw-base). A strict proper-crossing test on the DENSE walls
// therefore rejects every template and makes the mandated never-throws
// fallback impossible. What actually breaks the renderer/raycaster is a
// COLLAPSED wall, so this gate rejects finite violations, zero-length wall
// segments and non-adjacent EXACTLY-duplicate wall segments. The dense SPLINE
// samples themselves are still held to the full proper-crossing simplicity
// gate below, which keeps the centreline geometry sane.
function v5WallsSound(track) {
  var walls = [track.leftWall, track.rightWall];
  for (var s = 0; s < walls.length; s++) {
    var pts = walls[s];
    var n = pts.length;
    var grid = new Map();
    var boxes = new Array(n);
    var seen = new Int32Array(n);
    var i, j, cx, cy;
    for (i = 0; i < n; i++) {
      var a = pts[i], b = pts[(i + 1) % n];
      if (!Number.isFinite(a.x) || !Number.isFinite(a.y) ||
          !Number.isFinite(b.x) || !Number.isFinite(b.y)) return false;
      if (a.x === b.x && a.y === b.y) return false; // zero-length wall segment
      var minX = Math.floor(Math.min(a.x, b.x) / V5_SPATIAL_CELL), maxX = Math.floor(Math.max(a.x, b.x) / V5_SPATIAL_CELL);
      var minY = Math.floor(Math.min(a.y, b.y) / V5_SPATIAL_CELL), maxY = Math.floor(Math.max(a.y, b.y) / V5_SPATIAL_CELL);
      boxes[i] = [minX, maxX, minY, maxY];
      for (cx = minX; cx <= maxX; cx++) {
        for (cy = minY; cy <= maxY; cy++) {
          var key = v5CellKey(cx, cy);
          var bucket = grid.get(key);
          if (!bucket) { bucket = []; grid.set(key, bucket); }
          bucket.push(i);
        }
      }
    }
    for (i = 0; i < n; i++) {
      var a2 = pts[i], b2 = pts[(i + 1) % n];
      var box = boxes[i], stamp = i + 1;
      for (cx = box[0]; cx <= box[1]; cx++) {
        for (cy = box[2]; cy <= box[3]; cy++) {
          var bucket2 = grid.get(v5CellKey(cx, cy));
          if (!bucket2) continue;
          for (var bi = 0; bi < bucket2.length; bi++) {
            j = bucket2[bi];
            if (seen[j] === stamp) continue;
            seen[j] = stamp;
            if (j === i || j === i + 1 || j + 1 === i || (i === 0 && j === n - 1) || (j === 0 && i === n - 1)) continue;
            var c = pts[j], d = pts[(j + 1) % n];
            // Only exact segment collapse is fatal (identical endpoints).
            if ((a2.x === c.x && a2.y === c.y && b2.x === d.x && b2.y === d.y) ||
                (a2.x === d.x && a2.y === d.y && b2.x === c.x && b2.y === c.y)) return false;
          }
        }
      }
    }
  }
  return true;
}

// FIT-V6 cycle-3 S1 — dense-spline clipped-wall clearance gate.
// The car is a circle of radius 14px (car.js). Require >= 18px (CAR_RADIUS+4)
// between EVERY centreline sample and EVERY clipped wall segment. Reuse the
// single canonical Track built during validation; fail closed on any violation.
var V6_MIN_CLEARANCE = 18; // CAR_RADIUS(14) + 4

// Minimum circumradius over consecutive centreline samples (the effective
// spline turning radius through a hairpin apex). Infinity for a straight.
function v6SplineRadius(samples) {
  var N = samples.length, min = Infinity;
  for (var i = 0; i < N; i++) {
    var a = samples[(i - 1 + N) % N], b = samples[i], c = samples[(i + 1) % N];
    var abx = b.x - a.x, aby = b.y - a.y, cbx = c.x - a.x, cby = c.y - a.y;
    var cr = Math.abs(abx * cby - aby * cbx);
    if (cr < 1e-12) continue;
    var r = Math.sqrt(abx * abx + aby * aby) * Math.sqrt((b.x - c.x) * (b.x - c.x) + (b.y - c.y) * (b.y - c.y)) *
      Math.sqrt(cbx * cbx + cby * cby) / (2 * cr);
    if (r < min) min = r;
  }
  return min;
}

// Exact min distance track.samples x track.segments, spatial-hash accelerated.
// A segment within 18px of a sample must overlap that sample's 64px cell or an
// adjacent one, so a +/-1 cell ring is exact (verified against a brute-force
// O(samples x segments) sweep: 297/297 identical). Builds one grid per candidate.
function v6TrackClearance(track) {
  var CELL = 64, RING = 1, min2 = Infinity;
  var grid = new Map(), segs = track.segments;
  var key = function (cx, cy) { return (cx + 1024) * 8192 + (cy + 1024); };
  for (var si = 0; si < segs.length; si++) {
    var g = segs[si];
    var mnx = Math.floor(Math.min(g.x1, g.x2) / CELL), mxx = Math.floor(Math.max(g.x1, g.x2) / CELL);
    var mny = Math.floor(Math.min(g.y1, g.y2) / CELL), mxy = Math.floor(Math.max(g.y1, g.y2) / CELL);
    for (var cx = mnx; cx <= mxx; cx++) {
      for (var cy = mny; cy <= mxy; cy++) {
        var k = key(cx, cy), bucket = grid.get(k);
        if (!bucket) { bucket = []; grid.set(k, bucket); }
        bucket.push(si);
      }
    }
  }
  var samples = track.samples;
  for (var s = 0; s < samples.length; s++) {
    var sm = samples[s], scx = Math.floor(sm.x / CELL), scy = Math.floor(sm.y / CELL);
    for (var gx = scx - RING; gx <= scx + RING; gx++) {
      for (var gy = scy - RING; gy <= scy + RING; gy++) {
        var b = grid.get(key(gx, gy));
        if (!b) continue;
        for (var bi = 0; bi < b.length; bi++) {
          var g2 = segs[b[bi]], dx = g2.x2 - g2.x1, dy = g2.y2 - g2.y1, l2 = dx * dx + dy * dy;
          var t = l2 > 0 ? ((sm.x - g2.x1) * dx + (sm.y - g2.y1) * dy) / l2 : 0;
          if (t < 0) t = 0; else if (t > 1) t = 1;
          var ex = sm.x - (g2.x1 + t * dx), ey = sm.y - (g2.y1 + t * dy), d2 = ex * ex + ey * ey;
          if (d2 < min2) min2 = d2;
        }
      }
    }
    if (min2 < V6_MIN_CLEARANCE * V6_MIN_CLEARANCE) return Math.sqrt(min2);
  }
  return Math.sqrt(min2);
}

// The full step-7 validation gate, in the frozen order (FIT-V6 rewrite).
// Builds exactly ONE canonical clipped Track — the same object is returned to
// the generator on success, so the accepted candidate is never re-built
// ("reuse the single Track built during validation", plan C9). Returns the
// Track on success, false on rejection; the generator alone decides whether a
// rejection redraws or falls back.
function v5Valid(points, halfWidth) {
  // 1. finite coordinates
  if (!pointsAreFinite(points)) return false;
  var n = points.length;
  // 2. bounding box within the canvas. Step 5's margin-based scale targets the
  // inset band; the gate enforces the hard canvas bounds (the centroid-based
  // translation of step 5 is not bbox-centred, so an asymmetric template may
  // sit slightly inside the inset band but is always inside the canvas).
  for (var i = 0; i < n; i++) {
    if (points[i].x < -V5_BOUNDS_EPS || points[i].x > 1100 + V5_BOUNDS_EPS) return false;
    if (points[i].y < -V5_BOUNDS_EPS || points[i].y > 650 + V5_BOUNDS_EPS) return false;
  }
  // 3. control polygon simple
  if (controlPolygonSelfIntersects(points)) return false;
  // 4. tangent triple pts[n-1],pts[0],pts[1] within 1 degree
  if (v5TangentDeviation(points) > V5_TANGENT_TOL_DEG) return false;
  // 5. feasibility: apex radius >= 75 and arm separation 2R > 2*hw + 20
  if (v6ApexRadius(points) < V6_APEX_MIN_RADIUS) return false;
  if (v6ArmSeparation(points) <= 2 * halfWidth + V6_ARM_CLEARANCE) return false;
  // 6/7. canonical clipped Track geometry: G1 guarantees leftWall/rightWall
  // are one simple closed polyline per side and segments/stamps/grid are
  // rebuilt from them with their own lengths. Build it ONCE and validate the
  // invariants + canonical wall soundness on that SAME object. A
  // TrackWallError (or any throw) is a rejection, never a crash. Note: the
  // old FIT-V5 `v5PolylineSimple(track.samples)` gate is deliberately GONE —
  // the dense centreline spline legitimately folds at a genuine hairpin; the
  // whole point of the G1 clipped-wall model is to resolve that fold into one
  // simple wall polyline per side. Re-checking raw-sample simplicity here
  // would reject every sharp spike template (the exact defect FIT-V5 had).
  var track;
  try {
    track = new (v5TrackConstructor())(points, halfWidth);
  } catch (err) {
    return false;
  }
  if (track.leftWall.length < 3 || track.rightWall.length < 3) return false;
  if (track.leftWall.length !== track.rightWall.length) return false;
  if (track.segments.length !== 2 * track.leftWall.length) return false;
  if (track._segStamp && track._segStamp.length !== track.segments.length) return false;
  if (!pointsAreFinite(track.leftWall) || !pointsAreFinite(track.rightWall)) return false;
  if (!v5WallsSound(track)) return false;
  // 7b. FIT-V6 cycle-3 S1 (fail closed): the spline turning radius through
  // every hairpin apex and the centreline-to-CLIPPED-wall clearance on this
  // SAME canonical Track must clear the car circle. The old sharp templates
  // had a 0.41px spline radius / 0.86px clearance at hw 50 (S1 crash).
  if (v6SplineRadius(track.samples) < V6_APEX_MIN_RADIUS) return false;
  if (v6TrackClearance(track) < V6_MIN_CLEARANCE) return false;
  // (8. pairwise-distinct hash from other rotations is a caller/db check.)
  return track;
}

function generateTrackV2(rotationSeed, rotationIndex) {
  validateRotationSeed(rotationSeed);
  validateRotationIndex(rotationIndex);
  var difficulty = difficultyForRotation(rotationIndex);
  var d = Math.min(rotationIndex, 100);
  var halfWidth = halfWidthForDifficulty(difficulty);
  var templateIndex = ((rotationSeed ^ Math.imul(rotationIndex + 1, 0x9E3779B1)) >>> 0) % 6;
  var template = V5_TEMPLATES[templateIndex];
  var centroid = v5Centroid(template);
  var amplitude = v5Amplitude(d);
  var stream = rotationMulberry32((rotationSeed ^ Math.imul(rotationIndex + 1, 0x85EBCA6B)) >>> 0);
  var raw = new Array(template.length); // reused across attempts (no per-draw allocation)

  for (var attempt = 0; attempt < V5_MAX_ATTEMPTS; attempt++) {
    // Draw order: one uint32 for theta, then two uint32s (dx,dy) per point in
    // array order. A FULL redraw draws a fresh theta AND fresh dx/dy for every
    // point — no draw is ever reused from a previous attempt.
    var theta = TAU * (stream() * TWO_POW_NEG_32) * 0.10;
    v5Transform(template, centroid.x, centroid.y, theta, amplitude, stream, true, raw);
    // Cheap relaxed-tangent pre-filter. A uniform positive scale + translation
    // (step 5) preserve angles, so the deviation on `raw` equals the deviation
    // on the final points. Rounding can move it by <0.002deg, so a candidate
    // above 1.05deg here cannot pass the exact <=1.0deg gate. This avoids the
    // scale/round/Track work on the ~95% of draws the tangent gate rejects.
    if (!v5TangentWithinPreFilter(raw)) continue;
    var unrounded = v5ScaleTranslate(raw, centroid.x, centroid.y, halfWidth, false);
    var points = new Array(unrounded.length);
    for (var i = 0; i < unrounded.length; i++) {
      points[i] = { x: round3(unrounded[i].x), y: round3(unrounded[i].y) };
    }
    var acceptedTrack = v5Valid(points, halfWidth);
    if (acceptedTrack) {
      // Reuse the single Track built during validation — no rebuild.
      return { points: points, halfWidth: halfWidth, difficulty: difficulty, track: acceptedTrack };
    }
  }

  // Zero-perturbation, r-dependent fallback (O(1), non-recursive, function of
  // (seed, r, d)) — validated by the SAME step-7 gate. Fail closed: a broken
  // fallback throws a typed ValidationError instead of returning bad points.
  var fbTheta = TAU * ((((rotationSeed ^ rotationIndex) >>> 0) * TWO_POW_NEG_32)) * 0.10;
  var fbRaw = v5Transform(template, centroid.x, centroid.y, fbTheta, 0, null, false, new Array(template.length));
  var fallback = v5ScaleTranslate(fbRaw, centroid.x, centroid.y, halfWidth, true);
  var fallbackTrack = v5Valid(fallback, halfWidth);
  if (!fallbackTrack) {
    throw new ValidationError(
      'FIT-V6 zero-perturbation fallback failed validation (seed=' + rotationSeed + ', r=' + rotationIndex + ')'
    );
  }
  return { points: fallback, halfWidth: halfWidth, difficulty: difficulty, track: fallbackTrack };
}

// ---------------------------------------------------------------------------
// FIT-V4 version contract — ONE parser, TWO boundary modes. rotation.js is the
// single home loaded in every realm (main, fallback core, Worker importScripts
// list at train-worker.js:80, and Node tests). db.js re-exports these; no other
// module redefines an inline `=== 2` / `|| 1` check.
// ---------------------------------------------------------------------------
function parseGeneratorVersion(raw) {
  if (raw === undefined || raw === null) return 1;   // meta absent => legacy V1
  if (raw === 1 || raw === '1') return 1;
  if (raw === 2 || raw === '2') return 2;
  // '', whitespace-only and every other value fail closed (no silent downgrade).
  throw new ValidationError('invalid generator version: ' + String(raw));
}

function requireGeneratorVersion(raw) {
  // Message/state boundary: a missing or empty handshake field is malformed,
  // never a default. Reject before delegating to the canonical parser.
  if (raw === undefined || raw === null || raw === '') {
    throw new ValidationError('generator version is required');
  }
  return parseGeneratorVersion(raw);
}

// Canonical generator dispatch — the ONLY place that maps an integer version
// to a generator. No caller re-interprets the version inline.
var GENERATORS_BY_VERSION = Object.freeze({ 1: generateTrack, 2: generateTrackV2 });

function validateGeneratorVersion(version) {
  if (!Number.isInteger(version) || GENERATORS_BY_VERSION[version] === undefined) {
    throw new ValidationError('generator version must be integer 1 or 2, got ' + String(version));
  }
}

// resolveGeneratedTrack(rotationSeed, rotationIndex)
//   → { controlPoints: [{x,y} × 14], halfWidth, difficulty, trackHash }
// trackHash = cyrb53(canonicalJSON({ points: controlPoints, halfWidth })) —
// the SAME surrogate generations.track_hash uses. The bytes are canonical at
// the source (3-decimal points), so parse/serialize round-trips are
// byte-identical across runs and engines-in-family.
function resolveGeneratedTrack(rotationSeed, rotationIndex, version) {
  if (version === undefined) version = 1; // legacy 2-arg call sites only; production passes explicit version
  validateGeneratorVersion(version);
  validateRotationSeed(rotationSeed);
  validateRotationIndex(rotationIndex);
  var gen = GENERATORS_BY_VERSION[version](rotationSeed, rotationIndex);
  var canonicalJSONFn = (typeof globalThis !== 'undefined') ? globalThis.canonicalJSON : undefined;
  var cyrb53Fn = (typeof globalThis !== 'undefined') ? globalThis.cyrb53 : undefined;
  if (typeof canonicalJSONFn !== 'function' || typeof cyrb53Fn !== 'function') {
    throw new Error('rotation.js requires js/track.js to be loaded first (canonicalJSON / cyrb53)');
  }
  var trackHash = cyrb53Fn(canonicalJSONFn({ points: gen.points, halfWidth: gen.halfWidth }));
  // `track` is the SAME canonical clipped Track built during generator
  // validation (no rebuild). It is additive; the lineage bytes above are
  // unchanged and consumers that rebuild remain byte-compatible.
  return {
    controlPoints: gen.points,
    halfWidth: gen.halfWidth,
    difficulty: gen.difficulty,
    trackHash: trackHash,
    track: gen.track,
  };
}

// Browser-global exposure (boot contract asserts globalThis.Rotation) +
// guarded CommonJS export (node tests). ROTATE_EVERY is an integer constant;
// the exported object is frozen so the ladders cannot be mutated by
// consumers.
var Rotation = Object.freeze({
  ROTATE_EVERY: ROTATE_EVERY,
  TRACK_POINTS: TRACK_POINTS,
  TRACK_CENTER: Object.freeze({ x: TRACK_CENTER_X, y: TRACK_CENTER_Y }),
  TRACK_MAX_RADIUS: TRACK_MAX_RADIUS,
  TRACK_FALLBACK_RADIUS: TRACK_FALLBACK_RADIUS,
  TRACK_MAX_REJECTS: TRACK_MAX_REJECTS,
  rotationForGeneration: rotationForGeneration,
  difficultyForRotation: difficultyForRotation,
  halfWidthForDifficulty: halfWidthForDifficulty,
  harmonicsForDifficulty: harmonicsForDifficulty,
  amplitudeForDifficulty: amplitudeForDifficulty,
  generateTrack: generateTrack,
  // FIT-V4 additions (V1 entries above are unchanged).
  TRACK_POINTS_V2: TRACK_POINTS_V2,
  THETAS36: THETAS36,
  harmonicsV2ForDifficulty: harmonicsV2ForDifficulty,
  amplitudeV2ForDifficulty: amplitudeV2ForDifficulty,
  generateTrackV2: generateTrackV2,
  generateTrackV2Fallback: generateTrackV2Fallback,
  // FIT-V5 additions (V1 entries above are unchanged).
  V5_TEMPLATES: V5_TEMPLATES,
  parseGeneratorVersion: parseGeneratorVersion,
  requireGeneratorVersion: requireGeneratorVersion,
  resolveGeneratedTrack: resolveGeneratedTrack,
  // FIT-V6 G2 additions — feasibility-gate oracles (V1 entries unchanged).
  v6ApexRadius: v6ApexRadius,
  v6ArmSeparation: v6ArmSeparation,
  // FIT-V6 cycle-3 S1 seams (additive; the generator itself uses these
  // internally — exported for the clearance/fail-closed gate tests).
  v5Valid: v5Valid,
  v6TrackClearance: v6TrackClearance,
  v6SplineRadius: v6SplineRadius,
});

if (typeof globalThis !== 'undefined') {
  globalThis.Rotation = Rotation;
  globalThis.parseGeneratorVersion = parseGeneratorVersion;
  globalThis.requireGeneratorVersion = requireGeneratorVersion;
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    Rotation: Rotation,
    ROTATE_EVERY: ROTATE_EVERY,
    ValidationError: ValidationError,
    parseGeneratorVersion: parseGeneratorVersion,
    requireGeneratorVersion: requireGeneratorVersion,
    generateTrackV2: generateTrackV2,
    THETAS36: THETAS36,
    TRACK_POINTS_V2: TRACK_POINTS_V2,
    V5_TEMPLATES: V5_TEMPLATES,
    v6ApexRadius: v6ApexRadius,
    v6ArmSeparation: v6ArmSeparation,
    v5Valid: v5Valid,
    v6TrackClearance: v6TrackClearance,
    v6SplineRadius: v6SplineRadius,
  };
}
