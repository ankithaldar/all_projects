// tests/rotation-v2.mjs — versioned generator contract (FIT-V4 TASK A,
// migrated to the FIT-V6 shipped contract). Plain Node, guarded CommonJS
// exports, no dependencies, same style as tests/smoke.mjs.
// Run with:  node tests/rotation-v2.mjs
//
// FIT-V6 migration (G4): the shared half-width ladder is now 50 - 0.25 d and
// the V2 generator is the original sharp F1 template family. The V1 golden
// bytes below were regenerated against the new ladder (the old ladder's bytes
// are a globally-accepted breaking change) and the V2 golden bytes/shape
// assertions were regenerated against the shipped sharp-template generator.
// The comprehensive V2 corpus (495 hashes, apex/arm feasibility, 40k
// never-throws cases) is owned by tests/rotation-v6.mjs; this suite keeps the
// parser/version-dispatch contract and an independent 20-vector golden set.

import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { canonicalJSON, cyrb53 } = require('../js/track.js');
const { mulberry32 } = require('../js/nn.js');
const rotationMod = require('../js/rotation.js');
const { Rotation, ValidationError } = rotationMod;

// -----------------------------------------------------------------------------
// Tiny harness — one line per test, never throws, exit 1 on any failure.
// -----------------------------------------------------------------------------
let passed = 0;
const failures = [];
async function test(name, fn) {
  try {
    await fn();
    passed += 1;
    console.log('  ok   - ' + name);
  } catch (err) {
    failures.push({ name, err });
    const detail = String((err && err.stack) || err)
      .split('\n')
      .map((line) => '         ' + line)
      .join('\n');
    console.log('  FAIL - ' + name + '\n' + detail);
  }
}
function assert(cond, msg) {
  if (!cond) throw new Error('assert failed: ' + msg);
}
function assertValidationError(fn, msg) {
  let threw = null;
  try { fn(); } catch (err) { threw = err; }
  assert(threw !== null, msg + ' - expected ValidationError, nothing thrown');
  assert(threw.name === 'ValidationError', msg + ' - expected ValidationError, got ' + threw.name + ': ' + threw.message);
}

// -----------------------------------------------------------------------------
// Independent helpers (written from the frozen spec, NOT imported from prod).
// -----------------------------------------------------------------------------
const TAU = Math.PI * 2;
const T32 = 1 / 4294967296;
const CENTER = { x: 550, y: 325 };
const round3 = (v) => Number(v.toFixed(3));
function refMulberry(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (t ^ (t >>> 14)) >>> 0;
  };
}
const orient = (ax, ay, bx, by, cx, cy) => (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
function segmentsCross(ax, ay, bx, by, cx, cy, dx, dy) {
  const o1 = orient(ax, ay, bx, by, cx, cy);
  const o2 = orient(ax, ay, bx, by, dx, dy);
  const o3 = orient(cx, cy, dx, dy, ax, ay);
  const o4 = orient(cx, cy, dx, dy, bx, by);
  return ((o1 > 0 && o2 < 0) || (o1 < 0 && o2 > 0)) && ((o3 > 0 && o4 < 0) || (o3 < 0 && o4 > 0));
}
function polygonSelfIntersects(points) {
  const n = points.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (j === i + 1) continue;
      if (i === 0 && j === n - 1) continue;
      const a = points[i], b = points[(i + 1) % n], c = points[j], d = points[(j + 1) % n];
      if (segmentsCross(a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y)) return true;
    }
  }
  return false;
}
// turnCount = #{i : cross(e_i, e_{i+1}) < 0} on the closed control polygon.
function turnCount(points) {
  const n = points.length;
  let c = 0;
  for (let i = 0; i < n; i++) {
    const a = points[i], b = points[(i + 1) % n], d = points[(i + 2) % n];
    const e1x = b.x - a.x, e1y = b.y - a.y, e2x = d.x - b.x, e2y = d.y - b.y;
    if (e1x * e2y - e1y * e2x < 0) c += 1;
  }
  return c;
}
const hashOf = (points, halfWidth) => cyrb53(canonicalJSON({ points, halfWidth }));
const halfWidthOf = (r) => Number((50 - 0.25 * Math.min(r, 100)).toFixed(9));
const isFallback = (points) => points.every((p) =>
  Math.abs(Math.hypot(p.x - CENTER.x, p.y - CENTER.y) - 170) <= 1e-3);

// Independent V2 rho reference (frozen table): min/max rho over every attempt.
function refV2Rho(seed, r) {
  const d = Math.min(r, 100), K = 4 + Math.floor(d / 16);
  const A = Number((0.12 + 0.33 * (d / 100)).toFixed(9));
  const stream = refMulberry((seed ^ Math.imul(r + 1, 0x85EBCA6B)) >>> 0);
  let wSum = 0;
  for (let k = 1; k <= K; k++) wSum += 1 / Math.pow(k, 1.5);
  const amp = Array.from({ length: K }, (_, a) => A * (1 / Math.pow(a + 1, 1.5)) / wSum);
  let minRho = Infinity;
  for (let attempt = 0; attempt < 200; attempt++) {
    const phases = Array.from({ length: K }, () => TAU * (stream() * T32));
    const rho = new Array(36);
    let maxRho = -Infinity;
    for (let i = 0; i < 36; i++) {
      const theta = TAU * i / 36;
      let acc = 0;
      for (let a = 0; a < K; a++) acc += amp[a] * Math.sin((a + 1) * theta + phases[a]);
      rho[i] = 1 + acc;
      if (rho[i] < minRho) minRho = rho[i];
      if (rho[i] > maxRho) maxRho = rho[i];
    }
    // Mirror the generator's stream semantics: stop once a track is accepted.
    const r0 = 245 / Math.max(1, maxRho);
    const pts = Array.from({ length: 36 }, (_, i) => ({
      x: round3(CENTER.x + r0 * rho[i] * Math.cos(TAU * i / 36)),
      y: round3(CENTER.y + r0 * rho[i] * Math.sin(TAU * i / 36)),
    }));
    if (pts.every((p) => Number.isFinite(p.x) && Number.isFinite(p.y)) && !polygonSelfIntersects(pts)) break;
  }
  return minRho;
}

// -----------------------------------------------------------------------------
// FROZEN CORPORA
// -----------------------------------------------------------------------------
const SEEDS = [0, 1, 12345, 0xDEADBEEF >>> 0, 0xFFFFFFFF];
const RS = [1, 2, 50, 99];

// V1 golden hashes regenerated against the FIT-V6 50-0.25 d ladder (seed x r).
const GOLDEN_V1 = [
  [0, 1, '41da7bf2ff4bc5ed'], [0, 2, '4710ef3aa4609772'],
  [0, 50, 'cf93a3f4bb85dfa0'], [0, 99, '9fdb255e7d49c1fb'],
  [1, 1, 'caab32e6e15bd6d7'], [1, 2, '67139ff7454b023a'],
  [1, 50, '1399522a57b967e3'], [1, 99, 'b47f7b90209dde22'],
  [12345, 1, '1855f2d7ccca475f'], [12345, 2, '91fe7b82f45a8074'],
  [12345, 50, '7fe82c2ed11a487f'], [12345, 99, '1b9883149a735ef6'],
  [0xDEADBEEF>>>0, 1, '96c4eab5ebb61b40'], [0xDEADBEEF>>>0, 2, '80f9ea11f09f0595'],
  [0xDEADBEEF>>>0, 50, '700584ba66dcd5a0'], [0xDEADBEEF>>>0, 99, 'aa2edc869b297579'],
  [0xFFFFFFFF, 1, '99de0fe90dd709ae'], [0xFFFFFFFF, 2, 'f027c4cf4f5ae6be'],
  [0xFFFFFFFF, 50, '5223dede9a221269'], [0xFFFFFFFF, 99, 'eb8f63e7c0556139'],
];
// V2 golden hashes regenerated against the shipped sharp-template generator
// (FIT-V6). The full 5-seed x r=1..99 corpus lives in rotation-v6.mjs.
const GOLDEN_V2 = [
  [0, 1, '9a44a5c9dadd0597'], [0, 2, 'acf17e3f35df9371'], [0, 50, 'fe06aa68e72580a5'], [0, 99, '47fb42f3868710df'],
  [1, 1, 'f73efc03c72ad1fe'], [1, 2, 'c53b4eb9edab5861'], [1, 50, '0cc946187de240e7'], [1, 99, '84efa5da34c93c98'],
  [12345, 1, '6deb504c68c9cff3'], [12345, 2, 'f446281e6e01022c'], [12345, 50, '7fac3f5ae5f5007e'], [12345, 99, '912fc29e79c9c5c5'],
  [0xDEADBEEF>>>0, 1, 'f677a1db129583a3'], [0xDEADBEEF>>>0, 2, '1be2e6c74eabf6fd'], [0xDEADBEEF>>>0, 50, '87af15461ef5a1b1'], [0xDEADBEEF>>>0, 99, 'db8aa7b51953621f'],
  [0xFFFFFFFF, 1, '1d492f904ced3b79'], [0xFFFFFFFF, 2, '0b1390463c46d354'], [0xFFFFFFFF, 50, 'e039226d58a3b55f'], [0xFFFFFFFF, 99, 'bc8d3b1dce9499e6']

];

// Full canonical bytes for two V1 vectors (byte-identity proof), regenerated
// against the FIT-V6 50-0.25 d ladder.
const V1_BYTES = new Map([
  ['0|1', `{"points":[{"x":777.124,"y":325},{"x":745.961,"y":419.37},{"x":682.511,"y":491.163},{"x":597.586,"y":533.49},{"x":501.092,"y":539.28},{"x":408.62,"y":502.285},{"x":341.182,"y":425.561},{"x":315.74,"y":325},{"x":337.501,"y":222.666},{"x":401.317,"y":138.557},{"x":496.097,"y":88.836},{"x":604.518,"y":86.143},{"x":702.005,"y":134.392},{"x":763.77,"y":222.054}],"halfWidth":49.75}`],
  ['4294967295|99', `{"points":[{"x":773.235,"y":325},{"x":713.771,"y":403.868},{"x":650.237,"y":450.693},{"x":585.294,"y":479.632},{"x":519.314,"y":459.446},{"x":460.531,"y":437.191},{"x":381.026,"y":406.374},{"x":336.429,"y":325},{"x":343.671,"y":225.637},{"x":402.342,"y":139.843},{"x":500.366,"y":107.54},{"x":600.047,"y":105.729},{"x":701.303,"y":135.271},{"x":770.737,"y":218.698}],"halfWidth":25.25}`],
]);
// Full canonical bytes for two V2 vectors (byte-identity proof), regenerated
// against the shipped sharp-template generator.
const V2_BYTES = new Map([
  ['0|1', `{"points":[{"x":522.448,"y":70},{"x":545.422,"y":70},{"x":595.965,"y":70},{"x":693.434,"y":70},{"x":743.977,"y":70},{"x":766.951,"y":70},{"x":786.096,"y":70},{"x":799.995,"y":70.7},{"x":813.754,"y":72.794},{"x":827.232,"y":76.261},{"x":840.293,"y":81.064},{"x":852.806,"y":87.157},{"x":864.642,"y":94.476},{"x":875.683,"y":102.948},{"x":885.817,"y":112.487},{"x":894.94,"y":122.996},{"x":902.96,"y":134.369},{"x":909.797,"y":146.491},{"x":915.381,"y":159.239},{"x":919.655,"y":172.483},{"x":922.576,"y":186.09},{"x":924.114,"y":199.922},{"x":924.254,"y":213.838},{"x":922.994,"y":227.697},{"x":920.348,"y":241.36},{"x":916.342,"y":254.688},{"x":911.016,"y":267.546},{"x":904.425,"y":279.803},{"x":896.635,"y":291.335},{"x":887.725,"y":302.026},{"x":877.786,"y":311.767},{"x":866.918,"y":320.459},{"x":855.231,"y":328.016},{"x":838.651,"y":337.588},{"x":818.755,"y":349.075},{"x":774.983,"y":374.347},{"x":678.686,"y":429.944},{"x":627.364,"y":459.575},{"x":531.066,"y":515.172},{"x":487.295,"y":540.444},{"x":467.398,"y":551.931},{"x":450.818,"y":561.503},{"x":438.431,"y":567.846},{"x":425.469,"y":572.912},{"x":412.063,"y":576.649},{"x":398.35,"y":579.02},{"x":384.468,"y":580},{"x":370.557,"y":579.58},{"x":356.759,"y":577.763},{"x":343.214,"y":574.569},{"x":330.059,"y":570.029},{"x":317.426,"y":564.19},{"x":305.444,"y":557.111},{"x":294.235,"y":548.863},{"x":283.912,"y":539.53},{"x":270.374,"y":525.992},{"x":229.761,"y":485.379},{"x":216.223,"y":471.842},{"x":207.302,"y":462.02},{"x":199.364,"y":451.389},{"x":192.481,"y":440.047},{"x":186.717,"y":428.096},{"x":182.125,"y":415.648},{"x":178.747,"y":402.818},{"x":176.614,"y":389.722},{"x":175.746,"y":376.483},{"x":176.151,"y":363.221},{"x":177.826,"y":350.059},{"x":181.15,"y":331.205},{"x":191.124,"y":274.642},{"x":194.448,"y":255.788},{"x":197.457,"y":242.557},{"x":201.749,"y":229.684},{"x":207.281,"y":217.294},{"x":214.002,"y":205.506},{"x":221.845,"y":194.433},{"x":230.736,"y":184.183},{"x":240.589,"y":174.853},{"x":251.309,"y":166.534},{"x":266.992,"y":155.553},{"x":314.04,"y":122.61},{"x":329.723,"y":111.628},{"x":342.525,"y":103.668},{"x":356.118,"y":97.148},{"x":370.339,"y":92.146},{"x":385.021,"y":88.723},{"x":403.875,"y":85.399},{"x":460.438,"y":75.425},{"x":479.292,"y":72.101},{"x":491.252,"y":70.526},{"x":503.303,"y":70}],"halfWidth":49.75}`],
  ['3735928559|99', `{"points":[{"x":503.264,"y":96.56},{"x":520.956,"y":107.323},{"x":559.879,"y":131.002},{"x":645.511,"y":183.096},{"x":801.718,"y":278.125},{"x":887.35,"y":330.219},{"x":926.273,"y":353.897},{"x":943.966,"y":364.661},{"x":958.709,"y":373.63},{"x":968.989,"y":380.609},{"x":978.524,"y":388.576},{"x":987.217,"y":397.454},{"x":994.984,"y":407.154},{"x":1001.746,"y":417.577},{"x":1007.437,"y":428.623},{"x":1011.999,"y":440.181},{"x":1015.389,"y":452.135},{"x":1017.57,"y":464.367},{"x":1018.524,"y":476.756},{"x":1018.239,"y":489.178},{"x":1016.718,"y":501.51},{"x":1013.978,"y":513.629},{"x":1010.044,"y":525.416},{"x":1004.956,"y":536.751},{"x":998.765,"y":547.525},{"x":991.532,"y":557.628},{"x":983.329,"y":566.96},{"x":974.238,"y":575.43},{"x":964.347,"y":582.953},{"x":953.758,"y":589.453},{"x":942.573,"y":594.866},{"x":930.907,"y":599.139},{"x":918.871,"y":602.229},{"x":906.588,"y":604.106},{"x":894.179,"y":604.75},{"x":881.768,"y":604.156},{"x":869.479,"y":602.328},{"x":852.556,"y":598.943},{"x":832.249,"y":594.879},{"x":787.575,"y":585.941},{"x":689.29,"y":566.276},{"x":374.756,"y":503.345},{"x":276.471,"y":483.68},{"x":231.796,"y":474.742},{"x":211.49,"y":470.68},{"x":194.567,"y":467.293},{"x":182.272,"y":464.175},{"x":170.359,"y":459.824},{"x":158.948,"y":454.284},{"x":148.16,"y":447.613},{"x":138.106,"y":439.88},{"x":128.889,"y":431.166},{"x":120.607,"y":421.56},{"x":113.343,"y":411.162},{"x":107.173,"y":400.079},{"x":102.163,"y":388.426},{"x":98.362,"y":376.325},{"x":95.811,"y":363.901},{"x":93.205,"y":346.84},{"x":85.387,"y":295.661},{"x":82.78,"y":278.602},{"x":81.528,"y":266.434},{"x":81.476,"y":254.203},{"x":82.622,"y":242.026},{"x":84.956,"y":230.019},{"x":88.456,"y":218.299},{"x":93.088,"y":206.978},{"x":98.806,"y":196.167},{"x":105.558,"y":185.967},{"x":115.778,"y":172.062},{"x":146.438,"y":130.344},{"x":156.659,"y":116.438},{"x":164.508,"y":106.804},{"x":173.276,"y":98.001},{"x":182.877,"y":90.114},{"x":193.217,"y":83.222},{"x":204.191,"y":77.395},{"x":215.69,"y":72.688},{"x":227.601,"y":69.15},{"x":244.369,"y":65.067},{"x":294.672,"y":52.818},{"x":311.439,"y":48.735},{"x":324.259,"y":46.316},{"x":337.261,"y":45.25},{"x":350.305,"y":45.549},{"x":363.244,"y":47.211},{"x":375.939,"y":50.214},{"x":392.502,"y":55.062},{"x":442.191,"y":69.605},{"x":458.753,"y":74.453},{"x":469.046,"y":77.96},{"x":478.994,"y":82.35},{"x":488.52,"y":87.591}],"halfWidth":25.25}`]
]);

console.log('car_racer rotation V2 suite (FIT-V4 TASK A)');
console.log('-------------------------------------------');

// -----------------------------------------------------------------------------
await test('parseGeneratorVersion: allowlist (undefined/null/1/"1"->1, 2/"2"->2)', () => {
  for (const [raw, want] of [[undefined, 1], [null, 1], ['1', 1], [1, 1], ['2', 2], [2, 2]]) {
    assert(Rotation.parseGeneratorVersion(raw) === want, 'parseGeneratorVersion(' + String(raw) + ') === ' + want);
  }
});
await test('parseGeneratorVersion: every non-allowlisted value throws typed ValidationError', () => {
  const bad = ['', ' ', '\t', '\n', ' 1', '1 ', '2 ', '3', 0, 3, -1, 1.5, NaN, Infinity, true, false, {}, [], 'abc', '0x2'];
  for (const raw of bad) {
    assertValidationError(() => Rotation.parseGeneratorVersion(raw), 'parseGeneratorVersion rejects ' + JSON.stringify(raw));
  }
});
await test('requireGeneratorVersion: strict boundary rejects undefined/null/empty', () => {
  for (const raw of [undefined, null, '']) {
    assertValidationError(() => Rotation.requireGeneratorVersion(raw), 'requireGeneratorVersion rejects ' + String(raw));
  }
});
await test('requireGeneratorVersion: valid values delegate; whitespace still refused', () => {
  assert(Rotation.requireGeneratorVersion('1') === 1, 'requireGeneratorVersion("1") === 1');
  assert(Rotation.requireGeneratorVersion(2) === 2, 'requireGeneratorVersion(2) === 2');
  assertValidationError(() => Rotation.requireGeneratorVersion('  '), 'requireGeneratorVersion("  ") refused');
});
await test('parsers exported on Rotation, globalThis and CommonJS (same function identity)', () => {
  assert(typeof rotationMod.parseGeneratorVersion === 'function', 'CommonJS exports parseGeneratorVersion');
  assert(typeof rotationMod.requireGeneratorVersion === 'function', 'CommonJS exports requireGeneratorVersion');
  assert(rotationMod.parseGeneratorVersion === Rotation.parseGeneratorVersion, 'CommonJS === Rotation.parseGeneratorVersion');
  assert(rotationMod.requireGeneratorVersion === Rotation.requireGeneratorVersion, 'CommonJS === Rotation.requireGeneratorVersion');
  assert(globalThis.parseGeneratorVersion === Rotation.parseGeneratorVersion, 'globalThis === Rotation.parseGeneratorVersion');
  assert(globalThis.requireGeneratorVersion === Rotation.requireGeneratorVersion, 'globalThis === Rotation.requireGeneratorVersion');
});

// -----------------------------------------------------------------------------
await test('V1 constants/ladders literally unchanged', () => {
  assert(Rotation.TRACK_POINTS === 14, 'TRACK_POINTS still 14');
  assert(Rotation.generateTrack(0, 1).points.length === 14, 'V1 generator still emits 14 points (THETAS unchanged)');
  assert(Rotation.TRACK_MAX_RADIUS === 245 && Rotation.TRACK_FALLBACK_RADIUS === 170 && Rotation.TRACK_MAX_REJECTS === 200, 'V1 geometry constants unchanged');
  assert(Rotation.harmonicsForDifficulty(100) === 6, 'V1 HARMONICS(100) still 6');
  assert(Rotation.amplitudeForDifficulty(100) === 0.42, 'V1 AMPLITUDE(100) still 0.42');
  assert(Rotation.halfWidthForDifficulty(0) === 50 && Rotation.halfWidthForDifficulty(20) === 45 &&
         Rotation.halfWidthForDifficulty(50) === 37.5 && Rotation.halfWidthForDifficulty(100) === 25,
    'V1 HALFWIDTH ladder is the FIT-V6 50 - 0.25 d ladder (global break accepted)');
});
await test('V1 golden bytes regenerated under the FIT-V6 ladder: 20-vector hash corpus at version 1 AND legacy 2-arg', () => {
  for (const [s, r, want] of GOLDEN_V1) {
    assert(Rotation.resolveGeneratedTrack(s, r, 1).trackHash === want, 'V1 hash seed=' + s + ' r=' + r + ' (explicit version 1)');
    assert(Rotation.resolveGeneratedTrack(s, r).trackHash === want, 'V1 hash seed=' + s + ' r=' + r + ' (legacy 2-arg default)');
  }
});
await test('V1 full canonical bytes byte-identical for two full vectors', () => {
  for (const [key, want] of V1_BYTES) {
    const [s, r] = key.split('|').map(Number);
    const got = canonicalJSON({ points: Rotation.resolveGeneratedTrack(s, r, 1).controlPoints, halfWidth: Rotation.resolveGeneratedTrack(s, r, 1).halfWidth });
    assert(got === want, 'V1 canonical bytes seed=' + s + ' r=' + r + '\n got: ' + got + '\nwant: ' + want);
  }
});

// -----------------------------------------------------------------------------
await test('V2 golden vectors: 20-vector hash corpus at version 2', () => {
  for (const [s, r, want] of GOLDEN_V2) {
    const got = Rotation.resolveGeneratedTrack(s, r, 2);
    assert(got.trackHash === want, 'V2 hash seed=' + s + ' r=' + r + ' - got ' + got.trackHash + ' want ' + want);
  }
});
await test('V2 full canonical bytes byte-identical for two full vectors', () => {
  for (const [key, want] of V2_BYTES) {
    const [s, r] = key.split('|').map(Number);
    const res = Rotation.resolveGeneratedTrack(s, r, 2);
    const got = canonicalJSON({ points: res.controlPoints, halfWidth: res.halfWidth });
    assert(got === want, 'V2 canonical bytes seed=' + s + ' r=' + r + '\n got: ' + got + '\nwant: ' + want);
  }
});
await test('V2 resolver: arc-built 91-96 point F1 templates, halfWidth/difficulty ladders, hash = cyrb53(canonicalJSON(bytes))', () => {
  for (const [s, r] of [[0, 1], [1, 50], [12345, 99], [0xFFFFFFFF, 2]]) {
    const res = Rotation.resolveGeneratedTrack(s, r, 2);
    assert(res.controlPoints.length >= 91 && res.controlPoints.length <= 96,
      'V2 seed=' + s + ' r=' + r + ' - arc-built F1 template has 91-96 points, got ' + res.controlPoints.length);
    assert(res.halfWidth === halfWidthOf(r), 'V2 halfWidth ladder seed=' + s + ' r=' + r);
    assert(res.difficulty === Math.min(r, 100), 'V2 difficulty ladder seed=' + s + ' r=' + r);
    assert(res.trackHash === hashOf(res.controlPoints, res.halfWidth), 'V2 hash = cyrb53(canonicalJSON(bytes)) seed=' + s + ' r=' + r);
    assert(/^[0-9a-f]{16}$/.test(res.trackHash), 'V2 hash is 16 lowercase hex seed=' + s + ' r=' + r);
  }
});
await test('resolveGeneratedTrack version validation: integer 1|2 only', () => {
  const s = 42, r = 3;
  assertValidationError(() => Rotation.resolveGeneratedTrack(s, r, 0), 'version 0 refused');
  assertValidationError(() => Rotation.resolveGeneratedTrack(s, r, 3), 'version 3 refused');
  assertValidationError(() => Rotation.resolveGeneratedTrack(s, r, 1.5), 'version 1.5 refused');
  assertValidationError(() => Rotation.resolveGeneratedTrack(s, r, NaN), 'version NaN refused');
  assertValidationError(() => Rotation.resolveGeneratedTrack(s, r, '2'), 'string version refused (canonical integer only)');
});

// -----------------------------------------------------------------------------
await test('determinism: two V2 runs byte-identical (resolver + generateTrackV2 + THETAS36)', () => {
  assert(Array.isArray(Rotation.THETAS36) && Rotation.THETAS36.length === 36, 'THETAS36 has 36 entries');
  for (const [s, r] of [[0, 1], [12345, 50], [0xDEADBEEF>>>0, 99]]) {
    const a = Rotation.resolveGeneratedTrack(s, r, 2), b = Rotation.resolveGeneratedTrack(s, r, 2);
    assert(JSON.stringify(a) === JSON.stringify(b), 'resolveGeneratedTrack byte-identical seed=' + s + ' r=' + r);
    const g1 = Rotation.generateTrackV2(s, r), g2 = Rotation.generateTrackV2(s, r);
    assert(JSON.stringify(g1) === JSON.stringify(g2), 'generateTrackV2 byte-identical seed=' + s + ' r=' + r);
  }
});
await test('V2 geometry: arc-built 91-96 point templates, finite in-bounds coords, simple control polygon', () => {
  for (const [s, r] of GOLDEN_V2.map(([s, r]) => [s, r])) {
    const pts = Rotation.generateTrackV2(s, r).points;
    assert(pts.length >= 91 && pts.length <= 96, '91-96 points seed=' + s + ' r=' + r + ', got ' + pts.length);
    for (const p of pts) {
      assert(Number.isFinite(p.x) && Number.isFinite(p.y), 'finite coord seed=' + s + ' r=' + r + ': ' + JSON.stringify(p));
      assert(Math.abs(p.x) <= 1e6 && Math.abs(p.y) <= 1e6, 'coord within bounds seed=' + s + ' r=' + r);
    }
    assert(!polygonSelfIntersects(pts), 'no self-intersection seed=' + s + ' r=' + r);
    assert(!isFallback(pts), 'golden vector is not the fallback seed=' + s + ' r=' + r);
  }
});
await test('V2 fallback: regular 36-gon radius 170, finite, non-self-intersecting', () => {
  const fb = Rotation.generateTrackV2Fallback();
  assert(Array.isArray(fb) && fb.length === 36, 'fallback has 36 points');
  let maxErr = 0;
  for (const p of fb) {
    assert(Number.isFinite(p.x) && Number.isFinite(p.y), 'fallback point finite');
    const d = Math.hypot(p.x - CENTER.x, p.y - CENTER.y);
    if (Math.abs(d - 170) > maxErr) maxErr = Math.abs(d - 170);
  }
  assert(maxErr <= 1e-3, 'fallback radius 170 within rounding, maxErr ' + maxErr);
  assert(!polygonSelfIntersects(fb), 'fallback polygon non-self-intersecting');
  assert(Rotation.TRACK_POINTS_V2 === 36, 'TRACK_POINTS_V2 === 36');
});

// -----------------------------------------------------------------------------
// Turn-metric corpora, regenerated under the FIT-V6 sharp-template V2 generator
// (the old radial-formula baselines and the min-rho reference were specific to
// the removed generator and no longer apply). The sharp F1 templates have far
// MORE direction changes than the smooth V1 generator; the required >=2x ratio
// and 0-fallback properties still hold. Baselines below are measured from the
// shipped contract and pinned as regression gates.
// -----------------------------------------------------------------------------
await test('turn metric, golden-20 corpus: V1 0.15 / V2 9.55, ratio >= 2x', () => {
  let tc1 = 0, tc2 = 0;
  for (const [s, r] of SEEDS.flatMap((s) => RS.map((r) => [s, r]))) {
    tc1 += turnCount(Rotation.resolveGeneratedTrack(s, r, 1).controlPoints);
    tc2 += turnCount(Rotation.resolveGeneratedTrack(s, r, 2).controlPoints);
  }
  const mean1 = tc1 / 20, mean2 = tc2 / 20;
  assert(Math.abs(mean1 - 0.15) < 1e-9, 'golden-20 V1 mean turn === 0.15, got ' + mean1);
  assert(Math.abs(mean2 - 9.55) < 1e-9, 'golden-20 V2 mean turn === 9.55 (arc-built 91-96pt F1 templates), got ' + mean2);
  assert(mean2 >= 2 * mean1, 'golden-20 V2 >= 2x V1 turn count (V1 ' + mean1 + ', V2 ' + mean2 + ')');
});
await test('turn metric, frozen 400-seed/2000-sample sweep: V1 0.1505 / V2 11.6, ratio >= 2x, 0 fallbacks', () => {
  const srng = mulberry32(0xC0FFEE);
  const sweepSeeds = Array.from({ length: 400 }, () => srng() >>> 0);
  const sweepRs = [1, 25, 50, 75, 99];
  let sum1 = 0, sum2 = 0, fallbacks = 0, n = 0;
  for (const s of sweepSeeds) {
    for (const r of sweepRs) {
      const p1 = Rotation.resolveGeneratedTrack(s, r, 1).controlPoints;
      const p2 = Rotation.resolveGeneratedTrack(s, r, 2).controlPoints;
      sum1 += turnCount(p1);
      sum2 += turnCount(p2);
      if (isFallback(p2)) fallbacks += 1;
      n += 1;
    }
  }
  const mean1 = sum1 / n, mean2 = sum2 / n;
  assert(n === 2000, 'sweep sample count 2000, got ' + n);
  assert(Math.abs(mean1 - 0.1505) < 1e-9, 'sweep V1 mean turn === 0.1505, got ' + mean1);
  assert(Math.abs(mean2 - 11.6) < 1e-9, 'sweep V2 mean turn === 11.6 (arc-built 91-96pt F1 templates), got ' + mean2);
  assert(mean2 >= 2 * mean1, 'sweep V2 >= 2x V1 turn count (V1 ' + mean1 + ', V2 ' + mean2 + ')');
  assert(fallbacks === 0, 'sweep produced 0 fallbacks, got ' + fallbacks);
});

// -----------------------------------------------------------------------------
console.log('-------------------------------------------');
console.log('rotation-v2: ' + passed + ' passed, ' + failures.length + ' failed');
if (failures.length) process.exit(1);
