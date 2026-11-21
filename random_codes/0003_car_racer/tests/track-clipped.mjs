// tests/track-clipped.mjs — FIT-V6 TASK G1: byte-frozen clipped wall model.
//
// Plain Node, guarded CommonJS exports, no npm deps — same shape as
// tests/smoke.mjs. Run with:  node tests/track-clipped.mjs
//
// Legs (all must be green):
//   1  constructor invariant: left/right/segments OWN lengths, finite, >0 segs,
//      9-decimal rounding, halfWidth ladder = 50 - 0.25 d.
//   2  zero self-crossings on both clipped walls for all 6 templates ×
//      d ∈ {0,25,50,75,100}.
//   3  byte-for-byte wall golden corpus (30 vectors) — pinned from the
//      red-team-approved, byte-frozen G1 algorithm.
//   4  two-construction byte equality (deterministic rebuild).
//   5  10,000 rays: _raycastGrid === _raycastBrute on a clipped hairpin.
//   6  draw() does not throw when wall length != samples length (iterates wall
//      lengths, never samples.length).
//   7  adversarial .cartrack fuzz: termination, finite, no NaN, bounded work.
//   8b frozen miter formula + 8-point osculating-arc fallback (test seam).
//   8  sample budget enforced on ACTUAL count; TrackWallError export + type.

import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);

const {
  Track,
  cyrb53,
  MAX_TOTAL_SAMPLES,
  TrackWallError,
  ValidationError,
} = require('../js/track.js');

// -----------------------------------------------------------------------------
// Tiny harness (mirrors tests/smoke.mjs): counts legs, never throws out of a leg.
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

// -----------------------------------------------------------------------------
// Self-contained fixtures. The six frozen sharp-spike hairpin templates are
// copied verbatim from rotation.js V5_TEMPLATES (byte-frozen data) so this suite
// is independent of the generator task (G2) that owns rotation.js.
// -----------------------------------------------------------------------------
const TEMPLATES = [
  [[735.681,462.089],[781.5,381.5],[827.8,302.3],[876,220.3],[777,206.4],[683.4,188.2],[589.2,180.7],[497.5,160.2],[400.7,146.3],[307.9,134.2],[216.2,119.5],[121.5,107.6],[281.7,105.1],[435.9,109.4],[597,107.1],[726.9,107.7],[835.7,108.9],[911.6,243.8],[978.5,372.8],[833.6,461],[688.6,544.9]],
  [[684.906,181.99],[759.3,259.4],[836.2,331.6],[843.4,442.1],[840.9,539.3],[767.8,537.3],[707,537.6],[582.1,557.4],[529.3,491.2],[496.2,445.8],[459.5,369.3],[416.6,331],[384.5,272.3],[338.9,206.3],[307.6,149.4],[256.6,98],[350.1,109.1],[416.2,112.7],[485.3,99.3],[552.5,102.2],[599,92.6]],
  [[622.818,358.304],[639,289.9],[557.5,243.5],[479.7,193.7],[399.5,144.1],[319.4,97.6],[420.4,94],[521.2,96.3],[622.2,97.9],[702.3,96.6],[778.4,95.2],[777.5,155.8],[780.6,221.4],[779.1,283.8],[775.7,348.7],[729,400.9],[671.3,452.1],[623.7,505.7],[568.8,556],[587.3,492.7],[607.3,423.9]],
  [[184.973,180.789],[296.8,201],[411.2,210.1],[531.5,207.2],[652.3,186.3],[743.6,228.9],[876.8,302.9],[991.9,140.3],[952.2,492.5],[900.6,509.7],[828.1,494],[706.3,487.6],[624.9,356.4],[478.2,334.1],[355.6,356.7],[257.3,395.3],[152.7,425],[128.2,379.8],[148.1,289.1],[109.1,209.7],[108.1,161.6],[120.3,169.1]],
  [[240.524,270.508],[308,234.9],[416.4,192.5],[533.5,103.8],[581.2,130],[589.7,104.3],[720.8,190.9],[823.7,193.6],[956.4,277.6],[971.1,385.1],[923.5,424.6],[885.3,334.3],[825.1,376.2],[734.6,519.4],[679.9,546.2],[555,494.9],[479.2,379.7],[315.2,367.5],[228.9,333.4],[171.5,322.9],[128.9,287.2],[172.7,306.3]],
  [[749.274,500.728],[669.3,566.5],[540.3,521.6],[456.8,487.1],[373.6,461.3],[273.2,317.1],[253.1,403.2],[197.1,290.6],[139,207],[81,111.3],[248,83.5],[410.3,96.3],[528.2,175.1],[597.3,268.5],[676.5,332.1],[762.1,336],[847.2,316.9],[933.4,324.3],[1019,341.3],[929.9,340.4],[843.3,423.4]],
];
const DIFFICULTIES = [0, 25, 50, 75, 100];
// Byte-frozen narrow ladder: halfWidth(d) = 50 - 0.25 d  (50 .. 25).
function halfWidthFor(d) {
  return Number((50 - 0.25 * d).toFixed(9));
}
function templatePoints(t) {
  return t.map((p) => ({ x: p[0], y: p[1] }));
}

// Proper crossing test (non-adjacent closed segments), FOLD_EPS on the cross.
const FOLD_EPS = 1e-9;
function orient(ax, ay, bx, by, cx, cy) {
  return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}
function properCross(a, b, c, d) {
  const o1 = orient(a.x, a.y, b.x, b.y, c.x, c.y);
  const o2 = orient(a.x, a.y, b.x, b.y, d.x, d.y);
  const o3 = orient(c.x, c.y, d.x, d.y, a.x, a.y);
  const o4 = orient(c.x, c.y, d.x, d.y, b.x, b.y);
  return ((o1 > FOLD_EPS && o2 < -FOLD_EPS) || (o1 < -FOLD_EPS && o2 > FOLD_EPS)) &&
         ((o3 > FOLD_EPS && o4 < -FOLD_EPS) || (o3 < -FOLD_EPS && o4 > FOLD_EPS));
}
function crossingCount(poly) {
  const n = poly.length;
  let c = 0;
  for (let i = 0; i < n; i++) {
    const a = poly[i], b = poly[(i + 1) % n];
    for (let j = i + 2; j < n; j++) {
      if (i === 0 && j === n - 1) continue;
      if (properCross(a, b, poly[j], poly[(j + 1) % n])) c++;
    }
  }
  return c;
}

// Byte-frozen wall serialization: every coord rounded to 9 decimals, fixed
// key order, then hashed with the production cyrb53. Covers leftWall,
// rightWall AND segments (plan §18).
function wallHash(track) {
  const encPt = (p) => Number(p.x.toFixed(9)).toFixed(9) + ',' + Number(p.y.toFixed(9)).toFixed(9);
  const encWall = (w) => w.map(encPt).join(';');
  const encSeg = (s) =>
    Number(s.x1.toFixed(9)).toFixed(9) + ',' + Number(s.y1.toFixed(9)).toFixed(9) + ',' +
    Number(s.x2.toFixed(9)).toFixed(9) + ',' + Number(s.y2.toFixed(9)).toFixed(9) + ',' + s.side;
  const bytes = encWall(track.leftWall) + '|' + encWall(track.rightWall) + '|' +
    track.segments.map(encSeg).join(';');
  return cyrb53(bytes);
}

// -----------------------------------------------------------------------------
// Golden corpus — pinned from the byte-frozen G1 algorithm. Key = `${t}:${d}`.
// Filled once from a verified implementation; any drift is a breaking change.
// -----------------------------------------------------------------------------
const GOLDEN = {
  '0:0': '0e6a97540a6dd38e',
  '0:25': 'de42fa98c2c64918',
  '0:50': '1a685dc268e02a0b',
  '0:75': '43605d99713edcfd',
  '0:100': 'a078e56245542179',
  '1:0': 'f015f7de5e35e200',
  '1:25': '2c6d73e76d2be2b9',
  '1:50': '01556caea2886cc6',
  '1:75': '88d939b238d0eb11',
  '1:100': '7ca3dc5ca08dc4d7',
  '2:0': '97751c04a53432f0',
  '2:25': '9833011691c19d9d',
  '2:50': 'bd6415c5e558411f',
  '2:75': 'b3a748583b425eb5',
  '2:100': '40351f4a084640cb',
  '3:0': '09c0d063d1692d64',
  '3:25': 'b68dba1ca0357f3e',
  '3:50': '79285343e785c04c',
  '3:75': '00e4ab0eec88ae4b',
  '3:100': 'a52bd0b380198e55',
  '4:0': '9a1c63581e799f9f',
  '4:25': '73cda34b8d2138ab',
  '4:50': '179f868acef77115',
  '4:75': '5a892fdbeef83499',
  '4:100': 'be92bb929ed2dc1e',
  '5:0': 'c87754b400b247e8',
  '5:25': 'b226d790ec65585e',
  '5:50': 'af0daa150a391e0c',
  '5:75': '69982f7a68cd8a3d',
  '5:100': '87301e3ccfd23d21',
};

console.log('car_racer clipped-wall suite (FIT-V6 G1)');
console.log('---------------------------------------');

// -----------------------------------------------------------------------------
await leg('leg 1 — invariant: own lengths, finite, positive segments, 9-dec rounding', () => {
  assert(typeof TrackWallError === 'function', 'TrackWallError is exported');
  const err = new TrackWallError('x');
  assert(err.name === 'TrackWallError', 'TrackWallError.name is TrackWallError');
  assert(err instanceof Error, 'TrackWallError extends Error');

  for (let t = 0; t < TEMPLATES.length; t++) {
    for (const d of DIFFICULTIES) {
      const hw = halfWidthFor(d);
      const track = new Track(templatePoints(TEMPLATES[t]), hw);
      const L = track.leftWall, R = track.rightWall, S = track.segments;
      assert(L.length === R.length, `t${t} d${d}: leftWall.length === rightWall.length (${L.length} vs ${R.length})`);
      assert(S.length === 2 * L.length, `t${t} d${d}: segments.length === 2*wall.length (${S.length} vs ${2 * L.length})`);
      assert(track._segStamp.length === S.length, `t${t} d${d}: _segStamp.length === segments.length`);
      assert(L.length >= 3, `t${t} d${d}: wall has >= 3 points`);
      assert(track.samples.length <= MAX_TOTAL_SAMPLES, `t${t} d${d}: samples within budget`);
      const bound = 1e6 + hw;
      for (const w of [L, R]) {
        for (const p of w) {
          assert(Number.isFinite(p.x) && Number.isFinite(p.y), `t${t} d${d}: wall coord finite`);
          assert(Math.abs(p.x) <= bound && Math.abs(p.y) <= bound, `t${t} d${d}: wall coord in bounds`);
          assert(p.x === Number(p.x.toFixed(9)), `t${t} d${d}: x is 9-decimal rounded (${p.x})`);
          assert(p.y === Number(p.y.toFixed(9)), `t${t} d${d}: y is 9-decimal rounded (${p.y})`);
        }
      }
      for (let i = 0; i < S.length; i++) {
        const s = S[i];
        const dx = s.x2 - s.x1, dy = s.y2 - s.y1;
        assert(Math.hypot(dx, dy) > 0, `t${t} d${d}: segment ${i} has positive length`);
        assert(s.side === 'left' || s.side === 'right', `t${t} d${d}: segment side valid`);
      }
    }
  }
  assert(halfWidthFor(0) === 50 && halfWidthFor(25) === 43.75 &&
         halfWidthFor(50) === 37.5 && halfWidthFor(75) === 31.25 && halfWidthFor(100) === 25,
         'halfWidth ladder is 50 - 0.25 d -> 50/43.75/37.5/31.25/25');
});

// -----------------------------------------------------------------------------
await leg('leg 2 — zero self-crossings on both clipped walls (6 templates × 5 d)', () => {
  for (let t = 0; t < TEMPLATES.length; t++) {
    for (const d of DIFFICULTIES) {
      const track = new Track(templatePoints(TEMPLATES[t]), halfWidthFor(d));
      const lc = crossingCount(track.leftWall);
      const rc = crossingCount(track.rightWall);
      assert(lc === 0, `t${t} d${d}: leftWall has ${lc} crossings`);
      assert(rc === 0, `t${t} d${d}: rightWall has ${rc} crossings`);
    }
  }
});

// -----------------------------------------------------------------------------
await leg('leg 3 — byte-for-byte wall golden corpus (30 vectors)', () => {
  assert(Object.keys(GOLDEN).length === 30, 'golden corpus has 30 pinned vectors');
  let checked = 0;
  for (let t = 0; t < TEMPLATES.length; t++) {
    for (const d of DIFFICULTIES) {
      const key = t + ':' + d;
      const track = new Track(templatePoints(TEMPLATES[t]), halfWidthFor(d));
      const got = wallHash(track);
      assert(Object.prototype.hasOwnProperty.call(GOLDEN, key), 'golden key ' + key + ' exists');
      assert(got === GOLDEN[key], `t${t} d${d}: wall hash ${got} === golden ${GOLDEN[key]}`);
      checked++;
    }
  }
  assert(checked === 30, 'checked 30 golden vectors');
});

// -----------------------------------------------------------------------------
await leg('leg 4 — two-construction byte equality (deterministic rebuild)', () => {
  for (let t = 0; t < TEMPLATES.length; t++) {
    for (const d of DIFFICULTIES) {
      const pts = templatePoints(TEMPLATES[t]);
      const hw = halfWidthFor(d);
      const a = new Track(pts, hw);
      const b = new Track(pts, hw);
      assert(wallHash(a) === wallHash(b), `t${t} d${d}: two constructions byte-identical`);
      assert(JSON.stringify(a.segments) === JSON.stringify(b.segments), `t${t} d${d}: segments byte-identical`);
    }
  }
});

// -----------------------------------------------------------------------------
await leg('leg 5 — 10,000 rays: _raycastGrid === _raycastBrute on clipped hairpin', () => {
  // Template 4 at d=50 (hw 37.5) is a tight double-hairpin: several folds are
  // clipped on both sides. Rays originate on the centerline and off-road.
  const track = new Track(templatePoints(TEMPLATES[4]), halfWidthFor(50));
  let state = 0x9e3779b9;
  const rnd = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let z = state;
    z = Math.imul(z ^ (z >>> 15), z | 1);
    z ^= z + Math.imul(z ^ (z >>> 7), z | 61);
    return ((z ^ (z >>> 14)) >>> 0) / 4294967296;
  };
  const origins = track.samples.length;
  const MAX_DIST = 400;
  let mismatches = 0;
  for (let i = 0; i < 10000; i++) {
    const s = track.samples[Math.floor(rnd() * origins)];
    const offRoad = i % 5 === 0;
    const jitter = offRoad ? 800 : 60;
    const ox = s.x + (rnd() - 0.5) * jitter;
    const oy = s.y + (rnd() - 0.5) * jitter;
    const angle = rnd() * Math.PI * 2;
    const grid = track._raycastGrid(ox, oy, angle, MAX_DIST);
    const brute = track._raycastBrute(ox, oy, angle, MAX_DIST);
    if (grid !== brute) mismatches++;
    assert(Number.isFinite(grid) && grid >= 0 && grid <= MAX_DIST, 'ray distance finite and in range');
  }
  assert(mismatches === 0, 'grid === brute for all 10,000 rays (mismatches ' + mismatches + ')');
});

// -----------------------------------------------------------------------------
await leg('leg 6 — draw() no-throw with wall/sample length mismatch', () => {
  const track = new Track(templatePoints(TEMPLATES[0]), halfWidthFor(0));
  // Simulate clipped walls SHORTER than the centerline sample list.
  track.leftWall = track.leftWall.slice(0, 7);
  track.rightWall = track.rightWall.slice(0, 7);
  const ctx = {
    save() {}, restore() {}, beginPath() {}, closePath() {}, fill() {},
    setLineDash() {}, stroke() {},
    moveTo(x, y) { assert(Number.isFinite(x) && Number.isFinite(y), 'moveTo finite'); },
    lineTo(x, y) { assert(Number.isFinite(x) && Number.isFinite(y), 'lineTo finite (no undefined wall coord)'); },
  };
  track.draw(ctx, {}); // must not throw
  assert(track.samples.length > 7, 'sanity: samples longer than walls in this leg');
});

// -----------------------------------------------------------------------------
await leg('leg 7 — adversarial .cartrack fuzz: termination/finite/no-NaN/bounded', () => {
  let state = 0x1234567;
  const rnd = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let z = state;
    z = Math.imul(z ^ (z >>> 15), z | 1);
    z ^= z + Math.imul(z ^ (z >>> 7), z | 61);
    return ((z ^ (z >>> 14)) >>> 0) / 4294967296;
  };
  const chaos = [1e6, -1e6, 0, 1, -1, 1e-6, 999.999, 1000, -1000, 0.0000001, 700.5];
  const CASES = 240;
  const started = Date.now();
  let built = 0, refused = 0;
  for (let c = 0; c < CASES; c++) {
    const npts = 3 + Math.floor(rnd() * 22);
    const points = [];
    for (let i = 0; i < npts; i++) {
      const wild = rnd() < 0.25;
      const x = wild ? chaos[Math.floor(rnd() * chaos.length)] : (rnd() - 0.5) * 1400;
      const y = wild ? chaos[Math.floor(rnd() * chaos.length)] : (rnd() - 0.5) * 900;
      points.push({ x, y });
    }
    const hw = rnd() < 0.3 ? Math.floor(rnd() * 5) + 5 : Math.floor(rnd() * 90) + 25;
    const payload = JSON.stringify({ points, halfWidth: hw }); // a .cartrack file
    let track = null;
    try {
      track = Track.fromJSON(payload);
      built++;
    } catch (err) {
      // Refusals are fine (ValidationError / TrackWallError) — but must be
      // typed and must not be a generic TypeError from NaN/undefined math.
      assert(err && typeof err.name === 'string', 'fuzz refusal is a typed error');
      assert(err.name === 'ValidationError' || err.name === 'TrackWallError' ||
             err.name === 'RangeError', 'fuzz refusal type: ' + (err && err.name));
      refused++;
      continue;
    }
    // No NaN / non-finite anywhere in the built geometry.
    for (const s of track.samples) {
      assert(Number.isFinite(s.x) && Number.isFinite(s.y) && Number.isFinite(s.tangent) && Number.isFinite(s.arcPos), 'sample finite');
    }
    for (const w of [track.leftWall, track.rightWall]) {
      for (const p of w) assert(Number.isFinite(p.x) && Number.isFinite(p.y), 'wall coord finite');
      assert(w.length === track.leftWall.length, 'walls have equal length');
    }
    for (const s of track.segments) {
      assert([s.x1, s.y1, s.x2, s.y2].every(Number.isFinite), 'segment finite');
    }
    assert(track.segments.length === 2 * track.leftWall.length, 'segments length matches walls');
    assert(track.length > 0 && Number.isFinite(track.length), 'track length finite positive');
  }
  const elapsed = Date.now() - started;
  assert(built + refused === CASES, 'all fuzz cases either built or typed-refused');
  assert(elapsed < 12000, 'bounded work: ' + CASES + ' cases in ' + elapsed + ' ms');
  assert(built > 0, 'fuzz built at least some tracks (' + built + ')');
});

// -----------------------------------------------------------------------------
await leg('leg 8b — frozen miter formula + 8-point osculating-arc fallback', () => {
  assert(typeof Track._miterJoin === 'function', '_miterJoin test seam exists');
  const samples = [
    { x: 100, y: 100, tangent: 0 },
    { x: 100, y: 100, tangent: Math.PI },
  ];
  // Non-parallel: pinned operand order -> den=10, t=-0.5 -> (-5, 0).
  const miter = Track._miterJoin(
    { x: 0, y: 0 }, { x: 10, y: 0 },
    { x: 5, y: 1 }, { x: 15, y: 2 },
    samples, 0, 1, 25, 1
  );
  assert(miter.length === 1, 'non-parallel miter emits exactly 1 point');
  assert(miter[0].x === -5 && miter[0].y === 0, 'miter formula/operand order exact: (-5, 0), got ' + JSON.stringify(miter[0]));

  // Parallel (den = 0 < PARALLEL_EPS) -> osculating arc of exactly 8 points,
  // angular step turn/8, start angle atan2(normal.y, normal.x).
  const arc = Track._miterJoin(
    { x: 0, y: 0 }, { x: 10, y: 0 },
    { x: 5, y: 5 }, { x: 15, y: 5 },
    samples, 0, 1, 25, 1
  );
  assert(arc.length === 8, 'arc fallback emits exactly 8 points, got ' + arc.length);
  const start = Math.PI / 2; // normal at sample 0 (tangent 0) = (0, 1)
  const turn = Math.PI;      // tangent(pi) - tangent(0), wrapped
  for (let k = 0; k < 8; k++) {
    const ang = start + k * turn / 8;
    assert(arc[k].x === Number((100 + 25 * Math.cos(ang)).toFixed(9)), 'arc point ' + k + ' x exact');
    assert(arc[k].y === Number((100 + 25 * Math.sin(ang)).toFixed(9)), 'arc point ' + k + ' y exact');
  }
  assert(JSON.stringify(arc) === JSON.stringify(Track._miterJoin(
    { x: 0, y: 0 }, { x: 10, y: 0 }, { x: 5, y: 5 }, { x: 15, y: 5 }, samples, 0, 1, 25, 1
  )), 'arc fallback is deterministic');
});

// -----------------------------------------------------------------------------
await leg('leg 8c — arc fallback uses ORIGINAL centreline sample index (stale-pts regression)', () => {
  // Regression for FIT-V6 final gate Blocking 2. clipOffsetSide re-indexes `pts`
  // after every fold merge, so passing the (shrunken) pts index into miterJoin
  // makes the osculating-arc fallback read the WRONG centreline samples on the
  // 2nd+ merge — a wrong-but-finite wall that passes v5WallsSound and is
  // silently persisted. The fix carries `si` (the original sample index) on
  // every wall point and uses pts[i0].si / pts[j0].si at the join site.
  //
  // A natural 2-merge arc case cannot be built from the byte-frozen corpus
  // (the arc path has ZERO corpus hits: PARALLEL_EPS=1e-9 is unreachable once
  // coordinates are round9'd). So we exercise the exact seam with explicit si
  // values that model a post-merge pts whose indices != original sample indices.

  // ---- (A) seam: arc reads samples[siA], samples[siB], samples[mid] ----------
  const samples = new Array(12);
  for (let k = 0; k < 12; k++) samples[k] = { x: 1000 + k * 37, y: 2000 + k * 53, tangent: k * 0.13 };
  // The small (stale pts-style) indices carry deliberately WRONG geometry.
  samples[0] = { x: 111, y: 111, tangent: 2.9 };
  samples[1] = { x: 222, y: 222, tangent: 1.7 };
  // The real (post-merge) original sample indices 8 / 11 and their midpoint 9.
  // The midpoint has a DISTINCT position so the arc coordinates prove the
  // centre came from samples[floor((siA+siB)/2)] == samples[9], not from
  // samples[siA]/samples[siB] or a stale small index.
  samples[8]  = { x: 100, y: 100, tangent: 0 };
  samples[9]  = { x: 200, y: 300, tangent: Math.PI / 2 }; // floor((8+11)/2) = 9
  samples[11] = { x: 100, y: 100, tangent: Math.PI };

  const siA = 8, siB = 11, hw = 25, sign = 1;
  const arc = Track._miterJoin(
    { x: 0, y: 0 }, { x: 10, y: 0 },
    { x: 5, y: 5 }, { x: 15, y: 5 },
    samples, siA, siB, hw, sign
  );
  assert(arc.length === 8, 'arc has 8 points');
  const start = Math.PI / 2; // normal at samples[8] (tangent 0)
  const turn = Math.PI;      // samples[11].tangent - samples[8].tangent
  const cx = 200, cy = 300;  // samples[9] — the midpoint centre
  for (let k = 0; k < 8; k++) {
    const ang = start + k * turn / 8;
    assert(arc[k].x === Number((cx + hw * Math.cos(ang)).toFixed(9)), 'arc ' + k + ' x from samples[mid] centre');
    assert(arc[k].y === Number((cy + hw * Math.sin(ang)).toFixed(9)), 'arc ' + k + ' y from samples[mid] centre');
  }
  // Arc points carry the documented interpolated original sample index.
  for (let k = 0; k < 8; k++) {
    assert(arc[k].si === siA + Math.round((siB - siA) * k / 7), 'arc ' + k + ' si interpolated');
  }
  // A stale pts index (0/1) would produce a different arc — prove the seam is
  // actually driven by the si arguments, not by some fixed small index.
  const stale = Track._miterJoin(
    { x: 0, y: 0 }, { x: 10, y: 0 },
    { x: 5, y: 5 }, { x: 15, y: 5 },
    samples, 0, 1, hw, sign
  );
  assert(JSON.stringify(stale) !== JSON.stringify(arc), 'si values drive arc geometry (stale indices differ)');

  // Non-parallel miter point inherits siA (documented rule).
  const miter = Track._miterJoin(
    { x: 0, y: 0 }, { x: 10, y: 0 },
    { x: 5, y: 1 }, { x: 15, y: 2 },
    samples, siA, siB, hw, sign
  );
  assert(miter.length === 1 && miter[0].si === siA, 'miter point carries siA');

  // ---- (B) end-to-end: every wall point carries a valid original si ---------
  // pts[0] is never removed by a fold merge (adjacent pairs are skipped), so
  // it stays the raw offset of samples[0] with si === 0 on BOTH sides.
  for (let t = 0; t < TEMPLATES.length; t++) {
    for (const d of DIFFICULTIES) {
      const hwT = halfWidthFor(d);
      const track = new Track(templatePoints(TEMPLATES[t]), hwT);
      for (const w of [track.leftWall, track.rightWall]) {
        assert(w[0].si === 0, `t${t} d${d}: first wall point si === 0`);
        for (const p of w) {
          assert(Number.isInteger(p.si), `t${t} d${d}: wall point si is an integer`);
          assert(p.si >= 0 && p.si < track.samples.length, `t${t} d${d}: wall si in sample range`);
        }
      }
      // The first left wall point really is samples[0] offset by +normal.
      const s0 = track.samples[0];
      const nx = -Math.sin(s0.tangent), ny = Math.cos(s0.tangent);
      assert(track.leftWall[0].x === Number((s0.x + nx * hwT).toFixed(9)) &&
             track.leftWall[0].y === Number((s0.y + ny * hwT).toFixed(9)),
             `t${t} d${d}: first left wall point is raw offset of samples[0]`);
    }
  }
});

// -----------------------------------------------------------------------------
await leg('leg 8 — sample budget enforced on ACTUAL count; TrackWallError typed', () => {
  assert(MAX_TOTAL_SAMPLES === 20000, 'MAX_TOTAL_SAMPLES is 20000');
  // A 3-point bomb at the coordinate limit is refused before/after sampling.
  const bomb = [{ x: -1e6, y: -1e6 }, { x: 1e6, y: -1e6 }, { x: 0, y: 1e6 }];
  let refused = false;
  try {
    new Track(bomb, 50);
  } catch (err) {
    refused = true;
    assert(err.name === 'ValidationError' || err.name === 'TrackWallError', 'bomb refusal typed');
  }
  assert(refused, 'hostile large track is refused (sample budget)');
  assert(new TrackWallError('m') instanceof ValidationError, 'TrackWallError instanceof ValidationError');
});

console.log('');
console.log('legs passed: ' + legsPassed + ' / ' + (legsPassed + failures.length));
if (failures.length) {
  console.log('FAILURES: ' + failures.length);
  process.exitCode = 1;
} else {
  console.log('ALL GREEN');
}
