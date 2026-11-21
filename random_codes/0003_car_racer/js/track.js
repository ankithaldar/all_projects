// js/track.js — Closed-circuit track for the 0003_car_racer game.
//
// The track is a CLOSED, UNIFORM Catmull–Rom spline through a list of
// control points. We evaluate it at densely sampled t-steps such that the
// arc-length spacing between consecutive samples is <= ~4 px (spec: <= 5 px),
// and every sample carries { x, y, tangent (radians), arcPos (cumulative arc
// length from arcPos 0) }.
//
// From the samples we build:
//   * left/right wall polylines (samples offset along the unit normal by
//     ±halfWidth), and
//   * a flat `segments` array of wall line segments
//     { x1, y1, x2, y2, side: 'left' | 'right' },
//   * a uniform spatial hash grid over the segments (cell = 64 px) used by
//     the DDA raycaster. `_raycastGrid` and `_raycastBrute` compute the SAME
//     thing; the grid only visits cells the ray actually crosses. Both share
//     one segment-intersection helper so results are bit-identical —
//     tests/smoke.mjs leg 4 asserts exact equality on 10 000 seeded rays.
//
// Plain classic script; browser globals on globalThis, guarded CommonJS
// export at the bottom for node tests.

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

// TrackWallError — FIT-V6 G1. Thrown for ANY violation of the byte-frozen
// clipped-wall contract: fold-merge non-convergence, non-finite/out-of-bounds
// coordinate, collapsed wall, broken length invariant or grid AABB span cap.
// Extends ValidationError so all existing typed-error plumbing keeps working.
if (typeof globalThis.TrackWallError !== 'function') {
  globalThis.TrackWallError = class TrackWallError extends ValidationError {
    constructor(message) {
      super(message);
      this.name = 'TrackWallError';
    }
  };
}
var TrackWallError = globalThis.TrackWallError;

var TRACK_DEFAULT_HALF_WIDTH = 70;
var GRID_CELL_SIZE = 64;          // spatial hash cell size (px)
var SAMPLE_TARGET_SPACING = 4;    // px — comfortably inside the <= 5 px spec
// Hard cap on the TOTAL number of samples a track may allocate (F-2). The
// default circuit is ~500 samples; 20 000 (~80 km of spline at 4 px spacing)
// is enough for any sane track — 40x headroom — and bounds the heap (segments
// scale 2x samples, the grid inserts one AABB per segment) and the
// O(samples) nearestCenterline cost at construction time. Enforced in the
// constructor BEFORE any sample array is allocated, which closes the import /
// localStorage-boot / replay / editor arms at once.
var MAX_TOTAL_SAMPLES = 20000;    // samples — see doc comment above
var MIN_POINTS = 3;
var MAX_POINTS = 512;
var MAX_ABS_COORD = 1e6;
var MIN_HALF_WIDTH = 1;
var MAX_HALF_WIDTH = 2000;
var MAX_TRACK_JSON_LENGTH = 262144; // 256 KiB — SQLite column cap for track_json

// ---- FIT-V6 G1 — byte-frozen clipped-wall constants (red-team approved) ----
var FOLD_EPS = 1e-9;        // fold-loop proper-crossing threshold (on the cross)
var PARALLEL_EPS = 1e-9;    // miter-denominator parallel test -> arc fallback
var OSC_ARC_POINTS = 8;     // osculating-arc fallback point count
var MAX_CELL_SPAN = 1024;   // per-segment AABB cell-span cap (bounded work)

// VALIDATION — the shared, frozen validation-bound set. js/editor.js reads
// THESE (no literal twins anywhere); nothing else re-declares them.
var VALIDATION = Object.freeze({
  MIN_POINTS: MIN_POINTS,
  MAX_POINTS: MAX_POINTS,
  MAX_ABS_COORD: MAX_ABS_COORD,
  MIN_HALF_WIDTH: MIN_HALF_WIDTH,
  MAX_HALF_WIDTH: MAX_HALF_WIDTH,
  MAX_TRACK_JSON_LENGTH: MAX_TRACK_JSON_LENGTH,
});

// ---------------------------------------------------------------------------
// DEFAULT_CONTROL_POINTS — a sensible closed circuit for a ~1000x700 canvas.
// Rounded, clockwise (canvas y-down ⇒ positive angle is clockwise), fills
// most of the canvas with margins for the ±halfWidth walls. The start/finish
// line is arcPos 0, i.e. the first control point.
// ---------------------------------------------------------------------------
// Frozen (array + each point object): later files must not rewrite the
// default circuit. Constructor call sites still deep-copy points into the
// track instance, so freezing here only affects this shared constant.
var DEFAULT_CONTROL_POINTS = Object.freeze([
  Object.freeze({ x: 170, y: 180 }),
  Object.freeze({ x: 300, y: 120 }),
  Object.freeze({ x: 500, y: 100 }),
  Object.freeze({ x: 700, y: 120 }),
  Object.freeze({ x: 830, y: 190 }),
  Object.freeze({ x: 880, y: 340 }),
  Object.freeze({ x: 830, y: 490 }),
  Object.freeze({ x: 650, y: 570 }),
  Object.freeze({ x: 450, y: 590 }),
  Object.freeze({ x: 250, y: 560 }),
  Object.freeze({ x: 120, y: 400 }),
]);

// ---------------------------------------------------------------------------
// Uniform Catmull–Rom spline (closed loop).
//
// Standard formulas: for segment i running p1 → p2 with neighbours p0, p3,
//   q(t) = 0.5 * ( 2*p1
//                  + (-p0 + p2) * t
//                  + (2*p0 - 5*p1 + 4*p2 - p3) * t^2
//                  + (-p0 + 3*p1 - 3*p2 + p3) * t^3 ),   t ∈ [0,1]
//   q'(t) = 0.5 * ( (-p0 + p2) + 2*(2*p0 - 5*p1 + 4*p2 - p3)*t
//                                        + 3*(-p0 + 3*p1 - 3*p2 + p3)*t^2 )
// Uniform (not centripetal) parameterisation is deliberate: control points
// are roughly evenly spaced by design, and uniform keeps the tangent field
// smooth and cheap (no per-segment knot arithmetic on the hot path).
// ---------------------------------------------------------------------------
function catmullRomPoint(p0, p1, p2, p3, t) {
  var t2 = t * t;
  var t3 = t2 * t;
  return {
    x: 0.5 * (2 * p1.x + (-p0.x + p2.x) * t + (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 + (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3),
    y: 0.5 * (2 * p1.y + (-p0.y + p2.y) * t + (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 + (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3),
  };
}

function catmullRomTangent(p0, p1, p2, p3, t) {
  var t2 = t * t;
  var dx = 0.5 * ((-p0.x + p2.x) + 2 * (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t + 3 * (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t2);
  var dy = 0.5 * ((-p0.y + p2.y) + 2 * (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t + 3 * (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t2);
  return Math.atan2(dy, dx);
}

// ---------------------------------------------------------------------------
// Input validation shared by the constructor and fromJSON().
// Points must be an array of plain {x,y} objects with finite coordinates
// |x|,|y| <= 1e6; halfWidth finite and within [1, 2000].
// NOTE on halfWidth coercion: this helper coerces via Number() as a ONE-SHOT
// convenience for constructor call sites (e.g. numeric strings are accepted
// there). The STRICT boundary is Track.fromJSON, which pre-checks
// `typeof halfWidth === 'number'` BEFORE calling this helper — a non-number
// halfWidth never reaches this coercion on that path.
// ---------------------------------------------------------------------------
function validateControlPoints(points, halfWidth) {
  if (!Array.isArray(points)) {
    throw new ValidationError('control points must be an array');
  }
  if (points.length < MIN_POINTS || points.length > MAX_POINTS) {
    throw new ValidationError('control points must contain between ' + MIN_POINTS + ' and ' + MAX_POINTS + ' points, got ' + points.length);
  }
  var clean = new Array(points.length);
  for (var i = 0; i < points.length; i++) {
    var p = points[i];
    if (p === null || typeof p !== 'object' || Array.isArray(p)) {
      throw new ValidationError('point ' + i + ' must be an object {x, y}');
    }
    var x = p.x, y = p.y;
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      throw new ValidationError('point ' + i + ' has non-finite coordinates');
    }
    if (Math.abs(x) > MAX_ABS_COORD || Math.abs(y) > MAX_ABS_COORD) {
      throw new ValidationError('point ' + i + ' exceeds |coordinate| <= ' + MAX_ABS_COORD);
    }
    clean[i] = { x: x, y: y }; // deep copy — never keep caller references
  }
  var hw = Number(halfWidth);
  if (!Number.isFinite(hw) || hw < MIN_HALF_WIDTH || hw > MAX_HALF_WIDTH) {
    throw new ValidationError('halfWidth must be finite and within [' + MIN_HALF_WIDTH + ', ' + MAX_HALF_WIDTH + '], got ' + String(halfWidth));
  }
  return { points: clean, halfWidth: hw };
}

// ---------------------------------------------------------------------------
// FIT-V6 G1 — byte-frozen offset-wall fold clipping.
//
// The raw left/right offset polylines can self-cross at tight hairpins. Each
// side is reduced to ONE simple closed polyline by repeatedly merging the
// innermost fold loop into the miter intersection of the two segments that
// form it (or an 8-point osculating arc when those segments are parallel).
// Every emitted coordinate is rounded with Number(v.toFixed(9)) so the bytes
// are identical across engines/constructions. Array indices only; iteration
// is by increasing sample index; nested loops are merged innermost-first via
// an explicit LIFO stack (smallest span, tie -> later scan entry).
// ---------------------------------------------------------------------------
function round9(v) {
  return Number(v.toFixed(9));
}

// Proper crossing of two closed segments, with FOLD_EPS on the cross product
// (collinear/touching contacts are not folds).
function foldCross(a, b, c, d) {
  var o1 = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
  var o2 = (b.x - a.x) * (d.y - a.y) - (b.y - a.y) * (d.x - a.x);
  var o3 = (d.x - c.x) * (a.y - c.y) - (d.y - c.y) * (a.x - c.x);
  var o4 = (d.x - c.x) * (b.y - c.y) - (d.y - c.y) * (b.x - c.x);
  return ((o1 > FOLD_EPS && o2 < -FOLD_EPS) || (o1 < -FOLD_EPS && o2 > FOLD_EPS)) &&
         ((o3 > FOLD_EPS && o4 < -FOLD_EPS) || (o3 < -FOLD_EPS && o4 > FOLD_EPS));
}

// Miter intersection — FIXED operand order (frozen):
//   den = (p1x-p0x)*(q1y-q0y) - (p1y-p0y)*(q1x-q0x)
//   t   = ((q0x-p0x)*(q1y-q0y) - (q0y-p0y)*(q1x-q0x)) / den
//   point = p0 + t*(p1-p0)
// If |den| < PARALLEL_EPS -> osculating arc of OSC_ARC_POINTS (8) points,
// angular step turn/8, start angle atan2(normal.y, normal.x) with a
// sample-index tie-break.
//
// FIT-V6 final gate Blocking 2: `siA`/`siB` are the ORIGINAL centreline sample
// indices of p0/q0 (carried on each point as `.si`), NEVER indices into the
// `pts` array — `pts` shrinks after every fold merge, so its indices go stale
// on the 2nd+ merge and would select the wrong centreline samples below. The
// caller passes pts[i0].si / pts[j0].si. Every returned point carries an `si`
// so a later merge keeps the provenance chain intact. Deterministic rules:
//   * single miter point  -> siA (the fold's leading sample).
//   * arc points          -> si interpolated from siA..siB across the 8 points
//                            (siA + round((siB-siA)*k/(OSC_ARC_POINTS-1))).
function miterJoin(p0, p1, q0, q1, samples, siA, siB, hw, sign) {
  var p0x = p0.x, p0y = p0.y, p1x = p1.x, p1y = p1.y;
  var q0x = q0.x, q0y = q0.y, q1x = q1.x, q1y = q1.y;
  var den = (p1x - p0x) * (q1y - q0y) - (p1y - p0y) * (q1x - q0x);
  if (Math.abs(den) >= PARALLEL_EPS) {
    var t = ((q0x - p0x) * (q1y - q0y) - (q0y - p0y) * (q1x - q0x)) / den;
    return [{ x: round9(p0x + t * (p1x - p0x)), y: round9(p0y + t * (p1y - p0y)), si: siA }];
  }
  // Osculating-arc fallback (parallel miter).
  var sa = samples[siA], sb = samples[siB];
  var nax = -Math.sin(sa.tangent), nay = Math.cos(sa.tangent);
  var nbx = -Math.sin(sb.tangent), nby = Math.cos(sb.tangent);
  var aa = Math.atan2(nay, nax), ba = Math.atan2(nby, nbx);
  // sample-index tie-break: equal start angles -> the lower sample index wins.
  var start = (Math.abs(aa - ba) < FOLD_EPS) ? (siA <= siB ? aa : ba) : aa;
  var turn = sb.tangent - sa.tangent;
  while (turn > Math.PI) turn -= Math.PI * 2;
  while (turn <= -Math.PI) turn += Math.PI * 2;
  if (Math.abs(turn) < FOLD_EPS) turn = Math.PI; // degenerate parallel -> half turn
  var centre = samples[Math.floor((siA + siB) / 2)];
  var out = new Array(OSC_ARC_POINTS);
  for (var k = 0; k < OSC_ARC_POINTS; k++) {
    var ang = start + k * turn / OSC_ARC_POINTS;
    out[k] = {
      x: round9(centre.x + sign * hw * Math.cos(ang)),
      y: round9(centre.y + sign * hw * Math.sin(ang)),
      si: siA + Math.round((siB - siA) * k / (OSC_ARC_POINTS - 1)),
    };
  }
  return out;
}

// Find all fold loops with a spatial-hash sweep (never O(n²)). This is a
// transient pre-clip structure; the raycast grid below is the string-keyed Map.
// Returns an explicit stack of loops in increasing-index scan order.
function findFoldLoops(pts) {
  var n = pts.length;
  var cell = GRID_CELL_SIZE;
  var grid = new Map();
  var boxes = new Array(n);
  // Numeric key for this TRANSIENT fold sweep only (cx,cy are bounded well
  // inside ±1e5 by MAX_ABS_COORD/64 + MAX_CELL_SPAN). The raycast grid below
  // remains string-keyed; this is NOT v5CellKey and is rebuilt per merge.
  // Hoisted once and reused by both the insert sweep and the query sweep so
  // the two can never drift apart.
  var cellKey = function (x, y) { return (x + 100000) * 200000 + (y + 100000); };
  var i, cx, cy;
  for (i = 0; i < n; i++) {
    var a = pts[i], b = pts[(i + 1) % n];
    var minX = Math.floor(Math.min(a.x, b.x) / cell), maxX = Math.floor(Math.max(a.x, b.x) / cell);
    var minY = Math.floor(Math.min(a.y, b.y) / cell), maxY = Math.floor(Math.max(a.y, b.y) / cell);
    if ((maxX - minX + 1) > MAX_CELL_SPAN || (maxY - minY + 1) > MAX_CELL_SPAN) {
      throw new TrackWallError('offset segment AABB cell span exceeds cap');
    }
    boxes[i] = [minX, maxX, minY, maxY];
    for (cx = minX; cx <= maxX; cx++) {
      for (cy = minY; cy <= maxY; cy++) {
        var key = cellKey(cx, cy);
        var bucket = grid.get(key);
        if (!bucket) { bucket = []; grid.set(key, bucket); }
        bucket.push(i);
      }
    }
  }
  var stack = [];
  var seen = new Int32Array(n);
  for (i = 0; i < n; i++) {
    var a2 = pts[i], b2 = pts[(i + 1) % n];
    var box = boxes[i], stamp = i + 1;
    for (cx = box[0]; cx <= box[1]; cx++) {
      for (cy = box[2]; cy <= box[3]; cy++) {
        var bucket2 = grid.get(cellKey(cx, cy));
        if (!bucket2) continue;
        for (var bi = 0; bi < bucket2.length; bi++) {
          var j = bucket2[bi];
          if (seen[j] === stamp) continue;
          seen[j] = stamp;
          if (j <= i) continue; // each unordered pair once, j > i
          if (j === i + 1 || (i === 0 && j === n - 1)) continue; // adjacent
          var c = pts[j], d = pts[(j + 1) % n];
          if (foldCross(a2, b2, c, d)) stack.push({ i: i, j: j, span: j - i });
        }
      }
    }
  }
  return stack;
}

// Cheap O(n) necessary condition for a fold: an offset self-crossing always
// has >=1 segment whose displacement opposes the centreline tangent. When no
// such segment exists the polyline is already simple and the spatial sweep can
// be skipped entirely (keeps fold-free tracks well inside the time budget).
function offsetMayFold(raw, samples) {
  for (var i = 0; i < raw.length; i++) {
    var j = (i + 1) % raw.length;
    var tx = Math.cos(samples[i].tangent), ty = Math.sin(samples[i].tangent);
    if ((raw[j].x - raw[i].x) * tx + (raw[j].y - raw[i].y) * ty < -FOLD_EPS) return true;
  }
  return false;
}

// Reduce one raw offset polyline to a simple closed polyline.
function clipOffsetSide(raw, samples, hw, sign) {
  var pts = new Array(raw.length);
  // Every emitted coordinate is rounded with Number(v.toFixed(9)) — including
  // the raw offset points that survive clipping. `si` carries the ORIGINAL
  // centreline sample index (raw[i] came from samples[i]); it travels with the
  // point through every merge because `pts` is re-indexed after each one.
  for (var i = 0; i < raw.length; i++) {
    pts[i] = { x: round9(raw[i].x), y: round9(raw[i].y), si: i };
  }
  var hasFolds = offsetMayFold(raw, samples);
  var maxIters = 8 * raw.length + 64; // bounded work: each merge removes >=1 vertex
  var guard = 0;
  while (hasFolds && guard++ < maxIters) {
    var n = pts.length;
    // Explicit stack from the increasing-index spatial-hash sweep.
    // Pop the innermost (smallest span; tie -> later scan entry) first.
    var stack = findFoldLoops(pts);
    if (stack.length === 0) { hasFolds = false; break; } // simple closed polyline
    var pick = 0;
    for (var s = 1; s < stack.length; s++) {
      if (stack[s].span < stack[pick].span ||
          (stack[s].span === stack[pick].span && stack[s].i > stack[pick].i)) pick = s;
    }
    var loop = stack[pick];
    var i0 = loop.i, j0 = loop.j;
    // Pass the ORIGINAL sample indices, not the (shrunken) pts indices i0/j0.
    // On the 2nd+ merge i0/j0 no longer equal the sample positions; feeding
    // them to miterJoin would make the arc fallback read wrong-but-finite wall
    // geometry that passes validation. See FIT-V6 final gate Blocking 2.
    var join = miterJoin(pts[i0], pts[(i0 + 1) % n], pts[j0], pts[(j0 + 1) % n],
                         samples, pts[i0].si, pts[j0].si, hw, sign);
    // Emit in increasing sample index, then stack (merge) order.
    var next = pts.slice(0, i0 + 1).concat(join, pts.slice(j0 + 1));
    var clean = [];
    for (var q = 0; q < next.length; q++) {
      var p = next[q];
      if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) {
        throw new TrackWallError('clipped wall produced a non-finite coordinate');
      }
      if (clean.length && clean[clean.length - 1].x === p.x && clean[clean.length - 1].y === p.y) continue;
      clean.push(p);
    }
    if (clean.length > 1 && clean[0].x === clean[clean.length - 1].x && clean[0].y === clean[clean.length - 1].y) clean.pop();
    if (clean.length < 3) throw new TrackWallError('clipped wall collapsed below 3 points');
    pts = clean;
  }
  if (hasFolds && guard >= maxIters) throw new TrackWallError('fold merge did not converge');
  var bound = MAX_ABS_COORD + hw;
  for (var z = 0; z < pts.length; z++) {
    if (Math.abs(pts[z].x) > bound || Math.abs(pts[z].y) > bound) {
      throw new TrackWallError('clipped wall coordinate exceeds |MAX_ABS_COORD| + halfWidth');
    }
  }
  for (var z2 = 0; z2 < pts.length; z2++) {
    var pa = pts[z2], pb = pts[(z2 + 1) % pts.length];
    if (pa.x === pb.x && pa.y === pb.y) throw new TrackWallError('clipped wall has a zero-length segment');
  }
  return pts;
}

// Canonical-length alignment. The two independently clipped sides can lose a
// different number of fold vertices. The G1 invariant requires
// leftWall.length === rightWall.length === segments.length/2, so the shorter
// wall is brought up to the longer by inserting exact midpoints into its
// longest segment (lowest index on tie). This preserves the polyline's shape
// exactly (collinear split), keeps every segment length > 0, and is fully
// deterministic — unlike dropping real clipped geometry. An inserted midpoint
// inherits si = floor((si_a + si_b)/2) — the same lower-midpoint rule miterJoin
// uses for its arc centre — so the provenance chain stays integer and bounded.
function alignWallLengths(a, b) {
  while (a.length < b.length) splitLongestSegment(a);
  while (b.length < a.length) splitLongestSegment(b);
}

function splitLongestSegment(wall) {
  var best = 0, bestLen2 = -1;
  for (var i = 0; i < wall.length; i++) {
    var p = wall[i], q = wall[(i + 1) % wall.length];
    var d2 = (q.x - p.x) * (q.x - p.x) + (q.y - p.y) * (q.y - p.y);
    if (d2 > bestLen2) { bestLen2 = d2; best = i; }
  }
  var pa = wall[best], pb = wall[(best + 1) % wall.length];
  var mid = {
    x: round9((pa.x + pb.x) / 2),
    y: round9((pa.y + pb.y) / 2),
    si: Math.floor((pa.si + pb.si) / 2),
  };
  if ((mid.x === pa.x && mid.y === pa.y) || (mid.x === pb.x && mid.y === pb.y)) {
    throw new TrackWallError('cannot align wall lengths: degenerate segment');
  }
  wall.splice(best + 1, 0, mid);
}

// ---------------------------------------------------------------------------
// Ray ↔ segment intersection.
//
// Ray:   P = O + t * d,  d a UNIT direction vector ⇒ t is the distance in px.
// Seg:   A + s * (B − A), s ∈ [0,1].
// Solves O + t·d = A + s·(B − A) via 2D cross products:
//   t = cross(A − O, B − A) / cross(d, B − A)
//   s = cross(A − O, d)     / cross(d, B − A)
// Parallel/collinear rays (denominator exactly 0) report no hit — collinear
// walls are measure-zero for this use and treating them as misses keeps
// behaviour identical between brute force and the grid. Hits a hair BEHIND
// the origin (t ∈ [−1e−9, 0)) are accepted and clamped to 0 ("hits slightly
// ahead are included"); intersections with t > maxDist or s ∉ [0,1] are
// rejected. Near-parallel-but-nonzero denominators are left to the maxDist
// check rather than an epsilon band. Returns hit distance in px, or Infinity.
// ---------------------------------------------------------------------------
function raySegmentDistance(ox, oy, dx, dy, x1, y1, x2, y2, maxDist) {
  var sx = x2 - x1, sy = y2 - y1;
  var denom = dx * sy - dy * sx;
  if (denom === 0) return Infinity; // parallel (or zero-length segment)
  var qx = x1 - ox, qy = y1 - oy;
  var t = (qx * sy - qy * sx) / denom;
  var s = (qx * dy - qy * dx) / denom;
  if (t < -1e-9 || t > maxDist) return Infinity;
  if (s < 0 || s > 1) return Infinity;
  return t < 0 ? 0 : t;
}

// Squared distance from point P to segment AB (used by crash detection in car.js).
function segmentPointDistanceSquared(x1, y1, x2, y2, px, py) {
  var vx = x2 - x1, vy = y2 - y1;
  var wx = px - x1, wy = py - y1;
  var c1 = vx * wx + vy * wy;
  if (c1 <= 0) return wx * wx + wy * wy; // before A
  var c2 = vx * vx + vy * vy;
  if (c2 <= c1) { // beyond B
    var bx = px - x2, by = py - y2;
    return bx * bx + by * by;
  }
  var b = c1 / c2; // projection factor of P onto AB
  var pbx = x1 + b * vx - px, pby = y1 + b * vy - py;
  return pbx * pbx + pby * pby;
}

// ---------------------------------------------------------------------------
// cyrb53 — a 64-bit string hash (public-domain pattern by bryc).
// Produces two well-mixed 32-bit halves (h1 and h2); the return value is
// those two halves concatenated as one fixed 16-char lowercase hex string —
// 32 bits from each half, NOT a single 53-bit value. Seeded deterministically
// (seed defaults to 0). NON-CRYPTOGRAPHIC: it is an integrity/lineage SIGNAL
// for track bytes (same-track resume vs different-track fork), never an
// adversarial boundary.
// ---------------------------------------------------------------------------
function cyrb53(str, seed) {
  var h1 = 0xdeadbeef ^ (seed || 0);
  var h2 = 0x41c6ce57 ^ (seed || 0);
  for (var i = 0; i < str.length; i++) {
    var ch = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507);
  h1 ^= Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507);
  h2 ^= Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  // Two 32-bit halves, hex-padded to a stable 8 + 8 = 16-char string.
  return (h2 >>> 0).toString(16).padStart(8, '0') + (h1 >>> 0).toString(16).padStart(8, '0');
}

// canonicalJSON(trackLike) — THE canonical serialization for hashing.
// Accepts either a Track instance (uses .controlPoints / .halfWidth) or a
// plain parsed object { points, halfWidth } (e.g. from JSON.parse of a
// track_json column). Fixed key order { points[{x,y}], halfWidth } — EVERY
// coordinate is rounded via Number(v.toFixed(9)) — points AND halfWidth —
// so that JSON round-trips of the same geometry produce the identical
// string. Computed on the PARSED object, never on an arbitrary string.
function canonicalJSON(trackLike) {
  var rawPoints = null;
  var halfWidth;
  if (trackLike && Array.isArray(trackLike.points)) {
    // Plain parsed object shape.
    rawPoints = trackLike.points;
    halfWidth = trackLike.halfWidth;
  } else if (trackLike && Array.isArray(trackLike.controlPoints)) {
    // Track instance shape.
    rawPoints = trackLike.controlPoints;
    halfWidth = trackLike.halfWidth;
  } else {
    throw new ValidationError('canonicalJSON expects { points, halfWidth } or a Track instance');
  }
  var points = new Array(rawPoints.length);
  for (var i = 0; i < rawPoints.length; i++) {
    points[i] = {
      x: Number(Number(rawPoints[i].x).toFixed(9)),
      y: Number(Number(rawPoints[i].y).toFixed(9)),
    };
  }
  // Key insertion order IS the JSON serialization order — keep it fixed forever.
  var roundedHalfWidth = Number(Number(halfWidth).toFixed(9));
  return JSON.stringify({ points: points, halfWidth: roundedHalfWidth });
}

// ---------------------------------------------------------------------------
// Track — the built circuit.
//
// constructor(controlPoints /* [{x,y}…] */, halfWidth = 70)
//   → builds samples, walls, segments, the spatial grid, arc length.
// Exposed:
//   .samples       — [{ x, y, tangent (rad), arcPos (px) }], spacing ≤ ~4 px
//   .segments      — [{ x1, y1, x2, y2, side }] wall segments (both sides)
//   .length        — total arc length in px (closed loop)
//   .controlPoints — validated deep copy of the constructor input
//   .halfWidth
//   .leftWall / .rightWall — wall polylines (for draw() and analysis)
//
// nearestCenterline(x, y) → { arcPos, tangentAngle, dist }
//   O(samples) nearest-sample search. Nearest SAMPLE (not segment) — with
//   ≤ 4 px spacing the quantization error is invisible at this scale; a
//   segment refinement pass was deliberately skipped (documented, "nearest
//   sample is fine"). The nearest centerline point wins even off-corridor —
//   callers relying on arcPos for stuck-kill/progress get a steady,
//   small-step signal anywhere on the track, not only inside the road.
// ---------------------------------------------------------------------------
function Track(controlPoints, halfWidth) {
  var validated = validateControlPoints(
    controlPoints,
    halfWidth === undefined ? TRACK_DEFAULT_HALF_WIDTH : halfWidth
  );
  this.controlPoints = validated.points;
  this.halfWidth = validated.halfWidth;

  var pts = this.controlPoints;
  var n = pts.length;

  // ---- control-point perimeter / spline-length estimate --------------------
  // 16-subdivision polyline estimate per segment (a LOWER bound of the true
  // arc length). Pre-pass BEFORE any sample allocation: if the total estimate
  // would need more than MAX_TOTAL_SAMPLES steps, the track is refused HERE —
  // a hostile 3-point bomb at 1e6-coordinate scale never reaches the sampling
  // loop (F-2). The estimates are reused by the sampling loop (no double work).
  var estimates = new Array(n);
  var totalEstimate = 0;
  for (var pre = 0; pre < n; pre++) {
    var p0e = pts[(pre - 1 + n) % n], p1e = pts[pre], p2e = pts[(pre + 1) % n], p3e = pts[(pre + 2) % n];
    var acc = 0;
    var ae = catmullRomPoint(p0e, p1e, p2e, p3e, 0);
    for (var ke = 1; ke <= 16; ke++) {
      var be = catmullRomPoint(p0e, p1e, p2e, p3e, ke / 16);
      acc += Math.hypot(be.x - ae.x, be.y - ae.y);
      ae = be;
    }
    estimates[pre] = acc;
    totalEstimate += acc;
  }
  var estimatedSamples = Math.ceil(totalEstimate / SAMPLE_TARGET_SPACING);
  if (estimatedSamples > MAX_TOTAL_SAMPLES) {
    throw new ValidationError('track exceeds the sample budget (' + estimatedSamples + ' samples) — simplify it');
  }

  // ---- dense sampling of the closed spline ---------------------------------
  var samples = [];
  var arc = 0;
  var previous = null;
  for (var i = 0; i < n; i++) {
    var p0 = pts[(i - 1 + n) % n], p1 = pts[i], p2 = pts[(i + 1) % n], p3 = pts[(i + 2) % n];

    var estimate = estimates[i];
    var steps = Math.max(1, Math.ceil(estimate / SAMPLE_TARGET_SPACING));

    // Sample t = k/steps for k in [0, steps); each segment's endpoint sample
    // belongs to the next iteration, keeping the sample list duplicate-free.
    for (var s = 0; s < steps; s++) {
      var t = s / steps;
      var c = catmullRomPoint(p0, p1, p2, p3, t);
      if (previous) {
        var gap = Math.hypot(c.x - previous.x, c.y - previous.y);
        if (gap < 1e-6) continue; // degenerate — skip duplicate to keep arcPos advancing
        arc += gap;
      }
      samples.push({ x: c.x, y: c.y, tangent: catmullRomTangent(p0, p1, p2, p3, t), arcPos: arc });
      previous = samples[samples.length - 1];
    }
  }
  // FIT-V6 G1: enforce the cap on the ACTUAL sample count, not just the
  // lower-bound estimate above. The estimate is a lower bound of true arc
  // length, so a track can pass the pre-check and still allocate > budget.
  if (samples.length > MAX_TOTAL_SAMPLES) {
    throw new ValidationError('track exceeds the sample budget (' + samples.length + ' actual samples) — simplify it');
  }
  // Closing gap: last sample → first sample completes the loop (the spline is
  // closed, so the distance is exactly the wrap-around arc length).
  var first = samples[0];
  var lastSample = samples[samples.length - 1];
  this.length = arc + (first && lastSample ? Math.hypot(first.x - lastSample.x, first.y - lastSample.y) : 0);
  this.samples = samples;

  // ---- walls: raw ±normal offsets, then byte-frozen fold clipping ---------
  var hw = this.halfWidth;
  var rawLeft = new Array(samples.length);
  var rawRight = new Array(samples.length);
  for (var i2 = 0; i2 < samples.length; i2++) {
    var sm = samples[i2];
    var nx = -Math.sin(sm.tangent), ny = Math.cos(sm.tangent);
    rawLeft[i2] = { x: sm.x + nx * hw, y: sm.y + ny * hw };
    rawRight[i2] = { x: sm.x - nx * hw, y: sm.y - ny * hw };
  }
  var leftWall = clipOffsetSide(rawLeft, samples, hw, 1);
  var rightWall = clipOffsetSide(rawRight, samples, hw, -1);
  alignWallLengths(leftWall, rightWall);
  this.leftWall = leftWall;
  this.rightWall = rightWall;

  // ---- canonical segments REBUILT FROM the clipped walls (own lengths) ----
  var segments = [];
  var m = leftWall.length;
  for (var j2 = 0; j2 < m; j2++) {
    var j3 = (j2 + 1) % m;
    segments.push({ x1: leftWall[j2].x, y1: leftWall[j2].y, x2: leftWall[j3].x, y2: leftWall[j3].y, side: 'left' });
    segments.push({ x1: rightWall[j2].x, y1: rightWall[j2].y, x2: rightWall[j3].x, y2: rightWall[j3].y, side: 'right' });
  }
  this.segments = segments;

  // ---- pre-grid constructor invariant (FIT-V6 G1) -------------------------
  if (leftWall.length !== rightWall.length || segments.length !== 2 * leftWall.length) {
    throw new TrackWallError('wall/segment length invariant violated');
  }
  for (var inv = 0; inv < segments.length; inv++) {
    var iv = segments[inv];
    if (!Number.isFinite(iv.x1) || !Number.isFinite(iv.y1) || !Number.isFinite(iv.x2) || !Number.isFinite(iv.y2)) {
      throw new TrackWallError('non-finite wall segment coordinate');
    }
    if (Math.hypot(iv.x2 - iv.x1, iv.y2 - iv.y1) === 0) {
      throw new TrackWallError('zero-length wall segment after clipping');
    }
  }

  // ---- uniform spatial hash grid over wall segments (cell = 64 px) --------
  // Each segment is inserted into every cell its AABB overlaps. DDA traversal
  // visits only the cells the ray crosses; results are bit-identical to the
  // brute-force scan because both paths share raySegmentDistance() and take
  // the same minimum. `_segStamp` + `_rayStampId` dedupe segments that span
  // multiple visited cells WITHOUT clearing the structure between rays.
  //
  // The dedupe is a pure optimization: min-distance tracking is idempotent,
  // so even a stale stamp could only cause a redundant RE-TEST of an
  // already-counted segment and the result would be identical. A stamp id is
  // handed out per ray (monotonically increasing); ids of earlier rays are
  // never equal to the current ray's id, so no stamp can collide within one
  // ray. Plain Array of doubles (exact integers up to 2^53) — no wrap-around
  // bookkeeping needed.
  this._gridCell = GRID_CELL_SIZE;
  this._grid = new Map();
  this._segStamp = new Array(segments.length).fill(0);
  this._rayStampId = 0;
  for (var si = 0; si < segments.length; si++) {
    var seg = segments[si];
    var minX = Math.floor(Math.min(seg.x1, seg.x2) / GRID_CELL_SIZE);
    var maxX = Math.floor(Math.max(seg.x1, seg.x2) / GRID_CELL_SIZE);
    var minY = Math.floor(Math.min(seg.y1, seg.y2) / GRID_CELL_SIZE);
    var maxY = Math.floor(Math.max(seg.y1, seg.y2) / GRID_CELL_SIZE);
    if ((maxX - minX + 1) > MAX_CELL_SPAN || (maxY - minY + 1) > MAX_CELL_SPAN) {
      throw new TrackWallError('segment AABB cell span exceeds cap');
    }
    for (var cx = minX; cx <= maxX; cx++) {
      for (var cy = minY; cy <= maxY; cy++) {
        var key = cx + ':' + cy; // string-keyed Map — never v5CellKey
        var bucket = this._grid.get(key);
        if (!bucket) {
          bucket = [];
          this._grid.set(key, bucket);
        }
        bucket.push(si);
      }
    }
  }
}

// nearestCenterline — see class doc. O(number of samples).
Track.prototype.nearestCenterline = function nearestCenterline(x, y) {
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    throw new ValidationError('nearestCenterline(x, y) requires finite coordinates');
  }
  var bestDist2 = Infinity;
  var best = this.samples[0];
  for (var i = 0; i < this.samples.length; i++) {
    var s = this.samples[i];
    var dx = s.x - x, dy = s.y - y;
    var d2 = dx * dx + dy * dy;
    if (d2 < bestDist2) {
      bestDist2 = d2;
      best = s;
    }
  }
  return { arcPos: best.arcPos, tangentAngle: best.tangent, dist: Math.sqrt(bestDist2) };
};

// Shared argument validation for the two raycast paths.
Track.prototype._validateRaycast = function _validateRaycast(x, y, angle, maxDist) {
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(angle)) {
    throw new ValidationError('raycast requires finite x, y and angle');
  }
  var md = maxDist === undefined ? Number.POSITIVE_INFINITY : maxDist;
  if (!(md > 0) && md !== Number.POSITIVE_INFINITY) {
    throw new ValidationError('raycast maxDist must be > 0 or Infinity, got ' + String(maxDist));
  }
  return md;
};

// _raycastBrute — O(segments) scan, the reference implementation.
Track.prototype._raycastBrute = function _raycastBrute(x, y, angle, maxDist) {
  var md = this._validateRaycast(x, y, angle, maxDist); // _validateRaycast defaults undefined -> Infinity
  var dx = Math.cos(angle), dy = Math.sin(angle);
  var best = Infinity;
  var segments = this.segments;
  for (var i = 0; i < segments.length; i++) {
    var seg = segments[i];
    var d = raySegmentDistance(x, y, dx, dy, seg.x1, seg.y1, seg.x2, seg.y2, md);
    if (d < best) best = d;
  }
  return best === Infinity ? md : best;
};

// _raycastGrid — DDA traversal of the uniform grid. Bit-identical results to
// _raycastBrute: same intersection helper, same minimum, same candidates
// (every segment whose AABB overlaps a traversed cell was inserted into that
// cell, and DDA visits every cell the ray crosses within maxDist).
Track.prototype._raycastGrid = function _raycastGrid(x, y, angle, maxDist) {
  var md = this._validateRaycast(x, y, angle, maxDist);
  var dx = Math.cos(angle), dy = Math.sin(angle);
  var cell = this._gridCell;

  var cx = Math.floor(x / cell), cy = Math.floor(y / cell);
  var stepX = dx >= 0 ? 1 : -1, stepY = dy >= 0 ? 1 : -1;
  // tDelta: distance along the ray to cross one full cell on each axis.
  var tDeltaX = dx === 0 ? Infinity : Math.abs(cell / dx);
  var tDeltaY = dy === 0 ? Infinity : Math.abs(cell / dy);
  // tMax: distance along the ray to the FIRST cell boundary on each axis.
  var tMaxX = dx === 0 ? Infinity : (dx > 0 ? (cx + 1) * cell - x : x - cx * cell) / Math.abs(dx);
  var tMaxY = dy === 0 ? Infinity : (dy > 0 ? (cy + 1) * cell - y : y - cy * cell) / Math.abs(dy);

  var best = Infinity;
  var stampId = ++this._rayStampId; // one id per ray — no cross-ray collision within this loop
  var stamp = this._segStamp;
  var grid = this._grid;
  var segments = this.segments;

  // Safety cap: with finite maxDist the loop exits when the ray has crossed
  // maxDist px; with Infinity (or a huge track) we stop after 8192 cells
  // (8192 * 64 px = 512 000 px — far beyond any sane circuit).
  for (var guard = 0; guard < 8192; guard++) {
    var bucket = grid.get(cx + ':' + cy);
    if (bucket) {
      for (var i = 0; i < bucket.length; i++) {
        var idx = bucket[i];
        if (stamp[idx] === stampId) continue; // already tested via another cell in this ray
        stamp[idx] = stampId;
        var seg = segments[idx];
        var d = raySegmentDistance(x, y, dx, dy, seg.x1, seg.y1, seg.x2, seg.y2, md);
        if (d < best) best = d;
      }
    }
    var nextT;
    if (tMaxX < tMaxY) {
      nextT = tMaxX; cx += stepX; tMaxX += tDeltaX;
    } else if (tMaxY < tMaxX) {
      nextT = tMaxY; cy += stepY; tMaxY += tDeltaY;
    } else {
      // Corner crossing — advance both axes.
      nextT = tMaxX; cx += stepX; cy += stepY; tMaxX += tDeltaX; tMaxY += tDeltaY;
    }
    if (nextT > md) break; // everything beyond this boundary is out of range
  }
  return best === Infinity ? md : Math.min(best, md);
};

// raycast — public entry point, always the grid path. maxDist defaults to
// Infinity (unlimited scan); sensors always pass SENSOR_MAX.
Track.prototype.raycast = function raycast(x, y, angle, maxDist) {
  return this._raycastGrid(x, y, angle, maxDist);
};

// ---------------------------------------------------------------------------
// Serialization.
//
// toJSON() → STRING with FIXED key order { points:[{x,y}…], halfWidth } and
// RAW stored numbers (no rounding, no reformatting) — JSON.stringify emits
// the shortest round-trip representation, which keeps files small while
// guaranteeing parse(toJSON(t)) deep-equals t's control points exactly.
// canonicalJSON() is the separate, rounded form used for hashing.
//
// fromJSON(str) — strict validation:
//   * str must be a string of length <= 262144 (256 KiB)
//   * JSON.parse inside try/catch
//   * points: array, 3..512 entries, each {x,y} finite, |coord| <= 1e6
//   * halfWidth: finite, within [1, 2000]
//   * anything else → ValidationError
// ---------------------------------------------------------------------------
Track.prototype.toJSON = function toJSON() {
  var points = new Array(this.controlPoints.length);
  for (var i = 0; i < this.controlPoints.length; i++) {
    points[i] = { x: this.controlPoints[i].x, y: this.controlPoints[i].y };
  }
  return JSON.stringify({ points: points, halfWidth: this.halfWidth });
};

Track.fromJSON = function fromJSON(str) {
  if (typeof str !== 'string') {
    throw new ValidationError('Track.fromJSON expects a JSON string, got ' + typeof str);
  }
  if (str.length > MAX_TRACK_JSON_LENGTH) {
    throw new ValidationError('track JSON exceeds ' + MAX_TRACK_JSON_LENGTH + ' characters (got ' + str.length + ')');
  }
  var obj;
  try {
    obj = JSON.parse(str);
  } catch (err) {
    throw new ValidationError('track JSON is not valid JSON: ' + (err && err.message));
  }
  if (obj === null || typeof obj !== 'object' || Array.isArray(obj)) {
    throw new ValidationError('track JSON must be an object { points, halfWidth }');
  }
  // STRICT boundary: halfWidth must be an actual JSON number. No string
  // coercion here — '70' is rejected even though the constructor would
  // accept it for ergonomic reasons (documented in validateControlPoints).
  if (typeof obj.halfWidth !== 'number' || !Number.isFinite(obj.halfWidth)) {
    throw new ValidationError('halfWidth must be a finite number, got ' + JSON.stringify(obj.halfWidth));
  }
  if (obj.halfWidth < MIN_HALF_WIDTH || obj.halfWidth > MAX_HALF_WIDTH) {
    throw new ValidationError('halfWidth must be within [' + MIN_HALF_WIDTH + ', ' + MAX_HALF_WIDTH + '], got ' + obj.halfWidth);
  }
  var validated = validateControlPoints(obj.points, obj.halfWidth);
  return new Track(validated.points, validated.halfWidth);
};

// hash() — cyrb53 of the canonical serialization. Non-cryptographic lineage
// signal only (see cyrb53 doc comment).
Track.prototype.hash = function hash() {
  return cyrb53(canonicalJSON(this));
};

// startPose() — { x, y, heading } at arcPos 0 (the first sample).
Track.prototype.startPose = function startPose() {
  var s = this.samples[0];
  return { x: s.x, y: s.y, heading: s.tangent };
};

// ---------------------------------------------------------------------------
// draw(ctx, opts) — road, center dashes, red/white wall strips, start/finish.
// Background (grass etc.) is OUT of scope here — callers paint it first.
// Nothing in this method reads DOM; `ctx` is any object with the standard
// canvas 2D API surface.
// ---------------------------------------------------------------------------
Track.prototype.draw = function draw(ctx, opts) {
  if (!ctx || typeof ctx.beginPath !== 'function') {
    throw new ValidationError('Track.draw requires a canvas 2D-like context');
  }
  var o = opts || {};
  var asphalt = o.asphalt || '#3b3b40';
  var centerLine = o.centerLine || 'rgba(255,255,255,0.6)';
  var wallA = o.wallA || '#e9ecef';
  var wallB = o.wallB || '#e63946';
  var samples = this.samples;
  var leftWall = this.leftWall, rightWall = this.rightWall;
  var n = samples.length;          // centerline only (center dash + start line)
  var ln = leftWall.length, rn = rightWall.length; // walls have their OWN length
  var i;

  ctx.save();

  // ---- asphalt ribbon: left wall forward, right wall reversed -------------
  // FIT-V6 G1: iterate the WALL lengths, never samples.length — clipped walls
  // can be shorter than the centerline sample list.
  ctx.beginPath();
  ctx.moveTo(leftWall[0].x, leftWall[0].y);
  for (i = 1; i < ln; i++) ctx.lineTo(leftWall[i].x, leftWall[i].y);
  for (i = rn - 1; i >= 0; i--) ctx.lineTo(rightWall[i].x, rightWall[i].y);
  ctx.closePath();
  ctx.fillStyle = asphalt;
  ctx.fill();

  // ---- center dashed line --------------------------------------------------
  ctx.beginPath();
  ctx.moveTo(samples[0].x, samples[0].y);
  for (i = 1; i < n; i++) ctx.lineTo(samples[i].x, samples[i].y);
  ctx.closePath();
  ctx.setLineDash([22, 16]);
  ctx.lineWidth = 3;
  ctx.strokeStyle = centerLine;
  ctx.stroke();
  ctx.setLineDash([]);

  // ---- walls: alternating red / white strips (two offset dash passes) ------
  var walls = [this.leftWall, this.rightWall];
  for (var w = 0; w < walls.length; w++) {
    var wall = walls[w];
    ctx.beginPath();
    ctx.moveTo(wall[0].x, wall[0].y);
    for (i = 1; i < wall.length; i++) ctx.lineTo(wall[i].x, wall[i].y);
    ctx.closePath();
    ctx.lineWidth = 8;
    ctx.setLineDash([26, 26]);
    ctx.strokeStyle = wallA;
    ctx.lineDashOffset = 0;
    ctx.stroke();
    ctx.strokeStyle = wallB;
    ctx.lineDashOffset = 26;
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.lineDashOffset = 0;
  }

  // ---- start / finish line: perpendicular white bar with dark dashes ------
  var s0 = samples[0];
  var nx = -Math.sin(s0.tangent), ny = Math.cos(s0.tangent);
  var ax = s0.x + nx * this.halfWidth, ay = s0.y + ny * this.halfWidth;
  var bx = s0.x - nx * this.halfWidth, by = s0.y - ny * this.halfWidth;
  ctx.beginPath();
  ctx.moveTo(ax, ay);
  ctx.lineTo(bx, by);
  ctx.lineWidth = 12;
  ctx.strokeStyle = '#f8f9fa';
  ctx.stroke();
  ctx.setLineDash([10, 10]);
  ctx.strokeStyle = '#111417';
  ctx.lineDashOffset = 5;
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.lineDashOffset = 0;

  ctx.restore();
};

// drawTrackBackground(ctx, width, height) — the ONE grass-background pattern
// (dark fill + deterministic diagonal stripes + static dark "trees"). Shared
// by the game canvas (js/main.js — via its drawGrass wrapper, because
// js/spectate.js calls that classic-script global) and the in-browser
// editor (js/editor.js). No duplicated pattern anywhere.
function drawTrackBackground(ctx, width, height) {
  ctx.fillStyle = '#2a4a22';
  ctx.fillRect(0, 0, width, height);
  // Deterministic diagonal stripes for texture (no images, no assets).
  ctx.fillStyle = 'rgba(255, 255, 255, 0.035)';
  for (var x = -height; x < width + height; x += 90) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x + 45, 0);
    ctx.lineTo(x + 45 + height, height);
    ctx.lineTo(x + height, height);
    ctx.closePath();
    ctx.fill();
  }
  // A few static "trees" (dark dots) — fixed positions, purely decorative.
  ctx.fillStyle = '#1d3315';
  var spots = [[90, 90, 16], [1020, 110, 20], [1040, 560, 17], [60, 350, 13], [540, 630, 12], [330, 40, 11]];
  for (var i = 0; i < spots.length; i++) {
    ctx.beginPath();
    ctx.arc(spots[i][0], spots[i][1], spots[i][2], 0, Math.PI * 2);
    ctx.fill();
  }
}

// Shared helper: crash detection in car.js uses THIS implementation via the
// global Track (script order Track → Car is guaranteed). No duplicated
// collision math exists anywhere else.
Track.segmentPointDistanceSquared = segmentPointDistanceSquared;
Track.VALIDATION = VALIDATION;
Track.drawTrackBackground = drawTrackBackground;
// Test-visible seam for the byte-frozen miter/arc primitive (FIT-V6 G1). The
// parallel (|den| < PARALLEL_EPS) fallback is otherwise unreachable for
// non-degenerate splines; tests/track-clipped.mjs pins its 8-point output and
// the si provenance (Blocking 2: ia/ib are ORIGINAL sample indices, not pts
// indices — callers must pass pts[i].si).
Track._miterJoin = miterJoin;

// Browser-global exposure + guarded CommonJS export (node tests).
if (typeof globalThis !== 'undefined') {
  globalThis.Track = Track;
  globalThis.DEFAULT_CONTROL_POINTS = DEFAULT_CONTROL_POINTS;
  globalThis.canonicalJSON = canonicalJSON;
  globalThis.cyrb53 = cyrb53;
  globalThis.TrackWallError = TrackWallError;
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    Track: Track,
    DEFAULT_CONTROL_POINTS: DEFAULT_CONTROL_POINTS,
    ValidationError: ValidationError,
    TrackWallError: TrackWallError,
    canonicalJSON: canonicalJSON,
    cyrb53: cyrb53,
    MAX_TOTAL_SAMPLES: MAX_TOTAL_SAMPLES,
    VALIDATION: VALIDATION,
  };
}
