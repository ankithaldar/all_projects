// tests/rotation-v5.mjs — template-family V2 generator shape/validation gates,
// V1 byte-identity and version plumbing (FIT-V5 TASK F1, migrated to the
// FIT-V6 shipped contract). Plain Node, guarded CommonJS exports, no deps,
// same harness style as the other rotation suites.
// Run with:  node tests/rotation-v5.mjs
//
// FIT-V6 migration (G4): the old radial-V2 detector (legacyV2Hash) was removed
// and the shared half-width ladder is now 50 - 0.25 d. The detector legs are
// retired; in their place this suite asserts the symbol is fully removed. The
// V1/V2 golden bytes below were regenerated against the shipped contract (the
// old ladder's V1 bytes are a globally-accepted breaking change). The full V2
// corpus is owned by tests/rotation-v6.mjs; this suite keeps independent
// shape/tangent/timing/never-throws gates over the arc-built F1 hairpin templates.

import { createRequire } from 'node:module';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const ROOT = fileURLToPath(new URL('..', import.meta.url));
const { Track, canonicalJSON, cyrb53 } = require('../js/track.js');
const rotationMod = require('../js/rotation.js');
const { Rotation } = rotationMod;

// -----------------------------------------------------------------------------
// Harness — one line per test, never throws, exit 1 on any failure.
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
      .split('\n').map((line) => '         ' + line).join('\n');
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
// Independent helpers (from the frozen spec, NOT imported from prod).
// -----------------------------------------------------------------------------
const TAU = Math.PI * 2;
const T32 = 1 / 4294967296;
const CENTER = { x: 550, y: 325 };
const round3 = (v) => Number(v.toFixed(3));
const halfWidthOf = (r) => Number((50 - 0.25 * Math.min(r, 100)).toFixed(9));
const hashOf = (points, halfWidth) => cyrb53(canonicalJSON({ points, halfWidth }));
const templateIndex = (seed, r) => ((seed ^ Math.imul(r + 1, 0x9E3779B1)) >>> 0) % 6;
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
function orient(ax, ay, bx, by, cx, cy) {
  return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}
function segCross(ax, ay, bx, by, cx, cy, dx, dy) {
  const o1 = orient(ax, ay, bx, by, cx, cy);
  const o2 = orient(ax, ay, bx, by, dx, dy);
  const o3 = orient(cx, cy, dx, dy, ax, ay);
  const o4 = orient(cx, cy, dx, dy, bx, by);
  return ((o1 > 0 && o2 < 0) || (o1 < 0 && o2 > 0)) && ((o3 > 0 && o4 < 0) || (o3 < 0 && o4 > 0));
}
// Independent closed-polyline simplicity via a spatial-hash sweep (never O(n^2)
// over all pairs — only co-located segments are compared).
function polylineIsSimple(points, cell) {
  cell = cell || 64;
  const n = points.length;
  const grid = new Map();
  const boxes = new Array(n);
  for (let i = 0; i < n; i++) {
    const a = points[i], b = points[(i + 1) % n];
    const box = [
      Math.floor(Math.min(a.x, b.x) / cell), Math.floor(Math.max(a.x, b.x) / cell),
      Math.floor(Math.min(a.y, b.y) / cell), Math.floor(Math.max(a.y, b.y) / cell),
    ];
    boxes[i] = box;
    for (let cx = box[0]; cx <= box[1]; cx++) for (let cy = box[2]; cy <= box[3]; cy++) {
      const k = cx + ':' + cy;
      let arr = grid.get(k);
      if (!arr) { arr = []; grid.set(k, arr); }
      arr.push(i);
    }
  }
  for (let i = 0; i < n; i++) {
    const a = points[i], b = points[(i + 1) % n];
    const box = boxes[i], seen = new Set();
    for (let cx = box[0]; cx <= box[1]; cx++) for (let cy = box[2]; cy <= box[3]; cy++) {
      const arr = grid.get(cx + ':' + cy);
      if (!arr) continue;
      for (const j of arr) {
        if (seen.has(j)) continue;
        seen.add(j);
        if (j === i || j === i + 1 || j + 1 === i || (i === 0 && j === n - 1) || (j === 0 && i === n - 1)) continue;
        const c = points[j], d = points[(j + 1) % n];
        if (segCross(a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y)) return false;
      }
    }
  }
  return true;
}
// Independent wall-soundness check (same contract as the generator's spatial
// hash sweep): finite points, no zero-length segment, no non-adjacent exact
// duplicate segment. Brute force here (small walls) for independence.
function wallsSound(points) {
  const n = points.length;
  for (let i = 0; i < n; i++) {
    const a = points[i], b = points[(i + 1) % n];
    if (!Number.isFinite(a.x) || !Number.isFinite(a.y) || !Number.isFinite(b.x) || !Number.isFinite(b.y)) return false;
    if (a.x === b.x && a.y === b.y) return false;
  }
  for (let i = 0; i < n; i++) {
    const a = points[i], b = points[(i + 1) % n];
    for (let j = i + 1; j < n; j++) {
      if (j === i + 1 || j + 1 === i || (i === 0 && j === n - 1) || (j === 0 && i === n - 1)) continue;
      const c = points[j], d = points[(j + 1) % n];
      if ((a.x === c.x && a.y === c.y && b.x === d.x && b.y === d.y) ||
          (a.x === d.x && a.y === d.y && b.x === c.x && b.y === c.y)) return false;
    }
  }
  return true;
}
function tangentDeviation(points) {
  const n = points.length;
  const a = points[n - 1], b = points[0], c = points[1];
  const v1x = b.x - a.x, v1y = b.y - a.y, v2x = c.x - b.x, v2y = c.y - b.y;
  const dot = v1x * v2x + v1y * v2y;
  const cr = Math.abs(v1x * v2y - v1y * v2x);
  return Math.atan2(cr, dot) * 180 / Math.PI;
}
// Longest run of consecutive near-collinear control segments (<= 15 deg per
// junction) — the "long straight" metric for an F1-style circuit.
function longestStraight(points, thresholdDeg) {
  thresholdDeg = thresholdDeg || 15;
  const n = points.length;
  let best = 0;
  for (let s = 0; s < n; s++) {
    let len = 0, prevDir = null;
    for (let k = 0; k < n; k++) {
      const a = points[(s + k) % n], b = points[(s + k + 1) % n];
      const dx = b.x - a.x, dy = b.y - a.y;
      const seg = Math.hypot(dx, dy);
      const dir = Math.atan2(dy, dx);
      if (prevDir === null) {
        len += seg;
      } else {
        let d = dir - prevDir;
        while (d > Math.PI) d -= TAU;
        while (d < -Math.PI) d += TAU;
        if (Math.abs(d) * 180 / Math.PI <= thresholdDeg) len += seg;
        else break;
      }
      prevDir = dir;
      if (len > best) best = len;
    }
  }
  return best;
}
// Count genuine U-turns. Arc-built templates spread a hairpin's turn over many
// short (~13px) control segments, so a single-vertex internal-angle test is no
// longer meaningful: accumulate the largest continuous same-sign deflection
// around the closed polygon and count one U-turn when it reaches 150deg.
function uturnCount(points) {
  const n = points.length;
  let best = 0, run = 0, sign = 0;
  for (let i = 0; i < 2 * n; i++) {
    const a = points[(i - 1 + n) % n], b = points[i % n], d = points[(i + 1) % n];
    const e1x = b.x - a.x, e1y = b.y - a.y, e2x = d.x - b.x, e2y = d.y - b.y;
    const turn = Math.atan2(e1x * e2y - e1y * e2x, e1x * e2x + e1y * e2y) * 180 / Math.PI;
    const sg = Math.sign(turn);
    if (sg === 0) continue;
    if (sg !== sign) { sign = sg; run = 0; }
    run += Math.abs(turn);
    if (run > best) best = run;
  }
  return best >= 150 ? 1 : 0;
}

// -----------------------------------------------------------------------------
// FROZEN CORPORA
// -----------------------------------------------------------------------------
const SEEDS = [0, 1, 12345, 0xDEADBEEF >>> 0, 0xFFFFFFFF];
const RS = [1, 2, 50, 99];

// Old radial-ring V2 resolver hashes, captured BEFORE the generator was
// replaced (base64 of a compact {"seed":[99 hashes]} JSON object).
const LEGACY_GOLDEN = JSON.parse(Buffer.from('eyIwIjpbImNlZGQyOTc1ZmFlN2FkNDMiLCJjNTVjNjcxODFlMjVlYzEyIiwiNTQ4YWI2ZjMwNmRjM2NjNiIsIjM5ZTI5ZmY4MzBiOGJiYTMiLCJiZGNmZjQ5N2QwYTJiYzA5IiwiYWM4MWU2ODQwNGJhOTg0MCIsImE2YzBmNmFmYzYzZDVkYjYiLCI2MWRkMzc3NjliZGVhMWYzIiwiYWNhZmNlMGM1MmVhMmU3YiIsIjY4ODMyZDExMGZhZjlkMTciLCJlYWY5ZmQ5ODE0YjE2ZmNiIiwiYmQwNTIxOTYzZjNjODQ5NSIsImYxMmM4OTVmYmVlNzBlZDIiLCI1ODA1MGJjN2RiZDU2MTA5IiwiYzlmZTNlY2M2N2I4ZjBlMCIsIjFkNjE2ODc2MjMyMGQ5NjgiLCIzNmIzOGVmNjQxODNjNzMzIiwiMTZhOGRiZDJjMjkwMDIzZCIsIjZmNDdhOTgwMTZkMDQwMDAiLCI5ZmQ4NmVhOWVjMTM4ZWM0IiwiYzg5NWIwOTE1ZDlhMjkxMiIsIjI5M2RiNTM1OWUyNjRhZDMiLCI2ODYzMDkxNjNjMjRhNTljIiwiODBkOTQ2ZGFhMWUyZmViYSIsIjI4NzM3YzYzZmM1NzE3MjEiLCIwN2MyZjg3MDIzYjQ2N2MzIiwiMmVmZGFhZjZhMWYxN2E2NSIsIjQ1NmJiOWQyYjI2ZDI2YjkiLCI5MGNjYjc2NTkyOGVhODQxIiwiYTM1OTRmNzBlNzNjMGJlNyIsImVlNzBiMTY1ZDk1MzA4OTIiLCJhYjgwZmMwMGQ4MGRiNmVjIiwiYjYzMjRmOWM3MzQyMGMyYSIsIjEyMDRhMGI5OWY5YmIzODIiLCJiMTcwOWFmNDY4Y2FmNTEyIiwiZDUzYWNmMjVlMzhkYmY4NiIsIjViOTYwNDFjNDUyMGNmMmIiLCJjNzM1ODk5YWQyNTRhNGUwIiwiYjNlYTVhMGYwOTMyNGY4OCIsIjhjOTA0Mjk2ZmE1ODEyOTciLCIxYmUzNjUyYmM4ZTg5MDQyIiwiOTkyNjhjNzczMjJlYTQ3ZCIsIjJjOTliMmEwZjU5NjY3MmYiLCJjNWI2MDZlZmMwNTJiZWU1IiwiYTNlMjFiNWZhOTQwZDBjNiIsIjg0NDFhMWE3NDE1MDZkNjkiLCJjMDBmZTQyNTc3NGQ1ZjBhIiwiOTIxYjhjNDlmMmMxNDFmNyIsIjg2OWJjOWFkMmI3ZGI5ZWEiLCI3ZDcwNjE4NTkxODYzNTFhIiwiM2ZiMDRhMGU5YTYyOGVhYiIsImE5ZTRhNmM5MDdkYjkwZWQiLCI4YzZmZDY1YjIwOWI4M2ZlIiwiZmE3OTQwZjdjMjk2N2I4MyIsImI0N2NhMWY0ZTczZjFjMWUiLCI3MzJhYjJmNDQ2YWMwZGNjIiwiZDAzMmQ3MmMwODVmNzE3NiIsImViOGJlYjUwYTFlYjhiODgiLCJmMTJhZDY2ZDBiY2M4YjE4IiwiNTE4Y2M3MDlhM2Q0MWQ5YyIsImZkNjA4M2FlYTJkNzJlOTciLCIzZTRjMDdkYmI4NmFmOTVmIiwiOThlNDY5ZGMyYjFjZjRjMCIsIjI5YWFjY2Q0ZGYzY2I4ZTYiLCJiNjU0YWUxYzUyZmIyMDQxIiwiZTkxMDQyMzY2MzVhZTQ3NyIsImUxY2M5YjM0YzVkNmU2M2QiLCI1MWVjZDVhZmFkN2U3NWRmIiwiYTJmODcwOTE0MGVlMzhmZiIsIjg4ODFmMTkwYjUwZGU4NjgiLCJkZGFhMTkzN2JlMGZmZmI3IiwiNmQ2OTRiODY0MGRmMGQ4MiIsIjZjMWNhYTFlYjA2NmE1MmEiLCI2NDg2MTI0OTdhYzA2Y2JlIiwiMzQ5MzVlZWNmN2MyMzA0NiIsIjc2OTg4OGJmYThiNmYyZjUiLCJmYjkzNTUzNTllOTE3OTJmIiwiMGRhMGI0MjAxZTZkYmRiNiIsImYzODk3NmYzMmQ4NTRlZDciLCJiYjI5MmU5YzhlODk3NDdjIiwiYmZmMjM5Yzg1MjhlMjIzZiIsImE4ODQyYjY2ZmZjNjYzZjMiLCI4Mzg3MmI2Yjk0ZDkwOGMzIiwiYTFjY2JlOTViNmI3NWE1OSIsIjY1MDVkYTIwODgyODY2MjkiLCIyZDVlZTc1N2IyY2ZkNDA1IiwiNDcyY2Y4MTg0ZTNiNjQxYiIsImYyNmEyNWNlNTQ2OGMyODUiLCJlMzMzOTk0MjY1NDlhMDM0IiwiOTI1NDAyZGYyOTk1YWU2NyIsImYxNTA3NWFhNjhiNTQ5YzQiLCI0MjQ3YTBlYmEzYzIxOTYwIiwiMWQ1OWVhNTdmMzkxZTY0NiIsImNhZTc5YzUzOGQ4ZWM3NTAiLCIwNTEzZmJiYjFmZTc3N2ZmIiwiN2I1NGY4NzkzMWJmNmY3ZCIsIjAzOTI5MGE3MDJhZmNhODMiLCI0OWQ1NDNhNDJkNGNhNmQ5IiwiZTM3OGNmYjkwYThiOGFkYSJdLCIxIjpbImFiODEyNDE0Y2Q5OTFhYjQiLCJlY2M3NWFjYmU4MDZkOTM0IiwiY2RlZTkxZWFlM2FjMWU1NCIsImQ4YjQ2YmY4ZjBiZjRhZDEiLCIxY2I2ZjVkZWM2ODdiNDY2IiwiMWEzNzJlODllN2E3NzFhYyIsIjg2NDc4MTliNDg5NzE3NDEiLCI5MTk2ZTFkMGFkMzZhYjlkIiwiZWUxMDRjMmQ4YzJjOGFkYiIsIjJkNjMwOTgzODQ1YzM5ZDQiLCI2N2JmYjFmNDViMmUzZDFlIiwiN2E2YjM3NTY2MzlhNmZkMCIsIjc1MTJlYzlkYWYzOWY1NTciLCI3NTRkMjVjZjZhYmFkMDM1IiwiZWFmNDM1MTQ0ZTZkOWU4YSIsImRmYjMxMzdlNGJkMTE3N2MiLCJjMzgyNGQxMWRiZDBlNmVlIiwiMjIyZDA5ZmQ3N2RkNDkzMyIsIjAyN2NkNGMwNjEwMGQxMjkiLCIwYTQyMTgzY2ZkZTdlZWViIiwiYzczMGMzZmFlZDEwMDE1ZiIsIjMyZDdmOTZhMWI4ZmU0YTIiLCI0MmY4NjY1YzdiNDRkNjQ0IiwiYjkxNWE3ODI4ZjgyYzNjNCIsIjVlZDhiMjY4NjczZGVjYzgiLCJkNjI5MDhjMmJjYmRmMTA5IiwiZDMyY2M1ZmU1N2RiNGE1ZiIsImI3MzMxM2E5ZDk4Yjk5YmMiLCIyNDZjNjgxOGI5ZjkxOTc3IiwiYzBkNjA2OWRhNWY5YWY3MiIsImU5NmQ1Nzk2OTM0YTcxYmQiLCI0MjJkOWY1MWUzYjgxZjcxIiwiY2ZhNWMzNjkzNzI1YTljYSIsIjU4ZTYwYTgxNThjMDVkZGQiLCI0NDI5YmU3NTUyMWExYTBlIiwiNTJjZTRmZjBkYjQzOTdjZiIsIjk0NzRmNTZkNzI2NWJjNDYiLCIxNGViOWU2ZWQxY2UwYmY2IiwiMjI0YmMxMWNmOGYxYzNiNSIsIjEyOWE2MDExYzYzYTkxYzciLCI3YzEwNjFkMWMwMWM4MDNiIiwiMmZlN2VlMmU3MjM5OTc1OCIsIjI0ZDVlNDY1NjhkNjYwMmMiLCIyMWE5ZGRjMWIzMGNkOTM5IiwiZTRmZDY5NmRkYjdlYjgxZiIsIjNlYmU0NDgxNDM5YWRjOGMiLCI2Y2Q5ZDRlOWM4YTA4YzYwIiwiNDJkYWM5MDExYjVjOTc5ZSIsImI2MmY5YjMzYzRkOWVkOTQiLCI2Y2ExZmYyZDU4NTg3ZjM5IiwiODhhZTI4ZjJiYzhjZDYyYiIsIjRlODg5ZTM0YTVkNjI4YzgiLCIzYWViNTZkYmZhMzU5OGYxIiwiZGZiMmNjY2VjNjM3YzdlMyIsIjcxODU4NjQwZDI4MzdkOGMiLCJkMmM3ZmViYjJiYzNmNjVkIiwiOTI0OGE5YWY3OGQ0MzE3MCIsIjYxNTYyZTliM2E0Mjc4MWIiLCI2YjNhMjVjNDhkMWYyZjJiIiwiOWE5MDc4NWQwOGExZTI1NCIsImNmN2E2ODU1N2ZhNjVkMjciLCJmMTg2ODkxOGZiYWYwYTAyIiwiOWJiMTFhMGI4MDI0ZDU2MSIsIjU2NTcxOGZiOTM0YzdlNDQiLCIyYzQ4NTUwNjU5YzliNmMyIiwiYThhYWMzNjVmNmMwOGQ3ZCIsImZhYTljZmY3NTUzMjNkZTkiLCI5ZjY2MGI4NDg4YWNkMGY2IiwiODVlMDcyMGM0NWFmNjM0NCIsIjZiNzk2YmFmODNhN2ViM2EiLCJlMjAxYzI2ZWYxNWQxODEzIiwiOTc5OWZjNWZlZDIxNTYyNCIsImEyNjk4OTBjNWUxYWQ4MjMiLCI3NjVmZDlmZWQwNDRhNjI5IiwiNmQ1N2NlMTQwY2JkZDQ0NCIsIjg4YWRlMzc3NDc5ZWJlY2MiLCIxOTFhMWE5MjI2MmQzMThlIiwiMmQxNTNjYTU5YjNjMzk4OCIsImIzMDU4OGY5NTYwYzczYWUiLCIxYmQ5NDU4ZmNmZDQxMWQwIiwiMWZjMmQwN2YzOTYzZjg0ZSIsIjdmYmJhMjYwZmE0MWQxZjciLCI0MTc3YjA3NWU1NjMzNWE3IiwiNWQ2NzU1NDJkNmJkMDU2NCIsImYyMDkxY2QwMzViYzljMjAiLCJlNmJhZDkwMmJkNTcxN2I4IiwiNmMwZGUyYmUwMDZlYjgwZSIsIjUxYzI3ZDkxODQxNjFlODgiLCJkNjI2ZWRlZTk4NjliMDYxIiwiOGEwMDE0YmNhY2UxZjY4OSIsImQwMGQxODQ4ODk4Y2VmMzMiLCI5MWZkZTQ1ZjE1ZTkwNjc1IiwiMzg2OWE1MjUyNWFjNGZjYyIsIjZlYmNhZDg1ZTI3YWQ5OGIiLCI5ZjY1YjUwNjIzZWRkODQ4IiwiY2JlOTcyZTk0MjIyNDM0ZiIsImUxZDMxOTczNGNhNzcxNDAiLCI5ZGUyZTcyNzgxMTQ1MzVhIiwiMzJhZjkzZTc2MDU3YzUyNCJdLCIxMjM0NSI6WyI2ODE1YTRmMTEwMzVhMjI2IiwiZjU4ZjhlMTYzNTdjNmZkYiIsIjQ2YmU1ZDMzMjNkMjc1MTIiLCJiYjIwYjg4YjFkOGU4MGZlIiwiMDI1NGE1MmNhOWI1OGQ1YiIsImQwODE3YjFjMmNiODZhZTgiLCJlYjdiYzlhNjFlYTAyYWM4IiwiMWZiMGY3N2Q3MTUzMjNlZCIsIjEwYzI3NGU1MTQzZTQxNWEiLCJiYjdjN2MxOTlmNzA2MDZjIiwiZjBmYzgwY2MzOTNlYjQ5OCIsImQ0NjY2OTY1YzRkMzliNGMiLCI2YTdkZjgwMjA1YjEzNGM2IiwiN2U4NzU2MWEzYzE1MmIwNiIsIjMzYzk5MzVhYmY5ZWM5M2IiLCIyMjJlMDI1N2YzNmI1ZmE1IiwiZTM1ZDRhOGQzN2ZlYWU4ZiIsIjViYTg1MDE5MDJiNGE0MzgiLCJjOTg3YWQ4ZDdkNWQ0M2NkIiwiNjA2MWZhZjhkZWFiYjI0ZSIsIjBjMDZmODFjZGZmYWI1NDMiLCIyNjA2NDAwOWNkN2E4MjkxIiwiMDRmY2I1N2UwNTkzODQ3ZSIsImU4ZDI2MDg3OWQzMzMzMTUiLCJmMTNiMzZjZTRiNmY4NWE1IiwiM2ZlMjEzYjEyNmU5MzJiNSIsIjc0MWJjNzUxNzEwYmEzN2EiLCJjNzFlMDM1NWMxMDI0NWM0IiwiZTkxMjhmNmRmNmZkOWZhMiIsImMwYjM4YzY5MzI5OTc1NzkiLCI2NWEyOGViNjNiYTdhMzNkIiwiZjcxMWNmMTlhNGI5MGRhZiIsIjE4YWRkZmY3OGM5MWE1ZmMiLCIwMWY3Y2RmN2JiNzQ1N2M5IiwiNDE1Njg3YjUxYWQzYjRhMCIsImNhNTY4NzU1MjlhNWM4YWYiLCIwYjY4NmMxMjBjZDZlNTI2IiwiYzBlYjg4OTg3NjAwYWU0NSIsIjdiMDdiYjhmMDRmZjAwYWYiLCI2NmJiNTNmNGNmNjFkZTNjIiwiOTQxOTZlN2Q3ZDZiMWM5MCIsIjNiMDc2OGU3MjBmYzI0OTgiLCJkZjI1MTE0OWQ5NTk2MTYyIiwiNTVjNWY5YTM0ZTE3NGViMCIsIjA3MWJiMzZkNTg0YzVlM2UiLCIxNDU0YThmMzIyNGQ4NmEwIiwiNDA2ODUyMWVkYzE3YTgzMSIsIjBlNWVkZWEyYzYyMjJlZGQiLCIwODI2MmY1YjFhOGQ5MTI1IiwiNzBjYTdmM2NjYWZiMjc5NSIsImVjMTcwZDcxMDkyY2JiZTQiLCIwNmI4NTdmNGI1ZDBhOWQ0IiwiMWUxYWQ2YTIyODk0YmU3NyIsIjk0ZTc4MGMwZTk3YTQwNjciLCJlNTRlMmE5MjBkNDk1YzM4IiwiZTY0N2Q1NzcxNjIwYTkyNyIsIjIyMTBmYjBkNzdmMzMwNzUiLCIyMjM1NzllYmQ1Yzg1MmNkIiwiYjZmNzc4MjQ5YzZiZWYwMSIsIjY5OGVhNTBiM2NlOGY5ZjYiLCIyNGI5ZmViMDMzMWJjNWVhIiwiN2E5ODAzODFiOGY1YTFiNCIsIjg4YmI0MGU5Y2VhOWI1NGMiLCJhYmQ4NzNkYTQzNGExMTQ0IiwiNDQ2ZmY4YmVhMmQzZDhlNyIsIjg1MjViYTIyY2M3YjEzNmMiLCJjODkzYzMzNmQ3MDNhMzRmIiwiZDdkYWIzMjAxMDdhMjI3YSIsIjQ2NjE3ZWNhY2YzZThkYWQiLCI5MTVlNTE5ZTRmMDQ5ZWYxIiwiYzQzODAyM2UyNDVmY2ExYiIsIjk2NTc2ODc5ZDE0NGE2MWUiLCI4ZDJhNzBlMDllODkwNWYzIiwiMzg2ZjczMzRlMmNjYWRhMSIsIjRiMjc5MjkwYTg1YzMyMTMiLCIyYzMyNDgzZDNlYzllNjJmIiwiOTk2MDAzM2ZkYzUwYzk3MiIsImUwMTZlNjkxZGE4NTBlY2EiLCJkMDA0NDhlNTEzZjMwYzZiIiwiYzk1NzFlN2QyYmY5NjBiNCIsImQxYjgwMzU1OWZmZGE5OWEiLCI4MTQxNTMyODE2MmU3MTVhIiwiYmZkMmI0NzU4YWZiYmExYSIsImY3ZWE0MjI3ODhhZTIxYzUiLCJmNTk2YmFjZjA4Zjc4OTY3IiwiN2YzNDY3OTI5MDMxODNjYSIsIjg1ODk0ZmYwMjE5MmE4YzkiLCI5OTdjZDNjYzUxNWIzMDRhIiwiYTgxMjgyNmU2MjZhNGNiNSIsIjMwYTRlZGRmNTkwZDVlOTAiLCIxYzM0MmRkODQzYjQ4ZWVhIiwiZmI2M2Y1YWMwMGQ0MTA2ZiIsImVhMjkyNjMyOWYwYWY0ZTQiLCIxNDZmYjA5YjJmYmU4NGNiIiwiOTkzNzk3ZGJmMjI0ZjhjYSIsIjM3MDBiYTNlODU5NGQ4YjQiLCJiNjM2MDRkOWRhZjA1NDI3IiwiY2MzNjAzNTYxZGY0OGExMSIsIjgxMjRkNDE5MmQ2MDJiOTgiXSwiMzczNTkyODU1OSI6WyJlYTRkODZmM2FjZTNmMjU2IiwiNWY4NzUzZjFlMzE5ZjZhZSIsImJjMjQ0NWRlNzMwMzc1MmEiLCI4ZDQ4ZjBiMzM1ZDRkY2I1IiwiNjE2ZTJmZDBjMjcwMTBlNyIsImM1NjZhODc4Y2JhYjI1ZDkiLCI5Y2M4MDcxNmE2YTY4NDM0IiwiODVhMTM1ODJlMjkwZjA3YyIsIjkyNjI2YjExMTc2MThkOGQiLCI2MDQ5OGM5ZTQ2YTE2NTU5IiwiMTQ5NjFkZDkxMzg3MzZhYiIsIjFhYjMxOGZlN2U1NTdiZTMiLCI4YzgxYzM1Y2ViYzZhMmU1IiwiM2U1YjIxMzBhMzM4NDA2MSIsIjBiYmYyYWU3OWYxODlmMTkiLCI2MjFlZWU2ZTNhMmJkYWYxIiwiMDAyMWE4OTVjMzg2ZGU5MCIsIjQzZWYxOWU3ZjUyMDAwZWUiLCIwY2Y0ZTVkOTcyMjY3NTU1IiwiYTk3NjYwMTYzZjZlZTA4ZCIsImM5NjdiMmE5OThhZmQ2MmIiLCJmNDIxYzNhYTA0ODVlNzI2IiwiZDdiYzUxMTQzYzU3ZjRiNCIsIjkwZTE5NzY4OWI1ZDVhOTYiLCIyMjFiM2JiMWRjMTljZDdjIiwiNWJkOWIxZjQwZWM3MzRkNCIsImFjNjc0OGFjODVmZjc0MDciLCIzODA5ZjkzYjU1YjkzZTJjIiwiM2Y1YmQ0MWM5ZmI1ZmRiYiIsIjIwYTk5ZmRhNTljNWYyOTMiLCJkZTY3MmQxMGVkYzFlYWQ1IiwiZjJjYzI4ODRhNjQwNzc3MCIsIjkxMzQzNDNkMzFkNWQzZjkiLCIzMDE0YTM0MmRmZjQ1N2U5IiwiODdiNTUwZTkwZmFhZjU5YiIsImUyZTAyN2RmZjBmZDc1OGUiLCI0ZmE2MTIxNjc1MTNmMmM5IiwiOTVhNzI0ZTE1OGM2ZDE0MyIsIjgxODQ4MTI3OGNiNmM1NDIiLCIyOWZmYTFlNGFmYTM2YTBjIiwiZjVlNzVlYTMwZWM0NzNkZSIsImNiYTZmYjFmZWE0MDI0NzQiLCIxODdhYWMzY2VhNWUxZWNiIiwiM2Y0MjEzYWUzM2ZmMDk3MyIsImU2M2Y3OGNmOTI4NGE2ODUiLCJhOWUzNzMyMzk5MDAwOTE2IiwiZTIxZjFmNzZhNmE2NDYwZSIsIjM0YzdkZWUzYTAzNWZhZjEiLCI5YTM5MWM1ZWZkMzY1ZDEyIiwiODU4MzE4NWE4MTgyZWM1NCIsIjBiYWQ4NDEyZDYzNmUwY2YiLCJjZmI5NjBiNjY3ZjZlY2RiIiwiZTFiZmY0NTFmMzI2OWYyMyIsIjFhMjk4YTNiOTI4ZWYzOWEiLCJjYjQ2OTg5OThlNWM4ODU5IiwiNmM1ZWU5NTUxMmI5Y2IzOSIsIjAzYzg2YzFhMzQ2YzhmYzAiLCJhODY0OWE3NGE3OTlhMmIzIiwiOTFhYWRjMTQwODNiZDE1OCIsImQ4ZDBjZjQ5NmY3NzNlZmQiLCJhZTQ2NmM4YWIwYzZkYjBhIiwiN2IzMjBkNDdkMDE4ODY2ZSIsIjRjODVmZjFjNjZmNDA2ZGEiLCI5OTY3MDMzZmI4NWNkNDZjIiwiOTFiODc2NmI0YWVlYTUyMCIsIjA4NGJiYjczNjkwYWViYjkiLCIwOWIxYWNmZDZlNDM2ZTRhIiwiYTJlYWM3MTg0YTkwM2VmNCIsIjZjM2FlMDJhYTI4ZGFjOTQiLCJmZWZmNGIwYjlkMjM0YzI2IiwiMTI3YjRkM2MwMGYyYWVhNyIsIjc3Y2YyMWI5OWIxM2FjNWYiLCI4MTBjMmJmZTYxNmI5NDAzIiwiOTBlMjFkODI4YjU3ZTg5NCIsIjMwMWM0ZDMyMDVhMmVjYTUiLCJmYjhjMDE3YWNiOWYyZDc5IiwiMTQyNWYyZTc0OTA5YTM2OSIsIjllMWJlY2NhNDM0ODRjY2MiLCIzM2UzZWVmZDQwYjAwNGI2IiwiMWFlMGZjNTYyOGI2ODFmNiIsIjU0MTBhZDgwMDFiZWRjMGIiLCJiZmRmZTQxYjUzNmM0MmIyIiwiMWY5MTc0OGY2N2U3OTE4NCIsImFhZDdlNmNiZWU5NThlOTUiLCI3NjU5MWMxZmFlMTA1MDkyIiwiNzkxN2NhNDFjZDNlYjE3MyIsIjU4NzQ1NGUxMTdiZGE5YzciLCIxOTMwMzJlOWMxMTBjMmJiIiwiY2NhNDlmNmYwMWJkYzgxMyIsImI1YjY4ODJmZGJkNzdmNWQiLCI1NmRkODdmMDMzMzM2NjZhIiwiODI5OTM4OGY1YTMxZTczZiIsImUxZTAxYTEwNWNhYTdmMmIiLCJjMjE3OWNmYjlhNjFiMjMxIiwiOTUzOTk3MzA0NTZmNGQwNSIsImRmYWEzNzk5MjQyYzI4MjAiLCIxZjkxZGE3MzBmZmE0MjllIiwiN2QzZDQxM2YxZTIxZGFkZCIsIjQzZTA2ZjVkMDM3OGZhY2EiXSwiNDI5NDk2NzI5NSI6WyJlNDFiM2MxZGE1ZjlmNTNlIiwiNWYwYzg5NDZjZTJiNDAzMiIsImMzMGIzN2YwNjM3N2U5MzIiLCJjNGYyN2E4ODBhMDViMmMwIiwiNjllYzRlOTVkYjMxZjBlMSIsIjIzNzBkY2I2ZjY3NWZjOWIiLCJhMWJlOTIxMWYzOGExMWY2IiwiZGFkODdlMDcxNTdjZDFjYiIsImI3NTJhNzI0YWZhMDNhZjkiLCJkMTU0MTc1MjZhYjc0ZjkwIiwiYzQxM2FjMzJhZTVhMjg2NCIsImJhZjA0MjFkYmE4MzM2Y2UiLCI2ZWE5MGRjYzVhZWY1NDM4IiwiNTYwOGFlY2Q5ODE4OWJiMSIsImIyNzM0ZTM0MjU4MjBiMWYiLCI0M2UzZjkxM2MzYmM1N2I4IiwiYjY3NjdlZjFhMDNjMjQwOCIsImQ5MDllYWFiZWRjOGJlNTEiLCIxODNkNjBmOTE3OTUzMGIxIiwiY2IwMTAwNDAyZmEzMDVmNCIsIjcyZmQ5Y2UxZDAxMDVjYTkiLCI3ODUxYTA0NjI0MzkxYTNmIiwiMDA1NzM0OTZmODRlODJjNSIsIjcxNzQwMjNiZWE0MzVkNTIiLCI3ZDJjZTA2MTFiMjlkNDcxIiwiYjIxZDc2N2ZiMTNmNTRkYyIsIjJiN2Y5OGNjMmJkOTdjYjMiLCI0NmU3Y2RjODBjNWJiOGFhIiwiYmMxZGJhYmI5ZWMzODUwOCIsIjJjZDYwZWFlMGEwY2M1NmMiLCJjMjQ3OGMwNzc3OGNjZGE2IiwiNDhkZmUwNTMwMTBkZDYxZSIsImViMDVlNzM3MDljN2UwYzQiLCJjYTNkZGNhNjUwZjQ0MWM2IiwiZjUwNjQ0OWQ3MjMzZjE0MCIsIjY2YTE1MDgwMDQ2YTRjYjAiLCIyMjM2M2M0ZmI1YjQ3MzFiIiwiNjQwNmIzMjBmZGI2OTg3YiIsIjEyYjM4ZDVhYmM2MmYzM2YiLCI3NWE0ZWRlZmFkODUwYTRhIiwiMGE5YTI0MjBmYWUxOWNmNiIsIjAzMDliNmU4NjE4NzcyZWMiLCJjMTFhMmMzOGRkM2Y4ZDk2IiwiMTE0NmQzNTdiOGE2MTQ0MSIsIjExMzgwODc5ZTUwNGM0NTgiLCI1MzE1NWNiZTU5MTQ4ZmNkIiwiOWQ0NzZlNzAyZTQyNmU5NCIsIjI0MmZlMzVmOTdmOGQ2MmIiLCI4ODI2ODJjNDUxZDNiMjcyIiwiMmE3OWE5MjMwOWVmOGZiZSIsImU3MDNjYzlkMjhjNzg0MjIiLCIyN2MyMjUzYzg5YjYwOWVhIiwiYTcwZGExOTRkMzE3Zjk4NiIsImE0MDI4MDgwYTg0MTQxZGQiLCIyY2U4YjM1MmJkNWQ2MDJiIiwiOGQ2ZWMzMTIyNWFhMmZiYSIsIjRjYjg0MmYwNTlmNjEwNWMiLCI2MDk2M2MwZTMwNDE4YzY0IiwiOTE2NmNjM2ExZjE4OTk5OCIsIjE3OGU3NThhZDExN2U4Y2MiLCI3NzQ3NDA5MzMzNzYyNGYxIiwiZmJiMDFkZmNhNjAyODFlOSIsIjEzNThkNGExYTJhZDNhYzgiLCIxMjk3ODFjZTliYjQxOTNmIiwiY2Y4ZmI3MjNjZDY2NmU2ZCIsIjQ5ZTQ4Njk1MTZjZDBhNTYiLCI3MDU4MDVmNjZmNTRhZDE2IiwiZjM1NTNkNmMwNjJmNmU4OCIsImYyNjliNTNmMTllZWUzOWYiLCJmN2MyMTMyMTRhNTdjM2Q0IiwiZDY4MWYyODE1MmVmNDA1NiIsIjI5MTQwZDJiZjM1MGJkNWEiLCIzMWI1ZTUwNWM5NDkwZjRlIiwiNTgyYmU2ODhjYWVlYzMyZCIsIjFhZjJhNGUxZDFkMzFmODkiLCI1Y2FhYTZkYzk4ZTE1YWUxIiwiMDRkZWY3ZGNmYWVmN2EyOCIsIjBkNDQ0MjAzNmEzOGMzYjQiLCIwZDc0ZDc0Y2JkZjNhYzdlIiwiOTIzZTViMGIzZDVkZTE4OCIsIjM5NjJmZTk4YzJlYmMzOTEiLCIwY2I4MTkwYTIzZDhjODFiIiwiNzQwMTg5OTllODgwNzQ1OCIsIjg1MjRmZmY1YTMzYzlkMjEiLCJkYzkwZDlmMTI0MDU4OWYxIiwiYjFiYThjMTc0YzI0M2UyNSIsImFjODc5NDgyNTAzODU1MWIiLCIxZWYzZWU1Yjc5ODc5NzUyIiwiMmZiN2QxNDg2N2QwNzlhYSIsIjI2MmU3MjQyY2E4NmRiNmEiLCJjODU2ODUzN2Q5ZjVhYTQzIiwiNDU0OGNkNTk5NmQxMjNkYiIsIjU3NmY4N2IyMDg4ZDc4ZDkiLCJmZGUzMDBjZjg2OTY2NDkzIiwiNzg1Yjc1NTBiM2E5ZTMwMiIsIjI1MThkMTg3ZmI1ZGFjZWQiLCJmMDI2YjEzY2UzYzBhN2JhIiwiNzdlYzk4Nzk4NTZlOGM2NiIsImJmNmU2ZTE3MzZhYWJiMDAiXX0=', 'base64').toString('utf8'));

// V1 golden hashes regenerated against the FIT-V6 50 - 0.25 d ladder.
const GOLDEN_V1 = [
  [0, 1, '41da7bf2ff4bc5ed'], [0, 2, '4710ef3aa4609772'],
  [0, 50, 'cf93a3f4bb85dfa0'], [0, 99, '9fdb255e7d49c1fb'],
  [1, 1, 'caab32e6e15bd6d7'], [1, 2, '67139ff7454b023a'],
  [1, 50, '1399522a57b967e3'], [1, 99, 'b47f7b90209dde22'],
  [12345, 1, '1855f2d7ccca475f'], [12345, 2, '91fe7b82f45a8074'],
  [12345, 50, '7fe82c2ed11a487f'], [12345, 99, '1b9883149a735ef6'],
  [0xDEADBEEF >>> 0, 1, '96c4eab5ebb61b40'], [0xDEADBEEF >>> 0, 2, '80f9ea11f09f0595'],
  [0xDEADBEEF >>> 0, 50, '700584ba66dcd5a0'], [0xDEADBEEF >>> 0, 99, 'aa2edc869b297579'],
  [0xFFFFFFFF, 1, '99de0fe90dd709ae'], [0xFFFFFFFF, 2, 'f027c4cf4f5ae6be'],
  [0xFFFFFFFF, 50, '5223dede9a221269'], [0xFFFFFFFF, 99, 'eb8f63e7c0556139'],
];
const V1_BYTES = new Map([
  ['0|1', `{"points":[{"x":777.124,"y":325},{"x":745.961,"y":419.37},{"x":682.511,"y":491.163},{"x":597.586,"y":533.49},{"x":501.092,"y":539.28},{"x":408.62,"y":502.285},{"x":341.182,"y":425.561},{"x":315.74,"y":325},{"x":337.501,"y":222.666},{"x":401.317,"y":138.557},{"x":496.097,"y":88.836},{"x":604.518,"y":86.143},{"x":702.005,"y":134.392},{"x":763.77,"y":222.054}],"halfWidth":49.75}`],
  ['4294967295|99', `{"points":[{"x":773.235,"y":325},{"x":713.771,"y":403.868},{"x":650.237,"y":450.693},{"x":585.294,"y":479.632},{"x":519.314,"y":459.446},{"x":460.531,"y":437.191},{"x":381.026,"y":406.374},{"x":336.429,"y":325},{"x":343.671,"y":225.637},{"x":402.342,"y":139.843},{"x":500.366,"y":107.54},{"x":600.047,"y":105.729},{"x":701.303,"y":135.271},{"x":770.737,"y":218.698}],"halfWidth":25.25}`],
]);

// New template-family V2 golden hashes, regenerated against the FIT-V6 shipped
// sharp-template generator (the 495-vector corpus lives in rotation-v6.mjs).
const GOLDEN_V5 = [
  [0, 1, '9a44a5c9dadd0597'], [0, 2, 'acf17e3f35df9371'], [0, 50, 'fe06aa68e72580a5'], [0, 99, '47fb42f3868710df'],
  [1, 1, 'f73efc03c72ad1fe'], [1, 2, 'c53b4eb9edab5861'], [1, 50, '0cc946187de240e7'], [1, 99, '84efa5da34c93c98'],
  [12345, 1, '6deb504c68c9cff3'], [12345, 2, 'f446281e6e01022c'], [12345, 50, '7fac3f5ae5f5007e'], [12345, 99, '912fc29e79c9c5c5'],
  [0xDEADBEEF >>> 0, 1, 'f677a1db129583a3'], [0xDEADBEEF >>> 0, 2, '1be2e6c74eabf6fd'], [0xDEADBEEF >>> 0, 50, '87af15461ef5a1b1'], [0xDEADBEEF >>> 0, 99, 'db8aa7b51953621f'],
  [0xFFFFFFFF, 1, '1d492f904ced3b79'], [0xFFFFFFFF, 2, '0b1390463c46d354'], [0xFFFFFFFF, 50, 'e039226d58a3b55f'], [0xFFFFFFFF, 99, 'bc8d3b1dce9499e6']

];
const V5_BYTES = new Map([
  ['0|1', `{"points":[{"x":522.448,"y":70},{"x":545.422,"y":70},{"x":595.965,"y":70},{"x":693.434,"y":70},{"x":743.977,"y":70},{"x":766.951,"y":70},{"x":786.096,"y":70},{"x":799.995,"y":70.7},{"x":813.754,"y":72.794},{"x":827.232,"y":76.261},{"x":840.293,"y":81.064},{"x":852.806,"y":87.157},{"x":864.642,"y":94.476},{"x":875.683,"y":102.948},{"x":885.817,"y":112.487},{"x":894.94,"y":122.996},{"x":902.96,"y":134.369},{"x":909.797,"y":146.491},{"x":915.381,"y":159.239},{"x":919.655,"y":172.483},{"x":922.576,"y":186.09},{"x":924.114,"y":199.922},{"x":924.254,"y":213.838},{"x":922.994,"y":227.697},{"x":920.348,"y":241.36},{"x":916.342,"y":254.688},{"x":911.016,"y":267.546},{"x":904.425,"y":279.803},{"x":896.635,"y":291.335},{"x":887.725,"y":302.026},{"x":877.786,"y":311.767},{"x":866.918,"y":320.459},{"x":855.231,"y":328.016},{"x":838.651,"y":337.588},{"x":818.755,"y":349.075},{"x":774.983,"y":374.347},{"x":678.686,"y":429.944},{"x":627.364,"y":459.575},{"x":531.066,"y":515.172},{"x":487.295,"y":540.444},{"x":467.398,"y":551.931},{"x":450.818,"y":561.503},{"x":438.431,"y":567.846},{"x":425.469,"y":572.912},{"x":412.063,"y":576.649},{"x":398.35,"y":579.02},{"x":384.468,"y":580},{"x":370.557,"y":579.58},{"x":356.759,"y":577.763},{"x":343.214,"y":574.569},{"x":330.059,"y":570.029},{"x":317.426,"y":564.19},{"x":305.444,"y":557.111},{"x":294.235,"y":548.863},{"x":283.912,"y":539.53},{"x":270.374,"y":525.992},{"x":229.761,"y":485.379},{"x":216.223,"y":471.842},{"x":207.302,"y":462.02},{"x":199.364,"y":451.389},{"x":192.481,"y":440.047},{"x":186.717,"y":428.096},{"x":182.125,"y":415.648},{"x":178.747,"y":402.818},{"x":176.614,"y":389.722},{"x":175.746,"y":376.483},{"x":176.151,"y":363.221},{"x":177.826,"y":350.059},{"x":181.15,"y":331.205},{"x":191.124,"y":274.642},{"x":194.448,"y":255.788},{"x":197.457,"y":242.557},{"x":201.749,"y":229.684},{"x":207.281,"y":217.294},{"x":214.002,"y":205.506},{"x":221.845,"y":194.433},{"x":230.736,"y":184.183},{"x":240.589,"y":174.853},{"x":251.309,"y":166.534},{"x":266.992,"y":155.553},{"x":314.04,"y":122.61},{"x":329.723,"y":111.628},{"x":342.525,"y":103.668},{"x":356.118,"y":97.148},{"x":370.339,"y":92.146},{"x":385.021,"y":88.723},{"x":403.875,"y":85.399},{"x":460.438,"y":75.425},{"x":479.292,"y":72.101},{"x":491.252,"y":70.526},{"x":503.303,"y":70}],"halfWidth":49.75}`],
  ['3735928559|99', `{"points":[{"x":503.264,"y":96.56},{"x":520.956,"y":107.323},{"x":559.879,"y":131.002},{"x":645.511,"y":183.096},{"x":801.718,"y":278.125},{"x":887.35,"y":330.219},{"x":926.273,"y":353.897},{"x":943.966,"y":364.661},{"x":958.709,"y":373.63},{"x":968.989,"y":380.609},{"x":978.524,"y":388.576},{"x":987.217,"y":397.454},{"x":994.984,"y":407.154},{"x":1001.746,"y":417.577},{"x":1007.437,"y":428.623},{"x":1011.999,"y":440.181},{"x":1015.389,"y":452.135},{"x":1017.57,"y":464.367},{"x":1018.524,"y":476.756},{"x":1018.239,"y":489.178},{"x":1016.718,"y":501.51},{"x":1013.978,"y":513.629},{"x":1010.044,"y":525.416},{"x":1004.956,"y":536.751},{"x":998.765,"y":547.525},{"x":991.532,"y":557.628},{"x":983.329,"y":566.96},{"x":974.238,"y":575.43},{"x":964.347,"y":582.953},{"x":953.758,"y":589.453},{"x":942.573,"y":594.866},{"x":930.907,"y":599.139},{"x":918.871,"y":602.229},{"x":906.588,"y":604.106},{"x":894.179,"y":604.75},{"x":881.768,"y":604.156},{"x":869.479,"y":602.328},{"x":852.556,"y":598.943},{"x":832.249,"y":594.879},{"x":787.575,"y":585.941},{"x":689.29,"y":566.276},{"x":374.756,"y":503.345},{"x":276.471,"y":483.68},{"x":231.796,"y":474.742},{"x":211.49,"y":470.68},{"x":194.567,"y":467.293},{"x":182.272,"y":464.175},{"x":170.359,"y":459.824},{"x":158.948,"y":454.284},{"x":148.16,"y":447.613},{"x":138.106,"y":439.88},{"x":128.889,"y":431.166},{"x":120.607,"y":421.56},{"x":113.343,"y":411.162},{"x":107.173,"y":400.079},{"x":102.163,"y":388.426},{"x":98.362,"y":376.325},{"x":95.811,"y":363.901},{"x":93.205,"y":346.84},{"x":85.387,"y":295.661},{"x":82.78,"y":278.602},{"x":81.528,"y":266.434},{"x":81.476,"y":254.203},{"x":82.622,"y":242.026},{"x":84.956,"y":230.019},{"x":88.456,"y":218.299},{"x":93.088,"y":206.978},{"x":98.806,"y":196.167},{"x":105.558,"y":185.967},{"x":115.778,"y":172.062},{"x":146.438,"y":130.344},{"x":156.659,"y":116.438},{"x":164.508,"y":106.804},{"x":173.276,"y":98.001},{"x":182.877,"y":90.114},{"x":193.217,"y":83.222},{"x":204.191,"y":77.395},{"x":215.69,"y":72.688},{"x":227.601,"y":69.15},{"x":244.369,"y":65.067},{"x":294.672,"y":52.818},{"x":311.439,"y":48.735},{"x":324.259,"y":46.316},{"x":337.261,"y":45.25},{"x":350.305,"y":45.549},{"x":363.244,"y":47.211},{"x":375.939,"y":50.214},{"x":392.502,"y":55.062},{"x":442.191,"y":69.605},{"x":458.753,"y":74.453},{"x":469.046,"y":77.96},{"x":478.994,"y":82.35},{"x":488.52,"y":87.591}],"halfWidth":25.25}`]
]);

console.log('car_racer rotation V5 suite (FIT-V5 TASK F1)');
console.log('-------------------------------------------');

// -----------------------------------------------------------------------------
// F1-1 — the old radial-V2 detector was REMOVED in FIT-V6. Its frozen 495-vector
// corpus and distinctness legs are retired (the detector no longer exists);
// the removal itself is now the asserted contract, mirroring rotation-v6 G2-4.
// -----------------------------------------------------------------------------
await test('F1-1 legacyV2Hash fully removed (Rotation + CommonJS + source)', () => {
  assert(Rotation.legacyV2Hash === undefined, 'Rotation.legacyV2Hash absent');
  assert(rotationMod.legacyV2Hash === undefined, 'CommonJS legacyV2Hash absent');
  const src = readFileSync(ROOT + 'js/rotation.js', 'utf8');
  assert(!/legacyV2Hash/.test(src), 'no legacyV2Hash reference in rotation.js');
  assert(!/legacyGenerateTrackV2/.test(src), 'no legacyGenerateTrackV2 in rotation.js');
});

// -----------------------------------------------------------------------------
// F1-4..6 — V1 frozen (must stay byte-identical).
// -----------------------------------------------------------------------------
await test('F1-4 V1 constants/ladders literally unchanged', () => {
  assert(Rotation.TRACK_POINTS === 14, 'TRACK_POINTS still 14');
  assert(Rotation.generateTrack(0, 1).points.length === 14, 'V1 generator still emits 14 points');
  assert(Rotation.TRACK_MAX_RADIUS === 245 && Rotation.TRACK_FALLBACK_RADIUS === 170 && Rotation.TRACK_MAX_REJECTS === 200, 'V1 geometry constants unchanged');
  assert(Rotation.harmonicsForDifficulty(100) === 6, 'V1 HARMONICS(100) still 6');
  assert(Rotation.amplitudeForDifficulty(100) === 0.42, 'V1 AMPLITUDE(100) still 0.42');
  assert(Rotation.halfWidthForDifficulty(0) === 50 && Rotation.halfWidthForDifficulty(20) === 45 &&
         Rotation.halfWidthForDifficulty(50) === 37.5 && Rotation.halfWidthForDifficulty(100) === 25,
    'V1 HALFWIDTH ladder is the FIT-V6 50 - 0.25 d ladder (global break accepted)');
});
await test('F1-5 V1 golden bytes regenerated under the FIT-V6 ladder: 20-vector corpus at version 1 AND legacy 2-arg', () => {
  for (const [s, r, want] of GOLDEN_V1) {
    assert(Rotation.resolveGeneratedTrack(s, r, 1).trackHash === want, 'V1 hash seed=' + s + ' r=' + r + ' (explicit 1)');
    assert(Rotation.resolveGeneratedTrack(s, r).trackHash === want, 'V1 hash seed=' + s + ' r=' + r + ' (2-arg)');
  }
});
await test('F1-6 V1 full canonical bytes byte-identical for two vectors', () => {
  for (const [key, want] of V1_BYTES) {
    const [s, r] = key.split('|').map(Number);
    const res = Rotation.resolveGeneratedTrack(s, r, 1);
    const got = canonicalJSON({ points: res.controlPoints, halfWidth: res.halfWidth });
    assert(got === want, 'V1 canonical bytes seed=' + s + ' r=' + r + '\n got: ' + got + '\nwant: ' + want);
  }
});

// -----------------------------------------------------------------------------
// F1-7..9 — new template-family V2.
// -----------------------------------------------------------------------------
await test('F1-7 new V2 golden vectors: 20-vector hash corpus at version 2', () => {
  for (const [s, r, want] of GOLDEN_V5) {
    const got = Rotation.resolveGeneratedTrack(s, r, 2);
    assert(got.trackHash === want, 'V2 hash seed=' + s + ' r=' + r + ' - got ' + got.trackHash + ' want ' + want);
  }
});
await test('F1-8 new V2 full canonical bytes byte-identical for two vectors', () => {
  for (const [key, want] of V5_BYTES) {
    const [s, r] = key.split('|').map(Number);
    const res = Rotation.resolveGeneratedTrack(s, r, 2);
    const got = canonicalJSON({ points: res.controlPoints, halfWidth: res.halfWidth });
    assert(got === want, 'V2 canonical bytes seed=' + s + ' r=' + r + '\n got: ' + got + '\nwant: ' + want);
  }
});
await test('F1-9 new V2 resolver contract: ladders, arc-built 91-96 points, hash formula, version gate', () => {
  for (const [s, r] of [[0, 1], [1, 50], [12345, 99], [0xFFFFFFFF, 2]]) {
    const res = Rotation.resolveGeneratedTrack(s, r, 2);
    assert(res.controlPoints.length >= 91 && res.controlPoints.length <= 96,
      'V2 seed=' + s + ' r=' + r + ' - arc-built F1 template point count in [91,96], got ' + res.controlPoints.length);
    assert(res.halfWidth === halfWidthOf(r), 'V2 halfWidth ladder seed=' + s + ' r=' + r);
    assert(res.difficulty === Math.min(r, 100), 'V2 difficulty ladder seed=' + s + ' r=' + r);
    assert(res.trackHash === hashOf(res.controlPoints, res.halfWidth), 'V2 hash formula seed=' + s + ' r=' + r);
    assert(/^[0-9a-f]{16}$/.test(res.trackHash), 'V2 hash is 16 lowercase hex seed=' + s + ' r=' + r);
  }
  const s = 42, r = 3;
  assertValidationError(() => Rotation.resolveGeneratedTrack(s, r, 0), 'version 0 refused');
  assertValidationError(() => Rotation.resolveGeneratedTrack(s, r, 3), 'version 3 refused');
  assertValidationError(() => Rotation.resolveGeneratedTrack(s, r, 1.5), 'version 1.5 refused');
  assertValidationError(() => Rotation.resolveGeneratedTrack(s, r, NaN), 'version NaN refused');
  assertValidationError(() => Rotation.resolveGeneratedTrack(s, r, '2'), 'string version refused');
});

// -----------------------------------------------------------------------------
// F1-10 — start-tangent alignment for all six templates at d∈{0,25,50,75,100}.
// For each (template t, difficulty d) pick r = d (r=0 gives d=0) and seed =
// (imul(r+1,K)^t)>>>0 so the frozen templateIndex formula selects template t.
// -----------------------------------------------------------------------------
await test('F1-10 start-tangent triple pts[n-1],pts[0],pts[1] within 1° for all 6 templates at d∈{0,25,50,75,100}', () => {
  const K = 0x9E3779B1;
  let checked = 0;
  for (let t = 0; t < 6; t++) {
    for (const d of [0, 25, 50, 75, 100]) {
      const r = d;
      const C = Math.imul(r + 1, K) >>> 0;
      const seed = (C ^ t) >>> 0;
      assert(templateIndex(seed, r) === t, 'templateIndex selects t=' + t + ' at d=' + d);
      const pts = Rotation.resolveGeneratedTrack(seed, r, 2).controlPoints;
      const dev = tangentDeviation(pts);
      assert(dev <= 1.0, 'tangent deviation seed=' + seed + ' r=' + r + ' template=' + t + ' d=' + d + ' - got ' + dev.toFixed(6) + '°');
      checked += 1;
    }
  }
  assert(checked === 30, 'checked 30 (template,difficulty) pairs, got ' + checked);
});

// -----------------------------------------------------------------------------
// F1-11 — never throws across >=400 seeds × r=1..99.
// -----------------------------------------------------------------------------
await test('F1-11 resolveGeneratedTrack never throws across 400 seeds × r=1..99 (fallback passes all gates)', () => {
  const srng = refMulberry(0xC0FFEE);
  let calls = 0;
  for (let i = 0; i < 400; i++) {
    const seed = srng() >>> 0;
    for (let r = 1; r <= 99; r++) {
      let res;
      try {
        res = Rotation.resolveGeneratedTrack(seed, r, 2);
      } catch (err) {
        throw new Error('threw for seed=' + seed + ' r=' + r + ': ' + (err && err.message));
      }
      assert(res.controlPoints.length >= 91 && res.controlPoints.length <= 96, 'arc-built F1 template point count seed=' + seed + ' r=' + r);
      for (let k = 0; k < res.controlPoints.length; k++) {
        const p = res.controlPoints[k];
        assert(Number.isFinite(p.x) && Number.isFinite(p.y), 'finite coord seed=' + seed + ' r=' + r + ' k=' + k);
      }
      assert(/^[0-9a-f]{16}$/.test(res.trackHash), 'hash format seed=' + seed + ' r=' + r);
      calls += 1;
    }
  }
  assert(calls === 39600, 'asserted 39600 (seed,r) pairs, got ' + calls);
});

// -----------------------------------------------------------------------------
// F1-12 — distinct hashes across r=1..99.
// -----------------------------------------------------------------------------
await test('F1-12 distinct hashes across r=1..99 for every seed (99/99)', () => {
  for (const s of SEEDS) {
    const seen = new Set();
    for (let r = 1; r <= 99; r++) seen.add(Rotation.resolveGeneratedTrack(s, r, 2).trackHash);
    assert(seen.size === 99, 'distinct hashes seed=' + s + ' - got ' + seen.size + '/99');
  }
});

// -----------------------------------------------------------------------------
// F1-13 — shape properties (straights, U-turns, wall-safe, in-bounds).
// -----------------------------------------------------------------------------
await test('F1-13 shape properties: near-straight >120px, ≥1 U-turn, wall-safe, in-bounds', () => {
  let n = 0;
  for (const s of SEEDS) {
    for (const r of RS) {
      const res = Rotation.resolveGeneratedTrack(s, r, 2);
      const pts = res.controlPoints;
      const hw = res.halfWidth;
      for (const p of pts) {
        assert(p.x >= 0 && p.x <= 1100, 'x in canvas seed=' + s + ' r=' + r + ': ' + p.x);
        assert(p.y >= 0 && p.y <= 650, 'y in canvas seed=' + s + ' r=' + r + ': ' + p.y);
      }
      assert(longestStraight(pts) > 120, 'near-straight >120px seed=' + s + ' r=' + r + ' - got ' + longestStraight(pts).toFixed(1));
      assert(uturnCount(pts) >= 1, '≥1 U-turn seed=' + s + ' r=' + r + ' - got ' + uturnCount(pts));
      let track;
      try { track = new Track(pts, hw); } catch (err) {
        throw new Error('Track build failed (wall-safe) seed=' + s + ' r=' + r + ': ' + (err && err.message));
      }
      assert(track.segments.length > 0, 'wall segments exist seed=' + s + ' r=' + r);
      assert(wallsSound(track.leftWall), 'left wall sound seed=' + s + ' r=' + r);
      assert(wallsSound(track.rightWall), 'right wall sound seed=' + s + ' r=' + r);
      n += 1;
    }
  }
  assert(n === 20, 'checked 20 shape vectors, got ' + n);
});

// -----------------------------------------------------------------------------
// F1-14 — worst-case timing bound.
// -----------------------------------------------------------------------------
await test('F1-14 worst-case timing bound: no single resolve call exceeds 100ms', () => {
  const srng = refMulberry(0x1234ABCD);
  let maxMs = 0;
  for (let i = 0; i < 120; i++) {
    const seed = srng() >>> 0;
    const r = 1 + (srng() % 99);
    const t0 = Date.now();
    Rotation.resolveGeneratedTrack(seed, r, 2);
    const dt = Date.now() - t0;
    if (dt > maxMs) maxMs = dt;
  }
  assert(maxMs <= 100, 'max single-call time <= 100ms, got ' + maxMs + 'ms');
});

// -----------------------------------------------------------------------------
// F1-15 — no new version dispatch / parser drift.
// -----------------------------------------------------------------------------
await test('F1-15 parsers + generator dispatch unchanged (no inline version checks)', () => {
  for (const [raw, want] of [[undefined, 1], [null, 1], ['1', 1], [1, 1], ['2', 2], [2, 2]]) {
    assert(Rotation.parseGeneratorVersion(raw) === want, 'parseGeneratorVersion(' + String(raw) + ') === ' + want);
  }
  for (const raw of ['', ' ', '3', 0, 3, -1, 1.5, NaN, Infinity, true, false, {}, [], 'abc']) {
    assertValidationError(() => Rotation.parseGeneratorVersion(raw), 'parseGeneratorVersion rejects ' + JSON.stringify(raw));
  }
  assertValidationError(() => Rotation.requireGeneratorVersion(undefined), 'require rejects undefined');
  assertValidationError(() => Rotation.requireGeneratorVersion(null), 'require rejects null');
  assertValidationError(() => Rotation.requireGeneratorVersion(''), 'require rejects empty');
  assert(Rotation.requireGeneratorVersion('1') === 1 && Rotation.requireGeneratorVersion(2) === 2, 'require delegates');
  const src = readFileSync(ROOT + 'js/rotation.js', 'utf8');
  const genTables = (src.match(/var GENERATORS_BY_VERSION\s*=/g) || []).length;
  assert(genTables === 1, 'exactly one generator dispatch table, got ' + genTables);
  const inlineVersionEq2 = (src.match(/version\s*===\s*2/g) || []).length;
  assert(inlineVersionEq2 === 0, 'no inline `version === 2` check added, got ' + inlineVersionEq2);
  const rawEq2 = (src.match(/raw\s*===\s*2/g) || []).length;
  assert(rawEq2 === 1, 'exactly one parser `raw === 2` check, got ' + rawEq2);
  const jsFiles = readdirSync(ROOT + 'js').filter((f) => f.endsWith('.js'));
  let callSites = 0;
  for (const f of jsFiles) {
    if (f === 'rotation.js') continue;
    const text = readFileSync(ROOT + 'js/' + f, 'utf8');
    const re = /(?:Rotation|R)\.resolveGeneratedTrack\s*\(/g;
    let m;
    while ((m = re.exec(text))) callSites += 1;
  }
  assert(callSites === 5, 'exactly five external resolveGeneratedTrack call sites, got ' + callSites);
});

// -----------------------------------------------------------------------------
console.log('-------------------------------------------');
console.log('rotation-v5: ' + passed + ' passed, ' + failures.length + ' failed');
if (failures.length) process.exit(1);
