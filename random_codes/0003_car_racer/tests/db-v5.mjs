// tests/db-v5.mjs — FIT-V5 TASK F2, migrated to the FIT-V6 shipped contract.
//
// Plain Node, no npm deps, vendored sql.js via Db.loadNode — the same harness
// shape as tests/db-v5.mjs's predecessor and tests/db-v6.mjs. Run with:
//   node tests/db-v5.mjs
//   node --check tests/db-v5.mjs
//
// FIT-V6 migration (G4): V2GeneratorChangedError (the old-generator typed
// refusal) was REMOVED. The typed wall-model break is now carried by
// WallModelChangedError, whose full every-ingress / byte-preserving contract is
// owned by tests/db-v6.mjs. This suite keeps:
//   * the removal assertion for V2GeneratorChangedError everywhere;
//   * a compact cross-check that pre-v4 bytes are refused by the typed
//     WallModelChangedError on every ingress (no false accept);
//   * the still-live strictUint32 rotation_seed arm on uv4 V2-pinned bytes;
//   * the no-false-positive arms (valid uv4 V1/V2 files adopt normally).
// The obsolete "missing rotation_seed is refused" leg is retired: on uv4 the
// battery deliberately TOLERATES a missing seed on a V2-pinned file (the
// curriculum-row arm simply has no pin to verify), while a malformed seed is
// still refused as a typed ValidationError. Probe: 2026-11-19, this fixture.

import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const ROOT = fileURLToPath(new URL('..', import.meta.url));
const VENDOR_DIR = new URL('../vendor/', import.meta.url).pathname;

// Load order: track.js installs globalThis.canonicalJSON / cyrb53; rotation.js
// installs globalThis.Rotation. nn.js installs ValidationError/strictUint32.
require('../js/nn.js');
const { Track, DEFAULT_CONTROL_POINTS } = require('../js/track.js');
require('../js/rotation.js');
const requireDb = () => require('../js/db.js');
const { Rotation } = require('../js/rotation.js');
const { Db } = requireDb();
await Db.loadNode(VENDOR_DIR);

const SEED = 123456789;
const defaultTrack = new Track(DEFAULT_CONTROL_POINTS, 70);
const DEFAULT_TRACK_JSON = defaultTrack.toJSON();
const DEFAULT_TRACK_HASH = defaultTrack.hash();
const DEFAULT_POINTS_JSON = JSON.stringify(DEFAULT_CONTROL_POINTS.map((p) => ({ x: Number(p.x.toFixed(9)), y: Number(p.y.toFixed(9)) })));

const WALL_MESSAGE = requireDb().WALL_MODEL_CHANGED_MESSAGE;

let legsPassed = 0;
const failures = [];
async function leg(name, fn) {
  try {
    await fn();
    legsPassed += 1;
    console.log('  ok   — ' + name);
  } catch (err) {
    failures.push({ name, err });
    const detail = String((err && err.stack) || err).split('\n').map((line) => '         ' + line).join('\n');
    console.log('  FAIL — ' + name + '\n' + detail);
  }
}
function assert(condition, message) {
  if (!condition) throw new Error('assert failed: ' + message);
}
function expectTyped(err, name) {
  assert(err !== null && err !== undefined, (name || 'call') + ' — expected an error, nothing thrown');
  assert(err.name === name, (name || 'call') + ' — expected name ' + name + ', got ' + err.name + ': ' + err.message);
  return err;
}
function bytesCopy(bytes) { return new Uint8Array(bytes); }
function sameBytes(a, b) {
  if (!(a instanceof Uint8Array) || !(b instanceof Uint8Array)) return false;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// Structurally valid uv4 five-table lineage. opts.generatorVersion ('1'/'2'/
// absent), opts.seed (rotation_seed value; undefined ⇒ omit the key), and
// opts.uv (default 4; 3 ⇒ pre-v4 typed-refusal fixture) are controllable.
function buildV4Bytes(opts) {
  opts = opts || {};
  const SQL = globalThis.SQL;
  const { DDL_META, DDL_TRACKS, DDL_GENERATIONS, DDL_DRIVER_RESULTS, DDL_BEST_WEIGHTS } = requireDb();
  const uv = opts.uv === undefined ? 4 : opts.uv;
  const raw = new SQL.Database();
  raw.run('PRAGMA foreign_keys = ON');
  [DDL_META, DDL_TRACKS, DDL_GENERATIONS, DDL_DRIVER_RESULTS, DDL_BEST_WEIGHTS].forEach((ddl) => raw.run(ddl));
  raw.run('PRAGMA user_version = ' + uv);
  raw.run("INSERT INTO meta(key, value) VALUES ('schema_version', ?)", [String(uv)]);
  raw.run("INSERT INTO meta(key, value) VALUES ('wall_model', 'v6')");
  raw.run("INSERT INTO meta(key, value) VALUES ('architecture', '7-16-16-2')");
  raw.run("INSERT INTO meta(key, value) VALUES ('app_version', '1.1.0')");
  raw.run("INSERT INTO meta(key, value) VALUES ('created_at', '2026-01-01 00:00:00')");
  raw.run('INSERT INTO meta(key, value) VALUES (?, ?)', ['track_hash', DEFAULT_TRACK_HASH]);
  raw.run('INSERT INTO meta(key, value) VALUES (?, ?)', ['rng_state', String(SEED)]);
  if (opts.generatorVersion !== undefined) {
    raw.run('INSERT INTO meta(key, value) VALUES (?, ?)', ['generator_version', String(opts.generatorVersion)]);
  }
  if (opts.seed !== undefined) {
    raw.run('INSERT INTO meta(key, value) VALUES (?, ?)', ['rotation_seed', String(opts.seed)]);
  }
  raw.run('INSERT INTO tracks(track_hash, source, control_points, half_width, difficulty, created_at) VALUES (?,?,?,?,?,?)',
    [DEFAULT_TRACK_HASH, 'default', DEFAULT_POINTS_JSON, 70, 0, '2026-01-01 00:00:00']);
  if (opts.generatorVersion === '2') {
    for (let r = 1; r <= 99; r++) {
      const res = Rotation.resolveGeneratedTrack(SEED, r, 2);
      raw.run('INSERT INTO tracks(track_hash, source, control_points, half_width, difficulty, created_at) VALUES (?,?,?,?,?,?)',
        [res.trackHash, 'generated', JSON.stringify(res.controlPoints), res.halfWidth, r, '2026-01-01 00:00:00']);
    }
  }
  const bytes = raw.export();
  raw.close();
  return bytes;
}

// ===========================================================================
console.log('FIT-V6 — db-v5 migrated: old-generator refusal retired, typed wall-model contract lives in db-v6');
// ===========================================================================

await leg('V2GeneratorChangedError fully removed; WallModelChangedError is the typed replacement', async () => {
  assert(globalThis.V2GeneratorChangedError === undefined, 'globalThis.V2GeneratorChangedError absent');
  assert(requireDb().V2GeneratorChangedError === undefined, 'db.js must not export V2GeneratorChangedError');
  for (const file of ['db.js', 'main.js', 'db-session.js']) {
    const src = readFileSync(ROOT + 'js/' + file, 'utf8');
    for (const sym of ['V2GeneratorChangedError', 'V2_GENERATOR_CHANGED_MESSAGE']) {
      assert(src.indexOf(sym) === -1, file + ' must not reference ' + sym);
    }
  }
  const Ctor = globalThis.WallModelChangedError;
  assert(typeof Ctor === 'function', 'WallModelChangedError installed');
  assert(requireDb().WallModelChangedError === Ctor, 'module export === globalThis WallModelChangedError');
  assert(new Ctor() instanceof globalThis.ValidationError, 'WallModelChangedError extends ValidationError');
  assert(new Ctor().message === WALL_MESSAGE, 'WALL_MODEL_CHANGED_MESSAGE exported and default');
});

await leg('pre-v4 bytes refused by typed WallModelChangedError on every ingress (byte-preserving)', async () => {
  const bytes = buildV4Bytes({ uv: 3 });
  const before = bytesCopy(bytes);
  let thrown = null;
  try { Db.validateAdoptionBytes(bytes); } catch (err) { thrown = err; }
  expectTyped(thrown, 'WallModelChangedError');
  assert(sameBytes(thrown.refusedBytes, before), 'battery error carries refused bytes verbatim');

  const db = new Db();
  thrown = null;
  try { await db.init(bytes); } catch (err) { thrown = err; }
  expectTyped(thrown, 'WallModelChangedError');
  assert(db.db === null || db.db === undefined, 'init: zero adoption');
  assert(sameBytes(bytes, before), 'init: file byte-identical');

  thrown = null;
  try { await db.openExisting({ bytes }); } catch (err) { thrown = err; }
  expectTyped(thrown, 'WallModelChangedError');

  const db2 = new Db();
  await db2.init();
  let confirms = 0;
  thrown = null;
  try { await db2.importBytes(bytes, { confirm: () => { confirms++; return true; } }); } catch (err) { thrown = err; }
  expectTyped(thrown, 'WallModelChangedError');
  assert(confirms === 0, 'importBytes: confirm not consulted on refusal');
  db2.db.close();

  const offer = await Db.restoreFromLastKnownGood({ read: async () => bytes });
  assert(offer && offer.ok === false && offer.reason === 'validation', 'cache offer refuses with validation reason');
  assert(sameBytes(offer.refusedBytes, before), 'cache offer exposes refused bytes verbatim');
  assert(sameBytes(bytes, before), 'file byte-identical after all refusals');
});

await leg('malformed rotation_seed on a V2-pinned uv4 lineage is refused with a typed ValidationError', async () => {
  const bytes = buildV4Bytes({ generatorVersion: '2', seed: 'not-a-uint32' });
  const before = bytesCopy(bytes);
  let thrown = null;
  try { Db.validateAdoptionBytes(bytes); } catch (err) { thrown = err; }
  assert(thrown !== null, 'malformed seed — expected a typed refusal, got a pass');
  assert(thrown.name === 'ValidationError', 'malformed seed — expected ValidationError, got ' + thrown.name);
  assert(/rotation_seed/i.test(thrown.message), 'malformed seed — message must name rotation_seed: ' + thrown.message);
  assert(sameBytes(bytes, before), 'malformed seed refusal is read-only');
});

await leg('a valid V2-pinned uv4 file adopts normally — no false positive', async () => {
  const bytes = buildV4Bytes({ generatorVersion: '2', seed: SEED });
  const summary = Db.validateAdoptionBytes(bytes);
  assert(summary && summary.generation === 0, 'V2-pinned uv4 lineage must pass the battery unchanged');
  const db = new Db();
  await db.init(bytes);
  assert(db.resumeState().generatorVersion === 2, 'adopted V2-pinned lineage reports generatorVersion 2');
  db.db.close();
});

await leg('a valid unpinned uv4 file adopts normally — no false positive', async () => {
  const db = await (async () => { const d = new Db(); await d.init(); return d; })();
  const bytes = db.exportBytes();
  db.db.close();
  const summary = Db.validateAdoptionBytes(bytes);
  assert(summary && summary.generation === 0, 'fresh uv4 lineage must pass the battery unchanged');
  const db2 = new Db();
  await db2.init(bytes);
  assert(db2.resumeState().generatorVersion === 1, 'absent generator_version parses to V1');
  db2.db.close();
});

// ---------------------------------------------------------------------------
console.log('\nFIT-V6 db-v5 result: ' + legsPassed + ' passed, ' + failures.length + ' failed');
if (failures.length) {
  for (const f of failures) console.log('  FAILED: ' + f.name + ' — ' + ((f.err && f.err.message) || f.err));
  process.exitCode = 1;
}
