// tests/db-v6.mjs — FIT-V6 TASK G3: schema v4 + typed WallModelChangedError.
//
// Plain Node, no npm deps, vendored sql.js via Db.loadNode — the same harness
// shape as tests/db-v5.mjs / tests/smoke.mjs. Run with:
//   node tests/db-v6.mjs
//   node --check tests/db-v6.mjs
//
// Contract under test (TASK G3 handoff / plan §SCHEMA v4):
//   * Fresh init: user_version=4, meta.schema_version='4', meta.wall_model='v6'.
//   * Fresh -> export -> re-adopt (battery / init / openExisting) at uv4.
//   * Pre-v4 bytes (uv∈{1,2,3}) refused with a typed WallModelChangedError on
//     EVERY ingress: validateAdoptionBytes, init, openExisting, importBytes,
//     cache-offer (restoreFromLastKnownGood), persist step-2 verify.
//   * uv∉{1,2,3,4} refused with a GENERIC ValidationError (not the typed one).
//   * Refusal is byte-preserving: err.refusedBytes === original bytes; the
//     working DB is untouched; db-session does NOT clear the handle / LSG.
//   * No stale FIT-V5 symbols in db.js / main.js / db-session.js.

import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const VENDOR_DIR = new URL('../vendor/', import.meta.url).pathname;
const JS_DIR = new URL('../js/', import.meta.url).pathname;

// Load order mirrors index.html: nn.js -> track.js -> rotation.js -> db.js.
require('../js/nn.js');
const { Track, DEFAULT_CONTROL_POINTS, canonicalJSON, cyrb53 } = require('../js/track.js');
require('../js/rotation.js');
const requireDb = () => require('../js/db.js');

const { Db } = requireDb();
await Db.loadNode(VENDOR_DIR);

const SEED = 123456789;
const defaultTrack = new Track(DEFAULT_CONTROL_POINTS, 70);
const DEFAULT_TRACK_JSON = defaultTrack.toJSON();
const DEFAULT_TRACK_HASH = defaultTrack.hash();
const DEFAULT_POINTS_JSON = JSON.stringify(
  DEFAULT_CONTROL_POINTS.map((p) => ({ x: Number(p.x.toFixed(9)), y: Number(p.y.toFixed(9)) }))
);

const WALL_MODEL_MESSAGE_EXPECTED = /wall model/i;
const BANNER_MARK = 'WALL MODEL CHANGED';

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
function sameBytes(a, b) {
  if (!(a instanceof Uint8Array) || !(b instanceof Uint8Array)) return false;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}
function metaValue(db, key) {
  const rows = db.db.exec('SELECT value FROM meta WHERE key = ?', [key]);
  return rows && rows[0] && rows[0].values.length ? rows[0].values[0][0] : null;
}
function uvOf(db) {
  const rows = db.db.exec('SELECT COALESCE(user_version,0) FROM pragma_user_version');
  return rows && rows[0] && rows[0].values.length ? Number(rows[0].values[0][0]) : null;
}

// Build a structurally valid five-table SQLite lineage with controllable uv /
// schema_version / wall_model. No generations (empty lineage is legal), one
// default track row (satisfies the tracks content/hash arms).
function buildBytes(opts) {
  opts = opts || {};
  const uv = opts.uv === undefined ? 4 : opts.uv;
  const schemaVersion = opts.schemaVersion === undefined ? String(uv) : opts.schemaVersion;
  const SQL = globalThis.SQL;
  const {
    DDL_META, DDL_TRACKS, DDL_GENERATIONS, DDL_DRIVER_RESULTS, DDL_BEST_WEIGHTS,
  } = requireDb();
  const raw = new SQL.Database();
  raw.run('PRAGMA foreign_keys = ON');
  [DDL_META, DDL_TRACKS, DDL_GENERATIONS, DDL_DRIVER_RESULTS, DDL_BEST_WEIGHTS]
    .forEach((ddl) => raw.run(ddl));
  raw.run('PRAGMA user_version = ' + Number(uv));
  raw.run("INSERT INTO meta(key, value) VALUES ('schema_version', ?)", [String(schemaVersion)]);
  raw.run("INSERT INTO meta(key, value) VALUES ('architecture', '7-16-16-2')");
  raw.run("INSERT INTO meta(key, value) VALUES ('app_version', '1.1.0')");
  raw.run("INSERT INTO meta(key, value) VALUES ('created_at', '2026-01-01 00:00:00')");
  if (opts.wallModel !== undefined && opts.wallModel !== null) {
    raw.run("INSERT INTO meta(key, value) VALUES ('wall_model', ?)", [String(opts.wallModel)]);
  }
  raw.run('INSERT INTO meta(key, value) VALUES (?, ?)', ['track_hash', DEFAULT_TRACK_HASH]);
  raw.run('INSERT INTO meta(key, value) VALUES (?, ?)', ['rng_state', String(SEED)]);
  raw.run(
    'INSERT INTO tracks(track_hash, source, control_points, half_width, difficulty, created_at) VALUES (?,?,?,?,?,?)',
    [DEFAULT_TRACK_HASH, 'default', DEFAULT_POINTS_JSON, 70, 0, '2026-01-01 00:00:00']
  );
  const bytes = raw.export();
  raw.close();
  return bytes;
}

function preV4(over) {
  return buildBytes(Object.assign({ uv: 3, schemaVersion: '3' }, over || {}));
}

// ===========================================================================
console.log('FIT-V6 TASK G3 — schema v4 + typed wall-model refusal');
// ===========================================================================

await leg('WallModelChangedError is a typed ValidationError (global + exports) with actionable message', async () => {
  const Ctor = globalThis.WallModelChangedError;
  const ModCtor = requireDb().WallModelChangedError;
  assert(typeof Ctor === 'function', 'globalThis.WallModelChangedError must be installed');
  assert(ModCtor === Ctor, 'module.exports.WallModelChangedError must equal globalThis copy');
  const err = new Ctor();
  expectTyped(err, 'WallModelChangedError');
  assert(err instanceof globalThis.ValidationError, 'WallModelChangedError must extend ValidationError');
  assert(WALL_MODEL_MESSAGE_EXPECTED.test(err.message), 'message must name the wall model: ' + err.message);
  assert(/fresh lineage/i.test(err.message), 'message must instruct a fresh lineage: ' + err.message);
  assert(/untouched/i.test(err.message), 'message must state the original was untouched: ' + err.message);
  assert(requireDb().WALL_MODEL_CHANGED_MESSAGE === err.message, 'WALL_MODEL_CHANGED_MESSAGE must be exported and equal the default message');
});

await leg('fresh init seeds uv=4, schema_version=4, wall_model=v6 (reuses v3 DDL twins)', async () => {
  const db = new Db();
  await db.init();
  assert(uvOf(db) === 4, 'fresh user_version must be 4, got ' + uvOf(db));
  assert(metaValue(db, 'schema_version') === '4', "fresh meta.schema_version must be '4'");
  assert(metaValue(db, 'wall_model') === 'v6', "fresh meta.wall_model must be 'v6'");
  db.db.close();
});

await leg('fresh -> export -> re-adopt at uv4 (battery, init, openExisting)', async () => {
  const db = new Db();
  await db.init();
  const bytes = db.exportBytes();
  db.db.close();
  const summary = Db.validateAdoptionBytes(bytes);
  assert(summary && summary.generation === 0, 'uv4 fresh export must pass the battery');

  const db2 = new Db();
  await db2.init(bytes);
  assert(uvOf(db2) === 4, 'init(bytes) must adopt uv4');
  assert(metaValue(db2, 'wall_model') === 'v6', 'adopted meta.wall_model must be v6');
  db2.db.close();

  const db3 = new Db();
  const opened = await db3.openExisting({ bytes });
  assert(opened && opened.ok === true, 'openExisting must adopt uv4');
  assert(uvOf(db3) === 4, 'openExisting working DB must be uv4');
  db3.db.close();
});

await leg('battery: uv∈{1,2,3} typed WallModelChangedError with retrievable refused bytes', async () => {
  for (const uv of [1, 2, 3]) {
    const bytes = preV4({ uv, schemaVersion: String(uv) });
    const before = new Uint8Array(bytes);
    let thrown = null;
    try { Db.validateAdoptionBytes(bytes); } catch (err) { thrown = err; }
    expectTyped(thrown, 'WallModelChangedError');
    assert(sameBytes(thrown.refusedBytes, before), 'uv=' + uv + ' typed error must carry the refused bytes verbatim');
    assert(sameBytes(bytes, before), 'uv=' + uv + ' battery must not mutate the bytes');
  }
});

await leg('battery: uv∉{1,2,3,4} generic ValidationError (NOT the typed wall error)', async () => {
  const bytes = preV4({ uv: 999, schemaVersion: '999' });
  let thrown = null;
  try { Db.validateAdoptionBytes(bytes); } catch (err) { thrown = err; }
  assert(thrown !== null, 'uv=999 must be refused');
  assert(thrown.name === 'ValidationError', 'uv=999 must be generic ValidationError, got ' + thrown.name);
});

await leg('uv4 with wrong wall_model is refused (generic), uv4 with wrong schema_version is refused', async () => {
  const wrongWall = buildBytes({ uv: 4, schemaVersion: '4', wallModel: 'v5' });
  let t1 = null;
  try { Db.validateAdoptionBytes(wrongWall); } catch (err) { t1 = err; }
  assert(t1 && t1.name === 'ValidationError', 'uv4 wrong wall_model must refuse');

  const wrongSchema = buildBytes({ uv: 4, schemaVersion: '3', wallModel: 'v6' });
  let t2 = null;
  try { Db.validateAdoptionBytes(wrongSchema); } catch (err) { t2 = err; }
  assert(t2 && t2.name === 'ValidationError', 'uv4 wrong schema_version must refuse');
});

await leg('init(bytes) refuses pre-v4 typed, zero adoption, byte-identical, bytes retrievable', async () => {
  const bytes = preV4();
  const before = new Uint8Array(bytes);
  const db = new Db();
  let thrown = null;
  try { await db.init(bytes); } catch (err) { thrown = err; }
  expectTyped(thrown, 'WallModelChangedError');
  assert(db.db === null || db.db === undefined, 'working DB must remain unopened after refusal');
  assert(sameBytes(thrown.refusedBytes, before), 'refused bytes must be retrievable from the error');
  assert(sameBytes(bytes, before), 'pre-v4 file must be byte-identical after refusal');
});

await leg('openExisting({bytes}) refuses pre-v4 typed, bytes retrievable and untouched', async () => {
  const bytes = preV4();
  const before = new Uint8Array(bytes);
  const db = new Db();
  let thrown = null;
  try { await db.openExisting({ bytes }); } catch (err) { thrown = err; }
  expectTyped(thrown, 'WallModelChangedError');
  assert(sameBytes(thrown.refusedBytes, before), 'refused bytes retrievable from openExisting error');
  assert(sameBytes(bytes, before), 'file byte-identical after openExisting refusal');
});

await leg('importBytes refuses pre-v4 typed, confirm NOT consulted, bytes retrievable', async () => {
  const bytes = preV4();
  const before = new Uint8Array(bytes);
  const db = new Db();
  await db.init();
  let confirmCalls = 0;
  let thrown = null;
  try {
    await db.importBytes(bytes, { confirm: () => { confirmCalls += 1; return true; } });
  } catch (err) { thrown = err; }
  expectTyped(thrown, 'WallModelChangedError');
  assert(confirmCalls === 0, 'confirm callback must NOT be consulted on refusal');
  assert(sameBytes(thrown.refusedBytes, before), 'refused bytes retrievable from importBytes error');
  assert(sameBytes(bytes, before), 'file byte-identical after importBytes refusal');
  db.db.close();
});

await leg('persist step-2 _verifyForPersist refuses pre-v4 typed with retrievable bytes', async () => {
  const bytes = preV4();
  const before = new Uint8Array(bytes);
  const db = new Db();
  await db.init();
  let thrown = null;
  try { db._verifyForPersist(bytes); } catch (err) { thrown = err; }
  expectTyped(thrown, 'WallModelChangedError');
  assert(sameBytes(thrown.refusedBytes, before), 'refused bytes retrievable from persist verify error');
  assert(sameBytes(bytes, before), 'file byte-identical after persist verify refusal');
  // And a valid uv4 export still verifies.
  const good = db.exportBytes();
  const summary = db._verifyForPersist(good);
  assert(summary && typeof summary.maxGeneration === 'number', 'uv4 export must pass persist verify');
  db.db.close();
});

await leg('cache-offer restoreFromLastKnownGood refuses pre-v4 typed and exposes refused bytes', async () => {
  const bytes = preV4();
  const before = new Uint8Array(bytes);
  let res;
  try {
    res = await Db.restoreFromLastKnownGood({ read: async () => bytes });
  } catch (err) {
    assert(false, 'restoreFromLastKnownGood must not throw; got ' + err.name + ': ' + err.message);
  }
  assert(res && res.ok === false, 'cache offer must refuse, not adopt');
  assert(res.reason === 'validation', 'cache refusal reason must be validation');
  assert(String(res.detail || '').indexOf('wall model') !== -1 ||
         String(res.detail || '').indexOf(BANNER_MARK) !== -1,
    'cache refusal detail must carry the wall-model message');
  assert(sameBytes(res.refusedBytes, before), 'cache offer must expose the refused bytes verbatim');
  assert(sameBytes(bytes, before), 'refused cache bytes must be byte-identical');
});

await leg('_migrateIfNeeded is a verified no-op at uv4 and refuses uv<4', async () => {
  const db = new Db();
  await db.init();
  const res = db._migrateIfNeeded(db.db);
  assert(res && res.migrated === false && res.userVersion === 4, 'uv4 migration gate must be a no-op: ' + JSON.stringify(res));

  const old = new SQL.Database();
  old.run('PRAGMA user_version = 3');
  let thrown = null;
  try { db._migrateIfNeeded(old); } catch (err) { thrown = err; }
  assert(thrown && thrown.name === 'MigrationError', 'uv<4 must be refused at the migration gate');
  old.close();
  db.db.close();
});

await leg('_assertMigratedState accepts uv4 with wall_model v6', async () => {
  const db = new Db();
  await db.init();
  const res = db._assertMigratedState(db.db, 'test');
  assert(res && res.ok === true, 'uv4 verified set must pass');
  db.db.close();
});

// --- db-session byte-preservation (typed refusal keeps handle + LSG) -------
await leg('db-session typed refusal preserves handle + LSG and offers verbatim download', async () => {
  // Minimal fake IDB that records delete operations on the 'handles' store.
  const deletes = [];
  function fakeStore() {
    return {
      get: (key) => mkReq(handleLike),
      put: () => mkReq(undefined),
      delete: (key) => { deletes.push(key); return mkReq(undefined); },
    };
  }
  function mkReq(value) {
    const req = {};
    // defer resolution to microtask so await works
    queueMicrotask(() => { req.result = value; if (req.onsuccess) req.onsuccess(); });
    return req;
  }
  const handleLike = {
    name: 'lineage.sqlite',
    queryPermission: async () => 'granted',
    getFile: async () => ({ size: 1, arrayBuffer: async () => new Uint8Array([0]).buffer }),
  };
  globalThis.indexedDB = {
    open: () => {
      const req = {};
      queueMicrotask(() => {
        req.result = {
          objectStoreNames: { contains: () => true },
          objectStore: fakeStore,
          transaction: () => ({ objectStore: fakeStore }),
          close: () => {},
        };
        if (req.onsuccess) req.onsuccess();
      });
      return req;
    },
  };

  const bytes = preV4();
  let lsgClears = 0;
  const typedErr = new globalThis.WallModelChangedError();
  typedErr.refusedBytes = new Uint8Array(bytes);
  const stubDb = {
    openExisting: async () => { throw typedErr; },
    init: async () => {},
  };
  const prevDb = globalThis.Db;
  globalThis.Db = Object.assign(function () {}, prevDb, {
    restoreFromLastKnownGood: async () => ({
      ok: false, reason: 'validation',
      detail: typedErr.message,
      refusedBytes: new Uint8Array(bytes),
    }),
    clearLastKnownGood: async () => { lsgClears += 1; },
  });

  const requireDbSession = () => require('../js/db-session.js');
  const { DbSession } = requireDbSession();
  let result;
  try {
    result = await DbSession.restoreIfPossible(stubDb);
  } finally {
    globalThis.Db = prevDb;
    delete globalThis.indexedDB;
  }
  assert(result && typeof result === 'object', 'restoreIfPossible must resolve a result');
  assert(String(result.message || '').indexOf(BANNER_MARK) !== -1,
    'typed refusal banner must carry the wall-model marker: ' + result.message);
  assert(String(result.message || '').indexOf('untouched') !== -1,
    'typed refusal banner must state the original was untouched');
  assert(deletes.length === 0, 'typed refusal must NOT clear the handle, got ' + deletes.length + ' deletes');
  assert(lsgClears === 0, 'typed refusal must NOT clear the LSG slot, got ' + lsgClears + ' clears');
  assert(sameBytes(result.refusedBytes, bytes), 'typed refusal must expose refused bytes for verbatim download');
});

await leg('no stale FIT-V5 symbols in db.js / main.js / db-session.js', async () => {
  for (const file of ['db.js', 'main.js', 'db-session.js']) {
    const src = readFileSync(JS_DIR + file, 'utf8');
    for (const sym of ['V2GeneratorChangedError', 'V2_GENERATOR_CHANGED_MESSAGE', 'legacyV2Hash']) {
      assert(src.indexOf(sym) === -1, file + ' must not reference ' + sym);
    }
  }
  const mod = requireDb();
  assert(mod.V2GeneratorChangedError === undefined, 'db.js must not export V2GeneratorChangedError');
  assert(globalThis.V2GeneratorChangedError === undefined, 'globalThis must not carry V2GeneratorChangedError');
});

await leg('fresh init seeds an empty tracks table with the v3 schema twin (no DDL drift)', async () => {
  const db = new Db();
  await db.init();
  const { DDL_TRACKS, DDL_GENERATIONS } = requireDb();
  const norm = (s) => s.toLowerCase().replace(/\s+/g, '').replace(/;+$/, '').replace(/"/g, '');
  const rows = db.db.exec("SELECT sql FROM sqlite_master WHERE type='table' AND name='tracks'");
  assert(norm(String(rows[0].values[0][0])) === norm(DDL_TRACKS), 'tracks DDL twin drift');
  const grows = db.db.exec("SELECT sql FROM sqlite_master WHERE type='table' AND name='generations'");
  assert(norm(String(grows[0].values[0][0])) === norm(DDL_GENERATIONS), 'generations DDL twin drift');
  db.db.close();
});

// ---------------------------------------------------------------------------
console.log('');
console.log('legs passed: ' + legsPassed + ' / ' + (legsPassed + failures.length));
if (failures.length) {
  console.log('FAILURES:');
  for (const f of failures) console.log('  - ' + f.name + ': ' + String((f.err && f.err.message) || f.err));
  process.exitCode = 1;
} else {
  console.log('ALL GREEN');
}
