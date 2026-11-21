// tests/db-version.mjs — FIT-V4 TASK B: the DB generator-version contract
// (migrated to the FIT-V6 shipped contract; pre-v4 bytes are typed-refused).
//
// Plain Node, no npm deps, vendored sql.js via Db.loadNode — exactly the
// tests/smoke.mjs style: createRequire for the guarded classic-script exports,
// a tiny leg harness that never throws, loud failures, exit 1 on any failure.
//
// Run with:  node tests/db-version.mjs
// Also:      node --check tests/db-version.mjs
//
// Contract under test (red-team approved, cycle 4):
//   * Db re-exports the ONE canonical parser from rotation.js (alias).
//   * ensureLineage/forkLineage write meta.generator_version='2' ONCE, in the
//     same transaction BEFORE preMaterializeCurriculum, and RETURN the version.
//   * preMaterializeCurriculum(db, seed, context, versionOverride): override
//     fully replaces the db-derived version; post-invariant scoped to CURRENT
//     rotation_seed; 19,800-attempt ceiling.
//   * Migration (single call) forces V1 via override=1 AND deletes any stray
//     generator_version key → crafted v2 file with '2' key is adopted,
//     normalized to V1, crosses gen 51 with the V1 hash and zero errors.
//   * Allowlist: absent / 1 / 2 only; ''/'3'/whitespace/over-length → ValidationError.

import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const REPO_ROOT = fileURLToPath(new URL('..', import.meta.url));
const VENDOR_DIR = new URL('../vendor/', import.meta.url).pathname;

// Load order matters: track.js installs globalThis.canonicalJSON / cyrb53,
// which rotation.js's resolver uses; rotation.js installs globalThis.Rotation.
const { NeuralNetwork, mulberry32 } = require('../js/nn.js');
const { Track, DEFAULT_CONTROL_POINTS, canonicalJSON, cyrb53 } = require('../js/track.js');
require('../js/rotation.js');
const requireDb = () => require('../js/db.js');
const requireRotation = () => require('../js/rotation.js');

const POPULATION = 20;
const MUTATION_RATE = 0.25;
const MUTATION_SCALE = 0.30;
const SEED = 123456789;

const defaultTrack = new Track(DEFAULT_CONTROL_POINTS, 70);
const DEFAULT_TRACK_JSON = defaultTrack.toJSON();
const DEFAULT_TRACK_HASH = defaultTrack.hash();
const DEFAULT_POINTS_JSON = JSON.stringify(DEFAULT_CONTROL_POINTS.map((p) => ({ x: Number(p.x.toFixed(9)), y: Number(p.y.toFixed(9)) })));

// ---------------------------------------------------------------------------
// Tiny harness (same shape as tests/smoke.mjs).
// ---------------------------------------------------------------------------
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
      .split('\n').map((line) => '         ' + line).join('\n');
    console.log('  FAIL — ' + name + '\n' + detail);
  }
}
function assert(condition, message) {
  if (!condition) throw new Error('assert failed: ' + message);
}
function assertValidationError(fn, message) {
  let threw = null;
  try { fn(); } catch (err) { threw = err; }
  assert(threw !== null, (message || 'call') + ' — expected ValidationError, nothing thrown');
  assert(threw.name === 'ValidationError',
    (message || 'call') + ' — expected ValidationError, got ' + threw.name + ': ' + threw.message);
  return threw;
}

// ---------------------------------------------------------------------------
// Fixtures / helpers (mirrors tests/smoke.mjs).
// ---------------------------------------------------------------------------
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
function makeValidWeightsString() {
  return JSON.stringify(new NeuralNetwork(mulberry32(777)).getWeights());
}
function makeValidResults() {
  const results = [];
  for (let i = 0; i < POPULATION; i++) {
    results.push({ distance: 1000 - i * 10, topSpeed: 400 + i, timeAlive: 5.25, crashed: i % 4 === 0 ? 1 : 0, stuck: 0 });
  }
  return results;
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
    rngState: 1000 + (overrides && overrides.generation ? overrides.generation : 1),
    results: makeValidResults(),
    bestWeightsJson: makeValidWeightsString(),
  };
  return Object.assign(gen, overrides || {});
}
function logGen(db, n, overrides) {
  const gen = makeValidGen(Object.assign({ generation: n }, overrides || {}));
  db.logGeneration(gen);
  return gen;
}
function metaValue(db, key) {
  const rows = db.db.exec('SELECT value FROM meta WHERE key = ?', [key]);
  return rows && rows[0] && rows[0].values.length ? rows[0].values[0][0] : null;
}
function uvOf(db) {
  return Number(db.db.exec('SELECT COALESCE(user_version, 0) FROM pragma_user_version')[0].values[0][0]);
}
function sameBytesTyped(a, b) {
  if (!(a instanceof Uint8Array) || !(b instanceof Uint8Array)) return false;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// A faithful uv==2 fixture (zero tracks table), optionally carrying a stray
// meta.generator_version key — the crafted-legacy-v2 scenario under test.
function buildV2LineageBytes(versionKeyValue) {
  const SQL = globalThis.SQL;
  const { DDL_GENERATIONS_V2, DDL_DRIVER_RESULTS, DDL_BEST_WEIGHTS, DDL_META } = requireDb();
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
  raw.run('INSERT INTO meta(key, value) VALUES (?, ?)', ['track_hash', DEFAULT_TRACK_HASH]);
  raw.run("INSERT INTO meta(key, value) VALUES ('rng_state', ?)", [String(SEED)]);
  if (versionKeyValue !== undefined) {
    raw.run('INSERT INTO meta(key, value) VALUES (?, ?)', ['generator_version', versionKeyValue]);
  }
  const weights = makeValidWeightsString();
  for (let g = 1; g <= 2; g++) {
    raw.run('INSERT INTO generations (generation, drivers_evaluated, best_fitness, avg_fitness, worst_fitness, best_driver_index, mutation_rate, mutation_scale, track_json, track_hash, sim_seconds, rng_state) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
      [g, POPULATION, 1000 + g, 900 + g, 800 + g, 0, MUTATION_RATE, MUTATION_SCALE, DEFAULT_TRACK_JSON, DEFAULT_TRACK_HASH, 100 + g, 1000 + g]);
    for (let i = 0; i < POPULATION; i++) {
      raw.run('INSERT INTO driver_results (generation, driver_index, distance, top_speed, time_alive, crashed, stuck) VALUES (?,?,?,?,?,?,?)',
        [g, i, 500 + i, 300 + i, 10 + i, i % 4 === 0 ? 1 : 0, 0]);
    }
  }
  raw.run('INSERT INTO best_weights (generation, architecture, weights_json) VALUES (?,?,?)', [2, '7-16-16-2', weights]);
  const bytes = raw.export();
  raw.close();
  return bytes;
}

// FIT-V6: a structurally valid uv4 five-table lineage (schema_version='4',
// wall_model='v6'), optionally carrying a generator_version key. Used for the
// generator-version allowlist battery arm and the V1-by-default resume arm;
// pre-v4 bytes are now typed-refused before any generator_version consideration.
function buildV4LineageBytes(versionKeyValue) {
  const SQL = globalThis.SQL;
  const { DDL_META, DDL_TRACKS, DDL_GENERATIONS, DDL_DRIVER_RESULTS, DDL_BEST_WEIGHTS } = requireDb();
  const raw = new SQL.Database();
  raw.run('PRAGMA foreign_keys = ON');
  [DDL_META, DDL_TRACKS, DDL_GENERATIONS, DDL_DRIVER_RESULTS, DDL_BEST_WEIGHTS].forEach((ddl) => raw.run(ddl));
  raw.run('PRAGMA user_version = 4');
  raw.run("INSERT INTO meta(key, value) VALUES ('schema_version', '4')");
  raw.run("INSERT INTO meta(key, value) VALUES ('wall_model', 'v6')");
  raw.run("INSERT INTO meta(key, value) VALUES ('architecture', '7-16-16-2')");
  raw.run("INSERT INTO meta(key, value) VALUES ('app_version', '1.1.0')");
  raw.run("INSERT INTO meta(key, value) VALUES ('created_at', '2026-01-01 00:00:00')");
  raw.run('INSERT INTO meta(key, value) VALUES (?, ?)', ['track_hash', DEFAULT_TRACK_HASH]);
  raw.run('INSERT INTO meta(key, value) VALUES (?, ?)', ['rng_state', String(SEED)]);
  raw.run('INSERT INTO meta(key, value) VALUES (?, ?)', ['rotation_seed', String(SEED)]);
  if (versionKeyValue !== undefined) {
    raw.run('INSERT INTO meta(key, value) VALUES (?, ?)', ['generator_version', versionKeyValue]);
  }
  raw.run('INSERT INTO tracks(track_hash, source, control_points, half_width, difficulty, created_at) VALUES (?,?,?,?,?,?)',
    [DEFAULT_TRACK_HASH, 'default', DEFAULT_POINTS_JSON, 70, 0, '2026-01-01 00:00:00']);
  const bytes = raw.export();
  raw.close();
  return bytes;
}

function craftBytes(base, mutate) {
  const SQL = globalThis.SQL;
  const raw = new SQL.Database(base);
  mutate(raw);
  const bytes = raw.export();
  raw.close();
  return bytes;
}

function rot(seed, r, version) {
  const resolved = requireRotation().Rotation.resolveGeneratedTrack(seed, r, version);
  const json = JSON.stringify({ points: resolved.controlPoints, halfWidth: resolved.halfWidth });
  return { hash: resolved.trackHash, json: json, resolved: resolved };
}

function expectBatteryReject(Db, bytes, pattern, label) {
  let threw = null;
  try { Db.validateAdoptionBytes(bytes); } catch (err) { threw = err; }
  assert(threw !== null, (label || 'battery') + ' — expected rejection, got a pass');
  assert(threw.name === 'ValidationError', (label || 'battery') + ' — expected ValidationError, got ' + threw.name + ': ' + threw.message);
  if (pattern) assert(pattern.test(String(threw.message)), (label || 'battery') + ' — message ' + JSON.stringify(threw.message) + ' does not match ' + pattern);
  return threw;
}

// ===========================================================================
console.log('FIT-V4 TASK B — DB version contract');
// ===========================================================================

await leg('parser allowlist + Db re-export alias (never redefined)', async () => {
  const { Db } = requireDb();
  const Rotation = requireRotation().Rotation;
  assert(Db.parseGeneratorVersion === Rotation.parseGeneratorVersion,
    'Db.parseGeneratorVersion is the canonical Rotation parser (alias, not a redefinition)');
  const p = Db.parseGeneratorVersion;
  assert(p(undefined) === 1 && p(null) === 1, 'absent → V1');
  assert(p('1') === 1 && p(1) === 1, "'1'/1 → V1");
  assert(p('2') === 2 && p(2) === 2, "'2'/2 → V2");
  const bad = ['', '   ', '\t\n', '3', 3, '2 ', '02', '2x', 'x'.repeat(64), {}, [], NaN, Infinity, true];
  for (const v of bad) assertValidationError(() => p(v), 'invalid version ' + JSON.stringify(v) + ' refused');
  // Strict boundary mode still rejects missing/empty before delegating.
  const req = Rotation.requireGeneratorVersion;
  for (const v of [undefined, null, '']) assertValidationError(() => req(v), 'requireGeneratorVersion refuses ' + JSON.stringify(v));
  assert(req('1') === 1 && req('2') === 2 && req(2) === 2, 'requireGeneratorVersion passes valid values through');
});

await leg('ensureLineage writes generator_version=2 ONCE before materialization and returns it', async () => {
  const db = await freshDb();
  assert(metaValue(db, 'generator_version') === null, 'fresh DB has no generator_version key');
  const res = db.ensureLineage(DEFAULT_TRACK_HASH, SEED, DEFAULT_TRACK_JSON);
  assert(res && res.generatorVersion === 2, 'ensureLineage returns generatorVersion 2, got ' + JSON.stringify(res));
  assert(metaValue(db, 'generator_version') === '2', 'meta.generator_version written as string "2", got ' + metaValue(db, 'generator_version'));
  assert(countRows(db, 'tracks') === 100, '100 tracks rows (rotation 0 + 99) materialized, got ' + countRows(db, 'tracks'));
  // The 99 generated rows are V2 resolver hashes for the CURRENT seed.
  const Rotation = requireRotation().Rotation;
  for (let r = 1; r <= 99; r++) {
    const want = Rotation.resolveGeneratedTrack(SEED, r, 2).trackHash;
    const rows = db.db.exec('SELECT COUNT(*) FROM tracks WHERE track_hash = ? AND source = ?', [want, 'generated']);
    assert(rows && rows[0] && Number(rows[0].values[0][0]) === 1, 'rotation ' + r + ' has exactly one V2 generated row');
  }
  // Idempotent re-pin: same value → no-op, still 2, key still exactly one row.
  const res2 = db.ensureLineage(DEFAULT_TRACK_HASH, SEED, DEFAULT_TRACK_JSON);
  assert(res2 && res2.generatorVersion === 2, 'idempotent ensureLineage still returns 2');
  assert(Number(db.db.exec("SELECT COUNT(*) FROM meta WHERE key = 'generator_version'")[0].values[0][0]) === 1,
    'writer-once: exactly one generator_version meta row');
  db.db.close();
});

await leg('forkLineage writes generator_version=2 (new lineage) and returns it', async () => {
  const db = await freshDb();
  db.ensureLineage(DEFAULT_TRACK_HASH, SEED, DEFAULT_TRACK_JSON);
  const other = new Track(DEFAULT_CONTROL_POINTS.map((p, i) => ({ x: p.x + (i % 2 ? 1 : 0), y: p.y })), 70);
  const otherJson = other.toJSON();
  const otherHash = other.hash();
  const res = db.forkLineage(otherHash, SEED, otherJson);
  assert(res && res.generatorVersion === 2, 'forkLineage returns generatorVersion 2, got ' + JSON.stringify(res));
  assert(metaValue(db, 'generator_version') === '2', 'fork keeps generator_version=2 (new V2 lineage)');
  assert(countRows(db, 'generations') === 0, 'fork wiped generations');
  assert(countRows(db, 'tracks') >= 100, 'fork retained parent tracks rows + added new curriculum');
  db.db.close();
});

await leg('meta-tamper (invalid generator_version) is refused at every boundary', async () => {
  const { Db } = requireDb();
  // Battery: invalid stored values throw ValidationError. FIT-V6: the fixture
  // is uv4 (pre-v4 bytes are typed-refused at step 3b before this allowlist).
  for (const v of ['', '   ', '3', '2 ', 'x'.repeat(64)]) {
    const bytes = buildV4LineageBytes(v);
    expectBatteryReject(Db, bytes, /version|generator/i, 'battery refuses generator_version=' + JSON.stringify(v));
  }
  // Live DB: tampered meta causes resumeState + the boundary re-derivation to
  // halt closed (logGeneration reads the version at a rotation boundary, gen 51).
  const db = await freshDb();
  db.ensureLineage(DEFAULT_TRACK_HASH, SEED, DEFAULT_TRACK_JSON);
  logGen(db, 1);
  db.db.run("UPDATE meta SET value = '3' WHERE key = 'generator_version'");
  assertValidationError(() => db.resumeState(), 'resumeState refuses tampered meta version');
  // Restore, reach the boundary, tamper again — the boundary resolver must
  // parse meta (and therefore refuse the invalid value) before accepting any
  // payload hash.
  db.db.run("UPDATE meta SET value = '2' WHERE key = 'generator_version'");
  for (let g = 2; g <= 50; g++) logGen(db, g);
  const rot1 = rot(SEED, 1, 2);
  db.db.run("UPDATE meta SET value = '3' WHERE key = 'generator_version'");
  assertValidationError(() => logGen(db, 51, { trackJson: rot1.json, trackHash: rot1.hash, rngState: 9001 }),
    'gen-51 boundary refuses tampered meta version');
  assert(countRows(db, 'generations') === 50, 'tampered boundary logged nothing (rollback / pre-condition refusal)');
  db.db.close();
});

await leg('current-seed V2 rows verified; retained foreign-seed rows tolerated', async () => {
  const { Db } = requireDb();
  const db = await freshDb();
  db.ensureLineage(DEFAULT_TRACK_HASH, SEED, DEFAULT_TRACK_JSON);
  const base = db.exportBytes();
  db.db.close();
  // Baseline V2 lineage adopts cleanly.
  Db.validateAdoptionBytes(base);
  // A retained foreign-seed generated row is explicitly tolerated.
  const foreign = rot(999999, 1, 2);
  const tolerated = craftBytes(base, (raw) => {
    raw.run('INSERT INTO tracks (track_hash, source, control_points, half_width, difficulty, created_at) VALUES (?, ?, ?, ?, ?, ?)',
      [foreign.hash, 'generated', JSON.stringify(foreign.resolved.controlPoints), foreign.resolved.halfWidth, 1, '2026-01-01 00:00:00']);
  });
  Db.validateAdoptionBytes(tolerated);
  // A missing current-seed V2 row is refused by the scoped post-invariant.
  const wantR1 = rot(SEED, 1, 2);
  const corrupted = craftBytes(base, (raw) => {
    raw.run('DELETE FROM tracks WHERE track_hash = ?', [wantR1.hash]);
  });
  expectBatteryReject(Db, corrupted, /V2|rotation|generated|version/i, 'battery refuses missing current-seed V2 row');
});

await leg("crafted pre-v4 file with stray '2' key → typed WallModelChangedError, byte-preserving refusal", async () => {
  const { Db } = requireDb();
  const bytes = buildV2LineageBytes('2');
  const before = new Uint8Array(bytes);
  // FIT-V6: pre-v4 bytes are typed-refused at step 3b BEFORE the
  // generator_version allowlist or migration normalization could run. The
  // whole migrate-and-normalize lineage is intentionally out of scope.
  let thrown = null;
  try { Db.validateAdoptionBytes(bytes); } catch (err) { thrown = err; }
  assert(thrown !== null && thrown.name === 'WallModelChangedError',
    'battery typed refusal, got ' + (thrown && thrown.name));
  assert(sameBytesTyped(thrown.refusedBytes, before), 'typed error carries refused bytes verbatim');
  const db = new Db();
  thrown = null;
  try { await db.init(bytes); } catch (err) { thrown = err; }
  assert(thrown !== null && thrown.name === 'WallModelChangedError', 'init typed refusal');
  assert(db.db === null || db.db === undefined, 'zero adoption on refusal');
  assert(sameBytesTyped(bytes, before), 'refused file byte-identical');
});

await leg('legacy pre-v4 file (no generator_version key) → typed WallModelChangedError, byte-preserving refusal', async () => {
  const { Db } = requireDb();
  const bytes = buildV2LineageBytes();
  const before = new Uint8Array(bytes);
  let thrown = null;
  try { Db.validateAdoptionBytes(bytes); } catch (err) { thrown = err; }
  assert(thrown !== null && thrown.name === 'WallModelChangedError',
    'battery typed refusal, got ' + (thrown && thrown.name));
  assert(sameBytesTyped(thrown.refusedBytes, before), 'typed error carries refused bytes verbatim');
  assert(sameBytesTyped(bytes, before), 'file byte-identical after refusal');
});

await leg('fresh V2 lineage crosses gen 51 with the V2 hash, no divergence', async () => {
  const db = await freshDb();
  db.ensureLineage(DEFAULT_TRACK_HASH, SEED, DEFAULT_TRACK_JSON);
  for (let g = 1; g <= 50; g++) logGen(db, g);
  const rot1 = rot(SEED, 1, 2);
  logGen(db, 51, { trackJson: rot1.json, trackHash: rot1.hash, rngState: 9001 });
  assert(metaValue(db, 'track_hash') === rot1.hash, 'gen 51 moved the pin to the V2 rotation-1 hash');
  const rs = db.resumeState();
  assert(rs.generatorVersion === 2, 'resumeState exposes generatorVersion 2, got ' + rs.generatorVersion);
  assert(rs.maxGeneration === 51 && rs.rotationIndex === 1, 'resumeState reports gen 51 / rotation 1');
  // A payload carrying the OTHER version's hash is refused at the boundary.
  const staleDb = await freshDb();
  staleDb.ensureLineage(DEFAULT_TRACK_HASH, SEED, DEFAULT_TRACK_JSON);
  for (let g = 1; g <= 50; g++) logGen(staleDb, g);
  const wrong = rot(SEED, 1, 1);
  assertValidationError(() => logGen(staleDb, 51, { trackJson: wrong.json, trackHash: wrong.hash, rngState: 9002 }),
    'V2 boundary refuses a V1-hash payload (divergence halted)');
  staleDb.db.close();
  db.db.close();
});

await leg('zero-generation legacy re-pin becomes a NEW V2 lineage', async () => {
  const { Db } = requireDb();
  // A v3, zero-generation legacy lineage: no generator_version key.
  const db = await freshDb();
  const legacyZeroGen = craftBytes(db.exportBytes(), (raw) => {
    raw.run("DELETE FROM meta WHERE key = 'generator_version'");
  });
  db.db.close();
  const adopted = new Db();
  await adopted.init(legacyZeroGen);
  assert(metaValue(adopted, 'generator_version') === null, 'legacy zero-gen lineage starts with no key');
  assert(countRows(adopted, 'generations') === 0, 'legacy zero-gen lineage has no committed generations');
  const res = adopted.ensureLineage(DEFAULT_TRACK_HASH, SEED, DEFAULT_TRACK_JSON);
  assert(res && res.generatorVersion === 2, 'zero-gen re-pin returns V2');
  assert(metaValue(adopted, 'generator_version') === '2', 'zero-gen re-pin writes "2"');
  for (let r = 1; r <= 99; r++) {
    const want = requireRotation().Rotation.resolveGeneratedTrack(SEED, r, 2).trackHash;
    const rows = adopted.db.exec('SELECT COUNT(*) FROM tracks WHERE track_hash = ? AND source = ?', [want, 'generated']);
    assert(rows && rows[0] && Number(rows[0].values[0][0]) === 1, 'zero-gen re-pin rotation ' + r + ' is a V2 row');
  }
  adopted.db.close();
});

await leg('resumeState exposes generatorVersion on V1/V2/zero-gen lineages', async () => {
  const { Db } = requireDb();
  const fresh = await freshDb();
  assert(fresh.resumeState().generatorVersion === 1, 'fresh unpinned lineage reports V1');
  fresh.db.close();
  const v2db = await freshDb();
  v2db.ensureLineage(DEFAULT_TRACK_HASH, SEED, DEFAULT_TRACK_JSON);
  logGen(v2db, 1);
  assert(v2db.resumeState().generatorVersion === 2, 'committed V2 lineage reports V2');
  v2db.db.close();
  // FIT-V6: V1-by-default is now exercised on a uv4 lineage with no
  // generator_version key; pre-v4 migration-no-key lineages are typed-refused.
  const v1db = new Db();
  await v1db.init(buildV4LineageBytes());
  assert(v1db.resumeState().generatorVersion === 1, 'V1-by-default uv4 lineage reports V1');
  v1db.db.close();
});

// ---------------------------------------------------------------------------
console.log('');
console.log('DB-VERSION: ' + legsPassed + ' legs passed' + (failures.length ? ', ' + failures.length + ' FAILED' : ''));
if (failures.length) {
  for (const f of failures) console.log('  - ' + f.name + ' :: ' + String(f.err && f.err.message || f.err));
  process.exit(1);
}
