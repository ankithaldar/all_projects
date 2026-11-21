// js/db.js — The SQLite persistence layer for the 0003_car_racer evolutionary
// training lineage.
//
// WHY THIS FILE EXISTS (threat model in one paragraph):
// The training lineage is a SQLite database living in the browser tab and
// (optionally) in a user-picked file on disk via the File System Access API.
// Bytes can enter this file from OUTSIDE the session at any time — another
// tab, a hand-edited file, a downgraded/corrupted download, a malicious
// .sqlite from anywhere — and a poisoned database is not just a crash: a
// BEFORE INSERT trigger silently rewriting per-driver physics results, a
// view leaking lineage data, or a corrupt weights blob that survives all the
// way to the GPU would be invisible to a size check. The security boundary
// is therefore ONE shared function — Db.validateAdoptionBytes(), the
// "ADOPTION BATTERY" — and EVERY bytes-adoption path (init-from-bytes,
// openExisting(), importBytes(), the persist step-2 verify, the M4
// reconcile/reload flows) funnels through it or through its lighter sibling
// (_verifyForPersist, pure function of bytes, throwaway instance). Failure
// is always fail-closed: ZERO adoption, the working DB / file / IDB are
// left untouched, and the caller shows a blocking banner.
//
// ENVIRONMENT-AGNOSTIC: runs in node (smoke tests, via vendored sql-wasm.js)
// and in the browser (classic <script>). It never touches the DOM — it calls
// the hooks installed by setPersistenceHooks() instead and the caller renders
// banners. All SQL is literal static strings; every VALUE travels as a bound
// parameter; there are no template literals anywhere in this file.
// Plain script: no imports, no exports beyond the guarded module.exports at
// the bottom (and globalThis exposure for the browser), exactly like nn.js.
//
// LOADING SQL (fail-closed, no network, ever):
//   * The vendor files vendor/sql-wasm.js + vendor/sql-wasm.wasm are pinned
//     by SHA-256 in tests/smoke.mjs (hard-fail leg). Any vendor hash change
//     requires a new anti-team-security cycle — never a silent update.
//     vendor/sql-wasm-binary.js is the EMBEDDED base64 of the same
//     sql-wasm.wasm (auto-generated; sha256-pinned by leg 19) and exists
//     ONLY so file:// works — Chrome refuses fetch()/XHR there.
//   * Script order is LOAD-BEARING: index.html must load
//     vendor/sql-wasm-binary.js BEFORE vendor/sql-wasm.js, so
//     globalThis.SQL_WASM_BINARY_BASE64 is defined by the time db.js's
//     createBrowser() runs.
//   * The class consumes `globalThis.SQL` — the INITIALIZED sql.js namespace
//     (`await initSqlJs(...)`). Helpers:
//       - Db.loadNode(vendorDir)  — node/tests only: reads the vendored
//         sql-wasm.js + .wasm via require(), binds { wasmBinary } so nothing
//         is fetched, sets globalThis.SQL, returns the namespace. Throws
//         DbOfflineError outside node.
//       - Db.createBrowser()      — browser only: first tries the embedded
//         SQL_WASM_BINARY_BASE64 global (atob → { wasmBinary }), then falls
//         back to `initSqlJs({ locateFile: file => 'vendor/' + file })`.
//         Sets globalThis.SQL, returns it. Requires the vendor <script>s to
//         have run (window.initSqlJs).
//
// DETERMINISM / LINEAGE (see plan §DETERMINISM, §LINEAGE PIN):
//   * meta.rng_state (TEXT, decimal uint32) is AUTHORITATIVE for resume.
//     generations.rng_state (INTEGER) is audit-only. Both are committed in
//     the SAME transaction as the generation rows — never one without the
//     other. meta.rng_state is an UPDATE only; nothing here ever writes an
//     INSERT-or-whatever conflict clause — plain INSERT only (grep-gated).
//   * Resume boundary: spawn max(generation)+1. Resume NEVER re-inserts
//     generations <= max(generation). A UNIQUE conflict on INSERT aborts the
//     transaction and raises LineageIntegrityError (message contains
//     "UNIQUE") — the committed lineage is NEVER rewritten (N2 semantics).
//   * A non-finite/NaN fitness written BEFORE this build existed cannot be
//     caught by any battery — it can only be caught at write time, which is
//     what logGeneration() does (Number.isFinite on every numeric; fitness
//     may be signed, so no sign test applies). This
//     file does not attempt to detect pre-existing NaN payloads outside the
//     weights/track caps (NaN is a valid REAL in SQLite; integrity_check
//     does not flag it). Documented limitation, by design.
//
// PERSIST PIPELINE (plan §PERSIST — single ordered pipeline, one buffer
// end-to-end; EVERY failure path is fail-closed, no .crswap litter, and the
// banner never auto-hides). persist() is awaited by the session layer AFTER
// each generation's eval loop yields; this file performs the writes.
//
// The only I/O surfaces are the sqlite file handle (browser), IndexedDB
// (browser/node, feature-detected), crypto.subtle (browser/node,
// feature-detected) and — in getPersistCost() only — fs.writeFile to
// os.tmpdir() as a node proxy for the file write.

'use strict';

// ---------------------------------------------------------------------------
// Constants — the exact limits the plan pins. SQLite's own limits are far
// higher; these are OUR limits, tightened to the workload:
//   MAX_WEIGHTS_JSON_LENGTH  16384   — best_weights.weights_json column cap
//   MAX_TRACK_JSON_LENGTH    262144  — generations.track_json column cap (256 KB)
//   MAX_DB_BYTES             64 MiB  — ADOPTION BATTERY pre-parse cap (we
//                                      refuse to even open larger files; the
//                                      file the app writes can reach ~25–80 MB)
//   MAX_GENERATION_ROWS      100000  — battery cap: MAX_GENERATIONS x
//                                      POPULATION = 5000 x 20 = 100 000 (the
//                                      bound is INTENTIONAL and inclusive — a
//                                      complete full-lineage file passes)
//   MAX_DRIVER_RESULTS_ROWS  2000000 — battery cap: 100 000 gens x 20 drivers
//   POPULATION_SIZE          20      — the one number that must agree with
//                                      evolution.js's POPULATION (kept a
//                                      separate literal on purpose: db.js must
//                                      load without evolution.js)
// ---------------------------------------------------------------------------
var MAX_WEIGHTS_JSON_LENGTH = 16384;
var MAX_TRACK_JSON_LENGTH = 262144;
var MAX_DB_BYTES = 64 * 1024 * 1024;
var MAX_GENERATION_ROWS = 100000;
var MAX_DRIVER_RESULTS_ROWS = 2000000;
var MAX_TRACK_ROWS = 1000;          // FIT-V3: import-refusal bound for the tracks table (a full curriculum ≤ 101 rows; wipe/fork never delete tracks rows — documented in the plan §12)
var POPULATION_SIZE = 20;
var NN_ARCHITECTURE = '7-16-16-2';
var ROTATE_EVERY = 50;              // FIT-V3: kept a local literal on purpose (db.js must load without js/rotation.js — the same discipline as POPULATION_SIZE)
var MAX_PREMATERIALIZE_ATTEMPTS = 19800; // FIT-V4: 99 rotations x 200 generator attempts (rotation.js TRACK_MAX_REJECTS); fail-closed cumulative ceiling

// Static PRAGMA table_info statements — one literal per expected table, so no
// SQL is ever assembled at runtime.
var TABLE_INFO_SQL = {
  meta: 'PRAGMA table_info(meta)',
  generations: 'PRAGMA table_info(generations)',
  driver_results: 'PRAGMA table_info(driver_results)',
  best_weights: 'PRAGMA table_info(best_weights)',
  tracks: 'PRAGMA table_info(tracks)',
};
// Static PRAGMA foreign_key_list statements — literals, same policy.
var FK_LIST_SQL = {
  driver_results: 'PRAGMA foreign_key_list(driver_results)',
  best_weights: 'PRAGMA foreign_key_list(best_weights)',
};
// Value-domain batteries (F-8) — COUNT(*) with BOUND parameters. SQLite REAL
// can hold ±Infinity (bound doubles round-trip: Inf reads back as typeof
// 'real' with a JS-null payload) and a crafted file-format REAL can hold NaN;
// `x = x` is false for NaN, and Inf exceeds DBL_MAX. typeof() catches NULLs /
// texts / blobs hiding in REAL or INTEGER columns (REAL affinity stores an
// unconvertible value as-is). COUNT returns a single integer — no per-row
// materialization, safe at the 2 000 000-row cap.
var DBL_MAX = 1.7976931348623157e308;
var VALUE_DOMAIN_CHECKS = [
  { label: 'generations.generation', sql: "SELECT COUNT(*) FROM generations WHERE typeof(generation) != 'integer' OR generation < 1", params: [] },
  { label: 'generations.best_fitness', sql: "SELECT COUNT(*) FROM generations WHERE typeof(best_fitness) NOT IN ('integer','real') OR NOT (best_fitness = best_fitness) OR best_fitness > ? OR best_fitness < ?", params: [DBL_MAX, -DBL_MAX] },
  { label: 'generations.avg_fitness', sql: "SELECT COUNT(*) FROM generations WHERE typeof(avg_fitness) NOT IN ('integer','real') OR NOT (avg_fitness = avg_fitness) OR avg_fitness > ? OR avg_fitness < ?", params: [DBL_MAX, -DBL_MAX] },
  { label: 'generations.worst_fitness', sql: "SELECT COUNT(*) FROM generations WHERE typeof(worst_fitness) NOT IN ('integer','real') OR NOT (worst_fitness = worst_fitness) OR worst_fitness > ? OR worst_fitness < ?", params: [DBL_MAX, -DBL_MAX] },
  { label: 'generations.sim_seconds', sql: "SELECT COUNT(*) FROM generations WHERE typeof(sim_seconds) NOT IN ('integer','real') OR NOT (sim_seconds = sim_seconds) OR sim_seconds > ? OR sim_seconds < 0", params: [DBL_MAX] },
  { label: 'generations.drivers_evaluated', sql: "SELECT COUNT(*) FROM generations WHERE typeof(drivers_evaluated) != 'integer' OR drivers_evaluated < 1 OR drivers_evaluated > ?", params: [POPULATION_SIZE] },
  { label: 'generations.best_driver_index', sql: "SELECT COUNT(*) FROM generations WHERE typeof(best_driver_index) != 'integer' OR best_driver_index < 0 OR best_driver_index > ?", params: [POPULATION_SIZE - 1] },
  { label: 'generations.created_at', sql: "SELECT COUNT(*) FROM generations WHERE typeof(created_at) != 'text'", params: [] },
  { label: 'generations.rng_state', sql: "SELECT COUNT(*) FROM generations WHERE typeof(rng_state) != 'integer' OR rng_state < 0 OR rng_state > ?", params: [4294967295] },
  { label: 'generations.mutation_rate', sql: "SELECT COUNT(*) FROM generations WHERE typeof(mutation_rate) NOT IN ('integer','real') OR NOT (mutation_rate = mutation_rate) OR mutation_rate < 0 OR mutation_rate > 1", params: [] },
  { label: 'generations.mutation_scale', sql: "SELECT COUNT(*) FROM generations WHERE typeof(mutation_scale) NOT IN ('integer','real') OR NOT (mutation_scale = mutation_scale) OR mutation_scale < 0 OR mutation_scale > ?", params: [DBL_MAX] },
  { label: 'driver_results.driver_index', sql: "SELECT COUNT(*) FROM driver_results WHERE typeof(driver_index) != 'integer' OR driver_index < 0 OR driver_index > ?", params: [POPULATION_SIZE - 1] },
  { label: 'driver_results.distance', sql: "SELECT COUNT(*) FROM driver_results WHERE typeof(distance) NOT IN ('integer','real') OR NOT (distance = distance) OR distance > ? OR distance < ?", params: [DBL_MAX, -DBL_MAX] },
  { label: 'driver_results.top_speed', sql: "SELECT COUNT(*) FROM driver_results WHERE typeof(top_speed) NOT IN ('integer','real') OR NOT (top_speed = top_speed) OR top_speed > ? OR top_speed < 0", params: [DBL_MAX] },
  { label: 'driver_results.time_alive', sql: "SELECT COUNT(*) FROM driver_results WHERE typeof(time_alive) NOT IN ('integer','real') OR NOT (time_alive = time_alive) OR time_alive > ? OR time_alive < 0", params: [DBL_MAX] },
  { label: 'driver_results.crashed', sql: "SELECT COUNT(*) FROM driver_results WHERE typeof(crashed) != 'integer' OR crashed NOT IN (?, ?)", params: [0, 1] },
  { label: 'driver_results.stuck', sql: "SELECT COUNT(*) FROM driver_results WHERE typeof(stuck) != 'integer' OR stuck NOT IN (?, ?)", params: [0, 1] },
];
// Static PRAGMA user_version readback — table-valued pragma, literal (the
// exact shape the adoption/migration post-checks assert on; works on every
// sql.js this project vendors).
var USER_VERSION_READBACK_SQL = 'SELECT COALESCE(user_version, 0) FROM pragma_user_version';
// Exact column SETS (order-insensitive) — anything else is a foreign file.
// uv∈{1,2} raw-adoption paths keep today's four-table set EXACTLY; uv==3 raw
// bytes and every v3 verified-set path use the five-table set (with tracks).
var EXPECTED_TABLES = ['driver_results', 'best_weights', 'generations', 'meta'];
var EXPECTED_TABLES_V3 = ['driver_results', 'best_weights', 'generations', 'meta', 'tracks'];
var EXPECTED_COLUMNS = {
  meta: ['key', 'value'],
  generations: ['generation', 'created_at', 'drivers_evaluated', 'best_fitness', 'avg_fitness', 'worst_fitness', 'best_driver_index', 'mutation_rate', 'mutation_scale', 'track_json', 'track_hash', 'sim_seconds', 'rng_state'],
  driver_results: ['generation', 'driver_index', 'distance', 'top_speed', 'time_alive', 'crashed', 'stuck'],
  best_weights: ['generation', 'architecture', 'weights_json', 'created_at'],
};
var EXPECTED_COLUMNS_V3 = {
  meta: EXPECTED_COLUMNS.meta,
  generations: EXPECTED_COLUMNS.generations,
  driver_results: EXPECTED_COLUMNS.driver_results,
  best_weights: EXPECTED_COLUMNS.best_weights,
  tracks: ['track_hash', 'source', 'control_points', 'half_width', 'difficulty', 'created_at'],
};
// The v2 schema, verbatim (plan §DB SCHEMA v1 → v2 amendment). The four CREATE
// TABLE statements below are byte-for-byte the plan's DDL (modulo the plan's
// inline annotations) EXCEPT the single fitness-constraint change:
// generations.best_fitness dropped its `CHECK (best_fitness >= 0)` — fitness may
// be negative (net signed odometer + lap bonus) and must round-trip. NO other
// objects may ever exist in a lineage database.
var DDL_META = 'CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);';
// FITNESS-V2: `CHECK (best_fitness >= 0)` is GONE from DDL_GENERATIONS (v2).
// FIT-V3 (schema v3): DDL_GENERATIONS gains the declared FK
// `track_hash TEXT NOT NULL REFERENCES tracks(track_hash)` — the ONLY change vs
// FIT-V2. DDL_GENERATIONS is the v3 twin. FIT-V6 (schema v4) REUSES this exact
// twin verbatim — fresh v4 DBs and the uv>=3 battery's DDL compare consume it;
// the wall-model break is carried by meta.wall_model='v6', NOT by a new DDL
// literal. The FIT-V2 twin (no REFERENCES) is retained as DDL_GENERATIONS_V2
// for the uv==2 arm ONLY.
var DDL_GENERATIONS = 'CREATE TABLE generations(generation INTEGER PRIMARY KEY CHECK (generation >= 1), created_at TEXT NOT NULL DEFAULT (datetime(\'now\')), drivers_evaluated INTEGER NOT NULL CHECK (drivers_evaluated > 0), best_fitness REAL NOT NULL, avg_fitness REAL NOT NULL, worst_fitness REAL NOT NULL, best_driver_index INTEGER NOT NULL CHECK (best_driver_index BETWEEN 0 AND 19), mutation_rate REAL NOT NULL, mutation_scale REAL NOT NULL, track_json TEXT NOT NULL, track_hash TEXT NOT NULL REFERENCES tracks(track_hash), sim_seconds REAL NOT NULL, rng_state INTEGER NOT NULL CHECK (rng_state BETWEEN 0 AND 4294967295));';
// The FIT-V2 generations twin, kept EXACTLY for the uv==2 battery arm (a uv==2
// file relabelled v3 must not pass — and a v2 file must keep passing — the
// uv==2 twin compare) and for the v1 → v2 copy step. DDL_GENERATIONS_V2 is
// DERIVED from DDL_GENERATIONS_NEW_V2 (same literal, table-name swap) so the
// two can never drift apart.
var DDL_GENERATIONS_NEW_V2 = 'CREATE TABLE generations_new(generation INTEGER PRIMARY KEY CHECK (generation >= 1), created_at TEXT NOT NULL DEFAULT (datetime(\'now\')), drivers_evaluated INTEGER NOT NULL CHECK (drivers_evaluated > 0), best_fitness REAL NOT NULL, avg_fitness REAL NOT NULL, worst_fitness REAL NOT NULL, best_driver_index INTEGER NOT NULL CHECK (best_driver_index BETWEEN 0 AND 19), mutation_rate REAL NOT NULL, mutation_scale REAL NOT NULL, track_json TEXT NOT NULL, track_hash TEXT NOT NULL, sim_seconds REAL NOT NULL, rng_state INTEGER NOT NULL CHECK (rng_state BETWEEN 0 AND 4294967295));';
var DDL_GENERATIONS_V2 = DDL_GENERATIONS_NEW_V2.replace('generations_new', 'generations');
// The migration twin — MUST stay byte-identical to DDL_GENERATIONS modulo the
// table name (the v2 → v3 adoption copies generations through this exact
// shape). Any drift between the two means the migration produces a schema the
// fresh path would never create.
var DDL_GENERATIONS_NEW = 'CREATE TABLE generations_new(generation INTEGER PRIMARY KEY CHECK (generation >= 1), created_at TEXT NOT NULL DEFAULT (datetime(\'now\')), drivers_evaluated INTEGER NOT NULL CHECK (drivers_evaluated > 0), best_fitness REAL NOT NULL, avg_fitness REAL NOT NULL, worst_fitness REAL NOT NULL, best_driver_index INTEGER NOT NULL CHECK (best_driver_index BETWEEN 0 AND 19), mutation_rate REAL NOT NULL, mutation_scale REAL NOT NULL, track_json TEXT NOT NULL, track_hash TEXT NOT NULL REFERENCES tracks(track_hash), sim_seconds REAL NOT NULL, rng_state INTEGER NOT NULL CHECK (rng_state BETWEEN 0 AND 4294967295));';
// FIT-V3 schema v3 — Appendix A verbatim (one-line folded; the battery's twin
// compare folds case/whitespace before comparing, so the fold is lossless for
// these literals). The dedicated tracks table: one row per lineage-pinned or
// generated track; control_points is the BARE canonical JSON points array.
var DDL_TRACKS = 'CREATE TABLE tracks(track_hash TEXT PRIMARY KEY, source TEXT NOT NULL CHECK (source IN (\'default\',\'generated\',\'editor\')), control_points TEXT NOT NULL, half_width REAL NOT NULL CHECK (half_width >= 1 AND half_width <= 2000), difficulty INTEGER NOT NULL CHECK (difficulty BETWEEN 0 AND 100), created_at TEXT NOT NULL DEFAULT (datetime(\'now\')));';
// NOTE: the migration's INSERT … SELECT spells its explicit column list
// LITERALLY (both sides) inside migrateV1ToV2 — there is deliberately NO
// shared column-list constant to splice into SQL (the copy stays a pure
// single-literal statement).
// driver_results.distance carries FITNESS (net signed odometer + lap bonus;
// may be negative). The DDL column itself is unchanged from v1 (REAL NOT NULL
// is sign-agnostic) — only this comment records the fitness semantics.
var DDL_DRIVER_RESULTS = 'CREATE TABLE driver_results(generation INTEGER NOT NULL REFERENCES generations(generation) ON DELETE CASCADE, driver_index INTEGER NOT NULL CHECK (driver_index BETWEEN 0 AND 19), distance REAL NOT NULL, top_speed REAL NOT NULL, time_alive REAL NOT NULL, crashed INTEGER NOT NULL CHECK (crashed IN (0,1)), stuck INTEGER NOT NULL CHECK (stuck IN (0,1)), PRIMARY KEY (generation, driver_index));';
var DDL_BEST_WEIGHTS = "CREATE TABLE best_weights(generation INTEGER PRIMARY KEY REFERENCES generations(generation) ON DELETE CASCADE, architecture TEXT NOT NULL CHECK (architecture = '7-16-16-2'), weights_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now')));";

function errorMessage(err) {
  return String((err && err.message) || err);
}
function nowMs() {
  return globalThis.performance && typeof globalThis.performance.now === 'function'
    ? globalThis.performance.now()
    : Date.now();
}
function bytesToHex(bytes) {
  var out = '';
  for (var i = 0; i < bytes.length; i++) {
    var h = bytes[i].toString(16);
    out += h.length === 1 ? '0' + h : h;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Shared error classes (installed on globalThis if absent — same pattern as
// the pure-logic files; the browser exposes them for the M4 UI layer).
//   ValidationError       — bad caller input / battery violation / refusal.
//   LineageIntegrityError — a UNIQUE conflict escaped an INSERT (N2 semantics:
//                           the committed lineage is NEVER rewritten; the run
//                           HALTS; message always contains "UNIQUE").
//   DbOfflineError        — sql.js is not (or could not be) loaded: the Train
//                           tab must be blocked, Manual stays usable.
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

// ResumeTrackError — FIT-V3 (M6c): the D-5 resume-adoption arm refused. The
// stored tracks row the tip-generation pin names is ABSENT, or it fails the
// content/hash recombination arm. ALWAYS fail-closed with NO generator
// fallback (D-5/G3): resume must recompute from the stored bytes or refuse —
// never re-roll the curriculum under a committed lineage. A ValidationError
// subtype so existing `name === 'ValidationError'` callers keep working;
// main.js maps it to the dedicated LINEAGE GUARD banner.
if (typeof globalThis.ResumeTrackError !== 'function') {
  globalThis.ResumeTrackError = class ResumeTrackError extends globalThis.ValidationError {
    constructor(message) {
      super(message);
      this.name = 'ResumeTrackError';
    }
  };
}
var ResumeTrackError = globalThis.ResumeTrackError;

// WallModelChangedError — FIT-V6: a PRE-V4 lineage (created under the old
// naive ±halfWidth wall model) was offered for adoption. Its bytes encode a
// world this build can no longer reproduce, so it is refused byte-identically
// with a typed, actionable error naming the old wall model / no-resume /
// export + start a fresh lineage. A ValidationError subtype so existing
// `name === 'ValidationError'` callers keep working; main.js / db-session.js
// map it to the persistent export/fresh banner with a "download refused bytes
// verbatim" action. Refusal is BYTE-PRESERVING: the refused bytes travel with
// the error (err.refusedBytes) so no storage is cleared. Guarded install —
// first-loaded copy wins (same pattern as above).
var WALL_MODEL_CHANGED_MESSAGE = 'This lineage was created with an older wall model and cannot be resumed with this build. The original file was left untouched. Use Export .sqlite to download the refused bytes verbatim for your records, then start a fresh lineage.';
if (typeof globalThis.WallModelChangedError !== 'function') {
  globalThis.WallModelChangedError = class WallModelChangedError extends globalThis.ValidationError {
    constructor(message, refusedBytes) {
      super(message === undefined ? WALL_MODEL_CHANGED_MESSAGE : message);
      this.name = 'WallModelChangedError';
      if (refusedBytes !== undefined) this.refusedBytes = refusedBytes;
    }
  };
}
var WallModelChangedError = globalThis.WallModelChangedError;

// newWallModelChangedError(bytes) — the single factory for a typed wall-model
// refusal that carries the exact refused bytes verbatim for the UI download.
function newWallModelChangedError(bytes) {
  return new WallModelChangedError(undefined, bytes);
}

class LineageIntegrityError extends Error {
  constructor(message) {
    super(message);
    this.name = 'LineageIntegrityError';
  }
}

class DbOfflineError extends Error {
  constructor(message) {
    super(message);
    this.name = 'DbOfflineError';
  }
}

// MigrationError — the v1 → v2 schema migration aborted (any step: FK-OFF
// readback, BEGIN, collision pre-assert, create/copy/drop/rename, meta
// schema_version split, user_version, COMMIT, or the post-COMMIT verified
// set). ALWAYS fail-closed: the transaction is ROLLED BACK (if it began) and
// the instance is CLOSED/DISCARDED — zero adoption; the caller's bytes are
// never mutated. The committed bytes on disk (if any) keep their v1/v2
// form untouched; Manual mode boots with a banner.
class MigrationError extends Error {
  constructor(message) {
    super(message);
    this.name = 'MigrationError';
  }
}

// ---------------------------------------------------------------------------
// isLineageIntegrityFault(err) — the ONE shared lineage-integrity classifier
// (arch F-2 / security M2). Used by db.js (to wrap a conflict in
// LineageIntegrityError) and by the Trainer (to choose halt-vs-degrade for a
// logger throw). The two must never drift, so both delegate to this rule.
//
// Vendored sql-wasm build — OBSERVED MESSAGES (pinned by tests/smoke.mjs
// leg 23, which asserts the raw strings match this classifier):
//   "UNIQUE constraint failed: generations.generation"              (integer PK)
//   "UNIQUE constraint failed: driver_results.generation, driver_results.driver_index"  (composite PK)
//   "UNIQUE constraint failed: t.b"                                 (explicit UNIQUE)
// Every observed error carries code === undefined (plain Error); the regexes
// are therefore the load-bearing half with THIS vendor. The .code checks are
// for other builds (better-sqlite3 et al. surface SQLITE_CONSTRAINT_UNIQUE /
// SQLITE_CONSTRAINT_PRIMARYKEY on the error object). No "PRIMARY KEY
// constraint failed" spelling appears in this vendor build — kept in the
// regex anyway so the classifier is correct for builds that do emit it.
// ---------------------------------------------------------------------------
function isLineageIntegrityFault(err) {
  if (!err) return false;
  if (err.code === 'SQLITE_CONSTRAINT_UNIQUE' || err.code === 'SQLITE_CONSTRAINT_PRIMARYKEY') return true;
  if (err.code === 'FIT_V3_ROTATION_DIVERGENCE') return true; // FIT-V3: EVASION by the two derivation paths ⇒ halt, never log-disabled-continue
  return /\b(unique|primary key)\s+constraint\s+failed/i.test(String((err && err.message) || err));
}
// Guarded install — first-loaded copy wins (same pattern as ValidationError).
if (typeof globalThis.isLineageIntegrityFault !== 'function') {
  globalThis.isLineageIntegrityFault = isLineageIntegrityFault;
}

// strictUint32 — reused from nn.js when present (the ONE strict uint32 rule
// in the codebase). A local fallback is installed only if nn.js was not
// loaded (defensive; the documented load order always has nn.js first).
if (typeof globalThis.strictUint32 !== 'function') {
  globalThis.strictUint32 = function strictUint32Fallback(value, name) {
    var label = name || 'value';
    if (typeof value === 'number') {
      if (!Number.isInteger(value) || value < 0 || value > 4294967295) {
        throw new ValidationError(label + ' must be an integer uint32 [0, 4294967295], got ' + String(value));
      }
      return value;
    }
    if (typeof value === 'string' && /^(0|[1-9][0-9]{0,9})$/.test(value)) {
      var n = Number(value);
      if (n <= 4294967295) return n;
    }
    throw new ValidationError(label + ' must be a uint32 (integer number or decimal string), got ' + JSON.stringify(value));
  };
}
var strictUint32 = globalThis.strictUint32;

// ---------------------------------------------------------------------------
// FIT-V3 lazy requires — Track/js/rotation.js sit AFTER this file in script
// order (track.js before db.js — index.html pins it), so the browser fast
// path reads globalThis. In node db.js is loaded via CommonJS and the globals
// are present once the harness requires track.js/rotation.js — the require()
// fallback mirrors the sql.js loadNode pattern (absolute sibling path, no
// network ever).
// ---------------------------------------------------------------------------
var _trackModuleCache = null;
function requireTrack() {
  if (_trackModuleCache) return _trackModuleCache;
  var g = (typeof globalThis !== 'undefined') ? globalThis : {};
  if (typeof g.Track === 'function' && Array.isArray(g.DEFAULT_CONTROL_POINTS) &&
      typeof g.canonicalJSON === 'function' && typeof g.cyrb53 === 'function') {
    _trackModuleCache = { Track: g.Track, DEFAULT_CONTROL_POINTS: g.DEFAULT_CONTROL_POINTS, canonicalJSON: g.canonicalJSON, cyrb53: g.cyrb53 };
    return _trackModuleCache;
  }
  if (typeof require === 'function' && typeof __filename === 'string') {
    var path = require('node:path');
    var trackModule = require(path.join(path.dirname(__filename), 'track.js'));
    if (trackModule && typeof trackModule.Track === 'function' && Array.isArray(trackModule.DEFAULT_CONTROL_POINTS)) {
      _trackModuleCache = trackModule;
      return _trackModuleCache;
    }
  }
  throw new DbOfflineError('js/db.js requires js/track.js — load track.js before db.js');
}

var _rotationModuleCache = null;
function requireRotation() {
  if (_rotationModuleCache) return _rotationModuleCache;
  var g = (typeof globalThis !== 'undefined') ? globalThis : {};
  if (g.Rotation && typeof g.Rotation.resolveGeneratedTrack === 'function') {
    _rotationModuleCache = g.Rotation;
    return _rotationModuleCache;
  }
  if (typeof require === 'function' && typeof __filename === 'string') {
    var path = require('node:path');
    var rotationModule = require(path.join(path.dirname(__filename), 'rotation.js'));
    if (rotationModule && rotationModule.Rotation && typeof rotationModule.Rotation.resolveGeneratedTrack === 'function') {
      _rotationModuleCache = rotationModule.Rotation;
      return _rotationModuleCache;
    }
  }
  throw new DbOfflineError('js/db.js requires js/rotation.js — load rotation.js before db.js (FIT-V3 curriculum pre-materialization)');
}

// ---------------------------------------------------------------------------
// FIT-V4 version contract — the ONE canonical parser lives in rotation.js
// (Task A), the only module loaded in every realm. db.js DELEGATES / re-exports
// it; there is deliberately NO local `=== 2`, `== '2'` or `|| 1` version check
// anywhere in this file. The helpers below only *access* the canonical parser.
// ---------------------------------------------------------------------------
function requireVersionParser() {
  var Rotation = requireRotation();
  if (typeof Rotation.parseGeneratorVersion !== 'function') {
    throw new DbOfflineError('js/db.js requires Rotation.parseGeneratorVersion (FIT-V4) — load the versioned rotation.js before db.js');
  }
  return Rotation.parseGeneratorVersion;
}
function requireStrictVersionParser() {
  var Rotation = requireRotation();
  if (typeof Rotation.requireGeneratorVersion !== 'function') {
    throw new DbOfflineError('js/db.js requires Rotation.requireGeneratorVersion (FIT-V4) — load the versioned rotation.js before db.js');
  }
  return Rotation.requireGeneratorVersion;
}
var _canonicalV1 = null;
var _canonicalV2 = null;
function versionOne() {
  if (_canonicalV1 === null) _canonicalV1 = requireVersionParser()('1');
  return _canonicalV1;
}
function versionTwo() {
  if (_canonicalV2 === null) _canonicalV2 = requireVersionParser()('2');
  return _canonicalV2;
}
// Verification-only resolver entry. Every verification call routes through
// here with an EXPLICIT version; the plan's five-site grep gate counts the
// direct member calls (the two db.js boundary / materialization sites plus
// the three runtime sites in main/trainer/worker).
// Centralizing the verification calls here keeps each one explicit and leaves
// that canonical call-site set intact.
function resolveVersioned(seed, r, version) {
  return requireRotation().resolveGeneratedTrack(seed, r, version);
}

// The frozen DEFAULT circuit's hash — computed ONCE lazily on first use (NOT at
// import time — the node require must stay cheap and the browser path must not
// pay for a hash it never needs). The pinned default track is ALWAYS
// half_width 70; the frozen hash marks a lineage pin as source='default'.
var _frozenDefaultHash = null;
function frozenDefaultHash() {
  if (_frozenDefaultHash !== null) return _frozenDefaultHash;
  var t = requireTrack();
  var points = t.DEFAULT_CONTROL_POINTS.map(function (p) { return { x: p.x, y: p.y }; });
  _frozenDefaultHash = t.cyrb53(t.canonicalJSON({ points: points, halfWidth: 70 }));
  return _frozenDefaultHash;
}

// parseTrackJsonStrict — the {points, halfWidth} object out of a track_json
// payload. Track.fromJSON remains the judge of geometry validity; this is the
// structural pre-check (points ARRAY of objects, object halfWidth). Throws
// ValidationError with the caller's context prefix on ANY fault.
function parseTrackJsonStrict(trackJson, context) {
  var ctx = context || 'track_json';
  if (typeof trackJson !== 'string' || trackJson.length === 0) {
    throw new ValidationError(ctx + ': track_json must be a non-empty string, got ' + (typeof trackJson));
  }
  var parsed;
  try {
    parsed = JSON.parse(trackJson);
  } catch (err) {
    throw new ValidationError(ctx + ': track_json is not valid JSON — ' + errorMessage(err));
  }
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed) || !Array.isArray(parsed.points)) {
    throw new ValidationError(ctx + ': track_json must be an object { points, halfWidth } with a points array, got ' + (parsed === null ? 'null' : typeof parsed));
  }
  return { points: parsed.points, halfWidth: parsed.halfWidth };
}

// recombineTrackHash(points, halfWidth, context) — THE shared recombination
// path (plan D-14): the pair is passed through
// Track.fromJSON(JSON.stringify({points, halfWidth})) verbatim and re-hashed
// with the cyrb53(canonicalJSON(...)) surrogate every consumer uses. NOTHING
// recombines by string splicing. Throws ValidationError on any fault.
function recombineTrackHash(points, halfWidth, context) {
  var ctx = context || 'recombine';
  var t = requireTrack();
  var payload;
  try {
    payload = JSON.stringify({ points: points, halfWidth: halfWidth });
  } catch (err) {
    throw new ValidationError(ctx + ': could not serialize the recombined {points, halfWidth} — ' + errorMessage(err));
  }
  var track;
  try {
    track = t.Track.fromJSON(payload);
  } catch (err) {
    throw new ValidationError(ctx + ': recombined track rejected by Track.fromJSON — ' + errorMessage(err));
  }
  return t.cyrb53(t.canonicalJSON(track));
}

// generationRowHash(trackJson, context) — for generations rows: recombine the
// stored track_json to its hash (Track.fromJSON validates the bytes on the way).
function generationRowHash(trackJson, context) {
  var ctx = context || 'track_json';
  var parsed = parseTrackJsonStrict(trackJson, ctx);
  return recombineTrackHash(parsed.points, parsed.halfWidth, ctx);
}

// controlPointsJson(points) — the canonical BARE points array exactly as the
// tracks.control_points column stores it (coordinates rounded via
// Number(v.toFixed(9)) — the same rounding canonicalJSON applies).
function controlPointsJson(points) {
  var arr = new Array(points.length);
  for (var i = 0; i < points.length; i++) {
    arr[i] = {
      x: Number(Number(points[i].x).toFixed(9)),
      y: Number(Number(points[i].y).toFixed(9)),
    };
  }
  return JSON.stringify(arr);
}

// sourceForHash(hash) — the frozen source rule (plan D-12): 'default' iff the
// hash equals the frozen DEFAULT_CONTROL_POINTS hash, else 'editor'.
function sourceForHash(hash) {
  return hash === frozenDefaultHash() ? 'default' : 'editor';
}

// controlPointsContentError(parsed) — the code-side content arm for a parsed
// tracks.control_points payload (10b arm 5). Returns an error string or null.
// ≥3 finite {x,y} objects with |x|,|y| ≤ 1e6 — nothing more (the whole-row
// recombination arm validates the geometry for real).
function controlPointsContentError(parsed) {
  if (!Array.isArray(parsed)) return 'control_points must be a JSON array of {x,y} points';
  if (parsed.length < 3) return 'control_points holds ' + parsed.length + ' point(s) — at least 3 are required';
  for (var i = 0; i < parsed.length; i++) {
    var p = parsed[i];
    if (p === null || typeof p !== 'object' || Array.isArray(p)) {
      return 'control_points[' + i + '] must be an object {x, y}';
    }
    var keys = Object.keys(p);
    if (keys.length !== 2 || keys[0] !== 'x' || keys[1] !== 'y') {
      return 'control_points[' + i + '] must carry exactly the keys {x, y} (got ' + JSON.stringify(keys) + ')';
    }
    var x = p.x, y = p.y;
    if (typeof x !== 'number' || typeof y !== 'number' || !Number.isFinite(x) || !Number.isFinite(y)) {
      return 'control_points[' + i + '] has non-finite or non-number coordinates';
    }
    if (Math.abs(x) > 1e6 || Math.abs(y) > 1e6) {
      return 'control_points[' + i + '] exceeds |coordinate| <= 1e6';
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// FIT-V3 shared check helpers — the ONE implementation behind both the battery
// (adoption, ValidationError contract) and _assertMigratedState(v3) (post-
// migration verified set, MigrationError contract — the caller wraps). Each
// takes `context` as the message prefix and throws ValidationError naming the
// arm; callers that need a typed MigrationError wrap the throw.
// ---------------------------------------------------------------------------

// Count arm (refuse early): every generations.track_hash must resolve in tracks.
function checkDanglingGenerations(db, context) {
  var dangling = countScalar(db, 'SELECT COUNT(*) FROM generations g WHERE NOT EXISTS (SELECT 1 FROM tracks t WHERE t.track_hash = g.track_hash)');
  if (dangling > 0) {
    throw new ValidationError(context + ': ' + dangling + ' generations row(s) reference a track_hash with NO tracks row — refusing adoption');
  }
}

// Tracks value-domain arms (10b arms 1–4, 6, 7) — literal COUNT(*) arms with
// typeof-strict, NaN/NULL-safe predicates. Honest INTEGER widths pass arm 2.
function checkTracksValueDomain(db, context) {
  var tracksChecks = [
    { label: 'tracks.difficulty', sql: "SELECT COUNT(*) FROM tracks WHERE typeof(difficulty) != 'integer' OR difficulty < 0 OR difficulty > 100", params: [] },
    { label: 'tracks.half_width', sql: "SELECT COUNT(*) FROM tracks WHERE typeof(half_width) NOT IN ('integer','real') OR typeof(half_width) = 'blob' OR NOT (half_width = half_width) OR half_width < 1 OR half_width > 2000", params: [] },
    { label: 'tracks.source', sql: "SELECT COUNT(*) FROM tracks WHERE typeof(source) != 'text' OR source NOT IN ('default','generated','editor')", params: [] },
    { label: 'tracks.control_points', sql: "SELECT COUNT(*) FROM tracks WHERE typeof(control_points) != 'text' OR length(control_points) = 0 OR length(control_points) > ?", params: [MAX_TRACK_JSON_LENGTH] },
    { label: 'tracks.track_hash', sql: "SELECT COUNT(*) FROM tracks WHERE typeof(track_hash) != 'text' OR length(track_hash) < 1 OR length(track_hash) > 64", params: [] },
    { label: 'tracks.created_at', sql: "SELECT COUNT(*) FROM tracks WHERE typeof(created_at) != 'text'", params: [] },
  ];
  for (var tci = 0; tci < tracksChecks.length; tci++) {
    var check = tracksChecks[tci];
    var n = countScalar(db, check.sql, check.params);
    if (n > 0) {
      throw new ValidationError(context + ': ' + n + ' row(s) violate the value domain of ' + check.label + ' — the file was hand-crafted or corrupted; refusing adoption');
    }
  }
}

// Tracks content + whole-row recombination arms (10b arms 5 and 8). One row
// scan; the content arm runs BEFORE the recombination arm (cheaper refusal).
function checkTracksContentAndHash(db, context) {
  var rows = db.exec('SELECT track_hash, control_points, half_width FROM tracks');
  var values = (rows && rows[0] && rows[0].values) || [];
  for (var i = 0; i < values.length; i++) {
    var trackHash = values[i][0];
    var controlPoints = values[i][1];
    var halfWidth = values[i][2];
    var fp = null;
    if (typeof controlPoints === 'string') {
      try {
        fp = JSON.parse(controlPoints);
      } catch (err) {
        fp = null;
      }
    }
    var contentError = controlPointsContentError(fp);
    if (contentError) {
      throw new ValidationError(context + ': tracks row track_hash ' + JSON.stringify(trackHash) + ' — ' + contentError + '; refusing adoption');
    }
    var recombined = generationRowHash(JSON.stringify({ points: fp, halfWidth: halfWidth }), context);
    if (recombined !== trackHash) {
      throw new ValidationError(context + ': tracks row track_hash ' + JSON.stringify(trackHash) + ' — recombined hash ' + recombined + ' does not match — hash/bytes divergence (whole-row recombination arm); refusing adoption');
    }
  }
}

// Generations self-hash arm (battery step 8b / the v3 verified set's mirror):
// for EVERY generations row, the stored track_json must recombine to the row's
// track_hash. Bounded by MAX_GENERATION_ROWS (the battery's cap).
function checkGenerationsSelfHash(db, context) {
  var rows = db.exec('SELECT generation, track_json, track_hash FROM generations');
  var values = (rows && rows[0] && rows[0].values) || [];
  for (var i = 0; i < values.length; i++) {
    var recombined = generationRowHash(values[i][1], context);
    if (recombined !== values[i][2]) {
      throw new ValidationError(context + ': generations row generation ' + values[i][0] + ' — track_json recombined to ' + recombined + ' but the row carries track_hash ' + values[i][2] + ' (self-hash arm); refusing adoption');
    }
  }
}

// Generations FK arm (10b arm 10 / the verified set's dedicated arm): exactly
// one FK — (table='tracks', from='track_hash', to='track_hash') — with
// on_delete ∉ {SET NULL, SET DEFAULT, CASCADE} (only '' or 'NO ACTION').
function checkGenerationsForeignKey(db, context) {
  var rows = [];
  try {
    var res = db.exec('PRAGMA foreign_key_list(generations)');
    rows = (res && res[0] && res[0].values) || [];
  } catch (err) {
    throw new ValidationError(context + ': could not read the foreign keys of generations (' + errorMessage(err) + ') — refusing adoption');
  }
  var byId = {};
  var order = [];
  for (var i = 0; i < rows.length; i++) {
    var id = rows[i][0];
    if (!Object.prototype.hasOwnProperty.call(byId, id)) { byId[id] = []; order.push(id); }
    byId[id].push(rows[i]);
  }
  if (order.length !== 1) {
    throw new ValidationError(context + ': generations declares ' + order.length + ' foreign keys — exactly ONE (track_hash → tracks(track_hash), NO ACTION) is permitted; refusing adoption');
  }
  var cols = byId[order[0]];
  if (cols.length !== 1 || Number(cols[0][1]) !== 0) {
    throw new ValidationError(context + ': generations declares a multi-column foreign key — exactly ONE single-column FK on track_hash → tracks(track_hash) is permitted; refusing adoption');
  }
  var row = cols[0];
  var targetTable = String(row[2] || '').toLowerCase();
  var fromCol = String(row[3] || '').toLowerCase();
  // `to` may arrive quoted — normalize by stripping the quotes.
  var toCol = String(row[4] || '').replace(/^"(.*)"$/, '$1').toLowerCase();
  var onDelete = String(row[6] == null ? '' : row[6]).trim().toLowerCase();
  if (targetTable !== 'tracks' || fromCol !== 'track_hash' || toCol !== 'track_hash') {
    throw new ValidationError(context + ': generations foreign key mismatch — expected track_hash → tracks(track_hash), got ' + String(row[3]) + ' → ' + String(row[2]) + '(' + String(row[4]) + '); refusing adoption');
  }
  if (onDelete !== '' && onDelete !== 'no action') {
    throw new ValidationError(context + ': generations foreign key is ON DELETE ' + onDelete.toUpperCase() + " — only '' or 'NO ACTION' is accepted (a cascade/delete rule would let tracks rows vanish under a committed lineage); refusing adoption");
  }
}

// insertTrackRowIfAbsent(db, row, context) — SELECT-then-split (NO upsert
// spelling anywhere in this file): INSERT only when no tracks row with this
// track_hash exists — same-geometry pins and reforks are no-ops, never raw PK
// errors. Returns { inserted: true|false }.
function insertTrackRowIfAbsent(db, row, context) {
  var countRows = db.exec('SELECT COUNT(*) FROM tracks WHERE track_hash = ?', [row.track_hash]);
  var count = countRows && countRows[0] && countRows[0].values && countRows[0].values[0]
    ? Number(countRows[0].values[0][0])
    : NaN;
  if (!Number.isFinite(count)) {
    throw new ValidationError((context || 'tracks') + ': could not count tracks rows for track_hash ' + String(row.track_hash));
  }
  if (count > 0) return { inserted: false };
  db.run('INSERT INTO tracks (track_hash, source, control_points, half_width, difficulty, created_at) VALUES (?, ?, ?, ?, ?, ?)',
    [row.track_hash, row.source, row.control_points, row.half_width, row.difficulty, row.created_at]);
  return { inserted: true };
}

// preMaterializeCurriculum(db, rotationSeed, context, versionOverride) — plan
// D-12 + FIT-V4: rotations 1..99 resolved from (rotation_seed, r, version),
// BYTE-VERIFIED per row (cyrb53 of the canonical bytes == the row's own
// track_hash; mismatch ⇒ ValidationError — the caller rolls the pin/adoption
// back), then INSERT via SELECT-then-split. created_at is the migration-artefact
// sentinel, never datetime('now'). Any failure rolls the caller's transaction
// back (zero adoption / zero pin).
//
// FIT-V4 version selection (N4): when `versionOverride` is SUPPLIED it FULLY
// replaces the db-derived version for BOTH materialization and the post-
// invariant, and the db meta read is SKIPPED entirely. This is the migration
// path (override = V1): a crafted v2 file carrying a stray '2' key is still
// materialized as V1; the key is deleted by the migration so a later meta read
// also yields V1. When no override is supplied, the version comes from the
// per-lineage meta.generator_version via the ONE canonical parser (absent ⇒ V1).
//
// Post-invariant (N5), scoped to the CURRENT rotation_seed: for r=1..99,
// resolve(currentSeed, r, version) has exactly ONE matching generated row and
// resolve(currentSeed, r, otherVersion) has NONE. Retained foreign-seed rows
// from earlier forks are explicitly tolerated — they carry different hashes,
// so a per-r exact-hash check never mistakes them for a current-seed row.
// Cumulative attempt ceiling (N3): 99 x 200 = 19,800, fail-closed.
// Returns the integer version actually used.
function preMaterializeCurriculum(db, rotationSeed, context, versionOverride) {
  var resolvedSeed = strictUint32(rotationSeed, (context || 'curriculum') + ' rotation_seed');
  var t = requireTrack();
  var Rotation = requireRotation();
  var ctx = context || 'curriculum';

  // Version selection — override wins completely (and skips the db meta read).
  var version;
  if (arguments.length >= 4 && versionOverride !== undefined) {
    // A supplied override is an explicit caller value: require it to be
    // present and canonical (strict boundary mode) — never a silent default.
    version = requireStrictVersionParser()(versionOverride);
  } else {
    var versionRows = db.exec("SELECT value FROM meta WHERE key = 'generator_version'");
    var rawVersion = (versionRows && versionRows[0] && versionRows[0].values.length)
      ? versionRows[0].values[0][0]
      : undefined;
    version = requireVersionParser()(rawVersion); // permissive meta mode: absent ⇒ V1
  }

  var attempts = 0;
  for (var r = 1; r <= 99; r++) {
    // Every resolve consumes at most the generator's fixed 200-attempt reject
    // cap; the cumulative budget is 99 x 200 = 19,800. Above it we refuse
    // instead of doing unbounded generation work (DoS backstop, N3).
    attempts += 200;
    if (attempts > MAX_PREMATERIALIZE_ATTEMPTS) {
      throw new ValidationError(ctx + ': pre-materialization attempt ceiling ' + MAX_PREMATERIALIZE_ATTEMPTS + ' exceeded at rotation ' + r + ' — refusing');
    }
    var resolved = Rotation.resolveGeneratedTrack(resolvedSeed, r, version);
    var verify = t.cyrb53(t.canonicalJSON({ points: resolved.controlPoints, halfWidth: resolved.halfWidth }));
    if (verify !== resolved.trackHash) {
      throw new ValidationError(ctx + ': pre-materialized rotation ' + r + ' failed byte-verification (recomputed ' + verify + ' != resolver-reported ' + resolved.trackHash + ') — refusing to pin a corrupted curriculum');
    }
    insertTrackRowIfAbsent(db, {
      track_hash: resolved.trackHash,
      source: 'generated',
      control_points: controlPointsJson(resolved.controlPoints),
      half_width: resolved.halfWidth,
      difficulty: resolved.difficulty,
      created_at: '2026-01-01 00:00:00',
    }, context);
  }

  // Scoped post-invariant — exactly one row at `version`, zero at the other.
  var otherVersion = (version === versionOne()) ? versionTwo() : versionOne();
  for (var p = 1; p <= 99; p++) {
    var want = resolveVersioned(resolvedSeed, p, version).trackHash;
    var nWant = countScalar(db, 'SELECT COUNT(*) FROM tracks WHERE track_hash = ? AND source = ?', [want, 'generated']);
    if (nWant !== 1) {
      throw new ValidationError(ctx + ': post-invariant rotation ' + p + ' has ' + nWant + ' generated rows matching version ' + version + ' (expected exactly one) — refusing');
    }
    var other = resolveVersioned(resolvedSeed, p, otherVersion).trackHash;
    if (other !== want) {
      var nOther = countScalar(db, 'SELECT COUNT(*) FROM tracks WHERE track_hash = ? AND source = ?', [other, 'generated']);
      if (nOther !== 0) {
        throw new ValidationError(ctx + ': post-invariant rotation ' + p + ' has ' + nOther + ' generated row(s) matching the other version ' + otherVersion + ' (expected zero) — refusing');
      }
    }
  }
  return version;
}

// ---------------------------------------------------------------------------
// FIT-V4 generator_version meta arms — used by the adoption battery, the
// persist step-2 verify and the migration verified set. The allowlist is the
// canonical parser itself: absent / 1 / 2 pass; '', whitespace, over-length
// and every other value throw a typed ValidationError (no silent downgrade).
// ---------------------------------------------------------------------------
function metaGeneratorVersionRaw(db) {
  var rows = db.exec("SELECT value FROM meta WHERE key = 'generator_version'");
  return (rows && rows[0] && rows[0].values.length) ? rows[0].values[0][0] : undefined;
}
function checkGeneratorVersionAllowlist(db, context) {
  var raw = metaGeneratorVersionRaw(db);
  try {
    requireVersionParser()(raw); // absent ⇒ V1; invalid ⇒ typed ValidationError
  } catch (err) {
    throw new ValidationError(context + ': meta.generator_version ' + JSON.stringify(raw) + ' is not an allowed value — ' + errorMessage(err));
  }
}
// V2-pinned lineages: the CURRENT seed's 99 curriculum rows must resolve to
// exactly one generated row each at V2. MUST NOT run on the uv<=2 migration
// path (a crafted v2 file with a stray '2' key is adopted and normalized by
// DELETING that key in migration before any downstream meta read) — callers
// gate this arm to native v3 bytes / post-migration state (N2/N5).
function checkV2PinnedCurriculumRows(db, context) {
  var raw = metaGeneratorVersionRaw(db);
  var version = requireVersionParser()(raw);
  if (version !== versionTwo()) return; // V1 lineage: no V2-row assertion
  var seedRows = db.exec("SELECT value FROM meta WHERE key = 'rotation_seed'");
  if (!(seedRows && seedRows[0] && seedRows[0].values.length)) {
    return; // zero-generation V2 lineage (no pin yet): nothing to verify
  }
  var seed = strictUint32(seedRows[0].values[0][0], context + ' rotation_seed');
  for (var r = 1; r <= 99; r++) {
    var want = resolveVersioned(seed, r, version).trackHash;
    var n = countScalar(db, 'SELECT COUNT(*) FROM tracks WHERE track_hash = ? AND source = ?', [want, 'generated']);
    if (n !== 1) {
      throw new ValidationError(context + ': V2-pinned lineage rotation ' + r + ' has ' + n + ' generated rows matching the V2 resolver hash (expected exactly one) — refusing adoption');
    }
  }
}

// NeuralNetwork.fromJSON — the ONE weights validator. Reused by the battery
// (step 7) and getBestWeights()/resumeState() so bytes adopted from ANYWHERE
// are validated exactly the same way.
function requireNeuralNetwork() {
  var NeuralNetwork = globalThis.NeuralNetwork;
  if (!NeuralNetwork || typeof NeuralNetwork.fromJSON !== 'function') {
    throw new DbOfflineError('js/db.js requires NeuralNetwork (js/nn.js) — load nn.js before db.js');
  }
  return NeuralNetwork;
}

// requireSql — every SQL use in this file goes through this single accessor
// so a missing/blocked vendor bundle fails closed with a typed error.
function requireSql() {
  var SQL = globalThis.SQL;
  if (!SQL || typeof SQL.Database !== 'function') {
    throw new DbOfflineError('sql.js is not loaded — vendor/sql-wasm.js missing/corrupt/CSP-blocked; serve over localhost or fix the vendor files (see README)');
  }
  return SQL;
}

function countScalar(db, sql, params) {
  var rows = db.exec(sql, params);
  if (!rows || rows.length === 0 || !rows[0].values || rows[0].values.length === 0) return 0;
  var v = rows[0].values[0][0];
  return typeof v === 'number' ? v : Number(v) || 0;
}

// checkSingleFk / checkAdoptionSemantics — the battery's SEMANTIC arm (F-1,
// F-8). The structural arms prove the file is shaped like a lineage; these
// prove it BEHAVES like one:
//   (a) driver_results and best_weights each declare EXACTLY ONE foreign
//       key — generation → generations(generation) ON DELETE CASCADE (the
//       cascade forkLineage relies on so orphaned driver_results /
//       best_weights never survive a wipe). A hand-crafted file with the
//       exact column NAME sets but NO declared FK — or a multi-column /
//       mis-targeted / non-cascade FK — is REFUSED, naming what is missing.
//   (b) every generations / driver_results numeric value is inside its
//       domain (fitness finite but SIGNED — fitness may be negative; the
//       structural columns finite ≥ 0; integer ranges; crashed/stuck ∈ {0,1}) —
//       the CHECKs live only in OUR DDL, never in a foreign file.
// All checks are literal PRAGMAs / bound-parameter SELECTs over the rows —
// no interpolation anywhere.
function checkSingleFk(db, table, context) {
  var rows = [];
  try {
    var res = db.exec(FK_LIST_SQL[table]);
    rows = (res && res[0] && res[0].values) || [];
  } catch (err) {
    throw new ValidationError(context + ': could not read the foreign keys of ' + table + ' (' + errorMessage(err) + ') — refusing adoption');
  }
  if (rows.length === 0) {
    throw new ValidationError(context + ': ' + table + ' declares NO foreign key (expected ' + table + '.generation → generations(generation) ON DELETE CASCADE) — the file was hand-crafted or corrupted; refusing adoption');
  }
  // foreign_key_list columns: id, seq, table, from, to, on_update, on_delete, match.
  var byId = {};
  var order = [];
  for (var i = 0; i < rows.length; i++) {
    var id = rows[i][0];
    if (!Object.prototype.hasOwnProperty.call(byId, id)) { byId[id] = []; order.push(id); }
    byId[id].push(rows[i]);
  }
  if (order.length !== 1) {
    throw new ValidationError(context + ': ' + table + ' declares ' + order.length + ' foreign keys — exactly ONE (generation → generations(generation) ON DELETE CASCADE) is permitted; refusing adoption');
  }
  var cols = byId[order[0]];
  if (cols.length !== 1 || Number(cols[0][1]) !== 0) {
    throw new ValidationError(context + ': ' + table + ' declares a multi-column foreign key — exactly ONE single-column FK on generation → generations(generation) is permitted; refusing adoption');
  }
  var row = cols[0];
  var targetTable = String(row[2] || '').toLowerCase();
  var fromCol = String(row[3] || '').toLowerCase();
  // `to` may arrive quoted ('"generation"') — normalize by stripping the quotes.
  var toCol = String(row[4] || '').replace(/^"(.*)"$/, '$1').toLowerCase();
  var onDelete = String(row[6] == null ? '' : row[6]).trim().toLowerCase();
  if (targetTable !== 'generations' || fromCol !== 'generation' || toCol !== 'generation') {
    throw new ValidationError(context + ': ' + table + ' foreign key mismatch — expected generation → generations(generation), got ' + String(row[3]) + ' → ' + String(row[2]) + '(' + String(row[4]) + '); refusing adoption');
  }
  if (onDelete !== 'cascade') {
    throw new ValidationError(context + ': ' + table + ' foreign key on ' + table + '.generation is ON DELETE ' + (onDelete === '' ? '(empty)' : onDelete.toUpperCase()) + ' — only ON DELETE CASCADE is accepted (an empty on_delete or NO ACTION would orphan rows at fork/wipe); refusing adoption');
  }
}

function checkAdoptionSemantics(db, context) {
  checkSingleFk(db, 'driver_results', context);
  checkSingleFk(db, 'best_weights', context);
  for (var c = 0; c < VALUE_DOMAIN_CHECKS.length; c++) {
    var check = VALUE_DOMAIN_CHECKS[c];
    var n = countScalar(db, check.sql, check.params);
    if (n > 0) {
      throw new ValidationError(context + ': ' + n + ' row(s) violate the value domain of ' + check.label + ' (fitness finite but signed; structural integers in range; 0-1 flags) — the file was hand-crafted or corrupted; refusing adoption');
    }
  }
}

// scrubSqliteMaster — the SEC-18 scrub: EXACTLY the expected tables plus
// sqlite_autoindex_* entries; ZERO triggers, ZERO views, zero foreign
// indexes, zero anything else. Shared by the ADOPTION BATTERY and by init()
// (both fresh and adopted connections) — identical strictness at both gates.
// `expectedTables` selects the set (undefined ⇒ today's four-table set, kept
// byte-for-byte for the uv∈{1,2} paths; the v3 paths pass
// EXPECTED_TABLES_V3 — uv==3 raw bytes, post-migration and fresh init). The
// error strings name the right set.
function scrubSqliteMaster(db, context, expectedTables) {
  var rows = db.exec('SELECT type, name FROM sqlite_master');
  var objects = rows && rows[0] ? rows[0].values : [];
  var tablesList = expectedTables || EXPECTED_TABLES;
  var tableWord = tablesList.length === 5 ? 'five' : 'four';
  var tables = {};
  for (var i = 0; i < tablesList.length; i++) tables[tablesList[i]] = true;
  for (var j = 0; j < objects.length; j++) {
    var type = objects[j][0];
    var name = String(objects[j][1]);
    if (type === 'table') {
      if (tables[name]) continue;
      throw new ValidationError(context + ": unexpected sqlite_master object (type='table', name='" + name + "') — expected EXACTLY the " + tableWord + " tables + sqlite_autoindex entries");
    }
    // Autoindex entries must match the FULL shape sqlite emits, e.g.
    // 'sqlite_autoindex_generations_1' — 'sqlite_autoindex_evil' or
    // 'sqlite_autoindex_x_nope' are REJECTED like any other foreign object.
    if (type === 'index' && /^sqlite_autoindex_[A-Za-z0-9_]+_[0-9]+$/.test(name)) continue;
    if (type === 'trigger') {
      throw new ValidationError(context + ": sqlite_master object (type='trigger', name='" + name + "') rejected — triggers are forbidden in a lineage database");
    }
    if (type === 'view') {
      throw new ValidationError(context + ": sqlite_master object (type='view', name='" + name + "') rejected — views are forbidden in a lineage database");
    }
    throw new ValidationError(context + ": sqlite_master object (type='" + type + "', name='" + name + "') rejected — only sqlite_autoindex_* index entries are permitted");
  }
}

// ---------------------------------------------------------------------------
// Db — the persistence class. One instance per session/lineage; the working
// database is NEVER adopted/opened by two live instances. Constructor is
// cheap; call init() (fresh or init(bytes) adoption) before anything else.
// ---------------------------------------------------------------------------
class Db {
  constructor(opts) {
    opts = opts || {};
    this.db = null;                 // sql.js Database (the working in-memory db)
    this._fileHandle = opts.fileHandle || null; // browser FileSystemFileHandle or test stub
    this._hooks = { onBanner: null, persistSkipped: null };
    // _freshnessAtOpen: the tripwire baseline { size, sha256|null,
    // first4Kib|last4Kib } — recorded after a battery-passed adoption and
    // refreshed ONLY after a fully-verified persist succeeds (steps 2–6).
    // null means "no baseline yet" — the tripwire fails closed in that case.
    this._freshnessAtOpen = null;
    // _persistChain — same-tab serialization of persist() calls (F-5). Every
    // persist chains its pipeline onto this promise, so two same-tab persists
    // can NEVER interleave their export/verify/LSG/file-write steps, with or
    // without navigator.locks (which node / file:// / older browsers lack).
    // Layering: this in-instance chain (innermost, always present) < the
    // navigator.locks 'car_racer-db' exclusive lock taken inside the
    // pipeline (cross-tab ordering when supported). The chain never rejects
    // (settled with a swallowing catch) so one failure cannot wedge later
    // persists; each caller still sees ITS OWN promise resolve/reject as usual.
    this._persistChain = Promise.resolve();
  }

  // -------------------------------------------------------------------------
  // Capability detection / environment bridges (node + tests only / browser).
  // -------------------------------------------------------------------------

  // Db.loadNode(vendorDir) — NODE/TESTS ONLY. Reads the vendored
  // sql-wasm.js + .wasm from vendorDir, binds { wasmBinary } so sql.js NEVER
  // fetches anything, sets globalThis.SQL, returns the namespace. In the
  // browser this throws — use Db.createBrowser() there.
  static async loadNode(vendorDir) {
    var isNode = typeof process !== 'undefined' && process.versions && typeof process.versions.node === 'string';
    if (!isNode) {
      throw new DbOfflineError('Db.loadNode is node-only — in the browser use Db.createBrowser()');
    }
    if (typeof vendorDir !== 'string' || vendorDir.length === 0) {
      throw new ValidationError('Db.loadNode(vendorDir) requires a non-empty directory path');
    }
    var req;
    try {
      // require() exists in this module only when loaded as CJS (node) — the
      // guarded module.exports below proves that path; the browser never
      // reaches this branch (isNode gate above).
      req = require;
    } catch (err) {
      throw new DbOfflineError('Db.loadNode requires CommonJS require() — this copy of db.js was loaded as an ES module');
    }
    var path = req('node:path');
    var fs = req('node:fs');
    var wasmBinary = fs.readFileSync(path.join(vendorDir, 'sql-wasm.wasm'));
    var initSqlJs = req(path.join(vendorDir, 'sql-wasm.js'));
    if (typeof initSqlJs !== 'function') {
      throw new DbOfflineError('vendor/sql-wasm.js did not export the initSqlJs factory — refusing to load');
    }
    globalThis.SQL = await initSqlJs({ wasmBinary: wasmBinary });
    return globalThis.SQL;
  }

  // Db.createBrowser() — BROWSER. Requires vendor/sql-wasm.js to have run as
  // a classic <script> (it declares window.initSqlJs). file:// origins
  // REFUSE fetch()/XHR, so when vendor/sql-wasm-binary.js has defined
  // globalThis.SQL_WASM_BINARY_BASE64 (the embedded base64 of sql-wasm.wasm —
  // the documented script order is: sql-wasm-binary.js BEFORE sql-wasm.js),
  // the bytes are decoded here with atob() and bound as { wasmBinary } — no
  // network of any kind. Otherwise falls back to locateFile('vendor/…') for
  // HTTP serving. If BOTH paths are unavailable, DbOfflineError is thrown
  // with a reason naming both. No remote URL is ever used.
  static async createBrowser() {
    if (typeof globalThis.initSqlJs !== 'function') {
      throw new DbOfflineError('vendor/sql-wasm.js is missing or blocked — serve over localhost or fix the vendor files (see README)');
    }
    var b64 = typeof globalThis.SQL_WASM_BINARY_BASE64 === 'string' && globalThis.SQL_WASM_BINARY_BASE64.length > 0
      ? globalThis.SQL_WASM_BINARY_BASE64
      : null;
    if (b64) {
      try {
        var bin = atob(b64); // browsers + node ≥ 16 both expose atob
        var bytes = new Uint8Array(bin.length);
        for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        globalThis.SQL = await globalThis.initSqlJs({ wasmBinary: bytes });
        return globalThis.SQL;
      } catch (err) {
        // atob threw or wasm failed to instantiate from the embedded bytes —
        // fall through to the locateFile path; if that also fails, the
        // DbOfflineError below names BOTH paths.
        b64 = null;
      }
    }
    try {
      globalThis.SQL = await globalThis.initSqlJs({
        locateFile: function locateFile(file) { return 'vendor/' + file; },
      });
      return globalThis.SQL;
    } catch (err) {
      throw new DbOfflineError('sql-wasm-binary.js missing and vendor/ fetch unsupported here — serve over localhost or restore the vendor files (' + errorMessage(err) + ')');
    }
  }

  // -------------------------------------------------------------------------
  // init() — create the working database.
  //   init()        — fresh empty v4 schema (PRAGMA user_version = 4, seeded
  //                   meta with schema_version='4', wall_model='v6')
  //   init(bytes)   — ADOPT existing bytes (attack surface!): the ADOPTION
  //                   BATTERY runs FIRST; failure ⇒ zero adoption; then the
  //                   connection is opened, FK verified, sqlite_master is
  //                   scrubbed. Pre-v4 bytes are REFUSED by the battery (typed
  //                   WallModelChangedError, byte-preserving); uv4 bytes pass
  //                   the migration hook as a verified no-op. Post-COMMIT
  //                   verified set runs at the end
  //                   of init either way; the freshness baseline is recorded
  //                   FROM THE ADOPTED BYTES.
  // Always: PRAGMA foreign_keys = ON, and it is VERIFIED to have stuck —
  // per-connection, before any transaction — or init throws.
  // -------------------------------------------------------------------------
  async init(bytes) {
    if (this.db) {
      throw new Error('Db.init() called twice on the same instance — create a fresh Db per adoption/session');
    }
    var SQL = requireSql();

    // ---- Adoption path: battery BEFORE anything else. Failure ⇒ zero adoption.
    if (bytes !== undefined) {
      Db.validateAdoptionBytes(bytes); // throws on any violation — zero adoption
      var adoptedDigest;
      try {
        adoptedDigest = await this._fingerprintBytes(bytes);
      } catch (err) {
        // A fingerprint failure is an adoption FAILURE — typed ValidationError,
        // zero adoption, the working DB is untouched (arch F-5 / security M1).
        throw new ValidationError('init(bytes): fingerprint failed — refusing to adopt: ' + errorMessage(err));
      }
      var adopted;
      try {
        // Battery-passed bytes can STILL make sql.js throw (detached wasm-heap
        // views, OOM) — never let a raw escape out of here (security L6).
        adopted = new SQL.Database(bytes); // verified locally BEFORE this.db is touched
      } catch (err) {
        throw new ValidationError('init(bytes): database construction failed — refusing to adopt (working database untouched): ' + errorMessage(err));
      }
      try {
        this._enableAndVerifyForeignKeys(adopted);
        var rawUvRows = adopted.exec('PRAGMA user_version');
        var rawUv = rawUvRows && rawUvRows[0] && rawUvRows[0].values && rawUvRows[0].values[0]
          ? Number(rawUvRows[0].values[0][0])
          : NaN;
        // Belt-and-braces: adopted bytes are re-scrubbed on THIS connection —
        // five tables for uv>=3 raw bytes (the battery already refused a
        // tracks-less file and every pre-v4 lineage); four otherwise.
        // Anticipatory smuggles (a pre-carried tracks table in uv∈{1,2} bytes)
        // die at the battery's step 3b before ever reaching here.
        scrubSqliteMaster(adopted, 'init', rawUv >= 3 ? EXPECTED_TABLES_V3 : undefined);
        this._migrateIfNeeded(adopted);   // uv4 is a verified no-op; pre-v4 bytes were already refused by the battery
      } catch (err) {
        try { adopted.close(); } catch (closeErr) { /* best effort */ }
        throw err; // zero adoption — the working DB (none yet) is untouched and the instance is discarded
      }
      this.db = adopted;
      // init(bytes) adopts IN-MEMORY bytes — a handle attached at construction
      // can no longer describe the working DB; persist() would tripwire on
      // bytes it never saw (security L4).
      this._fileHandle = null;
      this._freshnessAtOpen = adoptedDigest; // adopting path owns the baseline reset (M-NEW5/F9)
      return this;
    }

    // ---- Fresh path: exact v4 schema (v3 DDL twins), user_version=4, seeded
    // meta with schema_version='4' and wall_model='v6'. NO tracks rows here —
    // ensureLineage()/forkLineage() insert the pin row and pre-materialize the
    // curriculum at lineage start; the battery's count arm treats an empty
    // lineage as legal (0 generations).
    this.db = new SQL.Database();
    // Fresh path creates an IN-MEMORY schema — a handle attached at
    // construction can no longer describe the working DB; persist() would
    // tripwire on bytes it never saw (security L4/F-17; fresh schema knows
    // nothing of any file — a constructor-supplied handle would persist a
    // false tripwire baseline).
    this._fileHandle = null;
    this._enableAndVerifyForeignKeys();
    var schemaStatements = [DDL_META, DDL_TRACKS, DDL_GENERATIONS, DDL_DRIVER_RESULTS, DDL_BEST_WEIGHTS];
    var begun = false;
    try {
      this.db.run('BEGIN');
      begun = true;
      for (var i = 0; i < schemaStatements.length; i++) {
        this.db.run(schemaStatements[i]);
      }
      this.db.run('PRAGMA user_version = 4');
      // Meta seed — schema_version, wall_model, architecture, app_version,
      // created_at. rng_state / track_hash / rotation_seed are deliberately
      // NOT seeded here: they are created by ensureLineage() at lineage start.
      this.db.run('INSERT INTO meta(key, value) VALUES (?, ?)', ['schema_version', '4']);
      this.db.run('INSERT INTO meta(key, value) VALUES (?, ?)', ['wall_model', 'v6']);
      this.db.run('INSERT INTO meta(key, value) VALUES (?, ?)', ['architecture', NN_ARCHITECTURE]);
      this.db.run('INSERT INTO meta(key, value) VALUES (?, ?)', ['app_version', '1.1.0']);
      this.db.run("INSERT INTO meta(key, value) VALUES (?, datetime('now'))", ['created_at']);
      this.db.run('COMMIT');
    } catch (err) {
      // BEGIN failure ⇒ begun === false ⇒ NO spurious ROLLBACK (logGeneration
      // F-6 template); rollback+close+null discipline otherwise unchanged and
      // data-at-risk-free (fresh schema — nothing was ever persisted).
      if (begun) {
        try { this.db.run('ROLLBACK'); } catch (rollbackErr) { /* best effort */ }
      }
      try { this.db.close(); } catch (closeErr) { /* best effort */ }
      this.db = null;
      throw err;
    }
    // Belt and braces: the exact-scrub runs on the connection EVERY init,
    // fresh or adopted — this is what refuses a pre-poisoned Database that
    // somehow reached us with extra objects.
    try {
      scrubSqliteMaster(this.db, 'init', EXPECTED_TABLES_V3);
      // Fresh v4 init gets the SAME post-COMMIT verified set as every adopted
      // instance — one uniform gate for every instance this factory produces.
      this._assertMigratedState(this.db, 'init (fresh v4)');
    } catch (err) {
      // FAIL CLOSED, like every adoption path — never leave this.db set on a
      // throw here (UI flips to setDbUnavailable; zombie allocation instead of
      // a half-trusted instance).
      try { this.db.close(); } catch (closeErr) { /* best effort */ }
      this.db = null;
      throw err;
    }
    return this;
  }

  // _enableAndVerifyForeignKeys(dbHandle, options) — set FK ON and READ BACK == 1.
  // options.mode selects the flavor:
  //   'init' (default) — plain Error, inline exec readback: the adoption/init
  //     gate; the message is the historical one and existing callers are
  //     untouched (they pass no options).
  //   'migration'       — MigrationError, _readForeignKeysPragma readback: the
  //     migration's post-COMMIT re-enable step (identical to the deleted twin).
  _enableAndVerifyForeignKeys(dbHandle, options) {
    var migration = !!(options && options.mode === 'migration');
    var target = dbHandle || this.db;
    target.run('PRAGMA foreign_keys = ON');
    var value;
    if (migration) {
      value = this._readForeignKeysPragma(target);
    } else {
      var rows = target.exec('PRAGMA foreign_keys');
      value = rows && rows[0] && rows[0].values && rows[0].values[0]
        ? rows[0].values[0][0]
        : null;
    }
    if (value !== 1) {
      if (migration) {
        throw new MigrationError('migration: PRAGMA foreign_keys did not re-enable after COMMIT (expected 1, got ' + JSON.stringify(value) + ') — the adopted instance is discarded');
      }
      throw new Error('init: PRAGMA foreign_keys did not stick (expected 1, got ' + JSON.stringify(value) + ') — refusing to continue (FK cascade is load-bearing)');
    }
    if (migration) return 1;
  }

  // _readForeignKeysPragma(target) — one readback helper shared by the
  // verified ON/OFF setters below. Returns the integer value (0|1) or null
  // when the pragma is unreadable — never throws.
  _readForeignKeysPragma(target) {
    try {
      var rows = target.exec('PRAGMA foreign_keys');
      var value = rows && rows[0] && rows[0].values && rows[0].values[0]
        ? rows[0].values[0][0]
        : null;
      return typeof value === 'number' ? value : null;
    } catch (err) {
      return null;
    }
  }

  // _enableAndVerifyForeignKeysOff(dbHandle) — set FK OFF and READ BACK == 0.
  // foreign_keys is a no-op INSIDE a transaction, so this MUST run before
  // BEGIN (the migration uses it) or the readback lies.
  _enableAndVerifyForeignKeysOff(dbHandle) {
    var target = dbHandle || this.db;
    target.run('PRAGMA foreign_keys = OFF');
    return this._verifyForeignKeysOff(dbHandle);
  }

  _verifyForeignKeysOff(dbHandle) {
    var target = dbHandle || this.db;
    var value = this._readForeignKeysPragma(target);
    if (value !== 0) {
      throw new MigrationError('migration: PRAGMA foreign_keys did not switch OFF (expected 0, got ' + JSON.stringify(value) + ') — refusing to run the v1 → v2 rebuild (FK is a no-op inside a transaction)');
    }
    return 0;
  }

  // _migrateIfNeeded(db) — the v4 gate. Called ONLY from the adoption/init
  // paths, after the ADOPTION BATTERY has already passed on these exact bytes
  // in the same call — migration is unreachable for battery-refused bytes and
  // never re-reads or re-opens the buffer. FIT-V6 closes ALL lineages: there
  // is no longer any v1 → v2 → v3 migration on the adoption path.
  //   uv==4 — VERIFIED NO-OP. The ADOPTION BATTERY already vetted these bytes
  //           in this same call (step 3b/3c uv/schema/wall_model + BOTH DDL
  //           twins, all value-domain arms, the self-hash arm, the five-table
  //           scrub and the tracks arms). No migration runs; the working DB
  //           is untouched. Fresh v4 init runs _assertMigratedState('init
  //           (fresh v4)') on its own path.
  //   uv<4  — refuse. Pre-v4 bytes were already refused byte-preservingly by
  //           the battery's typed WallModelChangedError; reaching this gate
  //           means a bypass/regression, so fail closed.
  // Any fault anywhere ⇒ the instance is discarded (init's catch-all closes
  // it) — zero adoption; the caller's bytes are never mutated.
  // Returns { migrated, userVersion }.
  _migrateIfNeeded(db) {
    var target = db || this.db;
    var uvRows = target.exec(USER_VERSION_READBACK_SQL);
    var uvValue = uvRows && uvRows[0] && uvRows[0].values && uvRows[0].values[0]
      ? Number(uvRows[0].values[0][0])
      : NaN;
    if (!Number.isFinite(uvValue)) {
      throw new MigrationError('migration: could not read user_version for the migration gate (got ' + JSON.stringify(uvValue) + ')');
    }
    if (uvValue === 4) {
      return { migrated: false, userVersion: 4 };
    }
    // Battery step 3b pins uv4 for adoption; any pre-v4 (or future) version
    // reaching this gate is a bypass — refuse to migrate or guess.
    throw new MigrationError('migration: user_version ' + uvValue + ' reached the v4 migration gate — pre-v4 lineages are refused byte-preservingly by the battery; refusing to migrate');
  }

  // migrateV1ToV2(db) — the ONE schema migration. Rebuilds `generations` in
  // place: CREATE generations_new (v2 DDL) → parity-checked copy → DROP →
  // RENAME → meta schema_version split (INSERT when absent, UPDATE when
  // present) → user_version = 2. NO legacy_alter_table. NO insert-or-conflict
  // clause of any kind (plain INSERT only, grep-gated).
  //
  // Order is pinned verbatim (SQLite's own 12-step procedure; RENAME-first
  // would corrupt the children's FK references):
  //   1. PRAGMA foreign_keys = OFF (BEFORE BEGIN) + readback == 0 — the
  //      pragma is a no-op inside a transaction, so it must flip first.
  //   2. BEGIN
  //   3. collision pre-assert: sqlite_master.name = 'generations_new' must
  //      count 0 — belt-and-braces against an interrupted prior attempt.
  //   4. CREATE TABLE generations_new (v2 DDL twin)
  //   5. parity BEFORE: count(*) FROM generations
  //   6. INSERT INTO generations_new (explicit list) SELECT (explicit list) FROM generations
  //   7. parity AFTER: count(*) FROM generations_new must equal parity BEFORE
  //   8. DROP TABLE generations
  //   9. ALTER TABLE generations_new RENAME TO generations
  //  10. meta schema_version — SELECT COUNT(*) … == 0 ⇒ INSERT, == 1 ⇒ UPDATE
  //      (a meta row is inserted by the fresh path in the same call for fresh
  //      DBs; an ADOPTED v1 file may legitimately lack it)
  //  11. PRAGMA user_version = 2
  //  12. COMMIT
  //  13. PRAGMA foreign_keys = ON + readback == 1 (AFTER COMMIT, never inside)
  //  14. _assertMigratedState() — the post-COMMIT verified set
  //
  // Failure anywhere ⇒ ROLLBACK (a failed rollback closes/discards the
  // instance) + typed MigrationError. The caller MUST discard the instance on
  // any throw — zero adoption; the bytes on disk / the working DB are never
  // touched by this method.
  migrateV1ToV2(db) {
    var target = db || this.db;
    var begun = false;
    try {
      this._enableAndVerifyForeignKeysOff(target);
      target.run('BEGIN');
      begun = true;
      // (3) collision pre-assert — BEFORE the create.
      var collisionRows = target.exec("SELECT COUNT(*) FROM sqlite_master WHERE name = 'generations_new'");
      var collisions = collisionRows && collisionRows[0] && collisionRows[0].values && collisionRows[0].values[0]
        ? Number(collisionRows[0].values[0][0])
        : NaN;
      if (collisions !== 0) {
        throw new MigrationError("migration aborted: 'generations_new' already exists in sqlite_master (count " + String(collisions) + ") — an earlier interrupted attempt may have left it behind; refusing to touch the adopted database");
      }
      // (4) CREATE the v2 twin (DDL_GENERATIONS_NEW_V2 — the v2 shape WITHOUT
      // the REFERENCES clause; the tracks table does not exist until the
      // v2 → v3 stage, and the v1 → v2 verified set requires EXACTLY the
      // four tables).
      target.run(DDL_GENERATIONS_NEW_V2);
      // (5) parity BEFORE.
      var beforeRows = target.exec('SELECT COUNT(*) FROM generations');
      var before = beforeRows && beforeRows[0] && beforeRows[0].values && beforeRows[0].values[0]
        ? Number(beforeRows[0].values[0][0])
        : NaN;
      if (!Number.isFinite(before)) {
        throw new MigrationError('migration: could not count generations before the copy (got ' + JSON.stringify(before) + ')');
      }
      // (6) the copy — EXPLICIT column list, both sides, ONE literal.
      target.run('INSERT INTO generations_new (generation, created_at, drivers_evaluated, best_fitness, avg_fitness, worst_fitness, best_driver_index, mutation_rate, mutation_scale, track_json, track_hash, sim_seconds, rng_state) SELECT generation, created_at, drivers_evaluated, best_fitness, avg_fitness, worst_fitness, best_driver_index, mutation_rate, mutation_scale, track_json, track_hash, sim_seconds, rng_state FROM generations');
      // (7) parity AFTER.
      var afterRows = target.exec('SELECT COUNT(*) FROM generations_new');
      var after = afterRows && afterRows[0] && afterRows[0].values && afterRows[0].values[0]
        ? Number(afterRows[0].values[0][0])
        : NaN;
      if (after !== before) {
        throw new MigrationError('migration parity check failed: generations held ' + before + ' rows, generations_new got ' + String(after) + ' — rolling back');
      }
      // (8) DROP the old table. Children (driver_results / best_weights) hold
      // FKs referencing generations — they were declared with ON DELETE
      // CASCADE but FK enforcement is OFF for the rebuild (SQLite's own
      // procedure); the children rows are NOT touched by this DROP.
      target.run('DROP TABLE generations');
      // (9) RENAME — the children's FK references still say "generations",
      // which is why the rebuild (not an in-place ALTER) is mandatory.
      target.run('ALTER TABLE generations_new RENAME TO generations');
      // (10) meta schema_version — plain INSERT/UPDATE split; meta exists
      // (the battery step 4 requires it with exactly columns key/value).
      // The split is decided by a SELECT; no conflict clause is used anywhere.
      var metaCountRows = target.exec("SELECT COUNT(*) FROM meta WHERE key = 'schema_version'");
      var metaCount = metaCountRows && metaCountRows[0] && metaCountRows[0].values && metaCountRows[0].values[0]
        ? Number(metaCountRows[0].values[0][0])
        : NaN;
      if (metaCount === 0) {
        target.run('INSERT INTO meta(key, value) VALUES (?, ?)', ['schema_version', '2']);
      } else if (metaCount === 1) {
        target.run("UPDATE meta SET value = '2' WHERE key = 'schema_version'");
      } else {
        throw new MigrationError("migration: meta has " + String(metaCount) + " rows for key 'schema_version' — expected 0 or 1; refusing to guess");
      }
      // (11) user_version = 2 (legal inside the transaction).
      target.run('PRAGMA user_version = 2');
      // (12) COMMIT.
      target.run('COMMIT');
      begun = false;
    } catch (err) {
      if (begun) {
        try { target.run('ROLLBACK'); } catch (rollbackErr) {
          // Rollback itself failed — the instance is unusable; discard it.
          try { target.close(); } catch (closeErr) { /* best effort */ }
        }
      }
      if (err && err.name === 'MigrationError') throw err;
      throw new MigrationError('migration failed: ' + errorMessage(err));
    }
    // (13) FK back ON — AFTER COMMIT — and verified == 1.
    this._enableAndVerifyForeignKeys(target, { mode: 'migration' });
    // (14) post-COMMIT verified set.
    this._assertMigratedState(target, 'migration (v1 → v2)');
    return { ok: true };
  }

  // migrateV2ToV3(db) — the v2 → v3 migration (the plan's §6 steps 1–11,
  // renumbered here). Rebuilds generations in place with the v3 twin
  // (REFERENCES tracks), backfills rotation-0 rows from the lineage's
  // earliest-track_hash rows (D-13), pre-materializes the adoption-time
  // curriculum (rotations 1..99 seeded with the meta.rng_state cursor, D-12),
  // and stamps meta.rotation_seed in the SAME transaction. NO
  // legacy_alter_table; NO insert-or-conflict clause of any kind (plain
  // INSERT only, grep-gated).
  //
  // Order is pinned verbatim:
  //   1. PRAGMA foreign_keys = OFF (BEFORE BEGIN) + readback == 0
  //   2. BEGIN
  //   3. collision pre-assert: BOTH 'tracks' and 'generations_new' absent
  //      (both reported when both fail)
  //   4. CREATE TABLE tracks (DDL_TRACKS verbatim) — BEFORE the rebuild:
  //      the generations copy must satisfy a parent that EXISTS at rename
  //      time, and backfill failures surface EARLIER in the transaction
  //   5. Backfill rotation-0 rows (D-13): for every DISTINCT
  //      generations.track_hash — corruption precondition FIRST (more than
  //      one distinct track_json ⇒ MigrationError BEFORE any INSERT), then
  //      self-hash arm on the EARLIEST generation's track_json, then INSERT
  //      (SELECT-then-split; source = 'default' iff the hash equals the
  //      frozen DEFAULT_CONTROL_POINTS hash else 'editor'; difficulty 0;
  //      created_at '2026-01-01 00:00:00')
  //   6. Read meta.rng_state cursor BEFORE any meta writes — missing cursor
  //      ⇒ typed MigrationError (a v1/v2 lineage ALWAYS carries it —
  //      ensureLineage pins it; missing ⇒ the file never ran through the
  //      pin path; refusing rather than fabricating a seed)
  //   7. Pre-materialize the curriculum (rotations 1..99): resolveGeneratedTrack
  //      per r, byte-verify (cyrb53 == the row's own track_hash; mismatch ⇒
  //      refuse), INSERT via SELECT-then-split (source='generated',
  //      difficulty=r). rotation 0 is the (5) INSERT — a no-op re-pin via
  //      SELECT-then-split.
  //   8. CREATE TABLE generations_new — the v3 twin (REFERENCES tracks)
  //   9. parity BEFORE; 10. explicit-column INSERT..SELECT; 11. parity AFTER
  //  12. DROP TABLE generations; ALTER TABLE generations_new RENAME TO generations
  //  13. meta schema_version split → '3' (0 ⇒ INSERT, 1 ⇒ UPDATE, else refuse)
  //  14. meta rotation_seed split — the adoption-time cursor (0 ⇒ INSERT,
  //      1 ⇒ UPDATE, else refuse) — D-11
  //  15. PRAGMA user_version = 3; COMMIT
  //  16. PRAGMA foreign_keys = ON + readback == 1 (AFTER COMMIT)
  //  17. _assertMigratedState(target, 'migration (v2 → v3)')
  //
  // Failure anywhere ⇒ ROLLBACK (a failed rollback closes/discards the
  // instance) + typed MigrationError. The caller MUST discard the instance on
  // any throw — zero adoption; the bytes on disk / the working DB are never
  // touched by this method.
  migrateV2ToV3(db) {
    var target = db || this.db;
    var begun = false;
    try {
      this._enableAndVerifyForeignKeysOff(target);
      target.run('BEGIN');
      begun = true;
      // (3) collision pre-assert — BOTH absent; report both if both fail.
      var collisionRows = target.exec("SELECT name FROM sqlite_master WHERE name IN ('tracks', 'generations_new') ORDER BY name");
      var collisionNames = ((collisionRows && collisionRows[0] && collisionRows[0].values) || []).map(function (r) { return String(r[0]); });
      if (collisionNames.length !== 0) {
        throw new MigrationError("migration (v2 → v3) aborted: sqlite_master already carries " + (collisionNames.length === 1 ? "'" + collisionNames[0] + "'" : "both '" + collisionNames.join("' and '") + "'") + " — an earlier interrupted attempt may have left them behind; refusing to touch the adopted database");
      }
      // (4) CREATE TABLE tracks — verbatim, BEFORE the generations rebuild.
      target.run(DDL_TRACKS);
      // (5) Backfill rotation-0 rows (D-13).
      var distinctRows = target.exec('SELECT DISTINCT track_hash FROM generations');
      var distinctValues = (distinctRows && distinctRows[0] && distinctRows[0].values) || [];
      for (var di = 0; di < distinctValues.length; di++) {
        var hash = String(distinctValues[di][0]);
        // Corruption precondition — re-checked BEFORE any tracks INSERT.
        var distinctJsonCount = countScalar(target, 'SELECT COUNT(DISTINCT track_json) FROM generations WHERE track_hash = ?', [hash]);
        if (distinctJsonCount > 1) {
          throw new MigrationError('migration (v2 → v3): track_hash ' + JSON.stringify(hash) + ' is carried by generations with ' + distinctJsonCount + ' DISTINCT track_json payloads — divergent bytes for one hash; refusing to backfill (zero adoption)');
        }
        var minRows = target.exec('SELECT MIN(generation) FROM generations WHERE track_hash = ?', [hash]);
        var minGen = minRows && minRows[0] && minRows[0].values && minRows[0].values[0]
          ? Number(minRows[0].values[0][0])
          : NaN;
        if (!Number.isFinite(minGen)) {
          throw new MigrationError('migration (v2 → v3): could not read MIN(generation) for track_hash ' + JSON.stringify(hash));
        }
        // Earliest generation wins (MIN(generation) — NEVER MIN(track_json)).
        var earlyRows = target.exec('SELECT track_json FROM generations WHERE generation = ?', [minGen]);
        var early = earlyRows && earlyRows[0] && earlyRows[0].values && earlyRows[0].values[0];
        var trackJson = early ? String(early[0]) : null;
        if (!trackJson) {
          throw new MigrationError('migration (v2 → v3): could not read the earliest track_json for track_hash ' + JSON.stringify(hash));
        }
        var parsed;
        try {
          parsed = parseTrackJsonStrict(trackJson, 'migration (v2 → v3)');
        } catch (err) {
          throw new MigrationError('migration (v2 → v3): earliest track_json for track_hash ' + JSON.stringify(hash) + ' rejected — ' + errorMessage(err));
        }
        // Self-hash arm (D-14(1)) — against the SAME earliest row.
        var recombined;
        try {
          recombined = recombineTrackHash(parsed.points, parsed.halfWidth, 'migration (v2 → v3)');
        } catch (err) {
          throw new MigrationError('migration (v2 → v3): earliest track_json for track_hash ' + JSON.stringify(hash) + ' rejected — ' + errorMessage(err));
        }
        if (recombined !== hash) {
          throw new MigrationError('migration (v2 → v3): earliest generation row for track_hash ' + JSON.stringify(hash) + ' (generation ' + minGen + ') recombined to ' + recombined + ' — hash/bytes divergence; refusing to backfill (zero adoption)');
        }
        insertTrackRowIfAbsent(target, {
          track_hash: hash,
          source: sourceForHash(hash),
          control_points: controlPointsJson(parsed.points),
          half_width: parsed.halfWidth,
          difficulty: 0,
          created_at: '2026-01-01 00:00:00',
        }, 'migration (v2 → v3)');
      }
      // (6) Read meta.rng_state cursor BEFORE any meta writes.
      var cursorRows = target.exec("SELECT value FROM meta WHERE key = 'rng_state'");
      var cursorRaw = (cursorRows && cursorRows[0] && cursorRows[0].values && cursorRows[0].values[0])
        ? cursorRows[0].values[0][0]
        : null;
      if (cursorRaw === null || cursorRaw === undefined) {
        throw new MigrationError('migration (v2 → v3): meta.rng_state is missing — every adoptable lineage carries the lineage pin (ensureLineage); refusing to fabricate a rotation seed (zero adoption)');
      }
      var cursorState = strictUint32(cursorRaw, 'migration (v2 → v3) rotation_seed cursor');
      // FIT-V4 (N2/N4): a crafted v2 file may carry a stray generator_version
      // key (adoption's allowlist ACCEPTS '2' — only invalid values throw at
      // the battery at db.js:924, BEFORE this migration at db.js:952). Migration
      // writes NO version key: DELETE any stray one in THIS transaction so the
      // permissive meta read after adoption sees absent ⇒ V1, and gen 51 crosses
      // with the V1 hash and zero errors.
      target.run("DELETE FROM meta WHERE key = 'generator_version'");
      // (7) Pre-materialize the curriculum — rotations 1..99, byte-verified.
      // The override FULLY replaces the db-derived version for materialization
      // AND the post-invariant (N4) — the db meta read is skipped on this path.
      preMaterializeCurriculum(target, cursorState, 'migration (v2 → v3)', versionOne());
      // (8) CREATE the v3 twin (REFERENCES tracks) — AFTER tracks exists.
      target.run(DDL_GENERATIONS_NEW);
      // (9) parity BEFORE.
      var beforeRows = target.exec('SELECT COUNT(*) FROM generations');
      var before = beforeRows && beforeRows[0] && beforeRows[0].values && beforeRows[0].values[0]
        ? Number(beforeRows[0].values[0][0])
        : NaN;
      if (!Number.isFinite(before)) {
        throw new MigrationError('migration (v2 → v3): could not count generations before the copy (got ' + JSON.stringify(before) + ')');
      }
      // (10) the copy — EXPLICIT column list, both sides, ONE literal.
      target.run('INSERT INTO generations_new (generation, created_at, drivers_evaluated, best_fitness, avg_fitness, worst_fitness, best_driver_index, mutation_rate, mutation_scale, track_json, track_hash, sim_seconds, rng_state) SELECT generation, created_at, drivers_evaluated, best_fitness, avg_fitness, worst_fitness, best_driver_index, mutation_rate, mutation_scale, track_json, track_hash, sim_seconds, rng_state FROM generations');
      // (11) parity AFTER.
      var afterRows = target.exec('SELECT COUNT(*) FROM generations_new');
      var after = afterRows && afterRows[0] && afterRows[0].values && afterRows[0].values[0]
        ? Number(afterRows[0].values[0][0])
        : NaN;
      if (after !== before) {
        throw new MigrationError('migration (v2 → v3) parity check failed: generations held ' + before + ' rows, generations_new got ' + String(after) + ' — rolling back');
      }
      // (12) DROP + RENAME (the v2 procedure verbatim).
      target.run('DROP TABLE generations');
      target.run('ALTER TABLE generations_new RENAME TO generations');
      // (13) meta schema_version — plain INSERT/UPDATE split → '3'.
      var metaCountRows = target.exec("SELECT COUNT(*) FROM meta WHERE key = 'schema_version'");
      var metaCount = metaCountRows && metaCountRows[0] && metaCountRows[0].values && metaCountRows[0].values[0]
        ? Number(metaCountRows[0].values[0][0])
        : NaN;
      if (metaCount === 0) {
        target.run('INSERT INTO meta(key, value) VALUES (?, ?)', ['schema_version', '3']);
      } else if (metaCount === 1) {
        target.run("UPDATE meta SET value = '3' WHERE key = 'schema_version'");
      } else {
        throw new MigrationError("migration (v2 → v3): meta has " + String(metaCount) + " rows for key 'schema_version' — expected 0 or 1; refusing to guess");
      }
      // (14) meta rotation_seed — the adoption-time cursor (D-11). Plain
      // INSERT/UPDATE split; absent ⇒ inserted, present ⇒ overwritten with
      // the cursor (the plan's exact adoption semantic), >1 ⇒ refuse.
      var seedCountRows = target.exec("SELECT COUNT(*) FROM meta WHERE key = 'rotation_seed'");
      var seedCount = seedCountRows && seedCountRows[0] && seedCountRows[0].values && seedCountRows[0].values[0]
        ? Number(seedCountRows[0].values[0][0])
        : NaN;
      if (seedCount === 0) {
        target.run('INSERT INTO meta(key, value) VALUES (?, ?)', ['rotation_seed', String(cursorState)]);
      } else if (seedCount === 1) {
        target.run('UPDATE meta SET value = ? WHERE key = ?', [String(cursorState), 'rotation_seed']);
      } else {
        throw new MigrationError("migration (v2 → v3): meta has " + String(seedCount) + " rows for key 'rotation_seed' — expected 0 or 1; refusing to guess");
      }
      // (15) PRAGMA user_version = 3; COMMIT.
      target.run('PRAGMA user_version = 3');
      target.run('COMMIT');
      begun = false;
    } catch (err) {
      if (begun) {
        try { target.run('ROLLBACK'); } catch (rollbackErr) {
          try { target.close(); } catch (closeErr) { /* best effort */ }
        }
      }
      if (err && err.name === 'MigrationError') throw err;
      throw new MigrationError('migration (v2 → v3) failed: ' + errorMessage(err));
    }
    // (16) FK back ON — AFTER COMMIT — and verified == 1.
    this._enableAndVerifyForeignKeys(target, { mode: 'migration' });
    // (17) post-COMMIT verified set — EVERY arm runs.
    this._assertMigratedState(target, 'migration (v2 → v3)');
    return { ok: true };
  }

  // _assertMigratedCore(target, ctx, expectedTables, uv, schemaVersion) — the
  // shared body of the verified set: the v1 → v2 (legacy four-table/v2) and
  // v3 cores differ ONLY in the three pinned values (the expected master
  // object-set and the uv/schema_version literals) — the SEC-18 scrub is
  // identical at both gates. Never weakens the v2 path.
  _assertMigratedCore(target, ctx, expectedTables, uv, schemaVersion) {
    // (a) FK is ON and verified.
    var fkRows = target.exec('PRAGMA foreign_keys');
    var fkValue = fkRows && fkRows[0] && fkRows[0].values && fkRows[0].values[0]
      ? fkRows[0].values[0][0]
      : null;
    if (fkValue !== 1) {
      throw new MigrationError(ctx + ': PRAGMA foreign_keys is ' + JSON.stringify(fkValue) + ' after migration (expected 1) — discarding the instance');
    }
    // (b) foreign_key_check — zero rows (no dangling child references).
    var fkcRows = target.exec('PRAGMA foreign_key_check');
    var fkcValues = (fkcRows && fkcRows[0] && fkcRows[0].values) ? fkcRows[0].values : [];
    if (fkcValues.length !== 0) {
      throw new MigrationError(ctx + ': PRAGMA foreign_key_check returned ' + fkcValues.length + ' row(s) after migration — a child row references a missing generation; discarding the instance');
    }
    // (c) integrity_check == 'ok'.
    var ic = target.exec('PRAGMA integrity_check');
    var icValue = ic && ic[0] && ic[0].values && ic[0].values[0]
      ? ic[0].values[0][0]
      : null;
    if (icValue !== 'ok') {
      throw new MigrationError(ctx + ': PRAGMA integrity_check returned ' + JSON.stringify(icValue) + " (expected 'ok') — discarding the instance");
    }
    // (d) the declared FKs — exactly one each, generations, CASCADE.
    checkSingleFk(target, 'driver_results', ctx);
    checkSingleFk(target, 'best_weights', ctx);
    // (e) sqlite_master object-set equality (the SEC-18 scrub, verbatim).
    scrubSqliteMaster(target, ctx, expectedTables);
    // (f) user_version == uv (table-valued pragma readback).
    var uvRows = target.exec(USER_VERSION_READBACK_SQL);
    var uvValue = uvRows && uvRows[0] && uvRows[0].values && uvRows[0].values[0]
      ? Number(uvRows[0].values[0][0])
      : NaN;
    if (uvValue !== uv) {
      throw new MigrationError(ctx + ': user_version is ' + String(uvValue) + ' after migration (expected ' + uv + ') — discarding the instance');
    }
    // (g) meta.schema_version == schemaVersion.
    var schemaRows = target.exec("SELECT value FROM meta WHERE key = 'schema_version'");
    var schemaValue = schemaRows && schemaRows[0] && schemaRows[0].values && schemaRows[0].values[0]
      ? schemaRows[0].values[0][0]
      : null;
    if (schemaValue !== schemaVersion) {
      throw new MigrationError(ctx + ": meta.schema_version is " + JSON.stringify(schemaValue) + " after migration (expected '" + schemaVersion + "') — discarding the instance");
    }
  }

  // _assertMigratedState(db, context) — the POST-COMMIT VERIFIED SET.
  // Every condition is mandatory; failure ⇒ throw (the caller discards the
  // instance — zero adoption, Manual + banner). The verified set BRANCHES on
  // the connection's user_version:
  //   uv==2 — EXACTLY the historical v1 → v2 arms (four-table scrub,
  //           uv==2, schema_version=='2') — byte-compatible behavior for the
  //           legacy stage of the chain.
  //   uv==4 — FIT-V6: the v3 arms below, pinned to uv==4 /
  //           schema_version=='4' PLUS meta.wall_model=='v6' (the wall-model
  //           break marker). Reuses the v3 DDL twins and table/column maps.
  //   uv==3 — the v3 arms (kept for the standalone v1→v2→v3 migration
  //           functions only; adoption never produces uv3):
  //     v1 → v2-inherited core arms (a)-(g), pinned to the five tables,
  //     uv==3, schema_version=='3' — UNCONDITIONAL (no pinnable-off flag);
  //     v3-specific arms ALWAYS run:
  //     (h) generations-FK dedicated arm — exactly one FK
  //         (table='tracks', from='track_hash', to='track_hash') with
  //         on_delete ∉ {SET NULL, SET DEFAULT, CASCADE} (only '' or
  //         'NO ACTION' legal — tracks rows are never deleted);
  //     (i) rotation_seed pin-detect — iff lineage is pinned (rng_state
  //         exists), rotation_seed MUST exist and be a strict uint32;
  //     (j) count arm — every generations.track_hash resolves in tracks;
  //     (k) tracks recombination arm — every tracks row's whole-row
  //         recombined hash == its track_hash;
  //     (l) value domains — every tracks row is well-typed and in-range
  //         (difficulty integer ⊆ [0,100]; half_width IN {integer,real} ⊆ [1,2000];
  //         source enum; track_hash text; created_at TEXT; control_points text);
  //     (m) adoption semantics — generations value-domain arms (the same 17 arms
  //         the adoption battery runs — fitness finite-but-signed, integers in
  //         range, 0/1 flags, crashed/stuck∈{0,1}) + children single-FK re-assert;
  //     (n) generations mirror — every generations row's recombined hash
  //         == its track_hash.
  // No flags, no escape hatches: every call to this runs every arm.
  _assertMigratedState(db, context) {
    var ctx = context || 'migration';
    var target = db || this.db;
    var uvRows = target.exec(USER_VERSION_READBACK_SQL);
    var uvValue = uvRows && uvRows[0] && uvRows[0].values && uvRows[0].values[0]
      ? Number(uvRows[0].values[0][0])
      : NaN;
    if (!Number.isFinite(uvValue)) {
      throw new MigrationError(ctx + ': could not read user_version for the verified set (got ' + JSON.stringify(uvValue) + ') — discarding the instance');
    }
    if (uvValue === 2) {
      // Legacy v1 → v2 stage — EXACTLY today's arms.
      this._assertMigratedCore(target, ctx, undefined, 2, '2');
      // FIT-V4: allowlist applies here too (a '2' claim parses as a valid
      // value; genuinely invalid values refuse). The key is deleted later,
      // in the v2 → v3 stage's single preMaterialize call.
      try { checkGeneratorVersionAllowlist(target, ctx); }
      catch (err) { throw new MigrationError(ctx + ': ' + errorMessage(err)); }
      return { ok: true };
    }
    if (uvValue !== 3 && uvValue !== 4) {
      throw new MigrationError(ctx + ': user_version is ' + String(uvValue) + ' — the verified set expects 2 (legacy v1 → v2), 3 or 4');
    }
    var expectedUv = uvValue;
    var expectedSchema = uvValue === 4 ? '4' : '3';
    var wrap = function (err) {
      if (err && err.name === 'MigrationError') return err;
      return new MigrationError(ctx + ': ' + errorMessage(err));
    };
    // ---- v1 → v2-inherited core arms (UNCONDITIONAL — no pinnable-off escape
    // hatch: a weakened verified set never ships silently; reviewer F-3).
    this._assertMigratedCore(target, ctx, EXPECTED_TABLES_V3, expectedUv, expectedSchema);
    // FIT-V6 — the wall-model break marker. uv4 pins 'v6'; uv3 predates it.
    if (uvValue === 4) {
      var wmRows = target.exec("SELECT value FROM meta WHERE key = 'wall_model'");
      var wmValue = wmRows && wmRows[0] && wmRows[0].values.length
        ? wmRows[0].values[0][0]
        : null;
      if (wmValue !== 'v6') {
        throw new MigrationError(ctx + ": meta.wall_model is " + JSON.stringify(wmValue) + " after init (expected 'v6') — discarding the instance");
      }
    }
    // FIT-V4: generator_version allowlist + (for a V2-pinned lineage) the
    // scoped 99-row verification. Migration deletes any stray key, so a
    // migrated legacy file parses to V1 here and skips the V2-row arm (N2).
    try { checkGeneratorVersionAllowlist(target, ctx); }
    catch (err) { throw wrap(err); }
    try { checkV2PinnedCurriculumRows(target, ctx); }
    catch (err) { throw wrap(err); }
    // ---- v3-specific arms (always).
    // (h) generations-FK dedicated arm.
    try {
      checkGenerationsForeignKey(target, ctx);
    } catch (err) {
      throw wrap(err);
    }
    // (i) rotation_seed pin-detect — iff lineage is pinned (rng_state
    // exists), rotation_seed must exist and be a strict uint32. An unpinned
    // (fresh, pre-ensureLineage) lineage is fine without the seed.
    var pinned = countScalar(target, "SELECT COUNT(*) FROM meta WHERE key = 'rng_state'");
    if (pinned > 0) {
      var seedRows = target.exec("SELECT value FROM meta WHERE key = 'rotation_seed'");
      var seedValue = seedRows && seedRows[0] && seedRows[0].values && seedRows[0].values[0]
        ? seedRows[0].values[0][0]
        : null;
      if (seedValue === null || seedValue === undefined) {
        throw new MigrationError(ctx + ": meta.rotation_seed is missing although meta.rng_state exists (the lineage is pinned) — the v2 → v3 migration writes it at adoption; discarding the instance");
      }
      try {
        strictUint32(seedValue, 'meta.rotation_seed');
      } catch (err) {
        throw new MigrationError(ctx + ': meta.rotation_seed is not a strict uint32: ' + errorMessage(err) + ' — discarding the instance');
      }
    }
    // (j) count arm — every generations.track_hash resolves in tracks.
    try {
      checkDanglingGenerations(target, ctx);
    } catch (err) {
      throw wrap(err);
    }
    // (k) tracks recombination arm — every tracks row's whole-row hash matches.
    try {
      checkTracksContentAndHash(target, ctx);
    } catch (err) {
      throw wrap(err);
    }
    // (l) value domains — every tracks row is well-typed and in-range BEFORE
    //     the caller relies on it (mirrors the battery's 10b arms so a
    //     mid-migration corruption that changed a difficulty survives re-adoption);
    //     ~6 COUNT queries on ≤ 101 tracks rows.
    try {
      checkTracksValueDomain(target, ctx);
    } catch (err) {
      throw wrap(err);
    }
    // (m) generations value domains + children single-FK assertions (the
    //     adoption battery's exact semantic arm — same 17 arms, same order;
    //     applied to the post-migration file as well so a mid-migration
    //     integrity slip is caught twice: at migration commit AND at any
    //     subsequent adoption).
    try {
      checkAdoptionSemantics(target, ctx);
    } catch (err) {
      throw wrap(err);
    }
    // (l) generations mirror — every generations row's self-hash matches.
    try {
      checkGenerationsSelfHash(target, ctx);
    } catch (err) {
      throw wrap(err);
    }
    return { ok: true };
  }

  _requireDb() {
    if (!this.db) {
      throw new ValidationError('Db not initialized — call init() (or init(bytes)) first');
    }
    return this.db;
  }

  _metaValue(key) {
    var rows = this.db.exec('SELECT value FROM meta WHERE key = ?', [key]);
    if (!rows || rows.length === 0 || rows[0].values.length === 0) return null;
    return rows[0].values[0][0];
  }

  _setMetaValue(key, value) {
    // Plain INSERT-or-UPDATE is decided by a SELECT first — there is NO
    // upsert spelling anywhere in this file (grep-gated).
    var existing = this._metaValue(key);
    if (existing === null) {
      this.db.run('INSERT INTO meta(key, value) VALUES (?, ?)', [key, value]);
    } else {
      this.db.run('UPDATE meta SET value = ? WHERE key = ?', [value, key]);
    }
  }

  // -------------------------------------------------------------------------
  // Db.validateAdoptionBytes(bytes) — THE ADOPTION BATTERY (SEC-22).
  // The single choke point EVERY bytes-adoption path funnels through:
  //   init(bytes) · openExisting() · importBytes() · persist step-2 verify
  //   (via _verifyForPersist, the lighter form) · the M4 reconcile/reload /
  //   cache-offer paths (they call this before offering ANY adopted bytes).
  // THROWS ValidationError on ANY violation — the caller must treat a throw
  // as "zero adoption; working DB / IDB / file untouched; blocking banner".
  // Returns { generation, architecture, trackHash, rngState } — the coherence
  // summary the reconcile UI shows.
  //
  // Order is load-bearing and deliberate: cheap pre-parse gate FIRST (never
  // open sqlite for an over-cap file), then a THROWAWAY instance that is
  // NEVER the live one, integrity_check, uv gating (uv∈{1,2,3} ⇒ typed
  // WallModelChangedError; uv∉{1,2,3,4} ⇒ generic; uv==4 proceeds), required
  // tables + exact column sets (five-table with tracks for uv>=3; the uv∈{1,2}
  // four-table arms are unreachable because step 3b refuses pre-v4 first),
  // architecture, row-count caps, per-row weights validation via
  // NeuralNetwork.fromJSON (the same validator the constructor uses — 434
  // finite floats, |v| <= 1e6, architecture string), the track_json cap, the
  // self-hash arm (step 8b), the sqlite_master scrub (five-table object-set
  // for uv>=3; zero triggers, zero views, zero foreign objects), and finally
  // the SEMANTIC arms (step 10a: declared foreign keys + value domains; step
  // 10b for uv>=3: the tracks arms). uv4 additionally pins
  // meta.wall_model=='v6'.
  //
  // Step 10a — the DECLARED foreign keys (PRAGMA foreign_key_list) on
  // driver_results/best_weights and the value domains of every adopted
  // generations/driver_results row (fitness finite but SIGNED; the
  // structural columns finite >= 0 / integer ranges / 0-1 flags). A
  // hand-crafted file with the exact column NAME sets but no FK /
  // non-cascade FK / out-of-domain values is refused here.
  //
  // Step 10b (uv>=3) — the generations self-hash arm runs in step 8b;
  // here: the count arm (no dangling generations.track_hash), the tracks
  // value-domain arms (typeof-strict), the control_points content arm, the
  // whole-row recombination arm and the generations-FK arm (exactly one FK,
  // tracks(track_hash), NO ACTION). The SAME invariants are also asserted by
  // _assertMigratedState (the verified set) — see the verified-set section.
  //
  // NOTE (documented behavior): the battery catches NaN/Infinity written by
  // an older/foreign build — via the value-domain arms (NOT (x = x) for
  // NaN; x > DBL_MAX for +Inf) plus logGeneration's write-time gate. A
  // pre-poisoned file can no longer adopt (F-8).
  // -------------------------------------------------------------------------
  static validateAdoptionBytes(bytes) {
    var NeuralNetwork = requireNeuralNetwork();

    // Step 1 — pre-parse gate. Reject BEFORE opening sqlite at all.
    if (!(bytes instanceof Uint8Array)) {
      throw new ValidationError('adoption battery: bytes must be a Uint8Array, got ' + (bytes === null ? 'null' : typeof bytes));
    }
    if (bytes.length === 0) {
      throw new ValidationError('adoption battery: bytes are empty (0 bytes) — refusing to adopt');
    }
    if (bytes.length > MAX_DB_BYTES) {
      throw new ValidationError('adoption battery: bytes length ' + bytes.length + ' exceeds the ' + MAX_DB_BYTES + '-byte (64 MB) pre-read cap');
    }

    // Step 2 — throwaway instance ONLY. The live working DB is never used
    // here, so a battery failure can never corrupt or even touch it.
    var SQL = requireSql();
    var throwaway = null;
    try {
      throwaway = new SQL.Database(bytes);
    } catch (err) {
      throw new ValidationError('adoption battery: failed to open the bytes as a SQLite database: ' + errorMessage(err));
    }

    try {
      // Step 3 — integrity_check must be exactly 'ok'.
      var ic = throwaway.exec('PRAGMA integrity_check');
      var icValue = ic && ic[0] && ic[0].values && ic[0].values[0]
        ? ic[0].values[0][0]
        : null;
      if (icValue !== 'ok') {
        throw new ValidationError("adoption battery: integrity_check returned " + JSON.stringify(icValue) + " (expected 'ok') — refusing adoption");
      }

      // Step 3a — PRAGMA foreign_key_check must be EMPTY (no orphans) on any
      // adopted bytes, v1 or v2 alike: the battery refuses orphans everywhere,
      // not just at migration time. (ADR-0001 accepted orphans in the past;
      // this closes the hole rather than document it.)
      var fkcRows = throwaway.exec('PRAGMA foreign_key_check');
      if (fkcRows && fkcRows[0] && fkcRows[0].values && fkcRows[0].values.length > 0) {
        throw new ValidationError('adoption battery: foreign_key_check returned ' + JSON.stringify(fkcRows[0].values) + ' — refusing adoption (orphan rows not permitted)');
      }

      // Step 3b — FIT-V6 break gate. uv∈{1,2,3} is a lineage built under the
      // OLD wall model: refuse with a TYPED WallModelChangedError that carries
      // the refused bytes verbatim (byte-preserving; no storage is cleared).
      // uv∉{1,2,3,4} (e.g. 999) is a foreign file → generic ValidationError.
      // Only uv==4 proceeds. This runs BEFORE any scrub/DDL arm so a pre-v4
      // file with a tracks table still gets the typed error, not a scrub one.
      var uvRows = throwaway.exec('PRAGMA user_version');
      var uvValue = uvRows && uvRows[0] && uvRows[0].values && uvRows[0].values[0]
        ? Number(uvRows[0].values[0][0])
        : NaN;
      if (uvValue === 1 || uvValue === 2 || uvValue === 3) {
        throw newWallModelChangedError(bytes);
      }
      if (uvValue !== 4) {
        throw new ValidationError('adoption battery: PRAGMA user_version ' + String(uvValue) + ' — refusing (expected 4; pre-v4 lineages are refused with WallModelChangedError)');
      }

      // Step 3c — the DDL twin readbacks. Folding is lossless for the DDL's
      // literals (lowercase + \s+ collapse + trailing-`;` strip + double-
      // quote strip — both sides). The quote strip exists because SQLite's
      // ALTER TABLE … RENAME stores the renamed table's sql with a QUOTED
      // table name ('CREATE TABLE "generations"(…') — the migration's own
      // output must re-adopt; identifiers in these DDLs never carry meaning
      // as double-quoted literals. Step 3b guarantees only uv==4 reaches here
      // (uv∈{1,2,3} typed-refused, uv∉{1,2,3,4} generic-refused), so the
      // uv>=3 arms pin schema_version=='4' + the v3 generations twin +
      // meta.wall_model=='v6' + the pinned tracks twin. The uv==2 arm is kept
      // (historical shape) but is unreachable on the adoption path.
      var generationsDdl = '';
      var tracksDdl = '';
      if (Number(uvValue) >= 2) {
        var svRows = throwaway.exec("SELECT value FROM meta WHERE key = 'schema_version'");
        var svValue = svRows && svRows[0] && svRows[0].values && svRows[0].values[0]
          ? String(svRows[0].values[0][0])
          : null;
        var expectedSchema = Number(uvValue) === 2 ? '2' : String(uvValue); // '2' or '4'
        if (svValue !== expectedSchema) {
          throw new ValidationError("adoption battery: user_version=" + uvValue + " but meta.schema_version is " + JSON.stringify(svValue) + " (expected '" + expectedSchema + "') — refusing adoption");
        }
        var ddlRows = throwaway.exec("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'generations'");
        generationsDdl = ddlRows && ddlRows[0] && ddlRows[0].values && ddlRows[0].values[0]
          ? String(ddlRows[0].values[0][0] || '')
          : '';
        var normDdl = generationsDdl.toLowerCase().replace(/\s+/g, '').replace(/;+$/, '').replace(/"/g, '');
        var expectedTwin = Number(uvValue) === 2 ? DDL_GENERATIONS_V2 : DDL_GENERATIONS;
        var normTwin = expectedTwin.toLowerCase().replace(/\s+/g, '').replace(/;+$/, '').replace(/"/g, '');
        if (normDdl !== normTwin) {
          throw new ValidationError("adoption battery: user_version=" + uvValue + " generations DDL does not match this build's native " + (Number(uvValue) === 2 ? 'v2' : 'v3') + " schema — refusing adoption");
        }
        // FIT-V6 — the wall-model break marker (uv4 pins 'v6').
        if (Number(uvValue) === 4) {
          var wmRows = throwaway.exec("SELECT value FROM meta WHERE key = 'wall_model'");
          var wmValue = wmRows && wmRows[0] && wmRows[0].values && wmRows[0].values[0]
            ? String(wmRows[0].values[0][0])
            : null;
          if (wmValue !== 'v6') {
            throw new ValidationError("adoption battery: user_version=4 but meta.wall_model is " + JSON.stringify(wmValue) + " (expected 'v6') — refusing adoption");
          }
        }
      }
      if (Number(uvValue) >= 3) {
        var tracksDdlRows = throwaway.exec("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'tracks'");
        tracksDdl = tracksDdlRows && tracksDdlRows[0] && tracksDdlRows[0].values && tracksDdlRows[0].values[0]
          ? String(tracksDdlRows[0].values[0][0] || '')
          : '';
        var normTracks = tracksDdl.toLowerCase().replace(/\s+/g, '').replace(/;+$/, '').replace(/"/g, '');
        var normTracksTwin = DDL_TRACKS.toLowerCase().replace(/\s+/g, '').replace(/;+$/, '').replace(/"/g, '');
        if (normTracks !== normTracksTwin) {
          throw new ValidationError("adoption battery: user_version=" + uvValue + " tracks DDL does not match this build's native v3 schema — refusing adoption");
        }
      }

      // Step 4 — required tables EXIST with the exact expected column sets.
      // uv>=3 ⇒ the five-table set (with tracks); uv∈{1,2} ⇒ the historical
      // four-table set (unreachable on the adoption path after step 3b).
      var fiveTable = Number(uvValue) >= 3;
      var columnsMap = fiveTable ? EXPECTED_COLUMNS_V3 : EXPECTED_COLUMNS;
      for (var table in columnsMap) {
        if (!Object.prototype.hasOwnProperty.call(columnsMap, table)) continue;
        var info = throwaway.exec(TABLE_INFO_SQL[table]);
        var names = info && info[0] && info[0].values ? info[0].values.map(function (row) { return row[1]; }) : [];
        if (names.length === 0) {
          throw new ValidationError('adoption battery: required table ' + table + ' is missing');
        }
        var expected = columnsMap[table].slice().sort();
        var actual = names.slice().sort();
        if (JSON.stringify(actual) !== JSON.stringify(expected)) {
          throw new ValidationError('adoption battery: table ' + table + ' columns mismatch — expected [' + expected.join(', ') + '], got [' + actual.join(', ') + ']');
        }
      }

      // Step 5 — architecture is pinned.
      var archRow = throwaway.exec("SELECT value FROM meta WHERE key = ?", ['architecture']);
      var archValue = archRow && archRow[0] && archRow[0].values.length
        ? archRow[0].values[0][0]
        : null;
      if (archValue !== NN_ARCHITECTURE) {
        throw new ValidationError("adoption battery: meta.architecture must be '" + NN_ARCHITECTURE + "', got " + JSON.stringify(archValue));
      }

      // Step 5b — FIT-V4 generator_version allowlist (absent/1/2 only).
      // Invalid values throw HERE at the door. (The v1→v2→v3 migration at
      // db.js is no longer on the adoption path in FIT-V6; this allowlist
      // remains a structural gate on uv4 bytes.)
      checkGeneratorVersionAllowlist(throwaway, 'adoption battery');
      // Step 5c — FIT-V4 V2-pinned curriculum verification, uv>=3 BYTES.
      // Pre-v4 bytes never reach here (typed-refused at step 3b).
      if (Number(uvValue) >= 3) {
        checkV2PinnedCurriculumRows(throwaway, 'adoption battery');
      }

      // Step 6 — row-count caps. 100 000 generations is the INTENTIONAL
      // inclusive bound (MAX_GENERATIONS x POPULATION); anything larger is a
      // foreign/corrupt file. 2 000 000 driver_results covers a full lineage.
      // FIT-V3: the tracks table carries a 1 000-row import-refusal cap
      // (uv>=3 only — pre-v4 bytes are typed-refused at step 3b; a full
      // curriculum is ≤ 101 rows).
      var generationCount = countScalar(throwaway, 'SELECT COUNT(*) FROM generations');
      if (generationCount > MAX_GENERATION_ROWS) {
        throw new ValidationError('adoption battery: generations row count ' + generationCount + ' exceeds ' + MAX_GENERATION_ROWS + ' (inclusive MAX_GENERATIONS x POPULATION bound)');
      }
      var driverResultsCount = countScalar(throwaway, 'SELECT COUNT(*) FROM driver_results');
      if (driverResultsCount > MAX_DRIVER_RESULTS_ROWS) {
        throw new ValidationError('adoption battery: driver_results row count ' + driverResultsCount + ' exceeds ' + MAX_DRIVER_RESULTS_ROWS);
      }
      if (fiveTable) {
        var trackCount = countScalar(throwaway, 'SELECT COUNT(*) FROM tracks');
        if (trackCount > MAX_TRACK_ROWS) {
          throw new ValidationError('adoption battery: tracks row count ' + trackCount + ' exceeds ' + MAX_TRACK_ROWS + ' (import-refusal bound — never a deletion)');
        }
      }

      // Step 7 — every best_weights.weights_json row: length cap first (SQL,
      // cheap), then per-row full validation through the SAME validator the
      // NN constructor uses (exactly 434 finite floats, architecture match,
      // |v| <= 1e6). Throws ValidationError on the first offender — the
      // wrap carries the generation number so an operator can find it.
      var fatWeights = countScalar(throwaway, 'SELECT COUNT(*) FROM best_weights WHERE length(weights_json) > ?', [MAX_WEIGHTS_JSON_LENGTH]);
      if (fatWeights > 0) {
        throw new ValidationError('adoption battery: ' + fatWeights + ' best_weights row(s) exceed the ' + MAX_WEIGHTS_JSON_LENGTH + '-character weights_json cap');
      }
      var weightsRows = throwaway.exec('SELECT generation, weights_json FROM best_weights');
      var weightValues = weightsRows && weightsRows[0] ? weightsRows[0].values : [];
      for (var w = 0; w < weightValues.length; w++) {
        var generation = weightValues[w][0];
        var weightsJson = weightValues[w][1];
        if (typeof weightsJson !== 'string') {
          throw new ValidationError('adoption battery: best_weights row generation ' + generation + ' has a non-string weights_json');
        }
        try {
          NeuralNetwork.fromJSON(weightsJson);
        } catch (err) {
          throw new ValidationError('adoption battery: best_weights row generation ' + generation + ' failed weights validation: ' + errorMessage(err));
        }
      }

      // Step 7b — a best_weights row for MAX(generation) is REQUIRED whenever
      // generations is non-empty (arch F-1 / security M2): a lineage whose tip
      // generation has no weights cannot be resumed honestly, so the battery
      // refuses — resumeState() later refuses with a typed ValidationError
      // instead of a TypeError.
      var maxGenForWeights = countScalar(throwaway, 'SELECT COALESCE(MAX(generation), 0) FROM generations');
      if (maxGenForWeights > 0) {
        var tipWeights = countScalar(throwaway, 'SELECT 1 FROM best_weights WHERE generation = (SELECT MAX(generation) FROM generations)');
        if (tipWeights !== 1) {
          throw new ValidationError('adoption battery: best_weights row missing for generation ' + maxGenForWeights + ' (the MAX(generation) of the lineage) — the lineage cannot be resumed honestly; refusing');
        }
      }

      // Step 8 — every generations.track_json row: length cap (256 KB).
      var fatTracks = countScalar(throwaway, "SELECT COUNT(*) FROM generations WHERE typeof(track_json) != 'text' OR length(track_json) > ?", [MAX_TRACK_JSON_LENGTH]);
      if (fatTracks > 0) {
        throw new ValidationError('adoption battery: ' + fatTracks + ' generations row(s) exceed the ' + MAX_TRACK_JSON_LENGTH + '-character track_json cap');
      }

      // Step 8b — SELF-HASH (all adopted bytes; D-14(2)): for EVERY generations row,
      // the stored track_json must recombine to the row's track_hash. Closes
      // the homogeneous forged-hash class wholesale — the file cannot carry
      // bytes that disagree with their surrogate. Bounded by
      // MAX_GENERATION_ROWS.
      checkGenerationsSelfHash(throwaway, 'adoption battery');

      // Step 9 — sqlite_master scrub: EXACTLY the expected tables +
      // sqlite_autoindex_*. uv>=3 ⇒ the five-table set; uv∈{1,2} ⇒ the
      // historical four-table set (unreachable after step 3b). ZERO triggers,
      // ZERO views, zero foreign objects either way.
      scrubSqliteMaster(throwaway, 'adoption battery', Number(uvValue) >= 3 ? EXPECTED_TABLES_V3 : undefined);

      // Step 10a — SEMANTICS (F-1/F-8): the DECLARED foreign keys on
      // driver_results/best_weights (exactly one, generation →
      // generations(generation), ON DELETE CASCADE) + the value domains of
      // every adopted generations / driver_results row. A file that passes
      // the structural arms but not these arms is refused — this is what
      // closes the FK-less hand-crafted file and the +Inf/99/7 adoption
      // class. A crafted file failing ANY arm ⇒ refusal at adoption.
      checkAdoptionSemantics(throwaway, 'adoption battery');

      // Step 10b — FIT-V3 (uv>=3): the tracks arms + the count arm
      // (refuse early) + the generations-FK arm. Pre-v4 bytes are
      // typed-refused at step 3b; uv>=3 always carries the tracks table.
      if (Number(uvValue) >= 3) {
        // Count arm FIRST — refuse early.
        checkDanglingGenerations(throwaway, 'adoption battery');
        // Arms 1–4, 6, 7 — value domains, typeof-strict, NaN/NULL-safe.
        checkTracksValueDomain(throwaway, 'adoption battery');
        // Arms 5 and 8 — control_points content (code-side) + whole-row
        // recombination (Track.fromJSON + cyrb53 == track_hash).
        checkTracksContentAndHash(throwaway, 'adoption battery');
        // Arm 10 — generations FK: exactly one, tracks(track_hash),
        // on_delete ∉ {SET NULL, SET DEFAULT, CASCADE} ('' or NO ACTION).
        checkGenerationsForeignKey(throwaway, 'adoption battery');
      }

      // Coherence summary for the reconcile UI.
      var maxGeneration = countScalar(throwaway, 'SELECT COALESCE(MAX(generation), 0) FROM generations');
      var trackHashValue = null;
      var trackRow = throwaway.exec("SELECT value FROM meta WHERE key = ?", ['track_hash']);
      if (trackRow && trackRow[0] && trackRow[0].values.length) {
        trackHashValue = trackRow[0].values[0][0];
      }
      var rngState = null;
      var rngRow = throwaway.exec("SELECT value FROM meta WHERE key = ?", ['rng_state']);
      if (rngRow && rngRow[0] && rngRow[0].values.length) {
        try {
          rngState = strictUint32(rngRow[0].values[0][0], 'meta.rng_state');
        } catch (err) {
          throw new ValidationError('adoption battery: meta.rng_state is not a strict uint32: ' + errorMessage(err));
        }
      }
      return { generation: maxGeneration, architecture: archValue, trackHash: trackHashValue, rngState: rngState };
    } catch (err) {
      // Preserve typed refusals: WallModelChangedError (and ResumeTrackError)
      // extend ValidationError but carry their own `name` (and, for the
      // wall-model refusal, the verbatim refused bytes). `instanceof` keeps
      // every ValidationError subclass typed at the boundary while wrapping
      // genuinely foreign faults as before.
      if (err instanceof ValidationError) throw err;
      throw new ValidationError('adoption battery: integrity check failed: ' + errorMessage(err));
    } finally {
      try { throwaway.close(); } catch (closeErr) { /* best effort */ }
    }
  }

  // -------------------------------------------------------------------------
  // Browser-cache slot readers/deleters — the SAME internal 'car_racer' IDB
  // kv store persist() writes (steps 4 + 8). Statics so the M4 session layer
  // (db-session.js restore, the reconcile "Restore from browser cache"
  // action) can read them WITHOUT a live working Db. Every method degrades
  // to null / typed { ok:false } when IDB is absent, blocked or unreadable —
  // never throws across the session-layer boundary. A bare `new Db()` is a
  // scratch vehicle for the instance IDB helpers only; it never touches the
  // working database. ADR-0004: 'car_racer.lastKnownGood' is THE single
  // browser-cache slot; 'car_racer.dbCache' (the duplicate) is retired.
  // -------------------------------------------------------------------------
  static async readLastKnownGood() {
    var probe = new Db();
    var res = await probe._idbGet('car_racer.lastKnownGood');
    if (!res || !res.ok || !res.value) return null;
    var bytes = normalizeClonedBytes(res.value.bytes);
    return bytes && bytes.length ? bytes : null;
  }

  // Db._readLastKnownGoodRecord() — INTERNAL (restore path only): the full
  // LSG record { bytes, tag } from IDB — the restore core needs the REJECTED
  // tag's maxGeneration to compare against the lineageSnapshot's tag (H).
  // Never throws; null when absent/unreadable.
  static async _readLastKnownGoodRecord() {
    var probe = new Db();
    var res = await probe._idbGet('car_racer.lastKnownGood');
    if (!res || !res.ok || !res.value) return null;
    var bytes = normalizeClonedBytes(res.value.bytes);
    if (!bytes || !bytes.length) return null;
    var tag = (res.value.tag && typeof res.value.tag === 'object') ? res.value.tag : null;
    return { bytes: bytes, tag: tag };
  }

  // Db._readLineageSnapshotBytes() — INTERNAL (restore path only): the full
  // lineageSnapshot record { bytes, tag } — the LAST-RESORT recovery slot
  // (F-7). The snapshot stores { maxGeneration, track_hash }-style tags; a
  // null-bytes value (the freshLineage:true reset marker) reads as null.
  // Never throws; null when absent/unreadable/null-bytes.
  static async _readLineageSnapshotBytes() {
    var probe = new Db();
    var res = await probe._idbGet('car_racer.lineageSnapshot');
    if (!res || !res.ok || !res.value) return null;
    var bytes = normalizeClonedBytes(res.value.bytes);
    if (!bytes || !bytes.length) return null;
    var tag = (res.value.tag && typeof res.value.tag === 'object') ? res.value.tag : null;
    return { bytes: bytes, tag: tag };
  }

  // clearLastKnownGood() — delete the LSG slot. Used by the restore-failure
  // path (a poisoned cache must never be re-offered or looped on). Typed
  // { ok } result — callers treat it as best-effort.
  static async clearLastKnownGood() {
    var probe = new Db();
    return probe._idbDelete('car_racer.lastKnownGood');
  }

  // -------------------------------------------------------------------------
  // Db.restoreFromLastKnownGood(opts) — the battery-gated recovery core
  // (ADR-0004). Node-testable: opts.read (async () => Uint8Array | null) is
  // injectable; the browser path reads the full LSG record (bytes + tag). On
  // ANY failure this static builds its OWN scratch instance — the working DB
  // is NEVER adopted, opened or touched. Returns
  //   { ok:true,  db, summary:{ generation, trackHash, rngState, slot } }
  //   { ok:false, reason:'no-cache' }                              (no bytes)
  //   { ok:false, reason:'validation', detail }                    (refused)
  // summary.slot names the winning slot: 'lastKnownGood' (preferred) or
  // 'lineageSnapshot' (H — last resort).
  //
  // Recovery order (H, F-7): LSG is tried FIRST. If LSG is absent or fails
  // the battery, the 'car_racer.lineageSnapshot' slot is tried as a LAST
  // RESORT — but ONLY when (a) the snapshot passes the SAME full battery, and
  // (b) the snapshot's tag maxGeneration is STRICTLY GREATER than LSG's
  // rejected tag's maxGeneration — or LSG was unreadable (no bytes, or opts.read
  // injected in tests with no tag on hand). An older/equal snapshot is NOT
  // restored: the LSG-first failure verdict stands. Either slot's success
  // builds a scratch instance; resumeState() is called exactly ONCE per
  // restored slot and the summary reuses it.
  // -------------------------------------------------------------------------
  static async restoreFromLastKnownGood(opts) {
    var read = (opts && opts.read) || null;
    var bytes = null;
    var lsgTag = null;      // the rejected LSG tag, when known (browser path)
    var lsgUnreadable = false;
    if (read) {
      bytes = await read(); // test seam — no tag on hand (treated as unreadable)
      lsgUnreadable = true;
    } else {
      var record = await Db._readLastKnownGoodRecord();
      if (record) {
        bytes = record.bytes;
        lsgTag = record.tag;
      } else {
        lsgUnreadable = true;
      }
    }
    var lsgReason = 'no-cache';
    var lsgDetail = null;
    var lsgRefusedBytes = null;
    if (bytes && bytes.length) {
      try {
        Db.validateAdoptionBytes(bytes);
        var fresh = new Db();
        await fresh.init(bytes);
        var rs = fresh.resumeState(); // ONE call — the summary reuses it
        return { ok: true, db: fresh, summary: { generation: rs.maxGeneration, trackHash: rs.trackHash, rngState: rs.rngState, slot: 'lastKnownGood' } };
      } catch (err) {
        lsgReason = 'validation';
        lsgDetail = String((err && err.message) || err);
        // FIT-V6 — byte-preserving refusal: expose the exact refused bytes on
        // the result so the UI can offer "download refused bytes verbatim".
        // The LSG slot itself is left untouched (db-session never clears it on
        // a typed wall-model refusal).
        if (err instanceof WallModelChangedError && err.refusedBytes) {
          lsgRefusedBytes = err.refusedBytes;
        }
      }
    }
    // ---- H — lineageSnapshot as LAST RESORT (LSG-first preference kept).
    var snapshot = null;
    try {
      snapshot = await Db._readLineageSnapshotBytes();
    } catch (err) {
      snapshot = null;
    }
    if (snapshot) {
      var snapshotBatteryOk = false;
      try {
        Db.validateAdoptionBytes(snapshot.bytes);
        snapshotBatteryOk = true;
      } catch (err) {
        snapshotBatteryOk = false;
      }
      if (snapshotBatteryOk) {
        var snapTagGen = snapshot.tag && Number.isFinite(Number(snapshot.tag.maxGeneration))
          ? Number(snapshot.tag.maxGeneration)
          : null;
        var lsgRejectedGen = (!lsgUnreadable && lsgTag && Number.isFinite(Number(lsgTag.maxGeneration)))
          ? Number(lsgTag.maxGeneration)
          : null;
        // A strictly-ahead snapshot is required whenever LSG's rejected tag is
        // known; an unreadable LSG imposes no comparison.
        var ahead = lsgRejectedGen === null || (snapTagGen !== null && snapTagGen > lsgRejectedGen);
        if (ahead) {
          try {
            var snapDb = new Db();
            await snapDb.init(snapshot.bytes);
            var rsSnap = snapDb.resumeState(); // ONE call — the summary reuses it
            return { ok: true, db: snapDb, summary: { generation: rsSnap.maxGeneration, trackHash: rsSnap.trackHash, rngState: rsSnap.rngState, slot: 'lineageSnapshot' } };
          } catch (err) {
            try { snapDb.db.close(); } catch (closeErr) { /* best effort */ }
            // fall through to the LSG-first failure verdict
          }
        }
      }
    }
    if (lsgReason === 'no-cache') return { ok: false, reason: 'no-cache' };
    var refusal = { ok: false, reason: 'validation', detail: lsgDetail };
    if (lsgRefusedBytes) refusal.refusedBytes = lsgRefusedBytes;
    return refusal;
  }

  // _verifyForPersist(bytes) — the persist step-2 check: throwaway instance,
  // integrity_check + uv==4 + meta.wall_model=='v6' + meta.architecture + a
  // coherence tag {maxGeneration, rngState, trackHash}. PURE FUNCTION OF BYTES
  // — no handle, no IDB, no DOM. Exposed for tests (tests/smoke.mjs leg 26/27)
  // and used by persist(). FIT-V6: a pre-v4 file throws the typed,
  // byte-preserving WallModelChangedError (with err.refusedBytes); any other
  // violation throws ValidationError. persist() treats either as "nothing is
  // cached, lastKnownGood untouched, blocking banner".
  _verifyForPersist(bytes) {
    if (!(bytes instanceof Uint8Array)) {
      throw new ValidationError('verifyForPersist: bytes must be a Uint8Array, got ' + typeof bytes);
    }
    if (bytes.length === 0) {
      throw new ValidationError('verifyForPersist: bytes are empty');
    }
    if (bytes.length > MAX_DB_BYTES) {
      throw new ValidationError('verifyForPersist: bytes exceed the ' + MAX_DB_BYTES + '-byte (64 MB) cap');
    }
    var SQL = requireSql();
    var throwaway = null;
    try {
      throwaway = new SQL.Database(bytes);
    } catch (err) {
      throw new ValidationError('verifyForPersist: failed to open bytes as SQLite: ' + errorMessage(err));
    }
    try {
      var ic = throwaway.exec('PRAGMA integrity_check');
      var icValue = ic && ic[0] && ic[0].values && ic[0].values[0]
        ? ic[0].values[0][0]
        : null;
      if (icValue !== 'ok') {
        throw new ValidationError("verifyForPersist: integrity_check returned " + JSON.stringify(icValue) + " (expected 'ok')");
      }
      // FIT-V6 — pin uv==4 + meta.wall_model=='v6' on the persist step-2
      // verify. Pre-v4 bytes get the typed, byte-preserving refusal so the
      // banner can offer "download refused bytes verbatim"; any other uv is a
      // generic refusal.
      var uvRows = throwaway.exec('PRAGMA user_version');
      var uvValue = uvRows && uvRows[0] && uvRows[0].values && uvRows[0].values[0]
        ? Number(uvRows[0].values[0][0])
        : NaN;
      if (uvValue === 1 || uvValue === 2 || uvValue === 3) {
        throw newWallModelChangedError(bytes);
      }
      if (uvValue !== 4) {
        throw new ValidationError('verifyForPersist: PRAGMA user_version ' + String(uvValue) + ' — refusing (expected 4)');
      }
      var wmRows = throwaway.exec("SELECT value FROM meta WHERE key = 'wall_model'");
      var wmValue = wmRows && wmRows[0] && wmRows[0].values.length
        ? wmRows[0].values[0][0]
        : null;
      if (wmValue !== 'v6') {
        throw new ValidationError("verifyForPersist: meta.wall_model must be 'v6', got " + JSON.stringify(wmValue));
      }
      var archRow = throwaway.exec("SELECT value FROM meta WHERE key = ?", ['architecture']);
      var archValue = archRow && archRow[0] && archRow[0].values.length
        ? archRow[0].values[0][0]
        : null;
      if (archValue !== NN_ARCHITECTURE) {
        throw new ValidationError("verifyForPersist: meta.architecture must be '" + NN_ARCHITECTURE + "', got " + JSON.stringify(archValue));
      }
      // FIT-V4: the same generator_version allowlist runs on the persist
      // step-2 verify (absent/1/2 only; invalid ⇒ fail-closed ValidationError).
      checkGeneratorVersionAllowlist(throwaway, 'verifyForPersist');
      var maxGeneration = countScalar(throwaway, 'SELECT COALESCE(MAX(generation), 0) FROM generations');
      var trackHashValue = null;
      var trackRow = throwaway.exec("SELECT value FROM meta WHERE key = ?", ['track_hash']);
      if (trackRow && trackRow[0] && trackRow[0].values.length) {
        trackHashValue = trackRow[0].values[0][0];
      }
      var rngState = null;
      var rngRow = throwaway.exec("SELECT value FROM meta WHERE key = ?", ['rng_state']);
      if (rngRow && rngRow[0] && rngRow[0].values.length) {
        rngState = strictUint32(rngRow[0].values[0][0], 'meta.rng_state');
      }
      return { maxGeneration: maxGeneration, rngState: rngState, trackHash: trackHashValue };
    } catch (err) {
      // Preserve typed refusals (WallModelChangedError + subclasses): keep
      // their name and err.refusedBytes intact for the banner/download action.
      if (err instanceof ValidationError) throw err;
      throw new ValidationError('verifyForPersist: unexpected failure (missing/alien schema?): ' + errorMessage(err));
    } finally {
      try { throwaway.close(); } catch (closeErr) { /* best effort */ }
    }
  }

  // -------------------------------------------------------------------------
  // logGeneration(gen) — the ONE write path for a completed generation.
  //
  // ONE transaction per generation (plan §DETERMINISM): BEGIN; INSERT INTO
  // generations; INSERT <POPULATION_SIZE> x driver_results; INSERT INTO
  // best_weights; UPDATE meta SET value=… WHERE key='rng_state'; COMMIT.
  // ANY error ⇒ ROLLBACK ⇒ a typed throw; the committed prefix is untouched.
  // ALL inserts are plain INSERT — no conflict clause may appear anywhere
  // (grep-gated); a UNIQUE conflict means the lineage integrity was violated
  // (N2): LineageIntegrityError is thrown (message contains "UNIQUE") and the
  // caller HALTS the run — a committed lineage is never rewritten.
  //
  // Validation happens BEFORE BEGIN (structural/range/type checks, NaN
  // included). The meta pre-conditions (rng_state exists; track_hash matches
  // the lineage pin when set — or, at a MAX(generation) ≡ 0 (mod 50) / 0
  // boundary, the §7 pin-move arm legalizes a re-pin to a tracks row whose
  // difficulty == the rotation ladder) also run before BEGIN. The payload's
  // track_json is recombined to payload.trackHash before BEGIN (D-14(3));
  // a boundary re-pin lands INSIDE the logging transaction (UPDATE meta
  // SET track_hash), never before BEGIN.
  //
  // gen shape: { generation (int ≥ 1), driversEvaluated (int ≥ 1),
  // bestFitness/avgFitness/worstFitness (finite; fitness may be NEGATIVE —
  // net signed odometer + lap bonus), bestDriverIndex (int 0–19),
  // mutationRate ([0,1]), mutationScale (finite ≥ 0), trackJson (non-empty
  // string ≤ 262144), trackHash (non-empty string), simSeconds
  // (finite ≥ 0), rngState (strict uint32 — number or decimal TEXT),
  // results[<20>] each { distance (finite; MAY BE NEGATIVE — fitness),
  // topSpeed, timeAlive (finite ≥ 0),
  // crashed, stuck (0|1; booleans are normalized — the Trainer payload
  // carries booleans and this is the normalization boundary) },
  // bestWeightsJson (JSON string ≤ 16384 or a weights object — normalized
  // to the string that is stored; validated by NeuralNetwork.fromJSON).
  // -------------------------------------------------------------------------
  logGeneration(gen) {
    this._requireDb();
    if (!gen || typeof gen !== 'object' || Array.isArray(gen)) {
      throw new ValidationError('logGeneration: gen must be an object, got ' + (gen === null ? 'null' : typeof gen));
    }
    var generation = gen.generation;
    if (!Number.isInteger(generation) || generation < 1) {
      throw new ValidationError('logGeneration: generation must be an integer >= 1, got ' + JSON.stringify(generation));
    }
    var driversEvaluated = gen.driversEvaluated;
    if (!Number.isInteger(driversEvaluated) || driversEvaluated < 1) {
      throw new ValidationError('logGeneration: driversEvaluated must be an integer >= 1, got ' + JSON.stringify(driversEvaluated));
    }
    var fitnessFields = [['bestFitness', gen.bestFitness], ['avgFitness', gen.avgFitness], ['worstFitness', gen.worstFitness]];
    for (var f = 0; f < fitnessFields.length; f++) {
      var fieldName = fitnessFields[f][0];
      var fieldValue = fitnessFields[f][1];
      // fitness may be negative (net signed odometer + lap bonus) — finite
      // only; NaN/±Infinity are still rejected.
      if (typeof fieldValue !== 'number' || !Number.isFinite(fieldValue)) {
        throw new ValidationError('logGeneration: ' + fieldName + ' must be a finite number, got ' + String(fieldValue));
      }
    }
    var bestDriverIndex = gen.bestDriverIndex;
    if (!Number.isInteger(bestDriverIndex) || bestDriverIndex < 0 || bestDriverIndex >= POPULATION_SIZE) {
      throw new ValidationError('logGeneration: bestDriverIndex must be an integer 0-' + (POPULATION_SIZE - 1) + ', got ' + JSON.stringify(bestDriverIndex));
    }
    var mutationRate = gen.mutationRate;
    if (typeof mutationRate !== 'number' || !Number.isFinite(mutationRate) || mutationRate < 0 || mutationRate > 1) {
      throw new ValidationError('logGeneration: mutationRate must be a finite number in [0,1], got ' + String(mutationRate));
    }
    var mutationScale = gen.mutationScale;
    if (typeof mutationScale !== 'number' || !Number.isFinite(mutationScale) || mutationScale < 0) {
      throw new ValidationError('logGeneration: mutationScale must be a finite number >= 0, got ' + String(mutationScale));
    }
    var trackJson = gen.trackJson;
    if (typeof trackJson !== 'string' || trackJson.length === 0) {
      throw new ValidationError('logGeneration: trackJson must be a non-empty string');
    }
    if (trackJson.length > MAX_TRACK_JSON_LENGTH) {
      throw new ValidationError('logGeneration: trackJson exceeds ' + MAX_TRACK_JSON_LENGTH + ' characters (got ' + trackJson.length + ')');
    }
    var trackHash = gen.trackHash;
    if (typeof trackHash !== 'string' || trackHash.length === 0) {
      throw new ValidationError('logGeneration: trackHash must be a non-empty string');
    }
    var simSeconds = gen.simSeconds;
    if (typeof simSeconds !== 'number' || !Number.isFinite(simSeconds) || simSeconds < 0) {
      throw new ValidationError('logGeneration: simSeconds must be a finite number >= 0, got ' + String(simSeconds));
    }
    var state = strictUint32(gen.rngState, 'logGeneration rngState');
    var results = gen.results;
    if (!Array.isArray(results) || results.length !== POPULATION_SIZE) {
      throw new ValidationError('logGeneration: results must be an array of exactly ' + POPULATION_SIZE + ' entries, got ' + (Array.isArray(results) ? results.length : typeof results));
    }
    var normalizedResults = [];
    for (var i = 0; i < results.length; i++) {
      var r = results[i];
      if (!r || typeof r !== 'object' || Array.isArray(r)) {
        throw new ValidationError('logGeneration: results[' + i + '] must be an object');
      }
      var distance = r.distance, topSpeed = r.topSpeed, timeAlive = r.timeAlive;
      // distance is FITNESS — net signed odometer + lap bonus; negative is legal.
      if (typeof distance !== 'number' || !Number.isFinite(distance)) {
        throw new ValidationError('logGeneration: results[' + i + '].distance must be a finite number, got ' + String(distance));
      }
      if (typeof topSpeed !== 'number' || !Number.isFinite(topSpeed) || topSpeed < 0) {
        throw new ValidationError('logGeneration: results[' + i + '].topSpeed must be a finite number >= 0, got ' + String(topSpeed));
      }
      if (typeof timeAlive !== 'number' || !Number.isFinite(timeAlive) || timeAlive < 0) {
        throw new ValidationError('logGeneration: results[' + i + '].timeAlive must be a finite number >= 0, got ' + String(timeAlive));
      }
      normalizedResults.push({
        distance: distance,
        topSpeed: topSpeed,
        timeAlive: timeAlive,
        crashed: normalizeBinaryFlag(r.crashed, 'results[' + i + '].crashed'),
        stuck: normalizeBinaryFlag(r.stuck, 'results[' + i + '].stuck'),
      });
    }
    // driversEvaluated must EQUAL the number of results actually logged —
    // an audit-column lie is rejected BEFORE BEGIN (prefix intact). The
    // results length itself is pinned to POPULATION_SIZE above (the one
    // number that must agree with evolution.js's POPULATION — kept a local
    // literal on purpose: db.js must load without evolution.js).
    if (driversEvaluated !== normalizedResults.length) {
      throw new ValidationError('logGeneration: driversEvaluated (' + driversEvaluated + ') must equal results.length (' + normalizedResults.length + ')');
    }
    var NeuralNetwork = requireNeuralNetwork();
    var weightsInput = gen.bestWeightsJson;
    var weightsString = null;
    if (typeof weightsInput === 'string') {
      if (weightsInput.length > MAX_WEIGHTS_JSON_LENGTH) {
        throw new ValidationError('logGeneration: bestWeightsJson exceeds ' + MAX_WEIGHTS_JSON_LENGTH + ' characters (got ' + weightsInput.length + ')');
      }
      NeuralNetwork.fromJSON(weightsInput); // full validation — throws on any violation
      weightsString = weightsInput;
    } else if (weightsInput !== null && typeof weightsInput === 'object' && !Array.isArray(weightsInput)) {
      NeuralNetwork.fromJSON(weightsInput); // validate the object payload too (434 finite floats, architecture, |v| ≤ 1e6)
      weightsString = JSON.stringify(weightsInput);
      if (weightsString.length > MAX_WEIGHTS_JSON_LENGTH) {
        throw new ValidationError('logGeneration: serialized bestWeightsJson exceeds ' + MAX_WEIGHTS_JSON_LENGTH + ' characters (got ' + weightsString.length + ')');
      }
    } else {
      throw new ValidationError('logGeneration: bestWeightsJson must be a JSON string or a weights object, got ' + (weightsInput === null ? 'null' : typeof weightsInput));
    }
    // ---- lineage pre-conditions (before BEGIN).
    var existingRng = this._metaValue('rng_state');
    if (existingRng === null) {
      throw new ValidationError("logGeneration: meta.rng_state is missing — call ensureLineage(trackHash, rngState, trackJson) at lineage start (the Trainer does this before generation 1)");
    }
    // FIT-V3 recombination arm (D-14(3)) — the payload's track_json must
    // recombine to payload.trackHash; mismatch ⇒ refusal before BEGIN.
    var recombinedPayloadHash = generationRowHash(trackJson, 'logGeneration');
    if (recombinedPayloadHash !== trackHash) {
      throw new ValidationError('logGeneration: track_json recombined to hash ' + recombinedPayloadHash + ' but payload.trackHash is ' + JSON.stringify(trackHash) + ' — hash/bytes divergence; refusing to log');
    }
    var existingHash = this._metaValue('track_hash');
    var pinMoveHash = null; // set iff the boundary arm legalizes a re-pin
    // FIT-V3 — rotation boundary (gen ≡ 1 (mod 50), N ≥ 51): the ONLY legal
    // track for the next generation is the byte-verified resolver row for the
    // upcoming rotation — a payload carrying the CURRENT pin, a stale editor row,
    // a fork-retention decoy, ANY hash other than the resolver row, is refused
    // regardless of the CURRENT pin's hash-equality status. (The old gate required
    // a MISMATCH to fire; a payload carrying the current pin at a boundary would
    // silently stall the curriculum ladder forever.)
    var maxRowPre = this.db.exec('SELECT COALESCE(MAX(generation), 0) FROM generations');
    var maxGenPre = maxRowPre && maxRowPre[0] && maxRowPre[0].values && maxRowPre[0].values[0]
      ? Number(maxRowPre[0].values[0][0]) || 0
      : 0;
    if (maxGenPre !== 0 && (maxGenPre % ROTATE_EVERY) === 0) {
      var Rotation = requireRotation();
      var seedPreRaw = this._metaValue('rotation_seed');
      if (seedPreRaw === null) {
        throw new ValidationError('logGeneration: the rotation boundary requires meta.rotation_seed — the lineage was pinned by an older build or corrupted adoption; refusing to log on any other track');
      }
      var rotationSeedPre = strictUint32(seedPreRaw, 'boundary rotation_seed');
      var boundaryRIdx = Math.floor(maxGenPre / ROTATE_EVERY);
      // FIT-V4: re-derive the boundary hash under the lineage's OWN generator
      // version (parsed from meta via the ONE canonical parser — absent ⇒ V1).
      // A V2 lineage cannot be satisfied by a V1 hash (and vice versa).
      var boundaryVersion = requireVersionParser()(this._metaValue('generator_version'));
      var boundaryHash = Rotation.resolveGeneratedTrack(rotationSeedPre, Math.min(boundaryRIdx, 99), boundaryVersion).trackHash;
      if (trackHash !== boundaryHash) {
        throw new ValidationError('logGeneration: at the gen ' + (maxGenPre + 1) + ' boundary the only legal track is the byte-verified resolver output for rotation ' + boundaryRIdx + ' (hash ' + boundaryHash + ') — payload carries ' + JSON.stringify(trackHash) + '; refusing to log on any other track');
      }
      var bRows = this.db.exec("SELECT COUNT(*), MIN(difficulty), MAX(difficulty) FROM tracks WHERE track_hash = ? AND source = 'generated'", [boundaryHash]);
      var bRow = bRows && bRows[0] && bRows[0].values && bRows[0].values[0];
      if (!bRow || Number(bRow[0]) !== 1 || Number(bRow[1]) !== Math.min(boundaryRIdx, 100)) {
        throw new ValidationError('logGeneration: the boundary track row for rotation ' + boundaryRIdx + ' is missing or its difficulty is off the ladder — the v3 curriculum may be corrupted; refusing');
      }
      pinMoveHash = trackHash;
    } else if (existingHash !== null && existingHash !== trackHash) {
      // non-boundary re-pin — impossible under the rotation model; a fork must go through forkLineage
      throw new ValidationError('logGeneration: track_hash mismatch — meta.track_hash is ' + JSON.stringify(existingHash) + ' (frozen pin) but the payload carries ' + JSON.stringify(trackHash) + ' at MAX(generation)=' + maxGenPre + ' (not a rotation boundary); a confirmed fork (forkLineage) is required, never a silent re-pin');
    }
    // ---- ONE transaction; any error ⇒ ROLLBACK ⇒ typed throw. BEGIN is
    //      INSIDE the try: a BEGIN failure (e.g. sqlite out of memory/disk)
    //      is a NON-lineage fault — begun stays false, no ROLLBACK is issued,
    //      the raw error propagates (never a LineageIntegrityError).
    var begun = false;
    try {
      this.db.run('BEGIN');
      begun = true;
      this.db.run('INSERT INTO generations (generation, drivers_evaluated, best_fitness, avg_fitness, worst_fitness, best_driver_index, mutation_rate, mutation_scale, track_json, track_hash, sim_seconds, rng_state) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
        [generation, driversEvaluated, gen.bestFitness, gen.avgFitness, gen.worstFitness, bestDriverIndex, mutationRate, mutationScale, trackJson, trackHash, simSeconds, state]);
      for (var j = 0; j < normalizedResults.length; j++) {
        var rr = normalizedResults[j];
        this.db.run('INSERT INTO driver_results (generation, driver_index, distance, top_speed, time_alive, crashed, stuck) VALUES (?,?,?,?,?,?,?)',
          [generation, j, rr.distance, rr.topSpeed, rr.timeAlive, rr.crashed, rr.stuck]);
      }
      this.db.run('INSERT INTO best_weights (generation, architecture, weights_json) VALUES (?,?,?)', [generation, NN_ARCHITECTURE, weightsString]);
      this.db.run('UPDATE meta SET value = ? WHERE key = ?', [String(state), 'rng_state']);
      // FIT-V3 pin-move — INSIDE the logging transaction: the boundary
      // re-pin (validated BEFORE BEGIN by the §7 arm) moves meta.track_hash
      // in the same transaction as the generation rows. No boundary INSERT —
      // pre-materialization owns tracks row creation; this arm asserts, it
      // doesn't write.
      if (pinMoveHash !== null) {
        this.db.run('UPDATE meta SET value = ? WHERE key = ?', [pinMoveHash, 'track_hash']);
      }
      this.db.run('COMMIT');
    } catch (err) {
      if (begun) {
        try { this.db.run('ROLLBACK'); } catch (rollbackErr) { /* best effort */ }
      }
      var msg = errorMessage(err);
      // Shared classifier (arch F-2 / security M2) — the SAME rule the Trainer
      // uses for halt-vs-degrade. Message keeps the literal "UNIQUE" so
      // operator tooling and the smoke legs can pin on it.
      if (isLineageIntegrityFault(err)) {
        throw new LineageIntegrityError('LINEAGE INTEGRITY ERROR — UNIQUE conflict on INSERT (' + msg + ') — committed lineage untouched; the run must HALT (N2)');
      }
      throw err; // non-UNIQUE (disk/quota/constraint) — caller applies the LOG-FAILURE session-disable policy
    }
    return { ok: true, generation: generation, driversEvaluated: driversEvaluated };
  }

  // -------------------------------------------------------------------------
  // Read paths. All bound-parameter; all fail closed on missing rows (null)
  // rather than inventing data.
  // -------------------------------------------------------------------------
  getGeneration(n) {
    var db = this._requireDb();
    assertPositiveInteger(n, 'getGeneration');
    var rows = db.exec('SELECT * FROM generations WHERE generation = ?', [n]);
    if (!rows || rows.length === 0 || rows[0].values.length === 0) return null;
    return rowObject(rows[0]);
  }

  getDriverResults(n) {
    var db = this._requireDb();
    assertPositiveInteger(n, 'getDriverResults');
    var rows = db.exec('SELECT driver_index, distance, top_speed, time_alive, crashed, stuck FROM driver_results WHERE generation = ? ORDER BY driver_index', [n]);
    // FAIL-CLOSED: sql.js exec() returns an ARRAY of result sets; rowObjects
    // expects the result SET — hand it rows[0] (undefined when empty ⇒ []).
    return rowObjects(rows && rows[0]);
  }

  // getBestWeights(n) — validated NeuralNetwork instance (fromJSON: exactly
  // 434 floats, finite, |v| ≤ 1e6, architecture). THROWS ValidationError on a
  // corrupt stored payload (fail closed — the caller surfaces REPLAY
  // BLOCKED). Returns null when the generation does not exist/has no weights.
  getBestWeights(n) {
    var db = this._requireDb();
    assertPositiveInteger(n, 'getBestWeights');
    var rows = db.exec('SELECT weights_json FROM best_weights WHERE generation = ?', [n]);
    if (!rows || rows.length === 0 || rows[0].values.length === 0) return null;
    return requireNeuralNetwork().fromJSON(rows[0].values[0][0]);
  }

  listGenerations() {
    var db = this._requireDb();
    var rows = db.exec('SELECT generation FROM generations ORDER BY generation');
    var out = [];
    if (rows && rows[0]) {
      for (var i = 0; i < rows[0].values.length; i++) out.push(rows[0].values[i][0]);
    }
    return out;
  }

  // listGenerationSummaries(limit = 50, offset = 0) — the Logs panel's page
  // query: newest first, one row per GENERATION with the per-generation
  // crash count pre-joined via subquery — no N+1 getDriverResults() round
  // trips. limit/offset are bound parameters; garbage input throws a typed
  // ValidationError (fail closed) — nothing is interpolated into the SQL.
  listGenerationSummaries(limit, offset) {
    var db = this._requireDb();
    var lim = limit === undefined ? 50 : limit;
    var off = offset === undefined ? 0 : offset;
    if (!Number.isInteger(lim) || lim < 1) {
      throw new ValidationError('listGenerationSummaries: limit must be an integer >= 1, got ' + JSON.stringify(lim));
    }
    if (!Number.isInteger(off) || off < 0) {
      throw new ValidationError('listGenerationSummaries: offset must be an integer >= 0, got ' + JSON.stringify(off));
    }
    var rows = db.exec(
      'SELECT generation, best_fitness, avg_fitness, worst_fitness, best_driver_index, drivers_evaluated, ' +
      '(SELECT COUNT(*) FROM driver_results WHERE driver_results.generation = g.generation AND crashed = 1) AS crashed_count, ' +
      'g.created_at, t.difficulty AS difficulty, t.source AS source ' +
      'FROM generations g LEFT JOIN tracks t ON t.track_hash = g.track_hash ORDER BY g.generation DESC LIMIT ? OFFSET ?',
      [lim, off]
    );
    var out = rowObjects(rows && rows[0]);
    // FIT-V3 — rotationIndex is derived code-side (never stored): the block
    // of ROTATE_EVERY generations this row belongs to. After v3, no row can
    // dangle (the count arm refuses the file at adoption), so the LEFT JOIN
    // never yields NULL difficulty/source here — kept honest-nullable anyway.
    for (var i = 0; i < out.length; i++) {
      out[i].rotationIndex = Math.floor((out[i].generation - 1) / ROTATE_EVERY);
    }
    return out;
  }

  // resumeState() — what the M4 boot/resume path needs. bestWeights is the
  // plain VALIDATED payload ({ architecture, w1..b3 } — exactly the shape
  // Evolution.resumeFromBest() and NeuralNetwork.fromJSON() consume), NOT a
  // NeuralNetwork instance; it is deep-copied by getBestWeights()'s path so
  // callers may mutate it freely. (getBestWeights(n), by contrast, returns
  // the validated NeuralNetwork INSTANCE for the replay path.) rngState is a
  // uint32 NUMBER (strict round-trip from the TEXT column). trackHash may be
  // null for a pre-pin foreign database — surfaced, never invented. Throws
  // ValidationError when the max(generation) has NO best_weights row (the
  // lineage cannot be resumed honestly — refusing, never a TypeError) or
  // when a committed lineage has no meta.rng_state (ditto).
  // FIT-V3 gains: { rotationIndex, difficulty, trackSource, rotationSeed }.
  // rotationIndex = floor((maxGeneration - 1) / ROTATE_EVERY) (0 for an
  // empty lineage). difficulty/trackSource are the STORED tracks row OF THE
  // TIP GENERATION (inner JOIN) — null when absent (honest), never an
  // invented 0. rotationSeed is meta.rotation_seed parsed as a strict
  // uint32 — null pre-pin (fresh uninitialized lineage); a committed lineage
  // always carries it post-FIT-V3. The trackHash read order is unchanged
  // (meta is the authority; the boundary invariant keeps meta.track_hash ==
  // generations(MAX).track_hash after any pin-move commit).
  resumeState() {
    var db = this._requireDb();
    var maxRow = db.exec('SELECT COALESCE(MAX(generation), 0) FROM generations');
    var maxGeneration = maxRow && maxRow[0] && maxRow[0].values.length
      ? Number(maxRow[0].values[0][0]) || 0
      : 0;
    var seedRows = this._metaValue('rotation_seed');
    var rotationSeed = seedRows === null ? null : strictUint32(seedRows, 'resumeState meta.rotation_seed');
    // FIT-V4: the lineage's generator version travels with the resume state.
    // Parsed via the ONE canonical parser (absent ⇒ V1); invalid ⇒ typed throw.
    var generatorVersion = requireVersionParser()(this._metaValue('generator_version'));
    if (maxGeneration === 0) {
      return { maxGeneration: 0, bestWeights: null, rngState: null, trackHash: null, rotationIndex: 0, difficulty: null, trackSource: null, rotationSeed: rotationSeed, generatorVersion: generatorVersion };
    }
    var bw = this.getBestWeights(maxGeneration);
    if (bw === null) {
      throw new ValidationError('resumeState: best_weights missing for generation ' + maxGeneration + ' — the lineage cannot be resumed honestly; refusing');
    }
    var bestWeights = bw.getWeights();
    var rngRaw = this._metaValue('rng_state');
    if (rngRaw === null) {
      throw new ValidationError('resumeState: meta.rng_state is missing — the lineage pin (ensureLineage) creates it at lineage start; refusing to resume an unpinned lineage');
    }
    // The stored tracks row of the TIP generation — the rotation the resume
    // lands on. Inner join on the tip; absent ⇒ honest nulls.
    var trackRow = null;
    try {
      var trackRes = db.exec('SELECT t.difficulty, t.source FROM tracks t JOIN generations g ON g.track_hash = t.track_hash WHERE g.generation = ?', [maxGeneration]);
      trackRow = trackRes && trackRes[0] && trackRes[0].values && trackRes[0].values[0] ? trackRes[0].values[0] : null;
    } catch (err) {
      // A v3 working database whose tracks read throws is corruption of the
      // very table the resume life-rake defends — refusing honestly instead
      // of returning null-born difficulty (which resume then silently adopts
      // as the only hint it has). The pre-v3 distinction does not exist on a
      // working v3 instance.
      throw new ValidationError('resumeState: the tip generation\'s tracks row could not be read (' + errorMessage(err) + ') — refusing to resume (the lineage cannot be resumed honestly; import a full export of the lineage, or start a fresh lineage)');
    }
    return {
      maxGeneration: maxGeneration,
      bestWeights: bestWeights,
      rngState: strictUint32(rngRaw, 'resumeState meta.rng_state'),
      trackHash: this._metaValue('track_hash'),
      rotationIndex: Math.floor((maxGeneration - 1) / ROTATE_EVERY),
      difficulty: trackRow ? Number(trackRow[0]) : null,
      trackSource: trackRow ? String(trackRow[1]) : null,
      rotationSeed: rotationSeed,
      generatorVersion: generatorVersion,
    };
  }

  // adoptPinnedTrack(trackHash) — FIT-V3 (M6c), ADDITIVE. The D-5 resume-
  // adoption arm for the tip-generation pin: reads the STORED tracks row for
  // `trackHash`, recombines control_points + half_width through the ONE shared
  // path (Track.fromJSON verbatim — never string splicing), re-hashes and
  // MUST equal the row's track_hash. ABSENCE or mismatch ⇒ typed
  // ResumeTrackError (a ValidationError subtype) with NO generator fallback
  // (D-5/G3 — refusal only). Returns a FRESH, fully validated Track built
  // from the stored bytes. The caller (main.js prepareResumeLineage) hands
  // THAT track to the Trainer — never the editor/Manual active track.
  adoptPinnedTrack(trackHash) {
    var db = this._requireDb();
    if (typeof trackHash !== 'string' || trackHash.length === 0) {
      throw new ResumeTrackError('adoptPinnedTrack: trackHash must be a non-empty string, got ' + (trackHash === null ? 'null' : typeof trackHash));
    }
    var res;
    try {
      res = db.exec('SELECT control_points, half_width FROM tracks WHERE track_hash = ?', [trackHash]);
    } catch (err) {
      throw new ResumeTrackError('adoptPinnedTrack: could not read the tracks row for ' + JSON.stringify(trackHash) + ' — ' + errorMessage(err));
    }
    var values = (res && res[0] && res[0].values) || [];
    if (values.length === 0) {
      throw new ResumeTrackError('adoptPinnedTrack: no tracks row for the pinned track_hash ' + JSON.stringify(trackHash) + ' — the track pinned for the resume generation is not recoverable from this file');
    }
    var controlPoints = values[0][0];
    var halfWidth = values[0][1];
    var parsed = null;
    if (typeof controlPoints === 'string') {
      try { parsed = JSON.parse(controlPoints); } catch (err) { parsed = null; }
    }
    var contentError = controlPointsContentError(parsed);
    if (contentError) {
      throw new ResumeTrackError('adoptPinnedTrack: tracks row for ' + JSON.stringify(trackHash) + ' — ' + contentError);
    }
    if (typeof halfWidth !== 'number' || !Number.isFinite(halfWidth) || halfWidth < 1 || halfWidth > 2000) {
      throw new ResumeTrackError('adoptPinnedTrack: tracks row for ' + JSON.stringify(trackHash) + ' carries half_width out of the [1, 2000] number domain, got ' + JSON.stringify(halfWidth));
    }
    var recombined;
    try {
      recombined = recombineTrackHash(parsed, halfWidth, 'adoptPinnedTrack');
    } catch (err) {
      throw new ResumeTrackError('adoptPinnedTrack: tracks row for ' + JSON.stringify(trackHash) + ' failed recombination — ' + errorMessage(err));
    }
    if (recombined !== trackHash) {
      throw new ResumeTrackError('adoptPinnedTrack: tracks row for ' + JSON.stringify(trackHash) + ' recombines to hash ' + recombined + ' — the pinned track is not recoverable from this file');
    }
    var t = requireTrack();
    var payload;
    try {
      payload = JSON.stringify({ points: parsed, halfWidth: halfWidth });
    } catch (err) {
      throw new ResumeTrackError('adoptPinnedTrack: could not serialize the stored bytes for ' + JSON.stringify(trackHash) + ' — ' + errorMessage(err));
    }
    return t.Track.fromJSON(payload);
  }

  // stats() — the Logs panel's numbers. Empty DB ⇒ generationCount 0,
  // latestGeneration 0, bestFitnessEver/avgOfLatest null (never 0-as-a-lie).
  stats() {
    var db = this._requireDb();
    var header = db.exec('SELECT COUNT(*), COALESCE(MAX(generation), 0), MAX(best_fitness) FROM generations');
    var generationCount = 0, latestGeneration = 0, bestFitnessEver = null;
    if (header && header[0] && header[0].values.length) {
      generationCount = Number(header[0].values[0][0]) || 0;
      latestGeneration = Number(header[0].values[0][1]) || 0;
      bestFitnessEver = header[0].values[0][2] === null || header[0].values[0][2] === undefined
        ? null
        : Number(header[0].values[0][2]);
    }
    var avgOfLatest = null;
    if (latestGeneration > 0) {
      var avgRow = db.exec('SELECT avg_fitness FROM generations WHERE generation = ?', [latestGeneration]);
      if (avgRow && avgRow[0] && avgRow[0].values.length) {
        avgOfLatest = Number(avgRow[0].values[0][0]);
      }
    }
    return { generationCount: generationCount, latestGeneration: latestGeneration, bestFitnessEver: bestFitnessEver, avgOfLatest: avgOfLatest };
  }

  // _writeGeneratorVersionOnce(context) — FIT-V4: a NEW lineage is V2. Written
  // EXACTLY ONCE, inside the caller's transaction, BEFORE preMaterializeCurriculum
  // (so the db-derived version read is already V2). Absent ⇒ INSERT '2';
  // already '2' ⇒ idempotent no-op; any other value ⇒ typed ValidationError
  // (allowed values are enforced by the canonical parser, never an inline check).
  // A zero-generation legacy lineage carries no key, so re-pinning it becomes
  // a NEW V2 lineage (N6); lineage with committed generations never re-pins.
  _writeGeneratorVersionOnce(context) {
    var existing = this._metaValue('generator_version');
    if (existing === null) {
      this.db.run('INSERT INTO meta(key, value) VALUES (?, ?)', ['generator_version', String(versionTwo())]);
    } else {
      var existingVersion = requireVersionParser()(existing); // allowlist; invalid ⇒ throw
      if (existingVersion !== versionTwo()) {
        throw new ValidationError(context + ': meta.generator_version already holds ' + JSON.stringify(existing) + ' — refusing to overwrite with ' + JSON.stringify(versionTwo()) + ' (lineage mismatch — a confirmed forkLineage() is required)');
      }
    }
    return versionTwo();
  }

  // -------------------------------------------------------------------------
  // Lineage pin helpers (plan §LINEAGE PIN + FIT-V3 D-12). ensureLineage
  // creates the meta rows at lineage start; forkLineage is the ONLY way to
  // move a lineage to a new track/rng (a user-confirmed wipe); both are ONE
  // transaction — meta pin + tracks rows (the rotation-0 pin row) + full
  // curriculum pre-materialization. FIT-V3: trackJson is REQUIRED (string) —
  // the rotation-0 tracks row and the curriculum are materialized FROM THE
  // PINNED TRACK'S BYTES; a hash without bytes is refused at the door.
  //   (a) trackJson parsed + recombined to trackHash BEFORE BEGIN (D-14(3));
  //   (b) the rotation-0 tracks row INSERT (SELECT-then-split; source
  //       'default' iff the hash equals the frozen DEFAULT_CONTROL_POINTS
  //       hash else 'editor'; difficulty 0; created_at sentinel);
  //   (c) rotations 1..99 pre-materialized from (rngState, r), byte-verified
  //       (D-12) — in the SAME transaction. Any failure ⇒ typed throw, the
  //       transaction rolled back, nothing written.
  // -------------------------------------------------------------------------
  ensureLineage(trackHash, rngState, trackJson) {
    this._requireDb();
    if (typeof trackHash !== 'string' || trackHash.trim().length === 0) {
      throw new ValidationError('ensureLineage: trackHash must be a non-empty string');
    }
    if (typeof trackJson !== 'string') {
      throw new ValidationError('ensureLineage: trackJson is REQUIRED (string) — the v3 curriculum is materialized from the pinned track\'s bytes; refusing to pin without them');
    }
    var state = strictUint32(rngState, 'ensureLineage rngState');
    var pin = this._pinPrecondition(trackHash, trackJson, 'ensureLineage');
    var begun = false;
    try {
      this.db.run('BEGIN');
      begun = true;
      var entries = [['track_hash', trackHash], ['rng_state', String(state)]];
      for (var i = 0; i < entries.length; i++) {
        var key = entries[i][0];
        var value = entries[i][1];
        var existing = this._metaValue(key);
        if (existing === null) {
          this.db.run('INSERT INTO meta(key, value) VALUES (?, ?)', [key, value]);
        } else if (existing !== value) {
          throw new ValidationError('ensureLineage: meta.' + key + ' already holds ' + JSON.stringify(existing) + ' — refusing to overwrite with ' + JSON.stringify(value) + ' (lineage mismatch — a confirmed forkLineage() is required)');
        }
      }
      this._insertRotationZeroRow(trackHash, pin, 'ensureLineage');
      // FIT-V4: write generator_version='2' ONCE, in this same transaction,
      // BEFORE materialization (F3). Returns the version we wrote.
      var generatorVersion = this._writeGeneratorVersionOnce('ensureLineage');
      preMaterializeCurriculum(this.db, state, 'ensureLineage');
      // meta.rotation_seed — written exactly once per lineage (D-11): INSERT
      // when absent, no-op when already equal, refuse when different.
      var existingSeed = this._metaValue('rotation_seed');
      if (existingSeed === null) {
        this.db.run('INSERT INTO meta(key, value) VALUES (?, ?)', ['rotation_seed', String(state)]);
      } else if (String(existingSeed) !== String(state)) {
        throw new ValidationError('ensureLineage: meta.rotation_seed already holds ' + JSON.stringify(existingSeed) + ' — refusing to overwrite with ' + JSON.stringify(state) + ' (lineage mismatch — a confirmed forkLineage() is required)');
      }
      this.db.run('COMMIT');
    } catch (err) {
      if (begun) {
        try { this.db.run('ROLLBACK'); } catch (rollbackErr) { /* best effort */ }
      }
      throw err;
    }
    return { ok: true, trackHash: trackHash, rngState: state, generatorVersion: generatorVersion };
  }

  // _pinPrecondition(trackHash, trackJson, context) — the shared
  // parse-and-re-hash gate (ensureLineage/forkLineage). Parses the payload,
  // recombines to a hash and REFUSES on divergence — BEFORE BEGIN, so a bad
  // payload never touches the working database. Returns the parsed pin.
  _pinPrecondition(trackHash, trackJson, context) {
    var parsed = parseTrackJsonStrict(trackJson, context);
    var recombined = recombineTrackHash(parsed.points, parsed.halfWidth, context);
    if (recombined !== trackHash) {
      throw new ValidationError(context + ': trackJson recombined to hash ' + recombined + ' but trackHash is ' + JSON.stringify(trackHash) + ' — hash/bytes divergence; refusing to pin');
    }
    return { points: parsed.points, halfWidth: parsed.halfWidth };
  }

  // _insertRotationZeroRow(trackHash, pin, context) — SELECT-then-split
  // (same-geometry pins and reforks are no-ops). source 'default' iff the
  // hash equals the frozen DEFAULT_CONTROL_POINTS hash else 'editor';
  // difficulty 0 (D-10/D-12); created_at the artefact sentinel. A pre-existing
  // row for this hash carrying difficulty != 0 (a hash that collides with a
  // generated curriculum row) is REFUSED — rotation 0 is difficulty 0 by law;
  // overwriting the curriculum row would corrupt the ladder.
  _insertRotationZeroRow(trackHash, pin, context) {
    var existing = this.db.exec('SELECT difficulty FROM tracks WHERE track_hash = ?', [trackHash]);
    var existingRows = (existing && existing[0] && existing[0].values) || [];
    if (existingRows.length > 0) {
      var existingDifficulty = Number(existingRows[0][0]);
      if (existingDifficulty !== 0) {
        throw new ValidationError(context + ': a tracks row for ' + JSON.stringify(trackHash) + ' already exists with difficulty ' + existingDifficulty + ' (not 0) — rotation 0 is difficulty 0 by law (D-10/D-12); the lineage cannot pin rotation 0 to a non-zero-difficulty track');
      }
      return { inserted: false };
    }
    return insertTrackRowIfAbsent(this.db, {
      track_hash: trackHash,
      source: sourceForHash(trackHash),
      control_points: controlPointsJson(pin.points),
      half_width: pin.halfWidth,
      difficulty: 0,
      created_at: '2026-01-01 00:00:00',
    }, context);
  }

  // forkLineage(newTrackHash, rngState, trackJson) — the confirmed-fork wipe
  // (LINEAGE PIN / Reconcile "Fork new lineage" / same-track wipe): ONE
  // transaction — DELETE FROM driver_results; DELETE FROM best_weights;
  // DELETE FROM generations (belt-and-braces: the children are deleted
  // EXPLICITLY first, so even an adopted DB that somehow lacks the FK
  // declarations can never leave orphaned rows behind — zero behavior change
  // for valid files, whose children already cascade), UPDATE meta rng_state +
  // track_hash + rotation_seed, INSERT the rotation-0 pin row (SELECT-then-
  // split, same source rule), pre-materialize the fork's curriculum. The
  // CALLER (M4 fork flow) owns the rest of the fork ritual: battery for any
  // adopted bytes, the fresh {size, sha256} baseline, and
  // resetLineageSnapshot() — all BEFORE the new lineage's first persist.
  // Committed history is destroyed here by explicit user confirmation;
  // nothing else in this file deletes a generation. FIT-V3: trackJson is
  // REQUIRED (see ensureLineage). Tracks rows for wiped generations are
  // RETAINED (unreferenced history — no FK violation; G4 blessed).
  forkLineage(newTrackHash, rngState, trackJson) {
    this._requireDb();
    if (typeof newTrackHash !== 'string' || newTrackHash.trim().length === 0) {
      throw new ValidationError('forkLineage: newTrackHash must be a non-empty string');
    }
    if (typeof trackJson !== 'string') {
      throw new ValidationError('forkLineage: trackJson is REQUIRED (string) — the v3 curriculum is materialized from the pinned track\'s bytes; refusing to fork without them');
    }
    var state = strictUint32(rngState, 'forkLineage rngState');
    var pin = this._pinPrecondition(newTrackHash, trackJson, 'forkLineage');
    var begun = false;
    try {
      this.db.run('BEGIN');
      begun = true;
      this.db.run('DELETE FROM driver_results');
      this.db.run('DELETE FROM best_weights');
      this.db.run('DELETE FROM generations');
      this._setMetaValue('rng_state', String(state));
      this._setMetaValue('track_hash', newTrackHash);
      // A fork is a NEW lineage — its rotation_seed is the fresh seed
      // (D-11). Overwrite is correct here (the fork is explicitly a new
      // lineage identity, never a silent divergence).
      this._setMetaValue('rotation_seed', String(state));
      // FIT-V4: a fork is explicitly a NEW lineage — pin V2 ONCE, in this same
      // transaction, BEFORE materialization (overwrite is correct here).
      this._setMetaValue('generator_version', String(versionTwo()));
      this._insertRotationZeroRow(newTrackHash, pin, 'forkLineage');
      preMaterializeCurriculum(this.db, state, 'forkLineage');
      this.db.run('COMMIT');
    } catch (err) {
      if (begun) {
        try { this.db.run('ROLLBACK'); } catch (rollbackErr) { /* best effort */ }
      }
      throw err;
    }
    return { ok: true, generation: 0, trackHash: newTrackHash, rngState: state, generatorVersion: versionTwo() };
  }

  exportBytes() {
    return this._requireDb().export();
  }

  // -------------------------------------------------------------------------
  // setPersistenceHooks({ onBanner, persistSkipped }) — the ONLY surface db.js
  // has to the outside world. db.js NEVER renders DOM; every failure taxonomy
  // path fires onBanner({ type, persistent: true, message, detail? }) and the
  // caller owns rendering (and the "never auto-hides" lifetime).
  //   onBanner(payload)      — receives typed banner payloads. payload.type ∈
  //                            'db-verify-failed' | 'reconcile-required' |
  //                            'db-not-persisting' | 'persist-skipped'
  //   persistSkipped()       — optional predicate: return true while the
  //                            session is logging-disabled ⇒ persist() skips
  //                            with the distinct SEC-26 banner and does NOTHING
  //                            (no file, no IDB, no verify).
  // -------------------------------------------------------------------------
  setPersistenceHooks(hooks) {
    hooks = hooks || {};
    if (hooks.onBanner !== undefined && typeof hooks.onBanner !== 'function') {
      throw new ValidationError('setPersistenceHooks: onBanner must be a function');
    }
    if (hooks.persistSkipped !== undefined && typeof hooks.persistSkipped !== 'function') {
      throw new ValidationError('setPersistenceHooks: persistSkipped must be a function');
    }
    this._hooks = {
      onBanner: hooks.onBanner || null,
      persistSkipped: hooks.persistSkipped || null,
    };
    return this;
  }

  _banner(payload) {
    if (!this._hooks.onBanner) return;
    try {
      this._hooks.onBanner(payload);
    } catch (err) {
      // A throwing banner hook must never break persistence — report and move on.
      console.error('[db] onBanner hook threw (persist protocol continues):', err);
    }
  }

  // -------------------------------------------------------------------------
  // IndexedDB — feature-detected; every method degrades to a typed
  // { ok: false, reason } instead of throwing in node or in browsers without
  // IDB. ONE object store ('kv') in the 'car_racer' database; keys are the
  // full plan key strings ('car_racer.lastKnownGood',
  // 'car_racer.lineageSnapshot'). Values are structured-clone objects
  // ({bytes, tag}). ADR-0004: 'car_racer.dbCache' is RETIRED — lastKnownGood
  // is THE single browser-cache slot.
  // -------------------------------------------------------------------------
  _idbAvailable() {
    return typeof indexedDB !== 'undefined' && indexedDB !== null;
  }

  _idbOpen() {
    return new Promise(function (resolve, reject) {
      var req;
      try {
        req = indexedDB.open('car_racer', 1);
      } catch (err) {
        reject(err);
        return;
      }
      req.onupgradeneeded = function () {
        var idb = req.result;
        if (!idb.objectStoreNames.contains('kv')) {
          idb.createObjectStore('kv', { keyPath: null });
        }
      };
      req.onsuccess = function () { resolve(req.result); };
      req.onerror = function () { reject(req.error || new Error('indexedDB.open failed')); };
      req.onblocked = function () { reject(new Error('indexedDB.open blocked')); };
    });
  }

  async _idbPut(key, value) {
    if (!this._idbAvailable()) return { ok: false, reason: 'idb-unavailable' };
    try {
      var idb = await this._idbOpen();
      return await new Promise(function (resolve) {
        var settled = false;
        function done(result) {
          if (settled) return;
          settled = true;
          try { idb.close(); } catch (closeErr) { /* best effort */ }
          resolve(result);
        }
        try {
          var tx = idb.transaction('kv', 'readwrite');
          tx.objectStore('kv').put(value, key);
          tx.oncomplete = function () { done({ ok: true }); };
          tx.onerror = function () { done({ ok: false, reason: 'idb-error', error: errorMessage(tx.error) }); };
          tx.onabort = function () { done({ ok: false, reason: 'idb-abort', error: errorMessage(tx.error) }); };
        } catch (err) {
          done({ ok: false, reason: 'idb-error', error: errorMessage(err) });
        }
      });
    } catch (err) {
      return { ok: false, reason: 'idb-error', error: errorMessage(err) };
    }
  }

  async _idbGet(key) {
    if (!this._idbAvailable()) return { ok: false, reason: 'idb-unavailable', value: null };
    try {
      var idb = await this._idbOpen();
      return await new Promise(function (resolve) {
        var settled = false;
        function done(result) {
          if (settled) return;
          settled = true;
          try { idb.close(); } catch (closeErr) { /* best effort */ }
          resolve(result);
        }
        try {
          var tx = idb.transaction('kv', 'readonly');
          var req = tx.objectStore('kv').get(key);
          req.onsuccess = function () {
            done({ ok: true, value: req.result === undefined ? null : req.result });
          };
          tx.onerror = function () { done({ ok: false, reason: 'idb-error', error: errorMessage(tx.error), value: null }); };
          tx.onabort = function () { done({ ok: false, reason: 'idb-abort', error: errorMessage(tx.error), value: null }); };
        } catch (err) {
          done({ ok: false, reason: 'idb-error', error: errorMessage(err), value: null });
        }
      });
    } catch (err) {
      return { ok: false, reason: 'idb-error', error: errorMessage(err), value: null };
    }
  }

  async _idbDelete(key) {
    if (!this._idbAvailable()) return { ok: false, reason: 'idb-unavailable' };
    try {
      var idb = await this._idbOpen();
      return await new Promise(function (resolve) {
        var settled = false;
        function done(result) {
          if (settled) return;
          settled = true;
          try { idb.close(); } catch (closeErr) { /* best effort */ }
          resolve(result);
        }
        try {
          var tx = idb.transaction('kv', 'readwrite');
          tx.objectStore('kv').delete(key);
          tx.oncomplete = function () { done({ ok: true }); };
          tx.onerror = function () { done({ ok: false, reason: 'idb-error', error: errorMessage(tx.error) }); };
          tx.onabort = function () { done({ ok: false, reason: 'idb-abort', error: errorMessage(tx.error) }); };
        } catch (err) {
          done({ ok: false, reason: 'idb-error', error: errorMessage(err) });
        }
      });
    } catch (err) {
      return { ok: false, reason: 'idb-error', error: errorMessage(err) };
    }
  }

  // -------------------------------------------------------------------------
  // Fingerprints — { size, sha256 | (first4Kib, last4Kib) }. SHA-256 via
  // crypto.subtle when available (browsers + node ≥ 19); otherwise the
  // documented fallback: size + first/last 4 KiB (both stored as hex). For
  // files ≤ 8 KiB the two 4 KiB edges overlap and together cover the whole
  // file; for larger files the fallback is siz+edges (equal-length corruption
  // mid-file is NOT detectable by the fallback — README documents this).
  // -------------------------------------------------------------------------
  async _fingerprintBytes(bytes) {
    var size = bytes.length;
    var subtle = globalThis.crypto && globalThis.crypto.subtle;
    if (subtle && typeof subtle.digest === 'function') {
      var digest = await subtle.digest('SHA-256', bytes);
      return { size: size, sha256: bytesToHex(new Uint8Array(digest)), first4Kib: null, last4Kib: null };
    }
    return {
      size: size,
      sha256: null,
      first4Kib: bytesToHex(bytes.subarray(0, 4096)),
      last4Kib: bytesToHex(bytes.subarray(Math.max(0, size - 4096))),
    };
  }

  async _fingerprintFile(file) {
    var buf = await file.arrayBuffer();
    return this._fingerprintBytes(new Uint8Array(buf));
  }

  _sameFingerprint(a, b) {
    if (!a || !b) return false;
    if (a.size !== b.size) return false;
    if (a.sha256 && b.sha256) return a.sha256 === b.sha256;
    if (a.sha256 || b.sha256) return false; // one side hashed, the other did not — cannot prove equality; fail closed
    return a.first4Kib === b.first4Kib && a.last4Kib === b.last4Kib;
  }

  // _coherenceTag(bytes) — PURE FUNCTION OF THE BYTES (F-4): opens a THROWAWAY
  // instance over the given export bytes (same pattern as _verifyForPersist)
  // and reads MAX(generation) FROM THE THROWAWAY — never from this.db. Step 4
  // writes the tag NEXT TO the bytes it describes; tagging from the live db
  // instead could stamp "gen N" onto bytes that predate gen N and poison the
  // _refreshLineageSnapshot freshness baseline. Returns
  // { maxGeneration, trackHash, rngState }. Throws ValidationError on any
  // failure (persist() catches it as a step-4-class failure).
  _coherenceTag(bytes) {
    if (!(bytes instanceof Uint8Array)) {
      throw new ValidationError('coherenceTag: bytes must be a Uint8Array, got ' + typeof bytes);
    }
    if (bytes.length === 0) {
      throw new ValidationError('coherenceTag: bytes are empty');
    }
    if (bytes.length > MAX_DB_BYTES) {
      throw new ValidationError('coherenceTag: bytes exceed the ' + MAX_DB_BYTES + '-byte (64 MB) cap');
    }
    var SQL = requireSql();
    var throwaway = null;
    try {
      throwaway = new SQL.Database(bytes);
    } catch (err) {
      throw new ValidationError('coherenceTag: failed to open bytes as SQLite: ' + errorMessage(err));
    }
    try {
      var maxRow = throwaway.exec('SELECT COALESCE(MAX(generation), 0) FROM generations');
      var maxGeneration = maxRow && maxRow[0] && maxRow[0].values.length
        ? Number(maxRow[0].values[0][0]) || 0
        : 0;
      var rngRaw = null;
      var rngRow = throwaway.exec("SELECT value FROM meta WHERE key = ?", ['rng_state']);
      if (rngRow && rngRow[0] && rngRow[0].values.length) {
        rngRaw = rngRow[0].values[0][0];
      }
      var rngState = null;
      if (rngRaw !== null && rngRaw !== undefined) {
        try { rngState = strictUint32(rngRaw, 'meta.rng_state'); } catch (err) { /* best effort */ rngState = null; }
      }
      var trackHash = null;
      var trackRow = throwaway.exec("SELECT value FROM meta WHERE key = ?", ['track_hash']);
      if (trackRow && trackRow[0] && trackRow[0].values.length) {
        trackHash = trackRow[0].values[0][0];
      }
      return { maxGeneration: maxGeneration, trackHash: trackHash, rngState: rngState, rotationIndex: Math.floor((maxGeneration > 0 ? maxGeneration - 1 : 0) / ROTATE_EVERY) };
    } finally {
      try { throwaway.close(); } catch (closeErr) { /* best effort */ }
    }
  }

  // _lsgShouldOverwrite(storedTag, newTag) — the LSG NO-REGRESS RULE (F-3),
  // exposed as a static pure function for tests. FIT-V3 re-spec — STRICTLY
  // incoming-newer wins, no exceptions:
  //   stored == null                    ⇒ overwrite (missing-stored case)
  //   newTag.maxGeneration > stored     ⇒ overwrite (strictly newer)
  //   newTag.maxGeneration < stored     ⇒ do NOT overwrite (stored ahead)
  //   equal generation                  ⇒ do NOT overwrite — whether the
  //                                        hash matches (same lineage, same
  //                                        tip — the cache is equal-or-ahead)
  //                                        or differs (cache/database
  //                                        diverge at generation N — the
  //                                        committed DATABASE is untouched,
  //                                        the cache slot keeps its copy).
  //                                        The USER-FACING banner for the
  //                                        divergent case is M6c's job — this
  //                                        layer returns the boolean only.
  // Unreadable tags (NaN) ⇒ overwrite (kept). (The stored tag accepts both
  // camelCase trackHash and a legacy snake_case track_hash spelling.) Pure
  // function — no I/O, no this.
  static _lsgShouldOverwrite(storedTag, newTag) {
    if (!storedTag) return true;
    var storedGen = Number(storedTag.maxGeneration);
    var newGen = newTag ? Number(newTag.maxGeneration) : NaN;
    if (!Number.isFinite(storedGen) || !Number.isFinite(newGen)) return true;
    return newGen > storedGen; // strictly incoming-newer wins; equal generation keeps the cache (either hash outcome)
  }

  // Create/refresh the never-auto-overwritten-within-a-lineage second slot
  // (_lineageSnapshot) under the FULL step-6 rule — DELEGATED verbatim to
  // _lsgShouldOverwrite (the FIT-V3 re-spec F-5: ONE rule, one function —
  // this site used to carry an inlined copy that could drift). The stored
  // tag keeps the legacy {maxGeneration, trackHash} shape plus rotationIndex
  // when the caller provides it (observability only — never load-bearing).
  async _refreshLineageSnapshot(tag, bytes) {
    var existing = await this._idbGet('car_racer.lineageSnapshot');
    var storedTag = existing && existing.ok && existing.value && existing.value.tag
      ? existing.value.tag
      : null;
    if (storedTag && !Db._lsgShouldOverwrite(storedTag, tag)) {
      return { ok: true, refreshed: false };
    }
    return this._idbPut('car_racer.lineageSnapshot', {
      bytes: bytes,
      tag: { maxGeneration: tag.maxGeneration, trackHash: tag.trackHash || null, rotationIndex: tag.rotationIndex },
      savedAt: new Date().toISOString(),
    });
  }

  // resetLineageSnapshot() — the ONLY sanctioned overwrite of the
  // "never-auto-overwritten" second slot: delete + recreate with the FRESH
  // tag namespace (M-NEW8). For a fresh/forked lineage the tag is
  // { maxGeneration: 0, trackHash: current or null } and bytes:null
  // ("nothing restorable yet") — reconcile must NOT offer a null-bytes
  // snapshot. The M4 fork flow calls this AFTER the bytes are adopted/fresh
  // and BEFORE the new lineage's first persist. In node (no IDB) this is a
  // typed no-op.
  async resetLineageSnapshot() {
    var trackHash = null;
    try {
      trackHash = this.db ? this._metaValue('track_hash') : null;
    } catch (err) {
      /* best effort */
    }
    var del = await this._idbDelete('car_racer.lineageSnapshot');
    if (!del.ok) return del;
    return this._idbPut('car_racer.lineageSnapshot', {
      bytes: null,
      tag: { maxGeneration: 0, trackHash: trackHash },
      savedAt: new Date().toISOString(),
      freshLineage: true,
    });
  }

  // -------------------------------------------------------------------------
  // persist() — plan §PERSIST, the single ordered pipeline, one buffer
  // end-to-end. Awaited by the session layer AFTER each generation yields.
  // NEVER throws for the defined failure taxonomy — every failure path fires
  // onBanner(persistent: true) and resolves { ok: false, reason }. Returns
  // { ok: true, bytesWritten, tag } on success (bytesWritten is the exported
  // buffer's byte count — the M4 caller compares it against the baseline;
  // nothing in the success object is named bare `bytes`).
  //
  //   Step 0 — SEC-26: while persistSkipped() reports the session is
  //            logging-disabled, persist() is SKIPPED (distinct banner,
  //            NOTHING is written — not the file, not IDB, not verify).
  //   Step 1 — exportBytes() ONCE; the same buffer is verified, written and
  //            digested (single buffer end-to-end).
  //   Step 2 — throwaway _verifyForPersist(bytes) (pure function of bytes).
  //            On failure: persistent banner, IDB/file UNTOUCHED, NOTHING
  //            cached — committed prefix intact; resume recomputes it
  //            identically (N1-E).
  //   Step 3 — freshness tripwire (file mode only): handle.getFile() →
  //            { size, sha256 } vs the baseline recorded at session open /
  //            after the last successful persist / after the last
  //            battery-passed adoption. ANY mismatch ⇒ DO NOT WRITE ⇒
  //            blocking Reconcile banner; logging stays disabled until the
  //            user resolves it.
  //   Step 4 — lastKnownGood: written AFTER 2–3 pass, BEFORE the file is
  //            touched (SEC-16) — an IDB copy of the last verified-good
  //            state always survives a failed persist. F-3 NO-REGRESS RULE:
  //            when IDB holds a stored tag with maxGeneration >= the incoming
  //            tag's AND the same trackHash, the slot is NOT overwritten — a
  //            'cache-ahead' banner fires and the (newer) cached copy is
  //            kept; steps 5–6 + the lineageSnapshot refresh run anyway. F-4:
  //            the tag is computed FROM THE SNAPSHOT BYTES (throwaway
  //            instance), never from this.db. On write failure we fail
  //            CLOSED: no file write without a cache copy (the plan's
  //            invariant) — but a no-regress skip is NOT a failure.
  //   Step 5 — write THE SAME verified bytes: createWritable({mode:
  //            'exclusive'}) (Chrome/Edge ≥ 121) — in node/tests the seam
  //            _createWritable(handle) is overridden instead. When
  //            navigator.locks is available, steps 3–6 run inside ONE
  //            exclusive 'car_racer-db' lock (SEC-25). Note: support cannot
  //            be probed without calling — a rejection at createWritable
  //            (NotAllowedError/NoModificationAllowedError/…) lands in the
  //            step-7 taxonomy (banner + training continues), never a crash.
  //            Write bytes; close().
  //   Step 6 — post-close verification: getFile(); size === bytes.length AND
  //            SHA-256 matches digest(bytes) (fallback: size + first/last
  //            4 KiB). Mismatch ⇒ persistent banner "DB NOT PERSISTING —
  //            export manually". On success ONLY: baseline refreshed +
  //            _refreshLineageSnapshot under the full tag rule.
  //   Step 7 — ANY failure at steps 2–6: writable.abort() if a writable is
  //            open (no .crswap litter), persistent banner, training
  //            continues; the banner never auto-hides.
  //   Step 8 — IDB-fallback mode (no file handle): steps 5–6 become "nothing
  //            further to write" — the LSG slot was ALREADY written in step 4
  //            (THE single browser-cache slot; the former 'car_racer.dbCache'
  //            name is retired — ADR-0004 / §PERSIST as amended) — + refresh
  //            the lineageSnapshot under the full step-6 tag rule. Export/
  //            Import remains the user's escape hatch.
  // -------------------------------------------------------------------------
  async persist() {
    // F-5 — same-tab serialization: EVERY persist() call chains onto the
    // in-instance promise chain. Lock layering (innermost first): this
    // _persistChain orders same-tab persists; the navigator.locks
    // 'car_racer-db' exclusive lock INSIDE the pipeline orders cross-tab
    // writers where supported. Either layer alone is insufficient (locks are
    // absent in node/file:///older engines), so both exist — the chain is
    // the unconditional floor. The chain never rejects (settled with a
    // swallowing catch) so one failure cannot wedge later persists; each
    // caller still sees ITS OWN promise resolve/reject as usual.
    var out = this._persistChain.then(function () {
      // Step 0 — SEC-26: while persistSkipped() reports the session is
      // logging-disabled, persist() is SKIPPED (distinct banner, NOTHING is
      // written — not the file, not IDB, not verify). Skipped calls chain
      // like any other persist (F-5: EVERY persist() chains) — a skip is a
      // zero-write persist, and chain order keeps banner ordering identical
      // to write ordering.
      if (this._hooks.persistSkipped && this._hooks.persistSkipped()) {
        this._banner({ type: 'persist-skipped', persistent: true, message: 'LOGGING DISABLED — persist skipped' });
        return { ok: false, skipped: true, reason: 'logging-disabled' };
      }
      this._requireDb();
      return this._persistPipeline();
    }.bind(this));
    this._persistChain = out.catch(function swallowPersistRejection() {
      // The chain must never reject — one failed persist cannot wedge the
      // next. The CALLER still sees `out` (with its own resolution/failure).
    });
    return out;
  }

  async _persistPipeline() {
    // Step 1 — export ONCE.
    var bytes;
    try {
      bytes = this.db.export();
    } catch (err) {
      return this._persistFail('export', err);
    }

    // Step 2 — throwaway verify; failure ⇒ nothing cached, nothing touched.
    try {
      this._verifyForPersist(bytes);
    } catch (err) {
      this._banner({ type: 'db-verify-failed', persistent: true, message: 'DB NOT PERSISTING — export manually', detail: 'step-2 throwaway verify failed: ' + errorMessage(err) });
      return { ok: false, reason: 'verify', error: errorMessage(err) };
    }

    var fingerprint;
    try {
      fingerprint = await this._fingerprintBytes(bytes);
    } catch (err) {
      // crypto.subtle rejection/permission/quota — SAME failure taxonomy as
      // any other persist failure: persistent banner, { ok:false }, and the
      // run CONTINUES (never a rejection escaping into the session layer).
      return this._persistFail('fingerprint', err);
    }

    // Steps 3–6 inside ONE exclusive navigator.locks request when available.
    var locks = (typeof navigator !== 'undefined' && navigator.locks && typeof navigator.locks.request === 'function')
      ? navigator.locks
      : null;
    if (locks) {
      try {
        return await locks.request('car_racer-db', { mode: 'exclusive' }, function () {
          return this._persistSteps3to6(bytes, fingerprint);
        }.bind(this));
      } catch (err) {
        return this._persistFail('lock', err);
      }
    }
    return this._persistSteps3to6(bytes, fingerprint);
  }

  async _persistSteps3to6(bytes, fingerprint) {
    var handle = this._fileHandle || null;
    var writable = null;
    var writableOpen = false;
    try {
      // Step 3 — freshness tripwire (file mode only).
      if (handle) {
        var file = await handle.getFile();
        var current = await this._fingerprintFile(file);
        if (!this._sameFingerprint(this._freshnessAtOpen, current)) {
          this._banner({ type: 'reconcile-required', persistent: true, message: 'Database file changed outside this tab — Reconcile required' });
          return { ok: false, reason: 'freshness', reconcile: true };
        }
      }
      // Step 4 — LSG AFTER 2–3, BEFORE the file is touched. Fail closed.
      // F-3 — LSG NO-REGRESS RULE: read the stored {maxGeneration,
      // track_hash} tag first; if the browser cache is AHEAD of (or equal
      // to) the incoming tag on the SAME lineage, DO NOT overwrite it —
      // emit a cache-ahead banner and let training continue. The LSG stays
      // the newer copy. Never blocks the file path: steps 5–6 and the
      // lineageSnapshot refresh below run regardless.
      var tag = this._coherenceTag(bytes); // F-4 — tag describes THE BYTES, not this.db
      var lsgSkip = false;
      try {
        var storedLsg = await this._idbGet('car_racer.lastKnownGood');
        var storedTag = (storedLsg && storedLsg.ok && storedLsg.value && storedLsg.value.tag) || null;
        if (storedTag && !Db._lsgShouldOverwrite(storedTag, tag)) {
          lsgSkip = true;
          var storedGenSkip = Number(storedTag.maxGeneration);
          // FIT-V3 (M6c, the F-5 rule) — same generation but DIFFERENT hash
          // means cache and database diverge at generation N. Both keep their
          // bytes (the cache slot stays; the committed database is untouched)
          // and the user gets a DISTINCT banner — never the 'cache is AHEAD'
          // wording, which is only honest for the same-lineage equal-or-ahead
          // case that follows.
          var storedHashRaw = storedTag.trackHash !== undefined && storedTag.trackHash !== null
            ? storedTag.trackHash
            : storedTag.track_hash; // tolerate the legacy snake_case spelling
          var sameLineageSkip = storedHashRaw != null && tag.trackHash != null && String(storedHashRaw) === String(tag.trackHash);
          if (storedGenSkip === Number(tag.maxGeneration) && !sameLineageSkip) {
            this._banner({
              type: 'cache-diverge',
              generation: storedGenSkip,
              reason: 'cache and database diverge at generation ' + storedGenSkip,
              message: 'cache and database diverge at generation ' + storedGenSkip + ' — kept the browser cache slot; the committed database on disk is untouched',
            });
          } else {
            this._banner({
              type: 'cache-ahead',
              generation: storedGenSkip,
              reason: 'browser cache is current or ahead (gen ' + storedGenSkip + ') — not overwritten by gen ' + tag.maxGeneration,
              message: 'browser cache is current or ahead (gen ' + storedGenSkip + ') — not overwritten by gen ' + tag.maxGeneration,
            });
          }
        }
      } catch (err) {
        // An unreadable stored tag must never block a persist — treat the
        // slot as absent and overwrite (the write itself fail-closes below).
        lsgSkip = false;
      }
      if (!lsgSkip) {
        var lsg = await this._idbPut('car_racer.lastKnownGood', {
          bytes: bytes,
          tag: { maxGeneration: tag.maxGeneration, rngState: tag.rngState, trackHash: tag.trackHash, rotationIndex: tag.rotationIndex },
          savedAt: new Date().toISOString(),
        });
        if (!lsg || !lsg.ok) {
          this._banner({ type: 'db-not-persisting', persistent: true, message: 'DB NOT PERSISTING — export manually', detail: 'lastKnownGood write failed: ' + ((lsg && lsg.reason) || 'unknown') });
          return { ok: false, reason: 'idb-lastknowngood' };
        }
      }
      // Step 5/6 — file mode: exclusive write + post-close verification.
      if (handle) {
        writable = await this._createWritable(handle);
        writableOpen = true;
        await writable.write(bytes);
        await writable.close();
        writableOpen = false;
        var verifyFile = await handle.getFile();
        var post = await this._fingerprintFile(verifyFile);
        if (post.size !== fingerprint.size || !this._sameFingerprint(fingerprint, post)) {
          this._banner({ type: 'db-not-persisting', persistent: true, message: 'DB NOT PERSISTING — export manually', detail: 'post-close verification mismatch (size or SHA-256)' });
          return { ok: false, reason: 'post-close' };
        }
      }
      // Step 8 — IDB-fallback (ADR-0004): the LSG slot was ALREADY written in
      // step 4 above — THE single browser-cache slot; the former
      // 'car_racer.dbCache' write is retired (§PERSIST as amended). Nothing
      // further to write here; the lineageSnapshot refresh below applies the
      // full step-6 tag rule.
      // SUCCESS — the ONLY place the baseline is refreshed.
      this._freshnessAtOpen = fingerprint;
      await this._refreshLineageSnapshot(tag, bytes);
      return { ok: true, bytesWritten: bytes.length, tag: tag };
    } catch (err) {
      // Step 7 — failure taxonomy: abort any open writable (no .crswap
      // litter), persistent banner, caller decides continue-vs-halt (a
      // UNIQUE lineage-integrity conflict never reaches persist()).
      if (writableOpen && writable) {
        try { await writable.abort(); } catch (abortErr) { /* best effort */ }
      }
      return this._persistFail('write', err);
    }
  }

  _persistFail(reason, err) {
    var msg = errorMessage(err);
    this._banner({ type: 'db-not-persisting', persistent: true, message: 'DB NOT PERSISTING — export manually', detail: reason + ': ' + msg });
    return { ok: false, reason: reason, error: msg };
  }

  // _createWritable(handle) — the test seam for node: tests override this
  // (or pass a stub handle) to drive the whole step 3–6 pipeline without a
  // browser. Default: the plan's exclusive-mode createWritable.
  async _createWritable(handle) {
    if (!handle || typeof handle.createWritable !== 'function') {
      throw new Error('persist: file handle does not support createWritable (exclusive write) — read-only mode or IDB fallback');
    }
    return handle.createWritable({ mode: 'exclusive' });
  }

  // -------------------------------------------------------------------------
  // openExisting({ bytes } | { handle } | { pick: true }) — adopt an
  // EXISTING database as the working DB (plan §PERSIST startup: "battery
  // before adoption; failure ⇒ zero adoption"). The ADOPTION BATTERY runs
  // BEFORE the working DB is touched; on failure the working DB is left
  // EXACTLY as it was. Returns { ok, generation, trackHash, rngState }.
  //   { bytes }  — tests/node (bytes path for the smoke legs)
  //   { handle } — an already-held FileSystemFileHandle (e.g. after showOpenFilePicker)
  //   { pick: true } — browser: showOpenFilePicker, .sqlite-ish filters
  // -------------------------------------------------------------------------
  async openExisting(opts) {
    opts = opts || {};
    var bytes = null;
    var handle = null;
    if (opts.bytes !== undefined) {
      if (!(opts.bytes instanceof Uint8Array)) {
        throw new ValidationError('openExisting({ bytes }) expects a Uint8Array');
      }
      bytes = opts.bytes;
    } else if (opts.handle) {
      handle = opts.handle;
      var handleFile;
      try {
        handleFile = await handle.getFile();
      } catch (e) {
        throw new ValidationError('openExisting: could not read file handle — ' + (e && e.message));
      }
      // Size gate BEFORE arrayBuffer() is ever called — a multi-gigabyte
      // "database" never even reaches the reader (security M3).
      if (handleFile.size > MAX_DB_BYTES) {
        throw new ValidationError('adoption battery: file too large (' + handleFile.size + ' bytes)');
      }
      try {
        bytes = new Uint8Array(await handleFile.arrayBuffer());
      } catch (err) {
        throw new ValidationError('openExisting: unable to read the file bytes: ' + errorMessage(err));
      }
    } else if (opts.pick) {
      if (typeof globalThis.showOpenFilePicker !== 'function') {
        throw new DbOfflineError('openExisting: showOpenFilePicker is not available in this browser — use Import .sqlite instead (see README)');
      }
      var handles = await globalThis.showOpenFilePicker({
        multiple: false,
        types: [{ description: 'SQLite database', accept: { 'application/octet-stream': ['.sqlite', '.sqlite3', '.db'] } }],
      });
      handle = handles && handles[0];
      if (!handle) {
        throw new ValidationError('openExisting: no file was picked');
      }
      var pickedFile;
      try {
        pickedFile = await handle.getFile();
      } catch (e) {
        throw new ValidationError('openExisting: could not read file handle — ' + (e && e.message));
      }
      if (pickedFile.size > MAX_DB_BYTES) {
        throw new ValidationError('adoption battery: file too large (' + pickedFile.size + ' bytes)');
      }
      try {
        bytes = new Uint8Array(await pickedFile.arrayBuffer());
      } catch (err) {
        throw new ValidationError('openExisting: unable to read the picked file bytes: ' + errorMessage(err));
      }
    } else {
      throw new ValidationError('openExisting requires { bytes } (tests/node) | { handle } | { pick: true }');
    }
    var summary = Db.validateAdoptionBytes(bytes); // battery FIRST — failure ⇒ zero adoption
    await this._activateBytes(bytes, handle);
    return { ok: true, generation: summary.generation, trackHash: summary.trackHash, rngState: summary.rngState };
  }

  // importBytes(bytes, { confirm }) — the Import .sqlite path. The battery
  // runs BEFORE the confirm callback; confirm()===true applies the bytes as
  // the new working DB (the file handle is DETACHED — M4 re-attaches
  // explicitly; the previously connected file holds the pre-import lineage
  // and persisting over it silently would corrupt the user's last copy).
  // WITHOUT a confirm callback: REFUSAL, typed ValidationError, zero adoption.
  // confirm()===false ⇒ { applied: false, reason: 'confirm-declined' }.
  async importBytes(bytes, opts) {
    opts = opts || {};
    if (typeof opts.confirm !== 'function') {
      throw new ValidationError('importBytes REFUSES to proceed without a confirm callback — pass { confirm: (generation) => boolean }; the working database is never replaced without explicit confirmation');
    }
    var summary = Db.validateAdoptionBytes(bytes); // battery FIRST; confirm is never consulted on failure
    var decision = opts.confirm(summary.generation);
    if (decision !== true) {
      return { applied: false, generation: summary.generation, reason: 'confirm-declined' };
    }
    await this._activateBytes(bytes, null); // detached; M4 re-attaches a file explicitly if wanted
    return { applied: true, generation: summary.generation, trackHash: summary.trackHash, rngState: summary.rngState };
  }

  // _activateBytes(bytes, handle) — shared adoption tail: baseline
  // from the adopted bytes (M-NEW5/F9), working DB swapped AFTER the battery
  // passed, FK verified, scrub re-run on the new connection. A fingerprint
  // failure is a typed ValidationError (the WORKING DB is untouched).
  async _activateBytes(bytes, handle) {
    var SQL = requireSql();
    var digest;
    try {
      digest = await this._fingerprintBytes(bytes); // baseline FROM THE ADOPTED BYTES
    } catch (err) {
      throw new ValidationError('adoption: fingerprint failed — refusing to adopt (working database untouched): ' + errorMessage(err));
    }
    var fresh;
    try {
      // Battery-passed bytes can STILL make sql.js throw — typed here, never
      // a raw escape (security L6); the WORKING DB is untouched.
      fresh = new SQL.Database(bytes); // verified locally BEFORE the working DB is touched
    } catch (err) {
      throw new ValidationError('adoption: database construction failed — refusing to adopt (working database untouched): ' + errorMessage(err));
    }
    try {
      this._enableAndVerifyForeignKeys(fresh);
      // Belt-and-braces (the uv-aware twin of init's own scrub): uv>=3 raw
      // bytes get the five-table scrub, uv∈{1,2} the four-table one — an
      // honest v4 export adopted here must not be refused by the gate that
      // exists to keep foreign objects out. Pre-v4 bytes never reach here
      // (typed-refused at the battery's step 3b).
      var rawUvRows = fresh.exec('PRAGMA user_version');
      var rawUv = rawUvRows && rawUvRows[0] && rawUvRows[0].values && rawUvRows[0].values[0]
        ? Number(rawUvRows[0].values[0][0])
        : NaN;
      scrubSqliteMaster(fresh, 'adoption', rawUv >= 3 ? EXPECTED_TABLES_V3 : undefined);
      // Battery-passed bytes hit the v4 migration gate here on this same call
      // (the only bytes that ever reach here — the battery runs on THESE bytes
      // first): uv4 is a verified no-op; pre-v4 bytes were already refused.
      this._migrateIfNeeded(fresh);
    } catch (err) {
      try { fresh.close(); } catch (closeErr) { /* best effort */ }
      throw err; // the WORKING DB is untouched — failure ⇒ zero adoption
    }
    if (this.db) {
      try { this.db.close(); } catch (closeErr) { /* best effort */ }
    }
    this.db = fresh;
    this._fileHandle = handle || null;
    this._freshnessAtOpen = digest; // adopting path OWNS the baseline reset
    return { ok: true };
  }

  // -------------------------------------------------------------------------
  // getPersistCost() — ONE honest per-persist measurement (L-NEW10): full
  // export + throwaway verify (integrity_check + architecture + coherence)
  // + (node only) fs.writeFile to os.tmpdir() as a proxy for the file-write
  // leg. Returns { bytes, exportMs, verifyMs, fsWriteMs, fsWriteError,
  // totalMs, note }. The M2 status.md must cite THESE numbers; no
  // 5000-generation wall-clock claim enters the README unmeasured.
  // -------------------------------------------------------------------------
  async getPersistCost() {
    var db = this._requireDb();
    var t0 = nowMs();
    var bytes = db.export();
    var t1 = nowMs();
    this._verifyForPersist(bytes);
    var t2 = nowMs();
    var fsWriteMs = null;
    var fsWriteError = null;
    var isNode = typeof process !== 'undefined' && process.versions && typeof process.versions.node === 'string';
    if (isNode) {
      try {
        var req;
        try {
          req = require;
        } catch (err) {
          throw err;
        }
        var fs = req('node:fs');
        var os = req('node:os');
        var path = req('node:path');
        var tmp = path.join(os.tmpdir(), 'car_racer-persist-cost-' + process.pid + '-' + String(Date.now()) + '.sqlite');
        var tA = nowMs();
        await fs.promises.writeFile(tmp, bytes);
        var tB = nowMs();
        try { await fs.promises.unlink(tmp); } catch (unlinkErr) { /* best effort */ }
        fsWriteMs = tB - tA;
      } catch (err) {
        fsWriteError = errorMessage(err);
      }
    }
    var t3 = nowMs();
    return {
      bytes: bytes.length,
      exportMs: t1 - t0,
      verifyMs: t2 - t1,
      fsWriteMs: fsWriteMs,
      fsWriteError: fsWriteError,
      totalMs: t3 - t0,
      note: 'fsWriteMs is a node proxy for the browser file-write leg; a real browser persist adds one getFile+arrayBuffer read (tripwire) and one more (post-close verify)',
    };
  }
}

// ---------------------------------------------------------------------------
// Module-level helpers (hoisted function declarations; used by the class).
// ---------------------------------------------------------------------------
// normalizeClonedBytes(value) — defensive normalization for bytes that made
// a structured-clone round trip through IDB. Browsers spec-return Uint8Array,
// but an ArrayBuffer / view is accepted and re-wrapped; anything else reads
// as null ("no usable bytes"). Never throws.
function normalizeClonedBytes(value) {
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (ArrayBuffer.isView && ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  }
  return null;
}

function assertPositiveInteger(value, name) {
  if (!Number.isInteger(value) || value < 1) {
    throw new ValidationError(name + ': generation must be an integer >= 1, got ' + JSON.stringify(value));
  }
}

// Trainer payloads carry boolean crashed/stuck; the schema stores INTEGER
// 0/1 with CHECK constraints — this is the normalization boundary.
function normalizeBinaryFlag(value, name) {
  if (value === 0 || value === 1) return value;
  if (value === true) return 1;
  if (value === false) return 0;
  throw new ValidationError(name + ' must be 0/1 (or boolean), got ' + JSON.stringify(value));
}

function rowObject(resultSet) {
  var obj = {};
  for (var c = 0; c < resultSet.columns.length; c++) {
    obj[resultSet.columns[c]] = resultSet.values[0][c];
  }
  return obj;
}

function rowObjects(resultSet) {
  var out = [];
  if (!resultSet || !resultSet.values) return out;
  for (var r = 0; r < resultSet.values.length; r++) {
    var obj = {};
    for (var c = 0; c < resultSet.columns.length; c++) {
      obj[resultSet.columns[c]] = resultSet.values[r][c];
    }
    out.push(obj);
  }
  return out;
}

// FIT-V4: Db re-exports the ONE canonical generator-version parser from
// rotation.js (Task A) — never a local redefinition. A getter keeps
// Db.parseGeneratorVersion === Rotation.parseGeneratorVersion in every realm.
Object.defineProperty(Db, 'parseGeneratorVersion', {
  configurable: true,
  enumerable: true,
  get: function () { return requireVersionParser(); },
});

// Browser-global exposure + guarded CommonJS export (node tests).
if (typeof globalThis !== 'undefined') {
  globalThis.Db = Db;
  globalThis.LineageIntegrityError = LineageIntegrityError;
  globalThis.DbOfflineError = DbOfflineError;
  globalThis.MigrationError = MigrationError;
  globalThis.MAX_WEIGHTS_JSON_LENGTH = MAX_WEIGHTS_JSON_LENGTH;
  globalThis.MAX_TRACK_JSON_LENGTH = MAX_TRACK_JSON_LENGTH;
  globalThis.MAX_DB_BYTES = MAX_DB_BYTES;
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    Db: Db,
    ValidationError: ValidationError,
    LineageIntegrityError: LineageIntegrityError,
    DbOfflineError: DbOfflineError,
    MigrationError: MigrationError,
    ResumeTrackError: ResumeTrackError,
    WallModelChangedError: WallModelChangedError,
    WALL_MODEL_CHANGED_MESSAGE: WALL_MODEL_CHANGED_MESSAGE,
    // FIT-V3: ALL DDL is exported for the smoke legs (Appendix-A twins and
    // the legacy v2 twins).
    DDL_META: DDL_META,
    DDL_TRACKS: DDL_TRACKS,
    DDL_GENERATIONS: DDL_GENERATIONS,
    DDL_GENERATIONS_NEW: DDL_GENERATIONS_NEW,
    DDL_GENERATIONS_V2: DDL_GENERATIONS_V2,
    DDL_GENERATIONS_NEW_V2: DDL_GENERATIONS_NEW_V2,
    DDL_DRIVER_RESULTS: DDL_DRIVER_RESULTS,
    DDL_BEST_WEIGHTS: DDL_BEST_WEIGHTS,
    MAX_WEIGHTS_JSON_LENGTH: MAX_WEIGHTS_JSON_LENGTH,
    MAX_TRACK_JSON_LENGTH: MAX_TRACK_JSON_LENGTH,
    MAX_DB_BYTES: MAX_DB_BYTES,
    MAX_GENERATION_ROWS: MAX_GENERATION_ROWS,
    MAX_DRIVER_RESULTS_ROWS: MAX_DRIVER_RESULTS_ROWS,
    MAX_TRACK_ROWS: MAX_TRACK_ROWS,
    POPULATION_SIZE: POPULATION_SIZE,
    NN_ARCHITECTURE: NN_ARCHITECTURE,
    isLineageIntegrityFault: isLineageIntegrityFault,
  };
}
