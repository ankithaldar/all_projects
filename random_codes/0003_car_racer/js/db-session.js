// js/db-session.js — Session persistence for the 0003_car_racer browser shell.
//
// WHY THIS FILE EXISTS
// The File System Access API can hold a FileSystemFileHandle across page
// reloads (via IndexedDB), which lets a returning session re-attach to the
// user's database file WITHOUT a picker round-trip. This module owns the
// small IndexedDB database 'car_racer-sessions' (schema v2 keeps the
// 'handles' store; the retired 'meta' store is no longer read or written).
//
//   store 'handles'  key 'current'  — the FileSystemFileHandle for the
//                     database the user connected/created last ('fs' mode)
//
// THE BROWSER-CACHE SLOT IS NOT HERE (ADR-0004). The single browser-cache
// slot is db.js's 'car_racer' kv key 'car_racer.lastKnownGood' — written by
// EVERY verified persist, read at boot by Db.restoreFromLastKnownGood()
// through the ADOPTION BATTERY. The former 'dbcache' store (a raw duplicate
// of the same bytes that was written per generation) is DELETED (schema
// version 2); 2× IDB footprint eliminated.
//
// EVERYTHING here is guarded:
//   * typeof indexedDB !== 'undefined' gate before opening anything,
//   * try/catch around every operation,
//   * every failure resolves to a typed { ok:false } / null instead of an
//     exception escaping into boot (boot must survive IDB being blocked,
//     private browsing, quota errors and handle-permission prompts).
//
// Plain classic script: exposes window.DbSession (globalThis.DbSession).
// No innerHTML, no eval, no network, no external anything.
//
// RESTORE SEMANTICS (boot flow, called by main.js once per page load)
//   1. A stored handle exists → handle.queryPermission({ mode:'readwrite' })
//      — NOTE: query ONLY. requestPermission() is NEVER called here; it may
//      only be called from inside a user-gesture click handler (main.js).
//      'granted'  → db.openExisting({ handle }) (ADOPTION BATTERY inside;
//                   failure ⇒ ZERO adoption; the FILE is NEVER adopted —
//                   but the battery-verified browser cache (LSG/snapshot,
//                   restoreFromLastKnownGood) IS tried FIRST as the session
//                   database; only if that also fails does a fresh in-memory
//                   session boot with a banner naming BOTH refusals. FIT-V6:
//                   on a typed WallModelChangedError the handle AND LSG slot
//                   are preserved (byte-preserving refusal) and the refused
//                   bytes ride on the result) ⇒ banner
//                   "Database connected from last session: <name> (N generations)".
//      not granted → a fresh in-memory DB is initialized (so the rest of the
//                   app works) and the result is
//                   { state:'needs-reconnect', handle, name } — main.js shows
//                   the "Connected handle needs permission — Reconnect database"
//                   banner with the Reconnect button.
//   2. No handle → Db.restoreFromLastKnownGood() reads db.js's
//      'car_racer.lastKnownGood' slot, falling back to the battery-verified
//      'car_racer.lineageSnapshot' slot when LSG is refused/unreadable and the
//      snapshot is STRICTLY AHEAD (ADOPTION BATTERY inside either way; total
//      failure ⇒ slot CLEARED, bytes NEVER adopted, fresh in-memory session
//      boots) ⇒ { state:'cache', db: <restored scratch Db> } on success; the
//      banner names WHICH slot won. No bytes ⇒ silent fresh in-memory session
//      ({ state:'memory', message:null }).
//   3. Neither → fresh in-memory DB. state:'memory', no banner.
//
// The db passed to restoreIfPossible() must be a FRESH Db (init() not yet
// called) — it is consumed on the handle paths and on the fresh-memory paths.
// The 'cache' path OWNS ITS OWN scratch Db (restore builds it internally) and
// hands it back in the result's `db` field; callers MUST use restored.db when
// it is present. Exactly one of init() / init(bytes) / openExisting() /
// restoreFromLastKnownGood() initializes the session's database.

'use strict';

var DbSession = (function () {
  var IDB_NAME = 'car_racer-sessions';
  var IDB_VERSION = 2; // v2 — drops the retired 'dbcache' store (ADR-0004)
  var STORE_HANDLES = 'handles';
  var STORE_CACHE = 'dbcache'; // legacy name — referenced ONLY by the v2 upgrade scrub
  var STORE_META = 'meta';
  var KEY_CURRENT = 'current';

  function idbAvailable() {
    return typeof indexedDB !== 'undefined' && indexedDB !== null;
  }

  function openDb() {
    return new Promise(function (resolve, reject) {
      var req;
      try {
        req = indexedDB.open(IDB_NAME, IDB_VERSION);
      } catch (err) {
        reject(err);
        return;
      }
      req.onupgradeneeded = function () {
        var idb = req.result;
        if (!idb.objectStoreNames.contains(STORE_HANDLES)) idb.createObjectStore(STORE_HANDLES, { keyPath: null });
        if (!idb.objectStoreNames.contains(STORE_META)) idb.createObjectStore(STORE_META, { keyPath: null });
        // ADR-0004 — the duplicate bytes copy is retired: drop it for every
        // existing install. Bytes live in db.js's 'car_racer' kv LSG slot.
        if (idb.objectStoreNames.contains(STORE_CACHE)) idb.deleteObjectStore(STORE_CACHE);
      };
      req.onsuccess = function () { resolve(req.result); };
      req.onerror = function () { reject(req.error || new Error('indexedDB.open(' + IDB_NAME + ') failed')); };
      req.onblocked = function () { reject(new Error('indexedDB.open(' + IDB_NAME + ') blocked')); };
    });
  }

  // One guarded store operation. op ∈ 'put' | 'get' | 'delete'.
  // Resolves { ok:true, value? } or { ok:false, reason } — never throws.
  async function storeOp(store, op, key, value) {
    if (!idbAvailable()) return { ok: false, reason: 'idb-unavailable' };
    var idb = null;
    try {
      idb = await openDb();
      var result = await new Promise(function (resolve, reject) {
        var tx = idb.transaction(store, op === 'get' ? 'readonly' : 'readwrite');
        var s = tx.objectStore(store);
        var req;
        if (op === 'get') req = s.get(key);
        else if (op === 'put') req = s.put(value, key);
        else req = s.delete(key);
        req.onsuccess = function () { resolve({ ok: true, value: req.result }); };
        req.onerror = function () { reject(req.error || new Error(store + '.' + op + ' failed')); };
      });
      try { idb.close(); } catch (closeErr) { /* best effort */ }
      return result;
    } catch (err) {
      try { if (idb) idb.close(); } catch (closeErr2) { /* best effort */ }
      return { ok: false, reason: String((err && err.message) || err) };
    }
  }

  function isHandleLike(h) {
    return h !== null && typeof h === 'object' && typeof h.queryPermission === 'function';
  }

  function errMsg(err) {
    return String((err && err.message) || err);
  }

  // FIT-V6 — map the typed WallModelChangedError to the persistent, actionable
  // refusal banner. BYTE-PRESERVING: on this typed refusal we never clear the
  // handle or the LSG slot, and we surface the refused bytes on the result so
  // the UI can offer "download refused bytes verbatim". The handle path catches
  // the error directly; the cache path sees the detail string + refused bytes.
  var WALL_MODEL_BANNER_MARK = 'WALL MODEL CHANGED';
  function wallModelRefusalBanner(text) {
    return WALL_MODEL_BANNER_MARK + ' — ' + text +
      ' — The original was left untouched. Use Export .sqlite to download the refused bytes verbatim, then Create / Fork to start a fresh lineage.';
  }
  function isTypedWallRefusal(err) {
    return !!(err && err.name === 'WallModelChangedError');
  }
  function adoptionRefusal(err) {
    return isTypedWallRefusal(err)
      ? wallModelRefusalBanner(errMsg(err))
      : errMsg(err);
  }
  function wallModelDetail(text) {
    text = String(text || '');
    return /wall model/i.test(text) ? wallModelRefusalBanner(text) : text;
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  // saveHandle(handle) — store the (cloneable) FileSystemFileHandle.
  // Only 'fs' mode should call this. Resolves {ok:true} | {ok:false, reason}.
  async function saveHandle(handle) {
    if (!handle || !isHandleLike(handle)) return { ok: false, reason: 'not-a-file-handle' };
    return storeOp(STORE_HANDLES, 'put', KEY_CURRENT, handle);
  }

  // loadHandle() — resolves the stored handle, or null when absent/unreadable.
  async function loadHandle() {
    var res = await storeOp(STORE_HANDLES, 'get', KEY_CURRENT, undefined);
    if (!res.ok || !res.value) return null;
    return isHandleLike(res.value) ? res.value : null;
  }

  async function clearHandle() {
    return storeOp(STORE_HANDLES, 'delete', KEY_CURRENT, undefined);
  }

  // ---------------------------------------------------------------------------
  // restoreIfPossible(db) — the boot-time restore routine (see file header).
  // db: a FRESH Db instance (init() not yet called). Resolves:
  //   { state:'fs',    name, generation, message }
  //   { state:'cache', name:'(browser cache)', generation, message, db }   — the restored scratch instance
  //   { state:'memory', generation:0, message:null }
  //   { state:'needs-reconnect', name, handle, message }   — banner + button
  //   { state:'unavailable', message }                       — db unusable
  // NEVER throws; every failure path resolves to a typed result.
  // ---------------------------------------------------------------------------
  async function restoreIfPossible(db) {
    // ---- path 1: a stored file handle --------------------------------------
    var handle = null;
    try { handle = await loadHandle(); } catch (err) { handle = null; }

    if (handle) {
      var permission = 'prompt';
      try {
        permission = await handle.queryPermission({ mode: 'readwrite' });
      } catch (err) {
        permission = 'prompt';
      }
      var handleName = typeof handle.name === 'string' ? handle.name : '(unnamed database)';

      if (permission === 'granted') {
        try {
          var summary = await db.openExisting({ handle: handle });
          var gen = summary && Number.isFinite(summary.generation) ? summary.generation : 0;
          return {
            state: 'fs',
            name: handleName,
            generation: gen,
            message: 'Database connected from last session: ' + handleName + ' (' + gen + ' generations)',
          };
        } catch (err) {
          // ADOPTION BATTERY refused (or file unreadable) — ZERO adoption; the
          // FILE is never adopted. FIRST try the battery-verified browser
          // cache: a kill-mid-commit or an external delete can leave newer
          // verified bytes in LSG (F-6). The file is still NEVER adopted — this
          // only prefers the verified cache copy over nothing.
          // FIT-V6 byte preservation: a typed WallModelChangedError must NOT
          // clear the handle or the LSG slot, and its refused bytes ride along
          // on the result for the "download verbatim" action.
          var wallTyped = isTypedWallRefusal(err);
          var refusedBytes = wallTyped && err.refusedBytes ? err.refusedBytes : null;
          var fileRefusal = adoptionRefusal(err);
          var cacheGen = 0;
          var cacheSlot = 'lastKnownGood';
          var restoredCache = null;
          try {
            restoredCache = globalThis.Db && typeof globalThis.Db.restoreFromLastKnownGood === 'function'
              ? await globalThis.Db.restoreFromLastKnownGood()
              : null;
          } catch (restoreErr) {
            restoredCache = null;
          }
          if (restoredCache && restoredCache.ok) {
            cacheGen = restoredCache.summary && Number.isFinite(restoredCache.summary.generation) ? restoredCache.summary.generation : 0;
            cacheSlot = restoredCache.summary && restoredCache.summary.slot === 'lineageSnapshot' ? 'lineageSnapshot' : 'lastKnownGood';
            if (!wallTyped) {
              try { await clearHandle(); } catch (clearErr) { /* best effort — the dead handle must not loop */ }
            }
            var cacheHit = {
              state: 'cache',
              name: '(browser cache)',
              generation: cacheGen,
              db: restoredCache.db,
              message: 'The file could not be adopted (' + fileRefusal + ') — loaded the battery-verified browser cache copy (' + cacheSlot + ', generation ' + cacheGen + ') instead — Export to keep it',
            };
            if (refusedBytes) cacheHit.refusedBytes = refusedBytes;
            return cacheHit;
          }
          var cacheReason = restoredCache && (restoredCache.detail || restoredCache.reason)
            ? String(restoredCache.detail || restoredCache.reason)
            : 'browser cache missing or unreadable';
          if (!wallTyped) {
            try { await clearHandle(); } catch (clearErr) { /* best effort */ }
            // The LSG slot held a battery-refused copy — the cache copy was NOT
            // adopted (the battery re-refuses it on every boot, so nothing ever
            // is) — but leaving it in place re-attempts it forever. Clear it
            // here, mirroring the no-handle path's validation-failure arm.
            try { await globalThis.Db.clearLastKnownGood(); } catch (clearCacheErr) { /* best effort */ }
          }
          try {
            await db.init();
          } catch (initErr) {
            return { state: 'unavailable', message: 'Database required — training disabled (Manual works). Reason: ' + errMsg(initErr) };
          }
          var memResult = {
            state: 'memory',
            name: '',
            generation: 0,
            message: 'Last session\'s database was refused by the adoption battery (' + fileRefusal + ') and the browser cache copy also failed (' + cacheReason + ') — started with a fresh in-memory database. ' + (wallTyped ? 'The original file and browser cache were left untouched.' : 'The file was left untouched.'),
          };
          if (refusedBytes) memResult.refusedBytes = refusedBytes;
          return memResult;
        }
      }

      // Permission not granted — DO NOT call requestPermission here (gesture-only).
      try {
        await db.init();
      } catch (err) {
        return { state: 'unavailable', message: 'Database required — training disabled (Manual works). Reason: ' + errMsg(err) };
      }
      return {
        state: 'needs-reconnect',
        name: handleName,
        generation: null,
        handle: handle,
        message: 'Connected handle needs permission — Reconnect database',
      };
    }

    // ---- path 2: no handle — the verified browser-cache slot (ADR-0004) -----
    var res = null;
    try {
      res = await globalThis.Db.restoreFromLastKnownGood(); // battery inside; failure ⇒ zero adoption
    } catch (err) {
      res = { ok: false, reason: (globalThis.Db ? 'restore-threw' : 'db-missing'), detail: errMsg(err) };
    }

    if (res && res.ok) {
      // The passed-in `db` is NOT used — the restore owns a scratch instance
      // (battery-first) and hands it back here. The working DB for this
      // session IS res.db; main.js adopts it via restored.db. The banner names
      // WHICH slot won (F-7 — the snapshot is the last-resort slot).
      var genCache = res.summary && Number.isFinite(res.summary.generation) ? res.summary.generation : 0;
      var slotWon = res.summary && res.summary.slot === 'lineageSnapshot' ? 'lineageSnapshot' : 'lastKnownGood';
      return {
        state: 'cache',
        name: '(browser cache)',
        generation: genCache,
        db: res.db,
        message: slotWon === 'lineageSnapshot'
          ? 'Loaded last session\'s database from the browser cache snapshot (lineageSnapshot, generation ' + genCache + ' — battery-verified) — Export to keep it'
          : 'Loaded last session\'s database from this browser\'s verified cache (lastKnownGood, generation ' + genCache + ') — Export to keep it',
      };
    }

    if (res && res.reason === 'no-cache') {
      // No bytes at all — a silent fresh in-memory session (today's empty state).
      try {
        await db.init();
      } catch (err) {
        return { state: 'unavailable', message: 'Database required — training disabled (Manual works). Reason: ' + errMsg(err) };
      }
      return { state: 'memory', name: '', generation: 0, message: null };
    }

    // Bytes were present but FAILED validation — normally drop the slot, never
    // load it. FIT-V6 byte preservation: a typed wall-model refusal (identified
    // by the refused bytes + the stable message phrase) leaves the LSG slot
    // untouched and surfaces the refused bytes for a verbatim download.
    var typedCache = !!(res && res.refusedBytes) && /wall model/i.test(String(res.detail || ''));
    if (!typedCache) {
      try { await globalThis.Db.clearLastKnownGood(); } catch (err) { /* best effort */ }
    }
    try {
      await db.init();
    } catch (initErr) {
      return { state: 'unavailable', message: 'Database required — training disabled (Manual works). Reason: ' + errMsg(initErr) };
    }
    var cacheDetail = wallModelDetail(res.detail || res.reason || 'unknown');
    var cacheResult = {
      state: 'memory',
      name: '',
      generation: 0,
      message: 'Cached database failed validation (' + cacheDetail + ') — NOT restored; starting fresh in-memory session — export will become available once you train or open a database',
    };
    if (typedCache && res.refusedBytes) cacheResult.refusedBytes = res.refusedBytes;
    return cacheResult;
  }

  return {
    saveHandle: saveHandle,
    clearHandle: clearHandle,
    restoreIfPossible: restoreIfPossible,
  };
})();

if (typeof globalThis !== 'undefined') {
  globalThis.DbSession = DbSession;
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { DbSession: DbSession };
}
