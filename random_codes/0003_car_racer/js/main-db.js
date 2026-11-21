// js/main-db.js — Banner system + everything around the `state.db` lifecycle
// for the 0003_car_racer browser shell. Plain classic script; exposes the
// banner/DB globals js/main.js calls (classic-script global functions, the
// same pattern as js/replay.js exposing Replay).
//
// WHAT THIS FILE OWNS
//   * The one persistent banner (#persist-banner): showBanner, the reconcile
//     banner (+ its four actions: reload-from-disk / export my copy / fork /
//     restore-from-browser-cache), the reconnect affordance, hideBanner,
//     refreshBannerState, setLoggingIndicator, persistFailSuffix.
//   * The banner-capable DB event hooks + persist plumbing: onDbBanner (the
//     Db setPersistenceHooks onBanner sink), setPersistPill, requireDb,
//     currentLatestGeneration.
//   * File-pick helpers for the import/create flows (isFilePickUnavailable /
//     showFilePickUnavailable — one-line banners; the working DB is never
//     touched on a picker failure).
//   * Export / import / create / open / fork flows: exportDb,
//     importBytesWithConfirm, importDb, onImportFileInputChange, createDb,
//     createDbAtHandle, createFreshCacheDb, openDbPicker, adoptOpenedHandle,
//     reloadFromDisk, restoreFromBrowserCache, forkNewLineage and its outcome
//     helpers (forkSnapshotOutcome / forkSnapshotOutcomeRejected),
//     reconnectDb, applyImportSuccess.
//   * setDbStatusPill (the header DB pill).
//
// This file owns the banner system + the DB lifecycle (create/open/import/
// export/fork/restore/reconnect — the Logs-tab reads stay in js/main.js).
// Sections moved from js/main.js EXCEPT the changes noted here — this file is
// NOT a verbatim copy: requireNoLineageWriters gates at every destructive door
// (F-16; the helper itself lives in js/main.js), the subtle-absent persistent
// banner arm, the create-picker re-stat refuse-overwrite (SEC F-1), and the
// fork/restore-time Db.clearLastKnownGood() calls. `state` lives in js/main.js
// (a shared global `var state`) — this file reads/writes it through the shared
// classic-script scope.
//
// Load position (LOAD-BEARING): AFTER js/replay.js and BEFORE js/spectate.js,
// js/db-session.js, js/editor.js and js/main.js. js/main.js is REQUIRED to
// load after this file — boot() calls its globals (refreshBannerState and
// friends).

'use strict';

var SQLITE_MAX_BYTES = globalThis.MAX_DB_BYTES;

var SQLITE_TYPES = [{ description: 'SQLite database', accept: { 'application/octet-stream': ['.sqlite', '.sqlite3', '.db'] } }];

// file:// / opaque-origin guard for the File System Access picker (M4-fix
// FIX-FILE-URL): showSaveFilePicker/showOpenFilePicker exist but THROW
// (sync TypeError, or a promise rejected with TypeError/NotAllowedError/
// SecurityError/AbortError-without-handle). The app is fully functional from
// a file:// double-click — the working session persists via the browser
// cache; only native file PICKS are unavailable. ONE shared one-line banner;
// the current database is NEVER touched on this path.
function isFilePickUnavailable(err) {
  if (!err) return false;
  var name = err.name || '';
  return name === 'TypeError' || name === 'NotAllowedError' || name === 'SecurityError' || name === 'AbortError';
}
function showFilePickUnavailable() {
  showBanner('File pick unavailable here — this session is saved in your browser cache; use Export DB to save a .sqlite file');
}
/* ============================================================
   2. The one persistent banner
   ============================================================ */

function showBanner(text) {
  var banner = el('persist-banner');
  el('persist-banner-text').textContent = String(text);
  el('persist-banner-actions').replaceChildren();
  el('btn-reconnect-db').classList.add('banner-hidden');
  banner.classList.remove('banner-hidden');
}

var RECONCILE_TEXT = 'Database file changed outside this tab — Reconcile required';

function persistFailSuffix() {
  return state.persistFailed ? ' — ALSO: DB NOT PERSISTING — export manually' : '';
}

function showReconcileBanner(note) {
  var banner = el('persist-banner');
  // Optional `note` (e.g. 'Restore refused: …') is folded INTO the reconcile
  // text so a failed action never wipes its own explanation (M9: the
  // refusal reason survives the re-render; the four actions stay visible).
  el('persist-banner-text').textContent = RECONCILE_TEXT + persistFailSuffix() + (note ? ' — ' + note : '') + ' — the file stays untouched until you pick one of these:';
  var actions = el('persist-banner-actions');
  actions.replaceChildren();
  var reloadBtn = document.createElement('button');
  reloadBtn.type = 'button';
  reloadBtn.textContent = 'Reload from disk (loses this tab\'s gens)';
  reloadBtn.addEventListener('click', reloadFromDisk);
  var exportBtn = document.createElement('button');
  exportBtn.type = 'button';
  exportBtn.textContent = 'Export my copy';
  exportBtn.addEventListener('click', exportDb);
  var forkBtn = document.createElement('button');
  forkBtn.type = 'button';
  forkBtn.textContent = 'Fork new lineage';
  forkBtn.addEventListener('click', function () { forkNewLineage({ onRefusedWhileActive: function (note) { showReconcileBanner(note); } }); });
  var restoreBtn = document.createElement('button');
  restoreBtn.type = 'button';
  restoreBtn.textContent = 'Restore from browser cache (validated first)';
  restoreBtn.addEventListener('click', function () { restoreFromBrowserCache(); });
  actions.appendChild(reloadBtn);
  actions.appendChild(exportBtn);
  actions.appendChild(forkBtn);
  actions.appendChild(restoreBtn);
  el('btn-reconnect-db').classList.add('banner-hidden');
  banner.classList.remove('banner-hidden');
}

function showReconnectBanner(message) {
  showBanner(message);
  el('btn-reconnect-db').classList.remove('banner-hidden');
}

function hideBanner() {
  el('persist-banner').classList.add('banner-hidden');
  el('persist-banner-text').textContent = '';
  el('persist-banner-actions').replaceChildren();
  el('btn-reconnect-db').classList.add('banner-hidden');
}

// close button — re-evaluate current state; never a silent dismiss.
// While state.persistFailed is set, the not-persisting fault stays visible
// IN ADDITION to whatever else is active (✕ does NOT clear it; only a
// successful persist / import / fresh session / verified restore does).
function refreshBannerState() {
  if (state.reconcileRequired) { showReconcileBanner(); return; } // reconcile text carries the not-persisting suffix
  if (state.haltedLineage) {
    showBanner('LINEAGE INTEGRITY ERROR — RUN HALTED — committed lineage untouched — reload and resume recomputes the unlogged generation identically' + persistFailSuffix());
    return;
  }
  if (state.sessionLoggingDisabled) {
    showBanner('GENERATION ' + (state.lastFailureGeneration || '?') + ' NOT LOGGED — this tab\'s committed prefix is intact through generation ' + Math.max(0, (state.lastFailureGeneration || 1) - 1) + ' — LOGGING DISABLED THIS SESSION — TRAINING CONTINUES WITHOUT LOGGING — resume later to recompute generation ' + (state.lastFailureGeneration || '?') + ' identically (the lineage guard requires the same track)' + persistFailSuffix());
    return;
  }
  if (state.persistFailed) {
    // Same message+detail the persistent onDbBanner fail path shows; if a
    // pending DB handle exists, the needs-attention Reconnect affordance
    // stays available next to it.
    showBanner(state.persistFailedText || 'DB NOT PERSISTING — export manually');
    if (state.pendingReconnectHandle) el('btn-reconnect-db').classList.remove('banner-hidden');
    return;
  }
  if (state.subtleAbsent) {
    // K — lowest-priority persistent note: re-rendered whenever no blocking
    // fault is active (never silently dismissed; ✕ re-evaluates to this).
    // The README bullet carrying this exact wording is the docs pass's job.
    showBanner('Browser without SHA-256 (subtle crypto) — file corruption detection reduced to size+edge compare');
    return;
  }
  if (state.cacheAheadNotice) {
    // N-1 — lowest-priority informational note: the browser cache held a copy
    // AHEAD of this write (the persist kept it — data intact). Re-rendered
    // whenever no blocking fault is active so it does not silently vanish on
    // re-render. NOT a fault — never blocks, never masks one, and the
    // banner payload stays non-persistent on the db.js side.
    showBanner(state.cacheAheadNotice);
    return;
  }
  hideBanner();
}

function setLoggingIndicator(disabled) {
  el('logging-indicator').hidden = !disabled;
}

/* ============================================================
   3. Banner-capable DB event hooks + session persistence helpers
   ============================================================ */

function onDbBanner(payload) {
  var type = payload && payload.type;
  var message = String((payload && payload.message) || 'Database event');
  var detail = payload && payload.detail ? String(payload.detail) : '';
  switch (type) {
    case 'reconcile-required':
      state.reconcileRequired = true;
      showReconcileBanner();
      setPersistPill('blocked — reconcile required', 'bad');
      break;
    case 'db-verify-failed':
    case 'db-not-persisting':
      state.persistFailed = true;
      state.persistFailedText = message + (detail ? ' — ' + detail : '');
      setPersistPill('failed — export manually', 'bad');
      showBanner(state.persistFailedText);
      break;
    case 'persist-skipped':
    case 'log-skipped':
      showBanner('LOGGING DISABLED — persist skipped');
      setPersistPill('skipped (logging disabled)', 'warn');
      break;
    case 'subtle-absent':
      // K — persistent note (never auto-hides): browser lacks SHA-256
      // (subtle crypto) — detection reduces to size + 4-KiB edges. NOT a
      // fault: training and persistence keep working. The README bullet
      // carrying this wording is the docs pass's job.
      state.subtleAbsent = true;
      showBanner(message);
      break;
    case 'cache-ahead':
      // N-1 — informational, NON-fault notice: the browser cache held a
      // copy ahead of this write (the LSG slot keeps the NEWER copy; the
      // file path was never blocked; training continues). Park the message
      // on state so refreshBannerState() re-renders it instead of it
      // silently vanishing — the db.js payload stays non-persistent.
      state.cacheAheadNotice = message;
      showBanner(message);
      break;
    case 'cache-diverge':
      // FIT-V3 (M6c, the F-5 rule) — informational, NON-fault notice: the
      // LSG slot and the database diverge at generation N (equal generation,
      // DIFFERENT track hash — a lineage fork/rotation boundary under a kept
      // stale slot, or two divergent lineages). Both keep their bytes — the
      // committed database on disk is untouched; the cache slot stays. Park
      // the message on the same slot as cache-ahead (refreshBannerState()
      // re-renders it); it never masks a blocking fault.
      state.cacheAheadNotice = message;
      showBanner(message);
      break;
    default:
      showBanner(message);
  }
  updateTrainUi();
}

function setPersistPill(text, kind) {
  var pill = el('persist-pill');
  pill.textContent = String(text);
  pill.className = 'pill ' + (kind === 'ok' ? 'pill-ok' : kind === 'warn' ? 'pill-warn' : kind === 'bad' ? 'pill-bad' : 'pill-unknown');
}

// persistSnapshotNow() was DELETED (ADR-0004 / D5): persist() itself writes
// the single browser-cache slot ('car_racer' kv 'car_racer.lastKnownGood')
// on every verified generation, so a second copy here was redundant — and
// unread. The LSG slot is what boot-restore and the reconcile "Restore from
// browser cache" path read, battery-gated; poisoned bytes are never adopted.

function requireDb(opName) {
  if (state.dbAvailable && state.db) return true;
  showBanner('Database required — ' + opName + ' is unavailable. Training/replay/logs are disabled; Manual works.');
  return false;
}

function currentLatestGeneration() {
  if (!state.db) return 0;
  try {
    var gens = state.db.listGenerations();
    return gens.length ? gens[gens.length - 1] : 0;
  } catch (err) {
    return 0;
  }
}

/* ============================================================
   4. Export / import / create / open / fork flows
   ============================================================ */

function exportDb() {
  if (!requireDb('Export')) return;
  var bytes;
  try {
    bytes = state.db.exportBytes();
  } catch (err) {
    showBanner('Export failed — ' + errMsg(err));
    return;
  }
  var latest = currentLatestGeneration();
  downloadBytes(bytes, 'car_racer-gen' + latest + '.sqlite');
}

function applyImportSuccess(generation) {
  // The import path detaches the file handle by design — never re-attach it.
  if (globalThis.DbSession) {
    if (DbSession.clearHandle) DbSession.clearHandle().catch(noop);
  }
  state.dbState = 'cache';
  state.dbName = '(browser cache)';
  state.reconcileRequired = false;
  state.sessionLoggingDisabled = false;
  state.haltedLineage = false;
  state.persistFailed = false;
  state.persistFailedText = '';
  state.pendingReconnectHandle = null;
  el('btn-reconnect-db').classList.add('banner-hidden');
  setLoggingIndicator(false);
  setPersistPill('idle', 'unknown');
  setDbStatusPill();
  showBanner('Imported generation ' + generation + ' — database replaced');
  refreshPanels();
  updateTrainUi();
}

function importBytesWithConfirm(bytes) {
  if (!requireNoLineageWriters('importing the database')) return;
  if (state.persistInFlight > 0) {
    showBanner('A persist is mid-flight — try again in a moment (it resolves by itself)');
    return;
  }
  var latestNow = currentLatestGeneration();
  state.db.importBytes(bytes, {
    confirm: function (generation) {
      return window.confirm(
        'Battery passed — the picked file contains generation ' + generation + '.\n\n' +
        'Importing REPLACES the current database (currently generation ' + latestNow + ').\n' +
        'Export first if you need the current lineage.\n\nReplace the working database?'
      );
    },
  }).then(function (res) {
    if (res && res.applied) {
      applyImportSuccess(res.generation);
    } else if (res && res.reason === 'confirm-declined') {
      showBanner('Import cancelled — database untouched');
    } else {
      showBanner('Import refused — unexpected result');
    }
  }, function (err) {
    showBanner('Import refused — ' + errMsg(err) + ' — working database untouched');
  });
}

function importDb() {
  if (!requireNoLineageWriters('importing the database')) return;
  if (state.persistInFlight > 0) {
    showBanner('A persist is mid-flight — try again in a moment (it resolves by itself)');
    return;
  }
  if (!requireDb('Import')) return;
  if (typeof globalThis.showOpenFilePicker === 'function') {
    // file:// / opaque-origin: the picker exists but throws — banner + keep
    // the working database untouched.
    var importOnError = function (err) {
      if (isFilePickUnavailable(err)) { showFilePickUnavailable(); return; }
      showBanner('Import refused — ' + errMsg(err));
    };
    try {
      globalThis.showOpenFilePicker({ multiple: false, types: SQLITE_TYPES }).then(function (handles) {
        var handle = handles && handles[0];
        if (!handle) return;
        handle.getFile().then(function (file) {
          if (file.size > SQLITE_MAX_BYTES) {
            showBanner('Import refused — file is ' + formatBytes(file.size) + '; the cap is ' + formatBytes(SQLITE_MAX_BYTES));
            return;
          }
          file.arrayBuffer().then(function (buf) {
            importBytesWithConfirm(new Uint8Array(buf));
          }, function (err) {
            showBanner('Import refused — could not read the file: ' + errMsg(err));
          });
        }, function (err) {
          showBanner('Import refused — could not read the file handle: ' + errMsg(err));
        });
      }, importOnError);
    } catch (err) {
      importOnError(err);
    }
  } else {
    // Fallback: hidden <input type=file> (Firefox/Safari)
    el('import-file-input').click();
  }
}

function onImportFileInputChange(event) {
  if (!requireNoLineageWriters('importing the database')) return;
  var input = event.target;
  var file = input.files && input.files[0];
  input.value = ''; // allow re-picking the same file later
  if (!file) return;
  if (file.size > SQLITE_MAX_BYTES) {
    showBanner('Import refused — file is ' + formatBytes(file.size) + '; the cap is ' + formatBytes(SQLITE_MAX_BYTES));
    return;
  }
  if (!requireDb('Import')) return;
  file.arrayBuffer().then(function (buf) {
    importBytesWithConfirm(new Uint8Array(buf));
  }, function (err) {
    showBanner('Import refused — could not read the file: ' + errMsg(err));
  });
}

function createDb() {
  if (!requireNoLineageWriters('creating the database')) return;
  if (!requireDb('Create')) return; // offline — a new file would be pointless
  var latest = currentLatestGeneration();
  if (latest > 0) {
    var ok = window.confirm(
      'Create a brand-new database? The current lineage (' + latest + ' generations) will NOT be copied into the new database.\n\nExport first if you need it. Continue?'
    );
    if (!ok) return;
  }
  if (typeof globalThis.showSaveFilePicker === 'function') {
    // file:// / opaque-origin: the picker exists but throws — stay in the
    // working in-memory session (it persists to the browser cache via the
    // normal flow); do NOT dead-end and do NOT replace the working DB.
    var pickOnError = function (err) {
      if (isFilePickUnavailable(err)) { showFilePickUnavailable(); return; }
      showBanner('Create database refused — ' + errMsg(err));
    };
    try {
      globalThis.showSaveFilePicker({ suggestedName: 'car_racer.sqlite', types: SQLITE_TYPES }).then(function (handle) {
        // SEC F-1 — create-picker TOCTOU: re-stat the picked handle and
        // require it to be EMPTY before ANY createWritable/write/saveHandle.
        // A file already existing at the path is refused outright — we never
        // overwrite bytes we did not write. Between this getFile() and the
        // exclusive createWritable there remains one event-loop turn; the
        // exclusive create shrinks the residual window further, but the
        // guarantee is: we never open for writing a path whose emptiness we
        // have not just verified. (A NotFoundError means the picker did not
        // pre-create the file — nothing to overwrite; proceed. Any other
        // getFile failure is fail-closed: refuse.)
        handle.getFile().then(function (file) {
          if (file.size !== 0) {
            showBanner('A file already exists at that path — Refusing to overwrite');
            return;
          }
          createDbAtHandle(handle);
        }, function (err) {
          if (err && err.name === 'NotFoundError') {
            createDbAtHandle(handle); // the save picker did not pre-create the file — nothing to overwrite
            return;
          }
          showBanner('Could not verify the picked file is empty — Refusing to overwrite');
        });
      }, pickOnError);
    } catch (err) {
      pickOnError(err);
    }
    return;
  }
  // No File System Access API — fresh database living in this browser's cache.
  createFreshCacheDb();
}

function createDbAtHandle(handle) {
  if (!requireNoLineageWriters('creating the database')) return;
  if (state.persistInFlight > 0) {
    showBanner('A persist is mid-flight — try again in a moment (it resolves by itself)');
    return;
  }
  var fresh;
  try {
    fresh = new Db();
  } catch (err) {
    showBanner('Create database failed — ' + errMsg(err));
    return;
  }
  fresh.init().then(function () {
    var bytes = fresh.exportBytes();
    var writable = null;
    return handle.createWritable({ mode: 'exclusive' }).then(function (w) {
      writable = w;
      return w.write(bytes).then(function () {
        return w.close();
      }, function (err) {
        try { w.abort(); } catch (abortErr) { noop(); }
        throw err;
      });
    }).then(function () {
      // Adopt the just-written file through the normal battery path —
      // baseline + freshness fingerprint are recorded from OUR bytes.
      return state.db.openExisting({ handle: handle });
    });
  }).then(function () {
    if (globalThis.DbSession) {
      if (DbSession.saveHandle) DbSession.saveHandle(handle).catch(noop);
    }
    state.dbState = 'fs';
    state.dbName = handle.name || 'car_racer.sqlite';
    state.reconcileRequired = false;
    state.sessionLoggingDisabled = false;
    state.haltedLineage = false;
    state.persistFailed = false;
    state.persistFailedText = '';
    setLoggingIndicator(false);
    setPersistPill('idle', 'unknown');
    setDbStatusPill();
    showBanner('Created new database ' + (handle.name || 'car_racer.sqlite') + ' — fresh lineage starts at generation 1');
    refreshPanels();
    updateTrainUi();
  }, function (err) {
    showBanner('Create database failed — ' + errMsg(err) + ' — your existing database was NOT touched');
  });
}

function createFreshCacheDb() {
  if (!requireNoLineageWriters('creating the database')) return;
  if (state.persistInFlight > 0) {
    showBanner('A persist is mid-flight — try again in a moment (it resolves by itself)');
    return;
  }
  var fresh;
  try {
    fresh = new Db();
  } catch (err) {
    showBanner('Create database failed — ' + errMsg(err));
    return;
  }
  fresh.init().then(function () {
    var old = state.db;
    state.db = fresh;
    if (old && old.db && typeof old.db.close === 'function') {
      try { old.db.close(); } catch (err) { noop(); }
    }
    fresh.setPersistenceHooks({ onBanner: onDbBanner, persistSkipped: function () { return state.sessionLoggingDisabled; } });
    state.dbState = 'cache';
    state.dbName = '(browser cache)';
    state.reconcileRequired = false;
    state.sessionLoggingDisabled = false;
    state.haltedLineage = false;
    state.persistFailed = false;
    state.persistFailedText = '';
    setLoggingIndicator(false);
    setPersistPill('idle', 'unknown');
    if (globalThis.DbSession) {
      if (DbSession.clearHandle) DbSession.clearHandle().catch(noop);
    }
    setDbStatusPill();
    showBanner('Created a fresh database in this browser\'s cache — Export .sqlite to keep it. Fresh lineage starts at generation 1.');
    refreshPanels();
    updateTrainUi();
  }, function (err) {
    showBanner('Create database failed — ' + errMsg(err) + ' — your existing database was NOT touched');
  });
}

function openDbPicker() {
  if (!requireNoLineageWriters('opening the database')) return;
  if (state.persistInFlight > 0) {
    showBanner('A persist is mid-flight — try again in a moment (it resolves by itself)');
    return;
  }
  if (!requireDb('Open')) return;
  if (typeof globalThis.showOpenFilePicker !== 'function') {
    showBanner('This browser cannot open file handles (File System Access API missing) — use Import .sqlite instead');
    return;
  }
  // file:// / opaque-origin: the picker exists but throws — banner + keep
  // the working database untouched.
  var openOnError = function (err) {
    if (isFilePickUnavailable(err)) { showFilePickUnavailable(); return; }
    showBanner('Open refused — ' + errMsg(err));
  };
  try {
    globalThis.showOpenFilePicker({ multiple: false, types: SQLITE_TYPES }).then(function (handles) {
      var handle = handles && handles[0];
      if (!handle) return;
      adoptOpenedHandle(handle, 'Opened');
    }, openOnError);
  } catch (err) {
    openOnError(err);
  }
}

function adoptOpenedHandle(handle, verb, opts) {
  if (!requireNoLineageWriters('replacing the database')) return;
  if (state.persistInFlight > 0) {
    showBanner('A persist is mid-flight — try again in a moment (it resolves by itself)');
    return;
  }
  state.db.openExisting({ handle: handle }).then(function (summary) {
    state.dbState = 'fs';
    state.dbName = handle.name || '(unnamed)';
    state.reconcileRequired = false;
    state.sessionLoggingDisabled = false;
    state.haltedLineage = false;
    state.persistFailed = false;
    state.persistFailedText = '';
    state.pendingReconnectHandle = null;
    el('btn-reconnect-db').classList.add('banner-hidden');
    setLoggingIndicator(false);
    setPersistPill('idle', 'unknown');
    if (globalThis.DbSession) {
      if (DbSession.saveHandle) DbSession.saveHandle(handle).catch(noop);
    }
    setDbStatusPill();
    showBanner(verb + ' ' + state.dbName + ' — generation ' + summary.generation + ' · track pin ' + shortHash(summary.trackHash));
    refreshPanels();
    updateTrainUi();
  }, function (err) {
    var refused = verb + ' refused — ' + errMsg(err) + ' — working database untouched';
    showBanner(refused);
    if (opts && typeof opts.onRefused === 'function') opts.onRefused(refused); // e.g. the reconcile flow re-shows its actions with the reason folded in
  });
}

function reloadFromDisk() {
  // Reconcile → "Reload from disk (loses this tab's gens)"
  if (!requireNoLineageWriters('reloading the database from disk')) return;
  if (state.persistInFlight > 0) {
    showReconcileBanner('A persist is mid-flight — try again in a moment (it resolves by itself)');
    return;
  }
  if (!state.db) return;
  if (typeof globalThis.showOpenFilePicker !== 'function') {
    showReconcileBanner(); // stay blocked; only Import can escape here
    showBanner('This browser cannot re-attach a file — use Export my copy to keep your work, then Import a good .sqlite');
    return;
  }
  var reloadOnError = function (err, note) {
    if (isFilePickUnavailable(err)) {
      // file:// / opaque-origin — banner + keep the working database AND the
      // reconcile banner (state unchanged; only the user's export/import can move it).
      showFilePickUnavailable();
      showReconcileBanner(note);
      return;
    }
    var refusal = 'Reload refused — ' + errMsg(err) + ' — working database untouched (zero adoption)';
    showBanner(note || refusal);
    showReconcileBanner(note || refusal); // keep the three actions visible; the refusal reason is folded into the text
  };
  try {
    globalThis.showOpenFilePicker({ multiple: false, types: SQLITE_TYPES }).then(function (handles) {
      var handle = handles && handles[0];
      if (!handle) { showReconcileBanner(); return; }
      adoptOpenedHandle(handle, 'Reloaded — this tab\'s uncommitted generations were discarded; the file at', { onRefused: function (note) { showReconcileBanner(note); } });
    }, reloadOnError);
  } catch (err) {
    reloadOnError(err);
  }
}

// restoreFromBrowserCache() — the reconcile banner's FOURTH action (M9):
// "Restore from browser cache (validated first)". Battery-gated exactly like
// every other adoption (Db.restoreFromLastKnownGood builds its own scratch
// instance and NEVER touches the working DB on failure). On success the
// working DB IS SWAPPED for the restored instance (state.db is reassignable —
// createFreshCacheDb does the same) and state is re-pinned (pills, logs,
// panels, hooks); on failure the working DB is left untouched and the
// reconcile banner with its actions is re-rendered — never overlaid away.
async function restoreFromBrowserCache() {
  // F-16 — one shared mid-run gate at every adoption door; fires BEFORE any work.
  if (!requireNoLineageWriters('restoring the browser-cached database')) {
    var restoreBusy = 'Stop the active run (Train → Stop), or wiping while a spectate generation is armed, before restoring the database — complete it or press the spectate toggle (Stop spectate) to end it, before continuing.';
    showBanner(restoreBusy);
    showReconcileBanner(restoreBusy);
    return;
  }
  // F3 — a swap mid-export would trip the freshness baseline bookkeeping.
  if (state.persistInFlight > 0) {
    showReconcileBanner('A persist is mid-flight — try again in a moment (it resolves by itself)');
    return;
  }
  showBanner('Restoring from browser cache…');
  var res = null;
  try {
    res = await Db.restoreFromLastKnownGood();
  } catch (err) {
    res = { ok: false, reason: 'restore-threw', detail: errMsg(err) };
  }
  if (!res || !res.ok) {
    if (res && res.reason === 'validation') {
      // The poison can never be offered again on every reconcile click —
      // clear the LSG slot (mirrors the boot refusal path in db-session.js).
      try { globalThis.Db.clearLastKnownGood(); } catch (e) { /* best-effort */ }
    }
    var restoreRefusal = 'Restore refused: ' + ((res && (res.detail || res.reason)) || 'unknown error') + ' — working database untouched';
    showBanner(restoreRefusal);
    showReconcileBanner(restoreRefusal); // never overlaid away — reconcile text carries the refusal reason too
    return;
  }
  var old = state.db;
  state.db = res.db;
  if (old && old.db && typeof old.db.close === 'function') {
    try { old.db.close(); } catch (err) { noop(); }
  }
  res.db.setPersistenceHooks({ onBanner: onDbBanner, persistSkipped: function () { return state.sessionLoggingDisabled; } });
  state.dbState = 'cache';
  state.dbName = '(browser cache)';
  state.reconcileRequired = false;
  state.sessionLoggingDisabled = false;
  state.haltedLineage = false;
  state.lastFailureGeneration = 0;
  state.persistFailed = false;
  state.persistFailedText = '';
  state.pendingReconnectHandle = null;
  el('btn-reconnect-db').classList.add('banner-hidden');
  setLoggingIndicator(false);
  setPersistPill('idle', 'unknown');
  if (globalThis.DbSession) {
    // Same detach semantics as the import path — a restored cache copy must
    // not silently re-attach the stale file handle.
    if (DbSession.clearHandle) DbSession.clearHandle().catch(noop);
  }
  setDbStatusPill();
  refreshPanels();
  updateTrainUi();
  refreshBannerState();
  showBanner('Restored from verified browser cache (' + (res.summary.slot || 'lastKnownGood') + ' — battery-verified) — generation ' + res.summary.generation + ' loaded');
}

function forkNewLineage(opts) {
  // Confirmed-fork wipe — shared by the banner and the Logs tab.
  // F3 — a mid-flight persist would race the snapshot reset/first-persist.
  if (state.persistInFlight > 0) {
    var midFlight = 'A persist is mid-flight — try again in a moment (it resolves by itself)';
    showBanner(midFlight);
    // The reconcile banner's Fork button passes onRefusedWhileActive — fold the reason in.
    if (opts && typeof opts.onRefusedWhileActive === 'function') opts.onRefusedWhileActive(midFlight);
    return;
  }
  if (!state.db) { showBanner('Database required'); return; }
  // F-16 — one shared mid-run gate at every adoption/swap door. The armed-spectate
  // escape copy + the reconcile banner folding (M4-FIX-3) are kept on top of the
  // helper's uniform refusal.
  if (!requireNoLineageWriters('wiping the lineage')) {
    var forkRefusal = 'Stop the active run (Train → Stop), or wiping while a spectate generation is armed, before wiping or forking the lineage — complete it or press the spectate toggle (Stop spectate) to end it, before continuing.';
    showBanner(forkRefusal);
    if (opts && typeof opts.onRefusedWhileActive === 'function') opts.onRefusedWhileActive(forkRefusal);
    return;
  }
  var ok = window.confirm(
    'NEW LINEAGE\n\nThis wipes EVERYTHING — all generations, per-driver results and best weights are deleted in one transaction. The committed lineage is destroyed.\n\nIt starts at rotation 0 (difficulty 0) and is replaced by the generated schedule from rotation 1 (generation 51).\n\nExport first if you need it. Continue?'
  );
  if (!ok) return;
  var seed = cryptoSeed();
  if (seed === null) { showBanner('crypto.getRandomValues is unavailable — cannot seed a new lineage'); return; }
  try {
    state.db.forkLineage(state.activeTrackHash, seed, state.activeTrack.toJSON());
  } catch (err) {
    var refused = 'Fork refused — ' + errMsg(err);
    showBanner(refused);
    // The reconcile banner's Fork button passes onRefusedWhileActive — reuse it
    // so the refusal text is folded INTO the reconcile banner (the actions stay).
    if (opts && typeof opts.onRefusedWhileActive === 'function') opts.onRefusedWhileActive(refused);
    return;
  }
  // F-3/L — a confirmed full-lineage wipe makes any cached LSG bytes
  // (pre-wipe lineage) stale: delete the LSG slot so the wipe is durable
  // across boots. Best effort — a failed clear never blocks the fork.
  try { globalThis.Db.clearLastKnownGood(); } catch (e) { /* best effort */ }
  // Fork = fresh namespace: the lineageSnapshot is DELETED AND RECREATED with
  // the fresh tag namespace BEFORE the new lineage's first persist (M-NEW8).
  // Never auto-overwrite the never-auto slot any other way.
  var snap = Promise.resolve(typeof state.db.resetLineageSnapshot === 'function' ? state.db.resetLineageSnapshot() : null);
  snap.then(forkSnapshotOutcome, forkSnapshotOutcomeRejected);
}

function forkSnapshotOutcome(res) {
  // The fork destroyed the faulty lineage — the halt notice and the stale
  // failure generation are obsolete the moment the wipe commits (SEC M2).
  state.reconcileRequired = false;
  state.sessionLoggingDisabled = false;
  state.haltedLineage = false;
  state.lastFailureGeneration = 0;
  setLoggingIndicator(false);
  setPersistPill('idle', 'unknown');
  refreshPanels();
  updateTrainUi();
  refreshBannerState(); // clear any stale halt/reconcile banner BEFORE rendering the outcome
  if (res && res.ok === false) {
    showBanner('Forked — but the browser-cache snapshot reset FAILED (' + (res.reason || 'idb') + ') — Export .sqlite now');
    return;
  }
  showBanner('Forked — new lineage starts at generation 1');
}

function forkSnapshotOutcomeRejected() {
  // resetLineageSnapshot REJECTED (not a typed {ok:false}) — same failure
  // posture as a typed failure: the fork committed, the safety net did NOT.
  forkSnapshotOutcome({ ok: false, reason: 'idb' });
}

function reconnectDb() {
  if (!requireNoLineageWriters('reconnecting the database')) return;
  var handle = state.pendingReconnectHandle;
  if (!handle || !state.db) return;
  // requestPermission may ONLY run inside a user gesture — this is one (click).
  handle.requestPermission({ mode: 'readwrite' }).then(function (permission) {
    if (permission !== 'granted') {
      showBanner('Reconnect refused — permission was not granted. The database stays detached; Import .sqlite works everywhere.');
      return;
    }
    adoptOpenedHandle(handle, 'Database connected from last session:');
  }, function (err) {
    showBanner('Reconnect failed — ' + errMsg(err));
  });
}

/* ============================================================
   DB status pill (moved from js/main.js section 5)
   ============================================================ */

function setDbStatusPill() {
  var pill = el('db-status-pill');
  if (!state.dbAvailable || !state.db) {
    pill.textContent = 'DB: unavailable';
    pill.className = 'pill pill-bad';
    return;
  }
  var label, kind;
  if (state.dbState === 'fs') { label = 'DB: file' + (state.dbName ? ' · ' + state.dbName : ''); kind = 'pill-ok'; }
  else if (state.dbState === 'cache') { label = 'DB: browser cache'; kind = 'pill-warn'; }
  else { label = 'DB: in-memory'; kind = 'pill-warn'; }
  pill.textContent = label;
  pill.className = 'pill ' + kind;
}

