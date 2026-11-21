// js/main.js — Browser shell for the 0003_car_racer evolutionary NN car racer.
//
// Script order (LOAD-BEARING, mirrored in index.html):
//   vendor/sql-wasm-binary.js → vendor/sql-wasm.js → js/driver-sim.js → js/track.js → js/rotation.js → js/car.js → js/sensors.js
//   → js/nn.js → js/evolution.js → js/db.js → js/trainer.js → js/replay.js
//   → js/main-db.js → js/spectate.js → js/db-session.js → js/editor.js (OPTIONAL) → js/main.js
//
// js/editor.js is an OPTIONAL module: boot() calls Editor.init() when — and
// only when — window.Editor exists. When it is absent/failed, the Editor tab
// stays hidden and the Logs "Fork to new track" button is disabled.
//
// BOOT CONTRACT: before anything else, the required globals are checked.
// If any is missing, #boot-error is filled with the exact list and EVERY
// control is disabled — the game stops here, cleanly, with readable output.
// window.Replay is OPTIONAL: the Replay tab shows "replay module not loaded"
// and everything else keeps working.
//
// DOM DISCIPLINE: this file NEVER uses innerHTML (grep gate). All dynamic DOM
// is built with textContent / createTextNode / createElement only. `ctx.fillText`
// (canvas text) is fine — nothing on canvas is HTML parsing.
//
// BANNER DISCIPLINE: #persist-banner is the single persistent banner. No banner
// here ever auto-hides. The close button only re-evaluates current state via
// refreshBannerState(): if a blocking condition persists, the banner is
// re-rendered; otherwise it hides.
//
// TRAINING MODES (Train tab)
//   Headless — Trainer + Web Worker (js/train-worker.js) with a MessageChannel
//   fallback, ONE Db transaction per generation, full LOG-FAILURE degrade
//   policy. Runs off the main thread.
//   Spectate — main-thread live eval of the same lineage spec: a local
//   Evolution instance, 20 cars rendered live with distinct hues. Logging
//   goes through the SAME db.logGeneration() transaction with the same
//   payload shape as the headless path; a [Next generation] button spawns
//   generation N+1 (elitism + Gaussian mutation) after the summary.
// Both modes honor the lineage guard (fresh: ensureLineage pin; resume:
// resumeState() + track_hash match) and the DB-required rule: if the database
// is unavailable, ALL training is disabled and Manual keeps working.

'use strict';

/* ============================================================
   0. Boot contract check — first thing this file does
   ============================================================ */

var REQUIRED_GLOBALS = [
  'Track',
  'Rotation',
  'DEFAULT_CONTROL_POINTS',
  'Car',
  'SensorFan',
  'NeuralNetwork',
  'mulberry32',
  'Evolution',
  'Db',
  'DbOfflineError',
  'Trainer',
  'DbSession',
  // js/main-db.js + js/spectate.js are REQUIRED files (NOT the optional
  // Replay/Editor pair) — their absence must fail this check, loudly, like a
  // missing main.js: boot() calls refreshBannerState synchronously (js/
  // main-db.js, and onDbBanner/setDbStatusPill during the async boot DB
  // init), and startSpectate is referenced by the wired Spectate handlers.
  'refreshBannerState',
  'startSpectate',
];

function missingGlobals() {
  var missing = [];
  for (var i = 0; i < REQUIRED_GLOBALS.length; i++) {
    var name = REQUIRED_GLOBALS[i];
    if (typeof globalThis[name] === 'undefined' || globalThis[name] === null) {
      missing.push(name);
    }
  }
  return missing;
}

function renderBootError(missing) {
  var panel = document.getElementById('boot-error');
  var lines = [
    'Car Racer failed to start.',
    '',
    'Missing globals: ' + (missing.length ? missing.join(', ') : '(unknown)'),
    '',
    'Required script order (see index.html):',
    '  vendor/sql-wasm-binary.js → vendor/sql-wasm.js → js/driver-sim.js → js/track.js → js/rotation.js → js/car.js → js/sensors.js → js/nn.js',
    '  → js/evolution.js → js/db.js → js/trainer.js → js/replay.js → js/main-db.js → js/spectate.js → js/db-session.js → js/editor.js (optional) → js/main.js',
    '',
    'Serve the folder over localhost (python3 -m http.server) or open it over file://.',
    'Note: if ONLY the database fails to load, the game starts with the database disabled and Manual works.',
  ];
  if (panel) {
    panel.textContent = lines.join('\n');
    panel.hidden = false;
  }
  var controls = document.querySelectorAll('button, input, select');
  for (var i = 0; i < controls.length; i++) {
    controls[i].disabled = true;
  }
  var boot = document.getElementById('status-line');
  if (boot) boot.textContent = 'boot failed — missing: ' + missing.join(', ');
}

var BOOT_CONTRACT_MISSING = missingGlobals();
if (BOOT_CONTRACT_MISSING.length > 0) {
  renderBootError(BOOT_CONTRACT_MISSING);
}

/* ============================================================
   1. Constants, tiny helpers, shared state
   ============================================================ */

// The load order is contract-pinned (index.html + the boot contract) — read
// the globals directly; no silent fallback twins. A missing global means a
// broken script order, which the boot-contract check reports loudly.
var SIM_DT = globalThis.SIM_DT;
var MAX_GENERATIONS_DEFAULT = globalThis.MAX_GENERATIONS;
var MUTATION_RATE = globalThis.MUTATION_RATE;
var MUTATION_SCALE = globalThis.MUTATION_SCALE;
var CANVAS_W = 1100, CANVAS_H = 650;

function el(id) { return document.getElementById(id); }
function noop() {}

// FIT-V6 — typed wall-model refusal → persistent, actionable banner. The
// message itself carries the export + fresh-lineage instruction (db.js's
// WallModelChangedError); the banner adds a stable marker and names the
// concrete UI actions. errMsg is shared by every catch site (including
// main-db.js's import handler), so mapping here covers all adoption doors.
// The refusal is BYTE-PRESERVING: err.refusedBytes can be downloaded verbatim
// via the dedicated banner action (showWallModelRefusalBanner) where wired.
var WALL_MODEL_BANNER_MARK = 'WALL MODEL CHANGED';
function isWallModelChanged(err) { return !!(err && err.name === 'WallModelChangedError'); }
function wallModelChangedBanner(err) {
  return WALL_MODEL_BANNER_MARK + ' — ' + String((err && err.message) || err) +
    ' — The original was left untouched. Use Export .sqlite to download the refused bytes verbatim, then Create / Fork to start a fresh lineage.';
}
function errMsg(err) {
  if (isWallModelChanged(err)) return wallModelChangedBanner(err);
  return String((err && err.message) || err);
}

// The one refusal banner that offers a "download refused bytes verbatim"
// action. Built here (rather than via showBanner, which clears actions) so the
// button survives until the user resolves the refusal. Never mutates storage.
function showWallModelRefusalBanner(refusedBytes, baseText) {
  var banner = el('persist-banner');
  el('persist-banner-text').textContent = String(baseText || (WALL_MODEL_BANNER_MARK + ' — the original was left untouched.'));
  var actions = el('persist-banner-actions');
  actions.replaceChildren();
  if (refusedBytes && typeof refusedBytes.length === 'number' && refusedBytes.length > 0) {
    var dl = document.createElement('button');
    dl.type = 'button';
    dl.textContent = 'Download refused bytes verbatim';
    dl.addEventListener('click', function () { downloadBytes(refusedBytes, 'refused-lineage.sqlite'); });
    actions.appendChild(dl);
  }
  el('btn-reconnect-db').classList.add('banner-hidden');
  banner.classList.remove('banner-hidden');
}
function shortHash(h) { var s = h === null || h === undefined ? '' : String(h); return s.length > 12 ? s.slice(0, 12) + '…' : s; }
function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return '? bytes';
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}
function formatPx(v) { return Number.isFinite(v) ? v.toFixed(1) + ' px' : '— px'; }
function formatSpeed(v) { return Number.isFinite(v) ? v.toFixed(0) + ' px/s' : '— px/s'; }
function formatSeconds(v) { return Number.isFinite(v) ? v.toFixed(2) + ' s' : '— s'; }
function clampInt(value, lo, hi, fallback) {
  var n = Number(value);
  if (!Number.isFinite(n)) n = fallback;
  n = Math.floor(n);
  return Math.max(lo, Math.min(hi, n));
}
function cryptoSeed() {
  try {
    var buf = new Uint32Array(1);
    if (!globalThis.crypto || typeof globalThis.crypto.getRandomValues !== 'function') return null;
    globalThis.crypto.getRandomValues(buf);
    return buf[0] >>> 0;
  } catch (err) {
    return null;
  }
}
// The ONE lineage-integrity classifier lives in js/db.js (globalThis.isLineageIntegrityFault);
// this is a thin guarded reference — outside a DbSession boot, no fault is assumed.
function isLineageIntegrity(err) {
  return typeof globalThis.isLineageIntegrityFault === 'function' && globalThis.isLineageIntegrityFault(err);
}
function downloadBytes(bytes, filename) {
  var blob = new Blob([bytes], { type: 'application/octet-stream' });
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
}

var state = {
  activeTab: 'manual',
  activeTrack: null,
  activeTrackHash: '',
  activeTrackName: 'default circuit',
  editorReady: false,            // true when js/editor.js loaded + Editor.init succeeded (boot)

  // Database
  db: null,
  dbAvailable: false,
  dbState: 'memory',       // 'fs' | 'cache' | 'memory' — only meaningful when dbAvailable
  dbName: '',
  sessionLoggingDisabled: false,
  haltedLineage: false,
  reconcileRequired: false,
  pendingReconnectHandle: null,
  persistInFlight: 0,
  persistFailed: false,
  persistFailedText: '',

  // Runs
  headlessRunning: false,
  trainer: null,
  spectate: null,          // { evo, cars:[{ sim, index }], running, waitingForNext, generation }
  replayRun: null,         // any object with stop() from Replay.createRun, or null
  replayRunning: false,

  // Logs pagination (rows currently rendered; the OFFSET for the next page)
  logPage: 0,

  // HUD / loop
  keys: Object.create(null),
  manualCar: null,
  // Latest committed generation — cached at refreshPanels() / lineage-event
  // time (reviewer #17-19): updateStatusLine() must NOT poll the table on a
  // 500 ms timer for a number that changes only on commit. Default 0.
  latestGen: 0,
  // Set at boot when crypto.subtle is unavailable (K) — drives the persistent
  // 'subtle-absent' banner in refreshBannerState(). Never auto-hides.
  subtleAbsent: false,
  // Set when a persist() is refused the browser-cache slot because the cache
  // is AHEAD on this lineage (F-3) — drives the informational 'cache-ahead'
  // banner in refreshBannerState(). NOT a fault — the cache keeps the newer
  // copy; training continues (N-1).
  cacheAheadNotice: null,

  // FIT-V3 — the rotation session (headless/spectate). trainPinTrack is the
  // track the CURRENT session pinned at lineage creation (fresh: the active
  // editor/manual track; resume: the stored tip-generation tracks row via
  // Db.adoptPinnedTrack — never the ambient editor/Manual track). The
  // trainTrack* triple caches the built curriculum track by rotation index —
  // rebuilt at most once per 50-generation block. spectateTrack is the
  // spectate render target (the block's track — cars and track always agree).
  trainPinTrack: null,
  rotationSeed: 0,
  generatorVersion: null, // FIT-V4 — canonical integer from the strict lineage pin; REQUIRED for the session resolver
  trainTrackBlock: -1,
  trainTrackObj: null,
  trainTrackHash: '',
  spectateTrack: null,

  // RESPONSIVE — last successful render inputs, replayed after a resize / tab
  // reveal so a resized (and therefore cleared) backing store is repainted.
  lastTrainStatus: null,     // { text, track }
  lastReplayFrame: null,     // last onFrame payload from js/replay.js
  currentTrackLabel: 'default circuit',
  canvasFitDpr: 0,           // last devicePixelRatio observed by canvasContext

  ticks: 0,
  fps: 60,
  manualAcc: 0,
  spectateAcc: 0,
  lastFrameTs: 0,
  lastFailureGeneration: 0,
};

/* ============================================================
   5. Panels refresh (the DB status pill lives in js/main-db.js)
   ============================================================ */

function updateTrainUi() {
  var canTrain = state.dbAvailable && !!state.db && !state.haltedLineage && !(state.spectate && state.spectate.waitingForNext); // J — an armed spectate generation (in-place lineage step) blocks new runs too
  el('btn-start-training').disabled = !canTrain || state.headlessRunning || (state.spectate && state.spectate.running);
  el('btn-stop-training').disabled = !state.headlessRunning;
  el('btn-spectate').disabled = (!canTrain || state.headlessRunning) && !(state.spectate && state.spectate.waitingForNext);
  el('btn-spectate').textContent = (state.spectate && (state.spectate.running || state.spectate.waitingForNext)) ? 'Stop spectate' : 'Spectate (20 cars live)';
  el('btn-next-generation').hidden = !(state.spectate && state.spectate.waitingForNext);
  var resumeSelected = el('lineage-resume').checked;
  el('resume-generation').disabled = !resumeSelected || state.headlessRunning;
  el('max-generations').disabled = state.headlessRunning;
  var blockedNote = el('train-blocked-note');
  if (!state.dbAvailable) {
    blockedNote.textContent = 'Database required — training is disabled; Manual still works.';
    blockedNote.hidden = false;
  } else if (state.haltedLineage) {
    blockedNote.textContent = 'Training halted (lineage-integrity) — reload and resume recomputes the unlogged generation identically.';
    blockedNote.hidden = false;
  } else {
    blockedNote.hidden = true;
  }
}

function updateResumeInputBounds() {
  var maxGen = currentLatestGeneration();
  state.latestGen = Math.max(state.latestGen || 0, maxGen); // never clobber a just-committed generation
  var input = el('resume-generation');
  input.max = String(Math.max(1, maxGen));
  if (Number(input.value) > maxGen) input.value = String(Math.max(1, maxGen));
  if (!input.value || Number(input.value) < 1) input.value = String(Math.max(1, maxGen || 1));
}

function setReplayDisabledNote(message) {
  var note = el('replay-disabled-note');
  if (message) {
    note.textContent = message;
    note.hidden = false;
  } else {
    note.hidden = true;
  }
}

function refreshReplayBounds() {
  var R = replayApi();
  var maxGen = currentLatestGeneration();
  state.latestGen = Math.max(state.latestGen || 0, maxGen); // commit-path cache — never regress mid-run
  var input = el('replay-generation');
  input.max = String(Math.max(1, maxGen));
  input.value = String(Math.max(1, Math.min(Math.max(1, Number(input.value) || 1), maxGen || 1)));
  var disabled = !state.dbAvailable || !state.db || maxGen < 1 || !R;
  input.disabled = disabled;
  el('replay-use-active').disabled = disabled;
  el('btn-run-replay').disabled = disabled || state.replayRunning;
  el('btn-stop-replay').disabled = !state.replayRunning;
  if (disabled) {
    if (!state.dbAvailable || !state.db) setReplayDisabledNote('Replay requires the database — unavailable (see banner).');
    else if (!R) setReplayDisabledNote('Replay module not loaded (js/replay.js missing) — the rest of the game works.');
    else setReplayDisabledNote('No generations committed yet — train first, then replay.');
  } else {
    setReplayDisabledNote(null);
  }
  var replayEmpty = el('replay-empty');
  if (replayEmpty) {
    replayEmpty.hidden = !(state.dbAvailable && state.db && Math.max(maxGen, state.latestGen || 0) < 1);
  }
  updateCanvasLabels(); // keep accessible names truthful as generations arrive
}

function refreshPanels() {
  state.latestGen = currentLatestGeneration(); // single cache write — readers use state.latestGen (reviewer #17-19)
  setDbStatusPill();
  updateResumeInputBounds();
  refreshReplayBounds();
  refreshLogs();
  updateTrainUi();
  updateStorageLine();
}

function updateStorageLine() {
  var line = el('storage-line');
  if (!globalThis.navigator || !globalThis.navigator.storage || typeof globalThis.navigator.storage.estimate !== 'function') {
    line.textContent = 'storage: estimate unavailable in this browser';
    return;
  }
  globalThis.navigator.storage.estimate().then(function (est) {
    var usage = est && Number.isFinite(est.usage) ? formatBytes(est.usage) : '?';
    var quota = est && Number.isFinite(est.quota) ? formatBytes(est.quota) : '?';
    var pct = est && Number.isFinite(est.usage) && Number.isFinite(est.quota) && est.quota > 0
      ? ' (' + ((est.usage / est.quota) * 100).toFixed(2) + '%)'
      : '';
    line.textContent = 'storage: ' + usage + ' of ' + quota + ' used' + pct;
  }, function () {
    line.textContent = 'storage: estimate unavailable';
  });
}

/* ============================================================
   6. Active track (default circuit or localStorage custom)
   ============================================================ */

function loadActiveTrack() {
  var raw = null;
  try {
    raw = globalThis.localStorage ? globalThis.localStorage.getItem('car_racer.track') : null;
  } catch (err) {
    raw = null; // storage blocked (private mode, file:// quirks) — use the default
  }
  if (raw) {
    try {
      var custom = globalThis.Track.fromJSON(raw);
      return {
        track: custom,
        name: 'localStorage custom track (' + custom.controlPoints.length + ' points)',
        note: null,
      };
    } catch (err) {
      return {
        track: new globalThis.Track(globalThis.DEFAULT_CONTROL_POINTS, 70),
        name: 'default circuit',
        note: 'Stored localStorage track was invalid (' + errMsg(err) + ') — using the default circuit.',
      };
    }
  }
  return {
    track: new globalThis.Track(globalThis.DEFAULT_CONTROL_POINTS, 70),
    name: 'default circuit',
    note: null,
  };
}

/* ============================================================
   6b. Track editor — hooks + the "Use as training track" lineage guard
   ============================================================ */

// Editing the track NEVER hot-swaps the active track — the active track comes
// from localStorage at boot, or from the explicit "Use as training track"
// flow below (save to localStorage + switch + lineage re-pin/fork). The M5
// acceptance path.

function currentGenerationCount() {
  if (!state.db) return 0;
  try {
    return state.db.listGenerations().length;
  } catch (err) {
    return 0;
  }
}

// resetTrainStatsUi — Train tab back to zero after a lineage wipe / fresh pin.
function resetTrainStatsUi() {
  el('stat-generation').textContent = '0';
  var rotationElReset = el('stat-rotation');
  if (rotationElReset) rotationElReset.textContent = 'R 0 · d 0';
  el('stat-best').textContent = '— px';
  el('stat-avg').textContent = '— px';
  el('stat-worst').textContent = '— px';
  el('stat-sim-seconds').textContent = '— s';
  el('mini-log').replaceChildren();
  el('spectate-summary').hidden = true;
  el('resume-generation').value = '1';
  // FIT-V3 — a wiped/forked lineage re-pins; drop any stale rotation block
  // cache and pin reference (the next session re-derives from ITS lineage).
  state.trainTrackBlock = -1;
  state.trainTrackObj = null;
  state.trainTrackHash = '';
  state.trainPinTrack = null;
  state.rotationSeed = 0;
  state.spectateTrack = null;
  scheduleTrainProgress(0, el('max-generations').value); // UX — reset progress on new lineage
  updateTrackLabel(0); // UX — back to the pinned/default circuit label
}

// clearLineageFaults — shared outcome reset after ANY fork wipe (the
// confirmed editor fork, or the empty-lineage re-pin fallback below): the
// halt / reconcile / logging-disabled states are obsolete the moment the
// committed lineage is destroyed or the DB is re-pinned. Same ritual as
// forkSnapshotOutcome for the Logs wipe.
function clearLineageFaults() {
  state.reconcileRequired = false;
  state.sessionLoggingDisabled = false;
  state.haltedLineage = false;
  state.lastFailureGeneration = 0;
  setLoggingIndicator(false);
  setPersistPill('fresh lineage', 'ok');
}

// forkSnapshotReset(okNote) — LINEAGE PIN ritual, M-NEW8: the lineageSnapshot
// is DELETED AND RECREATED with the fresh tag namespace BEFORE the new
// lineage's first persist. forkLineage() itself is synchronous and one
// transaction; this runs immediately after it inside the same button
// workflow. The ADOPTION BATTERY is N/A — the bytes are our own DB, no
// foreign bytes were adopted. Keeps the handle (same file, wiped lineage).
function forkSnapshotReset(okNote) {
  var snap = Promise.resolve(typeof state.db.resetLineageSnapshot === 'function' ? state.db.resetLineageSnapshot() : null);
  snap.then(function (res) {
    if (res && res.ok === false) {
      showBanner('Forked — but the browser-cache snapshot reset FAILED (' + (res.reason || 'idb') + ') — Export .sqlite now');
      return;
    }
    if (okNote) showBanner(okNote);
  }, function () {
    showBanner('Forked — but the browser-cache snapshot reset FAILED (idb) — Export .sqlite now');
  });
}

// setEditorStatusLine — mirrored by useEditorTrackAsTrainingTrack on every
// outcome so BOTH the editor's own button and the Logs "Fork to new track"
// button surface their result in the editor's status line. (The editor does
// the same via its own hooks path for its own button — idempotent.)
function setEditorStatusLine(text) {
  var statusEl = el('editor-status');
  if (statusEl) statusEl.textContent = String(text);
}

// useEditorTrackAsTrainingTrack — shared by the editor's primary button
// (hooks.confirmUseTrack, see editorHooks) and the Logs tab "Fork to new
// track" button. Builds the track from the EDITOR state (never from
// localStorage), forks the lineage when generations are committed, re-pins
// silently when none are, saves + activates the track ALWAYS (unless the
// user cancels the fork confirm). Returns { ok, cancelled?, message } — the
// editor renders `message` in #editor-status.
function useEditorTrackAsTrainingTrack() {
  // F-16 — one shared mid-run gate at every swap door; the editor keeps its own
  // status-line reporting on top of the helper's uniform refusal.
  if (!requireNoLineageWriters('swapping the training track')) {
    var refusal = 'Stop the active run (Train → Stop), then swap the training track';
    showBanner(refusal);
    setEditorStatusLine(refusal);
    return { ok: false, message: refusal };
  }
  var track = null;
  try {
    if (globalThis.window && window.Editor && typeof window.Editor.getTrack === 'function') {
      track = window.Editor.getTrack();
    }
  } catch (err) {
    track = null;
  }
  if (!track) {
    var broken = 'Editor track could not be built — nothing changed';
    showBanner(broken);
    setEditorStatusLine(broken);
    return { ok: false, message: broken };
  }
  var trackHash;
  var trackJson;
  try {
    trackHash = track.hash();
    trackJson = track.toJSON();
  } catch (err) {
    var hashFailed = 'Could not hash the editor track — ' + errMsg(err);
    showBanner(hashFailed);
    setEditorStatusLine(hashFailed);
    return { ok: false, message: hashFailed };
  }

  var dbReady = state.dbAvailable && !!state.db;
  var genCount = dbReady ? currentGenerationCount() : 0;
  var wiped = 0;
  var freshPin = false;
  var pinFailed = '';

  if (dbReady && genCount >= 1) {
    // (c) ≥1 committed generations — using this track forks a NEW lineage.
    var ok = window.confirm(
      'Using this track forks a NEW lineage — all ' + genCount + ' committed generations will be deleted.\n\n' +
      'It starts at rotation 0 (difficulty 0) and is replaced by the generated schedule from rotation 1 (generation 51).\n\n' +
      'Continue?'
    );
    if (!ok) {
      var cancelled = 'Fork cancelled — lineage and training track unchanged';
      setEditorStatusLine(cancelled);
      return { ok: false, cancelled: true, message: cancelled };
    }
    var seed = cryptoSeed();
    if (seed === null) {
      var noCrypto = 'crypto.getRandomValues is unavailable — cannot fork';
      showBanner(noCrypto);
      setEditorStatusLine(noCrypto);
      return { ok: false, message: noCrypto };
    }
    try {
      state.db.forkLineage(trackHash, seed, trackJson);
    } catch (err) {
      var refused = 'Fork refused — ' + errMsg(err);
      showBanner(refused);
      setEditorStatusLine(refused);
      return { ok: false, message: refused };
    }
    wiped = genCount;
    // F-3/L — a confirmed full-lineage wipe makes any cached LSG bytes
    // (pre-wipe lineage) stale: delete the LSG slot so the wipe is durable
    // across boots. Best effort — a failed clear never blocks the fork.
    try { globalThis.Db.clearLastKnownGood(); } catch (e) { /* best effort */ }
    clearLineageFaults();
    // The reset must land BEFORE the new lineage's first persist; it runs now,
    // inside the button workflow, immediately after the wipe transaction.
    forkSnapshotReset('Lineage wiped — new track active, generation 1 next');
  } else if (dbReady) {
    // (d) zero committed generations — silently re-pin; nothing is destroyed.
    var freshSeed = cryptoSeed();
    if (freshSeed === null) {
      pinFailed = 'crypto.getRandomValues is unavailable — cannot re-pin';
    } else {
      try {
        state.db.ensureLineage(trackHash, freshSeed, trackJson);
        freshPin = true;
      } catch (ensureErr) {
        // Empty-but-pinned lineage — the same shape prepareFreshLineage
        // handles: forkLineage on an EMPTY lineage destroys no rows, it just
        // re-pins track_hash + rng_state.
        try {
          state.db.forkLineage(trackHash, freshSeed, trackJson);
          freshPin = true;
          clearLineageFaults();
          forkSnapshotReset(null);
        } catch (forkErr) {
          pinFailed = errMsg(forkErr);
        }
      }
    }
  }

  // (e) save to localStorage 'car_racer.track' ALWAYS (fork decision or not)
  // and make the editor track the ACTIVE track for Manual/Train/Replay/Spectate.
  var storageBlocked = false;
  try {
    if (globalThis.localStorage) {
      globalThis.localStorage.setItem('car_racer.track', trackJson);
    } else {
      storageBlocked = true;
    }
  } catch (err) {
    storageBlocked = true;
  }
  state.activeTrack = track;
  state.activeTrackHash = trackHash;
  state.activeTrackName = 'editor track (' + track.controlPoints.length + ' points)';
  el('hud-track').textContent = state.activeTrackName;
  var trackNoteEl = el('manual-note');
  trackNoteEl.hidden = true;
  trackNoteEl.textContent = '';
  try {
    var pose = track.startPose();
    state.manualCar = new globalThis.Car(track, pose.x, pose.y, pose.heading);
    el('hud-speed').textContent = '0 px/s';
    el('hud-odometer').textContent = '0 px';
  } catch (err) {
    state.manualCar = null; // render path guards on a missing car
  }
  resetTrainStatsUi();
  refreshPanels();      // logs rows re-render empty, resume/replay clamps refresh, storage line
  updateTrainUi();
  refreshBannerState(); // clear any stale halt/reconcile banner BEFORE the outcome banner

  // (f) outcome banner + the editor's status line (#editor-status shows the
  // returned message — hash short form + wipe count, or 'fresh pin').
  var statusMsg;
  if (wiped > 0) {
    statusMsg = 'h' + shortHash(trackHash) + ' · ' + wiped + ' generations wiped';
    setPersistPill('fresh lineage', 'ok');
    showBanner('Lineage wiped — new track active, generation 1 next');
  } else if (freshPin) {
    statusMsg = 'h' + shortHash(trackHash) + ' · fresh pin';
    setPersistPill('fresh lineage', 'ok');
    showBanner('Track saved — new lineage pinned to geometry h' + shortHash(trackHash));
  } else if (!dbReady) {
    statusMsg = 'h' + shortHash(trackHash) + ' · saved — database offline';
    showBanner('Track saved — database unavailable; the lineage pin re-pins on the next fresh run');
  } else {
    statusMsg = 'h' + shortHash(trackHash) + ' · saved — lineage pin FAILED' + (pinFailed ? ' (' + pinFailed + ')' : '');
    showBanner('Track saved — but the lineage pin FAILED' + (pinFailed ? ' — ' + pinFailed : '') + ' — the next fresh run re-pins (empty lineage)');
  }
  if (storageBlocked) statusMsg += ' — browser storage blocked (track not cached for next reload)';
  setEditorStatusLine(statusMsg);
  return { ok: true, message: statusMsg };
}

function editorHooks() {
  return {
    // The editor auto-saves to localStorage on every structural change. The
    // ACTIVE track is deliberately NOT hot-swapped here — swapping mid-session
    // would move the lineage-guard target under a running/armed trainer. The
    // swap happens only via "Use as training track" (or the next page load,
    // which re-reads localStorage with the same Track.fromJSON validation).
    onStatus: function (text) {
      // Advisory mirror into the footer. Transient by design — the 500 ms
      // interval re-renders the canonical status line; the editor's own
      // #editor-status line persists the same text.
      var line = el('status-line');
      if (line) line.textContent = String(text);
    },
    confirmUseTrack: useEditorTrackAsTrainingTrack,
  };
}

/* ============================================================
   7. Canvas painting helpers (grass background)
   ============================================================ */

// Grass background — the ONE deterministic pattern lives in js/track.js
// (Track.drawTrackBackground). This wrapper stays because js/spectate.js and
// the internal render paths call the classic-script global drawGrass(ctx).
function drawGrass(ctx) {
  globalThis.Track.drawTrackBackground(ctx, CANVAS_W, CANVAS_H);
}

// RESPONSIVE — one entry point for EVERY render path (manual, headless
// placeholder, spectate, replay). The world stays 1100x650; this function
// sizes the bitmap to the CSS box * DPR and installs a UNIFORM scale with
// centered letterboxing so all existing drawing code is unchanged.
function canvasContext(canvas) {
  if (!canvas || typeof canvas.getContext !== 'function') return null;
  var ctx = canvas.getContext('2d');
  if (!ctx) return null;
  var dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
  state.canvasFitDpr = dpr;
  var rect = canvas.getBoundingClientRect();
  // A hidden tab has a zero-size box — keep the last bitmap/transform; nothing
  // is drawn until the tab is revealed (activateTab then triggers a redraw).
  if (!rect || rect.width <= 0 || rect.height <= 0) return ctx;
  var bw = Math.max(1, Math.round(rect.width * dpr));
  var bh = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw;   // changing either dimension clears the backing store
    canvas.height = bh;
  }
  var scale = Math.min(bw / CANVAS_W, bh / CANVAS_H);
  var ox = (bw - CANVAS_W * scale) / 2;
  var oy = (bh - CANVAS_H * scale) / 2;
  ctx.setTransform(scale, 0, 0, scale, ox, oy);
  return ctx;
}

// redrawAllCanvases() — re-paint the currently visible canvas(es) after a
// resize, a tab reveal, or a DPR change. Uses the LAST render inputs so a
// static (non-running) view still comes back after its bitmap was cleared.
function redrawAllCanvases() {
  if (state.activeTab === 'manual' && state.manualCar && state.activeTrack) {
    renderManual();
  } else if (state.activeTab === 'train') {
    if (state.spectate && state.spectate.running && typeof drawSpectateFrame === 'function') {
      drawSpectateFrame();
    } else if (state.lastTrainStatus) {
      drawTrainStatus(state.lastTrainStatus.text, state.lastTrainStatus.track);
    }
  } else if (state.activeTab === 'replay') {
    if (state.lastReplayFrame) {
      onReplayFrame(state.lastReplayFrame);
    } else {
      // No run yet — still fit the bitmap and show the (static) track behind
      // the empty-state message so the canvas is never a stale 1100×650 blob.
      var replayCtx = canvasContext(el('replay-canvas'));
      if (replayCtx && state.activeTrack) {
        drawGrass(replayCtx);
        state.activeTrack.draw(replayCtx);
      }
    }
  } else if (state.activeTab === 'editor' && globalThis.Editor && typeof globalThis.Editor.redraw === 'function') {
    globalThis.Editor.redraw();
  }
}

// One rAF-debounced resize listener for the whole app — never a rAF loop.
var resizeRaf = 0;
function onWindowResize() {
  if (resizeRaf) return;
  resizeRaf = requestAnimationFrame(function () {
    resizeRaf = 0;
    redrawAllCanvases();
  });
}

// drawCarGlyph(ctx, x, y, heading) — the ONE rotated car glyph, shared by the
// spectate rendering and the replay canvas: a 30×18 body with a dark
// windshield band. The CALLER owns the colors (spectate: per-car hue / done
// gray + alpha; replay: a single accent) — this helper owns geometry only.
function drawCarGlyph(ctx, x, y, heading) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(heading);
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.55)';
  ctx.lineWidth = 2;
  ctx.rect(-15, -9, 30, 18);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = 'rgba(10, 12, 14, 0.85)';
  ctx.fillRect(2, -5.5, 6, 11);
  ctx.restore();
}

/* ============================================================
   8. Manual mode — keyboard driving
   ============================================================ */

var KEY_MAP = {
  ArrowUp: 'gas',
  ArrowDown: 'brake',
  ArrowLeft: 'left',
  ArrowRight: 'right',
};

function isEditableTarget(target) {
  if (!target || !target.tagName) return false;
  var tag = target.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable === true;
}

function onManualKey(event, isDown) {
  if (isEditableTarget(event.target)) return; // typing in a number field must not drive
  var action = KEY_MAP[event.key];
  if (!action) return;
  if (event.repeat) { event.preventDefault(); return; }
  state.keys[action] = isDown;
  event.preventDefault();
}

function manualInputs() {
  var gas = (state.keys.gas ? 1 : 0) + (state.keys.brake ? -1 : 0);
  var steer = (state.keys.right ? 1 : 0) + (state.keys.left ? -1 : 0);
  return { gas: gas, steer: steer };
}

function resetManualCar() {
  if (!state.manualCar) return;
  state.manualCar.reset();
}

function stepManual() {
  var car = state.manualCar;
  if (!car) return;
  car.update(SIM_DT, manualInputs());
  state.ticks += 1;
}

function renderManual() {
  var canvas = el('manual-canvas');
  var ctx = canvasContext(canvas);
  if (!ctx || !state.activeTrack || !state.manualCar) return;
  drawGrass(ctx);
  state.activeTrack.draw(ctx);
  state.manualCar.draw(ctx);
  el('hud-speed').textContent = formatSpeed(state.manualCar.speed);
  el('hud-odometer').textContent = Math.round(state.manualCar.odometer) + ' px';
}

/* ============================================================
   9. Train — shared lineage resolution (fresh / resume)
   ============================================================ */

function lineageKind() {
  return el('lineage-resume').checked ? 'resume' : 'fresh';
}

// returnedGeneratorVersion(result) — Task B's ensureLineage/forkLineage return
// a result object carrying `generatorVersion` (a bare integer is also accepted
// so the parser stays the single source of truth in either shape).
function returnedGeneratorVersion(result) {
  return (result && typeof result === 'object') ? result.generatorVersion : result;
}

// prepareFreshLineage — returns { kind:'fresh', seed, trackHash } or null (banner shown).
function prepareFreshLineage() {
  var seed = cryptoSeed();
  if (seed === null) {
    showBanner('crypto.getRandomValues is unavailable — cannot seed a fresh lineage');
    return null;
  }
  var trackHash = state.activeTrackHash;
  var rs;
  try {
    rs = state.db.resumeState();
  } catch (err) {
    showBanner('Fresh lineage refused — could not read the database lineage state: ' + errMsg(err));
    return null;
  }
  if (rs.maxGeneration > 0) {
    showBanner('Fresh lineage refused — this database already holds a committed lineage (generation ' + rs.maxGeneration + '). Use "Wipe all generations — start a NEW lineage" in the Logs tab first.');
    return null;
  }
  var generatorVersion;
  try {
    generatorVersion = returnedGeneratorVersion(state.db.ensureLineage(trackHash, seed, state.activeTrack.toJSON()));
  } catch (err) {
    // Empty-but-pinned lineage (e.g. a previous fork/provisioning): re-pin —
    // forkLineage on an EMPTY lineage destroys no rows, it just re-pins
    // track_hash + rng_state for the new lineage.
    try {
      generatorVersion = returnedGeneratorVersion(state.db.forkLineage(trackHash, seed, state.activeTrack.toJSON()));
    } catch (err2) {
      showBanner('Fresh lineage refused — ' + errMsg(err2));
      return null;
    }
  }
  return { kind: 'fresh', seed: seed, rotationSeed: seed, generatorVersion: generatorVersion, trackHash: trackHash, track: state.activeTrack };
}

// prepareResumeLineage — returns { kind:'resume', generation, bestWeights, rngState, rotationSeed, trackHash, expectedTrackHash, track } or null.
// FIT-V3 (D-5) — the resume track is the STORED tracks row of the tip-
// generation pin (the row whose track_hash === rs.trackHash), adopted via
// Db.adoptPinnedTrack and rebuilt byte-identically (re-hash == rs.trackHash
// enforced there, typed ResumeTrackError otherwise — the LINEAGE GUARD
// banner). The ambient editor/Manual track is NEVER the resume track, and
// the old activeTrackHash comparison is gone: under rotation the lineage
// derives its tracks independently from the seed, so that check was
// meaningless by construction.
function prepareResumeLineage() {
  var rs;
  try {
    rs = state.db.resumeState();
  } catch (err) {
    showBanner('Resume refused — ' + errMsg(err));
    return null;
  }
  var latest = rs.maxGeneration;
  if (latest < 1) {
    showBanner('Nothing to resume — no generations are committed yet. Start a fresh lineage instead.');
    return null;
  }
  var picked = clampInt(el('resume-generation').value, 1, latest, latest);
  if (picked !== latest) {
    showBanner('Resume continues from the latest committed generation (' + latest + ') — re-evaluating generation ' + (picked + 1) + ' again would conflict with the committed prefix (UNIQUE) and is refused. Pick ' + latest + ', or wipe the lineage in the Logs tab to restart from generation 1.');
    return null;
  }
  if (rs.rotationSeed === null || rs.rotationSeed === undefined) {
    // No silent zero seed — a pinned v3 lineage always carries one.
    showBanner('Resume refused — this database carries no rotation_seed for the lineage (it was never pinned under the rotation model). Import a full export of the lineage, or start a fresh lineage.');
    return null;
  }
  if (!rs.trackHash) {
    showBanner('Warning — this database carries no track_hash pin; resume is allowed, but the lineage guard cannot protect against a different track.');
  } else {
    // Adopt the stored tracks row OF THE TIP PIN — the track the resumed
    // run continues on. Absent/unrecombinable row ⇒ typed refusal, no
    // generator fallback (D-5/G3).
    try {
      var adopted = state.db.adoptPinnedTrack(rs.trackHash);
      pushMiniLog('resumed from gen ' + latest + ' · rotation ' + rs.rotationIndex + ' · difficulty ' + (rs.difficulty === null || rs.difficulty === undefined ? '—' : rs.difficulty) + ' · source ' + (rs.trackSource || '—'));
      return {
        kind: 'resume',
        generation: latest,
        bestWeights: rs.bestWeights,
        rngState: rs.rngState,
        rotationSeed: rs.rotationSeed,
        generatorVersion: rs.generatorVersion,
        trackHash: rs.trackHash || null,
        expectedTrackHash: rs.trackHash || null,
        track: adopted,
      };
    } catch (err) {
      if (err && err.name === 'ResumeTrackError') {
        showBanner('LINEAGE GUARD — the track pinned for the last committed generation is not recoverable from this file — import a full export of this lineage to resume it; the committed database on disk is untouched. (' + errMsg(err) + ')');
      } else {
        showBanner('Resume refused — could not adopt the pinned track row: ' + errMsg(err));
      }
      return null;
    }
  }
  return {
    kind: 'resume',
    generation: latest,
    bestWeights: rs.bestWeights,
    rngState: rs.rngState,
    rotationSeed: rs.rotationSeed,
    generatorVersion: rs.generatorVersion,
    trackHash: rs.trackHash || null,
    expectedTrackHash: rs.trackHash || null,
    track: state.activeTrack, // pre-pin foreign database — no stored row to adopt; guarded by the banner above
  };
}

function resolveLineage() {
  if (!state.dbAvailable || !state.db) {
    showBanner('Database required — training is disabled; Manual still works.');
    return null;
  }
  return lineageKind() === 'resume' ? prepareResumeLineage() : prepareFreshLineage();
}

/* ============================================================
   10. Headless training (Trainer + Web Worker / fallback)
   ============================================================ */

// ---- FIT-V3: the rotation-aware training track resolution -----------------
// currentTrainingTrackFor(generation) — the Track the rotation curriculum
// dictates for `generation` in the CURRENT training session: rotation 0 is the
// lineage's pinned starting track (the track pinned at lineage creation; for a
// resume, the stored tip-generation tracks row that adoptPinnedTrack returned);
// rotation ≥ 1 is the generated curriculum track resolveGeneratedTrack(seed, r)
// — byte-identical to the pre-materialized tracks rows by construction.
// Cached by rotation index on state.trainTrackBlock — rebuilt at most once
// per 50-generation block; never within a block; never stale across a
// boundary. Returns a shared Track instance (the same class the cores build).
function currentTrainingTrackFor(generation) {
  var R = globalThis.Rotation;
  if (!R || typeof R.rotationForGeneration !== 'function') {
    throw new globalThis.ValidationError('Rotation is not available — js/rotation.js must load before training runs');
  }
  var ri = R.rotationForGeneration(generation);
  if (state.trainTrackBlock === ri && state.trainTrackObj) return state.trainTrackObj;
  var track;
  if (ri === 0) {
    track = state.trainPinTrack || state.activeTrack;
  } else {
    var resolved = R.resolveGeneratedTrack(state.rotationSeed, ri, state.generatorVersion);
    track = new globalThis.Track(resolved.controlPoints, resolved.halfWidth);
  }
  state.trainTrackBlock = ri;
  state.trainTrackObj = track;
  state.trainTrackHash = track.hash();
  return track;
}

// startRotationSession(lin) — pin the CURRENT session to the lineage
// resolveLineage() returned: fresh → the active track pinned at lineage
// creation; resume → the stored tracks row for the tip generation's rotation.
// Resets the block cache so the first generation re-derives from the pin.
function startRotationSession(lin) {
  // FIT-V4 — the canonical integer version is REQUIRED on every rotation
  // session (headless AND spectate). The strict parser lives in rotation.js;
  // a missing/empty field throws before any track resolution, never a default.
  var R = globalThis.Rotation;
  if (!R || typeof R.requireGeneratorVersion !== 'function') {
    throw new globalThis.ValidationError('Rotation.requireGeneratorVersion is not available — js/rotation.js must load before a rotation session starts');
  }
  state.generatorVersion = R.requireGeneratorVersion(lin && lin.generatorVersion);
  state.rotationSeed = (lin && lin.rotationSeed !== undefined && lin.rotationSeed !== null) ? lin.rotationSeed : 0;
  state.trainPinTrack = (lin && lin.track) || state.activeTrack;
  state.trainTrackBlock = -1;
  state.trainTrackObj = null;
  state.trainTrackHash = '';
}

// rotationLabelFor(generation) — the tiny 'R r · d d' tag text for the Train
// stats header. Difficulty is the ladder rung of the generation's rotation
// (D-15: difficulty consumed for display/audit only — never Selection/physics).
function rotationLabelFor(generation) {
  var R = globalThis.Rotation;
  if (!R || typeof R.rotationForGeneration !== 'function') return 'R — · d —';
  var ri = R.rotationForGeneration(generation);
  return 'R ' + ri + ' · d ' + R.difficultyForRotation(ri);
}

/* ---- UX: truthful evaluation-track label + contextual canvas names ------- */
// The HUD used to stay on "default circuit" forever even after rotation ≥ 1
// switched the drivers onto a generated circuit. Resolve the SAME track the
// rotation resolver uses and show its short hash. Rotation 0 keeps the pinned
// track's label. Updates Manual/Train/Replay HUDs and all canvas aria-labels.
function updateTrackLabel(generation) {
  var R = globalThis.Rotation;
  var ri = 0;
  if (Number(generation) >= 1 && R && typeof R.rotationForGeneration === 'function') {
    // rotationForGeneration refuses generation < 1; boot/reset call us with 0.
    try { ri = R.rotationForGeneration(Number(generation)); } catch (err) { ri = 0; }
  }
  var label;
  if (!ri || ri === 0) {
    label = state.activeTrackName || 'default circuit';
  } else {
    if (state.trainTrackBlock !== ri || !state.trainTrackObj) {
      // Best-effort: the session may not be pinned yet; the label still names
      // the rotation, and gains the hash as soon as the resolver has run.
      try { currentTrainingTrackFor(generation); } catch (err) { noop(); }
    }
    var hash = state.trainTrackHash || '';
    label = 'generated · R' + ri + (hash ? ' · ' + hash.slice(0, 8) + '…' : '');
  }
  state.currentTrackLabel = label;
  ['hud-track', 'train-track', 'replay-track'].forEach(function (id) {
    var node = el(id);
    if (node) node.textContent = label;
  });
  updateCanvasLabels();
}

// setCanvasLabel only writes when the text actually changed — avoids spamming
// assistive tech with redundant accessible-name mutations every frame.
function setCanvasLabel(id, text) {
  var c = el(id);
  if (c && c.getAttribute('aria-label') !== text) c.setAttribute('aria-label', text);
}

function updateCanvasLabels() {
  var R = globalThis.Rotation;
  var ri = state.trainTrackBlock >= 0 ? state.trainTrackBlock : 0;
  if (ri === 0 && R && typeof R.rotationForGeneration === 'function' && state.latestGen >= 1) {
    try { ri = R.rotationForGeneration(state.latestGen); } catch (err) { ri = 0; }
  }
  var diff = (R && typeof R.difficultyForRotation === 'function') ? R.difficultyForRotation(ri) : 0;
  var detail = ri > 0
    ? state.currentTrackLabel + ', rotation ' + ri + ', difficulty ' + diff
    : (state.activeTrackName || 'default circuit');
  setCanvasLabel('manual-canvas', 'Manual driving canvas: ' + detail);
  setCanvasLabel('train-canvas', 'Training canvas: ' + detail);
  setCanvasLabel('replay-canvas', 'Replay canvas: ' + detail);
  var editorCanvas = el('editor-canvas');
  if (editorCanvas) {
    var base = 'Track editor canvas — click empty space to add a point (inserted where it fits the loop best), drag a point to move it, right-click a point to delete it';
    setCanvasLabel('editor-canvas', base + ' Track: ' + detail);
  }
}

/* ---- UX: compact, accessible training progress -------------------------- */
var progressRaf = 0;
var pendingProgress = null;
function applyTrainProgress(gen, max) {
  var prog = el('train-progress');
  if (!prog) return;
  max = Math.max(1, Number(max) || MAX_GENERATIONS_DEFAULT);
  gen = Math.max(0, Number(gen) || 0);
  if (Number(prog.max) !== max) prog.max = String(max);
  prog.value = String(Math.min(gen, max));
  var txt = el('train-progress-text');
  if (txt) txt.textContent = gen + ' / ' + max;
}
function scheduleTrainProgress(gen, max) {
  pendingProgress = { gen: gen, max: max };
  if (progressRaf) return;
  progressRaf = requestAnimationFrame(function () {
    progressRaf = 0;
    var p = pendingProgress;
    pendingProgress = null;
    if (p) applyTrainProgress(p.gen, p.max);
  });
}

/* ---- UX: elements created from JS (index.html stays byte-pinned) --------- */
function ensureUiExtras() {
  if (el('train-progress')) return; // idempotent
  var controls = document.querySelector('#tab-train .panel[aria-label="Training controls"]');
  if (controls) {
    var wrap = document.createElement('div');
    wrap.className = 'train-progress';
    wrap.id = 'train-progress-wrap';
    var prog = document.createElement('progress');
    prog.id = 'train-progress';
    prog.max = String(MAX_GENERATIONS_DEFAULT);
    prog.value = '0';
    prog.setAttribute('aria-label', 'Training progress — generations committed');
    var text = document.createElement('span');
    text.id = 'train-progress-text';
    text.className = 'train-progress-text';
    text.textContent = '0 / ' + MAX_GENERATIONS_DEFAULT;
    wrap.appendChild(prog);
    wrap.appendChild(text);
    controls.appendChild(wrap);
  }
  var stats = document.querySelector('#tab-train .stats-grid');
  if (stats && !el('train-track')) {
    var stat = document.createElement('div');
    stat.className = 'stat';
    var statLabel = document.createElement('span');
    statLabel.className = 'hud-label';
    statLabel.textContent = 'Evaluation track';
    var statValue = document.createElement('span');
    statValue.id = 'train-track';
    statValue.className = 'stat-value';
    stat.appendChild(statLabel);
    stat.appendChild(statValue);
    stats.appendChild(stat);
  }
  var replayHud = document.querySelector('#tab-replay .hud-row');
  if (replayHud && !el('replay-track')) {
    var cell = document.createElement('div');
    cell.className = 'hud-cell';
    var cellLabel = document.createElement('span');
    cellLabel.className = 'hud-label';
    cellLabel.textContent = 'Track';
    var cellValue = document.createElement('span');
    cellValue.id = 'replay-track';
    cellValue.className = 'hud-value';
    cell.appendChild(cellLabel);
    cell.appendChild(cellValue);
    replayHud.appendChild(cell);
  }
  var replaySection = el('tab-replay');
  if (replaySection && !el('replay-empty')) {
    var empty = document.createElement('p');
    empty.id = 'replay-empty';
    empty.className = 'subtle-note';
    empty.textContent = 'No committed generations to replay yet — train first, then use Run replay.';
    empty.hidden = true;
    var controlsRow = replaySection.querySelector('.replay-controls');
    if (controlsRow && controlsRow.parentNode) controlsRow.parentNode.insertBefore(empty, controlsRow.nextSibling);
    else replaySection.insertBefore(empty, replaySection.firstChild);
  }
  var logsEmpty = el('logs-empty');
  if (logsEmpty) logsEmpty.textContent = 'No generations committed yet — start training in the Train tab.';
}

/* ---- UX: WAI-ARIA APG tabs keyboard support (roving tabindex) ------------ */
var TAB_ORDER = ['manual', 'train', 'replay', 'logs', 'editor'];
function enabledTabNames() {
  return TAB_ORDER.filter(function (name) {
    var b = el('tab-btn-' + name);
    return !!(b && !b.hidden && !b.disabled);
  });
}
function onTabsKeydown(event) {
  if (!event || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
  var key = event.key;
  if (key !== 'ArrowLeft' && key !== 'ArrowRight' && key !== 'Home' && key !== 'End') return;
  var tabs = enabledTabNames();
  if (!tabs.length) return;
  var focused = document.activeElement;
  var current = null;
  for (var i = 0; i < tabs.length; i++) {
    if (focused === el('tab-btn-' + tabs[i])) { current = tabs[i]; break; }
  }
  if (!current) current = state.activeTab;
  var idx = tabs.indexOf(current);
  if (idx < 0) return;
  var next;
  if (key === 'ArrowRight') next = tabs[(idx + 1) % tabs.length];
  else if (key === 'ArrowLeft') next = tabs[(idx - 1 + tabs.length) % tabs.length];
  else if (key === 'Home') next = tabs[0];
  else next = tabs[tabs.length - 1];
  if (!next || next === current) return;
  event.preventDefault();
  var btn = el('tab-btn-' + next);
  if (btn) btn.focus();
  activateTab(next); // panels are preloaded ⇒ automatic activation is correct
}

function buildLogGenerationPayload(p) {
  // p.results[].distance carries FITNESS (net signed odometer + lap bonus;
  // may be negative) — the SAME value selectBest consumed and spectate/headless
  // logged. buildLogGenerationPayload derives best/avg/worst from it; the
  // derivation is sign-safe (sums and comparisons, never absolutes) and the
  // written rows store fitness as-is.
  var results = p.results;
  var sum = 0, worst = Infinity, maxAlive = 0;
  for (var i = 0; i < results.length; i++) {
    var distance = results[i].distance;
    sum += distance;
    if (distance < worst) worst = distance;
    if (results[i].timeAlive > maxAlive) maxAlive = results[i].timeAlive;
  }
  // FIT-V3 — stamp the ROTATION track for p.generation, never the ambient
  // editor/Manual track for rotation generations. Same block ⇒ cache hit;
  // Manual contexts (no rotation session) fall back to the active track.
  var stampTrack = null;
  try {
    stampTrack = currentTrainingTrackFor(p.generation);
  } catch (err) {
    stampTrack = null;
  }
  // FIT-V3 — rotation-divergence assert (the plan's pin-move buddy). Two
  // INDEPENDENT derivations exist (the worker/fallback core vs this main
  // thread) and nothing compares them otherwise — with the SAME bytes they
  // agree forever; a divergence (e.g. a stale asset under file:// miscache)
  // would attribute fitness to the wrong track with zero downstream detection.
  // Refuse to log and halt the run — never log attribution of the NN to a
  // track it never drove on. Spectate never carries p.trackHash (single
  // thread derives both) and is unaffected; headless generations always do.
  if (stampTrack && typeof p.trackHash === 'string' && p.trackHash.length > 0 && stampTrack.hash() !== p.trackHash) {
    var divergenceErr = new Error('rotation divergence — the evaluation track for generation ' + p.generation + ' does not match the rotation resolver (worker track hash ' + JSON.stringify(p.trackHash) + ', resolver hash ' + JSON.stringify(stampTrack.hash()) + ') — the lineage is NOT updated; this tab stays halted');
    divergenceErr.name = 'LineageIntegrityError';
    divergenceErr.code = 'FIT_V3_ROTATION_DIVERGENCE';
    throw divergenceErr;
  }
  // Rotation sessions never fall back to the editor track — if the resolver
  // failed, that's a bug, not a signal to log the wrong geometry; halt.
  if (state.rotationSeed !== undefined && state.rotationSeed !== null && !stampTrack) {
    var resolverErr = new Error('rotation resolver failed for generation ' + p.generation + ' — refusing to log against the editor track (rotation sessions never fall back)');
    resolverErr.name = 'LineageIntegrityError';
    resolverErr.code = 'FIT_V3_ROTATION_DIVERGENCE';
    throw resolverErr;
  }
  return {
    generation: p.generation,
    driversEvaluated: results.length,
    bestFitness: results[p.bestIndex].distance,
    avgFitness: sum / results.length,
    worstFitness: worst,
    bestDriverIndex: p.bestIndex,
    mutationRate: MUTATION_RATE,
    mutationScale: MUTATION_SCALE,
    trackJson: stampTrack ? stampTrack.toJSON() : state.activeTrack.toJSON(),
    trackHash: stampTrack ? stampTrack.hash() : state.activeTrackHash,
    simSeconds: maxAlive,
    rngState: p.rngState,
    results: results,
    bestWeightsJson: p.bestWeights,
  };
}

function pushMiniLog(line) {
  var list = el('mini-log');
  var item = document.createElement('li');
  item.textContent = line;
  list.insertBefore(item, list.firstChild);
  while (list.childNodes.length > 5) list.removeChild(list.lastChild);
}

function trainingBlockedReason(err) {
  state.haltedLineage = true;
  updateTrainUi();
  showBanner('LINEAGE INTEGRITY ERROR — RUN HALTED — committed lineage untouched. (' + errMsg(err) + ') Reload and resume recomputes the unlogged generation identically.');
}

function onTrainProgress(progress) {
  state.latestGen = Math.max(state.latestGen || 0, Number(progress.generation) || 0); // commit-path cache (reviewer #17-19)
  el('stat-generation').textContent = String(progress.generation);
  // UX — the HUD must name the track the drivers were actually evaluated on,
  // not the stale "default circuit", once rotation ≥ 1 takes over.
  updateTrackLabel(progress.generation);
  scheduleTrainProgress(progress.generation, el('max-generations').value);
  var rotationElProgress = el('stat-rotation');
  if (rotationElProgress) {
    var rTag = Number.isInteger(progress.rotationIndex) ? progress.rotationIndex : null;
    var dTag = Number.isInteger(progress.difficulty) ? progress.difficulty : null;
    rotationElProgress.textContent = rTag !== null ? ('R ' + rTag + ' · d ' + (dTag !== null ? dTag : '—')) : rotationLabelFor(progress.generation);
  }
  el('stat-best').textContent = formatPx(progress.bestFitness);
  el('stat-avg').textContent = formatPx(progress.avgFitness);
  el('stat-worst').textContent = formatPx(progress.worstFitness);
  var sumAlive = 0;
  if (progress.results) {
    for (var i = 0; i < progress.results.length; i++) sumAlive += progress.results[i].timeAlive;
    el('stat-sim-seconds').textContent = formatSeconds(sumAlive / progress.results.length);
  }
  pushMiniLog('gen ' + progress.generation + ' · best ' + formatPx(progress.bestFitness) + ' · avg ' + formatPx(progress.avgFitness) + ' · worst ' + formatPx(progress.worstFitness) + ' · best #' + progress.bestIndex);
  var replayMax = el('replay-generation');
  replayMax.max = String(Math.max(1, progress.generation));
  refreshReplayBounds(); // generations committed — unlock replay as soon as one exists
  updateResumeInputBounds();
  drawTrainStatus('Headless training — generation ' + progress.generation + ' committed · best ' + formatPx(progress.bestFitness));
}

function onTrainerBanner(payload) {
  var type = payload && payload.type;
  var message = payload && payload.message ? String(payload.message) : 'Training event';
  var generation = payload && Number.isFinite(payload.generation) ? payload.generation : currentLatestGeneration();
  state.lastFailureGeneration = generation;
  switch (type) {
    case 'log-failed':
      if (payload.unique) {
        // SEC-24 carve-out — halt; Start/Resume disabled until reload.
        state.haltedLineage = true;
        showBanner('LINEAGE INTEGRITY ERROR — RUN HALTED — committed lineage untouched');
      } else {
        state.sessionLoggingDisabled = true;
        setLoggingIndicator(true);
        setPersistPill('skipped (logging disabled)', 'warn');
        showBanner(message); // trainer composes the exact SEC wording
      }
      break;
    case 'persist-skipped':
      showBanner('LOGGING DISABLED — persist skipped');
      setPersistPill('skipped (logging disabled)', 'warn');
      break;
    case 'lineage-mismatch':
      showBanner(message);
      break;
    case 'worker-failed-fallback':
      showBanner('Web Worker unavailable — running the in-thread MessageChannel backend (the lineage replays identically). ' + message);
      break;
    case 'worker-error':
    case 'halted':
      state.haltedLineage = true;
      showBanner(message); // trainer composes the extended LINEAGE wording
      break;
    default:
      showBanner(message);
  }
  updateTrainUi();
}

function onPersistRequested(payload) {
  if (!state.db) return;
  if (state.sessionLoggingDisabled) return; // trainer already skips under SEC-26
  state.persistInFlight += 1;
  setPersistPill('persisting…', 'warn');
  state.db.persist().then(function (res) {
    state.persistInFlight = Math.max(0, state.persistInFlight - 1);
    if (res && res.ok) {
      state.persistFailed = false;          // M4 — a successful persist clears the latch
      state.persistFailedText = '';
      setPersistPill('persisted · ' + formatBytes(res.bytesWritten) + ' · ' + new Date().toLocaleTimeString(), 'ok');
      refreshBannerState();                // a recovering pipeline clears the not-persisting banner (or re-renders the remaining one)
    } else if (res && res.skipped) {
      setPersistPill('skipped (logging disabled)', 'warn');
    } else if (res && res.reconcile) {
      state.reconcileRequired = true;
      setPersistPill('blocked — reconcile required', 'bad');
      showReconcileBanner();
    } else {
      state.persistFailed = true;          // res.ok === false (not skipped/reconciled) — the pipeline failed
      state.persistFailedText = 'DB NOT PERSISTING — export manually' + (res && res.reason ? ' — ' + res.reason : '');
      setPersistPill('failed — export manually', 'bad');
      refreshBannerState();                // the not-persisting banner re-renders — ✕ cannot clear it while the flag is set
    }
  }, function (err) {
    state.persistInFlight = Math.max(0, state.persistInFlight - 1);
    state.persistFailed = true;            // M4 — the persist() rejection branch
    state.persistFailedText = 'Persist pipeline error: ' + errMsg(err) + ' — export manually';
    setPersistPill('failed — export manually', 'bad');
    showBanner(state.persistFailedText);
  });
}

function startHeadless() {
  if (state.headlessRunning) return;
  var lin = resolveLineage();
  if (!lin) return;
  startRotationSession(lin); // FIT-V3 — pin the session to the lineage's seed + pin track; reset the block cache
  updateTrackLabel(lin.kind === 'resume' ? (Number(lin.generation) || 0) + 1 : 1); // UX — evaluate-track label from the first generation
  var maxGenerations = clampInt(el('max-generations').value, 1, MAX_GENERATIONS_DEFAULT, MAX_GENERATIONS_DEFAULT);
  var lineage = lin.kind === 'fresh'
    ? { seed: lin.seed, rotationSeed: lin.rotationSeed, generatorVersion: state.generatorVersion, trackHash: lin.trackHash, expectedTrackHash: lin.trackHash }
    : { resume: { bestWeights: lin.bestWeights, rngState: lin.rngState, generation: lin.generation }, rotationSeed: lin.rotationSeed, generatorVersion: state.generatorVersion, trackHash: lin.trackHash, expectedTrackHash: lin.expectedTrackHash };

  var logger = {
    logGeneration: function (p) {
      return state.db.logGeneration(buildLogGenerationPayload(p));
    },
  };

  var trainer;
  try {
    trainer = new globalThis.Trainer(lin.track || state.activeTrack, {
      lineage: lineage,
      logger: logger,
      onProgress: onTrainProgress,
      onBanner: onTrainerBanner,
      onPersist: onPersistRequested,
    });
  } catch (err) {
    showBanner('Training refused — ' + errMsg(err));
    return;
  }
  state.trainer = trainer;
  state.headlessRunning = true;
  var started;
  try {
    started = trainer.start(maxGenerations);
  } catch (err) {
    state.headlessRunning = false;
    state.trainer = null;
    showBanner('Training refused — ' + errMsg(err));
    updateTrainUi();
    return;
  }
  if (started === false) {
    state.headlessRunning = false; // lineage guard refused — onTrainerBanner already showed why
  }
  updateTrainUi();
}

function stopHeadless() {
  if (state.trainer) {
    try { state.trainer.stop(); } catch (err) { noop(); }
  }
  state.headlessRunning = false;
  scheduleTrainProgress(0, el('max-generations').value); // UX — reset on stop
  updateTrainUi();
}

// drawTrainStatus(text, track) — the ambient train-canvas placeholder.
// `track` optionally overrides state.activeTrack (spectate passes the
// rotation block's track so the canvas always shows what the cars run on).
function drawTrainStatus(text, track) {
  state.lastTrainStatus = { text: text, track: track || null };
  var canvas = el('train-canvas');
  var ctx = canvasContext(canvas);
  var drawTrack = track || state.activeTrack;
  if (!ctx || !drawTrack) return;
  drawGrass(ctx);
  drawTrack.draw(ctx);
  ctx.save();
  ctx.fillStyle = 'rgba(10, 12, 14, 0.55)';
  ctx.fillRect(0, CANVAS_H - 64, CANVAS_W, 64);
  ctx.fillStyle = '#ffd60a';
  ctx.font = '600 20px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(text, CANVAS_W / 2, CANVAS_H - 26);
  ctx.restore();
}

/* ============================================================
   12. Replay tab — defensive wiring against globalThis.Replay (M4-B)
   ============================================================ */

function replayApi() {
  var R = globalThis.Replay;
  // createRun is the one function the documented contract guarantees;
  // loadGeneration is used when present, with a DB-assembly fallback.
  if (R && typeof R.createRun === 'function') return R;
  return null;
}

// assembleReplayRun(loaded, useActive, n) — assembles the replay run inputs
// from EXACTLY the documented replay.js loadGeneration return shape:
//   { generation, results, bestWeights, track, stats, useActiveTrack, trackSource }
//   track  — active track (labeled toggle) or the loaded snapshot Track (a Track
//            built by the snapshot path).
//   weights — loaded.bestWeights (the plain VALIDATED payload createRun wants).
//   useActiveTrack / trackSource — 'active-track' iff the labeled toggle is
//            checked AND the active track exists; else 'generation-snapshot'.
//            BOTH are returned so runReplay can hand them to createRun —
//            without them the use-active-track warning banner is unreachable
//            and run.state.trackSource lies (F-4/SEC M1).
// replay.js is in-repo and loadGeneration() always returns the documented
// shape or throws — there is NO DB-assembly fallback. Throws (with a clear
// message) when the payload is missing/unusable; every consumed piece is
// asserted before it is used.
function assembleReplayRun(loaded, useActive, n) {
  var useActiveTrack = !!(useActive && state.activeTrack);
  var track = useActiveTrack ? state.activeTrack : null;
  var weights = null;
  if (!loaded || typeof loaded !== 'object') {
    throw new Error('Replay loadGeneration returned no payload — expected { generation, results, bestWeights, track, stats, useActiveTrack, trackSource }');
  }
  // Assert the documented shape — fail loudly instead of guessing payloads.
  if (loaded.track !== undefined && loaded.track !== null) {
    var t = loaded.track;
    if (!t || typeof t !== 'object' || typeof t.startPose !== 'function' || typeof t.nearestCenterline !== 'function') {
      throw new Error('Replay loadGeneration returned an unusable track — expected a Track built from the generation snapshot');
    }
    if (!track) track = t;
  }
  if (loaded.bestWeights !== undefined && loaded.bestWeights !== null) {
    if (typeof loaded.bestWeights !== 'object' || loaded.bestWeights === null) {
      throw new Error('Replay loadGeneration returned non-object bestWeights');
    }
    weights = loaded.bestWeights;
  }
  if (!track) {
    throw new Error('Replay loadGeneration returned no track and no active track is set — cannot assemble the replay for generation ' + n);
  }
  if (!weights) {
    throw new Error('Replay loadGeneration returned no bestWeights for generation ' + n);
  }
  return { track: track, weights: weights, useActiveTrack: useActiveTrack, trackSource: useActiveTrack ? 'active-track' : 'generation-snapshot' };
}

function replayCanvasCtx() {
  return canvasContext(el('replay-canvas'));
}

function onReplayFrame(frame) {
  if (frame && Number.isFinite(frame.x) && Number.isFinite(frame.y)) {
    state.lastReplayFrame = frame; // RESPONSIVE — replay after a resize/tab reveal
  }
  var canvas = el('replay-canvas');
  var ctx = canvasContext(canvas);
  if (!ctx || !state.activeTrack) return;
  var track = state.replayActiveTrack || state.activeTrack;
  drawGrass(ctx);
  track.draw(ctx);
  // The replay.js onFrame payload is FLAT — { x, y, heading, speed, odometer } —
  // draw the shared glyph straight from it (no car object, no .draw() speculation).
  if (frame && Number.isFinite(frame.x) && Number.isFinite(frame.y)) {
    if (Number.isFinite(frame.heading)) {
      ctx.globalAlpha = 1;
      ctx.fillStyle = '#ffd60a';
      drawCarGlyph(ctx, frame.x, frame.y, frame.heading);
    }
    el('replay-speed').textContent = formatSpeed(frame.speed);
    el('replay-odometer').textContent = Number.isFinite(frame.odometer) ? Math.round(frame.odometer) + ' px' : '— px';
  }
}

function onReplayFinish(generation, result) {
  state.replayRunning = false;
  state.replayRun = null;
  el('btn-run-replay').disabled = false;
  el('btn-stop-replay').disabled = true;
  el('replay-current-gen').textContent = '—';
  renderReplayStats(generation, result);
  refreshReplayBounds();
}

function onReplayBanner(payload) {
  var message = payload && payload.message ? String(payload.message) : (typeof payload === 'string' ? payload : 'Replay event');
  showBanner(message);
}

function renderReplayStats(generation, result) {
  var pre = el('replay-stats');
  var lines = [];
  // Committed stats from the database: best/avg/worst are fitness (net signed
  // odometer + lap bonus; may be negative, rendered by formatPx as-is).
  lines.push('generation ' + generation + ' — committed stats (from the database):');
  try {
    var gen = state.db.getGeneration(generation);
    if (gen) {
      lines.push('best ' + formatPx(gen.best_fitness) + ' · avg ' + formatPx(gen.avg_fitness) + ' · worst ' + formatPx(gen.worst_fitness));
      lines.push('best driver index #' + gen.best_driver_index + ' · sim ' + formatSeconds(gen.sim_seconds) + ' · recorded ' + String(gen.created_at));
      lines.push('track pin ' + shortHash(gen.track_hash));
      // FIT-V3 — the rotation/difficulty this generation belongs to (display
      // only, D-15; the ladder is pure math, no DB read needed).
      if (globalThis.Rotation && typeof globalThis.Rotation.rotationForGeneration === 'function') {
        var rIdx = globalThis.Rotation.rotationForGeneration(generation);
        lines.push('rotation ' + rIdx + ' · difficulty ' + globalThis.Rotation.difficultyForRotation(rIdx));
      }
    } else {
      lines.push('(no committed row found for generation ' + generation + ')');
    }
    var rows = state.db.getDriverResults(generation);
    var crashed = 0;
    for (var i = 0; i < rows.length; i++) if (Number(rows[i].crashed) === 1) crashed += 1;
    lines.push('drivers ' + rows.length + ' · crashed ' + crashed + '/' + rows.length);
  } catch (err) {
    lines.push('could not read generation stats: ' + errMsg(err));
  }
  if (result && typeof result === 'object') {
    if (result.trackSource === 'active-track') {
      lines.push('(active track — differs from committed pin)');
    }
    if (result.outcome) {
      // Outcome strings are rendered verbatim — 'error' (replay tick fault)
      // is the only mapped label; any other outcome (incl. 'finished') renders
      // as-is, never a crash (M10).
      lines.push('outcome: ' + (result.outcome === 'error' ? 'aborted — error (see console)' : result.outcome));
    }
    var extras = [];
    if (Number.isFinite(result.distance)) extras.push('run fitness ' + formatPx(result.distance));
    if (Number.isFinite(result.topSpeed)) extras.push('top speed ' + formatSpeed(result.topSpeed));
    if (Number.isFinite(result.timeAlive)) extras.push('time alive ' + formatSeconds(result.timeAlive));
    if (extras.length) lines.push('this run: ' + extras.join(' · '));
  }
  pre.textContent = lines.join('\n');
  pre.hidden = false;
}

function runReplay() {
  var R = replayApi();
  if (!R) {
    showBanner('Replay module not loaded (js/replay.js missing) — the rest of the game works.');
    return;
  }
  if (!requireDb('Replay')) return;
  var maxGen = currentLatestGeneration();
  if (maxGen < 1) {
    showBanner('No generations committed yet — train first, then replay.');
    return;
  }
  var n = clampInt(el('replay-generation').value, 1, maxGen, maxGen);
  var useActive = el('replay-use-active').checked;

  el('replay-stats').hidden = true;
  el('replay-stats').textContent = '';
  el('replay-current-gen').textContent = String(n);
  el('replay-speed').textContent = '— px/s';
  el('replay-odometer').textContent = '— px';
  state.replayRunning = true;
  el('btn-run-replay').disabled = true;
  el('btn-stop-replay').disabled = false;

  var p = Promise.resolve().then(function () { return R.loadGeneration(state.db, n, { useActiveTrack: useActive, activeTrack: state.activeTrack }); });

  p.then(function (loaded) {
    var assembled = assembleReplayRun(loaded, useActive, n);
    state.replayActiveTrack = assembled.track;
    var run = R.createRun({
      track: assembled.track,
      weights: assembled.weights,
      useActiveTrack: assembled.useActiveTrack,
      trackSource: assembled.trackSource,
      generation: n,
      onFrame: onReplayFrame,
      onFinish: function (result) { onReplayFinish(n, result); },
      onBanner: onReplayBanner,
    });
    state.replayRun = run && typeof run.stop === 'function' ? run : null;
    if (!state.replayRun) el('btn-stop-replay').disabled = true; // no handle to stop — run plays out
    // createRun returns an idle run with start() — start it (unguarded). A
    // start failure resets the Replay panel and says so; nothing plays.
    try {
      run.start();
    } catch (err) {
      state.replayRunning = false;
      state.replayRun = null;
      el('btn-run-replay').disabled = false;
      el('btn-stop-replay').disabled = true;
      showBanner('Replay failed to start: ' + errMsg(err));
    }
  }, function (err) {
    state.replayRunning = false;
    state.replayRun = null;
    el('btn-run-replay').disabled = false;
    el('btn-stop-replay').disabled = true;
    showBanner('Replay refused — ' + errMsg(err));
  });
}

function stopReplay() {
  if (state.replayRun && typeof state.replayRun.stop === 'function') {
    try { state.replayRun.stop(); } catch (err) { noop(); }
  }
  state.replayRunning = false;
  state.replayRun = null;
  el('btn-run-replay').disabled = false;
  el('btn-stop-replay').disabled = true;
  el('replay-current-gen').textContent = '—';
}

/* ============================================================
   13. Logs tab — table + expansion + buttons
   ============================================================ */

var LOGS_PAGE_SIZE = 50;
var showMoreLogsBtn = null; // lazily created by ensureShowMoreButton()

// buildLogRow(row) — ONE <tr> from a listGenerationSummaries() row.
function buildLogRow(row) {
  var tr = document.createElement('tr');
  tr.className = 'gen-row';
  tr.setAttribute('tabindex', '0');
  tr.setAttribute('data-generation', String(row.generation));
  tr.setAttribute('aria-expanded', 'false');

  var caretCell = document.createElement('td');
  var caret = document.createElement('span');
  caret.className = 'gen-caret';
  caret.textContent = '▸';
  caretCell.appendChild(caret);
  caretCell.appendChild(document.createTextNode(String(row.generation)));

  var cells = [
    formatPx(row.best_fitness),
    formatPx(row.avg_fitness),
    formatPx(row.worst_fitness),
    String(row.best_driver_index),
    String(row.crashed_count) + ' / 20',
    (row.rotationIndex === null || row.rotationIndex === undefined) ? '—' : String(row.rotationIndex),
    (row.difficulty === null || row.difficulty === undefined) ? '—' : String(row.difficulty),
    String(row.created_at),
  ];
  tr.appendChild(caretCell);
  for (var c = 0; c < cells.length; c++) {
    var td = document.createElement('td');
    td.className = 'num';
    td.textContent = cells[c];
    tr.appendChild(td);
  }
  return tr;
}

// listLogPage(offset) — fetch ONE page of generation summaries (newest
// first, at most LOGS_PAGE_SIZE) and append the rows to the tbody. Returns
// the number of rows appended. Throws on DB failure (caller renders the note).
function listLogPage(offset) {
  var rows = state.db.listGenerationSummaries(LOGS_PAGE_SIZE, offset);
  var fragment = document.createDocumentFragment();
  for (var i = 0; i < rows.length; i++) {
    fragment.appendChild(buildLogRow(rows[i]));
  }
  el('logs-tbody').appendChild(fragment);
  return rows.length;
}

// ensureShowMoreButton() — lazily creates the "Show 50 more" button (its
// label is set via textContent); it appends the next page on click.
function ensureShowMoreButton() {
  if (showMoreLogsBtn) return showMoreLogsBtn;
  var btn = document.createElement('button');
  btn.type = 'button';
  btn.id = 'btn-show-more-logs';
  btn.textContent = 'Show 50 more';
  btn.addEventListener('click', showMoreLogs);
  var anchor = el('logs-empty');
  var parent = anchor && anchor.parentNode;
  if (parent && typeof parent.insertBefore === 'function') {
    parent.insertBefore(btn, anchor);
  } else {
    el('tab-logs').appendChild(btn); // anchor not attached yet — fall back to the section
  }
  showMoreLogsBtn = btn;
  return btn;
}

function setShowMoreVisible(visible) {
  ensureShowMoreButton().hidden = !visible;
}

// showMoreLogs() — append the NEXT page (the LIMIT/OFFSET are computed from
// state.logPage, which refreshLogs() reseeds to 0).
function showMoreLogs() {
  if (!state.dbAvailable || !state.db) return;
  var added = 0;
  try {
    added = listLogPage(state.logPage);
  } catch (err) {
    var note = el('logs-disabled-note');
    note.textContent = 'Logs failed to load the next page: ' + errMsg(err);
    note.hidden = false;
    setShowMoreVisible(false);
    return;
  }
  state.logPage += added;
  setShowMoreVisible(added === LOGS_PAGE_SIZE);
}

function refreshLogs() {
  var note = el('logs-disabled-note');
  var tbody = el('logs-tbody');
  if (!state.dbAvailable || !state.db) {
    note.textContent = 'Logs require the database — unavailable (see banner). Manual works; training and replay are disabled.';
    note.hidden = false;
    tbody.replaceChildren();
    el('logs-empty').hidden = true;
    state.logPage = 0;
    setShowMoreVisible(false);
    return;
  }
  note.hidden = true;
  state.logPage = 0;
  tbody.replaceChildren();
  var added = 0;
  try {
    added = listLogPage(0);
  } catch (err) {
    note.textContent = 'Logs failed to load: ' + errMsg(err);
    note.hidden = false;
    setShowMoreVisible(false);
    return;
  }
  state.logPage = added;
  el('logs-empty').hidden = added > 0;
  setShowMoreVisible(added === LOGS_PAGE_SIZE);
}

function toggleLogRow(row) {
  var gen = Number(row.getAttribute('data-generation'));
  if (!Number.isInteger(gen)) return;
  var next = row.nextSibling;
  var expanded = row.getAttribute('aria-expanded') === 'true';
  if (next && next.classList && next.classList.contains('gen-expand')) {
    next.parentNode.removeChild(next);
    row.setAttribute('aria-expanded', 'false');
    var caret = row.querySelector('.gen-caret');
    if (caret) caret.textContent = '▸';
    return;
  }
  if (expanded) return;
  var results;
  try {
    results = state.db.getDriverResults(gen);
  } catch (err) {
    results = [];
  }
  var expand = document.createElement('tr');
  expand.className = 'gen-expand';
  var td = document.createElement('td');
  td.colSpan = 9;
  var table = document.createElement('table');
  table.className = 'driver-table';
  var thead = document.createElement('thead');
  var headRow = document.createElement('tr');
  var headers = ['Driver', 'Fitness', 'Top speed (px/s)', 'Time alive (s)', 'Crashed', 'Stuck'];
  for (var h = 0; h < headers.length; h++) {
    var th = document.createElement('th');
    th.scope = 'col';
    th.textContent = headers[h];
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);
  table.appendChild(thead);
  var tb = document.createElement('tbody');
  for (var i = 0; i < results.length; i++) {
    var r = results[i];
    var tr = document.createElement('tr');
    var values = [
      String(r.driver_index),
      Number(r.distance).toFixed(1),
      Number(r.top_speed).toFixed(1),
      Number(r.time_alive).toFixed(2),
      Number(r.crashed) === 1 ? 'yes' : 'no',
      Number(r.stuck) === 1 ? 'yes' : 'no',
    ];
    for (var v = 0; v < values.length; v++) {
      var cell = document.createElement('td');
      cell.className = 'num';
      cell.textContent = values[v];
      tr.appendChild(cell);
    }
    tb.appendChild(tr);
  }
  table.appendChild(tb);
  td.appendChild(table);
  expand.appendChild(td);
  row.parentNode.insertBefore(expand, row.nextSibling);
  row.setAttribute('aria-expanded', 'true');
  var caretEl = row.querySelector('.gen-caret');
  if (caretEl) caretEl.textContent = '▾';
}

function onLogsTableClick(event) {
  var row = event.target && event.target.closest ? event.target.closest('tr.gen-row') : null;
  if (row) toggleLogRow(row);
}

function onLogsTableKeydown(event) {
  if (event.key !== 'Enter' && event.key !== ' ') return;
  var row = event.target && event.target.closest ? event.target.closest('tr.gen-row') : null;
  if (!row) return;
  event.preventDefault();
  toggleLogRow(row);
}

/* ============================================================
   14. Tabs — never cancel a run; Manual/Replay entry confirms
   ============================================================ */

var TAB_SECTIONS = {
  manual: 'tab-manual',
  train: 'tab-train',
  replay: 'tab-replay',
  logs: 'tab-logs',
  editor: 'tab-editor',
};

// anyRunActive() — the destructive-path guard (F-1/ADR-0005): true while a
// LINEAGE-WRITING run is active (headless trainer or spectate). An armed
// (waiting-for-next) spectate generation counts too — it is an in-place
// armed lineage step, still not safe to swap the DB under it. Replay is
// read-only and deliberately EXCLUDED — a fork during replay cannot corrupt
// the committed lineage. Used by forkNewLineage (the Logs wipe and the
// reconcile Fork button both funnel through it) and by restoreFromBrowserCache.
function anyRunActive() {
  return state.headlessRunning || !!(state.spectate && (state.spectate.running || state.spectate.waitingForNext));
}

// anyVisibleRunActive() — the tab-switch leave-confirm: ALSO counts replay
// (a run would be left playing behind a hidden tab). Superset of anyRunActive.
function anyVisibleRunActive() {
  return anyRunActive() || state.replayRunning;
}

// A lineage-writing run is active — no DB swap, replace, import, open, reconnect,
// reload-from-disk, create, fresh-cache or fork may run. Refusal fires BEFORE any picker opens.
function requireNoLineageWriters(actionLabel) {
  if (anyRunActive()) {
    showBanner('Stop the active run (Train → Stop) before ' + actionLabel);
    return false;
  }
  return true;
}

function stopActiveRuns() {
  if (state.headlessRunning) stopHeadless();
  if (state.spectate && state.spectate.running) stopSpectate();
  if (state.replayRunning) stopReplay();
}

function activateTab(name) {
  if (!TAB_SECTIONS[name]) return;
  if (name === 'editor' && !state.editorReady) return; // editor module absent/failed — tab stays hidden
  var leaving = state.activeTab;
  if (leaving !== name && (name === 'manual' || name === 'replay') && anyVisibleRunActive()) {
    var ok = window.confirm('A run is active — stop it and switch tabs?');
    if (!ok) return;
    stopActiveRuns();
  }
  if (leaving === 'editor' && name !== 'editor' && globalThis.Editor && typeof globalThis.Editor.close === 'function') {
    try { globalThis.Editor.close(); } catch (err) { /* a best-effort cleanup must not break tab switching */ }
  }
  state.keys = Object.create(null); // fresh key set on every tab switch
  state.activeTab = name;
  for (var key in TAB_SECTIONS) {
    if (!Object.prototype.hasOwnProperty.call(TAB_SECTIONS, key)) continue;
    var btn = el('tab-btn-' + key);
    var section = el(TAB_SECTIONS[key]);
    var selected = key === name;
    if (btn) {
      btn.setAttribute('aria-selected', selected ? 'true' : 'false');
      btn.setAttribute('tabindex', selected ? '0' : '-1'); // APG roving tabindex
    }
    if (section) section.hidden = !selected;
  }
  redrawAllCanvases(); // a revealed canvas has a new CSS box / DPR-sized bitmap
  if (name === 'logs') refreshLogs();
  if (name === 'train') {
    if (state.spectate && state.spectate.running) {
      drawSpectateFrame();
    } else if (!state.spectate) {
      drawTrainStatus('Start training (headless) or press Spectate — 20 cars live');
    }
  }
  if (name === 'editor' && globalThis.Editor && typeof globalThis.Editor.open === 'function') {
    try { globalThis.Editor.open(); } catch (err) { /* a broken render must not break tab switching */ }
  }
}

/* ============================================================
   15. Status line + fps
   ============================================================ */

function dbStateText() {
  if (!state.dbAvailable || !state.db) return 'db: offline';
  if (state.dbState === 'fs') return 'db: fs (' + (state.dbName || 'file') + ')';
  if (state.dbState === 'cache') return 'db: browser cache';
  return 'db: in-memory';
}

function updateStatusLine() {
  // NO table poll — the cached value moves only on commit (refreshPanels /
  // resume/replay bound refreshes / onTrainProgress) — reviewer #17-19.
  var latest = state.dbAvailable && state.db ? state.latestGen : 0;
  el('status-line').textContent =
    'mode: ' + state.activeTab +
    ' · ' + dbStateText() +
    ' · generation ' + latest +
    ' · ticks ' + state.ticks +
    ' · fps ' + String(Math.round(state.fps));
}

/* ============================================================
   16. Static wiring — attach every handler exactly once
   ============================================================ */

function wireStaticUi() {
  ensureUiExtras(); // creates progress / track HUDs / empty states (JS-only, CSP-safe)
  window.addEventListener('resize', onWindowResize); // single rAF-debounced listener
  var tablist = document.querySelector('.tabs');
  if (tablist) tablist.addEventListener('keydown', onTabsKeydown);

  // Header DB buttons
  el('btn-open-db').addEventListener('click', openDbPicker);
  el('btn-create-db').addEventListener('click', createDb);
  el('btn-export-db').addEventListener('click', exportDb);
  el('btn-import-db').addEventListener('click', importDb);

  // Banner
  el('persist-banner-close').addEventListener('click', refreshBannerState);
  el('btn-reconnect-db').addEventListener('click', reconnectDb);

  // Tabs
  el('tab-btn-manual').addEventListener('click', function () { activateTab('manual'); });
  el('tab-btn-train').addEventListener('click', function () { activateTab('train'); });
  el('tab-btn-replay').addEventListener('click', function () { activateTab('replay'); });
  el('tab-btn-logs').addEventListener('click', function () { activateTab('logs'); });
  // Editor tab — the button ships ENABLED; boot disables + hides it when
  // js/editor.js is absent/failed (activateTab gates on state.editorReady).
  el('tab-btn-editor').addEventListener('click', function () { activateTab('editor'); });

  // Manual
  window.addEventListener('keydown', function (e) { onManualKey(e, true); }, { passive: false });
  window.addEventListener('keyup', function (e) { onManualKey(e, false); }, { passive: false });
  window.addEventListener('blur', function () { state.keys = Object.create(null); });
  el('btn-reset-car').addEventListener('click', resetManualCar);

  // Train — headless + spectate
  el('btn-start-training').addEventListener('click', startHeadless);
  el('btn-stop-training').addEventListener('click', stopHeadless);
  el('btn-spectate').addEventListener('click', function () {
    if (state.spectate && (state.spectate.running || state.spectate.waitingForNext)) {
      // running ⇒ stop the eval; armed (waiting-for-next) ⇒ the same click disarms (F-14)
      stopSpectate();
    } else {
      startSpectate();
    }
  });
  el('btn-next-generation').addEventListener('click', nextSpectateGeneration);
  el('lineage-fresh').addEventListener('change', updateTrainUi);
  el('lineage-resume').addEventListener('change', function () {
    if (el('lineage-resume').checked) {
      var latest = currentLatestGeneration();
      el('resume-generation').value = String(Math.max(1, latest || Number(el('resume-generation').min) || 1));
    }
    updateTrainUi();
  });
  el('resume-generation').addEventListener('change', function () {
    var latest = Math.max(1, currentLatestGeneration());
    el('resume-generation').value = String(clampInt(el('resume-generation').value, 1, latest, latest));
  });

  // Replay
  el('btn-run-replay').addEventListener('click', runReplay);
  el('btn-stop-replay').addEventListener('click', stopReplay);

  // Logs
  el('btn-refresh-logs').addEventListener('click', function () {
    refreshLogs();
    updateStorageLine();
  });
  el('btn-export-db-file').addEventListener('click', exportDb);
  el('btn-import-sqlite').addEventListener('click', importDb);
  el('btn-new-lineage').addEventListener('click', function () { forkNewLineage(); });
  // Fork to new track — uses the CURRENT EDITOR TRACK (edit it in the Editor
  // tab first) with the same confirm flow as the editor's own button.
  // Boot disables this button when js/editor.js is unavailable.
  el('btn-fork-track').addEventListener('click', function () { useEditorTrackAsTrainingTrack(); });
  el('logs-tbody').addEventListener('click', onLogsTableClick);
  el('logs-tbody').addEventListener('keydown', onLogsTableKeydown);

  // Import fallback file input
  el('import-file-input').addEventListener('change', onImportFileInputChange);
}

/* ============================================================
   17. rAF loop — manual + spectate stepping; fps tracking
   ============================================================ */

function frame(ts) {
  requestAnimationFrame(frame);
  if (!state.lastFrameTs) {
    state.lastFrameTs = ts;
    return;
  }
  var delta = Math.min(0.1, Math.max(0, (ts - state.lastFrameTs) / 1000));
  state.lastFrameTs = ts;
  if (delta > 0) state.fps = state.fps * 0.92 + (1 / delta) * 0.08;

  if (state.activeTab === 'manual') {
    state.manualAcc += delta;
    var steps = 0;
    while (state.manualAcc >= SIM_DT && steps < 16) {
      stepManual();
      state.manualAcc -= SIM_DT;
      steps += 1;
    }
    if (steps === 16) state.manualAcc = 0;
    renderManual();
  } else if (state.activeTab === 'train' && state.spectate && state.spectate.running) {
    state.spectateAcc += delta;
    var spectateSteps = 0;
    // F-6/SEC L4 — a tick that throws (non-finite NN output) is terminal:
    // running=false, waitingForNext=false (Next stays hidden — nothing was
    // evaluated, so spawning generation N+1 would be an unevaluated
    // selection), NO fabricated results, NO recordEvaluation for this driver,
    // NO halt/logging state change. The Spectate toggle re-arms for a
    // one-click retry. DriverSim.step() throws ValidationError on non-finite
    // [gas, steer] — exactly the old spectate behavior.
    try {
      while (state.spectateAcc >= SIM_DT && spectateSteps < 16) {
        var allDone = true;
        for (var i = 0; i < state.spectate.cars.length; i++) {
          var carEntry = state.spectate.cars[i];
          if (carEntry.sim.outcome === null) {
            carEntry.sim.step();
            state.ticks += 1;
          }
          if (carEntry.sim.outcome === null) allDone = false;
        }
        state.spectateAcc -= SIM_DT;
        spectateSteps += 1;
        if (allDone) {
          finalizeSpectate();
          break;
        }
      }
    } catch (err) {
      var failGen = state.spectate ? state.spectate.generation : '?';
      if (state.spectate) {
        state.spectate.running = false;
        state.spectate.waitingForNext = false;
      }
      showBanner('Spectate aborted — network produced non-finite output; Gen ' + failGen + ' not logged');
      updateTrainUi();
    }
    if (spectateSteps === 16) state.spectateAcc = 0;
    if (state.spectate) drawSpectateFrame();
  }
}

/* ============================================================
   18. Boot — DB init, session restore, loops, interval sync
   ============================================================ */

// knownVendorRejection(reason) — the emscripten/shader signatures ONLY. The
// vendor/sql-wasm.js emscripten loading chain can surface a rejection with no
// handler on the locateFile-failure edge; swallowing that exact signature
// keeps a stray console error from alarming users. The MATCH WINDOW IS
// DELIBERATELY NARROW — an exception from our own code can never be masked by
// this filter, because our typed errors never carry these strings. An error
// message merely CONTAINING 'WebAssembly' is no longer swallowed (earlier
// builds matched it; tightened): if the vendored loader itself throws, the
// message must independently contain one of the two fetch/compile signatures
// or 'Aborted(' to be excused here — anything else is reported, not hidden.
function knownVendorRejection(reason) {
  var text = String((reason && reason.message) || reason || '');
  // emscripten's fatal-abort wrapper always throws "Aborted(...)"; the
  // locateFile-failure edge surfaces as these streaming/sync-compile logs.
  return /Aborted\(|wasm streaming compile|both async and sync fetching/i.test(text);
}

function setDbUnavailable(err) {
  state.dbAvailable = false;
  state.db = null;
  state.dbState = 'memory';
  state.dbName = '';
  setDbStatusPill();
  setPersistPill('offline', 'bad');
  // FIT-V6 — a wall-model refusal is NOT a vendor/offline fault: show its
  // persistent export + verbatim-download banner, without the README suffix.
  var wallRefused = isWallModelChanged(err) ||
    String((err && err.message) || '').indexOf(WALL_MODEL_BANNER_MARK) !== -1;
  if (wallRefused) {
    showWallModelRefusalBanner(err && err.refusedBytes,
      isWallModelChanged(err) ? wallModelChangedBanner(err) : String(err.message));
  } else {
    showBanner('Database required — training disabled (Manual works). Reason: ' + errMsg(err) + ' — see README (serve over localhost or fix the vendor files).');
  }
  updateTrainUi();
  refreshPanels();
}

function onBootRestored(restored, bootDb) {
  state.db = bootDb; // boot passes restored.db when the restore owns its scratch instance (cache path)
  state.dbAvailable = true;
  state.persistFailed = false;
  state.persistFailedText = '';
  state.dbName = restored.name || '';
  state.dbState = restored.state === 'fs' ? 'fs' : restored.state === 'cache' ? 'cache' : 'memory';

  bootDb.setPersistenceHooks({
    onBanner: onDbBanner,
    persistSkipped: function () { return state.sessionLoggingDisabled; },
  });

  if (state.dbState === 'cache' && globalThis.navigator && globalThis.navigator.storage && typeof globalThis.navigator.storage.persist === 'function') {
    globalThis.navigator.storage.persist().catch(noop); // fallback mode — ask the browser to keep our bytes
  }

  if (restored.state === 'needs-reconnect') {
    state.pendingReconnectHandle = restored.handle || null;
    showReconnectBanner(restored.message || 'Connected handle needs permission — Reconnect database');
  } else if (restored.message) {
    if (restored.refusedBytes) {
      // FIT-V6 byte-preserving refusal from the session layer — offer the
      // verbatim download action alongside the refusal banner.
      showWallModelRefusalBanner(restored.refusedBytes, restored.message);
    } else {
      showBanner(restored.message); // "Database connected…" / "Loaded … from cache…"
    }
  }

  setDbStatusPill();
  refreshPanels();
  updateTrainUi();
}

function boot() {
  // Active track (default circuit or validated localStorage custom track)
  var loaded = loadActiveTrack();
  state.activeTrack = loaded.track;
  state.activeTrackHash = state.activeTrack.hash();
  state.activeTrackName = loaded.name;
  el('hud-track').textContent = loaded.name;
  if (loaded.note) {
    var note = el('manual-note');
    note.textContent = loaded.note;
    note.hidden = false;
  }

  // Manual car spawn
  var pose = state.activeTrack.startPose();
  state.manualCar = new globalThis.Car(state.activeTrack, pose.x, pose.y, pose.heading);
  el('hud-speed').textContent = '0 px/s';
  el('hud-odometer').textContent = '0 px';

  wireStaticUi();       // ensureUiExtras() runs here, before any label/progress write
  updateTrackLabel(0);  // UX — truthful initial track label + contextual canvas names
  applyTrainProgress(0, el('max-generations').value);
  refreshBannerState(); // hides the banner until a source shows it

  // K — subtle-crypto absence (F-24 class): crypto.subtle (SHA-256) is
  // missing ⇒ file corruption detection degrades to size + first/last-4-KiB
  // edges (equal-length mid-file corruption is undetectable). Persistent
  // one-line note via the banner system under its own distinct type
  // ('subtle-absent') — never auto-hides; refreshBannerState() re-renders it
  // whenever no higher-priority fault is active. Detection mirrors db.js's
  // own subtle feature-detect exactly. The README bullet carrying this exact
  // wording is the docs pass's job — do NOT paraphrase either side.
  if (!(globalThis.crypto && globalThis.crypto.subtle && typeof globalThis.crypto.subtle.digest === 'function')) {
    onDbBanner({ type: 'subtle-absent', persistent: true, message: 'Browser without SHA-256 (subtle crypto) — file corruption detection reduced to size+edge compare' });
  }

  // Track editor (OPTIONAL module — js/editor.js loads BEFORE js/main.js).
  // When window.Editor exists and init succeeds, the Editor tab works end to
  // end — including its output hooks (onSaved/onStatus) driving the footer and
  // the lineage-guarded "Use as training track" flow. When it does not, the
  // Editor tab stays hidden (and never activates) and "Fork to new track"
  // stays disabled — nothing else is affected.
  var editorReady = false;
  if (globalThis.window && window.Editor && typeof window.Editor.init === 'function') {
    try {
      window.Editor.init(el('editor-canvas'), editorHooks());
      editorReady = true;
    } catch (err) {
      try { console.error('[editor] Editor.init failed:', err); } catch (consoleErr) { /* console is best-effort */ }
      showBanner('Track editor failed to load (' + errMsg(err) + ') — Editor tab hidden; the rest of the game is unaffected');
    }
  }
  state.editorReady = editorReady;
  if (editorReady) {
    el('btn-fork-track').disabled = false;
  } else {
    var editorTabBtn = el('tab-btn-editor');
    if (editorTabBtn) editorTabBtn.hidden = true;
    var editorSection = el('tab-editor');
    if (editorSection) editorSection.hidden = true;
    var forkTrackBtn = el('btn-fork-track');
    forkTrackBtn.disabled = true;
    forkTrackBtn.title = 'Track editor unavailable — js/editor.js failed to load';
  }

  // Database boot: createBrowser → restore → hooks.
  // Suppression — the vendor/sql-wasm.js emscripten loading chain can surface
  // a rejection with no handler on the locateFile-failure edge (e.g. a corrupt
  // wasm served over http), so its failure shows up as an UNHANDLED rejection.
  // It can fire one microtask AFTER our own .then chain settles (removing the
  // listener earlier lost the race — verified over CDP), so this listener is
  // deliberately left installed for the session's life. It swallows ONLY the
  // emscripten/shader signatures matched by knownVendorRejection() — anything
  // else is reported, never masked.
  function onVendorUnhandled(event) {
    if (event && knownVendorRejection(event.reason) && typeof event.preventDefault === 'function') {
      event.preventDefault();
    }
  }
  if (typeof globalThis.addEventListener === 'function') {
    globalThis.addEventListener('unhandledrejection', onVendorUnhandled);
  }

  if (typeof Db.createBrowser !== 'function') {
    setDbUnavailable(new Error('Db.createBrowser is missing — js/db.js failed to load'));
  } else {
    var bootDb = new Db();
    Db.createBrowser().then(function () {
      return globalThis.DbSession.restoreIfPossible(bootDb);
    }).then(function (restored) {
      if (!restored || restored.state === 'unavailable') {
        setDbUnavailable(new Error(restored && restored.message ? restored.message : 'database initialization failed'));
        return;
      }
      // A cache restore owns its scratch instance — adopt it (the passed-in
      // bootDb stays a never-initialized shell and is discarded).
      onBootRestored(restored, restored.db || bootDb);
    }, function (err) {
      setDbUnavailable(err);
    });
  }

  // Spectate canvas placeholder
  drawTrainStatus('Start training (headless) or press Spectate — 20 cars live');

  // Train-canvas idle when switching to the tab happens before any run
  el('status-line').textContent = 'booting…';

  requestAnimationFrame(frame);
  setInterval(function () {
    // Keep the train UI in sync with the backend state (done/stopped/halted)
    if (state.trainer) {
      var s = state.trainer.state;
      if (state.headlessRunning && (s === 'done' || s === 'stopped' || s === 'halted')) {
        state.headlessRunning = false;
        updateTrainUi();
      }
    }
    updateStatusLine();
  }, 500);
}

if (BOOT_CONTRACT_MISSING.length === 0) {
  boot();
}
