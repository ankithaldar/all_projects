// js/editor.js — In-browser track editor for the 0003_car_racer game (M5).
//
// Plain classic script — no build step, no modules. Exposes exactly one
// global: globalThis.Editor. The module is OPTIONAL: js/editor.js loads
// BEFORE js/main.js, and main.js calls Editor.init(canvasEl, hooks) during
// boot when — and only when — globalThis.Editor exists. When this file fails to
// load, main.js hides the Editor tab and keeps the Logs "Fork to new track"
// button disabled; everything else works unchanged.
//
// DOM DISCIPLINE: this file NEVER uses innerHTML. Every dynamic DOM write is
// textContent / createElement / setAttribute. Canvas text (ctx.fillText) is
// allowed — nothing on canvas is HTML parsing. There is no fetch() anywhere.
//
// STATE:
//   points    — [{x, y}…] editor control points (seeded from the localStorage
//               'car_racer.track' track when it validates via Track.fromJSON,
//               else the DEFAULT_CONTROL_POINTS circuit, else a blank state
//               guarded by the editor-canvas' minimum-three-points rule).
//   halfWidth — road half-width in px, [1, 2000], default 70.
//   undo      — single-level snapshot { points, halfWidth } taken BEFORE the
//               last structural op (add / move / delete / width-change /
//               import / reset / undo itself restores and clears).
//   dirty     — road rebuild flag; the cached Track is rebuilt lazily on the
//               next render, never mid-pointer-move more than needed.
//
// EDITING RULES:
//   * Click empty space ⇒ a point is INSERTED where it best fits the closed
//     loop (the control-polygon edge nearest the click) — not appended, which
//     would scramble the loop's point order.
//   * Drag an existing point (pointerdown within 12 px) to move it; the
//     pointer is captured for the duration of the gesture.
//   * Right-click (contextmenu) a point to delete it; the closed loop is
//     never allowed below 3 points.
//   * Every structural op persists to localStorage 'car_racer.track' and
//     notifies the host via hooks.onSaved(trackJSON). The ACTIVE track is NOT
//     hot-swapped by editing — it changes only via "Use as training track"
//     (main.js) or the next page load (loadActiveTrack).
//   * Point coordinates are raw canvas pixels (1100×650); the slider and the
//     validation boundaries (|coord| ≤ 1e6, halfWidth ∈ [1, 2000], ≤ 512
//     points) match Track.fromJSON exactly — export → import is hash-identical.
//
// hooks = { onSaved(trackJSON), onStatus(text), confirmUseTrack() } — all
// optional; each is called defensively. confirmUseTrack() runs the whole
// "Use as training track" lineage-guard flow in main.js (fork-confirm when
// generations exist) and returns { ok, cancelled?, message } — the message is
// shown in #editor-status.

'use strict';

// Validation bounds — read from the ONE source of truth (js/track.js →
// Track.VALIDATION, a frozen object). No literal twins here; a missing
// track.js fails LOUDLY at load time (js/editor.js requires js/track.js —
// see the boot contract in index.html / js/main.js).
var _validation = (typeof globalThis !== 'undefined' && globalThis.Track && globalThis.Track.VALIDATION) || null;
if (!_validation) {
  throw new Error('js/track.js must load before js/editor.js — Track.VALIDATION is missing');
}
var EDITOR_MAX_POINTS = _validation.MAX_POINTS;
var EDITOR_MIN_POINTS = _validation.MIN_POINTS;
var EDITOR_HIT_RADIUS = 12;             // px — pointer grab radius
var EDITOR_CANVAS_W = 1100;
var EDITOR_CANVAS_H = 650;
var EDITOR_DEFAULT_HALF_WIDTH = 70;
var EDITOR_MIN_HALF_WIDTH = _validation.MIN_HALF_WIDTH;
var EDITOR_MAX_HALF_WIDTH = _validation.MAX_HALF_WIDTH;
var EDITOR_STORAGE_KEY = 'car_racer.track';
var EDITOR_EXPORT_NAME = 'car_racer-track.cartrack';
var EDITOR_MAX_IMPORT_BYTES = _validation.MAX_TRACK_JSON_LENGTH; // 256 KiB — matches Track's track_json cap

function deepCopyPoints(pts) {
  var out = new Array(pts.length);
  for (var i = 0; i < pts.length; i++) {
    out[i] = { x: pts[i].x, y: pts[i].y };
  }
  return out;
}

function editorErrMsg(err) { return String((err && err.message) || err); }

// Pointer event → canvas coordinate space (robust to CSS scaling of the canvas).
function eventPos(canvas, event) {
  if (!canvas) return null;
  var rect = canvas.getBoundingClientRect();
  if (!rect || rect.width <= 0 || rect.height <= 0) return null;
  var scaleX = canvas.width / rect.width;
  var scaleY = canvas.height / rect.height;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

var editor = {
  initialized: false,
  canvas: null,
  ctx: null,
  hooks: {},
  elements: null,
  points: [],
  halfWidth: EDITOR_DEFAULT_HALF_WIDTH,
  undo: null,               // single-level snapshot or null
  dragIndex: -1,            // point being dragged (gesture), -1 when idle
  hoverIndex: -1,           // point under the pointer (highlight), -1 when none
  selectedIndex: -1,        // last interacted point (highlight), -1 when none
  widthGestureArmed: false, // one undo snapshot per width-drag gesture
  dirty: true,              // road needs rebuilding
  cachedTrack: null,
};

function el(id) { return document.getElementById(id); }

function setStatus(text) {
  var msg = String(text);
  if (editor.elements && editor.elements.status) {
    editor.elements.status.textContent = msg;
  }
  if (editor.hooks && typeof editor.hooks.onStatus === 'function') {
    try { editor.hooks.onStatus(msg); } catch (err) { /* host hook must never break the editor */ }
  }
}

/* ---------- track building / rendering ---------- */

function buildTrack() {
  if (typeof globalThis.Track !== 'function') {
    throw new Error('Track is not loaded — js/track.js must load before js/editor.js');
  }
  return new globalThis.Track(deepCopyPoints(editor.points), editor.halfWidth);
}

// Builds (and caches) the road Track for rendering; returns null on failure.
function renderTrack() {
  if (!editor.dirty && editor.cachedTrack) return editor.cachedTrack;
  try {
    editor.cachedTrack = buildTrack();
    editor.dirty = false;
    return editor.cachedTrack;
  } catch (err) {
    editor.cachedTrack = null;
    setStatus('Cannot build track — ' + editorErrMsg(err));
    return null;
  }
}

function drawPoints(ctx) {
  ctx.save();
  ctx.font = 'bold 10px system-ui, -apple-system, "Segoe UI", Roboto, Ubuntu, Cantarell, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (var i = 0; i < editor.points.length; i++) {
    var p = editor.points[i];
    var selected = i === editor.selectedIndex;
    var hover = i === editor.hoverIndex;
    ctx.beginPath();
    ctx.arc(p.x, p.y, EDITOR_HIT_RADIUS - 1, 0, Math.PI * 2);
    ctx.fillStyle = selected ? '#ffb703' : (hover ? '#ffd60a' : '#4cc9f0');
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#111417';
    ctx.stroke();
    ctx.fillStyle = '#111417';
    ctx.fillText(String(i), p.x, p.y + 0.5);
  }
  ctx.restore();
}

function render() {
  var ctx = editor.ctx;
  if (!ctx) return;
  // The ONE deterministic grass pattern lives in js/track.js (shared with the
  // game canvas and spectate) — no duplicated pattern here.
  globalThis.Track.drawTrackBackground(ctx, EDITOR_CANVAS_W, EDITOR_CANVAS_H);
  var track = renderTrack();
  if (track) {
    try {
      track.draw(ctx);
    } catch (err) {
      setStatus('Cannot draw track — ' + editorErrMsg(err));
    }
  }
  drawPoints(ctx);
  updatePointsCount();
}

function updatePointsCount() {
  if (!editor.elements || !editor.elements.pointsCount) return;
  editor.elements.pointsCount.textContent = 'Points: ' + editor.points.length + ' / ' + EDITOR_MAX_POINTS;
}

function syncHalfWidthUi() {
  if (!editor.elements) return;
  editor.elements.halfWidth.value = String(editor.halfWidth);
  editor.elements.halfWidthValue.textContent = String(editor.halfWidth);
}

function updateUndoButton() {
  if (editor.elements && editor.elements.undo) {
    editor.elements.undo.disabled = !editor.undo;
  }
}

function clearSelection() {
  editor.dragIndex = -1;
  editor.hoverIndex = -1;
  editor.selectedIndex = -1;
}

/* ---------- persistence (localStorage) + host notification ---------- */

function trackJsonString() {
  var pts = new Array(editor.points.length);
  for (var i = 0; i < editor.points.length; i++) {
    pts[i] = { x: editor.points[i].x, y: editor.points[i].y };
  }
  return JSON.stringify({ points: pts, halfWidth: editor.halfWidth });
}

function trackHashShort() {
  try {
    var t = renderTrack(); // reuse the cache when it is fresh (dirty ⇒ rebuilt)
    if (t) return t.hash().slice(0, 8);
    return buildTrack().hash().slice(0, 8);
  } catch (err) {
    return null;
  }
}

// Writes the current editor state to localStorage and notifies the host.
// Returns { json, hashShort, storageOk }. The editor NEVER throws out of here.
function persist() {
  var json = trackJsonString();
  var hashShort = trackHashShort();
  var storageOk = true;
  try {
    if (globalThis.localStorage) {
      globalThis.localStorage.setItem(EDITOR_STORAGE_KEY, json);
    } else {
      storageOk = false;
    }
  } catch (err) {
    storageOk = false;
  }
  if (editor.hooks && typeof editor.hooks.onSaved === 'function') {
    try { editor.hooks.onSaved(json); } catch (err) { /* host hook must never break the editor */ }
  }
  return { json: json, hashShort: hashShort, storageOk: storageOk };
}

function announceSaved(res, prefix) {
  var msg = prefix || 'Saved';
  if (res.hashShort) msg += ' · h' + res.hashShort;
  if (!res.storageOk) msg += ' — browser storage blocked; track NOT cached for next reload';
  setStatus(msg);
}

/* ---------- undo (single level) ---------- */

function pushUndo() {
  editor.undo = { points: deepCopyPoints(editor.points), halfWidth: editor.halfWidth };
  updateUndoButton();
}

function undo() {
  if (!editor.undo) {
    setStatus('Nothing to undo');
    return;
  }
  editor.points = deepCopyPoints(editor.undo.points);
  editor.halfWidth = editor.undo.halfWidth;
  editor.undo = null;
  clearSelection();
  editor.dirty = true;
  syncHalfWidthUi();
  updateUndoButton();
  updatePointsCount();
  render();
  announceSaved(persist(), 'Undone');
}

/* ---------- structural ops ---------- */

function resetToDefaultState(clearUndo) {
  var def = globalThis.DEFAULT_CONTROL_POINTS;
  if (!def || !def.length) throw new Error('DEFAULT_CONTROL_POINTS is not loaded');
  editor.points = deepCopyPoints(def);
  editor.halfWidth = EDITOR_DEFAULT_HALF_WIDTH;
  editor.cachedTrack = null;
  editor.dirty = true;
  clearSelection();
  if (clearUndo) editor.undo = null;
  syncHalfWidthUi();
  updatePointsCount();
  updateUndoButton();
}

function resetToDefault() {
  pushUndo();
  try {
    resetToDefaultState(false);
  } catch (err) {
    setStatus('Reset failed — ' + editorErrMsg(err));
    return;
  }
  render();
  announceSaved(persist(), 'Reset to the default circuit');
}

function hitPoint(x, y) {
  var best = EDITOR_HIT_RADIUS * EDITOR_HIT_RADIUS;
  var idx = -1;
  for (var i = 0; i < editor.points.length; i++) {
    var dx = editor.points[i].x - x, dy = editor.points[i].y - y;
    var d2 = dx * dx + dy * dy;
    if (d2 <= best) { best = d2; idx = i; }
  }
  return idx;
}

// Insertion index for a NEW point: after the control-polygon edge (i, i+1)
// closest to (x, y). Points are pushed at the end when idx === points.length.
function bestInsertIndex(x, y) {
  var n = editor.points.length;
  var best = Infinity, idx = n;
  for (var i = 0; i < n; i++) {
    var a = editor.points[i], b = editor.points[(i + 1) % n];
    var d = globalThis.Track.segmentPointDistanceSquared(a.x, a.y, b.x, b.y, x, y);
    if (d < best) { best = d; idx = i + 1; }
  }
  return idx;
}

function addPoint(x, y) {
  if (editor.points.length >= EDITOR_MAX_POINTS) {
    setStatus('Maximum ' + EDITOR_MAX_POINTS + ' points — delete one first');
    return -1;
  }
  pushUndo();
  var at = bestInsertIndex(x, y);
  editor.points.splice(at, 0, { x: x, y: y });
  editor.dirty = true;
  updatePointsCount();
  render();
  announceSaved(persist(), 'Point ' + at + ' added');
  return at;
}

function deletePoint(i) {
  if (i < 0 || i >= editor.points.length) return;
  if (editor.points.length <= EDITOR_MIN_POINTS) {
    setStatus('Keep at least ' + EDITOR_MIN_POINTS + ' points — a closed loop needs three');
    return;
  }
  pushUndo();
  editor.points.splice(i, 1);
  clearSelection();
  editor.dirty = true;
  updatePointsCount();
  render();
  announceSaved(persist(), 'Point ' + i + ' deleted');
}

/* ---------- pointer handling ---------- */

function onPointerDown(event) {
  var pos = eventPos(editor.canvas, event);
  if (!pos) return;
  var hit = hitPoint(pos.x, pos.y);
  if (hit >= 0) {
    pushUndo(); // a move is a structural op — snapshot before it happens
    editor.dragIndex = hit;
    editor.selectedIndex = hit;
    editor.hoverIndex = hit;
  } else {
    var at = addPoint(pos.x, pos.y);
    if (at < 0) return;
    editor.dragIndex = at;   // keep dragging the freshly added point
    editor.selectedIndex = at;
    editor.hoverIndex = at;
  }
  try {
    editor.canvas.setPointerCapture(event.pointerId);
  } catch (err) { /* pointer capture is best-effort */ }
  if (event.cancelable) event.preventDefault();
  render();
}

function onPointerMove(event) {
  var pos = eventPos(editor.canvas, event);
  if (editor.dragIndex >= 0) {
    if (!pos) return;
    editor.points[editor.dragIndex].x = pos.x;
    editor.points[editor.dragIndex].y = pos.y;
    editor.dirty = true;
    if (event.cancelable) event.preventDefault();
    render(); // track cache is dirty — full road rebuild happens here
    return;
  }
  var hit = pos ? hitPoint(pos.x, pos.y) : -1;
  if (hit !== editor.hoverIndex) {
    editor.hoverIndex = hit;
    editor.canvas.style.cursor = hit >= 0 ? 'move' : 'crosshair';
    render(); // cached road — only the points overlay changes
  }
}

function onPointerEnd(event) {
  if (editor.dragIndex < 0) return;
  try {
    editor.canvas.releasePointerCapture(event.pointerId);
  } catch (err) { /* pointer capture is best-effort */ }
  editor.dragIndex = -1;
  render();
  announceSaved(persist(), 'Saved');
}

function onContextMenu(event) {
  if (event.cancelable) event.preventDefault(); // never show the menu over the canvas
  var pos = eventPos(editor.canvas, event);
  var hit = pos ? hitPoint(pos.x, pos.y) : -1;
  if (hit >= 0) deletePoint(hit);
  else setStatus('Right-click directly on a point to delete it');
}

/* ---------- width slider ---------- */

function onHalfWidthInput() {
  var v = Math.round(Number(editor.elements.halfWidth.value));
  if (!Number.isFinite(v)) return;
  v = Math.max(EDITOR_MIN_HALF_WIDTH, Math.min(EDITOR_MAX_HALF_WIDTH, v));
  if (!editor.widthGestureArmed) {
    pushUndo(); // one snapshot per gesture, not one per input tick
    editor.widthGestureArmed = true;
  }
  editor.halfWidth = v;
  editor.elements.halfWidthValue.textContent = String(v);
  editor.dirty = true;
  render();
  announceSaved(persist(), 'Width ' + v + ' px');
}

function onHalfWidthChange() {
  editor.widthGestureArmed = false; // gesture ends — next input snapshots afresh
}

/* ---------- export / import (.cartrack) ---------- */

function exportTrack() {
  var json = trackJsonString();
  try {
    var blob = new Blob([json], { type: 'application/json' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = EDITOR_EXPORT_NAME;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
    setStatus('Exported ' + EDITOR_EXPORT_NAME + ' · ' + editor.points.length + ' points');
  } catch (err) {
    setStatus('Export failed — ' + editorErrMsg(err));
  }
}

function readFileAsText(file) {
  if (file && typeof file.text === 'function') return file.text();
  return new Promise(function (resolve, reject) {
    var reader = new FileReader();
    reader.onload = function () { resolve(String(reader.result || '')); };
    reader.onerror = function () { reject(reader.error || new Error('could not read the file')); };
    reader.readAsText(file);
  });
}

function onImportFileChange(event) {
  var input = event.target;
  var file = input.files && input.files[0];
  input.value = ''; // allow re-picking the same file later
  if (!file) return;
  if (file.size > EDITOR_MAX_IMPORT_BYTES) {
    setStatus('Import refused — file is ' + formatBytes(file.size) + '; the cap is ' + formatBytes(EDITOR_MAX_IMPORT_BYTES));
    return;
  }
  readFileAsText(file).then(function (text) {
    var track;
    try {
      track = globalThis.Track.fromJSON(text); // full battery: ≤256 KiB, ≤512 pts, finite coords, halfWidth ∈ [1, 2000], closed loop
    } catch (err) {
      setStatus('Import refused — ' + editorErrMsg(err) + ' — editor unchanged');
      return;
    }
    pushUndo();
    editor.points = deepCopyPoints(track.controlPoints);
    editor.halfWidth = track.halfWidth;
    editor.cachedTrack = null;
    editor.dirty = true;
    clearSelection();
    syncHalfWidthUi();
    updatePointsCount();
    render();
    announceSaved(persist(), 'Imported · ' + editor.points.length + ' points');
  }, function (err) {
    setStatus('Import refused — could not read the file: ' + editorErrMsg(err) + ' — editor unchanged');
  });
}

/* ---------- "Use as training track" ---------- */

function onUseTrack() {
  if (editor.hooks && typeof editor.hooks.confirmUseTrack === 'function') {
    var result = editor.hooks.confirmUseTrack();
    if (result && typeof result.message === 'string' && result.message.length > 0) {
      setStatus(result.message);
      return;
    }
    setStatus(result && result.ok ? 'Track set as the training track' : 'Use as training track — no change');
    return;
  }
  setStatus('“Use as training track” is unavailable — the host did not wire the lineage-guard hook');
}

/* ---------- init / open / close / api ---------- */

function requireElement(id) {
  var node = el(id);
  if (!node) throw new Error('Editor.init: missing required element #' + id);
  return node;
}

function init(canvasEl, hooks) {
  if (editor.initialized) return api;
  var canvas = canvasEl || el('editor-canvas');
  if (!canvas || typeof canvas.getContext !== 'function') {
    throw new Error('Editor.init: canvas#editor-canvas missing or not a canvas element');
  }
  var ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Editor.init: the 2d canvas context is unavailable');
  if (typeof globalThis.Track !== 'function' || typeof globalThis.DEFAULT_CONTROL_POINTS === 'undefined') {
    throw new Error('Editor.init: Track / DEFAULT_CONTROL_POINTS are not loaded — js/track.js must load before js/editor.js');
  }

  editor.hooks = hooks || {};
  editor.canvas = canvas;
  editor.ctx = ctx;
  editor.elements = {
    pointsCount: requireElement('editor-points-count'),
    halfWidth: requireElement('editor-halfwidth'),
    halfWidthValue: requireElement('editor-halfwidth-value'),
    undo: requireElement('editor-undo'),
    reset: requireElement('btn-editor-reset'),
    exportBtn: requireElement('btn-editor-export'),
    importBtn: requireElement('btn-editor-import'),
    importInput: requireElement('import-track-input'),
    useTrack: requireElement('btn-editor-use-track'),
    status: requireElement('editor-status'),
  };

  // Events — attached exactly once per init.
  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerEnd);
  canvas.addEventListener('pointercancel', onPointerEnd);
  canvas.addEventListener('contextmenu', onContextMenu);
  editor.elements.halfWidth.addEventListener('input', onHalfWidthInput);
  editor.elements.halfWidth.addEventListener('change', onHalfWidthChange);
  editor.elements.undo.addEventListener('click', undo);
  editor.elements.reset.addEventListener('click', resetToDefault);
  editor.elements.exportBtn.addEventListener('click', exportTrack);
  editor.elements.importBtn.addEventListener('click', function () { editor.elements.importInput.click(); });
  editor.elements.importInput.addEventListener('change', onImportFileChange);
  editor.elements.useTrack.addEventListener('click', onUseTrack);
  canvas.style.cursor = 'crosshair';

  editor.initialized = true;
  load(); // seeds editor state from localStorage (or the default circuit) and renders
  return api;
}

function open() {
  if (!editor.initialized) return;
  render();
}

function close() {
  // Leaving the tab mid-drag: end the gesture cleanly without persisting a
  // partial drag (the drag's save happens in onPointerEnd / on tab switch is
  // not guaranteed). Rendering is idempotent, so nothing else is needed.
  editor.dragIndex = -1;
}

// Returns a FRESH Track built from the current editor state, or null on failure.
function getTrack() {
  if (editor.points.length < EDITOR_MIN_POINTS) return null;
  try {
    return buildTrack();
  } catch (err) {
    return null;
  }
}

// Loads the stored track (validated via Track.fromJSON) into the editor; falls
// back to the default circuit on missing/invalid storage. Returns true when a
// stored track was adopted, false otherwise.
function load() {
  var raw = null;
  try {
    raw = globalThis.localStorage ? globalThis.localStorage.getItem(EDITOR_STORAGE_KEY) : null;
  } catch (err) {
    raw = null; // storage blocked — fall back to the default circuit
  }
  if (raw) {
    try {
      var track = globalThis.Track.fromJSON(raw);
      editor.points = deepCopyPoints(track.controlPoints);
      editor.halfWidth = track.halfWidth;
      editor.undo = null;
      editor.cachedTrack = null;
      editor.dirty = true;
      clearSelection();
      syncHalfWidthUi();
      updateUndoButton();
      updatePointsCount();
      render();
      setStatus('Restored from browser storage · ' + editor.points.length + ' points · h' + track.hash().slice(0, 8));
      return true;
    } catch (err) {
      try {
        resetToDefaultState(true);
        render();
      } catch (resetErr) { /* the default circuit is a frozen constant — this cannot fail */ }
      setStatus('Stored track was invalid (' + editorErrMsg(err) + ') — showing the default circuit');
      return false;
    }
  }
  try {
    resetToDefaultState(true);
    render();
    setStatus('Default circuit loaded — click empty space to add a point, drag to move, right-click to delete');
  } catch (err) {
    editor.points = [];
    setStatus('Editor failed to load the default circuit — ' + editorErrMsg(err));
  }
  return false;
}

var api = {
  init: init,
  open: open,
  close: close,
  getTrack: getTrack,
};

if (typeof globalThis !== 'undefined') {
  globalThis.Editor = api;
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = api;
}
