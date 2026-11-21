/**
 * initial-state.js — the editable "Initial state" panel.
 *
 * The panel is PURE over an injected config snapshot: it never reads module
 * state from config.js/main.js. Prefill uses `prefillInitialState(snapshot)`;
 * reads use strict integer coercion with a prefill fallback and a single
 * stable CONFIG_INPUT_INVALID code (never raw input) for rejected values.
 *
 * DOM rules: createElement/textContent/setAttribute only — no innerHTML and no
 * inline handlers. Every emitted string passes through capString(...,120).
 * The null-prototype maps produced here are assigned BY REFERENCE downstream
 * (config.js::applyOverrides); validate.js remains the authoritative gate.
 */
import { prefillInitialState } from '../config.js';
import { KNOWN_TARGET_IDS, KNOWN_CRAFTABLE_IDS, ERROR_CODES, validateInitialState } from '../validate.js';
import { intInRange, capString } from './format.js';

/** UI coercion bounds (mirror the authoritative schema in validate.js). */
const MAX_TARGET = 1e7;
const MAX_INITIAL_COINS = 1e9;
const MAX_INITIAL_STASH = 1e9;

export const INITIAL_STATE_PANEL_ID = 'initial-state-panel';
export const INITIAL_COINS_ID = 'initial-coins';
export const INITIAL_APPLY_ID = 'initial-apply';
export const INITIAL_RESET_ID = 'initial-reset';
export const INITIAL_ZERO_ID = 'initial-zero';

export function initialTargetId(id) { return 'initial-target-' + id; }
export function initialStashId(id) { return 'initial-stash-' + id; }

function defaultDoc() {
  return typeof document !== 'undefined' ? document : null;
}

function isBoundedInt(v, min, max) {
  return typeof v === 'number' && Number.isFinite(v) && Number.isInteger(v) && v >= min && v <= max;
}

/**
 * Strict input coercion. Rejects empty/whitespace, non-numeric, fractional,
 * negative, over-cap and junk-suffixed strings; `onFail` flags the stable log.
 * `intInRange` does the numeric parse, `isBoundedInt` re-checks the result.
 */
function coerceCount(raw, fallback, min, max, onFail) {
  const s = raw === null || raw === undefined ? '' : String(raw).trim();
  if (s === '' || !/^\d+$/.test(s)) { onFail(); return fallback; }
  const n = intInRange(s, fallback, min, max);
  if (!isBoundedInt(n, min, max)) { onFail(); return fallback; }
  return n;
}

function readRaw(doc, id, onFail) {
  const el = doc && typeof doc.getElementById === 'function' ? doc.getElementById(id) : null;
  if (!el) { onFail(); return null; }
  return el.value;
}

/**
 * Read the panel into a validated override set: {targets, initialCoins,
 * initialStash} with null-prototype maps keyed only by the 11 target ids and
 * 19 craftable ids. Rejected values fall back to the prefill value; `onInvalid`
 * is called AT MOST ONCE with ERROR_CODES.CONFIG_INPUT_INVALID (no raw input).
 */
export function readInitialState(config = {}, doc = defaultDoc(), onInvalid = null) {
  const prefill = prefillInitialState(config);
  const targets = Object.create(null);
  const initialStash = Object.create(null);
  let invalid = false;
  const onFail = () => { invalid = true; };

  for (const id of KNOWN_TARGET_IDS) {
    const raw = readRaw(doc, initialTargetId(id), onFail);
    targets[id] = coerceCount(raw, prefill.targets[id], 0, MAX_TARGET, onFail);
  }
  for (const id of KNOWN_CRAFTABLE_IDS) {
    const raw = readRaw(doc, initialStashId(id), onFail);
    initialStash[id] = coerceCount(raw, prefill.initialStash[id], 0, MAX_INITIAL_STASH, onFail);
  }
  const initialCoins = coerceCount(
    readRaw(doc, INITIAL_COINS_ID, onFail), prefill.initialCoins,
    0, MAX_INITIAL_COINS, onFail
  );

  const out = { targets, initialCoins, initialStash };
  // Authoritative gate before the set leaves this module. A hostile injected
  // snapshot can never smuggle an out-of-range value past validate.js.
  let validationFailed = false;
  try {
    validateInitialState(out);
  } catch (_e) {
    validationFailed = true;
  }
  const code = ERROR_CODES.CONFIG_INPUT_INVALID || 'CONFIG_INPUT_INVALID';
  if (invalid && typeof onInvalid === 'function') onInvalid(code);
  if (validationFailed) return prefillInitialState({}); // all-zero, always valid
  return out; // per-field fallbacks preserve the valid prefill values
}

function writeValue(doc, id, value) {
  const el = doc && typeof doc.getElementById === 'function' ? doc.getElementById(id) : null;
  if (!el) return;
  const text = capString(value === null || value === undefined ? 0 : value);
  if (typeof el.setAttribute === 'function') el.setAttribute('value', text);
  el.value = text;
}

/**
 * Create the panel over an injected config snapshot. `prefill(next)` rewrites
 * all inputs from that snapshot; `read(onInvalid)` reads the live inputs.
 * Apply / Reset / Reset-to-0 buttons are wired here (exactly one listener each)
 * and delegate to the injected callbacks — no button wiring in main.js.
 */
export function createInitialStatePanel(config = {}, callbacks = {}, doc = defaultDoc()) {
  let snapshot = config || {};

  function prefill(next = snapshot) {
    snapshot = next || {};
    const data = prefillInitialState(snapshot);
    writeValue(doc, INITIAL_COINS_ID, data.initialCoins);
    for (const id of KNOWN_TARGET_IDS) writeValue(doc, initialTargetId(id), data.targets[id]);
    for (const id of KNOWN_CRAFTABLE_IDS) writeValue(doc, initialStashId(id), data.initialStash[id]);
    return data;
  }

  function read(onInvalid) {
    return readInitialState(snapshot, doc, onInvalid);
  }

  function bind(id, handler) {
    const el = doc && typeof doc.getElementById === 'function' ? doc.getElementById(id) : null;
    if (el && typeof el.addEventListener === 'function') el.addEventListener('click', handler);
  }

  bind(INITIAL_APPLY_ID, () => {
    if (typeof callbacks.onApply === 'function') callbacks.onApply(read(callbacks.onInvalid));
  });
  bind(INITIAL_RESET_ID, () => {
    if (typeof callbacks.onReset === 'function') callbacks.onReset();
  });
  // Fill-only: zero coins + starter stash and stop. This button must NOT start
  // the run; Apply remains the sole run trigger. Targets are untouched.
  bind(INITIAL_ZERO_ID, () => {
    writeValue(doc, INITIAL_COINS_ID, 0);
    for (const id of KNOWN_CRAFTABLE_IDS) writeValue(doc, initialStashId(id), 0);
  });

  prefill(snapshot);
  return { prefill, read, getSnapshot: () => snapshot };
}
