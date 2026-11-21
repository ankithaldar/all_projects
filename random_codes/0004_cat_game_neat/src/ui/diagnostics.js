/**
 * diagnostics.js — main-thread-only solver diagnostics view.
 *
 * Security contract (plan section 5 / acceptance criteria):
 *  - every node is built with createElement + textContent (never innerHTML,
 *    never insertAdjacentHTML);
 *  - all dynamic text passes through capString(..., MAX_STR = 120);
 *  - maps are keyed by a validated craftable-id allowlist, so unknown keys
 *    (including __proto__ / constructor and injected markup) are ignored;
 *  - every map is capped at DIAGNOSTICS_MAP_CAP entries;
 *  - the maps are null-prototype DTOs, so they cannot pollute Object.prototype.
 *
 * This module is pure presentation: no Worker, no fetch of its own, no globals
 * mutated. Diagnostics are computed on the main thread by main.js and handed
 * to render() as a plain object.
 */
import { capString, MAX_STR } from './format.js';
import { CRAFTABLE_IDS } from '../sim/items.js';

/** Hard bound on rendered map entries (plan: maps capped at 23). */
export const DIAGNOSTICS_MAP_CAP = 23;

/** Validated craftable ids only — the single allowlist for rendered keys. */
const KNOWN_IDS = new Set(CRAFTABLE_IDS);
const SECTIONS = ['criticality', 'caps'];

/** Return a bounded, null-prototype, allowlist-filtered copy of a diagnostics object. */
export function sanitizeDiagnostics(diag) {
  const out = Object.create(null);
  const src = (diag && typeof diag === 'object') ? diag : {};

  for (const section of SECTIONS) {
    const dst = Object.create(null);
    const map = src[section];
    if (map && typeof map === 'object') {
      let count = 0;
      for (const id of Object.keys(map)) {
        if (count >= DIAGNOSTICS_MAP_CAP) break;
        if (!KNOWN_IDS.has(id)) continue; // ignore forbidden / unknown keys
        const n = Number(map[id]);
        dst[id] = Number.isFinite(n) ? n : 0;
        count += 1;
      }
    }
    out[section] = dst;
  }

  const ticks = Number(src.estimatedTicks);
  const cost = Number(src.estimatedCost);
  out.estimatedTicks = Number.isFinite(ticks) ? ticks : 0;
  out.estimatedCost = Number.isFinite(cost) ? cost : 0;
  out.label = capString(src.label == null ? '' : src.label, MAX_STR);
  return out;
}

/**
 * createDiagnostics(el, doc?) -> { render(diag) }
 * `doc` is injectable so the view is testable outside a browser.
 */
export function createDiagnostics(el, doc = (typeof document !== 'undefined' ? document : null)) {
  function node(tag, text) {
    const n = doc.createElement(tag);
    if (text != null) n.textContent = capString(text, MAX_STR);
    return n;
  }

  function render(diag) {
    if (!el || !doc) return;
    const d = sanitizeDiagnostics(diag);
    // Clear by textContent (never innerHTML), then rebuild from bounded state.
    el.textContent = '';
    el.appendChild(node('div', 'Solver diagnostics (main thread)'));

    for (const section of SECTIONS) {
      const map = d[section];
      for (const id of Object.keys(map)) {
        el.appendChild(node('div', `${section}: ${id} = ${map[id]}`));
      }
    }

    el.appendChild(node('div',
      `estTicks=${d.estimatedTicks} estCost=${d.estimatedCost} ${d.label}`));
  }

  return { render };
}

export default createDiagnostics;
