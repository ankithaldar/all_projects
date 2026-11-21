/**
 * batch-quantizer.js — frozen, pure batch-quantum helpers (plan Amendment).
 *
 * No DOM, no globals beyond Math, no console.*. Every returned map is
 * null-prototype and keyed only by validated craftable ids, so inherited keys
 * (__proto__/constructor) can never masquerade as items.
 *
 * Exports (exact frozen contracts):
 *   computeBatchQuantums(items)            -> { [craftable id]: step }
 *                                             step = gcd of all downstream
 *                                             per-unit ingredient quantities;
 *                                             default 1 for a terminal item.
 *   effectiveCap(plannerCap, step, alignCap) -> single cap authority used by
 *                                             BOTH the planner and the engine.
 *   quantizeBatch(requested, step, remainingCap, batchCap, aligned)
 *                                           -> exact batch size or 0 (reject).
 */
import { MAX_BATCH } from '../config.js';

const ID_RE = /^[a-z][a-z0-9_]{0,39}$/;
const validId = (id) => typeof id === 'string' && ID_RE.test(id);
const isObj = (v) => !!v && typeof v === 'object' && !Array.isArray(v);

/** Finite integer clamped to [min,max]; non-finite / garbage -> fallback. */
function finiteInt(value, fallback, min, max) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, Math.floor(n)));
}

const gcd = (a, b) => (b ? gcd(b, a % b) : a);

/**
 * computeBatchQuantums — one step per validated craftable id.
 * The step is the gcd of every per-unit quantity with which that item is
 * consumed by a downstream recipe. Terminal items default to 1.
 */
export function computeBatchQuantums(items) {
  const list = Array.isArray(items) ? items : [];
  const byId = new Map();
  for (const item of list) {
    if (isObj(item) && validId(item.id) && item.type === 'craftable' && !byId.has(item.id)) {
      byId.set(item.id, item);
    }
  }

  // Collect downstream usages. `uses` is null-prototype and keyed only by
  // validated craftable ids, so a synthetic __proto__/constructor ingredient
  // can neither pollute Object.prototype nor appear in the output.
  const uses = Object.create(null);
  for (const item of byId.values()) {
    const ingredients = Array.isArray(item.ingredients) ? item.ingredients : [];
    for (const ing of ingredients) {
      if (!isObj(ing) || !validId(ing.item)) continue;
      if (!byId.has(ing.item)) continue; // only craftable feeders have a quantum
      const q = Number(ing.quantity);
      if (!Number.isFinite(q) || q < 1) continue;
      if (!uses[ing.item]) uses[ing.item] = [];
      uses[ing.item].push(Math.floor(q));
    }
  }

  const out = Object.create(null);
  for (const id of byId.keys()) {
    const arr = uses[id];
    out[id] = arr && arr.length ? arr.reduce((a, b) => gcd(a, b), arr[0]) : 1;
  }
  return out;
}

/** Round `cap` down to a positive multiple of `step` (floor at one step). */
function snapDown(cap, step) {
  if (step <= 1) return cap;
  return Math.max(step, Math.floor(cap / step) * step);
}

/**
 * effectiveCap — THE single cap authority. Planner and engine both call it.
 *
 *   alignCap > 0  -> min(plannerCap, alignCap)   (quantum rule relaxed)
 *   alignCap = 0  -> snapDown(plannerCap, step)  (positive step multiple)
 * result clamped to [1, MAX_BATCH].
 */
export function effectiveCap(plannerCap, step, alignCap) {
  const s = finiteInt(step, 1, 1, 1e9);
  const a = finiteInt(alignCap, 0, 0, 1e9);
  const requested = finiteInt(plannerCap, 1, 1, MAX_BATCH);
  const value = a > 0 ? Math.min(requested, a) : snapDown(requested, s);
  return Math.min(MAX_BATCH, Math.max(1, value));
}

/**
 * quantizeBatch — exact batch size for one accepted clip.
 *
 *   requested <= 0                 -> 0
 *   remainingCap < step            -> min(requested, remainingCap, batchCap)
 *                                     (exact final remainder; never snap up)
 *   aligned                        -> min(requested, remainingCap, batchCap)
 *                                     (per-product split sizes)
 *   n >= step                      -> max(step, floor(n/step)*step)
 *   0 < n < step                   -> 0 (REJECT / BELOW_QUANTUM; never snap up)
 *
 * `n` is min(requested, remainingCap, batchCap). The result never exceeds
 * remainingCap and is always a non-negative integer.
 */
export function quantizeBatch(requested, step, remainingCap, batchCap, aligned) {
  let req = Number(requested);
  if (!Number.isFinite(req)) req = 0;
  req = Math.floor(req);

  const s = finiteInt(step, 1, 1, 1e9);
  let rem = Number(remainingCap);
  rem = Number.isFinite(rem) && rem > 0 ? Math.floor(rem) : 0;
  const cap = finiteInt(batchCap, MAX_BATCH, 1, MAX_BATCH);
  const isAligned = aligned === true;

  if (req <= 0) return 0;
  const n = Math.min(req, rem, cap);
  if (n <= 0) return 0;

  // Exact final remainder (also for aligned items): never snap a sub-quantum
  // remainder up into an unaffordable / over-cap batch.
  if (rem < s) return Math.min(n, rem, cap);
  if (isAligned) return n;

  if (n >= s) {
    // floor(n/step)*step <= n <= rem, and step <= rem here, so this cannot
    // exceed remainingCap.
    return Math.min(rem, Math.max(s, Math.floor(n / s) * s));
  }
  // 0 < n < step with a full quantum still remaining: REJECT, never snap up.
  return 0;
}

export default { computeBatchQuantums, effectiveCap, quantizeBatch };
