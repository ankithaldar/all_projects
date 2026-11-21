/**
 * policy-caps.js — SINGLE shared implementation of per-item batch capping.
 *
 * Imported by BOTH the training controller (src/neat/controller.js) and the
 * main-thread display runner (src/main.js), so training and display apply the
 * exact same batch limits. No duplicate implementation exists elsewhere.
 *
 * The output map is null-prototype and keyed only by validated craftable ids,
 * so inherited keys (__proto__/constructor) can never masquerade as items.
 * Capping is applied at the POLICY layer only: the engine's own material/coin/
 * MAX_BATCH clipping is untouched, and no coin throttling happens at runtime.
 */
import { MAX_BATCH } from '../config.js';
import { CRAFTABLE_IDS } from '../sim/items.js';

const KNOWN_IDS = new Set(CRAFTABLE_IDS);
const ID_RE = /^[a-z][a-z0-9_]{0,39}$/;

/** True only for a known craftable id that also passes the allowlist. */
export function isKnownCraftableId(id) {
  return typeof id === 'string' && ID_RE.test(id) && KNOWN_IDS.has(id);
}

/** Clamp a cap value to an integer in [1, MAX_BATCH]; invalid -> MAX_BATCH. */
export function clampCap(value, maxBatch = MAX_BATCH) {
  const limitRaw = Number(maxBatch);
  const limit = Number.isFinite(limitRaw) && limitRaw >= 1
    ? Math.min(MAX_BATCH, Math.floor(limitRaw)) : MAX_BATCH;
  const n = Number(value);
  if (!Number.isFinite(n)) return limit;
  const c = Math.floor(n);
  if (c < 1) return 1;
  return Math.min(limit, c);
}

/**
 * capBatch(batch, cap) -> integer batch in [0, cap].
 * Non-positive / non-finite requests become 0 (the engine treats 0 as "no
 * craft this tick"); positive requests are clamped to the item cap.
 */
export function capBatch(batch, cap) {
  const limit = clampCap(cap);
  const n = Math.floor(Number(batch));
  if (!Number.isFinite(n) || n <= 0) return 0;
  return Math.min(limit, n);
}

/**
 * applyBatchCaps(policies, caps) -> null-prototype map of
 *   { [validCraftableId]: (context) => integerBatch }
 *
 * Each returned policy calls the original policy and clamps its batch to the
 * item's cap. Unknown ids and non-function policies are dropped. The helper is
 * idempotent (applying it twice yields the same batches).
 */
export function applyBatchCaps(policies, caps) {
  const out = Object.create(null);
  if (!policies || typeof policies !== 'object') return out;
  const capMap = (caps && typeof caps === 'object') ? caps : {};
  for (const id of Object.keys(policies)) {
    if (!isKnownCraftableId(id)) continue;
    const fn = policies[id];
    if (typeof fn !== 'function') continue;
    const cap = clampCap(capMap[id]);
    out[id] = (context) => capBatch(fn(context), cap);
  }
  return out;
}

export default applyBatchCaps;
