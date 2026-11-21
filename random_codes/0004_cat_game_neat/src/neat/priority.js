/**
 * priority.js — OFFLINE critical-chain prioritisation (main-thread diagnostics
 * and batch-cap allocation input only).
 *
 * `computeCriticality(items, targets)` returns a null-prototype map of
 *   { [validated craftable id]: number in [0,1] }
 * where a higher value means the item lies deeper on the dependency chain that
 * feeds the targets (the "critical chain" ending in artifact/elementstone).
 *
 * It is PURE and DOM-free. It is never put on a worker message, never written
 * into telemetry, and never used to throttle coins at runtime: all 19 items
 * feed target paths and none may be starved. The planner uses it offline to
 * give critical-chain items larger batch caps (fewer batches, less time) and
 * less-critical items smaller caps (cheaper per unit).
 */
import { topoOrder } from '../sim/demand.js';

const ID_RE = /^[a-z][a-z0-9_]{0,39}$/;
const isObj = (v) => !!v && typeof v === 'object' && !Array.isArray(v);
const validId = (id) => typeof id === 'string' && ID_RE.test(id);

/**
 * Longest-path depth from a base item over the craftable recipe DAG, computed
 * by topological relaxation. Only craftable ingredients contribute (bases are
 * depth 0 and never propagate). Returns a normal Map keyed by craftable id.
 */
function computeDepths(items, order, byId) {
  const depth = new Map();
  // A craftable that consumes only base items is depth 1 (bases are depth 0).
  for (const id of order) depth.set(id, 1);
  for (const id of order) {
    const item = byId.get(id);
    let best = 1;
    for (const ing of (item && item.ingredients) || []) {
      const src = byId.get(ing.item);
      if (!src || src.type !== 'craftable' || !depth.has(ing.item)) continue;
      best = Math.max(best, depth.get(ing.item) + 1);
    }
    depth.set(id, best);
  }
  return depth;
}

/**
 * Set of craftable ids that feed (or are) a target, found by a reverse BFS
 * over craftable ingredient edges starting at the target ids.
 */
function targetAncestors(order, byId, targets) {
  const onPath = new Set();
  for (const id of Object.keys(targets || {})) {
    if (byId.has(id) && byId.get(id).type === 'craftable' && order.includes(id)) onPath.add(id);
  }
  // Reverse relaxation: if a product is on a target path, all its craftable
  // ingredients are too. Iterate to a fixed point (order is topological).
  let changed = true;
  while (changed) {
    changed = false;
    for (const id of order) {
      if (!onPath.has(id)) continue;
      const item = byId.get(id);
      for (const ing of (item && item.ingredients) || []) {
        const src = byId.get(ing.item);
        if (src && src.type === 'craftable' && order.includes(ing.item) && !onPath.has(ing.item)) {
          onPath.add(ing.item);
          changed = true;
        }
      }
    }
  }
  return onPath;
}

/**
 * computeCriticality(items, targets) -> null-prototype { id: number in [0,1] }.
 * Invalid ids (including __proto__/constructor) are ignored. Values are finite.
 */
export function computeCriticality(items, targets) {
  const out = Object.create(null);
  const list = Array.isArray(items) ? items : [];
  if (list.length === 0) return out;

  const byId = new Map();
  for (const item of list) {
    if (isObj(item) && validId(item.id) && !byId.has(item.id)) byId.set(item.id, item);
  }
  let order;
  try { order = topoOrder(list); } catch (_e) { return out; }
  order = order.filter((id) => byId.has(id) && byId.get(id).type === 'craftable');
  if (order.length === 0) return out;

  const depth = computeDepths(list, order, byId);
  const onPath = targetAncestors(order, byId, isObj(targets) ? targets : {});
  const maxDepth = Math.max(1, ...order.map((id) => depth.get(id) || 0));

  for (const id of order) {
    if (!validId(id)) continue; // validated ids only
    const normalised = onPath.has(id) ? (depth.get(id) || 0) / maxDepth : 0;
    const value = Number.isFinite(normalised) ? Math.min(1, Math.max(0, normalised)) : 0;
    out[id] = value;
  }
  return out;
}

export default computeCriticality;
