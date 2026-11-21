/**
 * items.js — static, canonical item/DAG metadata derived from the embedded
 * config so every worker sees the same ids, tiers, recipes and topo order.
 *
 * NOTE: the YAML `tier` field is inconsistent with real dependencies (14 of
 * 19 craftables declare a tier below their true dependency depth). We keep
 * `tier` as-is for metadata/display fidelity, and compute `depth` — the
 * longest-path distance from a base item — from the recipe graph (DAG_EDGES).
 * `effectiveTier(id)` / `DEPTH_BY_ID` expose that corrected depth.
 */
import { EMBEDDED_CONFIG, MAX_BATCH, craftTicksForMinutes } from '../config.js';
import { topoOrder } from './demand.js';

export { MAX_BATCH };

/** Untouched source records (never mutated — config parity tests rely on it). */
const RAW_ITEMS = EMBEDDED_CONFIG.items;

/** ingredient -> product edges with quantity. */
export const DAG_EDGES = RAW_ITEMS.filter((i) => i.type === 'craftable').flatMap((i) =>
  i.ingredients.map((ing) => ({ from: ing.item, to: i.id, quantity: ing.quantity }))
);

/**
 * Longest-path depth from a base item (bases = 0), derived strictly from
 * DAG_EDGES. Computed over a topological relaxation so a node's depth is
 * max(depth(ingredients)) + 1.
 */
function computeDepths(items, edges) {
  const depth = {};
  for (const it of items) depth[it.id] = 0; // bases stay at 0

  const outgoing = new Map(); // from -> [to,...]
  const indeg = new Map();
  for (const it of items) indeg.set(it.id, 0);
  for (const e of edges) {
    if (!outgoing.has(e.from)) outgoing.set(e.from, []);
    outgoing.get(e.from).push(e.to);
    indeg.set(e.to, (indeg.get(e.to) || 0) + 1);
  }

  const index = new Map(items.map((it, i) => [it.id, i]));
  const sortReady = (arr) => arr.sort((a, b) => index.get(a) - index.get(b));
  const ready = sortReady(items.filter((it) => indeg.get(it.id) === 0).map((it) => it.id));
  let visited = 0;
  while (ready.length) {
    const id = ready.shift();
    visited += 1;
    for (const to of outgoing.get(id) || []) {
      depth[to] = Math.max(depth[to], depth[id] + 1);
      indeg.set(to, indeg.get(to) - 1);
      if (indeg.get(to) === 0) ready.push(to);
    }
    sortReady(ready);
  }
  if (visited !== items.length) throw new Error('DAG_CYCLE');
  return depth;
}

export const DEPTH_BY_ID = Object.freeze(computeDepths(RAW_ITEMS, DAG_EDGES));

/** Corrected dependency depth for an item id (undefined for unknown ids). */
export function effectiveTier(id) {
  return DEPTH_BY_ID[id];
}

/**
 * {id,name,tier,depth,type,ingredients:[{item,quantity}],coinCost,craftTimeMinutes}
 * `tier` is the raw YAML metadata; `depth` is the graph-derived effective tier.
 */
export const ITEMS = RAW_ITEMS.map((it) => ({ ...it, depth: DEPTH_BY_ID[it.id] }));

const BASE_IDS = ITEMS.filter((i) => i.type === 'base').map((i) => i.id);

/** Craftable ids in dependency-respecting topological order (19 nets). */
export const CRAFTABLE_IDS = topoOrder(ITEMS);

/** All item ids: bases (tier 0) first, then craftables in topo order. */
export const TOPO_ORDER = [...BASE_IDS, ...CRAFTABLE_IDS];

const BY_ID = new Map(ITEMS.map((i) => [i.id, i]));

export function getRecipe(id) {
  return BY_ID.get(id) || null;
}

/** ceil(craftTimeMinutes / 5), clamped to >= 1 — verified for 5 -> 1, 4320 -> 864. */
export function craftTicks(id) {
  const item = BY_ID.get(id);
  if (!item) return 0;
  return craftTicksForMinutes(item.craftTimeMinutes);
}
