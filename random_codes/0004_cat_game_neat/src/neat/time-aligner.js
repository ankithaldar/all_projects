/**
 * time-aligner.js — frozen, pure per-product lead-time alignment (plan Locked 4/5).
 *
 * No DOM, no globals beyond Math, no console.*. All maps are null-prototype
 * and keyed only by validated craftable ids.
 *
 * Alignment is computed PER PRODUCT UNIT from the per-product ingredient
 * quantity q (never from total mfr). For a product P with craftable
 * ingredients i:
 *   T_i_product = lcm(craftTicks(i)...)      (over P's craftable ingredients)
 *   k_i         = T / craftTicks(i)
 *   feasible_i  = k_i <= q_i
 *                 && k_i * craftTicks(i) <= MAX_TICKS
 *                 && ceil(q_i / k_i) <= MAX_BATCH
 * If ANY ingredient of P fails, the whole recipe is ALIGNMENT_INFEASIBLE and
 * falls back to k=1 (alignCap=0). An item feeding several recipes takes the
 * MINIMUM alignCap among its feasible k>1 recipes; natural k=1 and
 * infeasible/terminal items get alignCap=0.
 */
import { MAX_BATCH, deriveConstants, craftTicksForMinutes } from '../config.js';
import { topoOrder } from '../sim/demand.js';

const { MAX_TICKS } = deriveConstants();
const ID_RE = /^[a-z][a-z0-9_]{0,39}$/;
const validId = (id) => typeof id === 'string' && ID_RE.test(id);
const isObj = (v) => !!v && typeof v === 'object' && !Array.isArray(v);

/** Enum reported for infeasible recipes (and reused by planner relaxation). */
export const ALIGNMENT_STATUS = Object.freeze({
  FEASIBLE: 'ALIGNMENT_FEASIBLE',
  INFEASIBLE: 'ALIGNMENT_INFEASIBLE',
  RELAXED: 'ALIGNMENT_RELAXED'
});

function finiteInt(value, fallback, min, max) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, Math.floor(n)));
}

const gcd = (a, b) => (b ? gcd(b, a % b) : a);
const lcm = (a, b) => (a / gcd(a, b)) * b;

/**
 * computeAlignment(items, mfr, quantums) -> { alignments, infeasibleRecipes }
 *
 * `mfr` is accepted for interface parity with the plan (feasibility is driven
 * by per-product quantities q, NOT by mfr, so it is intentionally unused).
 * `quantums` is likewise accepted for parity; q always comes from the recipe.
 */
export function computeAlignment(items, mfr, quantums) {
  const list = Array.isArray(items) ? items : [];
  const byId = new Map();
  for (const item of list) {
    if (isObj(item) && validId(item.id) && item.type === 'craftable' && !byId.has(item.id)) {
      byId.set(item.id, item);
    }
  }
  let order;
  try { order = topoOrder(list); } catch (_e) { order = []; }
  order = order.filter((id) => byId.has(id) && validId(id));

  const alignments = Object.create(null);
  const infeasibleRecipes = Object.create(null);
  // Every validated craftable id gets a well-defined entry. alignCap=0 is the
  // sentinel for "no active alignment" (natural k=1, infeasible or terminal).
  for (const id of order) alignments[id] = { k: 1, alignCap: 0, totalTicks: 0 };

  for (const product of order) {
    const item = byId.get(product);
    const ingredients = Array.isArray(item.ingredients) ? item.ingredients : [];
    const craftableIngs = [];
    for (const ing of ingredients) {
      if (!isObj(ing) || !validId(ing.item) || !byId.has(ing.item)) continue;
      const q = finiteInt(ing.quantity, 0, 0, 1e9);
      if (q <= 0) continue;
      const t = craftTicksForMinutes(byId.get(ing.item).craftTimeMinutes);
      craftableIngs.push({ id: ing.item, q, t });
    }
    if (!craftableIngs.length) continue; // no craftable ingredients -> natural

    const totalTicks = craftableIngs.reduce((acc, ing) => lcm(acc, ing.t), 1);
    const candidates = craftableIngs.map((ing) => {
      const k = totalTicks / ing.t;
      const alignCap = Math.ceil(ing.q / k);
      const feasible = Number.isInteger(k) && k <= ing.q && k * ing.t <= MAX_TICKS && alignCap <= MAX_BATCH;
      return { id: ing.id, q: ing.q, t: ing.t, k, alignCap, feasible };
    });

    const failing = candidates.filter((c) => !c.feasible);
    if (failing.length) {
      // One infeasible ingredient makes the whole recipe fall back to k=1.
      const worst = failing.reduce((a, b) => (b.k > a.k ? b : a), failing[0]);
      infeasibleRecipes[product] = {
        status: ALIGNMENT_STATUS.INFEASIBLE,
        item: worst.id,
        k: worst.k,
        q: worst.q,
        totalTicks
      };
      continue;
    }

    // Feasible recipe: register every feasible k>1 candidate; keep the minimum
    // alignCap (tie -> smaller k) for items feeding multiple recipes.
    for (const c of candidates) {
      if (c.k <= 1 || !c.feasible) continue;
      const prev = alignments[c.id];
      if (prev.alignCap === 0 || c.alignCap < prev.alignCap || (c.alignCap === prev.alignCap && c.k < prev.k)) {
        alignments[c.id] = { k: c.k, alignCap: c.alignCap, totalTicks };
      }
    }
  }

  return { alignments, infeasibleRecipes };
}

export default computeAlignment;
