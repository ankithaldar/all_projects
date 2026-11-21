/**
 * demand.js — exact demand expansion in REVERSE topological order.
 *
 * required[name] = max(0, demand[name] - starterCredit[name])
 * computed final-products-first; a product's required units are distributed
 * to its (non-base) ingredients. Base items are excluded from `required`.
 */
/** Deterministic topological order of craftable items (ingredients first). */
export function topoOrder(items) {
  const craftables = items.filter((i) => i.type === 'craftable');
  const byId = new Map(items.map((i) => [i.id, i]));
  const indeg = new Map();
  const succ = new Map();
  for (const item of craftables) {
    indeg.set(item.id, 0);
    succ.set(item.id, []);
  }
  for (const item of craftables) {
    for (const ing of item.ingredients) {
      const src = byId.get(ing.item);
      if (src && src.type === 'craftable') {
        succ.get(ing.item).push(item.id);
        indeg.set(item.id, indeg.get(item.id) + 1);
      }
    }
  }
  const index = new Map(craftables.map((i, idx) => [i.id, idx]));
  const sortReady = (arr) => arr.sort((a, b) => index.get(a) - index.get(b));
  const ready = sortReady(craftables.filter((i) => indeg.get(i.id) === 0).map((i) => i.id));
  const order = [];
  while (ready.length) {
    const id = ready.shift();
    order.push(id);
    for (const s of succ.get(id)) {
      indeg.set(s, indeg.get(s) - 1);
      if (indeg.get(s) === 0) ready.push(s);
    }
    sortReady(ready);
  }
  if (order.length !== craftables.length) throw new Error('DAG_CYCLE');
  return order;
}

/**
 * expandDemand(items, targets, starterStash, starterMode) -> { required }
 *
 * starterMode:
 *   PREDELIVERED_RESERVED (default) / CONSUMABLE => starterCredit = starterStash
 *   FREE_INGREDIENTS                            => starterCredit = 0
 */
export function expandDemand(items, targets, starterStash, starterMode = 'PREDELIVERED_RESERVED') {
  const order = topoOrder(items);
  const byId = new Map(items.map((i) => [i.id, i]));
  const demand = {};
  for (const key of Object.keys(targets || {})) {
    demand[key] = Math.max(0, Math.floor(Number(targets[key]) || 0));
  }
  const free = starterMode === 'FREE_INGREDIENTS';
  const starter = starterStash || {};
  const credit = (id) => (free ? 0 : Math.max(0, Math.floor(Number(starter[id]) || 0)));

  const required = {};
  for (let i = order.length - 1; i >= 0; i--) {
    const id = order[i];
    const d = Math.max(0, Math.floor(demand[id] || 0));
    const r = Math.max(0, d - credit(id));
    required[id] = r;
    const item = byId.get(id);
    for (const ing of item.ingredients) {
      const src = byId.get(ing.item);
      if (src && src.type === 'base') continue; // bases: unlimited, excluded
      demand[ing.item] = (demand[ing.item] || 0) + r * ing.quantity;
    }
  }
  return { required };
}

/**
 * expandManufactureRequired(items, targets, starterStash, starterMode)
 *   -> { manufactureRequired }
 *
 * Hard-cap basis for the engine (NEW map — the shipped `required` is untouched).
 * Computed in REVERSE topo order: a product's gross requirement is distributed
 * to its craftable ingredients only. Base items are unlimited and never capped.
 *
 * Per item:
 *   PREDELIVERED_RESERVED (starter counts toward target, NOT consumable):
 *     target    = max(0, targetNeed - st) + ingredientNeed
 *     non-target= ingredientNeed
 *   CONSUMABLE (counts toward target AND consumable):
 *     target    = (targetNeed - min(st,targetNeed)) + max(0, ingredientNeed - max(0,st-min(st,targetNeed)))
 *     non-target= max(0, ingredientNeed - st)
 *   FREE_INGREDIENTS (starter does NOT count toward target, consumable):
 *     target / non-target = targetNeed + max(0, ingredientNeed - st)
 */
export function expandManufactureRequired(
  items, targets, starterStash, starterMode = 'PREDELIVERED_RESERVED'
) {
  const order = topoOrder(items);
  const byId = new Map(items.map((i) => [i.id, i]));
  const starter = starterStash || {};
  const ingredientNeed = {};
  const manufactureRequired = {};
  const hasOwn = (o, k) => Object.prototype.hasOwnProperty.call(o, k);

  const floor0 = (v) => {
    const n = Math.floor(Number(v) || 0);
    return Number.isFinite(n) && n > 0 ? n : 0;
  };

  for (let i = order.length - 1; i >= 0; i--) {
    const id = order[i];
    const item = byId.get(id);
    const targetNeed = floor0(targets ? targets[id] : 0);
    const need = floor0(ingredientNeed[id]);
    const st = floor0(starter[id]);
    const isTarget = hasOwn(targets || {}, id);

    let cap;
    if (starterMode === 'PREDELIVERED_RESERVED') {
      cap = isTarget ? Math.max(0, targetNeed - st) + need : need;
    } else if (starterMode === 'CONSUMABLE') {
      if (isTarget) {
        const usedForTarget = Math.min(st, targetNeed);
        const leftover = Math.max(0, st - usedForTarget);
        cap = (targetNeed - usedForTarget) + Math.max(0, need - leftover);
      } else {
        cap = Math.max(0, need - st);
      }
    } else { // FREE_INGREDIENTS
      cap = targetNeed + Math.max(0, need - st);
    }

    manufactureRequired[id] = cap;
    for (const ing of item.ingredients) {
      const src = byId.get(ing.item);
      if (!src || src.type === 'base') continue; // bases: unlimited, never propagated
      ingredientNeed[ing.item] = (ingredientNeed[ing.item] || 0) + cap * ing.quantity;
    }
  }

  return { manufactureRequired };
}
