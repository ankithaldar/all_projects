/**
 * batch-planner.js — frozen batch-cap planner (plan "Exact contracts" + Amendment).
 *
 * planBatchCaps(items, mfr, maxTicks, maxBatch, safety, quantums, alignments)
 *   -> { caps, estimatedCost, estimatedTicks, relaxed }
 *
 *   caps             null-prototype map { [validated craftable id]: int in [1,maxBatch] }
 *                    holding EFFECTIVE caps (idempotent under effectiveCap).
 *   estimatedTicks   EXACT critical path recomputed from the batch sizes the
 *                    engine actually realises via quantizeBatch. <= maxTicks.
 *   estimatedCost    real superlinear totalCost() of those realised batches.
 *   relaxed          ALIGNMENT_RELAXED reports for alignments dropped to
 *                    restore the hard horizon.
 *
 * quantums/alignments are explicit inputs (computed by the pure helpers when
 * omitted). Every cap decision goes through the SAME effectiveCap authority
 * the engine imports. If strict alignment pushes the path past maxTicks, the
 * worst aligned contributors are relaxed (alignCap=0) until the path fits.
 */
import { MAX_BATCH, craftTicksForMinutes } from '../config.js';
import { totalCost } from '../sim/cost.js';
import { topoOrder } from '../sim/demand.js';
import { computeCriticality } from './priority.js';
import { computeBatchQuantums, effectiveCap } from './batch-quantizer.js';
import { computeAlignment, ALIGNMENT_STATUS } from './time-aligner.js';

const ID_RE = /^[a-z][a-z0-9_]{0,39}$/;
const validId = (id) => typeof id === 'string' && ID_RE.test(id);
const isObj = (v) => !!v && typeof v === 'object' && !Array.isArray(v);
const hasOwn = (o, k) => Object.prototype.hasOwnProperty.call(o, k);
const clamp01 = (x) => (Number.isFinite(x) && x < 0 ? 0 : (Number.isFinite(x) && x > 1 ? 1 : (Number.isFinite(x) ? x : 0)));

function finiteInt(value, fallback, min, max) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, Math.floor(n)));
}

/**
 * batchCount — the exact number of batches the engine realises for one item,
 * in closed form (no per-batch loop). `cap` is an effective cap: a multiple of
 * `step` for non-aligned items, or an alignCap-sized split for aligned items.
 * Matches quantizeBatch(cap, step, remaining, cap, aligned) tick by tick.
 */
function batchCount(qty, step, cap, aligned) {
  if (qty <= 0 || cap <= 0) return 0;
  if (aligned) return Math.ceil(qty / cap);
  const full = Math.floor(qty / cap);
  const rem = qty % cap;
  if (rem === 0) return full;
  if (rem < step) return full + 1;               // exact final remainder
  return rem % step === 0 ? full + 1 : full + 2; // snapped batch + remainder
}

/**
 * planBatchCaps — deterministic, item-major, effective-cap allocation.
 */
export function planBatchCaps(items, mfr, maxTicks, maxBatch = MAX_BATCH, safety = 0.8, quantums, alignments) {
  const list = Array.isArray(items) ? items : [];
  const limit = finiteInt(maxBatch, MAX_BATCH, 1, MAX_BATCH);
  const horizonRaw = Number(maxTicks);
  const horizon = Number.isFinite(horizonRaw) && horizonRaw > 0 ? Math.floor(horizonRaw) : 8064;
  const safe = clamp01(Number(safety));
  const budget = Math.max(0, Math.floor(horizon * safe));

  const byId = new Map();
  for (const item of list) {
    if (isObj(item) && validId(item.id) && !byId.has(item.id)) byId.set(item.id, item);
  }
  let order;
  try { order = topoOrder(list); } catch (_e) { order = []; }
  order = order.filter((id) => byId.has(id) && byId.get(id).type === 'craftable' && validId(id));

  // Validated quantity map (null-prototype). Unknown / invalid ids are dropped.
  const qty = Object.create(null);
  const source = isObj(mfr) ? mfr : {};
  for (const id of order) qty[id] = finiteInt(source[id], 0, 0, 1e9);

  // Craft time and unit cost derived from the PASSED items (not a global
  // lookup), so custom DAGs in tests behave exactly like the shipped config.
  const craft = Object.create(null);
  const init = Object.create(null);
  const preds = Object.create(null);
  for (const id of order) {
    const item = byId.get(id);
    craft[id] = craftTicksForMinutes(item.craftTimeMinutes);
    const c = Number(item.coinCost);
    init[id] = Number.isFinite(c) && c > 0 ? c : 0;
    preds[id] = [];
    for (const ing of (item.ingredients) || []) {
      const src = byId.get(ing.item);
      if (src && src.type === 'craftable' && order.indexOf(ing.item) !== -1 && validId(ing.item)) {
        preds[id].push(ing.item);
      }
    }
  }

  // Quantum + alignment inputs. When omitted they are derived from the SAME
  // validated items/mfr, so the offline planner and the engine agree.
  const qIn = isObj(quantums) ? quantums : null;
  const aIn = isObj(alignments) ? alignments : null;
  const computedQuantums = qIn ? null : computeBatchQuantums(list);
  let computedAlignments = null;
  if (!aIn) {
    const qmap = Object.create(null);
    for (const id of order) qmap[id] = finiteInt(qIn ? qIn[id] : (computedQuantums && computedQuantums[id]), 1, 1, 1e9);
    computedAlignments = computeAlignment(list, source, qmap).alignments;
  }
  const step = Object.create(null);
  const alignCap = Object.create(null);
  for (const id of order) {
    step[id] = finiteInt(qIn ? qIn[id] : (computedQuantums && computedQuantums[id]), 1, 1, 1e9);
    const rawAlign = aIn ? aIn[id] : (computedAlignments && computedAlignments[id]);
    const capValue = isObj(rawAlign) ? rawAlign.alignCap : rawAlign;
    alignCap[id] = finiteInt(capValue, 0, 0, 1e9);
  }

  const caps = Object.create(null);
  const relaxed = [];

  const selfTicks = (id) => batchCount(qty[id] || 0, step[id], caps[id], alignCap[id] > 0) * craft[id];

  const pathTicks = () => {
    const dp = Object.create(null);
    let max = 0;
    for (const id of order) {
      let predMax = 0;
      for (const p of preds[id]) predMax = Math.max(predMax, dp[p] || 0);
      const value = selfTicks(id) + predMax;
      dp[id] = value;
      if (value > max) max = value;
    }
    return max;
  };

  const estimateCost = () => {
    let total = 0;
    for (const id of order) {
      const n = qty[id] || 0;
      const cap = caps[id];
      const s = step[id];
      if (n <= 0 || cap <= 0) continue;
      if (alignCap[id] > 0) {
        const full = Math.floor(n / cap);
        const rem = n % cap;
        total += full * totalCost(init[id], cap) + (rem > 0 ? totalCost(init[id], rem) : 0);
        continue;
      }
      const full = Math.floor(n / cap);
      const rem = n % cap;
      total += full * totalCost(init[id], cap);
      if (rem > 0) {
        if (rem < s) {
          total += totalCost(init[id], rem);
        } else {
          const snapped = Math.floor(rem / s) * s;
          total += totalCost(init[id], snapped);
          if (rem > snapped) total += totalCost(init[id], rem - snapped);
        }
      }
    }
    return Number.isFinite(total) ? total : 0;
  };

  // Initial allocation from offline criticality: deep-chain items -> large
  // caps, shallow/less-critical items -> small caps. Every value passes
  // through the single shared effectiveCap authority.
  const targets = Object.create(null);
  for (const id of order) if (qty[id] > 0) targets[id] = qty[id];
  const crit = computeCriticality(list, targets);
  for (const id of order) {
    const score = clamp01(hasOwn(crit, id) ? crit[id] : 0);
    const raw = finiteInt(Math.round(1 + score * (limit - 1)), limit, 1, limit);
    caps[id] = effectiveCap(raw, step[id], alignCap[id]);
  }

  function relax(id) {
    alignCap[id] = 0;
    isRelaxed[id] = true;
    relaxed.push({ id, status: ALIGNMENT_STATUS.RELAXED });
    caps[id] = effectiveCap(caps[id], step[id], 0);
  }

  // mfr-aware alignment relaxation FIRST: an alignment whose own single-item
  // self-time consumes too much of the safety budget is a throughput trap
  // (e.g. ribbon alignCap=1 over 1966 units = 5898 ticks). Relax the worst
  // such contributors before repairing, so repair does not inflate unrelated
  // caps to MAX_BATCH just to compensate for a pinned bottleneck.
  const isRelaxed = Object.create(null);
  let preGuard = order.length + 1;
  while (preGuard-- > 0) {
    let worst = null;
    let worstSelf = -1;
    for (const id of order) {
      if (alignCap[id] <= 0 || isRelaxed[id]) continue;
      const self = batchCount(qty[id] || 0, step[id], caps[id], true) * craft[id];
      if (self > worstSelf) { worstSelf = self; worst = id; }
    }
    if (!worst || worstSelf <= budget / 2) break;
    relax(worst);
  }

  // Bounded repair: grow the cap of the largest current time contributor (by
  // a step for non-aligned items, by one up to alignCap for aligned items)
  // until the exact critical path fits the (safety-scaled) budget.
  function repair(target) {
    let guard = order.length * limit + 1;
    while (pathTicks() > target && guard-- > 0) {
      let worst = null;
      let worstSelf = -1;
      for (const id of order) {
        if (caps[id] >= limit) continue;
        if (alignCap[id] > 0 && caps[id] >= alignCap[id]) continue;
        const self = selfTicks(id);
        if (self > worstSelf) { worstSelf = self; worst = id; }
      }
      if (worst === null) break;
      const next = alignCap[worst] > 0
        ? Math.min(alignCap[worst], caps[worst] + 1)
        : Math.min(limit, caps[worst] + step[worst]);
      if (next <= caps[worst]) break;
      caps[worst] = effectiveCap(next, step[worst], alignCap[worst]);
    }
  }
  repair(budget);

  // If alignments still pin the path past the safety budget after repair,
  // relax the worst remaining aligned contributors until the budget fits.
  let relaxGuard = order.length + 1;
  while (pathTicks() > budget && relaxGuard-- > 0) {
    let worst = null;
    let worstSelf = -1;
    for (const id of order) {
      if (alignCap[id] <= 0 || isRelaxed[id]) continue;
      const self = batchCount(qty[id] || 0, step[id], caps[id], true) * craft[id];
      if (self > worstSelf) { worstSelf = self; worst = id; }
    }
    if (worst === null) break;
    relax(worst);
    repair(budget);
  }

  // Defensive last resort: if any aligned bottleneck still overruns, relax
  // everything and grow all caps to the effective maximum.
  if (pathTicks() > horizon) {
    for (const id of order) {
      if (alignCap[id] > 0 && !isRelaxed[id]) {
        isRelaxed[id] = true;
        relaxed.push({ id, status: ALIGNMENT_STATUS.RELAXED });
      }
      alignCap[id] = 0;
      caps[id] = effectiveCap(limit, step[id], 0);
    }
    repair(budget);
  }

  return {
    caps,
    estimatedCost: estimateCost(),
    estimatedTicks: pathTicks(),
    relaxed
  };
}

export default planBatchCaps;
