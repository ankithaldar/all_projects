// Tests for src/neat/batch-planner.js — frozen planner contract.
import { test, assert, assertEqual, assertDeepEqual } from './harness.js';
import { planBatchCaps } from '../neat/batch-planner.js';
import { computeCriticality } from '../neat/priority.js';
import { applyBatchCaps } from '../neat/policy-caps.js';
import { greedyBatch } from '../neat/defaults.js';
import { createEngine } from '../sim/engine.js';
import { totalCost } from '../sim/cost.js';
import { craftTicksForMinutes, EMBEDDED_CONFIG, PRESETS, MAX_BATCH } from '../config.js';
import { expandManufactureRequired } from '../sim/demand.js';
import { CRAFTABLE_IDS, getRecipe } from '../sim/items.js';
import { computeBatchQuantums, effectiveCap, quantizeBatch } from '../neat/batch-quantizer.js';
import { computeAlignment, ALIGNMENT_STATUS } from '../neat/time-aligner.js';

const MAX_TICKS = 8064;

test('planner: returns the frozen contract shape with a clamped null-prototype caps map', () => {
  const items = EMBEDDED_CONFIG.items;
  const mfr = expandManufactureRequired(
    items, EMBEDDED_CONFIG.targets, PRESETS.reuse.initialStash, PRESETS.reuse.starterMode
  ).manufactureRequired;
  const out = planBatchCaps(items, mfr, MAX_TICKS, MAX_BATCH);
  assertEqual(Object.getPrototypeOf(out.caps), null, 'caps must be null-prototype');
  assertEqual(typeof out.estimatedCost, 'number');
  assertEqual(typeof out.estimatedTicks, 'number');
  assert(Number.isFinite(out.estimatedCost) && out.estimatedCost >= 0);
  assert(Number.isFinite(out.estimatedTicks) && out.estimatedTicks >= 0);
  assertEqual(Object.keys(out.caps).length, 19);
  for (const id of CRAFTABLE_IDS) {
    assert(Object.prototype.hasOwnProperty.call(out.caps, id), `missing cap for ${id}`);
    const c = out.caps[id];
    assert(Number.isInteger(c) && c >= 1 && c <= MAX_BATCH, `cap out of range for ${id}: ${c}`);
  }
});

test('planner: exact critical path equals an independent hand computation (clamped caps)', () => {
  // Small, fully hand-computable DAG: a <- base; b <- a (x2).
  const items = [
    { id: 'stone', type: 'base', tier: 0 },
    { id: 'a', type: 'craftable', tier: 1,
      ingredients: [{ item: 'stone', quantity: 1 }], coinCost: 10, craftTimeMinutes: 5 },
    { id: 'b', type: 'craftable', tier: 2,
      ingredients: [{ item: 'a', quantity: 2 }], coinCost: 20, craftTimeMinutes: 10 }
  ];
  const mfr = { a: 6, b: 3 };
  const out = planBatchCaps(items, mfr, MAX_TICKS, MAX_BATCH);

  // Independent recomputation from the RETURNED caps (a then b, topo order).
  const dp = {};
  dp.a = Math.ceil(mfr.a / out.caps.a) * craftTicksForMinutes(5);
  dp.b = Math.ceil(mfr.b / out.caps.b) * craftTicksForMinutes(10) + dp.a;
  assertEqual(out.estimatedTicks, Math.max(dp.a, dp.b), 'estimatedTicks is the exact clamped-cap critical path');

  // Independent superlinear cost recomputation with the real totalCost formula.
  let expectedCost = 0;
  for (const id of ['a', 'b']) {
    const init = id === 'a' ? 10 : 20;
    const qty = mfr[id];
    const cap = out.caps[id];
    const full = Math.floor(qty / cap);
    const rem = qty % cap;
    expectedCost += full * totalCost(init, cap) + (rem > 0 ? totalCost(init, rem) : 0);
  }
  assertEqual(out.estimatedCost, expectedCost, 'estimatedCost uses the real superlinear totalCost');
});

test('planner: critical chain gets larger caps than shallow items; path fits maxTicks', () => {
  const items = EMBEDDED_CONFIG.items;
  const mfr = expandManufactureRequired(
    items, EMBEDDED_CONFIG.targets, PRESETS.reuse.initialStash, PRESETS.reuse.starterMode
  ).manufactureRequired;
  const out = planBatchCaps(items, mfr, MAX_TICKS, MAX_BATCH);
  assert(out.caps.artifact > out.caps.string, 'artifact (critical chain) > string (shallow)');
  assert(out.caps.elementstone > out.caps.wood, 'elementstone (critical chain) > wood (shallow)');
  assert(out.estimatedTicks <= MAX_TICKS, `critical path must fit the horizon: ${out.estimatedTicks}`);
  // Safety factor 0.8 gives headroom under the hard stop.
  assert(out.estimatedTicks <= Math.floor(MAX_TICKS * 0.8), `safety headroom: ${out.estimatedTicks}`);
});

test('planner: cost is superlinear and cheaper than a max-batch allocation', () => {
  const items = EMBEDDED_CONFIG.items;
  const mfr = expandManufactureRequired(
    items, EMBEDDED_CONFIG.targets, PRESETS.reuse.initialStash, PRESETS.reuse.starterMode
  ).manufactureRequired;
  const capped = planBatchCaps(items, mfr, MAX_TICKS, MAX_BATCH);
  const allMax = Object.create(null);
  for (const id of CRAFTABLE_IDS) allMax[id] = MAX_BATCH;
  let maxCost = 0;
  for (const id of CRAFTABLE_IDS) {
    const qty = mfr[id];
    const init = getRecipe(id).coinCost;
    const full = Math.floor(qty / MAX_BATCH);
    const rem = qty % MAX_BATCH;
    maxCost += full * totalCost(init, MAX_BATCH) + (rem > 0 ? totalCost(init, rem) : 0);
  }
  assert(capped.estimatedCost < maxCost, 'non-critical caps must reduce cost vs all-max-batch');
});

test('planner: ignores invalid/forbidden mfr keys and is deterministic', () => {
  const items = EMBEDDED_CONFIG.items;
  const mfr = { artifact: 3, ['__proto__']: 999, constructor: 999, nope: 50 };
  const a = planBatchCaps(items, mfr, MAX_TICKS, MAX_BATCH);
  const b = planBatchCaps(items, mfr, MAX_TICKS, MAX_BATCH);
  assertEqual(Object.getPrototypeOf(a.caps), null);
  assertEqual(Object.prototype.hasOwnProperty.call(a.caps, '__proto__'), false);
  assertEqual(Object.prototype.hasOwnProperty.call(a.caps, 'constructor'), false);
  assertEqual(Object.prototype.hasOwnProperty.call(a.caps, 'nope'), false);
  assertDeepEqual(a, b, 'planner must be deterministic');
});

test('planner: reuse-preset real-engine regression is feasible, <=8064 ticks, spend<7111975', () => {
  const p = PRESETS.reuse;
  const config = {
    items: EMBEDDED_CONFIG.items,
    targets: EMBEDDED_CONFIG.targets,
    initialStash: p.initialStash,
    starterMode: p.starterMode,
    initialCoins: p.initialCoins
  };
  const mfr = expandManufactureRequired(
    config.items, config.targets, config.initialStash, config.starterMode
  ).manufactureRequired;
  const planned = planBatchCaps(config.items, mfr, MAX_TICKS, MAX_BATCH);
  assert(planned.estimatedTicks <= MAX_TICKS, 'planner critical path within horizon');

  // Apply the SAME shared helper the display runner uses.
  const base = {};
  for (const id of CRAFTABLE_IDS) base[id] = (ctx) => greedyBatch(id, ctx, mfr);
  const policyForItem = applyBatchCaps(base, planned.caps);
  assertEqual(Object.getPrototypeOf(policyForItem), null);

  const engine = createEngine({ config, policyForItem });
  let ticks = 0;
  while (!engine.done && ticks < MAX_TICKS) { engine.step(); ticks += 1; }
  const result = engine.result();
  assertEqual(result.feasible, true, `planned caps must stay feasible (shortfall=${JSON.stringify(result.shortfall)})`);
  assert(result.completionTick <= MAX_TICKS, `completionTick within horizon: ${result.completionTick}`);
  assert(result.totalSpent < 7111975, `spend below shipped feasible baseline: ${result.totalSpent}`);
});

test('planner: criticality map feeding the planner is null-prototype with known ids', () => {
  const crit = computeCriticality(EMBEDDED_CONFIG.items, EMBEDDED_CONFIG.targets);
  assertEqual(Object.getPrototypeOf(crit), null);
  assert(crit.artifact > crit.string);
});

// ---------------------------------------------------------------------------
// Amendment: explicit quantums/alignments, shared effectiveCap, relaxation.
// ---------------------------------------------------------------------------

function batchCount(qty, step, cap, aligned) {
  if (qty <= 0 || cap <= 0) return 0;
  let remaining = qty, count = 0, guard = qty + 1;
  while (remaining > 0 && guard-- > 0) {
    const b = quantizeBatch(cap, step, remaining, cap, aligned);
    if (b <= 0) break;
    remaining -= b; count++;
  }
  return count;
}

test('planner: explicit quantums/alignments are applied through the shared effectiveCap', () => {
  const items = EMBEDDED_CONFIG.items;
  const mfr = expandManufactureRequired(
    items, EMBEDDED_CONFIG.targets, PRESETS.reuse.initialStash, PRESETS.reuse.starterMode
  ).manufactureRequired;
  const quantums = computeBatchQuantums(items);
  const { alignments } = computeAlignment(items, mfr, quantums);
  const out = planBatchCaps(items, mfr, MAX_TICKS, MAX_BATCH, 0.8, quantums, alignments);
  assertEqual(Object.getPrototypeOf(out.caps), null);
  // Every returned cap is already an effective cap (idempotent single
  // authority), under the FINAL alignment (relaxed feeders use alignCap=0).
  const relaxedIds = new Set(out.relaxed.map((r) => r.id));
  for (const id of CRAFTABLE_IDS) {
    const alignCap = relaxedIds.has(id) ? 0 : alignments[id].alignCap;
    assertEqual(
      out.caps[id],
      effectiveCap(out.caps[id], quantums[id], alignCap),
      `${id} cap must be the shared effectiveCap`
    );
    assert(out.caps[id] >= 1 && out.caps[id] <= MAX_BATCH);
  }
  // Aligned feeders are capped by their per-product alignCap.
  assertEqual(out.caps.bronze, alignments.bronze.alignCap, 'bronze capped at alignCap');
  assertEqual(out.caps.orb, alignments.orb.alignCap, 'orb capped at alignCap');
  // Non-aligned feeders use step-multiple caps.
  assertEqual(out.caps.string % quantums.string, 0, 'string cap is a multiple of step');
  assert(out.estimatedTicks <= MAX_TICKS, `estimatedTicks within horizon: ${out.estimatedTicks}`);
});

test('planner: k=1 natural/infeasible items use snapped planner caps, not q-sized caps', () => {
  const items = EMBEDDED_CONFIG.items;
  const mfr = expandManufactureRequired(
    items, EMBEDDED_CONFIG.targets, PRESETS.reuse.initialStash, PRESETS.reuse.starterMode
  ).manufactureRequired;
  const quantums = computeBatchQuantums(items);
  const { alignments } = computeAlignment(items, mfr, quantums);
  const out = planBatchCaps(items, mfr, MAX_TICKS, MAX_BATCH, 0.8, quantums, alignments);
  const relaxedIds = new Set(out.relaxed.map((r) => r.id));
  for (const id of ['silver', 'gold', 'artifact', 'string']) {
    assertEqual(alignments[id].alignCap, 0, `${id} must have alignCap=0`);
    assertEqual(relaxedIds.has(id), false, `${id} must not be alignment-relaxed`);
    // Its effective cap is snapDown of the offline planner cap, never q-sized.
    const snapped = quantums[id] <= 1 ? out.caps[id] : Math.max(quantums[id], Math.floor(out.caps[id] / quantums[id]) * quantums[id]);
    assertEqual(out.caps[id], snapped, `${id} cap must be a step-snapped planner cap`);
  }
});

test('planner: estimatedTicks equals the path recomputed from realised engine batch sizes', () => {
  const items = EMBEDDED_CONFIG.items;
  const mfr = expandManufactureRequired(
    items, EMBEDDED_CONFIG.targets, PRESETS.reuse.initialStash, PRESETS.reuse.starterMode
  ).manufactureRequired;
  const quantums = computeBatchQuantums(items);
  const { alignments } = computeAlignment(items, mfr, quantums);
  const out = planBatchCaps(items, mfr, MAX_TICKS, MAX_BATCH, 0.8, quantums, alignments);

  // Independent recomputation: for each item the number of batches is exactly
  // what the engine's quantizeBatch produces for the returned effective cap.
  const byId = new Map(items.map((i) => [i.id, i]));
  let expected = 0;
  const dp = Object.create(null);
  for (const id of CRAFTABLE_IDS) {
    const item = byId.get(id);
    let predMax = 0;
    for (const ing of item.ingredients) {
      const src = byId.get(ing.item);
      if (src && src.type === 'craftable' && dp[ing.item] != null) predMax = Math.max(predMax, dp[ing.item]);
    }
    const batches = batchCount(mfr[id] || 0, quantums[id], out.caps[id], alignments[id].alignCap > 0);
    const self = batches * craftTicksForMinutes(item.craftTimeMinutes);
    dp[id] = self + predMax;
    expected = Math.max(expected, dp[id]);
  }
  assertEqual(out.estimatedTicks, expected, 'estimatedTicks is the realised-batch critical path');
});

test('planner: forced alignment overrun is relaxed (ALIGNMENT_RELAXED) and fits MAX_TICKS', () => {
  // Single aligned feeder f (alignCap=1) with a huge mfr forces a path > 8064.
  const items = [
    { id: 'base', type: 'base', tier: 0 },
    { id: 'f', type: 'craftable', tier: 1, ingredients: [{ item: 'base', quantity: 1 }], coinCost: 1, craftTimeMinutes: 5 },
    { id: 'g', type: 'craftable', tier: 1, ingredients: [{ item: 'base', quantity: 1 }], coinCost: 1, craftTimeMinutes: 10 },
    { id: 'p', type: 'craftable', tier: 2, ingredients: [{ item: 'f', quantity: 2 }, { item: 'g', quantity: 1 }], coinCost: 1, craftTimeMinutes: 5 }
  ];
  const mfr = { f: 20000, g: 1, p: 1 };
  const quantums = computeBatchQuantums(items);
  const { alignments } = computeAlignment(items, mfr, quantums);
  assertEqual(alignments.f.alignCap, 1, 'f has a forced alignCap=1 alignment');

  const out = planBatchCaps(items, mfr, MAX_TICKS, MAX_BATCH, 0.8, quantums, alignments);
  assert(out.relaxed.some((r) => r.id === 'f' && r.status === ALIGNMENT_STATUS.RELAXED),
    `f must be reported ALIGNMENT_RELAXED: ${JSON.stringify(out.relaxed)}`);
  assert(out.estimatedTicks <= MAX_TICKS, `relaxation must restore the horizon: ${out.estimatedTicks}`);
  assertEqual(out.caps.f % quantums.f, 0, 'relaxed feeder cap reverts to a step multiple');
});

test('planner: explicit-map path is deterministic and prototype-safe', () => {
  const items = EMBEDDED_CONFIG.items;
  const mfr = { artifact: 3, ['__proto__']: 999, constructor: 999 };
  const quantums = computeBatchQuantums(items);
  const { alignments } = computeAlignment(items, mfr, quantums);
  const a = planBatchCaps(items, mfr, MAX_TICKS, MAX_BATCH, 0.8, quantums, alignments);
  const b = planBatchCaps(items, mfr, MAX_TICKS, MAX_BATCH, 0.8, quantums, alignments);
  assertDeepEqual(a, b);
  assertEqual(Object.getPrototypeOf(a.caps), null);
  assertEqual(Object.prototype.hasOwnProperty.call(a.caps, '__proto__'), false);
  assertEqual(Object.prototype.hasOwnProperty.call(a.caps, 'constructor'), false);
  assertEqual(({}).polluted, undefined);
});
