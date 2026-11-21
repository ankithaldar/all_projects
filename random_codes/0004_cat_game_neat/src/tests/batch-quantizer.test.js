// Tests for the pure batch-quantizer contract (plan Locked 2/7 + quantizeBatch).
import { test, assert, assertEqual, assertDeepEqual } from './harness.js';
import {
  computeBatchQuantums, effectiveCap, quantizeBatch
} from '../neat/batch-quantizer.js';
import { EMBEDDED_CONFIG, MAX_BATCH } from '../config.js';
import { CRAFTABLE_IDS } from '../sim/items.js';

const EXPECTED_QUANTUMS = Object.freeze({
  string: 2, wood: 2, metal: 4, bronze: 3, amethyst: 7, orb: 2, ribbon: 1,
  needles: 2, sparkles: 1, silver: 1, gold: 1, pendant: 3, water: 2,
  necklace: 1, fire: 1, waterstone: 1, firestone: 1, elementstone: 1, artifact: 1
});

test('quantizer: full 19-item downstream-GCD table is exact', () => {
  const out = computeBatchQuantums(EMBEDDED_CONFIG.items);
  assertEqual(Object.getPrototypeOf(out), null, 'quantums must be null-prototype');
  assertDeepEqual({ ...out }, EXPECTED_QUANTUMS);
  assertEqual(Object.keys(out).length, 19);
});

test('quantizer: quantums keyed only by validated craftable ids (bases excluded)', () => {
  const out = computeBatchQuantums(EMBEDDED_CONFIG.items);
  for (const id of CRAFTABLE_IDS) assert(Object.prototype.hasOwnProperty.call(out, id), `missing ${id}`);
  for (const base of ['cotton', 'tree', 'rock', 'quartz']) {
    assertEqual(Object.prototype.hasOwnProperty.call(out, base), false, `base ${base} must be excluded`);
  }
});

test('quantizer: synthetic __proto__/constructor keys neither pollute nor appear', () => {
  const items = [
    { id: 'base', type: 'base', tier: 0 },
    { id: 'craft', type: 'craftable', tier: 1, ingredients: [
      { item: 'base', quantity: 2 }, { item: '__proto__', quantity: 3 },
      { item: 'constructor', quantity: 4 }
    ], coinCost: 1, craftTimeMinutes: 5 }
  ];
  const out = computeBatchQuantums(items);
  assertEqual(Object.getPrototypeOf(out), null);
  assertEqual(Object.prototype.hasOwnProperty.call(out, 'toString'), false);
  assertEqual(Object.prototype.hasOwnProperty.call(out, '__proto__'), false);
  assertEqual(Object.prototype.hasOwnProperty.call(out, 'constructor'), false);
  assertEqual(Object.keys(out).join(','), 'craft');
  assertEqual(({}).polluted, undefined);
});

test('quantizer: computeBatchQuantums is deterministic', () => {
  assertDeepEqual(
    { ...computeBatchQuantums(EMBEDDED_CONFIG.items) },
    { ...computeBatchQuantums(EMBEDDED_CONFIG.items) }
  );
});

test('quantizer: effectiveCap snaps DOWN to a step multiple when alignCap=0', () => {
  assertEqual(effectiveCap(20, 7, 0), 14);
  assertEqual(effectiveCap(20, 3, 0), 18);
  assertEqual(effectiveCap(20, 2, 0), 20);
  assertEqual(effectiveCap(10, 4, 0), 8);
  assertEqual(effectiveCap(7, 7, 0), 7);
  // cap below step is raised to one step (never stalls below a quantum).
  assertEqual(effectiveCap(3, 7, 0), 7);
});

test('quantizer: effectiveCap uses min(plannerCap, alignCap) for alignCap>0 (quantum relaxed)', () => {
  assertEqual(effectiveCap(20, 7, 4), 4);
  assertEqual(effectiveCap(3, 7, 4), 3);
  assertEqual(effectiveCap(20, 3, 2), 2);
  // alignCap>0 bypasses step snapping entirely.
  assertEqual(effectiveCap(20, 7, 5), 5);
});

test('quantizer: effectiveCap clamps to [1,MAX_BATCH] and tolerates garbage', () => {
  assertEqual(effectiveCap(999, 1, 0), MAX_BATCH);
  assertEqual(effectiveCap(-5, 1, 0), 1);
  assertEqual(effectiveCap(NaN, 1, 0), 1);
  assertEqual(effectiveCap(20, 0, 0), 20, 'invalid step falls back to 1');
  assertEqual(effectiveCap(20, 7, NaN), 14, 'invalid alignCap falls back to 0');
});

test('quantizer: effectiveCap is idempotent (single shared authority)', () => {
  for (const s of [1, 2, 3, 7]) {
    for (const a of [0, 1, 4]) {
      const once = effectiveCap(20, s, a);
      assertEqual(effectiveCap(once, s, a), once, `idempotent for step=${s} alignCap=${a}`);
    }
  }
});

test('quantizer: quantizeBatch snaps n>=step DOWN to a step multiple', () => {
  assertEqual(quantizeBatch(20, 7, 20, 14, false), 14);
  assertEqual(quantizeBatch(10, 3, 10, 10, false), 9);
  assertEqual(quantizeBatch(7, 7, 7, 7, false), 7);
});

test('quantizer: quantizeBatch requested<=0 -> 0', () => {
  assertEqual(quantizeBatch(0, 7, 20, 14, false), 0);
  assertEqual(quantizeBatch(-3, 7, 20, 14, false), 0);
  assertEqual(quantizeBatch(NaN, 7, 20, 14, false), 0);
});

test('quantizer: quantizeBatch final remainder (remaining<step) is exact, never snapped up', () => {
  assertEqual(quantizeBatch(20, 7, 3, 14, false), 3);
  assertEqual(quantizeBatch(20, 7, 1, 14, false), 1);
  assertEqual(quantizeBatch(20, 7, 0, 14, false), 0);
  // aligned items also take the exact final remainder.
  assertEqual(quantizeBatch(20, 7, 3, 2, true), 2);
});

test('quantizer: quantizeBatch aligned items use min(requested,remainingCap,batchCap) splits', () => {
  assertEqual(quantizeBatch(6, 7, 6, 2, true), 2);
  assertEqual(quantizeBatch(1, 7, 20, 2, true), 1);
  assertEqual(quantizeBatch(6, 7, 6, 10, true), 6);
});

test('quantizer: quantizeBatch REJECTS 0<n<step with remaining>=step (never snap up)', () => {
  assertEqual(quantizeBatch(2, 7, 20, 14, false), 0);
  assertEqual(quantizeBatch(6, 7, 20, 14, false), 0);
  assertEqual(quantizeBatch(3, 4, 10, 8, false), 0);
});

test('quantizer: quantizeBatch never exceeds remainingCap', () => {
  assertEqual(quantizeBatch(20, 7, 10, 14, false), 7);
  assertEqual(quantizeBatch(20, 3, 11, 18, false), 9);
  assertEqual(quantizeBatch(20, 7, 10, 20, true), 10);
});

test('quantizer: quantizeBatch invariant sweep (integer, bounded, multiple of step)', () => {
  for (const step of [1, 2, 3, 4, 7]) {
    for (let requested = 0; requested <= 20; requested++) {
      for (let remaining = 0; remaining <= 20; remaining++) {
        for (const aligned of [false, true]) {
          const cap = aligned ? Math.min(remaining || 1, 3) : effectiveCap(20, step, 0);
          const b = quantizeBatch(requested, step, remaining, cap, aligned);
          assert(Number.isInteger(b) && b >= 0, `bad integer b=${b}`);
          assert(b <= Math.min(requested, remaining, cap), `b=${b} exceeds bound for s=${step} r=${remaining} cap=${cap}`);
          if (!aligned && remaining >= step && b >= step) assertEqual(b % step, 0, 'non-aligned batch must be a step multiple');
        }
      }
    }
  }
});
