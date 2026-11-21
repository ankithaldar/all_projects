/**
 * cost.test.js — batch cost + affordability (plan Interfaces / Simulation Contract).
 */
import { test, assert, assertEqual, assertDeepEqual, assertThrows } from './harness.js';
import { unitCost, totalCost, maxAffordableBatch } from '../sim/cost.js';
import { deriveConstants } from '../config.js';

const { MAX_BATCH } = deriveConstants();

test('unitCost: n-th unit price grows by 50% per step', () => {
  assertEqual(unitCost(100, 1), 100);
  assertEqual(unitCost(100, 2), 150);
  assertEqual(unitCost(100, 3), 200);
  assertEqual(unitCost(50, 4), 125);
});

test('totalCost: b=1 equals init cost', () => {
  assertEqual(totalCost(100, 1), 100);
  assertEqual(totalCost(800, 1), 800);
});

test('totalCost: closed form matches sum of unit costs (large batch)', () => {
  let sum = 0;
  for (let n = 1; n <= 20; n++) sum += unitCost(100, n);
  assertEqual(totalCost(100, 20), sum);
  assertEqual(totalCost(100, 2), 250);
  assertEqual(totalCost(100, 20), 11500);
});

test('totalCost: b=0 is zero', () => {
  assertEqual(totalCost(100, 0), 0);
});

test('maxAffordableBatch: boundary at exact cost threshold', () => {
  assertEqual(maxAffordableBatch(100, 99), 0);
  assertEqual(maxAffordableBatch(100, 100), 1);
  assertEqual(maxAffordableBatch(100, 249), 1);
  assertEqual(maxAffordableBatch(100, 250), 2);
  assertEqual(maxAffordableBatch(100, 11500), 20);
});

test('maxAffordableBatch: never exceeds MAX_BATCH', () => {
  assertEqual(maxAffordableBatch(100, 10 ** 12), MAX_BATCH);
  assertEqual(maxAffordableBatch(100, 11501), MAX_BATCH);
});

test('maxAffordableBatch: zero init cost yields MAX_BATCH, negatives/zero coins yield 0', () => {
  assertEqual(maxAffordableBatch(0, 0), MAX_BATCH);
  assertEqual(maxAffordableBatch(100, 0), 0);
  assertEqual(maxAffordableBatch(100, -5), 0);
});

test('F6: negative and non-finite init costs are inert, never negative spend', () => {
  assertEqual(unitCost(-50, 20), 0);
  assertEqual(totalCost(-50, 20), 0);
  assertEqual(totalCost(Infinity, 20), 0);
  assertEqual(totalCost(NaN, 20), 0);
  assertEqual(maxAffordableBatch(-50, 100), 0);
  assertEqual(maxAffordableBatch(NaN, 100), 0);
  assertEqual(maxAffordableBatch(Infinity, 100), 0);
  assertEqual(maxAffordableBatch(100, Infinity), 0);
});

test('maxAffordableBatch: returned batch is affordable and next is not', () => {
  for (const init of [50, 100, 300, 1500, 10000]) {
    for (const coins of [0, 99, 250, 1000, 8000, 11500, 5998996]) {
      const b = maxAffordableBatch(init, coins);
      assert(b >= 0 && b <= MAX_BATCH, `range ${init}/${coins} -> ${b}`);
      assert(totalCost(init, b) <= coins, `affordable ${init}/${coins} -> ${b}`);
      if (b < MAX_BATCH) {
        assert(totalCost(init, b + 1) > coins, `maximality ${init}/${coins} -> ${b}`);
      }
    }
  }
});
