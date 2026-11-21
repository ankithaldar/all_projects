// Tests for src/neat/defaults.js — greedy later-item policy.
import { test, assert, assertEqual } from './harness.js';
import { greedyBatch, MAX_BATCH } from '../neat/defaults.js';

test('defaults: re-exports MAX_BATCH so policies share one cap', () => {
  assertEqual(MAX_BATCH, 20);
});

test('defaults: produces up to remaining demand, capped at MAX_BATCH', () => {
  assertEqual(greedyBatch('orb', { produced: 15 }, { orb: 35 }), MAX_BATCH);
  assertEqual(greedyBatch('orb', { produced: 25 }, { orb: 30 }), 5);
});

test('defaults: produces the full small goal when nothing has been made', () => {
  assertEqual(greedyBatch('silver', { produced: 0 }, { silver: 5 }), 5);
});

test('defaults: returns 0 when the goal is already met or overshot', () => {
  assertEqual(greedyBatch('silver', { produced: 5 }, { silver: 5 }), 0);
  assertEqual(greedyBatch('silver', { produced: 9 }, { silver: 5 }), 0);
});

test('defaults: falls back to stash[itemId] when produced is absent', () => {
  assertEqual(greedyBatch('string', { stash: { string: 7 } }, { string: 10 }), 3);
});

test('defaults: missing required or item yields 0', () => {
  assertEqual(greedyBatch('orb', { produced: 0 }, {}), 0);
  assertEqual(greedyBatch('orb', { produced: 0 }, null), 0);
});

test('defaults: never emits a negative or non-integer batch', () => {
  const b = greedyBatch('orb', { produced: 0 }, { orb: 2.2 });
  assert(Number.isInteger(b) && b >= 0 && b <= MAX_BATCH, `invalid greedy batch: ${b}`);
});
