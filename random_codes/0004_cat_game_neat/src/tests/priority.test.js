// Tests for src/neat/priority.js — OFFLINE critical-chain criticality map.
import { test, assert, assertEqual } from './harness.js';
import { computeCriticality } from '../neat/priority.js';
import { EMBEDDED_CONFIG } from '../config.js';
import { CRAFTABLE_IDS } from '../sim/items.js';

test('priority: computeCriticality returns a null-prototype map of validated craftable ids', () => {
  const out = computeCriticality(EMBEDDED_CONFIG.items, EMBEDDED_CONFIG.targets);
  assertEqual(Object.getPrototypeOf(out), null, 'criticality map must be null-prototype');
  assertEqual(Object.keys(out).length, 19);
  for (const id of CRAFTABLE_IDS) {
    assert(Object.prototype.hasOwnProperty.call(out, id), `missing craftable ${id}`);
    assert(Number.isFinite(out[id]), `non-finite criticality for ${id}`);
    assert(out[id] >= 0 && out[id] <= 1, `${id} criticality out of [0,1]: ${out[id]}`);
  }
});

test('priority: critical-chain (artifact/elementstone) outranks shallow base-tier items', () => {
  const out = computeCriticality(EMBEDDED_CONFIG.items, EMBEDDED_CONFIG.targets);
  assert(out.artifact > out.string, 'deep artifact must outrank shallow string');
  assert(out.artifact > out.wood, 'deep artifact must outrank shallow wood');
  assert(out.elementstone > out.ribbon, 'elementstone chain must outrank ribbon');
  assertEqual(out.artifact, 1, 'deepest target item normalises to 1');
});

test('priority: ignores forbidden prototype keys and non-craftable ids', () => {
  const items = EMBEDDED_CONFIG.items;
  const poisoned = Object.assign({}, EMBEDDED_CONFIG.targets, {
    ['__proto__']: 99, constructor: 99, toString: 99, not_an_item: 99
  });
  const out = computeCriticality(items, poisoned);
  assertEqual(Object.getPrototypeOf(out), null);
  for (const bad of ['__proto__', 'constructor', 'toString', 'not_an_item']) {
    assertEqual(Object.prototype.hasOwnProperty.call(out, bad), false, `must ignore ${bad}`);
  }
  // Reading inherited keys through the null-prototype map yields undefined.
  assertEqual(out['__proto__'], undefined);
  assertEqual(out.constructor, undefined);
});

test('priority: empty / invalid input yields an empty null-prototype map', () => {
  for (const items of [undefined, null, {}, [], 'nope']) {
    const out = computeCriticality(items, {});
    assertEqual(Object.getPrototypeOf(out), null);
    assertEqual(Object.keys(out).length, 0);
  }
});

test('priority: deterministic and independent of object insertion order', () => {
  const a = computeCriticality(EMBEDDED_CONFIG.items, EMBEDDED_CONFIG.targets);
  const b = computeCriticality([...EMBEDDED_CONFIG.items].reverse(), EMBEDDED_CONFIG.targets);
  assertEqual(Object.getPrototypeOf(b), null);
  for (const id of CRAFTABLE_IDS) assertEqual(a[id], b[id], `${id} criticality must be order-independent`);
});

test('priority: craftables not on any target path score zero', () => {
  const items = [
    { id: 'stone', type: 'base', tier: 0 },
    { id: 'a', type: 'craftable', tier: 1, ingredients: [{ item: 'stone', quantity: 1 }], coinCost: 1, craftTimeMinutes: 5 },
    { id: 'b', type: 'craftable', tier: 1, ingredients: [{ item: 'stone', quantity: 1 }], coinCost: 1, craftTimeMinutes: 5 }
  ];
  const out = computeCriticality(items, { a: 2 }); // b feeds no target
  assert(out.a > 0, 'a is a target and must be critical');
  assertEqual(out.b, 0, 'b is off the target path and must score zero');
});
