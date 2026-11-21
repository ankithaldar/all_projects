/**
 * demand.test.js — reverse-topo exact demand expansion + starter modes.
 */
import { test, assert, assertEqual, assertDeepEqual, assertThrows } from './harness.js';
import { expandDemand, topoOrder } from '../sim/demand.js';
import * as demand from '../sim/demand.js';
import { EMBEDDED_CONFIG, PRESETS } from '../config.js';

const ITEMS = EMBEDDED_CONFIG.items;

// Minimal linear DAG: A(cotton x1) -> B(A x2). B is built first, then A.
function linearItems() {
  return [
    { id: 'cotton', name: 'cotton', tier: 0, type: 'base', ingredients: [], coinCost: 0, craftTimeMinutes: 0 },
    { id: 'A', name: 'A', tier: 1, type: 'craftable', ingredients: [{ item: 'cotton', quantity: 1 }], coinCost: 100, craftTimeMinutes: 5 },
    { id: 'B', name: 'B', tier: 2, type: 'craftable', ingredients: [{ item: 'A', quantity: 2 }], coinCost: 100, craftTimeMinutes: 5 }
  ];
}

test('expandDemand: artifact:1 propagates downstream through entire DAG', () => {
  const { required } = expandDemand(ITEMS, { artifact: 1 }, {}, 'PREDELIVERED_RESERVED');
  assertEqual(required.artifact, 1);
  assertEqual(required.elementstone, 1);
  assertEqual(required.necklace, 1);
  assertEqual(required.waterstone, 1);
  assertEqual(required.firestone, 1);
  assertEqual(required.fire, 1);
  assertEqual(required.water, 2);
  assertEqual(required.pendant, 3);
  assertEqual(required.gold, 3);
  assertEqual(required.silver, 14);
  assertEqual(required.sparkles, 22);
  assertEqual(required.needles, 44);
  assertEqual(required.ribbon, 98);
  assertEqual(required.orb, 10);
  assertEqual(required.amethyst, 21);
  assertEqual(required.bronze, 42);
  assertEqual(required.metal, 176);
  assertEqual(required.wood, 196);
  assertEqual(required.string, 196);
});

test('expandDemand: downstream artifact:3 scales all contributions', () => {
  const { required } = expandDemand(ITEMS, { artifact: 3 }, {}, 'PREDELIVERED_RESERVED');
  assertEqual(required.artifact, 3);
  assertEqual(required.elementstone, 3);
  assertEqual(required.necklace, 3);
  assertEqual(required.silver, 42);
  assertEqual(required.ribbon, 294);
  assertEqual(required.string, 588);
});

test('expandDemand: base items are excluded from required', () => {
  const { required } = expandDemand(ITEMS, { artifact: 3 }, {}, 'PREDELIVERED_RESERVED');
  for (const base of ['cotton', 'tree', 'rock', 'quartz']) {
    assertEqual(required[base], undefined, `base ${base} must be excluded`);
  }
});

test('expandDemand: PREDELIVERED_RESERVED subtracts starter stash from demand', () => {
  const { required } = expandDemand(ITEMS, { silver: 12 }, { silver: 2, bronze: 30 }, 'PREDELIVERED_RESERVED');
  assertEqual(required.silver, 10);
  assertEqual(required.bronze, 0);
  assertEqual(required.sparkles, 10);
});

test('expandDemand: CONSUMABLE also credits starter stash (same required values)', () => {
  const { required } = expandDemand(ITEMS, { silver: 12 }, { silver: 2, bronze: 30 }, 'CONSUMABLE');
  assertEqual(required.silver, 10);
  assertEqual(required.bronze, 0);
  assertEqual(required.sparkles, 10);
});

test('expandDemand: FREE_INGREDIENTS does not credit starter stash toward targets', () => {
  const { required } = expandDemand(ITEMS, { silver: 12 }, { silver: 2, bronze: 30 }, 'FREE_INGREDIENTS');
  assertEqual(required.silver, 12);
  assertEqual(required.bronze, 36);
  assertEqual(required.sparkles, 12);
  assertEqual(required.needles, 24);
});

test('expandDemand: required never negative and defaults to PREDELIVERED_RESERVED', () => {
  const a = expandDemand(ITEMS, { silver: 12 }, { silver: 999 });
  const b = expandDemand(ITEMS, { silver: 12 }, { silver: 999 }, 'PREDELIVERED_RESERVED');
  assertDeepEqual(a, b);
  assertEqual(a.required.silver, 0);
  assert(a.required.sparkles === undefined || a.required.sparkles >= 0);
});

test('expandDemand: multiple targets combine additively', () => {
  const { required } = expandDemand(ITEMS, { silver: 2, pendant: 1 }, {}, 'PREDELIVERED_RESERVED');
  // pendant:1 -> amethyst7, silver2 ; silver target2 ; + pendant's silver2 => silver demand4
  assertEqual(required.pendant, 1);
  assertEqual(required.silver, 4);
  assertEqual(required.amethyst, 7);
});

// ---------------------------------------------------------------------------
// expandManufactureRequired — NEW basis for the engine hard cap. The shipped
// expandDemand/required map stays untouched; these tests pin the exact
// reverse-topo arithmetic and the three starter-mode rules from the amendment.
// ---------------------------------------------------------------------------
test('expandManufactureRequired: reuse preset golden arithmetic + all 19 craftables', () => {
  const { manufactureRequired } = demand.expandManufactureRequired(
    ITEMS, EMBEDDED_CONFIG.targets, PRESETS.reuse.initialStash, PRESETS.reuse.starterMode
  );
  assertEqual(manufactureRequired.bronze, 867);
  assertEqual(manufactureRequired.string, 3932);
  assertEqual(manufactureRequired.amethyst, 378);
  assertEqual(manufactureRequired.silver, 289);
  const craftIds = topoOrder(ITEMS);
  assertEqual(craftIds.length, 19);
  const keys = Object.keys(manufactureRequired).sort();
  assertDeepEqual(keys, craftIds.slice().sort());
  for (const id of craftIds) {
    const v = manufactureRequired[id];
    assert(Number.isInteger(v) && Number.isFinite(v) && v >= 0, `non-negative int for ${id}`);
  }
  for (const base of ['cotton', 'tree', 'rock', 'quartz']) {
    assertEqual(manufactureRequired[base], undefined, `base ${base} must be absent`);
  }
});

test('expandManufactureRequired: PREDELIVERED_RESERVED never credits starter stash as ingredients', () => {
  const items = linearItems();
  // A is a non-target; its 2 starter units cannot offset the 6 needed for B.
  const a = demand.expandManufactureRequired(items, { B: 3 }, { A: 2 }, 'PREDELIVERED_RESERVED').manufactureRequired;
  assertEqual(a.B, 3);
  assertEqual(a.A, 6);
  // Starter B counts toward the B target, so fewer B units are manufactured.
  const b = demand.expandManufactureRequired(items, { B: 3 }, { B: 1 }, 'PREDELIVERED_RESERVED').manufactureRequired;
  assertEqual(b.B, 2);
  assertEqual(b.A, 4);
});

test('expandManufactureRequired: CONSUMABLE uses leftover starter to offset ingredient need', () => {
  const items = linearItems();
  const a = demand.expandManufactureRequired(items, { B: 3 }, { A: 2 }, 'CONSUMABLE').manufactureRequired;
  assertEqual(a.B, 3);
  assertEqual(a.A, 4); // max(0, 6 - 2)
  const b = demand.expandManufactureRequired(items, { B: 3 }, { B: 1, A: 2 }, 'CONSUMABLE').manufactureRequired;
  assertEqual(b.B, 2); // min(1,3) used for target
  assertEqual(b.A, 2); // ingredient need 4, leftover starter 2 -> max(0, 4 - 2)
});

test('expandManufactureRequired: FREE_INGREDIENTS does not credit starter toward targets', () => {
  const items = linearItems();
  const a = demand.expandManufactureRequired(items, { B: 3 }, { B: 3 }, 'FREE_INGREDIENTS').manufactureRequired;
  assertEqual(a.B, 3); // target need is NOT reduced by starter stash
  const b = demand.expandManufactureRequired(items, { A: 5, B: 3 }, { A: 2 }, 'FREE_INGREDIENTS').manufactureRequired;
  assertEqual(b.B, 3);
  assertEqual(b.A, 9); // targetNeed 5 + max(0, ingredientNeed 6 - starter 2)
});

test('expandManufactureRequired: refills a target consumed as an ingredient in all 3 modes', () => {
  const items = linearItems();
  const targets = { A: 5, B: 3 }; // A is both a target and an ingredient of B
  const pr = (stash) => demand.expandManufactureRequired(items, targets, stash, 'PREDELIVERED_RESERVED').manufactureRequired;
  assertEqual(pr({}).A, 11); // target 5 + 6 consumed by B
  assertEqual(pr({ A: 5 }).A, 6); // starter covers target; refill only the 6 consumed
  const co = (stash) => demand.expandManufactureRequired(items, targets, stash, 'CONSUMABLE').manufactureRequired;
  assertEqual(co({ A: 8 }).A, 3); // used 5 for target, leftover 3 offsets 6 consumed
  assertEqual(co({ A: 5 }).A, 6);
  const fi = (stash) => demand.expandManufactureRequired(items, targets, stash, 'FREE_INGREDIENTS').manufactureRequired;
  assertEqual(fi({ A: 2 }).A, 9); // target 5 + (6 - 2 starter)
});

test('expandManufactureRequired: target detection uses hasOwn, ignoring inherited keys', () => {
  const items = linearItems();
  const targets = Object.create({ A: 99 }); // A inherited, NOT an own target
  targets.B = 3;
  const { manufactureRequired } = demand.expandManufactureRequired(items, targets, {}, 'PREDELIVERED_RESERVED');
  assertEqual(manufactureRequired.B, 3);
  assertEqual(manufactureRequired.A, 6); // non-target branch; must not use inherited 99
  assertEqual(Object.prototype.hasOwnProperty.call(manufactureRequired, 'cotton'), false);
});

test('expandManufactureRequired: expandDemand required map is unchanged', () => {
  const before = expandDemand(ITEMS, EMBEDDED_CONFIG.targets, PRESETS.reuse.initialStash, 'PREDELIVERED_RESERVED').required;
  demand.expandManufactureRequired(ITEMS, EMBEDDED_CONFIG.targets, PRESETS.reuse.initialStash, 'PREDELIVERED_RESERVED');
  const after = expandDemand(ITEMS, EMBEDDED_CONFIG.targets, PRESETS.reuse.initialStash, 'PREDELIVERED_RESERVED').required;
  assertDeepEqual(after, before);
  assertEqual(after.bronze, 807); // shipped value differs from manufactureRequired 867
});
