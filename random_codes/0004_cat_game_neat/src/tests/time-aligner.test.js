// Tests for the pure time-aligner contract (plan Locked 4/5 + worked table).
import { test, assert, assertEqual, assertDeepEqual } from './harness.js';
import { computeAlignment, ALIGNMENT_STATUS } from '../neat/time-aligner.js';
import { computeBatchQuantums } from '../neat/batch-quantizer.js';
import { EMBEDDED_CONFIG } from '../config.js';
import { CRAFTABLE_IDS } from '../sim/items.js';

const alignment = () => {
  const quantums = computeBatchQuantums(EMBEDDED_CONFIG.items);
  return computeAlignment(EMBEDDED_CONFIG.items, {}, quantums);
};

test('aligner: worked feasible alignment table is exact (k, alignCap, totalTicks)', () => {
  const { alignments } = alignment();
  const expected = {
    bronze:   { k: 2, alignCap: 2, totalTicks: 12 }, // silver<-bronze q3
    amethyst: { k: 2, alignCap: 4, totalTicks: 24 }, // pendant<-amethyst q7
    orb:      { k: 2, alignCap: 1, totalTicks: 24 }, // water<-orb q2 (also fire<-orb q6, cap1)
    sparkles: { k: 2, alignCap: 1, totalTicks: 24 }, // gold<-sparkles q2
    ribbon:   { k: 2, alignCap: 1, totalTicks: 6 },  // sparkles<-ribbon q2
    pendant:  { k: 2, alignCap: 2, totalTicks: 72 }  // necklace<-pendant q3
  };
  for (const id of Object.keys(expected)) assertDeepEqual({ ...alignments[id] }, expected[id], `${id} alignment`);
  assertEqual(Object.keys(expected).length, 6, 'exactly six feeder items have active alignment');
});

test('aligner: k=1 natural and infeasible items have alignCap=0 (silver, gold, artifact, string)', () => {
  const { alignments } = alignment();
  for (const id of ['silver', 'gold', 'artifact', 'string', 'water', 'fire', 'needles', 'metal', 'wood']) {
    assert(Object.prototype.hasOwnProperty.call(alignments, id), `missing alignment entry for ${id}`);
    assertEqual(alignments[id].alignCap, 0, `${id} must have alignCap=0`);
    assertEqual(alignments[id].k, 1, `${id} must fall back to k=1`);
  }
});

test('aligner: infeasible recipes reported with the enum and bounded numerics', () => {
  const { infeasibleRecipes } = alignment();
  assertEqual(Object.getPrototypeOf(infeasibleRecipes), null);
  assertEqual(Object.keys(infeasibleRecipes).length, 3);
  for (const id of ['waterstone', 'firestone', 'artifact']) {
    assert(Object.prototype.hasOwnProperty.call(infeasibleRecipes, id), `missing infeasible recipe ${id}`);
    assertEqual(infeasibleRecipes[id].status, ALIGNMENT_STATUS.INFEASIBLE);
    assert(Number.isInteger(infeasibleRecipes[id].k) && infeasibleRecipes[id].k >= 1);
    assert(Number.isInteger(infeasibleRecipes[id].q) && infeasibleRecipes[id].q >= 1);
  }
  assertEqual(infeasibleRecipes.waterstone.k, 24, 'waterstone<-ribbon k=24 > q10');
  assertEqual(infeasibleRecipes.firestone.k, 12, 'firestone<-sparkles k=12 > q2');
  assert(infeasibleRecipes.artifact.k >= 19, 'artifact<-elementstone k=19 / <-necklace k=36');
});

test('aligner: maps are null-prototype keyed by validated craftable ids', () => {
  const { alignments, infeasibleRecipes } = alignment();
  assertEqual(Object.getPrototypeOf(alignments), null);
  for (const id of CRAFTABLE_IDS) assert(Object.prototype.hasOwnProperty.call(alignments, id), `missing ${id}`);
  for (const base of ['cotton', 'tree', 'rock', 'quartz']) {
    assertEqual(Object.prototype.hasOwnProperty.call(alignments, base), false);
  }
});

test('aligner: synthetic prototype keys neither pollute nor appear in map keys', () => {
  const items = [
    { id: 'base', type: 'base', tier: 0 },
    { id: 'prod', type: 'craftable', tier: 1, ingredients: [
      { item: 'base', quantity: 2 }, { item: '__proto__', quantity: 3 },
      { item: 'constructor', quantity: 4 }
    ], coinCost: 1, craftTimeMinutes: 5 }
  ];
  const { alignments, infeasibleRecipes } = computeAlignment(items, {}, { prod: 1, base: 1 });
  assertEqual(Object.getPrototypeOf(alignments), null);
  assertEqual(Object.getPrototypeOf(infeasibleRecipes), null);
  assertEqual(Object.prototype.hasOwnProperty.call(alignments, '__proto__'), false);
  assertEqual(Object.prototype.hasOwnProperty.call(alignments, 'constructor'), false);
  assertEqual(Object.keys(alignments).join(','), 'prod');
  assertEqual(({}).polluted, undefined);
});

test('aligner: feasibility uses per-product q, not total mfr (mfr-invariant)', () => {
  const quantums = computeBatchQuantums(EMBEDDED_CONFIG.items);
  const a = computeAlignment(EMBEDDED_CONFIG.items, {}, quantums);
  const b = computeAlignment(EMBEDDED_CONFIG.items, { bronze: 100000, orb: 99999, amethyst: 7 }, quantums);
  assertDeepEqual(b, a, 'alignments must not depend on mfr quantities');
});

test('aligner: an item feeding several recipes takes the MINIMUM feasible alignCap', () => {
  // feeder f (t=1): p1 -> k=2 cap3, p2 -> k=4 cap2 -> expect cap2.
  const items = [
    { id: 'base', type: 'base', tier: 0 },
    { id: 'f', type: 'craftable', tier: 1, ingredients: [{ item: 'base', quantity: 1 }], coinCost: 1, craftTimeMinutes: 5 },
    { id: 'g', type: 'craftable', tier: 1, ingredients: [{ item: 'base', quantity: 1 }], coinCost: 1, craftTimeMinutes: 10 },
    { id: 'h', type: 'craftable', tier: 1, ingredients: [{ item: 'base', quantity: 1 }], coinCost: 1, craftTimeMinutes: 20 },
    { id: 'p1', type: 'craftable', tier: 2, ingredients: [{ item: 'f', quantity: 6 }, { item: 'g', quantity: 1 }], coinCost: 1, craftTimeMinutes: 5 },
    { id: 'p2', type: 'craftable', tier: 2, ingredients: [{ item: 'f', quantity: 8 }, { item: 'h', quantity: 1 }], coinCost: 1, craftTimeMinutes: 5 }
  ];
  const { alignments } = computeAlignment(items, {}, { f: 2, g: 1, h: 1, p1: 1, p2: 1 });
  assertEqual(alignments.f.alignCap, 2, 'minimum of cap3 and cap2');
  assertEqual(alignments.f.k, 4, 'k from the binding (min-cap) recipe');
});

test('aligner: k*t>MAX_TICKS makes a recipe infeasible', () => {
  const items = [
    { id: 'base', type: 'base', tier: 0 },
    { id: 'f', type: 'craftable', tier: 1, ingredients: [{ item: 'base', quantity: 1 }], coinCost: 1, craftTimeMinutes: 25000 }, // t=5000
    { id: 'g', type: 'craftable', tier: 1, ingredients: [{ item: 'base', quantity: 1 }], coinCost: 1, craftTimeMinutes: 50000 }, // t=10000
    { id: 'p', type: 'craftable', tier: 2, ingredients: [{ item: 'f', quantity: 2 }, { item: 'g', quantity: 1 }], coinCost: 1, craftTimeMinutes: 5 }
  ];
  const { alignments, infeasibleRecipes } = computeAlignment(items, {}, { f: 2, g: 1, p: 1 });
  assertEqual(infeasibleRecipes.p.status, ALIGNMENT_STATUS.INFEASIBLE);
  assertEqual(alignments.f.alignCap, 0, 'whole recipe falls back to k=1');
});

test('aligner: alignCap>MAX_BATCH makes a recipe infeasible', () => {
  const items = [
    { id: 'base', type: 'base', tier: 0 },
    { id: 'f', type: 'craftable', tier: 1, ingredients: [{ item: 'base', quantity: 1 }], coinCost: 1, craftTimeMinutes: 5 }, // t=1
    { id: 'g', type: 'craftable', tier: 1, ingredients: [{ item: 'base', quantity: 1 }], coinCost: 1, craftTimeMinutes: 10 }, // t=2
    { id: 'p', type: 'craftable', tier: 2, ingredients: [{ item: 'f', quantity: 100 }, { item: 'g', quantity: 1 }], coinCost: 1, craftTimeMinutes: 5 }
  ];
  // k_f=2, cap=ceil(100/2)=50 > MAX_BATCH(20) -> infeasible.
  const { infeasibleRecipes } = computeAlignment(items, {}, { f: 100, g: 1, p: 1 });
  assertEqual(infeasibleRecipes.p.status, ALIGNMENT_STATUS.INFEASIBLE);
});

test('aligner: deterministic', () => {
  assertDeepEqual(alignment(), alignment());
});
