/**
 * engine.test.js — deterministic tick engine, starvation, target status,
 * totalSpent and per-item attribution.
 */
import { test, assert, assertEqual, assertDeepEqual, assertThrows } from './harness.js';
import { createEngine } from '../sim/engine.js';
import {
  craftTicks, CRAFTABLE_IDS, ITEMS, DAG_EDGES, DEPTH_BY_ID, effectiveTier
} from '../sim/items.js';
import { EMBEDDED_CONFIG, PRESETS, deriveConstants, MAX_BATCH } from '../config.js';
import { expandDemand } from '../sim/demand.js';
import * as demand from '../sim/demand.js';
import { greedyBatch } from '../neat/defaults.js';

const { MAX_TICKS, TICK_MINUTES, COINS_PER_TICK } = deriveConstants();

// Purpose-built policy that chases the NEW manufactureRequired basis: it asks
// for exactly what is still outstanding (completed units subtracted), letting
// the engine clip to materials/coins/MAX_BATCH/remaining. Used by the
// feasibility + invariant tests; the shipped NEAT policies are unchanged.
function manufactureRequiredPolicy(items, targets, stash, mode) {
  const mr = demand.expandManufactureRequired(items, targets, stash, mode).manufactureRequired;
  return (id, ctx) => {
    const need = (mr[id] || 0) - (ctx.produced || 0);
    return need > 0 ? Math.min(need, MAX_BATCH) : 0;
  };
}

// FNV-1a of JSON.stringify(result()) captured from the shipped (uncapped)
// engine for the fixed scenario in the byte-identical regression test below.
// Quantized-engine golden for the fixed 300-tick / full-quantum-want scenario.
// (A want of 1 is a sub-quantum clip for most items and is frozen, so the
// scenario asks for a full 20-unit quantum to exercise real batching.)
const UNCAPPED_GOLDEN = {
  len: 34898,
  fnv1a: '4b298fe9',
  totalSpent: 6061950,
  produced: {
    cotton: 0, tree: 0, rock: 0, quartz: 0, string: 4130, wood: 4034, metal: 1300,
    bronze: 594, amethyst: 238, orb: 340, ribbon: 1302, needles: 314, sparkles: 152,
    silver: 130, gold: 6, pendant: 27, water: 42, necklace: 0, fire: 1,
    waterstone: 1, firestone: 0, elementstone: 0, artifact: 0
  }
};
function fnv1a(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

const RNG = { random: () => 0.5, int: () => 0, range: (a, b) => (a + b) / 2, gaussian: () => 0 };

function base(id, tier = 0) {
  return { id, name: id, tier, type: 'base', ingredients: [], coinCost: 0, craftTimeMinutes: 0 };
}
function craft(id, tier, ingredients, coinCost, craftTimeMinutes) {
  return { id, name: id, tier, type: 'craftable', ingredients, coinCost, craftTimeMinutes };
}
function cfg(items, targets, initialStash = {}, starterMode = 'PREDELIVERED_RESERVED', initialCoins = 0) {
  return { items, targets, initialStash, starterMode, initialCoins };
}
function engine(config, policy) {
  return createEngine({ config, policyForItem: policy, rng: RNG });
}

// ---------------------------------------------------------------------------
// craftTicks contract
// ---------------------------------------------------------------------------
test('craftTicks: ceil(craftTimeMinutes/5) for 5 and 4320', () => {
  assertEqual(craftTicks('string'), 1);   // 5 min
  assertEqual(craftTicks('artifact'), 864); // 4320 min
  assertEqual(craftTicks('metal'), 3);    // 15 min
  assertEqual(craftTicks('necklace'), 152); // 760 min
});

// ---------------------------------------------------------------------------
// graph-derived depth (YAML `tier` is inconsistent with dependencies)
// ---------------------------------------------------------------------------
test('items depth: effectiveTier from DAG_EDGES, raw tier unchanged, edges strictly increase', () => {
  assertEqual(effectiveTier('ribbon'), 2);
  assertEqual(effectiveTier('needles'), 3);   // needs metal(1) + ribbon(2)
  assertEqual(effectiveTier('sparkles'), 4);  // needs needles(3) + ribbon(2)
  assertEqual(effectiveTier('silver'), 5);    // needs bronze(1) + sparkles(4)
  assertEqual(effectiveTier('artifact'), 10); // needs necklace(7) + elementstone(9)
  assertEqual(DEPTH_BY_ID.cotton, 0);
  assertEqual(DEPTH_BY_ID.tree, 0);
  assertEqual(DEPTH_BY_ID.rock, 0);
  assertEqual(DEPTH_BY_ID.quartz, 0);
  for (const id of CRAFTABLE_IDS) assertEqual(effectiveTier(id), DEPTH_BY_ID[id]);

  for (const e of DAG_EDGES) {
    assert(DEPTH_BY_ID[e.from] < DEPTH_BY_ID[e.to],
      `depth(from) < depth(to) for ${e.from}->${e.to}`);
  }

  const byId = new Map(ITEMS.map((i) => [i.id, i]));
  assertEqual(byId.get('needles').tier, 2);   // raw YAML metadata preserved
  assertEqual(byId.get('needles').depth, 3);  // corrected depth added
  assertEqual(byId.get('artifact').tier, 8);
  assertEqual(byId.get('artifact').depth, 10);
  for (const it of ITEMS) assertEqual(it.depth, DEPTH_BY_ID[it.id]);
});

// ---------------------------------------------------------------------------
// tick order, countdown, completion, coin accounting
// ---------------------------------------------------------------------------
test('engine: tick0 accepts batch, +210 coins, spend, post-decrement countdown', () => {
  const config = cfg(
    [base('cotton'), craft('string', 1, [{ item: 'cotton', quantity: 3 }], 100, 15)],
    { string: 1 }
  );
  const e = engine(config, () => 1);
  const ev = e.step();
  assert(Array.isArray(ev));
  assert(ev.some((x) => x.type === 'COIN_INC'));
  assert(ev.some((x) => x.type === 'DECISION_ACCEPTED' && x.item === 'string'));
  assert(ev.some((x) => x.type === 'COIN_DEC' && x.item === 'string'));
  assert(ev.some((x) => x.type === 'MATERIAL_DEC' && x.item === 'string'));
  for (const x of ev) {
    assertDeepEqual(Object.keys(x).sort(), ['delta', 'item', 'reason', 'tick', 'type']);
    assertEqual(x.tick, 0);
  }
  const s = e.snapshot();
  assertEqual(s.tick, 0);
  assertEqual(s.coins, COINS_PER_TICK - 100); // 210 - 100
  assertEqual(s.totalSpent, 100);
  assertEqual(s.slots.string.remainingTicks, 2); // craftTicks(15)=3 -> 2 after decrement
  assertEqual(s.slots.string.batchSize, 1);
  assertEqual(s.slots.string.state, 'ACTIVE');
  assertEqual(s.stash.string, 0);
  assertEqual(s.done, false);
});

test('engine: timer advances and completion zeroes slot + adds stash', () => {
  const config = cfg(
    [base('cotton'), craft('string', 1, [{ item: 'cotton', quantity: 3 }], 100, 15)],
    { string: 1 }
  );
  const e = engine(config, () => 1);
  e.step(); // remaining 2
  const ev2 = e.step();
  assertEqual(e.snapshot().slots.string.remainingTicks, 1);
  assert(!ev2.some((x) => x.type === 'BATCH_DONE'));
  const ev3 = e.step(); // remaining 1 -> 0 -> completes
  assert(ev3.some((x) => x.type === 'BATCH_DONE'));
  const s = e.snapshot();
  assertEqual(s.slots.string.remainingTicks, 0);
  assertEqual(s.slots.string.batchSize, 0);
  assertEqual(s.slots.string.state, 'IDLE');
  assertEqual(s.stash.string, 1);
  assertEqual(s.status.string, true);
  assertEqual(s.done, true);
  assertEqual(e.result().completionTick, 2);
  assertEqual(e.result().elapsedTicks, 3);
  assertEqual(s.coins, 3 * COINS_PER_TICK - 100);
});

test('engine: step is a no-op once complete', () => {
  const config = cfg([base('cotton'), craft('string', 1, [{ item: 'cotton', quantity: 3 }], 100, 5)], { string: 1 });
  const e = engine(config, () => 1);
  e.step(); // craftTicks=1 -> completes in first tick
  assertEqual(e.snapshot().done, true);
  const before = e.snapshot().coins;
  const ev = e.step();
  assertEqual(ev.length, 0);
  assertEqual(e.snapshot().coins, before);
});

// ---------------------------------------------------------------------------
// starvation rule
// ---------------------------------------------------------------------------
test('starvation: material only when clipped to zero by materials', () => {
  const config = cfg(
    [base('cotton'), craft('needles', 1, [{ item: 'cotton', quantity: 3 }], 100, 5),
     craft('sparkles', 2, [{ item: 'needles', quantity: 2 }], 100, 5)],
    { sparkles: 1 }
  );
  const e = engine(config, () => 1);
  const ev = e.step();
  assertDeepEqual(e.snapshot().starvation.sparkles, { material: true, coin: false });
  assert(ev.some((x) => x.type === 'STARVED_MATERIAL' && x.item === 'sparkles'));
  assert(!ev.some((x) => x.type === 'STARVED_COIN' && x.item === 'sparkles'));
});

test('starvation: coin only when clipped to zero by affordability', () => {
  // Cost exceeds the +210/tick income, so the batch is coin-starved.
  const expensive = cfg(
    [base('cotton'), craft('string', 1, [{ item: 'cotton', quantity: 3 }], 1000, 5)],
    { string: 1 }
  );
  const e = engine(expensive, () => 1);
  const ev = e.step();
  assertDeepEqual(e.snapshot().starvation.string, { material: false, coin: true });
  assert(ev.some((x) => x.type === 'STARVED_COIN' && x.item === 'string'));
  assert(!ev.some((x) => x.type === 'STARVED_MATERIAL' && x.item === 'string'));
});

test('starvation: both flags true only when both limits are zero', () => {
  const config = cfg(
    [craft('sparkles', 2, [{ item: 'needles', quantity: 2 }], 1000, 5)],
    { sparkles: 1 }
  );
  const e = engine(config, () => 1);
  const ev = e.step();
  assertDeepEqual(e.snapshot().starvation.sparkles, { material: true, coin: true });
  assert(ev.some((x) => x.type === 'STARVED_MATERIAL'));
  assert(ev.some((x) => x.type === 'STARVED_COIN'));
});

test('starvation: clipped batch > 0 clears both flags (even when one limit was smaller)', () => {
  const config = cfg(
    [base('cotton'), craft('needles', 1, [{ item: 'cotton', quantity: 3 }], 100, 5),
     craft('sparkles', 2, [{ item: 'needles', quantity: 2 }], 150, 10)],
    { sparkles: 1 }, { needles: 2 }, 'CONSUMABLE'
  );
  const e = engine(config, (id) => (id === 'sparkles' ? 5 : 0)); // wants 5, clipped by materials to 1
  const ev = e.step();
  assertEqual(e.snapshot().slots.sparkles.batchSize, 1);
  assertDeepEqual(e.snapshot().starvation.sparkles, { material: false, coin: false });
  assert(!ev.some((x) => x.type.startsWith('STARVED')));
  assertEqual(e.snapshot().totalSpent, 150);
});

test('starvation: MAX_BATCH-only clipping is never starvation; busy slot never starved', () => {
  // MAX_BATCH clipping: huge want, abundant materials + coins -> N=20 > 0.
  // Target 100 keeps the new hard cap non-binding (manufactureRequired is
  // large), so this still isolates MAX_BATCH clipping, not cap clipping.
  const big = cfg(
    [base('cotton'), craft('needles', 1, [{ item: 'cotton', quantity: 3 }], 100, 10),
     craft('sparkles', 2, [{ item: 'needles', quantity: 2 }], 100, 10)],
    { sparkles: 100 }, { needles: 1000 }, 'CONSUMABLE', 100000
  );
  const e1 = engine(big, () => 999);
  const ev1 = e1.step();
  assertEqual(e1.snapshot().slots.sparkles.batchSize, 20);
  assertDeepEqual(e1.snapshot().starvation.sparkles, { material: false, coin: false });
  assert(!ev1.some((x) => x.type.startsWith('STARVED')));

  // Busy-slot rule: A consumes B; after A starts, B=0 so a free A would
  // starve for material, but a busy A must report both flags false.
  const busyCfg = cfg(
    [base('cotton'),
     craft('B', 1, [{ item: 'cotton', quantity: 3 }], 100, 5),
     craft('A', 2, [{ item: 'B', quantity: 1 }], 210, 10)],
    { A: 1, B: 1 }, { B: 1 }, 'CONSUMABLE'
  );
  const e2 = engine(busyCfg, (id) => (id === 'A' ? 1 : 0));
  e2.step(); // A starts (craftTicks=2), consumes the single B
  assertEqual(e2.snapshot().slots.A.state, 'ACTIVE');
  const ev2 = e2.step(); // A busy
  assertDeepEqual(e2.snapshot().starvation.A, { material: false, coin: false });
  assert(!ev2.some((x) => x.type === 'STARVED_MATERIAL' && x.item === 'A'));
});

// ---------------------------------------------------------------------------
// starter modes & target status
// ---------------------------------------------------------------------------
function starterScenario(mode) {
  return cfg(
    [base('cotton'),
     craft('silver', 1, [{ item: 'cotton', quantity: 3 }], 100, 5),
     craft('gold', 2, [{ item: 'silver', quantity: 2 }], 100, 5)],
    { silver: 2, gold: 1 }, { silver: 2 }, mode
  );
}

test('starter PREDELIVERED_RESERVED: starter counts toward target but cannot be consumed', () => {
  const e = engine(starterScenario('PREDELIVERED_RESERVED'), (id) => (id === 'gold' ? 1 : 0));
  const ev = e.step();
  assertEqual(e.snapshot().status.silver, true); // counts toward target
  assertDeepEqual(e.snapshot().starvation.gold, { material: true, coin: false });
  assert(!ev.some((x) => x.type === 'BATCH_DONE' && x.item === 'gold'));
});

test('starter CONSUMABLE: starter can be consumed and logs TARGET_UNMET', () => {
  const e = engine(starterScenario('CONSUMABLE'), (id) => (id === 'gold' ? 1 : 0));
  const ev = e.step();
  assertEqual(e.snapshot().stash.silver, 0);
  assertEqual(e.snapshot().status.silver, false);
  assert(ev.some((x) => x.type === 'TARGET_UNMET' && x.item === 'silver'));
});

test('starter FREE_INGREDIENTS: starter consumable but does NOT count toward target', () => {
  // Before any step: the 2 starter silver are physically present and
  // consumable, but they must NOT satisfy the silver:2 target.
  const before = engine(starterScenario('FREE_INGREDIENTS'), () => 0);
  const b = before.snapshot();
  assertEqual(b.stash.silver, 2);
  assertEqual(b.status.silver, false);

  // After gold consumes the starter silver, physical stash drops to 0 and
  // the target stays unmet (free starter never counted toward target).
  const e = engine(starterScenario('FREE_INGREDIENTS'), (id) => (id === 'gold' ? 1 : 0));
  const events = e.step();
  const s = e.snapshot();
  assertEqual(s.tick, 0);
  assertEqual(s.stash.silver, 0);
  assertEqual(s.status.silver, false);
  assert(events.some((x) => x.type === 'DECISION_ACCEPTED' && x.item === 'gold'));
});

// ---------------------------------------------------------------------------
// totalSpent + perItem attribution
// ---------------------------------------------------------------------------
test('totalSpent is explicit accumulator, never initialCoins - finalCoins', () => {
  const config = cfg(
    [base('cotton'), craft('string', 1, [{ item: 'cotton', quantity: 3 }], 100, 5)],
    { string: 1000000 } // unreachable target -> keeps running
  );
  const start = 5998996;
  config.initialCoins = start;
  // string quantum is 2: a want of 1 is a sub-quantum clip and is frozen;
  // ask for a full quantum so this test still exercises spend accounting.
  const e = engine(config, () => 2);
  for (let i = 0; i < 50; i++) e.step();
  const r = e.result();
  assert(r.totalSpent > 0);
  const finalCoins = e.snapshot().coins;
  assertEqual(finalCoins, start + 50 * COINS_PER_TICK - r.totalSpent);
  assert(start - finalCoins !== r.totalSpent);
  assertEqual(Object.values(r.perItem).reduce((a, x) => a + x.coinsSpent, 0), r.totalSpent);
});

test('perItem: sum(coinsSpent)===totalSpent and sum(timeMinutes)===total busy time', () => {
  // Ask for a full quantum: a want of 1 is now a sub-quantum clip for most
  // items and would be frozen, so no batches would exist to attribute.
  const e = engine(cfg(EMBEDDED_CONFIG.items, { artifact: 1000000 }, {}, 'CONSUMABLE', 5998996),
    () => 20);
  for (let i = 0; i < 300; i++) e.step();
  const r = e.result();
  let coinSum = 0, timeSum = 0, busySum = 0, unitSum = 0, batchSum = 0;
  for (const row of Object.values(r.perItem)) {
    coinSum += row.coinsSpent;
    timeSum += row.timeMinutes;
    busySum += row.busyTicks;
    unitSum += row.units;
    batchSum += row.batches;
    assertEqual(row.timeMinutes, row.busyTicks * TICK_MINUTES);
    assert(row.firstTick === -1 || row.firstTick >= 0);
    assert(row.lastTick >= row.firstTick);
  }
  assertEqual(coinSum, r.totalSpent);
  assertEqual(timeSum, busySum * TICK_MINUTES);
  assertEqual(unitSum, Object.values(r.produced).reduce((a, b) => a + b, 0));
  assert(batchSum > 0);
});

test('result: shortfall, feasibility and completionTick are well-formed', () => {
  const e = engine(cfg([base('cotton'), craft('string', 1, [{ item: 'cotton', quantity: 3 }], 100, 5)], { string: 1 }), () => 1);
  const r0 = e.result();
  assertEqual(r0.feasible, false);
  assertEqual(r0.shortfall.string, 1);
  assertEqual(r0.completionTick, null);
  e.step();
  const r1 = e.result();
  assertEqual(r1.feasible, true);
  assertEqual(r1.completionTick, 0);
  assertEqual(r1.shortfall.string, 0);
  assertEqual(r1.schedule.length, 1);
  assertDeepEqual(Object.keys(r1.schedule[0]).sort(), ['batch', 'item', 'tick']);
});

// ---------------------------------------------------------------------------
// snapshot DTO
// ---------------------------------------------------------------------------
test('snapshot DTO: all fields finite, bounded, correct shape', () => {
  const e = engine(cfg(EMBEDDED_CONFIG.items, { artifact: 3 }, {}, 'PREDELIVERED_RESERVED', 5998996), () => 0);
  const events = e.step();
  const s = e.snapshot();
  assertEqual(s.type, 'SNAPSHOT');
  assertEqual(s.tick, 0); // snapshot reflects the tick just processed
  assert(Number.isInteger(s.coins) && Number.isFinite(s.coins) && s.coins >= 0);
  assert(Number.isInteger(s.totalSpent) && Number.isFinite(s.totalSpent) && s.totalSpent >= 0);
  assert(events.every((x) => Number.isInteger(x.tick) && x.tick === 0));
  assertEqual(s.done, false); // artifact target not met with a no-op policy
  for (const id of CRAFTABLE_IDS) {
    const slot = s.slots[id];
    assert(slot, 'slot ' + id);
    assert(Number.isInteger(slot.remainingTicks) && slot.remainingTicks >= 0);
    assert(Number.isInteger(slot.batchSize) && slot.batchSize >= 0);
    assert(slot.state === 'IDLE' || slot.state === 'ACTIVE');
    assert(typeof s.starvation[id].material === 'boolean');
    assert(typeof s.starvation[id].coin === 'boolean');
  }
  for (const k of Object.keys(s.status)) assert(typeof s.status[k] === 'boolean');
});

// ---------------------------------------------------------------------------
// adversarial guards (F2 zero craft time, F6 bad numbers, F1 quiet mode)
// ---------------------------------------------------------------------------
test('F2: zero craft_time cannot leave a slot ACTIVE at remainingTicks 0 forever', () => {
  const config = cfg([base('cotton'), craft('zero', 1, [{ item: 'cotton', quantity: 3 }], 100, 0)], { zero: 1 });
  const e = engine(config, () => 1);
  const ev = e.step();
  assertEqual(e.snapshot().slots.zero.remainingTicks, 0);
  assertEqual(e.snapshot().slots.zero.state, 'IDLE');
  assertEqual(e.snapshot().stash.zero, 1); // produced, not stuck
  assert(ev.some((x) => x.type === 'BATCH_DONE' && x.item === 'zero'));
});

test('F6: non-finite initialCoins is clamped to a finite bounded value', () => {
  for (const bad of [Infinity, -Infinity, NaN, 1e30]) {
    const config = cfg([base('cotton'), craft('string', 1, [{ item: 'cotton', quantity: 3 }], 100, 5)],
      { string: 1000000 }, {}, 'PREDELIVERED_RESERVED', bad);
    const e = engine(config, () => 0);
    e.step();
    const s = e.snapshot();
    assert(Number.isFinite(s.coins), `coins must be finite for ${bad}`);
    assert(s.coins >= 0 && s.coins <= 1e12 + COINS_PER_TICK, `coins bounded for ${bad}`);
    assert(Number.isFinite(e.result().totalSpent));
  }
});

test('F1: quiet mode allocates no events but yields byte-identical result() totals', () => {
  const config = cfg(EMBEDDED_CONFIG.items, { artifact: 1000000 }, {}, 'CONSUMABLE', 5998996);
  const loud = createEngine({ config, policyForItem: () => 1, rng: RNG, quiet: false });
  const quiet = createEngine({ config, policyForItem: () => 1, rng: RNG, quiet: true });
  for (let i = 0; i < 200; i++) {
    const a = loud.step();
    const b = quiet.step();
    assert(a.length > 0);
    assertEqual(b.length, 0);
  }
  assertDeepEqual(quiet.result(), loud.result());
  assertDeepEqual(quiet.snapshot().stash, loud.snapshot().stash);
});

// ---------------------------------------------------------------------------
// snapshot.required (exact demand map for UI tooltips)
// ---------------------------------------------------------------------------
test('snapshot.required: engine exposes the exact reverse-topo demand map', () => {
  const config = cfg(EMBEDDED_CONFIG.items, { artifact: 1 }, {}, 'PREDELIVERED_RESERVED', 5998996);
  const expected = expandDemand(
    EMBEDDED_CONFIG.items, config.targets, config.initialStash, config.starterMode
  ).required;
  const e = engine(config, () => 0);
  const s = e.snapshot();
  assertDeepEqual(s.required, expected);
  assertEqual(s.required.artifact, 1);
  assertEqual(s.required.silver, 14);
  assertEqual(s.required.ribbon, 98);
  assertEqual(s.required.cotton, undefined); // bases excluded
  // Snapshots must be DTO copies: mutating one must not change the engine.
  s.required.artifact = 999;
  assertEqual(e.snapshot().required.artifact, 1);
});

// ---------------------------------------------------------------------------
// config + demand integration
// ---------------------------------------------------------------------------
test('engine accounts craftTicks using ceil, and demand + engine agree on units', () => {
  const config = cfg(EMBEDDED_CONFIG.items, { string: 5 }, {}, 'CONSUMABLE', 5998996);
  const { required } = expandDemand(EMBEDDED_CONFIG.items, config.targets, {}, 'CONSUMABLE');
  assertEqual(required.string, 5);
  const e = engine(config, (id) => (id === 'string' ? 5 : 0));
  for (let i = 0; i < 20; i++) e.step();
  assert(e.result().perItem.string.units >= 5);
  assertEqual(craftTicks('artifact'), 864);
  assertEqual(MAX_TICKS, 8064);
});

// ---------------------------------------------------------------------------
// Hard cap: manufactureRequired / committed / REQUIRED_MET (amendment)
// ---------------------------------------------------------------------------
test('cap: remaining===0 is rejected with exact REQUIRED_MET, no starvation/STARVED_*', () => {
  // Starter string:1 already satisfies the target, so manufactureRequired=0.
  // The policy still wants 1 and materials/coins are plentiful: the cap must
  // win, be evaluated FIRST, and emit the frozen REQUIRED_MET literal.
  const config = cfg(
    [base('cotton'), craft('string', 1, [{ item: 'cotton', quantity: 3 }], 100, 5)],
    { string: 1 }, { string: 1 }, 'PREDELIVERED_RESERVED', 1000
  );
  const e = engine(config, () => 1);
  assertEqual(e.snapshot().manufactureRequired.string, 0);
  const ev = e.step();
  assertDeepEqual(e.snapshot().starvation.string, { material: false, coin: false });
  const rejects = ev.filter((x) => x.type === 'DECISION_REJECTED' && x.item === 'string');
  assertEqual(rejects.length, 1);
  assertEqual(rejects[0].reason, 'REQUIRED_MET'); // exact frozen literal
  assert(!ev.some((x) => typeof x.type === 'string' && x.type.startsWith('STARVED')),
    'cap rejection must not emit any STARVED_* event');
  assert(!ev.some((x) => x.type === 'DECISION_ACCEPTED' && x.item === 'string'),
    'cap rejection must skip the item (no batch started)');
});

test('cap: in-flight committed accounting prevents a second batch before completion', () => {
  // manufactureRequired[string]=3, craftTicks(15)=3, quantum(string)=2.
  // A huge want clips to remaining=3 and is quantized DOWN to the largest
  // quantum multiple (2); the final remainder (1) is produced only after the
  // first batch completes. Never a second batch while one is in flight.
  // A downstream consumer with quantity 2 gives string quantum 2; it has no
  // demand of its own, so manufactureRequired.string stays 3.
  const config = cfg(
    [base('cotton'),
     craft('string', 1, [{ item: 'cotton', quantity: 3 }], 100, 15),
     craft('consumer', 2, [{ item: 'string', quantity: 2 }], 100, 5)],
    { string: 3 }, {}, 'PREDELIVERED_RESERVED', 100000
  );
  const e = engine(config, () => 999);
  assertEqual(e.snapshot().manufactureRequired.string, 3);
  assertEqual(e.quantum.string, 2);

  e.step(); // batch of 2 starts (3 clipped to remaining, quantized to 2)
  const s0 = e.snapshot();
  assertEqual(s0.slots.string.batchSize, 2);
  assertEqual(s0.slots.string.state, 'ACTIVE');
  assertEqual(s0.produced.string, 0);
  assertEqual(s0.committed.string, 2);

  const stringBatches = () => e.result().schedule.filter((x) => x.item === 'string');
  assertEqual(stringBatches().length, 1);
  assertEqual(stringBatches()[0].batch, 2);

  e.step(); // tick 1: still in flight (remainingTicks 2 -> 1)
  assertEqual(stringBatches().length, 1, 'no second batch while one is in flight');
  assertEqual(e.snapshot().committed.string, 2);
  assertEqual(e.snapshot().produced.string, 0);

  e.step(); // tick 2: completion (remainingTicks 1 -> 0)
  assertEqual(e.snapshot().produced.string, 2);
  assertEqual(e.snapshot().committed.string, 0);
  assertEqual(stringBatches().length, 1, 'still exactly one batch after completion');

  e.step(); // tick 3: remaining=1 < quantum -> final remainder accepted
  assertEqual(stringBatches().length, 2);
  assertDeepEqual(stringBatches().map((b) => b.batch), [2, 1]);
  e.step(); // tick 4: remainder in flight (2 -> 1)
  e.step(); // tick 5: remainder completes (1 -> 0)
  assertEqual(e.snapshot().produced.string, 3);
  assertEqual(e.snapshot().committed.string, 0);
  assertEqual(e.result().feasible, true);
});

test('snapshot: manufactureRequired/produced/committed are craftable-only maps; required unchanged', () => {
  const config = cfg(EMBEDDED_CONFIG.items, { artifact: 1 }, {}, 'PREDELIVERED_RESERVED', 5998996);
  const e = engine(config, () => 0);
  e.step();
  const s = e.snapshot();
  const craftIds = CRAFTABLE_IDS.slice().sort();
  for (const key of ['manufactureRequired', 'produced', 'committed']) {
    assert(s[key], `snapshot.${key} exists`);
    assertDeepEqual(Object.keys(s[key]).sort(), craftIds, `${key} keyed by craftable ids only`);
    for (const id of CRAFTABLE_IDS) {
      const v = s[key][id];
      assert(Number.isInteger(v) && Number.isFinite(v) && v >= 0, `${key}.${id} finite non-negative int`);
    }
    for (const base of ['cotton', 'tree', 'rock', 'quartz']) {
      assertEqual(Object.prototype.hasOwnProperty.call(s[key], base), false, `${key} excludes base ${base}`);
    }
  }
  // required stays the exact shipped demand map (unchanged by the cap).
  const expectedRequired = expandDemand(
    EMBEDDED_CONFIG.items, config.targets, config.initialStash, config.starterMode
  ).required;
  assertDeepEqual(s.required, expectedRequired);
  // DTO copies: mutating one snapshot must not mutate engine state.
  s.manufactureRequired.artifact = 999999;
  s.produced.artifact = 999999;
  s.committed.artifact = 999999;
  const s2 = e.snapshot();
  assertEqual(s2.manufactureRequired.artifact !== 999999, true);
  assertEqual(s2.produced.artifact, 0);
  assertEqual(s2.committed.artifact, 0);
});

test('cap invariant produced+committed<=manufactureRequired holds every tick, all 3 modes', () => {
  for (const mode of ['PREDELIVERED_RESERVED', 'CONSUMABLE', 'FREE_INGREDIENTS']) {
    const targets = EMBEDDED_CONFIG.targets;
    const stash = mode === 'PREDELIVERED_RESERVED' ? PRESETS.reuse.initialStash : {};
    const config = cfg(EMBEDDED_CONFIG.items, targets, stash, mode, 5998996);
    const e = engine(config, manufactureRequiredPolicy(EMBEDDED_CONFIG.items, targets, stash, mode));
    for (let tick = 0; tick < 400 && !e.done; tick++) {
      e.step();
      const s = e.snapshot();
      for (const id of CRAFTABLE_IDS) {
        assert(s.produced[id] + s.committed[id] <= s.manufactureRequired[id],
          `${mode}: ${id} produced+committed exceeded cap at tick ${s.tick}`);
      }
    }
  }
});

test('baseline non-regression: byte-identical result() when cap is non-binding', () => {
  // Huge unreachable target => manufactureRequired is enormous for every
  // craftable, so the cap never clips. result() must match the shipped
  // (pre-cap) golden byte-for-byte for this fixed scenario/seed.
  const config = cfg(EMBEDDED_CONFIG.items, { artifact: 1000000 }, {}, 'CONSUMABLE', 5998996);
  const e = engine(config, () => 20); // full quantum: sub-quantum wants are frozen
  for (let i = 0; i < 300; i++) e.step();
  const r = e.result();
  const json = JSON.stringify(r);
  assertEqual(json.length, UNCAPPED_GOLDEN.len, 'result() JSON length byte-identical');
  assertEqual(fnv1a(json), UNCAPPED_GOLDEN.fnv1a, 'result() JSON bytes byte-identical');
  assertEqual(r.totalSpent, UNCAPPED_GOLDEN.totalSpent);
  assertDeepEqual(r.produced, UNCAPPED_GOLDEN.produced);
});

test('baseline non-regression: greedy presets are no worse than the captured baseline', () => {
  // Captured greedy rollout under in-engine quantization (quantized engine,
  // unchanged greedy policy). Quantized batches change spend and reach.
  const baselines = {
    reuse: { totalSpent: 3686025, completionTick: null, producedCount: 19 },
    empty: { totalSpent: 1693250, completionTick: null, producedCount: 12 }
  };
  for (const preset of ['reuse', 'empty']) {
    const p = PRESETS[preset];
    const config = cfg(EMBEDDED_CONFIG.items, EMBEDDED_CONFIG.targets, p.initialStash, p.starterMode, p.initialCoins);
    const required = expandDemand(EMBEDDED_CONFIG.items, config.targets, config.initialStash, config.starterMode).required;
    const e = engine(config, (id, ctx) => greedyBatch(id, ctx, required));
    let ticks = 0;
    while (!e.done && ticks < MAX_TICKS) { e.step(); ticks++; }
    const r = e.result();
    const producedCount = Object.values(r.produced).filter((v) => v > 0).length;
    assertEqual(r.totalSpent, baselines[preset].totalSpent, `${preset} spend no worse than baseline`);
    assertEqual(r.completionTick, baselines[preset].completionTick, `${preset} completionTick unchanged`);
    assertEqual(producedCount, baselines[preset].producedCount, `${preset} produced item count unchanged`);
  }
});

test('baseline feasibility: reuse preset reaches feasible=true with a manufactureRequired policy', () => {
  // The greedy baseline cannot finish reuse (feasible=false). With the NEW
  // cap basis and an adequate policy, all 11 targets must be completable.
  const targets = EMBEDDED_CONFIG.targets;
  const stash = PRESETS.reuse.initialStash;
  const mode = PRESETS.reuse.starterMode;
  const config = cfg(EMBEDDED_CONFIG.items, targets, stash, mode, PRESETS.reuse.initialCoins);
  const e = engine(config, manufactureRequiredPolicy(EMBEDDED_CONFIG.items, targets, stash, mode));
  let ticks = 0;
  while (!e.done && ticks < MAX_TICKS) { e.step(); ticks++; }
  const r = e.result();
  assertEqual(r.feasible, true, `reuse must be feasible under the cap (shortfall=${JSON.stringify(r.shortfall)})`);
  for (const id of Object.keys(targets)) assertEqual(r.shortfall[id], 0, `shortfall ${id}`);
});

// ---------------------------------------------------------------------------
// policyContext: lazy read-only stash view (plan frozen contract)
// ---------------------------------------------------------------------------

test('policyContext.stash: eager-equivalent values, exact keys, spread-compatible', () => {
  const config = cfg(
    [base('cotton'),
     craft('string', 1, [{ item: 'cotton', quantity: 3 }], 100, 5),
     craft('wood', 1, [{ item: 'cotton', quantity: 1 }], 50, 5)],
    { string: 1000000 },
    { string: 7, wood: 3 }, 'CONSUMABLE', 5998996
  );
  let keys = null, spread = null, stash = null;
  const e = createEngine({
    config,
    // Read synchronously at decision time — this is the exact moment the old
    // eager implementation built its stash, so the values must be identical.
    policyForItem: (id, ctx) => {
      if (stash === null) {
        stash = ctx.stash;
        keys = Object.keys(stash).sort();
        spread = { ...stash };
      }
      return 0;
    },
    rng: RNG
  });
  e.step();
  assert(stash, 'policy did not receive a stash view');
  // Exact key set equals the eager build (every item, bases included).
  assertDeepEqual(keys, ['cotton', 'string', 'wood']);
  // Eager-equivalent scalars: base -> sentinel 1e6, craftable -> produced(0)
  // + starterRemaining (tick 0, no batch started because policy returns 0).
  assertDeepEqual(spread, { cotton: 1000000, string: 7, wood: 3 });
  assertEqual(Object.getPrototypeOf(spread), Object.prototype,
    'object spread yields a plain object');
  // `in` works; forbidden / non-item keys resolve to undefined and never leak.
  assert('string' in stash, 'item id present via in');
  assert(!('__proto__' in stash), 'no __proto__ key');
  assertEqual(stash.__proto__, undefined);
  assertEqual(stash.constructor, undefined);
  assertEqual(stash.nonExistentItem, undefined);
});

test('policyContext: all frozen keys present; required/remaining kept and eager-equivalent', () => {
  const config = cfg(
    [base('cotton'), craft('string', 1, [{ item: 'cotton', quantity: 3 }], 100, 5)],
    { string: 1 }
  );
  let ctx = null;
  const e = createEngine({
    config,
    policyForItem: (id, c) => { if (ctx === null) ctx = c; return 0; },
    rng: RNG
  });
  e.step();
  assert(ctx, 'policy did not receive a context');
  assertDeepEqual(Object.keys(ctx).sort(), [
    'tick', 'itemId', 'coins', 'totalSpent', 'stash', 'slots', 'starvation',
    'blocked', 'status', 'recipe', 'craftTicks', 'required', 'produced', 'remaining'
  ].sort());
  assertEqual(ctx.blocked.string, false, 'blocked telemetry present in policy context');
  assertEqual(ctx.itemId, 'string');
  assertEqual(ctx.tick, 0);
  assertEqual(ctx.coins, COINS_PER_TICK);
  assertEqual(ctx.totalSpent, 0);
  // required/remaining are documented as unused by the controller's
  // buildContext but must remain (not deleted) with their eager values.
  assertEqual(ctx.required, 1);
  assertEqual(ctx.produced, 0);
  assertEqual(ctx.remaining, 1);
  assert(ctx.recipe && ctx.recipe.id === 'string');
  assertEqual(ctx.craftTicks, 1);
});

test('policyContext.stash: every mutation attempt is rejected (throwing Proxy)', () => {
  const config = cfg(
    [base('cotton'), craft('string', 1, [{ item: 'cotton', quantity: 3 }], 100, 5)],
    { string: 1 }, {}, 'PREDELIVERED_RESERVED', 1000
  );
  let stash = null;
  const e = createEngine({
    config,
    policyForItem: (id, ctx) => { if (stash === null) stash = ctx.stash; return 0; },
    rng: RNG
  });
  e.step();
  assert(stash, 'policy did not receive a stash view');
  assertThrows(() => { stash.string = 999; }, 'set of existing key must throw');
  assertThrows(() => { stash.newItem = 1; }, 'set of new key must throw');
  assertThrows(() => Object.defineProperty(stash, 'x', { value: 1 }), 'defineProperty must throw');
  assertThrows(() => { delete stash.string; }, 'delete must throw');
  assertThrows(() => Object.setPrototypeOf(stash, {}), 'setPrototypeOf must throw');
  assertThrows(() => Object.preventExtensions(stash), 'preventExtensions must throw');
  // All values survive the failed mutations unchanged.
  assertEqual(stash.cotton, 1000000);
  assertEqual(stash.string, 0);
});

test('policyContext.stash: one memoised view reused, reads live engine state', () => {
  // Two craftables => two policyContext calls in the same tick: the stash
  // handed to both must be the SAME object (no per-item rebuild).
  const config = cfg(
    [base('cotton'),
     craft('string', 1, [{ item: 'cotton', quantity: 3 }], 100, 5),
     craft('wood', 1, [{ item: 'cotton', quantity: 1 }], 50, 5)],
    { string: 1000000, wood: 1000000 }, {}, 'CONSUMABLE', 5998996
  );
  const seen = [];
  const e = createEngine({
    config,
    policyForItem: (id, ctx) => { seen.push(ctx.stash); return 0; },
    rng: RNG
  });
  e.step();
  assertEqual(seen.length, 2);
  assert(seen[0] === seen[1], 'same lazy view shared across items in one tick');
  e.step();
  assertEqual(seen.length, 4);
  assert(seen[2] === seen[0], 'same lazy view reused across ticks');

  // The view computes values from current state on every read. A 3-tick batch
  // completes on tick 2, after which stash.string reflects the produced units.
  let live = null;
  const e2 = createEngine({
    config: cfg([base('cotton'), craft('string', 1, [{ item: 'cotton', quantity: 3 }], 100, 15)],
      { string: 1 }, {}, 'CONSUMABLE', 5998996),
    policyForItem: (id, ctx) => { live = ctx.stash; return 1; },
    rng: RNG
  });
  e2.step(); // batch starts (3 ticks), not complete
  assertEqual(live.string, 0);
  e2.step(); // remaining 2 -> 1
  assertEqual(live.string, 0);
  e2.step(); // remaining 1 -> 0 -> completes
  assertEqual(live.string, 1, 'stash reads current produced amount');
});

// ---------------------------------------------------------------------------
// Amendment: in-engine batch quantization + per-product lead-time alignment
// ---------------------------------------------------------------------------

test('batch metadata: engine derives the full quantum table and caps internally (no config.caps)', () => {
  const config = cfg(
    EMBEDDED_CONFIG.items, EMBEDDED_CONFIG.targets,
    PRESETS.reuse.initialStash, PRESETS.reuse.starterMode, 5998996
  );
  assertEqual(Object.prototype.hasOwnProperty.call(config, 'caps'), false, 'no config.caps field');
  const e = engine(config, () => 0);
  const q = e.quantum;
  const caps = e.batchCaps;
  const al = e.aligned;

  // All derived maps are null-prototype, keyed by validated craftable ids only.
  for (const map of [q, caps, al, e.blocked]) {
    assertEqual(Object.getPrototypeOf(map), null, 'derived map is null-prototype');
    assertDeepEqual(Object.keys(map).sort(), CRAFTABLE_IDS.slice().sort());
  }

  // Full 19-item GCD table (quantum 1 entries may be omitted by the pure
  // module, in which case the engine's fallback of 1 is the correct value).
  const table = {
    string: 2, wood: 2, metal: 4, bronze: 3, amethyst: 7, orb: 2, ribbon: 1,
    needles: 2, sparkles: 1, silver: 1, gold: 1, pendant: 3, water: 2,
    necklace: 1, fire: 1, waterstone: 1, firestone: 1, elementstone: 1, artifact: 1
  };
  for (const id of CRAFTABLE_IDS) assertEqual(q[id] || 1, table[id], `quantum ${id}`);

  // Worked-table alignments that survive the planner's horizon repair (the
  // planner may relax tight contributors, which shows up as aligned=false).
  const alignedCaps = { bronze: 2, orb: 1, amethyst: 4, pendant: 2 };
  for (const id of CRAFTABLE_IDS) {
    if (Object.prototype.hasOwnProperty.call(alignedCaps, id)) {
      assertEqual(al[id], true, `${id} is aligned`);
      assertEqual(caps[id], alignedCaps[id], `effective cap ${id}`);
    } else {
      assertEqual(al[id], false, `${id} is not aligned`);
      assert(caps[id] % (q[id] || 1) === 0, `${id} cap is a snapped planner cap`);
      assert(caps[id] <= MAX_BATCH && caps[id] >= 1, `${id} cap bounded`);
    }
  }
  // k=1 / infeasible fallbacks use the snapped planner cap, never a q-sized cap.
  for (const id of ['silver', 'gold', 'artifact', 'string', 'water']) {
    assertEqual(al[id], false, `${id} is not aligned`);
    assert(caps[id] % (q[id] || 1) === 0, `${id} cap is a step multiple`);
  }
  caps.bronze = 99; // DTO copy must not mutate engine state
  assertEqual(e.batchCaps.bronze, 2);
});

test('BELOW_QUANTUM: exact frozen literal, blocked telemetry, no state mutation / overspend', () => {
  // amethyst quantum is 7 (downstream per-unit qty 7). The "sink" recipe has a
  // second craftable ingredient with 96 ticks so lcm/t_amethyst = 8 > q=7:
  // alignment is infeasible -> alignCap=0 -> amethyst is non-aligned. With
  // initialCoins=1500, tick-0 coins=1710 give coinLimit=3 (< quantum 7).
  const items = [
    base('quartz'),
    craft('filler', 1, [{ item: 'quartz', quantity: 1 }], 100, 480),   // craftTicks 96
    craft('amethyst', 1, [{ item: 'quartz', quantity: 10 }], 300, 60),  // craftTicks 12, quantum 7
    craft('sink', 2, [
      { item: 'amethyst', quantity: 7 }, { item: 'filler', quantity: 1 }
    ], 100, 480)
  ];
  const config = cfg(items, { amethyst: 7 }, {}, 'PREDELIVERED_RESERVED', 1500);
  const e = engine(config, (id) => (id === 'amethyst' ? 7 : 0));
  assertEqual(e.snapshot().manufactureRequired.amethyst, 7);
  assertEqual(e.quantum.amethyst, 7);
  assertEqual(e.aligned.amethyst, false, 'forced-infeasible alignment falls back to non-aligned');

  const before = e.snapshot();
  const ev = e.step();
  const rejects = ev.filter((x) => x.type === 'DECISION_REJECTED' && x.item === 'amethyst');
  assertEqual(rejects.length, 1, 'exactly one amethyst rejection');
  assertEqual(rejects[0].reason, 'BELOW_QUANTUM'); // exact frozen literal, never interpolated
  assert(rejects[0].reason !== 'REQUIRED_MET', 'not the cap-rejection literal');
  assert(!ev.some((x) => typeof x.type === 'string' && x.type.startsWith('STARVED')),
    'sub-quantum rejection emits no STARVED_* event');
  assert(!ev.some((x) => x.type === 'DECISION_ACCEPTED' && x.item === 'amethyst'),
    'no batch started on rejection');
  assert(!ev.some((x) => x.type === 'DECISION_ACCEPTED' && x.item === 'amethyst' && x.delta > 0));

  const after = e.snapshot();
  assertEqual(after.coins, before.coins + COINS_PER_TICK, 'no coin spend (income only)');
  assertEqual(after.totalSpent, 0);
  assert(after.coins >= 0, 'no negative coin state');
  assertEqual(after.produced.amethyst, 0);
  assertEqual(after.committed.amethyst, 0);
  assertEqual(after.stash.amethyst, 0);
  assertEqual(after.starvation.amethyst.material, false, 'no material starvation flag');
  assertEqual(after.starvation.amethyst.coin, false, 'no coin starvation flag');
  assertDeepEqual(after.starvation.amethyst, { material: false, coin: false });
  assertEqual(after.blocked.amethyst, true, 'blocked telemetry latched in snapshot');
  after.blocked.amethyst = false; // DTO copy
  assertEqual(e.snapshot().blocked.amethyst, true);

  const r = e.result();
  assertEqual(r.perItem.amethyst.coinsSpent, 0);
  assertEqual(r.perItem.amethyst.batches, 0);
  assertEqual(r.perItem.amethyst.units, 0);
  assert(r.schedule.every((s) => s.item !== 'amethyst'), 'no schedule entry on rejection');
  assert(Object.values(after.produced).every((v) => v >= 0), 'no negative produced state');
});

test('aligned batches: bronze splits at cap 2 (q3), orb at cap 1 (q6)', () => {
  // Config A: silver:1 only -> mfr(bronze)=3, alignCap(bronze)=2 -> [2,1].
  const mk = (target) => cfg(EMBEDDED_CONFIG.items, target, {}, 'PREDELIVERED_RESERVED', 5998996);
  const run = (config) => {
    const e = engine(config, manufactureRequiredPolicy(
      EMBEDDED_CONFIG.items, config.targets, config.initialStash, config.starterMode
    ));
    let ticks = 0;
    while (!e.done && ticks < MAX_TICKS) { e.step(); ticks++; }
    return e;
  };
  const batches = (e, id) => e.result().schedule.filter((s) => s.item === id).map((s) => s.batch);

  const a = run(mk({ silver: 1 }));
  assertEqual(a.snapshot().manufactureRequired.bronze, 3);
  assertEqual(a.batchCaps.bronze, 2);
  assertDeepEqual(batches(a, 'bronze'), [2, 1]);
  assertEqual(a.result().perItem.bronze.units, 3);

  const b = run(mk({ fire: 1 }));
  assertEqual(b.snapshot().manufactureRequired.orb, 6);
  assertEqual(b.batchCaps.orb, 1);
  assertDeepEqual(batches(b, 'orb'), [1, 1, 1, 1, 1, 1]);
  assertEqual(b.result().perItem.orb.units, 6);
});

test('remaining basis: quantization clips to manufactureRequired, not policyContext.required', () => {
  // alpha's shipped `required` is 7 (starter 2 credited) but manufactureRequired
  // is 9 (PREDELIVERED_RESERVED does not credit starter as ingredient).
  // alpha quantum is 3; a clip to 9 stays 9, a clip to 7 would snap down to 6.
  const items = [
    base('cotton'),
    craft('alpha', 1, [{ item: 'cotton', quantity: 2 }], 100, 5),
    craft('beta', 2, [{ item: 'alpha', quantity: 3 }], 100, 5)
  ];
  const stash = { alpha: 2 };
  const config = cfg(items, { beta: 3 }, stash, 'PREDELIVERED_RESERVED', 5998996);
  const e = engine(config, () => 999);
  assertEqual(e.snapshot().required.alpha, 7);
  assertEqual(e.snapshot().manufactureRequired.alpha, 9);
  assertEqual(e.quantum.alpha, 3);
  assertEqual(e.aligned.alpha, false);

  const e2 = engine(config, manufactureRequiredPolicy(items, config.targets, stash, 'PREDELIVERED_RESERVED'));
  let ticks = 0;
  while (!e2.done && ticks < MAX_TICKS) { e2.step(); ticks++; }
  const r = e2.result();
  // Produced units converge on the mfr basis (9), never the legacy required (7).
  assertEqual(r.perItem.alpha.units, 9, 'produced alpha follows manufactureRequired (9)');
  assert(r.perItem.alpha.units !== e2.snapshot().required.alpha, 'not the legacy required basis');
  assertEqual(r.perItem.beta.units, 3);
  assertEqual(r.feasible, true);
  for (const s of r.schedule) {
    if (s.item === 'alpha') assertEqual(s.batch % e2.quantum.alpha, 0, 'alpha batch is a quantum multiple');
  }
});

test('non-aligned batches are positive quantum multiples except the exact final remainder', () => {
  const targets = EMBEDDED_CONFIG.targets;
  const stash = PRESETS.reuse.initialStash;
  const config = cfg(EMBEDDED_CONFIG.items, targets, stash, PRESETS.reuse.starterMode, 5998996);
  const e = engine(config, manufactureRequiredPolicy(EMBEDDED_CONFIG.items, targets, stash, PRESETS.reuse.starterMode));
  let ticks = 0;
  while (!e.done && ticks < MAX_TICKS) { e.step(); ticks++; }

  for (const id of ['string', 'wood', 'metal', 'needles', 'water']) {
    const step = e.quantum[id];
    assert(step > 1, `${id} has a non-trivial quantum`);
    assertEqual(e.aligned[id], false, `${id} is non-aligned`);
    const batches = e.result().schedule.filter((s) => s.item === id).map((s) => s.batch);
    assert(batches.length > 0, `${id} produced at least one batch`);
    for (let i = 0; i < batches.length; i++) {
      const b = batches[i];
      assert(b > 0, `${id} batch positive`);
      const last = i === batches.length - 1;
      assert(b % step === 0 || (last && b < step),
        `${id} batch ${b} must be a multiple of ${step} or a final remainder`);
    }
  }
});
