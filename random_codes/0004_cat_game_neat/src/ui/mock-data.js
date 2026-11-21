// Static placeholder dataset for the standalone UI preview.
// All values are illustrative; no simulation or config dependency.
import { ITEMS } from './item-registry.js';

export function placeholderSnapshot() {
  const slots = {}, starvation = {}, status = {}, stash = {};
  for (const it of ITEMS) {
    slots[it.id] = { state: 'IDLE', remainingTicks: 0, batchSize: 0 };
    starvation[it.id] = { material: false, coin: false };
    stash[it.id] = it.type === 'base' ? 40 + (Math.max(0, it.tier) + 1) * 7 : 6;
    status[it.id] = false;
  }
  Object.assign(stash, {
    cotton: 128, tree: 74, rock: 96, quartz: 51,
    fur: 128, fish: 74, flower: 51, stone: 96,
    string: 23, wood: 12, metal: 18, bronze: 6, amethyst: 4, orb: 2,
    ribbon: 9, needles: 7, sparkles: 3, silver: 0,
    gold: 5, pendant: 2, water: 4, necklace: 1, fire: 3,
    waterstone: 0, firestone: 2, elementstone: 1, artifact: 0,
  });
  // Active slots: live countdown + ×N batch badge.
  slots.string = { state: 'ACTIVE', remainingTicks: 12, batchSize: 4 };
  slots.orb = { state: 'ACTIVE', remainingTicks: 864, batchSize: 2 };
  slots.firestone = { state: 'ACTIVE', remainingTicks: 3, batchSize: 1 };
  // Starved free slots: material / coin / both.
  starvation.silver = { material: true, coin: false };
  starvation.gold = { material: false, coin: true };
  starvation.water = { material: true, coin: true };
  Object.assign(status, {
    string: true, wood: true, metal: true, amethyst: true,
    ribbon: true, needles: true, sparkles: true, silver: false,
    gold: true, water: true, fire: true, artifact: false,
  });
  return {
    type: 'SNAPSHOT', tick: 1234, coins: 1204510, totalSpent: 87340,
    stash, slots, starvation, status, done: false,
  };
}

export function placeholderEvents() {
  return [
    { tick: 1228, type: 'COIN_INC', item: '-', batch: 0, delta: 210, reason: 'income' },
    { tick: 1228, type: 'DECISION_ACCEPTED', item: 'string', batch: 4, delta: -220, reason: 'affordable' },
    { tick: 1228, type: 'COIN_DEC', item: 'string', batch: 4, delta: -220, reason: 'batch x4' },
    { tick: 1228, type: 'MATERIAL_DEC', item: 'string', batch: 4, delta: -8, reason: 'cotton x2' },
    { tick: 1228, type: 'BATCH_DONE', item: 'ribbon', batch: 3, delta: 3, reason: 'completed' },
    { tick: 1229, type: 'DECISION_REJECTED', item: 'silver', batch: 0, delta: 0, reason: 'MATERIAL_STARVED: stone insufficient' },
    { tick: 1229, type: 'STARVED_MATERIAL', item: 'silver', batch: 0, delta: 0, reason: 'no stone' },
    { tick: 1230, type: 'DECISION_REJECTED', item: 'gold', batch: 0, delta: 0, reason: 'COIN_STARVED: cannot afford batch' },
    { tick: 1230, type: 'STARVED_COIN', item: 'gold', batch: 0, delta: 0, reason: 'coins < batch cost' },
    { tick: 1231, type: 'STARVED_COIN', item: 'water', batch: 0, delta: 0, reason: 'both limits zero' },
    { tick: 1232, type: 'STASH_UPDATE', item: 'orb', batch: 2, delta: 2, reason: 'craft done' },
    { tick: 1233, type: 'TARGET_UNMET', item: 'artifact', batch: 0, delta: -1, reason: 'consumed as ingredient' },
    { tick: 1234, type: 'COIN_INC', item: '-', batch: 0, delta: 210, reason: 'income' },
    { tick: 1234, type: 'GENOME_RESULT', item: 'artifact', batch: 0, delta: 0, reason: 'gen 42 fitness -874301234' },
  ];
}

function perItem() {
  return {
    string: { coinsSpent: 8200, timeMinutes: 420, busyTicks: 84, batches: 21, units: 84 },
    silver: { coinsSpent: 12100, timeMinutes: 610, busyTicks: 122, batches: 31, units: 62 },
    gold: { coinsSpent: 15400, timeMinutes: 880, busyTicks: 176, batches: 44, units: 44 },
    orb: { coinsSpent: 9800, timeMinutes: 1240, busyTicks: 248, batches: 12, units: 12 },
    elementstone: { coinsSpent: 17340, timeMinutes: 1820, busyTicks: 364, batches: 8, units: 8 },
    artifact: { coinsSpent: 24500, timeMinutes: 2160, busyTicks: 432, batches: 3, units: 3 },
  };
}

export function placeholderResult() {
  return {
    feasible: true, completionTick: 6123, elapsedTicks: 6123, totalSpent: 87340,
    produced: { artifact: 3 },
    schedule: [
      { tick: 12, item: 'string', batch: 4 }, { tick: 40, item: 'ribbon', batch: 3 },
      { tick: 118, item: 'silver', batch: 2 }, { tick: 402, item: 'gold', batch: 1 },
      { tick: 911, item: 'orb', batch: 1 }, { tick: 2187, item: 'elementstone', batch: 1 },
      { tick: 6123, item: 'artifact', batch: 3 },
    ],
    shortfall: {}, perItem: perItem(),
  };
}

export function placeholderTelemetry() {
  return {
    itemId: 'artifact', generation: 42, genomeIndex: 7,
    fitness: -873406123, feasible: true, completionTick: 6123,
    elapsedTicks: 6123, totalSpent: 87340, perItem: perItem(),
  };
}
