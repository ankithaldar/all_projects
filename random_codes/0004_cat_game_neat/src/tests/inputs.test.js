// Tests for src/neat/inputs.js — per-item observation and batch decoding.
import { test, assert, assertEqual, assertDeepEqual } from './harness.js';
import {
  MAX_BATCH, OBSERVATION_SIZE, OBS_FIELDS, makeObservation, decodeBatch
} from '../neat/inputs.js';
import { MAX_BATCH as CONFIG_MAX_BATCH } from '../config.js';

test('inputs: exposes fixed observation contract', () => {
  assertEqual(MAX_BATCH, CONFIG_MAX_BATCH, 'MAX_BATCH must be the config.js single source');
  assertEqual(OBSERVATION_SIZE, 5);
  assertDeepEqual(OBS_FIELDS, ['remainingDemand', 'coins', 'ticksRemaining', 'materialAvailability', 'slotActive']);
});

test('inputs: observation length matches OBSERVATION_SIZE and is finite/bounded', () => {
  const obs = makeObservation({ required: 10, produced: 2, coins: 1000, tick: 100, maxTicks: 8064, materialsAvailable: 5, slotActive: true });
  assertEqual(obs.length, OBSERVATION_SIZE);
  for (const v of obs) {
    assert(Number.isFinite(v), `observation value must be finite: ${v}`);
    assert(v >= 0 && v <= 1, `observation value must be in [0,1]: ${v}`);
  }
});

test('inputs: remaining demand = required - produced (explicit produced)', () => {
  const a = makeObservation({ required: 10, produced: 0 });
  const b = makeObservation({ required: 10, produced: 4 });
  const c = makeObservation({ required: 10, produced: 10 });
  assert(a[0] > b[0], 'fewer produced -> more remaining demand');
  assert(b[0] > c[0], 'more produced -> less remaining demand');
  assertEqual(c[0], 0, 'remaining demand zero when fully produced');
});

test('inputs: remaining demand uses stash[itemId] when produced is absent', () => {
  const withStash = makeObservation({ itemId: 'orb', required: 8, stash: { orb: 3 } });
  const explicit = makeObservation({ itemId: 'orb', required: 8, produced: 3 });
  assert(withStash[0] > 0, 'stash must reduce but not erase remaining demand');
  assertEqual(withStash[0], explicit[0]);
});

test('inputs: remaining demand is 0 when required is missing or non-positive', () => {
  assertEqual(makeObservation({})[0], 0);
  assertEqual(makeObservation({ required: 0, produced: 0 })[0], 0);
});

test('inputs: coins are normalized and clamped to [0,1]', () => {
  assertEqual(makeObservation({ coins: 0 })[1], 0);
  const mid = makeObservation({ coins: 5000000 })[1];
  assert(mid > 0 && mid < 1, `mid coins should be in (0,1), got ${mid}`);
  assertEqual(makeObservation({ coins: 1e12 })[1], 1);
});

test('inputs: ticks remaining decreases as tick advances', () => {
  const start = makeObservation({ tick: 0, maxTicks: 8064 })[2];
  const mid = makeObservation({ tick: 4032, maxTicks: 8064 })[2];
  const end = makeObservation({ tick: 8064, maxTicks: 8064 })[2];
  assertEqual(start, 1);
  assertEqual(mid, 0.5);
  assertEqual(end, 0);
});

test('inputs: material availability and slot state are exposed', () => {
  const none = makeObservation({ materialsAvailable: 0, required: 10 });
  const some = makeObservation({ materialsAvailable: 5, required: 10 });
  assert(some[3] > none[3], 'more materials -> higher availability');
  assertEqual(makeObservation({ slotActive: false })[4], 0);
  assertEqual(makeObservation({ slotActive: true })[4], 1);
});

test('inputs: decodeBatch maps network output to 1..MAX_BATCH', () => {
  assertEqual(decodeBatch([0]), 1);
  assertEqual(decodeBatch([1]), MAX_BATCH);
  assertEqual(decodeBatch([0.5]), 11);
});

test('inputs: decodeBatch clamps and tolerates non-finite/scalar output', () => {
  assertEqual(decodeBatch([-5]), 1);
  assertEqual(decodeBatch([99]), MAX_BATCH);
  assertEqual(decodeBatch([NaN]), 1);
  assertEqual(decodeBatch(1), MAX_BATCH, 'scalar output accepted');
});

test('inputs: decodeBatch honors a custom maxBatch and stays integral', () => {
  assertEqual(decodeBatch([1], 5), 5);
  assertEqual(decodeBatch([0], 5), 1);
  const b = decodeBatch([0.37], 7);
  assert(Number.isInteger(b) && b >= 1 && b <= 7, `batch must be int in [1,7], got ${b}`);
});
