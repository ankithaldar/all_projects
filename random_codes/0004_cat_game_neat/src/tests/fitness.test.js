// Tests for src/neat/fitness.js — coins-first, time-second lexicographic fitness.
import { test, assert, assertEqual, assertDeepEqual } from './harness.js';
import {
  fitness, BIG, INFEASIBLE_BASE, progressBonus, trainingFitness, enrichResult, FITNESS_CLAMP
} from '../neat/fitness.js';

test('fitness: feasible lower totalSpent wins regardless of completionTick', () => {
  const cheapSlow = fitness({ feasible: true, totalSpent: 100, completionTick: 8000 });
  const expensiveFast = fitness({ feasible: true, totalSpent: 101, completionTick: 10 });
  assert(cheapSlow > expensiveFast,
    `cheap/slow (${cheapSlow}) must outrank expensive/fast (${expensiveFast})`);
});

test('fitness: feasible exact formula uses explicit totalSpent accumulator', () => {
  assertEqual(fitness({ feasible: true, totalSpent: 0, completionTick: 0 }), -(0 * BIG + 0));
  assertEqual(fitness({ feasible: true, totalSpent: 3, completionTick: 17 }), -(3 * BIG + 17));
});

test('fitness: completionTick is only a tie-breaker when totalSpent is equal', () => {
  const early = fitness({ feasible: true, totalSpent: 50, completionTick: 100 });
  const late = fitness({ feasible: true, totalSpent: 50, completionTick: 9000 });
  assert(early > late, `equal spend: earlier tick (${early}) must outrank later (${late})`);
});

test('fitness: infeasible always ranks below any feasible rollout', () => {
  const inf = fitness({ feasible: false, totalSpent: 0, completionTick: null });
  const feas = fitness({ feasible: true, totalSpent: 999999, completionTick: 8063 });
  assert(feas > inf, `feasible (${feas}) must outrank infeasible (${inf})`);
});

test('fitness: infeasible base is -1e12 plus a bounded progress bonus', () => {
  const noProgress = fitness({ feasible: false });
  assertEqual(noProgress, INFEASIBLE_BASE);
  const partial = fitness({
    feasible: false,
    targets: { a: 4, b: 2 },
    stash: { a: 2, b: 2 }
  });
  // progressBonus = min(2,4)/4 + min(2,2)/2 = 0.5 + 1 = 1.5
  assertEqual(partial, INFEASIBLE_BASE + 1.5);
});

test('fitness: progressBonus clamps over-production and ignores non-positive targets', () => {
  const bonus = progressBonus({ targets: { a: 2, b: 3, z: 0 }, stash: { a: 99, b: 0 } });
  assertEqual(bonus, 1.0); // a: min(99,2)/2=1, b: 0/3=0, z ignored
});

test('fitness: BIG exceeds the whole tick horizon for strict lexicographic ordering', () => {
  assert(BIG > 8064, `BIG (${BIG}) must exceed max ticks (8064)`);
});

test('fitness: never derives spend from an initial-vs-final coin delta', () => {
  // Same rollout but different coin deltas: fitness must depend ONLY on totalSpent.
  const a = fitness({ feasible: true, totalSpent: 200, completionTick: 500, initialCoins: 100, finalCoins: 0 });
  const b = fitness({ feasible: true, totalSpent: 200, completionTick: 500, initialCoins: 900000, finalCoins: 899500 });
  assertEqual(a, b);
});

// ---------------------------------------------------------------------------
// Bounded two-phase training fitness (selection signal only)
// ---------------------------------------------------------------------------

test('trainingFitness: enrichResult carries targets/mfr/maxTicks/stash/produced', () => {
  const r = { feasible: false, produced: { a: 2 }, totalSpent: 5 };
  const e = enrichResult(r, { a: 4 }, { a: 8 }, 8064);
  assertDeepEqual(e.targets, { a: 4 });
  assertDeepEqual(e.mfr, { a: 8 });
  assertEqual(e.maxTicks, 8064);
  assertDeepEqual(e.stash, { a: 2 });
  assertDeepEqual(e.produced, { a: 2 });
  assertEqual(r.produced.a, 2, 'enrichResult must not mutate the source');
});

test('trainingFitness: feasible (phase or result) returns the shipped lexicographic score', () => {
  const result = { feasible: true, totalSpent: 100, completionTick: 500, targets: { a: 2 }, stash: { a: 2 } };
  assertEqual(trainingFitness(result, 'feasible', { a: 2 }, 8064), fitness(result));
  assertEqual(trainingFitness(result, 'pre', { a: 2 }, 8064), fitness(result));
});

test('trainingFitness: pre-feasible is bounded, finite and strictly below every feasible score', () => {
  const mfr = { a: 8, b: 4, c: 12 };
  const targets = { a: 8, b: 4 };
  // Worst (most negative) feasible scores possible inside the wallet/horizon.
  const feasibleScores = [
    fitness({ feasible: true, totalSpent: 0, completionTick: 0 }),
    fitness({ feasible: true, totalSpent: 7110975, completionTick: 6210 }),
    fitness({ feasible: true, totalSpent: 7692436, completionTick: 8063 })
  ];
  const minFeasible = Math.min(...feasibleScores);
  const cases = [
    { produced: {}, elapsedTicks: 0 },
    { produced: { a: 8, b: 4 }, elapsedTicks: 0 },
    { produced: { a: 99, b: 99, c: 99 }, elapsedTicks: 8063 },
    { produced: { a: NaN, b: Infinity, c: -5 }, elapsedTicks: NaN },
    { produced: null, elapsedTicks: Infinity, completionTick: null }
  ];
  for (const c of cases) {
    const result = { feasible: false, ...c, targets, stash: c.produced || {} };
    const f = trainingFitness(result, 'pre', mfr, 8064);
    assert(Number.isFinite(f), `trainingFitness must be finite: ${f}`);
    assert(f >= -FITNESS_CLAMP && f <= FITNESS_CLAMP, `must be clamped: ${f}`);
    assert(f < minFeasible, `pre-feasible ${f} must be below min feasible ${minFeasible}`);
  }
});

test('trainingFitness: progress terms are within [0,1] and increase monotonically', () => {
  const mfr = { a: 10 };
  const targets = { a: 10 };
  const score = (produced, elapsed) => trainingFitness(
    { feasible: false, produced: { a: produced }, stash: { a: produced }, elapsedTicks: elapsed, targets },
    'pre', mfr, 8064
  );
  const none = score(0, 8064);
  const half = score(5, 4000);
  const full = score(10, 0);
  assert(none < half && half < full, `shaped score must grow with progress: ${none} ${half} ${full}`);
  // Max possible shaped value is INFEASIBLE_BASE + 1000 + 100 + 10.
  assert(full <= INFEASIBLE_BASE + 1110, 'shaped score cannot exceed the bounded ceiling');
});

test('trainingFitness: guarded division for zero/non-finite quantities never yields NaN', () => {
  const mfr = { a: 0, b: -3, c: NaN, d: Infinity, e: 5 };
  const targets = { a: 0, b: -1, c: NaN, z: 0 };
  const result = { feasible: false, produced: { a: 5, b: 5, c: 5, d: 5, e: 5 }, stash: { a: 5, e: 5 }, targets, elapsedTicks: 0 };
  const f = trainingFitness(result, 'pre', mfr, 8064);
  assert(Number.isFinite(f));
  assert(f >= INFEASIBLE_BASE, 'zero/negative/non-finite caps contribute no progress, not NaN');
});

test('trainingFitness: invalid maxTicks falls back to the fixed 8064 horizon', () => {
  const result = { feasible: false, produced: {}, stash: {}, targets: {}, elapsedTicks: 0 };
  const f = trainingFitness(result, 'pre', {}, 0);
  assert(Number.isFinite(f));
  const e = enrichResult(result, {}, {}, -5);
  assertEqual(e.maxTicks, 8064);
});
