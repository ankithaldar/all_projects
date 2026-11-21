// tests/f1-curriculum.mjs — F1 curriculum regression / falsifier suite.
//
// Proves the f1-01/f1-02 fix in js/evolution.js: deterministic re-diversification
// at each track-rotation boundary escapes the oval-specialist adaptation trap,
// `selectBest` breaks exact-distance ties on already-persisted fields only, and
// `resumeFromBest` threads the target generation explicitly at BOTH spawn call
// sites (security C1). Loads the app's REAL classic-script modules via the same
// guarded-CommonJS pattern as tests/smoke.mjs — no module under js/ is edited.
//
// Run:  node tests/f1-curriculum.mjs   (from this directory or any cwd)
// Exit: 0 when every leg passes, 1 otherwise. No silent skips.
//
// NOTE: this is a SEPARATE file; tests/smoke.mjs stays at 76 legs.

import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const PROJECT = fileURLToPath(new URL('..', import.meta.url)); // .../0003_car_racer/

// Load order is load-bearing (car.js reads globalThis.Track at load;
// driver-sim.js reads the sensor/car globals; evolution.js reads all five).
const { NeuralNetwork, mulberry32, NN_ARCH } = require(PROJECT + 'js/nn.js');
const { Track, DEFAULT_CONTROL_POINTS } = require(PROJECT + 'js/track.js');
require(PROJECT + 'js/car.js');
require(PROJECT + 'js/sensors.js');
const { DriverSim, EVAL_CAP_S, LAP_BONUS_TRACK_LENGTHS } = require(PROJECT + 'js/driver-sim.js');
const {
  Evolution, POPULATION, MUTATION_RATE, MUTATION_SCALE,
} = require(PROJECT + 'js/evolution.js');
require(PROJECT + 'js/rotation.js'); // installs globalThis.Rotation (+ canonicalJSON/cyrb53)
const Rotation = globalThis.Rotation;

// ---------------------------------------------------------------------------
// Fixed fixtures from the diagnosis (plan §43, flaw report Exp B).
// ---------------------------------------------------------------------------
const SEED = 4242;
const DEF_GENS = 50;
const F1_GENS = 200;                // gens 51..250 inclusive
const F1_HASH = '9a44a5c9dadd0597';
const F1_HALF_WIDTH = 49.75;
const THRESH_GEN250 = 38970;
const THRESH_LAST50_MEAN = 50000;
const KNOWN_RESULT_KEYS = ['crashed', 'distance', 'stuck', 'timeAlive', 'topSpeed'].sort();

function defaultTrack() {
  return new Track(DEFAULT_CONTROL_POINTS, 70);
}

function f1Track() {
  const resolved = Rotation.resolveGeneratedTrack(SEED, 1, 2);
  assert.equal(resolved.trackHash, F1_HASH, 'F1 track hash must be ' + F1_HASH);
  assert.equal(resolved.halfWidth, F1_HALF_WIDTH, 'F1 halfWidth must be ' + F1_HALF_WIDTH);
  return new Track(resolved.controlPoints, resolved.halfWidth);
}

// Null-safe normalization: false / 0 / undefined all mean "not crashed".
const notCrashed = (v) => !v;

const W = (net) => JSON.stringify(net.getWeights());

function constantNet(v) {
  return NeuralNetwork.fromJSON({
    architecture: NN_ARCH,
    w1: new Array(112).fill(v), b1: new Array(16).fill(v),
    w2: new Array(256).fill(v), b2: new Array(16).fill(v),
    w3: new Array(32).fill(v), b3: new Array(2).fill(v),
  });
}

function makeResults(base, overrides) {
  const arr = [];
  for (let i = 0; i < POPULATION; i++) arr.push({ ...base });
  for (const idx of Object.keys(overrides)) {
    arr[Number(idx)] = { ...arr[Number(idx)], ...overrides[idx] };
  }
  return arr;
}

// Drive one net (or policyOverride) to terminal state via the REAL DriverSim,
// so we can read sim.laps (Evolution.evaluateDriver surfaces fitness/flags but
// deliberately not the raw lap counter).
function drive(track, net, policyOverride) {
  const sim = new DriverSim(track, net, policyOverride);
  for (;;) if (sim.step().done) break;
  return {
    laps: sim.laps,
    crashed: sim.crashed,
    stuck: sim.stuck,
    outcome: sim.outcome,
    distance: sim.fitness(),
    timeAlive: sim.timeAlive,
  };
}

// Deterministically compute the expected children from a start rng cursor and
// target generation using the SAME two constructor paths the spec mandates.
function expectedSpawn(parent, startState, nextGeneration) {
  // v3: the ONLY restart boundary is target 51 — all 20 children are fresh
  // new NeuralNetwork(rng) nets (20 × 800 = 16,000 draws), elite clone dropped.
  // Every other target (EXPLICITLY including 101/151/201) is an ordinary mutant
  // spawn: child0 = exact elite clone (zero draws), children 1..19 = mutants.
  const boundary = nextGeneration === 51;
  const rng = mulberry32(0);
  rng.setState(startState);
  const out = [];
  if (boundary) {
    for (let i = 0; i < POPULATION; i++) out.push(new NeuralNetwork(rng)); // fresh, 800 draws each
  } else {
    out.push(parent.clone()); // child0 = exact elite clone, zero draws
    for (let i = 1; i < POPULATION; i++) {
      out.push(parent.clone().mutate(MUTATION_RATE, MUTATION_SCALE, rng));
    }
  }
  return { out, finalState: rng.state(), boundary };
}

// ---------------------------------------------------------------------------
// Tiny synchronous test harness — never throws out of a leg.
// ---------------------------------------------------------------------------
let legsPassed = 0;
const failures = [];

function leg(name, fn) {
  try {
    fn();
    legsPassed += 1;
    console.log('  ok   — ' + name);
  } catch (err) {
    failures.push({ name, err });
    console.log('  FAIL — ' + name);
    console.log('         ' + String((err && err.message) || err).split('\n').join('\n         '));
  }
}

function mean(xs) {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

// Populated by the curriculum leg; consumed by the real-driving leg.
let trainedPopulation = null;
const curriculum = { bestByGen: [], resumeChecks: [], f1TrackObj: null };

console.log('=== F1 curriculum regression / falsifier suite (seed ' + SEED + ') ===');

// ---------------------------------------------------------------------------
// Leg 1 — fixed F1 track identity (the exact diagnosis fixture).
// ---------------------------------------------------------------------------
leg('fixed F1 track identity: hash 9a44a5c9dadd0597, halfWidth 49.75', () => {
  const t = f1Track();
  assert.ok(t.length > 1900 && t.length < 2100, 'F1 track length sanity, got ' + t.length);
  assert.equal(EVAL_CAP_S, 45, 'EVAL_CAP_S must be 45s');
  assert.equal(LAP_BONUS_TRACK_LENGTHS, 2, 'lap bonus must be 2 track-lengths');
  curriculum.f1TrackObj = t;
});

// ---------------------------------------------------------------------------
// Leg 2 — reproduce Exp B with the fixed code and assert recovery. Also take
// byte-identity snapshots at each boundary of the uninterrupted run so C1 is
// proven on REAL evolved weights (not only fabricated fixtures).
// ---------------------------------------------------------------------------
leg('curriculum gens 1-50 default + 51-250 F1: recovery thresholds + real-run boundary resume', () => {
  const defTrack = defaultTrack();
  const f1 = curriculum.f1TrackObj || f1Track();
  curriculum.f1TrackObj = f1;
  const evo = new Evolution(SEED);
  evo.randomPopulation();

  const bestByGen = [];
  for (let g = 1; g <= DEF_GENS + F1_GENS; g++) {
    const track = g <= DEF_GENS ? defTrack : f1;
    const { results, bestIndex } = evo.evaluatePopulation(track); // REAL modules
    bestByGen.push(results[bestIndex].distance);

    // C1 on the REAL lineage: capture bestWeights + rngState at the END of gen N
    // (N in {50,100,150,200}), BEFORE the spawn that produces target N+1. Store
    // the uninterrupted post-spawn population/cursor as raw snapshots; Leg 6
    // performs the resumeFromBest(byte identity) asserts (consolidated, no 10th leg).
    if ((g % 50) === 0 && g < DEF_GENS + F1_GENS) {
      const weights = evo.bestWeights();
      const state = evo.rngState();
      evo.spawnNextGeneration(); // the uninterrupted path (target g+1)
      const uninterrupted = evo.population.map(W);
      const finalState = evo.rngState();
      curriculum.resumeChecks.push({
        completed: g,          // N
        target: g + 1,         // N+1 in {51,101,151,201}
        weights,
        state,
        uninterrupted,
        finalState,
      });
    } else if (g < DEF_GENS + F1_GENS) {
      evo.spawnNextGeneration();
    }
  }
  trainedPopulation = evo.population;
  curriculum.bestByGen = bestByGen;

  const at250 = bestByGen[249];
  const first50 = mean(bestByGen.slice(50, 100));   // gens 51..100
  const last50 = mean(bestByGen.slice(200, 250));   // gens 201..250

  console.log('         gen51 best = ' + bestByGen[50].toFixed(1) +
              ' (fresh nets; NOT a recovery baseline)');
  console.log('         gen250 best = ' + at250.toFixed(1) + ' (threshold >= ' + THRESH_GEN250 + ')');
  console.log('         mean best gens 51-100  = ' + first50.toFixed(1));
  console.log('         mean best gens 201-250 = ' + last50.toFixed(1) +
              ' (threshold >= ' + THRESH_LAST50_MEAN + ')');

  assert.ok(at250 >= THRESH_GEN250,
    'gen250 best fitness ' + at250.toFixed(1) + ' < ' + THRESH_GEN250);
  assert.ok(last50 >= THRESH_LAST50_MEAN,
    'last-50 mean ' + last50.toFixed(1) + ' < ' + THRESH_LAST50_MEAN);
  assert.ok(last50 > first50,
    'last-50 mean ' + last50.toFixed(1) + ' must exceed first-50 mean ' + first50.toFixed(1));
});

// ---------------------------------------------------------------------------
// Leg 3 — real driving: at least one trained driver completes a full lap on
// the R1 (generated F1) track without crashing. Null-safe !crashed.
// ---------------------------------------------------------------------------
leg('real driving on R1 track: >=1 driver with laps >= 1 AND !crashed', () => {
  assert.ok(trainedPopulation && trainedPopulation.length === POPULATION,
    'curriculum leg must produce a trained population');
  const f1 = curriculum.f1TrackObj || f1Track();
  let cleanLaps = 0;
  let maxLaps = 0;
  for (const net of trainedPopulation) {
    const r = drive(f1, net);
    if (r.laps > maxLaps) maxLaps = r.laps;
    if (r.laps >= 1 && notCrashed(r.crashed)) cleanLaps += 1;
  }
  console.log('         clean-lap drivers = ' + cleanLaps + '/' + POPULATION + ', max laps = ' + maxLaps);
  assert.ok(cleanLaps >= 1,
    'no driver achieved laps >= 1 with !crashed (max laps ' + maxLaps + ')');
});

// ---------------------------------------------------------------------------
// Leg 4 — falsifier: a stationary / wall-hugging policy cannot satisfy leg 3.
// Uses Evolution.evaluateDriver with policyOverride (the public evaluator),
// plus DriverSim for the raw lap counter.
// ---------------------------------------------------------------------------
leg('falsifier: stationary and wall-hugging policies cannot complete a clean lap', () => {
  const f1 = curriculum.f1TrackObj || f1Track();
  const lapBonus = LAP_BONUS_TRACK_LENGTHS * f1.length;

  const stationary = () => [0, 0];
  const viaEval = Evolution.evaluateDriver(f1, null, stationary); // REAL public API
  const viaSim = drive(f1, null, stationary);
  console.log('         stationary: distance=' + viaEval.distance.toFixed(1) +
              ' crashed=' + viaEval.crashed + ' stuck=' + viaEval.stuck +
              ' laps=' + viaSim.laps + ' (one-lap bonus=' + lapBonus.toFixed(0) + ')');
  assert.ok(viaEval.distance < lapBonus,
    'stationary policy distance must be below a full lap bonus');
  assert.equal(viaSim.laps, 0, 'stationary policy must have laps === 0');
  assert.equal(viaSim.outcome, 'stuck', 'stationary policy is killed by the stagnation timer');
  assert.ok(!(viaSim.laps >= 1 && notCrashed(viaSim.crashed)),
    'stationary policy must NOT satisfy criterion 3');
  assert.equal(typeof viaEval.crashed, 'boolean', 'evaluateDriver.crashed must be boolean');

  // Wall hugger: full gas, constant hard steer into the boundary.
  const hug = () => [1, 1];
  const hugEval = Evolution.evaluateDriver(f1, null, hug);
  const hugSim = drive(f1, null, hug);
  console.log('         wall-hug:   distance=' + hugEval.distance.toFixed(1) +
              ' crashed=' + hugEval.crashed + ' laps=' + hugSim.laps);
  assert.equal(hugSim.laps, 0, 'wall-hugging policy must have laps === 0');
  assert.ok(!(hugSim.laps >= 1 && notCrashed(hugSim.crashed)),
    'wall-hugging policy must NOT satisfy criterion 3');
});

// ---------------------------------------------------------------------------
// Leg 5 — v3 boundary predicate: ONE-TIME restart at 51 only (all 20 fresh
// nets, NO elite clone, 20×800 = 16,000 draws). 101/151/201 are ordinary
// mutant generations (exact elite clone + 19 mutants), as are 50/52/100/102.
// Byte-compare each child against the mandated constructor path.
// ---------------------------------------------------------------------------
leg('one-time restart boundary set = {51} only; 101/151/201 (and 50/52/100/102) stay mutated', () => {
  const parent = constantNet(0.9);
  const parentW = W(parent);
  // The single full-restart boundary: 20 fresh nets, NO elite clone.
  for (const g of [51]) {
    const evo = new Evolution(12345);
    const start = evo.rngState();
    const exp = expectedSpawn(parent, start, g);
    assert.equal(exp.boundary, true, 'test oracle: ' + g + ' is the one-time restart boundary');
    const actual = evo._spawnChildren(parent, g);
    for (let i = 0; i < POPULATION; i++) {
      assert.equal(W(actual[i]), W(exp.out[i]),
        'g=51 child ' + i + ' must be a FRESH rng net (full restart, no elite clone)');
    }
    assert.notEqual(W(actual[0]), parentW, 'g=51 child0 must NOT be the elite clone');
    assert.equal(evo.rngState(), exp.finalState, 'g=51 consumes 20*800 = 16000 draws');
  }
  // 101/151/201 were boundaries in v2; in v3 they are ordinary mutant spawns.
  for (const g of [101, 151, 201]) {
    const evo = new Evolution(12345);
    const start = evo.rngState();
    const exp = expectedSpawn(parent, start, g);
    assert.equal(exp.boundary, false, 'test oracle: ' + g + ' is NOT a restart boundary in v3');
    const actual = evo._spawnChildren(parent, g);
    assert.equal(W(actual[0]), parentW, 'g=' + g + ' child0 is exact elite clone');
    for (let i = 1; i < POPULATION; i++) {
      assert.equal(W(actual[i]), W(exp.out[i]),
        'g=' + g + ' child ' + i + ' must be a mutated clone (no re-diversification)');
    }
    assert.equal(evo.rngState(), exp.finalState, 'g=' + g + ' consumes the old mutation draws');
  }
  // Generations that were never restart candidates remain ordinary.
  for (const g of [50, 52, 100, 102]) {
    const evo = new Evolution(12345);
    const start = evo.rngState();
    const exp = expectedSpawn(parent, start, g);
    assert.equal(exp.boundary, false, 'test oracle: ' + g + ' is NOT a boundary');
    const actual = evo._spawnChildren(parent, g);
    assert.equal(W(actual[0]), parentW, 'g=' + g + ' child0 is exact elite clone');
    for (let i = 1; i < POPULATION; i++) {
      assert.equal(W(actual[i]), W(exp.out[i]),
        'g=' + g + ' child ' + i + ' must be a mutated clone (no re-diversification)');
    }
    assert.equal(evo.rngState(), exp.finalState, 'g=' + g + ' consumes the old mutation draws');
  }
});

// ---------------------------------------------------------------------------
// Leg 6 — security C1: resume must reproduce the uninterrupted population
// byte-for-byte. At 50→51 (the ONE-TIME fresh-restart boundary) omitting the
// explicit target would spawn mutated clones => this fails. At 100→101 an
// ORDINARY mutant generation must also resume byte-identically. The same
// byte-identity proof is consolidated here for the four REAL-lineage boundaries
// {50,100,150,200} captured by Leg 2 (no 10th leg).
// ---------------------------------------------------------------------------
leg('resume byte-identity: 50->51 fresh restart + 100->101 ordinary + real-lineage {50,100,150,200}', () => {
  // Fabricated pins (constantNet(0.9)) at completed 50 and 100.
  for (const completed of [50, 100]) {
    const target = completed + 1;
    const parent = constantNet(0.9);
    const weights = parent.getWeights();
    const evo = new Evolution(4242);
    evo.randomPopulation();
    evo.population[0] = parent.clone();
    evo.generation = completed;   // only target 51 is the fresh-restart boundary; 101 is ordinary
    evo.bestIndex = 0;
    const state = evo.rngState();

    evo.spawnNextGeneration();
    assert.equal(evo.generation, target);
    const uninterrupted = evo.population.map(W);
    const finalState = evo.rngState();

    const resumed = Evolution.resumeFromBest(weights, completed, state);
    assert.equal(resumed.generation, target, 'resumed generation must be ' + target);
    assert.deepEqual(resumed.population.map(W), uninterrupted,
      'resumed population must be byte-identical to uninterrupted run at target ' + target);
    assert.equal(resumed.rngState(), finalState, 'resumed rng cursor must match uninterrupted run');

    if (target === 51) {
      // A1: child0 is the FIRST fresh net; child1 is the SECOND fresh net.
      // The old pin compared population[1] to the FIRST fresh net — draw child0
      // first from the SAME seeded stream, then the expected child1.
      const freshRng = mulberry32(0);
      freshRng.setState(state);
      const child0 = new NeuralNetwork(freshRng); // FIRST fresh net (elite clone dropped)
      const child1 = new NeuralNetwork(freshRng);  // SECOND fresh net
      assert.notEqual(W(child0), W(child1), 'oracle sanity: the two fresh nets differ');
      assert.equal(W(resumed.population[0]), W(child0),
        'resumed child0 must be the FIRST fresh net at the 50->51 restart');
      assert.equal(W(resumed.population[1]), W(child1),
        'resumed child1 must be the SECOND fresh net, not a mutated clone');
    } else {
      // target 101: ORDINARY mutant generation — exact elite clone + 19 mutants.
      const mutRng = mulberry32(0);
      mutRng.setState(state);
      const expectedChild1 = parent.clone().mutate(MUTATION_RATE, MUTATION_SCALE, mutRng);
      assert.equal(W(resumed.population[0]), W(parent),
        'resumed child0 must be the exact elite clone at target 101');
      assert.equal(W(resumed.population[1]), W(expectedChild1),
        'resumed child1 must be a mutated clone at target 101');
    }
  }

  // Four REAL-lineage checks consolidated from Leg 2 (completed 50/100/150/200).
  assert.equal(curriculum.resumeChecks.length, 4,
    'four real-lineage resume snapshots from targets 51/101/151/201');
  for (const c of curriculum.resumeChecks) {
    const resumed = Evolution.resumeFromBest(c.weights, c.completed, c.state);
    assert.deepEqual(resumed.population.map(W), c.uninterrupted,
      'real-lineage population byte-identity at completedGeneration=' + c.completed);
    assert.equal(resumed.rngState(), c.finalState,
      'real-lineage rng cursor equality at completedGeneration=' + c.completed);
    assert.equal(resumed.generation, c.target,
      'real-lineage generation equality at completedGeneration=' + c.completed);
  }
});

// ---------------------------------------------------------------------------
// Leg 7 — tiebreak properties on fabricated EXACT distance ties + security C2
// (every REAL result passed to selectBest carries a boolean crashed).
// ---------------------------------------------------------------------------
leg('tiebreak: non-crashed wins, higher timeAlive wins, unequal distance never reorders; C2 boolean crashed', () => {
  // (a) non-crashed beats crashed at equal distance (crashed one has higher topSpeed).
  const a = makeResults({ distance: 1000, topSpeed: 1, timeAlive: 1, crashed: true }, {
    0: { crashed: true, timeAlive: 45, topSpeed: 999 },
    1: { crashed: false, timeAlive: 1, topSpeed: 1 },
  });
  assert.equal(Evolution.selectBest(a), 1, 'non-crashed must beat crashed at equal distance');

  // (b) equal distance + equal crash status: higher timeAlive wins.
  const b = makeResults({ distance: 1000, topSpeed: 100, timeAlive: 1, crashed: false }, {
    5: { timeAlive: 30 },
  });
  assert.equal(Evolution.selectBest(b), 5, 'higher timeAlive must win an exact tie');

  // (c) unequal distances never reorder, even when the max-distance driver crashed.
  const c = makeResults({ distance: 500, topSpeed: 999, timeAlive: 45, crashed: false }, {
    0: { distance: 2000, crashed: true, timeAlive: 1, topSpeed: 1 },
    10: { distance: 1000, crashed: false, timeAlive: 45, topSpeed: 999 },
  });
  assert.equal(Evolution.selectBest(c), 0, 'max distance always wins regardless of tiebreak fields');
  const c2 = [];
  for (let i = 0; i < POPULATION; i++) {
    c2.push({ distance: i * 7 + (i % 2), topSpeed: 1000 - i, timeAlive: 45 - i, crashed: i % 3 === 0 });
  }
  const p2 = Evolution.selectBest(c2);
  assert.equal(c2[p2].distance, Math.max(...c2.map((x) => x.distance)), 'primary distance ordering intact');

  // (d) null-safe crash normalization: 0 / false / undefined are "not crashed".
  const d = makeResults({ distance: 1000, topSpeed: 1, timeAlive: 1, crashed: 1 }, {
    0: { crashed: 1, timeAlive: 45, topSpeed: 999 },
    1: { crashed: 0, timeAlive: 1, topSpeed: 1 },
    2: { crashed: undefined, timeAlive: 1, topSpeed: 1 },
    3: { crashed: false, timeAlive: 1, topSpeed: 1 },
  });
  const pd = Evolution.selectBest(d);
  assert.ok([1, 2, 3].includes(pd), 'null-safe !crashed must pick a non-crashed driver, got ' + pd);

  // C2: spy on selectBest while the REAL evaluation path feeds it, then
  // assert every passed result object carries a BOOLEAN crashed field.
  const original = Evolution.selectBest;
  const seen = [];
  Evolution.selectBest = function (results) { seen.push(results); return original.call(this, results); };
  try {
    const evo = new Evolution(SEED);
    evo.randomPopulation();
    evo.evaluatePopulation(defaultTrack()); // recordEvaluation -> selectBest(results)
  } finally {
    Evolution.selectBest = original;
  }
  assert.ok(seen.length >= 1, 'selectBest must be exercised by evaluatePopulation');
  for (const results of seen) {
    for (const r of results) {
      assert.ok(Object.prototype.hasOwnProperty.call(r, 'crashed'),
        'every result passed to selectBest must carry a crashed field');
      assert.equal(typeof r.crashed, 'boolean',
        'crashed must be boolean (undefined = failure) on the real evaluation path');
    }
  }
});

// ---------------------------------------------------------------------------
// Leg 8 — ranking invariant: reproduce smoke leg 12's fixture verbatim.
// ---------------------------------------------------------------------------
leg('ranking invariant: smoke leg-12 fixture still selects driver #7', () => {
  const results = [];
  for (let i = 0; i < POPULATION; i++) {
    results.push({
      distance: i === 7 ? 10000 : 100 + i, // 100..106, 10000, 108..119
      topSpeed: i === 7 ? 500 : 100,
      timeAlive: i === 7 ? 45 : 30,
      crashed: false,
      stuck: false,
    });
  }
  assert.equal(Evolution.selectBest(results), 7, 'selectBest must pick #7 (max-distance driver)');
});

// ---------------------------------------------------------------------------
// Leg 9 — no selection-only field reaches the DB adapter: results keep exactly
// the five known fields after the selection path.
// ---------------------------------------------------------------------------
leg('result objects contain only the five known fields (no selection-only score)', () => {
  const evo = new Evolution(SEED);
  evo.randomPopulation();
  const { results } = evo.evaluatePopulation(defaultTrack());
  assert.equal(results.length, POPULATION);
  for (const r of results) {
    assert.deepEqual(Object.keys(r).sort(), KNOWN_RESULT_KEYS,
      'result field set must be exactly distance/topSpeed/timeAlive/crashed/stuck');
  }
});

// ---------------------------------------------------------------------------
console.log('');
console.log('legs passed: ' + legsPassed + ' / ' + (legsPassed + failures.length));
if (failures.length) {
  console.log('F1-CURRICULUM FAIL — ' + failures.length + ' leg(s) failed:');
  for (const f of failures) console.log('  - ' + f.name);
  process.exit(1);
}
console.log('F1-CURRICULUM OK — all ' + legsPassed + ' legs passed');
process.exit(0);
