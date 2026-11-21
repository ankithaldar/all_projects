// tests/benchmark.mjs — M2 throughput probe for the headless eval path.
//
// Print-only. NO fixed thresholds, NO pass/fail — machines differ and this is
// a measurement, not a gate. Exit code is always 0. NOT a smoke leg.
//
// Measures the raw evaluateDriver throughput on the DEFAULT track in node:
//   * runs until ≥ 2_000_000 ticks have been simulated (or MAX_DRIVERS hit),
//     tallying ticks as round(timeAlive / SIM_DT) per driver;
//   * reports ticks/s, µs/tick, observed average sim-time per driver and the
//     crash/stuck/finished/time-cap mix;
//   * projects the wall-clock for a 5000-generation run two ways:
//       worst-case  — every driver in every generation hits EVAL_CAP_S (45 s)
//       observed    — drivers average what the sample averaged here
//   * states explicitly that the per-persist cost is EXCLUDED here (measured
//     separately since M3 via Db.getPersistCost() — see smoke leg 27).
//   * prints the nearestCenterline complexity note (O(#samples) per tick).
//
// Run:  node tests/benchmark.mjs

import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);

require('../js/driver-sim.js');
require('../js/nn.js');
require('../js/track.js');
require('../js/car.js');
require('../js/sensors.js');
const { SIM_DT, EVAL_CAP_S, POPULATION } = require('../js/evolution.js');
const { DriverSim } = require('../js/driver-sim.js');
const { NeuralNetwork, mulberry32 } = require('../js/nn.js');
const { Track, DEFAULT_CONTROL_POINTS } = require('../js/track.js');

const TARGET_TICKS = 2_000_000;
const MAX_DRIVERS = 40000;

const track = new Track(DEFAULT_CONTROL_POINTS, 70);
const rng = mulberry32(20261116);

const t0 = performance.now();

let ticks = 0;
let drivers = 0;
let crashed = 0;
let stuck = 0;
let finished = 0;
let timeCap = 0;
let survivors = 0;
let sumSimS = 0;
let topSpeedSum = 0;

while (ticks < TARGET_TICKS && drivers < MAX_DRIVERS) {
  const net = NeuralNetwork.randomInit(rng);
  // DriverSim directly (not Evolution.evaluateDriver) so the outcome bucket
  // can count 'finished' / 'time-cap' explicitly (informational only).
  const sim = new DriverSim(track, net);
  for (;;) { if (sim.step().done) break; }
  const simTicks = Math.round(sim.timeAlive / SIM_DT);
  ticks += simTicks;
  drivers += 1;
  sumSimS += sim.timeAlive;
  topSpeedSum += sim.topSpeed();
  if (sim.outcome === 'crashed') crashed += 1;
  else if (sim.outcome === 'stuck') stuck += 1;
  else if (sim.outcome === 'finished') finished += 1;
  else if (sim.outcome === 'time-cap') timeCap += 1;
  if (!sim.crashed && !sim.stuck) survivors += 1;
}

const t1 = performance.now();
const wallS = (t1 - t0) / 1000;
const ticksPerSec = ticks / wallS;
const usPerTick = (wallS * 1e6) / ticks;
const avgSimS = sumSimS / drivers;
const avgTopSpeed = topSpeedSum / drivers;
const simTicksPerDriver = avgSimS / SIM_DT;

const TICKS_PER_GEN_WORST = POPULATION * EVAL_CAP_S * (1 / SIM_DT);
const TICKS_PER_GEN_AVG = POPULATION * simTicksPerDriver;
const genWallWorst = TICKS_PER_GEN_WORST / ticksPerSec;
const genWallAvg = TICKS_PER_GEN_AVG / ticksPerSec;

const fmt = (n, digits) => n.toFixed(digits === undefined ? 1 : digits);
const fmtHMS = (seconds) => {
  if (seconds < 90) return fmt(seconds, 1) + ' s';
  const m = seconds / 60;
  if (m < 90) return fmt(m, 1) + ' min';
  return fmt(m / 60, 2) + ' h';
};
const pct = (n) => fmt((n / drivers) * 100, 1) + '%';

console.log('=== 0003_car_racer M2 benchmark (default track, seed 20261116, node ' + process.version + ') ===');
console.log('');
console.log('track: ' + track.samples.length + ' samples, ' + track.segments.length + ' wall segments, halfWidth ' + track.halfWidth + ' px, arc length ' + fmt(track.length, 1) + ' px');
console.log('');
console.log('sample:');
console.log('  ticks simulated      : ' + ticks);
console.log('  drivers evaluated    : ' + drivers);
console.log('  wall time            : ' + fmt(wallS, 2) + ' s');
console.log('  throughput           : ' + Math.round(ticksPerSec) + ' ticks/s  (' + fmt(usPerTick, 2) + ' µs/tick)');
console.log('  avg sim-time/driver  : ' + fmt(avgSimS, 2) + ' s  (≈ ' + Math.round(simTicksPerDriver) + ' ticks)');
console.log('  avg top speed        : ' + fmt(avgTopSpeed, 1) + ' px/s');
console.log('  outcomes             : crashed ' + pct(crashed) + ', stuck ' + pct(stuck) + ', finished ' + pct(finished) + ', time-cap ' + pct(timeCap) + ', not-crashed-or-stuck ' + pct(survivors));
console.log('');
console.log('5000-generation projection (' + POPULATION + ' drivers/gen, ' + (1 / SIM_DT) + ' ticks/s-sim, excludes worker startup):');
console.log('  worst-case  (all drivers hit EVAL_CAP_S=' + EVAL_CAP_S + 's) : ' + TICKS_PER_GEN_WORST * 5000 + ' ticks ≈ ' + fmtHMS(TICKS_PER_GEN_WORST * 5000 / ticksPerSec));
console.log('  observed avg (this sample)               : ≈ ' + Math.round(TICKS_PER_GEN_AVG * 5000) + ' ticks ≈ ' + fmtHMS(TICKS_PER_GEN_AVG * 5000 / ticksPerSec));
console.log('  per-generation wall-clock at observed avg : ' + fmtHMS(genWallAvg) + '  (worst-case ' + fmtHMS(genWallWorst) + ')');
console.log('');
console.log('persist cost: EXCLUDED this milestone (no DB); M2 limitation, recorded in status.md');
console.log('');
console.log('nearestCenterline note: O(#samples = ' + track.samples.length + ') nearest-sample scan per tick (input + liveness share ONE call per tick)');
console.log('  ≈ ' + fmt((track.samples.length * ticks) / 1e9, 2) + ' G sample-comparisons for this run; grid raycast (DDA, 64 px cells) handles the 5 sensor rays');
console.log('  no O(n²) growth: per-tick cost is constant in track length (~samples + ~segments per collision scan ≈ ' + (track.samples.length + track.segments.length) + ' ops/tick core)');
process.exit(0);
