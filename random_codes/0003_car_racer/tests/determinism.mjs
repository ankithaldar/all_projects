// tests/determinism.mjs — M2 determinism probe for the training path.
//
// Builds the default track, seed 4242, and runs generations 1..3 EXACTLY the
// way the training path does — Evolution + evaluatePopulation, NO logging,
// NO DB — then prints JSON.stringify([gen1Weights, gen2Weights, gen3Weights])
// on stdout (no trailing newline, nothing else).
//
// Consumed by tests/smoke.mjs leg 13 as TWO fresh child processes whose
// stdout buffers are compared byte-for-byte (must match). It is also runnable
// standalone:  node tests/determinism.mjs
//
// Exit code is 0 on success; any failure prints to stderr and exits 1 (which
// makes the consuming smoke leg fail loudly rather than compare garbage).

import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);

require('../js/driver-sim.js');
require('../js/nn.js');
require('../js/track.js');
require('../js/car.js');
require('../js/sensors.js');
const { Evolution } = require('../js/evolution.js');
const { Track, DEFAULT_CONTROL_POINTS } = require('../js/track.js');

try {
  const track = new Track(DEFAULT_CONTROL_POINTS, 70);
  const evo = new Evolution(4242);
  evo.randomPopulation();

  const gens = [];
  for (let gen = 1; gen <= 3; gen++) {
    evo.evaluatePopulation(track);
    gens.push(evo.bestWeights());
    if (gen < 3) evo.spawnNextGeneration();
  }
  process.stdout.write(JSON.stringify(gens));
} catch (err) {
  process.stderr.write('determinism probe failed: ' + ((err && err.stack) || err) + '\n');
  process.exit(1);
}
