/**
 * adapter.js — thin adapter over the validated in-house canonical NEAT engine.
 *
 * Phase 0 (docs/neat-spike.md §3.7) proved the in-house module implements the
 * five required methods verbatim; the two extras (`activate`, `reset`) are
 * required for black-box rollouts. The vendored @stsoftware bundle is NOT
 * imported — its top-level init hangs over http://.
 *
 * This adapter is a DELIBERATE STABLE SEAM: the rest of the app (controller,
 * worker, tests) depends only on this small interface, never on inhouse-neat's
 * internals. If the evolution engine is later swapped, only this file changes.
 */
import { createInhouseNeat } from './inhouse-neat.js';

export function createNeatAdapter({ inputs, outputs, seed, config } = {}) {
  const eng = createInhouseNeat({ inputs, outputs, seed, config });

  const adapter = {
    inputs: eng.inputs,
    outputs: eng.outputs,
    seed: eng.seed,
    config: eng.config,

    evaluatePopulation: (fitnessFn) => eng.evaluatePopulation(fitnessFn),
    getBestGenome: () => eng.getBestGenome(),
    exportGenome: (g) => eng.exportGenome(g),
    importGenome: (g) => eng.importGenome(g),
    cloneFrozen: (g) => eng.cloneFrozen(g),

    // Required in addition for fitness.js / controller.js rollouts:
    activate: (g, obs) => eng.activate(g, obs),
    /** Reset the underlying engine; returns the adapter for chaining. */
    reset: (newSeed) => { eng.reset(newSeed); return adapter; }
  };

  return adapter;
}

export default createNeatAdapter;
