/**
 * inhouse-neat.js — self-contained canonical NEAT (NeuroEvolution of Augmenting Topologies).
 *
 * Fallback engine for the cat-game NEAT project. The vendored @stsoftware/neat-ai
 * browser bundle hangs on top-level initialisation when loaded over http://
 * (see docs/neat-spike.md), so this module provides the evolution substrate.
 *
 * Design constraints (from the Phase 0 spike):
 *  - Plain, structured-clone-safe genome objects (no methods, JSON round-trippable).
 *  - Fully synchronous and deterministic: one seeded PRNG drives every stochastic
 *    decision; same seed + same fitnessFn => byte-identical best genome.
 *  - Browser-safe: no Node/Deno globals, no workers, no WASM, no network, no timers.
 *  - Feed-forward topologies: an edge is only legal when depth(from) < depth(to).
 */

export const DEFAULT_CONFIG = Object.freeze({
  populationSize: 50,
  // Per-gene / per-bias mutation probabilities. weightRandomRate is checked first.
  weightRandomRate: 0.10,
  weightPerturbRate: 0.75,
  perturbStrength: 0.35,
  weightRange: 3,
  addConnectionRate: 0.06,
  addNodeRate: 0.04,
  activationRate: 0.01,
  // Structural caps keep rollout cost bounded.
  maxNodes: 128,
  maxGenes: 512,
  // Compatibility distance: c1*(disjoint+excess)/N + c3*meanWeightDelta.
  c1: 1.0,
  c3: 0.45,
  speciesThreshold: 2.6,
  elitism: 1,
  crossoverRate: 0.75
});

const ACTIVATIONS = Object.freeze({
  LOGISTIC: (x) => 1 / (1 + Math.exp(-x)),
  TANH: (x) => Math.tanh(x),
  RELU: (x) => (x > 0 ? x : 0),
  IDENTITY: (x) => x
});
const ACTIVATION_NAMES = Object.keys(ACTIVATIONS);

/** Deterministic mulberry32 PRNG with a serialisable state. */
export function createSeededRng(seed) {
  let a = seed >>> 0;
  const random = () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    random,
    int: (n) => Math.floor(random() * n),
    range: (lo, hi) => lo + (hi - lo) * random(),
    gaussian: () => {
      let u = 0, v = 0;
      while (u === 0) u = random();
      while (v === 0) v = random();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    },
    getState: () => a,
    setState: (s) => { a = s >>> 0; }
  };
}

function cloneNode(n) {
  return { id: n.id, type: n.type, activation: n.activation, bias: n.bias, depth: n.depth, frozen: !!n.frozen };
}
function cloneGene(g) {
  return {
    innovation: g.innovation, from: g.from, to: g.to,
    weight: g.weight, enabled: g.enabled !== false, frozen: !!g.frozen
  };
}
function cloneGenome(g) {
  return {
    nodes: g.nodes.map(cloneNode),
    genes: g.genes.map(cloneGene),
    frozen: !!g.frozen,
    fitness: g.fitness,
    adjustedFitness: g.adjustedFitness,
    speciesId: g.speciesId
  };
}

/**
 * Create a NEAT engine.
 * @param {{inputs:number, outputs:number, seed:number, populationSize?:number, config?:object}} options
 */
export function createInhouseNeat(options = {}) {
  const inputs = options.inputs | 0;
  const outputs = options.outputs | 0;
  if (!(inputs > 0) || !(outputs > 0)) throw new RangeError('inputs and outputs must be positive integers');
  const config = Object.freeze({ ...DEFAULT_CONFIG, ...(options.config || {}) });
  if (!(config.populationSize >= 2)) throw new RangeError('config.populationSize must be >= 2');

  let seed = (options.seed ^ 0) >>> 0;
  let rng, connInnov, nodeInnov, nextInnovation, nextNodeId, generation, population, bestGenome;

  function randomActivation() {
    return ACTIVATION_NAMES[rng.int(ACTIVATION_NAMES.length)];
  }
  function connectionInnovation(from, to) {
    const key = from + '->' + to;
    let id = connInnov.get(key);
    if (id === undefined) { id = nextInnovation++; connInnov.set(key, id); }
    return id;
  }
  function hiddenNodeIdFor(from, to) {
    const key = from + '->' + to;
    let id = nodeInnov.get(key);
    if (id === undefined) { id = nextNodeId++; nodeInnov.set(key, id); }
    return id;
  }

  function makeGenome() {
    const nodes = [];
    for (let i = 0; i < inputs; i++) {
      nodes.push({ id: i, type: 'input', activation: 'IDENTITY', bias: 0, depth: 0, frozen: false });
    }
    for (let o = 0; o < outputs; o++) {
      nodes.push({ id: inputs + o, type: 'output', activation: 'LOGISTIC', bias: 0, depth: 1, frozen: false });
    }
    const genes = [];
    for (let i = 0; i < inputs; i++) {
      for (let o = 0; o < outputs; o++) {
        genes.push({
          innovation: connectionInnovation(i, inputs + o),
          from: i, to: inputs + o, weight: rng.range(-1, 1), enabled: true, frozen: false
        });
      }
    }
    return { nodes, genes, frozen: false, fitness: 0, adjustedFitness: 0, speciesId: -1 };
  }

  function init(newSeed) {
    if (newSeed !== undefined) seed = newSeed >>> 0;
    rng = createSeededRng(seed);
    connInnov = new Map();
    nodeInnov = new Map();
    nextInnovation = 1;
    nextNodeId = inputs + outputs;
    generation = 0;
    population = [];
    bestGenome = null;
    for (let i = 0; i < config.populationSize; i++) population.push(makeGenome());
  }
  init(seed);

  /** Forward pass. Returns an array of `outputs` numbers. */
  function activate(genome, input) {
    if (!genome || !Array.isArray(genome.nodes) || !Array.isArray(genome.genes)) {
      throw new TypeError('activate: invalid genome');
    }
    if (!Array.isArray(input) && !ArrayBuffer.isView(input)) {
      throw new TypeError('activate: input must be an array-like of numbers');
    }
    const value = new Map();
    for (let i = 0; i < inputs; i++) {
      const v = Number(input[i]);
      value.set(i, Number.isFinite(v) ? v : 0);
    }
    const ordered = genome.nodes.slice().sort((a, b) => (a.depth - b.depth) || (a.id - b.id));
    for (const n of ordered) {
      if (n.type === 'input') continue;
      let sum = n.bias || 0;
      for (const g of genome.genes) {
        if (!g.enabled || g.to !== n.id) continue;
        const src = value.get(g.from);
        if (src !== undefined) sum += src * g.weight;
      }
      const fn = ACTIVATIONS[n.activation] || ACTIVATIONS.LOGISTIC;
      value.set(n.id, fn(sum));
    }
    const out = new Array(outputs);
    for (let o = 0; o < outputs; o++) {
      const v = value.get(inputs + o);
      out[o] = v === undefined ? 0 : v;
    }
    return out;
  }

  function addConnection(g) {
    if (g.nodes.length < 2) return false;
    for (let attempt = 0; attempt < 30; attempt++) {
      const a = g.nodes[rng.int(g.nodes.length)];
      const b = g.nodes[rng.int(g.nodes.length)];
      if (!a || !b || a === b) continue;
      if (!(a.depth < b.depth)) continue;
      if (a.type === 'output' || b.type === 'input') continue;
      const key = a.id + '->' + b.id;
      if (g.genes.some((x) => x.from === a.id && x.to === b.id)) continue;
      g.genes.push({
        innovation: connectionInnovation(a.id, b.id),
        from: a.id, to: b.id, weight: rng.range(-1, 1), enabled: true, frozen: false
      });
      return true;
    }
    return false;
  }

  function addNode(g) {
    const candidates = g.genes.filter((x) => x.enabled && !x.frozen);
    if (!candidates.length) return false;
    const gene = candidates[rng.int(candidates.length)];
    const fromNode = g.nodes.find((n) => n.id === gene.from);
    const toNode = g.nodes.find((n) => n.id === gene.to);
    if (!fromNode || !toNode) return false;
    gene.enabled = false;
    const hid = hiddenNodeIdFor(gene.from, gene.to);
    if (!g.nodes.some((n) => n.id === hid)) {
      g.nodes.push({
        id: hid, type: 'hidden', activation: 'LOGISTIC',
        bias: 0, depth: (fromNode.depth + toNode.depth) / 2, frozen: false
      });
    }
    const w1 = gene.weight;
    const w2 = rng.range(-1, 1);
    g.genes.push({ innovation: connectionInnovation(gene.from, hid), from: gene.from, to: hid, weight: w1, enabled: true, frozen: false });
    g.genes.push({ innovation: connectionInnovation(hid, gene.to), from: hid, to: gene.to, weight: w2, enabled: true, frozen: false });
    return true;
  }

  function mutate(g) {
    if (g.frozen) return g;
    for (const gene of g.genes) {
      if (gene.frozen) continue;
      const r = rng.random();
      if (r < config.weightRandomRate) gene.weight = rng.range(-config.weightRange, config.weightRange);
      else if (r < config.weightRandomRate + config.weightPerturbRate) gene.weight += rng.gaussian() * config.perturbStrength;
    }
    for (const n of g.nodes) {
      if (n.type === 'input' || n.frozen) continue;
      const r = rng.random();
      if (r < config.weightRandomRate) n.bias = rng.range(-config.weightRange, config.weightRange);
      else if (r < config.weightRandomRate + config.weightPerturbRate) n.bias += rng.gaussian() * config.perturbStrength;
      if (rng.random() < config.activationRate) n.activation = randomActivation();
    }
    if (rng.random() < config.addConnectionRate && g.genes.length < config.maxGenes) addConnection(g);
    if (rng.random() < config.addNodeRate && g.nodes.length < config.maxNodes) addNode(g);
    return g;
  }

  /** One-point innovation-aligned crossover. `a` is assumed at least as fit as `b`. */
  function crossover(a, b) {
    const child = { nodes: [], genes: [], frozen: false, fitness: 0, adjustedFitness: 0, speciesId: -1 };
    const mapB = new Map();
    for (const g of b.genes) mapB.set(g.innovation, g);
    const used = new Set();
    for (const ga of a.genes) {
      const gb = mapB.get(ga.innovation);
      if (gb && rng.random() < 0.5) child.genes.push(cloneGene(gb));
      else child.genes.push(cloneGene(ga));
      used.add(ga.innovation);
    }
    for (const gb of b.genes) if (!used.has(gb.innovation)) child.genes.push(cloneGene(gb));
    const idsA = new Set();
    for (const n of a.nodes) { child.nodes.push(cloneNode(n)); idsA.add(n.id); }
    for (const n of b.nodes) if (!idsA.has(n.id)) child.nodes.push(cloneNode(n));
    child.nodes.sort((x, y) => (x.depth - y.depth) || (x.id - y.id));
    return child;
  }

  function compatibility(a, b) {
    const mapA = new Map();
    const mapB = new Map();
    for (const g of a.genes) mapA.set(g.innovation, g);
    for (const g of b.genes) mapB.set(g.innovation, g);
    let matching = 0, weightDelta = 0, disjoint = 0;
    for (const [innov, ga] of mapA) {
      const gb = mapB.get(innov);
      if (gb) { matching++; weightDelta += Math.abs(ga.weight - gb.weight); }
      else disjoint++;
    }
    for (const innov of mapB.keys()) if (!mapA.has(innov)) disjoint++;
    const N = Math.max(mapA.size, mapB.size, 1);
    return config.c1 * (disjoint / N) + (matching ? config.c3 * (weightDelta / matching) : 0);
  }

  function speciate(pop) {
    const species = [];
    for (const g of pop) {
      let placed = false;
      for (const s of species) {
        if (compatibility(g, s.members[0]) < config.speciesThreshold) {
          s.members.push(g); placed = true; break;
        }
      }
      if (!placed) species.push({ id: species.length, members: [g], share: 0 });
    }
    for (const s of species) {
      for (const g of s.members) g.speciesId = s.id;
      let min = Infinity;
      for (const g of s.members) if (Number.isFinite(g.fitness)) min = Math.min(min, g.fitness);
      if (!Number.isFinite(min)) min = 0;
      // Positive shift makes fitness sharing well-behaved even for negative fitness
      // (the app's coins-first fitness is negative).
      let total = 0;
      const shares = s.members.map((g) => {
        const f = Number.isFinite(g.fitness) ? g.fitness : min;
        const v = Math.max(0, f - min) + 1e-6;
        total += v;
        return v;
      });
      s.share = total;
      s.members.forEach((g, i) => { g.adjustedFitness = shares[i] / s.members.length; });
    }
    return species;
  }

  function reproduce(pop, species) {
    const rankedGlobal = pop.slice().sort((x, y) => y.fitness - x.fitness);
    const next = [];
    for (let i = 0; i < Math.min(config.elitism, rankedGlobal.length); i++) {
      next.push(cloneGenome(rankedGlobal[i]));
    }
    const totalShare = species.reduce((a, s) => a + s.share, 0);
    for (const s of species) {
      const quota = species.length > 0 && totalShare > 0
        ? Math.round(config.populationSize * (s.share / totalShare))
        : Math.floor(config.populationSize / species.length);
      const ranked = s.members.slice().sort((x, y) => y.fitness - x.fitness);
      for (let i = 0; i < quota && next.length < config.populationSize; i++) {
        const p1 = ranked[rng.int(Math.min(2, ranked.length))];
        const p2 = ranked[rng.int(ranked.length)];
        if (!p1 || !p2) continue;
        const fitter = p1.fitness >= p2.fitness ? p1 : p2;
        const other = fitter === p1 ? p2 : p1;
        const child = (p1 !== p2 && rng.random() < config.crossoverRate)
          ? crossover(fitter, other)
          : cloneGenome(fitter);
        next.push(mutate(child));
      }
    }
    while (next.length < config.populationSize) {
      const parent = rankedGlobal[rng.int(rankedGlobal.length)];
      next.push(mutate(cloneGenome(parent)));
    }
    return next.slice(0, config.populationSize);
  }

  /**
   * Evaluate one generation. `fitnessFn(genome, engine) -> number` (higher is better).
   * The callback is a true black box: it may run any simulated rollout using
   * engine.activate(genome, observation).
   */
  function evaluatePopulation(fitnessFn) {
    if (typeof fitnessFn !== 'function') throw new TypeError('evaluatePopulation: fitnessFn must be a function');
    for (const g of population) {
      let f;
      try { f = fitnessFn(g, api); } catch (_) { f = -Infinity; }
      g.fitness = Number.isFinite(f) ? f : -Infinity;
    }
    const species = speciate(population);
    let generationBest = population[0];
    for (const g of population) if (g.fitness > generationBest.fitness) generationBest = g;
    if (!bestGenome || generationBest.fitness > bestGenome.fitness) bestGenome = cloneGenome(generationBest);
    population = reproduce(population, species);
    generation++;
    return {
      generation,
      bestFitness: bestGenome.fitness,
      generationBestFitness: generationBest.fitness,
      speciesCount: species.length,
      populationSize: population.length,
      bestGenome: cloneGenome(bestGenome)
    };
  }

  function exportGenome(g) { return cloneGenome(g); }

  function importGenome(obj) {
    if (!obj || typeof obj !== 'object' || !Array.isArray(obj.nodes) || !Array.isArray(obj.genes)) {
      throw new TypeError('importGenome: invalid genome object');
    }
    const g = cloneGenome(obj);
    g.nodes = g.nodes.map((n) => ({
      id: n.id | 0, type: n.type,
      activation: ACTIVATIONS[n.activation] ? n.activation : 'LOGISTIC',
      bias: Number.isFinite(Number(n.bias)) ? Number(n.bias) : 0,
      depth: Number.isFinite(Number(n.depth)) ? Number(n.depth) : 0,
      frozen: !!n.frozen
    }));
    g.genes = g.genes.map((x) => ({
      innovation: x.innovation | 0, from: x.from | 0, to: x.to | 0,
      weight: Number.isFinite(Number(x.weight)) ? Number(x.weight) : 0,
      enabled: x.enabled !== false, frozen: !!x.frozen
    }));
    g.fitness = Number.isFinite(Number(g.fitness)) ? Number(g.fitness) : 0;
    g.adjustedFitness = Number.isFinite(Number(g.adjustedFitness)) ? Number(g.adjustedFitness) : 0;
    g.speciesId = Number.isFinite(Number(g.speciesId)) ? Number(g.speciesId) : -1;
    // Keep innovation/node counters ahead of imported structure so later mutations
    // never collide with imported ids.
    for (const n of g.nodes) {
      if (n.id >= nextNodeId) nextNodeId = n.id + 1;
    }
    for (const x of g.genes) {
      if (x.innovation >= nextInnovation) nextInnovation = x.innovation + 1;
      connInnov.set(x.from + '->' + x.to, x.innovation);
    }
    return g;
  }

  function cloneFrozen(g) {
    const c = cloneGenome(g);
    c.frozen = true;
    for (const n of c.nodes) n.frozen = true;
    for (const x of c.genes) x.frozen = true;
    return c;
  }

  const api = {
    inputs, outputs, seed, config,
    createGenome: makeGenome,
    activate,
    evaluatePopulation,
    getBestGenome: () => (bestGenome ? cloneGenome(bestGenome) : null),
    getGeneration: () => generation,
    getPopulation: () => population.map(cloneGenome),
    exportGenome,
    importGenome,
    cloneFrozen,
    cloneGenome,
    compatibility,
    reset: (newSeed) => { init(newSeed === undefined ? seed : newSeed); return api; },
    rngState: () => rng.getState()
  };
  return api;
}
