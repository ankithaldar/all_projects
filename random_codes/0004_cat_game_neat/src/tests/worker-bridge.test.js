/**
 * worker-bridge.test.js — tests for the train-worker telemetry bridge.
 *
 * Exercises the exact pure helpers used by src/worker/train-worker.js:
 *  - the controller's telemetry is a ring-buffer OBJECT (toArray), not an array;
 *  - DONE reports the BEST feasible genome's own values (not OR/max/min);
 *  - a DONE/GENOME_RESULT flow emits >= 1 GENOME_RESULT and DONE equals best;
 *  - importing the worker module on the main thread installs no message listener.
 */
import { test, assert, assertEqual, assertDeepEqual, assertThrows } from './harness.js';
import { validateGenome, KNOWN_TARGET_IDS, KNOWN_CRAFTABLE_IDS } from '../validate.js';
import { createNeatAdapter } from '../neat/adapter.js';
import { EMBEDDED_CONFIG, PRESETS, prefillInitialState } from '../config.js';
import { createRingBuffer, DEFAULT_GENS as CTRL_GENS, DEFAULT_POP as CTRL_POP, DEFAULT_WALL_MS } from '../neat/controller.js';
import {
  collectTelemetry,
  isGenomeResultEvent,
  sanitizeGenomeResult,
  selectBestResult,
  buildDoneMessage,
  buildReplayMessages,
  buildTrainOptions,
  dispatchProgress,
  createCapState,
  sanitizeGenome,
  createDeadline,
  buildBudgetProgressMessage,
  DEFAULT_GENS,
  DEFAULT_POP,
  DEFAULT_MAX_WALL_MS
} from '../worker/train-worker.js';
import {
  dispatchWorkerMessage, collectGenomePayload, buildStartMessage, buildSolverDiagnostics,
  wireTabs, applyTabSelection, TAB_IDS,
  applyOverridesAndStart, startAll, resetAll, onPresetChange,
  getConfig, getBaseConfig, buildEngine, runDisplay, getRunToken,
  updateSimClockFromSnapshot
} from '../main.js';
import { createDiagnostics, sanitizeDiagnostics, DIAGNOSTICS_MAP_CAP } from '../ui/diagnostics.js';

function gr(overrides = {}) {
  return Object.assign({
    type: 'GENOME_RESULT',
    itemId: 'orb',
    generation: 0,
    genomeIndex: 0,
    fitness: -(1000 * 100000 + 100),
    feasible: true,
    completionTick: 100,
    elapsedTicks: 100,
    totalSpent: 1000,
    perItem: {}
  }, overrides);
}

test('collectTelemetry reads a ring-buffer object via toArray()', () => {
  const ring = createRingBuffer(4);
  const a = gr({ genomeIndex: 1 });
  const b = gr({ genomeIndex: 2 });
  ring.push(a);
  ring.push(b);
  assertEqual(Array.isArray(ring), false);
  const arr = collectTelemetry({ telemetry: ring });
  assertEqual(arr.length, 2);
  assertEqual(arr[0], a);
  assertEqual(arr[1], b);
});

test('selectBestResult picks the feasible genome with lowest totalSpent, not highest', () => {
  const events = [
    gr({ genomeIndex: 1, totalSpent: 300, completionTick: 100, feasible: true,
         fitness: -(300 * 100000 + 100) }),
    gr({ genomeIndex: 2, totalSpent: 100, completionTick: 500, feasible: true,
         fitness: -(100 * 100000 + 500) }),
    gr({ genomeIndex: 3, totalSpent: 0, completionTick: null, feasible: false,
         fitness: -1e12 + 0.5 })
  ];
  const best = selectBestResult(events);
  assertEqual(best.genomeIndex, 2);
  assertEqual(best.totalSpent, 100);
  assertEqual(best.completionTick, 500);
});

test('selectBestResult falls back to highest fitness when nothing is feasible', () => {
  const events = [
    gr({ genomeIndex: 1, feasible: false, fitness: -1e12 + 0.1 }),
    gr({ genomeIndex: 2, feasible: false, fitness: -1e12 + 0.9 })
  ];
  assertEqual(selectBestResult(events).genomeIndex, 2);
  assertEqual(selectBestResult([]), null);
});

test('isGenomeResultEvent recognizes both typed and shaped telemetry', () => {
  assert(isGenomeResultEvent(gr()));
  assert(isGenomeResultEvent({ perItem: {}, fitness: 1 }));
  assertEqual(isGenomeResultEvent({ type: 'BEST' }), false);
});

test('sanitizeGenomeResult caps strings, clamps ranges and drops invalid per-item keys', () => {
  const msg = sanitizeGenomeResult(gr({
    itemId: 'x'.repeat(200),
    generation: -5,
    totalSpent: -10,
    perItem: { ['a'.repeat(50)]: { coinsSpent: 10 }, silver: { coinsSpent: 7, timeMinutes: 3 } }
  }));
  assertEqual(msg.itemId.length, 40);
  assertEqual(msg.generation, 0);      // negative generation clamped
  assertEqual(msg.totalSpent, 0);      // negative spend clamped
  assertEqual(Object.keys(msg.perItem).length, 1);
  assertEqual(msg.perItem.silver.coinsSpent, 7);
});

test('buildDoneMessage consumes the controller ring buffer and reports the best genome values', () => {
  const ring = createRingBuffer(10);
  ring.push(gr({ genomeIndex: 1, totalSpent: 300, completionTick: 100,
                 fitness: -(300 * 100000 + 100) }));
  ring.push(gr({ genomeIndex: 2, totalSpent: 100, completionTick: 500,
                 fitness: -(100 * 100000 + 500) }));

  const done = buildDoneMessage({ telemetry: ring, genomes: { orb: {}, gold: {} } });
  assertEqual(done.type, 'DONE');
  assertEqual(done.feasible, true);
  // Old code used Math.max(spend)=300 and Math.min(tick)=100 and OR-ed feasible.
  assertEqual(done.totalSpent, 100);
  assertEqual(done.completionTick, 500);
  assertEqual(done.genomeCount, 2);
  assertEqual(done.telemetryCount, 2);
});

test('DONE / GENOME_RESULT flow emits >=1 GENOME_RESULT and DONE equals the best genome', () => {
  const ring = createRingBuffer(100);
  for (let i = 0; i < 25; i += 1) {
    ring.push(gr({ generation: 0, genomeIndex: i, totalSpent: 2000 + i,
                   completionTick: 200 + i, fitness: -((2000 + i) * 100000 + 200 + i) }));
  }
  for (let i = 0; i < 5; i += 1) {
    ring.push(gr({ generation: 1, genomeIndex: i, totalSpent: 500 + i,
                   completionTick: 50 + i, fitness: -((500 + i) * 100000 + 50 + i) }));
  }
  const result = { telemetry: ring, genomes: { orb: {} } };

  const posted = buildReplayMessages(result);
  const genomeResults = posted.filter((m) => m.type === 'GENOME_RESULT');
  assert(genomeResults.length >= 1);
  // Per-generation cap of 20: 20 from gen0 + 5 from gen1 = 25.
  assertEqual(genomeResults.length, 25);
  assert(genomeResults.every((m) => Number.isFinite(m.totalSpent) && m.fitness < 0));

  const done = buildDoneMessage(result);
  const best = selectBestResult(ring.toArray());
  assertEqual(done.feasible, best.feasible);
  assertEqual(done.totalSpent, best.totalSpent);
  assertEqual(done.completionTick, best.completionTick);
  assertEqual(best.totalSpent, 500);   // best is the lowest-spend feasible genome
  assertEqual(best.completionTick, 50);
});

test('buildDoneMessage reports infeasible defaults for an empty ring buffer', () => {
  const done = buildDoneMessage({ telemetry: createRingBuffer(2), genomes: {} });
  assertEqual(done.feasible, false);
  assertEqual(done.completionTick, 8064);
  assertEqual(done.totalSpent, 0);
  assertEqual(done.telemetryCount, 0);
});

test('worker DEFAULT_GENS / DEFAULT_POP / DEFAULT_MAX_WALL_MS match controller exports', () => {
  assertEqual(DEFAULT_GENS, CTRL_GENS);
  assertEqual(DEFAULT_POP, CTRL_POP);
  assertEqual(DEFAULT_MAX_WALL_MS, DEFAULT_WALL_MS);
});

test('buildTrainOptions forwards changed control values as adapterConfig.populationSize and maxGenerations', () => {
  const cfg = { items: [], targets: {} };
  const first = buildTrainOptions({ config: cfg, seed: 7, options: { populationSize: 37, generations: 7 } });
  assertEqual(first.adapterConfig.populationSize, 37);
  assertEqual(first.maxGenerations, 7);
  assertEqual(first.seed, 7);

  // Changing the control must change the value that reaches trainAll.
  const changed = buildTrainOptions({ config: cfg, seed: 7, options: { populationSize: 120, generations: 3 } });
  assertEqual(changed.adapterConfig.populationSize, 120);
  assertEqual(changed.maxGenerations, 3);
});

test('buildTrainOptions clamps pop-size to 2..500 and applies bounded defaults', () => {
  const cfg = { items: [], targets: {} };
  const clamped = buildTrainOptions({ config: cfg, options: { populationSize: 1, generations: 0, maxWallMs: -1 } });
  assertEqual(clamped.adapterConfig.populationSize, 2);
  assertEqual(clamped.maxGenerations, 1);
  assertEqual(clamped.maxWallMs, 1000);

  const defaults = buildTrainOptions({ config: cfg });
  assertEqual(defaults.adapterConfig.populationSize, DEFAULT_POP);
  assertEqual(defaults.maxGenerations, DEFAULT_GENS);
  assertEqual(defaults.maxWallMs, DEFAULT_MAX_WALL_MS);
  assert(DEFAULT_POP >= 2 && DEFAULT_POP < 50);
  assert(DEFAULT_GENS > 0 && DEFAULT_GENS < 100);
});

function handlerStub() {
  return {
    logs: [], status: [], genomes: [], rejects: 0,
    logEvent(e) { this.logs.push(e); },
    setStatus(s) { this.status.push(s); },
    showGenome(e) { this.genomes.push(e); },
    reject() { this.rejects += 1; }
  };
}

test('live runtime handler validates before dispatch and rejects malformed messages', () => {
  const h = handlerStub();
  dispatchWorkerMessage({ type: 'PROGRESS', itemId: 'orb', generation: 2, message: 'go', phase: 'eval' }, h);
  assertEqual(h.genomes.length, 0);
  assert(h.logs.some((e) => e.type === 'TRAIN_PROGRESS'));
  assert(h.status.length >= 1);

  const bad = handlerStub();
  dispatchWorkerMessage({ type: 'UNKNOWN_MESSAGE', foo: 1 }, bad);
  assertEqual(bad.rejects, 1);
  assert(bad.logs.some((e) => e.type === 'WORKER_MSG_REJECTED'));
});

test('live runtime handler strips markup from message text before logging', () => {
  const h = handlerStub();
  dispatchWorkerMessage({
    type: 'PROGRESS', itemId: '<img src=x onerror=alert(1)>', generation: 1,
    message: '<script>bad()</script>', phase: 'eval'
  }, h);
  assertEqual(h.logs.length, 1);
  assert(h.logs[0].item.indexOf('<') === -1);
  assert(h.logs[0].reason.indexOf('<') === -1);
});

/* ------------------------- FIX ROUND 2 findings ---------------------------- */

const trainingGenome = {
  nodes: [
    { id: 1, type: 'input', activation: 'IDENTITY', bias: 0, depth: 0, frozen: false },
    { id: 2, type: 'output', activation: 'LOGISTIC', bias: 0.1, depth: 1, frozen: false }
  ],
  genes: [{ innovation: 1, from: 1, to: 2, weight: 0.5, enabled: true, frozen: false }],
  frozen: false,
  fitness: -12345
};

test('controller-shaped PROGRESS carrying bestFitness emits PROGRESS, not BEST', () => {
  const caps = createCapState();
  const out = [];
  // This is the exact shape controller.js sends: type PROGRESS + bestFitness.
  const ev = {
    type: 'PROGRESS', itemId: 'silver', itemIndex: 9,
    generation: 3, generations: 10, bestFitness: -12345
  };
  const msg = dispatchProgress(ev, (m) => out.push(m), caps);
  assertEqual(msg.type, 'PROGRESS');
  assertEqual(out.length, 1);
  assertEqual(out[0].type, 'PROGRESS');
  assertEqual(out[0].itemId, 'silver');
  assert(out.every((m) => m.type !== 'BEST'));
});

test('explicit BEST kind still emits a BEST message', () => {
  const caps = createCapState();
  const out = [];
  const msg = dispatchProgress({ type: 'BEST', itemId: 'silver', generation: 3, fitness: -12345 }, (m) => out.push(m), caps);
  assertEqual(msg.type, 'BEST');
  assertEqual(out[0].itemId, 'silver');
});

test('per-(itemId|generation) cap lets two items share generation 1', () => {
  const caps = createCapState();
  const out = [];
  for (let i = 0; i < 25; i += 1) {
    dispatchProgress(gr({ itemId: 'orb', generation: 1, genomeIndex: i }), (m) => out.push(m), caps);
  }
  assertEqual(out.filter((m) => m.itemId === 'orb').length, 20);

  // Item 1 starts at generation 1 too: its own counter must be untouched.
  const other = dispatchProgress(gr({ itemId: 'silver', generation: 1, genomeIndex: 0 }), (m) => out.push(m), caps);
  assert(other && other.type === 'GENOME_RESULT');
  assertEqual(out.filter((m) => m.itemId === 'silver').length, 1);

  // PROGRESS has a separate counter from GENOME_RESULT for the same key.
  for (let i = 0; i < 25; i += 1) {
    dispatchProgress({ type: 'PROGRESS', itemId: 'orb', generation: 1, completed: i }, (m) => out.push(m), caps);
  }
  assertEqual(out.filter((m) => m.type === 'PROGRESS').length, 20);
  assertEqual(out.filter((m) => m.type === 'GENOME_RESULT').length, 21);
});

test('BEST carries a validated genome and DONE carries a bounded genomes map', () => {
  const caps = createCapState();
  const out = [];
  dispatchProgress({ type: 'BEST', itemId: 'orb', generation: 2, fitness: -12345, genome: trainingGenome }, (m) => out.push(m), caps);
  const best = out[0];
  assertEqual(best.type, 'BEST');
  assert(best.genome && Array.isArray(best.genome.nodes) && Array.isArray(best.genome.genes));
  assertEqual(best.genome.nodes[0].id, 1);
  assertEqual(best.genome.genes[0].to, 2);

  const done = buildDoneMessage({ telemetry: [], genomes: { orb: trainingGenome, silver: trainingGenome } });
  assertEqual(done.genomeCount, 2);
  assert(done.genomes && done.genomes.orb && done.genomes.silver);
  assertEqual(done.genomes.orb.nodes[0].id, 1);
  assertEqual(sanitizeGenome(trainingGenome).genes.length, 1);
});

test('collectGenomePayload collects trained genomes from BEST and DONE messages', () => {
  const store = {};
  collectGenomePayload({ type: 'BEST', itemId: 'orb', genome: trainingGenome }, store);
  collectGenomePayload({ type: 'DONE', genomes: { silver: trainingGenome } }, store);
  assertEqual(Object.keys(store).length, 2);
  assert(store.orb && store.silver);
});

test('sanitizeGenome preserves fractional hidden depth and replay activation parity', () => {
  // inhouse-neat creates hidden nodes at (from.depth + to.depth)/2 = 0.5 here.
  // Ids are chosen so that rounding 0.5 -> 1 would sort output before hidden
  // and skip the hidden->output gene, changing activate() output.
  const genome = {
    nodes: [
      { id: 1, type: 'input', activation: 'IDENTITY', bias: 0, depth: 0, frozen: false },
      { id: 5, type: 'hidden', activation: 'LOGISTIC', bias: 0, depth: 0.5, frozen: false },
      { id: 3, type: 'output', activation: 'LOGISTIC', bias: 0, depth: 1, frozen: false }
    ],
    genes: [
      { innovation: 1, from: 1, to: 5, weight: 1.2, enabled: true, frozen: false },
      { innovation: 2, from: 5, to: 3, weight: -0.7, enabled: true, frozen: false }
    ],
    frozen: false, fitness: 0, adjustedFitness: 0, speciesId: 0
  };

  const clean = sanitizeGenome(genome);
  assertEqual(clean.nodes.find((n) => n.id === 5).depth, 0.5);

  const validated = validateGenome(clean);
  assertEqual(validated.nodes.find((n) => n.id === 5).depth, 0.5);

  const adapter = createNeatAdapter({ inputs: 1, outputs: 1, seed: 4242 });
  const replayed = adapter.importGenome(validated);
  const obs = [0.42];
  assertDeepEqual(adapter.activate(replayed, obs), adapter.activate(genome, obs));
});

test('hard deadline posts a DONE from partial results and clears its timer (maxWallMs=1000)', () => {
  const posted = [];
  let timerCb = null;
  let timerId = 0;
  let cleared = 0;
  const guard = createDeadline({
    maxWallMs: 1000,
    sink: (m) => posted.push(m),
    makeTimer: (cb, ms) => {
      assertEqual(ms, 1000);
      timerId += 1;
      timerCb = cb;
      return timerId;
    },
    cancelTimer: (id) => { assertEqual(id, timerId); cleared += 1; },
    getPartial: () => ({ genomes: { orb: trainingGenome }, telemetry: [gr({ totalSpent: 42 })] })
  });

  guard.arm();
  assertEqual(guard.isArmed(), true);
  const t0 = Date.now();
  timerCb(); // simulate the 1000 ms deadline firing
  const elapsed = Date.now() - t0;
  assert(elapsed < 5000, 'deadline handling must be prompt');

  assertEqual(posted.length, 1);
  assertEqual(posted[0].type, 'DONE');
  assertEqual(posted[0].totalSpent, 42);
  assert(posted[0].genomes && posted[0].genomes.orb, 'partial genomes must reach DONE');
  assertEqual(guard.isFinished(), true);
  assertEqual(guard.isArmed(), false);
  assertEqual(cleared, 1);

  // Normal completion after the timer fired must not emit a duplicate DONE.
  assertEqual(guard.finish({ genomes: {}, telemetry: [] }), false);
  timerCb();
  assertEqual(posted.length, 1);
});

test('normal completion clears the deadline timer and suppresses a later timeout', () => {
  const posted = [];
  let timerCb = null;
  let cleared = 0;
  const guard = createDeadline({
    maxWallMs: 1000,
    sink: (m) => posted.push(m),
    makeTimer: (cb) => { timerCb = cb; return 7; },
    cancelTimer: () => { cleared += 1; },
    getPartial: () => null
  });
  guard.arm();
  assertEqual(guard.finish({ genomes: {}, telemetry: [] }), true);
  assertEqual(cleared, 1);
  assertEqual(guard.isArmed(), false);
  timerCb(); // late timeout is a no-op
  assertEqual(posted.length, 1);
  assertEqual(posted[0].type, 'DONE');
});

test('budget PROGRESS pings carry explicit elapsedMs / budgetMs / remainingMs', () => {
  const msg = buildBudgetProgressMessage(10000, 1000, 4, 10750);
  assertEqual(msg.type, 'PROGRESS');
  assertEqual(msg.phase, 'budget');
  assertEqual(msg.elapsedMs, 750);
  assertEqual(msg.budgetMs, 1000);
  assertEqual(msg.remainingMs, 250);
});

test('REAL module worker: START yields PROGRESS, GENOME_RESULT and DONE with no ERROR', async () => {
  // Skip gracefully outside a browser (Node has no DOM Worker + postMessage bridge).
  if (typeof Worker === 'undefined' || typeof document === 'undefined') return;

  const config = {
    items: EMBEDDED_CONFIG.items,
    targets: EMBEDDED_CONFIG.targets,
    initialCoins: 5998996,
    initialStash: {},
    starterMode: 'PREDELIVERED_RESERVED',
    maxTicks: 50 // short rollouts keep this E2E probe fast; still a real worker
  };
  const workerUrl = new URL('../worker/train-worker.js', import.meta.url);
  const worker = new Worker(workerUrl, { type: 'module' });
  const messages = [];
  const t0 = Date.now();

  try {
    const done = await new Promise((resolve, reject) => {
      // Guard timeout is deliberately larger than Chrome's virtual-time budget
      // so a virtual clock cannot fire it before the real worker finishes.
      const timer = setTimeout(() => {
        reject(new Error('REAL_WORKER_TIMEOUT'));
      }, 60000); // guard only; exceeds any Chrome virtual-time budget used in CI
      worker.onmessage = (event) => {
        const m = event.data;
        messages.push(m);
        if (m && m.type === 'DONE') { clearTimeout(timer); resolve(m); }
      };
      worker.onerror = (event) => {
        clearTimeout(timer);
        reject(new Error('REAL_WORKER_ERROR: ' + (event && event.message)));
      };
      worker.postMessage({
        type: 'START', config, seed: 4242,
        options: { populationSize: 2, generations: 2, maxWallMs: 1000 }
      });
    });

    const elapsedMs = Date.now() - t0;
    console.log(`REAL_WORKER_PROBE genomeCount=${done.genomeCount} elapsedMs=${elapsedMs} messages=${messages.length}`);
    assert(messages.some((m) => m.type === 'PROGRESS'), 'expected at least one PROGRESS');
    assert(messages.some((m) => m.type === 'GENOME_RESULT'), 'expected at least one GENOME_RESULT');
    assert(!messages.some((m) => m.type === 'ERROR'), 'worker must not emit ERROR');
    assert(done.genomeCount >= 1, 'DONE must report at least one trained genome');
    assert(elapsedMs < 20000, 'worker must finish within the test timeout');
  } finally {
    worker.terminate();
  }
});

test('importing worker modules installs no window message listener', async () => {
  // Loose comparison so the assertion also holds under a non-browser DOM stub
  // where window.onmessage is undefined rather than null.
  assert(window.onmessage == null);

  // Static imports are hoisted, so spy by source inspection instead: strip
  // comments (the header comment mentions this API) and reject any real call
  // to addEventListener('message') in either module.
  const stripComments = (src) => src
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/\/\/.*$/gm, '');
  for (const url of ['./src/main.js', './src/worker/train-worker.js']) {
    const response = await fetch(url);
    assert(response.ok, `fetch failed for ${url}`);
    const code = stripComments(await response.text());
    assert(!/\.addEventListener\(\s*['"]message['"]/i.test(code), `${url} installs a message listener`);
  }
});

/* --------------------- solver-integration (frozen contract) ---------------- */

test('START message uses only the frozen field set — no solver diagnostics fields', () => {
  const msg = buildStartMessage(
    { items: [], targets: {} },
    7,
    { populationSize: 42, generations: 5, criticality: { x: 1 }, caps: { x: 9 }, diagnostics: 'leak' }
  );
  assertDeepEqual(Object.keys(msg).sort(), ['config', 'options', 'seed', 'type']);
  assertDeepEqual(Object.keys(msg.options).sort(), ['generations', 'populationSize']);
  for (const forbidden of ['criticality', 'caps', 'diagnostics', 'estimatedTicks', 'estimatedCost', 'manufactureRequired']) {
    assertEqual(Object.prototype.hasOwnProperty.call(msg, forbidden), false, `no ${forbidden} on START`);
    assertEqual(Object.prototype.hasOwnProperty.call(msg.options, forbidden), false, `no ${forbidden} in START.options`);
  }
  assertEqual(msg.seed, 7);
  assertEqual(msg.options.populationSize, 42);
});

test('diagnostics are computed on the main thread from injected helpers (no Worker needed)', async () => {
  const config = {
    items: [{ id: 'a', type: 'craftable', ingredients: [], coinCost: 1, craftTimeMinutes: 5 }],
    targets: { a: 2 }, initialStash: {}, starterMode: 'CONSUMABLE'
  };
  let plannerCalls = 0, criticalityCalls = 0;
  const helpers = {
    planBatchCaps(items, mfr, maxTicks, maxBatch) {
      plannerCalls += 1;
      assertEqual(maxTicks, 8064);
      assertEqual(maxBatch, 20);
      assertEqual(mfr.a, 2, 'manufactureRequired basis passed to planner');
      return { caps: { a: 3 }, estimatedCost: 1234, estimatedTicks: 99 };
    },
    computeCriticality(items, targets) {
      criticalityCalls += 1;
      assertDeepEqual(targets, { a: 2 });
      return { a: 0.75 };
    }
  };
  // No Worker / postMessage is involved: this is pure main-thread computation.
  const diag = await buildSolverDiagnostics(config, helpers);
  assertEqual(plannerCalls, 1);
  assertEqual(criticalityCalls, 1);
  assertEqual(diag.caps.a, 3);
  assertEqual(diag.criticality.a, 0.75);
  assertEqual(diag.estimatedCost, 1234);
  assertEqual(diag.estimatedTicks, 99);
});

test('diagnostics view: textContent-only, allowlisted, capped, XSS-safe', async () => {
  const evil = '<img src=x onerror=alert(1)>';
  const many = {};
  for (let i = 0; i < 40; i += 1) many['orb' + i] = i;
  const diag = {
    criticality: Object.assign({}, many, { artifact: 5, ['__proto__']: 1, constructor: 2, [evil]: 3 }),
    caps: Object.assign({}, many, { artifact: 20, [evil]: 9 }),
    estimatedTicks: 8064,
    estimatedCost: 7111975,
    label: evil + 'x'.repeat(500)
  };

  const clean = sanitizeDiagnostics(diag);
  for (const section of ['criticality', 'caps']) {
    assertEqual(Object.getPrototypeOf(clean[section]), null, `${section} map is null-prototype`);
    assertEqual(Object.prototype.hasOwnProperty.call(clean[section], '__proto__'), false, 'no __proto__ key');
    assertEqual(Object.prototype.hasOwnProperty.call(clean[section], 'constructor'), false, 'no constructor key');
    assertEqual(Object.prototype.hasOwnProperty.call(clean[section], evil), false, 'no injected-markup key');
    assert(Object.keys(clean[section]).length <= DIAGNOSTICS_MAP_CAP, `${section} capped at 23`);
    assert(Object.prototype.hasOwnProperty.call(clean[section], 'artifact'), true, 'known id kept');
  }
  assert(clean.label.length <= 120, 'label capped at MAX_STR=120');

  // Minimal DOM stub with injectable document (no global mutation).
  function fakeDoc() {
    return {
      createElement(tag) {
        return {
          tag, className: '', textContent: '', children: [],
          appendChild(c) { this.children.push(c); }
        };
      }
    };
  }
  function collect(node, out) {
    if (node.textContent) out.push(node.textContent);
    for (const c of (node.children || [])) collect(c, out);
    return out;
  }
  const host = fakeDoc().createElement('div');
  const view = createDiagnostics(host, fakeDoc());
  view.render(diag);
  const texts = collect(host, []);
  assert(texts.length > 1, 'rendered several text nodes');
  // XSS safety = no executable element is ever created. The payload may appear
  // as inert TEXT (textContent), but must never become an <img>/<script> node.
  const tags = [];
  (function walk(n) {
    if (n.tag) tags.push(n.tag);
    for (const c of (n.children || [])) walk(c);
  })(host);
  assert(!tags.some((t) => t === 'img' || t === 'script'), 'no executable markup element created');
  const joined = texts.join('\n');
  assert(joined.indexOf('__proto__') === -1 && joined.indexOf('constructor') === -1,
    'forbidden keys never rendered');
  for (const t of texts) assert(t.length <= 120, 'every rendered string capped at 120');
});

test('diagnostics/main source: no markup-injection APIs and no solver fields in postMessage', async () => {
  const strip = (src) => src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');
  for (const url of ['./src/main.js', './src/ui/diagnostics.js']) {
    const response = await fetch(url);
    assert(response.ok, `fetch failed for ${url}`);
    const code = strip(await response.text());
    assert(!/innerHTML|insertAdjacentHTML/i.test(code), `${url} uses a markup-injection API`);
  }
  const mainCode = strip(await (await fetch('./src/main.js')).text());
  assert(/postMessage\(\s*buildStartMessage\(/.test(mainCode),
    'main posts only the frozen START builder');
  assert(!/postMessage\([^)]*(criticality|caps|diagnostics|estimatedTicks)/i.test(mainCode),
    'main never posts solver diagnostics fields');
});

/* --------------------------- tabs / panel wiring --------------------------- */

test('tabs: click shows only the selected panel and syncs is-active + aria-selected', () => {
  // Minimal DOM stub: the real handlers/listeners are exercised via wireTabs().
  function makeDoc() {
    const nodes = {};
    for (const id of TAB_IDS) {
      const classes = new Set(['tab']);
      if (id === 'tab-train') classes.add('is-active');
      nodes[id] = {
        id,
        classList: {
          toggle(cls, on) { if (on) classes.add(cls); else classes.delete(cls); },
          contains(cls) { return classes.has(cls); }
        },
        attrs: { 'aria-selected': id === 'tab-train' ? 'true' : 'false' },
        listeners: {},
        setAttribute(k, v) { this.attrs[k] = String(v); },
        getAttribute(k) { return this.attrs[k]; },
        addEventListener(ev, cb) { this.listeners[ev] = cb; },
        click() { if (this.listeners.click) this.listeners.click(); }
      };
    }
    for (const name of ['train', 'logger', 'results']) {
      const id = 'panel-' + name;
      nodes[id] = {
        id,
        attrs: name === 'train' ? {} : { hidden: '' },
        setAttribute(k, v) { this.attrs[k] = String(v); },
        getAttribute(k) { return this.attrs[k]; },
        removeAttribute(k) { delete this.attrs[k]; },
        hasAttribute(k) { return Object.prototype.hasOwnProperty.call(this.attrs, k); }
      };
    }
    return { getElementById: (id) => nodes[id] || null, nodes };
  }

  function assertSelection(doc, selected) {
    for (const id of TAB_IDS) {
      const name = id.slice('tab-'.length);
      const on = id === selected;
      assertEqual(doc.nodes[id].classList.contains('is-active'), on, `${id} is-active=${on}`);
      assertEqual(doc.nodes[id].getAttribute('aria-selected'), on ? 'true' : 'false', `${id} aria-selected`);
      assertEqual(doc.nodes['panel-' + name].hasAttribute('hidden'), !on, `panel-${name} hidden=${!on}`);
    }
  }

  const doc = makeDoc();
  wireTabs(doc);
  // Initial state matches the shipped markup (Train active, others hidden).
  assertSelection(doc, 'tab-train');

  doc.nodes['tab-logger'].click();
  assertSelection(doc, 'tab-logger');

  doc.nodes['tab-results'].click();
  assertSelection(doc, 'tab-results');

  doc.nodes['tab-train'].click();
  assertSelection(doc, 'tab-train');

  // applyTabSelection is also usable directly (pure of module state).
  assertEqual(applyTabSelection(doc, 'tab-results'), 'tab-results');
  assertSelection(doc, 'tab-results');
});

/* ---------------- editable initial-state propagation (TASK B) -------------- */

function baseConfigFor(preset = 'reuse') {
  const p = PRESETS[preset];
  return {
    items: EMBEDDED_CONFIG.items,
    targets: { ...EMBEDDED_CONFIG.targets },
    initialCoins: p.initialCoins,
    initialStash: { ...p.initialStash },
    starterMode: p.starterMode
  };
}

function overrideSet(base, opts = {}) {
  const out = prefillInitialState(base);
  for (const id of KNOWN_TARGET_IDS) out.targets[id] = 0;
  for (const id of KNOWN_CRAFTABLE_IDS) out.initialStash[id] = 0;
  out.targets.orb = opts.orb === undefined ? 77 : opts.orb;
  out.initialStash.wood = opts.wood === undefined ? 9 : opts.wood;
  out.initialCoins = opts.coins === undefined ? 123 : opts.coins;
  return out;
}

function noopDeps(extra = {}) {
  return Object.assign({
    controls: { speed: 'fast', seed: 7, preset: 'reuse', populationSize: 2, generations: 1 },
    loadConfig: async (preset) => baseConfigFor(preset),
    readInitialState: (base) => overrideSet(base),
    startWorkerTraining: () => {},
    runDisplay: () => {},
    refreshDiagnostics: () => {},
    panel: { prefill: () => {} }
  }, extra);
}

test('applyOverridesAndStart commits module config; START.config carries all three overrides by reference', () => {
  const base = baseConfigFor('reuse');
  const overrides = overrideSet(base);
  let startMessage = null;
  let engineCfg = null;
  const out = applyOverridesAndStart(base, overrides, {
    seed: 7,
    options: { populationSize: 2, generations: 1 },
    onStart: (msg) => { startMessage = msg; },
    onCreateEngine: (cfg) => { engineCfg = cfg; }
  });

  assertEqual(getBaseConfig(), base, 'pristine base config is kept at module level');
  assertEqual(getConfig(), out.config, 'module config is the committed effective config');
  assertEqual(startMessage.config, out.config, 'START.config is the module config itself');
  assertEqual(engineCfg, out.config, 'createEngine receives the same module config');
  assertEqual(startMessage.config.initialCoins, 123);
  assertEqual(startMessage.config.targets.orb, 77);
  assertEqual(startMessage.config.initialStash.wood, 9);
  assertEqual(Object.getPrototypeOf(startMessage.config.targets), null, 'null proto survives into START');
  assertEqual(Object.getPrototypeOf(startMessage.config.initialStash), null, 'null proto survives into START');
  assertDeepEqual(Object.keys(startMessage).sort(), ['config', 'options', 'seed', 'type'],
    'no new worker message fields');
  assertEqual(base.initialCoins, PRESETS.reuse.initialCoins, 'loaded config not mutated');
  assertNotEqualRef(base.targets, out.config.targets, 'targets assigned by reference, not copied into base');
});

function assertNotEqualRef(a, b) { assert(a !== b, 'expected different references'); }

test('buildEngine() reads the committed overrides through the real createEngine path', () => {
  const base = baseConfigFor('reuse');
  applyOverridesAndStart(base, overrideSet(base, { coins: 321, wood: 11 }), { seed: 3 });
  const snap = buildEngine().snapshot();
  assertEqual(snap.coins, 321, 'engine sees overridden coins');
  assertEqual(snap.stash.wood, 11, 'engine sees overridden starter stash');
});

test('startAll (stubbed I/O) applies DOM-read overrides through the same module config path', async () => {
  const base = baseConfigFor('reuse');
  let startMessage = null;
  let prefilled = null;
  let displayToken = null;
  const result = await startAll(noopDeps({
    loadConfig: async () => base,
    onStart: (msg) => { startMessage = msg; },
    panel: { prefill: (cfg) => { prefilled = cfg; } },
    runDisplay: (token) => { displayToken = token; }
  }));
  assert(!!result, 'startAll returns the committed config');
  assert(getConfig() === result, 'module config must be the exact object returned by startAll');
  assertEqual(getConfig().initialCoins, 123, 'module config coins');
  assertEqual(startMessage.config.initialCoins, 123);
  assertEqual(startMessage.config.targets.orb, 77);
  assertEqual(startMessage.config.initialStash.wood, 9);
  assertEqual(prefilled, result, 'panel re-prefilled from the effective config');
  assertEqual(typeof displayToken, 'number', 'display started after override commit');
});

test('reuse -> empty -> Start -> Reset yields 0 coins and empty stash; default Start/Reset restores preset', async () => {
  const reuse = baseConfigFor('reuse');
  const empty = baseConfigFor('empty');

  await startAll(noopDeps({
    controls: { speed: 'fast', seed: 7, preset: 'reuse', populationSize: 2, generations: 1 },
    loadConfig: async () => reuse,
    readInitialState: (base) => overrideSet(base, { coins: reuse.initialCoins, wood: 0 })
  }));
  assertEqual(getConfig().initialCoins, reuse.initialCoins);

  await startAll(noopDeps({
    controls: { speed: 'fast', seed: 7, preset: 'empty', populationSize: 2, generations: 1 },
    loadConfig: async () => empty,
    readInitialState: (base) => overrideSet(base, { coins: 0, wood: 0 })
  }));
  assertEqual(getConfig().initialCoins, 0);

  let resetPrefill = null;
  resetAll({ panel: { prefill: (cfg) => { resetPrefill = cfg; } } });
  assertEqual(getConfig(), getBaseConfig(), 'Reset restores the pristine base config');
  assertEqual(getConfig().initialCoins, 0, 'empty-preset coins after Reset');
  assertEqual(Object.keys(getConfig().initialStash).length, 0, 'empty stash after Reset');
  assertEqual(resetPrefill, getBaseConfig(), 'Reset clears overrides by re-prefilling the panel from base');

  await startAll(noopDeps({
    controls: { speed: 'fast', seed: 7, preset: 'reuse', populationSize: 2, generations: 1 },
    loadConfig: async () => reuse,
    readInitialState: (base) => overrideSet(base, { coins: reuse.initialCoins, wood: 0 })
  }));
  assertEqual(getConfig().initialCoins, PRESETS.reuse.initialCoins, 'default Start restores preset coins');
  resetAll();
  assertEqual(getConfig().initialCoins, PRESETS.reuse.initialCoins, 'default Reset keeps preset coins');
  assert(Object.keys(getConfig().initialStash).length > 0, 'default Reset restores preset stash');
});

test('onPresetChange refreshes baseConfig and re-prefills the panel from the new preset', async () => {
  let prefilled = null;
  let refreshed = 0;
  const base = await onPresetChange('empty', {
    loadConfig: async (preset) => baseConfigFor(preset),
    panel: { prefill: (cfg) => { prefilled = cfg; } },
    onRefresh: () => { refreshed += 1; }
  });
  assertEqual(base.initialCoins, 0, 'empty preset loaded');
  assertEqual(getBaseConfig(), base, 'baseConfig refreshed in the preset-change path');
  assertEqual(prefilled, base, 'panel re-prefilled from the newly loaded preset');
  assertEqual(refreshed, 1, 'diagnostics refreshed once');
});

test('forged overrides are rejected before START.config or createEngine can observe them', () => {
  const base = baseConfigFor('reuse');
  const before = getConfig();
  const forged = overrideSet(base);
  forged.initialCoins = 1e9 + 1;
  let reached = false;
  assertThrows(() => applyOverridesAndStart(base, forged, {
    onStart: () => { reached = true; },
    onCreateEngine: () => { reached = true; }
  }));
  assert(!reached, 'no downstream consumer sees a forged override');
  assertEqual(getConfig(), before, 'module config unchanged on rejection');
});

/* ------------------ simulated-time clock integration (tick-driven) --------- */

function fakeSimClock() {
  return {
    updates: [], resets: 0, freezes: [], frozen: false,
    update(t) { this.updates.push(t); },
    reset() { this.resets += 1; this.frozen = false; },
    freeze(t) { this.freezes.push(t); this.frozen = true; },
    isFrozen() { return this.frozen; },
    getMinutes() { return (this.updates.length ? this.updates[this.updates.length - 1] : 0) * 5; }
  };
}

test('sim clock resets to 0 on the Start/Apply path via an injected clock', async () => {
  const clock = fakeSimClock();
  await startAll(noopDeps({ simClock: clock }));
  assertEqual(clock.resets, 1, 'startAll (Start/Apply) resets the sim clock to 0');
  assertEqual(clock.freezes.length, 0, 'no completion freeze on start');
});

test('sim clock resets to 0 on the Reset path via an injected clock', () => {
  const clock = fakeSimClock();
  resetAll({ simClock: clock, panel: { prefill: () => {} } });
  assertEqual(clock.resets, 1, 'resetAll resets the sim clock to 0');
  assertEqual(clock.updates.length, 0, 'Reset never advances the clock');
});

test('updateSimClockFromSnapshot applies each injected snapshot tick to the clock', () => {
  const clock = fakeSimClock();
  for (const snap of [{ tick: 0 }, { tick: 3 }, { tick: 11 }, { tick: -2 }, {}]) {
    updateSimClockFromSnapshot(snap, clock);
  }
  assertDeepEqual(clock.updates, [0, 3, 11, 0, 0],
    'one update per snapshot, taken from snapshot.tick, invalid ticks normalized');
});

test('runDisplay updates the sim clock per snapshot and freezes on result.completionTick', async () => {
  const clock = fakeSimClock();
  const ticks = [0, 1, 2, 7];
  let idx = 0;
  const engine = {
    step: () => [],
    snapshot: () => {
      const tick = ticks[Math.min(idx, ticks.length - 1)];
      idx += 1;
      return { tick, done: idx >= ticks.length };
    },
    result: () => ({ completionTick: 3, elapsedTicks: 8 })
  };
  let sleeps = 0;
  await runDisplay(getRunToken(), {
    buildEngine: () => engine,
    simClock: clock,
    sleep: () => { sleeps += 1; }
  });
  assertDeepEqual(clock.updates, ticks,
    'one sim-clock update per display snapshot, from snapshot.tick');
  assertEqual(sleeps, ticks.length - 1, 'no real timer waiting: injected sleep counted only between ticks');
  assertDeepEqual(clock.freezes, [3], 'freeze uses result().completionTick, not the last snapshot tick');
  assertEqual(clock.isFrozen(), true, 'clock is frozen on completion');
});
