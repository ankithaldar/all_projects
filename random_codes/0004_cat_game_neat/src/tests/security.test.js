/**
 * security.test.js — allowlist validation, worker-message shape checks,
 * UI control validation and XSS-safe text sanitization.
 */
import { test, assert, assertEqual, assertDeepEqual, assertThrows } from './harness.js';
import {
  validateConfig, validateWorkerMessage, validateControl, sanitizeText,
  validateText, KNOWN_ITEM_IDS, KNOWN_CRAFTABLE_IDS, ERROR_CODES, isPlainObject
} from '../validate.js';
// Namespace import so a not-yet-implemented export fails as a normal test
// failure (RED) instead of aborting the whole test module at load time.
import * as validate from '../validate.js';
import { EMBEDDED_RAW } from '../config.js';

const EVIL = '<img src=x onerror=alert(1)>';

test('validateConfig: accepts the real embedded config', () => {
  assertEqual(validateConfig(EMBEDDED_RAW), true);
  assertEqual(KNOWN_ITEM_IDS.length, 23);
});

test('validateConfig: rejects unknown ids, bad numbers and oversized arrays', () => {
  const bad = (mutate) => {
    const cfg = JSON.parse(JSON.stringify(EMBEDDED_RAW));
    mutate(cfg);
    return cfg;
  };
  assertThrows(() => validateConfig(bad((c) => { c.items[0].id = 'not-an-item'; })));
  assertThrows(() => validateConfig(bad((c) => { c.items[4].recipe.ingredients[0].quantity = 0; })));
  assertThrows(() => validateConfig(bad((c) => { c.items[4].recipe.coin_cost = Infinity; })));
  assertThrows(() => validateConfig(bad((c) => { c.items[4].recipe.craft_time = 0; }))); // F2
  assertThrows(() => validateConfig(bad((c) => { c.items[4].recipe.craft_time = -1; })));
  assertThrows(() => validateConfig(bad((c) => { c.targets.artifact = -1; })));
  assertThrows(() => validateConfig(bad((c) => { c.items.push({ ...c.items[0] }); })));
  assertThrows(() => validateConfig({ items: [], targets: {} }));
});

test('validateWorkerMessage: accepts and sanitizes a valid message', () => {
  const out = validateWorkerMessage({ type: 'PROGRESS', pct: 42, reason: EVIL });
  assertEqual(out.type, 'PROGRESS');
  assertEqual(out.pct, 42);
  assertEqual(out.reason, 'img src=x onerror=alert(1)');
  assert(!out.reason.includes('<') && !out.reason.includes('>'));
});

test('validateWorkerMessage: rejects unknown types, bad numerics, forbidden keys', () => {
  assertThrows(() => validateWorkerMessage({ type: 'EVIL' }));
  assertThrows(() => validateWorkerMessage({ type: 'DONE', value: -1 }));
  assertThrows(() => validateWorkerMessage({ type: 'DONE', value: NaN }));
  assertThrows(() => validateWorkerMessage({ type: 'DONE', value: 1e13 }));
  assertThrows(() => validateWorkerMessage({ type: 'DONE', reason: 'x'.repeat(200) }));
  assertThrows(() => validateWorkerMessage(JSON.parse('{"type":"DONE","__proto__":{"x":1}}')));
});

const VALID_GENOME_RESULT = {
  type: 'GENOME_RESULT', itemId: 'silver', generation: 1, genomeIndex: 2,
  fitness: -123, feasible: false, completionTick: -1, elapsedTicks: 100, totalSpent: 5000,
  perItem: {
    silver: {
      coinsSpent: 5000, timeMinutes: 100, busyTicks: 20, batches: 2,
      units: 10, firstTick: 0, lastTick: 50
    }
  }
};

test('validateWorkerMessage: GENOME_RESULT accepts valid nested perItem breakdown', () => {
  const out = validateWorkerMessage(VALID_GENOME_RESULT);
  assertEqual(out.type, 'GENOME_RESULT');
  assertEqual(out.fitness, -123); // negative fitness (infeasible) must be allowed
  assertEqual(Object.getPrototypeOf(out.perItem), null); // null-prototype map
  assertDeepEqual(out.perItem.silver, VALID_GENOME_RESULT.perItem.silver);
});

test('validateWorkerMessage: GENOME_RESULT rejects __proto__ payload in perItem', () => {
  const payload = JSON.stringify(VALID_GENOME_RESULT)
    .replace('"silver":{', '"__proto__":{"polluted":true},"silver":{');
  const evil = JSON.parse(payload);
  assert(Object.prototype.hasOwnProperty.call(evil.perItem, '__proto__'));
  assertThrows(() => validateWorkerMessage(evil));
  assertEqual({}.polluted, undefined); // no prototype pollution escaped
});

test('validateWorkerMessage: perItem rejects unknown keys, forbidden keys and bad numerics', () => {
  const clone = (row) => JSON.parse(JSON.stringify({
    ...VALID_GENOME_RESULT, perItem: { silver: row }
  }));
  const row = VALID_GENOME_RESULT.perItem.silver;
  assertThrows(() => validateWorkerMessage({ ...VALID_GENOME_RESULT, perItem: { nope: row } }));
  assertThrows(() => validateWorkerMessage(JSON.parse(
    JSON.stringify(VALID_GENOME_RESULT).replace('"silver":{', '"constructor":{},"silver":{'))));
  assertThrows(() => validateWorkerMessage(clone({ ...row, coinsSpent: -1 })));
  assertThrows(() => validateWorkerMessage(clone({ ...row, busyTicks: 99999 })));
  assertThrows(() => validateWorkerMessage(clone({ ...row, firstTick: -5 })));
  assertThrows(() => validateWorkerMessage(clone({ ...row, timeMinutes: Infinity })));
  assertThrows(() => validateWorkerMessage({ ...VALID_GENOME_RESULT, perItem: null }));
  assertThrows(() => validateWorkerMessage({ type: 'GENOME_RESULT', totalSpent: 1 })); // perItem required
});

const VALID_GENOME = {
  nodes: [
    { id: 0, type: 'input', activation: 'IDENTITY', bias: 0, depth: 0, frozen: false },
    { id: 1, type: 'output', activation: 'LOGISTIC', bias: 0.5, depth: 1, frozen: false }
  ],
  genes: [
    { innovation: 0, from: 0, to: 1, weight: 0.25, enabled: true, frozen: false }
  ],
  frozen: false, fitness: 123.5, adjustedFitness: 100, speciesId: -1
};

test('validateWorkerMessage: BEST/DONE accept and deep-copy valid genome payloads', () => {
  const best = validateWorkerMessage({
    type: 'BEST', itemId: 'silver', generation: 1, fitness: 123.5, feasible: true,
    genome: VALID_GENOME
  });
  assertDeepEqual(best.genome, VALID_GENOME);
  assert(best.genome !== VALID_GENOME, 'genome must be a copy');
  assert(best.genome.nodes[0] !== VALID_GENOME.nodes[0], 'nodes deep-copied');
  assert(best.genome.genes[0] !== VALID_GENOME.genes[0], 'genes deep-copied');

  const done = validateWorkerMessage({
    type: 'DONE', feasible: true, completionTick: 100, totalSpent: 5,
    genomeCount: 19, telemetryCount: 50,
    genomes: { silver: VALID_GENOME, artifact: VALID_GENOME }
  });
  assertEqual(Object.getPrototypeOf(done.genomes), null);
  assertEqual(Object.keys(done.genomes).length, 2);
  assertDeepEqual(done.genomes.silver, VALID_GENOME);
  assert(done.genomes.silver !== VALID_GENOME);
});

test('validateWorkerMessage: rejects oversized genomes (nodes > 128, genes > 512)', () => {
  const tooManyNodes = {
    ...VALID_GENOME,
    nodes: Array.from({ length: 129 }, (_unused, i) => ({ ...VALID_GENOME.nodes[0], id: i }))
  };
  const tooManyGenes = {
    ...VALID_GENOME,
    genes: Array.from({ length: 513 }, (_unused, i) => ({ ...VALID_GENOME.genes[0], innovation: i }))
  };
  assertThrows(() => validateWorkerMessage({ type: 'BEST', genome: tooManyNodes }));
  assertThrows(() => validateWorkerMessage({ type: 'BEST', genome: tooManyGenes }));
  assertThrows(() => validateWorkerMessage({ type: 'DONE', genomes: { artifact: tooManyNodes } }));
});

test('validateWorkerMessage: rejects __proto__ payload inside node/gene/genomes', () => {
  const nodeEvil = JSON.parse(
    '{"type":"BEST","genome":{"nodes":[{"id":0,"type":"input","activation":"IDENTITY",' +
    '"bias":0,"__proto__":{"polluted":true},"depth":0,"frozen":false}],' +
    '"genes":[],"frozen":false,"fitness":0}}'
  );
  const geneEvil = JSON.parse(
    '{"type":"BEST","genome":{"nodes":[],"genes":[{"innovation":0,"from":0,"to":1,' +
    '"__proto__":{"polluted":true},"weight":0,"enabled":true,"frozen":false}],' +
    '"frozen":false,"fitness":0}}'
  );
  const mapEvil = JSON.parse(
    '{"type":"DONE","genomes":{"__proto__":{"polluted":true},"silver":' +
    JSON.stringify(VALID_GENOME) + '}}'
  );
  assertThrows(() => validateWorkerMessage(nodeEvil));
  assertThrows(() => validateWorkerMessage(geneEvil));
  assertThrows(() => validateWorkerMessage(mapEvil));
  assertEqual({}.polluted, undefined);
});

test('validateWorkerMessage: DONE.genomes allows only the 19 craftable ids', () => {
  assertThrows(() => validateWorkerMessage({ type: 'DONE', genomes: { nope: VALID_GENOME } }));
  assertThrows(() => validateWorkerMessage({ type: 'DONE', genomes: { cotton: VALID_GENOME } })); // base
  assertThrows(() => validateWorkerMessage({ type: 'DONE', genomes: VALID_GENOME })); // not a map
  const all = {};
  for (const id of KNOWN_CRAFTABLE_IDS) all[id] = VALID_GENOME;
  assertEqual(Object.keys(all).length, 19);
  const out = validateWorkerMessage({ type: 'DONE', genomes: all });
  assertEqual(Object.keys(out.genomes).length, 19);
});

test('validateControl: allowlists speed/seed/preset/popsize/generations', () => {
  assertEqual(validateControl('speed', '10x'), '10x');
  assertEqual(validateControl('seed', 4294967295), 4294967295);
  assertEqual(validateControl('preset', 'reuse'), 'reuse');
  assertEqual(validateControl('pop-size', 500), 500);
  assertEqual(validateControl('generations', 10000), 10000);
  assertThrows(() => validateControl('speed', '100x'));
  assertThrows(() => validateControl('seed', -1));
  assertThrows(() => validateControl('seed', 4294967296));
  assertThrows(() => validateControl('pop-size', 0));
  assertThrows(() => validateControl('generations', 10001));
  assertThrows(() => validateControl('unknown', 1));
});

test('XSS: markup as item name and reason is neutralized (no element/handler possible)', () => {
  const cleanName = sanitizeText(EVIL);
  const cleanReason = sanitizeText(EVIL, 120);
  for (const s of [cleanName, cleanReason]) {
    assert(!s.includes('<') && !s.includes('>'), 'angle brackets stripped');
    assert(!s.includes('"') && !s.includes("'"), 'quotes stripped');
    assert(!s.includes('`'), 'backticks stripped');
    assertEqual(s, 'img src=x onerror=alert(1)');
  }
  assertThrows(() => validateText(EVIL));
  assertEqual(validateText('silver'), 'silver');
  // Sanitization must remove the tag entirely.
  assert(!sanitizeText(EVIL).includes('<img'));
});

test('error codes are stable strings', () => {
  assertEqual(ERROR_CODES.CONFIG_INVALID, 'CONFIG_INVALID');
  assertEqual(ERROR_CODES.MESSAGE_INVALID, 'MESSAGE_INVALID');
  assertEqual(ERROR_CODES.CONTROL_INVALID, 'CONTROL_INVALID');
  assertEqual(ERROR_CODES.TEXT_INVALID, 'TEXT_INVALID');
  assertEqual(ERROR_CODES.DISPLAY_FAILED, 'DISPLAY_FAILED');
  assertEqual(ERROR_CODES.CONFIG_FAILED, 'CONFIG_FAILED');
  assertEqual(ERROR_CODES.CONFIG_INPUT_INVALID, 'CONFIG_INPUT_INVALID'); // initial-state input fallback code
  assert(isPlainObject({}));
  assert(!isPlainObject([]));
});

/* ---------------- editable initial-state validation (TASK A) ---------------- */

const TARGET_IDS = () => validate.KNOWN_TARGET_IDS || []; // may not exist yet (RED)

function nullMap(ids, value) {
  const m = Object.create(null);
  for (const id of ids) m[id] = value;
  return m;
}

function validOverrides() {
  return {
    targets: nullMap(TARGET_IDS(), 2),
    initialCoins: 1000,
    initialStash: nullMap(KNOWN_CRAFTABLE_IDS, 3)
  };
}

test('validateConfig: {items,targets} without optional initial-state keys is valid', () => {
  assertEqual(validateConfig({ items: EMBEDDED_RAW.items, targets: EMBEDDED_RAW.targets }), true);
});

test('validateConfig: optional initialCoins/initialStash accepted when in range', () => {
  const cfg = {
    items: EMBEDDED_RAW.items,
    targets: EMBEDDED_RAW.targets,
    initialCoins: 1e9,
    initialStash: nullMap(KNOWN_CRAFTABLE_IDS, 1e9)
  };
  assertEqual(validateConfig(cfg), true);
});

test('validateConfig: optional initialCoins rejects NaN/fractional/negative/over-cap', () => {
  const base = () => ({ items: EMBEDDED_RAW.items, targets: EMBEDDED_RAW.targets });
  for (const bad of [NaN, Infinity, -1, 1.5, 1e9 + 1, '1', null]) {
    assertThrows(() => validateConfig({ ...base(), initialCoins: bad }),
      `initialCoins must reject ${String(bad)}`);
  }
});

test('validateConfig: optional initialStash rejects bad keys/values but allows empty', () => {
  const base = () => ({ items: EMBEDDED_RAW.items, targets: EMBEDDED_RAW.targets });
  assertEqual(validateConfig({ ...base(), initialStash: {} }), true);
  assertThrows(() => validateConfig({ ...base(), initialStash: null }));
  assertThrows(() => validateConfig({ ...base(), initialStash: { nope: 1 } }));
  assertThrows(() => validateConfig({ ...base(), initialStash: { cotton: 1 } })); // base id
  assertThrows(() => validateConfig({ ...base(), initialStash: JSON.parse('{"__proto__":1}') }));
  assertThrows(() => validateConfig({ ...base(), initialStash: { silver: -1 } }));
  assertThrows(() => validateConfig({ ...base(), initialStash: { silver: 1e9 + 1 } }));
  assertThrows(() => validateConfig({ ...base(), initialStash: { silver: 1.5 } }));
});

test('validateInitialState: accepts a complete null-prototype override set', () => {
  assertEqual(typeof validate.validateInitialState, 'function');
  assertEqual(validate.validateInitialState(validOverrides()), true);
  assertEqual(KNOWN_CRAFTABLE_IDS.length, 19);
  assertEqual(TARGET_IDS().length, 11);
});

test('validateInitialState: requires all three fields and plain override object', () => {
  assertEqual(typeof validate.validateInitialState, 'function');
  const f = (o) => validate.validateInitialState(o);
  assertThrows(() => f(null));
  assertThrows(() => f([]));
  assertThrows(() => f({}));
  const ok = validOverrides();
  assertThrows(() => f({ initialCoins: ok.initialCoins, initialStash: ok.initialStash }));
  assertThrows(() => f({ targets: ok.targets, initialStash: ok.initialStash }));
  assertThrows(() => f({ targets: ok.targets, initialCoins: ok.initialCoins }));
});

test('validateInitialState: requires null-prototype maps (no prototype-pollution carriers)', () => {
  assertEqual(typeof validate.validateInitialState, 'function');
  const f = (o) => validate.validateInitialState(o);
  const ok = validOverrides();
  assertThrows(() => f({ ...ok, targets: { ...ok.targets } }));
  assertThrows(() => f({ ...ok, initialStash: { ...ok.initialStash } }));
  // Also reject a map whose prototype was poisoned via a forged __proto__ value.
  const poisoned = Object.create({ polluted: true });
  for (const id of KNOWN_CRAFTABLE_IDS) poisoned[id] = 1;
  assertThrows(() => f({ ...ok, initialStash: poisoned }));
  assertEqual({}.polluted, undefined);
});

test('validateInitialState: rejects forbidden keys in targets/stash maps', () => {
  assertEqual(typeof validate.validateInitialState, 'function');
  const f = (o) => validate.validateInitialState(o);
  const ok = validOverrides();
  for (const key of ['__proto__', 'constructor', 'prototype']) {
    const t = nullMap(TARGET_IDS(), 1);
    Object.defineProperty(t, key, { value: { polluted: true }, enumerable: true, configurable: true });
    assertThrows(() => f({ ...ok, targets: t }), `${key} in targets must be rejected`);
    const s = nullMap(KNOWN_CRAFTABLE_IDS, 1);
    Object.defineProperty(s, key, { value: { polluted: true }, enumerable: true, configurable: true });
    assertThrows(() => f({ ...ok, initialStash: s }), `${key} in stash must be rejected`);
  }
  assertEqual({}.polluted, undefined);
});

test('validateInitialState: rejects unknown/base ids in targets and stash', () => {
  assertEqual(typeof validate.validateInitialState, 'function');
  const f = (o) => validate.validateInitialState(o);
  const ok = validOverrides();
  const t = nullMap(TARGET_IDS(), 1); t['not-an-item'] = 1;
  assertThrows(() => f({ ...ok, targets: t }));
  const t2 = nullMap(TARGET_IDS(), 1); t2.cotton = 1; // base item is not a target
  assertThrows(() => f({ ...ok, targets: t2 }));
  const s = nullMap(KNOWN_CRAFTABLE_IDS, 1); s['not-an-item'] = 1;
  assertThrows(() => f({ ...ok, initialStash: s }));
  const s2 = nullMap(KNOWN_CRAFTABLE_IDS, 1); s2.rock = 1; // base item cannot be stashed
  assertThrows(() => f({ ...ok, initialStash: s2 }));
});

test('validateInitialState: single numeric schema — targets [0,1e7], coins/stash [0,1e9], ints only', () => {
  const f = (o) => validate.validateInitialState(o);
  const ok = validOverrides();
  // Boundary values are accepted.
  const edge = {
    targets: nullMap(TARGET_IDS(), 1e7),
    initialCoins: 1e9,
    initialStash: nullMap(KNOWN_CRAFTABLE_IDS, 1e9)
  };
  assertEqual(f(edge), true);
  assertEqual(f({ ...edge, targets: nullMap(TARGET_IDS(), 0), initialCoins: 0,
    initialStash: nullMap(KNOWN_CRAFTABLE_IDS, 0) }), true);
  for (const bad of [NaN, Infinity, -1, 1.5, '5', null]) {
    assertThrows(() => f({ ...ok, initialCoins: bad }), `coins reject ${String(bad)}`);
    const t = nullMap(TARGET_IDS(), 1); t.silver = bad;
    assertThrows(() => f({ ...ok, targets: t }), `target reject ${String(bad)}`);
    const s = nullMap(KNOWN_CRAFTABLE_IDS, 1); s.silver = bad;
    assertThrows(() => f({ ...ok, initialStash: s }), `stash reject ${String(bad)}`);
  }
  const tOver = nullMap(TARGET_IDS(), 1); tOver.silver = 1e7 + 1;
  assertThrows(() => f({ ...ok, targets: tOver }));
  const sOver = nullMap(KNOWN_CRAFTABLE_IDS, 1); sOver.silver = 1e9 + 1;
  assertThrows(() => f({ ...ok, initialStash: sOver }));
  assertThrows(() => f({ ...ok, initialCoins: 1e9 + 1 }));
});

test('validateInitialState: forged XSS/prototype payloads never pass', () => {
  assertEqual(typeof validate.validateInitialState, 'function');
  const f = (o) => validate.validateInitialState(o);
  const ok = validOverrides();
  const evil = JSON.parse('{"targets":{"__proto__":{"polluted":true}},"initialCoins":1,"initialStash":{}}');
  assertThrows(() => f(evil));
  assertThrows(() => f(JSON.parse('{"targets":{"<img src=x onerror=alert(1)>":1},"initialCoins":1,"initialStash":{}}')));
  assertThrows(() => f({ ...ok, initialCoins: '1e9' }));
  assertEqual({}.polluted, undefined);
});
