/**
 * config.test.js — constants, embedded/YAML parity, loader fallbacks & caps.
 */
import { test, assert, assertEqual, assertDeepEqual, assertThrows } from './harness.js';
import {
  loadConfig, deriveConstants, EMBEDDED_RAW, EMBEDDED_CONFIG, PRESETS, normalizeItems
} from '../config.js';
// Namespace imports so missing exports fail as normal RED test failures rather
// than aborting the whole module at load time.
import * as cfg from '../config.js';
import * as validate from '../validate.js';
import { validateConfig } from '../validate.js';

/**
 * Read the REAL data files (fs under Node, fetch in the browser) and parse
 * them with js-yaml 4.1.1 when present (tests.html loads it via CDN). Under
 * Node we use a small indentation-based parser for this YAML subset — it still
 * parses the real file text; it never returns the embedded JS copy.
 */
async function readYamlText(name) {
  const url = new URL(`../../config/${name}`, import.meta.url);
  if (typeof process !== 'undefined' && process.versions && process.versions.node) {
    const fs = await import('node:fs');
    return fs.readFileSync(url, 'utf8');
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error('HTTP ' + res.status);
  return res.text();
}

function parseYamlSubset(text) {
  const lines = [];
  for (const raw of text.split(/\r?\n/)) {
    const clean = raw.replace(/(^|\s)#.*$/, '').replace(/\s+$/, '');
    if (!clean.trim()) continue;
    lines.push({ indent: /^ */.exec(clean)[0].length, text: clean.trim() });
  }
  let i = 0;
  const scalar = (v) => {
    if (v === 'true') return true;
    if (v === 'false') return false;
    if (v === 'null' || v === '~') return null;
    if (/^[-+]?\d+$/.test(v) || /^[-+]?\d*\.\d+$/.test(v)) return Number(v);
    return v.replace(/^['"]|['"]$/g, '');
  };
  const kv = (t) => {
    const m = /^([^:]+):\s*(.*)$/.exec(t);
    return m ? [m[1].trim(), m[2].trim()] : null;
  };
  function block(indent) {
    if (i >= lines.length || lines[i].indent < indent) return undefined;
    const seq = lines[i].text.startsWith('- ');
    const out = seq ? [] : {};
    while (i < lines.length && lines[i].indent === indent) {
      const line = lines[i];
      if (seq) {
        if (!line.text.startsWith('- ')) break;
        const rest = line.text.slice(2);
        i += 1;
        const pair = kv(rest);
        if (!pair) { out.push(scalar(rest)); continue; }
        const obj = {};
        obj[pair[0]] = pair[1] === '' ? block(indent + 2) : scalar(pair[1]);
        while (i < lines.length && lines[i].indent === indent + 2 && !lines[i].text.startsWith('- ')) {
          const more = kv(lines[i].text);
          if (!more) break;
          i += 1;
          obj[more[0]] = more[1] === '' ? block(indent + 4) : scalar(more[1]);
        }
        out.push(obj);
      } else {
        const pair = kv(line.text);
        if (!pair) { i += 1; continue; }
        i += 1;
        out[pair[0]] = pair[1] === '' ? block(indent + 2) : scalar(pair[1]);
      }
    }
    return out;
  }
  return block(lines.length ? lines[0].indent : 0);
}

function parseRealYaml(text) {
  if (globalThis.jsyaml && typeof globalThis.jsyaml.load === 'function') {
    return globalThis.jsyaml.load(text, {
      schema: globalThis.jsyaml.CORE_SCHEMA, json: true
    });
  }
  return parseYamlSubset(text);
}

/** Read the REAL saved initial-state JSON (fs under Node, fetch in browser). */
async function readInitialStateText() {
  const url = new URL('../../data/initial-state.json', import.meta.url);
  if (typeof process !== 'undefined' && process.versions && process.versions.node) {
    const fs = await import('node:fs');
    return fs.readFileSync(url, 'utf8');
  }
  const res = await fetch(url); // real fetch, before any test stub is installed
  if (!res.ok) throw new Error('HTTP ' + res.status);
  return res.text();
}

function fakeResponse(text, { ok = true, status = 200 } = {}) {
  return { ok, status, headers: { get: () => null }, text: async () => text };
}

/** Stub fetch: saved-state JSON is served; YAML fetches fail -> embedded fallback. */
function stubSavedState(text) {
  return async (url) => {
    if (String(url).includes('initial-state.json')) return fakeResponse(text);
    throw new Error('NO_CONFIG_NETWORK');
  };
}

test('deriveConstants: exact plan-locked values', () => {
  assertDeepEqual(deriveConstants(), {
    MAX_TICKS: 8064, TICK_MINUTES: 5, COINS_PER_TICK: 210,
    MAX_BATCH: 20, BIG: 100000, LOG_CAP: 5000
  });
});

test('embedded config matches YAML facts (23 items, 19 craftable, 11 targets)', () => {
  assertEqual(EMBEDDED_CONFIG.items.length, 23);
  assertEqual(EMBEDDED_CONFIG.items.filter((i) => i.type === 'craftable').length, 19);
  assertEqual(Object.keys(EMBEDDED_CONFIG.targets).length, 11);
  const byId = new Map(EMBEDDED_CONFIG.items.map((i) => [i.id, i]));
  assertEqual(byId.get('string').craftTimeMinutes, 5);
  assertEqual(byId.get('artifact').craftTimeMinutes, 4320);
  assertEqual(byId.get('artifact').coinCost, 10000);
  assertDeepEqual(byId.get('artifact').ingredients, [
    { item: 'necklace', quantity: 1 }, { item: 'elementstone', quantity: 1 }
  ]);
  assertEqual(EMBEDDED_CONFIG.targets.silver, 12);
  assertEqual(EMBEDDED_CONFIG.targets.artifact, 3);
  assertEqual(EMBEDDED_RAW.items.length, 23);
});

test('normalizeItems drops require_crafted and maps YAML keys to canonical shape', () => {
  const norm = normalizeItems([{ id: 'x', type: 'craftable', tier: 1, recipe: {
    ingredients: [{ item: 'a', quantity: 2, require_crafted: true }], coin_cost: 50, craft_time: 5
  } }]);
  assertDeepEqual(norm, [{
    id: 'x', name: 'X', tier: 1, type: 'craftable',
    ingredients: [{ item: 'a', quantity: 2 }], coinCost: 50, craftTimeMinutes: 5
  }]);
});

test('loadConfig: defaults to the empty preset (0 coins/stash) with YAML targets', async () => {
  const c = await loadConfig();
  assertDeepEqual(Object.keys(c).sort(),
    ['initialCoins', 'initialStash', 'items', 'starterMode', 'targets']);
  assertEqual(c.initialCoins, 0);
  assertEqual(c.starterMode, 'FREE_INGREDIENTS');
  assertDeepEqual(c.initialStash, {});
  assertDeepEqual(c.items, EMBEDDED_CONFIG.items);
  assertDeepEqual(c.targets, EMBEDDED_CONFIG.targets);
});

test('fresh load / refresh is stateless: 0 coins + empty stash + YAML targets every time', async () => {
  const first = await loadConfig();
  const a = cfg.prefillInitialState(first);
  assertEqual(a.initialCoins, 0);
  for (const id of validate.KNOWN_CRAFTABLE_IDS) {
    assertEqual(a.initialStash[id], 0, `${id} stash is 0 on a fresh load`);
  }
  for (const id of validate.KNOWN_TARGET_IDS) {
    assertEqual(a.targets[id], EMBEDDED_RAW.targets[id], `${id} target comes from YAML`);
  }

  // Simulate user edits, then a page refresh (re-import / re-initialise).
  a.initialCoins = 12345;
  a.initialStash.wood = 999;
  a.targets.orb = 999;
  const second = await loadConfig();
  const b = cfg.prefillInitialState(second);
  assertEqual(b.initialCoins, 0, 'refresh resets edited coins to 0');
  for (const id of validate.KNOWN_CRAFTABLE_IDS) {
    assertEqual(b.initialStash[id], 0, `${id} stash is 0 again after refresh`);
  }
  for (const id of validate.KNOWN_TARGET_IDS) {
    assertEqual(b.targets[id], EMBEDDED_RAW.targets[id], `${id} target back to YAML after refresh`);
  }
});

test('loadConfig: empty preset uses zero coins and FREE_INGREDIENTS', async () => {
  const cfg = await loadConfig('empty');
  assertEqual(cfg.initialCoins, 0);
  assertEqual(cfg.starterMode, 'FREE_INGREDIENTS');
  assertDeepEqual(cfg.initialStash, {});
  assertEqual(PRESETS.reuse.initialCoins, 5998996);
});

test('config parity: real YAML files parse equal to the embedded config', async () => {
  const treeText = await readYamlText('crafting_tree.yaml');
  const targetsText = await readYamlText('targets.yaml');
  const raw = {
    items: parseRealYaml(treeText).items,
    targets: parseRealYaml(targetsText).targets
  };
  // The real parsed data must be independently valid, then match embedded.
  assertEqual(validateConfig(raw), true);
  assertDeepEqual(normalizeItems(raw.items), EMBEDDED_CONFIG.items);
  assertDeepEqual(raw.targets, EMBEDDED_CONFIG.targets);
});

test('loadConfig: parses real YAML text through jsyaml (CORE_SCHEMA + json:true)', async () => {
  const treeText = await readYamlText('crafting_tree.yaml');
  const targetsText = await readYamlText('targets.yaml');
  const origFetch = globalThis.fetch;
  const origYaml = globalThis.jsyaml;
  let opts = null;
  let parsedCalls = 0;
  try {
    globalThis.fetch = async (url) => ({
      ok: true, status: 200,
      headers: { get: () => null },
      text: async () => (String(url).includes('crafting_tree') ? treeText : targetsText)
    });
    globalThis.jsyaml = {
      CORE_SCHEMA: 'CORE_SCHEMA_SENTINEL',
      load: (text, o) => {
        opts = o; parsedCalls += 1;
        return parseYamlSubset(text); // real YAML text, parsed directly (no recursion)
      }
    };
    const cfg = await loadConfig();
    assertEqual(parsedCalls, 2);
    assertEqual(opts.schema, 'CORE_SCHEMA_SENTINEL');
    assertEqual(opts.json, true);
    assertDeepEqual(cfg.items, EMBEDDED_CONFIG.items);
    assertDeepEqual(cfg.targets, EMBEDDED_CONFIG.targets);
  } finally {
    globalThis.fetch = origFetch;
    globalThis.jsyaml = origYaml;
  }
});

test('loadConfig: preset ids reuse|empty honored; unknown ids fall back to reuse', async () => {
  // Stub fetch so reuse assertions are deterministic (red-team F2): with no
  // network the YAML path falls back to embedded and the saved-state path falls
  // back to the hardcoded preset (5998996), whether or not the file is served.
  const origFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () => { throw new Error('NO_NETWORK'); };
    const reuse = await loadConfig('reuse');
    assertEqual(reuse.initialCoins, 5998996);
    assertEqual(reuse.starterMode, 'PREDELIVERED_RESERVED');
    const empty = await loadConfig('empty');
    assertEqual(empty.initialCoins, 0);
    assertEqual(empty.starterMode, 'FREE_INGREDIENTS');
    for (const id of ['__proto__', 'constructor', 'unknown']) {
      const cfg = await loadConfig(id);
      assertEqual(cfg.initialCoins, 5998996, `id ${id} must fall back to reuse`);
      assertEqual(cfg.starterMode, 'PREDELIVERED_RESERVED');
    }
  } finally {
    globalThis.fetch = origFetch;
  }
});

test('loadConfig(reuse): reads saved JSON with null-proto maps and preset starterMode', async () => {
  const savedText = await readInitialStateText();
  const saved = JSON.parse(savedText);
  const origFetch = globalThis.fetch;
  let savedFetches = 0;
  try {
    globalThis.fetch = async (url) => {
      if (String(url).includes('initial-state.json')) {
        savedFetches += 1;
        return fakeResponse(savedText);
      }
      throw new Error('NO_CONFIG_NETWORK'); // YAML -> embedded fallback
    };
    const c = await loadConfig('reuse');
    assertEqual(savedFetches, 1, 'saved state fetched exactly once');
    assertEqual(c.initialCoins, saved.initialCoins, 'coins come from saved JSON');
    assertDeepEqual(c.targets, saved.targets, 'targets come from saved JSON');
    assertDeepEqual(c.initialStash, saved.initialStash, 'stash comes from saved JSON');
    assertEqual(Object.getPrototypeOf(c.targets), null, 'targets must be null-proto');
    assertEqual(Object.getPrototypeOf(c.initialStash), null, 'stash must be null-proto');
    assertEqual(c.starterMode, PRESETS.reuse.starterMode, 'starterMode comes from preset, not JSON');
    assertDeepEqual(c.items, EMBEDDED_CONFIG.items, 'items come from YAML/embedded, not JSON');
  } finally {
    globalThis.fetch = origFetch;
  }
});

test('loadConfig(reuse): missing field / null map fall back to hardcoded preset (F1 + C1)', async () => {
  const origFetch = globalThis.fetch;
  try {
    // (a) required field absent: Object.assign would silently accept it.
    globalThis.fetch = stubSavedState('{"targets":{"silver":1},"initialCoins":123}');
    let c = await loadConfig('reuse');
    assertEqual(c.initialCoins, PRESETS.reuse.initialCoins);
    assertDeepEqual(c.targets, EMBEDDED_CONFIG.targets);
    assertDeepEqual(c.initialStash, PRESETS.reuse.initialStash);
    assertEqual(Object.getPrototypeOf(c.targets), Object.prototype);

    // (b) "targets": null — a loose typeof check would accept it (red-team C1).
    globalThis.fetch = stubSavedState('{"targets":null,"initialCoins":123,"initialStash":{}}');
    c = await loadConfig('reuse');
    assertEqual(c.initialCoins, PRESETS.reuse.initialCoins);
    assertDeepEqual(c.targets, EMBEDDED_CONFIG.targets);

    // (c) each required field individually absent.
    for (const missing of ['targets', 'initialCoins', 'initialStash']) {
      const obj = { targets: { silver: 1 }, initialCoins: 123, initialStash: {} };
      delete obj[missing];
      globalThis.fetch = stubSavedState(JSON.stringify(obj));
      c = await loadConfig('reuse');
      assertEqual(c.initialCoins, PRESETS.reuse.initialCoins, `missing ${missing} -> fallback`);
    }
  } finally {
    globalThis.fetch = origFetch;
  }
});

test('loadConfig(reuse): malformed JSON, invalid values and 404 fall back', async () => {
  const origFetch = globalThis.fetch;
  try {
    const expectPreset = (c) => {
      assertEqual(c.initialCoins, PRESETS.reuse.initialCoins);
      assertDeepEqual(c.targets, EMBEDDED_CONFIG.targets);
      assertDeepEqual(c.initialStash, PRESETS.reuse.initialStash);
      assertEqual(c.starterMode, 'PREDELIVERED_RESERVED');
    };

    globalThis.fetch = stubSavedState('{ this is not json');
    expectPreset(await loadConfig('reuse'));

    globalThis.fetch = stubSavedState(JSON.stringify({
      targets: { not_an_id: 1 }, initialCoins: 5, initialStash: {}
    }));
    expectPreset(await loadConfig('reuse')); // validateInitialState throws

    globalThis.fetch = async (url) => {
      if (String(url).includes('initial-state.json')) {
        return fakeResponse('', { ok: false, status: 404 });
      }
      throw new Error('NO_CONFIG_NETWORK');
    };
    expectPreset(await loadConfig('reuse'));
  } finally {
    globalThis.fetch = origFetch;
  }
});

test('loadConfig(empty): ignores saved state even when the JSON is valid', async () => {
  const savedText = await readInitialStateText();
  const origFetch = globalThis.fetch;
  let savedFetches = 0;
  try {
    globalThis.fetch = async (url) => {
      if (String(url).includes('initial-state.json')) {
        savedFetches += 1;
        return fakeResponse(savedText);
      }
      throw new Error('NO_CONFIG_NETWORK');
    };
    const c = await loadConfig('empty');
    assertEqual(savedFetches, 0, 'empty preset must not fetch saved state');
    assertEqual(c.initialCoins, 0);
    assertDeepEqual(c.initialStash, {});
    assertEqual(c.starterMode, 'FREE_INGREDIENTS');
  } finally {
    globalThis.fetch = origFetch;
  }
});

test('256 KiB cap: oversized fetch falls back to embedded without parsing', async () => {
  const origFetch = globalThis.fetch;
  const origYaml = globalThis.jsyaml;
  let parsed = false;
  try {
    globalThis.fetch = async () => ({
      ok: true, status: 200,
      headers: { get: () => null },
      text: async () => 'x'.repeat(256 * 1024 + 1)
    });
    globalThis.jsyaml = {
      CORE_SCHEMA: 1,
      load: () => { parsed = true; throw new Error('SHOULD_NOT_PARSE'); }
    };
    const cfg = await loadConfig();
    assertEqual(parsed, false);
    assertDeepEqual(cfg.items, EMBEDDED_CONFIG.items);
  } finally {
    globalThis.fetch = origFetch;
    globalThis.jsyaml = origYaml;
  }
});

/* ---------------- editable initial-state helpers (TASK A) ---------------- */

const TARGET_IDS = () => validate.KNOWN_TARGET_IDS || []; // may not exist yet (RED)

function nullMap(ids, value) {
  const m = Object.create(null);
  for (const id of ids) m[id] = value;
  return m;
}

function sampleConfig() {
  return {
    items: EMBEDDED_CONFIG.items,
    targets: { ...EMBEDDED_CONFIG.targets },
    initialCoins: 5998996,
    initialStash: { ...PRESETS.reuse.initialStash },
    starterMode: 'PREDELIVERED_RESERVED'
  };
}

function sampleOverrides() {
  const targets = nullMap(TARGET_IDS(), 7);
  const initialStash = nullMap(validate.KNOWN_CRAFTABLE_IDS, 4);
  return { targets, initialCoins: 12345, initialStash };
}

test('applyOverrides: shallow copy assigns validated maps BY REFERENCE, never mutates config', () => {
  assertEqual(typeof cfg.applyOverrides, 'function');
  const base = sampleConfig();
  const overrides = sampleOverrides();
  const beforeTargets = { ...base.targets };
  const beforeStash = { ...base.initialStash };
  const out = cfg.applyOverrides(base, overrides);

  assert(out !== base, 'must return a new config object');
  assertEqual(out.targets, overrides.targets, 'targets assigned by reference');
  assertEqual(out.initialStash, overrides.initialStash, 'initialStash assigned by reference');
  assertEqual(out.initialCoins, overrides.initialCoins);
  assertEqual(Object.getPrototypeOf(out.targets), null); // null prototype preserved
  assertEqual(Object.getPrototypeOf(out.initialStash), null);
  assertEqual(out.items, base.items, 'items untouched');
  assertEqual(out.starterMode, base.starterMode, 'starterMode untouched');

  // Loaded config must be byte-for-byte unchanged.
  assertDeepEqual(base.targets, beforeTargets);
  assertDeepEqual(base.initialStash, beforeStash);
  assertEqual(base.initialCoins, 5998996);
});

test('applyOverrides: forged overrides are rejected before any config is produced', () => {
  assertEqual(typeof cfg.applyOverrides, 'function');
  const base = sampleConfig();
  const ok = sampleOverrides();
  const forged = JSON.parse('{"targets":{"__proto__":{"polluted":true}},"initialCoins":1,"initialStash":{}}');
  assertThrows(() => cfg.applyOverrides(base, forged));
  assertThrows(() => cfg.applyOverrides(base, { ...ok, initialCoins: NaN }));
  assertThrows(() => cfg.applyOverrides(base, { ...ok, initialCoins: 1e9 + 1 }));
  const t = nullMap(TARGET_IDS(), 1); t.silver = 1e7 + 1;
  assertThrows(() => cfg.applyOverrides(base, { ...ok, targets: t }));
  assertEqual({}.polluted, undefined);
  assertEqual(base.initialCoins, 5998996, 'config untouched after rejection');
});

test('applyOverrides: result passes validate.js (the single validation owner)', () => {
  const base = sampleConfig();
  const out = cfg.applyOverrides(base, sampleOverrides());
  // The exact assigned maps must satisfy the mandatory override schema.
  assertEqual(validate.validateInitialState(out), true);
  assertEqual(Object.getPrototypeOf(out.targets), null);
  assertEqual(Object.getPrototypeOf(out.initialStash), null);
  // Forged fields can never enter the result.
  const evil = JSON.parse('{"targets":{"__proto__":{"polluted":true}},"initialCoins":1,"initialStash":{}}');
  assertThrows(() => cfg.applyOverrides(base, evil));
  assertEqual({}.polluted, undefined);
});

test('prefillInitialState: null-proto 11 targets / 19 stash rows from injected config', () => {
  assertEqual(typeof cfg.prefillInitialState, 'function');
  const base = sampleConfig();
  const pre = cfg.prefillInitialState(base);

  assertDeepEqual(Object.keys(pre).sort(), ['initialCoins', 'initialStash', 'targets']);
  assertEqual(Object.getPrototypeOf(pre.targets), null);
  assertEqual(Object.getPrototypeOf(pre.initialStash), null);
  assertEqual(Object.keys(pre.targets).length, 11);
  assertEqual(Object.keys(pre.initialStash).length, 19);
  assertEqual(pre.initialCoins, 5998996);
  for (const id of TARGET_IDS()) {
    assertEqual(pre.targets[id], base.targets[id] === undefined ? 0 : base.targets[id]);
  }
  for (const id of validate.KNOWN_CRAFTABLE_IDS) {
    assertEqual(pre.initialStash[id], base.initialStash[id] === undefined ? 0 : base.initialStash[id]);
  }
});

test('prefillInitialState: pure over injected config — no mutation, independent maps', () => {
  const base = sampleConfig();
  const before = JSON.stringify(base);
  const a = cfg.prefillInitialState(base);
  const b = cfg.prefillInitialState(base);
  assertEqual(JSON.stringify(base), before, 'injected config unchanged');
  assert(a.targets !== b.targets && a.initialStash !== b.initialStash, 'fresh maps per call');
  a.targets.silver = 999;
  assertEqual(b.targets.silver, 12, 'no shared state between calls');
});

test('prefillInitialState: missing/zero fields fall back to 0 keys, not undefined', () => {
  const empty = cfg.prefillInitialState({});
  assertEqual(empty.initialCoins, 0);
  assertEqual(Object.keys(empty.targets).length, 11);
  assertEqual(Object.keys(empty.initialStash).length, 19);
  for (const id of TARGET_IDS()) assertEqual(empty.targets[id], 0);
  for (const id of validate.KNOWN_CRAFTABLE_IDS) assertEqual(empty.initialStash[id], 0);
});
