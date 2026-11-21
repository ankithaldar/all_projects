/**
 * config.js — YAML load (256 KiB cap) + SRI-ready js-yaml parse + allowlist
 * validation + embedded JS fallback identical to the two YAML data files.
 *
 * Browser: include js-yaml 4.1.1 UMD (global `jsyaml`) via <script> before
 * importing this module. Under file:// or CDN failure the embedded copy is
 * the sole source. No DOM access here.
 */
import {
  validateConfig, validateInitialState, isPlainObject, KNOWN_TARGET_IDS, KNOWN_CRAFTABLE_IDS
} from './validate.js';

/** Single source of truth for the batch cap (cost/items re-export from here). */
export const MAX_BATCH = 20;

/** Plan-locked constants. */
export function deriveConstants() {
  return Object.freeze({
    MAX_TICKS: 8064,
    TICK_MINUTES: 5,
    COINS_PER_TICK: 210,
    MAX_BATCH,
    BIG: 100000,
    LOG_CAP: 5000
  });
}

/**
 * ceil(craftTimeMinutes / 5), defensively clamped to >= 1 so a zero/invalid
 * craft time can never produce a slot stuck ACTIVE at remainingTicks 0 (F2).
 */
export function craftTicksForMinutes(craftTimeMinutes) {
  const minutes = Number.isFinite(craftTimeMinutes) && craftTimeMinutes > 0 ? craftTimeMinutes : 1;
  return Math.max(1, Math.ceil(minutes / 5));
}

const MAX_BYTES = 256 * 1024; // 256 KiB fetch cap

/** Preset (a) "reuse": training.yaml initial_stash + 5,998,996 coins. */
const REUSE_STASH = Object.freeze({
  string: 90, wood: 79, ribbon: 58, metal: 41, needles: 27,
  sparkles: 2, bronze: 30, silver: 2, gold: 0, amethyst: 163,
  pendant: 6, necklace: 0, orb: 44, water: 0, fire: 1,
  waterstone: 1, firestone: 3, elementstone: 1, artifact: 0
});

export const PRESETS = Object.freeze({
  reuse: Object.freeze({
    initialCoins: 5998996,
    initialStash: REUSE_STASH,
    starterMode: 'PREDELIVERED_RESERVED'
  }),
  empty: Object.freeze({
    initialCoins: 0,
    initialStash: Object.freeze({}),
    starterMode: 'FREE_INGREDIENTS'
  })
});

/**
 * Embedded raw config — byte-for-byte equivalent (semantically) to
 * config/crafting_tree.yaml + config/targets.yaml. Kept as a JS literal so the
 * app works from file:// with no fetch and no CDN.
 */
export const EMBEDDED_RAW = Object.freeze({
  items: [
    { id: 'cotton', type: 'base', tier: 0 },
    { id: 'tree', type: 'base', tier: 0 },
    { id: 'rock', type: 'base', tier: 0 },
    { id: 'quartz', type: 'base', tier: 0 },
    { id: 'string', type: 'craftable', tier: 1, recipe: { ingredients: [{ item: 'cotton', quantity: 3 }], coin_cost: 50, craft_time: 5 } },
    { id: 'wood', type: 'craftable', tier: 1, recipe: { ingredients: [{ item: 'tree', quantity: 3 }], coin_cost: 50, craft_time: 5 } },
    { id: 'metal', type: 'craftable', tier: 1, recipe: { ingredients: [{ item: 'rock', quantity: 3 }], coin_cost: 100, craft_time: 15 } },
    { id: 'bronze', type: 'craftable', tier: 1, recipe: { ingredients: [{ item: 'rock', quantity: 4 }], coin_cost: 200, craft_time: 30 } },
    { id: 'amethyst', type: 'craftable', tier: 1, recipe: { ingredients: [{ item: 'quartz', quantity: 10 }], coin_cost: 300, craft_time: 60 } },
    { id: 'orb', type: 'craftable', tier: 1, recipe: { ingredients: [{ item: 'quartz', quantity: 20 }], coin_cost: 300, craft_time: 60 } },
    { id: 'ribbon', type: 'craftable', tier: 2, recipe: { ingredients: [{ item: 'string', quantity: 2 }, { item: 'wood', quantity: 2 }], coin_cost: 100, craft_time: 15 } },
    { id: 'needles', type: 'craftable', tier: 2, recipe: { ingredients: [{ item: 'metal', quantity: 4 }, { item: 'ribbon', quantity: 1 }], coin_cost: 200, craft_time: 30 } },
    { id: 'sparkles', type: 'craftable', tier: 3, recipe: { ingredients: [{ item: 'needles', quantity: 2 }, { item: 'ribbon', quantity: 2 }], coin_cost: 300, craft_time: 60 } },
    { id: 'silver', type: 'craftable', tier: 3, recipe: { ingredients: [{ item: 'bronze', quantity: 3 }, { item: 'sparkles', quantity: 1 }], coin_cost: 300, craft_time: 120 } },
    { id: 'gold', type: 'craftable', tier: 4, recipe: { ingredients: [{ item: 'silver', quantity: 2 }, { item: 'sparkles', quantity: 2 }], coin_cost: 500, craft_time: 360 } },
    { id: 'pendant', type: 'craftable', tier: 4, recipe: { ingredients: [{ item: 'amethyst', quantity: 7 }, { item: 'silver', quantity: 2 }], coin_cost: 500, craft_time: 180 } },
    { id: 'water', type: 'craftable', tier: 4, recipe: { ingredients: [{ item: 'orb', quantity: 2 }, { item: 'silver', quantity: 1 }], coin_cost: 800, craft_time: 360 } },
    { id: 'necklace', type: 'craftable', tier: 5, recipe: { ingredients: [{ item: 'pendant', quantity: 3 }, { item: 'gold', quantity: 2 }], coin_cost: 800, craft_time: 760 } },
    { id: 'fire', type: 'craftable', tier: 5, recipe: { ingredients: [{ item: 'orb', quantity: 6 }, { item: 'gold', quantity: 1 }], coin_cost: 1000, craft_time: 720 } },
    { id: 'waterstone', type: 'craftable', tier: 6, recipe: { ingredients: [{ item: 'water', quantity: 2 }, { item: 'ribbon', quantity: 10 }], coin_cost: 1500, craft_time: 720 } },
    { id: 'firestone', type: 'craftable', tier: 6, recipe: { ingredients: [{ item: 'fire', quantity: 1 }, { item: 'sparkles', quantity: 2 }], coin_cost: 1500, craft_time: 720 } },
    { id: 'elementstone', type: 'craftable', tier: 7, recipe: { ingredients: [{ item: 'waterstone', quantity: 1 }, { item: 'firestone', quantity: 1 }], coin_cost: 5000, craft_time: 1440 } },
    { id: 'artifact', type: 'craftable', tier: 8, recipe: { ingredients: [{ item: 'necklace', quantity: 1 }, { item: 'elementstone', quantity: 1 }], coin_cost: 10000, craft_time: 4320 } }
  ],
  targets: Object.freeze({
    silver: 12, gold: 13, pendant: 1, necklace: 15, orb: 20,
    water: 5, fire: 4, waterstone: 6, firestone: 5,
    elementstone: 10, artifact: 3
  })
});

/** Human-readable display name derived from an id. */
export function titleCase(id) {
  return String(id).split('_').map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join(' ');
}

/** Normalize raw YAML/embedded item objects to the canonical ITEMS shape. */
export function normalizeItems(rawItems) {
  return rawItems.map((raw) => {
    if (raw.type === 'base') {
      return { id: raw.id, name: titleCase(raw.id), tier: raw.tier, type: 'base', ingredients: [], coinCost: 0, craftTimeMinutes: 0 };
    }
    const r = raw.recipe;
    return {
      id: raw.id,
      name: titleCase(raw.id),
      tier: raw.tier,
      type: 'craftable',
      ingredients: r.ingredients.map((i) => ({ item: i.item, quantity: i.quantity })),
      coinCost: r.coin_cost,
      craftTimeMinutes: r.craft_time
    };
  });
}

/** Normalized embedded config (the canonical in-memory fallback). */
export const EMBEDDED_CONFIG = Object.freeze({
  items: Object.freeze(normalizeItems(EMBEDDED_RAW.items)),
  targets: EMBEDDED_RAW.targets
});

function byteLength(text) {
  // Accurate for ASCII YAML; fallback that also works without TextEncoder.
  if (typeof TextEncoder !== 'undefined') return new TextEncoder().encode(text).length;
  if (typeof Blob !== 'undefined') return new Blob([text]).size;
  return text.length;
}

async function fetchCapped(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error('HTTP ' + res.status + ' for ' + url);
  const declared = Number(res.headers && res.headers.get && res.headers.get('content-length'));
  if (Number.isFinite(declared) && declared > MAX_BYTES) throw new Error('CONFIG_TOO_LARGE');
  const text = await res.text();
  if (byteLength(text) > MAX_BYTES) throw new Error('CONFIG_TOO_LARGE');
  return text;
}

function parseYaml(text) {
  const y = globalThis.jsyaml;
  if (!y || typeof y.load !== 'function') throw new Error('NO_YAML_PARSER');
  return y.load(text, { schema: y.CORE_SCHEMA, json: true });
}

/**
 * Load config. Returns
 *   { items, targets, initialCoins, initialStash, starterMode }.
 * Tries CDN-parsed YAML first; falls back to embedded on any failure. The
 * default preset is `empty` (0 coins, empty stash), so a fresh load/refresh
 * always returns the shipped YAML targets with no persisted edits.
 *
 * The `reuse` preset additionally loads the locally editable
 * `data/initial-state.json` (the same file tools/best-schedule.mjs reads).
 * Missing file (404 / file://), malformed JSON, a missing field, a non-object
 * map, or a validator rejection all fall back silently to the hardcoded preset
 * — consistent with the YAML fallback above. `empty` is unchanged.
 */
export async function loadConfig(presetId = 'empty') {
  // Strict lookup: only 'empty' and 'reuse' are honoured; ids such as
  // '__proto__' or 'constructor' must not resolve through the prototype chain.
  const preset = presetId === 'empty' ? PRESETS.empty : PRESETS.reuse;
  let raw = null;
  try {
    const [treeText, targetsText] = await Promise.all([
      fetchCapped('./config/crafting_tree.yaml'),
      fetchCapped('./config/targets.yaml')
    ]);
    const tree = parseYaml(treeText);
    const tgt = parseYaml(targetsText);
    raw = { items: tree.items, targets: tgt.targets };
    validateConfig(raw);
  } catch (_err) {
    raw = EMBEDDED_RAW;
  }

  let saved = null;
  if (presetId === 'reuse') {
    try {
      const parsed = JSON.parse(await fetchCapped('./data/initial-state.json'));
      // Explicit presence checks (red-team F1 + C1): do NOT rely on
      // Object.assign(..., parsed && ...) to detect absence — it silently
      // turns undefined/null into an empty map, which the validator accepts.
      // isPlainObject rejects null/arrays, so `"targets": null` cannot pass.
      if (!isPlainObject(parsed)) throw new Error('SAVED_STATE_INVALID');
      if (parsed.targets === undefined || parsed.initialCoins === undefined
          || parsed.initialStash === undefined) {
        throw new Error('SAVED_STATE_MISSING_FIELD');
      }
      if (!isPlainObject(parsed.targets) || !isPlainObject(parsed.initialStash)) {
        throw new Error('SAVED_STATE_MAP_INVALID');
      }
      const overrides = {
        targets: Object.assign(Object.create(null), parsed.targets),
        initialCoins: parsed.initialCoins,
        initialStash: Object.assign(Object.create(null), parsed.initialStash)
      };
      validateInitialState(overrides);
      saved = overrides;
    } catch (_err) {
      saved = null;
    }
  }

  return {
    items: normalizeItems(raw.items),
    targets: saved ? saved.targets : { ...raw.targets },
    initialCoins: saved ? saved.initialCoins : preset.initialCoins,
    initialStash: saved ? saved.initialStash : { ...preset.initialStash },
    starterMode: preset.starterMode
  };
}

/**
 * Apply validated user edits to a loaded config. The overrides MUST first pass
 * validateInitialState (the single validation owner). The validated maps are
 * assigned BY REFERENCE — never spread into a normal object — so their
 * null prototype survives into the worker START config and the engine. The
 * loaded config is never mutated. The result is re-validated with validateConfig.
 */
export function applyOverrides(config, overrides) {
  validateInitialState(overrides);
  const out = {
    ...config,
    targets: overrides.targets,
    initialCoins: overrides.initialCoins,
    initialStash: overrides.initialStash
  };
  // Re-validate the EXACT references that will be used downstream. This keeps
  // the result behind the single validate.js gate and closes a getter-based
  // time-of-check/time-of-use gap between validation and assignment.
  validateInitialState(out);
  return out;
}

/**
 * Pure prefill data for the "Initial state" panel, derived ONLY from an
 * injected config snapshot (not module state). Returns fresh null-prototype
 * maps keyed by the 11 target ids and 19 craftable ids; missing values fall
 * back to 0. Call again after a preset change to re-prefill the panel.
 */
export function prefillInitialState(config = {}) {
  const src = config || {};
  const targets = Object.create(null);
  const initialStash = Object.create(null);
  for (const id of KNOWN_TARGET_IDS) {
    const v = src.targets ? src.targets[id] : undefined;
    targets[id] = v === undefined ? 0 : v;
  }
  for (const id of KNOWN_CRAFTABLE_IDS) {
    const v = src.initialStash ? src.initialStash[id] : undefined;
    initialStash[id] = v === undefined ? 0 : v;
  }
  return {
    targets,
    initialCoins: src.initialCoins === undefined ? 0 : src.initialCoins,
    initialStash
  };
}
