/**
 * validate.js — single owner of config / UI-control / worker-message validators
 * and the text sanitizer used at every DOM boundary (plan Security Controls).
 *
 * No DOM access. Throws Error with a stable `code` property on rejection.
 * main.js imports validateWorkerMessage from here (no private divergent copy).
 */
// Plan-locked horizon; kept local to avoid a config.js <-> validate.js cycle.
const MAX_TICKS = 8064;

/** Allowlist of the 23 item ids in the copied crafting_tree.yaml. */
export const KNOWN_ITEM_IDS = Object.freeze([
  // base (tier 0)
  'cotton', 'tree', 'rock', 'quartz',
  // craftable (tier 1..8)
  'string', 'wood', 'metal', 'bronze', 'amethyst', 'orb',
  'ribbon', 'needles', 'sparkles', 'silver',
  'gold', 'pendant', 'water',
  'necklace', 'fire',
  'waterstone', 'firestone', 'elementstone', 'artifact'
]);

export const ERROR_CODES = Object.freeze({
  CONFIG_INVALID: 'CONFIG_INVALID',
  CONFIG_FAILED: 'CONFIG_FAILED',
  CONFIG_INPUT_INVALID: 'CONFIG_INPUT_INVALID',
  MESSAGE_INVALID: 'MESSAGE_INVALID',
  CONTROL_INVALID: 'CONTROL_INVALID',
  TEXT_INVALID: 'TEXT_INVALID',
  DISPLAY_FAILED: 'DISPLAY_FAILED'
});

/** The 19 craftable ids (bases excluded) — derived, not re-listed. */
export const KNOWN_CRAFTABLE_IDS = Object.freeze(
  KNOWN_ITEM_IDS.filter((id) => id !== 'cotton' && id !== 'tree' && id !== 'rock' && id !== 'quartz')
);

/**
 * The 11 editable target ids. Authoritative allowlist for the initial-state
 * override schema (mirrors config/targets.yaml); validate.js owns it so no
 * config.js <-> validate.js cycle is introduced.
 */
export const KNOWN_TARGET_IDS = Object.freeze([
  'silver', 'gold', 'pendant', 'necklace', 'orb',
  'water', 'fire', 'waterstone', 'firestone',
  'elementstone', 'artifact'
]);

const KNOWN_IDS = new Set(KNOWN_ITEM_IDS);
const CRAFTABLE_IDS = new Set(KNOWN_CRAFTABLE_IDS);
const TARGET_IDS = new Set(KNOWN_TARGET_IDS);
const FORBIDDEN_KEYS = new Set(['__proto__', 'constructor', 'prototype']);
const MAX_NUM = 1e12;

/** One authoritative numeric schema for the editable initial state. */
const MAX_TARGET = 1e7;
const MAX_INITIAL_COINS = 1e9;
const MAX_INITIAL_STASH = 1e9;

// Genome structural caps (mirror inhouse-neat DEFAULT_CONFIG maxNodes/maxGenes).
const MAX_GENOME_NODES = 128;
const MAX_GENOME_GENES = 512;
const ACTIVATIONS = new Set(['LOGISTIC', 'TANH', 'RELU', 'IDENTITY']);
const NODE_TYPES = new Set(['input', 'output', 'hidden']);

function fail(code, message) {
  const err = new Error(message || code);
  err.code = code;
  throw err;
}

export function isPlainObject(o) {
  if (o === null || typeof o !== 'object' || Array.isArray(o)) return false;
  const p = Object.getPrototypeOf(o);
  return p === Object.prototype || p === null;
}

function isFiniteNumber(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

function isBoundedInt(v, min, max) {
  return isFiniteNumber(v) && Number.isInteger(v) && v >= min && v <= max;
}

/**
 * Allowlist + bound a count map (targets / initialStash / override maps).
 * `requireNullProto` is true for user-supplied override maps so a forged
 * prototype can never ride into the worker; loaded config maps are plain
 * objects and are validated with it false. Own keys (including non-enumerable
 * ones) are checked so `__proto__` cannot hide from the allowlist.
 */
function validateCountMap(raw, allowed, max, label, requireNullProto) {
  if (!isPlainObject(raw)) fail(ERROR_CODES.CONFIG_INVALID, `${label} must be an object`);
  if (requireNullProto && Object.getPrototypeOf(raw) !== null) {
    fail(ERROR_CODES.CONFIG_INVALID, `${label} must have a null prototype`);
  }
  for (const key of Object.getOwnPropertyNames(raw)) {
    if (FORBIDDEN_KEYS.has(key)) fail(ERROR_CODES.CONFIG_INVALID, `forbidden ${label} key: ${key}`);
    if (!allowed.has(key)) fail(ERROR_CODES.CONFIG_INVALID, `unknown ${label} id: ${key}`);
    if (!isBoundedInt(raw[key], 0, max)) fail(ERROR_CODES.CONFIG_INVALID, `bad ${label} value for ${key}`);
  }
  return true;
}

/**
 * Escape untrusted text for text-only insertion. Strips angle brackets,
 * ampersands, quotes, backticks and control characters so a payload such as
 * `<img src=x onerror=alert(1)>` can never be parsed as markup.
 */
export function sanitizeText(value, maxLen = 120) {
  let s = (value === null || value === undefined) ? '' : String(value);
  s = s.replace(/[<>&"'`\u0000-\u001f\u007f]/g, '');
  s = s.replace(/\s+/g, ' ').trim();
  if (s.length > maxLen) s = s.slice(0, maxLen);
  return s;
}

/** Strict variant: reject any raw string that contains markup metacharacters. */
export function validateText(value, maxLen = 120) {
  if (typeof value !== 'string') fail(ERROR_CODES.TEXT_INVALID, 'text must be a string');
  if (/[<>&"'`\u0000-\u001f\u007f]/.test(value)) {
    fail(ERROR_CODES.TEXT_INVALID, 'text contains disallowed markup/control characters');
  }
  const s = value.replace(/\s+/g, ' ').trim();
  if (s.length > maxLen) fail(ERROR_CODES.TEXT_INVALID, 'text exceeds maximum length');
  return s;
}

function validateRawItem(raw, seen) {
  if (!isPlainObject(raw)) fail(ERROR_CODES.CONFIG_INVALID, 'item must be an object');
  if (typeof raw.id !== 'string' || !KNOWN_IDS.has(raw.id)) {
    fail(ERROR_CODES.CONFIG_INVALID, 'unknown item id: ' + String(raw.id));
  }
  if (seen.has(raw.id)) fail(ERROR_CODES.CONFIG_INVALID, 'duplicate item id: ' + raw.id);
  seen.add(raw.id);
  if (!isBoundedInt(raw.tier, 0, 8)) fail(ERROR_CODES.CONFIG_INVALID, 'bad tier for ' + raw.id);
  if (raw.type !== 'base' && raw.type !== 'craftable') {
    fail(ERROR_CODES.CONFIG_INVALID, 'bad type for ' + raw.id);
  }
  if (raw.type === 'base') return;

  const recipe = raw.recipe;
  if (!isPlainObject(recipe)) fail(ERROR_CODES.CONFIG_INVALID, 'missing recipe for ' + raw.id);
  if (!isFiniteNumber(recipe.coin_cost) || recipe.coin_cost < 0 || recipe.coin_cost > 1e9) {
    fail(ERROR_CODES.CONFIG_INVALID, 'bad coin_cost for ' + raw.id);
  }
  // F2: craft_time 0 leaves an engine slot ACTIVE at remainingTicks 0 forever.
  if (!isFiniteNumber(recipe.craft_time) || recipe.craft_time < 1 || recipe.craft_time > 1e7) {
    fail(ERROR_CODES.CONFIG_INVALID, 'bad craft_time for ' + raw.id);
  }
  const ingredients = recipe.ingredients;
  if (!Array.isArray(ingredients) || ingredients.length < 1 || ingredients.length > 8) {
    fail(ERROR_CODES.CONFIG_INVALID, 'bad ingredients for ' + raw.id);
  }
  for (const ing of ingredients) {
    if (!isPlainObject(ing)) fail(ERROR_CODES.CONFIG_INVALID, 'bad ingredient in ' + raw.id);
    if (typeof ing.item !== 'string' || !KNOWN_IDS.has(ing.item)) {
      fail(ERROR_CODES.CONFIG_INVALID, 'unknown ingredient id in ' + raw.id);
    }
    if (!isBoundedInt(ing.quantity, 1, 1000)) {
      fail(ERROR_CODES.CONFIG_INVALID, 'bad ingredient quantity in ' + raw.id);
    }
  }
}

/**
 * Validate a raw parsed config ({items, targets, [initialCoins], [initialStash]}).
 * initialCoins/initialStash are OPTIONAL: they are validated only when present
 * (!== undefined) so partial-object callers keep working. Returns true or throws.
 */
export function validateConfig(cfg) {
  if (!isPlainObject(cfg)) fail(ERROR_CODES.CONFIG_INVALID, 'config must be an object');
  if (!Array.isArray(cfg.items) || cfg.items.length === 0 || cfg.items.length > 100) {
    fail(ERROR_CODES.CONFIG_INVALID, 'items must be a bounded array');
  }
  const seen = new Set();
  for (const raw of cfg.items) validateRawItem(raw, seen);
  if (seen.size !== KNOWN_ITEM_IDS.length) {
    fail(ERROR_CODES.CONFIG_INVALID, 'item set does not match allowlist');
  }
  validateCountMap(cfg.targets, KNOWN_IDS, MAX_TARGET, 'targets', false);
  if (cfg.initialCoins !== undefined) {
    if (!isBoundedInt(cfg.initialCoins, 0, MAX_INITIAL_COINS)) {
      fail(ERROR_CODES.CONFIG_INVALID, 'bad initialCoins');
    }
  }
  if (cfg.initialStash !== undefined) {
    validateCountMap(cfg.initialStash, CRAFTABLE_IDS, MAX_INITIAL_STASH, 'initialStash', false);
  }
  return true;
}

/**
 * MANDATORY full schema for user-supplied initial-state overrides.
 * Three fields are required: null-prototype `targets` (11 allowed ids),
 * `initialCoins` and null-prototype `initialStash` (19 craftable ids).
 * Every value is a bounded integer from the single authoritative schema
 * above; forbidden/unknown keys are rejected. Throws with CONFIG_INVALID.
 */
export function validateInitialState(overrides) {
  if (!isPlainObject(overrides)) fail(ERROR_CODES.CONFIG_INVALID, 'initial state must be an object');
  for (const key of Object.getOwnPropertyNames(overrides)) {
    if (FORBIDDEN_KEYS.has(key)) fail(ERROR_CODES.CONFIG_INVALID, 'forbidden initial-state key: ' + key);
  }
  if (overrides.targets === undefined || overrides.initialCoins === undefined || overrides.initialStash === undefined) {
    fail(ERROR_CODES.CONFIG_INVALID, 'initial state requires targets, initialCoins and initialStash');
  }
  validateCountMap(overrides.targets, TARGET_IDS, MAX_TARGET, 'targets', true);
  if (!isBoundedInt(overrides.initialCoins, 0, MAX_INITIAL_COINS)) {
    fail(ERROR_CODES.CONFIG_INVALID, 'bad initialCoins');
  }
  validateCountMap(overrides.initialStash, CRAFTABLE_IDS, MAX_INITIAL_STASH, 'initialStash', true);
  return true;
}

const WORKER_TYPES = new Set(['PROGRESS', 'BEST', 'DONE', 'ERROR', 'GENOME_RESULT']);

// Type-aware numeric bounds. Unlisted numeric keys default to [0, MAX_NUM].
const NUM_RULES = {
  PROGRESS: { generation: [0, 100000], completed: [0, MAX_NUM], total: [0, MAX_NUM] },
  BEST: { generation: [0, 100000], fitness: [-1e15, 1e15] },
  GENOME_RESULT: {
    generation: [0, 100000], genomeIndex: [0, 1e6], fitness: [-1e15, 1e15],
    completionTick: [-1, MAX_TICKS], elapsedTicks: [0, MAX_TICKS + 1], totalSpent: [0, MAX_NUM]
  },
  DONE: {
    completionTick: [-1, MAX_TICKS], totalSpent: [0, MAX_NUM],
    genomeCount: [0, 100], telemetryCount: [0, 5000]
  },
  ERROR: {}
};

// [min, max, mustBeInteger] for every result().perItem row field.
const PER_ITEM_RULES = {
  coinsSpent: [0, MAX_NUM, false],
  timeMinutes: [0, 1e9, false],
  busyTicks: [0, MAX_TICKS, true],
  batches: [0, 1e6, true],
  units: [0, 1e9, true],
  firstTick: [-1, MAX_TICKS, true],
  lastTick: [-1, MAX_TICKS, true]
};

/**
 * Validate a GENOME_RESULT.perItem map. Output is a null-prototype object with
 * at most 23 keys, each a known item id, each row fully numeric-bounded.
 */
function validatePerItem(raw) {
  if (!isPlainObject(raw)) fail(ERROR_CODES.MESSAGE_INVALID, 'perItem must be an object');
  const keys = Object.keys(raw);
  if (keys.length > KNOWN_ITEM_IDS.length) {
    fail(ERROR_CODES.MESSAGE_INVALID, 'perItem key cap exceeded');
  }
  const out = Object.create(null);
  for (const key of keys) {
    if (FORBIDDEN_KEYS.has(key)) fail(ERROR_CODES.MESSAGE_INVALID, 'forbidden perItem key: ' + key);
    if (!KNOWN_IDS.has(key)) fail(ERROR_CODES.MESSAGE_INVALID, 'unknown perItem id: ' + key);
    const row = raw[key];
    if (!isPlainObject(row)) fail(ERROR_CODES.MESSAGE_INVALID, 'perItem row must be an object');
    const clean = {};
    for (const field of Object.keys(PER_ITEM_RULES)) {
      const [min, max, integer] = PER_ITEM_RULES[field];
      const v = row[field];
      if (!isFiniteNumber(v) || v < min || v > max || (integer && !Number.isInteger(v))) {
        fail(ERROR_CODES.MESSAGE_INVALID, `bad perItem.${key}.${field}`);
      }
      clean[field] = v;
    }
    out[key] = clean;
  }
  return out;
}

function assertNoForbiddenKeys(obj, label) {
  for (const key of Object.keys(obj)) {
    if (FORBIDDEN_KEYS.has(key)) {
      fail(ERROR_CODES.MESSAGE_INVALID, `forbidden key in ${label}: ${key}`);
    }
  }
}

function reqNumber(v, min, max, label) {
  if (!isFiniteNumber(v) || v < min || v > max) {
    fail(ERROR_CODES.MESSAGE_INVALID, `bad genome number ${label}`);
  }
  return v;
}

function reqInt(v, min, max, label) {
  if (!isBoundedInt(v, min, max)) {
    fail(ERROR_CODES.MESSAGE_INVALID, `bad genome integer ${label}`);
  }
  return v;
}

function reqBool(v, label) {
  if (typeof v !== 'boolean') fail(ERROR_CODES.MESSAGE_INVALID, `bad genome boolean ${label}`);
  return v;
}

/**
 * Validate + deep-copy a NEAT genome (nodes/genes arrays + metadata).
 * Unknown keys are dropped; FORBIDDEN_KEYS are rejected at every object level.
 */
export function validateGenome(raw) {
  if (!isPlainObject(raw)) fail(ERROR_CODES.MESSAGE_INVALID, 'genome must be an object');
  assertNoForbiddenKeys(raw, 'genome');
  if (!Array.isArray(raw.nodes) || raw.nodes.length > MAX_GENOME_NODES) {
    fail(ERROR_CODES.MESSAGE_INVALID, 'genome nodes out of bounds');
  }
  if (!Array.isArray(raw.genes) || raw.genes.length > MAX_GENOME_GENES) {
    fail(ERROR_CODES.MESSAGE_INVALID, 'genome genes out of bounds');
  }

  const nodes = raw.nodes.map((n) => {
    if (!isPlainObject(n)) fail(ERROR_CODES.MESSAGE_INVALID, 'genome node must be an object');
    assertNoForbiddenKeys(n, 'node');
    if (typeof n.activation !== 'string' || !ACTIVATIONS.has(n.activation)) {
      fail(ERROR_CODES.MESSAGE_INVALID, 'bad node activation');
    }
    if (typeof n.type !== 'string' || !NODE_TYPES.has(n.type)) {
      fail(ERROR_CODES.MESSAGE_INVALID, 'bad node type');
    }
    return {
      id: reqInt(n.id, 0, 1e6, 'node.id'),
      type: sanitizeText(n.type, 40),
      activation: sanitizeText(n.activation, 40),
      bias: reqNumber(n.bias, -1e6, 1e6, 'node.bias'),
      depth: reqNumber(n.depth, -1e6, 1e6, 'node.depth'),
      frozen: n.frozen === undefined ? false : reqBool(n.frozen, 'node.frozen')
    };
  });

  const genes = raw.genes.map((g) => {
    if (!isPlainObject(g)) fail(ERROR_CODES.MESSAGE_INVALID, 'genome gene must be an object');
    assertNoForbiddenKeys(g, 'gene');
    return {
      innovation: reqInt(g.innovation, -1, 1e9, 'gene.innovation'),
      from: reqInt(g.from, 0, 1e6, 'gene.from'),
      to: reqInt(g.to, 0, 1e6, 'gene.to'),
      weight: reqNumber(g.weight, -1e6, 1e6, 'gene.weight'),
      enabled: reqBool(g.enabled, 'gene.enabled'),
      frozen: g.frozen === undefined ? false : reqBool(g.frozen, 'gene.frozen')
    };
  });

  return {
    nodes,
    genes,
    frozen: raw.frozen === undefined ? false : reqBool(raw.frozen, 'genome.frozen'),
    fitness: raw.fitness === undefined ? 0 : reqNumber(raw.fitness, -1e15, 1e15, 'genome.fitness'),
    adjustedFitness: raw.adjustedFitness === undefined ? 0
      : reqNumber(raw.adjustedFitness, -1e15, 1e15, 'genome.adjustedFitness'),
    speciesId: raw.speciesId === undefined ? -1 : reqInt(raw.speciesId, -1, 1e6, 'genome.speciesId')
  };
}

/** Validate a DONE.genomes map keyed only by the 19 known craftable ids. */
function validateGenomeMap(raw) {
  if (!isPlainObject(raw)) fail(ERROR_CODES.MESSAGE_INVALID, 'genomes must be an object');
  assertNoForbiddenKeys(raw, 'genomes');
  const keys = Object.keys(raw);
  if (keys.length > KNOWN_CRAFTABLE_IDS.length) {
    fail(ERROR_CODES.MESSAGE_INVALID, 'genomes key cap exceeded');
  }
  const out = Object.create(null);
  for (const key of keys) {
    if (!CRAFTABLE_IDS.has(key)) fail(ERROR_CODES.MESSAGE_INVALID, 'unknown genomes id: ' + key);
    out[key] = validateGenome(raw[key]);
  }
  return out;
}

/**
 * Validate + sanitize a worker message. Returns a sanitized plain-object copy
 * (strings passed through sanitizeText; perItem is a null-prototype map).
 * Throws on unknown types, prototype-pollution keys, non-finite / negative /
 * oversized numerics. This is the single throw-on-invalid API for the UI.
 */
export function validateWorkerMessage(msg) {
  if (!isPlainObject(msg)) fail(ERROR_CODES.MESSAGE_INVALID, 'message must be an object');
  if (typeof msg.type !== 'string' || !WORKER_TYPES.has(msg.type)) {
    fail(ERROR_CODES.MESSAGE_INVALID, 'unknown message type');
  }
  const rules = NUM_RULES[msg.type] || {};
  const out = {};
  for (const key of Object.keys(msg)) {
    if (key === 'type') continue;
    if (FORBIDDEN_KEYS.has(key)) fail(ERROR_CODES.MESSAGE_INVALID, 'forbidden key: ' + key);
    const v = msg[key];
    if (key === 'perItem' && msg.type === 'GENOME_RESULT') {
      out.perItem = validatePerItem(v);
    } else if (key === 'genome' && msg.type === 'BEST') {
      out.genome = validateGenome(v);
    } else if (key === 'genomes' && msg.type === 'DONE') {
      out.genomes = validateGenomeMap(v);
    } else if (typeof v === 'number') {
      const range = rules[key] || [0, MAX_NUM];
      if (!Number.isFinite(v) || v < range[0] || v > range[1]) {
        fail(ERROR_CODES.MESSAGE_INVALID, 'numeric out of range for ' + key);
      }
      out[key] = v;
    } else if (typeof v === 'string') {
      if (v.length > 120) fail(ERROR_CODES.MESSAGE_INVALID, 'oversized string for ' + key);
      out[key] = sanitizeText(v, 120);
    } else if (typeof v === 'boolean') {
      out[key] = v;
    } else if (Array.isArray(v)) {
      if (v.length > 5000) fail(ERROR_CODES.MESSAGE_INVALID, 'array too large for ' + key);
      out[key] = v.map((x) => (typeof x === 'number' ? x : sanitizeText(x, 120)));
    } else if (v === null) {
      out[key] = null;
    } else {
      fail(ERROR_CODES.MESSAGE_INVALID, 'unsupported value for ' + key);
    }
  }
  if (msg.type === 'GENOME_RESULT' && !out.perItem) {
    fail(ERROR_CODES.MESSAGE_INVALID, 'GENOME_RESULT requires perItem');
  }
  out.type = msg.type;
  return out;
}

const CONTROL_RULES = {
  speed: (v) => (v === '1x' || v === '10x' || v === 'fast'),
  seed: (v) => isBoundedInt(v, 0, 4294967295),
  preset: (v) => (v === 'reuse' || v === 'empty'),
  'pop-size': (v) => isBoundedInt(v, 1, 500),
  generations: (v) => isBoundedInt(v, 1, 10000)
};

/** Validate a single UI control value. Returns the value or throws. */
export function validateControl(name, value) {
  const rule = CONTROL_RULES[name];
  if (!rule) fail(ERROR_CODES.CONTROL_INVALID, 'unknown control: ' + name);
  if (!rule(value)) fail(ERROR_CODES.CONTROL_INVALID, 'invalid value for ' + name);
  return value;
}
