#!/usr/bin/env node
/**
 * best-schedule.mjs — generate the app's "Best schedule" CSV
 * (`.opencode/plans/best-schedule-csv.md`, cycle-1 APPROVED; Phase 2: `reuse` preset).
 *
 * Pivot of `result().schedule` of the trained display run by tick:
 *   header `tick,` + the 19 craftable ids in topo order
 *   row t  = [t, batches produced at tick t for each craftable id, 0 when idle]
 *
 * App-faithful reproduction (src/main.js:514-573, src/neat/controller.js):
 *   - runConfig   = app load path: the app's own loadConfig(presetId) (src/config.js:158);
 *                   seed 1, pop 20, 100 gens, wallMs (default 180000; retry knob BEST_SCHEDULE_WALL_MS, capped 900000) — plan step 2
 *   - initial state = the saved `data/initial-state.json` when present — the app's own
 *                   initial-state shape `{targets, initialCoins, initialStash}` ("retrievable
 *                   and run simulation on") — validated with `validateInitialState` (src/validate.js:201);
 *                   when absent the app preset via loadConfig(PRESET_ID); a present-but-invalid
 *                   file throws loudly
 *   - training    = trainAll (controller.js:294) — plan step 3
 *   - bundle      = createDisplayPolicyBundle (controller.js:238) — plan step 4
 *   - caps        = planBatchCaps + applyBatchCaps — plan step 5
 *   - display run = createEngine + createSeededRng(1), step to done — plan step 6
 *
 * Run: node tools/best-schedule.mjs [reuse|empty]
 *       (from random_codes/0004_cat_game_neat; default: reuse)
 * Output: `data/best-schedule.csv`.
 *
 * Inline verification (structure / replay / determinism) runs before the
 * summary line; exit 0 when feasible, exit 1 when infeasible.
 */

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';

import { deriveConstants, loadConfig } from '../src/config.js';
import { validateInitialState } from '../src/validate.js';
import { createEngine } from '../src/sim/engine.js';
import { expandDemand, expandManufactureRequired } from '../src/sim/demand.js';
import { CRAFTABLE_IDS } from '../src/sim/items.js';
import { createNeatAdapter } from '../src/neat/adapter.js';
import { createDisplayPolicyBundle, trainAll } from '../src/neat/controller.js';
import { applyBatchCaps } from '../src/neat/policy-caps.js';
import { planBatchCaps } from '../src/neat/batch-planner.js';
import { createSeededRng } from '../src/neat/inhouse-neat.js';

const C = deriveConstants();
const MAX_TICKS = C.MAX_TICKS; // 8064
const MAX_BATCH = C.MAX_BATCH; // 20
const SEED = 1;
/** Preset: process.argv[2] in {reuse, empty}; default `reuse` (Phase 2). */
const PRESET_ID = (() => {
  const raw = process.argv[2];
  if (raw === undefined || raw === null || raw === '') return 'reuse';
  if (raw !== 'reuse' && raw !== 'empty') {
    console.error('usage: node tools/best-schedule.mjs [reuse|empty]  (default: reuse)');
    process.exit(2);
  }
  return raw;
})();

/** Training wall budget: 180000 ms default; BEST_SCHEDULE_WALL_MS overrides (retry path, max 900000). */
const MAX_WALL_MS = (() => {
  const raw = Number.parseInt(process.env.BEST_SCHEDULE_WALL_MS ?? '', 10);
  return Number.isInteger(raw) && raw >= 1000 && raw <= 900000 ? raw : 180000;
})();

const CSV_URL = new URL('../data/best-schedule.csv', import.meta.url);
const SAVED_STATE_URL = new URL('../data/initial-state.json', import.meta.url);

function assert(cond, msg) {
  if (!cond) {
    console.error('ASSERT FAIL: ' + msg);
    process.exit(1);
  }
}

/** Small deep-equal for arrays of plain numbers (rows / schedules). */
function deepEqual(a, b) {
  if (a === b) return true;
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && a.every((x, i) => deepEqual(x, b[i]));
  }
  if (a && b && typeof a === 'object' && typeof b === 'object') {
    const ka = Object.keys(a);
    const kb = Object.keys(b);
    return ka.length === kb.length
      && ka.every((k) => Object.prototype.hasOwnProperty.call(b, k) && deepEqual(a[k], b[k]));
  }
  return false;
}

/**
 * Read the saved initial state (`data/initial-state.json` — the app's own
 * `{targets, initialCoins, initialStash}` shape). Returns null when the file
 * is absent (ENOENT — nothing saved yet); throws loudly on malformed JSON, a
 * missing required field or a `validateInitialState` failure — the saved
 * state is the simulation input, so a bad one must not silently fall back.
 * Maps are null-prototype, matching `applyOverrides` (src/config.js:191-204)
 * and `validateInitialState` (src/validate.js:201-215).
 */
function readSavedInitialState() {
  let parsed;
  try {
    parsed = JSON.parse(readFileSync(SAVED_STATE_URL, 'utf8'));
  } catch (err) {
    if (err && err.code === 'ENOENT') return null; // no saved state yet
    throw err; // malformed JSON -> loud failure
  }
  if (!parsed || typeof parsed !== 'object') {
    throw new Error('data/initial-state.json: saved state must be a JSON object');
  }
  for (const key of ['targets', 'initialCoins', 'initialStash']) {
    if (parsed[key] === undefined) {
      throw new Error(`data/initial-state.json: missing required field "${key}"`); // loud-fail (red team L3)
    }
  }
  const overrides = {
    targets: Object.assign(Object.create(null), parsed.targets),      // null-proto (src/validate.js:95-97)
    initialCoins: parsed.initialCoins,
    initialStash: Object.assign(Object.create(null), parsed.initialStash)
  };
  validateInitialState(overrides); // throws CONFIG_INVALID on any bad field (src/validate.js:201-215)
  return overrides;
}

/** App load path (plan step 1) + the saved initial state when present. */
async function buildRunConfig() {
  const base = await loadConfig(PRESET_ID);
  const saved = readSavedInitialState();
  return {
    ...base,
    ...saved,                              // targets/initialCoins/initialStash from the saved state
    populationSize: 20,
    adapterConfig: { populationSize: 20 },
    generations: 100,
    maxGenerations: 100,
    maxWallMs: MAX_WALL_MS
  };
}

async function main() {
  const runConfig = await buildRunConfig();
  const saved = readSavedInitialState();

  console.error(`[best-schedule] preset ${PRESET_ID}: training seed 1, pop 20, 100 gens, ${MAX_WALL_MS} ms wall...`
    + (saved ? ' + initial state from data/initial-state.json (validated)' : ''));
  const t0 = Date.now();
  const { genomes } = await trainAll({
    adapterFactory: createNeatAdapter,
    engineFactory: createEngine,
    config: runConfig,
    seed: SEED,
    maxGenerations: 100,
    maxWallMs: MAX_WALL_MS,
    quiet: true
  });
  console.error(`[best-schedule] training done in ${Math.round((Date.now() - t0) / 1000)}s`);

  const required = expandDemand(runConfig.items, runConfig.targets, runConfig.initialStash, runConfig.starterMode).required;
  const manufactureRequired = expandManufactureRequired(runConfig.items, runConfig.targets, runConfig.initialStash, runConfig.starterMode).manufactureRequired;

  const bundle = createDisplayPolicyBundle({
    genomes,
    config: runConfig,
    required,
    manufactureRequired,
    adapterFactory: createNeatAdapter,
    seed: SEED,
    maxTicks: MAX_TICKS
  });

  const caps = planBatchCaps(runConfig.items, manufactureRequired, MAX_TICKS, MAX_BATCH).caps;

  function runDisplay() {
    const policyForItem = applyBatchCaps(bundle.policyForItem, caps);
    const engine = createEngine({ config: runConfig, policyForItem, rng: createSeededRng(SEED) });
    while (!engine.done && engine.tick < 8100) engine.step();
    return engine.result();
  }

  const result = runDisplay();

  const ids = CRAFTABLE_IDS;
  const end = Number.isFinite(result.completionTick)
    ? result.completionTick
    : Math.max(0, result.elapsedTicks - 1);

  /** Wide pivot #1 (used to build the CSV rows). */
  function pivot(schedule) {
    const byTick = new Map();
    for (const e of schedule) {
      if (!Number.isInteger(e.tick)) continue;
      let row = byTick.get(e.tick);
      if (!row) { row = Object.create(null); byTick.set(e.tick, row); }
      row[e.item] = (row[e.item] || 0) + e.batch;
    }
    const rows = [];
    for (let t = 0; t <= end; t++) {
      const cells = byTick.get(t) || {};
      rows.push([t, ...ids.map((id) => Number(cells[id] || 0))]);
    }
    return rows;
  }

  /** Wide pivot #2 — the independent replay implementation used by the check below. */
  function replayTable(schedule) {
    const acc = Object.create(null);
    for (const e of schedule) {
      if (!Number.isInteger(e.tick)) continue;
      if (!acc[e.tick]) acc[e.tick] = Object.create(null);
      acc[e.tick][e.item] = (acc[e.tick][e.item] || 0) + e.batch;
    }
    const out = [];
    for (let t = 0; t <= end; t++) {
      const src = acc[t] || {};
      out.push([t, ...ids.map((id) => Number(src[id] || 0))]);
    }
    return out;
  }

  const rows = pivot(result.schedule);
  const header = ['tick', ...ids];

  // ---- verification 1: CSV structure ----
  assert(rows.length === end + 1, `rows span 0..end (${rows.length} rows, end=${end})`);
  for (let i = 0; i < rows.length; i++) {
    assert(Array.isArray(rows[i]) && rows[i].length === header.length, `row ${i}: ${header.length} fields`);
    assert(rows[i][0] === i, `row ${i}: tick is ${rows[i][0]}`);
    for (let j = 1; j < rows[i].length; j++) {
      assert(Number.isInteger(rows[i][j]) && rows[i][j] >= 0,
        `row ${i} col ${j}: non-negative integer (got ${rows[i][j]})`);
    }
  }

  // ---- write CSV: header + rows (no trailing newline; every row has 20 fields) ----
  const csv = [header.join(','), ...rows.map((r) => r.join(','))].join('\n');
  mkdirSync(new URL('.', CSV_URL), { recursive: true }); // red team L4: data/ may not exist on a fresh clone
  writeFileSync(CSV_URL, csv);

  // ---- verification 2: replay — recompute the wide table from result().schedule ----
  const replayed = replayTable(result.schedule);
  assert(deepEqual(replayed, rows), 'replay: wide table from result().schedule deep-equals CSV rows');

  // Per-tick accounting: Σ cells === Σ e.batch; at most one event per (tick, id).
  const batchSums = new Map();
  const eventCounts = new Map();
  for (const e of result.schedule) {
    if (!Number.isInteger(e.tick)) continue;
    batchSums.set(e.tick, (batchSums.get(e.tick) || 0) + e.batch);
    eventCounts.set(e.tick, (eventCounts.get(e.tick) || 0) + 1);
  }
  for (let i = 0; i < rows.length; i++) {
    const cellSum = rows[i].slice(1).reduce((a, b) => a + b, 0);
    assert(cellSum === (batchSums.get(i) || 0),
      `row ${i}: cell sum ${cellSum} === Σ batch at tick ${i} (schedule events at tick: ${eventCounts.get(i) || 0})`);
    assert((eventCounts.get(i) || 0) <= ids.length,
      `row ${i}: at most one accepted event per craftable (events at tick: ${eventCounts.get(i) || 0})`);
  }

  // ---- verification 3: determinism — re-run the display engine, same bundle ----
  const result2 = runDisplay();
  assert(JSON.stringify(result2.schedule) === JSON.stringify(result.schedule),
    'determinism: second display run produced a byte-identical schedule');

  const summary = {
    preset: PRESET_ID,
    feasible: result.feasible,
    completionTick: result.completionTick,
    elapsedTicks: result.elapsedTicks,
    totalSpent: result.totalSpent,
    scheduleRows: rows.length,
    events: result.schedule.length
  };
  console.log(JSON.stringify(summary));
  if (!result.feasible) {
    console.error('[best-schedule] INFEASIBLE; retry once with BEST_SCHEDULE_WALL_MS (max 900000) if needed');
    process.exitCode = 1;
  }
}

main().catch((err) => {
  console.error('FAIL: ' + ((err && err.stack) || err));
  process.exit(1);
});
