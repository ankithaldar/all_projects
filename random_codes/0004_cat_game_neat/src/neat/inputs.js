/**
 * inputs.js — per-item observation builder and network-output -> batch decoder.
 *
 * The observation is a fixed-length vector of normalized numbers (all in [0,1])
 * so networks trained with different scales stay comparable. Field order is
 * part of the contract (OBS_FIELDS) and must not change without retraining.
 */

/** Maximum units craftable in one batch — single literal owned by config.js. */
import { MAX_BATCH } from '../config.js';

export { MAX_BATCH };
/** Number of network inputs consumed by every per-item policy network. */
export const OBSERVATION_SIZE = 5;
/** Observation field order, indexed by position in the vector. */
export const OBS_FIELDS = Object.freeze([
  'remainingDemand', 'coins', 'ticksRemaining', 'materialAvailability', 'slotActive'
]);

const COIN_SCALE = 1e7;
const DEFAULT_MAX_TICKS = 8064;

function clamp01(x) {
  if (!Number.isFinite(x)) return 0;
  return x < 0 ? 0 : (x > 1 ? 1 : x);
}

/**
 * Build the 5-element observation vector for one item.
 * Context fields (all optional, defensive defaults):
 *   itemId, required, produced | stash[itemId], coins, tick, maxTicks,
 *   materialsAvailable, slotActive.
 */
export function makeObservation(context = {}) {
  const ctx = (context && typeof context === 'object') ? context : {};

  const required = Number(ctx.required);
  const req = Number.isFinite(required) && required > 0 ? required : 0;

  let producedRaw = ctx.produced;
  if (producedRaw === undefined || producedRaw === null) {
    const stash = (ctx.stash && typeof ctx.stash === 'object') ? ctx.stash : {};
    producedRaw = stash[ctx.itemId];
  }
  const producedNum = Number(producedRaw);
  const produced = Number.isFinite(producedNum) ? Math.max(0, producedNum) : 0;
  const remaining = Math.max(0, req - produced);
  const remainingDemand = req > 0 ? Math.min(1, remaining / req) : 0;

  const coins = clamp01(Number(ctx.coins) / COIN_SCALE);

  const maxTicksNum = Number(ctx.maxTicks);
  const maxTicks = Number.isFinite(maxTicksNum) && maxTicksNum > 0 ? maxTicksNum : DEFAULT_MAX_TICKS;
  const tickNum = Number(ctx.tick);
  const tick = Number.isFinite(tickNum) ? Math.max(0, tickNum) : 0;
  const ticksRemaining = clamp01(Math.max(0, maxTicks - tick) / maxTicks);

  const materialsNum = Number(ctx.materialsAvailable);
  const available = Number.isFinite(materialsNum) ? Math.max(0, materialsNum) : 0;
  const materialAvailability = clamp01(available / Math.max(1, req || MAX_BATCH));

  const slotActive = ctx.slotActive === true || ctx.slotActive === 1 ? 1 : 0;

  return [remainingDemand, coins, ticksRemaining, materialAvailability, slotActive];
}

/**
 * Decode a network output (scalar or single-element array) into an integer
 * batch size in [1, maxBatch]. Outputs are treated as normalized [0,1]
 * activations by default (LOGISTIC output nodes); values are clamped.
 */
export function decodeBatch(output, maxBatch = MAX_BATCH) {
  let cap = Number(maxBatch);
  if (!Number.isFinite(cap) || cap < 1) cap = MAX_BATCH;
  cap = Math.floor(cap);

  const raw = Array.isArray(output) ? output[0] : output;
  let n = Number(raw);
  if (!Number.isFinite(n)) n = 0;
  if (n < 0) n = 0;
  if (n > 1) n = 1;

  const batch = Math.round(n * (cap - 1)) + 1;
  return Math.min(cap, Math.max(1, batch));
}

export default makeObservation;
