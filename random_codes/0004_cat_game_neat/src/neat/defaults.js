/**
 * defaults.js — greedy policy for items whose network has not been evolved yet.
 *
 * The production goal for an item is `required[item]` (exact reverse-topo
 * demand supplied by the controller). Each call asks for the remaining units,
 * capped at MAX_BATCH; coin/material clipping stays the engine's job.
 */
import { MAX_BATCH } from './inputs.js';

export { MAX_BATCH };

export function greedyBatch(itemId, context = {}, required = {}) {
  const reqMap = (required && typeof required === 'object') ? required : {};
  const goalRaw = Number(reqMap[itemId]);
  const goal = Number.isFinite(goalRaw) ? Math.max(0, Math.ceil(goalRaw)) : 0;
  if (goal === 0) return 0;

  const ctx = (context && typeof context === 'object') ? context : {};
  let made = ctx.produced;
  if (made === undefined || made === null) {
    const stash = (ctx.stash && typeof ctx.stash === 'object') ? ctx.stash : {};
    made = stash[itemId];
  }
  const madeNum = Number(made);
  const produced = Number.isFinite(madeNum) ? Math.max(0, madeNum) : 0;
  const remaining = Math.max(0, goal - produced);
  if (remaining === 0) return 0;

  return Math.min(MAX_BATCH, Math.ceil(remaining));
}

export default greedyBatch;
