/**
 * cost.js — batch cost and max-affordable batch.
 *
 * Verified closed form (plan "Verified Facts"):
 *   unitCost(init, n)  = init * (1 + 0.5 * (n - 1))      for n >= 1
 *   totalCost(init, b) = init * b * (1 + 0.25 * (b - 1)) = Σ_{n=1..b} unitCost(init, n)
 *
 * MAX_BATCH literal lives in config.js (single owner); re-exported here for
 * existing consumers. No DOM, no globals beyond Math. Pure + deterministic.
 */
import { MAX_BATCH } from '../config.js';

export { MAX_BATCH };

/** Non-finite / negative init costs are inert (F6). Zero remains "free". */
function safeInit(initCost) {
  return Number.isFinite(initCost) && initCost >= 0 ? initCost : 0;
}

/** Price of the n-th unit in a batch (n is 1-indexed). */
export function unitCost(initCost, n) {
  if (!Number.isFinite(n) || n <= 0) return 0;
  return safeInit(initCost) * (1 + 0.5 * (n - 1));
}

/** Total price of a batch of `batch` units (b >= 0). */
export function totalCost(initCost, batch) {
  if (!Number.isFinite(batch) || batch <= 0) return 0;
  return safeInit(initCost) * batch * (1 + 0.25 * (batch - 1));
}

/**
 * Largest b in [0, MAX_BATCH] with totalCost(initCost, b) <= coins.
 * Closed-form quadratic solve with ceil-correction + downward adjust to
 * guarantee the inequality even under floating point rounding.
 */
export function maxAffordableBatch(initCost, coins) {
  if (!Number.isFinite(initCost) || initCost < 0) return 0; // F6
  if (!Number.isFinite(coins) || coins < 0) return 0;        // F6
  if (initCost === 0) return MAX_BATCH;                       // free items

  // init*(0.25 b^2 + 0.75 b) <= coins  =>  b^2 + 3b - 16*coins/init <= 0
  // => b <= (-3 + sqrt(9 + 16*coins/init)) / 2
  const ratio = (16 * coins) / initCost;
  let b = Math.floor((-3 + Math.sqrt(9 + ratio)) / 2);
  if (!Number.isFinite(b)) b = 0;
  if (b > MAX_BATCH) b = MAX_BATCH;
  if (b < 0) b = 0;
  while (b > 0 && totalCost(initCost, b) > coins) b -= 1;
  while (b < MAX_BATCH && totalCost(initCost, b + 1) <= coins) b += 1;
  return b;
}
