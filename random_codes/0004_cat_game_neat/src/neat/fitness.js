/**
 * fitness.js — lexicographic, coins-first / time-second fitness with a hard
 * feasibility gate (plan "Fitness" section).
 *
 *   feasible:   fitness = -(totalSpent * BIG + completionTick)
 *   infeasible: fitness = -1e12 + progressBonus
 *
 * Higher fitness is better. `totalSpent` is the engine's explicit accumulator;
 * this module NEVER computes spend as initialCoins - finalCoins.
 *
 * `fitness()` is the SHIPPED score and stays the value written to telemetry.
 * `trainingFitness()` is a separate, bounded shaping signal used ONLY for NEAT
 * parent selection while a rollout is infeasible (plan decision 4 / finding 2).
 * Once a rollout is feasible it returns the shipped score immediately.
 */

/** > max tick horizon (8064) so spend strictly dominates time. */
export const BIG = 100000;
/** Floor for any infeasible rollout; any feasible rollout outranks it. */
export const INFEASIBLE_BASE = -1e12;
/** Hard finite clamp for every training-fitness value (security gate). */
export const FITNESS_CLAMP = 1e15;
const DEFAULT_MAX_TICKS = 8064;

const hasOwn = (o, k) => Object.prototype.hasOwnProperty.call(o, k);
const isObj = (v) => !!v && typeof v === 'object' && !Array.isArray(v);

function clamp01(x) {
  const n = Number(x);
  if (!Number.isFinite(n)) return 0;
  return n < 0 ? 0 : (n > 1 ? 1 : n);
}

/**
 * Progress achieved toward targets while infeasible, in [0, ...] but bounded
 * by the number of positive targets: sum of min(stash, target)/target.
 * Reads `result.targets` (or `result.required`) and `result.stash`
 * (or `result.produced`). Falls back to 0 when no target map is supplied.
 */
export function progressBonus(result) {
  if (!isObj(result)) return 0;
  const targets = isObj(result.targets) ? result.targets
    : isObj(result.required) ? result.required : null;
  if (!targets) return 0;
  const stash = isObj(result.stash) ? result.stash
    : isObj(result.produced) ? result.produced : {};
  let bonus = 0;
  for (const id of Object.keys(targets)) {
    const target = Number(targets[id]);
    if (!Number.isFinite(target) || target <= 0) continue;
    const haveRaw = Number(stash[id]);
    const have = Number.isFinite(haveRaw) ? Math.max(0, haveRaw) : 0;
    bonus += Math.min(have, target) / target;
  }
  return bonus;
}

/**
 * Enrich a DOM-free engine result with the data the shaped training fitness
 * needs: the targets, the manufactureRequired basis (`mfr`), `maxTicks`, and a
 * defensive `stash`/`produced` view. Pure and deterministic; the original
 * result object is never mutated.
 */
export function enrichResult(result, targets, mfr, maxTicks = DEFAULT_MAX_TICKS) {
  const base = isObj(result) ? result : {};
  const produced = isObj(base.produced) ? base.produced : {};
  const tickRaw = Number(maxTicks);
  const resolvedMaxTicks = Number.isFinite(tickRaw) && tickRaw > 0 ? tickRaw : DEFAULT_MAX_TICKS;
  return {
    ...base,
    targets: isObj(targets) ? targets : (isObj(base.targets) ? base.targets : {}),
    mfr: isObj(mfr) ? mfr : (isObj(base.mfr) ? base.mfr : {}),
    maxTicks: resolvedMaxTicks,
    stash: isObj(base.stash) ? base.stash : produced,
    produced
  };
}

/**
 * fitness(result) -> number. `result` is the DOM-free engine result enriched
 * (by the controller) with `targets`/`stash` for the infeasible progress bonus.
 * This is the SHIPPED score and is what telemetry reports.
 */
export function fitness(result) {
  if (!isObj(result)) return INFEASIBLE_BASE;
  if (result.feasible !== true) {
    return INFEASIBLE_BASE + progressBonus(result);
  }
  const spentRaw = Number(result.totalSpent);
  const tickRaw = Number(result.completionTick);
  const spent = Number.isFinite(spentRaw) ? spentRaw : 0;
  const tick = Number.isFinite(tickRaw) ? tickRaw : 0;
  return -(spent * BIG + tick);
}

/**
 * trainingFitness(result, phase, mfr, maxTicks) -> number.
 *
 * Selection signal only. `phase` is 'feasible' when the rollout met all
 * targets, otherwise 'pre'. Any feasible rollout (phase or result.feasible)
 * immediately uses the shipped lexicographic score. While infeasible, a
 * bounded shaped score rewards manufactured units, target progress and time:
 *
 *   INFEASIBLE_BASE + 1000*manufacturedProgress
 *                   + 100*targetProgress
 *                   + 10*timeProgress
 *
 * Each progress term is clamped to [0,1]; divisions are guarded (`qty > 0`),
 * the total is Number.isFinite-checked and finally clamped to
 * [-FITNESS_CLAMP, FITNESS_CLAMP]. The maximum shaped value is INFEASIBLE_BASE
 * + 1110, which is strictly below the smallest feasible score (the worst
 * feasible spend is bounded by the total wallet, far below 1e12).
 */
export function trainingFitness(result, phase, mfr, maxTicks = DEFAULT_MAX_TICKS) {
  const enriched = enrichResult(result, isObj(result) ? result.targets : null, mfr, maxTicks);
  if (phase === 'feasible' || enriched.feasible === true) return fitness(enriched);

  const produced = enriched.produced;
  const mfrMap = enriched.mfr;
  const targets = enriched.targets;
  const stash = enriched.stash;
  const horizon = enriched.maxTicks;

  // manufacturedProgress: mean of min(1, produced/mfr) over positive caps.
  let mSum = 0;
  let mCount = 0;
  for (const id of Object.keys(mfrMap)) {
    if (!hasOwn(mfrMap, id)) continue;
    const qty = Number(mfrMap[id]);
    if (!Number.isFinite(qty) || qty <= 0) continue;
    const haveRaw = Number(hasOwn(produced, id) ? produced[id] : (stash[id]));
    const have = Number.isFinite(haveRaw) ? Math.max(0, haveRaw) : 0;
    mSum += Math.min(1, have / qty); // qty > 0 here: division guarded
    mCount += 1;
  }
  const manufacturedProgress = mCount > 0 ? clamp01(mSum / mCount) : 0;

  // targetProgress: mean of min(1, stash/target) over positive targets.
  let tSum = 0;
  let tCount = 0;
  for (const id of Object.keys(targets)) {
    if (!hasOwn(targets, id)) continue;
    const qty = Number(targets[id]);
    if (!Number.isFinite(qty) || qty <= 0) continue;
    const haveRaw = Number(stash[id]);
    const have = Number.isFinite(haveRaw) ? Math.max(0, haveRaw) : 0;
    tSum += Math.min(1, have / qty); // qty > 0 here: division guarded
    tCount += 1;
  }
  const targetProgress = tCount > 0 ? clamp01(tSum / tCount) : 0;

  // timeProgress: fraction of the (fixed 8064) horizon still available.
  const elapsedRaw = Number(isObj(result) ? result.elapsedTicks : 0);
  let elapsed = Number.isFinite(elapsedRaw) ? Math.max(0, elapsedRaw) : 0;
  if (!Number.isFinite(elapsed) || elapsed === 0) {
    const tickRaw = Number(isObj(result) ? result.completionTick : 0);
    if (Number.isFinite(tickRaw) && tickRaw > 0) elapsed = tickRaw;
  }
  const timeProgress = clamp01((horizon - elapsed) / horizon); // horizon > 0

  let score = INFEASIBLE_BASE
    + 1000 * manufacturedProgress
    + 100 * targetProgress
    + 10 * timeProgress;
  if (!Number.isFinite(score)) score = INFEASIBLE_BASE;
  return Math.min(FITNESS_CLAMP, Math.max(-FITNESS_CLAMP, score));
}

export default fitness;
