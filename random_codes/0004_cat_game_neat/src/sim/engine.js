/**
 * engine.js — deterministic, DOM-free tick engine.
 *
 * Tick order (plan "Simulation Contract"):
 *   1. +210 coins (COIN_INC)
 *   2. each free craftable slot (topo order): ask policy, clip to
 *      min(materialsLimit, coinAffordableLimit, MAX_BATCH, mfr-remaining),
 *      then quantizeBatch() AFTER all clips. >0 => spend + start batch
 *      (DECISION_ACCEPTED / COIN_DEC / MATERIAL_DEC); 0 =>
 *      DECISION_REJECTED (REQUIRED_MET or the frozen BELOW_QUANTUM literal)
 *      + starvation flags only for genuine material/coin starvation.
 *   3. advance active timers; busy tick counted against that item.
 *   4. completed slots add batch to stash, reset (STASH_UPDATE / BATCH_DONE).
 *   5. evaluate targets against CURRENT stash (TARGET_UNMET on consumption).
 *   6. snapshot is emitted after 3-4 (post-decrement countdown).
 *
 * Every coin spend and every busy tick is attributed to the item whose slot
 * performed it, so sum(perItem.coinsSpent) === totalSpent and
 * sum(perItem.timeMinutes) === total manufacturing time, always.
 */
import { deriveConstants, craftTicksForMinutes } from '../config.js';
import { totalCost, maxAffordableBatch } from './cost.js';
import { topoOrder, expandDemand, expandManufactureRequired } from './demand.js';
import { makeSnapshot } from './state.js';
// Pure, read-only batch helpers (owned by the data-scientist). The engine never
// mutates these modules or their returned maps; there is no config.caps field.
import { computeBatchQuantums, effectiveCap, quantizeBatch } from '../neat/batch-quantizer.js';
import { computeAlignment } from '../neat/time-aligner.js';
import { planBatchCaps } from '../neat/batch-planner.js';

const { MAX_TICKS, TICK_MINUTES, COINS_PER_TICK, MAX_BATCH } = deriveConstants();
const BASE_STASH = 1000000; // finite sentinel for "unlimited" base items
const EMPTY_EVENTS = Object.freeze([]);
// Frozen cap-rejection reason literals — never interpolated (exact wire contract).
const REQUIRED_MET = 'REQUIRED_MET';
const BELOW_QUANTUM = 'BELOW_QUANTUM';

/** Finite positive integer, or `fallback` for garbage/absent values. */
function positiveInt(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? Math.max(1, Math.floor(n)) : fallback;
}
const hasOwn = (o, k) => Object.prototype.hasOwnProperty.call(o, k);

/**
 * createEngine({config, policyForItem, quiet}) -> {step(), snapshot(), result()}
 *
 * `quiet` (default false): fitness rollouts set it true to skip per-tick event
 * object allocation. Simulation state, result() and perItem are unaffected.
 */
export function createEngine({ config, policyForItem, quiet = false }) {
  const items = config.items;
  const byId = new Map(items.map((i) => [i.id, i]));
  const craftIds = topoOrder(items);
  const targets = config.targets || {};
  const mode = config.starterMode || 'PREDELIVERED_RESERVED';
  const starterStash = config.initialStash || {};
  const starterCountsTowardTarget = mode !== 'FREE_INGREDIENTS';
  const starterConsumable = mode !== 'PREDELIVERED_RESERVED';

  const { required: baseRequired } = expandDemand(items, targets, starterStash, mode);
  const { manufactureRequired } = expandManufactureRequired(items, targets, starterStash, mode);

  // ---- Batch metadata (derived once, read-only, from validated items + mfr) --
  // The engine owns no cap config: every quantum/alignment/cap is computed from
  // the same validated config.items and the mfr basis used by the hard cap.
  // All maps are null-prototype and keyed by craftable ids only.
  const quantums = computeBatchQuantums(items) || Object.create(null);
  const alignmentResult = computeAlignment(items, manufactureRequired, quantums) || {};
  const alignments = alignmentResult.alignments || Object.create(null);

  // The SAME offline planner the controller/main thread use, evaluated here
  // once from the engine's own mfr. This is where cap repair and any
  // ALIGNMENT_RELAXED decisions live; the engine never raises a cap itself.
  const planned = planBatchCaps(items, manufactureRequired, MAX_TICKS, MAX_BATCH);
  const plannerCaps = planned && planned.caps ? planned.caps : Object.create(null);
  const relaxedIds = new Set();
  for (const r of (planned && Array.isArray(planned.relaxed)) ? planned.relaxed : []) {
    if (r && typeof r.id === 'string') relaxedIds.add(r.id);
  }

  const quantStep = Object.create(null);   // per-item batch quantum (step)
  const batchCap = Object.create(null);    // effectiveCap(plannerCap, step, alignCap)
  const aligned = Object.create(null);     // true only for feasible k>1 alignments
  const blocked = Object.create(null);     // telemetry: sub-quantum rejection latched

  // Initialize per-item maps for every id referenced anywhere (defensive).
  const allIds = new Set(items.map((i) => i.id));
  for (const it of items) {
    if (it.ingredients) for (const ing of it.ingredients) allIds.add(ing.item);
  }
  const producedNet = {};      // produced - consumed (physical, consumable)
  const starterRemaining = {}; // unconsumed starter units
  const produced = {};
  const committed = {};        // in-flight units (one slot per craftable id)
  const consumed = {};
  const slots = {};
  const starvation = {};
  const perItem = {};
  for (const id of allIds) {
    producedNet[id] = 0;
    starterRemaining[id] = Math.max(0, Math.floor(Number(starterStash[id]) || 0));
    produced[id] = 0;
    committed[id] = 0;
    consumed[id] = 0;
  }
  for (const id of craftIds) {
    slots[id] = { remainingTicks: 0, batchSize: 0, state: 'IDLE' };
    starvation[id] = { material: false, coin: false };
    perItem[id] = { coinsSpent: 0, busyTicks: 0, batches: 0, units: 0, firstTick: -1, lastTick: -1 };

    const step = positiveInt(hasOwn(quantums, id) ? quantums[id] : undefined, 1);
    const align = alignments[id];
    const k = align ? positiveInt(align.k, 1) : 1;
    // A relaxed alignment is reported by the planner with alignCap=0 in its
    // caps; mirror that here so the quantum-multiple rule applies again.
    const isRelaxed = relaxedIds.has(id);
    const alignCap = (align && !isRelaxed) ? positiveInt(align.alignCap, 0) : 0;
    const isAligned = !isRelaxed && k > 1 && alignCap > 0;
    const plannerCap = hasOwn(plannerCaps, id)
      ? positiveInt(plannerCaps[id], MAX_BATCH) : MAX_BATCH;
    quantStep[id] = step;
    aligned[id] = isAligned;
    // Single cap authority: same effectiveCap the planner used, with the
    // planner's own (already repaired / relaxed) cap as the input.
    batchCap[id] = positiveInt(effectiveCap(plannerCap, step, alignCap), 1);
    blocked[id] = false;
  }

  // F6: initial coins must be finite + bounded (Infinity/NaN must not leak
  // into snapshot.coins or the fitness arithmetic).
  const rawCoins = Number(config.initialCoins);
  const initialCoins = Number.isFinite(rawCoins)
    ? Math.min(1e12, Math.max(0, Math.floor(rawCoins)))
    : 0;

  const st = {
    tick: -1, // step() increments to 0 before the first tick is processed
    coins: initialCoins,
    totalSpent: 0,
    producedNet, starterRemaining, produced, consumed, slots, starvation, perItem,
    status: {},
    schedule: [],
    completionTick: null,
    done: false
  };

  // Seed target status from current stash.
  for (const id of Object.keys(targets)) st.status[id] = counted(id) >= targets[id];

  function counted(id) {
    const net = producedNet[id] || 0;
    return starterCountsTowardTarget ? net + (starterRemaining[id] || 0) : net;
  }

  function availableFor(id) {
    const src = byId.get(id);
    if (src && src.type === 'base') return Infinity;
    const net = producedNet[id] || 0;
    return starterConsumable ? net + (starterRemaining[id] || 0) : net;
  }

  function emit(events, type, item, delta, reason) {
    if (quiet) return; // F1: no per-tick event objects during rollouts
    events.push({
      tick: st.tick,
      type,
      item: item === undefined ? null : item,
      delta: delta === undefined ? null : delta,
      reason: reason === undefined ? '' : String(reason).slice(0, 120)
    });
  }

  function consume(id, qty, events) {
    let need = qty;
    if (starterConsumable) {
      const fromStarter = Math.min(starterRemaining[id] || 0, need);
      starterRemaining[id] = (starterRemaining[id] || 0) - fromStarter;
      need -= fromStarter;
    }
    producedNet[id] = (producedNet[id] || 0) - need;
    consumed[id] = (consumed[id] || 0) + qty;

    if (Object.prototype.hasOwnProperty.call(targets, id)) {
      const before = st.status[id];
      const after = counted(id) >= targets[id];
      st.status[id] = after;
      if (before === true && after === false) emit(events, 'TARGET_UNMET', id, -qty, 'CONSUMED_BELOW_TARGET');
    }
  }

  function materialsLimitFor(item) {
    let limit = Infinity;
    for (const ing of item.ingredients) {
      const src = byId.get(ing.item);
      if (src && src.type === 'base') continue; // bases never contribute
      limit = Math.min(limit, Math.floor(availableFor(ing.item) / ing.quantity));
    }
    return limit;
  }

  function askPolicy(id, ctx) {
    let raw = 0;
    if (typeof policyForItem === 'function') raw = policyForItem(id, ctx);
    else if (policyForItem && typeof policyForItem[id] === 'function') raw = policyForItem[id](ctx);
    const n = Math.floor(Number(raw));
    return Number.isFinite(n) && n > 0 ? n : 0;
  }

  // ---- Lazy, read-only stash view for policyContext ------------------------
  // The previous implementation rebuilt a full stash object (one key per item)
  // for every item on every tick. That object is replaced by a single memoised
  // Proxy that computes each stash value on demand from the CURRENT engine
  // state and rejects every mutation attempt. It stays shape-compatible with
  // the eager plain object: property reads, `in`, Object.keys and object spread
  // all expose exactly the same item-id keys with the same scalar values.
  // No eval/Function is used.
  //
  // NOTE: policyContext.required and policyContext.remaining are intentionally
  // kept for backward compatibility (legacy UI/tooltips). The controller's
  // buildContext() does NOT read them - it uses the manufactureRequired basis
  // via ctx.produced / expandManufactureRequired. Do not delete these keys.
  let lazyStashView = null;
  const stashIds = items.map((it) => it.id); // fixed key set, same as the eager build

  function stashValue(id) {
    const item = byId.get(id);
    return item && item.type === 'base'
      ? BASE_STASH
      : (producedNet[id] || 0) + (starterRemaining[id] || 0);
  }

  function makeLazyStashView() {
    // Null-prototype target: no inherited keys (e.g. __proto__/constructor)
    // can leak into object spread, and there is no prototype-pollution path.
    const target = Object.create(null);
    const rejectMutation = () => { throw new TypeError('policyContext.stash is read-only'); };
    return new Proxy(target, {
      get(_t, prop) {
        if (typeof prop !== 'string') return undefined;
        return byId.has(prop) ? stashValue(prop) : undefined;
      },
      has(_t, prop) {
        return typeof prop === 'string' && byId.has(prop);
      },
      ownKeys() {
        return stashIds.slice();
      },
      getOwnPropertyDescriptor(_t, prop) {
        if (typeof prop !== 'string' || !byId.has(prop)) return undefined;
        // configurable:true satisfies the Proxy ownKeys/descriptor invariants
        // for an extensible target; enumerable:true keeps object spread exact.
        return { value: stashValue(prop), writable: false, enumerable: true, configurable: true };
      },
      set: rejectMutation,
      defineProperty: rejectMutation,
      deleteProperty: rejectMutation,
      setPrototypeOf: rejectMutation,
      preventExtensions: rejectMutation
    });
  }

  function stashView() {
    if (lazyStashView === null) lazyStashView = makeLazyStashView();
    return lazyStashView;
  }

  function policyContext(id) {
    return {
      tick: st.tick,
      itemId: id,
      coins: st.coins,
      totalSpent: st.totalSpent,
      stash: stashView(),
      slots: api.slots,
      starvation: api.starvation,
      blocked: api.blocked,
      status: api.status,
      recipe: byId.get(id),
      craftTicks: craftTicksForMinutes(byId.get(id).craftTimeMinutes),
      required: baseRequired[id] || 0,
      produced: produced[id] || 0,
      remaining: Math.max(0, (baseRequired[id] || 0) - (produced[id] || 0))
    };
  }

  function step() {
    if (st.done) return EMPTY_EVENTS;
    st.tick += 1;
    const events = quiet ? EMPTY_EVENTS : [];

    // 1. coin income
    st.coins += COINS_PER_TICK;
    emit(events, 'COIN_INC', null, COINS_PER_TICK, 'TICK_INCOME');

    // 2. decisions, topo order, free slots only
    for (const id of craftIds) {
      const slot = slots[id];
      const item = byId.get(id);
      if (slot.remainingTicks > 0) {
        starvation[id] = { material: false, coin: false }; // busy clears flags
        continue;
      }

      // Hard cap (evaluated FIRST, before want/material/coin): completed plus
      // in-flight units have already reached manufactureRequired -> reject with
      // the frozen REQUIRED_MET literal and emit no STARVED_* event.
      const remaining = Math.max(
        0, (manufactureRequired[id] || 0) - (produced[id] || 0) - (committed[id] || 0)
      );
      if (remaining === 0) {
        starvation[id] = { material: false, coin: false };
        emit(events, 'DECISION_REJECTED', id, 0, REQUIRED_MET);
        continue;
      }

      const materialsLimit = materialsLimitFor(item);
      const coinLimit = maxAffordableBatch(item.coinCost, st.coins);
      const want = askPolicy(id, policyContext(id));
      let n = Math.min(want, materialsLimit, coinLimit, MAX_BATCH, remaining);
      if (!Number.isFinite(n) || n < 0) n = 0;
      n = Math.floor(n);

      if (n > 0) {
        // Quantize AFTER every existing clip. `remaining` is the engine-only
        // manufactureRequired - produced - committed basis (never the legacy
        // policyContext.required). Rejection leaves all state untouched.
        const q = quantizeBatch(n, quantStep[id], remaining, batchCap[id], aligned[id]);
        if (q === 0 && n < quantStep[id] && remaining >= quantStep[id] && !aligned[id]) {
          // Sub-quantum clip with headroom still available: frozen rejection.
          // Telemetry only (blocked[id]); no starvation flags, no STARVED_*.
          starvation[id] = { material: false, coin: false };
          blocked[id] = true;
          emit(events, 'DECISION_REJECTED', id, 0, BELOW_QUANTUM);
          continue;
        }
        n = q;
      }

      if (n > 0) {
        const cost = totalCost(item.coinCost, n);
        st.coins -= cost;
        st.totalSpent += cost;
        const row = perItem[id];
        row.coinsSpent += cost;
        row.batches += 1;
        if (row.firstTick < 0) row.firstTick = st.tick;
        row.lastTick = st.tick;
        for (const ing of item.ingredients) {
          const src = byId.get(ing.item);
          if (src && src.type === 'base') continue;
          consume(ing.item, ing.quantity * n, events);
        }
        committed[id] = (committed[id] || 0) + n; // in-flight until completion
        slot.remainingTicks = craftTicksForMinutes(item.craftTimeMinutes);
        slot.batchSize = n;
        slot.state = 'ACTIVE';
        st.schedule.push({ tick: st.tick, item: id, batch: n });
        emit(events, 'DECISION_ACCEPTED', id, n, 'OK');
        emit(events, 'COIN_DEC', id, -cost, 'BATCH_COST');
        emit(events, 'MATERIAL_DEC', id, -n, 'INGREDIENTS');
        starvation[id] = { material: false, coin: false };
      } else {
        const material = materialsLimit === 0;
        const coin = coinLimit === 0;
        starvation[id] = { material, coin };
        const reason = want <= 0 ? 'POLICY_ZERO'
          : (material && coin ? 'MATERIAL_AND_COIN' : material ? 'MATERIAL' : coin ? 'COIN' : 'LIMITED');
        emit(events, 'DECISION_REJECTED', id, 0, reason);
        if (material) emit(events, 'STARVED_MATERIAL', id, 0, 'NO_INGREDIENTS');
        if (coin) emit(events, 'STARVED_COIN', id, 0, 'NO_COINS');
      }
    }

    // 3. advance timers + 4. completions
    for (const id of craftIds) {
      const slot = slots[id];
      if (slot.remainingTicks > 0) {
        slot.remainingTicks -= 1;
        perItem[id].busyTicks += 1;
        if (slot.remainingTicks === 0) {
          const n = slot.batchSize;
          producedNet[id] += n;
          produced[id] += n;
          committed[id] = 0; // batch completed: in-flight units now counted as produced
          perItem[id].units += n;
          slot.batchSize = 0;
          slot.state = 'IDLE';
          emit(events, 'STASH_UPDATE', id, n, 'BATCH_COMPLETE');
          emit(events, 'BATCH_DONE', id, n, 'CRAFT_DONE');
        }
      }
    }

    // 5. evaluate targets against current stash
    let allMet = true;
    for (const id of Object.keys(targets)) {
      const met = counted(id) >= targets[id];
      st.status[id] = met;
      if (!met) allMet = false;
    }
    if (allMet) {
      st.done = true;
      st.completionTick = st.tick;
    } else if (st.tick >= MAX_TICKS - 1) {
      st.done = true; // processed the last tick (8063)
    }
    return events;
  }

  function result() {
    const perItemOut = {};
    for (const id of craftIds) {
      const row = perItem[id];
      perItemOut[id] = {
        coinsSpent: row.coinsSpent,
        timeMinutes: row.busyTicks * TICK_MINUTES,
        busyTicks: row.busyTicks,
        batches: row.batches,
        units: row.units,
        firstTick: row.firstTick,
        lastTick: row.lastTick
      };
    }
    const shortfall = {};
    for (const id of Object.keys(targets)) shortfall[id] = Math.max(0, targets[id] - counted(id));
    return {
      feasible: Object.keys(shortfall).every((id) => shortfall[id] === 0),
      completionTick: st.completionTick,
      elapsedTicks: st.tick + 1,
      totalSpent: st.totalSpent,
      produced: { ...produced },
      schedule: st.schedule.slice(),
      shortfall,
      perItem: perItemOut
    };
  }

  const api = {
    step,
    result,
    // state.js stays untouched; augment its DTO with the blocked telemetry map.
    snapshot: () => ({ ...makeSnapshot(api), blocked: api.blocked }),
    get tick() { return st.tick; },
    get coins() { return st.coins; },
    get totalSpent() { return st.totalSpent; },
    get done() { return st.done; },
    get completionTick() { return st.completionTick; },
    get stash() {
      const out = {};
      for (const it of items) {
        out[it.id] = it.type === 'base'
          ? BASE_STASH
          : (producedNet[it.id] || 0) + (starterRemaining[it.id] || 0);
      }
      return out;
    },
    get slots() {
      const out = {};
      for (const id of craftIds) {
        const s = slots[id];
        out[id] = { remainingTicks: s.remainingTicks, batchSize: s.batchSize, state: s.state };
      }
      return out;
    },
    get starvation() {
      const out = {};
      for (const id of craftIds) out[id] = { ...starvation[id] };
      return out;
    },
    // Telemetry only (inputs.js unchanged): craftable-keyed DTO copy.
    get blocked() {
      const out = Object.create(null);
      for (const id of craftIds) out[id] = blocked[id] === true;
      return out;
    },
    // Null-prototype DTO copies of the internally derived batch metadata.
    get quantum() {
      const out = Object.create(null);
      for (const id of craftIds) out[id] = quantStep[id];
      return out;
    },
    get batchCaps() {
      const out = Object.create(null);
      for (const id of craftIds) out[id] = batchCap[id];
      return out;
    },
    get aligned() {
      const out = Object.create(null);
      for (const id of craftIds) out[id] = aligned[id];
      return out;
    },
    get status() { return { ...st.status }; },
    get required() { return { ...baseRequired }; },
    // Cap maps are keyed by craftable ids only (validated craftIds); bases
    // are absent. All are DTO copies so callers cannot mutate engine state.
    get manufactureRequired() {
      const out = {};
      for (const id of craftIds) out[id] = manufactureRequired[id] || 0;
      return out;
    },
    get produced() {
      const out = {};
      for (const id of craftIds) out[id] = produced[id] || 0;
      return out;
    },
    get committed() {
      const out = {};
      for (const id of craftIds) out[id] = committed[id] || 0;
      return out;
    }
  };

  return api;
}
