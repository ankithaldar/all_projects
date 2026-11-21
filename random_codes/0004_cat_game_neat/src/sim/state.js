/**
 * state.js — snapshot DTO builder. Reads getters off a DOM-free engine and
 * returns a structured-clone-safe plain object with bounded integer fields.
 */
export function makeSnapshot(engine) {
  return {
    type: 'SNAPSHOT',
    tick: engine.tick,
    coins: engine.coins,
    totalSpent: engine.totalSpent,
    stash: engine.stash,
    slots: engine.slots,
    starvation: engine.starvation,
    status: engine.status,
    required: engine.required,
    manufactureRequired: engine.manufactureRequired,
    produced: engine.produced,
    committed: engine.committed,
    done: engine.done
  };
}
