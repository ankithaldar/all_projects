// UI view of the canonical item registry (single source of truth: sim/items.js).
// No fallback dataset: the UI renders the real game only.
import { ITEMS } from '../sim/items.js';

export {
  ITEMS,
  DAG_EDGES,
  CRAFTABLE_IDS,
  craftTicks,
  effectiveTier,
} from '../sim/items.js';

const NAMES = new Map(ITEMS.map((i) => [i.id, i.name || i.id]));
export function nameOf(id) { return NAMES.get(id) || id; }
