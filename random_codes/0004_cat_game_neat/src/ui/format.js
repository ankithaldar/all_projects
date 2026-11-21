// Pure, DOM-free formatting helpers shared by all UI modules.
// Every dynamic string that reaches the DOM passes through capString().

export const SVG_NS = 'http://www.w3.org/2000/svg';
export const MAX_STR = 120;
export const LOG_CAP = 5000;
export const TICK_MINUTES = 5;

/** Coerce any value to a bounded, display-safe string (XSS-safe via textContent). */
export function capString(value, max = MAX_STR) {
  let s = value == null ? '' : String(value);
  if (s.length > max) s = s.slice(0, max);
  return s;
}

/**
 * Live countdown text from a slot's remaining ticks.
 * >= 60 minutes -> "Xd Yh Zm"; otherwise -> "Mm 00s".
 */
export function formatCountdown(remainingTicks, tickMinutes = TICK_MINUTES) {
  const t = Number.isFinite(remainingTicks) && remainingTicks > 0
    ? Math.floor(remainingTicks) : 0;
  const minutes = t * tickMinutes;
  if (minutes >= 60) {
    const d = Math.floor(minutes / 1440);
    const h = Math.floor((minutes % 1440) / 60);
    const m = minutes % 60;
    return `${d}d ${h}h ${m}m`;
  }
  // Minutes are tick-granular (multiples of 5), so seconds are always 00.
  return `${minutes}m 00s`;
}

/**
 * Classify a slot's bottleneck. Only free (IDLE, non-counting) slots can be
 * starved; a busy slot always renders its countdown, never a highlight.
 */
export function bottleneckKind(slot, starvation) {
  if (!slot || slot.state !== 'IDLE') return 'none';
  const m = !!(starvation && starvation.material);
  const c = !!(starvation && starvation.coin);
  if (m && c) return 'both';
  if (m) return 'material';
  if (c) return 'coin';
  return 'none';
}

/** Clamp an integer into [min,max], returning `fallback` when invalid. */
export function intInRange(value, fallback, min, max) {
  const n = typeof value === 'number' ? value : parseInt(String(value), 10);
  return Number.isFinite(n) && Number.isInteger(n) && n >= min && n <= max
    ? n : fallback;
}

/** Convert a tick count to a 28-day-completion day number (1-based). */
export function tickToDay(tick) {
  if (!Number.isFinite(tick) || tick < 0) return 0;
  return Math.floor(tick / 288) + 1;
}
