/**
 * harness.js — minimal zero-framework test harness (plan Interfaces).
 *
 * Export surface (exact contract):
 *   test(name, fn)        register a test (fn may be sync or return a Promise)
 *   assert(cond, msg)     throw when cond is falsy
 *   assertEqual(a,b,msg)  strict equality
 *   assertDeepEqual(a,b,msg)
 *   assertThrows(fn,msg)  pass only when fn throws
 *   runTests()            run all registered tests, render into #test-results,
 *                         set window.__TEST_RESULTS__ = {passed,failed,failures}
 *
 * Security: all rendering uses createElement + textContent (never innerHTML).
 */

/** @type {Array<{name:string, fn:Function}>} */
const tests = [];

export function test(name, fn) {
  tests.push({ name: String(name), fn });
}

export function assert(cond, msg) {
  if (!cond) throw new Error(msg || 'assertion failed');
}

export function assertEqual(a, b, msg) {
  if (a !== b) {
    throw new Error(msg || `expected equal: ${describe(a)} !== ${describe(b)}`);
  }
}

export function assertDeepEqual(a, b, msg) {
  if (!deepEqual(a, b)) {
    throw new Error(msg || `expected deep equal: ${describe(a)} !== ${describe(b)}`);
  }
}

export function assertThrows(fn, msg) {
  let threw = false;
  try {
    fn();
  } catch (_e) {
    threw = true;
  }
  if (!threw) throw new Error(msg || 'expected function to throw');
}

function describe(v) {
  try {
    const s = JSON.stringify(v);
    return s === undefined ? String(v) : s;
  } catch (_e) {
    return String(v);
  }
}

function deepEqual(a, b) {
  if (a === b) return true;
  if (typeof a === 'number' && typeof b === 'number') {
    return Number.isNaN(a) && Number.isNaN(b);
  }
  if (a === null || b === null || typeof a !== 'object' || typeof b !== 'object') {
    return false;
  }
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const ak = Object.keys(a);
  const bk = Object.keys(b);
  if (ak.length !== bk.length) return false;
  for (const k of ak) {
    if (!Object.prototype.hasOwnProperty.call(b, k)) return false;
    if (!deepEqual(a[k], b[k])) return false;
  }
  return true;
}

function renderRow(container, cls, label, detail) {
  const row = document.createElement('div');
  row.className = 'test-row ' + cls;
  const nameEl = document.createElement('span');
  nameEl.className = 'test-name';
  nameEl.textContent = label; // text-only: no markup injection
  row.appendChild(nameEl);
  if (detail) {
    const dEl = document.createElement('span');
    dEl.className = 'test-detail';
    // Cap detail length to avoid unbounded DOM text.
    dEl.textContent = String(detail).slice(0, 240);
    row.appendChild(dEl);
  }
  container.appendChild(row);
}

/**
 * Run all registered tests. Safe to call once from tests.html after imports.
 * Returns a Promise that resolves to the results object. Results are also
 * published incrementally to window.__TEST_RESULTS__ so partial state is visible.
 */
export async function runTests() {
  const result = { passed: 0, failed: 0, failures: [] };
  if (typeof window !== 'undefined') window.__TEST_RESULTS__ = result;

  const container = typeof document !== 'undefined'
    ? document.getElementById('test-results')
    : null;

  const summary = document.createElement ? document.createElement('div') : null;
  // Render order: rows first, summary at the end, so append summary lazily.

  for (const { name, fn } of tests) {
    try {
      const maybePromise = fn();
      if (maybePromise && typeof maybePromise.then === 'function') {
        await maybePromise;
      }
      result.passed += 1;
      if (container) renderRow(container, 'pass', 'PASS', name);
    } catch (err) {
      result.failed += 1;
      const message = (err && err.message) ? String(err.message) : String(err);
      result.failures.push({ name, message: message.slice(0, 240) });
      if (container) renderRow(container, 'fail', 'FAIL', `${name} :: ${message}`);
    }
  }

  if (container && summary) {
    summary.className = 'test-summary';
    summary.textContent = `TOTAL passed=${result.passed} failed=${result.failed}`;
    container.appendChild(summary);
  }

  if (typeof window !== 'undefined') window.__TEST_RESULTS__ = result;
  return result;
}
