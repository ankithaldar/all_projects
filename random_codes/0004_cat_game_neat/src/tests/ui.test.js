/**
 * ui.test.js — browser UI behavior (plan UI Requirements / Security Controls).
 *
 * Runs in tests.html with the real DOM, and in Node with the minimal DOM stub
 * below. All assertions use the zero-framework harness from harness.js.
 */
import { test, assert, assertEqual, assertDeepEqual } from './harness.js';
import { ITEMS, DAG_EDGES, effectiveTier, craftTicks } from '../sim/items.js';
import { createDag } from '../ui/dag.js';
import { createHud } from '../ui/hud.js';
import { createLogger } from '../ui/logger.js';
import { createControls } from '../ui/controls.js';
import { createResults } from '../ui/results.js';
import { KNOWN_TARGET_IDS, KNOWN_CRAFTABLE_IDS } from '../validate.js';
import { EMBEDDED_RAW } from '../config.js';
import {
  createSimClock, formatSimClock, SIM_CLOCK_ID, SIM_TICK_MINUTES
} from '../main.js';
import {
  readInitialState, createInitialStatePanel,
  INITIAL_COINS_ID, INITIAL_APPLY_ID, INITIAL_RESET_ID, INITIAL_ZERO_ID,
  initialTargetId, initialStashId
} from '../ui/initial-state.js';

const SVGNS = 'http://www.w3.org/2000/svg';

/* --------------------------- DOM stub (Node only) -------------------------- */

function ensureDocument() {
  if (typeof document !== 'undefined' && document.createElement) return;
  class ClassList {
    constructor(el) { this.el = el; }
    set() { return new Set(String(this.el.className || '').split(/\s+/).filter(Boolean)); }
    write(s) { this.el.className = Array.from(s).join(' '); }
    add(...c) { const s = this.set(); c.forEach((x) => s.add(String(x))); this.write(s); }
    remove(...c) { const s = this.set(); c.forEach((x) => s.delete(String(x))); this.write(s); }
    contains(c) { return this.set().has(String(c)); }
    toggle(c, force) {
      const on = force === undefined ? !this.contains(c) : !!force;
      if (on) this.add(c); else this.remove(c);
      return on;
    }
  }
  class Element {
    constructor(tag, ns = null) {
      this.localName = String(tag).toLowerCase();
      this.tagName = String(tag).toUpperCase();
      this.namespaceURI = ns;
      this.children = [];
      this.parentNode = null;
      this.className = '';
      this.classList = new ClassList(this);
      this.attrs = new Map();
      this.listeners = new Map();
      this.style = {};
      this.dataset = {};
      this.value = '';
      this.id = '';
      this.hidden = false;
      this.scrollTop = 0;
      this.scrollHeight = 0;
      this._text = '';
    }
    get firstChild() { return this.children[0] || null; }
    get lastChild() { return this.children[this.children.length - 1] || null; }
    appendChild(n) { n.parentNode = this; this.children.push(n); return n; }
    insertBefore(n, ref) {
      const i = this.children.indexOf(ref);
      this.children.splice(i < 0 ? this.children.length : i, 0, n);
      n.parentNode = this; return n;
    }
    removeChild(n) {
      const i = this.children.indexOf(n);
      if (i >= 0) { this.children.splice(i, 1); n.parentNode = null; }
      return n;
    }
    setAttribute(k, v) {
      const key = String(k), val = String(v);
      this.attrs.set(key, val);
      if (key === 'class') this.className = val;
      if (key === 'id') this.id = val;
    }
    getAttribute(k) { return this.attrs.has(String(k)) ? this.attrs.get(String(k)) : null; }
    removeAttribute(k) { this.attrs.delete(String(k)); }
    hasAttribute(k) { return this.attrs.has(String(k)); }
    set textContent(v) {
      this.children.forEach((c) => { c.parentNode = null; });
      this.children = [];
      this._text = v == null ? '' : String(v);
    }
    get textContent() {
      return this._text + this.children.map((c) => c.textContent).join('');
    }
    addEventListener(t, fn) {
      if (!this.listeners.has(t)) this.listeners.set(t, []);
      this.listeners.get(t).push(fn);
    }
    _fire(t, extra = {}) {
      (this.listeners.get(t) || []).slice().forEach((fn) => fn(Object.assign({
        type: t, target: this, preventDefault() {}, stopPropagation() {},
      }, extra)));
    }
  }
  globalThis.document = {
    createElement: (t) => new Element(t),
    createElementNS: (ns, t) => new Element(t, ns),
    getElementById: () => null,
    body: new Element('body'),
  };
}
ensureDocument();

/* ------------------------------- helpers ----------------------------------- */

function el(tag, ns) {
  return ns ? document.createElementNS(ns, tag) : document.createElement(tag);
}
function findAll(root, pred) {
  const out = [];
  (function walk(n) {
    for (const c of (n.children || [])) { if (pred(c)) out.push(c); walk(c); }
  })(root);
  return out;
}
function byClass(root, cls) {
  return findAll(root, (n) => n.classList && n.classList.contains(cls));
}
function byLocalName(root, name) {
  return findAll(root, (n) => n.localName === name);
}
function withAttr(root, key, value) {
  return findAll(root, (n) => typeof n.getAttribute === 'function' && n.getAttribute(key) === value);
}
function fireClick(node) {
  if (typeof node._fire === 'function') { node._fire('click'); return; }
  node.dispatchEvent(new MouseEvent('click', { bubbles: true }));
}
/** Works with both the Node stub (_fire) and a real browser DOM (dispatchEvent). */
function fireEvent(node, type) {
  if (typeof node._fire === 'function') node._fire(type);
  else node.dispatchEvent(new Event(type, { bubbles: true }));
}
function makeSnapshot(overrides = {}) {
  const stash = {}, slots = {}, starvation = {}, status = {}, required = {};
  for (const it of ITEMS) {
    stash[it.id] = 0;
    slots[it.id] = { state: 'IDLE', remainingTicks: 0, batchSize: 0 };
    starvation[it.id] = { material: false, coin: false };
    status[it.id] = false;
    required[it.id] = 0;
  }
  return Object.assign({
    type: 'SNAPSHOT', tick: 1, coins: 100, totalSpent: 0,
    stash, slots, starvation, status, required, done: false,
  }, overrides);
}

/* --------------------------------- tests ----------------------------------- */

test('logger: 5001 events cap the ring at 5000 and the DOM at 600 rows', () => {
  const list = el('ul');
  const log = createLogger(list);
  const events = [];
  for (let i = 0; i <= 5000; i++) {
    events.push({
      tick: i, type: 'DECISION_ACCEPTED', item: 'string', batch: 1,
      delta: -10, reason: i === 0 ? 'OLDEST_EVENT_MARKER'
        : i === 5000 ? 'NEWEST_EVENT_MARKER' : 'filler',
    });
  }
  log.append(events);
  assert(byClass(list, 'log-row').length <= 600, 'DOM rows must be capped at 600');
  log.setFilter((e) => e.tick === 0);
  assert(!list.textContent.includes('OLDEST_EVENT_MARKER'), 'event 0 must be evicted from the 5000 ring');
  log.setFilter((e) => e.tick === 5000);
  assert(list.textContent.includes('NEWEST_EVENT_MARKER'), 'newest event must remain');
  log.clear();
  assertEqual(list.children.length, 0, 'clear() must empty the list');
});

test('DAG click-to-trace highlights the full ancestry to depth-0 bases and clears', () => {
  const svg = el('svg', SVGNS);
  const holder = el('div');
  holder.appendChild(svg);
  createDag(svg);
  const groups = new Map(byClass(svg, 'dag-node')
    .map((g) => [g.getAttribute('data-item-id'), g]));

  const ancestors = new Set();
  (function walk(id) {
    if (ancestors.has(id)) return;
    ancestors.add(id);
    const item = ITEMS.find((i) => i.id === id);
    for (const ing of (item && item.ingredients) || []) walk(ing.item);
  })('artifact');
  assert(Array.from(ancestors).some((id) => (ITEMS.find((i) => i.id === id) || {}).type === 'base'),
    'ancestry must reach depth-0 bases');

  const artifact = groups.get('artifact');
  assert(!!artifact, 'artifact node must exist');
  fireClick(artifact);
  for (const id of ancestors) {
    assert(groups.get(id).classList.contains('trace-active'), `${id} must be traced`);
  }
  const toast = byClass(holder, 'trace-toast')[0];
  assert(toast && toast.textContent.includes('Trace'), 'toast must describe the trace');

  fireClick(artifact);
  assertEqual(byClass(svg, 'trace-dim').length, 0, 'second click clears trace');

  // A smaller root proves dimming: artifact is outside `string`'s ancestry.
  const small = groups.get('string');
  fireClick(small);
  assert(groups.get('artifact').classList.contains('trace-dim'),
    'nodes outside the traced ancestry must be dimmed');
  fireClick(small);
  assertEqual(byClass(svg, 'trace-dim').length, 0, 'second click clears the small trace');
  fireClick(artifact);
  fireClick(svg);
  assertEqual(byClass(svg, 'trace-dim').length, 0, 'canvas click clears trace');
});

test('tooltips show every required DAG node and edge field', () => {
  const svg = el('svg', SVGNS);
  const dag = createDag(svg);
  // Synthetic ACTIVE + coin-starved snapshot on `wood` (short enough that all
  // fields survive the MAX_STR=120 cap).
  dag.update(makeSnapshot({
    stash: { wood: 7 },
    slots: { wood: { state: 'ACTIVE', remainingTicks: 1, batchSize: 4 } },
    // Starvation is only rendered for FREE slots, so the starved node is a
    // different item in the same snapshot.
    starvation: { string: { material: false, coin: true } },
    required: { wood: 12 },
  }));
  const wood = withAttr(svg, 'data-item-id', 'wood')[0];
  const tip = byLocalName(wood, 'title')[0].textContent;
  const item = ITEMS.find((i) => i.id === 'wood');
  const requiredFields = [
    ['item name', item.name],
    ['raw tier', `tier ${item.tier}`],
    ['computed depth', `depth ${effectiveTier('wood')}`],
    ['recipe ingredient + quantity', '3x tree'],
    ['coin cost', String(item.coinCost)],
    ['craft minutes + ticks', `${item.craftTimeMinutes}m/${craftTicks('wood')}t`],
    ['current stash', 'stash 7'],
    ['remaining demand from snapshot.required', 'demand 5'],
    ['slot status', 'ACTIVE'],
    ['countdown', '5m 00s'],
    ['batch size', 'x4'],
  ];
  for (const [label, fragment] of requiredFields) {
    assert(tip.includes(fragment), `node tooltip missing ${label} (${fragment}): ${tip}`);
  }
  assert(tip.length <= 120, 'node tooltip must respect MAX_STR=120');

  const starved = withAttr(svg, 'data-item-id', 'string')[0];
  const starvedTip = byLocalName(starved, 'title')[0].textContent;
  assert(starvedTip.includes('starve coin starved'),
    `free-slot starvation reason must be shown: ${starvedTip}`);
  assert(tip.includes('starve none'),
    'busy slot must not show starvation (production rule)');

  const edgeDef = DAG_EDGES[0];
  const edgeLine = withAttr(svg, 'data-from', edgeDef.from)
    .filter((n) => n.getAttribute('data-to') === edgeDef.to)[0];
  const edgeTip = byLocalName(edgeLine, 'title')[0].textContent;
  const nameOfId = (id) => (ITEMS.find((i) => i.id === id) || {}).name || id;
  assert(edgeTip.includes(nameOfId(edgeDef.from)), 'edge tooltip must name the source');
  assert(edgeTip.includes('→'), 'edge tooltip must show direction');
  assert(edgeTip.includes(nameOfId(edgeDef.to)), 'edge tooltip must name the target');
  assert(edgeTip.includes(`qty ${edgeDef.quantity}`), 'edge tooltip must state required quantity');
});

test('tooltips show full HUD and logger-row details', () => {
  const stashEl = el('div'), coinEl = el('strong'), spentEl = el('strong');
  createHud({ stashEl, coinEl, spentEl })
    .update(makeSnapshot({ stash: { wood: 7 } }));
  for (const [label, node, fragments] of [
    ['coin balance', coinEl, ['Coin balance', '+210', 'batch cost']],
    ['totalSpent', spentEl, ['totalSpent', 'cumulative', 'initial - final']],
    ['stash row', withAttr(stashEl, 'data-item-id', 'wood')[0],
      ['Wood', 'tier 1', '7 units', 'every display snapshot']],
  ]) {
    const title = node && node.getAttribute('title');
    assert(title, `${label} tooltip must exist`);
    for (const f of fragments) {
      assert(title.includes(f), `${label} tooltip missing ${f}: ${title}`);
    }
  }

  const list = el('ul');
  createLogger(list).append({
    tick: 42, type: 'STARVED_COIN', item: 'wood', batch: 0,
    delta: -5, reason: 'no coins',
  });
  const rowTitle = byClass(list, 'log-row')[0].getAttribute('title');
  for (const f of ['tick 42', 'STARVED_COIN', 'wood', 'batch 0', 'delta -5', 'no coins']) {
    assert(rowTitle.includes(f), `logger tooltip missing ${f}: ${rowTitle}`);
  }
});

test('control tooltips in real index.html document actions and valid ranges', async () => {
  // Source inspection (same pattern as worker-bridge.test.js): read the REAL
  // index.html rather than stubbing values, so deleting a title fails the test.
  async function readSource(browserUrl, nodeRelative) {
    if (typeof window !== 'undefined' && typeof fetch === 'function') {
      const response = await fetch(browserUrl);
      assert(response.ok, `fetch failed for ${browserUrl}`);
      return await response.text();
    }
    const { readFileSync } = await import('node:fs');
    return readFileSync(new URL(nodeRelative, import.meta.url), 'utf8');
  }
  const source = await readSource('./index.html', '../../index.html');

  function titleAttr(id) {
    const tagRe = new RegExp('<[a-zA-Z][^>]*\\bid=["\']' + id + '["\'][^>]*>', 'i');
    const tag = source.match(tagRe);
    assert(tag, `#${id} element missing from index.html`);
    const title = tag[0].match(/\btitle\s*=\s*(["'])([\s\S]*?)\1/i);
    assert(title, `#${id} has no title attribute in index.html`);
    const value = title[2].trim();
    assert(value.length > 0, `#${id} title must not be empty`);
    return value;
  }

  const expectations = {
    'btn-start': ['Start', 'validate controls', 'display run'],
    'btn-pause': ['Pause', 'resume'],
    'btn-reset': ['Reset', 'clear the log'],
    speed: ['1x', '500 ms/tick', '10x', '50 ms/tick', 'fast'],
    seed: ['uint32', '0..4294967295'],
    preset: ['reuse', 'empty', '0 coins'],
    'pop-size': ['1..500'],
    generations: ['1..10000'],
    'log-filter': ['Filter log rows', 'type or item'],
    'log-clear': ['Clear all buffered', 'displayed log rows'],
  };
  // Every DOM-registry control id from the plan must appear in the assertions.
  assertEqual(Object.keys(expectations).length, 10, 'all 10 control ids must be checked');
  for (const id of Object.keys(expectations)) {
    const title = titleAttr(id);
    for (const fragment of expectations[id]) {
      assert(title.includes(fragment), `#${id} title missing "${fragment}": ${title}`);
    }
  }
});

test('DAG columns follow dependency depth, not raw YAML tier', () => {
  const svg = el('svg', SVGNS);
  createDag(svg);
  const groups = new Map(byClass(svg, 'dag-node')
    .map((g) => [g.getAttribute('data-item-id'), g]));
  const xOf = (id) => Number(groups.get(id).getAttribute('data-x'));
  for (const edge of DAG_EDGES) {
    assert(xOf(edge.to) > xOf(edge.from),
      `${edge.from} -> ${edge.to} must advance left-to-right by depth`);
  }
  const artifactDepth = effectiveTier('artifact');
  for (const it of ITEMS) {
    const node = groups.get(it.id);
    assertEqual(Number(node.getAttribute('data-depth')), effectiveTier(it.id),
      `${it.id} depth attribute`);
    if (effectiveTier(it.id) < artifactDepth) {
      assert(xOf(it.id) < xOf('artifact'), `${it.id} must be left of artifact`);
    }
  }
});

test('XSS payloads through createDag and createLogger create no element or handler', () => {
  const payload = '<img src=x onerror=alert(1)>';
  const svg = el('svg', SVGNS);
  const snapshot = makeSnapshot();
  snapshot.stash[payload] = 1;
  snapshot.slots[payload] = { state: 'ACTIVE', remainingTicks: 2, batchSize: 1 };
  createDag(svg).update(snapshot);
  const list = el('ul');
  createLogger(list).append({
    tick: 1, type: 'XSS', item: payload, batch: 0, delta: 0, reason: payload,
  });
  assertEqual(byLocalName(svg, 'img').length + byLocalName(list, 'img').length, 0,
    'no img element may be created');
  assertEqual(byLocalName(list, 'script').length, 0, 'no script element may be created');
  assertEqual(findAll(list, (n) => n.hasAttribute && n.hasAttribute('onerror')).length, 0,
    'no event handler attribute may be created');
});

test('results: telemetry TOTAL row equals the sum of per-item coins and minutes', () => {
  const host = el('div');
  const results = createResults(host);
  results.showGenomeTelemetry({
    itemId: 'artifact', generation: 1, genomeIndex: 0, fitness: -1,
    feasible: true, completionTick: 10, elapsedTicks: 10, totalSpent: 3000,
    perItem: {
      string: { coinsSpent: 1000, timeMinutes: 500, busyTicks: 10, batches: 1, units: 5 },
      wood: { coinsSpent: 2000, timeMinutes: 300, busyTicks: 20, batches: 2, units: 8 },
    },
  });
  const totalRow = byClass(host, 'results-total-row')[0];
  assert(!!totalRow, 'TOTAL row must exist');
  const cells = totalRow.children;
  assertEqual(cells[0].textContent, 'TOTAL');
  assertEqual(cells[1].textContent, '3000', 'coins TOTAL');
  assertEqual(cells[2].textContent, '800', 'minutes TOTAL');
});

test('results: telemetry view is bounded to the last 20 events', () => {
  const host = el('div');
  const results = createResults(host);
  for (let i = 0; i < 25; i++) {
    results.showGenomeTelemetry({
      itemId: 'artifact', generation: i, genomeIndex: i, fitness: -i,
      feasible: true, completionTick: i, elapsedTicks: i, totalSpent: i,
      perItem: { string: { coinsSpent: i, timeMinutes: i, busyTicks: 1, batches: 1, units: 1 } },
    });
  }
  const headers = byClass(host, 'results-title')
    .filter((n) => n.textContent.includes('Genome telemetry'));
  assert(headers.length <= 20, `telemetry sections must be capped at 20, got ${headers.length}`);
  assert(host.textContent.includes('gen 24'), 'newest telemetry event must be visible');
  assert(!host.textContent.includes('gen 4 '), 'telemetry events older than 20 must be dropped');
});

test('controls wires #log-filter and #log-clear to the active logger', () => {
  const original = globalThis.document.getElementById;
  const nodes = new Map();
  globalThis.document.getElementById = (id) => {
    if (!nodes.has(id)) {
      const node = el(id.startsWith('btn') || id === 'log-clear' ? 'button' : 'input');
      node.id = id;
      node.value = '';
      nodes.set(id, node);
    }
    return nodes.get(id);
  };
  const list = el('ul');
  const log = createLogger(list);
  log.append([
    { tick: 1, type: 'ALPHA', item: 'string', delta: 0 },
    { tick: 2, type: 'BETA', item: 'wood', delta: 0 },
  ]);
  createControls({ logger: log });
  const filterEl = nodes.get('log-filter');
  filterEl.value = 'BETA';
  if (typeof filterEl._fire === 'function') filterEl._fire('input');
  else filterEl.dispatchEvent(new Event('input', { bubbles: true }));
  assert(list.textContent.includes('BETA'), 'filter keeps matching events');
  assert(!list.textContent.includes('ALPHA'), 'filter hides non-matching events');
  const clearEl = nodes.get('log-clear');
  if (typeof clearEl._fire === 'function') clearEl._fire('click');
  else clearEl.click();
  assertEqual(list.children.length, 0, 'clear button empties the log');
  globalThis.document.getElementById = original;
});

function withControlValues(values, fn) {
  const ids = ['btn-start', 'btn-pause', 'btn-reset', 'speed', 'seed', 'preset', 'pop-size', 'generations'];
  const original = globalThis.document.getElementById;
  globalThis.document.getElementById = (id) => {
    if (!ids.includes(id)) return null;
    const node = el(id.startsWith('btn') ? 'button' : 'input');
    node.id = id;
    node.value = Object.prototype.hasOwnProperty.call(values, id) ? values[id] : '';
    return node;
  };
  try {
    return fn();
  } finally {
    globalThis.document.getElementById = original;
  }
}

test('controls: valid values pass validateControl unchanged', () => {
  const cfg = withControlValues({
    speed: '10x', seed: '7', preset: 'reuse', 'pop-size': '25', generations: '30',
  }, () => createControls({}).getConfig());
  assertEqual(cfg.speed, '10x');
  assertEqual(cfg.seed, 7);
  assertEqual(cfg.preset, 'reuse');
  assertEqual(cfg.popSize, 25);
  assertEqual(cfg.generations, 30);
});

test('controls: out-of-range values fall back to declared defaults', () => {
  const cfg = withControlValues({
    speed: 'bogus', seed: '4294967296', preset: 'nope', 'pop-size': '999', generations: '0',
  }, () => createControls({}).getConfig());
  assertEqual(cfg.speed, '1x', 'invalid speed falls back');
  assertEqual(cfg.seed, 1, 'seed above uint32 falls back');
  assertEqual(cfg.preset, 'reuse', 'unknown preset falls back');
  assertEqual(cfg.popSize, 20, 'pop-size above 500 falls back');
  assertEqual(cfg.generations, 100, 'generations below 1 falls back');
});

/* ------------- manufactured-count cap badge / tooltip (amendment) -------- */

const BASE_ID = (ITEMS.find((i) => i.type === 'base') || {}).id;
const CRAFT_ID = (ITEMS.find((i) => i.type === 'craftable') || {}).id;

function dagNode(svg, id) { return withAttr(svg, 'data-item-id', id)[0]; }
function badgeOf(group) { return byClass(group, 'node-badge')[0]; }
function tipOf(group) { return byLocalName(group, 'title')[0].textContent; }
function hasClass(group, cls) { return !!group && group.classList.contains(cls); }

test('DAG craftable badge renders normalized produced/manufactureRequired via textContent', () => {
  assert(!!CRAFT_ID && !!BASE_ID, 'fixture needs at least one craftable and one base');
  const svg = el('svg', SVGNS);
  const dag = createDag(svg);

  // Floors positives, clamps negatives, rejects non-finite/payload values.
  dag.update(makeSnapshot({
    manufactureRequired: { [CRAFT_ID]: 12.9 },
    produced: { [CRAFT_ID]: 5.7 },
    committed: { [CRAFT_ID]: 1 },
  }));
  const badge = badgeOf(dagNode(svg, CRAFT_ID));
  assert(!!badge, 'craftable must have its own badge element');
  assertEqual(badge.localName, 'text', 'badge must be an SVG <text>');
  assertEqual(badge.namespaceURI, SVGNS, 'badge must be in the SVG namespace');
  assertEqual(badge.textContent, '5/12', 'badge shows floored produced/manufactureRequired');
  assertEqual(badge.children.length, 0, 'badge text must be written via textContent only');

  dag.update(makeSnapshot({
    manufactureRequired: { [CRAFT_ID]: -4 },
    produced: { [CRAFT_ID]: -9 },
  }));
  assertEqual(badgeOf(dagNode(svg, CRAFT_ID)).textContent, '0/0',
    'negative finite values clamp to 0 (craftables may show 0/0)');

  dag.update(makeSnapshot({
    manufactureRequired: { [CRAFT_ID]: Infinity },
    produced: { [CRAFT_ID]: '<img src=x onerror=alert(1)>' },
  }));
  assertEqual(badgeOf(dagNode(svg, CRAFT_ID)).textContent, '0/0',
    'Infinity and payload strings normalize to 0');
});

test('DAG base nodes show n/a and never 0/0 or COMPLETE', () => {
  const svg = el('svg', SVGNS);
  const dag = createDag(svg);
  // Even if a malicious/incorrect snapshot assigns cap data to a base id,
  // base nodes must stay n/a and never enter a COMPLETE state.
  dag.update(makeSnapshot({
    manufactureRequired: { [BASE_ID]: 10 },
    produced: { [BASE_ID]: 99 },
    committed: { [BASE_ID]: 5 },
    status: { [BASE_ID]: true },
  }));
  const group = dagNode(svg, BASE_ID);
  assertEqual(badgeOf(group).textContent, 'n/a', 'base badge must read n/a');
  assert(!badgeOf(group).textContent.includes('0/0'), 'base badge must never read 0/0');
  assert(!hasClass(group, 'is-complete') && !hasClass(group, 'complete'),
    'base nodes must never be COMPLETE');
});

test('DAG COMPLETE requires finite values and outranks stash-based target-met', () => {
  const svg = el('svg', SVGNS);
  const dag = createDag(svg);
  const done = (mfr, prod, status) => {
    dag.update(makeSnapshot({
      manufactureRequired: { [CRAFT_ID]: mfr },
      produced: { [CRAFT_ID]: prod },
      status: { [CRAFT_ID]: status },
    }));
    return dagNode(svg, CRAFT_ID);
  };

  let g = done(10, 10, true);
  assert(hasClass(g, 'is-complete'), 'produced == manufactureRequired is COMPLETE');
  assert(!hasClass(g, 'target-met'), 'COMPLETE outranks target-met when both apply');

  g = done(10, 12, false);
  assert(hasClass(g, 'is-complete'), 'produced > manufactureRequired is COMPLETE');

  g = done(10, 3, true);
  assert(!hasClass(g, 'is-complete'), 'produced < manufactureRequired is not COMPLETE');
  assert(hasClass(g, 'target-met'), 'target-met applies when not COMPLETE');

  g = done(10, 'not-a-number', true);
  assert(!hasClass(g, 'is-complete'), 'non-finite produced must not be COMPLETE');
  assert(hasClass(g, 'target-met'), 'non-finite cap values fall back to target-met');

  g = done(Infinity, Infinity, true);
  assert(!hasClass(g, 'is-complete'), 'Infinite cap values must not be COMPLETE');
});

test('DAG cap-map payload strings and forbidden keys create no element or handler', () => {
  const payload = '<img src=x onerror=alert(1)>';
  const svg = el('svg', SVGNS);
  const dag = createDag(svg);
  // Own `__proto__` / `constructor` keys (JSON.parse keeps them as own keys),
  // plus a payload as a value and as an entire map.
  const poisoned = JSON.parse(
    '{"__proto__":"' + payload + '","constructor":"' + payload + '","' + CRAFT_ID + '":9}');
  dag.update(makeSnapshot({
    manufactureRequired: poisoned,
    produced: poisoned,
    committed: payload, // invalid map shape: must degrade, not throw/inject
    stash: { [payload]: 1 },
  }));
  assertEqual(byLocalName(svg, 'img').length, 0, 'no img element from cap-map payloads');
  assertEqual(byLocalName(svg, 'script').length, 0, 'no script element');
  assertEqual(findAll(svg, (n) => n.hasAttribute && n.hasAttribute('onerror')).length, 0,
    'no event-handler attribute from cap-map payloads');
  assertEqual(badgeOf(dagNode(svg, CRAFT_ID)).textContent, '9/9',
    'only the legitimate own registry key is read; badge stays numeric');

  // A craftable key inherited via the prototype must be ignored (no own key).
  const inherited = Object.create({ [CRAFT_ID]: 999 });
  dag.update(makeSnapshot({
    manufactureRequired: inherited,
    produced: { [CRAFT_ID]: 2 },
  }));
  assertEqual(badgeOf(dagNode(svg, CRAFT_ID)).textContent, '2/0',
    'inherited/prototype cap values are ignored, own values are used');
});

test('DAG tooltip shows separately labelled, capped stash/required/mfr/produced/committed', () => {
  const svg = el('svg', SVGNS);
  const dag = createDag(svg);
  dag.update(makeSnapshot({
    stash: { [CRAFT_ID]: 7 },
    required: { [CRAFT_ID]: 12 },
    manufactureRequired: { [CRAFT_ID]: 10 },
    produced: { [CRAFT_ID]: 3 },
    committed: { [CRAFT_ID]: 1 },
  }));
  const tip = tipOf(dagNode(svg, CRAFT_ID));
  for (const [label, value] of [
    ['stash', '7'], ['required', '12'], ['manufactureRequired', '10'],
    ['produced', '3'], ['committed', '1'],
  ]) {
    assert(tip.includes(value), `tooltip must show labelled ${label} value (${value}): ${tip}`);
  }
  for (const label of ['stash', 'req', 'mfr', 'prod', 'cmt']) {
    assert(tip.includes(label + ' '), `tooltip must label ${label}: ${tip}`);
  }
  assert(tip.length <= 120, 'tooltip must stay within MAX_STR=120');

  // A 300-char payload in a cap field must never surface raw text in the tip.
  dag.update(makeSnapshot({ manufactureRequired: { [CRAFT_ID]: 'A'.repeat(300) } }));
  const payloadTip = tipOf(dagNode(svg, CRAFT_ID));
  assert(!payloadTip.includes('AAAA'), 'cap-map payload strings must be capped/normalized away');
  assert(payloadTip.length <= 120, 'tooltip stays capped with hostile cap-map input');
});

test('DAG badge + COMPLETE have visible computed styles (browser path)', async () => {
  // Node's DOM stub has no CSS engine: skip gracefully there. In a real
  // browser, verify the ACTUAL stylesheet yields visible, distinct styling.
  if (typeof getComputedStyle !== 'function' || typeof fetch !== 'function') return;
  const res = await fetch('./styles.css');
  assert(res.ok, 'styles.css must be fetchable in the browser test path');
  const css = await res.text();
  assert(/\.node-badge\s*\{[^}]*fill\s*:/.test(css),
    'styles.css must define a visible .node-badge fill');
  assert(/\.is-complete\s+\.node-box\s*\{[^}]*stroke\s*:/.test(css),
    'styles.css must define an .is-complete node-box stroke');

  // tests.html has no <link> to styles.css; inject the real sheet so computed
  // styles match what index.html users see. textContent only — no HTML sinks.
  const style = document.createElement('style');
  style.textContent = css;
  (document.head || document.body).appendChild(style);

  const svg = el('svg', SVGNS);
  const holder = el('div');
  holder.appendChild(svg);
  document.body.appendChild(holder);
  const dag = createDag(svg);
  dag.update(makeSnapshot({
    manufactureRequired: { [CRAFT_ID]: 10 },
    produced: { [CRAFT_ID]: 3 },
  }));
  const group = dagNode(svg, CRAFT_ID);
  const badge = badgeOf(group);
  const box = byClass(group, 'node-box')[0];
  const badgeFill = getComputedStyle(badge).fill;
  const boxFill = getComputedStyle(box).fill;
  assert(badgeFill && badgeFill !== 'none' && badgeFill !== 'transparent'
    && badgeFill !== 'rgb(0, 0, 0)',
  `badge fill must be visible (not black/transparent), got ${badgeFill}`);
  assert(badgeFill !== boxFill,
    `badge fill (${badgeFill}) must differ from node-box fill (${boxFill})`);

  const normalStroke = getComputedStyle(box).stroke;
  dag.update(makeSnapshot({
    manufactureRequired: { [CRAFT_ID]: 10 },
    produced: { [CRAFT_ID]: 10 },
  }));
  assert(hasClass(group, 'is-complete'), 'fixture must reach COMPLETE');
  const completeStroke = getComputedStyle(box).stroke;
  const completeBadgeFill = getComputedStyle(badge).fill;
  assert(completeStroke !== normalStroke || completeBadgeFill !== badgeFill,
    'COMPLETE must change a computed style (stroke or badge fill) vs non-complete');
});

/* ------------------ editable initial-state panel (TASK B) ------------------ */

function initialConfig(opts = {}) {
  const targets = Object.create(null);
  const initialStash = Object.create(null);
  for (const id of KNOWN_TARGET_IDS) targets[id] = opts.target === undefined ? 7 : opts.target;
  for (const id of KNOWN_CRAFTABLE_IDS) initialStash[id] = opts.stash === undefined ? 3 : opts.stash;
  return {
    items: [],
    targets,
    initialCoins: opts.coins === undefined ? 12345 : opts.coins,
    initialStash,
    starterMode: 'PREDELIVERED_RESERVED'
  };
}

function initialDoc(values = {}) {
  const nodes = new Map();
  const mk = (id) => {
    if (!nodes.has(id)) {
      const node = el(id === INITIAL_APPLY_ID || id === INITIAL_RESET_ID || id === INITIAL_ZERO_ID
        ? 'button' : 'input');
      node.id = id;
      node.value = Object.prototype.hasOwnProperty.call(values, id) ? String(values[id]) : '';
      nodes.set(id, node);
    }
    return nodes.get(id);
  };
  const known = (id) => id === INITIAL_COINS_ID || id === INITIAL_APPLY_ID || id === INITIAL_RESET_ID
    || id === INITIAL_ZERO_ID
    || id.indexOf('initial-target-') === 0 || id.indexOf('initial-stash-') === 0;
  return { doc: { getElementById: (id) => (known(id) ? mk(id) : null), body: el('body') }, nodes, mk };
}

test('initial-state panel prefills 1 coins + 11 targets + 19 stash rows from the injected config', () => {
  const cfg = initialConfig();
  const { doc, nodes } = initialDoc();
  assert(!!createInitialStatePanel(cfg, {}, doc), 'panel must be created');
  assertEqual(nodes.get(INITIAL_COINS_ID).value, '12345', 'coins prefill');
  for (const id of KNOWN_TARGET_IDS) {
    assertEqual(nodes.get(initialTargetId(id)).value, '7', `${id} target prefill`);
  }
  for (const id of KNOWN_CRAFTABLE_IDS) {
    assertEqual(nodes.get(initialStashId(id)).value, '3', `${id} stash prefill`);
  }
  // Pure over the snapshot: mutating the source must not silently rewrite inputs.
  cfg.initialCoins = 999;
  assertEqual(nodes.get(INITIAL_COINS_ID).value, '12345', 'no hidden live binding to config');
});

test('initial-state panel re-prefill after a preset change replaces all edited values', () => {
  const { doc, nodes } = initialDoc();
  const panel = createInitialStatePanel(initialConfig(), {}, doc);
  nodes.get(INITIAL_COINS_ID).value = '777';
  nodes.get(initialTargetId('orb')).value = '99';
  nodes.get(initialStashId('wood')).value = '88';
  panel.prefill(initialConfig({ coins: 0, target: 0, stash: 0 }));
  assertEqual(nodes.get(INITIAL_COINS_ID).value, '0');
  assertEqual(nodes.get(initialTargetId('orb')).value, '0');
  assertEqual(nodes.get(initialStashId('wood')).value, '0');
});

test('readInitialState returns null-prototype allowlisted maps carrying the edited values', () => {
  const cfg = initialConfig();
  const { doc, nodes } = initialDoc();
  createInitialStatePanel(cfg, {}, doc);
  nodes.get(INITIAL_COINS_ID).value = '42';
  nodes.get(initialTargetId('orb')).value = '77';
  nodes.get(initialStashId('wood')).value = '9';
  const codes = [];
  const out = readInitialState(cfg, doc, (code) => codes.push(code));
  assertEqual(out.initialCoins, 42);
  assertEqual(Object.getPrototypeOf(out.targets), null, 'targets must be null-prototype');
  assertEqual(Object.getPrototypeOf(out.initialStash), null, 'stash must be null-prototype');
  assertEqual(Object.keys(out.targets).length, KNOWN_TARGET_IDS.length);
  assertEqual(Object.keys(out.initialStash).length, KNOWN_CRAFTABLE_IDS.length);
  assertEqual(out.targets.orb, 77);
  assertEqual(out.initialStash.wood, 9);
  assertEqual(codes.length, 0, 'valid input must not log the invalid-input code');
});

test('readInitialState rejects empty/fractional/negative/over-cap/junk input, falls back to prefill and logs only CONFIG_INPUT_INVALID', () => {
  const cfg = initialConfig();
  const { doc, nodes } = initialDoc();
  createInitialStatePanel(cfg, {}, doc);
  nodes.get(INITIAL_COINS_ID).value = '   ';
  nodes.get(initialTargetId('orb')).value = '1.5';
  nodes.get(initialTargetId('silver')).value = '10000001';
  nodes.get(initialStashId('wood')).value = '-4';
  nodes.get(initialStashId('orb')).value = '1000000001';
  nodes.get(initialStashId('needles')).value = '12abc';
  const codes = [];
  const out = readInitialState(cfg, doc, (code) => codes.push(code));
  assertEqual(out.initialCoins, 12345, 'whitespace coins falls back');
  assertEqual(out.targets.orb, 7, 'fractional target falls back');
  assertEqual(out.targets.silver, 7, 'over-cap target falls back');
  assertEqual(out.initialStash.wood, 3, 'negative stash falls back');
  assertEqual(out.initialStash.orb, 3, 'over-cap stash falls back');
  assertEqual(out.initialStash.needles, 3, 'junk stash falls back');
  assertDeepEqual(codes, ['CONFIG_INPUT_INVALID'], 'one stable code, no raw input');
});

test('readInitialState never creates an element or handler from hostile input', () => {
  const cfg = initialConfig();
  const { doc, nodes } = initialDoc();
  createInitialStatePanel(cfg, {}, doc);
  const before = nodes.size;
  const payload = '<img src=x onerror=alert(1)>';
  nodes.get(INITIAL_COINS_ID).value = payload;
  nodes.get(initialTargetId('orb')).value = payload;
  nodes.get(initialStashId('wood')).value = payload;
  const out = readInitialState(cfg, doc, () => {});
  assertEqual(nodes.size, before, 'no extra DOM nodes from payloads');
  for (const node of nodes.values()) {
    assert(!(typeof node.getAttribute === 'function' && node.getAttribute('onerror')),
      'no handler attribute may be created');
  }
  assertEqual(out.initialCoins, 12345);
  assertEqual(out.targets.orb, 7);
  assertEqual(out.initialStash.wood, 3);
});

test('initial-state panel wires Apply and Reset with exactly one listener each', () => {
  const { doc, nodes } = initialDoc();
  let applied = null;
  let applyCalls = 0;
  let resets = 0;
  createInitialStatePanel(initialConfig(), {
    onApply: (v) => { applied = v; applyCalls += 1; },
    onReset: () => { resets += 1; }
  }, doc);
  const apply = nodes.get(INITIAL_APPLY_ID);
  const reset = nodes.get(INITIAL_RESET_ID);
  fireEvent(apply, 'click');
  assert(applied && applied.initialCoins === 12345, 'Apply passes the read override set');
  assertEqual(applyCalls, 1, 'one Apply click -> one callback (no duplicate listener)');
  fireEvent(reset, 'click');
  assertEqual(resets, 1, 'one Reset click -> one callback');
  fireEvent(reset, 'click');
  assertEqual(resets, 2, 'still one callback per click (no duplicate listener)');
});

test('initial-state Reset to 0 fills only coins + 19 stash inputs, leaves targets and Apply untouched', () => {
  const { doc, nodes } = initialDoc();
  let applied = null;
  let applyCalls = 0;
  let resets = 0;
  const codes = [];
  createInitialStatePanel(initialConfig(), {
    onApply: (v) => { applied = v; applyCalls += 1; },
    onReset: () => { resets += 1; },
    onInvalid: (code) => codes.push(code)
  }, doc);

  // Pre-existing edits: targets must survive, coins/stash must be overwritten.
  nodes.get(initialTargetId('orb')).value = '99';
  nodes.get(initialTargetId('silver')).value = '23';
  nodes.get(initialStashId('wood')).value = '88';
  nodes.get(INITIAL_COINS_ID).value = '777';

  fireEvent(nodes.get(INITIAL_ZERO_ID), 'click');

  assertEqual(nodes.get(INITIAL_COINS_ID).value, '0', 'coins input zeroed');
  for (const id of KNOWN_CRAFTABLE_IDS) {
    assertEqual(nodes.get(initialStashId(id)).value, '0', `${id} stash input zeroed`);
  }
  for (const id of KNOWN_TARGET_IDS) {
    const expected = id === 'orb' ? '99' : id === 'silver' ? '23' : '7';
    assertEqual(nodes.get(initialTargetId(id)).value, expected, `${id} target preserved`);
  }

  assertEqual(applyCalls, 0, 'Reset to 0 must NOT invoke Apply/start');
  assertEqual(applied, null, 'no override set was read or applied');
  assertEqual(resets, 0, 'Reset to 0 must not call the preset Reset path');
  assertDeepEqual(codes, [], 'no validation/logging path is triggered');

  fireEvent(nodes.get(INITIAL_ZERO_ID), 'click');
  assertEqual(applyCalls, 0, 'still no Apply after repeated zero clicks');

  // Apply remains the sole run trigger and reads the fill-only zeroed form.
  fireEvent(nodes.get(INITIAL_APPLY_ID), 'click');
  assertEqual(applyCalls, 1, 'Apply is the sole run trigger');
  assert(applied, 'Apply callback received the override set');
  assertEqual(applied.initialCoins, 0, 'Apply reads the zeroed coins');
  for (const id of KNOWN_CRAFTABLE_IDS) {
    assertEqual(applied.initialStash[id], 0, `Apply reads the zeroed ${id} stash`);
  }
  assertEqual(applied.targets.orb, 99, 'Apply keeps current target values');
  assertEqual(applied.targets.silver, 23, 'Apply keeps current target values');
});

test('controls installs exactly one #preset change listener that invokes onPresetChange', () => {
  const ids = ['btn-start', 'btn-pause', 'btn-reset', 'speed', 'seed', 'preset', 'pop-size',
    'generations', 'log-filter', 'log-clear'];
  const nodes = new Map();
  const mk = (id) => {
    if (!nodes.has(id)) {
      const node = el(id.startsWith('btn') || id === 'log-clear' ? 'button' : 'input');
      node.id = id;
      node.value = '';
      nodes.set(id, node);
    }
    return nodes.get(id);
  };
  const original = globalThis.document.getElementById;
  globalThis.document.getElementById = (id) => (ids.includes(id) ? mk(id) : null);
  try {
    let calls = 0;
    let received = null;
    createControls({ onPresetChange: (cfg) => { calls += 1; received = cfg; } });
    const preset = nodes.get('preset');
    preset.value = 'empty';
    fireEvent(preset, 'change');
    assertEqual(calls, 1, 'one change -> one onPresetChange call (no duplicate listener)');
    assertEqual(received.preset, 'empty', 'callback receives the validated control set');
    fireEvent(preset, 'change');
    assertEqual(calls, 2, 'still one callback per change');
  } finally {
    globalThis.document.getElementById = original;
  }
});

test('index.html fresh load selects empty preset with 0 coins/stash and YAML targets', async () => {
  let source;
  if (typeof window !== 'undefined' && typeof fetch === 'function') {
    const res = await fetch('./index.html?fresh=' + Date.now());
    assert(res.ok, 'fetch failed for index.html');
    source = await res.text();
  } else {
    const { readFileSync } = await import('node:fs');
    source = readFileSync(new URL('../../index.html', import.meta.url), 'utf8');
  }
  const preset = source.match(/<select id="preset"[^>]*>([\s\S]*?)<\/select>/);
  assert(preset, 'preset select must exist');
  assert(/<option value="empty"[^>]*\bselected\b/.test(preset[1]), 'empty must be the selected preset');
  assert(!/<option value="reuse"[^>]*\bselected\b/.test(preset[1]), 'reuse must not be selected');

  const valueOf = (id) => {
    const tag = source.match(new RegExp('<[a-zA-Z][^>]*\\bid="' + id + '"[^>]*>', 'i'));
    assert(tag, `missing #${id}`);
    const m = tag[0].match(/\bvalue="([^"]*)"/);
    assert(m, `#${id} has no value attribute`);
    return m[1];
  };
  assertEqual(valueOf(INITIAL_COINS_ID), '0', 'fresh coins = 0');
  for (const id of KNOWN_CRAFTABLE_IDS) {
    assertEqual(valueOf(initialStashId(id)), '0', `fresh ${id} stash = 0`);
  }
  for (const id of KNOWN_TARGET_IDS) {
    assertEqual(valueOf(initialTargetId(id)), String(EMBEDDED_RAW.targets[id]),
      `fresh ${id} target comes from targets.yaml`);
  }
});

test('index.html Initial state panel has 34 ids with tooltips/ranges, no inline handlers, CSP/SRI intact', async () => {
  let source;
  if (typeof window !== 'undefined' && typeof fetch === 'function') {
    const res = await fetch('./index.html?v=' + Date.now());
    assert(res.ok, 'fetch failed for index.html');
    source = await res.text();
  } else {
    const { readFileSync } = await import('node:fs');
    source = readFileSync(new URL('../../index.html', import.meta.url), 'utf8');
  }
  const start = source.indexOf('id="initial-state-panel"');
  assert(start >= 0, 'root index.html must contain the Initial state panel');
  const end = source.indexOf('</section>', start);
  assert(end > start, 'panel section must be closed');
  const panel = source.slice(start, end);

  const ids = [INITIAL_COINS_ID, INITIAL_APPLY_ID, INITIAL_RESET_ID, INITIAL_ZERO_ID,
    ...KNOWN_TARGET_IDS.map(initialTargetId), ...KNOWN_CRAFTABLE_IDS.map(initialStashId)];
  assertEqual(ids.length, 34, '1 coins + 11 targets + 19 stash + 3 buttons');
  for (const id of ids) {
    assert(panel.includes('id="' + id + '"'), `panel missing #${id}`);
    const tag = panel.match(new RegExp('<[a-zA-Z][^>]*\\bid="' + id + '"[^>]*>', 'i'));
    assert(tag && /\btitle="[^"]+"/.test(tag[0]), `#${id} must carry a title tooltip`);
  }
  const coinsTag = panel.match(new RegExp('<[a-zA-Z][^>]*\\bid="' + INITIAL_COINS_ID + '"[^>]*>', 'i'))[0];
  assert(coinsTag.includes('0..1000000000'), 'coins tooltip states the 0..1e9 range');
  const targetTag = panel.match(new RegExp('<[a-zA-Z][^>]*\\bid="' + initialTargetId('orb') + '"[^>]*>', 'i'))[0];
  assert(targetTag.includes('0..10000000'), 'target tooltip states the 0..1e7 range');
  const stashTag = panel.match(new RegExp('<[a-zA-Z][^>]*\\bid="' + initialStashId('wood') + '"[^>]*>', 'i'))[0];
  assert(stashTag.includes('0..1000000000'), 'stash tooltip states the 0..1e9 range');
  assert(!/\bon(click|change|input|submit)\s*=/i.test(panel), 'no inline handlers in the panel');

  const csp = '<meta http-equiv="Content-Security-Policy" content="default-src \'self\'; script-src \'self\' https://cdn.jsdelivr.net https://esm.sh; style-src \'self\' \'unsafe-inline\'; img-src \'self\' data:; connect-src \'self\'; object-src \'none\'; base-uri \'none\'">';
  assert(source.includes(csp), 'CSP meta line must be preserved verbatim');
  assert(source.includes('integrity="sha384-ZeqCzuWczURac3RacSufGD7oSbzeaX7xxnnOr3PTcYTLx4Av0qBj0kBq7AeCtHLA"'),
    'SRI line must be preserved verbatim');
});

/* ---------------------- simulated-time clock (header) ---------------------- */

test('formatSimClock renders tick * 5 as "Sim time: N min"', () => {
  assertEqual(SIM_TICK_MINUTES, 5, 'one engine tick is 5 simulated minutes');
  assertEqual(formatSimClock(0), 'Sim time: 0 min');
  assertEqual(formatSimClock(1), 'Sim time: 5 min');
  assertEqual(formatSimClock(2), 'Sim time: 10 min');
  assertEqual(formatSimClock(7), 'Sim time: 35 min');
  assertEqual(formatSimClock(1.9), 'Sim time: 5 min', 'fractional ticks floor');
  assertEqual(formatSimClock(-5), 'Sim time: 0 min', 'negative clamps to zero');
  assertEqual(formatSimClock(NaN), 'Sim time: 0 min', 'non-finite clamps to zero');
  assertEqual(formatSimClock(Infinity), 'Sim time: 0 min', 'infinite clamps to zero');
});

test('sim clock: injected tick snapshots drive updates, freeze and reset to 0', () => {
  const host = { textContent: '', attrs: {}, setAttribute(k, v) { this.attrs[k] = String(v); } };
  const doc = { getElementById: (id) => (id === SIM_CLOCK_ID ? host : null) };
  const clock = createSimClock({}, doc);

  assertEqual(host.textContent, 'Sim time: 0 min', 'host present with initial sim-time format');
  assertEqual(clock.getMinutes(), 0, 'initial value is 0 min');
  assertEqual(host.attrs['aria-label'], 'Simulated time: 0 minutes', 'aria describes simulated time');

  // Drive injected snapshot ticks directly: no Date.now, no setInterval, no waiting.
  for (const tick of [1, 2, 5, 12]) {
    clock.update(tick);
    assertEqual(clock.getMinutes(), tick * 5, `tick ${tick} -> ${tick * 5} minutes`);
    assertEqual(host.textContent, `Sim time: ${tick * 5} min`, `tick ${tick} rendered`);
  }
  assertEqual(host.attrs['aria-label'], 'Simulated time: 60 minutes', 'aria tracks the live value');

  clock.freeze(12);
  assertEqual(clock.isFrozen(), true, 'freeze marks the clock frozen');
  clock.update(999); // stale post-completion snapshot must not change the display
  assertEqual(clock.getMinutes(), 60, 'frozen value survives later tick updates');
  assertEqual(host.textContent, 'Sim time: 60 min', 'frozen text unchanged');

  clock.reset();
  assertEqual(clock.getMinutes(), 0, 'reset returns to zero');
  assertEqual(clock.isFrozen(), false, 'reset unfreezes');
  assertEqual(host.textContent, 'Sim time: 0 min', 'reset redraws zero');
});

test('sim clock: completion freeze pins result.completionTick and hostile ticks are capped', () => {
  const host = { textContent: '' };
  const doc = { getElementById: (id) => (id === SIM_CLOCK_ID ? host : null) };
  const clock = createSimClock({}, doc);

  clock.update(4);
  clock.freeze(9); // final tick from result().completionTick
  assertEqual(clock.getMinutes(), 45, 'freeze(tick) pins the authoritative final tick');
  assertEqual(clock.isFrozen(), true);

  clock.reset();
  clock.update('<img src=x onerror=alert(1)>');
  assertEqual(clock.getMinutes(), 0, 'payload tick normalizes to 0');
  assert(!/img|onerror/i.test(host.textContent), 'no markup from a hostile tick');
  assert(host.textContent.length <= 120, 'text stays within the 120 cap');
});

test('index.html header hosts a simulated clock (id wall-clock) and no real-time clock code', async () => {
  async function read(relative, browserUrl) {
    if (typeof window !== 'undefined' && typeof fetch === 'function') {
      const res = await fetch(browserUrl);
      assert(res.ok, 'fetch failed for ' + browserUrl);
      return await res.text();
    }
    const { readFileSync } = await import('node:fs');
    return readFileSync(new URL(relative, import.meta.url), 'utf8');
  }
  const source = await read('../../index.html', './index.html?simclock=' + Date.now());
  const tag = source.match(/<[^>]*\bid="wall-clock"[^>]*>/);
  assert(tag, 'header must keep the #wall-clock element id');
  assert(/Sim time: 0 min/.test(source), 'initial text must use the Sim time: N min format');
  assert(/aria-label="[^"]*[Ss]imulated/.test(tag[0]), 'clock aria must describe simulated time');
  assert(!/Wall clock/.test(source), 'no real wall-clock label may remain');
  assert(!/\bon(click|change|input|submit)\s*=/i.test(tag[0]), 'no inline handlers on the clock');

  const main = await read('../main.js', './src/main.js?simclock=' + Date.now());
  const stripped = main.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');
  assert(!/formatWallClock|createWallClock|WALL_CLOCK_INTERVAL_MS/.test(stripped),
    'real wall-clock controller must be removed');
  assert(!/setInterval|Date\.now/.test(stripped), 'clock must not use timers or wall time');
  assert(source.includes('<meta http-equiv="Content-Security-Policy" content="default-src \'self\'; script-src \'self\' https://cdn.jsdelivr.net https://esm.sh; style-src \'self\' \'unsafe-inline\'; img-src \'self\' data:; connect-src \'self\'; object-src \'none\'; base-uri \'none\'">'),
    'CSP line must be preserved verbatim');
  assert(source.includes('integrity="sha384-ZeqCzuWczURac3RacSufGD7oSbzeaX7xxnnOr3PTcYTLx4Av0qBj0kBq7AeCtHLA"'),
    'SRI line must be preserved verbatim');
});

test('initial-state.js source uses no markup-injection API and no inline handlers', async () => {
  let source;
  if (typeof window !== 'undefined' && typeof fetch === 'function') {
    const res = await fetch('./src/ui/initial-state.js');
    assert(res.ok, 'fetch failed for initial-state.js');
    source = await res.text();
  } else {
    const { readFileSync } = await import('node:fs');
    source = readFileSync(new URL('../../src/ui/initial-state.js', import.meta.url), 'utf8');
  }
  const strip = (s) => s.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');
  const code = strip(source);
  assert(!/innerHTML|insertAdjacentHTML|outerHTML/i.test(code), 'no markup-injection API');
  assert(!/\bon(click|change|input)\s*=/i.test(code), 'no inline handlers');
});
