// Live left-to-right SVG DAG. Simple directed edges (ingredient -> product),
// live status badges, hover tooltips and recursive click-to-trace ancestry.
// Layout and recipes are derived entirely from the item registry.
import { ITEMS, DAG_EDGES, craftTicks, effectiveTier } from './item-registry.js';
import { SVG_NS, capString, MAX_STR, formatCountdown, bottleneckKind } from './format.js';

const NODE_W = 152;
const NODE_H = 56;
const COL_W = 224;
const ROW_H = 86;
const PAD_X = 40;
const PAD_Y = 40;
const RING_R = 7;
const RING_C = 2 * Math.PI * RING_R;

function svgEl(tag, attrs) {
  const el = document.createElementNS(SVG_NS, tag);
  if (attrs) for (const k in attrs) el.setAttribute(k, String(attrs[k]));
  return el;
}

/**
 * Normalize a manufactured-count value exactly as the cap contract requires:
 * only finite numbers are trusted, clamped to a non-negative integer. Strings
 * (including XSS payloads), NaN and Infinity all collapse to 0 and are never
 * treated as markup.
 */
function normalizeCount(v) {
  return Number.isFinite(v) ? Math.max(0, Math.floor(v)) : 0;
}

/**
 * Read a snapshot-map value only from an OWN key. The DAG's keys come from the
 * fixed ITEMS registry, never from Object.keys(snapshot maps), so inherited /
 * `__proto__` / `constructor` entries are never consulted.
 */
function ownValue(map, id) {
  return (map && typeof map === 'object' && Object.prototype.hasOwnProperty.call(map, id))
    ? map[id] : undefined;
}

/** capString(...)-wrapped normalized count (never raw untrusted input). */
function displayCount(v) {
  return capString(normalizeCount(v), MAX_STR);
}

export function createDag(svgElRoot) {
  const byId = new Map(ITEMS.map((i) => [i.id, i]));
  const ingredientsOf = new Map(ITEMS.map((i) => [i.id, (i.ingredients || []).map((e) => e.item)]));

  // --- dependency depth -> columns (NOT the inconsistent YAML tier label) --
  const tiers = new Map();
  for (const it of ITEMS) {
    const d = Math.max(0, effectiveTier(it.id));
    if (!tiers.has(d)) tiers.set(d, []);
    tiers.get(d).push(it);
  }
  const tierKeys = Array.from(tiers.keys()).sort((a, b) => a - b);
  const maxRows = Math.max(1, ...Array.from(tiers.values(), (a) => a.length));
  const width = PAD_X * 2 + tierKeys.length * COL_W;
  const height = PAD_Y * 2 + maxRows * ROW_H + 12;
  svgElRoot.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svgElRoot.setAttribute('preserveAspectRatio', 'xMinYMin meet');

  const defs = svgEl('defs');
  const marker = svgEl('marker', {
    id: 'dag-arrow', viewBox: '0 0 10 10', refX: '9', refY: '5',
    markerWidth: '6', markerHeight: '6', orient: 'auto-start-reverse',
  });
  marker.appendChild(svgEl('path', { d: 'M0,0 L10,5 L0,10 z', class: 'dag-arrow-head' }));
  defs.appendChild(marker);
  svgElRoot.appendChild(defs);

  const edgeLayer = svgEl('g', { class: 'dag-edge-layer' });
  const nodeLayer = svgEl('g', { class: 'dag-node-layer' });
  svgElRoot.appendChild(edgeLayer);
  svgElRoot.appendChild(nodeLayer);

  // --- nodes ---------------------------------------------------------------
  const nodes = new Map(); // id -> {item, refs, group}
  tierKeys.forEach((tier, col) => {
    const group = tiers.get(tier);
    group.forEach((it, row) => {
      const x = PAD_X + col * COL_W;
      const y = PAD_Y + row * ROW_H + (maxRows - group.length) * ROW_H / 2;
      const g = svgEl('g', {
        class: 'dag-node',
        'data-item-id': capString(it.id),
        'data-tier': String(it.tier),
        'data-depth': String(tier),
        'data-x': String(x),
        'data-y': String(y),
      });
      g.appendChild(svgEl('rect', {
        class: 'node-box', x, y, width: NODE_W, height: NODE_H, rx: '9',
      }));
      const ring = svgEl('circle', {
        class: 'node-progress', cx: x + 16, cy: y + NODE_H - 14,
        r: String(RING_R), fill: 'none', 'stroke-width': '3',
        'stroke-dasharray': String(RING_C), 'stroke-dashoffset': String(RING_C),
      });
      g.appendChild(ring);
      const name = svgEl('text', { class: 'node-name', x: x + 11, y: y + 22 });
      name.textContent = capString(it.name || it.id);
      g.appendChild(name);
      const stash = svgEl('text', {
        class: 'node-stash', x: x + NODE_W - 11, y: y + 22, 'text-anchor': 'end',
      });
      stash.textContent = '0';
      g.appendChild(stash);
      // Manufactured-count badge: its own SVG <text>, written via textContent
      // only. Craftables show `produced/manufactureRequired`; bases show `n/a`.
      const badge = svgEl('text', {
        class: 'node-badge', x: x + NODE_W / 2, y: y + NODE_H / 2 + 6,
        'text-anchor': 'middle',
      });
      badge.textContent = 'n/a';
      g.appendChild(badge);
      const timer = svgEl('text', { class: 'node-timer', x: x + 30, y: y + NODE_H - 10 });
      timer.textContent = '';
      g.appendChild(timer);
      const batch = svgEl('text', {
        class: 'node-batch', x: x + NODE_W - 11, y: y + NODE_H - 10, 'text-anchor': 'end',
      });
      batch.textContent = '';
      g.appendChild(batch);
      const tip = svgEl('title');
      tip.textContent = capString(
        `${it.name || it.id} · raw tier ${it.tier} · depth ${tier}`);
      g.appendChild(tip);
      nodeLayer.appendChild(g);

      const refs = { g, stash, timer, batch, ring, tip, badge };
      nodes.set(it.id, { item: it, refs, group: g, craftable: it.type === 'craftable' });

      g.addEventListener('click', (ev) => {
        if (ev && ev.stopPropagation) ev.stopPropagation();
        traced = (traced === it.id) ? null : it.id;
        applyTrace();
      });
    });
  });

  // --- edges (simple straight ingredient -> product lines) -----------------
  const rawEdges = (Array.isArray(DAG_EDGES) && DAG_EDGES.length)
    ? DAG_EDGES
    : ITEMS.flatMap((i) => (i.ingredients || [])
      .map((e) => ({ from: e.item, to: i.id, quantity: e.quantity })));
  const edgeRefs = [];
  for (const def of rawEdges) {
    const sb = nodes.get(def.from);
    const tb = nodes.get(def.to);
    if (!sb || !tb) continue;
    const sx = Number(sb.group.getAttribute('data-x')) + NODE_W;
    const sy = Number(sb.group.getAttribute('data-y')) + NODE_H / 2;
    const tx = Number(tb.group.getAttribute('data-x'));
    const ty = Number(tb.group.getAttribute('data-y')) + NODE_H / 2;
    const line = svgEl('line', {
      class: 'dag-edge',
      x1: sx, y1: sy, x2: tx, y2: ty,
      'marker-end': 'url(#dag-arrow)',
      'data-from': capString(def.from),
      'data-to': capString(def.to),
    });
    const edgeTip = svgEl('title');
    const qty = Number(def.quantity);
    edgeTip.textContent = capString(
      `${(byId.get(def.from) || {}).name || def.from} → ${(byId.get(def.to) || {}).name || def.to} · qty ${Number.isFinite(qty) ? qty : 1}`);
    line.appendChild(edgeTip);
    edgeLayer.appendChild(line);
    edgeRefs.push({ def, line });
    line.addEventListener('click', (ev) => {
      if (ev && ev.stopPropagation) ev.stopPropagation();
    });
  }

  // --- click-to-trace toast -------------------------------------------------
  const toast = document.createElement('div');
  toast.className = 'trace-toast';
  toast.setAttribute('role', 'status');
  toast.hidden = true;
  const host = (svgElRoot.parentNode && svgElRoot.parentNode.appendChild)
    ? svgElRoot.parentNode : document.body;
  host.appendChild(toast);

  let traced = null;

  function ancestry(rootId) {
    const foundNodes = new Set();
    const foundEdges = new Set();
    const order = [];
    const visit = (id) => {
      if (foundNodes.has(id)) return;
      foundNodes.add(id);
      order.push(id);
      for (const ing of (ingredientsOf.get(id) || [])) {
        if (!byId.has(ing)) continue;
        foundEdges.add(`${ing}>${id}`);
        visit(ing);
      }
    };
    visit(rootId);
    return { foundNodes, foundEdges, order };
  }

  function applyTrace() {
    const active = traced ? ancestry(traced) : null;
    for (const [id, n] of nodes) {
      n.refs.g.classList.toggle('trace-dim', !!active && !active.foundNodes.has(id));
      n.refs.g.classList.toggle('trace-active', !!active && active.foundNodes.has(id));
    }
    for (const { def, line } of edgeRefs) {
      const key = `${def.from}>${def.to}`;
      const on = !!active && active.foundEdges.has(key);
      line.classList.toggle('trace-dim', !!active && !on);
      line.classList.toggle('trace-active', on);
    }
    if (!active) {
      toast.hidden = true;
      toast.textContent = '';
    } else {
      const chain = active.order.map((id) => {
        const it = byId.get(id) || {};
        return `${capString(it.name || id, 12)}(d${effectiveTier(id)})`;
      });
      const root = byId.get(traced) || {};
      toast.textContent = capString(
        `Trace ${capString(root.name || traced, 14)} tier ${root.tier} depth ${effectiveTier(traced)} · ${chain.length} nodes: ${chain.join(' <- ')}`);
      toast.hidden = false;
    }
  }

  svgElRoot.addEventListener('click', () => {
    if (traced) { traced = null; applyTrace(); }
  });

  // --- live update ----------------------------------------------------------
  function update(snapshot) {
    if (!snapshot) return;
    const stash = snapshot.stash || {};
    const slots = snapshot.slots || {};
    const starvation = snapshot.starvation || {};
    const status = snapshot.status || {};
    // Engine-local cap maps: craftable ids only, base ids absent. May be
    // missing/invalid — ownValue()+normalizeCount() degrade to 0 safely.
    const manufactureRequired = snapshot.manufactureRequired;
    const producedMap = snapshot.produced;
    const committedMap = snapshot.committed;

    for (const [id, n] of nodes) {
      const count = Number.isFinite(stash[id]) ? Math.max(0, Math.floor(stash[id])) : 0;
      n.refs.stash.textContent = String(count);
      const slot = slots[id] || { state: 'IDLE', remainingTicks: 0, batchSize: 0 };
      const active = slot.state === 'ACTIVE' && Number(slot.remainingTicks) > 0;
      n.refs.g.classList.toggle('is-busy', active);
      n.refs.g.classList.toggle('is-idle', !active);
      if (active) {
        n.refs.timer.textContent = formatCountdown(slot.remainingTicks);
        const b = Number(slot.batchSize) || 0;
        n.refs.batch.textContent = b > 0 ? `×${b}` : '';
        const total = Math.max(1, craftTicks(id) || 1);
        const frac = Math.max(0, Math.min(1, Number(slot.remainingTicks) / total));
        n.refs.ring.setAttribute('stroke-dashoffset', String(RING_C * (1 - frac)));
      } else {
        n.refs.timer.textContent = '';
        n.refs.batch.textContent = '';
        n.refs.ring.setAttribute('stroke-dashoffset', String(RING_C));
      }
      const kind = bottleneckKind(slot, starvation[id]);
      n.refs.g.classList.toggle('starve-material', kind === 'material' || kind === 'both');
      n.refs.g.classList.toggle('starve-coin', kind === 'coin' || kind === 'both');
      n.refs.g.classList.toggle('starve-split', kind === 'both');

      // --- manufactured-count badge + COMPLETE state -----------------------
      // Read only OWN keys for this registry id; never iterate the maps and
      // never consult inherited/`__proto__`/`constructor` entries.
      const rawMfr = ownValue(manufactureRequired, id);
      const rawProd = ownValue(producedMap, id);
      const rawCmt = ownValue(committedMap, id);
      let complete = false;
      if (n.craftable) {
        const mfr = normalizeCount(rawMfr);
        const prod = normalizeCount(rawProd);
        // Own SVG <text>, textContent only, each number normalized + capped.
        n.refs.badge.textContent = capString(`${prod}/${mfr}`, MAX_STR);
        // COMPLETE needs both finite values; it outranks stash-based target-met.
        complete = Number.isFinite(rawMfr) && Number.isFinite(rawProd) && prod >= mfr;
      } else {
        // Base items are never manufactured: no ratio, never COMPLETE.
        n.refs.badge.textContent = 'n/a';
      }
      n.refs.g.classList.toggle('is-complete', complete);
      n.refs.g.classList.toggle('complete', complete);
      const targetMet = !!status[id];
      n.refs.g.classList.toggle('target-met', targetMet && !complete);
      n.refs.g.classList.toggle('is-done', !!snapshot.done);

      // Live, full hover tooltip — all dynamic values capped (textContent only).
      const it = n.item;
      const recipe = (it.ingredients || []).length
        ? it.ingredients.map((x) => `${capString(x.quantity)}x ${capString(x.item)}`).join(', ')
        : 'none (base item)';
      // Remaining demand comes ONLY from the snapshot's real `required` map.
      const requiredRaw = ownValue(snapshot.required, id);
      const demandText = Number.isFinite(requiredRaw)
        ? capString(Math.max(0, requiredRaw - count), MAX_STR) : 'n/a';
      const reason = kind === 'both' ? 'material + coin starved'
        : kind === 'material' ? 'material starved'
          : kind === 'coin' ? 'coin starved' : 'none';
      // Five separately labelled counts, each normalized/capped; compact labels
      // keep the whole tooltip within MAX_STR=120 (existing invariant preserved).
      n.refs.tip.textContent = capString(
        `${capString(it.name || id, 14)} tier ${it.tier} depth ${effectiveTier(id)}` +
        ` stash ${capString(count)} demand ${demandText}` +
        ` req ${displayCount(requiredRaw)} mfr ${displayCount(rawMfr)}` +
        ` prod ${displayCount(rawProd)} cmt ${displayCount(rawCmt)}` +
        ` starve ${reason} ${slot.state} ${active ? formatCountdown(slot.remainingTicks) : 'IDLE'}` +
        ` x${Math.max(0, Number(slot.batchSize) || 0)}` +
        ` ${capString(recipe, 30)} ${capString(it.coinCost || 0)}/${capString(it.craftTimeMinutes || 0)}m/${capString(craftTicks(id))}t`,
        MAX_STR);
    }

    // Missing-source edges for material-starved free slots use the same line.
    for (const { def, line } of edgeRefs) {
      const targetSlot = slots[def.to] || { state: 'IDLE' };
      const kind = bottleneckKind(targetSlot, starvation[def.to]);
      const sourceEmpty = (Number(stash[def.from]) || 0) <= 0;
      line.classList.toggle('edge-starved',
        (kind === 'material' || kind === 'both') && sourceEmpty);
    }
  }

  return { update };
}
