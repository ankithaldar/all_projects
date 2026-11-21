// Results tab: best schedule, totals, target status, feasibility verdict and
// a bounded per-genome telemetry view with per-item breakdown tables.
// Tables are built with createElement + textContent only — never innerHTML.
import { nameOf } from './item-registry.js';
import { capString, tickToDay } from './format.js';

const TELEMETRY_CAP = 20;

function h(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text != null) el.textContent = capString(text);
  return el;
}

function makeTable(headers, rows) {
  const table = h('table', 'results-table');
  const thead = h('thead');
  const hr = h('tr');
  for (const head of headers) hr.appendChild(h('th', null, head));
  thead.appendChild(hr);
  table.appendChild(thead);
  const tbody = h('tbody');
  for (const cells of rows) {
    const tr = h('tr', cells.className || '');
    for (const cell of cells.values) tr.appendChild(h('td', null, cell));
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  return table;
}

export function createResults(el) {
  let lastResult = null;
  const telemetry = []; // last TELEMETRY_CAP genome results only

  function renderResult(result) {
    if (!result || typeof result !== 'object') {
      el.appendChild(h('p', 'results-empty', 'No result available yet.'));
      return;
    }
    el.appendChild(h('h2', 'results-title', 'Best schedule'));
    const schedule = Array.isArray(result.schedule) ? result.schedule : [];
    el.appendChild(makeTable(
      ['Tick', 'Item', 'Batch'],
      schedule.map((s) => ({ values: [s.tick, nameOf(s.item), s.batch] })),
    ));

    const tick = Number.isFinite(result.completionTick) ? result.completionTick : result.elapsedTicks;
    el.appendChild(h('h2', 'results-title', 'Totals'));
    const totals = h('dl', 'results-totals');
    for (const [label, val] of [
      ['Total spent', result.totalSpent],
      ['Completion tick', tick],
      ['Completion day', tickToDay(tick)],
      ['Elapsed ticks', result.elapsedTicks],
    ]) {
      totals.appendChild(h('dt', null, label));
      totals.appendChild(h('dd', null, val == null ? '—' : val));
    }
    el.appendChild(totals);

    const shortfall = result.shortfall || {};
    const targetIds = Object.keys(shortfall);
    el.appendChild(h('h2', 'results-title', 'Target status'));
    if (targetIds.length === 0) {
      el.appendChild(h('p', 'results-note', 'All targets met.'));
    } else {
      el.appendChild(makeTable(
        ['Item', 'Shortfall'],
        targetIds.map((id) => ({ values: [nameOf(id), shortfall[id]] })),
      ));
    }
    el.appendChild(h('h2',
      result.feasible ? 'verdict verdict-ok' : 'verdict verdict-fail',
      result.feasible ? 'FEASIBLE' : 'INFEASIBLE'));
  }

  function renderTelemetry(event) {
    el.appendChild(h('h2', 'results-title',
      `Genome telemetry — ${nameOf(event.itemId)} · gen ${event.generation} · #${event.genomeIndex}`));

    const meta = h('dl', 'results-totals');
    for (const [label, val] of [
      ['Fitness', event.fitness],
      ['Total spent', event.totalSpent],
      ['Completion tick', event.completionTick],
      ['Elapsed ticks', event.elapsedTicks],
      ['Verdict', event.feasible ? 'FEASIBLE' : 'INFEASIBLE'],
    ]) {
      meta.appendChild(h('dt', null, label));
      meta.appendChild(h('dd', null, val == null ? '—' : val));
    }
    el.appendChild(meta);

    const perItem = (event.perItem && typeof event.perItem === 'object') ? event.perItem : {};
    const ids = Object.keys(perItem).slice(0, 23);
    const rows = ids.map((id) => {
      const p = perItem[id] || {};
      return { values: [nameOf(id), p.coinsSpent, p.timeMinutes, p.busyTicks, p.batches, p.units] };
    });
    const sum = (key) => ids.reduce((acc, id) => acc + (Number((perItem[id] || {})[key]) || 0), 0);
    rows.push({
      className: 'results-total-row',
      values: ['TOTAL', sum('coinsSpent'), sum('timeMinutes'),
        sum('busyTicks'), sum('batches'), sum('units')],
    });
    el.appendChild(h('h3', 'results-subtitle', 'PER-ITEM BREAKDOWN'));
    el.appendChild(makeTable(
      ['Item', 'Coins', 'Manufacturing min', 'Busy ticks', 'Batches', 'Units'],
      rows,
    ));
  }

  // Both entry points clear the DOM before rendering, then rebuild from the
  // bounded in-memory state — no unbounded append.
  function render() {
    el.textContent = '';
    if (lastResult) renderResult(lastResult);
    for (const event of telemetry) renderTelemetry(event);
  }

  function show(result) {
    lastResult = (result && typeof result === 'object') ? result : null;
    render();
  }

  function showGenomeTelemetry(event) {
    if (!event || typeof event !== 'object') return;
    telemetry.push(event);
    while (telemetry.length > TELEMETRY_CAP) telemetry.shift();
    render();
  }

  return { show, showGenomeTelemetry };
}
