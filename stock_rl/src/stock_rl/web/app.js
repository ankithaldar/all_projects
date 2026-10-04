/* stock_rl dashboard.
 *
 * Vanilla JavaScript, inline SVG, no dependencies and no build step. The
 * project's own research is explicit that the design doc's React stack
 * would be roughly four thousand lines to draw one curve and print two
 * tables; this file does both in a fraction of that, with nothing to
 * install and nothing to fetch.
 *
 * Two rules the rest of this file follows:
 *
 * 1. All text goes in through textContent, never innerHTML. These
 *    payloads carry symbol names and free-text reason strings, and a
 *    reason string is exactly the kind of value that must not be parsed
 *    as markup.
 * 2. Every panel renders independently. One endpoint failing shows an
 *    error in that panel and the rest of the page keeps its data. A
 *    dashboard that goes blank when the API hiccups is worse than no
 *    dashboard, because a blank screen looks like "no risk".
 */

'use strict';

const default_api = 'http://127.0.0.1:8765';
const svg_ns = 'http://www.w3.org/2000/svg';
const plot = {wide: 1000, tall: 280, left: 62, right: 16, top: 16,
              bottom: 28};

/* ---------------------------------------------------------------- DOM -- */

function el(tag, text, class_name) {
  const node = document.createElement(tag);
  if (text !== null && text !== undefined) {
    node.textContent = String(text);
  }
  if (class_name) {
    node.className = class_name;
  }
  return node;
}

function svg_el(tag, attributes) {
  const node = document.createElementNS(svg_ns, tag);
  for (const [key, value] of Object.entries(attributes)) {
    node.setAttribute(key, String(value));
  }
  return node;
}

function svg_text(parent, attributes, text) {
  const node = svg_el('text', attributes);
  node.textContent = text;
  parent.appendChild(node);
  return node;
}

function clear(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

function tbody_of(table_id) {
  const table = document.getElementById(table_id);
  const body = table.tBodies[0] || table.createTBody();
  clear(body);
  return body;
}

function row(body, cells, class_name) {
  const tr = el('tr', null, class_name || null);
  for (const [text, class_name_cell] of cells) {
    tr.appendChild(el('td', text, class_name_cell || null));
  }
  body.appendChild(tr);
  return tr;
}

function pair_row(body, label, value) {
  const tr = el('tr');
  tr.appendChild(el('td', label));
  tr.appendChild(el('td', value, 'num'));
  body.appendChild(tr);
  return tr;
}

function unavailable(table_id, reason) {
  const body = tbody_of(table_id);
  const tr = el('tr');
  const td = el('td', reason, 'empty');
  td.colSpan = 8;
  tr.appendChild(td);
  body.appendChild(tr);
}

function blank(host_id, reason) {
  const host = document.getElementById(host_id);
  clear(host);
  host.appendChild(el('p', reason, 'empty'));
}

/* ------------------------------------------------------------ numbers -- */

function is_number(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function num(value, digits) {
  if (!is_number(value)) {
    return '-';
  }
  return value.toFixed(digits === undefined ? 2 : digits);
}

function pct(value, digits) {
  if (!is_number(value)) {
    return '-';
  }
  return (value * 100).toFixed(digits === undefined ? 2 : digits) + '%';
}

function rupees(value) {
  if (!is_number(value)) {
    return '-';
  }
  const sign = value < 0 ? '-' : '';
  const units = [[1e7, 'cr'], [1e5, 'L']];
  for (const [scale, suffix] of units) {
    if (Math.abs(value) >= scale) {
      return sign + 'Rs ' + (value / scale).toFixed(2) + ' ' + suffix;
    }
  }
  return sign + 'Rs ' + value.toFixed(0);
}

/* -------------------------------------------------------------- fetch -- */

function api_base() {
  const field = document.getElementById('api-base');
  const typed = field.value.trim();
  return typed || default_api;
}

function file_hint() {
  if (window.location.protocol !== 'file:') {
    return '';
  }
  return ' This page was opened from file://, and a browser blocks those' +
    ' cross-origin requests by design. Serve the page from the API' +
    ' itself instead: python -m stock_rl.api, then open' +
    ' http://127.0.0.1:8765/';
}

async function get_json(path) {
  const url = api_base() + path;
  let response;
  try {
    response = await fetch(url, {headers: {Accept: 'application/json'}});
  } catch (err) {
    throw new Error('cannot reach ' + url + ' (' + err.message + ')' +
      file_hint());
  }
  if (!response.ok) {
    throw new Error(url + ' answered HTTP ' + response.status);
  }
  try {
    return await response.json();
  } catch (err) {
    throw new Error(url + ' did not answer with JSON (' + err.message +
      ')');
  }
}

async function settle(path) {
  try {
    return {path: path, data: await get_json(path)};
  } catch (err) {
    return {path: path, error: err.message};
  }
}

/* ------------------------------------------------------------- equity -- */

function draw_equity(series) {
  const host = document.getElementById('equity-chart');
  clear(host);
  if (!Array.isArray(series) || series.length < 2) {
    host.appendChild(el('p',
      'No equity curve yet. Run a backtest against this server;' +
      ' the dashboard shows whatever the last run produced.',
      'empty'));
    return;
  }
  const points = series.filter(is_number);
  if (points.length < 2) {
    host.appendChild(el('p',
      'The equity curve carries no usable numbers.', 'empty'));
    return;
  }
  let low = Math.min.apply(null, points.concat([1]));
  let high = Math.max.apply(null, points.concat([1]));
  if (high === low) {
    high = low + 0.01;
  }
  const span = high - low;
  low -= span * 0.08;
  high += span * 0.08;
  const inner_w = plot.wide - plot.left - plot.right;
  const inner_h = plot.tall - plot.top - plot.bottom;
  const x_of = (index) => plot.left + (index * inner_w) /
    Math.max(1, points.length - 1);
  const y_of = (value) => plot.top + (high - value) / (high - low) * inner_h;

  const svg = svg_el('svg', {
    viewBox: '0 0 ' + plot.wide + ' ' + plot.tall,
    role: 'img',
    'aria-label': 'Equity curve, ' + points.length + ' points'
  });

  /* The 1.0 line is the reference that makes the curve readable: an
   * equity line above it gained, below it lost. */
  svg.appendChild(svg_el('line', {
    x1: plot.left, x2: plot.wide - plot.right,
    y1: y_of(1), y2: y_of(1),
    stroke: '#b9bfca', 'stroke-dasharray': '4 4'
  }));

  let path = '';
  points.forEach((value, index) => {
    path += (index === 0 ? 'M' : 'L') + x_of(index).toFixed(2) + ' ' +
      y_of(value).toFixed(2) + ' ';
  });
  svg.appendChild(svg_el('path', {
    d: path.trim(),
    fill: 'none',
    stroke: '#1f4e9c',
    'stroke-width': '2',
    'stroke-linejoin': 'round'
  }));

  const last = points.length - 1;
  svg.appendChild(svg_el('circle', {
    cx: x_of(last), cy: y_of(points[last]), r: '3.5', fill: '#1f4e9c'
  }));
  svg_text(svg, {
    x: plot.left, y: y_of(high) - 2, fill: '#5f6672', 'font-size': '13',
    'text-anchor': 'start'
  }, num(high, 3));
  svg_text(svg, {
    x: plot.left, y: y_of(low) + 12, fill: '#5f6672', 'font-size': '13'
  }, num(low, 3));
  svg_text(svg, {
    x: plot.wide - plot.right, y: plot.tall - 8, fill: '#5f6672',
    'font-size': '13', 'text-anchor': 'end'
  }, 'bar 1 ... bar ' + points.length + '   final ' +
    num(points[last], 4));
  host.appendChild(svg);
}

/* ------------------------------------------------------------- panels -- */

function render_health(data) {
  const line = document.getElementById('status-line');
  line.textContent = 'v' + (data.version || '?') + ' - ' +
    (data.symbols || 0) + ' symbols, ' + (data.bars || 0) + ' bars - ' +
    'up ' + num(data.uptime_seconds, 1) + 's - kill switch ' +
    (data.kill_switch || 'unknown');
}

function render_metrics(data) {
  const body = tbody_of('metrics');
  pair_row(body, 'Sharpe (annualised)', num(data.sharpe, 3));
  pair_row(body, 'Max drawdown', pct(data.max_drawdown));
  pair_row(body, 'Turnover (one-way)', num(data.turnover, 3));
  pair_row(body, 'Total cost', rupees(data.total_cost));
  pair_row(body, 'Growth multiple', num(data.total_return_multiple, 4));
  document.getElementById('metrics-note').textContent =
    (data.points || 0) + ' bars of equity from ' + (data.source || 'none') +
    '. Sharpe and drawdown come from stock_rl.metrics, not from this page.';
}

function render_risk(data) {
  const body = tbody_of('risk');
  const sw = data.kill_switch || {};
  pair_row(body, 'Kill switch', sw.status || 'unknown');
  pair_row(body, 'Algo ID', sw.algo_id || '-');
  pair_row(body, 'Tripped', sw.tripped === null ? 'unknown'
    : (sw.tripped ? 'YES' : 'no'));
  pair_row(body, 'Recorded trips',
    String(Array.isArray(sw.trips) ? sw.trips.length : 0));
  const vr = data.var || {};
  pair_row(body, 'VaR 95% (1 bar)', pct(vr.one_period_fraction));
  pair_row(body, 'VaR 95% in rupees', rupees(vr.rupees));
  const breaches = Array.isArray(data.breaches) ? data.breaches : [];
  pair_row(body, 'Breached now',
    breaches.length ? breaches.join('; ') : 'none');
  document.getElementById('risk-note').textContent =
    (vr.note || '') + ' (' + (vr.observations || 0) +
    ' observations behind it)';
  render_halt(sw);
}

function render_halt(sw) {
  const banner = document.getElementById('halted');
  const status = sw.status || 'not_wired';
  let message = '';
  if (status === 'TRIPPED') {
    message = 'KILL SWITCH TRIPPED for algo ' + (sw.algo_id || '?') +
      '. Trading is halted and stays halted until a human clears it.' +
      ' Nothing on this page may be acted on.';
  } else if (status === 'unreadable') {
    message = 'The kill switch state could not be read: ' +
      (sw.error || 'no detail given') + ' This API reports the refusal' +
      ' rather than showing an untripped switch it cannot verify.';
  } else if (status === 'not_wired') {
    message = 'No kill switch is wired into this server, so the halt' +
      ' state below is unknown rather than clear.';
  }
  banner.hidden = message === '';
  banner.textContent = message;
}

function render_signals(data) {
  const body = tbody_of('signals');
  const rows = Array.isArray(data.signals) ? data.signals : [];
  document.getElementById('signals-note').textContent =
    rows.length + ' symbols - 12-1 momentum over ' + (data.lookback || '?') +
    ' bars, skipping the most recent ' + (data.skip || '?') +
    '. Confidence is a rank percentile across this panel, not a' +
    ' calibrated probability of being right.';
  if (!rows.length) {
    unavailable('signals', 'No signals. The server has no panels loaded.');
    return;
  }
  for (const item of rows) {
    const tr = el('tr');
    tr.appendChild(el('td', item.symbol));
    const action = el('td', item.action || '?',
      'action ' + String(item.action || '').toLowerCase());
    tr.appendChild(action);
    const conf = el('td', null, 'num');
    const wrap = el('span', pct(item.confidence, 0), 'confidence');
    const meter = el('span', null, 'meter');
    const fill = el('span');
    fill.style.width = Math.round(
      Math.max(0, Math.min(1, item.confidence || 0)) * 100) + '%';
    meter.appendChild(fill);
    wrap.appendChild(meter);
    conf.appendChild(wrap);
    tr.appendChild(conf);
    tr.appendChild(el('td',
      num(item.held_weight, 3) + ' -> ' + num(item.target_weight, 3),
      'num'));
    const reasons = el('td');
    const list = el('ul', null, 'reasons');
    for (const reason of item.reasons || ['no reason recorded']) {
      list.appendChild(el('li', reason));
    }
    reasons.appendChild(list);
    tr.appendChild(reasons);
    body.appendChild(tr);
  }
}

function render_baselines(data) {
  const body = tbody_of('baselines');
  const rows = Array.isArray(data.strategies) ? data.strategies : [];
  if (!rows.length) {
    unavailable('baselines', 'No baselines were reported.');
    return;
  }
  for (const item of rows) {
    const ok = item.status === 'ok';
    const tr = row(body, [
      [item.name],
      [num(item.sharpe, 3), 'num'],
      [pct(item.max_drawdown), 'num'],
      [num(item.turnover, 3), 'num'],
      [rupees(item.total_cost), 'num'],
      [ok ? 'measured' : (item.error || 'skipped')]
    ]);
    if (item.name === data.best_sharpe) {
      tr.className = 'best';
      tr.title = 'highest Sharpe of the measured rows';
    }
  }
}

/* ---------------------------------------------------------------- run -- */

async function load() {
  const button = document.getElementById('refresh');
  button.disabled = true;
  const error_box = document.getElementById('error');
  const answers = await Promise.all([
    settle('/api/health'),
    settle('/api/equity'),
    settle('/api/positions'),
    settle('/api/signals'),
    settle('/api/baselines'),
    settle('/api/risk')
  ]);
  const by_path = {};
  for (const answer of answers) {
    by_path[answer.path] = answer;
  }

  const health = by_path['/api/health'];
  if (health.error) {
    document.getElementById('status-line').textContent = 'API unreachable';
    blank('equity-chart', health.error);
  } else {
    render_health(health.data);
  }

  const equity = by_path['/api/equity'];
  if (equity.error) {
    unavailable('metrics', equity.error);
    blank('equity-chart', equity.error);
    document.getElementById('metrics-note').textContent = '';
  } else {
    render_metrics(equity.data);
    draw_equity(equity.data.equity);
    document.getElementById('equity-note').textContent =
      (equity.data.points || 0) + ' bars, normalised to 1.0 at the start' +
      ' - final ' + rupees(equity.data.final_value) + ' on ' +
      rupees(equity.data.capital) + ' of capital';
  }

  const risk = by_path['/api/risk'];
  if (risk.error) {
    unavailable('risk', risk.error);
    document.getElementById('risk-note').textContent = '';
    document.getElementById('halted').hidden = true;
  } else {
    render_risk(risk.data);
  }

  const signals = by_path['/api/signals'];
  if (signals.error) {
    unavailable('signals', signals.error);
    document.getElementById('signals-note').textContent = '';
  } else {
    render_signals(signals.data);
  }

  const baselines = by_path['/api/baselines'];
  if (baselines.error) {
    unavailable('baselines', baselines.error);
  } else {
    render_baselines(baselines.data);
  }

  const positions = by_path['/api/positions'];
  const held = positions.error ? null : positions.data;
  if (!positions.error) {
    document.getElementById('signals-note').textContent +=
      ' - book: ' + pct(held.invested, 1) + ' invested, ' +
      rupees(held.cash) + ' cash, drawdown ' + pct(held.drawdown);
  }

  const failures = answers.filter((answer) => answer.error);
  const messages = [];
  for (const failure of failures) {
    if (!messages.some((text) => text === failure.error)) {
      messages.push(failure.error);
    }
  }
  error_box.hidden = failures.length === 0;
  error_box.textContent = failures.length
    ? failures.length + ' of 6 endpoints failed: ' + messages.join(' | ')
    : '';
  button.disabled = false;
}

document.getElementById('api-base').value = default_api;
document.getElementById('api-base').addEventListener('change', load);
document.getElementById('refresh').addEventListener('click', load);
load();
