/* Dependency graph with news, for the stock_rl dashboard.
 *
 * Vanilla JavaScript, inline SVG, no dependencies, no build step, no
 * network. Same constraints and the same reasons as app.js: the whole
 * view is one SVG built with createElementNS and a handful of DOM
 * nodes, so the framework the design doc proposed would buy nothing and
 * cost a package.json, a node_modules tree and a build artefact.
 *
 * FOUR RULES THIS FILE FOLLOWS
 *
 * 1. All text goes in through textContent. No value is ever assigned as
 *    markup, and no HTML-parsing setter is called: the payloads this view
 *    renders carry node notes and free-text provenance sentences written
 *    by stock_rl.graph.news, and those are exactly the values that must
 *    not be parsed as markup. (tests/test_web_graph.py asserts that no
 *    markup-setting identifier appears in this file at all, this comment
 *    included, so the rule cannot be reintroduced in a docstring either.)
 *
 * 2. The data is a plain JSON object with a documented shape: whatever
 *    stock_rl.graph.news.GraphSlice.to_dict() produces. It comes from an
 *    inline <script type="application/json"> island when one is present,
 *    which is what makes this view work with the network unplugged and
 *    from file://. Only if no island is present does it fetch, and the URL
 *    it fetches is on the same loopback origin the rest of the dashboard
 *    already uses. No external origin is ever contacted, so no external
 *    font, script or style is needed either.
 *
 *    Every reading on screen is attached **not by matching text**. The
 *    headline is never compared against a node name of any kind; the
 *    symbol, the hop count and the key chain all come from the dependency
 *    graph, and the reason printed under each row says so in words a
 *    reader can check against the graph file.
 *
 * 3. Every edge is labelled with what it claims, and the unfitted caveat
 *    is always on screen. stock_rl.graph.edges says the seed is common
 *    sector reasoning rather than a measured sensitivity; an arrow drawn
 *    without that qualification is a measurement that does not exist.
 *    The caveat is not dismissible and not conditional.
 *
 * 4. No news is a sentence, and it is not an error. The three empty-ish
 *    states this view can be in -- has_news, no_news, no_dependents --
 *    are worded differently on purpose, and a fetch failure is a fourth,
 *    visually distinct state. A reader who cannot tell "nothing was
 *    public at the bar" from "the lookup found no route to any company"
 *    from "the request failed" will conclude something from whichever
 *    one they assume.
 *
 * PONYTAIL: columns by hop, one row per node, one SVG. Ceiling: no force
 * layout, no pan or zoom, no edge bundling, so a dense graph gets tall
 * rather than readable. Upgrade path: wrap the SVG in a viewBox plus a
 * scroll container and keep plan() as the only layout code.
 */

'use strict';

/* The one http string in this file. It is the SVG namespace identifier
 * that createElementNS requires, not an origin anything is fetched
 * from, and a browser never dereferences it. Kept as a named constant
 * and asserted as such by tests/test_web_graph.py so the "no external
 * origin" claim stays checkable rather than a promise. */
const svg_ns = 'http://www.w3.org/2000/svg';

/* Layout box. Column x is derived from hops, row y from the position
 * within the column, so the layout is a pure function of the payload
 * and needs no iteration, no collision detection and no randomness. */
const box = {
  wide: 960,
  pad: 24,
  node_w: 208,
  node_h: 58,
  col_gap: 132,
  row_gap: 26,
  label_gap: 14
};

const el = (tag, text, class_name) => {
  const node = document.createElement(tag);
  if (text !== null && text !== undefined) {
    node.textContent = String(text);
  }
  if (class_name) {
    node.className = class_name;
  }
  return node;
};

const svg_el = (tag, attributes) => {
  const node = document.createElementNS(svg_ns, tag);
  for (const [key, value] of Object.entries(attributes)) {
    node.setAttribute(key, String(value));
  }
  return node;
};

const svg_text = (parent, attributes, text) => {
  const node = svg_el('text', attributes);
  node.textContent = text === undefined || text === null ? '' : String(text);
  parent.appendChild(node);
  return node;
};

/* A <title> child is the platform's own tooltip, needs no JavaScript and
 * cannot execute anything, so the long form of every edge label rides
 * along with the drawn arrow. */
const svg_tip = (parent, text) => {
  const node = svg_el('title', {});
  node.textContent = String(text);
  parent.appendChild(node);
  return node;
};

const clear = (node) => {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
};

const is_number = (value) => typeof value === 'number' && Number.isFinite(value);

const num = (value, digits) => (
  is_number(value) ? value.toFixed(digits === undefined ? 2 : digits) : '-'
);

const pct = (value, digits) => (
  is_number(value) ? (value * 100).toFixed(digits === undefined ? 0 : digits) + '%' : '-'
);

/* ------------------------------------------------------------- payload -- */

/* The shape this view consumes is whatever stock_rl.graph.news produces.
 * Kept as a list of the fields it reads rather than a schema library,
 * because a payload from a server that does not exist yet has to fail as
 * a sentence and not as an exception on a property of undefined. */
function plan(payload, focus_key) {
  if (payload === null || typeof payload !== 'object') {
    return {
      ok: false,
      kind: 'error',
      message: 'No graph payload was supplied, so there is nothing to draw. '
        + 'Provide one as JSON: either an inline '
        + '<script type="application/json"> island with an id, or an object '
        + 'handed to mount(). Nothing is guessed and nothing is stubbed.'
    };
  }
  const rows = Array.isArray(payload.nodes) ? payload.nodes : [];
  if (!rows.length) {
    return {
      ok: false,
      kind: 'error',
      message: 'The graph payload carries no nodes, so there is no graph to '
        + 'draw. That is a payload problem, not an absence of news.'
    };
  }
  const readings = Array.isArray(payload.readings) ? payload.readings : [];
  const wanted = focus_key || payload.node || (rows[0] && rows[0].key) || '';
  const focus = rows.find((row) => row && row.key === wanted) ||
    rows.find((row) => row && row.role === 'focus') || rows[0];
  const columns = new Map();
  for (const row of rows) {
    if (!row || typeof row.key !== 'string') {
      continue;
    }
    const hop = is_number(row.hops) && row.hops > 0 ? Math.round(row.hops) : 0;
    if (!columns.has(hop)) {
      columns.set(hop, []);
    }
    columns.get(hop).push(row);
  }
  const hops = Array.from(columns.keys()).sort((left, right) => left - right);
  const tallest = hops.reduce((most, hop) => Math.max(
    most, columns.get(hop).length), 0);
  const height = box.pad * 2 + tallest * box.node_h +
    (tallest - 1) * box.row_gap;
  const placed = [];
  for (const hop of hops) {
    const group = columns.get(hop);
    const span = group.length * box.node_h + (group.length - 1) * box.row_gap;
    let y = (height - span) / 2;
    for (const row of group) {
      placed.push({
        key: row.key,
        label: row.label || row.key,
        kind: row.kind || '',
        note: row.note || '',
        hops: hop,
        role: row.role || 'path',
        focus: row.key === focus.key,
        news: readings.filter((item) => Array.isArray(item.via_path)
          && item.via_path.indexOf(row.key) >= 0).length,
        x: box.pad + hops.indexOf(hop) * (box.node_w + box.col_gap),
        y: y
      });
      y += box.node_h + box.row_gap;
    }
  }
  const boxes = new Map();
  for (const item of placed) {
    boxes.set(item.key, item);
  }
  const links = [];
  for (const edge of (Array.isArray(payload.edges) ? payload.edges : [])) {
    const tail = edge && boxes.get(edge.cause);
    const head = edge && boxes.get(edge.affected);
    if (!tail || !head) {
      continue;
    }
    const x1 = tail.x + box.node_w;
    const y1 = tail.y + box.node_h / 2;
    const x2 = head.x;
    const y2 = head.y + box.node_h / 2;
    const bend = (x2 - x1) / 2;
    links.push({
      cause: tail.key,
      affected: head.key,
      kind: edge.kind || '',
      short: edge.short || edge.kind || 'edge',
      meaning: edge.meaning || '',
      measured: edge.measured === true,
      x1: x1, y1: y1, x2: x2, y2: y2,
      mid_x: (x1 + x2) / 2,
      mid_y: Math.min(y1, y2) - box.label_gap,
      d: 'M' + x1.toFixed(1) + ' ' + y1.toFixed(1) + ' C' +
        (x1 + bend).toFixed(1) + ' ' + y1.toFixed(1) + ' ' +
        (x2 - bend).toFixed(1) + ' ' + y2.toFixed(1) + ' ' +
        x2.toFixed(1) + ' ' + y2.toFixed(1)
    });
  }
  return {
    ok: true,
    kind: 'graph',
    focus: focus,
    nodes: placed,
    links: links,
    readings: readings,
    routes: Array.isArray(payload.routes) ? payload.routes : [],
    checked: Array.isArray(payload.checked_symbols)
      ? payload.checked_symbols : [],
    status: payload.status || 'no_news',
    payload_message: payload.message || '',
    unfitted: payload.unfitted || '',
    not_advice: payload.not_advice || '',
    max_depth: payload.max_depth,
    wide: box.pad * 2 + hops.length * box.node_w + (hops.length - 1) * box.col_gap,
    tall: height
  };
}

/* What the news panel says for one node. Three states, worded apart:
 * has_news, no_news and no_dependents. A node that was merely reached by
 * the walk but has nothing filed under it is reported as an absence,
 * and the walk that reached it is named, because "we looked and found
 * nothing" and "we did not look" are different facts. */
function news_for(plan_result, key) {
  /* What the news panel should say for one node.
   *
   * `via` is the set of symbols whose readings are on screen. It matters
   * because a commodity or macro node has no news of its own: everything
   * under it belongs to the companies that consume it, and a heading that
   * read "News for Crude Oil" over three companies' headlines would be
   * exactly the kind of quiet misattribution this view exists to prevent.
   */
  const mine = plan_result.readings.filter(
    (item) => Array.isArray(item.via_path)
      && item.via_path.indexOf(key) >= 0);
  const node = plan_result.nodes.find((item) => item.key === key);
  const name = (node && node.label) || key;
  const via = Array.from(new Set(mine.map((item) => item.symbol))).sort();
  if (mine.length) {
    return {kind: 'news', readings: mine, message: '', label: name,
      via, node};
  }
  if (node && node.focus) {
    const why = plan_result.payload_message ||
      'The graph records no route from this node to a listed company at the '
      + 'stated depth, so there was nothing to look up.';
    return {kind: plan_result.status === 'has_news' ? 'empty' : plan_result.status,
      readings: [], message: why, label: name, via, node};
  }
  const symbols = plan_result.checked.length
    ? ' The companies the graph says are exposed to it were: ' +
      plan_result.checked.join(', ') + '.'
    : '';
  return {
    kind: 'empty',
    readings: [],
    message: 'No sentiment reading is attached to ' + name + '. Nothing was '
      + 'looked up for it, because no headline resolves to a node that is '
      + 'not a listed symbol.' + symbols,
    label: name,
    via,
    node
  };
}

function heading(view) {
  /* Return the panel heading, naming the symbols when there is more than one.
   *
   * "News for Crude Oil" alone would read as though crude said something.
   * What is actually shown is ONGC's and RELIANCE's news, attached through
   * the dependency edges, so the heading says so. */
  if (view.via && view.via.length > 1) {
    return 'News for ' + view.label + ', via ' + view.via.join(', ');
  }
  return 'News for ' + view.label;
}

/* ---------------------------------------------------------------- draw -- */

function tone(score) {
  if (!is_number(score)) {
    return 'flat';
  }
  if (score > 0) {
    return 'up';
  }
  if (score < 0) {
    return 'down';
  }
  return 'flat';
}

/* What a node is IN THIS DRAWING, which is not its kind.
 *
 * A ``peer`` is a listed company the graph says this node depends on. It
 * is drawn, because the edge is a real claim in the file, and it is
 * labelled a peer rather than an input, because nothing in the graph
 * established that one listed company supplies another. Printing the
 * role on the box is what stops the reader assuming otherwise.
 */
function role_word(node) {
  if (node.role === 'peer') {
    return 'peer, not an input';
  }
  if (node.role === 'stock') {
    return 'reached, news attached';
  }
  if (node.role === 'focus') {
    return 'focus';
  }
  return 'on the path';
}

function draw_canvas(plan_result, on_pick) {
  /* Build the SVG for a laid-out slice.
   *
   * Returns a detached <svg>. Nothing here reads the document, so the
   * whole drawing is a function of its two arguments and the tests can
   * drive it against a stub document. */
  const svg = svg_el('svg', {
    viewBox: '0 0 ' + Math.round(plan_result.wide) + ' ' +
      Math.round(plan_result.tall),
    role: 'img',
    'aria-label': 'Dependency graph for ' + plan_result.focus.key +
      ': ' + plan_result.nodes.length + ' nodes, ' +
      plan_result.links.length + ' edges'
  });
  for (const link of plan_result.links) {
    const path = svg_el('path', {
      d: link.d,
      fill: 'none',
      stroke: link.measured ? '#1f4e9c' : '#8b93a1',
      'stroke-width': link.measured ? '2.4' : '1.6',
      'stroke-dasharray': link.measured ? 'none' : '5 4'
    });
    svg_tip(path, link.meaning);
    svg.appendChild(path);
    /* Every edge carries its meaning on the canvas, not only in a
     * tooltip: the short label names the direction, and a hover or focus
     * reveals the sentence. */
    const label = svg_text(svg, {
      x: link.mid_x, y: link.mid_y,
      fill: '#5f6672', 'font-size': '11',
      'text-anchor': 'middle'
    }, link.short);
    svg_tip(label, link.meaning);
  }
  for (const node of plan_result.nodes) {
    const group = svg_el('g', {
      class: 'gnode gnode-' + node.role + (node.focus ? ' gnode-focus' : ''),
      tabindex: '0',
      role: 'button',
      'aria-label': node.label + ', ' + node.kind + ', ' + node.hops +
        ' hop(s) from ' + plan_result.focus.label + ', ' + node.news +
        ' reading(s) attached'
    });
    svg_tip(group, node.note || node.label);
    group.appendChild(svg_el('rect', {
      x: node.x, y: node.y, width: box.node_w, height: box.node_h,
      rx: '5'
    }));
    svg_text(group, {
      x: node.x + 10, y: node.y + 21, 'font-size': '13',
      'font-weight': '600'
    }, node.label);
    svg_text(group, {
      x: node.x + 10, y: node.y + 38, 'font-size': '11', fill: '#5f6672'
    }, node.kind + ' - ' + role_word(node) + ' - ' + node.hops + ' hop' +
      (node.hops === 1 ? '' : 's') + ' - ' + node.news + ' reading' +
      (node.news === 1 ? '' : 's'));
    group.addEventListener('click', () => on_pick(node.key));
    group.addEventListener('keydown', (event) => {
      const key_name = event.key;
      if (key_name === 'Enter' || key_name === ' ' || key_name === 'Spacebar') {
        event.preventDefault();
        on_pick(node.key);
      }
    });
    svg.appendChild(group);
  }
  return svg;
}

function source_line(item) {
  /* Return the sentence naming who produced a reading.
   *
   * An unlabelled source gets said out loud rather than printed as blank,
   * because stock_rl.sentiment.score collapses every unnamed feed into
   * one bucket precisely because an unlabelled feed cannot be shown to be
   * independent of another unlabelled feed. */
  return item.source
    ? 'source ' + item.source
    : 'source unnamed, so it cannot be shown independent of another '
      + 'unnamed feed';
}

function verdict_line(item) {
  /* Return the sentence a reader needs before trusting a score.
   *
   * The score itself is not the number to act on: the source minimum and
   * the cross-source spread are. A single-source reading and a mean whose
   * sources agree to within the convergence tolerance both look like
   * plain numbers in a table, so both get said out loud here. */
  const spread = 'cross-source spread ' + num(item.disagreement, 3);
  if (item.usable) {
    return 'verdict: usable aggregate, ' + spread
      + (item.converged ? ', and the sources agree suspiciously closely' : '');
  }
  const why = Array.isArray(item.rejected) && item.rejected.length
    ? ' - ' + item.rejected.join(', ')
    : '';
  return 'verdict: NOT usable' + why;
}

function draw_news(view) {
  /* Build the news panel for one node.
   *
   * Returns a detached div. The caller appends it, so a panel is either
   * built whole or not at all and can never half-render into the live
   * document. */
  const panel = el('div', null, 'graph-news');
  panel.appendChild(el('h3', heading(view)));
  if (view.node && view.node.note) {
    panel.appendChild(el('p', view.node.note, 'muted small'));
  }
  if (!view.readings.length) {
    /* The honest empty panel. It names what was looked for and says the
     * lookup ran, so an empty list is never mistaken for a broken one.
     * The class differs from graph-error on purpose: the two must not be
     * able to collapse into the same grey box on screen. */
    panel.appendChild(el('p', view.message,
      view.kind === 'error' ? 'graph-error' : 'graph-empty'));
    return panel;
  }
  panel.appendChild(el('p', view.readings.length + ' reading(s), each '
    + 'attached through the dependency graph rather than by matching the '
    + 'headline to this node. Read the reason under every row.',
  'muted small'));
  for (const item of view.readings) {
    /* gread-usable is what the stylesheet keys on to mute a
     * single-source score. It lives on the card rather than being
     * recomputed in CSS because "usable" is a property of the aggregate,
     * and duplicating that rule in a selector is how a stylesheet and a
     * payload start disagreeing. */
    const card = el('article', null,
      'gread' + (item.usable ? ' gread-usable' : ''));
    const line = el('div', null, 'gread-head');
    line.appendChild(el('span', item.symbol, 'gread-symbol'));
    line.appendChild(el('span',
      (item.score > 0 ? '+' : '') + num(item.score, 3),
      'gread-score ' + tone(item.score)));
    line.appendChild(el('span', 'intensity ' + pct(item.intensity),
      'gread-meta'));
    line.appendChild(el('span', 'public ' + item.available_from,
      'gread-meta'));
    card.appendChild(line);
    card.appendChild(el('p', source_line(item), 'gread-meta'));
    card.appendChild(el('p', verdict_line(item), 'gread-meta'));
    card.appendChild(el('p', item.reason, 'gread-reason'));
    card.appendChild(el('p', 'route: ' + item.via_path.join(' -> ') + ' ('
      + item.via_hops + ' hop(s), edge kinds ' + item.via_edges.join(', ')
      + ') | registry: ' + item.via_linkage, 'gread-route'));
    panel.appendChild(card);
  }
  return panel;
}

function draw_legend(plan_result) {
  /* Build the legend that lists every edge and what it claims.
   *
   * The short label on the canvas names a direction in two words. This is
   * where the full sentence lives, so "every edge is labelled with its
   * meaning" is not a claim about a tooltip the reader has to find. */
  const block = el('div', null, 'graph-legend');
  block.appendChild(el('h3', 'Every edge, and what it claims'));
  block.appendChild(el('p', 'An arrow runs from the cause to the node the '
    + 'shock reaches, which is the reverse of the stored edge. Solid arrows '
    + 'are measured; none here are, so every edge is dashed and carries its '
    + 'claim in full below.', 'muted small'));
  if (!plan_result.links.length) {
    block.appendChild(el('p', 'This slice draws no edges. The graph records '
      + 'no dependency between the nodes it returned, which is a statement '
      + 'about the file and not about the market.', 'graph-empty'));
    return block;
  }
  for (const link of plan_result.links) {
    block.appendChild(el('p', link.meaning,
      'graph-legend-row' + (link.measured ? '' : ' graph-unfitted')));
  }
  return block;
}

function render(host, plan_result, on_pick) {
  clear(host);
  if (!plan_result.ok) {
    /* A payload that cannot be laid out is an error, and it is drawn as
     * one. Rendering it as a blank canvas would let a broken feed look
     * exactly like a market with no dependencies. */
    host.appendChild(el('p', plan_result.message, 'graph-error'));
    return;
  }
  /* Never conditional and never dismissible. If the caveat were shown
   * only when convenient it would be absent exactly when a reader is in a
   * hurry, which is when a dashed arrow gets read as a measurement. */
  host.appendChild(el('p', plan_result.unfitted
    || 'Unfitted sector reasoning, not measured sensitivities.', 'graph-unfitted'));
  const canvas = el('div', null, 'graph-canvas');
  canvas.appendChild(draw_canvas(plan_result, on_pick));
  host.appendChild(canvas);
  host.appendChild(el('p', 'Focus: ' + plan_result.focus.label + ' ('
    + plan_result.focus.kind + ')'
    + (is_number(plan_result.max_depth)
      ? ', walked to at most ' + plan_result.max_depth + ' hop(s)' : '')
    + '. Click any node, or tab to it and press Enter, for its news. A '
    + 'dashed arrow is an unfitted dependency claim; a solid one would be a '
    + 'measured sensitivity, and this package holds none.', 'muted small'));
  host.appendChild(draw_legend(plan_result));
  if (plan_result.routes.length) {
    const routes = el('div', null, 'graph-routes');
    routes.appendChild(el('h3', 'How each symbol was reached'));
    for (const route of plan_result.routes) {
      routes.appendChild(el('p', route.symbol + ' - ' + route.describe,
        'gread-route'));
    }
    host.appendChild(routes);
  }
}

/* ----------------------------------------------------------------- api -- */

function island_text(doc) {
  const node = doc.getElementById('graph-payload');
  return node && typeof node.textContent === 'string' ? node.textContent : '';
}

function parse_payload(text) {
  try {
    const parsed = JSON.parse(text);
    return {ok: true, payload: parsed};
  } catch (err) {
    return {
      ok: false,
      message: 'The graph payload is not valid JSON (' + err.message
        + '), so nothing was drawn. A malformed payload is reported rather '
        + 'than rendered as an empty graph.'
    };
  }
}

/* The one place this view touches the network, and only when the page
 * carries no inline island. Same loopback origin as the rest of the
 * dashboard; no external host appears anywhere in this file. */
async function fetch_payload(url) {
  let response;
  try {
    response = await fetch(url, {headers: {Accept: 'application/json'}});
  } catch (err) {
    return {
      ok: false,
      message: 'Could not reach ' + url + ' (' + err.message + '). The graph '
        + 'was not drawn and no news is shown; this is a transport failure, '
        + 'not an absence of news.'
    };
  }
  if (!response.ok) {
    return {
      ok: false,
      message: url + ' answered HTTP ' + response.status + ', so there is no '
        + 'graph to draw. A server without this endpoint is a missing '
        + 'feature, and it is reported as one rather than as "no news".'
    };
  }
  return parse_payload(await response.text());
}

async function resolve_payload(host, doc, settings) {
  /* Find the payload, in one place and in a fixed order.
   *
   * An explicit object wins, then an inline JSON island on the page, then
   * a loopback URL. The order matters offline: the island is checked
   * before the network so a page that ships its own data never needs a
   * server, and a page that has neither ends up with a sentence rather
   * than a request to a host that was never configured.
   *
   * Returns {ok: true, payload} or {ok: false, message}. The two are kept
   * apart all the way to render() so "no payload configured" is drawn as
   * the error it is and never as an empty graph. */
  if (settings.payload !== undefined && settings.payload !== null) {
    return {ok: true, payload: settings.payload};
  }
  const inline = island_text(doc);
  if (inline) {
    return parse_payload(inline);
  }
  if (settings.url) {
    return fetch_payload(settings.url);
  }
  return {
    ok: false,
    message: 'No graph payload was supplied and this page carries no inline '
      + 'JSON island, so there is nothing to draw. Pass a payload to '
      + 'mount(), add a <script type="application/json"> island, or give a '
      + 'loopback URL. Nothing is fetched from an external origin and no '
      + 'placeholder graph is invented.'
  };
}

async function mount(host, options) {
  /* Draw the view into ``host`` and wire node clicks to the news panel.
   *
   * ``options.payload`` is the documented shape, which is whatever
   * stock_rl.graph.news.GraphSlice.to_dict produces. ``options.focus``
   * names a node key to centre on, defaulting to the slice's own focus.
   * ``options.url`` is used only when the page carries no inline island,
   * and must be a loopback origin.
   *
   * Returns the laid-out plan, or null when nothing could be drawn. */
  const settings = options || {};
  const doc = host.ownerDocument || (typeof document !== 'undefined'
    ? document : null);
  const found = await resolve_payload(host, doc || {getElementById: () => null},
    settings);
  if (!found.ok) {
    render(host, {ok: false, kind: 'error', message: found.message});
    return null;
  }
  const drawn = plan(found.payload, settings.focus);
  if (!drawn.ok) {
    render(host, drawn);
    return null;
  }
  const panel = doc ? doc.getElementById('graph-news') : null;
  const show = (key) => {
    if (!panel) {
      return;
    }
    clear(panel);
    panel.appendChild(draw_news(news_for(drawn, key)));
  };
  render(host, drawn, show);
  show(drawn.focus.key);
  return drawn;
}

const api = {
  plan: plan,
  news_for: news_for,
  parse_payload: parse_payload,
  resolve_payload: resolve_payload,
  render: render,
  draw_news: draw_news,
  fetch_payload: fetch_payload,
  mount: mount,
  layout: box,
  heading: heading,
  role_word: role_word,
  tone: tone
};

/* Loaded as a plain <script> in the browser and required by the test
 * suite under node. Both branches hand over the same object; nothing here
 * touches the DOM at load time, which is what lets the layout and the
 * message logic be exercised without a browser.
 *
 * The document-ready hook is the one piece of ambient behaviour, and it
 * is guarded on the host elements actually existing. A page that does not
 * want this view mount simply has no #graph-view, and nothing happens;
 * a page that does want it gets the payload from the island and never
 * reaches the network. */
if (typeof module !== 'undefined' && module.exports) {
  module.exports = api;
} else if (typeof window !== 'undefined') {
  window.StockRlGraph = api;
  const boot = () => {
    const anchor = window.document.getElementById('graph-view');
    if (anchor) {
      mount(anchor, {focus: anchor.getAttribute('data-focus') || undefined});
    }
  };
  if (window.document.readyState === 'loading') {
    window.document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
}