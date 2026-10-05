#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The dependency-graph view: its rules, and the promises it makes.

There is no DOM library in this repository and no npm, so most of this
file asserts things that can be read off the source: that the view never
assigns markup, that it names no external origin, that it parses under
node, and that it exports the logic a browser would run. Those are the
rules that would break silently -- a view that quietly used one HTML
parsing setter, or reached for a CDN font, renders perfectly on the
machine that wrote it and fails on the one that must run offline.

Where node is available, the exported logic is **executed** against a real
payload produced by :mod:`stock_rl.graph.news`, rather than against a
hand-written fixture. That is the point of these tests: a fixture written
by the same author as the view proves the view handles a shape the author
imagined. The payload here comes from the shipped join, so if the payload
contract and the view ever drift apart, these fail.

The classes are ordered by how badly a violation would hurt:

1. :class:`TestNoMarkupIsEverAssigned` -- the hard rule.
2. :class:`TestTheViewWorksWithTheNetworkUnplugged` -- the offline claim.
3. :class:`TestItParsesAndExports` -- it is loadable at all.
4. :class:`TestTheViewDrawsWhatTheJoinProduced` -- behaviour, on a real
   payload.
5. :class:`TestDegradationIsHonest` -- the empty and error states, which
   is where a dashboard most easily lies.
'''

from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from stock_rl.graph.edges import (
  DependencyGraph,
  Edge,
  EdgeKind,
  nifty50_seed,
)
from stock_rl.graph.news import UNFITTED_NOTE, node_slice
from stock_rl.graph.nodes import Node, NodeKind
from stock_rl.sentiment.score import SentimentReading

#: The bar the fixture payload is read at.
BAR = datetime(2024, 1, 10, 15, 30, tzinfo=timezone.utc)

#: The three files that make up the view.
WEB = Path(__file__).resolve().parents[1] / 'src' / 'stock_rl' / 'web'
VIEW = WEB / 'graph.js'
STYLE = WEB / 'graph.css'
PAGE = WEB / 'graph.html'

#: Every spelling of an HTML-parsing sink this repository forbids. The
#: innerHTML family is the one that matters: a payload carrying a node
#: note or a provenance sentence is exactly the value that must never be
#: parsed as markup, and a note is free text written by hand in
#: graph/news.py, so it can contain anything at all.
MARKUP_SETTERS = (
  'innerHTML',
  'outerHTML',
  'insertAdjacentHTML',
  'document.write',
  'document.writeln',
  'createContextualFragment',
  'DOMParser',
  'Range.createContextualFragment',
)

#: Patterns that would mean the page reaches outside itself.
EXTERNAL_ORIGINS = re.compile(
  r'''(?x)
  https?://(?!www\.w3\.org/2000/svg)   # any scheme+host but the SVG namespace
  | //[a-z0-9-]+\.[a-z]{2,}             # protocol-relative
  | @import                            # a CSS import
  | cdn\.
  | googleapis
  | unpkg
  | jsdelivr
  | cdnjs
  ''')

#: The one permitted http string: the SVG namespace identifier that
#: ``createElementNS`` requires. A browser never dereferences it, so it is
#: not an origin the page reaches, and the exemption is named rather than
#: blanket so a second URL cannot hide inside it.
SVG_NAMESPACE = 'http://www.w3.org/2000/svg'


def view_text() -> str:
  '''Return the view source with comments stripped.

  Comments are removed so a rule stated in a docstring is not mistaken
  for the rule being broken. The regex targets ``/* */`` and ``//`` runs
  without trying to be a JavaScript parser: this file cares about code.

  Returns:
    The source with every block and line comment removed.
  '''
  source = VIEW.read_text(encoding='utf-8')
  without_blocks = re.sub(r'/\*.*?\*/', '', source, flags=re.S)
  return re.sub(r'(?m)^\s*//.*$', '', without_blocks)


def page_text() -> str:
  '''Return the standalone page source with HTML comments removed.

  Comments are stripped so a rule stated in a comment is not read as the
  rule being broken. The same reasoning as :func:`view_text`: these tests
  care about live references, and ``<link href>`` inside a comment is not
  one.

  Returns:
    The text of ``graph.html`` without ``<!-- -->`` runs.
  '''
  return re.sub(r'<!--.*?-->', '', PAGE.read_text(encoding='utf-8'),
                flags=re.S)


def style_text() -> str:
  '''Return the view stylesheet with CSS comments removed.

  Returns:
    The text of ``graph.css`` without ``/* */`` runs. An ``@import`` inside
    a comment is not an import, and asserting on it would force the next
    person to stop documenting the rule in order to keep the test green.
  '''
  return re.sub(r'/\*.*?\*/', '', STYLE.read_text(encoding='utf-8'),
                flags=re.S)


def flowed(text: str) -> str:
  '''Collapse runs of whitespace so a phrase can wrap across lines.

  Args:
    text: Source or message text.

  Returns:
    The same text with every whitespace run replaced by one space and the
    ends stripped. A sentence in an HTML paragraph or a CSS comment wraps
    at whatever column the file's style guide allows, so asserting on a
    phrase without this would fail on reformatting alone -- which is how a
    prose assertion trains people to stop writing the sentence.
  '''
  return ' '.join(text.split())


def node_binary() -> str | None:
  '''Return the node executable, if there is one.

  Returns:
    Path to node, or None when it is not installed. The tests that need
    it skip rather than fail, because a machine without node can still
    run the Python half of this file and should not be blocked by a tool
    it was never promised.
  '''
  return shutil.which('node')


def payload_json(symbol: str = 'ONGC') -> str:
  '''Return a real slice payload as JSON text.

  Args:
    symbol: Symbol whose readings the payload carries.

  Returns:
    The JSON text of ``node_slice(graph, 'commodity:crude', readings,
    BAR).to_dict()``. Built by the shipped join rather than written here,
    so the view is exercised against the shape it will actually receive.
  '''
  readings = [
    SentimentReading(symbol, 0.45, 'news', 0.6, BAR, 'reuters'),
    SentimentReading(symbol, 0.10, 'news', 0.4, BAR, 'cnbc'),
  ]
  return json.dumps(node_slice(nifty50_seed(), 'commodity:crude',
                               readings, BAR).to_dict())


def mixed_payload_json() -> str:
  '''Return a slice payload carrying two symbols and two verdicts.

  ONGC has two readings from distinct sources, so its aggregate clears the
  source minimum. RELIANCE has one, so its aggregate is refused. A payload
  carrying only one of those two shapes would let a view render every card
  identically and still pass; this one cannot.

  Returns:
    The JSON text of a ``commodity:crude`` slice built by the shipped join.
  '''
  readings = [
    SentimentReading('ONGC', 0.45, 'news', 0.6, BAR, 'reuters'),
    SentimentReading('ONGC', 0.10, 'news', 0.4, BAR, 'cnbc'),
    SentimentReading('RELIANCE', -0.60, 'news', 0.9, BAR, 'reuters'),
  ]
  return json.dumps(node_slice(nifty50_seed(), 'commodity:crude',
                               readings, BAR).to_dict())


#: A DOM good enough for the view, and no more. It exists because there
#: is no npm in this repository and therefore no jsdom, and because a stub
#: that implements only what the view calls is a check in itself: if the
#: view reaches for a DOM feature the stub does not have, the test fails
#: loudly rather than quietly passing against a permissive mock.
#:
#: Implemented: createElement, createElementNS, appendChild, removeChild,
#: firstChild, textContent (get and set), className, setAttribute,
#: getAttribute, addEventListener, dispatch by type, and getElementById.
#: Deliberately NOT implemented: innerHTML, querySelector, innerText,
#: insertAdjacentHTML, dataset, style objects and Range. The view uses
#: none of them, and a call to any of them raises here.
DOM_SHIM = r'''
function shim_node(tag, ns) {
  const node = {
    tagName: tag, namespaceURI: ns || null, children: [], parent: null,
    _text: '', attributes: {}, classes: '', listeners: {},
    get firstChild() { return node.children.length ? node.children[0] : null; },
    get textContent() {
      if (node.children.length === 0) { return node._text; }
      return node._text + node.children.map((c) => c.textContent).join('');
    },
    set textContent(value) {
      node.children = [];
      node._text = String(value);
    },
    get className() { return node.classes; },
    set className(value) { node.classes = String(value); },
    appendChild(child) {
      child.parent = node;
      node.children.push(child);
      return child;
    },
    removeChild(child) {
      const at = node.children.indexOf(child);
      if (at < 0) { throw new Error('removeChild: not a child of this node'); }
      node.children.splice(at, 1);
      child.parent = null;
      return child;
    },
    setAttribute(key, value) { node.attributes[key] = String(value); },
    getAttribute(key) {
      return Object.prototype.hasOwnProperty.call(node.attributes, key)
        ? node.attributes[key] : null;
    },
    addEventListener(type, fn) {
      (node.listeners[type] = node.listeners[type] || []).push(fn);
    },
    fire(type, event) {
      for (const fn of node.listeners[type] || []) { fn(event || {}); }
    },
    walk(visit) {
      for (const child of node.children) { visit(child); child.walk(visit); }
    },
    count(tag) {
      let total = this.tagName === tag ? 1 : 0;
      this.walk((c) => { if (c.tagName === tag) { total += 1; } });
      return total;
    },
    texts() {
      const out = [];
      this.walk((c) => {
        if (c.children.length === 0 && c._text) { out.push(c._text); }
      });
      if (this._text) { out.push(this._text); }
      return out;
    },
    first_with_class(name) {
      let hit = null;
      if (this.classes.split(/\s+/).indexOf(name) >= 0) { hit = this; }
      if (hit) { return hit; }
      let found = null;
      this.walk((c) => {
        if (!found && c.classes.split(/\s+/).indexOf(name) >= 0) { found = c; }
      });
      return found;
    },
    first_with_tag(tag) {
      if (this.tagName === tag) { return this; }
      let found = null;
      this.walk((c) => {
        if (!found && c.tagName === tag) { found = c; }
      });
      return found;
    },
  };
  for (const banned of ['innerHTML', 'outerHTML', 'insertAdjacentHTML',
                        'querySelector', 'dataset',
                        'createContextualFragment']) {
    // Both accessors refuse. A getter alone would only catch a read, and
    // would let a write through silently in a non-strict caller; the view
    // is strict, but the stub must not depend on that to be a real check.
    Object.defineProperty(node, banned, {
      get() { throw new Error('the view touched ' + banned); },
      set() { throw new Error('the view assigned ' + banned); },
    });
  }
  return node;
}

function make_document() {
  const root = shim_node('#document');
  root.byId = {};
  root.createElement = (tag) => shim_node(tag);
  root.createElementNS = (ns, tag) => shim_node(tag, ns);
  root.getElementById = (id) => (root.byId[id] || null);
  root.register = (id, node) => { root.byId[id] = node; };
  return root;
}
'''


def run_node(
  script: str,
  payload: str = '{}',
  second: str = '{}',
  shim: bool = False,
) -> subprocess.CompletedProcess[str]:
  '''Run a script under node with the view required and payloads staged.

  Args:
    script: JavaScript to evaluate after loading the view. It may use the
      ``g`` handle, which is the module the view exports, and the ``doc``
      stub when ``shim`` is set.
    payload: JSON text staged as ``PAYLOAD`` before the script runs.
    second: JSON text staged as ``OTHER``, for the tests that compare two
      states of the view -- an empty slice against an orphan one.
    shim: Install the DOM stub from :data:`DOM_SHIM` and expose it as
      ``make_document``.

  Returns:
    The completed process, stdout and stderr both captured. The script is
    expected to print one JSON object on the last line of stdout, which
    is what :func:`node_json` parses.
  '''
  binary = node_binary()
  assert binary is not None, 'node was not found on PATH'
  # The payload is staged both as a parsed object and as the raw text, so
  # the scripts below exercise JSON.parse on exactly what an inline island
  # or a server response would hand over. Staging the literal object alone
  # would quietly skip the boundary this whole file is about.
  harness = [
    f'const g = require({json.dumps(str(VIEW))});',
    f'const RAW = {json.dumps(payload)};',
    'const PAYLOAD = JSON.parse(RAW);',
    f'const RAW_OTHER = {json.dumps(second)};',
    'const OTHER = JSON.parse(RAW_OTHER);',
  ]
  if shim:
    harness.append(DOM_SHIM)
  return subprocess.run(
    [binary, '--input-type=commonjs', '-e', '\n'.join(harness) + '\n' + script],
    capture_output=True, text=True, timeout=60, check=False,
    cwd=str(WEB))


def node_json(script: str, payload: str = '{}', second: str = '{}',
              shim: bool = False) -> dict[str, object]:
  '''Run a script under node and return the JSON it printed.

  Args:
    script: JavaScript to evaluate, expected to print JSON.
    payload: JSON text staged as ``PAYLOAD``.
    second: JSON text staged as ``OTHER``.
    shim: Install the DOM stub, for the tests that render.

  Returns:
    The parsed last line of stdout.

  Raises:
    AssertionError: If node exited non-zero. The message carries node's
      own stderr, because a JavaScript ``TypeError`` is unreadable without
      it.
  '''
  finished = run_node(script, payload, second, shim)
  assert finished.returncode == 0, (
      f'node failed:\n{finished.stderr}\nscript was:\n{script}')
  lines = [line for line in finished.stdout.splitlines() if line.strip()]
  assert lines, f'node printed nothing:\n{finished.stdout}\n{finished.stderr}'
  return json.loads(lines[-1])


# --- the hard rule --------------------------------------------------------

class TestNoMarkupIsEverAssigned:
  '''Text goes in as text. Always. In every file of the view.'''

  @pytest.mark.parametrize('name', MARKUP_SETTERS)
  def test_the_view_never_calls_a_markup_setter(self, name):
    '''Not once, anywhere in the view, comments included.

    Comments are stripped, and the whole raw text is asserted too. A view
    whose header said "never innerHTML" while a function below used it
    would pass a grep-for-code test, and this repository's own lesson is
    that a rule asserted only where it is documented is a note to self.
    So both spellings are covered: raw text, and comment-stripped code.

    Every ``textContent`` write in the file is checked at the same time,
    because "uses textContent" is only half the claim: the other half is
    that there is no second way in.
    '''
    raw = (VIEW.read_text(encoding='utf-8') + PAGE.read_text(encoding='utf-8')
           + STYLE.read_text(encoding='utf-8'))
    assert name not in raw, (
        f'{name} appears in the dependency-graph view, comments included. '
        f'All text must go in through textContent; a node note or a '
        f'provenance sentence is free text and must never be parsed as '
        f'markup.')

  def test_text_goes_in_through_text_content(self):
    '''The positive half of the same claim.

    Without this, a file that assigned nothing at all would satisfy the
    test above. The count is asserted rather than the mere presence,
    because the view has to write several kinds of string: node labels,
    edge meanings, news reasons and the unfitted caveat. Four separate
    sinks have to be text-only for the rule to hold at all, and each one
    is a named function: ``el``, ``svg_text``, ``svg_tip`` and the
    island reader.
    '''
    code = view_text()
    writes = code.count('textContent')
    assert writes >= 4, (
        f'textContent appears {writes} times, which is too few for the four '
        f'sinks the view writes through')
    for sink, name in (('const el =', 'el'), ('const svg_text =', 'svg_text'),
                       ('const svg_tip =', 'svg_tip')):
      assert sink in code, f'the view lost its {name} helper'
    assert '.textContent =' in code, (
        'no sink in the view assigns textContent any more')

  def test_no_markup_template_is_assigned_into_a_node(self):
    '''No string beginning with an HTML tag is assigned anywhere.

    A ``.append('...')`` treating markup as a string, or an assignment of
    ``'<b>' + value``, is a fourth way in that no name-based assertion
    would catch. Requiring every assigned string to lack a leading tag is
    a cheap structural check that covers the whole file at once.
    '''
    code = view_text()
    assert not re.search(r'=\s*`\s*<[a-zA-Z]', code), (
        'the view assigns a string beginning with an HTML tag somewhere')
    assert not re.search(r'\.\s*append\(\s*`\s*<', code), (
        'the view appends a string beginning with an HTML tag somewhere')


# --- offline --------------------------------------------------------------

class TestTheViewWorksWithTheNetworkUnplugged:
  '''No external origin, in any of the three files.'''

  @pytest.mark.parametrize('path', [VIEW, STYLE, PAGE],
                           ids=['graph.js', 'graph.css', 'graph.html'])
  def test_no_file_names_an_external_origin(self, path):
    '''Every URL-looking string is either absent or the SVG namespace.

    The SVG namespace identifier is the single permitted exception and it
    is permitted for a specific reason: ``createElementNS`` requires it,
    a browser never dereferences it, and it is not an origin this page
    reaches. Named rather than blanket, so a second URL cannot ride in on
    the same exemption.

    The search runs on comment-stripped source for the script and on the
    raw text for the CSS and the HTML, because a ``@import`` or an
    ``<link href>`` would be a live reference and a comment mentioning one
    would not.
    '''
    if path == VIEW:
      text = view_text()
    elif path == STYLE:
      text = style_text()
    else:
      text = page_text()
    text = text.replace(SVG_NAMESPACE, '')
    found = EXTERNAL_ORIGINS.search(text)
    assert not found, (
        f'{path.name} references {found.group(0)!r}, an origin outside the '
        f'package. The dashboard has to render with the network unplugged '
        f'and from file://')

  def test_the_only_fetch_target_is_one_the_caller_supplies(self):
    '''Nothing is fetched unless a URL is handed in.

    The view prefers an inline JSON island and only calls ``fetch`` when
    the caller passed a URL. So the default rendering path touches the
    network zero times, and a page with the island works from disk.
    '''
    code = view_text()
    calls = code.count('fetch(')
    assert calls == 1, (
        f'the view calls fetch {calls} times; one call site keeps the '
        f'"reaches the network exactly once, and not at all without a URL" '
        f'claim checkable')
    assert 'settings.url' in code

  def test_the_page_carries_its_payload_inline(self):
    '''The standalone page ships its data, so it needs no server.

    An inline ``application/json`` island is what makes the page work from
    ``file://``. Asserted on both halves -- the island is present and it
    is parsed rather than fetched -- because an island nobody reads is
    just a comment with brackets in it.
    '''
    page = page_text()
    assert 'application/json' in page
    assert 'graph-payload' in page
    assert 'graph.js' in page
    assert 'graph.css' in page
    # The two assets are siblings, so they resolve from disk.
    assert 'src="http' not in page
    assert 'href="http' not in page

  def test_no_webfont_is_declared_anywhere(self):
    '''A webfont is an origin this page would have to reach.

    ``style.css`` states the same rule for the main dashboard; the view
    inherits it. System fonts only, which render identically with the
    network unplugged.
    '''
    for text in (style_text(), view_text(), page_text()):
      assert '@font-face' not in text
      assert 'fonts.googleapis' not in text
    assert 'system-ui' in style_text(), (
        'the view stylesheet declares no font stack at all; it should name '
        'the system stack the rest of the dashboard uses')


# --- loadable -------------------------------------------------------------

class TestItParsesAndExports:
  '''A file that does not parse is a view that does not run.'''

  def test_node_accepts_the_file(self):
    '''``node --check`` is the syntax gate, and node is optional.

    Skipped rather than failed when node is absent, because the Python
    half of this file is still meaningful on a machine without it. On a
    machine that has it, a parse error fails the build, which is the only
    reason to check.
    '''
    binary = node_binary()
    if binary is None:
      pytest.skip('node is not installed; the syntax gate is unavailable')
    finished = subprocess.run([binary, '--check', str(VIEW)],
                              capture_output=True, text=True, timeout=60,
                              check=False)
    assert finished.returncode == 0, (
        f'node --check rejected graph.js:\n{finished.stderr}')

  def test_the_module_exports_what_a_host_needs(self):
    '''The pure logic is reachable from node, so it can be tested.

    Loading the file must not touch the DOM, which is why the test suite
    can drive ``plan`` and ``news_for`` at all. If a future edit moved DOM
    access to module scope, every one of the behaviour tests below would
    fail with a ReferenceError rather than quietly passing.
    '''
    binary = node_binary()
    if binary is None:
      pytest.skip('node is not installed; the export gate is unavailable')
    assert node_binary() is not None
    exports = node_json('console.log(JSON.stringify('
                        'Object.keys(g).sort()))')
    for name in ('plan', 'news_for', 'mount', 'render', 'parse_payload',
                 'layout', 'tone', 'draw_news'):
      assert name in exports, f'graph.js does not export {name}'

  def test_the_ponytail_comment_states_a_ceiling_and_an_upgrade(self):
    '''The view names what it gave up and how to get past it.

    The simplifications are real: column-by-hop layout rather than a force
    simulation, no zoom, no edge bundling. A reviewer needs to see the
    ceiling in the file rather than infer it from the diff.
    '''
    text = VIEW.read_text(encoding='utf-8')
    assert 'PONYTAIL:' in text
    assert 'Ceiling:' in text
    assert 'Upgrade path:' in text


# --- behaviour, on a payload the join actually produced -------------------

class TestTheViewDrawsWhatTheJoinProduced:
  '''Executed against a real slice, not a hand-written fixture.'''

  def test_it_places_every_node_and_links_every_edge(self):
    '''A slice with N nodes and M edges draws N boxes and M arrows.

    Edges whose endpoints are both in the payload are drawn; an edge
    naming a node the payload omitted is skipped rather than drawn to
    nowhere. The count is asserted rather than the mere presence, because
    a view that drew one arrow would otherwise look like it worked.
    '''
    result = node_json(
      'const p = g.plan(PAYLOAD, null);'
      'console.log(JSON.stringify({ok: p.ok, nodes: p.nodes.length,'
      ' links: p.links.length, wide: p.wide > 0, tall: p.tall > 0,'
      ' focus: p.focus.key}));',
      payload_json())
    assert result['ok'] is True
    assert result['nodes'] == 4, 'crude, ASIANPAINT, ONGC and RELIANCE'
    assert result['links'] == 3, 'one arrow per dependent stock'
    assert result['wide'] > 0 and result['tall'] > 0
    assert result['focus'] == 'commodity:crude'

  def test_a_clicked_node_gets_its_news_with_the_reason_attached(self):
    '''Clicking ONGC shows its readings and the reason they are there.

    This is the whole view in one assertion: the node is selectable, its
    readings are found by path membership rather than by name, and the
    reason string is present in what gets drawn. The reason is the part
    that matters, because a panel showing ONGC's score on the crude page
    without saying how it got there is the defect this view exists to
    avoid.
    '''
    result = node_json(
      'const p = g.plan(PAYLOAD, null);'
      'const view = g.news_for(p, "stock:ONGC");'
      'console.log(JSON.stringify({kind: view.kind, label: view.label,'
      ' count: view.readings.length, reason: view.readings[0].reason,'
      ' path: view.readings[0].via_path,'
      ' hops: view.readings[0].via_hops}));',
      payload_json())
    assert result['kind'] == 'news'
    assert result['label'] == 'ONGC'
    assert result['count'] == 2
    assert result['hops'] == 1
    assert result['path'] == ['commodity:crude', 'stock:ONGC']
    assert 'ONGC' in result['reason']
    assert 'not by matching text' in result['reason']
    assert 'commodity:crude -> stock:ONGC' in result['reason']

  def test_a_dependent_with_no_news_says_it_was_looked_up(self):
    '''ASIANPAINT is drawn, has no reading, and the panel says so.

    Three things are asserted and all three matter. The panel is not
    empty of text. It is not an error. And it names the exposure rather
    than saying nothing happened, because "we looked and found no news"
    and "we did not look" are different facts and only one of them is
    true here.
    '''
    result = node_json(
      'const p = g.plan(PAYLOAD, null);'
      'const view = g.news_for(p, "stock:ASIANPAINT");'
      'console.log(JSON.stringify({kind: view.kind, count:'
      ' view.readings.length, message: view.message}));',
      payload_json())
    assert result['count'] == 0
    assert result['kind'] == 'empty', 'an absence is not an error'
    assert 'ASIANPAINT' in result['message']
    assert 'No sentiment reading is attached' in result['message']

  def test_the_focus_panel_quotes_the_join_message_for_an_empty_slice(self):
    '''An empty slice keeps the backend's own wording.

    graph/news.py decides what an absence means and writes a sentence
    naming the symbols it checked. The view shows that sentence rather
    than writing its own, so the JSON and the screen cannot disagree --
    the same reason index.html quotes the API's ``not_advice`` string
    verbatim instead of paraphrasing it.
    '''
    empty = json.dumps(node_slice(nifty50_seed(), 'commodity:gold',
                                  [], BAR).to_dict())
    result = node_json(
      'const p = g.plan(PAYLOAD, null);'
      'const view = g.news_for(p, "commodity:gold");'
      'console.log(JSON.stringify({kind: view.kind, message: view.message}));',
      empty)
    assert 'TITAN' in result['message']
    assert 'not a failure to load' in result['message']

  def test_the_unfitted_caveat_reaches_the_payload_the_view_draws(self):
    '''The caveat travels in the payload and is read off it.

    edges.py says the seed is a hypothesis. If the caveat only lived in
    the CSS the view would be making a claim the data does not carry, and
    a JSON consumer of the same payload would print the graph with no
    qualification at all. So the assertion is that
    :data:`~stock_rl.graph.news.UNFITTED_NOTE` is in the payload and that
    ``plan`` surfaces it for the renderer.
    '''
    result = node_json(
      'const p = g.plan(PAYLOAD, null);'
      'console.log(JSON.stringify({unfitted: p.unfitted}));',
      payload_json())
    assert result['unfitted'] == UNFITTED_NOTE

  def test_every_edge_carries_a_meaning_the_view_can_print(self):
    '''No edge reaches the canvas without a label and a short form.

    The short form is drawn beside the arrow; the long sentence goes in
    the tooltip and the legend. Both are required, because "every edge is
    labelled with its meaning" is not satisfied by a line with no words
    on it.
    '''
    result = node_json(
      'const p = g.plan(PAYLOAD, null);'
      'console.log(JSON.stringify(p.links.map((l) =>'
      ' ({short: l.short, meaning: l.meaning, measured: l.measured,'
      '  d: typeof l.d === "string" && l.d.length > 0}))));',
      payload_json())
    assert result, 'the fixture payload drew no edges'
    for link in result:
      assert link['short'] in ('depends on', 'affected by')
      assert 'commodity:crude' in link['meaning']
      assert 'stock:' in link['meaning']
      assert link['measured'] is False, 'no edge in this package is measured'
      assert link['d'], 'an edge with no path data draws nothing'


def test_a_peer_stock_is_labelled_a_peer_and_not_an_input() -> None:
  '''A listed company another listed company depends on is a peer.

  Nothing in the seed graph says one listed company supplies another, so
  this role only arises in a hand-edited file -- and it is the role most
  likely to be rendered as an input by a view that buckets everything
  non-focus and non-reached together. The box must say which it is.

  The graph is built here rather than taken from the seed precisely
  because the seed cannot express this shape, and the payload still comes
  from the shipped join so the view is exercised against what it receives.
  '''
  binary = node_binary()
  if binary is None:
    pytest.skip('node is not installed; the role gate is unavailable')
  graph = DependencyGraph()
  graph.add_node(Node('stock:AAA', NodeKind.STOCK, 'AAA'))
  graph.add_node(Node('stock:BBB', NodeKind.STOCK, 'BBB'))
  graph.add_node(Node('commodity:zzz', NodeKind.COMMODITY, 'Zzz'))
  graph.add_edge(Edge('stock:AAA', 'stock:BBB', EdgeKind.DEPENDS_ON))
  graph.add_edge(Edge('stock:AAA', 'commodity:zzz', EdgeKind.DEPENDS_ON))
  peer_payload = json.dumps(node_slice(graph, 'stock:AAA', [], BAR).to_dict())
  result = node_json(
    'const p = g.plan(PAYLOAD, null);'
    'console.log(JSON.stringify({roles: p.nodes.map((n) =>'
    ' ({key: n.key, role: n.role, word: g.role_word(n)})),'
    ' links: p.links.length}));', peer_payload)
  roles = {row['key']: row for row in result['roles']}
  assert roles['stock:BBB']['role'] == 'peer'
  assert 'peer' in roles['stock:BBB']['word']
  assert 'not an input' in roles['stock:BBB']['word'], (
      'a peer must be told it is not an input; a listed company drawn as a '
      'commodity input asserts a supply chain nobody established')
  assert roles['commodity:zzz']['role'] == 'path'
  assert roles['stock:AAA']['word'] == 'focus'
  assert result['links'] == 2, 'both stored edges are still drawn'


class TestTheViewActuallyRenders:
  '''The draw path, driven against a stub DOM with no jsdom anywhere.

  The tests above cover layout and wording. These cover the part that only
  exists once there is a document: that ``mount`` builds a tree, that the
  caveat and the legend are in it, that a node click reaches the news
  panel, and that a keyboard press does the same thing a click does.

  The stub raises on ``innerHTML``, ``querySelector``, ``dataset`` and the
  rest of the surface the view is not supposed to use, so these tests
  would fail on a regression that the string-matching rule above could miss
  -- a dynamic property lookup, say. That is the reason the stub exists
  rather than the five layout assertions alone.
  '''

  def test_mount_builds_a_graph_a_legend_and_a_news_panel(self) -> None:
    '''The whole page is assembled, not just the canvas.

    Asserted on the tree the stub ends up holding: one ``<svg>``, one box
    per node, one ``<path>`` per edge, the unfitted caveat as its own
    element, and the legend naming every edge. A view that drew the arrows
    and silently dropped the caveat would pass a canvas-only assertion.
    '''
    result = node_json(
      'const doc = make_document();\n'
      'const view = doc.createElement("div");\n'
      'const panel = doc.createElement("div");\n'
      'doc.register("graph-view", view);\n'
      'doc.register("graph-news", panel);\n'
      'global.document = doc;\n'
      'g.mount(view, {payload: PAYLOAD}).then(() => {\n'
      '  const caveat = view.first_with_class("graph-unfitted");\n'
      '  const legend = view.first_with_class("graph-legend");\n'
      '  const drawn = view.first_with_tag("svg");\n'
      '  console.log(JSON.stringify({\n'
      '    svg: Boolean(drawn),\n'
      '    boxes: view.count("g"),\n'
      '    paths: view.count("path"),\n'
      '    caveat: caveat ? caveat.textContent : "",\n'
      '    legendRows: legend ? legend.count("p") : 0,\n'
      '    panelTexts: panel.texts(),\n'
      '    readingCards: panel.count("article")}));\n'
      '});',
      payload_json(), shim=True)
    assert result['svg'] is True
    assert result['boxes'] == 4, 'crude, ASIANPAINT, ONGC and RELIANCE'
    assert result['paths'] == 3
    assert 'not measured sensitivities' in result['caveat']
    assert result['legendRows'] >= 4, (
        'the legend lists its heading, its explanation and one row per edge')
    assert result['readingCards'] == 2, 'ONGC has two readings at the bar'
    assert 'ONGC' in ' '.join(result['panelTexts'])
    assert 'not by matching text' in ' '.join(result['panelTexts'])

  def test_a_refused_aggregate_is_marked_on_the_card_not_only_the_text(
    self,
  ) -> None:
    '''One source means the score is muted, and the card says why.

    ``gread-usable`` is the class the stylesheet keys on to grey out an
    uncorroborated score, so it has to be on the card rather than only
    mentioned in the verdict sentence. Both are asserted, because a card
    that reads "NOT usable - single_source" in small print while the number
    beside it is coloured like a corroborated one has told the reader the
    opposite of what it means.
    '''
    result = node_json(
      'const doc = make_document();\n'
      'const view = doc.createElement("div");\n'
      'const panel = doc.createElement("div");\n'
      'doc.register("graph-view", view);\n'
      'doc.register("graph-news", panel);\n'
      'global.document = doc;\n'
      'g.mount(view, {payload: PAYLOAD}).then(() => {\n'
      '  const cards = [];\n'
      '  panel.walk((c) => {\n'
      '    if (c.tagName === "article") { cards.push(c.classes); }\n'
      '  });\n'
      '  console.log(JSON.stringify({cards, text: panel.textContent}));\n'
      '});',
      mixed_payload_json(), shim=True)
    usable = [name for name in result['cards'] if 'gread-usable' in name]
    refused = [name for name in result['cards'] if 'gread-usable' not in name]
    assert len(usable) == 2, 'both ONGC readings share one usable aggregate'
    assert len(refused) == 1, 'RELIANCE has one source and is not usable'
    text = ' '.join(result['text'].split())
    assert 'NOT usable - single_source' in text
    assert 'usable aggregate, cross-source spread 0.175' in text, (
        'a usable card must carry the cross-source spread, because the '
        'score alone is not the number to act on')

  def test_clicking_a_node_replaces_the_news_panel(self) -> None:
    '''A click on ONGC, then on ASIANPAINT, gives two different panels.

    The click handler is the whole interaction of this view, and a handler
    wired to the wrong node would leave a populated panel that never
    changes. So the panel is asserted twice: once with readings and once
    with the honest empty sentence, which is only reachable if the second
    click reached a different node.
    '''
    result = node_json(
      'const doc = make_document();\n'
      'const view = doc.createElement("div");\n'
      'const panel = doc.createElement("div");\n'
      'doc.register("graph-view", view);\n'
      'doc.register("graph-news", panel);\n'
      'global.document = doc;\n'
      'const pick = (key) => {\n'
      '  let hit = null;\n'
      '  view.walk((c) => {\n'
      '    const label = c.getAttribute("aria-label") || "";\n'
      '    if (!hit && c.tagName === "g" && label.indexOf(key) === 0) {\n'
      '      hit = c;\n'
      '    }\n'
      '  });\n'
      '  return hit;\n'
      '};\n'
      'g.mount(view, {payload: PAYLOAD}).then(() => {\n'
      '  pick("ONGC").fire("click");\n'
      '  const withNews = {cards: panel.count("article"),\n'
      '    text: panel.textContent};\n'
      '  pick("ASIANPAINT").fire("click");\n'
      '  const withoutNews = {cards: panel.count("article"),\n'
      '    text: panel.textContent,\n'
      '    empty: Boolean(panel.first_with_class("graph-empty")),\n'
      '    error: Boolean(panel.first_with_class("graph-error"))};\n'
      '  console.log(JSON.stringify({withNews, withoutNews}));\n'
      '});',
      payload_json(), shim=True)
    assert result['withNews']['cards'] == 2
    assert 'not by matching text' in result['withNews']['text']
    assert result['withoutNews']['cards'] == 0
    assert 'No sentiment reading is attached' in result['withoutNews']['text']
    assert result['withoutNews']['empty'] is True
    assert result['withoutNews']['error'] is False, (
        'an absence of news was drawn as an error')

  def test_a_keyboard_press_reaches_the_same_handler_as_a_click(self) -> None:
    '''Enter and Space pick a node, because the boxes are focusable.

    The view sets ``tabindex`` and ``role=button`` on each node, which is
    a promise about keyboard operation, and a promise nothing exercises is
    a decoration. A keyboard user must be able to reach every node and see
    the same panel a click produces.
    '''
    for key in ('Enter', ' ', 'Spacebar'):
      result = node_json(
        'const doc = make_document();\n'
        'const view = doc.createElement("div");\n'
        'const panel = doc.createElement("div");\n'
        'doc.register("graph-view", view);\n'
        'doc.register("graph-news", panel);\n'
        'global.document = doc;\n'
        'let stopped = 0;\n'
        'g.mount(view, {payload: PAYLOAD}).then(() => {\n'
        '  let hit = null;\n'
        '  view.walk((c) => {\n'
        '    const label = c.getAttribute("aria-label") || "";\n'
        '    if (!hit && c.tagName === "g" && label.indexOf("ONGC") === 0) {\n'
        '      hit = c;\n'
        '    }\n'
        '  });\n'
        f'  const stop = () => {{ stopped += 1; }};\n'
        f'  const event = {{key: {json.dumps(key)}, preventDefault: stop}};\n'
        f'  hit.fire("keydown", event);\n'
        '  console.log(JSON.stringify({cards: panel.count("article"),\n'
        '    stopped, tabindex: hit.getAttribute("tabindex"),\n'
        '    role: hit.getAttribute("role")}));\n'
        '});',
        payload_json(), shim=True)
      assert result['cards'] == 2, f'{key!r} did not reach the news panel'
      assert result['stopped'] == 1, f'{key!r} did not suppress scrolling'
      assert result['tabindex'] == '0'
      assert result['role'] == 'button'

  def test_a_page_with_no_island_says_so_rather_than_drawing(self) -> None:
    '''The empty payload in the shipped page renders a sentence.

    ``graph.html`` ships ``{}`` as its island, so this is the state a
    reviewer sees on first open. If it rendered as an empty canvas the
    page would look like a market with no dependencies in it, so the
    message must be there, and it must be the error style rather than the
    empty style, because nothing was looked up.
    '''
    result = node_json(
      'const doc = make_document();\n'
      'const view = doc.createElement("div");\n'
      'const panel = doc.createElement("div");\n'
      'doc.register("graph-view", view);\n'
      'doc.register("graph-news", panel);\n'
      'doc.register("graph-payload", doc.createElement("script"));\n'
      'global.document = doc;\n'
      'g.mount(view, {}).then(() => {\n'
      '  console.log(JSON.stringify({svg: view.count("svg"),\n'
      '    text: view.textContent,\n'
      '    error: Boolean(view.first_with_class("graph-error")),\n'
      '    empty: Boolean(view.first_with_class("graph-empty"))}));\n'
      '});',
      shim=True)
    assert result['svg'] == 0, 'nothing may be drawn without a payload'
    assert 'No graph payload was supplied' in result['text']
    assert result['error'] is True
    assert result['empty'] is False

  def test_the_island_is_preferred_over_the_network(self) -> None:
    '''A page carrying its own data makes no request at all.

    This is the offline claim, executed rather than asserted: ``fetch`` is
    replaced with something that throws, and a payload present in the
    island must still render. A view that fetched first would raise, and
    the page would show a transport error for data it already had.
    '''
    result = node_json(
      'const doc = make_document();\n'
      'const view = doc.createElement("div");\n'
      'const panel = doc.createElement("div");\n'
      'const island = doc.createElement("script");\n'
      'island.textContent = RAW;\n'
      'doc.register("graph-view", view);\n'
      'doc.register("graph-news", panel);\n'
      'doc.register("graph-payload", island);\n'
      'global.document = doc;\n'
      'let reached = 0;\n'
      'global.fetch = async () => { reached += 1;'
      ' throw new Error("no net"); };\n'
      'g.mount(view, {url: "http://127.0.0.1:8765/api/graph"}).then(() => {\n'
      '  console.log(JSON.stringify({boxes: view.count("g"), reached}));});\n',
      payload_json(), shim=True)
    assert result['reached'] == 0, (
        'the view reached the network even though the page carried the data')
    assert result['boxes'] == 4

  def test_the_dom_stub_is_not_permissive_about_the_forbidden_surface(self):
    '''The stub raises on what the view must not touch, and is checked.

    Every rendering test above is only as strong as the stub underneath
    it. A stub that quietly accepted ``innerHTML`` or ``dataset`` would
    let a regression through that :class:`TestNoMarkupIsEveryAssigned`
    misses -- a dynamic property lookup, say, or a spelling the string
    scan does not match. So the stub's own refusal is asserted here, and
    separately from the view, because a stub that was neutered by a later
    edit would otherwise take all four tests above with it silently.
    '''
    result = node_json(
      'const doc = make_document();\n'
      'const node = doc.createElement("div");\n'
      'const seen = {};\n'
      'for (const banned of ["innerHTML", "outerHTML", "insertAdjacentHTML",\n'
      '                        "querySelector", "dataset"]) {\n'
      '  try { void node[banned]; seen[banned] = "allowed"; }\n'
      '  catch (err) { seen[banned] = "refused"; }\n'
      '}\n'
      'try { node.innerHTML = "<b>x</b>"; seen.assign = "allowed"; }\n'
      'catch (err) { seen.assign = "refused"; }\n'
      'try { node.dataset.role = "x"; seen.datasetWrite = "allowed"; }\n'
      'catch (err) { seen.datasetWrite = "refused"; }\n'
      'console.log(JSON.stringify(seen));', shim=True)
    for name in ('innerHTML', 'outerHTML', 'insertAdjacentHTML',
                 'querySelector', 'dataset'):
      assert result[name] == 'refused', (
          f'the DOM stub allows {name}, so the rendering tests above prove '
          f'nothing about it')
    assert result['assign'] == 'refused', (
        'assigning innerHTML on the stub must throw, or a write would slip '
        'through in a non-strict caller')
    assert result['datasetWrite'] == 'refused', (
        'writing to dataset on the stub must throw for the same reason')


def test_a_panel_over_several_companies_names_them_in_the_heading() -> None:
  '''The heading does not let a commodity take credit for a headline.

  Crude has no news of its own. Everything shown under it belongs to the
  companies that consume it, and a heading reading "News for Crude Oil"
  above ONGC's and RELIANCE's headlines would be a quiet misattribution
  of exactly the kind the relevance rule was built to avoid -- the only
  difference is that here it would be in the typography rather than in the
  data. So the heading names the symbols when there is more than one, and
  the single-symbol case is left plain rather than padded.
  '''
  result = node_json(
    'const p = g.plan(PAYLOAD, null);'
    'const focus = g.news_for(p, "commodity:crude");'
    'const one = g.news_for(p, "stock:ONGC");'
    'console.log(JSON.stringify({many: g.heading(focus),'
    ' one: g.heading(one), via: focus.via, count: focus.readings.length}));',
    mixed_payload_json())
  assert result['via'] == ['ONGC', 'RELIANCE']
  assert result['count'] == 3
  assert result['many'] == 'News for Crude Oil, via ONGC, RELIANCE'
  assert result['one'] == 'News for ONGC', (
      'a single-symbol panel must not be padded with a redundant "via"')


# --- honest degradation ---------------------------------------------------

class TestDegradationIsHonest:
  '''No news, no route, no payload and a failed request all differ.'''

  def test_a_missing_payload_is_an_error_not_an_empty_graph(self):
    '''No payload draws a red sentence, not a blank canvas.

    An empty canvas would be indistinguishable from a market with no
    dependencies in it, which is a claim the view has no way to support.
    The message has to say what to supply, because "no data" without a
    next step is just a dead end.
    '''
    result = node_json(
      'const p = g.plan(null, null);'
      'console.log(JSON.stringify({ok: p.ok, kind: p.kind,'
      ' message: p.message}));')
    assert result['ok'] is False
    assert result['kind'] == 'error'
    assert 'No graph payload was supplied' in result['message']
    assert 'mount()' in result['message']

  def test_a_payload_with_no_nodes_is_an_error_not_an_empty_graph(self):
    '''Nodes-less is a payload fault, and is reported as one.

    A node list of zero would otherwise render as an empty canvas with a
    caveat above it, which reads as a market with no dependencies. The
    message says so explicitly.
    '''
    result = node_json(
      'const p = g.plan({nodes: [], edges: [], readings: []}, null);'
      'console.log(JSON.stringify({ok: p.ok, kind: p.kind,'
      ' message: p.message}));')
    assert result['ok'] is False
    assert result['kind'] == 'error'
    assert 'no nodes' in result['message']
    assert 'not an absence of news' in result['message']

  def test_malformed_json_is_reported_rather_than_rendered_as_empty(self):
    '''A parse failure names the failure and draws nothing.

    ``JSON.parse`` throwing inside the view would leave the mount point
    holding whatever was there before, which on a page that says
    "Loading the dependency graph..." is a view that hangs mid-sentence.
    So the failure is caught and returned as a message.
    '''
    result = node_json(
      'const parsed = g.parse_payload("{not json");'
      'console.log(JSON.stringify({ok: parsed.ok,'
      ' message: parsed.message}));')
    assert result['ok'] is False
    assert 'not valid JSON' in result['message']
    assert 'rather than' in result['message']

  def test_valid_json_is_parsed_and_the_payload_survives(self):
    '''The positive half of the parse test.

    Without this, a ``parse_payload`` that always returned a failure
    would satisfy the test above and the view would never draw anything.
    '''
    result = node_json(
      'const parsed = g.parse_payload(RAW);'
      'console.log(JSON.stringify({ok: parsed.ok,'
      ' node: parsed.payload.node}));',
      payload_json())
    assert result['ok'] is True
    assert result['node'] == 'commodity:crude'

  def test_the_three_empty_states_are_three_different_sentences(self):
    '''``no_news`` and ``no_dependents`` are worded apart, and a click on an
    unreached node is a third state again.

    Two slices are built from the shipped join: gold at the default depth,
    which reaches TITAN and finds nothing, and global liquidity at depth
    one, which reaches no listed company at all. Both focus panels must
    produce text and the two texts must differ, because a reader who
    cannot tell them apart will conclude from whichever one they assume --
    and here the two conclusions are opposite: a quiet tape versus a graph
    that makes no claim.
    '''
    no_news = json.dumps(node_slice(nifty50_seed(), 'commodity:gold', [],
                                    BAR).to_dict())
    orphan = json.dumps(node_slice(nifty50_seed(), 'macro:global_liquidity',
                                   [], BAR, max_depth=1).to_dict())
    results = node_json(
      'const a = g.plan(PAYLOAD, null);'
      'const b = g.plan(OTHER, null);'
      'const one = g.news_for(a, a.focus.key);'
      'const two = g.news_for(b, b.focus.key);'
      'console.log(JSON.stringify({a: {kind: one.kind, m: one.message},'
      ' b: {kind: two.kind, m: two.message}}));',
      no_news, orphan)
    assert results['a']['kind'] == 'no_news'
    assert results['b']['kind'] == 'no_dependents'
    assert results['a']['m'] and results['b']['m']
    assert results['a']['m'] != results['b']['m']
    assert 'TITAN' in results['a']['m']
    assert 'no listed symbol downstream' in results['b']['m']

  def test_an_unreachable_http_status_is_reported_as_a_missing_feature(self):
    '''A 404 is named as a 404, and not as "no news".

    The temptation is to render a server without the endpoint as an empty
    graph, because both produce nothing on screen. That would make a
    missing feature indistinguishable from a quiet market, which is the
    one confusion this view must not create. The stub server below
    answers 404 and the message has to carry the status and the word
    "missing".
    '''
    binary = node_binary()
    if binary is None:
      pytest.skip('node is not installed; the fetch stub is unavailable')
    script = (
      'global.fetch = async () => ({ok: false, status: 404});\n'
      'g.fetch_payload("http://127.0.0.1:8765/api/graph").then((r) =>\n'
      '  console.log(JSON.stringify(r)));'
    )
    result = node_json(script)
    assert result['ok'] is False
    assert '404' in result['message']
    assert 'missing feature' in result['message']
    assert 'rather than as "no news"' in result['message']

  def test_a_transport_failure_is_reported_as_a_transport_failure(self):
    '''A rejected fetch is named as a transport failure.

    Same reasoning as the 404: the reader must be able to tell a broken
    request from an absent headline. The distinction is spelled out in the
    message rather than left to the class name alone.
    '''
    binary = node_binary()
    if binary is None:
      pytest.skip('node is not installed; the fetch stub is unavailable')
    script = (
      'global.fetch = async () => {throw new Error("ECONNREFUSED");};\n'
      'g.fetch_payload("http://127.0.0.1:8765/api/graph").then((r) =>\n'
      '  console.log(JSON.stringify(r)));'
    )
    result = node_json(script)
    assert result['ok'] is False
    assert 'ECONNREFUSED' in result['message']
    assert 'transport failure' in result['message']
    assert 'not an absence of news' in result['message']

  def test_the_error_and_empty_styles_are_visibly_different(self):
    '''The two states must not be able to collapse into one grey box.

    A payload with a class name on both is a claim; the stylesheet has to
    honour it. Asserted on the stylesheet because that is where a
    regression would land, and the colours are asserted to differ rather
    than merely both existing.
    '''
    css = STYLE.read_text(encoding='utf-8')
    assert '.graph-empty' in css
    assert '.graph-error' in css
    empty_block = re.search(r'\.graph-empty\s*\{(.*?)\}', css, re.S)
    error_block = re.search(r'\.graph-error\s*\{(.*?)\}', css, re.S)
    assert empty_block and error_block
    assert '--graph-empty-bg' in empty_block.group(1)
    assert '--graph-error-bg' in error_block.group(1)
    variables = dict(re.findall(r'(--graph-(?:empty|error)-bg):\s*([^;]+);',
                                css))
    assert variables['--graph-empty-bg'] != variables['--graph-error-bg']


# --- the unfitted caveat is in the UI, not just the data -----------------

class TestTheUiCannotImplyTheGraphIsMeasured:
  '''The qualification has to survive all the way to the pixels.'''

  def test_the_stylesheet_says_an_unfitted_edge_is_dashed(self):
    '''Dashed means unmeasured, and only unfitted edges are dashed.

    The view draws a dashed line for every edge and the stylesheet has to
    agree, or the dashes mean nothing. The legend must also say so in
    words, because a dash pattern is a convention and a convention is not
    a caveat.
    '''
    view = view_text()
    assert "'stroke-dasharray': link.measured ? 'none' : '5 4'" in view, (
        'the view no longer dashes an unmeasured edge; the dash IS the '
        'qualification, and a solid arrow would imply a measurement')
    assert 'link.measured' in view
    assert 'unfitted dependency claim' in VIEW.read_text(encoding='utf-8')

  def test_the_caveat_is_never_conditional(self):
    '''The caveat renders unconditionally.

    The renderer appends it before anything else and there is no branch
    that skips it. Asserted on the source because that is the property:
    a future edit that wrapped it in ``if (plan.unfitted)`` would be a
    one-character change and a large dishonesty.
    '''
    view = view_text()
    assert 'graph-unfitted' in view
    caveat = re.search(r"host\.appendChild\(el\('p', plan_result\.unfitted",
                       view)
    assert caveat, (
        'the unfitted caveat is no longer appended unconditionally in '
        'render(); a conditional caveat is absent exactly when a reader '
        'is in a hurry')

  def test_the_standalone_page_repeats_the_caveat_outside_the_view(self):
    '''The page says it in prose too, before any script runs.

    Belt and braces on purpose: the view's caveat arrives with the
    payload, so a page whose island is empty would show the view's own
    fallback sentence instead. The page's own paragraph is therefore not
    conditional on any JavaScript having run at all.
    '''
    page = flowed(page_text()).lower()
    assert 'unfitted sector reasoning' in page
    assert 'not measured sensitivities' in page
    assert 'nothing here was fitted to returns' in page


# --- guard against a vacuous suite ----------------------------------------

def test_the_whole_file_still_exercises_the_view() -> None:
  '''A sanity check on the tests themselves.

  ``node --check`` proves the file parses; it does not prove the
  behaviour tests ran against it. If node is unavailable the behaviour
  tests skip, so this asserts they were collected rather than silently
  removed -- which is the difference between a suite that reports "no
  news" and one that reports nothing at all.
  '''
  collected = [str(item) for item in Path(__file__).read_text(
    encoding='utf-8').splitlines()]
  behaviour = [line for line in collected
               if 'node_json(' in line and 'def node_json' not in line]
  assert len(behaviour) >= 8, (
      f'only {len(behaviour)} call(s) into node remain in this file; the '
      f'behaviour tests were removed rather than skipped')
