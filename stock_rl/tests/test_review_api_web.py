#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Adversarial review of the API, the dashboard and the sentiment gate.

Every test here calls the shipped module. Nothing in this file
re-implements the arithmetic under test: a test that reproduces a
suspected calculation inline proves nothing about the code that
performs it.

The file is an executable version of the review report. Run

    uv run pytest tests/test_review_api_web.py -q --no-cov

and the red tests are the findings, each of which names the file and
line it is about in its own docstring. The green tests are the claimed
invariants that survive an attempt to break them, and they are here so
the red ones cannot be dismissed wholesale.
'''

from __future__ import annotations

import contextlib
import json
import re
import socket
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from http.client import HTTPConnection

import pytest

from stock_rl import api
from stock_rl.bars import Bar
from stock_rl.context.fuse import (
  CONTEXT_ENABLED,
  CONTEXT_WIDTH,
  ContextArm,
  SymbolContext,
  context_names,
  state_vector,
  symbol_context,
)
from stock_rl.graph import traversal
from stock_rl.graph.edges import DependencyGraph, Edge, EdgeKind, nifty50_seed
from stock_rl.graph.nodes import Node, NodeKind, make_key
from stock_rl.portfolio import run_portfolio
from stock_rl.sentiment.entity_link import LinkStatus, resolve
from stock_rl.sentiment.score import (
  Rejection,
  SentimentReading,
  aggregate,
  require_visible,
)

#: Bar date every price fixture starts from.
bar_date = date(2020, 1, 1)

#: The decision bar every look-ahead test is taken at.
decision_bar = datetime(2024, 1, 10, 15, 30, tzinfo=timezone.utc)


def rising_panel(
  symbols: Sequence[str] = ('AAA', 'BBB', 'CCC', 'DDD'),
  bars: int = 250,
  step: float = 1.02,
) -> dict[str, list[Bar]]:
  '''Return aligned panels that move by ``step`` on every bar.

  Args:
    symbols: Bare symbols to build a panel for each of.
    bars: Bars in each panel.
    step: Multiplicative step per bar. 1.0 gives a flat panel.

  Returns:
    Mapping of symbol to bars in ascending time order.
  '''
  panels: dict[str, list[Bar]] = {}
  for offset, symbol in enumerate(symbols):
    price = 100.0 + offset
    series: list[Bar] = []
    for index in range(bars):
      price *= step
      rounded = round(price, 2)
      series.append(Bar(
        bar_date + timedelta(days=index), rounded, rounded, rounded,
        rounded, 1000))
    panels[symbol] = series
  return panels


def leaked_readings() -> list[SentimentReading]:
  '''Return two readings dated three days after the decision bar.

  Returns:
    Readings from two distinct sources, so they clear the default
    source minimum, and both public only after :data:`decision_bar`.
  '''
  published = decision_bar + timedelta(days=3)
  return [
    SentimentReading('INFY', -0.9, 'earnings', 0.95, published, 'cnbc'),
    SentimentReading('INFY', -0.8, 'earnings', 0.9, published, 'reuters'),
  ]


@contextlib.contextmanager
def serving(
  service: api.ApiService | None = None,
) -> Iterator[tuple[str, int]]:
  '''Yield the address of a real loopback server.

  Args:
    service: Service to serve. A fresh empty one by default.

  Yields:
    Tuple of (host, port) on an ephemeral port.
  '''
  server = api.build_server(service or api.ApiService(), '127.0.0.1', 0)
  host, port = api.bound_address(server)
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()
  try:
    yield host, port
  finally:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def fetch(
  host: str,
  port: int,
  path: str,
  headers: Mapping[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
  '''Issue one real HTTP GET and return the whole answer.

  Args:
    host: Loopback host the server bound.
    port: Port the server bound.
    path: Request path.
    headers: Extra request headers.

  Returns:
    Tuple of (status, response headers, body bytes).
  '''
  request = urllib.request.Request(
    f'http://{host}:{port}{path}', headers=dict(headers or {}))
  try:
    with urllib.request.urlopen(request, timeout=20) as answer:
      return answer.status, dict(answer.headers), answer.read()
  except urllib.error.HTTPError as error:
    return error.code, dict(error.headers), error.read()


# --------------------------------------------------------------------------
# FINDING 1 -- the dashboard cannot load the files it references.
# api.py:1625 registers no '/app.js' and no '/style.css', while
# index.html:10 and index.html:128 reference them by relative URL.
# --------------------------------------------------------------------------

def test_served_dashboard_can_load_every_file_it_references() -> None:
  '''Every asset the served document references must be routable.

  index.html links ``style.css`` and ``app.js`` with relative URLs, so
  the browser resolves them against the serving origin and they have to
  be routes. api.py:1625 registers neither, so the document arrives
  with no stylesheet and no script and every table on the page stays
  empty for the life of the process.
  '''
  document = api.index_html()
  references = re.findall(r'(?:src|href)="([^"]+)"', document)
  assert references, 'the document references no sibling asset at all'
  with serving() as (host, port):
    status, _, _ = fetch(host, port, '/')
    assert status == 200
    for reference in references:
      assert not reference.startswith(('http://', 'https://', '//')), (
          f'{reference!r} is an external reference; the dashboard has to '
          f'be usable offline')
      sub_status, headers, body = fetch(host, port, '/' + reference)
      assert sub_status == 200, (
          f'the served page references {reference!r} but GET /{reference} '
          f'answered {sub_status}; the dashboard renders no data')
      assert body, f'{reference!r} is empty'
      assert 'json' not in headers.get('Content-Type', ''), (
          f'{reference!r} answered JSON, so what came back is a 404 body')


# --------------------------------------------------------------------------
# FINDING 2 -- a reading dated after the bar reaches the state vector.
# fuse.py:274 defaults symbol_context's decision_bar to None, which
# makes score.py:352 skip the availability filter entirely.
# --------------------------------------------------------------------------

def test_a_reading_after_the_bar_cannot_reach_the_state_vector() -> None:
  '''A future reading must not survive the documented default call.

  fuse.py:310-313 claims "a single-source or leaked reading cannot reach
  a state vector as a real-looking number". The default call
  ``symbol_context(symbol, readings)`` omits ``decision_bar``, so
  score.py:352 skips the filter and two readings dated three days after
  the bar are averaged into ``sentiment_score`` with
  ``sentiment_usable == 1.0``.
  '''
  readings = leaked_readings()
  assert all(reading.available_from > decision_bar for reading in readings)
  context = symbol_context('INFY', readings)
  assert context.sentiment_usable == 0.0, (
      f'a reading public only after the bar produced '
      f'sentiment_usable={context.sentiment_usable}')
  assert context.sentiment_score == 0.0, (
      f'the leaked mean reached the context: {context.sentiment_score}')
  price = [0.1, -0.2, 0.3]
  state = state_vector(price, context, arm=ContextArm.NUMERIC_CONTEXT,
                       enabled=True)
  offset = len(price) + context_names().index('sentiment_score')
  assert state[offset] == 0.0, (
      f'the leaked mean reached the state vector at index {offset}: '
      f'{state[offset]}')


def test_the_strict_helper_still_refuses_the_same_readings() -> None:
  '''The guard exists; prove it fires rather than assuming it does.

  score.py:269 require_visible is the loud counterpart. This is the
  control for the test above: the same readings are refused here, so
  the leak there is a missing filter on one path, not a filter that
  does not work.
  '''
  with pytest.raises(ValueError) as caught:
    require_visible(leaked_readings(), decision_bar)
  assert 'not public' in str(caught.value)


# --------------------------------------------------------------------------
# FINDING 3 -- a leaked reading is dropped but the aggregate is still
# reported usable, against the docstring at score.py:100-103.
# --------------------------------------------------------------------------

def test_aggregate_is_refused_when_any_reading_leaked() -> None:
  '''A leak must make the aggregate unusable, not merely annotate it.

  score.py:100-103 says of :attr:`Rejection.LOOK_AHEAD`: "The reading
  is dropped **and the aggregate is refused**". The implementation at
  score.py:383 computes ``usable`` from the surviving source count
  alone, so two public readings beside one leaked third return
  ``usable=True`` with a ``look_ahead`` entry in ``rejected``.
  '''
  readings = [
    SentimentReading('INFY', 0.4, 'news', 0.5, decision_bar, 'reuters'),
    SentimentReading('INFY', 0.6, 'news', 0.5, decision_bar, 'bloomberg'),
    SentimentReading('INFY', -0.9, 'news', 0.9,
                     decision_bar + timedelta(hours=1), 'cnbc'),
  ]
  result = aggregate(readings, 'INFY', decision_bar)
  assert Rejection.LOOK_AHEAD.value in result.rejected
  assert not result.usable, (
      f'the aggregate was refused according to its own enum but reports '
      f'usable=True with score={result.score}; a caller that checks only '
      f'the flag proceeds on a panel that leaked')
  assert result.score == 0.0


# --------------------------------------------------------------------------
# FINDING 4 -- GET /api/baselines measures an unstated configuration and
# disagrees with POST /api/backtest for the same strategy name.
# api.py:787 calls run_portfolio with the library defaults.
# --------------------------------------------------------------------------

def test_baselines_row_matches_the_backtest_of_the_same_request() -> None:
  '''The control arm must measure what the trigger measures.

  api.py:787 runs ``run_portfolio(panels, provider)`` with the library
  defaults, ignoring the request. The same named strategy therefore
  carries two Sharpes on one dashboard, the Baselines panel's and the
  Metrics panel's, and the Baselines payload never says which
  ``rebalance_days`` or ``history`` produced its numbers.
  '''
  request = {'strategy': 'momentum_ranked', 'rebalance_days': 5,
             'max_weight': 0.3, 'history': 10, 'capital': 10_000_000.0}
  service = api.ApiService(panels=rising_panel())
  table = json.loads(api.dispatch(service, 'GET', '/api/baselines').body)
  triggered = json.loads(
      api.dispatch(service, 'POST', '/api/backtest',
                   json.dumps(request).encode('utf-8')).body)
  row = next(item for item in table['strategies']
             if item['name'] == request['strategy'])
  assert row['status'] == 'ok'
  for name in ('capital', 'rebalance_days', 'max_weight', 'history'):
    assert name in table or name in row, (
        f'GET /api/baselines does not disclose the {name!r} it measured, '
        f'so its Sharpe cannot be attributed to a configuration')
  measured = row['sharpe']
  asked = triggered['sharpe']
  wanted = request['strategy']
  assert measured == pytest.approx(asked), (
      f'GET /api/baselines reports Sharpe {measured} for {wanted} while '
      f'POST /api/backtest reports {asked} for the same strategy under '
      f'the requested parameters')


# --------------------------------------------------------------------------
# FINDING 5 -- positions() mixes two denominators. api.py:899 computes
# cash as capital * (1 - invested); api.py:717 values each position
# against state.value.
# --------------------------------------------------------------------------

def test_positions_cash_and_holdings_share_one_denominator() -> None:
  '''cash plus the position values must add up to the reported value.

  On a book that rose from ten million to fifty-one million, api.py:899
  reports cash as sixty percent of *capital* while api.py:717 values
  each position against *value*. The payload's own three numbers
  therefore disagree by twenty-five million rupees, and app.js:457
  prints them side by side.
  '''
  service = api.ApiService(panels=rising_panel())
  service.backtest(api.BacktestRequest(
    symbols=('AAA', 'BBB', 'CCC', 'DDD'), strategy='equal_weight'))
  book = service.positions()
  holdings = sum(row['value'] for row in book['positions'])
  assert book['value'] > 4.0 * book['capital'], (
      'the fixture did not move far enough to expose the bug')
  reported_cash = book['cash']
  reported_value = book['value']
  assert holdings + reported_cash == pytest.approx(reported_value, rel=1e-6), (
      f'holdings {holdings:.2f} plus cash {reported_cash:.2f} is not the '
      f'reported value {reported_value:.2f}; cash is a share of capital '
      f'and the positions are a share of value')



# --------------------------------------------------------------------------
# FINDING 6 -- the evidence string cites the module constants, not the
# arguments. api.py:1109 and api.py:1118.
# --------------------------------------------------------------------------

def test_signal_reasons_name_the_window_that_was_actually_used() -> None:
  '''A reason must describe the computation it is evidence for.

  api.py:1118 formats ``signal_lookback`` and ``signal_skip``, the
  module constants, rather than the ``lookback`` and ``skip`` arguments
  the score was built from. Ask for a five-bar, zero-skip window on a
  forty-bar panel and the payload still claims "over 252 bars, skipping
  the most recent 21", for a panel that has no such window.
  '''
  panels = rising_panel(('AAA',), bars=40)
  rows = api.momentum_signals(panels, {}, top=2, lookback=5, skip=0)
  assert rows, 'no signal was produced'
  reasons = ' '.join(clause for row in rows for clause in row.reasons)
  assert '5 bars' in reasons, (
      f'the reasons never mention the five-bar window actually used: '
      f'{reasons!r}')
  assert 'skipping the most recent 0' in reasons, (
      f'the reasons never mention skip=0 actually used: {reasons!r}')
  assert '252 bars' not in reasons, (
      f'the reasons cite the 252-bar default for a forty-bar panel: '
      f'{reasons!r}')


# --------------------------------------------------------------------------
# FINDING 7 -- GET /api/signals carries no bar as-of date.
# api.py:660 scores the whole panel including its final bar and emits
# generated_at only.
# --------------------------------------------------------------------------

def test_signal_payload_names_the_bar_the_signal_was_measured_on() -> None:
  '''A signal built on the last close must say which bar that was.

  api.py:660 scores the entire panel including its final bar and emits
  ``generated_at`` only. Nothing in the payload ties the action to a
  bar date, so a signal computed on a bar whose close is not yet
  tradeable is indistinguishable from one computed on yesterday's, and
  ``held_weight`` is compared against it regardless.
  '''
  service = api.ApiService(panels=rising_panel(bars=400))
  payload = service.signals()
  assert 'as_of' in payload or 'bar_date' in payload or 'last_bar' in payload, (
      'the signal payload names no bar as-of, so the action cannot be '
      'attributed to a date')


# --------------------------------------------------------------------------
# FINDING 8 -- the entity linker resolves a query that is not a prefix.
# entity_link.py:337, the ``compact.startswith(lowered)`` clause.
# --------------------------------------------------------------------------

def test_entity_linker_refuses_a_query_that_is_not_a_prefix() -> None:
  '''A superstring of a symbol must not resolve to that symbol.

  entity_link.py:337 accepts ``compact.startswith(lowered)``, which is
  the inverse of a prefix test. 'ITC-INFRA' therefore resolves to ITC
  with the reason "unambiguous prefix match", which is the silent
  wrong-entity resolution entity_link.py:9-12 says this module exists
  to prevent.
  '''
  for text in ('ITC-INFRA', 'ITCXYZ', 'RELIANCEEXTRA', 'AXISBANKING'):
    result = resolve(text)
    assert result.status is not LinkStatus.RESOLVED, (
        f'{text!r} resolved to {result.symbol} by the {result.matched_by} '
        f'rule; a query that is a prefix of no symbol and no company name '
        f'must come back ambiguous or unknown')
    assert result.symbol is None


# --------------------------------------------------------------------------
# FINDING 9 -- index.html:6-8 claims the page runs from file:// and
# app.js:143-151 tells the operator to serve it instead. Neither path
# works: there is no asset route and no CORS header on any response.
# --------------------------------------------------------------------------

def test_the_api_admits_the_origin_the_dashboard_documents() -> None:
  '''A browser reading the API cross-origin needs a CORS header.

  index.html:6-8 says the page "runs from file:// with no build step
  and no network at all". A file:// page sends ``Origin: null`` and the
  answer must carry ``Access-Control-Allow-Origin`` for the fetch at
  app.js:157 to succeed. api.py:1857 sends three headers and none of
  them is that one, and there is no OPTIONS handler, so a preflight is
  answered 501.
  '''
  with serving() as (host, port):
    status, headers, _ = fetch(host, port, '/api/health', {'Origin': 'null'})
    assert status == 200
    assert headers.get('Access-Control-Allow-Origin'), (
        'no Access-Control-Allow-Origin on any response, so the file:// '
        'path documented in index.html:6-8 cannot read the API')
    connection = HTTPConnection(host, port, timeout=20)
    try:
      connection.request('OPTIONS', '/api/health',
                         headers={'Origin': 'null',
                                  'Access-Control-Request-Method': 'GET'})
      answer = connection.getresponse()
      answer.read()
      preflight = answer.status
    finally:
      connection.close()
    assert preflight != 501, (
        f'the preflight answered {preflight}, so no cross-origin read '
        f'is possible from any origin')


# --------------------------------------------------------------------------
# FINDING 10 -- no Host-header check on a server with no authentication.
# --------------------------------------------------------------------------

def test_a_host_header_the_server_did_not_bind_is_refused() -> None:
  '''The Host header must be one this process is entitled to answer.

  api.py binds loopback and serves a trading book with no
  authentication. Answering a request whose ``Host`` names somebody
  else's origin is what turns that into a DNS-rebinding target, because
  the browser will treat the reply as coming from the attacker's
  origin. api.py:1726 checks nothing.
  '''
  with serving() as (host, port):
    status, _, body = fetch(host, port, '/api/health',
                            {'Host': 'trading-desk.example.com'})
    assert status in (400, 403, 421), (
        f'a request for Host trading-desk.example.com was answered '
        f'{status}, leaking {len(body)} bytes of book state')


# --------------------------------------------------------------------------
# The invariants that SURVIVED an attempt to break them.
# --------------------------------------------------------------------------

def test_the_loopback_gate_holds_for_every_address_shape() -> None:
  '''api.py may only bind loopback, and the check must be exact.

  A gate that accepts ``::ffff:127.0.0.1`` and ``127.0.0.53`` and
  refuses ``0.0.0.0``, ``127.1`` and ``0177.0.0.1`` is right. One that
  accepts any of the last three is not.
  '''
  for host in ('127.0.0.1', '127.0.0.53', '::1', '0:0:0:0:0:0:0:1',
               '::ffff:127.0.0.1', 'localhost', 'LOCALHOST'):
    assert api.is_loopback(host), f'{host!r} should be loopback'
  for host in ('0.0.0.0', '::', '192.168.1.5', '10.0.0.1', '8.8.8.8',
               '169.254.169.254', '127.1', '0177.0.0.1', '', '*'):
    assert not api.is_loopback(host), f'{host!r} must not be loopback'
    with pytest.raises(ValueError):
      api.build_server(api.ApiService(), host, 0)


def test_ipv6_loopback_is_actually_bindable() -> None:
  '''``::1`` must pass the gate and then work as a socket.

  api.py:1963 picks the server class on the presence of a colon, so
  this exercises ApiServer6 and ``bound_address`` on the four-tuple an
  AF_INET6 socket reports.
  '''
  server = api.build_server(api.ApiService(), '::1', 0)
  try:
    assert isinstance(server, api.ApiServer6)
    assert server.address_family == socket.AF_INET6
    host, port = api.bound_address(server)
    assert host == '::1'
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
      connection = HTTPConnection(host, port, timeout=20)
      try:
        connection.request('GET', '/api/health')
        answer = connection.getresponse()
        payload = json.loads(answer.read())
        assert answer.status == 200
        assert payload['status'] == 'ok'
      finally:
        connection.close()
    finally:
      thread.join(timeout=5)
  finally:
    server.shutdown()
    server.server_close()


def test_control_arm_is_byte_identical_to_the_no_context_path() -> None:
  '''Arm A must be the untouched feature vector, byte for byte.

  Not close: the same repr, the same JSON, the same length. A control
  arm that zero-pads or re-encodes a float is not a control.
  '''
  assert CONTEXT_ENABLED is False
  assert CONTEXT_WIDTH == len(context_names())
  price = [0.01, -0.02, 0.03, 0.0, 1.5, -0.125]
  rich = SymbolContext(symbol='INFY', sentiment_score=0.9,
                       sentiment_disagreement=0.3, graph_depth=2.0)
  with_context = state_vector(price, rich, arm=ContextArm.CONTROL)
  without_context = state_vector(price, None, arm=ContextArm.CONTROL)
  assert repr(with_context) == repr(without_context)
  assert json.dumps(with_context) == json.dumps(without_context)
  assert len(with_context) == len(price)
  arm_b = state_vector(price, rich, arm=ContextArm.NUMERIC_CONTEXT)
  assert arm_b[len(price):] == (0.0,) * CONTEXT_WIDTH


def test_a_raising_provider_is_visibly_marked_not_dropped() -> None:
  '''A broken control arm must appear as an error row.

  api.py:792 catches broadly on purpose. The requirement is that the row
  survives with its error attached, so the control arm cannot quietly
  lose a member and read as "the other four beat it".
  '''
  service = api.ApiService(panels=rising_panel())
  table = service.baselines()
  assert len(table['strategies']) == len(api.strategies())
  names = [row['name'] for row in table['strategies']]
  assert 'momentum_ranked' in names
  assert 'equal_weight' in names
  # pylint: disable=protected-access
  row = api.ApiService(panels=rising_panel())._baseline_row(
    'explode', _explode)
  assert row['status'] == 'error'
  assert 'provider is on fire' in row['error']
  assert 'sharpe' not in row


def _explode(visible: Mapping[str, Sequence[Bar]]) -> dict[str, float]:
  '''Return no weights because this provider is broken.

  Args:
    visible: Panels visible at the decision bar.

  Raises:
    RuntimeError: Always.
  '''
  del visible
  raise RuntimeError('provider is on fire')


def test_seed_graph_is_acyclic_and_fan_out_is_not_a_loop() -> None:
  '''Cycle detection must be directed, and the seed must be clean.

  traversal.py:154 walks the dependency direction. Undirected
  enumeration would report a thirty-way fan-out from one crude node as
  a loop; the directed walk must not, and a genuine two-key
  contradiction must still be caught.
  '''
  graph = nifty50_seed()
  assert not traversal.has_cycle(graph)
  assert not traversal.cycles(graph)
  fan = DependencyGraph()
  crude = make_key(NodeKind.COMMODITY, 'crude')
  fan.add_node(Node(crude, NodeKind.COMMODITY, 'Crude'))
  for index in range(30):
    symbol = f'S{index:02d}'
    key = make_key(NodeKind.STOCK, symbol)
    fan.add_node(Node(key, NodeKind.STOCK, symbol))
    fan.add_edge(Edge(key, crude, EdgeKind.DEPENDS_ON))
  assert not traversal.has_cycle(fan)
  assert not traversal.cycles(fan)
  assert len(traversal.affected_stocks(fan, crude, 1)) == 30
  looped = DependencyGraph()
  sector = make_key(NodeKind.SECTOR, 'fmcg')
  macro = make_key(NodeKind.MACRO, 'gst')
  looped.add_node(Node(sector, NodeKind.SECTOR, 'fmcg'))
  looped.add_node(Node(macro, NodeKind.MACRO, 'gst'))
  looped.add_edge(Edge(sector, macro, EdgeKind.DEPENDS_ON))
  looped.add_edge(Edge(macro, sector, EdgeKind.DEPENDS_ON))
  assert traversal.has_cycle(looped)
  assert traversal.cycles(looped)
  with pytest.raises(ValueError):
    traversal.affected_stocks(graph, 'commodity:crude',
                              traversal.max_traversal_depth + 1)


def test_an_oversized_body_is_refused_with_a_413() -> None:
  '''A backtest trigger is a few hundred bytes; one mebibyte is the cap.

  api.py:172 caps acceptance. The 413 has to arrive, which means the
  body is drained rather than buffered, and the connection is closed
  rather than left mid-body for the next request to misparse.
  '''
  with serving() as (host, port):
    connection = HTTPConnection(host, port, timeout=30)
    try:
      body = b'{}' + b' ' * (api.max_body_bytes + 1024)
      connection.request('POST', '/api/backtest', body=body,
                         headers={'Content-Type': 'application/json'})
      answer = connection.getresponse()
      payload = json.loads(answer.read())
      assert answer.status == 413
      assert payload['status'] == 413
      assert 'not_advice' in payload
    finally:
      connection.close()


def test_the_reported_turnover_is_two_way_not_one_way() -> None:
  '''Pin what api.py hands to app.js under the label "one-way".

  portfolio.py:261 adds ``notional / mark`` on the buy leg and on the
  sell leg, so the number is two-way. app.js:275 prints it as
  "Turnover (one-way)". This test documents the two-way nature so the
  label cannot be argued with; it passes today.
  '''
  panels = rising_panel(('AAA', 'BBB'), bars=200, step=1.0)

  def swap(visible: Mapping[str, Sequence[Bar]]) -> dict[str, float]:
    index = max((len(series) - 1 for series in visible.values()), default=0)
    if index % 2 == 0:
      return {'AAA': 1.0, 'BBB': 0.0}
    return {'AAA': 0.0, 'BBB': 1.0}

  result = run_portfolio(panels, swap, capital=1_000_000.0,
                         rebalance_days=1, max_weight=1.0, history=1)
  per_rebalance = result.turnover / max(1, result.rebalances)
  assert per_rebalance == pytest.approx(2.0, abs=0.05), (
      f'a book that swaps its entire contents every rebalance shows 2.0 '
      f'two-way turnover; got {per_rebalance}')
