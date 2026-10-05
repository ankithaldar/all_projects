#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The join between a graph node and the news attached to it.

The tests that matter here are about **relevance and honesty**, not
arithmetic. There is no arithmetic in this module: it walks a table and
copies a list. What can go wrong is attaching a headline to a node for a
reason nobody can state, and the symptom is a panel that looks informed
while resting on nothing. So each test below attacks one specific way of
being wrong, and the assertions are on the *reason string*, not merely on
the count.

The ordering of that concern is deliberate:

* :func:`readings_for_node` returns a reading if and only if the graph
  reaches a **stock node** whose name equals the reading's symbol. No hop
  count, no text similarity, no commodity-name match. The tests pin that
  rule from both directions: crude does reach ONGC through the graph, and
  a reading filed under ``'crude'`` reaches nothing at all.
* Provenance is carried, not implied. Every returned reading names the
  symbol, the hop count, the key path and the edge kind of each hop, plus
  the entity linker's own reason for that symbol. A test that only counted
  rows would pass against a module that returned the right readings for
  the wrong reasons, which is the defect this repo keeps finding.
* "No news" and "this failed" are different answers. The first is a
  status on a returned object; the second is an exception. They are
  asserted to be distinguishable in both directions.
* A reading dated after the decision bar is **refused**, not filtered.
  Look-ahead has been reintroduced as a permissive default in this
  repository twice, so the test uses
  :func:`stock_rl.sentiment.score.require_visible` semantics as the
  oracle and asserts the refusal rather than the absence of the row.

Nothing here re-implements the graph walk. Every fixture comes from
:func:`stock_rl.graph.edges.nifty50_seed` or from a hand-built
:class:`~stock_rl.graph.edges.DependencyGraph`, so a test that passes
proves something about the shipped traversal and the shipped linker.
'''

from __future__ import annotations

import inspect
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from stock_rl.sentiment.entity_link import (
  LinkStatus,
  SymbolLinker,
  SymbolRecord,
  resolve,
)

from stock_rl.graph import news as news_module
from stock_rl.graph.edges import DependencyGraph, Edge, EdgeKind, nifty50_seed
from stock_rl.graph.news import (
  UNFITTED_NOTE,
  GraphSlice,
  NodeNews,
  NodeNewsStatus,
  PathRoute,
  ReadingRef,
  edge_meaning,
  node_slice,
  not_advice,
  readings_for_node,
)
from stock_rl.graph.nodes import Node, NodeKind
from stock_rl.graph.traversal import default_max_depth, max_traversal_depth
from stock_rl.sentiment.score import (
  LookAheadError,
  Rejection,
  SentimentReading,
  require_visible,
)

#: The bar every fixture is read at. Timezone-aware, because the module
#: under test refuses a naive one and a test that used a naive bar would be
#: testing the refusal instead of the join.
BAR = datetime(2024, 1, 10, 15, 30, tzinfo=timezone.utc)

#: CRUDE is the design doc's worked example, so it is the node every
#: relevance test is written against.
CRUDE = 'commodity:crude'

#: The three names the design doc says a crude shock reaches.
CRUDE_NAMES = ('ASIANPAINT', 'ONGC', 'RELIANCE')


def at_bar(
  hours: float = 0.0,
  symbol: str = 'ONGC',
  score: float = 0.4,
  source: str = 'reuters',
  event_type: str = 'news',
) -> SentimentReading:
  '''Return one reading, public at :data:`BAR` unless told otherwise.

  Args:
    hours: Hours after the bar the reading became public. Zero is public
      at the bar exactly, which is the boundary ``is_public_at`` uses.
    symbol: Symbol the reading is filed under. Deliberately a parameter,
      because the rule under test is a statement about this field and a
      fixture that hardcoded it could not falsify it.
    score: Directional score.
    source: Source identifier. One source is deliberate where the aggregate
      is not under test, so a fixture cannot accidentally clear the
      two-source minimum and hide a single-source verdict.
    event_type: Free-form event label.

  Returns:
    A validated reading. :class:`~stock_rl.sentiment.score.SentimentReading`
    rejects a naive timestamp at construction, so the aware ``BAR`` is
    used directly.
  '''
  return SentimentReading(symbol, score, event_type, 0.5,
                          BAR + timedelta(hours=hours), source)


def two_sources(
  symbol: str = 'ONGC',
  hours: float = 0.0,
) -> list[SentimentReading]:
  '''Return two readings from distinct sources, which clears the minimum.

  Args:
    symbol: Symbol both readings are filed under.
    hours: Hours after the bar both became public.

  Returns:
    Two readings. Two and not three, because
    :data:`stock_rl.sentiment.score.default_min_sources` is two and a
    fixture that cleared it by accident would hide a verdict the tests
    mean to inspect.
  '''
  return [
    SentimentReading(symbol, 0.5, 'news', 0.5,
                     BAR + timedelta(hours=hours), 'reuters'),
    SentimentReading(symbol, 0.1, 'news', 0.4,
                     BAR + timedelta(hours=hours), 'cnbc'),
  ]


def one_off_symbol_graph() -> DependencyGraph:
  '''Return a graph whose commodity node depends on nothing.

  A two-node graph: a commodity that no stock points at, and one stock
  that points at a different commodity. Asking the first commodity for
  news has to come back empty, because the package holds no claim
  connecting it to any listed company.

  Returns:
    A fresh graph, so mutating it in one test cannot affect another.
  '''
  graph = DependencyGraph()
  graph.add_node(Node('commodity:unrelated', NodeKind.COMMODITY, 'Unrelated'))
  graph.add_node(Node('commodity:real', NodeKind.COMMODITY, 'Real'))
  graph.add_node(Node('stock:AAA', NodeKind.STOCK, 'AAA'))
  graph.add_edge(Edge('stock:AAA', 'commodity:real', EdgeKind.DEPENDS_ON))
  return graph


# --- the relevance rule ---------------------------------------------------

class TestRelevanceComesFromTheGraph:
  '''A commodity node has no symbol, so the graph supplies the bridge.'''

  def test_a_commodity_reaches_news_through_its_dependent_stocks(self):
    '''``commodity:crude`` attaches ONGC's readings, naming the hop.

    This is the gap the module exists to close. ONGC is a stock node one
    edge from crude in the shipped seed, so its readings are the news of
    a company the package claims consumes crude. Without the hop, a
    commodity node attaches nothing at all, which is the state before
    this module existed.
    '''
    news = readings_for_node(nifty50_seed(), CRUDE,
                             two_sources('ONGC'), BAR)
    assert news.status is NodeNewsStatus.HAS_NEWS
    assert news.news_count == 2
    assert {entry.symbol for entry in news.entries} == {'ONGC'}
    assert news.checked_symbols == CRUDE_NAMES, (
        f'the design doc names {CRUDE_NAMES} at one hop from crude; the walk '
        f'found {news.checked_symbols}')

  def test_a_reading_filed_under_the_commodity_is_attached_to_nothing(self):
    '''No headline is matched to a commodity name, at any spelling.

    The forbidden design is to resolve the node's own label -- 'Crude',
    'Crude Oil', 'crude_oil' -- against the headline and attach on a hit.
    This test does the bluntest possible version of that: it hands the
    join readings filed under the commodity's own name from two distinct
    sources, so a text-matching implementation would clear the
    two-source minimum and report them. The lookup rule has no way to
    attach them, because 'crude' is not the name of any stock node, and
    so they stay unattached.

    The refusal is structural rather than a check that happens to be in
    the way: hop 1 of the rule is "reaches a stock node", and no stock
    node is named 'crude'. That is why the test uses a name which would
    match under a looser rule rather than asserting on the wording of an
    error.
    '''
    readings = [
      SentimentReading('crude', 0.9, 'news', 0.8, BAR, 'reuters'),
      SentimentReading('crude', 0.8, 'news', 0.8, BAR, 'cnbc'),
    ]
    news = readings_for_node(nifty50_seed(), CRUDE, readings, BAR)
    assert news.news_count == 0, (
        f'{news.news_count} reading(s) were attached to {CRUDE} by matching '
        f'the commodity name; the graph is the only sanctioned bridge')
    assert news.status is NodeNewsStatus.NO_NEWS
    assert news.checked_symbols == CRUDE_NAMES

  def test_a_reading_about_a_symbol_the_graph_does_not_reach_is_dropped(self):
    '''Being in the same news day is not relevance.

    MARUTI is a real symbol the linker resolves, and it has no path to
    crude in the seed graph. Attaching its readings to crude would be the
    most tempting wrong answer available: the symbol is valid, the reading
    is public at the bar, and nothing about it is malformed. It is still
    wrong, because the only thing connecting crude to a symbol is the
    graph and the graph does not make that claim.
    '''
    news = readings_for_node(nifty50_seed(), CRUDE, two_sources('MARUTI'), BAR)
    assert news.news_count == 0
    assert news.status is NodeNewsStatus.NO_NEWS
    assert 'MARUTI' not in news.checked_symbols

  def test_a_stock_node_returns_only_its_own_readings(self):
    '''A stock is its own zero-hop route, and nothing else joins it.

    ONGC depends on crude, so a reader might expect crude's news to show
    up here too. It must not: the join is directional, and mixing a
    node's inputs into its news would put a number next to a company that
    never made a statement about it.
    '''
    news = readings_for_node(nifty50_seed(), 'stock:ONGC',
                             two_sources('ONGC'), BAR)
    assert [entry.symbol for entry in news.entries] == ['ONGC', 'ONGC']
    route = news.entries[0].route
    assert route.direct
    assert route.hops == 0
    assert route.path == ('stock:ONGC',)
    assert route.edges == ()

  def test_the_depth_argument_bounds_the_walk_and_the_answer_says_so(self):
    '''One hop from ``macro:global_liquidity`` reaches nobody.

    edges.py documents this shape: ``global_liquidity -> fii_net ->
    sector:banking -> stock:HDFCBANK`` is three hops, and at one hop the
    only thing reached is another macro. That is why depth is an argument
    and not a constant buried in a call site -- a default of one returns
    an empty panel for a global driver and reads exactly like "no news".
    The returned object carries the bound that produced it, so two
    different universes can never be compared without the reader knowing.
    '''
    graph = nifty50_seed()
    liquidity = 'macro:global_liquidity'
    shallow = readings_for_node(graph, liquidity, [], BAR, max_depth=1)
    assert shallow.max_depth == 1
    assert not shallow.checked_symbols
    assert shallow.status is NodeNewsStatus.NO_DEPENDENTS
    three = readings_for_node(graph, liquidity, [], BAR, max_depth=3)
    assert three.max_depth == 3
    assert 'HDFCBANK' in three.checked_symbols, (
        f'three hops from a global driver should reach the banks, got '
        f'{three.checked_symbols}')
    assert three.status is NodeNewsStatus.NO_NEWS

  def test_crude_reaches_exactly_the_three_names_the_design_doc_names(self):
    '''The worked example is the test of the graph, at the shipped depth.

    Not a depth-sensitivity test: a bound on the walk. At the default
    depth crude must reach ASIANPAINT, ONGC and RELIANCE and nothing
    else, because those are the three edges the design doc itself names.
    A walk that also returned IT names would be widening the universe
    without a claim to widen it.
    '''
    news = readings_for_node(nifty50_seed(), CRUDE, [], BAR)
    assert news.max_depth == default_max_depth
    assert news.checked_symbols == CRUDE_NAMES

  def test_a_multi_hop_route_names_every_node_between(self):
    '''SHREECEM reaches coal only through the cement sector.

    edges.py says this shape is load-bearing: a traversal whose depth is
    one hop short silently returns the wrong universe. So the fixture is
    the two-hop case, and the assertion is on the key chain rather than
    on the hop count, because a reader has to be able to see *through*
    the sector to reject the edge they disagree with.
    '''
    news = readings_for_node(nifty50_seed(), 'commodity:coal',
                             two_sources('SHREECEM'), BAR)
    assert news.status is NodeNewsStatus.HAS_NEWS
    route = news.entries[0].route
    assert route.symbol == 'SHREECEM'
    assert route.hops == 2
    assert route.path == ('commodity:coal', 'sector:cement',
                          'stock:SHREECEM')
    assert route.edges == (EdgeKind.DEPENDS_ON, EdgeKind.DEPENDS_ON)

  def test_an_event_edge_is_reported_as_an_event_not_a_dependency(self):
    '''An ``affected_by`` hop is labelled as one.

    Traversal mixes the two kinds on purpose, and this module inherits
    that. But a reader who sees an RBI page listing a bank under a
    "depends on the repo rate" edge is being told something different
    from one who sees it under "affected by". The kind travels with the
    hop, so the two are never silently conflated.

    HDFCBANK is reached twice over -- once by the direct
    ``_stock_events`` row and once through the banking sector's exposure
    to the same event -- and the shortest route wins, which is what the
    traversal module documents. What is asserted is that every hop on the
    reported route is an event edge, because that is the claim the reader
    has to be able to check.
    '''
    news = readings_for_node(nifty50_seed(), 'event:rbi_rate_decision',
                             two_sources('HDFCBANK'), BAR)
    assert news.status is NodeNewsStatus.HAS_NEWS
    route = news.entries[0].route
    assert route.symbol == 'HDFCBANK'
    assert route.edges, 'the route names no edge kind at all'
    assert set(route.edges) == {EdgeKind.AFFECTED_BY}, (
        f'the route kinds are {set(route.edges)}; an event edge reported as '
        f'a dependency misstates what the graph claims')
    assert route.path[0] == 'event:rbi_rate_decision'
    assert route.path[-1] == 'stock:HDFCBANK'


# --- provenance -----------------------------------------------------------

class TestProvenanceNamesTheHop:
  '''Every reading says why it is attached, in words a reader can use.'''

  def test_provenance_names_symbol_path_edge_kinds_and_source(self):
    '''The three things the brief asks for, plus the registry's reason.

    Symbol, path, source. The assertion is deliberately on substrings
    rather than on equality, because this is a display string and pinning
    it byte-for-byte would make a wording change look like a behaviour
    change. What is pinned is that all four facts are present: the symbol
    ONGC, the full key chain, the edge kind of the hop, and the linker's
    reason for ONGC, which is that the exchange symbol itself is a
    registry row.
    '''
    news = readings_for_node(nifty50_seed(), CRUDE, two_sources('ONGC'), BAR)
    route = news.entries[0].route
    described = route.describe()
    assert 'ONGC' in described
    assert 'commodity:crude -> stock:ONGC' in described
    assert 'depends_on' in described
    assert route.linkage, (
        'the route carries no linkage string, so the reader cannot tell '
        'whether ONGC was resolved from text or merely filed by symbol')
    assert route.linkage in described
    assert route.to_dict()['describe'] == described

  def test_each_reading_carries_its_own_route_not_the_node(self):
    '''Two stocks attached to one node carry different routes.

    The failure this guards against is a single provenance object shared
    across the result: the first symbol's path would then be printed
    beside every reading, and ONGC's news would appear to arrive via
    RELIANCE. Both symbols in this fixture are one hop from crude, so the
    paths differ in their tail, which is enough to catch a shared object.
    '''
    readings = two_sources('ONGC') + two_sources('RELIANCE')
    news = readings_for_node(nifty50_seed(), CRUDE, readings, BAR)
    assert len(news.entries) == 4
    routes = {(entry.symbol, entry.route.symbol,
               entry.route.path[-1]) for entry in news.entries}
    assert routes == {
      ('ONGC', 'ONGC', 'stock:ONGC'),
      ('RELIANCE', 'RELIANCE', 'stock:RELIANCE'),
    }, f'each reading must carry its own route, got {routes}'

  def test_the_reason_says_the_link_is_a_graph_route_not_a_text_match(self):
    '''The reason must state the mechanism, not just the outcome.

    A reader who sees "ONGC -0.4" on the crude page and is told only
    that it is "related" has learned nothing they can audit. The reason
    has to name the symbol, the hop count and the chain, and it has to say
    that the headline was not matched to the commodity.
    '''
    news = readings_for_node(nifty50_seed(), CRUDE, two_sources('ONGC'), BAR)
    reason = news.entries[0].reason
    assert 'ONGC' in reason
    assert '1 hop' in reason
    assert 'commodity:crude -> stock:ONGC' in reason
    assert 'not by matching text' in reason, (
        f'the reason does not say how the reading was attached: {reason!r}')

  def test_the_direct_route_says_no_hop_was_used(self):
    '''A stock node's own news must not claim a graph route.

    Attaching RELIANCE's reading to RELIANCE through the graph would be
    a fiction: there is no edge walked. The sentence has to say so, or a
    reader concludes the graph vouches for every reading on a company page.
    '''
    news = readings_for_node(nifty50_seed(), 'stock:RELIANCE',
                             two_sources('RELIANCE'), BAR)
    reason = news.entries[0].reason
    assert 'Attached directly' in reason
    assert 'no graph hop was used' in news.entries[0].route.describe()


class TestTheRegistryIsAskedRatherThanAssumed:
  '''A stock node the linker has never heard of is reported, not hidden.'''

  def test_a_symbol_with_no_registry_row_says_so(self):
    '''The seed graph names a symbol absent from the starter registry.

    'AAA' is in the one-off fixture graph above, and it is not a row in
    the linker registry. A reading filed under it is still legitimately
    attached -- it arrived with the symbol already on it -- but the
    provenance must say that no free text resolves to it, because that is
    the difference between "news we could have found by reading
    headlines" and "news somebody handed us under a bare symbol".
    '''
    graph = one_off_symbol_graph()
    readings = two_sources('AAA')
    news = readings_for_node(graph, 'commodity:real', readings, BAR)
    assert news.status is NodeNewsStatus.HAS_NEWS
    linkage = news.entries[0].route.linkage
    assert 'no registry row' in linkage, (
        f'the linkage string does not report the missing row: {linkage!r}')
    assert 'AAA' in linkage

  def test_a_registered_symbol_reports_the_registry_s_own_reason(self):
    '''The positive case, so the negative one cannot pass vacuously.

    ONGC resolves through the exchange symbol row, so its linkage string
    must be the linker's wording rather than a generic success. A test
    that only asserted "no registry row" in the other case would pass
    against an implementation that always printed the failure.
    '''
    news = readings_for_node(nifty50_seed(), CRUDE, two_sources('ONGC'), BAR)
    linkage = news.entries[0].route.linkage
    assert 'no registry row' not in linkage
    assert 'exact exchange symbol' in linkage or 'table' in linkage

  def test_a_caller_supplied_registry_is_the_one_that_is_asked(self):
    '''The registry is an argument, so a deployment can bring its own.

    The starter registry has no ambiguous row, so the collision branch of
    the linkage string is unreachable with it. A registry is *data* --
    :class:`~stock_rl.sentiment.entity_link.SymbolLinker` exists so one can
    be supplied -- so the join asks the caller's registry and reports a
    collision as a collision rather than as a success. Reading this as an
    escape hatch would be wrong: the reading is still attached, because it
    arrived filed under the symbol; only the provenance string changes.
    '''
    clashing = SymbolLinker((
      SymbolRecord('AAA', 'Alpha Metals'),
      SymbolRecord('AXA', 'Beta Wire', ('AAA',)),
      SymbolRecord('CCC', 'Charcoal Traders'),
    ))
    graph = DependencyGraph()
    graph.add_node(Node('stock:AAA', NodeKind.STOCK, 'AAA'))
    graph.add_node(Node('stock:CCC', NodeKind.STOCK, 'CCC'))
    news = readings_for_node(graph, 'stock:AAA', two_sources('AAA'), BAR,
                             linker=clashing)
    linkage = news.entries[0].route.linkage
    assert 'is ambiguous' in linkage, (
        f'a registry collision was not reported as one: {linkage!r}')
    assert 'AXA' in linkage, 'the collision must name who else claims the form'
    assert news.status is NodeNewsStatus.HAS_NEWS, (
        'an ambiguous registry row must not unattach a reading that was '
        'filed under the symbol')


# --- honest absence -------------------------------------------------------

class TestAbsenceIsNotFailure:
  '''No news, no route, and a broken call must read differently.'''

  def test_a_node_with_no_dependents_returns_empty_rather_than_guessing(self):
    '''No stock downstream means no lookup, and no fallback universe.

    The tempting wrong answers are all specific: fall back to the
    sector's news, fall back to every reading in hand, fall back to the
    most-mentioned symbol. Each of those would render a populated panel
    for a node the package holds no claim about, and a populated panel
    reads as evidence. The empty result carries the symbols that were
    checked, so the emptiness is attributable rather than mysterious.
    '''
    graph = one_off_symbol_graph()
    news = readings_for_node(graph, 'commodity:unrelated',
                             two_sources('AAA'), BAR)
    assert news.status is NodeNewsStatus.NO_DEPENDENTS
    assert news.news_count == 0
    assert not news.checked_symbols
    assert not news.entries
    assert not news.routes

  def test_no_news_and_no_dependents_are_different_statuses(self):
    '''Both are empty and they are not the same answer.

    One means the graph named the companies and none of them had news.
    The other means the graph named nobody. Collapsing them produces a
    panel that says "no news" for a node nothing was ever checked against,
    which is a claim the package cannot support.
    '''
    graph = one_off_symbol_graph()
    news = readings_for_node(graph, 'commodity:real', [], BAR)
    orphan = readings_for_node(graph, 'commodity:unrelated', [], BAR)
    assert news.status is NodeNewsStatus.NO_NEWS
    assert orphan.status is NodeNewsStatus.NO_DEPENDENTS
    assert news.status is not orphan.status
    assert news.message() != orphan.message()

  def test_the_no_news_message_names_what_was_looked_up(self):
    '''An absence of news must be attributable to a lookup that ran.

    "No news" alone is indistinguishable from a filter that never
    matched anything. Naming the symbols turns the sentence into evidence
    that the join ran and the feed was empty, which is the difference
    between a quiet tape and a broken query.
    '''
    news = readings_for_node(nifty50_seed(), CRUDE, [], BAR)
    assert news.status is NodeNewsStatus.NO_NEWS
    message = news.message()
    for symbol in CRUDE_NAMES:
      assert symbol in message, (
          f'the message does not name {symbol}, so the empty result is not '
          f'attributable to a lookup that ran: {message!r}')
    assert 'not a failure to load' in message

  def test_an_error_is_an_exception_and_not_an_empty_status(self):
    '''A malformed request raises; it does not report no news.

    An unknown node and a naive bar are programming errors. Returning a
    status for either would let a typo surface as a quiet panel, which is
    the failure mode this whole module is shaped around.
    '''
    graph = nifty50_seed()
    with pytest.raises(ValueError, match='unknown node'):
      readings_for_node(graph, 'commodity:unobtainium', [], BAR)
    with pytest.raises(ValueError, match='timezone-aware'):
      readings_for_node(graph, CRUDE, [], datetime(2024, 1, 10, 15, 30))
    with pytest.raises(ValueError, match='max_depth'):
      readings_for_node(graph, CRUDE, [], BAR, max_depth=0)

  def test_a_single_source_reading_is_attached_and_marked_unusable(self):
    '''Attached is not the same as usable, and the panel must say which.

    One source clears the relevance rule -- the symbol matches and the
    hop exists -- and fails the source minimum. A view that showed the
    score without the refusal would present a single opinion as a
    corroborated number, which is the failure the whole sentiment module
    exists to prevent.
    '''
    news = readings_for_node(nifty50_seed(), CRUDE, [at_bar()], BAR)
    assert news.status is NodeNewsStatus.HAS_NEWS
    assert news.news_count == 1
    entry = news.entries[0]
    assert not entry.aggregate.usable
    assert Rejection.SINGLE_SOURCE.value in entry.aggregate.rejected
    assert entry.to_dict()['usable'] is False


# --- look-ahead -----------------------------------------------------------

class TestReadingAfterTheBarIsRefused:
  '''The gate this repository has reintroduced twice must stay shut.'''

  def test_a_reading_after_the_bar_is_refused_not_dropped(self):
    '''The call raises, rather than quietly returning the survivors.

    Two of the four readings are public three days after the bar. A
    filtering implementation returns two rows and looks correct. The
    oracle is :func:`stock_rl.sentiment.score.require_visible`, called
    with the same readings and the same bar, so the test proves the join
    is *refusing* rather than filtering by showing the strict helper
    raises on the identical input the join refused.

    The other half is asserted too, because a refusal that still returned
    the public survivors as a result would satisfy the first assertion
    while quietly publishing a number built on data a trader could not
    have had.
    '''
    readings = two_sources('ONGC', hours=-72) + two_sources('ONGC', hours=72)
    leaked = [item for item in readings if item.available_from > BAR]
    assert len(leaked) == 2
    with pytest.raises(LookAheadError, match='not public'):
      readings_for_node(nifty50_seed(), CRUDE, readings, BAR)
    with pytest.raises(LookAheadError, match='not public'):
      require_visible(readings, BAR)
    # The control for the oracle: the same readings at a later bar are
    # public, so the refusal above is the date and not a broken filter.
    later = readings_for_node(nifty50_seed(), CRUDE, readings,
                              BAR + timedelta(days=4))
    assert later.status is NodeNewsStatus.HAS_NEWS
    assert later.news_count == 4

  def test_a_leak_for_a_symbol_outside_the_walk_does_not_refuse(self):
    '''The guard scopes to the readings that could reach the answer.

    A reading dated a year ahead for WIPRO cannot appear in a crude
    answer, so raising on it would train a caller to catch and ignore the
    exception -- which is how a strict guard becomes a permissive one.
    The scoping is asserted explicitly so a later edit that widens the
    filter to every reading in hand is caught, and so nobody reads this
    as the guard being absent.
    '''
    leak = SentimentReading('WIPRO', 0.9, 'news', 0.9,
                            BAR + timedelta(days=365), 'reuters')
    news = readings_for_node(nifty50_seed(), CRUDE,
                             two_sources('ONGC') + [leak], BAR)
    assert news.status is NodeNewsStatus.HAS_NEWS
    assert [entry.symbol for entry in news.entries] == ['ONGC', 'ONGC']
    assert all(entry.symbol != 'WIPRO' for entry in news.entries)

  def test_the_bar_argument_is_required(self):
    '''There is no default bar, so the permissive call cannot be written.

    score.py's ``aggregate`` once defaulted ``decision_bar`` to None,
    which skipped the filter entirely and let four readings dated a year
    after the bar through at full strength. The signature here repeats
    the fix: ``decision_bar`` has no default, so calling this without a
    bar is a TypeError at the call site rather than a leak at the bar.
    '''
    signature = inspect.signature(readings_for_node)
    bar = signature.parameters['decision_bar']
    assert bar.default is inspect.Parameter.empty, (
        'decision_bar has a default, so the permissive call is writable '
        'again')
    # The module carries `from __future__ import annotations`, so the
    # annotation is the string 'datetime' rather than the class. Compared
    # as text for that reason, and pinned so the parameter cannot quietly
    # become an optional one by being retyped.
    assert bar.annotation == 'datetime'
    with pytest.raises(TypeError):
      # pylint: disable=no-value-for-parameter
      # The missing argument IS the assertion. pylint catches the same
      # mistake statically here, which is the point: the check is what
      # stops a permissive default from being added back, and this call is
      # the only place in the package that makes the call deliberately.
      readings_for_node(nifty50_seed(), CRUDE, [])

  def test_the_depth_argument_is_bounded_by_the_traversal_ceiling(self):
    '''The hop bound is the one traversal.py publishes, not a local one.

    A second, looser bound here would mean a join could walk further than
    any other caller in the package, on a graph that may hold a cycle.
    ``max_traversal_depth`` is checked and the request is refused rather
    than clamped, so a caller who asked for depth 100 finds out.
    '''
    with pytest.raises(ValueError, match=str(max_traversal_depth)):
      readings_for_node(nifty50_seed(), CRUDE, [], BAR,
                        max_depth=max_traversal_depth + 1)
    assert readings_for_node(nifty50_seed(), CRUDE, [], BAR,
                             max_depth=max_traversal_depth).max_depth == (
                               max_traversal_depth)


# --- the drawn slice ------------------------------------------------------

class TestSliceIsRenderableAndHonest:
  '''What the view is handed, and what it is told about the edges.'''

  def test_the_slice_carries_the_focus_node_its_paths_and_its_news(self):
    '''A dependency graph for one stock, with the path between them.

    The brief asks for the company node, its commodity and macro inputs,
    and the path between them. ``stock:RELIANCE`` depends directly on
    crude, usdinr and its own sector, so its slice must contain all four
    keys with the right hops, and the edges must be drawn **cause to
    affected** -- which is the reverse of the stored edge, and the
    assertion is on both ends so an inverted drawing cannot pass.
    '''
    slice_result = node_slice(nifty50_seed(), 'stock:RELIANCE',
                              two_sources('RELIANCE'), BAR)
    keys = {node.key for node in slice_result.nodes}
    assert keys >= {'stock:RELIANCE', 'commodity:crude', 'macro:usdinr'}
    hops = {node.key: node.hops for node in slice_result.nodes}
    assert hops['stock:RELIANCE'] == 0
    assert hops['commodity:crude'] == 1, 'crude is a direct input of RELIANCE'
    assert hops['macro:usdinr'] == 1
    roles = {node.key: node.role for node in slice_result.nodes}
    assert roles['stock:RELIANCE'] == 'focus'
    assert roles['commodity:crude'] == 'path'
    direct = {(edge.cause, edge.affected) for edge in slice_result.edges
              if edge.hops == 1}
    assert direct >= {
      ('commodity:crude', 'stock:RELIANCE'),
      ('macro:usdinr', 'stock:RELIANCE'),
    }, f'the direct inputs are drawn as {direct}, cause to affected'
    for edge in slice_result.edges:
      assert edge.meaning, (
          f'the edge {edge.cause} -> {edge.affected} carries no label')
      assert edge.short in ('depends on', 'affected by')

  def test_an_event_hop_is_drawn_and_labelled_on_a_stock_slice(self):
    '''The upstream walk surfaces event edges, labelled as events.

    ``sector:oil_gas`` is ``affected_by`` the OPEC cut, and a stock slice
    that walked its sector reaches that edge. If the slice dropped
    non-dependency edges the drawn graph would be a strictly smaller
    universe than the file, and the reader would not know what had been
    left out.
    '''
    slice_result = node_slice(nifty50_seed(), 'stock:RELIANCE', [], BAR)
    events = {(edge.cause, edge.affected): edge for edge in slice_result.edges
              if edge.kind is EdgeKind.AFFECTED_BY}
    assert ('event:opec_supply_cut', 'sector:oil_gas') in events, (
        f'the OPEC edge is missing from the slice; drawn edges are '
        f'{sorted((e.cause, e.affected) for e in slice_result.edges)}')
    assert events[('event:opec_supply_cut', 'sector:oil_gas')].short == (
      'affected by')

  def test_every_edge_is_marked_unmeasured_and_says_why(self):
    '''No edge in this package is a measured sensitivity.

    edges.py says the seed is a hypothesis, and the payload has to carry
    that to the view rather than leaving the UI to imply otherwise. The
    assertion is on the field AND on the sentence, because a boolean that
    nothing reads is exactly the shape of defect this repo has shipped
    twice. The word "Unfitted" is required in every meaning string, so a
    consumer cannot read one edge's claim without seeing the caveat on it.
    '''
    slice_result = node_slice(nifty50_seed(), 'stock:RELIANCE', [], BAR)
    assert slice_result.edges
    for edge in slice_result.edges:
      assert edge.measured is False, (
          f'{edge.cause} -> {edge.affected} is marked measured')
      assert 'Unfitted' in edge.meaning
      assert 'estimated from returns' in edge.meaning
    payload = slice_result.to_dict()
    assert payload['unfitted'] == UNFITTED_NOTE
    assert 'not measured sensitivities' in payload['unfitted']
    for row in payload['edges']:
      assert row['measured'] is False
      assert row['meaning']

  def test_the_payload_is_json_serialisable_and_complete(self):
    '''The view consumes a JSON object, so the shape must survive a round trip.

    The dashboard serves and parses JSON, so a dataclass whose ``to_dict``
    returns a datetime or an enum member would raise at the boundary. The
    round trip also proves the view is not being handed a Python repr
    dressed as data.
    '''
    slice_result = node_slice(nifty50_seed(), CRUDE, two_sources('ONGC'), BAR)
    payload = slice_result.to_dict()
    text = json.dumps(payload)
    restored = json.loads(text)
    assert restored['node'] == CRUDE
    assert restored['status'] == NodeNewsStatus.HAS_NEWS.value
    assert restored['readings'][0]['symbol'] == 'ONGC'
    assert restored['readings'][0]['via_path'] == [CRUDE, 'stock:ONGC']
    assert restored['readings'][0]['via_hops'] == 1
    assert restored['unfitted'] == UNFITTED_NOTE
    assert restored['not_advice'] == not_advice
    assert 'Not financial advice' in restored['not_advice']
    assert isinstance(restored['message'], str) and restored['message']

  def test_the_slice_keeps_a_reached_stock_that_has_no_news(self):
    '''A dependent with nothing to say is still drawn.

    Dropping it would make the graph look smaller than the file says it
    is, and would make ASIANPAINT's absence of news indistinguishable
    from ASIANPAINT not being exposed to crude at all. The view marks the
    difference itself: the node is present with a zero count.
    '''
    graph = nifty50_seed()
    slice_result = node_slice(graph, CRUDE, two_sources('ONGC'), BAR)
    keys = {node.key for node in slice_result.nodes}
    assert 'stock:ASIANPAINT' in keys
    assert 'stock:RELIANCE' in keys
    assert {route.symbol for route in slice_result.news.routes} == set(
      CRUDE_NAMES)

  def test_a_stock_that_is_a_cause_is_not_drawn_as_an_input(self):
    '''A stock another stock depends on is a peer, not an input.

    Nothing in the seed graph says one listed company is an input of
    another, so this shape only arises in a hand-edited file. Drawing it
    as an input would assert a supply-chain relationship between two
    companies, which is a much stronger claim than "these two are in the
    same portfolio". The edge itself is still drawn; the node is still
    shown as reached, because hiding it would misrepresent the file.

    The assertion is on the *role* rather than on absence, because a walk
    that simply stopped at the stock would pass an absence check while
    quietly truncating the subgraph.
    '''
    graph = DependencyGraph()
    graph.add_node(Node('stock:AAA', NodeKind.STOCK, 'AAA'))
    graph.add_node(Node('stock:BBB', NodeKind.STOCK, 'BBB'))
    graph.add_node(Node('commodity:zzz', NodeKind.COMMODITY, 'Zzz'))
    graph.add_edge(Edge('stock:AAA', 'stock:BBB', EdgeKind.DEPENDS_ON))
    graph.add_edge(Edge('stock:AAA', 'commodity:zzz', EdgeKind.DEPENDS_ON))
    graph.add_edge(Edge('stock:BBB', 'commodity:zzz', EdgeKind.DEPENDS_ON))
    result = node_slice(graph, 'stock:AAA', [], BAR)
    roles = {node.key: node.role for node in result.nodes}
    assert roles['commodity:zzz'] == 'path', 'the real input is missing'
    got = roles['stock:BBB']
    assert got == 'peer', (
        f'stock:BBB is labelled {got!r}; calling a listed company an input '
        f'of another asserts a supply-chain relationship the graph never '
        f'established')
    drawn = {(edge.cause, edge.affected) for edge in result.edges}
    assert ('commodity:zzz', 'stock:AAA') in drawn
    assert ('stock:BBB', 'stock:AAA') in drawn, (
        'the peer edge is a stored edge and is drawn; what is withheld is '
        'the claim that BBB is an INPUT of AAA')

  def test_an_empty_slice_still_carries_the_honest_message(self):
    '''No news renders a sentence, not a blank panel.

    ``message()`` is what the view prints, so it is asserted for the empty
    case as carefully as the data is for the full one. A view given an
    empty payload must still say what was checked.
    '''
    slice_result = node_slice(nifty50_seed(), 'commodity:gold', [], BAR)
    payload = slice_result.to_dict()
    assert payload['status'] == NodeNewsStatus.NO_NEWS.value
    assert payload['readings'] == []
    assert 'TITAN' in payload['message']
    assert payload['checked_symbols'] == ['TITAN']
    assert slice_result.edges, 'gold reaches TITAN, so the edge is drawn'


class TestEdgeMeaningNamesBothEnds:
  '''The word on the arrow has to survive being read alone.'''

  @pytest.mark.parametrize('kind', [EdgeKind.DEPENDS_ON, EdgeKind.AFFECTED_BY])
  def test_both_kinds_name_the_cause_and_the_affected_node(self, kind):
    '''Every edge label says which node depends on, or is affected by, which.

    edges.py keeps one orientation -- affected to cause -- while the view
    draws cause to affected, because that is the direction a shock
    travels. Reversing the arrow on screen without saying so in the label
    would invert the graph silently, which is the kind of error that
    looks like a working feature.
    '''
    meaning = edge_meaning(kind, 'commodity:crude', 'stock:ONGC')
    assert 'commodity:crude' in meaning
    assert 'stock:ONGC' in meaning
    assert 'reverse of the stored edge' in meaning
    if kind is EdgeKind.DEPENDS_ON:
      assert 'depending on' in meaning
    else:
      assert 'affected by' in meaning


class TestResultObjectsAreWhatTheyClaim:
  '''The dataclasses hold what their docstrings promise.'''

  def test_every_reading_is_the_same_object_the_caller_handed_over(self):
    '''The reading is copied by reference, not rebuilt.

    A join that re-wrapped a reading would let the two drift, and the
    score on screen would stop being the score that was scored. Identity
    is the cheapest way to pin that.
    '''
    supplied = two_sources('ONGC')
    news = readings_for_node(nifty50_seed(), CRUDE, supplied, BAR)
    returned = [entry.reading for entry in news.entries]
    assert set(map(id, returned)) <= set(map(id, supplied))

  def test_a_route_and_a_reading_are_frozen(self):
    '''Provenance cannot be edited after the fact.

    Both are frozen dataclasses. That is what stops a caller rewriting the
    path on one reading to tidy a display, which would leave the reason
    string disagreeing with the route it is attached to.
    '''
    with pytest.raises(Exception):
      at_bar().source = 'edited'  # type: ignore[misc]
    route = PathRoute(symbol='ONGC', hops=1,
                      path=(CRUDE, 'stock:ONGC'),
                      edges=(EdgeKind.DEPENDS_ON,), linkage='x')
    with pytest.raises(Exception):
      route.hops = 4  # type: ignore[misc]

  def test_the_dataclasses_are_the_types_the_module_claims(self):
    '''The public names are the documented types, not something else.

    Cheap, and it catches a refactor that swaps a dataclass for a plain
    tuple and leaves every attribute access in the view working by luck.
    '''
    news = readings_for_node(nifty50_seed(), CRUDE, two_sources('ONGC'), BAR)
    assert isinstance(news, NodeNews)
    assert isinstance(news.entries[0], ReadingRef)
    assert isinstance(news.entries[0].route, PathRoute)
    assert isinstance(node_slice(nifty50_seed(), CRUDE, [], BAR), GraphSlice)


class TestTheRuleIsStatedInTheSource:
  '''The docstring is evidence, so it is asserted like evidence.'''

  def test_the_module_says_the_headline_is_never_matched_to_a_commodity(self):
    '''The prohibition is written down where a reader will find it.

    A test that only checks behaviour pins the behaviour; it does not
    stop the next person from "fixing" the module by adding a name match
    and updating the fixture. Asserting the promise is in the module
    docstring means the docstring and the code have to agree, which is the
    pairing this repository's process lessons ask for.
    '''
    source = Path(news_module_path()).read_text(encoding='utf-8')
    assert 'never matched against a commodity' in source, (
        'graph/news.py no longer states that a headline is never matched '
        'against a commodity name, so the prohibition a reviewer relies on '
        'has been deleted from the file that implements it')
    assert 'not by matching text' in Path(
      news_view_path()).read_text(encoding='utf-8'), (
        'the view no longer states that a reading is attached through the '
        'graph rather than by matching the headline')

  def test_the_ponytail_comment_is_present_with_a_ceiling(self):
    '''Deliberate simplifications are labelled, as the repo requires.

    The simplifications here are real and named in the code: a shortest
    path per symbol, no hop decay, and no measurement of whether a
    commodity move ever moved the stock. A reviewer needs to be able to
    see the ceiling without reading the diff.
    '''
    source = Path(news_module_path()).read_text(encoding='utf-8')
    assert 'PONYTAIL:' in source
    assert 'Ceiling:' in source
    assert 'Upgrade path:' in source


def news_module_path() -> str:
  '''Return the path of the module under test.

  Returns:
    Absolute path string, so a failure message names the file rather than
    an ambiguous relative one.
  '''
  return str(Path(news_module.__file__))


def news_view_path() -> str:
  '''Return the path of the view script.

  Returns:
    Absolute path string of ``web/graph.js``.
  '''
  return str(Path(__file__).resolve().parents[1]
             / 'src' / 'stock_rl' / 'web' / 'graph.js')


# A guard against a fixture that stops being meaningful: the two-source
# helper must clear the source minimum, or every "usable" assertion in this
# file would be asserting nothing.
def test_the_two_source_fixture_clears_the_source_minimum() -> None:
  '''The fixture is capable of being usable, so the refusal tests bite.

  Without this, an implementation that refused every aggregate would pass
  every test above, because none of them would have had a usable case to
  fail.
  '''
  news = readings_for_node(nifty50_seed(), CRUDE, two_sources('ONGC'), BAR)
  assert news.entries[0].aggregate.usable, (
      'two_sources() no longer produces a usable aggregate, so every '
      'single-source assertion in this file is vacuous')
  assert news.entries[0].aggregate.source_count == 2


def test_the_fixture_symbols_are_all_registry_rows() -> None:
  '''Every symbol this file asserts on is one the linker actually knows.

  The provenance tests read the linker reason, so a fixture symbol that
  is not a registry row would make them assert on the failure branch and
  hide a regression in the success branch.
  '''
  for symbol in CRUDE_NAMES + ('SHREECEM', 'HDFCBANK', 'RELIANCE', 'TITAN'):
    found = resolve(symbol)
    assert found.status is LinkStatus.RESOLVED, (
        f'{symbol} is not a registry row, so the provenance assertions in '
        f'this file are reading the failure branch: {found.reason}')


def test_the_look_ahead_refusal_names_the_offending_reading() -> None:
  '''The refusal must be diagnosable from its own message.

  An error that says "leak" without naming the symbol and the timestamp
  sends whoever hits it back to the data. Both are in the message, which
  is also what makes this a usable assertion rather than a bare raises.
  '''
  leak = SentimentReading('ONGC', 0.9, 'news', 0.9,
                          BAR + timedelta(hours=6), 'cnbc')
  with pytest.raises(LookAheadError) as caught:
    readings_for_node(nifty50_seed(), CRUDE, [leak], BAR)
  message = str(caught.value)
  assert 'ONGC' in message
  assert re.search(r'\d{4}-\d{2}-\d{2}', message), (
      f'the refusal names no timestamp, so it cannot be diagnosed: {message!r}')
