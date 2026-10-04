#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the static dependency graph.

The traversal tests matter more than the serialisation tests. A JSON
round-trip either works or it does not and you see it; a traversal that
quietly returns the wrong universe because a depth default was one hop
too small produces a feature value that looks fine and is wrong, which
is the same failure mode as look-ahead and just as hard to see.

The cyclic-graph test is the one that must never be removed. A
hand-maintained dependency graph acquires a cycle the first time someone
adds a plausible pair of contradictory edges, and the symptom of not
guarding is a backtest that hangs rather than an exception.
'''

import pytest

from stock_rl.graph import (
  DependencyGraph,
  Edge,
  EdgeKind,
  Node,
  NodeKind,
  affected_stocks,
  cycles,
  default_max_depth,
  dependencies,
  dependents,
  has_cycle,
  make_key,
  max_traversal_depth,
  nifty50_seed,
  nifty50_stocks,
  parse_key,
  schema_version,
  sector_labels,
  stock_sectors,
)

CRUDE = 'commodity:crude'
COAL = 'commodity:coal'
LIQUIDITY = 'macro:global_liquidity'
OPEC = 'event:opec_supply_cut'


def make_graph() -> DependencyGraph:
  '''Return the starter Nifty-50 graph.

  Returns:
    A fresh graph, so a test that mutates it cannot affect another.
  '''
  return nifty50_seed()


def cyclic_graph() -> DependencyGraph:
  '''Return a small graph containing a dependency loop.

  The loop is ``stock:A -> macro:x -> stock:B -> stock:A``, which is what
  "sector depends on a macro and the macro depends on the sector" looks
  like once it has been written down by hand.

  Returns:
    A graph with four nodes, four edges and one genuine cycle.
  '''
  graph = DependencyGraph()
  for key, kind in (
    ('stock:A', NodeKind.STOCK),
    ('stock:B', NodeKind.STOCK),
    ('stock:C', NodeKind.STOCK),
    ('macro:x', NodeKind.MACRO),
  ):
    graph.add_node(Node(key, kind, key.split(':')[1]))
  for source, target in (
    ('stock:A', 'macro:x'),
    ('macro:x', 'stock:B'),
    ('stock:B', 'stock:A'),
    ('stock:C', 'stock:A'),
  ):
    graph.add_edge(Edge(source, target, EdgeKind.DEPENDS_ON))
  return graph


# --- keys and nodes -------------------------------------------------------

def test_make_key_lowercases_and_namespaces() -> None:
  '''Names are slugs; stock symbols keep their exchange casing.'''
  assert make_key(NodeKind.COMMODITY, 'crude') == CRUDE
  assert make_key(NodeKind.COMMODITY, 'Crude Oil') == 'commodity:crude_oil'
  assert make_key(NodeKind.MACRO, 'USD/INR') == 'macro:usd_inr'
  assert make_key(NodeKind.STOCK, 'ongc') == 'stock:ONGC'


def test_make_key_rejects_empty_name() -> None:
  '''A name that normalises to nothing is not a node.'''
  with pytest.raises(ValueError):
    make_key(NodeKind.COMMODITY, '   ')


def test_parse_key_round_trips_every_kind() -> None:
  '''Every kind parses back to the kind it was built from.'''
  for kind, expected in ((NodeKind.STOCK, 'CRUDE'),
                         (NodeKind.COMMODITY, 'crude'),
                         (NodeKind.SECTOR, 'crude'),
                         (NodeKind.MACRO, 'crude'),
                         (NodeKind.EVENT, 'crude')):
    key = make_key(kind, 'crude')
    assert parse_key(key) == (kind, expected)


def test_parse_key_rejects_junk() -> None:
  '''A key with no namespace, or an unknown one, is refused.'''
  with pytest.raises(ValueError):
    parse_key('crude')
  with pytest.raises(ValueError):
    parse_key('planet:mars')


def test_node_rejects_kind_namespace_mismatch() -> None:
  '''A commodity key filed as a macro is refused at construction.'''
  with pytest.raises(ValueError):
    Node(CRUDE, NodeKind.MACRO, 'Crude Oil')


def test_node_rejects_blank_label_and_key() -> None:
  '''Both the key and the label must be readable.'''
  with pytest.raises(ValueError):
    Node(' ', NodeKind.MACRO, 'x')
  with pytest.raises(ValueError):
    Node('macro:usdinr', NodeKind.MACRO, ' ')


def test_node_dict_round_trip() -> None:
  '''A node survives its own JSON shape.'''
  node = Node(CRUDE, NodeKind.COMMODITY, 'Crude Oil', 'design doc example')
  assert Node.from_dict(node.to_dict()) == node


def test_node_from_dict_reports_missing_and_unknown() -> None:
  '''Malformed node records fail with a message that names the fault.'''
  with pytest.raises(ValueError, match='missing fields'):
    Node.from_dict({'key': CRUDE, 'kind': 'commodity'})
  with pytest.raises(ValueError, match='unknown kind'):
    Node.from_dict({'key': CRUDE, 'kind': 'moon', 'label': 'x'})


def test_node_name_and_is_stock() -> None:
  '''Namespace removal and the stock predicate agree with the kind.'''
  stock = Node('stock:RELIANCE', NodeKind.STOCK, 'RELIANCE')
  assert stock.name == 'RELIANCE'
  assert stock.is_stock
  assert not Node(CRUDE, NodeKind.COMMODITY, 'Crude Oil').is_stock


def test_stock_sector_table_is_a_copy() -> None:
  '''Mutating the returned map must not corrupt the module constant.'''
  sectors = stock_sectors()
  sectors['RELIANCE'] = 'tampered'
  assert stock_sectors()['RELIANCE'] == 'oil_gas'
  assert len(nifty50_stocks) == len(stock_sectors())
  assert all(sector in sector_labels for _, sector in nifty50_stocks)


# --- edges and the container ---------------------------------------------

def test_edge_rejects_blank_endpoint_and_negative_weight() -> None:
  '''An edge needs two endpoints and a non-negative weight.'''
  with pytest.raises(ValueError):
    Edge(' ', 'macro:x', EdgeKind.DEPENDS_ON)
  with pytest.raises(ValueError):
    Edge('stock:A', ' ', EdgeKind.DEPENDS_ON)
  with pytest.raises(ValueError):
    Edge('stock:A', 'macro:x', EdgeKind.DEPENDS_ON, weight=-1.0)


def test_edge_dict_round_trip_and_errors() -> None:
  '''Edges round-trip, and a bad kind is named rather than guessed.'''
  edge = Edge('stock:ONGC', CRUDE, EdgeKind.DEPENDS_ON, 0.5)
  assert Edge.from_dict(edge.to_dict()) == edge
  with pytest.raises(ValueError, match='missing fields'):
    Edge.from_dict({'source': 'stock:ONGC', 'target': CRUDE})
  with pytest.raises(ValueError, match='unknown kind'):
    Edge.from_dict({'source': 'a', 'target': 'b', 'kind': 'likes'})


def test_add_edge_requires_known_endpoints() -> None:
  '''A dangling edge is a typo, and a typo must be loud.'''
  graph = DependencyGraph()
  graph.add_node(Node('stock:A', NodeKind.STOCK, 'A'))
  with pytest.raises(ValueError, match='unknown node'):
    graph.add_edge(Edge('stock:A', 'macro:ghost', EdgeKind.DEPENDS_ON))


def test_add_node_is_idempotent_but_rejects_contradiction() -> None:
  '''The same node twice is fine; a different label is not.'''
  graph = DependencyGraph()
  graph.add_node(Node(CRUDE, NodeKind.COMMODITY, 'Crude Oil'))
  graph.add_node(Node(CRUDE, NodeKind.COMMODITY, 'Crude Oil'))
  assert graph.node_count == 1
  with pytest.raises(ValueError, match='conflicting'):
    graph.add_node(Node(CRUDE, NodeKind.COMMODITY, 'Brent'))


def test_add_edge_rejects_duplicates() -> None:
  '''A duplicate edge would double any naive degree count.'''
  graph = cyclic_graph()
  with pytest.raises(ValueError, match='duplicate'):
    graph.add_edge(Edge('stock:A', 'macro:x', EdgeKind.DEPENDS_ON))


def test_container_lookups_and_indexes() -> None:
  '''Counts, membership, symbols and both adjacency directions.'''
  graph = make_graph()
  assert len(graph) == graph.node_count
  assert graph.has('stock:RELIANCE')
  assert not graph.has('stock:GHOST')
  assert graph.node('stock:RELIANCE').label == 'RELIANCE'
  assert 'RELIANCE' in graph.symbols()
  with pytest.raises(KeyError):
    graph.node('stock:GHOST')
  assert CRUDE in dependencies(graph, 'stock:ONGC')
  assert 'stock:ONGC' in dependents(graph, CRUDE)
  assert not dependencies(graph, 'macro:ghost')
  assert not dependents(graph, 'macro:ghost')


def test_edge_budget_flag() -> None:
  '''The 2000-edge research ceiling is surfaced, not enforced.'''
  graph = make_graph()
  assert not graph.edge_budget_exceeded
  for index in range(2100):
    source = f'macro:probe{index}'
    target = f'macro:probe{index + 1}'
    graph.add_node(Node(source, NodeKind.MACRO, source))
    graph.add_node(Node(target, NodeKind.MACRO, target))
    graph.add_edge(Edge(source, target, EdgeKind.DEPENDS_ON))
  assert graph.edge_budget_exceeded


def test_graph_json_round_trip(tmp_path) -> None:
  '''The graph is a file: writing and reading it must be lossless.'''
  graph = make_graph()
  text = graph.to_json()
  assert DependencyGraph.from_json(text).to_json() == text
  path = graph.save(tmp_path / 'nested' / 'graph.json')
  assert path.exists()
  assert DependencyGraph.load(path).to_dict() == graph.to_dict()


def test_graph_dict_carries_the_schema_version() -> None:
  '''A loader must be able to refuse a future file, so version it.'''
  payload = make_graph().to_dict()
  assert payload['version'] == schema_version
  assert payload['nodes'] and payload['edges']


def test_from_dict_rejects_bad_documents() -> None:
  '''Wrong version, non-array sections and non-JSON all raise.'''
  with pytest.raises(ValueError, match='unsupported graph schema'):
    DependencyGraph.from_dict({'version': 99, 'nodes': [], 'edges': []})
  with pytest.raises(ValueError, match='JSON arrays'):
    DependencyGraph.from_dict({'version': schema_version,
                               'nodes': {}, 'edges': []})
  with pytest.raises(ValueError, match='not valid JSON'):
    DependencyGraph.from_json('{not json')
  with pytest.raises(ValueError, match='JSON object'):
    DependencyGraph.from_json('[1, 2]')


def test_from_dict_rejects_dangling_edge() -> None:
  '''An edge pointing at an absent node is refused on load too.'''
  with pytest.raises(ValueError, match='unknown node'):
    DependencyGraph.from_dict({
      'version': schema_version,
      'nodes': [{'key': 'stock:A', 'kind': 'stock', 'label': 'A',
                 'note': ''}],
      'edges': [{'source': 'stock:A', 'target': 'stock:GHOST',
                 'kind': 'depends_on', 'weight': 1.0}],
    })


# --- the seed graph -------------------------------------------------------

def test_seed_graph_is_a_starter_not_a_database() -> None:
  '''Inside the researched edge range, and only the five node kinds.'''
  graph = make_graph()
  assert 0 < graph.edge_count <= 2000
  assert not graph.edge_budget_exceeded
  kinds = {node.kind for node in graph.nodes.values()}
  assert kinds == {NodeKind.STOCK, NodeKind.SECTOR, NodeKind.COMMODITY,
                   NodeKind.MACRO, NodeKind.EVENT}


def test_seed_contains_the_design_docs_named_edges() -> None:
  '''The doc's own examples must hold or the seed is fiction.'''
  graph = make_graph()
  assert set(dependencies(graph, 'stock:ONGC')) >= {CRUDE}
  assert set(dependencies(graph, 'stock:RELIANCE')) >= {CRUDE,
                                                        'macro:usdinr'}
  assert CRUDE in dependencies(graph, 'stock:ASIANPAINT')
  assert 'macro:usdinr' in dependencies(graph, 'stock:INFY')
  assert 'macro:repo_rate' in dependencies(graph, 'stock:HDFCBANK')


def test_seed_has_a_sector_for_every_symbol() -> None:
  '''Each listed symbol depends on exactly one sector node.'''
  graph = make_graph()
  for symbol in graph.symbols():
    sectors = [key for key in dependencies(graph, f'stock:{symbol}')
               if key.startswith('sector:')]
    assert len(sectors) == 1


# --- traversal ------------------------------------------------------------

def test_crude_shock_reaches_the_design_docs_named_stocks() -> None:
  '''The doc's worked example: Crude +5% -> ONGC, RELIANCE, ASIANPAINT.'''
  reached = affected_stocks(make_graph(), CRUDE, max_depth=1)
  assert set(reached) == {'ONGC', 'RELIANCE', 'ASIANPAINT'}
  assert set(reached.values()) == {1}


def test_traversal_travels_against_the_stored_edge_direction() -> None:
  '''Information moves from cause to effect, not the other way round.'''
  graph = make_graph()
  assert CRUDE in dependencies(graph, 'stock:ONGC')
  # A shock at a stock reaches nothing: stocks are leaves, which is what
  # makes a missing upstream edge a silent coverage hole worth auditing.
  assert not affected_stocks(graph, 'stock:ONGC', max_depth=3)
  # Depth adds reach, never invented names: crude has no second tier.
  assert (affected_stocks(graph, CRUDE, max_depth=2)
          == affected_stocks(graph, CRUDE, max_depth=1))


def test_depth_bound_is_respected() -> None:
  '''A hop that needs three edges does not appear at depth two.'''
  graph = make_graph()
  assert 'HDFCBANK' not in affected_stocks(graph, LIQUIDITY, max_depth=2)
  deep = affected_stocks(graph, LIQUIDITY, max_depth=3)
  assert deep['HDFCBANK'] == 3
  assert deep['INFY'] == 3
  # SHREECEM reaches coal only through sector:cement.
  assert 'SHREECEM' not in affected_stocks(graph, COAL, max_depth=1)
  assert affected_stocks(graph, COAL, max_depth=2)['SHREECEM'] == 2


def test_default_depth_is_within_the_hard_cap() -> None:
  '''The shipped default must be usable without an argument.'''
  assert 1 <= default_max_depth <= max_traversal_depth
  assert affected_stocks(make_graph(), CRUDE) == affected_stocks(
    make_graph(), CRUDE, default_max_depth)


def test_depth_argument_is_validated() -> None:
  '''Zero, negative and absurd depths are refused, not clamped.'''
  graph = make_graph()
  with pytest.raises(ValueError, match='max_depth must be >= 1'):
    affected_stocks(graph, CRUDE, max_depth=0)
  with pytest.raises(ValueError, match='max_depth must be <='):
    affected_stocks(graph, CRUDE, max_depth=max_traversal_depth + 1)
  with pytest.raises(ValueError, match='unknown node'):
    affected_stocks(graph, 'macro:ghost')


def test_event_edges_and_exposure_edges_travel_together() -> None:
  '''affected_by edges are traversed like depends_on edges.'''
  graph = make_graph()
  reached = affected_stocks(graph, OPEC, max_depth=2)
  assert reached['ONGC'] == 1
  assert reached['RELIANCE'] == 2


def test_origin_is_not_reported_as_affected_by_itself() -> None:
  '''A shock at a stock does not make that stock downstream of itself.'''
  assert affected_stocks(make_graph(), 'stock:ONGC', max_depth=3).get(
    'ONGC') is None


def test_cyclic_graph_terminates_and_still_terminates() -> None:
  '''The hang test. A loop must not turn traversal into an infinite loop.'''
  graph = cyclic_graph()
  reached = affected_stocks(graph, 'macro:x', max_depth=max_traversal_depth)
  assert reached == {'A': 1, 'B': 2, 'C': 2}
  again = affected_stocks(graph, 'stock:B', max_depth=max_traversal_depth)
  assert again == {'A': 2, 'C': 3}
  # The depth bound still binds inside a loop: one hop from x reaches
  # only A, even though the loop would otherwise walk back round.
  assert affected_stocks(graph, 'macro:x', max_depth=1) == {'A': 1}
  # C depends on A and nothing depends on C, so C is a leaf.
  assert not affected_stocks(graph, 'stock:C', max_depth=2)


def test_has_cycle_and_cycles_report_the_loop() -> None:
  '''A loop is a review finding, and it is reportable as such.'''
  graph = cyclic_graph()
  assert has_cycle(graph)
  reported = cycles(graph)
  assert ('macro:x', 'stock:B', 'stock:A', 'macro:x') in reported
  assert not has_cycle(make_graph())
  assert not cycles(make_graph())


def test_cycles_arguments_are_validated() -> None:
  '''Cycle enumeration bounds are checked, not assumed.'''
  graph = cyclic_graph()
  with pytest.raises(ValueError, match='max_cycles'):
    cycles(graph, max_cycles=0)
  with pytest.raises(ValueError, match='max_length'):
    cycles(graph, max_length=2)


def test_short_cycles_only_reports_short_loops() -> None:
  '''A length bound is what keeps enumeration off the clock.'''
  graph = cyclic_graph()
  assert cycles(graph, max_length=3) == (
    ('macro:x', 'stock:B', 'stock:A', 'macro:x'),)


def test_self_loop_is_not_reported_as_a_dependency_loop() -> None:
  '''A one-key loop has no review value and is walked through safely.'''
  graph = DependencyGraph()
  graph.add_node(Node('macro:x', NodeKind.MACRO, 'X'))
  graph.add_edge(Edge('macro:x', 'macro:x', EdgeKind.DEPENDS_ON))
  assert has_cycle(graph)
  assert not cycles(graph)
  assert not affected_stocks(graph, 'macro:x', max_depth=2)


def test_from_dict_treats_absent_sections_as_empty() -> None:
  '''A version-only document is an empty graph, not a failure.'''
  empty = DependencyGraph.from_dict({'version': schema_version})
  assert empty.node_count == 0
  assert empty.edge_count == 0
  assert DependencyGraph.from_json(empty.to_json()).to_json() == (
    empty.to_json())


def test_from_dict_rejects_wrongly_typed_sections() -> None:
  '''An object where an array belongs is a corrupt file.'''
  with pytest.raises(ValueError, match='JSON arrays'):
    DependencyGraph.from_dict({'version': schema_version,
                               'nodes': {}, 'edges': []})
  with pytest.raises(ValueError, match='JSON arrays'):
    DependencyGraph.from_dict({'version': schema_version,
                               'nodes': [], 'edges': 'many'})


def test_cycle_enumeration_stops_at_max_cycles() -> None:
  '''The count bound is what stops a tangle from becoming a hang.'''
  graph = DependencyGraph()
  for name in ('stock:A', 'stock:B', 'stock:C', 'stock:D'):
    graph.add_node(Node(name, NodeKind.STOCK, name.split(':')[1]))
  graph.add_edge(Edge('stock:A', 'stock:B', EdgeKind.DEPENDS_ON))
  graph.add_edge(Edge('stock:B', 'stock:C', EdgeKind.DEPENDS_ON))
  graph.add_edge(Edge('stock:C', 'stock:D', EdgeKind.DEPENDS_ON))
  graph.add_edge(Edge('stock:D', 'stock:A', EdgeKind.DEPENDS_ON))
  found = cycles(graph, max_cycles=1)
  assert len(found) == 1
  # A four-key loop needs max_length 4; at three it is off the record.
  assert not cycles(graph, max_length=3)
  assert cycles(graph, max_length=4)


def test_load_reports_a_missing_file() -> None:
  '''A missing graph file is an OSError, not an empty graph.'''
  with pytest.raises(OSError):
    DependencyGraph.load('/nonexistent/stock_rl/graph.json')


def test_seed_carries_no_temporal_claim() -> None:
  '''Documented ceiling: the graph is static, with no validity dates.'''
  payload = make_graph().to_dict()
  assert all(set(node) == {'key', 'kind', 'label', 'note'}
             for node in payload['nodes'])
  assert all(set(edge) == {'source', 'target', 'kind', 'weight'}
             for edge in payload['edges'])
