#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Edges of the static market-dependency graph, and the graph itself.

Both edge kinds point the same way: **from the affected node to the
cause**. ``stock:ONGC depends_on commodity:crude`` reads as ONGC depends
on crude, and information therefore travels from crude to ONGC, i.e.
*against* the stored direction. ``affected_by`` is the same orientation
with an event or a news cause at the far end, so
``stock:HDFCBANK affected_by event:rbi_rate_decision`` is a direct
exposure while ``sector:banking affected_by event:rbi_rate_decision``
is an exposure every bank inherits through the sector.

Keeping one orientation is what lets traversal be one BFS rather than
two, and it means a new edge kind cannot silently reverse what a
traversal means. Traversal lives in :mod:`stock_rl.graph.traversal`.

**A file, not a database.** The design doc's answer is Neo4j with a
gRPC client. The research review puts a Nifty-50 dependency graph at
~300-2000 static edges and calls Neo4j "second only to Redis" in
indefensibility; the same review notes that a Cypher dependency breaks
the standard-library isolation which makes backtests reproducible. So the
graph serialises to JSON with a schema version, and
:meth:`DependencyGraph.save` writes a file you can diff.

**The seed graph is a starter, not gospel.** :func:`nifty50_seed`
encodes common sector reasoning, and it is a *hypothesis* about
sensitivity, not a measured one. Nothing here was fitted to returns, and
nothing here should be treated as evidence that a crude move moves
ONGC. It exists so a traversal is testable and so a reviewer has
something concrete to correct. Review it before using it.

PONYTAIL: two edge kinds, one weight, no hypergraph. Ceiling: no temporal
validity, no per-stock beta, no path-specific multipliers -- a shock
three hops out has the same weight as one hop out unless the caller
decays it. Upgrade path: ``schema_version`` plus an optional edge field.
'''

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from stock_rl.graph.nodes import (
  Node,
  NodeKind,
  make_key,
  nifty50_stocks,
  sector_labels,
)

__all__ = [
  'DependencyGraph',
  'Edge',
  'EdgeKind',
  'nifty50_seed',
  'schema_version',
]

#: Version of the JSON schema. A loader rejects anything else rather than
#: guessing at a field it does not understand.
schema_version = 1


class EdgeKind(StrEnum):
  '''The two edge kinds the design doc asks for.

  Attributes:
    DEPENDS_ON: The source node's exposure flows from the target. The
      design doc's example: a stock depends on crude.
    AFFECTED_BY: The source node is affected by the target, where the
      target is typically an event rather than a price. Structurally
      identical to :attr:`depends_on` in the direction information
      travels; it exists because an event edge and an exposure edge mean
      different things to a reviewer even though traversal treats them
      alike.
  '''

  DEPENDS_ON = 'depends_on'
  AFFECTED_BY = 'affected_by'


@dataclass(frozen=True, slots=True)
class Edge:
  '''One directed dependency edge.

  Attributes:
    source: Node key of the affected party, e.g. ``'stock:ONGC'``.
    target: Node key of the cause, e.g. ``'commodity:crude'``.
    kind: One of :class:`EdgeKind`.
    weight: Relative strength in ``[0, inf)``, 1.0 by default. Purely
      declarative metadata: traversal returns hop counts, not weighted
      sums, so a weight never smuggles in an unstated model.
  '''

  source: str
  target: str
  kind: EdgeKind
  weight: float = 1.0

  def __post_init__(self) -> None:
    '''Validate the edge at construction time.

    Raises:
      ValueError: If an endpoint is blank or the weight is negative.
    '''
    if not self.source.strip():
      raise ValueError('edge source must not be empty')
    if not self.target.strip():
      raise ValueError('edge target must not be empty')
    if self.weight < 0.0:
      raise ValueError(f'edge weight must be >= 0, got {self.weight}')

  def to_dict(self) -> dict[str, object]:
    '''Serialise to the JSON object shape used by the graph file.

    Returns:
      Mapping with ``source``, ``target``, ``kind`` and ``weight``.
    '''
    return {
      'source': self.source,
      'target': self.target,
      'kind': self.kind.value,
      'weight': self.weight,
    }

  @classmethod
  def from_dict(cls, payload: dict[str, object]) -> Edge:
    '''Rebuild an edge from its serialised form.

    Args:
      payload: Mapping produced by :meth:`to_dict`.

    Returns:
      The reconstructed edge.

    Raises:
      ValueError: If a required field is absent or the kind is unknown.
    '''
    missing = {'source', 'target', 'kind'} - set(payload)
    if missing:
      raise ValueError(f'edge record missing fields: {sorted(missing)}')
    source = str(payload['source'])
    target = str(payload['target'])
    raw_kind = payload['kind']
    try:
      kind = EdgeKind(str(raw_kind))
    except ValueError as exc:
      raise ValueError(
        f'edge {source!r} -> {target!r} has unknown '
        f'kind {raw_kind!r}') from exc
    return cls(
      source=source,
      target=target,
      kind=kind,
      weight=float(payload.get('weight', 1.0)),
    )


class DependencyGraph:
  '''An in-memory static dependency graph with two adjacency indexes.

  Attributes:
    nodes: Mapping of node key to :class:`~stock_rl.graph.nodes.Node`.
    edges: The edges, in insertion order.
  '''

  def __init__(self) -> None:
    '''Build an empty graph.'''
    self.nodes: dict[str, Node] = {}
    self.edges: list[Edge] = []
    self._seen: set[tuple[str, str, str]] = set()
    self._causes: dict[str, list[str]] = {}
    self._exposed: dict[str, list[str]] = {}

  def __len__(self) -> int:
    '''Return the node count.'''
    return len(self.nodes)

  @property
  def node_count(self) -> int:
    '''Return the number of nodes.'''
    return len(self.nodes)

  @property
  def edge_count(self) -> int:
    '''Return the number of edges.'''
    return len(self.edges)

  @property
  def edge_budget_exceeded(self) -> bool:
    '''Return True when the graph is larger than the reviewed range.

    The research review sizes a Nifty-50 dependency graph at ~300-2000
    static edges. Crossing 2000 means the graph has either grown stale
    or become a database in disguise, and either way a reviewer should
    look before a traversal is trusted. Note that the **seed** graph is
    far below the low end: :func:`nifty50_seed` builds 76 edges, so this
    flag is False on it by a factor of four, not by a little.
    '''
    return self.edge_count > 2000

  def add_node(self, node: Node) -> Node:
    '''Add a node, or return the existing one with the same key.

    Idempotent on purpose: the seed builder adds sectors and symbols from
    two different tables and the overlap is intentional.

    Args:
      node: Node to add.

    Returns:
      The node now in the graph.

    Raises:
      ValueError: If the key is already present with a different kind or
        a different label, which would make the file self-contradictory.
    '''
    existing = self.nodes.get(node.key)
    if existing is not None:
      if existing.kind is not node.kind or existing.label != node.label:
        raise ValueError(
          f'conflicting node definitions for {node.key}: '
          f'{existing.kind.value}/{existing.label!r} vs '
          f'{node.kind.value}/{node.label!r}')
      return existing
    self.nodes[node.key] = node
    return node

  def add_edge(self, edge: Edge) -> Edge:
    '''Add an edge between two existing nodes.

    Args:
      edge: Edge to add.

    Returns:
      The edge now in the graph.

    Raises:
      ValueError: If either endpoint is unknown, or the edge already
        exists. Duplicates are rejected rather than merged because a
        duplicated edge silently doubles any naive degree count, which
        is the sort of error that survives into a feature value.
    '''
    for endpoint in (edge.source, edge.target):
      if endpoint not in self.nodes:
        raise ValueError(f'unknown node {endpoint!r} on edge {edge!r}')
    signature = (edge.source, edge.target, edge.kind.value)
    if signature in self._seen:
      raise ValueError(f'duplicate edge {edge.source} -> {edge.target}')
    self._seen.add(signature)
    self.edges.append(edge)
    self._causes.setdefault(edge.source, []).append(edge.target)
    self._exposed.setdefault(edge.target, []).append(edge.source)
    return edge

  def has(self, key: str) -> bool:
    '''Return True when ``key`` is a node in this graph.

    Args:
      key: Node key to test.
    '''
    return key in self.nodes

  def node(self, key: str) -> Node:
    '''Return one node by key.

    Args:
      key: Node key to look up.

    Returns:
      The node.

    Raises:
      KeyError: If the key is not in the graph.
    '''
    return self.nodes[key]

  def symbols(self) -> tuple[str, ...]:
    '''Return every listed symbol in the graph, sorted.

    Returns:
      Bare NSE symbols without their ``stock:`` namespace.
    '''
    return tuple(sorted(node.name for node in self.nodes.values()
                        if node.is_stock))

  def causes_of(self, key: str) -> tuple[str, ...]:
    '''Return the nodes ``key`` points at.

    Args:
      key: Node key.

    Returns:
      Targets of this node's outgoing edges, in insertion order.
    '''
    return tuple(self._causes.get(key, ()))

  def exposed_to(self, key: str) -> tuple[str, ...]:
    '''Return the nodes that point at ``key``.

    This is the direction information travels: a shock at ``key`` reaches
    every node in this tuple.

    Args:
      key: Node key.

    Returns:
      Sources of this node's incoming edges, in insertion order.
    '''
    return tuple(self._exposed.get(key, ()))

  def to_dict(self) -> dict[str, object]:
    '''Serialise the whole graph to plain JSON-compatible objects.

    Nodes and edges are sorted by key so the file has a stable diff,
    which matters because the graph is meant to be reviewed.

    Returns:
      Mapping with ``version``, ``nodes`` and ``edges``.
    '''
    return {
      'version': schema_version,
      'nodes': [self.nodes[key].to_dict() for key in sorted(self.nodes)],
      'edges': [edge.to_dict()
                for edge in sorted(self.edges,
                                   key=lambda item: (item.source, item.target,
                                                     item.kind.value))],
    }

  @classmethod
  def from_dict(cls, payload: dict[str, object]) -> DependencyGraph:
    '''Rebuild a graph from its serialised form.

    Args:
      payload: Mapping produced by :meth:`to_dict`.

    Returns:
      The reconstructed graph.

    Raises:
      ValueError: If the schema version is unsupported, a node or edge
        record is malformed, or an edge references a missing node.
    '''
    version = payload.get('version')
    if version != schema_version:
      raise ValueError(
        f'unsupported graph schema version {version!r}, '
        f'expected {schema_version}')
    nodes = payload.get('nodes')
    edges = payload.get('edges')
    if nodes is None:
      nodes = []
    if edges is None:
      edges = []
    if not isinstance(nodes, list) or not isinstance(edges, list):
      raise ValueError('nodes and edges must be JSON arrays')
    graph = cls()
    for record in nodes:
      graph.add_node(Node.from_dict(record))
    for record in edges:
      graph.add_edge(Edge.from_dict(record))
    return graph

  def to_json(self) -> str:
    '''Serialise the graph to an indented JSON string.

    Returns:
      Pretty-printed JSON, stable across runs of the same graph.
    '''
    return json.dumps(self.to_dict(), indent=2)

  @classmethod
  def from_json(cls, text: str) -> DependencyGraph:
    '''Rebuild a graph from a JSON string.

    Args:
      text: JSON text produced by :meth:`to_json`.

    Returns:
      The reconstructed graph.

    Raises:
      ValueError: If the text is not valid JSON or the payload is
        malformed.
    '''
    try:
      payload = json.loads(text)
    except json.JSONDecodeError as exc:
      raise ValueError(f'graph file is not valid JSON: {exc}') from exc
    if not isinstance(payload, dict):
      raise ValueError('graph file must contain a JSON object')
    return cls.from_dict(payload)

  def save(self, path: str | Path) -> Path:
    '''Write the graph to a JSON file.

    Args:
      path: Destination path. Parent directories are created.

    Returns:
      The path written.
    '''
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(self.to_json() + '\n', encoding='utf-8')
    return target

  @classmethod
  def load(cls, path: str | Path) -> DependencyGraph:
    '''Read a graph from a JSON file.

    Args:
      path: Path written by :meth:`save`.

    Returns:
      The loaded graph.

    Raises:
      OSError: If the file cannot be read.
      ValueError: If the contents are not a valid graph document.
    '''
    return cls.from_json(Path(path).read_text(encoding='utf-8'))


#: Commodity nodes: (name, label, note).
_commodities: tuple[tuple[str, str, str], ...] = (
  ('crude', 'Crude Oil',
   'The design doc\'s worked example: a crude shock reaches '
   'ONGC, RELIANCE, ASIANPAINT.'),
  ('natural_gas', 'Natural Gas', ''),
  ('coal', 'Coal', 'Input cost for both the metals and the cement chain.'),
  ('iron_ore', 'Iron Ore', ''),
  ('copper', 'Copper', 'LME-linked; the metals chain is the main Nifty use.'),
  ('gold', 'Gold', 'Jewellery demand and the consumer-durables basket.'),
)

#: Macro nodes: (name, label, note).
_macros: tuple[tuple[str, str, str], ...] = (
  ('usdinr', 'USD/INR', 'Exporter margin headwind, IT revenue tailwind.'),
  ('repo_rate', 'RBI Repo Rate',
   'Bank lending rate and the discount rate everything else discounts '
   'at. Priced in before it is announced, so this edge is about the '
   'surprise, not the level.'),
  ('gst', 'Goods and Services Tax',
   'Domestic demand and packer pricing power.'),
  ('fii_net', 'FII/DII Net Flow',
   'ONE India-wide number, so it is a global state variable and never a '
   'per-stock feature. Handing it to a per-stock "FII Flow Agent" is '
   'the design doc\'s structurally incoherent step.'),
  ('global_liquidity', 'Global Liquidity',
   'Second-order driver of foreign flow into India.'),
)

#: Event nodes: (name, label, note).
_events: tuple[tuple[str, str, str], ...] = (
  ('opec_supply_cut', 'OPEC Supply Cut', ''),
  ('rbi_rate_decision', 'RBI Policy Decision', ''),
  ('monsoon', 'Monsoon', 'Rural demand and agri-linked input costs.'),
)

#: Sector-level commodity and macro inputs.
_sector_inputs: dict[str, tuple[str, ...]] = {
  'auto': ('usdinr',),
  'banking': ('repo_rate', 'fii_net'),
  'cement': ('coal',),
  'consumer': ('gold',),
  'fmcg': ('gst',),
  'it': ('usdinr',),
  'metals': ('copper', 'iron_ore', 'coal'),
  'oil_gas': ('crude', 'natural_gas'),
  'paints': ('crude', 'gst'),
  'pharma': ('usdinr', 'gst'),
  'telecom': ('repo_rate',),
  'utilities': ('coal', 'natural_gas'),
}

#: Extra direct inputs per symbol, beyond the symbol's sector and the
#: sector's own inputs. The design doc's named cases are the ones worth
#: stating explicitly; the rest add only what is common sector reasoning
#: rather than inventing precision.
#:
#: What this table costs, exactly: the seed graph is **76** edges over
#: **55** nodes -- 29 stock-to-sector, 19 sector-to-input, 21 of these
#: direct inputs, 2 macro chains and 5 event edges. The review sizes a
#: Nifty-50 dependency graph at ~300-2000 static edges, so the seed sits
#: at about a **quarter of the stated low end**, not near it. That is a
#: known and deliberate gap, stated here rather than implied: the table
#: covers 14 of the 29 starter symbols, so 15 carry no direct input at
#: all beyond their sector, and a reviewer who wants a denser graph
#: should add rows here and say why.
#: :attr:`DependencyGraph.edge_budget_exceeded` still trips at 2000.
_stock_inputs: dict[str, tuple[str, ...]] = {
  'ASIANPAINT': ('crude',),
  'BHARTIARTL': ('usdinr',),
  'HDFCBANK': ('repo_rate',),
  'HINDALCO': ('copper', 'usdinr'),
  'INFY': ('usdinr',),
  'ITC': ('gst',),
  'JSWSTEEL': ('coal', 'iron_ore'),
  'ONGC': ('crude',),
  'POWERGRID': ('coal',),
  'RELIANCE': ('crude', 'usdinr'),
  'TATASTEEL': ('coal', 'iron_ore'),
  'TATAMOTORS': ('usdinr', 'gst'),
  'TITAN': ('gold', 'usdinr'),
  'ULTRACEMCO': ('coal', 'natural_gas'),
}

#: Macro-to-macro chains. Two hops are enough to demonstrate a bounded
#: traversal reaching a bank, which is the case a depth default has to
#: get right.
_macro_chains: tuple[tuple[str, str], ...] = (
  ('fii_net', 'global_liquidity'),
  ('usdinr', 'fii_net'),
)

#: Sector-level event exposures: (sector, event).
_sector_events: tuple[tuple[str, str], ...] = (
  ('banking', 'rbi_rate_decision'),
  ('fmcg', 'monsoon'),
  ('oil_gas', 'opec_supply_cut'),
)

#: Direct event exposures on a symbol: (symbol, event).
_stock_events: tuple[tuple[str, str], ...] = (
  ('HDFCBANK', 'rbi_rate_decision'),
  ('ONGC', 'opec_supply_cut'),
)


def nifty50_seed() -> DependencyGraph:
  '''Build the starter Nifty-50 dependency graph.

  **Review this before trusting it.** Every edge is common sector
  reasoning, not a measured sensitivity: nothing here was fitted to
  returns, no beta was estimated, and no edge carries a date. The
  purpose is to make a traversal testable and to give a reviewer
  something concrete to correct -- a graph nobody has argued with is a
  graph nobody has looked at.

  Structure of the seed, and the reason it is shaped this way:

  * The design doc's worked example is the test of the graph: a crude
    shock reaches ``ONGC``, ``RELIANCE`` and ``ASIANPAINT`` at one hop
    each, because those are the edges the doc itself names.
  * Depth is load-bearing. ``SHREECEM`` reaches ``commodity:coal`` only
    through ``sector:cement``, and every bank reaches
    ``macro:global_liquidity`` through ``macro:fii_net`` and then
    ``sector:banking``. A traversal with too small a depth silently
    returns the wrong universe, which is why depth is an argument and
    not a constant buried in the call site.
  * ``macro:fii_net`` exists as a node precisely so a reviewer can see
    that it is global. It is not a per-stock feature and must not become
    one; the research review is explicit that a per-stock "FII flow
    agent" is structurally incoherent.

  Returns:
    A fresh graph; the module tables are not shared between calls.
  '''
  graph = DependencyGraph()
  for name, label, note in _commodities:
    graph.add_node(Node(make_key(NodeKind.COMMODITY, name),
                        NodeKind.COMMODITY, label, note))
  for name, label, note in _macros:
    graph.add_node(Node(make_key(NodeKind.MACRO, name),
                        NodeKind.MACRO, label, note))
  for name, label, note in _events:
    graph.add_node(Node(make_key(NodeKind.EVENT, name),
                        NodeKind.EVENT, label, note))
  for _, sector in nifty50_stocks:
    graph.add_node(Node(make_key(NodeKind.SECTOR, sector), NodeKind.SECTOR,
                        sector_labels.get(sector, sector)))
  for symbol, _ in nifty50_stocks:
    graph.add_node(Node(make_key(NodeKind.STOCK, symbol), NodeKind.STOCK,
                        symbol))
  for sector, inputs in _sector_inputs.items():
    for input_name in inputs:
      graph.add_edge(Edge(
        source=_sector_key(sector),
        target=make_key(_input_kind(input_name), input_name),
        kind=EdgeKind.DEPENDS_ON,
      ))
  for symbol, sector in nifty50_stocks:
    graph.add_edge(Edge(
      source=_stock_key(symbol),
      target=_sector_key(sector),
      kind=EdgeKind.DEPENDS_ON,
    ))
  for symbol, inputs in _stock_inputs.items():
    for input_name in inputs:
      graph.add_edge(Edge(
        source=_stock_key(symbol),
        target=make_key(_input_kind(input_name), input_name),
        kind=EdgeKind.DEPENDS_ON,
      ))
  for child, parent in _macro_chains:
    graph.add_edge(Edge(
      source=make_key(NodeKind.MACRO, child),
      target=make_key(NodeKind.MACRO, parent),
      kind=EdgeKind.DEPENDS_ON,
    ))
  for sector, event in _sector_events:
    graph.add_edge(Edge(
      source=_sector_key(sector),
      target=make_key(NodeKind.EVENT, event),
      kind=EdgeKind.AFFECTED_BY,
    ))
  for symbol, event in _stock_events:
    graph.add_edge(Edge(
      source=_stock_key(symbol),
      target=make_key(NodeKind.EVENT, event),
      kind=EdgeKind.AFFECTED_BY,
    ))
  return graph


def _sector_key(sector: str) -> str:
  '''Return the graph key for a sector name.

  Args:
    sector: Bare sector name, e.g. ``'oil_gas'``.

  Returns:
    Namespaced node key.
  '''
  return make_key(NodeKind.SECTOR, sector)


def _stock_key(symbol: str) -> str:
  '''Return the graph key for a listed symbol.

  Args:
    symbol: Bare NSE symbol, e.g. ``'ONGC'``.

  Returns:
    Namespaced node key.
  '''
  return make_key(NodeKind.STOCK, symbol)


def _input_kind(name: str) -> NodeKind:
  '''Return the node kind of a bare input name.

  Args:
    name: Bare input name such as ``'crude'`` or ``'usdinr'``.

  Returns:
    :attr:`~stock_rl.graph.nodes.NodeKind.MACRO` when the name is one of
    the macro nodes, otherwise
    :attr:`~stock_rl.graph.nodes.NodeKind.COMMODITY`.

  Raises:
    ValueError: If the name is neither a macro nor a commodity node, so a
      typo in a seed table fails here instead of at traversal time.
  '''
  macros = {name for name, _, _ in _macros}
  if name in macros:
    return NodeKind.MACRO
  commodities = {name for name, _, _ in _commodities}
  if name in commodities:
    return NodeKind.COMMODITY
  raise ValueError(f'{name!r} is not a known macro or commodity node')
