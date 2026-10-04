#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Nodes of the static market-dependency graph.

The design doc asks for a Neo4j "Knowledge Graph Service" whose nodes are
``Stock, Sector, Commodity, Event``. This module is that node type, and it
is a frozen dataclass, because the same research review that rejects the
agent debate also measures the graph at **~300-2000 static edges** for a
Nifty-50 dependency structure. A JVM, a container, a password and a
network hop to answer "which Nifty names care about crude" is a Cypher
dependency on top of a codebase that is deliberately pure standard
library, and that breaks the isolation which makes backtests
reproducible. The graph is a JSON file. See :mod:`stock_rl.graph.edges`.

**Keys are namespaced ``kind:name``** -- ``stock:RELIANCE``,
``commodity:crude``, ``macro:usdinr``. Without the namespace a commodity
named ``coal`` and a sector named ``coal`` would silently become the same
node, and the resulting graph would be wrong in a way that still looks
correct.

**The five kinds are the only kinds.** The design doc's ``Event`` node is
kept because an event is a legitimate cause (``event:opec_supply_cut``),
but nothing else may be added without editing this module, so the set stays
small enough to reason about by hand.

PONYTAIL: five kinds, namespaced keys, and a plain dataclass. Ceiling: no
temporal or probabilistic edges, no weights derived from measured
sensitivities, no graph algorithm beyond bounded traversal. Upgrade path:
add an edge field rather than a node kind -- the JSON schema version in
:mod:`stock_rl.graph.edges` exists for exactly that.
'''

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = [
  'Node',
  'NodeKind',
  'make_key',
  'nifty50_stocks',
  'parse_key',
  'sector_labels',
  'stock_sectors',
]


class NodeKind(StrEnum):
  '''The five node kinds the dependency graph admits.

  Attributes:
    STOCK: A listed NSE symbol. The only kind :mod:`stock_rl.graph
      .traversal` returns by name.
    SECTOR: A sector grouping. Exists because sector-level commodity
      exposure is what a crude shock actually travels through, and
      encoding it once is cheaper than restating it per stock.
    COMMODITY: A traded input. ``crude`` is the design doc's worked
      example.
    MACRO: An economy-wide driver. These are **global**, not per-stock:
      FII/DII flow is one India-wide number and duplicating it across 50
      symbols as 50 features is what the research review calls a
      structurally incoherent per-stock "flow agent".
    EVENT: A dated, discrete cause. Kept minimal and mostly referenced by
      sectors, because event edges are where a hand-maintained graph
      decays fastest.
  '''

  STOCK = 'stock'
  SECTOR = 'sector'
  COMMODITY = 'commodity'
  MACRO = 'macro'
  EVENT = 'event'


#: Namespace prefix per node kind, in ``kind:name`` keys.
_prefixes: dict[NodeKind, str] = {
  NodeKind.STOCK: 'stock:',
  NodeKind.SECTOR: 'sector:',
  NodeKind.COMMODITY: 'commodity:',
  NodeKind.MACRO: 'macro:',
  NodeKind.EVENT: 'event:',
}


#: Display labels for the starter sectors.
sector_labels: dict[str, str] = {
  'auto': 'Automobiles',
  'banking': 'Banking',
  'cement': 'Cement',
  'consumer': 'Consumer Durables',
  'fmcg': 'Fast Moving Consumer Goods',
  'it': 'Information Technology',
  'metals': 'Metals and Mining',
  'oil_gas': 'Oil and Gas',
  'paints': 'Paints and Chemicals',
  'pharma': 'Pharmaceuticals',
  'telecom': 'Telecommunications',
  'utilities': 'Power and Utilities',
}


#: Starter symbol-to-sector table for a Nifty-50 style universe.
#:
#: NOT the index. Sector classification changes with every index review,
#: and a stale symbol here fails loudly at the dependency lookup rather
#: than silently widening a traversal.
nifty50_stocks: tuple[tuple[str, str], ...] = (
  ('RELIANCE', 'oil_gas'),
  ('ONGC', 'oil_gas'),
  ('ASIANPAINT', 'paints'),
  ('HINDALCO', 'metals'),
  ('JSWSTEEL', 'metals'),
  ('TATASTEEL', 'metals'),
  ('INFY', 'it'),
  ('TCS', 'it'),
  ('HCLTECH', 'it'),
  ('WIPRO', 'it'),
  ('HDFCBANK', 'banking'),
  ('ICICIBANK', 'banking'),
  ('SBIN', 'banking'),
  ('AXISBANK', 'banking'),
  ('KOTAKBANK', 'banking'),
  ('TATAMOTORS', 'auto'),
  ('MARUTI', 'auto'),
  ('TVSMOTOR', 'auto'),
  ('SUNPHARMA', 'pharma'),
  ('DRREDDY', 'pharma'),
  ('CIPLA', 'pharma'),
  ('HINDUNILVR', 'fmcg'),
  ('ITC', 'fmcg'),
  ('BHARTIARTL', 'telecom'),
  ('NTPC', 'utilities'),
  ('POWERGRID', 'utilities'),
  ('ULTRACEMCO', 'cement'),
  ('SHREECEM', 'cement'),
  ('TITAN', 'consumer'),
)


def _slug(name: str) -> str:
  '''Normalise a node name into the ``name`` half of a key.

  Args:
    name: Free-form name, e.g. ``"Crude Oil"``.

  Returns:
    Lowercase name with runs of non-alphanumeric characters collapsed to
    a single underscore.
  '''
  out: list[str] = []
  previous_underscore = True
  for character in name.strip().lower():
    if character.isalnum():
      out.append(character)
      previous_underscore = False
    elif not previous_underscore:
      out.append('_')
      previous_underscore = True
  return ''.join(out).strip('_')


def make_key(kind: NodeKind, name: str) -> str:
  '''Build the canonical namespaced key for a node.

  Args:
    kind: Node kind.
    name: Unqualified name, e.g. ``"ONGC"`` or ``"Crude Oil"``.

  Returns:
    Key of the form ``kind:name``, e.g. ``'stock:RELIANCE'``. Names are
    lowercased slugs, except stock keys, which are uppercased because the
    exchange symbol is the identity.

  Raises:
    ValueError: If the name normalises to nothing.
  '''
  slug = _slug(name)
  clean = slug.upper() if kind is NodeKind.STOCK else slug
  if not clean:
    raise ValueError(f'node name must not be empty, got {name!r}')
  return f'{_prefixes[NodeKind(kind)]}{clean}'


def parse_key(key: str) -> tuple[NodeKind, str]:
  '''Split a namespaced key into its kind and name.

  Args:
    key: Key of the form ``kind:name``.

  Returns:
    Tuple of (node kind, name).

  Raises:
    ValueError: If the key has no namespace or an unknown one.
  '''
  prefix, separator, name = key.partition(':')
  if not separator or not name:
    raise ValueError(f'malformed node key: {key!r}')
  for kind, candidate in _prefixes.items():
    if candidate == f'{prefix}:':
      return kind, name
  raise ValueError(f'unknown node kind prefix in {key!r}')


def stock_sectors() -> dict[str, str]:
  '''Return the starter symbol-to-sector mapping.

  Returns:
    Mapping of symbol to sector name, a copy of :data:`nifty50_stocks`
    so a caller cannot mutate the module constant.
  '''
  return dict(nifty50_stocks)


@dataclass(frozen=True, slots=True)
class Node:
  '''One node of the dependency graph.

  Attributes:
    key: Canonical namespaced identifier, e.g. ``'commodity:crude'``.
    kind: One of :class:`NodeKind`.
    label: Human-readable name for logs and audit records.
    note: Provenance or caveat, e.g. why the edge is believed. Present
      because a hand-maintained dependency claim with no stated reason is
      an unfalsifiable assertion, and the research review's whole method
      is refusing to accept unsupported claims.
  '''

  key: str
  kind: NodeKind
  label: str
  note: str = ''

  def __post_init__(self) -> None:
    '''Validate the node at construction time.

    Raises:
      ValueError: If the key or label is blank, or the key's namespace
        disagrees with ``kind``. The second check is what stops a
        ``commodity:crude`` node being filed as a macro.
    '''
    if not self.key.strip():
      raise ValueError('node key must not be empty')
    if not self.label.strip():
      raise ValueError(f'{self.key}: node label must not be empty')
    kind, _ = parse_key(self.key)
    if kind is not self.kind:
      raise ValueError(
        f'{self.key}: key namespace {kind.value} disagrees with '
        f'declared kind {self.kind.value}')

  @property
  def name(self) -> str:
    '''Return the node name with its kind namespace removed.'''
    return parse_key(self.key)[1]

  @property
  def is_stock(self) -> bool:
    '''Return True when this node is a listed symbol.'''
    return self.kind is NodeKind.STOCK

  def to_dict(self) -> dict[str, str]:
    '''Serialise to the JSON object shape used by the graph file.

    Returns:
      Mapping with ``key``, ``kind``, ``label`` and ``note``.
    '''
    return {
      'key': self.key,
      'kind': self.kind.value,
      'label': self.label,
      'note': self.note,
    }

  @classmethod
  def from_dict(cls, payload: dict[str, object]) -> Node:
    '''Rebuild a node from its serialised form.

    Args:
      payload: Mapping produced by :meth:`to_dict`.

    Returns:
      The reconstructed node.

    Raises:
      ValueError: If a required field is absent or the kind is unknown.
    '''
    missing = {'key', 'kind', 'label'} - set(payload)
    if missing:
      raise ValueError(f'node record missing fields: {sorted(missing)}')
    key = str(payload['key'])
    raw_kind = payload['kind']
    try:
      kind = NodeKind(str(raw_kind))
    except ValueError as exc:
      raise ValueError(
        f'node {key!r} has unknown kind {raw_kind!r}') from exc
    return cls(
      key=key,
      kind=kind,
      label=str(payload['label']),
      note=str(payload.get('note', '')),
    )
