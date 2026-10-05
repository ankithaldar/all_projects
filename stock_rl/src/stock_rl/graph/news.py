#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Joining a graph node to the news that is actually about it.

**The gap this closes.** ``commodity:crude`` is the design doc's worked
example, and nothing in the package could put a headline next to it.
:mod:`stock_rl.sentiment.entity_link` resolves free text to a *listed
symbol* by table lookup, so it has nothing to say about a node that has
no symbol; :mod:`stock_rl.sentiment.score` aggregates readings that are
already filed under a symbol. The graph knows the bridge between the
two, and this module is that bridge.

**The relevance rule, stated once and implemented literally.** A reading
is attached to a node ``N`` when all of the following hold:

1. ``N`` is reachable, within ``max_depth`` hops, to a **stock node**
   ``stock:S``. Traversal follows the direction information travels, which
   is *against* the stored edge direction -- ``exposed_to``, the same
   relation :func:`stock_rl.graph.traversal.affected_stocks` walks.
2. The reading's ``symbol`` is exactly ``S``, the exchange symbol the
   stock node's name carries.

Nothing else attaches a reading. In particular:

* **A headline is never matched against a commodity, a macro or an event
  name.** There is no text similarity, no alias, no substring, nothing.
  A reading filed under ``symbol='crude'`` is attached to nothing at all,
  because ``crude`` is not the name of a stock node and hop 1 of the rule
  above is where a commodity name would have to enter. The only sanctioned
  bridge from a symbolless node to a symbol is the graph.
* **A reading about a symbol the graph does not reach is not attached.**
  Being in the same news day is not relevance.
* **A node with no stock downstream returns empty.** Not a fallback, not
  the nearest sector's news, not "all news for the day". There is nothing
  to look up, and the result says so in a status rather than in prose
  that could be mistaken for a hit.

**Why the dependent-hop rule is defensible rather than a guess.** The
alternative -- resolve the node's own label, e.g. feed ``"Crude Oil"``
to the entity linker -- is the thing this repository has already tried
and reverted. The linker is a lookup table keyed on exchange symbols,
company names and human-written alias rows; ``"Crude Oil"`` is in none of
them, and the only way to make it resolve is to start inferring what text
means. The graph route is the opposite: it never inspects the headline.
It asks "which listed companies does this package already claim are
exposed to crude", gets an answer from a reviewable table, and then uses
:mod:`stock_rl.sentiment.score`, which keys strictly on the symbol the
reading was filed under. Both hops are lookups. That is the whole defence,
and it is also why the answer ships with the path attached: a reader who
does not accept the edge ``sector:oil_gas depends_on commodity:crude`` can
reject every reading that used it, and see which ones those are.

**Provenance is not decoration.** Every returned reading carries the
symbol it was filed under, the hop count, the key path from the node to
that symbol, the ``EdgeKind`` of each stored edge traversed, and the
linker's own reason string for that symbol. A claim with no stated route
is the thing :mod:`stock_rl.graph.nodes` refuses at construction, so this
module does the same at the join.

**Look-ahead is refused, not filtered.** ``decision_bar`` is a required
argument with no default, because the permissive default has been
reintroduced as a bug twice in this repository. The readings that are
candidates for this node go through
:func:`stock_rl.sentiment.score.require_visible`, which raises
:class:`~stock_rl.sentiment.score.LookAheadError` on the first one dated
after the bar. Readings for symbols this node does not reach are not
consulted and are therefore not in scope: they cannot reach a number this
call produces.

**The drawn slice walks both directions, and the two are not
interchangeable.** :func:`readings_for_node` walks *downstream*, because
that is the direction a shock travels and therefore the direction news
arrives from. :func:`node_slice` adds an *upstream* walk of the focused
node's own causes, because a stock node's news is its own readings at zero
hops and a slice built from the news walk alone would be one box with a
company's name on it -- true, useless, and missing the "company, its
commodity and macro inputs, and the path between them" a graph view exists
to draw. Both directions are emitted with the same orientation on screen
(cause to affected, the reverse of the stored edge) and the same hop
numbering, so a reader is never shown two incompatible pictures of the
same file. An upstream node that happens to be a **stock** gets the
``'peer'`` role rather than an input role: the edge is a real claim in the
file, but nothing in the graph established that one listed company
supplies another, and a drawing that implied it would be a fabricated
supply chain.

**The edges are unfitted.** :mod:`stock_rl.graph.edges` says so itself: the
seed is common sector reasoning, no beta was estimated, no edge carries a
date. So this module reports :data:`UNFITTED_NOTE` in every payload, and
:func:`edge_meaning` words each edge as a claim rather than a
sensitivity. A UI that draws these arrows without that qualification is
drawing a measurement that does not exist.

**The registry is an argument.** ``readings_for_node`` and
:func:`node_slice` take a :class:`~stock_rl.sentiment.entity_link.SymbolLinker`
so a deployment whose symbol file differs from the starter registry can
hand its own, rather than having the graph and the linker disagree in
silence. The registry only shapes the *provenance string*: whether a
symbol resolves, is ambiguous, or is absent from the table never changes
whether a reading is attached, because the reading arrived already filed
under the symbol.

PONYTAIL: breadth-first, shortest path per symbol, one aggregate per
symbol, no inference from text anywhere. Ceiling: a symbol reached by two
routes reports only the shortest, so the second route is invisible; hops
are not decayed, so a three-hop exposure is presented exactly like a
one-hop one; and nothing here measures whether a commodity move ever moved
the stock. Upgrade path: carry a list of paths per symbol instead of one
:class:`PathRoute`, and add an edge field carrying a measured beta
rather than widening this module's inference.
'''

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from stock_rl.graph.edges import DependencyGraph, EdgeKind
from stock_rl.graph.traversal import default_max_depth, max_traversal_depth
from stock_rl.sentiment.entity_link import (
  LinkStatus,
  SymbolLinker,
  default_linker,
)
from stock_rl.sentiment.score import (
  SentimentAggregate,
  SentimentReading,
  aggregate,
  default_min_sources,
  require_visible,
)

__all__ = [
  'GraphSlice',
  'NodeNews',
  'NodeNewsStatus',
  'PathRoute',
  'ReadingRef',
  'SliceEdge',
  'SliceNode',
  'UNFITTED_NOTE',
  'edge_meaning',
  'node_slice',
  'not_advice',
  'readings_for_node',
]

#: Attached to every slice this module produces. The wording is lifted from
#: :mod:`stock_rl.graph.edges`, which is where the caveat is made in the
#: source, so the JSON and the module that produced the graph cannot
#: disagree about what the arrows mean.
UNFITTED_NOTE = (
  'Unfitted sector reasoning, not measured sensitivities. No beta was '
  'estimated, no edge carries a date, and the weight field is declarative '
  'metadata that no traversal consumes. Treat every edge as a reviewable '
  'claim about direction only.'
)

#: Same sentence the dashboard already prints above every other panel. It
#: is repeated here rather than imported so this module stays importable
#: without the API package, which is owned elsewhere and in flux.
not_advice = (
  'Research output from a backtested model. Not financial advice, not a '
  'recommendation, and not a solicitation to buy or sell anything.'
)

#: Short edge label drawn on the canvas beside each arrow. Long enough to
#: name a direction, short enough to fit between two node boxes.
_SHORT_MEANING: dict[EdgeKind, str] = {
  EdgeKind.DEPENDS_ON: 'depends on',
  EdgeKind.AFFECTED_BY: 'affected by',
}

#: What the linker says when a stock node's name is not in the registry.
#: Not an error: the reading arrives filed under the symbol already, so it
#: is reachable by symbol while being unreachable from any headline text.
#: Formatted with the symbol, because "no registry row" without saying
#: which symbol is a sentence nobody can act on.
_UNLINKED = (
  '{symbol} has no registry row, so no free text resolves to it; a '
  'reading can still be filed under it by symbol'
)


class NodeNewsStatus(StrEnum):
  '''Why a node has the news it has.

  The distinction that matters is between the last two members. ``no_news``
  is an answer: the graph named the companies that would be exposed and
  none of them carried a public reading at the bar. An **error** is not a
  member of this enum -- it is an exception, raised by
  :func:`~stock_rl.sentiment.score.require_visible` for a look-ahead leak
  and by this module for a malformed request. A panel that cannot tell
  those apart renders "no data" and "broken" identically.

  Attributes:
    HAS_NEWS: At least one reading was attached, with its provenance.
    NO_NEWS: Stock nodes were reached and none carried a public reading.
      ``checked_symbols`` names every symbol that was looked up, so the
      absence is attributable rather than merely reported.
    NO_DEPENDENTS: No stock node was reached within ``max_depth``. The
      package holds no claim connecting this node to any listed company,
      so there is nothing to look up. Empty, and it stays empty.
  '''

  HAS_NEWS = 'has_news'
  NO_NEWS = 'no_news'
  NO_DEPENDENTS = 'no_dependents'


@dataclass(frozen=True, slots=True)
class PathRoute:
  '''How one symbol was reached from the node that was asked about.

  Attributes:
    symbol: Bare exchange symbol of the stock node reached, e.g.
      ``'ONGC'``.
    hops: Number of stored edges walked. Zero when the node asked about
      *is* the stock, which is the only case needing no graph at all.
    path: Node keys from the node asked about to ``stock:SYMBOL``,
      inclusive at both ends. Length is ``hops + 1``.
    edges: ``EdgeKind`` of each stored edge walked, in the order walked,
      so a reader can see whether the exposure was declared as a
      dependency or as an event effect.
    linkage: The entity linker's own reason string for ``symbol``. Carried
      rather than recomputed so the UI can show what the registry said
      instead of asserting that resolution "worked".
  '''

  symbol: str
  hops: int
  path: tuple[str, ...]
  edges: tuple[EdgeKind, ...]
  linkage: str

  @property
  def direct(self) -> bool:
    '''Return True when no graph hop was needed to name this symbol.'''
    return self.hops == 0

  def describe(self) -> str:
    '''Return a sentence naming the symbol, the hops and the route.

    Returns:
      Plain-language provenance. For a stock node it says the headline was
      filed under that symbol and no hop was used. Otherwise it prints
      the whole key chain and the edge kind of every hop, because "ONGC
      reads the crude page" is only auditable if the reader is shown the
      route and can reject the particular edge they disagree with.
    '''
    if self.direct:
      return (
        f'{self.symbol} is the node itself, so the reading was filed under '
        f'that exchange symbol upstream and no graph hop was used. The '
        f'registry says: {self.linkage}'
      )
    chain = ' -> '.join(self.path)
    kinds = ', '.join(edge.value for edge in self.edges)
    return (
      f'{self.symbol} was reached in {self.hops} hop(s) along {chain}. Each '
      f'hop is a stored edge of kind [{kinds}] pointing from the node that '
      f'depends to the node that causes it, so the shock travels this way '
      f'and not the other. The registry says: {self.linkage}'
    )

  def to_dict(self) -> dict[str, object]:
    '''Serialise to the JSON object shape the dashboard view consumes.

    Returns:
      Mapping with ``symbol``, ``hops``, ``path``, ``edges``, ``linkage``,
      ``direct`` and ``describe``.
    '''
    return {
      'symbol': self.symbol,
      'hops': self.hops,
      'path': list(self.path),
      'edges': [edge.value for edge in self.edges],
      'linkage': self.linkage,
      'direct': self.direct,
      'describe': self.describe(),
    }


@dataclass(frozen=True, slots=True)
class ReadingRef:
  '''One reading attached to a node, with the reason it is attached.

  Attributes:
    reading: The reading as filed, untouched.
    route: Which symbol it was filed under and how that symbol was
      reached from the node. Never None, including for a stock node,
      where it is the zero-hop route.
    aggregate: :func:`~stock_rl.sentiment.score.aggregate` for the symbol
      at ``decision_bar``. Carried per symbol rather than per node: the
      source minimum, the cross-source spread and the anti-convergence
      guard are all properties of one symbol's readings, and averaging
      across the dependents of a commodity would silently invent a
      sentiment for the commodity itself.
    reason: The sentence to show a reader. States the symbol, the hop
      count and the route, and says plainly that the link is a structural
      proxy from an unfitted graph rather than a statement about the
      commodity.
  '''

  reading: SentimentReading
  route: PathRoute
  aggregate: SentimentAggregate
  reason: str

  @property
  def symbol(self) -> str:
    '''Return the bare exchange symbol this reading was filed under.'''
    return self.reading.symbol

  def to_dict(self) -> dict[str, object]:
    '''Serialise to the JSON object shape the dashboard view consumes.

    Returns:
      Flat mapping. The reading's own fields are spread at the top level
      because a news row is mostly reading, and the provenance fields are
      prefixed with ``via_`` so a reader cannot mistake one for the other.
    '''
    return {
      'symbol': self.reading.symbol,
      'score': self.reading.score,
      'intensity': self.reading.intensity,
      'event_type': self.reading.event_type,
      'source': self.reading.source,
      'source_count': self.reading.source_count,
      'available_from': self.reading.available_from.isoformat(),
      'via_hops': self.route.hops,
      'via_path': list(self.route.path),
      'via_edges': [edge.value for edge in self.route.edges],
      'via_linkage': self.route.linkage,
      'usable': self.aggregate.usable,
      'disagreement': self.aggregate.disagreement,
      'converged': self.aggregate.converged,
      'rejected': list(self.aggregate.rejected),
      'reason': self.reason,
    }


@dataclass(frozen=True, slots=True)
class NodeNews:
  '''Everything the view needs about one node's news.

  Attributes:
    node: Node key that was asked about.
    label: Its display label, e.g. ``'Crude Oil'``.
    kind: Its :class:`~stock_rl.graph.nodes.NodeKind` value as a string.
    note: The node's own caveat, when it carries one.
    status: One of :class:`NodeNewsStatus`.
    max_depth: Hop bound that was used, so a caller can see the answer is
      bounded rather than total.
    checked_symbols: Every stock symbol the traversal reached, sorted.
      Populated even when nothing was attached, which is the difference
      between "no news for ONGC, RELIANCE, ASIANPAINT" and a panel with
      no explanation.
    routes: One :class:`PathRoute` per reached symbol, ordered by hops
      then symbol.
    entries: The attached readings, ordered by hops, then symbol, then
      publication time, then source. Deterministic, so a test can pin it.
  '''

  node: str
  label: str
  kind: str
  note: str
  status: NodeNewsStatus
  max_depth: int
  checked_symbols: tuple[str, ...]
  routes: tuple[PathRoute, ...]
  entries: tuple[ReadingRef, ...]

  @property
  def news_count(self) -> int:
    '''Return how many readings were attached.'''
    return len(self.entries)

  def message(self) -> str:
    '''Return the honest one-line summary for an empty or full result.

    Returns:
      A sentence naming the situation. The three cases are worded
      differently on purpose, because a reader who cannot tell "no news"
      from "no route to any company" from "this failed" will draw a
      conclusion from whichever one they assumed.
    '''
    if self.status is NodeNewsStatus.HAS_NEWS:
      symbols = ', '.join(sorted({entry.symbol for entry in self.entries}))
      return (
        f'{self.news_count} reading(s) attached across {symbols}, each via '
        f'the dependency path printed beside it. These edges are unfitted '
        f'sector reasoning, not measured sensitivities.'
      )
    if self.status is NodeNewsStatus.NO_DEPENDENTS:
      return (
        f'The graph records no listed symbol downstream of {self.node} '
        f'within {self.max_depth} hops, so there is nothing to look up. No '
        f'news is attached to this node and none was invented.'
      )
    listed = ', '.join(self.checked_symbols)
    return (
      f'No reading public at the decision bar for any of the '
      f'{len(self.checked_symbols)} symbol(s) the graph says are exposed to '
      f'this node ({listed}). That is an absence of news, not a failure to '
      f'load: the lookup ran and found nothing.'
    )

  def to_dict(self) -> dict[str, object]:
    '''Serialise to the JSON object shape the dashboard view consumes.

    Returns:
      Mapping with the node, the status, the honest message, the routes
      and the readings, plus the unfitted caveat and the decision bar.
    '''
    return {
      'node': self.node,
      'label': self.label,
      'kind': self.kind,
      'note': self.note,
      'status': self.status.value,
      'message': self.message(),
      'news_count': self.news_count,
      'max_depth': self.max_depth,
      'checked_symbols': list(self.checked_symbols),
      'routes': [route.to_dict() for route in self.routes],
      'readings': [entry.to_dict() for entry in self.entries],
    }


@dataclass(frozen=True, slots=True)
class SliceNode:
  '''One node of the drawn subgraph.

  Attributes:
    key: Node key.
    label: Display label.
    kind: Node kind as a string.
    note: The node's own caveat, often empty.
    hops: Distance from the node the view is focused on. Zero for focus.
    role: What the node is in this drawing, which is not the same as its
      kind. ``'focus'`` is the node asked about; ``'stock'`` is a listed
      company the news walk reached, so its news is on screen; ``'peer'``
      is a listed company that points at the focus in the graph, which is
      a claim in the file but not an input, so it is drawn and labelled as
      what it is; ``'path'`` is an intermediate node on a route, such as a
      sector between a stock and a commodity.
  '''

  key: str
  label: str
  kind: str
  note: str
  hops: int
  role: str

  def to_dict(self) -> dict[str, object]:
    '''Serialise to the JSON object shape the dashboard view consumes.

    Returns:
      Mapping with ``key``, ``label``, ``kind``, ``note``, ``hops`` and
      ``role``.
    '''
    return {
      'key': self.key,
      'label': self.label,
      'kind': self.kind,
      'note': self.note,
      'hops': self.hops,
      'role': self.role,
    }


@dataclass(frozen=True, slots=True)
class SliceEdge:
  '''One edge of the drawn subgraph, in the direction it propagates.

  The stored edges point affected-to-cause, so a shock travels backwards
  along them. This is drawn the way a shock travels, and the reversal is
  stated in :attr:`meaning` rather than left for the reader to notice.

  Attributes:
    cause: Node key at the tail, where the shock starts.
    affected: Node key at the head, which the shock reaches.
    kind: The stored :class:`~stock_rl.graph.edges.EdgeKind`.
    hops: Distance of ``affected`` from the focused node.
    meaning: A sentence saying which node is claimed to depend on or be
      affected by which. Present on every edge so no arrow in the UI can
      be unlabelled.
    measured: Always False. Present as a field rather than only as prose
      so a consumer cannot drop the caveat by forgetting to print it.
  '''

  cause: str
  affected: str
  kind: EdgeKind
  hops: int
  meaning: str
  measured: bool = False

  @property
  def short(self) -> str:
    '''Return the two- or three-word label drawn beside the arrow.'''
    return _SHORT_MEANING[self.kind]

  def to_dict(self) -> dict[str, object]:
    '''Serialise to the JSON object shape the dashboard view consumes.

    Returns:
      Mapping with ``cause``, ``affected``, ``kind``, ``hops``, ``meaning``,
      ``short`` and ``measured``.
    '''
    return {
      'cause': self.cause,
      'affected': self.affected,
      'kind': self.kind.value,
      'hops': self.hops,
      'meaning': self.meaning,
      'short': self.short,
      'measured': self.measured,
    }


@dataclass(frozen=True, slots=True)
class GraphSlice:
  '''A node, the routes out of it, and the news on those routes.

  Attributes:
    news: The join result for the focused node.
    nodes: The focused node plus every node on a route, ordered by hops
      then key. A stock with no reading is still in here: dropping it
      would make the reader think the graph does not connect it.
    edges: Every stored edge on a route, drawn cause-to-affected, ordered
      by hops then cause then affected.
  '''

  news: NodeNews
  nodes: tuple[SliceNode, ...]
  edges: tuple[SliceEdge, ...]

  def to_dict(self) -> dict[str, object]:
    '''Serialise the whole slice for the dashboard view.

    Returns:
      Mapping carrying the node join, the subgraph and the two sentences
      that must accompany any drawing of it. ``not_advice`` is included
      because the view renders these numbers on the same page as Sharpe
      ratios and the disclaimer already attached there is exactly this
      text.
    '''
    payload = self.news.to_dict()
    payload['nodes'] = [node.to_dict() for node in self.nodes]
    payload['edges'] = [edge.to_dict() for edge in self.edges]
    payload['unfitted'] = UNFITTED_NOTE
    payload['not_advice'] = not_advice
    return payload


def edge_meaning(kind: EdgeKind, cause: str, affected: str) -> str:
  '''Return what one edge claims, worded as a claim.

  Args:
    kind: The stored edge kind.
    cause: Node key the shock starts at.
    affected: Node key the shock reaches.

  Returns:
    A sentence naming both keys and the direction. "depends on" alone is
    ambiguous on a canvas where the arrow points the other way from the
    stored edge, so the sentence spells out which node is claimed to
    depend on or be affected by which.
  '''
  if kind is EdgeKind.DEPENDS_ON:
    claim = f'{affected} is recorded as depending on {cause}'
  else:
    claim = f'{affected} is recorded as affected by {cause}'
  return (
    f'{claim}. Unfitted: the arrow is drawn from the cause to the affected '
    f'node, which is the reverse of the stored edge, and nothing here was '
    f'estimated from returns.'
  )


def readings_for_node(
  graph: DependencyGraph,
  node_key: str,
  readings: list[SentimentReading],
  decision_bar: datetime,
  max_depth: int = default_max_depth,
  min_sources: int = default_min_sources,
  linker: SymbolLinker | None = None,
) -> NodeNews:
  '''Return the sentiment readings relevant to a node, with provenance.

  The rule, implemented literally: walk the graph in the direction
  information travels, collect the stock nodes reached within
  ``max_depth``, and attach every public reading whose ``symbol`` is
  exactly one of them. The headline is never inspected. See the module
  docstring for why that is the only defensible bridge available.

  Args:
    graph: Graph to walk.
    node_key: Node being asked about, e.g. ``'commodity:crude'``. A stock
      key returns that symbol's own readings and nothing else.
    readings: All readings in hand, of any symbol. Only those filed under
      a symbol this node reaches are consulted.
    decision_bar: The bar the reader is standing on, timezone-aware.
      **Required, with no default.** A permissive default has been
      reintroduced as a look-ahead bug twice in this repository, so the
      signature forces the caller to name the bar.
    max_depth: Hop bound, in ``[1, max_traversal_depth]``.
    min_sources: Distinct-source minimum for the per-symbol aggregate.
      Passed through to
      :func:`~stock_rl.sentiment.score.aggregate`; it does not decide
      whether a reading is *attached*.
    linker: Registry to ask about each reached symbol, defaulting to
      :data:`stock_rl.sentiment.entity_link.default_linker`. It is an
      argument because the registry is a *table*, and a table is data:
      a deployment whose symbol file differs from the starter registry
      should be able to hand its own rather than have the graph and the
      linker disagree in silence.

  Returns:
    A :class:`NodeNews`. ``entries`` is empty and ``status`` is
    ``NO_DEPENDENTS`` or ``NO_NEWS`` when nothing was attached, never a
    fallback set.

  Raises:
    ValueError: If ``node_key`` is not in the graph, or ``max_depth`` is
      out of range, or ``decision_bar`` is naive.
    stock_rl.sentiment.score.LookAheadError: If a candidate reading became
      public after ``decision_bar``. Subclasses ``ValueError``, so a
      caller catching the ordinary case still catches the leak.
  '''
  _check_request(graph, node_key, max_depth)
  routes = _routes(graph, node_key, max_depth, linker)
  checked = tuple(sorted(route.symbol for route in routes))
  candidates = [reading for reading in readings if reading.symbol in checked]
  # The strict filter, on exactly the readings that could reach a number
  # this call returns. Readings about symbols this node does not reach
  # cannot enter the answer, so raising on them would be noise that
  # trains the caller to catch and ignore the real thing.
  require_visible(candidates, decision_bar)
  entries: list[ReadingRef] = []
  for route in routes:
    mine = [reading for reading in candidates
            if reading.symbol == route.symbol]
    if not mine:
      continue
    summary = aggregate(candidates, route.symbol, decision_bar,
                        min_sources=min_sources)
    for reading in sorted(mine, key=_reading_order):
      entries.append(ReadingRef(
        reading=reading,
        route=route,
        aggregate=summary,
        reason=_reason(route),
      ))
  if entries:
    status = NodeNewsStatus.HAS_NEWS
  elif checked:
    status = NodeNewsStatus.NO_NEWS
  else:
    status = NodeNewsStatus.NO_DEPENDENTS
  node = graph.node(node_key)
  return NodeNews(
    node=node.key,
    label=node.label,
    kind=node.kind.value,
    note=node.note,
    status=status,
    max_depth=max_depth,
    checked_symbols=checked,
    routes=routes,
    entries=tuple(entries),
  )


def node_slice(
  graph: DependencyGraph,
  node_key: str,
  readings: list[SentimentReading],
  decision_bar: datetime,
  max_depth: int = default_max_depth,
  min_sources: int = default_min_sources,
  linker: SymbolLinker | None = None,
) -> GraphSlice:
  '''Return the drawable subgraph for a node together with its news.

  Args:
    graph: Graph to walk.
    node_key: Node the view is focused on.
    readings: All readings in hand.
    decision_bar: Bar the reader is standing on, required and aware.
    max_depth: Hop bound.
    min_sources: Distinct-source minimum for the per-symbol aggregate.
    linker: Registry to ask about each reached symbol, forwarded to
      :func:`readings_for_node`.

  Returns:
    A :class:`GraphSlice` whose ``nodes`` and ``edges`` cover three things:
    the focused node, the node's own inputs walked upstream, and every
    route a returned reading used. The upstream half is what makes a
    stock's slice a dependency graph rather than a one-box list -- the
    company, its commodity and macro inputs, and the path between them.
  '''
  news = readings_for_node(graph, node_key, readings, decision_bar,
                           max_depth, min_sources, linker)
  by_pair = _edge_index(graph)
  # Stored edges run affected -> cause. The two walks produce chains in
  # opposite senses: a news route is already cause-to-affected
  # ('crude -> ONGC'), while an upstream chain follows causes_of and so is
  # affected-to-cause ('RELIANCE -> crude'). Both are focus-first and index
  # i is the node i hops from the focus; only the stored pair read off them
  # differs, which is what ``stored_pair`` settles in one place instead of at
  # two call sites.
  chains = [(list(route.path), True) for route in news.routes]
  chains.extend((chain, False)
                for chain in _upstream(graph, news.node, max_depth))
  hops: dict[str, int] = {news.node: 0}
  for chain, _ in chains:
    for index, key in enumerate(chain):
      hops[key] = min(hops.get(key, index), index)
  # Which stocks the news walk actually reached, as opposed to which
  # stocks the graph happens to contain. The difference decides a stock's
  # role: a reached stock is exposed to the focus and its news is on
  # screen; an upstream stock is neither, and calling it an input would
  # assert a supply-chain relationship between two listed companies.
  reached = {route.path[-1] for route in news.routes}
  ordered: list[SliceNode] = []
  for key in sorted(hops, key=lambda item: (hops[item], item)):
    node = graph.node(key)
    if key == news.node:
      role = 'focus'
    elif key in reached:
      role = 'stock'
    elif node.is_stock:
      role = 'peer'
    else:
      role = 'path'
    ordered.append(SliceNode(
      key=key,
      label=node.label,
      kind=node.kind.value,
      note=node.note,
      hops=hops[key],
      role=role,
    ))
  # Every edge is emitted cause-to-affected, the direction a shock travels,
  # which is the reverse of the stored pair. Both halves of the slice
  # therefore draw the same way once the stored pair is read correctly.
  seen_edges: set[tuple[str, str]] = set()
  drawn: list[SliceEdge] = []
  for chain, cause_first in chains:
    for index in range(1, len(chain)):
      stored = _stored_pair(chain, index, cause_first)
      pair = (stored[1], stored[0])
      if pair in seen_edges:
        continue
      seen_edges.add(pair)
      kind = by_pair[stored]
      drawn.append(SliceEdge(
        cause=pair[0],
        affected=pair[1],
        kind=kind,
        hops=index,
        meaning=edge_meaning(kind, pair[0], pair[1]),
      ))
  drawn.sort(key=lambda edge: (edge.hops, edge.cause, edge.affected))
  return GraphSlice(news=news, nodes=tuple(ordered), edges=tuple(drawn))


def _stored_pair(
  chain: list[str],
  index: int,
  cause_first: bool,
) -> tuple[str, str]:
  '''Return the ``(source, target)`` of the stored edge at ``chain[index]``.

  Stored edges run **affected -> cause** (see :mod:`stock_rl.graph.edges`),
  so a chain that lists the cause at the lower index has to be read
  backwards to find the edge object, while a chain that lists the affected
  node there is already in stored order. Getting this backwards is the one
  mistake in this module that would invert the whole drawing while still
  drawing every node, which is why both cases are spelled out here rather
  than left to a conditional at the call site.

  Args:
    chain: Focus-first key chain.
    index: Index in ``chain`` of the node at the far end of the hop.
    cause_first: True when the chain lists the cause at the lower index,
      as a news route does (``'crude -> ONGC'``); False when it lists the
      affected node there, as an upstream walk does (``'ONGC -> crude'``).

  Returns:
    Tuple in stored order, so it indexes the graph's edge table directly.
  '''
  near, far = chain[index - 1], chain[index]
  return (far, near) if cause_first else (near, far)


def _upstream(
  graph: DependencyGraph,
  node_key: str,
  max_depth: int,
) -> list[list[str]]:
  '''Return the key chains of the things ``node_key`` depends on.

  The mirror of :func:`_routes`. A stock node's own news is its own
  readings at zero hops, so without this walk a stock's slice would be a
  single box labelled with the company's name: technically correct,
  visually useless, and missing the thing the view is for. Walking
  ``causes_of`` gives the commodity and macro inputs and the sector
  between them, which is the "company, its inputs, and the path"
  structure the brief asks a graph view to draw.

  Args:
    graph: Graph to walk.
    node_key: Node to start from.
    max_depth: Hop bound, already validated.

  Returns:
    One chain per cause reachable, each focus-first, ordered by length then
    key. A visited set keeps a cyclic file from looping.

    Stock causes are included in the chains, because the stored edge
    between two listed companies is a real claim in the file and hiding
    it would make the drawing smaller than the data without saying so.
    What is withheld is the *input* claim: :func:`node_slice` gives such a
    node the ``'peer'`` role rather than the ``'path'`` one an input would
    carry, so a reader is never told that one listed company supplies
    another.
  '''
  chains: list[list[str]] = []
  visited = {node_key}
  frontier: list[list[str]] = [[node_key]]
  while frontier:
    grown: list[list[str]] = []
    for chain in frontier:
      if len(chain) > max_depth:
        continue
      for cause in graph.causes_of(chain[-1]):
        if cause in visited:
          continue
        visited.add(cause)
        extended = [*chain, cause]
        chains.append(extended)
        grown.append(extended)
    frontier = grown
  return sorted(chains, key=lambda chain: (len(chain), chain))


def _check_request(
  graph: DependencyGraph,
  node_key: str,
  max_depth: int,
) -> None:
  '''Validate a join request before anything is walked.

  Args:
    graph: Graph the node must belong to.
    node_key: Node key asked about.
    max_depth: Requested hop bound.

  Raises:
    ValueError: If the node is unknown, or the depth is outside
      ``[1, max_traversal_depth]``. Both refusals are raised before the
      traversal so a typo cannot be reported as "no news", which is the
      failure this module exists to avoid.
  '''
  if not graph.has(node_key):
    raise ValueError(f'unknown node {node_key!r}')
  if max_depth < 1 or max_depth > max_traversal_depth:
    raise ValueError(
      f'max_depth must be in [1, {max_traversal_depth}], got {max_depth}; a '
      f'deep request on a graph that may hold a cycle has no meaning')


def _routes(
  graph: DependencyGraph,
  node_key: str,
  max_depth: int,
  linker: SymbolLinker | None,
) -> tuple[PathRoute, ...]:
  '''Return one shortest route per stock node reachable from ``node_key``.

  Breadth-first, following ``exposed_to`` -- the direction a shock
  travels, which is against the stored edge direction. A visited set keeps
  a cyclic file from looping, and a predecessor map is kept alongside it so
  the path can be reconstructed: :func:`affected_stocks` returns a hop
  count and cannot say how a symbol was reached, and "two hops" without
  the two nodes is not provenance.

  Args:
    graph: Graph to walk.
    node_key: Node the walk starts at.
    max_depth: Hop bound, already validated.
    linker: Registry to ask about each reached symbol, or None for the
      starter one.

  Returns:
    Routes ordered by hops then symbol. A stock node asked about directly
    yields its own zero-hop route, which is the one case that needs no
    graph at all.
  '''
  # Total over exposed_to: the adjacency index is written by add_edge from
  # the same edge objects, so a pair reported by exposed_to always has an
  # edge here. That is what lets the hop carry its kind without a fallback
  # branch that would only ever be dead.
  by_pair = _edge_index(graph)
  depth = {node_key: 0}
  came_from: dict[str, tuple[str, EdgeKind]] = {}
  frontier = deque([node_key])
  while frontier:
    current = frontier.popleft()
    if depth[current] >= max_depth:
      continue
    for affected in graph.exposed_to(current):
      if affected in depth:
        continue
      depth[affected] = depth[current] + 1
      came_from[affected] = (current, by_pair[(affected, current)])
      frontier.append(affected)
  found: list[PathRoute] = []
  for key, hops in depth.items():
    node = graph.node(key)
    if not node.is_stock:
      continue
    found.append(PathRoute(
      symbol=node.name,
      hops=hops,
      path=_chain(key, came_from),
      edges=_kinds(key, came_from),
      linkage=_linkage(node.name, linker),
    ))
  return tuple(sorted(found, key=lambda route: (route.hops, route.symbol)))


def _chain(
  key: str,
  came_from: dict[str, tuple[str, EdgeKind]],
) -> tuple[str, ...]:
  '''Reconstruct the node keys from the walk's start to ``key``.

  Args:
    key: Node reached.
    came_from: Predecessor map produced by the walk.

  Returns:
    Keys with the starting node first and ``key`` last. The start node is
    not in ``came_from``, so the walk back stops there.
  '''
  path = [key]
  while path[-1] in came_from:
    path.append(came_from[path[-1]][0])
  path.reverse()
  return tuple(path)


def _kinds(
  key: str,
  came_from: dict[str, tuple[str, EdgeKind]],
) -> tuple[EdgeKind, ...]:
  '''Return the edge kind of every hop between the start node and ``key``.

  Args:
    key: Node reached.
    came_from: Predecessor map produced by the walk.

  Returns:
    Kinds in the order walked. Empty for the starting node.
  '''
  kinds: list[EdgeKind] = []
  cursor = key
  while cursor in came_from:
    previous, kind = came_from[cursor]
    kinds.append(kind)
    cursor = previous
  kinds.reverse()
  return tuple(kinds)


def _linkage(symbol: str, linker: SymbolLinker | None) -> str:
  '''Return the entity linker's own reason for a stock symbol.

  Args:
    symbol: Bare exchange symbol.
    linker: Registry to ask, or None for the starter one. A caller with
      its own symbol file passes it here rather than patching this module,
      which is the whole reason :func:`readings_for_node` takes a linker.

  Returns:
    The registry's reason string when the symbol is a row, a sentence
    naming the collision when a registry form is claimed by two symbols,
    and a sentence saying it is absent otherwise. The registry is asked
    rather than assumed, so that a graph node naming a company the linker
    has never heard of is visible as such instead of being silently
    treated as resolvable.

    Note that none of these three outcomes changes whether a reading is
    attached. The reading arrived filed under the symbol, so the symbol is
    reachable by symbol whatever the registry says; this string only
    records how a reader could have got there from text.
  '''
  found = (linker or default_linker).link(symbol)
  if found.status is LinkStatus.RESOLVED:
    return f'{found.matched_by}: {found.reason}'
  if found.status is LinkStatus.AMBIGUOUS:
    others = ', '.join(found.candidates)
    return (
      f'{symbol} is ambiguous: that registry form is claimed by '
      f'{len(found.candidates)} symbols ({others}), so none was chosen')
  return _UNLINKED.format(symbol=symbol)


def _reason(route: PathRoute) -> str:
  '''Return the sentence shown beside one attached reading.

  Args:
    route: The route that produced the reading.

  Returns:
    Provenance in the reader's terms: the symbol, the hop count, the
    route, and the standing caveat that a commodity node has no news of
    its own -- this is the news of the companies that consume it.
  '''
  if route.direct:
    lead = (
      f'Attached directly: the reading is filed under {route.symbol} and '
      f'this node is {route.symbol}.')
  else:
    chain = ' -> '.join(route.path)
    lead = (
      f'Attached through the dependency graph, not by matching text: '
      f'{route.symbol} is {route.hops} hop(s) downstream of this node along '
      f'{chain}, and the registry resolved that symbol from the exchange '
      f'symbol itself.')
  return (
    f'{lead} This node has no news of its own -- entity_link resolves '
    f'listed symbols, not commodities -- so what is shown is the news of '
    f'the companies the package claims consume it, over an edge that was '
    f'never fitted to returns.')


def _reading_order(reading: SentimentReading) -> tuple[str, str, str]:
  '''Return the sort key that makes an attached reading order stable.

  Args:
    reading: The reading being ordered.

  Returns:
    Tuple of publication time, then source, then event type. Score is
    deliberately absent: the order is a publication order, and sorting by
    how bearish a headline reads would be a way of deciding what a reader
    sees before they do.
  '''
  return (reading.available_from.isoformat(), reading.source,
          reading.event_type)


def _edge_index(
  graph: DependencyGraph,
) -> dict[tuple[str, str], EdgeKind]:
  '''Return the stored edge kind for every ``(affected, cause)`` pair.

  Args:
    graph: Graph to index.

  Returns:
    Mapping of ``(edge.source, edge.target)`` to the edge kind. Built once
    per join because the routes need the kind of every hop and a linear
    rescan of the edge list per hop would be quadratic on a file near the
    documented 2000-edge budget. :meth:`DependencyGraph.add_edge` is what
    writes the adjacency index, so this mapping covers everything
    ``exposed_to`` can report.
  '''
  return {(edge.source, edge.target): edge.kind for edge in graph.edges}
