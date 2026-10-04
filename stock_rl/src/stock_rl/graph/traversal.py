#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Bounded traversal of the static dependency graph.

The design doc's one graph-shaped sentence is
``Crude +5% -> ONGC, RELIANCE, ASIANPAINT``. This module is that, and
the whole reason it is 150 lines rather than a Cypher query is that the
operation is a breadth-first walk over an adjacency dict.

**Both depth and cycles are handled, because both will happen.** A
hand-maintained dependency graph acquires a cycle the first time someone
adds ``sector:fmcg depends_on macro:gst`` and ``macro:gst depends_on
sector:fmcg`` because it seemed reasonable. A naive recursive traversal
then hangs, and it hangs *inside a backtest*, so the symptom is a run
that never finishes rather than a stack trace. Every traversal here
carries a visited set, and depth is bounded twice: by the caller's
``max_depth`` and by :data:`max_traversal_depth`, which is a hard cap so
that a request for "everything" fails loudly instead of silently
returning the transitive closure of a cyclic graph.

**Enumerating loops is gated on the linear check, because enumerating
costs ``fanout ** max_length`` however few loops there are.** On an
honestly acyclic 333-node, 1977-edge graph -- well inside the edge budget
this package documents -- listing every simple path up to ``max_length``
takes 616 million path extensions to report *nothing*, while
:func:`has_cycle` settles the same question in about a millisecond. A
file with no loop has none to name, so :func:`cycles` answers from the
linear pass and walks nothing; a file that does have one is walked
against :data:`max_cycle_expansions`, which raises and names the graph
rather than grinding. The same lesson as the paragraph above, applied to
the other expensive operation: a bound nothing raises on is not a bound.

**A bounded traversal is a statement about the world, not a
performance hack.** At depth 1 from crude you get the three names the
design doc names. At depth 3 from ``macro:global_liquidity`` you get the
banks and the IT names, which reach it only through ``macro:fii_net``.
Both are correct answers to different questions, which is why the depth
is an argument and never a default buried in the call site.

PONYTAIL: breadth-first, hop counts, no weights, no path enumeration.
Ceiling: the returned depth is the *minimum* hop count, so a node reached
by two routes reports the shorter one and the other route is invisible.
Upgrade path: return a list of paths per symbol instead of an int; the
signature is a dict for exactly that reason.
'''

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from stock_rl.graph.edges import DependencyGraph

__all__ = [
  'affected_stocks',
  'cycles',
  'default_max_depth',
  'dependencies',
  'dependents',
  'has_cycle',
  'max_cycle_expansions',
  'max_traversal_depth',
]

#: Default hop bound. Three, because three is the depth at which a global
#: driver actually reaches a bank in the seed graph:
#: ``macro:global_liquidity -> macro:fii_net -> sector:banking ->
#: stock:HDFCBANK``. One hop returns only directly named exposures and
#: misses the sector channel; two reaches the sector but none of its
#: constituents. It is a default, not a rule: pass the depth that answers
#: the question being asked.
default_max_depth = 3

#: Hard cap on any traversal. A request beyond this raises instead of
#: running, because in a graph with cycles "depth 100" has no meaning
#: and quietly returning a closure is how a wrong universe becomes a
#: feature.
max_traversal_depth = 6

#: Default ceiling on path extensions during cycle enumeration. An
#: extension is one node pushed onto a partial path, which is the unit of
#: work the walk actually performs, and one costs about 0.6us on the
#: hardware these figures were measured on. So the number is chosen twice
#: over.
#:
#: Capacity first. The seed graph's whole enumeration is 56 extensions, and
#: a 40-node, 115-edge file carrying two short loops spends 247k of them at
#: the default ``max_cycles``. A million is ~18000x the seed and ~4x that
#: file, which is the shape a reviewer is actually holding.
#:
#: Cost second. A million is ~0.6s: two to three orders of magnitude above
#: :func:`has_cycle` on the largest graph in budget, and nowhere near the
#: 342s the same 616-million-extension enumeration took. An audit that
#: spends 0.6s finishing is slow and obvious; one that never finishes is a
#: run with no stack trace. It does not make the 2000-edge top of the
#: reviewed range comfortable -- that file spends 2.1M extensions naming
#: one loop and then keeps walking for the other nine -- which is the
#: point: past this ceiling the caller's judgement is required, so it is
#: passed as ``max_expansions`` rather than assumed.
max_cycle_expansions = 1_000_000


@dataclass(slots=True)
class _Meter:
  '''Counts path extensions against a ceiling.

  Counting extensions is the only honest bound available here:
  :func:`has_cycle` says whether a loop exists, not how many paths lead
  nowhere, and the number of those paths is invisible in the arguments.
  An extension is the thing the walk does, so it is what gets counted.

  Attributes:
    left: Extensions still permitted. Never negative.
    ceiling: The ceiling this meter was built with, so the error names
      the number the caller passed rather than a running total.
  '''

  left: int
  ceiling: int

  def spend(self) -> bool:
    '''Consume one extension.

    Returns:
      False once the ceiling is reached, which is the signal to unwind
      immediately rather than finish the path in hand. Checked at the top
      of every frame, so the overshoot is bounded by the path depth
      rather than by the size of the search space.
    '''
    self.left -= 1
    return self.left > 0


def affected_stocks(
  graph: DependencyGraph,
  key: str,
  max_depth: int = default_max_depth,
) -> dict[str, int]:
  '''Return every listed symbol reachable downstream of ``key``.

  Information travels against the stored edge direction: a stock that
  depends on crude is downstream of crude. Traversal mixes ``depends_on``
  and ``affected_by`` edges on purpose, because an RBI policy decision
  reaches banks by the same mechanism a repo-rate move does.

  Args:
    graph: Graph to walk.
    key: Node key the shock starts at, e.g. ``'commodity:crude'``.
    max_depth: Maximum number of hops. Must be in
      ``[1, max_traversal_depth]``.

  Returns:
    Mapping of bare NSE symbol to the **minimum** hop count at which it
    was reached. The origin node is excluded, so a shock at
    ``'stock:RELIANCE'`` does not report ``RELIANCE`` as affected by
    itself.

  Raises:
    ValueError: If ``key`` is not a node, or ``max_depth`` is out of
      range.
  '''
  _check_depth(max_depth)
  if not graph.has(key):
    raise ValueError(f'unknown node {key!r}')
  reached: dict[str, int] = {}
  visited = {key}
  frontier = deque([(key, 0)])
  while frontier:
    current, depth = frontier.popleft()
    if depth >= max_depth:
      continue
    for neighbour in graph.exposed_to(current):
      if neighbour in visited:
        continue
      visited.add(neighbour)
      if graph.node(neighbour).is_stock:
        reached[graph.node(neighbour).name] = depth + 1
      frontier.append((neighbour, depth + 1))
  return reached


def dependencies(graph: DependencyGraph, key: str) -> tuple[str, ...]:
  '''Return the nodes ``key`` points at, one hop out.

  This is the audit direction: for a stock it is the list of things the
  graph claims that stock cares about, which is what a reviewer needs to
  see when they are checking an edge.

  Args:
    graph: Graph to inspect.
    key: Node key.

  Returns:
    Target keys of ``key``'s outgoing edges, in insertion order. Empty
    when the key is absent from the graph, because "this node depends on
    nothing recorded" is the correct answer for a leaf.

  Raises:
    ValueError: Never, by design. An unknown key is treated as a leaf
      rather than an error, so an audit over a partial graph does not
      abort halfway through.
  '''
  return graph.causes_of(key)


def dependents(graph: DependencyGraph, key: str) -> tuple[str, ...]:
  '''Return the nodes pointing at ``key``, one hop out.

  Args:
    graph: Graph to inspect.
    key: Node key.

  Returns:
    Source keys of ``key``'s incoming edges, in insertion order. Empty
    when nothing points at the key.
  '''
  return graph.exposed_to(key)


def has_cycle(graph: DependencyGraph) -> bool:
  '''Report whether the graph contains a logical dependency loop.

  This is the cheap, linear question, and the right one to ask of a
  graph file before committing it. A loop is always an editorial mistake
  in this domain -- nothing is genuinely self-causing -- and it is
  exactly the mistake a hand-maintained file makes without anyone
  noticing.

  The walk follows the *dependency* direction, what a node points at,
  not the propagation direction. That distinction is the whole point:
  one crude node with ten dependents is a fan-out, not a loop, and an
  undirected search would report it as one. Only ``a depends_on b``
  together with ``b depends_on a`` is a loop.

  Args:
    graph: Graph to inspect.

  Returns:
    True when some node sits on a dependency loop. Iterative, so a long
    chain cannot raise ``RecursionError``.
  '''
  colour: dict[str, int] = {}
  for root in sorted(graph.nodes):
    if colour.get(root):
      continue
    colour[root] = 1
    stack: list[tuple[str, int]] = [(root, 0)]
    while stack:
      key, index = stack[-1]
      targets = graph.causes_of(key)
      if index >= len(targets):
        colour[key] = 2
        stack.pop()
        continue
      stack[-1] = (key, index + 1)
      target = targets[index]
      state = colour.get(target, 0)
      if state == 1:
        return True
      if state == 0:
        colour[target] = 1
        stack.append((target, 0))
  return False


def cycles(
  graph: DependencyGraph,
  max_cycles: int = 10,
  max_length: int = 8,
  max_expansions: int = max_cycle_expansions,
) -> tuple[tuple[str, ...], ...]:
  '''Report short dependency loops, so a reviewer can fix them.

  A loop is a *fact* about a hand-maintained graph rather than a runtime
  error: traversal stops expanding at a node it has already visited, so
  a cyclic graph walks more slowly but stays correct. What is not
  acceptable is a loop nobody noticed, because it almost always means
  two entries contradicting each other -- and in a file meant to be
  reviewed, a contradiction nobody sees is a contradiction nobody fixes.

  Only *short* loops are reported. Every simple loop in a dense graph is
  combinatorial, and the seed graph has 47 once fan-out is counted
  undirected; that is why this walk is directed and length-bounded.

  **An acyclic graph is answered, not walked.** :func:`has_cycle` is
  linear and settles the question in about a millisecond on the
  1977-edge graph, while enumerating that same acyclic graph to
  ``max_length`` costs 616 million path extensions and reports nothing.
  So the linear answer comes first and an acyclic file returns ``()``
  having walked nothing. That is the common case, not a corner case: the
  seed graph is honestly acyclic.

  Args:
    graph: Graph to inspect.
    max_cycles: Cap on how many loops are *reported*. It bounds the
      output, not the work: the walk can be long over before ten loops
      are found, which is what :func:`has_cycle` and ``max_expansions``
      are for.
    max_length: Longest loop reported, counted in distinct keys. At
      least 3: a two-key loop needs two contradictory edges, and a
      one-key loop needs a self-edge.
    max_expansions: Ceiling on path extensions, the walk's unit of work.
      At least 1. See :data:`max_cycle_expansions` for the default and
      the measurements behind it.

  Returns:
    Tuple of loops, each a key sequence whose first key is repeated as
    its last. Deterministic -- nodes and targets are walked in sorted
    order, and each loop is reported once, from its alphabetically first
    member. Empty for an acyclic graph, a fan-out, and a loop longer than
    ``max_length``.

  Raises:
    ValueError: If ``max_cycles`` is below 1, ``max_length`` below 3, or
      ``max_expansions`` below 1.
    RuntimeError: If the walk reaches ``max_expansions`` without
      collecting ``max_cycles`` loops. The message names the ceiling and
      the graph size, because the honest options at that point are to
      raise the ceiling deliberately or to ask a narrower question with
      ``max_cycles`` or ``max_length``.
  '''
  if max_cycles < 1:
    raise ValueError(f'max_cycles must be >= 1, got {max_cycles}')
  if max_length < 3:
    raise ValueError(f'max_length must be >= 3, got {max_length}')
  if max_expansions < 1:
    raise ValueError(f'max_expansions must be >= 1, got {max_expansions}')
  if not has_cycle(graph):
    # The gate, and the cheapest fix in this module: an acyclic file has
    # no loop to name, and answering it costs one linear pass instead of
    # the 616 million extensions enumeration spends reporting nothing on
    # the 1977-edge graph. Not a gate on a cyclic one, which is the case
    # enumeration exists for.
    return ()
  adjacency = {key: sorted(graph.causes_of(key))
               for key in sorted(graph.nodes)}
  found: list[tuple[str, ...]] = []
  reported: set[frozenset[str]] = set()
  meter = _Meter(left=max_expansions, ceiling=max_expansions)
  for start in sorted(graph.nodes):
    _walk(start, start, adjacency, [start], {start}, found, reported,
          max_cycles, max_length, meter)
    # A caller who asked for N loops and got N is answered, whatever the
    # walk cost on the way, so neither the rest of the graph nor the
    # ceiling is consulted while the audit still wants something. The
    # ceiling second, so a full count is never reported as a refusal.
    if len(found) >= max_cycles:
      break
    if not meter.left:
      raise RuntimeError(
        f'cycle enumeration exhausted its budget of {meter.ceiling} path '
        f'extensions on a graph of {graph.node_count} nodes and '
        f'{graph.edge_count} edges; a file this size either holds a loop '
        f'longer than max_length={max_length} or loops too tangled to '
        f'enumerate cheaply -- raise max_expansions deliberately, or ask a '
        f'narrower question with max_cycles or max_length')
  return tuple(found)


def _walk(
  current: str,
  start: str,
  adjacency: dict[str, list[str]],
  path: list[str],
  on_path: set[str],
  found: list[tuple[str, ...]],
  reported: set[frozenset[str]],
  max_cycles: int,
  max_length: int,
  meter: _Meter,
) -> None:
  '''Depth-first search for dependency loops starting at ``start``.

  Args:
    current: Node key being expanded.
    start: Node key the search began at.
    adjacency: Directed dependency adjacency, sorted.
    path: Current path, a list mutated in place.
    on_path: Keys currently on ``path``, the loop detector.
    found: Accumulator of reported loops.
    reported: Node sets already reported, so one loop found from two
      starting points appears once.
    max_cycles: Cap on loops reported.
    max_length: Longest loop reported, counted in distinct keys. Checked
      before the closure is tested, so ``max_length=3`` still reports a
      three-key loop.
    meter: Shared extension budget. Spent once per frame on entry, and
      the frame returns immediately when it is dry, so an exhausted
      budget unwinds the search instead of finishing it.
  '''
  if not meter.spend():
    return
  if len(found) >= max_cycles or len(path) > max_length:
    return
  for target in adjacency.get(current, ()):
    if target == start and len(path) > 1:
      signature = frozenset(path)
      if signature not in reported:
        reported.add(signature)
        found.append(tuple(path + [start]))
      continue
    if target in on_path or target < start:
      # ``target < start`` reports each loop once, from its
      # alphabetically first member, with no canonicalisation pass.
      continue
    path.append(target)
    on_path.add(target)
    _walk(target, start, adjacency, path, on_path, found, reported,
          max_cycles, max_length, meter)
    on_path.discard(target)
    path.pop()
    if not meter.left:
      return


def _check_depth(max_depth: int) -> None:
  '''Validate a requested traversal depth.

  Args:
    max_depth: The requested depth.

  Raises:
    ValueError: If the depth is not in ``[1, max_traversal_depth]``.
  '''
  if max_depth < 1:
    raise ValueError(f'max_depth must be >= 1, got {max_depth}')
  if max_depth > max_traversal_depth:
    raise ValueError(
      f'max_depth must be <= {max_traversal_depth}, got {max_depth}; a '
      f'deep request on a cyclic graph has no meaning')
