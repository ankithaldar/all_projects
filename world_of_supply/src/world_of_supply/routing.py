#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Path-finding over the traversable part of the world grid.

The module is intentionally free of any knowledge about facilities or the
world object itself: it operates on plain sizes plus an ``is_traversable``
predicate (Strategy pattern). Caching of the built graph is a concern of the
caller (:class:`world_of_supply.world.World`).
'''

from __future__ import annotations

from collections.abc import Callable

import networkx as nx

CellCoord = tuple[int, int]


def build_traversable_graph(
    size_x: int,
    size_y: int,
    is_traversable: Callable[[int, int], bool],
) -> nx.Graph:
  '''Build an undirected graph connecting adjacent traversable cells.

  Args:
    size_x: Grid width.
    size_y: Grid height.
    is_traversable: Callable ``(x, y) -> bool`` deciding passability.

  Returns:
    nx.Graph: Graph whose nodes encode coordinates as ``x * size_y + y``.
  '''
  graph = nx.Graph()
  for x in range(1, size_x - 1):
    for y in range(1, size_y - 1):
      if not is_traversable(x, y):
        continue
      node = x * size_y + y
      for nx_, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
        if is_traversable(nx_, ny):
          graph.add_edge(node, nx_ * size_y + ny)
  return graph


def shortest_path(graph: nx.Graph, start: CellCoord, goal: CellCoord, size_y: int) -> list[CellCoord] | None:
  '''Find an A* shortest path between two coordinates.

  Args:
    graph: Graph produced by :func:`build_traversable_graph`.
    start: ``(x, y)`` origin.
    goal: ``(x, y)`` destination.
    size_y: Grid height used to decode integer nodes back to coordinates.

  Returns:
    list[tuple[int, int]] | None: Coordinate sequence from start to goal
    inclusive, or ``None`` when the goal is unreachable.
  '''
  source = start[0] * size_y + start[1]
  target = goal[0] * size_y + goal[1]
  try:
    nodes = nx.astar_path(graph, source=source, target=target)
  except (nx.NetworkXNoPath, nx.NodeNotFound):
    return None
  return [(node // size_y, node % size_y) for node in nodes]
