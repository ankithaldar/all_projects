#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Static market-dependency graph, as dataclasses and a JSON file.

The design doc's answer to "which stocks does a crude shock reach" is a
Neo4j Knowledge Graph Service. The research review is blunt about this:
a Nifty-50 dependency graph is ~300-2000 **static** edges, Neo4j is
"second only to Redis" in indefensibility for this project, and a Cypher
dependency breaks the standard-library isolation that makes backtests
reproducible. So the graph here is frozen dataclasses with two adjacency
dicts, serialised to a reviewable JSON file.

**No agent, no LLM, no debate.** The graph is one of the independent
feature extractors the LLM-agents review says is the only defensible
architecture: extractors with equal weight and an explicit anti-convergence
guard, not a bull/bear/judge loop that loses to equal-weight two thirds
of the time.

What is here, and what is deliberately not:

* :mod:`stock_rl.graph.nodes` -- the five node kinds, namespaced keys.
* :mod:`stock_rl.graph.edges` -- two edge kinds, the graph container,
  JSON round-tripping, and a **starter** Nifty-50 seed that is a hypothesis
  about exposure rather than a measured one.
* :mod:`stock_rl.graph.traversal` -- bounded, cycle-safe reachability.

Not here, on purpose: node embeddings, a vector store, a learned
similarity over nodes, event extraction, and any graph write path. All of
those were proposed; none of them survived the evidence review.
'''

from stock_rl.graph.edges import (
  DependencyGraph,
  Edge,
  EdgeKind,
  nifty50_seed,
  schema_version,
)
from stock_rl.graph.nodes import (
  Node,
  NodeKind,
  make_key,
  nifty50_stocks,
  parse_key,
  sector_labels,
  stock_sectors,
)
from stock_rl.graph.traversal import (
  affected_stocks,
  cycles,
  default_max_depth,
  dependencies,
  dependents,
  has_cycle,
  max_traversal_depth,
)

__all__ = [
  'DependencyGraph',
  'Edge',
  'EdgeKind',
  'Node',
  'NodeKind',
  'affected_stocks',
  'cycles',
  'default_max_depth',
  'dependencies',
  'dependents',
  'has_cycle',
  'make_key',
  'max_traversal_depth',
  'nifty50_seed',
  'nifty50_stocks',
  'parse_key',
  'schema_version',
  'sector_labels',
  'stock_sectors',
]
