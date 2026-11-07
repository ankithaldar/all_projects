#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Discovery graph builder.

v1 pipeline: load_profile → build_plan → (Send fan-out) fetch_pair →
normalize_dedupe. Later features append enrichment/embedding/scoring nodes.

State shape notes: ``enriched`` values are stored as plain dicts
(consistent across enrich_jds and checkpoint serialization), and
``errors`` entries are ``NodeError`` instances matching the
``List[NodeError]`` annotation in graph/state.py.
'''

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langgraph.graph import StateGraph
from langgraph.types import Send
from job_hunter.graph import nodes as node_fns
from job_hunter.graph.state import DiscoveryState, initial_state

logger = logging.getLogger(__name__)

MAX_FANOUT = 50


def route_targets(state: Dict[str, Any]) -> list:
  '''Fan out one Send per planned target, capped at 50.

  Args:
    state: Current state after build_plan.

  Returns:
    List of Send payloads (at most MAX_FANOUT).
  '''
  plan = state.get('plan') or {}
  targets = (
    plan.get('targets')
    if isinstance(plan, dict)
    else getattr(plan, 'targets', []) or []
  )
  targets = list(targets or [])
  if len(targets) > MAX_FANOUT:
    logger.warning(
      'route_targets: truncating fan-out from %s to %s targets',
      len(targets),
      MAX_FANOUT,
    )
    targets = targets[:MAX_FANOUT]
  return [Send('fetch_pair', target) for target in targets]


def build_discovery_graph(checkpointer: Optional[Any] = None):
  '''Compile the v1 discovery graph.

  Args:
    checkpointer: Optional LangGraph checkpointer for resumability.

  Returns:
    Compiled StateGraph.
  '''
  builder = StateGraph(DiscoveryState)
  builder.add_node('load_profile', node_fns.load_profile)
  builder.add_node('build_plan', node_fns.build_plan)
  builder.add_node('fetch_pair', node_fns.fetch_pair)
  builder.add_node('normalize_dedupe', node_fns.normalize_dedupe)
  builder.add_node('enrich_jds', node_fns.enrich_jds)
  builder.add_node('compute_embeddings', node_fns.compute_embeddings)
  builder.add_node('score_rank_persist', node_fns.score_rank_persist)
  builder.add_node('summarize_run', node_fns.summarize_run)

  builder.set_entry_point('load_profile')
  builder.add_edge('load_profile', 'build_plan')
  builder.add_conditional_edges('build_plan', route_targets, ['fetch_pair'])
  builder.add_edge('fetch_pair', 'normalize_dedupe')
  builder.add_edge('normalize_dedupe', 'enrich_jds')
  builder.add_edge('enrich_jds', 'compute_embeddings')
  builder.add_edge('compute_embeddings', 'score_rank_persist')
  builder.add_edge('score_rank_persist', 'summarize_run')
  builder.add_edge('summarize_run', '__end__')
  return builder.compile(checkpointer=checkpointer)


async def run_discovery(
  settings,
  run_id: int,
  triggered_by: str = 'scheduler',
  quick_poll: bool = False,
) -> Dict[str, Any]:
  '''Execute the discovery graph for one run id.

  Args:
    settings: Application settings.
    run_id: Run row id.
    triggered_by: Origin label.
    quick_poll: Whether this is a lightweight pinned-company poll.

  Returns:
    Final state snapshot.
  '''
  from job_hunter.graph.checkpointing import open_checkpointer, thread_config

  seed = initial_state(run_id, triggered_by)
  seed['plan'] = {'quick_poll': quick_poll}
  async with open_checkpointer(settings.checkpoint_path) as saver:
    graph = build_discovery_graph(checkpointer=saver)
    result = await graph.ainvoke(seed, config=thread_config(run_id, settings))
  return result
