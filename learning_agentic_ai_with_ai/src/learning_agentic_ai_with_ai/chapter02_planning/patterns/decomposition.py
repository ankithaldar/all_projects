#!/usr/bin/env python
# -- coding: utf-8 --

'''Task decomposition and dependency-aware planning.

Pattern: PLANNER-EXECUTOR with a dependency DAG

    goal ──▶ PLANNER (LLM, structured JSON) ──▶ Plan{nodes, depends_on}
                                                     │
                                       validate (pydantic + networkx:
                                       duplicate ids? unknown deps? cycle?)
                                                     │
                                       topological_waves() = execution order
                                                     │
                       ┌───────────── wave 1 ──────────────┐
                       │  node A        node B             │
                       └───────┬────────────┬──────────────┘
                               ▼            ▼
                       ┌───────────── wave 2 ──────────────┐
                       │  node C (depends on A and B)      │
                       └───────────────────────────────────┘
                                                     │
                                                     ▼
                                    EXECUTOR ─▶ NodeResults ─▶ report

Why a DAG instead of a linear chain: real operations tasks have partial
order. "Pull low-stock items" and "pull sales trends" can happen before
"size the restock order", and "notify the store" depends on the order
existing. Explicit dependencies let independent work run first, let the
executor skip exactly the nodes invalidated by a failure (instead of the
whole plan), and make replanning a local edit rather than a restart.

Production guardrails:
  - the planner's JSON is validated before execution; invalid or cyclic
    plans degrade to a safe single-node plan (never a crash),
  - node count and wave depth are capped,
  - a failed node marks only its dependents as skipped; independent branches
    still complete (safe partial completion).
'''


from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx

from agentic_common.logging import get_logger, log_event
from chapter02_planning.config import PlanningConfig
from chapter02_planning.llm import ReasoningLLM
from chapter02_planning.prompting import (
  PLAN_PROMPT,
  PLAN_SYSTEM_TEMPLATE,
  output_contract,
  render_tool_catalog,
)
from chapter02_planning.schemas import (
  ExecutionReport,
  NodeResult,
  Plan,
  PlanNode,
  ToolSpec,
)

logger = get_logger(__name__)

NodeRunner = Callable[[PlanNode, Dict[str, NodeResult]], NodeResult]


def plan_to_graph(plan: Plan) -> nx.DiGraph:
  '''Build a directed graph from a plan.

  Edges point from dependency to dependent, so a topological order of the
  graph is a valid execution order.

  Args:
    plan: Validated plan.

  Returns:
    networkx DiGraph with one node per plan node.
  '''
  graph = nx.DiGraph()
  for node in plan.nodes:
    graph.add_node(node.id)
  for node in plan.nodes:
    for dependency in node.depends_on:
      graph.add_edge(dependency, node.id)
  return graph


def topological_waves(plan: Plan) -> List[List[str]]:
  '''Group plan nodes into dependency generations.

  Nodes in the same wave have no dependencies on each other and may run in
  any order (or in parallel in a future async executor).

  Args:
    plan: Validated plan.

  Returns:
    List of waves; each wave is a sorted list of node ids.

  Raises:
    ValueError: When the plan contains a cycle.
  '''
  graph = plan_to_graph(plan)
  if not nx.is_directed_acyclic_graph(graph):
    cycle = nx.find_cycle(graph)
    raise ValueError(f'plan contains a dependency cycle: {cycle}')

  generations = nx.topological_generations(graph)
  return [sorted(generation) for generation in generations]


def validate_plan(
  plan: Plan,
  max_nodes: int = 8,
  max_waves: int = 6,
) -> List[str]:
  '''Validate plan size and acyclicity before execution.

  Structural checks (duplicate ids, unknown/self dependencies) already ran
  in the pydantic validator; this adds the graph-level and budget checks.

  Args:
    plan: Proposed plan.
    max_nodes: Maximum accepted node count.
    max_waves: Maximum accepted dependency depth.

  Returns:
    List of error strings (empty when the plan is executable).
  '''
  errors: List[str] = []

  if not plan.nodes:
    return ['plan has no nodes']

  if len(plan.nodes) > max_nodes:
    errors.append(
      f'plan has {len(plan.nodes)} nodes (max {max_nodes})'
    )

  graph = plan_to_graph(plan)
  if not nx.is_directed_acyclic_graph(graph):
    cycle = nx.find_cycle(graph)
    errors.append(f'plan contains a dependency cycle: {cycle}')
    return errors

  waves = [sorted(gen) for gen in nx.topological_generations(graph)]
  if len(waves) > max_waves:
    errors.append(
      f'plan has {len(waves)} dependency waves (max {max_waves})'
    )

  return errors


class TaskPlanner:
  '''LLM planner that emits a validated, dependency-aware plan.'''

  def __init__(
    self,
    llm: ReasoningLLM,
    config: PlanningConfig,
  ) -> None:
    '''Initialize the planner.

    Args:
      llm: Gateway-backed reasoning LLM wrapper.
      config: Chapter 2 configuration.
    '''
    self._llm = llm
    self._config = config

  def plan(
    self,
    goal: str,
    tools: Optional[List[ToolSpec]] = None,
  ) -> Tuple[Plan, List[str]]:
    '''Produce a plan for a goal, with a safe fallback.

    Args:
      goal: Natural-language goal.
      tools: Tools available to execute the plan.

    Returns:
      (plan, errors) where errors explains any fallback that happened. The
      returned plan is always executable (validated structurally).
    '''
    tool_specs = tools or []
    system = PLAN_SYSTEM_TEMPLATE.render(
      max_nodes=self._config.max_plan_nodes,
    )
    prompt = PLAN_PROMPT.render(
      goal=goal,
      tools=render_tool_catalog(tool_specs),
      contract=output_contract(
        Plan,
        'Return the plan as JSON. Each node needs id, objective, and '
        'depends_on (use [] for the first nodes).',
      ),
    )

    plan = self._llm.complete_json(
      prompt,
      Plan,
      system=system,
      temperature=self._config.planner_temperature,
      label='plan',
    )
    if plan is None:
      fallback = self._fallback_plan(
        goal, tools=tool_specs,
        reason=self._llm.last_error or 'planner returned no plan',
      )
      return fallback, ['planner fallback: ' + (self._llm.last_error or '')]

    plan = self._normalize_suggested_tools(plan, tool_specs)
    errors = validate_plan(
      plan, self._config.max_plan_nodes, self._config.max_plan_waves,
    )
    if errors:
      reason = '; '.join(errors)
      fallback = self._fallback_plan(
        goal, tools=tool_specs, reason=reason,
      )
      return fallback, [f'planner fallback: {reason}']

    log_event(
      logger, 20, 'plan_ready',
      nodes=len(plan.nodes),
      waves=len(topological_waves(plan)),
    )
    return plan, []

  def _normalize_suggested_tools(
    self,
    plan: Plan,
    tools: List[ToolSpec],
  ) -> Plan:
    '''Keep only real tools in each node's suggestions.

    Models invent tool names ("analytics platform") or drop the server
    prefix. Filtering to the advertised catalog (accepting short names as
    aliases) keeps node routing honest: a node with no real tools routes to
    CoT/direct instead of attempting impossible ReAct calls.

    Args:
      plan: Parsed plan from the model.
      tools: Tools actually available.

    Returns:
      Plan with normalized `suggested_tools`.
    '''
    known = {tool.name for tool in tools}
    aliases = {
      name.split('.')[-1].lower(): name for name in known
    }
    normalized_nodes: List[PlanNode] = []
    dropped = 0

    for node in plan.nodes:
      normalized: List[str] = []
      for suggested in node.suggested_tools:
        candidate = suggested if suggested in known else aliases.get(
          suggested.lower().split('.')[-1],
        )
        if candidate is None:
          dropped += 1
          continue
        if candidate not in normalized:
          normalized.append(candidate)
      normalized_nodes.append(
        node.model_copy(update={'suggested_tools': normalized})
      )

    if dropped:
      log_event(
        logger, 20, 'plan_tools_filtered', dropped=dropped,
      )
    return plan.model_copy(update={'nodes': normalized_nodes})

  def _fallback_plan(
    self,
    goal: str,
    tools: List[ToolSpec],
    reason: str,
  ) -> Plan:
    '''Build a single-node plan used when planning fails.

    This is deliberate graceful degradation: an agent that cannot plan
    should still *try the task once* rather than refuse.

    Args:
      goal: Original goal.
      tools: Available tools (decides the node route).
      reason: Why the fallback triggered.

    Returns:
      A one-node Plan.
    '''
    route = 'react' if tools else 'cot'
    log_event(logger, 30, 'plan_fallback', reason=reason, route=route)
    return Plan(
      goal=goal,
      nodes=[
        PlanNode(
          id='execute_goal',
          objective=goal,
          route=route,
          success_criteria='produce a useful answer for the goal',
        )
      ],
      notes=f'fallback single-node plan ({reason})',
    )


class PlanExecutor:
  '''Executes a validated plan wave by wave with safe partial completion.'''

  def __init__(
    self,
    node_runner: NodeRunner,
    config: PlanningConfig,
  ) -> None:
    '''Initialize the executor.

    Args:
      node_runner: Callable executing one node with prior results as context.
      config: Chapter 2 configuration.
    '''
    self._node_runner = node_runner
    self._config = config

  def execute(self, plan: Plan) -> ExecutionReport:
    '''Execute every node in dependency order.

    Args:
      plan: Validated plan.

    Returns:
      ExecutionReport with one NodeResult per node and aggregate status.
    '''
    try:
      waves = topological_waves(plan)
    except ValueError as exc:
      log_event(logger, 40, 'plan_invalid', error=str(exc))
      return ExecutionReport(
        goal=plan.goal, status='error', errors=[str(exc)],
      )

    node_map = plan.node_map()
    results: Dict[str, NodeResult] = {}
    ordered: List[NodeResult] = []
    errors: List[str] = []

    for wave in waves:
      for node_id in wave:
        node = node_map[node_id]
        failed_deps = [
          dependency for dependency in node.depends_on
          if results[dependency].status != 'completed'
        ]
        if failed_deps:
          deps_text = ', '.join(failed_deps)
          result = NodeResult(
            node_id=node_id,
            status='skipped',
            skipped_reason=f'dependencies not completed: {deps_text}',
          )
        else:
          result = self._run_node(node, results)
        results[node_id] = result
        ordered.append(result)
        errors.extend(result.errors)

    status = self._aggregate_status(ordered)
    report = ExecutionReport(
      goal=plan.goal,
      waves=waves,
      results=ordered,
      status=status,
      errors=errors,
      usage=self._sum_usage(ordered),
    )
    log_event(
      logger, 20, 'plan_executed',
      status=status,
      nodes=len(ordered),
      failed=sum(1 for r in ordered if r.status == 'failed'),
      skipped=sum(1 for r in ordered if r.status == 'skipped'),
    )
    return report

  # ------------------------------------------------------------------
  # Internals
  # ------------------------------------------------------------------

  def _run_node(
    self,
    node: PlanNode,
    prior: Dict[str, NodeResult],
  ) -> NodeResult:
    '''Execute one node, converting crashes into failed results.

    Args:
      node: Node to execute.
      prior: Results of previously executed nodes.

    Returns:
      NodeResult (status failed on exception).
    '''
    try:
      result = self._node_runner(node, prior)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      log_event(
        logger, 40, 'node_crashed',
        node_id=node.id, error=str(exc),
      )
      return NodeResult(
        node_id=node.id, status='failed',
        errors=[f'node crashed: {exc}'],
      )

    if result.node_id != node.id:
      result = result.model_copy(update={'node_id': node.id})
    return result

  @staticmethod
  def _aggregate_status(results: List[NodeResult]) -> str:
    '''Map node outcomes to an aggregate run status.

    Args:
      results: All node results.

    Returns:
      'completed', 'needs_review', or 'error'.
    '''
    if results and all(result.status == 'completed' for result in results):
      return 'completed'
    if any(result.status == 'completed' for result in results):
      return 'needs_review'
    return 'error'

  @staticmethod
  def _sum_usage(results: List[NodeResult]) -> Dict[str, int]:
    '''Sum usage dicts across node results.

    Args:
      results: Node results carrying usage dicts.

    Returns:
      Aggregated usage dict.
    '''
    totals = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
    for result in results:
      for key in totals:
        totals[key] += int(result.usage.get(key, 0) or 0)
    return totals
