#!/usr/bin/env python
# -- coding: utf-8 --

'''Layer 3 - Decision: strategy-driven planning with introspection.

The Decision layer turns `Percept + MemoryContext + StrategyProfile` into a
validated `Decision`. It supports two plan shapes behind one pipeline:

  - **DAG planner** - the model returns JSON steps (`CognitivePlan`),
    validated structurally by pydantic and acyclically by networkx. Best
    for multi-step tasks that interleave reads, reasoning, and writes.
  - **Code planner** - the model writes a `solve(context)` Python function
    that uses a capability-based tool object. Best for computation-heavy
    tasks where arithmetic in JSON prompts is fragile. The code is statically
    validated before execution (see `codegen.py`).

Both planners degrade to a single-step **fallback plan** (`kind='reason'`)
when generation or validation fails, so the Action layer always has
something safe to execute.

`PlannerIntrospector` reviews the plan *before* execution: capability
coverage, write-before-read ordering, unknown tools, step/call estimates,
risk level, and confidence. This is the chapter's "planner introspection":
the agent inspects its own plan the way a senior engineer reviews a runbook.
'''


from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from agentic_common.logging import get_logger, log_event
from chapter03_cognition.codegen import validate_code
from chapter03_cognition.config import CognitionConfig
from chapter03_cognition.jsonio import JsonParseError, extract_code_block
from chapter03_cognition.llm import ReasoningLLM
from chapter03_cognition.prompts import (
  CODE_PLAN_PROMPT,
  CODE_PLAN_SYSTEM,
  DAG_PLAN_PROMPT,
  DAG_PLAN_SYSTEM,
  compact_json,
  format_memory_notes,
  output_contract,
  render_tool_catalog,
)
from chapter03_cognition.schemas import (
  CognitivePlan,
  Decision,
  MemoryContext,
  Percept,
  PlanInsight,
  PlanStep,
  StrategyProfile,
  ToolSpec,
)

logger = get_logger(__name__)

CAPABILITY_TOOLS: Dict[str, Set[str]] = {
  'inventory_read': {'retail-ops.retail_low_stock_report'},
  'sales_read': {'retail-ops.retail_sales_trend'},
  'restock_write': {'retail-ops.retail_restock_order'},
  'site_read': {
    'telecom-ops.telecom_site_status',
    'telecom-ops.telecom_degraded_sites',
  },
  'dispatch_write': {'telecom-ops.telecom_dispatch_technician'},
}

_PLACEHOLDER_RE = re.compile(r'^<.*>$|\.\.\.$|^the\s', re.IGNORECASE)


def plan_to_graph(plan: CognitivePlan) -> nx.DiGraph:
  '''Build a directed graph from a plan (dependency -> dependent).

  Args:
    plan: Plan with steps.

  Returns:
    networkx DiGraph with one node per step.
  '''
  graph = nx.DiGraph()
  for step in plan.steps:
    graph.add_node(step.id)
  for step in plan.steps:
    for dependency in step.depends_on:
      graph.add_edge(dependency, step.id)
  return graph


def topological_waves(plan: CognitivePlan) -> List[List[str]]:
  '''Group steps into dependency generations.

  Args:
    plan: Plan with steps.

  Returns:
    List of waves; each wave is a sorted list of step ids.

  Raises:
    ValueError: When the plan contains a cycle.
  '''
  graph = plan_to_graph(plan)
  if not nx.is_directed_acyclic_graph(graph):
    cycle = nx.find_cycle(graph)
    raise ValueError(f'plan contains a dependency cycle: {cycle}')
  return [
    sorted(generation) for generation in nx.topological_generations(graph)
  ]


def validate_plan(plan: CognitivePlan, max_steps: int = 8) -> List[str]:
  '''Validate plan size and acyclicity before execution.

  Args:
    plan: Proposed plan.
    max_steps: Maximum accepted step count.

  Returns:
    List of error strings (empty when executable).
  '''
  errors: List[str] = []
  if plan.source == 'code':
    if not plan.code.strip():
      errors.append('code plan has no code')
    return errors

  if not plan.steps:
    return ['plan has no steps']
  if len(plan.steps) > max_steps:
    errors.append(f'plan has {len(plan.steps)} steps (max {max_steps})')
  graph = plan_to_graph(plan)
  if not nx.is_directed_acyclic_graph(graph):
    cycle = nx.find_cycle(graph)
    errors.append(f'plan contains a dependency cycle: {cycle}')
  return errors


class DagPlanner:
  '''Structured JSON planner producing a dependency DAG.'''

  def __init__(self, llm: ReasoningLLM, config: CognitionConfig) -> None:
    '''Initialize the planner.

    Args:
      llm: Gateway-backed reasoning LLM wrapper.
      config: Chapter 3 configuration.
    '''
    self._llm = llm
    self._config = config

  def plan(
    self,
    goal: str,
    percept: Percept,
    memory: Optional[MemoryContext],
    profile: StrategyProfile,
    tools: List[ToolSpec],
  ) -> Tuple[CognitivePlan, List[str]]:
    '''Generate a DAG plan, with safe fallback.

    Args:
      goal: Task goal.
      percept: Perception-layer output.
      memory: Optional memory context.
      profile: Active strategy profile.
      tools: Available tools.

    Returns:
      (plan, notes) where notes describe normalizations or fallbacks.
    '''
    max_steps = min(profile.max_steps, self._config.max_steps)
    prompt = DAG_PLAN_PROMPT.render(
      goal=goal,
      domain=percept.signals.domain,
      risk=percept.signals.risk,
      data_need=percept.signals.data_need,
      multi_step=percept.signals.multi_step,
      entities=compact_json(percept.entities, 400),
      memory=format_memory_notes(
        memory.recall_notes if memory is not None else [],
      ),
      tools=render_tool_catalog(tools),
      max_steps=max_steps,
      contract=output_contract(
        CognitivePlan,
        'Return the plan as JSON. Each step needs id, objective, kind, '
        'and depends_on ([] for first steps).',
      ),
    )
    plan = self._llm.complete_json(
      prompt,
      CognitivePlan,
      system=DAG_PLAN_SYSTEM,
      temperature=profile.temperature,
      label='dag_plan',
    )
    if plan is None:
      reason = self._llm.last_error or 'planner returned no plan'
      fallback = fallback_plan(goal, profile, reason)
      return fallback, [f'planner fallback: {reason}']

    plan.source = 'dag'
    plan.strategy = profile.name
    plan.goal = plan.goal or goal
    plan, notes = self._normalize_steps(plan, tools)

    errors = validate_plan(plan, max_steps)
    if errors:
      reason = '; '.join(errors)
      fallback = fallback_plan(goal, profile, reason)
      return fallback, notes + [f'planner fallback: {reason}']

    log_event(
      logger, 20, 'dag_plan_ready',
      steps=len(plan.steps), waves=len(topological_waves(plan)),
    )
    return plan, notes

  def _normalize_steps(
    self,
    plan: CognitivePlan,
    tools: List[ToolSpec],
  ) -> Tuple[CognitivePlan, List[str]]:
    '''Filter unknown tools and drop empty steps.

    A step that names a tool the agent does not have is converted to a
    reasoning step so the executor is not set up for a guaranteed failure;
    the conversion is reported as a planner note.

    Args:
      plan: Parsed plan.
      tools: Available tools.

    Returns:
      (normalized plan, notes) pair.
    '''
    known = {tool.name for tool in tools}
    aliases = {name.split('.')[-1].lower(): name for name in known}
    notes: List[str] = []
    normalized_steps: List[PlanStep] = []

    for step in plan.steps:
      if not step.objective.strip():
        continue
      updated = step
      if step.tool and step.tool not in known:
        candidate = aliases.get(step.tool.lower().split('.')[-1])
        if candidate is not None:
          updated = step.model_copy(update={'tool': candidate})
        else:
          notes.append(
            f'step {step.id}: unknown tool {step.tool!r} converted to '
            'reasoning'
          )
          updated = step.model_copy(
            update={'tool': '', 'kind': 'reason', 'arguments': {}},
          )
      if updated.fallback_tool and updated.fallback_tool not in known:
        candidate = aliases.get(
          updated.fallback_tool.lower().split('.')[-1],
        )
        if candidate is not None:
          updated = updated.model_copy(
            update={'fallback_tool': candidate},
          )
        else:
          updated = updated.model_copy(update={'fallback_tool': ''})

      if updated.kind in ('tool', 'write') and not updated.tool:
        updated = updated.model_copy(update={'kind': 'reason'})
      normalized_steps.append(updated)

    normalized = plan.model_copy(update={'steps': normalized_steps})
    return normalized, notes


class CodePlanner:
  '''Agent-written Python planner producing a `solve()` code plan.'''

  def __init__(self, llm: ReasoningLLM, config: CognitionConfig) -> None:
    '''Initialize the planner.

    Args:
      llm: Gateway-backed reasoning LLM wrapper.
      config: Chapter 3 configuration.
    '''
    self._llm = llm
    self._config = config

  def plan(
    self,
    goal: str,
    percept: Percept,
    memory: Optional[MemoryContext],
    profile: StrategyProfile,
    tools: List[ToolSpec],
  ) -> Tuple[CognitivePlan, List[str]]:
    '''Generate a code plan, with one static-validation retry.

    Args:
      goal: Task goal.
      percept: Perception-layer output.
      memory: Optional memory context.
      profile: Active strategy profile.
      tools: Available tools.

    Returns:
      (plan, notes) pair; falls back on failure.
    '''
    facts: Dict[str, object] = {}
    if memory is not None:
      facts = dict(memory.facts)
    facts['perception'] = {
      'domain': percept.signals.domain,
      'entities': percept.entities,
      'risk': percept.signals.risk,
      'data_need': percept.signals.data_need,
    }

    prompt = CODE_PLAN_PROMPT.render(
      goal=goal,
      facts=compact_json(facts, 1200),
      tools=render_tool_catalog(tools),
      call_budget=self._config.code_call_budget,
    )
    notes: List[str] = []
    code = self._generate_code(prompt, profile, notes)
    if code is None:
      reason = self._llm.last_error or 'code generation failed'
      fallback = fallback_plan(goal, profile, reason)
      return fallback, notes + [f'code planner fallback: {reason}']

    plan = CognitivePlan(
      goal=goal,
      source='code',
      strategy=profile.name,
      code=code,
      entrypoint='solve',
      steps=[
        PlanStep(
          id='run_code',
          objective=goal,
          kind='code',
          expected_output='structured solve() result',
        )
      ],
      rationale='computation-heavy task: deterministic Python plan',
      assumptions=['numbers come from the task or tool results'],
    )
    log_event(logger, 20, 'code_plan_ready', chars=len(code))
    return plan, notes

  def _generate_code(
    self,
    prompt: str,
    profile: StrategyProfile,
    notes: List[str],
  ) -> Optional[str]:
    '''Generate and statically validate code, retrying once.

    Args:
      prompt: Code planning prompt.
      profile: Active strategy profile.
      notes: Mutable note collector.

    Returns:
      Validated code, or None.
    '''
    result = self._llm.complete(
      prompt,
      system=CODE_PLAN_SYSTEM,
      temperature=profile.temperature,
      label='code_plan',
    )
    if result is None:
      return None

    try:
      code = extract_code_block(result.content)
    except JsonParseError as exc:
      self._llm.last_error = str(exc)
      return None

    errors = validate_code(code, 'solve', self._config.code_max_chars)
    if not errors:
      return code

    notes.append('code plan rejected once: ' + '; '.join(errors[:3]))
    retry_prompt = (
      f'{prompt}\n\nYour previous code was rejected by the sandbox:\n'
      + '\n'.join(f'- {error}' for error in errors)
      + '\nRewrite the function without those violations. Output only code.'
    )
    retry = self._llm.complete(
      retry_prompt,
      system=CODE_PLAN_SYSTEM,
      temperature=0.0,
      label='code_plan.retry',
    )
    if retry is None:
      return None
    try:
      code = extract_code_block(retry.content)
    except JsonParseError as exc:
      self._llm.last_error = str(exc)
      return None
    errors = validate_code(code, 'solve', self._config.code_max_chars)
    if errors:
      self._llm.last_error = 'sandbox rejected generated code: ' + '; '.join(
        errors[:3],
      )
      return None
    return code


class PlannerIntrospector:
  '''Static review of a plan before execution.'''

  def introspect(
    self,
    plan: CognitivePlan,
    percept: Percept,
    tools: List[ToolSpec],
    profile: StrategyProfile,
    plan_notes: Optional[List[str]] = None,
  ) -> PlanInsight:
    '''Analyze a plan for risk, coverage, and likely failure modes.

    Args:
      plan: Plan about to execute.
      percept: Perception-layer output.
      tools: Available tools.
      profile: Active strategy profile.
      plan_notes: Planner notes to fold into introspection.

    Returns:
      PlanInsight.
    '''
    known = {tool.name for tool in tools}
    write_names = {
      tool.name for tool in tools if not tool.read_only
    }
    notes = list(plan_notes or [])
    warnings: List[str] = []

    if plan.source == 'code':
      insight = self._code_insight(profile, warnings)
    else:
      insight = self._dag_insight(
        plan, known, write_names, profile, warnings,
      )

    missing = self._missing_capabilities(plan, percept, known)
    if missing:
      warnings.append(
        'plan does not cover required capabilities: ' + ', '.join(missing)
      )

    for error in validate_plan(plan, max(profile.max_steps, 1)):
      warnings.append(error)

    insight.missing_capabilities = missing
    insight.warnings = sorted(set(warnings))
    insight.notes = notes + insight.notes
    insight.risk_level = self._risk_level(insight)
    insight.confidence = self._confidence(insight, plan)
    log_event(
      logger, 20, 'plan_introspected',
      source=plan.source,
      risk=insight.risk_level,
      warnings=len(insight.warnings),
      confidence=round(insight.confidence, 2),
    )
    return insight

  # ------------------------------------------------------------------
  # Internals
  # ------------------------------------------------------------------

  @staticmethod
  def _dag_insight(
    plan: CognitivePlan,
    known: Set[str],
    write_names: Set[str],
    profile: StrategyProfile,
    warnings: List[str],
  ) -> PlanInsight:
    '''Introspect a DAG or fallback plan.

    Args:
      plan: Plan to inspect.
      known: Known tool names.
      write_names: Write tool names.
      profile: Active strategy.
      warnings: Mutable warning list.

    Returns:
      Partial PlanInsight (capabilities filled by the caller).
    '''
    tool_calls = 0
    has_writes = False
    first_write_index: Optional[int] = None
    first_read_index: Optional[int] = None

    for index, step in enumerate(plan.steps):
      if step.kind in ('tool', 'write'):
        tool_calls += 1
      if step.tool and step.tool not in known:
        warnings.append(f'unknown tool in plan: {step.tool}')
      for value in step.arguments.values():
        if isinstance(value, str) and _PLACEHOLDER_RE.match(value.strip()):
          warnings.append(
            f'step {step.id}: argument looks like a placeholder: {value!r}'
          )
          break
      is_write = step.kind == 'write' or step.tool in write_names
      if is_write:
        has_writes = True
        if first_write_index is None:
          first_write_index = index
      elif step.kind == 'tool' and first_read_index is None:
        first_read_index = index

    if first_write_index is not None and (
      first_read_index is None or first_write_index < first_read_index
    ):
      warnings.append('plan writes before gathering any read evidence')

    if has_writes and not profile.retry.allow_writes:
      warnings.append(
        'plan contains writes but the strategy blocks them; they will be '
        'refused at execution'
      )

    try:
      waves = len(topological_waves(plan))
    except ValueError:
      waves = 0

    return PlanInsight(
      step_count=len(plan.steps),
      estimated_tool_calls=tool_calls,
      has_writes=has_writes,
      notes=[f'{len(plan.steps)} step(s) across {waves} wave(s)'],
    )

  @staticmethod
  def _code_insight(
    profile: StrategyProfile,
    warnings: List[str],
  ) -> PlanInsight:
    '''Introspect a code plan.

    Args:
      profile: Active strategy.
      warnings: Mutable warning list.

    Returns:
      Partial PlanInsight.
    '''
    if not profile.retry.allow_code_execution:
      warnings.append(
        'strategy disallows code execution; the plan will be replaced'
      )
    return PlanInsight(
      step_count=1,
      estimated_tool_calls=0,
      has_writes=False,
      notes=['agent-written solve() plan executed in a sandbox'],
    )

  @staticmethod
  def _missing_capabilities(
    plan: CognitivePlan,
    percept: Percept,
    known: Set[str],
  ) -> List[str]:
    '''List required capabilities the plan does not address.

    Args:
      plan: Plan to inspect.
      percept: Perception-layer output.
      known: Known tool names.

    Returns:
      Missing capability names.
    '''
    planned_tools = {
      step.tool for step in plan.steps if step.tool
    } | {
      step.fallback_tool for step in plan.steps if step.fallback_tool
    }
    if plan.source == 'code':
      planned_tools |= known

    missing: List[str] = []
    for capability in percept.required_capabilities:
      required = CAPABILITY_TOOLS.get(capability, set())
      if required and not required.intersection(planned_tools):
        missing.append(capability)
    return missing

  @staticmethod
  def _risk_level(insight: PlanInsight) -> str:
    '''Map insight features to a risk level.

    Args:
      insight: Insight under construction.

    Returns:
      'low', 'medium', or 'high'.
    '''
    if insight.has_writes or insight.missing_capabilities:
      return 'high'
    if insight.warnings or insight.step_count > 4:
      return 'medium'
    return 'low'

  @staticmethod
  def _confidence(insight: PlanInsight, plan: CognitivePlan) -> float:
    '''Heuristic confidence that the plan can succeed.

    Args:
      insight: Insight under construction.
      plan: The inspected plan.

    Returns:
      Confidence in 0..1.
    '''
    base = {
      'dag': 0.75,
      'code': 0.7,
      'fallback': 0.3,
    }.get(plan.source, 0.5)
    penalty = 0.1 * len(insight.warnings)
    if insight.missing_capabilities:
      penalty += 0.2
    return max(0.05, min(1.0, base - penalty))


def fallback_plan(
  goal: str,
  profile: StrategyProfile,
  reason: str,
) -> CognitivePlan:
  '''Build the single-step safe degradation plan.

  Args:
    goal: Task goal.
    profile: Active strategy profile.
    reason: Why the fallback triggered.

  Returns:
    A one-step reasoning plan.
  '''
  log_event(logger, 30, 'plan_fallback', reason=reason)
  return CognitivePlan(
    goal=goal,
    source='fallback',
    strategy=profile.name,
    steps=[
      PlanStep(
        id='answer_from_memory',
        objective=goal,
        kind='reason',
        expected_output='best-effort answer with explicit caveats',
      )
    ],
    rationale=f'fallback plan ({reason})',
    assumptions=['live data could not be planned reliably'],
  )


class DecisionLayer:
  '''Selects the planner and produces a validated, introspected Decision.'''

  def __init__(self, llm: ReasoningLLM, config: CognitionConfig) -> None:
    '''Initialize the layer.

    Args:
      llm: Gateway-backed reasoning LLM wrapper.
      config: Chapter 3 configuration.
    '''
    self._llm = llm
    self._config = config
    self._dag_planner = DagPlanner(llm, config)
    self._code_planner = CodePlanner(llm, config)
    self._introspector = PlannerIntrospector()

  def decide(
    self,
    percept: Percept,
    memory: Optional[MemoryContext],
    profile: StrategyProfile,
    tools: List[ToolSpec],
    plan_mode: str = 'auto',
  ) -> Decision:
    '''Produce a strategy-bound, validated, introspected plan.

    Args:
      percept: Perception-layer output.
      memory: Optional memory context.
      profile: Active strategy profile.
      tools: Available tools.
      plan_mode: 'auto', 'dag', or 'code'.

    Returns:
      Decision.
    '''
    use_code = self._use_code(percept, profile, plan_mode)
    if use_code:
      plan, notes = self._code_planner.plan(
        percept.task, percept, memory, profile, tools,
      )
    else:
      plan, notes = self._dag_planner.plan(
        percept.task, percept, memory, profile, tools,
      )

    if plan.source == 'code' and not (
      self._config.code_execution_enabled
      and profile.retry.allow_code_execution
    ):
      plan = fallback_plan(
        percept.task, profile, 'code execution disabled for this run',
      )
      notes.append('code plan replaced: execution disabled')

    insight = self._introspector.introspect(
      plan, percept, tools, profile, notes,
    )
    planner_notes = list(memory.recall_notes) if memory is not None else []
    return Decision(
      strategy=profile.name,
      strategy_reason=profile.description,
      plan=plan,
      insight=insight,
      planner_notes=planner_notes,
    )

  @staticmethod
  def _use_code(
    percept: Percept,
    profile: StrategyProfile,
    plan_mode: str,
  ) -> bool:
    '''Decide between code and DAG planning.

    Args:
      percept: Perception-layer output.
      profile: Active strategy profile.
      plan_mode: Caller override.

    Returns:
      True when a code plan should be produced.
    '''
    if plan_mode == 'code':
      return True
    if plan_mode == 'dag':
      return False
    if percept.signals.write_intent:
      return False
    quantities = percept.entities.get('quantities', [])
    computational = (
      len(quantities) >= 2
      and percept.signals.multi_step < 0.5
      and percept.signals.data_need < 0.5
    )
    return profile.prefer_code_plans and computational
