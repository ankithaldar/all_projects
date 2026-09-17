#!/usr/bin/env python
# -- coding: utf-8 --

'''PlanningAgent: the Chapter 2 orchestrator.

Architecture (one run):

    PlanningTaskInput
        │
        ▼
    ReasoningPolicy.decide ──────▶ PolicyDecision {route, rationale, scores}
        │
        ├─ direct ─▶ one LLM call
        ├─ cot    ─▶ ChainOfThoughtReasoner (step labels + self-consistency)
        ├─ react  ─▶ ReActAgent (Thought/Action/Observation over gated tools)
        └─ plan   ─▶ TaskPlanner ─▶ PlanExecutor (DAG waves; each node runs
                     as its own direct/cot/react sub-task) ─▶ synthesis
        │
        ▼
    SelfValidator (deterministic checks + critic + bounded revision)
        │
        ▼
    PlanningTaskOutput + events/memory/audits/traces persisted

Design principles applied:
  - Single Responsibility: routing (`policy`), reasoning (`patterns`),
    tool access (`tools`), validation (`self_validation`) and orchestration
    (this class) are separate and independently testable.
  - Open/Closed: a new route is added by implementing a pattern and one
    branch here; providers plug in behind the `ToolProvider` interface.
  - Dependency Inversion: depends on the `ToolProvider` abstraction and on
    the gateway facade, not on MCP transports or provider SDKs.
  - Fail-safe: every run returns a structured output. Crashes, gateway
    outages, budget exhaustion, and invalid plans degrade to typed results
    with machine-readable status and errors - never an exception to callers.
'''


from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from agentic_common.logging import get_logger, log_event
from agentic_common.persistence import AgentStore
from agentic_common.settings import Settings, default_settings
from agentic_common.tracing import TokenUsage, Tracer
from chapter02_planning.config import PlanningConfig, load_planning_config
from chapter02_planning.llm import ReasoningLLM, usage_delta
from chapter02_planning.patterns.chain_of_thought import (
  ChainOfThoughtReasoner,
)
from chapter02_planning.patterns.decomposition import (
  PlanExecutor,
  TaskPlanner,
)
from chapter02_planning.patterns.react import ReActAgent
from chapter02_planning.patterns.self_validation import SelfValidator
from chapter02_planning.policy import ReasoningPolicy, choose_node_route
from chapter02_planning.prompting import (
  DIRECT_SYSTEM,
  NODE_PROMPT,
  NODE_SYSTEM,
  SYNTHESIS_PROMPT,
  SYNTHESIS_SYSTEM,
)
from chapter02_planning.schemas import (
  ExecutionReport,
  NodeResult,
  Plan,
  PlanningTaskInput,
  PlanningTaskOutput,
  PlanNode,
  ReActTrace,
  ReasoningRoute,
  ToolExecutionAudit,
  ValidationReport,
)
from chapter02_planning.tools.provider import (
  Approver,
  GatedToolProvider,
  ToolProvider,
)

logger = get_logger(__name__)


class PlanningAgent:
  '''Routes tasks to reasoning patterns, executes them, and validates.'''

  def __init__(
    self,
    llm: Any,
    tools: Optional[ToolProvider] = None,
    config: Optional[PlanningConfig] = None,
    settings: Optional[Settings] = None,
    store: Optional[AgentStore] = None,
    tracer: Optional[Tracer] = None,
    write_approver: Optional[Approver] = None,
  ) -> None:
    '''Initialize the agent.

    Args:
      llm: GatewayClient or MockGateway (same `complete()` interface).
      tools: Raw tool provider (the agent applies the safety gate per run).
      config: Chapter 2 configuration; loaded from env/settings when omitted.
      settings: Optional shared settings seeding the config.
      store: Optional persistence for sessions, events, memory, audits.
      tracer: Optional tracer for spans.
      write_approver: Optional callback `(tool, args) -> bool` for writes.
    '''
    self._gateway = llm
    self._settings = settings or default_settings()
    self._config = config or load_planning_config(self._settings)
    self._tools = tools
    self._store = store
    self._tracer = tracer
    self._write_approver = write_approver
    self._policy = ReasoningPolicy(self._config)
    self._audits: List[ToolExecutionAudit] = []
    self._current_session_id = ''

  def close(self) -> None:
    '''Release the tool provider (and any MCP sessions) it owns.'''
    if self._tools is not None:
      try:
        self._tools.close()
      except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(
          'tool provider close failed',
          extra={'extra_fields': {'error': str(exc)}},
        )

  # ------------------------------------------------------------------
  # Public entry point
  # ------------------------------------------------------------------

  def run_task(self, task_input: PlanningTaskInput) -> PlanningTaskOutput:
    '''Run one task end-to-end.

    Args:
      task_input: Validated task input.

    Returns:
      PlanningTaskOutput. Guaranteed to return; internal crashes are
      captured as `status='error'` with the error recorded.
    '''
    started = time.perf_counter()
    trace_id = (
      self._tracer.new_trace_id() if self._tracer is not None
      else uuid.uuid4().hex
    )
    self._audits = []
    session_id = task_input.session_id

    self._persist_start(session_id, task_input)

    try:
      output = self._run_inner(task_input, trace_id)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      log_event(
        logger, 40, 'task_crashed', session_id=session_id, error=str(exc),
      )
      output = PlanningTaskOutput(
        session_id=session_id,
        answer='',
        status='error',
        errors=[f'agent crashed: {exc}'],
      )

    output.tool_calls = list(self._audits)
    output.duration_ms = (time.perf_counter() - started) * 1000.0
    self._persist_finish(session_id, output)

    log_event(
      logger, 20, 'task_done',
      session_id=session_id,
      route=output.route,
      status=output.status,
      tool_calls=len(output.tool_calls),
      errors=len(output.errors),
      duration_ms=round(output.duration_ms, 1),
    )
    return output

  # ------------------------------------------------------------------
  # Run internals
  # ------------------------------------------------------------------

  def _run_inner(
    self,
    task_input: PlanningTaskInput,
    trace_id: str,
  ) -> PlanningTaskOutput:
    '''Execute the routing + pattern + validation pipeline.

    Args:
      task_input: Validated task input.
      trace_id: Trace id for this run.

    Returns:
      PlanningTaskOutput without duration/tool audit aggregation.
    '''
    session_id = task_input.session_id
    tools_available = self._tools is not None and self._has_tools()
    decision = self._policy.decide(
      task_input.task,
      has_tools=tools_available,
      route_hint=task_input.prefer_route,
    )
    self._persist_event(
      session_id, 'route_decision',
      {
        'route': decision.route,
        'rationale': decision.rationale,
        'confidence': decision.confidence,
        'scores': decision.scores,
        'forced': decision.forced,
      },
    )

    llm = ReasoningLLM(
      self._gateway,
      self._config,
      tracer=self._tracer,
      trace_id=trace_id,
      session_id=session_id,
    )
    gate = self._build_gate(task_input, trace_id)

    output = PlanningTaskOutput(
      session_id=session_id,
      route=decision.route,
      policy=decision,
    )
    answer = ''
    evidence = ''
    route_status = 'completed'

    if decision.route == 'direct':
      answer, errors = self._run_direct(
        llm, task_input.task, model_label='direct',
      )
      evidence = task_input.task
      output.errors.extend(errors)
      route_status = 'completed' if answer else 'error'

    elif decision.route == 'cot':
      cot = ChainOfThoughtReasoner(llm, self._config).reason(
        task_input.task, samples=task_input.cot_samples,
      )
      answer = cot.answer
      output.reasoning_steps = cot.steps
      output.errors.extend(cot.errors)
      evidence = '\n'.join(cot.steps) or task_input.task
      route_status = cot.status

    elif decision.route == 'react':
      trace = self._run_react(llm, gate, task_input)
      answer = trace.final_answer
      output.react = trace
      evidence = self._react_evidence(trace)
      route_status = trace.status
      if not answer:
        answer, errors = self._force_answer(llm, task_input.task, trace)
        output.errors.extend(errors)

    else:
      plan, plan_errors, execution = self._run_plan(
        llm, gate, task_input,
      )
      output.plan = plan
      output.execution = execution
      output.errors.extend(plan_errors)
      evidence = self._plan_evidence(execution)
      route_status = execution.status
      answer, errors = self._synthesize(llm, task_input.task, execution)
      output.errors.extend(errors)

    if llm.budget.exceeded:
      output.errors.append('token budget exhausted')
      if route_status == 'completed':
        route_status = 'error'

    answer, validation, status = self._validate(
      llm, task_input, answer, evidence, decision.route, route_status,
      node_results=(
        output.execution.results if output.execution is not None else None
      ),
    )
    output.validation = validation
    if validation is not None:
      output.errors.extend(validation.errors)

    output.answer = answer
    output.status = status
    output.usage = llm.usage_snapshot()

    self._persist_event(
      session_id, 'pattern_finished',
      {
        'route': decision.route,
        'status': status,
        'answer_preview': answer[:300],
        'usage': output.usage,
      },
    )
    self._remember(session_id, output)
    return output

  def _has_tools(self) -> bool:
    '''Check whether the tool provider offers any tool.

    Returns:
      True when at least one tool is listed.
    '''
    try:
      return bool(self._tools.list_tools()) if self._tools else False
    except Exception:  # pylint: disable=broad-exception-caught
      return False

  def _build_gate(
    self,
    task_input: PlanningTaskInput,
    trace_id: str,
  ) -> Optional[GatedToolProvider]:
    '''Wrap the raw tool provider in the safety gate for this run.

    Args:
      task_input: Task input (carries `allow_writes`).
      trace_id: Trace id for span correlation.

    Returns:
      Gated provider, or None when no tools are configured.
    '''
    if self._tools is None:
      return None
    return GatedToolProvider(
      inner=self._tools,
      allow_writes=task_input.allow_writes,
      require_approval=self._config.require_write_approval,
      approver=self._write_approver,
      max_result_chars=self._config.max_result_chars,
      sanitize=self._config.sanitize_observations,
      audit_sink=self._on_audit,
      tracer=self._tracer,
      trace_id=trace_id,
    )

  def _run_direct(
    self,
    llm: ReasoningLLM,
    task: str,
    model_label: str = 'direct',
    context: str = '',
    system: str = DIRECT_SYSTEM,
  ) -> Tuple[str, List[str]]:
    '''Answer with a single LLM call (no visible reasoning).

    Args:
      llm: Per-run reasoning LLM.
      task: Task text.
      model_label: Span/log label.
      context: Optional evidence context.
      system: System prompt (node sub-tasks use NODE_SYSTEM).

    Returns:
      (answer, errors) pair.
    '''
    prompt = f'Task:\n{task}'
    if context:
      prompt += f'\n\nContext:\n{context}'
    result = llm.complete(
      prompt,
      system=system,
      temperature=self._config.base_temperature,
      label=model_label,
    )
    if result is None:
      return '', [f'{model_label} call failed: {llm.last_error}']
    return result.content.strip(), []

  def _run_react(
    self,
    llm: ReasoningLLM,
    gate: Optional[GatedToolProvider],
    task_input: PlanningTaskInput,
  ) -> ReActTrace:
    '''Run the ReAct loop (or a safe error trace without tools).

    Args:
      llm: Per-run reasoning LLM.
      gate: Gated tool provider.
      task_input: Task input.

    Returns:
      ReActTrace.
    '''
    if gate is None:
      return ReActTrace(
        task=task_input.task,
        status='error',
        stop_reason='no_tools_available',
      )
    return ReActAgent(llm, gate, self._config).run(
      task_input.task, max_steps=task_input.max_steps,
    )

  def _force_answer(
    self,
    llm: ReasoningLLM,
    task: str,
    trace: ReActTrace,
  ) -> Tuple[str, List[str]]:
    '''Coerce a stalled ReAct run into a best-effort answer.

    Args:
      llm: Per-run reasoning LLM.
      task: Original task.
      trace: Stalled ReAct trace.

    Returns:
      (answer, errors) pair.
    '''
    observations = self._react_evidence(trace)
    answer, errors = self._run_direct(
      llm,
      f'{task}\n\nUse only this evidence and answer now.',
      model_label='react.force_answer',
      context=observations[:6000] or '(no evidence gathered)',
    )
    if not answer:
      errors.append('could not produce a final answer from ReAct state')
    return answer, errors

  def _run_plan(
    self,
    llm: ReasoningLLM,
    gate: Optional[GatedToolProvider],
    task_input: PlanningTaskInput,
  ) -> Tuple[Plan, List[str], ExecutionReport]:
    '''Plan, then execute the dependency DAG.

    Args:
      llm: Per-run reasoning LLM.
      gate: Gated tool provider (may be None).
      task_input: Task input.

    Returns:
      (plan, planner errors, execution report) triple.
    '''
    tools = gate.list_tools() if gate is not None else []
    plan, plan_errors = TaskPlanner(llm, self._config).plan(
      task_input.task, tools=tools,
    )
    executor = PlanExecutor(
      node_runner=self._node_runner(llm, gate, task_input.task),
      config=self._config,
    )
    execution = executor.execute(plan)
    return plan, plan_errors, execution

  def _node_runner(
    self,
    llm: ReasoningLLM,
    gate: Optional[GatedToolProvider],
    goal: str,
  ) -> Any:
    '''Build the executor callback for one plan node.

    Args:
      llm: Per-run reasoning LLM.
      gate: Gated tool provider.
      goal: Overall plan goal (for node context).

    Returns:
      Callable `(node, prior) -> NodeResult`.
    '''
    has_tools = gate is not None
    tools = gate.list_tools() if gate is not None else []

    def run_node(
      node: PlanNode,
      prior: Dict[str, NodeResult],
    ) -> NodeResult:
      '''Execute one node as its own bounded sub-task.

      Args:
        node: Node to execute.
        prior: Results of previously executed nodes.

      Returns:
        NodeResult with output, audits captured globally, and usage.
      '''
      before = self._spent(llm)
      audit_start = len(self._audits)
      context = self._dependency_context(node, prior)
      tool_lines = []
      for tool in tools:
        kind = 'read' if tool.read_only else 'write'
        tool_lines.append(f'- {tool.name} ({kind})')
      node_task = NODE_PROMPT.render(
        goal=goal,
        objective=node.objective,
        success_criteria=node.success_criteria or '(not specified)',
        context=context or '(none)',
        tools='\n'.join(tool_lines) or '(no tools available)',
      )
      route = choose_node_route(node, has_tools)
      errors: List[str] = []

      if route == 'react' and gate is not None:
        trace = ReActAgent(llm, gate, self._config).run(
          node_task, max_steps=min(4, self._config.max_steps),
        )
        output = trace.final_answer
        if trace.status != 'completed':
          errors.append(
            f'node {node.id}: react stopped with {trace.stop_reason}'
          )
      elif route == 'cot':
        cot = ChainOfThoughtReasoner(llm, self._config).reason(
          node_task, context=context, samples=1,
        )
        output = cot.answer
        errors.extend(cot.errors)
      else:
        output, direct_errors = self._run_direct(
          llm, node_task, model_label=f'node.{node.id}',
          context=context, system=NODE_SYSTEM,
        )
        errors.extend(direct_errors)

      usage = usage_delta(before, self._spent(llm))
      if llm.budget.exceeded:
        errors.append('token budget exhausted during plan execution')

      if output and not llm.budget.exceeded:
        status = 'completed'
      else:
        status = 'failed'

      return NodeResult(
        node_id=node.id,
        status=status,
        output=output,
        tool_calls=list(self._audits[audit_start:]),
        errors=errors,
        usage=usage,
      )

    return run_node

  def _synthesize(
    self,
    llm: ReasoningLLM,
    goal: str,
    execution: ExecutionReport,
  ) -> Tuple[str, List[str]]:
    '''Fold node outputs into one final answer.

    Args:
      llm: Per-run reasoning LLM.
      goal: Plan goal.
      execution: Execution report.

    Returns:
      (answer, errors) pair. Falls back to a deterministic join when the
      synthesis call fails.
    '''
    completed = [
      result for result in execution.results
      if result.status == 'completed'
    ]
    node_outputs = '\n'.join(
      f'- [{result.node_id}] {result.output[:500]}'
      for result in completed
    ) or '(no steps completed)'
    failed = []
    for result in execution.results:
      if result.status == 'completed':
        continue
      detail = result.skipped_reason or '; '.join(result.errors)
      failed.append(f'{result.node_id}: {detail}')

    prompt = SYNTHESIS_PROMPT.render(
      goal=goal,
      node_outputs=node_outputs,
      errors='\n'.join(failed) or '(none)',
    )
    result = llm.complete(
      prompt,
      system=SYNTHESIS_SYSTEM,
      temperature=self._config.base_temperature,
      label='synthesis',
    )
    if result is not None and result.content.strip():
      return result.content.strip(), []

    fallback = '; '.join(
      item.output for item in completed if item.output
    ) or 'No plan steps completed successfully.'
    errors = ['synthesis call failed; used deterministic join']
    if failed:
      errors.append('unresolved steps: ' + '; '.join(failed))
    return fallback, errors

  def _validate(
    self,
    llm: ReasoningLLM,
    task_input: PlanningTaskInput,
    answer: str,
    evidence: str,
    route: ReasoningRoute,
    route_status: str,
    node_results: Optional[List[NodeResult]] = None,
  ) -> Tuple[str, Optional[ValidationReport], str]:
    '''Run self-validation when enabled and adjust the run status.

    Args:
      llm: Per-run reasoning LLM.
      task_input: Task input.
      answer: Candidate answer.
      evidence: Evidence text.
      route: Route that produced the answer.
      route_status: Status from the pattern stage.
      node_results: Plan execution results (plan route only).

    Returns:
      (final answer, validation report or None, final status) triple.
    '''
    if not task_input.run_validation:
      return answer, None, route_status

    if llm.budget.exceeded:
      return answer, None, route_status

    validation = SelfValidator(llm, self._config).validate(
      task=task_input.task,
      answer=answer,
      evidence=evidence,
      route=route,
      tool_calls=list(self._audits),
      node_results=list(node_results) if node_results else None,
    )
    final_answer = validation.final_answer or answer
    status = route_status
    if route_status == 'completed' and not validation.passed:
      status = 'needs_review'
    return final_answer, validation, status

  # ------------------------------------------------------------------
  # Evidence helpers
  # ------------------------------------------------------------------

  @staticmethod
  def _react_evidence(trace: ReActTrace) -> str:
    '''Build evidence text from a ReAct trace.

    Args:
      trace: ReAct trace.

    Returns:
      Concatenated observations.
    '''
    return '\n'.join(
      f'[{step.index}] {step.observation}'
      for step in trace.steps if step.observation
    )

  @staticmethod
  def _plan_evidence(execution: ExecutionReport) -> str:
    '''Build evidence text from plan node outputs.

    Args:
      execution: Execution report.

    Returns:
      Concatenated completed outputs.
    '''
    return '\n'.join(
      f'[{result.node_id}] {result.output}'
      for result in execution.results if result.output
    )

  @staticmethod
  def _dependency_context(
    node: PlanNode,
    prior: Dict[str, NodeResult],
  ) -> str:
    '''Render dependency outputs as node context.

    Args:
      node: Node being executed.
      prior: Previously executed node results.

    Returns:
      Context text.
    '''
    lines: List[str] = []
    for dependency in node.depends_on:
      result = prior.get(dependency)
      if result is not None and result.output:
        lines.append(f'[{dependency}] {result.output[:400]}')
    return '\n'.join(lines)

  @staticmethod
  def _spent(llm: ReasoningLLM) -> TokenUsage:
    '''Snapshot the LLM token usage.

    Args:
      llm: Reasoning LLM.

    Returns:
      Current cumulative usage.
    '''
    return TokenUsage(
      input_tokens=llm.budget.spent.input_tokens,
      output_tokens=llm.budget.spent.output_tokens,
      total_tokens=llm.budget.spent.total_tokens,
    )

  # ------------------------------------------------------------------
  # Persistence and audit wiring
  # ------------------------------------------------------------------

  def _on_audit(self, audit: ToolExecutionAudit) -> None:
    '''Collect a tool audit and persist it when a store exists.

    Args:
      audit: Audit record produced by the safety gate.
    '''
    self._audits.append(audit)
    if self._store is None:
      return
    server, _, tool = audit.tool.partition('.')
    provider_name = server or (self._tools.name if self._tools else 'unknown')
    try:
      self._store.log_tool_call(
        session_id=self._current_session_id,
        server=provider_name,
        tool=tool or audit.tool,
        args=audit.args,
        result={'preview': audit.result_preview},
        ok=audit.ok,
        error=audit.error,
        latency_ms=audit.latency_ms,
        approved=audit.approved,
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      logger.warning(
        'audit persist failed',
        extra={'extra_fields': {'error': str(exc)}},
      )

  def _persist_start(
    self,
    session_id: str,
    task_input: PlanningTaskInput,
  ) -> None:
    '''Create the session and log task_start.

    Args:
      session_id: Session id.
      task_input: Task input.
    '''
    self._current_session_id = session_id
    if self._store is None:
      return
    try:
      self._store.ensure_session(
        session_id, meta={'chapter': '02_planning'},
      )
      self._store.log_event(
        session_id,
        'task_start',
        {
          'task': task_input.task[:500],
          'allow_writes': task_input.allow_writes,
        },
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      logger.warning(
        'persist start failed',
        extra={'extra_fields': {'error': str(exc)}},
      )

  def _persist_finish(
    self,
    session_id: str,
    output: PlanningTaskOutput,
  ) -> None:
    '''Persist the final event and last-answer memory.

    Args:
      session_id: Session id.
      output: Final output.
    '''
    if self._store is None:
      return
    try:
      self._store.log_event(
        session_id,
        'task_finish',
        {
          'route': output.route,
          'status': output.status,
          'answer_preview': output.answer[:300],
          'usage': output.usage,
          'errors': output.errors,
          'tool_calls': [
            audit.model_dump() for audit in output.tool_calls
          ],
          'duration_ms': round(output.duration_ms, 1),
        },
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      logger.warning(
        'persist finish failed',
        extra={'extra_fields': {'error': str(exc)}},
      )

  def _persist_event(
    self,
    session_id: str,
    event_type: str,
    payload: Dict[str, Any],
  ) -> None:
    '''Persist one execution-history event.

    Args:
      session_id: Session id.
      event_type: Event name.
      payload: Event payload.
    '''
    if self._store is None:
      return
    try:
      self._store.log_event(session_id, event_type, payload)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      logger.warning(
        'persist event failed',
        extra={'extra_fields': {'error': str(exc)}},
      )

  def _remember(
    self,
    session_id: str,
    output: PlanningTaskOutput,
  ) -> None:
    '''Store cross-run memory for the session.

    Args:
      session_id: Session id.
      output: Final output.
    '''
    if self._store is None:
      return
    try:
      self._store.remember(
        session_id,
        'last_answer',
        {
          'route': output.route,
          'status': output.status,
          'answer': output.answer[:500],
        },
      )
      if output.plan is not None:
        self._store.remember(
          session_id,
          'last_plan',
          output.plan.model_dump(mode='json'),
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      logger.warning(
        'remember failed',
        extra={'extra_fields': {'error': str(exc)}},
      )
