#!/usr/bin/env python
# -- coding: utf-8 --

'''Layer 4 - Action: execute plans with bounded adaptation.

The Action layer is the only place that touches the outside world. It
executes either shape the Decision layer produced:

  - **DAG plans** run wave by wave (topological order). Each step is a tool
    call, a write, or a reasoning call; failures are handled by the adaptive
    loop (retry / tool switching) and dependents of a failed step are
    skipped while independent branches still complete.
  - **Code plans** run the agent-written `solve()` function through the
    configured sandbox. Generated code receives only `task`, `facts`, and a
    `ToolCaller` capability object that enforces the allowlist, write
    policy, call budget, and deadline.

Cross-step adaptation (whole-plan replans and strategy escalation) is
deliberately *not* implemented here: it changes the plan, which belongs to
the Decision layer, so the agent drives it in a bounded loop. This module
owns only what happens inside one plan execution, and records every
adaptation it takes.

The answer is synthesized from completed steps (or taken directly from a
code plan), always with concrete evidence and honest mention of failures.
'''


from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from agentic_common.logging import get_logger, log_event
from agentic_common.security import sanitize_untrusted
from agentic_common.tracing import Tracer
from chapter03_cognition.adaptive import (
  AdaptiveLoop,
  classify_code_error,
  classify_tool_error,
)
from chapter03_cognition.codegen import (
  Deadline,
  ToolCaller,
  build_code_runner,
)
from chapter03_cognition.config import CognitionConfig
from chapter03_cognition.llm import ReasoningLLM
from chapter03_cognition.planner import topological_waves
from chapter03_cognition.prompts import (
  REASON_PROMPT,
  REASON_SYSTEM,
  SYNTHESIS_PROMPT,
  SYNTHESIS_SYSTEM,
  compact_json,
)
from chapter03_cognition.schemas import (
  ActionResult,
  AdaptationRecord,
  CognitivePlan,
  Decision,
  MemoryContext,
  Percept,
  PlanStep,
  StepAttempt,
  StepResult,
  ToolExecutionAudit,
)
from chapter03_cognition.strategies import profile_for

logger = get_logger(__name__)


class ActionLayer:
  '''Executes a Decision's plan with bounded adaptation.'''

  def __init__(
    self,
    llm: ReasoningLLM,
    toolbox: Any,
    config: CognitionConfig,
    tracer: Optional[Tracer] = None,
    trace_id: str = '',
  ) -> None:
    '''Initialize the layer.

    Args:
      llm: Gateway-backed reasoning LLM wrapper (reason/synthesis steps).
      toolbox: ToolBox or compatible object exposing `call`, `list_tools`,
        `allowed_names`, and `write_tools`.
      config: Chapter 3 configuration.
      tracer: Optional tracer for span correlation.
      trace_id: Trace id for span correlation.
    '''
    self._llm = llm
    self._toolbox = toolbox
    self._config = config
    self._tracer = tracer
    self._trace_id = trace_id

  def execute(
    self,
    decision: Decision,
    percept: Percept,
    memory: Optional[MemoryContext],
    allow_writes: bool,
    code_runner_name: Optional[str] = None,
    audit_log: Optional[List[ToolExecutionAudit]] = None,
  ) -> ActionResult:
    '''Execute the decision's plan.

    Args:
      decision: Decision-layer output.
      percept: Perception-layer output.
      memory: Optional memory context (code plan facts).
      allow_writes: Whether writes are permitted for this run.
      code_runner_name: Optional runner override ('restricted'/'subprocess').
      audit_log: Mutable list the toolbox appends audits to.

    Returns:
      ActionResult; never raises.
    '''
    started = time.perf_counter()
    plan = decision.plan
    adaptations: List[AdaptationRecord] = []

    try:
      if plan.source == 'code':
        action = self._execute_code(
          plan, percept, memory, allow_writes,
          code_runner_name, adaptations,
        )
      else:
        action = self._execute_dag(
          plan, decision, allow_writes, adaptations,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      log_event(
        logger, 40, 'action_crashed',
        session_id=percept.session_id, error=str(exc),
      )
      action = ActionResult(
        goal=plan.goal,
        status='error',
        errors=[f'action layer crashed: {exc}'],
      )

    action.goal = plan.goal
    action.adaptations = adaptations
    action.tool_audits = list(audit_log or [])
    action.duration_ms = (time.perf_counter() - started) * 1000.0
    action.usage = self._llm.usage_snapshot()
    log_event(
      logger, 20, 'action_done',
      session_id=percept.session_id,
      source=plan.source,
      status=action.status,
      steps=len(action.steps),
      adaptations=len(adaptations),
    )
    return action

  # ------------------------------------------------------------------
  # Code plans
  # ------------------------------------------------------------------

  def _execute_code(
    self,
    plan: CognitivePlan,
    percept: Percept,
    memory: Optional[MemoryContext],
    allow_writes: bool,
    code_runner_name: Optional[str],
    adaptations: List[AdaptationRecord],
  ) -> ActionResult:
    '''Run an agent-written code plan in the configured sandbox.

    Args:
      plan: Code plan.
      percept: Perception-layer output.
      memory: Optional memory context.
      allow_writes: Whether writes are permitted.
      code_runner_name: Optional runner override.
      adaptations: Mutable adaptation log.

    Returns:
      ActionResult with the code plan's answer.
    '''
    runner_name = (
      code_runner_name
      or self._config.default_code_runner
    )
    runner = build_code_runner(runner_name, self._config)
    deadline = Deadline(self._config.code_timeout_seconds)
    caller = ToolCaller(
      invoker=self._toolbox.call,
      allowed_tools=self._toolbox.allowed_names(),
      write_tools=self._toolbox.write_tools(),
      deadline=deadline,
      call_budget=self._config.code_call_budget,
      allow_writes=allow_writes,
    )
    facts: Dict[str, Any] = {}
    if memory is not None:
      facts.update(memory.facts)
    facts['perception'] = {
      'domain': percept.signals.domain,
      'entities': percept.entities,
    }

    result = runner.run(
      code=plan.code,
      entrypoint=plan.entrypoint,
      context={'task': percept.task, 'facts': facts, 'tools': caller},
      timeout_seconds=self._config.code_timeout_seconds,
      call_budget=self._config.code_call_budget,
    )
    error_class = classify_code_error(result)
    if error_class != 'none':
      adaptations.append(
        AdaptationRecord(
          kind='fallback',
          step_id='run_code',
          reason=error_class,
          detail=result.error or 'code plan failed',
        )
      )

    output = result.output or {}
    answer = str(output.get('answer', '')).strip()
    step = StepResult(
      step_id='run_code',
      status='completed' if result.ok and answer else 'failed',
      output=answer or (result.error or ''),
      tool_used='code',
      attempts=[
        StepAttempt(
          attempt=1,
          tool='code',
          ok=result.ok,
          error_class=error_class,
          error=result.error,
          observation_preview=answer[:500],
          latency_ms=result.latency_ms,
        )
      ],
      errors=[] if result.ok else [result.error or 'code plan failed'],
      duration_ms=result.latency_ms,
    )
    status = 'completed'
    if not result.ok or not answer:
      status = 'blocked' if error_class == 'blocked' else 'error'

    return ActionResult(
      goal=plan.goal,
      steps=[step],
      status=status,
      answer=answer,
      code_runner=runner.name,
      errors=[] if status == 'completed' else step.errors,
    )

  # ------------------------------------------------------------------
  # DAG plans
  # ------------------------------------------------------------------

  def _execute_dag(
    self,
    plan: CognitivePlan,
    decision: Decision,
    allow_writes: bool,
    adaptations: List[AdaptationRecord],
  ) -> ActionResult:
    '''Execute a DAG (or fallback) plan wave by wave.

    Args:
      plan: DAG/fallback plan.
      decision: Decision (strategy profile carries the retry policy).
      allow_writes: Whether writes are permitted.
      adaptations: Mutable adaptation log.

    Returns:
      ActionResult with per-step results and a synthesized answer.
    '''
    profile = profile_for(decision.strategy)
    loop = AdaptiveLoop(profile, self._config)
    step_map = plan.step_map()
    results: Dict[str, StepResult] = {}
    ordered: List[StepResult] = []

    try:
      waves = topological_waves(plan)
    except ValueError as exc:
      return ActionResult(
        goal=plan.goal, status='error', errors=[str(exc)],
      )

    for wave in waves:
      for step_id in wave:
        step = step_map[step_id]
        failed_deps = [
          dep for dep in step.depends_on
          if results.get(dep) is None
          or results[dep].status != 'completed'
        ]
        if failed_deps:
          result = StepResult(
            step_id=step_id,
            status='skipped',
            skipped_reason=(
              'dependencies not completed: ' + ', '.join(failed_deps)
            ),
          )
        else:
          context = self._dependency_context(step, results)
          result = loop.run_step(
            step,
            attempt_fn=self._make_attempt_fn(context, allow_writes),
            adaptations=adaptations,
          )
        results[step_id] = result
        ordered.append(result)

    status = self._aggregate_status(ordered)
    errors = [
      f'{result.step_id}: {error}'
      for result in ordered for error in result.errors
    ]
    answer, synthesis_errors = self._synthesize(
      plan.goal, ordered, errors,
    )
    errors.extend(synthesis_errors)

    return ActionResult(
      goal=plan.goal,
      steps=ordered,
      status=status,
      answer=answer,
      errors=errors,
    )

  def _make_attempt_fn(
    self,
    context: str,
    allow_writes: bool,
  ) -> Any:
    '''Build the per-attempt callable for one step.

    Args:
      context: Dependency context text.
      allow_writes: Whether writes are permitted.

    Returns:
      Callable `(step, tool, args, attempt) -> StepAttempt`.
    '''
    def attempt_fn(
      active_step: PlanStep,
      tool: str,
      args: Dict[str, object],
      attempt: int,
    ) -> StepAttempt:
      '''Perform one attempt of the active step.

      Args:
        active_step: Step being executed.
        tool: Tool name for this attempt.
        args: Arguments for this attempt.
        attempt: 1-based attempt number.

      Returns:
        StepAttempt.
      '''
      if active_step.kind == 'reason' or not tool:
        return self._attempt_reason(active_step, context, attempt)
      return self._attempt_tool(
        tool, dict(args), attempt, allow_writes,
      )

    return attempt_fn

  def _attempt_tool(
    self,
    tool: str,
    args: Dict[str, Any],
    attempt: int,
    allow_writes: bool,
  ) -> StepAttempt:
    '''Call one tool through the gate.

    Args:
      tool: Qualified tool name.
      args: Tool arguments.
      attempt: Attempt number.
      allow_writes: Whether writes are permitted.

    Returns:
      StepAttempt with classified error.
    '''
    del allow_writes
    started = time.perf_counter()
    result = self._toolbox.call(tool, args)
    error_class = classify_tool_error(result)
    preview = sanitize_untrusted(result.text or '', 500)
    ok = bool(result.ok and (result.text or '').strip())
    return StepAttempt(
      attempt=attempt,
      tool=tool,
      ok=ok,
      error_class=error_class,
      error=result.error,
      observation_preview=preview,
      latency_ms=result.latency_ms or (
        (time.perf_counter() - started) * 1000.0
      ),
    )

  def _attempt_reason(
    self,
    step: PlanStep,
    context: str,
    attempt: int,
  ) -> StepAttempt:
    '''Run a reasoning step through the gateway.

    Args:
      step: Reasoning step.
      context: Dependency context text.
      attempt: Attempt number.

    Returns:
      StepAttempt.
    '''
    started = time.perf_counter()
    prompt = REASON_PROMPT.render(
      goal='complete the plan',
      objective=step.objective,
      expected=step.expected_output or '(not specified)',
      context=context or '(no prior evidence)',
    )
    result = self._llm.complete(
      prompt,
      system=REASON_SYSTEM,
      temperature=self._config.reason_temperature,
      label=f'reason.{step.id}',
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    if result is None:
      return StepAttempt(
        attempt=attempt,
        tool='',
        ok=False,
        error_class='gateway_error',
        error=self._llm.last_error or 'reasoning call failed',
        latency_ms=latency_ms,
      )
    content = result.content.strip()
    return StepAttempt(
      attempt=attempt,
      tool='',
      ok=bool(content),
      error_class='none' if content else 'empty_output',
      observation_preview=sanitize_untrusted(content, 2000),
      latency_ms=latency_ms,
    )

  # ------------------------------------------------------------------
  # Synthesis and aggregation
  # ------------------------------------------------------------------

  def _synthesize(
    self,
    goal: str,
    results: List[StepResult],
    errors: List[str],
  ) -> tuple[str, List[str]]:
    '''Fold step outputs into one answer.

    Args:
      goal: Plan goal.
      results: Per-step results.
      errors: Aggregate errors so far.

    Returns:
      (answer, errors) pair. Falls back to a deterministic join when the
      synthesis call fails.
    '''
    completed = [
      result for result in results if result.status == 'completed'
    ]
    if not completed:
      return (
        'No plan step completed; the task could not be answered from '
        'available tools.',
        [],
      )
    data_like = any(
      result.output.strip().startswith(('{', '['))
      for result in completed
    )
    if len(completed) == 1 and len(results) == 1 and not data_like:
      return completed[0].output, []

    step_text = '\n'.join(
      f'- [{result.step_id}] {result.output[:500]}'
      for result in completed
    )
    unresolved: List[str] = []
    for result in results:
      if result.status == 'completed':
        continue
      detail = result.skipped_reason or '; '.join(result.errors)
      unresolved.append(f'{result.step_id}: {detail}')
    prompt = SYNTHESIS_PROMPT.render(
      goal=goal,
      steps=step_text,
      errors='\n'.join(unresolved + errors) or '(none)',
    )
    result = self._llm.complete(
      prompt,
      system=SYNTHESIS_SYSTEM,
      temperature=self._config.reason_temperature,
      label='synthesis',
    )
    if result is not None and result.content.strip():
      return result.content.strip(), []

    fallback = '; '.join(
      item.output for item in completed if item.output
    ) or 'Plan steps completed without textual output.'
    return fallback, ['synthesis call failed; used deterministic join']

  @staticmethod
  def _dependency_context(
    step: PlanStep,
    results: Dict[str, StepResult],
  ) -> str:
    '''Render dependency outputs as context.

    Args:
      step: Step being executed.
      results: Completed step results.

    Returns:
      Context text.
    '''
    lines: List[str] = []
    for dependency in step.depends_on:
      result = results.get(dependency)
      if result is not None and result.output:
        lines.append(f'[{dependency}] {result.output[:500]}')
    return '\n'.join(lines)

  @staticmethod
  def _aggregate_status(results: List[StepResult]) -> str:
    '''Map step outcomes to an aggregate run status.

    Args:
      results: All step results.

    Returns:
      'completed', 'partial', 'blocked', or 'error'.
    '''
    if not results:
      return 'error'
    completed = sum(1 for result in results if result.status == 'completed')
    if completed == len(results):
      return 'completed'
    if completed > 0:
      return 'partial'
    blocked = all(
      any(attempt.error_class == 'blocked' for attempt in result.attempts)
      for result in results
    )
    return 'blocked' if blocked else 'error'


def compact_context(payload: Dict[str, Any], max_chars: int = 800) -> str:
  '''Serialize context compactly (helper for logs/tests).

  Args:
    payload: JSON-serializable mapping.
    max_chars: Maximum length.

  Returns:
    JSON string.
  '''
  return compact_json(payload, max_chars)
