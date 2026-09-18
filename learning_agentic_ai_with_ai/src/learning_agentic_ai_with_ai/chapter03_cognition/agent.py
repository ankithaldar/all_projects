#!/usr/bin/env python
# -- coding: utf-8 --

'''CognitiveAgent: the 4-layer pipeline and its adaptive outer loop.

    task
      │
      ▼
  ┌─────────────┐   Percept      ┌─────────────┐  MemoryContext
  │  PERCEPTION │ ─────────────▶ │   MEMORY    │ ────────────────┐
  └─────────────┘                └─────────────┘                 │
                                                                 ▼
                                                        ┌─────────────────┐
                                                        │    DECISION     │
                                                        │ strategy select │
                                                        │ plan + introspect│
                                                        └────────┬────────┘
                                                                 │ Decision
                                                                 ▼
                                                        ┌─────────────────┐
                                                        │     ACTION      │
                                                        │ DAG/code exec   │
                                                        │ retry/switch    │
                                                        └────────┬────────┘
                                                                 │ ActionResult
                                                                 ▼
                          reflection: episode + strategy stats + facts
                          (memory improves the NEXT run's planning)

Two nested adaptive loops:

  - **inner** (ActionLayer/AdaptiveLoop): bounded retry and tool switching
    inside one plan execution.
  - **outer** (this class): if a plan fails and the strategy allows it,
    escalate the strategy (conservative -> exploratory -> fallback),
    re-plan once per escalation, and stop when the ladder ends or the
    replan budget is exhausted. Every escalation is recorded.

Fail-safe: the agent always returns a typed `CognitiveTaskOutput`; crashes,
gateway outages, budget exhaustion, and sandbox refusals degrade to a
status plus errors, never an exception to the caller.
'''


from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from agentic_common.logging import get_logger, log_event
from agentic_common.persistence import AgentStore
from agentic_common.settings import Settings, default_settings
from agentic_common.tracing import Tracer
from chapter03_cognition.action import ActionLayer
from chapter03_cognition.config import CognitionConfig, load_cognition_config
from chapter03_cognition.llm import ReasoningLLM, usage_delta
from chapter03_cognition.memory import MemoryLayer, MemoryStore
from chapter03_cognition.perception import PerceptionLayer
from chapter03_cognition.planner import DecisionLayer
from chapter03_cognition.prompts import (
  FALLBACK_PROMPT,
  FALLBACK_SYSTEM,
  compact_json,
  format_memory_notes,
)
from chapter03_cognition.schemas import (
  ActionResult,
  AdaptationRecord,
  CognitiveTaskInput,
  CognitiveTaskOutput,
  Decision,
  Episode,
  MemoryContext,
  Percept,
  StrategyName,
  ToolExecutionAudit,
)
from chapter03_cognition.strategies import (
  StrategySelector,
  profile_for,
)

logger = get_logger(__name__)

ESCALATION_LADDER: Dict[StrategyName, StrategyName] = {
  'conservative': 'exploratory',
  'exploratory': 'fallback',
  'fallback': 'fallback',
}


class CognitiveAgent:
  '''Runs the 4-layer cognitive pipeline with adaptive escalation.'''

  def __init__(
    self,
    llm: Any,
    toolbox: Any,
    config: Optional[CognitionConfig] = None,
    settings: Optional[Settings] = None,
    store: Optional[AgentStore] = None,
    memory: Optional[MemoryStore] = None,
    tracer: Optional[Tracer] = None,
  ) -> None:
    '''Initialize the agent.

    Args:
      llm: GatewayClient or MockGateway (same `complete()` interface).
      toolbox: ToolBox or FaultInjector (exposes call/list_tools/...).
      config: Chapter 3 configuration; loaded from env when omitted.
      settings: Optional shared settings seeding the config.
      store: Optional shared execution-history store.
      memory: Optional cognitive memory store.
      tracer: Optional tracer for spans.
    '''
    self._gateway = llm
    self._toolbox = toolbox
    self._settings = settings or default_settings()
    self._config = config or load_cognition_config(self._settings)
    self._store = store
    self._memory_store = memory
    self._tracer = tracer
    self._perception = PerceptionLayer()
    self._selector = StrategySelector()
    self._audits: List[ToolExecutionAudit] = []
    self._current_session_id = ''

  def attach_toolbox(self, toolbox: Any) -> None:
    '''Attach the tool backend after construction.

    Useful when the toolbox needs the agent's `audit_sink` at build time.

    Args:
      toolbox: ToolBox or compatible object.
    '''
    self._toolbox = toolbox

  def close(self) -> None:
    '''Release the tool backend (memory store is owned by the caller).'''
    if self._toolbox is not None:
      try:
        self._toolbox.close()
      except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(
          'toolbox close failed',
          extra={'extra_fields': {'error': str(exc)}},
        )

  # ------------------------------------------------------------------
  # Public entry point
  # ------------------------------------------------------------------

  def run_task(self, task_input: CognitiveTaskInput) -> CognitiveTaskOutput:
    '''Run one task through all four layers.

    Args:
      task_input: Validated task input.

    Returns:
      CognitiveTaskOutput; guaranteed to return.
    '''
    started = time.perf_counter()
    trace_id = (
      self._tracer.new_trace_id() if self._tracer is not None
      else uuid.uuid4().hex
    )
    self._audits = []
    self._current_session_id = task_input.session_id
    self._persist_start(task_input)

    try:
      output = self._run_inner(task_input, trace_id)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      log_event(
        logger, 40, 'task_crashed',
        session_id=task_input.session_id, error=str(exc),
      )
      output = CognitiveTaskOutput(
        session_id=task_input.session_id,
        answer='',
        status='error',
        errors=[f'cognitive agent crashed: {exc}'],
      )

    output.duration_ms = (time.perf_counter() - started) * 1000.0
    self._reflect(task_input, output)
    self._persist_finish(task_input.session_id, output)

    log_event(
      logger, 20, 'task_done',
      session_id=task_input.session_id,
      strategy=output.decision.strategy if output.decision else 'unknown',
      status=output.status,
      adaptations=(
        len(output.action.adaptations) if output.action else 0
      ),
      duration_ms=round(output.duration_ms, 1),
    )
    return output

  # ------------------------------------------------------------------
  # Pipeline
  # ------------------------------------------------------------------

  def _run_inner(
    self,
    task_input: CognitiveTaskInput,
    trace_id: str,
  ) -> CognitiveTaskOutput:
    '''Execute the four layers plus the outer adaptive loop.

    Args:
      task_input: Validated task input.
      trace_id: Trace id for this run.

    Returns:
      CognitiveTaskOutput without reflection/persistence fields.
    '''
    session_id = task_input.session_id

    percept = self._perception.perceive(task_input.task, session_id)
    memory_context = self._memory_context(percept)
    profile, strategy_reason = self._selector.select(
      percept, memory_context, task_input.strategy_hint,
    )
    self._persist_event(
      session_id, 'percept_and_strategy',
      {
        'domain': percept.signals.domain,
        'risk': percept.signals.risk,
        'capabilities': percept.required_capabilities,
        'strategy': profile.name,
        'strategy_reason': strategy_reason,
      },
    )

    llm = ReasoningLLM(
      self._gateway,
      self._config,
      tracer=self._tracer,
      trace_id=trace_id,
      session_id=session_id,
    )
    set_trace = getattr(self._toolbox, 'set_trace', None)
    if callable(set_trace):
      set_trace(self._tracer, trace_id)
    decision_layer = DecisionLayer(llm, self._config)
    action_layer = ActionLayer(
      llm, self._toolbox, self._config,
      tracer=self._tracer, trace_id=trace_id,
    )

    tools = (
      self._toolbox.list_tools() if self._toolbox is not None else []
    )
    escalations: List[AdaptationRecord] = []
    attempts_allowed = 1 + profile.retry.max_replans
    current_profile = profile
    decision: Optional[Decision] = None
    action: Optional[ActionResult] = None

    for attempt in range(attempts_allowed):
      if llm.budget.exceeded:
        break
      decision = decision_layer.decide(
        percept, memory_context, current_profile, tools,
        plan_mode=task_input.plan_mode,
      )
      action = action_layer.execute(
        decision=decision,
        percept=percept,
        memory=memory_context,
        allow_writes=(
          task_input.allow_writes and current_profile.retry.allow_writes
        ),
        code_runner_name=task_input.code_runner,
        audit_log=self._audits,
      )
      if action.status == 'completed':
        break
      next_name = ESCALATION_LADDER[current_profile.name]
      can_escalate = (
        current_profile.retry.escalate_on_failure
        and next_name != current_profile.name
        and attempt + 1 < attempts_allowed
      )
      if not can_escalate:
        break
      escalations.append(
        AdaptationRecord(
          kind='strategy_escalation',
          attempt=attempt + 1,
          reason=action.status,
          detail=(
            f'escalating {current_profile.name} -> {next_name} '
            f'after {action.status}'
          ),
        )
      )
      current_profile = profile_for(next_name)

    if action is None or decision is None:
      return CognitiveTaskOutput(
        session_id=session_id,
        answer='',
        status='error',
        percept=percept,
        memory=memory_context,
        errors=['planning was skipped: token budget exhausted'],
        usage=llm.usage_snapshot(),
      )

    if escalations:
      action.adaptations = list(action.adaptations) + escalations
    answer = action.answer
    errors = list(action.errors)
    status = action.status

    if not answer:
      answer, degraded_errors = self._degraded_answer(
        llm, percept, memory_context,
      )
      errors.extend(degraded_errors)
      if status == 'completed':
        status = 'needs_review'

    if llm.budget.exceeded:
      errors.append('token budget exhausted')
      if status == 'completed':
        status = 'error'

    return CognitiveTaskOutput(
      session_id=session_id,
      answer=answer,
      status=status,
      percept=percept,
      memory=memory_context,
      decision=decision,
      action=action,
      usage=llm.usage_snapshot(),
      errors=errors,
    )

  # ------------------------------------------------------------------
  # Layer helpers
  # ------------------------------------------------------------------

  def _memory_context(self, percept: Percept) -> MemoryContext:
    '''Build memory context when a memory store exists.

    Args:
      percept: Perception-layer output.

    Returns:
      MemoryContext (empty when no store is configured).
    '''
    if self._memory_store is None:
      return MemoryContext(recall_notes=['memory store not configured'])
    return MemoryLayer(
      self._memory_store, self._config.memory_recall_limit,
    ).build_context(percept)

  def _degraded_answer(
    self,
    llm: ReasoningLLM,
    percept: Percept,
    memory: MemoryContext,
  ) -> Tuple[str, List[str]]:
    '''Produce a caveated fallback answer when execution produced none.

    Args:
      llm: Per-run reasoning LLM.
      percept: Perception-layer output.
      memory: Memory context.

    Returns:
      (answer, errors) pair.
    '''
    prompt = FALLBACK_PROMPT.render(
      task=percept.task,
      facts=compact_json(memory.facts, 800),
      memory=format_memory_notes(memory.recall_notes),
    )
    result = llm.complete(
      prompt,
      system=FALLBACK_SYSTEM,
      temperature=self._config.reason_temperature,
      label='fallback_answer',
    )
    if result is None or not result.content.strip():
      return (
        'Task could not be completed and no fallback answer was '
        'produced. Re-run with tools enabled or provide more context.',
        ['fallback answer call failed'],
      )
    return result.content.strip(), ['answered in degraded mode']

  # ------------------------------------------------------------------
  # Reflection (learning)
  # ------------------------------------------------------------------

  def _reflect(
    self,
    task_input: CognitiveTaskInput,
    output: CognitiveTaskOutput,
  ) -> None:
    '''Write the episode, update strategy stats, and store facts.

    Args:
      task_input: Task input.
      output: Final output.
    '''
    if self._memory_store is None:
      return
    try:
      tags = self._episode_tags(output)
      tool_sequence = [audit.tool for audit in self._audits]
      error_classes = self._error_classes(output)
      adaptations = [
        record.kind
        for record in (
          output.action.adaptations if output.action else []
        )
      ]
      episode = Episode(
        episode_id=uuid.uuid4().hex[:16],
        session_id=task_input.session_id,
        task=task_input.task[:500],
        domain=(
          output.percept.signals.domain if output.percept else 'unknown'
        ),
        strategy=(
          output.decision.strategy if output.decision else 'conservative'
        ),
        status=output.status,
        success=output.status == 'completed',
        answer_preview=output.answer[:300],
        tool_sequence=tool_sequence,
        error_classes=error_classes,
        adaptations=sorted(set(adaptations)),
        duration_ms=output.duration_ms,
        total_tokens=int(output.usage.get('total_tokens', 0) or 0),
        tags=tags,
      )
      self._memory_store.save_episode(episode)
      self._memory_store.record_strategy_result(
        strategy=episode.strategy,
        success=episode.success,
        duration_ms=output.duration_ms,
        tokens=episode.total_tokens,
      )
      self._memory_store.set_fact(
        f'last_answer.{task_input.session_id}',
        {
          'status': output.status,
          'strategy': episode.strategy,
          'answer': output.answer[:300],
        },
        domain=episode.domain,
        tags=tags[:6],
      )
      log_event(
        logger, 20, 'episode_saved',
        episode_id=episode.episode_id,
        success=episode.success,
        strategy=episode.strategy,
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      logger.warning(
        'reflection failed',
        extra={'extra_fields': {'error': str(exc)}},
      )

  @staticmethod
  def _episode_tags(output: CognitiveTaskOutput) -> List[str]:
    '''Build retrieval tags for an episode.

    Args:
      output: Final output.

    Returns:
      Tag list.
    '''
    tags: List[str] = []
    if output.percept is not None:
      tags.append(output.percept.signals.domain)
      for values in output.percept.entities.values():
        tags.extend(values[:4])
      tags.extend(output.percept.signals.keywords[:6])
    return list(dict.fromkeys(tag for tag in tags if tag))

  @staticmethod
  def _error_classes(output: CognitiveTaskOutput) -> List[str]:
    '''Collect distinct error classes from step attempts.

    Args:
      output: Final output.

    Returns:
      Sorted error class list.
    '''
    classes = set()
    if output.action is not None:
      for step in output.action.steps:
        for attempt in step.attempts:
          if attempt.error_class not in ('none',):
            classes.add(attempt.error_class)
    return sorted(classes)

  # ------------------------------------------------------------------
  # Persistence (shared execution history)
  # ------------------------------------------------------------------

  def _persist_start(self, task_input: CognitiveTaskInput) -> None:
    '''Create the session and log task_start.

    Args:
      task_input: Task input.
    '''
    if self._store is None:
      return
    try:
      self._store.ensure_session(
        task_input.session_id, meta={'chapter': '03_cognition'},
      )
      self._store.log_event(
        task_input.session_id, 'task_start',
        {
          'task': task_input.task[:500],
          'strategy_hint': task_input.strategy_hint,
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
    output: CognitiveTaskOutput,
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
        session_id, 'task_finish',
        {
          'status': output.status,
          'strategy': (
            output.decision.strategy if output.decision else 'unknown'
          ),
          'plan_source': (
            output.decision.plan.source if output.decision else 'unknown'
          ),
          'answer_preview': output.answer[:300],
          'usage': output.usage,
          'errors': output.errors,
          'adaptations': (
            [record.model_dump() for record in output.action.adaptations]
            if output.action else []
          ),
          'duration_ms': round(output.duration_ms, 1),
        },
      )
      self._store.remember(
        session_id,
        'last_answer',
        {
          'status': output.status,
          'answer': output.answer[:500],
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

  def audit_sink(self, audit: ToolExecutionAudit) -> None:
    '''Collect and persist one toolbox audit.

    Wire this into the toolbox (`audit_sink=agent.audit_sink`) so tool
    calls appear in execution history and in the final output.

    Args:
      audit: Audit record from the toolbox gate.
    '''
    self._audits.append(audit)
    if self._store is None:
      return
    server, _, tool = audit.tool.partition('.')
    try:
      self._store.log_tool_call(
        session_id=self._current_session_id,
        server=server or 'toolbox',
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


def usage_snapshot_of(llm: ReasoningLLM) -> Dict[str, Any]:
  '''Return a usage snapshot (helper for callers/tests).

  Args:
    llm: Reasoning LLM.

  Returns:
    Usage snapshot dict.
  '''
  return llm.usage_snapshot()


def usage_of_llm(
  before: Any,
  after: Any,
) -> Dict[str, int]:
  '''Compute usage delta between two TokenUsage snapshots.

  Args:
    before: Usage before.
    after: Usage after.

  Returns:
    Delta dict.
  '''
  return usage_delta(before, after)
