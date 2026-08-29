#!/usr/bin/env python
# -- coding: utf-8 --

'''The agentic tool-use loop (Chapter 1 agent).

Pattern: TOOL-USE LOOP (a.k.a. ReAct-style act/observe loop)

       ┌──────────────────────────────────────────────────────┐
       │                                                      │
       ▼                                                      │
   ┌───────┐  tool_calls   ┌─────────────┐  results   ┌────┴─────┐
   │  LLM  │ ─────────────▶│ MCP Toolbox │───────────▶│ messages │
   │gateway│ ◀──────────── │ (+ policy)  │            └──────────┘
   └───────┘               └─────────────┘
       │        no tool_calls → final answer → DONE
       └── each iteration = one gateway completion

Exit conditions: final answer, iteration budget, or token budget.
Every hop (LLM call, tool call) is logged, traced, audited, persisted.
'''


from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from agentic_common.logging import log_event
from agentic_common.persistence import AgentStore
from agentic_common.settings import Settings, default_settings
from agentic_common.tracing import TokenUsage, Tracer
from chapter01_mcp.agent.orchestrator import (
  MCPSessionManager,
  ops_policy_engine,
)
from chapter01_mcp.agent.tools_bridge import MCPToolbox
from chapter01_mcp.client.registry import (
  ServerCatalog,
  pick_servers,
  pick_tools,
)
from chapter01_mcp.logging_util import get_mcp_logger
from chapter01_mcp.schemas import (
  AgentTaskInput,
  AgentTaskOutput,
  ToolCallAudit,
)
from llm_gateway.schemas import ChatMessage, GatewayRequest

logger = get_mcp_logger(__name__)


SYSTEM_PROMPT = (
  'You are an operations decision agent for a retail and telecom company. '
  'Use the provided MCP tools to gather facts before deciding. Prefer read '
  'tools first; call write tools only when the task requires action and '
  'stays within policy. Summarize evidence with concrete numbers, then give '
  'a short, decisive recommendation.'
)


class OpsAgent:
  '''Tool-use-loop agent over MCP tools and the user's LLM gateway.'''

  def __init__(
    self,
    llm: Any,
    settings: Optional[Settings] = None,
    store: Optional[AgentStore] = None,
    tracer: Optional[Tracer] = None,
  ) -> None:
    '''Initialize the agent.

    Args:
      llm: GatewayClient or MockGateway (same complete() interface).
      settings: Runtime settings.
      store: Optional persistence for execution history.
      tracer: Optional tracer.
    '''
    self._llm = llm
    self._settings = settings or default_settings()
    self._store = store
    self._tracer = tracer
    self._sessions: Optional[MCPSessionManager] = None

  def run_task(self, task_input: AgentTaskInput) -> AgentTaskOutput:
    '''Run one task end-to-end.

    Args:
      task_input: Validated task input.

    Returns:
      Validated AgentTaskOutput.
    '''
    started = time.perf_counter()
    trace_id = self._new_trace_id()

    if self._store is not None:
      try:
        self._store.ensure_session(task_input.session_id)
        self._store.log_event(
          task_input.session_id, 'task_start',
          {'task': task_input.task[:500]},
        )
      except Exception as exc:  # pylint: disable=broad-exception-caught
        # Persistence failures must not break the agent run.
        logger.warning(
          'persist failed',
          extra={'extra_fields': {'error': str(exc)}},
        )

    answer = ''
    status = 'completed'
    audits: List[ToolCallAudit] = []
    errors: List[str] = []
    usage_total = TokenUsage()
    iterations = 0

    toolbox: Optional[Any] = None
    try:
      toolbox = self._open_toolbox(task_input.task, trace_id)

      if toolbox is None:
        return self._finalize(
          session_id=task_input.session_id,
          answer=(
            'No MCP servers could be reached; the task was aborted safely.'
          ),
          status='error',
          iterations=0,
          audits=[],
          usage=usage_total,
          errors=['all MCP servers failed to start'],
          duration_ms=(time.perf_counter() - started) * 1000.0,
        )

      (answer, usage_total, audits, iterations, status, errors) = self._loop(
        task_input, toolbox, task_input.session_id, trace_id,
      )

    except Exception as exc:  # pylint: disable=broad-exception-caught
      # Last-resort guard: a task must always return a structured output.
      status = 'error'
      errors.append(f'agent crashed: {exc}')
      log_event(
        logger, 40, 'task_crashed',
        session_id=task_input.session_id, error=str(exc),
      )
    finally:
      self._close_toolbox()

    return self._finalize(
      session_id=task_input.session_id,
      answer=answer,
      status=status,
      iterations=iterations,
      audits=audits,
      usage=usage_total,
      errors=errors,
      duration_ms=(time.perf_counter() - started) * 1000.0,
    )

  # ------------------------------------------------------------------
  # Internals
  # ------------------------------------------------------------------

  def _new_trace_id(self) -> str:
    '''Generate a trace id.

    Returns:
      Trace id string.
    '''
    if self._tracer is not None:
      return self._tracer.new_trace_id()
    return uuid.uuid4().hex

  def _open_toolbox(
    self,
    task: str,
    trace_id: str,
  ) -> Optional[Any]:
    '''Discovery -> sessions -> toolbox construction.

    Args:
      task: Task text.
      trace_id: Trace id.

    Returns:
      MCPToolbox or None when no server is reachable.
    '''
    catalog = ServerCatalog()
    selected = pick_servers(task, catalog, max_servers=2)
    log_event(
      logger, 20, 'server_selection',
      trace_id=trace_id,
      picked=[d.name for d in selected],
    )

    sessions = MCPSessionManager(
      catalog=catalog,
      selected=[d.name for d in selected],
      timeout_seconds=self._settings.tool_timeout_seconds,
    )
    sessions.open_all()

    if not sessions.clients:
      return None

    filtered_tools: Dict[str, List[Any]] = {}
    for name, tools in sessions.tools_by_server.items():
      filtered_tools[name] = pick_tools(task, tools, max_tools=6)

    self._sessions = sessions
    return MCPToolbox(
      clients=sessions.clients,
      tools_by_server=filtered_tools,
      policy_engine=ops_policy_engine(self._settings),
      store=self._store,
      tracer=self._tracer,
      trace_id=trace_id,
    )

  def _close_toolbox(self) -> None:
    '''Close MCP sessions if any are open.'''
    if self._sessions is not None:
      self._sessions.close()
      self._sessions = None

  def _loop(
    self,
    task_input: AgentTaskInput,
    toolbox: Any,
    session_id: str,
    trace_id: str,
  ) -> tuple:
    '''The core tool-use loop.

    Args:
      task_input: Validated task input.
      toolbox: Connected MCPToolbox.
      session_id: Session id.
      trace_id: Trace id.

    Returns:
      (answer, usage, audits, iterations, status, errors)
    '''
    messages: List[Dict[str, Any]] = [
      {'role': 'system', 'content': SYSTEM_PROMPT},
      {'role': 'user', 'content': task_input.task},
    ]

    gw_tools = toolbox.to_gateway_tools()
    usage_total = TokenUsage()
    audits: List[ToolCallAudit] = []
    errors: List[str] = []

    for iteration in range(1, task_input.max_iterations + 1):
      response = self._safe_complete(messages, gw_tools, session_id, trace_id)

      if response is None:
        errors.append('llm gateway unavailable')
        return '', usage_total, audits, iteration, 'error', errors

      usage_total = usage_total.add(
        TokenUsage(
          input_tokens=response.usage.input_tokens,
          output_tokens=response.usage.output_tokens,
          total_tokens=response.usage.total_tokens,
        )
      )

      tool_calls = list(response.tool_calls or [])
      if not tool_calls:
        return (
          response.content, usage_total, audits, iteration, 'completed', errors
        )

      messages.append(
        {
          'role': 'assistant',
          'content': response.content or '',
          'tool_calls': [call.model_dump(mode='json') for call in tool_calls],
        }
      )

      for call in tool_calls:
        try:
          arguments = json.loads(call.function.arguments or '{}')
        except json.JSONDecodeError:
          arguments = {}
          errors.append(f'malformed arguments for {call.function.name}')

        _, text, audit = toolbox.execute(
          call.function.name, arguments, task_input.session_id,
        )
        audits.append(audit)

        messages.append(
          {
            'role': 'tool',
            'tool_call_id': call.id or call.function.name,
            'name': call.function.name,
            'content': text,
          }
        )

      budget = self._settings.agent_token_budget
      if usage_total.total_tokens > budget:
        errors.append('token budget exhausted')
        return '', usage_total, audits, iteration, 'error', errors

    max_iters = task_input.max_iterations
    return (
      '', usage_total, audits, max_iters, 'max_iterations', errors
    )

  def _safe_complete(
    self,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    session_id: str,
    trace_id: str,
  ) -> Optional[Any]:
    '''One gateway completion with tracing and safe failure handling.

    Args:
      messages: Conversation so far.
      tools: Gateway tool definitions.
      session_id: Session id.
      trace_id: Trace id.

    Returns:
      GatewayResponse, or None when the gateway failed.
    '''
    chat_messages = [
      ChatMessage.model_validate(message) for message in messages
    ]

    request = GatewayRequest(
      messages=chat_messages,
      tools=tools,
      tool_choice='auto',
      temperature=self._settings.llm_temperature,
      max_tokens=self._settings.llm_max_tokens,
      session_id=session_id,
      metadata={'chapter': '01_mcp', 'trace_id': trace_id},
    )

    try:
      response = self._llm.complete(request)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      # Gateway/provider failures of any shape are normalized to None.
      log_event(
        logger,
        40,
        'llm_call_failed',
        session_id=session_id,
        error=str(exc),
      )
      return None

    log_event(
      logger,
      20,
      'llm_response',
      session_id=session_id,
      provider=response.provider,
      model=response.model,
      tool_calls=len(response.tool_calls),
      input_tokens=response.usage.input_tokens,
      output_tokens=response.usage.output_tokens,
      latency_ms=round(response.latency_ms, 1),
    )
    return response

  def _finalize(
    self,
    session_id: str,
    answer: str,
    status: str,
    iterations: int,
    audits: List[ToolCallAudit],
    usage: TokenUsage,
    errors: List[str],
    duration_ms: float,
  ) -> AgentTaskOutput:
    '''Assemble, persist, and log the final output.

    Args:
      session_id: Session id.
      answer: Final answer.
      status: Run status.
      iterations: Iterations used.
      audits: Tool audits.
      usage: Total usage.
      errors: Errors.
      duration_ms: Wall duration in ms.

    Returns:
      AgentTaskOutput.
    '''
    if self._store is not None:
      try:
        self._store.log_event(
          session_id,
          'task_finish',
          {
            'status': status,
            'iterations': iterations,
            'usage': usage.model_dump(),
            'errors': errors,
            'answer_preview': (answer or '')[:300],
            'tool_calls': [audit.model_dump() for audit in audits],
            'duration_ms': round(duration_ms, 1),
          },
        )
      except Exception as exc:  # pylint: disable=broad-exception-caught
        # Persistence failures must not break the agent run.
        logger.warning(
          'persist failed',
          extra={'extra_fields': {'error': str(exc)}},
        )

    log_event(
      logger,
      20,
      'task_done',
      session_id=session_id,
      status=status,
      iterations=iterations,
      total_tokens=usage.total_tokens,
      error_count=len(errors),
    )

    return AgentTaskOutput(
      session_id=session_id,
      answer=answer,
      status=status,
      iterations=iterations,
      tool_calls=audits,
      usage=usage.model_dump(),
      errors=errors,
    )
