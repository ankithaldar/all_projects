#!/usr/bin/env python
# -- coding: utf-8 --

'''Tool provider interface and the safety gate decorator.

Chapter 2 must be agnostic about *where* tools live. The `ToolProvider`
interface hides that decision behind three methods (`list_tools`,
`execute`, `close`), so the reasoning patterns never know whether a call
went to an in-process Chapter 1 server core or over MCP stdio to a
subprocess (or, later, to a remote HTTP server).

`GatedToolProvider` is a decorator that adds the safety and hygiene layer
required before any model-proposed call reaches a real system:

  - unknown tools are refused without touching the backend,
  - state-changing tools are blocked unless the run explicitly allows
    writes (least privilege per run),
  - optional human/operator approval callback for writes,
  - every observation is sanitized (control chars, secrets heuristics,
    length cap) and wrapped in explicit untrusted-data markers,
  - every call produces a `ToolExecutionAudit` for logs, traces, and
    persistence.

This decorator pattern keeps policy out of the tool implementations and out
of the reasoning loops: one place to audit, one place to harden.
'''


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from agentic_common.logging import get_logger, log_event
from agentic_common.security import sanitize_untrusted
from agentic_common.tracing import Tracer
from chapter02_planning.schemas import (
  ToolExecutionAudit,
  ToolResult,
  ToolSpec,
)

logger = get_logger(__name__)

UntrustedWrapper = Callable[[str], str]
AuditSink = Callable[[ToolExecutionAudit], None]
Approver = Callable[[str, Dict[str, Any]], bool]

UNTRUSTED_OPEN = '[BEGIN UNTRUSTED TOOL DATA]'
UNTRUSTED_CLOSE = '[END UNTRUSTED TOOL DATA]'


def wrap_untrusted(text: str) -> str:
  '''Wrap tool output in explicit untrusted-data markers.

  Args:
    text: Sanitized tool output.

  Returns:
    Wrapped text with delimiters.
  '''
  return f'{UNTRUSTED_OPEN}\n{text}\n{UNTRUSTED_CLOSE}'


def _blocked(tool: str, args: Dict[str, Any], reason: str) -> ToolResult:
  '''Build a blocked ToolResult.

  Args:
    tool: Tool name.
    args: Proposed arguments.
    reason: Human-readable block reason.

  Returns:
    ToolResult with ok=False.
  '''
  return ToolResult(
    tool=tool,
    args=args,
    ok=False,
    text=f'BLOCKED by safety gate: {reason}',
    error=reason,
  )


class ToolProvider(ABC):
  '''Abstract source of tools: specs in, observations out.

  Implementations must be safe to call after `close()`; they should degrade
  to empty tool lists and failed results rather than raising.
  '''

  name: str = 'provider'

  @abstractmethod
  def list_tools(self) -> List[ToolSpec]:
    '''List available tools.

    Returns:
      Tool specs, qualified with a provider-specific namespace.
    '''

  @abstractmethod
  def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
    '''Execute one tool call.

    Args:
      tool_name: Qualified tool name from `list_tools`.
      arguments: Model-proposed arguments (untrusted).

    Returns:
      Normalized ToolResult; transport failures must not raise.
    '''

  @abstractmethod
  def close(self) -> None:
    '''Release any resources held by the provider.'''

  def get(self, tool_name: str) -> Optional[ToolSpec]:
    '''Look up one tool spec by name.

    Args:
      tool_name: Qualified tool name.

    Returns:
      Tool spec or None.
    '''
    for tool in self.list_tools():
      if tool.name == tool_name:
        return tool
    return None


class GatedToolProvider(ToolProvider):
  '''Safety/hygiene decorator around any ToolProvider.'''

  def __init__(
    self,
    inner: ToolProvider,
    allow_writes: bool = False,
    require_approval: bool = True,
    approver: Optional[Approver] = None,
    max_result_chars: int = 4000,
    sanitize: bool = True,
    audit_sink: Optional[AuditSink] = None,
    tracer: Optional[Tracer] = None,
    trace_id: str = '',
  ) -> None:
    '''Initialize the gate.

    Args:
      inner: Wrapped provider.
      allow_writes: Whether state-changing tools may run in this run.
      require_approval: Whether writes additionally need `approver`.
      approver: Optional callback `(tool, args) -> bool`.
      max_result_chars: Observation length cap.
      sanitize: Whether to sanitize and wrap observations.
      audit_sink: Optional callback receiving each audit record.
      tracer: Optional tracer for `tool.call` spans.
      trace_id: Trace id for span correlation.
    '''
    self._inner = inner
    self._allow_writes = allow_writes
    self._require_approval = require_approval
    self._approver = approver
    self._max_result_chars = max(200, int(max_result_chars))
    self._sanitize = sanitize
    self._audit_sink = audit_sink
    self._tracer = tracer
    self._trace_id = trace_id
    self.name = inner.name
    self.last_audit: Optional[ToolExecutionAudit] = None

  def list_tools(self) -> List[ToolSpec]:  # noqa: D102
    return self._inner.list_tools()

  def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
    '''Gate, execute, sanitize, and audit one tool call.

    Args:
      tool_name: Qualified tool name.
      arguments: Model-proposed arguments (untrusted).

    Returns:
      Normalized ToolResult with sanitized observation text.
    '''
    safe_args = arguments if isinstance(arguments, dict) else {}
    spec = self._inner.get(tool_name)

    if spec is None:
      result = _blocked(tool_name, safe_args, f'unknown tool {tool_name!r}')
      self._record(result, spec=None)
      return result

    if not spec.read_only and not self._allow_writes:
      result = _blocked(
        tool_name, safe_args,
        'state-changing tools are disabled for this run '
        '(set allow_writes=True to enable)',
      )
      self._record(result, spec)
      return result

    if (
      not spec.read_only
      and self._require_approval
      and self._approver is not None
      and not self._approver(tool_name, safe_args)
    ):
      result = _blocked(
        tool_name, safe_args, 'write call was not approved',
      )
      self._record(result, spec)
      return result

    result = self._inner.execute(tool_name, safe_args)
    if self._sanitize and result.text:
      result.text = wrap_untrusted(
        sanitize_untrusted(result.text, self._max_result_chars)
      )
    elif len(result.text) > self._max_result_chars:
      result.text = result.text[: self._max_result_chars - 3] + '...'
      result.truncated = True

    self._record(result, spec)
    return result

  def close(self) -> None:  # noqa: D102
    self._inner.close()

  # ------------------------------------------------------------------
  # Internals
  # ------------------------------------------------------------------

  def _record(
    self,
    result: ToolResult,
    spec: Optional[ToolSpec],
  ) -> None:
    '''Build and publish the audit record for one call.

    Args:
      result: The tool result (possibly blocked).
      spec: The tool spec when known.
    '''
    blocked = (
      not result.ok and result.error is not None
      and result.text.startswith('BLOCKED by safety gate')
    )
    audit = ToolExecutionAudit(
      tool=result.tool,
      args=result.args,
      ok=result.ok,
      approved=not blocked,
      blocked=blocked,
      error=result.error,
      latency_ms=result.latency_ms,
      result_preview=sanitize_untrusted(result.text, 200),
    )
    self.last_audit = audit

    log_event(
      logger, 20, 'tool_call',
      tool=result.tool,
      read_only=spec.read_only if spec else None,
      ok=result.ok,
      blocked=blocked,
      latency_ms=round(result.latency_ms, 2),
      error=result.error,
    )

    if self._audit_sink is not None:
      try:
        self._audit_sink(audit)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(
          'audit sink failed',
          extra={'extra_fields': {'error': str(exc)}},
        )

    if self._tracer is not None and self._trace_id:
      with self._tracer.span(
        self._trace_id,
        'tool.call',
        tool=result.tool,
        blocked=blocked,
      ) as span:
        span.set_attr('ok', result.ok)
        span.set_attr('latency_ms', round(result.latency_ms, 1))
