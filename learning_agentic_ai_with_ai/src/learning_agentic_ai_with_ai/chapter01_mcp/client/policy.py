#!/usr/bin/env python
# -- coding: utf-8 --

'''Tool-call security policy enforced by the CLIENT (defense at the boundary).

Server-side validation alone is not enough: the agent must also decide
1. are the arguments inside the advertised schema? (untrusted model input)
2. does a WRITE need human approval? (approval callback)
3. is the result payload safe to feed back into the LLM? (truncation)
'''


from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from agentic_common.security import (
  sanitize_untrusted,
  validate_against_json_schema,
)
from chapter01_mcp.logging_util import get_mcp_logger
from chapter01_mcp.schemas import ToolCallResult, ToolDescriptor

logger = get_mcp_logger(__name__)


class ToolPolicyEngine:
  '''Central gate for every agent tool call.

  Checks, in order:
  1. JSON-Schema validation of arguments (against the advertised schema)
  2. write tools require the approval callback -> else blocked with reason
  '''

  def __init__(
    self,
    write_tools: set[str] | None = None,
    max_result_chars: int = 6000,
    approval_callback: Optional[Callable[[Dict[str, Any], str], bool]] = None,
  ) -> None:
    '''Initialize policy.

    Args:
      write_tools: Set of `server.tool` names classified as write tools.
      max_result_chars: Result truncation cap.
      approval_callback: Optional callable(arguments, qualified_name) -> bool.
    '''
    self._write_tools = write_tools or set()
    self._max_result_chars = max_result_chars
    self._approver = approval_callback or (lambda args, tool: True)

  def check(
    self,
    server: str,
    tool: str,
    arguments: Dict[str, Any],
    descriptor: ToolDescriptor,
  ) -> Tuple[bool, str, bool]:
    '''Evaluate the policy for one planned tool call.

    Args:
      server: Server name.
      tool: Tool name.
      arguments: Raw arguments from the LLM.
      descriptor: Tool descriptor with JSON schema.

    Returns:
      (allowed, reason, approved) triple.
    '''
    qualified = f'{server}.{tool}'

    # 1. JSON-Schema validation of arguments from the *untrusted* model.
    errors = validate_against_json_schema(
      arguments,
      descriptor.inputSchema or {'type': 'object'},
    )
    if errors:
      return False, f'args violate schema: {errors[0]}', False

    # 2. Write approval.
    is_write = self.is_write(server, tool)
    approved = True
    if is_write and self._approver is not None:
      approved = bool(self._approver(arguments, qualified))
      if not approved:
        return False, 'write call not approved by policy', False

    return True, 'ok', approved

  def is_write(self, server: str, tool: str) -> bool:
    '''Whether this tool mutates state.

    Args:
      server: Server name.
      tool: Tool name.

    Returns:
      True for write tools.
    '''
    return f'{server}.{tool}' in self._write_tools

  def sanitize_result(self, result: ToolCallResult) -> str:
    '''Prepare a tool result for the LLM context.

    Args:
      result: Raw tool result.

    Returns:
      Truncated text payload.
    '''
    text = result.text or ''
    if len(text) > self._max_result_chars:
      text = text[: self._max_result_chars - 3] + '...'
    return sanitize_untrusted(text, max_chars=self._max_result_chars)
