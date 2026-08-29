#!/usr/bin/env python
# -- coding: utf-8 --

'''Tool-call security policy enforced by the CLIENT (defense at the boundary).

Server-side validation alone is not enough: the agent must also decide
1. is this tool allowed at all? (read vs write classification)
2. are the arguments inside policy limits? (e.g. max restock quantity)
3. does a WRITE need human approval? (approval callback)
4. is the result payload safe to feed back into the LLM? (truncation)
'''


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from agentic_common.security import (
  sanitize_untrusted,
  validate_against_json_schema,
)
from chapter01_mcp.logging_util import get_mcp_logger
from chapter01_mcp.schemas import ToolCallResult, ToolDescriptor

logger = get_mcp_logger(__name__)


@dataclass
class ToolPolicy:
  '''Policy for one tool (or defaults for all tools on a server).'''

  requires_approval: bool = False
  max_result_chars: int = 6000
  # Optional extra argument constraints enforced *before* JSON-Schema check.
  argument_limits: Optional[Dict[str, Tuple[int, int]]] = None
  allowed_values: Optional[Dict[str, set]] = None


@dataclass
class PolicyDecision:
  '''Outcome of a policy evaluation.'''

  allowed: bool
  reason: str = ''
  approved: bool = True


class ToolPolicyEngine:
  '''Central gate for every agent tool call.

  Checks, in order:
  1. tool known to policy? (unknown tools on a server are still allowed for
     read-only classification but logged)
  2. JSON-Schema validation of arguments (against the advertised schema)
  3. argument range overrides (tighter than the schema, e.g. restock cap)
  4. allowed-values overrides (e.g. dispatch priority)
  5. write tools require approval callback -> else blocked with reason
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
    self._overrides: Dict[str, ToolPolicy] = {}
    self._defaults: Dict[str, ToolPolicy] = {}

  def set_server_policy(self, server: str, policy: ToolPolicy) -> None:
    '''Configure defaults for all tools on a server.

    Args:
      server: Server name.
      policy: Default policy.
    '''
    self._defaults[server] = policy

  def set_tool_policy(self, qualified: str, policy: ToolPolicy) -> None:
    '''Set a tool-specific policy override.

    Args:
      qualified: `server.tool` name.
      policy: The policy.
    '''
    self._overrides[qualified] = policy

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

    policy = self._policy_for(server, tool)

    # 2. Argument range checks (stricter than schema).
    if policy.argument_limits:
      for arg, (low, high) in policy.argument_limits.items():
        value = arguments.get(arg)
        in_range = isinstance(value, (int, float)) and low <= value <= high
        if not in_range:
          detail = f'{arg}={value} outside allowed range [{low}, {high}]'
          return False, detail, False

    # 3. Allowed values.
    if policy.allowed_values:
      for arg, allowed in policy.allowed_values.items():
        value = arguments.get(arg)
        if arg in arguments and value not in allowed:
          return False, f'{arg}={value!r} not permitted', False

    # 4. Write approval.
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

  def _policy_for(self, server: str, tool: str) -> ToolPolicy:
    '''Resolve effective policy (tool override > server default > empty).

    Args:
      server: Server name.
      tool: Tool name.

    Returns:
      Resolved ToolPolicy.
    '''
    override = self._overrides.get(f'{server}.{tool}')
    if override:
      return override
    return self._defaults.get(server, ToolPolicy())
