#!/usr/bin/env python
# -- coding: utf-8 --

'''Tool provider factory: one place to choose the tool backend.

`inproc`  -> Chapter 1 server cores in-process (fast, offline-friendly).
`mcp`     -> live MCP servers over stdio (production transport path).

Both return the same `ToolProvider` interface with identical qualified tool
names, so switching backends changes no agent code.
'''


from __future__ import annotations

from chapter02_planning.tools.in_process import InProcessToolProvider
from chapter02_planning.tools.mcp_provider import MCPToolProvider
from chapter02_planning.tools.provider import ToolProvider

BACKENDS = ('inproc', 'mcp')


def build_tool_provider(
  backend: str = 'inproc',
  task: str = '',
  timeout_seconds: float = 15.0,
) -> ToolProvider:
  '''Build a raw (un-gated) tool provider.

  The agent wraps providers with `GatedToolProvider` per run, so callers
  should pass the raw provider to `PlanningAgent`.

  Args:
    backend: 'inproc' or 'mcp'.
    task: Task text used by MCP hint-based server discovery.
    timeout_seconds: MCP per-request timeout.

  Returns:
    A ToolProvider implementation.

  Raises:
    ValueError: When the backend name is unknown.
  '''
  if backend == 'mcp':
    return MCPToolProvider(task=task, timeout_seconds=timeout_seconds)
  if backend == 'inproc':
    return InProcessToolProvider()
  raise ValueError(f'unknown tool backend {backend!r}; use one of {BACKENDS}')
