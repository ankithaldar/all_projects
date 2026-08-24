#!/usr/bin/env python
# -- coding: utf-8 --

'''MCP tool discovery/execution and gateway tool-calling loop.'''


from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any, Awaitable, Callable, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from job_hunter.core.config import AppSettings


class MCPClientManager:
  '''Owns stdio sessions to configured MCP servers.

  Attributes:
    sessions: Mapping of server name to active ClientSession.
  '''

  def __init__(self, settings: AppSettings) -> None:
    '''Store server configs; call open() before use.

    Args:
      settings: Application settings.
    '''
    self._settings = settings
    self._stack: Optional[AsyncExitStack] = None
    self.sessions: Dict[str, ClientSession] = {}

  async def open(self) -> None:
    '''Launch all configured servers over stdio.'''
    if self._stack is not None:
      return
    stack = AsyncExitStack()
    for name, spec in self._settings.mcp.items():
      command = str(spec.get('command', '{python}')).replace(
        '{python}', __import__('sys').executable,
      )
      params = StdioServerParameters(command=command, args=list(spec.get('args', [])))
      read, write = await stack.enter_async_context(stdio_client(params))
      session = await stack.enter_async_context(ClientSession(read, write))
      await session.initialize()
      self.sessions[name] = session
    self._stack = stack

  async def close(self) -> None:
    '''Shut down all sessions.'''
    if self._stack is not None:
      await self._stack.aclose()
      self._stack = None
      self.sessions.clear()

  async def list_tools(self) -> List[Dict[str, Any]]:
    '''Collect OpenAI-style tool definitions from every session.

    Returns:
      Tool definition mappings compatible with GatewayRequest.tools.
    '''
    tools: List[Dict[str, Any]] = []
    for name, session in self.sessions.items():
      result = await session.list_tools()
      for tool in result.tools:
        tools.append({
          'type': 'function',
          'function': {
            'name': f'{name}.{tool.name}',
            'description': tool.description or '',
            'parameters': tool.inputSchema or {'type': 'object'},
          },
        })
    return tools

  async def call_tool(self, qualified: str, arguments: Dict[str, Any]) -> str:
    '''Invoke one tool by qualified name server.tool.

    Args:
      qualified: Qualified tool name.
      arguments: JSON arguments.

    Returns:
      Text content of the first result item.
    '''
    server_name, tool_name = qualified.split('.', 1)
    session = self.sessions[server_name]
    result = await asyncio.wait_for(session.call_tool(tool_name, arguments), timeout=90)
    parts = []
    for item in result.content:
      text = getattr(item, 'text', None)
      if text:
        parts.append(text)
    return '\n'.join(parts) or json.dumps({'ok': True})


async def run_tool_loop(
  client: Any,
  manager: MCPClientManager,
  instruction: str,
  session_id: str,
  max_rounds: int = 6,
) -> str:
  '''ReAct-style loop: model proposes tool calls, we execute via MCP.

  Args:
    client: GatewayClient instance.
    manager: Connected MCPClientManager.
    instruction: User task description.
    session_id: Correlation id.
    max_rounds: Safety cap on tool rounds.

  Returns:
    Final assistant text content.
  '''
  tools = await manager.list_tools()

  def find(qualified: str) -> Optional[Dict[str, Any]]:
    for tool in tools:
      if tool['function']['name'] == qualified:
        return tool
    return None

  messages: List[Dict[str, Any]] = [{'role': 'user', 'content': instruction}]
  for _ in range(max_rounds):
    response = await client.acomplete_text(
      session_id=session_id,
      messages=messages,
      tools=tools,
      temperature=0.1,
    )
    calls = response.tool_calls or []
    if not calls:
      return response.content or ''
    messages.append({
      'role': 'assistant',
      'content': response.content or '',
      'tool_calls': [call.model_dump(exclude_none=True) for call in calls],
    })
    for call in calls:
      name = call.function.name
      try:
        arguments = json.loads(call.function.arguments or '{}')
        output = (
          await manager.call_tool(name, arguments)
          if find(name) else f'unknown tool {name}'
        )
      except Exception as exc:
        output = f'tool error: {exc}'
      messages.append({
        'role': 'tool',
        'tool_call_id': call.id,
        'content': output[:8000],
      })
  final = await client.acomplete_text(session_id=session_id, messages=messages)
  return final.content or ''


Executor = Callable[[str, Dict[str, Any]], Awaitable[str]]
