#!/usr/bin/env python
# -- coding: utf-8 --

'''Pydantic schemas for MCP entities used across Chapter 1.

Everything that crosses a process boundary (JSON-RPC messages, tool
descriptors, tool results, server descriptors, agent task inputs/outputs) is
modeled here so validation happens at the boundary, not deep inside logic.
'''


from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Wire-format field names follow the MCP spec exactly (camelCase);
# they are protocol constants, not style violations.
# pylint: disable=invalid-name


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 message models
# ---------------------------------------------------------------------------

class JsonRpcError(BaseModel):
  '''JSON-RPC error object.'''

  model_config = ConfigDict(extra='ignore')

  code: int
  message: str
  data: Optional[Dict[str, Any]] = None


class JsonRpcRequest(BaseModel):
  '''A JSON-RPC request expecting a response.'''

  model_config = ConfigDict(extra='ignore')

  jsonrpc: Literal['2.0'] = '2.0'
  id: int | str
  method: str
  params: Dict[str, Any] = Field(default_factory=dict)


class JsonRpcNotification(BaseModel):
  '''A JSON-RPC notification (no id, no response expected).'''

  model_config = ConfigDict(extra='ignore')

  jsonrpc: Literal['2.0'] = '2.0'
  method: str
  params: Dict[str, Any] = Field(default_factory=dict)


class JsonRpcSuccess(BaseModel):
  '''A successful JSON-RPC response.'''

  model_config = ConfigDict(extra='ignore')

  jsonrpc: Literal['2.0'] = '2.0'
  id: int | str
  result: Dict[str, Any] = Field(default_factory=dict)


class JsonRpcFailure(BaseModel):
  '''A failed JSON-RPC response.'''

  model_config = ConfigDict(extra='ignore')

  jsonrpc: Literal['2.0'] = '2.0'
  id: int | str | None = None
  error: JsonRpcError


# ---------------------------------------------------------------------------
# MCP lifecycle models
# ---------------------------------------------------------------------------

class InitializeParams(BaseModel):
  '''Client -> server initialize request parameters.'''

  model_config = ConfigDict(extra='ignore')

  protocolVersion: str
  capabilities: Dict[str, Any] = Field(default_factory=dict)
  clientInfo: Dict[str, str] = Field(default_factory=dict)


class InitializeResult(BaseModel):
  '''Server -> client initialize result.'''

  model_config = ConfigDict(extra='ignore')

  protocolVersion: str
  capabilities: Dict[str, Any] = Field(default_factory=dict)
  serverInfo: Dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# MCP tool models
# ---------------------------------------------------------------------------

class ToolDescriptor(BaseModel):
  '''A tool as advertised by an MCP server (tools/list entry).

  `inputSchema` is a standard JSON Schema object describing the tool's
  arguments - exactly what an LLM function-calling API expects.
  '''

  model_config = ConfigDict(extra='ignore')

  name: str
  description: str = ''
  inputSchema: Dict[str, Any] = Field(default_factory=dict)


class TextContent(BaseModel):
  '''One text block of a tool result.'''

  model_config = ConfigDict(extra='ignore')

  type: Literal['text'] = 'text'
  text: str = ''


class ToolCallResult(BaseModel):
  '''Result envelope returned by tools/call.

  Key MCP design point: *tool-level* failures are normal results with
  `isError: true` (the model can read the error and recover). Only
  *protocol-level* failures use JSON-RPC errors.
  '''

  model_config = ConfigDict(extra='ignore')

  content: List[TextContent] = Field(default_factory=list)
  isError: bool = False

  @property
  def text(self) -> str:
    '''Concatenated text of all content blocks.

    Returns:
      Combined text payload.
    '''
    return '\n'.join(block.text for block in self.content)


class CallToolParams(BaseModel):
  '''Parameters of the tools/call request.'''

  model_config = ConfigDict(extra='ignore')

  name: str
  arguments: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Server descriptor (how the client knows how to reach a server)
# ---------------------------------------------------------------------------

class ServerDescriptor(BaseModel):
  '''Declarative description of an MCP server the agent may use.

  Attributes:
    name: Unique logical server name.
    transport: 'stdio' (subprocess) or 'sse' (HTTP).
    command: Executable for stdio transport.
    args: Command arguments for stdio transport.
    cwd: Working directory for the subprocess.
    env: Extra environment variables for the subprocess.
    url: Base URL for SSE transport.
    summary: One-line human summary (used by dynamic tool discovery).
    hints: Keywords describing what this server is good for.
    enabled: Whether the catalog may use this server.
  '''

  model_config = ConfigDict(extra='ignore')

  name: str
  transport: Literal['stdio', 'sse'] = 'stdio'
  command: str = ''
  args: List[str] = Field(default_factory=list)
  cwd: str = ''
  env: Dict[str, str] = Field(default_factory=dict)
  url: str = ''
  summary: str = ''
  hints: List[str] = Field(default_factory=list)
  enabled: bool = True


# ---------------------------------------------------------------------------
# Agent task input/output
# ---------------------------------------------------------------------------

class AgentTaskInput(BaseModel):
  '''Validated input for one agent task run.'''

  model_config = ConfigDict(extra='ignore')

  task: str = Field(min_length=1, max_length=4000)
  session_id: str = Field(min_length=1, max_length=128)
  max_iterations: int = Field(default=8, ge=1, le=32)


class ToolCallAudit(BaseModel):
  '''Audit view of one tool call made by the agent.'''

  model_config = ConfigDict(extra='ignore')

  server: str
  tool: str
  args: Dict[str, Any] = Field(default_factory=dict)
  ok: bool = True
  approved: bool = True
  error: Optional[str] = None
  latency_ms: float = 0.0
  result_preview: str = ''


class AgentTaskOutput(BaseModel):
  '''Validated output of one agent task run.'''

  model_config = ConfigDict(extra='ignore')

  session_id: str
  answer: str
  status: Literal['completed', 'max_iterations', 'error'] = 'completed'
  iterations: int = 0
  tool_calls: List[ToolCallAudit] = Field(default_factory=list)
  usage: Dict[str, Any] = Field(default_factory=dict)
  errors: List[str] = Field(default_factory=list)
