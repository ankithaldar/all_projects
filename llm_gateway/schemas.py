#!/usr/bin/env python
# -- coding: utf-8 --

'''Pydantic schemas for gateway requests, responses, tools, and logging.'''


from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FunctionCall(BaseModel):
  '''Normalized function call emitted by a model.'''

  model_config = ConfigDict(extra='ignore')

  name: str
  arguments: str = '{}'


class ToolCall(BaseModel):
  '''Normalized tool call emitted by a model.'''

  model_config = ConfigDict(extra='ignore')

  id: str = ''
  type: str = 'function'
  function: FunctionCall


class ChatMessage(BaseModel):
  '''Chat message compatible with OpenAI-style chat APIs.'''

  model_config = ConfigDict(extra='ignore')

  role: str
  content: Optional[str] = None
  name: Optional[str] = None
  tool_calls: Optional[List[ToolCall]] = None
  tool_call_id: Optional[str] = None


class ToolFunctionDefinition(BaseModel):
  '''Function definition used by tool calling.'''

  model_config = ConfigDict(extra='ignore')

  name: str
  description: str = ''
  parameters: Dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
  '''Tool definition compatible with OpenAI-style tools.'''

  model_config = ConfigDict(extra='ignore')

  type: str = 'function'
  function: ToolFunctionDefinition


class GatewayRequest(BaseModel):
  '''Standard request submitted by the main application.'''

  model_config = ConfigDict(extra='ignore')

  prompt: str = ''
  system_prompt: Optional[str] = None
  messages: Optional[List[ChatMessage]] = None
  tools: Optional[List[ToolDefinition]] = None
  tool_choice: Optional[Union[str, Dict[str, Any]]] = None
  temperature: Optional[float] = None
  max_tokens: Optional[int] = None
  session_id: Optional[str] = None
  metadata: Dict[str, Any] = Field(default_factory=dict)

  @model_validator(mode='after')
  def validate_prompt_or_messages(self) -> 'GatewayRequest':
    '''Ensure that either prompt or messages are supplied.

    Returns:
      The validated request instance.

    Raises:
      ValueError: If neither prompt nor messages are provided.
    '''
    if not self.prompt and not self.messages:
      raise ValueError('prompt or messages is required')
    return self

  def build_messages(
    self,
    default_system_prompt: str = '',
  ) -> List[ChatMessage]:
    '''Build normalized chat messages.

    Args:
      default_system_prompt: System prompt used when no explicit system prompt
        is present.

    Returns:
      A list of chat messages.
    '''
    messages = list(self.messages) if self.messages else []

    if not messages and self.prompt:
      messages = [ChatMessage(role='user', content=self.prompt)]

    system_prompt = self.system_prompt
    if system_prompt is None and default_system_prompt:
      system_prompt = default_system_prompt

    if system_prompt:
      if not messages or messages[0].role != 'system':
        messages.insert(0, ChatMessage(role='system', content=system_prompt))

    return messages


class ProviderRequest(BaseModel):
  '''Provider-level request after gateway routing.'''

  model_config = ConfigDict(extra='ignore')

  provider: str
  model: str
  messages: List[ChatMessage]
  tools: Optional[List[ToolDefinition]] = None
  tool_choice: Optional[Union[str, Dict[str, Any]]] = None
  temperature: float = 0.2
  max_tokens: int = 1024
  stream: bool = False


class Usage(BaseModel):
  '''Token usage information.'''

  model_config = ConfigDict(extra='ignore')

  input_tokens: int = 0
  output_tokens: int = 0
  total_tokens: int = 0


class ProviderResponse(BaseModel):
  '''Normalized response returned by a provider.'''

  model_config = ConfigDict(extra='ignore')

  provider: str
  model: str
  content: str = ''
  tool_calls: List[ToolCall] = Field(default_factory=list)
  usage: Usage = Field(default_factory=Usage)
  raw: Dict[str, Any] = Field(default_factory=dict)


class ProviderChunk(BaseModel):
  '''Normalized streaming chunk returned by a provider.'''

  model_config = ConfigDict(extra='ignore')

  provider: str
  model: str
  delta_content: str = ''
  delta_tool_calls: List[ToolCall] = Field(default_factory=list)
  finish_reason: Optional[str] = None
  raw: Dict[str, Any] = Field(default_factory=dict)


class GatewayResponse(BaseModel):
  '''Final gateway response returned to the main application.'''

  model_config = ConfigDict(extra='ignore')

  request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
  provider: str
  model: str
  alias: str = ''
  content: str = ''
  tool_calls: List[ToolCall] = Field(default_factory=list)
  usage: Usage = Field(default_factory=Usage)
  cached: bool = False
  latency_ms: float = 0.0
  cost: float = 0.0
  temperature: float = 0.0
  system_prompt: str = ''
  prompt_chars: int = 0
  response_chars: int = 0
  session_id: Optional[str] = None


class LogRecord(BaseModel):
  '''Record persisted for every gateway-provider API call attempt.'''

  model_config = ConfigDict(extra='ignore')

  timestamp: datetime = Field(
    default_factory=lambda: datetime.now(timezone.utc),
  )
  provider: str
  model_used: str
  input_tokens: int = 0
  output_tokens: int = 0
  latency_ms: float = 0.0
  status: str
  error: Optional[str] = None
  prompt_chars: int = 0
  response_chars: int = 0
  cost: float = 0.0
  temperature: float = 0.0
  system_prompt: str = ''
  session_id: Optional[str] = None
