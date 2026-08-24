#!/usr/bin/env python
# -- coding: utf-8 --

'''GatewayClient: process-wide facade over llm_gateway.LLMGateway.'''


from __future__ import annotations

import threading
from typing import Optional

from job_hunter.core.config import AppSettings
from llm_gateway import GatewayRequest, LLMGateway

_lock = threading.Lock()
_instance: Optional['GatewayClient'] = None


class GatewayClient:
  '''Owns the LLMGateway instance and adds run-scoped helpers.'''

  def __init__(self, settings: AppSettings) -> None:
    '''Build the underlying gateway with absolute paths.

    Args:
      settings: Application settings.
    '''
    self._settings = settings
    self._gateway = LLMGateway(
      config_path=settings.gateway_config_path,
      env_path=settings.gateway_env_path,
      db_path=settings.gateway_db_path,
      cache_path=settings.gateway_cache_path,
    )

  @property
  def gateway(self) -> LLMGateway:
    '''Return the raw gateway for advanced use.

    Returns:
      Underlying LLMGateway.
    '''
    return self._gateway

  async def acomplete_text(
    self,
    session_id: str,
    prompt: str = '',
    messages: Optional[list] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    tools: Optional[list] = None,
    tool_choice: Optional[object] = None,
  ) -> 'GatewayResponse':
    '''Async completion returning the full gateway response.

    Args:
      session_id: Correlation id for gateway cost logs.
      prompt: User prompt when messages not supplied.
      messages: Optional full message list overriding prompt.
      system_prompt: Optional system prompt.
      temperature: Optional sampling temperature.
      max_tokens: Optional output token cap.
      tools: Optional OpenAI-style tool definitions.
      tool_choice: Optional tool choice hint.

    Returns:
      GatewayResponse instance.

    Raises:
      GatewayUnavailableError: When every provider failed.
    '''
    from job_hunter.core.errors import GatewayUnavailableError
    request = GatewayRequest(
      prompt=prompt,
      messages=messages,
      system_prompt=system_prompt,
      temperature=temperature,
      max_tokens=max_tokens,
      tools=tools,
      tool_choice=tool_choice,
      session_id=session_id,
    )
    try:
      return await self._gateway.acomplete(request)
    except Exception as exc:
      raise GatewayUnavailableError(f'all providers failed: {exc}') from exc

  def stream_text(self, prompt: str, session_id: str):
    '''Synchronous sanitized stream of chunks (UI niceties).

    Args:
      prompt: User prompt.
      session_id: Correlation id.

    Yields:
      ProviderChunk items.
    '''
    return self._gateway.stream(GatewayRequest(prompt=prompt, session_id=session_id))


def get_client(settings: AppSettings) -> GatewayClient:
  '''Return the process-wide client, constructing it once.

  Args:
    settings: Application settings.

  Returns:
    Shared GatewayClient.
  '''
  global _instance
  if _instance is None:
    with _lock:
      if _instance is None:
        _instance = GatewayClient(settings)
  return _instance


def reset_client() -> None:
  '''Drop the shared instance (tests only).'''
  global _instance
  _instance = None
