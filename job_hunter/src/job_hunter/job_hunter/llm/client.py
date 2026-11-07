#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''GatewayClient: process-wide facade over llm_gateway.LLMGateway.'''


from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from job_hunter.core.config import AppSettings
from llm_gateway import GatewayRequest, LLMGateway

if TYPE_CHECKING:
  from llm_gateway.schemas import GatewayResponse

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_clients: Dict[Tuple[str, str, str, str], 'GatewayClient'] = {}
# Legacy alias kept for the pre-keying singleton test/shim: the last
# client handed out by get_client. Prefer the keyed _clients cache.
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

    .. deprecated::
      Raw access is deprecated; use :meth:`acomplete_text` instead.
      Kept for backwards compatibility.

    Returns:
      Underlying LLMGateway.
    '''
    message = (
      'GatewayClient.gateway raw escape is deprecated; '
      'use acomplete_text instead'
    )
    logger.warning(message)
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


def get_client(settings: AppSettings) -> GatewayClient:
  '''Return the cached client for the settings' gateway paths.

  Clients are keyed by (gateway_db_path, gateway_cache_path,
  gateway_config_path, gateway_env_path) so two data dirs or two
  gateway configs never share one gateway; the same key returns
  the same instance.

  Args:
    settings: Application settings.

  Returns:
    Cached GatewayClient for those paths.
  '''
  global _instance
  # getattr with '' default keeps legacy test doubles that only
  # supply the two original paths working; real AppSettings always
  # provides all four, so distinct configs never share a client.
  key = (
    str(settings.gateway_db_path),
    str(settings.gateway_cache_path),
    str(getattr(settings, 'gateway_config_path', '')),
    str(getattr(settings, 'gateway_env_path', '')),
  )
  client = _clients.get(key)
  if client is None:
    with _lock:
      client = _clients.get(key)
      if client is None:
        client = GatewayClient(settings)
        _clients[key] = client
  with _lock:
    # Legacy alias kept for tests only (see test_reset_client in
    # test_fix_services_llm); assign under lock to avoid races.
    _instance = client
  return client


def reset_client() -> None:
  '''Drop every cached client (tests only).'''
  global _instance
  with _lock:
    _clients.clear()
    # Legacy alias kept for tests only; assign under lock.
    _instance = None
