#!/usr/bin/env python
# -- coding: utf-8 --

'''LLM access layer.

All LLM traffic in this course flows through the user's local LLM gateway
(in-process import). This module is the single integration point:

- `GatewayClient` wraps `llm_gateway.LLMGateway`, resolving paths and
  converting failures into a stable error type.
- `MockGateway` implements the same interface with deterministic scripted
  behavior so demos/tests run fully offline.
'''


from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional

from agentic_common import paths
from agentic_common.logging import get_logger
from agentic_common.tracing import TokenUsage
from llm_gateway import GatewayRequest, GatewayResponse, LLMGateway

logger = get_logger(__name__)


class GatewayUnavailableError(RuntimeError):
  '''Raised when no gateway provider could serve a completion.'''


class GatewayClient:
  '''Thin facade over the user's LLM gateway (in-process).

  The gateway already provides provider routing, retries, rate limiting,
  caching, and a local Ollama fallback - this facade only resolves paths and
  normalizes errors.
  '''

  _instance: Optional['GatewayClient'] = None
  _instance_lock = threading.Lock()

  def __init__(
    self,
    config_path: Optional[str] = None,
    env_path: Optional[str] = None,
  ) -> None:
    '''Create the gateway-backed client.

    Args:
      config_path: Optional gateway YAML path override.
      env_path: Optional gateway .env path override.

    Raises:
      GatewayUnavailableError: If the gateway cannot be constructed.
    '''
    # llm_gateway.LLMGateway is the SanitizedLLMGateway variant, which
    # strips provider think-tokens from content - important for agents.
    try:
      self._gateway = LLMGateway(
        config_path=config_path or str(paths.GATEWAY_CONFIG),
        env_path=env_path or str(paths.GATEWAY_ENV),
        db_path=paths.GATEWAY_DB_PATH,
        cache_path=paths.GATEWAY_CACHE_PATH,
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      # Any construction failure (config, DB, keys) is terminal for the client.
      raise GatewayUnavailableError(
        f'failed to initialize gateway: {exc}'
      ) from exc

  @classmethod
  def shared(cls) -> 'GatewayClient':
    '''Return a process-wide shared client.

    Returns:
      The shared GatewayClient instance.
    '''
    with cls._instance_lock:
      if cls._instance is None:
        cls._instance = GatewayClient()
      return cls._instance

  def complete(self, request: GatewayRequest) -> GatewayResponse:
    '''Run one gateway completion.

    Args:
      request: The gateway request.

    Returns:
      The gateway response.

    Raises:
      GatewayUnavailableError: If every provider failed.
    '''
    try:
      return self._gateway.complete(request)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      # AllProvidersFailedError or provider chain failure: normalize once.
      raise GatewayUnavailableError(
        'LLM gateway could not complete the request. Check provider API keys '
        f'in the gateway .env or run with mock mode. Original error: {exc}'
      ) from exc

  def close(self) -> None:
    '''Release gateway resources.'''
    try:
      self._gateway.close()
    except Exception:  # pylint: disable=broad-exception-caught
      pass  # close() is best-effort; nothing to recover


# A scripted planner receives the messages list and the pending tool results
# and returns either a final answer or the next tool calls.
ScriptedPlanner = Callable[[List[Dict[str, Any]]], GatewayResponse]


class MockGateway:
  '''Deterministic stand-in for GatewayClient (offline demos and tests).

  The constructor takes a `planner` callable: given the normalized message
  list it returns a `GatewayResponse` (content and/or tool_calls). This keeps
  the agent code identical for mock and live runs - only the transport of
  intelligence changes, which is exactly the point of the gateway layer.
  '''

  def __init__(self, planner: ScriptedPlanner) -> None:
    '''Initialize with a scripted planner.

    Args:
      planner: Callable mapping messages to a GatewayResponse.
    '''
    self._planner = planner
    self.calls: List[List[Dict[str, Any]]] = []
    self._counter = 0

  def complete(self, request: GatewayRequest) -> GatewayResponse:  # noqa: D102
    messages = request.build_messages()
    self.calls.append([m.model_dump(mode='json') for m in messages])
    self._counter += 1
    response = self._planner([m.model_dump(mode='json') for m in messages])
    # Simulate token usage proportional to prompt/response size so the
    # observability paths are exercised in mock mode too.
    prompt_chars = sum(len(m.content or '') for m in messages)
    resp_chars = len(response.content) + sum(
      len(call.function.arguments) for call in response.tool_calls
    )
    response.usage = TokenUsage(
      input_tokens=max(1, prompt_chars // 4),
      output_tokens=max(1, resp_chars // 4),
      total_tokens=(prompt_chars + resp_chars) // 4 + 2,
    )
    response.prompt_chars = prompt_chars
    response.response_chars = resp_chars
    return response

  def close(self) -> None:  # noqa: D102
    return None

  @property
  def call_count(self) -> int:
    '''Number of complete() calls made so far.

    Returns:
      Call count.
    '''
    return self._counter
