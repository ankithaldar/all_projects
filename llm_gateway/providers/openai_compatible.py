#!/usr/bin/env python
# -- coding: utf-8 --

'''OpenAI-compatible provider implementations.'''

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import httpx
import requests
from llm_gateway.config import ProviderConfig
from llm_gateway.errors import (AuthenticationError, ProviderError,
                                RateLimitedProviderError,
                                TransientProviderError)
from llm_gateway.key_rotator import APIKeyRotator
from llm_gateway.providers.base import LLMProvider
from llm_gateway.schemas import (ChatMessage, FunctionCall, ProviderChunk,
                                 ProviderRequest, ProviderResponse, ToolCall,
                                 Usage)


class OpenAICompatibleProvider(LLMProvider):
  '''Provider implementation for OpenAI-compatible HTTP APIs.'''

  def __init__(
    self,
    name: str,
    config: ProviderConfig,
    rotator: Optional[APIKeyRotator] = None,
    requests_session: Optional[requests.Session] = None,
    httpx_client: Optional[httpx.AsyncClient] = None,
  ) -> None:
    '''Initialize provider.

    Args:
      name: Provider name.
      config: Provider configuration.
      rotator: Optional API key rotator.
      requests_session: Optional synchronous HTTP session.
      httpx_client: Optional asynchronous HTTP client.
    '''
    super().__init__(name)
    self._config = config
    self._base_url = config.base_url.rstrip('/')
    self._timeout = config.timeout_seconds
    self._rotator = rotator
    self._session = requests_session or requests.Session()
    self._httpx_client = httpx_client

  def chat(self, request: ProviderRequest) -> ProviderResponse:
    '''Execute synchronous chat call.

    Args:
      request: Provider request.

    Returns:
      Provider response.

    Raises:
      ProviderError: For non-transient HTTP or request errors.
      RateLimitedProviderError: For HTTP 429 responses.
      TransientProviderError: For timeouts and connection errors.
    '''
    payload = self._payload(request, stream=False)
    headers = self._headers(self._next_key())

    try:
      response = self._session.post(
        self._chat_url(),
        json=payload,
        headers=headers,
        timeout=self._timeout,
      )
    except (
      requests.exceptions.Timeout,
      requests.exceptions.ConnectionError,
      requests.exceptions.ChunkedEncodingError,
    ) as exc:
      raise TransientProviderError(str(exc), provider=self.name) from exc
    except requests.exceptions.RequestException as exc:
      raise ProviderError(str(exc), provider=self.name) from exc

    if response.status_code != 200:
      self._raise_for_status(response.status_code, response.text)

    try:
      data = response.json()
    except ValueError as exc:
      raise ProviderError(
        'invalid JSON response from provider',
        provider=self.name,
      ) from exc

    return self._parse_response(data, request)

  async def achat(self, request: ProviderRequest) -> ProviderResponse:
    '''Execute asynchronous chat call.

    Args:
      request: Provider request.

    Returns:
      Provider response.

    Raises:
      ProviderError: For non-transient HTTP or request errors.
      RateLimitedProviderError: For HTTP 429 responses.
      TransientProviderError: For timeouts and connection errors.
    '''
    payload = self._payload(request, stream=False)
    headers = self._headers(self._next_key())

    client = self._httpx_client
    should_close = False

    if client is None:
      client = httpx.AsyncClient(timeout=self._timeout)
      should_close = True

    try:
      response = await client.post(
        self._chat_url(),
        json=payload,
        headers=headers,
        timeout=self._timeout,
      )

      if response.status_code != 200:
        self._raise_for_status(response.status_code, response.text)

      try:
        data = response.json()
      except ValueError as exc:
        raise ProviderError(
          'invalid JSON response from provider',
          provider=self.name,
        ) from exc

      return self._parse_response(data, request)
    except (
      httpx.TimeoutException,
      httpx.ConnectError,
      httpx.RemoteProtocolError,
    ) as exc:
      raise TransientProviderError(str(exc), provider=self.name) from exc
    except httpx.HTTPError as exc:
      raise ProviderError(str(exc), provider=self.name) from exc
    finally:
      if should_close:
        await client.aclose()

  def stream(self, request: ProviderRequest) -> Iterator[ProviderChunk]:
    '''Execute synchronous streaming call.

    Args:
      request: Provider request.

    Yields:
      Provider streaming chunks.

    Raises:
      ProviderError: For non-transient HTTP or request errors.
      RateLimitedProviderError: For HTTP 429 responses.
      TransientProviderError: For timeouts and connection errors.
    '''
    payload = self._payload(request, stream=True)
    headers = self._headers(self._next_key())

    try:
      with self._session.post(
        self._chat_url(),
        json=payload,
        headers=headers,
        timeout=self._timeout,
        stream=True,
      ) as response:
        if response.status_code != 200:
          response.read()
          self._raise_for_status(response.status_code, response.text)

        for line in response.iter_lines(decode_unicode=True):
          chunk = self._parse_sse_line(line, request)
          if chunk is not None:
            yield chunk
    except (
      requests.exceptions.Timeout,
      requests.exceptions.ConnectionError,
      requests.exceptions.ChunkedEncodingError,
    ) as exc:
      raise TransientProviderError(str(exc), provider=self.name) from exc
    except requests.exceptions.RequestException as exc:
      raise ProviderError(str(exc), provider=self.name) from exc

  async def astream(
    self,
    request: ProviderRequest,
  ) -> AsyncIterator[ProviderChunk]:
    '''Execute asynchronous streaming call.

    Args:
      request: Provider request.

    Yields:
      Provider streaming chunks.

    Raises:
      ProviderError: For non-transient HTTP or request errors.
      RateLimitedProviderError: For HTTP 429 responses.
      TransientProviderError: For timeouts and connection errors.
    '''
    payload = self._payload(request, stream=True)
    headers = self._headers(self._next_key())

    client = self._httpx_client
    should_close = False

    if client is None:
      client = httpx.AsyncClient(timeout=self._timeout)
      should_close = True

    try:
      async with client.stream(
        'POST',
        self._chat_url(),
        json=payload,
        headers=headers,
        timeout=self._timeout,
      ) as response:
        if response.status_code != 200:
          body = await response.aread()
          self._raise_for_status(
            response.status_code,
            body.decode('utf-8', 'ignore'),
          )

        async for line in response.aiter_lines():
          chunk = self._parse_sse_line(line, request)
          if chunk is not None:
            yield chunk
    except (
      httpx.TimeoutException,
      httpx.ConnectError,
      httpx.RemoteProtocolError,
    ) as exc:
      raise TransientProviderError(str(exc), provider=self.name) from exc
    except httpx.HTTPError as exc:
      raise ProviderError(str(exc), provider=self.name) from exc
    finally:
      if should_close:
        await client.aclose()

  def _chat_url(self) -> str:
    '''Build chat completion URL.

    Returns:
      Chat completion endpoint URL.
    '''
    return f'{self._base_url}/chat/completions'

  def _next_key(self) -> str:
    '''Get next API key from rotator.

    Returns:
      API key or empty string.
    '''
    if self._rotator is None:
      return ''

    return self._rotator.get_and_rotate()

  def _headers(self, api_key: str) -> Dict[str, str]:
    '''Build request headers.

    Args:
      api_key: Optional API key.

    Returns:
      Header dictionary.
    '''
    headers = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    }

    if api_key:
      headers['Authorization'] = f'Bearer {api_key}'

    return headers

  def _payload(
    self,
    request: ProviderRequest,
    stream: bool,
  ) -> Dict[str, Any]:
    '''Build OpenAI-compatible request payload.

    Args:
      request: Provider request.
      stream: Whether to request streaming.

    Returns:
      JSON payload.
    '''
    payload: Dict[str, Any] = {
      'model': request.model,
      'messages': [
        self._serialize_message(message)
        for message in request.messages
      ],
      'temperature': request.temperature,
      'max_tokens': request.max_tokens,
      'stream': stream,
    }

    if request.tools:
      payload['tools'] = [
        tool.model_dump(exclude_none=True, mode='json')
        for tool in request.tools
      ]

    if request.tool_choice is not None:
      payload['tool_choice'] = request.tool_choice

    return payload

  def _serialize_message(self, message: ChatMessage) -> Dict[str, Any]:
    '''Serialize one message.

    Args:
      message: Chat message.

    Returns:
      JSON-compatible message dictionary.
    '''
    return message.model_dump(exclude_none=True, mode='json')

  def _raise_for_status(
    self,
    status_code: int,
    text: str,
  ) -> None:
    '''Convert HTTP status into typed provider errors.

    Args:
      status_code: HTTP status code.
      text: Response body text.

    Raises:
      RateLimitedProviderError: For 429 status.
      TransientProviderError: For 5xx and known transient status codes.
      AuthenticationError: For 401 and 403 status codes.
      ProviderError: For other non-success status codes.
    '''
    snippet = (text or '')[:500]

    if status_code == 429:
      raise RateLimitedProviderError(
        f'rate limited: {snippet}',
        provider=self.name,
        status_code=status_code,
      )

    if status_code in {408, 500, 502, 503, 504, 520, 529}:
      raise TransientProviderError(
        f'transient provider error: {snippet}',
        provider=self.name,
        status_code=status_code,
      )

    if status_code in {401, 403}:
      raise AuthenticationError(
        f'authentication failed: {snippet}',
        provider=self.name,
        status_code=status_code,
      )

    if status_code >= 400:
      raise ProviderError(
        f'provider error: {snippet}',
        provider=self.name,
        status_code=status_code,
      )

  def _parse_response(
    self,
    data: Dict[str, Any],
    request: ProviderRequest,
  ) -> ProviderResponse:
    '''Parse OpenAI-compatible JSON response.

    Args:
      data: Raw response payload.
      request: Original provider request.

    Returns:
      Normalized provider response.
    '''
    choices = data.get('choices') or [{}]
    choice = choices[0] if choices else {}
    message = choice.get('message') or {}

    content = message.get('content') or ''
    tool_calls = self._parse_tool_calls(message.get('tool_calls') or [])

    usage_raw = data.get('usage') or {}
    usage = Usage(
      input_tokens=int(usage_raw.get('prompt_tokens') or 0),
      output_tokens=int(usage_raw.get('completion_tokens') or 0),
      total_tokens=int(usage_raw.get('total_tokens') or 0),
    )

    return ProviderResponse(
      provider=self.name,
      model=str(data.get('model') or request.model),
      content=content,
      tool_calls=tool_calls,
      usage=usage,
      raw=data,
    )

  def _parse_tool_calls(
    self,
    raw_calls: List[Dict[str, Any]],
  ) -> List[ToolCall]:
    '''Parse tool calls from provider response.

    Args:
      raw_calls: Raw tool call dictionaries.

    Returns:
      Normalized tool calls.
    '''
    calls: List[ToolCall] = []

    for raw in raw_calls or []:
      fn = raw.get('function') or {}

      calls.append(
        ToolCall(
          id=str(raw.get('id') or ''),
          type=str(raw.get('type') or 'function'),
          function=FunctionCall(
            name=str(fn.get('name') or ''),
            arguments=str(fn.get('arguments') or '{}'),
          ),
        )
      )

    return calls

  def _parse_sse_line(
    self,
    line: str,
    request: ProviderRequest,
  ) -> Optional[ProviderChunk]:
    '''Parse one SSE line.

    Args:
      line: Raw SSE line.
      request: Original provider request.

    Returns:
      Normalized streaming chunk or None.
    '''
    if not line:
      return None

    line = line.strip()

    if not line.startswith('data:'):
      return None

    data = line[5:].strip()

    if data == '[DONE]':
      return None

    try:
      parsed = json.loads(data)
    except json.JSONDecodeError:
      return None

    choices = parsed.get('choices') or []
    if not choices:
      return None

    choice = choices[0] or {}
    delta = choice.get('delta') or {}
    content = delta.get('content') or ''
    tool_calls = self._parse_tool_calls(delta.get('tool_calls') or [])

    return ProviderChunk(
      provider=self.name,
      model=str(parsed.get('model') or request.model),
      delta_content=content,
      delta_tool_calls=tool_calls,
      finish_reason=choice.get('finish_reason'),
      raw=parsed,
    )


class BytezProvider(OpenAICompatibleProvider):
  '''Bytez provider'''


class OpenRouterProvider(OpenAICompatibleProvider):
  '''OpenRouter provider.'''


class GroqProvider(OpenAICompatibleProvider):
  '''Groq provider.'''


class CerebrasProvider(OpenAICompatibleProvider):
  '''Cerebras provider.'''


class NvidiaProvider(OpenAICompatibleProvider):
  '''Nvidia provider.'''

class GithubProvider(OpenAICompatibleProvider):
  '''Github provider.'''


class OllamaProvider(OpenAICompatibleProvider):
  '''Local Ollama provider using OpenAI-compatible endpoint.'''
