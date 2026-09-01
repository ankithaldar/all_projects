#!/usr/bin/env python
# -- coding: utf-8 --

'''Gateway wrapper that removes think tokens from displayed output.'''

from __future__ import annotations

from typing import AsyncIterator, Iterator, List, Optional

from llm_gateway.app import LLMGateway
from llm_gateway.cache import SQLiteCache
from llm_gateway.sanitize import StreamThinkSanitizer, sanitize_think_tokens
from llm_gateway.schemas import GatewayRequest, GatewayResponse, ProviderChunk


class SanitizedLLMGateway(LLMGateway):
  '''Gateway that sanitizes displayed output while preserving raw cache.'''

  def complete(self, request: GatewayRequest) -> GatewayResponse:
    '''Execute a synchronous sanitized completion.

    Args:
      request: Gateway request.

    Returns:
      Gateway response with sanitized content.
    '''
    response = super().complete(request)
    return self._sanitize_complete_response(request, response)

  async def acomplete(self, request: GatewayRequest) -> GatewayResponse:
    '''Execute an asynchronous sanitized completion.

    Args:
      request: Gateway request.

    Returns:
      Gateway response with sanitized content.
    '''
    response = await super().acomplete(request)
    return self._sanitize_complete_response(request, response)

  def stream(self, request: GatewayRequest) -> Iterator[ProviderChunk]:
    '''Execute a synchronized sanitized stream.

    Args:
      request: Gateway request.

    Yields:
      Sanitized streaming chunks.
    '''
    cache_key = self._cache_key(request)
    cached_chunk = self._cached_chunk(cache_key)

    if cached_chunk is not None:
      yield cached_chunk
      return

    sanitizer = StreamThinkSanitizer()
    raw_parts: List[str] = []
    safe_parts: List[str] = []
    last_chunk: Optional[ProviderChunk] = None
    from_cache = False

    for chunk in super().stream(request):
      if chunk.provider == 'cache':
        from_cache = True

      last_chunk = chunk
      raw_delta = chunk.delta_content or ''

      if raw_delta:
        raw_parts.append(raw_delta)

      safe_delta = sanitizer.feed(raw_delta)

      if safe_delta:
        safe_parts.append(safe_delta)

      if safe_delta or chunk.delta_tool_calls or chunk.finish_reason:
        yield chunk.model_copy(update={'delta_content': safe_delta})

    tail = sanitizer.flush()

    if tail:
      safe_parts.append(tail)
      yield self._tail_chunk(last_chunk, tail)

    if not from_cache:
      self._cache_stream(cache_key, raw_parts, safe_parts, last_chunk)

  async def astream(
    self,
    request: GatewayRequest,
  ) -> AsyncIterator[ProviderChunk]:
    '''Execute an asynchronous sanitized stream.

    Args:
      request: Gateway request.

    Yields:
      Sanitized streaming chunks.
    '''
    cache_key = self._cache_key(request)
    cached_chunk = self._cached_chunk(cache_key)

    if cached_chunk is not None:
      yield cached_chunk
      return

    sanitizer = StreamThinkSanitizer()
    raw_parts: List[str] = []
    safe_parts: List[str] = []
    last_chunk: Optional[ProviderChunk] = None
    from_cache = False

    async for chunk in super().astream(request):
      if chunk.provider == 'cache':
        from_cache = True

      last_chunk = chunk
      raw_delta = chunk.delta_content or ''

      if raw_delta:
        raw_parts.append(raw_delta)

      safe_delta = sanitizer.feed(raw_delta)

      if safe_delta:
        safe_parts.append(safe_delta)

      if safe_delta or chunk.delta_tool_calls or chunk.finish_reason:
        yield chunk.model_copy(update={'delta_content': safe_delta})

    tail = sanitizer.flush()

    if tail:
      safe_parts.append(tail)
      yield self._tail_chunk(last_chunk, tail)

    if not from_cache:
      self._cache_stream(cache_key, raw_parts, safe_parts, last_chunk)

  def _sanitize_complete_response(
    self,
    request: GatewayRequest,
    response: GatewayResponse,
  ) -> GatewayResponse:
    '''Sanitize a completion response and persist raw content to cache.

    Args:
      request: Original gateway request.
      response: Raw gateway response.

    Returns:
      Sanitized gateway response.
    '''
    raw_content = response.content or ''
    safe_content = sanitize_think_tokens(raw_content)

    if self._cache is not None and not response.cached:
      cache_key = SQLiteCache.make_key(
        request,
        self._config.cache.include_temperature,
      )

      payload = response.model_dump(mode='json')
      payload['content'] = safe_content
      payload['raw_content'] = raw_content

      self._cache.set(cache_key, payload)

    return response.model_copy(update={'content': safe_content})

  def _cache_key(self, request: GatewayRequest) -> Optional[str]:
    '''Build cache key when caching is enabled.

    Args:
      request: Gateway request.

    Returns:
      Cache key or None.
    '''
    if self._cache is None:
      return None

    return SQLiteCache.make_key(
      request,
      self._config.cache.include_temperature,
    )

  def _cached_chunk(
    self,
    cache_key: Optional[str],
  ) -> Optional[ProviderChunk]:
    '''Build a sanitized cached stream chunk.

    Args:
      cache_key: Cache key.

    Returns:
      Cached chunk or None.
    '''
    if cache_key is None or self._cache is None:
      return None

    cached = self._cache.get(cache_key)

    if cached is None:
      return None

    raw_content = str(
      cached.get('raw_content')
      or cached.get('content')
      or ''
    )
    safe_content = sanitize_think_tokens(raw_content)

    return ProviderChunk(
      provider='cache',
      model=str(cached.get('model') or ''),
      delta_content=safe_content,
      finish_reason='stop',
    )

  def _cache_stream(
    self,
    cache_key: Optional[str],
    raw_parts: List[str],
    safe_parts: List[str],
    last_chunk: Optional[ProviderChunk],
  ) -> None:
    '''Cache streamed output with raw and sanitized content.

    Args:
      cache_key: Cache key.
      raw_parts: Raw streamed text pieces.
      safe_parts: Sanitized streamed text pieces.
      last_chunk: Last observed chunk.
    '''
    if cache_key is None or self._cache is None:
      return

    raw_content = ''.join(raw_parts)
    safe_content = ''.join(safe_parts)

    if not raw_content:
      return

    provider = last_chunk.provider if last_chunk is not None else ''
    model = last_chunk.model if last_chunk is not None else ''

    output_tokens = self._token_counter.count_text(raw_content, model)

    payload = {
      'provider': provider,
      'model': model,
      'alias': '',
      'content': safe_content,
      'raw_content': raw_content,
      'tool_calls': [],
      'usage': {
        'input_tokens': 0,
        'output_tokens': output_tokens,
        'total_tokens': output_tokens,
      },
      'temperature': 0.0,
    }

    self._cache.set(cache_key, payload)

  def _tail_chunk(
    self,
    last_chunk: Optional[ProviderChunk],
    tail: str,
  ) -> ProviderChunk:
    '''Create a chunk for remaining sanitized text.

    Args:
      last_chunk: Last observed chunk.
      tail: Remaining sanitized text.

    Returns:
      Provider chunk.
    '''
    provider = last_chunk.provider if last_chunk is not None else 'gateway'
    model = last_chunk.model if last_chunk is not None else ''

    return ProviderChunk(
      provider=provider,
      model=model,
      delta_content=tail,
      finish_reason=None,
    )
