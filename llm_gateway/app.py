#!/usr/bin/env python
# -- coding: utf-8 --

'''Core gateway orchestration.'''


from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from dotenv import load_dotenv

from llm_gateway.cache import SQLiteCache
from llm_gateway.config import (ExecutionStep, GatewayConfig, ModelConfig,
                                load_gateway_config)
from llm_gateway.db_logger import DBLogger
from llm_gateway.errors import (AllProvidersFailedError, ProviderError,
                                RateLimitedProviderError,
                                TransientProviderError)
from llm_gateway.key_rotator import APIKeyRotator
from llm_gateway.providers.openai_compatible import (BytezProvider,
                                                     CerebrasProvider,
                                                     GithubProvider,
                                                     GroqProvider,
                                                     NvidiaProvider,
                                                     OllamaProvider,
                                                     OpenAICompatibleProvider,
                                                     OpenRouterProvider)
from llm_gateway.rate_limiter import RPMRateLimiter
from llm_gateway.schemas import (ChatMessage, GatewayRequest, GatewayResponse,
                                 LogRecord, ProviderChunk, ProviderRequest,
                                 ProviderResponse, ToolCall, Usage)
from llm_gateway.token_counter import TokenCounter


class LLMGateway:
  '''Central LLM routing gateway.'''

  def __init__(
    self,
    config_path: str | Path = 'config/gateway.yaml',
    env_path: str | Path = '.env',
    db_path: Optional[str | Path] = None,
    cache_path: Optional[str | Path] = None,
    provider_overrides: Optional[Dict[str, Any]] = None,
  ) -> None:
    '''Initialize gateway.

    Args:
      config_path: Path to YAML configuration.
      env_path: Path to gateway-specific .env file.
      db_path: Optional override for logging database path.
      cache_path: Optional override for cache database path.
      provider_overrides: Optional provider instances, primarily for tests.
    '''
    env_path = Path(env_path)
    if env_path.exists():
      load_dotenv(env_path, override=True)

    self._config: GatewayConfig = load_gateway_config(config_path)

    resolved_db_path = Path(
      db_path
      or os.getenv('GATEWAY_DB_PATH', './data/gateway.db')
    )
    resolved_cache_path = Path(
      cache_path
      or os.getenv('GATEWAY_CACHE_PATH', './data/cache.db')
    )

    self._db_logger = DBLogger(resolved_db_path)
    self._token_counter = TokenCounter()

    self._cache: Optional[SQLiteCache] = None
    if self._config.cache.enabled:
      self._cache = SQLiteCache(
        resolved_cache_path,
        ttl_seconds=self._config.cache.ttl_seconds,
      )

    self._rate_limiters: Dict[str, RPMRateLimiter] = {
      name: RPMRateLimiter(provider.rpm)
      for name, provider in self._config.providers.items()
    }

    self._providers = self._build_providers(provider_overrides or {})

  def close(self) -> None:
    '''Close background logger.'''
    self._db_logger.close()

  def complete(self, request: GatewayRequest) -> GatewayResponse:
    '''Execute synchronous routing and completion.

    Args:
      request: Gateway request.

    Returns:
      Gateway response.

    Raises:
      AllProvidersFailedError: If every provider fails.
    '''
    cache_key: Optional[str] = None

    if self._cache is not None:
      cache_key = SQLiteCache.make_key(
        request,
        self._config.cache.include_temperature,
      )
      cached = self._cache.get(cache_key)
      if cached is not None:
        return self._response_from_cache(cached, request)

    for step in self._config.execution_order:
      response = self._execute_step(step, request, is_fallback=False)
      if response is not None:
        if self._cache is not None and cache_key is not None:
          self._cache.set(cache_key, response.model_dump(mode='json'))
        return response

    if self._config.fallback is not None:
      response = self._execute_step(
        self._config.fallback,
        request,
        is_fallback=True,
      )
      if response is not None:
        if self._cache is not None and cache_key is not None:
          self._cache.set(cache_key, response.model_dump(mode='json'))
        return response

    raise AllProvidersFailedError('all providers failed or were rate limited')

  async def acomplete(self, request: GatewayRequest) -> GatewayResponse:
    '''Execute asynchronous routing and completion.

    Args:
      request: Gateway request.

    Returns:
      Gateway response.

    Raises:
      AllProvidersFailedError: If every provider fails.
    '''
    cache_key: Optional[str] = None

    if self._cache is not None:
      cache_key = SQLiteCache.make_key(
        request,
        self._config.cache.include_temperature,
      )
      cached = self._cache.get(cache_key)
      if cached is not None:
        return self._response_from_cache(cached, request)

    for step in self._config.execution_order:
      response = await self._execute_step_async(
        step,
        request,
        is_fallback=False,
      )
      if response is not None:
        if self._cache is not None and cache_key is not None:
          self._cache.set(cache_key, response.model_dump(mode='json'))
        return response

    if self._config.fallback is not None:
      response = await self._execute_step_async(
        self._config.fallback,
        request,
        is_fallback=True,
      )
      if response is not None:
        if self._cache is not None and cache_key is not None:
          self._cache.set(cache_key, response.model_dump(mode='json'))
        return response

    raise AllProvidersFailedError('all providers failed or were rate limited')

  def stream(self, request: GatewayRequest) -> Iterator[ProviderChunk]:
    '''Execute synchronous streaming routing.

    Args:
      request: Gateway request.

    Yields:
      Streaming chunks.

    Raises:
      AllProvidersFailedError: If every provider fails.
    '''
    if self._cache is not None:
      cache_key = SQLiteCache.make_key(
        request,
        self._config.cache.include_temperature,
      )
      cached = self._cache.get(cache_key)
      if cached is not None:
        yield ProviderChunk(
          provider='cache',
          model=str(cached.get('model') or ''),
          delta_content=str(cached.get('content') or ''),
          finish_reason='stop',
        )
        return

    for step in self._config.execution_order:
      generator = self._stream_step(step, request, is_fallback=False)
      if generator is not None:
        yield from generator
        return

    if self._config.fallback is not None:
      generator = self._stream_step(
        self._config.fallback,
        request,
        is_fallback=True,
      )
      if generator is not None:
        yield from generator
        return

    raise AllProvidersFailedError('all providers failed or were rate limited')

  async def astream(
    self,
    request: GatewayRequest,
  ) -> AsyncIterator[ProviderChunk]:
    '''Execute asynchronous streaming routing.

    Args:
      request: Gateway request.

    Yields:
      Streaming chunks.

    Raises:
      AllProvidersFailedError: If every provider fails.
    '''
    if self._cache is not None:
      cache_key = SQLiteCache.make_key(
        request,
        self._config.cache.include_temperature,
      )
      cached = self._cache.get(cache_key)
      if cached is not None:
        yield ProviderChunk(
          provider='cache',
          model=str(cached.get('model') or ''),
          delta_content=str(cached.get('content') or ''),
          finish_reason='stop',
        )
        return

    for step in self._config.execution_order:
      generator = await self._stream_step_async(
        step,
        request,
        is_fallback=False,
      )
      if generator is not None:
        async for chunk in generator:
          yield chunk
        return

    if self._config.fallback is not None:
      generator = await self._stream_step_async(
        self._config.fallback,
        request,
        is_fallback=True,
      )
      if generator is not None:
        async for chunk in generator:
          yield chunk
        return

    raise AllProvidersFailedError('all providers failed or were rate limited')

  def _build_providers(
    self,
    overrides: Dict[str, Any],
  ) -> Dict[str, Any]:
    '''Build provider instances from configuration.

    Args:
      overrides: Optional injected provider instances.

    Returns:
      Mapping of provider name to provider instance.
    '''
    registry = {
      'bytez': BytezProvider,
      'openrouter': OpenRouterProvider,
      'groq': GroqProvider,
      'cerebras': CerebrasProvider,
      'nvidia': NvidiaProvider,
      'github': GithubProvider,
      'ollama': OllamaProvider,
    }

    providers: Dict[str, Any] = {}

    for name, provider_config in self._config.providers.items():
      if not provider_config.enabled:
        continue

      if name in overrides:
        providers[name] = overrides[name]
        continue

      rotator = APIKeyRotator(self._keys_from_env(provider_config.api_key_env))
      provider_class = registry.get(name, OpenAICompatibleProvider)

      providers[name] = provider_class(
        name=name,
        config=provider_config,
        rotator=rotator,
      )

    return providers

  def _keys_from_env(self, env_name: str) -> List[str]:
    '''Read comma-separated keys from environment.

    Args:
      env_name: Environment variable name.

    Returns:
      List of API keys.
    '''
    if not env_name:
      return []

    raw = os.getenv(env_name, '')
    return [part.strip() for part in raw.split(',') if part.strip()]

  def _execute_step(
    self,
    step: ExecutionStep,
    request: GatewayRequest,
    is_fallback: bool,
  ) -> Optional[GatewayResponse]:
    '''Execute one synchronous provider step with retries.

    Args:
      step: Execution step.
      request: Gateway request.
      is_fallback: Whether this step is the ultimate fallback.

    Returns:
      Gateway response or None when the step should be skipped/failed.
    '''
    provider = self._providers.get(step.provider)
    if provider is None:
      return None

    try:
      model_cfg = self._config.providers[step.provider].models[step.alias]
    except KeyError:
      return None

    limiter = self._rate_limiters.get(step.provider)
    if limiter is not None and not is_fallback and not limiter.allow():
      self._log_skipped(step, request, model_cfg)
      return None

    try:
      provider_request = self._provider_request(step, request, model_cfg)
    except Exception as exc:
      self._log_call(
        provider=step.provider,
        model=model_cfg.model,
        input_tokens=0,
        output_tokens=0,
        latency_ms=0.0,
        status='error',
        error=str(exc),
        prompt_chars=len(request.prompt or ''),
        response_chars=0,
        cost=0.0,
        temperature=model_cfg.temperature,
        system_prompt=request.system_prompt or self._config.default_system_prompt,
        session_id=request.session_id,
      )
      return None

    input_tokens = self._token_counter.count_messages(
      provider_request.messages,
      model_cfg.model,
    )
    system_prompt = self._extract_system_prompt(provider_request.messages)
    prompt_chars = self._prompt_characters(provider_request.messages)
    attempts = max(1, self._config.retry.max_attempts)

    for attempt in range(attempts):
      started = time.perf_counter()

      try:
        provider_response: ProviderResponse = provider.chat(provider_request)
        latency_ms = (time.perf_counter() - started) * 1000.0

        output_tokens = provider_response.usage.output_tokens
        if not output_tokens:
          output_tokens = self._token_counter.count_text(
            provider_response.content,
            model_cfg.model,
          )

        if provider_response.usage.input_tokens:
          input_tokens = provider_response.usage.input_tokens

        cost = self._estimate_cost(model_cfg, input_tokens, output_tokens)
        response_chars = self._response_characters(
          provider_response.content,
          provider_response.tool_calls,
        )

        response = GatewayResponse(
          provider=step.provider,
          model=provider_response.model,
          alias=step.alias,
          content=provider_response.content,
          tool_calls=provider_response.tool_calls,
          usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
          ),
          cached=False,
          latency_ms=latency_ms,
          cost=cost,
          temperature=provider_request.temperature,
          system_prompt=system_prompt,
          prompt_chars=prompt_chars,
          response_chars=response_chars,
          session_id=request.session_id,
        )

        self._log_call(
          provider=step.provider,
          model=provider_response.model,
          input_tokens=input_tokens,
          output_tokens=output_tokens,
          latency_ms=latency_ms,
          status='success',
          error=None,
          prompt_chars=prompt_chars,
          response_chars=response_chars,
          cost=cost,
          temperature=provider_request.temperature,
          system_prompt=system_prompt,
          session_id=request.session_id,
        )

        return response

      except RateLimitedProviderError as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._log_call(
          provider=step.provider,
          model=model_cfg.model,
          input_tokens=input_tokens,
          output_tokens=0,
          latency_ms=latency_ms,
          status='rate_limited',
          error=str(exc),
          prompt_chars=prompt_chars,
          response_chars=0,
          cost=0.0,
          temperature=provider_request.temperature,
          system_prompt=system_prompt,
          session_id=request.session_id,
        )
        return None

      except TransientProviderError as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._log_call(
          provider=step.provider,
          model=model_cfg.model,
          input_tokens=input_tokens,
          output_tokens=0,
          latency_ms=latency_ms,
          status='error',
          error=str(exc),
          prompt_chars=prompt_chars,
          response_chars=0,
          cost=0.0,
          temperature=provider_request.temperature,
          system_prompt=system_prompt,
          session_id=request.session_id,
        )

        if attempt + 1 < attempts:
          time.sleep(self._backoff_delay(attempt))

        continue

      except ProviderError as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._log_call(
          provider=step.provider,
          model=model_cfg.model,
          input_tokens=input_tokens,
          output_tokens=0,
          latency_ms=latency_ms,
          status='error',
          error=str(exc),
          prompt_chars=prompt_chars,
          response_chars=0,
          cost=0.0,
          temperature=provider_request.temperature,
          system_prompt=system_prompt,
          session_id=request.session_id,
        )
        return None

      except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._log_call(
          provider=step.provider,
          model=model_cfg.model,
          input_tokens=input_tokens,
          output_tokens=0,
          latency_ms=latency_ms,
          status='error',
          error=str(exc),
          prompt_chars=prompt_chars,
          response_chars=0,
          cost=0.0,
          temperature=provider_request.temperature,
          system_prompt=system_prompt,
          session_id=request.session_id,
        )
        return None

    return None

  async def _execute_step_async(
    self,
    step: ExecutionStep,
    request: GatewayRequest,
    is_fallback: bool,
  ) -> Optional[GatewayResponse]:
    '''Execute one asynchronous provider step with retries.

    Args:
      step: Execution step.
      request: Gateway request.
      is_fallback: Whether this step is the ultimate fallback.

    Returns:
      Gateway response or None when the step should be skipped/failed.
    '''
    provider = self._providers.get(step.provider)
    if provider is None:
      return None

    try:
      model_cfg = self._config.providers[step.provider].models[step.alias]
    except KeyError:
      return None

    limiter = self._rate_limiters.get(step.provider)
    if limiter is not None and not is_fallback and not limiter.allow():
      self._log_skipped(step, request, model_cfg)
      return None

    try:
      provider_request = self._provider_request(step, request, model_cfg)
    except Exception as exc:
      self._log_call(
        provider=step.provider,
        model=model_cfg.model,
        input_tokens=0,
        output_tokens=0,
        latency_ms=0.0,
        status='error',
        error=str(exc),
        prompt_chars=len(request.prompt or ''),
        response_chars=0,
        cost=0.0,
        temperature=model_cfg.temperature,
        system_prompt=request.system_prompt or self._config.default_system_prompt,
        session_id=request.session_id,
      )
      return None

    input_tokens = self._token_counter.count_messages(
      provider_request.messages,
      model_cfg.model,
    )
    system_prompt = self._extract_system_prompt(provider_request.messages)
    prompt_chars = self._prompt_characters(provider_request.messages)
    attempts = max(1, self._config.retry.max_attempts)

    for attempt in range(attempts):
      started = time.perf_counter()

      try:
        provider_response = await provider.achat(provider_request)
        latency_ms = (time.perf_counter() - started) * 1000.0

        output_tokens = provider_response.usage.output_tokens
        if not output_tokens:
          output_tokens = self._token_counter.count_text(
            provider_response.content,
            model_cfg.model,
          )

        if provider_response.usage.input_tokens:
          input_tokens = provider_response.usage.input_tokens

        cost = self._estimate_cost(model_cfg, input_tokens, output_tokens)
        response_chars = self._response_characters(
          provider_response.content,
          provider_response.tool_calls,
        )

        response = GatewayResponse(
          provider=step.provider,
          model=provider_response.model,
          alias=step.alias,
          content=provider_response.content,
          tool_calls=provider_response.tool_calls,
          usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
          ),
          cached=False,
          latency_ms=latency_ms,
          cost=cost,
          temperature=provider_request.temperature,
          system_prompt=system_prompt,
          prompt_chars=prompt_chars,
          response_chars=response_chars,
          session_id=request.session_id,
        )

        self._log_call(
          provider=step.provider,
          model=provider_response.model,
          input_tokens=input_tokens,
          output_tokens=output_tokens,
          latency_ms=latency_ms,
          status='success',
          error=None,
          prompt_chars=prompt_chars,
          response_chars=response_chars,
          cost=cost,
          temperature=provider_request.temperature,
          system_prompt=system_prompt,
          session_id=request.session_id,
        )

        return response

      except RateLimitedProviderError as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._log_call(
          provider=step.provider,
          model=model_cfg.model,
          input_tokens=input_tokens,
          output_tokens=0,
          latency_ms=latency_ms,
          status='rate_limited',
          error=str(exc),
          prompt_chars=prompt_chars,
          response_chars=0,
          cost=0.0,
          temperature=provider_request.temperature,
          system_prompt=system_prompt,
          session_id=request.session_id,
        )
        return None

      except TransientProviderError as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._log_call(
          provider=step.provider,
          model=model_cfg.model,
          input_tokens=input_tokens,
          output_tokens=0,
          latency_ms=latency_ms,
          status='error',
          error=str(exc),
          prompt_chars=prompt_chars,
          response_chars=0,
          cost=0.0,
          temperature=provider_request.temperature,
          system_prompt=system_prompt,
          session_id=request.session_id,
        )

        if attempt + 1 < attempts:
          await asyncio.sleep(self._backoff_delay(attempt))

        continue

      except ProviderError as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._log_call(
          provider=step.provider,
          model=model_cfg.model,
          input_tokens=input_tokens,
          output_tokens=0,
          latency_ms=latency_ms,
          status='error',
          error=str(exc),
          prompt_chars=prompt_chars,
          response_chars=0,
          cost=0.0,
          temperature=provider_request.temperature,
          system_prompt=system_prompt,
          session_id=request.session_id,
        )
        return None

      except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._log_call(
          provider=step.provider,
          model=model_cfg.model,
          input_tokens=input_tokens,
          output_tokens=0,
          latency_ms=latency_ms,
          status='error',
          error=str(exc),
          prompt_chars=prompt_chars,
          response_chars=0,
          cost=0.0,
          temperature=provider_request.temperature,
          system_prompt=system_prompt,
          session_id=request.session_id,
        )
        return None

    return None

  def _stream_step(
    self,
    step: ExecutionStep,
    request: GatewayRequest,
    is_fallback: bool,
  ) -> Optional[Iterator[ProviderChunk]]:
    '''Prepare a synchronous streaming step.

    Args:
      step: Execution step.
      request: Gateway request.
      is_fallback: Whether this step is the ultimate fallback.

    Returns:
      Streaming generator or None when the step should be skipped/failed.
    '''
    provider = self._providers.get(step.provider)
    if provider is None:
      return None

    try:
      model_cfg = self._config.providers[step.provider].models[step.alias]
    except KeyError:
      return None

    limiter = self._rate_limiters.get(step.provider)
    if limiter is not None and not is_fallback and not limiter.allow():
      self._log_skipped(step, request, model_cfg)
      return None

    try:
      provider_request = self._provider_request(
        step,
        request,
        model_cfg,
        stream=True,
      )
    except Exception as exc:
      self._log_call(
        provider=step.provider,
        model=model_cfg.model,
        input_tokens=0,
        output_tokens=0,
        latency_ms=0.0,
        status='error',
        error=str(exc),
        prompt_chars=len(request.prompt or ''),
        response_chars=0,
        cost=0.0,
        temperature=model_cfg.temperature,
        system_prompt=request.system_prompt or self._config.default_system_prompt,
        session_id=request.session_id,
      )
      return None

    input_tokens = self._token_counter.count_messages(
      provider_request.messages,
      model_cfg.model,
    )
    attempts = max(1, self._config.retry.max_attempts)

    for attempt in range(attempts):
      started = time.perf_counter()

      try:
        iterator = provider.stream(provider_request)
        first = next(iterator, None)
      except RateLimitedProviderError as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._log_call(
          provider=step.provider,
          model=model_cfg.model,
          input_tokens=input_tokens,
          output_tokens=0,
          latency_ms=latency_ms,
          status='rate_limited',
          error=str(exc),
          prompt_chars=self._prompt_characters(provider_request.messages),
          response_chars=0,
          cost=0.0,
          temperature=provider_request.temperature,
          system_prompt=self._extract_system_prompt(provider_request.messages),
          session_id=request.session_id,
        )
        return None

      except TransientProviderError as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._log_call(
          provider=step.provider,
          model=model_cfg.model,
          input_tokens=input_tokens,
          output_tokens=0,
          latency_ms=latency_ms,
          status='error',
          error=str(exc),
          prompt_chars=self._prompt_characters(provider_request.messages),
          response_chars=0,
          cost=0.0,
          temperature=provider_request.temperature,
          system_prompt=self._extract_system_prompt(provider_request.messages),
          session_id=request.session_id,
        )

        if attempt + 1 < attempts:
          time.sleep(self._backoff_delay(attempt))

        continue

      except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._log_call(
          provider=step.provider,
          model=model_cfg.model,
          input_tokens=input_tokens,
          output_tokens=0,
          latency_ms=latency_ms,
          status='error',
          error=str(exc),
          prompt_chars=self._prompt_characters(provider_request.messages),
          response_chars=0,
          cost=0.0,
          temperature=provider_request.temperature,
          system_prompt=self._extract_system_prompt(provider_request.messages),
          session_id=request.session_id,
        )
        return None

      return self._wrap_sync_stream(
        iterator=iterator,
        first=first,
        step=step,
        provider_request=provider_request,
        model_cfg=model_cfg,
        input_tokens=input_tokens,
        started=started,
        request=request,
      )

    return None

  async def _stream_step_async(
    self,
    step: ExecutionStep,
    request: GatewayRequest,
    is_fallback: bool,
  ) -> Optional[AsyncIterator[ProviderChunk]]:
    '''Prepare an asynchronous streaming step.

    Args:
      step: Execution step.
      request: Gateway request.
      is_fallback: Whether this step is the ultimate fallback.

    Returns:
      Streaming async generator or None when skipped/failed.
    '''
    provider = self._providers.get(step.provider)
    if provider is None:
      return None

    try:
      model_cfg = self._config.providers[step.provider].models[step.alias]
    except KeyError:
      return None

    limiter = self._rate_limiters.get(step.provider)
    if limiter is not None and not is_fallback and not limiter.allow():
      self._log_skipped(step, request, model_cfg)
      return None

    try:
      provider_request = self._provider_request(
        step,
        request,
        model_cfg,
        stream=True,
      )
    except Exception as exc:
      self._log_call(
        provider=step.provider,
        model=model_cfg.model,
        input_tokens=0,
        output_tokens=0,
        latency_ms=0.0,
        status='error',
        error=str(exc),
        prompt_chars=len(request.prompt or ''),
        response_chars=0,
        cost=0.0,
        temperature=model_cfg.temperature,
        system_prompt=request.system_prompt or self._config.default_system_prompt,
        session_id=request.session_id,
      )
      return None

    input_tokens = self._token_counter.count_messages(
      provider_request.messages,
      model_cfg.model,
    )
    attempts = max(1, self._config.retry.max_attempts)

    for attempt in range(attempts):
      started = time.perf_counter()

      try:
        iterator = provider.astream(provider_request)
        first = await anext(iterator, None)
      except RateLimitedProviderError as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._log_call(
          provider=step.provider,
          model=model_cfg.model,
          input_tokens=input_tokens,
          output_tokens=0,
          latency_ms=latency_ms,
          status='rate_limited',
          error=str(exc),
          prompt_chars=self._prompt_characters(provider_request.messages),
          response_chars=0,
          cost=0.0,
          temperature=provider_request.temperature,
          system_prompt=self._extract_system_prompt(provider_request.messages),
          session_id=request.session_id,
        )
        return None

      except TransientProviderError as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._log_call(
          provider=step.provider,
          model=model_cfg.model,
          input_tokens=input_tokens,
          output_tokens=0,
          latency_ms=latency_ms,
          status='error',
          error=str(exc),
          prompt_chars=self._prompt_characters(provider_request.messages),
          response_chars=0,
          cost=0.0,
          temperature=provider_request.temperature,
          system_prompt=self._extract_system_prompt(provider_request.messages),
          session_id=request.session_id,
        )

        if attempt + 1 < attempts:
          await asyncio.sleep(self._backoff_delay(attempt))

        continue

      except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._log_call(
          provider=step.provider,
          model=model_cfg.model,
          input_tokens=input_tokens,
          output_tokens=0,
          latency_ms=latency_ms,
          status='error',
          error=str(exc),
          prompt_chars=self._prompt_characters(provider_request.messages),
          response_chars=0,
          cost=0.0,
          temperature=provider_request.temperature,
          system_prompt=self._extract_system_prompt(provider_request.messages),
          session_id=request.session_id,
        )
        return None

      return self._wrap_async_stream(
        iterator=iterator,
        first=first,
        step=step,
        provider_request=provider_request,
        model_cfg=model_cfg,
        input_tokens=input_tokens,
        started=started,
        request=request,
      )

    return None

  def _wrap_sync_stream(
    self,
    iterator: Iterator[ProviderChunk],
    first: Optional[ProviderChunk],
    step: ExecutionStep,
    provider_request: ProviderRequest,
    model_cfg: ModelConfig,
    input_tokens: int,
    started: float,
    request: GatewayRequest,
  ) -> Iterator[ProviderChunk]:
    '''Wrap synchronous provider stream and log final metrics.

    Args:
      iterator: Provider stream iterator.
      first: First already-consumed chunk, if any.
      step: Execution step.
      provider_request: Provider request.
      model_cfg: Model configuration.
      input_tokens: Estimated input tokens.
      started: Start timestamp.
      request: Original gateway request.

    Yields:
      Streaming chunks.
    '''
    content_parts: List[str] = []
    tool_calls: List[ToolCall] = []
    system_prompt = self._extract_system_prompt(provider_request.messages)
    prompt_chars = self._prompt_characters(provider_request.messages)

    def generator() -> Iterator[ProviderChunk]:
      '''Inner generator that tracks streamed output.

      Yields:
        Streaming chunks.
      '''
      try:
        if first is not None:
          if first.delta_content:
            content_parts.append(first.delta_content)
          tool_calls.extend(first.delta_tool_calls)
          yield first

        for chunk in iterator:
          if chunk.delta_content:
            content_parts.append(chunk.delta_content)
          tool_calls.extend(chunk.delta_tool_calls)
          yield chunk

        latency_ms = (time.perf_counter() - started) * 1000.0
        content = ''.join(content_parts)

        output_tokens = self._token_counter.count_text(content, model_cfg.model)
        if not content and tool_calls:
          output_tokens = self._token_counter.count_text(
            json.dumps([call.model_dump() for call in tool_calls], default=str),
            model_cfg.model,
          )

        response_chars = self._response_characters(content, tool_calls)
        cost = self._estimate_cost(model_cfg, input_tokens, output_tokens)

        self._log_call(
          provider=step.provider,
          model=provider_request.model,
          input_tokens=input_tokens,
          output_tokens=output_tokens,
          latency_ms=latency_ms,
          status='success',
          error=None,
          prompt_chars=prompt_chars,
          response_chars=response_chars,
          cost=cost,
          temperature=provider_request.temperature,
          system_prompt=system_prompt,
          session_id=request.session_id,
        )
      except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        content = ''.join(content_parts)
        response_chars = self._response_characters(content, tool_calls)

        self._log_call(
          provider=step.provider,
          model=provider_request.model,
          input_tokens=input_tokens,
          output_tokens=0,
          latency_ms=latency_ms,
          status='error',
          error=str(exc),
          prompt_chars=prompt_chars,
          response_chars=response_chars,
          cost=0.0,
          temperature=provider_request.temperature,
          system_prompt=system_prompt,
          session_id=request.session_id,
        )
        raise

    return generator()

  def _wrap_async_stream(
    self,
    iterator: AsyncIterator[ProviderChunk],
    first: Optional[ProviderChunk],
    step: ExecutionStep,
    provider_request: ProviderRequest,
    model_cfg: ModelConfig,
    input_tokens: int,
    started: float,
    request: GatewayRequest,
  ) -> AsyncIterator[ProviderChunk]:
    '''Wrap asynchronous provider stream and log final metrics.

    Args:
      iterator: Provider async stream iterator.
      first: First already-consumed chunk, if any.
      step: Execution step.
      provider_request: Provider request.
      model_cfg: Model configuration.
      input_tokens: Estimated input tokens.
      started: Start timestamp.
      request: Original gateway request.

    Returns:
      Async generator of chunks.
    '''
    content_parts: List[str] = []
    tool_calls: List[ToolCall] = []
    system_prompt = self._extract_system_prompt(provider_request.messages)
    prompt_chars = self._prompt_characters(provider_request.messages)

    async def generator() -> AsyncIterator[ProviderChunk]:
      '''Inner async generator that tracks streamed output.

      Yields:
        Streaming chunks.
      '''
      try:
        if first is not None:
          if first.delta_content:
            content_parts.append(first.delta_content)
          tool_calls.extend(first.delta_tool_calls)
          yield first

        async for chunk in iterator:
          if chunk.delta_content:
            content_parts.append(chunk.delta_content)
          tool_calls.extend(chunk.delta_tool_calls)
          yield chunk

        latency_ms = (time.perf_counter() - started) * 1000.0
        content = ''.join(content_parts)

        output_tokens = self._token_counter.count_text(content, model_cfg.model)
        if not content and tool_calls:
          output_tokens = self._token_counter.count_text(
            json.dumps([call.model_dump() for call in tool_calls], default=str),
            model_cfg.model,
          )

        response_chars = self._response_characters(content, tool_calls)
        cost = self._estimate_cost(model_cfg, input_tokens, output_tokens)

        self._log_call(
          provider=step.provider,
          model=provider_request.model,
          input_tokens=input_tokens,
          output_tokens=output_tokens,
          latency_ms=latency_ms,
          status='success',
          error=None,
          prompt_chars=prompt_chars,
          response_chars=response_chars,
          cost=cost,
          temperature=provider_request.temperature,
          system_prompt=system_prompt,
          session_id=request.session_id,
        )
      except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        content = ''.join(content_parts)
        response_chars = self._response_characters(content, tool_calls)

        self._log_call(
          provider=step.provider,
          model=provider_request.model,
          input_tokens=input_tokens,
          output_tokens=0,
          latency_ms=latency_ms,
          status='error',
          error=str(exc),
          prompt_chars=prompt_chars,
          response_chars=response_chars,
          cost=0.0,
          temperature=provider_request.temperature,
          system_prompt=system_prompt,
          session_id=request.session_id,
        )
        raise

    return generator()

  def _provider_request(
    self,
    step: ExecutionStep,
    request: GatewayRequest,
    model_cfg: ModelConfig,
    stream: bool = False,
  ) -> ProviderRequest:
    '''Build a provider request from gateway request and model config.

    Args:
      step: Execution step.
      request: Gateway request.
      model_cfg: Model configuration.
      stream: Whether streaming is requested.

    Returns:
      Provider request.
    '''
    messages = request.build_messages(self._config.default_system_prompt)

    temperature = request.temperature
    if temperature is None:
      temperature = model_cfg.temperature

    max_tokens = request.max_tokens
    if max_tokens is None:
      max_tokens = model_cfg.max_tokens

    return ProviderRequest(
      provider=step.provider,
      model=model_cfg.model,
      messages=messages,
      tools=request.tools,
      tool_choice=request.tool_choice,
      temperature=temperature,
      max_tokens=max_tokens,
      stream=stream,
    )

  def _response_from_cache(
    self,
    cached: Dict[str, Any],
    request: GatewayRequest,
  ) -> GatewayResponse:
    '''Build gateway response from cache.

    Args:
      cached: Cached payload.
      request: Original gateway request.

    Returns:
      Gateway response marked as cached.
    '''
    usage = Usage.model_validate(cached.get('usage') or {})
    messages = request.build_messages(self._config.default_system_prompt)
    system_prompt = self._extract_system_prompt(messages)
    prompt_chars = self._prompt_characters(messages)
    content = str(cached.get('content') or '')
    response_chars = len(content)

    return GatewayResponse(
      request_id=str(uuid.uuid4()),
      provider='cache',
      model=str(cached.get('model') or ''),
      alias=str(cached.get('alias') or ''),
      content=content,
      tool_calls=cached.get('tool_calls') or [],
      usage=usage,
      cached=True,
      latency_ms=0.0,
      cost=0.0,
      temperature=float(cached.get('temperature') or 0.0),
      system_prompt=system_prompt,
      prompt_chars=prompt_chars,
      response_chars=response_chars,
      session_id=request.session_id,
    )

  def _estimate_cost(
    self,
    model_cfg: ModelConfig,
    input_tokens: int,
    output_tokens: int,
  ) -> float:
    '''Estimate call cost from token usage.

    Args:
      model_cfg: Model configuration.
      input_tokens: Input token count.
      output_tokens: Output token count.

    Returns:
      Estimated cost.
    '''
    input_cost = input_tokens * model_cfg.cost_per_million_input
    output_cost = output_tokens * model_cfg.cost_per_million_output
    return (input_cost + output_cost) / 1_000_000.0

  def _backoff_delay(self, attempt: int) -> float:
    '''Calculate exponential backoff delay.

    Args:
      attempt: Zero-based attempt number.

    Returns:
      Delay in seconds.
    '''
    base = self._config.retry.base_delay_seconds
    maximum = self._config.retry.max_delay_seconds
    delay = min(maximum, base * (2 ** attempt))
    return delay + random.uniform(0.0, 0.05)

  def _extract_system_prompt(self, messages: List[ChatMessage]) -> str:
    '''Extract system prompt from messages.

    Args:
      messages: Chat messages.

    Returns:
      System prompt or empty string.
    '''
    for message in messages:
      if message.role == 'system':
        return message.content or ''

    return ''

  def _prompt_characters(self, messages: List[ChatMessage]) -> int:
    '''Count prompt characters including tool call payloads.

    Args:
      messages: Chat messages.

    Returns:
      Character count.
    '''
    chars = 0

    for message in messages:
      if message.content:
        chars += len(message.content)

      if message.tool_calls:
        chars += len(
          json.dumps(
            [call.model_dump() for call in message.tool_calls],
            default=str,
          )
        )

    return chars

  def _response_characters(
    self,
    content: str,
    tool_calls: List[ToolCall],
  ) -> int:
    '''Count response characters including tool call payloads.

    Args:
      content: Response content.
      tool_calls: Response tool calls.

    Returns:
      Character count.
    '''
    chars = len(content or '')

    if tool_calls:
      chars += len(
        json.dumps(
          [call.model_dump() for call in tool_calls],
          default=str,
        )
      )

    return chars

  def _log_skipped(
    self,
    step: ExecutionStep,
    request: GatewayRequest,
    model_cfg: ModelConfig,
  ) -> None:
    '''Log a locally rate-limited skip.

    Args:
      step: Execution step.
      request: Gateway request.
      model_cfg: Model configuration.
    '''
    self._log_call(
      provider=step.provider,
      model=model_cfg.model,
      input_tokens=0,
      output_tokens=0,
      latency_ms=0.0,
      status='rate_limited_skipped',
      error='local RPM limit reached',
      prompt_chars=len(request.prompt or ''),
      response_chars=0,
      cost=0.0,
      temperature=model_cfg.temperature,
      system_prompt=request.system_prompt or self._config.default_system_prompt,
      session_id=request.session_id,
    )

  def _log_call(
    self,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    status: str,
    error: Optional[str],
    prompt_chars: int,
    response_chars: int,
    cost: float,
    temperature: float,
    system_prompt: str,
    session_id: Optional[str],
  ) -> None:
    '''Send a log record to background persistence.

    Args:
      provider: Provider name.
      model: Model used.
      input_tokens: Input tokens.
      output_tokens: Output tokens.
      latency_ms: Latency in milliseconds.
      status: Call status.
      error: Error message if any.
      prompt_chars: Prompt character count.
      response_chars: Response character count.
      cost: Estimated cost.
      temperature: Temperature used.
      system_prompt: System prompt used.
      session_id: Optional session id.
    '''
    record = LogRecord(
      provider=provider,
      model_used=model,
      input_tokens=input_tokens,
      output_tokens=output_tokens,
      latency_ms=latency_ms,
      status=status,
      error=error or None,
      prompt_chars=prompt_chars,
      response_chars=response_chars,
      cost=cost,
      temperature=temperature,
      system_prompt=system_prompt,
      session_id=session_id,
    )

    self._db_logger.log(record)
