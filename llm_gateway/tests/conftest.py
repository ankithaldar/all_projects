#!/usr/bin/env python
# -- coding: utf-8 --

'''Shared test fixtures and deterministic dummy provider.'''


from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Iterator

import pytest
import yaml
from llm_gateway.app import LLMGateway
from llm_gateway.errors import RateLimitedProviderError, TransientProviderError
from llm_gateway.providers.base import LLMProvider
from llm_gateway.schemas import (ProviderChunk, ProviderRequest,
                                 ProviderResponse, Usage)


class DummyProvider(LLMProvider):
  '''Deterministic provider used by gateway tests.'''

  def __init__(
    self,
    name: str,
    text: str = 'ok',
    fail_transient: bool = False,
    fail_rate_limited: bool = False,
  ) -> None:
    '''Initialize dummy provider.

    Args:
      name: Provider name.
      text: Response text.
      fail_transient: Whether to raise transient errors.
      fail_rate_limited: Whether to raise rate limit errors.
    '''
    super().__init__(name)
    self.text = text
    self.fail_transient = fail_transient
    self.fail_rate_limited = fail_rate_limited
    self.calls = 0

  def chat(self, request: ProviderRequest) -> ProviderResponse:
    '''Return deterministic response.

    Args:
      request: Provider request.

    Returns:
      Provider response.

    Raises:
      RateLimitedProviderError: When configured to rate limit.
      TransientProviderError: When configured to fail transiently.
    '''
    self.calls += 1

    if self.fail_rate_limited:
      raise RateLimitedProviderError(
        'rate limited',
        provider=self.name,
        status_code=429,
      )

    if self.fail_transient:
      raise TransientProviderError(
        'transient failure',
        provider=self.name,
      )

    return ProviderResponse(
      provider=self.name,
      model=request.model,
      content=self.text,
      tool_calls=[],
      usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
      raw={},
    )

  async def achat(self, request: ProviderRequest) -> ProviderResponse:
    '''Return deterministic async response.

    Args:
      request: Provider request.

    Returns:
      Provider response.
    '''
    return self.chat(request)

  def stream(self, request: ProviderRequest) -> Iterator[ProviderChunk]:
    '''Yield deterministic stream chunk.

    Args:
      request: Provider request.

    Yields:
      One provider chunk.

    Raises:
      RateLimitedProviderError: When configured to rate limit.
      TransientProviderError: When configured to fail transiently.
    '''
    self.calls += 1

    if self.fail_rate_limited:
      raise RateLimitedProviderError(
        'rate limited',
        provider=self.name,
        status_code=429,
      )

    if self.fail_transient:
      raise TransientProviderError(
        'transient failure',
        provider=self.name,
      )

    yield ProviderChunk(
      provider=self.name,
      model=request.model,
      delta_content=self.text,
      finish_reason='stop',
    )

  async def astream(
    self,
    request: ProviderRequest,
  ) -> AsyncIterator[ProviderChunk]:
    '''Yield deterministic async stream chunk.

    Args:
      request: Provider request.

    Yields:
      One provider chunk.

    Raises:
      RateLimitedProviderError: When configured to rate limit.
      TransientProviderError: When configured to fail transiently.
    '''
    self.calls += 1

    if self.fail_rate_limited:
      raise RateLimitedProviderError(
        'rate limited',
        provider=self.name,
        status_code=429,
      )

    if self.fail_transient:
      raise TransientProviderError(
        'transient failure',
        provider=self.name,
      )

    yield ProviderChunk(
      provider=self.name,
      model=request.model,
      delta_content=self.text,
      finish_reason='stop',
    )


@pytest.fixture
def gateway_factory(
  tmp_path: Path,
) -> Callable[[Dict[str, DummyProvider], Dict[str, int], bool], LLMGateway]:
  '''Create gateway instances backed by temporary config and databases.

  Args:
    tmp_path: Pytest temporary directory.

  Returns:
    Factory function.
  '''

  def _make(
    providers: Dict[str, DummyProvider],
    rpm_by_provider: Dict[str, int],
    cache_enabled: bool = False,
  ) -> LLMGateway:
    '''Build a gateway with dummy providers.

    Args:
      providers: Provider instances keyed by provider name.
      rpm_by_provider: RPM limits keyed by provider name.
      cache_enabled: Whether cache should be enabled.

    Returns:
      Configured gateway instance.
    '''
    model_config = {
      'fast': {
        'model': 'dummy-model',
        'temperature': 0.0,
        'max_tokens': 10,
        'cost_per_million_input': 0.0,
        'cost_per_million_output': 0.0,
        'supports_tools': True,
      }
    }

    providers_cfg = {}
    for name in providers:
      providers_cfg[name] = {
        'enabled': True,
        'base_url': 'http://dummy.invalid/v1',
        'api_key_env': '',
        'rpm': rpm_by_provider.get(name, 100),
        'timeout_seconds': 1.0,
        'models': model_config,
      }

    execution_order = [
      {'provider': name, 'alias': 'fast'}
      for name in providers
      if not name.startswith('fallback_')
    ]

    fallback_name = next(
      (name for name in providers if name.startswith('fallback_')),
      None,
    )

    fallback = None
    if fallback_name is not None:
      fallback = {
        'provider': fallback_name,
        'alias': 'fast',
      }

    config = {
      'default_system_prompt': 'test system',
      'retry': {
        'max_attempts': 1,
        'base_delay_seconds': 0.0,
        'max_delay_seconds': 0.0,
      },
      'cache': {
        'enabled': cache_enabled,
        'ttl_seconds': 60,
        'include_temperature': False,
      },
      'providers': providers_cfg,
      'execution_order': execution_order,
      'fallback': fallback,
    }

    config_path = tmp_path / 'gateway.yaml'
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')

    env_path = tmp_path / '.env'
    env_path.write_text('', encoding='utf-8')

    return LLMGateway(
      config_path=str(config_path),
      env_path=str(env_path),
      db_path=tmp_path / 'logs.db',
      cache_path=tmp_path / 'cache.db',
      provider_overrides=providers,
    )

  return _make
