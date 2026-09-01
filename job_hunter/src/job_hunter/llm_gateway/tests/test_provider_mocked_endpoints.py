#!/usr/bin/env python
# -- coding: utf-8 --

'''Provider tests with mocked HTTP endpoints.'''

from __future__ import annotations

import asyncio
from typing import Any, Dict

import httpx
from llm_gateway.config import ModelConfig, ProviderConfig
from llm_gateway.providers.openai_compatible import OpenAICompatibleProvider
from llm_gateway.schemas import ChatMessage, ProviderRequest


class FakeResponse:
  '''Fake requests response.'''

  def __init__(self, status_code: int, json_data: Dict[str, Any]) -> None:
    '''Initialize fake response.

    Args:
      status_code: HTTP status code.
      json_data: JSON payload.
    '''
    self.status_code = status_code
    self._json_data = json_data

  def json(self) -> Dict[str, Any]:
    '''Return JSON payload.

    Returns:
      JSON dictionary.
    '''
    return self._json_data

  @property
  def text(self) -> str:
    '''Return text representation.

    Returns:
      Stringified payload.
    '''
    return str(self._json_data)


class FakeSession:
  '''Fake requests session.'''

  def post(
    self,
    url: str,
    json: Dict[str, Any] | None = None,
    headers: Dict[str, str] | None = None,
    timeout: float | None = None,
    stream: bool = False,
  ) -> FakeResponse:
    '''Return mocked endpoint response.

    Args:
      url: Request URL.
      json: Request payload.
      headers: Request headers.
      timeout: Request timeout.
      stream: Streaming flag.

    Returns:
      Fake response.
    '''
    return FakeResponse(
      status_code=200,
      json_data={
        'model': 'dummy-model',
        'choices': [
          {
            'message': {
              'role': 'assistant',
              'content': 'hello',
            },
            'finish_reason': 'stop',
          }
        ],
        'usage': {
          'prompt_tokens': 1,
          'completion_tokens': 1,
          'total_tokens': 2,
        },
      },
    )


def _provider_config() -> ProviderConfig:
  '''Build provider config for tests.

  Returns:
    Provider configuration.
  '''
  return ProviderConfig(
    enabled=True,
    base_url='http://dummy.invalid/v1',
    api_key_env='',
    rpm=100,
    timeout_seconds=1.0,
    models={
      'fast': ModelConfig(
        model='dummy-model',
        temperature=0.0,
        max_tokens=10,
      )
    },
  )


def _provider_request() -> ProviderRequest:
  '''Build provider request for tests.

  Returns:
    Provider request.
  '''
  return ProviderRequest(
    provider='dummy',
    model='dummy-model',
    messages=[ChatMessage(role='user', content='hi')],
    temperature=0.0,
    max_tokens=10,
    stream=False,
  )


def test_sync_provider_with_mocked_requests_endpoint() -> None:
  '''Synchronous provider parses mocked requests response.'''
  provider = OpenAICompatibleProvider(
    name='dummy',
    config=_provider_config(),
    requests_session=FakeSession(),
  )

  response = provider.chat(_provider_request())

  assert response.provider == 'dummy'
  assert response.content == 'hello'
  assert response.usage.input_tokens == 1
  assert response.usage.output_tokens == 1


def test_async_provider_with_mocked_httpx_endpoint() -> None:
  '''Asynchronous provider parses mocked httpx response.'''

  def handler(request: httpx.Request) -> httpx.Response:
    '''Mock httpx endpoint.

    Args:
      request: Incoming request.

    Returns:
      HTTP response.
    '''
    return httpx.Response(
      status_code=200,
      json={
        'model': 'dummy-model',
        'choices': [
          {
            'message': {
              'role': 'assistant',
              'content': 'hello',
            },
            'finish_reason': 'stop',
          }
        ],
        'usage': {
          'prompt_tokens': 1,
          'completion_tokens': 1,
          'total_tokens': 2,
        },
      },
    )

  async def run() -> str:
    '''Run async provider call.

    Returns:
      Response content.
    '''
    async with httpx.AsyncClient(
      transport=httpx.MockTransport(handler),
    ) as client:
      provider = OpenAICompatibleProvider(
        name='dummy',
        config=_provider_config(),
        httpx_client=client,
      )
      response = await provider.achat(_provider_request())
      return response.content

  content = asyncio.run(run())
  assert content == 'hello'
