#!/usr/bin/env python
# -- coding: utf-8 --

'''Gateway routing, fallback, cache, and streaming tests.'''


from __future__ import annotations

from typing import Callable, Dict

from llm_gateway.app import LLMGateway
from llm_gateway.schemas import GatewayRequest
from tests.conftest import DummyProvider


def test_rate_limit_routes_to_second_provider(
  gateway_factory: Callable[[Dict[str, DummyProvider], Dict[str, int], bool], LLMGateway],
) -> None:
  '''When first provider hits local RPM, second provider is used.'''
  p1 = DummyProvider('p1', text='p1')
  p2 = DummyProvider('p2', text='p2')

  gateway = gateway_factory(
    {'p1': p1, 'p2': p2},
    {'p1': 1, 'p2': 100},
    cache_enabled=False,
  )

  first = gateway.complete(GatewayRequest(prompt='one'))
  second = gateway.complete(GatewayRequest(prompt='two'))

  assert first.provider == 'p1'
  assert second.provider == 'p2'
  assert p1.calls == 1
  assert p2.calls == 1

  gateway.close()


def test_fallback_after_transient_failure(
  gateway_factory: Callable[[Dict[str, DummyProvider], Dict[str, int], bool], LLMGateway],
) -> None:
  '''Fallback provider is used when primary fails transiently.'''
  p1 = DummyProvider('p1', fail_transient=True)
  local = DummyProvider('fallback_local', text='local')

  gateway = gateway_factory(
    {'p1': p1, 'fallback_local': local},
    {'p1': 100, 'fallback_local': 100},
    cache_enabled=False,
  )

  response = gateway.complete(GatewayRequest(prompt='hello'))

  assert response.provider == 'fallback_local'
  assert response.content == 'local'
  assert p1.calls == 1
  assert local.calls == 1

  gateway.close()


def test_cache_returns_cached_response(
  gateway_factory: Callable[[Dict[str, DummyProvider], Dict[str, int], bool], LLMGateway],
) -> None:
  '''Identical prompt returns cached response without extra API call.'''
  p1 = DummyProvider('p1', text='cached-response')

  gateway = gateway_factory(
    {'p1': p1},
    {'p1': 100},
    cache_enabled=True,
  )

  request = GatewayRequest(prompt='same prompt')

  first = gateway.complete(request)
  second = gateway.complete(request)

  assert first.cached is False
  assert second.cached is True
  assert second.content == 'cached-response'
  assert p1.calls == 1

  gateway.close()


def test_sync_streaming(
  gateway_factory: Callable[[Dict[str, DummyProvider], Dict[str, int], bool], LLMGateway],
) -> None:
  '''Synchronous streaming yields provider chunks.'''
  p1 = DummyProvider('p1', text='chunk')

  gateway = gateway_factory(
    {'p1': p1},
    {'p1': 100},
    cache_enabled=False,
  )

  chunks = list(gateway.stream(GatewayRequest(prompt='stream test')))

  assert len(chunks) == 1
  assert chunks[0].delta_content == 'chunk'
  assert p1.calls == 1

  gateway.close()
