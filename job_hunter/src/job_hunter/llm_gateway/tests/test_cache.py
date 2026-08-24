#!/usr/bin/env python
# -- coding: utf-8 --

'''Tests for SQLite cache.'''


from __future__ import annotations

from pathlib import Path

from llm_gateway.cache import SQLiteCache
from llm_gateway.schemas import GatewayRequest


def test_cache_roundtrip(tmp_path: Path) -> None:
  '''Cache stores and retrieves payload.'''
  cache = SQLiteCache(tmp_path / 'cache.db', ttl_seconds=60)

  cache.set('k1', {'content': 'hello'})
  assert cache.get('k1') == {'content': 'hello'}


def test_cache_key_changes_with_prompt() -> None:
  '''Different prompts produce different cache keys.'''
  request_one = GatewayRequest(prompt='one')
  request_two = GatewayRequest(prompt='two')

  key_one = SQLiteCache.make_key(request_one, include_temperature=False)
  key_two = SQLiteCache.make_key(request_two, include_temperature=False)

  assert key_one != key_two
