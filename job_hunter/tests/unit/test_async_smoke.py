#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Native async smoke tests using MockTransport (no network).'''


from __future__ import annotations

import httpx
from job_hunter.adapters.http_client import RobotsCache


async def test_robots_cache_allows_and_disallows() -> None:
  '''Prove RobotsCache honours Allow and Disallow robots rules.'''
  def _handler(request: httpx.Request) -> httpx.Response:
    '''Return robots.txt allowing root but disallowing /private.'''
    _ = request
    return httpx.Response(
      200, text='User-agent: *\nAllow: /\nDisallow: /private\n',
    )

  cache = RobotsCache()
  transport = httpx.MockTransport(_handler)
  async with httpx.AsyncClient(transport=transport) as client:
    assert await cache.allowed(client, 'https://example.com/page')
    assert not await cache.allowed(client, 'https://example.com/private')


async def test_robots_cache_missing_robots_allows() -> None:
  '''Prove RobotsCache allows when robots.txt is unavailable (404).'''
  def _handler(request: httpx.Request) -> httpx.Response:
    '''Return 404 for every request.'''
    _ = request
    return httpx.Response(404)

  cache = RobotsCache()
  transport = httpx.MockTransport(_handler)
  async with httpx.AsyncClient(transport=transport) as client:
    assert await cache.allowed(client, 'https://example.com/any')
