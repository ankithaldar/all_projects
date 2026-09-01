#!/usr/bin/env python
# -- coding: utf-8 --

'''Tests for RPM rate limiting.'''


from __future__ import annotations

from llm_gateway.rate_limiter import RPMRateLimiter


def test_rate_limiter_allows_up_to_limit() -> None:
  '''Limiter allows exactly RPM calls in a window.'''
  limiter = RPMRateLimiter(2)

  assert limiter.allow() is True
  assert limiter.allow() is True
  assert limiter.allow() is False


def test_rate_limiter_unlimited_when_zero() -> None:
  '''Zero RPM means unlimited.'''
  limiter = RPMRateLimiter(0)

  for _ in range(10):
    assert limiter.allow() is True
