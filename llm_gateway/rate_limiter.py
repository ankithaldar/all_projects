#!/usr/bin/env python
# -- coding: utf-8 --

'''Sliding-window RPM rate limiter.'''


from __future__ import annotations

from collections import deque
import threading
import time


class RPMRateLimiter:
  '''Thread-safe sliding-window rate limiter for requests per minute.'''

  def __init__(self, rpm: int) -> None:
    '''Initialize the limiter.

    Args:
      rpm: Maximum requests per minute. Values <= 0 mean unlimited.
    '''
    self._rpm = rpm
    self._calls: deque[float] = deque()
    self._lock = threading.Lock()

  def allow(self) -> bool:
    '''Check whether a request is allowed and record it if allowed.

    Returns:
      True if the request is allowed, otherwise False.
    '''
    if self._rpm <= 0:
      return True

    now = time.time()
    window_start = now - 60.0

    with self._lock:
      while self._calls and self._calls[0] <= window_start:
        self._calls.popleft()

      if len(self._calls) < self._rpm:
        self._calls.append(now)
        return True

      return False
