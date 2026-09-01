#!/usr/bin/env python
# -- coding: utf-8 --

'''API key rotation utility.'''


from __future__ import annotations

import threading
from typing import List


class APIKeyRotator:
  '''Rotates API keys for a provider.'''

  def __init__(self, keys: List[str]) -> None:
    '''Initialize the rotator.

    Args:
      keys: List of API keys. Empty strings are removed.
    '''
    self._keys = [key.strip() for key in keys if key.strip()]
    self._index = 0
    self._lock = threading.Lock()

  def get_and_rotate(self) -> str:
    '''Return the current key and advance the rotation pointer.

    Returns:
      The current API key, or an empty string when no keys are configured.
    '''
    with self._lock:
      if not self._keys:
        return ''

      key = self._keys[self._index]
      self._index = (self._index + 1) % len(self._keys)
      return key
