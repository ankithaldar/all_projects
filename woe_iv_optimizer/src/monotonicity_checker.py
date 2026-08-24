#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Monotonicity checker class"""


# imports
import numpy as np
from typing import Union
#    script imports
# imports


# constants
# constants


# classes
class MonotonicityChecker:
  """Monotonicity checker class"""

  @staticmethod
  def is_monotonic(arr: Union[list, np.ndarray]) -> bool:
    arr = np.array(arr)
    diffs = np.diff(arr)
    return np.all(diffs >= 0) or np.all(diffs <= 0)
# classes
