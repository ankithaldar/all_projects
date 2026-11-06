#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cost calculations for crafting batches.'''

from __future__ import annotations

import math


class CostCalculator:
  '''Computes escalating crafting costs.'''

  @staticmethod
  def unit_cost(init_cost: int, n: int) -> float:
    '''Computes the cost of the nth unit.

    Args:
      init_cost: Initial unit cost.
      n: One-indexed unit number.

    Returns:
      Cost of the nth unit.

    Raises:
      ValueError: If n is less than 1.
    '''
    if n < 1:
      raise ValueError(f'Unit index must be >= 1, got {n}')
    return init_cost * (1 + 0.5 * (n - 1))

  @staticmethod
  def total_cost(init_cost: int, batch_size: int) -> float:
    '''Computes total cost for a batch.

    Args:
      init_cost: Initial unit cost.
      batch_size: Number of units in the batch.

    Returns:
      Total cost for the batch.
    '''
    if batch_size <= 0:
      return 0.0
    return init_cost * batch_size * (1 + 0.25 * (batch_size - 1))

  @staticmethod
  def max_affordable_batch(init_cost: int, coins: int) -> int:
    '''Computes the largest affordable batch size.

    Args:
      init_cost: Initial unit cost.
      coins: Available coins.

    Returns:
      Largest batch size whose total cost fits the coins.
    '''
    if init_cost <= 0 or coins <= 0:
      return 0
    discriminant = 9 + 16 * coins / init_cost
    b_float = (-3 + math.sqrt(discriminant)) / 2
    b = max(0, int(b_float))
    while b > 0 and math.ceil(
        CostCalculator.total_cost(init_cost, b)) > coins:
      b -= 1
    if math.ceil(CostCalculator.total_cost(init_cost, b + 1)) <= coins:
      b += 1
    return b
