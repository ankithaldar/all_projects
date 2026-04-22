from __future__ import annotations

import math


class CostCalculator:
    @staticmethod
    def unit_cost(init_cost: int, n: int) -> float:
        """Cost of the nth unit (1-indexed): init_cost * (1 + 0.5*(n-1))."""
        if n < 1:
            raise ValueError(f"Unit index must be >= 1, got {n}")
        return init_cost * (1 + 0.5 * (n - 1))

    @staticmethod
    def total_cost(init_cost: int, batch_size: int) -> float:
        """Total cost for a batch: sum of init_cost*(1 + 0.5*(n-1)) for n=1..batch_size.

        Closed form: init_cost * batch_size * (1 + 0.25*(batch_size - 1))
        = init_cost * (0.75 * batch_size + 0.25 * batch_size^2)
        """
        if batch_size <= 0:
            return 0.0
        return init_cost * batch_size * (1 + 0.25 * (batch_size - 1))

    @staticmethod
    def max_affordable_batch(init_cost: int, coins: int) -> int:
        """Largest batch_size B such that total_cost(init_cost, B) <= coins.

        Solve: init_cost * B * (1 + 0.25*(B-1)) <= coins
        => 0.25*B^2 + 0.75*B <= coins/init_cost
        => B^2 + 3*B - 4*coins/init_cost <= 0
        => B <= (-3 + sqrt(9 + 16*coins/init_cost)) / 2
        """
        if init_cost <= 0 or coins <= 0:
            return 0
        discriminant = 9 + 16 * coins / init_cost
        b_float = (-3 + math.sqrt(discriminant)) / 2
        b = max(0, int(b_float))
        if CostCalculator.total_cost(init_cost, b + 1) <= coins:
            b += 1
        return b
