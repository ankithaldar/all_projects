from __future__ import annotations

import pytest

from src.core.cost_calculator import CostCalculator


class TestUnitCost:
  def test_first_unit(self):
    assert CostCalculator.unit_cost(100, 1) == 100.0

  def test_second_unit(self):
    assert CostCalculator.unit_cost(100, 2) == 150.0

  def test_third_unit(self):
    assert CostCalculator.unit_cost(100, 3) == 200.0

  def test_fifth_unit(self):
    assert CostCalculator.unit_cost(100, 5) == 300.0

  def test_invalid_n(self):
    with pytest.raises(ValueError):
      CostCalculator.unit_cost(100, 0)

  def test_negative_n(self):
    with pytest.raises(ValueError):
      CostCalculator.unit_cost(100, -1)


class TestTotalCost:
  def test_batch_zero(self):
    assert CostCalculator.total_cost(100, 0) == 0.0

  def test_batch_one(self):
    assert CostCalculator.total_cost(100, 1) == 100.0

  def test_batch_two(self):
    assert CostCalculator.total_cost(100, 2) == 250.0

  def test_batch_five(self):
    expected = sum(CostCalculator.unit_cost(100, n) for n in range(1, 6))
    assert CostCalculator.total_cost(100, 5) == pytest.approx(expected)

  def test_batch_ten(self):
    expected = sum(CostCalculator.unit_cost(200, n) for n in range(1, 11))
    assert CostCalculator.total_cost(200, 10) == pytest.approx(expected)

  def test_closed_form_matches_summation(self):
    for init_cost in [50, 100, 300, 500, 1000]:
      for batch in range(1, 21):
        manual = sum(
          CostCalculator.unit_cost(init_cost, n)
          for n in range(1, batch + 1)
        )
        closed = CostCalculator.total_cost(init_cost, batch)
        assert closed == pytest.approx(manual, rel=1e-9), (
          f"Mismatch at init_cost={init_cost}, batch={batch}"
        )


class TestMaxAffordableBatch:
  def test_zero_coins(self):
    assert CostCalculator.max_affordable_batch(100, 0) == 0

  def test_zero_init_cost(self):
    assert CostCalculator.max_affordable_batch(0, 1000) == 0

  def test_exact_one(self):
    assert CostCalculator.max_affordable_batch(100, 100) == 1

  def test_exact_two(self):
    assert CostCalculator.max_affordable_batch(100, 250) == 2

  def test_between(self):
    assert CostCalculator.max_affordable_batch(100, 200) == 1

  def test_large_budget(self):
    batch = CostCalculator.max_affordable_batch(50, 100000)
    assert CostCalculator.total_cost(50, batch) <= 100000
    assert CostCalculator.total_cost(50, batch + 1) > 100000

  def test_inverse_of_total_cost(self):
    for init_cost in [50, 100, 500]:
      for b in range(1, 15):
        cost = CostCalculator.total_cost(init_cost, b)
        max_b = CostCalculator.max_affordable_batch(init_cost, int(cost))
        assert max_b >= b, (
          f"init_cost={init_cost}, b={b}, cost={cost}, max_b={max_b}"
        )

  def test_large_batch_no_overflow(self):
    cost = CostCalculator.total_cost(10000, 100)
    assert cost > 0
    assert isinstance(cost, float)
    batch = CostCalculator.max_affordable_batch(10000, int(cost))
    assert batch >= 100

  def test_negative_coins(self):
    assert CostCalculator.max_affordable_batch(100, -50) == 0

  def test_ceil_boundary(self):
    import math
    result = CostCalculator.max_affordable_batch(100, 249)
    assert result == 1
    assert math.ceil(CostCalculator.total_cost(100, 1)) <= 249
    assert math.ceil(CostCalculator.total_cost(100, 2)) > 249

  def test_returned_batch_always_affordable(self):
    import math
    for init_cost in [50, 100, 300, 1000]:
      for coins in range(1, 5000, 47):
        b = CostCalculator.max_affordable_batch(init_cost, coins)
        if b > 0:
          actual = math.ceil(CostCalculator.total_cost(init_cost, b))
          assert actual <= coins, (
            f"init={init_cost}, coins={coins}, b={b}, cost={actual}"
          )
