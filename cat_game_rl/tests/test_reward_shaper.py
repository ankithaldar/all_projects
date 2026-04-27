from __future__ import annotations

import numpy as np
import pytest

from src.core.items import NUM_ITEMS, NUM_CRAFTABLE
from src.cat_game_env.reward_shaper import (
  SlotUtilizationReward,
  WasteMinimizationReward,
  TargetCompletionReward,
  ExcessInventoryPenalty,
  CoinEfficiencyReward,
  TimeEfficiencyReward,
  BatchOptimizationReward,
  RewardShaper,
)


def _make_state(
  active_slots=0,
  fraction_complete=0.0,
  targets_complete=False,
  stash=None,
  targets_remaining=None,
  targets_total=None,
  tick=0,
):
  return {
    "active_slot_count": active_slots,
    "fraction_complete": fraction_complete,
    "targets_complete": targets_complete,
    "stash": stash if stash is not None else np.zeros(NUM_ITEMS),
    "targets_remaining": (
      targets_remaining if targets_remaining is not None
      else np.zeros(NUM_ITEMS)
    ),
    "targets_total": (
      targets_total if targets_total is not None
      else np.ones(NUM_ITEMS)
    ),
    "tick": tick,
  }


class TestSlotUtilization:
  def test_zero_slots(self):
    r = SlotUtilizationReward()
    state = _make_state(active_slots=0)
    assert r.compute({}, {}, state) == 0.0

  def test_full_slots(self):
    r = SlotUtilizationReward()
    state = _make_state(active_slots=NUM_CRAFTABLE)
    assert r.compute({}, {}, state) == pytest.approx(1.0)

  def test_half_slots(self):
    r = SlotUtilizationReward()
    state = _make_state(active_slots=10)
    assert r.compute({}, {}, state) == pytest.approx(10 / NUM_CRAFTABLE)

  def test_name(self):
    assert SlotUtilizationReward().name == "slot_utilization"


class TestWasteMinimization:
  def test_no_waste(self):
    r = WasteMinimizationReward()
    state = _make_state()
    assert r.compute({}, {}, state) == 0.0

  def test_with_excess(self):
    r = WasteMinimizationReward()
    remaining = np.zeros(NUM_ITEMS)
    remaining[4] = -10  # STRING over-produced
    total = np.ones(NUM_ITEMS)
    total[4] = 5
    state = _make_state(targets_remaining=remaining, targets_total=total)
    value = r.compute({}, {}, state)
    assert value < 0

  def test_name(self):
    assert WasteMinimizationReward().name == "waste_minimization"


class TestTargetCompletion:
  def test_no_progress(self):
    r = TargetCompletionReward()
    before = _make_state(fraction_complete=0.0)
    after = _make_state(fraction_complete=0.0)
    value = r.compute(before, {}, after)
    assert value == pytest.approx(-0.001)

  def test_progress(self):
    r = TargetCompletionReward()
    before = _make_state(fraction_complete=0.0)
    after = _make_state(fraction_complete=0.1)
    value = r.compute(before, {}, after)
    assert value > 0

  def test_completion_bonus(self):
    r = TargetCompletionReward()
    before = _make_state(fraction_complete=0.9)
    after = _make_state(fraction_complete=1.0, targets_complete=True)
    value = r.compute(before, {}, after)
    assert value > 100.0

  def test_name(self):
    assert TargetCompletionReward().name == "target_completion"


class TestExcessInventory:
  def test_empty_stash(self):
    r = ExcessInventoryPenalty()
    state = _make_state(stash=np.zeros(NUM_ITEMS))
    assert r.compute({}, {}, state) == 0.0

  def test_large_stash(self):
    r = ExcessInventoryPenalty()
    stash = np.full(NUM_ITEMS, 100)
    state = _make_state(stash=stash)
    value = r.compute({}, {}, state)
    assert value < 0

  def test_name(self):
    assert ExcessInventoryPenalty().name == "excess_inventory"


class TestCoinEfficiency:
  def test_no_spending(self):
    r = CoinEfficiencyReward()
    value = r.compute({}, {"total_coins_spent": 0, "applied": {}}, {})
    assert value == 0.0

  def test_spending(self):
    from src.core.items import ItemId
    r = CoinEfficiencyReward()
    action_info = {
      "total_coins_spent": 500,
      "applied": {ItemId.STRING: 5},
    }
    value = r.compute({}, action_info, {})
    assert value < 0

  def test_name(self):
    assert CoinEfficiencyReward().name == "coin_efficiency"


class TestTimeEfficiency:
  def test_not_complete(self):
    r = TimeEfficiencyReward()
    state = _make_state(targets_complete=False, tick=100)
    assert r.compute({}, {}, state) == 0.0

  def test_early_completion(self):
    r = TimeEfficiencyReward()
    state = _make_state(targets_complete=True, tick=500)
    value = r.compute({}, {}, state)
    assert value > 0

  def test_late_completion_lower_reward(self):
    r = TimeEfficiencyReward()
    early = _make_state(targets_complete=True, tick=500)
    late = _make_state(targets_complete=True, tick=1500)
    assert r.compute({}, {}, early) > r.compute({}, {}, late)

  def test_name(self):
    assert TimeEfficiencyReward().name == "time_efficiency"


class TestBatchOptimization:
  def test_no_actions(self):
    r = BatchOptimizationReward()
    assert r.compute({}, {"applied": {}}, {}) == 0.0

  def test_large_batches(self):
    from src.core.items import ItemId
    r = BatchOptimizationReward()
    action_info = {"applied": {ItemId.STRING: 20}}
    value = r.compute({}, action_info, {})
    assert value == pytest.approx(1.0)

  def test_small_batches(self):
    from src.core.items import ItemId
    r = BatchOptimizationReward()
    action_info = {"applied": {ItemId.STRING: 5}}
    value = r.compute({}, action_info, {})
    assert value == pytest.approx(0.25)

  def test_name(self):
    assert BatchOptimizationReward().name == "batch_optimization"


class TestRewardShaper:
  def test_default_weights(self):
    shaper = RewardShaper()
    before = _make_state(fraction_complete=0.0)
    after = _make_state(active_slots=5, fraction_complete=0.1)
    reward, breakdown = shaper.compute(before, {}, after)
    assert isinstance(reward, float)
    assert "total" in breakdown
    assert "slot_utilization" in breakdown

  def test_custom_weights(self):
    shaper = RewardShaper(weights={
      "slot_utilization": 0.0,
      "waste_minimization": 0.0,
      "target_completion": 1.0,
      "excess_inventory": 0.0,
    })
    before = _make_state(fraction_complete=0.0)
    after = _make_state(active_slots=19, fraction_complete=0.0)
    reward, breakdown = shaper.compute(before, {}, after)
    assert breakdown["slot_utilization_weighted"] == 0.0

  def test_all_components_present(self):
    shaper = RewardShaper()
    before = _make_state()
    after = _make_state()
    _, breakdown = shaper.compute(before, {}, after)
    expected_keys = {
      "slot_utilization", "slot_utilization_weighted",
      "waste_minimization", "waste_minimization_weighted",
      "target_completion", "target_completion_weighted",
      "excess_inventory", "excess_inventory_weighted",
      "coin_efficiency", "coin_efficiency_weighted",
      "time_efficiency", "time_efficiency_weighted",
      "batch_optimization", "batch_optimization_weighted",
      "total",
    }
    assert expected_keys.issubset(set(breakdown.keys()))
