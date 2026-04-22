from __future__ import annotations

import numpy as np
import pytest

from src.core.items import NUM_ITEMS, NUM_CRAFTABLE
from src.env.reward_shaper import (
    SlotUtilizationReward,
    WasteMinimizationReward,
    TargetCompletionReward,
    ExcessInventoryPenalty,
    RewardShaper,
)


def _make_state(
    active_slots=0,
    fraction_complete=0.0,
    targets_complete=False,
    stash=None,
    targets_remaining=None,
    targets_total=None,
):
    return {
        "active_slot_count": active_slots,
        "fraction_complete": fraction_complete,
        "targets_complete": targets_complete,
        "stash": stash if stash is not None else np.zeros(NUM_ITEMS),
        "targets_remaining": targets_remaining if targets_remaining is not None else np.zeros(NUM_ITEMS),
        "targets_total": targets_total if targets_total is not None else np.ones(NUM_ITEMS),
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
            "total",
        }
        assert expected_keys.issubset(set(breakdown.keys()))
