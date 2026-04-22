from __future__ import annotations

import numpy as np
import pytest

from src.core.items import ItemId, CraftingTree, NUM_CRAFTABLE
from src.core.slot_scheduler import SlotScheduler, ManufacturingSlot


class TestManufacturingSlot:
    def test_default_inactive(self):
        slot = ManufacturingSlot(item_id=ItemId.STRING)
        assert not slot.is_active
        assert slot.batch_size == 0
        assert slot.remaining_ticks == 0

    def test_active_when_remaining(self):
        slot = ManufacturingSlot(
            item_id=ItemId.STRING, batch_size=5, remaining_ticks=3
        )
        assert slot.is_active


class TestSlotScheduler:
    def test_init_all_idle(self, slot_scheduler: SlotScheduler):
        assert slot_scheduler.active_count() == 0

    def test_start_success(
        self, slot_scheduler: SlotScheduler, crafting_tree: CraftingTree
    ):
        assert slot_scheduler.start(ItemId.STRING, 5) is True
        assert slot_scheduler.is_busy(ItemId.STRING) is True
        assert slot_scheduler.active_count() == 1

    def test_start_busy_rejected(self, slot_scheduler: SlotScheduler):
        slot_scheduler.start(ItemId.STRING, 5)
        assert slot_scheduler.start(ItemId.STRING, 3) is False

    def test_start_zero_batch_rejected(self, slot_scheduler: SlotScheduler):
        assert slot_scheduler.start(ItemId.STRING, 0) is False

    def test_tick_decrements(self, slot_scheduler: SlotScheduler):
        slot_scheduler.start(ItemId.STRING, 5)
        slot = slot_scheduler.get_slot(ItemId.STRING)
        initial_ticks = slot.remaining_ticks
        slot_scheduler.tick()
        assert slot.remaining_ticks == initial_ticks - 1

    def test_tick_completes_item(self, slot_scheduler: SlotScheduler):
        slot_scheduler.start(ItemId.STRING, 3)
        completed = slot_scheduler.tick()
        assert len(completed) == 1
        assert completed[0] == (ItemId.STRING, 3)
        assert not slot_scheduler.is_busy(ItemId.STRING)

    def test_tick_no_completion(self, slot_scheduler: SlotScheduler):
        slot_scheduler.start(ItemId.METAL, 2)
        completed = slot_scheduler.tick()
        assert len(completed) == 0
        assert slot_scheduler.is_busy(ItemId.METAL)

    def test_multiple_slots(self, slot_scheduler: SlotScheduler):
        slot_scheduler.start(ItemId.STRING, 5)
        slot_scheduler.start(ItemId.WOOD, 3)
        slot_scheduler.start(ItemId.METAL, 2)
        assert slot_scheduler.active_count() == 3

    def test_get_slot_array_shape(self, slot_scheduler: SlotScheduler):
        arr = slot_scheduler.get_slot_array()
        assert arr.shape == (NUM_CRAFTABLE, 3)
        assert arr.dtype == np.int32

    def test_get_slot_array_active(self, slot_scheduler: SlotScheduler):
        slot_scheduler.start(ItemId.STRING, 5)
        arr = slot_scheduler.get_slot_array()
        assert arr[0, 0] == 1  # active
        assert arr[0, 2] == 5  # batch_size

    def test_reset(self, slot_scheduler: SlotScheduler):
        slot_scheduler.start(ItemId.STRING, 5)
        slot_scheduler.start(ItemId.METAL, 2)
        slot_scheduler.reset()
        assert slot_scheduler.active_count() == 0

    def test_metal_takes_three_ticks(self, slot_scheduler: SlotScheduler):
        slot_scheduler.start(ItemId.METAL, 1)
        assert slot_scheduler.tick() == []
        assert slot_scheduler.tick() == []
        completed = slot_scheduler.tick()
        assert completed == [(ItemId.METAL, 1)]
