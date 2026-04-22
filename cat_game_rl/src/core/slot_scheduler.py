from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from src.core.items import CraftingTree, ItemId, CRAFTABLE_ITEM_IDS, NUM_CRAFTABLE


@dataclass
class ManufacturingSlot:
    item_id: ItemId
    batch_size: int = 0
    remaining_ticks: int = 0

    @property
    def is_active(self) -> bool:
        return self.remaining_ticks > 0


class SlotScheduler:
    def __init__(self, crafting_tree: CraftingTree):
        self._tree = crafting_tree
        self._slots: Dict[ItemId, ManufacturingSlot] = {}
        for item_id in CRAFTABLE_ITEM_IDS:
            self._slots[ItemId(item_id)] = ManufacturingSlot(item_id=ItemId(item_id))

    def start(self, item_id: ItemId, batch_size: int) -> bool:
        slot = self._slots[item_id]
        if slot.is_active:
            return False
        if batch_size <= 0:
            return False
        slot.batch_size = batch_size
        slot.remaining_ticks = self._tree.craft_time_ticks(item_id)
        return True

    def tick(self) -> List[Tuple[ItemId, int]]:
        completed = []
        for slot in self._slots.values():
            if slot.is_active:
                slot.remaining_ticks -= 1
                if slot.remaining_ticks <= 0:
                    completed.append((slot.item_id, slot.batch_size))
                    slot.batch_size = 0
                    slot.remaining_ticks = 0
        return completed

    def is_busy(self, item_id: ItemId) -> bool:
        return self._slots[item_id].is_active

    def get_slot(self, item_id: ItemId) -> ManufacturingSlot:
        return self._slots[item_id]

    def get_slot_array(self) -> np.ndarray:
        arr = np.zeros((NUM_CRAFTABLE, 3), dtype=np.int32)
        for i, item_id_int in enumerate(CRAFTABLE_ITEM_IDS):
            slot = self._slots[ItemId(item_id_int)]
            arr[i, 0] = 1 if slot.is_active else 0
            arr[i, 1] = slot.remaining_ticks
            arr[i, 2] = slot.batch_size
        return arr

    def active_count(self) -> int:
        return sum(1 for s in self._slots.values() if s.is_active)

    def reset(self) -> None:
        for slot in self._slots.values():
            slot.batch_size = 0
            slot.remaining_ticks = 0
