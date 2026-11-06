#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Scheduling of manufacturing slots.'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.core.items import CraftingTree, ItemId, CRAFTABLE_ITEM_IDS
from src.core.items import NUM_CRAFTABLE


@dataclass
class ManufacturingSlot:
  '''State of a single manufacturing slot.

  Attributes:
    item_id: Item being crafted in the slot.
    batch_size: Size of the current batch.
    remaining_ticks: Ticks left for the current batch.
  '''

  item_id: ItemId
  batch_size: int = 0
  remaining_ticks: int = 0

  @property
  def is_active(self) -> bool:
    '''Returns whether the slot is actively crafting.'''
    return self.remaining_ticks > 0


class SlotScheduler:
  '''Schedules crafting across manufacturing slots.'''

  def __init__(self, crafting_tree: CraftingTree):
    '''Initializes slots for every craftable item.

    Args:
      crafting_tree: Tree providing crafting times.
    '''
    self._tree = crafting_tree
    self._slots: Dict[ItemId, ManufacturingSlot] = {}
    for item_id in CRAFTABLE_ITEM_IDS:
      self._slots[ItemId(item_id)] = ManufacturingSlot(
          item_id=ItemId(item_id))

  def start(self, item_id: ItemId, batch_size: int) -> bool:
    '''Starts crafting in a slot.

    Args:
      item_id: Item to craft.
      batch_size: Size of the batch to craft.

    Returns:
      True if crafting started, else False.
    '''
    slot = self._slots[item_id]
    if slot.is_active:
      return False
    if batch_size <= 0:
      return False
    slot.batch_size = batch_size
    slot.remaining_ticks = self._tree.craft_time_ticks(item_id)
    return True

  def tick(self) -> List[Tuple[ItemId, int]]:
    '''Advances every active slot by one tick.

    Returns:
      Completed batches as item and batch size pairs.
    '''
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
    '''Checks whether a slot is busy.

    Args:
      item_id: Item whose slot to check.

    Returns:
      True if the slot is busy, else False.
    '''
    return self._slots[item_id].is_active

  def get_slot(self, item_id: ItemId) -> ManufacturingSlot:
    '''Returns the slot for an item.

    Args:
      item_id: Item whose slot to return.

    Returns:
      The manufacturing slot for the item.
    '''
    return self._slots[item_id]

  def get_slot_array(self) -> np.ndarray:
    '''Returns slot states as an array.

    Returns:
      Array with active flag, remaining ticks and batch size.
    '''
    arr = np.zeros((NUM_CRAFTABLE, 3), dtype=np.int32)
    for i, item_id_int in enumerate(CRAFTABLE_ITEM_IDS):
      slot = self._slots[ItemId(item_id_int)]
      arr[i, 0] = 1 if slot.is_active else 0
      arr[i, 1] = slot.remaining_ticks
      arr[i, 2] = slot.batch_size
    return arr

  def active_count(self) -> int:
    '''Returns the number of active slots.

    Returns:
      Count of active slots.
    '''
    return sum(1 for s in self._slots.values() if s.is_active)

  def reset(self) -> None:
    '''Clears every slot.'''
    for slot in self._slots.values():
      slot.batch_size = 0
      slot.remaining_ticks = 0
