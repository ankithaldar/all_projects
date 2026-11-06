#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Shared base-material replenishment for crafting environments.'''

from __future__ import annotations

from src.core.inventory import Stash
from src.core.items import BASE_ITEM_IDS, MAX_AFFORDABLE_BATCH


def replenish_base_materials(
  stash: Stash,
  cap: int = MAX_AFFORDABLE_BATCH,
) -> None:
  '''Refill every base material in a stash up to a cap.

  Args:
    stash: Stash whose base materials are replenished in place.
    cap: Per-item quantity to top each base material up to. Defaults
      to MAX_AFFORDABLE_BATCH.

  Returns:
    None. The stash is mutated in place; items already at or above
    the cap are left unchanged.
  '''
  for base_id in BASE_ITEM_IDS:
    deficit = max(0, cap - stash.get(base_id))
    if deficit > 0:
      stash.add(base_id, deficit)
