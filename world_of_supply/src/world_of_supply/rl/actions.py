#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Action decoding: flat agent actions to facility controls.'''

from __future__ import annotations

from collections import defaultdict

import numpy as np

from world_of_supply.facility import FacilityControl
from world_of_supply.rl.agents import facility_id_of, is_producer
from world_of_supply.world import Control, World

PRICE_LEVELS: tuple[int, ...] = (400, 600, 1000, 1200, 1400, 1600, 1800, 2000)
RATE_LEVELS: tuple[int, ...] = (0, 2, 4, 6, 8, 10)
QUANTITY_LEVELS: tuple[int, ...] = (0, 2, 4, 6, 8, 10)


class ActionDecoder:
  '''Translates per-agent discrete actions into :class:`Control` objects.

  Producer actions are ``(price_index, rate_index)`` pairs; consumer actions
  are ``(product_index, source_index, quantity_index)`` triples.
  '''

  def __init__(self, product_ids: list[str]) -> None:
    '''Capture the global product ordering used by consumer actions.

    Args:
      product_ids: Sorted product ids (action column order).
    '''
    self.product_ids = list(product_ids)

  def decode(self, action_dict: dict[str, object], world: World) -> Control:
    '''Convert a multi-agent action dictionary into facility controls.

    Args:
      action_dict: Mapping of agent id to action array/list.
      world: Current world providing facility lookup.

    Returns:
      Control: Controls keyed by facility id.
    '''
    actions_by_facility: dict[str, list[tuple[str, object]]] = defaultdict(list)
    for agent_id, action in action_dict.items():
      actions_by_facility[facility_id_of(agent_id)].append((agent_id, action))

    controls: dict[str, FacilityControl] = {}
    for facility_id, actions in actions_by_facility.items():
      controls[facility_id] = self._decode_facility(
          world.facilities[facility_id], actions
      )
    return Control(controls)

  def _decode_facility(self, facility, actions: list[tuple[str, object]]) -> FacilityControl:
    '''Merge all role actions of one facility into a single control.

    Args:
      facility: Target facility.
      actions: ``(agent_id, action)`` pairs belonging to this facility.

    Returns:
      FacilityControl: Combined control.
    '''
    control = FacilityControl()
    n_sources = len(facility.consumer.sources) if facility.consumer is not None else 0

    def element(action, index: int) -> int:
      '''Safely read one component of an action vector.

      Args:
        action: Raw action array or scalar.
        index: Component index.

      Returns:
        int: Component value as int, zero when absent.
      '''
      flat = np.asarray(action).flatten()
      return int(flat[index]) if index < len(flat) else 0

    for agent_id, action in actions:
      if is_producer(agent_id):
        control.unit_price = PRICE_LEVELS[element(action, 0)]
        control.production_rate = RATE_LEVELS[element(action, 1)]
      else:
        source_index = min(element(action, 1), max(n_sources - 1, 0))
        control.consumer_product_id = self.product_ids[element(action, 0) % len(self.product_ids)]
        control.consumer_source_id = source_index
        control.consumer_quantity = (
            QUANTITY_LEVELS[element(action, 2)] if n_sources > 0 else 0
        )
    return control
