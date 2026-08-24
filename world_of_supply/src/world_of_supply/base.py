#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Core abstractions shared by every acting entity in the simulation.'''

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
  from world_of_supply.economy import BalanceSheet
  from world_of_supply.facility import FacilityControl


class Agent(Protocol):
  '''Protocol for anything that can be stepped once per simulation tick.

  Implementations include transport vehicles and all facility units
  (storage, manufacturing, distribution, consumer, seller).
  '''

  def act(self, control: 'FacilityControl | None') -> 'BalanceSheet':
    '''Advance one time step under the given control.

    Args:
      control: Facility-level control carrying this unit's parameters, or
        ``None`` to run with defaults / perform no action.

    Returns:
      BalanceSheet: The monetary effect of this step.
    '''
    ...
