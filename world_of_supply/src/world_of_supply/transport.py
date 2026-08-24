#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Transport vehicles moving products along railroad paths.

A :class:`Transport` cycles through load → move → unload → return. Its state
machine is driven by two integers: ``step`` (+1 outbound, -1 returning,
0 idle) and ``location_pointer`` (index into the precomputed path).
'''

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from world_of_supply.base import Agent
from world_of_supply.economy import BalanceSheet

if TYPE_CHECKING:
  from world_of_supply.facility import FacilityCell
  from world_of_supply.world import World


class TransportState(Enum):
  '''High-level state of a vehicle, derived from the low-level pointers.

  Attributes:
    IDLE: Parked at home with no assignment.
    LOADING: At origin waiting for requested payload.
    EN_ROUTE: Moving toward the destination.
    UNLOADING: At destination waiting to unload.
    RETURNING: Heading back home.
  '''

  IDLE = auto()
  LOADING = auto()
  EN_ROUTE = auto()
  UNLOADING = auto()
  RETURNING = auto()


@dataclass
class TransportEconomy:
  '''Cost parameters of a vehicle.

  Attributes:
    unit_transport_cost: Cost of moving one payload unit one cell.
  '''

  unit_transport_cost: int = 1

  def step_balance_sheet(self, payload: int, step: int) -> BalanceSheet:
    '''Compute the freight cost for one time step.

    Args:
      payload: Units currently carried.
      step: Movement direction magnitude (+1/-1 moving, 0 stationary).

    Returns:
      BalanceSheet: Loss proportional to payload and distance.
    '''
    return BalanceSheet(0, -payload * self.unit_transport_cost * abs(step))


class Transport(Agent):
  '''One truck operating out of a home facility.

  Attributes:
    source: Home facility the truck operates from.
    economy: Freight-cost parameters.
    destination: Delivery target facility, or ``None`` when idle.
    product_id: Product carried on the current assignment.
    requested_quantity: Quantity requested when scheduled.
    path: Precomputed coordinate sequence from source to destination.
    location_pointer: Current index into ``path``.
    step: Direction flag (+1 outbound, -1 returning, 0 idle).
    payload: Units currently loaded.
  '''

  def __init__(self, source: 'FacilityCell', economy: TransportEconomy) -> None:
    '''Create an idle truck at its home facility.

    Args:
      source: Home facility.
      economy: Freight-cost parameters.
    '''
    self.source = source
    self.economy = economy
    self.destination: 'FacilityCell | None' = None
    self.product_id: str = ''
    self.requested_quantity: int = 0
    self.path: list[tuple[int, int]] | None = None
    self.location_pointer: int = 0
    self.step: int = 0
    self.payload: int = 0

  @property
  def state(self) -> TransportState:
    '''Derive the high-level state from movement pointers.'''
    if self.destination is None:
      return TransportState.IDLE
    at_origin = self.location_pointer == 0
    at_goal = self.path is not None and self.location_pointer == len(self.path) - 1
    if self.step > 0 and at_origin and self.payload == 0:
      return TransportState.LOADING
    if self.step > 0:
      return TransportState.EN_ROUTE
    if self.step < 0 and at_goal and self.payload > 0:
      return TransportState.UNLOADING
    return TransportState.RETURNING

  def schedule(
      self,
      world: 'World',
      destination: 'FacilityCell',
      product_id: str,
      quantity: int,
  ) -> None:
    '''Assign a delivery job and precompute the route.

    Args:
      world: World used for path finding.
      destination: Facility that should receive the goods.
      product_id: Product to deliver.
      quantity: Requested delivery size.

    Raises:
      ValueError: If no traversable route exists to the destination.
    '''
    path = world.find_path((self.source.x, self.source.y), (destination.x, destination.y))
    if path is None:
      raise ValueError(f'Destination {destination} is unreachable')
    self.destination = destination
    self.product_id = product_id
    self.requested_quantity = quantity
    self.path = path
    self.step = 1
    self.location_pointer = 0
    self.payload = 0

  def path_len(self) -> int:
    '''Return the length of the current route.

    Returns:
      int: Zero when no route has been planned yet.
    '''
    return len(self.path) if self.path is not None else 0

  def is_enroute(self) -> bool:
    '''Return True when the truck carries an active assignment.'''
    return self.destination is not None

  def current_location(self) -> tuple[int, int]:
    '''Return the truck grid position.

    Returns:
      tuple[int, int]: Coordinate on the path, or home position when idle.
    '''
    if self.path is None:
      return (self.source.x, self.source.y)
    return self.path[self.location_pointer]

  def try_loading(self, quantity: int) -> bool:
    '''Attempt to take the requested payload from the home facility.

    Args:
      quantity: Number of units to load.

    Returns:
      bool: True when the full quantity was available and loaded.
    '''
    loaded = self.source.storage.try_take_units({self.product_id: quantity})
    if loaded:
      self.payload = quantity
    return loaded

  def try_unloading(self) -> int:
    '''Deposit the payload into the destination storage.

    Units that do not fit are lost; a successful deposit notifies the
    destination consumer so its open-order book is updated.

    Returns:
      int: Number of units actually delivered.
    '''
    assert self.destination is not None
    deposited = self.destination.storage.try_add_units({self.product_id: self.payload}, all_or_nothing=False)
    delivered = deposited.get(self.product_id, 0)
    if delivered > 0:
      self.destination.consumer.on_order_reception(self.source.id, self.product_id, delivered)
      self.payload -= delivered
    return delivered

  def act(self, control=None) -> BalanceSheet:
    '''Advance the load/move/unload/return cycle by one tick.

    Args:
      control: Unused; accepted to satisfy the :class:`Agent` protocol.

    Returns:
      BalanceSheet: Freight cost incurred during this tick.
    '''
    if self.path is None:
      return BalanceSheet()
    last_index = len(self.path) - 1
    if self.step > 0:
      if self.location_pointer == 0 and self.payload == 0:
        self.try_loading(self.requested_quantity)
      if self.payload > 0:
        if self.location_pointer < last_index:
          self.location_pointer += self.step
        else:
          self.step = -1
    elif self.step < 0:
      if self.location_pointer == last_index and self.payload > 0:
        self.try_unloading()
      if self.payload == 0:
        if self.location_pointer > 0:
          self.location_pointer += self.step
        else:
          self.step = 0
          self.destination = None
    return self.economy.step_balance_sheet(self.payload, abs(self.step))
