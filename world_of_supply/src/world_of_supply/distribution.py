#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Distribution units: order queues plus truck fleets.'''

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from world_of_supply.base import Agent
from world_of_supply.economy import BalanceSheet
from world_of_supply.transport import Transport, TransportEconomy

if TYPE_CHECKING:
  from world_of_supply.facility import FacilityCell, FacilityControl


@dataclass
class Order:
  '''A purchase order placed by some consumer facility.

  Attributes:
    destination: Facility that placed the order.
    product_id: Requested product.
    quantity: Requested number of units.
    unit_price: Price locked in at placement time.
  '''

  destination: 'FacilityCell'
  product_id: str
  quantity: int
  unit_price: int = 0


@dataclass
class DistributionEconomy:
  '''Commercial parameters and statistics of distribution.

  Attributes:
    unit_price: Price charged per shipped unit.
    wrong_order_penalty: Penalty per unit for foreign products.
    pending_order_penalty: Fee per queued order per tick.
    order_checkin: Revenue accumulator flushed once per act.
    total_wrong_order_penalties: Lifetime wrong-order total.
    total_pending_order_penalties: Lifetime pending-order total.
  '''

  unit_price: int = 0
  wrong_order_penalty: int = 500
  pending_order_penalty: int = 4
  order_checkin: int = field(default=0)
  total_wrong_order_penalties: int = 0
  total_pending_order_penalties: int = 0

  def profit(self, units_sold: int) -> int:
    '''Compute revenue for a given shipped volume.

    Args:
      units_sold: Number of units to be shipped.

    Returns:
      int: ``unit_price * units_sold``.
    '''
    return self.unit_price * units_sold


class DistributionUnit(Agent):
  '''Runs the outbound logistics of a facility.

  Orders arrive via :meth:`place_order`, sit in a FIFO queue, and
  are picked up by idle trucks. Every queued order costs a fee;
  foreign-product orders are penalized.

  Attributes:
    facility: Owning facility.
    fleet: Trucks operating from this facility.
    order_queue: FIFO queue of unassigned orders.
    economy: Commercial parameters.
  '''

  def __init__(
      self,
      facility: 'FacilityCell',
      fleet_size: int,
      economy: DistributionEconomy,
      transport_economy: TransportEconomy,
  ) -> None:
    '''Create the unit with its truck fleet.

    Args:
      facility: Owning facility.
      fleet_size: Number of trucks.
      economy: Commercial parameters.
      transport_economy: Per-truck freight-cost parameters.

    Returns:
      None: The unit is mutated in place.
    '''
    self.facility = facility
    self.fleet = [
        Transport(facility, transport_economy)
        for _ in range(fleet_size)
    ]
    self.order_queue: deque[Order] = deque()
    self.economy = economy

  def place_order(self, order: Order) -> int:
    '''Register an incoming order and return the buyer payment.

    Valid orders join the queue and generate immediate revenue.
    Foreign-product orders incur a tracked penalty.

    Args:
      order: The incoming order.

    Returns:
      int: Positive amount the buyer has to pay.
    '''
    if order.quantity <= 0:
      return 0
    if order.product_id != self.facility.bom.output_product_id:
      penalty = self.economy.wrong_order_penalty * order.quantity
      self.economy.total_wrong_order_penalties -= penalty
      return penalty
    order.unit_price = self.economy.unit_price
    self.order_queue.append(order)
    payment = self.economy.profit(order.quantity)
    self.economy.order_checkin += payment
    return payment

  def act(
      self,
      control: 'FacilityControl | None' = None,
  ) -> BalanceSheet:
    '''Dispatch trucks, apply penalties, and flush revenue.

    Args:
      control: Carries ``unit_price`` updates; None keeps the price.

    Returns:
      BalanceSheet: Revenue minus pending fees and freight costs.
    '''
    if control is not None and control.unit_price is not None:
      self.economy.unit_price = control.unit_price
    sheets = BalanceSheet()
    for truck in self.fleet:
      if len(self.order_queue) > 0 and not truck.is_enroute():
        order = self.order_queue.popleft()
        truck.schedule(
            self.facility.world,
            order.destination,
            order.product_id,
            order.quantity,
        )
      sheets += truck.act()
    pending = self.economy.pending_order_penalty * len(self.order_queue)
    self.economy.total_pending_order_penalties -= pending
    revenue = self.economy.order_checkin
    self.economy.order_checkin = 0
    return BalanceSheet(revenue, -pending) + sheets
