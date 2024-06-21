#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''Distribution Unit'''


# imports
from collections import deque
from dataclasses import dataclass
#   script imports
from .agent import Agent
from .balance_sheet import BalanceSheet
from .cell import Cell
from .transport import Transport

# imports


class DistributionUnit(Agent):
  '''Distribution Unit'''

  @dataclass
  class Economy:
    '''Economy of Distribution Unit'''

    unit_price: int = 0
    wrong_order_penatly: int = 0
    pending_order_penalty: int = 0
    order_checkin: int = 0  # balance for the current time step

    total_wrong_order_penalties: int = 0
    total_pending_order_penalties: int = 0

    def profit(self, units_sold):
      return self.unit_price * units_sold

  @dataclass
  class Config:
    fleet_size: int
    unit_transport_cost: int
    wrong_order_penatly: int
    pending_order_penalty: int

  @dataclass
  class Control:
    unit_price: int

  @dataclass
  class Order:
    destination: Cell
    product_id: str
    quantity: int
    unit_price: int = 0

  def __init__(self,
    facility,
    fleet_size,
    distribution_economy,
    transport_economy
  ):
    self.facility = facility
    self.fleet = [Transport(facility, transport_economy)
    for i in range(fleet_size)]
    self.order_queue = deque()
    self.economy = distribution_economy

  def place_order(self, order):
    if order.quantity > 0:
      if self.facility.bom.output_product_id == order.product_id:
        order.unit_price = self.economy.unit_price
        self.order_queue.append(order)   # add order to the queue
        order_total = self.economy.profit(order.quantity)
        self.economy.order_checkin += order_total
        return -order_total
      else:
        penalty = -self.economy.wrong_order_penatly * order.quantity
        self.economy.total_wrong_order_penalties += penalty
        return -self.economy.wrong_order_penatly * order.quantity
    else:
      return 0

  def act(self, control):
    if control is not None:
      self.economy.unit_price = control.unit_price    # update unit price

    transportation_balance = BalanceSheet()
    for vechicle in self.fleet:
      if len(self.order_queue) > 0 and not vechicle.is_enroute():
        order = self.order_queue.popleft()
        vechicle.schedule(self.facility.world,
          order.destination,
          order.product_id,
          order.quantity
        )
        transportation_balance -= vechicle.act(None)
      else:
        transportation_balance -= vechicle.act(None)

    pending_orders_loss = - self.economy.pending_order_penalty\
      * len(self.order_queue)
    self.economy.total_pending_order_penalties += pending_orders_loss
    order_step_profit = self.economy.order_checkin
    self.economy.order_checkin = 0
    return BalanceSheet(
      order_step_profit,
      pending_orders_loss
    ) + transportation_balance
