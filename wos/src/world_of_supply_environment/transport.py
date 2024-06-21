#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Transportation unit
'''

from dataclasses import dataclass

# imports
from .agent import Agent
from .balance_sheet import BalanceSheet

# imports


class Transport(Agent):

  '''
  Transportation unit
  '''

  @dataclass
  class Economy:
    unit_transport_cost: int   # cost per unit per movement

    def step_balance_sheet(self, transport):
      return BalanceSheet(
        0,
        -transport.payload * self.unit_transport_cost * abs(transport.step)
      )

  @dataclass
  class Control:
    pass

  def __init__(self, source, economy):
    self.source = source
    self.destination = None
    self.path = None
    self.location_pointer = 0
    self.step = 0
    self.payload = 0  # units
    self.economy = economy

  def schedule(self, world, destination, product_id, quantity):
    self.destination = destination
    self.product_id = product_id
    self.requested_quantity = quantity
    self.path = world.find_path(
      self.source.x,
      self.source.y,
      self.destination.x,
      self.destination.y
    )

    if self.path is None:
      raise Exception(f'Destination {destination} is unreachable')

    self.step = 1    # 1 - to destination, -1 - to source, 0 - finished
    self.location_pointer = 0

  def path_len(self):
    if self.path is None:
      return 0
    else:
      return len(self.path)

  def is_enroute(self):
    return self.destination is not None

  def current_location(self):
    if self.path is None:
      return (self.source.x, self.source.y)
    else:
      return self.path[self.location_pointer]

  def try_loading(self, quantity):
    if self.source.storage.try_take_units({self.product_id: quantity}):
      self.payload = quantity

  def try_unloading(self):
    unloaded = self.destination.storage.try_add_units(
      {self.product_id: self.payload},
      all_or_nothing=False
    )
    if len(unloaded) > 0:
      unloaded_units = sum(unloaded.values())
      self.destination.consumer.on_order_reception(
        self.source.id,
        self.product_id,
        unloaded_units
      )
      self.payload = 0
      # all units that were not sucessfully unloaded will be lost

  def act(self, control):
    if self.step > 0:
      if self.location_pointer == 0 and self.payload == 0:
        self.try_loading(self.requested_quantity)

      if self.payload > 0:     # will stay at the source until loaded
        if self.location_pointer < len(self.path) - 1:
          self.location_pointer += self.step
        else:
          self.step = -1   # arrived to the destination

    if self.step < 0:
      if self.location_pointer == len(self.path) - 1 and self.payload > 0:
        self.try_unloading()

      if self.payload == 0:    # will stay at the destination until unloaded
        if self.location_pointer > 0:
          self.location_pointer += self.step
        else:
          self.step = 0    # arrived back to the source
          self.destination = None

    return self.economy.step_balance_sheet(self)
