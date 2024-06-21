#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Simple Control Policy
'''

import random as rnd
from collections import Counter

# imports
from .facility_cell import FacilityCell
from .raw_materials_factory_cell import RawMaterialsFactoryCell
from .retailer_cell import RetailerCell
from .value_add_factory_cell import ValueAddFactoryCell
from .warehouse_cell import WarehouseCell
from .world import World

# imports


class SimpleControlPolicy:
  '''
  Simple Control Policy
  '''

  def compute_control(self, world):

    def default_facility_control(unit_price, product_id, source_id):
      return FacilityCell.Control(
        unit_price = unit_price,
        production_rate = 4,
        consumer_product_id = product_id,
        consumer_source_id = source_id,
        consumer_quantity = 8
      )

    ctrl = dict()
    for f in world.get_facilities(RawMaterialsFactoryCell):
      ctrl[f.id] = default_facility_control(400, None, None)

    for f in world.get_facilities(ValueAddFactoryCell):
      ctrl[f.id] = default_facility_control(1000, *self._find_source(f))

    for f in world.get_facilities(WarehouseCell):
      ctrl[f.id] = default_facility_control(1400, *self._find_source(f))

    for f in world.get_facilities(RetailerCell):
      ctrl[f.id] = default_facility_control(1800, *self._find_source(f))

    return World.Control(ctrl)

  def _find_source(self, facility):
    # stop placing orders when the facility ran out of money
    if facility.economy.total_balance.total() <= 0:
      return (None, None)

    inputs = facility.bom.inputs
    available_inventory = facility.storage.stock_levels
    in_transit_orders = sum(facility.consumer.open_orders.values(), Counter())
    booked_inventory = available_inventory + in_transit_orders

    # stop placing orders when the facilty runs out of capacity
    if sum(booked_inventory.values()) > facility.storage.max_capacity:
      return (None, None)

    most_neeed_product_id = None
    min_ratio = float('inf')
    for product_id, quantity in inputs.items():
      fulfillment_ratio = booked_inventory[product_id] / quantity
      if fulfillment_ratio < min_ratio:
        min_ratio = fulfillment_ratio
        most_neeed_product_id = product_id

    exporting_sources = SimpleControlPolicy.find_exporting_sources(facility, most_neeed_product_id)

    return (most_neeed_product_id, rnd.choice(exporting_sources))

  @staticmethod
  def find_exporting_sources(facility, product_id):
    exporting_sources = []
    if product_id is not None and facility.consumer is not None:
      for source_id, source in enumerate(facility.consumer.sources):
        if source.bom.output_product_id == product_id:
          exporting_sources.append(source_id)
    return exporting_sources
