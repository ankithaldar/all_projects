#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Tests for seller demand curves and consumer open-order bookkeeping.'''

from world_of_supply.facility import FacilityControl, RetailerCell
from world_of_supply.scenario import ScenarioConfig, WorldBuilder
from world_of_supply.seller import SellerEconomy


def test_market_demand_is_linear_and_non_negative():
  economy = SellerEconomy(price_demand_intercept=50, price_demand_slope=0.005)
  assert economy.market_demand(0) == 50
  assert economy.market_demand(1800) == 41
  assert economy.market_demand(100000) == 0


def test_seller_sells_only_available_stock():
  world = WorldBuilder.build(ScenarioConfig(), seed=1)
  retailer = world.get_facilities(RetailerCell)[0]
  retailer.storage.stock_levels['toy_car'] = 3

  sheet = retailer.seller.act(FacilityControl(unit_price=1000))

  assert sheet.profit == 3000
  assert retailer.storage.used_capacity() == 0
  assert retailer.seller.economy.total_units_sold == 3


def test_open_orders_prune_after_full_reception():
  world = WorldBuilder.build(ScenarioConfig(), seed=2)
  warehouse = [f for f in world.facilities.values() if f.__class__.__name__ == 'WarehouseCell'][0]
  source = warehouse.consumer.sources[0]

  warehouse.consumer._shift_open_order(source.id, 'toy_car', 5)
  assert sum(warehouse.consumer.open_orders[source.id].values()) == 5

  warehouse.consumer.on_order_reception(source.id, 'toy_car', 5)
  assert source.id not in warehouse.consumer.open_orders


def test_scripted_policy_counts_all_stock_toward_capacity():
  from world_of_supply.facility import FacilityControl
  from world_of_supply.policies import ScriptedSupplyChainPolicy

  world = WorldBuilder.build(ScenarioConfig(), seed=2)
  retailer = world.get_facilities(RetailerCell)[0]
  retailer.storage.try_add_units({'toy_car': 9})
  source = retailer.consumer.sources[0]
  retailer.consumer._shift_open_order(source.id, 'toy_car', 5)

  product, source_index = ScriptedSupplyChainPolicy()._select_source(retailer)

  assert product is None and source_index is None
