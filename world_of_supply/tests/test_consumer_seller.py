#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for seller demand curves and consumer open-order book.'''

from world_of_supply.facility import FacilityControl, RetailerCell
from world_of_supply.facility import WarehouseCell
from world_of_supply.policies import ScriptedSupplyChainPolicy
from world_of_supply.scenario import ScenarioConfig, WorldBuilder
from world_of_supply.seller import SellerEconomy


def test_market_demand_is_linear_and_non_negative():
  '''Verify market demand is linear and clipped at zero.

  Returns:
    None: Test completes without assertion failure.
  '''
  economy = SellerEconomy(price_demand_intercept=50, price_demand_slope=0.005)
  assert economy.market_demand(0) == 50
  assert economy.market_demand(1800) == 41
  assert economy.market_demand(100000) == 0


def test_seller_sells_only_available_stock():
  '''Verify seller sells only stocked units at the set price.

  Returns:
    None: Test completes without assertion failure.
  '''
  world = WorldBuilder.build(ScenarioConfig(), seed=1)
  retailer = world.get_facilities(RetailerCell)[0]
  retailer.storage.stock_levels['toy_car'] = 3
  sheet = retailer.seller.act(FacilityControl(unit_price=1000))
  assert sheet.profit == 3000
  assert retailer.storage.used_capacity() == 0
  assert retailer.seller.economy.total_units_sold == 3


def test_open_orders_prune_after_full_reception():
  '''Verify open orders prune once a delivery completes them.

  Returns:
    None: Test completes without assertion failure.
  '''
  world = WorldBuilder.build(ScenarioConfig(), seed=2)
  warehouse = world.get_facilities(WarehouseCell)[0]
  source = warehouse.consumer.sources[0]
  warehouse.consumer._shift_open_order(source.id, 'toy_car', 5)
  booked = warehouse.consumer.open_orders[source.id].values()
  assert sum(booked) == 5
  warehouse.consumer.on_order_reception(source.id, 'toy_car', 5)
  assert source.id not in warehouse.consumer.open_orders


def test_scripted_policy_counts_all_stock_toward_capacity():
  '''Verify scripted policy skips sources when storage is full.

  Returns:
    None: Test completes without assertion failure.
  '''
  world = WorldBuilder.build(ScenarioConfig(), seed=2)
  retailer = world.get_facilities(RetailerCell)[0]
  retailer.storage.try_add_units({'toy_car': 9})
  source = retailer.consumer.sources[0]
  retailer.consumer._shift_open_order(source.id, 'toy_car', 5)
  product, source_index = ScriptedSupplyChainPolicy()._select_source(retailer)
  assert product is None and source_index is None
