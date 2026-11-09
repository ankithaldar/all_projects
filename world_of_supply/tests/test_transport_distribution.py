#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for order placement, transport delivery, and penalties.'''

from world_of_supply.distribution import DistributionEconomy, Order
from world_of_supply.distribution import TransportEconomy
from world_of_supply.facility import FacilityControl, RetailerCell
from world_of_supply.facility import ToyFactoryCell, WarehouseCell
from world_of_supply.scenario import ScenarioConfig, WorldBuilder
from world_of_supply.transport import TransportState
from world_of_supply.world import Control


def build_world():
  '''Build the shared seeded world for delivery tests.

  Returns:
    World: Fresh world built with scenario defaults and seed 7.
  '''
  return WorldBuilder.build(ScenarioConfig(), seed=7)


def test_place_order_valid_product_queues_and_charges():
  '''Verify valid order queues and charges the buyer unit price.

  Returns:
    None: Test completes without assertion failure.
  '''
  world = build_world()
  toy_factory = world.get_facilities(ToyFactoryCell)[0]
  buyer = world.get_facilities(WarehouseCell)[0]
  payment = toy_factory.distribution.place_order(Order(buyer, 'toy_car', 5))
  price = toy_factory.distribution.economy.unit_price
  assert payment == price * 5
  assert len(toy_factory.distribution.order_queue) == 1


def test_place_order_wrong_product_penalizes_but_does_not_queue():
  '''Verify wrong-product order penalizes without queueing.

  Returns:
    None: Test completes without assertion failure.
  '''
  world = build_world()
  toy_factory = world.get_facilities(ToyFactoryCell)[0]
  buyer = world.get_facilities(WarehouseCell)[0]
  payment = toy_factory.distribution.place_order(Order(buyer, 'steel', 2))
  assert payment == 1000
  penalties = toy_factory.distribution.economy.total_wrong_order_penalties
  assert penalties == -1000
  assert len(toy_factory.distribution.order_queue) == 0


def test_consumer_order_books_payment_and_open_orders():
  '''Verify consumer order books payment and open-order totals.

  Returns:
    None: Test completes without assertion failure.
  '''
  world = WorldBuilder.build(ScenarioConfig(), seed=3)
  retailer = world.get_facilities(RetailerCell)[0]
  warehouse = retailer.consumer.sources[0]
  warehouse.distribution.economy.unit_price = 1400
  control = FacilityControl(
      unit_price=None,
      production_rate=None,
      consumer_product_id='toy_car',
      consumer_source_id=0,
      consumer_quantity=4,
  )
  sheet = retailer.consumer.act(control)
  assert sheet.total() == -1400 * 4
  booked = retailer.consumer.open_orders[warehouse.id].values()
  assert sum(booked) == 4
  assert retailer.consumer.economy.total_units_purchased == 4


def test_delivery_cycle_moves_goods_to_buyer():
  '''Verify delivery cycle moves ordered goods to the buyer.

  Returns:
    None: Test completes without assertion failure.
  '''
  world = WorldBuilder.build(ScenarioConfig(), seed=5)
  toy_factory = world.get_facilities(ToyFactoryCell)[0]
  warehouse = world.get_facilities(WarehouseCell)[0]
  toy_factory.storage.stock_levels['toy_car'] = 6
  sources = warehouse.consumer.sources
  warehouse.consumer.act(FacilityControl(
      consumer_product_id='toy_car',
      consumer_source_id=sources.index(toy_factory),
      consumer_quantity=3,
      unit_price=None,
      production_rate=None,
  ))
  for _ in range(60):
    world.act(Control({}))
    if warehouse.storage.stock_levels.get('toy_car', 0) >= 3:
      break
  assert warehouse.storage.stock_levels['toy_car'] >= 3
  open_orders = warehouse.consumer.open_orders.values()
  assert sum(sum(c.values()) for c in open_orders) == 0


def test_transport_state_transitions():
  '''Verify transport moves from idle through scheduling to unload.

  Returns:
    None: Test completes without assertion failure.
  '''
  world = build_world()
  toy_factory = world.get_facilities(ToyFactoryCell)[0]
  truck = toy_factory.distribution.fleet[0]
  assert truck.state == TransportState.IDLE
  destination = world.get_facilities(WarehouseCell)[0]
  truck.schedule(world, destination, 'toy_car', 2)
  assert truck.is_enroute()
  assert truck.current_location() == (toy_factory.x, toy_factory.y)
  truck.payload = 2
  for _ in range(truck.path_len()):
    truck.act()
  assert truck.state in (TransportState.UNLOADING, TransportState.RETURNING)


def test_unfitting_units_are_lost_not_retained():
  '''Verify units beyond buyer capacity are lost, not retained.

  Returns:
    None: Test completes without assertion failure.
  '''
  world = WorldBuilder.build(ScenarioConfig(), seed=5)
  toy_factory = world.get_facilities(ToyFactoryCell)[0]
  warehouse = world.get_facilities(WarehouseCell)[0]
  truck = toy_factory.distribution.fleet[0]
  truck.schedule(world, warehouse, 'toy_car', 5)
  truck.payload = 5
  warehouse.storage.try_add_units({'toy_car': 18})
  delivered = truck.try_unloading()
  assert delivered == 2
  assert truck.payload == 0
  assert warehouse.storage.used_capacity() == 20
  open_orders = warehouse.consumer.open_orders.values()
  assert sum(sum(c.values()) for c in open_orders) == 0


def test_distribution_unit_assembly_uses_config():
  '''Verify distribution profit and transport step cost formulas.

  Returns:
    None: Test completes without assertion failure.
  '''
  economy = DistributionEconomy(unit_price=10)
  assert economy.profit(3) == 30
  transport = TransportEconomy(unit_transport_cost=2)
  assert transport.step_balance_sheet(3, 2).loss == -12
