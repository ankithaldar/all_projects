#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Seam tests for world registry, router, and scenario builders.'''

import random

import pytest

from world_of_supply.base import Agent, ConsumerUnitLike
from world_of_supply.base import DistributionUnitLike, FacilityLike
from world_of_supply.base import StorageUnitLike
from world_of_supply.facility import DistributionConfig, FacilityConfig
from world_of_supply.facility import ManufacturingConfig, RetailerCell
from world_of_supply.facility import SellerConfig, StorageConfig
from world_of_supply.facility import ToyFactoryCell, _storage_unit
from world_of_supply.geography import RailroadCell
from world_of_supply.manufacturing import BillOfMaterials
from world_of_supply.scenario import RailroadBuilder, ScenarioConfig
from world_of_supply.scenario import TierBuilder, WorldBuilder
from world_of_supply.world import FacilityRegistry, World, WorldRouter


def _facility_config():
  '''Build a facility config with distinctive narrow values.

  Returns:
    FacilityConfig: Config with capacity 7 and cost 3 values.
  '''
  return FacilityConfig(
      bill_of_materials=BillOfMaterials(),
      max_storage_capacity=7,
      unit_storage_cost=3,
      fleet_size=2,
      unit_transport_cost=4,
      unit_manufacturing_cost=9,
      price_demand_intercept=11.0,
      price_demand_slope=0.5,
      wrong_order_penalty=12,
      pending_order_penalty=13,
      initial_balance=9,
  )


def _railroad_count(world, y=2):
  '''Count railroad cells on one horizontal grid row.

  Args:
    world: World holding the grid under test.
    y: Grid row to scan for railroad cells.

  Returns:
    int: Number of railroad cells found on the row.
  '''
  return sum(1 for x in range(world.size_x) if world.is_railroad(x, y))


def test_registry_generate_id_increments():
  '''Verify registry issues monotonically increasing numeric ids.

  Returns:
    None: Test completes without assertion failure.
  '''
  registry = FacilityRegistry()
  assert registry.generate_id() == 1
  assert registry.generate_id() == 2
  assert registry.id_counter == 2


def test_registry_get_by_type_filters_facilities():
  '''Verify registry lookup filters facilities by class.

  Returns:
    None: Test completes without assertion failure.
  '''
  world = WorldBuilder.build(seed=42)
  toys = world._registry.get_by_type(ToyFactoryCell)
  assert len(toys) == 3
  assert all(isinstance(facility, ToyFactoryCell) for facility in toys)


def test_router_find_path_and_invalidate():
  '''Verify router finds paths and rebuilds after invalidation.

  Returns:
    None: Test completes without assertion failure.
  '''
  world = World(10, 10)
  world.place_cell(RailroadCell(2, 2), RailroadCell(3, 2))
  assert world.find_path((2, 2), (3, 2)) == [(2, 2), (3, 2)]
  world.place_cell(RailroadCell(3, 3))
  assert world.find_path((2, 2), (3, 3)) is not None
  router = WorldRouter(10, 10, world.is_traversable)
  router.invalidate()
  assert router.find_path((2, 2), (3, 3)) is not None
  assert router.find_path((2, 2), (9, 9)) is None


def test_world_public_facilities_is_registry_dict():
  '''Verify world facilities property exposes the registry dict.

  Returns:
    None: Test completes without assertion failure.
  '''
  world = WorldBuilder.build(seed=42)
  assert isinstance(world.facilities, dict)
  assert world.facilities is world._registry.facilities
  assert len(world.facilities) == 9


def test_railroad_builder_creates_path():
  '''Verify railroad builder lays an L-shaped path on the grid.

  Returns:
    None: Test completes without assertion failure.
  '''
  world = World(20, 20)
  builder = RailroadBuilder(world, rng=random.Random(1), attempts=5)
  builder.build_railroad(2, 2, 8, 2)
  assert _railroad_count(world) >= 5
  assert world.is_railroad(3, 2) or world.is_railroad(4, 2)


def test_railroad_builder_connect_multiple_destinations():
  '''Verify railroad connect joins one source to destinations.

  Returns:
    None: Test completes without assertion failure.
  '''

  class FakeFacility:
    '''Minimal facility stand-in carrying grid coordinates.'''

    def __init__(self, x, y):
      '''Store grid coordinates for connection endpoints.

      Args:
        x: Horizontal grid position.
        y: Vertical grid position.

      Returns:
        None: Stand-in is mutated in place.
      '''
      self.x = x
      self.y = y

  world = World(20, 20)
  builder = RailroadBuilder(world, rng=random.Random(1), attempts=5)
  source = FakeFacility(2, 2)
  builder.connect(source, FakeFacility(8, 2), FakeFacility(14, 2))
  assert _railroad_count(world) >= 5
  assert world.is_railroad(3, 2) or world.is_railroad(4, 2)


def test_tier_builder_spread_and_base_config():
  '''Verify tier builder spreads tiers and projects config values.

  Returns:
    None: Test completes without assertion failure.
  '''
  config = ScenarioConfig()
  world = World(config.world_size_x, config.world_size_y)
  railroad = RailroadBuilder(world, rng=random.Random(1))
  builder = TierBuilder(config, world, railroad)
  assert builder.spread(2) == [2, 14]
  assert builder.spread(3) == [2, 8, 14]
  built = builder.base_config(BillOfMaterials(), None, 100, 5)
  assert built.initial_balance == 100
  assert built.max_storage_capacity == 5
  assert built.unit_storage_cost == config.unit_storage_cost
  assert built.fleet_size == config.fleet_size


def test_worldbuilder_connect_shim_delegates():
  '''Verify WorldBuilder connect shim lays connecting railroads.

  Returns:
    None: Test completes without assertion failure.
  '''

  class FakeFacility:
    '''Minimal facility stand-in carrying grid coordinates.'''

    def __init__(self, x, y):
      '''Store grid coordinates for connection endpoints.

      Args:
        x: Horizontal grid position.
        y: Vertical grid position.

      Returns:
        None: Stand-in is mutated in place.
      '''
      self.x = x
      self.y = y

  world = World(20, 20)
  source = FakeFacility(2, 2)
  destination = FakeFacility(8, 2)
  WorldBuilder._connect(world, random.Random(1), source, destination)
  assert _railroad_count(world) >= 5


def test_worldbuilder_build_railroad_shim_delegates():
  '''Verify WorldBuilder railroad shim lays the requested path.

  Returns:
    None: Test completes without assertion failure.
  '''
  world = World(20, 20)
  WorldBuilder._build_railroad(world, random.Random(1), 2, 2, 8, 2, 5)
  assert _railroad_count(world) >= 5


def test_worldbuilder_seeded_layout_positions():
  '''Verify seeded builder produces the expected facility layout.

  Returns:
    None: Test completes without assertion failure.
  '''
  world = WorldBuilder.build(seed=42)
  assert len(world.facilities) == 9
  positions = sorted((f.x, f.y) for f in world.facilities.values())
  expected = [(10, 6), (10, 10), (35, 2), (35, 8), (35, 14)]
  expected += [(50, 2), (50, 14), (70, 2), (70, 14)]
  assert positions == sorted(expected)


def test_narrow_configs_project_from_facility_config():
  '''Verify narrow configs project only their own parameters.

  Returns:
    None: Test completes without assertion failure.
  '''
  config = _facility_config()
  storage = StorageConfig.from_facility_config(config)
  assert storage.max_storage_capacity == 7
  assert storage.unit_storage_cost == 3
  distribution = DistributionConfig.from_facility_config(config)
  assert distribution.fleet_size == 2
  assert distribution.unit_transport_cost == 4
  assert distribution.wrong_order_penalty == 12
  assert distribution.pending_order_penalty == 13
  manufacturing = ManufacturingConfig.from_facility_config(config)
  assert manufacturing.unit_manufacturing_cost == 9
  seller = SellerConfig.from_facility_config(config)
  assert seller.price_demand_intercept == 11.0
  assert seller.price_demand_slope == 0.5


def test_storage_factory_uses_narrow_config():
  '''Verify storage factory honors narrow storage capacity.

  Returns:
    None: Test completes without assertion failure.
  '''
  config = _facility_config()
  unit = _storage_unit(StorageConfig.from_facility_config(config))
  assert unit.max_capacity == 7


def test_facility_require_narrowers():
  '''Verify require helpers return installed units or raise.

  Returns:
    None: Test completes without assertion failure.
  '''
  world = WorldBuilder.build(seed=42)
  retailer = world.get_facilities(RetailerCell)[0]
  assert retailer.require_storage() is retailer.storage
  assert retailer.require_consumer() is retailer.consumer
  with pytest.raises(AssertionError):
    retailer.require_distribution()


def test_facility_unit_act_order_preserved():
  '''Verify facility units execute in storage to distribution order.

  Returns:
    None: Test completes without assertion failure.
  '''
  world = WorldBuilder.build(seed=42)
  facility = world.get_facilities(ToyFactoryCell)[0]
  order = [type(unit).__name__ for unit in facility._units]
  expected = ['StorageUnit', 'ConsumerUnit', 'ManufacturingUnit',
              'DistributionUnit']
  assert order == expected


def test_base_narrow_protocols_expose_expected_members():
  '''Verify base protocols expose only their narrow member sets.

  Returns:
    None: Test completes without assertion failure.
  '''
  storage_members = set(StorageUnitLike.__annotations__)
  assert storage_members == {'max_capacity', 'stock_levels'}
  methods = [m for m in dir(StorageUnitLike) if not m.startswith('_')]
  expected = {'used_capacity', 'available_capacity', 'try_add_units',
              'try_take_units', 'take_available'}
  assert set(methods) == expected
  consumer_members = set(ConsumerUnitLike.__annotations__)
  assert consumer_members == {'sources', 'open_orders'}
  assert hasattr(ConsumerUnitLike, 'on_order_reception')
  assert hasattr(DistributionUnitLike, 'place_order')
  facility_members = set(FacilityLike.__annotations__)
  assert facility_members == {'x', 'y', 'id', 'bom', 'storage', 'consumer',
                              'manufacturing', 'distribution', 'seller'}
  assert hasattr(Agent, 'act')
