#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Default scenario construction (Builder pattern).

Builds the classic supply chain: steel and lumber on the west edge, toy
factories in the middle, warehouses east of them, retailers at the far east,
all connected by L-shaped railroads.
'''

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from world_of_supply.facility import (
    FacilityConfig,
    LumberFactoryCell,
    RetailerCell,
    SteelFactoryCell,
    ToyFactoryCell,
    WarehouseCell,
)
from world_of_supply.geography import RailroadCell
from world_of_supply.manufacturing import BillOfMaterials
from world_of_supply.world import World

STEEL = 'steel'
LUMBER = 'lumber'
TOY_CAR = 'toy_car'
PRODUCT_IDS: tuple[str, ...] = (LUMBER, STEEL, TOY_CAR)


@dataclass
class ScenarioConfig:
  '''Tunable parameters of the default scenario.

  Attributes:
    world_size_x / world_size_y: Grid dimensions.
    max_storage_capacity: Storage capacity for most facilities.
    retailer_storage_capacity: Smaller capacity forcing frequent reorders.
    unit_storage_cost / unit_transport_cost / unit_manufacturing_cost:
      Per-unit operating costs.
    fleet_size: Trucks per distribution unit.
    wrong_order_penalty / pending_order_penalty: Distribution penalties.
    price_demand_intercept / price_demand_slope: Retailer demand curve.
    raw_materials_balance / warehouse_balance / retailer_balance:
      Starting cash positions.
    n_toy_factories / n_warehouses / n_retailers: Tier sizes.
    railroad_elbow_attempts: Randomization attempts for road elbows.
  '''

  world_size_x: int = 80
  world_size_y: int = 16
  max_storage_capacity: int = 20
  retailer_storage_capacity: int = 10
  unit_storage_cost: int = 1
  unit_transport_cost: int = 1
  unit_manufacturing_cost: int = 100
  fleet_size: int = 1
  wrong_order_penalty: int = 500
  pending_order_penalty: int = 4
  price_demand_intercept: float = 50.0
  price_demand_slope: float = 0.005
  raw_materials_balance: int = 1000
  warehouse_balance: int = 2000
  retailer_balance: int = 3000
  n_toy_factories: int = 3
  n_warehouses: int = 2
  n_retailers: int = 2
  railroad_elbow_attempts: int = 5


class WorldBuilder:
  '''Constructs fully wired worlds from a :class:`ScenarioConfig`.'''

  @staticmethod
  def build(config: ScenarioConfig | None = None, seed: int | None = None) -> World:
    '''Build the default supply-chain scenario.

    Args:
      config: Scenario parameters; ``None`` uses the defaults.
      seed: Optional seed making railroad jitter reproducible.

    Returns:
      World: Populated world with all facilities registered.
    '''
    config = config or ScenarioConfig()
    rng = random.Random(seed)
    world = World(config.world_size_x, config.world_size_y)

    steel_bom = BillOfMaterials(Counter(), output_product_id=STEEL, output_lot_size=1)
    lumber_bom = BillOfMaterials(Counter(), output_product_id=LUMBER, output_lot_size=1)
    toy_bom = BillOfMaterials(Counter({LUMBER: 1, STEEL: 1}), output_product_id=TOY_CAR)
    passthrough_bom = BillOfMaterials(Counter({TOY_CAR: 1}), output_product_id=TOY_CAR)

    margin = 2
    usable_height = config.world_size_y - 2 * margin

    def base_config(bom: BillOfMaterials, sources: list | None, balance: int, capacity: int) -> FacilityConfig:
      '''Create a FacilityConfig from scenario-level parameters.

      Args:
        bom: Production recipe.
        sources: Upstream suppliers.
        balance: Starting capital.
        capacity: Storage capacity override.

      Returns:
        FacilityConfig: Fully parameterized configuration.
      '''
      return FacilityConfig(
          bill_of_materials=bom,
          max_storage_capacity=capacity,
          unit_storage_cost=config.unit_storage_cost,
          fleet_size=config.fleet_size,
          unit_transport_cost=config.unit_transport_cost,
          sources=sources,
          wrong_order_penalty=config.wrong_order_penalty,
          pending_order_penalty=config.pending_order_penalty,
          unit_manufacturing_cost=config.unit_manufacturing_cost,
          price_demand_intercept=config.price_demand_intercept,
          price_demand_slope=config.price_demand_slope,
          initial_balance=balance,
      )

    def spread(count: int) -> list[int]:
      '''Distribute ``count`` facilities vertically like the legacy layout.

      Args:
        count: Number of facilities.

      Returns:
        list[int]: Y positions top-to-bottom.
      '''
      return [int(usable_height / (count - 1) * i + margin) for i in range(count)]

    raw_materials = [
        SteelFactoryCell(10, 6, world, base_config(steel_bom, None, config.raw_materials_balance, config.max_storage_capacity)),
        LumberFactoryCell(10, 10, world, base_config(lumber_bom, None, config.raw_materials_balance, config.max_storage_capacity)),
    ]

    factories = []
    for y in spread(config.n_toy_factories):
      factory = ToyFactoryCell(
          35, y, world,
          base_config(toy_bom, raw_materials, config.raw_materials_balance, config.max_storage_capacity),
      )
      factories.append(factory)
      WorldBuilder._connect(world, rng, factory, *raw_materials, attempts=config.railroad_elbow_attempts)

    warehouses = []
    for y in spread(config.n_warehouses):
      warehouse = WarehouseCell(
          50, y, world,
          base_config(passthrough_bom, factories, config.warehouse_balance, config.max_storage_capacity),
      )
      warehouses.append(warehouse)
      WorldBuilder._connect(world, rng, warehouse, *factories, attempts=config.railroad_elbow_attempts)

    retailers = []
    for y in spread(config.n_retailers):
      retailer = RetailerCell(
          70, y, world,
          base_config(passthrough_bom, warehouses, config.retailer_balance, config.retailer_storage_capacity),
      )
      retailers.append(retailer)
      WorldBuilder._connect(world, rng, retailer, *warehouses, attempts=config.railroad_elbow_attempts)

    for facility in raw_materials + factories + warehouses + retailers:
      world.place_cell(facility)
      world.register_facility(facility)

    return world

  @staticmethod
  def _connect(world: World, rng: random.Random, source, *destinations, attempts: int = 5) -> None:
    '''Connect one facility to several destinations by rail.

    Args:
      world: Target world.
      rng: Random source for elbow jitter.
      source: Origin facility.
      *destinations: Facilities to connect to.
      attempts: Elbow-placement attempts before accepting a collision.
    '''
    for destination in destinations:
      WorldBuilder._build_railroad(world, rng, source.x, source.y, destination.x, destination.y, attempts)

  @staticmethod
  def _build_railroad(world: World, rng: random.Random, x1: int, y1: int, x2: int, y2: int, attempts: int) -> None:
    '''Lay an L-shaped railroad with a jittered vertical elbow.

    Unlike the legacy builder this always defines the elbow position: after
    ``attempts`` failed non-adjacency tries it accepts the last candidate
    instead of raising.

    Args:
      world: Target world.
      rng: Random source.
      x1 / y1: Start coordinates.
      x2 / y2: End coordinates.
      attempts: Placement attempts.
    '''
    step_x = (x2 > x1) - (x2 < x1)
    step_y = (y2 > y1) - (y2 < y1)
    xi = min(x1, x2) + int(abs(x2 - x1) * 0.5)
    for _ in range(attempts):
      candidate = min(x1, x2) + int(abs(x2 - x1) * rng.uniform(0.15, 0.85))
      adjacent_blocked = world.is_railroad(candidate - 1, y1 + step_y) or world.is_railroad(candidate + 1, y1 + step_y)
      if not adjacent_blocked:
        xi = candidate
        break

    for x in range(x1 + step_x, xi, step_x):
      world.create_cell(x, y1, RailroadCell)
    if step_y != 0:
      for y in range(y1, y2, step_y):
        world.create_cell(xi, y, RailroadCell)
    for x in range(xi, x2, step_x):
      world.create_cell(x, y2, RailroadCell)
