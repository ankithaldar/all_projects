#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Default scenario construction (Builder pattern).

Builds the classic supply chain: steel and lumber on the west
edge, toy factories in the middle, warehouses east of them,
retailers at the far east, all connected by L-shaped railroads.
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
    world_size_x: Grid width.
    world_size_y: Grid height.
    max_storage_capacity: Storage capacity for most facilities.
    retailer_storage_capacity: Smaller capacity for retailers.
    unit_storage_cost: Holding cost per unit per tick.
    unit_transport_cost: Freight cost per unit per cell.
    unit_manufacturing_cost: Production cost per unit.
    fleet_size: Trucks per distribution unit.
    wrong_order_penalty: Penalty per foreign-product unit.
    pending_order_penalty: Fee per queued order per tick.
    price_demand_intercept: Retailer demand-curve intercept.
    price_demand_slope: Retailer demand-curve slope.
    raw_materials_balance: Starting cash for raw materials.
    warehouse_balance: Starting cash for warehouses.
    retailer_balance: Starting cash for retailers.
    n_toy_factories: Number of toy factories.
    n_warehouses: Number of warehouses.
    n_retailers: Number of retailers.
    railroad_elbow_attempts: Randomization attempts for elbows.
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


class RailroadBuilder:
  '''Lays L-shaped railroads with jittered vertical elbows.

  Extracted from :class:`WorldBuilder` (Single Responsibility):
  only railroad geometry lives here.

  Attributes:
    world: Target world receiving railroad cells.
    rng: Random source for elbow jitter.
    attempts: Placement attempts before accepting a collision.
  '''

  def __init__(
      self,
      world: World,
      rng: random.Random | None = None,
      attempts: int = 5,
      seed: int | None = None,
  ) -> None:
    '''Bind the builder to a world and random source.

    Args:
      world: Target world.
      rng: Random source (created from ``seed`` when omitted).
      attempts: Placement attempts.
      seed: Optional seed used when ``rng`` is omitted.

    Returns:
      None: The builder is mutated in place.
    '''
    self.world = world
    self.rng = rng if rng is not None else random.Random(seed)
    self.attempts = attempts

  def connect(self, source, *destinations) -> None:
    '''Connect one facility to several destinations by rail.

    Args:
      source: Origin facility.
      *destinations: Facilities to connect to.

    Returns:
      None: The world grid is mutated in place.
    '''
    for destination in destinations:
      self.build_railroad(
          source.x, source.y, destination.x, destination.y,
      )

  def build_railroad(
      self, x1: int, y1: int, x2: int, y2: int,
  ) -> None:
    '''Lay an L-shaped railroad with a jittered vertical elbow.

    After ``attempts`` failed non-adjacency tries the last
    candidate is accepted instead of raising.

    Args:
      x1: Start x coordinate.
      y1: Start y coordinate.
      x2: End x coordinate.
      y2: End y coordinate.

    Returns:
      None: The world grid is mutated in place.
    '''
    self._build_railroad(x1, y1, x2, y2)

  def _build_railroad(
      self, x1: int, y1: int, x2: int, y2: int,
  ) -> None:
    '''Lay the railroad (internal worker).

    Args:
      x1: Start x coordinate.
      y1: Start y coordinate.
      x2: End x coordinate.
      y2: End y coordinate.

    Returns:
      None: The world grid is mutated in place.
    '''
    world = self.world
    rng = self.rng
    attempts = self.attempts
    step_x = (x2 > x1) - (x2 < x1)
    step_y = (y2 > y1) - (y2 < y1)
    span_x = abs(x2 - x1)
    xi = min(x1, x2) + int(span_x * 0.5)
    for _ in range(attempts):
      offset = int(span_x * rng.uniform(0.15, 0.85))
      candidate = min(x1, x2) + offset
      west = world.is_railroad(candidate - 1, y1 + step_y)
      east = world.is_railroad(candidate + 1, y1 + step_y)
      if not (west or east):
        xi = candidate
        break
    for x in range(x1 + step_x, xi, step_x):
      world.create_cell(x, y1, RailroadCell)
    if step_y != 0:
      for y in range(y1, y2, step_y):
        world.create_cell(xi, y, RailroadCell)
    for x in range(xi, x2, step_x):
      world.create_cell(x, y2, RailroadCell)


def _bills_of_materials() -> tuple:
  '''Create the bills of materials for the default scenario.

  Returns:
    tuple: ``(steel, lumber, toy, passthrough)`` BOMs.
  '''
  steel = BillOfMaterials(Counter(), STEEL, 1)
  lumber = BillOfMaterials(Counter(), LUMBER, 1)
  inputs = Counter({LUMBER: 1, STEEL: 1})
  toy = BillOfMaterials(inputs, TOY_CAR)
  passthrough = BillOfMaterials(Counter({TOY_CAR: 1}), TOY_CAR)
  return (steel, lumber, toy, passthrough)


class TierBuilder:
  '''Builds one supply-chain tier at a time.

  Extracted from :meth:`WorldBuilder.build` (Single Responsibility):
  each tier builder owns one column of facilities and its
  connections to the previous tier.

  Attributes:
    config: Scenario parameters.
    world: Target world.
    railroad: Railroad builder for inter-tier connections.
  '''

  def __init__(
      self,
      config: ScenarioConfig,
      world: World,
      railroad: RailroadBuilder,
  ) -> None:
    '''Bind the tier builder to a scenario and world.

    Args:
      config: Scenario parameters.
      world: Target world.
      railroad: Railroad builder for connections.

    Returns:
      None: The builder is mutated in place.
    '''
    self.config = config
    self.world = world
    self.railroad = railroad
    margin = 2
    self.margin = margin
    self.usable_height = config.world_size_y - 2 * margin

  def spread(self, count: int) -> list[int]:
    '''Distribute ``count`` facilities vertically.

    Args:
      count: Number of facilities.

    Returns:
      list[int]: Y positions top-to-bottom.
    '''
    height = self.usable_height
    result = [int(height / (count - 1) * i + self.margin) for i in range(count)]
    return result

  def base_config(
      self,
      bom: BillOfMaterials,
      sources: list | None,
      balance: int,
      capacity: int,
  ) -> FacilityConfig:
    '''Create a FacilityConfig from scenario-level parameters.

    Args:
      bom: Production recipe.
      sources: Upstream suppliers.
      balance: Starting capital.
      capacity: Storage capacity override.

    Returns:
      FacilityConfig: Fully parameterized configuration.
    '''
    cfg = self.config
    return FacilityConfig(
        bill_of_materials=bom,
        max_storage_capacity=capacity,
        unit_storage_cost=cfg.unit_storage_cost,
        fleet_size=cfg.fleet_size,
        unit_transport_cost=cfg.unit_transport_cost,
        sources=sources,
        wrong_order_penalty=cfg.wrong_order_penalty,
        pending_order_penalty=cfg.pending_order_penalty,
        unit_manufacturing_cost=cfg.unit_manufacturing_cost,
        price_demand_intercept=cfg.price_demand_intercept,
        price_demand_slope=cfg.price_demand_slope,
        initial_balance=balance,
    )

  def build_raw_materials(self, steel_bom, lumber_bom) -> list:
    '''Build the raw-materials tier (west edge).

    Args:
      steel_bom: Steel production recipe.
      lumber_bom: Lumber production recipe.

    Returns:
      list: Raw-materials facilities in creation order.
    '''
    cfg = self.config
    capacity = cfg.max_storage_capacity
    balance = cfg.raw_materials_balance
    steel = SteelFactoryCell(
        10, 6, self.world, self.base_config(steel_bom, None, balance, capacity),
    )
    lumber = LumberFactoryCell(
        10, 10, self.world,
        self.base_config(lumber_bom, None, balance, capacity),
    )
    return [steel, lumber]

  def build_factories(self, bom, sources) -> list:
    '''Build the toy-factories tier with connections.

    Args:
      bom: Toy production recipe.
      sources: Raw-materials facilities.

    Returns:
      list: Factory facilities in creation order.
    '''
    cfg = self.config
    capacity = cfg.max_storage_capacity
    balance = cfg.raw_materials_balance
    factories = []
    for y in self.spread(cfg.n_toy_factories):
      config = self.base_config(bom, sources, balance, capacity)
      factory = ToyFactoryCell(35, y, self.world, config)
      factories.append(factory)
      self.railroad.connect(factory, *sources)
    return factories

  def build_warehouses(self, bom, sources) -> list:
    '''Build the warehouses tier with connections.

    Args:
      bom: Passthrough production recipe.
      sources: Toy-factories facilities.

    Returns:
      list: Warehouse facilities in creation order.
    '''
    cfg = self.config
    capacity = cfg.max_storage_capacity
    balance = cfg.warehouse_balance
    warehouses = []
    for y in self.spread(cfg.n_warehouses):
      config = self.base_config(bom, sources, balance, capacity)
      warehouse = WarehouseCell(50, y, self.world, config)
      warehouses.append(warehouse)
      self.railroad.connect(warehouse, *sources)
    return warehouses

  def build_retailers(self, bom, sources) -> list:
    '''Build the retailers tier with connections.

    Args:
      bom: Passthrough production recipe.
      sources: Warehouse facilities.

    Returns:
      list: Retailer facilities in creation order.
    '''
    cfg = self.config
    capacity = cfg.retailer_storage_capacity
    balance = cfg.retailer_balance
    retailers = []
    for y in self.spread(cfg.n_retailers):
      config = self.base_config(bom, sources, balance, capacity)
      retailer = RetailerCell(70, y, self.world, config)
      retailers.append(retailer)
      self.railroad.connect(retailer, *sources)
    return retailers


class WorldBuilder:
  '''Constructs fully wired worlds from a :class:`ScenarioConfig`.

  The builder coordinates :class:`TierBuilder` and
  :class:`RailroadBuilder` without owning tier geometry itself.

  Attributes:
    None: Builder is stateless; configuration lives in the
      tier and railroad builders.
  '''

  @staticmethod
  def build(
      config: ScenarioConfig | None = None,
      seed: int | None = None,
  ) -> World:
    '''Build the default supply-chain scenario.

    Args:
      config: Scenario parameters; ``None`` uses the defaults.
      seed: Optional seed making railroad jitter reproducible.

    Returns:
      World: Populated world with all facilities registered.
    '''
    scenario = config or ScenarioConfig()
    rng = random.Random(seed)
    world = World(scenario.world_size_x, scenario.world_size_y)
    attempts = scenario.railroad_elbow_attempts
    railroad = RailroadBuilder(world, rng, attempts)
    builder = TierBuilder(scenario, world, railroad)
    steel_bom, lumber_bom, toy_bom, passthrough = _bills_of_materials()
    raw_materials = builder.build_raw_materials(steel_bom, lumber_bom)
    factories = builder.build_factories(toy_bom, raw_materials)
    warehouses = builder.build_warehouses(passthrough, factories)
    retailers = builder.build_retailers(passthrough, warehouses)
    tiers = raw_materials + factories + warehouses + retailers
    for facility in tiers:
      world.place_cell(facility)
      world.register_facility(facility)
    return world

  @staticmethod
  def _connect(world: World, rng, source, *destinations, attempts=5) -> None:
    '''Connect one facility to destinations (legacy shim).

    Args:
      world: Target world.
      rng: Random source for elbow jitter.
      source: Origin facility.
      *destinations: Facilities to connect to.
      attempts: Elbow-placement attempts.

    Returns:
      None: The world grid is mutated in place.
    '''
    builder = RailroadBuilder(world, rng, attempts)
    builder.connect(source, *destinations)

  @staticmethod
  def _build_railroad(world, rng, x1, y1, x2, y2, attempts) -> None:
    '''Lay an L-shaped railroad (legacy shim).

    Args:
      world: Target world.
      rng: Random source.
      x1: Start x coordinate.
      y1: Start y coordinate.
      x2: End x coordinate.
      y2: End y coordinate.
      attempts: Placement attempts.

    Returns:
      None: The world grid is mutated in place.
    '''
    builder = RailroadBuilder(world, rng, attempts)
    builder.build_railroad(x1, y1, x2, y2)
