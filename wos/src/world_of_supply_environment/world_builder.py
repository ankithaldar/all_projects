#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''World Builder'''


# imports
from collections import Counter
import numpy as np
import random as rnd
#   script imports
from .bill_of_materials import BillOfMaterials
from .facility_cell import FacilityCell
from .lumber_factory_cell import LumberFactoryCell
from .railroad_cell import RailroadCell
from .retailer_cell import RetailerCell
from .steel_factory_cell import SteelFactoryCell
from .terrain_cell import TerrainCell
from .toy_factory_cell import ToyFactoryCell
from .warehouse_cell import WarehouseCell
from .world import World
# imports


class WorldBuilder:
  '''World Builder'''

  @staticmethod
  def create(x=80, y=16):
    world = World(x, y)
    world.grid = [[TerrainCell(xi, yi) for yi in range(y)] for xi in range(x)]

    # parameters
    def default_facility_config(bom, sources):
      return FacilityCell.Config(
        bill_of_materials=bom,
        max_storage_capacity=20,
        unit_storage_cost=1,
        fleet_size=1,
        unit_transport_cost=1,
        sources=sources,
        wrong_order_penatly=500,
        pending_order_penalty=4,
        unit_manufacturing_cost=100,
        price_demand_intercept=50,
        price_demand_slope=0.005
      )

    def default_economy_config(initial_balance=1000):
      return FacilityCell.EconomyConfig(initial_balance)

    steel_bom = BillOfMaterials(Counter(), 'steel', 1)
    lumber_bom = BillOfMaterials(Counter(), 'lumber', 1)
    toy_bom = BillOfMaterials(Counter({'lumber': 1, 'steel': 1}), 'toy_car')
    distribution_bom = BillOfMaterials(Counter({'toy_car': 1}), 'toy_car')
    retailer_bom = BillOfMaterials(Counter({'toy_car': 1}), 'toy_car', 1)

    # facility placement
    map_margin = 2
    size_y_margins = world.size_y - 2*map_margin

    # raw materials
    steel_01 = SteelFactoryCell(
      x=10,
      y=6,
      world=world,
      config=default_facility_config(
        bom=steel_bom,
        sources=None
      ),
      economy_config=default_economy_config()
    )
    lumber_01 = LumberFactoryCell(
      x=10,
      y=10,
      world=world,
      config=default_facility_config(
        bom=lumber_bom,
        sources=None
      ),
      economy_config=default_economy_config()
    )
    raw_materials = [steel_01, lumber_01]
    world.place_cell(*raw_materials)

    # manufacturing
    n_toy_factories = 3
    factories = []
    for i in range(n_toy_factories):
      f = ToyFactoryCell(
        35,
        int(size_y_margins/(n_toy_factories - 1)*i + map_margin),
        world,
        default_facility_config(
          toy_bom,
          raw_materials
        ),
        default_economy_config()
      )
      world.place_cell(f)
      factories.append(f)
      WorldBuilder.connect_cells(world, f, *raw_materials)

    # distribution
    n_warehouses = 2
    warehouses = []
    for i in range(n_warehouses):
      w = WarehouseCell(
        50,
        int(size_y_margins/(n_warehouses - 1)*i + map_margin),
        world,
        default_facility_config(
          distribution_bom,
          factories
        ),
        default_economy_config(2000)
      )
      world.place_cell(w)
      warehouses.append(w)
      WorldBuilder.connect_cells(world, w, *factories)

    # final consumers
    n_retailers = 2
    retailers = []
    for i in range(n_retailers):
      retailer_config = default_facility_config(retailer_bom, warehouses)
      retailer_config.max_storage_capacity = 10
      r = RetailerCell(
        70,
        int(size_y_margins/(n_retailers - 1)*i + map_margin),
        world,
        retailer_config,
        default_economy_config(3000)
      )
      world.place_cell(r)
      retailers.append(r)
      WorldBuilder.connect_cells(world, r, *warehouses)

    for facility in raw_materials + factories + warehouses + retailers:
      world.facilities[facility.id] = facility

    return world


  @staticmethod
  def connect_cells(world, source, *destinations):
    for dest_cell in destinations:
      WorldBuilder.build_railroad(
          world, source.x, source.y, dest_cell.x, dest_cell.y)


  @staticmethod
  def build_railroad(world, x1, y1, x2, y2):
    step_x = np.sign(x2 - x1)
    step_y = np.sign(y2 - y1)

    # make several attempts to find a route non-adjacent to existing roads
    for _ in range(5):
      xi = min(x1, x2) + int(abs(x2 - x1) * rnd.uniform(0.15, 0.85))
      if not (world.is_railroad(xi-1, y1+step_y) or
      world.is_railroad(xi+1, y1+step_y)):
        break

    for x in range(x1 + step_x, xi, step_x):
      world.create_cell(x, y1, RailroadCell)

    if step_y != 0:
      for y in range(y1, y2, step_y):
        world.create_cell(xi, y, RailroadCell)

    for x in range(xi, x2, step_x):
      world.create_cell(x, y2, RailroadCell)


if __name__ == '__main__':
  wse = WorldBuilder.create()
