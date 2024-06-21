#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
World
'''

# imports
from dataclasses import dataclass
from functools import lru_cache

import networkx as nx
import numpy as np

# from balance_sheet import BalanceSheet
from .railroad_cell import RailroadCell
from .terrain_cell import TerrainCell

# imports



class World:
  '''
  World
  '''

  @dataclass
  class Economy:
    def __init__(self, world):
      self.world = world

    def global_balance(self):
      return sum(
        [f.economy.total_balance for f in self.world.facilities.values()]
      )


  @dataclass
  class Control:
    facility_controls: dict

  @dataclass
  class StepOutcome:
    facility_step_balance_sheets: dict

  def __init__(self, x, y):
    self.size_x = x
    self.size_y = y
    self.grid = None
    self.economy = World.Economy(self)
    self.facilities = dict()
    self.id_counter = 0
    self.time_step = 0

  def generate_id(self):
    self.id_counter += 1
    return self.id_counter

  def act(self, control):
    balance_sheets = dict()
    for facility in self.facilities.values():
      balance_sheets[facility.id] = facility.act(
        control.facility_controls.get(facility.id)
      )

    self.time_step += 1
    return World.StepOutcome(balance_sheets)

  def create_cell(self, x, y, clazz):
    self.grid[x][y] = clazz(x, y)

  def place_cell(self, *cells):
    for c in cells:
      self.grid[c.x][c.y] = c

  def is_railroad(self, x, y):
    return isinstance(self.grid[x][y], RailroadCell)

  def is_traversable(self, x, y):
    return not isinstance(self.grid[x][y], TerrainCell)

  @staticmethod
  def c_tobytes(x, y):
    return np.array([x, y]).tobytes()

  def map_to_graph(self):
    g = nx.Graph()
    for x in range(1, self.size_x-1):
      for y in range(1, self.size_y-1):
        for c in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
          if self.is_traversable(x, y) and self.is_traversable(c[0], c[1]):
            g.add_edge(World.c_tobytes(x, y), World.c_tobytes(c[0], c[1]))
    return g

  @lru_cache(maxsize = 32)  # speedup the simulation
  def find_path(self, x1, y1, x2, y2):
    g = self.map_to_graph()
    path = nx.astar_path(
      g,
      source=World.c_tobytes(x1, y1),
      target=World.c_tobytes(x2, y2)
    )
    path_np = [np.frombuffer(p, dtype=int) for p in path]
    return [(p[0], p[1]) for p in path_np]

  def get_facilities(self, clazz):
    return filter(lambda f: isinstance(f, clazz), self.facilities.values())
