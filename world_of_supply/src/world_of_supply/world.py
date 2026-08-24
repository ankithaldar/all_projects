#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''The world: a grid of cells, registered facilities, and the step engine.'''

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from world_of_supply.economy import BalanceSheet
from world_of_supply.facility import FacilityCell, FacilityControl
from world_of_supply.geography import RailroadCell, TerrainCell
from world_of_supply import routing


class WorldEconomy:
  '''Aggregated view over all facility balances.

  Attributes:
    world: Owning world.
  '''

  def __init__(self, world: 'World') -> None:
    '''Bind the economy to a world.

    Args:
      world: Owning world.
    '''
    self.world = world

  def global_balance(self) -> BalanceSheet:
    '''Sum the balances of all facilities.

    Returns:
      BalanceSheet: The system-wide net worth movement since creation.
    '''
    return sum((facility.economy.total_balance for facility in self.world.facilities.values()), BalanceSheet())


@dataclass
class Control:
  '''Controls for one simulation tick.

  Attributes:
    facility_controls: Mapping of facility id to its control; facilities
      without an entry act with defaults (no-ops).
  '''

  facility_controls: dict[str, FacilityControl]


@dataclass
class StepOutcome:
  '''Result of one :meth:`World.act` call.

  Attributes:
    facility_step_balance_sheets: Per-facility monetary effect of the tick.
  '''

  facility_step_balance_sheets: dict[str, BalanceSheet]


class World:
  '''A rectangular grid hosting facilities connected by railroads.

  Attributes:
    size_x: Grid width.
    size_y: Grid height.
    grid: ``grid[x][y]`` cell matrix.
    facilities: Registry of all facilities by id.
    economy: Aggregated balance view.
    time_step: Number of executed ticks.
  '''

  def __init__(self, x: int, y: int) -> None:
    '''Create an empty terrain-filled world.

    Args:
      x: Grid width.
      y: Grid height.
    '''
    self.size_x = x
    self.size_y = y
    self.grid: list[list[object]] = [
        [TerrainCell(xi, yi) for yi in range(y)] for xi in range(x)
    ]
    self.facilities: dict[str, FacilityCell] = {}
    self.economy = WorldEconomy(self)
    self.id_counter = 0
    self.time_step = 0
    self._graph: nx.Graph | None = None

  def generate_id(self) -> int:
    '''Produce the next unique numeric facility id.

    Returns:
      int: Monotonically increasing id.
    '''
    self.id_counter += 1
    return self.id_counter

  def register_facility(self, facility: FacilityCell) -> None:
    '''Add a facility to the registry.

    Args:
      facility: Facility to register.
    '''
    self.facilities[facility.id] = facility

  def place_cell(self, *cells) -> None:
    '''Write cells onto the grid and invalidate cached routing.

    Args:
      *cells: Cells with final ``x``/``y`` attributes.
    '''
    for cell in cells:
      self.grid[cell.x][cell.y] = cell
    self._graph = None

  def create_cell(self, x: int, y: int, clazz) -> None:
    '''Instantiate a cell class directly onto the grid.

    Args:
      x: Grid x position.
      y: Grid y position.
      clazz: Cell subclass constructor taking ``(x, y)``.
    '''
    self.place_cell(clazz(x, y))

  def is_railroad(self, x: int, y: int) -> bool:
    '''Check whether a coordinate holds a railroad cell.'''
    return isinstance(self.grid[x][y], RailroadCell)

  def is_traversable(self, x: int, y: int) -> bool:
    '''Check whether a coordinate can be entered by vehicles.'''
    return not isinstance(self.grid[x][y], TerrainCell)

  def find_path(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    '''Find a route between two coordinates (cached until the map changes).

    Args:
      start: ``(x, y)`` origin.
      goal: ``(x, y)`` destination.

    Returns:
      list[tuple[int, int]] | None: Path inclusive of both ends, or None.
    '''
    if self._graph is None:
      self._graph = routing.build_traversable_graph(self.size_x, self.size_y, self.is_traversable)
    return routing.shortest_path(self._graph, start, goal, self.size_y)

  def get_facilities(self, clazz: type) -> list[FacilityCell]:
    '''Filter facilities by class.

    Args:
      clazz: Facility subclass to select.

    Returns:
      list[FacilityCell]: Matching facilities in registration order.
    '''
    return [facility for facility in self.facilities.values() if isinstance(facility, clazz)]

  def act(self, control: Control) -> StepOutcome:
    '''Advance every facility by one tick.

    Args:
      control: Controls per facility id; missing ids mean no-op ticks.

    Returns:
      StepOutcome: Balance sheets booked by each facility this tick.
    '''
    sheets: dict[str, BalanceSheet] = {}
    for facility in self.facilities.values():
      sheets[facility.id] = facility.act(control.facility_controls.get(facility.id))
    self.time_step += 1
    return StepOutcome(sheets)
