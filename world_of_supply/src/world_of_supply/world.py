#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The world: grid, facility registry, router, and step engine.'''

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import networkx as nx

from world_of_supply.economy import BalanceSheet
from world_of_supply.facility import FacilityCell, FacilityControl
from world_of_supply.geography import RailroadCell, TerrainCell
from world_of_supply import routing


class FacilityRegistry:
  '''Registry of all facilities by id plus id generation.

  Extracted from :class:`World` (Single Responsibility):
  the registry only owns identity and lookup, not grid or routing.

  Attributes:
    facilities: Facilities keyed by string id.
    id_counter: Last issued numeric id.
  '''

  def __init__(self) -> None:
    '''Create an empty registry.

    Returns:
      None: The registry is mutated in place.
    '''
    self.facilities: dict[str, FacilityCell] = {}
    self.id_counter = 0

  def generate_id(self) -> int:
    '''Produce the next unique numeric facility id.

    Returns:
      int: Monotonically increasing id.
    '''
    self.id_counter += 1
    return self.id_counter

  def register(self, facility: FacilityCell) -> None:
    '''Add a facility to the registry.

    Args:
      facility: Facility to register.

    Returns:
      None: The registry is mutated in place.
    '''
    self.facilities[facility.id] = facility

  def get_by_type(self, clazz: type) -> list[FacilityCell]:
    '''Filter facilities by class.

    Args:
      clazz: Facility subclass to select.

    Returns:
      list[FacilityCell]: Matching facilities in registration order.
    '''
    return [
        facility
        for facility in self.facilities.values()
        if isinstance(facility, clazz)
    ]


class WorldRouter:
  '''Cached shortest-path routing over the traversable grid.

  Extracted from :class:`World` (Single Responsibility):
  the router only owns graph caching and path lookup.

  Attributes:
    size_x: Grid width.
    size_y: Grid height.
    is_traversable: Predicate deciding cell passability.
  '''

  def __init__(
      self,
      size_x: int,
      size_y: int,
      is_traversable: Callable[[int, int], bool],
  ) -> None:
    '''Bind the router to a grid size and passability predicate.

    Args:
      size_x: Grid width.
      size_y: Grid height.
      is_traversable: Callable ``(x, y) -> bool``.

    Returns:
      None: The router is mutated in place.
    '''
    self.size_x = size_x
    self.size_y = size_y
    self.is_traversable = is_traversable
    self._graph: nx.Graph | None = None

  def invalidate(self) -> None:
    '''Drop the cached traversable graph.

    Returns:
      None: The router is mutated in place.
    '''
    self._graph = None

  def find_path(
      self,
      start: tuple[int, int],
      goal: tuple[int, int],
  ) -> list[tuple[int, int]] | None:
    '''Find a route between two coordinates (cached).

    Args:
      start: ``(x, y)`` origin.
      goal: ``(x, y)`` destination.

    Returns:
      list[tuple[int, int]] | None: Path inclusive of both ends,
        or ``None`` when unreachable.
    '''
    if self._graph is None:
      self._graph = routing.build_traversable_graph(
          self.size_x, self.size_y, self.is_traversable,
      )
    return routing.shortest_path(
        self._graph, start, goal, self.size_y,
    )


class WorldEconomy:
  '''Aggregated view over all facility balances.

  Attributes:
    world: Owning world.
  '''

  def __init__(self, world: 'World') -> None:
    '''Bind the economy to a world.

    Args:
      world: Owning world.

    Returns:
      None: The economy is mutated in place.
    '''
    self.world = world

  def global_balance(self) -> BalanceSheet:
    '''Sum the balances of all facilities.

    Returns:
      BalanceSheet: System-wide net worth movement since creation.
    '''
    sheets = (
        facility.economy.total_balance
        for facility in self.world.facilities.values()
    )
    return sum(sheets, BalanceSheet())


@dataclass
class Control:
  '''Controls for one simulation tick.

  Attributes:
    facility_controls: Mapping of facility id to its control;
      facilities without an entry act with defaults (no-ops).
  '''

  facility_controls: dict[str, FacilityControl]


@dataclass
class StepOutcome:
  '''Result of one :meth:`World.act` call.

  Attributes:
    facility_step_balance_sheets: Per-facility monetary effect.
  '''

  facility_step_balance_sheets: dict[str, BalanceSheet]


class World:
  '''A rectangular grid hosting facilities connected by railroads.

  The grid step engine delegates identity to
  :class:`FacilityRegistry` and routing to :class:`WorldRouter`.

  Attributes:
    size_x: Grid width.
    size_y: Grid height.
    grid: ``grid[x][y]`` cell matrix.
    economy: Aggregated balance view.
    time_step: Number of executed ticks.
  '''

  def __init__(self, x: int, y: int) -> None:
    '''Create an empty terrain-filled world.

    Args:
      x: Grid width.
      y: Grid height.

    Returns:
      None: The world is mutated in place.
    '''
    self.size_x = x
    self.size_y = y
    self.grid: list[list[object]] = [
        [TerrainCell(xi, yi) for yi in range(y)] for xi in range(x)
    ]
    self._registry = FacilityRegistry()
    self._router = WorldRouter(x, y, self.is_traversable)
    self.economy = WorldEconomy(self)
    self.time_step = 0

  @property
  def facilities(self) -> dict[str, FacilityCell]:
    '''Expose the registry dict as a public attribute.

    Returns:
      dict[str, FacilityCell]: Live registry mapping.
    '''
    return self._registry.facilities

  @facilities.setter
  def facilities(self, value: dict[str, FacilityCell]) -> None:
    '''Replace the registry backing dict.

    Args:
      value: New registry mapping.

    Returns:
      None: The world is mutated in place.
    '''
    self._registry.facilities = value

  def generate_id(self) -> int:
    '''Produce the next unique numeric facility id.

    Returns:
      int: Monotonically increasing id.
    '''
    return self._registry.generate_id()

  def register_facility(self, facility: FacilityCell) -> None:
    '''Add a facility to the registry.

    Args:
      facility: Facility to register.

    Returns:
      None: The registry is mutated in place.
    '''
    self._registry.register(facility)

  def place_cell(self, *cells) -> None:
    '''Write cells onto the grid and invalidate cached routing.

    Args:
      *cells: Cells with final ``x``/``y`` attributes.

    Returns:
      None: The grid is mutated in place.
    '''
    for cell in cells:
      self.grid[cell.x][cell.y] = cell
    self._router.invalidate()

  def create_cell(self, x: int, y: int, clazz) -> None:
    '''Instantiate a cell class directly onto the grid.

    Args:
      x: Grid x position.
      y: Grid y position.
      clazz: Cell subclass constructor taking ``(x, y)``.

    Returns:
      None: The grid is mutated in place.
    '''
    self.place_cell(clazz(x, y))

  def is_railroad(self, x: int, y: int) -> bool:
    '''Check whether a coordinate holds a railroad cell.

    Args:
      x: Grid x position.
      y: Grid y position.

    Returns:
      bool: True when the cell is a railroad.
    '''
    return isinstance(self.grid[x][y], RailroadCell)

  def is_traversable(self, x: int, y: int) -> bool:
    '''Check whether a coordinate can be entered by vehicles.

    Args:
      x: Grid x position.
      y: Grid y position.

    Returns:
      bool: True when the cell is not terrain.
    '''
    return not isinstance(self.grid[x][y], TerrainCell)

  def find_path(
      self,
      start: tuple[int, int],
      goal: tuple[int, int],
  ) -> list[tuple[int, int]] | None:
    '''Find a route between two coordinates (cached).

    Args:
      start: ``(x, y)`` origin.
      goal: ``(x, y)`` destination.

    Returns:
      list[tuple[int, int]] | None: Path inclusive of both ends.
    '''
    return self._router.find_path(start, goal)

  def get_facilities(self, clazz: type) -> list[FacilityCell]:
    '''Filter facilities by class.

    Args:
      clazz: Facility subclass to select.

    Returns:
      list[FacilityCell]: Matching facilities in order.
    '''
    return self._registry.get_by_type(clazz)

  def act(self, control: Control) -> StepOutcome:
    '''Advance every facility by one tick.

    Args:
      control: Controls per facility id; missing ids mean no-op.

    Returns:
      StepOutcome: Balance sheets booked by each facility.
    '''
    sheets: dict[str, BalanceSheet] = {}
    for facility in self.facilities.values():
      sheets[facility.id] = facility.act(
          control.facility_controls.get(facility.id),
      )
    self.time_step += 1
    return StepOutcome(sheets)
