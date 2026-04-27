from __future__ import annotations

from typing import Protocol

import networkx as nx
import numpy as np

from truck_carton.config import AppConfig, GridConfig
from truck_carton.domain.models import (
  Carton,
  CellType,
  EpisodeData,
  GridWorld,
  Store,
  Truck,
  TruckState,
  Warehouse,
)


class RoadNetworkStrategy(Protocol):
  """Strategy for building road networks.
  Implement to add new grid layout types."""

  def build(
    self,
    grid: np.ndarray,
    rows: int,
    cols: int,
    facilities: list[tuple[int, int]],
    rng: np.random.Generator,
  ) -> None: ...


class MSTRoadStrategy:
  """Builds roads via MST with L-shaped segments."""

  def __init__(self, config: GridConfig) -> None:
    self._config = config

  def build(
    self,
    grid: np.ndarray,
    rows: int,
    cols: int,
    facilities: list[tuple[int, int]],
    rng: np.random.Generator,
  ) -> None:
    if len(facilities) < 2:
      return

    g = nx.Graph()
    for i, pos_a in enumerate(facilities):
      for j, pos_b in enumerate(facilities):
        if i < j:
          dist = (
            abs(pos_a[0] - pos_b[0])
            + abs(pos_a[1] - pos_b[1])
          )
          g.add_edge(i, j, weight=dist)

    mst = nx.minimum_spanning_tree(g)

    non_mst = [
      (u, v) for u, v in g.edges()
      if not mst.has_edge(u, v)
    ]
    extras = min(
      self._config.road_extra_edges,
      len(non_mst),
    )
    if extras > 0:
      chosen = rng.choice(
        len(non_mst), size=extras, replace=False
      )
      for idx in chosen:
        u, v = non_mst[idx]
        mst.add_edge(u, v)

    for u, v in mst.edges():
      self._build_segment(
        grid, rows, cols,
        facilities[u], facilities[v], rng,
      )

  def _build_segment(
    self,
    grid: np.ndarray,
    rows: int,
    cols: int,
    src: tuple[int, int],
    dst: tuple[int, int],
    rng: np.random.Generator,
  ) -> None:
    r1, c1 = src
    r2, c2 = dst

    if abs(c2 - c1) > 1:
      bend_c = int(
        min(c1, c2)
        + abs(c2 - c1)
        * rng.uniform(0.2, 0.8)
      )
    else:
      bend_c = c1

    step_r = int(np.sign(r2 - r1)) or 1
    step_c = int(np.sign(c2 - c1)) or 1

    c = c1
    while c != bend_c:
      c += step_c
      if 0 <= c < cols and 0 <= r1 < rows:
        if grid[r1, c] == CellType.TERRAIN:
          grid[r1, c] = CellType.ROAD

    r = r1
    while r != r2:
      r += step_r
      if 0 <= r < rows and 0 <= bend_c < cols:
        if grid[r, bend_c] == CellType.TERRAIN:
          grid[r, bend_c] = CellType.ROAD

    c = bend_c
    while c != c2:
      c += step_c
      if 0 <= c < cols and 0 <= r2 < rows:
        if grid[r2, c] == CellType.TERRAIN:
          grid[r2, c] = CellType.ROAD


class DataGenerator:
  """Generates random episode data with procedural
  grid layouts for each curriculum stage."""

  def __init__(
    self,
    config: AppConfig,
    rng: np.random.Generator,
    road_strategy: RoadNetworkStrategy | None = None,
  ) -> None:
    self._config = config
    self._rng = rng
    self._road_strategy = (
      road_strategy
      or MSTRoadStrategy(config.grid)
    )

  def generate(
    self,
    num_trucks: int,
    num_stores: int,
    num_cartons: int,
    num_warehouses: int = 1,
    grid_rows: int = 5,
    grid_cols: int = 5,
  ) -> EpisodeData:
    grid_world, wh_positions, st_positions = (
      self._generate_grid_world(
        grid_rows, grid_cols,
        num_warehouses, num_stores,
      )
    )
    warehouses = self._generate_warehouses(
      wh_positions
    )
    stores = self._generate_stores(
      num_stores, st_positions
    )
    trucks = self._generate_trucks(
      num_trucks, grid_world.depot_position
    )
    cartons = self._generate_cartons(
      num_cartons, stores, warehouses
    )
    return EpisodeData(
      trucks=trucks,
      stores=stores,
      cartons=cartons,
      warehouses=warehouses,
      grid_world=grid_world,
    )

  def _generate_grid_world(
    self,
    rows: int,
    cols: int,
    num_warehouses: int,
    num_stores: int,
  ) -> tuple[
    GridWorld,
    list[tuple[int, int]],
    list[tuple[int, int]],
  ]:
    grid = np.full(
      (rows, cols), CellType.TERRAIN, dtype=np.int8
    )

    depot = (rows // 2, cols // 2)
    grid[depot[0], depot[1]] = CellType.DEPOT

    gc = self._config.grid
    # Interior cells available (exclude border)
    interior = max((rows - 2) * (cols - 2), 1)
    safe_wh = min(num_warehouses, interior // 3)
    safe_st = min(
      num_stores, interior // 3
    )

    min_spacing = max(
      gc.min_facility_spacing,
      int(
        min(rows, cols)
        * gc.spacing_scale_factor
      ),
    )
    placed: list[tuple[int, int]] = [depot]

    wh_positions = self._place_facilities(
      grid, rows, cols, safe_wh,
      CellType.WAREHOUSE, placed, min_spacing,
    )
    placed.extend(wh_positions)

    st_positions = self._place_facilities(
      grid, rows, cols, safe_st,
      CellType.STORE, placed, min_spacing,
    )
    placed.extend(st_positions)

    self._road_strategy.build(
      grid, rows, cols, placed, self._rng
    )

    facility_positions = list(placed)
    distance_matrix, path_cache = (
      self._compute_distances(
        grid, rows, cols, facility_positions
      )
    )

    grid_world = GridWorld(
      rows=rows,
      cols=cols,
      grid=grid,
      depot_position=depot,
      distance_matrix=distance_matrix,
      facility_positions=facility_positions,
      path_cache=path_cache,
    )
    return grid_world, wh_positions, st_positions

  def _place_facilities(
    self,
    grid: np.ndarray,
    rows: int,
    cols: int,
    count: int,
    cell_type: CellType,
    existing: list[tuple[int, int]],
    min_spacing: int,
  ) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    attempts = 0
    max_attempts = count * 100

    while len(positions) < count:
      attempts += 1
      if attempts > max_attempts:
        if min_spacing > 1:
          min_spacing -= 1
          attempts = 0
          continue
        break

      r = int(self._rng.integers(1, rows - 1))
      c = int(self._rng.integers(1, cols - 1))

      if grid[r, c] != CellType.TERRAIN:
        continue

      too_close = False
      for er, ec in existing + positions:
        if abs(r - er) + abs(c - ec) < min_spacing:
          too_close = True
          break
      if too_close:
        continue

      grid[r, c] = cell_type
      positions.append((r, c))

    return positions

  def _compute_distances(
    self,
    grid: np.ndarray,
    rows: int,
    cols: int,
    facilities: list[tuple[int, int]],
  ) -> tuple[np.ndarray, dict]:
    g = nx.Graph()

    for r in range(rows):
      for c in range(cols):
        if grid[r, c] == CellType.TERRAIN:
          continue
        node = (r, c)
        for dr, dc in [
          (-1, 0), (1, 0), (0, -1), (0, 1)
        ]:
          nr, nc = r + dr, c + dc
          if (
            0 <= nr < rows
            and 0 <= nc < cols
            and grid[nr, nc] != CellType.TERRAIN
          ):
            g.add_edge(node, (nr, nc), weight=1)

    n = len(facilities)
    dist_matrix = np.full(
      (n, n), float('inf'), dtype=np.float64
    )
    path_cache: dict[
      tuple[tuple[int, int], tuple[int, int]],
      list[tuple[int, int]],
    ] = {}

    for i in range(n):
      dist_matrix[i, i] = 0.0
      if facilities[i] not in g:
        continue
      try:
        lengths = nx.single_source_dijkstra_path_length(
          g, facilities[i]
        )
        paths = nx.single_source_dijkstra_path(
          g, facilities[i]
        )
      except nx.NetworkXError:
        continue

      for j in range(n):
        if facilities[j] in lengths:
          dist_matrix[i, j] = lengths[
            facilities[j]
          ]
          path_cache[
            (facilities[i], facilities[j])
          ] = paths[facilities[j]]

    return dist_matrix, path_cache

  def _generate_warehouses(
    self, positions: list[tuple[int, int]]
  ) -> list[Warehouse]:
    return [
      Warehouse(warehouse_id=i, position=pos)
      for i, pos in enumerate(positions)
    ]

  def _generate_stores(
    self,
    num_stores: int,
    positions: list[tuple[int, int]],
  ) -> list[Store]:
    return [
      Store(
        store_id=i,
        route_position=i,
        position=positions[i]
        if i < len(positions)
        else (0, 0),
      )
      for i in range(num_stores)
    ]

  def _generate_trucks(
    self,
    num_trucks: int,
    depot_position: tuple[int, int],
  ) -> list[Truck]:
    tc = self._config.truck
    trucks: list[Truck] = []

    for i in range(num_trucks):
      trucks.append(Truck(
        truck_id=i,
        length=int(self._rng.integers(
          tc.length_range[0],
          tc.length_range[1] + 1,
        )),
        width=int(self._rng.integers(
          tc.width_range[0],
          tc.width_range[1] + 1,
        )),
        height=int(self._rng.integers(
          tc.height_range[0],
          tc.height_range[1] + 1,
        )),
        max_weight=float(self._rng.uniform(
          tc.weight_capacity_range[0],
          tc.weight_capacity_range[1],
        )),
        route=[],
        position=depot_position,
        state=TruckState.ROUTING,
      ))
    return trucks

  def _generate_cartons(
    self,
    num_cartons: int,
    stores: list[Store],
    warehouses: list[Warehouse],
  ) -> list[Carton]:
    cc = self._config.carton
    store_ids = (
      [s.store_id for s in stores] or [0]
    )
    wh_ids = (
      [w.warehouse_id for w in warehouses] or [0]
    )
    cartons: list[Carton] = []

    for i in range(num_cartons):
      cartons.append(Carton(
        carton_id=i + 1,
        length=int(self._rng.integers(
          cc.length_range[0],
          cc.length_range[1] + 1,
        )),
        width=int(self._rng.integers(
          cc.width_range[0],
          cc.width_range[1] + 1,
        )),
        height=int(self._rng.integers(
          cc.height_range[0],
          cc.height_range[1] + 1,
        )),
        weight=float(self._rng.uniform(
          cc.weight_range[0], cc.weight_range[1]
        )),
        is_fragile=bool(
          self._rng.random() < cc.fragile_probability
        ),
        priority=int(self._rng.integers(1, 4)),
        destination_store_id=int(
          self._rng.choice(store_ids)
        ),
        origin_warehouse_id=int(
          self._rng.choice(wh_ids)
        ),
      ))
    return cartons
