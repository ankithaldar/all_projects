import numpy as np

from truck_carton.config import AppConfig
from truck_carton.domain.data_generator import (
  DataGenerator,
)


def test_generate_correct_counts():
  config = AppConfig()
  gen = DataGenerator(
    config, np.random.default_rng(42)
  )
  data = gen.generate(
    num_trucks=3, num_stores=2, num_cartons=15
  )
  assert len(data.trucks) == 3
  assert len(data.stores) == 2
  assert len(data.cartons) == 15


def test_store_ids_sequential():
  config = AppConfig()
  gen = DataGenerator(
    config, np.random.default_rng(42)
  )
  data = gen.generate(
    num_trucks=2, num_stores=4, num_cartons=10
  )
  for i, store in enumerate(data.stores):
    assert store.store_id == i
    assert store.route_position == i


def test_truck_dimensions_in_range():
  config = AppConfig()
  gen = DataGenerator(
    config, np.random.default_rng(42)
  )
  data = gen.generate(
    num_trucks=5, num_stores=3, num_cartons=20
  )
  tc = config.truck
  for truck in data.trucks:
    assert tc.length_range[0] <= truck.length
    assert truck.length <= tc.length_range[1]
    assert tc.width_range[0] <= truck.width
    assert truck.width <= tc.width_range[1]
    assert tc.height_range[0] <= truck.height
    assert truck.height <= tc.height_range[1]
    assert (
      tc.weight_capacity_range[0]
      <= truck.max_weight
      <= tc.weight_capacity_range[1]
    )


def test_truck_routes_are_valid_store_ids():
  config = AppConfig()
  gen = DataGenerator(
    config, np.random.default_rng(42)
  )
  data = gen.generate(
    num_trucks=3, num_stores=3, num_cartons=10
  )
  store_ids = {s.store_id for s in data.stores}
  for truck in data.trucks:
    assert len(truck.route) >= 1
    for sid in truck.route:
      assert sid in store_ids
    assert truck.route == sorted(truck.route)


def test_carton_dimensions_in_range():
  config = AppConfig()
  gen = DataGenerator(
    config, np.random.default_rng(42)
  )
  data = gen.generate(
    num_trucks=2, num_stores=2, num_cartons=30
  )
  cc = config.carton
  store_ids = {s.store_id for s in data.stores}
  for carton in data.cartons:
    assert cc.length_range[0] <= carton.length
    assert carton.length <= cc.length_range[1]
    assert cc.width_range[0] <= carton.width
    assert carton.width <= cc.width_range[1]
    assert cc.height_range[0] <= carton.height
    assert carton.height <= cc.height_range[1]
    assert cc.weight_range[0] <= carton.weight
    assert carton.weight <= cc.weight_range[1]
    assert carton.priority in (1, 2, 3)
    assert (
      carton.destination_store_id in store_ids
    )


def test_unique_carton_ids():
  config = AppConfig()
  gen = DataGenerator(
    config, np.random.default_rng(42)
  )
  data = gen.generate(
    num_trucks=2, num_stores=2, num_cartons=20
  )
  ids = [c.carton_id for c in data.cartons]
  assert len(ids) == len(set(ids))


def test_fragile_distribution():
  config = AppConfig()
  gen = DataGenerator(
    config, np.random.default_rng(42)
  )
  data = gen.generate(
    num_trucks=2, num_stores=2, num_cartons=100
  )
  fragile_count = sum(
    1 for c in data.cartons if c.is_fragile
  )
  ratio = fragile_count / len(data.cartons)
  assert 0.05 < ratio < 0.45


def test_grid_world_generation():
  """Procedural grid must have depot,
  warehouses, stores, and roads."""
  from truck_carton.domain.models import CellType
  config = AppConfig()
  gen = DataGenerator(
    config, np.random.default_rng(42)
  )
  data = gen.generate(
    num_trucks=2, num_stores=2,
    num_cartons=10, num_warehouses=1,
    grid_rows=5, grid_cols=5,
  )
  gw = data.grid_world
  assert gw is not None
  assert gw.rows == 5
  assert gw.cols == 5
  assert gw.grid[
    gw.depot_position[0],
    gw.depot_position[1],
  ] == CellType.DEPOT
  assert len(data.warehouses) == 1
  assert len(data.stores) == 2


def test_grid_facilities_reachable():
  """All facilities must be reachable from
  the depot via the road network."""
  config = AppConfig()
  gen = DataGenerator(
    config, np.random.default_rng(42)
  )
  data = gen.generate(
    num_trucks=3, num_stores=3,
    num_cartons=15, num_warehouses=2,
    grid_rows=7, grid_cols=7,
  )
  gw = data.grid_world
  n = len(gw.facility_positions)
  for i in range(n):
    for j in range(n):
      if i != j:
        assert gw.distance_matrix[i, j] < (
          float('inf')
        ), (
          f'Facility {i} cannot reach {j}'
        )


def test_cartons_have_warehouse_origin():
  """Each carton must have a valid
  origin_warehouse_id."""
  config = AppConfig()
  gen = DataGenerator(
    config, np.random.default_rng(42)
  )
  data = gen.generate(
    num_trucks=2, num_stores=2,
    num_cartons=20, num_warehouses=2,
    grid_rows=6, grid_cols=6,
  )
  wh_ids = {
    w.warehouse_id for w in data.warehouses
  }
  for carton in data.cartons:
    assert carton.origin_warehouse_id in wh_ids


def test_trucks_start_at_depot():
  """All trucks must start at the depot."""
  config = AppConfig()
  gen = DataGenerator(
    config, np.random.default_rng(42)
  )
  data = gen.generate(
    num_trucks=3, num_stores=2,
    num_cartons=10, num_warehouses=1,
    grid_rows=5, grid_cols=5,
  )
  depot = data.grid_world.depot_position
  for truck in data.trucks:
    assert truck.position == depot
