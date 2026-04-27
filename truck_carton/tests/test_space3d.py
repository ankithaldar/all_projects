import numpy as np

from truck_carton.packing.space3d import Space3D


def test_empty_space():
  s = Space3D(4, 3, 2)
  assert s.shape == (4, 3, 2)
  assert s.get_occupancy_ratio() == 0.0
  assert np.all(s.grid == 0)


def test_place_and_remove():
  s = Space3D(4, 4, 4)
  s.place(1, 0, 0, 0, 2, 2, 2)
  assert s.grid[0, 0, 0] == 1
  assert s.grid[1, 1, 1] == 1
  assert s.get_occupied_volume() == 8
  s.remove(1)
  assert s.get_occupied_volume() == 0


def test_can_place_valid():
  s = Space3D(4, 4, 4)
  assert s.can_place(0, 0, 0, 2, 2, 2) is True


def test_can_place_out_of_bounds():
  s = Space3D(4, 4, 4)
  assert s.can_place(3, 3, 3, 2, 2, 2) is False


def test_can_place_overlap():
  s = Space3D(4, 4, 4)
  s.place(1, 0, 0, 0, 2, 2, 2)
  assert s.can_place(1, 1, 0, 2, 2, 2) is False


def test_occupancy_ratio():
  s = Space3D(4, 4, 4)
  s.place(1, 0, 0, 0, 4, 4, 4)
  assert s.get_occupancy_ratio() == 1.0

  s2 = Space3D(4, 4, 4)
  s2.place(1, 0, 0, 0, 2, 2, 2)
  expected = 8 / 64
  assert abs(
    s2.get_occupancy_ratio() - expected
  ) < 1e-9


def test_height_map_empty():
  s = Space3D(4, 4, 4)
  hm = s.get_height_map()
  assert hm.shape == (4, 4)
  assert np.all(hm == 0)


def test_height_map_with_placement():
  s = Space3D(4, 4, 4)
  s.place(1, 0, 0, 0, 2, 2, 2)
  hm = s.get_height_map()
  assert hm[0, 0] == 2
  assert hm[1, 1] == 2
  assert hm[2, 0] == 0
  assert hm[0, 2] == 0


def test_height_map_stacked():
  s = Space3D(4, 4, 4)
  s.place(1, 0, 0, 0, 2, 2, 1)
  s.place(2, 0, 0, 1, 2, 2, 1)
  hm = s.get_height_map()
  assert hm[0, 0] == 2
  assert hm[1, 1] == 2


def test_normalized_grid():
  s = Space3D(4, 4, 4)
  s.place(5, 0, 0, 0, 1, 1, 1)
  ng = s.get_normalized_grid(10)
  assert ng[0, 0, 0] == 0.5
  assert ng[1, 0, 0] == 0.0


def test_copy():
  s = Space3D(4, 4, 4)
  s.place(1, 0, 0, 0, 2, 2, 2)
  s2 = s.copy()
  assert np.array_equal(s.grid, s2.grid)
  s2.remove(1)
  assert s.get_occupied_volume() == 8
  assert s2.get_occupied_volume() == 0


def test_place_rejects_zero_carton_id():
  """Space3D.place() must reject carton_id=0 since
  0 is the sentinel for empty cells."""
  import pytest
  s = Space3D(4, 4, 4)
  with pytest.raises(ValueError):
    s.place(0, 0, 0, 0, 1, 1, 1)


def test_place_rejects_negative_carton_id():
  """Space3D.place() must reject negative IDs."""
  import pytest
  s = Space3D(4, 4, 4)
  with pytest.raises(ValueError):
    s.place(-1, 0, 0, 0, 1, 1, 1)


def test_lowest_valid_id_visible():
  """Carton with ID 1 (lowest valid) must be
  visible in height_map and occupancy."""
  s = Space3D(4, 4, 4)
  s.place(1, 0, 0, 0, 2, 2, 1)
  hm = s.get_height_map()
  assert hm[0, 0] == 1
  assert s.get_occupancy_ratio() > 0.0
  assert not s.can_place(0, 0, 0, 2, 2, 1)
