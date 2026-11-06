#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for integer dims validation.'''

import pytest

from truck_carton.domain.models import (
  Carton,
  PlacementInfo,
  Store,
  Truck,
  Warehouse,
)
from truck_carton.packing.placement import PlacementValidator
from truck_carton.packing.rotation import (
  Rotation,
  apply_rotation,
)
from truck_carton.packing.space3d import Space3D


def _validator():
  '''Build a validator over a 4x4x4 space.'''
  return PlacementValidator(Space3D(4, 4, 4), {})


@pytest.mark.parametrize('value', [1.5, 2.0, True, '1'])
def test_carton_rejects_non_int_dims(value):
  '''Carton rejects non-int dims with ValueError.'''
  with pytest.raises(ValueError):
    Carton(1, value, 1, 1, 1.0, False, 1, 0)
  with pytest.raises(ValueError):
    Carton(1, 1, value, 1, 1.0, False, 1, 0)
  with pytest.raises(ValueError):
    Carton(1, 1, 1, value, 1.0, False, 1, 0)


@pytest.mark.parametrize('value', [1.5, 2.0, True, '1'])
def test_truck_rejects_non_int_dims(value):
  '''Truck rejects non-int dims with ValueError.'''
  with pytest.raises(ValueError):
    Truck(0, value, 1, 1, 500.0, [0])
  with pytest.raises(ValueError):
    Truck(0, 1, value, 1, 500.0, [0])
  with pytest.raises(ValueError):
    Truck(0, 1, 1, value, 500.0, [0])


@pytest.mark.parametrize('value', [0.5, 1.0, True, '0'])
def test_warehouse_store_reject_non_int_position(value):
  '''Warehouse/Store reject non-int positions.'''
  with pytest.raises(ValueError):
    Warehouse(0, (value, 0))
  with pytest.raises(ValueError):
    Warehouse(0, (0, value))
  with pytest.raises(ValueError):
    Store(0, 0, (value, 0))
  with pytest.raises(ValueError):
    Store(0, 0, (0, value))


@pytest.mark.parametrize('value', [1.5, 2.0, True, '1'])
def test_placement_info_rejects_non_int_dims(value):
  '''PlacementInfo rejects non-int oriented_dims.'''
  with pytest.raises(ValueError):
    PlacementInfo(0, (0, 0, 0), (value, 1, 1), Rotation.LWH)
  with pytest.raises(ValueError):
    PlacementInfo(0, (0, 0, 0), (1, value, 1), Rotation.LWH)
  with pytest.raises(ValueError):
    PlacementInfo(0, (0, 0, 0), (1, 1, value), Rotation.LWH)


@pytest.mark.parametrize('value', [0.5, 1.0, True, '0'])
def test_placement_info_rejects_non_int_position(value):
  '''PlacementInfo rejects non-int positions.'''
  with pytest.raises(ValueError):
    PlacementInfo(0, (value, 0, 0), (1, 1, 1), Rotation.LWH)


@pytest.mark.parametrize(
  'args',
  [(0.5, 0, 0, 1, 1, 1), (0, 0, 0, 1.5, 1, 1),
   (True, 0, 0, 1, 1, 1), ('0', 0, 0, 1, 1, 1)],
)
def test_check_bounds_rejects_non_int(args):
  '''check_bounds rejects non-int args with ValueError.'''
  with pytest.raises(ValueError):
    _validator().check_bounds(*args)


@pytest.mark.parametrize(
  'args',
  [(0, 0, 1.0, 1, 1, 1), (0.5, 0, 1, 1, 1, 1),
   (True, 0, 1, 1, 1, 1), ('0', 0, 1, 1, 1, 1),
   (0, 0, 1, 1.5, 1, 1)],
)
def test_check_gravity_rejects_non_int(args):
  '''check_gravity_support rejects non-int with ValueError.'''
  with pytest.raises(ValueError):
    _validator().check_gravity_support(*args)


@pytest.mark.parametrize(
  'args',
  [(0.5, 0, 0, 1, 1, 1), (0, 0, 0, 1.5, 1, 1),
   (True, 0, 0, 1, 1, 1)],
)
def test_check_no_overlap_rejects_non_int(args):
  '''check_no_overlap rejects non-int with ValueError.'''
  with pytest.raises(ValueError):
    _validator().check_no_overlap(*args)


def test_validate_placement_rejects_non_int():
  '''validate_placement rejects non-int coords/dims.'''
  carton = Carton(1, 1, 1, 1, 1.0, False, 1, 0)
  v = _validator()
  with pytest.raises(ValueError):
    v.validate_placement(carton, 0.5, 0, 0, 1, 1, 1, 0.0, 500.0)
  with pytest.raises(ValueError):
    v.validate_placement(carton, 0, 0, 0, 1.5, 1, 1, 0.0, 500.0)
  with pytest.raises(ValueError):
    v.validate_placement(carton, True, 0, 0, 1, 1, 1, 0.0, 500.0)


def test_find_valid_positions_rejects_non_int_carton_dims():
  '''find_valid_positions rejects non-int carton dims.'''
  carton = Carton(1, 1, 1, 1, 1.0, False, 1, 0)
  carton.length = 1.5
  with pytest.raises(ValueError):
    _validator().find_valid_positions(carton, 0.0, 500.0)


def test_validate_placement_rejects_non_int_carton_dims():
  '''validate_placement rejects non-int carton dims.'''
  carton = Carton(1, 1, 1, 1, 1.0, False, 1, 0)
  carton.height = 2.0
  with pytest.raises(ValueError):
    _validator().validate_placement(
      carton, 0, 0, 0, 1, 1, 1, 0.0, 500.0
    )


@pytest.mark.parametrize('rotation', [1.0, 0.0, True, '0'])
def test_apply_rotation_rejects_non_int(rotation):
  '''apply_rotation rejects non-int rotation with ValueError.'''
  with pytest.raises(ValueError):
    apply_rotation(1, 2, 3, rotation)


def test_valid_int_dims_still_work():
  '''Valid integer dims still work everywhere.'''
  carton = Carton(1, 1, 1, 1, 1.0, False, 1, 0)
  truck = Truck(0, 4, 4, 4, 500.0, [0])
  assert truck.length == 4
  assert carton.length == 1
  assert apply_rotation(1, 2, 3, Rotation.LWH) == (1, 2, 3)
  assert apply_rotation(1, 2, 3, 0) == (1, 2, 3)
  assert _validator().check_bounds(0, 0, 0, 1, 1, 1) is True
