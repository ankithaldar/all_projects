#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for slice ordering geometry, including EDA-found edge cases."""

import numpy as np

from knee.helpers.geometry import SliceGeometry, order_slices, slice_normal


def make_slice(
  sop: str,
  instance: int,
  offset: float,
  orientation=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
) -> SliceGeometry:
  """Build a geometry record positioned along the orientation's normal.

  For the default IOP the normal is the x-axis, so ``offset`` maps to x.
  """
  normal = slice_normal(orientation)
  position = tuple(float(offset) * float(v) for v in normal)
  return SliceGeometry(
    sop_uid=sop,
    instance_number=instance,
    position=position,
    orientation=orientation,
  )


class TestSliceNormal:
  """Normal computation from IOP vectors."""

  def test_axis_aligned(self):
    normal = slice_normal((1.0, 0.0, 0.0, 0.0, 0.0, -1.0))
    assert np.allclose(np.abs(normal), np.array([0.0, 1.0, 0.0]))

  def test_oblique_unit_length(self):
    normal = slice_normal(
      (
        0.99495750665664,
        -0.0122263422235,
        0.09954915195703,
        0.10025131702423,
        0.15123197436332,
        -0.9834015369415,
      )
    )
    assert abs(np.linalg.norm(normal) - 1.0) < 1e-6


class TestOrdering:
  """Ordering strategies per BLUEPRINT Section 4."""

  def test_orders_by_projection_ascending(self):
    slices = [
      make_slice('b', 2, 5.0),
      make_slice('a', 1, 1.0),
      make_slice('c', 3, 9.0),
    ]
    assert order_slices(slices) == ['a', 'b', 'c']

  def test_instance_correlation_flips_direction(self):
    # InstanceNumber decreases as projection increases -> after the
    # direction flip, ordering must equal ascending InstanceNumber.
    slices = [
      make_slice('a', 3, 1.0),
      make_slice('b', 2, 5.0),
      make_slice('c', 1, 9.0),
    ]
    assert order_slices(slices) == ['c', 'b', 'a']

  def test_fallback_to_instance_number_when_geometry_missing(self):
    slices = [
      SliceGeometry('x', 10, None, None),
      SliceGeometry('y', 2, None, None),
      SliceGeometry('z', 6, (1.0, 1.0, 1.0), None),  # partial geometry
    ]
    assert order_slices(slices) == ['y', 'z', 'x']

  def test_input_order_last_resort_on_degenerate_instances(self):
    slices = [
      SliceGeometry('m', 0, None, None),
      SliceGeometry('n', 0, None, None),
    ]
    assert order_slices(slices) == ['m', 'n']

  def test_empty_list(self):
    assert order_slices([]) == []
