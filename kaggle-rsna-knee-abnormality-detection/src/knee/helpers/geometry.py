#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DICOM slice geometry: ordering and spacing extraction.

EDA findings (notebooks/01_EDA.ipynb) drive this module:

* Oblique acquisitions exist (IOP vectors tilted off-axis), so slices are
  ordered by projecting ``ImagePositionPatient`` onto the slice normal,
  never by filename.
* A minority of series miss IPP tags entirely; those fall back to
  ``InstanceNumber`` ordering, then to filename order as a last resort.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SliceGeometry:
  """Per-slice spatial record extracted during the header scan.

  Attributes:
      sop_uid: SOP Instance UID of the slice.
      instance_number: DICOM InstanceNumber (fallback ordering key).
      position: ImagePositionPatient triple or None when absent.
      orientation: ImageOrientationPatient six-tuple or None when absent.
  """

  sop_uid: str
  instance_number: int
  position: tuple[float, float, float] | None = None
  orientation: tuple[float, ...] | None = None

  @classmethod
  def from_dataset(cls, dataset) -> 'SliceGeometry':
    """Build a geometry record from a pydicom Dataset.

    Args:
        dataset: ``pydicom.dataset.Dataset`` read with stop_before_pixels.

    Returns:
        SliceGeometry instance tolerating missing optional tags.
    """
    position = getattr(dataset, 'ImagePositionPatient', None)
    orientation = getattr(dataset, 'ImageOrientationPatient', None)
    return cls(
      sop_uid=str(getattr(dataset, 'SOPInstanceUID', '')),
      instance_number=int(getattr(dataset, 'InstanceNumber', 0) or 0),
      position=tuple(float(x) for x in position)
      if position is not None
      else None,
      orientation=(
        tuple(float(x) for x in orientation)
        if orientation is not None and len(orientation) == 6
        else None
      ),
    )


def slice_normal(orientation: tuple[float, ...]) -> np.ndarray:
  """Compute the slice normal from ImageOrientationPatient.

  Args:
      orientation: Six-vector (row cosines, column cosines).

  Returns:
      Unit normal vector orthogonal to the acquisition plane.
  """
  row = np.asarray(orientation[0:3], dtype=np.float64)
  col = np.asarray(orientation[3:6], dtype=np.float64)
  normal = np.cross(row, col)
  norm = np.linalg.norm(normal)
  if norm == 0.0:
    return np.array([0.0, 0.0, 1.0])
  return normal / norm


def order_slices(slices: list[SliceGeometry]) -> list[str]:
  """Return SOP UIDs ordered along the through-plane axis.

  Strategy hierarchy per series:

  1. Project positions onto the series normal (majority orientation) when
     every slice carries full geometry; direction is disambiguated by
     correlating with InstanceNumber so head/foot order is stable.
  2. Fall back to InstanceNumber ascending when any geometry is missing.
  3. Fall back to input order when InstanceNumbers are degenerate (all zero).

  Args:
      slices: Geometry records for every slice of one series.

  Returns:
      Ordered list of SOP UIDs matching the physical slice sequence.
  """
  if not slices:
    return []
  if all(s.position is not None and s.orientation is not None for s in slices):
    orientations = np.stack([s.orientation for s in slices])
    median_orientation = np.median(orientations, axis=0)
    normal = slice_normal(tuple(median_orientation))
    projections = np.array(
      [np.dot(np.asarray(s.position), normal) for s in slices]
    )
    instances = np.array([s.instance_number for s in slices], dtype=np.float64)
    if np.std(projections) > 0:
      if np.std(instances) > 0:
        correlation = np.corrcoef(projections, instances)[0, 1]
        if not np.isnan(correlation) and correlation < 0:
          projections = -projections
      indices = np.argsort(projections, kind='stable')
      return [slices[i].sop_uid for i in indices]
    # Degenerate projection (constant): fall through to InstanceNumber.
  if len({s.instance_number for s in slices}) > 1:
    return [s.sop_uid for s in sorted(slices, key=lambda s: s.instance_number)]
  return [s.sop_uid for s in slices]


def extract_spacing(
  first_slice: SliceGeometry, pixel_spacing: float | None
) -> dict:
  """Summarize spacing statistics used later as model metadata.

  Args:
      first_slice: Any representative slice of the series.
      pixel_spacing: Mean in-plane spacing in mm, or None when unknown.

  Returns:
      Mapping with ``has_geometry`` flag and the in-plane spacing value.
  """
  return {
    'has_geometry': first_slice.position is not None,
    'pixel_spacing': float(pixel_spacing)
    if pixel_spacing is not None
    else -1.0,
  }
