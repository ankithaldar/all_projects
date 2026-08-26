#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Intensity normalization for knee MRI slices.

MRI carries no absolute unit analogous to CT Hounsfield values, so windowing
is replaced by per-series percentile normalization followed by uint8 casting.
"""

from __future__ import annotations

import numpy as np

UINT8_MAX = 255.0


def rescale_pixels(
  pixels: np.ndarray, slope: float, intercept: float
) -> np.ndarray:
  """Apply DICOM modality LUT (RescaleSlope / RescaleIntercept).

  Args:
      pixels: Raw decoded pixel array of any numeric dtype.
      slope: RescaleSlope value; defaults handled upstream as 1.0.
      intercept: RescaleIntercept value; defaults handled upstream as 0.0.

  Returns:
      Float32 array after linear rescaling.
  """
  return pixels.astype(np.float32) * np.float32(slope) + np.float32(intercept)


def normalize_percentile(
  image: np.ndarray,
  percentiles: tuple[float, float],
  eps: float = 1e-6,
) -> np.ndarray:
  """Normalize an image to [0, 1] using robust percentile clipping.

  Args:
      image: Float array after rescaling.
      percentiles: ``(low, high)`` percentile bounds from configuration.
      eps: Numerical floor guarding zero-range images.

  Returns:
      Float32 image clipped and scaled to [0, 1].
  """
  low, high = np.percentile(image, percentiles)
  scaled = (image.astype(np.float32) - low) / max(float(high - low), eps)
  return np.clip(scaled, 0.0, 1.0)


def to_uint8(image01: np.ndarray) -> np.ndarray:
  """Quantize a [0, 1] image to uint8 for cheap caching and albumentations.

  Args:
      image01: Image already normalized to [0, 1].

  Returns:
      uint8 array in [0, 255].
  """
  return (image01 * UINT8_MAX).astype(np.uint8)


def autocrop(
  image: np.ndarray, margin: float, background_threshold: float = 0.02
):
  """Crop away empty background around the knee while keeping a safety margin.

  The foreground mask uses a low relative threshold because MRI backgrounds
  are near zero after normalization (EDA cell 45 showed consistent knee CoM,
  so cropping only needs to remove air borders).

  Args:
      image: Normalized [0, 1] image.
      margin: Fraction of the bounding-box extent padded on every side.
      background_threshold: Relative intensity above which pixels count
          as foreground.

  Returns:
      Tuple ``(cropped_image, bbox)`` where bbox is ``(y0, y1, x0, x1)``;
      the original image and full bbox are returned when no foreground exists.
  """
  mask = image > background_threshold * (float(image.max()) + 1e-6)
  if not mask.any():
    return image, (0, image.shape[0], 0, image.shape[1])
  rows = np.any(mask, axis=1)
  cols = np.any(mask, axis=0)
  y0, y1 = np.where(rows)[0][[0, -1]]
  x0, x1 = np.where(cols)[0][[0, -1]]
  pad_y = int((y1 - y0) * margin)
  pad_x = int((x1 - x0) * margin)
  y0 = max(0, y0 - pad_y)
  y1 = min(image.shape[0], y1 + pad_y + 1)
  x0 = max(0, x0 - pad_x)
  x1 = min(image.shape[1], x1 + pad_x + 1)
  return image[y0:y1, x0:x1], (y0, y1, x0, x1)
