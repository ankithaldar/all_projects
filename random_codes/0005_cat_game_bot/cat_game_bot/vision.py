#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Bounded template library, OpenCV matcher and image color helpers."""

import io
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cat_game_bot.config import _resolve_path
from cat_game_bot.errors import ConfigError, VisionError
from cat_game_bot.models import Match, Rect, Screen

__all__ = (
  'TEMPLATE_SCORE_OVERRIDES', 'TemplateLibrary', 'TemplateMatcher',
  'non_max_suppress', 'dominant_color', 'color_distance',
)

# Per-template score overrides live in code so the manifest stays a pure
# name-to-filename map. craft_ready_check scores ~0.895 on a normal craft
# shelf and 1.000 on a completed tile, so it needs a stricter threshold to
# avoid false COLLECT taps. The real COINS! gift popup scores 0.506 on
# gift_collect, so its pinned 0.45 override lets the GIFT_DIALOG conjunction
# fire while staying below false-positive scores on other banners. The real
# CRAFT WOOD dialog scores 0.673 on recipe_marker and 0.760 on craft_button;
# normal craft shelves score <= 0.44 and <= 0.36 on them, so 0.65/0.70 admits
# the dialog without firing on a shelf.
TEMPLATE_SCORE_OVERRIDES: dict[str, float] = {
  'craft_ready_check': 0.97,
  'gift_collect': 0.45,
  'recipe_marker': 0.65,
  'craft_button': 0.70,
}


class TemplateLibrary:
  """Loads a confined manifest of bounded PNG/JPEG template images."""

  def __init__(self, directory: str, max_files: int = 200,
               max_manifest_bytes: int = 1048576,
               max_template_bytes: int = 2097152,
               max_template_pixels: int = 2592000) -> None:
    """Initialise an empty library rooted at a confined directory.

    Args:
      directory: Root directory for the manifest and template files.
      max_files: Maximum number of templates in a manifest.
      max_manifest_bytes: Byte cap for the manifest file.
      max_template_bytes: Byte cap for each template file.
      max_template_pixels: Pixel cap applied before template decoding.
    """
    self._directory = str(Path(directory).resolve())
    self._max_files = max_files
    self._max_manifest_bytes = max_manifest_bytes
    self._max_template_bytes = max_template_bytes
    self._max_template_pixels = max_template_pixels
    self._templates: dict[str, np.ndarray] = {}

  def load(self, manifest_path: str) -> None:
    """Load and validate a version-1 template manifest.

    Args:
      manifest_path: Path of the manifest JSON file.

    Raises:
      VisionError: On any confinement, format, version or cap violation.
    """
    try:
      path = _resolve_path(manifest_path, self._directory)
    except ConfigError as exc:
      raise VisionError('manifest path rejected') from exc
    raw = self._read_bounded(path, self._max_manifest_bytes, 'manifest')
    try:
      data = json.loads(raw)
    except ValueError as exc:
      raise VisionError('invalid manifest json') from exc
    version = data.get('version') if isinstance(data, dict) else None
    if (isinstance(version, bool) or not isinstance(version, int)
        or version != 1):
      raise VisionError('unsupported manifest version')
    entries = data.get('templates')
    if not isinstance(entries, dict):
      raise VisionError('manifest templates must be an object')

    loaded: dict[str, np.ndarray] = {}
    for name, filename in entries.items():
      if len(loaded) >= self._max_files:
        raise VisionError('too many templates')
      if not isinstance(name, str) or not isinstance(filename, str):
        raise VisionError('invalid template entry')
      try:
        image_path = _resolve_path(
          os.path.join(self._directory, filename), self._directory)
      except ConfigError as exc:
        raise VisionError('template path rejected') from exc
      image_bytes = self._read_bounded(
        image_path, self._max_template_bytes, 'template')
      loaded[name] = self._decode(image_bytes)
    self._templates = loaded

  def get(self, name: str) -> np.ndarray:
    """Return a loaded template by name.

    Args:
      name: Template name from the manifest.

    Returns:
      The decoded template pixel array.

    Raises:
      VisionError: If no template with that name was loaded.
    """
    if name not in self._templates:
      raise VisionError(f'unknown template: {name}')
    return self._templates[name]

  def names(self) -> list[str]:
    """Return the sorted names of all loaded templates.

    Returns:
      The sorted template names.
    """
    return sorted(self._templates)

  def _read_bounded(self, path: Path, cap: int, label: str) -> bytes:
    """Read a file only after confirming it fits within a byte cap.

    Args:
      path: File to read.
      cap: Maximum permitted size.
      label: Label used in error messages.

    Returns:
      The file bytes.

    Raises:
      VisionError: If the file is unreadable or exceeds the cap.
    """
    try:
      size = os.path.getsize(path)
    except OSError as exc:
      raise VisionError(f'{label} not readable') from exc
    if size > cap:
      raise VisionError(f'{label} exceeds byte cap')
    try:
      with open(path, 'rb') as handle:
        data = handle.read(cap + 1)
    except OSError as exc:
      raise VisionError(f'{label} not readable') from exc
    if len(data) > cap:
      raise VisionError(f'{label} exceeds byte cap')
    return data

  def _decode(self, image_bytes: bytes) -> np.ndarray:
    """Decode template bytes after enforcing the pixel cap.

    Args:
      image_bytes: Encoded image bytes.

    Returns:
      A writable BGR uint8 pixel array, matching cv2.imdecode screens.

    Raises:
      VisionError: If the image is invalid, a decompression bomb, or too
        large.
    """
    try:
      with Image.open(io.BytesIO(image_bytes)) as image:
        if image.width * image.height > self._max_template_pixels:
          raise VisionError('template exceeds pixel cap')
        rgb = np.asarray(image.convert('RGB'), dtype=np.uint8).copy()
        return rgb[:, :, ::-1]
    except (OSError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
      raise VisionError('template decode failed') from exc


def _iou(first: Rect, second: Rect) -> float:
  """Compute intersection-over-union for two rectangles.

  Args:
    first: The first rectangle.
    second: The second rectangle.

  Returns:
    The IoU ratio, zero for disjoint or empty rectangles.
  """
  left = max(first.x, second.x)
  top = max(first.y, second.y)
  right = min(first.x + first.width, second.x + second.width)
  bottom = min(first.y + first.height, second.y + second.height)
  overlap = max(0, right - left) * max(0, bottom - top)
  union = first.area() + second.area() - overlap
  return overlap / union if union else 0.0


_NMS_CELL = 64
# Dense tiny matches (a 1x1 template on a uniform screen) can fill a cell with
# thousands of non-overlapping boxes. Bounding the highest-scoring candidates
# before suppression keeps that case linear; 3000 preserves the existing
# large-input equivalence semantics.
_NMS_MAX_CANDIDATES = 3000


def non_max_suppress(matches: list[Match],
                     iou_threshold: float) -> list[Match]:
  """Keep the highest-scoring matches and suppress overlapping ones.

  Candidates are first bounded to the highest-scoring ``NMS_MAX_CANDIDATES``.
  A spatial grid then limits each candidate's IoU checks to kept matches that
  share one of its cells. Overlapping rectangles always share a cell, so the
  survivors are identical to a pairwise scan within the bounded set.

  Args:
    matches: Candidate matches.
    iou_threshold: Suppression threshold on intersection-over-union.

  Returns:
    The surviving matches ordered by descending score.
  """
  ordered = sorted(matches, key=lambda item: item.score, reverse=True)
  ordered = ordered[:_NMS_MAX_CANDIDATES]
  kept: list[Match] = []
  buckets: dict[tuple[int, int], list[Match]] = {}
  for candidate in ordered:
    rect = candidate.rect
    x0 = rect.x // _NMS_CELL
    x1 = (rect.x + max(rect.width, 1) - 1) // _NMS_CELL
    y0 = rect.y // _NMS_CELL
    y1 = (rect.y + max(rect.height, 1) - 1) // _NMS_CELL
    cells = [(gx, gy) for gx in range(x0, x1 + 1)
             for gy in range(y0, y1 + 1)]
    neighbours = {
      other for cell in cells for other in buckets.get(cell, ())}
    if all(_iou(candidate.rect, other.rect) < iou_threshold
           for other in neighbours):
      kept.append(candidate)
      for cell in cells:
        buckets.setdefault(cell, []).append(candidate)
  return kept


class TemplateMatcher:
  """Matches templates with normalized correlation and IoU suppression."""

  def __init__(self, library: TemplateLibrary, threshold: float = 0.8,
               iou_threshold: float = 0.5) -> None:
    """Initialise a matcher over a loaded template library.

    Args:
      library: The loaded template library.
      threshold: Minimum normalized correlation score.
      iou_threshold: IoU threshold used for suppression.
    """
    self._library = library
    self._threshold = threshold
    self._iou_threshold = iou_threshold

  def match(self, screen: Screen, name: str) -> list[Match]:
    """Find all suppressed occurrences of one template in a screen.

    Args:
      screen: The screen frame to search.
      name: Template name from the library. A template listed in
        TEMPLATE_SCORE_OVERRIDES uses its own score instead of the global
        threshold.

    Returns:
      The surviving matches ordered by descending score.
    """
    template = self._library.get(name)
    height, width = template.shape[:2]
    if height > screen.height or width > screen.width:
      return []
    result = cv2.matchTemplate(
      screen.pixels, template, cv2.TM_CCOEFF_NORMED)
    threshold = TEMPLATE_SCORE_OVERRIDES.get(name, self._threshold)
    ys, xs = np.where(result >= threshold)
    candidates = [
      Match(name, Rect(int(x), int(y), width, height), float(result[y, x]))
      for y, x in zip(ys, xs)
    ]
    return non_max_suppress(candidates, self._iou_threshold)

  def match_all(self, screen: Screen) -> list[Match]:
    """Find and suppress occurrences of every loaded template.

    Args:
      screen: The screen frame to search.

    Returns:
      The surviving matches ordered by descending score.
    """
    matches: list[Match] = []
    for name in self._library.names():
      matches.extend(self.match(screen, name))
    matches.sort(key=lambda item: item.score, reverse=True)
    return non_max_suppress(matches, self._iou_threshold)


def dominant_color(screen: Screen, rect: Rect) -> tuple[int, int, int]:
  """Return the most frequent BGR color inside a rectangle.

  Args:
    screen: The screen frame to sample.
    rect: The rectangle to sample.

  Returns:
    The most frequent BGR triple, or black for an empty region.
  """
  region = screen.pixels[rect.y:rect.y + rect.height,
                         rect.x:rect.x + rect.width]
  if region.size == 0:
    return (0, 0, 0)
  colors, counts = np.unique(
    region.reshape(-1, 3), axis=0, return_counts=True)
  index = int(np.argmax(counts))
  return tuple(int(value) for value in colors[index])


def color_distance(a: tuple[int, int, int],
                   b: tuple[int, int, int]) -> float:
  """Return the Euclidean distance between two colors.

  Args:
    a: The first color.
    b: The second color.

  Returns:
    The Euclidean distance as a float.
  """
  return math.dist(a, b)
