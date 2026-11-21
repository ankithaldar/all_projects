#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for cat_game_bot.vision template loading, matching and colors."""

import json
import shutil
import struct
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from cat_game_bot.errors import VisionError
from cat_game_bot.models import Match, Rect, Screen
from cat_game_bot.vision import (
  TEMPLATE_SCORE_OVERRIDES,
  TemplateLibrary,
  TemplateMatcher,
  color_distance,
  dominant_color,
  non_max_suppress,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def write_png(path, array):
  """Write an ndarray as a PNG file.

  Args:
    path: Destination path.
    array: Image pixel array.

  Returns:
    The destination path.
  """
  assert cv2.imwrite(str(path), array)
  return path


def make_template():
  """Build a non-uniform 10x10 template suitable for correlation.

  Returns:
    A BGR ndarray with a dark square on a light field.
  """
  image = np.full((10, 10, 3), 200, dtype=np.uint8)
  image[2:8, 2:8] = 40
  return image


def make_library(tmp_path, *, version=1, templates=None):
  """Create a template directory and manifest on disk.

  Args:
    tmp_path: Temporary directory used as the library root.
    version: Manifest version to write.
    templates: Template name-to-file mapping.

  Returns:
    The path of the written manifest.
  """
  templates = templates if templates is not None else {'mark': 'mark.png'}
  write_png(tmp_path / 'mark.png', make_template())
  manifest = tmp_path / 'manifest.json'
  manifest.write_text(json.dumps({'version': version,
                                  'templates': templates}),
                      encoding='utf-8')
  return manifest


def test_library_loads_and_exposes_templates(tmp_path):
  """A version-1 manifest loads and its templates are retrievable."""
  manifest = make_library(tmp_path)
  library = TemplateLibrary(str(tmp_path))
  library.load(str(manifest))
  assert library.names() == ['mark']
  assert library.get('mark').shape == (10, 10, 3)
  with pytest.raises(VisionError):
    library.get('missing')


def write_coloured_png(path):
  """Write a non-uniform PNG whose B, G and R channels all differ.

  Args:
    path: Destination path.

  Returns:
    A tuple of the written path and the source BGR ndarray.
  """
  image = np.zeros((10, 10, 3), dtype=np.uint8)
  image[:, :] = (40, 120, 220)
  image[2:8, 2:8] = (210, 30, 90)
  return write_png(path, image), image


def test_library_decodes_template_in_bgr_order(tmp_path):
  """Template pixels decode to BGR, matching cv2.imdecode screens."""
  path, _ = write_coloured_png(tmp_path / 'mark.png')
  encoded = path.read_bytes()
  manifest = tmp_path / 'manifest.json'
  manifest.write_text(json.dumps({'version': 1,
                                  'templates': {'mark': 'mark.png'}}),
                      encoding='utf-8')
  library = TemplateLibrary(str(tmp_path))
  library.load(str(manifest))
  expected = cv2.imdecode(
    np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
  np.testing.assert_array_equal(library.get('mark'), expected)


def test_matcher_scores_coloured_template_on_bgr_screen(tmp_path):
  """A coloured template matches a BGR screen built from the same image."""
  path, _ = write_coloured_png(tmp_path / 'mark.png')
  encoded = path.read_bytes()
  manifest = tmp_path / 'manifest.json'
  manifest.write_text(json.dumps({'version': 1,
                                  'templates': {'mark': 'mark.png'}}),
                      encoding='utf-8')
  library = TemplateLibrary(str(tmp_path))
  library.load(str(manifest))
  screen = np.zeros((60, 60, 3), dtype=np.uint8)
  screen[25:35, 40:50] = cv2.imdecode(
    np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
  matches = TemplateMatcher(library).match(Screen(screen), 'mark')
  assert matches
  assert matches[0].score >= 0.95
  assert (matches[0].rect.x, matches[0].rect.y) == (40, 25)


def test_library_rejects_wrong_version(tmp_path):
  """A manifest whose version is not 1 is rejected."""
  manifest = make_library(tmp_path, version=2)
  with pytest.raises(VisionError):
    TemplateLibrary(str(tmp_path)).load(str(manifest))


def test_library_rejects_bool_version(tmp_path):
  """A boolean manifest version is rejected even though True == 1."""
  manifest = make_library(tmp_path, version=True)
  with pytest.raises(VisionError):
    TemplateLibrary(str(tmp_path)).load(str(manifest))


def test_library_rejects_manifest_outside_directory(tmp_path):
  """A manifest path outside the library directory is rejected."""
  manifest = make_library(tmp_path)
  with pytest.raises(VisionError):
    TemplateLibrary(str(tmp_path / 'sub')).load(str(manifest))


def test_library_rejects_template_traversal(tmp_path):
  """A template filename traversing outside the directory is rejected."""
  manifest = make_library(tmp_path, templates={'mark': '../escape.png'})
  with pytest.raises(VisionError):
    TemplateLibrary(str(tmp_path)).load(str(manifest))


def test_library_rejects_absolute_template(tmp_path):
  """An absolute template filename is rejected."""
  manifest = make_library(tmp_path, templates={'mark': '/etc/passwd'})
  with pytest.raises(VisionError):
    TemplateLibrary(str(tmp_path)).load(str(manifest))


def test_library_rejects_symlinked_template(tmp_path):
  """A symlinked template file is rejected."""
  manifest = make_library(tmp_path, templates={'mark': 'linked.png'})
  (tmp_path / 'linked.png').symlink_to(tmp_path / 'mark.png')
  with pytest.raises(VisionError):
    TemplateLibrary(str(tmp_path)).load(str(manifest))


def test_library_rejects_oversized_manifest(tmp_path):
  """A manifest larger than the byte cap is rejected before parsing."""
  manifest = make_library(tmp_path)
  with pytest.raises(VisionError):
    TemplateLibrary(str(tmp_path), max_manifest_bytes=10).load(
      str(manifest))


def test_library_rejects_oversized_template_bytes(tmp_path):
  """A template larger than the byte cap is rejected before decode."""
  manifest = make_library(tmp_path)
  with pytest.raises(VisionError):
    TemplateLibrary(str(tmp_path), max_template_bytes=10).load(
      str(manifest))


def test_library_rejects_oversized_template_pixels(tmp_path):
  """A template larger than the pixel cap is rejected before decode."""
  manifest = make_library(tmp_path)
  with pytest.raises(VisionError):
    TemplateLibrary(str(tmp_path), max_template_pixels=5).load(
      str(manifest))


def _bomb_png():
  """Build a tiny valid PNG whose header declares a 50000x50000 image.

  Returns:
    Encoded PNG bytes with valid CRCs for all three chunks.
  """
  def chunk(kind, data):
    """Return one length/type/data/CRC PNG chunk."""
    crc = zlib.crc32(kind + data) & 0xffffffff
    return struct.pack('>I', len(data)) + kind + data + struct.pack(
      '>I', crc)

  header = struct.pack('>IIBBBBB', 50000, 50000, 8, 2, 0, 0, 0)
  return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header)
          + chunk(b'IDAT', zlib.compress(b'')) + chunk(b'IEND', b''))


def test_library_rejects_decompression_bomb(tmp_path):
  """A valid-CRC bomb header is mapped to VisionError before pixel load."""
  (tmp_path / 'bomb.png').write_bytes(_bomb_png())
  manifest = tmp_path / 'manifest.json'
  manifest.write_text(json.dumps({'version': 1,
                                  'templates': {'bomb': 'bomb.png'}}),
                      encoding='utf-8')
  with pytest.raises(VisionError):
    TemplateLibrary(str(tmp_path)).load(str(manifest))


def test_library_rejects_too_many_templates(tmp_path):
  """More templates than the file cap are rejected."""
  manifest = make_library(tmp_path, templates={
    'mark': 'mark.png', 'other': 'mark.png'})
  with pytest.raises(VisionError):
    TemplateLibrary(str(tmp_path), max_files=1).load(str(manifest))


def test_matcher_finds_template_at_expected_location(tmp_path):
  """match locates a template using normalized cross-correlation."""
  manifest = make_library(tmp_path)
  library = TemplateLibrary(str(tmp_path))
  library.load(str(manifest))
  screen = np.zeros((100, 100, 3), dtype=np.uint8)
  screen[60:70, 50:60] = make_template()
  matcher = TemplateMatcher(library)
  matches = matcher.match(Screen(screen), 'mark')
  assert matches
  best = matches[0]
  assert (best.rect.x, best.rect.y) == (50, 60)
  assert (best.rect.width, best.rect.height) == (10, 10)
  assert best.score >= 0.99


def test_match_all_uses_library_names(tmp_path):
  """match_all returns matches for every template in the library."""
  manifest = make_library(tmp_path)
  library = TemplateLibrary(str(tmp_path))
  library.load(str(manifest))
  screen = np.zeros((100, 100, 3), dtype=np.uint8)
  screen[60:70, 50:60] = make_template()
  matches = TemplateMatcher(library).match_all(Screen(screen))
  assert any(match.name == 'mark' for match in matches)


def test_non_max_suppress_keeps_highest_overlapping():
  """NMS keeps the highest-scoring match among overlapping boxes."""
  first = Match('a', Rect(0, 0, 10, 10), 0.9)
  second = Match('b', Rect(1, 1, 10, 10), 0.7)
  third = Match('c', Rect(50, 50, 10, 10), 0.6)
  kept = non_max_suppress([second, first, third], 0.5)
  assert kept == [first, third]


def _reference_iou(first, second):
  """Compute IoU for two rectangles using the pairwise reference."""
  left = max(first.x, second.x)
  top = max(first.y, second.y)
  right = min(first.x + first.width, second.x + second.width)
  bottom = min(first.y + first.height, second.y + second.height)
  overlap = max(0, right - left) * max(0, bottom - top)
  union = first.area() + second.area() - overlap
  return overlap / union if union else 0.0


def _reference_nms(matches, threshold):
  """Emulate the original quadratic suppression for semantic comparison."""
  ordered = sorted(matches, key=lambda match: match.score, reverse=True)
  kept = []
  for candidate in ordered:
    if all(_reference_iou(candidate.rect, other.rect) < threshold
           for other in kept):
      kept.append(candidate)
  return kept


def test_non_max_suppress_handles_thousands_of_matches_quickly():
  """NMS suppresses thousands of matches without quadratic scanning."""
  matches = [
    Match(f'm{i}', Rect((i % 100) * 40, (i // 100) * 40, 10, 10),
          i / 3000.0)
    for i in range(3000)
  ]
  start = time.perf_counter()
  kept = non_max_suppress(matches, 0.5)
  elapsed = time.perf_counter() - start
  assert elapsed < 0.5
  assert len(kept) == 3000


def test_non_max_suppress_bounds_dense_candidates():
  """Dense tiny matches are bounded before quadratic suppression."""
  matches = [
    Match(f'm{i}', Rect(i % 200, i // 200, 1, 1), 1.0)
    for i in range(6000)
  ]
  start = time.perf_counter()
  kept = non_max_suppress(matches, 0.5)
  elapsed = time.perf_counter() - start
  assert elapsed < 2.0
  assert len(kept) <= 3000


def test_match_dense_tiny_template_is_fast(tmp_path):
  """A 1x1 template on a uniform screen avoids quadratic NMS."""
  write_png(tmp_path / 'tiny.png', np.full((1, 1, 3), 200, np.uint8))
  manifest = tmp_path / 'manifest.json'
  manifest.write_text(json.dumps({'version': 1,
                                  'templates': {'tiny': 'tiny.png'}}),
                      encoding='utf-8')
  library = TemplateLibrary(str(tmp_path))
  library.load(str(manifest))
  screen = np.full((200, 200, 3), 200, dtype=np.uint8)
  start = time.perf_counter()
  matches = TemplateMatcher(library).match(Screen(screen), 'tiny')
  elapsed = time.perf_counter() - start
  assert elapsed < 4.0
  assert len(matches) <= 3000


def test_non_max_suppress_matches_pairwise_reference_semantics():
  """Grid suppression keeps exactly the pairwise-scan survivors."""
  rng = np.random.default_rng(3)
  matches = [
    Match(f'm{i}',
          Rect(int(rng.integers(0, 20)), int(rng.integers(0, 20)),
               12, 12),
          float(rng.random()))
    for i in range(120)
  ]
  assert non_max_suppress(matches, 0.5) == _reference_nms(matches, 0.5)


def test_dominant_color_returns_most_frequent_bgr():
  """dominant_color reports the most frequent BGR triple in a rect."""
  pixels = np.zeros((10, 10, 3), dtype=np.uint8)
  pixels[:4] = (10, 20, 30)
  pixels[4:] = (200, 100, 50)
  color = dominant_color(Screen(pixels), Rect(0, 0, 10, 10))
  assert color == (200, 100, 50)


def test_color_distance_is_euclidean():
  """color_distance computes the Euclidean distance between colors."""
  assert color_distance((0, 0, 0), (3, 4, 0)) == pytest.approx(5.0)
  assert color_distance((1, 2, 3), (1, 2, 3)) == 0.0


def graded_template():
  """Build a low-contrast template whose patches stay inside 0..255.

  Returns:
    A 10x10 BGR ndarray with a small bright square on a mid-grey field.
  """
  image = np.full((10, 10, 3), 128, dtype=np.uint8)
  image[2:8, 2:8] = 148
  return image


def patch_with_correlation(template, target):
  """Build a patch with a known TM_CCOEFF_NORMED score to ``template``.

  Args:
    template: The reference template pixel array.
    target: Desired normalized correlation in the open interval (0, 1).

  Returns:
    A same-size uint8 patch with correlation approximately ``target``.
  """
  base = template.astype(np.float64)
  mean = float(base.mean())
  centred = base - mean
  norm = float(np.linalg.norm(centred))
  other = np.random.default_rng(7).random(centred.shape)
  other -= other.mean()
  other -= (np.sum(other * centred) / np.sum(centred ** 2)) * centred
  other *= norm / np.linalg.norm(other)
  patch = target * centred + np.sqrt(1 - target ** 2) * other + mean
  return np.clip(patch, 0, 255).astype(np.uint8)


def test_craft_ready_check_has_locked_097_override():
  """craft_ready_check uses a 0.97 score override defined in code."""
  assert TEMPLATE_SCORE_OVERRIDES['craft_ready_check'] == 0.97


def test_per_template_score_override_controls_threshold(tmp_path,
                                                        monkeypatch):
  """An overridden template uses its own score, not the global threshold."""
  template = graded_template()
  write_png(tmp_path / 'mark.png', template)
  manifest = tmp_path / 'manifest.json'
  manifest.write_text(json.dumps({'version': 1,
                                  'templates': {'mark': 'mark.png'}}),
                      encoding='utf-8')
  library = TemplateLibrary(str(tmp_path))
  library.load(str(manifest))
  patch = patch_with_correlation(template, 0.90)
  screen_array = np.zeros((60, 60, 3), dtype=np.uint8)
  screen_array[25:35, 40:50] = patch
  screen = Screen(screen_array)
  raw = cv2.matchTemplate(screen_array, template, cv2.TM_CCOEFF_NORMED)
  assert 0.85 < float(raw.max()) < 0.95
  assert TemplateMatcher(library).match(screen, 'mark')
  monkeypatch.setitem(TEMPLATE_SCORE_OVERRIDES, 'mark', 0.97)
  assert not TemplateMatcher(library).match(screen, 'mark')


def test_real_craft_ready_template_rejects_0895_and_accepts_completed(
    tmp_path):
  """The real template only matches a completed tile, not a normal shelf."""
  shutil.copy(PROJECT_ROOT / 'templates' / 'craft_ready_check.png',
              tmp_path / 'craft_ready_check.png')
  manifest = tmp_path / 'manifest.json'
  manifest.write_text(json.dumps({
      'version': 1,
      'templates': {'craft_ready_check': 'craft_ready_check.png'},
  }), encoding='utf-8')
  library = TemplateLibrary(str(tmp_path))
  library.load(str(manifest))
  template = library.get('craft_ready_check')
  height, width = template.shape[:2]
  for target, expected in ((0.895, False), (1.0, True)):
    patch = patch_with_correlation(template, target)
    pixels = np.full((height + 40, width + 40, 3), 128, dtype=np.uint8)
    pixels[20:20 + height, 20:20 + width] = patch
    matched = TemplateMatcher(library).match(
        Screen(pixels), 'craft_ready_check')
    assert bool(matched) is expected


def test_recipe_dialog_overrides_are_pinned_in_code():
  """recipe_marker 0.65 and craft_button 0.70 overrides are pinned in code."""
  assert TEMPLATE_SCORE_OVERRIDES['recipe_marker'] == 0.65
  assert TEMPLATE_SCORE_OVERRIDES['craft_button'] == 0.70

