#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for Kaggle input-mount discovery (varying UI layouts)."""

from __future__ import annotations

from pathlib import Path

from knee.helpers.mounts import find_input_mount


def _tree(base: Path, relative_dirs: list[str]) -> Path:
  """Materialize a fake /kaggle/input tree.

  Args:
      base: Root to create the tree under.
      relative_dirs: Directory paths to create.

  Returns:
      The root path.
  """
  for rel in relative_dirs:
    (base / rel).mkdir(parents=True, exist_ok=True)
  return base


class TestFindInputMount:
  def test_classic_flat_layout(self, tmp_path: Path):
    _tree(tmp_path, ['ah2022-rsna-knee-abnormality-detection', 'other-ds'])
    assert find_input_mount(
      'ah2022-rsna-knee-abnormality-detection', root=tmp_path
    ) == str(tmp_path / 'ah2022-rsna-knee-abnormality-detection')

  def test_nested_datasets_owner_layout(self, tmp_path: Path):
    _tree(
      tmp_path,
      [
        'competitions/rsna-knee-abnormality-detection',
        'datasets/haldarankit/ah2022-rsna-knee-abnormality-detection',
      ],
    )
    found = find_input_mount(
      'ah2022-rsna-knee-abnormality-detection', root=tmp_path
    )
    assert found == str(
      tmp_path / 'datasets/haldarankit/ah2022-rsna-knee-abnormality-detection'
    )

  def test_shallower_match_wins(self, tmp_path: Path):
    _tree(
      tmp_path,
      ['a/deep/nest/slug', 'slug'],
    )
    found = find_input_mount('slug', root=tmp_path)
    assert found == str(tmp_path / 'slug')

  def test_missing_returns_empty(self, tmp_path: Path):
    _tree(tmp_path, ['something-else'])
    assert find_input_mount('ah2022-rsna-knee', root=tmp_path) == ''

  def test_missing_root_returns_empty(self, tmp_path: Path):
    assert find_input_mount('any', root=tmp_path / 'nope') == ''

  def test_never_descends_beyond_max_depth(self, tmp_path: Path):
    _tree(tmp_path, ['a/b/c/d/e/slug'])
    assert find_input_mount('slug', root=tmp_path, max_depth=3) == ''
