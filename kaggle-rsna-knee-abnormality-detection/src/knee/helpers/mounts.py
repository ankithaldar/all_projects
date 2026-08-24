#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Locate attached Kaggle inputs regardless of mount-path layout.

Kaggle has changed mount layouts over time: competition data appears
under ``/kaggle/input/competitions/<slug>``, classic datasets under
``/kaggle/input/<slug>``, and some newer UI flows nest user datasets
as ``/kaggle/input/datasets/<owner>/<slug>``. Code that hardcodes one
layout silently misses restores (the bug this module fixes).

``find_input_mount`` searches a *bounded depth* of the input tree --
never descending into the multi-hundred-GB DICOM payload -- and
returns the first directory whose basename equals the wanted slug.
"""

from __future__ import annotations

from pathlib import Path


def find_input_mount(
  name: str,
  root: str | Path = '/kaggle/input',
  max_depth: int = 3,
) -> str:
  """Find an attached input directory by its slug.

  Args:
      name: Directory name to match (the dataset slug).
      root: Input root to search.
      max_depth: Maximum directory depth below ``root``; bounded so
          huge data payloads are never walked.

  Returns:
      Absolute path of the first matching directory ('' when absent).
      Matches at shallower depths win; ties resolve alphabetically,
      making the choice deterministic across kernels.
  """
  base = Path(root)
  if not base.is_dir():
    return ''
  for depth in range(1, max_depth + 1):
    pattern = '/'.join(['*'] * depth)
    for candidate in sorted(base.glob(pattern)):
      if candidate.is_dir() and candidate.name == name:
        return str(candidate)
  return ''
