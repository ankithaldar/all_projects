#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression tests for the weak-label rule-precision measurement.

Covers the pandas merge-suffix trap: mask columns only get suffixed on
*collisions*, so ``measure_rule_precision`` must rename them explicitly
or ``{target}_m`` never exists (KeyError seen on Kaggle kernel 3).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from build_weak_labels import measure_rule_precision  # noqa: E402
from knee.config_params.schema import TARGETS  # noqa: E402

_T0, _T1 = list(TARGETS[:2])


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Build a tiny gold/probs/mask triple with known precision.

  Returns:
      (gold, probs, mask) frames for four studies.
  """
  targets = [_T0, _T1]
  gold = pd.DataFrame(
    {
      'StudyInstanceUID': [f's{i}' for i in range(4)],
      targets[0]: [1, 0, 1, 0],
      targets[1]: [0, 0, 1, 1],
    }
  )
  probs = pd.DataFrame(
    {
      'StudyInstanceUID': [f's{i}' for i in range(4)],
      targets[0]: [1.0, 1.0, 1.0, 0.0],
      targets[1]: [0.0, 1.0, 1.0, 0.0],
    }
  )
  mask = pd.DataFrame(
    {
      'StudyInstanceUID': [f's{i}' for i in range(4)],
      targets[0]: [True, True, True, False],
      targets[1]: [False, False, False, True],
    }
  )
  return gold, probs, mask


class TestMeasureRulePrecision:
  def test_computes_per_target_precision(self):
    gold, probs, mask = _frames()
    precision = measure_rule_precision(gold, probs, mask)
    # target0: seeds s0,s1,s2 all positive; gold positive in s0,s2 -> 2/3.
    assert precision[_T0] == pytest.approx(2 / 3)
    # target1: seed only s4 (negative seed) -> no positive seeds -> NaN.
    assert np.isnan(precision[_T1])

  def test_mask_columns_survive_merge(self):
    """The original failure: {target}_m missing after double merge."""
    gold, probs, mask = _frames()
    joined = gold.merge(
      probs, on='StudyInstanceUID', suffixes=('_gold', '_rule')
    ).merge(
      mask.rename(columns={t: f'{t}_m' for t in TARGETS}),
      on='StudyInstanceUID',
    )
    assert f'{_T0}_m' in joined.columns
    assert f'{_T1}_m' in joined.columns

  def test_no_seeds_yields_nan_series(self):
    gold, probs, _ = _frames()
    empty_mask = probs.copy()
    empty_mask[list(TARGETS[:2])] = False
    precision = measure_rule_precision(gold, probs, empty_mask)
    assert precision.notna().sum() == 0
