#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the label ranking average precision score.

The metric scores the **ranking** of the clip head, so a correct class
that is ranked last scores badly.  It reduces then divides: the number
is the mean of the per-item scores and not a pooled score.
'''

import pytest
import torch

from birdcall.metrics.pl_metric_lraps import SedLRAPS


def test_lraps_scores_the_ranking_of_the_clip_head(metric_output,
                                                    metric_labels):
  '''The relevant class is ranked third, so its precision is a third.'''
  clip = torch.zeros(1, 2, 6)
  clip[0, 0, 0] = 0.2
  clip[0, 0, 1] = 0.9
  clip[0, 0, 2] = 0.5
  metric = SedLRAPS()
  metric.update(metric_output(clip, labels=metric_labels([0])))
  # The relevant class is ranked third, so its average precision is 1/3.
  assert metric.compute() == pytest.approx(1.0 / 3.0)


def test_lraps_gives_a_label_less_row_a_free_one(metric_output,
                                                  metric_labels):
  '''Without the background branch a label-less row scores a perfect 1.'''
  clip = torch.full((1, 2, 6), 0.5)
  metric = SedLRAPS()
  metric.update(metric_output(clip, labels=metric_labels([])))
  assert metric.compute() == pytest.approx(1.0)


def test_lraps_background_branch_punishes_a_firing_row(metric_output,
                                                       metric_labels):
  '''With the background branch a firing row scores zero.'''
  clip = torch.full((1, 2, 6), 0.5)
  metric = SedLRAPS(with_background_branch=True)
  metric.update(metric_output(clip, labels=metric_labels([])))
  assert metric.compute() == pytest.approx(0.0)
  quiet = torch.full((1, 2, 6), 0.1)
  metric = SedLRAPS(with_background_branch=True)
  metric.update(metric_output(quiet, labels=metric_labels([])))
  assert metric.compute() == pytest.approx(1.0)


def test_lraps_does_not_apply_a_sigmoid(metric_output, metric_labels):
  '''The attention block already applied the sigmoid.'''
  # 0.4 is below the fire threshold, but sigmoid(0.4) is above it.
  clip = torch.full((1, 2, 6), 0.4)
  metric = SedLRAPS(with_background_branch=True)
  metric.update(metric_output(clip, labels=metric_labels([])))
  assert metric.compute() == pytest.approx(1.0)


def test_an_empty_accumulator_scores_zero():
  '''Nothing accumulated means nothing scored.'''
  assert SedLRAPS().compute() == pytest.approx(0.0)
