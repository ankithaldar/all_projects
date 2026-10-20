#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the frame gated micro F1.

The decision is the **AND** of two independent gates: the frame head at
the threshold, and the clip head at the hard coded `0.3`.  The `0.3` is
a literal, not a config value, and the two gates stay independent.
'''

import pytest
import torch

from birdcall.metrics.pl_metric_f1score_frame import CLIP_GATE
from birdcall.metrics.pl_metric_f1score_frame import SedF1ScoreFrame

BATCH = 2
SNIPPETS = 2
CLASSES = 6
FRAMES = 5


def test_frame_metric_background_branch(metric_output, metric_labels):
  '''A background row is scored by the decision, not by the frame head.'''
  frame = torch.full((1, SNIPPETS, FRAMES, CLASSES), 0.1)
  clip = torch.full((1, SNIPPETS, CLASSES), 0.1)
  metric = SedF1ScoreFrame()
  metric.update(metric_output(clip, frame, labels=metric_labels([])))
  assert metric.compute() == pytest.approx(1.0)
  # The frame head alone does not fire the decision, the clip gate at
  # 0.3 blocks it.
  frame[0, 0, 0, 0] = 0.9
  metric = SedF1ScoreFrame()
  metric.update(metric_output(clip, frame, labels=metric_labels([])))
  assert metric.compute() == pytest.approx(1.0)
  clip[0, 0, 0] = 0.9
  metric = SedF1ScoreFrame()
  metric.update(metric_output(clip, frame, labels=metric_labels([])))
  assert metric.compute() == pytest.approx(0.0)


def test_the_frame_gate_and_the_clip_gate_are_independent(metric_output,
                                                          metric_labels):
  '''The clip gate opens at exactly 0.3, independently of the frame one.'''
  # The frame head fires, the clip head does not: the clip gate at 0.3
  # blocks the detection, so the item contributes nothing.
  clip = torch.full((BATCH, SNIPPETS, CLASSES), 0.1)
  frame = torch.full((BATCH, SNIPPETS, FRAMES, CLASSES), 0.1)
  frame[:, :, :, 0] = 0.9
  metric = SedF1ScoreFrame()
  metric.update(metric_output(clip, frame, labels=metric_labels([0], [0])))
  assert metric.compute() == pytest.approx(0.0)
  # The clip gate opens at exactly 0.3, independently of the frame one.
  clip[:, :, 0] = 0.3
  metric = SedF1ScoreFrame()
  metric.update(metric_output(clip, frame, labels=metric_labels([0], [0])))
  assert metric.compute() == pytest.approx(1.0)


def test_frame_metric_needs_the_frame_gate_too(metric_output,
                                               metric_labels):
  '''The clip head alone does not fire the decision either.'''
  clip = torch.full((BATCH, SNIPPETS, CLASSES), 0.9)
  frame = torch.full((BATCH, SNIPPETS, FRAMES, CLASSES), 0.1)
  metric = SedF1ScoreFrame()
  metric.update(metric_output(clip, frame, labels=metric_labels([0], [0])))
  assert metric.compute() == pytest.approx(0.0)


def test_the_frame_metric_does_not_apply_a_sigmoid(metric_output,
                                                    metric_labels):
  '''The attention block already applied the sigmoid.'''
  # 0.4 is below the threshold, but sigmoid(0.4) is above it.
  clip = torch.full((BATCH, SNIPPETS, CLASSES), 0.4)
  frame = torch.full((BATCH, SNIPPETS, FRAMES, CLASSES), 0.4)
  metric = SedF1ScoreFrame()
  metric.update(metric_output(clip, frame, labels=metric_labels([0], [0])))
  assert metric.compute() == pytest.approx(0.0)


def test_the_clip_gate_is_a_literal():
  '''The 0.3 is hard coded, it is not a config value.'''
  assert CLIP_GATE == 0.3


def test_an_empty_accumulator_scores_zero():
  '''Nothing accumulated means nothing scored.'''
  assert SedF1ScoreFrame().compute() == pytest.approx(0.0)
