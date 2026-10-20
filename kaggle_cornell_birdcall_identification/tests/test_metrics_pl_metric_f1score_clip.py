#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the clip level micro F1.

A row with no positive label is a **background** row: it is scored 1.0
when nothing fires and 0.0 when something does, so a false alarm on a
background row costs a full item.  The metric reduces then divides, so
the number is the mean of the per-item scores and not a pooled micro F1.
'''

import pytest
import torch
from sklearn.metrics import f1_score

from birdcall.metrics.pl_metric_f1score_clip import SedF1ScoreClip

BATCH = 2
SNIPPETS = 2
CLASSES = 6


def test_background_row_with_nothing_firing_scores_one(metric_output,
                                                         metric_labels):
  '''A background row with a quiet prediction scores a perfect one.'''
  clip = torch.full((1, SNIPPETS, CLASSES), 0.1)
  metric = SedF1ScoreClip()
  metric.update(metric_output(clip, labels=metric_labels([])))
  assert metric.compute() == pytest.approx(1.0)


def test_background_row_with_something_firing_scores_zero(metric_output,
                                                           metric_labels):
  '''A false alarm on a background row costs a full item.'''
  clip = torch.full((1, SNIPPETS, CLASSES), 0.1)
  clip[0, 0, 0] = 0.9
  metric = SedF1ScoreClip()
  metric.update(metric_output(clip, labels=metric_labels([])))
  assert metric.compute() == pytest.approx(0.0)


def test_the_clip_metric_only_looks_at_the_clip_head(metric_output,
                                                      metric_labels):
  '''The frame head plays no part in the clip metric.'''
  clip = torch.full((BATCH, SNIPPETS, CLASSES), 0.3)
  frame = torch.full((BATCH, SNIPPETS, 5, CLASSES), 0.9)
  metric = SedF1ScoreClip()
  metric.update(metric_output(clip, frame, labels=metric_labels([0], [0])))
  # 0.3 is below the 0.5 threshold of the clip metric.
  assert metric.compute() == pytest.approx(0.0)


def test_metrics_reduce_then_divide(metric_output, metric_labels):
  '''The metric is the mean of the per-item scores, not a pooled one.'''
  # The first item is a perfect detection, the second one is a total
  # miss.
  clip = torch.zeros(BATCH, SNIPPETS, CLASSES)
  clip[0, 0, 0] = 0.9
  clip[0, 0, 1] = 0.9
  labels = metric_labels([0, 1], [2, 3])
  metric = SedF1ScoreClip()
  metric.update(metric_output(clip, labels=labels))
  assert metric.compute() == pytest.approx(0.5)
  # A pooled micro F1 over the two items would be 2/3.
  pooled = f1_score(labels[:, 0, :].numpy(),
                    clip.max(axis=1)[0].numpy() >= 0.5, average='micro')
  assert pooled == pytest.approx(2.0 / 3.0)
  assert metric.compute() != pytest.approx(pooled)


def test_the_clip_metric_does_not_apply_a_sigmoid(metric_output,
                                                   metric_labels):
  '''The attention block already applied the sigmoid.'''
  # 0.4 is below the threshold, but sigmoid(0.4) is above it.
  clip = torch.full((BATCH, SNIPPETS, CLASSES), 0.4)
  metric = SedF1ScoreClip()
  metric.update(metric_output(clip, labels=metric_labels([0], [0])))
  assert metric.compute() == pytest.approx(0.0)


def test_metric_reset_clears_the_accumulator(metric_output, metric_labels):
  '''The reset zeroes the sum and the count.'''
  clip = torch.zeros(BATCH, SNIPPETS, CLASSES)
  clip[0, 0, 0] = 0.9
  metric = SedF1ScoreClip()
  metric.update(metric_output(clip, labels=metric_labels([0], [0])))
  assert metric.compute() == pytest.approx(0.5)
  metric.reset()
  assert metric.compute() == pytest.approx(0.0)


def test_an_empty_accumulator_scores_zero():
  '''Nothing accumulated means nothing scored.'''
  assert SedF1ScoreClip().compute() == pytest.approx(0.0)
