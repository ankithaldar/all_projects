#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The cross-backend parity of the three validation metrics.

The torch metrics and the Keras metrics must score the same batch the
same way, to 1e-6.
'''

import numpy as np
import pytest
import torch

from birdcall.metrics.pl_metric_f1score_clip import SedF1ScoreClip as PlClip
from birdcall.metrics.pl_metric_f1score_frame import SedF1ScoreFrame as PlFrame
from birdcall.metrics.pl_metric_lraps import SedLRAPS as PlLrap
from birdcall.metrics.tf_metric_f1score_clip import SedF1ScoreClip as TfClip
from birdcall.metrics.tf_metric_f1score_frame import SedF1ScoreFrame as TfFrame
from birdcall.metrics.tf_metric_lraps import SedLRAPS as TfLrap

BATCH = 4
SNIPPETS = 2
FRAMES = 5
CLASSES = 10


def batch(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  '''Builds a deterministic batch of labels, clip scores and frame scores.

  Args:
    seed: The seed of the generator.

  Returns:
    A tuple ``(labels, clip scores, frame scores)``.
  '''
  rng = np.random.default_rng(seed)
  labels = (rng.random((BATCH, SNIPPETS, CLASSES)) > 0.7).astype('float32')
  clip = rng.random((BATCH, SNIPPETS, CLASSES)).astype('float32')
  frames = rng.random((BATCH, SNIPPETS, FRAMES, CLASSES)).astype('float32')
  return labels, clip, frames


def torch_output(labels, clip, frames) -> dict:
  '''Builds the output dict the torch metrics are fed with.

  Args:
    labels: The labels.
    clip: The clip scores.
    frames: The frame scores.

  Returns:
    The output dict.
  '''
  return {
      'y_pred': {
          'clipwise_output': torch.from_numpy(clip),
          'framewise_output': torch.from_numpy(frames),
      },
      'y': {'all_labels': torch.from_numpy(labels)},
  }


def keras_output(labels, clip, frames) -> tuple[dict, dict]:
  '''Builds the label dict and the output dict the Keras metrics get.

  Args:
    labels: The labels.
    clip: The clip scores.
    frames: The frame scores.

  Returns:
    A tuple ``(y_true, y_pred)``.
  '''
  return (
      {'all_labels': labels},
      {'clipwise_output': clip, 'framewise_output': frames},
  )


@pytest.mark.parity
def test_the_lraps_scores_agree() -> None:
  '''The label ranking score is the same number.'''
  labels, clip, frames = batch()
  torch_metric = PlLrap()
  torch_metric.update(torch_output(labels, clip, frames))
  keras_metric = TfLrap()
  keras_metric.update_state(*keras_output(labels, clip, frames))
  assert float(keras_metric.result()) == pytest.approx(
      float(torch_metric.compute()), abs=1e-6)


@pytest.mark.parity
def test_the_clip_f1_scores_agree() -> None:
  '''The clip level micro F1 is the same number.'''
  labels, clip, frames = batch(seed=1)
  torch_metric = PlClip(threshold=0.5)
  torch_metric.update(torch_output(labels, clip, frames))
  keras_metric = TfClip(threshold=0.5)
  keras_metric.update_state(*keras_output(labels, clip, frames))
  assert float(keras_metric.result()) == pytest.approx(
      float(torch_metric.compute()), abs=1e-6)


@pytest.mark.parity
def test_the_frame_f1_scores_agree() -> None:
  '''The frame gated micro F1 is the same number.'''
  labels, clip, frames = batch(seed=2)
  torch_metric = PlFrame(threshold=0.5)
  torch_metric.update(torch_output(labels, clip, frames))
  keras_metric = TfFrame(threshold=0.5)
  keras_metric.update_state(*keras_output(labels, clip, frames))
  assert float(keras_metric.result()) == pytest.approx(
      float(torch_metric.compute()), abs=1e-6)


@pytest.mark.parity
def test_the_background_branch_agrees() -> None:
  '''A label-less row is scored the same way by both backends.'''
  labels, clip, frames = batch(seed=3)
  labels[0] = 0.0
  clip[0] = 0.0
  torch_metric = PlClip(threshold=0.5)
  torch_metric.update(torch_output(labels, clip, frames))
  keras_metric = TfClip(threshold=0.5)
  keras_metric.update_state(*keras_output(labels, clip, frames))
  assert float(keras_metric.result()) == pytest.approx(
      float(torch_metric.compute()), abs=1e-6)


@pytest.mark.parity
def test_the_frame_gate_agrees() -> None:
  '''The hard coded 0.3 clip gate is the same in both backends.'''
  labels, clip, frames = batch(seed=4)
  torch_metric = PlFrame(threshold=0.5)
  torch_metric.update(torch_output(labels, clip, frames))
  keras_metric = TfFrame(threshold=0.5)
  keras_metric.update_state(*keras_output(labels, clip, frames))
  assert float(keras_metric.result()) == pytest.approx(
      float(torch_metric.compute()), abs=1e-6)
  # The gate is a literal, and both backends use the same one.
  from birdcall.metrics.tf_metric_f1score_frame import CLIP_GATE
  assert CLIP_GATE == 0.3
