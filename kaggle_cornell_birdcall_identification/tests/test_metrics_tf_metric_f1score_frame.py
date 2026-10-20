#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The tests of the frame gated micro F1.

The frame metric is the **AND** of two independent gates: the frame
threshold and a hard coded 0.3 clip gate.  The 0.3 is a literal, and
the two gates must stay independent.
'''


import tensorflow as tf
import numpy as np
import pytest
from sklearn.metrics import f1_score

from birdcall.metrics.tf_metric_f1score_frame import CLIP_GATE
from birdcall.metrics.tf_metric_f1score_frame import SedF1ScoreFrame


CLASSES = 12
FRAMES = 5

def batch(seed: int = 0, rows: int = 4) -> tuple[np.ndarray, np.ndarray,
                                                np.ndarray]:
  '''Builds a deterministic batch of labels, clip scores and frame scores.

  Args:
    seed: The seed of the generator.
    rows: The number of items.

  Returns:
    A tuple ``(labels, clip scores, frame scores)``.
  '''
  rng = np.random.default_rng(seed)
  labels = (rng.random((rows, CLASSES)) > 0.7).astype('float32')
  clip = rng.random((rows, CLASSES)).astype('float32')
  frames = rng.random((rows, FRAMES, CLASSES)).astype('float32')
  return labels, clip, frames

def sklearn_score(labels: np.ndarray, clip: np.ndarray,
                  frames: np.ndarray, threshold: float = 0.5) -> float:
  '''Computes the mean frame gated micro F1 of a batch with sklearn.

  Args:
    labels: The labels.
    clip: The clip scores.
    frames: The frame scores.
    threshold: The frame threshold.

  Returns:
    The mean of the per-item scores.
  '''
  pooled = frames.max(axis=1)
  values = []
  for label, pc, pf in zip(labels, clip, pooled):
    decision = (pf >= threshold) & (pc >= CLIP_GATE)
    if label.sum() == 0.0:
      values.append(0.0 if decision.any() else 1.0)
    else:
      values.append(
          float(f1_score([label], [decision], average='micro')))
  return float(np.mean(values))

def test_the_clip_gate_is_a_literal() -> None:
  '''The 0.3 gate is a literal, not a config value.'''
  assert CLIP_GATE == 0.3

def test_the_score_matches_sklearn() -> None:
  '''The score is sklearn's micro F1 over the gated decision.'''
  labels, clip, frames = batch()
  metric = SedF1ScoreFrame()
  metric.update_state({'all_labels': labels},
                      {'clipwise_output': clip, 'framewise_output': frames})
  assert float(metric.result()) == pytest.approx(
      sklearn_score(labels, clip, frames), abs=1e-6)

def test_the_frame_head_is_max_pooled_over_time() -> None:
  '''The frame axis is destroyed, exactly like the original.'''
  labels, clip, frames = batch(seed=1)
  metric = SedF1ScoreFrame()
  metric.update_state({'all_labels': labels},
                      {'clipwise_output': clip, 'framewise_output': frames})
  expected = sklearn_score(labels, clip, frames)
  assert float(metric.result()) == pytest.approx(expected, abs=1e-6)
  # A frame head whose every frame is the maximum gives the same score:
  # the frame axis is destroyed by the max pooling.
  pooled = frames.max(axis=1)
  other = np.repeat(pooled[:, None, :], FRAMES, axis=1)
  second = SedF1ScoreFrame()
  second.update_state({'all_labels': labels},
                       {'clipwise_output': clip, 'framewise_output': other})
  assert float(second.result()) == pytest.approx(expected, abs=1e-6)

def test_the_clip_gate_is_independent_of_the_frame_threshold() -> None:
  '''A frame that passes but whose clip is below 0.3 fires nothing.'''
  labels = np.zeros((1, CLASSES), dtype='float32')
  labels[0, 0] = 1.0
  clip = np.zeros((1, CLASSES), dtype='float32')
  clip[0, 0] = 0.29
  frames = np.zeros((1, FRAMES, CLASSES), dtype='float32')
  frames[:, :, 0] = 0.9
  metric = SedF1ScoreFrame()
  metric.update_state({'all_labels': labels},
                      {'clipwise_output': clip, 'framewise_output': frames})
  # The frame passes, but the clip does not, so nothing is detected.
  assert float(metric.result()) == pytest.approx(0.0, abs=1e-6)
  # The very same prediction with a clip score above the gate fires.
  clip[0, 0] = 0.31
  metric = SedF1ScoreFrame()
  metric.update_state({'all_labels': labels},
                      {'clipwise_output': clip, 'framewise_output': frames})
  assert float(metric.result()) == pytest.approx(1.0, abs=1e-6)

def test_the_frame_threshold_is_independent_of_the_clip_gate() -> None:
  '''A clip that passes but whose frame does not fires nothing.'''
  labels = np.zeros((1, CLASSES), dtype='float32')
  labels[0, 0] = 1.0
  clip = np.zeros((1, CLASSES), dtype='float32')
  clip[0, 0] = 0.9
  frames = np.zeros((1, FRAMES, CLASSES), dtype='float32')
  frames[:, :, 0] = 0.49
  metric = SedF1ScoreFrame()
  metric.update_state({'all_labels': labels},
                      {'clipwise_output': clip, 'framewise_output': frames})
  assert float(metric.result()) == pytest.approx(0.0, abs=1e-6)
  frames[:, :, 0] = 0.51
  metric = SedF1ScoreFrame()
  metric.update_state({'all_labels': labels},
                      {'clipwise_output': clip, 'framewise_output': frames})
  assert float(metric.result()) == pytest.approx(1.0, abs=1e-6)

def test_a_background_row_that_fires_nothing_scores_one() -> None:
  '''The background branch: nothing fires, so the row is perfect.'''
  labels = np.zeros((1, CLASSES), dtype='float32')
  clip = np.zeros((1, CLASSES), dtype='float32')
  frames = np.zeros((1, FRAMES, CLASSES), dtype='float32')
  metric = SedF1ScoreFrame()
  metric.update_state({'all_labels': labels},
                      {'clipwise_output': clip, 'framewise_output': frames})
  assert float(metric.result()) == pytest.approx(1.0, abs=1e-6)

def test_a_background_row_that_fires_scores_zero() -> None:
  '''The background branch: a false alarm costs the whole item.'''
  labels = np.zeros((1, CLASSES), dtype='float32')
  clip = np.zeros((1, CLASSES), dtype='float32')
  clip[0, 2] = 0.9
  frames = np.zeros((1, FRAMES, CLASSES), dtype='float32')
  frames[:, :, 2] = 0.9
  metric = SedF1ScoreFrame()
  metric.update_state({'all_labels': labels},
                      {'clipwise_output': clip, 'framewise_output': frames})
  assert float(metric.result()) == pytest.approx(0.0, abs=1e-6)

def test_the_threshold_is_honoured() -> None:
  '''The threshold decides what counts as a frame detection.'''
  labels, clip, frames = batch(seed=2)
  metric = SedF1ScoreFrame(threshold=0.9)
  metric.update_state({'all_labels': labels},
                      {'clipwise_output': clip, 'framewise_output': frames})
  assert float(metric.result()) == pytest.approx(
      sklearn_score(labels, clip, frames, 0.9), abs=1e-6)

def test_the_metric_survives_a_save_and_load_round_trip() -> None:
  '''The configuration of the metric can be read back.'''
  metric = SedF1ScoreFrame(threshold=0.3)
  config = tf.keras.utils.serialize_keras_object(metric)
  assert config['class_name'] == 'SedF1ScoreFrame'
  assert config['config']['threshold'] == 0.3
  restored = tf.keras.utils.deserialize_keras_object(config)
  assert isinstance(restored, SedF1ScoreFrame)
  assert restored.threshold == 0.3
  labels, clip, frames = batch()
  restored.update_state({'all_labels': labels},
                        {'clipwise_output': clip,
                         'framewise_output': frames})
  assert float(restored.result()) == pytest.approx(
      sklearn_score(labels, clip, frames, 0.3), abs=1e-6)
