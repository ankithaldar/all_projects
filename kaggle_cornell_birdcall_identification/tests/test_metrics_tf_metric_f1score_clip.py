#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The tests of the clip level micro F1.

The F1 is hand written, because ``keras.metrics.F1Score`` defaults to
a multiclass argmax, which is categorically wrong for 264 labels.
'''


import tensorflow as tf
import numpy as np
import pytest
from sklearn.metrics import f1_score

from birdcall.metrics.tf_metric_f1score_clip import SedF1ScoreClip


CLASSES = 12

def batch(seed: int = 0, rows: int = 5) -> tuple[np.ndarray, np.ndarray]:
  '''Builds a deterministic batch of labels and scores.

  Args:
    seed: The seed of the generator.
    rows: The number of items.

  Returns:
    A tuple ``(labels, scores)``.
  '''
  rng = np.random.default_rng(seed)
  labels = (rng.random((rows, CLASSES)) > 0.7).astype('float32')
  scores = rng.random((rows, CLASSES)).astype('float32')
  return labels, scores

def sklearn_score(labels: np.ndarray, scores: np.ndarray,
                  threshold: float = 0.5) -> float:
  '''Computes the mean micro F1 of a batch with sklearn.

  Args:
    labels: The labels.
    scores: The scores.
    threshold: The detection threshold.

  Returns:
    The mean of the per-item scores.
  '''
  values = []
  for label, score in zip(labels, scores):
    if label.sum() == 0.0:
      values.append(0.0 if (score >= threshold).any() else 1.0)
    else:
      values.append(
          float(f1_score([label], [score >= threshold], average='micro')))
  return float(np.mean(values))

def test_the_score_matches_sklearn() -> None:
  '''The score is sklearn's micro F1, to 1e-6.'''
  labels, scores = batch()
  metric = SedF1ScoreClip()
  metric.update_state(labels, scores)
  assert float(metric.result()) == pytest.approx(
      sklearn_score(labels, scores), abs=1e-6)

def test_the_score_matches_sklearn_with_ties() -> None:
  '''Tied scores are detected the same way.'''
  labels, scores = batch(seed=1)
  scores = np.round(scores, 1)
  metric = SedF1ScoreClip()
  metric.update_state(labels, scores)
  assert float(metric.result()) == pytest.approx(
      sklearn_score(labels, scores), abs=1e-6)

def test_the_threshold_is_honoured() -> None:
  '''The threshold decides what counts as a detection.'''
  labels, scores = batch(seed=2)
  metric = SedF1ScoreClip(threshold=0.8)
  metric.update_state(labels, scores)
  assert float(metric.result()) == pytest.approx(
      sklearn_score(labels, scores, 0.8), abs=1e-6)

def test_a_background_row_that_fires_nothing_scores_one() -> None:
  '''The background branch: nothing fires, so the row is perfect.'''
  labels = np.zeros((1, CLASSES), dtype='float32')
  scores = np.zeros((1, CLASSES), dtype='float32')
  metric = SedF1ScoreClip()
  metric.update_state(labels, scores)
  assert float(metric.result()) == pytest.approx(1.0, abs=1e-6)

def test_a_background_row_that_fires_scores_zero() -> None:
  '''The background branch: a false alarm costs the whole item.'''
  labels = np.zeros((1, CLASSES), dtype='float32')
  scores = np.zeros((1, CLASSES), dtype='float32')
  scores[0, 3] = 0.9
  metric = SedF1ScoreClip()
  metric.update_state(labels, scores)
  assert float(metric.result()) == pytest.approx(0.0, abs=1e-6)

def test_the_background_branch_and_the_normal_rows_are_mixed() -> None:
  '''A background row and a normal row are scored side by side.'''
  labels, scores = batch(seed=3, rows=2)
  labels[0] = 0.0
  scores[0] = 0.0
  metric = SedF1ScoreClip()
  metric.update_state(labels, scores)
  expected = (1.0 + sklearn_score(labels[1:], scores[1:])) / 2
  assert float(metric.result()) == pytest.approx(expected, abs=1e-6)

def test_the_three_dimensional_input_is_reduced() -> None:
  '''The clip head is max-pooled over the snippet axis.'''
  labels, scores = batch(seed=4, rows=2)
  labels_3d = np.stack([labels, labels], axis=1)
  scores_3d = np.stack([scores * 0.5, scores], axis=1)
  metric = SedF1ScoreClip()
  metric.update_state(labels_3d, scores_3d)
  assert float(metric.result()) == pytest.approx(
      sklearn_score(labels, scores), abs=1e-6)

def test_no_sigmoid_is_applied() -> None:
  '''The metric sees probabilities, and does not sigmoid them again.'''
  labels = np.zeros((1, CLASSES), dtype='float32')
  scores = np.full((1, CLASSES), 0.4, dtype='float32')
  metric = SedF1ScoreClip()
  metric.update_state(labels, scores)
  assert float(metric.result()) == pytest.approx(1.0, abs=1e-6)

def test_the_metric_resets() -> None:
  '''The accumulator can be zeroed.'''
  labels, scores = batch()
  metric = SedF1ScoreClip()
  metric.update_state(labels, scores)
  metric.reset_state()
  assert float(metric.result()) == 0.0

def test_the_metric_survives_a_save_and_load_round_trip() -> None:
  '''The configuration of the metric can be read back.'''
  metric = SedF1ScoreClip(threshold=0.3)
  config = tf.keras.utils.serialize_keras_object(metric)
  assert config['class_name'] == 'SedF1ScoreClip'
  assert config['config']['threshold'] == 0.3
  restored = tf.keras.utils.deserialize_keras_object(config)
  assert isinstance(restored, SedF1ScoreClip)
  assert restored.threshold == 0.3
  labels, scores = batch()
  restored.update_state(labels, scores)
  assert float(restored.result()) == pytest.approx(
      sklearn_score(labels, scores, 0.3), abs=1e-6)
