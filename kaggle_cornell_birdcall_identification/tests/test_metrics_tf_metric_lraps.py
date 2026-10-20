#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The tests of the label ranking average precision metric.

Keras has no label ranking metric of any kind, so the score is
reimplemented here, and it is checked against sklearn.
'''


import tensorflow as tf
import numpy as np
import pytest
from sklearn.metrics import label_ranking_average_precision_score

from birdcall.metrics.tf_metric_lraps import SedLRAPS


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

def sklearn_score(labels: np.ndarray, scores: np.ndarray) -> float:
  '''Computes the mean label ranking score of a batch with sklearn.

  Args:
    labels: The labels.
    scores: The scores.

  Returns:
    The mean of the per-item scores.
  '''
  return float(np.mean([
      label_ranking_average_precision_score([label], [score])
      for label, score in zip(labels, scores)]))

def test_the_score_matches_sklearn() -> None:
  '''The score is sklearn's, to 1e-6.'''
  labels, scores = batch()
  metric = SedLRAPS()
  metric.update_state(labels, scores)
  assert float(metric.result()) == pytest.approx(sklearn_score(labels, scores),
                                                 abs=1e-6)

def test_the_score_matches_sklearn_with_ties() -> None:
  '''Tied scores share the largest of their ranks.'''
  labels, scores = batch(seed=1)
  scores = np.round(scores, 1)
  metric = SedLRAPS()
  metric.update_state(labels, scores)
  assert float(metric.result()) == pytest.approx(sklearn_score(labels, scores),
                                                 abs=1e-6)

def test_an_all_zero_target_scores_one() -> None:
  '''sklearn's degenerate clause: no positive label is a free 1.0.'''
  labels, scores = batch(seed=2)
  labels[0] = 0.0
  metric = SedLRAPS()
  metric.update_state(labels, scores)
  expected = (1.0 + sum(
      label_ranking_average_precision_score([label], [score])
      for label, score in zip(labels[1:], scores[1:]))) / len(labels)
  assert float(metric.result()) == pytest.approx(expected, abs=1e-6)

def test_an_all_one_target_scores_one() -> None:
  '''Every label relevant is a free 1.0 as well.'''
  labels, scores = batch(seed=3)
  labels[0] = 1.0
  metric = SedLRAPS()
  metric.update_state(labels, scores)
  expected = (1.0 + sum(
      label_ranking_average_precision_score([label], [score])
      for label, score in zip(labels[1:], scores[1:]))) / len(labels)
  assert float(metric.result()) == pytest.approx(expected, abs=1e-6)

def test_a_perfect_ranking_scores_one() -> None:
  '''A perfect ranking of the positives scores one.'''
  labels = np.zeros((1, CLASSES), dtype='float32')
  labels[0, 0] = 1.0
  scores = np.linspace(1.0, 0.0, CLASSES, dtype='float32')[None, :]
  metric = SedLRAPS()
  metric.update_state(labels, scores)
  assert float(metric.result()) == pytest.approx(1.0, abs=1e-6)

def test_the_metric_reduces_then_divides() -> None:
  '''The score is the mean of the per-item scores.'''
  labels, scores = batch(seed=4, rows=4)
  metric = SedLRAPS()
  metric.update_state(labels, scores)
  per_item = [label_ranking_average_precision_score([label], [score])
              for label, score in zip(labels, scores)]
  assert float(metric.result()) == pytest.approx(float(np.mean(per_item)),
                                                 abs=1e-6)

def test_the_background_branch_scores_a_firing_row_zero() -> None:
  '''With the background branch, a false alarm costs a full item.'''
  labels = np.zeros((2, CLASSES), dtype='float32')
  scores = np.zeros((2, CLASSES), dtype='float32')
  scores[0, 0] = 0.9
  metric = SedLRAPS(with_background_branch=True)
  metric.update_state(labels, scores)
  assert float(metric.result()) == pytest.approx(0.5, abs=1e-6)

def test_the_background_branch_scores_a_silent_row_one() -> None:
  '''A label-less row that fires nothing scores one.'''
  labels = np.zeros((1, CLASSES), dtype='float32')
  scores = np.zeros((1, CLASSES), dtype='float32')
  metric = SedLRAPS(with_background_branch=True)
  metric.update_state(labels, scores)
  assert float(metric.result()) == pytest.approx(1.0, abs=1e-6)

def test_the_three_dimensional_input_is_reduced() -> None:
  '''The clip head is max-pooled over the snippet axis.'''
  labels, scores = batch(seed=5, rows=2)
  # Two items of two snippets each: the labels are the first snippet's
  # labels, and the clip score is the max over the snippets.
  labels_3d = np.stack([labels, labels], axis=1)
  scores_3d = np.stack([scores * 0.5, scores], axis=1)
  metric = SedLRAPS()
  metric.update_state(labels_3d, scores_3d)
  assert float(metric.result()) == pytest.approx(
      sklearn_score(labels, scores), abs=1e-6)

def test_the_metric_resets() -> None:
  '''The accumulator can be zeroed.'''
  labels, scores = batch()
  metric = SedLRAPS()
  metric.update_state(labels, scores)
  assert float(metric.result()) > 0.0
  metric.reset_state()
  assert float(metric.result()) == 0.0

def test_the_metric_survives_a_save_and_load_round_trip() -> None:
  '''The configuration of the metric can be read back.'''
  metric = SedLRAPS(with_background_branch=True)
  config = tf.keras.utils.serialize_keras_object(metric)
  assert config['class_name'] == 'SedLRAPS'
  assert config['config']['with_background_branch'] is True
  restored = tf.keras.utils.deserialize_keras_object(config)
  assert isinstance(restored, SedLRAPS)
  assert restored.with_background_branch is True
  labels, scores = batch()
  restored.update_state(labels, scores)
  assert float(restored.result()) == pytest.approx(
      sklearn_score(labels, scores), abs=1e-6)

def test_an_empty_metric_scores_zero() -> None:
  '''A metric that saw nothing scores zero, not a division by zero.'''
  assert float(SedLRAPS().result()) == 0.0
