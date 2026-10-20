#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The clip level micro F1, as a Keras metric.

Per item:

    pred   = y_pred['clipwise_output'].max(axis=1)   (max over snippets)
    target = y['all_labels'][:, 0, :]                 (snippet 0's labels)

A row with no positive label is a **background** row.  It is scored by
the background branch: 1.0 when nothing fires above the threshold and
0.0 when something does, so a false alarm on a background row costs a
full item.  Every other row is scored by the micro F1 over the 264
classes.

The metric **reduces then divides**: it accumulates a sum and a count
and returns their quotient, so the number is the mean of the per-item
scores and not a pooled micro F1.

``keras.metrics.F1Score`` is not used: its ``threshold`` defaults to
``None``, which means multiclass argmax, and it is categorically wrong
for a 264-label multi-label problem.  The F1 is therefore written out.

**No sigmoid is applied** -- the ``AttBlock`` already applied it.
'''


import tensorflow as tf


def micro_f1(labels, predictions):
  '''Computes the micro F1 of every item.

  Args:
    labels: The labels, shape ``(batch, classes)``.
    predictions: The binary predictions, shape ``(batch, classes)``.

  Returns:
    A ``(batch,)`` tensor of per-item micro F1 scores.
  '''
  true = tf.cast(labels > 0.0, tf.float32)
  pred = tf.cast(predictions, tf.float32)
  true_positive = tf.reduce_sum(true * pred, axis=-1)
  false_positive = tf.reduce_sum((1.0 - true) * pred, axis=-1)
  false_negative = tf.reduce_sum(true * (1.0 - pred), axis=-1)
  denominator = 2.0 * true_positive + false_positive + false_negative
  return tf.where(denominator > 0.0,
                  2.0 * true_positive / denominator, tf.zeros_like(denominator))

@tf.keras.utils.register_keras_serializable(package='birdcall',
                                           name='SedF1ScoreClip')
class SedF1ScoreClip(tf.keras.metrics.Metric):
  '''The epoch-wise accumulator of the clip level micro F1.

  Attributes:
    threshold: The score a class must reach to count as a detection.
    output_transform: An optional callable applied to the
      ``(y_true, y_pred)`` pair before it is scored.  The default scores
      them as they are.
  '''

  def __init__(self, threshold: float = 0.5, output_transform=None,
               name: str = 'f1score_clip', dtype=None) -> None:
    '''Builds the metric.

    Args:
      threshold: The score a class must reach to count as a detection.
      output_transform: An optional callable applied to the
        ``(y_true, y_pred)`` pair before it is scored.
      name: The metric name.
      dtype: The dtype of the metric.
    '''
    super().__init__(name=name, dtype=dtype)
    self.threshold = threshold
    self.output_transform = output_transform
    self._score = self.add_variable(shape=(), initializer='zeros',
                                     name='score')
    self._count = self.add_variable(shape=(), initializer='zeros',
                                     name='count')

  def _prepare(self, y_true, y_pred):
    '''Reduces the labels and the prediction to one row per item.

    Args:
      y_true: The label dict, or the labels themselves.
      y_pred: The model output dict, or the prediction itself.

    Returns:
      A tuple ``(labels, scores)``, each of shape ``(batch, classes)``.
    '''
    if isinstance(y_true, dict):
      y_true = y_true['all_labels']
    if isinstance(y_pred, dict):
      y_pred = y_pred['clipwise_output']
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    if len(y_true.shape) == 3:
      y_true = y_true[:, 0, :]
    if len(y_pred.shape) == 4:
      y_pred = tf.reduce_max(y_pred, axis=1)
      y_pred = tf.reduce_max(y_pred, axis=1)
    elif len(y_pred.shape) == 3:
      y_pred = tf.reduce_max(y_pred, axis=1)
    return y_true, y_pred

  def update_state(self, y_true, y_pred, sample_weight=None) -> None:
    '''Adds one batch to the accumulator.

    Args:
      y_true: The label dict, or the labels themselves.
      y_pred: The model output dict, or the prediction itself.
      sample_weight: Unused.  The metric is a mean of per-item scores.
    '''
    del sample_weight
    if self.output_transform is not None:
      y_true, y_pred = self.output_transform((y_true, y_pred))
    labels, scores = self._prepare(y_true, y_pred)
    detections = scores >= self.threshold
    values = micro_f1(labels, detections)
    background = tf.reduce_sum(labels, axis=-1) == 0.0
    fires = tf.reduce_any(detections, axis=-1)
    values = tf.where(background, tf.where(fires, 0.0, 1.0), values)
    self._score.assign_add(tf.reduce_sum(values))
    self._count.assign_add(tf.cast(tf.shape(values)[0], self._count.dtype))

  def result(self):
    '''Returns the mean of the per-item scores.

    Returns:
      The accumulated sum divided by the number of items, or 0.0 when
      nothing was accumulated.
    '''
    return tf.where(self._count > 0.0, self._score / self._count,
                    tf.zeros_like(self._score))

  def reset_state(self) -> None:
    '''Zeroes the accumulator.'''
    self._score.assign(0.0)
    self._count.assign(0.0)

  def get_config(self) -> dict:
    '''Returns the serialisable configuration of the metric.

    Returns:
      The configuration dictionary.
    '''
    config = super().get_config()
    config.update({'threshold': self.threshold})
    return config
