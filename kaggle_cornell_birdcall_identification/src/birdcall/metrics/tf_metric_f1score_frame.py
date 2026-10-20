#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The frame gated micro F1, as a Keras metric.

Per item the prediction is the **AND** of two independent gates:

    pf     = y_pred['framewise_output'].max(axis=1).max(axis=1)
    pc     = y_pred['clipwise_output'].max(axis=1)
    target = y['all_labels'][:, 0, :]
    decision = (pf >= threshold) & (pc >= 0.3)

The frame head is max-pooled over the snippets and then over the whole
time axis, so the metric is a file level metric despite its name.

The ``0.3`` is a **hard coded literal**, not a config value: it is the
in-repo ancestor of the winning ensemble rule, where a species is
accepted when the clip score and the frame score are both at least 0.3.
The two gates stay **independent**: a frame that passes the frame
threshold but whose clip score is below 0.3 contributes nothing.

A row with no positive label is a **background** row.  It is scored by
the background branch: 1.0 when the decision fires nowhere and 0.0 when
it fires somewhere.

The metric **reduces then divides**: it accumulates a sum and a count
and returns their quotient, so the number is the mean of the per-item
scores and not a pooled micro F1.

**No sigmoid is applied** -- the ``AttBlock`` already applied it.
'''


import tensorflow as tf

from birdcall.metrics.tf_metric_f1score_clip import micro_f1


CLIP_GATE = 0.3
'''The hard coded clip gate of the frame metric.'''

@tf.keras.utils.register_keras_serializable(package='birdcall',
                                     name='SedF1ScoreFrame')
class SedF1ScoreFrame(tf.keras.metrics.Metric):
  '''The epoch-wise accumulator of the frame gated micro F1.

  Attributes:
    threshold: The score the frame head must reach to count as a
      detection.
    output_transform: An optional callable applied to the
      ``(y_true, y_pred)`` pair before it is scored.  The default scores
      them as they are.
  '''

  def __init__(self, threshold: float = 0.5, output_transform=None,
               name: str = 'f1score_frame', dtype=None) -> None:
    '''Builds the metric.

    Args:
      threshold: The score the frame head must reach to count as a
        detection.
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
    '''Reduces the labels and the two heads to one row per item.

    Args:
      y_true: The label dict, or the labels themselves.
      y_pred: The model output dict, or the two heads themselves.

    Returns:
      A tuple ``(labels, frame scores, clip scores)``, each of shape
      ``(batch, classes)``.
    '''
    if isinstance(y_true, dict):
      y_true = y_true['all_labels']
    y_true = tf.convert_to_tensor(y_true)
    if len(y_true.shape) == 3:
      y_true = y_true[:, 0, :]
    if isinstance(y_pred, dict):
      frame = tf.convert_to_tensor(y_pred['framewise_output'])
      clip = tf.convert_to_tensor(y_pred['clipwise_output'])
    else:
      frame = tf.convert_to_tensor(y_pred)
      clip = frame
    if len(frame.shape) == 4:
      frame = tf.reduce_max(frame, axis=1)
      frame = tf.reduce_max(frame, axis=1)
    elif len(frame.shape) == 3:
      frame = tf.reduce_max(frame, axis=1)
    if len(clip.shape) == 4:
      clip = tf.reduce_max(clip, axis=1)
      clip = tf.reduce_max(clip, axis=1)
    elif len(clip.shape) == 3:
      clip = tf.reduce_max(clip, axis=1)
    return y_true, frame, clip

  def update_state(self, y_true, y_pred, sample_weight=None) -> None:
    '''Adds one batch to the accumulator.

    Args:
      y_true: The label dict, or the labels themselves.
      y_pred: The model output dict, or the two heads themselves.
      sample_weight: Unused.  The metric is a mean of per-item scores.
    '''
    del sample_weight
    if self.output_transform is not None:
      y_true, y_pred = self.output_transform((y_true, y_pred))
    labels, frame, clip = self._prepare(y_true, y_pred)
    # The two gates are independent.
    decision = (frame >= self.threshold) & (clip >= CLIP_GATE)
    values = micro_f1(labels, decision)
    background = tf.reduce_sum(labels, axis=-1) == 0.0
    fires = tf.reduce_any(decision, axis=-1)
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
