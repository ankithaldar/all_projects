#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The label ranking average precision score, as a Keras metric.

There is **no label ranking metric of any kind** in Keras.
``keras.metrics.AUC(multi_label=True)`` is not one either: it is
query independent, and the label ranking average precision is query
dependent.  The score is therefore reimplemented here, vectorised.

The score of one item is sklearn's
``label_ranking_average_precision_score`` of the clip head against the
labels of the first snippet:

    pred   = y_pred['clipwise_output'].max(axis=1)   (max over snippets)
    target = y['all_labels'][:, 0, :]                 (snippet 0's labels)

sklearn's own formulation, which this module reproduces exactly, is the
mean over the relevant labels of

    L_i / rank_i

where ``rank_i`` is the number of labels whose score is at least
``score_i``, and ``L_i`` is the number of **relevant** labels whose score
is at least ``score_i``.  Ties share the largest of their ranks, which
is what ``scipy.stats.rankdata(..., 'max')`` does.

Two degenerate cases are inherited from sklearn: a target with no
positive label and a target with **every** label positive both score
**1.0**, because the ranking carries no information.

The metric **reduces then divides**: it accumulates a sum and a count
and returns their quotient, so the number is the mean of the per-item
scores and not a pooled score.

**No sigmoid is applied** -- the ``AttBlock`` already applied it.
'''


import tensorflow as tf


FIRE_THRESHOLD = 0.5
'''The score a class must reach to count as a fire, for the background
branch.'''

def lrap_scores(y_true, y_score):
  '''Computes the label ranking average precision of every item.

  Args:
    y_true: The labels, shape ``(batch, classes)``, zero or one.
    y_score: The scores, shape ``(batch, classes)``.

  Returns:
    A ``(batch,)`` tensor of per-item scores.
  '''
  relevant = tf.cast(y_true > 0.0, tf.float32)
  num_relevant = tf.reduce_sum(relevant, axis=-1)
  # The descending order of the scores.  The sort is stable, so equal
  # scores keep their order, and the tie groups are found below.
  order = tf.argsort(-y_score, axis=-1, stable=True)
  sorted_score = tf.gather(y_score, order, batch_dims=1)
  sorted_relevant = tf.gather(relevant, order, batch_dims=1)
  classes = tf.shape(sorted_score)[1]
  # The last position of every group of equal scores.
  is_end = tf.concat(
      [sorted_score[:, 1:] != sorted_score[:, :-1],
       tf.ones((tf.shape(sorted_score)[0], 1), tf.bool)], axis=1)
  positions = tf.range(classes)
  candidates = tf.where(is_end, positions, tf.fill(tf.shape(positions),
                                                   classes))
  # The first group end at or after each position, found by a reversed
  # fold.  TensorFlow has no cumulative minimum, and `tf.scan` folds over
  # the first axis, so the array is transposed around it.
  group_end = tf.transpose(
      tf.scan(tf.minimum, tf.transpose(candidates), reverse=True))
  cumulative_relevant = tf.cumsum(sorted_relevant, axis=-1)
  relevant_ge = tf.gather(cumulative_relevant, group_end, batch_dims=1)
  rank_ge = tf.cast(group_end, tf.float32) + 1.0
  # Back to the original order.
  inverse = tf.argsort(order, axis=-1)
  relevant_count = tf.gather(relevant_ge, inverse, batch_dims=1)
  rank = tf.gather(rank_ge, inverse, batch_dims=1)
  ratio = relevant_count / rank
  total = tf.reduce_sum(tf.where(relevant > 0.0, ratio, 0.0), axis=-1)
  scores = total / tf.maximum(num_relevant, 1.0)
  degenerate = (num_relevant == 0.0) | (num_relevant ==
                                        tf.cast(classes, tf.float32))
  return tf.where(degenerate, tf.ones_like(scores), scores)

@tf.keras.utils.register_keras_serializable(package='birdcall', name='SedLRAPS')
class SedLRAPS(tf.keras.metrics.Metric):
  '''The epoch-wise accumulator of the label ranking average precision.

  Attributes:
    output_transform: An optional callable applied to the
      ``(y_true, y_pred)`` pair before it is scored.  The default scores
      them as they are.
    with_background_branch: Whether a label-less row is scored by the
      background rule instead of by sklearn's free 1.0.
  '''

  def __init__(self, output_transform=None,
               with_background_branch: bool = False,
               name: str = 'lraps', dtype=None) -> None:
    '''Builds the metric.

    Args:
      output_transform: An optional callable applied to the
        ``(y_true, y_pred)`` pair before it is scored.
      with_background_branch: Whether a label-less row is scored by the
        background rule instead of by sklearn's free 1.0.
      name: The metric name.
      dtype: The dtype of the metric.
    '''
    super().__init__(name=name, dtype=dtype)
    self.output_transform = output_transform
    self.with_background_branch = with_background_branch
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
    values = lrap_scores(labels, scores)
    if self.with_background_branch:
      background = tf.reduce_sum(labels, axis=-1) == 0.0
      fires = tf.reduce_any(scores >= FIRE_THRESHOLD, axis=-1)
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
    config.update({'with_background_branch': self.with_background_branch})
    return config
