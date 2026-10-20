#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The label ranking average precision score, from ``sed_lraps.py``.

Per item the score is sklearn's ``label_ranking_average_precision_score``
of the clip head against the labels of the first snippet:

    pred   = y_pred['clipwise_output'].max(axis=1)[0]   (max over snippets)
    target = y['all_labels'][:, 0, :]
    score  = label_ranking_average_precision_score([target], [pred])

The metric **reduces then divides**: it accumulates a sum and a count
and returns their quotient, so the number is the mean of the per-item
scores and not a pooled score.

Two details of the original are reproduced on purpose:

* **no sigmoid is applied** - the ``AttBlock`` already applied it;
* **there is no background branch** - sklearn scores a label-less row as
  a perfect ranking, so every background item contributes a free 1.0.
  The optional ``with_background_branch`` turns that into the same rule
  the two F1 metrics use: a label-less row scores 0.0 when something
  fires and 1.0 when nothing does.
'''

from sklearn.metrics import label_ranking_average_precision_score

FIRE_THRESHOLD = 0.5


class SedLRAPS:
  '''The epoch-wise accumulator of the label ranking average precision.

  Attributes:
    output_transform: The transform applied to the output before it is
      scored.  The default takes the output as it is.
    with_background_branch: Whether a label-less row is scored by the
      background rule instead of by sklearn's free 1.0.
  '''

  def __init__(self, output_transform=lambda x: x,
               with_background_branch: bool = False) -> None:
    '''Builds the metric.

    Args:
      output_transform: The transform applied to the output before it is
        scored.  The default takes the output as it is.
      with_background_branch: Whether a label-less row is scored by the
        background rule instead of by sklearn's free 1.0.
    '''
    self.output_transform = output_transform
    self.with_background_branch = with_background_branch
    self.reset()

  def reset(self) -> None:
    '''Zeroes the accumulator.'''
    self._score = 0.0
    self._num_items = 0

  def update(self, output: dict) -> None:
    '''Adds one batch to the accumulator.

    Args:
      output: The dict with ``y_pred`` (the clip and the frame heads) and
        ``y`` (the label dict).
    '''
    output = self.output_transform(output)
    y_pred = output['y_pred']
    y = output['y']
    pred = y_pred['clipwise_output'].max(axis=1)[0].detach().cpu().numpy()
    target = y['all_labels'][:, 0, :].detach().cpu().numpy()
    for p, t in zip(pred, target):
      if self.with_background_branch and t.sum() == 0.0:
        score = 0.0 if (p >= FIRE_THRESHOLD).any() else 1.0
      else:
        score = float(label_ranking_average_precision_score([t], [p]))
      self._score += score
      self._num_items += 1

  def compute(self) -> float:
    '''Returns the mean of the per-item scores.

    Returns:
      The accumulated sum divided by the number of items, or 0.0 when
      nothing was accumulated.
    '''
    if self._num_items == 0:
      return 0.0
    return self._score / self._num_items
