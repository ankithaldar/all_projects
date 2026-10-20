#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The clip level micro F1, from ``sed_f1score_clip.py``.

Per item:

    pred   = y_pred['clipwise_output'].max(axis=1)[0]   (max over snippets)
    target = y['all_labels'][:, 0, :]

A row with no positive label is a **background** row.  It is scored by
the background branch: 1.0 when nothing fires above the threshold and
0.0 when something does, so a false alarm on a background row costs a
full item.  Every other row is scored by the micro F1 over the 264
classes.

The metric **reduces then divides**: it accumulates a sum and a count
and returns their quotient, so the number is the mean of the per-item
scores and not a pooled micro F1.

**No sigmoid is applied** - the ``AttBlock`` already applied it.
'''

from sklearn.metrics import f1_score


class SedF1ScoreClip:
  '''The epoch-wise accumulator of the clip level micro F1.

  Attributes:
    threshold: The score a class must reach to count as a detection.
    output_transform: The transform applied to the output before it is
      scored.  The default takes the output as it is.
  '''

  def __init__(self, threshold: float = 0.5,
               output_transform=lambda x: x) -> None:
    '''Builds the metric.

    Args:
      threshold: The score a class must reach to count as a detection.
      output_transform: The transform applied to the output before it is
        scored.  The default takes the output as it is.
    '''
    self.threshold = threshold
    self.output_transform = output_transform
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
      if t.sum() == 0.0:
        score = 0.0 if (p >= self.threshold).any() else 1.0
      else:
        score = float(f1_score([t], [p >= self.threshold], average='micro'))
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
