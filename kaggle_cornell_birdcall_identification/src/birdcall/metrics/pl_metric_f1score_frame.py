#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The frame gated micro F1, from ``sed_f1score_frame.py``.

Per item the prediction is the **AND** of two independent gates:

    pf     = y_pred['framewise_output'].max(axis=1)[0].max(axis=1)[0]
    pc     = y_pred['clipwise_output'].max(axis=1)[0]
    target = y['all_labels'][:, 0, :]
    decision = (pf >= threshold) & (pc >= 0.3)

The frame head is max-pooled over the snippets and then over the whole
time axis, so the metric is a file level metric despite its name.

The ``0.3`` is a **hard coded literal**, not a config value: it is the
in-repo ancestor of the winning ensemble rule, where a species is
accepted when the clip score and the frame score are both at least 0.3.
The two gates stay independent.

A row with no positive label is a **background** row.  It is scored by
the background branch: 1.0 when the decision fires nowhere and 0.0 when
it fires somewhere.

The metric **reduces then divides**: it accumulates a sum and a count
and returns their quotient, so the number is the mean of the per-item
scores and not a pooled micro F1.

**No sigmoid is applied** - the ``AttBlock`` already applied it.
'''

from sklearn.metrics import f1_score

CLIP_GATE = 0.3


class SedF1ScoreFrame:
  '''The epoch-wise accumulator of the frame gated micro F1.

  Attributes:
    threshold: The score the frame head must reach to count as a
      detection.
    output_transform: The transform applied to the output before it is
      scored.  The default takes the output as it is.
  '''

  def __init__(self, threshold: float = 0.5,
               output_transform=lambda x: x) -> None:
    '''Builds the metric.

    Args:
      threshold: The score the frame head must reach to count as a
        detection.
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
    pred_frame = y_pred['framewise_output'].max(axis=1)[0].max(axis=1)[0]
    pred_frame = pred_frame.detach().cpu().numpy()
    pred_clip = y_pred['clipwise_output'].max(axis=1)[0].detach().cpu().numpy()
    target = y['all_labels'][:, 0, :].detach().cpu().numpy()
    for pf, pc, t in zip(pred_frame, pred_clip, target):
      decision = (pf >= self.threshold) & (pc >= CLIP_GATE)
      if t.sum() == 0.0:
        score = 0.0 if decision.any() else 1.0
      else:
        score = float(f1_score([t], [decision], average='micro'))
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
