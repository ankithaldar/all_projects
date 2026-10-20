#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The keep-best-N checkpoint callback.

Keras 3's ``ModelCheckpoint`` has **no** ``save_top_k``.  The original
kept the best five checkpoints, ranked by the tracked metric, and
deleted the rest; that behaviour is hand written here.

The score is written into the file name, so the ranking can be recovered
from the directory alone, exactly like the original's ignite
``Checkpoint`` did.  The name holds twelve decimals, which is more than
a float32 metric needs.
'''


import tensorflow as tf
import logging
import re
from pathlib import Path


LOGGER = logging.getLogger(__name__)

SCORE_PATTERN = re.compile(r'_(-?\d+\.\d+)\.keras$')
'''The pattern of the score in a checkpoint file name.'''

class KeepBestNCallback(tf.keras.callbacks.Callback):
  '''Keeps the best N checkpoints, ranked by the monitored metric.

  Attributes:
    save_dir: The directory the checkpoints are written to.
    n_saved: How many checkpoints are kept.
    monitor: The name of the metric the checkpoints are ranked by.
    monitor_prefix: The prefix of the validation metrics in the epoch
      logs.  The checkpoints are ranked by the validation metric, which
      only exists on the epochs a validation ran on.
    mode: ``'max'`` to rank the highest score first, ``'min'`` for the
      lowest.
    prefix_name: The filename prefix of a checkpoint.
    best_score: The best score seen so far, or ``None``.
  '''

  def __init__(self, save_dir, n_saved: int = 5, monitor: str = 'f1score_clip',
               mode: str = 'max', prefix_name: str = 'checkpoint',
               monitor_prefix: str = 'val_', save_fn=None) -> None:
    '''Builds the callback.

    Args:
      save_dir: The directory the checkpoints are written to.
      n_saved: How many checkpoints are kept.
      monitor: The name of the metric the checkpoints are ranked by.
      mode: ``'max'`` or ``'min'``.
      prefix_name: The filename prefix of a checkpoint.
      monitor_prefix: The prefix of the validation metrics in the epoch
        logs.
      save_fn: The callable that writes one checkpoint.  It is called
        with the path.  The default writes the model itself.

    Raises:
      ValueError: If the mode is not known, or the number of kept
        checkpoints is not positive.
    '''
    super().__init__()
    if mode not in ('max', 'min'):
      raise ValueError(f'Unknown mode: {mode!r}')
    if n_saved <= 0:
      raise ValueError('n_saved must be positive')
    self.save_dir = Path(save_dir)
    self.n_saved = int(n_saved)
    self.monitor = monitor
    self.monitor_prefix = monitor_prefix
    self.mode = mode
    self.prefix_name = prefix_name
    self.save_fn = save_fn
    self.best_score = None

  def _is_better(self, score: float) -> bool:
    '''Whether a score beats the best one seen so far.

    Args:
      score: The score of the candidate.

    Returns:
      True when the candidate is better.
    '''
    if self.best_score is None:
      return True
    if self.mode == 'max':
      return score > self.best_score
    return score < self.best_score

  def _save(self, path: Path) -> None:
    '''Writes one checkpoint.

    Args:
      path: The path of the checkpoint.
    '''
    self.save_dir.mkdir(parents=True, exist_ok=True)
    if self.save_fn is not None:
      self.save_fn(path)
      return
    self.model.save(path)

  def _checkpoints(self) -> list[tuple[float, Path]]:
    '''Lists the checkpoints of the directory, with their scores.

    Returns:
      A list of ``(score, path)`` pairs.
    '''
    found = []
    for path in self.save_dir.glob(f'{self.prefix_name}_*.keras'):
      match = SCORE_PATTERN.search(path.name)
      if match is None:
        continue
      found.append((float(match.group(1)), path))
    return found

  def prune(self) -> list[Path]:
    '''Deletes the checkpoints that are not among the best N.

    Returns:
      The paths that were deleted.
    '''
    checkpoints = self._checkpoints()
    reverse = self.mode == 'max'
    checkpoints.sort(key=lambda item: item[0], reverse=reverse)
    deleted = []
    for _, path in checkpoints[self.n_saved:]:
      try:
        path.unlink()
        deleted.append(path)
      except OSError as error:
        LOGGER.warning('Could not delete %s: %s', path, error)
    return deleted

  def on_epoch_end(self, epoch, logs=None) -> None:
    '''Writes a checkpoint when the monitored metric improves.

    The monitored metric is looked up under the validation prefix first:
    the validation metrics are the ones the checkpoints are ranked by,
    and they only exist on the epochs a validation ran on.

    Args:
      epoch: The index of the epoch.
      logs: The epoch logs.
    '''
    del epoch
    logs = logs or {}
    value = logs.get(f'{self.monitor_prefix}{self.monitor}')
    if value is None:
      value = logs.get(self.monitor)
    if value is None:
      LOGGER.warning('The monitored metric %r is not in the logs; no '
                     'checkpoint is written.', self.monitor)
      return
    score = float(value)
    if not self._is_better(score):
      return
    self.best_score = score
    self._save(self.save_dir /
               f'{self.prefix_name}_{score:.12f}.keras')
    self.prune()
