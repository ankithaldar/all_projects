#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The tests of the keep-best-N checkpoint callback.

Keras 3's ``ModelCheckpoint`` has no ``save_top_k``, so the behaviour of
the original's ignite ``Checkpoint`` is hand written here and is tested
as such.
'''

from pathlib import Path

import pytest

from birdcall.callbacks.tf_callback_keep_best_n import KeepBestNCallback


def write(path: Path, score: float) -> None:
  '''Writes a checkpoint file with a score in its name.

  Args:
    path: The path of the file.
    score: The score the file stands for.
  '''
  path.write_text(f'{score:.12f}', encoding='utf-8')


def callback(tmp_path: Path, n_saved: int = 3, mode: str = 'max'):
  '''Builds the callback of the test.

  Args:
    tmp_path: The directory the checkpoints go to.
    n_saved: How many checkpoints are kept.
    mode: The ranking mode.

  Returns:
    The callback.
  '''
  def save_fn(path):
    write(path, float(path.stem.split('_')[-1]))

  return KeepBestNCallback(save_dir=tmp_path, n_saved=n_saved,
                           monitor='f1score_clip', mode=mode,
                           prefix_name='checkpoint', save_fn=save_fn)


def test_nothing_is_written_without_the_monitored_metric(tmp_path) -> None:
  '''A monitor that is not logged never fires.'''
  keeper = callback(tmp_path)
  keeper.on_epoch_end(0, {'loss': 0.5})
  assert list(tmp_path.glob('*.keras')) == []


def test_a_checkpoint_is_written_when_the_metric_improves(tmp_path) -> None:
  '''The first epoch always writes a checkpoint.'''
  keeper = callback(tmp_path)
  keeper.on_epoch_end(0, {'f1score_clip': 0.5})
  assert len(list(tmp_path.glob('*.keras'))) == 1
  assert keeper.best_score == pytest.approx(0.5)


def test_a_worse_metric_writes_nothing(tmp_path) -> None:
  '''A worse score does not replace the best one.'''
  keeper = callback(tmp_path)
  keeper.on_epoch_end(0, {'f1score_clip': 0.5})
  keeper.on_epoch_end(1, {'f1score_clip': 0.4})
  assert len(list(tmp_path.glob('*.keras'))) == 1
  assert keeper.best_score == pytest.approx(0.5)


def test_only_the_best_n_files_are_kept(tmp_path) -> None:
  '''The rest of the checkpoints are deleted.'''
  keeper = callback(tmp_path, n_saved=3)
  for epoch, score in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
    keeper.on_epoch_end(epoch, {'f1score_clip': score})
  files = sorted(tmp_path.glob('*.keras'))
  # Six checkpoints were written, and the three best are kept.
  assert len(files) == 3
  kept = sorted(float(path.stem.split('_')[-1]) for path in files)
  assert kept == pytest.approx([0.4, 0.5, 0.6], abs=1e-9)


def test_only_the_improvements_are_written(tmp_path) -> None:
  '''A score that does not beat the best one writes nothing.'''
  keeper = callback(tmp_path, n_saved=5)
  for epoch, score in enumerate([0.1, 0.5, 0.3, 0.9, 0.7, 0.8]):
    keeper.on_epoch_end(epoch, {'f1score_clip': score})
  files = sorted(tmp_path.glob('*.keras'))
  assert len(files) == 3
  kept = sorted(float(path.stem.split('_')[-1]) for path in files)
  assert kept == pytest.approx([0.1, 0.5, 0.9], abs=1e-9)


def test_the_minimising_mode_keeps_the_smallest(tmp_path) -> None:
  '''The ranking follows the mode.'''
  keeper = callback(tmp_path, n_saved=2, mode='min')
  for epoch, score in enumerate([0.5, 0.2, 0.9, 0.1]):
    keeper.on_epoch_end(epoch, {'f1score_clip': score})
  files = sorted(tmp_path.glob('*.keras'))
  assert len(files) == 2
  kept = sorted(float(path.stem.split('_')[-1]) for path in files)
  assert kept == pytest.approx([0.1, 0.2], abs=1e-9)


def test_the_save_function_is_used(tmp_path) -> None:
  '''The checkpoint can be written by a caller.'''
  written = []

  def save_fn(path):
    written.append(path)
    write(path, 0.5)

  keeper = KeepBestNCallback(save_dir=tmp_path, n_saved=1,
                              monitor='f1score_clip', save_fn=save_fn)
  keeper.on_epoch_end(0, {'f1score_clip': 0.5})
  assert len(written) == 1
  assert written[0].exists()


def test_the_pruning_removes_the_extra_files(tmp_path) -> None:
  '''The pruning can be run on its own.'''
  for score in [0.1, 0.9, 0.5]:
    write(tmp_path / f'checkpoint_{score:.12f}.keras', score)
  keeper = callback(tmp_path, n_saved=1)
  deleted = keeper.prune()
  assert len(deleted) == 2
  assert len(list(tmp_path.glob('*.keras'))) == 1


def test_the_files_of_another_prefix_are_kept(tmp_path) -> None:
  '''The pruning only touches the files of its own prefix.'''
  write(tmp_path / 'other_0.500000000000.keras', 0.5)
  write(tmp_path / 'checkpoint_0.100000000000.keras', 0.1)
  write(tmp_path / 'checkpoint_0.900000000000.keras', 0.9)
  keeper = callback(tmp_path, n_saved=1)
  keeper.prune()
  assert (tmp_path / 'other_0.500000000000.keras').exists()
  assert len(list(tmp_path.glob('checkpoint_*.keras'))) == 1


def test_an_unknown_mode_is_refused(tmp_path) -> None:
  '''An unknown ranking mode raises.'''
  with pytest.raises(ValueError, match='Unknown mode'):
    callback(tmp_path, mode='median')


def test_a_non_positive_number_of_checkpoints_is_refused(tmp_path) -> None:
  '''Keeping no checkpoint at all is refused.'''
  with pytest.raises(ValueError, match='n_saved'):
    callback(tmp_path, n_saved=0)
