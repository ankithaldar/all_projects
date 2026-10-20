#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the console logger.

The console logger writes the same series as the tracker, so a
transcript of the original and of the port can be read side by side.
'''

from pathlib import Path

import pytest

from birdcall.config_params.checkpoint_params import CheckpointParams
from birdcall.config_params.dataset_params import DatasetParams
from birdcall.config_params.logger_params import LoggerParams
from birdcall.config_params.parameters import Parameters
from birdcall.loggers.pl_logger_print import PrintLoggerAdapter


def _parameters(**overrides) -> Parameters:
  '''Builds a validated configuration for the tests.

  Args:
    **overrides: The attributes that differ from the default.

  Returns:
    The configuration.
  '''
  dataset = DatasetParams(root_dir=Path('/tmp'),
                          csv_dir=Path('/tmp/fold.csv'),
                          period=30,
                          isTraining=True,
                          num_test_samples=1)
  payload = {
      'train_ds_params': dataset,
      'valid_ds_params': dataset,
      'test_ds_params': dataset,
      'logger_name': 'print',
      'logger_params': LoggerParams(name='run', log_every=3),
      'checkpoint_params': CheckpointParams(save_dir=Path('/tmp/ckpt')),
  }
  payload.update(overrides)
  return Parameters(**payload)


def test_the_series_are_written_to_stdout(capsys):
  '''The metrics are printed under the original's series names.'''
  logger = PrintLoggerAdapter(_parameters())
  logger.log_metrics({'train_loss': 0.5, 'valid_f1score_clip': 0.75}, step=2)
  printed = capsys.readouterr().out
  assert 'train/loss 2 0.5' in printed
  assert 'valid/f1score_clip 2 0.75' in printed


def test_every_series_is_recorded():
  '''The logger keeps the lines it wrote.'''
  logger = PrintLoggerAdapter(_parameters())
  logger.log_metrics({'train_loss': 0.5}, step=1)
  assert logger.lines == ['train/loss 1 0.5']


def test_the_hyper_parameters_are_written(capsys):
  '''The configuration of the run is printed too.'''
  logger = PrintLoggerAdapter(_parameters())
  logger.log_hyperparams({'lr': 0.001})
  printed = capsys.readouterr().out
  assert 'config/lr 0.001' in printed


def test_the_cadence_comes_from_the_configuration():
  '''The logger carries the cadence of the per-iteration series.'''
  assert PrintLoggerAdapter(_parameters()).log_every == 3


def test_the_logger_has_no_experiment():
  '''The console logger writes to stdout, not to a tracker.'''
  logger = PrintLoggerAdapter(_parameters())
  assert logger.experiment is None
  assert logger.save_dir is None
  assert logger.name == 'run'
  assert logger.version is None


def test_the_logger_is_a_lightning_logger():
  '''The console logger can be handed to a trainer.'''
  from lightning.pytorch.loggers import Logger
  assert isinstance(PrintLoggerAdapter(_parameters()), Logger)


def test_nothing_is_printed_when_there_is_nothing_to_print(capsys):
  '''An empty metric dict writes nothing.'''
  logger = PrintLoggerAdapter(_parameters())
  logger.log_metrics({}, step=1)
  assert capsys.readouterr().out == ''
  assert logger.lines == []
  with pytest.raises(AssertionError):
    assert logger.lines == ['train/loss 1 0.5']
