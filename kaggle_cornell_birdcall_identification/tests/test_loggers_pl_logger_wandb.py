#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the wandb logger.

The tracker is wandb, but the series names are the original's Neptune
names, because a dashboard built on the original's names has to keep
reading the right numbers.  No test needs a network: the runs are
offline, or disabled altogether.
'''

from pathlib import Path

import pytest

from birdcall.config_params.checkpoint_params import CheckpointParams
from birdcall.config_params.dataset_params import DatasetParams
from birdcall.config_params.logger_params import LoggerParams
from birdcall.config_params.parameters import Parameters
from birdcall.loggers.pl_logger_wandb import WandbLoggerAdapter


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
      'logger_name': 'wandb',
      'logger_params': LoggerParams(project_name='bird-song',
                                    name='run',
                                    prefix_name='group',
                                    tags=['a', 'b'],
                                    log_every=7),
      'checkpoint_params': CheckpointParams(save_dir=Path('/tmp/ckpt')),
  }
  payload.update(overrides)
  return Parameters(**payload)


def test_the_series_names_are_the_originals():
  '''Every series keeps the name the original's tracker used.'''
  assert WandbLoggerAdapter.series_name('train_loss') == 'train/loss'
  assert WandbLoggerAdapter.series_name('train_bce_loss') == 'train/bce_loss'
  assert WandbLoggerAdapter.series_name('train_F_loss') == 'train/F_loss'
  assert WandbLoggerAdapter.series_name('lr-AdamW') == 'train/lr'
  assert WandbLoggerAdapter.series_name('valid_lraps') == 'valid/lraps'
  assert (WandbLoggerAdapter.series_name('valid_f1score_clip') ==
          'valid/f1score_clip')
  assert (WandbLoggerAdapter.series_name('valid_f1score_frame') ==
          'valid/f1score_frame')


def test_an_unknown_name_is_kept():
  '''A name the adapter does not know is written as it is.'''
  assert WandbLoggerAdapter.series_name('epoch') == 'epoch'


def test_the_logger_params_map_onto_wandb(monkeypatch, tmp_path):
  '''The six keys of the configuration keep their meaning.'''
  monkeypatch.setenv('WANDB_MODE', 'offline')
  monkeypatch.setenv('WANDB_DIR', str(tmp_path))
  monkeypatch.setenv('WANDB_SILENT', 'true')
  parameters = _parameters(
      checkpoint_params=CheckpointParams(save_dir=tmp_path / 'ckpt'))
  adapter = WandbLoggerAdapter(parameters)
  inner = adapter.inner
  assert inner is not None
  assert inner._project == 'bird-song'
  assert inner._name == 'run'
  assert inner._wandb_init['group'] == 'group'
  assert inner._wandb_init['tags'] == ['a', 'b']
  assert inner._wandb_init['config']['model_name'] == 'sed_dense121att'
  assert adapter.log_every == 7


def test_the_logger_works_offline(monkeypatch, tmp_path):
  '''An offline run writes its metrics without a network.'''
  monkeypatch.setenv('WANDB_MODE', 'offline')
  monkeypatch.setenv('WANDB_DIR', str(tmp_path))
  monkeypatch.setenv('WANDB_SILENT', 'true')
  parameters = _parameters(
      checkpoint_params=CheckpointParams(save_dir=tmp_path / 'ckpt'))
  adapter = WandbLoggerAdapter(parameters)
  adapter.log_metrics({'train_loss': 0.5, 'valid_lraps': 0.9}, step=3)
  adapter.log_hyperparams({'lr': 0.001})
  adapter.save()


def test_the_logger_works_when_wandb_is_disabled(monkeypatch, tmp_path):
  '''A disabled wandb is not an error.'''
  monkeypatch.setenv('WANDB_MODE', 'offline')
  monkeypatch.setenv('WANDB_DISABLED', 'true')
  monkeypatch.setenv('WANDB_DIR', str(tmp_path))
  monkeypatch.setenv('WANDB_SILENT', 'true')
  parameters = _parameters(
      checkpoint_params=CheckpointParams(save_dir=tmp_path / 'ckpt'))
  adapter = WandbLoggerAdapter(parameters)
  adapter.log_metrics({'train_loss': 0.5}, step=1)
  assert adapter.failure is None


def test_a_wandb_failure_is_not_fatal(monkeypatch, tmp_path):
  '''A tracker that breaks does not break the run.'''
  monkeypatch.setenv('WANDB_MODE', 'offline')
  monkeypatch.setenv('WANDB_DIR', str(tmp_path))
  monkeypatch.setenv('WANDB_SILENT', 'true')
  parameters = _parameters(
      checkpoint_params=CheckpointParams(save_dir=tmp_path / 'ckpt'))
  adapter = WandbLoggerAdapter(parameters)

  def broken_log_metrics(self, metrics, step=None):
    '''Fails the way a broken tracker does.

    Args:
      self: The logger.
      metrics: The metrics.
      step: The step.

    Raises:
      RuntimeError: Always.
    '''
    raise RuntimeError('the tracker is down')

  monkeypatch.setattr('lightning.pytorch.loggers.WandbLogger.log_metrics',
                      broken_log_metrics)
  adapter.log_metrics({'train_loss': 0.5}, step=1)


def test_a_broken_wandb_install_is_not_fatal(monkeypatch, tmp_path):
  '''A wandb that cannot even start is downgraded to a warning.'''
  monkeypatch.setenv('WANDB_MODE', 'offline')
  monkeypatch.setenv('WANDB_DIR', str(tmp_path))

  def broken_init(*args, **kwargs):
    '''Fails the way a broken install does.

    Args:
      *args: Ignored.
      **kwargs: Ignored.

    Raises:
      RuntimeError: Always.
    '''
    raise RuntimeError('wandb is broken')

  monkeypatch.setattr('wandb.init', broken_init)
  parameters = _parameters(
      checkpoint_params=CheckpointParams(save_dir=tmp_path / 'ckpt'))
  adapter = WandbLoggerAdapter(parameters)
  adapter.log_metrics({'train_loss': 0.5}, step=1)
  adapter.log_hyperparams({'lr': 0.001})
  adapter.save()
  assert adapter.failure is None or adapter.inner is None
