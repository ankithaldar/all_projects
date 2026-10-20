#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the trainer, the checkpoint callback and the resume helpers.

The monitor name is the trap this file exists for: a monitor that is
never logged makes the checkpoint callback fire never, and the run
trains for fifty epochs without a checkpoint.
'''

from pathlib import Path

import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from birdcall.config_params.checkpoint_params import CheckpointParams
from birdcall.config_params.dataset_params import DatasetParams
from birdcall.config_params.parameters import Parameters
from birdcall.config_params.run_params import RunParams
from birdcall.engines.pl_trainer import build_model_checkpoint
from birdcall.engines.pl_trainer import build_trainer
from birdcall.engines.pl_trainer import dist_kwargs
from birdcall.engines.pl_trainer import load_weights_only
from birdcall.engines.pl_trainer import resume_checkpoint_path
from birdcall.engines.pl_trainer import track_metric_name


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
      'run_params': RunParams(max_epochs=7, epoch_length=11),
      'checkpoint_params': CheckpointParams(save_dir=Path('/tmp/ckpt'),
                                            n_saved=3,
                                            prefix_name='prefix'),
  }
  payload.update(overrides)
  return Parameters(**payload)


def test_the_trainer_maps_every_knob_of_the_configuration():
  '''Every field of the control plane reaches the trainer.'''
  parameters = _parameters(eval_every=3,
                           accumulation_steps=2,
                           gradient_clip_val=0.5,
                           add_pbar=False)
  trainer = build_trainer(parameters)
  assert trainer.max_epochs == 7
  assert trainer.accumulate_grad_batches == 2
  assert trainer.gradient_clip_val == pytest.approx(0.5)
  assert trainer.check_val_every_n_epoch == 3
  assert trainer.progress_bar_callback is None
  assert trainer.num_sanity_val_steps == 0
  assert str(trainer.default_root_dir) == '/tmp/ckpt'


def test_the_progress_bar_follows_the_configuration():
  '''The progress bar of the original is the trainer's one.'''
  trainer = build_trainer(_parameters(add_pbar=True))
  assert trainer.progress_bar_callback is not None


def test_the_trainer_takes_the_overrides():
  '''An override wins over the configuration.'''
  trainer = build_trainer(_parameters(), max_epochs=2, devices=1)
  assert trainer.max_epochs == 2
  assert trainer.num_devices == 1


def test_the_distributed_parameters_reach_the_trainer():
  '''The distributed parameters map onto the three trainer keywords.'''
  parameters = _parameters(dist_params={'strategy': 'ddp',
                                        'devices': 2,
                                        'accelerator': 'cpu'})
  trainer = build_trainer(parameters)
  assert type(trainer.strategy).__name__ == 'DDPStrategy'
  assert trainer.num_devices == 2
  assert type(trainer.accelerator).__name__ == 'CPUAccelerator'


def test_a_single_process_run_leaves_the_defaults_alone():
  '''An empty dist_params leaves the trainer on its own defaults.'''
  assert dist_kwargs({}) == {}
  trainer = build_trainer(_parameters())
  assert type(trainer.strategy).__name__ == 'SingleDeviceStrategy'


def test_an_unknown_distributed_parameter_is_a_clear_error():
  '''A dist_params key the trainer does not know is a clear error.'''
  with pytest.raises(ValueError, match='Unknown dist_params'):
    dist_kwargs({'backend': 'nccl'})


def test_the_checkpoint_monitor_is_the_name_that_is_logged():
  '''The monitor is the tracked metric with the ``valid_`` prefix.'''
  parameters = _parameters(track_metric='f1score_clip')
  checkpoint = build_model_checkpoint(parameters)
  assert isinstance(checkpoint, ModelCheckpoint)
  assert checkpoint.monitor == 'valid_f1score_clip'
  assert track_metric_name('lraps') == 'valid_lraps'
  assert track_metric_name('valid_lraps') == 'valid_lraps'


def test_the_checkpoint_mode_follows_the_metric_factor():
  '''A metric that is maximised is watched with ``max``.'''
  maximum = build_model_checkpoint(_parameters(metric_factor=1))
  assert maximum.mode == 'max'
  minimum = build_model_checkpoint(_parameters(metric_factor=-1))
  assert minimum.mode == 'min'


def test_the_checkpoint_keeps_the_configured_number_of_checkpoints():
  '''The number of kept checkpoints is the configured one.'''
  checkpoint = build_model_checkpoint(_parameters())
  assert checkpoint.save_top_k == 3
  assert checkpoint.save_last is True
  assert checkpoint.dirpath == '/tmp/ckpt'
  assert checkpoint.filename == 'prefix'


def test_load_weights_only_restores_the_model_entry(tmp_path):
  '''Only the weights are restored, and the optimiser is left alone.'''
  module = torch.nn.Linear(3, 4)
  optimizer = torch.optim.AdamW(module.parameters(), lr=0.001)
  other = torch.nn.Linear(3, 4)
  checkpoint = tmp_path / 'weights.ckpt'
  torch.save({'model': other.state_dict(), 'optimizer': 'ignored'},
             checkpoint)
  load_weights_only(module, checkpoint)
  assert torch.equal(module.weight, other.weight)
  assert optimizer.param_groups[0]['lr'] == pytest.approx(0.001)


def test_load_weights_only_reads_a_lightning_checkpoint(tmp_path):
  '''A checkpoint Lightning wrote itself is read as well.'''
  module = torch.nn.Linear(3, 4)
  other = torch.nn.Linear(3, 4)
  checkpoint = tmp_path / 'lightning.ckpt'
  torch.save({'state_dict': other.state_dict()}, checkpoint)
  load_weights_only(module, checkpoint)
  assert torch.equal(module.weight, other.weight)


def test_load_weights_only_rejects_a_checkpoint_without_weights(tmp_path):
  '''A checkpoint without weights is a clear error.'''
  module = torch.nn.Linear(3, 4)
  checkpoint = tmp_path / 'empty.ckpt'
  torch.save({'epoch': 3}, checkpoint)
  with pytest.raises(KeyError, match='model'):
    load_weights_only(module, checkpoint)


def test_load_weights_only_rejects_a_missing_file(tmp_path):
  '''A missing checkpoint is a clear error.'''
  with pytest.raises(FileNotFoundError):
    load_weights_only(torch.nn.Linear(3, 4), tmp_path / 'nope.ckpt')


def test_the_resume_path_is_none_without_a_checkpoint_directory():
  '''A run that does not resume has no checkpoint.'''
  assert resume_checkpoint_path(_parameters()) is None


def test_the_resume_path_is_the_file_itself(tmp_path):
  '''A checkpoint file is the resume path.'''
  path = tmp_path / 'run.ckpt'
  path.write_text('not a real checkpoint')
  parameters = _parameters(checkpoint_dir=path)
  assert resume_checkpoint_path(parameters) == path


def test_the_resume_path_prefers_the_last_checkpoint(tmp_path):
  '''A directory holds the ``last`` checkpoint of the run.'''
  (tmp_path / 'epoch=0-step=1.ckpt').write_text('older')
  (tmp_path / 'last.ckpt').write_text('newer')
  parameters = _parameters(checkpoint_dir=tmp_path)
  assert resume_checkpoint_path(parameters) == tmp_path / 'last.ckpt'


def test_the_resume_path_is_the_newest_checkpoint_of_a_directory(tmp_path):
  '''Without a ``last`` checkpoint the newest one is taken.'''
  first = tmp_path / 'epoch=0-step=1.ckpt'
  first.write_text('older')
  second = tmp_path / 'epoch=1-step=2.ckpt'
  second.write_text('newer')
  parameters = _parameters(checkpoint_dir=tmp_path)
  assert resume_checkpoint_path(parameters) == second
