#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the Lightning module.

The model is a stub: the PANNs backbone needs the ImageNet weights and
minutes per forward, and the point of these tests is the control plane,
not the model.  The stub keeps the two heads apart, so a loss that
touched the frame head would be visible.
'''

from pathlib import Path

import lightning
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Sampler

from birdcall.config_params.checkpoint_params import CheckpointParams
from birdcall.config_params.configs import get_bird_code
from birdcall.config_params.configs import get_inv_ebird_label
from birdcall.config_params.dataset_params import DatasetParams
from birdcall.config_params.parameters import Parameters
from birdcall.config_params.run_params import RunParams
from birdcall.datamodules import pl_datamodule_sed
from birdcall.datamodules.pl_datamodule_sed import SedDataModule
from birdcall.engines import pl_lightning_module
from birdcall.engines.pl_lightning_module import SedLightningModule

SAMPLE_RATE = 8000
DURATION = 0.5
PERIOD = 1
CLASSES = 264
FRAMES = 5


class StubModel(torch.nn.Module):
  '''A stand-in for the PANNs model.

  It keeps the two heads apart: the clip head reads the first sample of
  every row and the frame head reads the last one, so a loss that used
  the frame head would not be the loss of the clip head.
  '''

  def __init__(self, classes_num: int = CLASSES, frames: int = FRAMES) -> None:
    '''Builds the stub.

    Args:
      classes_num: The number of classes.
      frames: The number of frames of the frame head.
    '''
    super().__init__()
    self.classes_num = classes_num
    self.frames = frames
    self.clip = torch.nn.Linear(1, classes_num)
    self.frame = torch.nn.Linear(1, classes_num)

  def forward(self, inputs):
    '''Runs the stub.

    Args:
      inputs: The tuple ``(waveform, mixup_lambda)``.

    Returns:
      The two heads.
    '''
    waveform, mixup_lambda = inputs
    batch, snippets = waveform.shape[0], waveform.shape[1]
    rows = batch * snippets
    flat = waveform.reshape(rows, -1)
    clip = self.clip(flat[:, :1]).reshape(batch, snippets, self.classes_num)
    frames = self.frame(flat[:, -1:]).reshape(batch, snippets,
                                              self.classes_num)
    if mixup_lambda is not None:
      batch = (batch * snippets) // 2
      snippets = 1
      clip = clip.reshape(rows, self.classes_num)[:batch].reshape(
          batch, snippets, self.classes_num)
      frames = frames.reshape(rows, self.classes_num)[:batch].reshape(
          batch, snippets, self.classes_num)
    framewise = frames.unsqueeze(2).expand(batch, snippets, self.frames,
                                           self.classes_num)
    return {'clipwise_output': clip, 'framewise_output': framewise}


class SpyCriterion(torch.nn.Module):
  '''A criterion that remembers what it was given.

  Attributes:
    received: The tensor the criterion was called with.
    loss_keys: The names of the scalar components.
  '''

  loss_keys = ['bce_loss', 'F_loss']

  def __init__(self) -> None:
    '''Builds the spy.'''
    super().__init__()
    self.received = None

  def forward(self, y_pred, y_target):
    '''Scores the batch.

    Args:
      y_pred: The clip head.
      y_target: The labels.

    Returns:
      The scalar loss and its components.
    '''
    self.received = y_pred
    loss = y_pred.mean()
    return loss, {'bce_loss': loss.detach(), 'F_loss': loss.detach()}


class TinyDataset(Dataset):
  '''A dataset that remembers the epochs it was put in.

  Attributes:
    epochs: The epochs the dataset was told about.
  '''

  def __init__(self, size: int = 2, snippets: int = 1) -> None:
    '''Builds the dataset.

    Args:
      size: The number of items.
      snippets: The number of snippets per item.
    '''
    super().__init__()
    self.size = size
    self.snippets = snippets
    self.epochs: list[int] = []

  def set_epoch(self, epoch: int) -> None:
    '''Records the epoch.

    Args:
      epoch: The epoch index.
    '''
    self.epochs.append(epoch)

  def __len__(self) -> int:
    '''Returns the number of items.'''
    return self.size

  def __getitem__(self, idx: int) -> dict:
    '''Returns one item.

    Args:
      idx: The index of the item.

    Returns:
      The item.
    '''
    generator = torch.Generator().manual_seed(idx)
    waveform = torch.rand(self.snippets, SAMPLE_RATE * PERIOD,
                          generator=generator)
    labels = torch.zeros(self.snippets, CLASSES)
    labels[0, idx % CLASSES] = 1.0
    return {
        'filename': f'REC{idx}.wav',
        'waveforms': waveform,
        'all_labels': labels,
        'primary_labels': labels.clone(),
        'secondary_labels': torch.zeros(self.snippets, CLASSES),
    }


class TinySampler(Sampler):
  '''A sampler that remembers the epochs it was put in.

  Attributes:
    epochs: The epochs the sampler was told about.
  '''

  def __init__(self, size: int = 2) -> None:
    '''Builds the sampler.

    Args:
      size: The number of indices it draws.
    '''
    self.size = size
    self.epochs: list[int] = []

  def set_epoch(self, epoch: int) -> None:
    '''Records the epoch.

    Args:
      epoch: The epoch index.
    '''
    self.epochs.append(epoch)

  def __iter__(self):
    '''Returns the indices.'''
    return iter(range(self.size))

  def __len__(self) -> int:
    '''Returns the number of indices.'''
    return self.size


def _fold(tmp_path) -> tuple[Path, Path]:
  '''Builds a synthetic fold: a wav file per row and the fold CSV.

  Args:
    tmp_path: The pytest temporary directory.

  Returns:
    The audio root and the CSV of the fold.
  '''
  audio_root = tmp_path / 'audio'
  rows = []
  for index, species in enumerate(['aldfly', 'amecro']):
    directory = audio_root / species
    directory.mkdir(parents=True, exist_ok=True)
    filename = f'REC{index}.wav'
    time = np.arange(int(SAMPLE_RATE * DURATION)) / SAMPLE_RATE
    sf.write(directory / filename,
             (0.5 * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32),
             SAMPLE_RATE)
    rows.append({
        'filename': filename,
        'ebird_code': species,
        'secondary_labels': '[]',
    })
  csv_path = tmp_path / 'fold_0.csv'
  pd.DataFrame(rows).to_csv(csv_path, index=False)
  return audio_root, csv_path


def _parameters(**overrides) -> Parameters:
  '''Builds a validated configuration for the tests.

  Args:
    **overrides: The attributes that differ from the default.

  Returns:
    The configuration.
  '''
  dataset = DatasetParams(root_dir=Path('/tmp'),
                          csv_dir=Path('/tmp/fold.csv'),
                          period=PERIOD,
                          isTraining=True,
                          num_test_samples=1)
  payload = {
      'train_ds_params': dataset,
      'valid_ds_params': dataset,
      'test_ds_params': dataset,
      'run_params': RunParams(max_epochs=2, epoch_length=2),
      'checkpoint_params': CheckpointParams(save_dir=Path('/tmp/ckpt')),
  }
  payload.update(overrides)
  return Parameters(**payload)


def _module(monkeypatch, parameters=None, epoch_length=None,
            **overrides) -> SedLightningModule:
  '''Builds a module with the stub model.

  Args:
    monkeypatch: The pytest monkeypatch fixture.
    parameters: The configuration, or ``None`` for the default one.
    epoch_length: The epoch length of the module, or ``None``.
    **overrides: The configuration attributes that differ.

  Returns:
    The module.
  '''
  monkeypatch.setattr(pl_lightning_module, 'build_model',
                      lambda name, **params: StubModel())
  config = parameters or _parameters(**overrides)
  return SedLightningModule(config, epoch_length=epoch_length)


def _batch(batch_size: int = 2, snippets: int = 1) -> dict:
  '''Builds a batch.

  Args:
    batch_size: The number of items.
    snippets: The number of snippets per item.

  Returns:
    The batch.
  '''
  dataset = TinyDataset(size=batch_size, snippets=snippets)
  items = [dataset[index] for index in range(batch_size)]
  return {
      'filename': [item['filename'] for item in items],
      'waveforms': torch.stack([item['waveforms'] for item in items]),
      'all_labels': torch.stack([item['all_labels'] for item in items]),
      'primary_labels': torch.stack([item['primary_labels']
                                     for item in items]),
      'secondary_labels': torch.stack([item['secondary_labels']
                                       for item in items]),
  }


def _trainer(**kwargs) -> lightning.Trainer:
  '''Builds a trainer for the tests.

  Args:
    **kwargs: The trainer keywords that differ from the default.

  Returns:
    The trainer.
  '''
  payload = {
      'max_epochs': 1,
      'accelerator': 'cpu',
      'devices': 1,
      'logger': False,
      'enable_checkpointing': False,
      'enable_progress_bar': False,
      'num_sanity_val_steps': 0,
  }
  payload.update(kwargs)
  return lightning.Trainer(**payload)


def test_the_module_builds_the_model_the_criterion_and_the_mixer(
    monkeypatch):
  '''The module wires the three registries together.'''
  module = _module(monkeypatch, apply_mixup=True)
  assert isinstance(module.model, StubModel)
  assert module.criterion is not None
  assert module.mixup_augmenter is not None
  assert module.loss_keys == list(module.criterion.loss_keys)
  assert sorted(module.validation_metrics) == ['f1score_clip',
                                               'f1score_frame', 'lraps']


def test_a_run_without_mixup_has_no_mixer(monkeypatch):
  '''The mixup augmenter only exists when the run mixes.'''
  module = _module(monkeypatch, apply_mixup=False)
  assert module.mixup_augmenter is None


def test_an_unknown_model_name_is_a_clear_error(monkeypatch):
  '''An unknown model name does not fail later with an AttributeError.'''
  parameters = _parameters()
  parameters.model_name = 'sed_not_a_model'
  with pytest.raises(ValueError, match='Unknown model'):
    SedLightningModule(parameters)


def test_an_unknown_criterion_name_is_a_clear_error(monkeypatch):
  '''An unknown criterion name is a clear error too.'''
  parameters = _parameters()
  parameters.criterion_name = 'not_a_criterion'
  with pytest.raises(ValueError, match='Unknown criterion'):
    SedLightningModule(parameters)


def test_prepare_batch_keeps_the_batch_when_nothing_is_mixed(monkeypatch):
  '''Without mixup the batch is handed over untouched.'''
  module = _module(monkeypatch, apply_mixup=False)
  batch = _batch()
  inputs, labels = module.prepare_batch(batch, mode='valid')
  assert inputs[1] is None
  assert inputs[0].shape == batch['waveforms'].shape
  assert labels['all_labels'].shape == batch['all_labels'].shape


def test_prepare_batch_halves_the_labels_when_it_mixes(monkeypatch):
  '''The mixup halves the label batch, as the model halves the waveforms.'''
  module = _module(monkeypatch, apply_mixup=True)
  batch = _batch(batch_size=2, snippets=2)
  inputs, labels = module.prepare_batch(batch, mode='train')
  assert inputs[1] is not None
  assert len(inputs[1]) == 4
  assert labels['all_labels'].shape == (2, 1, CLASSES)
  assert labels['primary_labels'].shape == (2, 1, CLASSES)
  assert labels['secondary_labels'].shape == (2, 1, CLASSES)


def test_prepare_batch_reads_the_waveforms_in_the_test_mode(monkeypatch):
  '''The test branch reads the batch, the original read an undefined name.'''
  module = _module(monkeypatch)
  batch = _batch()
  inputs, labels = module.prepare_batch(batch, mode='test')
  assert inputs[0] is batch['waveforms']
  assert inputs[1] is None
  assert labels['filename'] == batch['filename']


def test_prepare_batch_rejects_an_unknown_mode(monkeypatch):
  '''An unknown mode is a clear error.'''
  module = _module(monkeypatch)
  with pytest.raises(ValueError, match='Unknown mode'):
    module.prepare_batch(_batch(), mode='predict')


def test_the_loss_is_computed_from_the_clip_head_only(monkeypatch):
  '''Only the clip head reaches the criterion, and the frame head gets no
  gradient.'''
  spy = SpyCriterion()
  monkeypatch.setattr(pl_lightning_module, 'build_criterion',
                      lambda name, **params: spy)
  module = _module(monkeypatch, apply_mixup=False)
  batch = _batch()
  loss = module.training_step(batch, 0)
  assert spy.received is not None
  assert spy.received.dim() == 3
  assert spy.received.shape[-1] == CLASSES
  expected = module.model((batch['waveforms'], None))['clipwise_output']
  assert torch.allclose(spy.received, expected)
  loss.backward()
  assert module.model.frame.weight.grad is None
  assert module.model.clip.weight.grad is not None


def test_the_loss_keys_are_logged(monkeypatch):
  '''Every loss key of the criterion is a logged metric.'''
  module = _module(monkeypatch, apply_mixup=False)
  loader = DataLoader(TinyDataset(size=2), batch_size=2)
  trainer = _trainer()
  trainer.fit(module, train_dataloaders=loader)
  metrics = trainer.callback_metrics
  assert 'train_loss' in metrics
  for key in module.loss_keys:
    assert f'train_{key}' in metrics


def test_the_validation_metrics_are_logged_with_the_valid_prefix(
    monkeypatch):
  '''The three validation metrics carry the ``valid_`` prefix.'''
  module = _module(monkeypatch, apply_mixup=False)
  loader = DataLoader(TinyDataset(size=2), batch_size=2)
  trainer = _trainer()
  trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
  metrics = trainer.callback_metrics
  for name in ('lraps', 'f1score_clip', 'f1score_frame'):
    assert f'valid_{name}' in metrics
  assert 'valid_loss' in metrics


def test_the_epoch_advances_the_sampler_and_the_dataset(monkeypatch):
  '''The sampler and the dataset are told the epoch.

  Lightning advances the sampler of the dataloader on its own, so the
  sampler is told twice; the dataset is only told by the module.
  '''
  module = _module(monkeypatch, apply_mixup=False)
  dataset = TinyDataset(size=2)
  sampler = TinySampler(size=2)
  loader = DataLoader(dataset, batch_size=2, sampler=sampler)
  trainer = _trainer(max_epochs=2)
  trainer.fit(module, train_dataloaders=loader)
  assert dataset.epochs == [0, 1]
  assert set(sampler.epochs) == {0, 1}
  assert sampler.epochs[-1] == 1


def test_the_epoch_reaches_the_sampler_of_the_datamodule(monkeypatch,
                                                          tmp_path):
  '''The epoch of a real run reaches the sampler that draws the indices.

  The datamodule hands out the same loader every time, so the sampler
  the module advances is the sampler the training loop draws from.
  '''
  monkeypatch.setattr(
      pl_datamodule_sed, 'build_transforms',
      lambda name, **kwargs: {'train': lambda x, sr: x,
                              'valid': lambda x, sr: x})
  audio_root, csv_path = _fold(tmp_path)
  dataset = DatasetParams(root_dir=audio_root,
                           csv_dir=csv_path,
                           period=PERIOD,
                           isTraining=True,
                           num_test_samples=1)
  parameters = _parameters(train_ds_params=dataset,
                           valid_ds_params=dataset,
                           test_ds_params=dataset,
                           train_bs=2,
                           train_num_workers=0)
  datamodule = SedDataModule(parameters)
  module = _module(monkeypatch, parameters, apply_mixup=False)
  trainer = _trainer(max_epochs=2)
  trainer.fit(module, datamodule)
  loader = datamodule.train_dataloader()
  assert loader is datamodule.train_dataloader()
  assert loader.sampler is datamodule.train_sampler
  assert datamodule.train_sampler.epoch == 1
  assert datamodule.train_dataset.epoch == 1


def test_configure_optimizers_returns_the_step_schedule(monkeypatch):
  '''The schedule is driven with every optimiser step.'''
  module = _module(monkeypatch, apply_mixup=False, epoch_length=2)
  config = module.configure_optimizers()
  assert isinstance(config['optimizer'], torch.optim.AdamW)
  assert config['lr_scheduler']['interval'] == 'step'
  assert config['lr_scheduler']['frequency'] == 1
  assert isinstance(config['lr_scheduler']['scheduler'],
                    torch.optim.lr_scheduler.LambdaLR)


def test_test_step_returns_the_predictions_without_a_loss(monkeypatch):
  '''The test step returns the predictions, the inputs and the labels.'''
  module = _module(monkeypatch)
  output = module.test_step(_batch(), 0)
  assert set(output) == {'y_pred', 'x', 'y'}
  assert 'clipwise_output' in output['y_pred']
  assert output['y']['filename'] == ['REC0.wav', 'REC1.wav']


def test_the_fold_csv_of_a_synthetic_dataset_is_read(tmp_path):
  '''The dataset the module is fed reads a one row fold CSV.

  This is the shape of the data the port runs on: a CSV of file names
  and species codes, and the wav files it points at.
  '''
  audio = tmp_path / 'aldfly'
  audio.mkdir()
  time = np.arange(int(SAMPLE_RATE * DURATION)) / SAMPLE_RATE
  sf.write(audio / 'REC0.wav',
           (0.5 * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32),
           SAMPLE_RATE)
  csv_path = tmp_path / 'fold_0.csv'
  pd.DataFrame([{
      'filename': 'REC0.wav',
      'ebird_code': 'aldfly',
      'secondary_labels': '[]',
  }]).to_csv(csv_path, index=False)
  assert csv_path.exists()
  assert get_bird_code()['aldfly'] == 0
  assert get_inv_ebird_label()['Empidonax alnorum_Alder Flycatcher'] == 'aldfly'
