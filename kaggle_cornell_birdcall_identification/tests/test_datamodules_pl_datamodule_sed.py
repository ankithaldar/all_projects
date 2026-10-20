#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the datamodule.

The data is synthetic: a one row fold CSV and a short wav file per row.
The waveform transform stack needs ``audiomentations``, which is not
installed here, so the identity transform stands in for it.
'''

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from birdcall.augmentations.registry import build_transforms
from birdcall.config_params.checkpoint_params import CheckpointParams
from birdcall.config_params.configs import get_bird_code
from birdcall.config_params.configs import get_inv_ebird_label
from birdcall.config_params.dataset_params import DatasetParams
from birdcall.config_params.parameters import Parameters
from birdcall.config_params.run_params import RunParams
from birdcall.datamodules import pl_datamodule_sed
from birdcall.datamodules.pl_datamodule_sed import SedDataModule
from birdcall.datasets.pl_sampler_imbalanced import ImbalancedDatasetSampler
from birdcall.helpers.pl_seed import worker_init_fn

SAMPLE_RATE = 8000
DURATION = 0.5
PERIOD = 1
SPECIES = ['aldfly', 'amecro', 'balori']


def _identity_transform(x, sr):
  '''Returns the waveform unchanged.

  Args:
    x: The waveform.
    sr: The sample rate.

  Returns:
    The waveform.
  '''
  return x


def _write_wav(path: Path, frequency: float) -> None:
  '''Writes a short mono tone.

  Args:
    path: The wav file to write.
    frequency: The frequency of the tone.
  '''
  time = np.arange(int(SAMPLE_RATE * DURATION)) / SAMPLE_RATE
  signal = 0.5 * np.sin(2.0 * np.pi * frequency * time)
  path.parent.mkdir(parents=True, exist_ok=True)
  sf.write(path, signal.astype(np.float32), SAMPLE_RATE)


def _fold(tmp_path) -> tuple[Path, Path]:
  '''Builds a synthetic fold.

  Args:
    tmp_path: The pytest temporary directory.

  Returns:
    The audio root and the CSV of the fold.
  '''
  audio_root = tmp_path / 'audio'
  rows = []
  for index, species in enumerate(SPECIES):
    for row in range(2):
      filename = f'REC{index}_{row}.wav'
      _write_wav(audio_root / species / filename, 440.0 * (index + 1))
      rows.append({
          'filename': filename,
          'ebird_code': species,
          'secondary_labels': '[]',
      })
  csv_path = tmp_path / 'fold_0.csv'
  pd.DataFrame(rows).to_csv(csv_path, index=False)
  return audio_root, csv_path


def _parameters(tmp_path, **overrides) -> Parameters:
  '''Builds a validated configuration over a synthetic fold.

  Args:
    tmp_path: The pytest temporary directory.
    **overrides: The attributes that differ from the default.

  Returns:
    The configuration.
  '''
  audio_root, csv_path = _fold(tmp_path)
  train = DatasetParams(root_dir=audio_root,
                        csv_dir=csv_path,
                        period=PERIOD,
                        isTraining=True,
                        num_test_samples=1)
  valid = DatasetParams(root_dir=audio_root,
                        csv_dir=csv_path,
                        period=PERIOD,
                        isTraining=False,
                        num_test_samples=2)
  payload = {
      'train_ds_params': train,
      'valid_ds_params': valid,
      'test_ds_params': valid,
      'train_bs': 2,
      'train_num_workers': 0,
      'valid_bs': 3,
      'valid_num_workers': 0,
      'run_params': RunParams(max_epochs=1, epoch_length=3),
      'checkpoint_params': CheckpointParams(save_dir=tmp_path / 'ckpt'),
  }
  payload.update(overrides)
  return Parameters(**payload)


def _datamodule(monkeypatch, tmp_path, **overrides) -> SedDataModule:
  '''Builds a datamodule over a synthetic fold.

  Args:
    monkeypatch: The pytest monkeypatch fixture.
    tmp_path: The pytest temporary directory.
    **overrides: The configuration attributes that differ.

  Returns:
    The datamodule, with its datasets built.
  '''
  monkeypatch.setattr(
      pl_datamodule_sed, 'build_transforms',
      lambda name, **kwargs: {'train': _identity_transform,
                              'valid': _identity_transform})
  module = SedDataModule(_parameters(tmp_path, **overrides))
  module.setup()
  return module


def test_the_three_datasets_are_built(monkeypatch, tmp_path):
  '''Every split of the configuration becomes a dataset.'''
  module = _datamodule(monkeypatch, tmp_path)
  assert len(module.train_dataset) == 6
  assert len(module.valid_dataset) == 6
  assert len(module.test_dataset) == 6
  assert module.train_dataset.isTraining is True
  assert module.valid_dataset.isTraining is False


def test_the_training_loader_uses_the_imbalanced_sampler(monkeypatch, tmp_path):
  '''The training loader draws with the inverse-frequency sampler.'''
  module = _datamodule(monkeypatch, tmp_path)
  loader = module.train_dataloader()
  assert isinstance(loader.sampler, ImbalancedDatasetSampler)
  assert loader.drop_last is True
  assert loader.batch_size == 2
  assert len(loader.sampler) == len(module.train_dataset)


def test_the_validation_and_test_loaders_are_unshuffled(monkeypatch,
                                                         tmp_path):
  '''The other two loaders are contiguous and keep every row.'''
  module = _datamodule(monkeypatch, tmp_path)
  valid = module.valid_dataloader()
  test = module.test_dataloader()
  assert valid.drop_last is False
  assert valid.batch_size == 3
  assert test.drop_last is False
  assert test.batch_size == 3


def test_every_loader_seeds_its_workers(monkeypatch, tmp_path):
  '''Every loader seeds the workers of the torch backend.'''
  module = _datamodule(monkeypatch, tmp_path)
  for loader in (module.train_dataloader(), module.valid_dataloader(),
                 module.test_dataloader()):
    assert loader.worker_init_fn is not None
    loader.worker_init_fn(0)
  assert module.valid_dataloader().worker_init_fn.func is worker_init_fn
  assert module.test_dataloader().worker_init_fn.func is worker_init_fn


def test_the_training_loader_is_built_once(monkeypatch, tmp_path):
  '''The loader keeps its identity, so the epoch reaches its sampler.'''
  module = _datamodule(monkeypatch, tmp_path)
  loader = module.train_dataloader()
  assert module.train_dataloader() is loader
  assert loader.sampler is module.train_sampler
  assert module.valid_dataloader() is module.valid_dataloader()


def test_a_batch_holds_the_waveforms_and_the_labels(monkeypatch, tmp_path):
  '''A batch is the item the original's dataset produced.'''
  module = _datamodule(monkeypatch, tmp_path)
  loader = module.train_dataloader()
  batch = next(iter(loader))
  assert batch['waveforms'].shape == (2, 1, SAMPLE_RATE * PERIOD)
  assert batch['all_labels'].shape == (2, 1, 264)
  assert batch['all_labels'].dtype == torch.float32
  assert batch['all_labels'].sum() == 2.0
  assert len(batch['filename']) == 2


def test_the_label_space_is_the_shared_one(monkeypatch, tmp_path):
  '''The label vectors are the shared 264 species.'''
  module = _datamodule(monkeypatch, tmp_path)
  item = module.train_dataset[0]
  assert item['all_labels'].shape == (1, 264)
  assert item['all_labels'][0][get_bird_code()['aldfly']] == 1.0
  assert get_inv_ebird_label()['Corvus brachyrhynchos_American Crow'] == (
      'amecro')


def test_an_unknown_augmentation_name_is_a_clear_error():
  '''The transform stack fails loudly on an unknown name.

  The stack itself needs ``audiomentations``, which is not installed
  here, so only the name check is exercised.
  '''
  with pytest.raises(ValueError, match='Unknown augmentation'):
    build_transforms('not_an_augmentation')
