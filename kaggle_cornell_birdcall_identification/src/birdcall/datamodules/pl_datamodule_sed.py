#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The datamodule of the SED pipeline.

The module builds the three dataloaders of the original: the training one
with the inverse-frequency sampler and ``drop_last``, the validation one
and the test one, both contiguous and unshuffled.

The workers are seeded by ``birdcall.helpers.pl_seed.worker_init_fn``.  The
training loader uses the dataset's own worker seed, which carries the
epoch, so a worker draw follows the epoch the dataset is in; the other
two use the plain seed, because they do not draw.
'''

from functools import partial

import lightning
from torch.utils.data import DataLoader

from birdcall.augmentations.registry import build_transforms
from birdcall.config_params.parameters import Parameters
from birdcall.datasets.pl_dataset_sed import SedDataset
from birdcall.datasets.pl_sampler_imbalanced import ImbalancedDatasetSampler
from birdcall.helpers.pl_seed import worker_init_fn


class SedDataModule(lightning.LightningDataModule):
  '''The three dataloaders of one run.

  Attributes:
    config: The validated configuration of the run.
    transforms: The waveform transforms, with the keys ``train`` and
      ``valid``.
    train_dataset: The training split.
    valid_dataset: The validation split.
    test_dataset: The test split.
    train_sampler: The inverse-frequency sampler of the training split.
  '''

  def __init__(self, parameters: Parameters) -> None:
    '''Builds the datamodule of a run.

    Args:
      parameters: The validated configuration.
    '''
    super().__init__()
    self.config = parameters
    self.transforms = None
    self.train_dataset = None
    self.valid_dataset = None
    self.test_dataset = None
    self.train_sampler = None
    self._train_loader = None
    self._valid_loader = None
    self._test_loader = None

  def setup(self, stage: str | None = None) -> None:
    '''Builds the three datasets and the transform stack.

    Args:
      stage: The Lightning stage, unused: the three splits are always
        built, because the inference path needs the test split.
    '''
    self.transforms = build_transforms(
        self.config.aug_name,
        bckgrd_aug_dir=self.config.bckgrd_aug_dir,
        secondary_bckgrd_aug_dir=self.config.secondary_bckgrd_aug_dir)
    self.train_dataset = self._build_dataset(self.config.train_ds_params,
                                             self.transforms['train'])
    self.valid_dataset = self._build_dataset(self.config.valid_ds_params,
                                             self.transforms['valid'])
    self.test_dataset = self._build_dataset(self.config.test_ds_params,
                                            self.transforms['valid'])
    self._train_loader = None
    self._valid_loader = None
    self._test_loader = None

  def _build_dataset(self, ds_params, transform) -> SedDataset:
    '''Builds one split.

    Args:
      ds_params: The dataset parameters of the split.
      transform: The waveform transform of the split.

    Returns:
      The dataset.
    '''
    return SedDataset(root_dir=ds_params.root_dir,
                      csv_dir=ds_params.csv_dir,
                      period=ds_params.period,
                      bird_code=ds_params.bird_code,
                      inv_ebird_label=ds_params.inv_ebird_label,
                      isTraining=ds_params.isTraining,
                      num_test_samples=ds_params.num_test_samples,
                      transform=transform,
                      background_audio_dir=ds_params.background_audio_dir,
                      seed=self.config.seed)

  def _ensure_setup(self) -> None:
    '''Builds the datasets when they are not built yet.

    Lightning calls ``setup`` before it asks for a dataloader, but the
    datamodule is usable on its own too.
    '''
    if self.train_dataset is None:
      self.setup()

  def train_dataloader(self) -> DataLoader:
    '''Returns the training dataloader.

    The loader is built once and then handed out again, because the
    sampler and the dataset are stateful: the module advances their
    epoch at the start of every training epoch, and that has to reach
    the loader the training loop draws from.

    Returns:
      The loader, with the inverse-frequency sampler and
      ``drop_last``, exactly like the original's.
    '''
    if self._train_loader is None:
      self._ensure_setup()
      dataset = self.train_dataset
      self.train_sampler = ImbalancedDatasetSampler(dataset,
                                                     dataset.get_label,
                                                     seed=self.config.seed)
      self._train_loader = DataLoader(
          dataset,
          batch_size=self.config.train_bs,
          num_workers=self.config.train_num_workers,
          shuffle=False,
          drop_last=True,
          sampler=self.train_sampler,
          worker_init_fn=dataset.init_workers_fn)
    return self._train_loader

  def valid_dataloader(self) -> DataLoader:
    '''Returns the validation dataloader.

    Returns:
      The loader, contiguous and unshuffled.
    '''
    if self._valid_loader is None:
      self._ensure_setup()
      self._valid_loader = self._eval_loader(self.valid_dataset)
    return self._valid_loader

  def test_dataloader(self) -> DataLoader:
    '''Returns the test dataloader.

    Returns:
      The loader, contiguous and unshuffled.
    '''
    if self._test_loader is None:
      self._ensure_setup()
      self._test_loader = self._eval_loader(self.test_dataset)
    return self._test_loader

  def _eval_loader(self, dataset: SedDataset) -> DataLoader:
    '''Builds one of the two unshuffled loaders.

    Args:
      dataset: The split the loader reads.

    Returns:
      The loader.
    '''
    return DataLoader(dataset,
                      batch_size=self.config.valid_bs,
                      num_workers=self.config.valid_num_workers,
                      shuffle=False,
                      drop_last=False,
                      worker_init_fn=partial(worker_init_fn,
                                             base_seed=self.config.seed))
