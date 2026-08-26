#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Lightning DataModule wiring study datasets per fold."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from knee.datasets.study_dataset import StudyDataset, collate_studies


class StudyDataModule(pl.LightningDataModule):
  """Provide train/val dataloaders for one CV fold.

  Assembly happens in ``main.py`` (paths and fold split are runtime state),
  while every tunable knob still originates from YAML configuration.
  """

  def __init__(
    self,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    pin_memory: bool,
    **kwargs: Any,
  ) -> None:
    """Initialize loader-level settings from datamodule.yaml.

    Args:
        batch_size: Studies per optimization step.
        num_workers: DataLoader worker processes.
        prefetch_factor: Batches prefetched per worker.
        persistent_workers: Keep workers alive between epochs.
        pin_memory: Pin host memory for async H2D copies.
        **kwargs: Absorbed to tolerate extra YAML keys.
    """
    super().__init__()
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.prefetch_factor = prefetch_factor if num_workers > 0 else 0
    self.persistent_workers = persistent_workers and num_workers > 0
    self.pin_memory = pin_memory
    self.train_dataset: StudyDataset | None = None
    self.valid_dataset: StudyDataset | None = None

  def attach(
    self,
    train_dataset: StudyDataset,
    valid_dataset: StudyDataset,
  ) -> None:
    """Inject fully-constructed datasets for the active fold.

    Args:
        train_dataset: Training-split dataset instance.
        valid_dataset: Validation-split dataset instance.
    """
    self.train_dataset = train_dataset
    self.valid_dataset = valid_dataset

  def _loader(self, dataset: StudyDataset, shuffle: bool) -> DataLoader:
    """Build a DataLoader with collation bound.

    Args:
        dataset: Dataset to wrap.
        shuffle: Whether to shuffle sampling order.

    Returns:
        Configured DataLoader instance.
    """
    return DataLoader(
      dataset,
      batch_size=self.batch_size,
      shuffle=shuffle,
      num_workers=self.num_workers,
      prefetch_factor=self.prefetch_factor or None,
      persistent_workers=self.persistent_workers,
      pin_memory=self.pin_memory,
      collate_fn=collate_studies,
      drop_last=shuffle,
    )

  def train_dataloader(self) -> DataLoader:
    """Return the shuffled training loader.

    Returns:
        DataLoader over the attached training dataset.

    Raises:
        RuntimeError: If datasets were not attached beforehand.
    """
    if self.train_dataset is None:
      raise RuntimeError('attach() must be called before requesting loaders')
    return self._loader(self.train_dataset, shuffle=True)

  def val_dataloader(self) -> DataLoader:
    """Return the deterministic validation loader.

    Returns:
        DataLoader over the attached validation dataset.

    Raises:
        RuntimeError: If datasets were not attached beforehand.
    """
    if self.valid_dataset is None:
      raise RuntimeError('attach() must be called before requesting loaders')
    return self._loader(self.valid_dataset, shuffle=False)


def load_index(path: str) -> pd.DataFrame:
  """Read the series index parquet produced by build-index.

  Args:
      path: Filesystem path to index.parquet.

  Returns:
      Index frame with one row per series.
  """
  return pd.read_parquet(path)
