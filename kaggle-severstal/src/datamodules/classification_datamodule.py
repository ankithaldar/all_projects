#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Doc String for the module'''


# imports

import pytorch_lightning as pl
from datasets.classification_dataset import ClassificationDataset
from torch.utils.data import DataLoader, random_split

# imports


# constants
# constants


# classes
class ClassificationDataModule(pl.LightningDataModule):
  '''Returns torch dataloaders'''

  def __init__(self, batch_size=8, val_split=0.05, num_workers=4):
    super().__init__()

    self.batch_size = batch_size
    self.val_split = val_split
    self.num_workers = num_workers

  def setup(self, stage=None):
    full_dataset = ClassificationDataset(blackout=True)

    val_size = int(len(full_dataset) * self.val_split)
    train_size = len(full_dataset) - val_size

    self.train_ds, self.val_ds = random_split(
      full_dataset,
      lengths=[train_size, val_size]
    )

    # For validation, turn off augmentations & cropping
    self.val_ds.dataset.train = False
    self.val_ds.dataset.aug = self.val_ds.dataset.aug = None

  def train_dataloader(self):
    return DataLoader(
      self.train_ds,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=self.num_workers,
      pin_memory=True,
      drop_last=True
    )

  def val_dataloader(self):
    return DataLoader(
      self.val_ds,
      batch_size=self.batch_size,
      shuffle=False,
      num_workers=self.num_workers,
      pin_memory=True,
      drop_last=False
    )

# classes


# functions
def function_name():
  pass
# functions


# main
def main():
  pass


# if main script
if __name__ == '__main__':
  main()
