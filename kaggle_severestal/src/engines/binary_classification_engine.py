#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Ankit Haldar

'''Engine to train classification models'''


# imports
#    script imports
from engines.base_engine import BaseEngine
from engines.engine_utils import get_augmentation
# imports


# constants
# constants


# classes
class BinaryClassificationEngine(BaseEngine):
  '''Engine to train Binary classification models'''

  def __init__(self, hparams):
    super().__init__(hparams)
    # pass

  def prepare_batch(self, batch, mode='valid'):
    pass

  def loss_fn(self, y_pred, y):
    pass

  def output_transform(self, x, y, y_pred, loss=None, dict_loss=None, mode = 'valid'):
    pass

  def _init_optimizer(self):
    pass

  def _init_criterion_function(self):
    pass

  def _init_scheduler(self):
    pass

  def _init_logger(self):
    pass

  def _init_metrics(self):
    pass

  def _init_model(self):
    pass

  def _init_augmentation(self):
    self.tfms = get_augmentation(self.hparams.aug_config)

  def _init_train_datalader(self):
    pass

  def _init_valid_dataloader(self):
    pass

  def _init_test_dataloader(self):
    pass



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
