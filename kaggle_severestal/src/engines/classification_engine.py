#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Doc String for the module'''


# imports
import tensorflow as tf
#    script imports
from engines.base_engine import BaseEngine
from dataloaders.classification_dataloader import ClassificationDataloader
# imports


# constants
# constants


# classes
class ClassificationEngine(BaseEngine):
  '''docstring for ClassificationEngine'''

  def __init__(self, hparams):
    super().__init__(hparams)
    self.dataloader = ClassificationDataloader(hparams)

    for each in self.dataloader:
      print(each)

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
