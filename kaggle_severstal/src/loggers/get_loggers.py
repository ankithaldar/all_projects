#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Doc String for the module'''


# imports

from dataclasses import dataclass

from helpers.module_importer import instantiate_class

# imports


# constants
# constants


# classes
@dataclass
class Logger:
  class_path: str
  init_args: dict


def get_loggers():
  return [
    Logger(
      class_path='pytorch_lightning.loggers.WandbLogger',
      init_args={'entity': None, 'project': 'kaggle_severstal', 'name': 'resnet34_01', 'tags': ["baseline", "debug"], 'notes': None, 'group': None, 'job_type': 'train', 'save_dir': "logs/wandb", 'save_code': False, 'offline': False, 'log_model': False}
    ),
    Logger(
      class_path='pytorch_lightning.loggers.CSVLogger',
      init_args={'save_dir': './logs/csv', 'name': 'metrics'}
    )
  ]

class Loggers:
  '''Loggers'''

  def __init__(self):
    self.logger_list = get_loggers()

  def __call__(self):
    loggers = []
    if isinstance(self.logger_list, list) and len(self.logger_list) > 0:
      loggers = [
        instantiate_class(logger.class_path, logger.init_args)
        for logger in self.logger_list
      ]

    return loggers

# classes
