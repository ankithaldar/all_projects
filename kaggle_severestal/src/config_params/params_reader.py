#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Model Parameter Processing'''


# imports
import os
import yaml

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
  from yaml import CLoader
except ImportError:
  from yaml import Loader as CLoader
#    script imports
# imports


# functions
def yaml_reader(file_path: str) -> dict:
  with open(file_path, mode='r', encoding='utf8') as f:
    return yaml.load(f, Loader=CLoader)


def create_path(path: str) -> None:
  if not os.path.exists(path):
    os.makedirs(path)
# functions


# classes
@dataclass
class Parameters:
  '''Model Parameter Processing'''

  def __init__(self, file_path: str):
    self.param_file_path = Path(file_path)
    self.set_parameters(self.decode_yaml())

  # call functions to create necessary attributes and paths
    self._define_dependent_params()
    self._manually_set_params()
    self._create_paths()
    self._validate()

  def decode_yaml(self) -> Any:
    '''read yaml file'''
    return yaml_reader(self.param_file_path)


  def set_parameters(self, params: dict) -> None:
    '''set model parameters'''
    self.__dict__ = {**self.__dict__, **dict(params.items())}


  def _define_dependent_params(self) -> None:
    ''' create parameters after loading params from yaml'''
    self.name = self.param_file_path.stem

    self.logger_params = {
    'project_name': self.project_name,
    'log_every': 10,
    'name': self.name,
    'prefix_name': f'{self.name}_best',
    'tags': [self.fold, self.name, self.model_name, self.loss_func_name],
    'params': {
        'bs': self.train_bs,
        'lr': self.lr,
        'name': self.name,
        'aug_name': self.aug_name,
        'model_name': self.model_name,
        'weight_decay': self.weight_decay,
        'apply_mixup': self.apply_mixup,
        'optimizer_name': self.optimizer_name,
        'loss_func_name': self.loss_func_name,
        'scheduler_name': self.scheduler_name,
        'fold': self.fold,
        'lr_scale_factor': self.lr_scale_factor,
        'period': self.PERIOD,
        **self.model_config,
        **self.loss_func_params
      }
    }

  def _manually_set_params(self):
    self.checkpoint_params['save_dir'] += self.name
    self.checkpoint_params['prefix_name'] = self.name


  def _create_paths(self) -> None:
    '''create paths'''
    pass

  def _validate(self) -> None:
    '''Validates all values in this config.'''
    # check specific logger settings
    self._validate_logger()


  def _validate_logger(self) -> None:
    '''validates values of logger'''
    if self.logger_name is None:
      self.logger_name = 'printlogger'

    if isinstance(self.logger_name, (tuple, list)):
      self.logger_name = self.logger_name[0]


# classes


# functions
def test_parameters():
  params = Parameters('./src/config_params/sample.yml')
  print(params)

def functions():
  pass
# functions


# main
def main():
  # test_parameters()
  pass


# if main script
if __name__ == '__main__':
  main()
