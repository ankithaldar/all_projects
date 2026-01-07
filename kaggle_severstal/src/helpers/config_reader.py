#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''read configs from yaml file'''


# imports
import os
import re
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import yaml

#    script imports
# imports


# constants
T = TypeVar('T')
# constants


# decorator
def validates(*field_names: str):
  '''Decorator to mark validation methods for specific fields.'''
  def decorator(method):
    method._validates_fields = field_names
    return method
  return decorator
# decorator


# errors
class ConfigValidationError(Exception):
  '''Raised when configuration validation fails.'''
  pass
# errors


# classes
@dataclass
class BaseConfig:
  '''Base class for all configuration dataclasses with validation support.'''

  def __post_init__(self):
    '''Run validation methods after initialization.'''
    for field_name, field_value in self.__dict__.items():
      self._validate_field(field_name, field_value)

  def _validate_field(self, field_name: str, value: Any):
    '''Validate a single field using registered validation methods.'''
    for method_name in dir(self):
      method = getattr(self, method_name)
      if hasattr(method, '_validates_fields') and field_name in method._validates_fields:
        try:
          method(value)
        except Exception as e:
          raise ConfigValidationError(
            f"Validation failed for field '{field_name}': {str(e)}"
          ) from e

  @classmethod
  def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    '''Create a config instance from a dictionary.'''
    field_types = {f.name: f.type for f in fields(cls)}
    kwargs = {}

    for field_name, field_type in field_types.items():
      if field_name not in data:
        continue

      value = data[field_name]

      # Handle nested dataclasses
      if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
        # Handle Optional[SomeDataclass]
        args = [a for a in field_type.__args__ if a != type(None)]
        if len(args) == 1 and hasattr(args[0], '__annotations__'):
          field_type = args[0]

      if hasattr(field_type, '__annotations__'):  # It's a dataclass
        if isinstance(value, dict):
          value = field_type.from_dict(value)
        elif value is None and field_name in cls.__annotations__:
          if not isinstance(None, cls.__annotations__[field_name]):
            value = field_type()

      # Handle lists of dataclasses
      elif (hasattr(field_type, '__origin__') and
          field_type.__origin__ is list and
          len(field_type.__args__) == 1 and
          hasattr(field_type.__args__[0], '__annotations__')):
        nested_type = field_type.__args__[0]
        value = [nested_type.from_dict(v) if isinstance(v, dict) else v for v in value]

      kwargs[field_name] = value

    return cls(**kwargs)


@dataclass
class HardwareConfig(BaseConfig):
  '''Hardware configuration settings.'''
  accelerator: str = 'gpu'
  devices: Union[int, List[int], str] = -1
  strategy: str = 'ddp'
  precision: str = '16-mixed'
  amp_backend: str = 'native'
  enable_progress_bar: bool = True
  num_nodes: int = 1


  @validates('accelerator')
  def validate_accelerator(self, value: str):
    valid = ['cpu', 'gpu', 'tpu', 'hpu', 'mps', 'auto']
    if value.lower() not in valid:
      raise ValueError(f'Accelerator must be one of {valid}')

  @validates('precision')
  def validate_precision(self, value: str):
    valid = ['transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed',
    '32-true', '64-true', '64', '32', '16', 'bf16']
    if value not in valid:
      raise ValueError(f'Precision must be one of {valid}')

  @validates('amp_backend')
  def validate_amp_backend(self, value: str):
    valid = ['native', 'apex']
    if value not in valid:
      raise ValueError(f'AMP Backend must be one of {valid}')

@dataclass
class GlobalConfig(BaseConfig):
  '''Global configuration settings.'''
  run_id: Optional[str] = None
  seed: Optional[int] = 42
  deterministic: bool = False
  accumulate_grad_batches: int = 16
  hardware: HardwareConfig = field(default_factory=HardwareConfig)
  tags: List[str] = field(default_factory=lambda: ['baseline', 'debug'])



@dataclass
class LoggerConfig(BaseConfig):
  '''Logger configuration.'''
  class_path: str
  init_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CallbackConfig(BaseConfig):
  '''Callback configuration.'''
  class_path: str
  init_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsConfig(BaseConfig):
  '''Metrics configuration.'''
  class_path: str
  init_args: Dict[str, Any] = field(default_factory=dict)



@dataclass
class CheckpointingConfig(BaseConfig):
  '''Checkpointing configuration.'''
  dirpath: str = './checkpoints'
  filename: str = '{epoch}-{step}-{val_loss:.3f}'
  monitor: str = 'val_loss'
  mode: str = 'min'
  save_top_k: int = 3
  save_last: bool = True
  save_weights_only: bool = False
  auto_insert_metric_name: bool = True
  every_n_epochs: Optional[int] = None
  every_n_steps: Optional[int] = None
  train_time_interval: Optional[str] = None















@dataclass
class Config(BaseConfig):
  '''Top-level PyTorch Lightning configuration.'''
  global_config: GlobalConfig = field(default_factory=GlobalConfig)
  loggers: List[LoggerConfig] = field(default_factory=list)
  callbacks: List[CallbackConfig] = field(default_factory=list)
  metrics: List[MetricsConfig] = field(default_factory=list)
  checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)


  @classmethod
  def from_yaml(cls, file_path: str) -> 'Config':
    '''Load configuration from YAML file'''
    with open(file_path, 'r', encoding='utf-8') as f:
      config_data = yaml.safe_load(f)

    return cls.from_dict(config_data)
# classes


# classes


# functions
def load_config(config_path: str) -> Config:
  '''loader'''
  return Config.from_yaml(config_path)
# functions


# main
def main():
  cfg = load_config('/run/media/helder/Code_Dump/all_projects_all_branches/competitions_main/kaggle_severstal/src/config_params/sample_config.yml')
  # cfg is a dataclass object you can pass around anywhere
  print(cfg)



# if main script
if __name__ == '__main__':
  main()
