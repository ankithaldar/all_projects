#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Generates the YAML configuration tree from the original solution.

The original kept 21 experiment configurations as Python modules, each one
repeating all 39 attributes.  That is the copy-paste the documentation
calls out as the configuration drift risk.

This script inverts the layout: it reads the original configurations, works
out which values are shared, writes those into one YAML file **per
functional module**, and then writes one YAML file per experiment holding
only what that experiment changes.  The loader merges the two.

The original sources are needed.  They live in the upstream repository:

    https://github.com/ryanwongsa/kaggle-birdsong-recognition

Usage:

    python -m birdcall.scripts.build_configs --source /path/to/src

The script is deterministic: running it twice produces identical files.
'''

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import yaml

from birdcall.config_params.criterion_params import CRITERION_PARAM_SCHEMAS

CONFIG_DIR = Path(__file__).resolve().parent.parent / 'config_params'
MODULES_DIR = CONFIG_DIR / 'modules'
EXPERIMENTS_DIR = CONFIG_DIR / 'experiments'

MODULE_FILES = {
    'model': ('model_name', 'model_config', 'pretrained_path'),
    'augmentation': ('aug_name', 'apply_mixup', 'bckgrd_aug_dir',
                     'secondary_bckgrd_aug_dir'),
    'loss': ('criterion_name', 'criterion_params'),
    'optimizer': ('optimizer_name', 'weight_decay', 'lr'),
    'scheduler': ('scheduler_name', 'lr_scale_factor'),
    'data': ('PERIOD', 'train_ds_params', 'valid_ds_params', 'test_ds_params',
             'train_bs', 'train_num_workers', 'valid_bs', 'valid_num_workers'),
    'metrics': ('metrics', 'track_metric', 'metric_factor'),
    'logging': ('logger_name', 'logger_params'),
    'checkpoint': ('checkpoint_params', 'checkpoint_dir', 'load_model_only'),
    'runtime': ('run_params', 'dist_params', 'val_length', 'eval_every',
                'add_pbar', 'accumulation_steps', 'gradient_clip_val',
                'seed', 'mixup_seed', 'deterministic'),
}

DATASET_KEYS = ('train_ds_params', 'valid_ds_params', 'test_ds_params')

# The attributes the original had no notion of.  They are seeded here so
# that every generated configuration carries them, and they become part of
# the recipe rather than a hidden default.
PROJECT_DEFAULTS: dict[str, Any] = {
    'seed': 42,
    'mixup_seed': 1234,
    'deterministic': True,
}

# The original tracked to Neptune.  This implementation tracks to wandb,
# keeping the original's metric series names.
LOGGER_NAMES = {'neptune': 'wandb'}

EXPERIMENT_GROUPS = ('example_config', 'final_sed', 'final_sed_5_fold')


def load_original_module(source: Path, relative: str) -> Any:
  '''Imports one original configuration module by path.

  Args:
    source: The root of the original ``src`` directory.
    relative: The path of the module, relative to ``source``.

  Returns:
    The imported module.
  '''
  path = source / relative
  name = 'original_config_' + relative.replace('/', '_').removesuffix('.py')
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  if str(source) not in sys.path:
    sys.path.insert(0, str(source))
  spec.loader.exec_module(module)
  return module


def normalise(value: Any) -> Any:
  '''Converts a value into something YAML can hold.

  Args:
    value: The value to convert.

  Returns:
    The converted value.
  '''
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, dict):
    return {key: normalise(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [normalise(item) for item in value]
  return value


def deep_equal(left: Any, right: Any) -> bool:
  '''Compares two nested structures.

  Args:
    left: The first structure.
    right: The second structure.

  Returns:
    Whether the two structures are equal.
  '''
  return normalise(left) == normalise(right)


def common_value(values: list[Any]) -> Any:
  '''Returns the value that most of the configurations agree on.

  Args:
    values: One value per configuration.

  Returns:
    The most common value.  Ties are broken by first appearance, so the
    result is deterministic.
  '''
  counts: dict[str, int] = {}
  first: dict[str, Any] = {}
  order: list[str] = []
  for value in values:
    key = yaml.safe_dump(normalise(value), sort_keys=True)
    if key not in counts:
      counts[key] = 0
      order.append(key)
      first[key] = normalise(value)
    counts[key] += 1
  best = max(order, key=lambda key: (counts[key], -order.index(key)))
  return first[best]


def collect_parameters(source: Path) -> dict[str, Any]:
  '''Instantiates every original configuration.

  Args:
    source: The root of the original ``src`` directory.

  Returns:
    A mapping of experiment name to the parameter instance.
  '''
  collected: dict[str, Any] = {}
  config_root = source / 'config_params'
  relatives = [('example_config', 'example_config.py')]
  for group in ('final_sed', 'final_sed_5_fold'):
    for path in sorted((config_root / group).glob('*.py')):
      relatives.append((group, f'{group}/{path.name}'))
  for group, relative in relatives:
    module = load_original_module(source, f'config_params/{relative}')
    instance = module.Parameters()
    collected[f'{group}/{Path(relative).stem}'] = instance
  return collected


def build_module_defaults(parameters: dict[str, Any]) -> dict[str, Any]:
  '''Works out the shared value of every attribute.

  Args:
    parameters: Every experiment's parameters.

  Returns:
    A mapping of module name to the values that module owns.
  '''
  instances = list(parameters.values())
  defaults: dict[str, Any] = {}
  for module_name, keys in MODULE_FILES.items():
    module_values: dict[str, Any] = {}
    for key in keys:
      if key in PROJECT_DEFAULTS:
        module_values[key] = PROJECT_DEFAULTS[key]
        continue
      values = [getattr(instance, key, None) for instance in instances]
      if key in DATASET_KEYS:
        module_values[key] = normalise(common_value(values))
      elif key == 'logger_params':
        params = [normalise(value) for value in values]
        shared = {}
        for sub_key in params[0]:
          candidates = [value.get(sub_key) for value in params]
          if sub_key == 'params':
            continue
          shared[sub_key] = normalise(common_value(candidates))
        module_values[key] = shared
      elif key == 'checkpoint_params':
        params = [normalise(value) for value in values]
        shared = {}
        for sub_key in params[0]:
          candidates = [value.get(sub_key) for value in params]
          if sub_key in ('save_dir', 'prefix_name'):
            continue
          shared[sub_key] = normalise(common_value(candidates))
        module_values[key] = shared
      else:
        value = normalise(common_value(values))
        if key == 'logger_name':
          value = LOGGER_NAMES.get(value, value)
        module_values[key] = value
    defaults[module_name] = module_values
  return defaults


def diff_against(base: Any, value: Any) -> Any:
  '''Returns the parts of a value that differ from a base.

  Args:
    base: The shared value.
    value: The experiment's value.

  Returns:
    The differing parts, or ``None`` when the value is identical.
  '''
  base = normalise(base)
  value = normalise(value)
  if isinstance(base, dict) and isinstance(value, dict):
    difference: dict[str, Any] = {}
    for key, item in value.items():
      if key not in base:
        difference[key] = item
        continue
      sub = diff_against(base[key], item)
      if sub is not None:
        difference[key] = sub
    return difference or None
  if base == value:
    return None
  return value


def write_yaml(path: Path, document: Any) -> None:
  '''Writes a YAML document.

  Args:
    path: Where to write it.
    document: The document to write.
  '''
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
      yaml.safe_dump(document, sort_keys=False, default_flow_style=False,
                     width=100),
      encoding='utf-8')


def generate(source: Path) -> None:
  '''Generates the whole YAML configuration tree.

  Args:
    source: The root of the original ``src`` directory.
  '''
  parameters = collect_parameters(source)
  defaults = build_module_defaults(parameters)

  for module_name, values in defaults.items():
    write_yaml(MODULES_DIR / f'{module_name}.yaml', values)

  for experiment, instance in parameters.items():
    overrides: dict[str, Any] = {}
    for module_name, keys in MODULE_FILES.items():
      for key in keys:
        base = defaults[module_name].get(key)
        value = getattr(instance, key, None)
        if key == 'logger_params':
          value = normalise(value)
          value.pop('params', None)
          base = normalise(base)
          base.pop('params', None)
        if key == 'logger_name':
          value = LOGGER_NAMES.get(value, value)
        difference = diff_against(base, value)
        if difference is not None:
          overrides[key] = difference
    document = {
        'name': getattr(instance, 'name'),
        'fold': getattr(instance, 'fold'),
        'modules': list(MODULE_FILES),
        'overrides': overrides,
    }
    write_yaml(EXPERIMENTS_DIR / f'{experiment}.yaml', document)


def validate_criteria(parameters: dict[str, Any]) -> None:
  '''Checks that every configuration's criterion parameters match.

  Args:
    parameters: Every experiment's parameters.

  Raises:
    ValueError: If a configuration's parameters do not fit its criterion.
  '''
  for name, instance in parameters.items():
    schema = CRITERION_PARAM_SCHEMAS[instance.criterion_name]
    try:
      schema(**instance.criterion_params)
    except TypeError as error:
      raise ValueError(f'{name}: {error}') from error


def main() -> None:
  '''The command line entry point.'''
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--source', type=Path, required=True,
      help='The src directory of the original solution.')
  arguments = parser.parse_args()
  parameters = collect_parameters(arguments.source)
  validate_criteria(parameters)
  generate(arguments.source)
  print(f'Wrote {len(MODULE_FILES)} module files and '
        f'{len(parameters)} experiment files to {CONFIG_DIR}.')


if __name__ == '__main__':
  main()
