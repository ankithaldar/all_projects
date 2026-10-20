#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Loading and validating a configuration.

A configuration is composed from three layers:

1. the **module files** under ``config_params/modules/``, one per
   functional module, holding the values every experiment shares;
2. the **experiment file** under ``config_params/experiments/``, holding
   only what that experiment changes;
3. the **backend selection** under ``config_params/run/backend.yaml``,
   which says whether the PyTorch Lightning backend, the Keras backend, or
   both are to be run.

The loader merges the layers and validates the result against the pydantic
schema.  The error report is the point of this module: a pydantic
validation error already says which key failed and what was expected, and
this adds the file it came from, so a failure in one of the twenty-one
experiment files is identifiable without a stack trace.
'''

from pathlib import Path
from typing import Any
from typing import Optional

import yaml
from pydantic import ValidationError

from birdcall.config_params.parameters import Parameters

CONFIG_DIR = Path(__file__).resolve().parent
MODULES_DIR = CONFIG_DIR / 'modules'
RUN_DIR = CONFIG_DIR / 'run'


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
  '''Merges two nested dictionaries, the override winning.

  Args:
    base: The lower layer.
    override: The higher layer.

  Returns:
    A new dictionary holding the merged result.
  '''
  merged = dict(base)
  for key, value in override.items():
    current = merged.get(key)
    if isinstance(current, dict) and isinstance(value, dict):
      merged[key] = deep_merge(current, value)
    else:
      merged[key] = value
  return merged


def read_yaml(path: str | Path) -> dict[str, Any]:
  '''Reads a YAML document and checks that it is a mapping.

  Args:
    path: The path of the YAML file.

  Returns:
    The parsed document.

  Raises:
    FileNotFoundError: If the file does not exist.
    ValueError: If the document is not a mapping.
  '''
  path = Path(path)
  if not path.exists():
    raise FileNotFoundError(f'Configuration file not found: {path}')
  document = yaml.safe_load(path.read_text(encoding='utf-8'))
  if document is None:
    return {}
  if not isinstance(document, dict):
    raise ValueError(f'Configuration file {path} is not a mapping.')
  return document


def module_names() -> list[str]:
  '''Lists the available configuration modules.

  Returns:
    The module names, in the order their files sort.
  '''
  return sorted(path.stem for path in MODULES_DIR.glob('*.yaml'))


def read_module(name: str) -> dict[str, Any]:
  '''Reads one module file.

  Args:
    name: The module name.

  Returns:
    The module's values.

  Raises:
    FileNotFoundError: If the module does not exist.
  '''
  path = MODULES_DIR / f'{name}.yaml'
  if not path.exists():
    raise FileNotFoundError(
        f'Configuration module {name!r} not found at {path}. Available: '
        f'{module_names()}')
  return read_yaml(path)


def read_config_document(path: str | Path) -> dict[str, Any]:
  '''Reads an experiment file without validating it.

  Args:
    path: The path of the YAML file.

  Returns:
    The parsed document.
  '''
  return read_yaml(path)


def load_parameters(path: str | Path) -> Parameters:
  '''Loads and validates a configuration.

  The module files listed by the experiment (or all of them, when the
  experiment lists none) are merged first, then the experiment's
  ``overrides``, then its top level keys.

  Args:
    path: The path of the experiment file.

  Returns:
    The validated configuration.

  Raises:
    FileNotFoundError: If a file does not exist.
    ValueError: If a document is not a mapping.
    pydantic.ValidationError: If the merged document does not satisfy the
      schema.
  '''
  path = Path(path)
  document = read_config_document(path)
  overrides = document.get('overrides') or {}
  if not isinstance(overrides, dict):
    raise ValueError(f'Configuration file {path}: overrides is not a mapping.')
  names = document.get('modules') or module_names()
  merged: dict[str, Any] = {}
  for name in names:
    merged = deep_merge(merged, read_module(name))
  merged = deep_merge(merged, overrides)
  top_level = {key: value for key, value in document.items()
               if key not in ('modules', 'overrides')}
  merged = deep_merge(merged, top_level)
  try:
    return Parameters(**merged)
  except ValidationError as error:
    raise ValidationError.from_exception_data(
        title=f'{path}: invalid configuration',
        line_errors=error.errors()) from error


def iter_experiment_paths() -> list[Path]:
  '''Lists every experiment file shipped with the package.

  Returns:
    The paths of the experiment YAML files, sorted.
  '''
  root = CONFIG_DIR / 'experiments'
  if not root.exists():
    return []
  return sorted(root.rglob('*.yaml'))


def backend_selection_path() -> Path:
  '''Returns the path of the backend selection file.

  Returns:
    The path of ``config_params/run/backend.yaml``.
  '''
  return RUN_DIR / 'backend.yaml'


def load_backend_selection(path: Optional[str | Path] = None) -> dict[str, Any]:
  '''Reads the backend selection.

  Args:
    path: The path of the file.  The packaged one is used by default.

  Returns:
    The selection document.
  '''
  return read_yaml(path or backend_selection_path())
