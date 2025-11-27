#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Import a class given its fully-qualified module name.'''


# imports
import importlib
from typing import Type

#    script imports
# imports


# constants
# constants


# functions
def import_class(fully_qualified_name: str) -> Type:
  '''
  Import a class given its fully-qualified module name.

  Parameters
  ----------
  fully_qualified_name : str
    E.g. 'package.submodule.MyClass'

  Returns
  -------
  Type
    The requested class object.

  Raises
  ------
  ImportError
    If the module or the class cannot be found.
  '''
  try:
    module_name, class_name = fully_qualified_name.rsplit('.', 1)
  except ValueError as exc:
    raise ImportError(
      f'{fully_qualified_name!r} is not a fully-qualified class name'
    ) from exc

  try:
    module = importlib.import_module(module_name)
  except ModuleNotFoundError as exc:
    raise ImportError(f'Module {module_name!r} not found') from exc

  try:
    return getattr(module, class_name)
  except AttributeError as exc:
    raise ImportError(
      f'Module {module_name!r} has no attribute {class_name!r}'
    ) from exc


def instantiate_class(class_path: str, params: dict = {}):
  module = import_class(class_path)
  return module(**params)
# functions


