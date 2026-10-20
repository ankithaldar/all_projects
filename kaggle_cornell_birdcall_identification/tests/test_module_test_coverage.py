#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Asserts that every module has a test file.

The project rule is one test file per source file, named after it.  This
module enforces it, so the rule cannot quietly stop being followed as the
tree grows.

It also asserts that no module is left unimported by the test suite, which
is the stronger guarantee: a module that no test imports is a module whose
behaviour is unverified.
'''

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / 'src' / 'birdcall'
TEST_ROOT = Path(__file__).resolve().parent


def source_modules() -> list[Path]:
  '''Lists every module of the package.

  Returns:
    The paths of the Python modules, excluding the package initialisers.
  '''
  modules = []
  for path in sorted(PACKAGE_ROOT.rglob('*.py')):
    if path.name == '__init__.py':
      continue
    modules.append(path)
  return modules


def expected_test_name(module: Path) -> str:
  '''Builds the name of the test file a module must have.

  The name carries the package, because three packages have a module with
  the same stem (``models/pl_registry.py``, ``losses/pl_registry.py`` and
  ``metrics/pl_registry.py``) and one test file cannot serve all three.

  Args:
    module: The module to name a test file for.

  Returns:
    The test file name.
  '''
  relative = module.relative_to(PACKAGE_ROOT).with_suffix('')
  return 'test_' + '_'.join(relative.parts) + '.py'


def test_package_root_exists() -> None:
  '''The package directory is where the modules are expected.'''
  assert PACKAGE_ROOT.is_dir(), f'{PACKAGE_ROOT} does not exist'


def test_every_module_has_a_test_file() -> None:
  '''Every module has a test file named after it.

  Raises:
    AssertionError: If a module has no matching test file.
  '''
  missing = []
  for module in source_modules():
    expected = TEST_ROOT / expected_test_name(module)
    if not expected.exists():
      missing.append(f'{module.relative_to(PACKAGE_ROOT)} -> '
                     f'{expected.name}')
  assert not missing, (
      'Modules without a matching test file:\n  ' + '\n  '.join(missing))


def test_every_module_is_imported_by_the_test_suite() -> None:
  '''Every module is imported by at least one test module.

  Raises:
    AssertionError: If a module is imported by no test.
  '''
  imported: set[str] = set()
  for path in sorted(TEST_ROOT.rglob('test_*.py')):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for node in ast.walk(tree):
      if isinstance(node, ast.Import):
        for alias in node.names:
          imported.add(alias.name)
      elif isinstance(node, ast.ImportFrom):
        if node.module:
          imported.add(node.module)
          # `from package import module` also imports the submodule, and
          # the submodule is what the coverage rule is about.
          for alias in node.names:
            imported.add(f'{node.module}.{alias.name}')
  missing = []
  for module in source_modules():
    relative = module.relative_to(PACKAGE_ROOT).with_suffix('')
    dotted = 'birdcall.' + '.'.join(relative.parts)
    if dotted not in imported:
      missing.append(dotted)
  assert not missing, (
      'Modules that no test imports:\n  ' + '\n  '.join(missing))


@pytest.mark.parametrize('module', source_modules(),
                         ids=lambda path: path.stem)
def test_module_has_a_docstring(module: Path) -> None:
  '''Every module has a docstring.

  Args:
    module: The module to check.
  '''
  source = module.read_text(encoding='utf-8')
  tree = ast.parse(source, filename=str(module))
  assert ast.get_docstring(tree), f'{module} has no module docstring'
