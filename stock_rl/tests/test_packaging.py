#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Enforce the zero-runtime-dependency constraint mechanically.

The design constraint is that ``stock_rl`` imports only the standard
library. That claim appears in the README, in ``pyproject.toml`` and in
several module docstrings, so it had better be true rather than merely
intended.

The declared ``dependencies`` list in ``pyproject.toml`` catches an
intentional addition, but it cannot catch an undeclared import, so the
whole package is checked module by module.

Two earlier approaches to that check were wrong, and both failures are
worth recording because either one would have shipped a green build that
meant nothing.

*Comparing ``__file__`` against ``sysconfig`` paths.* On Anaconda, Debian
and several other layouts ``site-packages`` sits inside the directory the
interpreter reports as its standard library, so every third-party package
was classified as standard library. The suite passed unconditionally.

*Diffing ``sys.modules`` around a reload.* This depends on import order
across the whole test session. A probe module containing only
``import pylint.lint`` was not detected, because pylint had already been
loaded by an earlier test, so the set difference was empty. The check
passed precisely when the offending import was already cached.

So the imports are read statically with :mod:`ast`. That is deterministic,
independent of what any other test has imported, and points at the source
line where the dependency is actually written.

Classification uses ``sys.stdlib_module_names``, the interpreter's own
authoritative list of top-level modules.

Parsing ``pyproject.toml`` uses ``tomllib``, standard library since 3.11.
'''

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

import pytest

import stock_rl

_ROOT = stock_rl.__name__.split('.', maxsplit=1)[0]
_CONFIG = Path(__file__).resolve().parents[1] / 'pyproject.toml'


def _package_modules() -> list[tuple[str, Path]]:
  '''Return every source file in the package, paired with its dotted name.

  Returns:
    Pairs of (dotted name, source path) for each ``.py`` file beneath the
    package directory. The filesystem is walked rather than
    ``pkgutil``-imported, deliberately: importing every module to discover
    it makes this file depend on the whole package being importable at the
    moment it runs. The check is about what is *written in the source*, so
    it must work on a module that does not yet import cleanly, and it must
    not fail merely because another file is mid-edit.
  '''
  root = Path(stock_rl.__file__).resolve().parent
  found = []
  for path in sorted(root.rglob('*.py')):
    relative = path.relative_to(root)
    if relative.name == '__init__.py':
      # The package's own __init__ is `stock_rl`; a subpackage's is
      # `stock_rl.<sub>`. Neither keeps __init__ in its dotted name.
      parts = list(relative.parts[:-1])
    else:
      parts = list(relative.with_suffix('').parts)
    found.append(('.'.join([_ROOT, *parts]), path))
  return found


def _classify(name: str) -> str:
  '''Return where a module comes from: stdlib, this package, or foreign.

  Args:
    name: Fully qualified module name.

  Returns:
    One of ``'stdlib'``, ``'own'`` or ``'foreign'``. Only ``'foreign'``
    may fail a test.
  '''
  top = name.split('.')[0]
  if top == _ROOT:
    return 'own'
  if top in sys.stdlib_module_names:
    return 'stdlib'
  return 'foreign'


def _imported_roots(path: Path) -> set[str]:
  '''Return every top-level module name imported by a source file.

  Args:
    path: Python source file to parse.

  Returns:
    Top-level names from ``import x``, ``import x.y as z``, ``from x import
    ...`` and ``from x.y import ...``. Relative imports are resolved
    against nothing and skipped, since they stay inside the package.
  '''
  tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
  roots: set[str] = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      for alias in node.names:
        roots.add(alias.name.split('.')[0])
    elif isinstance(node, ast.ImportFrom):
      if node.level:
        continue
      if node.module:
        roots.add(node.module.split('.')[0])
  roots.discard('__future__')
  return roots


class TestClassifierIsNotVacuous:
  '''The classifier must be proven before the rest of this file is trusted.

  These are the tests that would have caught the path-comparison bug
  described in the module docstring.
  '''

  @pytest.mark.parametrize('name', ['json', 'csv', 'socket', 'os.path',
                                    'xml.etree.ElementTree', 'zoneinfo'])
  def test_recognises_standard_library(self, name):
    assert _classify(name) == 'stdlib'

  @pytest.mark.parametrize('name', ['pylint', 'numpy', 'pandas', 'fastapi',
                                    'pytest', 'astroid', 'requests'])
  def test_rejects_known_third_party_packages(self, name):
    assert _classify(name) == 'foreign'

  def test_rejects_this_packages_own_modules_as_foreign(self):
    for name in ('stock_rl', 'stock_rl.costs', 'stock_rl.rl.hedge_env'):
      assert _classify(name) == 'own'

  def test_ast_scan_finds_a_planted_third_party_import(self, tmp_path):
    # The regression guard for the sys.modules bug: a file importing an
    # already-loaded package must still be reported. Written to tmp_path so
    # it never becomes part of the package under test.
    probe = tmp_path / 'probe.py'
    probe.write_text(
      'import pylint.lint\nfrom numpy import array\nimport json\n',
      encoding='utf-8')
    assert _imported_roots(probe) == {'pylint', 'numpy', 'json'}


class TestDeclaredDependencies:
  '''The declared dependency list must stay empty.'''

  def test_project_declares_no_runtime_dependencies(self):
    assert tomllib.loads(_CONFIG.read_text(encoding='utf-8'))[
      'project']['dependencies'] == []

  def test_requires_python_is_pinned(self):
    assert tomllib.loads(_CONFIG.read_text(encoding='utf-8'))[
      'project']['requires-python'] == '>=3.14'


class TestImportClosureIsStandardLibrary:
  '''No module may import anything outside the standard library.'''

  @pytest.mark.parametrize('name,path', _package_modules(),
                           ids=[name for name, _ in _package_modules()])
  def test_module_imports_only_standard_library(self, name, path):
    foreign = sorted(
      root for root in (_imported_roots(path) | _foreign_roots(path))
      if _classify(root) == 'foreign'
    )
    assert not foreign, (
      f'{name} imports third-party module(s) it does not declare: '
      + ', '.join(foreign)
    )

  def test_every_package_module_was_actually_exercised(self):
    # Guards against a discovery bug leaving this file green while checking
    # nothing at all.
    names = [name for name, _ in _package_modules()]
    assert len(names) > 30
    assert 'stock_rl.costs' in names
    assert 'stock_rl' in names

  def test_package_version_is_exposed(self):
    assert stock_rl.__version__


def _foreign_roots(path: Path) -> set[str]:
  '''Return third-party roots imported inside a function body.

  Args:
    path: Python source file to parse.

  Returns:
    Top-level names from imports nested in functions or methods, which a
    naive top-level-only scan would miss. Deferred imports are how a
    dependency sneaks in unnoticed.
  '''
  tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
  roots: set[str] = set()
  for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
      for inner in ast.walk(node):
        if isinstance(inner, ast.Import):
          for alias in inner.names:
            roots.add(alias.name.split('.')[0])
        elif isinstance(inner, ast.ImportFrom) and not inner.level:
          if inner.module:
            roots.add(inner.module.split('.')[0])
  return roots - {'__future__'}



