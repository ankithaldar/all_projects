#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Tests for the framework's entry points and published surface.

An entry point that cannot be resolved, or a symbol advertised in ``__all__`` that
the module does not define, fails at the moment someone tries to use it rather
than at build time. Both were true here: the ``kedro.project_commands`` entry
point named an attribute that did not exist, and ``routines.__all__`` advertised
two base classes the module never imported.

These tests resolve the surface the way its consumers do -- through the installed
distribution's entry-point table, and through a star-import -- so the check
exercises the same path that fails in production.
'''

import importlib.metadata as metadata

import pytest

from forecasting_ml_framework import cli
from forecasting_ml_framework.preprocessing import routines


def _entry_points(group: str) -> list:
  """Resolve an installed entry-point group.

  Args:
    group: The entry-point group name.

  Returns:
    The resolved entry points.
  """
  return list(metadata.entry_points(group=group))


class TestKedroEntryPoints:
  '''Kedro resolves these by name from the installed distribution.'''

  @pytest.fixture(scope='class')
  @classmethod
  def kedro_commands(cls) -> list:
    """Return the resolved ``kedro.project_commands`` entry points.

    Returns:
      The entry points.
    """
    return _entry_points('kedro.project_commands')

  def test_the_run_group_resolves(self, kedro_commands: list) -> None:
    entry = next(item for item in kedro_commands if item.name == 'run')
    assert entry.load() is cli.commands

  def test_the_project_group_resolves(self, kedro_commands: list) -> None:
    """The defect: the entry point named ``cli:project``, which did not exist."""
    entry = next(item for item in kedro_commands if item.name == 'project')
    assert entry.load() is cli.project_commands

  def test_the_project_alias_is_the_same_object(self) -> None:
    """Aliasing rather than renaming keeps the two spellings from diverging."""
    assert cli.project is cli.project_commands

  def test_the_project_group_exposes_its_subcommands(self, kedro_commands: list) -> None:
    group = next(item for item in kedro_commands if item.name == 'project').load()
    assert sorted(group.commands) == ['pipeline', 'routines']

  def test_the_run_group_exposes_its_subcommands(self, kedro_commands: list) -> None:
    group = next(item for item in kedro_commands if item.name == 'run').load()
    assert sorted(group.commands) == ['run', 'version']

  def test_the_console_script_resolves(self) -> None:
    entry = next(item for item in _entry_points('console_scripts') if item.name == 'forecasting-ml')
    assert callable(entry.load())


class TestOptionSpellings:
  '''Click keys options by flag string, so duplicate short flags shadow each other.'''

  def test_no_short_flag_is_declared_twice(self) -> None:
    for parameter in cli.run.params:
      for flag in parameter.opts:
        if flag.startswith('--'):
          continue
        declarations = [
          other
          for other in cli.run.params
          for candidate in other.opts
          if candidate == flag
        ]
        assert len(declarations) == 1, f'short flag {flag!r} is declared more than once'

  def test_the_node_selection_options_are_all_reachable(self) -> None:
    flags = {flag for parameter in cli.run.params for flag in parameter.opts}
    for expected in ('--from-nodes', '--to-nodes', '--nodes', '--tags', '--namespace'):
      assert expected in flags, f'{expected} is no longer declared'

  def test_the_selection_options_are_distinct(self) -> None:
    by_flag = {flag: parameter.name for parameter in cli.run.params for flag in parameter.opts}
    assert by_flag['--to-nodes'] == 'to_nodes'
    assert by_flag['--tags'] == 'tags'
    assert by_flag['--nodes'] == 'nodes'
    assert by_flag['--namespace'] == 'namespace'


class TestRoutineCatalogueSurface:
  '''A name advertised in ``__all__`` must exist in the module.'''

  def test_every_exported_name_exists(self) -> None:
    missing = [name for name in routines.__all__ if not hasattr(routines, name)]
    assert not missing, f'__all__ advertises names the module does not define: {missing}'

  def test_a_star_import_succeeds(self) -> None:
    """The defect: the base classes were advertised but never imported."""
    namespace: dict = {}
    exec('from forecasting_ml_framework.preprocessing.routines import *', namespace)  # noqa: S102
    assert 'AVAILABLE_ROUTINES' in namespace

  def test_the_base_classes_are_reachable(self) -> None:
    assert routines.SparkMlRoutine is not None
    assert routines.UdfRoutine is not None

  def test_every_registered_routine_is_exported(self) -> None:
    for name in routines.AVAILABLE_ROUTINES:
      assert name in routines.__all__

  def test_the_registry_key_matches_the_class_name(self) -> None:
    """Identity is resolved from the class name, so the two must agree."""
    for key, routine in routines.AVAILABLE_ROUTINES.items():
      assert key == routine.__name__, f'{key} is registered against {routine.__name__}'
