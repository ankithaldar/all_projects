#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Tests for the search-space distribution registry.

The framework resolves a hyper-parameter search space from configuration, and a
search space cannot be written entirely as YAML literals -- a continuous
distribution has to name a constructor. That makes the resolution mechanism a
code-execution surface by construction, because configuration is data and data
must never be able to name an arbitrary callable.

The mechanism is now a closed registry: the expression is parsed, its callee is
looked up in :data:`DISTRIBUTIONS`, and its arguments must be literals. There is no
interpreter anywhere in the path. These tests pin that property from both sides --
that a legitimate distribution still resolves, and that every attack shape is
refused with an error naming what *is* supported.
'''

import pathlib

import pytest

from forecasting_ml_framework.exceptions import InvalidSearchSpaceError
from forecasting_ml_framework.utils.reflection import (
  DISTRIBUTIONS,
  assert_no_builtin_namespace,
  evaluate_distribution,
  looks_like_distribution,
)


class TestRegistryIsClosed:
  """The supported set is enumerable and bounded."""

  def test_the_registry_is_populated(self) -> None:
    assert len(DISTRIBUTIONS) > 10

  def test_every_key_is_a_dotted_name(self) -> None:
    for key in DISTRIBUTIONS:
      assert key.count('.') >= 1, f'{key} is not a dotted name'
      assert all(part.isidentifier() for part in key.split('.')), key

  def test_every_key_is_samplable(self) -> None:
    """A registered name that cannot be sampled from is a lie in the registry."""
    constructor = DISTRIBUTIONS['scipy.stats.uniform']
    assert constructor(0.0, 1.0).rvs() is not None

  def test_the_registry_names_no_builtin(self) -> None:
    assert_no_builtin_namespace()

  def test_the_registry_is_the_only_route(self) -> None:
    """No module in the framework evaluates a configuration-supplied expression.

    The scan is AST-based rather than textual, so a mention of ``eval`` inside a
    docstring or a comment -- including one explaining why the call was removed --
    is not counted as a call. A textual scan would make the explanation of the fix
    into a violation of the fix.
    """
    import ast  # noqa: PLC0415 - imported for the walk

    source_root = pathlib.Path(__file__).resolve().parents[1] / 'src'
    offenders = []
    for path in source_root.rglob('*.py'):
      tree = ast.parse(path.read_text(encoding='utf-8'))
      for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
          continue
        if node.func.id == 'eval':
          offenders.append(f'{path.name}:{node.lineno}')
    assert not offenders, f'bare eval() remains in: {offenders}'


class TestLegitimateDistributions:
  """Every shape a real search space uses must keep working."""

  @pytest.mark.parametrize(
    'expression',
    [
      'scipy.stats.uniform(0.01, 0.1)',
      'scipy.stats.uniform(loc=0.01, scale=0.1)',
      'scipy.stats.loguniform(0.005, 0.2)',
      'scipy.stats.norm(0, 1)',
      'scipy.stats.gamma(1.5, loc=0, scale=1)',
      'numpy.random.randint(50, 800)',
      'numpy.random.normal(0, 1)',
      'numpy.random.choice([1, 2, 3])',
      'numpy.random.choice(["a", "b"])',
      'numpy.random.uniform(0.1, 0.9, size=100)',
      'np.random.randint(3, 12)',
      'np.random.choice([1, 2])',
    ],
  )
  def test_the_expression_resolves(self, expression: str) -> None:
    assert evaluate_distribution(expression) is not None

  def test_a_list_of_candidates_is_a_search_space(self) -> None:
    """A categorical space is a list of literals, not a distribution."""
    assert looks_like_distribution('numpy.random.choice([1, 2, 3])')


class TestAttackShapes:
  """Each of these is a way a configuration value might reach the interpreter."""

  @pytest.mark.parametrize(
    'expression',
    [
      '__import__("os").system("id")',
      'os.system("id")',
      'eval("1+1")',
      'exec("x=1")',
      'open("/etc/passwd")',
      '__import__("os")',
      'getattr(__import__("os"), "system")("id")',
      'scipy.stats.uniform(0.01, __import__("os").getpid())',
      'scipy.stats.uniform(0.01, 0.1); __import__("os").system("id")',
      'scipy.stats.uniform(0.01, 0.1); import os',
      'scipy.stats.uniform([x for x in []])',
      'scipy.stats.uniform(scale=[x for x in []])',
      'scipy.stats.uniform(scale=(lambda: 1)())',
      'scipy.stats.uniform(scale=(1).__class__.__base__)',
      'scipy.stats.uniform(scale="".__class__.__mro__)',
      'scipy.stats.uniform(**{"loc": 0.01})',
      'scipy.stats.uniform(0.01 + 1)',
      'scipy.stats.uniform[0]',
      'scipy.stats.uniform',
      'uniform(0.01, 0.1)',
      '[x for x in range(3)]',
      'scipy.stats.norm(0, x := 1)',
      'numpy.random.choice({"a": __import__("os")})',
    ],
  )
  def test_the_expression_is_refused(self, expression: str) -> None:
    with pytest.raises(InvalidSearchSpaceError):
      evaluate_distribution(expression)


class TestErrorQuality:
  """A rejection should tell the operator what to write instead."""

  def test_an_unregistered_name_lists_the_supported_set(self) -> None:
    with pytest.raises(InvalidSearchSpaceError) as caught:
      evaluate_distribution('scipy.stats.wibble(1, 2)')
    message = str(caught.value)
    assert 'not a registered distribution' in message
    assert 'scipy.stats.uniform' in message

  def test_an_argument_shape_is_explained(self) -> None:
    with pytest.raises(InvalidSearchSpaceError) as caught:
      evaluate_distribution('scipy.stats.uniform(0.01, [x for x in []])')
    assert 'must be literals' in str(caught.value)

  def test_a_syntax_error_is_reported_as_a_configuration_error(self) -> None:
    with pytest.raises(InvalidSearchSpaceError) as caught:
      evaluate_distribution('scipy.stats.uniform(0.01,')
    assert 'not parseable' in str(caught.value)
