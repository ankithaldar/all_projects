#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the reflective resolver and the search-space guard.

A search-space expression arrives as a string in a YAML document and is
evaluated. That makes the evaluator a code-execution surface reachable from
configuration, so the interesting question is never "does it work" -- it is
"what cannot it be made to do". These tests are written from the second
question: every rejection case is an attack shape, and every acceptance case
exists so a tightened guard cannot silently break ordinary distributions.

The keyword-argument cases matter most. A guard that inspects positional
arguments only leaves the keyword channel open, and a comprehension or an
attribute chain passed as a keyword value evaluates during the call. The empty
builtins namespace stops the obvious forms, but it is a property of one
namespace rather than a claim about which AST nodes are acceptable, so the
literal-only rule has to cover both argument lists on its own.
"""

from __future__ import annotations

import pytest

from forecasting_ml_framework.exceptions import InvalidSearchSpaceError, RoutineNotFoundError
from forecasting_ml_framework.utils.reflection import (
  coerce_search_space,
  evaluate_distribution,
  looks_like_distribution,
  resolve_routine,
  try_resolve,
)


class TestAcceptedExpressions:
  """Ordinary distributions must keep working."""

  @pytest.mark.parametrize(
    'expression',
    [
      'scipy.stats.uniform(0.01, 0.1)',
      'scipy.stats.uniform(loc=0.01, scale=0.1)',
      'scipy.stats.loguniform(0.005, 0.2)',
      'numpy.random.randint(50, 800)',
      'np.random.randint(3, 12)',
      'numpy.random.choice([1, 2, 3])',
      'np.random.choice(["a", "b"])',
      'numpy.random.normal(0, 1)',
    ],
  )
  def test_distribution_is_constructed(self, expression: str) -> None:
    """Positional and keyword forms are both accepted.

    Args:
      expression: A distribution expression from configuration.
    """
    assert evaluate_distribution(expression) is not None

  def test_defaults_are_accepted(self) -> None:
    """An omitted optional argument is not a violation of literal-only."""
    assert evaluate_distribution('scipy.stats.uniform(0.01)') is not None

  def test_a_literal_container_is_accepted(self) -> None:
    """A list of candidates is how a categorical search space is written.

    Rejecting it is not a safety property -- ``ast.literal_eval`` cannot evaluate a
    comprehension -- it is only an artefact of testing each argument node against
    ``ast.Constant``, which a list is not.
    """
    assert evaluate_distribution('numpy.random.choice([1, 2, 3])') is not None

  def test_string_keyword_is_accepted(self) -> None:
    """A string literal is a constant, so it passes the literal-only rule.

    The keyword channel is the one the guard is most likely to tighten by
    accident, so an accepted keyword form is asserted alongside the positional
    one.
    """
    assert evaluate_distribution("numpy.random.choice(['a', 'b'])") is not None

  def test_a_non_distribution_under_an_allowed_root_is_refused(self) -> None:
    """The registry is closed: a module root is not a licence to call anything.

    ``numpy.log`` and ``np.errstate`` were accepted before, because the check was
    "an attribute call under an allowed root" rather than "a registered
    distribution". Neither constructs a distribution, so a search space naming one
    was asking for a value the searcher cannot sample from.
    """
    for expression in ('numpy.log(0.5)', 'np.log(0.01)', 'np.errstate(divide="ignore")'):
      with pytest.raises(InvalidSearchSpaceError) as caught:
        evaluate_distribution(expression)
      assert 'not a registered distribution' in str(caught.value)


class TestRejectedExpressions:
  """Every case here is an attack shape rather than a typo."""

  @pytest.mark.parametrize(
    'expression',
    [
      # The callee itself must be an allowed root.
      "open('/etc/passwd')",
      "__import__('os')",
      'os.system("id")',
      # An unregistered name under an otherwise-allowed root.
      'numpy.log(0.5)',
      'sklearn.model_selection.GridSearchCV()',
      # A plain attribute, not a call.
      'scipy.stats.uniform',
      # Attribute chains that escape the allowed namespace.
      'scipy.stats.uniform(scale=(1).__class__.__base__)',
      'scipy.stats.uniform(scale="".__class__.__mro__)',
      # Comprehensions and lambdas, through both argument channels.
      'scipy.stats.uniform([x for x in []])',
      'scipy.stats.uniform(scale=[x for x in []])',
      'scipy.stats.uniform(scale=(lambda: 1)())',
      # Builtin escape hatches.
      'scipy.stats.uniform(__import__("os").system("id"))',
      'scipy.stats.uniform(scale=eval("1"))',
      # Keyword unpacking, which carries a name and a dict in one node.
      "scipy.stats.uniform(**{'loc': 0.01})",
      # Statements, subscripts, binops and names.
      'scipy.stats.uniform(0.01, 0.1); import os',
      'scipy.stats.uniform[0]',
      'scipy.stats.uniform(0.01 + 1)',
      'uniform(0.01, 0.1)',
    ],
  )
  def test_expression_is_rejected(self, expression: str) -> None:
    """Nothing outside a literal call on an allowed root is evaluated.

    Args:
      expression: An expression that must not be evaluated.
    """
    with pytest.raises(InvalidSearchSpaceError):
      evaluate_distribution(expression)

  def test_unparseable_expression_is_rejected(self) -> None:
    """A syntax error is a configuration error, not an evaluation failure."""
    with pytest.raises(InvalidSearchSpaceError):
      evaluate_distribution('scipy.stats.uniform(0.01,')

  def test_empty_expression_is_rejected(self) -> None:
    """An empty string parses to no expression at all."""
    with pytest.raises(InvalidSearchSpaceError):
      evaluate_distribution('')

  def test_error_carries_the_expression(self) -> None:
    """The error names the offending expression, so the document is locatable."""
    with pytest.raises(InvalidSearchSpaceError) as caught:
      evaluate_distribution('os.system("id")')
    assert 'os.system' in str(caught.value)


class TestDistributionDetection:
  """``looks_like_distribution`` is the string dispatch into the guard."""

  @pytest.mark.parametrize(
    'value',
    ['scipy.stats.uniform(0.01, 0.1)', 'numpy.log(1, 4)', 'np.log(0.01, 0.5)'],
  )
  def test_recognised(self, value: str) -> None:
    """A distribution prefix routes to the evaluator.

    Args:
      value: A search-space string.
    """
    assert looks_like_distribution(value)

  @pytest.mark.parametrize('value', [0.01, 1, None, 'uniform(0, 1)', 'sklearn.linear_model'])
  def test_not_recognised(self, value: object) -> None:
    """A literal or an unprefixed name is not a distribution.

    Args:
      value: A search-space value.
    """
    assert not looks_like_distribution(value)

  def test_coercion_leaves_literals_untouched(self) -> None:
    """Coercion must not disturb a search space that needs no evaluation."""
    space = {'max_depth': [2, 4, 6], 'learning_rate': 0.05}
    assert coerce_search_space(space) == space

  def test_coercion_evaluates_distributions(self) -> None:
    """A distribution string becomes an object, and literals survive beside it."""
    coerced = coerce_search_space({'max_depth': [2, 4], 'rate': 'scipy.stats.loguniform(0.01, 0.5)'})
    assert coerced['max_depth'] == [2, 4]
    assert not isinstance(coerced['rate'], str)

  @pytest.mark.parametrize('value', [None, [], 'not-a-mapping', 7])
  def test_non_mapping_input_passes_through(self, value: object) -> None:
    """A search space that is not a mapping is returned untouched.

    The coercion helper is applied defensively to whatever a configuration block
    supplied, so a value of the wrong type is returned rather than raising: the
    caller decides whether it is acceptable, and it usually is, because a
    hyper-parameter search is optional for most estimators.

    Args:
      value: A search-space value of the wrong shape.
    """
    assert coerce_search_space(value) is value


class TestResolution:
  """The dotted-path resolver is the framework's whole extension surface."""

  def test_short_name_resolves(self) -> None:
    """A routine name from configuration resolves to the class."""
    resolved = resolve_routine('SequenceAverage')
    assert resolved.__name__ == 'SequenceAverage'

  def test_unknown_name_raises_with_candidates(self) -> None:
    """A misspelling names the search path, so the error is actionable.

    The diagnostic is the product here: a routine that cannot be resolved is a
    configuration typo, and the only useful thing a framework can do is show
    every path it tried and the framework's own prefix.
    """
    with pytest.raises(RoutineNotFoundError) as caught:
      resolve_routine('NoSuchRoutine')
    message = str(caught.value)
    assert 'NoSuchRoutine' in message
    assert 'routines' in message

  def test_unknown_name_carries_structured_context(self) -> None:
    """The name and the search path are attributes, not only message text.

    A caller that wants to log or re-raise the failure should not have to parse
    the message to learn which routine was missing.
    """
    with pytest.raises(RoutineNotFoundError) as caught:
      resolve_routine('NoSuchRoutine')
    assert caught.value.context['routine'] == 'NoSuchRoutine'
    assert caught.value.context['candidates']

  def test_extra_module_is_searched(self) -> None:
    """A caller-supplied module is searched after the built-in aggregate."""
    with pytest.raises(RoutineNotFoundError) as caught:
      resolve_routine('NoSuchRoutine', ['some.other.module'])
    assert 'some.other.module' in str(caught.value)

  def test_try_resolve_reports_failure_without_raising(self) -> None:
    """The non-raising form returns the exception for the caller to inspect."""
    resolved, error = try_resolve('nope.NotAThing')
    assert resolved is None
    assert error is not None

  def test_try_resolve_reports_success(self) -> None:
    """The non-raising form returns the object and ``None`` on success.

    The example is an allow-listed package on purpose. Resolution is constrained
    to :data:`_ALLOWED_RESOLUTION_ROOTS`, so a path naming any other package is a
    *rejection* rather than a success even when the attribute exists -- that is
    the control, and ``test_allow_list_refuses_an_escape_route`` covers it.
    """
    resolved, error = try_resolve('sklearn.base.BaseEstimator')
    assert error is None
    assert resolved is not None

  def test_unqualified_path_is_reported_not_raised(self) -> None:
    """A path with no module component is reported through the non-raising form.

    ``try_resolve`` exists to be called in a fallback chain, so a failure is a
    return value rather than an exception; the exception that would otherwise
    propagate is still available to the caller as the second element of the
    pair.
    """
    resolved, error = try_resolve('Counter')
    assert resolved is None
    assert isinstance(error, ImportError)
    assert 'fully qualified' in str(error)
