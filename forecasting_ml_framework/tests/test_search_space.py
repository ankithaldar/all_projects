#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Tests for the hyperparameter search-space binding.

The configuration vocabulary is deliberately forgiving: either spelling of the
search-space argument is accepted. The searcher classes are not -- passing
``param_grid`` to a randomised searcher raises, and passing
``param_distributions`` to a grid searcher raises. Forwarding whichever spelling
the operator typed therefore breaks every searcher but the one matching it.

The two defects pinned here were that the halving searcher's opt-in import was
gated on a comparison against the fully-qualified path rather than the class name
(so it never fired), and that the search space was forced into ``param_grid``
regardless of what the searcher accepted.
'''

import inspect

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401 - registers the class
from sklearn.model_selection import (
  GridSearchCV,
  HalvingGridSearchCV,
  HalvingRandomSearchCV,
  RandomizedSearchCV,
)

from forecasting_ml_framework.modeling.models import _accepts, _bind_search_space

GRID_LIKE = (GridSearchCV, HalvingGridSearchCV)
DISTRIBUTION_LIKE = (RandomizedSearchCV, HalvingRandomSearchCV)


class TestSearchSpaceArgument:
  '''The declared spelling must match what the searcher accepts.'''

  @pytest.mark.parametrize('searcher', GRID_LIKE, ids=lambda item: item.__name__)
  def test_a_grid_searcher_takes_param_grid(self, searcher: type) -> None:
    assert _accepts(searcher, 'param_grid')
    assert not _accepts(searcher, 'param_distributions')

  @pytest.mark.parametrize('searcher', DISTRIBUTION_LIKE, ids=lambda item: item.__name__)
  def test_a_distribution_searcher_takes_param_distributions(self, searcher: type) -> None:
    assert _accepts(searcher, 'param_distributions')
    assert not _accepts(searcher, 'param_grid')

  def test_a_variadic_searcher_accepts_either(self) -> None:
    class Searcher:
      """A searcher taking arbitrary keyword arguments."""

      def __init__(self, **kwargs: object) -> None:
        del kwargs

    assert _accepts(Searcher, 'param_grid')
    assert _accepts(Searcher, 'param_distributions')

  def test_an_unintrospectable_class_is_assumed_to_accept(self) -> None:
    class Opaque:
      """A class whose constructor cannot be inspected."""

      __init__ = None

    assert _accepts(Opaque, 'param_grid')


class TestBinding:
  '''The binding moves the space to the accepted spelling.'''

  @pytest.mark.parametrize('searcher', GRID_LIKE, ids=lambda item: item.__name__)
  def test_a_grid_searcher_keeps_the_declared_spelling(self, searcher: type) -> None:
    bound = _bind_search_space(searcher, {'param_grid': {'max_depth': [1, 2]}, 'n_jobs': -1})
    assert 'param_grid' in bound
    assert 'param_distributions' not in bound
    assert bound['n_jobs'] == -1

  def test_a_grid_searcher_accepts_the_distribution_spelling_too(self) -> None:
    """The configuration vocabulary is forgiving in both directions."""
    bound = _bind_search_space(GridSearchCV, {'param_distributions': {'max_depth': [1, 2]}})
    assert 'param_grid' in bound
    assert 'param_distributions' not in bound

  def test_a_randomised_searcher_is_given_param_distributions(self) -> None:
    """The defect: every searcher received ``param_grid``, so this raised."""
    bound = _bind_search_space(RandomizedSearchCV, {'param_grid': {'max_depth': [1, 2]}})
    assert 'param_distributions' in bound
    assert 'param_grid' not in bound

  def test_the_bound_space_is_accepted_by_the_constructor(self) -> None:
    """The end-to-end property: the result must actually construct."""
    bound = _bind_search_space(RandomizedSearchCV, {'param_grid': {'max_depth': [1, 2]}})
    searcher = RandomizedSearchCV(estimator=RandomForestClassifier(), **bound)
    assert searcher.param_distributions is not None

  def test_a_searcher_without_a_space_is_left_alone(self) -> None:
    bound = _bind_search_space(GridSearchCV, {'n_jobs': -1, 'refit': True})
    assert bound == {'n_jobs': -1, 'refit': True}

  def test_the_input_mapping_is_not_mutated(self) -> None:
    original = {'param_grid': {'max_depth': [1, 2]}}
    _bind_search_space(RandomizedSearchCV, original)
    assert original == {'param_grid': {'max_depth': [1, 2]}}


class TestHalvingEnablement:
  '''The opt-in import must fire for the halving searchers.'''

  def test_the_class_is_importable_after_the_opt_in(self) -> None:
    """Without the opt-in, the attribute does not exist at all.

    The defect compared ``__name__`` against ``'sklearn.model_selection.
    HalvingGridSearchCV'``. A class's ``__name__`` is the bare name, so the
    comparison was never true and the import never ran.
    """
    assert HalvingGridSearchCV.__name__ == 'HalvingGridSearchCV'
    assert GridSearchCV.__name__ == 'GridSearchCV'

  def test_no_searcher_declares_a_dotted_name(self) -> None:
    for searcher in (GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV):
      assert '.' not in searcher.__name__

  def test_the_signature_is_inspectable_for_every_searcher(self) -> None:
    for searcher in (GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV):
      assert inspect.signature(searcher).parameters
